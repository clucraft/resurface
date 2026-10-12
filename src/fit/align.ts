/**
 * Rigid alignment: Kabsch/Horn best fit, ICP (point-to-plane / point-to-point, trimmed),
 * multi-scan global fine alignment, "Object Align" to the world axes and N-point alignment.
 *
 * All matrices are column-major Mat4 rigid transforms (proper rotations, det = +1).
 */
import type { ClosestPointHit, ClosestPointQuery, Mat4, ProgressFn, Vec3 } from "../core/types";
import { choleskySolve, eigenSym3, solveLinear } from "../core/linalg";
import { cross, dot, identity4, invertRigid, length, mul4, normalize, transformDir, transformPoint } from "../core/vec";
import { jacobiEigen, kthSmallest, sampleSlots, selectionCount } from "./common";
import { mat4FromRotation, orthonormalizeRigid, quatToRotation, rigidDifference, rotationAngle, rotationFromVector } from "./transform";

export interface RigidFit {
  /** maps src -> dst */
  matrix: Mat4;
  /** (weighted) RMS of |M src - dst| */
  rms: number;
}

// ===========================================================================
// Horn / Kabsch
// ===========================================================================

/**
 * Weighted least-squares rigid transform mapping packed `src` onto packed `dst` (n points),
 * Horn's quaternion method (always a proper rotation — the reflection case cannot occur).
 * Under-determined input (n < 3 or collinear points) yields the solution closest to identity.
 */
export function rigidFitPacked(src: ArrayLike<number>, dst: ArrayLike<number>, n: number, weights?: ArrayLike<number>): RigidFit {
  if (n < 1) throw new Error("kabsch: no point pairs");
  let sw = 0;
  let sx = 0, sy = 0, sz = 0, dx = 0, dy = 0, dz = 0;
  for (let i = 0; i < n; i++) {
    const w = weights ? weights[i] : 1;
    sw += w;
    sx += w * src[3 * i];
    sy += w * src[3 * i + 1];
    sz += w * src[3 * i + 2];
    dx += w * dst[3 * i];
    dy += w * dst[3 * i + 1];
    dz += w * dst[3 * i + 2];
  }
  if (!(sw > 0)) throw new Error("kabsch: weights sum to zero");
  sx /= sw;
  sy /= sw;
  sz /= sw;
  dx /= sw;
  dy /= sw;
  dz /= sw;
  // cross-covariance S_ab = Σ w a_src b_dst
  let Sxx = 0, Sxy = 0, Sxz = 0, Syx = 0, Syy = 0, Syz = 0, Szx = 0, Szy = 0, Szz = 0;
  let scaleRef = 0;
  for (let i = 0; i < n; i++) {
    const w = weights ? weights[i] : 1;
    const ax = src[3 * i] - sx, ay = src[3 * i + 1] - sy, az = src[3 * i + 2] - sz;
    const bx = dst[3 * i] - dx, by = dst[3 * i + 1] - dy, bz = dst[3 * i + 2] - dz;
    Sxx += w * ax * bx;
    Sxy += w * ax * by;
    Sxz += w * ax * bz;
    Syx += w * ay * bx;
    Syy += w * ay * by;
    Syz += w * ay * bz;
    Szx += w * az * bx;
    Szy += w * az * by;
    Szz += w * az * bz;
    scaleRef += w * (ax * ax + ay * ay + az * az + bx * bx + by * by + bz * bz);
  }
  const N = [
    Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx,
    Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz,
    Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy,
    Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz,
  ];
  let q: Float64Array;
  if (!(scaleRef > 0)) {
    q = Float64Array.of(1, 0, 0, 0);
  } else {
    const e = jacobiEigen(N, 4);
    const top = e.values[3];
    const gap = top - e.values[2];
    if (gap <= 1e-10 * Math.max(Math.abs(top), 1e-300 + scaleRef)) {
      // degenerate (e.g. collinear points): choose the quaternion of the top eigenspace closest to identity
      const v3 = e.vectors[3], v2 = e.vectors[2];
      const a = v3[0], b = v2[0];
      const l = Math.hypot(a, b);
      q = l > 1e-12 ? Float64Array.from({ length: 4 }, (_, k) => (a * v3[k] + b * v2[k]) / l) : v3;
    } else q = e.vectors[3];
  }
  const R = quatToRotation(q[0], q[1], q[2], q[3]);
  const t: Vec3 = [
    dx - (R[0] * sx + R[1] * sy + R[2] * sz),
    dy - (R[3] * sx + R[4] * sy + R[5] * sz),
    dz - (R[6] * sx + R[7] * sy + R[8] * sz),
  ];
  const matrix = mat4FromRotation(R, t);
  let ss = 0;
  for (let i = 0; i < n; i++) {
    const w = weights ? weights[i] : 1;
    const x = src[3 * i], y = src[3 * i + 1], z = src[3 * i + 2];
    const ex = R[0] * x + R[1] * y + R[2] * z + t[0] - dst[3 * i];
    const ey = R[3] * x + R[4] * y + R[5] * z + t[1] - dst[3 * i + 1];
    const ez = R[6] * x + R[7] * y + R[8] * z + t[2] - dst[3 * i + 2];
    ss += w * (ex * ex + ey * ey + ez * ez);
  }
  return { matrix, rms: Math.sqrt(ss / sw) };
}

/** Best-fit rigid transform (rotation + translation) mapping src[i] → dst[i], optional weights. */
export function kabsch(src: Vec3[], dst: Vec3[], weights?: number[]): RigidFit {
  if (src.length !== dst.length) throw new Error(`kabsch: src (${src.length}) and dst (${dst.length}) lengths differ`);
  if (weights && weights.length !== src.length) throw new Error("kabsch: weights length mismatch");
  const n = src.length;
  const a = new Float64Array(3 * n), b = new Float64Array(3 * n);
  for (let i = 0; i < n; i++) {
    a.set(src[i], 3 * i);
    b.set(dst[i], 3 * i);
  }
  return rigidFitPacked(a, b, n, weights);
}

// ===========================================================================
// ICP
// ===========================================================================

export interface IcpOptions {
  /** subset of moving vertices */
  indices?: ArrayLike<number>;
  /** normals of the moving points (enables normal-compatibility rejection) */
  normals?: ArrayLike<number> | null;
  /** default 50 */
  maxIterations?: number;
  /** point-to-plane (default true) or point-to-point */
  pointToPlane?: boolean;
  /** fraction of the worst correspondences rejected each iteration (default 0.1) */
  trimFraction?: number;
  /** correspondences farther than this are rejected (also passed to the query) */
  maxCorrespondenceDistance?: number;
  /** number of moving points sampled (default 5000) */
  sampleCount?: number;
  onProgress?: ProgressFn;
  /** initial moving → target transform */
  initial?: Mat4;
  /** max angle between corresponding normals when `normals` is given (default 60°) */
  normalThresholdDeg?: number;
  /** convergence: incremental rotation (rad) and translation relative to the sample size (default 1e-7) */
  tolerance?: number;
  seed?: number;
}

export interface IcpResult {
  /** maps moving -> target */
  matrix: Mat4;
  /** RMS closest-point distance of the kept (trimmed) correspondences at the final matrix */
  rms: number;
  iterations: number;
  converged: boolean;
  /** number of kept correspondences in the final evaluation */
  pairs: number;
}

interface IcpState {
  X: Float64Array;
  NX?: Float64Array;
  m: number;
  Y: Float64Array;
  Q: Float64Array;
  NQ: Float64Array;
  D: Float64Array;
  keep: Uint8Array;
  refScale: number;
  centroid: Vec3;
}

function prepareIcp(moving: ArrayLike<number>, opts: IcpOptions): IcpState {
  const count = selectionCount(moving, opts.indices);
  if (count < 3) throw new Error(`icp: needs at least 3 moving points, got ${count}`);
  const slots = sampleSlots(count, Math.max(3, opts.sampleCount ?? 5000), opts.seed ?? 1);
  let m = 0;
  const X = new Float64Array(3 * slots.length);
  const NX = opts.normals ? new Float64Array(3 * slots.length) : undefined;
  let cx = 0, cy = 0, cz = 0;
  for (let k = 0; k < slots.length; k++) {
    const i = opts.indices ? opts.indices[slots[k]] : slots[k];
    const x = moving[3 * i], y = moving[3 * i + 1], z = moving[3 * i + 2];
    if (!Number.isFinite(x + y + z)) continue;
    X[3 * m] = x;
    X[3 * m + 1] = y;
    X[3 * m + 2] = z;
    cx += x;
    cy += y;
    cz += z;
    if (NX && opts.normals) {
      const nx = opts.normals[3 * i], ny = opts.normals[3 * i + 1], nz = opts.normals[3 * i + 2];
      const l = Math.hypot(nx, ny, nz) || 1;
      NX[3 * m] = nx / l;
      NX[3 * m + 1] = ny / l;
      NX[3 * m + 2] = nz / l;
    }
    m++;
  }
  if (m < 3) throw new Error("icp: fewer than 3 finite moving points");
  cx /= m;
  cy /= m;
  cz /= m;
  let ss = 0;
  for (let i = 0; i < m; i++) ss += (X[3 * i] - cx) ** 2 + (X[3 * i + 1] - cy) ** 2 + (X[3 * i + 2] - cz) ** 2;
  return {
    X,
    NX,
    m,
    Y: new Float64Array(3 * m),
    Q: new Float64Array(3 * m),
    NQ: new Float64Array(3 * m),
    D: new Float64Array(m),
    keep: new Uint8Array(m),
    refScale: Math.sqrt(ss / m) || 1,
    centroid: [cx, cy, cz],
  };
}

/**
 * Find correspondences at transform M and trim. Returns the number of kept pairs and their RMS.
 * `absNormals` compares normals up to sign.
 */
function correspond(
  st: IcpState,
  M: Mat4,
  target: ClosestPointQuery,
  opts: IcpOptions,
  normalMode: { abs: boolean; rejectedByNormal: number; valid: number },
): { kept: number; rms: number } {
  const { X, NX, m, Y, Q, NQ, D, keep } = st;
  const maxD = opts.maxCorrespondenceDistance;
  const cosThr = Math.cos(((opts.normalThresholdDeg ?? 60) * Math.PI) / 180);
  let valid = 0, rejN = 0;
  for (let i = 0; i < m; i++) {
    const x = X[3 * i], y = X[3 * i + 1], z = X[3 * i + 2];
    const p: Vec3 = [M[0] * x + M[4] * y + M[8] * z + M[12], M[1] * x + M[5] * y + M[9] * z + M[13], M[2] * x + M[6] * y + M[10] * z + M[14]];
    Y[3 * i] = p[0];
    Y[3 * i + 1] = p[1];
    Y[3 * i + 2] = p[2];
    keep[i] = 0;
    const hit: ClosestPointHit | null = target(p, maxD);
    if (!hit) continue;
    const dist = hit.distance;
    if (!Number.isFinite(dist) || (maxD !== undefined && dist > maxD)) continue;
    let nq = hit.normal;
    const nl = Math.hypot(nq[0], nq[1], nq[2]);
    if (!(nl > 0)) continue;
    nq = [nq[0] / nl, nq[1] / nl, nq[2] / nl];
    if (NX) {
      const a = NX[3 * i], b = NX[3 * i + 1], c = NX[3 * i + 2];
      const nmx = M[0] * a + M[4] * b + M[8] * c, nmy = M[1] * a + M[5] * b + M[9] * c, nmz = M[2] * a + M[6] * b + M[10] * c;
      let cs = nmx * nq[0] + nmy * nq[1] + nmz * nq[2];
      if (normalMode.abs) cs = Math.abs(cs);
      if (cs < cosThr) {
        rejN++;
        continue;
      }
    }
    Q[3 * i] = hit.point[0];
    Q[3 * i + 1] = hit.point[1];
    Q[3 * i + 2] = hit.point[2];
    NQ[3 * i] = nq[0];
    NQ[3 * i + 1] = nq[1];
    NQ[3 * i + 2] = nq[2];
    D[i] = dist;
    keep[i] = 1;
    valid++;
  }
  normalMode.rejectedByNormal = rejN;
  normalMode.valid = valid;
  if (valid === 0) return { kept: 0, rms: NaN };
  // trimming
  const trim = Math.min(0.9, Math.max(0, opts.trimFraction ?? 0.1));
  let thr = Infinity;
  if (trim > 0) {
    const tmp = new Float64Array(valid);
    let t = 0;
    for (let i = 0; i < m; i++) if (keep[i]) tmp[t++] = D[i];
    const k = Math.max(0, Math.ceil((1 - trim) * valid) - 1);
    thr = kthSmallest(tmp, k, valid);
  }
  let kept = 0, ss = 0;
  for (let i = 0; i < m; i++) {
    if (!keep[i]) continue;
    if (D[i] > thr) {
      keep[i] = 0;
      continue;
    }
    kept++;
    ss += D[i] * D[i];
  }
  return { kept, rms: Math.sqrt(ss / Math.max(1, kept)) };
}

/** One linearised point-to-plane step on the kept pairs: incremental transform. */
function pointToPlaneStep(st: IcpState): Mat4 | null {
  const { Y, Q, NQ, keep, m } = st;
  let cx = 0, cy = 0, cz = 0, n = 0;
  for (let i = 0; i < m; i++) {
    if (!keep[i]) continue;
    cx += Y[3 * i];
    cy += Y[3 * i + 1];
    cz += Y[3 * i + 2];
    n++;
  }
  if (n < 6) return null;
  cx /= n;
  cy /= n;
  cz /= n;
  let ss = 0;
  for (let i = 0; i < m; i++) if (keep[i]) ss += (Y[3 * i] - cx) ** 2 + (Y[3 * i + 1] - cy) ** 2 + (Y[3 * i + 2] - cz) ** 2;
  const L = Math.sqrt(ss / n) || 1;
  const A = new Float64Array(36), b = new Float64Array(6), J = new Float64Array(6);
  for (let i = 0; i < m; i++) {
    if (!keep[i]) continue;
    const yx = (Y[3 * i] - cx) / L, yy = (Y[3 * i + 1] - cy) / L, yz = (Y[3 * i + 2] - cz) / L;
    const nx = NQ[3 * i], ny = NQ[3 * i + 1], nz = NQ[3 * i + 2];
    J[0] = yy * nz - yz * ny;
    J[1] = yz * nx - yx * nz;
    J[2] = yx * ny - yy * nx;
    J[3] = nx;
    J[4] = ny;
    J[5] = nz;
    const r = (Y[3 * i] - Q[3 * i]) * nx + (Y[3 * i + 1] - Q[3 * i + 1]) * ny + (Y[3 * i + 2] - Q[3 * i + 2]) * nz;
    for (let a = 0; a < 6; a++) {
      b[a] -= J[a] * r;
      for (let c = a; c < 6; c++) A[a * 6 + c] += J[a] * J[c];
    }
  }
  for (let a = 0; a < 6; a++) for (let c = 0; c < a; c++) A[a * 6 + c] = A[c * 6 + a];
  let tr = 0;
  for (let a = 0; a < 6; a++) tr += A[a * 6 + a];
  // tiny ridge: keeps unconstrained (sliding) directions at zero motion
  const ridge = 1e-9 * (tr / 6) + 1e-300;
  const x = choleskySolve(A, b, 6, ridge) ?? solveLinear(A, b, 6);
  if (!x) return null;
  const R = rotationFromVector(x[0] / L, x[1] / L, x[2] / L);
  // y' = R (y - c) + c + t
  const t: Vec3 = [
    cx + x[3] - (R[0] * cx + R[1] * cy + R[2] * cz),
    cy + x[4] - (R[3] * cx + R[4] * cy + R[5] * cz),
    cz + x[5] - (R[6] * cx + R[7] * cy + R[8] * cz),
  ];
  return mat4FromRotation(R, t);
}

function pointToPointStep(st: IcpState): Mat4 | null {
  const { Y, Q, keep, m } = st;
  let n = 0;
  for (let i = 0; i < m; i++) if (keep[i]) n++;
  if (n < 3) return null;
  const a = new Float64Array(3 * n), b = new Float64Array(3 * n);
  let t = 0;
  for (let i = 0; i < m; i++) {
    if (!keep[i]) continue;
    a[3 * t] = Y[3 * i];
    a[3 * t + 1] = Y[3 * i + 1];
    a[3 * t + 2] = Y[3 * i + 2];
    b[3 * t] = Q[3 * i];
    b[3 * t + 1] = Q[3 * i + 1];
    b[3 * t + 2] = Q[3 * i + 2];
    t++;
  }
  return rigidFitPacked(a, b, n).matrix;
}

/**
 * Iterative closest point. Maps `moving` (packed xyz, local coordinates) onto the surface behind
 * `target` (closest-point oracle in target/world coordinates). Point-to-plane uses the small-angle
 * linearisation solved through 6×6 normal equations (exact rotation applied); each iteration
 * rejects the worst `trimFraction` of the correspondences.
 */
export function icp(moving: ArrayLike<number>, target: ClosestPointQuery, opts: IcpOptions = {}): IcpResult {
  const st = prepareIcp(moving, opts);
  const maxIt = Math.max(0, opts.maxIterations ?? 50);
  const tol = opts.tolerance ?? 1e-7;
  const p2p = opts.pointToPlane ?? true;
  let M: Mat4 = opts.initial ? opts.initial.slice() : identity4();
  const mode = { abs: false, rejectedByNormal: 0, valid: 0 };
  let converged = false;
  let it = 0;
  let prevRms = Infinity, stall = 0;
  for (; it < maxIt; it++) {
    opts.onProgress?.(it / maxIt, `ICP iteration ${it + 1}`);
    let c = correspond(st, M, target, opts, mode);
    if (it === 0 && st.NX && mode.rejectedByNormal > mode.valid) {
      // most pairs fail the normal test: normals are probably oriented inconsistently → compare up to sign
      mode.abs = true;
      c = correspond(st, M, target, opts, mode);
    }
    if (c.kept < (p2p ? 6 : 3)) {
      throw new Error(`icp: too few correspondences (${c.kept}) — check the initial alignment / maxCorrespondenceDistance`);
    }
    const delta = p2p ? pointToPlaneStep(st) : pointToPointStep(st);
    if (!delta) throw new Error("icp: degenerate correspondence set");
    M = mul4(delta, M);
    // convergence on the increment (rotation angle and centroid displacement)
    const ang = rotationAngle(delta);
    const cw = transformPoint(M, st.centroid);
    const before = transformPoint(invertRigid(delta), cw);
    const move = Math.hypot(cw[0] - before[0], cw[1] - before[1], cw[2] - before[2]);
    if (ang < tol && move < tol * st.refScale) {
      converged = true;
      it++;
      break;
    }
    if (Math.abs(prevRms - c.rms) <= 1e-9 * Math.max(c.rms, 1e-300)) {
      if (++stall >= 3) {
        converged = true;
        it++;
        break;
      }
    } else stall = 0;
    prevRms = c.rms;
  }
  M = orthonormalizeRigid(M);
  const fin = correspond(st, M, target, opts, mode);
  opts.onProgress?.(1, "ICP done");
  return { matrix: M, rms: fin.rms, iterations: it, converged, pairs: fin.kept };
}

// ===========================================================================
// Global (multi-scan) alignment
// ===========================================================================

export interface GlobalAlignScan {
  /** packed xyz in the scan's own (local) coordinates */
  positions: ArrayLike<number>;
  /** closest-point oracle on this scan, in its local coordinates */
  query: ClosestPointQuery;
  normals?: ArrayLike<number> | null;
  indices?: ArrayLike<number>;
}

export interface GlobalAlignOptions {
  /** initial local → world matrices (scan 0's is kept fixed). Default identity. */
  initial?: Mat4[];
  /** max sweeps over all scans (default 10) */
  maxRounds?: number;
  /** ICP iterations per scan and sweep (default 30) */
  maxIterations?: number;
  pointToPlane?: boolean;
  trimFraction?: number;
  maxCorrespondenceDistance?: number;
  sampleCount?: number;
  normalThresholdDeg?: number;
  /** stop when every scan moved less than this (rotation in rad, translation relative to scan size); default 1e-6 */
  tolerance?: number;
  onProgress?: ProgressFn;
}

export interface GlobalAlignResult {
  /** per-scan local → world matrices (matrices[0] = initial[0] or identity) */
  matrices: Mat4[];
  /** per-scan trimmed RMS distance to the union of the other scans */
  rms: number[];
  rounds: number;
  converged: boolean;
}

/** Closest-point oracle over the union of scans (each in its local frame placed by M[j]), excluding `skip`. */
function unionQuery(scans: GlobalAlignScan[], M: Mat4[], skip: number): ClosestPointQuery {
  const inv = M.map((m) => invertRigid(m));
  return (p: Vec3, maxDistance?: number) => {
    let best: ClosestPointHit | null = null;
    for (let j = 0; j < scans.length; j++) {
      if (j === skip) continue;
      const h = scans[j].query(transformPoint(inv[j], p), best ? Math.min(best.distance, maxDistance ?? Infinity) : maxDistance);
      if (h && (!best || h.distance < best.distance)) {
        best = { point: transformPoint(M[j], h.point), normal: transformDir(M[j], h.normal), distance: h.distance, faceIndex: h.faceIndex };
      }
    }
    return best;
  };
}

/**
 * Multi-scan fine alignment: sweeps over scans 1..n-1, aligning each (ICP) against the union of
 * all others in their current poses (Gauss–Seidel), until nothing moves. Scan 0 stays fixed.
 */
export function globalAlign(scans: GlobalAlignScan[], opts: GlobalAlignOptions = {}): GlobalAlignResult {
  const n = scans.length;
  if (n < 2) throw new Error("globalAlign: needs at least 2 scans");
  if (opts.initial && opts.initial.length !== n) throw new Error("globalAlign: initial must have one matrix per scan");
  const M: Mat4[] = scans.map((_, i) => (opts.initial ? opts.initial[i].slice() : identity4()));
  const rms = new Array<number>(n).fill(NaN);
  const maxRounds = opts.maxRounds ?? 10;
  const tol = opts.tolerance ?? 1e-6;
  const sizes = scans.map((s) => {
    const st = prepareIcp(s.positions, { indices: s.indices, sampleCount: 2000 });
    return { c: st.centroid, r: st.refScale };
  });
  let converged = false;
  let round = 0;
  for (; round < maxRounds; round++) {
    let moved = false;
    for (let i = 1; i < n; i++) {
      opts.onProgress?.((round * (n - 1) + (i - 1)) / (maxRounds * (n - 1)), `Global alignment: round ${round + 1}, scan ${i + 1}/${n}`);
      const res = icp(scans[i].positions, unionQuery(scans, M, i), {
        indices: scans[i].indices,
        normals: scans[i].normals,
        initial: M[i],
        maxIterations: opts.maxIterations ?? 30,
        pointToPlane: opts.pointToPlane,
        trimFraction: opts.trimFraction,
        maxCorrespondenceDistance: opts.maxCorrespondenceDistance,
        sampleCount: opts.sampleCount,
        normalThresholdDeg: opts.normalThresholdDeg,
      });
      const diff = rigidDifference(M[i], res.matrix, sizes[i].c);
      if (diff.angle > tol || diff.translation > tol * sizes[i].r) moved = true;
      M[i] = res.matrix;
      rms[i] = res.rms;
    }
    if (!moved) {
      converged = true;
      round++;
      break;
    }
  }
  // residual of the fixed scan (evaluation only)
  const ev = icp(scans[0].positions, unionQuery(scans, M, 0), {
    indices: scans[0].indices,
    normals: scans[0].normals,
    initial: M[0],
    maxIterations: 0,
    trimFraction: opts.trimFraction,
    maxCorrespondenceDistance: opts.maxCorrespondenceDistance,
    sampleCount: opts.sampleCount,
    normalThresholdDeg: opts.normalThresholdDeg,
  });
  rms[0] = ev.rms;
  opts.onProgress?.(1, "Global alignment done");
  return { matrices: M, rms, rounds: round, converged };
}

// ===========================================================================
// Object align (to world) and N-point alignment
// ===========================================================================

export type WorldAxis = "X" | "Y" | "Z";

export interface AlignToWorldSpec {
  /** primary entity: a plane (direction = its normal) or an axis (direction = axis), with a point on it */
  primary: { kind: "plane" | "axis"; direction: Vec3; point: Vec3 };
  /** secondary entity: only its direction is used (orthogonalised against the primary) */
  secondary?: { kind: "plane" | "axis"; direction: Vec3; point?: Vec3 };
  /** point mapped to the world origin (default: primary.point, or the primary/secondary intersection, see below) */
  origin?: Vec3;
  primaryTarget: WorldAxis;
  secondaryTarget: WorldAxis;
  flipPrimary?: boolean;
  flipSecondary?: boolean;
}

const AXES: Record<WorldAxis, Vec3> = { X: [1, 0, 0], Y: [0, 1, 0], Z: [0, 0, 1] };

/**
 * QuickSurface-style "Object Align": rigid transform that maps the primary direction onto the
 * chosen world axis, the secondary direction (made orthogonal to the primary) onto the second
 * world axis, and `origin` onto (0,0,0).
 *
 * Without a secondary entity the rotation is the minimal one taking the primary direction to its
 * target. Without an explicit origin: when the primary is a plane and the secondary an axis with a
 * point, the axis/plane intersection is used; otherwise primary.point.
 */
export function alignToWorld(spec: AlignToWorldSpec): Mat4 {
  let P = normalize(spec.primary.direction);
  if (length(P) === 0) throw new Error("alignToWorld: zero primary direction");
  if (spec.flipPrimary) P = [-P[0], -P[1], -P[2]];
  const T1 = AXES[spec.primaryTarget];
  if (!T1) throw new Error(`alignToWorld: invalid primary target '${spec.primaryTarget}'`);
  let R: Float64Array; // row-major, maps source → world
  if (spec.secondary) {
    const T2 = AXES[spec.secondaryTarget];
    if (!T2) throw new Error(`alignToWorld: invalid secondary target '${spec.secondaryTarget}'`);
    if (spec.primaryTarget === spec.secondaryTarget) throw new Error("alignToWorld: primary and secondary targets must be different axes");
    let S = normalize(spec.secondary.direction);
    S = [S[0] - dot(S, P) * P[0], S[1] - dot(S, P) * P[1], S[2] - dot(S, P) * P[2]];
    if (length(S) < 1e-6) throw new Error("alignToWorld: secondary direction is parallel to the primary direction");
    S = normalize(S);
    if (spec.flipSecondary) S = [-S[0], -S[1], -S[2]];
    const U = cross(P, S);
    const T3 = cross(T1, T2);
    // R = [T1 T2 T3] [P S U]^T
    R = new Float64Array(9);
    for (let r = 0; r < 3; r++)
      for (let c = 0; c < 3; c++) R[r * 3 + c] = T1[r] * P[c] + T2[r] * S[c] + T3[r] * U[c];
  } else {
    // minimal rotation P → T1
    const v = cross(P, T1);
    const s = length(v), c = dot(P, T1);
    if (s < 1e-12) {
      if (c > 0) R = Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1);
      else {
        // 180° about any axis ⟂ P
        const ax = normalize(cross(P, Math.abs(P[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]));
        R = rotationFromVector(ax[0] * Math.PI, ax[1] * Math.PI, ax[2] * Math.PI);
      }
    } else {
      const ang = Math.atan2(s, c);
      R = rotationFromVector((v[0] / s) * ang, (v[1] / s) * ang, (v[2] / s) * ang);
    }
  }
  let o = spec.origin;
  if (!o) {
    o = spec.primary.point;
    const sec = spec.secondary;
    if (spec.primary.kind === "plane" && sec && sec.kind === "axis" && sec.point) {
      const d = normalize(sec.direction);
      const den = dot(d, P);
      if (Math.abs(den) > 1e-9) {
        const t = dot([spec.primary.point[0] - sec.point[0], spec.primary.point[1] - sec.point[1], spec.primary.point[2] - sec.point[2]], P) / den;
        o = [sec.point[0] + t * d[0], sec.point[1] + t * d[1], sec.point[2] + t * d[2]];
      }
    }
  }
  const t: Vec3 = [
    -(R[0] * o[0] + R[1] * o[1] + R[2] * o[2]),
    -(R[3] * o[0] + R[4] * o[1] + R[5] * o[2]),
    -(R[6] * o[0] + R[7] * o[1] + R[8] * o[2]),
  ];
  return mat4FromRotation(R, t);
}

export interface NPointAlignResult {
  /** maps moving -> fixed */
  matrix: Mat4;
  rms: number;
  /** |M moving_i - fixed_i| per pair */
  perPairErrors: number[];
}

/** Rigid alignment from ≥ 3 picked point pairs (non-collinear). */
export function nPointAlign(pairs: { moving: Vec3; fixed: Vec3 }[], weights?: number[]): NPointAlignResult {
  if (pairs.length < 3) throw new Error(`nPointAlign: needs at least 3 point pairs, got ${pairs.length}`);
  // collinearity check on the moving points
  const n = pairs.length;
  let cx = 0, cy = 0, cz = 0;
  for (const p of pairs) {
    cx += p.moving[0];
    cy += p.moving[1];
    cz += p.moving[2];
  }
  cx /= n;
  cy /= n;
  cz /= n;
  const cov = new Float64Array(9);
  for (const p of pairs) {
    const x = p.moving[0] - cx, y = p.moving[1] - cy, z = p.moving[2] - cz;
    cov[0] += x * x;
    cov[1] += x * y;
    cov[2] += x * z;
    cov[4] += y * y;
    cov[5] += y * z;
    cov[8] += z * z;
  }
  cov[3] = cov[1];
  cov[6] = cov[2];
  cov[7] = cov[5];
  const e = eigenSym3(cov);
  if (!(e.values[1] > 1e-10 * e.values[2])) throw new Error("nPointAlign: the moving points are (nearly) collinear — the rotation is undetermined");
  const fit = kabsch(
    pairs.map((p) => p.moving),
    pairs.map((p) => p.fixed),
    weights,
  );
  const perPairErrors = pairs.map((p) => {
    const q = transformPoint(fit.matrix, p.moving);
    return Math.hypot(q[0] - p.fixed[0], q[1] - p.fixed[1], q[2] - p.fixed[2]);
  });
  return { matrix: fit.matrix, rms: fit.rms, perPairErrors };
}
