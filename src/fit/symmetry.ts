/**
 * Symmetry plane estimation and the symmetry deviation map ("symmetry analyser").
 *
 * Each candidate plane is refined ICP-style: the sample is mirrored, closest points on the
 * original surface are found, a linearised point-to-plane rigid increment T is solved and the
 * improper map T∘H is projected back onto the nearest reflection (eigenvector of its symmetric
 * part) — so the update always stays a pure reflection.
 */
import type { ClosestPointQuery, PlaneDef, Vec3 } from "../core/types";
import { choleskySolve, eigenSym3, solveLinear } from "../core/linalg";
import { dot, normalize } from "../core/vec";
import { kthSmallest, sampleSlots, selectionCount } from "./common";
import { rotationFromVector } from "./transform";

export interface SymmetryOptions {
  /** per-vertex normals (enables normal-compatibility rejection of mirrored pairs) */
  normals?: ArrayLike<number> | null;
  /** closest-point oracle on the ORIGINAL surface, same coordinates as `pos` */
  query: ClosestPointQuery;
  /** additional user-provided starting plane */
  initial?: PlaneDef;
  /** number of sample points (default 3000) */
  sampleCount?: number;
  indices?: ArrayLike<number>;
  /** iterations per candidate (default 60) */
  maxIterations?: number;
  /** fraction of worst mirrored pairs rejected (default 0.2: parts are rarely perfectly symmetric / complete) */
  trimFraction?: number;
  /** max angle between mirrored normal and surface normal (default 60°) */
  normalThresholdDeg?: number;
  seed?: number;
}

export interface SymmetryCandidate {
  plane: PlaneDef;
  /** trimmed RMS distance between mirrored samples and the surface */
  rms: number;
  /** fraction of samples that found a compatible mirrored correspondence */
  coverage: number;
  iterations: number;
}

export interface SymmetryResult {
  plane: PlaneDef;
  rms: number;
  candidates: SymmetryCandidate[];
}

interface SymSample {
  /** centred, scaled coordinates */
  X: Float64Array;
  N?: Float64Array;
  m: number;
  c: Vec3;
  s: number;
}

function buildSymSample(pos: ArrayLike<number>, opts: SymmetryOptions): SymSample {
  const count = selectionCount(pos, opts.indices);
  if (count < 10) throw new Error(`estimateSymmetryPlane: needs at least 10 points, got ${count}`);
  const slots = sampleSlots(count, Math.max(10, opts.sampleCount ?? 3000), opts.seed ?? 1);
  const raw = new Float64Array(3 * slots.length);
  const N = opts.normals ? new Float64Array(3 * slots.length) : undefined;
  let m = 0, cx = 0, cy = 0, cz = 0;
  for (let k = 0; k < slots.length; k++) {
    const i = opts.indices ? opts.indices[slots[k]] : slots[k];
    const x = pos[3 * i], y = pos[3 * i + 1], z = pos[3 * i + 2];
    if (!Number.isFinite(x + y + z)) continue;
    raw[3 * m] = x;
    raw[3 * m + 1] = y;
    raw[3 * m + 2] = z;
    if (N && opts.normals) {
      const nx = opts.normals[3 * i], ny = opts.normals[3 * i + 1], nz = opts.normals[3 * i + 2];
      const l = Math.hypot(nx, ny, nz) || 1;
      N[3 * m] = nx / l;
      N[3 * m + 1] = ny / l;
      N[3 * m + 2] = nz / l;
    }
    cx += x;
    cy += y;
    cz += z;
    m++;
  }
  cx /= m;
  cy /= m;
  cz /= m;
  let ss = 0;
  for (let i = 0; i < m; i++) ss += (raw[3 * i] - cx) ** 2 + (raw[3 * i + 1] - cy) ** 2 + (raw[3 * i + 2] - cz) ** 2;
  const s = Math.sqrt(ss / m);
  if (!(s > 0)) throw new Error("estimateSymmetryPlane: all points coincide");
  const X = new Float64Array(3 * m);
  for (let i = 0; i < 3 * m; i += 3) {
    X[i] = (raw[i] - cx) / s;
    X[i + 1] = (raw[i + 1] - cy) / s;
    X[i + 2] = (raw[i + 2] - cz) / s;
  }
  return { X, N, m, c: [cx, cy, cz], s };
}

/**
 * Mirror the sample across (n, d) (normalised coords), query correspondences, trim.
 * Fills Y (mirrored), Q (closest), NQ, keep; returns kept count, trimmed RMS (normalised) and coverage.
 */
function mirrorPairs(
  S: SymSample,
  n: Vec3,
  d: number,
  query: ClosestPointQuery,
  trim: number,
  cosThr: number,
  buf: { Y: Float64Array; Q: Float64Array; NQ: Float64Array; D: Float64Array; keep: Uint8Array },
): { kept: number; rms: number; coverage: number } {
  const { X, N, m, c, s } = S;
  const { Y, Q, NQ, D, keep } = buf;
  let valid = 0;
  for (let i = 0; i < m; i++) {
    const x = X[3 * i], y = X[3 * i + 1], z = X[3 * i + 2];
    const k = 2 * (x * n[0] + y * n[1] + z * n[2] - d);
    const yx = x - k * n[0], yy = y - k * n[1], yz = z - k * n[2];
    Y[3 * i] = yx;
    Y[3 * i + 1] = yy;
    Y[3 * i + 2] = yz;
    keep[i] = 0;
    const hit = query([c[0] + s * yx, c[1] + s * yy, c[2] + s * yz]);
    if (!hit || !Number.isFinite(hit.distance)) continue;
    const nl = Math.hypot(hit.normal[0], hit.normal[1], hit.normal[2]);
    if (!(nl > 0)) continue;
    const nq: Vec3 = [hit.normal[0] / nl, hit.normal[1] / nl, hit.normal[2] / nl];
    if (N) {
      // mirrored normal H n
      const a = N[3 * i], b = N[3 * i + 1], cc = N[3 * i + 2];
      const kn = 2 * (a * n[0] + b * n[1] + cc * n[2]);
      const mx = a - kn * n[0], my = b - kn * n[1], mz = cc - kn * n[2];
      if (mx * nq[0] + my * nq[1] + mz * nq[2] < cosThr) continue;
    }
    Q[3 * i] = (hit.point[0] - c[0]) / s;
    Q[3 * i + 1] = (hit.point[1] - c[1]) / s;
    Q[3 * i + 2] = (hit.point[2] - c[2]) / s;
    NQ[3 * i] = nq[0];
    NQ[3 * i + 1] = nq[1];
    NQ[3 * i + 2] = nq[2];
    D[i] = hit.distance / s;
    keep[i] = 1;
    valid++;
  }
  if (valid < 6) return { kept: valid, rms: Infinity, coverage: valid / m };
  const tmp = new Float64Array(valid);
  let t = 0;
  for (let i = 0; i < m; i++) if (keep[i]) tmp[t++] = D[i];
  const thr = trim > 0 ? kthSmallest(tmp, Math.max(0, Math.ceil((1 - trim) * valid) - 1), valid) : Infinity;
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
  // rms is reported over all valid-but-trimmed pairs; penalise low coverage so that planes that
  // mirror the part mostly into empty space do not win
  return { kept, rms: Math.sqrt(ss / Math.max(1, kept)), coverage: valid / m };
}

/** One refinement step: returns the updated plane (n, d) in normalised coordinates. */
function reflectionUpdate(
  S: SymSample,
  n: Vec3,
  d: number,
  buf: { Y: Float64Array; Q: Float64Array; NQ: Float64Array; keep: Uint8Array },
): { n: Vec3; d: number } | null {
  const { X, m } = S;
  const { Y, Q, NQ, keep } = buf;
  let cx = 0, cy = 0, cz = 0, cnt = 0;
  let xx = 0, xy = 0, xz = 0;
  for (let i = 0; i < m; i++) {
    if (!keep[i]) continue;
    cx += Y[3 * i];
    cy += Y[3 * i + 1];
    cz += Y[3 * i + 2];
    xx += X[3 * i];
    xy += X[3 * i + 1];
    xz += X[3 * i + 2];
    cnt++;
  }
  if (cnt < 6) return null;
  cx /= cnt;
  cy /= cnt;
  cz /= cnt;
  const xbar: Vec3 = [xx / cnt, xy / cnt, xz / cnt];
  const A = new Float64Array(36), b = new Float64Array(6), J = new Float64Array(6);
  for (let i = 0; i < m; i++) {
    if (!keep[i]) continue;
    const yx = Y[3 * i] - cx, yy = Y[3 * i + 1] - cy, yz = Y[3 * i + 2] - cz;
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
  const x = choleskySolve(A, b, 6, 1e-9 * (tr / 6) + 1e-300) ?? solveLinear(A, b, 6);
  if (!x) return null;
  const R = rotationFromVector(x[0], x[1], x[2]);
  // increment: y' = R (y - c) + c + t  →  y' = R y + T
  const T: Vec3 = [
    cx + x[3] - (R[0] * cx + R[1] * cy + R[2] * cz),
    cy + x[4] - (R[3] * cx + R[4] * cy + R[5] * cz),
    cz + x[5] - (R[6] * cx + R[7] * cy + R[8] * cz),
  ];
  // composed improper map: x ↦ R H x + (2 d R n + T),   H = I - 2 n nᵀ
  const H = [1 - 2 * n[0] * n[0], -2 * n[0] * n[1], -2 * n[0] * n[2], -2 * n[1] * n[0], 1 - 2 * n[1] * n[1], -2 * n[1] * n[2], -2 * n[2] * n[0], -2 * n[2] * n[1], 1 - 2 * n[2] * n[2]];
  const Am = new Float64Array(9);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) Am[r * 3 + c] = R[r * 3] * H[c] + R[r * 3 + 1] * H[3 + c] + R[r * 3 + 2] * H[6 + c];
  const Rn: Vec3 = [R[0] * n[0] + R[1] * n[1] + R[2] * n[2], R[3] * n[0] + R[4] * n[1] + R[5] * n[2], R[6] * n[0] + R[7] * n[1] + R[8] * n[2]];
  const bv: Vec3 = [2 * d * Rn[0] + T[0], 2 * d * Rn[1] + T[1], 2 * d * Rn[2] + T[2]];
  // nearest reflection: n' = eigenvector of sym(A) with the smallest eigenvalue
  const sym = [Am[0], (Am[1] + Am[3]) / 2, (Am[2] + Am[6]) / 2, (Am[1] + Am[3]) / 2, Am[4], (Am[5] + Am[7]) / 2, (Am[2] + Am[6]) / 2, (Am[5] + Am[7]) / 2, Am[8]];
  let n2 = eigenSym3(sym).vectors[0];
  if (dot(n2, n) < 0) n2 = [-n2[0], -n2[1], -n2[2]];
  // offset: the midpoint of x̄ and its image lies on the plane
  const ax: Vec3 = [
    Am[0] * xbar[0] + Am[1] * xbar[1] + Am[2] * xbar[2] + bv[0],
    Am[3] * xbar[0] + Am[4] * xbar[1] + Am[5] * xbar[2] + bv[1],
    Am[6] * xbar[0] + Am[7] * xbar[1] + Am[8] * xbar[2] + bv[2],
  ];
  const d2 = 0.5 * dot(n2, [xbar[0] + ax[0], xbar[1] + ax[1], xbar[2] + ax[2]]);
  return { n: n2, d: d2 };
}

function refineCandidate(
  S: SymSample,
  n0: Vec3,
  d0: number,
  opts: SymmetryOptions,
  buf: { Y: Float64Array; Q: Float64Array; NQ: Float64Array; D: Float64Array; keep: Uint8Array },
): { n: Vec3; d: number; rms: number; coverage: number; iterations: number } {
  const trim = Math.min(0.9, Math.max(0, opts.trimFraction ?? 0.2));
  const cosThr = Math.cos(((opts.normalThresholdDeg ?? 60) * Math.PI) / 180);
  const maxIt = opts.maxIterations ?? 60;
  let n = normalize(n0), d = d0;
  let total = 0;
  // Phase 1 without trimming: the pairs that pin down "sliding" directions (e.g. the end faces of
  // a prismatic part) are exactly the ones a fixed trim would discard while the plane is still off.
  // Phase 2 with trimming removes genuinely asymmetric regions.
  const phases = trim > 0 ? [0, trim] : [0];
  for (const tr of phases) {
    const budget = tr === 0 && phases.length > 1 ? Math.ceil(maxIt / 2) : maxIt - total;
    for (let it = 0; it < budget; it++) {
      const pr = mirrorPairs(S, n, d, opts.query, tr, cosThr, buf);
      if (pr.kept < 6) break;
      const up = reflectionUpdate(S, n, d, buf);
      if (!up) break;
      const dAng = Math.acos(Math.min(1, Math.abs(dot(up.n, n))));
      const dOff = Math.abs(up.d - d);
      n = up.n;
      d = up.d;
      total++;
      if (dAng < 1e-9 && dOff < 1e-9) break;
    }
  }
  const fin = mirrorPairs(S, n, d, opts.query, trim, cosThr, buf);
  return { n, d, rms: fin.rms, coverage: fin.coverage, iterations: total };
}

/**
 * Estimate the best symmetry plane of a part. Candidates: the three PCA planes through the
 * centroid (+ `initial`), each refined iteratively; the one with the lowest trimmed RMS (with a
 * coverage penalty) wins. `rms` is in model units.
 */
export function estimateSymmetryPlane(pos: ArrayLike<number>, opts: SymmetryOptions): SymmetryResult {
  if (!opts || typeof opts.query !== "function") throw new Error("estimateSymmetryPlane: a closest-point query on the original surface is required");
  const S = buildSymSample(pos, opts);
  const cov = new Float64Array(9);
  for (let i = 0; i < S.m; i++) {
    const x = S.X[3 * i], y = S.X[3 * i + 1], z = S.X[3 * i + 2];
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
  const starts: { n: Vec3; d: number }[] = [
    { n: e.vectors[2], d: 0 },
    { n: e.vectors[1], d: 0 },
    { n: e.vectors[0], d: 0 },
  ];
  if (opts.initial) {
    const n = normalize(opts.initial.normal);
    const o = opts.initial.origin;
    const on: Vec3 = [(o[0] - S.c[0]) / S.s, (o[1] - S.c[1]) / S.s, (o[2] - S.c[2]) / S.s];
    starts.unshift({ n, d: dot(n, on) });
  }
  const buf = {
    Y: new Float64Array(3 * S.m),
    Q: new Float64Array(3 * S.m),
    NQ: new Float64Array(3 * S.m),
    D: new Float64Array(S.m),
    keep: new Uint8Array(S.m),
  };
  const candidates: (SymmetryCandidate & { score: number })[] = [];
  for (const st of starts) {
    const r = refineCandidate(S, st.n, st.d, opts, buf);
    const n = r.n;
    const origin: Vec3 = [S.c[0] + S.s * r.d * n[0], S.c[1] + S.s * r.d * n[1], S.c[2] + S.s * r.d * n[2]];
    const rms = r.rms * S.s;
    const cov = Math.max(r.coverage, 1e-3);
    candidates.push({ plane: { kind: "plane", origin, normal: n }, rms, coverage: r.coverage, iterations: r.iterations, score: rms / cov });
  }
  candidates.sort((a, b) => a.score - b.score);
  const best = candidates[0];
  if (!Number.isFinite(best.rms)) throw new Error("estimateSymmetryPlane: no candidate found enough mirrored correspondences");
  return {
    plane: best.plane,
    rms: best.rms,
    candidates: candidates.map(({ plane, rms, coverage, iterations }) => ({ plane, rms, coverage, iterations })),
  };
}

/**
 * Per-vertex signed distance of the mirrored point to the surface (positive on the outer side of
 * the surface at the closest point; NaN when the query finds nothing). Selection order.
 */
export function symmetryDeviation(
  pos: ArrayLike<number>,
  plane: PlaneDef,
  query: ClosestPointQuery,
  indices?: ArrayLike<number>,
  maxDistance?: number,
): Float32Array {
  const n = normalize(plane.normal);
  const o = plane.origin;
  const count = selectionCount(pos, indices);
  const out = new Float32Array(count);
  for (let k = 0; k < count; k++) {
    const i = indices ? indices[k] : k;
    const x = pos[3 * i], y = pos[3 * i + 1], z = pos[3 * i + 2];
    const s = 2 * ((x - o[0]) * n[0] + (y - o[1]) * n[1] + (z - o[2]) * n[2]);
    const p: Vec3 = [x - s * n[0], y - s * n[1], z - s * n[2]];
    const hit = query(p, maxDistance);
    if (!hit) {
      out[k] = NaN;
      continue;
    }
    const sd = (p[0] - hit.point[0]) * hit.normal[0] + (p[1] - hit.point[1]) * hit.normal[1] + (p[2] - hit.point[2]) * hit.normal[2];
    out[k] = sd < 0 ? -hit.distance : hit.distance;
  }
  return out;
}
