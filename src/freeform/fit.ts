/**
 * Snap-to-mesh fitting of a Catmull–Clark cage to a target surface (scan).
 *
 * The target is only accessed through a `ClosestPointQuery`. Each outer iteration:
 *   1. evaluates limit-surface samples Q = S P (S: sparse subdivision/limit stencil matrix),
 *   2. finds the closest target points T (and target normals n),
 *   3. solves the regularised least-squares problem
 *        min  sum_s w_s (S_s P - T_s)^T M_s (S_s P - T_s)  +  lambda |L P|^2  +  mu |P - P_prev|^2
 *      with M_s = (1 - omega) I + omega n n^T  (omega = point-to-plane weight),
 *      L the cage umbrella Laplacian (crease/boundary vertices use their 1D neighbours) — i.e. the
 *      3N x 3N SPD system (S^T M S + lambda L^T L + mu I) P = S^T M T + mu P_prev, solved with a
 *      block-Jacobi preconditioned CG (warm-started). Fixed vertices are Dirichlet constraints.
 */
import type { ClosestPointHit, ClosestPointQuery, ProgressFn, Vec3 } from "../core/types";
import { limitNormals, limitStencilMatrix, subdivisionStencils } from "./catmullClark";
import { buildTopology, cageSize, cloneQuadMesh, sharpNeighbours, VertexKind, type QuadMesh, type Topology } from "./quadMesh";
import {
  bsr3BlockIndex,
  bsr3FromRowColumns,
  CsrBuilder,
  csrMulPacked3,
  csrMultiply,
  csrMulVec3,
  csrTranspose,
  pcgBsr3,
  type CsrMatrix,
} from "./sparse";

export type RaycastFn = (origin: Vec3, dir: Vec3) => ClosestPointHit | null;

export interface FitCageOptions {
  /** subdivision levels used for the samples (1..3, default 2: 16 samples per quad) */
  levels?: number;
  /** outer closest-point iterations (default 8) */
  iterations?: number;
  /** fairness weight on the cage (umbrella Laplacian, dimensionless, default 0.005) */
  smoothness?: number;
  /** point-to-plane weight omega in [0, 1] (default 0.8; 0 = pure point-to-point) */
  pointToPlane?: number;
  /** relative damping towards the previous iterate (default 1e-3) */
  damping?: number;
  /** vertices that must not move (overrides qm.fixed) */
  fixed?: boolean[];
  /** samples whose closest target point is farther than this are ignored (outliers / holes) */
  maxDistance?: number;
  /**
   * Optional target for the samples on the cage boundary (e.g. the scan's boundary or a feature
   * curve as a closest-point oracle). Boundary samples are fitted point-to-point to it.
   */
  boundaryTarget?: ClosestPointQuery;
  /** weight of the boundary samples when `boundaryTarget` is given (default 4) */
  boundaryWeight?: number;
  /** stop when the rms improves by less than tolerance * cage size (default 1e-7) */
  tolerance?: number;
  /** CG relative tolerance (default 1e-9) */
  cgTolerance?: number;
  onProgress?: ProgressFn;
}

export interface CageFitResult {
  cage: QuadMesh;
  /** rms distance of the limit samples to the target after the fit */
  rms: number;
  /** max distance of the limit samples to the target after the fit */
  maxDev: number;
  /** outer iterations performed */
  iterations: number;
  /** rms before each iteration, plus the final value */
  history: number[];
  samples: number;
  /** samples with a valid target hit in the final evaluation */
  hits: number;
}

/** Umbrella Laplacian of the cage (rows: smooth/dart = v - mean(neighbours), crease = v - mean(crease neighbours), corner = 0). */
export function cageLaplacian(topo: Topology): CsrMatrix {
  const b = new CsrBuilder(topo.nv, topo.nv * 6);
  for (let v = 0; v < topo.nv; v++) {
    const nOut = topo.vOutStart[v + 1] - topo.vOutStart[v];
    const kind = topo.vKind[v];
    if (nOut > 0 && (kind === VertexKind.Smooth || kind === VertexKind.Dart)) {
      const s = topo.vEdgeStart[v], e = topo.vEdgeStart[v + 1];
      const n = e - s;
      if (n > 0) {
        b.add(v, 1);
        for (let p = s; p < e; p++) {
          const ed = topo.vEdges[p];
          b.add(topo.edgeV[2 * ed] === v ? topo.edgeV[2 * ed + 1] : topo.edgeV[2 * ed], -1 / n);
        }
      }
    } else if (nOut > 0 && kind === VertexKind.Crease) {
      const nb = sharpNeighbours(topo, v);
      if (nb.length === 2) {
        b.add(v, 1);
        b.add(nb[0], -0.5);
        b.add(nb[1], -0.5);
      }
    }
    b.endRow();
  }
  return b.build();
}

interface SampleGroups {
  order: Int32Array;
  start: Int32Array;
  support: Int32Array[];
  /** local column (index into the group's support) per non-zero of S */
  localCol: Int32Array;
}

function buildGroups(S: CsrMatrix, faceOfSample: Int32Array, nFaces: number): SampleGroups {
  const ns = S.nRows;
  const key = new Int32Array(ns);
  let extra = 0;
  for (let s = 0; s < ns; s++) key[s] = faceOfSample[s] >= 0 ? faceOfSample[s] : nFaces + extra++;
  const nG = nFaces + extra;
  const start = new Int32Array(nG + 1);
  for (let s = 0; s < ns; s++) start[key[s] + 1]++;
  for (let g = 0; g < nG; g++) start[g + 1] += start[g];
  const order = new Int32Array(ns);
  const fill = start.slice(0, nG);
  for (let s = 0; s < ns; s++) order[fill[key[s]]++] = s;
  const support: Int32Array[] = new Array(nG);
  const localCol = new Int32Array(S.colIdx.length);
  const stamp = new Int32Array(S.nCols).fill(-1);
  const pos = new Int32Array(S.nCols);
  for (let g = 0; g < nG; g++) {
    const cols: number[] = [];
    for (let k = start[g]; k < start[g + 1]; k++) {
      const s = order[k];
      for (let p = S.rowPtr[s]; p < S.rowPtr[s + 1]; p++) {
        const c = S.colIdx[p];
        if (stamp[c] !== g) {
          stamp[c] = g;
          cols.push(c);
        }
      }
    }
    cols.sort((a, b) => a - b);
    const sup = Int32Array.from(cols);
    support[g] = sup;
    for (let i = 0; i < sup.length; i++) pos[sup[i]] = i;
    for (let k = start[g]; k < start[g + 1]; k++) {
      const s = order[k];
      for (let p = S.rowPtr[s]; p < S.rowPtr[s + 1]; p++) localCol[p] = pos[S.colIdx[p]];
    }
  }
  return { order, start, support, localCol };
}

function resolveFixed(f: boolean[] | undefined, n: number): Uint8Array {
  const out = new Uint8Array(n);
  if (f) for (let i = 0; i < n; i++) out[i] = f[i] ? 1 : 0;
  return out;
}

/** Fit the cage so that its limit surface follows the target (see module doc). */
export function fitCageToTarget(qm: QuadMesh, target: ClosestPointQuery, opts: FitCageOptions = {}): CageFitResult {
  const levels = Math.max(0, Math.min(3, Math.round(opts.levels ?? 2)));
  const iterations = Math.max(0, Math.round(opts.iterations ?? 8));
  const omega = Math.max(0, Math.min(1, opts.pointToPlane ?? 0.8));
  const smoothness = Math.max(0, opts.smoothness ?? 0.005);
  const damping = Math.max(0, opts.damping ?? 1e-3);
  const maxDistance = opts.maxDistance;
  const boundaryWeight = opts.boundaryWeight ?? 4;
  const size = Math.max(cageSize(qm), 1e-12);
  const tol = (opts.tolerance ?? 1e-7) * size;
  const N = qm.vertices.length;
  const fixed = resolveFixed(opts.fixed ?? qm.fixed, N);

  const st = subdivisionStencils(qm, levels);
  const S = st.rows;
  const ns = S.nRows;
  const topo = buildTopology(qm);
  const groups = buildGroups(S, st.faceOfSample, qm.faces.length);
  const nG = groups.support.length;

  // regulariser
  const Lap = cageLaplacian(topo);
  const LtL = csrMultiply(csrTranspose(Lap), Lap);

  // block pattern
  const rowCols: number[][] = Array.from({ length: N }, () => []);
  for (let g = 0; g < nG; g++) {
    const sup = groups.support[g];
    for (let a = 0; a < sup.length; a++) {
      const r = rowCols[sup[a]];
      for (let b = 0; b < sup.length; b++) r.push(sup[b]);
    }
  }
  for (let i = 0; i < N; i++) for (let p = LtL.rowPtr[i]; p < LtL.rowPtr[i + 1]; p++) rowCols[i].push(LtL.colIdx[p]);
  const A = bsr3FromRowColumns(N, rowCols);
  rowCols.length = 0;
  const blockPos: Int32Array[] = new Array(nG);
  let maxM = 0;
  for (let g = 0; g < nG; g++) {
    const sup = groups.support[g];
    const m = sup.length;
    maxM = Math.max(maxM, m);
    const bp = new Int32Array(m * m);
    for (let a = 0; a < m; a++) for (let b = 0; b < m; b++) bp[a * m + b] = bsr3BlockIndex(A, sup[a], sup[b]);
    blockPos[g] = bp;
  }
  const ltlPos = new Int32Array(LtL.colIdx.length);
  for (let i = 0; i < N; i++) for (let p = LtL.rowPtr[i]; p < LtL.rowPtr[i + 1]; p++) ltlPos[p] = bsr3BlockIndex(A, i, LtL.colIdx[p]);

  let traceStS = 0;
  for (let p = 0; p < S.values.length; p++) traceStS += S.values[p] * S.values[p];
  let traceLtL = 0;
  for (let p = 0; p < Lap.values.length; p++) traceLtL += Lap.values[p] * Lap.values[p];
  const lambda = traceLtL > 0 ? (smoothness * traceStS) / traceLtL : 0;
  const mu = Math.max((damping * traceStS) / Math.max(1, N), 1e-12 * traceStS / Math.max(1, N));

  const P = new Float64Array(3 * N);
  qm.vertices.forEach((p, i) => P.set(p, 3 * i));
  const Q = new Float64Array(3 * ns);
  const T = new Float64Array(3 * ns);
  const Nrm = new Float64Array(3 * ns);
  const W = new Float64Array(ns);
  const PtoPl = new Uint8Array(ns);
  const rhs = new Float64Array(3 * N);
  const loc = new Float64Array(maxM * maxM * 9);
  const locR = new Float64Array(maxM * 3);
  const vals = A.values;
  const useBoundaryTarget = !!opts.boundaryTarget;

  const evaluate = (forFit: boolean): { rms: number; maxDev: number; hits: number } => {
    csrMulPacked3(S, P, Q);
    let sum = 0, cnt = 0, maxDev = 0;
    for (let s = 0; s < ns; s++) {
      const q: Vec3 = [Q[3 * s], Q[3 * s + 1], Q[3 * s + 2]];
      const hit = target(q, maxDistance);
      const valid = !!hit && Number.isFinite(hit.distance) && (maxDistance === undefined || hit.distance <= maxDistance);
      if (valid) {
        const d = Math.hypot(hit.point[0] - q[0], hit.point[1] - q[1], hit.point[2] - q[2]);
        sum += d * d;
        cnt++;
        if (d > maxDev) maxDev = d;
      }
      if (!forFit) continue;
      if (useBoundaryTarget && st.boundarySample[s]) {
        const bh = opts.boundaryTarget!(q);
        if (bh) {
          T[3 * s] = bh.point[0];
          T[3 * s + 1] = bh.point[1];
          T[3 * s + 2] = bh.point[2];
          W[s] = boundaryWeight;
          PtoPl[s] = 0;
          continue;
        }
      }
      if (!valid) {
        W[s] = 0;
        continue;
      }
      W[s] = 1;
      T[3 * s] = hit!.point[0];
      T[3 * s + 1] = hit!.point[1];
      T[3 * s + 2] = hit!.point[2];
      const n = hit!.normal;
      const nl = n ? Math.hypot(n[0], n[1], n[2]) : 0;
      if (omega > 0 && nl > 1e-12) {
        Nrm[3 * s] = n[0] / nl;
        Nrm[3 * s + 1] = n[1] / nl;
        Nrm[3 * s + 2] = n[2] / nl;
        PtoPl[s] = 1;
      } else PtoPl[s] = 0;
    }
    return { rms: cnt ? Math.sqrt(sum / cnt) : Infinity, maxDev: cnt ? maxDev : Infinity, hits: cnt };
  };

  const history: number[] = [];
  let it = 0;
  const M = new Float64Array(9);
  for (; it < iterations; it++) {
    const ev = evaluate(true);
    history.push(ev.rms);
    if (it > 0 && Math.abs(history[it - 1] - ev.rms) < tol) break;
    // ---- assemble
    vals.fill(0);
    rhs.fill(0);
    for (let g = 0; g < nG; g++) {
      const sup = groups.support[g];
      const m = sup.length;
      loc.fill(0, 0, m * m * 9);
      locR.fill(0, 0, m * 3);
      let any = false;
      for (let k = groups.start[g]; k < groups.start[g + 1]; k++) {
        const s = groups.order[k];
        const w = W[s];
        if (w === 0) continue;
        any = true;
        if (PtoPl[s]) {
          const nx = Nrm[3 * s], ny = Nrm[3 * s + 1], nz = Nrm[3 * s + 2];
          const a = w * (1 - omega), b = w * omega;
          M[0] = a + b * nx * nx; M[1] = b * nx * ny; M[2] = b * nx * nz;
          M[3] = M[1]; M[4] = a + b * ny * ny; M[5] = b * ny * nz;
          M[6] = M[2]; M[7] = M[5]; M[8] = a + b * nz * nz;
        } else {
          M.fill(0);
          M[0] = M[4] = M[8] = w;
        }
        const tx = T[3 * s], ty = T[3 * s + 1], tz = T[3 * s + 2];
        const mt0 = M[0] * tx + M[1] * ty + M[2] * tz;
        const mt1 = M[3] * tx + M[4] * ty + M[5] * tz;
        const mt2 = M[6] * tx + M[7] * ty + M[8] * tz;
        const p0 = S.rowPtr[s], p1 = S.rowPtr[s + 1];
        for (let p = p0; p < p1; p++) {
          const lj = groups.localCol[p];
          const aj = S.values[p];
          locR[3 * lj] += aj * mt0;
          locR[3 * lj + 1] += aj * mt1;
          locR[3 * lj + 2] += aj * mt2;
          for (let q = p0; q < p1; q++) {
            const c = aj * S.values[q];
            const o = (lj * m + groups.localCol[q]) * 9;
            loc[o] += c * M[0]; loc[o + 1] += c * M[1]; loc[o + 2] += c * M[2];
            loc[o + 3] += c * M[3]; loc[o + 4] += c * M[4]; loc[o + 5] += c * M[5];
            loc[o + 6] += c * M[6]; loc[o + 7] += c * M[7]; loc[o + 8] += c * M[8];
          }
        }
      }
      if (!any) continue;
      const bp = blockPos[g];
      for (let a = 0; a < m; a++) {
        const ga = 3 * sup[a];
        rhs[ga] += locR[3 * a];
        rhs[ga + 1] += locR[3 * a + 1];
        rhs[ga + 2] += locR[3 * a + 2];
        for (let b = 0; b < m; b++) {
          const o = (a * m + b) * 9;
          const d = 9 * bp[a * m + b];
          for (let r = 0; r < 9; r++) vals[d + r] += loc[o + r];
        }
      }
    }
    if (lambda > 0)
      for (let p = 0; p < ltlPos.length; p++) {
        const d = 9 * ltlPos[p];
        const v = lambda * LtL.values[p];
        vals[d] += v;
        vals[d + 4] += v;
        vals[d + 8] += v;
      }
    for (let i = 0; i < N; i++) {
      const d = 9 * A.diagPos[i];
      vals[d] += mu;
      vals[d + 4] += mu;
      vals[d + 8] += mu;
      rhs[3 * i] += mu * P[3 * i];
      rhs[3 * i + 1] += mu * P[3 * i + 1];
      rhs[3 * i + 2] += mu * P[3 * i + 2];
    }
    // Dirichlet constraints for fixed vertices
    for (let i = 0; i < N; i++) {
      for (let p = A.rowPtr[i]; p < A.rowPtr[i + 1]; p++) {
        const j = A.colIdx[p];
        const d = 9 * p;
        if (fixed[i]) {
          vals.fill(0, d, d + 9);
          if (j === i) vals[d] = vals[d + 4] = vals[d + 8] = 1;
        } else if (fixed[j]) {
          const x = P[3 * j], y = P[3 * j + 1], z = P[3 * j + 2];
          rhs[3 * i] -= vals[d] * x + vals[d + 1] * y + vals[d + 2] * z;
          rhs[3 * i + 1] -= vals[d + 3] * x + vals[d + 4] * y + vals[d + 5] * z;
          rhs[3 * i + 2] -= vals[d + 6] * x + vals[d + 7] * y + vals[d + 8] * z;
          vals.fill(0, d, d + 9);
        }
      }
      if (fixed[i]) {
        rhs[3 * i] = P[3 * i];
        rhs[3 * i + 1] = P[3 * i + 1];
        rhs[3 * i + 2] = P[3 * i + 2];
      }
    }
    // ---- solve (warm start from the current cage)
    const x = P.slice();
    pcgBsr3(A, rhs, x, opts.cgTolerance ?? 1e-9, Math.max(200, 4 * N));
    let finite = true;
    for (let i = 0; i < x.length; i++) if (!Number.isFinite(x[i])) finite = false;
    if (!finite) break;
    P.set(x);
    opts.onProgress?.((it + 1) / Math.max(1, iterations), `Fitting cage: iteration ${it + 1}, rms ${ev.rms.toPrecision(4)}`);
  }
  const fin = evaluate(false);
  history.push(fin.rms);
  const cage = cloneQuadMesh(qm);
  for (let i = 0; i < N; i++) cage.vertices[i] = [P[3 * i], P[3 * i + 1], P[3 * i + 2]];
  opts.onProgress?.(1, `Fitting done: rms ${fin.rms.toPrecision(4)}, max ${fin.maxDev.toPrecision(4)}`);
  return { cage, rms: fin.rms, maxDev: fin.maxDev, iterations: it, history, samples: ns, hits: fin.hits };
}

export interface SnapOptions {
  /** fixed-point iterations (default 20) */
  iterations?: number;
  /** vertices that must not move (overrides qm.fixed) */
  fixed?: boolean[];
  /**
   * Optional ray caster: the limit point is projected along its limit normal (both directions,
   * nearest hit wins) instead of to the closest point. Falls back to `target` when there is no hit.
   */
  raycast?: RaycastFn;
  maxDistance?: number;
  /** stop when the largest vertex move is below tolerance * cage size (default 1e-9) */
  tolerance?: number;
}

/**
 * Move cage vertices so that their LIMIT positions lie on the target:
 * repeat  p_i += target(limit_i(P)) - limit_i(P). Returns a new cage.
 */
export function snapVerticesToTarget(qm: QuadMesh, target: ClosestPointQuery, opts: SnapOptions = {}): QuadMesh {
  const iterations = Math.max(1, Math.round(opts.iterations ?? 20));
  const N = qm.vertices.length;
  const fixed = resolveFixed(opts.fixed ?? qm.fixed, N);
  const Lm = limitStencilMatrix(qm);
  const cage = cloneQuadMesh(qm);
  const tol = (opts.tolerance ?? 1e-9) * Math.max(cageSize(qm), 1e-12);
  const used = new Uint8Array(N);
  for (const f of qm.faces) for (const v of f) if (v >= 0 && v < N) used[v] = 1;
  for (let it = 0; it < iterations; it++) {
    const LP = csrMulVec3(Lm, cage.vertices);
    const normals = opts.raycast ? limitNormals(cage) : null;
    let maxMove = 0;
    const moves: (Vec3 | null)[] = new Array(N).fill(null);
    for (let i = 0; i < N; i++) {
      if (fixed[i] || !used[i]) continue;
      const q = LP[i];
      let hit: ClosestPointHit | null = null;
      if (opts.raycast && normals) {
        const n = normals[i];
        const h1 = opts.raycast(q, n);
        const h2 = opts.raycast(q, [-n[0], -n[1], -n[2]]);
        const d1 = h1 ? Math.hypot(h1.point[0] - q[0], h1.point[1] - q[1], h1.point[2] - q[2]) : Infinity;
        const d2 = h2 ? Math.hypot(h2.point[0] - q[0], h2.point[1] - q[1], h2.point[2] - q[2]) : Infinity;
        hit = d1 <= d2 ? h1 : h2;
        if (hit && opts.maxDistance !== undefined && Math.min(d1, d2) > opts.maxDistance) hit = null;
      }
      if (!hit) hit = target(q, opts.maxDistance);
      if (!hit) continue;
      const d: Vec3 = [hit.point[0] - q[0], hit.point[1] - q[1], hit.point[2] - q[2]];
      moves[i] = d;
      maxMove = Math.max(maxMove, Math.hypot(d[0], d[1], d[2]));
    }
    for (let i = 0; i < N; i++) {
      const d = moves[i];
      if (!d) continue;
      const p = cage.vertices[i];
      cage.vertices[i] = [p[0] + d[0], p[1] + d[1], p[2] + d[2]];
    }
    if (maxMove < tol) break;
  }
  return cage;
}
