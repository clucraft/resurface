/**
 * Internal helpers shared by the fitting modules: deterministic sampling, scale normalisation,
 * robust statistics, small dense eigen solvers and a manifold Levenberg–Marquardt driver.
 *
 * Everything here is allocation-light in the per-point paths (packed Float64Arrays, no per-point
 * objects) so it scales to multi-million point selections.
 */
import type { Vec3 } from "../core/types";
import { choleskySolve, solveLinear } from "../core/linalg";
import { anyPerpendicular, cross, normalize } from "../core/vec";

/** Packed xyz coordinates (Float32Array, Float64Array, number[] … possibly SharedArrayBuffer-backed). */
export type PointArray = ArrayLike<number>;

/** Default cap on the number of points used by iterative refinement. */
export const DEFAULT_MAX_POINTS = 200_000;

// ---------------------------------------------------------------------------
// Random numbers / sampling
// ---------------------------------------------------------------------------

/** Deterministic PRNG (mulberry32). Returns floats in [0, 1). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal deviates from a uniform generator (Box–Muller). */
export function gaussianFrom(rng: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const s = spare;
      spare = null;
      return s;
    }
    let u = 0;
    while (u <= 1e-300) u = rng();
    const v = rng();
    const r = Math.sqrt(-2 * Math.log(u));
    spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  };
}

/** Number of points in a selection. */
export function selectionCount(pos: PointArray, indices?: ArrayLike<number>): number {
  return indices ? indices.length : Math.floor(pos.length / 3);
}

/**
 * Deterministically choose up to `max` slots (0..count-1) of a selection: stratified jittered
 * sampling (one random slot per stride bucket, fixed seed) — uniform, sorted, alias-free.
 */
export function sampleSlots(count: number, max: number, seed = 1): Uint32Array {
  if (count <= max) {
    const r = new Uint32Array(count);
    for (let i = 0; i < count; i++) r[i] = i;
    return r;
  }
  const rnd = mulberry32(seed);
  const r = new Uint32Array(max);
  const step = count / max;
  for (let k = 0; k < max; k++) {
    let s = Math.floor((k + rnd()) * step);
    if (s >= count) s = count - 1;
    r[k] = s;
  }
  return r;
}

/**
 * A (sub)sample of a selection in normalised coordinates: x_n = (x - center) / scale, where
 * center is the sample centroid and scale the RMS distance to it. Normals (if given) are unit.
 */
export interface PointSample {
  /** packed normalised xyz, length 3m */
  xs: Float64Array;
  /** packed unit normals, length 3m (when available) */
  ns?: Float64Array;
  /** original vertex index of each sample point */
  vid: Uint32Array;
  /** selection slot (position within `indices`, or vertex index when no indices) of each sample point */
  slot: Uint32Array;
  m: number;
  center: Vec3;
  scale: number;
}

export interface SampleOptions {
  indices?: ArrayLike<number>;
  normals?: ArrayLike<number> | null;
  maxPoints?: number;
  seed?: number;
}

/** Build a normalised sample of a selection. Non-finite points are skipped. Throws on degenerate input. */
export function buildSample(pos: PointArray, opts: SampleOptions = {}, minPoints = 1, what = "fit"): PointSample {
  const count = selectionCount(pos, opts.indices);
  if (count < minPoints) {
    throw new Error(`${what}: needs at least ${minPoints} points, got ${count}`);
  }
  const slots = sampleSlots(count, Math.max(1, opts.maxPoints ?? DEFAULT_MAX_POINTS), opts.seed ?? 1);
  const ind = opts.indices;
  const nrm = opts.normals ?? undefined;
  const raw = new Float64Array(slots.length * 3);
  const vid = new Uint32Array(slots.length);
  const slot = new Uint32Array(slots.length);
  let m = 0;
  let cx = 0, cy = 0, cz = 0;
  for (let k = 0; k < slots.length; k++) {
    const s = slots[k];
    const i = ind ? ind[s] : s;
    const x = pos[3 * i], y = pos[3 * i + 1], z = pos[3 * i + 2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    raw[3 * m] = x;
    raw[3 * m + 1] = y;
    raw[3 * m + 2] = z;
    vid[m] = i;
    slot[m] = s;
    cx += x;
    cy += y;
    cz += z;
    m++;
  }
  if (m < minPoints) throw new Error(`${what}: needs at least ${minPoints} finite points, got ${m}`);
  cx /= m;
  cy /= m;
  cz /= m;
  let ss = 0;
  for (let k = 0; k < m; k++) {
    const dx = raw[3 * k] - cx, dy = raw[3 * k + 1] - cy, dz = raw[3 * k + 2] - cz;
    ss += dx * dx + dy * dy + dz * dz;
  }
  const scale = Math.sqrt(ss / m);
  if (!(scale > 1e-300) || !Number.isFinite(scale)) {
    throw new Error(`${what}: all points coincide (zero spatial extent)`);
  }
  const inv = 1 / scale;
  const xs = new Float64Array(3 * m);
  for (let k = 0; k < m; k++) {
    xs[3 * k] = (raw[3 * k] - cx) * inv;
    xs[3 * k + 1] = (raw[3 * k + 1] - cy) * inv;
    xs[3 * k + 2] = (raw[3 * k + 2] - cz) * inv;
  }
  let ns: Float64Array | undefined;
  if (nrm) {
    ns = new Float64Array(3 * m);
    for (let k = 0; k < m; k++) {
      const i = vid[k];
      const nx = nrm[3 * i], ny = nrm[3 * i + 1], nz = nrm[3 * i + 2];
      const l = Math.hypot(nx, ny, nz);
      if (l > 1e-20 && Number.isFinite(l)) {
        ns[3 * k] = nx / l;
        ns[3 * k + 1] = ny / l;
        ns[3 * k + 2] = nz / l;
      }
    }
  }
  return {
    xs,
    ns,
    vid: m === vid.length ? vid : vid.slice(0, m),
    slot: m === slot.length ? slot : slot.slice(0, m),
    m,
    center: [cx, cy, cz],
    scale,
  };
}

/** Deterministic stride sub-sample of a sample (same normalisation). */
export function subsample(S: PointSample, max: number, seed = 7): PointSample {
  if (S.m <= max) return S;
  const pick = sampleSlots(S.m, max, seed);
  const m = pick.length;
  const xs = new Float64Array(3 * m);
  const ns = S.ns ? new Float64Array(3 * m) : undefined;
  const vid = new Uint32Array(m);
  const slot = new Uint32Array(m);
  for (let k = 0; k < m; k++) {
    const j = pick[k];
    xs[3 * k] = S.xs[3 * j];
    xs[3 * k + 1] = S.xs[3 * j + 1];
    xs[3 * k + 2] = S.xs[3 * j + 2];
    if (ns && S.ns) {
      ns[3 * k] = S.ns[3 * j];
      ns[3 * k + 1] = S.ns[3 * j + 1];
      ns[3 * k + 2] = S.ns[3 * j + 2];
    }
    vid[k] = S.vid[j];
    slot[k] = S.slot[j];
  }
  return { xs, ns, vid, slot, m, center: S.center, scale: S.scale };
}

/** Centroid of a selection (all points, double precision, non-finite skipped). */
export function selectionCentroid(pos: PointArray, indices?: ArrayLike<number>): Vec3 {
  const count = selectionCount(pos, indices);
  let cx = 0, cy = 0, cz = 0, n = 0;
  for (let k = 0; k < count; k++) {
    const i = indices ? indices[k] : k;
    const x = pos[3 * i], y = pos[3 * i + 1], z = pos[3 * i + 2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    cx += x;
    cy += y;
    cz += z;
    n++;
  }
  if (n === 0) return [NaN, NaN, NaN];
  return [cx / n, cy / n, cz / n];
}

// ---------------------------------------------------------------------------
// Robust statistics
// ---------------------------------------------------------------------------

/** In-place quickselect: returns the k-th smallest value (0-based) and partially reorders `a[0..n)`. */
export function kthSmallest(a: Float64Array, k: number, n = a.length): number {
  let lo = 0, hi = n - 1;
  while (hi > lo) {
    // median of three pivot
    const mid = (lo + hi) >> 1;
    let x = a[mid];
    const lv = a[lo], hv = a[hi];
    if ((lv <= x && x <= hv) || (hv <= x && x <= lv)) {
      /* x is median */
    } else if ((x <= lv && lv <= hv) || (hv <= lv && lv <= x)) x = lv;
    else x = hv;
    let i = lo, j = hi;
    while (i <= j) {
      while (a[i] < x) i++;
      while (a[j] > x) j--;
      if (i <= j) {
        const t = a[i];
        a[i] = a[j];
        a[j] = t;
        i++;
        j--;
      }
    }
    if (k <= j) hi = j;
    else if (k >= i) lo = i;
    else return a[k];
  }
  return a[k];
}

/** Quantile q in [0,1] of the first n values (copy, does not modify input). */
export function quantile(values: ArrayLike<number>, q: number, n = values.length): number {
  if (n === 0) return NaN;
  const tmp = new Float64Array(n);
  for (let i = 0; i < n; i++) tmp[i] = values[i];
  const k = Math.min(n - 1, Math.max(0, Math.round(q * (n - 1))));
  return kthSmallest(tmp, k, n);
}

/** Median of |values| (copy). */
export function medianAbs(values: ArrayLike<number>, n = values.length): number {
  if (n === 0) return NaN;
  const tmp = new Float64Array(n);
  for (let i = 0; i < n; i++) tmp[i] = Math.abs(values[i]);
  return kthSmallest(tmp, (n - 1) >> 1, n);
}

/** Tukey biweight tuning constant (95 % efficiency at the normal distribution). */
export const TUKEY_C = 4.685;

/**
 * Tukey biweight weights for residuals r (assumed centred near 0). Scale is estimated as
 * 1.4826 * median|r| (floored by `minSigma`). Returns the robust sigma.
 */
export function tukeyWeights(r: ArrayLike<number>, m: number, out: Float64Array, minSigma = 1e-12): number {
  const sigma = Math.max(1.4826 * medianAbs(r, m), minSigma);
  const c = TUKEY_C * sigma;
  for (let i = 0; i < m; i++) {
    const u = r[i] / c;
    out[i] = Math.abs(u) < 1 ? (1 - u * u) * (1 - u * u) : 0;
  }
  return sigma;
}

// ---------------------------------------------------------------------------
// Small geometry helpers
// ---------------------------------------------------------------------------

/** Deterministic orthonormal pair (e1, e2) perpendicular to unit d, with e1 x e2 = d. */
export function perpBasis(d: Vec3): [Vec3, Vec3] {
  const e1 = anyPerpendicular(d);
  const e2 = normalize(cross(d, e1));
  return [e1, e2];
}

/** n roughly uniform unit directions on the upper hemisphere (z >= 0), Fibonacci lattice. */
export function fibonacciHemisphere(n: number): Vec3[] {
  const out: Vec3[] = [];
  const ga = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    const z = 1 - (i + 0.5) / n; // (0, 1]
    const r = Math.sqrt(Math.max(0, 1 - z * z));
    const phi = i * ga;
    out.push([r * Math.cos(phi), r * Math.sin(phi), z]);
  }
  return out;
}

/** Flip v so that its largest-magnitude component is positive (deterministic sign convention). */
export function canonicalSign(v: Vec3): Vec3 {
  let k = 0;
  if (Math.abs(v[1]) > Math.abs(v[k])) k = 1;
  if (Math.abs(v[2]) > Math.abs(v[k])) k = 2;
  return v[k] < 0 ? [-v[0], -v[1], -v[2]] : [v[0], v[1], v[2]];
}

// ---------------------------------------------------------------------------
// Dense symmetric eigen solver (cyclic Jacobi), any small n
// ---------------------------------------------------------------------------

/**
 * Eigen-decomposition of a symmetric n×n matrix (row-major). Returns eigenvalues ascending and
 * the matching unit eigenvectors (vectors[k] is the k-th eigenvector).
 */
export function jacobiEigen(Ain: ArrayLike<number>, n: number): { values: Float64Array; vectors: Float64Array[] } {
  const a = Float64Array.from(Ain as ArrayLike<number>);
  const v = new Float64Array(n * n);
  for (let i = 0; i < n; i++) v[i * n + i] = 1;
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0, diag = 0;
    for (let p = 0; p < n; p++) {
      diag += a[p * n + p] * a[p * n + p];
      for (let q = p + 1; q < n; q++) off += a[p * n + q] * a[p * n + q];
    }
    if (off <= 1e-30 * diag || off < 1e-300) break;
    for (let p = 0; p < n - 1; p++)
      for (let q = p + 1; q < n; q++) {
        const apq = a[p * n + q];
        if (Math.abs(apq) < 1e-300) continue;
        const theta = (a[q * n + q] - a[p * n + p]) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k * n + p], akq = a[k * n + q];
          a[k * n + p] = c * akp - s * akq;
          a[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p * n + k], aqk = a[q * n + k];
          a[p * n + k] = c * apk - s * aqk;
          a[q * n + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k * n + p], vkq = v[k * n + q];
          v[k * n + p] = c * vkp - s * vkq;
          v[k * n + q] = s * vkp + c * vkq;
        }
      }
  }
  const idx = Array.from({ length: n }, (_, i) => i).sort((i, j) => a[i * n + i] - a[j * n + j]);
  const values = new Float64Array(n);
  const vectors: Float64Array[] = [];
  idx.forEach((col, k) => {
    values[k] = a[col * n + col];
    const vec = new Float64Array(n);
    let l = 0;
    for (let r = 0; r < n; r++) {
      vec[r] = v[r * n + col];
      l += vec[r] * vec[r];
    }
    l = Math.sqrt(l) || 1;
    for (let r = 0; r < n; r++) vec[r] /= l;
    vectors.push(vec);
  });
  return { values, vectors };
}

// ---------------------------------------------------------------------------
// Levenberg–Marquardt on a manifold (retraction based)
// ---------------------------------------------------------------------------

/**
 * A nonlinear least-squares problem Σ w_i r_i(s)^2 over a state S that may live on a manifold.
 * `linearize` must accumulate J^T W J (upper triangle is sufficient) and J^T W r at state s,
 * where J is the Jacobian w.r.t. the local increment used by `retract`.
 */
export interface LMProblem<S> {
  nParams: number;
  linearize(s: S, JtJ: Float64Array, Jtr: Float64Array): number;
  cost(s: S): number;
  retract(s: S, delta: Float64Array): S;
}

export interface LMOptions {
  maxIterations?: number;
  /** relative cost decrease below which an accepted step ends the iteration (default 1e-10) */
  tolerance?: number;
  /** parameters held fixed (by index) */
  fixed?: ArrayLike<boolean>;
}

export interface LMResult<S> {
  state: S;
  cost: number;
  iterations: number;
  converged: boolean;
}

export function levenbergMarquardt<S>(prob: LMProblem<S>, s0: S, opts: LMOptions = {}): LMResult<S> {
  const n = prob.nParams;
  const maxIt = opts.maxIterations ?? 100;
  const tol = opts.tolerance ?? 1e-10;
  const fixed = opts.fixed;
  const JtJ = new Float64Array(n * n);
  const Jtr = new Float64Array(n);
  const A = new Float64Array(n * n);
  const b = new Float64Array(n);
  let s = s0;
  let cost = Infinity;
  let lambda = 1e-3;
  let needLin = true;
  let converged = false;
  let it = 0;
  for (; it < maxIt; it++) {
    if (needLin) {
      JtJ.fill(0);
      Jtr.fill(0);
      cost = prob.linearize(s, JtJ, Jtr);
      if (!Number.isFinite(cost)) break;
      for (let a = 0; a < n; a++) for (let c = 0; c < a; c++) JtJ[a * n + c] = JtJ[c * n + a];
      if (fixed) {
        for (let a = 0; a < n; a++) {
          if (!fixed[a]) continue;
          for (let c = 0; c < n; c++) {
            JtJ[a * n + c] = 0;
            JtJ[c * n + a] = 0;
          }
          JtJ[a * n + a] = 1;
          Jtr[a] = 0;
        }
      }
      needLin = false;
      if (cost === 0) {
        converged = true;
        break;
      }
    }
    let maxDiag = 0;
    for (let a = 0; a < n; a++) maxDiag = Math.max(maxDiag, JtJ[a * n + a]);
    if (!(maxDiag > 0)) {
      converged = true;
      break;
    }
    A.set(JtJ);
    for (let a = 0; a < n; a++) A[a * n + a] += lambda * Math.max(JtJ[a * n + a], 1e-9 * maxDiag);
    for (let a = 0; a < n; a++) b[a] = -Jtr[a];
    let delta = choleskySolve(A, b, n);
    if (!delta) delta = solveLinear(A, b, n);
    let finite = !!delta;
    if (delta) for (let a = 0; a < n; a++) if (!Number.isFinite(delta[a])) finite = false;
    if (!delta || !finite) {
      lambda *= 10;
      if (lambda > 1e20) break;
      continue;
    }
    const trial = prob.retract(s, delta);
    const c2 = prob.cost(trial);
    if (Number.isFinite(c2) && c2 <= cost) {
      const rel = (cost - c2) / Math.max(cost, 1e-300);
      s = trial;
      cost = c2;
      lambda = Math.max(lambda / 3, 1e-12);
      needLin = true;
      let dn = 0;
      for (let a = 0; a < n; a++) dn += delta[a] * delta[a];
      if (rel < tol || Math.sqrt(dn) < 1e-13) {
        converged = true;
        it++;
        break;
      }
    } else {
      lambda *= 4;
      if (lambda > 1e16) {
        // no descent possible any more: numerically at a minimum
        converged = true;
        break;
      }
    }
  }
  return { state: s, cost, iterations: it, converged };
}

/** Accumulate w * J J^T (upper triangle) and w * J r into JtJ / Jtr. */
export function accumulateNormal(JtJ: Float64Array, Jtr: Float64Array, J: Float64Array, n: number, w: number, r: number): void {
  for (let a = 0; a < n; a++) {
    const ja = J[a] * w;
    if (ja === 0) continue;
    Jtr[a] += ja * r;
    const row = a * n;
    for (let c = a; c < n; c++) JtJ[row + c] += ja * J[c];
  }
}
