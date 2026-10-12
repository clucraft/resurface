/**
 * NURBS core: basis functions (The NURBS Book, Piegl & Tiller), curve/surface evaluation and
 * derivatives, least-squares curve fitting / interpolation, surface tessellation, closest point
 * on surface and regularised least-squares surface fitting ("Fit Surface").
 *
 * Conventions follow `BSplineCurveDef` / `BSplineSurfaceDef` in core/types.ts: full (clamped)
 * knot vectors, surface control points row-major over U (index = i * nv + j).
 */
import type { BSplineCurveDef, BSplineSurfaceDef, FitStats, Frame, MeshData, Vec3 } from "../core/types";
import { eigenSym3 } from "../core/linalg";
import { cross, dot, length, normalize } from "../core/vec";
import { buildSample, canonicalSign, selectionCount } from "./common";
import type { PointArray } from "./common";
import { BandGeneral, BandSPD } from "./sparse";

// ===========================================================================
// Basis functions
// ===========================================================================

/**
 * Knot span index (A2.1). `n` is the index of the LAST control point (nCtrl - 1), `p` the degree.
 * Returns i with U[i] <= u < U[i+1] (u at the end of the domain maps to the last non-empty span).
 */
export function findSpan(n: number, p: number, u: number, U: ArrayLike<number>): number {
  if (u >= U[n + 1]) return n;
  if (u <= U[p]) return p;
  let low = p, high = n + 1;
  let mid = (low + high) >> 1;
  while (u < U[mid] || u >= U[mid + 1]) {
    if (u < U[mid]) high = mid;
    else low = mid;
    mid = (low + high) >> 1;
  }
  return mid;
}

/** Non-vanishing basis functions N_{i-p..i, p}(u) (A2.2). */
export function basisFuns(i: number, u: number, p: number, U: ArrayLike<number>, out = new Float64Array(p + 1)): Float64Array {
  const left = new Float64Array(p + 1), right = new Float64Array(p + 1);
  basisFunsInto(i, u, p, U, out, left, right);
  return out;
}

function basisFunsInto(i: number, u: number, p: number, U: ArrayLike<number>, N: Float64Array, left: Float64Array, right: Float64Array): void {
  N[0] = 1;
  for (let j = 1; j <= p; j++) {
    left[j] = u - U[i + 1 - j];
    right[j] = U[i + j] - u;
    let saved = 0;
    for (let r = 0; r < j; r++) {
      const den = right[r + 1] + left[j - r];
      const temp = den !== 0 ? N[r] / den : 0;
      N[r] = saved + right[r + 1] * temp;
      saved = left[j - r] * temp;
    }
    N[j] = saved;
  }
}

/**
 * Basis functions and derivatives up to order n (A2.3): ders[k][j] = k-th derivative of
 * N_{i-p+j, p} at u. Orders above p are zero.
 */
export function dersBasisFuns(i: number, u: number, p: number, n: number, U: ArrayLike<number>): Float64Array[] {
  const ws = new DersWorkspace(p, n);
  ws.compute(i, u, U);
  const out: Float64Array[] = [];
  for (let k = 0; k <= n; k++) out.push(ws.ders.slice(k * (p + 1), (k + 1) * (p + 1)));
  return out;
}

/** Allocation-free workspace for dersBasisFuns (flat ders[k*(p+1) + j]). */
class DersWorkspace {
  readonly p: number;
  readonly n: number;
  readonly ders: Float64Array;
  private ndu: Float64Array;
  private a: Float64Array;
  private left: Float64Array;
  private right: Float64Array;
  constructor(p: number, n: number) {
    this.p = p;
    this.n = n;
    this.ders = new Float64Array((n + 1) * (p + 1));
    this.ndu = new Float64Array((p + 1) * (p + 1));
    this.a = new Float64Array(2 * (p + 1));
    this.left = new Float64Array(p + 1);
    this.right = new Float64Array(p + 1);
  }
  compute(i: number, u: number, U: ArrayLike<number>): void {
    const { p, ders, ndu, a, left, right } = this;
    const w = p + 1;
    const nn = Math.min(this.n, p);
    ders.fill(0);
    ndu[0] = 1;
    for (let j = 1; j <= p; j++) {
      left[j] = u - U[i + 1 - j];
      right[j] = U[i + j] - u;
      let saved = 0;
      for (let r = 0; r < j; r++) {
        ndu[j * w + r] = right[r + 1] + left[j - r];
        const den = ndu[j * w + r];
        const temp = den !== 0 ? ndu[r * w + j - 1] / den : 0;
        ndu[r * w + j] = saved + right[r + 1] * temp;
        saved = left[j - r] * temp;
      }
      ndu[j * w + j] = saved;
    }
    for (let j = 0; j <= p; j++) ders[j] = ndu[j * w + p];
    for (let r = 0; r <= p; r++) {
      let s1 = 0, s2 = 1;
      a[0] = 1;
      for (let k = 1; k <= nn; k++) {
        let d = 0;
        const rk = r - k, pk = p - k;
        if (r >= k) {
          const den = ndu[(pk + 1) * w + rk];
          a[s2 * w] = den !== 0 ? a[s1 * w] / den : 0;
          d = a[s2 * w] * ndu[rk * w + pk];
        }
        const j1 = rk >= -1 ? 1 : -rk;
        const j2 = r - 1 <= pk ? k - 1 : p - r;
        for (let j = j1; j <= j2; j++) {
          const den = ndu[(pk + 1) * w + rk + j];
          a[s2 * w + j] = den !== 0 ? (a[s1 * w + j] - a[s1 * w + j - 1]) / den : 0;
          d += a[s2 * w + j] * ndu[(rk + j) * w + pk];
        }
        if (r <= pk) {
          const den = ndu[(pk + 1) * w + r];
          a[s2 * w + k] = den !== 0 ? -a[s1 * w + k - 1] / den : 0;
          d += a[s2 * w + k] * ndu[r * w + pk];
        }
        ders[k * w + r] = d;
        const t = s1;
        s1 = s2;
        s2 = t;
      }
    }
    let f = p;
    for (let k = 1; k <= nn; k++) {
      for (let j = 0; j <= p; j++) ders[k * w + j] *= f;
      f *= p - k;
    }
  }
}

/** Clamped knot vector with uniformly spaced interior knots on [0, 1]. */
export function clampedUniformKnots(nCtrl: number, degree: number): number[] {
  if (!(degree >= 1)) throw new Error("clampedUniformKnots: degree must be >= 1");
  if (nCtrl < degree + 1) throw new Error(`clampedUniformKnots: need at least degree+1 = ${degree + 1} control points, got ${nCtrl}`);
  const U: number[] = [];
  for (let i = 0; i <= degree; i++) U.push(0);
  const nInt = nCtrl - degree - 1;
  for (let j = 1; j <= nInt; j++) U.push(j / (nInt + 1));
  for (let i = 0; i <= degree; i++) U.push(1);
  return U;
}

/**
 * Clamped knot vector adapted to sorted parameters in [params[0], params[last]]:
 * de Boor averaging when nCtrl == params.length (interpolation, eq. 9.8), otherwise the
 * least-squares placement of eq. 9.68–9.69 (every knot span contains parameters).
 */
export function averagedKnots(params: ArrayLike<number>, nCtrl: number, degree: number): number[] {
  const m = params.length;
  const p = degree;
  const n = nCtrl - 1;
  if (nCtrl < p + 1) throw new Error("averagedKnots: need at least degree+1 control points");
  if (m < nCtrl) throw new Error(`averagedKnots: need at least nCtrl (${nCtrl}) parameters, got ${m}`);
  const a = params[0], b = params[m - 1];
  const U: number[] = new Array(nCtrl + p + 1);
  for (let i = 0; i <= p; i++) {
    U[i] = a;
    U[n + 1 + i] = b;
  }
  if (m === nCtrl) {
    for (let j = 1; j <= n - p; j++) {
      let s = 0;
      for (let i = j; i <= j + p - 1; i++) s += params[i];
      U[j + p] = s / p;
    }
  } else {
    const d = m / (n - p + 1);
    for (let j = 1; j <= n - p; j++) {
      const i = Math.floor(j * d);
      const alpha = j * d - i;
      U[p + j] = (1 - alpha) * params[i - 1] + alpha * params[i];
    }
  }
  return U;
}

// ===========================================================================
// Gauss–Legendre quadrature
// ===========================================================================

const glCache = new Map<number, { x: Float64Array; w: Float64Array }>();

/** n-point Gauss–Legendre nodes/weights on [-1, 1]. */
export function gaussLegendre(n: number): { x: Float64Array; w: Float64Array } {
  const c = glCache.get(n);
  if (c) return c;
  const x = new Float64Array(n), w = new Float64Array(n);
  for (let i = 0; i < Math.ceil(n / 2); i++) {
    let z = Math.cos((Math.PI * (i + 0.75)) / (n + 0.5));
    let dp = 1;
    for (let it = 0; it < 100; it++) {
      let p0 = 1, p1 = z;
      for (let k = 2; k <= n; k++) {
        const p2 = ((2 * k - 1) * z * p1 - (k - 1) * p0) / k;
        p0 = p1;
        p1 = p2;
      }
      if (n === 1) {
        p1 = z;
        p0 = 1;
      }
      dp = (n * (z * p1 - p0)) / (z * z - 1);
      const dz = p1 / dp;
      z -= dz;
      if (Math.abs(dz) < 1e-15) break;
    }
    x[i] = -z;
    x[n - 1 - i] = z;
    w[i] = w[n - 1 - i] = 2 / ((1 - z * z) * dp * dp);
  }
  if (n === 1) {
    x[0] = 0;
    w[0] = 2;
  }
  const r = { x, w };
  glCache.set(n, r);
  return r;
}

/**
 * Gram matrix G[i][j] = ∫ N_i^(r)(t) N_j^(r)(t) dt over the knot domain, as a dense row-major
 * nCtrl×nCtrl Float64Array (only |i-j| <= p non-zero).
 */
export function basisGram(U: ArrayLike<number>, p: number, nCtrl: number, order: number): Float64Array {
  const G = new Float64Array(nCtrl * nCtrl);
  if (order > p) return G;
  const gl = gaussLegendre(p + 1);
  const ws = new DersWorkspace(p, order);
  const w = p + 1;
  for (let span = p; span < nCtrl; span++) {
    const a = U[span], b = U[span + 1];
    if (!(b > a)) continue;
    const half = (b - a) / 2, mid = (a + b) / 2;
    for (let g = 0; g < gl.x.length; g++) {
      const t = mid + half * gl.x[g];
      ws.compute(span, t, U);
      const wt = gl.w[g] * half;
      for (let r = 0; r <= p; r++) {
        const vr = ws.ders[order * w + r];
        if (vr === 0) continue;
        const I = span - p + r;
        for (let c = 0; c <= p; c++) G[I * nCtrl + span - p + c] += wt * vr * ws.ders[order * w + c];
      }
    }
  }
  return G;
}

// ===========================================================================
// Curves
// ===========================================================================

function curveDomain(c: BSplineCurveDef): [number, number] {
  return [c.knots[c.degree], c.knots[c.controlPoints.length]];
}

function checkCurve(c: BSplineCurveDef): void {
  const n = c.controlPoints.length;
  if (c.knots.length !== n + c.degree + 1) throw new Error(`BSpline curve: knots.length must be nCtrl + degree + 1 (${n + c.degree + 1}), got ${c.knots.length}`);
  if (c.weights && c.weights.length !== n) throw new Error("BSpline curve: weights length mismatch");
}

/** Point on a (possibly rational) B-spline curve. */
export function evalCurve(c: BSplineCurveDef, t: number): Vec3 {
  return curveDerivs(c, t, 0)[0];
}

/** Point and derivatives up to order k: [C, C', C'', ...] (rational curves supported, A4.2). */
export function curveDerivs(c: BSplineCurveDef, t: number, k: number): Vec3[] {
  checkCurve(c);
  const p = c.degree;
  const n = c.controlPoints.length - 1;
  const [a, b] = curveDomain(c);
  const u = Math.min(b, Math.max(a, t));
  const span = findSpan(n, p, u, c.knots);
  const ws = new DersWorkspace(p, k);
  ws.compute(span, u, c.knots);
  const w = p + 1;
  const out: Vec3[] = [];
  if (!c.weights) {
    for (let d = 0; d <= k; d++) {
      let x = 0, y = 0, z = 0;
      if (d <= p)
        for (let j = 0; j <= p; j++) {
          const P = c.controlPoints[span - p + j];
          const N = ws.ders[d * w + j];
          x += N * P[0];
          y += N * P[1];
          z += N * P[2];
        }
      out.push([x, y, z]);
    }
    return out;
  }
  const A: Vec3[] = [], W: number[] = [];
  for (let d = 0; d <= k; d++) {
    let x = 0, y = 0, z = 0, ww = 0;
    if (d <= p)
      for (let j = 0; j <= p; j++) {
        const P = c.controlPoints[span - p + j];
        const wj = c.weights[span - p + j];
        const N = ws.ders[d * w + j] * wj;
        x += N * P[0];
        y += N * P[1];
        z += N * P[2];
        ww += N;
      }
    A.push([x, y, z]);
    W.push(ww);
  }
  for (let d = 0; d <= k; d++) {
    let v: Vec3 = [A[d][0], A[d][1], A[d][2]];
    for (let i = 1; i <= d; i++) {
      const bin = binomial(d, i) * W[i];
      v = [v[0] - bin * out[d - i][0], v[1] - bin * out[d - i][1], v[2] - bin * out[d - i][2]];
    }
    out.push([v[0] / W[0], v[1] / W[0], v[2] / W[0]]);
  }
  return out;
}

function binomial(n: number, k: number): number {
  let r = 1;
  for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i;
  return r;
}

export type ParamMethod = "chord" | "centripetal" | "uniform";

/** Normalised [0,1] parameters for ordered points. */
export function parametrize(points: Vec3[], method: ParamMethod = "centripetal"): Float64Array {
  const n = points.length;
  const t = new Float64Array(n);
  if (n === 1) return t;
  if (method === "uniform") {
    for (let i = 0; i < n; i++) t[i] = i / (n - 1);
    return t;
  }
  for (let i = 1; i < n; i++) {
    const d = Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1], points[i][2] - points[i - 1][2]);
    t[i] = t[i - 1] + (method === "centripetal" ? Math.sqrt(d) : d);
  }
  const L = t[n - 1];
  if (!(L > 0)) throw new Error("parametrize: all points coincide");
  for (let i = 1; i < n; i++) t[i] /= L;
  t[n - 1] = 1;
  return t;
}

export interface CurveFitOptions {
  /** number of control points (>= degree + 1) */
  nCtrl: number;
  /** default 3 (reduced automatically when nCtrl is small) */
  degree?: number;
  /** weight of ∫|C''|² dt relative to the mean squared error (dimensionless, default 0) */
  smoothing?: number;
  /** closed curves are not supported yet */
  closed?: false;
  /** parametrisation (default 'centripetal') */
  params?: ParamMethod;
  /** parameter-correction passes (default 0) */
  iterations?: number;
}

export interface CurveFitResult {
  curve: BSplineCurveDef;
  /** parameter of each input point */
  params: Float64Array;
  /** distances |C(t_k) - Q_k| (unsigned) */
  stats: FitStats;
}

/**
 * Least-squares B-spline curve approximation with end-point interpolation (P0 = Q0, Pn = Qlast)
 * and optional bending-energy smoothing. Knots from the parameter distribution (NURBS Book 9.69).
 */
export function fitBSplineCurve(points: Vec3[], opts: CurveFitOptions): CurveFitResult {
  if ((opts as { closed?: boolean }).closed) throw new Error("fitBSplineCurve: closed curves are not supported");
  const N = points.length;
  if (N < 2) throw new Error("fitBSplineCurve: needs at least 2 points");
  const lam = Math.max(0, opts.smoothing ?? 0);
  let nCtrl = Math.floor(opts.nCtrl);
  if (!(nCtrl >= 2)) throw new Error("fitBSplineCurve: nCtrl must be >= 2");
  if (nCtrl > N) {
    if (lam === 0) throw new Error(`fitBSplineCurve: nCtrl (${nCtrl}) exceeds the number of points (${N}); use smoothing or fewer control points`);
  }
  const p = Math.max(1, Math.min(opts.degree ?? 3, nCtrl - 1));
  const t = parametrize(points, opts.params ?? "centripetal");
  const U = nCtrl <= N ? averagedKnots(t, nCtrl, p) : clampedUniformKnots(nCtrl, p);
  let ctrl = solveCurveLS(points, t, U, p, nCtrl, lam);
  const iters = opts.iterations ?? 0;
  for (let it = 0; it < iters; it++) {
    const c: BSplineCurveDef = { degree: p, knots: U, controlPoints: ctrl };
    for (let k = 1; k < N - 1; k++) t[k] = projectOnCurveNewton(c, points[k], t[k]);
    ctrl = solveCurveLS(points, t, U, p, nCtrl, lam);
  }
  const curve: BSplineCurveDef = { degree: p, knots: U, controlPoints: ctrl };
  let ss = 0, mx = 0, s = 0;
  for (let k = 0; k < N; k++) {
    const q = evalCurve(curve, t[k]);
    const d = Math.hypot(q[0] - points[k][0], q[1] - points[k][1], q[2] - points[k][2]);
    ss += d * d;
    s += d;
    mx = Math.max(mx, d);
  }
  return { curve, params: t, stats: { rms: Math.sqrt(ss / N), maxAbs: mx, mean: s / N, count: N } };
}

function solveCurveLS(points: Vec3[], t: Float64Array, U: number[], p: number, nCtrl: number, lam: number): Vec3[] {
  const N = points.length;
  const n = nCtrl - 1;
  const Q0 = points[0], Qn = points[N - 1];
  const ctrl: Vec3[] = new Array(nCtrl);
  ctrl[0] = [Q0[0], Q0[1], Q0[2]];
  ctrl[n] = [Qn[0], Qn[1], Qn[2]];
  const nu = nCtrl - 2;
  if (nu <= 0) return ctrl;
  const A = new BandSPD(nu, p);
  const bx = new Float64Array(nu), by = new Float64Array(nu), bz = new Float64Array(nu);
  const Nb = new Float64Array(p + 1), left = new Float64Array(p + 1), right = new Float64Array(p + 1);
  const inv = 1 / N;
  for (let k = 0; k < N; k++) {
    const span = findSpan(n, p, t[k], U);
    basisFunsInto(span, t[k], p, U, Nb, left, right);
    // residual target after removing the fixed end control points
    let rx = points[k][0], ry = points[k][1], rz = points[k][2];
    for (let j = 0; j <= p; j++) {
      const I = span - p + j;
      if (I === 0) {
        rx -= Nb[j] * Q0[0];
        ry -= Nb[j] * Q0[1];
        rz -= Nb[j] * Q0[2];
      } else if (I === n) {
        rx -= Nb[j] * Qn[0];
        ry -= Nb[j] * Qn[1];
        rz -= Nb[j] * Qn[2];
      }
    }
    for (let a = 0; a <= p; a++) {
      const I = span - p + a;
      if (I === 0 || I === n) continue;
      const va = Nb[a] * inv;
      bx[I - 1] += va * rx;
      by[I - 1] += va * ry;
      bz[I - 1] += va * rz;
      for (let c = 0; c <= a; c++) {
        const J = span - p + c;
        if (J === 0 || J === n) continue;
        A.add(I - 1, J - 1, va * Nb[c]);
      }
    }
  }
  if (lam > 0) {
    const order = p >= 2 ? 2 : 1;
    const G = basisGram(U, p, nCtrl, order);
    for (let i = 1; i < n; i++) {
      for (let j = Math.max(1, i - p); j <= i; j++) A.add(i - 1, j - 1, lam * G[i * nCtrl + j]);
      // fixed end contributions move to the right-hand side
      const g0 = G[i * nCtrl], gn = G[i * nCtrl + n];
      bx[i - 1] -= lam * (g0 * Q0[0] + gn * Qn[0]);
      by[i - 1] -= lam * (g0 * Q0[1] + gn * Qn[1]);
      bz[i - 1] -= lam * (g0 * Q0[2] + gn * Qn[2]);
    }
  }
  // tiny relative ridge keeps control points of data-free spans defined
  A.addDiagonal(1e-12 * Math.max(A.meanDiagonal(), 1e-300));
  if (!A.factor()) throw new Error("fitBSplineCurve: singular system — too many control points for the data (add smoothing)");
  A.solve(bx);
  A.solve(by);
  A.solve(bz);
  for (let i = 1; i < n; i++) ctrl[i] = [bx[i - 1], by[i - 1], bz[i - 1]];
  return ctrl;
}

/** Newton projection of a point onto a curve starting from parameter t0 (clamped to the domain). */
export function projectOnCurveNewton(c: BSplineCurveDef, q: Vec3, t0: number, maxIter = 20): number {
  const [a, b] = curveDomain(c);
  let t = Math.min(b, Math.max(a, t0));
  for (let it = 0; it < maxIter; it++) {
    const [C, C1, C2] = curveDerivs(c, t, 2);
    const r: Vec3 = [C[0] - q[0], C[1] - q[1], C[2] - q[2]];
    const f = dot(C1, r);
    let df = dot(C2, r) + dot(C1, C1);
    if (!(df > 0)) df = dot(C1, C1);
    if (!(df > 0)) break;
    const tn = Math.min(b, Math.max(a, t - f / df));
    if (Math.abs(tn - t) < 1e-14 * (b - a)) {
      t = tn;
      break;
    }
    t = tn;
  }
  return t;
}

/** Closest point on a curve (sampling + Newton). */
export function closestPointOnCurve(c: BSplineCurveDef, q: Vec3, samples = 0): { t: number; point: Vec3; distance: number } {
  checkCurve(c);
  const [a, b] = curveDomain(c);
  const ns = samples > 0 ? samples : Math.max(16, 4 * c.controlPoints.length);
  let bestT = a, bestD = Infinity;
  for (let i = 0; i <= ns; i++) {
    const t = a + ((b - a) * i) / ns;
    const P = evalCurve(c, t);
    const d = Math.hypot(P[0] - q[0], P[1] - q[1], P[2] - q[2]);
    if (d < bestD) {
      bestD = d;
      bestT = t;
    }
  }
  const t = projectOnCurveNewton(c, q, bestT);
  const point = evalCurve(c, t);
  return { t, point, distance: Math.hypot(point[0] - q[0], point[1] - q[1], point[2] - q[2]) };
}

/** Global curve interpolation through all points (A9.1), averaged knots. */
export function interpolateBSplineCurve(points: Vec3[], degree = 3, params: ParamMethod = "centripetal"): BSplineCurveDef {
  const N = points.length;
  if (N < 2) throw new Error("interpolateBSplineCurve: needs at least 2 points");
  const p = Math.max(1, Math.min(degree, N - 1));
  const t = parametrize(points, params);
  for (let i = 1; i < N; i++) if (!(t[i] > t[i - 1])) throw new Error("interpolateBSplineCurve: consecutive duplicate points");
  const U = averagedKnots(t, N, p);
  const n = N - 1;
  const A = new BandGeneral(N, p, p);
  const Nb = new Float64Array(p + 1), left = new Float64Array(p + 1), right = new Float64Array(p + 1);
  for (let k = 0; k < N; k++) {
    const span = findSpan(n, p, t[k], U);
    basisFunsInto(span, t[k], p, U, Nb, left, right);
    for (let j = 0; j <= p; j++) {
      const col = span - p + j;
      if (Math.abs(col - k) <= p) A.set(k, col, Nb[j]);
    }
  }
  const cx = Float64Array.from(points, (q) => q[0]);
  const cy = Float64Array.from(points, (q) => q[1]);
  const cz = Float64Array.from(points, (q) => q[2]);
  A.solve([cx, cy, cz]);
  return { degree: p, knots: U, controlPoints: Array.from({ length: N }, (_, i) => [cx[i], cy[i], cz[i]] as Vec3) };
}

// ===========================================================================
// Surfaces
// ===========================================================================

function checkSurface(s: BSplineSurfaceDef): void {
  if (s.controlPoints.length !== s.nu * s.nv) throw new Error(`BSpline surface: controlPoints.length (${s.controlPoints.length}) != nu*nv (${s.nu * s.nv})`);
  if (s.knotsU.length !== s.nu + s.degreeU + 1) throw new Error("BSpline surface: knotsU.length must be nu + degreeU + 1");
  if (s.knotsV.length !== s.nv + s.degreeV + 1) throw new Error("BSpline surface: knotsV.length must be nv + degreeV + 1");
  if (s.weights && s.weights.length !== s.nu * s.nv) throw new Error("BSpline surface: weights length mismatch");
}

/** Parameter domain [[u0, u1], [v0, v1]]. */
export function surfaceDomain(s: BSplineSurfaceDef): [[number, number], [number, number]] {
  return [
    [s.knotsU[s.degreeU], s.knotsU[s.nu]],
    [s.knotsV[s.degreeV], s.knotsV[s.nv]],
  ];
}

/**
 * Allocation-free surface evaluator (point + derivatives up to 2nd order), rational supported.
 * out layout (18 numbers): S, Su, Sv, Suu, Suv, Svv.
 */
export class SurfaceEvaluator {
  readonly s: BSplineSurfaceDef;
  readonly p: number;
  readonly q: number;
  readonly u0: number;
  readonly u1: number;
  readonly v0: number;
  readonly v1: number;
  private cp: Float64Array; // homogeneous x*w, y*w, z*w, w
  private rational: boolean;
  private wsU: DersWorkspace;
  private wsV: DersWorkspace;
  private h = new Float64Array(24); // homogeneous derivative accumulators: 6 x (x, y, z, w)

  constructor(s: BSplineSurfaceDef) {
    checkSurface(s);
    this.s = s;
    this.p = s.degreeU;
    this.q = s.degreeV;
    [[this.u0, this.u1], [this.v0, this.v1]] = surfaceDomain(s);
    const n = s.nu * s.nv;
    this.cp = new Float64Array(4 * n);
    this.rational = !!s.weights;
    for (let i = 0; i < n; i++) {
      const w = s.weights ? s.weights[i] : 1;
      const P = s.controlPoints[i];
      this.cp[4 * i] = P[0] * w;
      this.cp[4 * i + 1] = P[1] * w;
      this.cp[4 * i + 2] = P[2] * w;
      this.cp[4 * i + 3] = w;
    }
    this.wsU = new DersWorkspace(this.p, 2);
    this.wsV = new DersWorkspace(this.q, 2);
  }

  /** Evaluate S and derivatives up to `order` (0, 1 or 2) at (u, v) (clamped to the domain). */
  eval(u: number, v: number, order: number, out: Float64Array): void {
    const { s, p, q, cp, h } = this;
    u = Math.min(this.u1, Math.max(this.u0, u));
    v = Math.min(this.v1, Math.max(this.v0, v));
    const su = findSpan(s.nu - 1, p, u, s.knotsU);
    const sv = findSpan(s.nv - 1, q, v, s.knotsV);
    this.wsU.compute(su, u, s.knotsU);
    this.wsV.compute(sv, v, s.knotsV);
    const Du = this.wsU.ders, Dv = this.wsV.ders;
    const wu = p + 1, wv = q + 1;
    h.fill(0);
    // derivative index combos: 0:(0,0) 1:(1,0) 2:(0,1) 3:(2,0) 4:(1,1) 5:(0,2)
    const nk = order >= 2 ? 6 : order >= 1 ? 3 : 1;
    for (let a = 0; a <= p; a++) {
      const row = (su - p + a) * s.nv;
      const nu0 = Du[a], nu1 = Du[wu + a], nu2 = Du[2 * wu + a];
      for (let b = 0; b <= q; b++) {
        const idx = 4 * (row + sv - q + b);
        const x = cp[idx], y = cp[idx + 1], z = cp[idx + 2], w = cp[idx + 3];
        const nv0 = Dv[b], nv1 = Dv[wv + b], nv2 = Dv[2 * wv + b];
        const f0 = nu0 * nv0;
        h[0] += f0 * x;
        h[1] += f0 * y;
        h[2] += f0 * z;
        h[3] += f0 * w;
        if (nk > 1) {
          const f1 = nu1 * nv0, f2 = nu0 * nv1;
          h[4] += f1 * x;
          h[5] += f1 * y;
          h[6] += f1 * z;
          h[7] += f1 * w;
          h[8] += f2 * x;
          h[9] += f2 * y;
          h[10] += f2 * z;
          h[11] += f2 * w;
          if (nk > 3) {
            const f3 = nu2 * nv0, f4 = nu1 * nv1, f5 = nu0 * nv2;
            h[12] += f3 * x;
            h[13] += f3 * y;
            h[14] += f3 * z;
            h[15] += f3 * w;
            h[16] += f4 * x;
            h[17] += f4 * y;
            h[18] += f4 * z;
            h[19] += f4 * w;
            h[20] += f5 * x;
            h[21] += f5 * y;
            h[22] += f5 * z;
            h[23] += f5 * w;
          }
        }
      }
    }
    if (!this.rational) {
      for (let k = 0; k < nk; k++) {
        out[3 * k] = h[4 * k];
        out[3 * k + 1] = h[4 * k + 1];
        out[3 * k + 2] = h[4 * k + 2];
      }
      return;
    }
    const w0 = h[3];
    for (let c = 0; c < 3; c++) {
      const S = h[c] / w0;
      out[c] = S;
      if (nk > 1) {
        const wu1 = h[7], wv1 = h[11];
        const Su = (h[4 + c] - wu1 * S) / w0;
        const Sv = (h[8 + c] - wv1 * S) / w0;
        out[3 + c] = Su;
        out[6 + c] = Sv;
        if (nk > 3) {
          const wuu = h[15], wuv = h[19], wvv = h[23];
          out[9 + c] = (h[12 + c] - 2 * wu1 * Su - wuu * S) / w0;
          out[12 + c] = (h[16 + c] - wu1 * Sv - wv1 * Su - wuv * S) / w0;
          out[15 + c] = (h[20 + c] - 2 * wv1 * Sv - wvv * S) / w0;
        }
      }
    }
  }

  /**
   * Newton closest-point iteration from (u, v). Returns [u, v, distance²] in `res`.
   * Falls back to Gauss–Newton steps when the full Hessian is indefinite.
   */
  project(px: number, py: number, pz: number, u: number, v: number, res: Float64Array, maxIter = 20): void {
    const out = this.scratch;
    const du = this.u1 - this.u0, dv = this.v1 - this.v0;
    u = Math.min(this.u1, Math.max(this.u0, u));
    v = Math.min(this.v1, Math.max(this.v0, v));
    for (let it = 0; it < maxIter; it++) {
      this.eval(u, v, 2, out);
      const rx = out[0] - px, ry = out[1] - py, rz = out[2] - pz;
      const f = rx * out[3] + ry * out[4] + rz * out[5];
      const g = rx * out[6] + ry * out[7] + rz * out[8];
      const SuSu = out[3] * out[3] + out[4] * out[4] + out[5] * out[5];
      const SvSv = out[6] * out[6] + out[7] * out[7] + out[8] * out[8];
      const SuSv = out[3] * out[6] + out[4] * out[7] + out[5] * out[8];
      let a = SuSu + rx * out[9] + ry * out[10] + rz * out[11];
      let b = SuSv + rx * out[12] + ry * out[13] + rz * out[14];
      let c = SvSv + rx * out[15] + ry * out[16] + rz * out[17];
      let det = a * c - b * b;
      if (!(a > 0) || !(det > 1e-30 * (a * c + 1e-300))) {
        a = SuSu;
        b = SuSv;
        c = SvSv;
        det = a * c - b * b;
        if (!(det > 1e-300)) break;
      }
      let ddu = -(c * f - b * g) / det;
      let ddv = -(a * g - b * f) / det;
      // limit the step to a quarter of the domain
      const lim = 0.25;
      if (Math.abs(ddu) > lim * du) ddu = Math.sign(ddu) * lim * du;
      if (Math.abs(ddv) > lim * dv) ddv = Math.sign(ddv) * lim * dv;
      const un = Math.min(this.u1, Math.max(this.u0, u + ddu));
      const vn = Math.min(this.v1, Math.max(this.v0, v + ddv));
      const moved = Math.abs(un - u) / du + Math.abs(vn - v) / dv;
      u = un;
      v = vn;
      if (moved < 1e-12) break;
    }
    this.eval(u, v, 0, out);
    res[0] = u;
    res[1] = v;
    res[2] = (out[0] - px) ** 2 + (out[1] - py) ** 2 + (out[2] - pz) ** 2;
  }
  private scratch = new Float64Array(18);
}

/** Point on a B-spline surface. */
export function evalSurface(s: BSplineSurfaceDef, u: number, v: number): Vec3 {
  const ev = new SurfaceEvaluator(s);
  const out = new Float64Array(18);
  ev.eval(u, v, 0, out);
  return [out[0], out[1], out[2]];
}

export interface SurfaceDerivs {
  point: Vec3;
  du: Vec3;
  dv: Vec3;
  /** unit normal du × dv (zero vector at singular points) */
  normal: Vec3;
}

/** Point, first partial derivatives and unit normal (du × dv). */
export function surfaceDerivs(s: BSplineSurfaceDef, u: number, v: number): SurfaceDerivs {
  const ev = new SurfaceEvaluator(s);
  const o = new Float64Array(18);
  ev.eval(u, v, 1, o);
  const du: Vec3 = [o[3], o[4], o[5]], dv: Vec3 = [o[6], o[7], o[8]];
  return { point: [o[0], o[1], o[2]], du, dv, normal: normalize(cross(du, dv)) };
}

/** Second partial derivatives as well (Suu, Suv, Svv). */
export function surfaceDerivs2(s: BSplineSurfaceDef, u: number, v: number): SurfaceDerivs & { duu: Vec3; duv: Vec3; dvv: Vec3 } {
  const ev = new SurfaceEvaluator(s);
  const o = new Float64Array(18);
  ev.eval(u, v, 2, o);
  const du: Vec3 = [o[3], o[4], o[5]], dv: Vec3 = [o[6], o[7], o[8]];
  return {
    point: [o[0], o[1], o[2]],
    du,
    dv,
    normal: normalize(cross(du, dv)),
    duu: [o[9], o[10], o[11]],
    duv: [o[12], o[13], o[14]],
    dvv: [o[15], o[16], o[17]],
  };
}

/**
 * Triangulate a surface on a regular (segU+1)×(segV+1) parameter grid. Triangles are CCW around
 * the surface normal du × dv; normals are analytic.
 */
export function tessellateSurface(s: BSplineSurfaceDef, segU: number, segV: number): MeshData {
  const nu = Math.max(1, Math.floor(segU)), nv = Math.max(1, Math.floor(segV));
  const ev = new SurfaceEvaluator(s);
  const o = new Float64Array(18);
  const nVert = (nu + 1) * (nv + 1);
  const positions = new Float32Array(3 * nVert);
  const normals = new Float32Array(3 * nVert);
  for (let i = 0; i <= nu; i++) {
    const u = ev.u0 + ((ev.u1 - ev.u0) * i) / nu;
    for (let j = 0; j <= nv; j++) {
      const v = ev.v0 + ((ev.v1 - ev.v0) * j) / nv;
      ev.eval(u, v, 1, o);
      const k = i * (nv + 1) + j;
      positions[3 * k] = o[0];
      positions[3 * k + 1] = o[1];
      positions[3 * k + 2] = o[2];
      let n = cross([o[3], o[4], o[5]], [o[6], o[7], o[8]]);
      if (length(n) < 1e-14) {
        // degenerate (collapsed edge): nudge inwards
        const e = 1e-6;
        ev.eval(Math.min(ev.u1, Math.max(ev.u0, u + (i === nu ? -e : e) * (ev.u1 - ev.u0))), Math.min(ev.v1, Math.max(ev.v0, v + (j === nv ? -e : e) * (ev.v1 - ev.v0))), 1, o);
        n = cross([o[3], o[4], o[5]], [o[6], o[7], o[8]]);
      }
      n = normalize(n);
      normals[3 * k] = n[0];
      normals[3 * k + 1] = n[1];
      normals[3 * k + 2] = n[2];
    }
  }
  const indices = new Uint32Array(6 * nu * nv);
  let t = 0;
  for (let i = 0; i < nu; i++)
    for (let j = 0; j < nv; j++) {
      const a = i * (nv + 1) + j, b = (i + 1) * (nv + 1) + j, c = (i + 1) * (nv + 1) + j + 1, d = i * (nv + 1) + j + 1;
      indices[t++] = a;
      indices[t++] = b;
      indices[t++] = c;
      indices[t++] = a;
      indices[t++] = c;
      indices[t++] = d;
    }
  return { positions, indices, normals };
}

export interface SurfaceClosestPoint {
  u: number;
  v: number;
  point: Vec3;
  /** unsigned distance */
  distance: number;
  /** distance signed by the surface normal (du × dv) */
  signedDistance: number;
  normal: Vec3;
}

/** Closest point on a surface: coarse grid search (or the guess) + Newton refinement. */
export function closestPointOnSurface(s: BSplineSurfaceDef, p: Vec3, guess?: { u: number; v: number }): SurfaceClosestPoint {
  const ev = new SurfaceEvaluator(s);
  return closestWithEvaluator(ev, p, guess);
}

function closestWithEvaluator(ev: SurfaceEvaluator, p: Vec3, guess?: { u: number; v: number }): SurfaceClosestPoint {
  const o = new Float64Array(18);
  const res = new Float64Array(3);
  const starts: [number, number][] = [];
  if (guess) starts.push([guess.u, guess.v]);
  const gu = Math.min(48, Math.max(8, 3 * (ev.s.nu - ev.p) + 1));
  const gv = Math.min(48, Math.max(8, 3 * (ev.s.nv - ev.q) + 1));
  let bu = ev.u0, bv = ev.v0, bd = Infinity;
  for (let i = 0; i <= gu; i++) {
    const u = ev.u0 + ((ev.u1 - ev.u0) * i) / gu;
    for (let j = 0; j <= gv; j++) {
      const v = ev.v0 + ((ev.v1 - ev.v0) * j) / gv;
      ev.eval(u, v, 0, o);
      const d = (o[0] - p[0]) ** 2 + (o[1] - p[1]) ** 2 + (o[2] - p[2]) ** 2;
      if (d < bd) {
        bd = d;
        bu = u;
        bv = v;
      }
    }
  }
  starts.push([bu, bv]);
  let best = { u: bu, v: bv, d2: Infinity };
  for (const [u, v] of starts) {
    ev.project(p[0], p[1], p[2], u, v, res, 30);
    if (res[2] < best.d2) best = { u: res[0], v: res[1], d2: res[2] };
  }
  ev.eval(best.u, best.v, 1, o);
  const point: Vec3 = [o[0], o[1], o[2]];
  const normal = normalize(cross([o[3], o[4], o[5]], [o[6], o[7], o[8]]));
  const distance = Math.sqrt(best.d2);
  const sd = dot([p[0] - point[0], p[1] - point[1], p[2] - point[2]], normal);
  return { u: best.u, v: best.v, point, distance, signedDistance: sd < 0 ? -distance : distance, normal };
}

// ===========================================================================
// Surface fitting
// ===========================================================================

export interface SurfaceFitOptions {
  indices?: ArrayLike<number>;
  /** used only to orient the default frame (surface normal towards the mean point normal) */
  normals?: ArrayLike<number> | null;
  /** control points in u (along frame.xAxis) */
  nu: number;
  /** control points in v (along frame.yAxis) */
  nv: number;
  /** degree in both directions (default 3) */
  degree?: number;
  /**
   * Thin-plate energy weight λ relative to the mean squared error (dimensionless and scale
   * invariant): minimise mean|S−P|² + λ·E_tp. A surface feature with f waves across the domain is
   * attenuated by ≈ λ(2πf)⁴, so useful values are ~1e-10 (detail) … 1e-5 (very smooth).
   * Default: 0.03 / (π (max(nu,nv) − degree))⁴, i.e. only control-net-scale wiggles are damped.
   * Any λ > 0 also defines the surface across holes and in the extension area.
   */
  smoothing?: number;
  /** fraction by which the parameter rectangle is extended beyond the data on each side (default 0.05) */
  extend?: number;
  /** parametrisation plane; default = best-fit plane with in-plane axes from PCA */
  frame?: Frame;
  /** parameter-correction passes (re-project points, refit), default 2 */
  iterations?: number;
  /** max points used by the fit (deterministic sub-sample, default 100k); residuals use all points */
  maxPoints?: number;
  seed?: number;
}

export interface SurfaceFitResult {
  surface: BSplineSurfaceDef;
  /** statistics of the signed closest-point distances of all selected points */
  stats: FitStats;
  /** signed distance of each selected point to the surface (selection order) */
  residuals: Float32Array;
  /** the parametrisation frame used */
  frame: Frame;
}

/** Default thin-plate weight for a control net: damps only wiggles at the control-net scale. */
export function defaultSurfaceSmoothing(nu: number, nv: number, degree = 3): number {
  const k = Math.max(1, Math.max(nu, nv) - degree);
  return 0.03 / Math.pow(Math.PI * k, 4);
}

/** Best-fit plane frame of a selection: origin = centroid, z = normal, x = principal direction. */
function defaultFrame(pos: PointArray, opts: SurfaceFitOptions): Frame {
  const S = buildSample(pos, { indices: opts.indices, normals: opts.normals, maxPoints: 100000, seed: opts.seed }, 3, "fitBSplineSurface");
  const cov = new Float64Array(9);
  for (let i = 0; i < S.m; i++) {
    const x = S.xs[3 * i], y = S.xs[3 * i + 1], z = S.xs[3 * i + 2];
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
  if (!(e.values[1] > 1e-10 * e.values[2])) throw new Error("fitBSplineSurface: points are collinear");
  let z = e.vectors[0];
  if (S.ns) {
    let mx = 0, my = 0, mz = 0;
    for (let i = 0; i < S.m; i++) {
      mx += S.ns[3 * i];
      my += S.ns[3 * i + 1];
      mz += S.ns[3 * i + 2];
    }
    if (mx * z[0] + my * z[1] + mz * z[2] < 0) z = [-z[0], -z[1], -z[2]];
  } else z = canonicalSign(z);
  let x = canonicalSign(e.vectors[2]);
  x = normalize([x[0] - dot(x, z) * z[0], x[1] - dot(x, z) * z[1], x[2] - dot(x, z) * z[2]]);
  const y = cross(z, x);
  return { origin: S.center, xAxis: x, yAxis: y, zAxis: z };
}

interface LocalData {
  /** local (frame) coordinates of the fit sample */
  L: Float64Array;
  m: number;
  u: Float64Array;
  v: Float64Array;
}

/**
 * Regularised least-squares B-spline surface fit ("Fit Surface"): points are parametrised by
 * projection onto a frame plane (bounding rectangle + extension → [0,1]²); the normal equations
 * (data + thin-plate energy) are assembled sparsely and solved with a banded Cholesky; optional
 * parameter-correction passes re-project the points onto the surface and refit.
 */
export function fitBSplineSurface(pos: PointArray, opts: SurfaceFitOptions): SurfaceFitResult {
  const p = Math.floor(opts.degree ?? 3);
  const nu = Math.floor(opts.nu), nv = Math.floor(opts.nv);
  if (!(p >= 1 && p <= 7)) throw new Error("fitBSplineSurface: degree must be in [1, 7]");
  if (!(nu >= p + 1 && nv >= p + 1)) throw new Error(`fitBSplineSurface: nu and nv must be >= degree + 1 (${p + 1})`);
  if (nu * nv > 250 * 250) throw new Error("fitBSplineSurface: too many control points (nu*nv > 62500)");
  const lam = opts.smoothing ?? defaultSurfaceSmoothing(nu, nv, p);
  if (!(lam >= 0)) throw new Error("fitBSplineSurface: smoothing must be >= 0");
  const ext = Math.max(0, opts.extend ?? 0.05);
  const frame = opts.frame ? orthonormalFrame(opts.frame) : defaultFrame(pos, opts);
  const { origin: O, xAxis: X, yAxis: Y, zAxis: Z } = frame;
  // bounding rectangle over ALL selected points
  const count = selectionCount(pos, opts.indices);
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (let k = 0; k < count; k++) {
    const i = opts.indices ? opts.indices[k] : k;
    const dx = pos[3 * i] - O[0], dy = pos[3 * i + 1] - O[1], dz = pos[3 * i + 2] - O[2];
    const lx = dx * X[0] + dy * X[1] + dz * X[2];
    const ly = dx * Y[0] + dy * Y[1] + dz * Y[2];
    if (!Number.isFinite(lx + ly)) continue;
    if (lx < xmin) xmin = lx;
    if (lx > xmax) xmax = lx;
    if (ly < ymin) ymin = ly;
    if (ly > ymax) ymax = ly;
  }
  const W0 = xmax - xmin, H0 = ymax - ymin;
  const size = Math.max(W0, H0);
  if (!(size > 0) || !(Math.min(W0, H0) > 1e-9 * size)) {
    throw new Error("fitBSplineSurface: the projection of the points onto the frame plane is degenerate (zero width) — choose another frame");
  }
  const s0 = xmin - ext * W0, t0 = ymin - ext * H0;
  const W = W0 * (1 + 2 * ext), H = H0 * (1 + 2 * ext);
  // fit sample in local coordinates
  const S = buildSample(pos, { indices: opts.indices, maxPoints: opts.maxPoints ?? 100000, seed: opts.seed }, 3, "fitBSplineSurface");
  const m = S.m;
  const data: LocalData = { L: new Float64Array(3 * m), m, u: new Float64Array(m), v: new Float64Array(m) };
  for (let k = 0; k < m; k++) {
    const wx = S.center[0] + S.scale * S.xs[3 * k] - O[0];
    const wy = S.center[1] + S.scale * S.xs[3 * k + 1] - O[1];
    const wz = S.center[2] + S.scale * S.xs[3 * k + 2] - O[2];
    const lx = wx * X[0] + wy * X[1] + wz * X[2];
    const ly = wx * Y[0] + wy * Y[1] + wz * Y[2];
    const lz = wx * Z[0] + wy * Z[1] + wz * Z[2];
    data.L[3 * k] = lx;
    data.L[3 * k + 1] = ly;
    data.L[3 * k + 2] = lz;
    data.u[k] = Math.min(1, Math.max(0, (lx - s0) / W));
    data.v[k] = Math.min(1, Math.max(0, (ly - t0) / H));
  }
  const U = clampedUniformKnots(nu, p);
  const V = clampedUniformKnots(nv, p);
  const aspect = H / W;
  const tp = thinPlateGrams(U, V, p, nu, nv);
  let local = solveSurfaceLS(data, U, V, p, nu, nv, lam, aspect, tp);
  const iters = Math.max(0, Math.floor(opts.iterations ?? 2));
  const res = new Float64Array(3);
  for (let it = 0; it < iters; it++) {
    const ev = new SurfaceEvaluator(local);
    for (let k = 0; k < m; k++) {
      ev.project(data.L[3 * k], data.L[3 * k + 1], data.L[3 * k + 2], data.u[k], data.v[k], res, 8);
      data.u[k] = res[0];
      data.v[k] = res[1];
    }
    local = solveSurfaceLS(data, U, V, p, nu, nv, lam, aspect, tp);
  }
  // residuals of all selected points (Newton from the planar parametrisation)
  const ev = new SurfaceEvaluator(local);
  const o = new Float64Array(18);
  const residuals = new Float32Array(count);
  let ss = 0, sum = 0, mx = 0, cnt = 0;
  for (let k = 0; k < count; k++) {
    const i = opts.indices ? opts.indices[k] : k;
    const dx = pos[3 * i] - O[0], dy = pos[3 * i + 1] - O[1], dz = pos[3 * i + 2] - O[2];
    const lx = dx * X[0] + dy * X[1] + dz * X[2];
    const ly = dx * Y[0] + dy * Y[1] + dz * Y[2];
    const lz = dx * Z[0] + dy * Z[1] + dz * Z[2];
    if (!Number.isFinite(lx + ly + lz)) {
      residuals[k] = NaN;
      continue;
    }
    ev.project(lx, ly, lz, (lx - s0) / W, (ly - t0) / H, res, 12);
    ev.eval(res[0], res[1], 1, o);
    const nx = o[4] * o[8] - o[5] * o[7], ny = o[5] * o[6] - o[3] * o[8], nz = o[3] * o[7] - o[4] * o[6];
    const sd = (lx - o[0]) * nx + (ly - o[1]) * ny + (lz - o[2]) * nz;
    const d = Math.sqrt(res[2]);
    const r = sd < 0 ? -d : d;
    residuals[k] = r;
    ss += r * r;
    sum += r;
    mx = Math.max(mx, d);
    cnt++;
  }
  // local → world control points
  const controlPoints: Vec3[] = local.controlPoints.map((P) => [
    O[0] + P[0] * X[0] + P[1] * Y[0] + P[2] * Z[0],
    O[1] + P[0] * X[1] + P[1] * Y[1] + P[2] * Z[1],
    O[2] + P[0] * X[2] + P[1] * Y[2] + P[2] * Z[2],
  ]);
  const surface: BSplineSurfaceDef = { ...local, controlPoints };
  const stats: FitStats = cnt ? { rms: Math.sqrt(ss / cnt), maxAbs: mx, mean: sum / cnt, count: cnt } : { rms: NaN, maxAbs: NaN, mean: NaN, count: 0 };
  return { surface, stats, residuals, frame };
}

function orthonormalFrame(f: Frame): Frame {
  const z = normalize(f.zAxis);
  if (length(z) === 0) throw new Error("fitBSplineSurface: frame.zAxis is zero");
  let x: Vec3 = [f.xAxis[0] - dot(f.xAxis, z) * z[0], f.xAxis[1] - dot(f.xAxis, z) * z[1], f.xAxis[2] - dot(f.xAxis, z) * z[2]];
  if (length(x) < 1e-9) throw new Error("fitBSplineSurface: frame.xAxis is parallel to zAxis");
  x = normalize(x);
  const y = cross(z, x);
  return { origin: [f.origin[0], f.origin[1], f.origin[2]], xAxis: x, yAxis: y, zAxis: z };
}

interface ThinPlateGrams {
  G0u: Float64Array;
  G1u: Float64Array;
  G2u: Float64Array;
  G0v: Float64Array;
  G1v: Float64Array;
  G2v: Float64Array;
}

function thinPlateGrams(U: number[], V: number[], p: number, nu: number, nv: number): ThinPlateGrams {
  return {
    G0u: basisGram(U, p, nu, 0),
    G1u: basisGram(U, p, nu, 1),
    G2u: basisGram(U, p, nu, Math.min(2, p)),
    G0v: basisGram(V, p, nv, 0),
    G1v: basisGram(V, p, nv, 1),
    G2v: basisGram(V, p, nv, Math.min(2, p)),
  };
}

/**
 * Solve min (1/m) Σ |S(u_k, v_k) - L_k|² + λ ∫∫ (a²|S_uu|² + 2|S_uv|² + a⁻²|S_vv|²) du dv
 * (a = H/W) for the control points in local frame coordinates.
 */
function solveSurfaceLS(
  data: LocalData,
  U: number[],
  V: number[],
  p: number,
  nu: number,
  nv: number,
  lam: number,
  aspect: number,
  tp: ThinPlateGrams,
): BSplineSurfaceDef {
  const n = nu * nv;
  // unknown ordering: major index along the larger dimension → bandwidth p*min(nu,nv)+p
  const uMajor = nu >= nv;
  const idx = uMajor ? (i: number, j: number) => i * nv + j : (i: number, j: number) => j * nu + i;
  const bw = uMajor ? p * nv + p : p * nu + p;
  const A = new BandSPD(n, bw);
  const bx = new Float64Array(n), by = new Float64Array(n), bz = new Float64Array(n);
  const Nu = new Float64Array(p + 1), Nv = new Float64Array(p + 1);
  const lu = new Float64Array(p + 1), ru = new Float64Array(p + 1);
  const K = (p + 1) * (p + 1);
  const ids = new Int32Array(K), vals = new Float64Array(K);
  const inv = 1 / data.m;
  for (let k = 0; k < data.m; k++) {
    const u = data.u[k], v = data.v[k];
    const su = findSpan(nu - 1, p, u, U);
    const sv = findSpan(nv - 1, p, v, V);
    basisFunsInto(su, u, p, U, Nu, lu, ru);
    basisFunsInto(sv, v, p, V, Nv, lu, ru);
    // nonzeros in increasing unknown index
    let t = 0;
    if (uMajor) {
      for (let a = 0; a <= p; a++)
        for (let b = 0; b <= p; b++) {
          ids[t] = idx(su - p + a, sv - p + b);
          vals[t++] = Nu[a] * Nv[b];
        }
    } else {
      for (let b = 0; b <= p; b++)
        for (let a = 0; a <= p; a++) {
          ids[t] = idx(su - p + a, sv - p + b);
          vals[t++] = Nu[a] * Nv[b];
        }
    }
    const lx = data.L[3 * k], ly = data.L[3 * k + 1], lz = data.L[3 * k + 2];
    for (let s = 0; s < K; s++) {
      const vs = vals[s] * inv;
      if (vs === 0) continue;
      const I = ids[s];
      bx[I] += vs * lx;
      by[I] += vs * ly;
      bz[I] += vs * lz;
      for (let r = 0; r <= s; r++) A.add(I, ids[r], vs * vals[r]);
    }
  }
  const a2 = aspect * aspect, ia2 = 1 / a2;
  const effLam = Math.max(lam, 1e-12);
  const { G0u, G1u, G2u, G0v, G1v, G2v } = tp;
  for (let i = 0; i < nu; i++)
    for (let j = 0; j < nv; j++) {
      const I = idx(i, j);
      for (let i2 = Math.max(0, i - p); i2 <= Math.min(nu - 1, i + p); i2++)
        for (let j2 = Math.max(0, j - p); j2 <= Math.min(nv - 1, j + p); j2++) {
          const J = idx(i2, j2);
          if (J > I) continue;
          const e =
            a2 * G2u[i * nu + i2] * G0v[j * nv + j2] + 2 * G1u[i * nu + i2] * G1v[j * nv + j2] + ia2 * G0u[i * nu + i2] * G2v[j * nv + j2];
          if (e !== 0) A.add(I, J, effLam * e);
        }
    }
  if (!A.factor()) {
    throw new Error("fitBSplineSurface: singular normal equations — increase smoothing or reduce nu/nv");
  }
  A.solve(bx);
  A.solve(by);
  A.solve(bz);
  const controlPoints: Vec3[] = new Array(n);
  for (let i = 0; i < nu; i++)
    for (let j = 0; j < nv; j++) {
      const I = idx(i, j);
      controlPoints[i * nv + j] = [bx[I], by[I], bz[I]];
    }
  return { degreeU: p, degreeV: p, knotsU: U, knotsV: V, nu, nv, controlPoints };
}
