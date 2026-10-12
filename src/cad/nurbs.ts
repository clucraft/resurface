/**
 * Small, dependency-free NURBS helpers used by the CAD layer (knot conversion, evaluation,
 * reversal, tensor-product interpolation). Pure functions — usable in workers and Node tests.
 *
 * Knot vectors are FULL clamped knot vectors (length = nPoles + degree + 1), matching
 * `BSplineCurveDef` / `BSplineSurfaceDef` in `src/core/types.ts`.
 */
import type { BSplineCurve2Def, BSplineCurveDef, BSplineSurfaceDef, Vec2, Vec3 } from "../core/types";

/** Full knot vector -> distinct knots + multiplicities (OCCT style). */
export function toDistinctKnots(full: readonly number[], tol = 1e-12): { knots: number[]; mults: number[] } {
  const knots: number[] = [];
  const mults: number[] = [];
  const span = full.length ? Math.abs(full[full.length - 1] - full[0]) : 0;
  const eps = Math.max(tol, span * 1e-12);
  for (const k of full) {
    if (knots.length && Math.abs(k - knots[knots.length - 1]) <= eps) mults[mults.length - 1]++;
    else {
      knots.push(k);
      mults.push(1);
    }
  }
  return { knots, mults };
}

/** Distinct knots + multiplicities -> full knot vector. */
export function toFullKnots(knots: readonly number[], mults: readonly number[]): number[] {
  const out: number[] = [];
  for (let i = 0; i < knots.length; i++) for (let m = 0; m < mults[i]; m++) out.push(knots[i]);
  return out;
}

/** Find the knot span index i such that knots[i] <= t < knots[i+1] (clamped at the end). */
export function findSpan(n: number, p: number, t: number, U: readonly number[]): number {
  // n = number of control points - 1
  if (t >= U[n + 1]) return n;
  if (t <= U[p]) return p;
  let lo = p, hi = n + 1;
  let mid = (lo + hi) >> 1;
  while (t < U[mid] || t >= U[mid + 1]) {
    if (t < U[mid]) hi = mid;
    else lo = mid;
    mid = (lo + hi) >> 1;
  }
  return mid;
}

/** Non-vanishing basis functions N_{span-p..span,p}(t) (The NURBS Book A2.2). */
export function basisFuns(span: number, t: number, p: number, U: readonly number[]): number[] {
  const N = new Array<number>(p + 1).fill(0);
  const left = new Array<number>(p + 1).fill(0);
  const right = new Array<number>(p + 1).fill(0);
  N[0] = 1;
  for (let j = 1; j <= p; j++) {
    left[j] = t - U[span + 1 - j];
    right[j] = U[span + j] - t;
    let saved = 0;
    for (let r = 0; r < j; r++) {
      const den = right[r + 1] + left[j - r];
      const tmp = den !== 0 ? N[r] / den : 0;
      N[r] = saved + right[r + 1] * tmp;
      saved = left[j - r] * tmp;
    }
    N[j] = saved;
  }
  return N;
}

/** Evaluate a (rational) B-spline curve of any dimension. */
export function evalCurveN(
  degree: number,
  knots: readonly number[],
  points: readonly (readonly number[])[],
  weights: readonly number[] | undefined,
  t: number,
): number[] {
  const n = points.length - 1;
  const dim = points[0].length;
  const span = findSpan(n, degree, t, knots);
  const N = basisFuns(span, t, degree, knots);
  const out = new Array<number>(dim).fill(0);
  let wsum = 0;
  for (let i = 0; i <= degree; i++) {
    const idx = span - degree + i;
    const w = weights ? weights[idx] : 1;
    const f = N[i] * w;
    wsum += f;
    const P = points[idx];
    for (let d = 0; d < dim; d++) out[d] += f * P[d];
  }
  if (weights && wsum !== 0) for (let d = 0; d < dim; d++) out[d] /= wsum;
  return out;
}

export const evalCurve3 = (c: BSplineCurveDef, t: number): Vec3 =>
  evalCurveN(c.degree, c.knots, c.controlPoints, c.weights, t) as Vec3;

export const evalCurve2 = (c: BSplineCurve2Def, t: number): Vec2 =>
  evalCurveN(c.degree, c.knots, c.controlPoints, c.weights, t) as Vec2;

export const curveDomain = (c: { degree: number; knots: readonly number[] }): [number, number] => [
  c.knots[c.degree],
  c.knots[c.knots.length - 1 - c.degree],
];

/** Evaluate a (rational) B-spline surface. */
export function evalSurface(s: BSplineSurfaceDef, u: number, v: number): Vec3 {
  const { degreeU: p, degreeV: q, knotsU, knotsV, nu, nv, controlPoints: P, weights: W } = s;
  const su = findSpan(nu - 1, p, u, knotsU);
  const sv = findSpan(nv - 1, q, v, knotsV);
  const Nu = basisFuns(su, u, p, knotsU);
  const Nv = basisFuns(sv, v, q, knotsV);
  let x = 0, y = 0, z = 0, w = 0;
  for (let a = 0; a <= p; a++) {
    const i = su - p + a;
    for (let b = 0; b <= q; b++) {
      const j = sv - q + b;
      const idx = i * nv + j;
      const f = Nu[a] * Nv[b] * (W ? W[idx] : 1);
      const c = P[idx];
      x += f * c[0];
      y += f * c[1];
      z += f * c[2];
      w += f;
    }
  }
  return W && w !== 0 ? [x / w, y / w, z / w] : [x, y, z];
}

export const surfaceDomain = (s: BSplineSurfaceDef): { u: [number, number]; v: [number, number] } => ({
  u: [s.knotsU[s.degreeU], s.knotsU[s.knotsU.length - 1 - s.degreeU]],
  v: [s.knotsV[s.degreeV], s.knotsV[s.knotsV.length - 1 - s.degreeV]],
});

/** Reverse a knot vector: k' = a + b - k (reversed order). */
export function reverseKnots(knots: readonly number[]): number[] {
  const a = knots[0], b = knots[knots.length - 1];
  const out = new Array<number>(knots.length);
  for (let i = 0; i < knots.length; i++) out[i] = a + b - knots[knots.length - 1 - i];
  return out;
}

/** Reverse the parametric direction of a curve (same parameter domain). */
export function reverseCurve<T extends BSplineCurveDef | BSplineCurve2Def>(c: T): T {
  return {
    ...c,
    knots: reverseKnots(c.knots),
    controlPoints: [...c.controlPoints].reverse(),
    weights: c.weights ? [...c.weights].reverse() : undefined,
  } as T;
}

/** Mirror the U direction of a surface: S'(u, v) = S(a + b - u, v). Flips the surface normal. */
export function mirrorSurfaceU(s: BSplineSurfaceDef): BSplineSurfaceDef {
  const { nu, nv } = s;
  const cps: Vec3[] = new Array(nu * nv);
  const ws: number[] | undefined = s.weights ? new Array(nu * nv) : undefined;
  for (let i = 0; i < nu; i++)
    for (let j = 0; j < nv; j++) {
      cps[i * nv + j] = s.controlPoints[(nu - 1 - i) * nv + j];
      if (ws) ws[i * nv + j] = s.weights![(nu - 1 - i) * nv + j];
    }
  return { ...s, knotsU: reverseKnots(s.knotsU), controlPoints: cps, weights: ws };
}

/** Map a 2D curve's control points u -> a + b - u (exact: the map is affine). */
export function mirrorCurve2U(c: BSplineCurve2Def, a: number, b: number): BSplineCurve2Def {
  return { ...c, controlPoints: c.controlPoints.map(([u, v]) => [a + b - u, v] as Vec2) };
}

// ---------------------------------------------------------------------------
// Interpolation
// ---------------------------------------------------------------------------

/** Clamped knot vector for interpolation at parameters t (de Boor averaging, NURBS Book 9.8). */
export function averagingKnots(t: readonly number[], p: number): number[] {
  const n = t.length - 1;
  const U: number[] = [];
  for (let i = 0; i <= p; i++) U.push(t[0]);
  for (let j = 1; j <= n - p; j++) {
    let s = 0;
    for (let i = j; i <= j + p - 1; i++) s += t[i];
    U.push(s / p);
  }
  for (let i = 0; i <= p; i++) U.push(t[n]);
  return U;
}

/** LU factorisation with partial pivoting of a dense n x n matrix (row-major). */
class DenseLU {
  private lu: Float64Array;
  private piv: Int32Array;
  constructor(A: Float64Array, private n: number) {
    this.lu = A.slice();
    this.piv = new Int32Array(n);
    const lu = this.lu;
    for (let i = 0; i < n; i++) this.piv[i] = i;
    for (let k = 0; k < n; k++) {
      let p = k, max = Math.abs(lu[k * n + k]);
      for (let i = k + 1; i < n; i++) {
        const v = Math.abs(lu[i * n + k]);
        if (v > max) { max = v; p = i; }
      }
      if (max < 1e-300) throw new Error("Interpolation matrix is singular");
      if (p !== k) {
        for (let j = 0; j < n; j++) {
          const tmp = lu[k * n + j]; lu[k * n + j] = lu[p * n + j]; lu[p * n + j] = tmp;
        }
        const t = this.piv[k]; this.piv[k] = this.piv[p]; this.piv[p] = t;
      }
      const d = lu[k * n + k];
      for (let i = k + 1; i < n; i++) {
        const f = (lu[i * n + k] /= d);
        if (f !== 0) for (let j = k + 1; j < n; j++) lu[i * n + j] -= f * lu[k * n + j];
      }
    }
  }
  solve(b: Float64Array): Float64Array {
    const { n, lu, piv } = this;
    const x = new Float64Array(n);
    for (let i = 0; i < n; i++) x[i] = b[piv[i]];
    for (let i = 0; i < n; i++) {
      let s = x[i];
      for (let j = 0; j < i; j++) s -= lu[i * n + j] * x[j];
      x[i] = s;
    }
    for (let i = n - 1; i >= 0; i--) {
      let s = x[i];
      for (let j = i + 1; j < n; j++) s -= lu[i * n + j] * x[j];
      x[i] = s / lu[i * n + i];
    }
    return x;
  }
}

function interpolationMatrix(t: readonly number[], p: number, U: readonly number[]): DenseLU {
  const n = t.length;
  const A = new Float64Array(n * n);
  for (let k = 0; k < n; k++) {
    const span = findSpan(n - 1, p, t[k], U);
    const N = basisFuns(span, t[k], p, U);
    for (let i = 0; i <= p; i++) A[k * n + span - p + i] = N[i];
  }
  return new DenseLU(A, n);
}

/**
 * Tensor-product interpolation of a grid of points Q[k][l] sampled at parameters (us[k], vs[l]).
 * Returns a non-rational B-spline surface of degree min(3, samples-1) whose parameterisation
 * reproduces the sampling parameters (so the surface approximates S(u,v) "in place").
 */
export function interpolateSurfaceGrid(
  us: readonly number[],
  vs: readonly number[],
  Q: readonly (readonly Vec3[])[],
  maxDegree = 3,
): BSplineSurfaceDef {
  const nu = us.length, nv = vs.length;
  const p = Math.min(maxDegree, nu - 1), q = Math.min(maxDegree, nv - 1);
  const U = averagingKnots(us, p), V = averagingKnots(vs, q);
  const luU = interpolationMatrix(us, p, U);
  const luV = interpolationMatrix(vs, q, V);
  // R[k][j]: interpolate along v for each row k
  const R: Vec3[][] = [];
  const rhs = new Float64Array(nv);
  for (let k = 0; k < nu; k++) {
    const row: Vec3[] = new Array(nv);
    const comps: Float64Array[] = [];
    for (let d = 0; d < 3; d++) {
      for (let l = 0; l < nv; l++) rhs[l] = Q[k][l][d];
      comps.push(luV.solve(rhs));
    }
    for (let l = 0; l < nv; l++) row[l] = [comps[0][l], comps[1][l], comps[2][l]];
    R.push(row);
  }
  const P: Vec3[] = new Array(nu * nv);
  const rhsU = new Float64Array(nu);
  for (let j = 0; j < nv; j++) {
    const comps: Float64Array[] = [];
    for (let d = 0; d < 3; d++) {
      for (let k = 0; k < nu; k++) rhsU[k] = R[k][j][d];
      comps.push(luU.solve(rhsU));
    }
    for (let i = 0; i < nu; i++) P[i * nv + j] = [comps[0][i], comps[1][i], comps[2][i]];
  }
  return { degreeU: p, degreeV: q, knotsU: U, knotsV: V, nu, nv, controlPoints: P };
}

/**
 * Interpolating B-spline curve (degree min(3, n-1)) through points `pts` at parameters `ts`,
 * any dimension. The curve parameterisation reproduces `ts`.
 */
export function interpolateCurveN(
  ts: readonly number[],
  pts: readonly (readonly number[])[],
  maxDegree = 3,
): { degree: number; knots: number[]; controlPoints: number[][] } {
  const n = ts.length;
  if (n < 2) throw new Error("interpolateCurveN: need at least 2 points");
  const p = Math.min(maxDegree, n - 1);
  const U = averagingKnots(ts, p);
  const lu = interpolationMatrix(ts, p, U);
  const dim = pts[0].length;
  const comps: Float64Array[] = [];
  const rhs = new Float64Array(n);
  for (let d = 0; d < dim; d++) {
    for (let k = 0; k < n; k++) rhs[k] = pts[k][d];
    comps.push(lu.solve(rhs));
  }
  const cps: number[][] = [];
  for (let i = 0; i < n; i++) cps.push(comps.map((c) => c[i]));
  return { degree: p, knots: U, controlPoints: cps };
}

/** Validate the basic shape invariants of a surface definition; returns an error string or null. */
export function checkSurfaceDef(s: BSplineSurfaceDef): string | null {
  if (!(s.degreeU >= 1 && s.degreeV >= 1)) return "degrees must be >= 1";
  if (s.nu < s.degreeU + 1 || s.nv < s.degreeV + 1) return "not enough control points for the degree";
  if (s.controlPoints.length !== s.nu * s.nv) return `controlPoints.length (${s.controlPoints.length}) != nu*nv (${s.nu * s.nv})`;
  if (s.knotsU.length !== s.nu + s.degreeU + 1) return `knotsU.length must be nu+degreeU+1 = ${s.nu + s.degreeU + 1}`;
  if (s.knotsV.length !== s.nv + s.degreeV + 1) return `knotsV.length must be nv+degreeV+1 = ${s.nv + s.degreeV + 1}`;
  if (s.weights && s.weights.length !== s.nu * s.nv) return "weights.length must equal nu*nv";
  return null;
}

export function checkCurveDef(c: { degree: number; knots: readonly number[]; controlPoints: readonly unknown[]; weights?: readonly number[] }): string | null {
  if (!(c.degree >= 1)) return "degree must be >= 1";
  if (c.controlPoints.length < c.degree + 1) return "not enough control points for the degree";
  if (c.knots.length !== c.controlPoints.length + c.degree + 1) return `knots.length must be controlPoints.length+degree+1 = ${c.controlPoints.length + c.degree + 1}`;
  if (c.weights && c.weights.length !== c.controlPoints.length) return "weights.length must equal controlPoints.length";
  return null;
}
