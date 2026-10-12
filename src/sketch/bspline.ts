/**
 * Planar B-spline helpers used by the sketcher: evaluation, derivatives, global interpolation
 * (open and closed/periodic-like), least-squares approximation, closest point and splitting.
 *
 * All curves are clamped (open knot vectors), see {@link BSplineCurve2Def}.
 */
import type { BSplineCurve2Def, Vec2 } from "../core/types";
import { solveLinear } from "../core/linalg";

/** Knot span index i such that U[i] <= u < U[i+1] (Piegl & Tiller A2.1). */
export function findSpan(nCtrl: number, p: number, u: number, U: ArrayLike<number>): number {
  const n = nCtrl - 1;
  if (u >= U[n + 1]) {
    // last non-empty span
    let k = n;
    while (k > p && U[k] >= U[n + 1]) k--;
    return k;
  }
  if (u <= U[p]) return p;
  let low = p;
  let high = n + 1;
  let mid = (low + high) >> 1;
  while (u < U[mid] || u >= U[mid + 1]) {
    if (u < U[mid]) high = mid;
    else low = mid;
    mid = (low + high) >> 1;
  }
  return mid;
}

/** Non-zero basis functions N[span-p .. span] at u (Piegl & Tiller A2.2). */
export function basisFuns(span: number, u: number, p: number, U: ArrayLike<number>, out = new Float64Array(p + 1)): Float64Array {
  const left = new Float64Array(p + 1);
  const right = new Float64Array(p + 1);
  out[0] = 1;
  for (let j = 1; j <= p; j++) {
    left[j] = u - U[span + 1 - j];
    right[j] = U[span + j] - u;
    let saved = 0;
    for (let r = 0; r < j; r++) {
      const den = right[r + 1] + left[j - r];
      const temp = den !== 0 ? out[r] / den : 0;
      out[r] = saved + right[r + 1] * temp;
      saved = left[j - r] * temp;
    }
    out[j] = saved;
  }
  return out;
}

/** Parameter domain [U[p], U[n+1]]. */
export function bsplineDomain(c: BSplineCurve2Def): [number, number] {
  return [c.knots[c.degree], c.knots[c.controlPoints.length]];
}

/** Evaluate a (possibly rational) curve at u. */
export function evalBSpline2(c: BSplineCurve2Def, u: number): Vec2 {
  const p = c.degree;
  const n = c.controlPoints.length;
  const [u0, u1] = bsplineDomain(c);
  u = Math.min(u1, Math.max(u0, u));
  const span = findSpan(n, p, u, c.knots);
  const N = basisFuns(span, u, p, c.knots);
  let x = 0, y = 0, w = 0;
  for (let j = 0; j <= p; j++) {
    const i = span - p + j;
    const cp = c.controlPoints[i];
    const wt = c.weights ? c.weights[i] : 1;
    const b = N[j] * wt;
    x += b * cp[0];
    y += b * cp[1];
    w += b;
  }
  return w !== 0 ? [x / w, y / w] : [x, y];
}

/** First derivative dC/du (analytic for non-rational curves, central difference for rational ones). */
export function derivBSpline2(c: BSplineCurve2Def, u: number): Vec2 {
  const p = c.degree;
  const n = c.controlPoints.length;
  const [u0, u1] = bsplineDomain(c);
  if (c.weights || p < 1) {
    const h = (u1 - u0) * 1e-6;
    const a = Math.max(u0, u - h), b = Math.min(u1, u + h);
    const pa = evalBSpline2(c, a), pb = evalBSpline2(c, b);
    const d = b - a || 1;
    return [(pb[0] - pa[0]) / d, (pb[1] - pa[1]) / d];
  }
  u = Math.min(u1, Math.max(u0, u));
  const U = c.knots;
  // derivative curve: degree p-1, control points Q_i = p (P_{i+1}-P_i)/(U_{i+p+1}-U_{i+1}), knots U[1..m-1]
  const span = findSpan(n, p, u, U);
  const Ud = U.slice(1, U.length - 1);
  const spanD = span - 1; // span in the derivative knot vector
  const N = basisFuns(spanD, u, p - 1, Ud);
  let x = 0, y = 0;
  for (let j = 0; j <= p - 1; j++) {
    const i = spanD - (p - 1) + j; // index of Q
    const den = U[i + p + 1] - U[i + 1];
    if (den === 0) continue;
    const f = (p / den) * N[j];
    x += f * (c.controlPoints[i + 1][0] - c.controlPoints[i][0]);
    y += f * (c.controlPoints[i + 1][1] - c.controlPoints[i][1]);
  }
  return [x, y];
}

/** Uniformly sample n+1 points over the parameter domain. */
export function sampleBSpline2(c: BSplineCurve2Def, n = 64): Vec2[] {
  const [u0, u1] = bsplineDomain(c);
  const out: Vec2[] = [];
  for (let i = 0; i <= n; i++) out.push(evalBSpline2(c, u0 + ((u1 - u0) * i) / n));
  return out;
}

/** Closest point on the curve to p: coarse sampling followed by Newton refinement. */
export function closestPointBSpline2(c: BSplineCurve2Def, p: Vec2, samples = 0): { u: number; point: Vec2; distance: number } {
  const [u0, u1] = bsplineDomain(c);
  const ns = samples > 0 ? samples : Math.max(32, c.controlPoints.length * 8);
  let bestU = u0;
  let bestD = Infinity;
  for (let i = 0; i <= ns; i++) {
    const u = u0 + ((u1 - u0) * i) / ns;
    const q = evalBSpline2(c, u);
    const d = (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2;
    if (d < bestD) {
      bestD = d;
      bestU = u;
    }
  }
  // Newton on f(u) = C'(u).(C(u)-p) with a numerically estimated second derivative
  let u = bestU;
  const h = (u1 - u0) * 1e-5;
  for (let it = 0; it < 20; it++) {
    const q = evalBSpline2(c, u);
    const d1 = derivBSpline2(c, u);
    const f = d1[0] * (q[0] - p[0]) + d1[1] * (q[1] - p[1]);
    const ua = Math.max(u0, u - h), ub = Math.min(u1, u + h);
    const da = derivBSpline2(c, ua), db = derivBSpline2(c, ub);
    const d2: Vec2 = [(db[0] - da[0]) / (ub - ua || 1), (db[1] - da[1]) / (ub - ua || 1)];
    const fp = d1[0] * d1[0] + d1[1] * d1[1] + d2[0] * (q[0] - p[0]) + d2[1] * (q[1] - p[1]);
    if (!(Math.abs(fp) > 1e-300)) break;
    let un = u - f / fp;
    un = Math.min(u1, Math.max(u0, un));
    if (Math.abs(un - u) < (u1 - u0) * 1e-14) {
      u = un;
      break;
    }
    const qn = evalBSpline2(c, un);
    const dn = (qn[0] - p[0]) ** 2 + (qn[1] - p[1]) ** 2;
    const dc = (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2;
    if (dn > dc) break;
    u = un;
  }
  const point = evalBSpline2(c, u);
  return { u, point, distance: Math.hypot(point[0] - p[0], point[1] - p[1]) };
}

/** Chord-length (alpha = 1) or centripetal (alpha = 0.5) parameters in [0, 1]. */
export function chordParams(points: Vec2[], alpha = 1): number[] {
  const n = points.length;
  const t = new Array<number>(n).fill(0);
  let total = 0;
  for (let i = 1; i < n; i++) {
    const d = Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]);
    total += Math.pow(d, alpha);
    t[i] = total;
  }
  if (total <= 0) return points.map((_, i) => (n > 1 ? i / (n - 1) : 0));
  for (let i = 0; i < n; i++) t[i] /= total;
  t[n - 1] = 1;
  return t;
}

function dedupe(points: Vec2[], eps: number): Vec2[] {
  const out: Vec2[] = [];
  for (const p of points) {
    const q = out[out.length - 1];
    if (!q || Math.hypot(p[0] - q[0], p[1] - q[1]) > eps) out.push([p[0], p[1]]);
  }
  return out;
}

function lineCurve(a: Vec2, b: Vec2): BSplineCurve2Def {
  return { degree: 1, knots: [0, 0, 1, 1], controlPoints: [[a[0], a[1]], [b[0], b[1]]] };
}

/**
 * Global interpolation through `points` (Piegl & Tiller A9.1, centripetal parameters, knot averaging).
 * With `closed`, the curve returns to the first point with matching first derivative (C1 seam);
 * closed curves are always cubic. Degenerate inputs fall back to lower degree / straight lines.
 */
export function interpolateBSpline2(points: Vec2[], opts: { degree?: number; closed?: boolean } = {}): BSplineCurve2Def {
  let scale = 0;
  for (const p of points) scale = Math.max(scale, Math.abs(p[0]), Math.abs(p[1]));
  const eps = 1e-12 * Math.max(1, scale);
  let pts = dedupe(points, eps);
  const closed = !!opts.closed;
  if (closed && pts.length > 1) {
    const f = pts[0], l = pts[pts.length - 1];
    if (Math.hypot(f[0] - l[0], f[1] - l[1]) <= eps) pts = pts.slice(0, -1);
  }
  if (pts.length === 0) return lineCurve([0, 0], [0, 0]);
  if (pts.length === 1) return lineCurve(pts[0], pts[0]);
  if (closed && pts.length >= 3) return interpolateClosed(pts);
  if (closed) pts = [...pts, pts[0]];
  const n = pts.length - 1;
  const p = Math.max(1, Math.min(opts.degree ?? 3, n));
  const u = chordParams(pts, 0.5);
  const U: number[] = [];
  for (let i = 0; i <= p; i++) U.push(0);
  for (let j = 1; j <= n - p; j++) {
    let s = 0;
    for (let i = j; i <= j + p - 1; i++) s += u[i];
    U.push(s / p);
  }
  for (let i = 0; i <= p; i++) U.push(1);
  const m = n + 1;
  const A = new Float64Array(m * m);
  for (let k = 0; k <= n; k++) {
    const span = findSpan(m, p, u[k], U);
    const N = basisFuns(span, u[k], p, U);
    for (let j = 0; j <= p; j++) A[k * m + span - p + j] = N[j];
  }
  const bx = pts.map((q) => q[0]);
  const by = pts.map((q) => q[1]);
  const X = solveLinear(A, bx, m);
  const Y = solveLinear(A, by, m);
  if (!X || !Y) {
    // fall back to a polyline-like degree-1 curve
    return { degree: 1, knots: [0, ...u, 1], controlPoints: pts.map((q) => [q[0], q[1]] as Vec2) };
  }
  const ctrl: Vec2[] = [];
  for (let i = 0; i < m; i++) ctrl.push([X[i], Y[i]]);
  return { degree: p, knots: U, controlPoints: ctrl };
}

/** Closed cubic interpolation with the seam derivative estimated by a central difference. */
function interpolateClosed(pts0: Vec2[]): BSplineCurve2Def {
  const pts = [...pts0, pts0[0]];
  const K = pts.length - 1; // pts[0..K], pts[K] == pts[0]
  const u = chordParams(pts, 0.5);
  const p = 3;
  const M = K + 3; // control points
  const U: number[] = [0, 0, 0, 0];
  for (let i = 1; i <= K - 1; i++) U.push(u[i]);
  U.push(1, 1, 1, 1);
  // derivative estimate at the seam
  const du = u[1] + (1 - u[K - 1]);
  const D: Vec2 = [(pts[1][0] - pts[K - 1][0]) / du, (pts[1][1] - pts[K - 1][1]) / du];
  const A = new Float64Array(M * M);
  const bx = new Float64Array(M);
  const by = new Float64Array(M);
  // row 0: P0 = Q0
  A[0] = 1;
  bx[0] = pts[0][0];
  by[0] = pts[0][1];
  // row 1: p/U[p+1] (P1 - P0) = D
  const f0 = p / (U[p + 1] - U[1]);
  A[1 * M + 0] = -f0;
  A[1 * M + 1] = f0;
  bx[1] = D[0];
  by[1] = D[1];
  // rows 2..K: interior interpolation
  for (let k = 1; k <= K - 1; k++) {
    const row = k + 1;
    const span = findSpan(M, p, u[k], U);
    const N = basisFuns(span, u[k], p, U);
    for (let j = 0; j <= p; j++) A[row * M + span - p + j] = N[j];
    bx[row] = pts[k][0];
    by[row] = pts[k][1];
  }
  // row K+1: derivative at end
  const f1 = p / (U[M - 1 + p] - U[M - 1]);
  A[(K + 1) * M + M - 2] = -f1;
  A[(K + 1) * M + M - 1] = f1;
  bx[K + 1] = D[0];
  by[K + 1] = D[1];
  // row K+2: P_last = Q_K
  A[(K + 2) * M + M - 1] = 1;
  bx[K + 2] = pts[K][0];
  by[K + 2] = pts[K][1];
  const X = solveLinear(A, bx, M);
  const Y = solveLinear(A, by, M);
  if (!X || !Y) return { degree: 1, knots: [0, ...u, 1], controlPoints: pts.map((q) => [q[0], q[1]] as Vec2) };
  const ctrl: Vec2[] = [];
  for (let i = 0; i < M; i++) ctrl.push([X[i], Y[i]]);
  return { degree: p, knots: U, controlPoints: ctrl };
}

/**
 * Least-squares approximation with the end points interpolated (Piegl & Tiller 9.4.1) for given
 * data parameters. nCtrl >= degree + 1. Returns null for degenerate systems.
 */
export function approximateBSpline2(points: Vec2[], params: number[], nCtrl: number, degree: number): BSplineCurve2Def | null {
  const m = points.length - 1;
  const n = nCtrl - 1;
  const p = Math.max(1, Math.min(degree, n));
  if (m < 1 || n < 1) return null;
  const U: number[] = [];
  for (let i = 0; i <= p; i++) U.push(0);
  const d = (m + 1) / (n - p + 1);
  for (let j = 1; j <= n - p; j++) {
    const i = Math.floor(j * d);
    const alpha = j * d - i;
    const a = params[Math.max(0, Math.min(m, i - 1))];
    const b = params[Math.max(0, Math.min(m, i))];
    U.push((1 - alpha) * a + alpha * b);
  }
  for (let i = 0; i <= p; i++) U.push(1);
  // enforce strictly non-decreasing interior knots with a minimal gap
  for (let i = p + 1; i <= n; i++) if (U[i] < U[i - 1]) U[i] = U[i - 1];
  const Q0 = points[0], Qm = points[m];
  if (n === 1) return { degree: 1, knots: [0, 0, 1, 1], controlPoints: [[Q0[0], Q0[1]], [Qm[0], Qm[1]]] };
  const nu = n - 1; // unknowns P1..P_{n-1}
  const NtN = new Float64Array(nu * nu);
  const Rx = new Float64Array(nu);
  const Ry = new Float64Array(nu);
  for (let k = 1; k < m; k++) {
    const span = findSpan(n + 1, p, params[k], U);
    const N = basisFuns(span, params[k], p, U);
    // full row of basis values (only p+1 nonzero)
    let n0 = 0, nn = 0;
    for (let j = 0; j <= p; j++) {
      const i = span - p + j;
      if (i === 0) n0 = N[j];
      if (i === n) nn = N[j];
    }
    const rx = points[k][0] - n0 * Q0[0] - nn * Qm[0];
    const ry = points[k][1] - n0 * Q0[1] - nn * Qm[1];
    for (let a = 0; a <= p; a++) {
      const ia = span - p + a;
      if (ia < 1 || ia > n - 1) continue;
      Rx[ia - 1] += N[a] * rx;
      Ry[ia - 1] += N[a] * ry;
      for (let b = 0; b <= p; b++) {
        const ib = span - p + b;
        if (ib < 1 || ib > n - 1) continue;
        NtN[(ia - 1) * nu + ib - 1] += N[a] * N[b];
      }
    }
  }
  // tiny ridge keeps empty knot spans solvable
  let tr = 0;
  for (let i = 0; i < nu; i++) tr += NtN[i * nu + i];
  const ridge = 1e-10 * (tr / Math.max(1, nu) || 1);
  for (let i = 0; i < nu; i++) NtN[i * nu + i] += ridge;
  const X = solveLinear(NtN, Rx, nu);
  const Y = solveLinear(NtN, Ry, nu);
  if (!X || !Y) return null;
  const ctrl: Vec2[] = [[Q0[0], Q0[1]]];
  for (let i = 0; i < nu; i++) ctrl.push([X[i], Y[i]]);
  ctrl.push([Qm[0], Qm[1]]);
  return { degree: p, knots: U, controlPoints: ctrl };
}

/** Insert knot u once (Boehm). Non-rational curves only (weights are carried by homogeneous coordinates). */
export function insertKnot2(c: BSplineCurve2Def, u: number): BSplineCurve2Def {
  const p = c.degree;
  const U = c.knots;
  const n = c.controlPoints.length;
  const k = findSpan(n, p, u, U);
  const w = c.weights;
  const P = c.controlPoints.map((q, i) => {
    const wt = w ? w[i] : 1;
    return [q[0] * wt, q[1] * wt, wt];
  });
  const Q: number[][] = [];
  for (let i = 0; i <= n; i++) {
    if (i <= k - p) Q.push(P[i].slice());
    else if (i > k) Q.push(P[i - 1].slice());
    else {
      const den = U[i + p] - U[i];
      const a = den !== 0 ? (u - U[i]) / den : 0;
      Q.push([0, 1, 2].map((d) => (1 - a) * P[i - 1][d] + a * P[i][d]));
    }
  }
  const knots = [...U.slice(0, k + 1), u, ...U.slice(k + 1)];
  const controlPoints = Q.map((q) => [q[0] / q[2], q[1] / q[2]] as Vec2);
  const out: BSplineCurve2Def = { degree: p, knots, controlPoints };
  if (w) out.weights = Q.map((q) => q[2]);
  return out;
}

/** Split a clamped curve at parameter u (strictly inside the domain) into two clamped curves. */
export function splitBSpline2(c: BSplineCurve2Def, u: number): [BSplineCurve2Def, BSplineCurve2Def] {
  const p = c.degree;
  let cur = c;
  const mult = () => cur.knots.filter((k) => Math.abs(k - u) <= 1e-14).length;
  // snap knots that are numerically equal to u
  let guard = 0;
  while (mult() < p + 1 && guard++ < p + 2) cur = insertKnot2(cur, u);
  const U = cur.knots;
  // first index of u in the knot vector
  let first = U.findIndex((k) => Math.abs(k - u) <= 1e-14);
  const r = mult();
  // left: knots U[0..first+r-1], control points 0..first-1
  const leftKnots = U.slice(0, first + r);
  const nLeft = leftKnots.length - p - 1;
  const rightKnots = U.slice(first);
  const nRight = rightKnots.length - p - 1;
  const cp = cur.controlPoints;
  const left: BSplineCurve2Def = { degree: p, knots: leftKnots, controlPoints: cp.slice(0, nLeft) };
  const right: BSplineCurve2Def = { degree: p, knots: rightKnots, controlPoints: cp.slice(cp.length - nRight) };
  if (cur.weights) {
    left.weights = cur.weights.slice(0, nLeft);
    right.weights = cur.weights.slice(cp.length - nRight);
  }
  return [left, right];
}

/** Reverse the parameter direction of a curve. */
export function reverseBSpline2(c: BSplineCurve2Def): BSplineCurve2Def {
  const a = c.knots[0], b = c.knots[c.knots.length - 1];
  const out: BSplineCurve2Def = {
    degree: c.degree,
    knots: c.knots.map((k) => a + b - k).reverse(),
    controlPoints: c.controlPoints.map((q) => [q[0], q[1]] as Vec2).reverse(),
  };
  if (c.weights) out.weights = c.weights.slice().reverse();
  return out;
}
