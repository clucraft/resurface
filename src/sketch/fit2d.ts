/**
 * 2D fitting for section polylines: lines, circles, arcs, B-splines, and automatic segmentation of a
 * noisy section into a minimal chain of lines and arcs (`autoSegment`) plus conversion to a
 * constrained sketch (`autoSketch`).
 */
import type { BSplineCurve2Def, FitStats, Segment2D, Vec2 } from "../core/types";
import { approximateBSpline2, bsplineDomain, chordParams, derivBSpline2, evalBSpline2 } from "./bspline";
import {
  TAU,
  angleOf,
  circleCircleIntersection,
  closestPointOnSegment,
  lineCircleIntersection,
  lineLineIntersection,
  normAngle,
  polar,
  wrapPi,
} from "./geom2d";
import { addArc, addCircle, addConstraint, addLine, addPoint, createSketch, type Sketch } from "./model";
import { solveSketch } from "./solver";

// ---------------------------------------------------------------------------
// Stats helpers
// ---------------------------------------------------------------------------

export function fitStats(residuals: ArrayLike<number>): FitStats {
  const n = residuals.length;
  let s = 0, s2 = 0, mx = 0;
  for (let i = 0; i < n; i++) {
    const r = residuals[i];
    s += r;
    s2 += r * r;
    mx = Math.max(mx, Math.abs(r));
  }
  return { rms: n ? Math.sqrt(s2 / n) : 0, maxAbs: mx, mean: n ? s / n : 0, count: n };
}

// ---------------------------------------------------------------------------
// Line
// ---------------------------------------------------------------------------

export interface LineFit2D {
  /** projection of the first data extreme onto the line */
  a: Vec2;
  /** projection of the last data extreme onto the line */
  b: Vec2;
  /** centroid (a point on the line) */
  point: Vec2;
  /** unit direction, oriented from the first towards the last data point */
  direction: Vec2;
  stats: FitStats;
}

/** Orthogonal (total least squares) line fit. a..b spans the extent of the data along the line. */
export function fitLine2D(points: Vec2[]): LineFit2D {
  const n = points.length;
  if (n === 0) return { a: [0, 0], b: [0, 0], point: [0, 0], direction: [1, 0], stats: fitStats([]) };
  let mx = 0, my = 0;
  for (const p of points) {
    mx += p[0];
    my += p[1];
  }
  mx /= n;
  my /= n;
  let sxx = 0, sxy = 0, syy = 0;
  for (const p of points) {
    const dx = p[0] - mx, dy = p[1] - my;
    sxx += dx * dx;
    sxy += dx * dy;
    syy += dy * dy;
  }
  const th = 0.5 * Math.atan2(2 * sxy, sxx - syy);
  let d: Vec2 = [Math.cos(th), Math.sin(th)];
  if (sxx + syy < 1e-300) {
    const f = points[0], l = points[n - 1];
    const L = Math.hypot(l[0] - f[0], l[1] - f[1]);
    d = L > 0 ? [(l[0] - f[0]) / L, (l[1] - f[1]) / L] : [1, 0];
  }
  const f = points[0], l = points[n - 1];
  if ((l[0] - f[0]) * d[0] + (l[1] - f[1]) * d[1] < 0) d = [-d[0], -d[1]];
  const res = new Float64Array(n);
  let tmin = Infinity, tmax = -Infinity;
  for (let i = 0; i < n; i++) {
    const dx = points[i][0] - mx, dy = points[i][1] - my;
    const t = dx * d[0] + dy * d[1];
    tmin = Math.min(tmin, t);
    tmax = Math.max(tmax, t);
    res[i] = d[0] * dy - d[1] * dx;
  }
  return {
    a: [mx + d[0] * tmin, my + d[1] * tmin],
    b: [mx + d[0] * tmax, my + d[1] * tmax],
    point: [mx, my],
    direction: d,
    stats: fitStats(res),
  };
}

// ---------------------------------------------------------------------------
// Circle / arc
// ---------------------------------------------------------------------------

export interface CircleFit2D {
  center: Vec2;
  radius: number;
  stats: FitStats;
}

/** Taubin algebraic circle fit (Chernov's Newton formulation). */
function circleTaubin(points: Vec2[]): { c: Vec2; r: number } | null {
  const n = points.length;
  if (n < 3) return null;
  let xm = 0, ym = 0;
  for (const p of points) {
    xm += p[0];
    ym += p[1];
  }
  xm /= n;
  ym /= n;
  let Mxx = 0, Myy = 0, Mxy = 0, Mxz = 0, Myz = 0, Mzz = 0;
  for (const p of points) {
    const Xi = p[0] - xm, Yi = p[1] - ym;
    const Zi = Xi * Xi + Yi * Yi;
    Mxy += Xi * Yi;
    Mxx += Xi * Xi;
    Myy += Yi * Yi;
    Mxz += Xi * Zi;
    Myz += Yi * Zi;
    Mzz += Zi * Zi;
  }
  Mxx /= n;
  Myy /= n;
  Mxy /= n;
  Mxz /= n;
  Myz /= n;
  Mzz /= n;
  const Mz = Mxx + Myy;
  const Cov_xy = Mxx * Myy - Mxy * Mxy;
  const Var_z = Mzz - Mz * Mz;
  const A3 = 4 * Mz;
  const A2 = -3 * Mz * Mz - Mzz;
  const A1 = Var_z * Mz + 4 * Cov_xy * Mz - Mxz * Mxz - Myz * Myz;
  const A0 = Mxz * (Mxz * Myy - Myz * Mxy) + Myz * (Myz * Mxx - Mxz * Mxy) - Var_z * Cov_xy;
  const A22 = A2 + A2, A33 = A3 + A3 + A3;
  let x = 0, y = A0;
  for (let it = 0; it < 99; it++) {
    const Dy = A1 + x * (A22 + A33 * x);
    const xnew = x - y / Dy;
    if (xnew === x || !isFinite(xnew)) break;
    const ynew = A0 + xnew * (A1 + xnew * (A2 + xnew * A3));
    if (Math.abs(ynew) >= Math.abs(y)) break;
    x = xnew;
    y = ynew;
  }
  const DET = x * x - x * Mz + Cov_xy;
  if (!(Math.abs(DET) > 1e-300)) return null;
  const Xc = (Mxz * (Myy - x) - Myz * Mxy) / DET / 2;
  const Yc = (Myz * (Mxx - x) - Mxz * Mxy) / DET / 2;
  const r = Math.sqrt(Xc * Xc + Yc * Yc + Mz);
  if (!isFinite(r) || !isFinite(Xc) || !isFinite(Yc)) return null;
  return { c: [Xc + xm, Yc + ym], r };
}

/** Geometric (orthogonal distance) circle refinement with Levenberg–Marquardt. */
function circleLM(points: Vec2[], c0: Vec2, r0: number, iters = 30): { c: Vec2; r: number } {
  let cx = c0[0], cy = c0[1], r = r0;
  const cost = (x: number, y: number, rr: number) => {
    let s = 0;
    for (const p of points) {
      const d = Math.hypot(p[0] - x, p[1] - y) - rr;
      s += d * d;
    }
    return s;
  };
  let f = cost(cx, cy, r);
  let lambda = 1e-3;
  for (let it = 0; it < iters; it++) {
    const A = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    const g = [0, 0, 0];
    for (const p of points) {
      const dx = p[0] - cx, dy = p[1] - cy;
      const L = Math.hypot(dx, dy) || 1e-300;
      const res = L - r;
      const j = [-dx / L, -dy / L, -1];
      for (let a = 0; a < 3; a++) {
        g[a] += j[a] * res;
        for (let b = 0; b < 3; b++) A[a * 3 + b] += j[a] * j[b];
      }
    }
    let improved = false;
    for (let tries = 0; tries < 8; tries++) {
      const M = A.slice();
      for (let a = 0; a < 3; a++) M[a * 4] *= 1 + lambda;
      const d = solve3(M, [-g[0], -g[1], -g[2]]);
      if (!d) {
        lambda *= 10;
        continue;
      }
      const nx = cx + d[0], ny = cy + d[1], nr = r + d[2];
      const nf = cost(nx, ny, nr);
      if (nf < f) {
        const rel = (f - nf) / Math.max(f, 1e-300);
        cx = nx;
        cy = ny;
        r = nr;
        f = nf;
        lambda = Math.max(1e-12, lambda / 10);
        improved = true;
        if (rel < 1e-14) it = iters;
        break;
      }
      lambda *= 10;
    }
    if (!improved) break;
  }
  return { c: [cx, cy], r: Math.abs(r) };
}

function solve3(M: number[], b: number[]): number[] | null {
  const [a, bb, c, d, e, f, g, h, i] = M;
  const det = a * (e * i - f * h) - bb * (d * i - f * g) + c * (d * h - e * g);
  if (!(Math.abs(det) > 1e-300)) return null;
  const x = (b[0] * (e * i - f * h) - bb * (b[1] * i - f * b[2]) + c * (b[1] * h - e * b[2])) / det;
  const y = (a * (b[1] * i - f * b[2]) - b[0] * (d * i - f * g) + c * (d * b[2] - b[1] * g)) / det;
  const z = (a * (e * b[2] - b[1] * h) - bb * (d * b[2] - b[1] * g) + b[0] * (d * h - e * g)) / det;
  return [x, y, z];
}

function extentOf(points: Vec2[]): number {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const p of points) {
    x0 = Math.min(x0, p[0]);
    y0 = Math.min(y0, p[1]);
    x1 = Math.max(x1, p[0]);
    y1 = Math.max(y1, p[1]);
  }
  return Math.hypot(x1 - x0, y1 - y0);
}

function circleCore(points: Vec2[]): { c: Vec2; r: number } | null {
  const alg = circleTaubin(points);
  if (!alg) return null;
  const ext = extentOf(points);
  if (!(ext > 0) || alg.r > 1e6 * ext) return null;
  return circleLM(points, alg.c, alg.r);
}

/** Circle fit: Taubin algebraic estimate refined by geometric Levenberg–Marquardt. Null for < 3 or collinear points. */
export function fitCircle2D(points: Vec2[]): CircleFit2D | null {
  const c = circleCore(points);
  if (!c) return null;
  const res = points.map((p) => Math.hypot(p[0] - c.c[0], p[1] - c.c[1]) - c.r);
  return { center: c.c, radius: c.r, stats: fitStats(res) };
}

export interface ArcFit2D {
  /** first data point projected onto the circle */
  start: Vec2;
  /** circle point at the angular middle of the swept range */
  mid: Vec2;
  /** last data point projected onto the circle */
  end: Vec2;
  center: Vec2;
  radius: number;
  /** signed sweep from start to end following the data (> 0 = CCW) */
  sweep: number;
  stats: FitStats;
}

/** Signed angular progression of points around c (sum of wrapped increments). */
function angularSweep(points: Vec2[], c: Vec2): number {
  let total = 0;
  let prev = angleOf([points[0][0] - c[0], points[0][1] - c[1]]);
  for (let i = 1; i < points.length; i++) {
    const a = angleOf([points[i][0] - c[0], points[i][1] - c[1]]);
    total += wrapPi(a - prev);
    prev = a;
  }
  return total;
}

function arcFromCircle(points: Vec2[], c: Vec2, r: number): Omit<ArcFit2D, "stats"> {
  const a0 = angleOf([points[0][0] - c[0], points[0][1] - c[1]]);
  const sweep = angularSweep(points, c);
  return { start: polar(c, r, a0), mid: polar(c, r, a0 + sweep / 2), end: polar(c, r, a0 + sweep), center: c, radius: r, sweep };
}

/** Arc fit: circle fit, with start/end at the first/last data points projected on the circle. */
export function fitArc2D(points: Vec2[]): ArcFit2D | null {
  const c = circleCore(points);
  if (!c) return null;
  const res = points.map((p) => Math.hypot(p[0] - c.c[0], p[1] - c.c[1]) - c.r);
  return { ...arcFromCircle(points, c.c, c.r), stats: fitStats(res) };
}

// ---------------------------------------------------------------------------
// B-spline
// ---------------------------------------------------------------------------

export interface BSplineFit2D {
  curve: BSplineCurve2Def;
  /** parameter of each data point after parameter correction */
  params: number[];
  stats: FitStats;
}

/**
 * Least-squares B-spline approximation with end points interpolated and a few rounds of parameter
 * correction (projection of the data onto the current curve).
 */
export function fitBSpline2D(points: Vec2[], nCtrl: number, degree = 3, opts: { iterations?: number } = {}): BSplineFit2D | null {
  const pts: Vec2[] = [];
  for (const p of points) {
    const q = pts[pts.length - 1];
    if (!q || Math.hypot(p[0] - q[0], p[1] - q[1]) > 1e-12) pts.push(p);
  }
  if (pts.length < 2) return null;
  const nc = Math.max(Math.min(degree, pts.length - 1) + 1, Math.min(nCtrl, pts.length));
  const params = chordParams(pts, 1);
  let curve = approximateBSpline2(pts, params, nc, degree);
  if (!curve) return null;
  const iters = opts.iterations ?? 4;
  for (let it = 0; it < iters; it++) {
    for (let k = 1; k < pts.length - 1; k++) {
      let u = params[k];
      for (let s = 0; s < 3; s++) {
        const q = evalBSpline2(curve, u);
        const d1 = derivBSpline2(curve, u);
        const den = d1[0] * d1[0] + d1[1] * d1[1];
        if (den < 1e-300) break;
        u += ((pts[k][0] - q[0]) * d1[0] + (pts[k][1] - q[1]) * d1[1]) / den;
        u = Math.min(1, Math.max(0, u));
      }
      params[k] = u;
    }
    // keep parameters monotone for a well-posed knot placement
    for (let k = 1; k < pts.length; k++) if (params[k] < params[k - 1]) params[k] = params[k - 1];
    const next = approximateBSpline2(pts, params, nc, degree);
    if (!next) break;
    curve = next;
  }
  const res = pts.map((p, k) => {
    const q = evalBSpline2(curve!, params[k]);
    const d = derivBSpline2(curve!, params[k]);
    const s = d[0] * (p[1] - q[1]) - d[1] * (p[0] - q[0]);
    return Math.sign(s || 1) * Math.hypot(p[0] - q[0], p[1] - q[1]);
  });
  return { curve, params, stats: fitStats(res) };
}

// ---------------------------------------------------------------------------
// Auto segmentation
// ---------------------------------------------------------------------------

export interface AutoSegmentOptions {
  /** maximum allowed deviation of the data from the segments (mm) */
  tolerance: number;
  /** treat the polyline as closed (default false) */
  closed?: boolean;
  /** minimum number of data points per segment (default 4) */
  minSegmentPoints?: number;
  /** prefer an arc over a line when both fit but the arc is clearly better (default true) */
  preferArcs?: boolean;
  /** allow arcs at all (default true; false gives a polyline approximation) */
  allowArcs?: boolean;
  /** turning angle that seeds a corner (default 30 deg) */
  cornerAngleDeg?: number;
  /** make nearly tangent junctions exactly tangent (default true) */
  snapTangent?: boolean;
  /** a closed polyline that fits one circle becomes a full circle segment (default true; false gives 2 arcs) */
  fullCircle?: boolean;
}

export interface AutoSegmentDetail {
  segments: Segment2D[];
  /** tangent[k]: junction between segment k and k+1 (closed: the last entry is last->first) is tangent */
  tangent: boolean[];
  /** inclusive data index range [start, end] per segment (closed ranges may wrap: end < start) */
  ranges: [number, number][];
  /** arcs: +1 if the data runs CCW, -1 if CW; lines/circles: 0 */
  arcDirection: number[];
  /** max distance of any data point from its segment */
  maxError: number;
  closed: boolean;
}

type Prim = { type: "line"; p: Vec2; d: Vec2 } | { type: "arc"; c: Vec2; r: number; dir: number };

interface Piece {
  /** unwrapped inclusive index range (closed data: may exceed N-1, use mod) */
  s: number;
  e: number;
  prim: Prim;
  err: number;
  ok: boolean;
}

const DEG = Math.PI / 180;

class Segmenter {
  readonly N: number;
  readonly P: Vec2[];
  readonly closed: boolean;
  readonly tol: number;
  readonly minPts: number;
  readonly preferArcs: boolean;
  readonly allowArcs: boolean;
  readonly extent: number;
  private cache = new Map<string, Piece>();

  constructor(P: Vec2[], opts: AutoSegmentOptions) {
    this.P = P;
    this.N = P.length;
    this.closed = !!opts.closed;
    this.tol = Math.max(opts.tolerance, 1e-12);
    this.minPts = Math.max(2, Math.floor(opts.minSegmentPoints ?? 4));
    this.preferArcs = opts.preferArcs ?? true;
    this.allowArcs = opts.allowArcs ?? true;
    this.extent = extentOf(P);
  }

  pt(i: number): Vec2 {
    const N = this.N;
    return this.P[((i % N) + N) % N];
  }

  points(s: number, e: number): Vec2[] {
    const out: Vec2[] = [];
    for (let i = s; i <= e; i++) out.push(this.pt(i));
    return out;
  }

  /** Best primitive for the inclusive range. */
  fit(s: number, e: number): Piece {
    const key = `${s}:${e}`;
    const hit = this.cache.get(key);
    if (hit) return hit;
    const pts = this.points(s, e);
    const lf = fitLine2D(pts);
    const line: Piece = { s, e, prim: { type: "line", p: lf.point, d: lf.direction }, err: lf.stats.maxAbs, ok: lf.stats.maxAbs <= this.tol };
    let best = line;
    if (this.allowArcs && pts.length >= 3 && (!line.ok || this.preferArcs)) {
      const c = circleCore(pts);
      if (c && this.validArc(pts, c.c, c.r)) {
        let err = 0, s2 = 0;
        for (const p of pts) {
          const d = Math.abs(Math.hypot(p[0] - c.c[0], p[1] - c.c[1]) - c.r);
          err = Math.max(err, d);
          s2 += d * d;
        }
        const sweep = angularSweep(pts, c.c);
        const arc: Piece = { s, e, prim: { type: "arc", c: c.c, r: c.r, dir: sweep >= 0 ? 1 : -1 }, err, ok: err <= this.tol };
        if (!line.ok) {
          if (arc.ok || arc.err < line.err) best = arc;
        } else if (arc.ok) {
          // both fit: take the arc only when it is clearly better and really curved
          const lineRms = lf.stats.rms;
          const arcRms = Math.sqrt(s2 / pts.length);
          if (Math.abs(sweep) >= 5 * DEG && arcRms < 0.5 * lineRms && line.err > 0.5 * this.tol) best = arc;
        }
      }
    }
    this.cache.set(key, best);
    return best;
  }

  /** Arc plausibility: bounded radius, monotone angular progression, sweep below a full turn. */
  validArc(pts: Vec2[], c: Vec2, r: number): boolean {
    if (!(r > 0) || r > 1e4 * Math.max(this.extent, this.tol)) return false;
    const sweep = angularSweep(pts, c);
    if (Math.abs(sweep) > TAU * 0.995) return false;
    const n = pts.length;
    const K = Math.min(n - 1, 12);
    const dir = sweep >= 0 ? 1 : -1;
    const slack = (3 * this.tol) / r;
    let prev = angleOf([pts[0][0] - c[0], pts[0][1] - c[1]]);
    for (let k = 1; k <= K; k++) {
      const i = Math.round((k * (n - 1)) / K);
      const a = angleOf([pts[i][0] - c[0], pts[i][1] - c[1]]);
      if (wrapPi(a - prev) * dir < -slack) return false;
      prev = a;
    }
    return true;
  }

  dist(prim: Prim, p: Vec2): number {
    if (prim.type === "line") return Math.abs(prim.d[0] * (p[1] - prim.p[1]) - prim.d[1] * (p[0] - prim.p[0]));
    return Math.abs(Math.hypot(p[0] - prim.c[0], p[1] - prim.c[1]) - prim.r);
  }

  /** Douglas–Peucker style split index inside (s, e). */
  splitIndex(s: number, e: number): number {
    const a = this.pt(s), b = this.pt(e);
    const dx = b[0] - a[0], dy = b[1] - a[1];
    const L = Math.hypot(dx, dy);
    let best = -1, bestD = -1;
    const lo = s + this.minPts - 1, hi = e - this.minPts + 1;
    for (let i = s + 1; i < e; i++) {
      const p = this.pt(i);
      const d = L > 1e-9 * Math.max(1, this.extent) ? Math.abs(dx * (p[1] - a[1]) - dy * (p[0] - a[0])) / L : Math.hypot(p[0] - a[0], p[1] - a[1]);
      if (d > bestD) {
        bestD = d;
        best = i;
      }
    }
    if (lo > hi) return Math.round((s + e) / 2);
    return Math.min(hi, Math.max(lo, best));
  }

  splitRec(s: number, e: number, out: Piece[], depth = 0): void {
    const f = this.fit(s, e);
    const n = e - s + 1;
    if (f.ok || n < 2 * this.minPts - 1 || depth > 60) {
      out.push(f);
      return;
    }
    const k = this.splitIndex(s, e);
    this.splitRec(s, k, out, depth + 1);
    this.splitRec(k, e, out, depth + 1);
  }

  /** Greedy merge of adjacent pieces while the merged fit stays within tolerance. */
  merge(pieces: Piece[]): Piece[] {
    let ps = pieces.slice();
    for (;;) {
      let bestK = -1;
      let best: Piece | null = null;
      const K = ps.length;
      const pairs = this.closed ? K : K - 1;
      if (K < 2) break;
      for (let k = 0; k < pairs; k++) {
        const A = ps[k];
        const B = ps[(k + 1) % K];
        const s = A.s;
        const e = k + 1 < K ? B.e : B.e + this.N;
        if (this.closed && e - s >= this.N) {
          // merging would cover the whole loop: only allowed when K == 2 (full circle handled elsewhere)
          continue;
        }
        const f = this.fit(s, e);
        if (!f.ok) continue;
        // prefer lines, then lower error relative to tolerance
        const score = f.err / this.tol + (f.prim.type === "line" ? 0 : 0.25);
        const bestScore = best ? best.err / this.tol + (best.prim.type === "line" ? 0 : 0.25) : Infinity;
        if (score < bestScore) {
          best = f;
          bestK = k;
        }
      }
      if (!best) break;
      if (bestK + 1 < K) ps.splice(bestK, 2, best);
      else {
        // wrap pair (last, first)
        ps = ps.slice(1, K - 1);
        ps.push(best);
      }
      ps = this.normalize(ps);
    }
    return ps;
  }

  /** Keep the first piece starting inside [0, N) for closed data. */
  normalize(ps: Piece[]): Piece[] {
    if (!this.closed || !ps.length) return ps;
    const N = this.N;
    const shift = Math.floor(ps[0].s / N) * N;
    if (shift === 0) return ps;
    return ps.map((p) => this.fit(p.s - shift, p.e - shift));
  }

  /**
   * Remove small pieces by letting their two neighbours absorb them (with the best split point inside
   * the removed range) when both neighbours still fit within tolerance.
   */
  eliminate(pieces: Piece[]): Piece[] {
    let ps = pieces.slice();
    const N = this.N;
    const small = Math.max(4 * this.minPts, 8);
    for (let guard = 0; guard < 200; guard++) {
      const K = ps.length;
      if (K < (this.closed ? 3 : 2)) break;
      const order = ps.map((_, k) => k).sort((a, b) => ps[a].e - ps[a].s - (ps[b].e - ps[b].s));
      let done = false;
      for (const k of order) {
        const pc = ps[k];
        if (pc.e - pc.s + 1 > small) break;
        const hasPrev = this.closed || k > 0;
        const hasNext = this.closed || k < K - 1;
        if (!hasPrev || !hasNext) continue; // end pieces: plain merging handles them
        const kp = (k - 1 + K) % K, kn = (k + 1) % K;
        const A = ps[kp], B = ps[kn];
        // unwrapped coordinates relative to the removed piece
        const As = kp < k ? A.s : A.s - N;
        const Be = kn > k ? B.e : B.e + N;
        let best: [Piece, Piece] | null = null, bestErr = Infinity;
        for (let b = pc.s; b <= pc.e; b++) {
          if (b - As + 1 < this.minPts || Be - b + 1 < this.minPts) continue;
          const nA = this.fit(As, b), nB = this.fit(b, Be);
          if (!nA.ok || !nB.ok) continue;
          const err = Math.max(nA.err, nB.err);
          if (err < bestErr) {
            bestErr = err;
            best = [nA, nB];
          }
        }
        if (!best) continue;
        // rebuild the cyclic/open list without piece k
        const next: Piece[] = [];
        for (let i = 0; i < K; i++) {
          if (i === k) continue;
          if (i === kp) next.push(best[0]);
          else if (i === kn) next.push(best[1]);
          else next.push(ps[i]);
        }
        // order: for closed data starting with the wrap, re-sort by unwrapped start after normalising
        ps = this.closed ? this.reorder(next) : next;
        done = true;
        break;
      }
      if (!done) break;
    }
    return ps;
  }

  /** Re-establish cyclic order (pieces consecutive, first piece starting in [0, N)) after edits. */
  reorder(ps: Piece[]): Piece[] {
    const N = this.N;
    const norm = ps.map((p) => {
      const shift = Math.floor(p.s / N) * N;
      return shift ? this.fit(p.s - shift, p.e - shift) : p;
    });
    norm.sort((a, b) => a.s - b.s);
    return norm;
  }

  /** Move junctions to minimise the summed squared distances to the two adjacent primitives. */
  refine(pieces: Piece[], rounds = 3): Piece[] {
    let ps = pieces.slice();
    const N = this.N;
    for (let round = 0; round < rounds; round++) {
      let changed = false;
      const K = ps.length;
      const pairs = this.closed ? (K >= 2 ? K : 0) : K - 1;
      for (let k = 0; k < pairs; k++) {
        const A = ps[k];
        const wrap = k + 1 >= K;
        const B0 = ps[(k + 1) % K];
        const Be = wrap ? B0.e + N : B0.e;
        const lo = A.s + this.minPts - 1, hi = Be - this.minPts + 1;
        if (lo > hi) continue;
        let acc = 0;
        // cost(b) = sum_{A.s..b} dA^2 + sum_{b..Be} dB^2
        const dA: number[] = [], dB: number[] = [];
        for (let i = A.s; i <= Be; i++) {
          const p = this.pt(i);
          dA.push(this.dist(A.prim, p) ** 2);
          dB.push(this.dist(B0.prim, p) ** 2);
        }
        const suffix = new Array<number>(dB.length + 1).fill(0);
        for (let i = dB.length - 1; i >= 0; i--) suffix[i] = suffix[i + 1] + dB[i];
        let bestB = A.e, bestC = Infinity;
        for (let i = 0; i < dA.length; i++) {
          acc += dA[i];
          const b = A.s + i;
          if (b < lo || b > hi) continue;
          const c = acc + suffix[i];
          if (c < bestC - 1e-15) {
            bestC = c;
            bestB = b;
          }
        }
        if (bestB === A.e) continue;
        const nA = this.fit(A.s, bestB);
        const nB = this.fit(bestB, Be);
        // accept only if both still fit (or the worse error does not grow)
        const oldWorst = Math.max(A.err, B0.err);
        const newWorst = Math.max(nA.err, nB.err);
        if ((nA.ok && nB.ok) || newWorst < oldWorst) {
          ps[k] = nA;
          ps[(k + 1) % K] = wrap ? this.fit(bestB - N, Be - N) : nB;
          changed = true;
        }
      }
      ps = this.normalize(ps);
      if (!changed) break;
    }
    return ps;
  }
}

/** Turning-angle corner candidates (indices), strongest first. */
function detectCorners(P: Vec2[], closed: boolean, tol: number, angleDeg: number): number[] {
  const N = P.length;
  if (N < 5) return [];
  // arc-length window: a few tolerances, at least ~2 sample spacings
  const S = [0];
  for (let i = 1; i < N; i++) S.push(S[i - 1] + Math.hypot(P[i][0] - P[i - 1][0], P[i][1] - P[i - 1][1]));
  const total = closed ? S[N - 1] + Math.hypot(P[0][0] - P[N - 1][0], P[0][1] - P[N - 1][1]) : S[N - 1];
  const spacing = total / Math.max(1, closed ? N : N - 1);
  const w = Math.max(5 * tol, 2.5 * spacing);
  const turn = new Float64Array(N);
  const at = (i: number) => P[((i % N) + N) % N];
  const sAt = (i: number) => {
    // unwrapped arc length
    const k = Math.floor(i / N);
    return S[((i % N) + N) % N] + k * total;
  };
  for (let i = 0; i < N; i++) {
    let lo = i, hi = i;
    while (sAt(i) - sAt(lo - 1) <= w && (closed ? i - lo < N / 3 : lo > 0)) lo--;
    while (sAt(hi + 1) - sAt(i) <= w && (closed ? hi - i < N / 3 : hi < N - 1)) hi++;
    lo = Math.min(lo, i - 1);
    hi = Math.max(hi, i + 1);
    if (!closed && (lo < 0 || hi > N - 1)) continue;
    const p = at(i), a = at(lo), b = at(hi);
    const d1: Vec2 = [p[0] - a[0], p[1] - a[1]], d2: Vec2 = [b[0] - p[0], b[1] - p[1]];
    const l1 = Math.hypot(...d1), l2 = Math.hypot(...d2);
    if (l1 < 1e-12 || l2 < 1e-12) continue;
    turn[i] = Math.abs(Math.atan2(d1[0] * d2[1] - d1[1] * d2[0], d1[0] * d2[0] + d1[1] * d2[1]));
  }
  const thr = angleDeg * DEG;
  const cands: number[] = [];
  const halfW = Math.max(1, Math.round(w / Math.max(spacing, 1e-12)));
  for (let i = 0; i < N; i++) {
    if (turn[i] < thr) continue;
    let isMax = true;
    for (let k = -halfW; k <= halfW && isMax; k++) {
      if (k === 0) continue;
      const j = i + k;
      if (!closed && (j < 0 || j >= N)) continue;
      const tj = turn[((j % N) + N) % N];
      if (tj > turn[i] || (tj === turn[i] && k < 0)) isMax = false;
    }
    if (isMax) cands.push(i);
  }
  return cands.sort((a, b) => turn[b] - turn[a]);
}

function cleanInput(points: Vec2[], closed: boolean): Vec2[] {
  let scale = 0;
  for (const p of points) scale = Math.max(scale, Math.abs(p[0]), Math.abs(p[1]));
  const eps = 1e-12 * Math.max(1, scale);
  const out: Vec2[] = [];
  for (const p of points) {
    if (!isFinite(p[0]) || !isFinite(p[1])) continue;
    const q = out[out.length - 1];
    if (!q || Math.hypot(p[0] - q[0], p[1] - q[1]) > eps) out.push([p[0], p[1]]);
  }
  if (closed && out.length > 2) {
    const f = out[0], l = out[out.length - 1];
    if (Math.hypot(f[0] - l[0], f[1] - l[1]) <= eps) out.pop();
  }
  return out;
}

// ----- tangency / junction post-processing --------------------------------

function primTangentAt(prim: Prim, p: Vec2): Vec2 {
  if (prim.type === "line") return prim.d;
  const rx = p[0] - prim.c[0], ry = p[1] - prim.c[1];
  const L = Math.hypot(rx, ry) || 1;
  return [(-ry / L) * prim.dir, (rx / L) * prim.dir];
}

function footOnLine(prim: Extract<Prim, { type: "line" }>, p: Vec2): Vec2 {
  const t = (p[0] - prim.p[0]) * prim.d[0] + (p[1] - prim.p[1]) * prim.d[1];
  return [prim.p[0] + prim.d[0] * t, prim.p[1] + prim.d[1] * t];
}

function signedLineDist(prim: Extract<Prim, { type: "line" }>, p: Vec2): number {
  return prim.d[0] * (p[1] - prim.p[1]) - prim.d[1] * (p[0] - prim.p[0]);
}

/** Is the junction (A -> B at data point q) nearly tangent within tol? */
function nearlyTangent(A: Prim, B: Prim, q: Vec2, tol: number): boolean {
  if (A.type === "line" && B.type === "line") return false;
  let T: Vec2;
  if (A.type === "line" || B.type === "line") {
    const L = (A.type === "line" ? A : B) as Extract<Prim, { type: "line" }>;
    const C = (A.type === "arc" ? A : B) as Extract<Prim, { type: "arc" }>;
    const h = Math.abs(signedLineDist(L, C.c));
    if (Math.abs(h - C.r) > tol) return false;
    T = footOnLine(L, C.c);
    const reach = Math.max(3 * tol, 4 * Math.sqrt(2 * C.r * tol));
    if (Math.hypot(T[0] - q[0], T[1] - q[1]) > reach) return false;
  } else {
    const a = A as Extract<Prim, { type: "arc" }>, b = B as Extract<Prim, { type: "arc" }>;
    const d = Math.hypot(b.c[0] - a.c[0], b.c[1] - a.c[1]);
    if (d < 1e-12) return false;
    const ext = Math.abs(d - (a.r + b.r)), int = Math.abs(d - Math.abs(a.r - b.r));
    if (Math.min(ext, int) > tol) return false;
    const u: Vec2 = [(b.c[0] - a.c[0]) / d, (b.c[1] - a.c[1]) / d];
    // tangent point on circle a toward/away from b
    T = ext <= int ? polar(a.c, a.r, angleOf(u)) : a.r >= b.r ? polar(a.c, a.r, angleOf(u)) : polar(a.c, a.r, angleOf(u) + Math.PI);
    const reach = Math.max(3 * tol, 4 * Math.sqrt(2 * Math.min(a.r, b.r) * tol));
    if (Math.hypot(T[0] - q[0], T[1] - q[1]) > reach) return false;
  }
  // travel directions must agree (no cusp)
  const ta = primTangentAt(A, T), tb = primTangentAt(B, T);
  return ta[0] * tb[0] + ta[1] * tb[1] > 0.9;
}

/** Robust 1D minimisation: coarse scan of [a, b] followed by golden section around the best sample. */
function scanMin(f: (t: number) => number, a: number, b: number, samples = 24): number {
  let best = a, bf = Infinity;
  const h = (b - a) / samples;
  for (let i = 0; i <= samples; i++) {
    const t = a + h * i;
    const v = f(t);
    if (v < bf) {
      bf = v;
      best = t;
    }
  }
  return golden(f, Math.max(a, best - h), Math.min(b, best + h));
}

/** 1D minimisation by golden section on [a, b]. */
function golden(f: (t: number) => number, a: number, b: number, iters = 60): number {
  const g = (Math.sqrt(5) - 1) / 2;
  let c = b - g * (b - a), d = a + g * (b - a);
  let fc = f(c), fd = f(d);
  for (let i = 0; i < iters; i++) {
    if (fc < fd) {
      b = d;
      d = c;
      fd = fc;
      c = b - g * (b - a);
      fc = f(c);
    } else {
      a = c;
      c = d;
      fc = fd;
      d = a + g * (b - a);
      fd = f(d);
    }
  }
  return (a + b) / 2;
}

function arcCost(pts: Vec2[], c: Vec2, r: number): number {
  let s = 0;
  for (const p of pts) {
    const d = Math.hypot(p[0] - c[0], p[1] - c[1]) - r;
    s += d * d;
  }
  return s;
}

/** Arc tangent to two lines: center on the bisector of the offset lines, 1-parameter fit. */
function fitArcTangentTo2Lines(pts: Vec2[], arc: Extract<Prim, { type: "arc" }>, L1: Extract<Prim, { type: "line" }>, L2: Extract<Prim, { type: "line" }>): { c: Vec2; r: number } | null {
  // sides of the center relative to each line
  const s1 = Math.sign(signedLineDist(L1, arc.c)) || 1;
  const s2 = Math.sign(signedLineDist(L2, arc.c)) || 1;
  const n1: Vec2 = [-L1.d[1] * s1, L1.d[0] * s1];
  const n2: Vec2 = [-L2.d[1] * s2, L2.d[0] * s2];
  // center(r) = intersection of L1 + r n1 and L2 + r n2 (linear in r) or, for parallel lines, r fixed and center slides
  const X0 = lineLineIntersection(L1.p, [L1.p[0] + L1.d[0], L1.p[1] + L1.d[1]], L2.p, [L2.p[0] + L2.d[0], L2.p[1] + L2.d[1]]);
  const cross = L1.d[0] * L2.d[1] - L1.d[1] * L2.d[0];
  if (X0 && Math.abs(cross) > 1e-3) {
    const p1: Vec2 = [L1.p[0] + n1[0], L1.p[1] + n1[1]];
    const p2: Vec2 = [L2.p[0] + n2[0], L2.p[1] + n2[1]];
    const X1 = lineLineIntersection(p1, [p1[0] + L1.d[0], p1[1] + L1.d[1]], p2, [p2[0] + L2.d[0], p2[1] + L2.d[1]]);
    if (!X1) return null;
    const v: Vec2 = [X1.point[0] - X0.point[0], X1.point[1] - X0.point[1]];
    const C = (r: number): Vec2 => [X0.point[0] + v[0] * r, X0.point[1] + v[1] * r];
    // stay near the unconstrained estimate (the junctions are tangent within tolerance)
    const r = scanMin((rr) => arcCost(pts, C(rr), rr), Math.max(1e-9, arc.r * 0.7), arc.r * 1.4);
    return { c: C(r), r };
  }
  // (nearly) parallel lines: radius = half the gap, center slides along the mid line
  const gap = Math.abs(signedLineDist(L1, L2.p));
  const r = gap / 2;
  const base = footOnLine(L1, arc.c);
  const c0: Vec2 = [base[0] + n1[0] * r, base[1] + n1[1] * r];
  const C = (t: number): Vec2 => [c0[0] + L1.d[0] * t, c0[1] + L1.d[1] * t];
  const t0 = (arc.c[0] - c0[0]) * L1.d[0] + (arc.c[1] - c0[1]) * L1.d[1];
  const t = scanMin((tt) => arcCost(pts, C(tt), r), t0 - 0.5 * r, t0 + 0.5 * r);
  return { c: C(t), r };
}

/** Arc tangent to one line: center = foot + s d + r n, 2-parameter Gauss–Newton fit. */
function fitArcTangentTo1Line(pts: Vec2[], arc: Extract<Prim, { type: "arc" }>, L: Extract<Prim, { type: "line" }>): { c: Vec2; r: number } {
  const sd = Math.sign(signedLineDist(L, arc.c)) || 1;
  const n: Vec2 = [-L.d[1] * sd, L.d[0] * sd];
  const foot = footOnLine(L, arc.c);
  let s = 0;
  let r = Math.abs(signedLineDist(L, arc.c));
  r = (r + arc.r) / 2;
  const C = (ss: number, rr: number): Vec2 => [foot[0] + L.d[0] * ss + n[0] * rr, foot[1] + L.d[1] * ss + n[1] * rr];
  for (let it = 0; it < 30; it++) {
    const c = C(s, r);
    let a11 = 0, a12 = 0, a22 = 0, g1 = 0, g2 = 0;
    for (const p of pts) {
      const dx = p[0] - c[0], dy = p[1] - c[1];
      const D = Math.hypot(dx, dy) || 1e-300;
      const res = D - r;
      // d res / d s = -(dx d.x + dy d.y)/D ; d res / d r = -(dx n.x + dy n.y)/D - 1
      const js = -(dx * L.d[0] + dy * L.d[1]) / D;
      const jr = -(dx * n[0] + dy * n[1]) / D - 1;
      a11 += js * js;
      a12 += js * jr;
      a22 += jr * jr;
      g1 += js * res;
      g2 += jr * res;
    }
    const det = a11 * a22 - a12 * a12;
    if (!(Math.abs(det) > 1e-300)) break;
    const ds = -(a22 * g1 - a12 * g2) / det;
    const dr = -(a11 * g2 - a12 * g1) / det;
    const f0 = arcCost(pts, c, r);
    let alpha = 1;
    while (alpha > 1e-4 && arcCost(pts, C(s + alpha * ds, r + alpha * dr), r + alpha * dr) > f0) alpha /= 2;
    if (alpha <= 1e-4) break;
    s += alpha * ds;
    r += alpha * dr;
    if (Math.abs(ds) + Math.abs(dr) < 1e-13 * Math.max(1, r)) break;
  }
  return { c: C(s, r), r: Math.abs(r) };
}

function nearestTo(cands: Vec2[], q: Vec2): Vec2 | null {
  let best: Vec2 | null = null, bd = Infinity;
  for (const c of cands) {
    const d = Math.hypot(c[0] - q[0], c[1] - q[1]);
    if (d < bd) {
      bd = d;
      best = c;
    }
  }
  return best;
}

function junctionPoint(A: Prim, B: Prim, q: Vec2, tangent: boolean): Vec2 {
  if (A.type === "line" && B.type === "line") {
    const r = lineLineIntersection(A.p, [A.p[0] + A.d[0], A.p[1] + A.d[1]], B.p, [B.p[0] + B.d[0], B.p[1] + B.d[1]]);
    const fa = footOnLine(A, q), fb = footOnLine(B, q);
    const mid: Vec2 = [(fa[0] + fb[0]) / 2, (fa[1] + fb[1]) / 2];
    if (!r) return mid;
    const sinA = Math.abs(A.d[0] * B.d[1] - A.d[1] * B.d[0]);
    if (sinA < Math.sin(2 * DEG) && Math.hypot(r.point[0] - q[0], r.point[1] - q[1]) > Math.hypot(fa[0] - fb[0], fa[1] - fb[1]) * 10) return mid;
    return r.point;
  }
  if (A.type === "line" || B.type === "line") {
    const L = (A.type === "line" ? A : B) as Extract<Prim, { type: "line" }>;
    const C = (A.type === "arc" ? A : B) as Extract<Prim, { type: "arc" }>;
    if (tangent) return footOnLine(L, C.c);
    const hits = lineCircleIntersection(L.p, [L.p[0] + L.d[0], L.p[1] + L.d[1]], C.c, C.r).map((h) => h.point);
    return nearestTo(hits, q) ?? footOnLine(L, C.c);
  }
  const a = A as Extract<Prim, { type: "arc" }>, b = B as Extract<Prim, { type: "arc" }>;
  const d = Math.hypot(b.c[0] - a.c[0], b.c[1] - a.c[1]);
  if (tangent && d > 1e-12) {
    const u = angleOf([b.c[0] - a.c[0], b.c[1] - a.c[1]]);
    const ext = Math.abs(d - (a.r + b.r)) <= Math.abs(d - Math.abs(a.r - b.r));
    if (ext) return polar(a.c, a.r, u);
    return a.r >= b.r ? polar(a.c, a.r, u) : polar(a.c, a.r, u + Math.PI);
  }
  const hits = circleCircleIntersection(a.c, a.r, b.c, b.r);
  const h = nearestTo(hits, q);
  if (h) return h;
  // no intersection: midpoint of the closest points of both circles to q
  const pa = polar(a.c, a.r, angleOf([q[0] - a.c[0], q[1] - a.c[1]]));
  const pb = polar(b.c, b.r, angleOf([q[0] - b.c[0], q[1] - b.c[1]]));
  return [(pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2];
}

function projectOnPrim(prim: Prim, p: Vec2): Vec2 {
  if (prim.type === "line") return footOnLine(prim, p);
  return polar(prim.c, prim.r, angleOf([p[0] - prim.c[0], p[1] - prim.c[1]]));
}

/**
 * Split a noisy section polyline into a minimal chain of lines and arcs within `tolerance`, with full
 * details (tangency flags, data ranges).
 */
export function autoSegmentDetailed(points: Vec2[], opts: AutoSegmentOptions): AutoSegmentDetail {
  const closed = !!opts.closed;
  const P = cleanInput(points, closed);
  const N = P.length;
  const empty: AutoSegmentDetail = { segments: [], tangent: [], ranges: [], arcDirection: [], maxError: 0, closed };
  if (N < 2) return empty;
  const tol = Math.max(opts.tolerance, 1e-12);
  if (N === 2 || (!closed && N < 3)) {
    return { segments: [{ type: "line", a: P[0], b: P[N - 1] }], tangent: [], ranges: [[0, N - 1]], arcDirection: [0], maxError: 0, closed: false };
  }
  const seg = new Segmenter(P, { ...opts, tolerance: tol, closed });
  // full circle?
  if (closed && (opts.allowArcs ?? true) && N >= 5) {
    const c = circleCore(P);
    if (c) {
      let err = 0;
      for (const p of P) err = Math.max(err, Math.abs(Math.hypot(p[0] - c.c[0], p[1] - c.c[1]) - c.r));
      const sweep = Math.abs(angularSweep([...P, P[0]], c.c));
      if (err <= tol && sweep > TAU * 0.99) {
        if (opts.fullCircle ?? true)
          return { segments: [{ type: "circle", center: c.c, radius: c.r }], tangent: [], ranges: [[0, N - 1]], arcDirection: [0], maxError: err, closed };
        // two arcs split at the first data point and the opposite side
        const a0 = angleOf([P[0][0] - c.c[0], P[0][1] - c.c[1]]);
        const dir = angularSweep([...P, P[0]], c.c) >= 0 ? 1 : -1;
        const pA = polar(c.c, c.r, a0), pB = polar(c.c, c.r, a0 + dir * Math.PI);
        const m1 = polar(c.c, c.r, a0 + (dir * Math.PI) / 2), m2 = polar(c.c, c.r, a0 + (dir * 3 * Math.PI) / 2);
        const half = Math.floor(N / 2);
        return {
          segments: [
            { type: "arc", start: pA, mid: m1, end: pB },
            { type: "arc", start: pB, mid: m2, end: pA },
          ],
          tangent: [true, true],
          ranges: [
            [0, half],
            [half, 0],
          ],
          arcDirection: [dir, dir],
          maxError: err,
          closed,
        };
      }
    }
  }

  // initial breakpoints from corners
  const corners = detectCorners(P, closed, tol, opts.cornerAngleDeg ?? 30);
  let pieces: Piece[] = [];
  if (closed) {
    let bps = corners.slice().sort((a, b) => a - b);
    if (bps.length === 0) bps = [0];
    if (bps.length === 1) {
      // add the farthest point as a second breakpoint
      const a = P[bps[0]];
      let far = (bps[0] + Math.floor(N / 2)) % N, fd = -1;
      for (let i = 0; i < N; i++) {
        const d = Math.hypot(P[i][0] - a[0], P[i][1] - a[1]);
        if (d > fd) {
          fd = d;
          far = i;
        }
      }
      bps = [bps[0], far].sort((x, y) => x - y);
      if (bps[0] === bps[1]) bps = [bps[0], (bps[0] + Math.floor(N / 2)) % N].sort((x, y) => x - y);
    }
    for (let k = 0; k < bps.length; k++) {
      const s = bps[k];
      const e = k + 1 < bps.length ? bps[k + 1] : bps[0] + N;
      seg.splitRec(s, e, pieces);
    }
  } else {
    const bps = [0, ...corners.filter((c) => c >= seg.minPts - 1 && c <= N - seg.minPts).sort((a, b) => a - b), N - 1];
    const uniqB = bps.filter((b, i) => i === 0 || b - bps[i - 1] >= 1);
    for (let k = 0; k + 1 < uniqB.length; k++) seg.splitRec(uniqB[k], uniqB[k + 1], pieces);
  }
  pieces = seg.normalize(pieces);
  pieces = seg.merge(pieces);
  pieces = seg.refine(pieces);
  pieces = seg.eliminate(pieces);
  pieces = seg.merge(pieces);
  pieces = seg.refine(pieces, 2);
  pieces = seg.eliminate(pieces);
  pieces = seg.merge(pieces);

  // closed loop collapsed into a single arc piece: report as circle
  const K = pieces.length;
  const prims: Prim[] = pieces.map((p) => p.prim);
  const q = (k: number) => seg.pt(pieces[k].e); // data point at junction k (between k and k+1)
  const nJ = closed ? K : K - 1;

  // tangency classification and enforcement
  const snap = opts.snapTangent ?? true;
  const tangent: boolean[] = new Array(nJ).fill(false);
  if (snap)
    for (let j = 0; j < nJ; j++) tangent[j] = nearlyTangent(prims[j], prims[(j + 1) % K], q(j), tol);
  const lineNeighbours = (k: number): Extract<Prim, { type: "line" }>[] => {
    const out: Extract<Prim, { type: "line" }>[] = [];
    const prevJ = k - 1 >= 0 ? k - 1 : closed ? K - 1 : -1;
    const nextJ = k < nJ ? k : -1;
    if (prevJ >= 0 && tangent[prevJ] && prims[(k - 1 + K) % K].type === "line") out.push(prims[(k - 1 + K) % K] as Extract<Prim, { type: "line" }>);
    if (nextJ >= 0 && tangent[nextJ] && prims[(k + 1) % K].type === "line") out.push(prims[(k + 1) % K] as Extract<Prim, { type: "line" }>);
    return out;
  };
  const constrainedArc = new Array<boolean>(K).fill(false);
  // max deviation of a candidate arc from its own data (junction points excluded: they are shared)
  const arcErr = (k: number, pr: Extract<Prim, { type: "arc" }>): number => {
    const pc = pieces[k];
    const trim = pc.e - pc.s + 1 > 4 ? 1 : 0;
    let mx = 0;
    for (let i = pc.s + trim; i <= pc.e - trim; i++) {
      const p = seg.pt(i);
      mx = Math.max(mx, Math.abs(Math.hypot(p[0] - pr.c[0], p[1] - pr.c[1]) - pr.r));
    }
    return mx;
  };
  for (let k = 0; k < K; k++) {
    const pr = prims[k];
    if (pr.type !== "arc") continue;
    const lines = lineNeighbours(k);
    if (!lines.length) continue;
    const pts = seg.points(pieces[k].s, pieces[k].e);
    const fit = lines.length === 2 && lines[0] !== lines[1] ? fitArcTangentTo2Lines(pts, pr, lines[0], lines[1]) : fitArcTangentTo1Line(pts, pr, lines[0]);
    const cand = fit && isFinite(fit.r) && fit.r > 0 ? ({ type: "arc", c: fit.c, r: fit.r, dir: pr.dir } as const) : null;
    if (cand && arcErr(k, cand) <= tol) {
      prims[k] = cand;
      constrainedArc[k] = true;
    } else {
      // tangency would break the tolerance: keep the free fit, junctions to lines become sharp
      const prevJ = k - 1 >= 0 ? k - 1 : closed ? K - 1 : -1;
      if (prevJ >= 0 && prims[(k - 1 + K) % K].type === "line") tangent[prevJ] = false;
      if (k < nJ && prims[(k + 1) % K].type === "line") tangent[k] = false;
    }
  }
  // arc-arc tangency: move an arc that is not yet constrained along the center line
  for (let j = 0; j < nJ; j++) {
    const a = prims[j], b = prims[(j + 1) % K];
    if (!tangent[j] || a.type !== "arc" || b.type !== "arc") continue;
    const d = Math.hypot(b.c[0] - a.c[0], b.c[1] - a.c[1]);
    if (d < 1e-12) {
      tangent[j] = false;
      continue;
    }
    const ext = Math.abs(d - (a.r + b.r)) <= Math.abs(d - Math.abs(a.r - b.r));
    const D = ext ? a.r + b.r : Math.abs(a.r - b.r);
    if (Math.abs(d - D) <= 1e-9 * Math.max(1, a.r, b.r)) continue;
    const u: Vec2 = [(b.c[0] - a.c[0]) / d, (b.c[1] - a.c[1]) / d];
    const kb = (j + 1) % K;
    const nb = { ...b, c: [a.c[0] + u[0] * D, a.c[1] + u[1] * D] as Vec2 };
    const na = { ...a, c: [b.c[0] - u[0] * D, b.c[1] - u[1] * D] as Vec2 };
    if (!constrainedArc[kb] && arcErr(kb, nb) <= tol) {
      prims[kb] = nb;
      constrainedArc[kb] = true;
    } else if (!constrainedArc[j] && arcErr(j, na) <= tol) {
      prims[j] = na;
      constrainedArc[j] = true;
    } else tangent[j] = false;
  }

  // junction points
  const J: Vec2[] = [];
  for (let j = 0; j < nJ; j++) J.push(junctionPoint(prims[j], prims[(j + 1) % K], q(j), tangent[j]));
  const startPt = closed ? J[nJ - 1] : projectOnPrim(prims[0], P[0]);
  const endPt = closed ? J[nJ - 1] : projectOnPrim(prims[K - 1], P[N - 1]);

  const segments: Segment2D[] = [];
  const arcDirection: number[] = [];
  for (let k = 0; k < K; k++) {
    const a = k === 0 ? startPt : J[k - 1];
    const b = k === K - 1 ? endPt : J[k];
    const pr = prims[k];
    if (pr.type === "line") {
      segments.push({ type: "line", a, b });
      arcDirection.push(0);
    } else {
      const as = angleOf([a[0] - pr.c[0], a[1] - pr.c[1]]);
      const ae = angleOf([b[0] - pr.c[0], b[1] - pr.c[1]]);
      let sweep = pr.dir > 0 ? normAngle(ae - as) : -normAngle(as - ae);
      if (Math.abs(sweep) < 1e-9) sweep = pr.dir * TAU;
      const mid = polar(pr.c, Math.hypot(a[0] - pr.c[0], a[1] - pr.c[1]), as + sweep / 2);
      if (Math.abs(sweep) >= TAU - 1e-9) {
        segments.push({ type: "circle", center: pr.c, radius: pr.r });
        arcDirection.push(0);
      } else {
        segments.push({ type: "arc", start: a, mid, end: b });
        arcDirection.push(pr.dir);
      }
    }
  }
  // final error
  let maxError = 0;
  const ranges: [number, number][] = [];
  for (let k = 0; k < K; k++) {
    const pc = pieces[k];
    ranges.push([((pc.s % N) + N) % N, ((pc.e % N) + N) % N]);
    // data near a junction may belong to either neighbour: measure against own and adjacent segments
    const near = [segments[k]];
    if (k > 0 || closed) near.push(segments[(k - 1 + K) % K]);
    if (k < K - 1 || closed) near.push(segments[(k + 1) % K]);
    for (let i = pc.s; i <= pc.e; i++) {
      const p = seg.pt(i);
      let d = Infinity;
      for (const g of near) d = Math.min(d, closestPointOnSegment(g, p).distance);
      if (d > maxError) maxError = d;
    }
  }
  return { segments, tangent, ranges, arcDirection, maxError, closed };
}

/** Split a noisy section polyline into a minimal sequence of lines and arcs within tolerance. */
export function autoSegment(points: Vec2[], opts: AutoSegmentOptions): Segment2D[] {
  return autoSegmentDetailed(points, opts).segments;
}

export interface AutoSketchOptions extends AutoSegmentOptions {
  /** add tangent constraints at tangent junctions (default true) */
  tangentConstraints?: boolean;
  /** add horizontal/vertical constraints to lines within `hvAngleDeg` of the axes (default false) */
  horizontalVertical?: boolean;
  /** angular threshold for horizontal/vertical detection (default 1 deg) */
  hvAngleDeg?: number;
  /** run the solver so added constraints hold exactly (default true) */
  solve?: boolean;
}

/**
 * Convert a section polyline into a sketch: `autoSegment` output becomes lines/arcs/circles sharing
 * their junction points, with tangent (and optionally horizontal/vertical) constraints.
 */
export function autoSketch(points: Vec2[], opts: AutoSketchOptions): Sketch {
  const det = autoSegmentDetailed(points, opts);
  return segmentsToSketch(det, opts);
}

/** Build a sketch from an autoSegment result (see autoSketch). */
export function segmentsToSketch(det: AutoSegmentDetail, opts: Omit<AutoSketchOptions, "tolerance"> = {}): Sketch {
  const s = createSketch();
  const segs = det.segments;
  const K = segs.length;
  if (!K) return s;
  if (K === 1 && segs[0].type === "circle") {
    addCircle(s, segs[0].center, segs[0].radius);
    return s;
  }
  const endOf = (g: Segment2D): [Vec2, Vec2] =>
    g.type === "line" ? [g.a, g.b] : g.type === "arc" ? [g.start, g.end] : [[0, 0], [0, 0]];
  // junction point ids
  const nPts = det.closed ? K : K + 1;
  const pid: string[] = [];
  for (let k = 0; k < nPts; k++) {
    const p = k < K ? endOf(segs[k])[0] : endOf(segs[K - 1])[1];
    pid.push(addPoint(s, p));
  }
  const eids: string[] = [];
  for (let k = 0; k < K; k++) {
    const g = segs[k];
    const a = pid[k], b = pid[(k + 1) % nPts];
    if (g.type === "line") eids.push(addLine(s, a, b));
    else if (g.type === "arc") {
      const c = circleThrough(g.start, g.mid, g.end);
      if (!c) {
        eids.push(addLine(s, a, b));
        continue;
      }
      const cId = addPoint(s, c);
      const ccw = (det.arcDirection[k] || arcIsCCW(g)) > 0;
      eids.push(ccw ? addArc(s, cId, a, b) : addArc(s, cId, b, a));
    } else if (g.type === "circle") eids.push(addCircle(s, g.center, g.radius));
  }
  if (opts.tangentConstraints ?? true) {
    const nJ = det.closed ? K : K - 1;
    for (let j = 0; j < nJ; j++) {
      if (!det.tangent[j]) continue;
      const e1 = s.entities.find((e) => e.id === eids[j]);
      const e2 = s.entities.find((e) => e.id === eids[(j + 1) % K]);
      if (!e1 || !e2 || (e1.type === "line" && e2.type === "line")) continue;
      addConstraint(s, { type: "tangent", refs: [eids[j], eids[(j + 1) % K]] });
    }
  }
  let added = false;
  if (opts.horizontalVertical) {
    const thr = (opts.hvAngleDeg ?? 1) * DEG;
    for (let k = 0; k < K; k++) {
      const g = segs[k];
      if (g.type !== "line") continue;
      const ang = Math.atan2(g.b[1] - g.a[1], g.b[0] - g.a[0]);
      const toH = Math.abs(wrapPi(ang - Math.round(ang / Math.PI) * Math.PI));
      const toV = Math.abs(wrapPi(ang - Math.PI / 2 - Math.round((ang - Math.PI / 2) / Math.PI) * Math.PI));
      if (toH <= thr) {
        addConstraint(s, { type: "horizontal", refs: [eids[k]] });
        added = true;
      } else if (toV <= thr) {
        addConstraint(s, { type: "vertical", refs: [eids[k]] });
        added = true;
      }
    }
  }
  if ((opts.solve ?? true) && (added || s.constraints.length)) {
    const r = solveSketch(s, { analyzeConflicts: false });
    if (r.status === "ok") return r.sketch;
  }
  return s;
}

function circleThrough(a: Vec2, b: Vec2, c: Vec2): Vec2 | null {
  const d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
  if (Math.abs(d) < 1e-14) return null;
  const a2 = a[0] ** 2 + a[1] ** 2, b2 = b[0] ** 2 + b[1] ** 2, c2 = c[0] ** 2 + c[1] ** 2;
  return [(a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / d, (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / d];
}

function arcIsCCW(g: Extract<Segment2D, { type: "arc" }>): number {
  const cr = (g.mid[0] - g.start[0]) * (g.end[1] - g.start[1]) - (g.mid[1] - g.start[1]) * (g.end[0] - g.start[0]);
  return cr > 0 ? 1 : -1;
}

/** Evaluate a fitted B-spline over its domain (convenience for display). */
export function bsplineFitPoint(curve: BSplineCurve2Def, t: number): Vec2 {
  const [u0, u1] = bsplineDomain(curve);
  return evalBSpline2(curve, u0 + (u1 - u0) * t);
}
