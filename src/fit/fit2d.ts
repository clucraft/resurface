/**
 * 2D line / circle / arc fitting (sketch sections) and 3D line fitting (reference lines).
 */
import type { FitResult, LineDef, Vec2, Vec3 } from "../core/types";
import { eigenSym3, solveLinear } from "../core/linalg";
import { DEFAULT_MAX_POINTS, accumulateNormal, buildSample, canonicalSign, levenbergMarquardt, selectionCount } from "./common";
import type { LMProblem, PointArray } from "./common";

/** 2D input: Vec2[] or packed xy. */
export type Points2D = Vec2[] | ArrayLike<number>;

function toPacked2(points: Points2D): Float64Array {
  if (Array.isArray(points) && points.length > 0 && Array.isArray(points[0])) {
    const p = points as Vec2[];
    const out = new Float64Array(p.length * 2);
    for (let i = 0; i < p.length; i++) {
      out[2 * i] = p[i][0];
      out[2 * i + 1] = p[i][1];
    }
    return out;
  }
  return Float64Array.from(points as ArrayLike<number>);
}

export interface Line2Fit {
  origin: Vec2;
  /** unit direction */
  direction: Vec2;
  /** RMS orthogonal distance */
  rms: number;
  maxAbs: number;
}

/** Total-least-squares 2D line (PCA). */
export function fitLine2D(points: Points2D): Line2Fit {
  const xy = toPacked2(points);
  const n = xy.length >> 1;
  if (n < 2) throw new Error("fitLine2D: needs at least 2 points");
  let cx = 0, cy = 0;
  for (let i = 0; i < n; i++) {
    cx += xy[2 * i];
    cy += xy[2 * i + 1];
  }
  cx /= n;
  cy /= n;
  let sxx = 0, sxy = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xy[2 * i] - cx, dy = xy[2 * i + 1] - cy;
    sxx += dx * dx;
    sxy += dx * dy;
    syy += dy * dy;
  }
  if (sxx + syy <= 1e-300) throw new Error("fitLine2D: all points coincide");
  // largest eigenvector of [[sxx,sxy],[sxy,syy]]
  const theta = 0.5 * Math.atan2(2 * sxy, sxx - syy);
  const dir: Vec2 = [Math.cos(theta), Math.sin(theta)];
  let ss = 0, mx = 0;
  for (let i = 0; i < n; i++) {
    const d = -(xy[2 * i] - cx) * dir[1] + (xy[2 * i + 1] - cy) * dir[0];
    ss += d * d;
    mx = Math.max(mx, Math.abs(d));
  }
  return { origin: [cx, cy], direction: dir, rms: Math.sqrt(ss / n), maxAbs: mx };
}

export interface Circle2Fit {
  center: Vec2;
  radius: number;
  /** RMS of radial residuals */
  rms: number;
  maxAbs: number;
}

/**
 * Algebraic (Kåsa) circle on packed xy (optionally on a subset of point indices).
 * Returns null for (near-)collinear input.
 */
export function kasaCircle(xy: ArrayLike<number>, n: number, subset?: ArrayLike<number>): { cx: number; cy: number; r: number } | null {
  const m = subset ? subset.length : n;
  if (m < 3) return null;
  let mx = 0, my = 0;
  for (let t = 0; t < m; t++) {
    const i = subset ? subset[t] : t;
    mx += xy[2 * i];
    my += xy[2 * i + 1];
  }
  mx /= m;
  my /= m;
  // normalise for conditioning
  let s2 = 0;
  for (let t = 0; t < m; t++) {
    const i = subset ? subset[t] : t;
    const dx = xy[2 * i] - mx, dy = xy[2 * i + 1] - my;
    s2 += dx * dx + dy * dy;
  }
  const sc = Math.sqrt(s2 / m);
  if (!(sc > 1e-300)) return null;
  const A = new Float64Array(9), b = new Float64Array(3);
  let cxx = 0, cxy = 0, cyy = 0;
  for (let t = 0; t < m; t++) {
    const i = subset ? subset[t] : t;
    const x = (xy[2 * i] - mx) / sc, y = (xy[2 * i + 1] - my) / sc;
    cxx += x * x;
    cxy += x * y;
    cyy += y * y;
    const z = -(x * x + y * y);
    A[0] += x * x;
    A[1] += x * y;
    A[2] += x;
    A[4] += y * y;
    A[5] += y;
    A[8] += 1;
    b[0] += x * z;
    b[1] += y * z;
    b[2] += z;
  }
  // (near-)collinear points: the smaller eigenvalue of the normalised 2D scatter vanishes
  const tr = (cxx + cyy) / m, det = (cxx * cyy - cxy * cxy) / (m * m);
  const lmin = tr / 2 - Math.sqrt(Math.max(0, (tr * tr) / 4 - det));
  if (!(lmin > 1e-12 * tr)) return null;
  A[3] = A[1];
  A[6] = A[2];
  A[7] = A[5];
  const sol = solveLinear(A, b, 3);
  if (!sol) return null;
  const cx = -sol[0] / 2, cy = -sol[1] / 2;
  const r2 = cx * cx + cy * cy - sol[2];
  if (!(r2 > 0) || !Number.isFinite(r2)) return null;
  return { cx: mx + cx * sc, cy: my + cy * sc, r: Math.sqrt(r2) * sc };
}

/** Geometric circle refinement (LM on |p - c| - r), optional fixed radius. */
export function refineCircle(
  xy: ArrayLike<number>,
  n: number,
  init: { cx: number; cy: number; r: number },
  opts: { weights?: ArrayLike<number>; fixedRadius?: boolean; maxIterations?: number } = {},
): { cx: number; cy: number; r: number } {
  const w = opts.weights;
  const cost = (s: Float64Array) => {
    let c = 0;
    for (let i = 0; i < n; i++) {
      const d = Math.hypot(xy[2 * i] - s[0], xy[2 * i + 1] - s[1]) - s[2];
      c += (w ? w[i] : 1) * d * d;
    }
    return c;
  };
  const prob: LMProblem<Float64Array> = {
    nParams: 3,
    linearize(s, JtJ, Jtr) {
      const J = new Float64Array(3);
      let c = 0;
      for (let i = 0; i < n; i++) {
        const dx = xy[2 * i] - s[0], dy = xy[2 * i + 1] - s[1];
        const dist = Math.hypot(dx, dy);
        const r = dist - s[2];
        const wi = w ? w[i] : 1;
        c += wi * r * r;
        if (dist > 1e-300) {
          J[0] = -dx / dist;
          J[1] = -dy / dist;
        } else {
          J[0] = 0;
          J[1] = 0;
        }
        J[2] = -1;
        accumulateNormal(JtJ, Jtr, J, 3, wi, r);
      }
      return c;
    },
    cost,
    retract: (s, d) => Float64Array.of(s[0] + d[0], s[1] + d[1], s[2] + d[2]),
  };
  const res = levenbergMarquardt(prob, Float64Array.of(init.cx, init.cy, init.r), {
    maxIterations: opts.maxIterations ?? 100,
    fixed: opts.fixedRadius ? [false, false, true] : undefined,
  });
  return { cx: res.state[0], cy: res.state[1], r: Math.abs(res.state[2]) };
}

/** Best-fit circle (algebraic initialisation + geometric refinement). */
export function fitCircle2D(points: Points2D, opts: { radius?: number } = {}): Circle2Fit {
  const xy = toPacked2(points);
  const n = xy.length >> 1;
  if (n < 3) throw new Error("fitCircle2D: needs at least 3 points");
  const init = kasaCircle(xy, n);
  if (!init) throw new Error("fitCircle2D: points are collinear — no circle");
  if (opts.radius !== undefined) init.r = opts.radius;
  const c = refineCircle(xy, n, init, { fixedRadius: opts.radius !== undefined });
  let ss = 0, mx = 0;
  for (let i = 0; i < n; i++) {
    const d = Math.hypot(xy[2 * i] - c.cx, xy[2 * i + 1] - c.cy) - c.r;
    ss += d * d;
    mx = Math.max(mx, Math.abs(d));
  }
  return { center: [c.cx, c.cy], radius: c.r, rms: Math.sqrt(ss / n), maxAbs: mx };
}

export interface Arc2Fit extends Circle2Fit {
  start: Vec2;
  mid: Vec2;
  end: Vec2;
  /** signed sweep angle (radians, CCW positive) from start to end following the point order */
  sweep: number;
}

/**
 * Circular arc through ordered points: circle fit + start/end from the first/last points
 * projected onto the circle, sweep direction following the point order.
 */
export function fitArc2D(points: Points2D, opts: { radius?: number } = {}): Arc2Fit {
  const xy = toPacked2(points);
  const n = xy.length >> 1;
  const c = fitCircle2D(xy, opts);
  const [cx, cy] = c.center;
  // accumulate signed angle increments along the ordered points
  let prev = Math.atan2(xy[1] - cy, xy[0] - cx);
  const a0 = prev;
  let total = 0;
  for (let i = 1; i < n; i++) {
    const a = Math.atan2(xy[2 * i + 1] - cy, xy[2 * i] - cx);
    let d = a - prev;
    while (d > Math.PI) d -= 2 * Math.PI;
    while (d < -Math.PI) d += 2 * Math.PI;
    total += d;
    prev = a;
  }
  const at = (ang: number): Vec2 => [cx + c.radius * Math.cos(ang), cy + c.radius * Math.sin(ang)];
  return { ...c, start: at(a0), mid: at(a0 + total / 2), end: at(a0 + total), sweep: total };
}

export interface Line3FitOptions {
  indices?: ArrayLike<number>;
  maxPoints?: number;
  seed?: number;
}

/** Total-least-squares 3D line through a selection (PCA). stats are orthogonal distances. */
export function fitLine3D(pos: PointArray, opts: Line3FitOptions = {}): FitResult<LineDef> & { tMin: number; tMax: number } {
  const S = buildSample(pos, { indices: opts.indices, maxPoints: opts.maxPoints ?? DEFAULT_MAX_POINTS, seed: opts.seed }, 2, "fitLine3D");
  const cov = new Float64Array(9);
  for (let k = 0; k < S.m; k++) {
    const x = S.xs[3 * k], y = S.xs[3 * k + 1], z = S.xs[3 * k + 2];
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
  const dir = canonicalSign(e.vectors[2]);
  const o = S.center;
  // stats + extent over all points
  const count = selectionCount(pos, opts.indices);
  let ss = 0, mx = 0, sum = 0, cnt = 0, tMin = Infinity, tMax = -Infinity;
  for (let k = 0; k < count; k++) {
    const i = opts.indices ? opts.indices[k] : k;
    const dx = pos[3 * i] - o[0], dy = pos[3 * i + 1] - o[1], dz = pos[3 * i + 2] - o[2];
    if (!Number.isFinite(dx + dy + dz)) continue;
    const t = dx * dir[0] + dy * dir[1] + dz * dir[2];
    const d = Math.sqrt(Math.max(0, dx * dx + dy * dy + dz * dz - t * t));
    ss += d * d;
    sum += d;
    mx = Math.max(mx, d);
    tMin = Math.min(tMin, t);
    tMax = Math.max(tMax, t);
    cnt++;
  }
  const origin: Vec3 = [o[0], o[1], o[2]];
  return {
    def: { kind: "line", origin, direction: dir },
    stats: { rms: Math.sqrt(ss / Math.max(1, cnt)), maxAbs: mx, mean: sum / Math.max(1, cnt), count: cnt },
    tMin,
    tMax,
  };
}
