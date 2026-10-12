/**
 * Analytic primitive fitting: plane, sphere, cylinder, cone.
 *
 *  - Inputs are packed xyz arrays (+ optional per-vertex normals, optional selection `indices`).
 *  - Iterative refinement runs on a deterministic sub-sample (≤ `maxPoints`, default 200k) in
 *    normalised coordinates (centroid at 0, RMS radius 1) — scale invariant; final statistics and
 *    extents are always computed on all selected points.
 *  - Geometric (orthogonal-distance) Levenberg–Marquardt with minimal parametrisations.
 *  - `inliers` (when present) is a per-selected-point mask in selection order (`indices` order, or
 *    vertex order when no indices) and `stats` are then computed over the inliers only.
 */
import type { ConeDef, CylinderDef, FitResult, FitStats, PlaneDef, PrimitiveDef, SphereDef, Vec3 } from "../core/types";
import { eigenSym3, solveLinear } from "../core/linalg";
import { cross, dot, length, normalize } from "../core/vec";
import {
  DEFAULT_MAX_POINTS,
  TUKEY_C,
  accumulateNormal,
  buildSample,
  canonicalSign,
  fibonacciHemisphere,
  levenbergMarquardt,
  mulberry32,
  perpBasis,
  selectionCentroid,
  selectionCount,
  subsample,
  tukeyWeights,
} from "./common";
import type { LMOptions, LMProblem, PointArray, PointSample } from "./common";
import { kasaCircle, refineCircle } from "./fit2d";
import { GridIndex, pcaNormalAt } from "./pointQuery";

export type PrimitiveKind = PrimitiveDef["kind"];

export interface FitOptions {
  /** vertex indices of the selection (default: all vertices) */
  indices?: ArrayLike<number>;
  /** per-vertex normals (packed xyz, same indexing as positions) */
  normals?: ArrayLike<number> | null;
  /** iteratively reweighted (Tukey biweight) refinement; sets `inliers` */
  robust?: boolean;
  /** max points used for iterative refinement (deterministic sub-sample), default 200k */
  maxPoints?: number;
  /** sub-sampling seed (default 1) */
  seed?: number;
  /** max LM iterations (default 100) */
  maxIterations?: number;
}

export interface SphereFitOptions extends FitOptions {
  /** start from this sphere instead of the algebraic fit */
  initial?: SphereDef;
}

export interface CylinderFitOptions extends FitOptions {
  /** start from this cylinder (skips the global axis search) */
  initial?: CylinderDef;
}

export interface ConeFitOptions extends FitOptions {
  /** start from this cone (skips the initialisation search) */
  initial?: ConeDef;
  /** |half angle| below which the result is flagged `nearCylinder` (radians, default 0.1°) */
  nearCylinderAngle?: number;
}

export interface ConeFitResult extends FitResult<ConeDef> {
  /** the fitted half angle is tiny: data is (nearly) cylindrical, apex is far away / ill-defined */
  nearCylinder: boolean;
  /** human readable explanation when nearCylinder */
  warning?: string;
  /** equivalent cylinder when nearCylinder */
  cylinder?: CylinderDef;
}

// ===========================================================================
// Distances, projections, statistics
// ===========================================================================

export type DistanceKernel = (x: number, y: number, z: number) => number;

/** Fast signed-distance closure for a primitive (positive outside / along the plane normal). */
export function distanceKernel(def: PrimitiveDef): DistanceKernel {
  switch (def.kind) {
    case "plane": {
      const [ox, oy, oz] = def.origin;
      const [nx, ny, nz] = normalize(def.normal);
      return (x, y, z) => (x - ox) * nx + (y - oy) * ny + (z - oz) * nz;
    }
    case "sphere": {
      const [cx, cy, cz] = def.center;
      const r = def.radius;
      return (x, y, z) => Math.hypot(x - cx, y - cy, z - cz) - r;
    }
    case "cylinder": {
      const [ox, oy, oz] = def.origin;
      const [ax, ay, az] = normalize(def.axis);
      const r = def.radius;
      return (x, y, z) => {
        const dx = x - ox, dy = y - oy, dz = z - oz;
        const h = dx * ax + dy * ay + dz * az;
        const wx = dx - h * ax, wy = dy - h * ay, wz = dz - h * az;
        return Math.sqrt(wx * wx + wy * wy + wz * wz) - r;
      };
    }
    case "cone": {
      const [ox, oy, oz] = def.apex;
      const [ax, ay, az] = normalize(def.axis);
      const ca = Math.cos(def.halfAngle), sa = Math.sin(def.halfAngle);
      return (x, y, z) => {
        const dx = x - ox, dy = y - oy, dz = z - oz;
        const h = dx * ax + dy * ay + dz * az;
        const wx = dx - h * ax, wy = dy - h * ay, wz = dz - h * az;
        const rho = Math.sqrt(wx * wx + wy * wy + wz * wz);
        const t = h * ca + rho * sa;
        return t >= 0 ? rho * ca - h * sa : Math.sqrt(dx * dx + dy * dy + dz * dz);
      };
    }
  }
}

/** Non-allocating outward-normal kernel: writes the unit normal at the foot point of (x,y,z) into out[0..3). */
export type NormalKernel = (x: number, y: number, z: number, out: Float64Array) => void;

export function normalKernel(def: PrimitiveDef): NormalKernel {
  switch (def.kind) {
    case "plane": {
      const [nx, ny, nz] = normalize(def.normal);
      return (_x, _y, _z, out) => {
        out[0] = nx;
        out[1] = ny;
        out[2] = nz;
      };
    }
    case "sphere": {
      const [cx, cy, cz] = def.center;
      return (x, y, z, out) => {
        const dx = x - cx, dy = y - cy, dz = z - cz;
        const l = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (l > 1e-300) {
          out[0] = dx / l;
          out[1] = dy / l;
          out[2] = dz / l;
        } else {
          out[0] = 1;
          out[1] = 0;
          out[2] = 0;
        }
      };
    }
    case "cylinder":
    case "cone": {
      const o = def.kind === "cylinder" ? def.origin : def.apex;
      const [ox, oy, oz] = o;
      const [ax, ay, az] = normalize(def.axis);
      const ca = def.kind === "cone" ? Math.cos(def.halfAngle) : 1;
      const sa = def.kind === "cone" ? Math.sin(def.halfAngle) : 0;
      const [px, py, pz] = perpBasis([ax, ay, az])[0];
      return (x, y, z, out) => {
        const dx = x - ox, dy = y - oy, dz = z - oz;
        const h = dx * ax + dy * ay + dz * az;
        let wx = dx - h * ax, wy = dy - h * ay, wz = dz - h * az;
        const l = Math.sqrt(wx * wx + wy * wy + wz * wz);
        if (l > 1e-300) {
          wx /= l;
          wy /= l;
          wz /= l;
        } else {
          wx = px;
          wy = py;
          wz = pz;
        }
        out[0] = ca * wx - sa * ax;
        out[1] = ca * wy - sa * ay;
        out[2] = ca * wz - sa * az;
      };
    }
  }
}

/** Signed distance from p to the primitive (positive outside / on the normal side of a plane). */
export function distanceToPrimitive(def: PrimitiveDef, p: Vec3): number {
  return distanceKernel(def)(p[0], p[1], p[2]);
}

/** Closest point on the (infinite) primitive surface. */
export function projectToPrimitive(def: PrimitiveDef, p: Vec3): Vec3 {
  switch (def.kind) {
    case "plane": {
      const n = normalize(def.normal);
      const d = dot([p[0] - def.origin[0], p[1] - def.origin[1], p[2] - def.origin[2]], n);
      return [p[0] - d * n[0], p[1] - d * n[1], p[2] - d * n[2]];
    }
    case "sphere": {
      const v: Vec3 = [p[0] - def.center[0], p[1] - def.center[1], p[2] - def.center[2]];
      const l = length(v);
      const u: Vec3 = l > 1e-300 ? [v[0] / l, v[1] / l, v[2] / l] : [1, 0, 0];
      return [def.center[0] + def.radius * u[0], def.center[1] + def.radius * u[1], def.center[2] + def.radius * u[2]];
    }
    case "cylinder": {
      const a = normalize(def.axis);
      const v: Vec3 = [p[0] - def.origin[0], p[1] - def.origin[1], p[2] - def.origin[2]];
      const h = dot(v, a);
      const f: Vec3 = [def.origin[0] + h * a[0], def.origin[1] + h * a[1], def.origin[2] + h * a[2]];
      let w: Vec3 = [p[0] - f[0], p[1] - f[1], p[2] - f[2]];
      const l = length(w);
      w = l > 1e-300 ? [w[0] / l, w[1] / l, w[2] / l] : perpBasis(a)[0];
      return [f[0] + def.radius * w[0], f[1] + def.radius * w[1], f[2] + def.radius * w[2]];
    }
    case "cone": {
      const a = normalize(def.axis);
      const v: Vec3 = [p[0] - def.apex[0], p[1] - def.apex[1], p[2] - def.apex[2]];
      const h = dot(v, a);
      let w: Vec3 = [v[0] - h * a[0], v[1] - h * a[1], v[2] - h * a[2]];
      const rho = length(w);
      w = rho > 1e-300 ? [w[0] / rho, w[1] / rho, w[2] / rho] : perpBasis(a)[0];
      const ca = Math.cos(def.halfAngle), sa = Math.sin(def.halfAngle);
      const t = h * ca + rho * sa;
      if (t <= 0) return [def.apex[0], def.apex[1], def.apex[2]];
      const g: Vec3 = [ca * a[0] + sa * w[0], ca * a[1] + sa * w[1], ca * a[2] + sa * w[2]];
      return [def.apex[0] + t * g[0], def.apex[1] + t * g[1], def.apex[2] + t * g[2]];
    }
  }
}

/** Outward unit surface normal of the primitive at the foot point of p. */
export function primitiveNormalAt(def: PrimitiveDef, p: Vec3): Vec3 {
  switch (def.kind) {
    case "plane":
      return normalize(def.normal);
    case "sphere": {
      const v = normalize([p[0] - def.center[0], p[1] - def.center[1], p[2] - def.center[2]]);
      return length(v) > 0 ? v : [1, 0, 0];
    }
    case "cylinder": {
      const a = normalize(def.axis);
      const v: Vec3 = [p[0] - def.origin[0], p[1] - def.origin[1], p[2] - def.origin[2]];
      const h = dot(v, a);
      const w = normalize([v[0] - h * a[0], v[1] - h * a[1], v[2] - h * a[2]]);
      return length(w) > 0 ? w : perpBasis(a)[0];
    }
    case "cone": {
      const a = normalize(def.axis);
      const v: Vec3 = [p[0] - def.apex[0], p[1] - def.apex[1], p[2] - def.apex[2]];
      const h = dot(v, a);
      let w = normalize([v[0] - h * a[0], v[1] - h * a[1], v[2] - h * a[2]]);
      if (length(w) === 0) w = perpBasis(a)[0];
      const ca = Math.cos(def.halfAngle), sa = Math.sin(def.halfAngle);
      return [ca * w[0] - sa * a[0], ca * w[1] - sa * a[1], ca * w[2] - sa * a[2]];
    }
  }
}

/** Signed distances of the selected points to the primitive (selection order). */
export function residuals(def: PrimitiveDef, pos: PointArray, indices?: ArrayLike<number>): Float32Array {
  const f = distanceKernel(def);
  const count = selectionCount(pos, indices);
  const out = new Float32Array(count);
  for (let k = 0; k < count; k++) {
    const i = indices ? indices[k] : k;
    out[k] = f(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
  }
  return out;
}

/** RMS / max|r| / mean / count of a residual array (non-finite entries ignored). */
export function fitStats(res: ArrayLike<number>): FitStats {
  let ss = 0, s = 0, mx = 0, n = 0;
  for (let i = 0; i < res.length; i++) {
    const r = res[i];
    if (!Number.isFinite(r)) continue;
    ss += r * r;
    s += r;
    const a = Math.abs(r);
    if (a > mx) mx = a;
    n++;
  }
  return n === 0 ? { rms: NaN, maxAbs: NaN, mean: NaN, count: 0 } : { rms: Math.sqrt(ss / n), maxAbs: mx, mean: s / n, count: n };
}

/** Streaming residual statistics over the selection (optionally only where mask[k] != 0). */
export function primitiveStats(def: PrimitiveDef, pos: PointArray, indices?: ArrayLike<number>, mask?: Uint8Array): FitStats {
  const f = distanceKernel(def);
  const count = selectionCount(pos, indices);
  let ss = 0, s = 0, mx = 0, n = 0;
  for (let k = 0; k < count; k++) {
    if (mask && !mask[k]) continue;
    const i = indices ? indices[k] : k;
    const r = f(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
    if (!Number.isFinite(r)) continue;
    ss += r * r;
    s += r;
    const a = Math.abs(r);
    if (a > mx) mx = a;
    n++;
  }
  return n === 0 ? { rms: NaN, maxAbs: NaN, mean: NaN, count: 0 } : { rms: Math.sqrt(ss / n), maxAbs: mx, mean: s / n, count: n };
}

/** Inlier mask |r| <= threshold over the selection. */
function inlierMask(def: PrimitiveDef, pos: PointArray, indices: ArrayLike<number> | undefined, threshold: number): Uint8Array {
  const f = distanceKernel(def);
  const count = selectionCount(pos, indices);
  const out = new Uint8Array(count);
  for (let k = 0; k < count; k++) {
    const i = indices ? indices[k] : k;
    const r = f(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
    out[k] = Math.abs(r) <= threshold ? 1 : 0;
  }
  return out;
}

/** min/max of (x - o)·a over the selection (optionally masked). */
function axialRange(pos: PointArray, indices: ArrayLike<number> | undefined, o: Vec3, a: Vec3, mask?: Uint8Array): [number, number] {
  const count = selectionCount(pos, indices);
  let lo = Infinity, hi = -Infinity;
  for (let k = 0; k < count; k++) {
    if (mask && !mask[k]) continue;
    const i = indices ? indices[k] : k;
    const h = (pos[3 * i] - o[0]) * a[0] + (pos[3 * i + 1] - o[1]) * a[1] + (pos[3 * i + 2] - o[2]) * a[2];
    if (!Number.isFinite(h)) continue;
    if (h < lo) lo = h;
    if (h > hi) hi = h;
  }
  if (lo > hi) return [0, 0];
  return [lo, hi];
}

// ===========================================================================
// Normalisation helpers
// ===========================================================================

const toNorm = (p: Vec3, S: PointSample): Vec3 => [
  (p[0] - S.center[0]) / S.scale,
  (p[1] - S.center[1]) / S.scale,
  (p[2] - S.center[2]) / S.scale,
];
const toWorld = (p: Vec3, S: PointSample): Vec3 => [
  S.center[0] + S.scale * p[0],
  S.center[1] + S.scale * p[1],
  S.center[2] + S.scale * p[2],
];

/** Convert a primitive in normalised sample coordinates back to world coordinates. */
function defToWorld<T extends PrimitiveDef>(def: T, S: PointSample): T {
  switch (def.kind) {
    case "plane":
      return { ...def, origin: toWorld(def.origin, S) };
    case "sphere":
      return { ...def, center: toWorld(def.center, S), radius: def.radius * S.scale };
    case "cylinder":
      return { ...def, origin: toWorld(def.origin, S), radius: def.radius * S.scale, hMin: def.hMin * S.scale, hMax: def.hMax * S.scale };
    case "cone":
      return { ...def, apex: toWorld(def.apex, S), hMin: def.hMin * S.scale, hMax: def.hMax * S.scale };
  }
  return def;
}

function defToNorm<T extends PrimitiveDef>(def: T, S: PointSample): T {
  switch (def.kind) {
    case "plane":
      return { ...def, origin: toNorm(def.origin, S), normal: normalize(def.normal) };
    case "sphere":
      return { ...def, center: toNorm(def.center, S), radius: def.radius / S.scale };
    case "cylinder":
      return { ...def, origin: toNorm(def.origin, S), axis: normalize(def.axis), radius: def.radius / S.scale, hMin: def.hMin / S.scale, hMax: def.hMax / S.scale };
    case "cone":
      return { ...def, apex: toNorm(def.apex, S), axis: normalize(def.axis), hMin: def.hMin / S.scale, hMax: def.hMax / S.scale };
  }
  return def;
}

// ===========================================================================
// Robust driver
// ===========================================================================

interface RobustSolve<S> {
  state: S;
  /** robust residual scale (normalised units), NaN when not robust */
  sigma: number;
}

function solveMaybeRobust<S>(
  makeProblem: (w?: Float64Array) => LMProblem<S>,
  resid: (s: S, out: Float64Array) => void,
  m: number,
  s0: S,
  robust: boolean | undefined,
  lmOpts: LMOptions,
): RobustSolve<S> {
  let s = levenbergMarquardt(makeProblem(undefined), s0, lmOpts).state;
  if (!robust) return { state: s, sigma: NaN };
  const r = new Float64Array(m), w = new Float64Array(m);
  let sigma = NaN;
  for (let round = 0; round < 10; round++) {
    resid(s, r);
    const sig = tukeyWeights(r, m, w, 1e-9);
    s = levenbergMarquardt(makeProblem(w), s, lmOpts).state;
    if (Number.isFinite(sigma) && Math.abs(sig - sigma) <= 1e-4 * sigma) {
      sigma = sig;
      break;
    }
    sigma = sig;
  }
  resid(s, r);
  sigma = tukeyWeights(r, m, w, 1e-9);
  return { state: s, sigma };
}

function finishStats(
  def: PrimitiveDef,
  pos: PointArray,
  indices: ArrayLike<number> | undefined,
  robustSigmaWorld: number,
): { stats: FitStats; inliers?: Uint8Array } {
  if (Number.isFinite(robustSigmaWorld)) {
    const inliers = inlierMask(def, pos, indices, TUKEY_C * robustSigmaWorld);
    return { stats: primitiveStats(def, pos, indices, inliers), inliers };
  }
  return { stats: primitiveStats(def, pos, indices) };
}

// ===========================================================================
// Plane
// ===========================================================================

function weightedPlane(xs: Float64Array, m: number, w?: Float64Array): { n: Vec3; c: Vec3; values: [number, number, number] } {
  let sw = 0, cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < m; i++) {
    const wi = w ? w[i] : 1;
    sw += wi;
    cx += wi * xs[3 * i];
    cy += wi * xs[3 * i + 1];
    cz += wi * xs[3 * i + 2];
  }
  if (!(sw > 0)) throw new Error("fitPlane: all points rejected as outliers");
  cx /= sw;
  cy /= sw;
  cz /= sw;
  const cov = new Float64Array(9);
  for (let i = 0; i < m; i++) {
    const wi = w ? w[i] : 1;
    if (wi === 0) continue;
    const x = xs[3 * i] - cx, y = xs[3 * i + 1] - cy, z = xs[3 * i + 2] - cz;
    cov[0] += wi * x * x;
    cov[1] += wi * x * y;
    cov[2] += wi * x * z;
    cov[4] += wi * y * y;
    cov[5] += wi * y * z;
    cov[8] += wi * z * z;
  }
  cov[3] = cov[1];
  cov[6] = cov[2];
  cov[7] = cov[5];
  const e = eigenSym3(cov);
  return { n: e.vectors[0], c: [cx, cy, cz], values: e.values };
}

function meanNormal(ns: Float64Array, m: number): Vec3 {
  let x = 0, y = 0, z = 0;
  for (let i = 0; i < m; i++) {
    x += ns[3 * i];
    y += ns[3 * i + 1];
    z += ns[3 * i + 2];
  }
  return [x, y, z];
}

/** Least-squares plane (PCA) with optional robust (Tukey IRLS) refinement. origin = selection centroid projected. */
export function fitPlane(pos: PointArray, opts: FitOptions = {}): FitResult<PlaneDef> {
  const S = buildSample(pos, opts, 3, "fitPlane");
  let { n, c, values } = weightedPlane(S.xs, S.m);
  if (!(values[1] > 1e-12 * values[2])) throw new Error("fitPlane: points are collinear — the plane is undefined");
  let sigma = NaN;
  if (opts.robust) {
    const r = new Float64Array(S.m), w = new Float64Array(S.m);
    const maxRounds = opts.maxIterations ?? 30;
    for (let round = 0; round < maxRounds; round++) {
      for (let i = 0; i < S.m; i++) r[i] = (S.xs[3 * i] - c[0]) * n[0] + (S.xs[3 * i + 1] - c[1]) * n[1] + (S.xs[3 * i + 2] - c[2]) * n[2];
      sigma = tukeyWeights(r, S.m, w, 1e-9);
      const prev = { n, c };
      ({ n, c } = weightedPlane(S.xs, S.m, w));
      const moved = Math.abs(dot([c[0] - prev.c[0], c[1] - prev.c[1], c[2] - prev.c[2]], n));
      if (1 - Math.abs(dot(n, prev.n)) < 1e-15 && moved < 1e-12) break;
    }
    for (let i = 0; i < S.m; i++) r[i] = (S.xs[3 * i] - c[0]) * n[0] + (S.xs[3 * i + 1] - c[1]) * n[1] + (S.xs[3 * i + 2] - c[2]) * n[2];
    sigma = tukeyWeights(r, S.m, w, 1e-9);
  }
  let normal: Vec3 = n;
  if (S.ns) {
    if (dot(meanNormal(S.ns, S.m), normal) < 0) normal = [-normal[0], -normal[1], -normal[2]];
  } else normal = canonicalSign(normal);
  const onPlane = toWorld(c, S);
  const centroid = selectionCentroid(pos, opts.indices);
  const d = dot([centroid[0] - onPlane[0], centroid[1] - onPlane[1], centroid[2] - onPlane[2]], normal);
  const origin: Vec3 = [centroid[0] - d * normal[0], centroid[1] - d * normal[1], centroid[2] - d * normal[2]];
  const def: PlaneDef = { kind: "plane", origin, normal };
  return { def, ...finishStats(def, pos, opts.indices, sigma * S.scale) };
}

/** Plane with a prescribed normal direction (parallel/perpendicular constraints): only the offset is fitted. */
export function fitPlaneWithNormal(pos: PointArray, normal: Vec3, opts: FitOptions = {}): FitResult<PlaneDef> {
  const n = normalize(normal);
  if (length(n) === 0) throw new Error("fitPlaneWithNormal: zero normal");
  const S = buildSample(pos, opts, 1, "fitPlaneWithNormal");
  const r = new Float64Array(S.m);
  for (let i = 0; i < S.m; i++) r[i] = S.xs[3 * i] * n[0] + S.xs[3 * i + 1] * n[1] + S.xs[3 * i + 2] * n[2];
  let off = 0;
  for (let i = 0; i < S.m; i++) off += r[i];
  off /= S.m;
  let sigma = NaN;
  if (opts.robust) {
    const res = new Float64Array(S.m), w = new Float64Array(S.m);
    for (let round = 0; round < (opts.maxIterations ?? 30); round++) {
      for (let i = 0; i < S.m; i++) res[i] = r[i] - off;
      sigma = tukeyWeights(res, S.m, w, 1e-9);
      let sw = 0, s = 0;
      for (let i = 0; i < S.m; i++) {
        sw += w[i];
        s += w[i] * r[i];
      }
      if (!(sw > 0)) break;
      const next = s / sw;
      const done = Math.abs(next - off) < 1e-13;
      off = next;
      if (done) break;
    }
    for (let i = 0; i < S.m; i++) res[i] = r[i] - off;
    sigma = tukeyWeights(res, S.m, w, 1e-9);
  }
  const onPlane = toWorld([n[0] * off, n[1] * off, n[2] * off], S);
  const centroid = selectionCentroid(pos, opts.indices);
  const d = dot([centroid[0] - onPlane[0], centroid[1] - onPlane[1], centroid[2] - onPlane[2]], n);
  const def: PlaneDef = { kind: "plane", origin: [centroid[0] - d * n[0], centroid[1] - d * n[1], centroid[2] - d * n[2]], normal: n };
  return { def, ...finishStats(def, pos, opts.indices, sigma * S.scale) };
}

/**
 * Plane whose normal is perpendicular to `direction` — i.e. the plane is parallel to / contains
 * that direction (perpendicular to a reference plane with normal `direction`, or parallel to a
 * reference axis). With `throughPoint` the plane also passes through that point (direction +
 * point = a plane through a reference line).
 */
export function fitPlanePerpendicular(
  pos: PointArray,
  direction: Vec3,
  opts: FitOptions & { throughPoint?: Vec3 } = {},
): FitResult<PlaneDef> {
  const dir = normalize(direction);
  if (length(dir) === 0) throw new Error("fitPlanePerpendicular: zero direction");
  const S = buildSample(pos, opts, 2, "fitPlanePerpendicular");
  const [e1, e2] = perpBasis(dir);
  const fixedC = opts.throughPoint ? toNorm(opts.throughPoint, S) : null;
  const solve = (w?: Float64Array): { n: Vec3; c: Vec3; spread: number } => {
    let c: Vec3;
    if (fixedC) c = fixedC;
    else {
      let sw = 0, cx = 0, cy = 0, cz = 0;
      for (let i = 0; i < S.m; i++) {
        const wi = w ? w[i] : 1;
        sw += wi;
        cx += wi * S.xs[3 * i];
        cy += wi * S.xs[3 * i + 1];
        cz += wi * S.xs[3 * i + 2];
      }
      if (!(sw > 0)) throw new Error("fitPlanePerpendicular: all points rejected as outliers");
      c = [cx / sw, cy / sw, cz / sw];
    }
    let a = 0, b = 0, d = 0;
    for (let i = 0; i < S.m; i++) {
      const wi = w ? w[i] : 1;
      const x = S.xs[3 * i] - c[0], y = S.xs[3 * i + 1] - c[1], z = S.xs[3 * i + 2] - c[2];
      const u = x * e1[0] + y * e1[1] + z * e1[2], v = x * e2[0] + y * e2[1] + z * e2[2];
      a += wi * u * u;
      b += wi * u * v;
      d += wi * v * v;
    }
    // smallest eigenvector of [[a, b], [b, d]]
    const th = 0.5 * Math.atan2(2 * b, a - d) + Math.PI / 2;
    const cu = Math.cos(th), su = Math.sin(th);
    const n = normalize([cu * e1[0] + su * e2[0], cu * e1[1] + su * e2[1], cu * e1[2] + su * e2[2]]);
    return { n, c, spread: (a + d) / Math.max(1e-300, w ? w.reduce((s, v) => s + v, 0) : S.m) };
  };
  let { n, c, spread } = solve();
  // all points on one line parallel to `direction` (through the point): the plane is undefined
  if (!(spread > 1e-20)) throw new Error("fitPlanePerpendicular: points lie on a line parallel to the direction — plane undefined");
  let sigma = NaN;
  if (opts.robust) {
    const r = new Float64Array(S.m), w = new Float64Array(S.m);
    for (let round = 0; round < (opts.maxIterations ?? 30); round++) {
      for (let i = 0; i < S.m; i++) r[i] = (S.xs[3 * i] - c[0]) * n[0] + (S.xs[3 * i + 1] - c[1]) * n[1] + (S.xs[3 * i + 2] - c[2]) * n[2];
      sigma = tukeyWeights(r, S.m, w, 1e-9);
      const prev = n;
      ({ n, c } = solve(w));
      if (1 - Math.abs(dot(prev, n)) < 1e-15) break;
    }
    for (let i = 0; i < S.m; i++) r[i] = (S.xs[3 * i] - c[0]) * n[0] + (S.xs[3 * i + 1] - c[1]) * n[1] + (S.xs[3 * i + 2] - c[2]) * n[2];
    sigma = tukeyWeights(r, S.m, w, 1e-9);
  }
  let normal: Vec3 = n;
  if (S.ns) {
    if (dot(meanNormal(S.ns, S.m), normal) < 0) normal = [-normal[0], -normal[1], -normal[2]];
  } else normal = canonicalSign(normal);
  const onPlane = toWorld(c, S);
  let origin: Vec3;
  if (opts.throughPoint) origin = [opts.throughPoint[0], opts.throughPoint[1], opts.throughPoint[2]];
  else {
    const centroid = selectionCentroid(pos, opts.indices);
    const dd = dot([centroid[0] - onPlane[0], centroid[1] - onPlane[1], centroid[2] - onPlane[2]], normal);
    origin = [centroid[0] - dd * normal[0], centroid[1] - dd * normal[1], centroid[2] - dd * normal[2]];
  }
  const def: PlaneDef = { kind: "plane", origin, normal };
  return { def, ...finishStats(def, pos, opts.indices, sigma * S.scale) };
}

// ===========================================================================
// Sphere
// ===========================================================================

/** Algebraic sphere x²+y²+z² + Dx + Ey + Fz + G = 0 (normalised coordinates). */
function algebraicSphere(xs: Float64Array, m: number): Float64Array | null {
  const A = new Float64Array(16), b = new Float64Array(4);
  const row = new Float64Array(4);
  for (let i = 0; i < m; i++) {
    const x = xs[3 * i], y = xs[3 * i + 1], z = xs[3 * i + 2];
    row[0] = x;
    row[1] = y;
    row[2] = z;
    row[3] = 1;
    const rhs = -(x * x + y * y + z * z);
    for (let a = 0; a < 4; a++) {
      b[a] += row[a] * rhs;
      for (let c = 0; c < 4; c++) A[a * 4 + c] += row[a] * row[c];
    }
  }
  // conditioning check: the 4x4 normal matrix of coplanar points is singular
  const e = eigenSym3([A[0], A[1], A[2], A[4], A[5], A[6], A[8], A[9], A[10]]);
  if (!(e.values[0] > 1e-10 * e.values[2])) return null;
  const s = solveLinear(A, b, 4);
  if (!s) return null;
  const cx = -s[0] / 2, cy = -s[1] / 2, cz = -s[2] / 2;
  const r2 = cx * cx + cy * cy + cz * cz - s[3];
  if (!(r2 > 0) || !Number.isFinite(r2)) return null;
  return Float64Array.of(cx, cy, cz, Math.sqrt(r2));
}

function sphereProblem(xs: Float64Array, m: number, w?: Float64Array): LMProblem<Float64Array> {
  return {
    nParams: 4,
    linearize(s, JtJ, Jtr) {
      const J = new Float64Array(4);
      J[3] = -1;
      let c = 0;
      for (let i = 0; i < m; i++) {
        const dx = xs[3 * i] - s[0], dy = xs[3 * i + 1] - s[1], dz = xs[3 * i + 2] - s[2];
        const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const r = d - s[3];
        const wi = w ? w[i] : 1;
        c += wi * r * r;
        if (d > 1e-300) {
          J[0] = -dx / d;
          J[1] = -dy / d;
          J[2] = -dz / d;
        } else J[0] = J[1] = J[2] = 0;
        accumulateNormal(JtJ, Jtr, J, 4, wi, r);
      }
      return c;
    },
    cost(s) {
      let c = 0;
      for (let i = 0; i < m; i++) {
        const dx = xs[3 * i] - s[0], dy = xs[3 * i + 1] - s[1], dz = xs[3 * i + 2] - s[2];
        const r = Math.sqrt(dx * dx + dy * dy + dz * dz) - s[3];
        c += (w ? w[i] : 1) * r * r;
      }
      return c;
    },
    retract: (s, d) => Float64Array.of(s[0] + d[0], s[1] + d[1], s[2] + d[2], s[3] + d[3]),
  };
}

function sphereResiduals(xs: Float64Array, m: number, s: Float64Array, out: Float64Array): void {
  for (let i = 0; i < m; i++) out[i] = Math.hypot(xs[3 * i] - s[0], xs[3 * i + 1] - s[1], xs[3 * i + 2] - s[2]) - s[3];
}

function fitSphereImpl(pos: PointArray, opts: SphereFitOptions, fixedRadius?: number): FitResult<SphereDef> {
  const S = buildSample(pos, opts, fixedRadius !== undefined ? 3 : 4, "fitSphere");
  let s0: Float64Array | null;
  if (opts.initial) {
    const c = toNorm(opts.initial.center, S);
    s0 = Float64Array.of(c[0], c[1], c[2], opts.initial.radius / S.scale);
  } else {
    s0 = algebraicSphere(S.xs, S.m);
    if (!s0 && fixedRadius !== undefined) {
      // coplanar points with a known radius: start above the best-fit plane
      const pl = weightedPlane(S.xs, S.m);
      s0 = Float64Array.of(pl.c[0] - pl.n[0] * 0.5, pl.c[1] - pl.n[1] * 0.5, pl.c[2] - pl.n[2] * 0.5, 1);
    }
    if (!s0) throw new Error("fitSphere: points are (nearly) coplanar — a sphere is undefined (fit a plane or circle instead)");
  }
  if (fixedRadius !== undefined) {
    if (!(fixedRadius > 0)) throw new Error("fitSphere: radius must be positive");
    s0[3] = fixedRadius / S.scale;
  }
  const lmOpts: LMOptions = { maxIterations: opts.maxIterations ?? 100, fixed: fixedRadius !== undefined ? [false, false, false, true] : undefined };
  const sol = solveMaybeRobust(
    (w) => sphereProblem(S.xs, S.m, w),
    (s, out) => sphereResiduals(S.xs, S.m, s, out),
    S.m,
    s0,
    opts.robust,
    lmOpts,
  );
  const s = sol.state;
  const def: SphereDef = defToWorld({ kind: "sphere", center: [s[0], s[1], s[2]], radius: Math.abs(s[3]) }, S);
  return { def, ...finishStats(def, pos, opts.indices, sol.sigma * S.scale) };
}

/** Sphere: algebraic initialisation + geometric Levenberg–Marquardt. */
export function fitSphere(pos: PointArray, opts: SphereFitOptions = {}): FitResult<SphereDef> {
  return fitSphereImpl(pos, opts);
}

/** Sphere with a prescribed radius (only the centre is fitted). */
export function fitSphereWithRadius(pos: PointArray, radius: number, opts: SphereFitOptions = {}): FitResult<SphereDef> {
  return fitSphereImpl(pos, opts, radius);
}

// ===========================================================================
// Normal estimation for normal-less samples
// ===========================================================================

/** Unoriented PCA normals at up to `maxQuery` sample points (neighbours searched in up to 60k sample points). */
function estimateSampleNormals(S: PointSample, maxQuery = 3000, k = 24): PointSample {
  const base = subsample(S, 60000, 3);
  const Q = subsample(S, maxQuery, 5);
  const grid = new GridIndex(base.xs, base.m);
  const ns = new Float64Array(3 * Q.m);
  const kk = Math.min(k, Math.max(3, base.m - 1));
  const si = new Int32Array(kk), sd = new Float64Array(kk);
  for (let i = 0; i < Q.m; i++) pcaNormalAt(grid, Q.xs[3 * i], Q.xs[3 * i + 1], Q.xs[3 * i + 2], kk, ns, 3 * i, si, sd);
  return { ...Q, ns };
}

/** Sample carrying normals: given ones, or estimated (unoriented). */
function samplesWithNormals(S: PointSample, maxQuery = 3000): PointSample {
  if (S.ns) return subsample(S, maxQuery, 5);
  return estimateSampleNormals(S, maxQuery);
}

// ===========================================================================
// Cylinder
// ===========================================================================

interface CylState {
  /** point on the axis (kept at the projection of the sample centroid = origin) */
  p: Vec3;
  /** unit axis */
  d: Vec3;
  r: number;
}

function cylinderProblem(xs: Float64Array, m: number, w?: Float64Array): LMProblem<CylState> {
  const cost = (s: CylState) => {
    const [px, py, pz] = s.p, [dx, dy, dz] = s.d;
    let c = 0;
    for (let i = 0; i < m; i++) {
      const qx = xs[3 * i] - px, qy = xs[3 * i + 1] - py, qz = xs[3 * i + 2] - pz;
      const h = qx * dx + qy * dy + qz * dz;
      const wx = qx - h * dx, wy = qy - h * dy, wz = qz - h * dz;
      const r = Math.sqrt(wx * wx + wy * wy + wz * wz) - s.r;
      c += (w ? w[i] : 1) * r * r;
    }
    return c;
  };
  return {
    nParams: 5,
    linearize(s, JtJ, Jtr) {
      const [e1, e2] = perpBasis(s.d);
      const [px, py, pz] = s.p, [dx, dy, dz] = s.d;
      const J = new Float64Array(5);
      J[4] = -1;
      let c = 0;
      for (let i = 0; i < m; i++) {
        const qx = xs[3 * i] - px, qy = xs[3 * i + 1] - py, qz = xs[3 * i + 2] - pz;
        const h = qx * dx + qy * dy + qz * dz;
        const wx = qx - h * dx, wy = qy - h * dy, wz = qz - h * dz;
        const rho = Math.sqrt(wx * wx + wy * wy + wz * wz);
        const r = rho - s.r;
        const wi = w ? w[i] : 1;
        c += wi * r * r;
        let n1 = 0, n2 = 0;
        if (rho > 1e-300) {
          n1 = (wx * e1[0] + wy * e1[1] + wz * e1[2]) / rho;
          n2 = (wx * e2[0] + wy * e2[1] + wz * e2[2]) / rho;
        }
        J[0] = -h * n1;
        J[1] = -h * n2;
        J[2] = -n1;
        J[3] = -n2;
        accumulateNormal(JtJ, Jtr, J, 5, wi, r);
      }
      return c;
    },
    cost,
    retract(s, dl) {
      const [e1, e2] = perpBasis(s.d);
      const d = normalize([s.d[0] + dl[0] * e1[0] + dl[1] * e2[0], s.d[1] + dl[0] * e1[1] + dl[1] * e2[1], s.d[2] + dl[0] * e1[2] + dl[1] * e2[2]]);
      const p: Vec3 = [s.p[0] + dl[2] * e1[0] + dl[3] * e2[0], s.p[1] + dl[2] * e1[1] + dl[3] * e2[1], s.p[2] + dl[2] * e1[2] + dl[3] * e2[2]];
      const t = dot(p, d);
      return { p: [p[0] - t * d[0], p[1] - t * d[1], p[2] - t * d[2]], d, r: s.r + dl[4] };
    },
  };
}

function cylResiduals(xs: Float64Array, m: number, s: CylState, out: Float64Array): void {
  const [px, py, pz] = s.p, [dx, dy, dz] = s.d;
  for (let i = 0; i < m; i++) {
    const qx = xs[3 * i] - px, qy = xs[3 * i + 1] - py, qz = xs[3 * i + 2] - pz;
    const h = qx * dx + qy * dy + qz * dz;
    out[i] = Math.hypot(qx - h * dx, qy - h * dy, qz - h * dz) - s.r;
  }
}

/** Project sample points onto the plane ⟂ d (packed 2D in the perpBasis of d). */
function projectPerp(xs: Float64Array, m: number, d: Vec3, subset?: ArrayLike<number>): Float64Array {
  const [e1, e2] = perpBasis(d);
  const cnt = subset ? subset.length : m;
  const out = new Float64Array(2 * cnt);
  for (let t = 0; t < cnt; t++) {
    const i = subset ? subset[t] : t;
    const x = xs[3 * i], y = xs[3 * i + 1], z = xs[3 * i + 2];
    out[2 * t] = x * e1[0] + y * e1[1] + z * e1[2];
    out[2 * t + 1] = x * e2[0] + y * e2[1] + z * e2[2];
  }
  return out;
}

/** RMS radial residual of the algebraic circle of the projection along d (Infinity if none). */
function circleScore(xs: Float64Array, m: number, d: Vec3): number {
  const uv = projectPerp(xs, m, d);
  const c = kasaCircle(uv, m);
  if (!c) return Infinity;
  let ss = 0;
  for (let i = 0; i < m; i++) {
    const r = Math.hypot(uv[2 * i] - c.cx, uv[2 * i + 1] - c.cy) - c.r;
    ss += r * r;
  }
  return Math.sqrt(ss / m);
}

/** Cylinder state for a given axis direction: circle fit of the perpendicular projection. */
function cylinderFromAxis(xs: Float64Array, m: number, d: Vec3, geometric = true): CylState | null {
  const uv = projectPerp(xs, m, d);
  let c = kasaCircle(uv, m);
  if (!c) return null;
  if (geometric) c = refineCircle(uv, m, c, { maxIterations: 30 });
  const [e1, e2] = perpBasis(d);
  const p: Vec3 = [c.cx * e1[0] + c.cy * e2[0], c.cx * e1[1] + c.cy * e2[1], c.cx * e1[2] + c.cy * e2[2]];
  return { p, d, r: c.r };
}

function angleDist(a: Vec3, b: Vec3): number {
  return Math.acos(Math.min(1, Math.abs(dot(a, b))));
}

/** Local pattern search on the axis direction minimising the circle score. */
function refineAxisDirection(xs: Float64Array, m: number, d0: Vec3, s0: number): { d: Vec3; score: number } {
  let d = d0, best = s0, step = 0.08;
  for (let guard = 0; guard < 80 && step > 0.002; guard++) {
    const [e1, e2] = perpBasis(d);
    let improved = false;
    for (const [a, b] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) {
      const c = normalize([d[0] + step * (a * e1[0] + b * e2[0]), d[1] + step * (a * e1[1] + b * e2[1]), d[2] + step * (a * e1[2] + b * e2[2])]);
      const sc = circleScore(xs, m, c);
      if (sc < best) {
        best = sc;
        d = c;
        improved = true;
        break;
      }
    }
    if (!improved) step /= 2;
  }
  return { d, score: best };
}

/** Candidate cylinder initialisations (normal-based axis + PCA + direction sampling). */
function cylinderCandidates(S: PointSample): CylState[] {
  const scoreSet = subsample(S, 1200, 11);
  const fitSet = subsample(S, 20000, 13);
  const dirs: Vec3[] = [];
  // axis from normals: eigenvector of Σ n nᵀ with the smallest eigenvalue
  const NS = samplesWithNormals(S, 3000);
  if (NS.ns) {
    const M = new Float64Array(9);
    for (let i = 0; i < NS.m; i++) {
      const x = NS.ns[3 * i], y = NS.ns[3 * i + 1], z = NS.ns[3 * i + 2];
      M[0] += x * x;
      M[1] += x * y;
      M[2] += x * z;
      M[4] += y * y;
      M[5] += y * z;
      M[8] += z * z;
    }
    M[3] = M[1];
    M[6] = M[2];
    M[7] = M[5];
    const e = eigenSym3(M);
    if (e.values[2] > 0) dirs.push(e.vectors[0]);
  }
  const preferred = dirs.length;
  // PCA directions of the points
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
  const pca = eigenSym3(cov);
  dirs.push(pca.vectors[2], pca.vectors[1], pca.vectors[0]);
  for (const d of fibonacciHemisphere(160)) dirs.push(d);
  const scored = dirs.map((d, i) => ({ d, i, score: circleScore(scoreSet.xs, scoreSet.m, d) }));
  const chosen: { d: Vec3; score: number }[] = [];
  // always keep the normal-based direction
  for (let i = 0; i < preferred; i++) if (Number.isFinite(scored[i].score)) chosen.push(scored[i]);
  scored.sort((a, b) => a.score - b.score);
  for (const c of scored) {
    if (chosen.length >= preferred + 3) break;
    if (!Number.isFinite(c.score)) continue;
    if (chosen.some((o) => angleDist(o.d, c.d) < 0.17)) continue;
    chosen.push(c);
  }
  const out: CylState[] = [];
  for (const c of chosen) {
    const r = refineAxisDirection(scoreSet.xs, scoreSet.m, c.d, c.score);
    const st = cylinderFromAxis(fitSet.xs, fitSet.m, r.d);
    if (st && Number.isFinite(st.r)) out.push(st);
  }
  return out;
}

/** Run LM from several initial states on a small sub-sample and return the best one. */
function pickBest<St>(states: St[], S: PointSample, make: (xs: Float64Array, m: number) => LMProblem<St>, fixed?: boolean[]): St {
  if (states.length === 1) return states[0];
  const sub = subsample(S, 4000, 17);
  const prob = make(sub.xs, sub.m);
  let best = states[0], bestCost = Infinity;
  for (const s of states) {
    const r = levenbergMarquardt(prob, s, { maxIterations: 40, fixed });
    if (Number.isFinite(r.cost) && r.cost < bestCost) {
      bestCost = r.cost;
      best = r.state;
    }
  }
  return best;
}

interface CylinderConstraints {
  axis?: Vec3;
  axisPoint?: Vec3;
  radius?: number;
}

function cylStateFromDef(def: CylinderDef, S: PointSample): CylState {
  const d = normalize(def.axis);
  const o = toNorm(def.origin, S);
  const t = dot(o, d);
  return { p: [o[0] - t * d[0], o[1] - t * d[1], o[2] - t * d[2]], d, r: def.radius / S.scale };
}

/** Solve the cylinder in normalised coordinates; returns state + robust sigma. */
function solveCylinder(S: PointSample, opts: CylinderFitOptions, cons: CylinderConstraints): RobustSolve<CylState> {
  const fixed = [!!cons.axis, !!cons.axis, !!(cons.axis && cons.axisPoint), !!(cons.axis && cons.axisPoint), cons.radius !== undefined];
  const rFix = cons.radius !== undefined ? cons.radius / S.scale : undefined;
  if (rFix !== undefined && !(rFix > 0)) throw new Error("fitCylinder: radius must be positive");
  let cands: CylState[] = [];
  if (cons.axis) {
    const d = normalize(cons.axis);
    if (length(d) === 0) throw new Error("fitCylinder: zero axis direction");
    if (cons.axisPoint) {
      const o = toNorm(cons.axisPoint, S);
      const t = dot(o, d);
      const p: Vec3 = [o[0] - t * d[0], o[1] - t * d[1], o[2] - t * d[2]];
      let rr = 0;
      const tmp = new Float64Array(S.m);
      cylResiduals(S.xs, S.m, { p, d, r: 0 }, tmp);
      for (let i = 0; i < S.m; i++) rr += tmp[i];
      cands = [{ p, d, r: rr / S.m }];
    } else {
      const st = cylinderFromAxis(S.m > 20000 ? subsample(S, 20000, 13).xs : S.xs, Math.min(S.m, 20000), d);
      if (!st) throw new Error("fitCylinder: the projection along the given axis is degenerate (collinear) — no circle");
      cands = [st];
    }
  } else if (opts.initial) {
    cands = [cylStateFromDef(opts.initial, S)];
  } else {
    cands = cylinderCandidates(S);
  }
  if (rFix !== undefined) for (const c of cands) c.r = rFix;
  if (!cands.length) throw new Error("fitCylinder: could not initialise — data too degenerate (planar/collinear?)");
  const start = pickBest(cands, S, (xs, m) => cylinderProblem(xs, m), fixed);
  return solveMaybeRobust(
    (w) => cylinderProblem(S.xs, S.m, w),
    (s, out) => cylResiduals(S.xs, S.m, s, out),
    S.m,
    start,
    opts.robust,
    { maxIterations: opts.maxIterations ?? 100, fixed },
  );
}

function finishCylinder(S: PointSample, sol: RobustSolve<CylState>, pos: PointArray, opts: FitOptions): FitResult<CylinderDef> {
  const s = sol.state;
  const axis = canonicalSign(s.d);
  const origin = toWorld(s.p, S);
  const radius = Math.abs(s.r) * S.scale;
  if (!Number.isFinite(radius) || !Number.isFinite(origin[0] + origin[1] + origin[2])) throw new Error("fitCylinder: did not converge (non-finite result)");
  const proto: CylinderDef = { kind: "cylinder", origin, axis, radius, hMin: 0, hMax: 0 };
  const fs = finishStats(proto, pos, opts.indices, sol.sigma * S.scale);
  const [hMin, hMax] = axialRange(pos, opts.indices, origin, axis, fs.inliers);
  const def: CylinderDef = { ...proto, hMin, hMax };
  return { def, ...fs };
}

/**
 * Cylinder: axis initialised from the normals' covariance (or, without normals, from PCA +
 * direction sampling with circle scoring), then geometric LM with a minimal parametrisation
 * (2 axis angles, 2 axis-point coordinates ⟂ axis, radius). Works on partial patches (≳45° arc).
 * origin = axis point closest to the data centroid, hMin/hMax = data extent along the axis.
 */
export function fitCylinder(pos: PointArray, opts: CylinderFitOptions = {}): FitResult<CylinderDef> {
  const S = buildSample(pos, opts, 5, "fitCylinder");
  return finishCylinder(S, solveCylinder(S, opts, {}), pos, opts);
}

/** Cylinder with a fixed axis direction (e.g. ⟂ to a reference plane); optionally a fixed axis point (coaxial). */
export function fitCylinderWithAxis(
  pos: PointArray,
  axis: Vec3,
  opts: CylinderFitOptions & { axisPoint?: Vec3; radius?: number } = {},
): FitResult<CylinderDef> {
  const S = buildSample(pos, opts, 3, "fitCylinderWithAxis");
  return finishCylinder(S, solveCylinder(S, opts, { axis, axisPoint: opts.axisPoint, radius: opts.radius }), pos, opts);
}

/** Cylinder with a prescribed radius (axis and position fitted). */
export function fitCylinderWithRadius(pos: PointArray, radius: number, opts: CylinderFitOptions = {}): FitResult<CylinderDef> {
  const S = buildSample(pos, opts, 4, "fitCylinderWithRadius");
  return finishCylinder(S, solveCylinder(S, opts, { radius }), pos, opts);
}

// ===========================================================================
// Cone
// ===========================================================================

interface ConeState {
  /** axis point (kept at the projection of the sample centroid) */
  p: Vec3;
  /** unit axis */
  d: Vec3;
  /** radius at p */
  r0: number;
  /** signed half angle: radius(h) = r0 + h tan(a), h along d from p */
  a: number;
}

const MAX_CONE_ANGLE = 1.45; // ~83°

function coneProblem(xs: Float64Array, m: number, w?: Float64Array): LMProblem<ConeState> {
  const cost = (s: ConeState) => {
    const [px, py, pz] = s.p, [dx, dy, dz] = s.d;
    const ca = Math.cos(s.a), sa = Math.sin(s.a);
    let c = 0;
    for (let i = 0; i < m; i++) {
      const qx = xs[3 * i] - px, qy = xs[3 * i + 1] - py, qz = xs[3 * i + 2] - pz;
      const h = qx * dx + qy * dy + qz * dz;
      const wx = qx - h * dx, wy = qy - h * dy, wz = qz - h * dz;
      const rho = Math.sqrt(wx * wx + wy * wy + wz * wz);
      const e = (rho - s.r0) * ca - h * sa;
      c += (w ? w[i] : 1) * e * e;
    }
    return c;
  };
  return {
    nParams: 6,
    linearize(s, JtJ, Jtr) {
      const [e1, e2] = perpBasis(s.d);
      const [px, py, pz] = s.p, [dx, dy, dz] = s.d;
      const ca = Math.cos(s.a), sa = Math.sin(s.a);
      const J = new Float64Array(6);
      J[4] = -ca;
      let c = 0;
      for (let i = 0; i < m; i++) {
        const qx = xs[3 * i] - px, qy = xs[3 * i + 1] - py, qz = xs[3 * i + 2] - pz;
        const h = qx * dx + qy * dy + qz * dz;
        const wx = qx - h * dx, wy = qy - h * dy, wz = qz - h * dz;
        const rho = Math.sqrt(wx * wx + wy * wy + wz * wz);
        const e = (rho - s.r0) * ca - h * sa;
        const wi = w ? w[i] : 1;
        c += wi * e * e;
        let n1 = 0, n2 = 0;
        if (rho > 1e-300) {
          n1 = (wx * e1[0] + wy * e1[1] + wz * e1[2]) / rho;
          n2 = (wx * e2[0] + wy * e2[1] + wz * e2[2]) / rho;
        }
        const g = h * ca + rho * sa;
        J[0] = -n1 * g;
        J[1] = -n2 * g;
        J[2] = -ca * n1;
        J[3] = -ca * n2;
        J[5] = -(rho - s.r0) * sa - h * ca;
        accumulateNormal(JtJ, Jtr, J, 6, wi, e);
      }
      return c;
    },
    cost,
    retract(s, dl) {
      const [e1, e2] = perpBasis(s.d);
      const d = normalize([s.d[0] + dl[0] * e1[0] + dl[1] * e2[0], s.d[1] + dl[0] * e1[1] + dl[1] * e2[1], s.d[2] + dl[0] * e1[2] + dl[1] * e2[2]]);
      const p: Vec3 = [s.p[0] + dl[2] * e1[0] + dl[3] * e2[0], s.p[1] + dl[2] * e1[1] + dl[3] * e2[1], s.p[2] + dl[2] * e1[2] + dl[3] * e2[2]];
      const a = Math.max(-MAX_CONE_ANGLE, Math.min(MAX_CONE_ANGLE, s.a + dl[5]));
      const r0 = s.r0 + dl[4];
      // slide p along the new axis to the projection of the centroid (origin); radius follows the cone
      const t = -dot(p, d);
      return { p: [p[0] + t * d[0], p[1] + t * d[1], p[2] + t * d[2]], d, r0: r0 + t * Math.tan(a), a };
    },
  };
}

function coneResiduals(xs: Float64Array, m: number, s: ConeState, out: Float64Array): void {
  const [px, py, pz] = s.p, [dx, dy, dz] = s.d;
  const ca = Math.cos(s.a), sa = Math.sin(s.a);
  for (let i = 0; i < m; i++) {
    const qx = xs[3 * i] - px, qy = xs[3 * i + 1] - py, qz = xs[3 * i + 2] - pz;
    const h = qx * dx + qy * dy + qz * dz;
    const rho = Math.hypot(qx - h * dx, qy - h * dy, qz - h * dz);
    out[i] = (rho - s.r0) * ca - h * sa;
  }
}

/** Cone initialisation from (possibly unoriented) normals: apex = LS intersection of tangent planes. */
function coneFromNormals(xs: Float64Array, ns: Float64Array, m: number): ConeState | null {
  const M = new Float64Array(9), b = new Float64Array(3);
  for (let i = 0; i < m; i++) {
    const nx = ns[3 * i], ny = ns[3 * i + 1], nz = ns[3 * i + 2];
    const nd = nx * xs[3 * i] + ny * xs[3 * i + 1] + nz * xs[3 * i + 2];
    M[0] += nx * nx;
    M[1] += nx * ny;
    M[2] += nx * nz;
    M[4] += ny * ny;
    M[5] += ny * nz;
    M[8] += nz * nz;
    b[0] += nx * nd;
    b[1] += ny * nd;
    b[2] += nz * nd;
  }
  M[3] = M[1];
  M[6] = M[2];
  M[7] = M[5];
  const e = eigenSym3(M);
  if (!(e.values[0] > 2e-3 * e.values[2])) return null; // normals do not span 3D → (near) cylinder / plane
  const apexS = solveLinear(M, b, 3);
  if (!apexS) return null;
  const apex: Vec3 = [apexS[0], apexS[1], apexS[2]];
  if (length(apex) > 1e3) return null;
  const C = new Float64Array(9);
  for (let i = 0; i < m; i++) {
    const u = normalize([xs[3 * i] - apex[0], xs[3 * i + 1] - apex[1], xs[3 * i + 2] - apex[2]]);
    const mm = cross(u, [ns[3 * i], ns[3 * i + 1], ns[3 * i + 2]]);
    C[0] += mm[0] * mm[0];
    C[1] += mm[0] * mm[1];
    C[2] += mm[0] * mm[2];
    C[4] += mm[1] * mm[1];
    C[5] += mm[1] * mm[2];
    C[8] += mm[2] * mm[2];
  }
  C[3] = C[1];
  C[6] = C[2];
  C[7] = C[5];
  let d = eigenSym3(C).vectors[0];
  let side = 0;
  for (let i = 0; i < m; i++) side += (xs[3 * i] - apex[0]) * d[0] + (xs[3 * i + 1] - apex[1]) * d[1] + (xs[3 * i + 2] - apex[2]) * d[2];
  if (side < 0) d = [-d[0], -d[1], -d[2]];
  let sumA = 0;
  for (let i = 0; i < m; i++) {
    const u = normalize([xs[3 * i] - apex[0], xs[3 * i + 1] - apex[1], xs[3 * i + 2] - apex[2]]);
    sumA += Math.acos(Math.max(-1, Math.min(1, dot(u, d))));
  }
  const alpha = sumA / m;
  if (!(alpha > 0.005 && alpha < MAX_CONE_ANGLE)) return null;
  const h0 = -dot(apex, d);
  if (!(h0 > 0)) return null;
  return { p: [apex[0] + h0 * d[0], apex[1] + h0 * d[1], apex[2] + h0 * d[2]], d, r0: h0 * Math.tan(alpha), a: alpha };
}

function coneStateFromDef(def: ConeDef, S: PointSample): ConeState {
  const d = normalize(def.axis);
  const apex = toNorm(def.apex, S);
  const h0 = -dot(apex, d);
  return { p: [apex[0] + h0 * d[0], apex[1] + h0 * d[1], apex[2] + h0 * d[2]], d, r0: h0 * Math.tan(def.halfAngle), a: def.halfAngle };
}

/** Mean |n·d| → half-angle estimate of a cone around d from normals. */
function halfAngleFromNormals(NS: PointSample, d: Vec3): number {
  if (!NS.ns) return 0;
  let s = 0;
  for (let i = 0; i < NS.m; i++) s += Math.abs(NS.ns[3 * i] * d[0] + NS.ns[3 * i + 1] * d[1] + NS.ns[3 * i + 2] * d[2]);
  return Math.asin(Math.min(1, s / NS.m));
}

function fitConeImpl(pos: PointArray, opts: ConeFitOptions, fixedAxis?: Vec3): ConeFitResult {
  const S = buildSample(pos, opts, fixedAxis ? 4 : 6, "fitCone");
  const fixed = fixedAxis ? [true, true, false, false, false, false] : undefined;
  let cands: ConeState[] = [];
  if (opts.initial && !fixedAxis) {
    cands = [coneStateFromDef(opts.initial, S)];
  } else {
    const NS = samplesWithNormals(S, 3000);
    if (opts.initial) {
      const st = coneStateFromDef({ ...opts.initial, axis: normalize(fixedAxis!) }, S);
      cands.push(st);
    }
    if (!fixedAxis && NS.ns) {
      const st = coneFromNormals(NS.xs, NS.ns, NS.m);
      if (st) cands.push(st);
    }
    // cylinder-based initialisation (half angle 0 and ± the normal-derived estimate)
    const cyl = solveCylinder(subsample(S, 20000, 19), { maxIterations: 50 }, fixedAxis ? { axis: fixedAxis } : {}).state;
    const base: ConeState = { p: cyl.p, d: cyl.d, r0: cyl.r, a: 0 };
    cands.push(base);
    const aEst = halfAngleFromNormals(NS, cyl.d);
    if (aEst > 0.02) {
      cands.push({ ...base, a: aEst }, { ...base, a: -aEst });
    }
  }
  if (!cands.length) throw new Error("fitCone: could not initialise");
  const start = pickBest(cands, S, (xs, m) => coneProblem(xs, m), fixed);
  const sol = solveMaybeRobust(
    (w) => coneProblem(S.xs, S.m, w),
    (s, out) => coneResiduals(S.xs, S.m, s, out),
    S.m,
    start,
    opts.robust,
    { maxIterations: opts.maxIterations ?? 100, fixed },
  );
  let { p, d, r0, a } = sol.state;
  if (a < 0) {
    d = [-d[0], -d[1], -d[2]];
    a = -a;
  }
  const nearAngle = opts.nearCylinderAngle ?? (0.1 * Math.PI) / 180;
  const nearCylinder = a < nearAngle;
  const aC = Math.max(a, 1e-6);
  const apexN: Vec3 = [p[0] - (r0 / Math.tan(aC)) * d[0], p[1] - (r0 / Math.tan(aC)) * d[1], p[2] - (r0 / Math.tan(aC)) * d[2]];
  const apex = toWorld(apexN, S);
  if (!Number.isFinite(apex[0] + apex[1] + apex[2])) throw new Error("fitCone: did not converge (non-finite apex)");
  const proto: ConeDef = { kind: "cone", apex, axis: d, halfAngle: aC, hMin: 0, hMax: 0 };
  const fs = finishStats(proto, pos, opts.indices, sol.sigma * S.scale);
  const [lo, hi] = axialRange(pos, opts.indices, apex, d, fs.inliers);
  const def: ConeDef = { ...proto, hMin: Math.max(0, lo), hMax: Math.max(0, hi) };
  const out: ConeFitResult = { def, ...fs, nearCylinder };
  if (nearCylinder) {
    const origin = toWorld(p, S);
    const axis = canonicalSign(d);
    const [cl, ch] = axialRange(pos, opts.indices, origin, axis, fs.inliers);
    out.cylinder = { kind: "cylinder", origin, axis, radius: Math.abs(r0) * S.scale, hMin: cl, hMax: ch };
    out.warning = `fitCone: half angle ${((a * 180) / Math.PI).toFixed(4)}° is below ${((nearAngle * 180) / Math.PI).toFixed(3)}° — the data is (nearly) cylindrical; the apex is far away and ill-conditioned. Consider a cylinder.`;
  }
  return out;
}

/**
 * Cone: initialised from normals (apex = least-squares intersection of tangent planes, axis ⟂ to
 * u×n for all points) and from a cylinder fit (half angle 0 / ±estimate), then geometric LM with
 * a parametrisation that is smooth through the cylinder case (axis point, radius at that point,
 * signed half angle). Near-cylindrical data is reported via `nearCylinder` (+ `cylinder`).
 */
export function fitCone(pos: PointArray, opts: ConeFitOptions = {}): ConeFitResult {
  return fitConeImpl(pos, opts);
}

/** Cone with a prescribed axis direction (e.g. ⟂ to a reference plane). */
export function fitConeWithAxis(pos: PointArray, axis: Vec3, opts: ConeFitOptions = {}): ConeFitResult {
  const a = normalize(axis);
  if (length(a) === 0) throw new Error("fitConeWithAxis: zero axis direction");
  return fitConeImpl(pos, opts, a);
}

// ===========================================================================
// Generic entry points
// ===========================================================================

export type AnyFitOptions = FitOptions & { initial?: PrimitiveDef };

/** Fit a primitive of the given kind. */
export function fitPrimitive(kind: PrimitiveKind, pos: PointArray, opts: AnyFitOptions = {}): FitResult<PrimitiveDef> {
  const init = opts.initial && opts.initial.kind === kind ? opts.initial : undefined;
  switch (kind) {
    case "plane":
      return fitPlane(pos, opts);
    case "sphere":
      return fitSphere(pos, { ...opts, initial: init as SphereDef | undefined });
    case "cylinder":
      return fitCylinder(pos, { ...opts, initial: init as CylinderDef | undefined });
    case "cone":
      return fitCone(pos, { ...opts, initial: init as ConeDef | undefined });
  }
  throw new Error(`fitPrimitive: unknown kind '${kind as string}'`);
}

/** Degrees of freedom of each primitive kind. */
export const PRIMITIVE_DOF: Record<PrimitiveKind, number> = { plane: 3, sphere: 4, cylinder: 5, cone: 6 };

export interface DetectOptions extends FitOptions {
  /** kinds to try (default all four) */
  kinds?: PrimitiveKind[];
  /** relative RMS penalty per extra degree of freedom over a plane (default 0.1) */
  complexityPenalty?: number;
}

export interface RankedPrimitive {
  kind: PrimitiveKind;
  result: FitResult<PrimitiveDef>;
  rms: number;
  /** ranking score (lower is better): rms * (1 + penalty * (dof - 3)) */
  score: number;
}

/**
 * Fit all primitive kinds and rank them by RMS with a complexity penalty (so a plane beats a
 * huge-radius sphere and a cylinder beats a near-zero-angle cone). Kinds that fail are omitted.
 */
export function detectBestPrimitive(pos: PointArray, opts: DetectOptions = {}): RankedPrimitive[] {
  const kinds = opts.kinds ?? (["plane", "sphere", "cylinder", "cone"] as PrimitiveKind[]);
  const pen = opts.complexityPenalty ?? 0.1;
  const out: RankedPrimitive[] = [];
  for (const kind of kinds) {
    try {
      const result = fitPrimitive(kind, pos, opts);
      const rms = result.stats.rms;
      if (!Number.isFinite(rms)) continue;
      let score = rms * (1 + pen * (PRIMITIVE_DOF[kind] - 3));
      if (kind === "cone" && (result as ConeFitResult).nearCylinder) score *= 1 + pen;
      out.push({ kind, result, rms, score });
    } catch {
      /* kind not applicable to this data */
    }
  }
  out.sort((a, b) => a.score - b.score);
  return out;
}

// ===========================================================================
// RANSAC
// ===========================================================================

export interface RansacOptions {
  /** inlier distance threshold in model units (default 1 % of the selection's RMS radius about its centroid) */
  threshold?: number;
  /** max angle between point normal and primitive normal for inliers (default 25°; ignored without normals) */
  normalThresholdDeg?: number;
  /** maximum number of hypotheses (default 1000; adaptive early termination at 99.9 % confidence) */
  iterations?: number;
  seed?: number;
  indices?: ArrayLike<number>;
  /** points used to score hypotheses (default 4000) */
  evalSamples?: number;
  /** points used by the least-squares refits (default 200k) */
  maxPoints?: number;
}

const MIN_SAMPLE: Record<PrimitiveKind, number> = { plane: 3, sphere: 2, cylinder: 2, cone: 3 };

function hypothesis(kind: PrimitiveKind, E: PointSample, rng: () => number, hasNormals: boolean): PrimitiveDef | null {
  const m = E.m;
  const pick = (): number => Math.min(m - 1, Math.floor(rng() * m));
  const P = (i: number): Vec3 => [E.xs[3 * i], E.xs[3 * i + 1], E.xs[3 * i + 2]];
  const N = (i: number): Vec3 => (E.ns ? [E.ns[3 * i], E.ns[3 * i + 1], E.ns[3 * i + 2]] : [0, 0, 0]);
  switch (kind) {
    case "plane": {
      const a = P(pick()), b = P(pick()), c = P(pick());
      const n = cross([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]]);
      const l = length(n);
      if (l < 1e-9) return null;
      return { kind: "plane", origin: a, normal: [n[0] / l, n[1] / l, n[2] / l] };
    }
    case "sphere": {
      if (!hasNormals) {
        // 4 points, algebraic
        const ids = [pick(), pick(), pick(), pick()];
        const xs = new Float64Array(12);
        ids.forEach((i, k) => xs.set(P(i), 3 * k));
        const s = algebraicSphere(xs, 4);
        if (!s) return null;
        return { kind: "sphere", center: [s[0], s[1], s[2]], radius: s[3] };
      }
      const i = pick(), j = pick();
      if (i === j) return null;
      const p1 = P(i), n1 = N(i), p2 = P(j), n2 = N(j);
      // closest points between the normal lines
      const w0: Vec3 = [p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]];
      const b = dot(n1, n2), dd = dot(n1, w0), e = dot(n2, w0);
      const den = 1 - b * b;
      if (den < 1e-6) return null;
      const s = (b * e - dd) / den, t = (e - b * dd) / den;
      const c: Vec3 = [
        0.5 * (p1[0] + s * n1[0] + p2[0] + t * n2[0]),
        0.5 * (p1[1] + s * n1[1] + p2[1] + t * n2[1]),
        0.5 * (p1[2] + s * n1[2] + p2[2] + t * n2[2]),
      ];
      const r1 = Math.hypot(p1[0] - c[0], p1[1] - c[1], p1[2] - c[2]);
      const r2 = Math.hypot(p2[0] - c[0], p2[1] - c[1], p2[2] - c[2]);
      if (!(r1 > 0) || Math.abs(r1 - r2) > 0.1 * (r1 + r2)) return null;
      return { kind: "sphere", center: c, radius: 0.5 * (r1 + r2) };
    }
    case "cylinder": {
      const i = pick(), j = pick();
      if (i === j) return null;
      const p1 = P(i), n1 = N(i), p2 = P(j), n2 = N(j);
      const ax = cross(n1, n2);
      const l = length(ax);
      if (l < 1e-3) return null;
      const d: Vec3 = [ax[0] / l, ax[1] / l, ax[2] / l];
      const [e1, e2] = perpBasis(d);
      const u1 = [dot(p1, e1), dot(p1, e2)], u2 = [dot(p2, e1), dot(p2, e2)];
      const m1 = [dot(n1, e1), dot(n1, e2)], m2 = [dot(n2, e1), dot(n2, e2)];
      // u1 + s m1 = u2 + t m2
      const det = m1[0] * -m2[1] - m1[1] * -m2[0];
      if (Math.abs(det) < 1e-9) return null;
      const rx = u2[0] - u1[0], ry = u2[1] - u1[1];
      const s = (rx * -m2[1] - ry * -m2[0]) / det;
      const cx = u1[0] + s * m1[0], cy = u1[1] + s * m1[1];
      const r1 = Math.hypot(u1[0] - cx, u1[1] - cy), r2 = Math.hypot(u2[0] - cx, u2[1] - cy);
      if (!(r1 > 0) || Math.abs(r1 - r2) > 0.1 * (r1 + r2)) return null;
      const o: Vec3 = [cx * e1[0] + cy * e2[0], cx * e1[1] + cy * e2[1], cx * e1[2] + cy * e2[2]];
      return { kind: "cylinder", origin: o, axis: d, radius: 0.5 * (r1 + r2), hMin: 0, hMax: 0 };
    }
    case "cone": {
      const ids = [pick(), pick(), pick()];
      if (ids[0] === ids[1] || ids[1] === ids[2] || ids[0] === ids[2]) return null;
      const A = new Float64Array(9), b = new Float64Array(3);
      ids.forEach((i, k) => {
        const n = N(i), p = P(i);
        A[3 * k] = n[0];
        A[3 * k + 1] = n[1];
        A[3 * k + 2] = n[2];
        b[k] = dot(n, p);
      });
      const apexS = solveLinear(A, b, 3);
      if (!apexS) return null;
      const apex: Vec3 = [apexS[0], apexS[1], apexS[2]];
      if (length(apex) > 100) return null;
      const u = ids.map((i) => normalize([E.xs[3 * i] - apex[0], E.xs[3 * i + 1] - apex[1], E.xs[3 * i + 2] - apex[2]]));
      let d = cross([u[1][0] - u[0][0], u[1][1] - u[0][1], u[1][2] - u[0][2]], [u[2][0] - u[0][0], u[2][1] - u[0][1], u[2][2] - u[0][2]]);
      if (length(d) < 1e-9) return null;
      d = normalize(d);
      if (dot(d, u[0]) < 0) d = [-d[0], -d[1], -d[2]];
      const alpha = Math.acos(Math.min(1, dot(d, u[0])));
      if (!(alpha > 0.01 && alpha < MAX_CONE_ANGLE)) return null;
      return { kind: "cone", apex, axis: d, halfAngle: alpha, hMin: 0, hMax: 0 };
    }
  }
  return null;
}

/** MSAC cost and inlier count of a model on the evaluation sample. */
function scoreModel(def: PrimitiveDef, E: PointSample, T: number, cosThr: number, useNormals: boolean): { cost: number; inliers: number } {
  const f = distanceKernel(def);
  const nk = normalKernel(def);
  const n = new Float64Array(3);
  const T2 = T * T;
  let cost = 0, inl = 0;
  for (let i = 0; i < E.m; i++) {
    const x = E.xs[3 * i], y = E.xs[3 * i + 1], z = E.xs[3 * i + 2];
    const r = f(x, y, z);
    let r2 = r * r;
    if (r2 < T2 && useNormals && E.ns) {
      nk(x, y, z, n);
      if (Math.abs(n[0] * E.ns[3 * i] + n[1] * E.ns[3 * i + 1] + n[2] * E.ns[3 * i + 2]) < cosThr) r2 = T2;
    }
    if (r2 < T2) inl++;
    cost += Math.min(r2, T2);
  }
  return { cost, inliers: inl };
}

/**
 * Inlier mask over the whole selection (distance + optional normal compatibility), plus the
 * vertex ids of a deterministic stride sub-sample of the inliers (≤ maxIds) for the refit.
 */
function classifyAll(
  def: PrimitiveDef,
  pos: PointArray,
  normals: ArrayLike<number> | null | undefined,
  indices: ArrayLike<number> | undefined,
  T: number,
  cosThr: number,
  maxIds: number,
): { mask: Uint8Array; count: number; ids: Uint32Array } {
  const f = distanceKernel(def);
  const nk = normalKernel(def);
  const pn = new Float64Array(3);
  const count = selectionCount(pos, indices);
  const mask = new Uint8Array(count);
  let n = 0;
  for (let k = 0; k < count; k++) {
    const i = indices ? indices[k] : k;
    const x = pos[3 * i], y = pos[3 * i + 1], z = pos[3 * i + 2];
    const r = f(x, y, z);
    if (!(Math.abs(r) <= T)) continue;
    if (normals) {
      const nx = normals[3 * i], ny = normals[3 * i + 1], nz = normals[3 * i + 2];
      const l = Math.sqrt(nx * nx + ny * ny + nz * nz);
      if (l > 0) {
        nk(x, y, z, pn);
        if (Math.abs(pn[0] * nx + pn[1] * ny + pn[2] * nz) / l < cosThr) continue;
      }
    }
    mask[k] = 1;
    n++;
  }
  // stride sub-sample of the inliers: take inlier number floor(j * n / m) for j = 0..m-1
  const m = Math.min(n, Math.max(1, maxIds));
  const ids = new Uint32Array(m);
  let seen = 0, t = 0;
  for (let k = 0; k < count && t < m; k++) {
    if (!mask[k]) continue;
    if (Math.floor((t * n) / m) === seen) ids[t++] = indices ? indices[k] : k;
    seen++;
  }
  return { mask, count: n, ids: t === m ? ids : ids.slice(0, t) };
}

function sameMask(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

/**
 * RANSAC (MSAC scoring) for one primitive kind, followed by least-squares refits on the inliers.
 * Used to ignore fillets / outliers / neighbouring faces inside a sloppy brush selection.
 * Cylinders and cones need normals; when `normals` is null they are estimated by local PCA.
 * Returns `inliers` (selection order) and stats over the inliers.
 */
export function ransacPrimitive(
  kind: PrimitiveKind,
  pos: PointArray,
  normals: ArrayLike<number> | null | undefined,
  opts: RansacOptions = {},
): FitResult<PrimitiveDef> {
  const minPts = { plane: 3, sphere: 4, cylinder: 5, cone: 6 }[kind];
  const S = buildSample(pos, { indices: opts.indices, normals, maxPoints: 60000, seed: opts.seed ?? 1 }, minPts, "ransacPrimitive");
  let E = subsample(S, opts.evalSamples ?? 4000, (opts.seed ?? 1) + 101);
  const needNormals = kind === "cylinder" || kind === "cone";
  const hasNormals = !!normals;
  if (needNormals && !hasNormals) E = estimateSampleNormalsFor(S, E);
  const T = (opts.threshold !== undefined ? opts.threshold : 0.01 * S.scale) / S.scale;
  if (!(T > 0)) throw new Error("ransacPrimitive: threshold must be positive");
  const cosThr = Math.cos(((opts.normalThresholdDeg ?? 25) * Math.PI) / 180);
  const useNormals = !!E.ns;
  const rng = mulberry32((opts.seed ?? 1) * 7919 + 17);
  const maxIt = Math.max(1, opts.iterations ?? 1000);
  let best: PrimitiveDef | null = null;
  let bestCost = Infinity;
  let limit = maxIt;
  const s = MIN_SAMPLE[kind] + (kind === "sphere" && !E.ns ? 2 : 0);
  for (let it = 0; it < limit; it++) {
    const h = hypothesis(kind, E, rng, !!E.ns);
    if (!h) continue;
    const sc = scoreModel(h, E, T, cosThr, useNormals);
    if (sc.cost < bestCost) {
      bestCost = sc.cost;
      best = h;
      const w = sc.inliers / E.m;
      const pAll = Math.pow(w, s);
      if (pAll > 1e-9) {
        const need = Math.ceil(Math.log(1 - 0.999) / Math.log(Math.max(1e-12, 1 - pAll)));
        limit = Math.min(maxIt, Math.max(need, 30));
      }
    }
  }
  if (!best) throw new Error(`ransacPrimitive: no valid ${kind} hypothesis found (data degenerate or normals missing)`);
  let def: PrimitiveDef = defToWorld(best, S);
  const Tw = T * S.scale;
  const maxIds = opts.maxPoints ?? DEFAULT_MAX_POINTS;
  let cls = classifyAll(def, pos, normals, opts.indices, Tw, cosThr, maxIds);
  for (let round = 0; round < 3; round++) {
    if (cls.count < minPts) break;
    try {
      // the refit sees a sub-sample of the inliers; final stats/extents below use the full mask
      def = fitPrimitive(kind, pos, { indices: cls.ids, normals, initial: def, maxPoints: maxIds, seed: opts.seed }).def;
    } catch {
      break;
    }
    const next = classifyAll(def, pos, normals, opts.indices, Tw, cosThr, maxIds);
    const same = sameMask(next.mask, cls.mask);
    cls = next;
    if (same) break;
  }
  if (cls.count < minPts) throw new Error(`ransacPrimitive: too few inliers (${cls.count}) for a ${kind}`);
  // final extents / stats over the final inliers
  if (def.kind === "cylinder") {
    const [lo, hi] = axialRange(pos, opts.indices, def.origin, def.axis, cls.mask);
    def = { ...def, hMin: lo, hMax: hi };
  } else if (def.kind === "cone") {
    const [lo, hi] = axialRange(pos, opts.indices, def.apex, def.axis, cls.mask);
    def = { ...def, hMin: Math.max(0, lo), hMax: Math.max(0, hi) };
  }
  return { def, stats: primitiveStats(def, pos, opts.indices, cls.mask), inliers: cls.mask };
}

function estimateSampleNormalsFor(S: PointSample, E: PointSample): PointSample {
  const grid = new GridIndex(S.xs, S.m);
  const ns = new Float64Array(3 * E.m);
  const k = Math.min(16, Math.max(3, S.m - 1));
  const si = new Int32Array(k), sd = new Float64Array(k);
  for (let i = 0; i < E.m; i++) pcaNormalAt(grid, E.xs[3 * i], E.xs[3 * i + 1], E.xs[3 * i + 2], k, ns, 3 * i, si, sd);
  return { ...E, ns };
}
