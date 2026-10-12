/**
 * Reference-geometry constructions (planes, lines/axes, points) used by the Reference Geometry
 * and alignment tools. All functions are pure; "null" means the construction is undefined
 * (parallel / coincident / degenerate input).
 */
import type { ConeDef, CylinderDef, LineDef, PlaneDef, PointDef, SphereDef, Vec3 } from "../core/types";
import { solveLinear } from "../core/linalg";
import { add, cross, dot, length, normalize, scale, sub } from "../core/vec";

const PARALLEL_EPS = 1e-9;

const unit = (v: Vec3, what: string): Vec3 => {
  const n = normalize(v);
  if (length(n) === 0) throw new Error(`${what}: zero-length direction`);
  return n;
};

/** Plane from a point and a normal. */
export function planeFromPointNormal(point: Vec3, normal: Vec3): PlaneDef {
  return { kind: "plane", origin: [point[0], point[1], point[2]], normal: unit(normal, "planeFromPointNormal") };
}

/** Line from a point and a direction. */
export function lineFromPointDirection(point: Vec3, direction: Vec3): LineDef {
  return { kind: "line", origin: [point[0], point[1], point[2]], direction: unit(direction, "lineFromPointDirection") };
}

/** Line through two points (null if they coincide). origin = a. */
export function lineFrom2Points(a: Vec3, b: Vec3): LineDef | null {
  const d = sub(b, a);
  const l = length(d);
  if (l < 1e-12 * Math.max(1, length(a), length(b))) return null;
  return { kind: "line", origin: [a[0], a[1], a[2]], direction: scale(d, 1 / l) };
}

/**
 * Intersection line of two planes, or null if they are parallel. The line origin is the point
 * of the line closest to the midpoint of the two plane origins.
 */
export function intersectPlanes(a: PlaneDef, b: PlaneDef): LineDef | null {
  const na = normalize(a.normal), nb = normalize(b.normal);
  const u = cross(na, nb);
  const l = length(u);
  if (l < 1e-9) return null;
  const dir = scale(u, 1 / l);
  const m = scale(add(a.origin, b.origin), 0.5);
  const A = [na[0], na[1], na[2], nb[0], nb[1], nb[2], dir[0], dir[1], dir[2]];
  const rhs = [dot(na, a.origin), dot(nb, b.origin), dot(dir, m)];
  const p = solveLinear(A, rhs, 3);
  if (!p) return null;
  return { kind: "line", origin: [p[0], p[1], p[2]], direction: dir };
}

/** Common point of three planes, or null if two of them are parallel / the three share a line. */
export function intersectThreePlanes(a: PlaneDef, b: PlaneDef, c: PlaneDef): Vec3 | null {
  const na = normalize(a.normal), nb = normalize(b.normal), nc = normalize(c.normal);
  const det = dot(na, cross(nb, nc));
  if (Math.abs(det) < 1e-9) return null;
  const p = solveLinear([...na, ...nb, ...nc], [dot(na, a.origin), dot(nb, b.origin), dot(nc, c.origin)], 3);
  return p ? [p[0], p[1], p[2]] : null;
}

/** Intersection point of a line and a plane, or null if parallel. */
export function intersectLinePlane(l: LineDef, p: PlaneDef): Vec3 | null {
  const n = normalize(p.normal);
  const d = normalize(l.direction);
  const den = dot(n, d);
  if (Math.abs(den) < PARALLEL_EPS) return null;
  const t = dot(n, sub(p.origin, l.origin)) / den;
  return [l.origin[0] + t * d[0], l.origin[1] + t * d[1], l.origin[2] + t * d[2]];
}

export interface LineLineClosest {
  /** closest point on l1 */
  p1: Vec3;
  /** closest point on l2 */
  p2: Vec3;
  /** parameters along the (unit) directions from the line origins */
  t1: number;
  t2: number;
  distance: number;
  /** true when the lines are parallel (p1 = l1.origin, p2 = its projection on l2) */
  parallel: boolean;
}

/** Closest points between two infinite lines. */
export function closestPointsBetweenLines(l1: LineDef, l2: LineDef): LineLineClosest {
  const d1 = normalize(l1.direction), d2 = normalize(l2.direction);
  const w0 = sub(l1.origin, l2.origin);
  const b = dot(d1, d2);
  const d = dot(d1, w0), e = dot(d2, w0);
  const den = 1 - b * b;
  let t1: number, t2: number, parallel = false;
  if (den < 1e-12) {
    parallel = true;
    t1 = 0;
    t2 = e;
  } else {
    t1 = (b * e - d) / den;
    t2 = (e - b * d) / den;
  }
  const p1: Vec3 = [l1.origin[0] + t1 * d1[0], l1.origin[1] + t1 * d1[1], l1.origin[2] + t1 * d1[2]];
  const p2: Vec3 = [l2.origin[0] + t2 * d2[0], l2.origin[1] + t2 * d2[1], l2.origin[2] + t2 * d2[2]];
  return { p1, p2, t1, t2, distance: length(sub(p1, p2)), parallel };
}

/** Intersection of two (nearly) intersecting lines: midpoint of the closest points, or null if parallel. */
export function intersectLines(l1: LineDef, l2: LineDef): Vec3 | null {
  const c = closestPointsBetweenLines(l1, l2);
  if (c.parallel) return null;
  return scale(add(c.p1, c.p2), 0.5);
}

/** Plane through three points (origin = their centroid, normal = (b-a)×(c-a)), or null if collinear. */
export function planeFrom3Points(a: Vec3, b: Vec3, c: Vec3): PlaneDef | null {
  const n = cross(sub(b, a), sub(c, a));
  const l = length(n);
  const sz = Math.max(length(sub(b, a)), length(sub(c, a)), 1e-300);
  if (l < 1e-12 * sz * sz) return null;
  return {
    kind: "plane",
    origin: [(a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3, (a[2] + b[2] + c[2]) / 3],
    normal: scale(n, 1 / l),
  };
}

/** Plane shifted by `distance` along its normal. */
export function offsetPlane(p: PlaneDef, distance: number): PlaneDef {
  const n = normalize(p.normal);
  return { kind: "plane", origin: [p.origin[0] + distance * n[0], p.origin[1] + distance * n[1], p.origin[2] + distance * n[2]], normal: n };
}

/** The same plane with the normal reversed. */
export function flipPlane(p: PlaneDef): PlaneDef {
  return { kind: "plane", origin: [p.origin[0], p.origin[1], p.origin[2]], normal: [-p.normal[0], -p.normal[1], -p.normal[2]] };
}

/**
 * Mid plane of two planes.
 *  - parallel planes: the plane halfway between them (normal = a.normal);
 *  - intersecting planes: the bisector through their intersection line. Of the two bisectors this
 *    returns the one whose normal is n_a + n_b' where n_b' = ±n_b is oriented to agree with n_a —
 *    i.e. the bisector lying "between" the planes in the sense that degenerates to the parallel
 *    case. Pass `other: true` to get the perpendicular bisector instead.
 */
export function midPlane(a: PlaneDef, b: PlaneDef, opts: { other?: boolean } = {}): PlaneDef {
  const na = normalize(a.normal);
  let nb = normalize(b.normal);
  if (dot(na, nb) < 0) nb = scale(nb, -1);
  const line = intersectPlanes(a, b);
  if (!line || length(cross(na, nb)) < 1e-6) {
    // (nearly) parallel: halfway offsets along na
    const da = dot(na, a.origin), db = dot(na, b.origin);
    const dm = 0.5 * (da + db);
    const mid = scale(add(a.origin, b.origin), 0.5);
    const s = dm - dot(na, mid);
    const origin: Vec3 = [mid[0] + s * na[0], mid[1] + s * na[1], mid[2] + s * na[2]];
    if (opts.other) {
      // a perpendicular plane through the midpoint containing the common normal (arbitrary in-plane direction)
      const t = normalize(cross(na, Math.abs(na[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]));
      return { kind: "plane", origin, normal: t };
    }
    return { kind: "plane", origin, normal: na };
  }
  const n = opts.other ? normalize(sub(na, nb)) : normalize(add(na, nb));
  // origin: point of the intersection line closest to the midpoint of the two origins
  const mid = scale(add(a.origin, b.origin), 0.5);
  const t = dot(sub(mid, line.origin), line.direction);
  const origin: Vec3 = [line.origin[0] + t * line.direction[0], line.origin[1] + t * line.direction[1], line.origin[2] + t * line.direction[2]];
  return { kind: "plane", origin, normal: n };
}

/** Axis line of a cylinder (origin = cylinder origin) or cone (origin = apex, direction into the opening). */
export function axisOf(def: CylinderDef | ConeDef): LineDef {
  if (def.kind === "cylinder") return { kind: "line", origin: [...def.origin] as Vec3, direction: normalize(def.axis) };
  return { kind: "line", origin: [...def.apex] as Vec3, direction: normalize(def.axis) };
}

/** Centre of a sphere as a reference point. */
export function centerOf(sphere: SphereDef): PointDef {
  return { kind: "point", position: [sphere.center[0], sphere.center[1], sphere.center[2]] };
}

/** Apex of a cone as a reference point. */
export function apexOf(cone: ConeDef): PointDef {
  return { kind: "point", position: [cone.apex[0], cone.apex[1], cone.apex[2]] };
}

/** Plane containing a line and a point (null if the point lies on the line). origin = line origin. */
export function planeThroughLineAndPoint(line: LineDef, p: Vec3): PlaneDef | null {
  const d = normalize(line.direction);
  const v = sub(p, line.origin);
  const n = cross(d, v);
  const l = length(n);
  if (l < 1e-12 * Math.max(1, length(v))) return null;
  return { kind: "plane", origin: [...line.origin] as Vec3, normal: scale(n, 1 / l) };
}

/** Plane containing two lines (parallel or intersecting), or null if skew / identical. */
export function planeThroughTwoLines(l1: LineDef, l2: LineDef, tol = 1e-6): PlaneDef | null {
  const c = closestPointsBetweenLines(l1, l2);
  if (c.parallel) return planeThroughLineAndPoint(l1, l2.origin);
  const scaleRef = Math.max(1, length(l1.origin), length(l2.origin));
  if (c.distance > tol * scaleRef) return null;
  const n = cross(normalize(l1.direction), normalize(l2.direction));
  return { kind: "plane", origin: scale(add(c.p1, c.p2), 0.5), normal: normalize(n) };
}

/** Plane through a point, perpendicular to a line direction (e.g. a section plane ⟂ an axis). */
export function planePerpendicularToLine(line: LineDef, through?: Vec3): PlaneDef {
  return { kind: "plane", origin: through ? [...through] as Vec3 : [...line.origin] as Vec3, normal: normalize(line.direction) };
}

/** Line through a point along the normal of a plane. */
export function lineNormalToPlane(plane: PlaneDef, through?: Vec3): LineDef {
  return { kind: "line", origin: through ? [...through] as Vec3 : [...plane.origin] as Vec3, direction: normalize(plane.normal) };
}

/** Orthogonal projection of a point onto a line. */
export function projectPointOnLine(p: Vec3, line: LineDef): Vec3 {
  const d = normalize(line.direction);
  const t = dot(sub(p, line.origin), d);
  return [line.origin[0] + t * d[0], line.origin[1] + t * d[1], line.origin[2] + t * d[2]];
}

/** Orthogonal projection of a point onto a plane. */
export function projectPointOnPlane(p: Vec3, plane: PlaneDef): Vec3 {
  const n = normalize(plane.normal);
  const d = dot(sub(p, plane.origin), n);
  return [p[0] - d * n[0], p[1] - d * n[1], p[2] - d * n[2]];
}

/** Projection of a line onto a plane (null if the line is perpendicular to the plane). */
export function projectLineOnPlane(line: LineDef, plane: PlaneDef): LineDef | null {
  const n = normalize(plane.normal);
  const d = normalize(line.direction);
  const dp = sub(d, scale(n, dot(d, n)));
  const l = length(dp);
  if (l < 1e-9) return null;
  return { kind: "line", origin: projectPointOnPlane(line.origin, plane), direction: scale(dp, 1 / l) };
}

/** Signed distance of a point to a plane (positive on the normal side). */
export function distancePointPlane(p: Vec3, plane: PlaneDef): number {
  return dot(sub(p, plane.origin), normalize(plane.normal));
}

/** Distance of a point to a line. */
export function distancePointLine(p: Vec3, line: LineDef): number {
  return length(sub(p, projectPointOnLine(p, line)));
}

/** Angle between two planes' normals, in [0, π/2] (unsigned dihedral, ignoring orientation). */
export function angleBetweenPlanes(a: PlaneDef, b: PlaneDef): number {
  const c = Math.abs(dot(normalize(a.normal), normalize(b.normal)));
  return Math.acos(Math.min(1, c));
}

/** Angle between two lines' directions, in [0, π/2]. */
export function angleBetweenLines(a: LineDef, b: LineDef): number {
  const c = Math.abs(dot(normalize(a.direction), normalize(b.direction)));
  return Math.acos(Math.min(1, c));
}
