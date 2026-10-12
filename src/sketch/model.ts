/**
 * Serialisable 2D sketch model (points, entities, constraints) plus builder/helper functions.
 *
 * Conventions
 *  - All coordinates are 2D sketch-plane coordinates in mm; angles in radians.
 *  - Ids are unique across points, entities AND constraints of one sketch (constraint refs may name
 *    either points or entities). New ids come from `sketch.nextId` ("p12", "l13", "k14", ...).
 *  - The builder helpers (`addPoint`, `addLine`, ..., `deleteEntity`) MUTATE the sketch passed in and
 *    return the new id(s); use {@link cloneSketch} first when an immutable update is needed.
 *  - Arcs run counter-clockwise from `start` to `end` around `center`; the radius is |start - center|.
 *
 * Constraint reference conventions (refs order):
 *  coincident [pointA, pointB] | [point, entity] (= pointOnEntity)
 *  horizontal / vertical [line] | [pointA, pointB]
 *  parallel / perpendicular / collinear [line1, line2]
 *  tangent [e1, e2] (line/arc/circle combinations; endpoint tangency when they share an endpoint)
 *  equal [line1, line2] (length) | [arc|circle, arc|circle] (radius)
 *  concentric [arc|circle|point, arc|circle|point]
 *  midpoint [point, line|arc]
 *  pointOnEntity [point, line|arc|circle]
 *  fix [point | entity]
 *  symmetric [pointA, pointB, line | point]
 *  distance [pointish, pointish] | [pointish, line] | [line, line] | [line] (pointish = point, or circle/arc = its center)
 *  horizontalDistance / verticalDistance [pointish, pointish] | [line]
 *  radius / diameter [arc|circle]
 *  angle [line1, line2]   (angle between the directed lines p1->p2, value in [0, PI])
 *  length [line | arc]
 */
import type { Segment2D, Vec2 } from "../core/types";
import { interpolateBSpline2 } from "./bspline";
import { TAU, angleOf, closestPointOnSegment, normAngle, polar } from "./geom2d";

export interface SketchPoint {
  id: string;
  x: number;
  y: number;
  /** locked in place (the solver never moves it) */
  fixed?: boolean;
}

export type SketchEntity =
  | { id: string; type: "line"; p1: string; p2: string; construction?: boolean }
  /** counter-clockwise from start to end */
  | { id: string; type: "arc"; center: string; start: string; end: string; construction?: boolean }
  | { id: string; type: "circle"; center: string; radius: number; construction?: boolean }
  /** interpolating spline through the fit points (cubic); `closed` makes a C1 periodic-like loop */
  | { id: string; type: "spline"; points: string[]; closed?: boolean; construction?: boolean };

export type SketchEntityType = SketchEntity["type"];

export type GeometricConstraintType =
  | "coincident"
  | "horizontal"
  | "vertical"
  | "parallel"
  | "perpendicular"
  | "tangent"
  | "equal"
  | "concentric"
  | "midpoint"
  | "pointOnEntity"
  | "fix"
  | "symmetric"
  | "collinear";

export type DimensionConstraintType =
  | "distance"
  | "horizontalDistance"
  | "verticalDistance"
  | "radius"
  | "diameter"
  | "angle"
  | "length";

export type ConstraintType = GeometricConstraintType | DimensionConstraintType;

export interface GeometricConstraint {
  id: string;
  type: GeometricConstraintType;
  refs: string[];
}

export interface DimensionConstraint {
  id: string;
  type: DimensionConstraintType;
  refs: string[];
  /** mm for distances/lengths/radii, radians for angles */
  value: number;
  /** driving dimensions are enforced by the solver; reference (driven) ones only display a measurement */
  driving: boolean;
  /** label placement relative to the dimension's anchor (display only) */
  labelOffset?: Vec2;
}

export type Constraint = GeometricConstraint | DimensionConstraint;

export interface Sketch {
  points: SketchPoint[];
  entities: SketchEntity[];
  constraints: Constraint[];
  /** id counter (optional; recomputed from existing ids when missing) */
  nextId?: number;
}

export const DIMENSION_TYPES: readonly DimensionConstraintType[] = [
  "distance",
  "horizontalDistance",
  "verticalDistance",
  "radius",
  "diameter",
  "angle",
  "length",
];

export function isDimension(c: { type: string }): c is DimensionConstraint {
  return (DIMENSION_TYPES as readonly string[]).includes(c.type);
}

// ---------------------------------------------------------------------------
// Basics
// ---------------------------------------------------------------------------

export function createSketch(): Sketch {
  return { points: [], entities: [], constraints: [], nextId: 1 };
}

/** Deep copy (JSON-safe). */
export function cloneSketch(s: Sketch): Sketch {
  return {
    points: s.points.map((p) => ({ ...p })),
    entities: s.entities.map((e) => (e.type === "spline" ? { ...e, points: e.points.slice() } : { ...e })),
    constraints: s.constraints.map((c) => {
      const o = { ...c, refs: c.refs.slice() };
      if ("labelOffset" in o && o.labelOffset) o.labelOffset = [o.labelOffset[0], o.labelOffset[1]];
      return o;
    }),
    nextId: s.nextId,
  };
}

/** Generate a fresh id with the given prefix (unique across the whole sketch). */
export function genId(s: Sketch, prefix: string): string {
  if (s.nextId === undefined || !isFinite(s.nextId)) {
    let max = 0;
    const scan = (id: string) => {
      const m = /(\d+)$/.exec(id);
      if (m) max = Math.max(max, parseInt(m[1], 10));
    };
    s.points.forEach((p) => scan(p.id));
    s.entities.forEach((e) => scan(e.id));
    s.constraints.forEach((c) => scan(c.id));
    s.nextId = max + 1;
  }
  let id = `${prefix}${s.nextId++}`;
  // guard against user-supplied ids colliding with the counter
  while (hasId(s, id)) id = `${prefix}${s.nextId++}`;
  return id;
}

function hasId(s: Sketch, id: string): boolean {
  return s.points.some((p) => p.id === id) || s.entities.some((e) => e.id === id) || s.constraints.some((c) => c.id === id);
}

export interface SketchIndex {
  points: Map<string, SketchPoint>;
  entities: Map<string, SketchEntity>;
  constraints: Map<string, Constraint>;
}

export function indexSketch(s: Sketch): SketchIndex {
  return {
    points: new Map(s.points.map((p) => [p.id, p])),
    entities: new Map(s.entities.map((e) => [e.id, e])),
    constraints: new Map(s.constraints.map((c) => [c.id, c])),
  };
}

export const getPoint = (s: Sketch, id: string): SketchPoint | undefined => s.points.find((p) => p.id === id);
export const getEntity = (s: Sketch, id: string): SketchEntity | undefined => s.entities.find((e) => e.id === id);
export const getConstraint = (s: Sketch, id: string): Constraint | undefined => s.constraints.find((c) => c.id === id);
export const pointPos = (p: SketchPoint): Vec2 => [p.x, p.y];

/** Point ids used by an entity (line: p1,p2; arc: center,start,end; circle: center; spline: fit points). */
export function entityPointIds(e: SketchEntity): string[] {
  switch (e.type) {
    case "line":
      return [e.p1, e.p2];
    case "arc":
      return [e.center, e.start, e.end];
    case "circle":
      return [e.center];
    case "spline":
      return e.points.slice();
  }
}

/** Endpoint ids of an open entity (line, arc, open spline); [] for circles and closed splines. */
export function entityEndpointIds(e: SketchEntity): string[] {
  switch (e.type) {
    case "line":
      return [e.p1, e.p2];
    case "arc":
      return [e.start, e.end];
    case "circle":
      return [];
    case "spline":
      return e.closed || e.points.length < 2 ? [] : [e.points[0], e.points[e.points.length - 1]];
  }
}

// ---------------------------------------------------------------------------
// Builders (mutating)
// ---------------------------------------------------------------------------

export type PointRef = string | Vec2;

export function addPoint(s: Sketch, p: Vec2, opts: { fixed?: boolean; id?: string } = {}): string {
  const id = opts.id ?? genId(s, "p");
  const pt: SketchPoint = { id, x: p[0], y: p[1] };
  if (opts.fixed) pt.fixed = true;
  s.points.push(pt);
  return id;
}

function refPoint(s: Sketch, r: PointRef): string {
  if (typeof r === "string") {
    if (!getPoint(s, r)) throw new Error(`Unknown point ${r}`);
    return r;
  }
  return addPoint(s, r);
}

function posOf(s: Sketch, r: PointRef): Vec2 {
  if (typeof r !== "string") return r;
  const p = getPoint(s, r);
  if (!p) throw new Error(`Unknown point ${r}`);
  return [p.x, p.y];
}

export function addLine(s: Sketch, a: PointRef, b: PointRef, opts: { construction?: boolean; id?: string } = {}): string {
  const p1 = refPoint(s, a);
  const p2 = refPoint(s, b);
  const id = opts.id ?? genId(s, "l");
  const e: SketchEntity = { id, type: "line", p1, p2 };
  if (opts.construction) e.construction = true;
  s.entities.push(e);
  return id;
}

/**
 * Arc CCW around `center` from `start` to `end`. When `end` is given as coordinates it is projected
 * onto the circle defined by center/start.
 */
export function addArc(s: Sketch, center: PointRef, start: PointRef, end: PointRef, opts: { construction?: boolean; id?: string } = {}): string {
  const cpos = posOf(s, center);
  const spos = posOf(s, start);
  let endRef = end;
  if (typeof end !== "string") {
    const r = Math.hypot(spos[0] - cpos[0], spos[1] - cpos[1]);
    const a = Math.atan2(end[1] - cpos[1], end[0] - cpos[0]);
    endRef = polar(cpos, r, a);
  }
  const c = refPoint(s, center);
  const st = refPoint(s, start);
  const en = refPoint(s, endRef);
  const id = opts.id ?? genId(s, "a");
  const e: SketchEntity = { id, type: "arc", center: c, start: st, end: en };
  if (opts.construction) e.construction = true;
  s.entities.push(e);
  return id;
}

/**
 * Three-point arc from p1 through pMid to p2. The stored arc is CCW, so for a clockwise input the
 * start/end points are swapped. Returns null when the points are collinear.
 */
export function addArc3(s: Sketch, p1: PointRef, pMid: Vec2, p2: PointRef, opts: { construction?: boolean } = {}): string | null {
  const a = posOf(s, p1), m = pMid, b = posOf(s, p2);
  const d = 2 * (a[0] * (m[1] - b[1]) + m[0] * (b[1] - a[1]) + b[0] * (a[1] - m[1]));
  const scale = Math.max(1, Math.hypot(b[0] - a[0], b[1] - a[1]));
  if (Math.abs(d) < 1e-9 * scale * scale) return null;
  const a2 = a[0] * a[0] + a[1] * a[1], m2 = m[0] * m[0] + m[1] * m[1], b2 = b[0] * b[0] + b[1] * b[1];
  const cx = (a2 * (m[1] - b[1]) + m2 * (b[1] - a[1]) + b2 * (a[1] - m[1])) / d;
  const cy = (a2 * (b[0] - m[0]) + m2 * (a[0] - b[0]) + b2 * (m[0] - a[0])) / d;
  const ccw = (m[0] - a[0]) * (b[1] - a[1]) - (m[1] - a[1]) * (b[0] - a[0]) > 0;
  const c = addPoint(s, [cx, cy]);
  return ccw ? addArc(s, c, p1, p2, opts) : addArc(s, c, p2, p1, opts);
}

export function addCircle(s: Sketch, center: PointRef, radius: number, opts: { construction?: boolean; id?: string } = {}): string {
  const c = refPoint(s, center);
  const id = opts.id ?? genId(s, "c");
  const e: SketchEntity = { id, type: "circle", center: c, radius: Math.abs(radius) };
  if (opts.construction) e.construction = true;
  s.entities.push(e);
  return id;
}

/** Axis-aligned rectangle from two opposite corners, with horizontal/vertical constraints (unless constrain = false). */
export function addRectangle(
  s: Sketch,
  c1: Vec2,
  c2: Vec2,
  opts: { construction?: boolean; constrain?: boolean } = {},
): { points: string[]; lines: string[]; constraints: string[] } {
  const x0 = Math.min(c1[0], c2[0]), x1 = Math.max(c1[0], c2[0]);
  const y0 = Math.min(c1[1], c2[1]), y1 = Math.max(c1[1], c2[1]);
  const pts = [addPoint(s, [x0, y0]), addPoint(s, [x1, y0]), addPoint(s, [x1, y1]), addPoint(s, [x0, y1])];
  const lines = [0, 1, 2, 3].map((i) => addLine(s, pts[i], pts[(i + 1) % 4], opts));
  const constraints: string[] = [];
  if (opts.constrain !== false) {
    constraints.push(addConstraint(s, { type: "horizontal", refs: [lines[0]] }));
    constraints.push(addConstraint(s, { type: "vertical", refs: [lines[1]] }));
    constraints.push(addConstraint(s, { type: "horizontal", refs: [lines[2]] }));
    constraints.push(addConstraint(s, { type: "vertical", refs: [lines[3]] }));
  }
  return { points: pts, lines, constraints };
}

export function addSpline(s: Sketch, pts: PointRef[], opts: { closed?: boolean; construction?: boolean; id?: string } = {}): string {
  const ids = pts.map((p) => refPoint(s, p));
  const id = opts.id ?? genId(s, "s");
  const e: SketchEntity = { id, type: "spline", points: ids };
  if (opts.closed) e.closed = true;
  if (opts.construction) e.construction = true;
  s.entities.push(e);
  return id;
}

export type ConstraintInput =
  | (Omit<GeometricConstraint, "id"> & { id?: string })
  | (Omit<DimensionConstraint, "id" | "driving" | "value"> & { id?: string; value?: number; driving?: boolean });

/**
 * Add a constraint after validating it (throws with a readable message when invalid). For dimensions,
 * a missing `value` defaults to the current measurement and `driving` defaults to true.
 */
export function addConstraint(s: Sketch, c: ConstraintInput): string {
  const err = validateConstraint(s, c);
  if (err) throw new Error(err);
  const id = c.id ?? genId(s, "k");
  if (isDimension(c)) {
    const dc = c as Omit<DimensionConstraint, "id" | "driving" | "value"> & { value?: number; driving?: boolean };
    const value = dc.value ?? measure(s, dc) ?? 0;
    const out: DimensionConstraint = { id, type: dc.type, refs: dc.refs.slice(), value, driving: dc.driving ?? true };
    if (dc.labelOffset) out.labelOffset = [dc.labelOffset[0], dc.labelOffset[1]];
    s.constraints.push(out);
  } else {
    s.constraints.push({ id, type: c.type as GeometricConstraintType, refs: c.refs.slice() });
  }
  return id;
}

type RefKind = "point" | "line" | "arc" | "circle" | "spline" | "missing";

function refKind(s: Sketch, id: string): RefKind {
  if (s.points.some((p) => p.id === id)) return "point";
  const e = getEntity(s, id);
  return e ? e.type : "missing";
}

/** Returns an error message when the constraint is not applicable to its refs, else null. */
export function validateConstraint(s: Sketch, c: { type: string; refs: string[]; value?: number }): string | null {
  const kinds = c.refs.map((r) => refKind(s, r));
  const missing = c.refs.filter((_, i) => kinds[i] === "missing");
  if (missing.length) return `Constraint ${c.type} references unknown id(s) ${missing.join(", ")}`;
  const isCurve = (k: RefKind) => k === "arc" || k === "circle";
  const pointish = (k: RefKind) => k === "point" || isCurve(k);
  const n = kinds.length;
  const k0 = kinds[0], k1 = kinds[1];
  let ok = false;
  switch (c.type) {
    case "coincident":
      ok = n === 2 && ((k0 === "point" && k1 === "point") || (k0 === "point" && k1 !== "spline") || (k1 === "point" && k0 !== "spline"));
      break;
    case "horizontal":
    case "vertical":
      ok = (n === 1 && k0 === "line") || (n === 2 && k0 === "point" && k1 === "point");
      break;
    case "parallel":
    case "perpendicular":
    case "collinear":
      ok = n === 2 && k0 === "line" && k1 === "line" && c.refs[0] !== c.refs[1];
      break;
    case "tangent":
      ok = n === 2 && c.refs[0] !== c.refs[1] && ((k0 === "line" && isCurve(k1)) || (k1 === "line" && isCurve(k0)) || (isCurve(k0) && isCurve(k1)));
      break;
    case "equal":
      ok = n === 2 && c.refs[0] !== c.refs[1] && ((k0 === "line" && k1 === "line") || (isCurve(k0) && isCurve(k1)));
      break;
    case "concentric":
      ok = n === 2 && pointish(k0) && pointish(k1) && (isCurve(k0) || isCurve(k1));
      break;
    case "midpoint":
      ok = n === 2 && ((k0 === "point" && (k1 === "line" || k1 === "arc")) || (k1 === "point" && (k0 === "line" || k0 === "arc")));
      break;
    case "pointOnEntity":
      ok = n === 2 && ((k0 === "point" && (k1 === "line" || isCurve(k1))) || (k1 === "point" && (k0 === "line" || isCurve(k0))));
      break;
    case "fix":
      ok = n === 1;
      break;
    case "symmetric":
      ok = n === 3 && k0 === "point" && k1 === "point" && (kinds[2] === "line" || kinds[2] === "point") && c.refs[0] !== c.refs[1];
      break;
    case "distance":
      ok =
        (n === 1 && k0 === "line") ||
        (n === 2 && ((pointish(k0) && pointish(k1)) || (pointish(k0) && k1 === "line") || (k0 === "line" && pointish(k1)) || (k0 === "line" && k1 === "line")));
      break;
    case "horizontalDistance":
    case "verticalDistance":
      ok = (n === 1 && k0 === "line") || (n === 2 && pointish(k0) && pointish(k1));
      break;
    case "radius":
    case "diameter":
      ok = n === 1 && isCurve(k0);
      break;
    case "angle":
      ok = n === 2 && k0 === "line" && k1 === "line" && c.refs[0] !== c.refs[1];
      break;
    case "length":
      ok = n === 1 && (k0 === "line" || k0 === "arc");
      break;
    default:
      return `Unknown constraint type ${c.type}`;
  }
  if (!ok) return `Constraint ${c.type} cannot be applied to (${kinds.join(", ")})`;
  if (c.value !== undefined && !isFinite(c.value)) return `Constraint ${c.type} has a non-finite value`;
  if ((c.type === "radius" || c.type === "diameter" || c.type === "length") && c.value !== undefined && c.value <= 0)
    return `Constraint ${c.type} needs a positive value`;
  return null;
}

// ---------------------------------------------------------------------------
// Deletion
// ---------------------------------------------------------------------------

export function deleteConstraint(s: Sketch, id: string): void {
  s.constraints = s.constraints.filter((c) => c.id !== id);
}

/** Remove constraints that reference any of the given ids. */
function dropConstraintsReferencing(s: Sketch, ids: Set<string>): void {
  s.constraints = s.constraints.filter((c) => !c.refs.some((r) => ids.has(r)));
}

/**
 * Delete an entity, every constraint that references it, and its points that are no longer used by
 * any other entity (together with the constraints on those points).
 */
export function deleteEntity(s: Sketch, id: string): void {
  deleteEntities(s, [id]);
}

export function deleteEntities(s: Sketch, ids: string[]): void {
  const del = new Set(ids);
  const victims = s.entities.filter((e) => del.has(e.id));
  if (!victims.length) return;
  s.entities = s.entities.filter((e) => !del.has(e.id));
  const used = new Set<string>();
  for (const e of s.entities) for (const p of entityPointIds(e)) used.add(p);
  const orphan = new Set<string>();
  for (const e of victims) for (const p of entityPointIds(e)) if (!used.has(p)) orphan.add(p);
  s.points = s.points.filter((p) => !orphan.has(p.id));
  dropConstraintsReferencing(s, new Set([...del, ...orphan]));
}

/** Delete a point together with the entities using it (and their orphaned points) and its constraints. */
export function deletePoint(s: Sketch, id: string): void {
  const users = s.entities.filter((e) => entityPointIds(e).includes(id)).map((e) => e.id);
  // splines just lose the fit point when they keep at least 2
  const keep: string[] = [];
  for (const uid of users) {
    const e = getEntity(s, uid)!;
    if (e.type === "spline" && e.points.filter((p) => p !== id).length >= 2) {
      e.points = e.points.filter((p) => p !== id);
      keep.push(uid);
    }
  }
  deleteEntities(
    s,
    users.filter((u) => !keep.includes(u)),
  );
  s.points = s.points.filter((p) => p.id !== id);
  dropConstraintsReferencing(s, new Set([id]));
}

/**
 * Merge point `remove` into point `keep`: all entity and constraint references are redirected and the
 * removed point is deleted (constraints that become trivial, e.g. coincident(keep, keep), are dropped).
 */
export function mergePoints(s: Sketch, keep: string, remove: string): void {
  if (keep === remove) return;
  for (const e of s.entities) {
    switch (e.type) {
      case "line":
        if (e.p1 === remove) e.p1 = keep;
        if (e.p2 === remove) e.p2 = keep;
        break;
      case "arc":
        if (e.center === remove) e.center = keep;
        if (e.start === remove) e.start = keep;
        if (e.end === remove) e.end = keep;
        break;
      case "circle":
        if (e.center === remove) e.center = keep;
        break;
      case "spline":
        e.points = e.points.map((p) => (p === remove ? keep : p));
        break;
    }
  }
  for (const c of s.constraints) c.refs = c.refs.map((r) => (r === remove ? keep : r));
  s.constraints = s.constraints.filter((c) => !(c.type === "coincident" && c.refs[0] === c.refs[1]));
  const rp = getPoint(s, remove);
  const kp = getPoint(s, keep);
  if (rp?.fixed && kp) kp.fixed = true;
  s.points = s.points.filter((p) => p.id !== remove);
}

// ---------------------------------------------------------------------------
// Geometry queries
// ---------------------------------------------------------------------------

export interface ArcInfo {
  center: Vec2;
  radius: number;
  startAngle: number;
  endAngle: number;
  /** CCW sweep in (0, 2PI] */
  sweep: number;
}

function P(s: Sketch, id: string, idx?: Map<string, SketchPoint>): Vec2 {
  const p = idx ? idx.get(id) : getPoint(s, id);
  if (!p) throw new Error(`Unknown point ${id}`);
  return [p.x, p.y];
}

export function arcInfo(s: Sketch, e: Extract<SketchEntity, { type: "arc" }>, idx?: Map<string, SketchPoint>): ArcInfo {
  const c = P(s, e.center, idx), a = P(s, e.start, idx), b = P(s, e.end, idx);
  const radius = Math.hypot(a[0] - c[0], a[1] - c[1]);
  const startAngle = angleOf([a[0] - c[0], a[1] - c[1]]);
  const endAngle = angleOf([b[0] - c[0], b[1] - c[1]]);
  let sweep = normAngle(endAngle - startAngle);
  if (sweep < 1e-12) sweep = TAU;
  return { center: c, radius, startAngle, endAngle, sweep };
}

/** Segment2D representation of an entity (arcs as 3-point arcs, splines as interpolating B-splines). */
export function entityToSegments(s: Sketch, entity: SketchEntity | string, idx?: Map<string, SketchPoint>): Segment2D[] {
  const e = typeof entity === "string" ? getEntity(s, entity) : entity;
  if (!e) return [];
  switch (e.type) {
    case "line":
      return [{ type: "line", a: P(s, e.p1, idx), b: P(s, e.p2, idx) }];
    case "arc": {
      const g = arcInfo(s, e, idx);
      if (g.radius < 1e-12) return [];
      if (g.sweep >= TAU - 1e-12) return [{ type: "circle", center: g.center, radius: g.radius }];
      return [{ type: "arc", start: P(s, e.start, idx), mid: polar(g.center, g.radius, g.startAngle + g.sweep / 2), end: P(s, e.end, idx) }];
    }
    case "circle":
      return [{ type: "circle", center: P(s, e.center, idx), radius: e.radius }];
    case "spline": {
      const pts = e.points.map((id) => P(s, id, idx));
      if (pts.length < 2) return [];
      return [{ type: "bspline", curve: interpolateBSpline2(pts, { closed: e.closed, degree: 3 }) }];
    }
  }
}

export interface PickResult {
  kind: "point" | "entity";
  id: string;
  distance: number;
  /** closest location on the picked item */
  point: Vec2;
}

/**
 * Pick the item nearest to p within tol: points have priority over entities. Construction geometry
 * is included unless `includeConstruction` is false; `filter` can exclude ids.
 */
export function findNearest(
  s: Sketch,
  p: Vec2,
  tol: number,
  opts: { includeConstruction?: boolean; filter?: (id: string) => boolean; pointsOnly?: boolean; entitiesOnly?: boolean } = {},
): PickResult | null {
  const filter = opts.filter ?? (() => true);
  if (!opts.entitiesOnly) {
    let best: PickResult | null = null;
    for (const q of s.points) {
      if (!filter(q.id)) continue;
      const d = Math.hypot(q.x - p[0], q.y - p[1]);
      if (d <= tol && (!best || d < best.distance)) best = { kind: "point", id: q.id, distance: d, point: [q.x, q.y] };
    }
    if (best || opts.pointsOnly) return best;
  }
  const idx = new Map(s.points.map((q) => [q.id, q]));
  let best: PickResult | null = null;
  for (const e of s.entities) {
    if (opts.includeConstruction === false && e.construction) continue;
    if (!filter(e.id)) continue;
    for (const seg of entityToSegments(s, e, idx)) {
      const c = closestPointOnSegment(seg, p);
      if (c.distance <= tol && (!best || c.distance < best.distance)) best = { kind: "entity", id: e.id, distance: c.distance, point: c.point };
    }
  }
  return best;
}

/** Bounding box of all points and entity geometry; null for an empty sketch. */
export function sketchBounds(s: Sketch): { min: Vec2; max: Vec2 } | null {
  if (!s.points.length) return null;
  const min: Vec2 = [Infinity, Infinity], max: Vec2 = [-Infinity, -Infinity];
  const add = (x: number, y: number) => {
    min[0] = Math.min(min[0], x);
    min[1] = Math.min(min[1], y);
    max[0] = Math.max(max[0], x);
    max[1] = Math.max(max[1], y);
  };
  const idx = new Map(s.points.map((q) => [q.id, q]));
  for (const q of s.points) add(q.x, q.y);
  for (const e of s.entities) {
    if (e.type === "circle") {
      const c = idx.get(e.center);
      if (c) {
        add(c.x - e.radius, c.y - e.radius);
        add(c.x + e.radius, c.y + e.radius);
      }
    } else if (e.type === "arc") {
      const g = arcInfo(s, e, idx);
      for (let k = 0; k <= 32; k++) {
        const q = polar(g.center, g.radius, g.startAngle + (g.sweep * k) / 32);
        add(q[0], q[1]);
      }
    }
  }
  return { min, max };
}

// ---------------------------------------------------------------------------
// Measurement (current values for dimension display)
// ---------------------------------------------------------------------------

type Resolved = { kind: "point"; p: Vec2 } | { kind: "line"; a: Vec2; b: Vec2 } | { kind: "curve"; c: Vec2; r: number; arc?: ArcInfo };

function resolve(s: Sketch, id: string, idx: Map<string, SketchPoint>): Resolved | null {
  const p = idx.get(id);
  if (p) return { kind: "point", p: [p.x, p.y] };
  const e = getEntity(s, id);
  if (!e) return null;
  if (e.type === "line") return { kind: "line", a: P(s, e.p1, idx), b: P(s, e.p2, idx) };
  if (e.type === "circle") return { kind: "curve", c: P(s, e.center, idx), r: e.radius };
  if (e.type === "arc") {
    const g = arcInfo(s, e, idx);
    return { kind: "curve", c: g.center, r: g.radius, arc: g };
  }
  return null;
}

function pointLineDistance(p: Vec2, a: Vec2, b: Vec2): number {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const L = Math.hypot(dx, dy);
  if (L < 1e-300) return Math.hypot(p[0] - a[0], p[1] - a[1]);
  return Math.abs(dx * (p[1] - a[1]) - dy * (p[0] - a[0])) / L;
}

/**
 * Current value of a dimension-like constraint {type, refs} (mm or radians), or null when it cannot
 * be measured. Distances are unsigned; angles are in [0, PI] between the directed lines.
 */
export function measure(s: Sketch, c: { type: string; refs: string[] }): number | null {
  const idx = new Map(s.points.map((q) => [q.id, q]));
  const r = c.refs.map((id) => resolve(s, id, idx));
  if (r.some((x) => !x)) return null;
  const R = r as Resolved[];
  const pt = (x: Resolved): Vec2 | null => (x.kind === "point" ? x.p : x.kind === "curve" ? x.c : null);
  switch (c.type) {
    case "distance": {
      if (R.length === 1) return R[0].kind === "line" ? Math.hypot(R[0].b[0] - R[0].a[0], R[0].b[1] - R[0].a[1]) : null;
      const [x, y] = R;
      const px = pt(x), py = pt(y);
      if (px && py) return Math.hypot(px[0] - py[0], px[1] - py[1]);
      if (px && y.kind === "line") return pointLineDistance(px, y.a, y.b);
      if (py && x.kind === "line") return pointLineDistance(py, x.a, x.b);
      if (x.kind === "line" && y.kind === "line") {
        const m: Vec2 = [(y.a[0] + y.b[0]) / 2, (y.a[1] + y.b[1]) / 2];
        return pointLineDistance(m, x.a, x.b);
      }
      return null;
    }
    case "horizontalDistance":
    case "verticalDistance": {
      const k = c.type === "horizontalDistance" ? 0 : 1;
      if (R.length === 1 && R[0].kind === "line") return Math.abs(R[0].b[k] - R[0].a[k]);
      if (R.length !== 2) return null;
      const a = pt(R[0]), b = pt(R[1]);
      return a && b ? Math.abs(b[k] - a[k]) : null;
    }
    case "radius":
      return R[0]?.kind === "curve" ? R[0].r : null;
    case "diameter":
      return R[0]?.kind === "curve" ? 2 * R[0].r : null;
    case "length": {
      const x = R[0];
      if (!x) return null;
      if (x.kind === "line") return Math.hypot(x.b[0] - x.a[0], x.b[1] - x.a[1]);
      if (x.kind === "curve" && x.arc) return x.arc.radius * x.arc.sweep;
      return null;
    }
    case "angle": {
      const [x, y] = R;
      if (x?.kind !== "line" || y?.kind !== "line") return null;
      const u: Vec2 = [x.b[0] - x.a[0], x.b[1] - x.a[1]];
      const w: Vec2 = [y.b[0] - y.a[0], y.b[1] - y.a[1]];
      return Math.abs(Math.atan2(u[0] * w[1] - u[1] * w[0], u[0] * w[0] + u[1] * w[1]));
    }
    default:
      return null;
  }
}
