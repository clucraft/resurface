/**
 * Sketch editing operations: trim, extend, corner fillet, offset, mirror, linear / circular pattern,
 * plus 2D intersection helpers.
 *
 * Like the model builders, operations update the sketch passed in, atomically: they work on a copy and
 * commit it into the given sketch object only on success. On failure (ok = false, with a message) the
 * sketch is left untouched. `OpResult.sketch` is the (same) updated sketch object.
 */
import type { Segment2D, Vec2 } from "../core/types";
import {
  TAU,
  angleOf,
  arcGeomFrom3,
  arcParamOf,
  circleCircleIntersection,
  closestPointOnSegment,
  lineCircleIntersection,
  lineLineIntersection,
  normAngle,
  polar,
  segmentIntersections,
  segmentPointAt,
  type SegmentHit,
} from "./geom2d";
import {
  addArc,
  addConstraint,
  addLine,
  addPoint,
  arcInfo,
  cloneSketch,
  deleteEntities,
  entityPointIds,
  entityToSegments,
  genId,
  getEntity,
  getPoint,
  isDimension,
  sketchBounds,
  validateConstraint,
  type Constraint,
  type Sketch,
  type SketchEntity,
} from "./model";

export {
  circleCircleIntersection,
  lineCircleIntersection,
  lineLineIntersection,
  segmentIntersections,
} from "./geom2d";

export interface OpResult {
  sketch: Sketch;
  ok: boolean;
  /** reason when ok is false */
  message?: string;
  /** ids of created points/entities/constraints */
  created: string[];
  /** ids of removed entities/points */
  removed: string[];
}

const fail = (s: Sketch, message: string): OpResult => ({ sketch: s, ok: false, message, created: [], removed: [] });

/** Commit the working copy into the caller's sketch object. */
function commit(target: Sketch, work: Sketch, created: string[], removed: string[]): OpResult {
  target.points = work.points;
  target.entities = work.entities;
  target.constraints = work.constraints;
  target.nextId = work.nextId;
  return { sketch: target, ok: true, created, removed };
}

// ---------------------------------------------------------------------------
// Intersections
// ---------------------------------------------------------------------------

/** Intersections of the bounded segment a-b with a 3-point arc segment. */
export function segmentArcIntersection(a: Vec2, b: Vec2, arc: Extract<Segment2D, { type: "arc" }>, tol = 1e-9): SegmentHit[] {
  return segmentIntersections({ type: "line", a, b }, arc, tol);
}

/** Intersections between two sketch entities (t1/t2 normalised along each entity, see segmentPointAt). */
export function entityIntersections(s: Sketch, idA: string, idB: string, tol?: number): SegmentHit[] {
  const A = entityToSegments(s, idA), B = entityToSegments(s, idB);
  if (A.length !== 1 || B.length !== 1) return [];
  return segmentIntersections(A[0], B[0], tol ?? defaultTol(s));
}

function defaultTol(s: Sketch): number {
  const b = sketchBounds(s);
  const size = b ? Math.max(1, Math.hypot(b.max[0] - b.min[0], b.max[1] - b.min[1])) : 1;
  return 1e-7 * size;
}

function pos(s: Sketch, id: string): Vec2 {
  const p = getPoint(s, id);
  if (!p) throw new Error(`Unknown point ${id}`);
  return [p.x, p.y];
}

/** Entities (other than `exclude`) using point id as an endpoint or center. */
function usersOf(s: Sketch, pid: string, exclude?: string): SketchEntity[] {
  return s.entities.filter((e) => e.id !== exclude && entityPointIds(e).includes(pid));
}

/** Remove a point when no entity uses it anymore (with its constraints). */
function dropIfOrphan(s: Sketch, pid: string, removed: string[]): void {
  if (usersOf(s, pid).length) return;
  if (!getPoint(s, pid)) return;
  s.points = s.points.filter((p) => p.id !== pid);
  s.constraints = s.constraints.filter((c) => !c.refs.includes(pid));
  removed.push(pid);
}

/** Constraints that depend on the length/extent of an entity (invalid after trimming/extending). */
function dropExtentConstraints(s: Sketch, eid: string): void {
  s.constraints = s.constraints.filter((c) => {
    if (!c.refs.includes(eid)) return true;
    if (c.type === "length" || c.type === "midpoint") return false;
    if ((c.type === "distance" || c.type === "horizontalDistance" || c.type === "verticalDistance") && c.refs.length === 1) return false;
    const e = getEntity(s, eid);
    if (c.type === "equal" && e?.type === "line") return false;
    return true;
  });
}

interface Cut {
  t: number;
  point: Vec2;
  other: string;
}

/** All cuts of entity e by other entities (param t normalised along e). */
function cutsOf(s: Sketch, e: SketchEntity, tol: number): Cut[] {
  const segE = entityToSegments(s, e);
  if (segE.length !== 1) return [];
  const out: Cut[] = [];
  for (const o of s.entities) {
    if (o.id === e.id) continue;
    const segO = entityToSegments(s, o);
    if (segO.length !== 1) continue;
    for (const h of segmentIntersections(segE[0], segO[0], tol)) out.push({ t: h.t1, point: h.point, other: o.id });
  }
  return out.sort((a, b) => a.t - b.t);
}

/**
 * Point id for a new end at `p` lying on entity `other`: reuses an endpoint of `other` at that spot,
 * otherwise creates a point constrained onto `other`.
 */
function endPointOn(s: Sketch, p: Vec2, other: string | null, tol: number, created: string[]): string {
  if (other) {
    const o = getEntity(s, other);
    if (o) {
      for (const pid of entityPointIds(o)) {
        if (o.type === "arc" && pid === o.center) continue;
        if (o.type === "circle") continue;
        const q = pos(s, pid);
        if (Math.hypot(q[0] - p[0], q[1] - p[1]) <= tol) return pid;
      }
    }
  }
  const id = addPoint(s, p);
  created.push(id);
  if (other) {
    const o = getEntity(s, other);
    if (o && (o.type === "line" || o.type === "arc" || o.type === "circle")) created.push(addConstraint(s, { type: "pointOnEntity", refs: [id, other] }));
  }
  return id;
}

// ---------------------------------------------------------------------------
// Trim / extend
// ---------------------------------------------------------------------------

/**
 * Trim the portion of an entity around `pickPoint` that lies between its nearest intersections with
 * other entities. Without any intersection on either side the entity is deleted.
 */
export function trimAt(sketch: Sketch, entityId: string, pickPoint: Vec2, tol?: number): OpResult {
  const s = cloneSketch(sketch);
  const e = getEntity(s, entityId);
  if (!e) return fail(sketch, `Unknown entity ${entityId}`);
  if (e.type === "spline") return fail(sketch, "Trimming splines is not supported");
  const TOL = tol ?? defaultTol(s);
  const seg = entityToSegments(s, e)[0];
  if (!seg) return fail(sketch, "Degenerate entity");
  const tp = closestPointOnSegment(seg, pickPoint).t;
  const created: string[] = [];
  const removed: string[] = [];
  const L = Math.max(1e-300, segLen(seg));
  let cuts = cutsOf(s, e, TOL);

  if (e.type === "circle") {
    // unique cut angles
    const ts: Cut[] = [];
    for (const c of cuts) if (!ts.some((x) => Math.min(Math.abs(x.t - c.t), 1 - Math.abs(x.t - c.t)) * L <= TOL)) ts.push(c);
    if (ts.length < 2) {
      deleteEntities(s, [e.id]);
      removed.push(e.id);
      return commit(sketch, s, created, removed);
    }
    let lo = ts[ts.length - 1], hi = ts[0];
    for (const c of ts) {
      if (c.t <= tp) lo = c;
    }
    for (let i = ts.length - 1; i >= 0; i--) if (ts[i].t > tp) hi = ts[i];
    if (tp < ts[0].t || tp >= ts[ts.length - 1].t) {
      lo = ts[ts.length - 1];
      hi = ts[0];
    }
    // remaining arc: CCW from hi to lo
    const startId = endPointOn(s, hi.point, hi.other, TOL, created);
    const endId = endPointOn(s, lo.point, lo.other, TOL, created);
    const idx = s.entities.findIndex((x) => x.id === e.id);
    const arcE: SketchEntity = { id: e.id, type: "arc", center: e.center, start: startId, end: endId };
    if (e.construction) arcE.construction = true;
    s.entities[idx] = arcE;
    return commit(sketch, s, created, removed);
  }

  // open entities: drop cuts at the own end points
  cuts = cuts.filter((c) => c.t * L > TOL && (1 - c.t) * L > TOL);
  let lo: Cut | null = null, hi: Cut | null = null;
  for (const c of cuts) {
    if (c.t < tp) lo = c;
    else if (!hi) hi = c;
  }
  if (!lo && !hi) {
    deleteEntities(s, [e.id]);
    removed.push(e.id);
    return commit(sketch, s, created, removed);
  }
  const startKey = e.type === "line" ? "p1" : "start";
  const endKey = e.type === "line" ? "p2" : "end";
  const ent = e as Record<string, unknown> & SketchEntity;
  const oldStart = ent[startKey] as string, oldEnd = ent[endKey] as string;
  dropExtentConstraints(s, e.id);
  if (lo && hi) {
    // split: [0, lo] stays in e, [hi, 1] becomes a new entity
    const a = endPointOn(s, lo.point, lo.other, TOL, created);
    const b = endPointOn(s, hi.point, hi.other, TOL, created);
    ent[endKey] = a;
    let nid: string;
    if (e.type === "line") {
      nid = addLine(s, b, oldEnd, { construction: e.construction });
      created.push(nid);
      for (const c of sketch.constraints)
        if ((c.type === "horizontal" || c.type === "vertical") && c.refs.length === 1 && c.refs[0] === e.id) created.push(addConstraint(s, { type: c.type, refs: [nid] }));
      created.push(addConstraint(s, { type: "collinear", refs: [e.id, nid] }));
    } else {
      const ae = e as Extract<SketchEntity, { type: "arc" }>;
      nid = addArc(s, ae.center, b, oldEnd, { construction: e.construction });
      created.push(nid);
      created.push(addConstraint(s, { type: "equal", refs: [e.id, nid] }));
    }
  } else if (lo) {
    ent[endKey] = endPointOn(s, lo.point, lo.other, TOL, created);
    dropIfOrphan(s, oldEnd, removed);
  } else if (hi) {
    ent[startKey] = endPointOn(s, hi.point, hi.other, TOL, created);
    dropIfOrphan(s, oldStart, removed);
  }
  return commit(sketch, s, created, removed);
}

function segLen(seg: Segment2D): number {
  if (seg.type === "line") return Math.hypot(seg.b[0] - seg.a[0], seg.b[1] - seg.a[1]);
  if (seg.type === "circle") return TAU * seg.radius;
  if (seg.type === "arc") {
    const g = arcGeomFrom3(seg.start, seg.mid, seg.end);
    return g ? g.radius * Math.abs(g.sweep) : Math.hypot(seg.end[0] - seg.start[0], seg.end[1] - seg.start[1]);
  }
  return 1;
}

/**
 * Extend the end of a line or arc nearest to `pickPoint` up to the nearest intersection with another
 * entity. Returns ok = false when nothing is hit.
 */
export function extend(sketch: Sketch, entityId: string, pickPoint: Vec2, tol?: number): OpResult {
  const s = cloneSketch(sketch);
  const e = getEntity(s, entityId);
  if (!e) return fail(sketch, `Unknown entity ${entityId}`);
  if (e.type !== "line" && e.type !== "arc") return fail(sketch, "Only lines and arcs can be extended");
  const TOL = tol ?? defaultTol(s);
  const created: string[] = [];
  const removed: string[] = [];
  const b = sketchBounds(s)!;
  const big = 10 * Math.max(1, Math.hypot(b.max[0] - b.min[0], b.max[1] - b.min[1]));
  let probe: Segment2D;
  let atEnd: boolean;
  let from: Vec2;
  if (e.type === "line") {
    const p1 = pos(s, e.p1), p2 = pos(s, e.p2);
    atEnd = Math.hypot(pickPoint[0] - p2[0], pickPoint[1] - p2[1]) <= Math.hypot(pickPoint[0] - p1[0], pickPoint[1] - p1[1]);
    const [o, f] = atEnd ? [p2, p1] : [p1, p2];
    const d: Vec2 = [o[0] - f[0], o[1] - f[1]];
    const l = Math.hypot(d[0], d[1]);
    if (l < 1e-300) return fail(sketch, "Degenerate line");
    from = o;
    probe = { type: "line", a: o, b: [o[0] + (d[0] / l) * big, o[1] + (d[1] / l) * big] };
  } else {
    const g = arcInfo(s, e);
    const st = pos(s, e.start), en = pos(s, e.end);
    atEnd = Math.hypot(pickPoint[0] - en[0], pickPoint[1] - en[1]) <= Math.hypot(pickPoint[0] - st[0], pickPoint[1] - st[1]);
    const free = TAU - g.sweep;
    if (free < 1e-9) return fail(sketch, "Arc is already closed");
    // the complementary arc, traversed away from the chosen end
    const a0 = atEnd ? g.startAngle + g.sweep : g.startAngle;
    const sw = atEnd ? free : -free;
    from = atEnd ? en : st;
    probe = { type: "arc", start: polar(g.center, g.radius, a0), mid: polar(g.center, g.radius, a0 + sw / 2), end: polar(g.center, g.radius, a0 + sw * (1 - 1e-9)) };
  }
  let best: { t: number; point: Vec2; other: string } | null = null;
  for (const o of s.entities) {
    if (o.id === e.id) continue;
    const segO = entityToSegments(s, o);
    if (segO.length !== 1) continue;
    for (const h of segmentIntersections(probe, segO[0], TOL)) {
      if (Math.hypot(h.point[0] - from[0], h.point[1] - from[1]) <= TOL * 10) continue;
      if (!best || h.t1 < best.t) best = { t: h.t1, point: h.point, other: o.id };
    }
  }
  if (!best) return fail(sketch, "Nothing to extend to");
  const key = e.type === "line" ? (atEnd ? "p2" : "p1") : atEnd ? "end" : "start";
  const ent = e as Record<string, unknown> & SketchEntity;
  const old = ent[key] as string;
  dropExtentConstraints(s, e.id);
  if (usersOf(s, old, e.id).length === 0 && !s.constraints.some((c) => c.refs.includes(old) && c.type === "fix") && !getPoint(s, old)!.fixed) {
    // free end: move the point and pin it on the target
    const p = getPoint(s, old)!;
    p.x = best.point[0];
    p.y = best.point[1];
    const tgt = getEntity(s, best.other);
    if (tgt && (tgt.type === "line" || tgt.type === "arc" || tgt.type === "circle")) created.push(addConstraint(s, { type: "pointOnEntity", refs: [old, best.other] }));
  } else {
    ent[key] = endPointOn(s, best.point, best.other, TOL, created);
  }
  return commit(sketch, s, created, removed);
}

// ---------------------------------------------------------------------------
// Fillet
// ---------------------------------------------------------------------------

/**
 * Round the corner where exactly two lines meet at `pointId` with a tangent arc of `radius`. The lines
 * are trimmed to the tangent points and tangent constraints (and a radius dimension) are added.
 */
export function filletCorner(sketch: Sketch, pointId: string, radius: number, opts: { dimension?: boolean } = {}): OpResult {
  const s = cloneSketch(sketch);
  if (!(radius > 0)) return fail(sketch, "Fillet radius must be positive");
  if (!getPoint(s, pointId)) return fail(sketch, `Unknown point ${pointId}`);
  const users = usersOf(s, pointId);
  const lines = users.filter((e) => e.type === "line") as Extract<SketchEntity, { type: "line" }>[];
  if (lines.length !== 2 || users.length !== 2) return fail(sketch, "Fillet needs exactly two lines meeting at the point");
  const P = pos(s, pointId);
  const far = lines.map((l) => (l.p1 === pointId ? l.p2 : l.p1));
  if (far[0] === pointId || far[1] === pointId) return fail(sketch, "Degenerate line");
  const A = pos(s, far[0]), B = pos(s, far[1]);
  const la = Math.hypot(A[0] - P[0], A[1] - P[1]), lb = Math.hypot(B[0] - P[0], B[1] - P[1]);
  if (la < 1e-12 || lb < 1e-12) return fail(sketch, "Degenerate line");
  const u1: Vec2 = [(A[0] - P[0]) / la, (A[1] - P[1]) / la];
  const u2: Vec2 = [(B[0] - P[0]) / lb, (B[1] - P[1]) / lb];
  const theta = Math.acos(Math.max(-1, Math.min(1, u1[0] * u2[0] + u1[1] * u2[1])));
  if (theta < 1e-6 || theta > Math.PI - 1e-6) return fail(sketch, "Lines are collinear");
  const d = radius / Math.tan(theta / 2);
  if (d >= la - 1e-9 || d >= lb - 1e-9) return fail(sketch, "Fillet radius too large for these lines");
  const T1: Vec2 = [P[0] + u1[0] * d, P[1] + u1[1] * d];
  const T2: Vec2 = [P[0] + u2[0] * d, P[1] + u2[1] * d];
  const bis: Vec2 = [u1[0] + u2[0], u1[1] + u2[1]];
  const bl = Math.hypot(bis[0], bis[1]);
  const h = radius / Math.sin(theta / 2);
  const C: Vec2 = [P[0] + (bis[0] / bl) * h, P[1] + (bis[1] / bl) * h];
  const created: string[] = [];
  const removed: string[] = [];
  const t1 = addPoint(s, T1), t2 = addPoint(s, T2), c = addPoint(s, C);
  created.push(t1, t2, c);
  for (const [i, l] of lines.entries()) {
    const t = i === 0 ? t1 : t2;
    if (l.p1 === pointId) l.p1 = t;
    else l.p2 = t;
    dropExtentConstraints(s, l.id);
  }
  const ccw = (T1[0] - C[0]) * (T2[1] - C[1]) - (T1[1] - C[1]) * (T2[0] - C[0]) > 0;
  const arc = ccw ? addArc(s, c, t1, t2) : addArc(s, c, t2, t1);
  created.push(arc);
  created.push(addConstraint(s, { type: "tangent", refs: [lines[0].id, arc] }));
  created.push(addConstraint(s, { type: "tangent", refs: [lines[1].id, arc] }));
  if (opts.dimension ?? true) created.push(addConstraint(s, { type: "radius", refs: [arc], value: radius }));
  dropIfOrphan(s, pointId, removed);
  return commit(sketch, s, created, removed);
}

// ---------------------------------------------------------------------------
// Offset
// ---------------------------------------------------------------------------

type Prim = { kind: "line"; a: Vec2; b: Vec2 } | { kind: "arc"; c: Vec2; r: number };

/**
 * Offset entities by `distance`. Connected selections are offset as chains with re-joined corners.
 * Sign convention: closed chains and lone circles/arcs grow outward for positive distances; open
 * chains move to the left of their direction (the direction of the first entity, lines p1->p2,
 * arcs CCW). Offset arcs reuse the original center point.
 */
export function offsetEntities(sketch: Sketch, ids: string[], distance: number): OpResult {
  const s = cloneSketch(sketch);
  const created: string[] = [];
  const sel = ids.map((id) => getEntity(s, id)).filter((e): e is SketchEntity => !!e);
  if (!sel.length) return fail(sketch, "Nothing selected");
  if (distance === 0) return fail(sketch, "Offset distance is zero");
  const TOL = defaultTol(s);

  // standalone circles and splines
  for (const e of sel) {
    if (e.type === "circle") {
      const r = e.radius + distance;
      if (r <= TOL) return fail(sketch, "Offset collapses a circle");
      const id = genId(s, "c");
      const ne: SketchEntity = { id, type: "circle", center: e.center, radius: r };
      if (e.construction) ne.construction = true;
      s.entities.push(ne);
      created.push(id);
    }
  }
  const chainable = sel.filter((e) => e.type === "line" || e.type === "arc" || e.type === "spline");
  const chains = buildChains(s, chainable);
  for (const ch of chains) {
    // orientation: closed chains CCW, offset outward (right side) => left offset with -distance
    let dist = distance;
    if (ch.closed) {
      const area = chainArea(s, ch);
      if (area > 0) dist = -distance;
    } else if (ch.items.length === 1 && ch.items[0].e.type === "arc") {
      // lone arc: outward = larger radius
      dist = ch.items[0].fwd ? -distance : distance;
    }
    const res = offsetChain(s, ch, dist, TOL, created);
    if (!res) return fail(sketch, "Offset collapses an arc or spline");
  }
  return commit(sketch, s, created, []);
}

interface ChainItem {
  e: SketchEntity;
  /** traversed from start to end (line p1->p2, arc start->end CCW, spline first->last) */
  fwd: boolean;
}

interface Chain {
  items: ChainItem[];
  closed: boolean;
}

function ends(e: SketchEntity): [string, string] {
  if (e.type === "line") return [e.p1, e.p2];
  if (e.type === "arc") return [e.start, e.end];
  if (e.type === "spline") return [e.points[0], e.points[e.points.length - 1]];
  return ["", ""];
}

function buildChains(s: Sketch, ents: SketchEntity[]): Chain[] {
  const byPoint = new Map<string, SketchEntity[]>();
  for (const e of ents) {
    if (e.type === "spline" && e.closed) continue;
    for (const p of ends(e)) {
      let l = byPoint.get(p);
      if (!l) byPoint.set(p, (l = []));
      l.push(e);
    }
  }
  const used = new Set<string>();
  const chains: Chain[] = [];
  const walkFrom = (start: SketchEntity, fwd: boolean): Chain => {
    const items: ChainItem[] = [{ e: start, fwd }];
    used.add(start.id);
    const first = fwd ? ends(start)[0] : ends(start)[1];
    let cur = fwd ? ends(start)[1] : ends(start)[0];
    for (;;) {
      const nb = (byPoint.get(cur) ?? []).filter((x) => !used.has(x.id));
      if ((byPoint.get(cur) ?? []).length !== 2 || nb.length !== 1) break;
      const n = nb[0];
      const [a, b] = ends(n);
      const f = a === cur;
      items.push({ e: n, fwd: f });
      used.add(n.id);
      cur = f ? b : a;
    }
    return { items, closed: cur === first && items.length > 1 };
  };
  // start at chain ends first
  for (const e of ents) {
    if (used.has(e.id) || (e.type === "spline" && e.closed)) continue;
    const [a, b] = ends(e);
    if ((byPoint.get(a) ?? []).length !== 2) chains.push(walkFrom(e, true));
    else if ((byPoint.get(b) ?? []).length !== 2) chains.push(walkFrom(e, false));
  }
  for (const e of ents) if (!used.has(e.id)) chains.push(walkFrom(e, true));
  // open chains run in the natural direction of their first entity
  return chains.map((c) =>
    !c.closed && !c.items[0].fwd ? { closed: false, items: c.items.slice().reverse().map((it) => ({ e: it.e, fwd: !it.fwd })) } : c,
  );
}

function chainArea(s: Sketch, ch: Chain): number {
  let a = 0;
  for (const it of ch.items) {
    const seg = entityToSegments(s, it.e)[0];
    if (!seg) continue;
    const pts: Vec2[] = [];
    for (let k = 0; k <= 16; k++) pts.push(segmentPointAt(seg, it.fwd ? k / 16 : 1 - k / 16));
    for (let k = 0; k < pts.length - 1; k++) a += pts[k][0] * pts[k + 1][1] - pts[k][1] * pts[k + 1][0];
  }
  return a / 2;
}

function offsetChain(s: Sketch, ch: Chain, dist: number, tol: number, created: string[]): boolean {
  // offset primitives (in traversal direction, left side positive)
  const prims: (Prim | null)[] = [];
  const naiveStart: Vec2[] = [];
  const naiveEnd: Vec2[] = [];
  for (const it of ch.items) {
    const e = it.e;
    const [ia, ib] = ends(e);
    const pa = pos(s, it.fwd ? ia : ib), pb = pos(s, it.fwd ? ib : ia);
    if (e.type === "line") {
      const d: Vec2 = [pb[0] - pa[0], pb[1] - pa[1]];
      const l = Math.hypot(d[0], d[1]) || 1;
      const n: Vec2 = [(-d[1] / l) * dist, (d[0] / l) * dist];
      const a: Vec2 = [pa[0] + n[0], pa[1] + n[1]], b: Vec2 = [pb[0] + n[0], pb[1] + n[1]];
      prims.push({ kind: "line", a, b });
      naiveStart.push(a);
      naiveEnd.push(b);
    } else if (e.type === "arc") {
      const g = arcInfo(s, e);
      // traversed CCW (fwd): left = toward center
      const r = it.fwd ? g.radius - dist : g.radius + dist;
      if (r <= tol) return false;
      prims.push({ kind: "arc", c: g.center, r });
      naiveStart.push(polar(g.center, r, angleOf([pa[0] - g.center[0], pa[1] - g.center[1]])));
      naiveEnd.push(polar(g.center, r, angleOf([pb[0] - g.center[0], pb[1] - g.center[1]])));
    } else {
      prims.push(null);
      naiveStart.push(pa);
      naiveEnd.push(pb);
    }
  }
  const n = ch.items.length;
  // junction points
  const J: Vec2[] = [];
  const nj = ch.closed ? n : n - 1;
  for (let j = 0; j < nj; j++) {
    const A = prims[j], B = prims[(j + 1) % n];
    const naive: Vec2 = [(naiveEnd[j][0] + naiveStart[(j + 1) % n][0]) / 2, (naiveEnd[j][1] + naiveStart[(j + 1) % n][1]) / 2];
    J.push(A && B ? primJunction(A, B, naive) : naive);
  }
  // splines: offset fit points along the curve normal
  const pointIds: string[] = [];
  const nPts = ch.closed ? n : n + 1;
  for (let k = 0; k < nPts; k++) {
    let p: Vec2;
    if (k === 0 && !ch.closed) p = naiveStart[0];
    else if (k === n && !ch.closed) p = naiveEnd[n - 1];
    else p = J[(k - 1 + nj) % nj];
    const id = addPoint(s, p);
    created.push(id);
    pointIds.push(id);
  }
  const newIds: string[] = [];
  for (let k = 0; k < n; k++) {
    const it = ch.items[k];
    const e = it.e;
    const a = pointIds[k], b = pointIds[(k + 1) % nPts];
    const constr = { construction: e.construction };
    let id: string;
    if (e.type === "line") id = it.fwd ? addLine(s, a, b, constr) : addLine(s, b, a, constr);
    else if (e.type === "arc") {
      // keep the arc CCW: forward traversal = start..end
      id = it.fwd ? addArc(s, e.center, a, b, constr) : addArc(s, e.center, b, a, constr);
    } else if (e.type === "spline") {
      const seg = entityToSegments(s, e)[0];
      const fit = e.points.map((pid) => pos(s, pid));
      const off: string[] = [];
      for (let i = 0; i < fit.length; i++) {
        const isFirst = i === 0, isLast = i === fit.length - 1;
        if ((isFirst && it.fwd) || (isLast && !it.fwd)) {
          off.push(a);
          continue;
        }
        if ((isLast && it.fwd) || (isFirst && !it.fwd)) {
          off.push(b);
          continue;
        }
        const c = seg ? closestPointOnSegment(seg, fit[i]) : null;
        const t = c ? c.t : 0;
        const q1 = seg ? segmentPointAt(seg, Math.min(1, t + 1e-4)) : fit[i];
        const q0 = seg ? segmentPointAt(seg, Math.max(0, t - 1e-4)) : fit[i];
        const d: Vec2 = [q1[0] - q0[0], q1[1] - q0[1]];
        const l = Math.hypot(d[0], d[1]) || 1;
        const sgn = it.fwd ? 1 : -1;
        const pid = addPoint(s, [fit[i][0] - (d[1] / l) * dist * sgn, fit[i][1] + (d[0] / l) * dist * sgn]);
        created.push(pid);
        off.push(pid);
      }
      id = genId(s, "s");
      const ne: SketchEntity = { id, type: "spline", points: off };
      if (e.construction) ne.construction = true;
      s.entities.push(ne);
    } else continue;
    created.push(id);
    newIds.push(id);
  }
  // copy tangent / orientation constraints between chain members
  const map = new Map(ch.items.map((it, k) => [it.e.id, newIds[k]]));
  for (const c of [...s.constraints]) {
    if (!["tangent", "parallel", "perpendicular", "horizontal", "vertical"].includes(c.type)) continue;
    if (!c.refs.every((r) => map.has(r))) continue;
    const refs = c.refs.map((r) => map.get(r)!);
    if (!validateConstraint(s, { type: c.type, refs })) created.push(addConstraint(s, { type: c.type as "tangent", refs }));
  }
  return true;
}

function primJunction(A: Prim, B: Prim, naive: Vec2): Vec2 {
  let cands: Vec2[] = [];
  if (A.kind === "line" && B.kind === "line") {
    const r = lineLineIntersection(A.a, A.b, B.a, B.b);
    return r ? r.point : naive;
  }
  if (A.kind === "line" && B.kind === "arc") cands = lineCircleIntersection(A.a, A.b, B.c, B.r).map((h) => h.point);
  else if (A.kind === "arc" && B.kind === "line") cands = lineCircleIntersection(B.a, B.b, A.c, A.r).map((h) => h.point);
  else if (A.kind === "arc" && B.kind === "arc") cands = circleCircleIntersection(A.c, A.r, B.c, B.r);
  let best = naive, bd = Infinity;
  for (const c of cands) {
    const d = Math.hypot(c[0] - naive[0], c[1] - naive[1]);
    if (d < bd) {
      bd = d;
      best = c;
    }
  }
  return best;
}

// ---------------------------------------------------------------------------
// Copy / mirror / patterns
// ---------------------------------------------------------------------------

interface CopyOptions {
  /** reflection: arcs swap start/end to stay CCW */
  reflect?: boolean;
  /** keep the original point (no copy) when this returns true, e.g. points on a mirror line */
  keepPoint?: (pid: string, p: Vec2, q: Vec2) => boolean;
  /** which internal constraints to copy (default none) */
  copyConstraint?: (c: Constraint) => boolean;
}

/** Copy entities through a point map. Returns maps old->new for points and entities. */
export function copyEntities(
  s: Sketch,
  ids: string[],
  map: (p: Vec2) => Vec2,
  opts: CopyOptions = {},
): { points: Map<string, string>; entities: Map<string, string>; created: string[] } {
  const created: string[] = [];
  const pmap = new Map<string, string>();
  const emap = new Map<string, string>();
  const ents = ids.map((id) => getEntity(s, id)).filter((e): e is SketchEntity => !!e);
  const P = (pid: string): string => {
    let q = pmap.get(pid);
    if (q) return q;
    const p = pos(s, pid);
    const m = map(p);
    if (opts.keepPoint?.(pid, p, m)) q = pid;
    else {
      q = addPoint(s, m);
      created.push(q);
    }
    pmap.set(pid, q);
    return q;
  };
  for (const e of ents) {
    let id: string;
    const constr = { construction: e.construction };
    switch (e.type) {
      case "line":
        id = addLine(s, P(e.p1), P(e.p2), constr);
        break;
      case "arc":
        id = opts.reflect ? addArc(s, P(e.center), P(e.end), P(e.start), constr) : addArc(s, P(e.center), P(e.start), P(e.end), constr);
        break;
      case "circle": {
        id = genId(s, "c");
        const ne: SketchEntity = { id, type: "circle", center: P(e.center), radius: e.radius };
        if (e.construction) ne.construction = true;
        s.entities.push(ne);
        break;
      }
      case "spline": {
        id = genId(s, "s");
        const ne: SketchEntity = { id, type: "spline", points: e.points.map(P) };
        if (e.closed) ne.closed = true;
        if (e.construction) ne.construction = true;
        s.entities.push(ne);
        break;
      }
    }
    emap.set(e.id, id);
    created.push(id);
  }
  if (opts.copyConstraint) {
    const remap = (r: string) => emap.get(r) ?? pmap.get(r);
    for (const c of [...s.constraints]) {
      if (!opts.copyConstraint(c)) continue;
      if (!c.refs.every((r) => remap(r) !== undefined)) continue;
      const refs = c.refs.map((r) => remap(r)!);
      if (refs.every((r, i) => r === c.refs[i])) continue;
      if (c.type === "fix") continue;
      const input = isDimension(c) ? { type: c.type, refs, value: c.value, driving: c.driving } : { type: c.type, refs };
      if (!validateConstraint(s, input)) created.push(addConstraint(s, input as never));
    }
  }
  return { points: pmap, entities: emap, created };
}

const ORIENTATION_FREE = new Set<string>([
  "coincident",
  "parallel",
  "perpendicular",
  "tangent",
  "equal",
  "concentric",
  "midpoint",
  "pointOnEntity",
  "collinear",
]);

/**
 * Mirror entities about a line entity. Points on the mirror line are shared; every other point pair
 * gets a `symmetric` constraint (and mirrored circles an `equal` constraint), unless symmetric = false
 * in which case orientation-free internal constraints are copied instead.
 */
export function mirrorEntities(sketch: Sketch, ids: string[], lineId: string, opts: { symmetric?: boolean } = {}): OpResult {
  const s = cloneSketch(sketch);
  const ml = getEntity(s, lineId);
  if (!ml || ml.type !== "line") return fail(sketch, "Mirror axis must be a line");
  const A = pos(s, ml.p1), B = pos(s, ml.p2);
  const d: Vec2 = [B[0] - A[0], B[1] - A[1]];
  const L2 = d[0] * d[0] + d[1] * d[1];
  if (L2 < 1e-300) return fail(sketch, "Degenerate mirror line");
  const TOL = defaultTol(s);
  const reflect = (p: Vec2): Vec2 => {
    const t = ((p[0] - A[0]) * d[0] + (p[1] - A[1]) * d[1]) / L2;
    const f: Vec2 = [A[0] + d[0] * t, A[1] + d[1] * t];
    return [2 * f[0] - p[0], 2 * f[1] - p[1]];
  };
  const symmetric = opts.symmetric ?? true;
  const sel = ids.filter((id) => id !== lineId);
  const res = copyEntities(s, sel, reflect, {
    reflect: true,
    keepPoint: (_pid, p, q) => Math.hypot(p[0] - q[0], p[1] - q[1]) <= 2 * TOL,
    copyConstraint: symmetric ? undefined : (c) => ORIENTATION_FREE.has(c.type) || (isDimension(c) && c.type !== "horizontalDistance" && c.type !== "verticalDistance"),
  });
  const created = res.created;
  if (symmetric) {
    for (const [o, n] of res.points) if (o !== n) created.push(addConstraint(s, { type: "symmetric", refs: [o, n, lineId] }));
    for (const [o, n] of res.entities) {
      const e = getEntity(s, o);
      if (e?.type === "circle") created.push(addConstraint(s, { type: "equal", refs: [o, n] }));
    }
  }
  return commit(sketch, s, created, []);
}

/**
 * Linear pattern: `count` instances in total (the original included), each shifted by `step`.
 * Internal geometric constraints are copied (dimensions too with copyDimensions).
 */
export function linearPattern(sketch: Sketch, ids: string[], step: Vec2, count: number, opts: { copyDimensions?: boolean } = {}): OpResult {
  const s = cloneSketch(sketch);
  if (!(count >= 2)) return fail(sketch, "Pattern count must be at least 2");
  const created: string[] = [];
  for (let k = 1; k < count; k++) {
    const r = copyEntities(s, ids, (p) => [p[0] + step[0] * k, p[1] + step[1] * k], {
      copyConstraint: (c) => !isDimension(c) || !!opts.copyDimensions,
    });
    created.push(...r.created);
  }
  return commit(sketch, s, created, []);
}

/**
 * Circular pattern about `center`: `count` instances in total spread over `totalAngle` (default full
 * turn; for a partial angle the last instance lands exactly on totalAngle). Points at the center are
 * shared. Orientation-free internal constraints are copied.
 */
export function circularPattern(
  sketch: Sketch,
  ids: string[],
  center: Vec2,
  count: number,
  totalAngle = TAU,
  opts: { copyDimensions?: boolean } = {},
): OpResult {
  const s = cloneSketch(sketch);
  if (!(count >= 2)) return fail(sketch, "Pattern count must be at least 2");
  const full = Math.abs(Math.abs(totalAngle) - TAU) < 1e-9;
  const stepA = full ? totalAngle / count : totalAngle / (count - 1);
  const TOL = defaultTol(s);
  const created: string[] = [];
  for (let k = 1; k < count; k++) {
    const a = stepA * k;
    const c = Math.cos(a), sn = Math.sin(a);
    const r = copyEntities(
      s,
      ids,
      (p) => [center[0] + (p[0] - center[0]) * c - (p[1] - center[1]) * sn, center[1] + (p[0] - center[0]) * sn + (p[1] - center[1]) * c],
      {
        keepPoint: (_pid, p) => Math.hypot(p[0] - center[0], p[1] - center[1]) <= TOL,
        copyConstraint: (cc) => ORIENTATION_FREE.has(cc.type) || (!!opts.copyDimensions && isDimension(cc) && cc.type !== "horizontalDistance" && cc.type !== "verticalDistance"),
      },
    );
    created.push(...r.created);
  }
  return commit(sketch, s, created, []);
}

/** Normalised arc parameter of a point on an arc entity (helper for tools). */
export function arcEntityParam(s: Sketch, arcId: string, p: Vec2): number | null {
  const e = getEntity(s, arcId);
  if (!e || e.type !== "arc") return null;
  const g = arcInfo(s, e);
  return arcParamOf({ center: g.center, radius: g.radius, a0: g.startAngle, sweep: g.sweep }, p);
}

/** Angle helper re-exported for tools. */
export const normalizeAngle = normAngle;
