/**
 * 2D geometry kernel for the sketcher: arc helpers, segment evaluation, closest points,
 * intersections (line/circle/arc/bspline) and polygon utilities. Pure functions on Vec2 tuples.
 */
import type { Segment2D, Vec2 } from "../core/types";
import { circleFrom3Points2 } from "../core/vec";
import {
  bsplineDomain,
  closestPointBSpline2,
  derivBSpline2,
  evalBSpline2,
  reverseBSpline2,
  sampleBSpline2,
  splitBSpline2,
} from "./bspline";

export const TAU = Math.PI * 2;

/** Angle normalised to [0, 2PI). */
export function normAngle(a: number): number {
  a %= TAU;
  if (a < 0) a += TAU;
  return a >= TAU ? 0 : a;
}

/** Angle normalised to (-PI, PI]. */
export function wrapPi(a: number): number {
  a = normAngle(a);
  return a > Math.PI ? a - TAU : a;
}

export const angleOf = (v: Vec2): number => Math.atan2(v[1], v[0]);
export const polar = (c: Vec2, r: number, a: number): Vec2 => [c[0] + r * Math.cos(a), c[1] + r * Math.sin(a)];

/** Circular arc geometry: center, radius, start angle and signed sweep (> 0 = CCW). */
export interface ArcGeom {
  center: Vec2;
  radius: number;
  a0: number;
  sweep: number;
}

/** Arc through three points start -> mid -> end, or null if (nearly) collinear. */
export function arcGeomFrom3(start: Vec2, mid: Vec2, end: Vec2): ArcGeom | null {
  const c = circleFrom3Points2(start, mid, end);
  if (!c || !isFinite(c.radius)) return null;
  const a0 = angleOf([start[0] - c.center[0], start[1] - c.center[1]]);
  const ae = angleOf([end[0] - c.center[0], end[1] - c.center[1]]);
  const am = angleOf([mid[0] - c.center[0], mid[1] - c.center[1]]);
  const ccwEnd = normAngle(ae - a0);
  const ccwMid = normAngle(am - a0);
  let sweep: number;
  if (ccwMid <= ccwEnd) sweep = ccwEnd === 0 ? TAU : ccwEnd;
  else sweep = -(normAngle(a0 - ae) || TAU);
  return { center: c.center, radius: c.radius, a0, sweep };
}

export function arcPoint(g: ArcGeom, t: number): Vec2 {
  return polar(g.center, g.radius, g.a0 + g.sweep * t);
}

/** Unit tangent of an arc at fraction t in its direction of travel. */
export function arcTangent(g: ArcGeom, t: number): Vec2 {
  const a = g.a0 + g.sweep * t;
  const s = g.sweep >= 0 ? 1 : -1;
  return [-Math.sin(a) * s, Math.cos(a) * s];
}

/** Three-point arc segment from arc geometry. */
export function arcSegment(g: ArcGeom): Segment2D {
  return { type: "arc", start: arcPoint(g, 0), mid: arcPoint(g, 0.5), end: arcPoint(g, 1) };
}

/**
 * Fraction t of point p along the arc (by angle). Points outside the swept range map to t < 0 or t > 1,
 * whichever end is angularly closer.
 */
export function arcParamOf(g: ArcGeom, p: Vec2): number {
  const a = angleOf([p[0] - g.center[0], p[1] - g.center[1]]);
  const sw = Math.abs(g.sweep);
  if (sw < 1e-300) return 0;
  const d = g.sweep >= 0 ? normAngle(a - g.a0) : normAngle(g.a0 - a);
  if (d <= sw) return d / sw;
  // outside: before start (negative) or after end
  const after = d - sw;
  const before = TAU - d;
  return after <= before ? 1 + after / sw : -before / sw;
}

// ---------------------------------------------------------------------------
// Segment evaluation
// ---------------------------------------------------------------------------

export function segmentArcGeom(s: Extract<Segment2D, { type: "arc" }>): ArcGeom | null {
  return arcGeomFrom3(s.start, s.mid, s.end);
}

export function segmentStart(s: Segment2D): Vec2 {
  switch (s.type) {
    case "line":
      return s.a;
    case "arc":
      return s.start;
    case "circle":
      return [s.center[0] + s.radius, s.center[1]];
    case "bspline":
      return evalBSpline2(s.curve, bsplineDomain(s.curve)[0]);
  }
}

export function segmentEnd(s: Segment2D): Vec2 {
  switch (s.type) {
    case "line":
      return s.b;
    case "arc":
      return s.end;
    case "circle":
      return [s.center[0] + s.radius, s.center[1]];
    case "bspline":
      return evalBSpline2(s.curve, bsplineDomain(s.curve)[1]);
  }
}

/** Point at normalised parameter t in [0,1] (circle: angle = 2PI t from +x; bspline: domain-normalised). */
export function segmentPointAt(s: Segment2D, t: number): Vec2 {
  switch (s.type) {
    case "line":
      return [s.a[0] + (s.b[0] - s.a[0]) * t, s.a[1] + (s.b[1] - s.a[1]) * t];
    case "arc": {
      const g = segmentArcGeom(s);
      if (!g) return [s.start[0] + (s.end[0] - s.start[0]) * t, s.start[1] + (s.end[1] - s.start[1]) * t];
      return arcPoint(g, t);
    }
    case "circle":
      return polar(s.center, s.radius, TAU * t);
    case "bspline": {
      const [u0, u1] = bsplineDomain(s.curve);
      return evalBSpline2(s.curve, u0 + (u1 - u0) * t);
    }
  }
}

/** Unit tangent (direction of travel) at normalised parameter t. */
export function segmentTangentAt(s: Segment2D, t: number): Vec2 {
  let d: Vec2;
  switch (s.type) {
    case "line":
      d = [s.b[0] - s.a[0], s.b[1] - s.a[1]];
      break;
    case "arc": {
      const g = segmentArcGeom(s);
      if (!g) d = [s.end[0] - s.start[0], s.end[1] - s.start[1]];
      else return arcTangent(g, t);
      break;
    }
    case "circle": {
      const a = TAU * t;
      return [-Math.sin(a), Math.cos(a)];
    }
    case "bspline": {
      const [u0, u1] = bsplineDomain(s.curve);
      d = derivBSpline2(s.curve, u0 + (u1 - u0) * t);
      break;
    }
  }
  const l = Math.hypot(d[0], d[1]);
  return l > 1e-300 ? [d[0] / l, d[1] / l] : [1, 0];
}

export function segmentLength(s: Segment2D): number {
  switch (s.type) {
    case "line":
      return Math.hypot(s.b[0] - s.a[0], s.b[1] - s.a[1]);
    case "arc": {
      const g = segmentArcGeom(s);
      return g ? g.radius * Math.abs(g.sweep) : Math.hypot(s.end[0] - s.start[0], s.end[1] - s.start[1]);
    }
    case "circle":
      return TAU * s.radius;
    case "bspline": {
      const pts = sampleBSpline2(s.curve, 128);
      let L = 0;
      for (let i = 1; i < pts.length; i++) L += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
      return L;
    }
  }
}

export function reverseSegment(s: Segment2D): Segment2D {
  switch (s.type) {
    case "line":
      return { type: "line", a: s.b, b: s.a };
    case "arc":
      return { type: "arc", start: s.end, mid: s.mid, end: s.start };
    case "circle":
      return s;
    case "bspline":
      return { type: "bspline", curve: reverseBSpline2(s.curve) };
  }
}

/** Is the segment closed by itself (circle or bspline whose ends coincide)? */
export function segmentIsClosed(s: Segment2D, tol = 1e-9): boolean {
  if (s.type === "circle") return true;
  if (s.type === "bspline") {
    const a = segmentStart(s), b = segmentEnd(s);
    return Math.hypot(a[0] - b[0], a[1] - b[1]) <= tol;
  }
  return false;
}

/**
 * Sample a segment into a polyline. `opts` is either a fixed number of intervals or a chordal
 * tolerance (default: 1e-3 relative to size, at most 256 intervals). Endpoints are included exactly.
 */
export function sampleSegment(s: Segment2D, opts?: number | { chordTol?: number; maxSegments?: number }): Vec2[] {
  if (s.type === "line") return [s.a, s.b];
  let n: number;
  if (typeof opts === "number") n = Math.max(1, Math.floor(opts));
  else {
    const maxN = opts?.maxSegments ?? 256;
    let sweep: number, r: number;
    if (s.type === "circle") {
      sweep = TAU;
      r = s.radius;
    } else if (s.type === "arc") {
      const g = segmentArcGeom(s);
      if (!g) return [s.start, s.end];
      sweep = Math.abs(g.sweep);
      r = g.radius;
    } else {
      sweep = TAU;
      r = segmentLength(s) / TAU;
    }
    const tol = opts?.chordTol ?? Math.max(1e-6, r * 1e-3);
    const da = r > tol ? 2 * Math.acos(Math.max(-1, 1 - tol / r)) : Math.PI / 2;
    n = Math.max(s.type === "bspline" ? 16 : 4, Math.min(maxN, Math.ceil(sweep / Math.max(da, 1e-6))));
    if (s.type === "bspline") n = Math.max(n, Math.min(maxN, s.curve.controlPoints.length * 8));
  }
  const out: Vec2[] = [];
  for (let i = 0; i <= n; i++) out.push(segmentPointAt(s, i / n));
  if (s.type === "arc") {
    out[0] = s.start;
    out[n] = s.end;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Closest points
// ---------------------------------------------------------------------------

export interface ClosestOnSegment {
  point: Vec2;
  /** normalised parameter (see segmentPointAt) */
  t: number;
  distance: number;
}

export function closestPointOnLineSegment(a: Vec2, b: Vec2, p: Vec2, clamp = true): ClosestOnSegment {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const L2 = dx * dx + dy * dy;
  let t = L2 > 1e-300 ? ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2 : 0;
  if (clamp) t = Math.max(0, Math.min(1, t));
  const q: Vec2 = [a[0] + dx * t, a[1] + dy * t];
  return { point: q, t, distance: Math.hypot(p[0] - q[0], p[1] - q[1]) };
}

export function closestPointOnSegment(s: Segment2D, p: Vec2): ClosestOnSegment {
  switch (s.type) {
    case "line":
      return closestPointOnLineSegment(s.a, s.b, p);
    case "circle": {
      const dx = p[0] - s.center[0], dy = p[1] - s.center[1];
      const a = Math.hypot(dx, dy) > 1e-300 ? Math.atan2(dy, dx) : 0;
      const q = polar(s.center, s.radius, a);
      return { point: q, t: normAngle(a) / TAU, distance: Math.hypot(p[0] - q[0], p[1] - q[1]) };
    }
    case "arc": {
      const g = segmentArcGeom(s);
      if (!g) return closestPointOnLineSegment(s.start, s.end, p);
      let t = arcParamOf(g, p);
      if (t < 0 || t > 1) {
        const ds = Math.hypot(p[0] - s.start[0], p[1] - s.start[1]);
        const de = Math.hypot(p[0] - s.end[0], p[1] - s.end[1]);
        return ds <= de ? { point: s.start, t: 0, distance: ds } : { point: s.end, t: 1, distance: de };
      }
      const q = arcPoint(g, t);
      return { point: q, t, distance: Math.hypot(p[0] - q[0], p[1] - q[1]) };
    }
    case "bspline": {
      const r = closestPointBSpline2(s.curve, p);
      const [u0, u1] = bsplineDomain(s.curve);
      return { point: r.point, t: (r.u - u0) / (u1 - u0 || 1), distance: r.distance };
    }
  }
}

// ---------------------------------------------------------------------------
// Primitive intersections (infinite lines / full circles)
// ---------------------------------------------------------------------------

/**
 * Intersection of the infinite lines a1a2 and b1b2. t and u are the parameters along a1->a2 and b1->b2
 * (point = a1 + t (a2 - a1)). Null for (nearly) parallel lines.
 */
export function lineLineIntersection(a1: Vec2, a2: Vec2, b1: Vec2, b2: Vec2): { point: Vec2; t: number; u: number } | null {
  const rx = a2[0] - a1[0], ry = a2[1] - a1[1];
  const sx = b2[0] - b1[0], sy = b2[1] - b1[1];
  const den = rx * sy - ry * sx;
  const lr = Math.hypot(rx, ry), ls = Math.hypot(sx, sy);
  if (lr < 1e-300 || ls < 1e-300 || Math.abs(den) <= 1e-12 * lr * ls) return null;
  const qx = b1[0] - a1[0], qy = b1[1] - a1[1];
  const t = (qx * sy - qy * sx) / den;
  const u = (qx * ry - qy * rx) / den;
  return { point: [a1[0] + rx * t, a1[1] + ry * t], t, u };
}

/** Intersections of the infinite line a->b with a circle; t is the parameter along a->b. */
export function lineCircleIntersection(a: Vec2, b: Vec2, c: Vec2, r: number, tangentTol = 0): { point: Vec2; t: number }[] {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const L2 = dx * dx + dy * dy;
  if (L2 < 1e-300) return [];
  const t0 = ((c[0] - a[0]) * dx + (c[1] - a[1]) * dy) / L2;
  const fx = a[0] + dx * t0 - c[0], fy = a[1] + dy * t0 - c[1];
  const h2 = fx * fx + fy * fy;
  const L = Math.sqrt(L2);
  if (h2 > r * r) {
    const h = Math.sqrt(h2);
    if (h - r <= tangentTol) return [{ point: [a[0] + dx * t0, a[1] + dy * t0], t: t0 }];
    return [];
  }
  const half = Math.sqrt(Math.max(0, r * r - h2)) / L;
  if (half * L <= 1e-12 * Math.max(1, r)) return [{ point: [a[0] + dx * t0, a[1] + dy * t0], t: t0 }];
  return [t0 - half, t0 + half].map((t) => ({ point: [a[0] + dx * t, a[1] + dy * t] as Vec2, t }));
}

/** Intersection points of two circles (0, 1 or 2). Coincident circles return []. */
export function circleCircleIntersection(c1: Vec2, r1: number, c2: Vec2, r2: number, tangentTol = 0): Vec2[] {
  const dx = c2[0] - c1[0], dy = c2[1] - c1[1];
  const d = Math.hypot(dx, dy);
  const scale = Math.max(1, r1, r2);
  if (d < 1e-12 * scale) return [];
  if (d > r1 + r2 + tangentTol || d < Math.abs(r1 - r2) - tangentTol) return [];
  const a = (r1 * r1 - r2 * r2 + d * d) / (2 * d);
  const h2 = r1 * r1 - a * a;
  const ux = dx / d, uy = dy / d;
  const px = c1[0] + ux * a, py = c1[1] + uy * a;
  if (h2 <= (1e-12 * scale) ** 2) return [[px, py]];
  const h = Math.sqrt(h2);
  return [
    [px - uy * h, py + ux * h],
    [px + uy * h, py - ux * h],
  ];
}

// ---------------------------------------------------------------------------
// Segment–segment intersections
// ---------------------------------------------------------------------------

export interface SegmentHit {
  point: Vec2;
  /** normalised parameter on the first segment */
  t1: number;
  /** normalised parameter on the second segment */
  t2: number;
}

type Prim =
  | { kind: "line"; a: Vec2; b: Vec2 }
  | { kind: "circ"; c: Vec2; r: number; arc: ArcGeom | null };

function toPrim(s: Segment2D): Prim | null {
  switch (s.type) {
    case "line":
      return { kind: "line", a: s.a, b: s.b };
    case "circle":
      return { kind: "circ", c: s.center, r: s.radius, arc: null };
    case "arc": {
      const g = segmentArcGeom(s);
      if (!g) return { kind: "line", a: s.start, b: s.end };
      return { kind: "circ", c: g.center, r: g.radius, arc: g };
    }
    default:
      return null;
  }
}

/** Parameter of p on a primitive and whether it lies within the bounded piece (with tolerance in length units). */
function primParam(pr: Prim, p: Vec2, tol: number): { t: number; inside: boolean } {
  if (pr.kind === "line") {
    const dx = pr.b[0] - pr.a[0], dy = pr.b[1] - pr.a[1];
    const L2 = dx * dx + dy * dy;
    const L = Math.sqrt(L2);
    const t = L2 > 1e-300 ? ((p[0] - pr.a[0]) * dx + (p[1] - pr.a[1]) * dy) / L2 : 0;
    const e = L > 0 ? tol / L : 0;
    return { t: Math.max(0, Math.min(1, t)), inside: t >= -e && t <= 1 + e };
  }
  if (!pr.arc) {
    const a = Math.atan2(p[1] - pr.c[1], p[0] - pr.c[0]);
    return { t: normAngle(a) / TAU, inside: true };
  }
  const t = arcParamOf(pr.arc, p);
  const e = pr.r * Math.abs(pr.arc.sweep) > 0 ? tol / (pr.r * Math.abs(pr.arc.sweep)) : 0;
  return { t: Math.max(0, Math.min(1, t)), inside: t >= -e && t <= 1 + e };
}

function primIntersect(p1: Prim, p2: Prim, tol: number): Vec2[] {
  if (p1.kind === "line" && p2.kind === "line") {
    const r = lineLineIntersection(p1.a, p1.b, p2.a, p2.b);
    if (r) return [r.point];
    // collinear overlap: report shared endpoints
    const out: Vec2[] = [];
    for (const q of [p2.a, p2.b]) if (closestPointOnLineSegment(p1.a, p1.b, q).distance <= tol) out.push(q);
    for (const q of [p1.a, p1.b]) if (closestPointOnLineSegment(p2.a, p2.b, q).distance <= tol) out.push(q);
    return out;
  }
  if (p1.kind === "line" && p2.kind === "circ") return lineCircleIntersection(p1.a, p1.b, p2.c, p2.r, tol).map((h) => h.point);
  if (p1.kind === "circ" && p2.kind === "line") return lineCircleIntersection(p2.a, p2.b, p1.c, p1.r, tol).map((h) => h.point);
  const c1 = p1 as Extract<Prim, { kind: "circ" }>, c2 = p2 as Extract<Prim, { kind: "circ" }>;
  return circleCircleIntersection(c1.c, c1.r, c2.c, c2.r, tol);
}

function dedupeHits(hits: SegmentHit[], tol: number): SegmentHit[] {
  const out: SegmentHit[] = [];
  for (const h of hits) if (!out.some((o) => Math.hypot(o.point[0] - h.point[0], o.point[1] - h.point[1]) <= tol)) out.push(h);
  return out;
}

/**
 * Intersections between two bounded segments (line, arc, circle, bspline). Tangential touches within
 * `tol` are reported. B-splines are handled by polyline sampling followed by Newton refinement.
 */
export function segmentIntersections(s1: Segment2D, s2: Segment2D, tol = 1e-9): SegmentHit[] {
  const p1 = toPrim(s1), p2 = toPrim(s2);
  if (p1 && p2) {
    const hits: SegmentHit[] = [];
    for (const q of primIntersect(p1, p2, tol)) {
      const a = primParam(p1, q, tol), b = primParam(p2, q, tol);
      if (a.inside && b.inside) hits.push({ point: q, t1: a.t, t2: b.t });
    }
    return dedupeHits(hits, Math.max(tol, 1e-12));
  }
  return dedupeHits(sampledIntersections(s1, s2, tol), Math.max(tol * 10, 1e-9));
}

function sampledIntersections(s1: Segment2D, s2: Segment2D, tol: number): SegmentHit[] {
  const n1 = s1.type === "line" ? 1 : 200, n2 = s2.type === "line" ? 1 : 200;
  const A = sampleSegment(s1, n1), B = sampleSegment(s2, n2);
  const hits: SegmentHit[] = [];
  for (let i = 0; i < A.length - 1; i++) {
    const a0 = A[i], a1 = A[i + 1];
    const minAx = Math.min(a0[0], a1[0]) - tol, maxAx = Math.max(a0[0], a1[0]) + tol;
    const minAy = Math.min(a0[1], a1[1]) - tol, maxAy = Math.max(a0[1], a1[1]) + tol;
    for (let j = 0; j < B.length - 1; j++) {
      const b0 = B[j], b1 = B[j + 1];
      if (Math.max(b0[0], b1[0]) < minAx || Math.min(b0[0], b1[0]) > maxAx) continue;
      if (Math.max(b0[1], b1[1]) < minAy || Math.min(b0[1], b1[1]) > maxAy) continue;
      const r = lineLineIntersection(a0, a1, b0, b1);
      if (!r || r.t < -1e-9 || r.t > 1 + 1e-9 || r.u < -1e-9 || r.u > 1 + 1e-9) continue;
      let t1 = (i + r.t) / (A.length - 1);
      let t2 = (j + r.u) / (B.length - 1);
      // Newton refinement on F(t1,t2) = P1(t1) - P2(t2)
      for (let it = 0; it < 12; it++) {
        const q1 = segmentPointAt(s1, t1), q2 = segmentPointAt(s2, t2);
        const fx = q1[0] - q2[0], fy = q1[1] - q2[1];
        if (Math.hypot(fx, fy) < 1e-13) break;
        const h = 1e-7;
        const d1a = segmentPointAt(s1, Math.min(1, t1 + h)), d1b = segmentPointAt(s1, Math.max(0, t1 - h));
        const d2a = segmentPointAt(s2, Math.min(1, t2 + h)), d2b = segmentPointAt(s2, Math.max(0, t2 - h));
        const h1 = Math.min(1, t1 + h) - Math.max(0, t1 - h), h2 = Math.min(1, t2 + h) - Math.max(0, t2 - h);
        const j11 = (d1a[0] - d1b[0]) / h1, j21 = (d1a[1] - d1b[1]) / h1;
        const j12 = -(d2a[0] - d2b[0]) / h2, j22 = -(d2a[1] - d2b[1]) / h2;
        const det = j11 * j22 - j12 * j21;
        if (Math.abs(det) < 1e-300) break;
        const dt1 = (fx * j22 - fy * j12) / det;
        const dt2 = (j11 * fy - j21 * fx) / det;
        t1 = Math.max(0, Math.min(1, t1 - dt1));
        t2 = Math.max(0, Math.min(1, t2 - dt2));
      }
      const q1 = segmentPointAt(s1, t1), q2 = segmentPointAt(s2, t2);
      if (Math.hypot(q1[0] - q2[0], q1[1] - q2[1]) <= Math.max(tol, 1e-6))
        hits.push({ point: [(q1[0] + q2[0]) / 2, (q1[1] + q2[1]) / 2], t1, t2 });
    }
  }
  return hits;
}

/** Split a segment at normalised parameter t into two pieces (circles: returns arcs [t..t+1/2], [t+1/2..t+1]). */
export function splitSegment(s: Segment2D, t: number): [Segment2D, Segment2D] {
  switch (s.type) {
    case "line": {
      const m = segmentPointAt(s, t);
      return [
        { type: "line", a: s.a, b: m },
        { type: "line", a: m, b: s.b },
      ];
    }
    case "arc": {
      const g = segmentArcGeom(s);
      if (!g) {
        const m = segmentPointAt(s, t);
        return [
          { type: "line", a: s.start, b: m },
          { type: "line", a: m, b: s.end },
        ];
      }
      const m = arcPoint(g, t);
      return [
        { type: "arc", start: s.start, mid: arcPoint(g, t / 2), end: m },
        { type: "arc", start: m, mid: arcPoint(g, (1 + t) / 2), end: s.end },
      ];
    }
    case "circle": {
      const a = TAU * t;
      const g1: ArcGeom = { center: s.center, radius: s.radius, a0: a, sweep: Math.PI };
      const g2: ArcGeom = { center: s.center, radius: s.radius, a0: a + Math.PI, sweep: Math.PI };
      return [arcSegment(g1), arcSegment(g2)];
    }
    case "bspline": {
      const [u0, u1] = bsplineDomain(s.curve);
      const [l, r] = splitBSpline2(s.curve, u0 + (u1 - u0) * t);
      return [
        { type: "bspline", curve: l },
        { type: "bspline", curve: r },
      ];
    }
  }
}

/**
 * Sub-segment between normalised parameters t0 < t1 (for circles t may exceed 1 to wrap around; the
 * result is then an arc).
 */
export function subSegment(s: Segment2D, t0: number, t1: number): Segment2D {
  switch (s.type) {
    case "line":
      return { type: "line", a: segmentPointAt(s, t0), b: segmentPointAt(s, t1) };
    case "arc": {
      const g = segmentArcGeom(s);
      if (!g) return { type: "line", a: segmentPointAt(s, t0), b: segmentPointAt(s, t1) };
      return arcSegment({ center: g.center, radius: g.radius, a0: g.a0 + g.sweep * t0, sweep: g.sweep * (t1 - t0) });
    }
    case "circle":
      return arcSegment({ center: s.center, radius: s.radius, a0: TAU * t0, sweep: TAU * (t1 - t0) });
    case "bspline": {
      let c = s.curve;
      const [u0, u1] = bsplineDomain(c);
      const ua = u0 + (u1 - u0) * t0, ub = u0 + (u1 - u0) * t1;
      if (ub < u1 - 1e-12 * (u1 - u0)) c = splitBSpline2(c, ub)[0];
      if (ua > u0 + 1e-12 * (u1 - u0)) c = splitBSpline2(c, ua)[1];
      return { type: "bspline", curve: c };
    }
  }
}

// ---------------------------------------------------------------------------
// Polygons
// ---------------------------------------------------------------------------

export function polygonSignedArea(poly: Vec2[]): number {
  let a = 0;
  for (let i = 0, n = poly.length; i < n; i++) {
    const p = poly[i], q = poly[(i + 1) % n];
    a += p[0] * q[1] - p[1] * q[0];
  }
  return a / 2;
}

/** Even-odd point-in-polygon test (polygon implicitly closed). */
export function pointInPolygon(poly: Vec2[], p: Vec2): boolean {
  let inside = false;
  const x = p[0], y = p[1];
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0], yi = poly[i][1], xj = poly[j][0], yj = poly[j][1];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/** Distance from p to the polygon boundary (closed). */
export function distanceToPolygon(poly: Vec2[], p: Vec2): number {
  let best = Infinity;
  for (let i = 0, n = poly.length; i < n; i++) {
    const d = closestPointOnLineSegment(poly[i], poly[(i + 1) % n], p).distance;
    if (d < best) best = d;
  }
  return best;
}

/** Exact signed area contribution  1/2 * integral(x dy - y dx)  of a segment (bsplines sampled). */
export function segmentAreaTerm(s: Segment2D): number {
  switch (s.type) {
    case "line":
      return (s.a[0] * s.b[1] - s.a[1] * s.b[0]) / 2;
    case "circle":
      return Math.PI * s.radius * s.radius;
    case "arc": {
      const g = segmentArcGeom(s);
      const chord = (s.start[0] * s.end[1] - s.start[1] * s.end[0]) / 2;
      if (!g) return chord;
      const phi = g.sweep;
      return chord + ((g.radius * g.radius) / 2) * (phi - Math.sin(phi));
    }
    case "bspline": {
      const pts = sampleSegment(s, 256);
      let a = 0;
      for (let i = 0; i < pts.length - 1; i++) a += pts[i][0] * pts[i + 1][1] - pts[i][1] * pts[i + 1][0];
      return a / 2;
    }
  }
}

/** Bounding box of a segment (arcs/circles exact, bsplines from the control polygon). */
export function segmentBounds(s: Segment2D): { min: Vec2; max: Vec2 } {
  const min: Vec2 = [Infinity, Infinity], max: Vec2 = [-Infinity, -Infinity];
  const add = (p: Vec2) => {
    if (p[0] < min[0]) min[0] = p[0];
    if (p[1] < min[1]) min[1] = p[1];
    if (p[0] > max[0]) max[0] = p[0];
    if (p[1] > max[1]) max[1] = p[1];
  };
  switch (s.type) {
    case "line":
      add(s.a);
      add(s.b);
      break;
    case "circle":
      add([s.center[0] - s.radius, s.center[1] - s.radius]);
      add([s.center[0] + s.radius, s.center[1] + s.radius]);
      break;
    case "arc": {
      add(s.start);
      add(s.end);
      const g = segmentArcGeom(s);
      if (!g) break;
      for (let k = 0; k < 4; k++) {
        const ang = (k * Math.PI) / 2;
        const t = arcParamOf(g, polar(g.center, g.radius, ang));
        if (t >= 0 && t <= 1) add(polar(g.center, g.radius, ang));
      }
      break;
    }
    case "bspline":
      for (const p of s.curve.controlPoints) add(p);
      break;
  }
  return { min, max };
}
