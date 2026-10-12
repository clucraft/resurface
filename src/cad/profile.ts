/**
 * 2D profile analysis for the CAD kernel (pure, no OCCT):
 *  - segment helpers (start/end/reverse/sampling),
 *  - loop closing/snapping,
 *  - nesting analysis: which loops are outer boundaries, which are holes, islands in holes.
 */
import type { Loop2D, Profile2D, Segment2D, Vec2, Wire2D } from "../core/types";
import { circleFrom3Points2, dist2 } from "../core/vec";
import { curveDomain, evalCurve2, reverseCurve } from "./nurbs";

/** Kernel-internal segment: like Segment2D but a circle may be traversed clockwise. */
export type KSegment =
  | Exclude<Segment2D, { type: "circle" }>
  | { type: "circle"; center: Vec2; radius: number; cw?: boolean };
export type KLoop = KSegment[];

export interface Region {
  /** counter-clockwise outer boundary */
  outer: KLoop;
  /** clockwise holes directly inside `outer` */
  holes: KLoop[];
}

export function segStart(s: KSegment): Vec2 {
  switch (s.type) {
    case "line": return s.a;
    case "arc": return s.start;
    case "circle": return [s.center[0] + s.radius, s.center[1]];
    case "bspline": return s.curve.controlPoints[0];
  }
}

export function segEnd(s: KSegment): Vec2 {
  switch (s.type) {
    case "line": return s.b;
    case "arc": return s.end;
    case "circle": return [s.center[0] + s.radius, s.center[1]];
    case "bspline": return s.curve.controlPoints[s.curve.controlPoints.length - 1];
  }
}

export function reverseSegment(s: KSegment): KSegment {
  switch (s.type) {
    case "line": return { type: "line", a: s.b, b: s.a };
    case "arc": return { type: "arc", start: s.end, mid: s.mid, end: s.start };
    case "circle": return { type: "circle", center: s.center, radius: s.radius, cw: !s.cw };
    case "bspline": return { type: "bspline", curve: reverseCurve(s.curve) };
  }
}

export const reverseLoop = (loop: KLoop): KLoop => loop.slice().reverse().map(reverseSegment);

const TWO_PI = Math.PI * 2;
const normAngle = (a: number) => ((a % TWO_PI) + TWO_PI) % TWO_PI;

/** Arc geometry from a three-point arc: centre, radius, start angle and signed sweep. */
export function arcGeometry(s: { start: Vec2; mid: Vec2; end: Vec2 }): { center: Vec2; radius: number; a0: number; sweep: number } | null {
  const c = circleFrom3Points2(s.start, s.mid, s.end);
  if (!c) return null;
  const a0 = Math.atan2(s.start[1] - c.center[1], s.start[0] - c.center[0]);
  const am = Math.atan2(s.mid[1] - c.center[1], s.mid[0] - c.center[0]);
  const a1 = Math.atan2(s.end[1] - c.center[1], s.end[0] - c.center[0]);
  const d1 = normAngle(a1 - a0);
  const dm = normAngle(am - a0);
  const sweep = dm < d1 ? d1 : -(TWO_PI - d1);
  return { center: c.center, radius: c.radius, a0, sweep };
}

/** Polyline approximation of a segment, including both end points. */
export function sampleSegment(s: KSegment, maxAngleStep = Math.PI / 36, bsplineSamples = 48): Vec2[] {
  switch (s.type) {
    case "line":
      return [s.a, s.b];
    case "arc": {
      const g = arcGeometry(s);
      if (!g) return [s.start, s.end];
      const n = Math.max(2, Math.ceil(Math.abs(g.sweep) / maxAngleStep));
      const out: Vec2[] = [];
      for (let i = 0; i <= n; i++) {
        const a = g.a0 + (g.sweep * i) / n;
        out.push([g.center[0] + g.radius * Math.cos(a), g.center[1] + g.radius * Math.sin(a)]);
      }
      out[0] = s.start;
      out[n] = s.end;
      return out;
    }
    case "circle": {
      const n = Math.max(8, Math.ceil(TWO_PI / maxAngleStep));
      const out: Vec2[] = [];
      for (let i = 0; i <= n; i++) {
        const a = ((s.cw ? -1 : 1) * TWO_PI * i) / n;
        out.push([s.center[0] + s.radius * Math.cos(a), s.center[1] + s.radius * Math.sin(a)]);
      }
      return out;
    }
    case "bspline": {
      const [t0, t1] = curveDomain(s.curve);
      const n = Math.max(bsplineSamples, s.curve.controlPoints.length * 4);
      const out: Vec2[] = [];
      for (let i = 0; i <= n; i++) out.push(evalCurve2(s.curve, t0 + ((t1 - t0) * i) / n));
      return out;
    }
  }
}

/** Closed polygon (no repeated closing vertex) approximating a loop. */
export function polygonize(loop: KLoop): Vec2[] {
  const out: Vec2[] = [];
  for (const s of loop) {
    const pts = sampleSegment(s);
    for (let i = out.length ? 1 : 0; i < pts.length; i++) out.push(pts[i]);
  }
  if (out.length > 1 && dist2(out[0], out[out.length - 1]) < 1e-12) out.pop();
  return out;
}

export function signedArea(poly: readonly Vec2[]): number {
  let a = 0;
  for (let i = 0, n = poly.length; i < n; i++) {
    const p = poly[i], q = poly[(i + 1) % n];
    a += p[0] * q[1] - q[0] * p[1];
  }
  return a / 2;
}

export function pointInPolygon(pt: Vec2, poly: readonly Vec2[]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if (yi > pt[1] !== yj > pt[1] && pt[0] < ((xj - xi) * (pt[1] - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function bboxDiag(points: readonly Vec2[]): number {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const p of points) {
    x0 = Math.min(x0, p[0]); y0 = Math.min(y0, p[1]);
    x1 = Math.max(x1, p[0]); y1 = Math.max(y1, p[1]);
  }
  return points.length ? Math.hypot(x1 - x0, y1 - y0) : 0;
}

/** Default closing tolerance for a set of segments, relative to their size. */
export function defaultTolerance(segments: readonly KSegment[]): number {
  const pts: Vec2[] = [];
  for (const s of segments) pts.push(...sampleSegment(s, Math.PI / 4, 8));
  return Math.max(1e-6, 1e-5 * bboxDiag(pts));
}

function withStart(s: KSegment, p: Vec2): KSegment {
  switch (s.type) {
    case "line": return { ...s, a: p };
    case "arc": return { ...s, start: p };
    case "bspline": {
      const cps = s.curve.controlPoints.slice();
      cps[0] = p;
      return { type: "bspline", curve: { ...s.curve, controlPoints: cps } };
    }
    case "circle": return s;
  }
}

/** Drop zero-length segments and make each segment start exactly where the previous ended. */
function snapChain(segs: readonly KSegment[], closed: boolean, tol: number, what: string): KSegment[] {
  const out: KSegment[] = segs.filter((s) => {
    if (s.type === "circle") return s.radius > tol;
    if (s.type === "line") return dist2(s.a, s.b) > tol;
    if (s.type === "arc") return dist2(s.start, s.end) > tol || dist2(s.start, s.mid) > tol;
    return true;
  });
  if (!out.length) throw new Error(`${what} has no (non-degenerate) segments`);
  if (out.some((s) => s.type === "circle") && out.length > 1)
    throw new Error(`${what}: a full circle must be the only segment of its loop`);
  if (out.length === 1 && out[0].type === "circle") {
    if (!closed) throw new Error(`${what}: a full circle cannot be part of an open wire`);
    return out;
  }
  const n = out.length;
  const last = closed ? n : n - 1;
  for (let i = 0; i < last; i++) {
    const j = (i + 1) % n;
    const e = segEnd(out[i]);
    const s = segStart(out[j]);
    const gap = dist2(e, s);
    if (gap > tol)
      throw new Error(
        `${what} is not ${closed && i === n - 1 ? "closed" : "connected"}: gap of ${gap.toPrecision(3)} between segment ${i} and segment ${j}`,
      );
    if (gap > 0) out[j] = withStart(out[j], e);
  }
  return out;
}

/** Snap a closed loop; throws a descriptive Error when it is open. */
export function snapLoop(loop: Loop2D | KLoop, tol?: number, what = "loop"): KLoop {
  return snapChain(loop as KLoop, true, tol ?? defaultTolerance(loop as KLoop), what);
}

/** Snap an open (or closed) wire. */
export function snapWire(wire: Wire2D, tol?: number): KLoop {
  return snapChain(wire.segments, wire.closed, tol ?? defaultTolerance(wire.segments), "wire");
}

/** Whether a loop closes within tolerance (does not throw). */
export function isLoopClosed(loop: Loop2D, tol?: number): boolean {
  try {
    snapLoop(loop, tol);
    return true;
  } catch {
    return false;
  }
}

/**
 * Classify the loops of a profile into regions (outer boundary + its direct holes), using
 * containment depth: even depth = outer boundary, odd depth = hole of the smallest enclosing
 * loop. Islands inside holes therefore become new regions. Orientation is normalised
 * (outer CCW, holes CW in the frame's 2D coordinates).
 */
export function buildRegions(profile: Profile2D, tol?: number): Region[] {
  if (!profile.loops.length) throw new Error("profile has no loops");
  const loops = profile.loops.map((l, i) => snapLoop(l, tol, `profile loop ${i}`));
  const polys = loops.map(polygonize);
  const areas = polys.map(signedArea);
  loops.forEach((_, i) => {
    if (Math.abs(areas[i]) < 1e-12) throw new Error(`profile loop ${i} encloses no area`);
  });
  const probe = polys.map((p) => {
    const a = p[0], b = p[1 % p.length];
    return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2] as Vec2;
  });
  const n = loops.length;
  const containers: number[][] = loops.map(() => []);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      if (i !== j && Math.abs(areas[j]) > Math.abs(areas[i]) && pointInPolygon(probe[i], polys[j])) containers[i].push(j);
  const depth = containers.map((c) => c.length);
  const regions = new Map<number, Region>();
  const oriented = (i: number, ccw: boolean): KLoop => (areas[i] > 0 === ccw ? loops[i] : reverseLoop(loops[i]));
  for (let i = 0; i < n; i++) if (depth[i] % 2 === 0) regions.set(i, { outer: oriented(i, true), holes: [] });
  for (let i = 0; i < n; i++) {
    if (depth[i] % 2 === 0) continue;
    let parent = -1;
    for (const j of containers[i])
      if (depth[j] === depth[i] - 1 && (parent < 0 || Math.abs(areas[j]) < Math.abs(areas[parent]))) parent = j;
    regions.get(parent)!.holes.push(oriented(i, false));
  }
  return [...regions.values()];
}
