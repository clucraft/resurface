import { describe, expect, it } from "vitest";
import type { Vec2 } from "../core/types";
import {
  addArc,
  addCircle,
  addConstraint,
  addLine,
  addPoint,
  addRectangle,
  arcInfo,
  createSketch,
  getEntity,
  getPoint,
  measure,
  type Sketch,
} from "./model";
import {
  circleCircleIntersection,
  circularPattern,
  entityIntersections,
  extend,
  filletCorner,
  lineCircleIntersection,
  lineLineIntersection,
  linearPattern,
  mirrorEntities,
  offsetEntities,
  segmentArcIntersection,
  trimAt,
} from "./ops";
import { solveSketch } from "./solver";
import { extractProfiles, loopSignedArea } from "./profiles";

const P = (s: Sketch, id: string): Vec2 => {
  const p = getPoint(s, id)!;
  return [p.x, p.y];
};
const near = (a: Vec2, b: Vec2, eps = 1e-9) => Math.hypot(a[0] - b[0], a[1] - b[1]) <= eps;

describe("intersection utilities", () => {
  it("line-line, line-circle, circle-circle, segment-arc", () => {
    const ll = lineLineIntersection([0, 0], [2, 2], [0, 2], [2, 0])!;
    expect(ll.point[0]).toBeCloseTo(1, 12);
    expect(ll.t).toBeCloseTo(0.5, 12);
    expect(lineLineIntersection([0, 0], [1, 0], [0, 1], [1, 1])).toBeNull();
    const lc = lineCircleIntersection([-5, 0], [5, 0], [0, 0], 2);
    expect(lc.map((h) => h.point[0]).sort()).toEqual([-2, 2]);
    expect(lineCircleIntersection([-5, 3], [5, 3], [0, 0], 2)).toEqual([]);
    expect(lineCircleIntersection([-5, 2], [5, 2], [0, 0], 2).length).toBe(1);
    const cc = circleCircleIntersection([0, 0], 5, [8, 0], 5);
    expect(cc.length).toBe(2);
    expect(cc[0][0]).toBeCloseTo(4, 12);
    expect(Math.abs(cc[0][1])).toBeCloseTo(3, 12);
    expect(circleCircleIntersection([0, 0], 1, [5, 0], 1)).toEqual([]);
    expect(circleCircleIntersection([0, 0], 1, [0, 0], 1)).toEqual([]);
    // quarter arc from (1,0) to (0,1): a horizontal segment at y=0.5 hits it once
    const arc = { type: "arc" as const, start: [1, 0] as Vec2, mid: [Math.SQRT1_2, Math.SQRT1_2] as Vec2, end: [0, 1] as Vec2 };
    const h = segmentArcIntersection([-2, 0.5], [2, 0.5], arc);
    expect(h.length).toBe(1);
    expect(h[0].point[0]).toBeCloseTo(Math.sqrt(0.75), 12);
    // segment that only hits the full circle outside the arc
    expect(segmentArcIntersection([-2, -0.5], [2, -0.5], arc)).toEqual([]);
  });

  it("entity intersections in a sketch", () => {
    const s = createSketch();
    const l = addLine(s, [-10, 0], [10, 0]);
    const c = addCircle(s, [0, 0], 4);
    const hits = entityIntersections(s, l, c);
    expect(hits.length).toBe(2);
  });
});

describe("trim", () => {
  it("splits a line crossed by two others and keeps the outer parts", () => {
    const s = createSketch();
    const l = addLine(s, [0, 0], [10, 0]);
    const v1 = addLine(s, [3, -2], [3, 2]);
    const v2 = addLine(s, [7, -2], [7, 2]);
    addConstraint(s, { type: "horizontal", refs: [l] });
    const r = trimAt(s, l, [5, 0.1]);
    expect(r.ok).toBe(true);
    const lines = r.sketch.entities.filter((e) => e.type === "line" && e.id !== v1 && e.id !== v2) as Extract<(typeof r.sketch.entities)[number], { type: "line" }>[];
    expect(lines.length).toBe(2);
    const spans = lines.map((e) => [P(r.sketch, e.p1)[0], P(r.sketch, e.p2)[0]].sort((a, b) => a - b));
    spans.sort((a, b) => a[0] - b[0]);
    expect(spans[0][0]).toBeCloseTo(0, 9);
    expect(spans[0][1]).toBeCloseTo(3, 9);
    expect(spans[1][0]).toBeCloseTo(7, 9);
    expect(spans[1][1]).toBeCloseTo(10, 9);
    // the new piece is horizontal and collinear, ends pinned on the cutters
    const cs = r.sketch.constraints;
    expect(cs.filter((c) => c.type === "horizontal").length).toBe(2);
    expect(cs.some((c) => c.type === "collinear")).toBe(true);
    expect(cs.filter((c) => c.type === "pointOnEntity").length).toBe(2);
    expect(solveSketch(r.sketch).status).toBe("ok");
    // updated in place
    expect(r.sketch).toBe(s);
    expect(s.entities.length).toBe(4);
  });

  it("leaves the sketch untouched when an operation fails", () => {
    const s = createSketch();
    const rect = addRectangle(s, [0, 0], [20, 10]);
    const before = JSON.stringify(s);
    const r = filletCorner(s, rect.points[2], 50);
    expect(r.ok).toBe(false);
    expect(r.message).toMatch(/too large/);
    expect(JSON.stringify(s)).toBe(before);
    expect(extend(s, rect.lines[0], [10, 0]).ok).toBe(false);
    expect(JSON.stringify(s)).toBe(before);
  });

  it("shortens a line with one cut and deletes an isolated one", () => {
    const s = createSketch();
    const l = addLine(s, [0, 0], [10, 0]);
    addLine(s, [6, -1], [6, 1]);
    addConstraint(s, { type: "length", refs: [l], value: 10 });
    const r = trimAt(s, l, [9, 0]);
    const e = getEntity(r.sketch, l) as { p1: string; p2: string };
    expect(P(r.sketch, e.p2)[0]).toBeCloseTo(6, 9);
    // length dimension removed (no longer valid), old end point removed
    expect(r.sketch.constraints.some((c) => c.type === "length")).toBe(false);
    expect(r.sketch.points.length).toBe(4);
    const lone = addLine(s, [20, 20], [30, 20]);
    const r2 = trimAt(s, lone, [25, 20]);
    expect(getEntity(r2.sketch, lone)).toBeUndefined();
    expect(r2.removed).toContain(lone);
  });

  it("trims a circle into an arc and an arc into two arcs", () => {
    const s = createSketch();
    const c = addCircle(s, [0, 0], 5);
    addLine(s, [-10, 0], [10, 0]);
    const r = trimAt(s, c, [0, 5]); // remove the upper half
    const a = getEntity(r.sketch, c)!;
    expect(a.type).toBe("arc");
    const g = arcInfo(r.sketch, a as never);
    expect(g.sweep).toBeCloseTo(Math.PI, 9);
    // remaining arc is the lower half: its midpoint has y < 0
    expect(Math.sin(g.startAngle + g.sweep / 2)).toBeLessThan(0);

    const s2 = createSketch();
    const arc = addArc(s2, [0, 0], [5, 0], [-5, 0]); // upper half CCW
    addLine(s2, [-2, -1], [-2, 10]);
    addLine(s2, [2, -1], [2, 10]);
    const r2 = trimAt(s2, arc, [0, 5]);
    const arcs = r2.sketch.entities.filter((e) => e.type === "arc");
    expect(arcs.length).toBe(2);
    expect(r2.sketch.constraints.some((k) => k.type === "equal")).toBe(true);
    expect(solveSketch(r2.sketch).status).toBe("ok");
  });

  it("refuses splines", () => {
    const s = createSketch();
    const sp = { id: "s1", type: "spline" as const, points: [addPoint(s, [0, 0]), addPoint(s, [1, 1]), addPoint(s, [2, 0])] };
    s.entities.push(sp);
    expect(trimAt(s, "s1", [1, 1]).ok).toBe(false);
  });
});

describe("extend", () => {
  it("extends a line to the next entity", () => {
    const s = createSketch();
    const l = addLine(s, [0, 0], [4, 0]);
    const wall = addLine(s, [10, -5], [10, 5]);
    const r = extend(s, l, [3.9, 0]);
    expect(r.ok).toBe(true);
    const e = getEntity(r.sketch, l) as { p2: string };
    expect(P(r.sketch, e.p2)[0]).toBeCloseTo(10, 9);
    expect(r.sketch.constraints.some((c) => c.type === "pointOnEntity" && c.refs[1] === wall)).toBe(true);
    // extending the other end: nothing there
    expect(extend(s, l, [0.1, 0]).ok).toBe(false);
  });

  it("extends an arc along its circle", () => {
    const s = createSketch();
    const a = addArc(s, [0, 0], [5, 0], [0, 5]);
    addLine(s, [-10, 0], [10, 0]);
    const r = extend(s, a, [0, 5]);
    expect(r.ok).toBe(true);
    const g = arcInfo(r.sketch, getEntity(r.sketch, a) as never);
    expect(g.sweep).toBeCloseTo(Math.PI, 9);
  });
});

describe("fillet", () => {
  it("rounds a rectangle corner with a tangent arc", () => {
    const s = createSketch();
    const rect = addRectangle(s, [0, 0], [20, 10]);
    const r = filletCorner(s, rect.points[2], 3);
    expect(r.ok).toBe(true);
    const q = r.sketch;
    expect(getPoint(q, rect.points[2])).toBeUndefined();
    const arc = q.entities.find((e) => e.type === "arc")!;
    const g = arcInfo(q, arc as never);
    expect(g.radius).toBeCloseTo(3, 12);
    expect(g.sweep).toBeCloseTo(Math.PI / 2, 12);
    expect(g.center[0]).toBeCloseTo(17, 12);
    expect(g.center[1]).toBeCloseTo(7, 12);
    expect(q.constraints.filter((c) => c.type === "tangent").length).toBe(2);
    expect(q.constraints.some((c) => c.type === "radius")).toBe(true);
    const res = solveSketch(q);
    expect(res.status).toBe("ok");
    expect(res.residual).toBeLessThan(1e-9);
    // profile is still a single closed loop with the arc
    const pr = extractProfiles(q);
    expect(pr.profile.loops.length).toBe(1);
    expect(pr.profile.loops[0].length).toBe(5);
    expect(loopSignedArea(pr.profile.loops[0])).toBeCloseTo(200 - (9 - (Math.PI * 9) / 4), 9);
  });

  it("rejects bad input", () => {
    const s = createSketch();
    const rect = addRectangle(s, [0, 0], [20, 10]);
    expect(filletCorner(s, rect.points[2], 50).ok).toBe(false);
    expect(filletCorner(s, rect.points[2], -1).ok).toBe(false);
    const lone = addPoint(s, [100, 100]);
    expect(filletCorner(s, lone, 1).ok).toBe(false);
  });
});

describe("offset", () => {
  it("offsets a closed rectangle outward with joined corners", () => {
    const s = createSketch();
    const rect = addRectangle(s, [0, 0], [20, 10]);
    const r = offsetEntities(s, rect.lines, 2);
    expect(r.ok).toBe(true);
    const pr = extractProfiles(r.sketch);
    const areas = pr.profile.loops.map(loopSignedArea).sort((a, b) => a - b);
    expect(areas[0]).toBeCloseTo(200, 9);
    expect(areas[1]).toBeCloseTo(24 * 14, 9);
    // copied horizontal/vertical constraints
    expect(r.sketch.constraints.filter((c) => c.type === "horizontal").length).toBe(4);
    const inward = offsetEntities(s, rect.lines, -2);
    const a2 = extractProfiles(inward.sketch).profile.loops.map(loopSignedArea).sort((a, b) => a - b);
    expect(a2[0]).toBeCloseTo(16 * 6, 9);
  });

  it("offsets a rounded chain keeping tangency and concentric arcs", () => {
    const s = createSketch();
    const rect = addRectangle(s, [0, 0], [20, 10]);
    const f = filletCorner(s, rect.points[2], 3).sketch;
    const ids = f.entities.map((e) => e.id);
    const r = offsetEntities(f, ids, 1);
    expect(r.ok).toBe(true);
    const arcs = r.sketch.entities.filter((e) => e.type === "arc");
    expect(arcs.length).toBe(2);
    const radii = arcs.map((a) => arcInfo(r.sketch, a as never).radius).sort();
    expect(radii[0]).toBeCloseTo(3, 9);
    expect(radii[1]).toBeCloseTo(4, 9);
    expect(solveSketch(r.sketch).status).toBe("ok");
    // circle offset
    const s2 = createSketch();
    const c = addCircle(s2, [0, 0], 5);
    const r2 = offsetEntities(s2, [c], 2);
    expect(r2.sketch.entities.filter((e) => e.type === "circle").map((e) => (e as { radius: number }).radius)).toEqual([5, 7]);
    expect(offsetEntities(s2, [c], -6).ok).toBe(false);
  });

  it("offsets an open line to its left", () => {
    const s = createSketch();
    const l = addLine(s, [0, 0], [10, 0]);
    const r = offsetEntities(s, [l], 3);
    const nl = r.sketch.entities.find((e) => e.id !== l) as { p1: string; p2: string };
    expect(P(r.sketch, nl.p1)).toEqual([0, 3]);
    expect(P(r.sketch, nl.p2)).toEqual([10, 3]);
  });
});

describe("mirror and patterns", () => {
  it("mirrors with symmetric constraints and shared on-axis points", () => {
    const s = createSketch();
    const axis = addLine(s, [0, -10], [0, 10], { construction: true });
    const a = addPoint(s, [0, 0]);
    const l = addLine(s, a, [5, 2]);
    const arc = addArc(s, [5, 5], [5, 2], [8, 5]);
    const c = addCircle(s, [3, 8], 1.5);
    const r = mirrorEntities(s, [l, arc, c], axis);
    expect(r.ok).toBe(true);
    const q = r.sketch;
    const lines = q.entities.filter((e) => e.type === "line" && !e.construction) as { p1: string; p2: string }[];
    expect(lines.length).toBe(2);
    // the on-axis point is shared
    expect(lines.filter((x) => x.p1 === a).length).toBe(2);
    const arcs = q.entities.filter((e) => e.type === "arc");
    const g = arcInfo(q, arcs[1] as never);
    expect(g.center[0]).toBeCloseTo(-5, 12);
    expect(g.sweep).toBeCloseTo(Math.PI / 2, 12);
    expect(q.constraints.filter((k) => k.type === "symmetric").length).toBe(5); // line end, arc center/end, circle center (arc start = line end shared)
    expect(q.constraints.filter((k) => k.type === "equal").length).toBe(1);
    const res = solveSketch(q);
    expect(res.status).toBe("ok");
    expect(res.residual).toBeLessThan(1e-9);
  });

  it("linear and circular patterns", () => {
    const s = createSketch();
    const rect = addRectangle(s, [0, 0], [2, 1]);
    const r = linearPattern(s, rect.lines, [5, 0], 4);
    expect(r.sketch.entities.length).toBe(16);
    expect(r.sketch.constraints.filter((c) => c.type === "horizontal").length).toBe(8);
    expect(extractProfiles(r.sketch).profile.loops.length).toBe(4);

    const s2 = createSketch();
    const hub = addPoint(s2, [0, 0]);
    const spoke = addLine(s2, hub, [10, 0]);
    const hole = addCircle(s2, [10, 0], 1);
    addConstraint(s2, { type: "radius", refs: [hole], value: 1 });
    const r2 = circularPattern(s2, [spoke, hole], [0, 0], 6);
    const lines = r2.sketch.entities.filter((e) => e.type === "line") as { p1: string; p2: string }[];
    expect(lines.length).toBe(6);
    expect(lines.every((l) => l.p1 === hub)).toBe(true);
    const ends = lines.map((l) => P(r2.sketch, l.p2));
    expect(ends.some((p) => near(p, [10 * Math.cos(Math.PI / 3), 10 * Math.sin(Math.PI / 3)], 1e-9))).toBe(true);
    // partial pattern: 3 instances over 90 degrees
    const r3 = circularPattern(s2, [spoke], [0, 0], 3, Math.PI / 2);
    const ends3 = (r3.sketch.entities.filter((e) => e.type === "line") as { p2: string }[]).map((l) => P(r3.sketch, l.p2));
    expect(ends3.some((p) => near(p, [0, 10], 1e-9))).toBe(true);
    expect(measure(r2.sketch, { type: "radius", refs: [hole] })).toBe(1);
  });
});
