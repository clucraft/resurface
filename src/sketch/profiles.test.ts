import { describe, expect, it } from "vitest";
import type { Loop2D, Vec2 } from "../core/types";
import { segmentEnd, segmentStart } from "./geom2d";
import { addArc, addCircle, addLine, addPoint, addRectangle, addSpline, createSketch } from "./model";
import { extractProfiles, loopContainsPoint, loopSignedArea, regionsFromLoops, sampleLoop, sampleSegment } from "./profiles";

function expectHeadToTail(loop: Loop2D, eps = 1e-9) {
  if (loop.length === 1) return;
  for (let i = 0; i < loop.length; i++) {
    const a = segmentEnd(loop[i]);
    const b = segmentStart(loop[(i + 1) % loop.length]);
    expect(Math.hypot(a[0] - b[0], a[1] - b[1])).toBeLessThan(eps);
  }
}

describe("extractProfiles", () => {
  it("finds a CCW rectangle loop and ignores construction geometry", () => {
    const s = createSketch();
    // clockwise input order to check re-orientation
    const p = [addPoint(s, [0, 0]), addPoint(s, [0, 10]), addPoint(s, [20, 10]), addPoint(s, [20, 0])];
    for (let i = 0; i < 4; i++) addLine(s, p[i], p[(i + 1) % 4]);
    addLine(s, [0, 0], [20, 10], { construction: true });
    const r = extractProfiles(s);
    expect(r.profile.loops.length).toBe(1);
    expect(r.profile.loops[0].length).toBe(4);
    expect(loopSignedArea(r.profile.loops[0])).toBeCloseTo(200, 12);
    expectHeadToTail(r.profile.loops[0]);
    expect(r.openWires).toEqual([]);
    expect(r.crossings).toEqual([]);
    expect(r.loopEntities[0].length).toBe(4);
  });

  it("nests holes and islands into regions", () => {
    const s = createSketch();
    addRectangle(s, [0, 0], [100, 100]);
    addRectangle(s, [10, 10], [60, 60]);
    addCircle(s, [35, 35], 10); // island in the hole
    addCircle(s, [80, 80], 5); // second hole
    const r = extractProfiles(s);
    expect(r.profile.loops.length).toBe(4);
    const regions = regionsFromLoops(r.profile);
    expect(regions.length).toBe(2);
    const big = regions.find((g) => g.holes.length === 2)!;
    expect(loopSignedArea(big.outerLoop)).toBeCloseTo(10000, 9);
    const sq = big.holeLoops.find((l) => l.length === 4)!;
    expect(loopSignedArea(sq)).toBeCloseTo(-2500, 9); // holes are clockwise
    const island = regions.find((g) => g.holes.length === 0)!;
    expect(island.outerLoop[0].type).toBe("circle");
  });

  it("splits at T-junctions into faces and reports branch points", () => {
    const s = createSketch();
    addRectangle(s, [0, 0], [20, 10]);
    addLine(s, [8, 0], [8, 10]); // endpoints lie on the edges (separate point ids)
    const r = extractProfiles(s);
    expect(r.profile.loops.length).toBe(2);
    const areas = r.profile.loops.map(loopSignedArea).sort((a, b) => a - b);
    expect(areas[0]).toBeCloseTo(80, 9);
    expect(areas[1]).toBeCloseTo(120, 9);
    expect(r.branchPoints.length).toBe(2);
    r.profile.loops.forEach((l) => expectHeadToTail(l));
  });

  it("dangling lines and bridges become open wires", () => {
    const s = createSketch();
    const a = addRectangle(s, [0, 0], [10, 10]);
    addLine(s, a.points[2], [15, 15]); // dangling
    const b = addRectangle(s, [30, 0], [40, 10]);
    addLine(s, a.points[1], b.points[0]); // bridge between the two squares
    const r = extractProfiles(s);
    expect(r.profile.loops.length).toBe(2);
    expect(r.openWires.length).toBe(2);
    expect(r.openWires.every((w) => !w.closed && w.segments.length === 1)).toBe(true);
  });

  it("reports crossings without splitting", () => {
    const s = createSketch();
    addLine(s, [0, 0], [10, 10]);
    addLine(s, [0, 10], [10, 0]);
    const r = extractProfiles(s);
    expect(r.profile.loops.length).toBe(0);
    expect(r.openWires.length).toBe(2);
    expect(r.crossings.length).toBe(1);
    expect(r.crossings[0].point[0]).toBeCloseTo(5, 12);
  });

  it("joins an open chain head-to-tail and flips segment directions", () => {
    const s = createSketch();
    const p0 = addPoint(s, [0, 0]);
    const p1 = addPoint(s, [10, 0]);
    const p2 = addPoint(s, [15, 5]);
    const p3 = addPoint(s, [15, 20]);
    addLine(s, p1, p0); // reversed
    addArc(s, [10, 5], p1, p2);
    addLine(s, p3, p2); // reversed
    const r = extractProfiles(s);
    expect(r.openWires.length).toBe(1);
    const w = r.openWires[0];
    expect(w.segments.map((g) => g.type)).toEqual(["line", "arc", "line"]);
    for (let i = 0; i + 1 < w.segments.length; i++) {
      const e = segmentEnd(w.segments[i]), b = segmentStart(w.segments[i + 1]);
      expect(Math.hypot(e[0] - b[0], e[1] - b[1])).toBeLessThan(1e-12);
    }
  });

  it("closes loops through geometrically coincident (unshared) points, arcs and splines", () => {
    const s = createSketch();
    // two arcs making a circle, endpoints duplicated but coincident
    addArc(s, [0, 0], [5, 0], [-5, 0]);
    addArc(s, [0, 0], [-5, 0], [5, 0]);
    const sp = [[20, 0], [30, 5], [25, 15], [15, 10]].map((p) => addPoint(s, p as Vec2));
    addSpline(s, sp, { closed: true });
    addCircle(s, [50, 0], 3);
    const r = extractProfiles(s);
    expect(r.profile.loops.length).toBe(3);
    const types = r.profile.loops.map((l) => l.map((g) => g.type).join(","));
    expect(types).toContain("arc,arc");
    expect(types).toContain("bspline");
    expect(types).toContain("circle");
    for (const l of r.profile.loops) {
      expect(loopSignedArea(l)).toBeGreaterThan(0);
      expectHeadToTail(l, 1e-6);
    }
    const two = r.profile.loops.find((l) => l.length === 2)!;
    expect(loopSignedArea(two)).toBeCloseTo(Math.PI * 25, 9);
  });

  it("splits a circle touched by a line endpoint", () => {
    const s = createSketch();
    addCircle(s, [0, 0], 5);
    addLine(s, [5, 0], [-5, 0]); // diameter with both ends on the circle
    const r = extractProfiles(s);
    expect(r.profile.loops.length).toBe(2);
    for (const l of r.profile.loops) expect(loopSignedArea(l)).toBeCloseTo((Math.PI * 25) / 2, 9);
  });

  it("handles figure-eight sharing one vertex and empty sketches", () => {
    const s = createSketch();
    const c = addPoint(s, [0, 0]);
    const a1 = addPoint(s, [5, 5]), a2 = addPoint(s, [5, -5]);
    const b1 = addPoint(s, [-5, 5]), b2 = addPoint(s, [-5, -5]);
    addLine(s, c, a1);
    addLine(s, a1, a2);
    addLine(s, a2, c);
    addLine(s, c, b1);
    addLine(s, b1, b2);
    addLine(s, b2, c);
    const r = extractProfiles(s);
    expect(r.profile.loops.length).toBe(2);
    const e = extractProfiles(createSketch());
    expect(e.profile.loops).toEqual([]);
    expect(e.openWires).toEqual([]);
  });
});

describe("loop utilities", () => {
  it("point containment and sampling", () => {
    const s = createSketch();
    addRectangle(s, [0, 0], [10, 10]);
    const loop = extractProfiles(s).profile.loops[0];
    expect(loopContainsPoint(loop, [5, 5])).toBe(true);
    expect(loopContainsPoint(loop, [15, 5])).toBe(false);
    expect(sampleLoop(loop).length).toBe(4);
    const circle: Loop2D = [{ type: "circle", center: [0, 0], radius: 2 }];
    expect(loopContainsPoint(circle, [1, 1])).toBe(true);
    expect(loopContainsPoint(circle, [2, 2])).toBe(false);
    const pts = sampleSegment({ type: "arc", start: [1, 0], mid: [0, 1], end: [-1, 0] }, 8);
    expect(pts.length).toBe(9);
    expect(pts[0]).toEqual([1, 0]);
    expect(pts[8]).toEqual([-1, 0]);
    expect(Math.hypot(pts[4][0], pts[4][1] - 1)).toBeLessThan(1e-12);
  });
});
