import { describe, expect, it } from "vitest";
import type { Segment2D, Vec2 } from "../core/types";
import {
  bsplineDomain,
  closestPointBSpline2,
  derivBSpline2,
  evalBSpline2,
  insertKnot2,
  interpolateBSpline2,
  reverseBSpline2,
  splitBSpline2,
} from "./bspline";
import {
  arcGeomFrom3,
  arcParamOf,
  closestPointOnSegment,
  pointInPolygon,
  polygonSignedArea,
  reverseSegment,
  segmentAreaTerm,
  segmentIntersections,
  segmentLength,
  segmentPointAt,
  segmentTangentAt,
  splitSegment,
  subSegment,
} from "./geom2d";

const d = (a: Vec2, b: Vec2) => Math.hypot(a[0] - b[0], a[1] - b[1]);

describe("arcs and segments", () => {
  it("arc geometry from three points (CCW and CW)", () => {
    const g = arcGeomFrom3([1, 0], [0, 1], [-1, 0])!;
    expect(g.radius).toBeCloseTo(1, 12);
    expect(g.sweep).toBeCloseTo(Math.PI, 12);
    const h = arcGeomFrom3([-1, 0], [0, 1], [1, 0])!;
    expect(h.sweep).toBeCloseTo(-Math.PI, 12);
    // large arc (> 180 deg)
    const big = arcGeomFrom3([1, 0], [-1, 0], [0, -1])!;
    expect(big.sweep).toBeCloseTo(1.5 * Math.PI, 12);
    expect(arcGeomFrom3([0, 0], [1, 1], [2, 2])).toBeNull();
    expect(arcParamOf(g, [0, 1])).toBeCloseTo(0.5, 12);
    expect(arcParamOf(g, [0.7, -0.7])).toBeLessThan(0);
    expect(arcParamOf(g, [-0.7, -0.7])).toBeGreaterThan(1);
  });

  it("evaluates, measures, reverses and splits segments", () => {
    const arc: Segment2D = { type: "arc", start: [2, 0], mid: [0, 2], end: [-2, 0] };
    expect(segmentLength(arc)).toBeCloseTo(2 * Math.PI, 12);
    expect(d(segmentPointAt(arc, 0.5), [0, 2])).toBeLessThan(1e-12);
    const t = segmentTangentAt(arc, 0);
    expect(t[0]).toBeCloseTo(0, 12);
    expect(t[1]).toBeCloseTo(1, 12);
    const r = reverseSegment(arc);
    expect(d(segmentPointAt(r, 0.25), segmentPointAt(arc, 0.75))).toBeLessThan(1e-12);
    const [a, b] = splitSegment(arc, 0.25);
    expect(segmentLength(a) + segmentLength(b)).toBeCloseTo(2 * Math.PI, 12);
    const sub = subSegment({ type: "circle", center: [0, 0], radius: 1 }, 0.75, 1.25);
    expect(segmentLength(sub)).toBeCloseTo(Math.PI, 12);
    expect(d(segmentPointAt(sub, 0.5), [1, 0])).toBeLessThan(1e-12);
    // closed loop area via area terms: half disc above the diameter
    const area = segmentAreaTerm(arc) + segmentAreaTerm({ type: "line", a: [-2, 0], b: [2, 0] });
    expect(area).toBeCloseTo(2 * Math.PI, 12);
    expect(segmentAreaTerm({ type: "circle", center: [5, 5], radius: 1 })).toBeCloseTo(Math.PI, 12);
  });

  it("closest points", () => {
    const l = closestPointOnSegment({ type: "line", a: [0, 0], b: [10, 0] }, [12, 3]);
    expect(l.point).toEqual([10, 0]);
    const c = closestPointOnSegment({ type: "circle", center: [0, 0], radius: 2 }, [0, 5]);
    expect(c.distance).toBeCloseTo(3, 12);
    expect(c.t).toBeCloseTo(0.25, 12);
    const a = closestPointOnSegment({ type: "arc", start: [2, 0], mid: [0, 2], end: [-2, 0] }, [1, -5]);
    expect(a.point).toEqual([2, 0]);
  });

  it("polygons", () => {
    const sq: Vec2[] = [[0, 0], [1, 0], [1, 1], [0, 1]];
    expect(polygonSignedArea(sq)).toBe(1);
    expect(pointInPolygon(sq, [0.5, 0.5])).toBe(true);
    expect(pointInPolygon(sq, [1.5, 0.5])).toBe(false);
  });
});

describe("B-splines", () => {
  const pts: Vec2[] = [[0, 0], [2, 3], [5, 2], [7, 6], [10, 4]];

  it("interpolates open data", () => {
    const c = interpolateBSpline2(pts);
    expect(c.degree).toBe(3);
    expect(c.knots.length).toBe(c.controlPoints.length + 4);
    // passes through every point at some parameter
    for (const p of pts) expect(closestPointBSpline2(c, p).distance).toBeLessThan(1e-9);
    const [u0, u1] = bsplineDomain(c);
    expect(d(evalBSpline2(c, u0), pts[0])).toBeLessThan(1e-12);
    expect(d(evalBSpline2(c, u1), pts[4])).toBeLessThan(1e-12);
    // derivative matches finite differences
    const u = 0.37, h = 1e-6;
    const fd: Vec2 = [(evalBSpline2(c, u + h)[0] - evalBSpline2(c, u - h)[0]) / (2 * h), (evalBSpline2(c, u + h)[1] - evalBSpline2(c, u - h)[1]) / (2 * h)];
    const an = derivBSpline2(c, u);
    expect(d(fd, an)).toBeLessThan(1e-5);
    // degenerate inputs
    expect(interpolateBSpline2([[1, 1]]).degree).toBe(1);
    expect(interpolateBSpline2([[0, 0], [1, 1]]).degree).toBe(1);
    expect(interpolateBSpline2([[0, 0], [1, 1], [2, 0]]).degree).toBe(2);
  });

  it("interpolates closed data with a C1 seam", () => {
    const c = interpolateBSpline2(pts, { closed: true });
    const [u0, u1] = bsplineDomain(c);
    expect(d(evalBSpline2(c, u0), evalBSpline2(c, u1))).toBeLessThan(1e-12);
    expect(d(derivBSpline2(c, u0), derivBSpline2(c, u1))).toBeLessThan(1e-9);
    for (const p of pts) expect(closestPointBSpline2(c, p).distance).toBeLessThan(1e-8);
  });

  it("knot insertion and splitting preserve the curve", () => {
    const c = interpolateBSpline2(pts);
    const k = insertKnot2(c, 0.4);
    for (const u of [0.1, 0.4, 0.77]) expect(d(evalBSpline2(c, u), evalBSpline2(k, u))).toBeLessThan(1e-12);
    const [l, r] = splitBSpline2(c, 0.6);
    expect(d(evalBSpline2(l, 0.3), evalBSpline2(c, 0.3))).toBeLessThan(1e-12);
    expect(d(evalBSpline2(r, 0.8), evalBSpline2(c, 0.8))).toBeLessThan(1e-12);
    expect(d(evalBSpline2(l, 0.6), evalBSpline2(r, 0.6))).toBeLessThan(1e-12);
    const rev = reverseBSpline2(c);
    expect(d(evalBSpline2(rev, 0.25), evalBSpline2(c, 0.75))).toBeLessThan(1e-12);
  });

  it("intersects a spline with a line", () => {
    const c = interpolateBSpline2(pts);
    const hits = segmentIntersections({ type: "bspline", curve: c }, { type: "line", a: [0, 3], b: [10, 3] });
    expect(hits.length).toBeGreaterThanOrEqual(2);
    for (const h of hits) expect(Math.abs(h.point[1] - 3)).toBeLessThan(1e-6);
  });
});
