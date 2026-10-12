import { describe, expect, it } from "vitest";
import type { LineDef, PlaneDef, Vec3 } from "../core/types";
import { cross, dot, normalize } from "../core/vec";
import {
  angleBetweenPlanes,
  axisOf,
  centerOf,
  closestPointsBetweenLines,
  distancePointLine,
  distancePointPlane,
  intersectLinePlane,
  intersectLines,
  intersectPlanes,
  intersectThreePlanes,
  lineFrom2Points,
  midPlane,
  offsetPlane,
  planeFrom3Points,
  planeThroughLineAndPoint,
  planeThroughTwoLines,
  projectLineOnPlane,
  projectPointOnLine,
  projectPointOnPlane,
} from "./reference";

const P = (origin: Vec3, normal: Vec3): PlaneDef => ({ kind: "plane", origin, normal: normalize(normal) });
const L = (origin: Vec3, direction: Vec3): LineDef => ({ kind: "line", origin, direction: normalize(direction) });

describe("reference geometry", () => {
  it("plane-plane intersection", () => {
    const a = P([0, 0, 1], [0, 0, 1]);
    const b = P([2, 0, 0], [1, 0, 0]);
    const l = intersectPlanes(a, b)!;
    expect(Math.abs(dot(l.direction, [0, 1, 0]))).toBeCloseTo(1, 12);
    expect(distancePointPlane(l.origin, a)).toBeCloseTo(0, 12);
    expect(distancePointPlane(l.origin, b)).toBeCloseTo(0, 12);
    expect(intersectPlanes(a, P([5, 5, 3], [0, 0, -1]))).toBeNull();
  });

  it("three planes, line-plane, line-line", () => {
    const p = intersectThreePlanes(P([1, 0, 0], [1, 0, 0]), P([0, 2, 0], [0, 1, 0]), P([0, 0, 3], [1, 1, 1]))!;
    expect(p[0]).toBeCloseTo(1, 12);
    expect(p[1]).toBeCloseTo(2, 12);
    expect(p[0] + p[1] + p[2]).toBeCloseTo(3, 12);
    expect(intersectThreePlanes(P([0, 0, 0], [1, 0, 0]), P([1, 0, 0], [1, 0, 0]), P([0, 0, 0], [0, 0, 1]))).toBeNull();
    const q = intersectLinePlane(L([0, 0, 5], [0, 0, -1]), P([0, 0, 2], [0, 0.1, 1]))!;
    expect(q[2]).toBeCloseTo(2, 12);
    expect(intersectLinePlane(L([0, 0, 5], [1, 0, 0]), P([0, 0, 2], [0, 0, 1]))).toBeNull();
    const c = closestPointsBetweenLines(L([0, 0, 0], [1, 0, 0]), L([0, 1, 3], [0, 1, 0]));
    expect(c.distance).toBeCloseTo(3, 12);
    expect(c.p1).toEqual([0, 0, 0]);
    expect(c.p2[2]).toBeCloseTo(3, 12);
    expect(c.parallel).toBe(false);
    const par = closestPointsBetweenLines(L([0, 0, 0], [1, 0, 0]), L([0, 2, 0], [-1, 0, 0]));
    expect(par.parallel).toBe(true);
    expect(par.distance).toBeCloseTo(2, 12);
    const x = intersectLines(L([0, 0, 0], [1, 1, 0]), L([2, 0, 0], [0, 1, 0]))!;
    expect(Math.hypot(x[0] - 2, x[1] - 2, x[2])).toBeLessThan(1e-12);
  });

  it("planes from points / lines, offsets, projections", () => {
    const pl = planeFrom3Points([0, 0, 0], [1, 0, 0], [0, 1, 0])!;
    expect(pl.normal).toEqual([0, 0, 1]);
    expect(planeFrom3Points([0, 0, 0], [1, 1, 1], [2, 2, 2])).toBeNull();
    expect(offsetPlane(pl, 2).origin[2]).toBeCloseTo(2, 12);
    const tl = planeThroughLineAndPoint(L([0, 0, 0], [1, 0, 0]), [0, 0, 5])!;
    expect(Math.abs(tl.normal[1])).toBeCloseTo(1, 12);
    expect(planeThroughLineAndPoint(L([0, 0, 0], [1, 0, 0]), [3, 0, 0])).toBeNull();
    const two = planeThroughTwoLines(L([0, 0, 0], [1, 0, 0]), L([0, 0, 0], [0, 0, 1]))!;
    expect(Math.abs(two.normal[1])).toBeCloseTo(1, 12);
    expect(planeThroughTwoLines(L([0, 0, 0], [1, 0, 0]), L([0, 1, 1], [0, 0, 1]))).toBeNull(); // skew
    expect(projectPointOnLine([3, 4, 5], L([0, 0, 0], [0, 0, 2]))).toEqual([0, 0, 5]);
    expect(projectPointOnPlane([3, 4, 5], pl)).toEqual([3, 4, 0]);
    expect(distancePointLine([3, 4, 5], L([0, 0, 0], [0, 0, 1]))).toBeCloseTo(5, 12);
    const pr = projectLineOnPlane(L([0, 0, 3], [1, 0, 1]), pl)!;
    expect(pr.direction).toEqual([1, 0, 0]);
    expect(projectLineOnPlane(L([0, 0, 3], [0, 0, 1]), pl)).toBeNull();
    expect(lineFrom2Points([1, 1, 1], [1, 1, 1])).toBeNull();
  });

  it("mid planes: parallel and intersecting", () => {
    const m = midPlane(P([0, 0, 1], [0, 0, 1]), P([5, 5, 7], [0, 0, -1]));
    expect(m.normal).toEqual([0, 0, 1]);
    expect(m.origin[2]).toBeCloseTo(4, 12);
    // V-shaped pair: bisector contains the intersection line and halves the angle
    const a = P([0, 0, 0], [-1, 0, 1]);
    const b = P([0, 0, 0], [1, 0, 1]);
    const bis = midPlane(a, b);
    expect(Math.abs(bis.normal[2])).toBeCloseTo(1, 12);
    const other = midPlane(a, b, { other: true });
    expect(Math.abs(other.normal[0])).toBeCloseTo(1, 12);
    expect(angleBetweenPlanes(bis, a)).toBeCloseTo(angleBetweenPlanes(bis, b), 12);
    expect(Math.abs(dot(cross(bis.normal, other.normal), [0, 1, 0]))).toBeCloseTo(1, 12);
  });

  it("axis and centre helpers", () => {
    expect(axisOf({ kind: "cylinder", origin: [1, 2, 3], axis: [0, 0, 2], radius: 1, hMin: 0, hMax: 1 })).toEqual({ kind: "line", origin: [1, 2, 3], direction: [0, 0, 1] });
    expect(axisOf({ kind: "cone", apex: [1, 2, 3], axis: [0, 1, 0], halfAngle: 0.3, hMin: 0, hMax: 1 }).origin).toEqual([1, 2, 3]);
    expect(centerOf({ kind: "sphere", center: [4, 5, 6], radius: 2 })).toEqual({ kind: "point", position: [4, 5, 6] });
  });
});
