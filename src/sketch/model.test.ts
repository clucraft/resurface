import { describe, expect, it } from "vitest";
import {
  addArc,
  addArc3,
  addCircle,
  addConstraint,
  addLine,
  addPoint,
  addRectangle,
  addSpline,
  arcInfo,
  cloneSketch,
  createSketch,
  deleteEntity,
  deletePoint,
  entityToSegments,
  findNearest,
  genId,
  getEntity,
  getPoint,
  measure,
  mergePoints,
  validateConstraint,
  type Sketch,
} from "./model";

describe("sketch model", () => {
  it("generates unique ids and survives JSON round trips", () => {
    const s = createSketch();
    const a = addPoint(s, [0, 0]);
    const l = addLine(s, a, [3, 4]);
    const c = addConstraint(s, { type: "length", refs: [l] });
    const ids = [...s.points.map((p) => p.id), ...s.entities.map((e) => e.id), ...s.constraints.map((k) => k.id)];
    expect(new Set(ids).size).toBe(ids.length);
    const json = JSON.parse(JSON.stringify(s)) as Sketch;
    expect(json).toEqual(s);
    // missing counter is recomputed from existing ids
    delete json.nextId;
    const n = genId(json, "x");
    expect(ids).not.toContain(n);
    // dimension value defaults to the current measurement
    expect((getEntity(s, l) as { p1: string }).p1).toBe(a);
    expect(s.constraints.find((k) => k.id === c)).toMatchObject({ value: 5, driving: true });
  });

  it("clones deeply", () => {
    const s = createSketch();
    const sp = addSpline(s, [[0, 0], [1, 1], [2, 0]]);
    const d = addConstraint(s, { type: "fix", refs: [s.points[0].id] });
    const c = cloneSketch(s);
    c.points[0].x = 99;
    (c.entities[0] as { points: string[] }).points.push("zz");
    c.constraints[0].refs.push("q");
    expect(s.points[0].x).toBe(0);
    expect((getEntity(s, sp) as { points: string[] }).points.length).toBe(3);
    expect(s.constraints.find((k) => k.id === d)!.refs).toEqual([s.points[0].id]);
  });

  it("builds arcs (CCW), 3-point arcs, circles, rectangles and splines", () => {
    const s = createSketch();
    const arc = addArc(s, [0, 0], [2, 0], [0, 5]); // end projected to radius 2
    const g = arcInfo(s, getEntity(s, arc) as never);
    expect(g.radius).toBeCloseTo(2, 12);
    expect(getPoint(s, (getEntity(s, arc) as { end: string }).end)!.y).toBeCloseTo(2, 12);
    expect(g.sweep).toBeCloseTo(Math.PI / 2, 12);
    // clockwise three points -> start/end swapped
    const a3 = addArc3(s, [0, 10], [10, 0], [0, -10])!;
    const e3 = getEntity(s, a3) as { start: string; end: string };
    expect(getPoint(s, e3.start)!.y).toBeCloseTo(-10, 12);
    const g3 = arcInfo(s, getEntity(s, a3) as never);
    expect(g3.sweep).toBeCloseTo(Math.PI, 9);
    expect(g3.center[0]).toBeCloseTo(0, 9);
    expect(addArc3(s, [0, 0], [1, 1], [2, 2])).toBeNull();
    const c = addCircle(s, [5, 5], -3);
    expect((getEntity(s, c) as { radius: number }).radius).toBe(3);
    const r = addRectangle(s, [10, 10], [0, 0]);
    expect(r.lines.length).toBe(4);
    expect(r.constraints.length).toBe(4);
    expect(getPoint(s, r.points[0])).toMatchObject({ x: 0, y: 0 });
    const segs = entityToSegments(s, arc);
    expect(segs[0].type).toBe("arc");
    const sp = addSpline(s, [[0, 0], [1, 2], [3, 1], [4, 4]]);
    expect(entityToSegments(s, sp)[0].type).toBe("bspline");
    expect(entityToSegments(s, c)[0]).toEqual({ type: "circle", center: [5, 5], radius: 3 });
  });

  it("validates constraints", () => {
    const s = createSketch();
    const l = addLine(s, [0, 0], [1, 0]);
    const c = addCircle(s, [0, 0], 1);
    const p = addPoint(s, [3, 3]);
    expect(validateConstraint(s, { type: "horizontal", refs: [l] })).toBeNull();
    expect(validateConstraint(s, { type: "horizontal", refs: [c] })).toMatch(/cannot be applied/);
    expect(validateConstraint(s, { type: "tangent", refs: [l, c] })).toBeNull();
    expect(validateConstraint(s, { type: "radius", refs: [c], value: -1 })).toMatch(/positive/);
    expect(validateConstraint(s, { type: "parallel", refs: [l, "nope"] })).toMatch(/unknown/);
    expect(validateConstraint(s, { type: "bogus", refs: [] })).toMatch(/Unknown constraint/);
    expect(validateConstraint(s, { type: "distance", refs: [p, l] })).toBeNull();
    expect(validateConstraint(s, { type: "concentric", refs: [p, c] })).toBeNull();
    expect(() => addConstraint(s, { type: "vertical", refs: [c] })).toThrow();
  });

  it("deletes entities with orphan cleanup", () => {
    const s = createSketch();
    const r = addRectangle(s, [0, 0], [10, 5]);
    addConstraint(s, { type: "length", refs: [r.lines[0]], value: 10 });
    addConstraint(s, { type: "fix", refs: [r.points[0]] });
    deleteEntity(s, r.lines[0]);
    // points still used by neighbours stay
    expect(s.points.length).toBe(4);
    expect(s.constraints.some((c) => c.type === "length")).toBe(false);
    expect(s.constraints.filter((c) => c.type === "horizontal").length).toBe(1);
    deleteEntity(s, r.lines[3]);
    // point 0 now unused -> removed together with its fix constraint
    expect(getPoint(s, r.points[0])).toBeUndefined();
    expect(s.constraints.some((c) => c.type === "fix")).toBe(false);
    deletePoint(s, r.points[2]);
    expect(s.entities.length).toBe(0);
    expect(s.points.length).toBe(0);
    expect(s.constraints.length).toBe(0);
  });

  it("merges points", () => {
    const s = createSketch();
    const l1 = addLine(s, [0, 0], [1, 0]);
    const l2 = addLine(s, [1, 0], [1, 1]);
    const a = (getEntity(s, l1) as { p2: string }).p2;
    const b = (getEntity(s, l2) as { p1: string }).p1;
    addConstraint(s, { type: "coincident", refs: [a, b] });
    mergePoints(s, a, b);
    expect((getEntity(s, l2) as { p1: string }).p1).toBe(a);
    expect(s.constraints.length).toBe(0);
    expect(s.points.length).toBe(3);
  });

  it("picks points before entities", () => {
    const s = createSketch();
    const l = addLine(s, [0, 0], [10, 0]);
    const c = addCircle(s, [20, 0], 5, { construction: true });
    expect(findNearest(s, [0.05, 0.05], 0.1)).toMatchObject({ kind: "point" });
    const hit = findNearest(s, [5, 0.05], 0.1)!;
    expect(hit).toMatchObject({ kind: "entity", id: l });
    expect(hit.point[1]).toBeCloseTo(0, 12);
    expect(findNearest(s, [25.05, 0], 0.1)).toMatchObject({ kind: "entity", id: c });
    expect(findNearest(s, [25.05, 0], 0.1, { includeConstruction: false })).toBeNull();
    expect(findNearest(s, [5, 3], 0.1)).toBeNull();
  });

  it("measures dimensions", () => {
    const s = createSketch();
    const a = addPoint(s, [0, 0]);
    const b = addPoint(s, [3, 4]);
    const l = addLine(s, a, b);
    const l2 = addLine(s, [0, 2], [10, 2]);
    const c = addCircle(s, [10, 10], 2.5);
    const arc = addArc(s, [0, 0], [1, 0], [0, 1]);
    expect(measure(s, { type: "distance", refs: [a, b] })).toBeCloseTo(5, 12);
    expect(measure(s, { type: "distance", refs: [l] })).toBeCloseTo(5, 12);
    expect(measure(s, { type: "distance", refs: [b, l2] })).toBeCloseTo(2, 12);
    expect(measure(s, { type: "distance", refs: [c, a] })).toBeCloseTo(Math.hypot(10, 10), 12);
    expect(measure(s, { type: "horizontalDistance", refs: [a, b] })).toBe(3);
    expect(measure(s, { type: "verticalDistance", refs: [l] })).toBe(4);
    expect(measure(s, { type: "radius", refs: [c] })).toBe(2.5);
    expect(measure(s, { type: "diameter", refs: [c] })).toBe(5);
    expect(measure(s, { type: "length", refs: [arc] })).toBeCloseTo(Math.PI / 2, 12);
    expect(measure(s, { type: "angle", refs: [l2, l] })).toBeCloseTo(Math.atan2(4, 3), 12);
    expect(measure(s, { type: "radius", refs: [l] })).toBeNull();
    expect(measure(s, { type: "distance", refs: ["missing", a] })).toBeNull();
  });
});
