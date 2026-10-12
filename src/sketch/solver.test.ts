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
import { solveSketch } from "./solver";

const pos = (s: Sketch, id: string): Vec2 => {
  const p = getPoint(s, id)!;
  return [p.x, p.y];
};
const dist = (a: Vec2, b: Vec2) => Math.hypot(a[0] - b[0], a[1] - b[1]);

describe("solver basics", () => {
  it("solves a dimensioned rectangle and reports 0 DOF", () => {
    const s = createSketch();
    const r = addRectangle(s, [0, 0], [9, 4.5]);
    addConstraint(s, { type: "fix", refs: [r.points[0]] });
    addConstraint(s, { type: "length", refs: [r.lines[0]], value: 10 });
    addConstraint(s, { type: "length", refs: [r.lines[1]], value: 5 });
    const res = solveSketch(s);
    expect(res.status).toBe("ok");
    expect(res.dof).toBe(0);
    expect(res.residual).toBeLessThan(1e-8);
    const q = res.sketch;
    expect(pos(q, r.points[2])[0]).toBeCloseTo(10, 9);
    expect(pos(q, r.points[2])[1]).toBeCloseTo(5, 9);
    expect(pos(q, r.points[0])).toEqual([0, 0]);
    // the input is untouched
    expect(pos(s, r.points[2])).toEqual([9, 4.5]);
  });

  it("counts DOF of under-constrained geometry and moves it minimally", () => {
    const s = createSketch();
    const a = addPoint(s, [0, 0]);
    const b = addPoint(s, [8, 0]);
    const l = addLine(s, a, b);
    expect(solveSketch(s).dof).toBe(4);
    addConstraint(s, { type: "length", refs: [l], value: 10 });
    const res = solveSketch(s);
    expect(res.status).toBe("ok");
    expect(res.dof).toBe(3);
    // minimum-norm: both endpoints move symmetrically by 1
    expect(pos(res.sketch, a)[0]).toBeCloseTo(-1, 9);
    expect(pos(res.sketch, b)[0]).toBeCloseTo(9, 9);
    expect(pos(res.sketch, a)[1]).toBeCloseTo(0, 12);
  });

  it("solves a triangle from three distances", () => {
    const s = createSketch();
    const p = [addPoint(s, [0, 0], { fixed: true }), addPoint(s, [4, 0.3]), addPoint(s, [1, 3])];
    const l = [addLine(s, p[0], p[1]), addLine(s, p[1], p[2]), addLine(s, p[2], p[0])];
    addConstraint(s, { type: "horizontal", refs: [l[0]] });
    addConstraint(s, { type: "distance", refs: [p[0], p[1]], value: 3 });
    addConstraint(s, { type: "distance", refs: [p[1], p[2]], value: 4 });
    addConstraint(s, { type: "distance", refs: [p[2], p[0]], value: 5 });
    const res = solveSketch(s);
    expect(res.status).toBe("ok");
    expect(res.dof).toBe(0);
    const q = res.sketch;
    expect(dist(pos(q, p[0]), pos(q, p[1]))).toBeCloseTo(3, 9);
    expect(dist(pos(q, p[1]), pos(q, p[2]))).toBeCloseTo(4, 9);
    expect(dist(pos(q, p[2]), pos(q, p[0]))).toBeCloseTo(5, 9);
    // stays on the same side (no flip)
    expect(pos(q, p[2])[1]).toBeGreaterThan(0);
  });

  it("handles parallel, perpendicular, equal, angle and midpoint", () => {
    const s = createSketch();
    const l1 = addLine(s, [0, 0], [10, 1]);
    const l2 = addLine(s, [0, 5], [7, 6.5]);
    const l3 = addLine(s, [2, -3], [3, 4]);
    const l4 = addLine(s, [12, 0], [15, 6]);
    addConstraint(s, { type: "parallel", refs: [l1, l2] });
    addConstraint(s, { type: "perpendicular", refs: [l1, l3] });
    addConstraint(s, { type: "equal", refs: [l1, l2] });
    addConstraint(s, { type: "angle", refs: [l1, l4], value: Math.PI / 3 });
    const m = addPoint(s, [5, 3]);
    addConstraint(s, { type: "midpoint", refs: [m, l3] });
    const res = solveSketch(s);
    expect(res.status).toBe("ok");
    const q = res.sketch;
    expect(measure(q, { type: "angle", refs: [l1, l2] })!).toBeLessThan(1e-8);
    expect(Math.abs(measure(q, { type: "angle", refs: [l1, l3] })! - Math.PI / 2)).toBeLessThan(1e-8);
    expect(measure(q, { type: "length", refs: [l1] })!).toBeCloseTo(measure(q, { type: "length", refs: [l2] })!, 8);
    expect(measure(q, { type: "angle", refs: [l1, l4] })!).toBeCloseTo(Math.PI / 3, 8);
    const e3 = getEntity(q, l3) as { p1: string; p2: string };
    const mid: Vec2 = [(pos(q, e3.p1)[0] + pos(q, e3.p2)[0]) / 2, (pos(q, e3.p1)[1] + pos(q, e3.p2)[1]) / 2];
    expect(dist(pos(q, m), mid)).toBeLessThan(1e-8);
  });

  it("solves tangencies (line-arc at shared endpoint, line-circle, circle-circle, arc-arc)", () => {
    const s = createSketch();
    const a = addPoint(s, [0, 0]);
    const b = addPoint(s, [10, 0.5]);
    const line = addLine(s, a, b);
    const c = addPoint(s, [10.5, 4]);
    const e = addPoint(s, [14.5, 4.2]);
    const arc = addArc(s, c, b, e);
    addConstraint(s, { type: "tangent", refs: [line, arc] });
    const circ = addCircle(s, [5, 6], 3);
    addConstraint(s, { type: "tangent", refs: [line, circ] });
    const circ2 = addCircle(s, [12, 12], 2);
    addConstraint(s, { type: "tangent", refs: [circ, circ2] });
    // arc-arc at shared endpoint
    const c2 = addPoint(s, [14.5, 9]);
    const e2 = addPoint(s, [19, 9]);
    const arc2 = addArc(s, c2, e2, e);
    addConstraint(s, { type: "tangent", refs: [arc, arc2] });
    const res = solveSketch(s);
    expect(res.status).toBe("ok");
    const q = res.sketch;
    // line-arc: radius perpendicular to the line at b
    const A = pos(q, a), B = pos(q, b), C = pos(q, c);
    const d: Vec2 = [B[0] - A[0], B[1] - A[1]];
    expect(Math.abs(d[0] * (B[0] - C[0]) + d[1] * (B[1] - C[1])) / Math.hypot(...d)).toBeLessThan(1e-8);
    // arc end lies on its circle
    const ai = arcInfo(q, getEntity(q, arc) as never);
    expect(dist(pos(q, e), ai.center)).toBeCloseTo(ai.radius, 8);
    // line-circle
    const ce = getEntity(q, circ) as { center: string; radius: number };
    const cc = pos(q, ce.center);
    const h = Math.abs(d[0] * (cc[1] - A[1]) - d[1] * (cc[0] - A[0])) / Math.hypot(...d);
    expect(h).toBeCloseTo(ce.radius, 8);
    // circle-circle external
    const ce2 = getEntity(q, circ2) as { center: string; radius: number };
    expect(dist(cc, pos(q, ce2.center))).toBeCloseTo(ce.radius + ce2.radius, 8);
    // arc-arc: centers collinear with the shared point
    const E = pos(q, e), C2 = pos(q, c2);
    const cr = (E[0] - C[0]) * (E[1] - C2[1]) - (E[1] - C[1]) * (E[0] - C2[0]);
    expect(Math.abs(cr) / (dist(E, C) * dist(E, C2))).toBeLessThan(1e-8);
  });

  it("supports radius/diameter, concentric, pointOnEntity, symmetric, collinear, h/v distances", () => {
    const s = createSketch();
    const c1 = addCircle(s, [0, 0], 4);
    const c2 = addCircle(s, [0.5, 0.2], 6);
    addConstraint(s, { type: "radius", refs: [c1], value: 5 });
    addConstraint(s, { type: "diameter", refs: [c2], value: 16 });
    addConstraint(s, { type: "concentric", refs: [c1, c2] });
    const p = addPoint(s, [5, 1]);
    addConstraint(s, { type: "pointOnEntity", refs: [p, c1] });
    const axis = addLine(s, [0, -10], [0.2, 10], { construction: true });
    const q1 = addPoint(s, [3, 3]);
    const q2 = addPoint(s, [-2, 3.5]);
    addConstraint(s, { type: "symmetric", refs: [q1, q2, axis] });
    const la = addLine(s, [20, 0], [25, 0.2]);
    const lb = addLine(s, [27, 1], [30, 0.5]);
    addConstraint(s, { type: "collinear", refs: [la, lb] });
    const u = addPoint(s, [40, 0]);
    const v = addPoint(s, [43, 2]);
    addConstraint(s, { type: "horizontalDistance", refs: [u, v], value: 5 });
    addConstraint(s, { type: "verticalDistance", refs: [u, v], value: 1 });
    const res = solveSketch(s);
    expect(res.status).toBe("ok");
    const q = res.sketch;
    const e1 = getEntity(q, c1) as { center: string; radius: number };
    const e2 = getEntity(q, c2) as { center: string; radius: number };
    expect(e1.radius).toBeCloseTo(5, 9);
    expect(e2.radius).toBeCloseTo(8, 9);
    expect(dist(pos(q, e1.center), pos(q, e2.center))).toBeLessThan(1e-8);
    expect(dist(pos(q, p), pos(q, e1.center))).toBeCloseTo(5, 8);
    // symmetric about axis line
    const ax = getEntity(q, axis) as { p1: string; p2: string };
    const A = pos(q, ax.p1), B = pos(q, ax.p2);
    const Q1 = pos(q, q1), Q2 = pos(q, q2);
    const d: Vec2 = [B[0] - A[0], B[1] - A[1]];
    expect(Math.abs((Q2[0] - Q1[0]) * d[0] + (Q2[1] - Q1[1]) * d[1])).toBeLessThan(1e-7);
    const M: Vec2 = [(Q1[0] + Q2[0]) / 2 - A[0], (Q1[1] + Q2[1]) / 2 - A[1]];
    expect(Math.abs(d[0] * M[1] - d[1] * M[0])).toBeLessThan(1e-7);
    expect(measure(q, { type: "angle", refs: [la, lb] })!).toBeLessThan(1e-8);
    expect(pos(q, v)[0] - pos(q, u)[0]).toBeCloseTo(5, 9);
    expect(pos(q, v)[1] - pos(q, u)[1]).toBeCloseTo(1, 9);
  });

  it("solves arc length and line-to-line distance", () => {
    const s = createSketch();
    const arc = addArc(s, [0, 0], [5, 0], [0, 5]);
    addConstraint(s, { type: "length", refs: [arc], value: 10 });
    addConstraint(s, { type: "radius", refs: [arc], value: 5 });
    const l1 = addLine(s, [10, 0], [20, 0]);
    const l2 = addLine(s, [10, 3], [20, 3.4]);
    addConstraint(s, { type: "parallel", refs: [l1, l2] });
    addConstraint(s, { type: "distance", refs: [l1, l2], value: 4 });
    const res = solveSketch(s);
    expect(res.status).toBe("ok");
    expect(measure(res.sketch, { type: "length", refs: [arc] })!).toBeCloseTo(10, 7);
    expect(measure(res.sketch, { type: "distance", refs: [l1, l2] })!).toBeCloseTo(4, 7);
  });
});

describe("solver redundancy and conflicts", () => {
  it("accepts consistent redundant constraints and reports them", () => {
    const s = createSketch();
    const r = addRectangle(s, [0, 0], [10, 5]);
    const par = addConstraint(s, { type: "parallel", refs: [r.lines[0], r.lines[2]] });
    const res = solveSketch(s);
    expect(res.status).toBe("ok");
    expect(res.redundant).toContain(par);
    // 8 coords - 4 independent equations
    expect(res.dof).toBe(4);
  });

  it("detects conflicting dimensions and blames the later one", () => {
    const s = createSketch();
    const a = addPoint(s, [0, 0], { fixed: true });
    const b = addPoint(s, [10, 0]);
    const l = addLine(s, a, b);
    addConstraint(s, { type: "horizontal", refs: [l] });
    addConstraint(s, { type: "length", refs: [l], value: 10 });
    const bad = addConstraint(s, { type: "horizontalDistance", refs: [a, b], value: 7 });
    const res = solveSketch(s);
    expect(res.status).toBe("overconstrained");
    expect(res.conflicting).toEqual([bad]);
    // the earlier constraints hold
    expect(measure(res.sketch, { type: "length", refs: [l] })!).toBeCloseTo(10, 8);
    expect(Number.isFinite(res.residual)).toBe(true);
  });

  it("reports conflicts between fixed geometry", () => {
    const s = createSketch();
    const a = addPoint(s, [0, 0], { fixed: true });
    const b = addPoint(s, [3, 0], { fixed: true });
    const k = addConstraint(s, { type: "distance", refs: [a, b], value: 5 });
    const res = solveSketch(s);
    expect(res.status).toBe("overconstrained");
    expect(res.conflicting).toEqual([k]);
  });

  it("fails gracefully on impossible triangles", () => {
    const s = createSketch();
    const p = [addPoint(s, [0, 0], { fixed: true }), addPoint(s, [1, 0]), addPoint(s, [0.5, 1])];
    addConstraint(s, { type: "distance", refs: [p[0], p[1]], value: 1 });
    addConstraint(s, { type: "distance", refs: [p[1], p[2]], value: 1 });
    addConstraint(s, { type: "distance", refs: [p[2], p[0]], value: 5 });
    const res = solveSketch(s);
    expect(res.status).not.toBe("ok");
    for (const q of res.sketch.points) expect(Number.isFinite(q.x) && Number.isFinite(q.y)).toBe(true);
  });

  it("does not explode on degenerate input", () => {
    const s = createSketch();
    const a = addPoint(s, [1, 1]);
    const l0 = addLine(s, a, a); // zero length
    const l1 = addLine(s, [0, 0], [5, 0]);
    addConstraint(s, { type: "parallel", refs: [l0, l1] });
    addConstraint(s, { type: "perpendicular", refs: [l0, l1] });
    const c1 = addCircle(s, [3, 3], 2);
    const c2 = addCircle(s, [3, 3], 2); // coincident centers
    addConstraint(s, { type: "tangent", refs: [c1, c2] });
    addConstraint(s, { type: "concentric", refs: [c1, c2] });
    const z = addArc(s, [8, 8], [8, 8], [8, 8]); // zero-radius arc
    addConstraint(s, { type: "radius", refs: [z], value: 1 });
    const res = solveSketch(s);
    expect(["ok", "failed", "overconstrained"]).toContain(res.status);
    for (const q of res.sketch.points) expect(Number.isFinite(q.x) && Number.isFinite(q.y)).toBe(true);
    for (const e of res.sketch.entities) if (e.type === "circle") expect(Number.isFinite(e.radius)).toBe(true);
  });

  it("ignores invalid constraints and reports them", () => {
    const s = createSketch();
    const l = addLine(s, [0, 0], [1, 1]);
    s.constraints.push({ id: "bad1", type: "horizontal", refs: ["nope"] });
    s.constraints.push({ id: "bad2", type: "radius", refs: [l], value: 3, driving: true });
    const res = solveSketch(s);
    expect(res.invalid).toEqual(["bad1", "bad2"]);
    expect(res.status).toBe("ok");
  });

  it("skips reference (non-driving) dimensions", () => {
    const s = createSketch();
    const l = addLine(s, [0, 0], [3, 4]);
    addConstraint(s, { type: "length", refs: [l], value: 99, driving: false });
    const res = solveSketch(s);
    expect(res.status).toBe("ok");
    expect(measure(res.sketch, { type: "length", refs: [l] })).toBeCloseTo(5, 12);
  });
});

describe("solver dragging", () => {
  it("moves a free point exactly to the target", () => {
    const s = createSketch();
    const a = addPoint(s, [0, 0]);
    const b = addPoint(s, [5, 0]);
    addLine(s, a, b);
    const res = solveSketch(s, { drag: { pointId: b, target: [7, 3] } });
    expect(pos(res.sketch, b)).toEqual([7, 3]);
    expect(pos(res.sketch, a)).toEqual([0, 0]);
  });

  it("keeps hard constraints while dragging (fixed-length line pivots / translates)", () => {
    const s = createSketch();
    const a = addPoint(s, [0, 0], { fixed: true });
    const b = addPoint(s, [5, 0]);
    const l = addLine(s, a, b);
    addConstraint(s, { type: "length", refs: [l], value: 5 });
    const res = solveSketch(s, { drag: { pointId: b, target: [3, 6] } });
    expect(res.status).toBe("ok");
    const B = pos(res.sketch, b);
    expect(Math.hypot(B[0], B[1])).toBeCloseTo(5, 9);
    // the point lands on the circle nearest the target direction
    expect(Math.atan2(B[1], B[0])).toBeCloseTo(Math.atan2(6, 3), 6);

    // free line: dragging one end translates the other to keep the length
    const s2 = createSketch();
    const c = addPoint(s2, [0, 0]);
    const d = addPoint(s2, [5, 0]);
    const l2 = addLine(s2, c, d);
    addConstraint(s2, { type: "length", refs: [l2], value: 5 });
    addConstraint(s2, { type: "horizontal", refs: [l2] });
    const r2 = solveSketch(s2, { drag: { pointId: d, target: [8, 2] } });
    expect(pos(r2.sketch, d)[0]).toBeCloseTo(8, 2);
    expect(pos(r2.sketch, d)[1]).toBeCloseTo(2, 2);
    expect(pos(r2.sketch, c)[0]).toBeCloseTo(3, 2);
    expect(pos(r2.sketch, c)[1]).toBeCloseTo(2, 2);
  });

  it("does not move fixed points when dragged", () => {
    const s = createSketch();
    const a = addPoint(s, [1, 2], { fixed: true });
    const res = solveSketch(s, { drag: { pointId: a, target: [5, 5] } });
    expect(pos(res.sketch, a)).toEqual([1, 2]);
  });
});

describe("solver free points", () => {
  it("reports which points can still move", () => {
    const s = createSketch();
    const r = addRectangle(s, [0, 0], [10, 5]);
    addConstraint(s, { type: "fix", refs: [r.points[0]] });
    addConstraint(s, { type: "length", refs: [r.lines[0]], value: 10 });
    const res = solveSketch(s, { computeFreePoints: true });
    expect(res.dof).toBe(1);
    // p0, p1 fully determined; p2, p3 can move vertically
    expect(res.freePoints!.sort()).toEqual([r.points[2], r.points[3]].sort());
  });
});

describe("solver performance", () => {
  /** Chain of 100 lines + 100 tangent arcs (200 entities) with dimensions. */
  function bigSketch(): { s: Sketch; drag: string } {
    const s = createSketch();
    const n = 100;
    const R = 200;
    let prev: string | null = null;
    let first: string | null = null;
    let lastArc: string | null = null;
    let drag = "";
    for (let i = 0; i < n; i++) {
      const a0 = (i / n) * Math.PI * 2;
      const a1 = ((i + 0.6) / n) * Math.PI * 2;
      const a2 = ((i + 1) / n) * Math.PI * 2;
      const p0: Vec2 = [R * Math.cos(a0), R * Math.sin(a0)];
      const p1: Vec2 = [R * Math.cos(a1), R * Math.sin(a1)];
      const pa: string = prev ?? addPoint(s, p0);
      if (!first) first = pa;
      const pb = addPoint(s, p1);
      const line = addLine(s, pa, pb);
      if (lastArc) addConstraint(s, { type: "tangent", refs: [lastArc, line] });
      const am = (a1 + a2) / 2;
      const center: Vec2 = [(R - 3) * Math.cos(am), (R - 3) * Math.sin(am)];
      const pe: string = i === n - 1 ? first! : addPoint(s, [R * Math.cos(a2), R * Math.sin(a2)]);
      const arc = addArc(s, addPoint(s, center), pb, pe);
      addConstraint(s, { type: "tangent", refs: [line, arc] });
      if (i % 3 === 0) addConstraint(s, { type: "length", refs: [line], value: 7 });
      if (i % 4 === 0) addConstraint(s, { type: "radius", refs: [arc], value: 3 });
      lastArc = arc;
      prev = pe;
      if (i === 37) drag = pb;
    }
    addConstraint(s, { type: "tangent", refs: [lastArc!, s.entities[0].id] });
    addConstraint(s, { type: "fix", refs: [first!] });
    return { s, drag };
  }

  it("solves ~200 entities and drags interactively", () => {
    const { s, drag } = bigSketch();
    expect(s.entities.length).toBe(200);
    let t0 = performance.now();
    const res = solveSketch(s);
    const tSolve = performance.now() - t0;
    expect(res.status).toBe("ok");
    expect(res.residual).toBeLessThan(1e-6);
    // drag steps
    let cur = res.sketch;
    const p = getPoint(cur, drag)!;
    const times: number[] = [];
    for (let k = 1; k <= 20; k++) {
      t0 = performance.now();
      const r = solveSketch(cur, { drag: { pointId: drag, target: [p.x + k * 0.2, p.y + k * 0.1] } });
      times.push(performance.now() - t0);
      expect(r.status).toBe("ok");
      cur = r.sketch;
    }
    times.sort((a, b) => a - b);
    const median = times[Math.floor(times.length / 2)];
    console.log(`[solver perf] full solve ${tSolve.toFixed(1)} ms, drag median ${median.toFixed(2)} ms, max ${times[times.length - 1].toFixed(2)} ms, dof ${res.dof}`);
    expect(median).toBeLessThan(20);
  });
});
