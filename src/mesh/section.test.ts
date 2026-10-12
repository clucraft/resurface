import { describe, expect, it } from "vitest";
import type { PlaneDef } from "../core/types";
import { sectionMesh, sectionMeshParallel } from "./section";
import { MeshSpatialIndex } from "./spatial";
import { makeCylinder, makeGrid, makeIcosphere, makeUVSphere } from "./util/generate";

const plane = (o: [number, number, number], n: [number, number, number]): PlaneDef => ({ kind: "plane", origin: o, normal: n });

describe("sectionMesh", () => {
  it("sphere section is one closed circle", () => {
    const s = makeIcosphere(10, 4);
    const res = sectionMesh(s, plane([0, 0, 3], [0, 0, 1]));
    expect(res.length).toBe(1);
    expect(res[0].closed).toBe(true);
    const r = Math.sqrt(100 - 9);
    for (const p of res[0].points) {
      expect(p[2]).toBeCloseTo(3, 5);
      expect(Math.hypot(p[0], p[1])).toBeGreaterThan(r * 0.99);
      expect(Math.hypot(p[0], p[1])).toBeLessThan(r * 1.0001);
    }
    // orientation consistent with the outward triangles: CCW around +Z seen from above
    let area2 = 0;
    const pts = res[0].points;
    for (let i = 0; i < pts.length; i++) {
      const a = pts[i], b = pts[(i + 1) % pts.length];
      area2 += a[0] * b[1] - a[1] * b[0];
    }
    expect(Math.abs(area2 / 2)).toBeGreaterThan(Math.PI * r * r * 0.97);
  });

  it("plane exactly through a ring of vertices still gives one clean loop", () => {
    // UV sphere with an even ring count has an equator ring at z = 0 exactly
    const s = makeUVSphere(5, 32, 16);
    const res = sectionMesh(s, plane([0, 0, 0], [0, 0, 1]));
    expect(res.length).toBe(1);
    expect(res[0].closed).toBe(true);
    expect(res[0].points.length).toBe(32);
    for (const p of res[0].points) expect(Math.hypot(p[0], p[1])).toBeCloseTo(5, 4);
  });

  it("open surfaces give open chains; two cylinders → two loops", () => {
    const g = makeGrid(10, 10, 10, 10);
    const res = sectionMesh(g, plane([3.3, 0, 0], [1, 0, 0]));
    expect(res.length).toBe(1);
    expect(res[0].closed).toBe(false);
    const ys = res[0].points.map((p) => p[1]);
    expect(Math.min(...ys)).toBeCloseTo(0, 6);
    expect(Math.max(...ys)).toBeCloseTo(10, 6);
    // diagonal plane through vertices of the grid
    const d = sectionMesh(g, plane([0, 0, 0], [Math.SQRT1_2, -Math.SQRT1_2, 0]));
    expect(d.length).toBe(1);
    expect(d[0].points.length).toBe(11);

    const c = makeCylinder(4, 10, 40, 10);
    const cs = sectionMesh(c, plane([0, 0, 0], [1, 0, 0]));
    expect(cs.length).toBe(2);
    for (const pl of cs) expect(pl.closed).toBe(false);
    const ring = sectionMesh(c, plane([0, 0, 5.5], [0, 0, 1]));
    expect(ring.length).toBe(1);
    expect(ring[0].closed).toBe(true);
  });

  it("candidates from the spatial index give the same result; parallel sections match", () => {
    const s = makeIcosphere(10, 4);
    const idx = new MeshSpatialIndex(s);
    const pl = plane([0, 0, -4.2], [0, 0.6, 0.8]);
    const cand = idx.trianglesIntersectingPlane(pl);
    expect(cand.length).toBeLessThan(s.indices.length / 3 / 4);
    const a = sectionMesh(s, pl);
    const b = sectionMesh(s, pl, { candidates: cand });
    expect(b.length).toBe(a.length);
    expect(b[0].points.length).toBe(a[0].points.length);
    const offsets = [-8, -3, 0, 2.5, 9.99, 12];
    const par = sectionMeshParallel(s, [0, 0, 0], [0, 0.6, 0.8], offsets);
    for (let i = 0; i < offsets.length; i++) {
      const single = sectionMesh(s, plane([0, 0.6 * offsets[i], 0.8 * offsets[i]], [0, 0.6, 0.8]));
      expect(par[i].length).toBe(single.length);
      if (single.length) expect(par[i][0].points.length).toBe(single[0].points.length);
    }
    expect(par[5].length).toBe(0);
    idx.dispose();
  });
});
