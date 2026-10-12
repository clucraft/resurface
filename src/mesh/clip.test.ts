import { describe, expect, it } from "vitest";
import type { PlaneDef } from "../core/types";
import { clipByPlane } from "./clip";
import { buildEdges, findBoundaryLoops, meshArea } from "./topology";
import { makeGrid, makeIcosphere } from "./util/generate";

const plane = (o: [number, number, number], n: [number, number, number]): PlaneDef => ({ kind: "plane", origin: o, normal: n });

describe("clipByPlane", () => {
  it("cuts a sphere exactly at the plane (watertight cut, single circular border)", () => {
    const s = makeIcosphere(10, 4);
    s.colors = new Uint8Array(s.positions.length).fill(200);
    const out = clipByPlane(s, plane([0, 0, 2], [0, 0, 1]), "positive", true);
    const P = out.positions;
    for (let v = 0; v < P.length / 3; v++) expect(P[3 * v + 2]).toBeGreaterThanOrEqual(2 - 1e-5);
    const loops = findBoundaryLoops(out);
    expect(loops.length).toBe(1);
    const r = Math.sqrt(100 - 4);
    for (const v of loops[0]) {
      expect(P[3 * v + 2]).toBeCloseTo(2, 4);
      expect(Math.hypot(P[3 * v], P[3 * v + 1])).toBeGreaterThan(r * 0.99);
    }
    // spherical cap area 2πRh with h = 8
    expect(meshArea(out)).toBeCloseTo(2 * Math.PI * 10 * 8, -1);
    // manifold: every edge has 1 or 2 triangles
    const topo = buildEdges(out);
    for (let e = 0; e < topo.edgeCount; e++) expect(topo.edgeFaceCount[e]).toBeLessThanOrEqual(2);
    expect(out.colors!.length).toBe(out.positions.length);
    expect(out.colors![out.colors!.length - 1]).toBe(200);
  });

  it("negative side + complementary areas", () => {
    const s = makeIcosphere(10, 3);
    const pl = plane([1, -2, 0.5], [0.3, 0.4, 0.866]);
    const a = clipByPlane(s, pl, "positive", true);
    const b = clipByPlane(s, pl, "negative", true);
    expect(meshArea(a) + meshArea(b)).toBeCloseTo(meshArea(s), 3);
  });

  it("without cut keeps/drops straddling triangles whole", () => {
    const g = makeGrid(10, 10, 10, 10);
    const out = clipByPlane(g, plane([4.5, 0, 0], [1, 0, 0]), "positive", false);
    // columns with centroid x > 4.5: cells 4..9 (cell 4 triangles centroids at 4.33 and 4.67)
    expect(out.indices.length / 3).toBe(10 * 2 * 5 + 10);
  });

  it("plane through a vertex column does not create slivers", () => {
    const g = makeGrid(4, 4, 4, 4);
    const out = clipByPlane(g, plane([2, 0, 0], [1, 0, 0]), "positive", true);
    expect(out.indices.length / 3).toBe(16);
    expect(out.positions.length / 3).toBe(15);
  });
});
