import { describe, expect, it } from "vitest";
import type { MeshData } from "../core/types";
import { decimate, decimateClustering, decimateQEM } from "./decimate";
import { buildEdges, computeFaceNormals, findBoundaryLoops, meshBounds, triangleCountOf } from "./topology";
import { makeGrid, makeIcosphere, makeUVSphere } from "./util/generate";

function sphereDeviation(m: MeshData, R: number) {
  let max = 0;
  const P = m.positions;
  for (let v = 0; v < P.length / 3; v++) max = Math.max(max, Math.abs(Math.hypot(P[3 * v], P[3 * v + 1], P[3 * v + 2]) - R));
  return max;
}

function outwardFraction(m: MeshData): number {
  const fn = computeFaceNormals(m);
  const P = m.positions, I = m.indices;
  let ok = 0;
  const nt = triangleCountOf(m);
  for (let t = 0; t < nt; t++) {
    let d = 0;
    for (let k = 0; k < 3; k++) for (let c = 0; c < 3; c++) d += (fn[3 * t + c] * P[3 * I[3 * t + k] + c]) / 3;
    if (d > 0) ok++;
  }
  return ok / nt;
}

describe("decimateQEM", () => {
  it("reduces a sphere to the target count, staying manifold, closed and on the surface", () => {
    const s = makeIcosphere(10, 5); // 20480 triangles
    const t0 = performance.now();
    const d = decimateQEM(s, { targetTriangles: 2000 });
    const ms = performance.now() - t0;
    const nt = triangleCountOf(d);
    expect(nt).toBeLessThanOrEqual(2000);
    expect(nt).toBeGreaterThan(1990);
    const topo = buildEdges(d);
    for (let e = 0; e < topo.edgeCount; e++) expect(topo.edgeFaceCount[e]).toBe(2);
    expect(d.positions.length / 3 - topo.edgeCount + nt).toBe(2); // still a sphere
    expect(sphereDeviation(d, 10)).toBeLessThan(0.15);
    expect(outwardFraction(d)).toBe(1);
    console.log(`[decimate] QEM 20480 → ${nt} in ${ms.toFixed(0)} ms`);
  });

  it("preserves the boundary of an open grid and its planarity", () => {
    const g = makeGrid(60, 40, 60, 40);
    const d = decimateQEM(g, { targetTriangles: 300 });
    expect(triangleCountOf(d)).toBeLessThanOrEqual(300);
    const bb = meshBounds(d);
    expect(bb.min).toEqual([0, 0, 0]);
    expect(bb.max).toEqual([60, 40, 0]);
    const loops = findBoundaryLoops(d);
    expect(loops.length).toBe(1);
    // boundary vertices stay on the rectangle border
    const P = d.positions;
    for (const v of loops[0]) {
      const x = P[3 * v], y = P[3 * v + 1];
      const onBorder = Math.abs(x) < 1e-4 || Math.abs(x - 60) < 1e-4 || Math.abs(y) < 1e-4 || Math.abs(y - 40) < 1e-4;
      expect(onBorder).toBe(true);
    }
    const fn = computeFaceNormals(d);
    for (let t = 0; t < fn.length / 3; t++) expect(fn[3 * t + 2]).toBeGreaterThan(0.999);
  });

  it("keeps sharp features of a curved surface reasonably and carries colours", () => {
    const g = makeGrid(50, 50, 50, 50, (x) => (x < 25 ? 0 : (x - 25) * 0.5));
    g.colors = new Uint8Array(g.positions.length);
    for (let v = 0; v < g.positions.length / 3; v++) g.colors[3 * v] = g.positions[3 * v] < 25 ? 255 : 0;
    const d = decimateQEM(g, { targetTriangles: 400 });
    // crease vertices x=25 should survive: max deviation from the height function stays tiny
    const P = d.positions;
    for (let v = 0; v < P.length / 3; v++) {
      const x = P[3 * v];
      const h = x < 25 ? 0 : (x - 25) * 0.5;
      expect(Math.abs(P[3 * v + 2] - h)).toBeLessThan(0.05);
    }
    expect(d.colors!.length).toBe(d.positions.length);
    let red = 0;
    for (let v = 0; v < P.length / 3; v++) if (d.colors![3 * v] > 128) red++;
    expect(red).toBeGreaterThan(0);
  });

  it("handles dirty input (duplicates, degenerate, non-manifold)", () => {
    const s = makeUVSphere(5, 32, 16);
    const extra = Uint32Array.of(0, 0, 1, s.indices[0], s.indices[1], s.indices[2], 2, 3, 4);
    const idx = new Uint32Array(s.indices.length + extra.length);
    idx.set(s.indices);
    idx.set(extra, s.indices.length);
    const d = decimateQEM({ positions: s.positions, indices: idx }, { targetTriangles: 200 });
    expect(triangleCountOf(d)).toBeLessThanOrEqual(260);
    expect(triangleCountOf(d)).toBeGreaterThan(100);
  });
});

describe("decimateClustering / decimate", () => {
  it("clusters to roughly the target and stays near the surface", () => {
    const s = makeIcosphere(10, 6); // 81920 triangles
    s.colors = new Uint8Array(s.positions.length).fill(50);
    const t0 = performance.now();
    const d = decimateClustering(s, { targetTriangles: 5000 });
    const ms = performance.now() - t0;
    const nt = triangleCountOf(d);
    console.log(`[decimate] clustering 81920 → ${nt} (target 5000) in ${ms.toFixed(0)} ms`);
    expect(nt).toBeGreaterThan(2500);
    expect(nt).toBeLessThan(10000);
    expect(sphereDeviation(d, 10)).toBeLessThan(0.3);
    expect(d.colors![0]).toBe(50);
    const fixed = decimateClustering(s, { cellSize: 2 });
    expect(triangleCountOf(fixed)).toBeLessThan(nt);
  });

  it("decimate chains clustering and QEM above the threshold", () => {
    const s = makeIcosphere(10, 6);
    const d = decimate(s, { targetTriangles: 3000, clusterAbove: 50_000 });
    expect(triangleCountOf(d)).toBeLessThanOrEqual(3000);
    expect(triangleCountOf(d)).toBeGreaterThan(2500);
    expect(sphereDeviation(d, 10)).toBeLessThan(0.3);
    const same = decimate(s, { targetTriangles: 1e9 });
    expect(triangleCountOf(same)).toBe(triangleCountOf(s));
  });
});
