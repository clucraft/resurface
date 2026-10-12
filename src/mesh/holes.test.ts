import { describe, expect, it } from "vitest";
import type { MeshData } from "../core/types";
import { defeature } from "./defeature";
import { fillHoles, listHoles } from "./holes";
import { buildEdges, compactMesh, findBoundaryLoops, triangleCountOf } from "./topology";
import { makeCylinder, makeGrid, makeIcosphere } from "./util/generate";

function centroid(m: MeshData, t: number): [number, number, number] {
  const P = m.positions, I = m.indices;
  const c: [number, number, number] = [0, 0, 0];
  for (let k = 0; k < 3; k++) for (let d = 0; d < 3; d++) c[d] += P[3 * I[3 * t + k] + d] / 3;
  return c;
}

function removeWhere(m: MeshData, pred: (c: [number, number, number]) => boolean): MeshData {
  const nt = triangleCountOf(m);
  const keep = new Uint8Array(nt);
  for (let t = 0; t < nt; t++) keep[t] = pred(centroid(m, t)) ? 0 : 1;
  return compactMesh(m, keep);
}

/** every directed edge at most once, every edge ≤ 2 faces */
function checkManifoldOriented(m: MeshData) {
  const topo = buildEdges(m);
  for (let e = 0; e < topo.edgeCount; e++) expect(topo.edgeFaceCount[e]).toBeLessThanOrEqual(2);
  const seen = new Set<string>();
  for (let t = 0; t < m.indices.length / 3; t++)
    for (let k = 0; k < 3; k++) {
      const key = `${m.indices[3 * t + k]}-${m.indices[3 * t + ((k + 1) % 3)]}`;
      expect(seen.has(key)).toBe(false);
      seen.add(key);
    }
}

function newVertexStats(m: MeshData, firstNew: number, f: (x: number, y: number, z: number) => number) {
  let max = 0, sum = 0, n = 0;
  const P = m.positions;
  for (let v = firstNew; v < P.length / 3; v++) {
    const d = Math.abs(f(P[3 * v], P[3 * v + 1], P[3 * v + 2]));
    max = Math.max(max, d);
    sum += d;
    n++;
  }
  return { max, mean: n ? sum / n : 0, n };
}

describe("fillHoles", () => {
  it("closes a spherical cap hole with a curvature-continuous patch", () => {
    const s = removeWhere(makeIcosphere(10, 4), (c) => c[2] > 7);
    expect(findBoundaryLoops(s).length).toBe(1);
    const nv0 = s.positions.length / 3;
    const { mesh, stats } = fillHoles(s, { smooth: true });
    expect(stats.holesFilled).toBe(1);
    expect(findBoundaryLoops(mesh).length).toBe(0);
    checkManifoldOriented(mesh);
    expect(stats.verticesAdded).toBeGreaterThan(20);
    const dev = newVertexStats(mesh, nv0, (x, y, z) => Math.hypot(x, y, z) - 10);
    // the faired dome follows the sphere (cap height is 3, half-angle 45°)
    console.log(`[holes] sphere cap: thin-plate dev max ${dev.max.toFixed(3)} mean ${dev.mean.toFixed(3)}`);
    expect(dev.max).toBeLessThan(0.8);
    expect(dev.mean).toBeLessThan(0.45);
    // membrane fill is much flatter
    const flat = fillHoles(s, { smooth: false });
    const devFlat = newVertexStats(flat.mesh, nv0, (x, y, z) => Math.hypot(x, y, z) - 10);
    console.log(`[holes] sphere cap: membrane dev max ${devFlat.max.toFixed(3)} mean ${devFlat.mean.toFixed(3)}`);
    expect(devFlat.mean).toBeGreaterThan(dev.mean);
  });

  it("fills a hole in a plane with a planar, refined, consistently oriented patch", () => {
    const g = removeWhere(makeGrid(20, 20, 20, 20), (c) => (c[0] - 10) ** 2 + (c[1] - 10) ** 2 < 16);
    const loops = findBoundaryLoops(g);
    expect(loops.length).toBe(2);
    const holeIdx = loops[0].length < loops[1].length ? 0 : 1;
    const nv0 = g.positions.length / 3;
    const { mesh, stats } = fillHoles(g, { smooth: true, loops: [holeIdx] });
    expect(stats.holesFilled).toBe(1);
    expect(findBoundaryLoops(mesh).length).toBe(1);
    checkManifoldOriented(mesh);
    const dev = newVertexStats(mesh, nv0, (_x, _y, z) => z);
    expect(dev.max).toBeLessThan(1e-4);
    // density matches the surrounding (~1 unit edges): area of hole ≈ 50 → ~50 vertices
    expect(stats.verticesAdded).toBeGreaterThan(15);
    expect(stats.verticesAdded).toBeLessThan(120);
    // maxBoundaryEdges filter keeps the outer border (80 edges) open
    const both = fillHoles(g, { smooth: false, maxBoundaryEdges: 60 });
    expect(both.stats.holesFilled).toBe(1);
    expect(both.stats.skipped).toBe(1);
  });

  it("handles large holes (ear clipping path) and non-planar saddles", () => {
    const big = removeWhere(makeGrid(160, 160, 160, 160), (c) => (c[0] - 80) ** 2 + (c[1] - 80) ** 2 < 60 ** 2);
    const loops = findBoundaryLoops(big);
    const hole = loops.findIndex((l) => l.length < 600);
    expect(loops[hole].length).toBeGreaterThan(300);
    const nv0 = big.positions.length / 3;
    const t0 = performance.now();
    const { mesh, stats } = fillHoles(big, { smooth: false, loops: [hole] });
    const ms = performance.now() - t0;
    expect(stats.holesFilled).toBe(1);
    expect(findBoundaryLoops(mesh).length).toBe(1);
    checkManifoldOriented(mesh);
    expect(newVertexStats(mesh, nv0, (_x, _y, z) => z).max).toBeLessThan(1e-3);
    console.log(`[holes] large hole: ${loops[hole].length} boundary edges, +${stats.verticesAdded} vertices, ${ms.toFixed(0)} ms`);

    const f = (x: number, y: number) => ((x - 10) ** 2 - (y - 10) ** 2) / 20;
    const saddle = removeWhere(makeGrid(40, 40, 20, 20, f), (c) => (c[0] - 10) ** 2 + (c[1] - 10) ** 2 < 25);
    const sl = findBoundaryLoops(saddle);
    const sh = sl.findIndex((l) => l.length < 100);
    const nvs = saddle.positions.length / 3;
    for (const smooth of [false, true]) {
      const r = fillHoles(saddle, { smooth, loops: [sh] });
      checkManifoldOriented(r.mesh);
      const dev = newVertexStats(r.mesh, nvs, (x, y, z) => z - f(x, y));
      // saddle height range inside the hole is ±1.25
      expect(dev.max).toBeLessThan(0.08);
    }
  });

  it("interpolates colours and keeps existing data untouched", () => {
    const g = removeWhere(makeGrid(10, 10, 10, 10), (c) => (c[0] - 5) ** 2 + (c[1] - 5) ** 2 < 4);
    g.colors = new Uint8Array(g.positions.length).fill(100);
    const loops = findBoundaryLoops(g);
    const hole = loops[0].length < loops[1].length ? 0 : 1;
    const { mesh } = fillHoles(g, { smooth: true, loops: [hole] });
    expect(mesh.colors!.length).toBe(mesh.positions.length);
    for (const c of mesh.colors!) expect(c).toBe(100);
    expect(Array.from(mesh.indices.subarray(0, g.indices.length))).toEqual(Array.from(g.indices));
    expect(Array.from(mesh.positions.subarray(0, g.positions.length))).toEqual(Array.from(g.positions));
  });

  it("returns the input when there is nothing to fill; listHoles", () => {
    const s = makeIcosphere(1, 2);
    const r = fillHoles(s, { smooth: true });
    expect(r.mesh).toBe(s);
    expect(r.stats.holesFound).toBe(0);
    const g = makeGrid(4, 4, 4, 4);
    const holes = listHoles(g);
    expect(holes.length).toBe(1);
    expect(holes[0].perimeter).toBeCloseTo(16, 5);
  });

  it("triangle hole and quad hole (minimal cases)", () => {
    const g = makeGrid(3, 3, 3, 3);
    const mask = new Uint8Array(18).fill(1);
    mask[8] = 0; // one triangle in the middle cell
    const tri = compactMesh(g, mask);
    const loops = findBoundaryLoops(tri);
    const hole = loops.findIndex((l) => l.length === 3);
    const r = fillHoles(tri, { smooth: true, loops: [hole] });
    expect(r.stats.trianglesAdded).toBe(1);
    expect(r.stats.verticesAdded).toBe(0);
    checkManifoldOriented(r.mesh);
  });
});

describe("fill accuracy on curved surfaces", () => {
  it("reproduces a cylinder inside a hole (quadric base)", () => {
    const c = removeWhere(makeCylinder(20, 60, 180, 90), (p) => p[0] > 0 && Math.hypot(p[1], p[2] - 30) < 6);
    const loops = findBoundaryLoops(c);
    const hole = loops.findIndex((l) => l.length < 150);
    const nv0 = c.positions.length / 3;
    const withQ = fillHoles(c, { smooth: true, loops: [hole] });
    const noQ = fillHoles(c, { smooth: true, quadricBase: false, loops: [hole] });
    const dq = newVertexStats(withQ.mesh, nv0, (x, y) => Math.hypot(x, y) - 20);
    const dn = newVertexStats(noQ.mesh, nv0, (x, y) => Math.hypot(x, y) - 20);
    expect(dq.max).toBeLessThan(0.02);
    expect(dq.max).toBeLessThan(dn.max);
    checkManifoldOriented(withQ.mesh);
  });
});

describe("defeature", () => {
  it("removes a bump from a sphere and restores the surface", () => {
    const s = makeIcosphere(10, 5);
    const P = s.positions;
    // stamp: raise vertices within 2.5 of the north pole by up to 1
    for (let v = 0; v < P.length / 3; v++) {
      const x = P[3 * v], y = P[3 * v + 1], z = P[3 * v + 2];
      const d = Math.hypot(x, y, z - 10);
      if (d < 2.5) {
        const f = 1 + (1 - d / 2.5) * 0.1;
        P[3 * v] = x * f;
        P[3 * v + 1] = y * f;
        P[3 * v + 2] = z * f;
      }
    }
    const nt = triangleCountOf(s);
    const mask = new Uint8Array(nt);
    for (let t = 0; t < nt; t++) {
      const c = centroid(s, t);
      if (Math.hypot(c[0], c[1], c[2] - 10) < 3.2) mask[t] = 1;
    }
    const { mesh, stats } = defeature(s, mask);
    expect(stats.holesFilled).toBe(1);
    expect(findBoundaryLoops(mesh).length).toBe(0);
    checkManifoldOriented(mesh);
    let maxDev = 0;
    const Q = mesh.positions;
    for (let v = 0; v < Q.length / 3; v++) maxDev = Math.max(maxDev, Math.abs(Math.hypot(Q[3 * v], Q[3 * v + 1], Q[3 * v + 2]) - 10));
    expect(maxDev).toBeLessThan(0.03); // bump was 1.0
  });

  it("does not fill pre-existing holes or a region touching the border", () => {
    const g = removeWhere(makeGrid(20, 20, 20, 20), (c) => (c[0] - 5) ** 2 + (c[1] - 5) ** 2 < 4);
    const nt = triangleCountOf(g);
    const mask = new Uint8Array(nt);
    for (let t = 0; t < nt; t++) {
      const c = centroid(g, t);
      if ((c[0] - 14) ** 2 + (c[1] - 14) ** 2 < 6) mask[t] = 1; // interior feature
      if (c[0] < 1 && c[1] > 15) mask[t] = 1; // touches the border
    }
    const { mesh, stats } = defeature(g, mask);
    expect(stats.holesFilled).toBe(1);
    // outer border + the pre-existing hole remain
    expect(findBoundaryLoops(mesh).length).toBe(2);
  });
});
