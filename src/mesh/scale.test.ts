/**
 * Scale smoke test: core linear-time operations on a ~2M-triangle mesh, with timings printed
 * so they can be extrapolated to the 20M-triangle target (×10, all ops are O(n) / O(n log n)).
 * Run with:  npx vitest run src/mesh/scale.test.ts --silent=false
 */
import { describe, expect, it } from "vitest";
import type { MeshData } from "../core/types";
import { clipByPlane } from "./clip";
import { vertexCurvature } from "./curvature";
import { decimate, decimateClustering, decimateQEM } from "./decimate";
import { fillHoles } from "./holes";
import { orientConsistently, removeDegenerateTriangles, removeSmallComponents, weldVertices } from "./repair";
import { sectionMesh } from "./section";
import { selectConnected, smartSelect } from "./select";
import { smoothTaubin } from "./smooth";
import { MeshSpatialIndex, buildChunks, reorderTriangles, spatialOrderTriangles } from "./spatial";
import {
  buildEdges,
  buildTriangleNeighbors,
  buildVertexTriangles,
  compactMesh,
  computeVertexNormals,
  connectedComponents,
  findBoundaryLoops,
  triangleCountOf,
} from "./topology";
import { makeGrid } from "./util/generate";

const N = 1000; // 1000 × 1000 cells → 2,000,000 triangles, 1,002,001 vertices

/** linear = cost proportional to mesh size (extrapolated ×10); otherwise per query / per region. */
function timed<T>(label: string, fn: () => T, rows: string[], linear = true): T {
  const t0 = performance.now();
  const r = fn();
  const ms = performance.now() - t0;
  const extra = linear ? `(→ 20M: ~${((ms * 10) / 1000).toFixed(1)} s)` : "(not size-proportional)";
  rows.push(`${label.padEnd(44)} ${ms.toFixed(0).padStart(7)} ms   ${extra}`);
  return r;
}

describe("scale smoke test (2M triangles)", () => {
  it("runs the core operations in linear time", { timeout: 600_000 }, () => {
    const rows: string[] = [];
    const mesh = timed("generate grid", () => makeGrid(N, N, 1000, 1000, (x, y) => 5 * Math.sin(x * 0.02) * Math.cos(y * 0.03)), rows);
    const nt = triangleCountOf(mesh);
    expect(nt).toBe(2_000_000);

    const normals = timed("computeVertexNormals", () => computeVertexNormals(mesh), rows);
    mesh.normals = normals;
    timed("buildVertexTriangles (CSR)", () => buildVertexTriangles(mesh), rows);
    const nbr = timed("buildTriangleNeighbors", () => buildTriangleNeighbors(mesh), rows);
    const topo = timed("buildEdges", () => buildEdges(mesh), rows);
    expect(topo.edgeCount).toBe(3 * N * N + 2 * N);
    const loops = timed("findBoundaryLoops", () => findBoundaryLoops(mesh, nbr), rows);
    expect(loops.length).toBe(1);
    const cc = timed("connectedComponents", () => connectedComponents(mesh), rows);
    expect(cc.count).toBe(1);
    const keep = new Uint8Array(nt).fill(1);
    for (let t = 0; t < nt; t += 7) keep[t] = 0;
    timed("compactMesh (drop 1/7 of triangles)", () => compactMesh(mesh, keep), rows);
    timed("removeDegenerateTriangles", () => removeDegenerateTriangles(mesh), rows);
    timed("removeSmallComponents", () => removeSmallComponents(mesh, 100), rows);
    timed("orientConsistently", () => orientConsistently(mesh, { neighbors: nbr }), rows);
    timed("weldVertices (exact)", () => weldVertices(mesh, 0), rows);
    timed("weldVertices (tolerance 0.01)", () => weldVertices(mesh, 0.01), rows);
    timed("smoothTaubin (1 iteration = 2 passes)", () => smoothTaubin(mesh, { iterations: 1 }), rows);
    timed("vertexCurvature", () => vertexCurvature(mesh), rows);
    const clipped = timed("clipByPlane (cut)", () => clipByPlane(mesh, { kind: "plane", origin: [500.3, 0, 0], normal: [1, 0.2, 0] }, "positive", true), rows);
    expect(triangleCountOf(clipped)).toBeGreaterThan(nt * 0.3);
    const sec = timed("sectionMesh (full scan)", () => sectionMesh(mesh, { kind: "plane", origin: [0, 400.5, 0], normal: [0, 1, 0] }), rows);
    expect(sec.length).toBe(1);
    const perm = timed("spatialOrderTriangles (Morton + radix)", () => spatialOrderTriangles(mesh), rows);
    const ordered = timed("reorderTriangles (+ vertices)", () => reorderTriangles(mesh, perm, { reorderVertices: true }), rows);
    timed("buildChunks (64k)", () => buildChunks(ordered, 65536), rows);
    const index = timed("MeshSpatialIndex build (BVH, indirect)", () => new MeshSpatialIndex(mesh), rows);
    const cand = index.trianglesIntersectingPlane({ kind: "plane", origin: [0, 400.5, 0], normal: [0, 1, 0] });
    timed("sectionMesh (BVH candidates)", () => sectionMesh(mesh, { kind: "plane", origin: [0, 400.5, 0], normal: [0, 1, 0] }, { candidates: cand }), rows, false);
    timed("10k closestPoint queries", () => {
      for (let i = 0; i < 10000; i++) index.closestPoint([(i * 37) % 1000, (i * 91) % 1000, 10]);
    }, rows, false);
    timed("10k raycasts", () => {
      for (let i = 0; i < 10000; i++) index.raycast([(i * 37) % 1000 + 0.3, (i * 91) % 1000 + 0.3, 50], [0, 0, -1]);
    }, rows, false);
    index.dispose();
    timed("selectConnected (whole mesh)", () => selectConnected(mesh, [0], { neighbors: nbr }), rows);
    timed("smartSelect (200k-triangle region)", () => smartSelect(mesh, 1000, { neighbors: nbr, angleTolDeg: 10, maxTriangles: 200_000 }), rows, false);
    const holed = compactMesh(mesh, (() => {
      const k = new Uint8Array(nt).fill(1);
      for (let j = 480; j < 520; j++) for (let i = 480; i < 520; i++) k[2 * (j * N + i)] = k[2 * (j * N + i) + 1] = 0;
      return k;
    })());
    const holeLoops = findBoundaryLoops(holed);
    const hole = holeLoops.findIndex((l) => l.length === 160);
    const filled = timed("fillHoles (1 hole, 160 edges, smooth)", () => fillHoles(holed, { smooth: true, loops: [hole] }), rows);
    expect(filled.stats.holesFilled).toBe(1);
    const clustered = timed("decimateClustering → ~200k", () => decimateClustering(mesh, { targetTriangles: 200_000 }), rows);
    expect(triangleCountOf(clustered)).toBeGreaterThan(100_000);
    const q = timed("decimateQEM 2M → 200k", () => decimateQEM(mesh, { targetTriangles: 200_000 }), rows);
    expect(triangleCountOf(q)).toBeLessThanOrEqual(200_000);
    // the chained path used above 8M triangles (here forced with a lower threshold)
    const chained = timed("decimate (cluster → 500k, QEM → 200k)", () => decimate(mesh, { targetTriangles: 200_000, clusterAbove: 1_000_000 }), rows);
    expect(triangleCountOf(chained)).toBeLessThanOrEqual(200_000);
    console.log(`\n[mesh scale] ${nt.toLocaleString()} triangles, ${(mesh.positions.length / 3).toLocaleString()} vertices\n` + rows.join("\n"));
  });
});

export type { MeshData };
