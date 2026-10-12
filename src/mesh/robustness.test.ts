/**
 * Every operation must survive "real scan" garbage: non-manifold edges, duplicate and flipped
 * faces, zero-area and repeated-index triangles, isolated vertices, bow-ties, inconsistent
 * winding, out-of-range indices and SharedArrayBuffer-backed inputs.
 */
import { describe, expect, it } from "vitest";
import type { MeshData } from "../core/types";
import { clipByPlane } from "./clip";
import { vertexCurvature } from "./curvature";
import { decimate, decimateClustering, decimateQEM } from "./decimate";
import { defeature } from "./defeature";
import { fillHoles } from "./holes";
import { deleteTriangles, orientConsistently, removeDegenerateTriangles, removeSmallComponents, weldVertices } from "./repair";
import { sectionMesh } from "./section";
import { growSelection, selectConnected, shrinkSelection, smartSelect } from "./select";
import { smoothTaubin } from "./smooth";
import { MeshSpatialIndex, buildChunks, reorderTriangles, spatialOrderTriangles } from "./spatial";
import { buildEdges, compactMesh, computeVertexNormals, connectedComponents, findBoundaryLoops, triangleCountOf } from "./topology";
import { makeGrid, makeIcosphere, seededRandom } from "./util/generate";

function dirtyMesh(): MeshData {
  const s = makeIcosphere(10, 3);
  const g = makeGrid(6, 6, 6, 6);
  const nvs = s.positions.length / 3;
  const nvg = g.positions.length / 3;
  const positions = new Float32Array((nvs + nvg + 3) * 3);
  positions.set(s.positions);
  positions.set(g.positions, nvs * 3);
  positions.set([50, 50, 50, 51, 50, 50, 52, 50, 50], (nvs + nvg) * 3); // collinear + isolated
  const idx: number[] = Array.from(s.indices);
  for (const i of g.indices) idx.push(i + nvs);
  const rnd = seededRandom(4);
  // random winding flips
  for (let t = 0; t < idx.length / 3; t += 1) if (rnd() < 0.1) [idx[3 * t + 1], idx[3 * t + 2]] = [idx[3 * t + 2], idx[3 * t + 1]];
  idx.push(s.indices[0], s.indices[1], s.indices[2]); // duplicate
  idx.push(s.indices[3], s.indices[5], s.indices[4]); // flipped duplicate
  idx.push(s.indices[0], s.indices[1], nvs + 3); // fin → non-manifold edge
  idx.push(0, 0, 1); // repeated index
  idx.push(nvs + nvg, nvs + nvg + 1, nvs + nvg + 2); // zero area
  idx.push(nvs + 3, nvs + 10, nvs + 11); // flipped duplicate of grid triangle (3, 11, 10)
  // carve a hole in the sphere
  const tris = idx.length / 3;
  const keep: number[] = [];
  for (let t = 0; t < tris; t++) {
    const a = idx[3 * t];
    if (a < nvs && positions[3 * a + 2] > 9.3 && t < s.indices.length / 3) continue;
    keep.push(idx[3 * t], idx[3 * t + 1], idx[3 * t + 2]);
  }
  return { positions, indices: Uint32Array.from(keep) };
}

function toShared(m: MeshData): MeshData {
  const sp = new Float32Array(new SharedArrayBuffer(m.positions.byteLength));
  sp.set(m.positions);
  const si = new Uint32Array(new SharedArrayBuffer(m.indices.byteLength));
  si.set(m.indices);
  return { positions: sp, indices: si };
}

describe("robustness on dirty, shared-memory input", () => {
  it("all operations run without throwing and produce sane output", () => {
    const m = toShared(dirtyMesh());
    const nt = triangleCountOf(m);
    expect(() => buildEdges(m)).not.toThrow();
    expect(findBoundaryLoops(m).length).toBeGreaterThan(0);
    expect(connectedComponents(m).count).toBeGreaterThanOrEqual(2);
    const n = computeVertexNormals(m);
    expect(n.length).toBe(m.positions.length);
    const clean = removeDegenerateTriangles(m);
    expect(triangleCountOf(clean)).toBe(nt - 5); // 3 duplicates + repeated index + zero area
    expect(triangleCountOf(removeSmallComponents(m, 50))).toBeGreaterThan(0);
    const o = orientConsistently(m);
    expect(triangleCountOf(o)).toBe(nt);
    expect(triangleCountOf(weldVertices(m, 0.001))).toBeGreaterThan(0);
    expect(triangleCountOf(smoothTaubin(m, { iterations: 2 }))).toBe(nt);
    expect(vertexCurvature(m).every(Number.isFinite)).toBe(true);
    expect(triangleCountOf(clipByPlane(m, { kind: "plane", origin: [0, 0, 0], normal: [0, 0, 1] }, "positive", true))).toBeGreaterThan(0);
    expect(sectionMesh(m, { kind: "plane", origin: [0, 0, 0.5], normal: [0, 0, 1] }).length).toBeGreaterThan(0);
    const filled = fillHoles(m, { smooth: true });
    expect(filled.stats.holesFilled).toBeGreaterThan(0);
    const mask = new Uint8Array(nt);
    for (let t = 0; t < 30; t++) mask[t] = 1;
    expect(triangleCountOf(defeature(m, mask).mesh)).toBeGreaterThan(0);
    expect(triangleCountOf(deleteTriangles(m, mask))).toBe(nt - 30);
    expect(triangleCountOf(decimateQEM(m, { targetTriangles: 200 }))).toBeLessThanOrEqual(200);
    expect(triangleCountOf(decimateClustering(m, { targetTriangles: 200 }))).toBeGreaterThan(0);
    expect(triangleCountOf(decimate(m, { targetTriangles: 300 }))).toBeLessThanOrEqual(300);
    expect(selectConnected(m, [0]).length).toBe(nt);
    expect(growSelection(m, mask, 2).length).toBe(nt);
    expect(shrinkSelection(m, mask, 1).length).toBe(nt);
    expect(smartSelect(m, 5).length).toBe(nt);
    const ordered = reorderTriangles(m, spatialOrderTriangles(m), { reorderVertices: true });
    expect(triangleCountOf(ordered)).toBe(nt);
    expect(buildChunks(ordered, 100).reduce((s, c) => s + c.triCount, 0)).toBe(nt);
    const idx = new MeshSpatialIndex(m);
    expect(idx.raycast([0, 0, -50], [0, 0, 1])).not.toBeNull();
    idx.dispose();
  });

  it("empty and tiny meshes", () => {
    const empty: MeshData = { positions: new Float32Array(0), indices: new Uint32Array(0) };
    expect(findBoundaryLoops(empty)).toEqual([]);
    expect(buildEdges(empty).edgeCount).toBe(0);
    expect(connectedComponents(empty).count).toBe(0);
    expect(triangleCountOf(compactMesh(empty))).toBe(0);
    expect(triangleCountOf(decimateQEM(empty, { targetTriangles: 0 }))).toBe(0);
    expect(sectionMesh(empty, { kind: "plane", origin: [0, 0, 0], normal: [0, 0, 1] })).toEqual([]);
    expect(fillHoles(empty, { smooth: true }).stats.holesFilled).toBe(0);
    expect(spatialOrderTriangles(empty).length).toBe(0);
    const one: MeshData = { positions: Float32Array.of(0, 0, 0, 1, 0, 0, 0, 1, 0), indices: Uint32Array.of(0, 1, 2) };
    expect(findBoundaryLoops(one).length).toBe(1);
    expect(fillHoles(one, { smooth: true }).stats.holesFilled).toBe(1);
    expect(triangleCountOf(decimateQEM(one, { targetTriangles: 1 }))).toBe(1);
    expect(triangleCountOf(clipByPlane(one, { kind: "plane", origin: [0.5, 0, 0], normal: [1, 0, 0] }, "negative", true))).toBe(2);
  });
});
