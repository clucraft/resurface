import { describe, expect, it } from "vitest";
import type { MeshData } from "../core/types";
import {
  BOUNDARY,
  NON_MANIFOLD,
  averageEdgeLength,
  boundaryVertexMask,
  buildEdges,
  buildTriangleNeighbors,
  buildVertexTriangles,
  compactMesh,
  compactMeshWithMap,
  computeFaceNormals,
  computeVertexNormals,
  connectedComponents,
  findBoundaryLoops,
  meshArea,
  meshBounds,
} from "./topology";
import { makeGrid, makeIcosphere, makeUVSphere } from "./util/generate";

function merge(a: MeshData, b: MeshData): MeshData {
  const nva = a.positions.length / 3;
  const positions = new Float32Array(a.positions.length + b.positions.length);
  positions.set(a.positions);
  positions.set(b.positions, a.positions.length);
  const indices = new Uint32Array(a.indices.length + b.indices.length);
  indices.set(a.indices);
  for (let i = 0; i < b.indices.length; i++) indices[a.indices.length + i] = b.indices[i] + nva;
  return { positions, indices };
}

describe("normals", () => {
  it("grid normals point +Z", () => {
    const g = makeGrid(4, 3);
    const fn = computeFaceNormals(g);
    for (let t = 0; t < fn.length / 3; t++) expect(fn[3 * t + 2]).toBeCloseTo(1, 6);
    const vn = computeVertexNormals(g);
    for (let v = 0; v < vn.length / 3; v++) expect(vn[3 * v + 2]).toBeCloseTo(1, 6);
  });
  it("sphere vertex normals point outward", () => {
    const s = makeIcosphere(10, 3);
    const vn = computeVertexNormals(s);
    for (let v = 0; v < vn.length / 3; v++) {
      const d = (vn[3 * v] * s.positions[3 * v] + vn[3 * v + 1] * s.positions[3 * v + 1] + vn[3 * v + 2] * s.positions[3 * v + 2]) / 10;
      expect(d).toBeGreaterThan(0.99);
    }
  });
  it("degenerate triangle gets zero face normal", () => {
    const m: MeshData = { positions: Float32Array.of(0, 0, 0, 1, 0, 0, 2, 0, 0), indices: Uint32Array.of(0, 1, 2) };
    expect(Array.from(computeFaceNormals(m))).toEqual([0, 0, 0]);
  });
});

describe("adjacency", () => {
  it("vertex→triangle CSR", () => {
    const g = makeGrid(2, 2);
    const vt = buildVertexTriangles(g);
    expect(vt.offsets.length).toBe(10);
    expect(vt.triangles.length).toBe(24);
    // centre vertex (index 4) touches 6 triangles
    expect(vt.offsets[5] - vt.offsets[4]).toBe(6);
    for (let v = 0; v < 9; v++)
      for (let j = vt.offsets[v]; j < vt.offsets[v + 1]; j++) {
        const t = vt.triangles[j];
        expect([g.indices[3 * t], g.indices[3 * t + 1], g.indices[3 * t + 2]]).toContain(v);
      }
  });

  it("closed sphere: E = 3F/2, all edges manifold, neighbours symmetric", () => {
    const s = makeUVSphere(5, 24, 12);
    const nt = s.indices.length / 3;
    const topo = buildEdges(s);
    expect(topo.edgeCount).toBe((3 * nt) / 2);
    for (let e = 0; e < topo.edgeCount; e++) {
      expect(topo.edgeFaceCount[e]).toBe(2);
      expect(topo.edgeVertices[2 * e]).toBeLessThan(topo.edgeVertices[2 * e + 1]);
    }
    const n = topo.triangleNeighbors;
    for (let t = 0; t < nt; t++)
      for (let k = 0; k < 3; k++) {
        const u = n[3 * t + k];
        expect(u).toBeGreaterThanOrEqual(0);
        expect([n[3 * u], n[3 * u + 1], n[3 * u + 2]]).toContain(t);
        const e = topo.triangleEdges[3 * t + k];
        const a = s.indices[3 * t + k], b = s.indices[3 * t + ((k + 1) % 3)];
        expect([topo.edgeVertices[2 * e], topo.edgeVertices[2 * e + 1]].sort()).toEqual([a, b].sort());
      }
    expect(Array.from(buildTriangleNeighbors(s))).toEqual(Array.from(n));
    // Euler characteristic of a sphere
    expect(s.positions.length / 3 - topo.edgeCount + nt).toBe(2);
  });

  it("grid boundary and non-manifold edges", () => {
    const g = makeGrid(3, 3);
    const n = buildTriangleNeighbors(g);
    let boundary = 0;
    for (let i = 0; i < n.length; i++) if (n[i] === BOUNDARY) boundary++;
    expect(boundary).toBe(12);
    // add a fin triangle on an interior edge → that edge becomes non-manifold
    const nv = g.positions.length / 3;
    const pos = new Float32Array(g.positions.length + 3);
    pos.set(g.positions);
    pos.set([1, 1, 1], g.positions.length);
    const a = g.indices[0], b = g.indices[2]; // diagonal of the first cell (interior edge)
    const idx = new Uint32Array(g.indices.length + 3);
    idx.set(g.indices);
    idx.set([a, b, nv], g.indices.length);
    const m: MeshData = { positions: pos, indices: idx };
    const topo = buildEdges(m);
    let nonManifold = 0;
    for (let e = 0; e < topo.edgeCount; e++) if (topo.edgeFaceCount[e] === 3) nonManifold++;
    expect(nonManifold).toBe(1);
    expect(topo.triangleNeighbors[0 * 3 + 2]).toBe(NON_MANIFOLD);
  });

  it("degenerate triangles do not crash adjacency", () => {
    const m: MeshData = { positions: makeGrid(1, 1).positions, indices: Uint32Array.of(0, 1, 3, 0, 3, 2, 1, 1, 3, 0, 0, 0) };
    const topo = buildEdges(m);
    expect(topo.edgeCount).toBe(5);
    expect(topo.triangleNeighbors[3 * 3]).toBe(NON_MANIFOLD);
  });
});

describe("boundary loops", () => {
  it("grid has one loop of the outer border, oriented with the triangles", () => {
    const g = makeGrid(4, 3);
    const loops = findBoundaryLoops(g);
    expect(loops.length).toBe(1);
    expect(loops[0].length).toBe(2 * (4 + 3));
    // consecutive loop vertices must be a directed edge of some triangle
    const directed = new Set<string>();
    for (let t = 0; t < g.indices.length / 3; t++)
      for (let k = 0; k < 3; k++) directed.add(`${g.indices[3 * t + k]}-${g.indices[3 * t + ((k + 1) % 3)]}`);
    const L = loops[0];
    for (let i = 0; i < L.length; i++) expect(directed.has(`${L[i]}-${L[(i + 1) % L.length]}`)).toBe(true);
    const bm = boundaryVertexMask(g);
    let nb = 0;
    for (const x of bm) nb += x;
    expect(nb).toBe(14);
  });

  it("grid with two holes → three loops", () => {
    const g = makeGrid(10, 10);
    const nt = g.indices.length / 3;
    const keep = new Uint8Array(nt).fill(1);
    // remove cell (2,2) and cells (6..7, 6)
    const cell = (i: number, j: number) => 2 * (j * 10 + i);
    for (const [i, j] of [[2, 2], [6, 6], [7, 6]]) keep[cell(i, j)] = keep[cell(i, j) + 1] = 0;
    const m = compactMesh(g, keep);
    const loops = findBoundaryLoops(m).map((l) => l.length).sort((a, b) => a - b);
    expect(loops).toEqual([4, 6, 40]);
  });

  it("splits loops at bow-tie vertices", () => {
    // two triangles' fans touching at one vertex: remove two diagonal cells sharing a corner
    const g = makeGrid(4, 4);
    const nt = g.indices.length / 3;
    const keep = new Uint8Array(nt).fill(1);
    const cell = (i: number, j: number) => 2 * (j * 4 + i);
    for (const [i, j] of [[1, 1], [2, 2]]) keep[cell(i, j)] = keep[cell(i, j) + 1] = 0;
    const m = compactMesh(g, keep);
    const loops = findBoundaryLoops(m).map((l) => l.length).sort((a, b) => a - b);
    expect(loops).toEqual([4, 4, 16]);
  });

  it("closed mesh has no loops", () => {
    expect(findBoundaryLoops(makeIcosphere(1, 2)).length).toBe(0);
  });
});

describe("components, bounds, area, compaction", () => {
  it("connected components", () => {
    const a = makeIcosphere(1, 1);
    const b = makeGrid(2, 2);
    const m = merge(merge(a, b), makeGrid(1, 1));
    const c = connectedComponents(m);
    expect(c.count).toBe(3);
    expect(Array.from(c.componentSizes)).toEqual([80, 8, 2]);
    expect(c.triangleComponent[0]).toBe(0);
    expect(c.triangleComponent[80]).toBe(1);
    expect(c.triangleComponent[88]).toBe(2);
  });

  it("bounds/area/edge length", () => {
    const g = makeGrid(10, 5, 20, 10);
    const bb = meshBounds(g);
    expect(bb.min).toEqual([0, 0, 0]);
    expect(bb.max).toEqual([20, 10, 0]);
    expect(meshArea(g)).toBeCloseTo(200, 4);
    const s = makeIcosphere(3, 4);
    expect(meshArea(s)).toBeCloseTo(4 * Math.PI * 9, 0);
    // grid: edges 2 (x), 2 (y), diagonal 2√2 in equal numbers per triangle
    expect(averageEdgeLength(g)).toBeCloseTo((2 + 2 + 2 * Math.SQRT2) / 3, 5);
  });

  it("compactMesh removes unreferenced vertices and carries attributes", () => {
    const g = makeGrid(2, 1);
    g.colors = new Uint8Array((g.positions.length / 3) * 3).map((_, i) => i);
    g.normals = computeVertexNormals(g);
    const keep = Uint8Array.of(1, 1, 0, 0);
    const { mesh, vertexMap, triangleMap } = compactMeshWithMap(g, keep);
    expect(mesh.positions.length / 3).toBe(4);
    expect(mesh.indices.length).toBe(6);
    expect(Array.from(triangleMap)).toEqual([0, 1]);
    expect(vertexMap[2]).toBe(-1);
    for (let v = 0; v < 6; v++) {
      const m2 = vertexMap[v];
      if (m2 < 0) continue;
      expect(mesh.positions[3 * m2]).toBe(g.positions[3 * v]);
      expect(mesh.colors![3 * m2]).toBe(g.colors[3 * v]);
    }
    expect(mesh.normals!.length).toBe(12);
  });
});
