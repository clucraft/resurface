import { describe, expect, it } from "vitest";
import type { MeshData } from "../core/types";
import {
  deleteTriangles,
  degenerateTriangleMask,
  flipNormals,
  orientConsistently,
  removeDegenerateTriangles,
  removeSmallComponents,
  weldVertices,
  weldVerticesWithMap,
} from "./repair";
import { buildEdges, computeFaceNormals, findBoundaryLoops } from "./topology";
import { makeGrid, makeIcosphere, seededRandom } from "./util/generate";

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

/** Unweld: every triangle gets its own 3 vertices. */
function soup(m: MeshData, jitter = 0, seed = 3): MeshData {
  const nt = m.indices.length / 3;
  const rnd = seededRandom(seed);
  const positions = new Float32Array(nt * 9);
  const indices = new Uint32Array(nt * 3);
  for (let c = 0; c < nt * 3; c++) {
    const v = m.indices[c];
    for (let k = 0; k < 3; k++) positions[3 * c + k] = m.positions[3 * v + k] + (rnd() * 2 - 1) * jitter;
    indices[c] = c;
  }
  return { positions, indices };
}

/** Every directed edge appears at most once and every undirected interior edge in both directions. */
function isConsistentlyOriented(m: MeshData): boolean {
  const seen = new Set<string>();
  for (let t = 0; t < m.indices.length / 3; t++)
    for (let k = 0; k < 3; k++) {
      const key = `${m.indices[3 * t + k]}-${m.indices[3 * t + ((k + 1) % 3)]}`;
      if (seen.has(key)) return false;
      seen.add(key);
    }
  return true;
}

function signedVolume(m: MeshData): number {
  const P = m.positions, I = m.indices;
  let vol = 0;
  for (let t = 0; t < I.length / 3; t++) {
    const a = 3 * I[3 * t], b = 3 * I[3 * t + 1], c = 3 * I[3 * t + 2];
    vol += P[a] * (P[b + 1] * P[c + 2] - P[b + 2] * P[c + 1]) - P[a + 1] * (P[b] * P[c + 2] - P[b + 2] * P[c]) + P[a + 2] * (P[b] * P[c + 1] - P[b + 1] * P[c]);
  }
  return vol / 6;
}

describe("repair", () => {
  it("removeSmallComponents keeps big shells", () => {
    const m = merge(merge(makeIcosphere(1, 2), makeGrid(1, 1)), makeGrid(3, 3));
    const out = removeSmallComponents(m, 10);
    expect(out.indices.length / 3).toBe(320 + 18);
    expect(out.positions.length / 3).toBe(162 + 16);
    const largest = removeSmallComponents(m, 0, { keepLargestOnly: true });
    expect(largest.indices.length / 3).toBe(320);
  });

  it("removeDegenerateTriangles removes repeated indices, zero area and duplicates", () => {
    const g = makeGrid(2, 2);
    const extra = Uint32Array.of(
      0, 0, 1, // repeated index
      0, 1, 2, // collinear (zero area): vertices 0,1,2 lie on the x axis
      g.indices[0], g.indices[1], g.indices[2], // exact duplicate of triangle 0
      g.indices[3], g.indices[5], g.indices[4], // duplicate of triangle 1 with flipped winding
    );
    const idx = new Uint32Array(g.indices.length + extra.length);
    idx.set(g.indices);
    idx.set(extra, g.indices.length);
    const m: MeshData = { positions: g.positions, indices: idx };
    const mask = degenerateTriangleMask(m);
    expect(Array.from(mask)).toEqual([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]);
    const out = removeDegenerateTriangles(m);
    expect(out.indices.length / 3).toBe(8);
    expect(Array.from(out.indices)).toEqual(Array.from(g.indices));
  });

  it("weldVertices (exact) restores grid topology from a triangle soup", () => {
    const g = makeGrid(6, 5);
    const s = soup(g);
    const w = weldVertices(s, 0);
    expect(w.positions.length / 3).toBe(7 * 6);
    expect(w.indices.length).toBe(g.indices.length);
    expect(findBoundaryLoops(w).length).toBe(1);
  });

  it("weldVertices with tolerance merges jittered copies and drops collapsed triangles", () => {
    const g = makeGrid(6, 5, 6, 5);
    const s = soup(g, 1e-4);
    const { mesh: w, vertexMap } = weldVerticesWithMap(s, 1e-3);
    expect(w.positions.length / 3).toBe(42);
    expect(w.indices.length).toBe(g.indices.length);
    expect(vertexMap.length).toBe(s.positions.length / 3);
    // tolerance bigger than the cells collapses everything
    const all = weldVertices(s, 100);
    expect(all.indices.length).toBe(0);
  });

  it("flipNormals reverses winding and normals", () => {
    const g = makeGrid(2, 2);
    g.normals = new Float32Array((g.positions.length / 3) * 3).fill(0).map((_, i) => (i % 3 === 2 ? 1 : 0));
    const f = flipNormals(g);
    const fn = computeFaceNormals(f);
    for (let t = 0; t < fn.length / 3; t++) expect(fn[3 * t + 2]).toBeCloseTo(-1, 6);
    expect(f.normals![2]).toBe(-1);
  });

  it("orientConsistently repairs random flips on a closed sphere and points normals outward", () => {
    const s = makeIcosphere(5, 3);
    const rnd = seededRandom(7);
    const I = s.indices;
    for (let t = 0; t < I.length / 3; t++)
      if (rnd() < 0.4) {
        const tmp = I[3 * t + 1];
        I[3 * t + 1] = I[3 * t + 2];
        I[3 * t + 2] = tmp;
      }
    expect(isConsistentlyOriented(s)).toBe(false);
    const o = orientConsistently(s);
    expect(isConsistentlyOriented(o)).toBe(true);
    expect(signedVolume(o)).toBeGreaterThan(0);
    // fully inverted closed sphere gets turned outward
    const inv = orientConsistently(flipNormals(makeIcosphere(5, 2)));
    expect(signedVolume(inv)).toBeGreaterThan(0);
  });

  it("orientConsistently on open patches keeps the majority orientation", () => {
    const g = makeGrid(8, 8);
    const I = g.indices;
    for (let t = 0; t < 10; t++) {
      const tmp = I[3 * t + 1];
      I[3 * t + 1] = I[3 * t + 2];
      I[3 * t + 2] = tmp;
    }
    const o = orientConsistently(g);
    expect(isConsistentlyOriented(o)).toBe(true);
    const fn = computeFaceNormals(o);
    for (let t = 0; t < fn.length / 3; t++) expect(fn[3 * t + 2]).toBeCloseTo(1, 6);
  });

  it("orientConsistently survives non-manifold input", () => {
    const g = makeGrid(3, 3);
    const nv = g.positions.length / 3;
    const pos = new Float32Array(g.positions.length + 3);
    pos.set(g.positions);
    pos.set([1, 1, 1], g.positions.length);
    const idx = new Uint32Array(g.indices.length + 6);
    idx.set(g.indices);
    idx.set([g.indices[0], g.indices[2], nv, g.indices[0], g.indices[0], nv], g.indices.length);
    const o = orientConsistently({ positions: pos, indices: idx });
    expect(o.indices.length).toBe(idx.length);
    expect(buildEdges(o).edgeCount).toBeGreaterThan(0);
  });

  it("deleteTriangles", () => {
    const g = makeGrid(2, 2);
    const mask = new Uint8Array(8);
    mask[0] = mask[1] = 1;
    const d = deleteTriangles(g, mask);
    expect(d.indices.length / 3).toBe(6);
    expect(d.positions.length / 3).toBe(8);
  });
});
