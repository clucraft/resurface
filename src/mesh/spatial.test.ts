import { describe, expect, it } from "vitest";
import { MeshSpatialIndex, buildChunks, pointTriangleDistSq, reorderTriangles, spatialOrderTriangles } from "./spatial";
import { meshArea, triangleCountOf } from "./topology";
import { makeGrid, makeIcosphere, seededRandom } from "./util/generate";

describe("MeshSpatialIndex", () => {
  const s = makeIcosphere(10, 4);
  const before = s.indices.slice();
  const idx = new MeshSpatialIndex(s);

  it("does not reorder the mesh triangles (indirect BVH)", () => {
    expect(Array.from(s.indices)).toEqual(Array.from(before));
  });

  it("raycast hits the sphere", () => {
    const hit = idx.raycast([0, 0, 50], [0, 0, -1]);
    expect(hit).not.toBeNull();
    expect(hit!.distance).toBeGreaterThan(39.9);
    expect(hit!.distance).toBeLessThanOrEqual(40.0001);
    expect(hit!.point[2]).toBeGreaterThan(9.9);
    expect(hit!.normal[2]).toBeGreaterThan(0.95);
    expect(hit!.faceIndex).toBeGreaterThanOrEqual(0);
    expect(idx.raycast([0, 0, 50], [0, 0, -1], 30)).toBeNull();
    expect(idx.raycast([0, 0, 50], [0, 0, 1])).toBeNull();
    // from inside (back faces count)
    const inside = idx.raycast([0, 0, 0], [1, 0, 0]);
    expect(inside!.distance).toBeGreaterThan(9.9);
  });

  it("closestPoint matches brute force", () => {
    const rnd = seededRandom(2);
    const P = s.positions, I = s.indices;
    const nt = triangleCountOf(s);
    for (let i = 0; i < 20; i++) {
      const p: [number, number, number] = [(rnd() - 0.5) * 30, (rnd() - 0.5) * 30, (rnd() - 0.5) * 30];
      let best = Infinity, bestT = -1;
      for (let t = 0; t < nt; t++) {
        const d = pointTriangleDistSq(p[0], p[1], p[2], P, I[3 * t], I[3 * t + 1], I[3 * t + 2]);
        if (d < best) {
          best = d;
          bestT = t;
        }
      }
      const hit = idx.closestPoint(p)!;
      expect(hit.distance).toBeCloseTo(Math.sqrt(best), 4);
      expect(pointTriangleDistSq(p[0], p[1], p[2], P, I[3 * hit.faceIndex!], I[3 * hit.faceIndex! + 1], I[3 * hit.faceIndex! + 2])).toBeCloseTo(best, 3);
      void bestT;
    }
    expect(idx.closestPoint([0, 0, 0], 5)).toBeNull();
    const q = idx.asQuery();
    expect(q([0, 0, 20])!.distance).toBeCloseTo(10, 1);
  });

  it("sphere/box/plane queries match brute force", () => {
    const P = s.positions, I = s.indices;
    const nt = triangleCountOf(s);
    const c: [number, number, number] = [3, 4, 8];
    const r = 3;
    const got = new Set(idx.trianglesInSphere(c, r));
    let expected = 0;
    for (let t = 0; t < nt; t++) {
      const inside = pointTriangleDistSq(c[0], c[1], c[2], P, I[3 * t], I[3 * t + 1], I[3 * t + 2]) <= r * r;
      if (inside) {
        expected++;
        expect(got.has(t)).toBe(true);
      }
    }
    expect(got.size).toBe(expected);

    const inBox = idx.trianglesInBox([-2, -2, 5], [2, 2, 11]);
    let expectedBox = 0;
    for (let t = 0; t < nt; t++) {
      let anyInside = false;
      for (let k = 0; k < 3; k++) {
        const v = I[3 * t + k];
        const x = P[3 * v], y = P[3 * v + 1], z = P[3 * v + 2];
        if (x >= -2 && x <= 2 && y >= -2 && y <= 2 && z >= 5 && z <= 11) anyInside = true;
      }
      if (anyInside) expectedBox++;
    }
    expect(inBox.length).toBeGreaterThanOrEqual(expectedBox);
    expect(inBox.length).toBeLessThan(expectedBox + 60);

    const cand = idx.trianglesIntersectingPlane({ kind: "plane", origin: [0, 0, 1], normal: [0, 0, 1] });
    for (const t of cand) {
      const zs = [0, 1, 2].map((k) => P[3 * I[3 * t + k] + 2] - 1);
      expect(Math.min(...zs)).toBeLessThanOrEqual(0);
      expect(Math.max(...zs)).toBeGreaterThanOrEqual(0);
    }
    expect(cand.length).toBeGreaterThan(50);
    expect(idx.bounds().max[2]).toBeCloseTo(10, 4);
  });

  it("refit after moving vertices; dispose", () => {
    const g = makeGrid(10, 10, 10, 10);
    const gi = new MeshSpatialIndex(g);
    expect(gi.raycast([5, 5, 10], [0, 0, -1])!.distance).toBeCloseTo(10, 5);
    for (let v = 0; v < g.positions.length / 3; v++) g.positions[3 * v + 2] = 2;
    gi.refit();
    expect(gi.raycast([5, 5, 10], [0, 0, -1])!.distance).toBeCloseTo(8, 5);
    gi.dispose();
    expect(() => gi.raycast([0, 0, 0], [0, 0, 1])).toThrow();
  });
});

describe("spatial ordering & chunks", () => {
  it("Morton order is a permutation that improves locality; chunks cover everything", () => {
    const g = makeGrid(64, 64, 64, 64);
    // scramble triangle order first
    const nt = triangleCountOf(g);
    const rnd = seededRandom(9);
    const scramble = new Uint32Array(nt).map((_, i) => i);
    for (let i = nt - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      const t = scramble[i];
      scramble[i] = scramble[j];
      scramble[j] = t;
    }
    const mixed = reorderTriangles(g, scramble);
    const perm = spatialOrderTriangles(mixed);
    const seen = new Uint8Array(nt);
    for (const p of perm) seen[p]++;
    expect(seen.every((x) => x === 1)).toBe(true);
    const ordered = reorderTriangles(mixed, perm, { reorderVertices: true });
    expect(meshArea(ordered)).toBeCloseTo(meshArea(g), 3);
    const volume = (chunks: ReturnType<typeof buildChunks>) =>
      chunks.reduce((s, c) => s + (c.bounds.max[0] - c.bounds.min[0]) * (c.bounds.max[1] - c.bounds.min[1]), 0);
    const a = buildChunks(mixed, 256), b = buildChunks(ordered, 256);
    expect(a.length).toBe(Math.ceil(nt / 256));
    expect(a.reduce((s, c) => s + c.triCount, 0)).toBe(nt);
    expect(volume(b)).toBeLessThan(volume(a) / 20);
    // vertex reordering: the first chunk's vertices are a contiguous low range
    let maxV = 0;
    for (let c = 0; c < 256 * 3; c++) maxV = Math.max(maxV, ordered.indices[c]);
    expect(maxV).toBeLessThan(256 * 3);
  });
});

describe("MeshSpatialIndex serialization", () => {
  it("rebuilds an identical index from serialized buffers (worker hand-off)", () => {
    const mesh = makeIcosphere(5, 5);
    const built = new MeshSpatialIndex(mesh);
    const copy = new MeshSpatialIndex(mesh, { serialized: built.serialize() });
    const rnd = seededRandom(3);
    for (let i = 0; i < 50; i++) {
      const p: [number, number, number] = [(rnd() - 0.5) * 30, (rnd() - 0.5) * 30, (rnd() - 0.5) * 30];
      const a = built.closestPoint(p)!, b = copy.closestPoint(p)!;
      expect(b.faceIndex).toBe(a.faceIndex);
      expect(b.distance).toBeCloseTo(a.distance, 9);
      const dir: [number, number, number] = [-p[0], -p[1], -p[2]];
      expect(copy.raycast(p, dir)?.faceIndex).toBe(built.raycast(p, dir)?.faceIndex);
    }
  });
});

