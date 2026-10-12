import { describe, expect, it } from "vitest";
import { GridIndex, createPointCloudQuery, estimateNormals } from "./pointQuery";
import { makeRand, planePatch } from "./testing/synthetic";
import { reflectionMatrix, reflectPoint, rigidDifference, rotationAngle, transformPoints, transformPrimitive } from "./transform";
import { rotationAxisAngle } from "../core/vec";

describe("GridIndex", () => {
  const r = makeRand(1);
  const n = 5000;
  const pts = new Float64Array(3 * n);
  for (let i = 0; i < 3 * n; i++) pts[i] = r.gauss() * (i % 3 === 2 ? 0.1 : 3);
  const grid = new GridIndex(pts, n);
  const brute = (x: number, y: number, z: number) => {
    let best = -1, bd = Infinity;
    for (let i = 0; i < n; i++) {
      const d = (pts[3 * i] - x) ** 2 + (pts[3 * i + 1] - y) ** 2 + (pts[3 * i + 2] - z) ** 2;
      if (d < bd) {
        bd = d;
        best = i;
      }
    }
    return { best, bd };
  };

  it("nearest matches brute force (inside and far outside)", () => {
    for (let k = 0; k < 200; k++) {
      const q = [r.gauss() * 4, r.gauss() * 4, r.gauss() * (k % 10 === 0 ? 50 : 1)];
      const i = grid.nearest(q[0], q[1], q[2]);
      expect(grid.lastDist2).toBeCloseTo(brute(q[0], q[1], q[2]).bd, 12);
      expect(i).toBeGreaterThanOrEqual(0);
    }
    expect(grid.nearest(100, 100, 100, 1)).toBe(-1);
  });

  it("knn returns the k closest sorted", () => {
    const idx = new Int32Array(8), d2 = new Float64Array(8);
    const q = [0.5, -0.3, 0.02];
    expect(grid.knn(q[0], q[1], q[2], 8, idx, d2)).toBe(8);
    const all = Array.from({ length: n }, (_, i) => (pts[3 * i] - q[0]) ** 2 + (pts[3 * i + 1] - q[1]) ** 2 + (pts[3 * i + 2] - q[2]) ** 2).sort((a, b) => a - b);
    for (let k = 0; k < 8; k++) expect(d2[k]).toBeCloseTo(all[k], 12);
  });
});

describe("point cloud helpers", () => {
  it("estimates normals and answers closest-point queries", () => {
    const c = planePatch({ origin: [0, 0, 0], normal: [0, 0, 1], size: 10, n: 4000, seed: 2 });
    const nr = estimateNormals(c.positions, { k: 12 });
    for (let i = 0; i < 4000; i += 97) expect(Math.abs(nr[3 * i + 2])).toBeCloseTo(1, 4);
    const q = createPointCloudQuery(c.positions, null, { projectToTangentPlane: true });
    const hit = q([0.1, 0.2, 0.5])!;
    expect(hit.distance).toBeCloseTo(0.5, 4);
    expect(Math.abs(hit.normal[2])).toBeCloseTo(1, 4);
    expect(q([0, 0, 5], 1)).toBeNull();
  });
});

describe("transform helpers", () => {
  it("reflection, rigid difference, primitive transform", () => {
    const pl = { kind: "plane" as const, origin: [0, 0, 2] as [number, number, number], normal: [0, 0, 1] as [number, number, number] };
    expect(reflectPoint([1, 1, 5], pl)).toEqual([1, 1, -1]);
    const m = reflectionMatrix(pl);
    expect(Array.from(transformPoints(m, [1, 1, 5]))).toEqual([1, 1, -1]);
    const R = rotationAxisAngle([0, 0, 1], 1e-7);
    expect(rotationAngle(R)).toBeCloseTo(1e-7, 15);
    expect(rigidDifference(R, rotationAxisAngle([0, 0, 1], 0)).angle).toBeCloseTo(1e-7, 15);
    const cyl = transformPrimitive(rotationAxisAngle([1, 0, 0], Math.PI / 2), { kind: "cylinder", origin: [0, 0, 0], axis: [0, 0, 1], radius: 1, hMin: 0, hMax: 1 });
    expect(cyl.axis[1]).toBeCloseTo(-1, 12);
  });
});
