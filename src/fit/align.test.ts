import { describe, expect, it } from "vitest";
import type { Mat4, Vec3 } from "../core/types";
import { identity4, invertRigid, mul4, normalize, transformDir, transformPoint } from "../core/vec";
import { alignToWorld, globalAlign, icp, kabsch, nPointAlign } from "./align";
import { createPointCloudQuery } from "./pointQuery";
import { boxQuery, boxSurface, heightField, makeRand, randomRigid, transformCloud } from "./testing/synthetic";
import { rigidDifference } from "./transform";

const det3 = (m: Mat4) =>
  m[0] * (m[5] * m[10] - m[9] * m[6]) - m[4] * (m[1] * m[10] - m[9] * m[2]) + m[8] * (m[1] * m[6] - m[5] * m[2]);

const surf = (x: number, y: number) => 0.3 * Math.sin(1.3 * x) * Math.cos(0.7 * y) + 0.05 * x * x - 0.03 * x * y + 0.1 * Math.sin(2 * y) + 0.02 * y * y * y;

describe("kabsch", () => {
  it("recovers a random rigid transform", () => {
    const r = makeRand(1);
    const T = randomRigid(r, Math.PI, [10, -20, 5]);
    const src: Vec3[] = Array.from({ length: 50 }, () => [r.gauss() * 10, r.gauss() * 5, r.gauss() * 2] as Vec3);
    const dst = src.map((p) => transformPoint(T, p));
    const fit = kabsch(src, dst);
    const d = rigidDifference(fit.matrix, T);
    expect(d.angle).toBeLessThan(1e-9);
    expect(d.translation).toBeLessThan(1e-8);
    expect(fit.rms).toBeLessThan(1e-8);
  });

  it("weighted fit with noise and never returns a reflection", () => {
    const r = makeRand(2);
    const src: Vec3[] = Array.from({ length: 30 }, () => [r.gauss(), r.gauss(), r.gauss()] as Vec3);
    // mirrored target (det -1): best proper rotation, not a reflection
    const dst = src.map((p) => [-p[0], p[1], p[2]] as Vec3);
    const fit = kabsch(src, dst, src.map(() => 1));
    expect(det3(fit.matrix)).toBeCloseTo(1, 9);
    expect(fit.rms).toBeGreaterThan(0.1);
  });

  it("degenerate inputs stay close to identity", () => {
    const one = kabsch([[1, 2, 3]], [[2, 2, 3]]);
    expect(rigidDifference(one.matrix, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1]).angle).toBeLessThan(1e-12);
    const two = kabsch(
      [
        [0, 0, 0],
        [1, 0, 0],
      ],
      [
        [0, 0, 0],
        [1, 0, 0],
      ],
    );
    expect(rigidDifference(two.matrix, identity4()).angle).toBeLessThan(1e-9);
    expect(() => kabsch([[0, 0, 0]], [])).toThrow();
  });
});

describe("nPointAlign", () => {
  it("aligns picked pairs and reports per-pair errors", () => {
    const T = randomRigid(makeRand(3), 2, [1, 2, 3]);
    const moving: Vec3[] = [
      [0, 0, 0],
      [10, 0, 0],
      [0, 5, 0],
      [3, 3, 4],
    ];
    const pairs = moving.map((m, i) => {
      const f = transformPoint(T, m);
      return { moving: m, fixed: [f[0] + (i === 3 ? 0.1 : 0), f[1], f[2]] as Vec3 };
    });
    const r = nPointAlign(pairs);
    expect(r.perPairErrors.length).toBe(4);
    expect(Math.max(...r.perPairErrors)).toBeLessThan(0.1);
    expect(r.rms).toBeGreaterThan(0);
    expect(rigidDifference(r.matrix, T).angle).toBeLessThan(0.01);
  });

  it("rejects too few or collinear pairs", () => {
    expect(() => nPointAlign([{ moving: [0, 0, 0], fixed: [0, 0, 0] }])).toThrow(/at least 3/);
    expect(() =>
      nPointAlign([
        { moving: [0, 0, 0], fixed: [0, 0, 0] },
        { moving: [1, 1, 1], fixed: [1, 1, 1] },
        { moving: [2, 2, 2], fixed: [2, 2, 2] },
      ]),
    ).toThrow(/collinear/);
  });
});

describe("icp", () => {
  const target = heightField(surf, { x0: -3, x1: 3, y0: -3, y1: 3, nx: 220, ny: 220, seed: 1 });
  const query = createPointCloudQuery(target.positions, target.normals, { projectToTangentPlane: true });

  it("point-to-plane recovers a 6° / 0.3 misalignment on a free-form surface", () => {
    const moving0 = heightField(surf, { x0: -2.4, x1: 2.4, y0: -2.4, y1: 2.4, nx: 70, ny: 70, jitter: 0.8, noise: 0.002, seed: 2 });
    const T = mul4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.2, -0.1, 0.15, 1], randomRigid(makeRand(4), (6 * Math.PI) / 180));
    const moving = transformCloud(T, moving0);
    const r = icp(moving.positions, query, { maxIterations: 60, pointToPlane: true, trimFraction: 0.1 });
    const d = rigidDifference(r.matrix, invertRigid(T));
    expect(r.converged).toBe(true);
    expect((d.angle * 180) / Math.PI).toBeLessThan(0.02);
    expect(d.translation).toBeLessThan(0.003);
    expect(r.rms).toBeLessThan(0.004);
  });

  it("point-to-point converges too (much slower, as expected)", () => {
    const moving0 = heightField(surf, { x0: -2.4, x1: 2.4, y0: -2.4, y1: 2.4, nx: 40, ny: 40, jitter: 0.8, seed: 3 });
    const T = mul4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.05, 0.05, -0.05, 1], randomRigid(makeRand(5), (3 * Math.PI) / 180));
    const moving = transformCloud(T, moving0);
    const r = icp(moving.positions, query, { maxIterations: 800, pointToPlane: false, trimFraction: 0.05, tolerance: 1e-9 });
    const d = rigidDifference(r.matrix, invertRigid(T));
    expect((d.angle * 180) / Math.PI).toBeLessThan(0.02);
    expect(d.translation).toBeLessThan(0.001);
  });

  it("box target with analytic closest points, initial guess, normals and progress", () => {
    const half: Vec3 = [2, 1.5, 1];
    const box = boxSurface(half, 4000, 7, 0.001);
    const T = mul4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.3, 0.2, -0.1, 1], randomRigid(makeRand(6), (8 * Math.PI) / 180));
    const moving = transformCloud(T, box);
    let calls = 0;
    const r = icp(moving.positions, boxQuery(half), {
      normals: moving.normals,
      initial: identity4(),
      onProgress: () => calls++,
      trimFraction: 0.05,
    });
    const d = rigidDifference(r.matrix, invertRigid(T));
    expect((d.angle * 180) / Math.PI).toBeLessThan(0.01);
    expect(d.translation).toBeLessThan(0.002);
    expect(calls).toBeGreaterThan(1);
  });

  it("fails clearly with no correspondences", () => {
    const pts = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 2, 2, 2, 3, 1, 2, 1, 1, 1]);
    expect(() => icp(pts, () => null)).toThrow(/correspondences/);
  });
});

describe("globalAlign", () => {
  it("aligns three overlapping scans (scan 0 fixed)", () => {
    const parts = [
      heightField(surf, { x0: -3, x1: 1.5, y0: -3, y1: 3, nx: 110, ny: 140, seed: 11 }),
      heightField(surf, { x0: -1.5, x1: 3, y0: -3, y1: 3, nx: 110, ny: 140, seed: 12 }),
      heightField(surf, { x0: -3, x1: 3, y0: -3, y1: 3, nx: 100, ny: 100, seed: 13 }),
    ];
    const r0 = makeRand(21);
    const truth: Mat4[] = [
      identity4(),
      mul4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.1, -0.05, 0.08, 1], randomRigid(r0, (3 * Math.PI) / 180)),
      mul4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -0.07, 0.1, -0.05, 1], randomRigid(r0, (3 * Math.PI) / 180)),
    ];
    // local coordinates: world = truth * local
    const scans = parts.map((p, i) => {
      const local = transformCloud(invertRigid(truth[i]), p);
      return { positions: local.positions, normals: local.normals, query: createPointCloudQuery(local.positions, local.normals, { projectToTangentPlane: true }) };
    });
    const res = globalAlign(scans, { maxRounds: 15, maxIterations: 20, trimFraction: 0.1, sampleCount: 3000 });
    expect(res.matrices[0]).toEqual(identity4());
    for (let i = 1; i < 3; i++) {
      const d = rigidDifference(res.matrices[i], truth[i], [0, 0, 0]);
      expect((d.angle * 180) / Math.PI).toBeLessThan(0.03);
      expect(d.translation).toBeLessThan(0.003);
      expect(res.rms[i]).toBeLessThan(0.003);
    }
    expect(res.rms[0]).toBeLessThan(0.003);
  });
});

describe("alignToWorld", () => {
  it("maps primary plane normal → Z, secondary axis → X, origin → 0", () => {
    const n = normalize([0.2, -0.3, 0.9] as Vec3);
    const ax0 = normalize([1, 0.5, 0.1] as Vec3);
    const point: Vec3 = [5, 6, 7];
    const m = alignToWorld({
      primary: { kind: "plane", direction: n, point },
      secondary: { kind: "axis", direction: ax0 },
      primaryTarget: "Z",
      secondaryTarget: "X",
    });
    expect(det3(m)).toBeCloseTo(1, 12);
    const tn = transformDir(m, n);
    expect(tn[2]).toBeCloseTo(1, 12);
    const ta = transformDir(m, ax0);
    expect(ta[1]).toBeCloseTo(0, 12); // in the XZ plane
    expect(ta[0]).toBeGreaterThan(0);
    const to = transformPoint(m, point);
    expect(Math.hypot(...to)).toBeLessThan(1e-12);
  });

  it("flips, explicit origin, axis-plane intersection origin and minimal rotation", () => {
    const m = alignToWorld({
      primary: { kind: "axis", direction: [0, 0, 1], point: [1, 1, 1] },
      secondary: { kind: "plane", direction: [1, 0, 0] },
      primaryTarget: "Y",
      secondaryTarget: "Z",
      flipPrimary: true,
      origin: [2, 0, 0],
    });
    expect(transformDir(m, [0, 0, 1])[1]).toBeCloseTo(-1, 12);
    expect(transformDir(m, [1, 0, 0])[2]).toBeCloseTo(1, 12);
    expect(Math.hypot(...transformPoint(m, [2, 0, 0]))).toBeLessThan(1e-12);
    // plane z = 3 + vertical axis through (4, 5, *) → origin at (4, 5, 3)
    const m2 = alignToWorld({
      primary: { kind: "plane", direction: [0, 0, 1], point: [0, 0, 3] },
      secondary: { kind: "axis", direction: [0.1, 0, 1], point: [4, 5, 10] },
      primaryTarget: "Z",
      secondaryTarget: "X",
    });
    const o = transformPoint(m2, [4 - 0.7, 5, 3]);
    expect(Math.hypot(...o)).toBeLessThan(1e-12);
    // primary axis (cylinder) + secondary plane with a point → origin at their intersection
    const m4 = alignToWorld({
      primary: { kind: "axis", direction: [0, 0, 1], point: [1, 2, 50] },
      secondary: { kind: "plane", direction: [1, 0, 0.5], point: [0, 0, 7] },
      primaryTarget: "Z",
      secondaryTarget: "X",
    });
    // axis point (1, 2, z) on plane x + 0.5 z = 3.5 → z = 5
    expect(Math.hypot(...transformPoint(m4, [1, 2, 5]))).toBeLessThan(1e-12);
    const m3 = alignToWorld({ primary: { kind: "plane", direction: [0, 1, 0], point: [0, 0, 0] }, primaryTarget: "Z", secondaryTarget: "X" });
    expect(transformDir(m3, [0, 1, 0])[2]).toBeCloseTo(1, 12);
    expect(transformDir(m3, [1, 0, 0])[0]).toBeCloseTo(1, 12); // X untouched by the minimal rotation
    expect(() =>
      alignToWorld({ primary: { kind: "plane", direction: [0, 0, 1], point: [0, 0, 0] }, secondary: { kind: "axis", direction: [0, 0, 2] }, primaryTarget: "Z", secondaryTarget: "X" }),
    ).toThrow(/parallel/);
    expect(() =>
      alignToWorld({ primary: { kind: "plane", direction: [0, 0, 1], point: [0, 0, 0] }, secondary: { kind: "axis", direction: [1, 0, 0] }, primaryTarget: "Z", secondaryTarget: "Z" }),
    ).toThrow(/different/);
  });
});
