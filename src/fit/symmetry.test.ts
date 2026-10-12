import { describe, expect, it } from "vitest";
import type { PlaneDef, Vec3 } from "../core/types";
import { invertRigid, mul4, normalize, transformDir, transformPoint } from "../core/vec";
import { createPointCloudQuery } from "./pointQuery";
import { estimateSymmetryPlane, symmetryDeviation } from "./symmetry";
import { axisAngleDeg, boxQuery, boxSurface, heightField, makeRand, randomRigid, transformCloud } from "./testing/synthetic";

// symmetric in x (mirror plane x = 0), asymmetric otherwise
const symSurf = (x: number, y: number) => 0.5 * Math.cos(1.2 * x) + 0.3 * y + 0.15 * y * y + 0.08 * x * x * y;

describe("estimateSymmetryPlane", () => {
  it("finds a symmetry plane of a rotated box", () => {
    const half: Vec3 = [3, 2, 1];
    const M = mul4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, -3, 2, 1], randomRigid(makeRand(1), 1.2));
    const cloud = transformCloud(M, boxSurface(half, 6000, 2));
    const r = estimateSymmetryPlane(cloud.positions, { query: boxQuery(half, M), normals: cloud.normals });
    expect(r.candidates.length).toBe(3);
    // every refined candidate is one of the box's symmetry planes
    const axes = [transformDir(M, [1, 0, 0]), transformDir(M, [0, 1, 0]), transformDir(M, [0, 0, 1])];
    const center = transformPoint(M, [0, 0, 0]);
    for (const c of r.candidates) {
      const best = Math.min(...axes.map((a) => axisAngleDeg(a, c.plane.normal)));
      expect(best).toBeLessThan(0.01);
      const off = (center[0] - c.plane.origin[0]) * c.plane.normal[0] + (center[1] - c.plane.origin[1]) * c.plane.normal[1] + (center[2] - c.plane.origin[2]) * c.plane.normal[2];
      expect(Math.abs(off)).toBeLessThan(1e-3);
      expect(c.rms).toBeLessThan(1e-3);
    }
    expect(r.rms).toBeLessThan(1e-3);
  });

  it("refines a tilted initial guess on a free-form part with a single symmetry plane", () => {
    const M = mul4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 20, -5, 1], randomRigid(makeRand(3), 2.5));
    const dense = transformCloud(M, heightField(symSurf, { x0: -3, x1: 3, y0: -2, y1: 3, nx: 240, ny: 200, seed: 4 }));
    const query = createPointCloudQuery(dense.positions, dense.normals, { projectToTangentPlane: true });
    const scan = transformCloud(M, heightField(symSurf, { x0: -3, x1: 3, y0: -2, y1: 3, nx: 80, ny: 70, jitter: 0.9, noise: 0.002, seed: 5 }));
    const trueN = transformDir(M, [1, 0, 0]);
    const trueO = transformPoint(M, [0, 0, 0]);
    // initial: 8° off and shifted by 0.3
    const tilt = normalize([Math.cos(0.14), Math.sin(0.14), 0] as Vec3);
    const initial: PlaneDef = { kind: "plane", origin: transformPoint(M, [0.3, 0, 0]), normal: transformDir(M, tilt) };
    const r = estimateSymmetryPlane(scan.positions, { query, normals: scan.normals, initial, sampleCount: 2500 });
    expect(axisAngleDeg(r.plane.normal, trueN)).toBeLessThan(0.05);
    const off = (trueO[0] - r.plane.origin[0]) * r.plane.normal[0] + (trueO[1] - r.plane.origin[1]) * r.plane.normal[1] + (trueO[2] - r.plane.origin[2]) * r.plane.normal[2];
    expect(Math.abs(off)).toBeLessThan(0.005);
    expect(r.rms).toBeLessThan(0.01);
    // the asymmetric PCA planes score worse
    expect(r.candidates[r.candidates.length - 1].rms).toBeGreaterThan(10 * r.rms);
  });

  it("ignores a one-sided asymmetric feature (trimming) starting from PCA candidates", () => {
    const bumpy = (x: number, y: number) => symSurf(x, y) + 0.4 * Math.exp(-((x - 1.8) ** 2 + (y - 1) ** 2) / 0.08);
    const M = mul4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -4, 2, 8, 1], randomRigid(makeRand(7), 2));
    const dense = transformCloud(M, heightField(bumpy, { x0: -3, x1: 3, y0: -2, y1: 3, nx: 240, ny: 200, seed: 8 }));
    const query = createPointCloudQuery(dense.positions, dense.normals, { projectToTangentPlane: true });
    const scan = transformCloud(M, heightField(bumpy, { x0: -3, x1: 3, y0: -2, y1: 3, nx: 80, ny: 70, jitter: 0.9, noise: 0.002, seed: 9 }));
    const r = estimateSymmetryPlane(scan.positions, { query, normals: scan.normals });
    expect(axisAngleDeg(r.plane.normal, transformDir(M, [1, 0, 0]))).toBeLessThan(0.1);
    const o = transformPoint(M, [0, 0, 0]);
    const off = (o[0] - r.plane.origin[0]) * r.plane.normal[0] + (o[1] - r.plane.origin[1]) * r.plane.normal[1] + (o[2] - r.plane.origin[2]) * r.plane.normal[2];
    expect(Math.abs(off)).toBeLessThan(0.01);
  });

  it("requires a query", () => {
    expect(() => estimateSymmetryPlane(new Float32Array(30), {} as never)).toThrow(/query/);
  });
});

describe("symmetryDeviation", () => {
  it("is ~0 on a symmetric surface and shows a one-sided bump", () => {
    const bump = (x: number, y: number) => symSurf(x, y) + 0.2 * Math.exp(-((x - 1.5) ** 2 + (y - 0.5) ** 2) / 0.1);
    const dense = heightField(bump, { x0: -3, x1: 3, y0: -2, y1: 3, nx: 240, ny: 200, seed: 6 });
    const query = createPointCloudQuery(dense.positions, dense.normals, { projectToTangentPlane: true });
    const pts = new Float32Array([0.5, 1, symSurf(0.5, 1), 1.5, 0.5, bump(1.5, 0.5), -1.5, 0.5, symSurf(-1.5, 0.5)]);
    const dev = symmetryDeviation(pts, { kind: "plane", origin: [0, 0, 0], normal: [1, 0, 0] }, query);
    expect(Math.abs(dev[0])).toBeLessThan(0.002);
    // the bump mirrored to the other side lies above the surface there
    expect(dev[1]).toBeGreaterThan(0.15);
    // the plain side mirrored onto the bump lies below it
    expect(dev[2]).toBeLessThan(-0.1);
    // identity check with an inverse transform (just exercising helpers)
    expect(invertRigid([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 2, 3, 1])[12]).toBe(-1);
  });
});
