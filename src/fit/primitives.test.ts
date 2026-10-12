import { describe, expect, it } from "vitest";
import type { ConeDef, CylinderDef, Vec3 } from "../core/types";
import { normalize } from "../core/vec";
import {
  detectBestPrimitive,
  distanceToPrimitive,
  fitCone,
  fitConeWithAxis,
  fitCylinder,
  fitCylinderWithAxis,
  fitCylinderWithRadius,
  fitPlane,
  fitPlaneWithNormal,
  fitPrimitive,
  fitSphere,
  fitSphereWithRadius,
  fitStats,
  primitiveNormalAt,
  projectToPrimitive,
  ransacPrimitive,
  residuals,
} from "./primitives";
import { axisAngleDeg, concat, conePatch, cylinderPatch, planePatch, pointLineDistance, spherePatch, withOutliers } from "./testing/synthetic";

const deg = (r: number) => (r * 180) / Math.PI;

describe("fitPlane", () => {
  it("recovers a noisy tilted plane", () => {
    const normal = normalize([0.3, -0.2, 0.93] as Vec3);
    const c = planePatch({ origin: [10, -5, 3], normal, size: 40, n: 5000, noise: 0.05, seed: 11 });
    const r = fitPlane(c.positions);
    expect(axisAngleDeg(r.def.normal, normal)).toBeLessThan(0.05);
    expect(Math.abs(distanceToPrimitive(r.def, [10, -5, 3]))).toBeLessThan(0.01);
    expect(r.stats.rms).toBeGreaterThan(0.045);
    expect(r.stats.rms).toBeLessThan(0.055);
    expect(r.stats.count).toBe(5000);
  });

  it("orients the normal with the point normals", () => {
    const normal = normalize([0, 0, -1] as Vec3);
    const c = planePatch({ origin: [0, 0, 0], normal, size: 10, n: 500, seed: 3 });
    expect(fitPlane(c.positions, { normals: c.normals }).def.normal[2]).toBeCloseTo(-1, 6);
    expect(fitPlane(c.positions).def.normal[2]).toBeCloseTo(1, 6); // canonical sign without normals
  });

  it("robust mode ignores 30% outliers", () => {
    const normal = normalize([0.1, 0.2, 1] as Vec3);
    const base = planePatch({ origin: [0, 0, 0], normal, size: 20, n: 3000, noise: 0.02, seed: 5 });
    const c = withOutliers(base, 0.3 / 0.7, 3, 77);
    const plain = fitPlane(c.positions);
    const rob = fitPlane(c.positions, { robust: true });
    expect(axisAngleDeg(rob.def.normal, normal)).toBeLessThan(0.1);
    expect(axisAngleDeg(plain.def.normal, normal)).toBeGreaterThan(axisAngleDeg(rob.def.normal, normal));
    expect(rob.inliers).toBeDefined();
    let inl = 0;
    for (let i = 0; i < 3000; i++) inl += rob.inliers![i];
    expect(inl).toBeGreaterThan(2950);
    expect(rob.stats.rms).toBeLessThan(0.03);
  });

  it("uses only the selected indices and subsamples big selections", () => {
    const a = planePatch({ origin: [0, 0, 0], normal: [0, 0, 1], size: 10, n: 300_000, noise: 0.01, seed: 1 });
    const b = planePatch({ origin: [0, 0, 5], normal: [1, 0, 0], size: 10, n: 1000, seed: 2 });
    const c = concat(a, b);
    const idx = new Uint32Array(300_000).map((_, i) => i);
    const r = fitPlane(c.positions, { indices: idx, maxPoints: 20_000 });
    expect(axisAngleDeg(r.def.normal, [0, 0, 1])).toBeLessThan(0.05);
    expect(r.stats.count).toBe(300_000);
    expect(Math.abs(r.def.origin[2])).toBeLessThan(0.002);
  });

  it("throws on collinear input", () => {
    const p = new Float32Array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]);
    expect(() => fitPlane(p)).toThrow(/collinear/);
    expect(() => fitPlane(new Float32Array([0, 0, 0, 1, 1, 1]))).toThrow(/at least 3/);
  });

  it("fitPlaneWithNormal keeps the normal", () => {
    const c = planePatch({ origin: [0, 0, 2], normal: [0, 0.05, 1], size: 10, n: 1000, seed: 4 });
    const r = fitPlaneWithNormal(c.positions, [0, 0, 1]);
    expect(r.def.normal).toEqual([0, 0, 1]);
    expect(r.def.origin[2]).toBeCloseTo(2, 1);
  });
});

describe("fitSphere", () => {
  it("recovers a full noisy sphere", () => {
    const c = spherePatch({ center: [5, 6, -7], radius: 12, capDeg: 180, n: 4000, noise: 0.05, seed: 3 });
    const r = fitSphere(c.positions);
    expect(Math.hypot(r.def.center[0] - 5, r.def.center[1] - 6, r.def.center[2] + 7)).toBeLessThan(0.01);
    expect(Math.abs(r.def.radius - 12)).toBeLessThan(0.01);
    expect(r.stats.rms).toBeLessThan(0.06);
  });

  it("recovers a 30° cap (partial data)", () => {
    const c = spherePatch({ center: [1, 2, 3], radius: 50, pole: [1, 1, 0], capDeg: 30, n: 4000, noise: 0.02, seed: 4 });
    const r = fitSphere(c.positions);
    expect(Math.abs(r.def.radius - 50)).toBeLessThan(0.15);
    expect(Math.hypot(r.def.center[0] - 1, r.def.center[1] - 2, r.def.center[2] - 3)).toBeLessThan(0.2);
  });

  it("robust sphere with outliers + fixed radius variant", () => {
    const base = spherePatch({ center: [0, 0, 0], radius: 10, capDeg: 90, n: 3000, noise: 0.02, seed: 8 });
    const c = withOutliers(base, 0.2, 2, 5);
    const r = fitSphere(c.positions, { robust: true });
    expect(Math.abs(r.def.radius - 10)).toBeLessThan(0.02);
    const f = fitSphereWithRadius(base.positions, 10);
    expect(f.def.radius).toBe(10);
    expect(Math.hypot(...f.def.center)).toBeLessThan(0.02);
  });

  it("throws on coplanar input", () => {
    const c = planePatch({ origin: [0, 0, 0], normal: [0, 0, 1], size: 10, n: 200, seed: 1 });
    expect(() => fitSphere(c.positions)).toThrow(/coplanar/);
  });
});

function cylErr(def: CylinderDef, origin: Vec3, axis: Vec3, radius: number) {
  return {
    axisDeg: axisAngleDeg(def.axis, axis),
    radius: Math.abs(def.radius - radius),
    axisDist: pointLineDistance(def.origin, origin, axis),
  };
}

describe("fitCylinder", () => {
  const origin: Vec3 = [3, -2, 7];
  const axis = normalize([0.2, 0.9, 0.4] as Vec3);

  it("60° patch with normals", () => {
    const c = cylinderPatch({ origin, axis, radius: 25, arcDeg: 60, length: 40, n: 5000, noise: 0.03, seed: 21 });
    const r = fitCylinder(c.positions, { normals: c.normals });
    const e = cylErr(r.def, origin, axis, 25);
    expect(e.axisDeg).toBeLessThan(0.05);
    expect(e.radius).toBeLessThan(0.05);
    expect(e.axisDist).toBeLessThan(0.05);
    expect(r.stats.rms).toBeLessThan(0.035);
    expect(r.def.hMax - r.def.hMin).toBeGreaterThan(39.5);
    expect(r.def.hMax - r.def.hMin).toBeLessThan(40.5);
  });

  it("60° patch without normals (short and wide)", () => {
    const c = cylinderPatch({ origin, axis, radius: 25, arcDeg: 60, length: 8, n: 5000, noise: 0.02, seed: 22 });
    const r = fitCylinder(c.positions);
    const e = cylErr(r.def, origin, axis, 25);
    expect(e.axisDeg).toBeLessThan(0.3);
    expect(e.radius).toBeLessThan(0.1);
  });

  it("45° patch without normals", () => {
    const c = cylinderPatch({ origin, axis, radius: 10, arcDeg: 45, length: 30, n: 5000, noise: 0.01, seed: 23 });
    const r = fitCylinder(c.positions);
    const e = cylErr(r.def, origin, axis, 10);
    expect(e.axisDeg).toBeLessThan(0.2);
    expect(e.radius).toBeLessThan(0.05);
    expect(e.axisDist).toBeLessThan(0.05);
  });

  it("full cylinder far from the origin (scale/translation invariance)", () => {
    const o2: Vec3 = [1e5, -2e5, 3e4];
    const c = cylinderPatch({ origin: o2, axis, radius: 0.5, arcDeg: 360, length: 2, n: 4000, noise: 0.001, seed: 24 });
    const r = fitCylinder(c.positions);
    const e = cylErr(r.def, o2, axis, 0.5);
    expect(e.axisDeg).toBeLessThan(0.2);
    // float32 positions at 1e5 have ~0.008 quantisation
    expect(e.radius).toBeLessThan(0.01);
  });

  it("robust: ignores 15% outliers", () => {
    const base = cylinderPatch({ origin, axis, radius: 15, arcDeg: 120, length: 30, n: 4000, noise: 0.02, seed: 25 });
    const c = withOutliers(base, 0.15, 1, 3);
    const r = fitCylinder(c.positions, { normals: c.normals, robust: true });
    const e = cylErr(r.def, origin, axis, 15);
    expect(e.axisDeg).toBeLessThan(0.1);
    expect(e.radius).toBeLessThan(0.03);
    expect(r.inliers).toBeDefined();
  });

  it("constrained: fixed axis, fixed axis line, fixed radius", () => {
    const c = cylinderPatch({ origin, axis, radius: 12, arcDeg: 90, length: 20, n: 3000, noise: 0.01, seed: 26 });
    const a = fitCylinderWithAxis(c.positions, axis);
    expect(axisAngleDeg(a.def.axis, axis)).toBeLessThan(1e-9);
    expect(Math.abs(a.def.radius - 12)).toBeLessThan(0.02);
    const co = fitCylinderWithAxis(c.positions, axis, { axisPoint: origin });
    expect(pointLineDistance(co.def.origin, origin, axis)).toBeLessThan(1e-9);
    expect(Math.abs(co.def.radius - 12)).toBeLessThan(0.02);
    const fr = fitCylinderWithRadius(c.positions, 12.5);
    expect(fr.def.radius).toBeCloseTo(12.5, 9);
    expect(axisAngleDeg(fr.def.axis, axis)).toBeLessThan(1);
    // a slightly tilted fixed axis yields a worse fit
    const tilted = fitCylinderWithAxis(c.positions, normalize([axis[0] + 0.05, axis[1], axis[2]] as Vec3));
    expect(tilted.stats.rms).toBeGreaterThan(a.stats.rms);
  });
});

function coneErr(def: ConeDef, apex: Vec3, axis: Vec3, half: number) {
  return {
    axisDeg: axisAngleDeg(def.axis, axis),
    sameDir: def.axis[0] * axis[0] + def.axis[1] * axis[1] + def.axis[2] * axis[2] > 0,
    halfDeg: Math.abs(deg(def.halfAngle - half)),
    apex: Math.hypot(def.apex[0] - apex[0], def.apex[1] - apex[1], def.apex[2] - apex[2]),
  };
}

describe("fitCone", () => {
  const apex: Vec3 = [-4, 2, 1];
  const axis = normalize([0.3, -0.2, 1] as Vec3);
  const half = (25 * Math.PI) / 180;

  it("90° patch with normals", () => {
    const c = conePatch({ apex, axis, halfAngle: half, hMin: 10, hMax: 30, arcDeg: 90, n: 5000, noise: 0.02, seed: 31 });
    const r = fitCone(c.positions, { normals: c.normals });
    const e = coneErr(r.def, apex, axis, half);
    expect(e.sameDir).toBe(true);
    expect(e.axisDeg).toBeLessThan(0.1);
    expect(e.halfDeg).toBeLessThan(0.05);
    expect(e.apex).toBeLessThan(0.1);
    expect(r.nearCylinder).toBe(false);
    expect(r.def.hMin).toBeGreaterThan(9.5);
    expect(r.def.hMax).toBeLessThan(30.5);
    expect(r.stats.rms).toBeLessThan(0.025);
  });

  it("full cone without normals", () => {
    const c = conePatch({ apex, axis, halfAngle: half, hMin: 5, hMax: 20, arcDeg: 360, n: 5000, noise: 0.01, seed: 32 });
    const r = fitCone(c.positions);
    const e = coneErr(r.def, apex, axis, half);
    expect(e.sameDir).toBe(true);
    expect(e.axisDeg).toBeLessThan(0.1);
    expect(e.halfDeg).toBeLessThan(0.05);
    expect(e.apex).toBeLessThan(0.05);
  });

  it("small draft angle (2°) on a 120° patch", () => {
    const h2 = (2 * Math.PI) / 180;
    const c = conePatch({ apex, axis, halfAngle: h2, hMin: 300, hMax: 340, arcDeg: 120, n: 5000, noise: 0.005, seed: 33 });
    const r = fitCone(c.positions, { normals: c.normals });
    const e = coneErr(r.def, apex, axis, h2);
    expect(e.halfDeg).toBeLessThan(0.05);
    expect(e.axisDeg).toBeLessThan(0.2);
    expect(r.nearCylinder).toBe(false);
  });

  it("reports near-cylinder degeneracy", () => {
    const c = cylinderPatch({ origin: [0, 0, 0], axis: [0, 0, 1], radius: 10, arcDeg: 180, length: 20, n: 3000, noise: 0.002, seed: 34 });
    const r = fitCone(c.positions, { normals: c.normals });
    expect(r.nearCylinder).toBe(true);
    expect(r.warning).toMatch(/cylind/);
    expect(r.cylinder).toBeDefined();
    expect(Math.abs(r.cylinder!.radius - 10)).toBeLessThan(0.01);
    expect(r.def.halfAngle).toBeGreaterThan(0);
    expect(r.stats.rms).toBeLessThan(0.003);
  });

  it("fitConeWithAxis keeps the axis", () => {
    const c = conePatch({ apex, axis, halfAngle: half, hMin: 10, hMax: 30, arcDeg: 180, n: 3000, noise: 0.01, seed: 35 });
    const r = fitConeWithAxis(c.positions, axis, { normals: c.normals });
    expect(axisAngleDeg(r.def.axis, axis)).toBeLessThan(1e-9);
    expect(coneErr(r.def, apex, axis, half).halfDeg).toBeLessThan(0.05);
  });
});

describe("residuals / projection / stats", () => {
  it("projection lies on the primitive and distances are signed", () => {
    const prims = [
      { kind: "plane" as const, origin: [1, 2, 3] as Vec3, normal: normalize([1, 1, 0] as Vec3) },
      { kind: "sphere" as const, center: [1, 2, 3] as Vec3, radius: 4 },
      { kind: "cylinder" as const, origin: [0, 0, 0] as Vec3, axis: [0, 0, 1] as Vec3, radius: 2, hMin: 0, hMax: 1 },
      { kind: "cone" as const, apex: [0, 0, 0] as Vec3, axis: [0, 0, 1] as Vec3, halfAngle: 0.5, hMin: 0, hMax: 1 },
    ];
    const pts: Vec3[] = [
      [5, -3, 2],
      [0.3, 0.2, 7],
      [-2, 1, -4],
      [0.1, 0.1, 0.1],
    ];
    for (const d of prims)
      for (const p of pts) {
        const q = projectToPrimitive(d, p);
        expect(Math.abs(distanceToPrimitive(d, q))).toBeLessThan(1e-9);
        const dist = Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
        expect(Math.abs(Math.abs(distanceToPrimitive(d, p)) - dist)).toBeLessThan(1e-9);
        // moving along the outward normal increases the signed distance
        const n = primitiveNormalAt(d, q);
        expect(distanceToPrimitive(d, [q[0] + 0.01 * n[0], q[1] + 0.01 * n[1], q[2] + 0.01 * n[2]])).toBeCloseTo(0.01, 6);
      }
    expect(distanceToPrimitive(prims[1], [1, 2, 3])).toBeCloseTo(-4, 12);
    // point behind the apex is closest to the apex
    expect(distanceToPrimitive(prims[3], [0, 0, -2])).toBeCloseTo(2, 12);
  });

  it("residuals + fitStats", () => {
    const pos = new Float32Array([0, 0, 1, 0, 0, -2, 1, 1, 3]);
    const res = residuals({ kind: "plane", origin: [0, 0, 0], normal: [0, 0, 1] }, pos, [0, 1, 2]);
    expect(Array.from(res)).toEqual([1, -2, 3]);
    const st = fitStats(res);
    expect(st.count).toBe(3);
    expect(st.maxAbs).toBe(3);
    expect(st.mean).toBeCloseTo(2 / 3, 12);
    expect(st.rms).toBeCloseTo(Math.sqrt(14 / 3), 12);
  });
});

describe("detectBestPrimitive", () => {
  const cases = [
    { name: "plane", cloud: planePatch({ origin: [0, 0, 0], normal: [0.1, 0.3, 1], size: 30, n: 3000, noise: 0.03, seed: 41 }) },
    { name: "sphere", cloud: spherePatch({ center: [1, 1, 1], radius: 20, capDeg: 50, n: 3000, noise: 0.03, seed: 42 }) },
    { name: "cylinder", cloud: cylinderPatch({ origin: [0, 0, 0], axis: [1, 0.2, 0.1], radius: 15, arcDeg: 90, length: 30, n: 3000, noise: 0.03, seed: 43 }) },
    { name: "cone", cloud: conePatch({ apex: [0, 0, 0], axis: [0, 1, 0.2], halfAngle: 0.4, hMin: 10, hMax: 25, arcDeg: 120, n: 3000, noise: 0.03, seed: 44 }) },
  ];
  for (const c of cases)
    it(`ranks a ${c.name} first`, () => {
      const ranked = detectBestPrimitive(c.cloud.positions, { normals: c.cloud.normals });
      expect(ranked[0].kind).toBe(c.name);
      expect(ranked[0].rms).toBeLessThan(0.04);
    });
});

describe("ransacPrimitive", () => {
  it("cylinder in a sloppy selection with a neighbouring plane and outliers", () => {
    const cyl = cylinderPatch({ origin: [0, 0, 0], axis: [0, 0, 1], radius: 10, arcDeg: 120, length: 30, n: 4000, noise: 0.01, seed: 51 });
    // neighbouring flat face tangent-ish to the cylinder (like a fillet into a wall)
    const wall = planePatch({ origin: [12, 0, 15], normal: [1, 0, 0], size: 14, n: 1500, noise: 0.01, seed: 52 });
    const mix = withOutliers(concat(cyl, wall), 0.05, 2, 53);
    const r = ransacPrimitive("cylinder", mix.positions, mix.normals, { threshold: 0.08, normalThresholdDeg: 20, seed: 1 });
    expect(r.def.kind).toBe("cylinder");
    const d = r.def as CylinderDef;
    expect(axisAngleDeg(d.axis, [0, 0, 1])).toBeLessThan(0.1);
    expect(Math.abs(d.radius - 10)).toBeLessThan(0.02);
    let inCyl = 0, inOther = 0;
    for (let i = 0; i < r.inliers!.length; i++) {
      if (!r.inliers![i]) continue;
      if (i < 4000) inCyl++;
      else inOther++;
    }
    expect(inCyl).toBeGreaterThan(3950);
    expect(inOther).toBeLessThan(80);
    expect(r.stats.rms).toBeLessThan(0.015);
  });

  it("plane without normals among outliers, and cylinder without normals", () => {
    const base = planePatch({ origin: [0, 0, 0], normal: [0, 0, 1], size: 20, n: 2000, noise: 0.01, seed: 54 });
    const mix = withOutliers(base, 1.0, 1, 55); // 50 % outliers
    const r = ransacPrimitive("plane", mix.positions, null, { threshold: 0.05, seed: 3 });
    expect(axisAngleDeg((r.def as { normal: Vec3 }).normal, [0, 0, 1])).toBeLessThan(0.1);
    const cyl = cylinderPatch({ origin: [0, 0, 0], axis: [0, 1, 0], radius: 5, arcDeg: 180, length: 20, n: 4000, noise: 0.005, seed: 56 });
    const r2 = ransacPrimitive("cylinder", cyl.positions, null, { threshold: 0.03, seed: 4 });
    expect(Math.abs((r2.def as CylinderDef).radius - 5)).toBeLessThan(0.01);
  });

  it("sphere and cone via RANSAC", () => {
    const s = withOutliers(spherePatch({ center: [2, 0, 0], radius: 8, capDeg: 70, n: 3000, noise: 0.01, seed: 57 }), 0.3, 2, 58);
    const rs = ransacPrimitive("sphere", s.positions, s.normals, { threshold: 0.05, seed: 5 });
    expect(Math.abs((rs.def as { radius: number }).radius - 8)).toBeLessThan(0.02);
    const c = withOutliers(conePatch({ apex: [0, 0, 0], axis: [0, 0, 1], halfAngle: 0.5, hMin: 5, hMax: 15, arcDeg: 180, n: 3000, noise: 0.01, seed: 59 }), 0.3, 2, 60);
    const rc = ransacPrimitive("cone", c.positions, c.normals, { threshold: 0.05, seed: 6 });
    expect(Math.abs((rc.def as ConeDef).halfAngle - 0.5)).toBeLessThan(0.002);
  });
});

describe("fitPrimitive", () => {
  it("dispatches by kind and honours the initial guess", () => {
    const c = cylinderPatch({ origin: [0, 0, 0], axis: [0, 0, 1], radius: 3, arcDeg: 200, length: 5, n: 1000, noise: 0.001, seed: 61 });
    const r = fitPrimitive("cylinder", c.positions, { initial: { kind: "cylinder", origin: [0.1, 0, 0], axis: [0, 0.05, 1], radius: 2.8, hMin: 0, hMax: 1 } });
    expect(Math.abs((r.def as CylinderDef).radius - 3)).toBeLessThan(0.001);
  });
});
