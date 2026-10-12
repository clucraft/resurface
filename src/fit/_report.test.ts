import { it } from "vitest";
import type { ConeDef, CylinderDef, Vec3 } from "../core/types";
import { normalize } from "../core/vec";
import { detectBestPrimitive, fitCone, fitCylinder, fitPlane, fitSphere, ransacPrimitive } from "./primitives";
import { axisAngleDeg, concat, conePatch, cylinderPatch, planePatch, pointLineDistance, spherePatch, withOutliers } from "./testing/synthetic";

it("report", () => {
  const out: string[] = [];
  const t = (name: string, f: () => string) => {
    const t0 = performance.now();
    const s = f();
    out.push(`${name}: ${s}  [${(performance.now() - t0).toFixed(0)} ms]`);
  };
  const axis = normalize([0.2, 0.9, 0.4] as Vec3);
  const origin: Vec3 = [3, -2, 7];
  t("plane noise .05", () => {
    const normal = normalize([0.3, -0.2, 0.93] as Vec3);
    const c = planePatch({ origin: [10, -5, 3], normal, size: 40, n: 5000, noise: 0.05, seed: 11 });
    const r = fitPlane(c.positions);
    return `normal err ${axisAngleDeg(r.def.normal, normal).toExponential(2)}°, rms ${r.stats.rms.toFixed(4)}`;
  });
  for (const [arc, len, nrm, noise] of [
    [60, 40, true, 0.03],
    [60, 8, false, 0.02],
    [45, 30, false, 0.01],
    [45, 30, true, 0.01],
    [30, 30, true, 0.01],
    [30, 30, false, 0.01],
  ] as const) {
    t(`cyl arc ${arc} len ${len} normals ${nrm} noise ${noise}`, () => {
      const R = arc === 60 ? 25 : 10;
      const c = cylinderPatch({ origin, axis, radius: R, arcDeg: arc, length: len, n: 5000, noise, seed: 21 });
      const r = fitCylinder(c.positions, { normals: nrm ? c.normals : undefined });
      return `axis ${axisAngleDeg(r.def.axis, axis).toExponential(2)}°, dR ${(r.def.radius - R).toExponential(2)}, axisDist ${pointLineDistance(r.def.origin, origin, axis).toExponential(2)}, rms ${r.stats.rms.toFixed(4)}`;
    });
  }
  t("cyl 1M points subsampled", () => {
    const c = cylinderPatch({ origin, axis, radius: 25, arcDeg: 90, length: 40, n: 1_000_000, noise: 0.03, seed: 21 });
    const r = fitCylinder(c.positions, { normals: c.normals });
    return `axis ${axisAngleDeg(r.def.axis, axis).toExponential(2)}°, dR ${(r.def.radius - 25).toExponential(2)}, count ${r.stats.count}`;
  });
  const apex: Vec3 = [-4, 2, 1];
  const ca = normalize([0.3, -0.2, 1] as Vec3);
  for (const [arc, nrm, ha] of [
    [90, true, 25],
    [90, false, 25],
    [360, false, 25],
    [60, true, 10],
    [120, true, 2],
  ] as const) {
    t(`cone arc ${arc} normals ${nrm} half ${ha}`, () => {
      const half = (ha * Math.PI) / 180;
      const hMin = ha === 2 ? 300 : 10, hMax = ha === 2 ? 340 : 30;
      const c = conePatch({ apex, axis: ca, halfAngle: half, hMin, hMax, arcDeg: arc, n: 5000, noise: 0.02, seed: 31 });
      const r = fitCone(c.positions, { normals: nrm ? c.normals : undefined });
      const d = r.def as ConeDef;
      return `axis ${axisAngleDeg(d.axis, ca).toExponential(2)}°, dHalf ${(((d.halfAngle - half) * 180) / Math.PI).toExponential(2)}°, apex ${Math.hypot(d.apex[0] - apex[0], d.apex[1] - apex[1], d.apex[2] - apex[2]).toExponential(2)}, rms ${r.stats.rms.toFixed(4)} near ${r.nearCylinder}`;
    });
  }
  t("sphere cap 30", () => {
    const c = spherePatch({ center: [1, 2, 3], radius: 50, pole: [1, 1, 0], capDeg: 30, n: 4000, noise: 0.02, seed: 4 });
    const r = fitSphere(c.positions);
    return `dR ${(r.def.radius - 50).toExponential(2)}, dC ${Math.hypot(r.def.center[0] - 1, r.def.center[1] - 2, r.def.center[2] - 3).toExponential(2)}`;
  });
  t("ransac cyl+wall", () => {
    const cyl = cylinderPatch({ origin: [0, 0, 0], axis: [0, 0, 1], radius: 10, arcDeg: 120, length: 30, n: 4000, noise: 0.01, seed: 51 });
    const wall = planePatch({ origin: [12, 0, 15], normal: [1, 0, 0], size: 14, n: 1500, noise: 0.01, seed: 52 });
    const mix = withOutliers(concat(cyl, wall), 0.05, 2, 53);
    const r = ransacPrimitive("cylinder", mix.positions, mix.normals, { threshold: 0.08, normalThresholdDeg: 20, seed: 1 });
    const d = r.def as CylinderDef;
    return `axis ${axisAngleDeg(d.axis, [0, 0, 1]).toExponential(2)}°, dR ${(d.radius - 10).toExponential(2)}, inl ${r.stats.count}`;
  });
  t("detect", () => {
    const c = cylinderPatch({ origin: [0, 0, 0], axis: [1, 0.2, 0.1], radius: 15, arcDeg: 90, length: 30, n: 3000, noise: 0.03, seed: 43 });
    return detectBestPrimitive(c.positions, { normals: c.normals }).map((r) => `${r.kind}:${r.rms.toFixed(4)}/${r.score.toFixed(4)}`).join(" ");
  });
  console.log(out.join("\n"));
});
