import { it } from "vitest";
import type { Vec3 } from "../core/types";
import { mul4, transformDir, transformPoint } from "../core/vec";
import { estimateSymmetryPlane } from "./symmetry";
import { axisAngleDeg, boxQuery, boxSurface, makeRand, randomRigid, transformCloud } from "./testing/synthetic";
it("dbg", () => {
  const half: Vec3 = [3, 2, 1];
  const M = mul4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, -3, 2, 1], randomRigid(makeRand(1), 1.2));
  const cloud = transformCloud(M, boxSurface(half, 6000, 2));
  for (const mi of [60, 200]) for (const nrm of [true, false]) {
  const r = estimateSymmetryPlane(cloud.positions, { query: boxQuery(half, M), normals: nrm ? cloud.normals : undefined, maxIterations: mi });
  const axes = [transformDir(M, [1, 0, 0]), transformDir(M, [0, 1, 0]), transformDir(M, [0, 0, 1])];
  const center = transformPoint(M, [0, 0, 0]);
  for (const c of r.candidates) {
    const best = axes.map((a) => axisAngleDeg(a, c.plane.normal).toFixed(4));
    const off = (center[0] - c.plane.origin[0]) * c.plane.normal[0] + (center[1] - c.plane.origin[1]) * c.plane.normal[1] + (center[2] - c.plane.origin[2]) * c.plane.normal[2];
    console.log(mi, nrm, best.join(","), off, c.rms, c.coverage, c.iterations);
  }
  }
});
