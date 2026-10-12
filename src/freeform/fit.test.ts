import { describe, expect, it } from "vitest";
import type { ClosestPointQuery, Vec3 } from "../core/types";
import { cylinderTarget, heightFieldTarget, planeTarget, sphereRaycast, sphereTarget, torusTarget, wavyHeightField } from "./analyticTargets";
import { patchDeviation, surfaceDeviation } from "./analysis";
import { limitPoints, subdivisionStencils } from "./catmullClark";
import { fitCageToTarget, snapVerticesToTarget } from "./fit";
import { toBicubicPatches } from "./patches";
import { createBoxCage, createCylinderCage, createGridCage, createTorusCage } from "./quadMesh";
import { bsr3BlockIndex, bsr3FromRowColumns, csrMulVec3, pcgBsr3 } from "./sparse";

describe("sparse PCG", () => {
  it("solves a block SPD system (1D Laplacian + shift, coupled 3x3 blocks)", () => {
    const n = 50;
    const rows = Array.from({ length: n }, (_, i) => [i - 1, i, i + 1].filter((j) => j >= 0 && j < n));
    const A = bsr3FromRowColumns(n, rows);
    for (let i = 0; i < n; i++)
      for (const j of rows[i]) {
        const b = 9 * bsr3BlockIndex(A, i, j);
        if (i === j) {
          A.values.set([2.5, 0.3, 0, 0.3, 2.5, 0.1, 0, 0.1, 2.5], b);
        } else A.values.set([-1, 0, 0, 0, -1, 0, 0, 0, -1], b);
      }
    const xTrue = Float64Array.from({ length: 3 * n }, (_, k) => Math.sin(k));
    const b = new Float64Array(3 * n);
    // b = A xTrue
    for (let i = 0; i < n; i++)
      for (let p = A.rowPtr[i]; p < A.rowPtr[i + 1]; p++) {
        const j = A.colIdx[p];
        for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) b[3 * i + r] += A.values[9 * p + 3 * r + c] * xTrue[3 * j + c];
      }
    const x = new Float64Array(3 * n);
    const res = pcgBsr3(A, b, x, 1e-12, 500);
    expect(res.converged).toBe(true);
    for (let k = 0; k < 3 * n; k++) expect(Math.abs(x[k] - xTrue[k])).toBeLessThan(1e-9);
  });
});

describe("snap-to-mesh fitting", () => {
  it("fits a 6x6x6 box cage to a sphere of radius 50 with rms < 0.05", () => {
    const target = sphereTarget([0, 0, 0], 50);
    const cage = createBoxCage([-45, -45, -45], [45, 45, 45], 6);
    const t0 = performance.now();
    const fit = fitCageToTarget(cage, target, { levels: 2, iterations: 10 });
    const ms = performance.now() - t0;
    expect(fit.rms).toBeLessThan(0.05);
    expect(fit.maxDev).toBeLessThan(0.25);
    // independent check on a finer sampling (level 3) and on the exported NURBS patches
    const dev3 = surfaceDeviation(fit.cage, target, 3);
    expect(dev3.rms).toBeLessThan(0.05);
    const patches = toBicubicPatches(fit.cage);
    const pd = patchDeviation(patches, target, 10);
    expect(pd.rms).toBeLessThan(0.1);
    console.log(
      `sphere R=50, box cage 6^3 (${cage.faces.length} faces): rms=${fit.rms.toExponential(3)} max=${fit.maxDev.toExponential(3)} ` +
        `(level-3 rms=${dev3.rms.toExponential(3)} max=${dev3.maxAbs.toExponential(3)}; patches rms=${pd.rms.toExponential(3)} max=${pd.maxAbs.toExponential(3)}) ` +
        `iterations=${fit.iterations} history=${fit.history.map((h) => h.toPrecision(3)).join(",")} time=${ms.toFixed(0)}ms`,
    );
  });

  it("fits a torus, an open cylinder and a wavy height field", () => {
    const report: string[] = [];
    // torus
    {
      const target = torusTarget([0, 0, 0], 40, 12);
      const cage = createTorusCage(38, 9, 24, 12);
      const fit = fitCageToTarget(cage, target, { iterations: 10 });
      expect(fit.rms).toBeLessThan(0.02);
      report.push(`torus R=40 r=12 (24x12 cage): rms=${fit.rms.toExponential(3)} max=${fit.maxDev.toExponential(3)}`);
    }
    // open cylinder (boundary rows free to slide along the infinite cylinder)
    {
      const target = cylinderTarget([0, 0, 0], [0, 0, 1], 25);
      const cage = createCylinderCage(18, 60, 8, 4, [0, 0, -30]);
      const fit = fitCageToTarget(cage, target, { iterations: 10 });
      expect(fit.rms).toBeLessThan(0.05);
      report.push(`cylinder r=25 (8x4 open cage): rms=${fit.rms.toExponential(3)} max=${fit.maxDev.toExponential(3)}`);
    }
    // wavy height field z = 3 sin(x/8) cos(y/10) over a 100x100 patch
    {
      const hf = wavyHeightField(3, 1 / 8, 1 / 10);
      const target = heightFieldTarget(hf);
      const cage = createGridCage(12, 12, { origin: [-50, -50, 0], uAxis: [100, 0, 0], vAxis: [0, 100, 0] });
      const fit = fitCageToTarget(cage, target, { iterations: 10, levels: 2 });
      expect(fit.rms).toBeLessThan(0.05);
      report.push(`height field (12x12 open grid): rms=${fit.rms.toExponential(3)} max=${fit.maxDev.toExponential(3)}`);
    }
    console.log(report.join("\n"));
  });

  it("keeps fixed vertices in place", () => {
    const target = sphereTarget([0, 0, 0], 50);
    const cage = createBoxCage([-45, -45, -45], [45, 45, 45], 3);
    const fixed = cage.vertices.map((_, i) => i % 7 === 0);
    const fit = fitCageToTarget(cage, target, { fixed, iterations: 5 });
    fixed.forEach((f, i) => {
      if (f) expect(fit.cage.vertices[i]).toEqual(cage.vertices[i]);
    });
    expect(fit.rms).toBeLessThan(surfaceDeviation(cage, target, 2).rms);
    // qm.fixed is honoured as well
    const c2 = { ...cage, fixed };
    const fit2 = fitCageToTarget(c2, target, { iterations: 2 });
    fixed.forEach((f, i) => {
      if (f) expect(fit2.cage.vertices[i]).toEqual(cage.vertices[i]);
    });
  });

  it("smoothness trades accuracy for fairness monotonically", () => {
    const target = sphereTarget([0, 0, 0], 50);
    const cage = createBoxCage([-45, -45, -45], [45, 45, 45], 4);
    const a = fitCageToTarget(cage, target, { smoothness: 0.001, iterations: 8 });
    const b = fitCageToTarget(cage, target, { smoothness: 1, iterations: 8 });
    expect(a.rms).toBeLessThan(b.rms);
  });

  it("boundaryTarget pulls the open boundary onto a curve", () => {
    // fit a planar grid to the plane z = 0 while its boundary is pulled onto the cylinder r = 40
    const plane = planeTarget([0, 0, 0], [0, 0, 1]);
    const circle: ClosestPointQuery = (p) => {
      const r = Math.hypot(p[0], p[1]) || 1;
      const q: Vec3 = [(40 * p[0]) / r, (40 * p[1]) / r, 0];
      return { point: q, normal: [p[0] / r, p[1] / r, 0], distance: Math.hypot(p[0] - q[0], p[1] - q[1], p[2]) };
    };
    const cage = createGridCage(6, 6, { origin: [-35, -35, 0], uAxis: [70, 0, 0], vAxis: [0, 70, 0] });
    const fit = fitCageToTarget(cage, plane, { boundaryTarget: circle, boundaryWeight: 10, iterations: 15, smoothness: 0.01 });
    const st = subdivisionStencils(fit.cage, 2);
    const Q = csrMulVec3(st.rows, fit.cage.vertices);
    let before = 0, after = 0;
    const Q0 = csrMulVec3(subdivisionStencils(cage, 2).rows, cage.vertices);
    for (let s = 0; s < Q.length; s++) {
      if (!st.boundarySample[s]) continue;
      before = Math.max(before, Math.abs(Math.hypot(Q0[s][0], Q0[s][1]) - 40));
      after = Math.max(after, Math.abs(Math.hypot(Q[s][0], Q[s][1]) - 40));
    }
    expect(after).toBeLessThan(0.3 * before);
    expect(fit.rms).toBeLessThan(1e-6); // the surface stays in the plane
  });
});

describe("snapVerticesToTarget", () => {
  it("moves cage vertices so that their limit points lie on the target", () => {
    const target = sphereTarget([0, 0, 0], 50);
    const cage = createBoxCage([-30, -30, -30], [30, 30, 30], 3);
    const snapped = snapVerticesToTarget(cage, target, { iterations: 60 });
    const L = limitPoints(snapped);
    let worst = 0;
    for (const p of L) worst = Math.max(worst, Math.abs(Math.hypot(p[0], p[1], p[2]) - 50));
    expect(worst).toBeLessThan(1e-6);
    // ray casting along the limit normal gives the same result on a sphere
    const snapped2 = snapVerticesToTarget(cage, target, { iterations: 60, raycast: sphereRaycast([0, 0, 0], 50) });
    let worst2 = 0;
    for (const p of limitPoints(snapped2)) worst2 = Math.max(worst2, Math.abs(Math.hypot(p[0], p[1], p[2]) - 50));
    expect(worst2).toBeLessThan(1e-6);
  });

  it("open grid: boundary and corner vertices snap too; fixed vertices stay", () => {
    const hf = wavyHeightField(3, 1 / 8, 1 / 10);
    const target = heightFieldTarget(hf);
    const cage = createGridCage(6, 6, { origin: [-30, -30, 0], uAxis: [60, 0, 0], vAxis: [0, 60, 0] });
    const fixed = cage.vertices.map((_, i) => i === 24);
    const snapped = snapVerticesToTarget(cage, target, { iterations: 80, fixed });
    expect(snapped.vertices[24]).toEqual(cage.vertices[24]);
    const L = limitPoints(snapped);
    let worst = 0;
    L.forEach((p, i) => {
      if (i === 24) return;
      worst = Math.max(worst, target(p)!.distance);
    });
    expect(worst).toBeLessThan(0.02);
  });
});
