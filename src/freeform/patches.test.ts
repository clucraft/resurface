import { describe, expect, it } from "vitest";
import type { BSplineSurfaceDef, Vec3 } from "../core/types";
import { cross, dot, length, normalize, sub } from "../core/vec";
import { heightFieldMesh, icosphereMesh, wavyHeightField } from "./analyticTargets";
import { autoCageFromMesh } from "./autosurface";
import { subdivisionStencils } from "./catmullClark";
import {
  bsplineGridToBezier,
  clampUniformCubic,
  evaluateSurface,
  evaluateSurfaceDerivs,
  mergeRegularRegions,
  regularFaceSplineGrids,
  toBicubicPatchesEx,
  type PatchSet,
} from "./patches";
import {
  buildTopology,
  cageSize,
  createBoxCage,
  createGridCage,
  createTorusCage,
  removeFaces,
  removeUnusedVertices,
  setCrease,
  type QuadMesh,
} from "./quadMesh";
import { csrMulVec3 } from "./sparse";

const dist = (a: Vec3, b: Vec3) => length(sub(a, b));

function bumpyGrid(n = 4): QuadMesh {
  const g = createGridCage(n, n, { uAxis: [40, 0, 0], vAxis: [0, 40, 0] });
  g.vertices = g.vertices.map(([x, y]) => [x + 0.7 * Math.sin(y), y, 6 * Math.sin(x / 9) * Math.cos(y / 11)] as Vec3);
  return g;
}

function mixedCage(): QuadMesh {
  const g = createGridCage(2, 1, { uAxis: [20, 0, 0], vAxis: [0, 10, 0] });
  const apex = g.vertices.length;
  g.vertices.push([5, 18, 3]);
  const p1 = g.vertices.length;
  g.vertices.push([16, 20, -2], [22, 14, 1]);
  g.faces.push([3, 4, apex], [4, 5, p1 + 1, p1, apex]);
  return g;
}

/** Box cage with some asymmetry so that nothing is accidentally exact. */
function wobblyBox(n = 2): QuadMesh {
  const b = createBoxCage([-20, -20, -20], [20, 20, 20], n);
  b.vertices = b.vertices.map(([x, y, z]) => [x * (1 + 0.1 * Math.sin(z / 7)), y + 0.15 * x, z * (1 + 0.05 * Math.cos(x / 9))] as Vec3);
  return b;
}

/** Three quads fanning around a boundary vertex (irregular boundary sector), with height variation. */
function boundaryFan(): QuadMesh {
  const vertices: Vec3[] = [[0, 0, 0]];
  for (let i = 0; i <= 3; i++) {
    const a = (i * Math.PI) / 3;
    vertices.push([10 * Math.cos(a), 10 * Math.sin(a), 2 * Math.sin(2 * a)]);
  }
  for (let i = 0; i < 3; i++) {
    const a = ((i + 0.5) * Math.PI) / 3;
    vertices.push([14 * Math.cos(a), 14 * Math.sin(a), 3 * Math.cos(a)]);
  }
  const faces = [0, 1, 2].map((i) => [0, 1 + i, 5 + i, 2 + i]);
  return { vertices, faces };
}

/** 6x6 sheet with the top-right quarter removed: the reflex boundary corner has 3 faces. */
function lShape(): QuadMesh {
  const g = createGridCage(6, 6, { uAxis: [60, 0, 0], vAxis: [0, 60, 0] });
  const del: number[] = [];
  g.faces.forEach((f, i) => {
    const c = g.vertices[f[0]];
    if (c[0] >= 30 && c[1] >= 30) del.push(i);
  });
  removeFaces(g, del);
  removeUnusedVertices(g);
  g.vertices = g.vertices.map(([x, y]) => [x, y, 4 * Math.sin(x / 12) * Math.cos(y / 15)] as Vec3);
  return g;
}

/** Parameter of edge k of a Bézier patch at t (corner k -> corner k+1). */
function edgeParam(k: number, t: number): [number, number] {
  switch (k) {
    case 0: return [t, 0];
    case 1: return [1, t];
    case 2: return [1 - t, 1];
    default: return [0, 1 - t];
  }
}

/** Max gap and max normal angle (deg) between neighbouring patches along all shared edges. */
function seamStats(ps: PatchSet, samples = 16): { gap: number; maxAngleDeg: number; maxAngleRegularDeg: number } {
  const topo = buildTopology(ps.mesh);
  const patchOfFace = new Int32Array(topo.nf).fill(-1);
  ps.faceOfPatch.forEach((f, i) => (patchOfFace[f] = i));
  let gap = 0, maxAng = 0, maxAngReg = 0;
  for (let h = 0; h < topo.nh; h++) {
    const t = topo.heTwin[h];
    if (t < 0 || t < h) continue;
    const sharp = topo.edgeSharp[topo.heEdge[h]] !== 0;
    const fa = topo.heF[h], fb = topo.heF[t];
    const ka = h - topo.faceStart[fa], kb = t - topo.faceStart[fb];
    const A = ps.patches[patchOfFace[fa]], B = ps.patches[patchOfFace[fb]];
    const bothExact = ps.exact[patchOfFace[fa]] && ps.exact[patchOfFace[fb]];
    for (let i = 0; i <= samples; i++) {
      const s = i / samples;
      const [ua, va] = edgeParam(ka, s);
      const [ub, vb] = edgeParam(kb, 1 - s);
      const ea = evaluateSurfaceDerivs(A, ua, va), eb = evaluateSurfaceDerivs(B, ub, vb);
      gap = Math.max(gap, dist(ea.p, eb.p));
      const na = normalize(cross(ea.du, ea.dv)), nb = normalize(cross(eb.du, eb.dv));
      if (!sharp && length(na) > 0.5 && length(nb) > 0.5) {
        const ang = (Math.acos(Math.max(-1, Math.min(1, dot(na, nb)))) * 180) / Math.PI;
        maxAng = Math.max(maxAng, ang);
        if (bothExact) maxAngReg = Math.max(maxAngReg, ang);
      }
    }
  }
  return { gap, maxAngleDeg: maxAng, maxAngleRegularDeg: maxAngReg };
}

/** Max distance between patches (evaluated at the sample uv) and the exact limit samples. */
function limitDeviation(qm: QuadMesh, ps: PatchSet, levels = 3, onlyExact = false): number {
  const st = subdivisionStencils(ps.mesh, levels);
  const Q = csrMulVec3(st.rows, ps.mesh.vertices);
  const patchOfFace = new Int32Array(ps.mesh.faces.length).fill(-1);
  ps.faceOfPatch.forEach((f, i) => (patchOfFace[f] = i));
  let maxErr = 0;
  for (let s = 0; s < Q.length; s++) {
    const p = patchOfFace[st.faceOfSample[s]];
    if (onlyExact && !ps.exact[p]) continue;
    const e = dist(evaluateSurface(ps.patches[p], st.uvOfSample[2 * s], st.uvOfSample[2 * s + 1]), Q[s]);
    maxErr = Math.max(maxErr, e);
  }
  void qm;
  return maxErr;
}

/** The 4 control points of edge k of a Bézier patch, from corner k to corner k+1. */
function edgeControlPoints(s: BSplineSurfaceDef, k: number): Vec3[] {
  const idx = [0, 1, 2, 3].map((t) => (k === 0 ? t * 4 : k === 1 ? 12 + t : k === 2 ? (3 - t) * 4 + 3 : 3 - t));
  return idx.map((i) => s.controlPoints[i]);
}

describe("bicubic patch export", () => {
  it("neighbouring patches share bit-identical boundary control points", () => {
    for (const qm of [wobblyBox(3), mixedCage(), bumpyGrid(3)]) {
      for (const g1Seams of [true, false]) {
        const ps = toBicubicPatchesEx(qm, { g1Seams });
        const topo = buildTopology(ps.mesh);
        let checked = 0;
        for (let h = 0; h < topo.nh; h++) {
          const t = topo.heTwin[h];
          if (t < 0) continue;
          const fa = topo.heF[h], fb = topo.heF[t];
          const A = edgeControlPoints(ps.patches[fa], h - topo.faceStart[fa]);
          const B = edgeControlPoints(ps.patches[fb], t - topo.faceStart[fb]).reverse();
          expect(A).toEqual(B);
          checked++;
        }
        expect(checked).toBeGreaterThan(0);
      }
    }
  });

  it("B-spline evaluation: derivatives match finite differences; clamping preserves the curve", () => {
    const ps = toBicubicPatchesEx(wobblyBox(1));
    const s = ps.patches[3];
    const e = evaluateSurfaceDerivs(s, 0.3, 0.6);
    const h = 1e-6;
    const fu = sub(evaluateSurface(s, 0.3 + h, 0.6), evaluateSurface(s, 0.3 - h, 0.6)).map((x) => x / (2 * h));
    const fv = sub(evaluateSurface(s, 0.3, 0.6 + h), evaluateSurface(s, 0.3, 0.6 - h)).map((x) => x / (2 * h));
    for (let c = 0; c < 3; c++) {
      expect(Math.abs(e.du[c] - fu[c])).toBeLessThan(1e-5);
      expect(Math.abs(e.dv[c] - fv[c])).toBeLessThan(1e-5);
    }
    // uniform cubic B-spline (7 control points, 4 spans) vs clamped representation
    const ctrl: Vec3[] = [[0, 0, 0], [1, 2, 0], [3, 3, 1], [4, 0, 2], [6, -1, 1], [7, 2, 0], [9, 1, -1]];
    const clamped = clampUniformCubic(ctrl);
    const a = ctrl.length - 3;
    const surf: BSplineSurfaceDef = {
      degreeU: 3, degreeV: 1, nu: ctrl.length, nv: 2,
      knotsU: [0, 0, 0, ...Array.from({ length: a + 1 }, (_, k) => k), a, a, a], knotsV: [0, 0, 1, 1],
      controlPoints: clamped.flatMap((p) => [p, p]),
    };
    const uniform = (u: number): Vec3 => {
      const span = Math.min(a - 1, Math.floor(u));
      const t = u - span;
      const b = [(1 - t) ** 3 / 6, (3 * t ** 3 - 6 * t ** 2 + 4) / 6, (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6, t ** 3 / 6];
      const p: Vec3 = [0, 0, 0];
      for (let k = 0; k < 4; k++) for (let c = 0; c < 3; c++) p[c] += b[k] * ctrl[span + k][c];
      return p;
    };
    for (let i = 0; i <= 40; i++) {
      const u = (a * i) / 40;
      expect(dist(evaluateSurface(surf, u, 0.5), uniform(u))).toBeLessThan(1e-12);
    }
  });

  it("regular faces reproduce the limit surface exactly (torus, open bumpy grid, fully creased box)", () => {
    const creased = createBoxCage([0, 0, 0], [30, 20, 10], [3, 2, 2]);
    const topo = buildTopology(creased);
    const allEdges: [number, number][] = [];
    for (let e = 0; e < topo.ne; e++) {
      const a = creased.vertices[topo.edgeV[2 * e]], b = creased.vertices[topo.edgeV[2 * e + 1]];
      // cube edges: both endpoints on two box planes
      const onPlanes = (p: Vec3) => [p[0] === 0 || p[0] === 30, p[1] === 0 || p[1] === 20, p[2] === 0 || p[2] === 10].filter(Boolean).length >= 2;
      if (onPlanes(a) && onPlanes(b)) allEdges.push([topo.edgeV[2 * e], topo.edgeV[2 * e + 1]]);
    }
    setCrease(creased, allEdges, true);
    for (const qm of [createTorusCage(30, 10, 10, 8), bumpyGrid(4), creased]) {
      const ps = toBicubicPatchesEx(qm);
      expect(ps.patches.length).toBe(qm.faces.length);
      expect([...ps.exact].every((x) => x === 1)).toBe(true);
      const err = limitDeviation(qm, ps, 3);
      expect(err).toBeLessThan(1e-9 * cageSize(qm));
      // patches equal the Bézier form of the uniform B-spline net of each face
      const grids = regularFaceSplineGrids(qm);
      grids.forEach((g, f) => {
        const B = bsplineGridToBezier(g!);
        const cp = ps.patches[f].controlPoints;
        for (let k = 0; k < 16; k++) expect(dist(B[k], cp[k])).toBeLessThan(1e-9 * cageSize(qm));
      });
      const seams = seamStats(ps);
      expect(seams.gap).toBe(0);
      expect(seams.maxAngleRegularDeg).toBeLessThan(1e-4);
    }
  });

  it("ACC patches are watertight (bit-identical seams) and close to the limit surface", () => {
    const report: string[] = [];
    const results: { tpe: boolean; gap: number; rel: number; ang: number; angReg: number }[] = [];
    for (const [name, qm] of [
      ["box 1x1x1 (8 EV of valence 3)", wobblyBox(1)],
      ["box 3x3x3", wobblyBox(3)],
      ["mixed tri/pentagon cage", mixedCage()],
      ["auto cage of an icosphere (valences 3..6)", autoCageFromMesh(icosphereMesh(30, 1))],
      ["boundary fan (crease sector with 3 faces)", boundaryFan()],
      ["L-shaped sheet (reflex boundary corner)", lShape()],
      ["auto cage of an open height field", autoCageFromMesh(heightFieldMesh(wavyHeightField(3, 1 / 8, 1 / 10), -30, 30, -30, 30, 5, 5))],
    ] as [string, QuadMesh][]) {
      for (const tpe of [true, false]) {
        const ps = toBicubicPatchesEx(qm, { g1Seams: tpe });
        const seams = seamStats(ps, 24);
        const err = limitDeviation(qm, ps, 4);
        const rel = err / cageSize(qm);
        report.push(
          `${name} g1Seams=${tpe}: patches=${ps.patches.length} exact=${[...ps.exact].filter((x) => x).length} ` +
            `seam gap=${seams.gap.toExponential(2)} max seam normal angle=${seams.maxAngleDeg.toFixed(2)}deg ` +
            `max |patch - limit|/size=${rel.toExponential(2)}`,
        );
        results.push({ tpe, gap: seams.gap, rel, ang: seams.maxAngleDeg, angReg: seams.maxAngleRegularDeg });
      }
    }
    console.log(report.join("\n"));
    for (const r of results) {
      expect(r.gap).toBeLessThan(1e-12);
      expect(r.rel).toBeLessThan(r.tpe ? 0.015 : 0.025);
      // near-G1 seams with g1Seams, plain ACC is only C0 there
      if (r.tpe) expect(r.ang).toBeLessThan(1);
      // regular-regular seams are smooth
      expect(r.angReg).toBeLessThan(1e-4);
    }
  });

  it("g1Seams: all patches share the limit normal at extraordinary vertices", () => {
    const qm = wobblyBox(2);
    const ps = toBicubicPatchesEx(qm, { g1Seams: true });
    const topo = buildTopology(ps.mesh);
    // at every vertex, the corner normals of all incident patches agree
    const corner: [number, number][] = [[0, 0], [1, 0], [1, 1], [0, 1]];
    const normalsAt = new Map<number, Vec3[]>();
    ps.faceOfPatch.forEach((f, pi) => {
      for (let k = 0; k < 4; k++) {
        const v = topo.heV[topo.faceStart[f] + k];
        const e = evaluateSurfaceDerivs(ps.patches[pi], corner[k][0], corner[k][1]);
        const n = normalize(cross(e.du, e.dv));
        const l = normalsAt.get(v);
        if (l) l.push(n);
        else normalsAt.set(v, [n]);
      }
    });
    let worst = 0;
    for (const l of normalsAt.values()) for (const n of l) worst = Math.max(worst, Math.acos(Math.min(1, dot(n, l[0]))));
    expect((worst * 180) / Math.PI).toBeLessThan(1e-4);
  });

  it("mergeRegularRegions builds few large C2 surfaces that coincide with the per-face patches", () => {
    const torus = createTorusCage(30, 10, 12, 8);
    const merged = mergeRegularRegions(torus);
    expect(merged.surfaces.length).toBeLessThan(torus.faces.length / 4);
    const per = toBicubicPatchesEx(torus);
    // every merged surface: evaluate at face-local parameters and compare with that face's patch
    merged.faceGroups.forEach((group, si) => {
      const s = merged.surfaces[si];
      if (group.length === 1 && s.nu === 4) return;
      const W = s.knotsU[s.knotsU.length - 1], H = s.knotsV[s.knotsV.length - 1];
      expect(W * H).toBe(group.length);
      // locate each face of the group by evaluating the merged surface at the cell centre
      for (let i = 0; i < W; i++)
        for (let j = 0; j < H; j++) {
          const pc = evaluateSurface(s, i + 0.5, j + 0.5);
          let best = Infinity;
          for (const f of group) best = Math.min(best, dist(pc, evaluateSurface(per.patches[f], 0.5, 0.5)));
          expect(best).toBeLessThan(1e-9);
        }
    });
    // grid with an extraordinary region: box -> merged surfaces + ACC patches
    const box = wobblyBox(4);
    const mb = mergeRegularRegions(box);
    const covered = mb.faceGroups.reduce((s, g) => s + g.length, 0);
    expect(covered).toBe(box.faces.length);
    expect(mb.surfaces.length).toBeLessThan(box.faces.length);
    console.log(`merge: torus 12x8 -> ${merged.surfaces.length} surface(s); box 4x4x4 (${box.faces.length} faces) -> ${mb.surfaces.length} surfaces (${mb.mergedCount} merged)`);
  });
});
