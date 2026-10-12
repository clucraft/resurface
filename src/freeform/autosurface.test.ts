import { describe, expect, it } from "vitest";
import type { MeshData } from "../core/types";
import { heightFieldMesh, heightFieldTarget, icosphereMesh, sphereTarget, wavyHeightField } from "./analyticTargets";
import { cageQuality, deviationColors, patchDeviation, surfaceDeviation } from "./analysis";
import { autoCageFromMesh, autosurface, pairTrianglesToQuads } from "./autosurface";
import { createBoxCage, createGridCage, orient, quadMeshFromTriangles, signedVolume, validate, weldVertices } from "./quadMesh";

/** Explode a mesh into an STL-like triangle soup (3 unique vertices per triangle). */
function soup(m: MeshData): MeshData {
  const nt = m.indices.length / 3;
  const positions = new Float32Array(nt * 9);
  for (let k = 0; k < m.indices.length; k++) positions.set(m.positions.subarray(3 * m.indices[k], 3 * m.indices[k] + 3), 3 * k);
  return { positions, indices: Uint32Array.from({ length: nt * 3 }, (_, i) => i) };
}

describe("autoCageFromMesh", () => {
  it("closed icosphere -> valid all-quad closed cage, pairing reduces extraordinary vertices", () => {
    const ico = icosphereMesh(50, 2); // 320 triangles
    const cage = autoCageFromMesh(ico);
    const v = validate(cage);
    expect(v.ok).toBe(true);
    expect(v.closed).toBe(true);
    expect(v.eulerCharacteristic).toBe(2);
    expect(cage.faces.every((f) => f.length === 4)).toBe(true);
    expect(signedVolume(cage)).toBeGreaterThan(0);
    const q = cageQuality(cage);
    const plain = autoCageFromMesh(ico, { quadDominant: false });
    const qp = cageQuality(plain);
    expect(validate(plain).ok).toBe(true);
    expect(q.faceCount).toBeLessThan(qp.faceCount);
    expect(q.extraordinaryCount).toBeLessThan(qp.extraordinaryCount);
    // pairing statistics
    let tri = quadMeshFromTriangles(ico);
    weldVertices(tri, 1e-6);
    orient(tri);
    const paired = pairTrianglesToQuads(tri);
    const leftover = paired.faces.filter((f) => f.length === 3).length;
    expect(validate(paired).ok).toBe(true);
    console.log(
      `icosphere 320 tris: paired -> ${paired.faces.length - leftover} quads + ${leftover} tris; ` +
        `cage ${q.faceCount} quads, ${q.extraordinaryCount} EVs (min angle ${q.minAngleDeg.toFixed(1)}°) vs ` +
        `tri-split cage ${qp.faceCount} quads, ${qp.extraordinaryCount} EVs`,
    );
  });

  it("welds STL-style triangle soups and repairs inverted orientation", () => {
    const ico = icosphereMesh(50, 1);
    const ref = autoCageFromMesh(ico);
    const s = soup(ico);
    const c = autoCageFromMesh(s);
    expect(c.vertices.length).toBe(ref.vertices.length);
    expect(c.faces.length).toBe(ref.faces.length);
    // inverted winding
    const inv: MeshData = { positions: ico.positions, indices: Uint32Array.from(ico.indices) };
    for (let t = 0; t < inv.indices.length; t += 3) [inv.indices[t + 1], inv.indices[t + 2]] = [inv.indices[t + 2], inv.indices[t + 1]];
    const ci = autoCageFromMesh(inv);
    expect(validate(ci).ok).toBe(true);
    expect(signedVolume(ci)).toBeGreaterThan(0);
  });

  it("open meshes keep a single boundary loop; quad-dominant mode keeps triangles", () => {
    const hf = wavyHeightField(3, 1 / 8, 1 / 10);
    const m = heightFieldMesh(hf, -50, 50, -50, 50, 10, 10);
    const cage = autoCageFromMesh(m);
    const v = validate(cage);
    expect(v.ok).toBe(true);
    expect(v.boundaryLoopCount).toBe(1);
    expect(cage.faces.every((f) => f.length === 4)).toBe(true);
    const qd = autoCageFromMesh(m, { split: false });
    expect(validate(qd).ok).toBe(true);
    expect(qd.faces.filter((f) => f.length === 4).length).toBeGreaterThan(qd.faces.length / 2);
  });
});

describe("autosurface", () => {
  it("sphere: cage + fit + patches with sub-0.05 rms", () => {
    const target = sphereTarget([0, 0, 0], 50);
    const ico = icosphereMesh(50, 2);
    const t0 = performance.now();
    const steps: number[] = [];
    const res = autosurface(ico, target, { onProgress: (f) => steps.push(f) });
    const ms = performance.now() - t0;
    expect(res.rms).toBeLessThan(0.05);
    expect(res.patches.length).toBe(res.cage.faces.length);
    expect(steps[steps.length - 1]).toBe(1);
    const pd = patchDeviation(res.patches, target, 6);
    expect(pd.maxAbs).toBeLessThan(0.5);
    const merged = autosurface(ico, target, { mergeRegular: true, iterations: 3 });
    expect(merged.patches.length).toBeLessThanOrEqual(res.patches.length);
    console.log(
      `autosurface sphere (icosphere 320 tris): cage ${res.cage.faces.length} quads, ${res.quality.extraordinaryCount} EVs, ` +
        `rms=${res.rms.toExponential(3)} max=${res.maxDev.toExponential(3)}, patches=${res.patches.length} ` +
        `(patch rms=${pd.rms.toExponential(3)} max=${pd.maxAbs.toExponential(3)}), merged surfaces=${merged.patches.length}, time=${ms.toFixed(0)}ms`,
    );
  });

  it("open wavy height field", () => {
    const hf = wavyHeightField(3, 1 / 8, 1 / 10);
    const target = heightFieldTarget(hf);
    const m = heightFieldMesh(hf, -50, 50, -50, 50, 10, 10);
    const res = autosurface(m, target);
    expect(res.rms).toBeLessThan(0.05);
    expect(validate(res.cage).boundaryLoopCount).toBe(1);
    console.log(`autosurface height field (200 tris): cage ${res.cage.faces.length} quads, rms=${res.rms.toExponential(3)} max=${res.maxDev.toExponential(3)}`);
  });
});

describe("analysis", () => {
  it("cageQuality", () => {
    const box = createBoxCage([0, 0, 0], [1, 1, 1], 2);
    const q = cageQuality(box);
    expect(q.extraordinaryCount).toBe(8);
    expect(q.nonQuadCount).toBe(0);
    expect(q.minAngleDeg).toBeCloseTo(90, 10);
    expect(q.maxAngleDeg).toBeCloseTo(90, 10);
    expect(q.valenceHistogram[3]).toBe(8);
    expect(q.valenceHistogram[4]).toBe(box.vertices.length - 8);
    const g = cageQuality(createGridCage(3, 3));
    expect(g.extraordinaryCount).toBe(0);
    expect(g.boundaryVertexCount).toBe(12);
  });

  it("surfaceDeviation is signed and matches the stencil samples; colour map", () => {
    const target = sphereTarget([0, 0, 0], 50);
    const box = createBoxCage([-25, -25, -25], [25, 25, 25], 2);
    const d = surfaceDeviation(box, target, 2);
    expect(d.distances.length).toBe(d.positions.length / 3);
    // the CC surface of a box cage of half-size 25 lies inside the sphere of radius 50 -> negative
    expect(Math.max(...d.distances)).toBeLessThan(0);
    expect(d.misses).toBe(0);
    const col = deviationColors([-1, 0, 1, NaN], 1);
    expect([...col]).toEqual([0, 0, 255, 0, 255, 0, 255, 0, 0, 128, 128, 128]);
  });
});
