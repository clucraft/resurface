import { describe, expect, it } from "vitest";
import type { Vec3 } from "../core/types";
import { planeTarget, sphereTarget } from "./analyticTargets";
import { cageQuality, surfaceDeviation } from "./analysis";
import { autoCageFromMesh } from "./autosurface";
import { limitMesh, limitNormals, limitPoints, subdivide, subdivisionStencils } from "./catmullClark";
import { fitCageToTarget, snapVerticesToTarget } from "./fit";
import { mergeRegularRegions, toBicubicPatches } from "./patches";
import { addVertex, createGridCage, validate, type QuadMesh } from "./quadMesh";

const finite = (pts: Vec3[]) => pts.every((p) => p.every(Number.isFinite));

function cases(): [string, QuadMesh][] {
  const single: QuadMesh = { vertices: [[0, 0, 0], [10, 0, 0], [10, 10, 1], [0, 10, 0]], faces: [[0, 1, 2, 3]] };
  const tri: QuadMesh = { vertices: [[0, 0, 0], [10, 0, 0], [0, 10, 0]], faces: [[0, 1, 2]] };
  // three quads sharing one edge (non-manifold "book")
  const book: QuadMesh = {
    vertices: [[0, 0, 0], [0, 0, 10], [10, 0, 0], [10, 0, 10], [-5, 8, 0], [-5, 8, 10], [-5, -8, 0], [-5, -8, 10]],
    faces: [[0, 2, 3, 1], [0, 1, 5, 4], [1, 0, 6, 7]],
  };
  // two quads with inconsistent orientation
  const flipped = createGridCage(2, 1);
  flipped.faces[1] = flipped.faces[1].slice().reverse();
  // bowtie vertex
  const bow: QuadMesh = {
    vertices: [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0], [-10, 0, 0], [-10, -10, 0], [0, -10, 0]],
    faces: [[0, 1, 2, 3], [0, 4, 5, 6]],
  };
  // unused vertex + degenerate face entry
  const junk = createGridCage(2, 2);
  addVertex(junk, [500, 500, 500]);
  junk.faces.push([0, 0, 1]);
  junk.creaseEdges = [[0, 1], [1, 1], [99, 100]];
  // creased grid with a dart (single crease edge ending inside)
  const dart = createGridCage(3, 3);
  dart.creaseEdges = [[5, 6]];
  const empty: QuadMesh = { vertices: [], faces: [] };
  return [["single quad", single], ["triangle", tri], ["book", book], ["flipped", flipped], ["bowtie", bow], ["junk", junk], ["dart", dart], ["empty", empty]];
}

describe("robustness on degenerate / non-manifold cages", () => {
  for (const [name, qm] of cases()) {
    it(`${name}: no throw, finite output`, () => {
      validate(qm);
      cageQuality(qm);
      const s = subdivide(qm);
      expect(finite(s.vertices)).toBe(true);
      expect(finite(limitPoints(qm))).toBe(true);
      expect(finite(limitNormals(qm))).toBe(true);
      const lm = limitMesh(qm, 2);
      expect([...lm.positions].every(Number.isFinite)).toBe(true);
      expect([...lm.normals!].every(Number.isFinite)).toBe(true);
      const st = subdivisionStencils(qm, 2);
      expect(st.rows.nRows).toBeGreaterThanOrEqual(qm.vertices.length);
      const patches = toBicubicPatches(qm);
      for (const p of patches) expect(finite(p.controlPoints)).toBe(true);
      const merged = mergeRegularRegions(qm);
      for (const p of merged.surfaces) expect(finite(p.controlPoints)).toBe(true);
      const target = planeTarget([0, 0, 0], [0, 0, 1]);
      const fit = fitCageToTarget(qm, target, { iterations: 2 });
      expect(finite(fit.cage.vertices)).toBe(true);
      const snapped = snapVerticesToTarget(qm, target, { iterations: 3 });
      expect(finite(snapped.vertices)).toBe(true);
      surfaceDeviation(qm, target, 1);
    });
  }

  it("single quad: patch interpolates the corners (bilinear-like corner rule)", () => {
    const single: QuadMesh = { vertices: [[0, 0, 0], [10, 0, 0], [10, 10, 1], [0, 10, 0]], faces: [[0, 1, 2, 3]] };
    const [p] = toBicubicPatches(single);
    expect(p.controlPoints[0]).toEqual([0, 0, 0]);
    expect(p.controlPoints[15]).toEqual([10, 10, 1]);
  });

  it("targets that return null (holes) are ignored by the fit", () => {
    const sphere = sphereTarget([0, 0, 0], 50);
    const holey = (p: Vec3) => (p[2] > 30 ? null : sphere(p));
    const qm = autoCageFromMesh({
      positions: Float32Array.from([0, 0, 50, 50, 0, 0, 0, 50, 0, -50, 0, 0, 0, -50, 0, 0, 0, -50]),
      indices: Uint32Array.from([0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1, 5, 2, 1, 5, 3, 2, 5, 4, 3, 5, 1, 4]),
    });
    const fit = fitCageToTarget(qm, holey, { iterations: 4 });
    expect(Number.isFinite(fit.rms)).toBe(true);
    expect(fit.hits).toBeLessThan(fit.samples);
  });
});
