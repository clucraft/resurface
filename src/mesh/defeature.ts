/**
 * De-feature ("remove stamps / markers / engravings in one click"): delete a triangle selection
 * and fill the resulting hole(s) with a faired, curvature-continuous patch.
 */
import type { MeshData, ProgressFn } from "../core/types";
import { fillBoundaryLoops, type FillHolesResult } from "./holes";
import { compactMeshWithMap, findBoundaryLoops, triangleCountOf, vertexCountOf } from "./topology";

export interface DefeatureOptions {
  /** curvature-continuous (bi-Laplacian) fill; false = membrane fill. Default true. */
  smooth?: boolean;
  /** insert interior vertices matching the surrounding density (default true) */
  refine?: boolean;
  onProgress?: ProgressFn;
}

/**
 * Delete the masked triangles and fill the holes this creates. Only loops whose vertices all
 * touched deleted triangles are filled; pre-existing holes and open borders stay open. A loop
 * where the deleted region merged with an existing boundary (selection reaching the mesh border)
 * is not filled and is counted in stats.skipped.
 *
 * For best results select the feature plus a small margin of clean surface around it: the
 * patch blends tangent- and curvature-continuously into the surface around the hole boundary.
 */
export function defeature(mesh: MeshData, triMask: Uint8Array, opts: DefeatureOptions = {}): FillHolesResult {
  const I = mesh.indices;
  const nt = triangleCountOf(mesh);
  const nv = vertexCountOf(mesh);
  const touched = new Uint8Array(nv);
  const keep = new Uint8Array(nt);
  for (let t = 0; t < nt; t++) {
    if (triMask[t]) touched[I[3 * t]] = touched[I[3 * t + 1]] = touched[I[3 * t + 2]] = 1;
    else keep[t] = 1;
  }
  const { mesh: cut, vertexMap } = compactMeshWithMap(mesh, keep);
  const touchedNew = new Uint8Array(vertexCountOf(cut));
  for (let v = 0; v < nv; v++) if (vertexMap[v] >= 0 && touched[v]) touchedNew[vertexMap[v]] = 1;
  const loops = findBoundaryLoops(cut);
  const chosen: Uint32Array[] = [];
  let skipped = 0;
  for (const L of loops) {
    let all = true, any = false;
    for (let i = 0; i < L.length; i++) {
      if (touchedNew[L[i]]) any = true;
      else all = false;
    }
    if (all) chosen.push(L);
    else if (any) skipped++;
  }
  const res = fillBoundaryLoops(cut, chosen, { smooth: opts.smooth ?? true, refine: opts.refine ?? true, onProgress: opts.onProgress });
  res.stats.holesFound = loops.length;
  res.stats.skipped += skipped;
  return res;
}
