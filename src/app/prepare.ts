/**
 * Mesh preparation shared by the UI thread and compute workers: order triangles along a
 * space-filling curve (contiguous ranges become spatially coherent render chunks), compute
 * vertex normals, and move buffers to shared memory when available.
 */
import type { MeshData } from "../core/types";
import { computeVertexNormals } from "../mesh/topology";
import { reorderTriangles, spatialOrderTriangles } from "../mesh/spatial";
import { toShared } from "./shared";

export function prepareMeshData(mesh: MeshData, opts: { keepOrder?: boolean } = {}): MeshData {
  let m = mesh;
  const triCount = m.indices.length / 3;
  if (triCount > 1 && !opts.keepOrder) m = reorderTriangles(m, spatialOrderTriangles(m));
  const normals = m.normals && m.normals.length === m.positions.length ? m.normals : computeVertexNormals(m);
  return {
    positions: toShared(m.positions),
    indices: toShared(m.indices),
    normals: toShared(normals),
    colors: m.colors ? toShared(m.colors) : undefined,
  };
}
