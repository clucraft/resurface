/**
 * Taubin λ|μ smoothing (low-pass filter without the shrinkage of plain Laplacian smoothing).
 *
 * The umbrella operator is evaluated per triangle (each triangle adds its two other corners to
 * every corner), so no adjacency structure is needed: per iteration it is two linear sweeps
 * over the triangles + one over the vertices. Interior edges are seen from both incident
 * triangles, so neighbours get equal weight; boundary edges get half weight (harmless — boundary
 * vertices are fixed by default). Differences (p_j − p_i) are accumulated rather than absolute
 * positions so float32 sums keep precision far from the origin.
 *
 * Memory: positions copy (12 B/vertex) + accumulators (16 B/vertex).
 */
import type { MeshData, ProgressFn } from "../core/types";
import { boundaryVertexMask, computeVertexNormals, triangleCountOf, vertexCountOf } from "./topology";

export interface TaubinOptions {
  iterations: number;
  /** positive shrink factor, default 0.5 */
  lambda?: number;
  /** negative inflate factor, |mu| slightly > lambda, default -0.53 */
  mu?: number;
  /** per-vertex: 1 = may move. Omitted → all vertices may move. */
  mask?: Uint8Array;
  /** keep boundary vertices fixed (default true) */
  preserveBoundary?: boolean;
  /** precomputed triangle neighbours (used only to find boundary vertices) */
  neighbors?: Int32Array;
  onProgress?: ProgressFn;
}

export function smoothTaubin(mesh: MeshData, opts: TaubinOptions): MeshData {
  const nv = vertexCountOf(mesh);
  const nt = triangleCountOf(mesh);
  const I = mesh.indices;
  const P = mesh.positions.slice();
  const lambda = opts.lambda ?? 0.5;
  const mu = opts.mu ?? -0.53;
  const iterations = Math.max(0, Math.floor(opts.iterations));

  // movable flags & inverse neighbour weights
  const movable = new Uint8Array(nv);
  if (opts.mask) for (let v = 0; v < nv; v++) movable[v] = opts.mask[v] ? 1 : 0;
  else movable.fill(1);
  if (opts.preserveBoundary ?? true) {
    const b = boundaryVertexMask(mesh, opts.neighbors);
    for (let v = 0; v < nv; v++) if (b[v]) movable[v] = 0;
  }
  const invW = new Float32Array(nv);
  for (let c = 0; c < nt * 3; c++) invW[I[c]] += 2;
  for (let v = 0; v < nv; v++) invW[v] = invW[v] > 0 ? 1 / invW[v] : 0;

  const acc = new Float32Array(nv * 3);
  const totalSteps = iterations * 2;
  let step = 0;
  for (let it = 0; it < iterations; it++) {
    for (let pass = 0; pass < 2; pass++) {
      const f = pass === 0 ? lambda : mu;
      acc.fill(0);
      for (let t = 0; t < nt; t++) {
        const a = 3 * I[3 * t], b = 3 * I[3 * t + 1], c = 3 * I[3 * t + 2];
        const ax = P[a], ay = P[a + 1], az = P[a + 2];
        const bx = P[b], by = P[b + 1], bz = P[b + 2];
        const cx = P[c], cy = P[c + 1], cz = P[c + 2];
        acc[a] += bx + cx - 2 * ax;
        acc[a + 1] += by + cy - 2 * ay;
        acc[a + 2] += bz + cz - 2 * az;
        acc[b] += ax + cx - 2 * bx;
        acc[b + 1] += ay + cy - 2 * by;
        acc[b + 2] += az + cz - 2 * bz;
        acc[c] += ax + bx - 2 * cx;
        acc[c + 1] += ay + by - 2 * cy;
        acc[c + 2] += az + bz - 2 * cz;
      }
      for (let v = 0; v < nv; v++) {
        if (!movable[v]) continue;
        const s = f * invW[v];
        P[3 * v] += s * acc[3 * v];
        P[3 * v + 1] += s * acc[3 * v + 1];
        P[3 * v + 2] += s * acc[3 * v + 2];
      }
      step++;
      opts.onProgress?.(step / totalSteps, "Smoothing");
    }
  }
  const out: MeshData = { positions: P, indices: mesh.indices.slice() };
  if (mesh.normals) out.normals = computeVertexNormals(out);
  if (mesh.colors) out.colors = mesh.colors.slice();
  return out;
}

/**
 * Plain Laplacian (umbrella) smoothing, same conventions as smoothTaubin (shrinks!). Mostly
 * useful for small masked regions.
 */
export function smoothLaplacian(mesh: MeshData, opts: Omit<TaubinOptions, "mu" | "lambda"> & { factor?: number }): MeshData {
  return smoothTaubin(mesh, { ...opts, lambda: opts.factor ?? 0.5, mu: 0 });
}
