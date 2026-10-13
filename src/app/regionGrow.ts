/**
 * Primitive-aware region growing for "smart" selection: fit the surface type around the seed
 * (plane / cylinder / cone / sphere), then flood-fill over edge neighbours while triangles stay
 * within tolerance of the (periodically refitted) primitive and their normals agree with it.
 * Stops reliably at fillet boundaries and feature edges even when the surface is tangent
 * continuous. Falls back to curvature-based growing for freeform regions.
 */
import type { MeshData, PrimitiveDef, Vec3 } from "../core/types";
import { detectBestPrimitive, distanceKernel, primitiveNormalAt, type PrimitiveKind, type RankedPrimitive } from "../fit/primitives";
import { smartSelect } from "../mesh/select";

export interface RegionGrowResult {
  mask: Uint8Array;
  count: number;
  kind: PrimitiveKind | "freeform";
  def: PrimitiveDef | null;
  rms: number;
}

export interface RegionGrowOptions {
  angleTolDeg: number;
  /** distance tolerance multiplier on the seed fit RMS (default 4) */
  tolFactor?: number;
  maxTriangles?: number;
}

/** BFS over triangle neighbours collecting up to `max` triangles around the seed. */
function seedPatch(nbr: Int32Array, seed: number, max: number): number[] {
  const out = [seed];
  const seen = new Set<number>([seed]);
  for (let i = 0; i < out.length && out.length < max; i++) {
    const t = out[i];
    for (let k = 0; k < 3; k++) {
      const u = nbr[3 * t + k];
      if (u >= 0 && !seen.has(u)) {
        seen.add(u);
        out.push(u);
      }
    }
  }
  return out;
}

function verticesOf(mesh: MeshData, tris: ArrayLike<number>, maxVerts = 60000): Uint32Array {
  const I = mesh.indices;
  const set = new Set<number>();
  const step = Math.max(1, Math.floor(tris.length / maxVerts));
  for (let i = 0; i < tris.length; i += step) {
    const t = tris[i];
    set.add(I[3 * t]);
    set.add(I[3 * t + 1]);
    set.add(I[3 * t + 2]);
  }
  return Uint32Array.from(set);
}

function fitBest(mesh: MeshData, verts: Uint32Array, kinds?: PrimitiveKind[]) {
  const ranked = detectBestPrimitive(mesh.positions, { indices: verts, normals: mesh.normals, kinds, complexityPenalty: 0.25 });
  return ranked[0] ?? null;
}

function extentOf(P: ArrayLike<number>, verts: Uint32Array): number {
  let ext = 0;
  const c: Vec3 = [P[3 * verts[0]], P[3 * verts[0] + 1], P[3 * verts[0] + 2]];
  for (const v of verts) ext = Math.max(ext, Math.hypot(P[3 * v] - c[0], P[3 * v + 1] - c[1], P[3 * v + 2] - c[2]));
  return ext;
}

export function primitiveRegionGrow(mesh: MeshData, nbr: Int32Array, seed: number, opts: RegionGrowOptions): RegionGrowResult | null {
  const P = mesh.positions, I = mesh.indices;
  const nt = I.length / 3;
  // Multi-scale seed: a large patch identifies the surface type more reliably, but on narrow
  // faces (thin ribs, grille slats) it spills over the edges — then fall back to smaller patches.
  let best: RankedPrimitive | null = null;
  let ext = 0;
  for (const size of [400, 160, 60]) {
    const patch = seedPatch(nbr, seed, size);
    if (patch.length < 20) continue;
    const verts = verticesOf(mesh, patch);
    const e = extentOf(P, verts);
    const fit = fitBest(mesh, verts);
    if (fit && fit.rms <= e * 0.01) {
      best = fit;
      ext = e;
      break;
    }
  }
  if (!best) return null;
  let def = best.result.def;
  // the seed residual is essentially scanner noise: grow within a few times that
  const tol = Math.max(best.rms * (opts.tolFactor ?? 4), ext * 1e-4);
  const cosTol = Math.cos((opts.angleTolDeg * Math.PI) / 180);
  let dist = distanceKernel(def);
  const mask = new Uint8Array(nt);
  const queue = new Int32Array(Math.min(nt, opts.maxTriangles ?? nt) + 1);
  let head = 0, tail = 0, count = 0;
  let nextRefit = 1000;
  /** boundary triangles rejected by the current model (re-tested after a refit) */
  let rejected: number[] = [];

  const accept = (t: number): boolean => {
    const a = 3 * I[3 * t], b = 3 * I[3 * t + 1], cc = 3 * I[3 * t + 2];
    const cx = (P[a] + P[b] + P[cc]) / 3, cy = (P[a + 1] + P[b + 1] + P[cc + 1]) / 3, cz = (P[a + 2] + P[b + 2] + P[cc + 2]) / 3;
    if (Math.abs(dist(cx, cy, cz)) > tol) return false;
    // also test the vertices: catches triangles straddling a tangent edge
    if (Math.abs(dist(P[a], P[a + 1], P[a + 2])) > tol * 1.5) return false;
    if (Math.abs(dist(P[b], P[b + 1], P[b + 2])) > tol * 1.5) return false;
    if (Math.abs(dist(P[cc], P[cc + 1], P[cc + 2])) > tol * 1.5) return false;
    const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2];
    const vx = P[cc] - P[a], vy = P[cc + 1] - P[a + 1], vz = P[cc + 2] - P[a + 2];
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const l = Math.hypot(nx, ny, nz) || 1;
    nx /= l;
    ny /= l;
    nz /= l;
    const pn = primitiveNormalAt(def, [cx, cy, cz]);
    return Math.abs(nx * pn[0] + ny * pn[1] + nz * pn[2]) >= cosTol;
  };

  /**
   * Refit on the grown region. The kind may change (a nearly flat seed on a gently curved bumper
   * starts as a plane and becomes a large cylinder once the region shows the curvature).
   */
  const refit = (): boolean => {
    const region = queue.subarray(0, tail);
    const r = fitBest(mesh, verticesOf(mesh, region));
    if (!r || r.rms > tol * 0.6) return false;
    best = r;
    def = r.result.def;
    dist = distanceKernel(def);
    return true;
  };

  mask[seed] = 1;
  queue[tail++] = seed;
  count = 1;
  for (let round = 0; round < 12; round++) {
    while (head < tail) {
      const t = queue[head++];
      for (let k = 0; k < 3; k++) {
        const u = nbr[3 * t + k];
        if (u < 0 || mask[u]) continue;
        if (!accept(u)) {
          rejected.push(u);
          continue;
        }
        mask[u] = 1;
        count++;
        if (tail < queue.length) queue[tail++] = u;
      }
      if (count >= nextRefit) {
        refit();
        nextRefit *= 2;
      }
      if (opts.maxTriangles && count >= opts.maxTriangles) return { mask, count, kind: best!.kind, def, rms: best!.rms };
    }
    // frontier exhausted: refit on everything found so far and re-test the rejected boundary
    if (!refit()) break;
    const retry = rejected;
    rejected = [];
    let added = 0;
    for (const u of retry) {
      if (mask[u] || !accept(u)) {
        if (!mask[u]) rejected.push(u);
        continue;
      }
      mask[u] = 1;
      count++;
      added++;
      if (tail < queue.length) queue[tail++] = u;
    }
    if (!added) break;
  }
  return { mask, count, kind: best!.kind, def, rms: best!.rms };
}

/** Smart select: primitive region growing with a curvature-based fallback for freeform areas. */
export function smartRegion(mesh: MeshData, nbr: Int32Array, seed: number, angleTolDeg: number): RegionGrowResult {
  const prim = primitiveRegionGrow(mesh, nbr, seed, { angleTolDeg });
  if (prim && prim.count > 50) return prim;
  const mask = smartSelect(mesh, seed, { angleTolDeg, neighbors: nbr });
  let count = 0;
  for (let i = 0; i < mask.length; i++) count += mask[i];
  return { mask, count, kind: "freeform", def: null, rms: 0 };
}
