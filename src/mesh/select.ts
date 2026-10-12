/**
 * Triangle selection helpers. Selections are per-triangle Uint8Array masks (1 = selected).
 *
 * Functions that need triangle adjacency accept a precomputed `neighbors` table
 * (buildTriangleNeighbors) — the app should cache it per mesh, because building it is O(n).
 * Region-growing tools (selectConnected, smartSelect) then cost O(size of the region).
 */
import type { MeshData, Vec3 } from "../core/types";
import { buildTriangleNeighbors, buildVertexTriangles, triangleCountOf, vertexCountOf } from "./topology";
import { GrowableU32 } from "./util/growable";
import { IntFloatMap } from "./util/int-map";

/**
 * Grow by `rings` rings of triangles sharing a vertex with the selection.
 * O(rings · n), 1 B/vertex scratch.
 */
export function growSelection(mesh: MeshData, triMask: Uint8Array, rings = 1): Uint8Array {
  const I = mesh.indices;
  const nt = triangleCountOf(mesh);
  const nv = vertexCountOf(mesh);
  let cur = triMask.slice(0, nt);
  const vmark = new Uint8Array(nv);
  for (let r = 0; r < rings; r++) {
    vmark.fill(0);
    for (let t = 0; t < nt; t++) if (cur[t]) vmark[I[3 * t]] = vmark[I[3 * t + 1]] = vmark[I[3 * t + 2]] = 1;
    const next = new Uint8Array(nt);
    for (let t = 0; t < nt; t++) next[t] = cur[t] || vmark[I[3 * t]] || vmark[I[3 * t + 1]] || vmark[I[3 * t + 2]] ? 1 : 0;
    cur = next;
  }
  return cur;
}

/** Shrink by `rings` rings: deselect triangles touching a vertex of an unselected triangle. */
export function shrinkSelection(mesh: MeshData, triMask: Uint8Array, rings = 1): Uint8Array {
  const I = mesh.indices;
  const nt = triangleCountOf(mesh);
  const nv = vertexCountOf(mesh);
  let cur = triMask.slice(0, nt);
  const vmark = new Uint8Array(nv);
  for (let r = 0; r < rings; r++) {
    vmark.fill(0);
    for (let t = 0; t < nt; t++) if (!cur[t]) vmark[I[3 * t]] = vmark[I[3 * t + 1]] = vmark[I[3 * t + 2]] = 1;
    const next = new Uint8Array(nt);
    for (let t = 0; t < nt; t++) next[t] = cur[t] && !vmark[I[3 * t]] && !vmark[I[3 * t + 1]] && !vmark[I[3 * t + 2]] ? 1 : 0;
    cur = next;
  }
  return cur;
}

/** 1 − mask. */
export function invertSelection(triMask: Uint8Array): Uint8Array {
  const out = new Uint8Array(triMask.length);
  for (let i = 0; i < triMask.length; i++) out[i] = triMask[i] ? 0 : 1;
  return out;
}

/** Number of selected triangles. */
export function countSelected(triMask: Uint8Array): number {
  let n = 0;
  for (let i = 0; i < triMask.length; i++) if (triMask[i]) n++;
  return n;
}

/** Per-vertex mask: vertices of selected triangles. */
export function selectionVertices(mesh: MeshData, triMask: Uint8Array): Uint8Array {
  const I = mesh.indices;
  const nt = triangleCountOf(mesh);
  const vm = new Uint8Array(vertexCountOf(mesh));
  for (let t = 0; t < nt; t++) if (triMask[t]) vm[I[3 * t]] = vm[I[3 * t + 1]] = vm[I[3 * t + 2]] = 1;
  return vm;
}

export interface ConnectedOptions {
  /** Precomputed triangle neighbours (edge mode). */
  neighbors?: Int32Array;
  /**
   * 'edge' (default): flood across manifold edges only. 'vertex': across shared vertices
   * (also crosses non-manifold edges / bow-tie vertices; builds a vertex→triangle table).
   */
  via?: "edge" | "vertex";
  /** Restrict the flood to triangles with allowed[t] !== 0. */
  allowed?: Uint8Array;
}

/** Flood fill from the seed triangles. */
export function selectConnected(mesh: MeshData, seedTris: ArrayLike<number>, opts: ConnectedOptions = {}): Uint8Array {
  const I = mesh.indices;
  const nt = triangleCountOf(mesh);
  const out = new Uint8Array(nt);
  const queue = new GrowableU32(1024);
  const allowed = opts.allowed;
  for (let i = 0; i < seedTris.length; i++) {
    const t = seedTris[i];
    if (t < nt && !out[t] && (!allowed || allowed[t])) {
      out[t] = 1;
      queue.push(t);
    }
  }
  if (opts.via === "vertex") {
    const vt = buildVertexTriangles(mesh);
    for (let h = 0; h < queue.length; h++) {
      const t = queue.data[h];
      for (let k = 0; k < 3; k++) {
        const v = I[3 * t + k];
        for (let j = vt.offsets[v], e = vt.offsets[v + 1]; j < e; j++) {
          const n = vt.triangles[j];
          if (out[n] || (allowed && !allowed[n])) continue;
          out[n] = 1;
          queue.push(n);
        }
      }
    }
    return out;
  }
  const nbr = opts.neighbors ?? buildTriangleNeighbors(mesh);
  for (let h = 0; h < queue.length; h++) {
    const t = queue.data[h];
    for (let k = 0; k < 3; k++) {
      const n = nbr[3 * t + k];
      if (n < 0 || out[n] || (allowed && !allowed[n])) continue;
      out[n] = 1;
      queue.push(n);
    }
  }
  return out;
}

/** Select triangles whose unit normal is within angleTolDeg of `direction`. */
export function selectByNormal(mesh: MeshData, direction: Vec3, angleTolDeg: number): Uint8Array {
  const P = mesh.positions, I = mesh.indices;
  const nt = triangleCountOf(mesh);
  const dl = Math.hypot(direction[0], direction[1], direction[2]) || 1;
  const dx = direction[0] / dl, dy = direction[1] / dl, dz = direction[2] / dl;
  const cosTol = Math.cos((angleTolDeg * Math.PI) / 180);
  const out = new Uint8Array(nt);
  for (let t = 0; t < nt; t++) {
    const a = 3 * I[3 * t], b = 3 * I[3 * t + 1], c = 3 * I[3 * t + 2];
    const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2];
    const vx = P[c] - P[a], vy = P[c + 1] - P[a + 1], vz = P[c + 2] - P[a + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const l = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (l > 0 && (nx * dx + ny * dy + nz * dz) >= cosTol * l) out[t] = 1;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Smart (curvature-aware) region growing
// ---------------------------------------------------------------------------

export interface SmartSelectOptions {
  /** Max angle between adjacent triangle normals (stops at sharp edges). Default 20°. */
  angleTolDeg?: number;
  /**
   * Max |K(t) − K_region| where K is a smoothed signed curvature estimate (1/length) and
   * K_region the running mean over the selected region. Default: automatic from the seed
   * neighbourhood (≈ 4 robust std. dev., ≥ 25 % of |K_seed|, ≥ 0.005 / edge length).
   */
  curvatureTol?: number;
  /** Precomputed triangle neighbours (strongly recommended for large meshes). */
  neighbors?: Int32Array;
  /** Stop after this many triangles (safety). Default: no limit. */
  maxTriangles?: number;
}

/**
 * Region growing from `seedTri` that stays on one smooth, primitive-like surface region: it
 * crosses an edge only when the dihedral angle is below `angleTolDeg` AND the neighbour's
 * curvature matches the region's curvature. So it spreads over a whole plane or cylinder face
 * but stops at sharp edges and at fillet boundaries (where curvature jumps although the surface
 * is tangent-continuous).
 *
 * Curvature per triangle is computed lazily only for visited triangles: the per-edge
 * normal-curvature estimate κ = (n_t − n_u)·(c_t − c_u)/|c_t − c_u|² between face normals and
 * centroids of adjacent triangles, averaged over the triangle's neighbours and then smoothed
 * over two rings of neighbours. Cost is proportional to the selected region.
 */
export function smartSelect(mesh: MeshData, seedTri: number, opts: SmartSelectOptions = {}): Uint8Array {
  const P = mesh.positions, I = mesh.indices;
  const nt = triangleCountOf(mesh);
  const out = new Uint8Array(nt);
  if (seedTri < 0 || seedTri >= nt) return out;
  const nbr = opts.neighbors ?? buildTriangleNeighbors(mesh);
  const cosTol = Math.cos(((opts.angleTolDeg ?? 20) * Math.PI) / 180);
  const maxTris = opts.maxTriangles ?? Infinity;

  // --- lazily evaluated per-triangle geometry -------------------------------------------
  const tn = new Float64Array(6); // scratch normals
  const tc = new Float64Array(6); // scratch centroids
  const faceGeom = (t: number, o: number) => {
    const a = 3 * I[3 * t], b = 3 * I[3 * t + 1], c = 3 * I[3 * t + 2];
    const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2];
    const vx = P[c] - P[a], vy = P[c + 1] - P[a + 1], vz = P[c + 2] - P[a + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const l = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
    tn[o] = nx / l;
    tn[o + 1] = ny / l;
    tn[o + 2] = nz / l;
    tc[o] = (P[a] + P[b] + P[c]) / 3;
    tc[o + 1] = (P[a + 1] + P[b + 1] + P[c + 1]) / 3;
    tc[o + 2] = (P[a + 2] + P[b + 2] + P[c + 2]) / 3;
  };
  const rawMemo = new IntFloatMap(1024);
  const k1Memo = new IntFloatMap(1024);
  const k2Memo = new IntFloatMap(1024);
  let edgeLenSum = 0, edgeLenCount = 0;
  const rawK = (t: number): number => {
    const m = rawMemo.get(t);
    if (m === m) return m; // not NaN
    faceGeom(t, 0);
    let s = 0, n = 0;
    for (let k = 0; k < 3; k++) {
      const u = nbr[3 * t + k];
      if (u < 0) continue;
      faceGeom(u, 3);
      const dx = tc[0] - tc[3], dy = tc[1] - tc[4], dz = tc[2] - tc[5];
      const d2 = dx * dx + dy * dy + dz * dz;
      if (!(d2 > 0)) continue;
      s += ((tn[0] - tn[3]) * dx + (tn[1] - tn[4]) * dy + (tn[2] - tn[5]) * dz) / d2;
      n++;
      edgeLenSum += Math.sqrt(d2);
      edgeLenCount++;
    }
    const v = n ? s / n : 0;
    rawMemo.set(t, v);
    return v;
  };
  const ringMean = (t: number, f: (u: number) => number): number => {
    let s = f(t), n = 1;
    for (let k = 0; k < 3; k++) {
      const u = nbr[3 * t + k];
      if (u < 0) continue;
      s += f(u);
      n++;
    }
    return s / n;
  };
  const k1 = (t: number): number => {
    const m = k1Memo.get(t);
    if (m === m) return m;
    const v = ringMean(t, rawK);
    k1Memo.set(t, v);
    return v;
  };
  const curv = (t: number): number => {
    const m = k2Memo.get(t);
    if (m === m) return m;
    const v = ringMean(t, k1);
    k2Memo.set(t, v);
    return v;
  };
  const normalDot = (t: number, u: number): number => {
    faceGeom(t, 0);
    faceGeom(u, 3);
    return tn[0] * tn[3] + tn[1] * tn[4] + tn[2] * tn[5];
  };

  // --- seed statistics (small BFS patch around the seed, respecting the angle test) ------
  const patch = new GrowableU32(64);
  const inPatch = new IntFloatMap(64);
  patch.push(seedTri);
  inPatch.set(seedTri, 1);
  for (let h = 0; h < patch.length && patch.length < 40; h++) {
    const t = patch.data[h];
    for (let k = 0; k < 3; k++) {
      const u = nbr[3 * t + k];
      if (u < 0 || inPatch.has(u)) continue;
      if (normalDot(t, u) < cosTol) continue;
      inPatch.set(u, 1);
      patch.push(u);
    }
  }
  const vals = new Float64Array(patch.length);
  for (let i = 0; i < patch.length; i++) vals[i] = curv(patch.data[i]);
  const sorted = Float64Array.from(vals).sort();
  const median = sorted[sorted.length >> 1];
  const dev = new Float64Array(vals.length);
  for (let i = 0; i < vals.length; i++) dev[i] = Math.abs(vals[i] - median);
  dev.sort();
  const mad = dev[dev.length >> 1];
  const h = edgeLenCount ? edgeLenSum / edgeLenCount : 1;
  const tol = opts.curvatureTol ?? Math.max(4 * 1.4826 * mad, 0.25 * Math.abs(median), 0.005 / h);

  // --- region growing ---------------------------------------------------------------------
  let refSum = 0, refN = 0;
  const queue = new GrowableU32(1024);
  out[seedTri] = 1;
  queue.push(seedTri);
  refSum += median;
  refN++;
  let selected = 1;
  for (let q = 0; q < queue.length && selected < maxTris; q++) {
    const t = queue.data[q];
    for (let k = 0; k < 3; k++) {
      const u = nbr[3 * t + k];
      if (u < 0 || out[u]) continue;
      if (normalDot(t, u) < cosTol) continue;
      const ku = curv(u);
      if (Math.abs(ku - refSum / refN) > tol) continue;
      out[u] = 1;
      queue.push(u);
      refSum += ku;
      refN++;
      selected++;
      if (selected >= maxTris) break;
    }
  }
  return out;
}
