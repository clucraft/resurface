/**
 * Hole filling (Liepa 2003, "Filling Holes in Meshes", with a CG-based fairing).
 *
 * Per boundary loop:
 *  1. Triangulate the loop polygon.
 *     - up to DP_LIMIT vertices: dynamic programming minimising, lexicographically, the largest
 *       dihedral angle (between new triangles and with the surrounding mesh) and then the total
 *       area — this handles strongly non-planar loops well (O(m³) time, O(m²) memory);
 *     - larger loops: ear clipping in the loop's best-fit plane (smallest interior angle first,
 *       with containment tests), O(m²).
 *     Diagonals that already exist as mesh edges are never created (prevents non-manifold edges).
 *  2. Refine: split triangles at their centroid until edge lengths match the density of the
 *     surrounding mesh (scale attribute σ propagated from the loop edges), relaxing edges with
 *     Delaunay-like flips after each split and between passes.
 *  3. Fair: solve for the new interior vertices with the loop vertices (and the surrounding
 *     mesh) fixed —
 *       smooth = true : minimise Σ‖L(v)‖² over interior AND loop vertices (bi-Laplacian / thin
 *                        plate). Including the loop vertices' Laplacians, whose one-ring reaches
 *                        into the existing mesh, makes the patch blend tangent- and
 *                        curvature-continuously with the surroundings;
 *       smooth = false: L(v) = 0 at interior vertices (membrane — minimal-area-like, only C0).
 *     Uniform (umbrella) weights; matrix-free conjugate gradients.
 *     With smooth = true (and quadricBase, default on) a quadratic height field fitted to the
 *     hole's surroundings serves as base surface and only the residual is faired, so locally
 *     quadratic shapes (sphere / cylinder / saddle patches) are reproduced inside the hole:
 *     e.g. a hole of radius 3 in a sphere of radius 10 is filled with < 0.012 max deviation
 *     (0.06 without the quadric base, 0.43 with the membrane).
 *  Colours of new vertices are interpolated harmonically from the loop colours.
 *
 * Patch-local data (sized by the hole, not by the mesh) uses small typed arrays; the only
 * mesh-sized work is one linear scan to collect the one-ring around loop vertices.
 */
import { solveLinear } from "../core/linalg";
import type { MeshData, ProgressFn } from "../core/types";
import {
  buildTriangleNeighbors,
  computeVertexNormals,
  findBoundaryLoops,
  triangleCountOf,
  vertexCountOf,
} from "./topology";
import { GrowableF32, GrowableF64, GrowableI32, GrowableU32 } from "./util/growable";
import { MinHeap } from "./util/heap";
import { IntFloatMap } from "./util/int-map";
import { PairHashMap } from "./util/pair-hash";

export interface FillHolesOptions {
  /** Only fill loops with at most this many boundary edges (default: no limit). */
  maxBoundaryEdges?: number;
  /** Indices into findBoundaryLoops(mesh) of the loops to fill (default: all loops). */
  loops?: number[];
  /** true: curvature-continuous (bi-Laplacian) fairing; false: membrane fairing. */
  smooth: boolean;
  /** Insert interior vertices to match the surrounding edge length (default true). */
  refine?: boolean;
  /**
   * With smooth = true: fit a quadratic height field to the hole surroundings and fair only the
   * residual, so locally quadratic surfaces (spheres, cylinders, saddles) are reproduced inside
   * the hole. Default true; automatically skipped when the hole is not a height field.
   */
  quadricBase?: boolean;
  /** Precomputed triangle neighbours (for findBoundaryLoops). */
  neighbors?: Int32Array;
  /** Safety cap on vertices added per hole (default 2,000,000). */
  maxNewVerticesPerHole?: number;
  onProgress?: ProgressFn;
}

export interface FillHolesStats {
  /** loops found in the mesh */
  holesFound: number;
  holesFilled: number;
  /** loops selected but not filled (triangulation failed or over the size limit) */
  skipped: number;
  trianglesAdded: number;
  verticesAdded: number;
}

export interface FillHolesResult {
  mesh: MeshData;
  stats: FillHolesStats;
}

/** Ear-clipping is used above this loop size; the O(m³) DP below. */
const DP_LIMIT = 400;
const SQRT2 = Math.SQRT2;

/**
 * Fill boundary loops of the mesh. Returns a new mesh (the input object itself when nothing
 * was filled) and statistics. New vertices are appended after the existing ones; existing
 * vertex and triangle indices are unchanged (new triangles are appended). If the input has
 * vertex normals they are recomputed for the whole output.
 */
export function fillHoles(mesh: MeshData, opts: FillHolesOptions): FillHolesResult {
  const all = findBoundaryLoops(mesh, opts.neighbors);
  const maxE = opts.maxBoundaryEdges ?? Infinity;
  const chosen: Uint32Array[] = [];
  let skipped = 0;
  const ids = opts.loops ?? all.map((_, i) => i);
  for (const i of ids) {
    const L = all[i];
    if (!L) continue;
    if (L.length > maxE) {
      skipped++;
      continue;
    }
    chosen.push(L);
  }
  const res = fillBoundaryLoops(mesh, chosen, opts);
  res.stats.holesFound = all.length;
  res.stats.skipped += skipped;
  return res;
}

// ---------------------------------------------------------------------------
// Core
// ---------------------------------------------------------------------------

/**
 * Fill the given boundary loops (as returned by findBoundaryLoops — ordered along the adjacent
 * triangles' edges). Lower-level entry point used by fillHoles and defeature.
 */
export function fillBoundaryLoops(mesh: MeshData, loops: Uint32Array[], opts: FillHolesOptions): FillHolesResult {
  const stats: FillHolesStats = { holesFound: loops.length, holesFilled: 0, skipped: 0, trianglesAdded: 0, verticesAdded: 0 };
  if (loops.length === 0) return { mesh, stats };
  const P = mesh.positions, I = mesh.indices, C = mesh.colors;
  const nv = vertexCountOf(mesh);
  const nt = triangleCountOf(mesh);

  // --- one-ring triangles of all loop vertices (CSR over loop "slots") ---------------------
  const slotOf = new Int32Array(nv).fill(-1);
  let nSlots = 0;
  for (const L of loops) for (let i = 0; i < L.length; i++) if (slotOf[L[i]] < 0) slotOf[L[i]] = nSlots++;
  const ringOff = new Uint32Array(nSlots + 1);
  for (let c = 0; c < nt * 3; c++) {
    const s = slotOf[I[c]];
    if (s >= 0) ringOff[s + 1]++;
  }
  for (let s = 0; s < nSlots; s++) ringOff[s + 1] += ringOff[s];
  const ringTri = new Uint32Array(ringOff[nSlots]);
  {
    const cur = ringOff.slice(0, nSlots);
    for (let c = 0; c < nt * 3; c++) {
      const s = slotOf[I[c]];
      if (s >= 0) ringTri[cur[s]++] = (c / 3) | 0;
    }
  }
  const ctx: MeshCtx = { P, I, C, slotOf, ringOff, ringTri };

  const addPos = new GrowableF32(1024);
  const addCol = C ? new GrowableF32(1024) : null;
  const addTri = new GrowableU32(1024);
  const maxNew = opts.maxNewVerticesPerHole ?? 2_000_000;
  for (let li = 0; li < loops.length; li++) {
    const ok = fillOneLoop(ctx, loops[li], nv + addPos.length / 3, opts.smooth, opts.refine ?? true, opts.quadricBase ?? true, maxNew, addPos, addCol, addTri);
    if (ok) stats.holesFilled++;
    else stats.skipped++;
    opts.onProgress?.((li + 1) / loops.length, "Filling holes");
  }
  if (stats.holesFilled === 0) return { mesh, stats };

  const nNew = addPos.length / 3;
  stats.verticesAdded = nNew;
  stats.trianglesAdded = addTri.length / 3;
  const positions = new Float32Array((nv + nNew) * 3);
  positions.set(P.subarray(0, nv * 3));
  positions.set(addPos.view(), nv * 3);
  const indices = new Uint32Array(I.length + addTri.length);
  indices.set(I);
  indices.set(addTri.view(), I.length);
  const out: MeshData = { positions, indices };
  if (C && addCol) {
    const colors = new Uint8Array((nv + nNew) * 3);
    colors.set(C.subarray(0, nv * 3));
    const ac = addCol.view();
    for (let i = 0; i < ac.length; i++) colors[nv * 3 + i] = Math.max(0, Math.min(255, Math.round(ac[i])));
    out.colors = colors;
  }
  if (mesh.normals) out.normals = computeVertexNormals(out);
  return { mesh: out, stats };
}

interface MeshCtx {
  P: Float32Array;
  I: Uint32Array;
  C?: Uint8Array;
  slotOf: Int32Array;
  ringOff: Uint32Array;
  ringTri: Uint32Array;
}

/**
 * Fill one loop. `L` is ordered along the adjacent triangles' edges; the patch polygon is the
 * reverse. Appends to the output builders; returns false if nothing could be done.
 */
function fillOneLoop(
  ctx: MeshCtx,
  L: Uint32Array,
  firstNewId: number,
  smooth: boolean,
  refine: boolean,
  quadricBase: boolean,
  maxNew: number,
  addPos: GrowableF32,
  addCol: GrowableF32 | null,
  addTri: GrowableU32,
): boolean {
  const { P, I, C, slotOf, ringOff, ringTri } = ctx;
  const m = L.length;
  if (m < 3) return false;
  // polygon Q = reversed loop
  const Q = new Uint32Array(m);
  for (let i = 0; i < m; i++) Q[i] = L[m - 1 - i];
  const local = new IntFloatMap(m * 2); // global vertex -> local loop index
  for (let i = 0; i < m; i++) local.set(Q[i], i);

  const px = new Float64Array(m * 3);
  for (let i = 0; i < m; i++) for (let k = 0; k < 3; k++) px[3 * i + k] = P[3 * Q[i] + k];

  // normals of the mesh triangles adjacent to the polygon edges (Q[i], Q[i+1]); the mesh has the
  // directed edge Q[i+1] -> Q[i]
  const edgeN = new Float64Array(m * 3);
  for (let i = 0; i < m; i++) {
    const a = Q[(i + 1) % m], b = Q[i];
    const s = slotOf[a];
    for (let j = ringOff[s]; j < ringOff[s + 1]; j++) {
      const t = ringTri[j];
      let found = false;
      for (let k = 0; k < 3; k++) if (I[3 * t + k] === a && I[3 * t + (k === 2 ? 0 : k + 1)] === b) found = true;
      if (found) {
        triNormalGlobal(P, I[3 * t], I[3 * t + 1], I[3 * t + 2], edgeN, 3 * i);
        break;
      }
    }
  }

  // existing mesh edges between non-consecutive loop vertices must not become diagonals
  const forbidden = new PairHashMap(16);
  for (let i = 0; i < m; i++) {
    const s = slotOf[Q[i]];
    for (let j = ringOff[s]; j < ringOff[s + 1]; j++) {
      const t = ringTri[j];
      for (let k = 0; k < 3; k++) {
        const w = I[3 * t + k];
        const lj = local.get(w);
        if (!(lj === lj) || lj === i) continue;
        const d = Math.abs(lj - i);
        if (d === 1 || d === m - 1) continue;
        forbidden.set(Math.min(i, lj), Math.max(i, lj), 1);
      }
    }
  }
  const isForbidden = (i: number, k: number) => forbidden.get(Math.min(i, k), Math.max(i, k)) >= 0;

  // 1. initial triangulation (local loop indices)
  const tri0 = m === 3 ? Int32Array.of(0, 1, 2) : m <= DP_LIMIT ? triangulateDP(m, px, edgeN, isForbidden) : triangulateEars(m, px, isForbidden);
  if (!tri0) return false;

  // 2. patch: positions (local), σ, triangles, neighbours
  const patch = new Patch(px, m);
  for (let i = 0; i < m; i++) {
    const ip = (i + m - 1) % m, inx = (i + 1) % m;
    patch.sigma.push(0.5 * (dist3(px, i, px, ip) + dist3(px, i, px, inx)));
  }
  if (!patch.setTriangles(tri0)) return false;
  if (refine) patch.refine(maxNew);

  const nInterior = patch.vertexCount - m;
  // 3. fairing
  if (nInterior > 0) fairPatch(ctx, patch, Q, local, smooth, quadricBase, edgeN);

  // 4. emit
  const pv = patch.pos.data;
  for (let v = m; v < patch.vertexCount; v++) addPos.push3(pv[3 * v], pv[3 * v + 1], pv[3 * v + 2]);
  if (addCol && C) {
    const col = interpolateColors(patch, Q, C);
    for (let i = 0; i < col.length; i++) addCol.push(col[i]);
  }
  const T = patch.tri.data;
  for (let t = 0; t < patch.triCount; t++) {
    for (let k = 0; k < 3; k++) {
      const v = T[3 * t + k];
      addTri.push(v < m ? Q[v] : firstNewId + (v - m));
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Triangulation
// ---------------------------------------------------------------------------

function triNormalGlobal(P: Float32Array, a: number, b: number, c: number, out: Float64Array, o: number): void {
  const ax = P[3 * a], ay = P[3 * a + 1], az = P[3 * a + 2];
  const ux = P[3 * b] - ax, uy = P[3 * b + 1] - ay, uz = P[3 * b + 2] - az;
  const vx = P[3 * c] - ax, vy = P[3 * c + 1] - ay, vz = P[3 * c + 2] - az;
  const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
  const l = Math.sqrt(nx * nx + ny * ny + nz * nz);
  if (l > 0) {
    out[o] = nx / l;
    out[o + 1] = ny / l;
    out[o + 2] = nz / l;
  }
}

/** Unnormalised normal of local triangle (i,j,k) into out[o..o+2]; returns its length (= 2·area). */
function triCross(px: Float64Array, i: number, j: number, k: number, out: Float64Array, o: number): number {
  const ax = px[3 * i], ay = px[3 * i + 1], az = px[3 * i + 2];
  const ux = px[3 * j] - ax, uy = px[3 * j + 1] - ay, uz = px[3 * j + 2] - az;
  const vx = px[3 * k] - ax, vy = px[3 * k + 1] - ay, vz = px[3 * k + 2] - az;
  const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
  out[o] = nx;
  out[o + 1] = ny;
  out[o + 2] = nz;
  return Math.sqrt(nx * nx + ny * ny + nz * nz);
}

function dist3(a: Float64Array, i: number, b: Float64Array, j: number): number {
  return Math.hypot(a[3 * i] - b[3 * j], a[3 * i + 1] - b[3 * j + 1], a[3 * i + 2] - b[3 * j + 2]);
}

/**
 * Liepa's DP: weight of a triangulation of sub-polygon (i..k) = (max dihedral angle, area),
 * compared lexicographically. Returns local triangles (3 per triangle) or null.
 */
function triangulateDP(n: number, px: Float64Array, edgeN: Float64Array, forbidden: (i: number, k: number) => boolean): Int32Array | null {
  // Work with cosines of dihedral angles (larger = flatter) and cache the unit normal of the
  // chosen apex triangle of every sub-polygon, so the inner loop needs one cross product.
  const N2 = n * n;
  const wc = new Float64Array(N2).fill(-Infinity); // min cos over the sub-triangulation
  const wr = new Float64Array(N2).fill(Infinity); // area
  const O = new Int32Array(N2).fill(-1);
  const nrm = new Float64Array(N2 * 3); // unit normal of triangle (i, O[i,k], k)
  for (let i = 0; i + 1 < n; i++) {
    wc[i * n + i + 1] = 1;
    wr[i * n + i + 1] = 0;
  }
  // unit edge normals (zero = unknown → no constraint)
  const en = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) {
    const l = Math.hypot(edgeN[3 * i], edgeN[3 * i + 1], edgeN[3 * i + 2]);
    if (l > 0) for (let c = 0; c < 3; c++) en[3 * i + c] = edgeN[3 * i + c] / l;
  }
  const cosWith = (x: number, y: number, z: number, arr: Float64Array, o: number): number => {
    const ax = arr[o], ay = arr[o + 1], az = arr[o + 2];
    if (ax === 0 && ay === 0 && az === 0) return 1;
    return x * ax + y * ay + z * az;
  };
  const EPS = 1e-9;
  for (let len = 2; len < n; len++) {
    for (let i = 0; i + len < n; i++) {
      const k = i + len;
      if (len < n - 1 && forbidden(i, k)) continue;
      let bestC = -Infinity, bestR = Infinity, bestJ = -1, bx = 0, by = 0, bz = 0;
      const ix = px[3 * i], iy = px[3 * i + 1], iz = px[3 * i + 2];
      const kx = px[3 * k] - ix, ky = px[3 * k + 1] - iy, kz = px[3 * k + 2] - iz;
      for (let j = i + 1; j < k; j++) {
        const c1 = wc[i * n + j], c2 = wc[j * n + k];
        if (c1 === -Infinity || c2 === -Infinity) continue;
        const jx = px[3 * j] - ix, jy = px[3 * j + 1] - iy, jz = px[3 * j + 2] - iz;
        let nx = jy * kz - jz * ky, ny = jz * kx - jx * kz, nz = jx * ky - jy * kx;
        const twoArea = Math.sqrt(nx * nx + ny * ny + nz * nz);
        let c = c1 < c2 ? c1 : c2;
        if (twoArea > 0) {
          nx /= twoArea;
          ny /= twoArea;
          nz /= twoArea;
          // across edge (i,j), (j,k) and — for the outermost triangle — the closing edge (n-1, 0)
          const ca = j === i + 1 ? cosWith(nx, ny, nz, en, 3 * i) : cosWith(nx, ny, nz, nrm, 3 * (i * n + j));
          if (ca < c) c = ca;
          const cb = k === j + 1 ? cosWith(nx, ny, nz, en, 3 * j) : cosWith(nx, ny, nz, nrm, 3 * (j * n + k));
          if (cb < c) c = cb;
          if (i === 0 && k === n - 1) {
            const cc = cosWith(nx, ny, nz, en, 3 * (n - 1));
            if (cc < c) c = cc;
          }
        } else {
          c = -1; // degenerate triangle: worst possible
          nx = ny = nz = 0;
        }
        const area = wr[i * n + j] + wr[j * n + k] + 0.5 * twoArea;
        if (c > bestC + EPS || (c >= bestC - EPS && area < bestR)) {
          bestC = c;
          bestR = area;
          bestJ = j;
          bx = nx;
          by = ny;
          bz = nz;
        }
      }
      if (bestJ >= 0) {
        const o = i * n + k;
        wc[o] = bestC;
        wr[o] = bestR;
        O[o] = bestJ;
        nrm[3 * o] = bx;
        nrm[3 * o + 1] = by;
        nrm[3 * o + 2] = bz;
      }
    }
  }
  if (O[n - 1] < 0) return null;
  const out = new Int32Array((n - 2) * 3);
  let o = 0;
  const stack = new GrowableI32(64);
  stack.push2(0, n - 1);
  while (stack.length > 0) {
    const k = stack.data[--stack.length];
    const i = stack.data[--stack.length];
    if (k - i < 2) continue;
    const j = O[i * n + k];
    if (j < 0) return null;
    out[o++] = i;
    out[o++] = j;
    out[o++] = k;
    stack.push2(i, j);
    stack.push2(j, k);
  }
  return out;
}

/**
 * Ear clipping in the best-fit plane (Newell normal) for large loops. Picks the ear with the
 * smallest interior angle that contains no other polygon vertex; if none is valid (the projected
 * polygon self-overlaps), the smallest-angle convex ear is clipped regardless.
 */
function triangulateEars(n: number, px: Float64Array, forbidden: (i: number, k: number) => boolean): Int32Array | null {
  // Newell normal & centroid
  let nx = 0, ny = 0, nz = 0, cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    const xi = px[3 * i], yi = px[3 * i + 1], zi = px[3 * i + 2];
    const xj = px[3 * j], yj = px[3 * j + 1], zj = px[3 * j + 2];
    nx += (yi - yj) * (zi + zj);
    ny += (zi - zj) * (xi + xj);
    nz += (xi - xj) * (yi + yj);
    cx += xi;
    cy += yi;
    cz += zi;
  }
  const nl = Math.hypot(nx, ny, nz);
  if (!(nl > 0)) return null;
  nx /= nl;
  ny /= nl;
  nz /= nl;
  cx /= n;
  cy /= n;
  cz /= n;
  // in-plane basis
  let ex = 0, ey = 0, ez = 0;
  if (Math.abs(nx) <= Math.abs(ny) && Math.abs(nx) <= Math.abs(nz)) ex = 1;
  else if (Math.abs(ny) <= Math.abs(nz)) ey = 1;
  else ez = 1;
  let ux = ey * nz - ez * ny, uy = ez * nx - ex * nz, uz = ex * ny - ey * nx;
  const ul = Math.hypot(ux, uy, uz);
  ux /= ul;
  uy /= ul;
  uz /= ul;
  const vx = ny * uz - nz * uy, vy = nz * ux - nx * uz, vz = nx * uy - ny * ux;
  const X = new Float64Array(n), Y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const dx = px[3 * i] - cx, dy = px[3 * i + 1] - cy, dz = px[3 * i + 2] - cz;
    X[i] = dx * ux + dy * uy + dz * uz;
    Y[i] = dx * vx + dy * vy + dz * vz;
  }
  const prev = new Int32Array(n), next = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    prev[i] = (i + n - 1) % n;
    next[i] = (i + 1) % n;
  }
  const alive = new Uint8Array(n).fill(1);
  const version = new Uint32Array(n);
  const heap = new MinHeap(n * 2);
  // key: interior angle (convex ears); reflex vertices get angle > π and are kept with a large key
  const keyOf = (i: number): number => {
    const a = prev[i], b = next[i];
    const ax = X[a] - X[i], ay = Y[a] - Y[i];
    const bx = X[b] - X[i], by = Y[b] - Y[i];
    // interior angle measured from (next - i) to (prev - i), CCW polygon
    let ang = Math.atan2(bx * ay - by * ax, bx * ax + by * ay);
    if (ang < 0) ang += 2 * Math.PI;
    return ang;
  };
  // heap ids encode (vertex, version) as vertex + n * (version & 0xff) — versions are checked
  const pushV = (i: number) => {
    version[i]++;
    heap.push(keyOf(i), i + n * (version[i] & 0xff));
  };
  for (let i = 0; i < n; i++) pushV(i);
  const pointInTri = (p: number, a: number, b: number, c: number): boolean => {
    const x = X[p], y = Y[p];
    const d1 = (X[b] - X[a]) * (y - Y[a]) - (Y[b] - Y[a]) * (x - X[a]);
    const d2 = (X[c] - X[b]) * (y - Y[b]) - (Y[c] - Y[b]) * (x - X[b]);
    const d3 = (X[a] - X[c]) * (y - Y[c]) - (Y[a] - Y[c]) * (x - X[c]);
    return d1 >= 0 && d2 >= 0 && d3 >= 0;
  };
  const out = new Int32Array((n - 2) * 3);
  let o = 0;
  let remaining = n;
  const deferred = new GrowableI32(64);
  while (remaining > 3) {
    if (heap.size === 0) {
      // every candidate was rejected: force the smallest-angle convex ear
      let best = -1, bestK = Infinity;
      for (let i = 0; i < n; i++) {
        if (!alive[i]) continue;
        const k = keyOf(i);
        if (k < bestK && !forbidden(prev[i], next[i])) {
          bestK = k;
          best = i;
        }
      }
      if (best < 0) return null;
      clip(best);
      continue;
    }
    const id = heap.pop();
    const i = id % n;
    if (!alive[i] || (id - i) / n !== (version[i] & 0xff)) continue;
    const a = prev[i], b = next[i];
    let ok = keyOf(i) < Math.PI && !forbidden(a, b);
    if (ok) {
      for (let p = next[b]; p !== a; p = next[p]) {
        if (pointInTri(p, a, i, b) && (X[p] !== X[a] || Y[p] !== Y[a]) && (X[p] !== X[b] || Y[p] !== Y[b])) {
          ok = false;
          break;
        }
      }
    }
    if (!ok) {
      deferred.push(i);
      continue;
    }
    clip(i);
  }
  // last triangle
  let s = 0;
  while (!alive[s]) s++;
  out[o++] = prev[s];
  out[o++] = s;
  out[o++] = next[s];
  return o === out.length ? out : null;

  function clip(i: number) {
    const a = prev[i], b = next[i];
    out[o++] = a;
    out[o++] = i;
    out[o++] = b;
    alive[i] = 0;
    next[a] = b;
    prev[b] = a;
    remaining--;
    pushV(a);
    pushV(b);
    // give previously rejected ears another chance (the polygon changed)
    for (let d = 0; d < deferred.length; d++) {
      const v = deferred.data[d];
      if (alive[v] && v !== a && v !== b) pushV(v);
    }
    deferred.clear();
  }
}

// ---------------------------------------------------------------------------
// Patch with local adjacency: refinement (centroid splits + edge flips)
// ---------------------------------------------------------------------------

class Patch {
  pos: GrowableF64;
  sigma: GrowableF64;
  tri: GrowableI32;
  nbr: GrowableI32;
  readonly nLoop: number;

  constructor(px: Float64Array, nLoop: number) {
    this.nLoop = nLoop;
    this.pos = new GrowableF64(px.length * 4);
    for (let i = 0; i < px.length; i++) this.pos.push(px[i]);
    this.sigma = new GrowableF64(nLoop * 4);
    this.tri = new GrowableI32(nLoop * 12);
    this.nbr = new GrowableI32(nLoop * 12);
  }

  get vertexCount(): number {
    return this.pos.length / 3;
  }
  get triCount(): number {
    return this.tri.length / 3;
  }

  /** Set the initial triangles and build neighbours. Returns false on non-manifold input. */
  setTriangles(tris: Int32Array): boolean {
    const nt = tris.length / 3;
    for (let i = 0; i < tris.length; i++) {
      this.tri.push(tris[i]);
      this.nbr.push(-1);
    }
    const map = new PairHashMap(nt * 3);
    const T = this.tri.data, N = this.nbr.data;
    for (let t = 0; t < nt; t++) {
      for (let k = 0; k < 3; k++) {
        const a = T[3 * t + k], b = T[3 * t + (k + 1) % 3];
        // twin is b->a
        const twin = map.get(b, a);
        if (twin >= 0) {
          N[3 * t + k] = (twin / 3) | 0;
          N[twin] = t;
        } else {
          if (map.insert(a, b, 3 * t + k) >= 0) return false; // duplicate directed edge
        }
      }
    }
    return true;
  }

  private triCrossLocal(a: number, b: number, c: number, out: Float64Array, o: number): number {
    return triCross(this.pos.data, a, b, c, out, o);
  }

  /** Liepa refinement. */
  refine(maxNew: number): void {
    const flipQueue = new GrowableI32(256);
    let added = 0;
    for (let pass = 0; pass < 64; pass++) {
      let anySplit = false;
      const ntNow = this.triCount;
      for (let t = 0; t < ntNow; t++) {
        if (added >= maxNew) break;
        if (this.shouldSplit(t)) {
          this.split(t, flipQueue);
          added++;
          anySplit = true;
          this.relaxQueue(flipQueue);
        }
      }
      if (!anySplit || added >= maxNew) break;
      // full relaxation pass
      for (let iter = 0; iter < 8; iter++) {
        for (let t = 0; t < this.triCount; t++) for (let k = 0; k < 3; k++) flipQueue.push2(t, k);
        if (this.relaxQueue(flipQueue) === 0) break;
      }
    }
  }

  private shouldSplit(t: number): boolean {
    const T = this.tri.data, p = this.pos.data, s = this.sigma.data;
    const a = T[3 * t], b = T[3 * t + 1], c = T[3 * t + 2];
    const cx = (p[3 * a] + p[3 * b] + p[3 * c]) / 3;
    const cy = (p[3 * a + 1] + p[3 * b + 1] + p[3 * c + 1]) / 3;
    const cz = (p[3 * a + 2] + p[3 * b + 2] + p[3 * c + 2]) / 3;
    const sc = (s[a] + s[b] + s[c]) / 3;
    for (const v of [a, b, c]) {
      const d = SQRT2 * Math.hypot(cx - p[3 * v], cy - p[3 * v + 1], cz - p[3 * v + 2]);
      if (!(d > sc) || !(d > s[v])) return false;
    }
    return true;
  }

  /** 1→3 split at the centroid. Pushes the three outer edges onto the flip queue. */
  private split(t: number, flipQueue: GrowableI32): void {
    const p = this.pos.data;
    const T0 = this.tri.data;
    const a = T0[3 * t], b = T0[3 * t + 1], c = T0[3 * t + 2];
    const v = this.vertexCount;
    this.pos.push3((p[3 * a] + p[3 * b] + p[3 * c]) / 3, (p[3 * a + 1] + p[3 * b + 1] + p[3 * c + 1]) / 3, (p[3 * a + 2] + p[3 * b + 2] + p[3 * c + 2]) / 3);
    const s = this.sigma.data;
    this.sigma.push((s[a] + s[b] + s[c]) / 3);
    const N0 = this.nbr.data;
    const nab = N0[3 * t], nbc = N0[3 * t + 1], nca = N0[3 * t + 2];
    const t1 = this.triCount, t2 = t1 + 1;
    this.tri.push3(b, c, v);
    this.tri.push3(c, a, v);
    this.nbr.push3(nbc, t2, t);
    this.nbr.push3(nca, t, t1);
    const T = this.tri.data, N = this.nbr.data;
    T[3 * t + 2] = v;
    N[3 * t] = nab;
    N[3 * t + 1] = t1;
    N[3 * t + 2] = t2;
    if (nbc >= 0) this.replaceNbr(nbc, t, t1);
    if (nca >= 0) this.replaceNbr(nca, t, t2);
    flipQueue.push2(t, 0);
    flipQueue.push2(t1, 0);
    flipQueue.push2(t2, 0);
  }

  private replaceNbr(x: number, from: number, to: number): void {
    const N = this.nbr.data;
    for (let k = 0; k < 3; k++)
      if (N[3 * x + k] === from) {
        N[3 * x + k] = to;
        return;
      }
  }

  /** Process the flip queue (Lawson style). Returns number of flips. */
  private relaxQueue(queue: GrowableI32): number {
    let flips = 0;
    const cap = 20 * this.triCount + 100;
    let head = 0;
    while (head < queue.length) {
      const t = queue.data[head++];
      const k = queue.data[head++];
      if (this.tryFlip(t, k, queue)) {
        flips++;
        if (flips > cap) break;
      }
      if (head > 1 << 16 && head * 2 > queue.length) {
        // compact the consumed prefix
        queue.data.copyWithin(0, head, queue.length);
        queue.length -= head;
        head = 0;
      }
    }
    queue.clear();
    return flips;
  }

  private tmp = new Float64Array(12);

  /** Flip edge k of triangle t if the angle-sum (Delaunay) criterion asks for it and it is safe. */
  private tryFlip(t: number, k: number, queue: GrowableI32): boolean {
    const T = this.tri.data, N = this.nbr.data;
    const u = N[3 * t + k];
    if (u < 0) return false;
    const a = T[3 * t + k], b = T[3 * t + (k + 1) % 3], c = T[3 * t + (k + 2) % 3];
    let j = -1;
    for (let q = 0; q < 3; q++) if (T[3 * u + q] === b && T[3 * u + (q + 1) % 3] === a) j = q;
    if (j < 0) return false;
    const d = T[3 * u + (j + 2) % 3];
    if (c === d) return false;
    const p = this.pos.data;
    // angle sum at c and d opposite to edge (a,b)
    const angC = angleAt(p, c, a, b), angD = angleAt(p, d, a, b);
    if (angC + angD <= Math.PI + 1e-9) return false;
    // new edge c-d must not exist already
    if (this.hasEdge(c, d, t)) return false;
    // no fold: new normals must agree with the old ones
    const w = this.tmp;
    const l1 = this.triCrossLocal(a, b, c, w, 0);
    const l2 = this.triCrossLocal(b, a, d, w, 3);
    const l3 = this.triCrossLocal(c, a, d, w, 6);
    const l4 = this.triCrossLocal(d, b, c, w, 9);
    if (!(l3 > 1e-14 * (l1 + l2)) || !(l4 > 1e-14 * (l1 + l2))) return false;
    const ox = w[0] / (l1 || 1) + w[3] / (l2 || 1), oy = w[1] / (l1 || 1) + w[4] / (l2 || 1), oz = w[2] / (l1 || 1) + w[5] / (l2 || 1);
    if (w[6] * ox + w[7] * oy + w[8] * oz <= 0 || w[9] * ox + w[10] * oy + w[11] * oz <= 0) return false;
    if (w[6] * w[9] + w[7] * w[10] + w[8] * w[11] <= 0) return false;
    // perform flip: t = (c,a,d), u = (d,b,c)
    const nbc = N[3 * t + (k + 1) % 3], nca = N[3 * t + (k + 2) % 3];
    const nad = N[3 * u + (j + 1) % 3], ndb = N[3 * u + (j + 2) % 3];
    T[3 * t] = c;
    T[3 * t + 1] = a;
    T[3 * t + 2] = d;
    N[3 * t] = nca;
    N[3 * t + 1] = nad;
    N[3 * t + 2] = u;
    T[3 * u] = d;
    T[3 * u + 1] = b;
    T[3 * u + 2] = c;
    N[3 * u] = ndb;
    N[3 * u + 1] = nbc;
    N[3 * u + 2] = t;
    if (nad >= 0) this.replaceNbr(nad, u, t);
    if (nbc >= 0) this.replaceNbr(nbc, t, u);
    queue.push2(t, 0);
    queue.push2(t, 1);
    queue.push2(u, 0);
    queue.push2(u, 1);
    return true;
  }

  /** Is (c,d) an edge of the patch? Walks the triangle fan around c starting at triangle t. */
  private hasEdge(c: number, d: number, t: number): boolean {
    const T = this.tri.data, N = this.nbr.data;
    for (let dir = 0; dir < 2; dir++) {
      let x = t;
      for (let guard = 0; guard < 4096; guard++) {
        let pc = -1;
        for (let q = 0; q < 3; q++) if (T[3 * x + q] === c) pc = q;
        if (pc < 0) return false;
        if (T[3 * x + (pc + 1) % 3] === d || T[3 * x + (pc + 2) % 3] === d) return true;
        // rotate: dir 0 crosses edge (c -> next), dir 1 crosses edge (prev -> c)
        const y = dir === 0 ? N[3 * x + pc] : N[3 * x + (pc + 2) % 3];
        if (y < 0) break;
        if (y === t) return false; // full fan visited
        x = y;
      }
    }
    return false;
  }
}

function angleAt(p: Float64Array, v: number, a: number, b: number): number {
  const ax = p[3 * a] - p[3 * v], ay = p[3 * a + 1] - p[3 * v + 1], az = p[3 * a + 2] - p[3 * v + 2];
  const bx = p[3 * b] - p[3 * v], by = p[3 * b + 1] - p[3 * v + 1], bz = p[3 * b + 2] - p[3 * v + 2];
  const la = Math.hypot(ax, ay, az), lb = Math.hypot(bx, by, bz);
  if (!(la > 0) || !(lb > 0)) return Math.PI;
  const d = (ax * bx + ay * by + az * bz) / (la * lb);
  return Math.acos(d > 1 ? 1 : d < -1 ? -1 : d);
}

// ---------------------------------------------------------------------------
// Fairing
// ---------------------------------------------------------------------------

/**
 * Builds local one-ring lists (patch + surrounding mesh around loop vertices) and solves for the
 * interior vertex positions with CG. Local ids: [0, m) loop, [m, V) interior, [V, V+R) outer
 * ring (fixed, positions in `outerPos`).
 *
 * smooth = false: membrane (L x = 0 at interior vertices).
 * smooth = true : membrane as initial guess, then thin plate (min Σ‖L‖² over interior + loop
 *   vertices). Additionally, when the hole region is a height field over the loop's best-fit
 *   plane, a quadratic height function z = q(x, y) is fitted to the loop and the surrounding
 *   ring, and the thin plate is applied to the residual z − q instead: surfaces that are locally
 *   quadratic (spheres, cylinders, saddles, gentle free-form) are then reproduced inside the hole
 *   (curvature continuity), while deviations from the quadric still blend smoothly.
 */
function fairPatch(ctx: MeshCtx, patch: Patch, Q: Uint32Array, local: IntFloatMap, smooth: boolean, quadricBase: boolean, edgeN: Float64Array): void {
  const { P, I, slotOf, ringOff, ringTri } = ctx;
  const m = patch.nLoop;
  const V = patch.vertexCount;
  const nU = V - m;
  const T = patch.tri.data, NB = patch.nbr.data;
  const nTri = patch.triCount;
  // --- adjacency lists (CSR) for patch vertices, including outer mesh neighbours of loop vertices
  const outerId = new IntFloatMap(64); // global -> local id >= V
  const outerPos = new GrowableF64(64);
  const pairs = new GrowableI32(nTri * 6);
  for (let t = 0; t < nTri; t++) {
    for (let k = 0; k < 3; k++) {
      const a = T[3 * t + k], b = T[3 * t + (k + 1) % 3];
      if (NB[3 * t + k] < 0 || a < b) {
        pairs.push2(a, b);
        pairs.push2(b, a);
      }
    }
  }
  for (let i = 0; i < m; i++) {
    const g = Q[i];
    const s = slotOf[g];
    for (let j = ringOff[s]; j < ringOff[s + 1]; j++) {
      const t = ringTri[j];
      for (let k = 0; k < 3; k++) {
        const w = I[3 * t + k];
        if (w === g) continue;
        let lid = local.get(w);
        if (!(lid === lid)) {
          lid = outerId.get(w);
          if (!(lid === lid)) {
            lid = V + outerPos.length / 3;
            outerId.set(w, lid);
            outerPos.push3(P[3 * w], P[3 * w + 1], P[3 * w + 2]);
          }
        }
        pairs.push2(i, lid);
      }
    }
  }
  // CSR with dedupe (lists are short)
  const off = new Uint32Array(V + 1);
  const pd = pairs.data;
  for (let q = 0; q < pairs.length; q += 2) off[pd[q] + 1]++;
  for (let v = 0; v < V; v++) off[v + 1] += off[v];
  const adj = new Int32Array(off[V]);
  const fill = off.slice(0, V);
  const deg = new Uint32Array(V);
  for (let q = 0; q < pairs.length; q += 2) {
    const a = pd[q], b = pd[q + 1];
    let dup = false;
    for (let r = off[a]; r < fill[a]; r++)
      if (adj[r] === b) {
        dup = true;
        break;
      }
    if (!dup) adj[fill[a]++] = b;
  }
  for (let v = 0; v < V; v++) deg[v] = fill[v] - off[v];

  const pos = patch.pos.data;
  const op = outerPos.data;
  const nOuter = outerPos.length / 3;
  const isUnknown = (v: number) => v >= m && v < V;
  const x = new Float64Array(nU);
  const rhs = new Float64Array(nU);

  // --- scalar solvers over the patch graph; `fixed(v)` gives the value at loop/outer vertices,
  //     x holds the initial guess for the unknowns and receives the result.
  const applyK = (xx: Float64Array, out: Float64Array) => {
    for (let u = 0; u < nU; u++) {
      const v = m + u;
      let s = deg[v] * xx[u];
      for (let r = off[v], e = fill[v]; r < e; r++) {
        const w = adj[r];
        if (isUnknown(w)) s -= xx[w - m];
      }
      out[u] = s;
    }
  };
  const solveMembrane = (fixed: (v: number) => number) => {
    for (let u = 0; u < nU; u++) {
      const v = m + u;
      let f = 0;
      for (let r = off[v], e = fill[v]; r < e; r++) {
        const w = adj[r];
        if (!isUnknown(w)) f += fixed(w);
      }
      rhs[u] = f;
    }
    conjugateGradient(applyK, rhs, x, Math.min(5000, 200 + 40 * Math.ceil(Math.sqrt(nU))), 1e-10);
  };
  // thin plate: minimise Σ_{s in patch} ‖L(s)‖², L(s) = mean(N(s)) − p_s; A: rows = patch
  // vertices, cols = unknowns; normal equations AᵀA x = −Aᵀ b
  const Ax = new Float64Array(V);
  const applyAtA = (xx: Float64Array, out: Float64Array) => {
    for (let s = 0; s < V; s++) {
      const d = deg[s];
      if (d === 0) {
        Ax[s] = 0;
        continue;
      }
      let acc = 0;
      for (let r = off[s], e = fill[s]; r < e; r++) {
        const w = adj[r];
        if (isUnknown(w)) acc += xx[w - m];
      }
      acc /= d;
      if (isUnknown(s)) acc -= xx[s - m];
      Ax[s] = acc;
    }
    out.fill(0);
    for (let s = 0; s < V; s++) {
      const d = deg[s];
      if (d === 0) continue;
      const r0 = Ax[s];
      const rd = r0 / d;
      for (let r = off[s], e = fill[s]; r < e; r++) {
        const w = adj[r];
        if (isUnknown(w)) out[w - m] += rd;
      }
      if (isUnknown(s)) out[s - m] -= r0;
    }
  };
  const bRow = new Float64Array(V);
  const solveThinPlate = (fixed: (v: number) => number) => {
    for (let s = 0; s < V; s++) {
      const d = deg[s];
      if (d === 0) {
        bRow[s] = 0;
        continue;
      }
      let acc = 0;
      for (let r = off[s], e = fill[s]; r < e; r++) {
        const w = adj[r];
        if (!isUnknown(w)) acc += fixed(w);
      }
      acc /= d;
      if (!isUnknown(s)) acc -= fixed(s);
      bRow[s] = acc;
    }
    rhs.fill(0);
    for (let s = 0; s < V; s++) {
      const d = deg[s];
      if (d === 0) continue;
      const r0 = bRow[s];
      for (let r = off[s], e = fill[s]; r < e; r++) {
        const w = adj[r];
        if (isUnknown(w)) rhs[w - m] -= r0 / d;
      }
      if (isUnknown(s)) rhs[s - m] += r0;
    }
    conjugateGradient(applyAtA, rhs, x, Math.min(20000, 500 + 40 * nU), 1e-11);
  };

  // --- 3D fairing, coordinate by coordinate
  for (let c = 0; c < 3; c++) {
    const fixed = (v: number) => (v < V ? pos[3 * v + c] : op[3 * (v - V) + c]);
    for (let u = 0; u < nU; u++) x[u] = pos[3 * (m + u) + c];
    solveMembrane(fixed);
    if (smooth) solveThinPlate(fixed);
    for (let u = 0; u < nU; u++) pos[3 * (m + u) + c] = x[u];
  }
  if (!smooth || !quadricBase) return;

  // --- quadric base + thin-plate residual (see doc comment) --------------------------------
  // frame: loop centroid, Newell normal of the (fill-oriented) loop polygon
  let cx = 0, cy = 0, cz = 0, nx = 0, ny = 0, nz = 0;
  for (let i = 0; i < m; i++) {
    const j = (i + 1) % m;
    const xi = pos[3 * i], yi = pos[3 * i + 1], zi = pos[3 * i + 2];
    const xj = pos[3 * j], yj = pos[3 * j + 1], zj = pos[3 * j + 2];
    nx += (yi - yj) * (zi + zj);
    ny += (zi - zj) * (xi + xj);
    nz += (xi - xj) * (yi + yj);
    cx += xi / m;
    cy += yi / m;
    cz += zi / m;
  }
  const nl = Math.hypot(nx, ny, nz);
  if (!(nl > 0)) return;
  nx /= nl;
  ny /= nl;
  nz /= nl;
  // height-field test: the surrounding surface must face the frame normal
  for (let i = 0; i < m; i++) {
    const ex = edgeN[3 * i], ey = edgeN[3 * i + 1], ez = edgeN[3 * i + 2];
    if (ex === 0 && ey === 0 && ez === 0) continue;
    if (ex * nx + ey * ny + ez * nz < 0.3) return;
  }
  let ux: number, uy: number, uz: number;
  if (Math.abs(nx) <= Math.abs(ny) && Math.abs(nx) <= Math.abs(nz)) {
    ux = 0;
    uy = nz;
    uz = -ny;
  } else if (Math.abs(ny) <= Math.abs(nz)) {
    ux = -nz;
    uy = 0;
    uz = nx;
  } else {
    ux = ny;
    uy = -nx;
    uz = 0;
  }
  const ul = Math.hypot(ux, uy, uz);
  ux /= ul;
  uy /= ul;
  uz /= ul;
  const vx = ny * uz - nz * uy, vy = nz * ux - nx * uz, vz = nx * uy - ny * ux;
  // local frame coordinates of every involved vertex
  const nAll = V + nOuter;
  const L = new Float64Array(nAll * 3);
  let rad = 0;
  for (let v = 0; v < nAll; v++) {
    const px = v < V ? pos[3 * v] : op[3 * (v - V)];
    const py = v < V ? pos[3 * v + 1] : op[3 * (v - V) + 1];
    const pz = v < V ? pos[3 * v + 2] : op[3 * (v - V) + 2];
    const dx = px - cx, dy = py - cy, dz = pz - cz;
    L[3 * v] = dx * ux + dy * uy + dz * uz;
    L[3 * v + 1] = dx * vx + dy * vy + dz * vz;
    L[3 * v + 2] = dx * nx + dy * ny + dz * nz;
    if (v < m) rad += Math.hypot(L[3 * v], L[3 * v + 1]) / m;
  }
  if (!(rad > 0)) return;
  // least-squares fit z = a x² + b xy + c y² + d x + e y + f over loop + outer ring (scaled coords)
  const M = new Float64Array(36), R = new Float64Array(6), row = new Float64Array(6);
  let nSamples = 0;
  for (let v = 0; v < nAll; v++) {
    if (isUnknown(v)) continue;
    const X = L[3 * v] / rad, Y = L[3 * v + 1] / rad, Z = L[3 * v + 2];
    row[0] = X * X;
    row[1] = X * Y;
    row[2] = Y * Y;
    row[3] = X;
    row[4] = Y;
    row[5] = 1;
    for (let a = 0; a < 6; a++) {
      R[a] += row[a] * Z;
      for (let b = 0; b < 6; b++) M[6 * a + b] += row[a] * row[b];
    }
    nSamples++;
  }
  if (nSamples < 12) return;
  const coef = solveLinear(M, R, 6);
  if (!coef || !coef.every(Number.isFinite)) return;
  const qz = (v: number) => {
    const X = L[3 * v] / rad, Y = L[3 * v + 1] / rad;
    return coef[0] * X * X + coef[1] * X * Y + coef[2] * Y * Y + coef[3] * X + coef[4] * Y + coef[5];
  };
  // residual field: fixed at loop/outer, thin plate inside
  const fixedRes = (v: number) => L[3 * v + 2] - qz(v);
  x.fill(0);
  solveThinPlate(fixedRes);
  for (let u = 0; u < nU; u++) {
    const v = m + u;
    const z = qz(v) + x[u];
    const X = L[3 * v], Y = L[3 * v + 1];
    pos[3 * v] = cx + X * ux + Y * vx + z * nx;
    pos[3 * v + 1] = cy + X * uy + Y * vy + z * ny;
    pos[3 * v + 2] = cz + X * uz + Y * vz + z * nz;
  }
}

/** Plain CG for SPD operators; x holds the initial guess and receives the solution. */
function conjugateGradient(apply: (x: Float64Array, out: Float64Array) => void, b: Float64Array, x: Float64Array, maxIter: number, relTol: number): number {
  const n = b.length;
  const r = new Float64Array(n), p = new Float64Array(n), Ap = new Float64Array(n);
  apply(x, Ap);
  let bn = 0;
  for (let i = 0; i < n; i++) {
    r[i] = b[i] - Ap[i];
    p[i] = r[i];
    bn += b[i] * b[i];
  }
  let rr = 0;
  for (let i = 0; i < n; i++) rr += r[i] * r[i];
  const stop = relTol * relTol * Math.max(bn, 1e-300);
  let it = 0;
  for (; it < maxIter && rr > stop; it++) {
    apply(p, Ap);
    let pAp = 0;
    for (let i = 0; i < n; i++) pAp += p[i] * Ap[i];
    if (!(pAp > 0)) break;
    const alpha = rr / pAp;
    let rr2 = 0;
    for (let i = 0; i < n; i++) {
      x[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
      rr2 += r[i] * r[i];
    }
    const beta = rr2 / rr;
    rr = rr2;
    for (let i = 0; i < n; i++) p[i] = r[i] + beta * p[i];
  }
  return it;
}

/** Harmonic (membrane) interpolation of loop colours to the interior patch vertices. */
function interpolateColors(patch: Patch, Q: Uint32Array, C: Uint8Array): Float32Array {
  const m = patch.nLoop;
  const V = patch.vertexCount;
  const nU = V - m;
  const out = new Float32Array(nU * 3);
  if (nU === 0) return out;
  const T = patch.tri.data, NB = patch.nbr.data;
  const nTri = patch.triCount;
  // neighbour sums via edge list (Jacobi iterations are enough: colours are low-precision)
  const ea = new GrowableI32(nTri * 3), eb = new GrowableI32(nTri * 3);
  for (let t = 0; t < nTri; t++)
    for (let k = 0; k < 3; k++) {
      const a = T[3 * t + k], b = T[3 * t + (k + 1) % 3];
      if (NB[3 * t + k] < 0 || a < b) {
        ea.push(a);
        eb.push(b);
      }
    }
  const val = new Float32Array(V * 3);
  let avg0 = 0, avg1 = 0, avg2 = 0;
  for (let i = 0; i < m; i++) {
    val[3 * i] = C[3 * Q[i]];
    val[3 * i + 1] = C[3 * Q[i] + 1];
    val[3 * i + 2] = C[3 * Q[i] + 2];
    avg0 += val[3 * i];
    avg1 += val[3 * i + 1];
    avg2 += val[3 * i + 2];
  }
  for (let v = m; v < V; v++) {
    val[3 * v] = avg0 / m;
    val[3 * v + 1] = avg1 / m;
    val[3 * v + 2] = avg2 / m;
  }
  const sum = new Float32Array(V * 3), cnt = new Float32Array(V);
  for (let e = 0; e < ea.length; e++) {
    cnt[ea.data[e]]++;
    cnt[eb.data[e]]++;
  }
  const iters = Math.min(200, 20 + Math.ceil(Math.sqrt(nU)) * 4);
  for (let it = 0; it < iters; it++) {
    sum.fill(0);
    for (let e = 0; e < ea.length; e++) {
      const a = ea.data[e], b = eb.data[e];
      for (let c = 0; c < 3; c++) {
        sum[3 * a + c] += val[3 * b + c];
        sum[3 * b + c] += val[3 * a + c];
      }
    }
    for (let v = m; v < V; v++) if (cnt[v] > 0) for (let c = 0; c < 3; c++) val[3 * v + c] = sum[3 * v + c] / cnt[v];
  }
  out.set(val.subarray(m * 3));
  return out;
}

/** Convenience: per-loop summary for UIs listing holes. */
export interface HoleInfo {
  index: number;
  edgeCount: number;
  /** loop perimeter length */
  perimeter: number;
}

export function listHoles(mesh: MeshData, neighbors?: Int32Array): HoleInfo[] {
  const loops = findBoundaryLoops(mesh, neighbors ?? buildTriangleNeighbors(mesh));
  const P = mesh.positions;
  return loops.map((L, index) => {
    let per = 0;
    for (let i = 0; i < L.length; i++) {
      const a = L[i], b = L[(i + 1) % L.length];
      per += Math.hypot(P[3 * a] - P[3 * b], P[3 * a + 1] - P[3 * b + 1], P[3 * a + 2] - P[3 * b + 2]);
    }
    return { index, edgeCount: L.length, perimeter: per };
  });
}
