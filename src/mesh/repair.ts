/**
 * Mesh repair & basic editing: small-shell removal, degenerate/duplicate face removal,
 * vertex welding, normal flipping, consistent orientation, triangle deletion.
 *
 * All functions return a new MeshData (inputs are never mutated) and run in linear or
 * near-linear time with typed-array storage only.
 */
import type { MeshData, ProgressFn } from "../core/types";
import {
  BOUNDARY,
  buildTriangleNeighbors,
  compactMesh,
  compactMeshWithMap,
  computeVertexNormals,
  connectedComponents,
  triangleCountOf,
  vertexCountOf,
} from "./topology";
import { ExactVertexWelder, fmix32 } from "./util/weld-hash";

const PROGRESS_STEP = 1 << 20;

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

export interface RemoveSmallComponentsOptions {
  /** Keep only the largest component (overrides minTriangles). */
  keepLargestOnly?: boolean;
  onProgress?: ProgressFn;
}

/**
 * Remove connected components ("shells", vertex-connected) with fewer than `minTriangles`
 * triangles. Unreferenced vertices are removed as well.
 */
export function removeSmallComponents(mesh: MeshData, minTriangles: number, opts: RemoveSmallComponentsOptions = {}): MeshData {
  const nt = triangleCountOf(mesh);
  const comps = connectedComponents(mesh, opts.onProgress);
  const keepComp = new Uint8Array(comps.count);
  if (opts.keepLargestOnly) {
    let best = -1, bestSize = -1;
    for (let c = 0; c < comps.count; c++)
      if (comps.componentSizes[c] > bestSize) {
        bestSize = comps.componentSizes[c];
        best = c;
      }
    if (best >= 0) keepComp[best] = 1;
  } else {
    for (let c = 0; c < comps.count; c++) keepComp[c] = comps.componentSizes[c] >= minTriangles ? 1 : 0;
  }
  const keep = new Uint8Array(nt);
  for (let t = 0; t < nt; t++) keep[t] = keepComp[comps.triangleComponent[t]];
  return compactMesh(mesh, keep);
}

// ---------------------------------------------------------------------------
// Degenerate & duplicate triangles
// ---------------------------------------------------------------------------

export interface DegenerateOptions {
  /**
   * A triangle is degenerate when |cross(e1,e2)| <= relAreaTolerance * (longest edge)^2,
   * i.e. 2·area relative to the squared longest edge (≈ height/length). Default 1e-7, which
   * catches float32 rounding noise on collinear vertices. Triangles with repeated vertex
   * indices are always removed.
   */
  relAreaTolerance?: number;
  /** Remove duplicate faces (same vertex set, any orientation), keeping the first. Default true. */
  removeDuplicates?: boolean;
  onProgress?: ProgressFn;
}

/** Remove zero-area / repeated-index triangles and duplicate faces; drops unreferenced vertices. */
export function removeDegenerateTriangles(mesh: MeshData, opts: DegenerateOptions = {}): MeshData {
  const keep = degenerateTriangleMask(mesh, opts);
  for (let t = 0; t < keep.length; t++) keep[t] = keep[t] ? 0 : 1;
  return compactMesh(mesh, keep);
}

/** Per-triangle mask: 1 = degenerate or duplicate (would be removed by removeDegenerateTriangles). */
export function degenerateTriangleMask(mesh: MeshData, opts: DegenerateOptions = {}): Uint8Array {
  const P = mesh.positions, I = mesh.indices;
  const nt = triangleCountOf(mesh);
  const nv = vertexCountOf(mesh);
  const tol = opts.relAreaTolerance ?? 1e-7;
  const bad = new Uint8Array(nt);
  for (let t = 0; t < nt; t++) {
    const ia = I[3 * t], ib = I[3 * t + 1], ic = I[3 * t + 2];
    if (ia === ib || ib === ic || ic === ia || ia >= nv || ib >= nv || ic >= nv) {
      bad[t] = 1;
      continue;
    }
    const a = 3 * ia, b = 3 * ib, c = 3 * ic;
    const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2];
    const vx = P[c] - P[a], vy = P[c + 1] - P[a + 1], vz = P[c + 2] - P[a + 2];
    const wx = P[c] - P[b], wy = P[c + 1] - P[b + 1], wz = P[c + 2] - P[b + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const cr = Math.sqrt(nx * nx + ny * ny + nz * nz);
    const l2 = Math.max(ux * ux + uy * uy + uz * uz, vx * vx + vy * vy + vz * vz, wx * wx + wy * wy + wz * wz);
    if (!(cr > tol * l2)) bad[t] = 1; // also catches NaN
    if (opts.onProgress && (t & (PROGRESS_STEP - 1)) === 0) opts.onProgress((0.5 * t) / nt, "Checking triangles");
  }
  if (opts.removeDuplicates ?? true) markDuplicateFaces(mesh, bad);
  return bad;
}

/**
 * Marks (in `bad`) every face whose sorted vertex triple equals that of an earlier face that is
 * not already marked. Bucketing by the smallest vertex (counting sort) → linear time.
 */
function markDuplicateFaces(mesh: MeshData, bad: Uint8Array): void {
  const I = mesh.indices;
  const nt = triangleCountOf(mesh);
  const nv = vertexCountOf(mesh);
  const offsets = new Uint32Array(nv + 1);
  for (let t = 0; t < nt; t++) {
    if (bad[t]) continue;
    offsets[Math.min(I[3 * t], I[3 * t + 1], I[3 * t + 2]) + 1]++;
  }
  for (let v = 0; v < nv; v++) offsets[v + 1] += offsets[v];
  const bucket = new Uint32Array(offsets[nv]);
  for (let t = 0; t < nt; t++) {
    if (bad[t]) continue;
    bucket[offsets[Math.min(I[3 * t], I[3 * t + 1], I[3 * t + 2])]++] = t;
  }
  for (let v = nv; v > 0; v--) offsets[v] = offsets[v - 1];
  offsets[0] = 0;
  for (let v = 0; v < nv; v++) {
    const s = offsets[v], e = offsets[v + 1];
    if (e - s < 2) continue;
    // buckets are tiny on real meshes: quadratic scan is fine; fall back to sorting for huge fans
    if (e - s <= 64) {
      for (let i = s; i < e; i++) {
        const ti = bucket[i];
        if (bad[ti]) continue;
        const a = I[3 * ti], b = I[3 * ti + 1], c = I[3 * ti + 2];
        const mid = a + b + c - v - Math.max(a, b, c), mx = Math.max(a, b, c);
        for (let j = i + 1; j < e; j++) {
          const tj = bucket[j];
          if (bad[tj]) continue;
          const a2 = I[3 * tj], b2 = I[3 * tj + 1], c2 = I[3 * tj + 2];
          const mx2 = Math.max(a2, b2, c2);
          if (mx2 === mx && a2 + b2 + c2 - v - mx2 === mid) bad[tj] = 1;
        }
      }
    } else {
      const list = Array.from(bucket.subarray(s, e));
      const key = (t: number) => {
        const a = I[3 * t], b = I[3 * t + 1], c = I[3 * t + 2];
        const mx = Math.max(a, b, c);
        return [a + b + c - v - mx, mx];
      };
      list.sort((x, y) => {
        const kx = key(x), ky = key(y);
        return kx[0] - ky[0] || kx[1] - ky[1] || x - y;
      });
      for (let i = 1; i < list.length; i++) {
        const k0 = key(list[i - 1]), k1 = key(list[i]);
        if (k0[0] === k1[0] && k0[1] === k1[1]) bad[list[i]] = 1;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Welding
// ---------------------------------------------------------------------------

export interface WeldResult {
  mesh: MeshData;
  /** old vertex → new vertex (-1 if the vertex ended up unreferenced) */
  vertexMap: Int32Array;
}

/**
 * Merge vertices closer than `tolerance` (exactly equal positions when tolerance <= 0).
 * Greedy clustering in input order: each vertex snaps to the nearest earlier representative
 * within tolerance (uniform hash grid with cell = tolerance, 27-cell search). Triangles that
 * become degenerate (repeated vertex) are removed, as are unreferenced vertices. Normals and
 * colours of the representative are kept.
 */
export function weldVertices(mesh: MeshData, tolerance: number, onProgress?: ProgressFn): MeshData {
  return weldVerticesWithMap(mesh, tolerance, onProgress).mesh;
}

export function weldVerticesWithMap(mesh: MeshData, tolerance: number, onProgress?: ProgressFn): WeldResult {
  const nv = vertexCountOf(mesh);
  const repOf = tolerance > 0 ? weldGridRepresentatives(mesh.positions, tolerance, onProgress) : weldExactRepresentatives(mesh.positions);
  // remap triangles to representatives, drop degenerate
  const I = mesh.indices;
  const nt = triangleCountOf(mesh);
  const newIdx = new Uint32Array(nt * 3);
  const keep = new Uint8Array(nt);
  for (let t = 0; t < nt; t++) {
    const a = I[3 * t], b = I[3 * t + 1], c = I[3 * t + 2];
    if (a >= nv || b >= nv || c >= nv) continue;
    const ra = repOf[a], rb = repOf[b], rc = repOf[c];
    newIdx[3 * t] = ra;
    newIdx[3 * t + 1] = rb;
    newIdx[3 * t + 2] = rc;
    keep[t] = ra !== rb && rb !== rc && rc !== ra ? 1 : 0;
  }
  const tmp: MeshData = { positions: mesh.positions, indices: newIdx };
  if (mesh.normals) tmp.normals = mesh.normals;
  if (mesh.colors) tmp.colors = mesh.colors;
  const res = compactMeshWithMap(tmp, keep);
  // compose: old -> rep -> compacted
  const vertexMap = new Int32Array(nv);
  for (let v = 0; v < nv; v++) vertexMap[v] = res.vertexMap[repOf[v]];
  return { mesh: res.mesh, vertexMap };
}

function weldExactRepresentatives(P: Float32Array): Uint32Array {
  const nv = Math.floor(P.length / 3);
  const welder = new ExactVertexWelder(nv);
  const firstOf = new Uint32Array(nv); // unique id -> first vertex having it
  const rep = new Uint32Array(nv);
  for (let v = 0; v < nv; v++) {
    const before = welder.count;
    const id = welder.add(P[3 * v], P[3 * v + 1], P[3 * v + 2]);
    if (id === before) firstOf[id] = v;
    rep[v] = firstOf[id];
  }
  return rep;
}

/**
 * Greedy tolerance clustering on a hash grid. Returns rep[v] = representative vertex (an input
 * vertex index; rep[r] === r for representatives).
 */
function weldGridRepresentatives(P: Float32Array, tol: number, onProgress?: ProgressFn): Uint32Array {
  const nv = Math.floor(P.length / 3);
  const rep = new Uint32Array(nv);
  const nextRep = new Int32Array(nv).fill(-1); // chain of representatives in the same cell
  let cap = 1024;
  while (cap < nv * 1.5) cap *= 2;
  let table = new Int32Array(cap).fill(-1); // slot -> head representative of a cell
  let mask = cap - 1;
  let used = 0;
  const inv = 1 / tol;
  const tol2 = tol * tol;
  const cellHash = (cx: number, cy: number, cz: number) =>
    fmix32((cx >>> 0) ^ fmix32((cy >>> 0) ^ fmix32((cz >>> 0) + 0x632be5ab)));
  const findSlot = (cx: number, cy: number, cz: number): number => {
    let s = cellHash(cx, cy, cz) & mask;
    for (;;) {
      const h = table[s];
      if (h < 0) return s;
      if (Math.floor(P[3 * h] * inv) === cx && Math.floor(P[3 * h + 1] * inv) === cy && Math.floor(P[3 * h + 2] * inv) === cz) return s;
      s = (s + 1) & mask;
    }
  };
  const rehash = () => {
    const old = table;
    cap *= 2;
    table = new Int32Array(cap).fill(-1);
    mask = cap - 1;
    for (let s = 0; s < old.length; s++) {
      const h = old[s];
      if (h < 0) continue;
      const slot = findSlot(Math.floor(P[3 * h] * inv), Math.floor(P[3 * h + 1] * inv), Math.floor(P[3 * h + 2] * inv));
      table[slot] = h;
    }
  };
  for (let v = 0; v < nv; v++) {
    const x = P[3 * v], y = P[3 * v + 1], z = P[3 * v + 2];
    const cx = Math.floor(x * inv), cy = Math.floor(y * inv), cz = Math.floor(z * inv);
    let best = -1, bestD = tol2;
    for (let dx = -1; dx <= 1; dx++)
      for (let dy = -1; dy <= 1; dy++)
        for (let dz = -1; dz <= 1; dz++) {
          const h = table[findSlot(cx + dx, cy + dy, cz + dz)];
          for (let r = h; r >= 0; r = nextRep[r]) {
            const ex = P[3 * r] - x, ey = P[3 * r + 1] - y, ez = P[3 * r + 2] - z;
            const d = ex * ex + ey * ey + ez * ez;
            if (d <= bestD) {
              bestD = d;
              best = r;
            }
          }
        }
    if (best >= 0) {
      rep[v] = best;
      continue;
    }
    rep[v] = v;
    const slot = findSlot(cx, cy, cz);
    if (table[slot] < 0) {
      table[slot] = v;
      used++;
      if (used * 2 > cap) rehash();
    } else {
      nextRep[v] = table[slot];
      table[slot] = v;
    }
    if (onProgress && (v & (PROGRESS_STEP - 1)) === 0) onProgress(v / nv, "Welding vertices");
  }
  return rep;
}

// ---------------------------------------------------------------------------
// Orientation
// ---------------------------------------------------------------------------

/** Reverse every triangle's winding (and negate vertex normals when present). */
export function flipNormals(mesh: MeshData): MeshData {
  const I = mesh.indices;
  const nt = triangleCountOf(mesh);
  const indices = new Uint32Array(nt * 3);
  for (let t = 0; t < nt; t++) {
    indices[3 * t] = I[3 * t];
    indices[3 * t + 1] = I[3 * t + 2];
    indices[3 * t + 2] = I[3 * t + 1];
  }
  const out: MeshData = { positions: mesh.positions.slice(), indices };
  if (mesh.normals) {
    const n = mesh.normals.slice();
    for (let i = 0; i < n.length; i++) n[i] = -n[i];
    out.normals = n;
  }
  if (mesh.colors) out.colors = mesh.colors.slice();
  return out;
}

export interface OrientOptions {
  /** precomputed triangle neighbours (buildTriangleNeighbors) */
  neighbors?: Int32Array;
  onProgress?: ProgressFn;
}

/**
 * Make triangle windings consistent within every edge-connected patch (BFS over manifold
 * edges). Then per patch: closed patches are oriented so the enclosed volume is positive
 * (normals outward); open patches keep the orientation shared by the majority of their
 * triangles (fewest flips). Non-orientable patches (Möbius-like) are left as found by the BFS.
 * Returns a new mesh; vertex normals, if present, are recomputed by the caller when needed
 * (they are copied, and negated per vertex only if all incident triangles flipped — simpler to
 * recompute; here we drop them if any triangle flipped).
 */
export function orientConsistently(mesh: MeshData, opts: OrientOptions = {}): MeshData {
  const I = mesh.indices, P = mesh.positions;
  const nt = triangleCountOf(mesh);
  const nbr = opts.neighbors ?? buildTriangleNeighbors(mesh);
  const flip = new Uint8Array(nt);
  const visited = new Uint8Array(nt);
  const queue = new Uint32Array(nt);
  let qTail = 0;
  let anyFlip = false;
  for (let seed = 0; seed < nt; seed++) {
    if (visited[seed]) continue;
    const qStart = qTail;
    let qHead = qTail;
    queue[qTail++] = seed;
    visited[seed] = 1;
    let open = false;
    while (qHead < qTail) {
      const t = queue[qHead++];
      const dt = flip[t] ? -1 : 1;
      for (let k = 0; k < 3; k++) {
        const n = nbr[3 * t + k];
        if (n < 0) {
          if (n === BOUNDARY) open = true;
          continue;
        }
        if (visited[n]) continue;
        const a = I[3 * t + k], b = I[3 * t + (k === 2 ? 0 : k + 1)];
        // does n (original winding) contain the directed edge a->b ?
        let same = false;
        for (let j = 0; j < 3; j++) {
          if (I[3 * n + j] === a) {
            same = I[3 * n + (j === 2 ? 0 : j + 1)] === b;
            break;
          }
        }
        const dnOrig = same ? 1 : -1;
        flip[n] = dnOrig === dt ? 1 : 0;
        visited[n] = 1;
        queue[qTail++] = n;
      }
    }
    // choose the global orientation of this patch
    let invert = false;
    if (!open) {
      let vol = 0;
      for (let q = qStart; q < qTail; q++) {
        const t = queue[q];
        let a = 3 * I[3 * t], b = 3 * I[3 * t + 1], c = 3 * I[3 * t + 2];
        if (flip[t]) {
          const tmp = b;
          b = c;
          c = tmp;
        }
        vol +=
          P[a] * (P[b + 1] * P[c + 2] - P[b + 2] * P[c + 1]) -
          P[a + 1] * (P[b] * P[c + 2] - P[b + 2] * P[c]) +
          P[a + 2] * (P[b] * P[c + 1] - P[b + 1] * P[c]);
      }
      invert = vol < 0;
    } else {
      let flipped = 0;
      for (let q = qStart; q < qTail; q++) flipped += flip[queue[q]];
      invert = flipped * 2 > qTail - qStart;
    }
    for (let q = qStart; q < qTail; q++) {
      const t = queue[q];
      if (invert) flip[t] ^= 1;
      if (flip[t]) anyFlip = true;
    }
    if (opts.onProgress && (seed & (PROGRESS_STEP - 1)) === 0) opts.onProgress(qTail / nt, "Orienting triangles");
  }
  const indices = new Uint32Array(nt * 3);
  for (let t = 0; t < nt; t++) {
    indices[3 * t] = I[3 * t];
    if (flip[t]) {
      indices[3 * t + 1] = I[3 * t + 2];
      indices[3 * t + 2] = I[3 * t + 1];
    } else {
      indices[3 * t + 1] = I[3 * t + 1];
      indices[3 * t + 2] = I[3 * t + 2];
    }
  }
  const out: MeshData = { positions: mesh.positions.slice(), indices };
  if (mesh.normals) out.normals = anyFlip ? computeVertexNormals(out) : mesh.normals.slice();
  if (mesh.colors) out.colors = mesh.colors.slice();
  return out;
}

// ---------------------------------------------------------------------------
// Deletion
// ---------------------------------------------------------------------------

/** Delete the triangles with mask[t] !== 0 and any vertices left unreferenced. */
export function deleteTriangles(mesh: MeshData, mask: Uint8Array): MeshData {
  const nt = triangleCountOf(mesh);
  const keep = new Uint8Array(nt);
  for (let t = 0; t < nt; t++) keep[t] = mask[t] ? 0 : 1;
  return compactMesh(mesh, keep);
}
