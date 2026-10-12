/**
 * Mesh topology & basic geometric queries on indexed triangle meshes.
 *
 * Everything here is linear (or near-linear) time and allocates only typed arrays:
 *  - vertex → triangle adjacency in CSR form (offsets + flat list),
 *  - edge / triangle-neighbour tables built by bucketing half-edges on their smaller vertex
 *    (a counting sort, no hashing) and grouping each bucket by the larger vertex,
 *  - boundary loops, connected components (union–find on vertices), bounds, area, compaction.
 *
 * Conventions: triangle t has corners indices[3t..3t+2]; its edge k (k = 0,1,2) runs from
 * corner k to corner (k+1)%3; "half-edge" id he = 3t + k.
 */
import type { BBox3, MeshData, ProgressFn } from "../core/types";
import { GrowableI32, GrowableU32 } from "./util/growable";

/** triangleNeighbors value for a boundary edge (exactly one incident triangle). */
export const BOUNDARY = -1;
/** triangleNeighbors value for a non-manifold edge (> 2 triangles) or a degenerate edge (a == b). */
export const NON_MANIFOLD = -2;
/** triangleEdges value for a degenerate edge (both endpoints equal) or an invalid index. */
export const NO_EDGE = 0xffffffff;

export const triangleCountOf = (mesh: MeshData): number => Math.floor(mesh.indices.length / 3);
export const vertexCountOf = (mesh: MeshData): number => Math.floor(mesh.positions.length / 3);

const PROGRESS_STEP = 1 << 20;

// ---------------------------------------------------------------------------
// Normals
// ---------------------------------------------------------------------------

/** Unit face normals (3 per triangle, CCW = outward). Degenerate triangles get (0,0,0). */
export function computeFaceNormals(mesh: MeshData): Float32Array {
  const P = mesh.positions, I = mesh.indices;
  const nt = triangleCountOf(mesh);
  const out = new Float32Array(nt * 3);
  for (let t = 0; t < nt; t++) {
    const a = 3 * I[3 * t], b = 3 * I[3 * t + 1], c = 3 * I[3 * t + 2];
    const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2];
    const vx = P[c] - P[a], vy = P[c + 1] - P[a + 1], vz = P[c + 2] - P[a + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const l = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (l > 0) {
      out[3 * t] = nx / l;
      out[3 * t + 1] = ny / l;
      out[3 * t + 2] = nz / l;
    }
  }
  return out;
}

/**
 * Area-weighted vertex normals (sum of un-normalised face normals, then normalised).
 * Vertices not referenced by any non-degenerate triangle get (0,0,0).
 */
export function computeVertexNormals(mesh: MeshData): Float32Array {
  const P = mesh.positions, I = mesh.indices;
  const nt = triangleCountOf(mesh);
  const nv = vertexCountOf(mesh);
  const N = new Float32Array(nv * 3);
  for (let t = 0; t < nt; t++) {
    const ia = I[3 * t], ib = I[3 * t + 1], ic = I[3 * t + 2];
    const a = 3 * ia, b = 3 * ib, c = 3 * ic;
    const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2];
    const vx = P[c] - P[a], vy = P[c + 1] - P[a + 1], vz = P[c + 2] - P[a + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    N[a] += nx; N[a + 1] += ny; N[a + 2] += nz;
    N[b] += nx; N[b + 1] += ny; N[b + 2] += nz;
    N[c] += nx; N[c + 1] += ny; N[c + 2] += nz;
  }
  for (let v = 0; v < nv; v++) {
    const x = N[3 * v], y = N[3 * v + 1], z = N[3 * v + 2];
    const l = Math.sqrt(x * x + y * y + z * z);
    if (l > 0) {
      N[3 * v] = x / l;
      N[3 * v + 1] = y / l;
      N[3 * v + 2] = z / l;
    }
  }
  return N;
}

// ---------------------------------------------------------------------------
// Vertex → triangle CSR
// ---------------------------------------------------------------------------

export interface VertexTriangles {
  /** length vertexCount + 1; triangles of vertex v are triangles[offsets[v] .. offsets[v+1]) */
  offsets: Uint32Array;
  /** flat triangle ids (a triangle appears once per corner referencing v) */
  triangles: Uint32Array;
}

/** Vertex → incident triangles in CSR form. O(n), 4 B per vertex + 12 B per triangle. */
export function buildVertexTriangles(mesh: MeshData): VertexTriangles {
  const I = mesh.indices;
  const nv = vertexCountOf(mesh);
  const nc = triangleCountOf(mesh) * 3;
  const offsets = new Uint32Array(nv + 1);
  for (let c = 0; c < nc; c++) {
    const v = I[c];
    if (v < nv) offsets[v + 1]++;
  }
  for (let v = 0; v < nv; v++) offsets[v + 1] += offsets[v];
  const triangles = new Uint32Array(offsets[nv]);
  // fill using offsets[v] as a cursor, then shift back
  for (let c = 0; c < nc; c++) {
    const v = I[c];
    if (v < nv) triangles[offsets[v]++] = (c / 3) | 0;
  }
  for (let v = nv; v > 0; v--) offsets[v] = offsets[v - 1];
  offsets[0] = 0;
  return { offsets, triangles };
}

// ---------------------------------------------------------------------------
// Edges & triangle neighbours (sort-based half-edge pairing)
// ---------------------------------------------------------------------------

export interface EdgeTopology {
  edgeCount: number;
  /** 2 per edge: (lo, hi) vertex ids, lo < hi. Edges are sorted by (lo, hi). */
  edgeVertices: Uint32Array;
  /** 2 per edge: first two incident triangles (-1 when absent). */
  edgeTriangles: Int32Array;
  /** number of incident triangles per edge (saturates at 255). 1 = boundary, 2 = manifold, >2 = non-manifold. */
  edgeFaceCount: Uint8Array;
  /** 3 per triangle: edge id of edge k (corner k → corner k+1), NO_EDGE for degenerate edges. */
  triangleEdges: Uint32Array;
  /**
   * 3 per triangle: neighbouring triangle across edge k; BOUNDARY (-1) on a boundary edge,
   * NON_MANIFOLD (-2) on an edge shared by more than two triangles or a degenerate edge.
   */
  triangleNeighbors: Int32Array;
}

/**
 * Buckets half-edges by their smaller vertex (counting sort) and sorts every bucket by the
 * larger vertex, in place. Returns the CSR so callers can walk runs of equal (lo, hi).
 */
function bucketHalfEdges(mesh: MeshData): { offsets: Uint32Array; bucket: Uint32Array } {
  const I = mesh.indices;
  const nv = vertexCountOf(mesh);
  const nt = triangleCountOf(mesh);
  const offsets = new Uint32Array(nv + 1);
  for (let t = 0; t < nt; t++) {
    const o = 3 * t;
    const v0 = I[o], v1 = I[o + 1], v2 = I[o + 2];
    if (v0 >= nv || v1 >= nv || v2 >= nv) continue;
    if (v0 !== v1) offsets[(v0 < v1 ? v0 : v1) + 1]++;
    if (v1 !== v2) offsets[(v1 < v2 ? v1 : v2) + 1]++;
    if (v2 !== v0) offsets[(v2 < v0 ? v2 : v0) + 1]++;
  }
  for (let v = 0; v < nv; v++) offsets[v + 1] += offsets[v];
  const bucket = new Uint32Array(offsets[nv]);
  for (let t = 0; t < nt; t++) {
    const o = 3 * t;
    const v0 = I[o], v1 = I[o + 1], v2 = I[o + 2];
    if (v0 >= nv || v1 >= nv || v2 >= nv) continue;
    if (v0 !== v1) bucket[offsets[v0 < v1 ? v0 : v1]++] = o;
    if (v1 !== v2) bucket[offsets[v1 < v2 ? v1 : v2]++] = o + 1;
    if (v2 !== v0) bucket[offsets[v2 < v0 ? v2 : v0]++] = o + 2;
  }
  for (let v = nv; v > 0; v--) offsets[v] = offsets[v - 1];
  offsets[0] = 0;

  // sort each bucket by the larger endpoint
  let keys = new Float64Array(64);
  for (let v = 0; v < nv; v++) {
    const s = offsets[v], e = offsets[v + 1];
    const n = e - s;
    if (n < 2) continue;
    if (n <= 16) {
      // insertion sort by hi
      for (let i = s + 1; i < e; i++) {
        const he = bucket[i];
        const hk = hiOf(I, he);
        let j = i - 1;
        while (j >= s && hiOf(I, bucket[j]) > hk) {
          bucket[j + 1] = bucket[j];
          j--;
        }
        bucket[j + 1] = he;
      }
    } else if (n < 1 << 21) {
      // pack (hi, local position) into an exactly representable double and sort natively
      if (keys.length < n) keys = new Float64Array(n * 2);
      const kv = keys.subarray(0, n);
      for (let i = 0; i < n; i++) kv[i] = hiOf(I, bucket[s + i]) * 2097152 + i;
      kv.sort();
      const tmp = bucket.slice(s, e);
      for (let i = 0; i < n; i++) bucket[s + i] = tmp[kv[i] % 2097152];
    } else {
      const sub = bucket.subarray(s, e);
      const tmp = Array.from(sub);
      tmp.sort((x, y) => hiOf(I, x) - hiOf(I, y) || x - y);
      sub.set(tmp);
    }
  }
  return { offsets, bucket };
}

function hiOf(I: Uint32Array, he: number): number {
  const k = he % 3;
  const a = I[he], b = I[k === 2 ? he - 2 : he + 1];
  return a > b ? a : b;
}

/**
 * Triangle neighbours only (3 per triangle, see EdgeTopology.triangleNeighbors). Lighter than
 * buildEdges: 12 B per triangle output + ~16 B per triangle transient.
 */
export function buildTriangleNeighbors(mesh: MeshData, onProgress?: ProgressFn): Int32Array {
  const I = mesh.indices;
  const nv = vertexCountOf(mesh);
  const nt = triangleCountOf(mesh);
  const nbr = new Int32Array(nt * 3).fill(NON_MANIFOLD);
  const { offsets, bucket } = bucketHalfEdges(mesh);
  for (let v = 0; v < nv; v++) {
    const e = offsets[v + 1];
    let i = offsets[v];
    while (i < e) {
      const hi = hiOf(I, bucket[i]);
      let j = i + 1;
      while (j < e && hiOf(I, bucket[j]) === hi) j++;
      const run = j - i;
      if (run === 1) nbr[bucket[i]] = BOUNDARY;
      else if (run === 2) {
        const h0 = bucket[i], h1 = bucket[i + 1];
        const t0 = (h0 / 3) | 0, t1 = (h1 / 3) | 0;
        if (t0 !== t1) {
          nbr[h0] = t1;
          nbr[h1] = t0;
        }
      }
      i = j;
    }
    if (onProgress && (v & (PROGRESS_STEP - 1)) === 0) onProgress(v / nv, "Building triangle adjacency");
  }
  return nbr;
}

/**
 * Unique undirected edges with their incident triangles, the triangle→edge table and the
 * triangle-neighbour table. Memory ≈ 4 B/vertex + ~60 B/triangle (output) for a manifold mesh.
 */
export function buildEdges(mesh: MeshData, onProgress?: ProgressFn): EdgeTopology {
  const I = mesh.indices;
  const nv = vertexCountOf(mesh);
  const nt = triangleCountOf(mesh);
  const { offsets, bucket } = bucketHalfEdges(mesh);
  // pass 1: count unique edges
  let ne = 0;
  for (let v = 0; v < nv; v++) {
    const e = offsets[v + 1];
    let i = offsets[v];
    let prev = -1;
    for (; i < e; i++) {
      const hi = hiOf(I, bucket[i]);
      if (hi !== prev) {
        ne++;
        prev = hi;
      }
    }
  }
  const edgeVertices = new Uint32Array(ne * 2);
  const edgeTriangles = new Int32Array(ne * 2).fill(-1);
  const edgeFaceCount = new Uint8Array(ne);
  const triangleEdges = new Uint32Array(nt * 3).fill(NO_EDGE);
  const triangleNeighbors = new Int32Array(nt * 3).fill(NON_MANIFOLD);
  let eid = 0;
  for (let v = 0; v < nv; v++) {
    const e = offsets[v + 1];
    let i = offsets[v];
    while (i < e) {
      const hi = hiOf(I, bucket[i]);
      let j = i + 1;
      while (j < e && hiOf(I, bucket[j]) === hi) j++;
      const run = j - i;
      edgeVertices[2 * eid] = v;
      edgeVertices[2 * eid + 1] = hi;
      edgeFaceCount[eid] = run > 255 ? 255 : run;
      edgeTriangles[2 * eid] = (bucket[i] / 3) | 0;
      if (run > 1) edgeTriangles[2 * eid + 1] = (bucket[i + 1] / 3) | 0;
      for (let k = i; k < j; k++) triangleEdges[bucket[k]] = eid;
      if (run === 1) triangleNeighbors[bucket[i]] = BOUNDARY;
      else if (run === 2) {
        const h0 = bucket[i], h1 = bucket[i + 1];
        const t0 = (h0 / 3) | 0, t1 = (h1 / 3) | 0;
        if (t0 !== t1) {
          triangleNeighbors[h0] = t1;
          triangleNeighbors[h1] = t0;
        }
      }
      eid++;
      i = j;
    }
    if (onProgress && (v & (PROGRESS_STEP - 1)) === 0) onProgress(v / nv, "Building edges");
  }
  return { edgeCount: ne, edgeVertices, edgeTriangles, edgeFaceCount, triangleEdges, triangleNeighbors };
}

/** Per-vertex flag: 1 if the vertex lies on a boundary edge. */
export function boundaryVertexMask(mesh: MeshData, neighbors?: Int32Array): Uint8Array {
  const nbr = neighbors ?? buildTriangleNeighbors(mesh);
  const I = mesh.indices;
  const nv = vertexCountOf(mesh);
  const mask = new Uint8Array(nv);
  const nh = nbr.length;
  for (let he = 0; he < nh; he++) {
    if (nbr[he] !== BOUNDARY) continue;
    const k = he % 3;
    mask[I[he]] = 1;
    mask[I[k === 2 ? he - 2 : he + 1]] = 1;
  }
  return mask;
}

// ---------------------------------------------------------------------------
// Boundary loops
// ---------------------------------------------------------------------------

/**
 * Ordered boundary loops (closed chains of boundary edges), each as vertex indices.
 *
 * Each loop is ordered along the direction of its adjacent triangles' edges (for a
 * consistently oriented mesh, the hole lies to the right of the walking direction when viewed
 * from the outward side; a patch filling the hole must use the reversed order).
 * Loops that pass several times through one vertex ("bow-tie" vertices) are split into simple
 * loops. Open boundary chains (only possible on broken topology) and loops with < 3 vertices
 * are dropped. Accepts precomputed triangle neighbours.
 */
export function findBoundaryLoops(mesh: MeshData, neighbors?: Int32Array): Uint32Array[] {
  const nbr = neighbors ?? buildTriangleNeighbors(mesh);
  const I = mesh.indices;
  const nv = vertexCountOf(mesh);
  // collect boundary half-edges
  const B = new GrowableU32(1024);
  for (let he = 0; he < nbr.length; he++) if (nbr[he] === BOUNDARY) B.push(he);
  const nb = B.length;
  const loops: Uint32Array[] = [];
  if (nb === 0) return loops;
  const be = B.data;
  const fromV = new Uint32Array(nb), toV = new Uint32Array(nb);
  for (let i = 0; i < nb; i++) {
    const he = be[i];
    const k = he % 3;
    fromV[i] = I[he];
    toV[i] = I[k === 2 ? he - 2 : he + 1];
  }
  // vertex -> incident boundary edges (CSR)
  const vOff = new Uint32Array(nv + 1);
  for (let i = 0; i < nb; i++) {
    vOff[fromV[i] + 1]++;
    vOff[toV[i] + 1]++;
  }
  for (let v = 0; v < nv; v++) vOff[v + 1] += vOff[v];
  const vList = new Uint32Array(nb * 2);
  for (let i = 0; i < nb; i++) {
    vList[vOff[fromV[i]]++] = i;
    vList[vOff[toV[i]]++] = i;
  }
  for (let v = nv; v > 0; v--) vOff[v] = vOff[v - 1];
  vOff[0] = 0;

  const used = new Uint8Array(nb);
  const pos = new Int32Array(nv).fill(-1);
  const stackV = new GrowableU32(256);
  const stackD = new GrowableI32(256);

  const emit = (p: number) => {
    const top = stackV.length - 1;
    const n = top - p + 1;
    if (n < 3) return;
    let dirSum = 0;
    for (let i = p; i <= top; i++) dirSum += stackD.data[i];
    const loop = new Uint32Array(n);
    if (dirSum >= 0) for (let i = 0; i < n; i++) loop[i] = stackV.data[p + i];
    else {
      // reverse while keeping the first vertex first
      loop[0] = stackV.data[p];
      for (let i = 1; i < n; i++) loop[i] = stackV.data[top + 1 - i];
    }
    loops.push(loop);
  };

  for (let e0 = 0; e0 < nb; e0++) {
    if (used[e0]) continue;
    used[e0] = 1;
    stackV.clear();
    stackD.clear();
    stackV.push(fromV[e0]);
    pos[fromV[e0]] = 0;
    stackD.push(1);
    let cur = toV[e0];
    for (;;) {
      const p = pos[cur];
      if (p >= 0) {
        emit(p);
        for (let i = p + 1; i < stackV.length; i++) pos[stackV.data[i]] = -1;
        stackV.length = p + 1;
        stackD.length = p;
      } else {
        pos[cur] = stackV.length;
        stackV.push(cur);
      }
      // next unused boundary edge at cur, preferring one leaving cur (consistent orientation)
      let pick = -1, pickDir = 0;
      for (let j = vOff[cur], je = vOff[cur + 1]; j < je; j++) {
        const e = vList[j];
        if (used[e]) continue;
        if (fromV[e] === cur) {
          pick = e;
          pickDir = 1;
          break;
        }
        if (pick < 0) {
          pick = e;
          pickDir = -1;
        }
      }
      if (pick < 0) break; // closed and exhausted, or an open chain (dropped)
      used[pick] = 1;
      stackD.push(pickDir);
      cur = pickDir > 0 ? toV[pick] : fromV[pick];
    }
    for (let i = 0; i < stackV.length; i++) pos[stackV.data[i]] = -1;
  }
  return loops;
}

// ---------------------------------------------------------------------------
// Connected components
// ---------------------------------------------------------------------------

export interface ComponentInfo {
  /** component id per triangle (0..count-1), components numbered by first triangle */
  triangleComponent: Uint32Array;
  /** triangles per component */
  componentSizes: Uint32Array;
  count: number;
}

/** Union–find root with path halving (iterative). */
export function ufFind(parent: Int32Array, x: number): number {
  while (parent[x] !== x) {
    const g = parent[parent[x]];
    parent[x] = g;
    x = g;
  }
  return x;
}

/**
 * Connected components of triangles (triangles sharing a vertex are connected).
 * Union–find over vertices, O(n α(n)), 4 B/vertex + 4 B/triangle.
 */
export function connectedComponents(mesh: MeshData, onProgress?: ProgressFn): ComponentInfo {
  const I = mesh.indices;
  const nv = vertexCountOf(mesh);
  const nt = triangleCountOf(mesh);
  const parent = new Int32Array(nv);
  for (let v = 0; v < nv; v++) parent[v] = v;
  for (let t = 0; t < nt; t++) {
    const a = ufFind(parent, I[3 * t]);
    const b = ufFind(parent, I[3 * t + 1]);
    const c = ufFind(parent, I[3 * t + 2]);
    // union by smaller index keeps it deterministic
    const r = a < b ? (a < c ? a : c) : b < c ? b : c;
    parent[a] = r;
    parent[b] = r;
    parent[c] = r;
    if (onProgress && (t & (PROGRESS_STEP - 1)) === 0) onProgress((0.8 * t) / nt, "Connected components");
  }
  // label roots in order of first triangle
  const label = new Int32Array(nv).fill(-1);
  const triangleComponent = new Uint32Array(nt);
  const sizes = new GrowableU32(64);
  for (let t = 0; t < nt; t++) {
    const r = ufFind(parent, I[3 * t]);
    let l = label[r];
    if (l < 0) {
      l = label[r] = sizes.length;
      sizes.push(0);
    }
    triangleComponent[t] = l;
    sizes.data[l]++;
  }
  return { triangleComponent, componentSizes: sizes.toArray(), count: sizes.length };
}

// ---------------------------------------------------------------------------
// Bounds, area, edge length
// ---------------------------------------------------------------------------

/** Axis-aligned bounds of all vertices. Empty mesh → min = max = [0,0,0]. */
export function meshBounds(mesh: MeshData): BBox3 {
  const P = mesh.positions;
  const nv = vertexCountOf(mesh);
  if (nv === 0) return { min: [0, 0, 0], max: [0, 0, 0] };
  let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
  for (let i = 0; i < nv * 3; i += 3) {
    const x = P[i], y = P[i + 1], z = P[i + 2];
    if (x < x0) x0 = x;
    if (x > x1) x1 = x;
    if (y < y0) y0 = y;
    if (y > y1) y1 = y;
    if (z < z0) z0 = z;
    if (z > z1) z1 = z;
  }
  return { min: [x0, y0, z0], max: [x1, y1, z1] };
}

/** Total surface area. */
export function meshArea(mesh: MeshData): number {
  const P = mesh.positions, I = mesh.indices;
  const nt = triangleCountOf(mesh);
  let sum = 0;
  for (let t = 0; t < nt; t++) sum += triangleArea(P, I[3 * t], I[3 * t + 1], I[3 * t + 2]);
  return sum;
}

export function triangleArea(P: ArrayLike<number>, ia: number, ib: number, ic: number): number {
  const a = 3 * ia, b = 3 * ib, c = 3 * ic;
  const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2];
  const vx = P[c] - P[a], vy = P[c + 1] - P[a + 1], vz = P[c + 2] - P[a + 2];
  const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
  return 0.5 * Math.sqrt(nx * nx + ny * ny + nz * nz);
}

/**
 * Mean triangle edge length. For large meshes a uniform stride sample of about `maxSamples`
 * edges is used. Returns 0 for an empty mesh.
 */
export function averageEdgeLength(mesh: MeshData, maxSamples = 1_000_000): number {
  const P = mesh.positions, I = mesh.indices;
  const nt = triangleCountOf(mesh);
  if (nt === 0) return 0;
  const stride = Math.max(1, Math.floor((nt * 3) / maxSamples));
  let sum = 0, n = 0;
  for (let t = 0; t < nt; t += stride) {
    for (let k = 0; k < 3; k++) {
      const a = 3 * I[3 * t + k], b = 3 * I[3 * t + (k === 2 ? 0 : k + 1)];
      const dx = P[a] - P[b], dy = P[a + 1] - P[b + 1], dz = P[a + 2] - P[b + 2];
      sum += Math.sqrt(dx * dx + dy * dy + dz * dz);
      n++;
    }
  }
  return n ? sum / n : 0;
}

// ---------------------------------------------------------------------------
// Compaction
// ---------------------------------------------------------------------------

export interface CompactResult {
  mesh: MeshData;
  /** old vertex index → new vertex index, -1 when removed */
  vertexMap: Int32Array;
  /** new triangle index → old triangle index */
  triangleMap: Uint32Array;
}

/**
 * New mesh containing the triangles with keepTriangleMask[t] !== 0 (all when omitted), with
 * unreferenced vertices removed (normals/colours carried over). Triangles referencing
 * out-of-range vertices are dropped. Vertex order is preserved.
 */
export function compactMesh(mesh: MeshData, keepTriangleMask?: Uint8Array): MeshData {
  return compactMeshWithMap(mesh, keepTriangleMask).mesh;
}

export function compactMeshWithMap(mesh: MeshData, keepTriangleMask?: Uint8Array): CompactResult {
  const I = mesh.indices;
  const nv = vertexCountOf(mesh);
  const nt = triangleCountOf(mesh);
  const vertexMap = new Int32Array(nv).fill(-1);
  let keptTris = 0;
  for (let t = 0; t < nt; t++) {
    if (keepTriangleMask && !keepTriangleMask[t]) continue;
    const a = I[3 * t], b = I[3 * t + 1], c = I[3 * t + 2];
    if (a >= nv || b >= nv || c >= nv) continue;
    vertexMap[a] = vertexMap[b] = vertexMap[c] = 0;
    keptTris++;
  }
  let nvNew = 0;
  for (let v = 0; v < nv; v++) if (vertexMap[v] === 0) vertexMap[v] = nvNew++;
  const indices = new Uint32Array(keptTris * 3);
  const triangleMap = new Uint32Array(keptTris);
  let o = 0;
  for (let t = 0; t < nt; t++) {
    if (keepTriangleMask && !keepTriangleMask[t]) continue;
    const a = I[3 * t], b = I[3 * t + 1], c = I[3 * t + 2];
    if (a >= nv || b >= nv || c >= nv) continue;
    triangleMap[o] = t;
    indices[3 * o] = vertexMap[a];
    indices[3 * o + 1] = vertexMap[b];
    indices[3 * o + 2] = vertexMap[c];
    o++;
  }
  const positions = new Float32Array(nvNew * 3);
  const normals = mesh.normals ? new Float32Array(nvNew * 3) : undefined;
  const colors = mesh.colors ? new Uint8Array(nvNew * 3) : undefined;
  const P = mesh.positions, N = mesh.normals, C = mesh.colors;
  for (let v = 0; v < nv; v++) {
    const m = vertexMap[v];
    if (m < 0) continue;
    positions[3 * m] = P[3 * v];
    positions[3 * m + 1] = P[3 * v + 1];
    positions[3 * m + 2] = P[3 * v + 2];
    if (normals && N) {
      normals[3 * m] = N[3 * v];
      normals[3 * m + 1] = N[3 * v + 1];
      normals[3 * m + 2] = N[3 * v + 2];
    }
    if (colors && C) {
      colors[3 * m] = C[3 * v];
      colors[3 * m + 1] = C[3 * v + 1];
      colors[3 * m + 2] = C[3 * v + 2];
    }
  }
  const out: MeshData = { positions, indices };
  if (normals) out.normals = normals;
  if (colors) out.colors = colors;
  return { mesh: out, vertexMap, triangleMap };
}

/** Shallow structural copy with fresh typed arrays (deep copy of the data). */
export function cloneMesh(mesh: MeshData): MeshData {
  const out: MeshData = { positions: mesh.positions.slice(), indices: mesh.indices.slice() };
  if (mesh.normals) out.normals = mesh.normals.slice();
  if (mesh.colors) out.colors = mesh.colors.slice();
  return out;
}
