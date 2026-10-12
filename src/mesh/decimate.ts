/**
 * Mesh decimation.
 *
 *  - decimateQEM: Garland–Heckbert quadric edge collapse. Typed-array binary heap over the unique
 *    edges with lazy re-evaluation (quadric costs only grow when quadrics are summed, so a stale
 *    key is a lower bound: popped stale edges are re-scored and re-pushed). Dynamic vertex →
 *    corner adjacency as singly linked lists over triangle corners (splice on collapse). Checks:
 *    link condition (keeps the mesh manifold), normal flips / sliver creation, boundary
 *    preservation (boundary edges collapse only along the boundary, heavy boundary quadrics).
 *    Quadrics are float32 in coordinates normalised to the unit box (precision), positions
 *    float64. Working set ≈ 130 B per input triangle.
 *  - decimateClustering: Lindstrom-style out-of-core vertex clustering with per-cell quadrics.
 *    One linear pass, memory ∝ output size (+ 4 B per input vertex).
 *  - decimate: clustering pre-pass for very large inputs, then QEM.
 */
import type { MeshData, ProgressFn } from "../core/types";
import { removeDegenerateTriangles } from "./repair";
import { BOUNDARY, buildEdges, computeVertexNormals, meshArea, meshBounds, triangleCountOf, vertexCountOf } from "./topology";
import { GrowableF64, GrowableU32 } from "./util/growable";
import { MinHeap } from "./util/heap";
import { PairHashMap } from "./util/pair-hash";

export interface DecimateQEMOptions {
  targetTriangles: number;
  /** Keep the open boundary in place (default true). */
  preserveBoundary?: boolean;
  /** Reject collapses that rotate a triangle normal by more than this (degrees). Default 75. */
  maxNormalChangeDeg?: number;
  onProgress?: ProgressFn;
}

/** Triangle shape quality 4√3·area / Σ|e|² (1 = equilateral, 0 = degenerate). */
function quality(ax: number, ay: number, az: number, bx: number, by: number, bz: number, cx: number, cy: number, cz: number): number {
  const ux = bx - ax, uy = by - ay, uz = bz - az;
  const vx = cx - ax, vy = cy - ay, vz = cz - az;
  const wx = cx - bx, wy = cy - by, wz = cz - bz;
  const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
  const s = ux * ux + uy * uy + uz * uz + vx * vx + vy * vy + vz * vz + wx * wx + wy * wy + wz * wz;
  return s > 0 ? (2 * Math.sqrt(3) * Math.sqrt(nx * nx + ny * ny + nz * nz)) / s : 0;
}

export function decimateQEM(mesh: MeshData, opts: DecimateQEMOptions): MeshData {
  const nv = vertexCountOf(mesh);
  const nt = triangleCountOf(mesh);
  const target = Math.max(0, Math.floor(opts.targetTriangles));
  const preserveBoundary = opts.preserveBoundary ?? true;
  const cosMax = Math.cos(((opts.maxNormalChangeDeg ?? 75) * Math.PI) / 180);
  const onProgress = opts.onProgress;
  if (nt <= target || nt === 0) return { ...copyMesh(mesh) };

  // --- normalised float64 positions -------------------------------------------------------
  const bb = meshBounds(mesh);
  const cx0 = (bb.min[0] + bb.max[0]) / 2, cy0 = (bb.min[1] + bb.max[1]) / 2, cz0 = (bb.min[2] + bb.max[2]) / 2;
  const ext = Math.max(bb.max[0] - bb.min[0], bb.max[1] - bb.min[1], bb.max[2] - bb.min[2]) || 1;
  const sc = 2 / ext;
  const P = mesh.positions;
  const pos = new Float64Array(nv * 3);
  for (let v = 0; v < nv; v++) {
    pos[3 * v] = (P[3 * v] - cx0) * sc;
    pos[3 * v + 1] = (P[3 * v + 1] - cy0) * sc;
    pos[3 * v + 2] = (P[3 * v + 2] - cz0) * sc;
  }
  const col = mesh.colors ? Float32Array.from(mesh.colors) : null;

  // --- topology -----------------------------------------------------------------------------
  const idx = mesh.indices.slice();
  const topo = buildEdges(mesh, onProgress ? (f) => onProgress(0.1 * f, "Decimating: topology") : undefined);
  const ne = topo.edgeCount;
  const edgeV = topo.edgeVertices; // owned copy from buildEdges, updated with resolved ends
  const triEdge = topo.triangleEdges;
  const alive = new Uint8Array(nt);
  let live = 0;
  for (let t = 0; t < nt; t++) {
    const a = idx[3 * t], b = idx[3 * t + 1], c = idx[3 * t + 2];
    if (a !== b && b !== c && c !== a && a < nv && b < nv && c < nv) {
      alive[t] = 1;
      live++;
    }
  }
  const isB = new Uint8Array(nv);
  const nbr = topo.triangleNeighbors;

  // --- quadrics (a2 ab ac ad b2 bc bd c2 cd d2) -----------------------------------------------
  const Qd = new Float32Array(nv * 10);
  const addPlane = (v: number, nx: number, ny: number, nz: number, d: number, w: number) => {
    const o = 10 * v;
    Qd[o] += w * nx * nx;
    Qd[o + 1] += w * nx * ny;
    Qd[o + 2] += w * nx * nz;
    Qd[o + 3] += w * nx * d;
    Qd[o + 4] += w * ny * ny;
    Qd[o + 5] += w * ny * nz;
    Qd[o + 6] += w * ny * d;
    Qd[o + 7] += w * nz * nz;
    Qd[o + 8] += w * nz * d;
    Qd[o + 9] += w * d * d;
  };
  const boundaryWeight = preserveBoundary ? 1000 : 1;
  for (let t = 0; t < nt; t++) {
    if (!alive[t]) continue;
    const a = idx[3 * t], b = idx[3 * t + 1], c = idx[3 * t + 2];
    const ax = pos[3 * a], ay = pos[3 * a + 1], az = pos[3 * a + 2];
    const ux = pos[3 * b] - ax, uy = pos[3 * b + 1] - ay, uz = pos[3 * b + 2] - az;
    const vx = pos[3 * c] - ax, vy = pos[3 * c + 1] - ay, vz = pos[3 * c + 2] - az;
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const l = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (!(l > 0)) continue;
    nx /= l;
    ny /= l;
    nz /= l;
    const d = -(nx * ax + ny * ay + nz * az);
    const area = 0.5 * l;
    addPlane(a, nx, ny, nz, d, area);
    addPlane(b, nx, ny, nz, d, area);
    addPlane(c, nx, ny, nz, d, area);
    // boundary constraint planes (perpendicular to the triangle through the boundary edge)
    for (let k = 0; k < 3; k++) {
      if (nbr[3 * t + k] !== BOUNDARY) continue;
      const p = idx[3 * t + k], q = idx[3 * t + (k === 2 ? 0 : k + 1)];
      isB[p] = isB[q] = 1;
      const ex = pos[3 * q] - pos[3 * p], ey = pos[3 * q + 1] - pos[3 * p + 1], ez = pos[3 * q + 2] - pos[3 * p + 2];
      let mx = ey * nz - ez * ny, my = ez * nx - ex * nz, mz = ex * ny - ey * nx;
      const ml = Math.sqrt(mx * mx + my * my + mz * mz);
      if (!(ml > 0)) continue;
      mx /= ml;
      my /= ml;
      mz /= ml;
      const md = -(mx * pos[3 * p] + my * pos[3 * p + 1] + mz * pos[3 * p + 2]);
      const w = boundaryWeight * (ex * ex + ey * ey + ez * ez);
      addPlane(p, mx, my, mz, md, w);
      addPlane(q, mx, my, mz, md, w);
    }
  }
  // non-manifold edges: treat their vertices like boundary vertices (don't move them off)
  for (let e = 0; e < ne; e++) if (topo.edgeFaceCount[e] > 2) isB[edgeV[2 * e]] = isB[edgeV[2 * e + 1]] = 1;

  // --- vertex -> corner linked lists ---------------------------------------------------------
  const head = new Int32Array(nv).fill(-1);
  const nextC = new Int32Array(nt * 3).fill(-1);
  for (let c = nt * 3 - 1; c >= 0; c--) {
    if (!alive[(c / 3) | 0]) continue;
    const v = idx[c];
    nextC[c] = head[v];
    head[v] = c;
  }
  const rep = new Int32Array(nv);
  for (let v = 0; v < nv; v++) rep[v] = v;
  const find = (x: number): number => {
    while (rep[x] !== x) {
      const g = rep[rep[x]];
      rep[x] = g;
      x = g;
    }
    return x;
  };
  const lastMod = new Uint32Array(nv);
  const edgeStamp = new Uint32Array(ne);
  const inHeap = new Uint8Array(ne);
  let clock = 1;

  // --- edge cost -----------------------------------------------------------------------------
  const opt = new Float64Array(3); // optimal position output
  const q = new Float64Array(10);
  const evalQ = (x: number, y: number, z: number): number =>
    q[0] * x * x + 2 * q[1] * x * y + 2 * q[2] * x * z + 2 * q[3] * x + q[4] * y * y + 2 * q[5] * y * z + 2 * q[6] * y + q[7] * z * z + 2 * q[8] * z + q[9];
  /** cost of collapsing (a,b); writes the target position to `opt` */
  const edgeCost = (a: number, b: number): number => {
    const oa = 10 * a, ob = 10 * b;
    for (let i = 0; i < 10; i++) q[i] = Qd[oa + i] + Qd[ob + i];
    const ba = isB[a], bb2 = isB[b];
    if (preserveBoundary && ba !== bb2) {
      // keep the boundary vertex in place
      const f = ba ? a : b;
      opt[0] = pos[3 * f];
      opt[1] = pos[3 * f + 1];
      opt[2] = pos[3 * f + 2];
      return Math.max(0, evalQ(opt[0], opt[1], opt[2]));
    }
    // solve [a2 ab ac; ab b2 bc; ac bc c2] x = -[ad bd cd]
    const a11 = q[0], a12 = q[1], a13 = q[2], a22 = q[4], a23 = q[5], a33 = q[7];
    const r1 = -q[3], r2 = -q[6], r3 = -q[8];
    const c11 = a22 * a33 - a23 * a23, c12 = a13 * a23 - a12 * a33, c13 = a12 * a23 - a13 * a22;
    const det = a11 * c11 + a12 * c12 + a13 * c13;
    const scaleN = Math.abs(a11) + Math.abs(a22) + Math.abs(a33);
    const ax = pos[3 * a], ay = pos[3 * a + 1], az = pos[3 * a + 2];
    const bx = pos[3 * b], by = pos[3 * b + 1], bz = pos[3 * b + 2];
    if (Math.abs(det) > 1e-9 * scaleN * scaleN * scaleN && scaleN > 0) {
      const c22 = a11 * a33 - a13 * a13, c23 = a12 * a13 - a11 * a23, c33 = a11 * a22 - a12 * a12;
      const x = (c11 * r1 + c12 * r2 + c13 * r3) / det;
      const y = (c12 * r1 + c22 * r2 + c23 * r3) / det;
      const z = (c13 * r1 + c23 * r2 + c33 * r3) / det;
      // guard against far-away optima of nearly singular systems
      const mx = (ax + bx) / 2, my = (ay + by) / 2, mz = (az + bz) / 2;
      const el2 = (ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2;
      if ((x - mx) ** 2 + (y - my) ** 2 + (z - mz) ** 2 <= 4 * el2) {
        opt[0] = x;
        opt[1] = y;
        opt[2] = z;
        return Math.max(0, evalQ(x, y, z));
      }
    }
    // fallback: best of a, b, midpoint
    let best = evalQ(ax, ay, az);
    opt[0] = ax;
    opt[1] = ay;
    opt[2] = az;
    const eb = evalQ(bx, by, bz);
    if (eb < best) {
      best = eb;
      opt[0] = bx;
      opt[1] = by;
      opt[2] = bz;
    }
    const mx = (ax + bx) / 2, my = (ay + by) / 2, mz = (az + bz) / 2;
    const em = evalQ(mx, my, mz);
    if (em < best) {
      best = em;
      opt[0] = mx;
      opt[1] = my;
      opt[2] = mz;
    }
    return Math.max(0, best);
  };

  // --- initial heap ------------------------------------------------------------------------------
  const keys = new Float64Array(ne);
  const ids = new Uint32Array(ne);
  let nh = 0;
  for (let e = 0; e < ne; e++) {
    const a = edgeV[2 * e], b = edgeV[2 * e + 1];
    keys[nh] = edgeCost(a, b);
    ids[nh++] = e;
    inHeap[e] = 1;
  }
  const heap = MinHeap.from(keys, ids, nh);
  onProgress?.(0.15, "Decimating");

  // --- collapse loop -----------------------------------------------------------------------------
  const mark = new Uint32Array(nv);
  let markStamp = 1;
  const startLive = live;
  const toRemove = Math.max(1, startLive - target);
  let collapses = 0;
  const newPos = new Float64Array(3);

  /** walk live corners of v, pruning dead ones from the list */
  const forEachCorner = (v: number, fn: (c: number) => void) => {
    let prev = -1;
    let c = head[v];
    while (c >= 0) {
      const nx = nextC[c];
      if (!alive[(c / 3) | 0]) {
        if (prev < 0) head[v] = nx;
        else nextC[prev] = nx;
      } else {
        fn(c);
        prev = c;
      }
      c = nx;
    }
  };

  let shared = 0, common = 0, ok = true;
  // scratch state for checks (closures avoid allocations)
  let chkA = 0, chkB = 0;
  const markNeighborsOfA = (c: number) => {
    const t = (c / 3) | 0;
    const o = 3 * t;
    for (let k = 0; k < 3; k++) {
      const w = idx[o + k];
      if (w !== chkA) mark[w] = markStamp;
    }
  };
  const countFromB = (c: number) => {
    const t = (c / 3) | 0;
    const o = 3 * t;
    const i0 = idx[o], i1 = idx[o + 1], i2 = idx[o + 2];
    if (i0 === chkA || i1 === chkA || i2 === chkA) {
      shared++;
    }
    for (let k = 0; k < 3; k++) {
      const w = idx[o + k];
      if (w === chkB || w === chkA) continue;
      if (mark[w] === markStamp) {
        common++;
        mark[w] = markStamp + 1;
      }
    }
  };
  /** normal-flip & sliver check of triangles around `v` (not containing `other`) when v moves to newPos */
  const checkMove = (v: number, other: number) => (c: number) => {
    if (!ok) return;
    const t = (c / 3) | 0;
    const o = 3 * t;
    const i0 = idx[o], i1 = idx[o + 1], i2 = idx[o + 2];
    if (i0 === other || i1 === other || i2 === other) return;
    const ax = pos[3 * i0], ay = pos[3 * i0 + 1], az = pos[3 * i0 + 2];
    const bx = pos[3 * i1], by = pos[3 * i1 + 1], bz = pos[3 * i1 + 2];
    const cx = pos[3 * i2], cy = pos[3 * i2 + 1], cz = pos[3 * i2 + 2];
    const n0x = (by - ay) * (cz - az) - (bz - az) * (cy - ay);
    const n0y = (bz - az) * (cx - ax) - (bx - ax) * (cz - az);
    const n0z = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    let px = ax, py = ay, pz = az, qx = bx, qy = by, qz = bz, rx = cx, ry = cy, rz = cz;
    if (i0 === v) {
      px = newPos[0];
      py = newPos[1];
      pz = newPos[2];
    } else if (i1 === v) {
      qx = newPos[0];
      qy = newPos[1];
      qz = newPos[2];
    } else {
      rx = newPos[0];
      ry = newPos[1];
      rz = newPos[2];
    }
    const n1x = (qy - py) * (rz - pz) - (qz - pz) * (ry - py);
    const n1y = (qz - pz) * (rx - px) - (qx - px) * (rz - pz);
    const n1z = (qx - px) * (ry - py) - (qy - py) * (rx - px);
    const l0 = Math.sqrt(n0x * n0x + n0y * n0y + n0z * n0z), l1 = Math.sqrt(n1x * n1x + n1y * n1y + n1z * n1z);
    if (!(l1 > 0)) {
      ok = false;
      return;
    }
    if (l0 > 0 && n0x * n1x + n0y * n1y + n0z * n1z < cosMax * l0 * l1) {
      ok = false;
      return;
    }
    const qOld = quality(ax, ay, az, bx, by, bz, cx, cy, cz);
    const qNew = quality(px, py, pz, qx, qy, qz, rx, ry, rz);
    if (qNew < 0.02 && qNew < 0.5 * qOld) ok = false;
  };

  let survivor = 0;
  const reroute = (c: number) => {
    const t = (c / 3) | 0;
    const o = 3 * t;
    if (idx[o] === survivor || idx[o + 1] === survivor || idx[o + 2] === survivor) {
      alive[t] = 0;
      live--;
    } else {
      idx[c] = survivor;
    }
  };
  const schedule = (e: number) => {
    if (e === 0xffffffff || inHeap[e]) return;
    const a = find(edgeV[2 * e]), b = find(edgeV[2 * e + 1]);
    if (a === b) return;
    edgeV[2 * e] = a;
    edgeV[2 * e + 1] = b;
    heap.push(edgeCost(a, b), e);
    edgeStamp[e] = clock;
    inHeap[e] = 1;
  };
  const repush = (c: number) => {
    // edges of this corner's triangle incident to the survivor that are not in the heap
    const t = (c / 3) | 0;
    const k = c - 3 * t;
    schedule(triEdge[c]); // edge (c -> c+1)
    schedule(triEdge[3 * t + (k === 0 ? 2 : k - 1)]); // edge (c-1 -> c)
  };

  while (live > target && heap.size > 0) {
    const e = heap.pop();
    let a = find(edgeV[2 * e]), b = find(edgeV[2 * e + 1]);
    if (a === b) {
      inHeap[e] = 0;
      continue;
    }
    edgeV[2 * e] = a;
    edgeV[2 * e + 1] = b;
    if (edgeStamp[e] < lastMod[a] || edgeStamp[e] < lastMod[b]) {
      // stale (cost can only have grown): re-score and re-insert
      const cost = edgeCost(a, b);
      edgeStamp[e] = clock;
      heap.push(cost, e);
      continue;
    }
    inHeap[e] = 0;
    // survivor: keep boundary vertex if only one is on the boundary
    if (isB[b] && !isB[a]) {
      const t = a;
      a = b;
      b = t;
    }
    // link condition
    markStamp += 2;
    chkA = a;
    chkB = b;
    forEachCorner(a, markNeighborsOfA);
    shared = 0;
    common = 0;
    forEachCorner(b, countFromB);
    if (shared === 0 || shared > 2 || common !== shared) continue;
    // an interior edge joining two boundary vertices would pinch the surface into a bow-tie
    if (isB[a] && isB[b] && shared !== 1) continue;
    // target position & geometric checks
    edgeCost(a, b);
    newPos[0] = opt[0];
    newPos[1] = opt[1];
    newPos[2] = opt[2];
    ok = true;
    forEachCorner(a, checkMove(a, b));
    if (ok) forEachCorner(b, checkMove(b, a));
    if (!ok) continue;

    // --- collapse b into a
    clock++;
    const oa = 10 * a, ob = 10 * b;
    for (let i = 0; i < 10; i++) Qd[oa + i] += Qd[ob + i];
    if (col) {
      // inverse-distance weighted colour
      const da = Math.hypot(pos[3 * a] - newPos[0], pos[3 * a + 1] - newPos[1], pos[3 * a + 2] - newPos[2]);
      const db = Math.hypot(pos[3 * b] - newPos[0], pos[3 * b + 1] - newPos[1], pos[3 * b + 2] - newPos[2]);
      const wa = da + db > 0 ? db / (da + db) : 0.5;
      for (let k = 0; k < 3; k++) col[3 * a + k] = wa * col[3 * a + k] + (1 - wa) * col[3 * b + k];
    }
    pos[3 * a] = newPos[0];
    pos[3 * a + 1] = newPos[1];
    pos[3 * a + 2] = newPos[2];
    isB[a] = isB[a] | isB[b];
    rep[b] = a;
    lastMod[a] = clock;
    survivor = a;
    forEachCorner(b, reroute);
    // splice b's (pruned) list in front of a's
    let tail = -1;
    for (let c = head[b]; c >= 0; c = nextC[c]) tail = c;
    if (tail >= 0) {
      nextC[tail] = head[a];
      head[a] = head[b];
    }
    head[b] = -1;
    // make sure every edge around the survivor is (re)scheduled
    forEachCorner(a, repush);
    collapses++;
    if (onProgress && (collapses & 0xffff) === 0) onProgress(0.15 + (0.8 * (startLive - live)) / toRemove, "Decimating");
  }

  // --- output --------------------------------------------------------------------------------------
  const vmap = new Int32Array(nv).fill(-1);
  let nOutV = 0, nOutT = 0;
  for (let t = 0; t < nt; t++) {
    if (!alive[t]) continue;
    nOutT++;
    for (let k = 0; k < 3; k++) {
      const v = idx[3 * t + k];
      if (vmap[v] < 0) vmap[v] = nOutV++;
    }
  }
  const outPos = new Float32Array(nOutV * 3);
  const outCol = col ? new Uint8Array(nOutV * 3) : undefined;
  const inv = 1 / sc;
  for (let v = 0; v < nv; v++) {
    const m = vmap[v];
    if (m < 0) continue;
    outPos[3 * m] = pos[3 * v] * inv + cx0;
    outPos[3 * m + 1] = pos[3 * v + 1] * inv + cy0;
    outPos[3 * m + 2] = pos[3 * v + 2] * inv + cz0;
    if (outCol && col) for (let k = 0; k < 3; k++) outCol[3 * m + k] = Math.max(0, Math.min(255, Math.round(col[3 * v + k])));
  }
  const outIdx = new Uint32Array(nOutT * 3);
  let o = 0;
  for (let t = 0; t < nt; t++) {
    if (!alive[t]) continue;
    outIdx[o++] = vmap[idx[3 * t]];
    outIdx[o++] = vmap[idx[3 * t + 1]];
    outIdx[o++] = vmap[idx[3 * t + 2]];
  }
  const out: MeshData = { positions: outPos, indices: outIdx };
  if (outCol) out.colors = outCol;
  if (mesh.normals) out.normals = computeVertexNormals(out);
  onProgress?.(1, "Decimating");
  return out;
}

function copyMesh(mesh: MeshData): MeshData {
  const out: MeshData = { positions: mesh.positions.slice(), indices: mesh.indices.slice() };
  if (mesh.normals) out.normals = mesh.normals.slice();
  if (mesh.colors) out.colors = mesh.colors.slice();
  return out;
}

// ---------------------------------------------------------------------------
// Vertex clustering (Lindstrom 2000, "Out-of-core simplification of large polygonal models")
// ---------------------------------------------------------------------------

export interface ClusteringOptions {
  /** Grid cell size (mesh units). Either this or targetTriangles must be given. */
  cellSize?: number;
  /** Approximate output triangle count (used to derive cellSize). */
  targetTriangles?: number;
  onProgress?: ProgressFn;
}

/**
 * Uniform-grid vertex clustering with per-cell quadrics: every vertex is snapped to its cell's
 * representative (the point minimising the cell's summed triangle-plane quadric, clamped to the
 * cell; mean position when the quadric is singular). Triangles whose vertices fall in three
 * distinct cells survive; duplicates are removed. One pass over the triangles; memory is
 * O(output) + 4 B per input vertex. Output may be non-manifold where thin parts collapse.
 * The triangle count achieved for a given targetTriangles is approximate (typically ±30 %).
 */
export function decimateClustering(mesh: MeshData, opts: ClusteringOptions): MeshData {
  const P = mesh.positions, I = mesh.indices, C = mesh.colors;
  const nv = vertexCountOf(mesh);
  const nt = triangleCountOf(mesh);
  const bb = meshBounds(mesh);
  let cell = opts.cellSize ?? 0;
  if (!(cell > 0)) {
    const target = Math.max(4, opts.targetTriangles ?? nt / 4);
    // occupied cells ≈ 1.5·A/cell² (surfaces cut cells obliquely), output ≈ 2 triangles / cell
    cell = Math.sqrt((3 * meshArea(mesh)) / target);
    if (!(cell > 0)) cell = 1;
  }
  const inv = 1 / cell;
  const ox = bb.min[0], oy = bb.min[1], oz = bb.min[2];
  const gx = Math.floor((bb.max[0] - ox) * inv) + 1, gy = Math.floor((bb.max[1] - oy) * inv) + 1;
  // cell key = ix + gx*(iy + gy*iz) < 2^53 as long as the grid has < 2^53 cells
  const cellMap = new PairHashMap(1024);
  const vCell = new Int32Array(nv);
  let nCells = 0;
  const cellQ = new GrowableF64(10 * 1024);
  const cellSum = new GrowableF64(4 * 1024); // x y z count
  const cellCol = C ? new GrowableF64(3 * 1024) : null;
  for (let v = 0; v < nv; v++) {
    const x = P[3 * v], y = P[3 * v + 1], z = P[3 * v + 2];
    const ix = Math.min(gx - 1, Math.max(0, Math.floor((x - ox) * inv)));
    const iy = Math.min(gy - 1, Math.max(0, Math.floor((y - oy) * inv)));
    const iz = Math.max(0, Math.floor((z - oz) * inv));
    const key = ix + gx * (iy + gy * iz);
    const lo = key % 4294967296, hi = Math.floor(key / 4294967296);
    let id = cellMap.get(lo, hi);
    if (id < 0) {
      id = nCells++;
      cellMap.set(lo, hi, id);
      for (let k = 0; k < 10; k++) cellQ.push(0);
      cellSum.push2(0, 0);
      cellSum.push2(0, 0);
      if (cellCol) cellCol.push3(0, 0, 0);
    }
    vCell[v] = id;
    const s = cellSum.data;
    s[4 * id] += x;
    s[4 * id + 1] += y;
    s[4 * id + 2] += z;
    s[4 * id + 3] += 1;
    if (cellCol && C) {
      const cc = cellCol.data;
      cc[3 * id] += C[3 * v];
      cc[3 * id + 1] += C[3 * v + 1];
      cc[3 * id + 2] += C[3 * v + 2];
    }
    if (opts.onProgress && (v & 0xfffff) === 0) opts.onProgress((0.3 * v) / nv, "Clustering vertices");
  }
  const outTri = new GrowableU32(Math.min(nt * 3, 3 * (opts.targetTriangles ?? nt) * 2 + 1024));
  const Q = cellQ.data;
  for (let t = 0; t < nt; t++) {
    const a = I[3 * t], b = I[3 * t + 1], c = I[3 * t + 2];
    if (a >= nv || b >= nv || c >= nv) continue;
    const ca = vCell[a], cb = vCell[b], cc = vCell[c];
    // quadric of the triangle plane (coordinates relative to the grid origin for precision)
    const ax = P[3 * a] - ox, ay = P[3 * a + 1] - oy, az = P[3 * a + 2] - oz;
    const ux = P[3 * b] - P[3 * a], uy = P[3 * b + 1] - P[3 * a + 1], uz = P[3 * b + 2] - P[3 * a + 2];
    const vx = P[3 * c] - P[3 * a], vy = P[3 * c + 1] - P[3 * a + 1], vz = P[3 * c + 2] - P[3 * a + 2];
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const l = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (l > 0) {
      nx /= l;
      ny /= l;
      nz /= l;
      const d = -(nx * ax + ny * ay + nz * az);
      const w = 0.5 * l;
      for (const id of [ca, cb, cc]) {
        const o = 10 * id;
        Q[o] += w * nx * nx;
        Q[o + 1] += w * nx * ny;
        Q[o + 2] += w * nx * nz;
        Q[o + 3] += w * nx * d;
        Q[o + 4] += w * ny * ny;
        Q[o + 5] += w * ny * nz;
        Q[o + 6] += w * ny * d;
        Q[o + 7] += w * nz * nz;
        Q[o + 8] += w * nz * d;
        Q[o + 9] += w * d * d;
      }
    }
    if (ca !== cb && cb !== cc && cc !== ca) outTri.push3(ca, cb, cc);
    if (opts.onProgress && (t & 0xfffff) === 0) opts.onProgress(0.3 + (0.5 * t) / nt, "Clustering triangles");
  }
  // representatives
  const positions = new Float32Array(nCells * 3);
  const S = cellSum.data;
  for (let id = 0; id < nCells; id++) {
    const n = S[4 * id + 3];
    const mx = S[4 * id] / n - ox, my = S[4 * id + 1] / n - oy, mz = S[4 * id + 2] / n - oz;
    let x = mx, y = my, z = mz;
    const o = 10 * id;
    const a11 = Q[o], a12 = Q[o + 1], a13 = Q[o + 2], a22 = Q[o + 4], a23 = Q[o + 5], a33 = Q[o + 7];
    const r1 = -Q[o + 3], r2 = -Q[o + 6], r3 = -Q[o + 8];
    const c11 = a22 * a33 - a23 * a23, c12 = a13 * a23 - a12 * a33, c13 = a12 * a23 - a13 * a22;
    const det = a11 * c11 + a12 * c12 + a13 * c13;
    const tr = a11 + a22 + a33;
    if (tr > 0 && Math.abs(det) > 1e-6 * tr * tr * tr) {
      const c22 = a11 * a33 - a13 * a13, c23 = a12 * a13 - a11 * a23, c33 = a11 * a22 - a12 * a12;
      const sx = (c11 * r1 + c12 * r2 + c13 * r3) / det;
      const sy = (c12 * r1 + c22 * r2 + c23 * r3) / det;
      const sz = (c13 * r1 + c23 * r2 + c33 * r3) / det;
      // accept if it stays within the (slightly enlarged) cell around the mean
      if (Math.abs(sx - mx) <= cell && Math.abs(sy - my) <= cell && Math.abs(sz - mz) <= cell) {
        x = sx;
        y = sy;
        z = sz;
      }
    }
    positions[3 * id] = x + ox;
    positions[3 * id + 1] = y + oy;
    positions[3 * id + 2] = z + oz;
  }
  const out: MeshData = { positions, indices: outTri.toArray() };
  if (cellCol) {
    const colors = new Uint8Array(nCells * 3);
    const cc = cellCol.data;
    for (let id = 0; id < nCells; id++) {
      const n = S[4 * id + 3];
      for (let k = 0; k < 3; k++) colors[3 * id + k] = Math.max(0, Math.min(255, Math.round(cc[3 * id + k] / n)));
    }
    out.colors = colors;
  }
  opts.onProgress?.(0.9, "Removing duplicate triangles");
  const cleaned = removeDegenerateTriangles(out, { relAreaTolerance: 0 });
  if (mesh.normals) cleaned.normals = computeVertexNormals(cleaned);
  opts.onProgress?.(1, "Clustering done");
  return cleaned;
}

// ---------------------------------------------------------------------------
// Combined
// ---------------------------------------------------------------------------

export interface DecimateOptions {
  targetTriangles: number;
  preserveBoundary?: boolean;
  /** Use a clustering pre-pass above this many triangles (default 8,000,000). */
  clusterAbove?: number;
  onProgress?: ProgressFn;
}

/**
 * Decimate to `targetTriangles`: inputs larger than `clusterAbove` are first reduced by vertex
 * clustering (linear time, low memory) to about max(2 × target, clusterAbove / 2) triangles,
 * then quadric edge collapse produces the exact target with good quality.
 */
export function decimate(mesh: MeshData, opts: DecimateOptions): MeshData {
  const nt = triangleCountOf(mesh);
  const target = Math.max(0, Math.floor(opts.targetTriangles));
  const clusterAbove = opts.clusterAbove ?? 8_000_000;
  const onProgress = opts.onProgress;
  if (nt <= target) return copyMesh(mesh);
  let work = mesh;
  let base = 0;
  if (nt > clusterAbove) {
    const inter = Math.max(2 * target, Math.floor(clusterAbove / 2));
    if (inter < nt) {
      work = decimateClustering(mesh, { targetTriangles: inter, onProgress: onProgress ? (f, m) => onProgress(0.4 * f, m) : undefined });
      base = 0.4;
    }
  }
  if (triangleCountOf(work) <= target) return work === mesh ? copyMesh(mesh) : work;
  return decimateQEM(work, {
    targetTriangles: target,
    preserveBoundary: opts.preserveBoundary,
    onProgress: onProgress ? (f, m) => onProgress(base + (1 - base) * f, m) : undefined,
  });
}
