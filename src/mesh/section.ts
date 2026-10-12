/**
 * Planar sections of a triangle mesh → polylines.
 *
 * Robustness: vertex sides are classified with a symbolic perturbation (d >= 0 counts as the
 * positive side), so every triangle is either untouched or crossed by exactly two of its edges
 * and the crossing edges of adjacent triangles coincide. Crossing points are keyed by their
 * (undirected) mesh edge, so segments of neighbouring triangles join exactly via shared edges —
 * no distance tolerance is involved. Vertices lying exactly on the plane yield several
 * coincident crossing points; zero-length steps are removed from the output.
 */
import type { MeshData, PlaneDef, Vec3 } from "../core/types";
import { triangleCountOf, vertexCountOf } from "./topology";
import { GrowableF64, GrowableI32, GrowableU32 } from "./util/growable";
import { PairHashMap } from "./util/pair-hash";

export interface SectionPolyline {
  points: Vec3[];
  closed: boolean;
}

export interface SectionOptions {
  /**
   * Optional candidate triangles (e.g. from MeshSpatialIndex.trianglesIntersectingPlane):
   * only these are tested. Without it every triangle is classified (3 dot products each).
   */
  candidates?: ArrayLike<number>;
  /** Consecutive output points closer than this are merged. Default 1e-9 × (|origin| + 1). */
  mergeDistance?: number;
}

/**
 * Intersect the mesh with a plane. Returns closed loops and open chains (open where the section
 * leaves the mesh through a boundary, or at non-manifold edges). Closed loops do not repeat the
 * first point at the end. Polylines are oriented consistently with the triangle winding where
 * the mesh orientation allows it.
 */
export function sectionMesh(mesh: MeshData, plane: PlaneDef, opts: SectionOptions = {}): SectionPolyline[] {
  const P = mesh.positions, I = mesh.indices;
  const nt = triangleCountOf(mesh);
  const nv = vertexCountOf(mesh);
  const [nx, ny, nz] = plane.normal;
  const [ox, oy, oz] = plane.origin;
  const dist = (v: number) => (P[3 * v] - ox) * nx + (P[3 * v + 1] - oy) * ny + (P[3 * v + 2] - oz) * nz;

  const nodeOf = new PairHashMap(1024);
  const nodePos = new GrowableF64(1024);
  const segA = new GrowableU32(1024);
  const segB = new GrowableU32(1024);

  const node = (u: number, v: number, du: number, dv: number): number => {
    let lo = u, hi = v, dlo = du, dhi = dv;
    if (u > v) {
      lo = v;
      hi = u;
      dlo = dv;
      dhi = du;
    }
    const existing = nodeOf.get(lo, hi);
    if (existing >= 0) return existing;
    const id = nodePos.length / 3;
    // dlo and dhi have different (perturbed) signs, so dlo - dhi != 0 unless both are 0,
    // which cannot happen (0 counts as positive).
    const s = dlo / (dlo - dhi);
    nodePos.push3(
      P[3 * lo] + s * (P[3 * hi] - P[3 * lo]),
      P[3 * lo + 1] + s * (P[3 * hi + 1] - P[3 * lo + 1]),
      P[3 * lo + 2] + s * (P[3 * hi + 2] - P[3 * lo + 2]),
    );
    nodeOf.set(lo, hi, id);
    return id;
  };

  const processTriangle = (t: number) => {
    const a = I[3 * t], b = I[3 * t + 1], c = I[3 * t + 2];
    if (a >= nv || b >= nv || c >= nv) return;
    const da = dist(a), db = dist(b), dc = dist(c);
    const pa = da >= 0, pb = db >= 0, pc = dc >= 0;
    if (pa === pb && pb === pc) return;
    // find the edge leaving the positive side (+ → −) and the one entering it (− → +)
    let exitNode = -1, entryNode = -1;
    if (pa !== pb) {
      const id = node(a, b, da, db);
      if (pa) exitNode = id;
      else entryNode = id;
    }
    if (pb !== pc) {
      const id = node(b, c, db, dc);
      if (pb) exitNode = id;
      else entryNode = id;
    }
    if (pc !== pa) {
      const id = node(c, a, dc, da);
      if (pc) exitNode = id;
      else entryNode = id;
    }
    // oriented segment entry -> exit
    segA.push(entryNode);
    segB.push(exitNode);
  };

  if (opts.candidates) {
    const cand = opts.candidates;
    for (let i = 0; i < cand.length; i++) {
      const t = cand[i];
      if (t < nt) processTriangle(t);
    }
  } else {
    for (let t = 0; t < nt; t++) processTriangle(t);
  }

  return joinSegments(nodePos.view(), segA.view(), segB.view(), opts.mergeDistance ?? 1e-9 * (Math.hypot(ox, oy, oz) + 1));
}

/**
 * Join directed segments (node ids) into polylines. Nodes with more than two incident segments
 * (non-manifold) keep their first two links; the others become chain ends.
 */
function joinSegments(nodePos: Float64Array, segA: Uint32Array, segB: Uint32Array, mergeDist: number): SectionPolyline[] {
  const nn = nodePos.length / 3;
  const ns = segA.length;
  // two segment slots per node
  const slot = new Int32Array(nn * 2).fill(-1);
  const segUsable = new Uint8Array(ns);
  for (let s = 0; s < ns; s++) {
    const a = segA[s], b = segB[s];
    if (a === b) continue;
    // skip exact duplicates (e.g. duplicate faces) — a already linked to b
    const a0 = slot[2 * a], a1 = slot[2 * a + 1];
    if ((a0 >= 0 && other(segA, segB, a0, a) === b) || (a1 >= 0 && other(segA, segB, a1, a) === b)) continue;
    const fa = a0 < 0 ? 0 : a1 < 0 ? 1 : -1;
    const fb = slot[2 * b] < 0 ? 0 : slot[2 * b + 1] < 0 ? 1 : -1;
    if (fa < 0 || fb < 0) continue; // non-manifold overflow
    slot[2 * a + fa] = s;
    slot[2 * b + fb] = s;
    segUsable[s] = 1;
  }
  const visited = new Uint8Array(nn);
  const usedSeg = new Uint8Array(ns);
  const out: SectionPolyline[] = [];
  const chain = new GrowableU32(256);
  const dirs = new GrowableI32(256);

  const walk = (start: number) => {
    chain.clear();
    dirs.clear();
    let cur = start;
    visited[cur] = 1;
    chain.push(cur);
    let closed = false;
    for (;;) {
      let s = slot[2 * cur];
      if (s < 0 || usedSeg[s]) s = slot[2 * cur + 1];
      if (s < 0 || usedSeg[s]) break;
      usedSeg[s] = 1;
      const nxt = other(segA, segB, s, cur);
      dirs.push(segA[s] === cur ? 1 : -1);
      if (nxt === start) {
        closed = true;
        break;
      }
      if (visited[nxt]) break; // should not happen with 2 slots; be safe
      visited[nxt] = 1;
      chain.push(nxt);
      cur = nxt;
    }
    emitChain(chain.view(), dirs.view(), closed);
  };

  const emitChain = (ids: Uint32Array, d: Int32Array, closed: boolean) => {
    let dirSum = 0;
    for (let i = 0; i < d.length; i++) dirSum += d[i];
    const n = ids.length;
    const pts: Vec3[] = [];
    const md2 = mergeDist * mergeDist;
    for (let k = 0; k < n; k++) {
      const id = ids[dirSum >= 0 ? k : n - 1 - k];
      const p: Vec3 = [nodePos[3 * id], nodePos[3 * id + 1], nodePos[3 * id + 2]];
      const last = pts[pts.length - 1];
      if (last) {
        const dx = p[0] - last[0], dy = p[1] - last[1], dz = p[2] - last[2];
        if (dx * dx + dy * dy + dz * dz <= md2) continue;
      }
      pts.push(p);
    }
    if (closed && pts.length > 1) {
      const f = pts[0], l = pts[pts.length - 1];
      const dx = f[0] - l[0], dy = f[1] - l[1], dz = f[2] - l[2];
      if (dx * dx + dy * dy + dz * dz <= md2) pts.pop();
    }
    if (closed ? pts.length >= 3 : pts.length >= 2) out.push({ points: pts, closed });
  };

  // open chains first: start at nodes with exactly one usable link
  for (let v = 0; v < nn; v++) {
    if (visited[v]) continue;
    const deg = (slot[2 * v] >= 0 ? 1 : 0) + (slot[2 * v + 1] >= 0 ? 1 : 0);
    if (deg === 1) walk(v);
  }
  // remaining nodes lie on closed loops (or are isolated)
  for (let v = 0; v < nn; v++) {
    if (visited[v]) continue;
    const deg = (slot[2 * v] >= 0 ? 1 : 0) + (slot[2 * v + 1] >= 0 ? 1 : 0);
    if (deg === 0) {
      visited[v] = 1;
      continue;
    }
    walk(v);
  }
  return out;
}

function other(segA: Uint32Array, segB: Uint32Array, s: number, n: number): number {
  return segA[s] === n ? segB[s] : segA[s];
}

/**
 * Several parallel sections at once (planes with a common normal at the given signed offsets
 * from `origin`). Each triangle is visited once per plane it actually spans, by bucketing
 * triangles on their distance range. Returns one result list per offset.
 */
export function sectionMeshParallel(mesh: MeshData, origin: Vec3, normal: Vec3, offsets: number[], opts: Omit<SectionOptions, "candidates"> = {}): SectionPolyline[][] {
  const P = mesh.positions, I = mesh.indices;
  const nt = triangleCountOf(mesh);
  const m = offsets.length;
  if (m === 0) return [];
  // sort plane offsets, remember original order
  const order = offsets.map((o, i) => [o, i] as [number, number]).sort((a, b) => a[0] - b[0]);
  const sorted = order.map((o) => o[0]);
  // per plane candidate lists (CSR) built from each triangle's [dmin, dmax] interval
  const lower = (x: number) => {
    let lo = 0, hi = m;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (sorted[mid] < x) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  };
  const counts = new Uint32Array(m + 1);
  const range = new Uint32Array(nt * 2);
  const [nx, ny, nz] = normal;
  const [ox, oy, oz] = origin;
  const dist = (v: number) => (P[3 * v] - ox) * nx + (P[3 * v + 1] - oy) * ny + (P[3 * v + 2] - oz) * nz;
  for (let t = 0; t < nt; t++) {
    const da = dist(I[3 * t]), db = dist(I[3 * t + 1]), dc = dist(I[3 * t + 2]);
    const dmin = Math.min(da, db, dc), dmax = Math.max(da, db, dc);
    // planes p with dmin < offset <= dmax ... use a closed interval to be safe
    const i0 = lower(dmin);
    let i1 = lower(dmax);
    while (i1 < m && sorted[i1] <= dmax) i1++;
    range[2 * t] = i0;
    range[2 * t + 1] = i1;
    for (let i = i0; i < i1; i++) counts[i + 1]++;
  }
  for (let i = 0; i < m; i++) counts[i + 1] += counts[i];
  const lists = new Uint32Array(counts[m]);
  const fill = counts.slice(0, m);
  for (let t = 0; t < nt; t++) for (let i = range[2 * t]; i < range[2 * t + 1]; i++) lists[fill[i]++] = t;
  const result: SectionPolyline[][] = new Array(m);
  for (let i = 0; i < m; i++) {
    const off = sorted[i];
    const plane: PlaneDef = { kind: "plane", origin: [ox + nx * off, oy + ny * off, oz + nz * off], normal: [nx, ny, nz] };
    result[order[i][1]] = sectionMesh(mesh, plane, { ...opts, candidates: lists.subarray(counts[i], counts[i + 1]) });
  }
  return result;
}
