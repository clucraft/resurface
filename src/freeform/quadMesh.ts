/**
 * Quad control cage ("QuadMesh") for freeform subdivision modelling.
 *
 *  - data structure + generators (grid, box, cylinder, torus cages)
 *  - topology: a half-edge view of the polygon soup with edge sharpness and the vertex
 *    classification used by the Catmull–Clark rules (smooth / dart / crease / corner)
 *  - validation (manifoldness, orientation) and `orient`
 *  - editing operations called by the UI (add vertex/face, remove face, insert edge loop,
 *    extrude boundary edges, merge/weld vertices)
 *
 * Conventions: faces are CCW seen from outside (normal = right-hand rule); quads are the norm,
 * triangles / n-gons are allowed in the cage. Editing operations MUTATE the given mesh in place
 * (use `cloneQuadMesh` first when an undo snapshot is needed).
 */
import type { MeshData, Vec3 } from "../core/types";
import { add, cross, dot, length, lerp, normalize, scale, sub } from "../core/vec";

export interface QuadMesh {
  vertices: Vec3[];
  /** vertex indices per face, CCW seen from outside; mostly quads */
  faces: number[][];
  /** sharp (crease) edges as unordered vertex pairs */
  creaseEdges?: [number, number][];
  /** per-vertex "do not move" flags used by the fitting */
  fixed?: boolean[];
}

/** Vertex classification for the subdivision rules. */
export const VertexKind = {
  /** interior, no sharp edge */
  Smooth: 0,
  /** interior, exactly one sharp edge (smooth vertex rule) */
  Dart: 1,
  /** exactly two sharp edges (boundary vertices with >= 2 faces, interior crease vertices) */
  Crease: 2,
  /** >= 3 sharp edges, boundary vertex with a single face, non-manifold or isolated vertex */
  Corner: 3,
} as const;
export type VertexKind = (typeof VertexKind)[keyof typeof VertexKind];

export const EdgeSharpness = {
  None: 0,
  /** edge with exactly one face */
  Boundary: 1,
  /** user crease */
  Crease: 2,
  /** more than two faces or inconsistent orientation of the two faces */
  NonManifold: 3,
} as const;

/** Half-edge view of a QuadMesh. Half-edge h of face f: faceStart[f] <= h < faceStart[f+1], starting at heV[h]. */
export interface Topology {
  nv: number;
  nf: number;
  /** number of half-edges */
  nh: number;
  /** number of undirected edges */
  ne: number;
  faceStart: Int32Array;
  /** 1 if the face takes part in the topology (>= 3 distinct, valid indices) */
  faceValid: Uint8Array;
  heV: Int32Array;
  heF: Int32Array;
  heNext: Int32Array;
  hePrev: Int32Array;
  /** opposite half-edge or -1 (boundary / non-manifold / inconsistent) */
  heTwin: Int32Array;
  heEdge: Int32Array;
  /** 2 per edge: (lo, hi) */
  edgeV: Int32Array;
  /** one half-edge of each edge */
  edgeHe: Int32Array;
  edgeFaceCount: Int32Array;
  edgeSharp: Uint8Array;
  /** CSR of outgoing half-edges per vertex */
  vOutStart: Int32Array;
  vOut: Int32Array;
  /** CSR of incident undirected edges per vertex */
  vEdgeStart: Int32Array;
  vEdges: Int32Array;
  vSharpCount: Int32Array;
  vKind: Uint8Array;
  vManifold: Uint8Array;
  vBoundary: Uint8Array;
  /** key lo * nv + hi -> edge id */
  edgeIndex: Map<number, number>;
}

/** Ordered one-ring of a manifold vertex (CCW around the outward normal). */
export interface VertexRing {
  v: number;
  /** neighbour vertices; for open (boundary) rings length = faces.length + 1 */
  nbr: number[];
  /** undirected edge id per neighbour */
  edges: number[];
  /** face i lies between nbr[i] and nbr[(i + 1) % nbr.length] */
  faces: number[];
  /** half-edge of face i starting at v (it runs v -> nbr[i]) */
  faceHe: number[];
  closed: boolean;
  manifold: boolean;
}

/** A run of faces around a vertex between two sharp edges (or the whole ring). Indices refer to the ring arrays. */
export interface RingSector {
  /** ring neighbour indices (k + 1 entries for k faces; first/last are the bounding sharp edges) */
  nbrIdx: number[];
  /** ring face indices (k entries) */
  faceIdx: number[];
  /** true if the sector is the full closed ring without sharp edges */
  full: boolean;
}

// ---------------------------------------------------------------------------
// Basic helpers
// ---------------------------------------------------------------------------

export function cloneQuadMesh(qm: QuadMesh): QuadMesh {
  const out: QuadMesh = {
    vertices: qm.vertices.map((p) => [p[0], p[1], p[2]] as Vec3),
    faces: qm.faces.map((f) => f.slice()),
  };
  if (qm.creaseEdges) out.creaseEdges = qm.creaseEdges.map((e) => [e[0], e[1]] as [number, number]);
  if (qm.fixed) out.fixed = qm.fixed.slice();
  return out;
}

/** Canonical string key of an undirected edge. */
export const edgeKey = (a: number, b: number): string => (a < b ? `${a}_${b}` : `${b}_${a}`);

export function isAllQuads(qm: QuadMesh): boolean {
  for (const f of qm.faces) if (f.length !== 4) return false;
  return true;
}

function isFaceValid(face: number[], nv: number): boolean {
  const n = face.length;
  if (n < 3) return false;
  for (let i = 0; i < n; i++) {
    const v = face[i];
    if (!Number.isInteger(v) || v < 0 || v >= nv) return false;
    for (let j = 0; j < i; j++) if (face[j] === v) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Topology
// ---------------------------------------------------------------------------

export function buildTopology(qm: QuadMesh): Topology {
  return buildTopologyRaw(qm.vertices.length, qm.faces, qm.creaseEdges);
}

/** Topology from raw connectivity (positions are not needed). */
export function buildTopologyRaw(nv: number, faces: number[][], creaseEdges?: [number, number][]): Topology {
  const nf = faces.length;
  const faceStart = new Int32Array(nf + 1);
  const faceValid = new Uint8Array(nf);
  let nh = 0;
  for (let f = 0; f < nf; f++) {
    faceStart[f] = nh;
    if (isFaceValid(faces[f], nv)) {
      faceValid[f] = 1;
      nh += faces[f].length;
    }
  }
  faceStart[nf] = nh;
  const heV = new Int32Array(nh), heF = new Int32Array(nh), heNext = new Int32Array(nh), hePrev = new Int32Array(nh);
  for (let f = 0; f < nf; f++) {
    if (!faceValid[f]) continue;
    const s = faceStart[f], face = faces[f], n = face.length;
    for (let k = 0; k < n; k++) {
      heV[s + k] = face[k];
      heF[s + k] = f;
      heNext[s + k] = s + ((k + 1) % n);
      hePrev[s + k] = s + ((k + n - 1) % n);
    }
  }
  // undirected edges
  const edgeIndex = new Map<number, number>();
  const heEdge = new Int32Array(nh);
  const ev: number[] = [];
  const eh0: number[] = [];
  const eh1: number[] = [];
  const ecount: number[] = [];
  for (let h = 0; h < nh; h++) {
    const a = heV[h], b = heV[heNext[h]];
    const lo = a < b ? a : b, hi = a < b ? b : a;
    const key = lo * nv + hi;
    let e = edgeIndex.get(key);
    if (e === undefined) {
      e = ecount.length;
      edgeIndex.set(key, e);
      ev.push(lo, hi);
      eh0.push(h);
      eh1.push(-1);
      ecount.push(0);
    } else if (eh1[e] === -1) eh1[e] = h;
    ecount[e]++;
    heEdge[h] = e;
  }
  const ne = ecount.length;
  const edgeV = Int32Array.from(ev);
  const edgeHe = Int32Array.from(eh0);
  const edgeFaceCount = Int32Array.from(ecount);
  const edgeSharp = new Uint8Array(ne);
  const heTwin = new Int32Array(nh).fill(-1);
  for (let e = 0; e < ne; e++) {
    const c = ecount[e];
    if (c === 1) edgeSharp[e] = EdgeSharpness.Boundary;
    else if (c === 2) {
      const h0 = eh0[e], h1 = eh1[e];
      if (heV[h0] !== heV[h1]) {
        heTwin[h0] = h1;
        heTwin[h1] = h0;
      } else edgeSharp[e] = EdgeSharpness.NonManifold;
    } else edgeSharp[e] = EdgeSharpness.NonManifold;
  }
  if (creaseEdges) {
    for (const ce of creaseEdges) {
      const a = ce[0], b = ce[1];
      if (a === b || a < 0 || b < 0 || a >= nv || b >= nv) continue;
      const lo = a < b ? a : b, hi = a < b ? b : a;
      const e = edgeIndex.get(lo * nv + hi);
      if (e !== undefined && edgeSharp[e] === EdgeSharpness.None) edgeSharp[e] = EdgeSharpness.Crease;
    }
  }
  // vertex -> outgoing half-edges
  const vOutStart = new Int32Array(nv + 1);
  for (let h = 0; h < nh; h++) vOutStart[heV[h] + 1]++;
  for (let v = 0; v < nv; v++) vOutStart[v + 1] += vOutStart[v];
  const vOut = new Int32Array(nh);
  {
    const fill = vOutStart.slice(0, nv);
    for (let h = 0; h < nh; h++) vOut[fill[heV[h]]++] = h;
  }
  // vertex -> incident edges
  const vEdgeStart = new Int32Array(nv + 1);
  for (let e = 0; e < ne; e++) {
    vEdgeStart[edgeV[2 * e] + 1]++;
    vEdgeStart[edgeV[2 * e + 1] + 1]++;
  }
  for (let v = 0; v < nv; v++) vEdgeStart[v + 1] += vEdgeStart[v];
  const vEdges = new Int32Array(2 * ne);
  {
    const fill = vEdgeStart.slice(0, nv);
    for (let e = 0; e < ne; e++) {
      vEdges[fill[edgeV[2 * e]]++] = e;
      vEdges[fill[edgeV[2 * e + 1]]++] = e;
    }
  }
  const vSharpCount = new Int32Array(nv);
  const vBoundary = new Uint8Array(nv);
  for (let v = 0; v < nv; v++) {
    for (let p = vEdgeStart[v]; p < vEdgeStart[v + 1]; p++) {
      const e = vEdges[p];
      if (edgeSharp[e] !== EdgeSharpness.None) vSharpCount[v]++;
      if (edgeFaceCount[e] === 1) vBoundary[v] = 1;
    }
  }
  const vManifold = new Uint8Array(nv);
  const vKind = new Uint8Array(nv);
  for (let v = 0; v < nv; v++) {
    const s = vOutStart[v], e = vOutStart[v + 1];
    const nOut = e - s;
    if (nOut === 0) {
      vManifold[v] = 1;
      vKind[v] = VertexKind.Corner;
      continue;
    }
    let nb = 0, start = vOut[s];
    for (let p = s; p < e; p++)
      if (heTwin[vOut[p]] === -1) {
        nb++;
        start = vOut[p];
      }
    let manifold = false;
    if (nb <= 1) {
      let h = start, count = 0;
      for (;;) {
        count++;
        const t = heTwin[hePrev[h]];
        if (t === -1) {
          manifold = nb === 1 && count === nOut;
          break;
        }
        h = t;
        if (h === start) {
          manifold = nb === 0 && count === nOut;
          break;
        }
        if (count > nOut) break;
      }
    }
    // the number of incident edges must match a disc / half-disc
    const nEdges = vEdgeStart[v + 1] - vEdgeStart[v];
    if (manifold && nEdges !== nOut + (nb === 1 ? 1 : 0)) manifold = false;
    vManifold[v] = manifold ? 1 : 0;
    if (!manifold) {
      vKind[v] = VertexKind.Corner;
      continue;
    }
    const sc = vSharpCount[v];
    if (sc === 0) vKind[v] = VertexKind.Smooth;
    else if (sc === 1) vKind[v] = VertexKind.Dart;
    else if (sc === 2) vKind[v] = nb === 1 && nOut === 1 ? VertexKind.Corner : VertexKind.Crease;
    else vKind[v] = VertexKind.Corner;
  }
  return {
    nv, nf, nh, ne, faceStart, faceValid, heV, heF, heNext, hePrev, heTwin, heEdge,
    edgeV, edgeHe, edgeFaceCount, edgeSharp, vOutStart, vOut, vEdgeStart, vEdges,
    vSharpCount, vKind, vManifold, vBoundary, edgeIndex,
  };
}

/** Edge id of (a, b) or -1. */
export function findEdge(topo: Topology, a: number, b: number): number {
  if (a < 0 || b < 0 || a >= topo.nv || b >= topo.nv) return -1;
  const lo = a < b ? a : b, hi = a < b ? b : a;
  return topo.edgeIndex.get(lo * topo.nv + hi) ?? -1;
}

/** Number of incident edges. */
export const valenceOf = (topo: Topology, v: number): number => topo.vEdgeStart[v + 1] - topo.vEdgeStart[v];
/** Number of incident faces (face corners). */
export const faceCountOf = (topo: Topology, v: number): number => topo.vOutStart[v + 1] - topo.vOutStart[v];
/** Other endpoint of edge e. */
export const otherEnd = (topo: Topology, e: number, v: number): number =>
  topo.edgeV[2 * e] === v ? topo.edgeV[2 * e + 1] : topo.edgeV[2 * e];

/** The neighbours of v across its sharp edges (in edge order). */
export function sharpNeighbours(topo: Topology, v: number): number[] {
  const out: number[] = [];
  for (let p = topo.vEdgeStart[v]; p < topo.vEdgeStart[v + 1]; p++) {
    const e = topo.vEdges[p];
    if (topo.edgeSharp[e] !== EdgeSharpness.None) out.push(otherEnd(topo, e, v));
  }
  return out;
}

/** Ordered (CCW) one-ring of v. For non-manifold vertices `manifold` is false and the arrays are unordered. */
export function vertexRing(topo: Topology, v: number): VertexRing {
  const s = topo.vOutStart[v], e = topo.vOutStart[v + 1];
  const ring: VertexRing = { v, nbr: [], edges: [], faces: [], faceHe: [], closed: false, manifold: topo.vManifold[v] === 1 };
  if (e === s) return ring;
  if (!ring.manifold) {
    for (let p = s; p < e; p++) {
      const h = topo.vOut[p];
      ring.faces.push(topo.heF[h]);
      ring.faceHe.push(h);
      ring.nbr.push(topo.heV[topo.heNext[h]]);
      ring.edges.push(topo.heEdge[h]);
    }
    return ring;
  }
  let start = topo.vOut[s];
  for (let p = s; p < e; p++) if (topo.heTwin[topo.vOut[p]] === -1) start = topo.vOut[p];
  let h = start;
  const guard = e - s + 2;
  for (let it = 0; it < guard; it++) {
    ring.faces.push(topo.heF[h]);
    ring.faceHe.push(h);
    ring.nbr.push(topo.heV[topo.heNext[h]]);
    ring.edges.push(topo.heEdge[h]);
    const pr = topo.hePrev[h];
    const t = topo.heTwin[pr];
    if (t === -1) {
      ring.nbr.push(topo.heV[pr]);
      ring.edges.push(topo.heEdge[pr]);
      ring.closed = false;
      break;
    }
    h = t;
    if (h === start) {
      ring.closed = true;
      break;
    }
  }
  return ring;
}

/** Split a ring into sectors at sharp edges. A smooth closed ring yields one `full` sector. */
export function ringSectors(topo: Topology, ring: VertexRing): RingSector[] {
  const m = ring.nbr.length;
  const isSharp = (i: number) => topo.edgeSharp[ring.edges[i]] !== EdgeSharpness.None;
  const out: RingSector[] = [];
  if (m === 0) return out;
  if (!ring.closed) {
    let start = 0;
    for (let i = 1; i < m; i++) {
      if (i === m - 1 || isSharp(i)) {
        const nbrIdx: number[] = [], faceIdx: number[] = [];
        for (let j = start; j <= i; j++) nbrIdx.push(j);
        for (let j = start; j < i; j++) faceIdx.push(j);
        out.push({ nbrIdx, faceIdx, full: false });
        start = i;
      }
    }
    return out;
  }
  const sharp: number[] = [];
  for (let i = 0; i < m; i++) if (isSharp(i)) sharp.push(i);
  if (sharp.length === 0) {
    const nbrIdx: number[] = [], faceIdx: number[] = [];
    for (let i = 0; i < m; i++) {
      nbrIdx.push(i);
      faceIdx.push(i);
    }
    out.push({ nbrIdx, faceIdx, full: true });
    return out;
  }
  for (let k = 0; k < sharp.length; k++) {
    const a = sharp[k];
    const b = k + 1 < sharp.length ? sharp[k + 1] : sharp[0] + m;
    const nbrIdx: number[] = [], faceIdx: number[] = [];
    for (let j = a; j <= b; j++) nbrIdx.push(j % m);
    for (let j = a; j < b; j++) faceIdx.push(j % m);
    out.push({ nbrIdx, faceIdx, full: false });
  }
  return out;
}

// ---------------------------------------------------------------------------
// Convenience topology queries
// ---------------------------------------------------------------------------

/** Number of incident edges per vertex. */
export function vertexValence(qm: QuadMesh): Int32Array {
  const topo = buildTopology(qm);
  const out = new Int32Array(topo.nv);
  for (let v = 0; v < topo.nv; v++) out[v] = valenceOf(topo, v);
  return out;
}

/** Unique undirected edges (lo, hi) of the cage. */
export function cageEdges(qm: QuadMesh): [number, number][] {
  const topo = buildTopology(qm);
  const out: [number, number][] = [];
  for (let e = 0; e < topo.ne; e++) out.push([topo.edgeV[2 * e], topo.edgeV[2 * e + 1]]);
  return out;
}

/** Map edgeKey -> faces using the edge. */
export function edgeFaceMap(qm: QuadMesh): Map<string, number[]> {
  const m = new Map<string, number[]>();
  qm.faces.forEach((f, fi) => {
    for (let i = 0; i < f.length; i++) {
      const k = edgeKey(f[i], f[(i + 1) % f.length]);
      const l = m.get(k);
      if (l) l.push(fi);
      else m.set(k, [fi]);
    }
  });
  return m;
}

/** For each face and each of its edges (f[i], f[i+1]) the neighbouring face across it, or -1. */
export function faceAdjacency(qm: QuadMesh): number[][] {
  const topo = buildTopology(qm);
  return qm.faces.map((face, f) => {
    if (!topo.faceValid[f]) return face.map(() => -1);
    const s = topo.faceStart[f];
    return face.map((_, k) => {
      const t = topo.heTwin[s + k];
      return t >= 0 ? topo.heF[t] : -1;
    });
  });
}

/** Per-vertex boundary flags. */
export function boundaryVertices(qm: QuadMesh): boolean[] {
  const topo = buildTopology(qm);
  return Array.from(topo.vBoundary, (b) => b === 1);
}

/** Boundary edges as oriented pairs (a, b) following the face orientation. */
export function boundaryEdges(qm: QuadMesh): [number, number][] {
  const topo = buildTopology(qm);
  const out: [number, number][] = [];
  for (let h = 0; h < topo.nh; h++)
    if (topo.edgeFaceCount[topo.heEdge[h]] === 1) out.push([topo.heV[h], topo.heV[topo.heNext[h]]]);
  return out;
}

/** Ordered boundary loops (vertex lists following the face orientation). */
export function boundaryLoops(qm: QuadMesh): number[][] {
  const topo = buildTopology(qm);
  return boundaryLoopsTopo(topo);
}

export function boundaryLoopsTopo(topo: Topology): number[][] {
  const outBoundary = new Map<number, number[]>();
  const bnd: number[] = [];
  for (let h = 0; h < topo.nh; h++) {
    if (topo.edgeFaceCount[topo.heEdge[h]] !== 1) continue;
    bnd.push(h);
    const v = topo.heV[h];
    const l = outBoundary.get(v);
    if (l) l.push(h);
    else outBoundary.set(v, [h]);
  }
  const used = new Set<number>();
  const loops: number[][] = [];
  for (const h0 of bnd) {
    if (used.has(h0)) continue;
    const loop: number[] = [];
    let h = h0;
    while (!used.has(h)) {
      used.add(h);
      loop.push(topo.heV[h]);
      const w = topo.heV[topo.heNext[h]];
      const cand = outBoundary.get(w);
      let next = -1;
      if (cand) for (const c of cand) if (!used.has(c)) { next = c; break; }
      if (next < 0) break;
      h = next;
    }
    loops.push(loop);
  }
  return loops;
}

// ---------------------------------------------------------------------------
// Validation and orientation
// ---------------------------------------------------------------------------

export interface ValidationReport {
  /** manifold, consistently oriented, all faces valid */
  ok: boolean;
  errors: string[];
  warnings: string[];
  invalidFaces: number[];
  nonManifoldEdges: [number, number][];
  /** edges used twice in the same direction (orientation flips) */
  inconsistentEdges: [number, number][];
  nonManifoldVertices: number[];
  unusedVertices: number[];
  boundaryEdgeCount: number;
  boundaryLoopCount: number;
  components: number;
  eulerCharacteristic: number;
  closed: boolean;
}

export function validate(qm: QuadMesh): ValidationReport {
  const topo = buildTopology(qm);
  const errors: string[] = [];
  const warnings: string[] = [];
  const invalidFaces: number[] = [];
  for (let f = 0; f < topo.nf; f++) if (!topo.faceValid[f]) invalidFaces.push(f);
  const nonManifoldEdges: [number, number][] = [];
  const inconsistentEdges: [number, number][] = [];
  let boundaryEdgeCount = 0;
  for (let e = 0; e < topo.ne; e++) {
    const pair: [number, number] = [topo.edgeV[2 * e], topo.edgeV[2 * e + 1]];
    const c = topo.edgeFaceCount[e];
    if (c === 1) boundaryEdgeCount++;
    else if (c > 2) nonManifoldEdges.push(pair);
    else if (c === 2 && topo.heTwin[topo.edgeHe[e]] === -1) inconsistentEdges.push(pair);
  }
  const nonManifoldVertices: number[] = [];
  const unusedVertices: number[] = [];
  for (let v = 0; v < topo.nv; v++) {
    if (faceCountOf(topo, v) === 0) unusedVertices.push(v);
    else if (!topo.vManifold[v]) nonManifoldVertices.push(v);
  }
  // only report vertex non-manifoldness not already explained by edges
  const edgeBad = new Set<number>();
  for (const [a, b] of [...nonManifoldEdges, ...inconsistentEdges]) {
    edgeBad.add(a);
    edgeBad.add(b);
  }
  const pureVertexIssues = nonManifoldVertices.filter((v) => !edgeBad.has(v));
  // components via union-find over faces
  const parent = new Int32Array(topo.nf);
  for (let i = 0; i < topo.nf; i++) parent[i] = i;
  const find = (x: number): number => {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  for (let h = 0; h < topo.nh; h++) {
    const t = topo.heTwin[h];
    if (t >= 0) {
      const a = find(topo.heF[h]), b = find(topo.heF[t]);
      if (a !== b) parent[a] = b;
    }
  }
  // also union faces sharing an edge without twins (non-manifold) so components reflect connectivity
  for (let e = 0; e < topo.ne; e++) {
    if (topo.edgeFaceCount[e] < 2) continue;
    const h0 = topo.edgeHe[e];
    const f0 = topo.heF[h0];
    for (let p = topo.vOutStart[topo.edgeV[2 * e]]; p < topo.vOutStart[topo.edgeV[2 * e] + 1]; p++) {
      const h = topo.vOut[p];
      if (topo.heEdge[h] === e || topo.heEdge[topo.hePrev[h]] === e) {
        const a = find(f0), b = find(topo.heF[h]);
        if (a !== b) parent[a] = b;
      }
    }
  }
  const roots = new Set<number>();
  for (let f = 0; f < topo.nf; f++) if (topo.faceValid[f]) roots.add(find(f));
  const usedV = topo.nv - unusedVertices.length;
  let validFaces = 0;
  for (let f = 0; f < topo.nf; f++) if (topo.faceValid[f]) validFaces++;
  const eulerCharacteristic = usedV - topo.ne + validFaces;
  const loops = boundaryLoopsTopo(topo);
  if (invalidFaces.length) errors.push(`${invalidFaces.length} invalid face(s) (fewer than 3 vertices, bad or repeated indices)`);
  if (nonManifoldEdges.length) errors.push(`${nonManifoldEdges.length} non-manifold edge(s) (more than two faces)`);
  if (inconsistentEdges.length) errors.push(`${inconsistentEdges.length} edge(s) with inconsistent face orientation`);
  if (pureVertexIssues.length) errors.push(`${pureVertexIssues.length} non-manifold vertex/vertices`);
  if (unusedVertices.length) warnings.push(`${unusedVertices.length} unused vertex/vertices`);
  let nonQuads = 0;
  for (const f of qm.faces) if (f.length !== 4) nonQuads++;
  if (nonQuads) warnings.push(`${nonQuads} non-quad face(s)`);
  return {
    ok: errors.length === 0,
    errors,
    warnings,
    invalidFaces,
    nonManifoldEdges,
    inconsistentEdges,
    nonManifoldVertices,
    unusedVertices,
    boundaryEdgeCount,
    boundaryLoopCount: loops.length,
    components: roots.size,
    eulerCharacteristic,
    closed: boundaryEdgeCount === 0 && validFaces > 0,
  };
}

/** Signed volume enclosed by the given faces (fan triangulation); > 0 for outward-oriented closed meshes. */
export function signedVolume(qm: QuadMesh, faces?: number[]): number {
  let vol = 0;
  const list = faces ?? qm.faces.map((_, i) => i);
  for (const fi of list) {
    const f = qm.faces[fi];
    if (f.length < 3) continue;
    const p0 = qm.vertices[f[0]];
    for (let k = 1; k + 1 < f.length; k++) vol += dot(p0, cross(qm.vertices[f[k]], qm.vertices[f[k + 1]]));
  }
  return vol / 6;
}

export interface OrientResult {
  /** number of faces whose vertex order was reversed */
  flipped: number;
  /** false if some component is non-orientable (e.g. a Möbius strip) */
  orientable: boolean;
}

/**
 * Make face orientations consistent across manifold edges. Closed components are oriented so
 * their signed volume is positive (outward normals); open components keep the orientation of the
 * majority of their faces. Mutates qm.
 */
export function orient(qm: QuadMesh): OrientResult {
  const nf = qm.faces.length;
  const topo = buildTopology(qm);
  // edge -> faces with direction (+1 if face uses lo->hi)
  const edgeFaces: number[][] = Array.from({ length: topo.ne }, () => []);
  for (let h = 0; h < topo.nh; h++) {
    const e = topo.heEdge[h];
    const dir = topo.heV[h] === topo.edgeV[2 * e] ? 1 : -1;
    edgeFaces[e].push(topo.heF[h], dir);
  }
  const flip = new Int8Array(nf).fill(-1);
  let orientable = true;
  let flipped = 0;
  const queue: number[] = [];
  for (let seed = 0; seed < nf; seed++) {
    if (!topo.faceValid[seed] || flip[seed] !== -1) continue;
    const comp: number[] = [];
    let closedComp = true;
    flip[seed] = 0;
    queue.length = 0;
    queue.push(seed);
    while (queue.length) {
      const f = queue.pop()!;
      comp.push(f);
      for (let h = topo.faceStart[f]; h < topo.faceStart[f + 1]; h++) {
        const e = topo.heEdge[h];
        const list = edgeFaces[e];
        if (list.length !== 4) {
          if (list.length === 2) closedComp = false;
          continue; // boundary or non-manifold: no propagation
        }
        const myDir = topo.heV[h] === topo.edgeV[2 * e] ? 1 : -1;
        const g = list[0] === f && list[1] === myDir ? list[2] : list[0];
        const gDir = list[0] === f && list[1] === myDir ? list[3] : list[1];
        if (g === f) continue;
        const same = gDir === myDir ? 1 : 0;
        const want = flip[f] ^ same;
        if (flip[g] === -1) {
          flip[g] = want;
          queue.push(g);
        } else if (flip[g] !== want) orientable = false;
      }
    }
    let nFlip = 0;
    for (const f of comp) nFlip += flip[f];
    if (closedComp) {
      // orient outward by signed volume (after the tentative flips)
      let vol = 0;
      for (const f of comp) {
        const face = qm.faces[f];
        const p0 = qm.vertices[face[0]];
        let fv = 0;
        for (let k = 1; k + 1 < face.length; k++) fv += dot(p0, cross(qm.vertices[face[k]], qm.vertices[face[k + 1]]));
        vol += flip[f] ? -fv : fv;
      }
      if (vol < 0) for (const f of comp) flip[f] ^= 1;
    } else if (2 * nFlip > comp.length) {
      for (const f of comp) flip[f] ^= 1;
    }
  }
  for (let f = 0; f < nf; f++) {
    if (flip[f] === 1) {
      const face = qm.faces[f];
      qm.faces[f] = [face[0], ...face.slice(1).reverse()];
      flipped++;
    }
  }
  return { flipped, orientable };
}

// ---------------------------------------------------------------------------
// Editing operations (mutating)
// ---------------------------------------------------------------------------

function ensureFixed(qm: QuadMesh): boolean[] {
  if (!qm.fixed) qm.fixed = [];
  while (qm.fixed.length < qm.vertices.length) qm.fixed.push(false);
  return qm.fixed;
}

/** Append a vertex; returns its index. */
export function addVertex(qm: QuadMesh, p: Vec3, fixed = false): number {
  qm.vertices.push([p[0], p[1], p[2]]);
  if (qm.fixed || fixed) {
    const fx = ensureFixed(qm);
    fx[qm.vertices.length - 1] = fixed;
  }
  return qm.vertices.length - 1;
}

/**
 * Append a face (any polygon). With `autoOrient` (default) the vertex order is reversed when that
 * makes it consistent with the faces it shares edges with. Returns the new face index.
 */
export function addFace(qm: QuadMesh, verts: number[], autoOrient = true): number {
  if (verts.length < 3) throw new Error("addFace: a face needs at least 3 vertices");
  const nv = qm.vertices.length;
  for (const v of verts) if (!Number.isInteger(v) || v < 0 || v >= nv) throw new Error(`addFace: bad vertex index ${v}`);
  if (new Set(verts).size !== verts.length) throw new Error("addFace: repeated vertex");
  let face = verts.slice();
  if (autoOrient && qm.faces.length) {
    const directed = new Set<number>();
    for (const f of qm.faces) for (let i = 0; i < f.length; i++) directed.add(f[i] * nv + f[(i + 1) % f.length]);
    let same = 0, opposite = 0;
    for (let i = 0; i < face.length; i++) {
      const a = face[i], b = face[(i + 1) % face.length];
      if (directed.has(a * nv + b)) same++;
      if (directed.has(b * nv + a)) opposite++;
    }
    if (same > 0 && opposite === 0) face = [face[0], ...face.slice(1).reverse()];
  }
  qm.faces.push(face);
  return qm.faces.length - 1;
}

/** Append a quad [a, b, c, d] (auto-oriented against its neighbours). */
export function addQuad(qm: QuadMesh, quad: [number, number, number, number] | number[], autoOrient = true): number {
  if (quad.length !== 4) throw new Error("addQuad: expected 4 vertices");
  return addFace(qm, quad, autoOrient);
}

function pruneCreases(qm: QuadMesh): void {
  if (!qm.creaseEdges) return;
  const present = new Set<string>();
  for (const f of qm.faces) for (let i = 0; i < f.length; i++) present.add(edgeKey(f[i], f[(i + 1) % f.length]));
  const seen = new Set<string>();
  qm.creaseEdges = qm.creaseEdges.filter(([a, b]) => {
    const k = edgeKey(a, b);
    if (a === b || !present.has(k) || seen.has(k)) return false;
    seen.add(k);
    return true;
  });
}

/** Remove a face (later face indices shift down by one). Vertices are kept. Returns the removed face. */
export function removeFace(qm: QuadMesh, face: number): number[] {
  if (face < 0 || face >= qm.faces.length) throw new Error(`removeFace: bad face index ${face}`);
  const [removed] = qm.faces.splice(face, 1);
  pruneCreases(qm);
  return removed;
}

/** Remove several faces at once. */
export function removeFaces(qm: QuadMesh, faces: number[]): void {
  const del = new Set(faces);
  qm.faces = qm.faces.filter((_, i) => !del.has(i));
  pruneCreases(qm);
}

/** Drop vertices no face references. Returns old -> new index map (-1 for removed). */
export function removeUnusedVertices(qm: QuadMesh): Int32Array {
  const nv = qm.vertices.length;
  const used = new Uint8Array(nv);
  for (const f of qm.faces) for (const v of f) if (v >= 0 && v < nv) used[v] = 1;
  const remap = new Int32Array(nv).fill(-1);
  const verts: Vec3[] = [];
  const fixed: boolean[] = [];
  for (let v = 0; v < nv; v++) {
    if (!used[v]) continue;
    remap[v] = verts.length;
    verts.push(qm.vertices[v]);
    if (qm.fixed) fixed.push(!!qm.fixed[v]);
  }
  qm.vertices = verts;
  if (qm.fixed) qm.fixed = fixed;
  qm.faces = qm.faces.map((f) => f.map((v) => remap[v]));
  if (qm.creaseEdges)
    qm.creaseEdges = qm.creaseEdges
      .map(([a, b]) => [remap[a], remap[b]] as [number, number])
      .filter(([a, b]) => a >= 0 && b >= 0);
  return remap;
}

/** Apply a vertex substitution map to faces / creases and clean up degenerate faces. */
function applyVertexSubstitution(qm: QuadMesh, sub: (v: number) => number): void {
  const faces: number[][] = [];
  const seenFaces = new Set<string>();
  for (const f of qm.faces) {
    const g: number[] = [];
    for (const v0 of f) {
      const v = sub(v0);
      if (g.length && g[g.length - 1] === v) continue;
      g.push(v);
    }
    while (g.length > 1 && g[0] === g[g.length - 1]) g.pop();
    if (g.length < 3 || new Set(g).size !== g.length) continue;
    const key = g.slice().sort((a, b) => a - b).join(",");
    if (seenFaces.has(key)) continue;
    seenFaces.add(key);
    faces.push(g);
  }
  qm.faces = faces;
  if (qm.creaseEdges) qm.creaseEdges = qm.creaseEdges.map(([a, b]) => [sub(a), sub(b)] as [number, number]);
  pruneCreases(qm);
}

/**
 * Merge the given vertices into one (the first of the list), placed at `position` (default: their
 * average). Faces collapsing to < 3 vertices are removed. With `compact` (default true) unused
 * vertices are removed afterwards. Returns old -> new index map.
 */
export function mergeVertices(qm: QuadMesh, verts: number[], opts: { position?: Vec3; compact?: boolean } = {}): Int32Array {
  const nv = qm.vertices.length;
  const list = [...new Set(verts)].filter((v) => v >= 0 && v < nv);
  if (list.length === 0) return Int32Array.from({ length: nv }, (_, i) => i);
  const keep = list[0];
  let pos: Vec3 = [0, 0, 0];
  if (opts.position) pos = [opts.position[0], opts.position[1], opts.position[2]];
  else {
    for (const v of list) pos = add(pos, qm.vertices[v]);
    pos = scale(pos, 1 / list.length);
  }
  qm.vertices[keep] = pos;
  if (qm.fixed) {
    const fx = ensureFixed(qm);
    fx[keep] = list.some((v) => fx[v]);
  }
  const mergeSet = new Set(list);
  applyVertexSubstitution(qm, (v) => (mergeSet.has(v) ? keep : v));
  if (opts.compact === false) {
    const m = Int32Array.from({ length: nv }, (_, i) => i);
    for (const v of list) m[v] = keep;
    return m;
  }
  const remap = removeUnusedVertices(qm);
  for (const v of list) remap[v] = remap[keep];
  return remap;
}

/** Merge vertices closer than `tolerance` (spatial hashing). Returns old -> new index map. */
export function weldVertices(qm: QuadMesh, tolerance: number, compact = true): Int32Array {
  const nv = qm.vertices.length;
  const tol = Math.max(tolerance, 0);
  const rep = new Int32Array(nv);
  for (let i = 0; i < nv; i++) rep[i] = i;
  if (tol > 0) {
    const cell = tol;
    const grid = new Map<string, number[]>();
    const key = (x: number, y: number, z: number) => `${x},${y},${z}`;
    for (let i = 0; i < nv; i++) {
      const p = qm.vertices[i];
      const cx = Math.floor(p[0] / cell), cy = Math.floor(p[1] / cell), cz = Math.floor(p[2] / cell);
      let found = -1;
      for (let dx = -1; dx <= 1 && found < 0; dx++)
        for (let dy = -1; dy <= 1 && found < 0; dy++)
          for (let dz = -1; dz <= 1 && found < 0; dz++) {
            const l = grid.get(key(cx + dx, cy + dy, cz + dz));
            if (!l) continue;
            for (const j of l) {
              const q = qm.vertices[j];
              if (Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]) <= tol) {
                found = j;
                break;
              }
            }
          }
      if (found >= 0) rep[i] = found;
      else {
        const k = key(cx, cy, cz);
        const l = grid.get(k);
        if (l) l.push(i);
        else grid.set(k, [i]);
      }
    }
  }
  if (qm.fixed) {
    const fx = ensureFixed(qm);
    for (let i = 0; i < nv; i++) if (rep[i] !== i && fx[i]) fx[rep[i]] = true;
  }
  applyVertexSubstitution(qm, (v) => rep[v]);
  if (!compact) return rep;
  const remap = removeUnusedVertices(qm);
  const out = new Int32Array(nv);
  for (let i = 0; i < nv; i++) out[i] = remap[rep[i]];
  return out;
}

export interface EdgeLoopResult {
  /** indices of the inserted vertices (one per crossed edge) */
  newVertices: number[];
  /** indices of the faces appended by the split (the other halves keep the original indices) */
  newFaces: number[];
  /** the quads that were split */
  splitFaces: number[];
  /** true if the loop closed on itself */
  closed: boolean;
}

/**
 * Insert an edge loop crossing `edge` (like "insert row/column"): walks across the ring of quads
 * through the edge in both directions, splitting every crossed edge at parameter `t` (measured
 * from edge[0]'s side) and every quad of the ring into two. The walk stops at boundaries and at
 * non-quad faces (which receive the new vertex on their edge). Mutates qm.
 */
export function splitEdgeLoop(qm: QuadMesh, edge: [number, number], t = 0.5): EdgeLoopResult {
  const topo = buildTopology(qm);
  const e0 = findEdge(topo, edge[0], edge[1]);
  if (e0 < 0) throw new Error(`splitEdgeLoop: edge (${edge[0]}, ${edge[1]}) not found`);
  const P = qm.vertices;
  // crossing per edge id: a-side and b-side vertex
  const crossA = new Map<number, number>();
  const crossB = new Map<number, number>();
  const ringFaces: number[] = [];
  const ringEntry: number[] = [];
  const inRing = new Uint8Array(topo.nf);
  const register = (h: number, aIsOrigin: boolean) => {
    const e = topo.heEdge[h];
    if (crossA.has(e)) return;
    const o = topo.heV[h], d = topo.heV[topo.heNext[h]];
    crossA.set(e, aIsOrigin ? o : d);
    crossB.set(e, aIsOrigin ? d : o);
  };
  const h0 = topo.edgeHe[e0];
  const a0IsOrigin = topo.heV[h0] === edge[0];
  register(h0, a0IsOrigin);
  let closed = false;
  const walk = (hStart: number, aIsOrigin: boolean): boolean => {
    let h = hStart;
    for (let guard = 0; guard <= topo.nf; guard++) {
      const f = topo.heF[h];
      if (inRing[f]) return ringFaces.length > 0 && f === ringFaces[0] && topo.heEdge[h] === e0;
      if (topo.faceStart[f + 1] - topo.faceStart[f] !== 4) return false;
      inRing[f] = 1;
      ringFaces.push(f);
      ringEntry.push(h);
      const o = topo.heNext[topo.heNext[h]];
      const next = topo.heTwin[o];
      // a-side on the opposite edge: destination of o if the entry's origin is the a-side
      register(o, !aIsOrigin);
      if (next < 0) return false;
      h = next;
    }
    return false;
  };
  closed = walk(h0, a0IsOrigin);
  if (!closed) {
    const t0 = topo.heTwin[h0];
    if (t0 >= 0) walk(t0, !a0IsOrigin);
    else {
      // boundary start edge: nothing on the other side
    }
  }
  // create the new vertices
  const newVert = new Map<number, number>();
  const newVertices: number[] = [];
  const fixed = qm.fixed ? ensureFixed(qm) : undefined;
  for (const [e, a] of crossA) {
    const b = crossB.get(e)!;
    const idx = qm.vertices.length;
    qm.vertices.push(lerp(P[a], P[b], t));
    if (fixed) fixed.push(false);
    newVert.set(e, idx);
    newVertices.push(idx);
  }
  const crossKey = new Map<string, number>();
  for (const [e, idx] of newVert) crossKey.set(edgeKey(topo.edgeV[2 * e], topo.edgeV[2 * e + 1]), idx);
  // split ring quads
  const newFaces: number[] = [];
  for (let i = 0; i < ringFaces.length; i++) {
    const f = ringFaces[i];
    const h = ringEntry[i];
    const x = topo.heV[h];
    const y = topo.heV[topo.heNext[h]];
    const c = topo.heV[topo.heNext[topo.heNext[h]]];
    const d = topo.heV[topo.hePrev[h]];
    const m1 = newVert.get(topo.heEdge[h])!;
    const m2 = newVert.get(topo.heEdge[topo.heNext[topo.heNext[h]]])!;
    qm.faces[f] = [x, m1, m2, d];
    qm.faces.push([m1, y, c, m2]);
    newFaces.push(qm.faces.length - 1);
  }
  // insert crossing vertices into any face still containing a crossed edge (n-gon ends, conflicts)
  for (let f = 0; f < qm.faces.length; f++) {
    const face = qm.faces[f];
    let changed = false;
    const out: number[] = [];
    for (let k = 0; k < face.length; k++) {
      const a = face[k], b = face[(k + 1) % face.length];
      out.push(a);
      const m = crossKey.get(edgeKey(a, b));
      if (m !== undefined) {
        out.push(m);
        changed = true;
      }
    }
    if (changed) qm.faces[f] = out;
  }
  // creases: split crossed crease edges
  if (qm.creaseEdges) {
    const out: [number, number][] = [];
    for (const [a, b] of qm.creaseEdges) {
      const m = crossKey.get(edgeKey(a, b));
      if (m === undefined) out.push([a, b]);
      else out.push([a, m], [m, b]);
    }
    qm.creaseEdges = out;
  }
  return { newVertices, newFaces, splitFaces: ringFaces.slice(), closed };
}

export interface ExtrudeResult {
  /** pairs [originalVertex, newVertex] */
  vertexMap: [number, number][];
  newFaces: number[];
  /** requested edges that are not boundary edges (skipped) */
  skipped: [number, number][];
}

/**
 * Extrude boundary edges: each boundary edge (a, b) gets a new quad [b, a, a', b'] with
 * a' = a + offset (new vertices are shared between consecutive edges). `offset` is a vector or a
 * function of the original vertex. Mutates qm.
 */
export function extrudeBoundaryEdges(
  qm: QuadMesh,
  edges: [number, number][],
  offset: Vec3 | ((vertex: number, position: Vec3) => Vec3),
): ExtrudeResult {
  const topo = buildTopology(qm);
  const map = new Map<number, number>();
  const newFaces: number[] = [];
  const skipped: [number, number][] = [];
  const fixed = qm.fixed ? ensureFixed(qm) : undefined;
  const dup = (v: number): number => {
    let n = map.get(v);
    if (n === undefined) {
      const p = qm.vertices[v];
      const off = typeof offset === "function" ? offset(v, p) : offset;
      n = qm.vertices.length;
      qm.vertices.push(add(p, off));
      if (fixed) fixed.push(false);
      map.set(v, n);
    }
    return n;
  };
  for (const [a0, b0] of edges) {
    const e = findEdge(topo, a0, b0);
    if (e < 0 || topo.edgeFaceCount[e] !== 1) {
      skipped.push([a0, b0]);
      continue;
    }
    const h = topo.edgeHe[e];
    const a = topo.heV[h], b = topo.heV[topo.heNext[h]];
    qm.faces.push([b, a, dup(a), dup(b)]);
    newFaces.push(qm.faces.length - 1);
  }
  return { vertexMap: [...map.entries()], newFaces, skipped };
}

/** Toggle / set crease flags on edges. */
export function setCrease(qm: QuadMesh, edges: [number, number][], sharp: boolean): void {
  const cur = new Map<string, [number, number]>();
  for (const e of qm.creaseEdges ?? []) cur.set(edgeKey(e[0], e[1]), e);
  for (const [a, b] of edges) {
    const k = edgeKey(a, b);
    if (sharp) cur.set(k, [a, b]);
    else cur.delete(k);
  }
  qm.creaseEdges = [...cur.values()];
}

/** Reverse the orientation of all faces. */
export function flipFaces(qm: QuadMesh): void {
  qm.faces = qm.faces.map((f) => [f[0], ...f.slice(1).reverse()]);
}

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

/** Triangle MeshData -> QuadMesh with triangle faces (no welding). */
export function quadMeshFromTriangles(mesh: MeshData): QuadMesh {
  const n = mesh.positions.length / 3;
  const vertices: Vec3[] = new Array(n);
  for (let i = 0; i < n; i++) vertices[i] = [mesh.positions[3 * i], mesh.positions[3 * i + 1], mesh.positions[3 * i + 2]];
  const faces: number[][] = [];
  for (let t = 0; t + 2 < mesh.indices.length; t += 3) faces.push([mesh.indices[t], mesh.indices[t + 1], mesh.indices[t + 2]]);
  return { vertices, faces };
}

/** The cage polygons as a flat-shaded triangle mesh (fan triangulation) — handy for display/picking. */
export function cageToMeshData(qm: QuadMesh): MeshData {
  const positions = new Float32Array(qm.vertices.length * 3);
  qm.vertices.forEach((p, i) => positions.set(p, 3 * i));
  const idx: number[] = [];
  for (const f of qm.faces) for (let k = 1; k + 1 < f.length; k++) idx.push(f[0], f[k], f[k + 1]);
  return { positions, indices: Uint32Array.from(idx) };
}

/** Line segments of the cage edges (2 points per edge, packed xyz) for wireframe display. */
export function cageWireframe(qm: QuadMesh): Float32Array {
  const edges = cageEdges(qm);
  const out = new Float32Array(edges.length * 6);
  edges.forEach(([a, b], i) => {
    out.set(qm.vertices[a], 6 * i);
    out.set(qm.vertices[b], 6 * i + 3);
  });
  return out;
}

/** Newell normal of a polygon (not normalised; length = 2 * area). */
export function polygonNormal(points: Vec3[]): Vec3 {
  let x = 0, y = 0, z = 0;
  for (let i = 0; i < points.length; i++) {
    const a = points[i], b = points[(i + 1) % points.length];
    x += (a[1] - b[1]) * (a[2] + b[2]);
    y += (a[2] - b[2]) * (a[0] + b[0]);
    z += (a[0] - b[0]) * (a[1] + b[1]);
  }
  return [x, y, z];
}

export function faceNormal(qm: QuadMesh, f: number): Vec3 {
  return normalize(polygonNormal(qm.faces[f].map((v) => qm.vertices[v])));
}

export function faceCentroid(qm: QuadMesh, f: number): Vec3 {
  const face = qm.faces[f];
  let c: Vec3 = [0, 0, 0];
  for (const v of face) c = add(c, qm.vertices[v]);
  return scale(c, 1 / Math.max(1, face.length));
}

// ---------------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------------

export interface GridCageOptions {
  origin?: Vec3;
  /** full extent vector along u (default [100, 0, 0]) */
  uAxis?: Vec3;
  /** full extent vector along v (default [0, 100, 0]) */
  vAxis?: Vec3;
}

/** Planar nu x nv grid of quads; normal = uAxis x vAxis. Vertex (i, j) has index j * (nu + 1) + i. */
export function createGridCage(nu: number, nv: number, opts: GridCageOptions = {}): QuadMesh {
  const o = opts.origin ?? [0, 0, 0];
  const U = opts.uAxis ?? [100, 0, 0];
  const V = opts.vAxis ?? [0, 100, 0];
  const vertices: Vec3[] = [];
  for (let j = 0; j <= nv; j++)
    for (let i = 0; i <= nu; i++) vertices.push(add(o, add(scale(U, i / nu), scale(V, j / nv))));
  const faces: number[][] = [];
  const id = (i: number, j: number) => j * (nu + 1) + i;
  for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) faces.push([id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)]);
  return { vertices, faces };
}

/** Closed box cage (surface lattice of an axis-aligned box) with outward CCW quads. */
export function createBoxCage(min: Vec3, max: Vec3, divisions: number | [number, number, number] = 1): QuadMesh {
  const [nx, ny, nz] = typeof divisions === "number" ? [divisions, divisions, divisions] : divisions;
  const vertices: Vec3[] = [];
  const index = new Map<string, number>();
  const vid = (i: number, j: number, k: number): number => {
    const key = `${i},${j},${k}`;
    let v = index.get(key);
    if (v === undefined) {
      v = vertices.length;
      index.set(key, v);
      vertices.push([
        min[0] + ((max[0] - min[0]) * i) / nx,
        min[1] + ((max[1] - min[1]) * j) / ny,
        min[2] + ((max[2] - min[2]) * k) / nz,
      ]);
    }
    return v;
  };
  const faces: number[][] = [];
  // x = 0 (normal -x) and x = nx (normal +x)
  for (let j = 0; j < ny; j++)
    for (let k = 0; k < nz; k++) {
      faces.push([vid(0, j, k), vid(0, j, k + 1), vid(0, j + 1, k + 1), vid(0, j + 1, k)]);
      faces.push([vid(nx, j, k), vid(nx, j + 1, k), vid(nx, j + 1, k + 1), vid(nx, j, k + 1)]);
    }
  for (let i = 0; i < nx; i++)
    for (let k = 0; k < nz; k++) {
      faces.push([vid(i, 0, k), vid(i + 1, 0, k), vid(i + 1, 0, k + 1), vid(i, 0, k + 1)]);
      faces.push([vid(i, ny, k), vid(i, ny, k + 1), vid(i + 1, ny, k + 1), vid(i + 1, ny, k)]);
    }
  for (let i = 0; i < nx; i++)
    for (let j = 0; j < ny; j++) {
      faces.push([vid(i, j, 0), vid(i, j + 1, 0), vid(i + 1, j + 1, 0), vid(i + 1, j, 0)]);
      faces.push([vid(i, j, nz), vid(i + 1, j, nz), vid(i + 1, j + 1, nz), vid(i, j + 1, nz)]);
    }
  return { vertices, faces };
}

/** Open tube along +z: nAround x nAlong quads, outward normals. Vertex (i, j) = j * nAround + i. */
export function createCylinderCage(radius: number, height: number, nAround: number, nAlong: number, center: Vec3 = [0, 0, 0]): QuadMesh {
  const vertices: Vec3[] = [];
  for (let j = 0; j <= nAlong; j++)
    for (let i = 0; i < nAround; i++) {
      const a = (2 * Math.PI * i) / nAround;
      vertices.push([center[0] + radius * Math.cos(a), center[1] + radius * Math.sin(a), center[2] + (height * j) / nAlong]);
    }
  const faces: number[][] = [];
  const id = (i: number, j: number) => j * nAround + (i % nAround);
  for (let j = 0; j < nAlong; j++) for (let i = 0; i < nAround; i++) faces.push([id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)]);
  return { vertices, faces };
}

/** Torus cage around +z (all vertices valence 4). Vertex (i, j) = j * nU + i. */
export function createTorusCage(R: number, r: number, nU: number, nV: number, center: Vec3 = [0, 0, 0]): QuadMesh {
  const vertices: Vec3[] = [];
  for (let j = 0; j < nV; j++)
    for (let i = 0; i < nU; i++) {
      const u = (2 * Math.PI * i) / nU, v = (2 * Math.PI * j) / nV;
      const rho = R + r * Math.cos(v);
      vertices.push([center[0] + rho * Math.cos(u), center[1] + rho * Math.sin(u), center[2] + r * Math.sin(v)]);
    }
  const faces: number[][] = [];
  const id = (i: number, j: number) => (j % nV) * nU + (i % nU);
  for (let j = 0; j < nV; j++) for (let i = 0; i < nU; i++) faces.push([id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)]);
  return { vertices, faces };
}

/** Bounding-box diagonal length of the cage vertices. */
export function cageSize(qm: QuadMesh): number {
  if (qm.vertices.length === 0) return 0;
  const lo: Vec3 = [Infinity, Infinity, Infinity], hi: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (const p of qm.vertices)
    for (let c = 0; c < 3; c++) {
      if (p[c] < lo[c]) lo[c] = p[c];
      if (p[c] > hi[c]) hi[c] = p[c];
    }
  return length(sub(hi, lo));
}
