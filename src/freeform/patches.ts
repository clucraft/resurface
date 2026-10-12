/**
 * Export of a Catmull–Clark cage to bicubic NURBS patches.
 *
 * One degree 3x3 Bézier patch (4x4 control points, knots [0,0,0,0,1,1,1,1]) per quad face (cages
 * with triangles / n-gons are subdivided once first). The Bézier control points are assembled
 * from SHARED per-vertex / per-edge / per-face-corner quantities, so neighbouring patches have
 * bit-identical boundary curves (watertight):
 *
 *  - corner point of vertex v: its limit position
 *  - interior point of face-corner (v, f):  (n v + 2 a + 2 b + d) / (n + 5), with a, b the face
 *    neighbours of v and d the diagonal vertex (Loop & Schaefer 2008, "ACC" geometry patches)
 *  - edge point near v on edge (v, w): mean of the two interior points adjacent to the edge, or
 *    the cubic B-spline curve point (2 v + w) / 3 on boundary / crease edges
 *
 * For valence-4 vertices (and regular boundary / crease / corner configurations, where the
 * effective valence n is 4) these masks are exactly the uniform bicubic B-spline -> Bézier
 * conversion, so patches of regular faces reproduce the limit surface exactly and join C2.
 * Around extraordinary vertices the patches start from Loop–Schaefer ACC geometry patches; with
 * option `g1Seams` (default on) the control points next to each extraordinary vertex are then
 * adjusted to satisfy the sufficient G1 conditions d_A + d_B = alpha(t) c'(t) row by row (exact
 * limit tangent plane at the vertex, near-G1 along the seams: measured < 0.4° vs 4–8° for plain
 * ACC on test cages), still without touching any control point of an exact (regular) patch.
 */
import { solveLinear } from "../core/linalg";
import type { BSplineSurfaceDef, Vec3 } from "../core/types";
import { add, cross, dot, length, normalize, scale, sub } from "../core/vec";
import { smoothLimitTangents, subdivideWithMap, vertexSectorNormals } from "./catmullClark";
import {
  buildTopology,
  EdgeSharpness,
  isAllQuads,
  ringSectors,
  valenceOf,
  VertexKind,
  vertexRing,
  type QuadMesh,
  type RingSector,
  type Topology,
  type VertexRing,
} from "./quadMesh";

export interface PatchOptions {
  /**
   * Near-G1 seams around extraordinary vertices (default true): edge control points on the exact
   * limit tangent directions and interior points satisfying the G1 conditions of the first Bézier
   * rows (see g1Correction). false = plain Loop–Schaefer ACC geometry patches (C0 at seams near
   * extraordinary vertices). Regular (exact) patches are identical in both modes.
   */
  g1Seams?: boolean;
}

export interface PatchSet {
  patches: BSplineSurfaceDef[];
  /** the all-quad mesh the patches were built on (the cage itself, or its first CC refinement) */
  mesh: QuadMesh;
  /** face of `mesh` per patch */
  faceOfPatch: Int32Array;
  /** face of the input cage per patch */
  cageFaceOfPatch: Int32Array;
  /** 1 if the patch is exactly the limit surface (regular configuration) */
  exact: Uint8Array;
}

const NONMANIFOLD = 4;

interface CornerInfo {
  kind: number;
  ring: VertexRing;
  sectors: RingSector[];
  /** sector index per ring face */
  faceSector: Int32Array;
}

function cornerInfo(topo: Topology, v: number): CornerInfo {
  const ring = vertexRing(topo, v);
  let kind: number = topo.vKind[v];
  if (!ring.manifold) kind = NONMANIFOLD;
  const sectors = kind === NONMANIFOLD ? [] : ringSectors(topo, ring);
  const faceSector = new Int32Array(ring.faces.length);
  sectors.forEach((s, si) => {
    for (const fi of s.faceIdx) faceSector[fi] = si;
  });
  return { kind, ring, sectors, faceSector };
}

/** Sector (and position inside it) of the face corner given by half-edge h. */
function sectorOf(info: CornerInfo, h: number): { sector: RingSector | null; ringFace: number } {
  const i = info.ring.faceHe.indexOf(h);
  if (i < 0 || info.sectors.length === 0) return { sector: null, ringFace: i };
  return { sector: info.sectors[info.faceSector[i]], ringFace: i };
}

/** Effective valence used in the interior-point mask, and whether the corner is "regular" for that face. */
function effectiveValence(topo: Topology, P: Vec3[], info: CornerInfo, h: number): { n: number; regular: boolean } {
  const v = topo.heV[h];
  const kind = info.kind;
  if (kind === NONMANIFOLD) return { n: 4, regular: false };
  if (kind === VertexKind.Smooth || kind === VertexKind.Dart) {
    const n = valenceOf(topo, v);
    return { n, regular: kind === VertexKind.Smooth && n === 4 };
  }
  // Boundary / crease / corner sectors: the boundary-curve rules give subdominant eigenvalue 1/2
  // for any sector size, so the regular mask (n = 4) is used; it is exact for regular sectors
  // (crease: 2 faces, corner: 1 face) and measured best for irregular ones (vs n = 2k, k + 2).
  const { sector } = sectorOf(info, h);
  const k = sector ? sector.faceIdx.length : 1;
  if (kind === VertexKind.Crease) return { n: 4, regular: k === 2 };
  return { n: 4, regular: k === 1 };
}

function bezierSurface(B: Vec3[]): BSplineSurfaceDef {
  return {
    degreeU: 3,
    degreeV: 3,
    knotsU: [0, 0, 0, 0, 1, 1, 1, 1],
    knotsV: [0, 0, 0, 0, 1, 1, 1, 1],
    nu: 4,
    nv: 4,
    controlPoints: B,
  };
}

interface PatchData {
  topo: Topology;
  P: Vec3[];
  info: CornerInfo[];
  corner: Vec3[];
  interior: Vec3[];
  edgePt: Vec3[];
  regularCorner: Uint8Array;
}

function computePatchData(mesh: QuadMesh, opts: PatchOptions): PatchData {
  const topo = buildTopology(mesh);
  const P = mesh.vertices;
  const nv = topo.nv;
  const project = opts.g1Seams !== false;
  const info: CornerInfo[] = new Array(nv);
  const corner: Vec3[] = new Array(nv);
  for (let v = 0; v < nv; v++) {
    const ci = cornerInfo(topo, v);
    info[v] = ci;
    const V = P[v];
    if (ci.ring.faces.length === 0 || ci.kind === NONMANIFOLD || ci.kind === VertexKind.Corner) corner[v] = [V[0], V[1], V[2]];
    else if (ci.kind === VertexKind.Crease) {
      const sn: number[] = [];
      for (let i = 0; i < ci.ring.edges.length; i++)
        if (topo.edgeSharp[ci.ring.edges[i]] !== EdgeSharpness.None) sn.push(ci.ring.nbr[i]);
      if (sn.length === 2) corner[v] = scale(add(add(P[sn[0]], scale(V, 4)), P[sn[1]]), 1 / 6);
      else corner[v] = [V[0], V[1], V[2]];
    } else {
      const n = valenceOf(topo, v);
      let s = scale(V, n * n);
      for (const nb of ci.ring.nbr) s = add(s, scale(P[nb], 4));
      for (const h of ci.ring.faceHe) s = add(s, P[topo.heV[topo.heNext[topo.heNext[h]]]]);
      corner[v] = scale(s, 1 / (n * (n + 5)));
    }
  }
  const interior: Vec3[] = new Array(topo.nh);
  const regularCorner = new Uint8Array(topo.nh);
  for (let h = 0; h < topo.nh; h++) {
    const v = topo.heV[h];
    const { n, regular } = effectiveValence(topo, P, info[v], h);
    regularCorner[h] = regular ? 1 : 0;
    const a = P[topo.heV[topo.heNext[h]]];
    const d = P[topo.heV[topo.heNext[topo.heNext[h]]]];
    const b = P[topo.heV[topo.hePrev[h]]];
    const V = P[v];
    const w = 1 / (n + 5);
    interior[h] = [
      (n * V[0] + 2 * a[0] + 2 * b[0] + d[0]) * w,
      (n * V[1] + 2 * a[1] + 2 * b[1] + d[1]) * w,
      (n * V[2] + 2 * a[2] + 2 * b[2] + d[2]) * w,
    ];
  }
  const edgePt: Vec3[] = new Array(2 * topo.ne);
  for (let e = 0; e < topo.ne; e++) {
    for (let s = 0; s < 2; s++) {
      const v = topo.edgeV[2 * e + s];
      const w = topo.edgeV[2 * e + 1 - s];
      const kind = info[v].kind;
      const V = P[v], W = P[w];
      const curvePoint: Vec3 = [(2 * V[0] + W[0]) / 3, (2 * V[1] + W[1]) / 3, (2 * V[2] + W[2]) / 3];
      const h0 = topo.edgeHe[e];
      const sharpAtV = topo.edgeSharp[e] !== EdgeSharpness.None && (kind === VertexKind.Crease || kind === VertexKind.Corner || kind === NONMANIFOLD);
      if (sharpAtV || topo.heTwin[h0] < 0) {
        edgePt[2 * e + s] = curvePoint;
        continue;
      }
      const h1 = topo.heV[h0] === v ? h0 : topo.heTwin[h0];
      const h2 = topo.heNext[topo.heTwin[h1]];
      const p1 = interior[h1], p2 = interior[h2];
      edgePt[2 * e + s] = [(p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5, (p1[2] + p2[2]) * 0.5];
    }
  }
  if (project) g1Correction(topo, P, info, corner, interior, edgePt);
  return { topo, P, info, corner, interior, edgePt, regularCorner };
}

/**
 * Improve tangent continuity around irregular vertices without touching any control point of an
 * exact (regular) patch. Uses the sufficient G1 condition across an edge with shared boundary
 * curve c(t):  d_A(t) + d_B(t) = alpha(t) c'(t)  (d_X = cross-boundary derivative into patch X,
 * alpha linear from alpha0 at v to alpha1 at w), imposed row by row on the Bézier nets.
 *
 * "Jobs" are smooth extraordinary vertices / darts (full ring, angles 2πj/n) and crease or
 * boundary sectors with >= 3 faces (half ring, angles πj/k, the crease edges keep their exact
 * B-spline curve points). For each job:
 *  - row 0: edge points are placed on the limit tangent directions
 *      e_j = p + cos(θ_j) U + sin(θ_j) W
 *    (smooth: U, W = lambda * limit tangents t1, t2 from the tangent masks, lambda fitted to the
 *    ACC edge points; crease: U = crease curve tangent fixed by the curve, W = fitted cross
 *    tangent). Then e_{j-1} + e_{j+1} - 2p = alpha0 (e_j - p), alpha0 = 2cos(Δθ): all patches at the
 *    vertex share its limit tangent plane.
 *  - row 2 at the far end w of each edge (when w is not itself a job, i.e. alpha1 = 0): w's edge
 *    point is slid along the edge by alpha0 c2 / (alpha0 - 6), which satisfies that row exactly.
 *  - row 1: the interior points x at the vertex solve, in the least-squares sense with a tiny
 *    regularisation towards ACC,  x_a + x_b = 2 e_j + (2 alpha0 c1 + alpha1 c0) / 3  per edge
 *    (exact for odd valence; even valence is the classical vertex-enclosure obstruction).
 */
function g1Correction(topo: Topology, P: Vec3[], info: CornerInfo[], corner: Vec3[], interior: Vec3[], edgePt: Vec3[]): void {
  const nv = topo.nv;
  const edgeSlot = (e: number, v: number) => 2 * e + (topo.edgeV[2 * e] === v ? 0 : 1);
  const alphaSlot = new Float64Array(2 * topo.ne).fill(NaN);
  interface Job {
    v: number;
    /** interior-point half-edges (one per face of the job) */
    faces: number[];
    /** edges with the positions (in `faces`) of their two adjacent faces and their alpha0 */
    edges: { e: number; w: number; a: number; b: number; alpha0: number }[];
  }
  const jobs: Job[] = [];
  const isQuadFace = (h: number) => topo.faceStart[topo.heF[h] + 1] - topo.faceStart[topo.heF[h]] === 4;
  // phase 1: tangent-direction edge points
  for (let v = 0; v < nv; v++) {
    const ci = info[v];
    const ring = ci.ring;
    if (!ring.manifold || ring.faces.length === 0 || !ring.faceHe.every(isQuadFace)) continue;
    const p = corner[v];
    if ((ci.kind === VertexKind.Smooth || ci.kind === VertexKind.Dart) && ring.closed) {
      const n = ring.faces.length;
      if ((ci.kind === VertexKind.Smooth && n === 4) || n < 3) continue;
      const [t1, t2] = smoothLimitTangents(topo, P, ring);
      const T: Vec3[] = [];
      let num = 0, den = 0;
      for (let j = 0; j < n; j++) {
        const th = (2 * Math.PI * j) / n;
        const Tj = add(scale(t1, Math.cos(th)), scale(t2, Math.sin(th)));
        T.push(Tj);
        num += dot(sub(edgePt[edgeSlot(ring.edges[j], v)], p), Tj);
        den += dot(Tj, Tj);
      }
      if (!(den > 0) || !(num > 0)) continue;
      // 1.1: empirically minimises the deviation from the limit surface (scan of 0.85..1.5 on test cages)
      const lambda = (num / den) * 1.1;
      const alpha0 = 2 * Math.cos((2 * Math.PI) / n);
      const job: Job = { v, faces: ring.faceHe.slice(), edges: [] };
      for (let j = 0; j < n; j++) {
        const slot = edgeSlot(ring.edges[j], v);
        edgePt[slot] = add(p, scale(T[j], lambda));
        alphaSlot[slot] = alpha0;
        job.edges.push({ e: ring.edges[j], w: ring.nbr[j], a: (j + n - 1) % n, b: j, alpha0 });
      }
      jobs.push(job);
    } else if (ci.kind === VertexKind.Crease) {
      let normals: Vec3[] | null = null;
      ci.sectors.forEach((sec, si) => {
        const k = sec.faceIdx.length;
        if (k < 3) return;
        normals = normals ?? vertexSectorNormals(topo, P, v).normals;
        const N = normals[si];
        if (!N) return;
        // keep ACC's tangential positions, project into the sector tangent plane
        const slots = sec.nbrIdx.map((ri) => edgeSlot(ring.edges[ri], v));
        for (let j = 1; j < k; j++) {
          const ep = edgePt[slots[j]];
          edgePt[slots[j]] = sub(ep, scale(N, dot(sub(ep, p), N)));
        }
        const job: Job = { v, faces: sec.faceIdx.map((fi) => ring.faceHe[fi]), edges: [] };
        for (let j = 1; j < k; j++) {
          // per-edge alpha0: best fit of e_{j-1} + e_{j+1} - 2p = alpha0 (e_j - p)
          const c0 = sub(edgePt[slots[j]], p);
          const s0 = add(sub(edgePt[slots[j - 1]], p), sub(edgePt[slots[j + 1]], p));
          const cc = dot(c0, c0);
          const alpha0 = cc > 0 ? Math.max(-1.9, Math.min(1.9, dot(s0, c0) / cc)) : 0;
          alphaSlot[slots[j]] = alpha0;
          job.edges.push({ e: ring.edges[sec.nbrIdx[j]], w: ring.nbr[sec.nbrIdx[j]], a: j - 1, b: j, alpha0 });
        }
        jobs.push(job);
      });
    }
  }
  // phase 2: row 2 at far ends that are not jobs themselves (alpha1 = 0)
  for (const job of jobs) {
    for (const { e, w, alpha0 } of job.edges) {
      const slot = edgeSlot(e, w);
      if (!Number.isNaN(alphaSlot[slot]) || topo.edgeSharp[e] !== EdgeSharpness.None || topo.edgeFaceCount[e] !== 2) continue;
      const kw = info[w].kind;
      if (kw === NONMANIFOLD || kw === VertexKind.Corner) continue;
      const c2 = sub(corner[w], edgePt[slot]);
      edgePt[slot] = add(edgePt[slot], scale(c2, alpha0 / (alpha0 - 6)));
    }
  }
  // phase 3: interior points (row 1)
  for (const job of jobs) {
    const m = job.faces.length;
    const p = corner[job.v];
    const AtA = new Float64Array(m * m);
    const rhs = [new Float64Array(m), new Float64Array(m), new Float64Array(m)];
    for (const { e, w, a, b, alpha0 } of job.edges) {
      const ej = edgePt[edgeSlot(e, job.v)], ew = edgePt[edgeSlot(e, w)];
      const c0 = sub(ej, p), c1 = sub(ew, ej);
      const aw = alphaSlot[edgeSlot(e, w)];
      const a1 = Number.isNaN(aw) ? 0 : -aw;
      AtA[a * m + a] += 1;
      AtA[b * m + b] += 1;
      AtA[a * m + b] += 1;
      AtA[b * m + a] += 1;
      for (let k = 0; k < 3; k++) {
        const r = 2 * ej[k] + (2 * alpha0 * c1[k] + a1 * c0[k]) / 3;
        rhs[k][a] += r;
        rhs[k][b] += r;
      }
    }
    const mu = 1e-6;
    for (let i = 0; i < m; i++) {
      AtA[i * m + i] += mu;
      const x0 = interior[job.faces[i]];
      for (let k = 0; k < 3; k++) rhs[k][i] += mu * x0[k];
    }
    const sol = rhs.map((b) => solveLinear(AtA, b, m));
    if (sol.some((x) => !x)) continue;
    for (let i = 0; i < m; i++) {
      const x: Vec3 = [sol[0]![i], sol[1]![i], sol[2]![i]];
      if (x.every(Number.isFinite)) interior[job.faces[i]] = x;
    }
  }
}

function edgePoint(d: PatchData, e: number, v: number): Vec3 {
  return d.edgePt[2 * e + (d.topo.edgeV[2 * e] === v ? 0 : 1)];
}

/** Bézier control net (16 points, index i*4+j, i along u) of the face whose corner 0 is half-edge h0. */
function facePatch(d: PatchData, h0: number): Vec3[] {
  const { topo } = d;
  const h1 = topo.heNext[h0], h2 = topo.heNext[h1], h3 = topo.heNext[h2];
  const q0 = topo.heV[h0], q1 = topo.heV[h1], q2 = topo.heV[h2], q3 = topo.heV[h3];
  const B: Vec3[] = new Array(16);
  const set = (i: number, j: number, p: Vec3) => (B[i * 4 + j] = [p[0], p[1], p[2]]);
  set(0, 0, d.corner[q0]);
  set(3, 0, d.corner[q1]);
  set(3, 3, d.corner[q2]);
  set(0, 3, d.corner[q3]);
  const e0 = topo.heEdge[h0], e1 = topo.heEdge[h1], e2 = topo.heEdge[h2], e3 = topo.heEdge[h3];
  set(1, 0, edgePoint(d, e0, q0));
  set(2, 0, edgePoint(d, e0, q1));
  set(3, 1, edgePoint(d, e1, q1));
  set(3, 2, edgePoint(d, e1, q2));
  set(2, 3, edgePoint(d, e2, q2));
  set(1, 3, edgePoint(d, e2, q3));
  set(0, 2, edgePoint(d, e3, q3));
  set(0, 1, edgePoint(d, e3, q0));
  set(1, 1, d.interior[h0]);
  set(2, 1, d.interior[h1]);
  set(2, 2, d.interior[h2]);
  set(1, 2, d.interior[h3]);
  return B;
}

function quadMeshForPatches(qm: QuadMesh): { mesh: QuadMesh; cageFace: Int32Array } {
  if (isAllQuads(qm)) return { mesh: qm, cageFace: Int32Array.from({ length: qm.faces.length }, (_, i) => i) };
  const r = subdivideWithMap(qm, false);
  return { mesh: r.mesh, cageFace: Int32Array.from(r.refinement.faceParent) };
}

/** Bicubic patches with bookkeeping (see PatchSet). */
export function toBicubicPatchesEx(qm: QuadMesh, opts: PatchOptions = {}): PatchSet {
  const { mesh, cageFace } = quadMeshForPatches(qm);
  const d = computePatchData(mesh, opts);
  const { topo } = d;
  const patches: BSplineSurfaceDef[] = [];
  const faceOfPatch: number[] = [];
  const cageFaceOfPatch: number[] = [];
  const exact: number[] = [];
  for (let f = 0; f < topo.nf; f++) {
    if (!topo.faceValid[f] || topo.faceStart[f + 1] - topo.faceStart[f] !== 4) continue;
    const h0 = topo.faceStart[f];
    patches.push(bezierSurface(facePatch(d, h0)));
    faceOfPatch.push(f);
    cageFaceOfPatch.push(cageFace[f]);
    let ex = 1;
    for (let k = 0; k < 4; k++) if (!d.regularCorner[h0 + k]) ex = 0;
    exact.push(ex);
  }
  return {
    patches,
    mesh,
    faceOfPatch: Int32Array.from(faceOfPatch),
    cageFaceOfPatch: Int32Array.from(cageFaceOfPatch),
    exact: Uint8Array.from(exact),
  };
}

/** One bicubic Bézier patch (as a clamped B-spline surface) per quad face of the cage. */
export function toBicubicPatches(qm: QuadMesh, opts: PatchOptions = {}): BSplineSurfaceDef[] {
  return toBicubicPatchesEx(qm, opts).patches;
}

// ---------------------------------------------------------------------------
// Uniform B-spline control nets of regular faces and region merging
// ---------------------------------------------------------------------------

type G = (Vec3 | null)[];

/**
 * 4x4 uniform bicubic B-spline control net (index i*4+j, i along u) reproducing the limit surface
 * of a regular face (frame: corner 0 = origin of half-edge h0). Missing control points beyond
 * boundaries / creases are linear extrapolations ("ghost" points). Returns null if the face is
 * not regular.
 */
function faceSplineGrid(d: PatchData, h0: number): Vec3[] | null {
  const { topo, P } = d;
  const grid: G = new Array(16).fill(null);
  const hs = [h0, topo.heNext[h0], topo.heNext[topo.heNext[h0]], topo.hePrev[h0]];
  const pos: [number, number][] = [[1, 1], [2, 1], [2, 2], [1, 2]];
  const dNext: [number, number][] = [[1, 0], [0, 1], [-1, 0], [0, -1]];
  const dPrev: [number, number][] = [[0, 1], [-1, 0], [0, -1], [1, 0]];
  const put = (i: number, j: number, p: Vec3) => {
    if (i < 0 || i > 3 || j < 0 || j > 3) return;
    if (!grid[i * 4 + j]) grid[i * 4 + j] = p;
  };
  for (let c = 0; c < 4; c++) {
    const h = hs[c];
    if (!d.regularCorner[h]) return null;
    const v = topo.heV[h];
    const info = d.info[v];
    const dir = (o: number): [number, number] => {
      const m = ((o % 4) + 4) % 4;
      const a = dNext[c], b = dPrev[c];
      return m === 0 ? a : m === 1 ? b : m === 2 ? [-a[0], -a[1]] : [-b[0], -b[1]];
    };
    const [pi, pj] = pos[c];
    put(pi, pj, P[v]);
    const ring = info.ring;
    const i0 = ring.faceHe.indexOf(h);
    if (i0 < 0) return null;
    const diagOf = (fi: number) => P[topo.heV[topo.heNext[topo.heNext[ring.faceHe[fi]]]]];
    if (info.kind === VertexKind.Smooth) {
      for (let o = 0; o < 4; o++) {
        const dn = dir(o), dd = dir(o + 1);
        put(pi + dn[0], pj + dn[1], P[ring.nbr[(i0 + o) % 4]]);
        put(pi + dn[0] + dd[0], pj + dn[1] + dd[1], diagOf((i0 + o) % 4));
      }
    } else {
      const sec = info.sectors[info.faceSector[i0]];
      const m = sec.faceIdx.indexOf(i0);
      for (let j = 0; j < sec.nbrIdx.length; j++) {
        const dn = dir(j - m);
        put(pi + dn[0], pj + dn[1], P[ring.nbr[sec.nbrIdx[j]]]);
      }
      for (let j = 0; j < sec.faceIdx.length; j++) {
        const dn = dir(j - m), dd = dir(j - m + 1);
        put(pi + dn[0] + dd[0], pj + dn[1] + dd[1], diagOf(sec.faceIdx[j]));
      }
    }
  }
  const at = (i: number, j: number) => grid[i * 4 + j];
  const extra = (a: Vec3, b: Vec3): Vec3 => [2 * a[0] - b[0], 2 * a[1] - b[1], 2 * a[2] - b[2]];
  for (let j = 0; j < 4; j++) {
    if (!at(0, j) && at(1, j) && at(2, j)) grid[j] = extra(at(1, j)!, at(2, j)!);
    if (!at(3, j) && at(1, j) && at(2, j)) grid[12 + j] = extra(at(2, j)!, at(1, j)!);
  }
  for (let i = 0; i < 4; i++) {
    if (!at(i, 0) && at(i, 1) && at(i, 2)) grid[i * 4] = extra(at(i, 1)!, at(i, 2)!);
    if (!at(i, 3) && at(i, 1) && at(i, 2)) grid[i * 4 + 3] = extra(at(i, 2)!, at(i, 1)!);
  }
  for (const g of grid) if (!g) return null;
  return grid as Vec3[];
}

/** Uniform B-spline 4x4 net -> Bézier 4x4 net (both index i*4+j). */
export function bsplineGridToBezier(G: Vec3[]): Vec3[] {
  const M = [
    [1 / 6, 4 / 6, 1 / 6, 0],
    [0, 4 / 6, 2 / 6, 0],
    [0, 2 / 6, 4 / 6, 0],
    [0, 1 / 6, 4 / 6, 1 / 6],
  ];
  const out: Vec3[] = [];
  for (let i = 0; i < 4; i++)
    for (let j = 0; j < 4; j++) {
      const p: Vec3 = [0, 0, 0];
      for (let a = 0; a < 4; a++) {
        if (M[i][a] === 0) continue;
        for (let b = 0; b < 4; b++) {
          const w = M[i][a] * M[j][b];
          if (w === 0) continue;
          const g = G[a * 4 + b];
          p[0] += w * g[0];
          p[1] += w * g[1];
          p[2] += w * g[2];
        }
      }
      out.push(p);
    }
  return out;
}

/** B-spline control nets of all regular faces (null for irregular faces); for diagnostics / tests. */
export function regularFaceSplineGrids(qm: QuadMesh): (Vec3[] | null)[] {
  if (!isAllQuads(qm)) throw new Error("regularFaceSplineGrids: all-quad mesh required");
  const d = computePatchData(qm, {});
  return qm.faces.map((_, f) => (d.topo.faceValid[f] ? faceSplineGrid(d, d.topo.faceStart[f]) : null));
}

/** Insert knot u once into a curve given by knots / control points (Boehm). */
function insertKnot(knots: number[], ctrl: Vec3[], p: number, u: number): { knots: number[]; ctrl: Vec3[] } {
  const n = ctrl.length;
  let k = -1;
  for (let i = 0; i < knots.length - 1; i++) if (knots[i] <= u && u < knots[i + 1]) k = i;
  // at the end of the domain (u == knots[n]) use the last non-empty span: same result, valid indices
  if (k > n - 1) k = n - 1;
  if (k < p) throw new Error("insertKnot: parameter outside the knot vector");
  const Q: Vec3[] = new Array(n + 1);
  for (let i = 0; i <= k - p; i++) Q[i] = ctrl[i];
  for (let i = k - p + 1; i <= k; i++) {
    const a = (u - knots[i]) / (knots[i + p] - knots[i]);
    const A = ctrl[i - 1], B = ctrl[i];
    Q[i] = [(1 - a) * A[0] + a * B[0], (1 - a) * A[1] + a * B[1], (1 - a) * A[2] + a * B[2]];
  }
  for (let i = k + 1; i <= n; i++) Q[i] = ctrl[i - 1];
  return { knots: [...knots.slice(0, k + 1), u, ...knots.slice(k + 1)], ctrl: Q };
}

/**
 * Uniform cubic B-spline with m >= 4 control points (domain [0, m-3]) -> clamped representation
 * with knots [0,0,0,0,1,..,a-1,a,a,a,a] (a = m - 3) and m control points.
 */
export function clampUniformCubic(ctrl: Vec3[]): Vec3[] {
  const m = ctrl.length;
  const a = m - 3;
  let knots: number[] = [];
  for (let i = 0; i < m + 4; i++) knots.push(i - 3);
  let Q = ctrl;
  for (const u of [0, 0, a, a]) {
    const r = insertKnot(knots, Q, 3, u);
    knots = r.knots;
    Q = r.ctrl;
  }
  // after insertion knot 0 sits at indices 3..5; the relevant control points start at index 2
  return Q.slice(2, 2 + m);
}

export interface MergedSurfaces {
  /** merged B-spline surfaces followed by single Bézier patches for the remaining faces */
  surfaces: BSplineSurfaceDef[];
  /** faces of `mesh` covered by each surface */
  faceGroups: number[][];
  /** the all-quad mesh used (cage or its first CC refinement) */
  mesh: QuadMesh;
  /** number of merged (multi-face) surfaces at the start of `surfaces` */
  mergedCount: number;
}

/**
 * Combine rectangular grids of regular faces into single uniform-knot bicubic B-spline surfaces
 * (C2 inside, exact limit surface); remaining faces become single patches as in toBicubicPatches.
 * Regions never cross creases/boundaries.
 */
export function mergeRegularRegions(qm: QuadMesh, opts: PatchOptions & { maxSpan?: number; minFaces?: number } = {}): MergedSurfaces {
  const { mesh } = quadMeshForPatches(qm);
  const d = computePatchData(mesh, opts);
  const { topo } = d;
  const maxSpan = Math.max(1, opts.maxSpan ?? 64);
  const minFaces = Math.max(2, opts.minFaces ?? 2);
  const nf = topo.nf;
  const regular = new Uint8Array(nf);
  for (let f = 0; f < nf; f++) {
    if (!topo.faceValid[f] || topo.faceStart[f + 1] - topo.faceStart[f] !== 4) continue;
    let r = 1;
    for (let k = 0; k < 4; k++) if (!d.regularCorner[topo.faceStart[f] + k]) r = 0;
    regular[f] = r;
  }
  const assigned = new Uint8Array(nf);
  const okStep = (h: number, region: Set<number>): boolean => {
    if (h < 0) return false;
    const f = topo.heF[h];
    return regular[f] === 1 && !assigned[f] && !region.has(f) && topo.faceStart[f + 1] - topo.faceStart[f] === 4;
  };
  const stepU = (h: number): number => {
    const t = topo.heTwin[topo.heNext[h]];
    if (t < 0 || topo.edgeSharp[topo.heEdge[t]] !== EdgeSharpness.None) return -1;
    return topo.heNext[t];
  };
  const stepV = (h: number): number => {
    const t = topo.heTwin[topo.heNext[topo.heNext[h]]];
    if (t < 0 || topo.edgeSharp[topo.heEdge[t]] !== EdgeSharpness.None) return -1;
    return t;
  };
  const grow = (h00: number): number[][] => {
    const region = new Set<number>([topo.heF[h00]]);
    const row0 = [h00];
    while (row0.length < maxSpan) {
      const n = stepU(row0[row0.length - 1]);
      if (!okStep(n, region)) break;
      row0.push(n);
      region.add(topo.heF[n]);
    }
    const rows = [row0];
    while (rows.length < maxSpan) {
      const prev = rows[rows.length - 1];
      const row: number[] = [];
      const added: number[] = [];
      let ok = true;
      for (let i = 0; i < prev.length && ok; i++) {
        const n = stepV(prev[i]);
        if (!okStep(n, region)) ok = false;
        else if (i > 0 && stepU(row[i - 1]) !== n) ok = false;
        else {
          row.push(n);
          region.add(topo.heF[n]);
          added.push(topo.heF[n]);
        }
      }
      if (!ok) {
        for (const f of added) region.delete(f);
        break;
      }
      rows.push(row);
    }
    return rows;
  };
  const surfaces: BSplineSurfaceDef[] = [];
  const faceGroups: number[][] = [];
  for (let f = 0; f < nf; f++) {
    if (!regular[f] || assigned[f]) continue;
    const s = topo.faceStart[f];
    const a = grow(s);
    const b = grow(topo.heNext[s]);
    const area = (r: number[][]) => r.length * r[0].length;
    const rows = area(b) > area(a) ? b : a;
    const H = rows.length, W = rows[0].length;
    if (W * H < minFaces) continue;
    const grids: Vec3[][][] = [];
    let ok = true;
    for (let j = 0; j < H && ok; j++) {
      grids.push([]);
      for (let i = 0; i < W; i++) {
        const g = faceSplineGrid(d, rows[j][i]);
        if (!g) {
          ok = false;
          break;
        }
        grids[j].push(g);
      }
    }
    if (!ok) continue;
    // assemble the (W+3) x (H+3) net
    const NU = W + 3, NV = H + 3;
    const net: Vec3[][] = Array.from({ length: NU }, () => new Array<Vec3>(NV));
    for (let j = 0; j < H; j++)
      for (let i = 0; i < W; i++) {
        const g = grids[j][i];
        for (let a2 = 0; a2 < 4; a2++) for (let b2 = 0; b2 < 4; b2++) if (!net[i + a2][j + b2]) net[i + a2][j + b2] = g[a2 * 4 + b2];
      }
    // clamp along u then along v
    const cu: Vec3[][] = Array.from({ length: NU }, () => new Array<Vec3>(NV));
    for (let j = 0; j < NV; j++) {
      const line = clampUniformCubic(net.map((col) => col[j]));
      for (let i = 0; i < NU; i++) cu[i][j] = line[i];
    }
    const cp: Vec3[] = new Array(NU * NV);
    for (let i = 0; i < NU; i++) {
      const line = clampUniformCubic(cu[i]);
      for (let j = 0; j < NV; j++) cp[i * NV + j] = line[j];
    }
    const knots = (n: number) => [0, 0, 0, ...Array.from({ length: n + 1 }, (_, k) => k), n, n, n];
    surfaces.push({ degreeU: 3, degreeV: 3, knotsU: knots(W), knotsV: knots(H), nu: NU, nv: NV, controlPoints: cp });
    const group: number[] = [];
    for (const r of rows) for (const h of r) {
      assigned[topo.heF[h]] = 1;
      group.push(topo.heF[h]);
    }
    faceGroups.push(group);
  }
  const mergedCount = surfaces.length;
  for (let f = 0; f < nf; f++) {
    if (assigned[f] || !topo.faceValid[f] || topo.faceStart[f + 1] - topo.faceStart[f] !== 4) continue;
    surfaces.push(bezierSurface(facePatch(d, topo.faceStart[f])));
    faceGroups.push([f]);
  }
  return { surfaces, faceGroups, mesh, mergedCount };
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

function findSpan(n: number, p: number, u: number, U: number[]): number {
  if (u >= U[n + 1]) return n;
  if (u <= U[p]) return p;
  let low = p, high = n + 1, mid = (low + high) >> 1;
  while (u < U[mid] || u >= U[mid + 1]) {
    if (u < U[mid]) high = mid;
    else low = mid;
    mid = (low + high) >> 1;
  }
  return mid;
}

/** Basis functions and first derivatives (Piegl & Tiller A2.3, k = 1). */
function basisDers(span: number, u: number, p: number, U: number[]): { N: number[]; dN: number[] } {
  const ndu: number[][] = Array.from({ length: p + 1 }, () => new Array(p + 1).fill(0));
  const left = new Array(p + 1).fill(0), right = new Array(p + 1).fill(0);
  ndu[0][0] = 1;
  for (let j = 1; j <= p; j++) {
    left[j] = u - U[span + 1 - j];
    right[j] = U[span + j] - u;
    let saved = 0;
    for (let r = 0; r < j; r++) {
      ndu[j][r] = right[r + 1] + left[j - r];
      const temp = ndu[r][j - 1] / ndu[j][r];
      ndu[r][j] = saved + right[r + 1] * temp;
      saved = left[j - r] * temp;
    }
    ndu[j][j] = saved;
  }
  const N: number[] = [];
  const dN: number[] = [];
  for (let r = 0; r <= p; r++) {
    N.push(ndu[r][p]);
    if (p === 0) {
      dN.push(0);
      continue;
    }
    const a = r >= 1 ? ndu[r - 1][p - 1] / ndu[p][r - 1] : 0;
    const b = r <= p - 1 ? ndu[r][p - 1] / ndu[p][r] : 0;
    dN.push(p * (a - b));
  }
  return { N, dN };
}

export interface SurfacePoint {
  p: Vec3;
  du: Vec3;
  dv: Vec3;
}

/** Point and first partial derivatives of a (non-rational) B-spline surface. */
export function evaluateSurfaceDerivs(s: BSplineSurfaceDef, u: number, v: number): SurfacePoint {
  const su = findSpan(s.nu - 1, s.degreeU, u, s.knotsU);
  const sv = findSpan(s.nv - 1, s.degreeV, v, s.knotsV);
  const bu = basisDers(su, u, s.degreeU, s.knotsU);
  const bv = basisDers(sv, v, s.degreeV, s.knotsV);
  const p: Vec3 = [0, 0, 0], du: Vec3 = [0, 0, 0], dv: Vec3 = [0, 0, 0];
  for (let a = 0; a <= s.degreeU; a++) {
    const i = su - s.degreeU + a;
    for (let b = 0; b <= s.degreeV; b++) {
      const j = sv - s.degreeV + b;
      const c = s.controlPoints[i * s.nv + j];
      const w = bu.N[a] * bv.N[b], wu = bu.dN[a] * bv.N[b], wv = bu.N[a] * bv.dN[b];
      for (let k = 0; k < 3; k++) {
        p[k] += w * c[k];
        du[k] += wu * c[k];
        dv[k] += wv * c[k];
      }
    }
  }
  return { p, du, dv };
}

/** Point on a (non-rational) B-spline surface. */
export function evaluateSurface(s: BSplineSurfaceDef, u: number, v: number): Vec3 {
  return evaluateSurfaceDerivs(s, u, v).p;
}

/** Unit normal of a B-spline surface (du x dv). */
export function surfaceNormal(s: BSplineSurfaceDef, u: number, v: number): Vec3 {
  const r = evaluateSurfaceDerivs(s, u, v);
  return normalize(cross(r.du, r.dv));
}

/** Parameter domain of a surface. */
export function surfaceDomain(s: BSplineSurfaceDef): { u0: number; u1: number; v0: number; v1: number } {
  return { u0: s.knotsU[s.degreeU], u1: s.knotsU[s.nu], v0: s.knotsV[s.degreeV], v1: s.knotsV[s.nv] };
}
