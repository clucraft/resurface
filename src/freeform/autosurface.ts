/**
 * One-click auto-surfacing of organic shapes:
 *   coarse (decimated) triangle mesh -> quad cage -> snap-to-mesh fit -> bicubic NURBS patches
 * ("G2 everywhere except G1 at extraordinary points" in the limit; the exported patches are
 * exact B-spline patches on regular regions and ACC patches around extraordinary vertices).
 *
 * Cage construction:
 *   1. weld duplicate vertices, drop degenerate / duplicate triangles, orient consistently
 *   2. (quadDominant) pair adjacent triangles into quads greedily by quality (planarity, corner
 *      angles close to 90°, aspect, convexity) with a bonus for pairings that move vertex
 *      valences towards 4 (fewer extraordinary vertices); one pass of length-3 augmenting swaps
 *      reduces leftover triangles
 *   3. (split) one Catmull–Clark-topology split with linear positions: quad -> 4 quads,
 *      triangle -> 3 quads, so the cage is all-quad (always valid)
 * Open meshes are supported (boundary curves follow the cubic B-spline boundary rules).
 */
import type { BSplineSurfaceDef, ClosestPointQuery, MeshData, ProgressFn, Vec3 } from "../core/types";
import { add, cross, dot, length, normalize, sub } from "../core/vec";
import { cageQuality, type CageQuality } from "./analysis";
import { subdivideLinear } from "./catmullClark";
import { fitCageToTarget, snapVerticesToTarget, type CageFitResult } from "./fit";
import { mergeRegularRegions, toBicubicPatches } from "./patches";
import { buildTopology, EdgeSharpness, orient, quadMeshFromTriangles, weldVertices, type QuadMesh } from "./quadMesh";

export interface AutoCageOptions {
  /** pair triangles into quads before splitting (default true) */
  quadDominant?: boolean;
  /** apply one CC-topology split so the cage is all quads (default true) */
  split?: boolean;
  /** minimum quality (0..1) of a paired quad (default 0.3) */
  minQuadQuality?: number;
  /** mark edges with a dihedral angle above this as creases (default: no creases) */
  creaseAngleDeg?: number;
  /** vertex weld tolerance (default 1e-6 x bounding-box diagonal) */
  weldTolerance?: number;
}

function triNormal(a: Vec3, b: Vec3, c: Vec3): Vec3 {
  return normalize(cross(sub(b, a), sub(c, a)));
}

/** Quality of a convex quad (0 = unusable, 1 = square). n1, n2: unit normals of the two source triangles. */
function quadQuality(p: Vec3[], n1: Vec3, n2: Vec3): number {
  const planar = dot(n1, n2);
  if (!(planar > 0.25)) return 0;
  const N = normalize(add(n1, n2));
  let worst = 0;
  for (let k = 0; k < 4; k++) {
    const a = p[k], next = p[(k + 1) % 4], prev = p[(k + 3) % 4];
    const e1 = sub(next, a), e2 = sub(prev, a);
    const l1 = length(e1), l2 = length(e2);
    if (l1 < 1e-300 || l2 < 1e-300) return 0;
    if (dot(cross(e1, e2), N) <= 1e-12 * l1 * l2) return 0;
    const ang = Math.acos(Math.max(-1, Math.min(1, dot(e1, e2) / (l1 * l2))));
    worst = Math.max(worst, Math.abs(ang - Math.PI / 2));
  }
  const angleQ = Math.max(0, 1 - worst / (Math.PI / 2));
  const L = [0, 1, 2, 3].map((k) => length(sub(p[(k + 1) % 4], p[k])));
  const s1 = L[0] + L[2], s2 = L[1] + L[3];
  const aspect = Math.min(s1, s2) / Math.max(s1, s2, 1e-300);
  return angleQ * (0.5 + 0.5 * planar) * Math.sqrt(aspect);
}

interface Candidate {
  e: number;
  f1: number;
  f2: number;
  a: number;
  b: number;
  quad: [number, number, number, number];
  q: number;
}

/** Binary max-heap of [score, id]. */
class MaxHeap {
  private s: number[] = [];
  private id: number[] = [];
  get size() {
    return this.s.length;
  }
  push(score: number, id: number) {
    this.s.push(score);
    this.id.push(id);
    let i = this.s.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.s[p] > this.s[i] || (this.s[p] === this.s[i] && this.id[p] < this.id[i])) break;
      this.swap(i, p);
      i = p;
    }
  }
  pop(): [number, number] {
    const top: [number, number] = [this.s[0], this.id[0]];
    const ls = this.s.pop()!, li = this.id.pop()!;
    if (this.s.length) {
      this.s[0] = ls;
      this.id[0] = li;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        const better = (x: number, y: number) => this.s[x] > this.s[y] || (this.s[x] === this.s[y] && this.id[x] < this.id[y]);
        if (l < this.s.length && better(l, m)) m = l;
        if (r < this.s.length && better(r, m)) m = r;
        if (m === i) break;
        this.swap(i, m);
        i = m;
      }
    }
    return top;
  }
  private swap(a: number, b: number) {
    [this.s[a], this.s[b]] = [this.s[b], this.s[a]];
    [this.id[a], this.id[b]] = [this.id[b], this.id[a]];
  }
}

/** Pair triangles into quads (input: oriented triangle cage). Returns a quad-dominant mesh. */
export function pairTrianglesToQuads(qm: QuadMesh, minQuality = 0.3): QuadMesh {
  const topo = buildTopology(qm);
  const P = qm.vertices;
  const nv = topo.nv;
  const valence = new Int32Array(nv);
  for (let v = 0; v < nv; v++) valence[v] = topo.vEdgeStart[v + 1] - topo.vEdgeStart[v];
  const boundary = topo.vBoundary;
  // interior angle sum at boundary vertices (sharp boundary corners may become single-face corners)
  const angleSum = new Float64Array(nv);
  for (let h = 0; h < topo.nh; h++) {
    const v = topo.heV[h];
    if (!boundary[v]) continue;
    const a = sub(P[topo.heV[topo.heNext[h]]], P[v]), b = sub(P[topo.heV[topo.hePrev[h]]], P[v]);
    angleSum[v] += Math.acos(Math.max(-1, Math.min(1, dot(a, b) / Math.max(1e-300, length(a) * length(b)))));
  }
  const isTri = (f: number) => topo.faceValid[f] && topo.faceStart[f + 1] - topo.faceStart[f] === 3;
  const cands: Candidate[] = [];
  for (let e = 0; e < topo.ne; e++) {
    if (topo.edgeSharp[e] !== EdgeSharpness.None) continue;
    const h = topo.edgeHe[e], t = topo.heTwin[h];
    if (t < 0) continue;
    const f1 = topo.heF[h], f2 = topo.heF[t];
    if (!isTri(f1) || !isTri(f2) || f1 === f2) continue;
    const a = topo.heV[h], b = topo.heV[topo.heNext[h]], c = topo.heV[topo.hePrev[h]], d = topo.heV[topo.hePrev[t]];
    if (c === d) continue;
    const quad: [number, number, number, number] = [a, d, b, c];
    const n1 = triNormal(P[a], P[b], P[c]), n2 = triNormal(P[b], P[a], P[d]);
    const q = quadQuality(quad.map((i) => P[i]), n1, n2);
    if (q < minQuality) continue;
    cands.push({ e, f1, f2, a, b, quad, q });
  }
  const bonus = (v: number): number => {
    const val = valence[v];
    if (boundary[v]) {
      // regular boundary vertex: valence 3 (2 faces); valence 2 = single-face corner
      if (val <= 3) return angleSum[v] < (2 * Math.PI) / 3 ? 0 : -Infinity;
      return val === 4 ? 0.25 : 0.12;
    }
    if (val <= 3) return -Infinity;
    if (val === 4) return -0.25;
    return val === 5 ? 0.25 : 0.12;
  };
  const score = (c: Candidate) => c.q + bonus(c.a) + bonus(c.b);
  const heap = new MaxHeap();
  cands.forEach((c, i) => {
    const s = score(c);
    if (s > -Infinity) heap.push(s, i);
  });
  const pairedWith = new Int32Array(topo.nf).fill(-1);
  const pairCand = new Int32Array(topo.nf).fill(-1);
  while (heap.size) {
    const [s, i] = heap.pop();
    const c = cands[i];
    if (pairedWith[c.f1] >= 0 || pairedWith[c.f2] >= 0) continue;
    const now = score(c);
    if (now === -Infinity) continue;
    if (now < s - 1e-12) {
      heap.push(now, i);
      continue;
    }
    pairedWith[c.f1] = c.f2;
    pairedWith[c.f2] = c.f1;
    pairCand[c.f1] = pairCand[c.f2] = i;
    valence[c.a]--;
    valence[c.b]--;
  }
  // one pass of augmenting swaps: unpaired t, neighbour u paired with w, w has unpaired neighbour x
  const candOf = new Map<string, number>();
  cands.forEach((c, i) => candOf.set(c.f1 < c.f2 ? `${c.f1}_${c.f2}` : `${c.f2}_${c.f1}`, i));
  const candBetween = (x: number, y: number) => candOf.get(x < y ? `${x}_${y}` : `${y}_${x}`);
  const triNeighbours = (f: number): number[] => {
    const out: number[] = [];
    for (let h = topo.faceStart[f]; h < topo.faceStart[f + 1]; h++) {
      const t = topo.heTwin[h];
      if (t >= 0) out.push(topo.heF[t]);
    }
    return out;
  };
  const okVal = (v: number, delta: number) => (boundary[v] ? valence[v] + delta >= 3 || angleSum[v] < (2 * Math.PI) / 3 : valence[v] + delta >= 3);
  for (let t = 0; t < topo.nf; t++) {
    if (!isTri(t) || pairedWith[t] >= 0) continue;
    let done = false;
    for (const u of triNeighbours(t)) {
      if (done) break;
      const w = pairedWith[u];
      if (w < 0) continue;
      const ctu = candBetween(t, u);
      if (ctu === undefined) continue;
      for (const x of triNeighbours(w)) {
        if (x === t || x === u || pairedWith[x] >= 0) continue;
        const cwx = candBetween(w, x);
        if (cwx === undefined) continue;
        const old = cands[pairCand[u]];
        const A = cands[ctu], B = cands[cwx];
        // valence bookkeeping: restore old diagonal, remove the two new ones
        valence[old.a]++;
        valence[old.b]++;
        const dec = new Map<number, number>();
        for (const v of [A.a, A.b, B.a, B.b]) dec.set(v, (dec.get(v) ?? 0) + 1);
        let ok = true;
        for (const [v, k] of dec) if (!okVal(v, -k)) ok = false;
        if (!ok) {
          valence[old.a]--;
          valence[old.b]--;
          continue;
        }
        valence[A.a]--;
        valence[A.b]--;
        valence[B.a]--;
        valence[B.b]--;
        pairedWith[t] = u;
        pairedWith[u] = t;
        pairedWith[w] = x;
        pairedWith[x] = w;
        pairCand[t] = pairCand[u] = ctu;
        pairCand[w] = pairCand[x] = cwx;
        done = true;
        break;
      }
    }
  }
  const faces: number[][] = [];
  for (let f = 0; f < topo.nf; f++) {
    if (!topo.faceValid[f]) continue;
    const g = pairedWith[f];
    if (g < 0) faces.push(qm.faces[f].slice());
    else if (f < g) faces.push(cands[pairCand[f]].quad.slice());
  }
  const out: QuadMesh = { vertices: qm.vertices.map((p) => [p[0], p[1], p[2]] as Vec3), faces };
  if (qm.creaseEdges) out.creaseEdges = qm.creaseEdges.map((e) => [e[0], e[1]] as [number, number]);
  return out;
}

/** Mark edges whose dihedral angle exceeds the threshold as creases. */
function detectCreases(qm: QuadMesh, angleDeg: number): [number, number][] {
  const topo = buildTopology(qm);
  const cosT = Math.cos((angleDeg * Math.PI) / 180);
  const fn = qm.faces.map((f) => {
    const p = f.map((v) => qm.vertices[v]);
    return normalize(cross(sub(p[1], p[0]), sub(p[p.length - 1], p[0])));
  });
  const out: [number, number][] = [];
  for (let e = 0; e < topo.ne; e++) {
    const h = topo.edgeHe[e], t = topo.heTwin[h];
    if (t < 0) continue;
    if (dot(fn[topo.heF[h]], fn[topo.heF[t]]) < cosT) out.push([topo.edgeV[2 * e], topo.edgeV[2 * e + 1]]);
  }
  return out;
}

/** Build a quad cage from a coarse (decimated) triangle mesh of the scan. */
export function autoCageFromMesh(coarse: MeshData, opts: AutoCageOptions = {}): QuadMesh {
  let qm = quadMeshFromTriangles(coarse);
  let lo: Vec3 = [Infinity, Infinity, Infinity], hi: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (const p of qm.vertices)
    for (let c = 0; c < 3; c++) {
      lo[c] = Math.min(lo[c], p[c]);
      hi[c] = Math.max(hi[c], p[c]);
    }
  const diag = qm.vertices.length ? length(sub(hi, lo)) : 0;
  weldVertices(qm, opts.weldTolerance ?? 1e-6 * diag, true);
  orient(qm);
  if (opts.creaseAngleDeg !== undefined) {
    const cr = detectCreases(qm, opts.creaseAngleDeg);
    if (cr.length) qm.creaseEdges = cr;
  }
  if (opts.quadDominant !== false) qm = pairTrianglesToQuads(qm, opts.minQuadQuality ?? 0.3);
  if (opts.split !== false) qm = subdivideLinear(qm);
  return qm;
}

export interface AutosurfaceOptions extends AutoCageOptions {
  /** fitting sample level (default 2) */
  levels?: number;
  /** fitting iterations (default 6) */
  iterations?: number;
  smoothness?: number;
  pointToPlane?: number;
  maxDistance?: number;
  /** snap cage vertices so their limit points lie on the target before fitting (default true) */
  snap?: boolean;
  /** merge rectangular regular regions into larger B-spline surfaces (default false) */
  mergeRegular?: boolean;
  onProgress?: ProgressFn;
}

export interface AutosurfaceResult {
  cage: QuadMesh;
  patches: BSplineSurfaceDef[];
  rms: number;
  maxDev: number;
  quality: CageQuality;
  fit: CageFitResult;
}

/** autoCageFromMesh + fitCageToTarget + toBicubicPatches. */
export function autosurface(coarse: MeshData, target: ClosestPointQuery, opts: AutosurfaceOptions = {}): AutosurfaceResult {
  const progress = opts.onProgress;
  progress?.(0, "Building quad cage");
  let cage = autoCageFromMesh(coarse, opts);
  if (opts.snap !== false) {
    progress?.(0.1, "Snapping cage to the scan");
    cage = snapVerticesToTarget(cage, target, { iterations: 8, maxDistance: opts.maxDistance });
  }
  const fit = fitCageToTarget(cage, target, {
    levels: opts.levels ?? 2,
    iterations: opts.iterations ?? 6,
    smoothness: opts.smoothness,
    pointToPlane: opts.pointToPlane,
    maxDistance: opts.maxDistance,
    onProgress: progress ? (f, m) => progress(0.15 + 0.75 * f, m) : undefined,
  });
  progress?.(0.92, "Converting to NURBS patches");
  const patches = opts.mergeRegular ? mergeRegularRegions(fit.cage).surfaces : toBicubicPatches(fit.cage);
  const quality = cageQuality(fit.cage);
  progress?.(1, `Autosurface done: ${patches.length} patches, rms ${fit.rms.toPrecision(3)}`);
  return { cage: fit.cage, patches, rms: fit.rms, maxDev: fit.maxDev, quality, fit };
}
