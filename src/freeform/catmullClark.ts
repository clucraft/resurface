/**
 * Catmull–Clark subdivision for quad cages (n-gons allowed), with
 *  - boundary rules: boundary edges / vertices follow the cubic B-spline curve rules, boundary
 *    vertices with a single face are interpolated corners ("edge and corner" boundary mode)
 *  - sharp crease edges (same rules as boundaries), darts (1 crease) use the smooth vertex rule,
 *    vertices with >= 3 sharp edges are corners
 *  - closed-form limit positions and limit normals (tangent masks, per crease sector)
 *  - triangulated limit-surface meshes for display
 *  - a sparse linear stencil matrix mapping cage vertices to limit-surface samples (fitting)
 *
 * Refinement ordering: new vertices = [vertex points (same index as the coarse vertex),
 * edge points (nv + edge id), face points (nv + ne + valid face rank)]. The sub-faces of coarse
 * face f are emitted in order, corner k first: sub-face index == half-edge index of corner k.
 */
import type { MeshData, Vec3 } from "../core/types";
import { add, cross, dot, length, normalize, scale, sub } from "../core/vec";
import {
  buildTopology,
  buildTopologyRaw,
  EdgeSharpness,
  isAllQuads,
  polygonNormal,
  ringSectors,
  sharpNeighbours,
  VertexKind,
  vertexRing,
  type QuadMesh,
  type RingSector,
  type Topology,
  type VertexRing,
} from "./quadMesh";
import { CsrBuilder, csrMultiply, csrMulVec3, csrSelectRows, type CsrMatrix } from "./sparse";

// ---------------------------------------------------------------------------
// One refinement step as a sparse linear operator
// ---------------------------------------------------------------------------

export interface Refinement {
  /** subdivision matrix: rows = fine vertices, columns = coarse vertices */
  S: CsrMatrix;
  faces: number[][];
  creaseEdges: [number, number][];
  fixed?: boolean[];
  /** coarse parent face of every fine face */
  faceParent: Int32Array;
  /** corner index within the parent face */
  faceCorner: Int32Array;
  nVertices: number;
  edgePointStart: number;
  facePointStart: number;
  /** fine vertex index of the face point of every coarse face (-1 for invalid faces) */
  facePointIndex: Int32Array;
}

/**
 * One level of refinement for the given topology. `linear` = bilinear/centroid split (same
 * topology as Catmull–Clark, positions not smoothed).
 */
export function refineTopology(topo: Topology, opts: { linear?: boolean; fixed?: boolean[] } = {}): Refinement {
  const linear = !!opts.linear;
  const { nv, ne, nf } = topo;
  const facePointIndex = new Int32Array(nf).fill(-1);
  let nValidFaces = 0;
  for (let f = 0; f < nf; f++) if (topo.faceValid[f]) facePointIndex[f] = nv + ne + nValidFaces++;
  const nNew = nv + ne + nValidFaces;
  const b = new CsrBuilder(nv, nNew * 8);
  const { heV, heF, heTwin, faceStart, vOutStart, vOut, vEdgeStart, vEdges, edgeV, edgeHe, edgeSharp, vKind } = topo;

  // vertex points
  for (let v = 0; v < nv; v++) {
    const nOut = vOutStart[v + 1] - vOutStart[v];
    const kind = vKind[v];
    if (linear || nOut === 0 || kind === VertexKind.Corner) {
      b.add(v, 1);
    } else if (kind === VertexKind.Crease) {
      const nb = sharpNeighbours(topo, v);
      if (nb.length === 2) {
        b.add(nb[0], 0.125);
        b.add(v, 0.75);
        b.add(nb[1], 0.125);
      } else b.add(v, 1);
    } else {
      const n = vEdgeStart[v + 1] - vEdgeStart[v];
      const inv2 = 1 / (n * n);
      b.add(v, (n - 2) / n);
      for (let p = vEdgeStart[v]; p < vEdgeStart[v + 1]; p++) {
        const e = vEdges[p];
        b.add(edgeV[2 * e] === v ? edgeV[2 * e + 1] : edgeV[2 * e], inv2);
      }
      for (let p = vOutStart[v]; p < vOutStart[v + 1]; p++) {
        const f = heF[vOut[p]];
        const s = faceStart[f], m = faceStart[f + 1] - s;
        const w = inv2 / m;
        for (let k = 0; k < m; k++) b.add(heV[s + k], w);
      }
    }
    b.endRow();
  }
  // edge points
  for (let e = 0; e < ne; e++) {
    const a = edgeV[2 * e], c = edgeV[2 * e + 1];
    const h0 = edgeHe[e], h1 = heTwin[h0];
    if (linear || edgeSharp[e] !== EdgeSharpness.None || h1 < 0) {
      b.add(a, 0.5);
      b.add(c, 0.5);
    } else {
      b.add(a, 0.25);
      b.add(c, 0.25);
      for (const h of [h0, h1]) {
        const f = heF[h];
        const s = faceStart[f], m = faceStart[f + 1] - s;
        const w = 0.25 / m;
        for (let k = 0; k < m; k++) b.add(heV[s + k], w);
      }
    }
    b.endRow();
  }
  // face points
  for (let f = 0; f < nf; f++) {
    if (!topo.faceValid[f]) continue;
    const s = faceStart[f], m = faceStart[f + 1] - s;
    for (let k = 0; k < m; k++) b.add(heV[s + k], 1 / m);
    b.endRow();
  }
  const S = b.build();

  // fine faces
  const faces: number[][] = new Array(topo.nh);
  const faceParent = new Int32Array(topo.nh);
  const faceCorner = new Int32Array(topo.nh);
  for (let f = 0; f < nf; f++) {
    if (!topo.faceValid[f]) continue;
    const s = faceStart[f], m = faceStart[f + 1] - s;
    const fp = facePointIndex[f];
    for (let k = 0; k < m; k++) {
      const h = s + k;
      faces[h] = [heV[h], nv + topo.heEdge[h], fp, nv + topo.heEdge[topo.hePrev[h]]];
      faceParent[h] = f;
      faceCorner[h] = k;
    }
  }
  const creaseEdges: [number, number][] = [];
  for (let e = 0; e < ne; e++) {
    if (edgeSharp[e] !== EdgeSharpness.Crease) continue;
    creaseEdges.push([edgeV[2 * e], nv + e], [nv + e, edgeV[2 * e + 1]]);
  }
  let fixed: boolean[] | undefined;
  if (opts.fixed) {
    fixed = new Array(nNew).fill(false);
    for (let v = 0; v < nv; v++) fixed[v] = !!opts.fixed[v];
  }
  return { S, faces, creaseEdges, fixed, faceParent, faceCorner, nVertices: nNew, edgePointStart: nv, facePointStart: nv + ne, facePointIndex };
}

function refineMesh(qm: QuadMesh, linear: boolean): { mesh: QuadMesh; refinement: Refinement } {
  const topo = buildTopology(qm);
  const r = refineTopology(topo, { linear, fixed: qm.fixed });
  const mesh: QuadMesh = { vertices: csrMulVec3(r.S, qm.vertices), faces: r.faces };
  if (r.creaseEdges.length) mesh.creaseEdges = r.creaseEdges;
  if (r.fixed) mesh.fixed = r.fixed;
  return { mesh, refinement: r };
}

/** One refinement level plus the parent maps (`refinement.faceParent` etc.). */
export function subdivideWithMap(qm: QuadMesh, linear = false): { mesh: QuadMesh; refinement: Refinement } {
  return refineMesh(qm, linear);
}

/** One level of Catmull–Clark subdivision (n-gons -> quads, creases and boundaries kept sharp). */
export function subdivide(qm: QuadMesh): QuadMesh {
  return refineMesh(qm, false).mesh;
}

/** Same topology split as `subdivide` but with linear positions (edge midpoints, face centroids). */
export function subdivideLinear(qm: QuadMesh): QuadMesh {
  return refineMesh(qm, true).mesh;
}

/** `levels` levels of Catmull–Clark subdivision (control points of the refined cage, not limit points). */
export function subdivideLevels(qm: QuadMesh, levels: number): QuadMesh {
  let m = qm;
  for (let l = 0; l < levels; l++) m = subdivide(m);
  return m;
}

// ---------------------------------------------------------------------------
// Limit stencils
// ---------------------------------------------------------------------------

/**
 * Limit-position stencils for every vertex of a quad mesh (rows = vertices, cols = vertices).
 * Faces that are not quads contribute their centroid as "diagonal" (approximation) — call on
 * all-quad meshes for exact results (see `limitStencilMatrix`).
 */
export function limitMatrix(topo: Topology): CsrMatrix {
  const { nv, heV, heNext, heF, faceStart, vOutStart, vOut, vEdgeStart, vEdges, edgeV, vKind } = topo;
  const b = new CsrBuilder(nv, nv * 10);
  for (let v = 0; v < nv; v++) {
    const nOut = vOutStart[v + 1] - vOutStart[v];
    const kind = vKind[v];
    if (nOut === 0 || kind === VertexKind.Corner) b.add(v, 1);
    else if (kind === VertexKind.Crease) {
      const nb = sharpNeighbours(topo, v);
      if (nb.length === 2) {
        b.add(nb[0], 1 / 6);
        b.add(v, 4 / 6);
        b.add(nb[1], 1 / 6);
      } else b.add(v, 1);
    } else {
      const n = vEdgeStart[v + 1] - vEdgeStart[v];
      const c = 1 / (n * (n + 5));
      b.add(v, n * n * c);
      for (let p = vEdgeStart[v]; p < vEdgeStart[v + 1]; p++) {
        const e = vEdges[p];
        b.add(edgeV[2 * e] === v ? edgeV[2 * e + 1] : edgeV[2 * e], 4 * c);
      }
      for (let p = vOutStart[v]; p < vOutStart[v + 1]; p++) {
        const h = vOut[p];
        const f = heF[h];
        const s = faceStart[f], m = faceStart[f + 1] - s;
        if (m === 4) b.add(heV[heNext[heNext[h]]], c);
        else for (let k = 0; k < m; k++) b.add(heV[s + k], c / m);
      }
    }
    b.endRow();
  }
  return b.build();
}

/** Exact limit stencils of the cage vertices for any cage (refines once internally if it has non-quads). */
export function limitStencilMatrix(qm: QuadMesh): CsrMatrix {
  const topo = buildTopology(qm);
  if (isAllQuads(qm)) return limitMatrix(topo);
  const r = refineTopology(topo);
  const topo1 = buildTopologyRaw(r.nVertices, r.faces, r.creaseEdges);
  const lim1 = limitMatrix(topo1);
  const rows = new Int32Array(qm.vertices.length);
  for (let i = 0; i < rows.length; i++) rows[i] = i;
  return csrMultiply(csrSelectRows(lim1, rows), r.S);
}

/** Limit-surface position of every cage vertex. */
export function limitPoints(qm: QuadMesh): Vec3[] {
  return csrMulVec3(limitStencilMatrix(qm), qm.vertices);
}

// ---------------------------------------------------------------------------
// Limit normals
// ---------------------------------------------------------------------------

function diagonalPoint(topo: Topology, P: ArrayLike<Vec3>, h: number): Vec3 {
  const f = topo.heF[h];
  const s = topo.faceStart[f], m = topo.faceStart[f + 1] - s;
  if (m === 4) return P[topo.heV[topo.heNext[topo.heNext[h]]]];
  let c: Vec3 = [0, 0, 0];
  for (let k = 0; k < m; k++) c = add(c, P[topo.heV[s + k]]);
  return scale(c, 1 / m);
}

/** Area-weighted normal of the ring faces with the given ring face indices. */
function fanNormal(topo: Topology, P: ArrayLike<Vec3>, ring: VertexRing, faceIdx: number[]): Vec3 {
  let n: Vec3 = [0, 0, 0];
  for (const i of faceIdx) {
    const f = ring.faces[i];
    const pts: Vec3[] = [];
    for (let h = topo.faceStart[f]; h < topo.faceStart[f + 1]; h++) pts.push(P[topo.heV[h]]);
    n = add(n, polygonNormal(pts));
  }
  return n;
}

/**
 * Limit tangent vectors (unnormalised) of a smooth / dart vertex with a closed ring
 * (Halstead et al. masks). The limit tangent along ring edge j is cos(2πj/n) t1 + sin(2πj/n) t2.
 */
export function smoothLimitTangents(topo: Topology, P: ArrayLike<Vec3>, ring: VertexRing): [Vec3, Vec3] {
  const n = ring.nbr.length;
  const V = P[ring.v];
  const A = 1 + Math.cos((2 * Math.PI) / n) + Math.cos(Math.PI / n) * Math.sqrt(2 * (9 + Math.cos((2 * Math.PI) / n)));
  let t1: Vec3 = [0, 0, 0], t2: Vec3 = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const ci = Math.cos((2 * Math.PI * i) / n), ci1 = Math.cos((2 * Math.PI * (i + 1)) / n);
    const si = Math.sin((2 * Math.PI * i) / n), si1 = Math.sin((2 * Math.PI * (i + 1)) / n);
    const E = sub(P[ring.nbr[i]], V);
    const F = sub(diagonalPoint(topo, P, ring.faceHe[i]), V);
    t1 = add(t1, add(scale(E, A * ci), scale(F, ci + ci1)));
    t2 = add(t2, add(scale(E, A * si), scale(F, si + si1)));
  }
  return [t1, t2];
}

/** Limit normal of a smooth (or dart) vertex with a closed ring (Halstead et al. tangent masks). */
function smoothLimitNormal(topo: Topology, P: ArrayLike<Vec3>, ring: VertexRing): Vec3 {
  const [t1, t2] = smoothLimitTangents(topo, P, ring);
  return cross(t1, t2);
}

/** Limit normal of one crease sector (tangent along the crease x cross tangent into the sector). */
function creaseSectorNormal(V: Vec3, E: Vec3[], F: Vec3[]): Vec3 {
  const k = F.length;
  const tan1 = scale(sub(E[0], E[k]), 0.5);
  let tan2: Vec3;
  if (k === 1) {
    tan2 = add(sub(E[0], V), sub(E[1], V));
  } else if (k === 2) {
    tan2 = scale(
      add(add(add(scale(sub(E[0], V), -1), scale(sub(E[1], V), 4)), scale(sub(E[2], V), -1)), add(sub(F[0], V), sub(F[1], V))),
      1 / 6,
    );
  } else {
    const theta = Math.PI / k;
    const c = Math.cos(theta), s = Math.sin(theta);
    const R = (c + 1) / s;
    const creaseW = -R * (1 + 2 * c);
    // vertex weight 4R(c-1) is absorbed by writing everything relative to V (weights sum to zero)
    tan2 = add(scale(sub(E[0], V), creaseW), scale(sub(E[k], V), creaseW));
    tan2 = add(tan2, scale(sub(F[0], V), s));
    for (let i = 1; i < k; i++) {
      const sI = Math.sin(i * theta), sI1 = Math.sin((i + 1) * theta);
      tan2 = add(tan2, scale(sub(E[i], V), 4 * sI));
      tan2 = add(tan2, scale(sub(F[i], V), sI + sI1));
    }
  }
  return cross(tan1, tan2);
}

/** Normal of a corner sector. */
function cornerSectorNormal(V: Vec3, E: Vec3[], F: Vec3[]): Vec3 {
  const k = F.length;
  if (k === 1) return cross(sub(E[0], V), sub(E[1], V));
  let n: Vec3 = [0, 0, 0];
  for (let j = 0; j < k; j++) {
    n = add(n, cross(sub(E[j], V), sub(F[j], V)));
    n = add(n, cross(sub(F[j], V), sub(E[j + 1], V)));
  }
  return n;
}

export interface VertexSectorNormals {
  ring: VertexRing;
  sectors: RingSector[];
  /** unit normal per sector */
  normals: Vec3[];
  /** sector index of each ring face */
  faceSector: Int32Array;
}

/** Limit normals of vertex v, one per sector (sectors are separated by sharp edges). */
export function vertexSectorNormals(topo: Topology, P: ArrayLike<Vec3>, v: number): VertexSectorNormals {
  const ring = vertexRing(topo, v);
  const kind = topo.vKind[v];
  const V = P[v];
  const nFaces = ring.faces.length;
  const faceSector = new Int32Array(nFaces);
  const all = Array.from({ length: nFaces }, (_, i) => i);
  if (nFaces === 0) return { ring, sectors: [], normals: [], faceSector };
  const finish = (n: Vec3, faceIdx: number[]): Vec3 => {
    let r = n;
    const l = length(r);
    let ref = 0;
    for (const nb of ring.nbr) ref = Math.max(ref, length(sub(P[nb], V)));
    if (!(l > 1e-12 * ref * ref)) r = fanNormal(topo, P, ring, faceIdx);
    const r2 = normalize(r);
    return length(r2) > 0 ? r2 : [0, 0, 1];
  };
  if (!ring.manifold) {
    return { ring, sectors: [{ nbrIdx: [], faceIdx: all, full: false }], normals: [finish(fanNormal(topo, P, ring, all), all)], faceSector };
  }
  if ((kind === VertexKind.Smooth || kind === VertexKind.Dart) && ring.closed) {
    const sector: RingSector = { nbrIdx: all.slice(), faceIdx: all, full: true };
    return { ring, sectors: [sector], normals: [finish(smoothLimitNormal(topo, P, ring), all)], faceSector };
  }
  const sectors = ringSectors(topo, ring);
  const normals: Vec3[] = [];
  sectors.forEach((sec, si) => {
    for (const fi of sec.faceIdx) faceSector[fi] = si;
    if (sec.full) {
      normals.push(finish(smoothLimitNormal(topo, P, ring), sec.faceIdx));
      return;
    }
    const E = sec.nbrIdx.map((i) => P[ring.nbr[i]]);
    const F = sec.faceIdx.map((i) => diagonalPoint(topo, P, ring.faceHe[i]));
    const n = kind === VertexKind.Crease ? creaseSectorNormal(V, E, F) : cornerSectorNormal(V, E, F);
    // guard against a flipped sector normal (e.g. degenerate cage geometry)
    const fan = fanNormal(topo, P, ring, sec.faceIdx);
    const nn = dot(n, fan) < 0 && length(fan) > 0 ? fan : n;
    normals.push(finish(nn, sec.faceIdx));
  });
  return { ring, sectors, normals, faceSector };
}

/** Normal of the limit surface at every vertex (crease vertices: average of their sector normals). */
export function limitNormalsTopo(topo: Topology, P: ArrayLike<Vec3>): Vec3[] {
  const out: Vec3[] = new Array(topo.nv);
  for (let v = 0; v < topo.nv; v++) {
    const s = vertexSectorNormals(topo, P, v);
    if (s.normals.length === 0) out[v] = [0, 0, 1];
    else if (s.normals.length === 1) out[v] = s.normals[0];
    else {
      let n: Vec3 = [0, 0, 0];
      for (const x of s.normals) n = add(n, x);
      const nn = normalize(n);
      out[v] = length(nn) > 0 ? nn : s.normals[0];
    }
  }
  return out;
}

/** Limit-surface unit normal at every cage vertex. */
export function limitNormals(qm: QuadMesh): Vec3[] {
  if (isAllQuads(qm)) return limitNormalsTopo(buildTopology(qm), qm.vertices);
  const r = refineMesh(qm, false);
  const topo1 = buildTopology(r.mesh);
  return limitNormalsTopo(topo1, r.mesh.vertices).slice(0, qm.vertices.length);
}

// ---------------------------------------------------------------------------
// Limit mesh for display
// ---------------------------------------------------------------------------

export interface LimitMeshOptions {
  /** duplicate vertices along crease/boundary sectors so creases render sharp (default true) */
  splitCreases?: boolean;
}

export interface LimitMeshResult {
  mesh: MeshData;
  /** for every output vertex the index of the subdivision sample it comes from (see subdivisionStencils) */
  sampleOfVertex: Int32Array;
  /** cage face of every output triangle */
  faceOfTriangle: Int32Array;
  /** the refined control mesh (level `levels`) */
  refined: QuadMesh;
}

/** Subdivide `levels` times and return the triangulated limit surface with exact limit positions/normals. */
export function limitMeshWithMap(qm: QuadMesh, levels = 3, opts: LimitMeshOptions = {}): LimitMeshResult {
  const L = Math.max(0, Math.min(8, Math.round(levels)));
  let mesh: QuadMesh = { vertices: qm.vertices, faces: qm.faces, creaseEdges: qm.creaseEdges };
  let root = Int32Array.from({ length: qm.faces.length }, (_, i) => i);
  for (let l = 0; l < L; l++) {
    const r = refineMesh(mesh, false);
    const nr = new Int32Array(r.refinement.faceParent.length);
    for (let i = 0; i < nr.length; i++) nr[i] = root[r.refinement.faceParent[i]];
    root = nr;
    mesh = r.mesh;
  }
  const topo = buildTopology(mesh);
  const nv = mesh.vertices.length;
  let LP: Vec3[];
  const vertexNormal: Vec3[] = new Array(nv);
  // extra (duplicated) vertices for crease sectors
  const extraPos: Vec3[] = [];
  const extraNrm: Vec3[] = [];
  const extraSrc: number[] = [];
  // corner (face, k) -> output vertex for split sectors
  const cornerOverride = new Map<number, number>(); // key: half-edge index
  const quads = isAllQuads(mesh);
  if (!quads) {
    LP = limitPoints(mesh);
    const N = limitNormals(mesh);
    for (let v = 0; v < nv; v++) vertexNormal[v] = N[v];
  } else {
    LP = csrMulVec3(limitMatrix(topo), mesh.vertices);
    const split = opts.splitCreases !== false;
    for (let v = 0; v < nv; v++) {
      const s = vertexSectorNormals(topo, mesh.vertices, v);
      if (s.normals.length === 0) {
        vertexNormal[v] = [0, 0, 1];
        continue;
      }
      vertexNormal[v] = s.normals[0];
      if (s.normals.length > 1) {
        if (split) {
          const outIdx: number[] = [v];
          for (let si = 1; si < s.normals.length; si++) {
            outIdx.push(nv + extraPos.length);
            extraPos.push(LP[v]);
            extraNrm.push(s.normals[si]);
            extraSrc.push(v);
          }
          for (let i = 0; i < s.ring.faces.length; i++) {
            const si = s.faceSector[i];
            if (si > 0) cornerOverride.set(s.ring.faceHe[i], outIdx[si]);
          }
        } else {
          let n: Vec3 = [0, 0, 0];
          for (const x of s.normals) n = add(n, x);
          vertexNormal[v] = normalize(n);
        }
      }
    }
  }
  const total = nv + extraPos.length;
  const positions = new Float32Array(3 * total);
  const normals = new Float32Array(3 * total);
  const sampleOfVertex = new Int32Array(total);
  for (let v = 0; v < nv; v++) {
    positions.set(LP[v], 3 * v);
    normals.set(vertexNormal[v], 3 * v);
    sampleOfVertex[v] = v;
  }
  for (let i = 0; i < extraPos.length; i++) {
    positions.set(extraPos[i], 3 * (nv + i));
    normals.set(extraNrm[i], 3 * (nv + i));
    sampleOfVertex[nv + i] = extraSrc[i];
  }
  const idx: number[] = [];
  const tri: number[] = [];
  for (let f = 0; f < mesh.faces.length; f++) {
    if (!topo.faceValid[f]) continue;
    const s = topo.faceStart[f], m = topo.faceStart[f + 1] - s;
    const vid = (k: number) => cornerOverride.get(s + k) ?? topo.heV[s + k];
    if (m === 4) {
      const a = topo.heV[s], b = topo.heV[s + 1], c = topo.heV[s + 2], d = topo.heV[s + 3];
      const dac = length(sub(LP[a], LP[c])), dbd = length(sub(LP[b], LP[d]));
      if (dac <= dbd) idx.push(vid(0), vid(1), vid(2), vid(0), vid(2), vid(3));
      else idx.push(vid(0), vid(1), vid(3), vid(1), vid(2), vid(3));
      tri.push(root[f], root[f]);
    } else {
      for (let k = 1; k + 1 < m; k++) {
        idx.push(vid(0), vid(k), vid(k + 1));
        tri.push(root[f]);
      }
    }
  }
  return {
    mesh: { positions, indices: Uint32Array.from(idx), normals },
    sampleOfVertex,
    faceOfTriangle: Int32Array.from(tri),
    refined: mesh,
  };
}

/** Triangulated limit surface after `levels` subdivisions (positions + normals on the limit surface). */
export function limitMesh(qm: QuadMesh, levels = 3, opts: LimitMeshOptions = {}): MeshData {
  return limitMeshWithMap(qm, levels, opts).mesh;
}

// ---------------------------------------------------------------------------
// Linear stencils: control vertices -> limit samples
// ---------------------------------------------------------------------------

export interface SubdivisionStencils {
  /** (samples x control vertices): limit position of sample s = sum_j rows[s, j] * P[j] */
  rows: CsrMatrix;
  /** cage face each sample is attributed to (-1 for isolated vertices) */
  faceOfSample: Int32Array;
  /**
   * (u, v) per sample inside its cage face: quad faces use [0,1]^2 with (0,0) at face vertex 0,
   * u along edge 0->1 and v along edge 0->3. For non-quad faces the uv refers to the sub-quad
   * at corner `cornerOfSample` ((0,0) at the corner vertex, u towards the next vertex).
   */
  uvOfSample: Float32Array;
  /** sub-quad (corner) index for samples of non-quad cage faces, -1 for quads */
  cornerOfSample: Int8Array;
  /** faces of the refined mesh in sample indices (all quads when levels >= 1) */
  faces: number[][];
  /** 1 if the sample lies on the cage boundary (on a limit boundary curve) */
  boundarySample: Uint8Array;
  levels: number;
}

/**
 * Sparse linear map from cage vertices to the limit-surface points of the `levels`-times refined
 * mesh (one sample per refined vertex; sample i == vertex i of `limitMesh(qm, levels)` before
 * crease splitting). Samples 0..nv-1 are the limit points of the cage vertices.
 */
export function subdivisionStencils(qm: QuadMesh, levels = 2): SubdivisionStencils {
  const L = Math.max(0, Math.min(5, Math.round(levels)));
  let topo = buildTopology(qm);
  let faces = qm.faces;
  let total: CsrMatrix | null = null;
  const rootQuad = qm.faces.map((f) => f.length === 4);
  let faceRoot = Int32Array.from({ length: faces.length }, (_, i) => i);
  let faceCornerArr = new Int8Array(faces.length).fill(-1);
  let faceUV = new Float64Array(faces.length * 8);
  const unit = [0, 0, 1, 0, 1, 1, 0, 1];
  for (let f = 0; f < faces.length; f++) if (rootQuad[f]) faceUV.set(unit, 8 * f);
  for (let l = 0; l < L; l++) {
    const r = refineTopology(topo);
    total = total ? csrMultiply(r.S, total) : r.S;
    const nfNew = r.faces.length;
    const nRoot = new Int32Array(nfNew);
    const nCorner = new Int8Array(nfNew).fill(-1);
    const nUV = new Float64Array(nfNew * 8);
    for (let i = 0; i < nfNew; i++) {
      if (!r.faces[i]) continue;
      const f = r.faceParent[i], k = r.faceCorner[i];
      nRoot[i] = faceRoot[f];
      if (l === 0 && !rootQuad[f]) {
        nCorner[i] = k;
        nUV.set(unit, 8 * i);
      } else {
        nCorner[i] = faceCornerArr[f];
        const o = 8 * f;
        const cu = (j: number) => faceUV[o + 2 * ((j + 4) % 4)];
        const cv = (j: number) => faceUV[o + 2 * ((j + 4) % 4) + 1];
        const mu = (cu(0) + cu(1) + cu(2) + cu(3)) / 4, mv = (cv(0) + cv(1) + cv(2) + cv(3)) / 4;
        const q = 8 * i;
        nUV[q] = cu(k);
        nUV[q + 1] = cv(k);
        nUV[q + 2] = (cu(k) + cu(k + 1)) / 2;
        nUV[q + 3] = (cv(k) + cv(k + 1)) / 2;
        nUV[q + 4] = mu;
        nUV[q + 5] = mv;
        nUV[q + 6] = (cu(k - 1) + cu(k)) / 2;
        nUV[q + 7] = (cv(k - 1) + cv(k)) / 2;
      }
    }
    faceRoot = nRoot;
    faceCornerArr = nCorner;
    faceUV = nUV;
    faces = r.faces;
    topo = buildTopologyRaw(r.nVertices, faces, r.creaseEdges);
  }
  let rows: CsrMatrix;
  if (L === 0 && !isAllQuads(qm)) rows = limitStencilMatrix(qm);
  else {
    const lim = limitMatrix(topo);
    rows = total ? csrMultiply(lim, total) : lim;
  }
  const ns = rows.nRows;
  const faceOfSample = new Int32Array(ns).fill(-1);
  const uvOfSample = new Float32Array(2 * ns);
  const cornerOfSample = new Int8Array(ns).fill(-1);
  for (let f = 0; f < faces.length; f++) {
    const face = faces[f];
    if (!topo.faceValid[f]) continue;
    for (let k = 0; k < face.length; k++) {
      const v = face[k];
      if (faceOfSample[v] !== -1) continue;
      faceOfSample[v] = faceRoot[f];
      if (L === 0 && !rootQuad[f]) {
        cornerOfSample[v] = k;
        uvOfSample[2 * v] = 0;
        uvOfSample[2 * v + 1] = 0;
      } else {
        cornerOfSample[v] = faceCornerArr[f];
        uvOfSample[2 * v] = faceUV[8 * f + 2 * k];
        uvOfSample[2 * v + 1] = faceUV[8 * f + 2 * k + 1];
      }
    }
  }
  return { rows, faceOfSample, uvOfSample, cornerOfSample, faces, boundarySample: topo.vBoundary.slice(), levels: L };
}

/** Classification helper re-exported for convenience. */
export function vertexKinds(qm: QuadMesh): Uint8Array {
  return buildTopology(qm).vKind.slice();
}
