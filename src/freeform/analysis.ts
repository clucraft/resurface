/**
 * Cage quality metrics and surface-to-scan deviation analysis (for colour maps).
 */
import type { BSplineSurfaceDef, ClosestPointQuery, Vec3 } from "../core/types";
import { subdivisionStencils } from "./catmullClark";
import { evaluateSurface, surfaceDomain } from "./patches";
import { buildTopology, ringSectors, valenceOf, VertexKind, vertexRing, type QuadMesh, type Topology } from "./quadMesh";
import { csrMulVec3 } from "./sparse";

/**
 * True if the vertex makes the surface around it non-B-spline: interior valence != 4, crease
 * sectors with != 2 faces, corner sectors with != 1 face, darts, non-manifold vertices.
 */
export function isExtraordinaryVertex(topo: Topology, v: number): boolean {
  const nOut = topo.vOutStart[v + 1] - topo.vOutStart[v];
  if (nOut === 0) return false;
  if (!topo.vManifold[v]) return true;
  const kind = topo.vKind[v];
  if (kind === VertexKind.Smooth) return valenceOf(topo, v) !== 4;
  if (kind === VertexKind.Dart) return true;
  const sectors = ringSectors(topo, vertexRing(topo, v));
  const want = kind === VertexKind.Crease ? 2 : 1;
  return sectors.some((s) => s.faceIdx.length !== want);
}

export interface CageQuality {
  vertexCount: number;
  faceCount: number;
  quadCount: number;
  triangleCount: number;
  /** faces with more than 4 vertices */
  polygonCount: number;
  nonQuadCount: number;
  extraordinaryCount: number;
  extraordinaryVertices: number[];
  boundaryVertexCount: number;
  creaseEdgeCount: number;
  /** smallest / largest face corner angle in degrees */
  minAngleDeg: number;
  maxAngleDeg: number;
  /** mean |corner angle - 90°| over quad corners, degrees */
  meanQuadAngleDeviationDeg: number;
  /** number of vertices per valence (incident edge count) */
  valenceHistogram: Record<number, number>;
}

export function cageQuality(qm: QuadMesh): CageQuality {
  const topo = buildTopology(qm);
  let quadCount = 0, triangleCount = 0, polygonCount = 0;
  for (const f of qm.faces) {
    if (f.length === 4) quadCount++;
    else if (f.length === 3) triangleCount++;
    else if (f.length > 4) polygonCount++;
  }
  const extraordinaryVertices: number[] = [];
  const valenceHistogram: Record<number, number> = {};
  let boundaryVertexCount = 0;
  for (let v = 0; v < topo.nv; v++) {
    if (topo.vOutStart[v + 1] === topo.vOutStart[v]) continue;
    if (isExtraordinaryVertex(topo, v)) extraordinaryVertices.push(v);
    const val = valenceOf(topo, v);
    valenceHistogram[val] = (valenceHistogram[val] ?? 0) + 1;
    if (topo.vBoundary[v]) boundaryVertexCount++;
  }
  let creaseEdgeCount = 0;
  for (let e = 0; e < topo.ne; e++) if (topo.edgeSharp[e] === 2) creaseEdgeCount++;
  let minA = Infinity, maxA = -Infinity, devSum = 0, devCnt = 0;
  for (const f of qm.faces) {
    const n = f.length;
    if (n < 3) continue;
    for (let k = 0; k < n; k++) {
      const p = qm.vertices[f[k]], a = qm.vertices[f[(k + 1) % n]], b = qm.vertices[f[(k + n - 1) % n]];
      const ux = a[0] - p[0], uy = a[1] - p[1], uz = a[2] - p[2];
      const wx = b[0] - p[0], wy = b[1] - p[1], wz = b[2] - p[2];
      const lu = Math.hypot(ux, uy, uz), lw = Math.hypot(wx, wy, wz);
      if (lu < 1e-300 || lw < 1e-300) {
        minA = 0;
        continue;
      }
      const c = Math.max(-1, Math.min(1, (ux * wx + uy * wy + uz * wz) / (lu * lw)));
      const ang = (Math.acos(c) * 180) / Math.PI;
      minA = Math.min(minA, ang);
      maxA = Math.max(maxA, ang);
      if (n === 4) {
        devSum += Math.abs(ang - 90);
        devCnt++;
      }
    }
  }
  return {
    vertexCount: qm.vertices.length,
    faceCount: qm.faces.length,
    quadCount,
    triangleCount,
    polygonCount,
    nonQuadCount: qm.faces.length - quadCount,
    extraordinaryCount: extraordinaryVertices.length,
    extraordinaryVertices,
    boundaryVertexCount,
    creaseEdgeCount,
    minAngleDeg: Number.isFinite(minA) ? minA : 0,
    maxAngleDeg: Number.isFinite(maxA) ? maxA : 0,
    meanQuadAngleDeviationDeg: devCnt ? devSum / devCnt : 0,
    valenceHistogram,
  };
}

export interface DeviationResult {
  /** signed distance per sample (positive = surface outside the target along the target normal); NaN = no hit */
  distances: Float32Array;
  /** limit-surface sample positions, packed xyz (== limitMesh(qm, levels) vertices before crease splitting) */
  positions: Float32Array;
  faceOfSample: Int32Array;
  rms: number;
  maxAbs: number;
  mean: number;
  /** samples without a target hit */
  misses: number;
}

/** Deviation of the limit surface (sampled at subdivision level `levels`) from the target. */
export function surfaceDeviation(qm: QuadMesh, target: ClosestPointQuery, levels = 2, maxDistance?: number): DeviationResult {
  const st = subdivisionStencils(qm, levels);
  const Q = csrMulVec3(st.rows, qm.vertices);
  return deviationOfPoints(Q, target, maxDistance, st.faceOfSample);
}

function deviationOfPoints(Q: Vec3[], target: ClosestPointQuery, maxDistance: number | undefined, faceOfSample: Int32Array): DeviationResult {
  const n = Q.length;
  const distances = new Float32Array(n);
  const positions = new Float32Array(3 * n);
  let sum2 = 0, sum = 0, cnt = 0, maxAbs = 0, misses = 0;
  for (let i = 0; i < n; i++) {
    const q = Q[i];
    positions[3 * i] = q[0];
    positions[3 * i + 1] = q[1];
    positions[3 * i + 2] = q[2];
    const hit = target(q, maxDistance);
    if (!hit) {
      distances[i] = NaN;
      misses++;
      continue;
    }
    const dx = q[0] - hit.point[0], dy = q[1] - hit.point[1], dz = q[2] - hit.point[2];
    const d = Math.hypot(dx, dy, dz);
    const side = dx * hit.normal[0] + dy * hit.normal[1] + dz * hit.normal[2];
    const sd = side < 0 ? -d : d;
    distances[i] = sd;
    sum2 += d * d;
    sum += sd;
    cnt++;
    if (d > maxAbs) maxAbs = d;
  }
  return {
    distances,
    positions,
    faceOfSample,
    rms: cnt ? Math.sqrt(sum2 / cnt) : NaN,
    maxAbs: cnt ? maxAbs : NaN,
    mean: cnt ? sum / cnt : NaN,
    misses,
  };
}

export interface PatchDeviationResult {
  rms: number;
  maxAbs: number;
  /** max |distance| per patch */
  perPatchMax: Float32Array;
}

/** Deviation of exported surfaces from the target, sampled on a (samplesPerSide+1)^2 grid per surface. */
export function patchDeviation(patches: BSplineSurfaceDef[], target: ClosestPointQuery, samplesPerSide = 8): PatchDeviationResult {
  const perPatchMax = new Float32Array(patches.length);
  let sum2 = 0, cnt = 0, maxAbs = 0;
  patches.forEach((s, pi) => {
    const dom = surfaceDomain(s);
    let pm = 0;
    for (let i = 0; i <= samplesPerSide; i++)
      for (let j = 0; j <= samplesPerSide; j++) {
        const u = dom.u0 + ((dom.u1 - dom.u0) * i) / samplesPerSide;
        const v = dom.v0 + ((dom.v1 - dom.v0) * j) / samplesPerSide;
        const p = evaluateSurface(s, u, v);
        const hit = target(p);
        if (!hit) continue;
        const d = Math.hypot(p[0] - hit.point[0], p[1] - hit.point[1], p[2] - hit.point[2]);
        sum2 += d * d;
        cnt++;
        if (d > pm) pm = d;
      }
    perPatchMax[pi] = pm;
    if (pm > maxAbs) maxAbs = pm;
  });
  return { rms: cnt ? Math.sqrt(sum2 / cnt) : NaN, maxAbs, perPatchMax };
}

/**
 * Blue (-tol) -> green (0) -> red (+tol) colour map for signed deviations, RGB 0..255 per value
 * (grey for NaN). Values beyond +-tol saturate.
 */
export function deviationColors(distances: ArrayLike<number>, tolerance: number): Uint8Array {
  const out = new Uint8Array(distances.length * 3);
  const tol = Math.max(tolerance, 1e-300);
  for (let i = 0; i < distances.length; i++) {
    const d = distances[i];
    let r = 128, g = 128, b = 128;
    if (Number.isFinite(d)) {
      const t = Math.max(-1, Math.min(1, d / tol));
      if (t >= 0) {
        r = Math.round(255 * t);
        g = Math.round(255 * (1 - t));
        b = 0;
      } else {
        r = 0;
        g = Math.round(255 * (1 + t));
        b = Math.round(-255 * t);
      }
    }
    out[3 * i] = r;
    out[3 * i + 1] = g;
    out[3 * i + 2] = b;
  }
  return out;
}
