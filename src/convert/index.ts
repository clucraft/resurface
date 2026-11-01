/**
 * STL → CAD conversion ("Auto Convert"): segment the mesh into analytic surface regions and
 * derive a B-rep plan the CAD kernel turns into a solid (see OcctKernel.brepFromPlan).
 * Unrecognised triangles are kept as faceted faces so the result always closes.
 */
import type { MeshData, ProgressFn } from "../core/types";
import { buildTriangleNeighbors, meshBounds } from "../mesh/topology";
import { weldVerticesWithMap } from "../mesh/repair";
import { buildBrepPlan, type BrepPlan, type FaceKind } from "./brepPlan";
import { segmentMesh, type Region } from "./segment";
import type { SurfKind } from "./surfaces";

export type { BrepPlan, FaceKind } from "./brepPlan";
export type { Region } from "./segment";
export type { ConvSurface, SurfKind, TorusDef } from "./surfaces";

export interface ConvertOptions {
  /** max vertex distance from a fitted surface (mm); default follows the model size */
  tolerance?: number;
  /** base facet-normal tolerance (deg), default 2 */
  angleTolDeg?: number;
  kinds?: SurfKind[];
  onProgress?: ProgressFn;
}

export interface ConvertStats {
  triangles: number;
  regions: number;
  /** regions per surface kind */
  kinds: Partial<Record<SurfKind, number>>;
  facetedTriangles: number;
  /** share of the surface area covered by analytic surfaces (0..1) */
  coverage: number;
  /** largest vertex deviation over all regions (mm) */
  maxDeviation: number;
  tolerance: number;
  planFaces: number;
  planEdges: number;
}

export interface ConvertResult {
  plan: BrepPlan;
  /** region per triangle (-1 = faceted) */
  labels: Int32Array;
  regions: Region[];
  stats: ConvertStats;
}

/** Default fitting tolerance: a small fraction of the model size. */
export function defaultTolerance(mesh: MeshData): number {
  const b = meshBounds(mesh);
  const diag = Math.hypot(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]);
  return Math.max(diag * 2e-4, 1e-4);
}

/**
 * Weld vertices that coincide within float noise (CAD tessellators leave duplicated pole vertices)
 * and drop the triangles that collapse. Returns the original → cleaned triangle map.
 */
export function cleanMesh(mesh: MeshData, eps: number): { mesh: MeshData; triMap: Int32Array } {
  const nt = mesh.indices.length / 3;
  const identity = () => ({ mesh, triMap: Int32Array.from({ length: nt }, (_, i) => i) });
  const w = weldVerticesWithMap(mesh, eps);
  const I = mesh.indices, vm = w.vertexMap;
  const triMap = new Int32Array(nt).fill(-1);
  let k = 0;
  for (let t = 0; t < nt; t++) {
    const a = vm[I[3 * t]], b = vm[I[3 * t + 1]], c = vm[I[3 * t + 2]];
    if (a < 0 || b < 0 || c < 0 || a === b || b === c || a === c) continue;
    triMap[t] = k++;
  }
  // the welder keeps triangle order: verify before trusting the map
  if (k !== w.mesh.indices.length / 3) return identity();
  return { mesh: w.mesh, triMap };
}

export function convertMesh(input: MeshData, opts: ConvertOptions = {}): ConvertResult {
  if (!(input.indices.length / 3)) throw new Error("the mesh has no triangles");
  const tolerance = opts.tolerance ?? defaultTolerance(input);
  const { mesh, triMap } = cleanMesh(input, tolerance * 0.1);
  if (!(mesh.indices.length / 3)) throw new Error("the mesh has no triangles");
  const size = sizeOf(mesh);
  const progress = opts.onProgress;
  progress?.(0, "Building adjacency");
  const nbr = buildTriangleNeighbors(mesh);
  const seg = segmentMesh(mesh, {
    tolerance,
    angleTolDeg: opts.angleTolDeg ?? noiseAngleDeg(mesh, nbr),
    size,
    kinds: opts.kinds,
    neighbors: nbr,
    onProgress: progress ? (f, m) => progress(0.05 + 0.85 * f, m) : undefined,
  });
  progress?.(0.92, "Building edges");
  return finish(mesh, triMap, nbr, seg.labels, seg.regions, tolerance, size);
}

/**
 * Rebuild the plan after demoting some regions to faceted triangles (their faces could not be
 * built validly). `prev` is an earlier result for the same mesh and options.
 */
export function replanMesh(input: MeshData, prev: Pick<ConvertResult, "labels" | "regions" | "stats">, demoteRegions: number[]): ConvertResult {
  const tolerance = prev.stats.tolerance;
  const { mesh, triMap } = cleanMesh(input, tolerance * 0.1);
  const size = sizeOf(mesh);
  const demote = new Set(demoteRegions);
  const remap = new Int32Array(prev.regions.length).fill(-1);
  const regions: Region[] = [];
  prev.regions.forEach((r, i) => {
    if (demote.has(i)) return;
    remap[i] = regions.length;
    regions.push({ ...r, id: regions.length });
  });
  const labels = new Int32Array(mesh.indices.length / 3).fill(-1);
  for (let t = 0; t < triMap.length; t++) {
    const c = triMap[t];
    if (c >= 0 && prev.labels[t] >= 0) labels[c] = remap[prev.labels[t]];
  }
  return finish(mesh, triMap, buildTriangleNeighbors(mesh), labels, regions, tolerance, size);
}

/**
 * Facet-normal allowance. CAD exports have exact facets: many neighbouring triangles are exactly
 * coplanar (flat faces, strip diagonals), and 2° is plenty. On a scan almost no neighbours are
 * coplanar and the normals wobble with the noise: allow about the typical facet-to-facet angle.
 */
export function noiseAngleDeg(mesh: MeshData, nbr: Int32Array): number {
  const P = mesh.positions, I = mesh.indices;
  const nt = I.length / 3;
  const fn = (t: number, out: number[]) => {
    const a = 3 * I[3 * t], b = 3 * I[3 * t + 1], c = 3 * I[3 * t + 2];
    const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2];
    const vx = P[c] - P[a], vy = P[c + 1] - P[a + 1], vz = P[c + 2] - P[a + 2];
    const x = uy * vz - uz * vy, y = uz * vx - ux * vz, z = ux * vy - uy * vx;
    const l = Math.hypot(x, y, z) || 1;
    out[0] = x / l;
    out[1] = y / l;
    out[2] = z / l;
  };
  const angles: number[] = [];
  const n1 = [0, 0, 0], n2 = [0, 0, 0];
  const step = Math.max(1, Math.floor(nt / 20000));
  for (let t = 0; t < nt; t += step) {
    const u = nbr[3 * t];
    if (u < 0) continue;
    fn(t, n1);
    fn(u, n2);
    angles.push((Math.acos(Math.max(-1, Math.min(1, n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2]))) * 180) / Math.PI);
  }
  if (!angles.length) return 2;
  const coplanar = angles.filter((a) => a < 0.05).length / angles.length;
  if (coplanar > 0.15) return 2;
  angles.sort((x, y) => x - y);
  return Math.min(25, Math.max(2, 1.5 * angles[Math.floor(angles.length / 2)]));
}

function sizeOf(mesh: MeshData): number {
  const b = meshBounds(mesh);
  return Math.hypot(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]);
}

function finish(mesh: MeshData, triMap: Int32Array, nbr: Int32Array, cleanLabels: Int32Array, regions: Region[], tolerance: number, size: number): ConvertResult {
  const nt = mesh.indices.length / 3;
  const plan = buildBrepPlan(mesh, nbr, cleanLabels, regions, { tolerance, size });
  const kinds: Partial<Record<SurfKind, number>> = {};
  let covered = 0, total = 0, maxDev = 0;
  for (const r of regions) {
    kinds[r.surface.kind] = (kinds[r.surface.kind] ?? 0) + 1;
    covered += r.area;
    maxDev = Math.max(maxDev, r.maxDev);
  }
  let faceted = 0;
  const P = mesh.positions, I = mesh.indices;
  for (let t = 0; t < nt; t++) {
    const a = 3 * I[3 * t], bb = 3 * I[3 * t + 1], c = 3 * I[3 * t + 2];
    const ux = P[bb] - P[a], uy = P[bb + 1] - P[a + 1], uz = P[bb + 2] - P[a + 2];
    const vx = P[c] - P[a], vy = P[c + 1] - P[a + 1], vz = P[c + 2] - P[a + 2];
    const ar = Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx) / 2;
    total += ar;
    if (cleanLabels[t] < 0) faceted++;
  }
  // labels of the input triangles (collapsed ones: faceted)
  const labels = new Int32Array(triMap.length);
  for (let t = 0; t < triMap.length; t++) labels[t] = triMap[t] >= 0 ? cleanLabels[triMap[t]] : -1;
  return {
    plan,
    labels,
    regions,
    stats: {
      triangles: nt,
      regions: regions.length,
      kinds,
      facetedTriangles: faceted,
      coverage: total > 0 ? Math.min(1, covered / total) : 0,
      maxDeviation: maxDev,
      tolerance,
      planFaces: plan.faces.length,
      planEdges: plan.edges.length,
    },
  };
}

/** Regions whose faces the kernel could not build validly (to demote with replanMesh). */
export function regionsToDemote(plan: BrepPlan, built: { failedFaces: number[]; invalidFaces: number[] }): number[] {
  const out = new Set<number>();
  for (const f of [...built.failedFaces, ...built.invalidFaces]) {
    const r = plan.faces[f]?.region ?? -1;
    if (r >= 0) out.add(r);
  }
  return [...out];
}

export const FACE_KIND_LABEL: Record<FaceKind, string> = {
  plane: "Plane",
  cylinder: "Cylinder",
  cone: "Cone",
  sphere: "Sphere",
  torus: "Torus",
  faceted: "Faceted",
};
