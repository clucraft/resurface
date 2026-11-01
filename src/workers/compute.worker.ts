/// <reference lib="webworker" />
/**
 * Compute worker: runs named heavy jobs off the UI thread. Results are moved into shared
 * memory (when available) or transferred, so large meshes are never copied twice.
 */
import type { MeshData, ProgressFn } from "../core/types";
import { toShared, sharedMemoryAvailable } from "../app/shared";

type Job = (args: any, progress: ProgressFn) => unknown | Promise<unknown>;

const jobs: Record<string, Job> = {
  /** Parse a scan file and prepare the mesh (spatial order + normals) in one go. */
  async import({ name, data }: { name: string; data: ArrayBuffer }, progress) {
    const { importMeshFile } = await import("../mesh/io");
    const res = importMeshFile(name, data, (f, m) => progress(f * 0.8, m ?? "Parsing"));
    if (res.kind === "mesh") {
      progress(0.85, "Preparing mesh");
      const { prepareMeshData } = await import("../app/prepare");
      return { kind: "mesh", mesh: prepareMeshData(res.mesh) };
    }
    return res;
  },
  async sample({ id }: { id: "bracket" | "turned" | "organic" }) {
    const { makeSample } = await import("../samples/samples");
    const { prepareMeshData } = await import("../app/prepare");
    return prepareMeshData(makeSample(id));
  },
  /** dev/perf: large synthetic scan, prepared */
  async bigTest({ triangles }: { triangles: number }, progress) {
    const { bumpySphere } = await import("../dev/bigmesh");
    const { prepareMeshData } = await import("../app/prepare");
    const t0 = performance.now();
    const m = bumpySphere(triangles);
    const t1 = performance.now();
    progress(0.5, "Preparing");
    const p = prepareMeshData(m);
    const t2 = performance.now();
    return { mesh: p, timings: { generate: t1 - t0, prepare: t2 - t1 } };
  },
  async prepare({ mesh, keepOrder }: { mesh: MeshData; keepOrder?: boolean }) {
    const { prepareMeshData } = await import("../app/prepare");
    return prepareMeshData(mesh, { keepOrder });
  },
  async decimate({ mesh, targetTriangles, fast }: { mesh: MeshData; targetTriangles: number; fast?: boolean }, progress) {
    const d = await import("../mesh/decimate");
    const tris = mesh.indices.length / 3;
    if (fast && tris > targetTriangles * 2 && d.decimateClustering) return d.decimateClustering(mesh, { targetTriangles, onProgress: progress });
    return d.decimate(mesh, { targetTriangles, onProgress: progress });
  },
  async smooth({ mesh, iterations, lambda, mu, mask }: { mesh: MeshData; iterations: number; lambda: number; mu: number; mask?: Uint8Array }, progress) {
    const { smoothTaubin } = await import("../mesh/smooth");
    return smoothTaubin(mesh, { iterations, lambda, mu, mask, onProgress: progress });
  },
  async clip({ mesh, plane, keep, cut }: { mesh: MeshData; plane: import("../core/types").PlaneDef; keep: "positive" | "negative"; cut: boolean }) {
    const { clipByPlane } = await import("../mesh/clip");
    return clipByPlane(mesh, plane, keep, cut);
  },
  async components({ mesh }: { mesh: MeshData }, progress) {
    const { connectedComponents } = await import("../mesh/topology");
    const c = connectedComponents(mesh, progress);
    return { count: c.count, sizes: c.componentSizes };
  },
  /**
   * Signed distances from points[start..end) to a target triangle mesh (NaN beyond maxDistance).
   * For closed solids the sign comes from a ray-parity inside test (robust at edges/corners);
   * for open surfaces from the face normal at the closest point.
   */
  async deviation(
    { points, start, end, target, maxDistance, closed }: { points: Float32Array; start: number; end: number; target: MeshData; maxDistance: number; closed?: boolean },
    progress,
  ) {
    const { MeshSpatialIndex } = await import("../mesh/spatial");
    const index = new MeshSpatialIndex(target);
    const n = end - start;
    const out = new Float32Array(n);
    const p: [number, number, number] = [0, 0, 0];
    // skewed direction avoids grazing the axis-aligned edges typical of CAD parts
    const dir: [number, number, number] = [0.5773, 0.5779, 0.5767];
    const inside = (q: [number, number, number]): boolean => {
      let o: [number, number, number] = [q[0], q[1], q[2]];
      let count = 0;
      for (let k = 0; k < 64; k++) {
        const h = index.raycast(o, dir);
        if (!h) break;
        count++;
        o = [h.point[0] + dir[0] * 1e-4, h.point[1] + dir[1] * 1e-4, h.point[2] + dir[2] * 1e-4];
      }
      return (count & 1) === 1;
    };
    for (let i = 0; i < n; i++) {
      const v = start + i;
      p[0] = points[3 * v];
      p[1] = points[3 * v + 1];
      p[2] = points[3 * v + 2];
      const hit = index.closestPoint(p, maxDistance);
      if (!hit || hit.distance > maxDistance) out[i] = NaN;
      else if (closed) out[i] = inside(p) ? -hit.distance : hit.distance;
      else {
        const s = (p[0] - hit.point[0]) * hit.normal[0] + (p[1] - hit.point[1]) * hit.normal[1] + (p[2] - hit.point[2]) * hit.normal[2];
        out[i] = s >= 0 ? hit.distance : -hit.distance;
      }
      if ((i & 16383) === 0) progress(i / n);
    }
    index.dispose();
    return out;
  },
  async curvature({ mesh, kind }: { mesh: MeshData; kind: "mean" | "signedMean" | "rms" | "max" }) {
    const { vertexCurvature, smoothVertexScalar } = await import("../mesh/curvature");
    return smoothVertexScalar(mesh, vertexCurvature(mesh, { kind }), 1);
  },
  /**
   * One-click autosurface (optionally restricted to a triangle mask): decimate to a coarse
   * triangle mesh, build a quad cage, snap & fit it to the scan.
   */
  async autosurface(
    { mesh, mask, coarseTriangles, opts }: { mesh: MeshData; mask?: Uint8Array; coarseTriangles: number; opts: Record<string, unknown> },
    progress,
  ) {
    const { compactMesh } = await import("../mesh/topology");
    const { decimate } = await import("../mesh/decimate");
    const { MeshSpatialIndex } = await import("../mesh/spatial");
    const { autosurface } = await import("../freeform/autosurface");
    progress(0, "Preparing target");
    let target = mask ? compactMesh(mesh, mask) : mesh;
    if (target.indices.length / 3 > 400_000) target = decimate(target, { targetTriangles: 400_000 });
    progress(0.05, "Building coarse layout");
    const coarse = decimate(target, { targetTriangles: Math.max(8, coarseTriangles) });
    const index = new MeshSpatialIndex(target);
    const res = autosurface(coarse, index.asQuery(), { ...opts, onProgress: (f, m) => progress(0.1 + 0.9 * f, m) });
    index.dispose();
    // report the face count the CAD body will actually have (regular regions are merged)
    let patches = res.patches.length;
    try {
      const { mergeRegularRegions } = await import("../freeform/patches");
      patches = mergeRegularRegions(res.cage).surfaces.length;
    } catch {
      /* keep the bicubic count */
    }
    return { cage: res.cage, patches, rms: res.rms, maxDev: res.maxDev, quality: res.quality };
  },
  /**
   * STL → CAD (Auto Convert): segment the mesh into analytic surfaces and build the B-rep plan.
   * With `prev` + `demote`, rebuild the plan with those regions turned into facets instead.
   */
  async convert(
    { mesh, tolerance, prev, demote }: { mesh: MeshData; tolerance?: number; prev?: import("../convert").ConvertResult; demote?: number[] },
    progress,
  ) {
    const { convertMesh, replanMesh } = await import("../convert");
    if (prev && demote) return replanMesh(mesh, prev, demote);
    return convertMesh(mesh, { tolerance, onProgress: progress });
  },
  /** Fit an existing cage to the scan (optionally to a masked region only). */
  async fitCage({ mesh, mask, cage, opts }: { mesh: MeshData; mask?: Uint8Array; cage: import("../freeform/quadMesh").QuadMesh; opts: Record<string, unknown> }, progress) {
    const { compactMesh } = await import("../mesh/topology");
    const { decimate } = await import("../mesh/decimate");
    const { MeshSpatialIndex } = await import("../mesh/spatial");
    const { fitCageToTarget } = await import("../freeform/fit");
    let target = mask ? compactMesh(mesh, mask) : mesh;
    if (target.indices.length / 3 > 600_000) target = decimate(target, { targetTriangles: 600_000 });
    const index = new MeshSpatialIndex(target);
    const res = fitCageToTarget(cage, index.asQuery(), { ...opts, onProgress: progress });
    index.dispose();
    return { cage: res.cage, rms: res.rms, maxDev: res.maxDev };
  },
  /** Build the BVH off the UI thread; buffers are shared (SAB) or transferred back. */
  async spatialIndex({ mesh }: { mesh: MeshData }) {
    const { MeshSpatialIndex } = await import("../mesh/spatial");
    const index = new MeshSpatialIndex(mesh, { shared: sharedMemoryAvailable() });
    return index.serialize();
  },
  async neighbors({ mesh }: { mesh: MeshData }) {
    const { buildTriangleNeighbors } = await import("../mesh/topology");
    return buildTriangleNeighbors(mesh);
  },
  async boundaryLoops({ mesh }: { mesh: MeshData }) {
    const { findBoundaryLoops } = await import("../mesh/topology");
    return findBoundaryLoops(mesh);
  },
  async fillHoles({ mesh, opts }: { mesh: MeshData; opts: Record<string, unknown> }) {
    const { fillHoles } = await import("../mesh/holes");
    return fillHoles(mesh, opts as never).mesh;
  },
  async removeSmallComponents({ mesh, minTriangles, degenerate }: { mesh: MeshData; minTriangles: number; degenerate?: boolean }, progress) {
    const { removeSmallComponents, removeDegenerateTriangles } = await import("../mesh/repair");
    let m = removeSmallComponents(mesh, minTriangles, { onProgress: progress });
    if (degenerate) m = removeDegenerateTriangles(m);
    return m;
  },
  async defeature({ mesh, mask }: { mesh: MeshData; mask: Uint8Array }) {
    const { defeature } = await import("../mesh/defeature");
    return defeature(mesh, mask).mesh;
  },
};

/** Move typed arrays inside a result to shared memory, or collect transferables. */
function prepareResult(value: unknown, transfer: Transferable[], depth = 0): unknown {
  if (depth > 4 || value === null || typeof value !== "object") return value;
  if (value instanceof ArrayBuffer) {
    if (!transfer.includes(value)) transfer.push(value);
    return value;
  }
  if (typeof SharedArrayBuffer !== "undefined" && value instanceof SharedArrayBuffer) return value;
  if (ArrayBuffer.isView(value)) {
    const ta = value as Float32Array;
    if (ta.buffer instanceof SharedArrayBuffer) return ta;
    if (sharedMemoryAvailable() && ta.byteLength > 1 << 16) return toShared(ta as never);
    if (ta.byteOffset === 0 && ta.byteLength === ta.buffer.byteLength && !transfer.includes(ta.buffer)) transfer.push(ta.buffer);
    return ta;
  }
  if (Array.isArray(value)) return value.map((v) => prepareResult(v, transfer, depth + 1));
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(value as Record<string, unknown>)) out[k] = prepareResult(v, transfer, depth + 1);
  return out;
}

self.onmessage = async (ev: MessageEvent) => {
  const { id, job, args } = ev.data as { id: number; job: string; args: unknown };
  let last = 0;
  const progress: ProgressFn = (fraction, message) => {
    const now = performance.now();
    if (now - last < 100 && fraction < 1) return;
    last = now;
    (self as unknown as Worker).postMessage({ id, type: "progress", fraction, message });
  };
  try {
    const fn = jobs[job];
    if (!fn) throw new Error(`Unknown compute job "${job}"`);
    const transfer: Transferable[] = [];
    let value = await fn(args, progress);
    // optional post-processing for mesh results: "reorder" = spatial sort + normals, "normals" = normals only
    const prep = (args as { prepare?: "reorder" | "normals" })?.prepare;
    if (prep && value && typeof value === "object" && "positions" in (value as object) && "indices" in (value as object)) {
      const { prepareMeshData } = await import("../app/prepare");
      const m = value as MeshData;
      value = prepareMeshData({ ...m, normals: undefined }, { keepOrder: prep === "normals" });
    }
    const result = prepareResult(value, transfer);
    (self as unknown as Worker).postMessage({ id, type: "done", result }, transfer);
  } catch (err) {
    console.error(err);
    (self as unknown as Worker).postMessage({ id, type: "done", error: err instanceof Error ? err.message : String(err) });
  }
};
