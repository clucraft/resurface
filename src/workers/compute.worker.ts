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
  /** Signed distances from points[start..end) to a target triangle mesh (NaN beyond maxDistance). */
  async deviation({ points, start, end, target, maxDistance }: { points: Float32Array; start: number; end: number; target: MeshData; maxDistance: number }, progress) {
    const { MeshSpatialIndex } = await import("../mesh/spatial");
    const index = new MeshSpatialIndex(target);
    const n = end - start;
    const out = new Float32Array(n);
    const p: [number, number, number] = [0, 0, 0];
    for (let i = 0; i < n; i++) {
      const v = start + i;
      p[0] = points[3 * v];
      p[1] = points[3 * v + 1];
      p[2] = points[3 * v + 2];
      const hit = index.closestPoint(p, maxDistance);
      if (!hit) out[i] = NaN;
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
    return { cage: res.cage, patches: res.patches.length, rms: res.rms, maxDev: res.maxDev, quality: res.quality };
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
