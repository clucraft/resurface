/**
 * Per-mesh runtime caches that are not part of the document: render chunks, spatial index
 * (BVH), triangle selection mask and the LOD proxy. Rebuilt whenever the entity's MeshData
 * object changes (meshes are immutable snapshots).
 */
import type { MeshData, Vec3 } from "../core/types";
import { MeshSpatialIndex, buildChunks, type SerializedSpatialIndex } from "../mesh/spatial";
import { compute } from "./compute";
import { buildTriangleNeighbors } from "../mesh/topology";
import type { MeshChunk, MeshView } from "../viewer/MeshView";
import { prepareMeshData } from "./prepare";
import { nextFrame } from "../core/frame";

export const CHUNK_TRIANGLES = 262_144;

const prepared = new WeakSet<MeshData>();

/** Prepare a mesh for display/interaction (see prepareMeshData); idempotent per MeshData object. */
export function prepareMesh(mesh: MeshData, opts: { keepOrder?: boolean } = {}): MeshData {
  if (prepared.has(mesh)) return mesh;
  const out = prepareMeshData(mesh, opts);
  prepared.add(out);
  return out;
}

export function markPrepared(mesh: MeshData): void {
  prepared.add(mesh);
}

export class MeshRuntime {
  readonly chunks: MeshChunk[];
  private _spatial: MeshSpatialIndex | null = null;
  selection: Uint8Array;
  selectionCount = 0;
  view: MeshView | null = null;
  proxy: MeshData | null = null;
  private dirtyChunks = new Set<number>();
  private flushQueued = false;
  onSelectionChanged: (() => void) | null = null;

  /** above this size the BVH / adjacency are built in a worker right after the mesh appears */
  static BACKGROUND_THRESHOLD = 600_000;
  private disposed = false;
  /** resolves when background indexing has finished (or immediately for small meshes) */
  readonly indexed: Promise<void>;
  onIndexed: (() => void) | null = null;

  constructor(
    readonly entityId: string,
    readonly mesh: MeshData,
  ) {
    this.chunks = buildChunks(mesh, CHUNK_TRIANGLES) as MeshChunk[];
    this.selection = new Uint8Array(mesh.indices.length / 3);
    this.indexed = this.triangleCount >= MeshRuntime.BACKGROUND_THRESHOLD ? this.indexInBackground() : Promise.resolve();
  }

  /** Build the BVH and triangle adjacency in compute workers (UI stays responsive). */
  private async indexInBackground(): Promise<void> {
    const [bvh, nbr] = await Promise.allSettled([
      compute.run<SerializedSpatialIndex>("spatialIndex", { mesh: this.mesh }),
      compute.run<Int32Array>("neighbors", { mesh: this.mesh }),
    ]);
    if (this.disposed) return;
    if (bvh.status === "fulfilled" && !this._spatial) this._spatial = new MeshSpatialIndex(this.mesh, { serialized: bvh.value });
    if (nbr.status === "fulfilled" && !this._neighbors) this._neighbors = nbr.value;
    this.onIndexed?.();
  }

  get triangleCount(): number {
    return this.mesh.indices.length / 3;
  }

  get vertexCount(): number {
    return this.mesh.positions.length / 3;
  }

  /** Spatial index (built synchronously if background indexing has not finished yet). */
  get spatial(): MeshSpatialIndex {
    if (!this._spatial) this._spatial = new MeshSpatialIndex(this.mesh);
    return this._spatial;
  }

  private _neighbors: Int32Array | null = null;
  /** Triangle edge-neighbours (3 per triangle, -1 on boundaries), built on first use. */
  get neighbors(): Int32Array {
    if (!this._neighbors) this._neighbors = buildTriangleNeighbors(this.mesh);
    return this._neighbors;
  }

  get hasSpatial(): boolean {
    return !!this._spatial;
  }

  /** Picking: never blocks — returns null while a large mesh is still being indexed. */
  raycast = (origin: Vec3, dir: Vec3) => {
    if (!this._spatial && this.triangleCount >= MeshRuntime.BACKGROUND_THRESHOLD) return null;
    const hit = this.spatial.raycast(origin, dir);
    if (!hit) return null;
    return { point: hit.point, normal: hit.normal, distance: hit.distance, faceIndex: hit.faceIndex ?? -1 };
  };

  // -------------------------------------------------------------------------
  // Selection
  // -------------------------------------------------------------------------

  private chunkOf(tri: number): number {
    const c = this.chunks;
    let lo = 0, hi = c.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (c[mid].triStart <= tri) lo = mid;
      else hi = mid - 1;
    }
    return lo;
  }

  /** Set/clear a list of triangles. mode: add | remove | replace (replace clears first). */
  applyTriangles(tris: ArrayLike<number>, mode: "add" | "remove" | "replace"): void {
    if (mode === "replace") this.clearSelection(false);
    const v = mode === "remove" ? 0 : 1;
    const sel = this.selection;
    let delta = 0;
    let lastChunk = -1, lastEnd = -1, lastStart = 0;
    for (let i = 0; i < tris.length; i++) {
      const t = tris[i];
      if (sel[t] === v) continue;
      sel[t] = v;
      delta += v ? 1 : -1;
      if (t < lastStart || t >= lastEnd) {
        lastChunk = this.chunkOf(t);
        lastStart = this.chunks[lastChunk].triStart;
        lastEnd = lastStart + this.chunks[lastChunk].triCount;
      }
      this.dirtyChunks.add(lastChunk);
    }
    this.selectionCount += delta;
    this.scheduleFlush();
  }

  /** Replace the whole mask (e.g. results of grow/shrink/invert). */
  setMask(mask: Uint8Array): void {
    let count = 0;
    for (let i = 0; i < mask.length; i++) count += mask[i] ? 1 : 0;
    this.selection = mask;
    this.selectionCount = count;
    for (let i = 0; i < this.chunks.length; i++) this.dirtyChunks.add(i);
    this.scheduleFlush();
  }

  clearSelection(notify = true): void {
    if (this.selectionCount === 0) return;
    this.selection.fill(0);
    this.selectionCount = 0;
    for (let i = 0; i < this.chunks.length; i++) this.dirtyChunks.add(i);
    if (notify) this.scheduleFlush();
  }

  /** Indices of selected triangles. */
  selectedTriangles(): Uint32Array {
    const out = new Uint32Array(this.selectionCount);
    let k = 0;
    const sel = this.selection;
    for (let i = 0; i < sel.length && k < out.length; i++) if (sel[i]) out[k++] = i;
    return out;
  }

  /** Unique vertex indices of the selected triangles. */
  selectedVertices(): Uint32Array {
    const mark = new Uint8Array(this.vertexCount);
    const idx = this.mesh.indices;
    const sel = this.selection;
    let count = 0;
    for (let t = 0; t < sel.length; t++) {
      if (!sel[t]) continue;
      for (let k = 0; k < 3; k++) {
        const v = idx[3 * t + k];
        if (!mark[v]) {
          mark[v] = 1;
          count++;
        }
      }
    }
    const out = new Uint32Array(count);
    let k = 0;
    for (let v = 0; v < mark.length; v++) if (mark[v]) out[k++] = v;
    return out;
  }

  private scheduleFlush(): void {
    if (this.flushQueued) return;
    this.flushQueued = true;
    nextFrame(() => {
      this.flushQueued = false;
      this.flush();
    });
  }

  flush(): void {
    if (this.view && this.dirtyChunks.size) this.view.setSelection(this.selection, this.dirtyChunks);
    this.dirtyChunks.clear();
    this.onSelectionChanged?.();
  }

  dispose(): void {
    this.disposed = true;
    this._spatial?.dispose();
    this._spatial = null;
    this.view = null;
  }
}
