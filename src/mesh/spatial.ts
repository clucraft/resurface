/**
 * Spatial queries on a mesh (three-mesh-bvh) and spatial ordering / chunking of triangles.
 *
 * MeshSpatialIndex builds a MeshBVH over a BufferGeometry whose position and index attributes
 * wrap the MeshData arrays directly (no copies). The BVH is built with `indirect: true`, so the
 * mesh's triangle order is NOT modified; the BVH keeps its own permutation instead.
 *
 * Memory (on top of the mesh itself): indirect triangle buffer 4 B/triangle + BVH nodes
 * 32 B/node (≈ 2·n / leafSize nodes → ~6–13 B/triangle for the default leaf size of 10), i.e.
 * ~10–17 B per triangle (≈ 200–340 MB at 20 M triangles). Build time is O(n log n).
 *
 * The index references the mesh arrays: if the mesh is edited (positions changed in place),
 * call `refit()`; if the topology changes, build a new index.
 *
 * This is the only module in src/mesh allowed to import three / three-mesh-bvh.
 */
import { BufferAttribute, BufferGeometry, DoubleSide, Box3, Ray, Sphere, Triangle, Vector3 } from "three";
import { CENTER, CONTAINED, INTERSECTED, MeshBVH, NOT_INTERSECTED, SAH } from "three-mesh-bvh";
import type { BBox3, ClosestPointHit, ClosestPointQuery, MeshData, PlaneDef, ProgressFn, Vec3 } from "../core/types";
import { triangleCountOf, vertexCountOf } from "./topology";
import { GrowableU32 } from "./util/growable";
import { radixArgsortU32 } from "./util/sort";

export interface RayHit {
  point: Vec3;
  /** unit geometric normal of the hit triangle (CCW winding; not flipped towards the ray) */
  normal: Vec3;
  distance: number;
  faceIndex: number;
}

export interface SpatialIndexOptions {
  /** target triangles per BVH leaf (default 10) */
  leafSize?: number;
  /** 'sah' (slower build, faster queries) or 'center' (default, fast build) */
  strategy?: "center" | "sah";
  onProgress?: ProgressFn;
  /** allocate BVH buffers in SharedArrayBuffers (so a worker-built index can be shared without copying) */
  shared?: boolean;
  /** adopt a BVH built elsewhere (see serialize()); skips the build */
  serialized?: SerializedSpatialIndex;
}

/** Transferable form of a built index (BVH node buffers + indirect triangle permutation). */
export interface SerializedSpatialIndex {
  version: number;
  roots: ArrayBufferLike[];
  indirectBuffer: Uint32Array | null;
}

export class MeshSpatialIndex {
  readonly mesh: MeshData;
  private geometry: BufferGeometry | null;
  private bvh: MeshBVH | null;
  // scratch objects (per instance)
  private readonly v0 = new Vector3();
  private readonly ray = new Ray();
  private readonly tri = new Triangle();
  private readonly box = new Box3();

  constructor(mesh: MeshData, opts: SpatialIndexOptions = {}) {
    this.mesh = mesh;
    const geometry = new BufferGeometry();
    geometry.setAttribute("position", new BufferAttribute(mesh.positions, 3));
    geometry.setIndex(new BufferAttribute(mesh.indices, 1));
    this.geometry = geometry;
    if (opts.serialized) {
      const data = { ...opts.serialized, index: null } as unknown as Parameters<typeof MeshBVH.deserialize>[0];
      this.bvh = MeshBVH.deserialize(data, geometry, { setIndex: false });
    } else {
      this.bvh = new MeshBVH(geometry, {
        indirect: true,
        targetLeafSize: opts.leafSize ?? 10,
        strategy: opts.strategy === "sah" ? SAH : CENTER,
        useSharedArrayBuffer: !!opts.shared,
        onProgress: opts.onProgress ? (p: number) => opts.onProgress!(p, "Building spatial index") : undefined,
      });
    }
  }

  /** Buffers of the built BVH (not copied) — rebuild elsewhere with `new MeshSpatialIndex(mesh, { serialized })`. */
  serialize(): SerializedSpatialIndex {
    const s = MeshBVH.serialize(this.b, { cloneBuffers: false }) as unknown as { version?: number; roots: ArrayBufferLike[]; indirectBuffer: Uint32Array | null };
    return { version: s.version ?? 1, roots: s.roots, indirectBuffer: s.indirectBuffer };
  }

  private get b(): MeshBVH {
    if (!this.bvh) throw new Error("MeshSpatialIndex has been disposed");
    return this.bvh;
  }

  /** Recompute node bounds after vertex positions were modified in place (same topology). */
  refit(): void {
    if (this.geometry) this.geometry.attributes.position.needsUpdate = true;
    this.b.refit();
  }

  /** Bounds of the whole mesh. */
  bounds(): BBox3 {
    const bb = this.b.getBoundingBox(this.box);
    return { min: [bb.min.x, bb.min.y, bb.min.z], max: [bb.max.x, bb.max.y, bb.max.z] };
  }

  /** First hit along origin + t·dir (dir need not be unit), t in [0, maxDist]. Both triangle sides count. */
  raycast(origin: Vec3, dir: Vec3, maxDist = Infinity): RayHit | null {
    const l = Math.hypot(dir[0], dir[1], dir[2]);
    if (!(l > 0)) return null;
    this.ray.origin.set(origin[0], origin[1], origin[2]);
    this.ray.direction.set(dir[0] / l, dir[1] / l, dir[2] / l);
    const hit = this.b.raycastFirst(this.ray, DoubleSide, 0, maxDist);
    if (!hit || hit.faceIndex == null) return null;
    return {
      point: [hit.point.x, hit.point.y, hit.point.z],
      normal: this.faceNormal(hit.faceIndex),
      distance: hit.distance,
      faceIndex: hit.faceIndex,
    };
  }

  /** Closest point on the mesh to p (null if farther than maxDistance). */
  closestPoint(p: Vec3, maxDistance = Infinity): ClosestPointHit | null {
    this.v0.set(p[0], p[1], p[2]);
    const hit = this.b.closestPointToPoint(this.v0, undefined, 0, maxDistance);
    if (!hit) return null;
    return {
      point: [hit.point.x, hit.point.y, hit.point.z],
      normal: this.faceNormal(hit.faceIndex),
      distance: hit.distance,
      faceIndex: hit.faceIndex,
    };
  }

  /** Closest-point oracle bound to this index. */
  asQuery(): ClosestPointQuery {
    return (p, maxDistance) => this.closestPoint(p, maxDistance);
  }

  /** Unit normal of triangle t (CCW). */
  faceNormal(t: number): Vec3 {
    const P = this.mesh.positions, I = this.mesh.indices;
    const a = 3 * I[3 * t], b = 3 * I[3 * t + 1], c = 3 * I[3 * t + 2];
    const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2];
    const vx = P[c] - P[a], vy = P[c + 1] - P[a + 1], vz = P[c + 2] - P[a + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const len = Math.hypot(nx, ny, nz);
    return len > 0 ? [nx / len, ny / len, nz / len] : [0, 0, 0];
  }

  /** Triangles with any point within `radius` of `center`. */
  trianglesInSphere(center: Vec3, radius: number): Uint32Array {
    const out = new GrowableU32(256);
    const [cx, cy, cz] = center;
    const r2 = radius * radius;
    const sphere = new Sphere(new Vector3(cx, cy, cz), radius);
    const P = this.mesh.positions, I = this.mesh.indices;
    const bvh = this.b;
    bvh.shapecast({
      intersectsBounds: (box: Box3) => {
        if (!sphere.intersectsBox(box)) return NOT_INTERSECTED;
        const dx = Math.max(Math.abs(cx - box.min.x), Math.abs(cx - box.max.x));
        const dy = Math.max(Math.abs(cy - box.min.y), Math.abs(cy - box.max.y));
        const dz = Math.max(Math.abs(cz - box.min.z), Math.abs(cz - box.max.z));
        return dx * dx + dy * dy + dz * dz <= r2 ? CONTAINED : INTERSECTED;
      },
      intersectsRange: (offset: number, count: number, contained: boolean) => {
        for (let i = offset, e = offset + count; i < e; i++) {
          const t = bvh.resolveTriangleIndex(i);
          if (contained || pointTriangleDistSq(cx, cy, cz, P, I[3 * t], I[3 * t + 1], I[3 * t + 2]) <= r2) out.push(t);
        }
        return false;
      },
    });
    return out.toArray();
  }

  /** Triangles intersecting the axis-aligned box [min, max]. */
  trianglesInBox(min: Vec3, max: Vec3): Uint32Array {
    const out = new GrowableU32(256);
    const query = new Box3(new Vector3(min[0], min[1], min[2]), new Vector3(max[0], max[1], max[2]));
    const P = this.mesh.positions, I = this.mesh.indices;
    const tri = this.tri;
    const bvh = this.b;
    bvh.shapecast({
      intersectsBounds: (box: Box3) => {
        if (!query.intersectsBox(box)) return NOT_INTERSECTED;
        return query.containsBox(box) ? CONTAINED : INTERSECTED;
      },
      intersectsRange: (offset: number, count: number, contained: boolean) => {
        for (let i = offset, e = offset + count; i < e; i++) {
          const t = bvh.resolveTriangleIndex(i);
          if (contained) {
            out.push(t);
            continue;
          }
          const a = 3 * I[3 * t], b = 3 * I[3 * t + 1], c = 3 * I[3 * t + 2];
          tri.a.set(P[a], P[a + 1], P[a + 2]);
          tri.b.set(P[b], P[b + 1], P[b + 2]);
          tri.c.set(P[c], P[c + 1], P[c + 2]);
          if (query.intersectsTriangle(tri)) out.push(t);
        }
        return false;
      },
    });
    return out.toArray();
  }

  /**
   * Triangles that touch or straddle a plane (candidates for sectionMesh). A triangle is
   * reported when its vertices are not all strictly on one side.
   */
  trianglesIntersectingPlane(plane: PlaneDef): Uint32Array {
    const out = new GrowableU32(1024);
    const [nx, ny, nz] = plane.normal;
    const [ox, oy, oz] = plane.origin;
    const P = this.mesh.positions, I = this.mesh.indices;
    const bvh = this.b;
    const d = (v: number) => (P[3 * v] - ox) * nx + (P[3 * v + 1] - oy) * ny + (P[3 * v + 2] - oz) * nz;
    bvh.shapecast({
      intersectsBounds: (box: Box3) => {
        const cx = (box.min.x + box.max.x) / 2, cy = (box.min.y + box.max.y) / 2, cz = (box.min.z + box.max.z) / 2;
        const ex = (box.max.x - box.min.x) / 2, ey = (box.max.y - box.min.y) / 2, ez = (box.max.z - box.min.z) / 2;
        const s = (cx - ox) * nx + (cy - oy) * ny + (cz - oz) * nz;
        const r = ex * Math.abs(nx) + ey * Math.abs(ny) + ez * Math.abs(nz);
        return Math.abs(s) <= r * (1 + 1e-9) + 1e-12 ? INTERSECTED : NOT_INTERSECTED;
      },
      intersectsRange: (offset: number, count: number) => {
        for (let i = offset, e = offset + count; i < e; i++) {
          const t = bvh.resolveTriangleIndex(i);
          const da = d(I[3 * t]), db = d(I[3 * t + 1]), dc = d(I[3 * t + 2]);
          const pos = (da >= 0 ? 1 : 0) + (db >= 0 ? 1 : 0) + (dc >= 0 ? 1 : 0);
          if (pos !== 0 && pos !== 3) out.push(t);
          else if (pos === 3 && (da === 0 || db === 0 || dc === 0)) out.push(t);
        }
        return false;
      },
    });
    return out.toArray();
  }

  /** Release the BVH and geometry wrapper (the MeshData itself is untouched). */
  dispose(): void {
    this.bvh = null;
    if (this.geometry) {
      // drop references to the mesh arrays without touching GPU resources
      this.geometry.deleteAttribute("position");
      this.geometry.setIndex(null);
      this.geometry = null;
    }
  }
}

/** Squared distance from point (px,py,pz) to triangle (a,b,c) of a packed position array (Ericson). */
export function pointTriangleDistSq(px: number, py: number, pz: number, P: ArrayLike<number>, ia: number, ib: number, ic: number): number {
  const ax = P[3 * ia], ay = P[3 * ia + 1], az = P[3 * ia + 2];
  const bx = P[3 * ib], by = P[3 * ib + 1], bz = P[3 * ib + 2];
  const cx = P[3 * ic], cy = P[3 * ic + 1], cz = P[3 * ic + 2];
  const abx = bx - ax, aby = by - ay, abz = bz - az;
  const acx = cx - ax, acy = cy - ay, acz = cz - az;
  const apx = px - ax, apy = py - ay, apz = pz - az;
  const d1 = abx * apx + aby * apy + abz * apz, d2 = acx * apx + acy * apy + acz * apz;
  let qx: number, qy: number, qz: number;
  if (d1 <= 0 && d2 <= 0) {
    qx = ax; qy = ay; qz = az;
  } else {
    const bpx = px - bx, bpy = py - by, bpz = pz - bz;
    const d3 = abx * bpx + aby * bpy + abz * bpz, d4 = acx * bpx + acy * bpy + acz * bpz;
    if (d3 >= 0 && d4 <= d3) {
      qx = bx; qy = by; qz = bz;
    } else {
      const vc = d1 * d4 - d3 * d2;
      if (vc <= 0 && d1 >= 0 && d3 <= 0) {
        const v = d1 / (d1 - d3);
        qx = ax + v * abx; qy = ay + v * aby; qz = az + v * abz;
      } else {
        const cpx = px - cx, cpy = py - cy, cpz = pz - cz;
        const d5 = abx * cpx + aby * cpy + abz * cpz, d6 = acx * cpx + acy * cpy + acz * cpz;
        if (d6 >= 0 && d5 <= d6) {
          qx = cx; qy = cy; qz = cz;
        } else {
          const vb = d5 * d2 - d1 * d6;
          if (vb <= 0 && d2 >= 0 && d6 <= 0) {
            const w = d2 / (d2 - d6);
            qx = ax + w * acx; qy = ay + w * acy; qz = az + w * acz;
          } else {
            const va = d3 * d6 - d5 * d4;
            if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
              const w = (d4 - d3) / (d4 - d3 + (d5 - d6));
              qx = bx + w * (cx - bx); qy = by + w * (cy - by); qz = bz + w * (cz - bz);
            } else {
              const denom = 1 / (va + vb + vc);
              const v = vb * denom, w = vc * denom;
              qx = ax + abx * v + acx * w; qy = ay + aby * v + acy * w; qz = az + abz * v + acz * w;
            }
          }
        }
      }
    }
  }
  const dx = px - qx, dy = py - qy, dz = pz - qz;
  return dx * dx + dy * dy + dz * dz;
}

// ---------------------------------------------------------------------------
// Spatial ordering & chunking
// ---------------------------------------------------------------------------

/** Spread the low 10 bits of x so there are two zero bits between each. */
function part1by2(x: number): number {
  x &= 0x3ff;
  x = (x | (x << 16)) & 0x030000ff;
  x = (x | (x << 8)) & 0x0300f00f;
  x = (x | (x << 4)) & 0x030c30c3;
  x = (x | (x << 2)) & 0x09249249;
  return x;
}

/**
 * Permutation of the triangles along a Morton (Z-order) curve of their centroids (10 bits per
 * axis over the mesh bounds), computed with an LSD radix sort — linear time, 20 B/triangle
 * transient. perm[i] = old triangle index placed at position i.
 */
export function spatialOrderTriangles(mesh: MeshData, onProgress?: ProgressFn): Uint32Array {
  const P = mesh.positions, I = mesh.indices;
  const nt = triangleCountOf(mesh);
  const nv = vertexCountOf(mesh);
  let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
  for (let i = 0; i < nv * 3; i += 3) {
    const x = P[i], y = P[i + 1], z = P[i + 2];
    if (x < x0) x0 = x;
    if (x > x1) x1 = x;
    if (y < y0) y0 = y;
    if (y > y1) y1 = y;
    if (z < z0) z0 = z;
    if (z > z1) z1 = z;
  }
  const s = 1023.999 / Math.max(x1 - x0, y1 - y0, z1 - z0, 1e-30);
  const codes = new Uint32Array(nt);
  for (let t = 0; t < nt; t++) {
    const a = 3 * I[3 * t], b = 3 * I[3 * t + 1], c = 3 * I[3 * t + 2];
    const cx = ((P[a] + P[b] + P[c]) / 3 - x0) * s;
    const cy = ((P[a + 1] + P[b + 1] + P[c + 1]) / 3 - y0) * s;
    const cz = ((P[a + 2] + P[b + 2] + P[c + 2]) / 3 - z0) * s;
    codes[t] = (part1by2(cx) | (part1by2(cy) << 1) | (part1by2(cz) << 2)) >>> 0;
    if (onProgress && (t & 0xfffff) === 0) onProgress((0.5 * t) / nt, "Ordering triangles");
  }
  const perm = radixArgsortU32(codes, 30);
  onProgress?.(1, "Ordering triangles");
  return perm;
}

export interface ReorderOptions {
  /**
   * Also renumber vertices in order of first use by the reordered triangles, so each chunk's
   * vertices are (mostly) contiguous. Changes vertex indices! Default false.
   */
  reorderVertices?: boolean;
}

/** New mesh with triangle i = old triangle perm[i] (attributes carried along). */
export function reorderTriangles(mesh: MeshData, perm: Uint32Array, opts: ReorderOptions = {}): MeshData {
  const I = mesh.indices;
  const nt = triangleCountOf(mesh);
  if (perm.length !== nt) throw new Error(`reorderTriangles: permutation length ${perm.length} != triangle count ${nt}`);
  const indices = new Uint32Array(nt * 3);
  for (let i = 0; i < nt; i++) {
    const t = perm[i];
    indices[3 * i] = I[3 * t];
    indices[3 * i + 1] = I[3 * t + 1];
    indices[3 * i + 2] = I[3 * t + 2];
  }
  if (!opts.reorderVertices) {
    const out: MeshData = { positions: mesh.positions.slice(), indices };
    if (mesh.normals) out.normals = mesh.normals.slice();
    if (mesh.colors) out.colors = mesh.colors.slice();
    return out;
  }
  const nv = vertexCountOf(mesh);
  const map = new Int32Array(nv).fill(-1);
  let n = 0;
  for (let c = 0; c < nt * 3; c++) {
    const v = indices[c];
    if (map[v] < 0) map[v] = n++;
    indices[c] = map[v];
  }
  // unreferenced vertices go to the end, keeping their relative order
  for (let v = 0; v < nv; v++) if (map[v] < 0) map[v] = n++;
  const positions = new Float32Array(nv * 3);
  const normals = mesh.normals ? new Float32Array(nv * 3) : undefined;
  const colors = mesh.colors ? new Uint8Array(nv * 3) : undefined;
  const P = mesh.positions, N = mesh.normals, C = mesh.colors;
  for (let v = 0; v < nv; v++) {
    const m = map[v];
    positions[3 * m] = P[3 * v];
    positions[3 * m + 1] = P[3 * v + 1];
    positions[3 * m + 2] = P[3 * v + 2];
    if (normals && N) {
      normals[3 * m] = N[3 * v];
      normals[3 * m + 1] = N[3 * v + 1];
      normals[3 * m + 2] = N[3 * v + 2];
    }
    if (colors && C) {
      colors[3 * m] = C[3 * v];
      colors[3 * m + 1] = C[3 * v + 1];
      colors[3 * m + 2] = C[3 * v + 2];
    }
  }
  const out: MeshData = { positions, indices };
  if (normals) out.normals = normals;
  if (colors) out.colors = colors;
  return out;
}

export interface TriangleChunk {
  triStart: number;
  triCount: number;
  bounds: BBox3;
}

/**
 * Split the (already spatially ordered) triangle list into consecutive ranges of at most
 * `maxTrianglesPerChunk` triangles and compute each range's bounds.
 */
export function buildChunks(mesh: MeshData, maxTrianglesPerChunk: number): TriangleChunk[] {
  const P = mesh.positions, I = mesh.indices;
  const nt = triangleCountOf(mesh);
  const size = Math.max(1, Math.floor(maxTrianglesPerChunk));
  const chunks: TriangleChunk[] = [];
  for (let s = 0; s < nt; s += size) {
    const e = Math.min(nt, s + size);
    let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
    for (let c = 3 * s; c < 3 * e; c++) {
      const o = 3 * I[c];
      const x = P[o], y = P[o + 1], z = P[o + 2];
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
      if (z < z0) z0 = z;
      if (z > z1) z1 = z;
    }
    chunks.push({ triStart: s, triCount: e - s, bounds: { min: [x0, y0, z0], max: [x1, y1, z1] } });
  }
  return chunks;
}
