/**
 * Uniform-grid spatial index over packed points: nearest neighbour, k-nearest neighbours,
 * PCA normal estimation and a `ClosestPointQuery` oracle for raw point clouds.
 *
 * Used internally (normal estimation for normal-less fits, RANSAC) and handy for point clouds
 * that have no triangle mesh / BVH (e.g. PTX imports, tests).
 */
import type { ClosestPointHit, ClosestPointQuery, Vec3 } from "../core/types";
import { eigenSym3 } from "../core/linalg";

const MAX_CELLS = 1 << 22;

export class GridIndex {
  readonly pts: ArrayLike<number>;
  readonly n: number;
  readonly cell: number;
  readonly min: Vec3;
  readonly nx: number;
  readonly ny: number;
  readonly nz: number;
  /** cellStart[c] .. cellStart[c+1] index into `order` */
  readonly cellStart: Uint32Array;
  readonly order: Uint32Array;

  /**
   * @param pts packed xyz
   * @param n number of points (default pts.length / 3)
   * @param cellSize grid cell size (default: ~ 1.5 x mean surface spacing estimate)
   */
  constructor(pts: ArrayLike<number>, n = Math.floor(pts.length / 3), cellSize?: number) {
    if (n <= 0) throw new Error("GridIndex: no points");
    this.pts = pts;
    this.n = n;
    let minX = Infinity, minY = Infinity, minZ = Infinity, maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < n; i++) {
      const x = pts[3 * i], y = pts[3 * i + 1], z = pts[3 * i + 2];
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (z < minZ) minZ = z;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
      if (z > maxZ) maxZ = z;
    }
    if (!Number.isFinite(minX + minY + minZ + maxX + maxY + maxZ)) throw new Error("GridIndex: non-finite coordinates");
    const ex = maxX - minX, ey = maxY - minY, ez = maxZ - minZ;
    const diag = Math.hypot(ex, ey, ez) || 1;
    let cell = cellSize ?? (1.5 * diag) / Math.sqrt(n);
    if (!(cell > 0)) cell = diag;
    let nx = 0, ny = 0, nz = 0;
    for (let guard = 0; guard < 64; guard++) {
      nx = Math.floor(ex / cell) + 1;
      ny = Math.floor(ey / cell) + 1;
      nz = Math.floor(ez / cell) + 1;
      const total = nx * ny * nz;
      if (total <= MAX_CELLS) break;
      cell *= Math.cbrt(total / MAX_CELLS) * 1.01;
    }
    this.cell = cell;
    this.min = [minX, minY, minZ];
    this.nx = nx;
    this.ny = ny;
    this.nz = nz;
    const total = nx * ny * nz;
    const counts = new Uint32Array(total + 1);
    const key = new Uint32Array(n);
    const inv = 1 / cell;
    for (let i = 0; i < n; i++) {
      const ix = Math.min(nx - 1, Math.floor((pts[3 * i] - minX) * inv));
      const iy = Math.min(ny - 1, Math.floor((pts[3 * i + 1] - minY) * inv));
      const iz = Math.min(nz - 1, Math.floor((pts[3 * i + 2] - minZ) * inv));
      const c = ix + nx * (iy + ny * iz);
      key[i] = c;
      counts[c + 1]++;
    }
    for (let c = 0; c < total; c++) counts[c + 1] += counts[c];
    const fill = counts.slice(0, total);
    const order = new Uint32Array(n);
    for (let i = 0; i < n; i++) order[fill[key[i]]++] = i;
    this.cellStart = counts;
    this.order = order;
  }

  /** Visit candidate points in Chebyshev ring r around cell (qi,qj,qk); returns false if the ring is fully outside. */
  private visitRing(qi: number, qj: number, qk: number, r: number, visit: (idx: number) => void): boolean {
    const { nx, ny, nz } = this;
    const i0 = Math.max(0, qi - r), i1 = Math.min(nx - 1, qi + r);
    const j0 = Math.max(0, qj - r), j1 = Math.min(ny - 1, qj + r);
    const k0 = Math.max(0, qk - r), k1 = Math.min(nz - 1, qk + r);
    if (i0 > i1 || j0 > j1 || k0 > k1) return false;
    const cs = this.cellStart, order = this.order;
    for (let i = i0; i <= i1; i++) {
      const edgeI = i === qi - r || i === qi + r;
      for (let j = j0; j <= j1; j++) {
        const edgeJ = edgeI || j === qj - r || j === qj + r;
        if (edgeJ) {
          for (let k = k0; k <= k1; k++) {
            const c = i + nx * (j + ny * k);
            for (let t = cs[c]; t < cs[c + 1]; t++) visit(order[t]);
          }
        } else {
          const ka = qk - r, kb = qk + r;
          if (ka >= 0 && ka < nz) {
            const c = i + nx * (j + ny * ka);
            for (let t = cs[c]; t < cs[c + 1]; t++) visit(order[t]);
          }
          if (kb !== ka && kb >= 0 && kb < nz) {
            const c = i + nx * (j + ny * kb);
            for (let t = cs[c]; t < cs[c + 1]; t++) visit(order[t]);
          }
        }
      }
    }
    return true;
  }

  private cellCoords(x: number, y: number, z: number): [number, number, number, number, number] {
    const inv = 1 / this.cell;
    const qi = Math.floor((x - this.min[0]) * inv);
    const qj = Math.floor((y - this.min[1]) * inv);
    const qk = Math.floor((z - this.min[2]) * inv);
    // first ring that intersects the grid, and ring after which the whole grid is covered
    const d0 = Math.max(0, -qi, qi - (this.nx - 1), -qj, qj - (this.ny - 1), -qk, qk - (this.nz - 1));
    const dMax = Math.max(Math.abs(qi), Math.abs(qi - (this.nx - 1)), Math.abs(qj), Math.abs(qj - (this.ny - 1)), Math.abs(qk), Math.abs(qk - (this.nz - 1)));
    return [qi, qj, qk, d0, dMax];
  }

  /** Index of the nearest point within maxDist (or -1). Squared distance in `lastDist2`. */
  lastDist2 = Infinity;
  nearest(x: number, y: number, z: number, maxDist = Infinity): number {
    const [qi, qj, qk, d0, dMax] = this.cellCoords(x, y, z);
    const pts = this.pts;
    let best = -1;
    let bestD2 = maxDist === Infinity ? Infinity : maxDist * maxDist;
    const visit = (idx: number) => {
      const dx = pts[3 * idx] - x, dy = pts[3 * idx + 1] - y, dz = pts[3 * idx + 2] - z;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < bestD2) {
        bestD2 = d2;
        best = idx;
      }
    };
    for (let r = d0; r <= dMax; r++) {
      // all points in rings > r-1 are at least (r-1)*cell away
      const lb = Math.max(0, r - 1) * this.cell;
      if (lb * lb >= bestD2) break;
      this.visitRing(qi, qj, qk, r, visit);
    }
    this.lastDist2 = bestD2;
    return best;
  }

  /**
   * k nearest neighbours of (x,y,z). Fills outIdx/outD2 sorted by distance and returns the count
   * (< k only when the index holds fewer points).
   */
  knn(x: number, y: number, z: number, k: number, outIdx: Int32Array, outD2: Float64Array): number {
    const [qi, qj, qk, d0, dMax] = this.cellCoords(x, y, z);
    const pts = this.pts;
    let cnt = 0;
    const visit = (idx: number) => {
      const dx = pts[3 * idx] - x, dy = pts[3 * idx + 1] - y, dz = pts[3 * idx + 2] - z;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (cnt < k) {
        let p = cnt++;
        while (p > 0 && outD2[p - 1] > d2) {
          outD2[p] = outD2[p - 1];
          outIdx[p] = outIdx[p - 1];
          p--;
        }
        outD2[p] = d2;
        outIdx[p] = idx;
      } else if (d2 < outD2[k - 1]) {
        let p = k - 1;
        while (p > 0 && outD2[p - 1] > d2) {
          outD2[p] = outD2[p - 1];
          outIdx[p] = outIdx[p - 1];
          p--;
        }
        outD2[p] = d2;
        outIdx[p] = idx;
      }
    };
    for (let r = d0; r <= dMax; r++) {
      const lb = Math.max(0, r - 1) * this.cell;
      if (cnt === k && lb * lb >= outD2[k - 1]) break;
      this.visitRing(qi, qj, qk, r, visit);
    }
    return cnt;
  }
}

/**
 * PCA normal (unoriented, unit) of the k nearest neighbours in `index` around (x,y,z).
 * Writes into out[o..o+3). Returns the "surface variation" λ0/(λ0+λ1+λ2) (0 = perfectly flat).
 */
export function pcaNormalAt(
  index: GridIndex,
  x: number,
  y: number,
  z: number,
  k: number,
  out: Float64Array | Float32Array,
  o: number,
  scratchIdx = new Int32Array(k),
  scratchD2 = new Float64Array(k),
): number {
  const cnt = index.knn(x, y, z, k, scratchIdx, scratchD2);
  const pts = index.pts;
  if (cnt < 3) {
    out[o] = 0;
    out[o + 1] = 0;
    out[o + 2] = 1;
    return 1;
  }
  let cx = 0, cy = 0, cz = 0;
  for (let t = 0; t < cnt; t++) {
    const i = scratchIdx[t];
    cx += pts[3 * i];
    cy += pts[3 * i + 1];
    cz += pts[3 * i + 2];
  }
  cx /= cnt;
  cy /= cnt;
  cz /= cnt;
  let xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
  for (let t = 0; t < cnt; t++) {
    const i = scratchIdx[t];
    const dx = pts[3 * i] - cx, dy = pts[3 * i + 1] - cy, dz = pts[3 * i + 2] - cz;
    xx += dx * dx;
    xy += dx * dy;
    xz += dx * dz;
    yy += dy * dy;
    yz += dy * dz;
    zz += dz * dz;
  }
  const e = eigenSym3([xx, xy, xz, xy, yy, yz, xz, yz, zz]);
  const nv = e.vectors[0];
  out[o] = nv[0];
  out[o + 1] = nv[1];
  out[o + 2] = nv[2];
  const tr = e.values[0] + e.values[1] + e.values[2];
  return tr > 0 ? Math.max(0, e.values[0]) / tr : 1;
}

/**
 * Estimate unoriented unit normals by local PCA of the k nearest neighbours.
 * @param positions packed xyz
 * @param opts.k neighbourhood size (default 16)
 * @param opts.indices restrict to these vertices (result is per selected vertex, in selection order)
 */
export function estimateNormals(positions: ArrayLike<number>, opts: { k?: number; indices?: ArrayLike<number> } = {}): Float32Array {
  const k = Math.max(3, opts.k ?? 16);
  const ind = opts.indices;
  let pts: ArrayLike<number> = positions;
  let n = Math.floor(positions.length / 3);
  if (ind) {
    const p = new Float64Array(ind.length * 3);
    for (let t = 0; t < ind.length; t++) {
      const i = ind[t];
      p[3 * t] = positions[3 * i];
      p[3 * t + 1] = positions[3 * i + 1];
      p[3 * t + 2] = positions[3 * i + 2];
    }
    pts = p;
    n = ind.length;
  }
  const index = new GridIndex(pts, n);
  const out = new Float32Array(n * 3);
  const si = new Int32Array(k), sd = new Float64Array(k);
  for (let i = 0; i < n; i++) pcaNormalAt(index, pts[3 * i], pts[3 * i + 1], pts[3 * i + 2], k, out, 3 * i, si, sd);
  return out;
}

export interface PointCloudQueryOptions {
  /**
   * Return the foot point on the tangent plane of the nearest sample instead of the sample itself
   * (more accurate distances for smooth, densely sampled surfaces). Default false.
   */
  projectToTangentPlane?: boolean;
  /** neighbourhood size for normal estimation when `normals` is not given (default 16) */
  k?: number;
}

/**
 * A `ClosestPointQuery` against a raw point cloud (nearest sample + its normal). When normals are
 * not supplied they are estimated lazily by local PCA (unoriented).
 */
export function createPointCloudQuery(
  positions: ArrayLike<number>,
  normals?: ArrayLike<number> | null,
  opts: PointCloudQueryOptions = {},
): ClosestPointQuery {
  const index = new GridIndex(positions);
  const n = index.n;
  const k = opts.k ?? 16;
  let est: Float32Array | null = null;
  let done: Uint8Array | null = null;
  const si = new Int32Array(k), sd = new Float64Array(k);
  const normalOf = (i: number): Vec3 => {
    if (normals) {
      const nx = normals[3 * i], ny = normals[3 * i + 1], nz = normals[3 * i + 2];
      const l = Math.hypot(nx, ny, nz) || 1;
      return [nx / l, ny / l, nz / l];
    }
    if (!est) {
      est = new Float32Array(3 * n);
      done = new Uint8Array(n);
    }
    if (!done![i]) {
      pcaNormalAt(index, positions[3 * i], positions[3 * i + 1], positions[3 * i + 2], k, est, 3 * i, si, sd);
      done![i] = 1;
    }
    return [est[3 * i], est[3 * i + 1], est[3 * i + 2]];
  };
  return (p: Vec3, maxDistance?: number): ClosestPointHit | null => {
    const idx = index.nearest(p[0], p[1], p[2], maxDistance ?? Infinity);
    if (idx < 0) return null;
    const q: Vec3 = [positions[3 * idx], positions[3 * idx + 1], positions[3 * idx + 2]];
    const nrm = normalOf(idx);
    let point = q;
    if (opts.projectToTangentPlane) {
      const s = (p[0] - q[0]) * nrm[0] + (p[1] - q[1]) * nrm[1] + (p[2] - q[2]) * nrm[2];
      point = [p[0] - s * nrm[0], p[1] - s * nrm[1], p[2] - s * nrm[2]];
    }
    const distance = Math.hypot(p[0] - point[0], p[1] - point[1], p[2] - point[2]);
    if (maxDistance !== undefined && distance > maxDistance) return null;
    return { point, normal: nrm, distance, faceIndex: idx };
  };
}
