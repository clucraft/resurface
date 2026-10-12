/**
 * Naive Surface Nets: extracts a watertight triangle mesh from a signed distance function
 * sampled on a regular grid (negative inside). Vertices are refined onto the zero set with a
 * gradient step, which gives scan-like uniform triangles with slightly rounded sharp edges.
 */
import type { MeshData, Vec3 } from "../core/types";

export type SDF = (x: number, y: number, z: number) => number;

export function surfaceNets(sdf: SDF, min: Vec3, max: Vec3, h: number): MeshData {
  const nx = Math.ceil((max[0] - min[0]) / h) + 2;
  const ny = Math.ceil((max[1] - min[1]) / h) + 2;
  const nz = Math.ceil((max[2] - min[2]) / h) + 2;
  const ox = min[0] - h, oy = min[1] - h, oz = min[2] - h;
  const sx = 1, sy = nx, sz = nx * ny;
  const field = new Float32Array(nx * ny * nz);
  for (let k = 0; k < nz; k++)
    for (let j = 0; j < ny; j++)
      for (let i = 0; i < nx; i++) field[i + j * sy + k * sz] = sdf(ox + i * h, oy + j * h, oz + k * h);

  // one vertex per cell that straddles the surface
  const cellVert = new Int32Array((nx - 1) * (ny - 1) * (nz - 1)).fill(-1);
  const cx = nx - 1, cxy = (nx - 1) * (ny - 1);
  const verts: number[] = [];
  const corners = [
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
    [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
  ];
  const edges = [
    [0, 1], [2, 3], [4, 5], [6, 7],
    [0, 2], [1, 3], [4, 6], [5, 7],
    [0, 4], [1, 5], [2, 6], [3, 7],
  ];
  const v = new Float32Array(8);
  for (let k = 0; k < nz - 1; k++)
    for (let j = 0; j < ny - 1; j++)
      for (let i = 0; i < nx - 1; i++) {
        let mask = 0;
        for (let c = 0; c < 8; c++) {
          const [a, b, d] = corners[c];
          v[c] = field[i + a + (j + b) * sy + (k + d) * sz];
          if (v[c] < 0) mask |= 1 << c;
        }
        if (mask === 0 || mask === 255) continue;
        let px = 0, py = 0, pz = 0, n = 0;
        for (const [e0, e1] of edges) {
          const a = v[e0], b = v[e1];
          if (a < 0 === b < 0) continue;
          const t = a / (a - b);
          const c0 = corners[e0], c1 = corners[e1];
          px += c0[0] + (c1[0] - c0[0]) * t;
          py += c0[1] + (c1[1] - c0[1]) * t;
          pz += c0[2] + (c1[2] - c0[2]) * t;
          n++;
        }
        cellVert[i + j * cx + k * cxy] = verts.length / 3;
        verts.push(ox + (i + px / n) * h, oy + (j + py / n) * h, oz + (k + pz / n) * h);
      }

  // quads across sign-changing grid edges
  const tris: number[] = [];
  const cell = (i: number, j: number, k: number) => cellVert[i + j * cx + k * cxy];
  const quad = (a: number, b: number, c: number, d: number, flip: boolean) => {
    if (a < 0 || b < 0 || c < 0 || d < 0) return;
    if (flip) tris.push(a, c, b, a, d, c);
    else tris.push(a, b, c, a, c, d);
  };
  for (let k = 1; k < nz - 1; k++)
    for (let j = 1; j < ny - 1; j++)
      for (let i = 1; i < nx - 1; i++) {
        const f0 = field[i + j * sy + k * sz];
        const in0 = f0 < 0;
        // edge along +x
        if (i < nx - 1 && in0 !== field[i + 1 + j * sy + k * sz] < 0)
          quad(cell(i, j - 1, k - 1), cell(i, j, k - 1), cell(i, j, k), cell(i, j - 1, k), !in0);
        // edge along +y
        if (j < ny - 1 && in0 !== field[i + (j + 1) * sy + k * sz] < 0)
          quad(cell(i - 1, j, k - 1), cell(i - 1, j, k), cell(i, j, k), cell(i, j, k - 1), !in0);
        // edge along +z
        if (k < nz - 1 && in0 !== field[i + j * sy + (k + 1) * sz] < 0)
          quad(cell(i - 1, j - 1, k), cell(i, j - 1, k), cell(i, j, k), cell(i - 1, j, k), !in0);
      }

  const positions = Float32Array.from(verts);
  // project vertices onto the zero level set (two Newton steps with a central-difference gradient)
  const e = h * 0.25;
  for (let it = 0; it < 2; it++)
    for (let p = 0; p < positions.length; p += 3) {
      const x = positions[p], y = positions[p + 1], z = positions[p + 2];
      const d = sdf(x, y, z);
      const gx = (sdf(x + e, y, z) - sdf(x - e, y, z)) / (2 * e);
      const gy = (sdf(x, y + e, z) - sdf(x, y - e, z)) / (2 * e);
      const gz = (sdf(x, y, z + e) - sdf(x, y, z - e)) / (2 * e);
      const g2 = gx * gx + gy * gy + gz * gz;
      if (g2 < 1e-12) continue;
      const s = Math.max(-h, Math.min(h, d / g2));
      positions[p] = x - gx * s;
      positions[p + 1] = y - gy * s;
      positions[p + 2] = z - gz * s;
    }
  return { positions, indices: Uint32Array.from(tris) };
}

/** Deterministic Gaussian noise along vertex normals (simulated scanner noise). */
export function addScanNoise(mesh: MeshData, sigma: number, seed = 1): void {
  const { positions, indices } = mesh;
  const n = positions.length / 3;
  const normals = new Float32Array(positions.length);
  for (let t = 0; t < indices.length; t += 3) {
    const a = indices[t], b = indices[t + 1], c = indices[t + 2];
    const e1x = positions[3 * b] - positions[3 * a], e1y = positions[3 * b + 1] - positions[3 * a + 1], e1z = positions[3 * b + 2] - positions[3 * a + 2];
    const e2x = positions[3 * c] - positions[3 * a], e2y = positions[3 * c + 1] - positions[3 * a + 1], e2z = positions[3 * c + 2] - positions[3 * a + 2];
    const nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
    for (const v of [a, b, c]) {
      normals[3 * v] += nx;
      normals[3 * v + 1] += ny;
      normals[3 * v + 2] += nz;
    }
  }
  let s = seed >>> 0 || 1;
  const rnd = () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return ((s >>> 0) + 0.5) / 4294967296;
  };
  for (let i = 0; i < n; i++) {
    const l = Math.hypot(normals[3 * i], normals[3 * i + 1], normals[3 * i + 2]) || 1;
    const g = Math.sqrt(-2 * Math.log(rnd())) * Math.cos(2 * Math.PI * rnd()) * sigma;
    positions[3 * i] += (normals[3 * i] / l) * g;
    positions[3 * i + 1] += (normals[3 * i + 1] / l) * g;
    positions[3 * i + 2] += (normals[3 * i + 2] / l) * g;
  }
}
