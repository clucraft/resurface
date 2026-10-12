/**
 * Procedural test/sample meshes (grids, spheres, cylinders, extruded profiles). Used by the unit
 * tests and handy for demos. All meshes are CCW = outward.
 */
import type { MeshData } from "../../core/types";

/**
 * Regular grid in the XY plane, (nx × ny) cells, 2 triangles per cell, normal +Z.
 * Vertex (i, j) has index j·(nx+1) + i and position (x0 + i·dx, y0 + j·dy, z(x, y)).
 */
export function makeGrid(nx: number, ny: number, sizeX = nx, sizeY = ny, z?: (x: number, y: number) => number, x0 = 0, y0 = 0): MeshData {
  const nvx = nx + 1, nvy = ny + 1;
  const positions = new Float32Array(nvx * nvy * 3);
  const dx = sizeX / nx, dy = sizeY / ny;
  for (let j = 0; j < nvy; j++)
    for (let i = 0; i < nvx; i++) {
      const v = j * nvx + i;
      const x = x0 + i * dx, y = y0 + j * dy;
      positions[3 * v] = x;
      positions[3 * v + 1] = y;
      positions[3 * v + 2] = z ? z(x, y) : 0;
    }
  const indices = new Uint32Array(nx * ny * 6);
  let o = 0;
  for (let j = 0; j < ny; j++)
    for (let i = 0; i < nx; i++) {
      const a = j * nvx + i, b = a + 1, c = a + nvx, d = c + 1;
      indices[o++] = a;
      indices[o++] = b;
      indices[o++] = d;
      indices[o++] = a;
      indices[o++] = d;
      indices[o++] = c;
    }
  return { positions, indices };
}

/** Closed UV sphere (shared pole vertices), `segments` around, `rings` from pole to pole. */
export function makeUVSphere(radius: number, segments: number, rings: number, center: [number, number, number] = [0, 0, 0]): MeshData {
  const nv = 2 + (rings - 1) * segments;
  const positions = new Float32Array(nv * 3);
  const [cx, cy, cz] = center;
  positions.set([cx, cy, cz + radius], 0); // north pole: 0
  positions.set([cx, cy, cz - radius], 3); // south pole: 1
  for (let r = 1; r < rings; r++) {
    const th = (Math.PI * r) / rings;
    const z = Math.cos(th), s = Math.sin(th);
    for (let k = 0; k < segments; k++) {
      const ph = (2 * Math.PI * k) / segments;
      const v = 2 + (r - 1) * segments + k;
      positions[3 * v] = cx + radius * s * Math.cos(ph);
      positions[3 * v + 1] = cy + radius * s * Math.sin(ph);
      positions[3 * v + 2] = cz + radius * z;
    }
  }
  const ring = (r: number, k: number) => 2 + (r - 1) * segments + (((k % segments) + segments) % segments);
  const idx: number[] = [];
  for (let k = 0; k < segments; k++) idx.push(0, ring(1, k), ring(1, k + 1));
  for (let r = 1; r < rings - 1; r++)
    for (let k = 0; k < segments; k++) {
      const a = ring(r, k), b = ring(r, k + 1), c = ring(r + 1, k), d = ring(r + 1, k + 1);
      idx.push(a, c, d, a, d, b);
    }
  for (let k = 0; k < segments; k++) idx.push(1, ring(rings - 1, k + 1), ring(rings - 1, k));
  return { positions, indices: Uint32Array.from(idx) };
}

/** Icosphere with `subdivisions` levels (20·4^s triangles), uniform triangles, outward. */
export function makeIcosphere(radius: number, subdivisions: number): MeshData {
  const t = (1 + Math.sqrt(5)) / 2;
  const verts: number[] = [-1, t, 0, 1, t, 0, -1, -t, 0, 1, -t, 0, 0, -1, t, 0, 1, t, 0, -1, -t, 0, 1, -t, t, 0, -1, t, 0, 1, -t, 0, -1, -t, 0, 1];
  let faces: number[] = [0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8, 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1];
  for (let s = 0; s < subdivisions; s++) {
    const mid = new Map<number, number>();
    const nf: number[] = [];
    const midpoint = (a: number, b: number) => {
      const key = a < b ? a * 1e7 + b : b * 1e7 + a;
      let m = mid.get(key);
      if (m === undefined) {
        m = verts.length / 3;
        verts.push((verts[3 * a] + verts[3 * b]) / 2, (verts[3 * a + 1] + verts[3 * b + 1]) / 2, (verts[3 * a + 2] + verts[3 * b + 2]) / 2);
        mid.set(key, m);
      }
      return m;
    };
    for (let f = 0; f < faces.length; f += 3) {
      const a = faces[f], b = faces[f + 1], c = faces[f + 2];
      const ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a);
      nf.push(a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca);
    }
    faces = nf;
  }
  const positions = new Float32Array(verts.length);
  for (let v = 0; v < verts.length / 3; v++) {
    const x = verts[3 * v], y = verts[3 * v + 1], z = verts[3 * v + 2];
    const l = Math.hypot(x, y, z);
    positions[3 * v] = (x / l) * radius;
    positions[3 * v + 1] = (y / l) * radius;
    positions[3 * v + 2] = (z / l) * radius;
  }
  return { positions, indices: Uint32Array.from(faces) };
}

/**
 * Open surface obtained by sweeping a 2D profile (points (x, z)) along +Y over [0, length] with
 * `nAlong` segments. Orientation: a profile running in +x at z = 0 gets normal +Z (the normal is
 * the profile tangent rotated by +90° in the XZ plane).
 */
export function makeExtrudedProfile(profile: [number, number][], length: number, nAlong: number): MeshData {
  const np = profile.length;
  const nvy = nAlong + 1;
  const positions = new Float32Array(np * nvy * 3);
  for (let j = 0; j < nvy; j++)
    for (let i = 0; i < np; i++) {
      const v = j * np + i;
      positions[3 * v] = profile[i][0];
      positions[3 * v + 1] = (length * j) / nAlong;
      positions[3 * v + 2] = profile[i][1];
    }
  const indices = new Uint32Array((np - 1) * nAlong * 6);
  let o = 0;
  for (let j = 0; j < nAlong; j++)
    for (let i = 0; i < np - 1; i++) {
      const a = j * np + i, b = a + 1, c = a + np, d = c + 1;
      indices[o++] = a;
      indices[o++] = b;
      indices[o++] = d;
      indices[o++] = a;
      indices[o++] = d;
      indices[o++] = c;
    }
  return { positions, indices };
}

/** Open cylinder side (axis Z, radius r, height h), outward normals. */
export function makeCylinder(radius: number, height: number, segments: number, rows: number): MeshData {
  const nv = segments * (rows + 1);
  const positions = new Float32Array(nv * 3);
  for (let j = 0; j <= rows; j++)
    for (let k = 0; k < segments; k++) {
      const v = j * segments + k;
      const ph = (2 * Math.PI * k) / segments;
      positions[3 * v] = radius * Math.cos(ph);
      positions[3 * v + 1] = radius * Math.sin(ph);
      positions[3 * v + 2] = (height * j) / rows;
    }
  const idx: number[] = [];
  for (let j = 0; j < rows; j++)
    for (let k = 0; k < segments; k++) {
      const a = j * segments + k, b = j * segments + ((k + 1) % segments);
      const c = a + segments, d = b + segments;
      idx.push(a, b, d, a, d, c);
    }
  return { positions, indices: Uint32Array.from(idx) };
}

/** Deterministic pseudo-random generator (mulberry32) for reproducible noise in tests. */
export function seededRandom(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Displace every vertex along a random direction by up to `amplitude` (in place). */
export function addNoise(mesh: MeshData, amplitude: number, seed = 1): MeshData {
  const rnd = seededRandom(seed);
  const P = mesh.positions;
  for (let i = 0; i < P.length; i++) P[i] += (rnd() * 2 - 1) * amplitude;
  return mesh;
}
