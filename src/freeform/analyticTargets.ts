/**
 * Closed-form closest-point oracles for analytic surfaces (sphere, torus, cylinder, plane,
 * height field) plus small mesh generators. Used by the freeform tests and handy for demos.
 */
import type { ClosestPointHit, ClosestPointQuery, MeshData, Vec3 } from "../core/types";

const hitOf = (p: Vec3, point: Vec3, normal: Vec3): ClosestPointHit => ({
  point,
  normal,
  distance: Math.hypot(p[0] - point[0], p[1] - point[1], p[2] - point[2]),
});

const within = (h: ClosestPointHit, maxDistance?: number): ClosestPointHit | null =>
  maxDistance !== undefined && h.distance > maxDistance ? null : h;

export function sphereTarget(center: Vec3, radius: number): ClosestPointQuery {
  return (p, maxDistance) => {
    let d: Vec3 = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
    let l = Math.hypot(d[0], d[1], d[2]);
    if (l < 1e-300) {
      d = [0, 0, 1];
      l = 1;
    }
    const n: Vec3 = [d[0] / l, d[1] / l, d[2] / l];
    return within(hitOf(p, [center[0] + radius * n[0], center[1] + radius * n[1], center[2] + radius * n[2]], n), maxDistance);
  };
}

/** Ray cast against a sphere (nearest intersection with t >= 0). */
export function sphereRaycast(center: Vec3, radius: number): (o: Vec3, dir: Vec3) => ClosestPointHit | null {
  return (o, dir) => {
    const l = Math.hypot(dir[0], dir[1], dir[2]) || 1;
    const d: Vec3 = [dir[0] / l, dir[1] / l, dir[2] / l];
    const oc: Vec3 = [o[0] - center[0], o[1] - center[1], o[2] - center[2]];
    const b = oc[0] * d[0] + oc[1] * d[1] + oc[2] * d[2];
    const c = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - radius * radius;
    const disc = b * b - c;
    if (disc < 0) return null;
    const sq = Math.sqrt(disc);
    let t = -b - sq;
    if (t < 0) t = -b + sq;
    if (t < 0) return null;
    const point: Vec3 = [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
    const n: Vec3 = [(point[0] - center[0]) / radius, (point[1] - center[1]) / radius, (point[2] - center[2]) / radius];
    return { point, normal: n, distance: t };
  };
}

/** Torus around the +z axis through `center`: major radius R, minor radius r. */
export function torusTarget(center: Vec3, R: number, r: number): ClosestPointQuery {
  return (p, maxDistance) => {
    const x = p[0] - center[0], y = p[1] - center[1], z = p[2] - center[2];
    let rho = Math.hypot(x, y);
    let cx = 1, cy = 0;
    if (rho > 1e-300) {
      cx = x / rho;
      cy = y / rho;
    } else rho = 0;
    const ring: Vec3 = [R * cx, R * cy, 0];
    let d: Vec3 = [x - ring[0], y - ring[1], z];
    let l = Math.hypot(d[0], d[1], d[2]);
    if (l < 1e-300) {
      d = [cx, cy, 0];
      l = 1;
    }
    const n: Vec3 = [d[0] / l, d[1] / l, d[2] / l];
    return within(hitOf(p, [center[0] + ring[0] + r * n[0], center[1] + ring[1] + r * n[1], center[2] + r * n[2]], n), maxDistance);
  };
}

/** Infinite cylinder: axis through `origin` with unit direction `axis`. */
export function cylinderTarget(origin: Vec3, axis: Vec3, radius: number): ClosestPointQuery {
  const al = Math.hypot(axis[0], axis[1], axis[2]) || 1;
  const a: Vec3 = [axis[0] / al, axis[1] / al, axis[2] / al];
  return (p, maxDistance) => {
    const v: Vec3 = [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]];
    const t = v[0] * a[0] + v[1] * a[1] + v[2] * a[2];
    let r: Vec3 = [v[0] - t * a[0], v[1] - t * a[1], v[2] - t * a[2]];
    let l = Math.hypot(r[0], r[1], r[2]);
    if (l < 1e-300) {
      r = Math.abs(a[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
      l = 1;
    }
    const n: Vec3 = [r[0] / l, r[1] / l, r[2] / l];
    const q: Vec3 = [origin[0] + t * a[0] + radius * n[0], origin[1] + t * a[1] + radius * n[1], origin[2] + t * a[2] + radius * n[2]];
    return within(hitOf(p, q, n), maxDistance);
  };
}

export function planeTarget(origin: Vec3, normal: Vec3): ClosestPointQuery {
  const nl = Math.hypot(normal[0], normal[1], normal[2]) || 1;
  const n: Vec3 = [normal[0] / nl, normal[1] / nl, normal[2] / nl];
  return (p, maxDistance) => {
    const s = (p[0] - origin[0]) * n[0] + (p[1] - origin[1]) * n[1] + (p[2] - origin[2]) * n[2];
    return within(hitOf(p, [p[0] - s * n[0], p[1] - s * n[1], p[2] - s * n[2]], n), maxDistance);
  };
}

export interface HeightField {
  f: (x: number, y: number) => number;
  /** gradient [df/dx, df/dy] */
  grad: (x: number, y: number) => [number, number];
}

/** Wavy height field z = A sin(kx x) cos(ky y). */
export function wavyHeightField(A: number, kx: number, ky: number): HeightField {
  return {
    f: (x, y) => A * Math.sin(kx * x) * Math.cos(ky * y),
    grad: (x, y) => [A * kx * Math.cos(kx * x) * Math.cos(ky * y), -A * ky * Math.sin(kx * x) * Math.sin(ky * y)],
  };
}

/** Closest point on the graph z = f(x, y) (Gauss–Newton with step control; exact to ~1e-12 for gentle fields). */
export function heightFieldTarget(hf: HeightField): ClosestPointQuery {
  return (p, maxDistance) => {
    let x = p[0], y = p[1];
    const g = (x: number, y: number) => {
      const dz = hf.f(x, y) - p[2];
      return (x - p[0]) ** 2 + (y - p[1]) ** 2 + dz * dz;
    };
    let cur = g(x, y);
    for (let it = 0; it < 100; it++) {
      const z = hf.f(x, y);
      const [fx, fy] = hf.grad(x, y);
      const r = z - p[2];
      const gx = x - p[0] + r * fx, gy = y - p[1] + r * fy;
      // Gauss–Newton normal matrix of the residual vector (x - px, y - py, f - pz)
      const a = 1 + fx * fx, b = fx * fy, c = 1 + fy * fy;
      const det = a * c - b * b;
      const dx = -(c * gx - b * gy) / det, dy = -(-b * gx + a * gy) / det;
      let step = 1;
      let nx = x + dx, ny = y + dy, nv = g(nx, ny);
      while (nv > cur && step > 1e-6) {
        step *= 0.5;
        nx = x + step * dx;
        ny = y + step * dy;
        nv = g(nx, ny);
      }
      if (!(nv <= cur)) break;
      const moved = Math.hypot(nx - x, ny - y);
      x = nx;
      y = ny;
      cur = nv;
      if (moved < 1e-13 * (1 + Math.abs(x) + Math.abs(y))) break;
    }
    const z = hf.f(x, y);
    const [fx, fy] = hf.grad(x, y);
    const nl = Math.hypot(fx, fy, 1);
    return within(hitOf(p, [x, y, z], [-fx / nl, -fy / nl, 1 / nl]), maxDistance);
  };
}

/** Icosphere triangle mesh (outward CCW). subdivisions = 0 -> 20 triangles, 1 -> 80, 2 -> 320, 3 -> 1280. */
export function icosphereMesh(radius: number, subdivisions: number, center: Vec3 = [0, 0, 0]): MeshData {
  const t = (1 + Math.sqrt(5)) / 2;
  let verts: Vec3[] = [
    [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
    [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
    [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
  ];
  let faces: [number, number, number][] = [
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
    [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
    [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
  ];
  const norm = (v: Vec3): Vec3 => {
    const l = Math.hypot(v[0], v[1], v[2]);
    return [v[0] / l, v[1] / l, v[2] / l];
  };
  verts = verts.map(norm);
  for (let s = 0; s < subdivisions; s++) {
    const mid = new Map<string, number>();
    const midpoint = (a: number, b: number): number => {
      const k = a < b ? `${a}_${b}` : `${b}_${a}`;
      let m = mid.get(k);
      if (m === undefined) {
        const p = verts[a], q = verts[b];
        m = verts.length;
        verts.push(norm([(p[0] + q[0]) / 2, (p[1] + q[1]) / 2, (p[2] + q[2]) / 2]));
        mid.set(k, m);
      }
      return m;
    };
    const nf: [number, number, number][] = [];
    for (const [a, b, c] of faces) {
      const ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a);
      nf.push([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]);
    }
    faces = nf;
  }
  const positions = new Float32Array(verts.length * 3);
  verts.forEach((v, i) => positions.set([center[0] + radius * v[0], center[1] + radius * v[1], center[2] + radius * v[2]], 3 * i));
  return { positions, indices: Uint32Array.from(faces.flat()) };
}

/** Regular triangulated height-field patch over [x0,x1] x [y0,y1] (open mesh, normals up). */
export function heightFieldMesh(hf: HeightField, x0: number, x1: number, y0: number, y1: number, nx: number, ny: number): MeshData {
  const positions = new Float32Array((nx + 1) * (ny + 1) * 3);
  for (let j = 0; j <= ny; j++)
    for (let i = 0; i <= nx; i++) {
      const x = x0 + ((x1 - x0) * i) / nx, y = y0 + ((y1 - y0) * j) / ny;
      positions.set([x, y, hf.f(x, y)], 3 * (j * (nx + 1) + i));
    }
  const idx: number[] = [];
  const id = (i: number, j: number) => j * (nx + 1) + i;
  for (let j = 0; j < ny; j++)
    for (let i = 0; i < nx; i++) {
      // alternate the diagonal for a less biased triangulation
      if ((i + j) % 2 === 0) idx.push(id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j), id(i + 1, j + 1), id(i, j + 1));
      else idx.push(id(i, j), id(i + 1, j), id(i, j + 1), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1));
    }
  return { positions, indices: Uint32Array.from(idx) };
}
