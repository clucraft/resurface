/** Signed distance primitives and CSG operators for procedural sample parts. */
import type { SDF } from "./surfaceNets";

export const box = (cx: number, cy: number, cz: number, hx: number, hy: number, hz: number, r = 0): SDF => (x, y, z) => {
  const qx = Math.abs(x - cx) - hx + r, qy = Math.abs(y - cy) - hy + r, qz = Math.abs(z - cz) - hz + r;
  const ox = Math.max(qx, 0), oy = Math.max(qy, 0), oz = Math.max(qz, 0);
  return Math.hypot(ox, oy, oz) + Math.min(Math.max(qx, qy, qz), 0) - r;
};

export const sphere = (cx: number, cy: number, cz: number, r: number): SDF => (x, y, z) => Math.hypot(x - cx, y - cy, z - cz) - r;

/** Finite cylinder along an axis ('x' | 'y' | 'z'). */
export const cylinder = (axis: "x" | "y" | "z", cx: number, cy: number, cz: number, r: number, halfLen: number): SDF => (x, y, z) => {
  let radial: number, along: number;
  if (axis === "z") {
    radial = Math.hypot(x - cx, y - cy);
    along = Math.abs(z - cz);
  } else if (axis === "y") {
    radial = Math.hypot(x - cx, z - cz);
    along = Math.abs(y - cy);
  } else {
    radial = Math.hypot(y - cy, z - cz);
    along = Math.abs(x - cx);
  }
  const dx = radial - r, dy = along - halfLen;
  return Math.min(Math.max(dx, dy), 0) + Math.hypot(Math.max(dx, 0), Math.max(dy, 0));
};

export const union = (...fs: SDF[]): SDF => (x, y, z) => {
  let d = Infinity;
  for (const f of fs) d = Math.min(d, f(x, y, z));
  return d;
};

export const subtract = (a: SDF, ...bs: SDF[]): SDF => (x, y, z) => {
  let d = a(x, y, z);
  for (const b of bs) d = Math.max(d, -b(x, y, z));
  return d;
};

/** Smooth union (polynomial smooth-min) — creates fillet-like blends of radius ~k. */
export const smoothUnion = (k: number, ...fs: SDF[]): SDF => (x, y, z) => {
  let d = fs[0](x, y, z);
  for (let i = 1; i < fs.length; i++) {
    const b = fs[i](x, y, z);
    const h = Math.max(k - Math.abs(d - b), 0) / k;
    d = Math.min(d, b) - (h * h * k) / 4;
  }
  return d;
};

/** Solid of revolution around Z from a 2D profile SDF in (r, z). */
export const revolve = (profile: (r: number, z: number) => number): SDF => (x, y, z) => profile(Math.hypot(x, y), z);

/** 2D signed distance to a closed polygon (r,z) — negative inside. */
export function polygon2D(pts: [number, number][]): (x: number, y: number) => number {
  return (px, py) => {
    let d = Infinity;
    let inside = false;
    for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
      const [xi, yi] = pts[i], [xj, yj] = pts[j];
      const ex = xj - xi, ey = yj - yi;
      const wx = px - xi, wy = py - yi;
      const t = Math.max(0, Math.min(1, (wx * ex + wy * ey) / (ex * ex + ey * ey)));
      d = Math.min(d, Math.hypot(wx - ex * t, wy - ey * t));
      if (yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) inside = !inside;
    }
    return inside ? -d : d;
  };
}
