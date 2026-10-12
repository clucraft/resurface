/**
 * Synthetic point-cloud generators for tests and demos (deterministic, seeded).
 */
import type { Mat4, Vec3 } from "../../core/types";
import { cross, normalize } from "../../core/vec";
import { gaussianFrom, mulberry32 } from "../common";
import { mat4FromRotation, quatToRotation } from "../transform";

export interface Cloud {
  positions: Float32Array;
  normals: Float32Array;
}

export interface Rand {
  uniform: () => number;
  gauss: () => number;
}

export function makeRand(seed: number): Rand {
  const u = mulberry32(seed);
  return { uniform: u, gauss: gaussianFrom(u) };
}

/** Orthonormal pair perpendicular to unit a. */
export function perp(a: Vec3): [Vec3, Vec3] {
  const h: Vec3 = Math.abs(a[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const e1 = normalize(cross(a, h));
  return [e1, cross(a, e1)];
}

/** Uniformly random rotation (as Mat4) with optional translation. */
export function randomRigid(r: Rand, maxAngleRad = Math.PI, translation: Vec3 = [0, 0, 0]): Mat4 {
  // random axis, angle uniform in [0, maxAngle]
  const axis = normalize([r.gauss(), r.gauss(), r.gauss()]);
  const ang = r.uniform() * maxAngleRad;
  const s = Math.sin(ang / 2);
  const R = quatToRotation(Math.cos(ang / 2), axis[0] * s, axis[1] * s, axis[2] * s);
  return mat4FromRotation(R, translation);
}

function finish(pts: number[], nrm: number[], noise: number, r: Rand): Cloud {
  const n = pts.length / 3;
  const positions = new Float32Array(3 * n), normals = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    const g = noise > 0 ? noise * r.gauss() : 0;
    for (let c = 0; c < 3; c++) {
      positions[3 * i + c] = pts[3 * i + c] + g * nrm[3 * i + c];
      normals[3 * i + c] = nrm[3 * i + c];
    }
  }
  return { positions, normals };
}

/** Square planar patch of side `size` centred at origin. */
export function planePatch(o: { origin: Vec3; normal: Vec3; size: number; n: number; noise?: number; seed?: number }): Cloud {
  const r = makeRand(o.seed ?? 1);
  const nn = normalize(o.normal);
  const [e1, e2] = perp(nn);
  const pts: number[] = [], nrm: number[] = [];
  for (let i = 0; i < o.n; i++) {
    const a = (r.uniform() - 0.5) * o.size, b = (r.uniform() - 0.5) * o.size;
    for (let c = 0; c < 3; c++) pts.push(o.origin[c] + a * e1[c] + b * e2[c]);
    nrm.push(...nn);
  }
  return finish(pts, nrm, o.noise ?? 0, r);
}

/** Spherical cap around direction `pole` with polar half-angle `capDeg` (180 = full sphere). */
export function spherePatch(o: { center: Vec3; radius: number; pole?: Vec3; capDeg: number; n: number; noise?: number; seed?: number }): Cloud {
  const r = makeRand(o.seed ?? 2);
  const pole = normalize(o.pole ?? [0, 0, 1]);
  const [e1, e2] = perp(pole);
  const cmin = Math.cos((o.capDeg * Math.PI) / 180);
  const pts: number[] = [], nrm: number[] = [];
  for (let i = 0; i < o.n; i++) {
    const z = cmin + (1 - cmin) * r.uniform();
    const phi = 2 * Math.PI * r.uniform();
    const s = Math.sqrt(Math.max(0, 1 - z * z));
    const d: Vec3 = [0, 1, 2].map((c) => z * pole[c] + s * Math.cos(phi) * e1[c] + s * Math.sin(phi) * e2[c]) as Vec3;
    for (let c = 0; c < 3; c++) pts.push(o.center[c] + o.radius * d[c]);
    nrm.push(...d);
  }
  return finish(pts, nrm, o.noise ?? 0, r);
}

/** Cylinder patch: arc of `arcDeg` (centred on angle 0 = perp(axis)[0]) and given length along the axis from `origin`. */
export function cylinderPatch(o: {
  origin: Vec3;
  axis: Vec3;
  radius: number;
  arcDeg: number;
  length: number;
  n: number;
  noise?: number;
  seed?: number;
  startDeg?: number;
}): Cloud {
  const r = makeRand(o.seed ?? 3);
  const a = normalize(o.axis);
  const [e1, e2] = perp(a);
  const pts: number[] = [], nrm: number[] = [];
  const a0 = ((o.startDeg ?? -o.arcDeg / 2) * Math.PI) / 180;
  for (let i = 0; i < o.n; i++) {
    const th = a0 + (r.uniform() * o.arcDeg * Math.PI) / 180;
    const h = r.uniform() * o.length;
    const d: Vec3 = [0, 1, 2].map((c) => Math.cos(th) * e1[c] + Math.sin(th) * e2[c]) as Vec3;
    for (let c = 0; c < 3; c++) pts.push(o.origin[c] + h * a[c] + o.radius * d[c]);
    nrm.push(...d);
  }
  return finish(pts, nrm, o.noise ?? 0, r);
}

/** Cone patch between heights hMin..hMax from the apex (measured along the axis). */
export function conePatch(o: {
  apex: Vec3;
  axis: Vec3;
  halfAngle: number;
  hMin: number;
  hMax: number;
  arcDeg: number;
  n: number;
  noise?: number;
  seed?: number;
}): Cloud {
  const r = makeRand(o.seed ?? 4);
  const a = normalize(o.axis);
  const [e1, e2] = perp(a);
  const ta = Math.tan(o.halfAngle), ca = Math.cos(o.halfAngle), sa = Math.sin(o.halfAngle);
  const pts: number[] = [], nrm: number[] = [];
  for (let i = 0; i < o.n; i++) {
    const th = ((r.uniform() - 0.5) * o.arcDeg * Math.PI) / 180;
    // uniform area sampling: h ~ sqrt(uniform between hMin², hMax²)
    const h = Math.sqrt(o.hMin * o.hMin + r.uniform() * (o.hMax * o.hMax - o.hMin * o.hMin));
    const d: Vec3 = [0, 1, 2].map((c) => Math.cos(th) * e1[c] + Math.sin(th) * e2[c]) as Vec3;
    for (let c = 0; c < 3; c++) pts.push(o.apex[c] + h * a[c] + h * ta * d[c]);
    nrm.push(...([0, 1, 2].map((c) => ca * d[c] - sa * a[c]) as number[]));
  }
  return finish(pts, nrm, o.noise ?? 0, r);
}

/** Uniform random outliers inside the bounding box of `cloud` grown by `margin`. Returns a new cloud. */
export function withOutliers(cloud: Cloud, fraction: number, margin: number, seed = 9): Cloud {
  const r = makeRand(seed);
  const n = cloud.positions.length / 3;
  const k = Math.round(n * fraction);
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++)
    for (let c = 0; c < 3; c++) {
      lo[c] = Math.min(lo[c], cloud.positions[3 * i + c]);
      hi[c] = Math.max(hi[c], cloud.positions[3 * i + c]);
    }
  const positions = new Float32Array(3 * (n + k)), normals = new Float32Array(3 * (n + k));
  positions.set(cloud.positions);
  normals.set(cloud.normals);
  for (let i = n; i < n + k; i++) {
    const nn = normalize([r.gauss(), r.gauss(), r.gauss()]);
    for (let c = 0; c < 3; c++) {
      positions[3 * i + c] = lo[c] - margin + r.uniform() * (hi[c] - lo[c] + 2 * margin);
      normals[3 * i + c] = nn[c];
    }
  }
  return { positions, normals };
}

/** Concatenate clouds. */
export function concat(...clouds: Cloud[]): Cloud {
  const n = clouds.reduce((s, c) => s + c.positions.length, 0);
  const positions = new Float32Array(n), normals = new Float32Array(n);
  let o = 0;
  for (const c of clouds) {
    positions.set(c.positions, o);
    normals.set(c.normals, o);
    o += c.positions.length;
  }
  return { positions, normals };
}

/** Apply a rigid transform to a cloud (new arrays). */
export function transformCloud(m: Mat4, c: Cloud): Cloud {
  const n = c.positions.length / 3;
  const positions = new Float32Array(3 * n), normals = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    const x = c.positions[3 * i], y = c.positions[3 * i + 1], z = c.positions[3 * i + 2];
    positions[3 * i] = m[0] * x + m[4] * y + m[8] * z + m[12];
    positions[3 * i + 1] = m[1] * x + m[5] * y + m[9] * z + m[13];
    positions[3 * i + 2] = m[2] * x + m[6] * y + m[10] * z + m[14];
    const a = c.normals[3 * i], b = c.normals[3 * i + 1], d = c.normals[3 * i + 2];
    normals[3 * i] = m[0] * a + m[4] * b + m[8] * d;
    normals[3 * i + 1] = m[1] * a + m[5] * b + m[9] * d;
    normals[3 * i + 2] = m[2] * a + m[6] * b + m[10] * d;
  }
  return { positions, normals };
}

/**
 * Height field z = f(x, y) sampled on a jittered grid over [x0,x1]×[y0,y1] with analytic normals
 * (from the numerical gradient).
 */
export function heightField(
  f: (x: number, y: number) => number,
  o: { x0: number; x1: number; y0: number; y1: number; nx: number; ny: number; noise?: number; seed?: number; jitter?: number; skip?: (x: number, y: number) => boolean },
): Cloud {
  const r = makeRand(o.seed ?? 5);
  const pts: number[] = [], nrm: number[] = [];
  const hx = (o.x1 - o.x0) / (o.nx - 1), hy = (o.y1 - o.y0) / (o.ny - 1);
  const jit = o.jitter ?? 0;
  for (let i = 0; i < o.nx; i++)
    for (let j = 0; j < o.ny; j++) {
      const x = Math.min(o.x1, Math.max(o.x0, o.x0 + i * hx + (r.uniform() - 0.5) * jit * hx));
      const y = Math.min(o.y1, Math.max(o.y0, o.y0 + j * hy + (r.uniform() - 0.5) * jit * hy));
      if (o.skip && o.skip(x, y)) continue;
      const e = 1e-5 * Math.max(1, Math.abs(o.x1 - o.x0));
      const fx = (f(x + e, y) - f(x - e, y)) / (2 * e), fy = (f(x, y + e) - f(x, y - e)) / (2 * e);
      const nn = normalize([-fx, -fy, 1]);
      pts.push(x, y, f(x, y));
      nrm.push(...nn);
    }
  return finish(pts, nrm, o.noise ?? 0, r);
}

/** Closed-form closest point on an axis-aligned box surface (for alignment / symmetry tests). */
export function boxQuery(half: Vec3, m?: Mat4) {
  // m maps box-local → world; queries in world
  const inv = m ? invertRigidLocal(m) : null;
  return (p: Vec3) => {
    const q: Vec3 = inv ? apply(inv, p) : p;
    const ax = Math.abs(q[0]) - half[0], ay = Math.abs(q[1]) - half[1], az = Math.abs(q[2]) - half[2];
    let cp: Vec3, n: Vec3;
    if (ax <= 0 && ay <= 0 && az <= 0) {
      // inside: nearest face
      if (ax >= ay && ax >= az) {
        cp = [Math.sign(q[0] || 1) * half[0], q[1], q[2]];
        n = [Math.sign(q[0] || 1), 0, 0];
      } else if (ay >= az) {
        cp = [q[0], Math.sign(q[1] || 1) * half[1], q[2]];
        n = [0, Math.sign(q[1] || 1), 0];
      } else {
        cp = [q[0], q[1], Math.sign(q[2] || 1) * half[2]];
        n = [0, 0, Math.sign(q[2] || 1)];
      }
    } else {
      cp = [Math.max(-half[0], Math.min(half[0], q[0])), Math.max(-half[1], Math.min(half[1], q[1])), Math.max(-half[2], Math.min(half[2], q[2]))];
      // normal of the face with the largest outside excess
      if (ax >= ay && ax >= az) n = [Math.sign(q[0]), 0, 0];
      else if (ay >= az) n = [0, Math.sign(q[1]), 0];
      else n = [0, 0, Math.sign(q[2])];
    }
    const pw = m ? apply(m, cp) : cp;
    const nw = m ? normalize(applyDir(m, n)) : n;
    return { point: pw, normal: nw, distance: Math.hypot(p[0] - pw[0], p[1] - pw[1], p[2] - pw[2]) };
  };
}

/** Points sampled uniformly on the surface of an axis-aligned box (area-weighted). */
export function boxSurface(half: Vec3, n: number, seed = 6, noise = 0): Cloud {
  const r = makeRand(seed);
  const [a, b, c] = half;
  const areas = [b * c, b * c, a * c, a * c, a * b, a * b];
  const tot = areas.reduce((s, v) => s + v, 0);
  const pts: number[] = [], nrm: number[] = [];
  for (let i = 0; i < n; i++) {
    let t = r.uniform() * tot, f = 0;
    while (f < 5 && t > areas[f]) t -= areas[f++];
    const u = 2 * r.uniform() - 1, v = 2 * r.uniform() - 1;
    const s = f % 2 === 0 ? 1 : -1;
    let p: Vec3, nn: Vec3;
    if (f < 2) {
      p = [s * a, u * b, v * c];
      nn = [s, 0, 0];
    } else if (f < 4) {
      p = [u * a, s * b, v * c];
      nn = [0, s, 0];
    } else {
      p = [u * a, v * b, s * c];
      nn = [0, 0, s];
    }
    pts.push(...p);
    nrm.push(...nn);
  }
  return finish(pts, nrm, noise, r);
}

function apply(m: Mat4, p: Vec3): Vec3 {
  return [m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12], m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13], m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14]];
}
function applyDir(m: Mat4, p: Vec3): Vec3 {
  return [m[0] * p[0] + m[4] * p[1] + m[8] * p[2], m[1] * p[0] + m[5] * p[1] + m[9] * p[2], m[2] * p[0] + m[6] * p[1] + m[10] * p[2]];
}
function invertRigidLocal(m: Mat4): Mat4 {
  const r: Mat4 = [m[0], m[4], m[8], 0, m[1], m[5], m[9], 0, m[2], m[6], m[10], 0, 0, 0, 0, 1];
  const t = applyDir(r, [m[12], m[13], m[14]]);
  r[12] = -t[0];
  r[13] = -t[1];
  r[14] = -t[2];
  return r;
}

/** Angle in degrees between two directions, ignoring sign. */
export function axisAngleDeg(a: Vec3, b: Vec3): number {
  const c = Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (Math.hypot(...a) * Math.hypot(...b));
  return (Math.acos(Math.min(1, c)) * 180) / Math.PI;
}

/** Distance from point p to the line (o, d). */
export function pointLineDistance(p: Vec3, o: Vec3, d: Vec3): number {
  const dn = normalize(d);
  const v: Vec3 = [p[0] - o[0], p[1] - o[1], p[2] - o[2]];
  const t = v[0] * dn[0] + v[1] * dn[1] + v[2] * dn[2];
  return Math.hypot(v[0] - t * dn[0], v[1] - t * dn[1], v[2] - t * dn[2]);
}
