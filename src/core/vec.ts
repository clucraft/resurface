/** Small allocation-light vector helpers for Vec2/Vec3 tuples. */
import type { Frame, Mat4, Vec2, Vec3 } from "./types";

export const v3 = (x = 0, y = 0, z = 0): Vec3 => [x, y, z];
export const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const scale = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
export const addScaled = (a: Vec3, b: Vec3, s: number): Vec3 => [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s];
export const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
export const lengthSq = (a: Vec3): number => a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
export const length = (a: Vec3): number => Math.sqrt(lengthSq(a));
export const distance = (a: Vec3, b: Vec3): number => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
export const distanceSq = (a: Vec3, b: Vec3): number => {
  const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
  return dx * dx + dy * dy + dz * dz;
};
export const normalize = (a: Vec3): Vec3 => {
  const l = length(a);
  return l > 1e-300 ? [a[0] / l, a[1] / l, a[2] / l] : [0, 0, 0];
};
export const negate = (a: Vec3): Vec3 => [-a[0], -a[1], -a[2]];
export const lerp = (a: Vec3, b: Vec3, t: number): Vec3 => [
  a[0] + (b[0] - a[0]) * t,
  a[1] + (b[1] - a[1]) * t,
  a[2] + (b[2] - a[2]) * t,
];
export const mid = (a: Vec3, b: Vec3): Vec3 => lerp(a, b, 0.5);
export const clone = (a: Vec3): Vec3 => [a[0], a[1], a[2]];
export const equals = (a: Vec3, b: Vec3, eps = 1e-9): boolean =>
  Math.abs(a[0] - b[0]) <= eps && Math.abs(a[1] - b[1]) <= eps && Math.abs(a[2] - b[2]) <= eps;

/** Read vertex i from a packed xyz array. */
export const getV = (arr: ArrayLike<number>, i: number): Vec3 => [arr[3 * i], arr[3 * i + 1], arr[3 * i + 2]];

/** Any unit vector perpendicular to n (n must be unit). */
export function anyPerpendicular(n: Vec3): Vec3 {
  const ax = Math.abs(n[0]), ay = Math.abs(n[1]), az = Math.abs(n[2]);
  const helper: Vec3 = ax <= ay && ax <= az ? [1, 0, 0] : ay <= az ? [0, 1, 0] : [0, 0, 1];
  return normalize(cross(n, helper));
}

/** Angle between two vectors in radians, [0, PI]. */
export function angleBetween(a: Vec3, b: Vec3): number {
  const d = dot(a, b) / Math.sqrt(lengthSq(a) * lengthSq(b));
  return Math.acos(Math.max(-1, Math.min(1, d)));
}

/** Project p onto the line (o, unit d); returns parameter t with point = o + t d. */
export const projectOnLineT = (p: Vec3, o: Vec3, d: Vec3): number => dot(sub(p, o), d);

/** Signed distance of p to plane (o, unit n). */
export const planeSignedDistance = (p: Vec3, o: Vec3, n: Vec3): number => dot(sub(p, o), n);

// ---------------------------------------------------------------------------
// Frames
// ---------------------------------------------------------------------------

/** Build an orthonormal frame from an origin, a normal and an optional preferred x direction. */
export function frameFromNormal(origin: Vec3, normal: Vec3, xHint?: Vec3): Frame {
  const z = normalize(normal);
  let x: Vec3;
  if (xHint) {
    x = sub(xHint, scale(z, dot(xHint, z)));
    if (length(x) < 1e-9) x = anyPerpendicular(z);
    x = normalize(x);
  } else {
    // Prefer world X projected into the plane, then world Y — gives stable, predictable sketches.
    const wx: Vec3 = [1, 0, 0];
    x = sub(wx, scale(z, dot(wx, z)));
    if (length(x) < 1e-6) {
      const wy: Vec3 = [0, 1, 0];
      x = sub(wy, scale(z, dot(wy, z)));
    }
    x = normalize(x);
  }
  const y = cross(z, x);
  return { origin: clone(origin), xAxis: x, yAxis: y, zAxis: z };
}

export const WORLD_FRAME: Frame = { origin: [0, 0, 0], xAxis: [1, 0, 0], yAxis: [0, 1, 0], zAxis: [0, 0, 1] };

/** 2D frame coordinates -> 3D world point. */
export const frameToWorld = (f: Frame, p: Vec2): Vec3 => [
  f.origin[0] + f.xAxis[0] * p[0] + f.yAxis[0] * p[1],
  f.origin[1] + f.xAxis[1] * p[0] + f.yAxis[1] * p[1],
  f.origin[2] + f.xAxis[2] * p[0] + f.yAxis[2] * p[1],
];

/** 3D world point -> 2D frame coordinates (orthogonal projection onto the frame plane). */
export function worldToFrame(f: Frame, p: Vec3): Vec2 {
  const d = sub(p, f.origin);
  return [dot(d, f.xAxis), dot(d, f.yAxis)];
}

/** Frame -> column-major Mat4 (local to world). */
export function frameToMat4(f: Frame): Mat4 {
  const { xAxis: x, yAxis: y, zAxis: z, origin: o } = f;
  return [x[0], x[1], x[2], 0, y[0], y[1], y[2], 0, z[0], z[1], z[2], 0, o[0], o[1], o[2], 1];
}

// ---------------------------------------------------------------------------
// Mat4 (column-major)
// ---------------------------------------------------------------------------

export const identity4 = (): Mat4 => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

export function mul4(a: Mat4, b: Mat4): Mat4 {
  const r = new Array(16).fill(0);
  for (let c = 0; c < 4; c++)
    for (let rIdx = 0; rIdx < 4; rIdx++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + rIdx] * b[c * 4 + k];
      r[c * 4 + rIdx] = s;
    }
  return r;
}

export const transformPoint = (m: Mat4, p: Vec3): Vec3 => [
  m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
  m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
  m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
];

export const transformDir = (m: Mat4, d: Vec3): Vec3 => [
  m[0] * d[0] + m[4] * d[1] + m[8] * d[2],
  m[1] * d[0] + m[5] * d[1] + m[9] * d[2],
  m[2] * d[0] + m[6] * d[1] + m[10] * d[2],
];

/** Inverse of a rigid transform (rotation + translation). */
export function invertRigid(m: Mat4): Mat4 {
  // R^T, -R^T t
  const r: Mat4 = [m[0], m[4], m[8], 0, m[1], m[5], m[9], 0, m[2], m[6], m[10], 0, 0, 0, 0, 1];
  const t: Vec3 = [m[12], m[13], m[14]];
  const it = transformDir(r, t);
  r[12] = -it[0];
  r[13] = -it[1];
  r[14] = -it[2];
  return r;
}

/** General 4x4 inverse. Returns null if singular. */
export function invert4(m: Mat4): Mat4 | null {
  const inv = new Array(16);
  inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
  inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
  inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
  inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
  inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
  inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
  inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
  inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
  inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
  inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
  inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
  inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
  inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
  inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
  inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
  inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
  let det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
  if (Math.abs(det) < 1e-300) return null;
  det = 1 / det;
  for (let i = 0; i < 16; i++) inv[i] *= det;
  return inv;
}

/** Rotation matrix about a unit axis through the origin. */
export function rotationAxisAngle(axis: Vec3, angle: number): Mat4 {
  const [x, y, z] = normalize(axis);
  const c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
  return [
    t * x * x + c, t * x * y + s * z, t * x * z - s * y, 0,
    t * x * y - s * z, t * y * y + c, t * y * z + s * x, 0,
    t * x * z + s * y, t * y * z - s * x, t * z * z + c, 0,
    0, 0, 0, 1,
  ];
}

export const translation4 = (t: Vec3): Mat4 => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, t[0], t[1], t[2], 1];

/** Apply a Mat4 to a packed xyz Float32Array in place (and optionally normals). */
export function transformPositionsInPlace(m: Mat4, positions: Float32Array, normals?: Float32Array): void {
  const n = positions.length / 3;
  for (let i = 0; i < n; i++) {
    const x = positions[3 * i], y = positions[3 * i + 1], z = positions[3 * i + 2];
    positions[3 * i] = m[0] * x + m[4] * y + m[8] * z + m[12];
    positions[3 * i + 1] = m[1] * x + m[5] * y + m[9] * z + m[13];
    positions[3 * i + 2] = m[2] * x + m[6] * y + m[10] * z + m[14];
  }
  if (normals) {
    // assumes rotation (+uniform scale); renormalise
    for (let i = 0; i < n; i++) {
      const x = normals[3 * i], y = normals[3 * i + 1], z = normals[3 * i + 2];
      let nx = m[0] * x + m[4] * y + m[8] * z;
      let ny = m[1] * x + m[5] * y + m[9] * z;
      let nz = m[2] * x + m[6] * y + m[10] * z;
      const l = Math.hypot(nx, ny, nz) || 1;
      normals[3 * i] = nx / l;
      normals[3 * i + 1] = ny / l;
      normals[3 * i + 2] = nz / l;
    }
  }
}

// ---------------------------------------------------------------------------
// 2D helpers
// ---------------------------------------------------------------------------

export const add2 = (a: Vec2, b: Vec2): Vec2 => [a[0] + b[0], a[1] + b[1]];
export const sub2 = (a: Vec2, b: Vec2): Vec2 => [a[0] - b[0], a[1] - b[1]];
export const scale2 = (a: Vec2, s: number): Vec2 => [a[0] * s, a[1] * s];
export const dot2 = (a: Vec2, b: Vec2): number => a[0] * b[0] + a[1] * b[1];
export const cross2 = (a: Vec2, b: Vec2): number => a[0] * b[1] - a[1] * b[0];
export const len2 = (a: Vec2): number => Math.hypot(a[0], a[1]);
export const dist2 = (a: Vec2, b: Vec2): number => Math.hypot(a[0] - b[0], a[1] - b[1]);
export const norm2 = (a: Vec2): Vec2 => {
  const l = len2(a);
  return l > 1e-300 ? [a[0] / l, a[1] / l] : [0, 0];
};
export const lerp2 = (a: Vec2, b: Vec2, t: number): Vec2 => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];

/** Circle through three 2D points, or null if (nearly) collinear. */
export function circleFrom3Points2(a: Vec2, b: Vec2, c: Vec2): { center: Vec2; radius: number } | null {
  const d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
  if (Math.abs(d) < 1e-12) return null;
  const a2 = a[0] * a[0] + a[1] * a[1], b2 = b[0] * b[0] + b[1] * b[1], c2 = c[0] * c[0] + c[1] * c[1];
  const ux = (a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / d;
  const uy = (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / d;
  return { center: [ux, uy], radius: Math.hypot(a[0] - ux, a[1] - uy) };
}
