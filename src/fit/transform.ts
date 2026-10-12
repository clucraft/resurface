/**
 * Rigid-transform helpers (column-major Mat4, like three.js) used by alignment and symmetry.
 */
import type { ConeDef, CylinderDef, LineDef, Mat4, PlaneDef, PointDef, PrimitiveDef, SphereDef, Vec3 } from "../core/types";
import { normalize, transformDir, transformPoint } from "../core/vec";

/** New Float32Array with every xyz of `pos` transformed by m (optionally only `indices`, in that order). */
export function transformPoints(m: Mat4, pos: ArrayLike<number>, indices?: ArrayLike<number>): Float32Array {
  const n = indices ? indices.length : Math.floor(pos.length / 3);
  const out = new Float32Array(3 * n);
  for (let k = 0; k < n; k++) {
    const i = indices ? indices[k] : k;
    const x = pos[3 * i], y = pos[3 * i + 1], z = pos[3 * i + 2];
    out[3 * k] = m[0] * x + m[4] * y + m[8] * z + m[12];
    out[3 * k + 1] = m[1] * x + m[5] * y + m[9] * z + m[13];
    out[3 * k + 2] = m[2] * x + m[6] * y + m[10] * z + m[14];
  }
  return out;
}

/** New Float32Array of normals rotated by the linear part of m and renormalised. */
export function transformNormals(m: Mat4, normals: ArrayLike<number>): Float32Array {
  const n = Math.floor(normals.length / 3);
  const out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    const x = normals[3 * i], y = normals[3 * i + 1], z = normals[3 * i + 2];
    const nx = m[0] * x + m[4] * y + m[8] * z;
    const ny = m[1] * x + m[5] * y + m[9] * z;
    const nz = m[2] * x + m[6] * y + m[10] * z;
    const l = Math.hypot(nx, ny, nz) || 1;
    out[3 * i] = nx / l;
    out[3 * i + 1] = ny / l;
    out[3 * i + 2] = nz / l;
  }
  return out;
}

/** Row-major 3x3 rotation part of a Mat4. */
export function rotationOf(m: Mat4): Float64Array {
  return Float64Array.of(m[0], m[4], m[8], m[1], m[5], m[9], m[2], m[6], m[10]);
}

/** Mat4 from a row-major 3x3 rotation and a translation. */
export function mat4FromRotation(R: ArrayLike<number>, t: Vec3 = [0, 0, 0]): Mat4 {
  return [R[0], R[3], R[6], 0, R[1], R[4], R[7], 0, R[2], R[5], R[8], 0, t[0], t[1], t[2], 1];
}

/** Row-major 3x3 rotation of a unit quaternion (w, x, y, z). */
export function quatToRotation(w: number, x: number, y: number, z: number): Float64Array {
  const l = Math.hypot(w, x, y, z) || 1;
  w /= l;
  x /= l;
  y /= l;
  z /= l;
  return Float64Array.of(
    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
  );
}

/** Row-major rotation for the rotation vector ω (axis * angle), Rodrigues. */
export function rotationFromVector(wx: number, wy: number, wz: number): Float64Array {
  const th = Math.hypot(wx, wy, wz);
  if (th < 1e-300) return Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1);
  const x = wx / th, y = wy / th, z = wz / th;
  const c = Math.cos(th), s = Math.sin(th), t = 1 - c;
  return Float64Array.of(
    t * x * x + c, t * x * y - s * z, t * x * z + s * y,
    t * x * y + s * z, t * y * y + c, t * y * z - s * x,
    t * x * z - s * y, t * y * z + s * x, t * z * z + c,
  );
}

/** Rotation angle (radians) of the linear part of a rigid Mat4 (atan2 form: accurate for tiny angles). */
export function rotationAngle(m: Mat4): number {
  const tr = m[0] + m[5] + m[10];
  // skew part = 2 sin(θ) axis ; R[r][c] = m[c*4 + r]
  const sx = m[6] - m[9], sy = m[8] - m[2], sz = m[1] - m[4];
  return Math.atan2(0.5 * Math.hypot(sx, sy, sz), (tr - 1) / 2);
}

/** Re-orthonormalise the rotation part of a rigid Mat4 (Gram–Schmidt on columns). */
export function orthonormalizeRigid(m: Mat4): Mat4 {
  let x: Vec3 = normalize([m[0], m[1], m[2]]);
  let y: Vec3 = [m[4], m[5], m[6]];
  const dxy = x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
  y = normalize([y[0] - dxy * x[0], y[1] - dxy * x[1], y[2] - dxy * x[2]]);
  const z: Vec3 = [x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]];
  return [x[0], x[1], x[2], 0, y[0], y[1], y[2], 0, z[0], z[1], z[2], 0, m[12], m[13], m[14], 1];
}

/** Difference between two rigid transforms: rotation angle of a⁻¹b and translation distance at `ref`. */
export function rigidDifference(a: Mat4, b: Mat4, ref: Vec3 = [0, 0, 0]): { angle: number; translation: number } {
  // relative rotation Rrel = Ra^T Rb, Rrel[i][j] = Σ_r Ra[r][i] Rb[r][j]
  const rel = (i: number, j: number) => a[i * 4] * b[j * 4] + a[i * 4 + 1] * b[j * 4 + 1] + a[i * 4 + 2] * b[j * 4 + 2];
  const tr = rel(0, 0) + rel(1, 1) + rel(2, 2);
  const sx = rel(2, 1) - rel(1, 2), sy = rel(0, 2) - rel(2, 0), sz = rel(1, 0) - rel(0, 1);
  const angle = Math.atan2(0.5 * Math.hypot(sx, sy, sz), (tr - 1) / 2);
  const pa = transformPoint(a, ref), pb = transformPoint(b, ref);
  return { angle, translation: Math.hypot(pa[0] - pb[0], pa[1] - pb[1], pa[2] - pb[2]) };
}

/** Householder reflection across a plane as a Mat4 (det = -1). */
export function reflectionMatrix(plane: PlaneDef): Mat4 {
  const n = normalize(plane.normal);
  const d = n[0] * plane.origin[0] + n[1] * plane.origin[1] + n[2] * plane.origin[2];
  const [a, b, c] = n;
  return [
    1 - 2 * a * a, -2 * a * b, -2 * a * c, 0,
    -2 * a * b, 1 - 2 * b * b, -2 * b * c, 0,
    -2 * a * c, -2 * b * c, 1 - 2 * c * c, 0,
    2 * d * a, 2 * d * b, 2 * d * c, 1,
  ];
}

/** Reflect a point across a plane. */
export function reflectPoint(p: Vec3, plane: PlaneDef): Vec3 {
  const n = normalize(plane.normal);
  const s = 2 * ((p[0] - plane.origin[0]) * n[0] + (p[1] - plane.origin[1]) * n[1] + (p[2] - plane.origin[2]) * n[2]);
  return [p[0] - s * n[0], p[1] - s * n[1], p[2] - s * n[2]];
}

/** Apply a rigid transform to a plane. */
export function transformPlane(m: Mat4, p: PlaneDef): PlaneDef {
  return { kind: "plane", origin: transformPoint(m, p.origin), normal: normalize(transformDir(m, p.normal)) };
}

/** Apply a rigid transform to a line. */
export function transformLine(m: Mat4, l: LineDef): LineDef {
  return { kind: "line", origin: transformPoint(m, l.origin), direction: normalize(transformDir(m, l.direction)) };
}

/** Apply a rigid transform to a reference point. */
export function transformPointDef(m: Mat4, p: PointDef): PointDef {
  return { kind: "point", position: transformPoint(m, p.position) };
}

/** Apply a rigid transform to any analytic primitive (extents are preserved). */
export function transformPrimitive<T extends PrimitiveDef>(m: Mat4, def: T): T {
  switch (def.kind) {
    case "plane":
      return transformPlane(m, def) as T;
    case "sphere":
      return { ...(def as SphereDef), center: transformPoint(m, def.center) } as T;
    case "cylinder":
      return { ...(def as CylinderDef), origin: transformPoint(m, def.origin), axis: normalize(transformDir(m, def.axis)) } as T;
    case "cone":
      return { ...(def as ConeDef), apex: transformPoint(m, def.apex), axis: normalize(transformDir(m, def.axis)) } as T;
  }
  return def;
}
