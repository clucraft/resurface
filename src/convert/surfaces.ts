/**
 * Analytic surfaces used by the STL → CAD converter: the reference primitives (plane, cylinder,
 * cone, sphere) plus the torus (fillets around round features), with distance / normal kernels
 * and fast fits that start from a good initial guess derived from facet normals.
 */
import type { ConeDef, CylinderDef, PlaneDef, PrimitiveDef, SphereDef, Vec3 } from "../core/types";
import { eigenSym3 } from "../core/linalg";
import { cross, dot, normalize } from "../core/vec";
import {
  distanceKernel as primDistance,
  fitCone,
  fitCylinder,
  fitPlane,
  fitSphere,
  normalKernel as primNormal,
  projectToPrimitive,
  type DistanceKernel,
  type NormalKernel,
} from "../fit/primitives";
import { levenbergMarquardt, perpBasis, type LMProblem } from "../fit/common";
import { kasaCircle, refineCircle } from "../fit/fit2d";

export interface TorusDef {
  kind: "torus";
  center: Vec3;
  /** unit axis */
  axis: Vec3;
  /** radius of the spine circle */
  major: number;
  /** tube radius */
  minor: number;
}

export type ConvSurface = PrimitiveDef | TorusDef;
export type SurfKind = ConvSurface["kind"];

/** Degrees of freedom (model-selection tie breaks prefer fewer). */
export const SURF_DOF: Record<SurfKind, number> = { plane: 3, sphere: 4, cylinder: 5, cone: 6, torus: 7 };

export function surfDistance(s: ConvSurface): DistanceKernel {
  if (s.kind !== "torus") return primDistance(s);
  const [cx, cy, cz] = s.center;
  const [ax, ay, az] = s.axis;
  const R = s.major, r = s.minor;
  return (x, y, z) => {
    const dx = x - cx, dy = y - cy, dz = z - cz;
    const h = dx * ax + dy * ay + dz * az;
    const wx = dx - h * ax, wy = dy - h * ay, wz = dz - h * az;
    const rho = Math.sqrt(wx * wx + wy * wy + wz * wz);
    return Math.hypot(rho - R, h) - r;
  };
}

export function surfNormal(s: ConvSurface): NormalKernel {
  if (s.kind !== "torus") return primNormal(s);
  const [cx, cy, cz] = s.center;
  const [ax, ay, az] = s.axis;
  const R = s.major;
  const [px, py, pz] = perpBasis(s.axis)[0];
  return (x, y, z, out) => {
    const dx = x - cx, dy = y - cy, dz = z - cz;
    const h = dx * ax + dy * ay + dz * az;
    let wx = dx - h * ax, wy = dy - h * ay, wz = dz - h * az;
    const rho = Math.sqrt(wx * wx + wy * wy + wz * wz);
    if (rho > 1e-300) {
      wx /= rho;
      wy /= rho;
      wz /= rho;
    } else {
      wx = px;
      wy = py;
      wz = pz;
    }
    // from the spine point towards p
    const sx = dx - R * wx, sy = dy - R * wy, sz = dz - R * wz;
    const l = Math.sqrt(sx * sx + sy * sy + sz * sz) || 1;
    out[0] = sx / l;
    out[1] = sy / l;
    out[2] = sz / l;
  };
}

export function projectToSurface(s: ConvSurface, p: Vec3): Vec3 {
  if (s.kind !== "torus") return projectToPrimitive(s, p);
  const d = surfDistance(s)(p[0], p[1], p[2]);
  const n = new Float64Array(3);
  surfNormal(s)(p[0], p[1], p[2], n);
  return [p[0] - d * n[0], p[1] - d * n[1], p[2] - d * n[2]];
}

/** Smallest radius of curvature (Infinity for planes) — used to scale normal tolerances. */
export function minCurvatureRadius(s: ConvSurface): number {
  switch (s.kind) {
    case "plane":
      return Infinity;
    case "cylinder":
    case "sphere":
      return s.radius;
    case "cone":
      return Math.max(1e-9, Math.abs(s.hMin) * Math.tan(s.halfAngle));
    case "torus":
      return s.minor;
  }
}

/** Largest geometric size of the surface parameters (for sanity checks against the model size). */
export function surfaceScale(s: ConvSurface): number {
  switch (s.kind) {
    case "plane":
      return 0;
    case "cylinder":
    case "sphere":
      return s.radius;
    case "cone":
      return Math.abs(s.hMax) / Math.max(Math.cos(s.halfAngle), 1e-9);
    case "torus":
      return s.major + s.minor;
  }
}

// ---------------------------------------------------------------------------
// Fitting
// ---------------------------------------------------------------------------

export interface FitInput {
  /** packed xyz of the mesh */
  positions: ArrayLike<number>;
  /** vertex indices of the points to fit */
  verts: ArrayLike<number>;
  /** unit facet normals with area weights (oriented outward) of the triangles around the points */
  normals: Vec3[];
  weights: number[];
}

/** Fit a surface of the given kind; null when the kind does not apply to the data. */
export function fitSurface(kind: SurfKind, input: FitInput, initial?: ConvSurface): ConvSurface | null {
  try {
    switch (kind) {
      case "plane":
        return fitPlane(input.positions, { indices: input.verts, maxIterations: 30 }).def;
      case "cylinder":
        return fitCylinderFast(input, initial?.kind === "cylinder" ? initial : undefined);
      case "cone":
        return fitConeFast(input, initial?.kind === "cone" ? initial : undefined);
      case "sphere":
        return fitSphere(input.positions, { indices: input.verts, maxIterations: 40, initial: initial?.kind === "sphere" ? initial : undefined }).def;
      case "torus":
        return fitTorus(input, initial?.kind === "torus" ? initial : undefined);
    }
  } catch {
    return null;
  }
  return null;
}

/** Weighted scatter matrix of the normals (optionally centred). */
function normalScatter(input: FitInput, centred: boolean): Float64Array {
  const m = new Float64Array(9);
  let W = 0;
  const mean: Vec3 = [0, 0, 0];
  if (centred) {
    input.normals.forEach((n, i) => {
      const w = input.weights[i];
      W += w;
      mean[0] += w * n[0];
      mean[1] += w * n[1];
      mean[2] += w * n[2];
    });
    if (W > 0) for (let k = 0; k < 3; k++) mean[k] /= W;
  }
  input.normals.forEach((n0, i) => {
    const w = input.weights[i];
    const n = [n0[0] - mean[0], n0[1] - mean[1], n0[2] - mean[2]];
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) m[3 * a + b] += w * n[a] * n[b];
  });
  return m;
}

function centroid(input: FitInput): Vec3 {
  const c: Vec3 = [0, 0, 0];
  const P = input.positions;
  const n = input.verts.length;
  for (let k = 0; k < n; k++) {
    const i = input.verts[k];
    c[0] += P[3 * i];
    c[1] += P[3 * i + 1];
    c[2] += P[3 * i + 2];
  }
  return [c[0] / n, c[1] / n, c[2] / n];
}

/** Project the points onto the plane ⟂ axis (2D coords in the perpendicular basis). */
function projectPerp(input: FitInput, axis: Vec3, origin: Vec3): { xy: Float64Array; u: Vec3; v: Vec3; h: Float64Array } {
  const [u, v] = perpBasis(axis);
  const n = input.verts.length;
  const xy = new Float64Array(2 * n);
  const h = new Float64Array(n);
  const P = input.positions;
  for (let k = 0; k < n; k++) {
    const i = input.verts[k];
    const d: Vec3 = [P[3 * i] - origin[0], P[3 * i + 1] - origin[1], P[3 * i + 2] - origin[2]];
    xy[2 * k] = dot(d, u);
    xy[2 * k + 1] = dot(d, v);
    h[k] = dot(d, axis);
  }
  return { xy, u, v, h };
}

function fitCylinderFast(input: FitInput, initial?: CylinderDef): CylinderDef | null {
  const opts = { indices: input.verts, maxIterations: 40 };
  if (initial) return fitCylinder(input.positions, { ...opts, initial }).def;
  if (input.verts.length < 5) return null;
  // cylinder facets contain the axis direction: their normals are ⟂ to it
  const { values, vectors } = eigenSym3(normalScatter(input, false));
  const axis = normalize(vectors[0]);
  if (!(values[0] <= 0.05 * (values[1] + values[2]))) return null;
  const c = centroid(input);
  const { xy, u, v } = projectPerp(input, axis, c);
  const n = input.verts.length;
  const k0 = kasaCircle(xy, n);
  if (!k0 || !Number.isFinite(k0.r)) return null;
  const k1 = refineCircle(xy, n, k0, { maxIterations: 20 });
  const origin: Vec3 = [c[0] + k1.cx * u[0] + k1.cy * v[0], c[1] + k1.cx * u[1] + k1.cy * v[1], c[2] + k1.cx * u[2] + k1.cy * v[2]];
  const init: CylinderDef = { kind: "cylinder", origin, axis, radius: k1.r, hMin: 0, hMax: 0 };
  return fitCylinder(input.positions, { ...opts, initial: init }).def;
}

/**
 * Point on the axis (direction known) where the normal lines meet: every normal of a surface of
 * revolution intersects its axis, so in the plane ⟂ axis the projected normal lines all pass
 * through the axis point (2D least-squares intersection).
 */
function axisPointFromNormals(input: FitInput, axis: Vec3, triCentres: Vec3[]): Vec3 | null {
  const [u, v] = perpBasis(axis);
  let a11 = 0, a12 = 0, a22 = 0, b1 = 0, b2 = 0;
  input.normals.forEach((n, i) => {
    const w = input.weights[i];
    const nx = dot(n, u), ny = dot(n, v);
    const l = Math.hypot(nx, ny);
    if (l < 1e-6) return;
    // line through q with direction d: constraint (x - q) · m = 0 with m ⟂ d
    const mx = -ny / l, my = nx / l;
    const q = triCentres[i];
    const qx = dot(q, u), qy = dot(q, v);
    const c = mx * qx + my * qy;
    a11 += w * mx * mx;
    a12 += w * mx * my;
    a22 += w * my * my;
    b1 += w * mx * c;
    b2 += w * my * c;
  });
  const det = a11 * a22 - a12 * a12;
  if (!(Math.abs(det) > 1e-12 * Math.max(1, a11 * a22))) return null;
  const x = (b1 * a22 - b2 * a12) / det;
  const y = (a11 * b2 - a12 * b1) / det;
  return [x * u[0] + y * v[0], x * u[1] + y * v[1], x * u[2] + y * v[2]];
}

/** Facet centres are passed through `normals`' companion array set by the caller (see FitInputEx). */
export interface FitInputEx extends FitInput {
  centres: Vec3[];
}

function fitConeFast(input: FitInput, initial?: ConeDef): ConeDef | null {
  const opts = { indices: input.verts, maxIterations: 40 };
  if (initial) {
    const r = fitCone(input.positions, { ...opts, initial });
    return r.nearCylinder ? null : r.def;
  }
  const centres = (input as FitInputEx).centres;
  if (!centres || input.verts.length < 6) return null;
  // cone normals keep a constant angle with the axis: their tips lie in a plane ⟂ axis
  const { vectors } = eigenSym3(normalScatter(input, true));
  let axis = normalize(vectors[0]);
  let W = 0, s = 0;
  input.normals.forEach((n, i) => {
    W += input.weights[i];
    s += input.weights[i] * dot(n, axis);
  });
  const mean = s / Math.max(W, 1e-300);
  if (Math.abs(mean) < Math.sin((0.5 * Math.PI) / 180)) return null; // a cylinder
  const p0 = axisPointFromNormals(input, axis, centres);
  if (!p0) return null;
  // profile: rho = rho0 + k h  → apex where rho = 0
  const { xy, h } = projectPerp(input, axis, p0);
  const n = input.verts.length;
  let sh = 0, sr = 0, shh = 0, shr = 0;
  for (let k = 0; k < n; k++) {
    const rho = Math.hypot(xy[2 * k], xy[2 * k + 1]);
    sh += h[k];
    sr += rho;
    shh += h[k] * h[k];
    shr += h[k] * rho;
  }
  const den = n * shh - sh * sh;
  if (!(Math.abs(den) > 1e-18)) return null;
  const k = (n * shr - sh * sr) / den;
  const rho0 = (sr - k * sh) / n;
  if (!(Math.abs(k) > 1e-9)) return null;
  // apex where the profile reaches the axis; the cone axis points into the opening
  const hApex = -rho0 / k;
  const apex: Vec3 = [p0[0] + hApex * axis[0], p0[1] + hApex * axis[1], p0[2] + hApex * axis[2]];
  if (k < 0) axis = [-axis[0], -axis[1], -axis[2]];
  const half = Math.atan(Math.abs(k));
  if (!(half > 0.5 * (Math.PI / 180) && half < 89 * (Math.PI / 180))) return null;
  const init: ConeDef = { kind: "cone", apex, axis, halfAngle: half, hMin: 0, hMax: 0 };
  const r = fitCone(input.positions, { ...opts, initial: init });
  return r.nearCylinder ? null : r.def;
}

// ---------------------------------------------------------------------------
// Torus
// ---------------------------------------------------------------------------

interface TorusState {
  c: Vec3;
  a: Vec3;
  R: number;
  r: number;
}

/** Axis of a surface of revolution from point/normal pairs (normals all meet the axis). */
function revolutionAxis(input: FitInputEx): { axis: Vec3; point: Vec3 } | null {
  // Plücker coordinates: normal line (n, p × n) meets the axis (a, ā) iff a·(p × n) + ā·n = 0.
  // Normalise coordinates for conditioning.
  const c0 = centroid(input);
  let scale = 0;
  input.centres.forEach((p) => (scale = Math.max(scale, Math.hypot(p[0] - c0[0], p[1] - c0[1], p[2] - c0[2]))));
  if (!(scale > 0)) return null;
  const M = new Float64Array(36);
  input.normals.forEach((n, i) => {
    const w = input.weights[i];
    const q = input.centres[i];
    const p: Vec3 = [(q[0] - c0[0]) / scale, (q[1] - c0[1]) / scale, (q[2] - c0[2]) / scale];
    const m = cross(p, n);
    const row = [m[0], m[1], m[2], n[0], n[1], n[2]];
    for (let a = 0; a < 6; a++) for (let b = 0; b < 6; b++) M[6 * a + b] += w * row[a] * row[b];
  });
  // minimise over ā for fixed a: ā = -C⁻¹ Bᵀ a; then a = smallest eigvec of A - B C⁻¹ Bᵀ
  const A = new Float64Array(9), B = new Float64Array(9), C = new Float64Array(9);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) {
      A[3 * i + j] = M[6 * i + j];
      B[3 * i + j] = M[6 * i + j + 3];
      C[3 * i + j] = M[6 * (i + 3) + j + 3];
    }
  const Ci = inv3(C);
  if (!Ci) return null;
  // S = A - B Ci Bᵀ
  const BCi = mul3(B, Ci);
  const S = new Float64Array(9);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) {
      let v = 0;
      for (let k = 0; k < 3; k++) v += BCi[3 * i + k] * B[3 * j + k];
      S[3 * i + j] = A[3 * i + j] - v;
    }
  const { vectors } = eigenSym3(S);
  const a = normalize(vectors[0]);
  // ā = -Ci Bᵀ a
  const bt: Vec3 = [0, 0, 0];
  for (let i = 0; i < 3; i++) for (let k = 0; k < 3; k++) bt[i] += B[3 * k + i] * a[k];
  const abar: Vec3 = [0, 0, 0];
  for (let i = 0; i < 3; i++) for (let k = 0; k < 3; k++) abar[i] -= Ci[3 * i + k] * bt[k];
  // ā = point × a  → point closest to the origin: a × ā / |a|²
  const pt = cross(a, abar);
  return { axis: a, point: [c0[0] + pt[0] * scale, c0[1] + pt[1] * scale, c0[2] + pt[2] * scale] };
}

function inv3(m: Float64Array): Float64Array | null {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  const tr = Math.abs(a) + Math.abs(e) + Math.abs(i);
  if (!(Math.abs(det) > 1e-14 * tr * tr * tr)) return null;
  return Float64Array.from([A, -(b * i - c * h), b * f - c * e, B, a * i - c * g, -(a * f - c * d), C, -(a * h - b * g), a * e - b * d].map((x) => x / det));
}

function mul3(x: Float64Array, y: Float64Array): Float64Array {
  const o = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) o[3 * i + j] += x[3 * i + k] * y[3 * k + j];
  return o;
}

function torusResiduals(st: TorusState, P: ArrayLike<number>, verts: ArrayLike<number>, out: Float64Array): void {
  const [cx, cy, cz] = st.c;
  const [ax, ay, az] = st.a;
  for (let k = 0; k < verts.length; k++) {
    const i = verts[k];
    const dx = P[3 * i] - cx, dy = P[3 * i + 1] - cy, dz = P[3 * i + 2] - cz;
    const h = dx * ax + dy * ay + dz * az;
    const wx = dx - h * ax, wy = dy - h * ay, wz = dz - h * az;
    const rho = Math.sqrt(wx * wx + wy * wy + wz * wz);
    out[k] = Math.hypot(rho - st.R, h) - st.r;
  }
}

/** Levenberg–Marquardt torus refinement (analytic Jacobian) from one starting state. */
function refineTorus(P: ArrayLike<number>, verts: ArrayLike<number>, st0: TorusState, maxIterations: number): { st: TorusState; cost: number } {
  const n = verts.length;
  const J = new Float64Array(7);
  const step = (s: TorusState, d: ArrayLike<number>): TorusState => {
    const [u, v] = perpBasis(s.a);
    const a = normalize([s.a[0] + d[3] * u[0] + d[4] * v[0], s.a[1] + d[3] * u[1] + d[4] * v[1], s.a[2] + d[3] * u[2] + d[4] * v[2]]);
    return { c: [s.c[0] + d[0], s.c[1] + d[1], s.c[2] + d[2]], a, R: s.R + d[5], r: s.r + d[6] };
  };
  const res = new Float64Array(n);
  const prob: LMProblem<TorusState> = {
    nParams: 7,
    cost(s) {
      torusResiduals(s, P, verts, res);
      let c = 0;
      for (let k = 0; k < n; k++) c += res[k] * res[k];
      return c;
    },
    linearize(s, JtJ, Jtr) {
      const [u, v] = perpBasis(s.a);
      const [ax, ay, az] = s.a;
      let cost = 0;
      for (let k = 0; k < n; k++) {
        const i = verts[k];
        const qx = P[3 * i] - s.c[0], qy = P[3 * i + 1] - s.c[1], qz = P[3 * i + 2] - s.c[2];
        const h = qx * ax + qy * ay + qz * az;
        const wx = qx - h * ax, wy = qy - h * ay, wz = qz - h * az;
        const rho = Math.sqrt(wx * wx + wy * wy + wz * wz);
        const e = rho - s.R;
        const D = Math.sqrt(e * e + h * h);
        const r = D - s.r;
        cost += r * r;
        if (!(D > 1e-12) || !(rho > 1e-12)) continue;
        const hx = wx / rho, hy = wy / rho, hz = wz / rho;
        J[0] = -(e * hx + h * ax) / D;
        J[1] = -(e * hy + h * ay) / D;
        J[2] = -(e * hz + h * az) / D;
        const qu = qx * u[0] + qy * u[1] + qz * u[2], qv = qx * v[0] + qy * v[1] + qz * v[2];
        const wu = hx * u[0] + hy * u[1] + hz * u[2], wv = hx * v[0] + hy * v[1] + hz * v[2];
        J[3] = (-e * h * wu + h * qu) / D;
        J[4] = (-e * h * wv + h * qv) / D;
        J[5] = -e / D;
        J[6] = -1;
        for (let p = 0; p < 7; p++) {
          Jtr[p] += J[p] * r;
          for (let q = p; q < 7; q++) JtJ[7 * p + q] += J[p] * J[q];
        }
      }
      return cost;
    },
    retract: step,
  };
  const out = levenbergMarquardt(prob, st0, { maxIterations });
  return { st: out.state, cost: out.cost };
}

/**
 * Geometric torus fit (7 dof: centre, axis (2), R, r). Starts from the revolution axis of the
 * facet normals and — because a narrow fillet patch hardly shows the spine curvature — also from a
 * local cylinder (the tube) bent around several candidate spine centres.
 */
export function fitTorus(input: FitInput, initial?: TorusDef): TorusDef | null {
  const P = input.positions;
  let verts = input.verts;
  if (verts.length < 7) return null;
  if (verts.length > 800) {
    const step = verts.length / 800;
    const sub: number[] = [];
    for (let t = 0; t < 800; t++) sub.push(verts[Math.floor(t * step)]);
    verts = sub;
  }
  const starts: TorusState[] = [];
  if (initial) starts.push({ c: initial.center, a: initial.axis, R: initial.major, r: initial.minor });
  else {
    const ex = input as FitInputEx;
    const ax = ex.centres ? revolutionAxis(ex) : null;
    if (ax) {
      // profile in (rho, h) → circle (R, h0, r)
      const n = verts.length;
      const xy = new Float64Array(2 * n);
      for (let k = 0; k < n; k++) {
        const i = verts[k];
        const d: Vec3 = [P[3 * i] - ax.point[0], P[3 * i + 1] - ax.point[1], P[3 * i + 2] - ax.point[2]];
        const h = dot(d, ax.axis);
        xy[2 * k] = Math.hypot(d[0] - h * ax.axis[0], d[1] - h * ax.axis[1], d[2] - h * ax.axis[2]);
        xy[2 * k + 1] = h;
      }
      const k0 = kasaCircle(xy, n);
      if (k0 && Number.isFinite(k0.r)) {
        const k1 = refineCircle(xy, n, k0, { maxIterations: 20 });
        starts.push({ c: [ax.point[0] + k1.cy * ax.axis[0], ax.point[1] + k1.cy * ax.axis[1], ax.point[2] + k1.cy * ax.axis[2]], a: ax.axis, R: k1.cx, r: k1.r });
      }
    }
    let cyl: CylinderDef | null = null;
    try {
      cyl = fitCylinderFast(input);
    } catch {
      cyl = null;
    }
    if (cyl) {
      const d = normalize(cyl.axis);
      const [u, v] = perpBasis(d);
      for (let ai = 0; ai < 8; ai++) {
        const ang = (ai * Math.PI) / 4;
        const m: Vec3 = [Math.cos(ang) * u[0] + Math.sin(ang) * v[0], Math.cos(ang) * u[1] + Math.sin(ang) * v[1], Math.cos(ang) * u[2] + Math.sin(ang) * v[2]];
        const A = normalize(cross(d, m));
        for (const f of [1.5, 3, 6, 12]) {
          const R = cyl.radius * f;
          starts.push({ c: [cyl.origin[0] + R * m[0], cyl.origin[1] + R * m[1], cyl.origin[2] + R * m[2]], a: A, R, r: cyl.radius });
        }
      }
    }
  }
  let best: { st: TorusState; cost: number } | null = null;
  for (const st of starts) {
    if (!(st.R > 0 && st.r > 0)) continue;
    const r = refineTorus(P, verts, st, initial ? 40 : 30);
    if (!Number.isFinite(r.cost) || !(r.st.R > r.st.r * 1.0001 && r.st.r > 0)) continue;
    if (!best || r.cost < best.cost) best = r;
    if (best.cost <= 1e-18 * verts.length) break;
  }
  if (!best || !best.st.a.every(Number.isFinite)) return null;
  const out = best.st;
  return { kind: "torus", center: out.c, axis: out.a, major: out.R, minor: out.r };
}

/** Plane through a triangle (used as a region-growing seed). */
export function planeOfTriangle(origin: Vec3, normal: Vec3): PlaneDef {
  return { kind: "plane", origin, normal };
}

export type { CylinderDef, ConeDef, SphereDef, PlaneDef };
