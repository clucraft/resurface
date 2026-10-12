/**
 * Shared, framework-free data contracts used across Resurface modules.
 *
 * Conventions:
 *  - Units are millimetres, angles are radians unless a name says `Deg`.
 *  - World is Z-up, right-handed.
 *  - Plain arrays/typed arrays only (no three.js types) so these structures can be
 *    transferred to/from Web Workers and used in Node unit tests.
 */

export type Vec2 = [number, number];
export type Vec3 = [number, number, number];
/** 4x4 matrix, column-major (same layout as three.js Matrix4.elements). */
export type Mat4 = number[];

/** Indexed triangle mesh. */
export interface MeshData {
  /** xyz per vertex, length = 3 * vertexCount */
  positions: Float32Array;
  /** 3 vertex indices per triangle, CCW = outward, length = 3 * triangleCount */
  indices: Uint32Array;
  /** Optional per-vertex unit normals, length = 3 * vertexCount */
  normals?: Float32Array;
  /** Optional per-vertex RGB colours 0..255, length = 3 * vertexCount */
  colors?: Uint8Array;
}

export interface PointCloudData {
  positions: Float32Array;
  normals?: Float32Array;
  colors?: Uint8Array;
}

/** Orthonormal right-handed frame. zAxis is the plane normal for sketch planes. */
export interface Frame {
  origin: Vec3;
  xAxis: Vec3;
  yAxis: Vec3;
  zAxis: Vec3;
}

// ---------------------------------------------------------------------------
// Analytic primitives (reference geometry)
// ---------------------------------------------------------------------------

export interface PlaneDef {
  kind: "plane";
  origin: Vec3;
  /** unit normal */
  normal: Vec3;
}

export interface SphereDef {
  kind: "sphere";
  center: Vec3;
  radius: number;
}

export interface CylinderDef {
  kind: "cylinder";
  /** a point on the axis */
  origin: Vec3;
  /** unit axis direction */
  axis: Vec3;
  radius: number;
  /** extent of the fitted data along the axis, measured from origin (hMin <= hMax) */
  hMin: number;
  hMax: number;
}

export interface ConeDef {
  kind: "cone";
  apex: Vec3;
  /** unit axis, pointing from the apex into the opening of the cone */
  axis: Vec3;
  /** half opening angle in radians, 0 < halfAngle < PI/2 */
  halfAngle: number;
  /** extent of the fitted data along the axis, measured from the apex (0 <= hMin <= hMax) */
  hMin: number;
  hMax: number;
}

export type PrimitiveDef = PlaneDef | SphereDef | CylinderDef | ConeDef;

export interface LineDef {
  kind: "line";
  origin: Vec3;
  /** unit direction */
  direction: Vec3;
}

export interface PointDef {
  kind: "point";
  position: Vec3;
}

/** Signed-distance residual statistics of a fit. */
export interface FitStats {
  /** root-mean-square of residuals */
  rms: number;
  /** max |residual| */
  maxAbs: number;
  /** mean residual (signed) */
  mean: number;
  /** number of points used */
  count: number;
}

export interface FitResult<T> {
  def: T;
  stats: FitStats;
  /** optional per-input-point inlier flags (1 = inlier) when a robust method was used */
  inliers?: Uint8Array;
}

// ---------------------------------------------------------------------------
// NURBS
// ---------------------------------------------------------------------------

/** Clamped (open) B-spline curve. knots is the FULL knot vector: length = controlPoints.length + degree + 1. */
export interface BSplineCurveDef {
  degree: number;
  knots: number[];
  controlPoints: Vec3[];
  /** optional rational weights, same length as controlPoints */
  weights?: number[];
}

export interface BSplineCurve2Def {
  degree: number;
  knots: number[];
  controlPoints: Vec2[];
  weights?: number[];
}

/**
 * Clamped B-spline surface. Full knot vectors:
 *   knotsU.length = nu + degreeU + 1, knotsV.length = nv + degreeV + 1.
 * controlPoints is row-major over U: index = i * nv + j  (i in [0,nu), j in [0,nv)).
 */
export interface BSplineSurfaceDef {
  degreeU: number;
  degreeV: number;
  knotsU: number[];
  knotsV: number[];
  nu: number;
  nv: number;
  controlPoints: Vec3[];
  weights?: number[];
}

// ---------------------------------------------------------------------------
// 2D profiles (sketch output consumed by the CAD kernel)
// ---------------------------------------------------------------------------

export type Segment2D =
  | { type: "line"; a: Vec2; b: Vec2 }
  /** circular arc through three points, from `start` via `mid` to `end` */
  | { type: "arc"; start: Vec2; mid: Vec2; end: Vec2 }
  /** full circle */
  | { type: "circle"; center: Vec2; radius: number }
  | { type: "bspline"; curve: BSplineCurve2Def };

/** A closed loop: segments are ordered and each segment's end equals the next one's start
 * (within tolerance). A loop may consist of a single circle or a single closed bspline. */
export type Loop2D = Segment2D[];

/**
 * A planar profile made of closed loops expressed in the 2D coordinates of a sketch Frame
 * (x along frame.xAxis, y along frame.yAxis). Nesting (outer boundary vs. holes) is
 * derived by the kernel from containment.
 */
export interface Profile2D {
  loops: Loop2D[];
}

/** Open or closed chain of 2D segments (used for surface extrudes/revolves of open sketches). */
export interface Wire2D {
  segments: Segment2D[];
  closed: boolean;
}

// ---------------------------------------------------------------------------
// Misc
// ---------------------------------------------------------------------------

export interface BBox3 {
  min: Vec3;
  max: Vec3;
}

/** Result of a closest-point query against a reference surface (mesh). */
export interface ClosestPointHit {
  point: Vec3;
  /** unit surface normal at the hit (face normal is fine) */
  normal: Vec3;
  /** unsigned distance from query point to `point` */
  distance: number;
  /** triangle index, when the target is a triangle mesh */
  faceIndex?: number;
}

/** Closest-point oracle. Returns null if nothing within maxDistance (when given). */
export type ClosestPointQuery = (p: Vec3, maxDistance?: number) => ClosestPointHit | null;

/** Progress callback: fraction in [0,1] and optional message. */
export type ProgressFn = (fraction: number, message?: string) => void;
