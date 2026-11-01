/**
 * Plain, serialisable data contracts of the CAD kernel (shared by OcctKernel, the kernel worker
 * and KernelClient). No OCCT types here.
 */
import type {
  BBox3,
  BSplineCurveDef,
  BSplineSurfaceDef,
  Frame,
  Mat4,
  PrimitiveDef,
  Profile2D,
  Vec3,
  Wire2D,
} from "../core/types";
import type { BrepPlan, FaceKind } from "../convert/brepPlan";

/** Handle of a B-rep shape held in the kernel's registry. */
export type ShapeId = number;

export interface ExtrudeArgs {
  frame: Frame;
  profile: Profile2D;
  /** length along `direction` (may be negative) */
  distance: number;
  /** additional length in the opposite direction (two-sided extrude), default 0 */
  distance2?: number;
  /** extrusion direction, default `frame.zAxis` (need not be perpendicular to the sketch plane) */
  direction?: Vec3;
  /** true (default): capped solid(s) from the profile regions; false: lateral surfaces only */
  solid?: boolean;
  /** draft angle in degrees, positive tapers the walls inwards along `direction`; neutral plane = sketch plane */
  draftDeg?: number;
}

export interface ExtrudeWireArgs {
  frame: Frame;
  wire: Wire2D;
  distance: number;
  distance2?: number;
  direction?: Vec3;
}

export interface RevolveArgs {
  frame: Frame;
  /** closed profile (solid revolve, or surfaces when solid=false) */
  profile?: Profile2D;
  /** open/closed wire (surface revolve) — used when `profile` is absent */
  wire?: Wire2D;
  axisOrigin: Vec3;
  axisDir: Vec3;
  /** sweep angle in degrees; |angle| >= 360 gives a full revolution */
  angleDeg: number;
  solid?: boolean;
}

export interface BSplineFaceArgs {
  surface: BSplineSurfaceDef;
  uRange?: [number, number];
  vRange?: [number, number];
}

export interface PatchesToShapeArgs {
  surfaces: BSplineSurfaceDef[];
  sewTolerance: number;
  makeSolid: boolean;
}

export type PrimitiveBounds = { uMin: number; uMax: number; vMin: number; vMax: number } | { size: number };

export interface LoftSection {
  frame: Frame;
  /** must consist of exactly one closed loop */
  profile: Profile2D;
}

export interface LoftArgs {
  sections: LoftSection[] | { curves: BSplineCurveDef[] };
  solid: boolean;
  ruled?: boolean;
}

export type SweepPath = BSplineCurveDef | { points: Vec3[]; /** interpolate a smooth spline (default) or use a polyline */ smooth?: boolean };

export interface SweepArgs {
  frame: Frame;
  profile: Profile2D;
  path: SweepPath;
  /** default true */
  solid?: boolean;
  /** Frenet trihedron instead of corrected Frenet (default false) */
  frenet?: boolean;
}

export type BooleanOp = "union" | "cut" | "intersect";

export interface BooleanOptions {
  /** merge coplanar/co-cylindrical faces of the result (ShapeUpgrade_UnifySameDomain) */
  simplify?: boolean;
}

export interface PatternOptions {
  /** fuse the copies into one shape (default false: compound) */
  fuse?: boolean;
}

export interface TessellateOptions {
  linearDeflection: number;
  angularDeflectionDeg: number;
}

export interface TessellationEdge {
  /** edge index (TopExp::MapShapes order) — the id accepted by fillet/chamfer */
  id: number;
  /** polyline xyz */
  points: Float32Array;
}

export interface Tessellation {
  positions: Float32Array;
  normals: Float32Array;
  /** CCW = outward */
  indices: Uint32Array;
  /** per face (MapShapes order): [firstTriangle, triangleCount]; triangles are ordered by face */
  faceTriangleRanges: Uint32Array;
  edges: TessellationEdge[];
}

export type ShapeKind = "solid" | "shell" | "face" | "compound" | "other";

export interface ShapeInfo {
  type: ShapeKind;
  faces: number;
  edges: number;
  solids: number;
  shells: number;
  closed: boolean;
  /** only when the shape contains solids */
  volume?: number;
  area: number;
  bbox: BBox3;
  valid: boolean;
}

export interface ExportItem {
  id: ShapeId;
  name?: string;
}

export interface StlExportOptions {
  linearDeflection: number;
  angularDeflectionDeg?: number;
}

export interface IgesExportOptions {
  author?: string;
  organization?: string;
  fileName?: string;
  /** max deviation when non-NURBS faces must be approximated (default 1e-4 mm) */
  approximationTolerance?: number;
}

export interface PlanBuildResult {
  id: ShapeId;
  /** kind of every face (TopExp::MapShapes order, as in `tessellate`) */
  faceKinds: FaceKind[];
  /** plan faces that could not be built */
  failedFaces: number[];
  /** plan faces built but not valid (the caller may demote their regions to facets and rebuild) */
  invalidFaces: number[];
  failedEdges: number;
  /** edges left unconnected after sewing (0 for a closed solid) */
  freeEdges: number;
}

/** The kernel API (synchronous in OcctKernel, Promise-returning through KernelClient). */
export interface KernelApi {
  extrude(args: ExtrudeArgs): ShapeId;
  extrudeWire(args: ExtrudeWireArgs): ShapeId;
  revolve(args: RevolveArgs): ShapeId;
  bsplineSurfaceFace(args: BSplineFaceArgs): ShapeId;
  bsplinePatchesToShape(args: PatchesToShapeArgs): ShapeId;
  primitiveFace(def: PrimitiveDef, bounds: PrimitiveBounds): ShapeId;
  /** Shape from a converter B-rep plan (STL → CAD): faces on analytic surfaces, sewn into solids. */
  brepFromPlan(plan: BrepPlan): PlanBuildResult;
  loft(args: LoftArgs): ShapeId;
  sweep(args: SweepArgs): ShapeId;
  boolean(op: BooleanOp, a: ShapeId, b: ShapeId, opts?: BooleanOptions): ShapeId;
  split(a: ShapeId, tool: ShapeId): ShapeId;
  fillet(shape: ShapeId, edgeIds: number[], radius: number): ShapeId;
  chamfer(shape: ShapeId, edgeIds: number[], distance: number): ShapeId;
  shell(shape: ShapeId, faceIdsToRemove: number[], thickness: number): ShapeId;
  thicken(shape: ShapeId, thickness: number): ShapeId;
  transform(shape: ShapeId, m: Mat4): ShapeId;
  mirror(shape: ShapeId, plane: { origin: Vec3; normal: Vec3 }): ShapeId;
  linearPattern(shape: ShapeId, dir: Vec3, count: number, spacing: number, opts?: PatternOptions): ShapeId;
  circularPattern(shape: ShapeId, axisOrigin: Vec3, axisDir: Vec3, count: number, totalAngleDeg: number, opts?: PatternOptions): ShapeId;
  compound(ids: ShapeId[]): ShapeId;
  tessellate(shape: ShapeId, opts: TessellateOptions): Tessellation;
  info(shape: ShapeId): ShapeInfo;
  exportSTEP(items: ExportItem[]): string;
  exportIGES(items: ExportItem[], opts?: IgesExportOptions): string;
  exportSTL(items: ExportItem[], opts: StlExportOptions): Uint8Array<ArrayBuffer>;
  /** OCCT BREP text of a shape (for persistence / undo) */
  serialize(shape: ShapeId): string;
  deserialize(brep: string): ShapeId;
  release(id: ShapeId): void;
  releaseAll(): void;
  count(): number;
}

export type KernelMethod = keyof KernelApi;
