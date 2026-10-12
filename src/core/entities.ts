/**
 * Document entity definitions. Entities are immutable snapshots: every change produces a new
 * object (large typed arrays are shared by reference unless replaced), which makes undo/redo
 * and React change detection trivial.
 */
import type {
  BSplineCurveDef,
  BSplineSurfaceDef,
  FitStats,
  Frame,
  LineDef,
  MeshData,
  PointCloudData,
  PointDef,
  PrimitiveDef,
  Vec2,
  Vec3,
} from "./types";
import type { Sketch } from "../sketch/model";
import type { QuadMesh } from "../freeform/quadMesh";

export type EntityKind =
  | "mesh"
  | "points"
  | "primitive"
  | "refline"
  | "refpoint"
  | "sketch"
  | "sketch3d"
  | "body";

export interface BaseEntity {
  id: string;
  kind: EntityKind;
  name: string;
  visible: boolean;
  /** CSS colour string; optional override of the default per-kind colour */
  color?: string;
  /** monotonically increasing per-entity revision (bumped by Document.update) */
  rev: number;
}

export interface MeshEntity extends BaseEntity {
  kind: "mesh";
  mesh: MeshData;
  /** where it came from (file name) */
  source?: string;
}

export interface PointsEntity extends BaseEntity {
  kind: "points";
  points: PointCloudData;
  source?: string;
}

export interface PrimitiveEntity extends BaseEntity {
  kind: "primitive";
  def: PrimitiveDef;
  stats?: FitStats;
  /** mesh the primitive was extracted from */
  meshId?: string;
  /** display extent for planes: half-size of the drawn square (mm) */
  displaySize?: number;
  /** true for the symmetry-plane analyser result */
  symmetry?: boolean;
}

export interface RefLineEntity extends BaseEntity {
  kind: "refline";
  def: LineDef;
  /** drawn segment as parameters along the line */
  tMin: number;
  tMax: number;
}

export interface RefPointEntity extends BaseEntity {
  kind: "refpoint";
  def: PointDef;
}

/** A section polyline expressed in the sketch's 2D coordinates. */
export interface SectionPolyline {
  points: Vec2[];
  closed: boolean;
}

export interface SketchEntity extends BaseEntity {
  kind: "sketch";
  frame: Frame;
  sketch: Sketch;
  /** reference points from mesh sections (not part of the profile) */
  sections: SectionPolyline[];
  /** mesh the section was cut from */
  meshId?: string;
  /** offset of the section plane along frame.zAxis relative to the plane it was defined from */
  planeSource?: { type: "world"; axis: "XY" | "YZ" | "ZX"; offset: number } | { type: "entity"; id: string; offset: number };
}

export interface Curve3D {
  points: Vec3[];
  closed: boolean;
  bspline?: BSplineCurveDef;
}

export interface Sketch3DEntity extends BaseEntity {
  kind: "sketch3d";
  curves: Curve3D[];
  meshId?: string;
}

// ---------------------------------------------------------------------------
// CAD bodies (feature definitions evaluated by the OCCT kernel)
// ---------------------------------------------------------------------------

export type FeatureDef =
  | {
      type: "extrude";
      sketchId: string;
      distance: number;
      distance2: number;
      solid: boolean;
      draftDeg?: number;
    }
  | {
      type: "revolve";
      sketchId: string;
      /** axis in world coordinates */
      axisOrigin: Vec3;
      axisDir: Vec3;
      /** where the axis came from, for UI */
      axisRef?: string;
      angleDeg: number;
      solid: boolean;
    }
  | { type: "fitSurface"; surface: BSplineSurfaceDef; meshId?: string; stats?: FitStats }
  | { type: "freeform"; cage: QuadMesh; meshId?: string; solid: boolean; stats?: FitStats }
  | { type: "primitive"; def: PrimitiveDef; size: number }
  | { type: "loft"; sketchIds: string[]; solid: boolean }
  | { type: "boolean"; op: "union" | "cut" | "intersect"; targetId: string; toolId: string }
  | { type: "fillet"; baseId: string; edges: number[]; radius: number }
  | { type: "chamfer"; baseId: string; edges: number[]; distance: number }
  | { type: "mirror"; baseId: string; planeOrigin: Vec3; planeNormal: Vec3 };

export interface BodyEntity extends BaseEntity {
  kind: "body";
  feature: FeatureDef;
  /** bodies consumed by this one (hidden in the tree under it) */
  consumes?: string[];
}

export type Entity =
  | MeshEntity
  | PointsEntity
  | PrimitiveEntity
  | RefLineEntity
  | RefPointEntity
  | SketchEntity
  | Sketch3DEntity
  | BodyEntity;

export type EntityOf<K extends EntityKind> = Extract<Entity, { kind: K }>;

export const KIND_LABEL: Record<EntityKind, string> = {
  mesh: "Mesh",
  points: "Point cloud",
  primitive: "Primitive",
  refline: "Line",
  refpoint: "Point",
  sketch: "2D sketch",
  sketch3d: "3D sketch",
  body: "Body",
};
