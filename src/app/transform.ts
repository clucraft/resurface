/** Rigid transforms of document entities (used by alignment tools). */
import type { Entity, FeatureDef } from "../core/entities";
import type { BSplineSurfaceDef, Frame, Mat4, MeshData, PrimitiveDef, Vec3 } from "../core/types";
import { normalize, transformDir, transformPoint } from "../core/vec";
import { toShared } from "./shared";

const tp = transformPoint;
const td = (m: Mat4, d: Vec3) => normalize(transformDir(m, d));

export function transformFrame(m: Mat4, f: Frame): Frame {
  return { origin: tp(m, f.origin), xAxis: td(m, f.xAxis), yAxis: td(m, f.yAxis), zAxis: td(m, f.zAxis) };
}

export function transformPrimitive(m: Mat4, d: PrimitiveDef): PrimitiveDef {
  switch (d.kind) {
    case "plane":
      return { ...d, origin: tp(m, d.origin), normal: td(m, d.normal) };
    case "sphere":
      return { ...d, center: tp(m, d.center) };
    case "cylinder":
      return { ...d, origin: tp(m, d.origin), axis: td(m, d.axis) };
    case "cone":
      return { ...d, apex: tp(m, d.apex), axis: td(m, d.axis) };
  }
}

export function transformMesh(m: Mat4, mesh: MeshData): MeshData {
  const pos = new Float32Array(mesh.positions.length);
  const src = mesh.positions;
  for (let i = 0; i < src.length; i += 3) {
    const x = src[i], y = src[i + 1], z = src[i + 2];
    pos[i] = m[0] * x + m[4] * y + m[8] * z + m[12];
    pos[i + 1] = m[1] * x + m[5] * y + m[9] * z + m[13];
    pos[i + 2] = m[2] * x + m[6] * y + m[10] * z + m[14];
  }
  let normals: Float32Array | undefined;
  if (mesh.normals) {
    normals = new Float32Array(mesh.normals.length);
    const n = mesh.normals;
    for (let i = 0; i < n.length; i += 3) {
      normals[i] = m[0] * n[i] + m[4] * n[i + 1] + m[8] * n[i + 2];
      normals[i + 1] = m[1] * n[i] + m[5] * n[i + 1] + m[9] * n[i + 2];
      normals[i + 2] = m[2] * n[i] + m[6] * n[i + 1] + m[10] * n[i + 2];
    }
  }
  return { positions: toShared(pos), indices: mesh.indices, normals: normals ? toShared(normals) : undefined, colors: mesh.colors };
}

function transformSurface(m: Mat4, s: BSplineSurfaceDef): BSplineSurfaceDef {
  return { ...s, controlPoints: s.controlPoints.map((p) => tp(m, p)) };
}

function transformFeature(m: Mat4, f: FeatureDef): FeatureDef {
  switch (f.type) {
    case "revolve":
      return { ...f, axisOrigin: tp(m, f.axisOrigin), axisDir: td(m, f.axisDir) };
    case "fitSurface":
      return { ...f, surface: transformSurface(m, f.surface) };
    case "freeform":
      return { ...f, cage: { ...f.cage, vertices: f.cage.vertices.map((v) => tp(m, v)) } };
    case "primitive":
      return { ...f, def: transformPrimitive(m, f.def) };
    case "mirror":
      return { ...f, planeOrigin: tp(m, f.planeOrigin), planeNormal: td(m, f.planeNormal) };
    default:
      // extrude/loft follow their sketches; booleans/fillets follow their inputs
      return f;
  }
}

/** Patch that moves an entity by a rigid transform (meshes return a new MeshData). */
export function transformEntityPatch(m: Mat4, e: Entity): Partial<Entity> | null {
  switch (e.kind) {
    case "mesh":
      return { mesh: transformMesh(m, e.mesh) } as Partial<Entity>;
    case "points": {
      const tmp = transformMesh(m, { positions: e.points.positions, indices: new Uint32Array(0), normals: e.points.normals });
      return { points: { ...e.points, positions: tmp.positions, normals: tmp.normals } } as Partial<Entity>;
    }
    case "primitive":
      return { def: transformPrimitive(m, e.def) } as Partial<Entity>;
    case "refline":
      return { def: { ...e.def, origin: tp(m, e.def.origin), direction: td(m, e.def.direction) } } as Partial<Entity>;
    case "refpoint":
      return { def: { ...e.def, position: tp(m, e.def.position) } } as Partial<Entity>;
    case "sketch":
      return { frame: transformFrame(m, e.frame) } as Partial<Entity>;
    case "sketch3d":
      return {
        curves: e.curves.map((c) => ({
          ...c,
          points: c.points.map((p) => tp(m, p)),
          bspline: c.bspline ? { ...c.bspline, controlPoints: c.bspline.controlPoints.map((p) => tp(m, p)) } : undefined,
        })),
      } as Partial<Entity>;
    case "body":
      return { feature: transformFeature(m, e.feature) } as Partial<Entity>;
  }
}
