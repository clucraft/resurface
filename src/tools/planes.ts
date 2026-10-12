/**
 * Plane sources used by section, clip and other tools: world planes, reference planes,
 * planes perpendicular to an axis, and planes containing an axis (for revolve profiles).
 */
import type { App } from "../app/App";
import type { Frame, Vec3 } from "../core/types";
import type { PrimitiveEntity, RefLineEntity } from "../core/entities";
import { anyPerpendicular, cross, dot, frameFromNormal, normalize, sub } from "../core/vec";
import { meshBounds } from "../mesh/topology";

export interface PlaneOption {
  value: string;
  label: string;
}

export function planeSourceOptions(app: App, opts: { axial?: boolean } = {}): PlaneOption[] {
  const out: PlaneOption[] = [
    { value: "world:XY", label: "World XY (top)" },
    { value: "world:ZX", label: "World ZX (front)" },
    { value: "world:YZ", label: "World YZ (right)" },
  ];
  for (const e of app.doc.all()) {
    if (e.kind === "primitive" && e.def.kind === "plane") out.push({ value: `entity:${e.id}`, label: e.name });
    if (e.kind === "primitive" && (e.def.kind === "cylinder" || e.def.kind === "cone")) {
      out.push({ value: `perp:${e.id}`, label: `${e.name} — perpendicular to axis` });
      if (opts.axial !== false) out.push({ value: `axial:${e.id}`, label: `${e.name} — through axis` });
    }
    if (e.kind === "refline") {
      out.push({ value: `perp:${e.id}`, label: `${e.name} — perpendicular` });
      if (opts.axial !== false) out.push({ value: `axial:${e.id}`, label: `${e.name} — through line` });
    }
  }
  return out;
}

function axisOf(app: App, id: string): { origin: Vec3; dir: Vec3 } | null {
  const e = app.doc.get(id);
  if (!e) return null;
  if (e.kind === "refline") return { origin: (e as RefLineEntity).def.origin, dir: normalize((e as RefLineEntity).def.direction) };
  if (e.kind === "primitive") {
    const d = (e as PrimitiveEntity).def;
    if (d.kind === "cylinder") return { origin: d.origin, dir: normalize(d.axis) };
    if (d.kind === "cone") return { origin: d.apex, dir: normalize(d.axis) };
  }
  return null;
}

/** Base frame of a plane source (offset 0). zAxis is the plane normal. */
export function resolvePlaneSource(app: App, key: string, rotationDeg = 0): Frame | null {
  const [type, ref] = key.split(":");
  if (type === "world") {
    if (ref === "XY") return { origin: [0, 0, 0], xAxis: [1, 0, 0], yAxis: [0, 1, 0], zAxis: [0, 0, 1] };
    if (ref === "ZX") return { origin: [0, 0, 0], xAxis: [1, 0, 0], yAxis: [0, 0, 1], zAxis: [0, -1, 0] };
    if (ref === "YZ") return { origin: [0, 0, 0], xAxis: [0, 1, 0], yAxis: [0, 0, 1], zAxis: [1, 0, 0] };
  }
  if (type === "entity") {
    const e = app.doc.get<PrimitiveEntity>(ref);
    if (e?.kind === "primitive" && e.def.kind === "plane") return frameFromNormal(e.def.origin, e.def.normal);
  }
  if (type === "perp") {
    const ax = axisOf(app, ref);
    if (ax) return frameFromNormal(ax.origin, ax.dir);
  }
  if (type === "axial") {
    const ax = axisOf(app, ref);
    if (ax) {
      // sketch Y axis = rotation axis, so revolve profiles are drawn with the axis vertical
      const y = ax.dir;
      let x = cross(y, [0, 0, 1]);
      if (Math.hypot(...x) < 1e-6) x = anyPerpendicular(y);
      x = normalize(x);
      if (rotationDeg) {
        const a = (rotationDeg * Math.PI) / 180;
        const z0 = cross(x, y);
        x = normalize([x[0] * Math.cos(a) + z0[0] * Math.sin(a), x[1] * Math.cos(a) + z0[1] * Math.sin(a), x[2] * Math.cos(a) + z0[2] * Math.sin(a)]);
      }
      const z = normalize(cross(x, y));
      return { origin: ax.origin, xAxis: x, yAxis: y, zAxis: z };
    }
  }
  return null;
}

/** Shift a frame along its normal. */
export function offsetFrame(f: Frame, offset: number): Frame {
  return {
    ...f,
    origin: [f.origin[0] + f.zAxis[0] * offset, f.origin[1] + f.zAxis[1] * offset, f.origin[2] + f.zAxis[2] * offset],
  };
}

/** Range of offsets along the frame normal that intersect the active mesh (with a margin). */
export function meshOffsetRange(app: App, f: Frame): { min: number; max: number; mid: number } {
  const m = app.activeMesh();
  if (!m) return { min: -100, max: 100, mid: 0 };
  const b = meshBounds(m.mesh);
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < 8; i++) {
    const p: Vec3 = [i & 1 ? b.max[0] : b.min[0], i & 2 ? b.max[1] : b.min[1], i & 4 ? b.max[2] : b.min[2]];
    const d = dot(sub(p, f.origin), f.zAxis);
    lo = Math.min(lo, d);
    hi = Math.max(hi, d);
  }
  return { min: lo, max: hi, mid: (lo + hi) / 2 };
}
