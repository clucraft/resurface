/** CAD test parts for the converter tests (built with the OCCT kernel, tessellated like an STL export). */
import type { Loop2D, MeshData, Vec2, Vec3 } from "../core/types";
import { WORLD_FRAME } from "../core/vec";
import type { OcctKernel } from "../cad/OcctKernel";
import type { ShapeId } from "../cad/kernelTypes";
import { weldExactPositions } from "../mesh/util/weld-hash";

export const rect = (x0: number, y0: number, x1: number, y1: number): Loop2D => {
  const p: Vec2[] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
  return p.map((a, i) => ({ type: "line" as const, a, b: p[(i + 1) % 4] }));
};
export const circle = (cx: number, cy: number, r: number): Loop2D => [{ type: "circle", center: [cx, cy], radius: r }];
const XZ = { origin: [0, 0, 0] as Vec3, xAxis: [1, 0, 0] as Vec3, yAxis: [0, 0, 1] as Vec3, zAxis: [0, -1, 0] as Vec3 };

function edgeIds(k: OcctKernel, id: ShapeId, pred: (pts: Vec3[]) => boolean): number[] {
  const t = k.tessellate(id, { linearDeflection: 0.1, angularDeflectionDeg: 20 });
  return t.edges
    .filter((e) => {
      const pts: Vec3[] = [];
      for (let i = 0; i < e.points.length; i += 3) pts.push([e.points[i], e.points[i + 1], e.points[i + 2]]);
      return pts.length > 1 && pred(pts);
    })
    .map((e) => e.id);
}

export const TEST_SHAPES: Record<string, (k: OcctKernel) => ShapeId> = {
  /** toothed ring with holes and filleted teeth (a larger, busier part) */
  gear: (k) => {
    const teeth = 24, ro = 30, ri = 26;
    const pts: Vec2[] = [];
    for (let i = 0; i < teeth; i++) {
      const a0 = (i / teeth) * 2 * Math.PI, w = (2 * Math.PI) / teeth;
      for (const [f, r] of [[0, ri], [0.15, ro], [0.5, ro], [0.65, ri]] as const) pts.push([r * Math.cos(a0 + f * w), r * Math.sin(a0 + f * w)]);
    }
    const outer: Loop2D = pts.map((a, i) => ({ type: "line" as const, a, b: pts[(i + 1) % pts.length] }));
    const loops: Loop2D[] = [outer, circle(0, 0, 10)];
    for (let i = 0; i < 6; i++) loops.push(circle(18 * Math.cos((i * Math.PI) / 3), 18 * Math.sin((i * Math.PI) / 3), 3));
    const g = k.extrude({ frame: WORLD_FRAME, profile: { loops }, distance: 8, solid: true });
    const vertical = edgeIds(k, g, (p) => Math.abs(p[0][0] - p[p.length - 1][0]) < 1e-6 && Math.abs(p[0][1] - p[p.length - 1][1]) < 1e-6 && Math.hypot(p[0][0], p[0][1]) > 27);
    return k.fillet(g, vertical, 0.8);
  },
  plate: (k) => k.extrude({ frame: WORLD_FRAME, profile: { loops: [circle(20, 15, 5), rect(0, 0, 40, 30)] }, distance: 10, solid: true }),
  filletBox: (k) => {
    const box = k.extrude({ frame: WORLD_FRAME, profile: { loops: [rect(0, 0, 40, 30)] }, distance: 15, solid: true });
    return k.fillet(box, edgeIds(k, box, (p) => Math.abs(p[0][0] - p[p.length - 1][0]) < 1e-6 && Math.abs(p[0][1] - p[p.length - 1][1]) < 1e-6), 4);
  },
  boss: (k) => {
    const plate = k.extrude({ frame: WORLD_FRAME, profile: { loops: [rect(-30, -30, 30, 30)] }, distance: 8, solid: true });
    const boss = k.extrude({ frame: { ...WORLD_FRAME, origin: [0, 0, 8] }, profile: { loops: [circle(0, 0, 10)] }, distance: 12, solid: true });
    const u = k.boolean("union", plate, boss, { simplify: true });
    return k.fillet(u, edgeIds(k, u, (p) => p.every((q) => Math.abs(q[2] - 8) < 1e-6 && Math.abs(Math.hypot(q[0], q[1]) - 10) < 1e-4)), 3);
  },
  roundBox: (k) => {
    const box = k.extrude({ frame: WORLD_FRAME, profile: { loops: [rect(0, 0, 40, 30)] }, distance: 20, solid: true });
    return k.fillet(box, edgeIds(k, box, () => true), 3);
  },
  turned: (k) => {
    const pts: Vec2[] = [[0, 0], [12, 0], [12, 10], [8, 14], [8, 24]];
    const loop: Loop2D = [];
    for (let i = 0; i + 1 < pts.length; i++) loop.push({ type: "line", a: pts[i], b: pts[i + 1] });
    loop.push({ type: "arc", start: [8, 24], mid: [6, 28], end: [8, 32] });
    loop.push({ type: "line", a: [8, 32], b: [8, 40] });
    loop.push({ type: "arc", start: [8, 40], mid: [8 * Math.SQRT1_2, 40 + 8 * Math.SQRT1_2], end: [0, 48] });
    loop.push({ type: "line", a: [0, 48], b: [0, 0] });
    return k.revolve({ frame: XZ, profile: { loops: [loop] }, axisOrigin: [0, 0, 0], axisDir: [0, 0, 1], angleDeg: 360, solid: true });
  },
  countersink: (k) => {
    const plate = k.extrude({ frame: WORLD_FRAME, profile: { loops: [rect(-20, -20, 20, 20)] }, distance: 10, solid: true });
    const tool: Loop2D = [
      { type: "line", a: [0, -1], b: [4, -1] },
      { type: "line", a: [4, -1], b: [4, 6] },
      { type: "line", a: [4, 6], b: [9, 11] },
      { type: "line", a: [9, 11], b: [0, 11] },
      { type: "line", a: [0, 11], b: [0, -1] },
    ];
    const cutter = k.revolve({ frame: XZ, profile: { loops: [tool] }, axisOrigin: [0, 0, 0], axisDir: [0, 0, 1], angleDeg: 360, solid: true });
    return k.boolean("cut", plate, cutter);
  },
};

/** Tessellate like a CAD STL export and weld coincident vertices (as STL import does). */
export function stlOf(k: OcctKernel, id: ShapeId, lin = 0.05, ang = 15): MeshData {
  const t = k.tessellate(id, { linearDeflection: lin, angularDeflectionDeg: ang });
  const { positions, remap } = weldExactPositions(t.positions);
  const indices = new Uint32Array(t.indices.length);
  for (let i = 0; i < indices.length; i++) indices[i] = remap[t.indices[i]];
  return { positions, indices };
}
