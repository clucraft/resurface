/**
 * "Mounting plate (CAD STL)": a real CAD part built with the kernel and exported as a
 * tessellated STL — the kind of file Auto Convert (STL → STEP) is made for. Planes, cylinders,
 * countersink cones and a fillet torus.
 */
import type { Frame, Loop2D, Vec2, Vec3 } from "../core/types";
import type { KernelClient } from "../cad/KernelClient";
import type { ShapeId } from "../cad/kernelTypes";

const WORLD: Frame = { origin: [0, 0, 0], xAxis: [1, 0, 0], yAxis: [0, 1, 0], zAxis: [0, 0, 1] };
const rect = (x0: number, y0: number, x1: number, y1: number): Loop2D => {
  const p: Vec2[] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
  return p.map((a, i) => ({ type: "line" as const, a, b: p[(i + 1) % 4] }));
};
const circle = (cx: number, cy: number, r: number): Loop2D => [{ type: "circle", center: [cx, cy], radius: r }];

async function edgesWhere(k: KernelClient, id: ShapeId, pred: (pts: Vec3[]) => boolean): Promise<number[]> {
  const t = await k.tessellate(id, { linearDeflection: 0.1, angularDeflectionDeg: 20 });
  return t.edges
    .filter((e) => {
      const pts: Vec3[] = [];
      for (let i = 0; i < e.points.length; i += 3) pts.push([e.points[i], e.points[i + 1], e.points[i + 2]]);
      return pts.length > 1 && pred(pts);
    })
    .map((e) => e.id);
}

/** Build the part and return it as binary STL bytes. */
export async function buildCadSampleStl(k: KernelClient): Promise<Uint8Array> {
  const made: ShapeId[] = [];
  const keep = async (p: Promise<ShapeId>) => {
    const id = await p;
    made.push(id);
    return id;
  };
  try {
    let part = await keep(k.extrude({ frame: WORLD, profile: { loops: [rect(-40, -25, 40, 25)] }, distance: 8, solid: true }));
    // rounded corners
    const corners = await edgesWhere(k, part, (p) => Math.abs(p[0][0] - p[p.length - 1][0]) < 1e-6 && Math.abs(p[0][1] - p[p.length - 1][1]) < 1e-6);
    part = await keep(k.fillet(part, corners, 6));
    // boss with a fillet at its foot
    const boss = await keep(k.extrude({ frame: { ...WORLD, origin: [0, 0, 8] }, profile: { loops: [circle(0, 0, 12)] }, distance: 14, solid: true }));
    part = await keep(k.boolean("union", part, boss, { simplify: true }));
    const foot = await edgesWhere(k, part, (p) => p.every((q) => Math.abs(q[2] - 8) < 1e-6 && Math.abs(Math.hypot(q[0], q[1]) - 12) < 1e-4));
    part = await keep(k.fillet(part, foot, 3));
    // four countersunk holes (Ø6, 90° countersink to Ø12)
    const holes: Loop2D[] = [];
    for (const [x, y] of [[-30, -15], [30, -15], [-30, 15], [30, 15]]) holes.push(circle(x, y, 3));
    const drills = await keep(k.extrude({ frame: { ...WORLD, origin: [0, 0, -1] }, profile: { loops: holes }, distance: 30, solid: true }));
    part = await keep(k.boolean("cut", part, drills));
    for (const [x, y] of [[-30, -15], [30, -15], [-30, 15], [30, 15]]) {
      const frame: Frame = { origin: [x, y, 0], xAxis: [1, 0, 0], yAxis: [0, 0, 1], zAxis: [0, -1, 0] };
      const tool: Loop2D = [
        { type: "line", a: [0, 4.5], b: [3, 4.5] },
        { type: "line", a: [3, 4.5], b: [7, 8.5] },
        { type: "line", a: [7, 8.5], b: [0, 8.5] },
        { type: "line", a: [0, 8.5], b: [0, 4.5] },
      ];
      const sink = await keep(k.revolve({ frame, profile: { loops: [tool] }, axisOrigin: [x, y, 0], axisDir: [0, 0, 1], angleDeg: 360, solid: true }));
      part = await keep(k.boolean("cut", part, sink));
    }
    // bore through the boss with a chamfer at the top
    const bore = await keep(k.extrude({ frame: { ...WORLD, origin: [0, 0, -1] }, profile: { loops: [circle(0, 0, 6)] }, distance: 30, solid: true }));
    part = await keep(k.boolean("cut", part, bore));
    const cf: Frame = { origin: [0, 0, 0], xAxis: [1, 0, 0], yAxis: [0, 0, 1], zAxis: [0, -1, 0] };
    const chamfer: Loop2D = [
      { type: "line", a: [0, 20], b: [6, 20] },
      { type: "line", a: [6, 20], b: [8.5, 22.5] },
      { type: "line", a: [8.5, 22.5], b: [0, 22.5] },
      { type: "line", a: [0, 22.5], b: [0, 20] },
    ];
    const cham = await keep(k.revolve({ frame: cf, profile: { loops: [chamfer] }, axisOrigin: [0, 0, 0], axisDir: [0, 0, 1], angleDeg: 360, solid: true }));
    part = await keep(k.boolean("cut", part, cham));
    return await k.exportSTL([{ id: part, name: "Mounting plate" }], { linearDeflection: 0.03, angularDeflectionDeg: 12 });
  } finally {
    for (const id of made) void k.release(id);
  }
}
