import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type { Loop2D, MeshData, Vec2, Vec3 } from "../core/types";
import { WORLD_FRAME } from "../core/vec";
import { OcctKernel } from "../cad/OcctKernel";
import type { ShapeId, Tessellation } from "../cad/kernelTypes";
import { weldExactPositions } from "../mesh/util/weld-hash";
import { convertMesh, regionsToDemote, replanMesh } from "./index";
import { TEST_SHAPES } from "./testShapes";

const rect = (x0: number, y0: number, x1: number, y1: number): Loop2D => {
  const p: Vec2[] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
  return p.map((a, i) => ({ type: "line" as const, a, b: p[(i + 1) % 4] }));
};
const circle = (cx: number, cy: number, r: number): Loop2D => [{ type: "circle", center: [cx, cy], radius: r }];

let k: OcctKernel;
beforeAll(async () => {
  k = await OcctKernel.create();
});
afterAll(() => k.releaseAll());

/** Tessellate like a CAD STL export and weld coincident vertices (as STL import does). */
function stlOf(id: ShapeId, lin = 0.05, ang = 15): MeshData {
  const t = k.tessellate(id, { linearDeflection: lin, angularDeflectionDeg: ang });
  const { positions, remap } = weldExactPositions(t.positions);
  const indices = new Uint32Array(t.indices.length);
  for (let i = 0; i < indices.length; i++) indices[i] = remap[t.indices[i]];
  return { positions, indices };
}

function edgesWhere(t: Tessellation, pred: (pts: Vec3[]) => boolean): number[] {
  return t.edges
    .filter((e) => {
      const pts: Vec3[] = [];
      for (let i = 0; i < e.points.length; i += 3) pts.push([e.points[i], e.points[i + 1], e.points[i + 2]]);
      return pts.length > 1 && pred(pts);
    })
    .map((e) => e.id);
}

function roundTrip(id: ShapeId, lin = 0.05, ang = 15) {
  const mesh = stlOf(id, lin, ang);
  const t0 = performance.now();
  let res = convertMesh(mesh);
  const t1 = performance.now();
  let built = k.brepFromPlan(res.plan);
  for (let pass = 0; pass < 3; pass++) {
    const demote = regionsToDemote(res.plan, built);
    if (!demote.length) break;
    res = replanMesh(mesh, res, demote);
    built = k.brepFromPlan(res.plan);
  }
  const t2 = performance.now();
  const info = k.info(built.id);
  const ref = k.info(id);
  // eslint-disable-next-line no-console
  console.log(
    `${mesh.indices.length / 3} tris → regions ${JSON.stringify(res.stats.kinds)}, faceted ${res.stats.facetedTriangles}, faces ${info.faces} ` +
      `(failed ${built.failedFaces.length}, free edges ${built.freeEdges}), valid ${info.valid}, closed ${info.closed}, ` +
      `vol ${info.volume?.toFixed(3)} vs ${ref.volume?.toFixed(3)}; segment ${(t1 - t0).toFixed(0)} ms, build ${(t2 - t1).toFixed(0)} ms`,
  );
  return { mesh, res, built, info, ref };
}

describe("convertMesh — CAD STL round trips", () => {
  it("plate with a through hole: planes + one cylinder", () => {
    const id = k.extrude({ frame: WORLD_FRAME, profile: { loops: [circle(20, 15, 5), rect(0, 0, 40, 30)] }, distance: 10, solid: true });
    const { res, info, ref, built } = roundTrip(id);
    expect(res.stats.kinds.plane).toBe(6);
    expect(res.stats.kinds.cylinder).toBe(1);
    expect(res.stats.facetedTriangles).toBe(0);
    expect(built.failedFaces).toEqual([]);
    expect(info.type).toBe("solid");
    expect(info.valid).toBe(true);
    expect(info.faces).toBe(7);
    expect(info.volume!).toBeCloseTo(ref.volume!, 0);
    expect(built.faceKinds.filter((x) => x === "cylinder").length).toBe(1);
  });

  it("box with filleted vertical edges: planes + four fillet cylinders", () => {
    const box = k.extrude({ frame: WORLD_FRAME, profile: { loops: [rect(0, 0, 40, 30)] }, distance: 15, solid: true });
    const t = k.tessellate(box, { linearDeflection: 0.1, angularDeflectionDeg: 20 });
    const vertical = edgesWhere(t, (p) => Math.abs(p[0][0] - p[p.length - 1][0]) < 1e-6 && Math.abs(p[0][1] - p[p.length - 1][1]) < 1e-6);
    expect(vertical.length).toBe(4);
    const id = k.fillet(box, vertical, 4);
    const { res, info, ref } = roundTrip(id);
    expect(res.stats.kinds.cylinder).toBe(4);
    expect(res.stats.kinds.plane).toBe(6);
    expect(info.valid).toBe(true);
    expect(info.type).toBe("solid");
    expect(info.volume!).toBeCloseTo(ref.volume!, 0);
  });

  it("boss with a base fillet: torus", () => {
    const plate = k.extrude({ frame: WORLD_FRAME, profile: { loops: [rect(-30, -30, 30, 30)] }, distance: 8, solid: true });
    const boss = k.extrude({ frame: { ...WORLD_FRAME, origin: [0, 0, 8] }, profile: { loops: [circle(0, 0, 10)] }, distance: 12, solid: true });
    const u = k.boolean("union", plate, boss, { simplify: true });
    const t = k.tessellate(u, { linearDeflection: 0.1, angularDeflectionDeg: 20 });
    const base = edgesWhere(t, (p) => p.every((q) => Math.abs(q[2] - 8) < 1e-6 && Math.abs(Math.hypot(q[0], q[1]) - 10) < 1e-4));
    expect(base.length).toBeGreaterThan(0);
    const id = k.fillet(u, base, 3);
    const { res, info, ref } = roundTrip(id);
    expect(res.stats.kinds.torus).toBe(1);
    expect(info.valid).toBe(true);
    expect(info.type).toBe("solid");
    expect(info.volume!).toBeCloseTo(ref.volume!, 0);
  });

  it("box with every edge filleted: cylinders and spherical corners", () => {
    const box = k.extrude({ frame: WORLD_FRAME, profile: { loops: [rect(0, 0, 40, 30)] }, distance: 20, solid: true });
    const t = k.tessellate(box, { linearDeflection: 0.1, angularDeflectionDeg: 20 });
    const id = k.fillet(box, t.edges.map((e) => e.id), 3);
    const { res, info, ref } = roundTrip(id);
    expect(res.stats.kinds.plane).toBe(6);
    expect(res.stats.kinds.cylinder).toBe(12);
    expect(res.stats.kinds.sphere).toBe(8);
    expect(info.valid).toBe(true);
    expect(info.type).toBe("solid");
    // a few slivers where three fillets meet stay faceted
    expect(Math.abs(info.volume! - ref.volume!) / ref.volume!).toBeLessThan(1e-3);
  });

  it("turned part: cylinders, cones, a groove torus and a spherical end", () => {
    // half profile in the XZ plane (x = radius, y = height along Z)
    const XZ = { origin: [0, 0, 0] as Vec3, xAxis: [1, 0, 0] as Vec3, yAxis: [0, 0, 1] as Vec3, zAxis: [0, -1, 0] as Vec3 };
    const pts: Vec2[] = [[0, 0], [12, 0], [12, 10], [8, 14], [8, 24]];
    const loop: Loop2D = [];
    for (let i = 0; i + 1 < pts.length; i++) loop.push({ type: "line", a: pts[i], b: pts[i + 1] });
    // groove: concave arc from (8,24) to (8,32) bulging inwards to r = 6
    loop.push({ type: "arc", start: [8, 24], mid: [6, 28], end: [8, 32] });
    loop.push({ type: "line", a: [8, 32], b: [8, 40] });
    // spherical end cap of radius 8 centred on the axis at z = 40
    loop.push({ type: "arc", start: [8, 40], mid: [8 * Math.SQRT1_2, 40 + 8 * Math.SQRT1_2], end: [0, 48] });
    loop.push({ type: "line", a: [0, 48], b: [0, 0] });
    const id = k.revolve({ frame: XZ, profile: { loops: [loop] }, axisOrigin: [0, 0, 0], axisDir: [0, 0, 1], angleDeg: 360, solid: true });
    const { res, info, ref } = roundTrip(id);
    expect(res.stats.kinds.cone).toBe(1);
    expect(res.stats.kinds.torus).toBe(1);
    expect(res.stats.kinds.sphere).toBe(1);
    expect(res.stats.facetedTriangles).toBe(0);
    expect(info.valid).toBe(true);
    expect(info.type).toBe("solid");
    expect(info.volume!).toBeCloseTo(ref.volume!, 0);
  });

  it("countersunk hole: cone between the hole and the top face", () => {
    const plate = k.extrude({ frame: WORLD_FRAME, profile: { loops: [rect(-20, -20, 20, 20)] }, distance: 10, solid: true });
    const XZ = { origin: [0, 0, 0] as Vec3, xAxis: [1, 0, 0] as Vec3, yAxis: [0, 0, 1] as Vec3, zAxis: [0, -1, 0] as Vec3 };
    const tool: Loop2D = [
      { type: "line", a: [0, -1], b: [4, -1] },
      { type: "line", a: [4, -1], b: [4, 6] },
      { type: "line", a: [4, 6], b: [9, 11] },
      { type: "line", a: [9, 11], b: [0, 11] },
      { type: "line", a: [0, 11], b: [0, -1] },
    ];
    const cutter = k.revolve({ frame: XZ, profile: { loops: [tool] }, axisOrigin: [0, 0, 0], axisDir: [0, 0, 1], angleDeg: 360, solid: true });
    const id = k.boolean("cut", plate, cutter);
    const { res, info, ref } = roundTrip(id);
    expect(res.stats.kinds.cone).toBe(1);
    expect(res.stats.kinds.cylinder).toBe(1);
    expect(info.valid).toBe(true);
    expect(info.volume!).toBeCloseTo(ref.volume!, 0);
  });

  it("toothed ring with holes and filleted teeth, coarse and fine", () => {
    const id = TEST_SHAPES.gear(k);
    for (const [lin, ang] of [[0.2, 30], [0.005, 5]]) {
      const { res, info, ref } = roundTrip(id, lin, ang);
      expect(res.stats.kinds.plane).toBe(98);
      expect(res.stats.kinds.cylinder).toBe(55);
      expect(info.valid).toBe(true);
      expect(info.type).toBe("solid");
      expect(Math.abs(info.volume! - ref.volume!) / ref.volume!).toBeLessThan(1e-3);
    }
  }, 60_000);

  it("built-in CAD sample (mounting plate STL) converts to a clean solid", async () => {
    const { buildCadSampleStl } = await import("../samples/cadSample");
    const { parseSTL } = await import("../mesh/io/stl");
    const mesh = parseSTL(await buildCadSampleStl(k as never));
    let res = convertMesh(mesh);
    let built = k.brepFromPlan(res.plan);
    for (let pass = 0; pass < 3; pass++) {
      const demote = regionsToDemote(res.plan, built);
      if (!demote.length) break;
      res = replanMesh(mesh, res, demote);
      built = k.brepFromPlan(res.plan);
    }
    const info = k.info(built.id);
    expect(res.stats.kinds).toEqual({ plane: 7, cylinder: 10, cone: 5, torus: 1 });
    expect(res.stats.facetedTriangles).toBe(0);
    expect(info.valid).toBe(true);
    expect(info.type).toBe("solid");
    expect(info.faces).toBe(23);
  }, 60_000);
});
