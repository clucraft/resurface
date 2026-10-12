import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type { BSplineSurfaceDef, Frame, Loop2D, Vec2, Vec3 } from "../core/types";
import { WORLD_FRAME } from "../core/vec";
import { OcctKernel } from "./OcctKernel";
import type { Tessellation } from "./kernelTypes";

const rect = (x0: number, y0: number, x1: number, y1: number): Loop2D => {
  const p: Vec2[] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
  return p.map((a, i) => ({ type: "line" as const, a, b: p[(i + 1) % 4] }));
};
const circle = (cx: number, cy: number, r: number): Loop2D => [{ type: "circle", center: [cx, cy], radius: r }];
const XZ_FRAME: Frame = { origin: [0, 0, 0], xAxis: [1, 0, 0], yAxis: [0, 0, 1], zAxis: [0, -1, 0] };

/** Bicubic planar patch spanning c0 + s*du + t*dv. */
function planarPatch(c0: Vec3, du: Vec3, dv: Vec3): BSplineSurfaceDef {
  const cps: Vec3[] = [];
  for (let i = 0; i < 4; i++)
    for (let j = 0; j < 4; j++) cps.push([0, 1, 2].map((k) => c0[k] + (du[k] * i) / 3 + (dv[k] * j) / 3) as Vec3);
  const knots = [0, 0, 0, 0, 1, 1, 1, 1];
  return { degreeU: 3, degreeV: 3, knotsU: knots, knotsV: knots, nu: 4, nv: 4, controlPoints: cps };
}

function cubePatches(L: number): BSplineSurfaceDef[] {
  return [
    planarPatch([0, 0, 0], [0, L, 0], [L, 0, 0]),
    planarPatch([0, 0, L], [L, 0, 0], [0, L, 0]),
    planarPatch([0, 0, 0], [L, 0, 0], [0, 0, L]),
    planarPatch([0, L, 0], [0, 0, L], [L, 0, 0]),
    planarPatch([0, 0, 0], [0, 0, L], [0, L, 0]),
    planarPatch([L, 0, 0], [0, L, 0], [0, 0, L]),
  ];
}

/** Signed volume enclosed by the triangles (positive when windings are outward). */
function meshVolume(t: Tessellation): number {
  const v = t.positions, idx = t.indices;
  let vol = 0;
  for (let i = 0; i < idx.length; i += 3) {
    const a = 3 * idx[i], b = 3 * idx[i + 1], c = 3 * idx[i + 2];
    vol += (v[a] * (v[b + 1] * v[c + 2] - v[b + 2] * v[c + 1]) - v[a + 1] * (v[b] * v[c + 2] - v[b + 2] * v[c]) + v[a + 2] * (v[b] * v[c + 1] - v[b + 1] * v[c])) / 6;
  }
  return vol;
}

/** Edge ids whose polyline is a straight segment from p to q (either direction). */
function findEdge(t: Tessellation, p: Vec3, q: Vec3, tol = 1e-4): number {
  const near = (pts: Float32Array, k: number, r: Vec3) => Math.hypot(pts[3 * k] - r[0], pts[3 * k + 1] - r[1], pts[3 * k + 2] - r[2]) < tol;
  const e = t.edges.find(({ points }) => {
    const n = points.length / 3;
    return n >= 2 && ((near(points, 0, p) && near(points, n - 1, q)) || (near(points, 0, q) && near(points, n - 1, p)));
  });
  if (!e) throw new Error("edge not found");
  return e.id;
}

let k: OcctKernel;
const t0 = performance.now();
let initMs = 0;

beforeAll(async () => {
  k = await OcctKernel.create();
  initMs = performance.now() - t0;
});

afterAll(() => {
  k.releaseAll();
  // eslint-disable-next-line no-console
  console.log(`OCCT init: ${initMs.toFixed(0)} ms`);
});

describe("OcctKernel — extrude / revolve", () => {
  it("extrudes a rectangle with a circular hole into a closed valid solid", () => {
    const id = k.extrude({ frame: WORLD_FRAME, profile: { loops: [circle(20, 10, 5), rect(0, 0, 40, 20)] }, distance: 10, solid: true });
    const info = k.info(id);
    expect(info.type).toBe("solid");
    expect(info.valid).toBe(true);
    expect(info.closed).toBe(true);
    expect(info.faces).toBe(7);
    expect(info.volume!).toBeCloseTo((800 - 25 * Math.PI) * 10, 6);
    expect(info.bbox.min[2]).toBeCloseTo(0, 6);
    expect(info.bbox.max[2]).toBeCloseTo(10, 6);
  });

  it("handles several regions, holes and islands in holes", () => {
    const profile = {
      loops: [rect(0, 0, 100, 100), rect(20, 20, 80, 80), rect(40, 40, 60, 60), circle(150, 50, 10)],
    };
    const id = k.extrude({ frame: WORLD_FRAME, profile, distance: 5 });
    const info = k.info(id);
    expect(info.type).toBe("compound");
    expect(info.solids).toBe(3);
    expect(info.valid).toBe(true);
    expect(info.volume!).toBeCloseTo((100 * 100 - 60 * 60 + 20 * 20 + Math.PI * 100) * 5, 5);
  });

  it("supports two-sided extrusion, custom direction and negative distance", () => {
    const two = k.info(k.extrude({ frame: WORLD_FRAME, profile: { loops: [rect(0, 0, 10, 10)] }, distance: 4, distance2: 6 }));
    expect(two.volume!).toBeCloseTo(1000, 6);
    expect(two.bbox.min[2]).toBeCloseTo(-6, 6);
    expect(two.bbox.max[2]).toBeCloseTo(4, 6);
    const neg = k.info(k.extrude({ frame: WORLD_FRAME, profile: { loops: [rect(0, 0, 10, 10)] }, distance: -3 }));
    expect(neg.volume!).toBeCloseTo(300, 6);
    expect(neg.bbox.max[2]).toBeCloseTo(0, 6);
    const oblique = k.info(k.extrude({ frame: WORLD_FRAME, profile: { loops: [rect(0, 0, 10, 10)] }, distance: Math.SQRT2 * 5, direction: [1, 0, 1] }));
    expect(oblique.volume!).toBeCloseTo(500, 5); // base area x height
    expect(oblique.valid).toBe(true);
  });

  it("applies a draft angle to the side walls", () => {
    const id = k.extrude({ frame: WORLD_FRAME, profile: { loops: [rect(0, 0, 20, 10)] }, distance: 10, draftDeg: 5 });
    const info = k.info(id);
    expect(info.valid).toBe(true);
    // walls lean inwards: section at height z is (20 - 2 z t) x (10 - 2 z t), t = tan(5deg)
    const d = 10 * Math.tan((5 * Math.PI) / 180);
    const expected = 10 * (200 - 30 * d + (4 / 3) * d * d);
    expect(info.volume!).toBeCloseTo(expected, 6);
    expect(info.volume!).toBeLessThan(2000);
  });

  it("extrudes profiles and open wires as surfaces", () => {
    const shell = k.info(k.extrude({ frame: WORLD_FRAME, profile: { loops: [rect(0, 0, 10, 10)] }, distance: 5, solid: false }));
    expect(shell.type).toBe("shell");
    expect(shell.faces).toBe(4);
    expect(shell.closed).toBe(false);
    expect(shell.area).toBeCloseTo(200, 6);
    const open = k.info(
      k.extrudeWire({
        frame: WORLD_FRAME,
        wire: { closed: false, segments: [{ type: "line", a: [0, 0], b: [10, 0] }, { type: "arc", start: [10, 0], mid: [15, 5], end: [20, 0] }] },
        distance: 3,
      }),
    );
    expect(open.faces).toBe(2);
    expect(open.area).toBeCloseTo(3 * (10 + 5 * Math.PI), 4);
  });

  it("revolves a rectangle offset from the axis into a ring", () => {
    const profile = { loops: [rect(10, 0, 12, 4)] }; // x = radius, y = height (XZ plane)
    const full = k.info(k.revolve({ frame: XZ_FRAME, profile, axisOrigin: [0, 0, 0], axisDir: [0, 0, 1], angleDeg: 360, solid: true }));
    expect(full.type).toBe("solid");
    expect(full.valid).toBe(true);
    expect(full.closed).toBe(true);
    expect(full.volume!).toBeCloseTo(Math.PI * (144 - 100) * 4, 6);
    const half = k.info(k.revolve({ frame: XZ_FRAME, profile, axisOrigin: [0, 0, 0], axisDir: [0, 0, 1], angleDeg: 180 }));
    expect(half.volume!).toBeCloseTo((Math.PI * (144 - 100) * 4) / 2, 6);
    const surf = k.info(k.revolve({ frame: XZ_FRAME, wire: { closed: false, segments: [{ type: "line", a: [10, 0], b: [10, 5] }] }, axisOrigin: [0, 0, 0], axisDir: [0, 0, 1], angleDeg: 360 }));
    expect(surf.faces).toBe(1);
    expect(surf.area).toBeCloseTo(2 * Math.PI * 10 * 5, 6);
  });

  it("reports clear errors", () => {
    expect(() => k.extrude({ frame: WORLD_FRAME, profile: { loops: [] }, distance: 5 })).toThrow(/^Extrude failed: profile has no closed loop/);
    const open: Loop2D = [{ type: "line", a: [0, 0], b: [10, 0] }, { type: "line", a: [10, 0], b: [10, 10] }];
    expect(() => k.extrude({ frame: WORLD_FRAME, profile: { loops: [open] }, distance: 5 })).toThrow(/Extrude failed: profile loop 0 is not closed/);
    expect(() => k.info(123456)).toThrow(/unknown shape id 123456/);
    expect(() => k.extrude({ frame: WORLD_FRAME, profile: { loops: [rect(0, 0, 1, 1)] }, distance: 0 })).toThrow(/zero extrusion length/);
    expect(() =>
      k.revolve({ frame: XZ_FRAME, profile: { loops: [rect(-2, 0, 2, 4)] }, axisOrigin: [0, 0, 0], axisDir: [0, 0, 1], angleDeg: 360 }),
    ).toThrow(/Revolve failed/);
  });
});

describe("OcctKernel — NURBS faces and sewing", () => {
  it("builds a face from a 4x4 bicubic control net and tessellates it", () => {
    const cps: Vec3[] = [];
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) cps.push([i * 10, j * 10, (i === 1 || i === 2) && (j === 1 || j === 2) ? 8 : 0]);
    const knots = [0, 0, 0, 0, 1, 1, 1, 1];
    const surface: BSplineSurfaceDef = { degreeU: 3, degreeV: 3, knotsU: knots, knotsV: knots, nu: 4, nv: 4, controlPoints: cps };
    const id = k.bsplineSurfaceFace({ surface });
    const info = k.info(id);
    expect(info.type).toBe("face");
    expect(info.faces).toBe(1);
    expect(info.valid).toBe(true);
    const t = k.tessellate(id, { linearDeflection: 0.05, angularDeflectionDeg: 10 });
    expect(t.indices.length).toBeGreaterThan(30);
    expect(t.positions.length).toBeGreaterThan(30);
    expect(t.normals.length).toBe(t.positions.length);
    expect(Array.from(t.faceTriangleRanges)).toEqual([0, t.indices.length / 3]);
    expect(t.edges.length).toBe(4);
    // trimmed parameter range gives a smaller face
    const sub = k.info(k.bsplineSurfaceFace({ surface, uRange: [0, 0.5], vRange: [0, 1] }));
    expect(sub.area).toBeLessThan(info.area * 0.75);
    // weights are accepted
    const w = k.info(k.bsplineSurfaceFace({ surface: { ...surface, weights: cps.map((_, i) => (i === 5 ? 2 : 1)) } }));
    expect(w.valid).toBe(true);
  });

  it("sews six planar B-spline patches into a closed 10 mm cube", () => {
    const id = k.bsplinePatchesToShape({ surfaces: cubePatches(10), sewTolerance: 1e-3, makeSolid: true });
    const info = k.info(id);
    expect(info.type).toBe("solid");
    expect(info.closed).toBe(true);
    expect(info.valid).toBe(true);
    expect(info.volume!).toBeCloseTo(1000, 6);
    // without makeSolid: a closed shell
    const sh = k.info(k.bsplinePatchesToShape({ surfaces: cubePatches(10), sewTolerance: 1e-3, makeSolid: false }));
    expect(sh.type).toBe("shell");
    expect(sh.closed).toBe(true);
    // open set of patches stays a shell
    const open = k.info(k.bsplinePatchesToShape({ surfaces: cubePatches(10).slice(0, 5), sewTolerance: 1e-3, makeSolid: true }));
    expect(open.type).toBe("shell");
    expect(open.closed).toBe(false);
  });

  it("creates bounded primitive faces", () => {
    const pl = k.info(k.primitiveFace({ kind: "plane", origin: [0, 0, 0], normal: [0, 0, 1] }, { size: 20 }));
    expect(pl.area).toBeCloseTo(400, 6);
    const cyl = k.info(k.primitiveFace({ kind: "cylinder", origin: [0, 0, 0], axis: [0, 0, 1], radius: 5, hMin: 0, hMax: 10 }, { size: 1 }));
    expect(cyl.area).toBeCloseTo(2 * Math.PI * 5 * 10, 4);
    const cone = k.info(k.primitiveFace({ kind: "cone", apex: [0, 0, 0], axis: [0, 0, 1], halfAngle: Math.PI / 4, hMin: 0, hMax: 10 }, { size: 1 }));
    expect(cone.area).toBeCloseTo(Math.PI * 10 * Math.hypot(10, 10), 3);
    expect(cone.bbox.max[2]).toBeCloseTo(10, 4);
    const sph = k.info(k.primitiveFace({ kind: "sphere", center: [1, 2, 3], radius: 4 }, { size: 1 }));
    expect(sph.area).toBeCloseTo(4 * Math.PI * 16, 3);
    const quarter = k.info(k.primitiveFace({ kind: "cylinder", origin: [0, 0, 0], axis: [0, 0, 1], radius: 5, hMin: 0, hMax: 0 }, { uMin: 0, uMax: Math.PI / 2, vMin: 0, vMax: 4 }));
    expect(quarter.area).toBeCloseTo((Math.PI / 2) * 5 * 4, 5);
  });
});

describe("OcctKernel — booleans, fillets and other operations", () => {
  const box = (x: number, y: number, z: number, size: number) =>
    k.extrude({ frame: { ...WORLD_FRAME, origin: [0, 0, z] }, profile: { loops: [rect(x, y, x + size, y + size)] }, distance: size });

  it("cuts a cylinder from a box and fillets an outer edge", () => {
    const b = box(0, 0, 0, 20);
    const cyl = k.extrude({ frame: { ...WORLD_FRAME, origin: [0, 0, -5] }, profile: { loops: [circle(10, 10, 5)] }, distance: 30 });
    const cut = k.boolean("cut", b, cyl);
    const ci = k.info(cut);
    expect(ci.type).toBe("solid");
    expect(ci.valid).toBe(true);
    expect(ci.volume!).toBeCloseTo(8000 - Math.PI * 25 * 20, 5);
    const t = k.tessellate(cut, { linearDeflection: 0.05, angularDeflectionDeg: 10 });
    const edge = findEdge(t, [0, 0, 0], [0, 0, 20]);
    const r = 2;
    const fil = k.fillet(cut, [edge], r);
    const fi = k.info(fil);
    expect(fi.valid).toBe(true);
    expect(fi.volume!).toBeCloseTo(ci.volume! - (1 - Math.PI / 4) * r * r * 20, 4);
    expect(fi.faces).toBe(ci.faces + 1);
    const ch = k.info(k.chamfer(cut, [edge], r));
    expect(ch.volume!).toBeCloseTo(ci.volume! - (r * r * 20) / 2, 4);
    expect(() => k.fillet(cut, [edge], 50)).toThrow(/^Fillet failed/);
    expect(() => k.fillet(cut, [9999], 1)).toThrow(/edge id 9999 out of range/);
  });

  it("unions and intersects", () => {
    const a = box(0, 0, 0, 10), b = box(5, 5, 5, 10);
    const u = k.info(k.boolean("union", a, b));
    expect(u.type).toBe("solid");
    expect(u.volume!).toBeCloseTo(2000 - 125, 5);
    const i = k.info(k.boolean("intersect", a, b));
    expect(i.volume!).toBeCloseTo(125, 5);
    const simplified = k.info(k.boolean("union", a, box(10, 0, 0, 10), { simplify: true }));
    expect(simplified.faces).toBe(6);
    expect(() => k.boolean("intersect", a, box(50, 50, 50, 1))).toThrow(/Boolean intersect failed: result is empty/);
  });

  it("splits a solid with a plane face", () => {
    const b = box(0, 0, 0, 10);
    const plane = k.primitiveFace({ kind: "plane", origin: [5, 5, 5], normal: [1, 0, 0] }, { size: 40 });
    const sp = k.info(k.split(b, plane));
    expect(sp.solids).toBe(2);
    expect(sp.volume!).toBeCloseTo(1000, 5);
  });

  it("shells, thickens, transforms, mirrors and patterns", () => {
    const b = box(0, 0, 0, 20);
    const t = k.tessellate(b, { linearDeflection: 0.1, angularDeflectionDeg: 20 });
    // top face: the face whose triangles all lie at z = 20
    let top = -1;
    for (let f = 0; f < t.faceTriangleRanges.length / 2; f++) {
      const [s, n] = [t.faceTriangleRanges[2 * f], t.faceTriangleRanges[2 * f + 1]];
      let allTop = n > 0;
      for (let i = 3 * s; i < 3 * (s + n) && allTop; i++) allTop = Math.abs(t.positions[3 * t.indices[i] + 2] - 20) < 1e-6;
      if (allTop) top = f;
    }
    expect(top).toBeGreaterThanOrEqual(0);
    const sh = k.info(k.shell(b, [top], 1));
    expect(sh.valid).toBe(true);
    expect(sh.volume!).toBeCloseTo(8000 - 18 * 18 * 19, 3);

    const face = k.primitiveFace({ kind: "plane", origin: [0, 0, 0], normal: [0, 0, 1] }, { size: 10 });
    const th = k.info(k.thicken(face, 2));
    expect(th.type).toBe("solid");
    expect(th.volume!).toBeCloseTo(200, 4);

    // rotation 90deg about z + translation (column-major)
    const m = [0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 100, 0, 0, 1];
    const tr = k.info(k.transform(b, m));
    expect(tr.bbox.min[0]).toBeCloseTo(80, 5);
    expect(tr.bbox.max[1]).toBeCloseTo(20, 5);
    expect(tr.volume!).toBeCloseTo(8000, 5);

    const mi = k.info(k.mirror(b, { origin: [-10, 0, 0], normal: [1, 0, 0] }));
    expect(mi.bbox.min[0]).toBeCloseTo(-40, 5);
    expect(mi.volume!).toBeCloseTo(8000, 5);
    expect(mi.valid).toBe(true);

    const lp = k.info(k.linearPattern(b, [1, 0, 0], 3, 30));
    expect(lp.solids).toBe(3);
    expect(lp.bbox.max[0]).toBeCloseTo(80, 5);
    const lpf = k.info(k.linearPattern(b, [1, 0, 0], 3, 10, { fuse: true }));
    expect(lpf.solids).toBe(1);
    expect(lpf.volume!).toBeCloseTo(8000 + 2 * 4000, 3);

    const small = box(30, -2, 0, 4);
    const cp = k.info(k.circularPattern(small, [0, 0, 0], [0, 0, 1], 6, 360));
    expect(cp.solids).toBe(6);
    expect(cp.volume!).toBeCloseTo(6 * 64, 5);
    const comp = k.info(k.compound([b, small]));
    expect(comp.solids).toBe(2);
  });

  it("lofts and sweeps", () => {
    const sec = (z: number, r: number) => ({ frame: { ...WORLD_FRAME, origin: [0, 0, z] as Vec3 }, profile: { loops: [circle(0, 0, r)] } });
    const loft = k.info(k.loft({ sections: [sec(0, 5), sec(10, 3)], solid: true }));
    expect(loft.valid).toBe(true);
    expect(loft.volume!).toBeCloseTo(((Math.PI * 10) / 3) * (25 + 15 + 9), 3);
    const ruled = k.info(k.loft({ sections: [sec(0, 5), sec(10, 5), sec(20, 2)], solid: true, ruled: true }));
    expect(ruled.volume!).toBeCloseTo(Math.PI * 25 * 10 + ((Math.PI * 10) / 3) * (25 + 10 + 4), 3);

    const sw = k.info(k.sweep({ frame: WORLD_FRAME, profile: { loops: [circle(0, 0, 1)] }, path: { points: [[0, 0, 0], [0, 0, 30]], smooth: false } }));
    expect(sw.volume!).toBeCloseTo(Math.PI * 30, 4);
    const curved = k.info(
      k.sweep({ frame: WORLD_FRAME, profile: { loops: [circle(0, 0, 1)] }, path: { points: [[0, 0, 0], [0, 0, 10], [5, 0, 20], [10, 0, 25]] } }),
    );
    expect(curved.valid).toBe(true);
    expect(curved.volume!).toBeGreaterThan(Math.PI * 25);
    const tube = k.info(
      k.sweep({ frame: WORLD_FRAME, profile: { loops: [circle(0, 0, 2), circle(0, 0, 1)] }, path: { points: [[0, 0, 0], [0, 0, 10]], smooth: false } }),
    );
    expect(tube.volume!).toBeCloseTo(Math.PI * 3 * 10, 3);
  });
});

describe("OcctKernel — tessellation, export and registry", () => {
  it("tessellates with outward triangles and consistent face/edge indexing", () => {
    const b = k.extrude({ frame: WORLD_FRAME, profile: { loops: [rect(0, 0, 20, 20)] }, distance: 20 });
    const cyl = k.extrude({ frame: { ...WORLD_FRAME, origin: [0, 0, -5] }, profile: { loops: [circle(10, 10, 5)] }, distance: 30 });
    const moved = k.transform(k.boolean("cut", b, cyl), [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 100, 50, 0, 1]);
    const info = k.info(moved);
    const t = k.tessellate(moved, { linearDeflection: 0.01, angularDeflectionDeg: 5 });
    expect(t.faceTriangleRanges.length).toBe(2 * info.faces);
    let sum = 0;
    for (let f = 0; f < info.faces; f++) {
      expect(t.faceTriangleRanges[2 * f]).toBe(sum);
      expect(t.faceTriangleRanges[2 * f + 1]).toBeGreaterThan(0);
      sum += t.faceTriangleRanges[2 * f + 1];
    }
    expect(sum * 3).toBe(t.indices.length);
    expect(meshVolume(t)).toBeCloseTo(info.volume!, -1);
    expect(Math.abs(meshVolume(t) - info.volume!) / info.volume!).toBeLessThan(2e-3);
    expect(t.edges.map((e) => e.id)).toEqual([...Array(info.edges).keys()]);
    for (const e of t.edges) expect(e.points.length).toBeGreaterThanOrEqual(6);
    // per-vertex normals agree with the triangle windings
    let bad = 0;
    for (let i = 0; i < t.indices.length; i += 3) {
      const [a, b2, c] = [t.indices[i], t.indices[i + 1], t.indices[i + 2]].map((x) => 3 * x);
      const v = t.positions, n = t.normals;
      const ux = v[b2] - v[a], uy = v[b2 + 1] - v[a + 1], uz = v[b2 + 2] - v[a + 2];
      const wx = v[c] - v[a], wy = v[c + 1] - v[a + 1], wz = v[c + 2] - v[a + 2];
      const cx = uy * wz - uz * wy, cy = uz * wx - ux * wz, cz = ux * wy - uy * wx;
      if (cx * (n[a] + n[b2] + n[c]) + cy * (n[a + 1] + n[b2 + 1] + n[c + 1]) + cz * (n[a + 2] + n[b2 + 2] + n[c + 2]) < 0) bad++;
    }
    expect(bad).toBe(0);
  });

  it("exports STEP with names and B-spline geometry", () => {
    const cube = k.bsplinePatchesToShape({ surfaces: cubePatches(10), sewTolerance: 1e-3, makeSolid: true });
    const plate = k.extrude({ frame: WORLD_FRAME, profile: { loops: [rect(20, 0, 30, 10)] }, distance: 2 });
    const step = k.exportSTEP([{ id: cube, name: "FreeformCube" }, { id: plate, name: "Plate" }]);
    expect(step.startsWith("ISO-10303-21;")).toBe(true);
    expect(step).toContain("MANIFOLD_SOLID_BREP");
    expect(step).toContain("B_SPLINE_SURFACE_WITH_KNOTS");
    expect(step).toContain("FreeformCube");
    expect(step).toContain("Plate");
    expect(step).toMatch(/SI_UNIT\(\.MILLI\.,\.METRE\.\)/);
    const single = k.exportSTEP([{ id: plate }]);
    expect(single).toContain("MANIFOLD_SOLID_BREP");
    expect(k.oc.FS.readdir("/tmp").filter((f: string) => f.endsWith(".step"))).toEqual([]);
  });

  it("exports binary STL", () => {
    const b = k.extrude({ frame: WORLD_FRAME, profile: { loops: [rect(0, 0, 10, 10)] }, distance: 10 });
    const stl = k.exportSTL([{ id: b }], { linearDeflection: 0.1 });
    const dv = new DataView(stl.buffer);
    const n = dv.getUint32(80, true);
    expect(n).toBe(12);
    expect(stl.byteLength).toBe(84 + 50 * n);
  });

  it("serialises, releases and counts shapes", () => {
    const before = k.count();
    const b = k.extrude({ frame: WORLD_FRAME, profile: { loops: [rect(0, 0, 10, 10)] }, distance: 10 });
    const brep = k.serialize(b);
    expect(brep).toContain("CASCADE Topology");
    const c = k.deserialize(brep);
    expect(k.info(c).volume!).toBeCloseTo(1000, 6);
    expect(k.count()).toBe(before + 2);
    k.release(b);
    k.release(c);
    expect(k.count()).toBe(before);
    expect(() => k.info(b)).toThrow(/unknown shape id/);
  });
});
