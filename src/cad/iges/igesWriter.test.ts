import { beforeAll, describe, expect, it } from "vitest";
import type { BSplineCurve2Def, BSplineCurveDef, BSplineSurfaceDef, Frame, Loop2D, Vec2, Vec3 } from "../../core/types";
import { WORLD_FRAME } from "../../core/vec";
import { OcctKernel } from "../OcctKernel";
import { curveDomain, evalCurve2, evalCurve3, evalSurface } from "../nurbs";
import { distanceToCurve } from "../occt/brepToIges";
import {
  curveComponents,
  decodeBSplineCurve,
  decodeBSplineCurve2,
  decodeBSplineSurface,
  entityByDE,
  parseIGES,
  tokenizeParams,
  type IgesFile,
} from "./igesReader";
import { formatReal, writeIGES, type TrimLoop } from "./igesWriter";

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

interface DecodedFace {
  surface: BSplineSurfaceDef;
  loops: { outer: boolean; c2: BSplineCurve2Def[]; c3: BSplineCurveDef[] }[];
}

function decodeFaces(file: IgesFile): DecodedFace[] {
  return file.entities
    .filter((e) => e.type === 144)
    .map((e) => {
      const [pts, n1, n2, pto, ...pti] = e.params as number[];
      const surface = decodeBSplineSurface(entityByDE(file, pts)).surface;
      const loopPtrs = [...(n1 ? [pto] : []), ...pti.slice(0, n2)];
      const loops = loopPtrs.map((de, i) => {
        const cos = entityByDE(file, de);
        expect(cos.type).toBe(142);
        const [, sptr, bptr, cptr] = cos.params as number[];
        expect(sptr).toBe(pts);
        return {
          outer: i === 0 && n1 === 1,
          c2: bptr ? curveComponents(file, bptr).map(decodeBSplineCurve2) : [],
          c3: cptr ? curveComponents(file, cptr).map((x) => decodeBSplineCurve(x).curve) : [],
        };
      });
      return { surface, loops };
    });
}

const ends2 = (c: BSplineCurve2Def): [Vec2, Vec2] => {
  const [a, b] = curveDomain(c);
  return [evalCurve2(c, a), evalCurve2(c, b)];
};
const ends3 = (c: BSplineCurveDef): [Vec3, Vec3] => {
  const [a, b] = curveDomain(c);
  return [evalCurve3(c, a), evalCurve3(c, b)];
};
const dist = (a: number[], b: number[]) => Math.hypot(...a.map((x, i) => x - b[i]));

/** Signed area of a 2D loop (sampled). */
function loopArea(c2: BSplineCurve2Def[]): number {
  const pts: Vec2[] = [];
  for (const c of c2) {
    const [a, b] = curveDomain(c);
    for (let k = 0; k < 32; k++) pts.push(evalCurve2(c, a + ((b - a) * k) / 32));
  }
  let A = 0;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i], q = pts[(i + 1) % pts.length];
    A += p[0] * q[1] - q[0] * p[1];
  }
  return A / 2;
}

/** Checks loops are closed chains and that S(B(t)) lies on the 3D boundary. Returns max deviation. */
function checkFace(f: DecodedFace): number {
  let maxDev = 0;
  for (const loop of f.loops) {
    for (let i = 0; i < loop.c2.length; i++) {
      const e = ends2(loop.c2[i])[1], s = ends2(loop.c2[(i + 1) % loop.c2.length])[0];
      expect(dist(e, s)).toBeLessThan(1e-6);
    }
    for (let i = 0; i < loop.c3.length; i++) {
      const e = ends3(loop.c3[i])[1], s = ends3(loop.c3[(i + 1) % loop.c3.length])[0];
      expect(dist(e, s)).toBeLessThan(1e-5);
    }
    for (const c of loop.c2) {
      const [a, b] = curveDomain(c);
      for (let k = 0; k <= 9; k++) {
        // off-grid samples (k + 0.37) so we do not only hit interpolation nodes
        const uv = evalCurve2(c, a + ((b - a) * Math.min(9, k + 0.37)) / 9.37);
        const p = evalSurface(f.surface, uv[0], uv[1]);
        maxDev = Math.max(maxDev, Math.min(...loop.c3.map((c3) => distanceToCurve(c3, p, 32))));
      }
    }
    const area = loopArea(loop.c2);
    if (loop.c2.length) expect(area > 0).toBe(loop.outer);
  }
  return maxDev;
}

function checkStructure(text: string): IgesFile {
  const lines = text.split("\n").filter((l) => l.length);
  for (const l of lines) expect(l.length).toBe(80);
  const file = parseIGES(text); // validates sequence numbers, back pointers and Terminate counts
  expect(file.global[0]).toBe(",");
  expect(file.global[1]).toBe(";");
  expect(file.global[13]).toBe(2); // units flag: mm
  expect(file.global[14]).toBe("MM");
  expect(file.global[22]).toBe(11); // IGES 5.3
  return file;
}

const count = (file: IgesFile, type: number) => file.entities.filter((e) => e.type === type).length;

// ---------------------------------------------------------------------------
// writer unit tests (no OCCT)
// ---------------------------------------------------------------------------

describe("IGES writer", () => {
  it("formats reals and Hollerith-aware tokens", () => {
    expect(formatReal(1)).toBe("1.0");
    expect(formatReal(-0.5)).toBe("-0.5");
    expect(formatReal(1e-7)).toBe("1.0E-7");
    expect(formatReal(123456.789)).toBe("123456.789");
    expect(formatReal(42)).toBe("42.0");
    expect(Number(formatReal(Math.SQRT1_2))).toBe(Math.SQRT1_2);
    expect(formatReal(2.5e21)).toBe("2.5E21");
    expect(tokenizeParams("128,1,2.5,3HA,B,,1.0D-3;")).toEqual([128, 1, 2.5, "A,B", null, 0.001]);
  });

  it("writes a trimmed planar patch with a hole and reads it back", () => {
    const surface: BSplineSurfaceDef = {
      degreeU: 1, degreeV: 1, nu: 2, nv: 2,
      knotsU: [0, 0, 10, 10], knotsV: [0, 0, 10, 10],
      controlPoints: [[0, 0, 0], [0, 10, 0], [10, 0, 0], [10, 10, 0]],
    };
    const line2 = (a: Vec2, b: Vec2): BSplineCurve2Def => ({ degree: 1, knots: [0, 0, 1, 1], controlPoints: [a, b] });
    const line3 = (a: Vec2, b: Vec2): BSplineCurveDef => ({ degree: 1, knots: [0, 0, 1, 1], controlPoints: [[a[0], a[1], 0], [b[0], b[1], 0]] });
    const sq: Vec2[] = [[1, 1], [9, 1], [9, 9], [1, 9]];
    const outer: TrimLoop = sq.map((p, i) => ({ curve2d: line2(p, sq[(i + 1) % 4]), curve3d: line3(p, sq[(i + 1) % 4]) }));
    // clockwise rational circle (r = 2 around (5,5)) as 2D + 3D curves
    const w = Math.SQRT1_2;
    const circ2: Vec2[] = [[7, 5], [7, 3], [5, 3], [3, 3], [3, 5], [3, 7], [5, 7], [7, 7], [7, 5]];
    const knots = [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4];
    const weights = [1, w, 1, w, 1, w, 1, w, 1];
    const hole: TrimLoop = [
      {
        curve2d: { degree: 2, knots, controlPoints: circ2, weights },
        curve3d: { degree: 2, knots, controlPoints: circ2.map(([x, y]) => [x, y, 0] as Vec3), weights },
      },
    ];
    const text = writeIGES({ surfaces: [{ surface, trims: { outer, inner: [hole] }, name: "PLATE" }], units: "mm", author: "Test" });
    const file = checkStructure(text);
    expect(count(file, 144)).toBe(1);
    expect(count(file, 128)).toBe(1);
    expect(count(file, 142)).toBe(2);
    expect(count(file, 102)).toBe(2); // outer loop 2D + 3D composites; the hole is a single curve
    expect(count(file, 126)).toBe(10);
    const e144 = file.entities.find((e) => e.type === 144)!;
    expect(e144.label).toBe("PLATE");
    expect(e144.status).toBe("00000000");
    expect(file.entities.find((e) => e.type === 128)!.status).toBe("00010000");
    expect(file.entities.filter((e) => e.type === 126 && e.status === "00010500").length).toBe(5);
    const [face] = decodeFaces(file);
    expect(face.surface).toEqual({ ...surface, weights: undefined });
    expect(checkFace(face)).toBeLessThan(1e-9);
    const holeCurve = face.loops[1].c3[0];
    expect(holeCurve.weights).toEqual(weights);
  });

  it("writes untrimmed surfaces as independent 128 entities and free curves as 126", () => {
    const surface: BSplineSurfaceDef = {
      degreeU: 1, degreeV: 1, nu: 2, nv: 2, knotsU: [0, 0, 1, 1], knotsV: [0, 0, 1, 1],
      controlPoints: [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]],
    };
    const text = writeIGES({ surfaces: [{ surface }], curves: [{ degree: 1, knots: [0, 0, 1, 1], controlPoints: [[0, 0, 0], [1, 2, 3]] }], units: "mm" });
    const file = checkStructure(text);
    expect(file.entities.map((e) => [e.type, e.status])).toEqual([[128, "00000000"], [126, "00000000"]]);
  });
});

// ---------------------------------------------------------------------------
// B-rep -> IGES through the kernel
// ---------------------------------------------------------------------------

const rect = (x0: number, y0: number, x1: number, y1: number): Loop2D => {
  const p: Vec2[] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
  return p.map((a, i) => ({ type: "line" as const, a, b: p[(i + 1) % 4] }));
};
const circle = (cx: number, cy: number, r: number): Loop2D => [{ type: "circle", center: [cx, cy], radius: r }];

describe("OcctKernel.exportIGES", () => {
  let k: OcctKernel;
  beforeAll(async () => {
    k = await OcctKernel.create();
  });

  it("exports a box and a cylinder as trimmed NURBS surfaces", () => {
    const box = k.extrude({ frame: WORLD_FRAME, profile: { loops: [rect(0, 0, 10, 20)] }, distance: 30 });
    const cyl = k.extrude({ frame: { ...WORLD_FRAME, origin: [40, 0, 0] }, profile: { loops: [circle(0, 0, 5)] }, distance: 10 });
    const t0 = performance.now();
    const text = k.exportIGES([{ id: box, name: "Box" }, { id: cyl, name: "Cylinder" }], { author: "vitest" });
    const ms = performance.now() - t0;
    const file = checkStructure(text);
    expect(count(file, 144)).toBe(6 + 3);
    expect(count(file, 128)).toBe(9);
    expect(count(file, 142)).toBe(9);
    // every 142 has both parameter-space and model-space curves, parametric preferred
    for (const e of file.entities.filter((x) => x.type === 142)) {
      expect(e.params[2]).not.toBe(0);
      expect(e.params[3]).not.toBe(0);
      expect(e.params[4]).toBe(1);
    }
    const faces = decodeFaces(file);
    let maxDev = 0;
    for (const f of faces) maxDev = Math.max(maxDev, checkFace(f));
    expect(maxDev).toBeLessThan(1e-3);
    // outward normals: the box top face normal (via surface partials) points +z
    const top = faces.find((f) => f.surface.controlPoints.every((p) => Math.abs(p[2] - 30) < 1e-9))!;
    const [p00, p01, p10] = [top.surface.controlPoints[0], top.surface.controlPoints[1], top.surface.controlPoints[2]];
    const du = [p10[0] - p00[0], p10[1] - p00[1]], dv = [p01[0] - p00[0], p01[1] - p00[1]];
    expect(du[0] * dv[1] - du[1] * dv[0]).toBeGreaterThan(0);
    const bottom = faces.find((f) => f.surface.controlPoints.every((p) => Math.abs(p[2]) < 1e-9) && f.surface.controlPoints[0][0] < 20)!;
    const [q00, q01, q10] = [bottom.surface.controlPoints[0], bottom.surface.controlPoints[1], bottom.surface.controlPoints[2]];
    const eu = [q10[0] - q00[0], q10[1] - q00[1]], ev = [q01[0] - q00[0], q01[1] - q00[1]];
    expect(eu[0] * ev[1] - eu[1] * ev[0]).toBeLessThan(0);
    // eslint-disable-next-line no-console
    console.log(`IGES box+cylinder: ${ms.toFixed(1)} ms, ${text.split("\n").length} lines, max trim deviation ${maxDev.toExponential(2)}`);
  });

  it("handles holes, spheres (degenerate edges), B-spline solids and located shapes", () => {
    const plate = k.extrude({ frame: WORLD_FRAME, profile: { loops: [rect(0, 0, 40, 20), circle(20, 10, 5)] }, distance: 5 });
    const xz: Frame = { origin: [0, 0, 0], xAxis: [1, 0, 0], yAxis: [0, 0, 1], zAxis: [0, -1, 0] };
    const sphere = k.revolve({
      frame: xz,
      profile: { loops: [[{ type: "arc", start: [0, -6], mid: [6, 0], end: [0, 6] }, { type: "line", a: [0, 6], b: [0, -6] }]] },
      axisOrigin: [0, 0, 0], axisDir: [0, 0, 1], angleDeg: 360,
    });
    const L = 10;
    const patch = (c0: Vec3, du: Vec3, dv: Vec3): BSplineSurfaceDef => {
      const cps: Vec3[] = [];
      for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) cps.push([0, 1, 2].map((q) => c0[q] + (du[q] * i) / 3 + (dv[q] * j) / 3) as Vec3);
      return { degreeU: 3, degreeV: 3, knotsU: [0, 0, 0, 0, 1, 1, 1, 1], knotsV: [0, 0, 0, 0, 1, 1, 1, 1], nu: 4, nv: 4, controlPoints: cps };
    };
    const cube = k.bsplinePatchesToShape({
      surfaces: [
        patch([0, 0, 0], [0, L, 0], [L, 0, 0]), patch([0, 0, L], [L, 0, 0], [0, L, 0]), patch([0, 0, 0], [L, 0, 0], [0, 0, L]),
        patch([0, L, 0], [0, 0, L], [L, 0, 0]), patch([0, 0, 0], [0, 0, L], [0, L, 0]), patch([L, 0, 0], [0, L, 0], [0, 0, L]),
      ],
      sewTolerance: 1e-3, makeSolid: true,
    });
    const movedCube = k.transform(cube, [0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 100, 0, 0, 1]);
    const text = k.exportIGES([{ id: plate }, { id: sphere }, { id: movedCube }]);
    const file = checkStructure(text);
    const faces = decodeFaces(file);
    expect(faces.length).toBe(7 + 1 + 6);
    const holeFace = faces.filter((f) => f.loops.length === 2);
    expect(holeFace.length).toBe(2); // top and bottom of the plate
    let maxDev = 0;
    for (const f of faces) maxDev = Math.max(maxDev, checkFace(f));
    expect(maxDev).toBeLessThan(1e-3);
    // located cube: all control points moved to x in [90, 100]
    const cubeFaces = faces.slice(8);
    for (const f of cubeFaces) for (const p of f.surface.controlPoints) expect(p[0]).toBeGreaterThan(89.999);
  });
});
