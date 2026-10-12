import { describe, expect, it } from "vitest";
import type { Loop2D, Vec2, Vec3 } from "../core/types";
import { approximateSurface } from "./occt/brepToIges";
import {
  evalCurve3,
  evalSurface,
  interpolateCurveN,
  mirrorSurfaceU,
  reverseCurve,
  toDistinctKnots,
  toFullKnots,
} from "./nurbs";
import { arcGeometry, buildRegions, polygonize, signedArea, snapLoop, snapWire } from "./profile";

const rect = (x0: number, y0: number, x1: number, y1: number, cw = false): Loop2D => {
  let p: Vec2[] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
  if (cw) p = p.reverse();
  return p.map((a, i) => ({ type: "line" as const, a, b: p[(i + 1) % 4] }));
};

describe("profile analysis", () => {
  it("classifies outer loops, holes and islands and normalises orientation", () => {
    const regions = buildRegions({
      loops: [rect(40, 40, 60, 60), rect(0, 0, 100, 100, true), rect(20, 20, 80, 80), [{ type: "circle", center: [150, 0], radius: 5 }]],
    });
    expect(regions.length).toBe(3);
    const big = regions.find((r) => r.holes.length === 1)!;
    expect(signedArea(polygonize(big.outer))).toBeCloseTo(10000, 6);
    expect(signedArea(polygonize(big.holes[0]))).toBeCloseTo(-3600, 6);
    const island = regions.find((r) => Math.abs(signedArea(polygonize(r.outer)) - 400) < 1e-6);
    expect(island).toBeDefined();
    expect(island!.holes.length).toBe(0);
  });

  it("snaps tiny gaps and rejects open loops", () => {
    const loop: Loop2D = [
      { type: "line", a: [0, 0], b: [10, 0] },
      { type: "arc", start: [10, 1e-9], mid: [15, 5], end: [10, 10] },
      { type: "line", a: [10, 10], b: [0, 10] },
      { type: "line", a: [0, 10], b: [0, 0] },
    ];
    const s = snapLoop(loop);
    expect(s[1].type === "arc" && s[1].start).toEqual([10, 0]);
    expect(() => snapLoop([{ type: "line", a: [0, 0], b: [10, 0] }, { type: "line", a: [10, 0], b: [10, 10] }])).toThrow(/not closed/);
    expect(snapWire({ closed: false, segments: [{ type: "line", a: [0, 0], b: [1, 0] }] }).length).toBe(1);
  });

  it("computes arc direction from three points", () => {
    const ccw = arcGeometry({ start: [1, 0], mid: [0, 1], end: [-1, 0] })!;
    expect(ccw.sweep).toBeCloseTo(Math.PI, 12);
    const cw = arcGeometry({ start: [1, 0], mid: [0, -1], end: [-1, 0] })!;
    expect(cw.sweep).toBeCloseTo(-Math.PI, 12);
  });
});

describe("nurbs helpers", () => {
  it("converts knot vectors and reverses curves", () => {
    const full = [0, 0, 0, 0.5, 1, 1, 1];
    const d = toDistinctKnots(full);
    expect(d).toEqual({ knots: [0, 0.5, 1], mults: [3, 1, 3] });
    expect(toFullKnots(d.knots, d.mults)).toEqual(full);
    const c = { degree: 2, knots: full, controlPoints: [[0, 0, 0], [1, 2, 0], [3, 2, 0], [4, 0, 0]] as Vec3[] };
    const r = reverseCurve(c);
    for (const t of [0, 0.2, 0.5, 0.9]) {
      const a = evalCurve3(c, t), b = evalCurve3(r, 1 - t);
      expect(Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])).toBeLessThan(1e-12);
    }
  });

  it("interpolates curves and approximates surfaces in their own parameterisation", () => {
    const ts = [0, 0.5, 1.2, 2, 3];
    const pts = ts.map((t) => [Math.cos(t), Math.sin(t)]);
    const fit = interpolateCurveN(ts, pts);
    expect(fit.degree).toBe(3);
    // cylinder r = 50 over the full angle: tolerance 1e-4 respected at off-grid parameters
    const cyl = (u: number, v: number): Vec3 => [50 * Math.cos(u), 50 * Math.sin(u), v];
    const { surface, maxError } = approximateSurface(cyl, 0, 2 * Math.PI, 0, 10, 1e-4);
    expect(maxError).toBeLessThan(1e-4);
    for (let k = 0; k < 50; k++) {
      const u = (2 * Math.PI * (k + 0.31)) / 50, v = (10 * ((k * 7) % 50)) / 50;
      const p = evalSurface(surface, u, v), q = cyl(u, v);
      expect(Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2])).toBeLessThan(2e-4);
    }
    const m = mirrorSurfaceU(surface);
    const a = evalSurface(m, 1, 3), b = evalSurface(surface, 2 * Math.PI - 1, 3);
    expect(Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])).toBeLessThan(1e-9);
  });
});
