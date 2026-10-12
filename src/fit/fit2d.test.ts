import { describe, expect, it } from "vitest";
import type { Vec2 } from "../core/types";
import { fitArc2D, fitCircle2D, fitLine2D, fitLine3D } from "./fit2d";
import { makeRand } from "./testing/synthetic";

describe("2D fitting", () => {
  it("circle from a noisy 60° arc", () => {
    const r = makeRand(1);
    const pts: Vec2[] = Array.from({ length: 300 }, (_, i) => {
      const a = 0.2 + (i / 299) * (Math.PI / 3);
      const rr = 40 + 0.01 * r.gauss();
      return [7 + rr * Math.cos(a), -3 + rr * Math.sin(a)];
    });
    const c = fitCircle2D(pts);
    expect(Math.abs(c.radius - 40)).toBeLessThan(0.02);
    expect(Math.hypot(c.center[0] - 7, c.center[1] + 3)).toBeLessThan(0.02);
    expect(c.rms).toBeLessThan(0.011);
    const fixed = fitCircle2D(pts, { radius: 41 });
    expect(fixed.radius).toBe(41);
  });

  it("arc sweep follows the point order", () => {
    const ccw: Vec2[] = Array.from({ length: 50 }, (_, i) => [Math.cos(i * 0.05), Math.sin(i * 0.05)]);
    const a = fitArc2D(ccw);
    expect(a.sweep).toBeCloseTo(49 * 0.05, 9);
    expect(a.start[0]).toBeCloseTo(1, 9);
    const cw = fitArc2D([...ccw].reverse());
    expect(cw.sweep).toBeCloseTo(-49 * 0.05, 9);
    expect(a.mid[0]).toBeCloseTo(Math.cos(49 * 0.025), 9);
  });

  it("lines", () => {
    const l = fitLine2D([
      [0, 0],
      [1, 1],
      [2, 2.0],
      [3, 3],
    ]);
    expect(Math.abs(l.direction[0])).toBeCloseTo(Math.SQRT1_2, 12);
    expect(l.rms).toBeCloseTo(0, 12);
    expect(() => fitCircle2D([[0, 0], [1, 1], [2, 2]])).toThrow(/collinear/);
    const pos = new Float32Array([0, 0, 0, 1, 2, 3, 2, 4, 6, 3, 6, 9.01]);
    const l3 = fitLine3D(pos);
    expect(l3.def.direction[2] / l3.def.direction[0]).toBeCloseTo(3, 2);
    expect(l3.tMax - l3.tMin).toBeGreaterThan(11);
  });
});
