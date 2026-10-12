import { describe, expect, it } from "vitest";
import type { Segment2D, Vec2 } from "../core/types";
import { closestPointOnSegment, segmentArcGeom, segmentEnd, segmentStart } from "./geom2d";
import { autoSegment, autoSegmentDetailed, autoSketch, fitArc2D, fitBSpline2D, fitCircle2D, fitLine2D } from "./fit2d";
import { solveSketch } from "./solver";
import { extractProfiles } from "./profiles";

/** Deterministic PRNG + gaussian noise. */
function rng(seed: number) {
  let s = seed >>> 0;
  const u = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return (s + 0.5) / 4294967296;
  };
  return { u, g: () => Math.sqrt(-2 * Math.log(u())) * Math.cos(2 * Math.PI * u()) };
}

/** Sample a chain of segments (exact geometry) every `step` mm and add gaussian noise (normal to nothing in particular). */
function sampleChain(segs: Segment2D[], step: number, sigma: number, seed = 1, closed = true): Vec2[] {
  const r = rng(seed);
  const out: Vec2[] = [];
  for (const s of segs) {
    let L: number;
    let at: (t: number) => Vec2;
    if (s.type === "line") {
      L = Math.hypot(s.b[0] - s.a[0], s.b[1] - s.a[1]);
      at = (t) => [s.a[0] + (s.b[0] - s.a[0]) * t, s.a[1] + (s.b[1] - s.a[1]) * t];
    } else if (s.type === "arc") {
      const g = segmentArcGeom(s)!;
      L = g.radius * Math.abs(g.sweep);
      at = (t) => [g.center[0] + g.radius * Math.cos(g.a0 + g.sweep * t), g.center[1] + g.radius * Math.sin(g.a0 + g.sweep * t)];
    } else if (s.type === "circle") {
      L = 2 * Math.PI * s.radius;
      at = (t) => [s.center[0] + s.radius * Math.cos(2 * Math.PI * t), s.center[1] + s.radius * Math.sin(2 * Math.PI * t)];
    } else throw new Error("unsupported");
    const n = Math.max(1, Math.round(L / step));
    for (let i = 0; i < n; i++) {
      const p = at(i / n);
      out.push([p[0] + sigma * r.g(), p[1] + sigma * r.g()]);
    }
  }
  if (!closed) {
    const last = segs[segs.length - 1];
    const p = segmentEnd(last);
    out.push([p[0] + sigma * r.g(), p[1] + sigma * r.g()]);
  }
  return out;
}

const arc = (c: Vec2, r: number, a0: number, a1: number): Segment2D => {
  const am = (a0 + a1) / 2;
  return {
    type: "arc",
    start: [c[0] + r * Math.cos(a0), c[1] + r * Math.sin(a0)],
    mid: [c[0] + r * Math.cos(am), c[1] + r * Math.sin(am)],
    end: [c[0] + r * Math.cos(a1), c[1] + r * Math.sin(a1)],
  };
};
const line = (a: Vec2, b: Vec2): Segment2D => ({ type: "line", a, b });

/** CCW rounded rectangle [0,W]x[0,H] with fillet radius R. */
function roundedRect(W: number, H: number, R: number): Segment2D[] {
  const h = Math.PI / 2;
  return [
    line([R, 0], [W - R, 0]),
    arc([W - R, R], R, -h, 0),
    line([W, R], [W, H - R]),
    arc([W - R, H - R], R, 0, h),
    line([W - R, H], [R, H]),
    arc([R, H - R], R, h, 2 * h),
    line([0, H - R], [0, R]),
    arc([R, R], R, 2 * h, 3 * h),
  ];
}

function maxDeviation(points: Vec2[], segs: Segment2D[]): number {
  let mx = 0;
  for (const p of points) {
    let best = Infinity;
    for (const s of segs) best = Math.min(best, closestPointOnSegment(s, p).distance);
    mx = Math.max(mx, best);
  }
  return mx;
}

function expectChained(segs: Segment2D[], closed: boolean) {
  for (let i = 0; i + 1 < segs.length; i++) expect(segmentEnd(segs[i])).toEqual(segmentStart(segs[i + 1]));
  if (closed && segs.length > 1) expect(segmentEnd(segs[segs.length - 1])).toEqual(segmentStart(segs[0]));
}

describe("primitive fits", () => {
  it("fits a line with extents and stats", () => {
    const r = rng(3);
    const pts: Vec2[] = [];
    for (let i = 0; i <= 100; i++) pts.push([i * 0.5 + 0.01 * r.g(), 2 + i * 0.25 + 0.01 * r.g()]);
    const f = fitLine2D(pts);
    expect(Math.abs(f.direction[1] / f.direction[0] - 0.5)).toBeLessThan(1e-3);
    expect(f.a[0]).toBeCloseTo(0, 1);
    expect(f.b[0]).toBeCloseTo(50, 1);
    expect(f.stats.rms).toBeLessThan(0.02);
    expect(f.stats.count).toBe(101);
  });

  it("fits circles and arcs (noisy, partial)", () => {
    const r = rng(5);
    const pts: Vec2[] = [];
    for (let i = 0; i <= 60; i++) {
      const a = 0.3 + (i / 60) * 1.2; // ~69 deg arc
      pts.push([10 + 7 * Math.cos(a) + 0.01 * r.g(), -4 + 7 * Math.sin(a) + 0.01 * r.g()]);
    }
    const c = fitCircle2D(pts)!;
    expect(c.center[0]).toBeCloseTo(10, 1);
    expect(c.center[1]).toBeCloseTo(-4, 1);
    expect(c.radius).toBeCloseTo(7, 1);
    expect(c.stats.rms).toBeLessThan(0.015);
    const a = fitArc2D(pts)!;
    expect(a.sweep).toBeGreaterThan(1.15);
    expect(a.sweep).toBeLessThan(1.25);
    expect(Math.hypot(a.start[0] - pts[0][0], a.start[1] - pts[0][1])).toBeLessThan(0.05);
    expect(Math.hypot(a.end[0] - pts[60][0], a.end[1] - pts[60][1])).toBeLessThan(0.05);
    // reversed data gives a CW sweep
    expect(fitArc2D(pts.slice().reverse())!.sweep).toBeLessThan(0);
    // collinear -> null
    expect(fitCircle2D([[0, 0], [1, 1], [2, 2], [3, 3]])).toBeNull();
    expect(fitCircle2D([[0, 0], [1, 1]])).toBeNull();
  });

  it("fits a B-spline", () => {
    const pts: Vec2[] = [];
    for (let i = 0; i <= 100; i++) {
      const x = (i / 100) * 10;
      pts.push([x, Math.sin(x)]);
    }
    const f = fitBSpline2D(pts, 12, 3)!;
    expect(f.curve.controlPoints.length).toBe(12);
    expect(f.stats.maxAbs).toBeLessThan(0.03);
    const g = fitBSpline2D(pts, 24, 3)!;
    expect(g.stats.maxAbs).toBeLessThan(2e-3);
    expect(g.curve.knots.length).toBe(24 + 4);
    // ends interpolated
    expect(g.curve.controlPoints[0]).toEqual(pts[0]);
    expect(g.curve.controlPoints[23]).toEqual(pts[100]);
  });
});

describe("autoSegment", () => {
  it("rounded rectangle -> 4 lines + 4 tangent arcs", () => {
    const truth = roundedRect(100, 60, 8);
    const pts = sampleChain(truth, 0.5, 0.02, 11);
    // start the data in the middle of the bottom edge
    const rot = pts.slice(40).concat(pts.slice(0, 40));
    const det = autoSegmentDetailed(rot, { tolerance: 0.1, closed: true });
    const types = det.segments.map((s) => s.type);
    expect(types.filter((t) => t === "line").length).toBe(4);
    expect(types.filter((t) => t === "arc").length).toBe(4);
    expect(det.tangent.every((t) => t)).toBe(true);
    for (const s of det.segments)
      if (s.type === "arc") {
        const g = segmentArcGeom(s)!;
        expect(Math.abs(g.radius - 8)).toBeLessThan(0.1);
        expect(Math.abs(Math.abs(g.sweep) - Math.PI / 2)).toBeLessThan(0.03);
      }
    expectChained(det.segments, true);
    expect(det.maxError).toBeLessThan(0.15);
    expect(maxDeviation(rot, det.segments)).toBeLessThan(0.15);
  });

  it("slot -> 2 lines + 2 arcs", () => {
    const truth = [line([0, 0], [50, 0]), arc([50, 10], 10, -Math.PI / 2, Math.PI / 2), line([50, 20], [0, 20]), arc([0, 10], 10, Math.PI / 2, 1.5 * Math.PI)];
    const pts = sampleChain(truth, 0.4, 0.015, 7);
    const segs = autoSegment(pts, { tolerance: 0.08, closed: true });
    expect(segs.filter((s) => s.type === "line").length).toBe(2);
    expect(segs.filter((s) => s.type === "arc").length).toBe(2);
    for (const s of segs) if (s.type === "arc") expect(Math.abs(segmentArcGeom(s)!.radius - 10)).toBeLessThan(0.08);
    expectChained(segs, true);
  });

  it("circle -> 1 circle (or 2 arcs)", () => {
    const pts = sampleChain([{ type: "circle", center: [5, 5], radius: 25 }], 0.5, 0.02, 3);
    const segs = autoSegment(pts, { tolerance: 0.1, closed: true });
    expect(segs.length).toBe(1);
    expect(segs[0].type).toBe("circle");
    if (segs[0].type === "circle") expect(segs[0].radius).toBeCloseTo(25, 1);
    const two = autoSegment(pts, { tolerance: 0.1, closed: true, fullCircle: false });
    expect(two.map((s) => s.type)).toEqual(["arc", "arc"]);
    expectChained(two, true);
  });

  it("L-bracket -> 6 lines with sharp corners", () => {
    const P: Vec2[] = [[0, 0], [80, 0], [80, 10], [10, 10], [10, 50], [0, 50]];
    const truth = P.map((p, i) => line(p, P[(i + 1) % P.length]));
    const pts = sampleChain(truth, 0.5, 0.02, 9);
    const det = autoSegmentDetailed(pts, { tolerance: 0.1, closed: true });
    expect(det.segments.length).toBe(6);
    expect(det.segments.every((s) => s.type === "line")).toBe(true);
    expect(det.tangent.some((t) => t)).toBe(false);
    // corners recovered precisely
    for (const s of det.segments) {
      const a = (s as { a: Vec2 }).a;
      const near = Math.min(...P.map((p) => Math.hypot(p[0] - a[0], p[1] - a[1])));
      expect(near).toBeLessThan(0.05);
    }
    expectChained(det.segments, true);
  });

  it("L-bracket with an inner fillet and open sections", () => {
    const truth = [line([0, 0], [80, 0]), line([80, 0], [80, 10]), line([80, 10], [16, 10]), arc([16, 16], 6, -Math.PI / 2, -Math.PI), line([10, 16], [10, 50]), line([10, 50], [0, 50]), line([0, 50], [0, 0])];
    const pts = sampleChain(truth, 0.5, 0.02, 21);
    const det = autoSegmentDetailed(pts, { tolerance: 0.1, closed: true });
    expect(det.segments.filter((s) => s.type === "line").length).toBe(6);
    expect(det.segments.filter((s) => s.type === "arc").length).toBe(1);
    expect(det.tangent.filter((t) => t).length).toBe(2);
    // open: line - arc - line
    const open = [line([0, 0], [30, 0]), arc([30, 10], 10, -Math.PI / 2, 0), line([40, 10], [40, 40])];
    const op = sampleChain(open, 0.3, 0.01, 4, false);
    const od = autoSegmentDetailed(op, { tolerance: 0.05 });
    expect(od.segments.map((s) => s.type)).toEqual(["line", "arc", "line"]);
    expect(od.tangent).toEqual([true, true]);
    expectChained(od.segments, false);
    expect(Math.hypot(segmentStart(od.segments[0])[0], segmentStart(od.segments[0])[1])).toBeLessThan(0.05);
  });

  it("merges collinear pieces and handles tiny inputs", () => {
    const pts: Vec2[] = [];
    for (let i = 0; i <= 200; i++) pts.push([i * 0.5, 0.001 * Math.sin(i)]);
    expect(autoSegment(pts, { tolerance: 0.01 }).length).toBe(1);
    expect(autoSegment([], { tolerance: 0.1 })).toEqual([]);
    expect(autoSegment([[0, 0]], { tolerance: 0.1 })).toEqual([]);
    expect(autoSegment([[0, 0], [1, 0]], { tolerance: 0.1 }).length).toBe(1);
    expect(autoSegment([[0, 0], [1, 0], [1, 0], [2, 0.5]], { tolerance: 0.01 }).length).toBeGreaterThan(0);
  });

  it("polyline mode without arcs stays within tolerance", () => {
    const pts = sampleChain([{ type: "circle", center: [0, 0], radius: 10 }], 0.2, 0, 1);
    const segs = autoSegment(pts, { tolerance: 0.05, closed: true, allowArcs: false });
    expect(segs.every((s) => s.type === "line")).toBe(true);
    expect(maxDeviation(pts, segs)).toBeLessThan(0.05 + 1e-9);
  });
});

describe("autoSegment robustness", () => {
  const h = Math.PI / 2;
  const shapes: Record<string, { segs: Segment2D[]; lines: number; arcs: number }> = {
    rrect: { segs: roundedRect(100, 60, 3), lines: 4, arcs: 4 },
    slot: { segs: [line([0, 0], [50, 0]), arc([50, 10], 10, -h, h), line([50, 20], [0, 20]), arc([0, 10], 10, h, 3 * h)], lines: 2, arcs: 2 },
    cam: { segs: [arc([0, 0], 20, -h, h), line([0, 20], [-30, 12]), arc([-30, 0], 12, h, 3 * h), line([-30, -12], [0, -20])], lines: 2, arcs: 2 },
  };
  it("recovers the primitive counts over seeds, rotations, start offsets and densities", () => {
    for (const sh of Object.values(shapes))
      for (let seed = 1; seed <= 6; seed++)
        for (const step of [0.3, 0.8]) {
          const r0 = sampleChain(sh.segs, step, 0.025, seed + 50);
          const c = Math.cos(seed), sn = Math.sin(seed);
          const rotd = r0.map(([x, y]) => [x * c - y * sn, x * sn + y * c] as Vec2);
          const off = (seed * 41) % rotd.length;
          const pts = rotd.slice(off).concat(rotd.slice(0, off));
          const det = autoSegmentDetailed(pts, { tolerance: 0.1, closed: true });
          expect(det.segments.filter((s) => s.type === "line").length).toBe(sh.lines);
          expect(det.segments.filter((s) => s.type === "arc").length).toBe(sh.arcs);
          expect(det.maxError).toBeLessThan(0.1 * 1.25);
          expectChained(det.segments, true);
        }
  });

  it("approximates free-form data within tolerance", () => {
    const ell: Vec2[] = [];
    for (let i = 0; i < 2000; i++) {
      const a = (i / 2000) * 2 * Math.PI;
      ell.push([60 * Math.cos(a), 25 * Math.sin(a)]);
    }
    const det = autoSegmentDetailed(ell, { tolerance: 0.05, closed: true });
    expect(det.segments.length).toBeLessThan(30);
    expect(maxDeviation(ell, det.segments)).toBeLessThan(0.05 * 1.1);
    expectChained(det.segments, true);
  });
});

describe("autoSketch", () => {
  it("builds a constrained sketch with shared points and tangent constraints", () => {
    const truth = roundedRect(100, 60, 8);
    const pts = sampleChain(truth, 0.5, 0.02, 11);
    const sk = autoSketch(pts, { tolerance: 0.1, closed: true, horizontalVertical: true });
    expect(sk.entities.filter((e) => e.type === "line").length).toBe(4);
    expect(sk.entities.filter((e) => e.type === "arc").length).toBe(4);
    expect(sk.constraints.filter((c) => c.type === "tangent").length).toBe(8);
    expect(sk.constraints.filter((c) => c.type === "horizontal").length).toBe(2);
    expect(sk.constraints.filter((c) => c.type === "vertical").length).toBe(2);
    // 8 junction points + 4 centers
    expect(sk.points.length).toBe(12);
    const res = solveSketch(sk);
    expect(res.status).toBe("ok");
    expect(res.residual).toBeLessThan(1e-8);
    // the profile is one closed loop
    const pr = extractProfiles(sk);
    expect(pr.profile.loops.length).toBe(1);
    expect(pr.profile.loops[0].length).toBe(8);
    expect(pr.openWires.length).toBe(0);
  });

  it("circle section becomes a circle entity", () => {
    const pts = sampleChain([{ type: "circle", center: [5, 5], radius: 25 }], 0.5, 0.02, 3);
    const sk = autoSketch(pts, { tolerance: 0.1, closed: true });
    expect(sk.entities.map((e) => e.type)).toEqual(["circle"]);
  });
});
