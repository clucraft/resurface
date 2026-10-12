import { it } from "vitest";
import type { Segment2D, Vec2 } from "../core/types";
import { closestPointOnSegment, segmentArcGeom } from "./geom2d";
import { autoSegmentDetailed } from "./fit2d";
function rng(seed: number) {
  let s = seed >>> 0;
  const u = () => { s = (s * 1664525 + 1013904223) >>> 0; return (s + 0.5) / 4294967296; };
  return { u, g: () => Math.sqrt(-2 * Math.log(u())) * Math.cos(2 * Math.PI * u()) };
}
function sampleChain(segs: Segment2D[], step: number, sigma: number, seed = 1, rotA = 0): Vec2[] {
  const r = rng(seed); const out: Vec2[] = [];
  const c = Math.cos(rotA), s0 = Math.sin(rotA);
  for (const s of segs) {
    let L: number; let at: (t: number) => Vec2;
    if (s.type === "line") { L = Math.hypot(s.b[0] - s.a[0], s.b[1] - s.a[1]); at = (t) => [s.a[0] + (s.b[0] - s.a[0]) * t, s.a[1] + (s.b[1] - s.a[1]) * t]; }
    else { const g = segmentArcGeom(s as any)!; L = g.radius * Math.abs(g.sweep); at = (t) => [g.center[0] + g.radius * Math.cos(g.a0 + g.sweep * t), g.center[1] + g.radius * Math.sin(g.a0 + g.sweep * t)]; }
    const n = Math.max(1, Math.round(L / step));
    for (let i = 0; i < n; i++) { const p = at(i / n); const q: Vec2 = [p[0] * c - p[1] * s0, p[0] * s0 + p[1] * c]; out.push([q[0] + sigma * r.g(), q[1] + sigma * r.g()]); }
  }
  return out;
}
const arc = (c: Vec2, r: number, a0: number, a1: number): Segment2D => { const am = (a0 + a1) / 2; return { type: "arc", start: [c[0] + r * Math.cos(a0), c[1] + r * Math.sin(a0)], mid: [c[0] + r * Math.cos(am), c[1] + r * Math.sin(am)], end: [c[0] + r * Math.cos(a1), c[1] + r * Math.sin(a1)] }; };
const line = (a: Vec2, b: Vec2): Segment2D => ({ type: "line", a, b });
const h = Math.PI / 2;
const shapes: Record<string, { segs: Segment2D[]; lines: number; arcs: number }> = {
  rrect: { segs: [line([8, 0], [92, 0]), arc([92, 8], 8, -h, 0), line([100, 8], [100, 52]), arc([92, 52], 8, 0, h), line([92, 60], [8, 60]), arc([8, 52], 8, h, 2 * h), line([0, 52], [0, 8]), arc([8, 8], 8, 2 * h, 3 * h)], lines: 4, arcs: 4 },
  rrectSmall: { segs: [line([3, 0], [97, 0]), arc([97, 3], 3, -h, 0), line([100, 3], [100, 57]), arc([97, 57], 3, 0, h), line([97, 60], [3, 60]), arc([3, 57], 3, h, 2 * h), line([0, 57], [0, 3]), arc([3, 3], 3, 2 * h, 3 * h)], lines: 4, arcs: 4 },
  slot: { segs: [line([0, 0], [50, 0]), arc([50, 10], 10, -h, h), line([50, 20], [0, 20]), arc([0, 10], 10, h, 3 * h)], lines: 2, arcs: 2 },
  lbr: { segs: [line([0, 0], [80, 0]), line([80, 0], [80, 10]), line([80, 10], [16, 10]), arc([16, 16], 6, -h, -2 * h), line([10, 16], [10, 50]), line([10, 50], [0, 50]), line([0, 50], [0, 0])], lines: 6, arcs: 1 },
  cam: { segs: [arc([0, 0], 20, -h, h), line([0, 20], [-30, 12]), arc([-30, 0], 12, h, 3 * h), line([-30, -12], [0, -20])], lines: 2, arcs: 2 },
};
it.skip("stress", () => {
  for (const [name, sh] of Object.entries(shapes)) {
    let ok = 0, total = 0; const fails: string[] = [];
    for (let seed = 1; seed <= 12; seed++) for (const step of [0.25, 0.5, 1.0]) {
      const rotA = seed * 0.37;
      const pts0 = sampleChain(sh.segs, step, 0.02, seed, rotA);
      const off = (seed * 37) % pts0.length;
      const pts = pts0.slice(off).concat(pts0.slice(0, off));
      const det = autoSegmentDetailed(pts, { tolerance: 0.1, closed: true });
      const nl = det.segments.filter((s) => s.type === "line").length, na = det.segments.filter((s) => s.type === "arc").length;
      total++;
      if (nl === sh.lines && na === sh.arcs && det.maxError < 0.15) ok++;
      else fails.push(`seed${seed}/step${step}: ${nl}L ${na}A err ${det.maxError.toFixed(3)}`);
    }
    console.log(name, ok, "/", total, fails.slice(0, 6).join("; "));
  }
});
it("detail", () => {
  const cases: [string, number, number][] = [["rrect", 6, 0.5], ["slot", 5, 0.25], ["cam", 2, 0.5], ["cam", 1, 1]];
  for (const [name, seed, step] of cases) {
    const sh = shapes[name];
    const rotA = seed * 0.37;
    const pts0 = sampleChain(sh.segs, step, 0.02, seed, rotA);
    const off = (seed * 37) % pts0.length;
    const pts = pts0.slice(off).concat(pts0.slice(0, off));
    const det = autoSegmentDetailed(pts, { tolerance: 0.1, closed: true });
    console.log("==", name, seed, step, "N", pts.length, "err", det.maxError.toFixed(3));
    det.segments.forEach((s, k) => {
      const g = s.type === "arc" ? segmentArcGeom(s)! : null;
      const len = s.type === "line" ? Math.hypot(s.b[0] - s.a[0], s.b[1] - s.a[1]) : g ? g.radius * Math.abs(g.sweep) : 0;
      console.log("  ", k, s.type, "range", det.ranges[k].join("-"), "len", len.toFixed(2), g ? "r " + g.radius.toFixed(3) + " sw " + (g.sweep * 180 / Math.PI).toFixed(1) : "", "tan", det.tangent[k]);
    });
  }
});
