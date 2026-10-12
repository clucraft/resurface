import { it } from "vitest";
import type { Segment2D, Vec2 } from "../core/types";
import { closestPointOnSegment, segmentArcGeom } from "./geom2d";
import { autoSegmentDetailed } from "./fit2d";
function rng(seed: number) {
  let s = seed >>> 0;
  const u = () => { s = (s * 1664525 + 1013904223) >>> 0; return (s + 0.5) / 4294967296; };
  return { u, g: () => Math.sqrt(-2 * Math.log(u())) * Math.cos(2 * Math.PI * u()) };
}
function sampleChain(segs: Segment2D[], step: number, sigma: number, seed = 1): Vec2[] {
  const r = rng(seed); const out: Vec2[] = [];
  for (const s of segs) {
    let L: number; let at: (t: number) => Vec2;
    if (s.type === "line") { L = Math.hypot(s.b[0] - s.a[0], s.b[1] - s.a[1]); at = (t) => [s.a[0] + (s.b[0] - s.a[0]) * t, s.a[1] + (s.b[1] - s.a[1]) * t]; }
    else { const g = segmentArcGeom(s as any)!; L = g.radius * Math.abs(g.sweep); at = (t) => [g.center[0] + g.radius * Math.cos(g.a0 + g.sweep * t), g.center[1] + g.radius * Math.sin(g.a0 + g.sweep * t)]; }
    const n = Math.max(1, Math.round(L / step));
    for (let i = 0; i < n; i++) { const p = at(i / n); out.push([p[0] + sigma * r.g(), p[1] + sigma * r.g()]); }
  }
  return out;
}
const arc = (c: Vec2, r: number, a0: number, a1: number): Segment2D => { const am = (a0 + a1) / 2; return { type: "arc", start: [c[0] + r * Math.cos(a0), c[1] + r * Math.sin(a0)], mid: [c[0] + r * Math.cos(am), c[1] + r * Math.sin(am)], end: [c[0] + r * Math.cos(a1), c[1] + r * Math.sin(a1)] }; };
const line = (a: Vec2, b: Vec2): Segment2D => ({ type: "line", a, b });
function roundedRect(W: number, H: number, R: number): Segment2D[] {
  const h = Math.PI / 2;
  return [line([R, 0], [W - R, 0]), arc([W - R, R], R, -h, 0), line([W, R], [W, H - R]), arc([W - R, H - R], R, 0, h), line([W - R, H], [R, H]), arc([R, H - R], R, h, 2 * h), line([0, H - R], [0, R]), arc([R, R], R, 2 * h, 3 * h)];
}
it("dbg", () => {
  const pts = sampleChain(roundedRect(100, 60, 8), 0.5, 0.02, 11);
  const rot = pts.slice(40).concat(pts.slice(0, 40));
  const det = autoSegmentDetailed(rot, { tolerance: 0.1, closed: true });
  console.log("N", rot.length, "maxErr", det.maxError);
  det.segments.forEach((s, k) => {
    const [a, b] = det.ranges[k];
    let mx = 0, at = -1;
    const N = rot.length;
    for (let i = a; ; i = (i + 1) % N) { const d = closestPointOnSegment(s, rot[i]).distance; if (d > mx) { mx = d; at = i; } if (i === b) break; }
    console.log(k, s.type, JSON.stringify(s).slice(0, 200), "range", a, b, "err", mx.toFixed(4), "at", at, det.tangent[k]);
  });
});
