/**
 * Profile extraction: turns the non-construction sketch geometry into closed loops (Profile2D) and
 * open wires for the CAD kernel, plus region nesting (outer boundary + holes).
 *
 * Topology rules
 *  - entities are split where an endpoint of another entity lies on them (T-junctions); interior
 *    crossings are NOT split, only reported in `crossings`;
 *  - dangling chains and bridges become open wires;
 *  - the remaining graph is traced into bounded faces (left-most turn walk) so that simple chains
 *    through degree-2 nodes give one loop each and branched graphs give their minimal faces;
 *  - circles and closed splines with nothing attached are loops by themselves.
 * Loops are ordered head-to-tail, counter-clockwise, arcs as 3-point segments.
 */
import type { Loop2D, Profile2D, Segment2D, Vec2, Wire2D } from "../core/types";
import {
  closestPointOnSegment,
  pointInPolygon,
  reverseSegment,
  sampleSegment,
  segmentAreaTerm,
  segmentBounds,
  segmentEnd,
  segmentIntersections,
  segmentIsClosed,
  segmentLength,
  segmentPointAt,
  segmentStart,
  subSegment,
} from "./geom2d";
import { entityToSegments, sketchBounds, type Sketch } from "./model";

export { sampleSegment } from "./geom2d";

export interface ProfileCrossing {
  point: Vec2;
  entities: [string, string];
}

export interface ProfileExtraction {
  profile: Profile2D;
  openWires: Wire2D[];
  /** source entity ids of each loop's segments (parallel to profile.loops) */
  loopEntities: string[][];
  /** source entity ids of each open wire's segments (parallel to openWires) */
  wireEntities: string[][];
  /** interior intersections that were not split (the profile may be invalid where they occur) */
  crossings: ProfileCrossing[];
  /** graph nodes where more than two loop edges meet */
  branchPoints: Vec2[];
}

interface Edge {
  id: number;
  u: number;
  v: number;
  seg: Segment2D;
  src: string;
  removed: boolean;
  open: boolean;
}

class NodeIndex {
  readonly pos: Vec2[] = [];
  private grid = new Map<string, number[]>();
  constructor(private tol: number) {}
  private key(x: number, y: number) {
    return `${Math.floor(x / (this.tol * 4))},${Math.floor(y / (this.tol * 4))}`;
  }
  find(p: Vec2): number {
    const cs = this.tol * 4;
    const gx = Math.floor(p[0] / cs), gy = Math.floor(p[1] / cs);
    let best = -1, bd = Infinity;
    for (let dx = -1; dx <= 1; dx++)
      for (let dy = -1; dy <= 1; dy++) {
        const l = this.grid.get(`${gx + dx},${gy + dy}`);
        if (!l) continue;
        for (const i of l) {
          const d = Math.hypot(this.pos[i][0] - p[0], this.pos[i][1] - p[1]);
          if (d <= this.tol && d < bd) {
            bd = d;
            best = i;
          }
        }
      }
    return best;
  }
  get(p: Vec2): number {
    const f = this.find(p);
    if (f >= 0) return f;
    const id = this.pos.length;
    this.pos.push([p[0], p[1]]);
    const k = this.key(p[0], p[1]);
    let l = this.grid.get(k);
    if (!l) this.grid.set(k, (l = []));
    l.push(id);
    return id;
  }
}

/** Replace a segment's end points by exact node positions (keeps loops exactly head-to-tail). */
function snapEnds(seg: Segment2D, a: Vec2, b: Vec2): Segment2D {
  switch (seg.type) {
    case "line":
      return { type: "line", a, b };
    case "arc":
      return { type: "arc", start: a, mid: seg.mid, end: b };
    case "bspline": {
      const cp = seg.curve.controlPoints.map((q) => [q[0], q[1]] as Vec2);
      cp[0] = [a[0], a[1]];
      cp[cp.length - 1] = [b[0], b[1]];
      return { type: "bspline", curve: { ...seg.curve, controlPoints: cp } };
    }
    default:
      return seg;
  }
}

function boxesOverlap(a: { min: Vec2; max: Vec2 }, b: { min: Vec2; max: Vec2 }, tol: number): boolean {
  return a.min[0] <= b.max[0] + tol && b.min[0] <= a.max[0] + tol && a.min[1] <= b.max[1] + tol && b.min[1] <= a.max[1] + tol;
}

/** Extract closed loops and open wires from the non-construction geometry of a sketch. */
export function extractProfiles(sketch: Sketch, tol?: number): ProfileExtraction {
  const bounds = sketchBounds(sketch);
  const scale = bounds ? Math.max(1, Math.hypot(bounds.max[0] - bounds.min[0], bounds.max[1] - bounds.min[1]), ...bounds.max.map(Math.abs), ...bounds.min.map(Math.abs)) : 1;
  const TOL = tol ?? 1e-6 * scale;
  const idx = new Map(sketch.points.map((p) => [p.id, p]));
  const src: { id: string; seg: Segment2D; box: { min: Vec2; max: Vec2 } }[] = [];
  for (const e of sketch.entities) {
    if (e.construction) continue;
    let segs: Segment2D[];
    try {
      segs = entityToSegments(sketch, e, idx);
    } catch {
      continue;
    }
    for (const seg of segs) {
      if (seg.type !== "circle" && !segmentIsClosed(seg, TOL) && segmentLength(seg) <= TOL) continue; // degenerate
      if (seg.type === "circle" && seg.radius <= TOL) continue;
      src.push({ id: e.id, seg, box: segmentBounds(seg) });
    }
  }

  const nodes = new NodeIndex(TOL);
  // endpoints
  for (const s of src) {
    if (s.seg.type === "circle") continue;
    nodes.get(segmentStart(s.seg));
    if (!segmentIsClosed(s.seg, TOL)) nodes.get(segmentEnd(s.seg));
  }
  const endpointCount = nodes.pos.length;
  // T-junction split parameters
  const splits: number[][] = src.map(() => []);
  for (let n = 0; n < endpointCount; n++) {
    const p = nodes.pos[n];
    for (let k = 0; k < src.length; k++) {
      const s = src[k];
      if (p[0] < s.box.min[0] - TOL || p[0] > s.box.max[0] + TOL || p[1] < s.box.min[1] - TOL || p[1] > s.box.max[1] + TOL) continue;
      const c = closestPointOnSegment(s.seg, p);
      if (c.distance > TOL) continue;
      if (s.seg.type !== "circle") {
        const a = segmentStart(s.seg), b = segmentEnd(s.seg);
        if (Math.hypot(a[0] - p[0], a[1] - p[1]) <= TOL || Math.hypot(b[0] - p[0], b[1] - p[1]) <= TOL) continue;
      }
      splits[k].push(c.t);
    }
  }

  const edges: Edge[] = [];
  const loops: Loop2D[] = [];
  const loopEntities: string[][] = [];
  const addEdge = (seg: Segment2D, srcId: string) => {
    const a = segmentStart(seg), b = segmentEnd(seg);
    const u = nodes.get(a), v = nodes.get(b);
    edges.push({ id: edges.length, u, v, seg: snapEnds(seg, nodes.pos[u], nodes.pos[v]), src: srcId, removed: false, open: false });
  };
  for (let k = 0; k < src.length; k++) {
    const { seg, id } = src[k];
    const L = Math.max(segmentLength(seg), 1e-300);
    let ts = splits[k].sort((x, y) => x - y).filter((t, i, arr) => i === 0 || (t - arr[i - 1]) * L > TOL);
    if (seg.type === "circle") {
      if (!ts.length) {
        loops.push([seg]);
        loopEntities.push([id]);
        continue;
      }
      if (ts.length > 1 && (ts[0] + 1 - ts[ts.length - 1]) * L <= TOL) ts.pop();
      if (ts.length === 1) ts = [ts[0], (ts[0] + 0.5) % 1].sort((x, y) => x - y);
      for (let i = 0; i < ts.length; i++) {
        const t0 = ts[i], t1 = i + 1 < ts.length ? ts[i + 1] : ts[0] + 1;
        addEdge(subSegment(seg, t0, t1), id);
      }
      continue;
    }
    const cuts = [0, ...ts.filter((t) => t * L > TOL && (1 - t) * L > TOL), 1];
    for (let i = 0; i + 1 < cuts.length; i++) addEdge(cuts.length === 2 ? seg : subSegment(seg, cuts[i], cuts[i + 1]), id);
  }

  // incidence
  const nN = nodes.pos.length;
  const inc: number[][] = Array.from({ length: nN }, () => []);
  for (const e of edges) {
    inc[e.u].push(e.id);
    inc[e.v].push(e.id);
  }
  const degree = (n: number) => inc[n].filter((id) => !edges[id].removed).length;

  // prune dangling chains
  const prune = () => {
    const q: number[] = [];
    for (let n = 0; n < nN; n++) if (degree(n) === 1) q.push(n);
    while (q.length) {
      const n = q.pop()!;
      const live = inc[n].filter((id) => !edges[id].removed);
      if (live.length !== 1) continue;
      const e = edges[live[0]];
      e.removed = true;
      e.open = true;
      const o = e.u === n ? e.v : e.u;
      if (degree(o) === 1) q.push(o);
    }
  };
  prune();
  // bridges (Tarjan, iterative), removed into open wires
  {
    const disc = new Int32Array(nN).fill(-1);
    const low = new Int32Array(nN);
    let time = 0;
    for (let root = 0; root < nN; root++) {
      if (disc[root] >= 0 || degree(root) === 0) continue;
      const stack: { n: number; pe: number; i: number }[] = [{ n: root, pe: -1, i: 0 }];
      disc[root] = low[root] = time++;
      while (stack.length) {
        const fr = stack[stack.length - 1];
        const list = inc[fr.n];
        if (fr.i < list.length) {
          const eid = list[fr.i++];
          const e = edges[eid];
          if (e.removed || eid === fr.pe) continue;
          if (e.u === e.v) continue; // self loop
          const w = e.u === fr.n ? e.v : e.u;
          if (disc[w] < 0) {
            disc[w] = low[w] = time++;
            stack.push({ n: w, pe: eid, i: 0 });
          } else low[fr.n] = Math.min(low[fr.n], disc[w]);
        } else {
          stack.pop();
          if (fr.pe >= 0) {
            const parent = stack[stack.length - 1].n;
            low[parent] = Math.min(low[parent], low[fr.n]);
            if (low[fr.n] > disc[parent]) {
              edges[fr.pe].removed = true;
              edges[fr.pe].open = true;
            }
          }
        }
      }
    }
  }
  prune();

  // face tracing on the remaining edges
  type Half = { e: number; fwd: boolean; from: number; to: number; ang: number };
  const halves: Half[] = [];
  const out: number[][] = Array.from({ length: nN }, () => []);
  const delta = 1e-4;
  for (const e of edges) {
    if (e.removed) continue;
    for (const fwd of [true, false]) {
      const from = fwd ? e.u : e.v;
      const to = fwd ? e.v : e.u;
      const q = segmentPointAt(e.seg, fwd ? delta : 1 - delta);
      const p = nodes.pos[from];
      halves.push({ e: e.id, fwd, from, to, ang: Math.atan2(q[1] - p[1], q[0] - p[0]) });
      out[from].push(halves.length - 1);
    }
  }
  for (let n = 0; n < nN; n++) out[n].sort((a, b) => halves[a].ang - halves[b].ang);
  const posInNode = new Int32Array(halves.length);
  for (let n = 0; n < nN; n++) out[n].forEach((h, i) => (posInNode[h] = i));
  const twin = (h: number) => (halves[h].fwd ? h + 1 : h - 1);
  const used = new Uint8Array(halves.length);
  const branchPoints: Vec2[] = [];
  for (let n = 0; n < nN; n++) if (out[n].length > 2) branchPoints.push(nodes.pos[n]);
  for (let h0 = 0; h0 < halves.length; h0++) {
    if (used[h0]) continue;
    const face: number[] = [];
    let h = h0;
    let guard = 0;
    while (!used[h] && guard++ <= halves.length) {
      used[h] = 1;
      face.push(h);
      const t = twin(h);
      const at = halves[t].from;
      const list = out[at];
      const i = posInNode[t];
      h = list[(i - 1 + list.length) % list.length];
    }
    if (h !== h0) continue; // malformed walk
    const segs: Segment2D[] = face.map((hh) => {
      const e = edges[halves[hh].e];
      return halves[hh].fwd ? e.seg : reverseSegment(e.seg);
    });
    let area = 0;
    for (const s of segs) area += segmentAreaTerm(s);
    if (area > TOL * TOL) {
      loops.push(segs);
      loopEntities.push(face.map((hh) => edges[halves[hh].e].src));
    }
  }
  // orient standalone loops CCW (circles have no orientation)
  for (let i = 0; i < loops.length; i++) {
    const lp = loops[i];
    if (lp.length === 1 && lp[0].type === "bspline" && loopSignedArea(lp) < 0) loops[i] = reverseLoop(lp);
  }

  // open wires: maximal chains of open edges through nodes of open-degree 2
  const openEdges = edges.filter((e) => e.open);
  const oinc: number[][] = Array.from({ length: nN }, () => []);
  for (const e of openEdges) {
    oinc[e.u].push(e.id);
    oinc[e.v].push(e.id);
  }
  const seen = new Set<number>();
  const openWires: Wire2D[] = [];
  const wireEntities: string[][] = [];
  const walk = (startNode: number, firstEdge: number) => {
    const segs: Segment2D[] = [];
    const ids: string[] = [];
    let n = startNode;
    let eid = firstEdge;
    while (eid >= 0 && !seen.has(eid)) {
      seen.add(eid);
      const e = edges[eid];
      const fwd = e.u === n;
      segs.push(fwd ? e.seg : reverseSegment(e.seg));
      ids.push(e.src);
      n = fwd ? e.v : e.u;
      if (oinc[n].length !== 2) break;
      eid = oinc[n].find((x) => !seen.has(x)) ?? -1;
    }
    const closed = segs.length > 0 && n === startNode;
    openWires.push({ segments: segs, closed });
    wireEntities.push(ids);
  };
  for (let n = 0; n < nN; n++) if (oinc[n].length !== 2) for (const eid of oinc[n]) if (!seen.has(eid)) walk(n, eid);
  for (const e of openEdges) if (!seen.has(e.id)) walk(e.u, e.id);

  // crossings that were not split
  const crossings: ProfileCrossing[] = [];
  for (let i = 0; i < src.length; i++)
    for (let j = i + 1; j < src.length; j++) {
      if (!boxesOverlap(src[i].box, src[j].box, TOL)) continue;
      for (const h of segmentIntersections(src[i].seg, src[j].seg, TOL)) {
        if (nodes.find(h.point) >= 0) continue;
        crossings.push({ point: h.point, entities: [src[i].id, src[j].id] });
      }
    }

  return { profile: { loops }, openWires, loopEntities, wireEntities, crossings, branchPoints };
}

// ---------------------------------------------------------------------------
// Loop utilities
// ---------------------------------------------------------------------------

/** Exact signed area of a closed loop (CCW positive). */
export function loopSignedArea(loop: Loop2D): number {
  let a = 0;
  for (const s of loop) a += segmentAreaTerm(s);
  if (loop.length === 1 && loop[0].type === "circle") return Math.PI * loop[0].radius ** 2;
  return a;
}

export function reverseLoop(loop: Loop2D): Loop2D {
  return loop
    .slice()
    .reverse()
    .map((s) => reverseSegment(s));
}

/** Polygon approximation of a loop (no repeated closing vertex). */
export function sampleLoop(loop: Loop2D, opts?: number | { chordTol?: number; maxSegments?: number }): Vec2[] {
  const poly: Vec2[] = [];
  for (const s of loop) {
    const pts = sampleSegment(s, opts);
    for (let i = 0; i < pts.length - 1; i++) poly.push(pts[i]);
    if (loop.length === 1 && s.type !== "circle" && !segmentIsClosed(s)) poly.push(pts[pts.length - 1]);
  }
  return poly;
}

/** Even-odd containment test of p in the (sampled) loop. */
export function loopContainsPoint(loop: Loop2D, p: Vec2, opts?: number | { chordTol?: number; maxSegments?: number }): boolean {
  if (loop.length === 1 && loop[0].type === "circle") {
    const c = loop[0];
    return Math.hypot(p[0] - c.center[0], p[1] - c.center[1]) < c.radius;
  }
  return pointInPolygon(sampleLoop(loop, opts), p);
}

export interface Region {
  /** index of the outer loop in profile.loops */
  outer: number;
  /** indices of the hole loops */
  holes: number[];
  /** outer loop oriented counter-clockwise */
  outerLoop: Loop2D;
  /** hole loops oriented clockwise */
  holeLoops: Loop2D[];
}

/** A point strictly inside the region bounded by the loop (a little to the left of a long segment, for CCW). */
function interiorProbe(loop: Loop2D, poly: Vec2[], area: number): Vec2 {
  if (loop.length === 1 && loop[0].type === "circle") return loop[0].center;
  const sign = area >= 0 ? 1 : -1;
  let best: Vec2 | null = null, bestScore = -Infinity;
  const ext = Math.sqrt(Math.abs(area)) || 1;
  for (const s of loop) {
    const L = segmentLength(s);
    for (const t of [0.5, 0.31, 0.69]) {
      const p = segmentPointAt(s, t);
      const q = segmentPointAt(s, Math.min(1, t + 1e-4));
      const r = segmentPointAt(s, Math.max(0, t - 1e-4));
      const d: Vec2 = [q[0] - r[0], q[1] - r[1]];
      const dl = Math.hypot(d[0], d[1]) || 1;
      const eps = Math.min(L, ext) * 1e-3;
      const probe: Vec2 = [p[0] - (d[1] / dl) * eps * sign, p[1] + (d[0] / dl) * eps * sign];
      if (pointInPolygon(poly, probe) && L > bestScore) {
        bestScore = L;
        best = probe;
      }
    }
  }
  return best ?? poly[0];
}

/**
 * Nest loops by containment: even-depth loops are outer boundaries, odd-depth loops are holes of their
 * immediate parent. Islands inside holes form new regions.
 */
export function regionsFromLoops(profile: Profile2D): Region[] {
  const loops = profile.loops;
  const n = loops.length;
  const polys = loops.map((l) => sampleLoop(l));
  const areas = loops.map((l) => loopSignedArea(l));
  const absA = areas.map(Math.abs);
  const probes = loops.map((l, i) => interiorProbe(l, polys[i], areas[i]));
  const parent = new Array<number>(n).fill(-1);
  const depth = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i++) {
    let best = -1;
    for (let j = 0; j < n; j++) {
      if (i === j || absA[j] <= absA[i]) continue;
      if (!loopContainsPoint(loops[j], probes[i])) continue;
      // also require a boundary point of i to be inside j (adjacent faces share edges, so use the probe primarily)
      depth[i]++;
      if (best < 0 || absA[j] < absA[best]) best = j;
    }
    parent[i] = best;
  }
  const regions: Region[] = [];
  const regionOf = new Map<number, Region>();
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => depth[a] - depth[b]);
  for (const i of order) {
    if (depth[i] % 2 === 0) {
      const r: Region = { outer: i, holes: [], outerLoop: areas[i] >= 0 ? loops[i] : reverseLoop(loops[i]), holeLoops: [] };
      regions.push(r);
      regionOf.set(i, r);
    }
  }
  for (const i of order) {
    if (depth[i] % 2 === 1 && parent[i] >= 0) {
      const r = regionOf.get(parent[i]);
      if (!r) continue;
      r.holes.push(i);
      const lp = loops[i];
      r.holeLoops.push(lp.length === 1 && lp[0].type === "circle" ? lp : areas[i] <= 0 ? lp : reverseLoop(lp));
    }
  }
  return regions;
}
