/**
 * Turns a segmented mesh into a B-rep "plan": faces (one per region, one per faceted triangle),
 * their boundary loops, the edges shared between faces with an analytic curve each (line, arc,
 * circle or interpolating spline) and the vertices where three or more faces meet.
 *
 * Plain data only — the kernel builds the OCCT shape from it (OcctKernel.brepFromPlan).
 */
import type { MeshData, Vec3 } from "../core/types";
import { cross, dot, normalize } from "../core/vec";
import { eigenSym3 } from "../core/linalg";
import { perpBasis } from "../fit/common";
import { kasaCircle, refineCircle } from "../fit/fit2d";
import { surfNormal, type ConvSurface, type SurfKind } from "./surfaces";
import type { Region } from "./segment";

export type FaceKind = SurfKind | "faceted";

export type PlanCurve =
  | { type: "line" }
  /** open arc from v0 through `mid` to v1 */
  | { type: "arc"; mid: Vec3 }
  /** closed full circle through v0 (v0 === v1) */
  | { type: "circle"; center: Vec3; normal: Vec3; radius: number }
  /** interpolating spline through `points` (open: includes both end vertices; closed: v0 first, no repeat) */
  | { type: "spline"; points: Vec3[]; closed: boolean }
  /** straight segments through `points` (both end vertices included): short chains with no clear curve */
  | { type: "polyline"; points: Vec3[] };

export interface PlanEdge {
  v0: number;
  v1: number;
  curve: PlanCurve;
}

export interface EdgeUse {
  edge: number;
  reversed: boolean;
}

/** Placement of the analytic surface: origin, main axis and the seam direction (u = 0). */
export interface SurfacePlacement {
  origin: Vec3;
  z: Vec3;
  x: Vec3;
}

export interface PlanFace {
  kind: FaceKind;
  /** null for faceted faces (the plane of their triangle) */
  surface: ConvSurface | null;
  placement: SurfacePlacement | null;
  /** surface normal agrees with the outward mesh normal */
  forward: boolean;
  loops: EdgeUse[][];
  /** region id, or -1 for faceted triangles */
  region: number;
  /** area of the mesh triangles the face replaces (regions only): a sanity check for the built face */
  area?: number;
  /**
   * Faces bounded only by full circles about the surface axis (a hole wall, a fillet ring): two
   * circles bound two complementary bands on a periodic surface, so these are built directly as
   * the parametric patch u ∈ [0, 2π], v ∈ [v0, v1] (v in OCCT's parametrisation of the surface).
   */
  band?: { v0: number; v1: number };
}

export interface BrepPlan {
  vertices: Vec3[];
  edges: PlanEdge[];
  faces: PlanFace[];
  tolerance: number;
}

export interface PlanOptions {
  tolerance: number;
  size: number;
}

const getP = (P: ArrayLike<number>, v: number): Vec3 => [P[3 * v], P[3 * v + 1], P[3 * v + 2]];

/** Common rotation axis of two surfaces (their intersection is then a set of circles about it). */
function commonAxis(a: ConvSurface | null, b: ConvSurface | null, tol: number): { point: Vec3; dir: Vec3 } | null {
  if (!a || !b) return null;
  const axisOf = (s: ConvSurface): { point: Vec3; dir: Vec3 } | null => {
    switch (s.kind) {
      case "cylinder":
        return { point: s.origin, dir: normalize(s.axis) };
      case "cone":
        return { point: s.apex, dir: normalize(s.axis) };
      case "torus":
        return { point: s.center, dir: normalize(s.axis) };
      default:
        return null;
    }
  };
  const ax = axisOf(a) ?? axisOf(b);
  const other = axisOf(a) ? b : a;
  const first = axisOf(a) ? a : b;
  if (ax) {
    const d = ax.dir;
    const onAxis = (p: Vec3) => {
      const w: Vec3 = [p[0] - ax.point[0], p[1] - ax.point[1], p[2] - ax.point[2]];
      const h = dot(w, d);
      return Math.hypot(w[0] - h * d[0], w[1] - h * d[1], w[2] - h * d[2]) <= 2 * tol;
    };
    const ang = (u: Vec3) => Math.abs(dot(normalize(u), d)) >= Math.cos(0.2 * (Math.PI / 180));
    switch (other.kind) {
      case "plane":
        return ang(other.normal) ? ax : null;
      case "sphere":
        return onAxis(other.center) ? ax : null;
      case "cylinder":
      case "cone":
      case "torus": {
        if (other === first) return ax;
        const o = axisOf(other)!;
        return ang(o.dir) && onAxis(o.point) ? ax : null;
      }
    }
    return null;
  }
  // no axial surface: plane–sphere or sphere–sphere
  if (a.kind === "sphere" && b.kind === "sphere") {
    const d: Vec3 = [b.center[0] - a.center[0], b.center[1] - a.center[1], b.center[2] - a.center[2]];
    if (Math.hypot(...d) < tol) return null;
    return { point: a.center, dir: normalize(d) };
  }
  if (a.kind === "sphere" && b.kind === "plane") return { point: a.center, dir: normalize(b.normal) };
  if (b.kind === "sphere" && a.kind === "plane") return { point: b.center, dir: normalize(a.normal) };
  return null;
}

/** Circle about `axis` through the points (mean height/radius), when every point fits it. */
function axialCircle(pts: Vec3[], axis: { point: Vec3; dir: Vec3 }, tol: number): { center: Vec3; normal: Vec3; radius: number } | null {
  const d = axis.dir;
  let sh = 0, sr = 0;
  const hr = pts.map((p) => {
    const w: Vec3 = [p[0] - axis.point[0], p[1] - axis.point[1], p[2] - axis.point[2]];
    const h = dot(w, d);
    const r = Math.hypot(w[0] - h * d[0], w[1] - h * d[1], w[2] - h * d[2]);
    sh += h;
    sr += r;
    return [h, r];
  });
  const h0 = sh / pts.length, r0 = sr / pts.length;
  if (!(r0 > 2 * tol)) return null;
  for (const [h, r] of hr) if (Math.hypot(h - h0, r - r0) > 1.5 * tol) return null;
  return { center: [axis.point[0] + h0 * d[0], axis.point[1] + h0 * d[1], axis.point[2] + h0 * d[2]], normal: d, radius: r0 };
}

/** Least-squares circle in 3D (plane fit + 2D circle). */
function fitCircle3(pts: Vec3[], tol: number, size: number): { center: Vec3; normal: Vec3; radius: number } | null {
  // three points always lie on some circle: it takes a fourth to show the edge really is one
  if (pts.length < 4) return null;
  const c: Vec3 = [0, 0, 0];
  for (const p of pts) for (let k = 0; k < 3; k++) c[k] += p[k] / pts.length;
  const m = new Float64Array(9);
  for (const p of pts) {
    const q = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) m[3 * a + b] += q[a] * q[b];
  }
  const { vectors } = eigenSym3(m);
  const n = normalize(vectors[0]);
  for (const p of pts) if (Math.abs(dot([p[0] - c[0], p[1] - c[1], p[2] - c[2]], n)) > tol) return null;
  const [u, v] = perpBasis(n);
  const xy = new Float64Array(2 * pts.length);
  pts.forEach((p, i) => {
    const q: Vec3 = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
    xy[2 * i] = dot(q, u);
    xy[2 * i + 1] = dot(q, v);
  });
  const k0 = kasaCircle(xy, pts.length);
  if (!k0 || !Number.isFinite(k0.r)) return null;
  const k = refineCircle(xy, pts.length, k0, { maxIterations: 20 });
  if (!(k.r > 2 * tol && k.r < 10 * size)) return null;
  const center: Vec3 = [c[0] + k.cx * u[0] + k.cy * v[0], c[1] + k.cx * u[1] + k.cy * v[1], c[2] + k.cx * u[2] + k.cy * v[2]];
  for (let i = 0; i < pts.length; i++) if (Math.abs(Math.hypot(xy[2 * i] - k.cx, xy[2 * i + 1] - k.cy) - k.r) > tol) return null;
  return { center, normal: n, radius: k.r };
}

function pointOnCircle(circ: { center: Vec3; normal: Vec3; radius: number }, p: Vec3): Vec3 {
  const n = circ.normal;
  const w: Vec3 = [p[0] - circ.center[0], p[1] - circ.center[1], p[2] - circ.center[2]];
  const h = dot(w, n);
  let r: Vec3 = [w[0] - h * n[0], w[1] - h * n[1], w[2] - h * n[2]];
  const l = Math.hypot(...r);
  r = l > 1e-300 ? [r[0] / l, r[1] / l, r[2] / l] : perpBasis(n)[0];
  return [circ.center[0] + circ.radius * r[0], circ.center[1] + circ.radius * r[1], circ.center[2] + circ.radius * r[2]];
}

/** Point of the circle arc between a and b on the side of `through` (or the minor arc). */
function arcMid(circ: { center: Vec3; normal: Vec3; radius: number }, a: Vec3, b: Vec3, through: Vec3 | null): Vec3 {
  const c = circ.center, n = circ.normal;
  const rel = (p: Vec3): Vec3 => [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
  const [u, v] = perpBasis(n);
  const angle = (p: Vec3) => Math.atan2(dot(rel(p), v), dot(rel(p), u));
  const ta = angle(a);
  const norm = (x: number) => ((x % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
  // counter-clockwise sweep from a to b
  const sweep = norm(angle(b) - ta);
  let mid: number;
  if (through) {
    const tt = norm(angle(through) - ta);
    mid = tt <= sweep ? ta + sweep / 2 : ta - (2 * Math.PI - sweep) / 2;
  } else mid = sweep <= Math.PI ? ta + sweep / 2 : ta - (2 * Math.PI - sweep) / 2;
  return [
    c[0] + circ.radius * (Math.cos(mid) * u[0] + Math.sin(mid) * v[0]),
    c[1] + circ.radius * (Math.cos(mid) * u[1] + Math.sin(mid) * v[1]),
    c[2] + circ.radius * (Math.cos(mid) * u[2] + Math.sin(mid) * v[2]),
  ];
}

function linePoints(pts: Vec3[], tol: number): boolean {
  const a = pts[0], b = pts[pts.length - 1];
  const d: Vec3 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const L = Math.hypot(...d);
  if (!(L > 0)) return false;
  for (let i = 1; i < pts.length - 1; i++) {
    const w: Vec3 = [pts[i][0] - a[0], pts[i][1] - a[1], pts[i][2] - a[2]];
    const x = cross(w, d);
    if (Math.hypot(...x) / L > tol) return false;
    const t = dot(w, d) / (L * L);
    if (t < -1e-9 || t > 1 + 1e-9) return false;
  }
  return true;
}

/** Seam direction ⟂ axis in the middle of the largest angular gap of the points. */
function seamDirection(pts: Iterable<Vec3>, origin: Vec3, axis: Vec3): Vec3 {
  const [u, v] = perpBasis(axis);
  const angs: number[] = [];
  for (const p of pts) {
    const w: Vec3 = [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]];
    const x = dot(w, u), y = dot(w, v);
    if (Math.hypot(x, y) > 1e-12) angs.push(Math.atan2(y, x));
  }
  if (angs.length < 2) return u;
  angs.sort((a, b) => a - b);
  let best = angs[0] + 2 * Math.PI - angs[angs.length - 1];
  let at = angs[angs.length - 1] + best / 2;
  for (let i = 1; i < angs.length; i++) {
    const g = angs[i] - angs[i - 1];
    if (g > best) {
      best = g;
      at = angs[i - 1] + g / 2;
    }
  }
  return normalize([Math.cos(at) * u[0] + Math.sin(at) * v[0], Math.cos(at) * u[1] + Math.sin(at) * v[1], Math.cos(at) * u[2] + Math.sin(at) * v[2]]);
}

function placementFor(s: ConvSurface, pts: Vec3[]): SurfacePlacement {
  switch (s.kind) {
    case "plane": {
      const z = normalize(s.normal);
      return { origin: s.origin, z, x: perpBasis(z)[0] };
    }
    case "cylinder": {
      const z = normalize(s.axis);
      return { origin: s.origin, z, x: seamDirection(pts, s.origin, z) };
    }
    case "cone": {
      const z = normalize(s.axis);
      return { origin: s.apex, z, x: seamDirection(pts, s.apex, z) };
    }
    case "torus": {
      const z = normalize(s.axis);
      return { origin: s.center, z, x: seamDirection(pts, s.center, z) };
    }
    case "sphere": {
      // keep the poles away from the face: of a few candidate pole axes (⟂ to the mean direction
      // for a patch, the principal axes for a zone around an axis), the one the face stays
      // farthest from — a pole inside the face makes the kernel close it into a cap
      const dirs: Vec3[] = [];
      const m: Vec3 = [0, 0, 0];
      const cov = new Array<number>(9).fill(0);
      for (const p of pts) {
        const w: Vec3 = [p[0] - s.center[0], p[1] - s.center[1], p[2] - s.center[2]];
        if (Math.hypot(...w) < 1e-12) continue;
        const d = normalize(w);
        dirs.push(d);
        for (let i = 0; i < 3; i++) {
          m[i] += d[i];
          for (let j = 0; j < 3; j++) cov[3 * i + j] += d[i] * d[j];
        }
      }
      const md = Math.hypot(...m) > 1e-12 ? normalize(m) : ([1, 0, 0] as Vec3);
      const candidates: Vec3[] = [...perpBasis(md), ...eigenSym3(cov).vectors];
      let z = candidates[0], best = Infinity;
      for (const c of candidates) {
        let worst = 0;
        for (const d of dirs) worst = Math.max(worst, Math.abs(dot(d, c)));
        if (worst < best - 1e-9) {
          best = worst;
          z = c;
        }
      }
      return { origin: s.center, z, x: seamDirection(pts, s.center, z) };
    }
  }
}

/** OCCT v parameter of p on the surface placed at `pl`. */
function vParam(s: ConvSurface, pl: SurfacePlacement, p: Vec3): number {
  const w: Vec3 = [p[0] - pl.origin[0], p[1] - pl.origin[1], p[2] - pl.origin[2]];
  const h = dot(w, pl.z);
  switch (s.kind) {
    case "cylinder":
      return h;
    case "cone":
      return h / Math.cos(s.halfAngle);
    case "sphere":
      return Math.asin(Math.max(-1, Math.min(1, h / s.radius)));
    case "torus": {
      const rho = Math.hypot(w[0] - h * pl.z[0], w[1] - h * pl.z[1], w[2] - h * pl.z[2]);
      return Math.atan2(h, rho - s.major);
    }
    default:
      return 0;
  }
}

function bandOf(
  s: ConvSurface,
  loops: EdgeUse[][],
  edges: PlanEdge[],
  vertices: Vec3[],
  pts: Vec3[],
  tol: number,
): { placement: SurfacePlacement; v0: number; v1: number } | null {
  if (s.kind === "plane" || !loops.length || !pts.length) return null;
  const circles: { center: Vec3; normal: Vec3; radius: number; v: number }[] = [];
  for (const l of loops) {
    if (l.length !== 1) return null;
    const e = edges[l[0].edge];
    if (e.curve.type !== "circle") return null;
    circles.push({ ...e.curve, v: e.v0 });
  }
  let origin: Vec3, z: Vec3;
  switch (s.kind) {
    case "cylinder":
      origin = s.origin;
      z = normalize(s.axis);
      break;
    case "cone":
      origin = s.apex;
      z = normalize(s.axis);
      break;
    case "torus":
      origin = s.center;
      z = normalize(s.axis);
      break;
    case "sphere":
      origin = s.center;
      z = normalize(circles[0].normal);
      break;
  }
  for (const c of circles) if (Math.abs(dot(normalize(c.normal), z)) < Math.cos(0.5 * (Math.PI / 180))) return null;
  // seam (u = 0) through the first circle's vertex so sewing finds matching closed edges
  const pv = vertices[circles[0].v];
  const w: Vec3 = [pv[0] - origin[0], pv[1] - origin[1], pv[2] - origin[2]];
  const hh = dot(w, z);
  let x: Vec3 = [w[0] - hh * z[0], w[1] - hh * z[1], w[2] - hh * z[2]];
  x = Math.hypot(...x) > 1e-12 ? normalize(x) : perpBasis(z)[0];
  const placement: SurfacePlacement = { origin, z, x };
  const vs = pts.map((p) => vParam(s, placement, p));
  let v0: number, v1: number;
  if (s.kind === "torus") {
    // periodic: the band is the complement of the largest gap
    vs.sort((a, b) => a - b);
    let gap = vs[0] + 2 * Math.PI - vs[vs.length - 1];
    v0 = vs[0];
    v1 = vs[vs.length - 1];
    for (let i = 1; i < vs.length; i++) {
      const g = vs[i] - vs[i - 1];
      if (g > gap) {
        gap = g;
        v0 = vs[i];
        v1 = vs[i - 1] + 2 * Math.PI;
      }
    }
  } else {
    v0 = Math.min(...vs);
    v1 = Math.max(...vs);
  }
  // close the band at a pole / apex the region reaches (float noise leaves a tiny hole otherwise)
  if (s.kind === "sphere") {
    if (s.radius * Math.cos(v1) < tol) v1 = Math.PI / 2;
    if (s.radius * Math.cos(v0) < tol) v0 = -Math.PI / 2;
  } else if (s.kind === "cone" && Math.abs(v0) * Math.sin(s.halfAngle) < tol) v0 = 0;
  if (!(v1 - v0 > 1e-9)) return null;
  return { placement, v0, v1 };
}

export function buildBrepPlan(mesh: MeshData, nbr: Int32Array, labels: Int32Array, regions: Region[], opts: PlanOptions): BrepPlan {
  const P = mesh.positions, I = mesh.indices;
  const nt = I.length / 3;
  const nv = P.length / 3;
  const tol = opts.tolerance;
  // faces: regions first, then one per faceted triangle
  const faceOf = new Int32Array(nt);
  let nf = regions.length;
  const facetTri: number[] = [];
  for (let t = 0; t < nt; t++) {
    if (labels[t] >= 0) faceOf[t] = labels[t];
    else {
      faceOf[t] = nf++;
      facetTri.push(t);
    }
  }
  const surfOf = (f: number): ConvSurface | null => (f >= 0 && f < regions.length ? regions[f].surface : null);
  const isBoundary = (he: number): boolean => {
    const u = nbr[he];
    return u < 0 || faceOf[u] !== faceOf[(he / 3) | 0];
  };
  const next = (he: number) => (he % 3 === 2 ? he - 2 : he + 1);
  const src = (he: number) => I[he];
  const dst = (he: number) => I[next(he)];
  /** half-edge of triangle u going a → b */
  const findHe = (u: number, a: number, b: number): number => {
    for (let j = 0; j < 3; j++) if (I[3 * u + j] === a && I[3 * u + ((j + 1) % 3)] === b) return 3 * u + j;
    return -1;
  };
  const twin = (he: number): number => {
    const u = nbr[he];
    return u < 0 ? -1 : findHe(u, dst(he), src(he));
  };

  // --- junction vertices
  const junction = new Uint8Array(nv);
  {
    // distinct faces (incl. the open boundary as a pseudo face) and outgoing boundary edges per vertex
    const faceA = new Int32Array(nv).fill(-3), faceB = new Int32Array(nv).fill(-3);
    const many = new Uint8Array(nv);
    const add = (v: number, f: number) => {
      if (faceA[v] === f || faceB[v] === f) return;
      if (faceA[v] === -3) faceA[v] = f;
      else if (faceB[v] === -3) faceB[v] = f;
      else many[v] = 1;
    };
    const outCount = new Map<number, number>();
    for (let t = 0; t < nt; t++)
      for (let k = 0; k < 3; k++) {
        const he = 3 * t + k;
        const v = I[he];
        add(v, faceOf[t]);
        if (nbr[he] < 0) {
          add(v, -1);
          add(dst(he), -1);
        }
        if (isBoundary(he)) {
          const key = v * 1 + faceOf[t] * nv;
          const c = (outCount.get(key) ?? 0) + 1;
          outCount.set(key, c);
          if (c > 1) junction[v] = 1;
        }
      }
    for (let v = 0; v < nv; v++) if (many[v]) junction[v] = 1;
    for (const t of facetTri) for (let k = 0; k < 3; k++) junction[I[3 * t + k]] = 1;
  }

  /** next boundary half-edge of the same face, rotating around the end vertex */
  const nextBoundary = (he: number): number => {
    let h = next(he);
    for (let guard = 0; guard < 10000; guard++) {
      if (isBoundary(h)) return h;
      const tw = twin(h);
      if (tw < 0) return -1;
      h = next(tw);
    }
    return -1;
  };

  const vertices: Vec3[] = [];
  const vertexId = new Int32Array(nv).fill(-1);
  const vid = (v: number): number => {
    if (vertexId[v] < 0) {
      vertexId[v] = vertices.length;
      vertices.push(getP(P, v));
    }
    return vertexId[v];
  };
  const edges: PlanEdge[] = [];
  /** half-edge → edge id (+1, negative = reversed use) */
  const heEdge = new Int32Array(3 * nt);
  const faces: PlanFace[] = [];
  const faceLoops: EdgeUse[][][] = Array.from({ length: nf }, () => []);
  const visited = new Uint8Array(3 * nt);

  const makeEdge = (chain: number[], f: number, closedIn: boolean): number => {
    const vs = [src(chain[0]), ...chain.map(dst)];
    // a loop with a single junction is a closed edge through that vertex
    const closed = closedIn || (vs.length > 3 && vs[0] === vs[vs.length - 1]);
    const pts = vs.map((v) => getP(P, v));
    const g = (() => {
      const tw = twin(chain[0]);
      return tw < 0 ? -1 : faceOf[(tw / 3) | 0];
    })();
    const sa = surfOf(f), sb = surfOf(g);
    const facet = f >= regions.length || g >= regions.length;
    let curve: PlanCurve;
    const e = edges.length;
    if (closed) {
      const ax = commonAxis(sa, sb, tol);
      const circ = (ax && axialCircle(pts, ax, tol)) || fitCircle3(pts, tol, opts.size);
      const v0 = vid(vs[0]);
      if (circ) {
        // the seam vertex of a closed edge is private to it: move it onto the circle
        vertices[v0] = pointOnCircle(circ, vertices[v0]);
        // orient the circle with the traversal direction
        const t1 = pts[1], t0 = pts[0];
        const dirv = cross([t0[0] - circ.center[0], t0[1] - circ.center[1], t0[2] - circ.center[2]], [t1[0] - t0[0], t1[1] - t0[1], t1[2] - t0[2]]);
        const normal = dot(dirv, circ.normal) >= 0 ? circ.normal : ([-circ.normal[0], -circ.normal[1], -circ.normal[2]] as Vec3);
        curve = { type: "circle", center: circ.center, normal, radius: circ.radius };
      } else curve = { type: "spline", points: pts.slice(0, -1), closed: true };
      edges.push({ v0, v1: v0, curve });
    } else {
      const v0 = vid(vs[0]), v1 = vid(vs[vs.length - 1]);
      if (facet || pts.length === 2) {
        const ax = !facet ? commonAxis(sa, sb, tol) : null;
        const circ = ax && axialCircle(pts, ax, tol);
        curve = circ ? { type: "arc", mid: arcMid(circ, pts[0], pts[1], null) } : { type: "line" };
      } else {
        const ax = commonAxis(sa, sb, tol);
        const circ = ax && axialCircle(pts, ax, tol);
        if (circ) curve = { type: "arc", mid: arcMid(circ, pts[0], pts[pts.length - 1], pts[Math.floor(pts.length / 2)]) };
        else if (linePoints(pts, tol)) curve = { type: "line" };
        else {
          const c2 = fitCircle3(pts, tol, opts.size);
          curve = c2
            ? { type: "arc", mid: arcMid(c2, pts[0], pts[pts.length - 1], pts[Math.floor(pts.length / 2)]) }
            : pts.length <= 5
              ? { type: "polyline", points: pts }
              : { type: "spline", points: pts, closed: false };
        }
      }
      edges.push({ v0, v1, curve });
    }
    for (const he of chain) {
      heEdge[he] = e + 1;
      const tw = twin(he);
      if (tw >= 0) heEdge[tw] = -(e + 1);
    }
    return e;
  };

  /**
   * A region touching itself at a vertex traces one loop through that vertex twice: split it there
   * into simple loops (an outer boundary and the hole touching it), as the kernel expects.
   */
  const splitPinched = (loop: number[]): number[][] => {
    const out: number[][] = [];
    const stack: number[] = [];
    const at = new Map<number, number>();
    for (const h of loop) {
      const i = at.get(src(h));
      if (i !== undefined) {
        const sub = stack.splice(i);
        for (const x of sub) at.delete(src(x));
        out.push(sub);
      }
      at.set(src(h), stack.length);
      stack.push(h);
    }
    if (stack.length) out.push(stack);
    return out;
  };
  /** edge uses of a closed loop of boundary half-edges of face f */
  const loopUses = (loop: number[], f: number): EdgeUse[] => {
    // rotate to start at a junction
    const start = loop.findIndex((h) => junction[src(h)]);
    const uses: EdgeUse[] = [];
    if (start < 0) {
      // closed edge without vertices
      const existing = heEdge[loop[0]];
      if (existing) uses.push({ edge: Math.abs(existing) - 1, reversed: existing < 0 });
      else uses.push({ edge: makeEdge(loop, f, true), reversed: false });
    } else {
      const rot = loop.slice(start).concat(loop.slice(0, start));
      let chain: number[] = [];
      const flush = () => {
        if (!chain.length) return;
        const existing = heEdge[chain[0]];
        if (existing) uses.push({ edge: Math.abs(existing) - 1, reversed: existing < 0 });
        else uses.push({ edge: makeEdge(chain, f, false), reversed: false });
        chain = [];
      };
      for (const h of rot) {
        if (chain.length && junction[src(h)]) flush();
        chain.push(h);
      }
      flush();
    }
    return uses;
  };

  for (let t = 0; t < nt; t++) {
    for (let k = 0; k < 3; k++) {
      const he0 = 3 * t + k;
      if (visited[he0] || !isBoundary(he0)) continue;
      // trace the loop
      const loop: number[] = [];
      let he = he0;
      let ok = true;
      while (!visited[he]) {
        visited[he] = 1;
        loop.push(he);
        he = nextBoundary(he);
        if (he < 0) {
          ok = false;
          break;
        }
      }
      if (!ok || he !== he0) continue; // malformed loop (non-manifold mesh): leave to the faceted fallback
      const f = faceOf[t];
      for (const simple of splitPinched(loop)) faceLoops[f].push(loopUses(simple, f));
    }
  }

  // --- faces (orientation and seam placement from the region's triangles)
  for (let f = 0; f < nf; f++) {
    if (f < regions.length)
      faces.push({ kind: regions[f].surface.kind, surface: regions[f].surface, placement: null, forward: true, loops: faceLoops[f], region: f, area: regions[f].area });
    else faces.push({ kind: "faceted", surface: null, placement: null, forward: true, loops: faceLoops[f], region: -1 });
  }
  const nb = new Float64Array(3);
  const dotSum = new Float64Array(regions.length);
  const regionPts: Vec3[][] = regions.map(() => []);
  const seenV = new Int32Array(nv).fill(-1);
  for (let t = 0; t < nt; t++) {
    const f = labels[t];
    if (f < 0) continue;
    const a = 3 * I[3 * t], b = 3 * I[3 * t + 1], c = 3 * I[3 * t + 2];
    const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2];
    const vx = P[c] - P[a], vy = P[c + 1] - P[a + 1], vz = P[c + 2] - P[a + 2];
    const n: Vec3 = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx];
    surfNormal(regions[f].surface)((P[a] + P[b] + P[c]) / 3, (P[a + 1] + P[b + 1] + P[c + 1]) / 3, (P[a + 2] + P[b + 2] + P[c + 2]) / 3, nb);
    dotSum[f] += n[0] * nb[0] + n[1] * nb[1] + n[2] * nb[2];
    for (let k = 0; k < 3; k++) {
      const v = I[3 * t + k];
      if (seenV[v] === f) continue;
      seenV[v] = f;
      regionPts[f].push(getP(P, v));
    }
  }
  regions.forEach((r, f) => {
    let surface = r.surface;
    if (surface.kind === "plane" && dotSum[f] < 0) {
      // planes: point the normal outward so the face is built forward
      surface = { ...surface, normal: [-surface.normal[0], -surface.normal[1], -surface.normal[2]] };
      faces[f].surface = surface;
      faces[f].forward = true;
    } else faces[f].forward = dotSum[f] >= 0;
    faces[f].placement = placementFor(surface, regionPts[f]);
    const band = bandOf(surface, faces[f].loops, edges, vertices, regionPts[f], tol);
    if (band) {
      faces[f].placement = band.placement;
      faces[f].band = { v0: band.v0, v1: band.v1 };
    }
  });

  return { vertices, edges, faces, tolerance: tol };
}
