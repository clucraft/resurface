/**
 * Geometric constraint solver for 2D sketches.
 *
 * Unknowns are the free point coordinates, circle radii and one auxiliary radius per arc (with the two
 * implicit equations |start-center| = r, |end-center| = r). Every constraint contributes residual
 * equations with analytic gradients (a few rarely used ones use central differences).
 *
 * The system is split into independent connected sub-systems. Each is solved with damped Gauss–Newton
 * using minimum-norm steps  dx = -W^-1 J^T (J W^-1 J^T + mu D)^-1 F  so that under-constrained geometry
 * moves as little as possible. J W^-1 J^T is sparse; it is assembled in envelope storage after a reverse
 * Cuthill–McKee ordering and factorised with a rank-revealing Cholesky that drops dependent rows (the
 * dropped rows identify redundant / conflicting constraints and give the Jacobian rank for the DOF).
 *
 * Dragging: the dragged point is first moved onto the target, then the solve runs with a large weight
 * on that point's coordinates, i.e. it is pulled toward the target (soft) while the hard constraints
 * are restored by moving the other geometry as little as possible.
 */
import type { Vec2 } from "../core/types";
import {
  cloneSketch,
  isDimension,
  validateConstraint,
  type Constraint,
  type DimensionConstraint,
  type Sketch,
  type SketchEntity,
} from "./model";
import { EnvelopeMatrix, rcmOrder } from "./sparse";

export interface SolveOptions {
  /** drag a point toward a target (soft), keeping hard constraints satisfied */
  drag?: { pointId: string; target: Vec2 };
  /** Gauss–Newton iteration limit per sub-system (default 60) */
  maxIterations?: number;
  /** max |residual| considered solved (default 1e-10 * sketch scale) */
  tolerance?: number;
  /** compute degrees of freedom (default true) */
  computeDof?: boolean;
  /** on redundancy/conflict, re-run in constraint order so later constraints are blamed (default: true unless dragging) */
  analyzeConflicts?: boolean;
  /** report which points can still move (default false) */
  computeFreePoints?: boolean;
  /** weight of the dragged point (default 1000) */
  dragWeight?: number;
}

export type SolveStatus = "ok" | "failed" | "overconstrained";

export interface SolveResult {
  /** solved copy of the input (failed sub-systems keep their original positions) */
  sketch: Sketch;
  status: SolveStatus;
  /** max |residual| over all equations after solving */
  residual: number;
  /** remaining degrees of freedom (free parameters minus Jacobian rank) */
  dof: number;
  /** constraint ids that cannot be satisfied together with the others */
  conflicting?: string[];
  /** constraint ids that are redundant (consistent but not independent) */
  redundant?: string[];
  /** constraint ids that were ignored because they are invalid */
  invalid?: string[];
  /** points that are not fully determined (only with computeFreePoints) */
  freePoints?: string[];
  /** total Gauss–Newton iterations */
  iterations: number;
}

// ---------------------------------------------------------------------------
// Equations
// ---------------------------------------------------------------------------

type Pair = [number, number];

interface Eq {
  /** owning constraint id ("" for implicit arc equations) */
  cid: string;
  /** global parameter indices of the local variables (may repeat) */
  idx: Int32Array;
  f: (v: Float64Array) => number;
  g?: (v: Float64Array, out: Float64Array) => void;
  v: Float64Array;
  gl: Float64Array;
}

function mk(cid: string, idx: number[], f: Eq["f"], g?: Eq["g"]): Eq {
  return { cid, idx: Int32Array.from(idx), f, g, v: new Float64Array(idx.length), gl: new Float64Array(idx.length) };
}

let TINY = 1e-12;

function eqLinear(cid: string, idx: number[], coef: number[], k: number): Eq {
  return mk(
    cid,
    idx,
    (v) => {
      let s = -k;
      for (let i = 0; i < coef.length; i++) s += coef[i] * v[i];
      return s;
    },
    (_v, g) => {
      for (let i = 0; i < coef.length; i++) g[i] = coef[i];
    },
  );
}

/** |q - p| - target - sum(coef_j * extra_j) */
function eqDist(cid: string, p: Pair, q: Pair, target: number, extra: [number, number][] = []): Eq {
  const idx = [p[0], p[1], q[0], q[1], ...extra.map((e) => e[0])];
  const coef = extra.map((e) => e[1]);
  return mk(
    cid,
    idx,
    (v) => {
      let s = Math.hypot(v[2] - v[0], v[3] - v[1]) - target;
      for (let j = 0; j < coef.length; j++) s -= coef[j] * v[4 + j];
      return s;
    },
    (v, g) => {
      const dx = v[2] - v[0], dy = v[3] - v[1];
      const L = Math.hypot(dx, dy);
      let ux = 1, uy = 0;
      if (L > TINY) {
        ux = dx / L;
        uy = dy / L;
      }
      g[0] = -ux;
      g[1] = -uy;
      g[2] = ux;
      g[3] = uy;
      for (let j = 0; j < coef.length; j++) g[4 + j] = -coef[j];
    },
  );
}

/** |b - a| - |d - c| */
function eqDistDiff(cid: string, a: Pair, b: Pair, c: Pair, d: Pair): Eq {
  return mk(
    cid,
    [a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1]],
    (v) => Math.hypot(v[2] - v[0], v[3] - v[1]) - Math.hypot(v[6] - v[4], v[7] - v[5]),
    (v, g) => {
      const ux = v[2] - v[0], uy = v[3] - v[1], wx = v[6] - v[4], wy = v[7] - v[5];
      const lu = Math.hypot(ux, uy), lw = Math.hypot(wx, wy);
      const au = lu > TINY ? 1 / lu : 0, aw = lw > TINY ? 1 / lw : 0;
      g[0] = -ux * au;
      g[1] = -uy * au;
      g[2] = ux * au;
      g[3] = uy * au;
      g[4] = wx * aw;
      g[5] = wy * aw;
      g[6] = -wx * aw;
      g[7] = -wy * aw;
    },
  );
}

/** sigma * signedDistance(P, line AB) - target - sum(coef_j * extra_j)   (signed: left of A->B positive) */
function eqPointLine(cid: string, p: Pair, a: Pair, b: Pair, sigma: number, target: number, extra: [number, number][] = []): Eq {
  const idx = [p[0], p[1], a[0], a[1], b[0], b[1], ...extra.map((e) => e[0])];
  const coef = extra.map((e) => e[1]);
  return mk(
    cid,
    idx,
    (v) => {
      const dx = v[4] - v[2], dy = v[5] - v[3];
      const L = Math.hypot(dx, dy);
      let s = -target;
      for (let j = 0; j < coef.length; j++) s -= coef[j] * v[6 + j];
      if (L <= TINY) return 0;
      return s + (sigma * (dx * (v[1] - v[3]) - dy * (v[0] - v[2]))) / L;
    },
    (v, g) => {
      const px = v[0], py = v[1], ax = v[2], ay = v[3], bx = v[4], by = v[5];
      const dx = bx - ax, dy = by - ay;
      const L = Math.hypot(dx, dy);
      g.fill(0);
      if (L <= TINY) return;
      const c = dx * (py - ay) - dy * (px - ax);
      const iL = 1 / L, cL3 = c / (L * L * L);
      g[0] = sigma * (-dy * iL);
      g[1] = sigma * (dx * iL);
      g[2] = sigma * ((by - py) * iL - cL3 * -dx);
      g[3] = sigma * ((px - bx) * iL - cL3 * -dy);
      g[4] = sigma * ((py - ay) * iL - cL3 * dx);
      g[5] = sigma * (-(px - ax) * iL - cL3 * dy);
      for (let j = 0; j < coef.length; j++) g[6 + j] = -coef[j];
    },
  );
}

/** Relation between directions u = b - a and w = d - c: normalised cross (parallel), dot (perpendicular) or angle. */
function eqDirPair(cid: string, a: Pair, b: Pair, c: Pair, d: Pair, mode: "cross" | "dot" | "angle", target = 0): Eq {
  return mk(
    cid,
    [a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1]],
    (v) => {
      const ux = v[2] - v[0], uy = v[3] - v[1], wx = v[6] - v[4], wy = v[7] - v[5];
      const lu = Math.hypot(ux, uy), lw = Math.hypot(wx, wy);
      if (lu <= TINY || lw <= TINY) return 0;
      if (mode === "cross") return (ux * wy - uy * wx) / (lu * lw);
      if (mode === "dot") return (ux * wx + uy * wy) / (lu * lw);
      return wrapPi(Math.atan2(ux * wy - uy * wx, ux * wx + uy * wy) - target);
    },
    (v, g) => {
      const ux = v[2] - v[0], uy = v[3] - v[1], wx = v[6] - v[4], wy = v[7] - v[5];
      const lu = Math.hypot(ux, uy), lw = Math.hypot(wx, wy);
      g.fill(0);
      if (lu <= TINY || lw <= TINY) return;
      let gux: number, guy: number, gwx: number, gwy: number;
      if (mode === "angle") {
        const lu2 = lu * lu, lw2 = lw * lw;
        gux = uy / lu2;
        guy = -ux / lu2;
        gwx = -wy / lw2;
        gwy = wx / lw2;
      } else {
        const iuw = 1 / (lu * lw), lu2 = lu * lu, lw2 = lw * lw;
        if (mode === "cross") {
          const s = (ux * wy - uy * wx) * iuw;
          gux = wy * iuw - (s * ux) / lu2;
          guy = -wx * iuw - (s * uy) / lu2;
          gwx = -uy * iuw - (s * wx) / lw2;
          gwy = ux * iuw - (s * wy) / lw2;
        } else {
          const s = (ux * wx + uy * wy) * iuw;
          gux = wx * iuw - (s * ux) / lu2;
          guy = wy * iuw - (s * uy) / lu2;
          gwx = ux * iuw - (s * wx) / lw2;
          gwy = uy * iuw - (s * wy) / lw2;
        }
      }
      g[0] = -gux;
      g[1] = -guy;
      g[2] = gux;
      g[3] = guy;
      g[4] = -gwx;
      g[5] = -gwy;
      g[6] = gwx;
      g[7] = gwy;
    },
  );
}

/** Generic equation with central-difference gradient. */
function eqFD(cid: string, idx: number[], f: (v: Float64Array) => number): Eq {
  return mk(cid, idx, f);
}

function wrapPi(a: number): number {
  const t = Math.PI * 2;
  a = a % t;
  if (a <= -Math.PI) a += t;
  else if (a > Math.PI) a -= t;
  return a;
}

const sgn = (x: number) => (x < 0 ? -1 : 1);

// ---------------------------------------------------------------------------
// System construction
// ---------------------------------------------------------------------------

interface Built {
  x: Float64Array;
  fixed: Uint8Array;
  eqs: Eq[];
  pointBase: Map<string, number>;
  circleR: Map<string, number>;
  invalid: string[];
  scale: number;
}

type RefInfo =
  | { kind: "point"; id: string; p: Pair }
  | { kind: "line"; e: SketchEntity & { type: "line" }; a: Pair; b: Pair }
  | { kind: "arc"; e: SketchEntity & { type: "arc" }; c: Pair; s: Pair; en: Pair; r: number }
  | { kind: "circle"; e: SketchEntity & { type: "circle" }; c: Pair; r: number }
  | { kind: "spline"; e: SketchEntity & { type: "spline" } };

function buildSystem(sk: Sketch): Built {
  const pointBase = new Map<string, number>();
  const xs: number[] = [];
  let scale = 1;
  for (const p of sk.points) {
    pointBase.set(p.id, xs.length);
    xs.push(p.x, p.y);
    scale = Math.max(scale, Math.abs(p.x), Math.abs(p.y));
  }
  const ents = new Map<string, SketchEntity>();
  // entities that reference missing points are ignored (and so are constraints on them)
  const broken = new Set<string>();
  for (const e of sk.entities) {
    const pts = e.type === "line" ? [e.p1, e.p2] : e.type === "arc" ? [e.center, e.start, e.end] : e.type === "circle" ? [e.center] : e.points;
    if (pts.some((id) => !pointBase.has(id))) broken.add(e.id);
    else ents.set(e.id, e);
  }
  const circleR = new Map<string, number>();
  const arcR = new Map<string, number>();
  const pp = (id: string): Pair => {
    const b = pointBase.get(id)!;
    return [b, b + 1];
  };
  for (const e of sk.entities) {
    if (broken.has(e.id)) continue;
    if (e.type === "circle") {
      circleR.set(e.id, xs.length);
      xs.push(e.radius);
      scale = Math.max(scale, Math.abs(e.radius));
    } else if (e.type === "arc" && pointBase.has(e.center) && pointBase.has(e.start)) {
      const c = pp(e.center), s = pp(e.start);
      arcR.set(e.id, xs.length);
      xs.push(Math.hypot(xs[s[0]] - xs[c[0]], xs[s[1]] - xs[c[1]]));
    }
  }
  const x = Float64Array.from(xs);
  const fixed = new Uint8Array(x.length);
  TINY = 1e-12 * scale;
  const fixPoint = (id: string) => {
    const b = pointBase.get(id);
    if (b !== undefined) fixed[b] = fixed[b + 1] = 1;
  };
  for (const p of sk.points) if (p.fixed) fixPoint(p.id);

  const invalid: string[] = [];
  const eqs: Eq[] = [];
  // implicit arc equations
  for (const e of sk.entities) {
    if (e.type !== "arc" || broken.has(e.id)) continue;
    const r = arcR.get(e.id);
    if (r === undefined || !pointBase.has(e.end)) continue;
    eqs.push(eqDist("", pp(e.center), pp(e.start), 0, [[r, 1]]));
    eqs.push(eqDist("", pp(e.center), pp(e.end), 0, [[r, 1]]));
  }

  const ref = (id: string): RefInfo | null => {
    if (pointBase.has(id)) return { kind: "point", id, p: pp(id) };
    const e = ents.get(id);
    if (!e) return null;
    switch (e.type) {
      case "line":
        return { kind: "line", e, a: pp(e.p1), b: pp(e.p2) };
      case "arc":
        return { kind: "arc", e, c: pp(e.center), s: pp(e.start), en: pp(e.end), r: arcR.get(e.id)! };
      case "circle":
        return { kind: "circle", e, c: pp(e.center), r: circleR.get(e.id)! };
      case "spline":
        return { kind: "spline", e };
    }
  };
  const val = (i: number) => x[i];
  const centerOf = (r: RefInfo): Pair | null => (r.kind === "point" ? r.p : r.kind === "arc" || r.kind === "circle" ? r.c : null);
  const radiusOf = (r: RefInfo): number | null => (r.kind === "arc" || r.kind === "circle" ? r.r : null);
  const cross = (p: Pair, a: Pair, b: Pair) =>
    (val(b[0]) - val(a[0])) * (val(p[1]) - val(a[1])) - (val(b[1]) - val(a[1])) * (val(p[0]) - val(a[0]));
  const lin2 = (cid: string, a: Pair, b: Pair, ca: number, cb: number, kx = 0, ky = 0) => {
    eqs.push(eqLinear(cid, [a[0], b[0]], [ca, cb], kx));
    eqs.push(eqLinear(cid, [a[1], b[1]], [ca, cb], ky));
  };
  const pointOn = (cid: string, p: Pair, r: RefInfo): boolean => {
    if (r.kind === "line") eqs.push(eqPointLine(cid, p, r.a, r.b, 1, 0));
    else if (r.kind === "arc" || r.kind === "circle") eqs.push(eqDist(cid, r.c, p, 0, [[r.r, 1]]));
    else return false;
    return true;
  };
  const endpointsOf = (r: RefInfo): string[] =>
    r.kind === "line" ? [r.e.p1, r.e.p2] : r.kind === "arc" ? [r.e.start, r.e.end] : [];

  for (const c of sk.constraints) {
    if (validateConstraint(sk, c) || c.refs.some((r) => broken.has(r))) {
      invalid.push(c.id);
      continue;
    }
    if (isDimension(c) && !c.driving) continue;
    const R = c.refs.map(ref) as RefInfo[];
    const cid = c.id;
    let ok = true;
    switch (c.type) {
      case "fix":
        if (R[0].kind === "point") fixPoint(R[0].id);
        else if (R[0].kind === "line") {
          fixPoint(R[0].e.p1);
          fixPoint(R[0].e.p2);
        } else if (R[0].kind === "arc") {
          fixPoint(R[0].e.center);
          fixPoint(R[0].e.start);
          fixPoint(R[0].e.end);
        } else if (R[0].kind === "circle") {
          fixPoint(R[0].e.center);
          fixed[R[0].r] = 1;
        } else if (R[0].kind === "spline") R[0].e.points.forEach(fixPoint);
        break;
      case "coincident": {
        const [a, b] = R;
        if (a.kind === "point" && b.kind === "point") {
          if (a.id !== b.id) lin2(cid, a.p, b.p, 1, -1);
        } else if (a.kind === "point") ok = pointOn(cid, a.p, b);
        else if (b.kind === "point") ok = pointOn(cid, b.p, a);
        else ok = false;
        break;
      }
      case "pointOnEntity": {
        const [a, b] = R;
        ok = a.kind === "point" ? pointOn(cid, a.p, b) : b.kind === "point" ? pointOn(cid, b.p, a) : false;
        break;
      }
      case "horizontal":
      case "vertical": {
        const k = c.type === "horizontal" ? 1 : 0;
        const [p, q] = R[0].kind === "line" ? [R[0].a, R[0].b] : [(R[0] as { p: Pair }).p, (R[1] as { p: Pair }).p];
        eqs.push(eqLinear(cid, [p[k], q[k]], [-1, 1], 0));
        break;
      }
      case "parallel":
      case "perpendicular": {
        const [l1, l2] = R as Extract<RefInfo, { kind: "line" }>[];
        eqs.push(eqDirPair(cid, l1.a, l1.b, l2.a, l2.b, c.type === "parallel" ? "cross" : "dot"));
        break;
      }
      case "collinear": {
        const [l1, l2] = R as Extract<RefInfo, { kind: "line" }>[];
        eqs.push(eqPointLine(cid, l2.a, l1.a, l1.b, 1, 0));
        eqs.push(eqPointLine(cid, l2.b, l1.a, l1.b, 1, 0));
        break;
      }
      case "tangent": {
        let [a, b] = R;
        if (b.kind === "line") [a, b] = [b, a];
        const shared = endpointsOf(a).find((id) => endpointsOf(b).includes(id));
        if (a.kind === "line" && (b.kind === "arc" || b.kind === "circle")) {
          if (shared && b.kind === "arc") {
            const P = pointBase.get(shared)!;
            eqs.push(eqDirPair(cid, b.c, [P, P + 1], a.a, a.b, "dot"));
          } else {
            const side = sgn(cross(b.c, a.a, a.b));
            eqs.push(eqPointLine(cid, b.c, a.a, a.b, side, 0, [[b.r, 1]]));
          }
        } else if ((a.kind === "arc" || a.kind === "circle") && (b.kind === "arc" || b.kind === "circle")) {
          if (shared) {
            const P = pointBase.get(shared)!;
            eqs.push(eqDirPair(cid, a.c, [P, P + 1], b.c, [P, P + 1], "cross"));
          } else {
            const d = Math.hypot(val(a.c[0]) - val(b.c[0]), val(a.c[1]) - val(b.c[1]));
            const r1 = val(a.r), r2 = val(b.r);
            if (Math.abs(d - (r1 + r2)) <= Math.abs(d - Math.abs(r1 - r2))) eqs.push(eqDist(cid, a.c, b.c, 0, [[a.r, 1], [b.r, 1]]));
            else {
              const s = sgn(r1 - r2);
              eqs.push(eqDist(cid, a.c, b.c, 0, [[a.r, s], [b.r, -s]]));
            }
          }
        } else ok = false;
        break;
      }
      case "equal": {
        const [a, b] = R;
        if (a.kind === "line" && b.kind === "line") eqs.push(eqDistDiff(cid, a.a, a.b, b.a, b.b));
        else {
          const ra = radiusOf(a), rb = radiusOf(b);
          if (ra === null || rb === null) ok = false;
          else eqs.push(eqLinear(cid, [ra, rb], [1, -1], 0));
        }
        break;
      }
      case "concentric": {
        const ca = centerOf(R[0]), cb = centerOf(R[1]);
        if (!ca || !cb) ok = false;
        else if (ca[0] !== cb[0]) lin2(cid, ca, cb, 1, -1);
        break;
      }
      case "midpoint": {
        let [p, e] = R;
        if (p.kind !== "point") [p, e] = [e, p];
        if (p.kind !== "point") {
          ok = false;
          break;
        }
        if (e.kind === "line") {
          const P = p.p, A = e.a, B = e.b;
          eqs.push(eqLinear(cid, [P[0], A[0], B[0]], [2, -1, -1], 0));
          eqs.push(eqLinear(cid, [P[1], A[1], B[1]], [2, -1, -1], 0));
        } else if (e.kind === "arc") {
          eqs.push(eqDist(cid, e.c, p.p, 0, [[e.r, 1]]));
          eqs.push(eqDistDiff(cid, p.p, e.s, p.p, e.en));
        } else ok = false;
        break;
      }
      case "symmetric": {
        const [p, q, m] = R as [Extract<RefInfo, { kind: "point" }>, Extract<RefInfo, { kind: "point" }>, RefInfo];
        if (m.kind === "point") {
          eqs.push(eqLinear(cid, [p.p[0], q.p[0], m.p[0]], [1, 1, -2], 0));
          eqs.push(eqLinear(cid, [p.p[1], q.p[1], m.p[1]], [1, 1, -2], 0));
        } else if (m.kind === "line") {
          const { a, b } = m;
          eqs.push(
            eqFD(cid, [p.p[0], p.p[1], q.p[0], q.p[1], a[0], a[1], b[0], b[1]], (v) => {
              const dx = v[6] - v[4], dy = v[7] - v[5];
              const L = Math.hypot(dx, dy);
              if (L <= TINY) return 0;
              const mx = (v[0] + v[2]) / 2 - v[4], my = (v[1] + v[3]) / 2 - v[5];
              return (dx * my - dy * mx) / L;
            }),
          );
          eqs.push(
            eqFD(cid, [p.p[0], p.p[1], q.p[0], q.p[1], a[0], a[1], b[0], b[1]], (v) => {
              const dx = v[6] - v[4], dy = v[7] - v[5];
              const L = Math.hypot(dx, dy);
              if (L <= TINY) return 0;
              return ((v[2] - v[0]) * dx + (v[3] - v[1]) * dy) / L;
            }),
          );
        } else ok = false;
        break;
      }
      default:
        ok = buildDimension(c as DimensionConstraint, R, eqs, x, centerOf);
    }
    if (!ok) invalid.push(c.id);
  }
  return { x, fixed, eqs, pointBase, circleR, invalid, scale };
}

function buildDimension(
  c: DimensionConstraint,
  R: RefInfo[],
  eqs: Eq[],
  x: Float64Array,
  centerOf: (r: RefInfo) => Pair | null,
): boolean {
  const cid = c.id;
  const v = c.value;
  const val = (i: number) => x[i];
  switch (c.type) {
    case "distance": {
      if (R.length === 1) {
        if (R[0].kind !== "line") return false;
        eqs.push(eqDist(cid, R[0].a, R[0].b, Math.abs(v)));
        return true;
      }
      const [a, b] = R;
      const pa = centerOf(a), pb = centerOf(b);
      if (pa && pb) {
        if (Math.abs(v) <= TINY) {
          eqs.push(eqLinear(cid, [pa[0], pb[0]], [1, -1], 0));
          eqs.push(eqLinear(cid, [pa[1], pb[1]], [1, -1], 0));
        } else eqs.push(eqDist(cid, pa, pb, Math.abs(v)));
        return true;
      }
      const pl = (p: Pair, l: Extract<RefInfo, { kind: "line" }>) => {
        const cr =
          (val(l.b[0]) - val(l.a[0])) * (val(p[1]) - val(l.a[1])) - (val(l.b[1]) - val(l.a[1])) * (val(p[0]) - val(l.a[0]));
        eqs.push(eqPointLine(cid, p, l.a, l.b, sgn(cr), Math.abs(v)));
      };
      if (pa && b.kind === "line") {
        pl(pa, b);
        return true;
      }
      if (pb && a.kind === "line") {
        pl(pb, a);
        return true;
      }
      if (a.kind === "line" && b.kind === "line") {
        const idx = [b.a[0], b.a[1], b.b[0], b.b[1], a.a[0], a.a[1], a.b[0], a.b[1]];
        const sd = (w: ArrayLike<number>) => {
          const dx = w[6] - w[4], dy = w[7] - w[5];
          const L = Math.hypot(dx, dy);
          if (L <= TINY) return 0;
          const mx = (w[0] + w[2]) / 2 - w[4], my = (w[1] + w[3]) / 2 - w[5];
          return (dx * my - dy * mx) / L;
        };
        const s = sgn(sd(idx.map(val)));
        eqs.push(eqFD(cid, idx, (w) => s * sd(w) - Math.abs(v)));
        return true;
      }
      return false;
    }
    case "horizontalDistance":
    case "verticalDistance": {
      const k = c.type === "horizontalDistance" ? 0 : 1;
      let p: Pair | null, q: Pair | null;
      if (R.length === 1) {
        if (R[0].kind !== "line") return false;
        p = R[0].a;
        q = R[0].b;
      } else {
        p = centerOf(R[0]);
        q = centerOf(R[1]);
      }
      if (!p || !q) return false;
      const s = sgn(val(q[k]) - val(p[k]));
      eqs.push(eqLinear(cid, [p[k], q[k]], [-s, s], Math.abs(v)));
      return true;
    }
    case "radius":
    case "diameter": {
      const r = R[0];
      if (r.kind !== "arc" && r.kind !== "circle") return false;
      eqs.push(eqLinear(cid, [r.r], [1], c.type === "radius" ? v : v / 2));
      return true;
    }
    case "angle": {
      const [a, b] = R;
      if (a.kind !== "line" || b.kind !== "line") return false;
      const ux = val(a.b[0]) - val(a.a[0]), uy = val(a.b[1]) - val(a.a[1]);
      const wx = val(b.b[0]) - val(b.a[0]), wy = val(b.b[1]) - val(b.a[1]);
      const cur = Math.atan2(ux * wy - uy * wx, ux * wx + uy * wy);
      eqs.push(eqDirPair(cid, a.a, a.b, b.a, b.b, "angle", sgn(cur) * Math.abs(v)));
      return true;
    }
    case "length": {
      const r = R[0];
      if (r.kind === "line") {
        eqs.push(eqDist(cid, r.a, r.b, Math.abs(v)));
        return true;
      }
      if (r.kind === "arc") {
        eqs.push(
          eqFD(cid, [r.c[0], r.c[1], r.s[0], r.s[1], r.en[0], r.en[1], r.r], (w) => {
            const a0 = Math.atan2(w[3] - w[1], w[2] - w[0]);
            const a1 = Math.atan2(w[5] - w[1], w[4] - w[0]);
            let sw = (a1 - a0) % (2 * Math.PI);
            if (sw <= 0) sw += 2 * Math.PI;
            return w[6] * sw - Math.abs(v);
          }),
        );
        return true;
      }
      return false;
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// Sub-system (connected component) numerics
// ---------------------------------------------------------------------------

class SubSystem {
  readonly m: number;
  readonly n: number;
  /** local column -> global parameter index */
  readonly params: Int32Array;
  /** per equation: offset into the flat Jacobian values */
  readonly rowOff: Int32Array;
  /** flat Jacobian: column per entry */
  readonly jCol: Int32Array;
  readonly jVal: Float64Array;
  /** per equation, per local variable: entry index (or -1 for fixed params) */
  readonly slot: Int32Array[];
  /** position of each equation in the elimination order */
  readonly pos: Int32Array;
  readonly A: EnvelopeMatrix;
  /** assembly triples: entry1, entry2, envelope index */
  private readonly pe1: Int32Array;
  private readonly pe2: Int32Array;
  private readonly ptg: Int32Array;
  readonly winv: Float64Array;
  readonly F: Float64Array;
  /** homotopy offset: residuals are F(x) - offsetScale * offset */
  offset: Float64Array | null = null;
  offsetScale = 0;

  constructor(
    readonly eqs: Eq[],
    params: number[],
    colOf: Int32Array,
    ordering: "rcm" | "natural",
    weight: (param: number) => number,
  ) {
    const m = (this.m = eqs.length);
    const n = (this.n = params.length);
    this.params = Int32Array.from(params);
    this.winv = new Float64Array(n);
    for (let k = 0; k < n; k++) this.winv[k] = 1 / weight(params[k]);
    // entries
    const rowOff = new Int32Array(m + 1);
    const slots: Int32Array[] = [];
    const cols: number[] = [];
    for (let i = 0; i < m; i++) {
      const e = eqs[i];
      const sl = new Int32Array(e.idx.length).fill(-1);
      const local = new Map<number, number>();
      const base = cols.length;
      for (let l = 0; l < e.idx.length; l++) {
        const c = colOf[e.idx[l]];
        if (c < 0) continue;
        let s = local.get(c);
        if (s === undefined) {
          s = cols.length - base;
          local.set(c, s);
          cols.push(c);
        }
        sl[l] = s;
      }
      slots.push(sl);
      rowOff[i + 1] = cols.length;
    }
    this.rowOff = rowOff;
    this.slot = slots;
    this.jCol = Int32Array.from(cols);
    this.jVal = new Float64Array(cols.length);
    this.F = new Float64Array(m);
    // column -> entries
    const colEntries: number[][] = Array.from({ length: n }, () => []);
    const entryRow = new Int32Array(cols.length);
    for (let i = 0; i < m; i++)
      for (let e = rowOff[i]; e < rowOff[i + 1]; e++) {
        colEntries[cols[e]].push(e);
        entryRow[e] = i;
      }
    // ordering
    let perm: Int32Array;
    if (ordering === "rcm" && m > 2) {
      const adjSets: Set<number>[] = Array.from({ length: m }, () => new Set<number>());
      for (const list of colEntries)
        for (let a = 0; a < list.length; a++)
          for (let b = a + 1; b < list.length; b++) {
            const i = entryRow[list[a]], j = entryRow[list[b]];
            if (i !== j) {
              adjSets[i].add(j);
              adjSets[j].add(i);
            }
          }
      perm = rcmOrder(
        m,
        adjSets.map((s) => Array.from(s)),
      );
    } else perm = Int32Array.from({ length: m }, (_, i) => i);
    const pos = (this.pos = new Int32Array(m));
    for (let k = 0; k < m; k++) pos[perm[k]] = k;
    // envelope
    const fc = new Int32Array(m);
    for (let k = 0; k < m; k++) fc[k] = k;
    for (const list of colEntries)
      for (let a = 0; a < list.length; a++)
        for (let b = 0; b < list.length; b++) {
          const pi = pos[entryRow[list[a]]], pj = pos[entryRow[list[b]]];
          if (pj < fc[pi]) fc[pi] = pj;
        }
    this.A = new EnvelopeMatrix(m, fc);
    // assembly triples (lower triangle incl. diagonal)
    const e1: number[] = [], e2: number[] = [], tg: number[] = [];
    for (const list of colEntries)
      for (let a = 0; a < list.length; a++)
        for (let b = 0; b < list.length; b++) {
          const pi = pos[entryRow[list[a]]], pj = pos[entryRow[list[b]]];
          if (pj > pi) continue;
          e1.push(list[a]);
          e2.push(list[b]);
          tg.push(this.A.index(pi, pj));
        }
    this.pe1 = Int32Array.from(e1);
    this.pe2 = Int32Array.from(e2);
    this.ptg = Int32Array.from(tg);
  }

  /** Evaluate residuals (into F) and optionally the Jacobian at x. Returns max |F|. */
  evaluate(x: Float64Array, withJacobian: boolean): number {
    let mx = 0;
    const { eqs, rowOff, jVal } = this;
    for (let i = 0; i < eqs.length; i++) {
      const e = eqs[i];
      const v = e.v;
      for (let l = 0; l < e.idx.length; l++) v[l] = x[e.idx[l]];
      let f = e.f(v);
      if (this.offset) f -= this.offsetScale * this.offset[i];
      if (!isFinite(f)) f = 1e30;
      this.F[i] = f;
      const af = Math.abs(f);
      if (af > mx) mx = af;
      if (!withJacobian) continue;
      const g = e.gl;
      if (e.g) e.g(v, g);
      else fdGrad(e, v, g);
      for (let k = rowOff[i]; k < rowOff[i + 1]; k++) jVal[k] = 0;
      const sl = this.slot[i];
      for (let l = 0; l < sl.length; l++) {
        const s = sl[l];
        if (s >= 0) jVal[rowOff[i] + s] += isFinite(g[l]) ? g[l] : 0;
      }
    }
    return mx;
  }

  /** Assemble J W^-1 J^T (+ mu * diag) and factorise. Returns the rank. */
  factor(mu: number, relTol: number): number {
    const { A, pe1, pe2, ptg, jVal, jCol, winv } = this;
    A.clear();
    const vals = A.vals;
    for (let k = 0; k < ptg.length; k++) vals[ptg[k]] += jVal[pe1[k]] * jVal[pe2[k]] * winv[jCol[pe1[k]]];
    if (mu > 0) for (let i = 0; i < this.m; i++) vals[A.index(i, i)] *= 1 + mu;
    return A.factor(relTol);
  }

  /** Minimum (weighted) norm step for J dx = -F using the current factorisation. */
  step(out: Float64Array): void {
    const { m, pos, rowOff, jCol, jVal, winv, F, A } = this;
    const b = new Float64Array(m);
    for (let i = 0; i < m; i++) b[pos[i]] = F[i];
    const lam = A.solve(b);
    out.fill(0);
    for (let i = 0; i < m; i++) {
      const li = lam[pos[i]];
      if (li === 0) continue;
      for (let e = rowOff[i]; e < rowOff[i + 1]; e++) out[jCol[e]] -= jVal[e] * li;
    }
    for (let k = 0; k < out.length; k++) out[k] *= winv[k];
  }

  isDropped(i: number): boolean {
    return this.A.dropped[this.pos[i]] === 1;
  }

  /** Sum of squared residuals over the non-dropped rows. */
  merit(): number {
    let s = 0;
    for (let i = 0; i < this.m; i++) if (!this.isDropped(i)) s += this.F[i] * this.F[i];
    return s;
  }

  /** Is unit direction e_k in the row space of J (i.e. parameter k fully determined)? Uses W = I factorisation. */
  determined(k: number, tol: number): boolean {
    // projection of e_k onto rowspace(J): J^T (J J^T)^-1 J e_k
    const { m, pos, rowOff, jCol, jVal, A } = this;
    const b = new Float64Array(m);
    for (let i = 0; i < m; i++)
      for (let e = rowOff[i]; e < rowOff[i + 1]; e++) if (jCol[e] === k) b[pos[i]] += jVal[e];
    const lam = A.solve(b);
    let pk = 0;
    for (let i = 0; i < m; i++) {
      const li = lam[pos[i]];
      if (li === 0) continue;
      for (let e = rowOff[i]; e < rowOff[i + 1]; e++) if (jCol[e] === k) pk += jVal[e] * li;
    }
    return Math.abs(1 - pk) < tol;
  }
}

function fdGrad(e: Eq, v: Float64Array, g: Float64Array): void {
  for (let l = 0; l < v.length; l++) {
    const x0 = v[l];
    const h = 1e-7 * Math.max(1, Math.abs(x0));
    v[l] = x0 + h;
    const fp = e.f(v);
    v[l] = x0 - h;
    const fm = e.f(v);
    v[l] = x0;
    g[l] = (fp - fm) / (2 * h);
  }
}

interface ComponentOutcome {
  status: SolveStatus;
  iterations: number;
  rank: number;
  conflicting: string[];
  redundant: string[];
  sys: SubSystem;
}

function solveSub(sys: SubSystem, x: Float64Array, tol: number, maxIt: number): ComponentOutcome {
  const n = sys.n;
  const dx = new Float64Array(n);
  const x0 = new Float64Array(n);
  const RANK_TOL = 1e-10;
  let mu = 0;
  let it = 0;
  sys.evaluate(x, true);
  let rank = sys.factor(0, RANK_TOL);
  let status: SolveStatus = "failed";
  while (true) {
    // convergence check over all rows
    let keptMax = 0, allMax = 0;
    for (let i = 0; i < sys.m; i++) {
      const a = Math.abs(sys.F[i]);
      if (a > allMax) allMax = a;
      if (!sys.isDropped(i) && a > keptMax) keptMax = a;
    }
    if (allMax <= tol) {
      status = "ok";
      break;
    }
    if (keptMax <= tol) {
      status = "overconstrained";
      break;
    }
    if (it >= maxIt) break;
    it++;
    sys.step(dx);
    const m0 = sys.merit();
    for (let k = 0; k < n; k++) x0[k] = x[sys.params[k]];
    let accepted = false;
    for (let alpha = 1; alpha >= 1 / 64; alpha /= 2) {
      for (let k = 0; k < n; k++) x[sys.params[k]] = x0[k] + alpha * dx[k];
      sys.evaluate(x, false);
      const m1 = sys.merit();
      if (m1 <= (1 - 1e-4 * alpha) * m0 || Math.sqrt(m1) <= tol) {
        accepted = true;
        break;
      }
    }
    if (!accepted) {
      for (let k = 0; k < n; k++) x[sys.params[k]] = x0[k];
      sys.evaluate(x, true);
      mu = mu === 0 ? 1e-6 : mu * 10;
      if (mu > 1e6) break;
      rank = sys.factor(mu, RANK_TOL);
      continue;
    }
    mu = mu > 1e-9 ? mu / 10 : 0;
    sys.evaluate(x, true);
    rank = sys.factor(mu, RANK_TOL);
  }
  if (mu !== 0) {
    sys.evaluate(x, true);
    rank = sys.factor(0, RANK_TOL);
  }
  const conflicting: string[] = [];
  const redundant: string[] = [];
  for (let i = 0; i < sys.m; i++) {
    if (!sys.isDropped(i)) continue;
    const cid = sys.eqs[i].cid;
    if (!cid) continue;
    if (Math.abs(sys.F[i]) > tol) conflicting.push(cid);
    else redundant.push(cid);
  }
  return { status, iterations: it, rank, conflicting: uniq(conflicting), redundant: uniq(redundant), sys };
}

const uniq = (a: string[]) => Array.from(new Set(a));

/**
 * Gauss–Newton first; when that does not converge, fall back to a residual homotopy
 * F(x) - (1 - lambda) F(x0) = 0, lambda: 0 -> 1, which follows large dimension changes robustly.
 */
function solveRobust(sys: SubSystem, x: Float64Array, tol: number, maxIt: number): ComponentOutcome {
  const params = sys.params;
  const xs0 = Float64Array.from(params, (p) => x[p]);
  const first = solveSub(sys, x, tol, Math.min(maxIt, 25));
  if (first.status !== "failed" || maxIt <= 25) return first;
  let iterations = first.iterations;
  params.forEach((p, k) => (x[p] = xs0[k]));
  sys.offset = null;
  sys.evaluate(x, false);
  const F0 = Float64Array.from(sys.F);
  sys.offset = F0;
  let lambda = 0;
  let step = 1 / 8;
  const xs = new Float64Array(params.length);
  let budget = maxIt * 4;
  while (lambda < 1 && budget > 0) {
    const next = Math.min(1, lambda + step);
    sys.offsetScale = 1 - next;
    params.forEach((p, k) => (xs[k] = x[p]));
    const r = solveSub(sys, x, tol, 15);
    iterations += r.iterations;
    budget -= r.iterations + 1;
    if (r.status === "failed") {
      params.forEach((p, k) => (x[p] = xs[k]));
      step /= 2;
      if (step < 1 / 512) break;
    } else {
      lambda = next;
      step = Math.min(0.5, step * 1.5);
    }
  }
  sys.offset = null;
  sys.offsetScale = 0;
  const out = solveSub(sys, x, tol, maxIt);
  out.iterations += iterations;
  return out;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/** Solve all constraints of a sketch. Returns a new sketch; the input is not modified. */
export function solveSketch(sketch: Sketch, opts: SolveOptions = {}): SolveResult {
  const B = buildSystem(sketch);
  const { x, fixed, eqs, pointBase, circleR } = B;
  const tol = opts.tolerance ?? 1e-10 * B.scale;
  const maxIt = opts.maxIterations ?? 60;
  const analyze = opts.analyzeConflicts ?? !opts.drag;
  const xInit = Float64Array.from(x);
  const nParams = x.length;

  // drag: move the point onto the target and weight it
  const dragParams = new Set<number>();
  if (opts.drag) {
    const b = pointBase.get(opts.drag.pointId);
    if (b !== undefined && !fixed[b] && !fixed[b + 1]) {
      x[b] = opts.drag.target[0];
      x[b + 1] = opts.drag.target[1];
      dragParams.add(b).add(b + 1);
    }
  }
  const xStart = Float64Array.from(x);
  const dragW = opts.dragWeight ?? 1000;
  const weight = (p: number) => (dragParams.has(p) ? dragW : 1);

  // union-find over free params connected through equations
  const parent = new Int32Array(nParams);
  for (let i = 0; i < nParams; i++) parent[i] = i;
  const find = (a: number): number => {
    while (parent[a] !== a) {
      parent[a] = parent[parent[a]];
      a = parent[a];
    }
    return a;
  };
  const inEq = new Uint8Array(nParams);
  const constEqs: Eq[] = [];
  const eqRoot: number[] = [];
  for (const e of eqs) {
    let first = -1;
    for (let l = 0; l < e.idx.length; l++) {
      const p = e.idx[l];
      if (fixed[p]) continue;
      inEq[p] = 1;
      if (first < 0) first = p;
      else {
        const ra = find(first), rb = find(p);
        if (ra !== rb) parent[rb] = ra;
      }
    }
    if (first < 0) constEqs.push(e);
    eqRoot.push(first);
  }
  const compEqs = new Map<number, Eq[]>();
  for (let i = 0; i < eqs.length; i++) {
    if (eqRoot[i] < 0) continue;
    const r = find(eqRoot[i]);
    let l = compEqs.get(r);
    if (!l) compEqs.set(r, (l = []));
    l.push(eqs[i]);
  }
  const compParams = new Map<number, number[]>();
  let freeParams = 0;
  for (let p = 0; p < nParams; p++) {
    if (fixed[p]) continue;
    freeParams++;
    if (!inEq[p]) continue;
    const r = find(p);
    let l = compParams.get(r);
    if (!l) compParams.set(r, (l = []));
    l.push(p);
  }

  const colOf = new Int32Array(nParams).fill(-1);
  let status: SolveStatus = "ok";
  let iterations = 0;
  let rankSum = 0;
  const conflicting: string[] = [];
  const redundant: string[] = [];
  const freePointParams = new Set<number>();
  const worse = (a: SolveStatus, b: SolveStatus): SolveStatus =>
    a === "failed" || b === "failed" ? "failed" : a === "overconstrained" || b === "overconstrained" ? "overconstrained" : "ok";

  for (const [root, ceqs] of compEqs) {
    const params = compParams.get(root)!;
    params.forEach((p, k) => (colOf[p] = k));
    let sys = new SubSystem(ceqs, params, colOf, "rcm", weight);
    let out = solveRobust(sys, x, tol, maxIt);
    if (out.status === "overconstrained" || out.redundant.length) {
      if (analyze) {
        // re-run in natural (constraint) order from the initial state so that later constraints get blamed
        const xs = Float64Array.from(params.map((p) => x[p]));
        params.forEach((p) => (x[p] = xStart[p]));
        const sysN = new SubSystem(ceqs, params, colOf, "natural", weight);
        const outN = solveRobust(sysN, x, tol, maxIt);
        if (outN.status === "failed" && out.status !== "failed") {
          params.forEach((p, k) => (x[p] = xs[k]));
        } else {
          out = outN;
          sys = sysN;
        }
      }
    }
    iterations += out.iterations;
    if (out.status === "failed") {
      // keep original geometry for this sub-system
      for (const p of params) x[p] = xInit[p];
    }
    status = worse(status, out.status);
    rankSum += out.rank;
    conflicting.push(...out.conflicting);
    redundant.push(...out.redundant);
    if (opts.computeFreePoints) {
      // determinacy with W = I
      const sysI = dragParams.size ? new SubSystem(ceqs, params, colOf, "rcm", () => 1) : sys;
      sysI.evaluate(x, true);
      sysI.factor(0, 1e-10);
      for (let k = 0; k < params.length; k++) if (!sysI.determined(k, 1e-6)) freePointParams.add(params[k]);
    }
    params.forEach((p) => (colOf[p] = -1));
  }

  // equations without unknowns: must already hold
  for (const e of constEqs) {
    for (let l = 0; l < e.idx.length; l++) e.v[l] = x[e.idx[l]];
    const f = e.f(e.v);
    if (!(Math.abs(f) <= tol)) {
      status = worse(status, "overconstrained");
      if (e.cid) conflicting.push(e.cid);
    } else if (e.cid) redundant.push(e.cid);
  }

  // residual over everything
  let residual = 0;
  for (const e of eqs) {
    for (let l = 0; l < e.idx.length; l++) e.v[l] = x[e.idx[l]];
    const f = Math.abs(e.f(e.v));
    if (f > residual || !isFinite(f)) residual = isFinite(f) ? f : Infinity;
  }

  // write back
  const out = cloneSketch(sketch);
  for (const p of out.points) {
    const b = pointBase.get(p.id)!;
    if (isFinite(x[b]) && isFinite(x[b + 1])) {
      p.x = x[b];
      p.y = x[b + 1];
    }
  }
  for (const e of out.entities) {
    if (e.type !== "circle" || !circleR.has(e.id)) continue;
    const r = x[circleR.get(e.id)!];
    if (isFinite(r)) e.radius = Math.abs(r);
  }

  const result: SolveResult = {
    sketch: out,
    status,
    residual,
    dof: opts.computeDof === false ? -1 : freeParams - rankSum,
    iterations,
  };
  const confl = uniq(conflicting);
  const red = uniq(redundant).filter((c) => !confl.includes(c));
  if (confl.length) result.conflicting = confl;
  if (red.length) result.redundant = red;
  if (B.invalid.length) result.invalid = uniq(B.invalid);
  if (opts.computeFreePoints) {
    const free: string[] = [];
    for (const p of sketch.points) {
      const b = pointBase.get(p.id)!;
      const isFree = (k: number) => !fixed[k] && (!inEq[k] || freePointParams.has(k));
      if (isFree(b) || isFree(b + 1)) free.push(p.id);
    }
    result.freePoints = free;
  }
  return result;
}

/** Convenience: degrees of freedom of a sketch without moving anything (solves a copy). */
export function sketchDof(sketch: Sketch): number {
  return solveSketch(sketch, { analyzeConflicts: false }).dof;
}

/** Is a single constraint satisfied by the current geometry (within tol * sketch scale)? */
export function isConstraintSatisfied(sketch: Sketch, c: Constraint, tol = 1e-7): boolean {
  const single: Sketch = { points: sketch.points, entities: sketch.entities, constraints: [c] };
  const B = buildSystem(single);
  if (B.invalid.length) return false;
  for (const e of B.eqs) {
    if (e.cid !== c.id) continue;
    for (let l = 0; l < e.idx.length; l++) e.v[l] = B.x[e.idx[l]];
    if (!(Math.abs(e.f(e.v)) <= tol * Math.max(1, B.scale))) return false;
  }
  return true;
}
