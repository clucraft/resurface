/**
 * Helpers around the OCCT embind API: memory scopes, error conversion, geometry builders,
 * B-spline conversion and deterministic sub-shape indexing (TopExp::MapShapes order).
 */
import type {
  Geom2d_BSplineCurve,
  Geom_BSplineCurve,
  Geom_BSplineSurface,
  TopAbs_ShapeEnum,
  TopoDS_Shape,
} from "replicad-opencascadejs";
import type { BSplineCurve2Def, BSplineCurveDef, BSplineSurfaceDef, Frame, Vec2, Vec3 } from "../../core/types";
import { toDistinctKnots, toFullKnots } from "../nurbs";
import type { OC } from "./loadOcct";

export interface Deletable {
  delete(): void;
}

/**
 * Collects embind objects and deletes them (in reverse order) on `dispose()`.
 * Everything created inside a kernel operation goes into a scope, except the result shape
 * that is handed to the registry.
 */
export class Scope {
  private items: Deletable[] = [];
  add<T extends Deletable>(o: T): T {
    if (o && typeof o.delete === "function") this.items.push(o);
    return o;
  }
  /** Stop tracking `o` (it survives the scope, e.g. a result handed to the registry). */
  keep<T>(o: T): T {
    const i = this.items.lastIndexOf(o as unknown as Deletable);
    if (i >= 0) this.items.splice(i, 1);
    return o;
  }
  /** Track an OCCT "result envelope" (object with owned Handle fields and Symbol.dispose). */
  envelope<T extends object>(env: T): T {
    this.items.push({ delete: () => disposeEnvelope(env) });
    return env;
  }
  dispose(): void {
    for (let i = this.items.length - 1; i >= 0; i--) {
      try {
        this.items[i].delete();
      } catch {
        /* already deleted */
      }
    }
    this.items.length = 0;
  }
}

export function disposeEnvelope(env: object): void {
  const symDispose = (Symbol as unknown as { dispose?: symbol }).dispose;
  const fn = symDispose ? (env as Record<symbol, unknown>)[symDispose] : undefined;
  if (typeof fn === "function") {
    (fn as () => void).call(env);
    return;
  }
  for (const v of Object.values(env)) {
    if (v && typeof (v as Deletable).delete === "function") {
      try {
        (v as Deletable).delete();
      } catch {
        /* ignore */
      }
    }
  }
}

/** Convert anything thrown by OCCT/embind into an Error with a readable message. */
export function occtError(oc: OC, e: unknown, context: string): Error {
  if (e instanceof Error) {
    const err = new Error(e.message.startsWith(context) ? e.message : `${context}: ${e.message}`);
    return err;
  }
  const WasmException = (globalThis as unknown as { WebAssembly?: { Exception?: new (...a: unknown[]) => unknown } }).WebAssembly?.Exception;
  if (WasmException && e instanceof WasmException) {
    try {
      const [type, msg] = (oc.getExceptionMessage as (x: unknown) => [string, string | undefined])(e);
      return new Error(`${context}: OCCT ${type}${msg ? ` (${msg})` : ""}`);
    } catch {
      return new Error(`${context}: OCCT exception`);
    } finally {
      try {
        (oc.decrementExceptionRefcount as (x: unknown) => void)(e);
      } catch {
        /* ignore */
      }
    }
  }
  if (typeof e === "number") return new Error(`${context}: OCCT exception (code ${e})`);
  return new Error(`${context}: ${String(e)}`);
}

export const HASH_MAX = 2147483647;

// ---------------------------------------------------------------------------
// gp builders (all registered in the scope)
// ---------------------------------------------------------------------------

export class Gp {
  constructor(private oc: OC, private s: Scope) {}
  pnt(p: Vec3) {
    return this.s.add(new this.oc.gp_Pnt(p[0], p[1], p[2]));
  }
  pnt2(p: Vec2) {
    return this.s.add(new this.oc.gp_Pnt2d(p[0], p[1]));
  }
  dir(d: Vec3) {
    const l = Math.hypot(d[0], d[1], d[2]);
    if (!(l > 1e-300)) throw new Error("zero-length direction vector");
    return this.s.add(new this.oc.gp_Dir(d[0] / l, d[1] / l, d[2] / l));
  }
  vec(v: Vec3) {
    return this.s.add(new this.oc.gp_Vec(v[0], v[1], v[2]));
  }
  ax1(o: Vec3, d: Vec3) {
    return this.s.add(new this.oc.gp_Ax1(this.pnt(o), this.dir(d)));
  }
  ax2(o: Vec3, n: Vec3, x?: Vec3) {
    return this.s.add(x ? new this.oc.gp_Ax2(this.pnt(o), this.dir(n), this.dir(x)) : new this.oc.gp_Ax2(this.pnt(o), this.dir(n)));
  }
  ax3(o: Vec3, n: Vec3, x?: Vec3) {
    return this.s.add(x ? new this.oc.gp_Ax3(this.pnt(o), this.dir(n), this.dir(x)) : new this.oc.gp_Ax3(this.pnt(o), this.dir(n)));
  }
  frameAx3(f: Frame) {
    return this.ax3(f.origin, f.zAxis, f.xAxis);
  }
  progress() {
    return this.s.add(new this.oc.Message_ProgressRange());
  }
  arrDouble(values: readonly number[]) {
    const a = this.s.add(new this.oc.NCollection_Array1_double(1, values.length));
    values.forEach((v, i) => a.SetValue(i + 1, v));
    return a;
  }
  arrInt(values: readonly number[]) {
    const a = this.s.add(new this.oc.NCollection_Array1_int(1, values.length));
    values.forEach((v, i) => a.SetValue(i + 1, v));
    return a;
  }
  arrPnt(points: readonly Vec3[]) {
    const a = this.s.add(new this.oc.NCollection_Array1_gp_Pnt(1, points.length));
    points.forEach((p, i) => a.SetValue(i + 1, this.pnt(p)));
    return a;
  }
  arrPnt2(points: readonly Vec2[]) {
    const a = this.s.add(new this.oc.NCollection_Array1_gp_Pnt2d(1, points.length));
    points.forEach((p, i) => a.SetValue(i + 1, this.pnt2(p)));
    return a;
  }
  shapeList(shapes: readonly TopoDS_Shape[]) {
    const l = this.s.add(new this.oc.NCollection_List_TopoDS_Shape());
    for (const sh of shapes) l.Append(sh);
    return l;
  }
}

// ---------------------------------------------------------------------------
// B-spline construction / extraction
// ---------------------------------------------------------------------------

export function makeBSplineCurve(oc: OC, gp: Gp, s: Scope, c: BSplineCurveDef): Geom_BSplineCurve {
  const { knots, mults } = toDistinctKnots(c.knots);
  const poles = gp.arrPnt(c.controlPoints);
  const K = gp.arrDouble(knots), M = gp.arrInt(mults);
  return s.add(
    c.weights
      ? new oc.Geom_BSplineCurve(poles, gp.arrDouble(c.weights), K, M, c.degree, false, true)
      : new oc.Geom_BSplineCurve(poles, K, M, c.degree, false),
  );
}

export function makeBSplineCurve2d(oc: OC, gp: Gp, s: Scope, c: BSplineCurve2Def): Geom2d_BSplineCurve {
  const { knots, mults } = toDistinctKnots(c.knots);
  const poles = gp.arrPnt2(c.controlPoints);
  const K = gp.arrDouble(knots), M = gp.arrInt(mults);
  return s.add(
    c.weights
      ? new oc.Geom2d_BSplineCurve(poles, gp.arrDouble(c.weights), K, M, c.degree, false)
      : new oc.Geom2d_BSplineCurve(poles, K, M, c.degree, false),
  );
}

export function makeBSplineSurface(oc: OC, gp: Gp, s: Scope, d: BSplineSurfaceDef): Geom_BSplineSurface {
  const ku = toDistinctKnots(d.knotsU), kv = toDistinctKnots(d.knotsV);
  const poles = s.add(new oc.NCollection_Array2_gp_Pnt(1, d.nu, 1, d.nv));
  for (let i = 0; i < d.nu; i++) for (let j = 0; j < d.nv; j++) poles.SetValue(i + 1, j + 1, gp.pnt(d.controlPoints[i * d.nv + j]));
  const UK = gp.arrDouble(ku.knots), VK = gp.arrDouble(kv.knots), UM = gp.arrInt(ku.mults), VM = gp.arrInt(kv.mults);
  if (d.weights) {
    const w = s.add(new oc.NCollection_Array2_double(1, d.nu, 1, d.nv));
    for (let i = 0; i < d.nu; i++) for (let j = 0; j < d.nv; j++) w.SetValue(i + 1, j + 1, d.weights[i * d.nv + j]);
    return s.add(new oc.Geom_BSplineSurface(poles, w, UK, VK, UM, VM, d.degreeU, d.degreeV, false, false));
  }
  return s.add(new oc.Geom_BSplineSurface(poles, UK, VK, UM, VM, d.degreeU, d.degreeV, false, false));
}

/** Read a (non-periodic) Geom_BSplineCurve into a plain definition. */
export function readBSplineCurve(s: Scope, c: Geom_BSplineCurve): BSplineCurveDef {
  if (c.IsPeriodic()) c.SetNotPeriodic();
  const n = c.NbPoles();
  const rational = c.IsRational();
  const cps: Vec3[] = [];
  const w: number[] = [];
  for (let i = 1; i <= n; i++) {
    const p = s.add(c.Pole(i));
    cps.push([p.X(), p.Y(), p.Z()]);
    if (rational) w.push(c.Weight(i));
  }
  const knots: number[] = [], mults: number[] = [];
  for (let i = 1; i <= c.NbKnots(); i++) {
    knots.push(c.Knot(i));
    mults.push(c.Multiplicity(i));
  }
  return { degree: c.Degree(), knots: toFullKnots(knots, mults), controlPoints: cps, weights: rational ? w : undefined };
}

export function readBSplineCurve2d(s: Scope, c: Geom2d_BSplineCurve): BSplineCurve2Def {
  if (c.IsPeriodic()) c.SetNotPeriodic();
  const n = c.NbPoles();
  const rational = c.IsRational();
  const cps: Vec2[] = [];
  const w: number[] = [];
  for (let i = 1; i <= n; i++) {
    const p = s.add(c.Pole(i));
    cps.push([p.X(), p.Y()]);
    if (rational) w.push(c.Weight(i));
  }
  const knots: number[] = [], mults: number[] = [];
  for (let i = 1; i <= c.NbKnots(); i++) {
    knots.push(c.Knot(i));
    mults.push(c.Multiplicity(i));
  }
  return { degree: c.Degree(), knots: toFullKnots(knots, mults), controlPoints: cps, weights: rational ? w : undefined };
}

export function readBSplineSurface(s: Scope, b: Geom_BSplineSurface): BSplineSurfaceDef {
  if (b.IsUPeriodic()) b.SetUNotPeriodic();
  if (b.IsVPeriodic()) b.SetVNotPeriodic();
  const nu = b.NbUPoles(), nv = b.NbVPoles();
  const rational = b.IsURational() || b.IsVRational();
  const cps: Vec3[] = new Array(nu * nv);
  const w: number[] = rational ? new Array(nu * nv) : [];
  for (let i = 1; i <= nu; i++)
    for (let j = 1; j <= nv; j++) {
      const p = s.add(b.Pole(i, j));
      cps[(i - 1) * nv + (j - 1)] = [p.X(), p.Y(), p.Z()];
      if (rational) w[(i - 1) * nv + (j - 1)] = b.Weight(i, j);
    }
  const uk: number[] = [], um: number[] = [], vk: number[] = [], vm: number[] = [];
  for (let i = 1; i <= b.NbUKnots(); i++) { uk.push(b.UKnot(i)); um.push(b.UMultiplicity(i)); }
  for (let i = 1; i <= b.NbVKnots(); i++) { vk.push(b.VKnot(i)); vm.push(b.VMultiplicity(i)); }
  return {
    degreeU: b.UDegree(),
    degreeV: b.VDegree(),
    knotsU: toFullKnots(uk, um),
    knotsV: toFullKnots(vk, vm),
    nu,
    nv,
    controlPoints: cps,
    weights: rational ? w : undefined,
  };
}

// ---------------------------------------------------------------------------
// Topology indexing
// ---------------------------------------------------------------------------

/**
 * Unique sub-shapes of `type` in TopExp::MapShapes order (explorer order, first occurrence,
 * de-duplicated with IsSame). Returned shapes are owned by the scope.
 */
export function mapShapes(oc: OC, s: Scope, shape: TopoDS_Shape, type: TopAbs_ShapeEnum): { shapes: TopoDS_Shape[]; hashes: number[] } {
  const shapes: TopoDS_Shape[] = [];
  const hashes: number[] = [];
  const buckets = new Map<number, number[]>();
  const ex = s.add(new oc.TopExp_Explorer(shape, type, oc.TopAbs_ShapeEnum.TopAbs_SHAPE));
  for (; ex.More(); ex.Next()) {
    const cur = ex.Current();
    const h = oc.ReplicadShapeHasher.HashCode(cur, HASH_MAX);
    const b = buckets.get(h);
    if (b && b.some((i) => shapes[i].IsSame(cur))) {
      cur.delete();
      continue;
    }
    s.add(cur);
    const idx = shapes.push(cur) - 1;
    hashes.push(h);
    if (b) b.push(idx);
    else buckets.set(h, [idx]);
  }
  return { shapes, hashes };
}

/** All sub-shapes of `type` in explorer order, without de-duplication (e.g. seam edges twice). */
export function exploreAll(oc: OC, s: Scope, shape: TopoDS_Shape, type: TopAbs_ShapeEnum): TopoDS_Shape[] {
  const out: TopoDS_Shape[] = [];
  const ex = s.add(new oc.TopExp_Explorer(shape, type, oc.TopAbs_ShapeEnum.TopAbs_SHAPE));
  for (; ex.More(); ex.Next()) out.push(s.add(ex.Current()));
  return out;
}

export function countShapes(oc: OC, shape: TopoDS_Shape, type: TopAbs_ShapeEnum): number {
  const s = new Scope();
  try {
    return mapShapes(oc, s, shape, type).shapes.length;
  } finally {
    s.dispose();
  }
}
