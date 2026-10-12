/**
 * B-rep -> neutral NURBS data for the IGES writer.
 *
 * Per face:
 *  - surface: B-spline faces are read exactly; planes become exact bilinear patches in the
 *    plane's own (u, v) parameterisation; every other surface type (cylinder, cone, sphere,
 *    torus, revolution, extrusion, offset...) is approximated by tensor-product cubic
 *    interpolation sampled in the ORIGINAL (u, v) parameterisation, so the face's pcurves stay
 *    valid on the approximation (deviation <= tolerance, adaptively refined).
 *  - trimming loops: one per wire, edges chained head to tail; each edge gives its 3D curve
 *    (BRep_Tool::Curve -> GeomConvert) and its pcurve (BRepAdaptor_Curve2d -> Geom2dConvert).
 *  - reversed faces get their U direction mirrored (surface and pcurves) so that the IGES
 *    surface normal points out of the solid and outer loops run counter-clockwise in (u, v).
 *
 * (This OCCT build has no BRepBuilderAPI_NurbsConvert / IGESControl_Writer.)
 */
import type { BRepAdaptor_Surface, TopoDS_Edge, TopoDS_Face, TopoDS_Shape } from "replicad-opencascadejs";
import type { BSplineCurve2Def, BSplineCurveDef, BSplineSurfaceDef, Vec2, Vec3 } from "../../core/types";
import type { IgesSurfaceItem, IgesTrimCurve, TrimLoop } from "../iges/igesWriter";
import {
  curveDomain,
  evalCurveN,
  evalSurface,
  interpolateCurveN,
  interpolateSurfaceGrid,
  mirrorCurve2U,
  mirrorSurfaceU,
  reverseCurve,
  surfaceDomain,
} from "../nurbs";
import type { OC } from "./loadOcct";
import { exploreAll, mapShapes, readBSplineCurve, readBSplineCurve2d, readBSplineSurface, Scope } from "./occtUtils";

export interface BrepToIgesOptions {
  /** max deviation for approximated surfaces (mm) */
  tolerance: number;
  /** label prefix for the faces */
  name?: string;
}

interface OrientedCurve extends IgesTrimCurve {
  s2?: Vec2;
  e2?: Vec2;
  s3?: Vec3;
  e3?: Vec3;
}

const linspace = (a: number, b: number, n: number) => Array.from({ length: n + 1 }, (_, i) => a + ((b - a) * i) / n);
const d3 = (a: Vec3, b: Vec3) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const d2 = (a: Vec2, b: Vec2) => Math.hypot(a[0] - b[0], a[1] - b[1]);

/** Adaptive interpolation of an arbitrary surface evaluator over [u0,u1]x[v0,v1]. */
export function approximateSurface(
  evalFn: (u: number, v: number) => Vec3,
  u0: number, u1: number, v0: number, v1: number,
  tol: number,
  maxSegments = 256,
): { surface: BSplineSurfaceDef; maxError: number } {
  let nu = 4, nv = 4;
  for (;;) {
    const us = linspace(u0, u1, nu), vs = linspace(v0, v1, nv);
    const grid = us.map((u) => vs.map((v) => evalFn(u, v)));
    const surf = interpolateSurfaceGrid(us, vs, grid);
    let errU = 0, errV = 0;
    for (let i = 0; i < nu; i++) {
      const um = (us[i] + us[i + 1]) / 2;
      for (const v of vs) errU = Math.max(errU, d3(evalSurface(surf, um, v), evalFn(um, v)));
    }
    for (let j = 0; j < nv; j++) {
      const vm = (vs[j] + vs[j + 1]) / 2;
      for (const u of us) errV = Math.max(errV, d3(evalSurface(surf, u, vm), evalFn(u, vm)));
    }
    const growU = errU > tol && nu < maxSegments;
    const growV = errV > tol && nv < maxSegments;
    if (!growU && !growV) return { surface: surf, maxError: Math.max(errU, errV) };
    if (growU) nu *= 2;
    if (growV) nv *= 2;
  }
}

function faceSurface(oc: OC, s: Scope, face: TopoDS_Face, ad: BRepAdaptor_Surface, tol: number): BSplineSurfaceDef {
  const uv = oc.BRepTools.UVBounds(face, 0, 0, 0, 0);
  const { UMin: u0, UMax: u1, VMin: v0, VMax: v1 } = uv;
  if (!(u1 > u0) || !(v1 > v0)) throw new Error("face has an empty parameter range");
  const evalFn = (u: number, v: number): Vec3 => {
    const p = ad.Value(u, v);
    const r: Vec3 = [p.X(), p.Y(), p.Z()];
    p.delete();
    return r;
  };
  const type = ad.GetType();
  const T = oc.GeomAbs_SurfaceType;
  if (type === T.GeomAbs_Plane) {
    return {
      degreeU: 1, degreeV: 1, nu: 2, nv: 2,
      knotsU: [u0, u0, u1, u1], knotsV: [v0, v0, v1, v1],
      controlPoints: [evalFn(u0, v0), evalFn(u0, v1), evalFn(u1, v0), evalFn(u1, v1)],
    };
  }
  if (type === T.GeomAbs_BSplineSurface) {
    // BRepAdaptor_Surface::BSpline() returns a copy with the face location applied
    const bs = s.add(ad.BSpline());
    return readBSplineSurface(s, bs);
  }
  return approximateSurface(evalFn, u0, u1, v0, v1, tol).surface;
}

function edgeCurve3d(oc: OC, s: Scope, edge: TopoDS_Edge): BSplineCurveDef | undefined {
  if (oc.BRep_Tool.Degenerated(edge)) return undefined;
  const env = s.envelope(oc.BRep_Tool.Curve(edge, 0, 0));
  const c = env.returnValue;
  if (!c || (c as unknown as { isNull?: () => boolean }).isNull?.()) return undefined;
  const { First: f, Last: l } = env;
  try {
    const tr = s.add(new oc.Geom_TrimmedCurve(c, f, l, true, true));
    const bs = s.add(oc.GeomConvert.CurveToBSplineCurve(tr, oc.Convert_ParameterisationType.Convert_TgtThetaOver2));
    return readBSplineCurve(s, bs);
  } catch {
    // fallback: interpolate samples
    const ad = s.add(new oc.BRepAdaptor_Curve(edge));
    const ts = linspace(ad.FirstParameter(), ad.LastParameter(), 32);
    const pts = ts.map((t) => {
      const p = ad.Value(t);
      const r = [p.X(), p.Y(), p.Z()];
      p.delete();
      return r;
    });
    const fit = interpolateCurveN(ts, pts);
    return { degree: fit.degree, knots: fit.knots, controlPoints: fit.controlPoints as Vec3[] };
  }
}

function edgeCurve2d(oc: OC, s: Scope, edge: TopoDS_Edge, face: TopoDS_Face): BSplineCurve2Def | undefined {
  const ad = s.add(new oc.BRepAdaptor_Curve2d(edge, face));
  const f = ad.FirstParameter(), l = ad.LastParameter();
  try {
    const c = s.add(ad.Curve());
    const tr = s.add(new oc.Geom2d_TrimmedCurve(c, f, l, true, true));
    const bs = s.add(oc.Geom2dConvert.CurveToBSplineCurve(tr, oc.Convert_ParameterisationType.Convert_TgtThetaOver2));
    return readBSplineCurve2d(s, bs);
  } catch {
    try {
      const ts = linspace(f, l, 32);
      const pts = ts.map((t) => {
        const p = ad.Value(t);
        const r = [p.X(), p.Y()];
        p.delete();
        return r;
      });
      const fit = interpolateCurveN(ts, pts);
      return { degree: fit.degree, knots: fit.knots, controlPoints: fit.controlPoints as Vec2[] };
    } catch {
      return undefined;
    }
  }
}

function endpoints<T extends number[]>(c: { degree: number; knots: readonly number[]; controlPoints: readonly T[] }): [T, T] {
  // clamped curves interpolate their end control points
  return [c.controlPoints[0], c.controlPoints[c.controlPoints.length - 1]];
}

/** Chain curves head-to-tail (explorer order is usually right; this makes it certain). */
function chainLoop(curves: OrientedCurve[]): OrientedCurve[] {
  if (curves.length <= 2) return curves;
  const left = curves.slice(1);
  const out = [curves[0]];
  while (left.length) {
    const last = out[out.length - 1];
    let best = 0, bestD = Infinity;
    left.forEach((c, i) => {
      const d = last.e2 && c.s2 ? d2(last.e2, c.s2) : last.e3 && c.s3 ? d3(last.e3, c.s3) : Infinity;
      if (d < bestD) { bestD = d; best = i; }
    });
    out.push(left.splice(best, 1)[0]);
  }
  return out;
}

function faceToIges(oc: OC, face: TopoDS_Face, opts: BrepToIgesOptions, label?: string): IgesSurfaceItem {
  const s = new Scope();
  try {
    const ad = s.add(new oc.BRepAdaptor_Surface(face, true));
    let surface = faceSurface(oc, s, face, ad, opts.tolerance);
    const reversed = face.Orientation() === oc.TopAbs_Orientation.TopAbs_REVERSED;
    const outerWire = s.add(oc.BRepTools.OuterWire(face));
    const wires = mapShapes(oc, s, face, oc.TopAbs_ShapeEnum.TopAbs_WIRE).shapes;
    let outer: TrimLoop | null = null;
    const inner: TrimLoop[] = [];
    const { u: [ua, ub] } = surfaceDomain(surface);
    for (const w of wires) {
      const curves: OrientedCurve[] = [];
      for (const e of exploreAll(oc, s, w, oc.TopAbs_ShapeEnum.TopAbs_EDGE)) {
        const edge = s.add(oc.TopoDS.Edge(e));
        const rev = edge.Orientation() === oc.TopAbs_Orientation.TopAbs_REVERSED;
        let c3 = edgeCurve3d(oc, s, edge);
        let c2 = edgeCurve2d(oc, s, edge, face);
        if (rev) {
          if (c3) c3 = reverseCurve(c3);
          if (c2) c2 = reverseCurve(c2);
        }
        if (reversed && c2) c2 = mirrorCurve2U(c2, ua, ub);
        if (!c3 && !c2) continue;
        const oc_: OrientedCurve = { curve3d: c3, curve2d: c2 };
        if (c2) [oc_.s2, oc_.e2] = endpoints(c2);
        if (c3) [oc_.s3, oc_.e3] = endpoints(c3);
        curves.push(oc_);
      }
      if (!curves.length) continue;
      const loop: TrimLoop = chainLoop(curves).map(({ curve3d, curve2d }) => ({ curve3d, curve2d }));
      if (!outer && w.IsSame(outerWire)) outer = loop;
      else inner.push(loop);
    }
    if (reversed) surface = mirrorSurfaceU(surface);
    if (!outer) {
      if (!inner.length) return { surface, name: label };
      outer = inner.shift()!;
    }
    return { surface, trims: { outer, inner }, name: label };
  } finally {
    s.dispose();
  }
}

/** Convert all faces of a shape (MapShapes order) into IGES surface items. */
export function shapeToIgesSurfaces(oc: OC, shape: TopoDS_Shape, opts: BrepToIgesOptions): IgesSurfaceItem[] {
  const s = new Scope();
  try {
    const faces = mapShapes(oc, s, shape, oc.TopAbs_ShapeEnum.TopAbs_FACE).shapes;
    return faces.map((f, i) => {
      const face = s.add(oc.TopoDS.Face(f));
      try {
        return faceToIges(oc, face, opts, opts.name);
      } catch (e) {
        throw new Error(`face ${i}: ${e instanceof Error ? e.message : String(e)}`);
      }
    });
  } finally {
    s.dispose();
  }
}

/**
 * Max distance between S(B(t)) and the 3D trimming curve C over all loops of an item
 * (consistency of surface, pcurves and 3D curves; used by tests and diagnostics).
 */
export function trimConsistency(item: IgesSurfaceItem, samples = 16): number {
  let maxErr = 0;
  const loops = item.trims ? [item.trims.outer, ...item.trims.inner] : [];
  for (const loop of loops)
    for (const c of loop) {
      if (!c.curve2d || !c.curve3d) continue;
      const [a2, b2] = curveDomain(c.curve2d);
      for (let k = 0; k <= samples; k++) {
        const uv = evalCurveAt(c.curve2d, a2 + ((b2 - a2) * k) / samples);
        const ps = evalSurface(item.surface, uv[0], uv[1]);
        maxErr = Math.max(maxErr, distanceToCurve(c.curve3d, ps));
      }
    }
  return maxErr;
}

const evalCurveAt = (c: { degree: number; knots: number[]; controlPoints: number[][]; weights?: number[] }, t: number) =>
  evalCurveN(c.degree, c.knots, c.controlPoints, c.weights, t);

/** Distance from p to a curve: coarse sampling + golden-section refinement. */
export function distanceToCurve(c: BSplineCurveDef, p: Vec3, n = 64): number {
  const [a, b] = curveDomain(c);
  const f = (t: number) => d3(evalCurveAt(c, t) as Vec3, p);
  let bestK = 0, best = Infinity;
  for (let k = 0; k <= n; k++) {
    const d = f(a + ((b - a) * k) / n);
    if (d < best) { best = d; bestK = k; }
  }
  let lo = a + ((b - a) * Math.max(0, bestK - 1)) / n, hi = a + ((b - a) * Math.min(n, bestK + 1)) / n;
  const g = (Math.sqrt(5) - 1) / 2;
  let x1 = hi - g * (hi - lo), x2 = lo + g * (hi - lo), f1 = f(x1), f2 = f(x2);
  for (let it = 0; it < 60; it++) {
    if (f1 < f2) { hi = x2; x2 = x1; f2 = f1; x1 = hi - g * (hi - lo); f1 = f(x1); }
    else { lo = x1; x1 = x2; f1 = f2; x2 = lo + g * (hi - lo); f2 = f(x2); }
  }
  return Math.min(best, f1, f2);
}
