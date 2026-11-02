/**
 * Builds an OCCT shape from a converter B-rep plan (src/convert/brepPlan.ts): shared vertices
 * and edges (line / arc / circle / spline), faces on analytic surfaces assembled by ShapeFix_Face
 * (pcurves, seams, wire orientation), then sewn into shells and closed shells into solids.
 */
import type { Geom_Surface, TopoDS_Edge, TopoDS_Face, TopoDS_Shape, TopoDS_Vertex, TopoDS_Wire } from "replicad-opencascadejs";
import type { Vec3 } from "../../core/types";
import type { BrepPlan, FaceKind, PlanFace } from "../../convert/brepPlan";
import { mapShapes, HASH_MAX, type Gp, type Scope } from "./occtUtils";
import type { OC } from "./loadOcct";

export interface PlanBuild {
  shape: TopoDS_Shape;
  /** kind of every face of `shape` in TopExp::MapShapes order */
  faceKinds: FaceKind[];
  /** plan faces that could not be built (left as holes) */
  failedFaces: number[];
  /** plan faces built on a surface but not valid (wire/pcurve problems) */
  invalidFaces: number[];
  failedEdges: number;
  freeEdges: number;
}

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const len = (a: Vec3) => Math.hypot(a[0], a[1], a[2]);

export function buildFromPlan(oc: OC, s: Scope, gp: Gp, plan: BrepPlan, onProgress?: (f: number, msg: string) => void): PlanBuild {
  const T = oc.TopAbs_ShapeEnum;
  const tol = Math.max(plan.tolerance, 1e-7);
  const V: TopoDS_Vertex[] = plan.vertices.map((p) => s.add(s.add(new oc.BRepBuilderAPI_MakeVertex(gp.pnt(p))).Vertex()));

  // --- edges
  let failedEdges = 0;
  /** polyline plan edges become several line edges (with their own inner vertices) */
  const polyEdges = new Map<number, TopoDS_Edge[]>();
  const E: (TopoDS_Edge | null)[] = plan.edges.map((e, ei) => {
    const p0 = plan.vertices[e.v0], p1 = plan.vertices[e.v1];
    const line = (): TopoDS_Edge | null => {
      if (e.v0 === e.v1 || len(sub(p0, p1)) < 1e-9) return null;
      const mk = s.add(new oc.BRepBuilderAPI_MakeEdge(V[e.v0], V[e.v1]));
      return mk.IsDone() ? s.add(mk.Edge()) : null;
    };
    try {
      const c = e.curve;
      switch (c.type) {
        case "line":
          return line() ?? (failedEdges++, null);
        case "arc": {
          const arc = s.add(new oc.GC_MakeArcOfCircle(gp.pnt(p0), gp.pnt(c.mid), gp.pnt(p1)));
          if (!arc.IsDone()) return line();
          const mk = s.add(new oc.BRepBuilderAPI_MakeEdge(s.add(arc.Value()), V[e.v0], V[e.v1]));
          return mk.IsDone() ? s.add(mk.Edge()) : line();
        }
        case "circle": {
          const x = sub(p0, c.center);
          const ax = gp.ax2(c.center, c.normal, len(x) > 1e-12 ? x : undefined);
          const mk = s.add(new oc.BRepBuilderAPI_MakeEdge(s.add(new oc.gp_Circ(ax, c.radius))));
          return mk.IsDone() ? s.add(mk.Edge()) : (failedEdges++, null);
        }
        case "polyline": {
          const vs: TopoDS_Vertex[] = [V[e.v0]];
          if (c.points.length < 2) return line();
          for (let i = 1; i + 1 < c.points.length; i++) vs.push(s.add(s.add(new oc.BRepBuilderAPI_MakeVertex(gp.pnt(c.points[i]))).Vertex()));
          vs.push(V[e.v1]);
          const list: TopoDS_Edge[] = [];
          for (let i = 0; i + 1 < vs.length; i++) {
            const mk = s.add(new oc.BRepBuilderAPI_MakeEdge(vs[i], vs[i + 1]));
            if (mk.IsDone()) list.push(s.add(mk.Edge()));
          }
          if (!list.length) return line();
          polyEdges.set(ei, list);
          return list[0];
        }
        case "spline": {
          // ends exactly on the (possibly snapped) plan vertices
          const pts = c.points.slice();
          if (pts.length < 2) return line();
          pts[0] = p0;
          if (!c.closed) pts[pts.length - 1] = p1;
          const arr = s.add(new oc.NCollection_HArray1_gp_Pnt(gp.arrPnt(pts)));
          const ip = s.add(new oc.GeomAPI_Interpolate(arr, c.closed, 1e-9));
          ip.Perform();
          if (!ip.IsDone()) return c.closed ? (failedEdges++, null) : line();
          const curve = s.add(ip.Curve());
          const mk = s.add(c.closed ? new oc.BRepBuilderAPI_MakeEdge(curve) : new oc.BRepBuilderAPI_MakeEdge(curve, V[e.v0], V[e.v1]));
          return mk.IsDone() ? s.add(mk.Edge()) : c.closed ? (failedEdges++, null) : line();
        }
      }
    } catch {
      failedEdges++;
      return null;
    }
  });

  // --- faces
  const built: { face: TopoDS_Face; kind: FaceKind; plan: number }[] = [];
  const failedFaces: number[] = [];
  const invalidFaces: number[] = [];
  const surfaceOf = (f: PlanFace): Geom_Surface | null => {
    const sf = f.surface!, pl = f.placement!;
    const ax3 = () => gp.ax3(pl.origin, pl.z, pl.x);
    const fromFace = (face: TopoDS_Face) => s.add(oc.BRep_Tool.Surface(face));
    switch (sf.kind) {
      case "plane":
        return fromFace(s.add(s.add(new oc.BRepBuilderAPI_MakeFace(s.add(new oc.gp_Pln(ax3())))).Face()));
      case "cylinder":
        return s.add(new oc.Geom_CylindricalSurface(ax3(), sf.radius));
      case "cone":
        return s.add(new oc.Geom_ConicalSurface(ax3(), sf.halfAngle, 0));
      case "sphere":
        return fromFace(s.add(s.add(new oc.BRepBuilderAPI_MakeFace(s.add(new oc.gp_Sphere(ax3(), sf.radius)))).Face()));
      case "torus": {
        const mk = s.add(new oc.BRepPrimAPI_MakeTorus(gp.ax2(pl.origin, pl.z, pl.x), sf.major, sf.minor));
        const faces = mapShapes(oc, s, s.add(mk.Shape()), T.TopAbs_FACE).shapes;
        if (!faces.length) return null;
        return fromFace(s.add(oc.TopoDS.Face(faces[0])));
      }
    }
  };
  const wireOf = (loop: PlanFace["loops"][number]): TopoDS_Wire | null => {
    const edges: TopoDS_Edge[] = [];
    const rev = (e: TopoDS_Edge) => s.add(oc.TopoDS.Edge(s.add(e.Reversed())));
    for (const u of loop) {
      const e = E[u.edge];
      if (!e) return null;
      const poly = polyEdges.get(u.edge);
      if (poly) edges.push(...(u.reversed ? poly.slice().reverse().map(rev) : poly));
      else edges.push(u.reversed ? rev(e) : e);
    }
    if (!edges.length) return null;
    const mw = s.add(new oc.BRepBuilderAPI_MakeWire());
    for (const e of edges) mw.Add(e);
    if (mw.IsDone()) return s.add(mw.Wire());
    // disconnected (shouldn't happen with shared vertices): let ShapeFix sort it out
    const w = s.add(new oc.TopoDS_Wire());
    const b = s.add(new oc.TopoDS_Builder());
    b.MakeWire(w);
    for (const e of edges) b.Add(w, e);
    return w;
  };

  const areaMatches = (face: TopoDS_Face, meshArea: number): boolean => {
    const p = s.add(new oc.GProp_GProps());
    oc.BRepGProp.SurfaceProperties(face, p, false, false);
    const a = Math.abs(p.Mass()), slack = 50 * tol * tol;
    return a <= 1.4 * meshArea + slack && a >= 0.7 * meshArea - slack;
  };

  plan.faces.forEach((f, fi) => {
    if ((fi & 127) === 0) onProgress?.(fi / plan.faces.length, "Building faces");
    try {
      const wires = f.loops.map(wireOf);
      if (!wires.length || wires.some((w) => !w)) {
        failedFaces.push(fi);
        return;
      }
      let face: TopoDS_Face;
      let invalid = false;
      if (f.band && f.surface) {
        const surf = surfaceOf(f);
        const mf = surf && s.add(new oc.BRepBuilderAPI_MakeFace(surf, 0, 2 * Math.PI, f.band.v0, f.band.v1, 1e-7));
        if (!mf || !mf.IsDone()) {
          failedFaces.push(fi);
          return;
        }
        face = s.add(mf.Face());
        if (!f.forward) face = s.add(oc.TopoDS.Face(s.add(face.Reversed())));
      } else if (!f.surface) {
        const mf = s.add(new oc.BRepBuilderAPI_MakeFace(wires[0]!, true));
        if (!mf.IsDone()) {
          failedFaces.push(fi);
          return;
        }
        face = s.add(mf.Face());
      } else {
        const surf = surfaceOf(f);
        if (!surf) {
          failedFaces.push(fi);
          return;
        }
        // build along the surface normal (loops reversed when the outward side is the back) and
        // flip the finished face: ShapeFix on a reversed face leaves the loops inside out
        const fix = s.add(new oc.ShapeFix_Face());
        fix.Init(surf, tol, true);
        for (const w of wires) fix.Add(f.forward ? w! : s.add(oc.TopoDS.Wire(s.add(w!.Reversed()))));
        fix.Perform(gp.progress());
        face = s.add(fix.Face());
        if (face.IsNull()) {
          failedFaces.push(fi);
          return;
        }
        if (!f.forward) face = s.add(oc.TopoDS.Face(s.add(face.Reversed())));
        // ShapeFix can leave pcurves (loops crossing the seam of a periodic surface) outside their
        // edge tolerances: same-parameter before judging the face; one still invalid is reported
        // so the caller can fall back to facets
        const valid = () => s.add(new oc.BRepCheck_Analyzer(face, true, false, false)).IsValid();
        if (!valid()) {
          oc.BRepLib.SameParameter(face, tol, true);
          oc.BRepLib.UpdateTolerances(face, true);
          invalid = !valid();
        }
      }
      // a valid face can still cover the wrong part of its surface (a cap closing a hole, the
      // complement of a band): compare with the area of the triangles it replaces
      if (!invalid && f.area !== undefined && f.surface) invalid = !areaMatches(face, f.area);
      if (invalid) invalidFaces.push(fi);
      built.push({ face, kind: f.kind, plan: fi });
    } catch {
      failedFaces.push(fi);
    }
  });
  if (!built.length) throw new Error("no face could be built");

  // --- sew
  onProgress?.(0.9, "Sewing faces");
  const sew = s.add(new oc.BRepBuilderAPI_Sewing(Math.max(2 * tol, 1e-6), true, true, true, false));
  for (const b of built) sew.Add(b.face);
  sew.Perform(gp.progress());
  const sewed: TopoDS_Shape = s.add(sew.SewedShape());
  const freeEdges = sew.NbFreeEdges();
  // kinds of the sewn faces
  const kindByHash = new Map<number, { face: TopoDS_Shape; kind: FaceKind }[]>();
  for (const b of built) {
    const m: TopoDS_Shape = sew.IsModified(b.face) ? s.add(sew.Modified(b.face)) : b.face;
    const h = oc.ReplicadShapeHasher.HashCode(m, HASH_MAX);
    const l = kindByHash.get(h);
    if (l) l.push({ face: m, kind: b.kind });
    else kindByHash.set(h, [{ face: m, kind: b.kind }]);
  }
  const G = oc.GeomAbs_SurfaceType;
  const kindOf = (f: TopoDS_Shape, h: number): FaceKind => {
    const k = kindByHash.get(h)?.find((x) => x.face.IsSame(f))?.kind;
    if (k) return k;
    // the final repair may have replaced the face: classify by its surface type
    const ad = s.add(new oc.BRepAdaptor_Surface(s.add(oc.TopoDS.Face(f)), true));
    const t = ad.GetType();
    return t === G.GeomAbs_Plane ? "plane" : t === G.GeomAbs_Cylinder ? "cylinder" : t === G.GeomAbs_Cone ? "cone" : t === G.GeomAbs_Sphere ? "sphere" : t === G.GeomAbs_Torus ? "torus" : "faceted";
  };

  // closed shells → solids
  const parts: TopoDS_Shape[] = [];
  for (const sh of mapShapes(oc, s, sewed, T.TopAbs_SHELL).shapes) {
    if (oc.BRep_Tool.IsClosed(sh)) {
      const fix = s.add(new oc.ShapeFix_Solid());
      let solid: TopoDS_Shape = s.add(fix.SolidFromShell(s.add(oc.TopoDS.Shell(sh))));
      // final repair pass (face/wire orientation, tolerances)
      const fix2 = s.add(new oc.ShapeFix_Solid(s.add(oc.TopoDS.Solid(solid))));
      fix2.Perform(gp.progress());
      solid = s.add(fix2.Solid());
      const p = s.add(new oc.GProp_GProps());
      oc.BRepGProp.VolumeProperties(solid, p, false, false, false);
      if (p.Mass() < 0) solid = s.add(solid.Reversed());
      parts.push(solid);
    } else parts.push(sh);
  }
  const free = s.add(new oc.TopExp_Explorer(sewed, T.TopAbs_FACE, T.TopAbs_SHELL));
  for (; free.More(); free.Next()) parts.push(s.add(free.Current()));
  let shape: TopoDS_Shape;
  if (parts.length === 1) shape = parts[0];
  else {
    const comp = s.add(new oc.TopoDS_Compound());
    const b = s.add(new oc.TopoDS_Builder());
    b.MakeCompound(comp);
    for (const p of parts) b.Add(comp, p);
    shape = comp;
  }
  const finalFaces = mapShapes(oc, s, shape, T.TopAbs_FACE);
  const faceKinds = finalFaces.shapes.map((f, i) => kindOf(f, finalFaces.hashes[i]));
  return { shape, faceKinds, failedFaces, invalidFaces, failedEdges, freeEdges };
}
