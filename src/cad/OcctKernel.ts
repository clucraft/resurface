/**
 * OcctKernel — B-rep modelling on top of OpenCASCADE (WebAssembly).
 *
 * Shapes live in a registry (ShapeId -> TopoDS_Shape); every method takes and returns plain,
 * serialisable data so the kernel can run inside a Web Worker (see kernel.worker.ts /
 * KernelClient.ts). All temporary embind objects are released through `Scope`s.
 *
 * Sub-shape ids (faces/edges) follow TopExp::MapShapes order (explorer order, de-duplicated),
 * consistently in `tessellate`, `fillet`, `chamfer` and `shell`.
 */
import type { TopoDS_Edge, TopoDS_Face, TopoDS_Shape, TopoDS_Wire } from "replicad-opencascadejs";
import type { BSplineCurveDef, Frame, Mat4, PrimitiveDef, Vec2, Vec3 } from "../core/types";
import { addScaled, anyPerpendicular, cross, dot, frameToWorld, length, negate, normalize, scale } from "../core/vec";
import { writeIGES } from "./iges/igesWriter";
import type {
  BooleanOp,
  BooleanOptions,
  BSplineFaceArgs,
  ExportItem,
  ExtrudeArgs,
  ExtrudeWireArgs,
  IgesExportOptions,
  KernelApi,
  LoftArgs,
  PatchesToShapeArgs,
  PatternOptions,
  PrimitiveBounds,
  RevolveArgs,
  ShapeId,
  ShapeInfo,
  ShapeKind,
  StlExportOptions,
  SweepArgs,
  TessellateOptions,
  Tessellation,
  TessellationEdge,
} from "./kernelTypes";
import { checkCurveDef, checkSurfaceDef } from "./nurbs";
import { shapeToIgesSurfaces } from "./occt/brepToIges";
import { loadOcct, type LoadOcctOptions, type OC } from "./occt/loadOcct";
import { Gp, makeBSplineCurve, makeBSplineSurface, mapShapes, occtError, Scope } from "./occt/occtUtils";
import { arcGeometry, buildRegions, reverseLoop, snapLoop, snapWire, type KSegment, type Region } from "./profile";
import { writeBinarySTL } from "./stl";

const DEG = Math.PI / 180;

export class OcctKernel implements KernelApi {
  private shapes = new Map<ShapeId, TopoDS_Shape>();
  private nextId = 1;
  private tmpCounter = 0;

  constructor(readonly oc: OC) {}

  static async create(opts?: LoadOcctOptions): Promise<OcctKernel> {
    return new OcctKernel(await loadOcct(opts));
  }

  // -------------------------------------------------------------------------
  // Registry
  // -------------------------------------------------------------------------

  private register(shape: TopoDS_Shape): ShapeId {
    if (shape.IsNull()) {
      shape.delete();
      throw new Error("operation produced an empty shape");
    }
    const id = this.nextId++;
    this.shapes.set(id, shape);
    return id;
  }

  private get(id: ShapeId): TopoDS_Shape {
    const s = this.shapes.get(id);
    if (!s) throw new Error(`unknown shape id ${id}`);
    return s;
  }

  /** Direct access to a registered OCCT shape (for advanced in-worker use; do not delete it). */
  getShape(id: ShapeId): TopoDS_Shape {
    return this.get(id);
  }

  /** Register an externally built OCCT shape; the kernel takes ownership. */
  adoptShape(shape: TopoDS_Shape): ShapeId {
    return this.register(shape);
  }

  release(id: ShapeId): void {
    const s = this.shapes.get(id);
    if (s) {
      s.delete();
      this.shapes.delete(id);
    }
  }

  releaseAll(): void {
    for (const s of this.shapes.values()) s.delete();
    this.shapes.clear();
  }

  count(): number {
    return this.shapes.size;
  }

  /** Run an operation inside a memory scope; errors become `${name} failed: <reason>`. */
  private run<T>(name: string, fn: (s: Scope, gp: Gp) => T): T {
    const s = new Scope();
    const gp = new Gp(this.oc, s);
    try {
      return fn(s, gp);
    } catch (e) {
      throw occtError(this.oc, e, `${name} failed`);
    } finally {
      s.dispose();
    }
  }

  /** Register a result created inside `s` (removing it from the scope). */
  private keepAndRegister(s: Scope, shape: TopoDS_Shape): ShapeId {
    return this.register(s.keep(shape));
  }

  // -------------------------------------------------------------------------
  // Geometry builders
  // -------------------------------------------------------------------------

  private segmentEdge(s: Scope, gp: Gp, f: Frame, seg: KSegment): TopoDS_Edge {
    const oc = this.oc;
    const P = (p: Vec2) => gp.pnt(frameToWorld(f, p));
    const lineEdge = (a: Vec2, b: Vec2) => s.add(new oc.BRepBuilderAPI_MakeEdge(P(a), P(b)));
    let mk;
    switch (seg.type) {
      case "line":
        mk = lineEdge(seg.a, seg.b);
        break;
      case "arc": {
        if (!arcGeometry(seg)) {
          mk = lineEdge(seg.start, seg.end);
          break;
        }
        const arc = s.add(new oc.GC_MakeArcOfCircle(P(seg.start), P(seg.mid), P(seg.end)));
        if (!arc.IsDone()) throw new Error("degenerate arc in profile");
        mk = s.add(new oc.BRepBuilderAPI_MakeEdge(s.add(arc.Value())));
        break;
      }
      case "circle": {
        const ax = gp.ax2(frameToWorld(f, seg.center), seg.cw ? negate(f.zAxis) : f.zAxis, f.xAxis);
        mk = s.add(new oc.BRepBuilderAPI_MakeEdge(s.add(new oc.gp_Circ(ax, seg.radius))));
        break;
      }
      case "bspline": {
        const err = checkCurveDef(seg.curve);
        if (err) throw new Error(`invalid bspline segment: ${err}`);
        const def: BSplineCurveDef = { ...seg.curve, controlPoints: seg.curve.controlPoints.map((p) => frameToWorld(f, p)) };
        mk = s.add(new oc.BRepBuilderAPI_MakeEdge(makeBSplineCurve(oc, gp, s, def)));
        break;
      }
    }
    if (!mk.IsDone()) throw new Error(`cannot build ${seg.type} edge`);
    return s.add(mk.Edge());
  }

  private chainWire(s: Scope, gp: Gp, f: Frame, segs: readonly KSegment[]): TopoDS_Wire {
    const mw = s.add(new this.oc.BRepBuilderAPI_MakeWire());
    for (const seg of segs) mw.Add(this.segmentEdge(s, gp, f, seg));
    if (!mw.IsDone()) throw new Error(`wire construction failed (${String(mw.Error())})`);
    return s.add(mw.Wire());
  }

  private curveWire(s: Scope, gp: Gp, c: BSplineCurveDef): TopoDS_Wire {
    const err = checkCurveDef(c);
    if (err) throw new Error(`invalid curve: ${err}`);
    const mk = s.add(new this.oc.BRepBuilderAPI_MakeEdge(makeBSplineCurve(this.oc, gp, s, c)));
    if (!mk.IsDone()) throw new Error("cannot build curve edge");
    const mw = s.add(new this.oc.BRepBuilderAPI_MakeWire(s.add(mk.Edge())));
    return s.add(mw.Wire());
  }

  private regionFace(s: Scope, gp: Gp, f: Frame, r: Region): TopoDS_Face {
    const oc = this.oc;
    const outer = this.chainWire(s, gp, f, r.outer);
    const pln = s.add(new oc.gp_Pln(gp.frameAx3(f)));
    const mf = s.add(new oc.BRepBuilderAPI_MakeFace(pln, outer, true));
    for (const h of r.holes) mf.Add(this.chainWire(s, gp, f, h));
    if (!mf.IsDone()) throw new Error("planar face construction failed");
    let face = s.add(mf.Face());
    const an = s.add(new oc.BRepCheck_Analyzer(face, true, false, false));
    if (!an.IsValid()) {
      const fix = s.add(new oc.ShapeFix_Face(face));
      fix.Perform(gp.progress());
      face = s.add(fix.Face());
    }
    return face;
  }

  private makeCompound(s: Scope, parts: readonly TopoDS_Shape[]): TopoDS_Shape {
    if (parts.length === 1) return parts[0];
    const comp = s.add(new this.oc.TopoDS_Compound());
    const b = s.add(new this.oc.TopoDS_Builder());
    b.MakeCompound(comp);
    for (const p of parts) b.Add(comp, p);
    return comp;
  }

  /** A single-child compound becomes its solid/shell/face. */
  private unwrap(s: Scope, shape: TopoDS_Shape): TopoDS_Shape {
    const T = this.oc.TopAbs_ShapeEnum;
    if (shape.ShapeType() !== T.TopAbs_COMPOUND || shape.NbChildren() !== 1) return shape;
    const solids = mapShapes(this.oc, s, shape, T.TopAbs_SOLID).shapes;
    if (solids.length === 1) return solids[0];
    if (solids.length) return shape;
    const shells = mapShapes(this.oc, s, shape, T.TopAbs_SHELL).shapes;
    if (shells.length === 1) return shells[0];
    if (shells.length) return shape;
    const faces = mapShapes(this.oc, s, shape, T.TopAbs_FACE).shapes;
    return faces.length === 1 ? faces[0] : shape;
  }

  /** A fresh handle on a registered shape (so it can be registered/kept independently). */
  private copyHandle(s: Scope, shape: TopoDS_Shape): TopoDS_Shape {
    return s.add(shape.Oriented(shape.Orientation()));
  }

  private volumeOf(s: Scope, shape: TopoDS_Shape): number {
    const p = s.add(new this.oc.GProp_GProps());
    this.oc.BRepGProp.VolumeProperties(shape, p, false, false, false);
    return p.Mass();
  }

  private isValid(s: Scope, shape: TopoDS_Shape): boolean {
    return s.add(new this.oc.BRepCheck_Analyzer(shape, true, false, false)).IsValid();
  }

  /** Solid from a closed shell, outward oriented. */
  private solidFromShell(s: Scope, shell: TopoDS_Shape): TopoDS_Shape {
    const oc = this.oc;
    const fix = s.add(new oc.ShapeFix_Solid());
    let solid: TopoDS_Shape = s.add(fix.SolidFromShell(s.add(oc.TopoDS.Shell(shell))));
    if (this.volumeOf(s, solid) < 0) solid = s.add(solid.Reversed());
    return solid;
  }

  private fixInverted(s: Scope, shape: TopoDS_Shape): TopoDS_Shape {
    const T = this.oc.TopAbs_ShapeEnum;
    if (shape.ShapeType() === T.TopAbs_SOLID && this.volumeOf(s, shape) < 0) return s.add(shape.Reversed());
    return shape;
  }

  private fuseAll(s: Scope, gp: Gp, parts: TopoDS_Shape[]): TopoDS_Shape {
    if (parts.length === 1) return parts[0];
    const fuse = s.add(new this.oc.BRepAlgoAPI_Fuse());
    fuse.SetArguments(gp.shapeList([parts[0]]));
    fuse.SetTools(gp.shapeList(parts.slice(1)));
    fuse.Build(gp.progress());
    if (fuse.HasErrors() || !fuse.IsDone()) throw new Error("fusing the pattern copies failed");
    return this.unwrap(s, s.add(fuse.Shape()));
  }

  // -------------------------------------------------------------------------
  // Features
  // -------------------------------------------------------------------------

  extrude(a: ExtrudeArgs): ShapeId {
    return this.run("Extrude", (s, gp) => {
      const oc = this.oc;
      const { frame } = a;
      const d1 = a.distance, d2 = a.distance2 ?? 0;
      if (!Number.isFinite(d1) || !Number.isFinite(d2)) throw new Error("distance must be a finite number");
      const total = d1 + d2;
      if (Math.abs(total) < 1e-9) throw new Error("zero extrusion length");
      const dir = normalize(a.direction ?? frame.zAxis);
      if (Math.abs(dot(dir, normalize(frame.zAxis))) < 1e-6) throw new Error("extrusion direction lies in the sketch plane");
      const base: Frame = { ...frame, origin: addScaled(frame.origin, dir, -d2) };
      const vec = gp.vec(scale(dir, total));
      if (!a.profile?.loops?.length) throw new Error("profile has no closed loop");
      if (a.solid === false) {
        const parts = a.profile.loops.map((l, i) => {
          const w = this.chainWire(s, gp, base, snapLoop(l, undefined, `profile loop ${i}`));
          return s.add(s.add(new oc.BRepPrimAPI_MakePrism(w, vec, false, true)).Shape());
        });
        return this.keepAndRegister(s, this.makeCompound(s, parts));
      }
      const regions = buildRegions(a.profile);
      const parts = regions.map((r) => {
        const face = this.regionFace(s, gp, base, r);
        const mk = s.add(new oc.BRepPrimAPI_MakePrism(face, vec, false, true));
        if (!mk.IsDone()) throw new Error("prism construction failed");
        let shape: TopoDS_Shape = s.add(mk.Shape());
        if (a.draftDeg) {
          const first = s.add(mk.FirstShape());
          const last = s.add(mk.LastShape());
          const da = s.add(new oc.BRepOffsetAPI_DraftAngle(shape));
          const neutral = s.add(new oc.gp_Pln(gp.pnt(frame.origin), gp.dir(frame.zAxis)));
          const pull = gp.dir(dir);
          for (const f of mapShapes(oc, s, shape, oc.TopAbs_ShapeEnum.TopAbs_FACE).shapes) {
            if (f.IsSame(first) || f.IsSame(last)) continue;
            da.Add(s.add(oc.TopoDS.Face(f)), pull, a.draftDeg * DEG, neutral, true);
            if (!da.AddDone()) throw new Error("draft angle cannot be applied to a side face (angle too large?)");
          }
          da.Build(gp.progress());
          if (!da.IsDone()) throw new Error("draft angle computation failed");
          shape = s.add(da.Shape());
        }
        return shape;
      });
      return this.keepAndRegister(s, this.makeCompound(s, parts));
    });
  }

  extrudeWire(a: ExtrudeWireArgs): ShapeId {
    return this.run("Extrude", (s, gp) => {
      const d1 = a.distance, d2 = a.distance2 ?? 0;
      const total = d1 + d2;
      if (!Number.isFinite(total) || Math.abs(total) < 1e-9) throw new Error("zero extrusion length");
      const dir = normalize(a.direction ?? a.frame.zAxis);
      const base: Frame = { ...a.frame, origin: addScaled(a.frame.origin, dir, -d2) };
      const w = this.chainWire(s, gp, base, snapWire(a.wire));
      const mk = s.add(new this.oc.BRepPrimAPI_MakePrism(w, gp.vec(scale(dir, total)), false, true));
      if (!mk.IsDone()) throw new Error("prism construction failed");
      return this.keepAndRegister(s, s.add(mk.Shape()));
    });
  }

  revolve(a: RevolveArgs): ShapeId {
    return this.run("Revolve", (s, gp) => {
      const oc = this.oc;
      const axis = gp.ax1(a.axisOrigin, a.axisDir);
      const ang = a.angleDeg * DEG;
      if (!Number.isFinite(ang) || Math.abs(ang) < 1e-9) throw new Error("zero revolve angle");
      const full = Math.abs(a.angleDeg) >= 360 - 1e-9;
      let inputs: TopoDS_Shape[];
      if (a.profile) {
        if (!a.profile.loops.length) throw new Error("profile has no closed loop");
        inputs =
          a.solid === false
            ? a.profile.loops.map((l, i) => this.chainWire(s, gp, a.frame, snapLoop(l, undefined, `profile loop ${i}`)))
            : buildRegions(a.profile).map((r) => this.regionFace(s, gp, a.frame, r));
      } else if (a.wire) inputs = [this.chainWire(s, gp, a.frame, snapWire(a.wire))];
      else throw new Error("either a profile or a wire is required");
      const parts = inputs.map((sh) => {
        const mk = s.add(full ? new oc.BRepPrimAPI_MakeRevol(sh, axis, false) : new oc.BRepPrimAPI_MakeRevol(sh, axis, ang, false));
        if (!mk.IsDone()) throw new Error("revolution failed");
        return s.add(mk.Shape());
      });
      const result = this.makeCompound(s, parts);
      if (a.profile && a.solid !== false && !this.isValid(s, result))
        throw new Error("result is not a valid solid (does the profile cross the axis?)");
      return this.keepAndRegister(s, result);
    });
  }

  bsplineSurfaceFace(a: BSplineFaceArgs): ShapeId {
    return this.run("B-spline face", (s, gp) => {
      const err = checkSurfaceDef(a.surface);
      if (err) throw new Error(err);
      const surf = makeBSplineSurface(this.oc, gp, s, a.surface);
      let mf;
      if (a.uRange || a.vRange) {
        const u = a.uRange ?? [a.surface.knotsU[a.surface.degreeU], a.surface.knotsU[a.surface.nu]];
        const v = a.vRange ?? [a.surface.knotsV[a.surface.degreeV], a.surface.knotsV[a.surface.nv]];
        mf = s.add(new this.oc.BRepBuilderAPI_MakeFace(surf, u[0], u[1], v[0], v[1], 1e-6));
      } else mf = s.add(new this.oc.BRepBuilderAPI_MakeFace(surf, 1e-6));
      if (!mf.IsDone()) throw new Error("face construction failed");
      return this.keepAndRegister(s, s.add(mf.Face()));
    });
  }

  bsplinePatchesToShape(a: PatchesToShapeArgs): ShapeId {
    return this.run("Sewing", (s, gp) => {
      const oc = this.oc;
      const T = oc.TopAbs_ShapeEnum;
      if (!a.surfaces.length) throw new Error("no surfaces given");
      const tol = a.sewTolerance > 0 ? a.sewTolerance : 1e-3;
      const sew = s.add(new oc.BRepBuilderAPI_Sewing(tol, true, true, true, false));
      a.surfaces.forEach((d, i) => {
        const err = checkSurfaceDef(d);
        if (err) throw new Error(`surface ${i}: ${err}`);
        const mf = s.add(new oc.BRepBuilderAPI_MakeFace(makeBSplineSurface(oc, gp, s, d), 1e-6));
        if (!mf.IsDone()) throw new Error(`surface ${i}: face construction failed`);
        sew.Add(s.add(mf.Face()));
      });
      sew.Perform(gp.progress());
      const sewed: TopoDS_Shape = s.add(sew.SewedShape());
      if (!a.makeSolid) return this.keepAndRegister(s, this.unwrap(s, sewed));
      const out: TopoDS_Shape[] = [];
      for (const sh of mapShapes(oc, s, sewed, T.TopAbs_SHELL).shapes)
        out.push(oc.BRep_Tool.IsClosed(sh) ? this.solidFromShell(s, sh) : sh);
      const free = s.add(new oc.TopExp_Explorer(sewed, T.TopAbs_FACE, T.TopAbs_SHELL));
      for (; free.More(); free.Next()) out.push(s.add(free.Current()));
      if (!out.length) throw new Error("sewing produced no faces");
      return this.keepAndRegister(s, this.makeCompound(s, out));
    });
  }

  primitiveFace(def: PrimitiveDef, bounds: PrimitiveBounds): ShapeId {
    return this.run("Primitive face", (s, gp) => {
      const oc = this.oc;
      const sized = "size" in bounds ? bounds.size : null;
      if (sized !== null && !(sized > 0)) throw new Error("size must be > 0");
      let mf;
      switch (def.kind) {
        case "plane": {
          const n = normalize(def.normal);
          const ax = gp.ax3(def.origin, n, anyPerpendicular(n));
          const b = sized !== null ? { uMin: -sized / 2, uMax: sized / 2, vMin: -sized / 2, vMax: sized / 2 } : (bounds as Exclude<PrimitiveBounds, { size: number }>);
          mf = s.add(new oc.BRepBuilderAPI_MakeFace(s.add(new oc.gp_Pln(ax)), b.uMin, b.uMax, b.vMin, b.vMax));
          break;
        }
        case "cylinder": {
          const n = normalize(def.axis);
          const ax = gp.ax3(def.origin, n, anyPerpendicular(n));
          const hasH = def.hMax > def.hMin;
          const b =
            sized !== null
              ? { uMin: 0, uMax: 2 * Math.PI, vMin: hasH ? def.hMin : -sized / 2, vMax: hasH ? def.hMax : sized / 2 }
              : (bounds as Exclude<PrimitiveBounds, { size: number }>);
          mf = s.add(new oc.BRepBuilderAPI_MakeFace(s.add(new oc.gp_Cylinder(ax, def.radius)), b.uMin, b.uMax, b.vMin, b.vMax));
          break;
        }
        case "cone": {
          const n = normalize(def.axis);
          const ax = gp.ax3(def.apex, n, anyPerpendicular(n));
          const surf = s.add(new oc.Geom_ConicalSurface(ax, def.halfAngle, 0));
          const hasH = def.hMax > def.hMin;
          const b =
            sized !== null
              ? { uMin: 0, uMax: 2 * Math.PI, vMin: hasH ? def.hMin : 0, vMax: hasH ? def.hMax : sized }
              : (bounds as Exclude<PrimitiveBounds, { size: number }>);
          // Geom_ConicalSurface's v runs along the generatrix: axial height = v cos(halfAngle)
          const c = Math.cos(def.halfAngle);
          mf = s.add(new oc.BRepBuilderAPI_MakeFace(surf, b.uMin, b.uMax, b.vMin / c, b.vMax / c, 1e-6));
          break;
        }
        case "sphere": {
          const ax = gp.ax3(def.center, [0, 0, 1], [1, 0, 0]);
          const sph = s.add(new oc.gp_Sphere(ax, def.radius));
          if (sized !== null) mf = s.add(new oc.BRepBuilderAPI_MakeFace(sph));
          else {
            const b = bounds as Exclude<PrimitiveBounds, { size: number }>;
            mf = s.add(new oc.BRepBuilderAPI_MakeFace(sph, b.uMin, b.uMax, b.vMin, b.vMax));
          }
          break;
        }
        default:
          throw new Error(`unsupported primitive '${(def as { kind: string }).kind}'`);
      }
      if (!mf.IsDone()) throw new Error("face construction failed");
      return this.keepAndRegister(s, s.add(mf.Face()));
    });
  }

  loft(a: LoftArgs): ShapeId {
    return this.run("Loft", (s, gp) => {
      let wires: TopoDS_Wire[];
      if (Array.isArray(a.sections)) {
        wires = a.sections.map((sec, i) => {
          const regions = buildRegions(sec.profile);
          if (regions.length !== 1 || regions[0].holes.length) throw new Error(`section ${i} must be a single closed loop`);
          return this.chainWire(s, gp, sec.frame, regions[0].outer);
        });
      } else wires = a.sections.curves.map((c) => this.curveWire(s, gp, c));
      if (wires.length < 2) throw new Error("at least two sections are required");
      const ts = s.add(new this.oc.BRepOffsetAPI_ThruSections(a.solid, a.ruled ?? false, 1e-6));
      for (const w of wires) ts.AddWire(w);
      ts.Build(gp.progress());
      if (!ts.IsDone()) throw new Error("sections are incompatible or the loft did not converge");
      return this.keepAndRegister(s, s.add(ts.Shape()));
    });
  }

  sweep(a: SweepArgs): ShapeId {
    return this.run("Sweep", (s, gp) => {
      const oc = this.oc;
      const solid = a.solid !== false;
      let spine: TopoDS_Wire;
      let polyline = false;
      if ("points" in a.path) {
        const pts = a.path.points;
        if (pts.length < 2) throw new Error("path needs at least two points");
        if (a.path.smooth !== false && pts.length > 2) {
          const h = s.add(new oc.NCollection_HArray1_gp_Pnt(gp.arrPnt(pts)));
          const ip = s.add(new oc.GeomAPI_Interpolate(h, false, 1e-7));
          ip.Perform();
          if (!ip.IsDone()) throw new Error("path interpolation failed");
          const mk = s.add(new oc.BRepBuilderAPI_MakeEdge(s.add(ip.Curve())));
          spine = s.add(s.add(new oc.BRepBuilderAPI_MakeWire(s.add(mk.Edge()))).Wire());
        } else {
          polyline = true;
          const mw = s.add(new oc.BRepBuilderAPI_MakeWire());
          for (let i = 0; i + 1 < pts.length; i++) {
            if (length([pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1], pts[i + 1][2] - pts[i][2]]) < 1e-9) continue;
            mw.Add(s.add(s.add(new oc.BRepBuilderAPI_MakeEdge(gp.pnt(pts[i]), gp.pnt(pts[i + 1]))).Edge()));
          }
          spine = s.add(mw.Wire());
        }
      } else spine = this.curveWire(s, gp, a.path);
      const pipe = (w: TopoDS_Wire): TopoDS_Shape => {
        const ps = s.add(new oc.BRepOffsetAPI_MakePipeShell(spine));
        ps.SetMode(a.frenet ?? false);
        if (polyline) ps.SetTransitionMode(oc.BRepBuilderAPI_TransitionMode.BRepBuilderAPI_RoundCorner);
        ps.Add(w, false, false);
        ps.Build(gp.progress());
        if (!ps.IsDone()) throw new Error("pipe shell construction failed");
        if (solid && !ps.MakeSolid()) throw new Error("could not close the swept shape into a solid");
        return this.fixInverted(s, s.add(ps.Shape()));
      };
      const parts = buildRegions(a.profile).map((r) => {
        let body = pipe(this.chainWire(s, gp, a.frame, r.outer));
        for (const h of r.holes) {
          const tool = pipe(this.chainWire(s, gp, a.frame, reverseLoop(h)));
          if (!solid) {
            body = this.makeCompound(s, [body, tool]);
            continue;
          }
          const cut = s.add(new oc.BRepAlgoAPI_Cut(body, tool, gp.progress()));
          if (cut.HasErrors()) throw new Error("subtracting a profile hole failed");
          body = this.unwrap(s, s.add(cut.Shape()));
        }
        return body;
      });
      return this.keepAndRegister(s, this.makeCompound(s, parts));
    });
  }

  // -------------------------------------------------------------------------
  // Operations
  // -------------------------------------------------------------------------

  boolean(op: BooleanOp, a: ShapeId, b: ShapeId, opts?: BooleanOptions): ShapeId {
    return this.run(`Boolean ${op}`, (s, gp) => {
      const oc = this.oc;
      const A = this.get(a), B = this.get(b);
      const algo =
        op === "union"
          ? s.add(new oc.BRepAlgoAPI_Fuse(A, B, gp.progress()))
          : op === "cut"
            ? s.add(new oc.BRepAlgoAPI_Cut(A, B, gp.progress()))
            : op === "intersect"
              ? s.add(new oc.BRepAlgoAPI_Common(A, B, gp.progress()))
              : null;
      if (!algo) throw new Error(`unknown boolean operation '${op}'`);
      if (algo.HasErrors() || !algo.IsDone()) throw new Error("the boolean algorithm reported errors");
      if (opts?.simplify) algo.SimplifyResult(true, true, 1e-6);
      const res = this.unwrap(s, s.add(algo.Shape()));
      if (!mapShapes(oc, s, res, oc.TopAbs_ShapeEnum.TopAbs_FACE).shapes.length) throw new Error("result is empty");
      return this.keepAndRegister(s, res);
    });
  }

  split(a: ShapeId, tool: ShapeId): ShapeId {
    return this.run("Split", (s, gp) => {
      const sp = s.add(new this.oc.BRepAlgoAPI_Splitter());
      sp.SetArguments(gp.shapeList([this.get(a)]));
      sp.SetTools(gp.shapeList([this.get(tool)]));
      sp.Build(gp.progress());
      if (sp.HasErrors() || !sp.IsDone()) throw new Error("the splitter reported errors");
      return this.keepAndRegister(s, s.add(sp.Shape()));
    });
  }

  private pickEdges(s: Scope, shape: TopoDS_Shape, ids: readonly number[]): TopoDS_Edge[] {
    if (!ids.length) throw new Error("no edges selected");
    const edges = mapShapes(this.oc, s, shape, this.oc.TopAbs_ShapeEnum.TopAbs_EDGE).shapes;
    return ids.map((id) => {
      if (!Number.isInteger(id) || id < 0 || id >= edges.length) throw new Error(`edge id ${id} out of range (shape has ${edges.length} edges)`);
      return s.add(this.oc.TopoDS.Edge(edges[id]));
    });
  }

  fillet(shape: ShapeId, edgeIds: number[], radius: number): ShapeId {
    return this.run("Fillet", (s, gp) => {
      if (!(radius > 0)) throw new Error("radius must be > 0");
      const sh = this.get(shape);
      const mk = s.add(new this.oc.BRepFilletAPI_MakeFillet(sh, this.oc.ChFi3d_FilletShape.ChFi3d_Rational));
      for (const e of this.pickEdges(s, sh, edgeIds)) mk.Add(radius, e);
      mk.Build(gp.progress());
      if (!mk.IsDone()) throw new Error("radius too large or edges cannot be filleted");
      return this.keepAndRegister(s, this.unwrap(s, s.add(mk.Shape())));
    });
  }

  chamfer(shape: ShapeId, edgeIds: number[], distance: number): ShapeId {
    return this.run("Chamfer", (s, gp) => {
      if (!(distance > 0)) throw new Error("distance must be > 0");
      const sh = this.get(shape);
      const mk = s.add(new this.oc.BRepFilletAPI_MakeChamfer(sh));
      for (const e of this.pickEdges(s, sh, edgeIds)) mk.Add(distance, e);
      mk.Build(gp.progress());
      if (!mk.IsDone()) throw new Error("distance too large or edges cannot be chamfered");
      return this.keepAndRegister(s, this.unwrap(s, s.add(mk.Shape())));
    });
  }

  /** Hollow a solid: removes the given faces, walls of `thickness` (positive = material inside). */
  shell(shape: ShapeId, faceIdsToRemove: number[], thickness: number): ShapeId {
    return this.run("Shell", (s, gp) => {
      const oc = this.oc;
      if (!thickness) throw new Error("thickness must be non-zero");
      const sh = this.get(shape);
      const faces = mapShapes(oc, s, sh, oc.TopAbs_ShapeEnum.TopAbs_FACE).shapes;
      const removed = faceIdsToRemove.map((id) => {
        if (!Number.isInteger(id) || id < 0 || id >= faces.length) throw new Error(`face id ${id} out of range (shape has ${faces.length} faces)`);
        return faces[id];
      });
      const mk = s.add(new oc.BRepOffsetAPI_MakeThickSolid());
      mk.MakeThickSolidByJoin(
        sh, gp.shapeList(removed), -thickness, 1e-3,
        oc.BRepOffset_Mode.BRepOffset_Skin, false, false, oc.GeomAbs_JoinType.GeomAbs_Arc, false, gp.progress(),
      );
      if (!mk.IsDone()) throw new Error("thick solid computation failed (thickness too large?)");
      return this.keepAndRegister(s, this.unwrap(s, s.add(mk.Shape())));
    });
  }

  /** Faces/shells -> solid by offsetting by `thickness` along the normals; solids are offset. */
  thicken(shape: ShapeId, thickness: number): ShapeId {
    return this.run("Thicken", (s, gp) => {
      const oc = this.oc;
      const T = oc.TopAbs_ShapeEnum;
      if (!thickness) throw new Error("thickness must be non-zero");
      const sh = this.get(shape);
      const hasSolid = mapShapes(oc, s, sh, T.TopAbs_SOLID).shapes.length > 0;
      let res: TopoDS_Shape;
      if (!hasSolid) {
        const mk = s.add(new oc.BRepOffsetAPI_MakeThickSolid());
        mk.MakeThickSolidBySimple(sh, thickness);
        if (!mk.IsDone()) throw new Error("thickening failed");
        res = s.add(mk.Shape());
      } else {
        const mk = s.add(new oc.BRepOffsetAPI_MakeOffsetShape());
        mk.PerformByJoin(sh, thickness, 1e-3, oc.BRepOffset_Mode.BRepOffset_Skin, false, false, oc.GeomAbs_JoinType.GeomAbs_Arc, false, gp.progress());
        if (!mk.IsDone()) throw new Error("offset failed");
        res = s.add(mk.Shape());
      }
      return this.keepAndRegister(s, this.fixInverted(s, this.unwrap(s, res)));
    });
  }

  transform(shape: ShapeId, m: Mat4): ShapeId {
    return this.run("Transform", (s) => {
      if (m.length !== 16) throw new Error("matrix must have 16 entries (column-major)");
      const tr = s.add(new this.oc.gp_Trsf());
      tr.SetValues(m[0], m[4], m[8], m[12], m[1], m[5], m[9], m[13], m[2], m[6], m[10], m[14]);
      const c0: Vec3 = [m[0], m[1], m[2]], c1: Vec3 = [m[4], m[5], m[6]], c2: Vec3 = [m[8], m[9], m[10]];
      const det = dot(c0, cross(c1, c2));
      const rigid = Math.abs(det - 1) < 1e-9;
      const mk = s.add(new this.oc.BRepBuilderAPI_Transform(this.get(shape), tr, !rigid, false));
      return this.keepAndRegister(s, s.add(mk.Shape()));
    });
  }

  mirror(shape: ShapeId, plane: { origin: Vec3; normal: Vec3 }): ShapeId {
    return this.run("Mirror", (s, gp) => {
      const tr = s.add(new this.oc.gp_Trsf());
      tr.SetMirror(gp.ax2(plane.origin, plane.normal));
      const mk = s.add(new this.oc.BRepBuilderAPI_Transform(this.get(shape), tr, true, false));
      return this.keepAndRegister(s, s.add(mk.Shape()));
    });
  }

  private pattern(name: string, shape: ShapeId, count: number, trsfAt: (i: number) => import("replicad-opencascadejs").gp_Trsf, opts?: PatternOptions): ShapeId {
    return this.run(name, (s, gp) => {
      if (!Number.isInteger(count) || count < 1) throw new Error("count must be an integer >= 1");
      const base = this.get(shape);
      const parts: TopoDS_Shape[] = [this.copyHandle(s, base)];
      for (let i = 1; i < count; i++) {
        const mk = s.add(new this.oc.BRepBuilderAPI_Transform(base, s.add(trsfAt(i)), false, false));
        parts.push(s.add(mk.Shape()));
      }
      const res = opts?.fuse ? this.fuseAll(s, gp, parts) : this.makeCompound(s, parts);
      return this.keepAndRegister(s, res);
    });
  }

  linearPattern(shape: ShapeId, dir: Vec3, count: number, spacing: number, opts?: PatternOptions): ShapeId {
    const d = normalize(dir);
    return this.pattern("Linear pattern", shape, count, (i) => {
      const tr = new this.oc.gp_Trsf();
      const v = new this.oc.gp_Vec(d[0] * spacing * i, d[1] * spacing * i, d[2] * spacing * i);
      tr.SetTranslation(v);
      v.delete();
      return tr;
    }, opts);
  }

  circularPattern(shape: ShapeId, axisOrigin: Vec3, axisDir: Vec3, count: number, totalAngleDeg: number, opts?: PatternOptions): ShapeId {
    const full = Math.abs(totalAngleDeg) >= 360 - 1e-9;
    const step = (full ? totalAngleDeg / count : count > 1 ? totalAngleDeg / (count - 1) : 0) * DEG;
    const d = normalize(axisDir);
    return this.pattern("Circular pattern", shape, count, (i) => {
      const tr = new this.oc.gp_Trsf();
      const p = new this.oc.gp_Pnt(axisOrigin[0], axisOrigin[1], axisOrigin[2]);
      const dd = new this.oc.gp_Dir(d[0], d[1], d[2]);
      const ax = new this.oc.gp_Ax1(p, dd);
      tr.SetRotation(ax, step * i);
      ax.delete(); dd.delete(); p.delete();
      return tr;
    }, opts);
  }

  compound(ids: ShapeId[]): ShapeId {
    return this.run("Compound", (s) => {
      if (!ids.length) throw new Error("no shapes given");
      const parts = ids.map((id) => this.copyHandle(s, this.get(id)));
      return this.keepAndRegister(s, this.makeCompound(s, parts));
    });
  }

  // -------------------------------------------------------------------------
  // Queries
  // -------------------------------------------------------------------------

  private tessellateShape(s: Scope, shape: TopoDS_Shape, lin: number, angDeg: number, withEdges: boolean): Tessellation {
    const oc = this.oc;
    if (!(lin > 0)) throw new Error("linearDeflection must be > 0");
    const ang = Math.max(0.5, angDeg || 20) * DEG;
    const T = oc.TopAbs_ShapeEnum;
    const faceHashes = mapShapes(oc, s, shape, T.TopAbs_FACE).hashes;
    const data = s.add(oc.ReplicadMeshExtractor.extract(shape, lin, ang, false));
    const positions = new Float32Array(oc.wasmMemory.buffer, data.getVerticesPtr(), data.getVerticesSize()).slice();
    const normals = new Float32Array(oc.wasmMemory.buffer, data.getNormalsPtr(), data.getNormalsSize()).slice();
    const tris = new Uint32Array(oc.wasmMemory.buffer, data.getTrianglesPtr(), data.getTrianglesSize()).slice();
    const groups = new Int32Array(oc.wasmMemory.buffer, data.getFaceGroupsPtr(), data.getFaceGroupsSize()).slice();
    const groupOf = assignGroups(faceHashes, groups);
    const indices = new Uint32Array(tris.length);
    const faceTriangleRanges = new Uint32Array(2 * faceHashes.length);
    let w = 0;
    groupOf.forEach((g, fi) => {
      faceTriangleRanges[2 * fi] = w / 3;
      if (g < 0) return;
      const start = groups[3 * g], cnt = groups[3 * g + 1];
      indices.set(tris.subarray(start, start + cnt), w);
      w += cnt;
      faceTriangleRanges[2 * fi + 1] = cnt / 3;
    });
    const edges: TessellationEdge[] = [];
    if (withEdges) {
      const edgeHashes = mapShapes(oc, s, shape, T.TopAbs_EDGE).hashes;
      const ed = s.add(oc.ReplicadEdgeMeshExtractor.extract(shape, lin, ang));
      const lines = new Float32Array(oc.wasmMemory.buffer, ed.getLinesPtr(), ed.getLinesSize()).slice();
      const eg = new Int32Array(oc.wasmMemory.buffer, ed.getEdgeGroupsPtr(), ed.getEdgeGroupsSize()).slice();
      assignGroups(edgeHashes, eg).forEach((g, id) => {
        const pts = g < 0 ? new Float32Array(0) : lines.slice(3 * eg[3 * g], 3 * (eg[3 * g] + eg[3 * g + 1]));
        edges.push({ id, points: pts });
      });
    }
    return { positions, normals, indices: w === indices.length ? indices : indices.slice(0, w), faceTriangleRanges, edges };
  }

  tessellate(shape: ShapeId, opts: TessellateOptions): Tessellation {
    return this.run("Tessellation", (s) => this.tessellateShape(s, this.get(shape), opts.linearDeflection, opts.angularDeflectionDeg, true));
  }

  info(shape: ShapeId): ShapeInfo {
    return this.run("Info", (s) => {
      const oc = this.oc;
      const T = oc.TopAbs_ShapeEnum;
      const sh = this.get(shape);
      const st = sh.ShapeType();
      const type: ShapeKind =
        st === T.TopAbs_SOLID ? "solid"
          : st === T.TopAbs_SHELL ? "shell"
            : st === T.TopAbs_FACE ? "face"
              : st === T.TopAbs_COMPOUND || st === T.TopAbs_COMPSOLID ? "compound"
                : "other";
      const n = (t: (typeof T)[keyof typeof T]) => mapShapes(oc, s, sh, t).shapes.length;
      const faces = n(T.TopAbs_FACE), edges = n(T.TopAbs_EDGE), solids = n(T.TopAbs_SOLID);
      const shellList = mapShapes(oc, s, sh, T.TopAbs_SHELL).shapes;
      const shells = shellList.length;
      // closed: every shell is closed (no free edges) and no stray faces outside shells
      const stray = s.add(new oc.TopExp_Explorer(sh, T.TopAbs_FACE, T.TopAbs_SHELL)).More();
      const closed = shells > 0 && !stray && shellList.every((x) => oc.BRep_Tool.IsClosed(x));
      const sp = s.add(new oc.GProp_GProps());
      oc.BRepGProp.SurfaceProperties(sh, sp, false, false);
      const box = s.add(new oc.Bnd_Box());
      oc.BRepBndLib.AddOptimal(sh, box, false, false);
      const bbox = box.IsVoid()
        ? { min: [0, 0, 0] as Vec3, max: [0, 0, 0] as Vec3 }
        : { min: [box.GetXMin(), box.GetYMin(), box.GetZMin()] as Vec3, max: [box.GetXMax(), box.GetYMax(), box.GetZMax()] as Vec3 };
      return {
        type, faces, edges, solids, shells, closed,
        volume: solids > 0 ? this.volumeOf(s, sh) : undefined,
        area: sp.Mass(),
        bbox,
        valid: this.isValid(s, sh),
      };
    });
  }

  // -------------------------------------------------------------------------
  // Export / persistence
  // -------------------------------------------------------------------------

  exportSTEP(items: ExportItem[]): string {
    return this.run("STEP export", (s, gp) => {
      const oc = this.oc;
      if (!items.length) throw new Error("nothing to export");
      oc.Interface_Static.SetCVal("write.step.unit", "MM");
      const path = `/tmp/resurface_${++this.tmpCounter}.step`;
      try {
        const doc = s.add(new oc.TDocStd_Document(s.add(new oc.TCollection_ExtendedString("XmlOcaf", true))));
        const tool = s.add(oc.XCAFDoc_DocumentTool.ShapeTool(s.add(doc.Main())));
        items.forEach((it, i) => {
          const label = s.add(tool.NewShape());
          tool.SetShape(label, this.get(it.id));
          const name = s.add(new oc.TCollection_ExtendedString(it.name || `Shape${i + 1}`, true));
          s.add(oc.TDataStd_Name.Set(label, name));
        });
        tool.UpdateAssemblies();
        const ws = s.add(new oc.XSControl_WorkSession());
        const writer = s.add(new oc.STEPCAFControl_Writer(ws, false));
        writer.SetNameMode(true);
        if (!writer.Perform(doc, path, gp.progress())) throw new Error("the STEP writer reported an error");
        return oc.FS.readFile(path, { encoding: "utf8" });
      } finally {
        try {
          oc.FS.unlink(path);
        } catch {
          /* not written */
        }
      }
    });
  }

  exportIGES(items: ExportItem[], opts: IgesExportOptions = {}): string {
    return this.run("IGES export", () => {
      if (!items.length) throw new Error("nothing to export");
      const tolerance = opts.approximationTolerance ?? 1e-4;
      const surfaces = items.flatMap((it) => shapeToIgesSurfaces(this.oc, this.get(it.id), { tolerance, name: it.name }));
      return writeIGES({
        surfaces,
        units: "mm",
        author: opts.author,
        organization: opts.organization,
        fileName: opts.fileName,
      });
    });
  }

  exportSTL(items: ExportItem[], opts: StlExportOptions): Uint8Array<ArrayBuffer> {
    return this.run("STL export", (s) => {
      if (!items.length) throw new Error("nothing to export");
      const parts = items.map((it) => this.tessellateShape(s, this.get(it.id), opts.linearDeflection, opts.angularDeflectionDeg ?? 20, false));
      return writeBinarySTL(parts);
    });
  }

  serialize(shape: ShapeId): string {
    return this.run("Serialize", () => this.oc.BRepToolsWrapper.Write(this.get(shape)));
  }

  deserialize(brep: string): ShapeId {
    return this.run("Deserialize", (s) => this.keepAndRegister(s, s.add(this.oc.BRepToolsWrapper.Read(brep))));
  }
}

/** Map extractor groups [start, count, hash]* onto MapShapes indices (same order, hash-checked). */
function assignGroups(hashes: readonly number[], groups: Int32Array): Int32Array {
  const pending = new Map<number, number[]>();
  hashes.forEach((h, i) => {
    const l = pending.get(h);
    if (l) l.push(i);
    else pending.set(h, [i]);
  });
  const out = new Int32Array(hashes.length).fill(-1);
  for (let g = 0; 3 * g + 2 < groups.length; g++) {
    const l = pending.get(groups[3 * g + 2]);
    if (!l || !l.length) continue;
    out[l.shift()!] = g;
  }
  return out;
}
