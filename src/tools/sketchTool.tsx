/**
 * 2D sectioning and sketching (QuickSurface "2D Sketch"):
 *  - SectionTool: choose a plane, preview the mesh section live, create one or several sketches.
 *  - SketchEditTool: draw/fit lines, arcs, circles, splines over the section points, add
 *    constraints and dimensions (geometric solver), trim and fillet.
 */
import * as THREE from "three";
import {
  MousePointer2, Slash, Spline as SplineIcon, Circle as CircleIcon, RectangleHorizontal, Ruler, Scissors, Squircle,
  WandSparkles, Trash2, Focus, Minus, CornerDownRight, Waypoints,
} from "lucide-react";
import type { App } from "../app/App";
import { newId } from "../core/document";
import type { MeshEntity, SectionPolyline, SketchEntity as SketchDoc } from "../core/entities";
import type { Frame, PlaneDef, Vec2, Vec3 } from "../core/types";
import { frameToWorld, worldToFrame, dist2 } from "../core/vec";
import { sectionMesh } from "../mesh/section";
import {
  addArc, addArc3, addCircle, addConstraint, addLine, addPoint, addRectangle, addSpline, cloneSketch, createSketch, deleteConstraint,
  deleteEntities, deletePoint, entityToSegments, findNearest, getEntity, getPoint, indexSketch, isDimension, measure,
  validateConstraint, type Constraint, type DimensionConstraint, type Sketch,
} from "../sketch/model";
import { solveSketch } from "../sketch/solver";
import { Tool, type ToolPointerEvent } from "./Tool";
import { lazy } from "../app/lazy";
import { Overlay } from "./overlay";
import { Section, Field, Select, NumberInput, Slider, Check, Stats, int } from "../ui/widgets";
import { fatSegments, lineMaterial, pointSprites, polylinesToPairs, sampleSegment2D, translucentMaterial } from "../viewer/draw";
import { meshOffsetRange, offsetFrame, planeSourceOptions, resolvePlaneSource } from "./planes";
import type { SketchView } from "../viewer/SketchView";
import { COLORS } from "../viewer/views";
import { SECTION_COLOR, CONSTRUCTION_COLOR } from "../viewer/SketchView";

// ---------------------------------------------------------------------------
// Section computation helpers
// ---------------------------------------------------------------------------

function sectionCandidates(app: App, meshId: string, plane: PlaneDef): Uint32Array | undefined {
  const rt = app.sync?.meshRuntime(meshId);
  if (!rt) return undefined;
  const n = plane.normal, o = plane.origin;
  let count = 0;
  const hits: { s: number; c: number }[] = [];
  for (const ch of rt.chunks) {
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < 8; i++) {
      const x = i & 1 ? ch.bounds.max[0] : ch.bounds.min[0];
      const y = i & 2 ? ch.bounds.max[1] : ch.bounds.min[1];
      const z = i & 4 ? ch.bounds.max[2] : ch.bounds.min[2];
      const d = (x - o[0]) * n[0] + (y - o[1]) * n[1] + (z - o[2]) * n[2];
      lo = Math.min(lo, d);
      hi = Math.max(hi, d);
    }
    if (lo <= 0 && hi >= 0) {
      hits.push({ s: ch.triStart, c: ch.triCount });
      count += ch.triCount;
    }
  }
  if (count === rt.triangleCount) return undefined;
  const out = new Uint32Array(count);
  let k = 0;
  for (const h of hits) for (let t = 0; t < h.c; t++) out[k++] = h.s + t;
  return out;
}

export function computeSection(app: App, meshId: string, frame: Frame): SectionPolyline[] {
  const mesh = app.doc.get<MeshEntity>(meshId)?.mesh;
  if (!mesh) return [];
  const plane: PlaneDef = { kind: "plane", origin: frame.origin, normal: frame.zAxis };
  const pls = sectionMesh(mesh, plane, { candidates: sectionCandidates(app, meshId, plane) });
  return pls.map((pl) => ({ points: pl.points.map((p) => worldToFrame(frame, p)), closed: pl.closed }));
}

function sectionsTo3D(frame: Frame, secs: SectionPolyline[]) {
  return secs.map((s) => ({ points: s.points.map((p) => frameToWorld(frame, p)), closed: s.closed }));
}

// ---------------------------------------------------------------------------
// Section (plane setup) tool
// ---------------------------------------------------------------------------

interface SectionState {
  source: string;
  offset: number;
  rotation: number;
  count: number;
  spacing: number;
  stats: { loops: number; points: number } | null;
}

class SectionTool extends Tool<SectionState> {
  id: string;
  label: string;
  hint = "Choose the section plane. The orange curves show where it cuts the scan; they become reference points for sketching.";
  okLabel: string;
  private overlay = new Overlay(this.app);
  private meshId: string;
  private sectionMat = lineMaterial({ color: SECTION_COLOR, width: 2, depthTest: false });
  private timer = 0;

  constructor(app: App, private multi: boolean) {
    super(app, { source: "world:XY", offset: 0, rotation: 0, count: 5, spacing: 10, stats: null });
    this.id = multi ? "multiSection" : "sketch2d";
    this.label = multi ? "Parallel Sections" : "2D Sketch — Section";
    this.okLabel = multi ? "Create sections" : "Start sketch";
    const m = app.activeMesh();
    if (!m) throw new Error("Import a mesh first");
    this.meshId = m.id;
    // default: a plane through the middle of the part
    const f = resolvePlaneSource(app, "world:XY")!;
    const r = meshOffsetRange(app, f);
    this.set({ offset: Math.round(r.mid * 100) / 100, spacing: Math.max(0.5, Math.round(((r.max - r.min) / 6) * 10) / 10) });
  }

  activate() {
    this.update();
  }

  private frames(): Frame[] {
    const base = resolvePlaneSource(this.app, this.s.source, this.s.rotation);
    if (!base) return [];
    if (!this.multi) return [offsetFrame(base, this.s.offset)];
    const out: Frame[] = [];
    const n = Math.max(1, this.s.count);
    const start = this.s.offset - ((n - 1) * this.s.spacing) / 2;
    for (let i = 0; i < n; i++) out.push(offsetFrame(base, start + i * this.s.spacing));
    return out;
  }

  private update() {
    window.clearTimeout(this.timer);
    this.timer = window.setTimeout(() => this.drawPreview(), 30);
  }

  private drawPreview() {
    this.overlay.clear();
    const frames = this.frames();
    let loops = 0, points = 0;
    const size = this.app.sceneSize() * 0.4;
    for (const f of frames) {
      const secs = computeSection(this.app, this.meshId, f);
      loops += secs.length;
      for (const s of secs) points += s.points.length;
      const pairs = polylinesToPairs(sectionsTo3D(f, secs));
      if (pairs.length) this.overlay.add(fatSegments(pairs, this.sectionMat)).renderOrder = 25;
      const m = new THREE.Mesh(new THREE.PlaneGeometry(size, size), translucentMaterial("#ffb347", frames.length > 1 ? 0.08 : 0.14));
      const basis = new THREE.Matrix4().makeBasis(new THREE.Vector3(...f.xAxis), new THREE.Vector3(...f.yAxis), new THREE.Vector3(...f.zAxis));
      m.quaternion.setFromRotationMatrix(basis);
      m.position.set(...f.origin);
      this.overlay.add(m);
    }
    this.set({ stats: { loops, points } });
  }

  renderPanel() {
    const s = this.s;
    const base = resolvePlaneSource(this.app, s.source, s.rotation);
    const range = base ? meshOffsetRange(this.app, base) : { min: -100, max: 100 };
    const upd = (p: Partial<SectionState>) => {
      this.set(p);
      this.update();
    };
    return (
      <>
        <Section title="Plane">
          <Field label="Plane">
            <Select
              value={s.source}
              options={planeSourceOptions(this.app)}
              onChange={(source) => {
                const f = resolvePlaneSource(this.app, source);
                upd({ source, offset: f && source.startsWith("world") ? Math.round(meshOffsetRange(this.app, f).mid * 100) / 100 : 0, rotation: 0 });
              }}
            />
          </Field>
          <Field label="Offset">
            <NumberInput value={s.offset} unit="mm" step={0.5} onChange={(offset) => upd({ offset })} />
          </Field>
          <Slider value={s.offset} min={range.min} max={range.max} step={(range.max - range.min) / 1000 || 0.1} onChange={(offset) => upd({ offset })} format={(v) => v.toFixed(1)} />
          {s.source.startsWith("axial") && (
            <Field label="Rotation">
              <NumberInput value={s.rotation} unit="°" step={5} onChange={(rotation) => upd({ rotation })} />
            </Field>
          )}
        </Section>
        {this.multi && (
          <Section title="Sections">
            <Field label="Count">
              <NumberInput value={s.count} min={2} max={50} digits={0} onChange={(count) => upd({ count })} />
            </Field>
            <Field label="Spacing">
              <NumberInput value={s.spacing} min={0.01} unit="mm" onChange={(spacing) => upd({ spacing })} />
            </Field>
          </Section>
        )}
        <Section title="Result">
          <Stats rows={[["Curves", s.stats ? int(s.stats.loops) : "—"], ["Points", s.stats ? int(s.stats.points) : "—"]]} />
        </Section>
      </>
    );
  }

  async commit() {
    const frames = this.frames();
    if (!frames.length) return false;
    const ids: string[] = [];
    this.app.doc.transact(this.multi ? "Create sections" : "Create sketch", () => {
      for (const f of frames) {
        const sections = computeSection(this.app, this.meshId, f);
        const e = this.app.doc.add<SketchDoc>({
          id: newId("s"),
          kind: "sketch",
          name: this.app.doc.uniqueName(this.multi ? "Section" : "Sketch"),
          visible: true,
          frame: f,
          sketch: createSketch(),
          sections,
          meshId: this.meshId,
          planeSource: { type: "world", axis: "XY", offset: this.s.offset },
        });
        ids.push(e.id);
      }
    });
    if (!this.multi && ids[0]) {
      const id = ids[0];
      // open the sketch editor right after this tool closes
      setTimeout(() => editSketch(this.app, id), 0);
    } else this.app.toast("success", `Created ${ids.length} section sketches`);
  }

  deactivate() {
    window.clearTimeout(this.timer);
    this.overlay.dispose();
    this.sectionMat.dispose();
  }
}

export function startSectionTool(app: App, multi: boolean): void {
  app.startTool(new SectionTool(app, multi));
}

// ---------------------------------------------------------------------------
// Sketch editor
// ---------------------------------------------------------------------------

type DrawMode = "select" | "line" | "arc" | "circle" | "rect" | "spline" | "fitLine" | "fitArc" | "fitCircle" | "autoFit" | "trim" | "fillet" | "dimension";

interface SketchEditState {
  mode: DrawMode;
  selection: string[];
  dof: number;
  status: string;
  conflicting: string[];
  filletRadius: number;
  autoTol: number;
  clipMesh: boolean;
  showSection: boolean;
  construction: boolean;
  autoHV: boolean;
  version: number;
}

const MODES: { mode: DrawMode; label: string; icon: typeof Slash; tip: string }[] = [
  { mode: "select", label: "Select", icon: MousePointer2, tip: "Select and drag geometry (Esc)" },
  { mode: "line", label: "Line", icon: Slash, tip: "Polyline: click points, double-click or Esc to finish (L)" },
  { mode: "arc", label: "Arc", icon: CornerDownRight, tip: "3-point arc: start, end, then a point on the arc (A)" },
  { mode: "circle", label: "Circle", icon: CircleIcon, tip: "Centre, then radius (C)" },
  { mode: "rect", label: "Rect", icon: RectangleHorizontal, tip: "Two corners (R)" },
  { mode: "spline", label: "Spline", icon: SplineIcon, tip: "Click fit points, double-click to finish (S)" },
  { mode: "fitLine", label: "Fit line", icon: Minus, tip: "Drag a box over section points to fit a line" },
  { mode: "fitArc", label: "Fit arc", icon: Waypoints, tip: "Drag a box over section points to fit an arc" },
  { mode: "fitCircle", label: "Fit circle", icon: CircleIcon, tip: "Drag a box over section points to fit a circle" },
  { mode: "autoFit", label: "Auto fit", icon: WandSparkles, tip: "Click a section curve (or drag a box) to fit lines & arcs automatically" },
  { mode: "trim", label: "Trim", icon: Scissors, tip: "Click the part of an entity to remove" },
  { mode: "fillet", label: "Fillet", icon: Squircle, tip: "Click a corner shared by two lines" },
  { mode: "dimension", label: "Dimension", icon: Ruler, tip: "Click one or two items to dimension them (D)" },
];

const CONSTRAINT_BUTTONS: { type: Constraint["type"]; label: string; tip: string }[] = [
  { type: "horizontal", label: "H", tip: "Horizontal" },
  { type: "vertical", label: "V", tip: "Vertical" },
  { type: "coincident", label: "●", tip: "Coincident / point on curve" },
  { type: "parallel", label: "∥", tip: "Parallel" },
  { type: "perpendicular", label: "⊥", tip: "Perpendicular" },
  { type: "tangent", label: "T", tip: "Tangent" },
  { type: "equal", label: "=", tip: "Equal length / radius" },
  { type: "concentric", label: "◎", tip: "Concentric" },
  { type: "collinear", label: "—", tip: "Collinear" },
  { type: "midpoint", label: "M", tip: "Midpoint" },
  { type: "fix", label: "⚓", tip: "Fix in place" },
];

const DIM_LABEL: Record<string, string> = {
  distance: "Distance",
  horizontalDistance: "Horizontal",
  verticalDistance: "Vertical",
  radius: "Radius",
  diameter: "Diameter",
  angle: "Angle",
  length: "Length",
};

class SketchEditTool extends Tool<SketchEditState> {
  id = "sketch2d";
  label = "Sketch";
  okLabel: string | null = "Finish sketch";
  private overlay = new Overlay(this.app);
  private dynOverlay = new Overlay(this.app);
  private sketch: Sketch;
  private savedRef: Sketch;
  private matGeom = lineMaterial({ color: COLORS.sketch, width: 2.6, depthTest: false });
  private matSel = lineMaterial({ color: COLORS.highlight, width: 3.6, depthTest: false });
  private matCons = lineMaterial({ color: CONSTRUCTION_COLOR, width: 1.4, dashed: true, dashSize: 2, depthTest: false });
  private matSec = lineMaterial({ color: SECTION_COLOR, width: 1.5, depthTest: false });
  private matRubber = lineMaterial({ color: "#ffffff", width: 1.6, dashed: true, dashSize: 2.5, depthTest: false });
  private matConflict = lineMaterial({ color: "#f45d5d", width: 3.2, depthTest: false });
  private clipPlane = new THREE.Plane();
  /** points of an in-progress drawing operation (2D) */
  private pending: { pts: Vec2[]; ids: (string | null)[] } = { pts: [], ids: [] };
  private cursor: Vec2 | null = null;
  private drag: { pointId: string; moved: boolean; start: Vec2 } | null = null;
  private box: { a: Vec2; b: Vec2; screenA: { x: number; y: number } } | null = null;
  private lastClick = 0;

  constructor(app: App, private sketchId: string) {
    super(app, {
      mode: "select",
      selection: [],
      dof: 0,
      status: "ok",
      conflicting: [],
      filletRadius: 2,
      autoTol: 0.1,
      clipMesh: true,
      showSection: true,
      construction: false,
      autoHV: true,
      version: 0,
    });
    const e = this.entity;
    this.sketch = cloneSketch(e.sketch);
    this.savedRef = e.sketch;
    this.label = `Sketch — ${e.name}`;
    this.hint = "Draw over the orange section points. Right-drag rotates the view; use “Normal view” to look straight at the sketch plane.";
  }

  private get entity(): SketchDoc {
    const e = this.app.doc.get<SketchDoc>(this.sketchId);
    if (!e || e.kind !== "sketch") throw new Error("Sketch was deleted");
    return e;
  }

  private get frame(): Frame {
    return this.entity.frame;
  }

  activate() {
    this.app.sync?.view<SketchView>(this.sketchId)?.setEditing(true);
    this.normalView();
    this.applyClip();
    this.solve(false);
    this.redraw();
  }

  normalView() {
    const f = this.frame;
    const secs = this.entity.sections;
    let min: Vec2 = [Infinity, Infinity], max: Vec2 = [-Infinity, -Infinity];
    const add = (p: Vec2) => {
      min = [Math.min(min[0], p[0]), Math.min(min[1], p[1])];
      max = [Math.max(max[0], p[0]), Math.max(max[1], p[1])];
    };
    for (const s of secs) for (const p of s.points) add(p);
    for (const p of this.sketch.points) add([p.x, p.y]);
    let center: Vec3 = f.origin, radius: number | undefined;
    if (Number.isFinite(min[0])) {
      center = frameToWorld(f, [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2]);
      radius = Math.max(1, Math.hypot(max[0] - min[0], max[1] - min[1]) * 0.55);
    }
    this.app.viewer?.lookAlong(f.zAxis, f.yAxis, center, radius);
  }

  private applyClip() {
    const f = this.frame;
    for (const m of this.app.doc.ofKind("mesh")) {
      const v = this.app.sync?.view<import("../viewer/MeshView").MeshView>(m.id);
      if (!v) continue;
      if (this.s.clipMesh) {
        // hide everything in front of the sketch plane (camera side) so the section is visible
        const n = new THREE.Vector3(...f.zAxis).negate();
        this.clipPlane.setFromNormalAndCoplanarPoint(n, new THREE.Vector3(...f.origin));
        this.clipPlane.constant += 1e-3;
        v.material.clippingPlanes = [this.clipPlane];
      } else v.material.clippingPlanes = null;
      v.material.needsUpdate = true;
    }
    this.app.viewer?.requestRender();
  }

  // -------------------------------------------------------------------------
  // Model updates
  // -------------------------------------------------------------------------

  private solve(dragging: boolean, drag?: { pointId: string; target: Vec2 }) {
    if (!this.sketch.constraints.length && !drag) {
      this.set({ dof: this.sketch.points.length * 2, status: "ok", conflicting: [] });
      return;
    }
    try {
      const r = solveSketch(this.sketch, { drag, analyzeConflicts: !dragging });
      this.sketch = r.sketch;
      this.set({ dof: r.dof, status: r.status, conflicting: r.conflicting ?? [] });
    } catch (e) {
      console.error(e);
      this.set({ status: "failed" });
    }
  }

  /** Store the working sketch into the document (one undo step). */
  private save(label: string) {
    const copy = cloneSketch(this.sketch);
    this.savedRef = copy;
    this.app.doc.transact(label, () => this.app.doc.update<SketchDoc>(this.sketchId, { sketch: copy }));
    this.set({ version: this.s.version + 1 });
  }

  onDocumentChanged(ids: Set<string>) {
    if (!ids.has(this.sketchId)) return;
    const e = this.app.doc.get<SketchDoc>(this.sketchId);
    if (!e) {
      void this.app.endTool(false);
      return;
    }
    if (e.sketch !== this.savedRef) {
      // undo/redo changed the sketch underneath us
      this.sketch = cloneSketch(e.sketch);
      this.savedRef = e.sketch;
      this.set({ selection: [] });
      this.solve(false);
      this.redraw();
    }
  }

  private edit(label: string, fn: (s: Sketch) => void, solve = true) {
    const backup = cloneSketch(this.sketch);
    try {
      fn(this.sketch);
      if (solve) this.solve(false);
      this.save(label);
    } catch (e) {
      this.sketch = backup;
      this.app.toast("error", e instanceof Error ? e.message : String(e));
    }
    this.redraw();
  }

  // -------------------------------------------------------------------------
  // Coordinates & snapping
  // -------------------------------------------------------------------------

  private to2D(ray: THREE.Ray): Vec2 | null {
    const f = this.frame;
    const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(new THREE.Vector3(...f.zAxis), new THREE.Vector3(...f.origin));
    const p = ray.intersectPlane(plane, new THREE.Vector3());
    if (!p) return null;
    return worldToFrame(f, [p.x, p.y, p.z]);
  }

  /** sketch units per screen pixel near the given 2D point */
  private pxSize(p: Vec2): number {
    return this.app.viewer!.pixelSize(new THREE.Vector3(...frameToWorld(this.frame, p)));
  }

  private snap(p: Vec2, exclude?: string): { p: Vec2; id: string | null } {
    const tol = this.pxSize(p) * 10;
    const near = findNearest(this.sketch, p, tol, { pointsOnly: true, filter: (id) => id !== exclude });
    if (near) return { p: near.point, id: near.id };
    // snap to section points
    if (this.s.showSection) {
      let best: Vec2 | null = null, bd = this.pxSize(p) * 7;
      for (const s of this.entity.sections)
        for (const q of s.points) {
          const d = dist2(q, p);
          if (d < bd) {
            bd = d;
            best = q;
          }
        }
      if (best) return { p: [best[0], best[1]], id: null };
    }
    return { p, id: null };
  }

  private sectionPointsInBox(a: Vec2, b: Vec2): Vec2[] {
    const x0 = Math.min(a[0], b[0]), x1 = Math.max(a[0], b[0]), y0 = Math.min(a[1], b[1]), y1 = Math.max(a[1], b[1]);
    const out: Vec2[] = [];
    for (const s of this.entity.sections) for (const q of s.points) if (q[0] >= x0 && q[0] <= x1 && q[1] >= y0 && q[1] <= y1) out.push(q);
    return out;
  }

  /** Ordered points of the section chain inside the box (keeps polyline order for arcs/auto-fit). */
  private sectionChainInBox(a: Vec2, b: Vec2): { points: Vec2[]; closed: boolean }[] {
    const x0 = Math.min(a[0], b[0]), x1 = Math.max(a[0], b[0]), y0 = Math.min(a[1], b[1]), y1 = Math.max(a[1], b[1]);
    const inside = (q: Vec2) => q[0] >= x0 && q[0] <= x1 && q[1] >= y0 && q[1] <= y1;
    const out: { points: Vec2[]; closed: boolean }[] = [];
    for (const s of this.entity.sections) {
      const pts = s.points;
      if (pts.every(inside)) {
        out.push({ points: pts, closed: s.closed });
        continue;
      }
      // rotate closed loops so runs are not split at index 0
      let start = 0;
      if (s.closed) {
        const firstOut = pts.findIndex((q) => !inside(q));
        start = firstOut < 0 ? 0 : firstOut;
      }
      let run: Vec2[] = [];
      for (let k = 0; k < pts.length; k++) {
        const q = pts[(start + k) % pts.length];
        if (inside(q)) run.push(q);
        else if (run.length) {
          out.push({ points: run, closed: false });
          run = [];
        }
      }
      if (run.length) out.push({ points: run, closed: false });
    }
    return out.filter((r) => r.points.length >= 2);
  }

  // -------------------------------------------------------------------------
  // Pointer handling
  // -------------------------------------------------------------------------

  onPointerDown(e: ToolPointerEvent): boolean {
    if (e.button !== 0) return false;
    const p = this.to2D(e.ray);
    if (!p) return true;
    const now = performance.now();
    const dbl = now - this.lastClick < 320;
    this.lastClick = now;
    const mode = this.s.mode;
    const sn = this.snap(p);

    switch (mode) {
      case "select": {
        const hit = findNearest(this.sketch, p, this.pxSize(p) * 8);
        if (!hit) {
          if (!e.shift) this.set({ selection: [] });
          this.redraw();
          return true;
        }
        const sel = new Set(e.shift || e.ctrl ? this.s.selection : []);
        if (sel.has(hit.id) && (e.shift || e.ctrl)) sel.delete(hit.id);
        else sel.add(hit.id);
        this.set({ selection: [...sel] });
        if (hit.kind === "point") this.drag = { pointId: hit.id, moved: false, start: p };
        this.redraw();
        return true;
      }
      case "line": {
        if (dbl) {
          this.pending = { pts: [], ids: [] };
          this.redrawDynamic();
          return true;
        }
        if (!this.pending.pts.length) {
          this.pending = { pts: [sn.p], ids: [sn.id] };
        } else {
          const a = this.pending.pts.at(-1)!, aId = this.pending.ids.at(-1)!;
          if (dist2(a, sn.p) < 1e-9) return true;
          const closing = this.pending.ids[0] !== null && sn.id === this.pending.ids[0];
          let newEnd: string | null = null;
          this.edit("Line", (s) => {
            const ia = aId ?? addPoint(s, a);
            const ib = sn.id ?? addPoint(s, sn.p);
            const lid = addLine(s, ia, ib, { construction: this.s.construction });
            if (this.s.autoHV) this.autoHV(s, lid, a, sn.p);
            this.pending.ids[this.pending.ids.length - 1] = ia;
            newEnd = ib;
          });
          if (closing) this.pending = { pts: [], ids: [] };
          else {
            this.pending.pts.push(sn.p);
            this.pending.ids.push(newEnd);
          }
        }
        this.redrawDynamic();
        return true;
      }
      case "arc": {
        this.pending.pts.push(sn.p);
        this.pending.ids.push(sn.id);
        if (this.pending.pts.length === 3) {
          const [a, b, m] = this.pending.pts;
          const [ia, ib] = this.pending.ids;
          this.edit("Arc", (s) => {
            const r = addArc3(s, ia ?? a, m, ib ?? b, { construction: this.s.construction });
            if (!r) throw new Error("The three points are collinear");
          });
          this.pending = { pts: [], ids: [] };
        }
        this.redrawDynamic();
        return true;
      }
      case "circle": {
        this.pending.pts.push(sn.p);
        this.pending.ids.push(sn.id);
        if (this.pending.pts.length === 2) {
          const [c, r] = this.pending.pts;
          const cid = this.pending.ids[0];
          this.edit("Circle", (s) => {
            addCircle(s, cid ?? c, dist2(c, r), { construction: this.s.construction });
          });
          this.pending = { pts: [], ids: [] };
        }
        this.redrawDynamic();
        return true;
      }
      case "rect": {
        this.pending.pts.push(sn.p);
        if (this.pending.pts.length === 2) {
          const [a, b] = this.pending.pts;
          this.edit("Rectangle", (s) => {
            addRectangle(s, a, b, { construction: this.s.construction });
          });
          this.pending = { pts: [], ids: [] };
        }
        this.redrawDynamic();
        return true;
      }
      case "spline": {
        if (dbl && this.pending.pts.length >= 2) {
          this.finishSpline();
          return true;
        }
        this.pending.pts.push(sn.p);
        this.pending.ids.push(sn.id);
        this.redrawDynamic();
        return true;
      }
      case "fitLine":
      case "fitArc":
      case "fitCircle":
      case "autoFit": {
        this.box = { a: p, b: p, screenA: { x: e.x, y: e.y } };
        return true;
      }
      case "trim": {
        const hit = findNearest(this.sketch, p, this.pxSize(p) * 8, { entitiesOnly: true });
        if (!hit) return true;
        void lazy<typeof import("../sketch/ops")>("../sketch/ops.ts").then((ops) => this.edit("Trim", (s) => ops.trimAt(s, hit.id, p) as unknown as void));
        return true;
      }
      case "fillet": {
        const hit = findNearest(this.sketch, p, this.pxSize(p) * 10, { pointsOnly: true });
        if (!hit) return true;
        void lazy<typeof import("../sketch/ops")>("../sketch/ops.ts").then((ops) => this.edit("Fillet", (s) => ops.filletCorner(s, hit.id, this.s.filletRadius) as unknown as void));
        return true;
      }
      case "dimension": {
        const hit = findNearest(this.sketch, p, this.pxSize(p) * 8);
        if (!hit) {
          this.set({ selection: [] });
          this.redraw();
          return true;
        }
        const sel = [...this.s.selection, hit.id];
        this.set({ selection: sel });
        this.tryDimension(sel, hit.kind === "entity" && sel.length === 1);
        this.redraw();
        return true;
      }
    }
    return true;
  }

  onPointerMove(e: ToolPointerEvent): boolean {
    const p = this.to2D(e.ray);
    if (!p) return false;
    this.cursor = p;
    if (this.drag && (e.native.buttons & 1)) {
      this.drag.moved = true;
      const pt = getPoint(this.sketch, this.drag.pointId);
      if (pt && !pt.fixed) {
        if (this.sketch.constraints.length) this.solve(true, { pointId: this.drag.pointId, target: p });
        else {
          pt.x = p[0];
          pt.y = p[1];
        }
        this.redraw();
      }
      return true;
    }
    if (this.box && (e.native.buttons & 1)) {
      this.box.b = p;
      this.redrawDynamic();
      return true;
    }
    if (this.s.mode !== "select") {
      this.redrawDynamic();
      return true;
    }
    return false;
  }

  onPointerUp(e: ToolPointerEvent): boolean {
    if (this.drag) {
      const moved = this.drag.moved;
      this.drag = null;
      if (moved) {
        this.solve(false);
        this.save("Drag");
        this.redraw();
      }
      return true;
    }
    if (this.box) {
      const { a, b } = this.box;
      this.box = null;
      this.applyFitBox(a, b, e);
      this.redrawDynamic();
      return true;
    }
    return false;
  }

  onKeyDown(e: KeyboardEvent): boolean {
    const k = e.key.toLowerCase();
    if (e.key === "Escape") {
      if (this.pending.pts.length) {
        if (this.s.mode === "spline" && this.pending.pts.length >= 2) this.finishSpline();
        this.pending = { pts: [], ids: [] };
        this.redrawDynamic();
      } else if (this.s.mode !== "select") this.setMode("select");
      else if (this.s.selection.length) {
        this.set({ selection: [] });
        this.redraw();
      } else return false;
      return true;
    }
    if (e.key === "Enter" && this.s.mode === "spline" && this.pending.pts.length >= 2) {
      this.finishSpline();
      return true;
    }
    if (e.key === "Delete" || e.key === "Backspace") {
      this.deleteSelection();
      return true;
    }
    if (e.ctrlKey || e.metaKey || e.altKey) return false;
    const shortcuts: Record<string, DrawMode> = { l: "line", a: "arc", c: "circle", r: "rect", s: "spline", d: "dimension", t: "trim" };
    if (shortcuts[k]) {
      this.setMode(shortcuts[k]);
      return true;
    }
    if (k === "n") {
      this.normalView();
      return true;
    }
    return false;
  }

  private setMode(mode: DrawMode) {
    this.pending = { pts: [], ids: [] };
    this.set({ mode, selection: mode === "dimension" ? [] : this.s.selection });
    this.app.setStatus(MODES.find((m) => m.mode === mode)?.tip ?? "");
    this.redrawDynamic();
  }

  private finishSpline() {
    const pts = this.pending.pts.slice();
    const ids = this.pending.ids.slice();
    this.pending = { pts: [], ids: [] };
    if (pts.length < 2) return;
    const closed = ids[0] !== null && ids.at(-1) === ids[0];
    this.edit("Spline", (s) => {
      const refs = pts.map((p, i) => ids[i] ?? p);
      if (closed) refs.pop();
      addSpline(s, refs, { closed, construction: this.s.construction });
    });
  }

  private autoHV(s: Sketch, lineId: string, a: Vec2, b: Vec2) {
    const ang = Math.abs(Math.atan2(b[1] - a[1], b[0] - a[0]));
    const tol = (1.5 * Math.PI) / 180;
    const type = ang < tol || Math.abs(ang - Math.PI) < tol ? "horizontal" : Math.abs(ang - Math.PI / 2) < tol ? "vertical" : null;
    if (type && !validateConstraint(s, { type, refs: [lineId] })) addConstraint(s, { type, refs: [lineId] });
  }

  /** Merge a newly created endpoint with an existing point at the same location. */
  private pointAt(s: Sketch, p: Vec2): string {
    const near = findNearest(s, p, this.pxSize(p) * 6, { pointsOnly: true });
    return near ? near.id : addPoint(s, p);
  }

  private async applyFitBox(a: Vec2, b: Vec2, e: ToolPointerEvent) {
    const mode = this.s.mode;
    const tiny = dist2(a, b) < this.pxSize(a) * 4;
    let chains = tiny ? [] : this.sectionChainInBox(a, b);
    if (tiny && mode === "autoFit") {
      // click on a section curve: take the whole polyline
      let best: SectionPolyline | null = null, bd = this.pxSize(a) * 10;
      for (const s of this.entity.sections)
        for (const q of s.points) {
          const d = dist2(q, a);
          if (d < bd) {
            bd = d;
            best = s;
          }
        }
      if (best) chains = [{ points: best.points, closed: best.closed }];
    }
    const pts = chains.flatMap((c) => c.points);
    if (pts.length < 3) {
      if (!tiny) this.app.toast("info", "Not enough section points in the box");
      return;
    }
    const fit = await lazy<typeof import("../sketch/fit2d")>("../sketch/fit2d.ts");
    if (mode === "fitLine") {
      const r = fit.fitLine2D(pts);
      this.edit("Fit line", (s) => {
        const lid = addLine(s, this.pointAt(s, r.a), this.pointAt(s, r.b), { construction: this.s.construction });
        if (this.s.autoHV) this.autoHV(s, lid, r.a, r.b);
      });
      this.app.setStatus(`Line fitted: RMS ${r.stats.rms.toFixed(4)} mm over ${pts.length} points`);
    } else if (mode === "fitCircle") {
      const r = fit.fitCircle2D(pts);
      this.edit("Fit circle", (s) => {
        addCircle(s, addPoint(s, r.center), r.radius, { construction: this.s.construction });
      });
      this.app.setStatus(`Circle fitted: Ø${(2 * r.radius).toFixed(3)} mm, RMS ${r.stats.rms.toFixed(4)} mm`);
    } else if (mode === "fitArc") {
      const chain = chains.sort((x, y) => y.points.length - x.points.length)[0].points;
      const r = fit.fitArc2D(chain);
      this.edit("Fit arc", (s) => {
        const id = addArc3(s, this.pointAt(s, r.start), r.mid, this.pointAt(s, r.end), { construction: this.s.construction });
        if (!id) throw new Error("Arc fit failed (points are collinear)");
      });
      this.app.setStatus(`Arc fitted: R${r.radius.toFixed(3)} mm, RMS ${r.stats.rms.toFixed(4)} mm`);
    } else if (mode === "autoFit") {
      const tol = this.s.autoTol;
      let created = 0;
      this.edit("Auto fit", (s) => {
        for (const c of chains) {
          const sub = fit.autoSketch(c.points, { tolerance: tol, closed: c.closed, constrainHV: this.s.autoHV });
          created += mergeSketch(s, sub, (p) => this.pointAt(s, p));
        }
      });
      this.app.setStatus(`Auto fit created ${created} entities (tolerance ${tol} mm)`);
    }
    void e;
  }

  private tryDimension(sel: string[], waitForSecond: boolean) {
    const s = this.sketch;
    const kinds = sel.map((id) => (getPoint(s, id) ? "point" : getEntity(s, id)?.type ?? "missing"));
    let input: { type: DimensionConstraint["type"]; refs: string[] } | null = null;
    if (sel.length === 1) {
      if (kinds[0] === "circle") input = { type: "diameter", refs: sel };
      else if (kinds[0] === "arc") input = { type: "radius", refs: sel };
      else if (waitForSecond) return; // line: wait — a second click may make it a distance/angle
    } else if (sel.length === 2) {
      if (kinds[0] === "line" && kinds[1] === "line") {
        const ang = measure(s, { type: "angle", refs: sel });
        input = ang !== null && Math.abs(Math.sin(ang)) > 1e-3 ? { type: "angle", refs: sel } : { type: "distance", refs: sel };
      } else input = { type: "distance", refs: sel };
    }
    if (sel.length === 1 && kinds[0] === "line") input = { type: "distance", refs: sel };
    if (!input) return;
    const err = validateConstraint(s, input);
    if (err) {
      this.app.toast("error", err);
      this.set({ selection: [] });
      return;
    }
    this.edit("Dimension", (sk) => {
      addConstraint(sk, input!);
    });
    this.set({ selection: [] });
  }

  private deleteSelection() {
    const sel = this.s.selection;
    if (!sel.length) return;
    this.edit("Delete", (s) => {
      const ents = sel.filter((id) => getEntity(s, id));
      const cons = sel.filter((id) => s.constraints.some((c) => c.id === id));
      const pts = sel.filter((id) => getPoint(s, id));
      for (const c of cons) deleteConstraint(s, c);
      if (ents.length) deleteEntities(s, ents);
      for (const p of pts) if (getPoint(s, p)) deletePoint(s, p);
    });
    this.set({ selection: [] });
  }

  addConstraintFromSelection(type: Constraint["type"]) {
    const refs = this.s.selection.filter((id) => getPoint(this.sketch, id) || getEntity(this.sketch, id));
    const input = { type, refs } as { type: Constraint["type"]; refs: string[] };
    // single line + horizontal/vertical; two points + horizontal/vertical also valid
    const err = validateConstraint(this.sketch, input);
    if (err) {
      this.app.toast("error", err);
      return;
    }
    this.edit(`Constraint: ${type}`, (s) => {
      addConstraint(s, input as never);
    });
  }

  setDimensionValue(id: string, value: number) {
    this.edit("Edit dimension", (s) => {
      const c = s.constraints.find((k) => k.id === id);
      if (c && isDimension(c)) c.value = c.type === "angle" ? (value * Math.PI) / 180 : value;
    });
  }

  toggleConstruction() {
    const ids = this.s.selection.filter((id) => getEntity(this.sketch, id));
    if (!ids.length) return;
    this.edit("Construction", (s) => {
      for (const id of ids) {
        const e = getEntity(s, id)!;
        e.construction = !e.construction;
      }
    }, false);
  }

  // -------------------------------------------------------------------------
  // Drawing
  // -------------------------------------------------------------------------

  private redraw() {
    this.overlay.clear();
    const f = this.frame;
    const s = this.sketch;
    // section reference points
    if (this.s.showSection) {
      const secs = sectionsTo3D(f, this.entity.sections);
      const pairs = polylinesToPairs(secs);
      if (pairs.length) this.overlay.add(fatSegments(pairs, this.matSec)).renderOrder = 20;
    }
    const idx = indexSketch(s).points;
    const sel = new Set(this.s.selection);
    const conflictEntities = new Set<string>();
    for (const cid of this.s.conflicting) for (const r of s.constraints.find((c) => c.id === cid)?.refs ?? []) conflictEntities.add(r);
    const groups: Record<string, { points: Vec3[] }[]> = { geom: [], sel: [], cons: [], conflict: [] };
    for (const ent of s.entities) {
      let segs;
      try {
        segs = entityToSegments(s, ent, idx);
      } catch {
        continue;
      }
      const key = sel.has(ent.id) ? "sel" : conflictEntities.has(ent.id) ? "conflict" : ent.construction ? "cons" : "geom";
      for (const seg of segs) groups[key].push({ points: sampleSegment2D(seg, 0.02).map((p) => frameToWorld(f, p)) });
    }
    const add = (k: string, mat: typeof this.matGeom, order: number) => {
      if (groups[k].length) this.overlay.add(fatSegments(polylinesToPairs(groups[k]), mat)).renderOrder = order;
    };
    add("cons", this.matCons, 21);
    add("geom", this.matGeom, 22);
    add("conflict", this.matConflict, 23);
    add("sel", this.matSel, 24);
    // points
    const normal: number[] = [], selected: number[] = [], fixed: number[] = [];
    for (const p of s.points) {
      const w = frameToWorld(f, [p.x, p.y]);
      (sel.has(p.id) ? selected : p.fixed ? fixed : normal).push(...w);
    }
    if (normal.length) this.overlay.add(pointSprites(normal, "#e8eef5", 7, { depthTest: false })).renderOrder = 26;
    if (fixed.length) this.overlay.add(pointSprites(fixed, "#ff5d8f", 8, { depthTest: false })).renderOrder = 26;
    if (selected.length) this.overlay.add(pointSprites(selected, COLORS.highlight, 9, { depthTest: false })).renderOrder = 27;
    // dimension labels
    for (const c of s.constraints) {
      if (!isDimension(c)) continue;
      const anchor = this.dimensionAnchor(c);
      if (!anchor) continue;
      const txt = c.type === "angle" ? `${((c.value * 180) / Math.PI).toFixed(2)}°` : `${c.type === "diameter" ? "Ø" : c.type === "radius" ? "R" : ""}${c.value.toFixed(3)}`;
      const label = this.overlay.label(txt, frameToWorld(f, anchor), `viewer-label dim${sel.has(c.id) ? " warn" : ""}`);
      if (label) {
        label.el.onclick = () => {
          this.set({ selection: [c.id] });
          this.redraw();
        };
      }
    }
    this.redrawDynamic();
  }

  private dimensionAnchor(c: DimensionConstraint): Vec2 | null {
    const s = this.sketch;
    const pts: Vec2[] = [];
    for (const r of c.refs) {
      const p = getPoint(s, r);
      if (p) {
        pts.push([p.x, p.y]);
        continue;
      }
      const e = getEntity(s, r);
      if (!e) continue;
      if (e.type === "line") {
        const a = getPoint(s, e.p1)!, b = getPoint(s, e.p2)!;
        pts.push([(a.x + b.x) / 2, (a.y + b.y) / 2]);
      } else if (e.type === "circle" || e.type === "arc") {
        const ctr = getPoint(s, e.center)!;
        const r = e.type === "circle" ? e.radius : Math.hypot(getPoint(s, e.start)!.x - ctr.x, getPoint(s, e.start)!.y - ctr.y);
        pts.push([ctr.x + r * 0.7071, ctr.y + r * 0.7071]);
      }
    }
    if (!pts.length) return null;
    const m: Vec2 = [pts.reduce((a, p) => a + p[0], 0) / pts.length, pts.reduce((a, p) => a + p[1], 0) / pts.length];
    const off = c.labelOffset ?? [0, 0];
    return [m[0] + off[0], m[1] + off[1]];
  }

  private redrawDynamic() {
    this.dynOverlay.clear();
    const f = this.frame;
    const cur = this.cursor ? this.snap(this.cursor).p : null;
    const segs: { points: Vec3[] }[] = [];
    const P = this.pending.pts;
    const mode = this.s.mode;
    if (cur && P.length) {
      if (mode === "line" || mode === "spline") segs.push({ points: [...P, cur].map((p) => frameToWorld(f, p)) });
      if (mode === "circle") {
        const r = dist2(P[0], cur);
        segs.push({ points: sampleSegment2D({ type: "circle", center: P[0], radius: r }).map((p) => frameToWorld(f, p)) });
      }
      if (mode === "rect") {
        const [a] = P;
        segs.push({ points: [a, [cur[0], a[1]], cur, [a[0], cur[1]], a].map((p) => frameToWorld(f, p as Vec2)) });
      }
      if (mode === "arc") {
        if (P.length === 1) segs.push({ points: [P[0], cur].map((p) => frameToWorld(f, p)) });
        if (P.length === 2) segs.push({ points: sampleSegment2D({ type: "arc", start: P[0], mid: cur, end: P[1] }).map((p) => frameToWorld(f, p)) });
      }
    }
    if (this.box) {
      const { a, b } = this.box;
      segs.push({ points: [a, [b[0], a[1]], b, [a[0], b[1]], a].map((p) => frameToWorld(f, p as Vec2)) });
    }
    if (segs.length) this.dynOverlay.add(fatSegments(polylinesToPairs(segs), this.matRubber)).renderOrder = 30;
    if (cur && mode !== "select") {
      const snapped = this.cursor ? this.snap(this.cursor) : null;
      if (snapped && (snapped.id || snapped.p !== this.cursor)) this.dynOverlay.add(pointSprites(frameToWorld(f, snapped.p), snapped.id ? "#ffd23f" : SECTION_COLOR, 10, { depthTest: false })).renderOrder = 31;
    }
  }

  // -------------------------------------------------------------------------
  // Panel
  // -------------------------------------------------------------------------

  renderPanel() {
    const st = this.s;
    const sk = this.sketch;
    const dims = sk.constraints.filter(isDimension) as DimensionConstraint[];
    const geo = sk.constraints.filter((c) => !isDimension(c));
    const sel = new Set(st.selection);
    const statusRow: [string, string, ("good" | "warn" | "bad")?] =
      st.status === "ok"
        ? ["Solver", st.dof === 0 && sk.entities.length ? "fully constrained" : `${st.dof} DOF free`, st.dof === 0 && sk.entities.length ? "good" : undefined]
        : ["Solver", st.status === "overconstrained" ? "over-constrained" : "failed", "bad"];
    return (
      <>
        <Section title="Draw">
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 3 }}>
            {MODES.map(({ mode, label, icon: Icon, tip }) => (
              <button key={mode} className={`rbtn${st.mode === mode ? " active" : ""}`} style={{ width: "auto", padding: "5px 2px" }} title={tip} onClick={() => this.setMode(mode)}>
                <Icon size={17} strokeWidth={1.7} />
                <span style={{ fontSize: 10.5 }}>{label}</span>
              </button>
            ))}
          </div>
          <div style={{ height: 8 }} />
          {st.mode === "fillet" && (
            <Field label="Fillet radius">
              <NumberInput value={st.filletRadius} min={0.001} unit="mm" onChange={(filletRadius) => this.set({ filletRadius })} />
            </Field>
          )}
          {st.mode === "autoFit" && (
            <Field label="Tolerance">
              <NumberInput value={st.autoTol} min={0.001} unit="mm" step={0.05} onChange={(autoTol) => this.set({ autoTol })} />
            </Field>
          )}
          <Check checked={st.construction} onChange={(construction) => this.set({ construction })}>
            Construction geometry
          </Check>
          <Check checked={st.autoHV} onChange={(autoHV) => this.set({ autoHV })}>
            Auto horizontal / vertical
          </Check>
        </Section>
        <Section title="Constraints" right={<span style={{ textTransform: "none" }}>{st.selection.length} selected</span>}>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 3 }}>
            {CONSTRAINT_BUTTONS.map((c) => (
              <button key={c.type} className="btn small" style={{ minWidth: 30, padding: "0 6px" }} title={c.tip} onClick={() => this.addConstraintFromSelection(c.type)}>
                {c.label}
              </button>
            ))}
          </div>
          <div className="btn-row" style={{ marginTop: 6 }}>
            <button className="btn small" onClick={() => this.toggleConstruction()} disabled={!st.selection.length}>
              Toggle construction
            </button>
            <button className="btn small danger" onClick={() => this.deleteSelection()} disabled={!st.selection.length}>
              <Trash2 size={12} /> Delete
            </button>
          </div>
          <div style={{ height: 8 }} />
          <Stats rows={[statusRow, ["Entities", int(sk.entities.length)], ["Constraints", int(geo.length)]]} />
          {st.conflicting.length > 0 && <div className="note err" style={{ marginTop: 6 }}>Conflicting constraints are shown in red — delete one of them.</div>}
        </Section>
        {dims.length > 0 && (
          <Section title="Dimensions">
            {dims.map((d) => (
              <div key={d.id} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 5 }}>
                <span
                  style={{ width: 82, fontSize: 12, color: sel.has(d.id) ? "var(--orange)" : "var(--text-2)", cursor: "pointer" }}
                  onClick={() => {
                    this.set({ selection: [d.id] });
                    this.redraw();
                  }}
                >
                  {DIM_LABEL[d.type] ?? d.type}
                </span>
                <div style={{ flex: 1 }}>
                  <NumberInput
                    value={d.type === "angle" ? (d.value * 180) / Math.PI : d.value}
                    unit={d.type === "angle" ? "°" : "mm"}
                    min={d.type === "angle" ? 0 : 0}
                    onChange={(v) => this.setDimensionValue(d.id, v)}
                  />
                </div>
                <button
                  className="tree-action always"
                  style={{ opacity: 1 }}
                  title="Delete dimension"
                  onClick={() => this.edit("Delete dimension", (s) => deleteConstraint(s, d.id))}
                >
                  <Trash2 size={13} />
                </button>
              </div>
            ))}
          </Section>
        )}
        <Section title="View">
          <div className="btn-row">
            <button className="btn small" onClick={() => this.normalView()}>
              <Focus size={13} /> Normal view (N)
            </button>
          </div>
          <div style={{ height: 8 }} />
          <Check
            checked={st.clipMesh}
            onChange={(clipMesh) => {
              this.set({ clipMesh });
              this.applyClip();
            }}
          >
            Cut away the mesh in front of the plane
          </Check>
          <Check
            checked={st.showSection}
            onChange={(showSection) => {
              this.set({ showSection });
              this.redraw();
            }}
          >
            Show section points
          </Check>
        </Section>
      </>
    );
  }

  async commit() {
    this.save("Finish sketch");
  }

  deactivate() {
    for (const m of this.app.doc.ofKind("mesh")) {
      const v = this.app.sync?.view<import("../viewer/MeshView").MeshView>(m.id);
      if (v) {
        v.material.clippingPlanes = null;
        v.material.needsUpdate = true;
      }
    }
    this.app.sync?.view<SketchView>(this.sketchId)?.setEditing(false);
    this.overlay.dispose();
    this.dynOverlay.dispose();
    for (const m of [this.matGeom, this.matSel, this.matCons, this.matSec, this.matRubber, this.matConflict]) m.dispose();
  }
}

/** Copy another sketch's points/entities/constraints into `s`, merging coincident endpoints. */
function mergeSketch(s: Sketch, sub: Sketch, pointAt: (p: Vec2) => string): number {
  const map = new Map<string, string>();
  const endpointIds = new Set<string>();
  for (const e of sub.entities) {
    if (e.type === "line") [e.p1, e.p2].forEach((id) => endpointIds.add(id));
    if (e.type === "arc") [e.start, e.end].forEach((id) => endpointIds.add(id));
  }
  for (const p of sub.points) {
    map.set(p.id, endpointIds.has(p.id) ? pointAt([p.x, p.y]) : addPoint(s, [p.x, p.y]));
  }
  const entMap = new Map<string, string>();
  for (const e of sub.entities) {
    let id: string;
    if (e.type === "line") id = addLine(s, map.get(e.p1)!, map.get(e.p2)!, { construction: e.construction });
    else if (e.type === "arc") id = addArc(s, map.get(e.center)!, map.get(e.start)!, map.get(e.end)!, { construction: e.construction });
    else if (e.type === "circle") id = addCircle(s, map.get(e.center)!, e.radius, { construction: e.construction });
    else id = addSpline(s, e.points.map((p) => map.get(p)!), { closed: e.closed, construction: e.construction });
    entMap.set(e.id, id);
  }
  for (const c of sub.constraints) {
    const refs = c.refs.map((r) => map.get(r) ?? entMap.get(r) ?? r);
    try {
      addConstraint(s, { ...c, id: undefined, refs } as never);
    } catch {
      /* skip constraints that do not apply after merging */
    }
  }
  return sub.entities.length;
}

export function editSketch(app: App, id: string): void {
  app.startTool(new SketchEditTool(app, id));
}
