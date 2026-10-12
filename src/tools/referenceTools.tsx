/** Reference construction tools: lines, points, derived planes and the symmetry plane analyser. */
import * as THREE from "three";
import type { App } from "../app/App";
import { newId } from "../core/document";
import type { MeshEntity, PrimitiveEntity, RefLineEntity, RefPointEntity } from "../core/entities";
import type { LineDef, PlaneDef, Vec3 } from "../core/types";
import { cross, dot, normalize, rotationAxisAngle, sub, transformDir } from "../core/vec";
import {
  axisOf, intersectLinePlane, intersectPlanes, intersectThreePlanes, lineFrom2Points, midPlane, offsetPlane,
  planeFrom3Points, planePerpendicularToLine, planeThroughLineAndPoint,
} from "../fit/reference";
import { Tool, type ToolClass, type ToolPointerEvent } from "./Tool";
import { Overlay } from "./overlay";
import { Section, Field, Select, NumberInput, Segmented, Stats, Slider, mm } from "../ui/widgets";
import { fatSegments, lineMaterial, pointSprites, translucentMaterial } from "../viewer/draw";
import { deviationColors, robustRange } from "../analysis/colormap";
import type { MeshView } from "../viewer/MeshView";

const planesOf = (app: App) =>
  app.doc.ofKind("primitive").filter((e) => e.def.kind === "plane").map((e) => ({ value: e.id, label: e.name }));
const axesOf = (app: App) => [
  ...app.doc.ofKind("primitive").filter((e) => e.def.kind === "cylinder" || e.def.kind === "cone").map((e) => ({ value: e.id, label: `${e.name} axis` })),
  ...app.doc.ofKind("refline").map((e) => ({ value: e.id, label: e.name })),
];

function planeDef(app: App, id: string): PlaneDef | null {
  const e = app.doc.get<PrimitiveEntity>(id);
  return e?.kind === "primitive" && e.def.kind === "plane" ? e.def : null;
}

function lineDef(app: App, id: string): { line: LineDef; tMin: number; tMax: number } | null {
  const e = app.doc.get(id);
  if (e?.kind === "refline") return { line: (e as RefLineEntity).def, tMin: (e as RefLineEntity).tMin, tMax: (e as RefLineEntity).tMax };
  if (e?.kind === "primitive") {
    const d = (e as PrimitiveEntity).def;
    if (d.kind === "cylinder" || d.kind === "cone") {
      const ext = (d.hMax - d.hMin) * 0.2 + 1;
      return { line: axisOf(d), tMin: d.hMin - ext, tMax: d.hMax + ext };
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Reference line
// ---------------------------------------------------------------------------

type LineMode = "axis" | "planes" | "points";

class RefLineTool extends Tool<{ mode: LineMode; axis: string; planeA: string; planeB: string; picks: Vec3[] }> {
  id = "refLine";
  label = "Reference Line";
  hint = "Create a line from a cylinder/cone axis, the intersection of two planes, or two picked points.";
  private overlay = new Overlay(this.app);
  private mat = lineMaterial({ color: "#00c2a8", width: 3, dashed: true, dashSize: 4 });

  constructor(app: App) {
    const ax = axesOf(app), pl = planesOf(app);
    super(app, { mode: ax.length ? "axis" : pl.length >= 2 ? "planes" : "points", axis: ax[0]?.value ?? "", planeA: pl[0]?.value ?? "", planeB: pl[1]?.value ?? "", picks: [] });
  }

  activate() {
    this.preview();
  }

  result(): { line: LineDef; tMin: number; tMax: number } | null {
    const s = this.s;
    const size = this.app.sceneSize() * 0.6;
    if (s.mode === "axis") return lineDef(this.app, s.axis);
    if (s.mode === "planes") {
      const a = planeDef(this.app, s.planeA), b = planeDef(this.app, s.planeB);
      if (!a || !b) return null;
      const l = intersectPlanes(a, b);
      return l ? { line: l, tMin: -size / 2, tMax: size / 2 } : null;
    }
    if (s.picks.length === 2) {
      const l = lineFrom2Points(s.picks[0], s.picks[1]);
      if (!l) return null;
      const len = dot(sub(s.picks[1], s.picks[0]), l.direction);
      return { line: l, tMin: Math.min(0, len), tMax: Math.max(0, len) };
    }
    return null;
  }

  private preview() {
    this.overlay.clear();
    const r = this.result();
    if (this.s.picks.length) this.overlay.add(pointSprites(this.s.picks.flat(), "#ffd23f", 9, { depthTest: false }));
    if (!r) return;
    const { origin: o, direction: d } = r.line;
    this.overlay.add(fatSegments([...o.map((v, i) => v + d[i] * r.tMin), ...o.map((v, i) => v + d[i] * r.tMax)], this.mat));
  }

  onPointerDown(e: ToolPointerEvent) {
    if (this.s.mode !== "points" || e.button !== 0) return false;
    const hit = e.pick();
    if (!hit) return true;
    const picks = this.s.picks.length >= 2 ? [hit.point] : [...this.s.picks, hit.point];
    this.set({ picks });
    this.preview();
    return true;
  }

  renderPanel() {
    const s = this.s;
    const upd = (p: Partial<typeof s>) => {
      this.set(p);
      this.preview();
    };
    return (
      <Section title="Definition">
        <Segmented
          value={s.mode}
          options={[{ value: "axis", label: "Axis" }, { value: "planes", label: "2 planes" }, { value: "points", label: "2 points" }]}
          onChange={(mode) => upd({ mode, picks: [] })}
        />
        <div style={{ height: 10 }} />
        {s.mode === "axis" && (
          <Field label="Cylinder / cone">
            <Select value={s.axis} options={axesOf(this.app)} placeholder="Choose…" onChange={(axis) => upd({ axis })} />
          </Field>
        )}
        {s.mode === "planes" && (
          <>
            <Field label="Plane A">
              <Select value={s.planeA} options={planesOf(this.app)} placeholder="Choose…" onChange={(planeA) => upd({ planeA })} />
            </Field>
            <Field label="Plane B">
              <Select value={s.planeB} options={planesOf(this.app)} placeholder="Choose…" onChange={(planeB) => upd({ planeB })} />
            </Field>
          </>
        )}
        {s.mode === "points" && <div className="note">Click two points on the model ({s.picks.length}/2).</div>}
      </Section>
    );
  }

  async commit() {
    const r = this.result();
    if (!r) {
      this.app.toast("error", "The line is not fully defined");
      return false;
    }
    const e = this.app.doc.add<RefLineEntity>({ id: newId("l"), kind: "refline", name: this.app.doc.uniqueName("Line"), visible: true, def: r.line, tMin: r.tMin, tMax: r.tMax });
    this.app.setActiveEntity(e.id);
  }

  deactivate() {
    this.overlay.dispose();
    this.mat.dispose();
  }
}

// ---------------------------------------------------------------------------
// Reference point
// ---------------------------------------------------------------------------

type PointMode = "pick" | "center" | "planes3" | "linePlane";

class RefPointTool extends Tool<{ mode: PointMode; prim: string; a: string; b: string; c: string; picks: Vec3[] }> {
  id = "refPoint";
  label = "Reference Point";
  hint = "Pick points on the model, or construct them from a sphere/cone, three planes or a line and a plane.";
  private overlay = new Overlay(this.app);

  constructor(app: App) {
    super(app, { mode: "pick", prim: "", a: "", b: "", c: "", picks: [] });
  }

  activate() {
    this.preview();
  }

  private constructed(): Vec3 | null {
    const s = this.s;
    if (s.mode === "center") {
      const e = this.app.doc.get<PrimitiveEntity>(s.prim);
      if (e?.def.kind === "sphere") return e.def.center;
      if (e?.def.kind === "cone") return e.def.apex;
      return null;
    }
    if (s.mode === "planes3") {
      const a = planeDef(this.app, s.a), b = planeDef(this.app, s.b), c = planeDef(this.app, s.c);
      return a && b && c ? intersectThreePlanes(a, b, c) : null;
    }
    if (s.mode === "linePlane") {
      const l = lineDef(this.app, s.a), p = planeDef(this.app, s.b);
      return l && p ? intersectLinePlane(l.line, p) : null;
    }
    return null;
  }

  private points(): Vec3[] {
    if (this.s.mode === "pick") return this.s.picks;
    const p = this.constructed();
    return p ? [p] : [];
  }

  private preview() {
    this.overlay.clear();
    const pts = this.points();
    if (pts.length) this.overlay.add(pointSprites(pts.flat(), "#ffd23f", 11, { depthTest: false })).renderOrder = 30;
  }

  onPointerDown(e: ToolPointerEvent) {
    if (this.s.mode !== "pick" || e.button !== 0) return false;
    const hit = e.pick();
    if (hit) {
      this.set({ picks: [...this.s.picks, hit.point] });
      this.preview();
    }
    return true;
  }

  renderPanel() {
    const s = this.s;
    const upd = (p: Partial<typeof s>) => {
      this.set(p);
      this.preview();
    };
    const centers = this.app.doc
      .ofKind("primitive")
      .filter((e) => e.def.kind === "sphere" || e.def.kind === "cone")
      .map((e) => ({ value: e.id, label: `${e.name} ${e.def.kind === "sphere" ? "centre" : "apex"}` }));
    const pl = planesOf(this.app);
    return (
      <Section title="Definition">
        <Select
          value={s.mode}
          options={[
            { value: "pick", label: "Pick on model" },
            { value: "center", label: "Sphere centre / cone apex" },
            { value: "planes3", label: "Intersection of 3 planes" },
            { value: "linePlane", label: "Line ∩ plane" },
          ]}
          onChange={(mode) => upd({ mode })}
        />
        <div style={{ height: 10 }} />
        {s.mode === "pick" && (
          <>
            <div className="note">Click on the model to place points ({s.picks.length} placed).</div>
            {s.picks.length > 0 && (
              <button className="btn small" style={{ marginTop: 8 }} onClick={() => upd({ picks: s.picks.slice(0, -1) })}>
                Remove last
              </button>
            )}
          </>
        )}
        {s.mode === "center" && (
          <Field label="Primitive">
            <Select value={s.prim} options={centers} placeholder="Choose…" onChange={(prim) => upd({ prim })} />
          </Field>
        )}
        {s.mode === "planes3" &&
          (["a", "b", "c"] as const).map((k, i) => (
            <Field key={k} label={`Plane ${i + 1}`}>
              <Select value={s[k]} options={pl} placeholder="Choose…" onChange={(v) => upd({ [k]: v } as Partial<typeof s>)} />
            </Field>
          ))}
        {s.mode === "linePlane" && (
          <>
            <Field label="Line / axis">
              <Select value={s.a} options={axesOf(this.app)} placeholder="Choose…" onChange={(a) => upd({ a })} />
            </Field>
            <Field label="Plane">
              <Select value={s.b} options={pl} placeholder="Choose…" onChange={(b) => upd({ b })} />
            </Field>
          </>
        )}
      </Section>
    );
  }

  async commit() {
    const pts = this.points();
    if (!pts.length) {
      this.app.toast("error", "No point defined");
      return false;
    }
    this.app.doc.transact("Add points", () => {
      for (const p of pts)
        this.app.doc.add<RefPointEntity>({ id: newId("pt"), kind: "refpoint", name: this.app.doc.uniqueName("Point"), visible: true, def: { kind: "point", position: p } });
    });
  }

  deactivate() {
    this.overlay.dispose();
  }
}

// ---------------------------------------------------------------------------
// Derived planes
// ---------------------------------------------------------------------------

type PlaneMode = "offset" | "world" | "points3" | "mid" | "lineThroughPoint" | "perpLine";

class RefPlaneTool extends Tool<{ mode: PlaneMode; a: string; b: string; offset: number; world: "XY" | "YZ" | "ZX"; picks: Vec3[] }> {
  id = "refPlane";
  label = "Construct Plane";
  hint = "Create planes from existing references: offset, mid-plane, 3 points, through an axis, or world planes.";
  private overlay = new Overlay(this.app);

  constructor(app: App) {
    const pl = planesOf(app);
    super(app, { mode: pl.length ? "offset" : "world", a: pl[0]?.value ?? "", b: pl[1]?.value ?? "", offset: 10, world: "XY", picks: [] });
  }

  activate() {
    this.preview();
  }

  plane(): PlaneDef | null {
    const s = this.s;
    switch (s.mode) {
      case "offset": {
        const p = planeDef(this.app, s.a);
        return p ? offsetPlane(p, s.offset) : null;
      }
      case "world": {
        const n: Vec3 = s.world === "XY" ? [0, 0, 1] : s.world === "YZ" ? [1, 0, 0] : [0, 1, 0];
        return { kind: "plane", origin: [n[0] * s.offset, n[1] * s.offset, n[2] * s.offset], normal: n };
      }
      case "points3":
        return s.picks.length === 3 ? planeFrom3Points(s.picks[0], s.picks[1], s.picks[2]) : null;
      case "mid": {
        const a = planeDef(this.app, s.a), b = planeDef(this.app, s.b);
        return a && b ? midPlane(a, b) : null;
      }
      case "lineThroughPoint": {
        const l = lineDef(this.app, s.a);
        return l && s.picks[0] ? planeThroughLineAndPoint(l.line, s.picks[0]) : null;
      }
      case "perpLine": {
        const l = lineDef(this.app, s.a);
        return l ? planePerpendicularToLine(l.line, s.picks[0]) : null;
      }
    }
  }

  private preview() {
    this.overlay.clear();
    if (this.s.picks.length) this.overlay.add(pointSprites(this.s.picks.flat(), "#ffd23f", 9, { depthTest: false }));
    const p = this.plane();
    if (!p) return;
    const size = this.app.sceneSize() * 0.35;
    const m = new THREE.Mesh(new THREE.PlaneGeometry(size, size), translucentMaterial("#3d8bfd", 0.3));
    m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), new THREE.Vector3(...p.normal));
    m.position.set(...p.origin);
    this.overlay.add(m);
  }

  private neededPicks(): number {
    return this.s.mode === "points3" ? 3 : this.s.mode === "lineThroughPoint" || this.s.mode === "perpLine" ? 1 : 0;
  }

  onPointerDown(e: ToolPointerEvent) {
    const need = this.neededPicks();
    if (!need || e.button !== 0) return false;
    const hit = e.pick();
    if (hit) {
      const picks = this.s.picks.length >= need ? [hit.point] : [...this.s.picks, hit.point];
      this.set({ picks });
      this.preview();
    }
    return true;
  }

  renderPanel() {
    const s = this.s;
    const upd = (p: Partial<typeof s>) => {
      this.set(p);
      this.preview();
    };
    const pl = planesOf(this.app);
    return (
      <Section title="Definition">
        <Select
          value={s.mode}
          options={[
            { value: "offset", label: "Offset from plane" },
            { value: "world", label: "World plane" },
            { value: "mid", label: "Mid-plane of two planes" },
            { value: "points3", label: "Through 3 points" },
            { value: "lineThroughPoint", label: "Through axis and point" },
            { value: "perpLine", label: "Perpendicular to axis" },
          ]}
          onChange={(mode) => upd({ mode, picks: [] })}
        />
        <div style={{ height: 10 }} />
        {(s.mode === "offset" || s.mode === "mid") && (
          <Field label={s.mode === "mid" ? "Plane A" : "Plane"}>
            <Select value={s.a} options={pl} placeholder="Choose…" onChange={(a) => upd({ a })} />
          </Field>
        )}
        {s.mode === "mid" && (
          <Field label="Plane B">
            <Select value={s.b} options={pl} placeholder="Choose…" onChange={(b) => upd({ b })} />
          </Field>
        )}
        {s.mode === "world" && (
          <Field label="Plane">
            <Segmented value={s.world} options={[{ value: "XY", label: "XY" }, { value: "YZ", label: "YZ" }, { value: "ZX", label: "ZX" }]} onChange={(world) => upd({ world })} />
          </Field>
        )}
        {(s.mode === "offset" || s.mode === "world") && (
          <Field label="Offset">
            <NumberInput value={s.offset} unit="mm" onChange={(offset) => upd({ offset })} />
          </Field>
        )}
        {(s.mode === "lineThroughPoint" || s.mode === "perpLine") && (
          <Field label="Axis / line">
            <Select value={s.a} options={axesOf(this.app)} placeholder="Choose…" onChange={(a) => upd({ a })} />
          </Field>
        )}
        {this.neededPicks() > 0 && (
          <div className="note">
            Click {this.neededPicks() === 3 ? "three points" : "a point"} on the model ({s.picks.length}/{this.neededPicks()}).
            {s.mode === "perpLine" && " Optional — defaults to the axis origin."}
          </div>
        )}
      </Section>
    );
  }

  async commit() {
    const p = this.plane();
    if (!p) {
      this.app.toast("error", "The plane is not fully defined");
      return false;
    }
    const e = this.app.doc.add<PrimitiveEntity>({ id: newId("r"), kind: "primitive", name: this.app.doc.uniqueName("Plane"), visible: true, def: p, displaySize: this.app.sceneSize() * 0.2 });
    this.app.setActiveEntity(e.id);
  }

  deactivate() {
    this.overlay.dispose();
  }
}

// ---------------------------------------------------------------------------
// Symmetry plane with analyser
// ---------------------------------------------------------------------------

interface SymState {
  plane: PlaneDef | null;
  rms: number | null;
  tiltA: number;
  tiltB: number;
  shift: number;
  showMap: boolean;
}

class SymmetryTool extends Tool<SymState> {
  id = "symmetry";
  label = "Symmetry Plane";
  hint = "The symmetry plane is estimated automatically. The colour map shows how far the mirrored scan is from the original — fine-tune with the sliders.";
  okLabel = "Create plane";
  private overlay = new Overlay(this.app);
  private meshId: string;
  private base: PlaneDef | null = null;
  private sampleIdx: Uint32Array | null = null;
  private timer = 0;

  constructor(app: App) {
    super(app, { plane: null, rms: null, tiltA: 0, tiltB: 0, shift: 0, showMap: true });
    const m = app.activeMesh();
    if (!m) throw new Error("Import a mesh first");
    this.meshId = m.id;
  }

  async activate() {
    const mesh = this.app.doc.get<MeshEntity>(this.meshId)!.mesh;
    const rt = this.app.sync!.meshRuntime(this.meshId)!;
    const { estimateSymmetryPlane } = await import("../fit/symmetry");
    const res = await this.app.busy("Estimating symmetry plane", () =>
      estimateSymmetryPlane(mesh.positions, { normals: mesh.normals, query: rt.spatial.asQuery(), sampleCount: 20000 }),
    );
    if (!res) {
      void this.app.endTool(false);
      return;
    }
    this.base = res.plane;
    this.set({ plane: res.plane, rms: res.rms });
    this.update(true);
  }

  private current(): PlaneDef | null {
    const b = this.base;
    if (!b) return null;
    const s = this.s;
    const u = normalize(cross(b.normal, Math.abs(b.normal[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0]));
    const v = cross(b.normal, u);
    let n = transformDir(rotationAxisAngle(u, (s.tiltA * Math.PI) / 180), b.normal);
    n = normalize(transformDir(rotationAxisAngle(v, (s.tiltB * Math.PI) / 180), n));
    const o: Vec3 = [b.origin[0] + n[0] * s.shift, b.origin[1] + n[1] * s.shift, b.origin[2] + n[2] * s.shift];
    return { kind: "plane", origin: o, normal: n };
  }

  private update(full: boolean) {
    const p = this.current();
    this.overlay.clear();
    if (!p) return;
    const size = this.app.sceneSize() * 0.45;
    const m = new THREE.Mesh(new THREE.PlaneGeometry(size, size), translucentMaterial("#b46cff", 0.25));
    m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), new THREE.Vector3(...p.normal));
    m.position.set(...p.origin);
    this.overlay.add(m);
    window.clearTimeout(this.timer);
    this.timer = window.setTimeout(() => void this.colorMap(p, full), full ? 0 : 120);
  }

  private async colorMap(p: PlaneDef, _full: boolean) {
    const e = this.app.doc.get<MeshEntity>(this.meshId);
    const rt = this.app.sync?.meshRuntime(this.meshId);
    const mv = this.app.sync?.view<MeshView>(this.meshId);
    if (!e || !rt || !mv) return;
    const { symmetryDeviation } = await import("../fit/symmetry");
    const dev = symmetryDeviation(e.mesh.positions, p, rt.spatial.asQuery());
    let sum = 0, n = 0;
    for (let i = 0; i < dev.length; i++) if (Number.isFinite(dev[i])) {
      sum += dev[i] * dev[i];
      n++;
    }
    const rms = Math.sqrt(sum / Math.max(1, n));
    this.set({ plane: p, rms });
    if (this.s.showMap) {
      const range = robustRange(dev);
      mv.setColorMode("map", deviationColors(dev, range, range * 0.1));
      this.app.ui.patch({ legend: { title: "Symmetry deviation", min: -range, max: range, tolerance: range * 0.1, stats: [`RMS ${rms.toFixed(3)} mm`] } });
    }
    void this.sampleIdx;
  }

  renderPanel() {
    const s = this.s;
    const upd = (p: Partial<SymState>) => {
      this.set(p);
      this.update(false);
    };
    if (!s.plane) return <Section><div className="note">Estimating…</div></Section>;
    return (
      <>
        <Section title="Quality">
          <Stats rows={[["Mirror RMS", s.rms === null ? "—" : mm(s.rms)]]} />
        </Section>
        <Section title="Fine tune">
          <Field label="Tilt A">
            <Slider value={s.tiltA} min={-10} max={10} step={0.05} onChange={(tiltA) => upd({ tiltA })} format={(v) => `${v.toFixed(2)}°`} />
          </Field>
          <Field label="Tilt B">
            <Slider value={s.tiltB} min={-10} max={10} step={0.05} onChange={(tiltB) => upd({ tiltB })} format={(v) => `${v.toFixed(2)}°`} />
          </Field>
          <Field label="Shift">
            <NumberInput value={s.shift} unit="mm" step={0.1} onChange={(shift) => upd({ shift })} />
          </Field>
          <button className="btn small" onClick={() => upd({ tiltA: 0, tiltB: 0, shift: 0 })}>
            Reset
          </button>
        </Section>
      </>
    );
  }

  async commit() {
    const p = this.current();
    if (!p) return false;
    const e = this.app.doc.add<PrimitiveEntity>({
      id: newId("r"),
      kind: "primitive",
      name: this.app.doc.uniqueName("Symmetry plane"),
      visible: true,
      def: p,
      symmetry: true,
      meshId: this.meshId,
      displaySize: this.app.sceneSize() * 0.25,
      stats: this.s.rms !== null ? { rms: this.s.rms, maxAbs: NaN, mean: 0, count: 0 } : undefined,
    });
    this.app.setActiveEntity(e.id);
  }

  deactivate() {
    window.clearTimeout(this.timer);
    this.overlay.dispose();
    const mv = this.app.sync?.view<MeshView>(this.meshId);
    const e = this.app.doc.get<MeshEntity>(this.meshId);
    if (mv && e) {
      mv.setColorMode("solid", null);
      this.app.applyMeshDisplay(mv, e);
    }
    this.app.ui.patch({ legend: null });
  }
}

export function startReferenceTool(app: App, id: string): void {
  const map: Record<string, ToolClass> = {
    refLine: RefLineTool,
    refPoint: RefPointTool,
    refPlane: RefPlaneTool,
    symmetry: SymmetryTool,
  };
  const T = map[id];
  if (T) app.startTool(new T(app));
}
