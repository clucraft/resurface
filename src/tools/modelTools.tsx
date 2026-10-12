/**
 * CAD feature tools: extrude, revolve, loft, primitive surface, boolean, fillet/chamfer, mirror.
 * Each tool previews the result through the OCCT kernel and commits a BodyEntity holding the
 * feature definition (re-evaluated by FeatureEvaluator when inputs change).
 */
import * as THREE from "three";
import type { App } from "../app/App";
import { newId } from "../core/document";
import type { BodyEntity, FeatureDef, PrimitiveEntity, RefLineEntity, SketchEntity } from "../core/entities";
import type { Frame, Vec3 } from "../core/types";
import { frameToWorld, normalize } from "../core/vec";
import { Tool, type ToolClass, type ToolPointerEvent } from "./Tool";
import { Overlay } from "./overlay";
import { Section, Field, Select, NumberInput, Check, Segmented, Stats, mm } from "../ui/widgets";
import { fatSegments, lineMaterial } from "../viewer/draw";
import type { BodyTessellation, BodyView } from "../viewer/views";
import { lazy } from "../app/lazy";
import { getEntity, getPoint } from "../sketch/model";

// ---------------------------------------------------------------------------
// Preview helper
// ---------------------------------------------------------------------------

class KernelPreview {
  private overlay: Overlay;
  private shape: number | null = null;
  private token = 0;
  private timer = 0;
  private faceMat = new THREE.MeshStandardMaterial({ color: "#7fb2ff", roughness: 0.5, metalness: 0.1, transparent: true, opacity: 0.6, side: THREE.DoubleSide, depthWrite: false });
  private edgeMat = lineMaterial({ color: "#cfe3ff", width: 1.4 });
  error: string | null = null;

  constructor(private app: App, private onError: (msg: string | null) => void) {
    this.overlay = new Overlay(app);
  }

  /** Debounced: build a shape with `build`, tessellate and show it. */
  update(build: (k: import("../cad/KernelClient").KernelClient) => Promise<number>, delay = 120): void {
    window.clearTimeout(this.timer);
    this.timer = window.setTimeout(() => void this.run(build), delay);
  }

  private async run(build: (k: import("../cad/KernelClient").KernelClient) => Promise<number>) {
    const t = ++this.token;
    try {
      const k = await this.app.getKernel();
      const id = await build(k);
      if (t !== this.token) {
        void k.release(id);
        return;
      }
      const tess = await k.tessellate(id, { linearDeflection: 0.1, angularDeflectionDeg: 15 });
      if (t !== this.token) {
        void k.release(id);
        return;
      }
      this.releaseShape();
      this.shape = id;
      this.show(tess);
      this.error = null;
      this.onError(null);
    } catch (e) {
      if (t !== this.token) return;
      this.overlay.clear();
      this.error = e instanceof Error ? e.message : String(e);
      this.onError(this.error);
    }
  }

  private show(t: BodyTessellation) {
    this.overlay.clear();
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(t.positions, 3));
    g.setAttribute("normal", new THREE.BufferAttribute(t.normals, 3));
    g.setIndex(new THREE.BufferAttribute(t.indices, 1));
    const m = new THREE.Mesh(g, this.faceMat);
    m.renderOrder = 5;
    this.overlay.add(m);
    let n = 0;
    for (const e of t.edges) n += Math.max(0, e.points.length / 3 - 1);
    const pairs = new Float32Array(n * 6);
    let k = 0;
    for (const e of t.edges)
      for (let i = 0; i + 5 < e.points.length; i += 3) {
        pairs.set(e.points.subarray(i, i + 6), k);
        k += 6;
      }
    if (pairs.length) this.overlay.add(fatSegments(pairs, this.edgeMat));
  }

  private releaseShape() {
    if (this.shape !== null) {
      const id = this.shape;
      this.shape = null;
      void this.app.getKernel().then((k) => k.release(id));
    }
  }

  dispose() {
    this.token++;
    window.clearTimeout(this.timer);
    this.releaseShape();
    this.overlay.dispose();
    this.faceMat.dispose();
    this.edgeMat.dispose();
  }
}

const sketchOptions = (app: App) => app.doc.ofKind("sketch").map((s) => ({ value: s.id, label: s.name }));
const bodyOptions = (app: App) => app.doc.ofKind("body").map((b) => ({ value: b.id, label: b.name }));

function defaultSketch(app: App): string {
  const active = app.doc.get(app.ui.get().activeEntityId);
  if (active?.kind === "sketch") return active.id;
  const all = app.doc.ofKind("sketch");
  return all[all.length - 1]?.id ?? "";
}

function defaultBody(app: App, exclude?: string): string {
  const active = app.doc.get(app.ui.get().activeEntityId);
  if (active?.kind === "body" && active.id !== exclude) return active.id;
  return app.doc.ofKind("body").find((b) => b.id !== exclude && b.visible)?.id ?? "";
}

abstract class FeatureTool<S extends object> extends Tool<S> {
  protected preview: KernelPreview;
  /** when editing an existing body */
  protected editId: string | null = null;

  constructor(app: App, initial: S) {
    super(app, { ...initial, error: null } as S);
    this.preview = new KernelPreview(app, (error) => this.set({ error } as unknown as Partial<S>));
  }

  /** Kernel shape for the current parameters (or null when inputs are incomplete). */
  protected abstract feature(): FeatureDef | null;
  protected abstract baseName(): string;

  activate() {
    if (this.editId) this.app.sync?.view<BodyView>(this.editId)?.setVisible(false);
    this.refresh();
  }

  /** Recompute the preview. Uses a temporary BodyEntity-like evaluation through the kernel. */
  protected refresh() {
    const f = this.feature();
    if (!f) {
      this.preview.update(async () => {
        throw new Error("Incomplete input");
      }, 0);
      return;
    }
    this.preview.update((k) => buildFeature(this.app, k, f));
  }

  protected upd(p: Partial<S>) {
    this.set(p);
    this.refresh();
  }

  protected errorNote() {
    const err = (this.s as { error?: string | null }).error;
    return err && err !== "Incomplete input" ? <div className="note err" style={{ marginTop: 8 }}>{err}</div> : null;
  }

  async commit() {
    const f = this.feature();
    if (!f) {
      this.app.toast("error", "Complete the inputs first");
      return false;
    }
    if (this.editId) {
      this.app.doc.transact(`Edit ${this.baseName()}`, () => this.app.doc.update<BodyEntity>(this.editId!, { feature: f }));
      return;
    }
    const consumes = f.type === "boolean" ? [f.targetId, f.toolId] : f.type === "fillet" || f.type === "chamfer" ? [f.baseId] : undefined;
    const e = this.app.doc.transact(this.baseName(), () => {
      for (const id of consumes ?? []) if (this.app.doc.get(id)?.visible) this.app.doc.update(id, { visible: false });
      return this.app.doc.add<BodyEntity>({
        id: newId("b"),
        kind: "body",
        name: this.app.doc.uniqueName(this.baseName()),
        visible: true,
        feature: f,
        consumes,
      });
    });
    this.app.setActiveEntity(e.id);
  }

  deactivate() {
    if (this.editId) this.app.sync?.view<BodyView>(this.editId)?.setVisible(this.app.doc.get(this.editId)?.visible ?? true);
    this.preview.dispose();
  }
}

/** Build a kernel shape for a feature definition without a document entity (preview). */
async function buildFeature(app: App, k: import("../cad/KernelClient").KernelClient, f: FeatureDef): Promise<number> {
  const { extractProfiles } = await lazy<typeof import("../sketch/profiles")>("../sketch/profiles.ts");
  const sk = (id: string) => {
    const s = app.doc.get<SketchEntity>(id);
    if (!s) throw new Error("Sketch not found");
    return s;
  };
  switch (f.type) {
    case "extrude": {
      const s = sk(f.sketchId);
      const ex = extractProfiles(s.sketch.entities.length ? s.sketch : s.sketch);
      if (ex.profile.loops.length)
        return k.extrude({ frame: s.frame, profile: ex.profile, distance: f.distance, distance2: f.distance2, solid: f.solid, draftDeg: f.draftDeg });
      if (ex.openWires.length) return k.extrudeWire({ frame: s.frame, wire: ex.openWires[0], distance: f.distance, distance2: f.distance2 });
      throw new Error("The sketch has no closed profile or curve to extrude");
    }
    case "revolve": {
      const s = sk(f.sketchId);
      const ex = extractProfiles(s.sketch);
      if (ex.profile.loops.length) return k.revolve({ frame: s.frame, profile: ex.profile, axisOrigin: f.axisOrigin, axisDir: f.axisDir, angleDeg: f.angleDeg, solid: f.solid });
      if (ex.openWires.length) return k.revolve({ frame: s.frame, wire: ex.openWires[0], axisOrigin: f.axisOrigin, axisDir: f.axisDir, angleDeg: f.angleDeg, solid: false });
      throw new Error("The sketch has no geometry to revolve");
    }
    case "loft":
      return k.loft({ sections: f.sketchIds.map((id) => ({ frame: sk(id).frame, profile: extractProfiles(sk(id).sketch).profile })), solid: f.solid });
    case "primitive":
      return k.primitiveFace(f.def, { size: f.size });
    case "boolean": {
      const a = await app.features.ensureShape(f.targetId);
      const b = await app.features.ensureShape(f.toolId);
      return k.boolean(f.op, a, b, { simplify: true });
    }
    case "fillet":
      return k.fillet(await app.features.ensureShape(f.baseId), f.edges, f.radius);
    case "chamfer":
      return k.chamfer(await app.features.ensureShape(f.baseId), f.edges, f.distance);
    case "mirror":
      return k.mirror(await app.features.ensureShape(f.baseId), { origin: f.planeOrigin, normal: f.planeNormal });
    default:
      throw new Error(`Preview not supported for ${f.type}`);
  }
}

// ---------------------------------------------------------------------------
// Extrude
// ---------------------------------------------------------------------------

interface ExtrudeState {
  sketchId: string;
  distance: number;
  distance2: number;
  mode: "one" | "symmetric" | "two";
  solid: boolean;
  draftDeg: number;
  error: string | null;
}

class ExtrudeTool extends FeatureTool<ExtrudeState> {
  id = "extrude";
  label = "Extrude";
  hint = "Extrudes the closed loops of a sketch (holes are subtracted automatically). Open curves become surfaces.";
  private arrow = new Overlay(this.app);

  constructor(app: App, edit?: BodyEntity) {
    const f = edit?.feature.type === "extrude" ? edit.feature : null;
    super(app, {
      sketchId: f?.sketchId ?? defaultSketch(app),
      distance: f?.distance ?? Math.round(app.sceneSize() * 0.1),
      distance2: f?.distance2 ?? 0,
      mode: f ? (f.distance2 === 0 ? "one" : f.distance2 === f.distance ? "symmetric" : "two") : "one",
      solid: f?.solid ?? true,
      draftDeg: f?.draftDeg ?? 0,
      error: null,
    });
    if (edit) this.editId = edit.id;
    if (!this.s.sketchId) throw new Error("Create a sketch first");
  }

  baseName() {
    return "Extrude";
  }

  feature(): FeatureDef | null {
    const s = this.s;
    if (!s.sketchId) return null;
    const d2 = s.mode === "one" ? 0 : s.mode === "symmetric" ? s.distance : s.distance2;
    return { type: "extrude", sketchId: s.sketchId, distance: s.distance, distance2: d2, solid: s.solid, draftDeg: s.draftDeg || undefined };
  }

  protected refresh() {
    super.refresh();
    this.arrow.clear();
    const sk = this.app.doc.get<SketchEntity>(this.s.sketchId);
    if (!sk) return;
    const f = sk.frame;
    const L = this.s.distance;
    const a = new THREE.ArrowHelper(new THREE.Vector3(...f.zAxis).multiplyScalar(Math.sign(L) || 1), new THREE.Vector3(...f.origin), Math.abs(L) || 1, 0x2dd4bf, Math.min(6, Math.abs(L) * 0.2), Math.min(3, Math.abs(L) * 0.1));
    this.arrow.add(a);
  }

  renderPanel() {
    const s = this.s;
    return (
      <>
        <Section title="Profile">
          <Field label="Sketch">
            <Select value={s.sketchId} options={sketchOptions(this.app)} onChange={(sketchId) => this.upd({ sketchId })} />
          </Field>
          <Check checked={s.solid} onChange={(solid) => this.upd({ solid })}>
            Solid (unchecked: surface)
          </Check>
        </Section>
        <Section title="Extent">
          <Field label="Direction">
            <Segmented
              value={s.mode}
              options={[{ value: "one", label: "One side" }, { value: "symmetric", label: "Symmetric" }, { value: "two", label: "Two sides" }]}
              onChange={(mode) => this.upd({ mode })}
            />
          </Field>
          <Field label={s.mode === "symmetric" ? "Each side" : "Distance"}>
            <NumberInput value={s.distance} unit="mm" step={1} onChange={(distance) => this.upd({ distance })} />
          </Field>
          {s.mode === "two" && (
            <Field label="Opposite side">
              <NumberInput value={s.distance2} unit="mm" step={1} onChange={(distance2) => this.upd({ distance2 })} />
            </Field>
          )}
          <Field label="Draft angle">
            <NumberInput value={s.draftDeg} unit="°" step={0.5} min={-45} max={45} onChange={(draftDeg) => this.upd({ draftDeg })} />
          </Field>
          <button className="btn small" onClick={() => this.upd({ distance: -s.distance })}>
            Flip direction
          </button>
          {this.errorNote()}
        </Section>
      </>
    );
  }

  deactivate() {
    super.deactivate();
    this.arrow.dispose();
  }
}

// ---------------------------------------------------------------------------
// Revolve
// ---------------------------------------------------------------------------

interface RevolveState {
  sketchId: string;
  axis: string;
  angleDeg: number;
  solid: boolean;
  error: string | null;
}

function axisOptions(app: App, sketchId: string): { value: string; label: string }[] {
  const out = [
    { value: "sketchY", label: "Sketch Y axis" },
    { value: "sketchX", label: "Sketch X axis" },
  ];
  const sk = app.doc.get<SketchEntity>(sketchId);
  for (const e of sk?.sketch.entities ?? []) if (e.type === "line") out.push({ value: `line:${e.id}`, label: `Sketch line ${e.id}${e.construction ? " (construction)" : ""}` });
  for (const p of app.doc.all()) {
    if (p.kind === "refline") out.push({ value: `ref:${p.id}`, label: p.name });
    if (p.kind === "primitive" && (p.def.kind === "cylinder" || p.def.kind === "cone")) out.push({ value: `ref:${p.id}`, label: `${p.name} axis` });
  }
  return out;
}

function resolveAxis(app: App, sketchId: string, key: string): { origin: Vec3; dir: Vec3 } | null {
  const sk = app.doc.get<SketchEntity>(sketchId);
  if (!sk) return null;
  const f: Frame = sk.frame;
  if (key === "sketchY") return { origin: f.origin, dir: f.yAxis };
  if (key === "sketchX") return { origin: f.origin, dir: f.xAxis };
  if (key.startsWith("line:")) {
    const e = getEntity(sk.sketch, key.slice(5));
    if (e?.type !== "line") return null;
    const a = getPoint(sk.sketch, e.p1)!, b = getPoint(sk.sketch, e.p2)!;
    const pa = frameToWorld(f, [a.x, a.y]), pb = frameToWorld(f, [b.x, b.y]);
    return { origin: pa, dir: normalize([pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]]) };
  }
  if (key.startsWith("ref:")) {
    const e = app.doc.get(key.slice(4));
    if (e?.kind === "refline") return { origin: (e as RefLineEntity).def.origin, dir: (e as RefLineEntity).def.direction };
    if (e?.kind === "primitive") {
      const d = (e as PrimitiveEntity).def;
      if (d.kind === "cylinder") return { origin: d.origin, dir: d.axis };
      if (d.kind === "cone") return { origin: d.apex, dir: d.axis };
    }
  }
  return null;
}

class RevolveTool extends FeatureTool<RevolveState> {
  id = "revolve";
  label = "Revolve";
  hint = "Revolves a sketch profile around an axis. For turned parts, sketch on a section through the axis and use the sketch Y axis.";
  private axisOverlay = new Overlay(this.app);
  private axisMat = lineMaterial({ color: "#00c2a8", width: 2.5, dashed: true, dashSize: 4, depthTest: false });

  constructor(app: App, edit?: BodyEntity) {
    const f = edit?.feature.type === "revolve" ? edit.feature : null;
    super(app, { sketchId: f?.sketchId ?? defaultSketch(app), axis: f?.axisRef ?? "sketchY", angleDeg: f?.angleDeg ?? 360, solid: f?.solid ?? true, error: null });
    if (edit) this.editId = edit.id;
    if (!this.s.sketchId) throw new Error("Create a sketch first");
  }

  baseName() {
    return "Revolve";
  }

  feature(): FeatureDef | null {
    const ax = resolveAxis(this.app, this.s.sketchId, this.s.axis);
    if (!ax) return null;
    return { type: "revolve", sketchId: this.s.sketchId, axisOrigin: ax.origin, axisDir: ax.dir, axisRef: this.s.axis, angleDeg: this.s.angleDeg, solid: this.s.solid };
  }

  protected refresh() {
    super.refresh();
    this.axisOverlay.clear();
    const ax = resolveAxis(this.app, this.s.sketchId, this.s.axis);
    if (!ax) return;
    const L = this.app.sceneSize() * 0.6;
    const a = ax.origin.map((v, i) => v - ax.dir[i] * L);
    const b = ax.origin.map((v, i) => v + ax.dir[i] * L);
    this.axisOverlay.add(fatSegments([...a, ...b], this.axisMat)).renderOrder = 30;
  }

  renderPanel() {
    const s = this.s;
    return (
      <Section title="Revolve">
        <Field label="Sketch">
          <Select value={s.sketchId} options={sketchOptions(this.app)} onChange={(sketchId) => this.upd({ sketchId })} />
        </Field>
        <Field label="Axis">
          <Select value={s.axis} options={axisOptions(this.app, s.sketchId)} onChange={(axis) => this.upd({ axis })} />
        </Field>
        <Field label="Angle">
          <NumberInput value={s.angleDeg} unit="°" step={15} min={-360} max={360} onChange={(angleDeg) => this.upd({ angleDeg })} />
        </Field>
        <Check checked={s.solid} onChange={(solid) => this.upd({ solid })}>
          Solid (unchecked: surface)
        </Check>
        {this.errorNote()}
      </Section>
    );
  }

  deactivate() {
    super.deactivate();
    this.axisOverlay.dispose();
    this.axisMat.dispose();
  }
}

// ---------------------------------------------------------------------------
// Loft
// ---------------------------------------------------------------------------

class LoftTool extends FeatureTool<{ order: string[]; solid: boolean; error: string | null }> {
  id = "loft";
  label = "Loft";
  hint = "Lofts through the closed profiles of several sketches, in the listed order. Use Sketch → Sections to create parallel sections, then sketch a closed profile on each.";
  constructor(app: App, edit?: BodyEntity) {
    const f = edit?.feature.type === "loft" ? edit.feature : null;
    super(app, { order: f?.sketchIds ?? [], solid: f?.solid ?? true, error: null });
    if (edit) this.editId = edit.id;
  }
  baseName() {
    return "Loft";
  }
  feature(): FeatureDef | null {
    return this.s.order.length >= 2 ? { type: "loft", sketchIds: this.s.order, solid: this.s.solid } : null;
  }
  renderPanel() {
    const s = this.s;
    const all = this.app.doc.ofKind("sketch");
    const toggle = (id: string) => this.upd({ order: s.order.includes(id) ? s.order.filter((x) => x !== id) : [...s.order, id] });
    return (
      <Section title="Sections (in order)">
        <div className="list">
          {all.map((sk) => {
            const i = s.order.indexOf(sk.id);
            return (
              <div key={sk.id} className={`list-row${i >= 0 ? " on" : ""}`} onClick={() => toggle(sk.id)}>
                <input type="checkbox" readOnly checked={i >= 0} />
                {sk.name}
                <span className="mono">{i >= 0 ? `#${i + 1}` : ""}</span>
              </div>
            );
          })}
        </div>
        <div style={{ height: 8 }} />
        <Check checked={s.solid} onChange={(solid) => this.upd({ solid })}>
          Solid
        </Check>
        {this.errorNote()}
      </Section>
    );
  }
}

// ---------------------------------------------------------------------------
// Primitive → CAD surface
// ---------------------------------------------------------------------------

class PrimitiveBodyTool extends FeatureTool<{ primId: string; size: number; error: string | null }> {
  id = "primitiveBody";
  label = "Surface from Primitive";
  hint = "Turns a fitted plane, cylinder, cone or sphere into a CAD surface that can be trimmed or used in booleans.";
  constructor(app: App) {
    const prims = app.doc.ofKind("primitive");
    if (!prims.length) throw new Error("Fit a primitive first (Reference tab)");
    const active = app.doc.get(app.ui.get().activeEntityId);
    super(app, { primId: active?.kind === "primitive" ? active.id : prims[0].id, size: Math.round(app.sceneSize() * 0.3), error: null });
  }
  baseName() {
    return this.app.doc.get(this.s.primId)?.name ?? "Surface";
  }
  feature(): FeatureDef | null {
    const p = this.app.doc.get<PrimitiveEntity>(this.s.primId);
    if (!p) return null;
    return { type: "primitive", def: p.def, size: this.s.size };
  }
  renderPanel() {
    const s = this.s;
    const def = this.app.doc.get<PrimitiveEntity>(s.primId)?.def;
    return (
      <Section title="Primitive">
        <Field label="Primitive">
          <Select value={s.primId} options={this.app.doc.ofKind("primitive").map((p) => ({ value: p.id, label: p.name }))} onChange={(primId) => this.upd({ primId })} />
        </Field>
        {def?.kind === "plane" && (
          <Field label="Size">
            <NumberInput value={s.size} unit="mm" min={1} onChange={(size) => this.upd({ size })} />
          </Field>
        )}
        <div className="note">Cylinders and cones use the extent of the fitted data along their axis.</div>
        {this.errorNote()}
      </Section>
    );
  }
}

// ---------------------------------------------------------------------------
// Boolean
// ---------------------------------------------------------------------------

class BooleanTool extends FeatureTool<{ targetId: string; toolId: string; op: "union" | "cut" | "intersect"; error: string | null }> {
  id = "boolean";
  label = "Boolean";
  hint = "Combine two bodies. The input bodies are hidden and kept in the history.";
  constructor(app: App, edit?: BodyEntity) {
    const f = edit?.feature.type === "boolean" ? edit.feature : null;
    const t = f?.targetId ?? defaultBody(app);
    super(app, { targetId: t, toolId: f?.toolId ?? defaultBody(app, t), op: f?.op ?? "cut", error: null });
    if (edit) this.editId = edit.id;
  }
  baseName() {
    return this.s.op === "union" ? "Union" : this.s.op === "cut" ? "Cut" : "Intersect";
  }
  feature(): FeatureDef | null {
    const s = this.s;
    return s.targetId && s.toolId && s.targetId !== s.toolId ? { type: "boolean", op: s.op, targetId: s.targetId, toolId: s.toolId } : null;
  }
  renderPanel() {
    const s = this.s;
    const bodies = bodyOptions(this.app).filter((b) => b.value !== this.editId);
    return (
      <Section title="Boolean">
        <Field label="Operation">
          <Segmented value={s.op} options={[{ value: "union", label: "Unite" }, { value: "cut", label: "Subtract" }, { value: "intersect", label: "Intersect" }]} onChange={(op) => this.upd({ op })} />
        </Field>
        <Field label="Target body">
          <Select value={s.targetId} options={bodies} onChange={(targetId) => this.upd({ targetId })} />
        </Field>
        <Field label="Tool body">
          <Select value={s.toolId} options={bodies.filter((b) => b.value !== s.targetId)} placeholder="Choose…" onChange={(toolId) => this.upd({ toolId })} />
        </Field>
        {this.errorNote()}
      </Section>
    );
  }
}

// ---------------------------------------------------------------------------
// Fillet / chamfer (edge picking)
// ---------------------------------------------------------------------------

class FilletTool extends FeatureTool<{ baseId: string; edges: number[]; radius: number; kind: "fillet" | "chamfer"; error: string | null }> {
  id = "fillet";
  label = "Fillet / Chamfer";
  hint = "Click the edges to round (click again to deselect). Adjust the radius and watch the preview.";
  private edgeOverlay = new Overlay(this.app);
  private selMat = lineMaterial({ color: "#ffb020", width: 4, depthTest: false });

  constructor(app: App, edit?: BodyEntity) {
    const f = edit?.feature.type === "fillet" || edit?.feature.type === "chamfer" ? edit.feature : null;
    super(app, {
      baseId: f?.baseId ?? defaultBody(app),
      edges: f?.edges ?? [],
      radius: f ? (f.type === "fillet" ? f.radius : f.distance) : 2,
      kind: f?.type === "chamfer" ? "chamfer" : "fillet",
      error: null,
    });
    if (edit) this.editId = edit.id;
    if (!this.s.baseId) throw new Error("Create a body first");
  }
  baseName() {
    return this.s.kind === "fillet" ? "Fillet" : "Chamfer";
  }
  feature(): FeatureDef | null {
    const s = this.s;
    if (!s.baseId || !s.edges.length) return null;
    return s.kind === "fillet" ? { type: "fillet", baseId: s.baseId, edges: s.edges, radius: s.radius } : { type: "chamfer", baseId: s.baseId, edges: s.edges, distance: s.radius };
  }

  private tess(): BodyTessellation | null {
    return this.app.sync?.view<BodyView>(this.s.baseId)?.tess ?? null;
  }

  onPointerDown(e: ToolPointerEvent): boolean {
    if (e.button !== 0) return false;
    const t = this.tess();
    if (!t) return true;
    const viewer = this.app.viewer!;
    // nearest edge polyline in screen space
    let best = -1, bestD = 10;
    for (const ed of t.edges) {
      const p = ed.points;
      for (let i = 0; i + 5 < p.length; i += 3) {
        const a = viewer.worldToScreen([p[i], p[i + 1], p[i + 2]]);
        const b = viewer.worldToScreen([p[i + 3], p[i + 4], p[i + 5]]);
        const dx = b.x - a.x, dy = b.y - a.y;
        const L2 = dx * dx + dy * dy || 1;
        const u = Math.max(0, Math.min(1, ((e.x - a.x) * dx + (e.y - a.y) * dy) / L2));
        const d = Math.hypot(a.x + u * dx - e.x, a.y + u * dy - e.y);
        if (d < bestD) {
          bestD = d;
          best = ed.id;
        }
      }
    }
    if (best >= 0) {
      const edges = this.s.edges.includes(best) ? this.s.edges.filter((x) => x !== best) : [...this.s.edges, best];
      this.upd({ edges });
      this.drawEdges();
    }
    return true;
  }

  private drawEdges() {
    this.edgeOverlay.clear();
    const t = this.tess();
    if (!t) return;
    const sel = new Set(this.s.edges);
    const pts: number[] = [];
    for (const ed of t.edges) {
      if (!sel.has(ed.id)) continue;
      const p = ed.points;
      for (let i = 0; i + 5 < p.length; i += 3) pts.push(p[i], p[i + 1], p[i + 2], p[i + 3], p[i + 4], p[i + 5]);
    }
    if (pts.length) this.edgeOverlay.add(fatSegments(pts, this.selMat)).renderOrder = 40;
  }

  activate() {
    super.activate();
    this.drawEdges();
  }

  renderPanel() {
    const s = this.s;
    return (
      <Section title="Edges">
        <Field label="Body">
          <Select value={s.baseId} options={bodyOptions(this.app).filter((b) => b.value !== this.editId)} onChange={(baseId) => { this.upd({ baseId, edges: [] }); this.drawEdges(); }} />
        </Field>
        <Field label="Type">
          <Segmented value={s.kind} options={[{ value: "fillet", label: "Fillet" }, { value: "chamfer", label: "Chamfer" }]} onChange={(kind) => this.upd({ kind })} />
        </Field>
        <Field label={s.kind === "fillet" ? "Radius" : "Distance"}>
          <NumberInput value={s.radius} unit="mm" min={0.001} step={0.25} onChange={(radius) => this.upd({ radius })} />
        </Field>
        <Stats rows={[["Selected edges", String(s.edges.length)]]} />
        {this.errorNote()}
      </Section>
    );
  }

  deactivate() {
    super.deactivate();
    this.edgeOverlay.dispose();
    this.selMat.dispose();
  }
}

// ---------------------------------------------------------------------------
// Mirror
// ---------------------------------------------------------------------------

class MirrorTool extends FeatureTool<{ baseId: string; plane: string; error: string | null }> {
  id = "mirror";
  label = "Mirror";
  hint = "Mirror a body about a reference plane (e.g. the symmetry plane) or a world plane.";
  constructor(app: App, edit?: BodyEntity) {
    const planes = app.doc.ofKind("primitive").filter((p) => p.def.kind === "plane");
    super(app, { baseId: defaultBody(app), plane: planes.find((p) => p.symmetry)?.id ?? planes[0]?.id ?? "world:YZ", error: null });
    if (edit) this.editId = edit.id;
  }
  baseName() {
    return "Mirror";
  }
  private planeDef(): { origin: Vec3; normal: Vec3 } | null {
    const p = this.s.plane;
    if (p === "world:YZ") return { origin: [0, 0, 0], normal: [1, 0, 0] };
    if (p === "world:ZX") return { origin: [0, 0, 0], normal: [0, 1, 0] };
    if (p === "world:XY") return { origin: [0, 0, 0], normal: [0, 0, 1] };
    const e = this.app.doc.get<PrimitiveEntity>(p);
    return e?.def.kind === "plane" ? { origin: e.def.origin, normal: e.def.normal } : null;
  }
  feature(): FeatureDef | null {
    const pd = this.planeDef();
    return pd && this.s.baseId ? { type: "mirror", baseId: this.s.baseId, planeOrigin: pd.origin, planeNormal: pd.normal } : null;
  }
  renderPanel() {
    const s = this.s;
    const planes = [
      ...this.app.doc.ofKind("primitive").filter((p) => p.def.kind === "plane").map((p) => ({ value: p.id, label: p.name })),
      { value: "world:YZ", label: "World YZ" },
      { value: "world:ZX", label: "World ZX" },
      { value: "world:XY", label: "World XY" },
    ];
    return (
      <Section title="Mirror">
        <Field label="Body">
          <Select value={s.baseId} options={bodyOptions(this.app)} onChange={(baseId) => this.upd({ baseId })} />
        </Field>
        <Field label="Plane">
          <Select value={s.plane} options={planes} onChange={(plane) => this.upd({ plane })} />
        </Field>
        {this.errorNote()}
      </Section>
    );
  }
}

export function startModelTool(app: App, id: string): void {
  const map: Record<string, ToolClass> = {
    extrude: ExtrudeTool,
    revolve: RevolveTool,
    loft: LoftTool,
    primitiveBody: PrimitiveBodyTool,
    boolean: BooleanTool,
    fillet: FilletTool,
    mirror: MirrorTool,
  };
  const T = map[id];
  if (T) app.startTool(new T(app));
}

export async function editBody(app: App, id: string): Promise<void> {
  const b = app.doc.get<BodyEntity>(id);
  if (!b) return;
  const f = b.feature;
  switch (f.type) {
    case "extrude":
      return app.startTool(new ExtrudeTool(app, b));
    case "revolve":
      return app.startTool(new RevolveTool(app, b));
    case "loft":
      return app.startTool(new LoftTool(app, b));
    case "boolean":
      return app.startTool(new BooleanTool(app, b));
    case "fillet":
    case "chamfer":
      return app.startTool(new FilletTool(app, b));
    case "mirror":
      return app.startTool(new MirrorTool(app, b));
    case "fitSurface": {
      const m = await import("./fitSurfaceTool");
      return m.startFitSurface(app, id);
    }
    case "freeform": {
      const m = await import("./freeformTools");
      return m.startFreeformTool(app, "quadSurface", id);
    }
    default:
      app.toast("info", "This feature has no editable parameters");
  }
}

void mm;
