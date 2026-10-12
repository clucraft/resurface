/** Alignment tools: object align to world, numeric move, N-point and fine (ICP) registration. */
import * as THREE from "three";
import type { App } from "../app/App";
import type { MeshEntity, PrimitiveEntity, RefLineEntity, RefPointEntity } from "../core/entities";
import type { Mat4, Vec3 } from "../core/types";
import { invertRigid, mul4, rotationAxisAngle, transformDir, transformPoint, translation4 } from "../core/vec";
import { alignToWorld, icp, nPointAlign, type WorldAxis } from "../fit/align";
import { Tool, type ToolClass, type ToolPointerEvent } from "./Tool";
import { Overlay } from "./overlay";
import { Section, Field, Select, Check, NumberInput, Stats, Segmented, mm, int } from "../ui/widgets";
import { pointSprites } from "../viewer/draw";

type DirSource = { kind: "plane" | "axis"; direction: Vec3; point: Vec3 };

function directionSources(app: App): { value: string; label: string }[] {
  const out: { value: string; label: string }[] = [];
  for (const e of app.doc.all()) {
    if (e.kind === "primitive" && e.def.kind !== "sphere") out.push({ value: e.id, label: `${e.name} (${e.def.kind === "plane" ? "normal" : "axis"})` });
    if (e.kind === "refline") out.push({ value: e.id, label: `${e.name} (line)` });
  }
  return out;
}

function pointSources(app: App): { value: string; label: string }[] {
  const out: { value: string; label: string }[] = [{ value: "", label: "Automatic" }];
  for (const e of app.doc.all()) {
    if (e.kind === "refpoint") out.push({ value: e.id, label: e.name });
    if (e.kind === "primitive" && e.def.kind === "sphere") out.push({ value: e.id, label: `${e.name} (centre)` });
    if (e.kind === "primitive" && e.def.kind === "cone") out.push({ value: e.id, label: `${e.name} (apex)` });
  }
  return out;
}

function dirOf(app: App, id: string): DirSource | null {
  const e = app.doc.get(id);
  if (!e) return null;
  if (e.kind === "primitive") {
    const d = (e as PrimitiveEntity).def;
    if (d.kind === "plane") return { kind: "plane", direction: d.normal, point: d.origin };
    if (d.kind === "cylinder") return { kind: "axis", direction: d.axis, point: d.origin };
    if (d.kind === "cone") return { kind: "axis", direction: d.axis, point: d.apex };
  }
  if (e.kind === "refline") return { kind: "axis", direction: (e as RefLineEntity).def.direction, point: (e as RefLineEntity).def.origin };
  return null;
}

function pointOf(app: App, id: string): Vec3 | undefined {
  const e = app.doc.get(id);
  if (!e) return undefined;
  if (e.kind === "refpoint") return (e as RefPointEntity).def.position;
  if (e.kind === "primitive") {
    const d = (e as PrimitiveEntity).def;
    if (d.kind === "sphere") return d.center;
    if (d.kind === "cone") return d.apex;
  }
  return undefined;
}

/** All entities that move together when the object is aligned (everything except other scans). */
function modelEntityIds(app: App): string[] {
  return app.doc.all().map((e) => e.id);
}

function drawAxes(overlay: Overlay, m: Mat4 | null, size: number) {
  overlay.clear();
  if (!m) return;
  // world axes as seen in the current (unaligned) coordinates
  const inv = invertRigid(m);
  const o = transformPoint(inv, [0, 0, 0]);
  const axes: [Vec3, number][] = [
    [[1, 0, 0], 0xf05454],
    [[0, 1, 0], 0x5bd16b],
    [[0, 0, 1], 0x4f8ff7],
  ];
  for (const [a, color] of axes) {
    const d = transformDir(inv, a);
    const arrow = new THREE.ArrowHelper(new THREE.Vector3(...d), new THREE.Vector3(...o), size, color, size * 0.12, size * 0.06);
    arrow.traverse((c) => {
      const mat = (c as THREE.Mesh).material as THREE.Material | undefined;
      if (mat) {
        mat.depthTest = false;
        mat.transparent = true;
      }
      c.renderOrder = 30;
    });
    overlay.add(arrow);
  }
  overlay.label("New origin", o);
}

// ---------------------------------------------------------------------------
// Object align
// ---------------------------------------------------------------------------

interface ObjectAlignState {
  primary: string;
  primaryTarget: WorldAxis;
  flipPrimary: boolean;
  secondary: string;
  secondaryTarget: WorldAxis;
  flipSecondary: boolean;
  origin: string;
  error: string | null;
}

class ObjectAlignTool extends Tool<ObjectAlignState> {
  id = "objectAlign";
  label = "Object Align";
  hint = "Choose the primitives that define the part's coordinate system. Typically: a base plane → Z, a side plane or axis → X, and a point/sphere/axis intersection as origin.";
  okLabel = "Align";
  private overlay = new Overlay(this.app);

  constructor(app: App) {
    const dirs = directionSources(app);
    if (!dirs.length) throw new Error("Fit at least one plane, cylinder or line first (Reference tab)");
    super(app, {
      primary: dirs[0].value,
      primaryTarget: "Z",
      flipPrimary: false,
      secondary: dirs[1]?.value ?? "",
      secondaryTarget: "X",
      flipSecondary: false,
      origin: "",
      error: null,
    });
  }

  activate() {
    this.preview();
  }

  matrix(): Mat4 | null {
    const s = this.s;
    const p = dirOf(this.app, s.primary);
    if (!p) return null;
    const sec = s.secondary ? dirOf(this.app, s.secondary) : null;
    try {
      const m = alignToWorld({
        primary: p,
        secondary: sec ?? undefined,
        origin: s.origin ? pointOf(this.app, s.origin) : undefined,
        primaryTarget: s.primaryTarget,
        secondaryTarget: s.secondaryTarget,
        flipPrimary: s.flipPrimary,
        flipSecondary: s.flipSecondary,
      });
      if (this.s.error) this.set({ error: null });
      return m;
    } catch (e) {
      this.set({ error: e instanceof Error ? e.message : String(e) });
      return null;
    }
  }

  private preview() {
    drawAxes(this.overlay, this.matrix(), this.app.sceneSize() * 0.25);
  }

  private upd(p: Partial<ObjectAlignState>) {
    this.set(p);
    this.preview();
  }

  renderPanel() {
    const s = this.s;
    const dirs = directionSources(this.app);
    const axisOpts = (["X", "Y", "Z"] as WorldAxis[]).map((a) => ({ value: a, label: a }));
    return (
      <>
        <Section title="Primary direction">
          <Field label="Entity">
            <Select value={s.primary} options={dirs} onChange={(primary) => this.upd({ primary })} />
          </Field>
          <Field label="Maps to">
            <Segmented value={s.primaryTarget} options={axisOpts} onChange={(primaryTarget) => this.upd({ primaryTarget, secondaryTarget: primaryTarget === s.secondaryTarget ? (primaryTarget === "X" ? "Y" : "X") : s.secondaryTarget })} />
          </Field>
          <Check checked={s.flipPrimary} onChange={(flipPrimary) => this.upd({ flipPrimary })}>
            Reverse direction
          </Check>
        </Section>
        <Section title="Secondary direction">
          <Field label="Entity">
            <Select value={s.secondary} options={[{ value: "", label: "None" }, ...dirs.filter((d) => d.value !== s.primary)]} onChange={(secondary) => this.upd({ secondary })} />
          </Field>
          <Field label="Maps to">
            <Segmented value={s.secondaryTarget} options={axisOpts.filter((a) => a.value !== s.primaryTarget)} onChange={(secondaryTarget) => this.upd({ secondaryTarget })} />
          </Field>
          <Check checked={s.flipSecondary} onChange={(flipSecondary) => this.upd({ flipSecondary })}>
            Reverse direction
          </Check>
        </Section>
        <Section title="Origin">
          <Field label="Point">
            <Select value={s.origin} options={pointSources(this.app)} onChange={(origin) => this.upd({ origin })} />
          </Field>
          <div className="note">Automatic uses the intersection of the primary plane with the secondary axis, or a point on the primary entity.</div>
          {s.error && <div className="note err" style={{ marginTop: 6 }}>{s.error}</div>}
        </Section>
      </>
    );
  }

  async commit() {
    const m = this.matrix();
    if (!m) return false;
    this.app.transformEntities(modelEntityIds(this.app), m, "Object align");
    this.app.viewer?.fitAll("iso");
    this.app.toast("success", "Scan aligned to the world coordinate system");
  }

  deactivate() {
    this.overlay.dispose();
  }
}

// ---------------------------------------------------------------------------
// Numeric move / rotate
// ---------------------------------------------------------------------------

interface TransformState {
  tx: number;
  ty: number;
  tz: number;
  rx: number;
  ry: number;
  rz: number;
  scope: "mesh" | "all";
  pivot: "origin" | "center";
}

class TransformTool extends Tool<TransformState> {
  id = "transform";
  label = "Move / Rotate";
  hint = "Enter a translation and rotation. Rotations are applied X, then Y, then Z about the pivot.";
  okLabel = "Apply";
  private meshId: string;

  constructor(app: App) {
    super(app, { tx: 0, ty: 0, tz: 0, rx: 0, ry: 0, rz: 0, scope: "all", pivot: "center" });
    const m = app.activeMesh();
    if (!m) throw new Error("Import a mesh first");
    this.meshId = m.id;
  }

  private pivotPoint(): Vec3 {
    if (this.s.pivot === "origin") return [0, 0, 0];
    const box = this.app.viewer!.sceneBounds(false);
    const c = box.getCenter(new THREE.Vector3());
    return [c.x, c.y, c.z];
  }

  matrix(): Mat4 {
    const s = this.s;
    const p = this.pivotPoint();
    const d2r = Math.PI / 180;
    let m = translation4([-p[0], -p[1], -p[2]]);
    m = mul4(rotationAxisAngle([1, 0, 0], s.rx * d2r), m);
    m = mul4(rotationAxisAngle([0, 1, 0], s.ry * d2r), m);
    m = mul4(rotationAxisAngle([0, 0, 1], s.rz * d2r), m);
    m = mul4(translation4([p[0] + s.tx, p[1] + s.ty, p[2] + s.tz]), m);
    return m;
  }

  private ids(): string[] {
    return this.s.scope === "all" ? modelEntityIds(this.app) : [this.meshId];
  }

  private preview() {
    const m = new THREE.Matrix4().fromArray(this.matrix());
    for (const id of modelEntityIds(this.app)) {
      const v = this.app.sync?.view<{ group: THREE.Group }>(id);
      if (!v?.group) continue;
      const on = this.ids().includes(id);
      v.group.matrixAutoUpdate = false;
      v.group.matrix.copy(on ? m : new THREE.Matrix4());
      v.group.matrixWorldNeedsUpdate = true;
    }
    this.app.viewer?.requestRender();
  }

  private upd(p: Partial<TransformState>) {
    this.set(p);
    this.preview();
  }

  renderPanel() {
    const s = this.s;
    const num = (k: keyof TransformState, unit: string, step: number) => (
      <NumberInput value={s[k] as number} unit={unit} step={step} onChange={(v) => this.upd({ [k]: v } as Partial<TransformState>)} />
    );
    return (
      <>
        <Section title="Apply to">
          <Segmented value={s.scope} options={[{ value: "all", label: "Whole model" }, { value: "mesh", label: "Active mesh" }]} onChange={(scope) => this.upd({ scope })} />
          <div style={{ height: 8 }} />
          <Field label="Rotation pivot">
            <Segmented value={s.pivot} options={[{ value: "center", label: "Model centre" }, { value: "origin", label: "World origin" }]} onChange={(pivot) => this.upd({ pivot })} />
          </Field>
        </Section>
        <Section title="Translate">
          <Field label="X">{num("tx", "mm", 1)}</Field>
          <Field label="Y">{num("ty", "mm", 1)}</Field>
          <Field label="Z">{num("tz", "mm", 1)}</Field>
        </Section>
        <Section title="Rotate">
          <Field label="About X">{num("rx", "°", 5)}</Field>
          <Field label="About Y">{num("ry", "°", 5)}</Field>
          <Field label="About Z">{num("rz", "°", 5)}</Field>
        </Section>
      </>
    );
  }

  private resetPreview() {
    for (const id of modelEntityIds(this.app)) {
      const v = this.app.sync?.view<{ group: THREE.Group }>(id);
      if (!v?.group) continue;
      v.group.matrix.identity();
      v.group.matrixAutoUpdate = true;
      v.group.matrixWorldNeedsUpdate = true;
    }
    this.app.viewer?.requestRender();
  }

  async commit() {
    const m = this.matrix();
    this.resetPreview();
    this.app.transformEntities(this.ids(), m, "Move / rotate");
  }

  deactivate() {
    this.resetPreview();
  }
}

// ---------------------------------------------------------------------------
// N-point alignment
// ---------------------------------------------------------------------------

interface NPointState {
  movingId: string;
  fixedId: string;
  moving: Vec3[];
  fixed: Vec3[];
  refine: boolean;
  rms: number | null;
}

class NPointAlignTool extends Tool<NPointState> {
  id = "nPointAlign";
  label = "N-Point Alignment";
  hint = "Click a point on the moving scan, then the matching point on the fixed scan. Repeat for at least 3 well-spread pairs.";
  okLabel = "Align";
  private overlay = new Overlay(this.app);

  constructor(app: App) {
    const meshes = app.doc.ofKind("mesh");
    if (meshes.length < 2) throw new Error("N-point alignment needs two meshes");
    super(app, { movingId: meshes[1].id, fixedId: meshes[0].id, moving: [], fixed: [], refine: true, rms: null });
  }

  private nextIsMoving() {
    return this.s.moving.length <= this.s.fixed.length;
  }

  onPointerDown(e: ToolPointerEvent): boolean {
    if (e.button !== 0) return false;
    const target = this.nextIsMoving() ? this.s.movingId : this.s.fixedId;
    const hit = e.pick((id) => id === target);
    if (!hit) {
      this.app.setStatus(`Click on "${this.app.doc.get(target)?.name}"`);
      return true;
    }
    if (this.nextIsMoving()) this.set({ moving: [...this.s.moving, hit.point] });
    else this.set({ fixed: [...this.s.fixed, hit.point] });
    this.update();
    return true;
  }

  private update() {
    this.overlay.clear();
    const s = this.s;
    const size = 9;
    if (s.moving.length) this.overlay.add(pointSprites(s.moving.flat(), "#ff8a3d", size, { depthTest: false })).renderOrder = 30;
    if (s.fixed.length) this.overlay.add(pointSprites(s.fixed.flat(), "#2dd4bf", size, { depthTest: false })).renderOrder = 30;
    s.moving.forEach((p, i) => this.overlay.label(`${i + 1}`, p));
    s.fixed.forEach((p, i) => this.overlay.label(`${i + 1}′`, p));
    const n = Math.min(s.moving.length, s.fixed.length);
    let rms: number | null = null;
    if (n >= 3) {
      try {
        rms = nPointAlign(s.moving.slice(0, n).map((m, i) => ({ moving: m, fixed: s.fixed[i] }))).rms;
      } catch {
        rms = null;
      }
    }
    this.set({ rms });
    this.app.setStatus(this.nextIsMoving() ? `Pick point ${s.moving.length + 1} on the moving scan` : `Pick the matching point ${s.fixed.length + 1} on the fixed scan`);
  }

  renderPanel() {
    const s = this.s;
    const meshes = this.app.doc.ofKind("mesh").map((m) => ({ value: m.id, label: m.name }));
    const n = Math.min(s.moving.length, s.fixed.length);
    return (
      <>
        <Section title="Scans">
          <Field label="Moving">
            <Select value={s.movingId} options={meshes} onChange={(movingId) => this.set({ movingId, moving: [], fixed: [] })} />
          </Field>
          <Field label="Fixed">
            <Select value={s.fixedId} options={meshes.filter((m) => m.value !== s.movingId)} onChange={(fixedId) => this.set({ fixedId, moving: [], fixed: [] })} />
          </Field>
        </Section>
        <Section title="Point pairs">
          <Stats rows={[["Pairs", `${n}`], ["RMS", s.rms === null ? "—" : mm(s.rms), s.rms !== null && s.rms < 1 ? "good" : undefined]]} />
          <div className="btn-row" style={{ marginTop: 8 }}>
            <button
              className="btn small"
              onClick={() => {
                if (s.fixed.length >= s.moving.length && s.fixed.length) this.set({ fixed: s.fixed.slice(0, -1) });
                else this.set({ moving: s.moving.slice(0, -1) });
                this.update();
              }}
            >
              Undo last
            </button>
            <button className="btn small" onClick={() => { this.set({ moving: [], fixed: [] }); this.update(); }}>
              Clear
            </button>
          </div>
          <div style={{ height: 8 }} />
          <Check checked={s.refine} onChange={(refine) => this.set({ refine })}>
            Refine with fine alignment (ICP)
          </Check>
        </Section>
      </>
    );
  }

  async commit() {
    const s = this.s;
    const n = Math.min(s.moving.length, s.fixed.length);
    if (n < 3) {
      this.app.toast("error", "Pick at least 3 point pairs");
      return false;
    }
    const res = nPointAlign(s.moving.slice(0, n).map((m, i) => ({ moving: m, fixed: s.fixed[i] })));
    let m = res.matrix;
    if (s.refine) {
      const moving = this.app.doc.get<MeshEntity>(s.movingId)!;
      const rt = this.app.sync!.meshRuntime(s.fixedId)!;
      const r = await this.app.busy("Refining alignment", () =>
        icp(moving.mesh.positions, rt.spatial.asQuery(), { initial: m, normals: moving.mesh.normals, sampleCount: 8000, maxIterations: 60 }),
      );
      if (r) m = r.matrix;
      if (r) this.app.toast("success", `Aligned — RMS ${r.rms.toFixed(4)} mm`);
    }
    this.app.transformEntities([s.movingId], m, "N-point align");
  }

  deactivate() {
    this.overlay.dispose();
  }
}

// ---------------------------------------------------------------------------
// Fine align (ICP)
// ---------------------------------------------------------------------------

class FineAlignTool extends Tool<{ movingId: string; fixedId: string; iterations: number; maxDist: number; result: string | null }> {
  id = "fineAlign";
  label = "Fine Align";
  hint = "Iterative closest point registration of two overlapping scans. Roughly pre-align them first (N-points).";
  okLabel = "Align";
  constructor(app: App) {
    const meshes = app.doc.ofKind("mesh");
    if (meshes.length < 2) throw new Error("Fine alignment needs two meshes");
    super(app, { movingId: meshes[1].id, fixedId: meshes[0].id, iterations: 50, maxDist: 0, result: null });
  }
  renderPanel() {
    const s = this.s;
    const meshes = this.app.doc.ofKind("mesh").map((m) => ({ value: m.id, label: m.name }));
    return (
      <Section title="Scans">
        <Field label="Moving">
          <Select value={s.movingId} options={meshes} onChange={(movingId) => this.set({ movingId })} />
        </Field>
        <Field label="Fixed">
          <Select value={s.fixedId} options={meshes.filter((m) => m.value !== s.movingId)} onChange={(fixedId) => this.set({ fixedId })} />
        </Field>
        <Field label="Iterations">
          <NumberInput value={s.iterations} min={5} max={500} step={5} digits={0} onChange={(iterations) => this.set({ iterations })} />
        </Field>
        <Field label="Max distance">
          <NumberInput value={s.maxDist} min={0} unit="mm" onChange={(maxDist) => this.set({ maxDist })} />
        </Field>
        <div className="note">0 = automatic. Limit it to ignore non-overlapping areas.</div>
        {s.result && <div className="note" style={{ marginTop: 8 }}>{s.result}</div>}
      </Section>
    );
  }
  async commit() {
    const s = this.s;
    const moving = this.app.doc.get<MeshEntity>(s.movingId)!;
    const rt = this.app.sync!.meshRuntime(s.fixedId)!;
    const r = await this.app.busy("Fine alignment", () =>
      icp(moving.mesh.positions, rt.spatial.asQuery(), {
        normals: moving.mesh.normals,
        maxIterations: s.iterations,
        sampleCount: 10000,
        maxCorrespondenceDistance: s.maxDist > 0 ? s.maxDist : undefined,
      }),
    );
    if (!r) return false;
    this.app.transformEntities([s.movingId], r.matrix, "Fine align");
    this.app.toast("success", `Fine alignment: RMS ${r.rms.toFixed(4)} mm after ${r.iterations} iterations (${int(r.pairs)} pairs)`);
  }
}

export function startAlignTool(app: App, id: string): void {
  const map: Record<string, ToolClass> = {
    objectAlign: ObjectAlignTool,
    transform: TransformTool,
    nPointAlign: NPointAlignTool,
    fineAlign: FineAlignTool,
  };
  const T = map[id];
  if (T) app.startTool(new T(app));
}
