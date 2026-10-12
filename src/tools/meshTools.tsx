/** Mesh optimisation / repair tools with property panels. */
import * as THREE from "three";
import type { App } from "../app/App";
import type { MeshData, PlaneDef } from "../core/types";
import { compute } from "../app/compute";
import { markPrepared } from "../app/MeshRuntime";
import { Tool, type ToolClass } from "./Tool";
import { Overlay } from "./overlay";
import { Section, Field, NumberInput, Slider, Check, Stats, Select, Segmented, int, mm } from "../ui/widgets";
import { fatSegments, lineMaterial, polylinesToPairs, translucentMaterial } from "../viewer/draw";
import { meshOffsetRange, offsetFrame, planeSourceOptions, resolvePlaneSource } from "./planes";
import type { MeshView } from "../viewer/MeshView";
import type { Vec3 } from "../core/types";

abstract class MeshTool<S extends object> extends Tool<S> {
  protected meshId: string;
  constructor(app: App, initial: S) {
    super(app, initial);
    const e = app.activeMesh();
    if (!e) throw new Error("Import a mesh first");
    this.meshId = e.id;
  }
  protected get entity() {
    const e = this.app.doc.get(this.meshId);
    if (!e || e.kind !== "mesh") throw new Error("The mesh was deleted");
    return e;
  }
  protected get runtime() {
    return this.app.sync!.meshRuntime(this.meshId)!;
  }
  protected async applyResult(label: string, job: string, args: Record<string, unknown>, prepare: "reorder" | "normals") {
    const name = this.entity.name;
    const before = this.entity.mesh.indices.length / 3;
    const result = await this.app.busy(`${label}: ${name}`, async () => {
      const mesh = await compute.run<MeshData>(job, { mesh: this.entity.mesh, prepare, ...args }, { onProgress: (f, m) => this.app.setProgress(f, m) });
      markPrepared(mesh);
      this.app.replaceMesh(this.meshId, mesh, label, true);
      return mesh;
    });
    if (result) {
      const after = result.indices.length / 3;
      this.app.toast("success", `${label}: ${int(before)} → ${int(after)} triangles`);
    }
    return result;
  }
}

// ---------------------------------------------------------------------------
// Reduce
// ---------------------------------------------------------------------------

class ReduceTool extends MeshTool<{ percent: number; target: number; mode: "percent" | "count" }> {
  id = "reduce";
  label = "Reduce Mesh";
  hint = "Quadric decimation keeps detail where curvature is high. Large scans are first clustered, then refined.";
  okLabel = "Reduce";
  constructor(app: App) {
    super(app, { percent: 50, target: 0, mode: "percent" });
    this.set({ target: Math.round(this.tris * 0.5) });
  }
  get tris() {
    return this.entity.mesh.indices.length / 3;
  }
  get targetCount() {
    return this.s.mode === "percent" ? Math.max(4, Math.round((this.tris * this.s.percent) / 100)) : Math.max(4, Math.min(this.tris, this.s.target));
  }
  renderPanel() {
    return (
      <Section title="Target">
        <Field label="Mode">
          <Segmented value={this.s.mode} options={[{ value: "percent", label: "Percent" }, { value: "count", label: "Triangles" }]} onChange={(mode) => this.set({ mode })} />
        </Field>
        {this.s.mode === "percent" ? (
          <Field label="Keep">
            <Slider value={this.s.percent} min={1} max={99} onChange={(percent) => this.set({ percent })} format={(v) => `${v}%`} />
          </Field>
        ) : (
          <Field label="Triangles">
            <NumberInput value={this.s.target} step={1000} min={4} digits={0} onChange={(target) => this.set({ target })} />
          </Field>
        )}
        <Stats rows={[["Current", int(this.tris)], ["Result", int(this.targetCount)]]} />
      </Section>
    );
  }
  async commit() {
    await this.applyResult("Reduce", "decimate", { targetTriangles: this.targetCount }, "reorder");
  }
}

// ---------------------------------------------------------------------------
// Smooth
// ---------------------------------------------------------------------------

class SmoothTool extends MeshTool<{ iterations: number; strength: number; selectionOnly: boolean }> {
  id = "smooth";
  label = "Smooth Mesh";
  hint = "Taubin smoothing removes scanner noise without shrinking the part. Restrict it to the selection to protect sharp details.";
  okLabel = "Smooth";
  allowMeshSelection = true;
  constructor(app: App) {
    super(app, { iterations: 5, strength: 0.5, selectionOnly: false });
    if (this.runtime.selectionCount > 0) this.set({ selectionOnly: true });
  }
  renderPanel() {
    const sel = this.runtime.selectionCount;
    return (
      <Section title="Settings">
        <Field label="Iterations">
          <Slider value={this.s.iterations} min={1} max={50} onChange={(iterations) => this.set({ iterations })} />
        </Field>
        <Field label="Strength">
          <Slider value={this.s.strength} min={0.1} max={0.9} step={0.05} onChange={(strength) => this.set({ strength })} format={(v) => v.toFixed(2)} />
        </Field>
        <Check checked={this.s.selectionOnly && sel > 0} onChange={(selectionOnly) => this.set({ selectionOnly })}>
          Selected area only {sel > 0 ? `(${int(sel)} ▲)` : "(no selection)"}
        </Check>
      </Section>
    );
  }
  async commit() {
    const rt = this.runtime;
    let mask: Uint8Array | undefined;
    if (this.s.selectionOnly && rt.selectionCount > 0) {
      mask = new Uint8Array(rt.vertexCount);
      const verts = rt.selectedVertices();
      for (let i = 0; i < verts.length; i++) mask[verts[i]] = 1;
    }
    const lambda = this.s.strength;
    await this.applyResult("Smooth", "smooth", { iterations: this.s.iterations, lambda, mu: -(lambda + 0.03), mask }, "normals");
  }
  onSelectionChanged() {
    this.state.patch({});
  }
}

// ---------------------------------------------------------------------------
// Fill holes
// ---------------------------------------------------------------------------

interface HoleInfo {
  index: number;
  edges: number;
  perimeter: number;
  center: Vec3;
  points: Vec3[];
}

class FillHolesTool extends MeshTool<{ holes: HoleInfo[] | null; selected: number[]; maxEdges: number; smooth: boolean }> {
  id = "fillHoles";
  label = "Fill Holes";
  hint = "Holes are listed by size. The largest boundary is usually the open rim of the scan — leave it unchecked unless you want to close it.";
  okLabel = "Fill";
  private overlay = new Overlay(this.app);
  private mat = lineMaterial({ color: "#ff5d8f", width: 3, depthTest: false });
  private matSel = lineMaterial({ color: "#2dd4bf", width: 3.5, depthTest: false });

  constructor(app: App) {
    super(app, { holes: null, selected: [], maxEdges: 2000, smooth: true });
  }

  activate() {
    void this.scan();
  }

  private async scan() {
    const mesh = this.entity.mesh;
    const loops = await compute.run<Uint32Array[]>("boundaryLoops", { mesh });
    const pos = mesh.positions;
    const holes: HoleInfo[] = loops.map((loop, index) => {
      let per = 0;
      const pts: Vec3[] = [];
      let cx = 0, cy = 0, cz = 0;
      for (let i = 0; i < loop.length; i++) {
        const a = loop[i], b = loop[(i + 1) % loop.length];
        per += Math.hypot(pos[3 * a] - pos[3 * b], pos[3 * a + 1] - pos[3 * b + 1], pos[3 * a + 2] - pos[3 * b + 2]);
        pts.push([pos[3 * a], pos[3 * a + 1], pos[3 * a + 2]]);
        cx += pos[3 * a];
        cy += pos[3 * a + 1];
        cz += pos[3 * a + 2];
      }
      return { index, edges: loop.length, perimeter: per, center: [cx / loop.length, cy / loop.length, cz / loop.length], points: pts };
    });
    holes.sort((a, b) => a.perimeter - b.perimeter);
    // pre-select all but the largest loop when there are several (largest = scan rim)
    const selected = holes.length > 1 ? holes.slice(0, -1).filter((h) => h.edges <= this.s.maxEdges).map((h) => h.index) : holes.map((h) => h.index);
    this.set({ holes, selected });
    this.draw();
  }

  private draw() {
    this.overlay.clear();
    const holes = this.s.holes ?? [];
    const sel = new Set(this.s.selected);
    const a = holes.filter((h) => !sel.has(h.index)).map((h) => ({ points: h.points, closed: true }));
    const b = holes.filter((h) => sel.has(h.index)).map((h) => ({ points: h.points, closed: true }));
    if (a.length) this.overlay.add(fatSegments(polylinesToPairs(a), this.mat)).renderOrder = 20;
    if (b.length) this.overlay.add(fatSegments(polylinesToPairs(b), this.matSel)).renderOrder = 21;
  }

  private toggle(i: number) {
    const s = new Set(this.s.selected);
    if (s.has(i)) s.delete(i);
    else s.add(i);
    this.set({ selected: [...s] });
    this.draw();
  }

  renderPanel() {
    const holes = this.s.holes;
    if (!holes) return <Section><div className="note">Scanning boundaries…</div></Section>;
    if (!holes.length)
      return (
        <Section>
          <div className="note">No holes — the mesh is watertight.</div>
        </Section>
      );
    const sel = new Set(this.s.selected);
    return (
      <>
        <Section title={`Holes (${holes.length})`} right={
          <span className="btn-row">
            <button className="btn small" onClick={() => { this.set({ selected: holes.map((h) => h.index) }); this.draw(); }}>All</button>
            <button className="btn small" onClick={() => { this.set({ selected: [] }); this.draw(); }}>None</button>
          </span>
        }>
          <div className="list">
            {holes.map((h, i) => (
              <div key={h.index} className={`list-row${sel.has(h.index) ? " on" : ""}`} onClick={() => this.toggle(h.index)}>
                <input type="checkbox" readOnly checked={sel.has(h.index)} />
                Hole {i + 1}
                <span className="mono">
                  {h.edges} edges · {h.perimeter.toFixed(1)} mm
                </span>
              </div>
            ))}
          </div>
        </Section>
        <Section title="Options">
          <Check checked={this.s.smooth} onChange={(smooth) => this.set({ smooth })}>
            Curvature-matched (smooth) fill
          </Check>
        </Section>
      </>
    );
  }

  async commit() {
    if (!this.s.selected.length) return;
    await this.applyResult("Fill holes", "fillHoles", { opts: { loops: this.s.selected, smooth: this.s.smooth } }, "reorder");
  }

  deactivate() {
    this.overlay.dispose();
    this.mat.dispose();
    this.matSel.dispose();
  }
}

// ---------------------------------------------------------------------------
// Clean (small shells)
// ---------------------------------------------------------------------------

class CleanTool extends MeshTool<{ sizes: number[] | null; minTriangles: number; degenerate: boolean }> {
  id = "clean";
  label = "Clean Mesh";
  hint = "Removes floating noise shells and degenerate triangles left by the scanner.";
  okLabel = "Clean";
  constructor(app: App) {
    super(app, { sizes: null, minTriangles: 100, degenerate: true });
  }
  activate() {
    void compute.run<{ count: number; sizes: Uint32Array }>("components", { mesh: this.entity.mesh }).then((c) => {
      const sizes = [...c.sizes].sort((a, b) => b - a);
      const largest = sizes[0] ?? 0;
      this.set({ sizes, minTriangles: Math.max(10, Math.round(largest * 0.01)) });
    });
  }
  renderPanel() {
    const sizes = this.s.sizes;
    const removed = sizes ? sizes.filter((s) => s < this.s.minTriangles) : [];
    return (
      <Section title="Shells">
        {!sizes ? (
          <div className="note">Analysing connectivity…</div>
        ) : (
          <>
            <Stats
              rows={[
                ["Shells", int(sizes.length)],
                ["Largest", `${int(sizes[0] ?? 0)} ▲`],
                ["To remove", `${int(removed.length)} (${int(removed.reduce((a, b) => a + b, 0))} ▲)`, removed.length ? "warn" : undefined],
              ]}
            />
            <div style={{ height: 10 }} />
            <Field label="Min. shell size">
              <NumberInput value={this.s.minTriangles} min={1} step={10} digits={0} unit="▲" onChange={(minTriangles) => this.set({ minTriangles })} />
            </Field>
            <Check checked={this.s.degenerate} onChange={(degenerate) => this.set({ degenerate })}>
              Remove degenerate triangles
            </Check>
          </>
        )}
      </Section>
    );
  }
  async commit() {
    await this.applyResult("Clean", "removeSmallComponents", { minTriangles: this.s.minTriangles, degenerate: this.s.degenerate }, "normals");
  }
}

// ---------------------------------------------------------------------------
// Clip with plane
// ---------------------------------------------------------------------------

class ClipTool extends MeshTool<{ source: string; offset: number; flip: boolean; cut: boolean }> {
  id = "clip";
  label = "Clip Mesh";
  hint = "Cuts away everything on one side of a plane. The preview hides the part that will be removed.";
  okLabel = "Clip";
  private overlay = new Overlay(this.app);
  private clipPlane = new THREE.Plane();

  constructor(app: App) {
    super(app, { source: "world:XY", offset: 0, flip: false, cut: true });
    const f = resolvePlaneSource(app, "world:XY")!;
    this.set({ offset: meshOffsetRange(app, f).mid });
  }

  private plane(): PlaneDef | null {
    const base = resolvePlaneSource(this.app, this.s.source);
    if (!base) return null;
    const f = offsetFrame(base, this.s.offset);
    const n = this.s.flip ? ([-f.zAxis[0], -f.zAxis[1], -f.zAxis[2]] as Vec3) : f.zAxis;
    return { kind: "plane", origin: f.origin, normal: n };
  }

  activate() {
    this.update();
  }

  private update() {
    this.overlay.clear();
    const p = this.plane();
    const mv = this.app.sync?.view<MeshView>(this.meshId);
    if (!p || !mv) return;
    // keep the positive side (normal side); three.js clipping keeps the side where dot(n,x)+c > 0... it removes the negative
    this.clipPlane.setFromNormalAndCoplanarPoint(new THREE.Vector3(...p.normal), new THREE.Vector3(...p.origin));
    mv.material.clippingPlanes = [this.clipPlane];
    mv.material.needsUpdate = true;
    const size = this.app.sceneSize() * 0.6;
    const g = new THREE.PlaneGeometry(size, size);
    const m = new THREE.Mesh(g, translucentMaterial("#ff8a3d", 0.22));
    m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), new THREE.Vector3(...p.normal));
    m.position.set(...p.origin);
    this.overlay.add(m);
    this.app.viewer?.requestRender();
  }

  renderPanel() {
    const base = resolvePlaneSource(this.app, this.s.source);
    const range = base ? meshOffsetRange(this.app, base) : { min: -100, max: 100 };
    const setU = (patch: Partial<typeof this.s>) => {
      this.set(patch);
      this.update();
    };
    return (
      <Section title="Plane">
        <Field label="Plane">
          <Select
            value={this.s.source}
            options={planeSourceOptions(this.app, { axial: false })}
            onChange={(source) => {
              const f = resolvePlaneSource(this.app, source);
              setU({ source, offset: f && source.startsWith("world") ? meshOffsetRange(this.app, f).mid : 0 });
            }}
          />
        </Field>
        <Field label="Offset">
          <NumberInput value={this.s.offset} unit="mm" step={0.5} onChange={(offset) => setU({ offset })} />
        </Field>
        <Slider value={this.s.offset} min={range.min} max={range.max} step={(range.max - range.min) / 500 || 0.1} onChange={(offset) => setU({ offset })} format={(v) => mm(v, 1)} />
        <div style={{ height: 8 }} />
        <Check checked={this.s.flip} onChange={(flip) => setU({ flip })}>
          Keep the other side
        </Check>
        <Check checked={this.s.cut} onChange={(cut) => this.set({ cut })}>
          Cut triangles exactly at the plane
        </Check>
      </Section>
    );
  }

  async commit() {
    const p = this.plane();
    if (!p) return false;
    await this.applyResult("Clip", "clip", { plane: p, keep: "positive", cut: this.s.cut }, "normals");
  }

  deactivate() {
    const mv = this.app.sync?.view<MeshView>(this.meshId);
    if (mv) {
      mv.material.clippingPlanes = null;
      mv.material.needsUpdate = true;
    }
    this.overlay.dispose();
  }
}

export function startMeshTool(app: App, id: string): void {
  const map: Record<string, ToolClass> = {
    reduce: ReduceTool,
    smooth: SmoothTool,
    fillHoles: FillHolesTool,
    clean: CleanTool,
    clip: ClipTool,
  };
  const T = map[id];
  if (T) app.startTool(new T(app));
}
