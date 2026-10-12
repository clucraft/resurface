/**
 * Freeform surfacing:
 *  - QuadSurfaceTool: build a quad control cage on the scan by clicking points (or generate it
 *    from the selection); the Catmull–Clark limit surface snaps to the scan and exports as
 *    bicubic NURBS patches.
 *  - AutoSurfaceTool: one-click surfacing of organic shapes (coarse decimation → quad cage →
 *    snap & fit → G2 patches, G1 at extraordinary points).
 */
import * as THREE from "three";
import type { App } from "../app/App";
import { newId } from "../core/document";
import type { BodyEntity, MeshEntity } from "../core/entities";
import type { MeshData, Vec3 } from "../core/types";
import { compute } from "../app/compute";
import { Tool, type ToolClass, type ToolPointerEvent } from "./Tool";
import { Overlay } from "./overlay";
import { Section, Field, Slider, Stats, Check, Segmented, NumberInput, mm, int } from "../ui/widgets";
import { fatSegments, lineMaterial, pointSprites } from "../viewer/draw";
import { addQuad, addVertex, boundaryEdges, cloneQuadMesh, removeFace, removeUnusedVertices, type QuadMesh } from "../freeform/quadMesh";
import { limitMesh, subdivide } from "../freeform/catmullClark";
import { cageQuality } from "../freeform/analysis";
import type { BodyView } from "../viewer/views";

const SURFACE_MAT = () =>
  new THREE.MeshStandardMaterial({ color: "#7fb2ff", roughness: 0.35, metalness: 0.1, transparent: true, opacity: 0.72, side: THREE.DoubleSide, depthWrite: false });

function limitObject(cage: QuadMesh, mat: THREE.Material, levels = 3): THREE.Mesh | null {
  if (!cage.faces.length) return null;
  let m: MeshData;
  try {
    m = limitMesh(cage, levels);
  } catch {
    return null;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(m.positions, 3));
  if (m.normals) g.setAttribute("normal", new THREE.BufferAttribute(m.normals, 3));
  else g.computeVertexNormals();
  g.setIndex(new THREE.BufferAttribute(m.indices, 1));
  const mesh = new THREE.Mesh(g, mat);
  mesh.renderOrder = 6;
  return mesh;
}

function cagePairs(cage: QuadMesh): number[] {
  const seen = new Set<string>();
  const out: number[] = [];
  for (const f of cage.faces)
    for (let i = 0; i < f.length; i++) {
      const a = f[i], b = f[(i + 1) % f.length];
      const k = a < b ? `${a}_${b}` : `${b}_${a}`;
      if (seen.has(k)) continue;
      seen.add(k);
      out.push(...cage.vertices[a], ...cage.vertices[b]);
    }
  return out;
}

function isClosed(cage: QuadMesh): boolean {
  try {
    return cage.faces.length > 0 && boundaryEdges(cage).length === 0;
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// Quad surface (manual cage)
// ---------------------------------------------------------------------------

type QuadMode = "quad" | "move" | "delete" | "fix";

interface QuadState {
  mode: QuadMode;
  pending: number;
  fitStats: { rms: number; maxDev: number } | null;
  density: number;
  smoothness: number;
  showCage: boolean;
  version: number;
}

class QuadSurfaceTool extends Tool<QuadState> {
  id = "quadSurface";
  label = "Quad Surface";
  hint = "Add quads by clicking 4 points on the scan (corners snap to existing cage points), or generate a cage from the selection. Then ‘Fit to scan’. Drag points in Move mode.";
  okLabel = "Create surface";
  allowMeshSelection = true;
  private overlay = new Overlay(this.app);
  private cage: QuadMesh = { vertices: [], faces: [] };
  private history: QuadMesh[] = [];
  private pending: number[] = [];
  private drag: number | null = null;
  private surfMat = SURFACE_MAT();
  private cageMat = lineMaterial({ color: "#ff4fa3", width: 1.6, depthTest: false });
  private meshId: string;
  private editId: string | null = null;

  constructor(app: App, editId?: string) {
    super(app, { mode: "quad", pending: 0, fitStats: null, density: 120, smoothness: 0.005, showCage: true, version: 0 });
    const m = app.activeMesh();
    if (!m) throw new Error("Import a mesh first");
    this.meshId = m.id;
    if (editId) {
      const b = app.doc.get<BodyEntity>(editId);
      if (b?.feature.type === "freeform") {
        this.editId = editId;
        this.cage = cloneQuadMesh(b.feature.cage);
        if (b.feature.meshId) this.meshId = b.feature.meshId;
      }
    }
  }

  activate() {
    if (this.editId) this.app.sync?.view<BodyView>(this.editId)?.setVisible(false);
    this.app.ui.patch({ selectionMode: "smart" });
    this.redraw();
  }

  private push() {
    this.history.push(cloneQuadMesh(this.cage));
    if (this.history.length > 50) this.history.shift();
  }

  private changed() {
    this.set({ version: this.s.version + 1, pending: this.pending.length });
    this.redraw();
  }

  private nearestVertex(e: ToolPointerEvent, px = 12): number {
    const v = this.app.viewer!;
    let best = -1, bd = px;
    this.cage.vertices.forEach((p, i) => {
      const s = v.worldToScreen(p);
      const d = Math.hypot(s.x - e.x, s.y - e.y);
      if (d < bd) {
        bd = d;
        best = i;
      }
    });
    return best;
  }

  private meshHit(e: ToolPointerEvent): Vec3 | null {
    const hit = e.pick((id) => this.app.doc.get(id)?.kind === "mesh");
    return hit ? hit.point : null;
  }

  onPointerDown(e: ToolPointerEvent): boolean {
    if (e.button !== 0) return false;
    const mode = this.s.mode;
    if (mode === "quad") {
      let vi = this.nearestVertex(e);
      if (vi < 0) {
        const p = this.meshHit(e);
        if (!p) return true;
        this.push();
        vi = addVertex(this.cage, p);
      }
      if (this.pending.includes(vi)) return true;
      this.pending.push(vi);
      if (this.pending.length === 4) {
        try {
          if (!this.history.length || this.history[this.history.length - 1].faces.length === this.cage.faces.length) this.push();
          addQuad(this.cage, this.pending);
        } catch (err) {
          this.app.toast("error", err instanceof Error ? err.message : String(err));
        }
        this.pending = [];
      }
      this.changed();
      return true;
    }
    if (mode === "move" || mode === "fix") {
      const vi = this.nearestVertex(e);
      if (vi < 0) return mode === "move" ? false : true;
      if (mode === "fix") {
        this.push();
        const fixed = this.cage.fixed ?? this.cage.vertices.map(() => false);
        fixed[vi] = !fixed[vi];
        this.cage.fixed = fixed;
        this.changed();
        return true;
      }
      this.push();
      this.drag = vi;
      return true;
    }
    if (mode === "delete") {
      const f = this.pickFace(e);
      if (f >= 0) {
        this.push();
        removeFace(this.cage, f);
        removeUnusedVertices(this.cage);
        this.changed();
      }
      return true;
    }
    return false;
  }

  onPointerMove(e: ToolPointerEvent): boolean {
    if (this.drag === null || !(e.native.buttons & 1)) return false;
    const p = this.meshHit(e);
    if (p) {
      this.cage.vertices[this.drag] = p;
      this.redraw(false);
    }
    return true;
  }

  onPointerUp(): boolean {
    if (this.drag === null) return false;
    this.drag = null;
    this.changed();
    return true;
  }

  onKeyDown(e: KeyboardEvent): boolean {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z" && this.history.length) {
      this.cage = this.history.pop()!;
      this.pending = [];
      this.changed();
      return true;
    }
    if (e.key === "Escape" && this.pending.length) {
      this.pending = [];
      this.changed();
      return true;
    }
    return false;
  }

  /** Face whose projected polygon contains the cursor (closest to the camera). */
  private pickFace(e: ToolPointerEvent): number {
    const v = this.app.viewer!;
    let best = -1, bestZ = Infinity;
    this.cage.faces.forEach((f, fi) => {
      const pts = f.map((i) => v.worldToScreen(this.cage.vertices[i]));
      let inside = false;
      for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
        if (pts[i].y > e.y !== pts[j].y > e.y && e.x < ((pts[j].x - pts[i].x) * (e.y - pts[i].y)) / (pts[j].y - pts[i].y) + pts[i].x) inside = !inside;
      }
      const z = pts.reduce((a, p) => a + p.z, 0) / pts.length;
      if (inside && z < bestZ) {
        bestZ = z;
        best = fi;
      }
    });
    return best;
  }

  private redraw(full = true) {
    this.overlay.clear();
    const c = this.cage;
    if (full || c.faces.length < 400) {
      const surf = limitObject(c, this.surfMat, c.faces.length > 600 ? 2 : 3);
      if (surf) this.overlay.add(surf);
    }
    if (this.s.showCage) {
      const pairs = cagePairs(c);
      if (pairs.length) this.overlay.add(fatSegments(pairs, this.cageMat)).renderOrder = 30;
      const fixed = c.fixed ?? [];
      const normal: number[] = [], fx: number[] = [], pend: number[] = [];
      c.vertices.forEach((p, i) => (this.pending.includes(i) ? pend : fixed[i] ? fx : normal).push(...p));
      if (normal.length) this.overlay.add(pointSprites(normal, "#ffffff", 6, { depthTest: false })).renderOrder = 31;
      if (fx.length) this.overlay.add(pointSprites(fx, "#f45d5d", 8, { depthTest: false })).renderOrder = 31;
      if (pend.length) this.overlay.add(pointSprites(pend, "#ffd23f", 10, { depthTest: false })).renderOrder = 32;
    }
  }

  private async fromSelection() {
    const rt = this.app.sync?.meshRuntime(this.meshId);
    const mesh = this.app.doc.get<MeshEntity>(this.meshId)?.mesh;
    if (!rt || !mesh) return;
    if (!rt.selectionCount) {
      this.app.toast("error", "Select a region first (smart select or brush)");
      return;
    }
    const res = await this.app.busy("Building cage from selection", () =>
      compute.run<{ cage: QuadMesh; rms: number; maxDev: number }>(
        "autosurface",
        { mesh, mask: rt.selection.slice(), coarseTriangles: this.s.density, opts: { smoothness: this.s.smoothness, iterations: 6 } },
        { onProgress: (f, m) => this.app.setProgress(f, m) },
      ),
    );
    if (!res) return;
    this.push();
    this.cage = res.cage;
    this.pending = [];
    this.set({ fitStats: { rms: res.rms, maxDev: res.maxDev } });
    rt.clearSelection();
    this.changed();
  }

  private async fit() {
    const mesh = this.app.doc.get<MeshEntity>(this.meshId)?.mesh;
    if (!mesh || !this.cage.faces.length) return;
    const res = await this.app.busy("Fitting to scan", () =>
      compute.run<{ cage: QuadMesh; rms: number; maxDev: number }>(
        "fitCage",
        { mesh, cage: this.cage, opts: { smoothness: this.s.smoothness, iterations: 8, maxDistance: this.app.sceneSize() * 0.05 } },
        { onProgress: (f, m) => this.app.setProgress(f, m) },
      ),
    );
    if (!res) return;
    this.push();
    this.cage = res.cage;
    this.set({ fitStats: { rms: res.rms, maxDev: res.maxDev } });
    this.changed();
  }

  private refine() {
    if (!this.cage.faces.length) return;
    this.push();
    this.cage = subdivide(this.cage);
    this.changed();
  }

  renderPanel() {
    const s = this.s;
    let q: ReturnType<typeof cageQuality> | null = null;
    try {
      q = this.cage.faces.length ? cageQuality(this.cage) : null;
    } catch {
      q = null;
    }
    return (
      <>
        <Section title="Edit cage">
          <Segmented
            value={s.mode}
            options={[
              { value: "quad", label: "Add quad" },
              { value: "move", label: "Move" },
              { value: "fix", label: "Pin" },
              { value: "delete", label: "Delete" },
            ]}
            onChange={(mode) => {
              this.pending = [];
              this.set({ mode, pending: 0 });
              this.redraw();
            }}
          />
          <div className="note" style={{ marginTop: 8 }}>
            {s.mode === "quad" && `Click 4 corners (${s.pending}/4). Shared corners connect quads.`}
            {s.mode === "move" && "Drag a cage point along the scan."}
            {s.mode === "fix" && "Click points to pin/unpin them (pinned points are not moved by the fit)."}
            {s.mode === "delete" && "Click inside a quad to delete it."}
          </div>
          <div className="btn-row" style={{ marginTop: 8 }}>
            <button className="btn small" onClick={() => void this.fit()} disabled={!this.cage.faces.length}>
              Fit to scan
            </button>
            <button className="btn small" onClick={() => this.refine()} disabled={!this.cage.faces.length}>
              Subdivide
            </button>
            <button
              className="btn small"
              disabled={!this.history.length}
              onClick={() => {
                this.cage = this.history.pop()!;
                this.changed();
              }}
            >
              Undo
            </button>
          </div>
        </Section>
        <Section title="Cage from selection">
          <Field label="Density">
            <Slider value={s.density} min={20} max={1500} step={10} onChange={(density) => this.set({ density })} format={(v) => `${v} ▲`} />
          </Field>
          <button className="btn small" onClick={() => void this.fromSelection()}>
            Generate from selection
          </button>
        </Section>
        <Section title="Fit">
          <Field label="Smoothness">
            <NumberInput value={s.smoothness} min={0} step={0.001} digits={4} onChange={(smoothness) => this.set({ smoothness })} />
          </Field>
          <Check checked={s.showCage} onChange={(showCage) => { this.set({ showCage }); this.redraw(); }}>
            Show control cage
          </Check>
          <Stats
            rows={[
              ["Faces", int(this.cage.faces.length)],
              ["Extraordinary pts", q ? int(q.extraordinaryCount) : "—"],
              ["RMS deviation", s.fitStats ? mm(s.fitStats.rms, 4) : "—"],
              ["Max deviation", s.fitStats ? mm(s.fitStats.maxDev, 4) : "—"],
              ["Closed", isClosed(this.cage) ? "yes (solid)" : "no (surface)"],
            ]}
          />
        </Section>
      </>
    );
  }

  async commit() {
    if (!this.cage.faces.length) {
      this.app.toast("error", "The cage is empty");
      return false;
    }
    const feature = {
      type: "freeform" as const,
      cage: cloneQuadMesh(this.cage),
      meshId: this.meshId,
      solid: isClosed(this.cage),
      stats: this.s.fitStats ? { rms: this.s.fitStats.rms, maxAbs: this.s.fitStats.maxDev, mean: 0, count: 0 } : undefined,
    };
    if (this.editId) {
      this.app.doc.transact("Edit freeform", () => this.app.doc.update<BodyEntity>(this.editId!, { feature }));
      return;
    }
    const e = this.app.doc.add<BodyEntity>({ id: newId("b"), kind: "body", name: this.app.doc.uniqueName("Freeform"), visible: true, feature });
    this.app.setActiveEntity(e.id);
  }

  deactivate() {
    if (this.editId) this.app.sync?.view<BodyView>(this.editId)?.setVisible(this.app.doc.get(this.editId)?.visible ?? true);
    this.overlay.dispose();
    this.surfMat.dispose();
    this.cageMat.dispose();
  }
}

// ---------------------------------------------------------------------------
// Autosurface
// ---------------------------------------------------------------------------

interface AutoState {
  density: number;
  smoothness: number;
  creaseAngle: number;
  useSelection: boolean;
  result: { cage: QuadMesh; rms: number; maxDev: number; patches: number; extraordinary: number } | null;
  running: boolean;
}

class AutoSurfaceTool extends Tool<AutoState> {
  id = "autoSurface";
  label = "Autosurface";
  hint = "One-click NURBS surfacing for organic shapes. Increase the density for more patches and closer fit; the result is G2 except at extraordinary points.";
  okLabel = "Create surface";
  allowMeshSelection = true;
  private overlay = new Overlay(this.app);
  private surfMat = SURFACE_MAT();
  private cageMat = lineMaterial({ color: "#ff4fa3", width: 1, opacity: 0.6 });
  private meshId: string;

  constructor(app: App) {
    super(app, { density: 300, smoothness: 0.004, creaseAngle: 0, useSelection: false, result: null, running: false });
    const m = app.activeMesh();
    if (!m) throw new Error("Import a mesh first");
    this.meshId = m.id;
    const rt = app.sync?.meshRuntime(m.id);
    if (rt?.selectionCount) this.set({ useSelection: true });
  }

  activate() {
    void this.run();
  }

  private async run() {
    const mesh = this.app.doc.get<MeshEntity>(this.meshId)?.mesh;
    const rt = this.app.sync?.meshRuntime(this.meshId);
    if (!mesh || !rt) return;
    const mask = this.s.useSelection && rt.selectionCount ? rt.selection.slice() : undefined;
    this.set({ running: true });
    const t0 = performance.now();
    const res = await this.app.busy("Autosurfacing", () =>
      compute.run<{ cage: QuadMesh; rms: number; maxDev: number; patches: number; quality: { extraordinaryCount: number } }>(
        "autosurface",
        {
          mesh,
          mask,
          coarseTriangles: this.s.density,
          opts: { smoothness: this.s.smoothness, iterations: 8, creaseAngleDeg: this.s.creaseAngle > 0 ? this.s.creaseAngle : undefined },
        },
        { onProgress: (f, m) => this.app.setProgress(f, m) },
      ),
    );
    this.set({ running: false });
    if (!res) return;
    this.set({ result: { cage: res.cage, rms: res.rms, maxDev: res.maxDev, patches: res.patches, extraordinary: res.quality.extraordinaryCount } });
    this.app.setStatus(`Autosurface: ${res.patches} patches, RMS ${res.rms.toFixed(4)} mm in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    this.redraw();
  }

  private redraw() {
    this.overlay.clear();
    const r = this.s.result;
    if (!r) return;
    const surf = limitObject(r.cage, this.surfMat, r.cage.faces.length > 1500 ? 2 : 3);
    if (surf) this.overlay.add(surf);
    this.overlay.add(fatSegments(cagePairs(r.cage), this.cageMat)).renderOrder = 7;
  }

  renderPanel() {
    const s = this.s;
    const rt = this.app.sync?.meshRuntime(this.meshId);
    const r = s.result;
    return (
      <>
        <Section title="Settings">
          <Field label="Patch density">
            <Slider value={s.density} min={20} max={4000} step={10} onChange={(density) => this.set({ density })} format={(v) => `${v}`} />
          </Field>
          <Field label="Smoothness">
            <NumberInput value={s.smoothness} min={0} step={0.001} digits={4} onChange={(smoothness) => this.set({ smoothness })} />
          </Field>
          <Field label="Sharp edges ≥">
            <NumberInput value={s.creaseAngle} min={0} max={170} unit="°" step={5} onChange={(creaseAngle) => this.set({ creaseAngle })} />
          </Field>
          <Check checked={s.useSelection && !!rt?.selectionCount} onChange={(useSelection) => this.set({ useSelection })}>
            Selected region only
          </Check>
          <button className="btn small" disabled={s.running} onClick={() => void this.run()}>
            Recompute
          </button>
        </Section>
        <Section title="Result">
          {r ? (
            <Stats
              rows={[
                ["Patches", int(r.patches)],
                ["Cage faces", int(r.cage.faces.length)],
                ["Extraordinary pts", int(r.extraordinary)],
                ["RMS deviation", mm(r.rms, 4), r.rms < 0.05 ? "good" : r.rms < 0.2 ? "warn" : "bad"],
                ["Max deviation", mm(r.maxDev, 4)],
                ["Closed", isClosed(r.cage) ? "yes (solid)" : "no (surface)"],
              ]}
            />
          ) : (
            <div className="note">{s.running ? "Computing…" : "—"}</div>
          )}
        </Section>
      </>
    );
  }

  async commit() {
    const r = this.s.result;
    if (!r) return false;
    const e = this.app.doc.add<BodyEntity>({
      id: newId("b"),
      kind: "body",
      name: this.app.doc.uniqueName("Autosurface"),
      visible: true,
      feature: { type: "freeform", cage: r.cage, meshId: this.meshId, solid: isClosed(r.cage), stats: { rms: r.rms, maxAbs: r.maxDev, mean: 0, count: 0 } },
    });
    this.app.setActiveEntity(e.id);
  }

  deactivate() {
    this.overlay.dispose();
    this.surfMat.dispose();
    this.cageMat.dispose();
  }
}

export function startFreeformTool(app: App, id: string, editId?: string): void {
  if (id === "quadSurface") app.startTool(new QuadSurfaceTool(app, editId));
  else if (id === "autoSurface") app.startTool(new AutoSurfaceTool(app));
}

export type { ToolClass };
