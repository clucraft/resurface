/**
 * Fit Surface: approximates the selected mesh region with a NURBS surface, extended beyond the
 * selection for later trimming, with a real-time deviation colour map on the scan.
 */
import * as THREE from "three";
import type { App } from "../app/App";
import { newId } from "../core/document";
import type { BodyEntity, MeshEntity, PrimitiveEntity } from "../core/entities";
import type { BSplineSurfaceDef, FitStats, Frame } from "../core/types";
import { frameFromNormal } from "../core/vec";
import { fitBSplineSurface, tessellateSurface } from "../fit/bspline";
import { Tool } from "./Tool";
import { Overlay } from "./overlay";
import { Section, Field, Slider, Stats, Select, Check, mm, int } from "../ui/widgets";
import { deviationColors } from "../analysis/colormap";
import type { MeshView } from "../viewer/MeshView";
import { fatSegments, lineMaterial } from "../viewer/draw";

interface FitSurfaceState {
  nu: number;
  nv: number;
  smoothLog: number;
  extend: number;
  frameSource: string;
  showCage: boolean;
  stats: FitStats | null;
  error: string | null;
  points: number;
}

class FitSurfaceTool extends Tool<FitSurfaceState> {
  id = "fitSurface";
  label = "Fit Surface";
  hint = "Select a smooth region of the scan (brush/lasso/smart). The NURBS surface updates live; the colour map shows its deviation from the selected points.";
  okLabel = "Create surface";
  allowMeshSelection = true;
  private overlay = new Overlay(this.app);
  private meshId: string;
  private timer = 0;
  private surface: BSplineSurfaceDef | null = null;
  private surfMat = new THREE.MeshStandardMaterial({
    color: "#7fb2ff",
    roughness: 0.4,
    metalness: 0.1,
    transparent: true,
    opacity: 0.6,
    side: THREE.DoubleSide,
    depthWrite: false,
    polygonOffset: true,
    polygonOffsetFactor: -2,
    polygonOffsetUnits: -4,
  });
  private cageMat = lineMaterial({ color: "#ff4fa3", width: 1, opacity: 0.8 });
  private editId: string | null = null;

  constructor(app: App, editId?: string) {
    super(app, { nu: 10, nv: 10, smoothLog: -8, extend: 10, frameSource: "auto", showCage: false, stats: null, error: null, points: 0 });
    const m = app.activeMesh();
    if (!m) throw new Error("Import a mesh first");
    this.meshId = m.id;
    if (editId) {
      const b = app.doc.get<BodyEntity>(editId);
      if (b?.feature.type === "fitSurface") {
        this.editId = editId;
        this.surface = b.feature.surface;
        this.set({ nu: b.feature.surface.nu, nv: b.feature.surface.nv });
      }
    }
  }

  activate() {
    if (this.editId) this.app.sync?.view<{ setVisible(v: boolean): void }>(this.editId)?.setVisible(false);
    this.app.sync?.view<MeshView>(this.meshId)?.setSelectionVisible(false);
    this.schedule(0);
  }

  onSelectionChanged() {
    this.schedule(150);
  }

  private schedule(ms = 120) {
    window.clearTimeout(this.timer);
    this.timer = window.setTimeout(() => this.fit(), ms);
  }

  private frame(): Frame | undefined {
    const src = this.s.frameSource;
    if (src === "auto") return undefined;
    const e = this.app.doc.get<PrimitiveEntity>(src);
    if (e?.def.kind === "plane") return frameFromNormal(e.def.origin, e.def.normal);
    return undefined;
  }

  private fit() {
    const rt = this.app.sync?.meshRuntime(this.meshId);
    const mesh = this.app.doc.get<MeshEntity>(this.meshId)?.mesh;
    if (!rt || !mesh) return;
    if (rt.selectionCount < 20) {
      if (!this.editId) {
        this.overlay.clear();
        this.surface = null;
      } else this.draw();
      this.set({ stats: null, error: rt.selectionCount ? "Select a larger region" : null, points: 0 });
      return;
    }
    const verts = rt.selectedVertices();
    try {
      const s = this.s;
      const res = fitBSplineSurface(mesh.positions, {
        indices: verts,
        normals: mesh.normals,
        nu: s.nu,
        nv: s.nv,
        smoothing: Math.pow(10, s.smoothLog),
        extend: s.extend / 100,
        frame: this.frame(),
        iterations: 2,
        maxPoints: 60000,
      });
      this.surface = res.surface;
      this.set({ stats: res.stats, error: null, points: verts.length });
      this.colorMap(verts, res.residuals, res.stats);
      this.draw();
    } catch (e) {
      this.set({ error: e instanceof Error ? e.message : String(e) });
    }
  }

  private colorMap(verts: Uint32Array, residuals: Float32Array, stats: FitStats) {
    const mv = this.app.sync?.view<MeshView>(this.meshId);
    const mesh = this.app.doc.get<MeshEntity>(this.meshId)?.mesh;
    if (!mv || !mesh) return;
    const n = mesh.positions.length / 3;
    const values = new Float32Array(n).fill(NaN);
    for (let i = 0; i < verts.length; i++) values[verts[i]] = residuals[i];
    const range = Math.max(stats.maxAbs, 1e-3);
    const colors = deviationColors(values, range, Math.max(stats.rms * 0.5, 1e-4));
    // keep the unselected part neutral grey
    for (let v = 0; v < n; v++) if (!Number.isFinite(values[v])) colors.set([170, 178, 188], 3 * v);
    mv.updateMapColors(colors);
    this.app.ui.patch({ legend: { title: "Surface deviation", min: -range, max: range, tolerance: Math.max(stats.rms * 0.5, 1e-4), stats: [`RMS ${stats.rms.toFixed(4)} mm`, `Max ${stats.maxAbs.toFixed(4)} mm`] } });
  }

  private draw() {
    this.overlay.clear();
    const s = this.surface;
    if (!s) return;
    const m = tessellateSurface(s, Math.max(16, s.nu * 6), Math.max(16, s.nv * 6));
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(m.positions, 3));
    if (m.normals) g.setAttribute("normal", new THREE.BufferAttribute(m.normals, 3));
    else g.computeVertexNormals();
    g.setIndex(new THREE.BufferAttribute(m.indices, 1));
    const mesh = new THREE.Mesh(g, this.surfMat);
    mesh.renderOrder = 6;
    this.overlay.add(mesh);
    if (this.s.showCage) {
      const pts: number[] = [];
      const cp = s.controlPoints;
      for (let i = 0; i < s.nu; i++)
        for (let j = 0; j < s.nv; j++) {
          const p = cp[i * s.nv + j];
          if (j + 1 < s.nv) pts.push(...p, ...cp[i * s.nv + j + 1]);
          if (i + 1 < s.nu) pts.push(...p, ...cp[(i + 1) * s.nv + j]);
        }
      this.overlay.add(fatSegments(pts, this.cageMat));
    }
  }

  renderPanel() {
    const s = this.s;
    const upd = (p: Partial<FitSurfaceState>) => {
      this.set(p);
      this.schedule(60);
    };
    const planes = this.app.doc.ofKind("primitive").filter((p) => p.def.kind === "plane");
    return (
      <>
        <Section title="Surface">
          <Field label="Control pts U">
            <Slider value={s.nu} min={4} max={40} onChange={(nu) => upd({ nu })} />
          </Field>
          <Field label="Control pts V">
            <Slider value={s.nv} min={4} max={40} onChange={(nv) => upd({ nv })} />
          </Field>
          <Field label="Smoothness">
            <Slider value={s.smoothLog} min={-11} max={-4} step={0.25} onChange={(smoothLog) => upd({ smoothLog })} format={(v) => `1e${v.toFixed(1)}`} />
          </Field>
          <Field label="Extend">
            <Slider value={s.extend} min={0} max={50} onChange={(extend) => upd({ extend })} format={(v) => `${v}%`} />
          </Field>
          <Field label="Projection">
            <Select value={s.frameSource} options={[{ value: "auto", label: "Best-fit plane" }, ...planes.map((p) => ({ value: p.id, label: p.name }))]} onChange={(frameSource) => upd({ frameSource })} />
          </Field>
          <Check checked={s.showCage} onChange={(showCage) => { this.set({ showCage }); this.draw(); }}>
            Show control net
          </Check>
        </Section>
        <Section title="Deviation">
          {s.stats ? (
            <Stats
              rows={[
                ["Points", int(s.points)],
                ["RMS", mm(s.stats.rms, 4), s.stats.rms < 0.05 ? "good" : s.stats.rms < 0.2 ? "warn" : "bad"],
                ["Max", mm(s.stats.maxAbs, 4)],
              ]}
            />
          ) : (
            <div className="note">{s.error ?? "Select a region of the scan to fit."}</div>
          )}
          {s.stats && s.error && <div className="note err">{s.error}</div>}
        </Section>
      </>
    );
  }

  async commit() {
    if (!this.surface) {
      this.app.toast("error", "Select a region to fit first");
      return false;
    }
    const feature = { type: "fitSurface" as const, surface: this.surface, meshId: this.meshId, stats: this.s.stats ?? undefined };
    if (this.editId) {
      this.app.doc.transact("Edit surface", () => this.app.doc.update<BodyEntity>(this.editId!, { feature }));
      return;
    }
    const e = this.app.doc.add<BodyEntity>({ id: newId("b"), kind: "body", name: this.app.doc.uniqueName("Surface"), visible: true, feature });
    this.app.activeMeshRuntime()?.clearSelection();
    this.app.setActiveEntity(e.id);
  }

  deactivate() {
    window.clearTimeout(this.timer);
    if (this.editId) this.app.sync?.view<{ setVisible(v: boolean): void }>(this.editId)?.setVisible(this.app.doc.get(this.editId)?.visible ?? true);
    this.overlay.dispose();
    this.surfMat.dispose();
    this.cageMat.dispose();
    const mv = this.app.sync?.view<MeshView>(this.meshId);
    const e = this.app.doc.get<MeshEntity>(this.meshId);
    if (mv && e) {
      mv.setSelectionVisible(true);
      mv.setColorMode("solid", null);
      this.app.applyMeshDisplay(mv, e);
    }
    this.app.ui.patch({ legend: null });
  }
}

export function startFitSurface(app: App, editId?: string): void {
  app.startTool(new FitSurfaceTool(app, editId));
}
