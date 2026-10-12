/**
 * Application controller: owns the document, UI state, viewer, scene sync, active tool,
 * selection, the CAD kernel client and the compute workers. UI components call into it;
 * tools receive it in their constructor.
 */
import * as THREE from "three";
import { Document, newId } from "../core/document";
import type { BodyEntity, Entity, MeshEntity } from "../core/entities";
import { Store } from "../core/emitter";
import type { MeshData } from "../core/types";
import { Viewer } from "../viewer/Viewer";
import { setLineResolution } from "../viewer/draw";
import { LOD_THRESHOLD, type MeshView } from "../viewer/MeshView";
import type { KernelClient } from "../cad/KernelClient";
import { SceneSync } from "./SceneSync";
import { SelectionController } from "./selection";
import { initialUiState, type Toast, type UiState } from "./ui-state";
import { FeatureEvaluator } from "./features";
import { prepareMesh, markPrepared } from "./MeshRuntime";
import type { Tool, ToolPointerEvent } from "../tools/Tool";
import { compute } from "./compute";
import { transformEntityPatch } from "./transform";
import { lazy } from "./lazy";

let toastId = 1;

export class App {
  readonly doc = new Document();
  readonly ui = new Store<UiState>(initialUiState);
  viewer: Viewer | null = null;
  sync: SceneSync | null = null;
  readonly selection = new SelectionController(this);
  readonly features = new FeatureEvaluator(this);
  private tool: Tool<object> | null = null;
  readonly toolStore = new Store<Tool<object> | null>(null);
  private kernelPromise: Promise<KernelClient> | null = null;
  /** body id -> last evaluation error */
  readonly bodyErrors = new Store<Record<string, string>>({});
  private proxyJobs = new Set<string>();

  constructor() {
    this.doc.on("changed", ({ ids }) => {
      this.tool?.onDocumentChanged(ids);
      const ui = this.ui.get();
      if (ui.activeEntityId && !this.doc.has(ui.activeEntityId)) this.ui.patch({ activeEntityId: null });
      if (ui.activeMeshId && !this.doc.has(ui.activeMeshId)) this.ui.patch({ activeMeshId: null, selectedTriangles: 0 });
      if (!this.ui.get().activeMeshId) {
        const m = this.doc.ofKind("mesh")[0];
        if (m) this.ui.patch({ activeMeshId: m.id });
      }
      this.onMeshSelectionChanged(this.ui.get().activeMeshId ?? "");
    });
  }

  // -------------------------------------------------------------------------
  // Viewer lifecycle
  // -------------------------------------------------------------------------

  attachViewer(container: HTMLElement): Viewer {
    const viewer = new Viewer(container);
    this.viewer = viewer;
    viewer.on("resize", ({ width, height }) => setLineResolution(width, height));
    setLineResolution(viewer.size.width, viewer.size.height);
    viewer.showGrid = this.ui.get().showGrid;
    this.sync = new SceneSync(this);
    this.sync.syncAll();
    viewer.fitAll("iso");
    return viewer;
  }

  detachViewer(): void {
    this.sync?.dispose();
    this.viewer?.dispose();
    this.viewer = null;
    this.sync = null;
  }

  // -------------------------------------------------------------------------
  // Notifications & busy state
  // -------------------------------------------------------------------------

  toast(kind: Toast["kind"], text: string, ms = kind === "error" ? 7000 : 3500): void {
    const t: Toast = { id: toastId++, kind, text };
    this.ui.patch({ toasts: [...this.ui.get().toasts, t] });
    window.setTimeout(() => this.ui.patch({ toasts: this.ui.get().toasts.filter((x) => x.id !== t.id) }), ms);
  }

  setStatus(text: string): void {
    this.ui.patch({ status: text });
  }

  setProgress(fraction: number, message?: string): void {
    const b = this.ui.get().busy;
    if (b) this.ui.patch({ busy: { message: message ?? b.message, progress: fraction } });
  }

  /** Show a blocking busy overlay while `fn` runs; errors become toasts. */
  async busy<T>(message: string, fn: () => Promise<T> | T): Promise<T | undefined> {
    this.ui.patch({ busy: { message } });
    // let the overlay paint before heavy synchronous work starts (rAF is paused in hidden windows,
    // so fall back to a timer)
    await new Promise<void>((resolve) => {
      let done = false;
      const go = () => {
        if (done) return;
        done = true;
        setTimeout(resolve, 0);
      };
      requestAnimationFrame(go);
      setTimeout(go, 80);
    });
    try {
      return await fn();
    } catch (err) {
      console.error(err);
      this.toast("error", err instanceof Error ? err.message : String(err));
      return undefined;
    } finally {
      this.ui.patch({ busy: null });
    }
  }

  // -------------------------------------------------------------------------
  // Kernel
  // -------------------------------------------------------------------------

  getKernel(): Promise<KernelClient> {
    if (!this.kernelPromise) {
      this.kernelPromise = lazy<typeof import("../cad/KernelClient")>("../cad/KernelClient.ts").then(async ({ KernelClient }) => {
        const k = new KernelClient();
        await k.ready();
        return k;
      });
      this.kernelPromise.catch((e) => {
        this.kernelPromise = null;
        this.toast("error", `CAD kernel failed to start: ${e instanceof Error ? e.message : e}`);
      });
    }
    return this.kernelPromise;
  }

  onBodyEvaluated(id: string, error: string | null): void {
    const cur = { ...this.bodyErrors.get() };
    if (error) {
      cur[id] = error;
      const name = this.doc.get(id)?.name ?? "Body";
      this.toast("error", `${name}: ${error}`);
    } else delete cur[id];
    this.bodyErrors.set(cur);
  }

  // -------------------------------------------------------------------------
  // Entities
  // -------------------------------------------------------------------------

  setActiveEntity(id: string | null): void {
    const e = this.doc.get(id);
    const patch: Partial<UiState> = { activeEntityId: id };
    if (e?.kind === "mesh") patch.activeMeshId = e.id;
    this.ui.patch(patch);
    this.sync?.setHighlight(e && e.kind !== "mesh" ? e.id : null);
  }

  /** Add a mesh entity (prepares buffers for display/interaction). */
  addMesh(name: string, mesh: MeshData, source?: string): MeshEntity {
    const prepared = prepareMesh(mesh);
    const e = this.doc.add<MeshEntity>({
      id: newId("m"),
      kind: "mesh",
      name: this.doc.uniqueName(name),
      visible: true,
      mesh: prepared,
      source,
    });
    this.ui.patch({ activeMeshId: e.id });
    return e;
  }

  /** Replace a mesh's geometry (undoable). */
  replaceMesh(id: string, mesh: MeshData, label: string, alreadyPrepared = false): void {
    const prepared = alreadyPrepared ? (markPrepared(mesh), mesh) : prepareMesh(mesh);
    this.doc.transact(label, () => this.doc.update<MeshEntity>(id, { mesh: prepared }));
  }

  /** Apply a rigid transform to entities (undoable as one step). */
  transformEntities(ids: string[], m: number[], label: string): void {
    this.doc.transact(label, () => {
      for (const id of ids) {
        const e = this.doc.get(id);
        if (!e) continue;
        const patch = transformEntityPatch(m, e);
        if (!patch) continue;
        if (e.kind === "mesh") (patch as Partial<MeshEntity>).mesh = prepareMesh((patch as MeshEntity).mesh, { keepOrder: true });
        this.doc.update(id, patch);
      }
    });
  }

  deleteEntities(ids: string[]): void {
    if (!ids.length) return;
    this.doc.transact(ids.length === 1 ? `Delete ${this.doc.get(ids[0])?.name}` : `Delete ${ids.length} items`, () => {
      for (const id of ids) this.doc.remove(id);
    });
  }

  setVisible(id: string, visible: boolean): void {
    this.doc.transact(visible ? "Show" : "Hide", () => this.doc.update(id, { visible }));
  }

  rename(id: string, name: string): void {
    this.doc.transact("Rename", () => this.doc.update(id, { name }));
  }

  /** The mesh that receives selections/operations: active mesh, or the only visible one. */
  activeMesh(): MeshEntity | undefined {
    const id = this.ui.get().activeMeshId;
    const e = this.doc.get<MeshEntity>(id);
    if (e?.kind === "mesh") return e;
    return this.doc.ofKind("mesh").find((m) => m.visible);
  }

  activeMeshRuntime() {
    const m = this.activeMesh();
    return m ? this.sync?.meshRuntime(m.id) : undefined;
  }

  onMeshSelectionChanged(meshId: string): void {
    const rt = this.sync?.meshRuntime(this.ui.get().activeMeshId);
    const count = rt?.selectionCount ?? 0;
    if (this.ui.get().selectedTriangles !== count) this.ui.patch({ selectedTriangles: count });
    if (meshId) this.tool?.onSelectionChanged();
  }

  applyMeshDisplay(view: MeshView, e: MeshEntity): void {
    const mode = this.ui.get().meshDisplay;
    view.setWireframe(mode === "wireframe" && e.mesh.indices.length / 3 < 2_000_000);
    if (view.colorMode !== "map") view.setColorMode(mode === "scan-colors" && e.mesh.colors ? "scan" : "solid");
  }

  setMeshDisplay(mode: UiState["meshDisplay"]): void {
    this.ui.patch({ meshDisplay: mode });
    for (const m of this.doc.ofKind("mesh")) {
      const v = this.sync?.view<MeshView>(m.id);
      if (v) this.applyMeshDisplay(v, m);
    }
  }

  /** Build a decimated LOD proxy for very large meshes (in a worker). */
  ensureMeshProxy(id: string): void {
    const e = this.doc.get<MeshEntity>(id);
    if (!e || e.mesh.indices.length / 3 < LOD_THRESHOLD || this.proxyJobs.has(id)) return;
    const mesh = e.mesh;
    this.proxyJobs.add(id);
    compute
      .run<MeshData>("decimate", { mesh, targetTriangles: 1_000_000, fast: true })
      .then((proxy) => {
        const rt = this.sync?.meshRuntime(id);
        if (!rt || rt.mesh !== mesh) return;
        rt.proxy = proxy;
        rt.view?.setProxy(proxy);
      })
      .catch((err) => console.warn("LOD proxy failed", err))
      .finally(() => this.proxyJobs.delete(id));
  }

  // -------------------------------------------------------------------------
  // Tools
  // -------------------------------------------------------------------------

  get activeTool(): Tool<object> | null {
    return this.tool;
  }

  startTool(tool: Tool<object>): void {
    if (this.tool) this.endTool(false);
    this.tool = tool;
    this.toolStore.set(tool);
    this.ui.patch({ toolId: tool.id, status: tool.hint || this.ui.get().status });
    try {
      tool.activate();
    } catch (err) {
      this.toast("error", err instanceof Error ? err.message : String(err));
      this.endTool(false);
    }
  }

  /** Close the active tool; `commit` runs its commit() first. */
  async endTool(commit: boolean): Promise<void> {
    const t = this.tool;
    if (!t) return;
    if (commit) {
      try {
        const ok = await t.commit();
        if (ok === false) return;
      } catch (err) {
        console.error(err);
        this.toast("error", err instanceof Error ? err.message : String(err));
        return;
      }
    } else {
      t.cancel();
    }
    if (this.tool !== t) return;
    t.deactivate();
    this.tool = null;
    this.toolStore.set(null);
    this.ui.patch({ toolId: null, status: "Ready" });
    this.viewer?.requestRender();
  }

  // -------------------------------------------------------------------------
  // Pointer & keyboard routing (called by the Viewport component)
  // -------------------------------------------------------------------------

  makePointerEvent(native: PointerEvent): ToolPointerEvent {
    const viewer = this.viewer!;
    const local = viewer.clientToLocal(native.clientX, native.clientY);
    const ndc = viewer.clientToNdc(native.clientX, native.clientY);
    const ray = viewer.rayFromNdc(ndc);
    const cache = new Map<unknown, ReturnType<Viewer["pick"]>>();
    return {
      native,
      button: native.button,
      shift: native.shiftKey,
      ctrl: native.ctrlKey || native.metaKey,
      alt: native.altKey,
      x: local.x,
      y: local.y,
      ndc,
      ray,
      pick: (filter) => {
        if (!cache.has(filter)) cache.set(filter, viewer.pickRay(ray, filter));
        return cache.get(filter) ?? null;
      },
    };
  }

  handlePointer(type: "down" | "move" | "up", native: PointerEvent): void {
    if (!this.viewer) return;
    const e = this.makePointerEvent(native);
    const t = this.tool;
    let used = false;
    if (t) {
      used = type === "down" ? t.onPointerDown(e) : type === "move" ? t.onPointerMove(e) : t.onPointerUp(e);
    }
    if (!used && (!t || t.allowMeshSelection)) {
      if (type === "down") this.selection.onPointerDown(e);
      else if (type === "move") this.selection.onPointerMove(e);
      else this.selection.onPointerUp(e);
    }
    if (type === "move" && e.button <= 0 && !native.buttons) this.updateCursor(e);
  }

  private cursorFrame = 0;
  private updateCursor(e: ToolPointerEvent): void {
    // throttle picking for the status-bar coordinate readout
    if (this.cursorFrame) return;
    this.cursorFrame = requestAnimationFrame(() => {
      this.cursorFrame = 0;
      const hit = e.pick();
      this.ui.patch({ cursor: hit ? hit.point : null });
    });
  }

  handleKey(e: KeyboardEvent): boolean {
    if (this.tool?.onKeyDown(e)) return true;
    const ctrl = e.ctrlKey || e.metaKey;
    if (ctrl && e.key.toLowerCase() === "z" && !e.shiftKey) {
      this.doc.undo();
      return true;
    }
    if (ctrl && (e.key.toLowerCase() === "y" || (e.key.toLowerCase() === "z" && e.shiftKey))) {
      this.doc.redo();
      return true;
    }
    if (e.key === "Escape") {
      if (this.tool) void this.endTool(false);
      else this.activeMeshRuntime()?.clearSelection();
      return true;
    }
    if (e.key === "Enter" && this.tool) {
      void this.endTool(true);
      return true;
    }
    if (e.key === "Delete") {
      const active = this.ui.get().activeEntityId;
      const rt = this.activeMeshRuntime();
      if (rt && rt.selectionCount > 0) {
        void import("../tools/meshCommands").then((m) => m.deleteSelectedTriangles(this));
      } else if (active) this.deleteEntities([active]);
      return true;
    }
    if (!ctrl && !e.altKey) {
      const k = e.key.toLowerCase();
      if (k === "f") {
        this.viewer?.fitAll();
        return true;
      }
      const modes: Record<string, UiState["selectionMode"]> = { "1": "pick", "2": "brush", "3": "rect", "4": "lasso", "5": "smart" };
      if (modes[k]) {
        this.ui.patch({ selectionMode: modes[k] });
        return true;
      }
      if (k === "[" || k === "]") {
        const r = this.ui.get().brushRadius;
        this.ui.patch({ brushRadius: Math.max(4, Math.min(300, Math.round(k === "]" ? r * 1.2 : r / 1.2))) });
        return true;
      }
    }
    return false;
  }

  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------

  /** World-space size of the scene (used for default sizes of planes, offsets, ...). */
  sceneSize(): number {
    const box = this.viewer?.sceneBounds(false);
    if (!box || box.isEmpty()) return 100;
    return box.getSize(new THREE.Vector3()).length();
  }

  entityBodies(): BodyEntity[] {
    return this.doc.ofKind("body");
  }

  allEntities(): Entity[] {
    return this.doc.all();
  }
}

export const app = new App();
(globalThis as { resurface?: App }).resurface = app;
