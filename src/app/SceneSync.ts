/**
 * Keeps the three.js scene in sync with the document: creates, updates and disposes entity
 * views, maintains mesh runtimes, and re-evaluates CAD bodies (asynchronously, through the
 * OCCT kernel worker) when their feature or any dependency changes.
 */
import type { BodyEntity, Entity, MeshEntity } from "../core/entities";
import { MeshView } from "../viewer/MeshView";
import { BodyView, PointsView, PrimitiveView, RefLineView, RefPointView, Sketch3DView, type EntityView } from "../viewer/views";
import { SketchView } from "../viewer/SketchView";
import { MeshRuntime } from "./MeshRuntime";
import type { App } from "./App";
import { featureDependencies } from "./features";

interface Slot {
  entity: Entity;
  view: EntityView | MeshView;
}

const featureIds = new WeakMap<object, number>();
let nextFeatureId = 1;
/** identity-based id for immutable objects (sketch model / frame) */
function sketchIds(o: object): number {
  let id = featureIds.get(o);
  if (id === undefined) {
    id = nextFeatureId++;
    featureIds.set(o, id);
  }
  return id;
}

export class SceneSync {
  private slots = new Map<string, Slot>();
  readonly meshRuntimes = new Map<string, MeshRuntime>();
  private highlighted: string | null = null;
  /** body id -> evaluation token to drop stale async results */
  private bodyTokens = new Map<string, number>();
  private bodyDeps = new Map<string, string>();

  private unsubscribe: () => void;

  constructor(private app: App) {
    this.unsubscribe = app.doc.on("changed", ({ ids }) => this.sync(ids));
  }

  /** Detach from the document and dispose all views (viewer is being destroyed). */
  dispose(): void {
    this.unsubscribe();
    for (const [id, slot] of [...this.slots]) {
      slot.view.dispose();
      this.slots.delete(id);
    }
    for (const rt of this.meshRuntimes.values()) rt.dispose();
    this.meshRuntimes.clear();
  }

  /** Full sync of all entities (initial mount / project load). */
  syncAll(): void {
    const ids = new Set<string>([...this.slots.keys(), ...this.app.doc.all().map((e) => e.id)]);
    this.sync(ids);
  }

  meshRuntime(id: string | null | undefined): MeshRuntime | undefined {
    return id ? this.meshRuntimes.get(id) : undefined;
  }

  view<T = EntityView>(id: string): T | undefined {
    return this.slots.get(id)?.view as T | undefined;
  }

  private sync(ids: Set<string>): void {
    const viewer = this.app.viewer;
    if (!viewer) return;
    for (const id of ids) {
      const e = this.app.doc.get(id);
      const slot = this.slots.get(id);
      if (!e) {
        if (slot) this.disposeSlot(id, slot);
        continue;
      }
      if (slot && slot.entity === e) continue;
      if (slot && this.canUpdateInPlace(slot.entity, e)) {
        if (e.kind === "mesh") {
          const mv = slot.view as MeshView;
          mv.setVisible(e.visible);
          if (e.color) mv.setColor(e.color);
        } else {
          (slot.view as EntityView).update(e);
          (slot.view as EntityView).setVisible(e.visible);
        }
        slot.entity = e;
      } else {
        if (slot) this.disposeSlot(id, slot);
        this.createSlot(e);
      }
    }
    // Re-evaluate bodies whose feature or dependencies changed.
    for (const body of this.app.doc.ofKind("body")) this.maybeEvaluateBody(body, ids);
    if (this.highlighted && !this.slots.has(this.highlighted)) this.highlighted = null;
    viewer.updateGrid();
    viewer.requestRender();
  }

  private canUpdateInPlace(prev: Entity, next: Entity): boolean {
    if (prev.kind !== next.kind) return false;
    if (prev.kind === "mesh") return (prev as MeshEntity).mesh === (next as MeshEntity).mesh;
    return true;
  }

  private createSlot(e: Entity): void {
    const viewer = this.app.viewer!;
    let view: EntityView | MeshView;
    switch (e.kind) {
      case "mesh": {
        let rt = this.meshRuntimes.get(e.id);
        if (!rt || rt.mesh !== e.mesh) {
          rt?.dispose();
          rt = new MeshRuntime(e.id, e.mesh);
          rt.onSelectionChanged = () => this.app.onMeshSelectionChanged(e.id);
          this.meshRuntimes.set(e.id, rt);
        }
        const mv = new MeshView(viewer, e.id, e.mesh, rt.chunks, { color: e.color, raycast: rt.raycast });
        rt.view = mv;
        if (rt.selectionCount) mv.setSelection(rt.selection);
        if (rt.proxy) mv.setProxy(rt.proxy);
        mv.setVisible(e.visible);
        this.app.applyMeshDisplay(mv, e);
        view = mv;
        this.app.ensureMeshProxy(e.id);
        break;
      }
      case "points":
        view = new PointsView(viewer, e);
        break;
      case "primitive":
        view = new PrimitiveView(viewer, e);
        break;
      case "refline":
        view = new RefLineView(viewer, e);
        break;
      case "refpoint":
        view = new RefPointView(viewer, e);
        break;
      case "sketch":
        view = new SketchView(viewer, e);
        break;
      case "sketch3d":
        view = new Sketch3DView(viewer, e);
        break;
      case "body":
        view = new BodyView(viewer, e);
        this.bodyDeps.delete(e.id);
        break;
    }
    this.slots.set(e.id, { entity: e, view });
    if (this.highlighted === e.id) (view as EntityView).setHighlight?.(true);
  }

  private disposeSlot(id: string, slot: Slot): void {
    slot.view.dispose();
    this.slots.delete(id);
    if (slot.entity.kind === "mesh") {
      const rt = this.meshRuntimes.get(id);
      const stillExists = this.app.doc.get<MeshEntity>(id);
      if (rt && (!stillExists || stillExists.mesh !== rt.mesh)) {
        rt.dispose();
        this.meshRuntimes.delete(id);
      }
    }
    if (slot.entity.kind === "body") {
      this.bodyTokens.delete(id);
      this.bodyDeps.delete(id);
      this.app.features.release(id);
    }
  }

  setHighlight(id: string | null): void {
    if (this.highlighted === id) return;
    if (this.highlighted) (this.slots.get(this.highlighted)?.view as EntityView | undefined)?.setHighlight?.(false);
    this.highlighted = id;
    if (id) (this.slots.get(id)?.view as EntityView | undefined)?.setHighlight?.(true);
    this.app.viewer?.requestRender();
  }

  /** Signature of a body's inputs: its own revision plus the revisions of what it depends on. */
  private bodySignature(body: BodyEntity): string {
    // Feature objects are immutable, so object identity tells us whether the definition changed
    // (visibility/colour edits keep the same feature object and do not trigger re-evaluation).
    let fid = featureIds.get(body.feature);
    if (fid === undefined) {
      fid = nextFeatureId++;
      featureIds.set(body.feature, fid);
    }
    const deps = featureDependencies(body.feature);
    return [String(fid), ...deps.map((d) => `${d}:${this.depSignature(d)}`)].join("|");
  }

  /** Revision of a dependency, ignoring pure display changes for bodies (feature identity). */
  private depSignature(id: string): string {
    const e = this.app.doc.get(id);
    if (!e) return "x";
    if (e.kind === "body") return this.bodySignature(e);
    if (e.kind === "sketch") return String(sketchIds(e.sketch) + ":" + sketchIds(e.frame));
    return String(e.rev);
  }

  private maybeEvaluateBody(body: BodyEntity, _changed: Set<string>): void {
    const sig = this.bodySignature(body);
    if (this.bodyDeps.get(body.id) === sig) return;
    this.bodyDeps.set(body.id, sig);
    const token = (this.bodyTokens.get(body.id) ?? 0) + 1;
    this.bodyTokens.set(body.id, token);
    this.app.features
      .evaluate(body)
      .then((tess) => {
        if (this.bodyTokens.get(body.id) !== token) return;
        const v = this.slots.get(body.id)?.view as BodyView | undefined;
        v?.setTessellation(tess);
        this.app.onBodyEvaluated(body.id, null);
      })
      .catch((err: unknown) => {
        if (this.bodyTokens.get(body.id) !== token) return;
        const v = this.slots.get(body.id)?.view as BodyView | undefined;
        v?.setTessellation(null);
        this.app.onBodyEvaluated(body.id, err instanceof Error ? err.message : String(err));
      });
  }
}
