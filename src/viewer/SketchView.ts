/** Displays a 2D sketch entity: its section reference curves and sketch geometry in 3D. */
import * as THREE from "three";
import type { SketchEntity as SketchDocEntity } from "../core/entities";
import type { Vec3 } from "../core/types";
import { frameToWorld } from "../core/vec";
import { entityToSegments, indexSketch } from "../sketch/model";
import type { PickTarget, Viewer } from "./Viewer";
import type { EntityView } from "./views";
import { COLORS } from "./views";
import { fatSegments, lineMaterial, pointSprites, polylinesToPairs, sampleSegment2D } from "./draw";

export const SECTION_COLOR = "#ffb347";
export const CONSTRUCTION_COLOR = "#7d8a9c";

export class SketchView implements EntityView {
  readonly group = new THREE.Group();
  private geomMat = lineMaterial({ color: COLORS.sketch, width: 2.6 });
  private consMat = lineMaterial({ color: CONSTRUCTION_COLOR, width: 1.4, dashed: true, dashSize: 2 });
  private sectionMat = lineMaterial({ color: SECTION_COLOR, width: 1.6 });
  private removePick: () => void;
  private highlighted = false;
  /** while the sketch is being edited the tool draws it; the static view hides itself */
  editing = false;

  constructor(private viewer: Viewer, private entity: SketchDocEntity) {
    viewer.layers.sketches.add(this.group);
    const rc = new THREE.Raycaster();
    const target: PickTarget = {
      entityId: entity.id,
      priority: 2,
      enabled: () => this.group.visible && !this.editing,
      raycast: (ray, ctx) => {
        rc.ray.copy(ray);
        rc.camera = ctx.camera;
        (rc.params as { Line2?: { threshold: number } }).Line2 = { threshold: 6 };
        const hits = rc.intersectObjects(this.group.children.filter((c) => !c.userData.noPick), false);
        if (!hits.length) return null;
        const h = hits[0];
        return { entityId: entity.id, point: [h.point.x, h.point.y, h.point.z], distance: h.distance };
      },
    };
    this.removePick = viewer.addPickTarget(target);
    this.update(entity);
  }

  update(e: SketchDocEntity): void {
    this.entity = e;
    for (const c of [...this.group.children]) {
      this.group.remove(c);
      (c as THREE.Mesh).geometry?.dispose();
      if (c instanceof THREE.Points) (c.material as THREE.Material).dispose();
    }
    const f = e.frame;
    // section reference curves
    if (e.sections.length) {
      const pls = e.sections.map((s) => ({ points: s.points.map((p) => frameToWorld(f, p)), closed: s.closed }));
      const sec = fatSegments(polylinesToPairs(pls), this.sectionMat);
      this.group.add(sec);
    }
    // sketch geometry
    const sk = e.sketch;
    if (sk && sk.entities.length) {
      const idx = indexSketch(sk).points;
      const solid: { points: Vec3[] }[] = [];
      const cons: { points: Vec3[] }[] = [];
      for (const ent of sk.entities) {
        let segs;
        try {
          segs = entityToSegments(sk, ent, idx);
        } catch {
          continue;
        }
        for (const s of segs) {
          const pts = sampleSegment2D(s).map((p) => frameToWorld(f, p));
          (ent.construction ? cons : solid).push({ points: pts });
        }
      }
      if (solid.length) this.group.add(fatSegments(polylinesToPairs(solid), this.geomMat));
      if (cons.length) this.group.add(fatSegments(polylinesToPairs(cons), this.consMat));
      const ptPos: number[] = [];
      for (const ent of sk.entities) {
        if (ent.type === "line") for (const id of [ent.p1, ent.p2]) push(id);
        else if (ent.type === "arc") for (const id of [ent.start, ent.end]) push(id);
      }
      function push(id: string) {
        const p = idx.get(id);
        if (p) ptPos.push(...frameToWorld(f, [p.x, p.y]));
      }
      if (ptPos.length) {
        const pts = pointSprites(ptPos, COLORS.sketch, 5);
        pts.userData.noPick = true;
        this.group.add(pts);
      }
    }
    this.group.visible = e.visible && !this.editing;
    this.applyHighlight();
    this.viewer.markSceneChanged();
  }

  setEditing(on: boolean): void {
    this.editing = on;
    this.group.visible = this.entity.visible && !on;
    this.viewer.requestRender();
  }

  private applyHighlight() {
    this.geomMat.color.set(this.highlighted ? COLORS.highlight : COLORS.sketch);
    this.geomMat.linewidth = this.highlighted ? 3.4 : 2.6;
  }

  setHighlight(on: boolean): void {
    this.highlighted = on;
    this.applyHighlight();
    this.viewer.requestRender();
  }

  setVisible(on: boolean): void {
    this.group.visible = on && !this.editing;
    this.viewer.markSceneChanged();
  }

  dispose(): void {
    this.removePick();
    this.viewer.layers.sketches.remove(this.group);
    for (const c of this.group.children) (c as THREE.Mesh).geometry?.dispose();
    this.geomMat.dispose();
    this.consMat.dispose();
    this.sectionMat.dispose();
    this.viewer.markSceneChanged();
  }
}
