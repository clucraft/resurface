/** Views for reference geometry, point clouds, 3D sketches and CAD bodies. */
import * as THREE from "three";
import { MeshBVH, acceleratedRaycast } from "three-mesh-bvh";
import type {
  BodyEntity,
  PointsEntity,
  PrimitiveEntity,
  RefLineEntity,
  RefPointEntity,
  Sketch3DEntity,
} from "../core/entities";
import type { Vec3 } from "../core/types";
import { anyPerpendicular, cross, normalize } from "../core/vec";
import type { PickHit, PickTarget, Viewer } from "./Viewer";
import { disposeObject, fatSegments, lineMaterial, pointCloudObject, pointSprites, polylinesToPairs, translucentMaterial } from "./draw";

export interface EntityView {
  update(entity: unknown): void;
  setHighlight(on: boolean): void;
  setVisible(on: boolean): void;
  dispose(): void;
}

export const COLORS = {
  primitive: "#3d8bfd",
  primitiveHi: "#7fb2ff",
  symmetry: "#b46cff",
  refline: "#00c2a8",
  refpoint: "#ffd23f",
  body: "#9fb7d6",
  bodyEdge: "#1a2230",
  sketch: "#2ee6a6",
  sketch3d: "#ff4fa3",
  highlight: "#ffb020",
};

/** Raycast a generic three object (small geometry) and turn the result into a PickHit. */
function objectPick(entityId: string, root: THREE.Object3D, priority = 0, sub?: (o: THREE.Intersection) => string | number | undefined): PickTarget {
  const rc = new THREE.Raycaster();
  return {
    entityId,
    priority,
    enabled: () => root.visible && !!root.parent,
    raycast(ray, ctx) {
      rc.ray.copy(ray);
      rc.camera = ctx.camera;
      const px = ctx.pixelSize(ray.origin.clone().addScaledVector(ray.direction, 100));
      rc.params.Line = { threshold: Math.max(px * 6, 1e-3) };
      rc.params.Points = { threshold: Math.max(px * 6, 1e-3) };
      (rc.params as { Line2?: { threshold: number } }).Line2 = { threshold: 8 };
      const hits = rc.intersectObject(root, true).filter((h) => h.object.visible && !h.object.userData.noPick);
      if (!hits.length) return null;
      const h = hits[0];
      const n = h.face ? h.face.normal.clone().transformDirection(h.object.matrixWorld) : undefined;
      return {
        entityId,
        point: [h.point.x, h.point.y, h.point.z],
        normal: n ? [n.x, n.y, n.z] : undefined,
        distance: h.distance,
        faceIndex: h.faceIndex ?? undefined,
        sub: sub?.(h),
      } satisfies PickHit;
    },
  };
}

function orientToAxis(obj: THREE.Object3D, axis: Vec3): void {
  obj.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(...normalize(axis)));
}

// ---------------------------------------------------------------------------
// Primitives (plane / cylinder / cone / sphere)
// ---------------------------------------------------------------------------

export class PrimitiveView implements EntityView {
  readonly group = new THREE.Group();
  private removePick: () => void;
  private fillMat: THREE.MeshPhongMaterial;
  private edgeMat = lineMaterial({ color: COLORS.primitive, width: 1.5 });
  private highlighted = false;
  private baseColor = COLORS.primitive;

  constructor(private viewer: Viewer, private entity: PrimitiveEntity) {
    this.fillMat = translucentMaterial(COLORS.primitive, 0.28);
    viewer.layers.references.add(this.group);
    this.removePick = viewer.addPickTarget(objectPick(entity.id, this.group, 0.5));
    this.update(entity);
  }

  update(e: PrimitiveEntity): void {
    this.entity = e;
    for (const c of [...this.group.children]) {
      this.group.remove(c);
      (c as THREE.Mesh).geometry?.dispose();
    }
    this.baseColor = e.color ?? (e.symmetry ? COLORS.symmetry : COLORS.primitive);
    this.applyColor();
    const def = e.def;
    const edges: { points: Vec3[]; closed?: boolean }[] = [];
    if (def.kind === "plane") {
      const s = e.displaySize ?? 50;
      const g = new THREE.PlaneGeometry(2 * s, 2 * s);
      const m = new THREE.Mesh(g, this.fillMat);
      const n = new THREE.Vector3(...def.normal);
      m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), n);
      m.position.set(...def.origin);
      this.group.add(m);
      const u = anyPerpendicular(def.normal);
      const v = cross(def.normal, u);
      const corner = (a: number, b: number): Vec3 => [
        def.origin[0] + s * (a * u[0] + b * v[0]),
        def.origin[1] + s * (a * u[1] + b * v[1]),
        def.origin[2] + s * (a * u[2] + b * v[2]),
      ];
      edges.push({ points: [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)], closed: true });
      // normal tick
      const tip: Vec3 = [def.origin[0] + def.normal[0] * s * 0.25, def.origin[1] + def.normal[1] * s * 0.25, def.origin[2] + def.normal[2] * s * 0.25];
      edges.push({ points: [def.origin, tip] });
    } else if (def.kind === "cylinder" || def.kind === "cone") {
      const axis = normalize(def.axis);
      const base = def.kind === "cylinder" ? def.origin : def.apex;
      const h0 = def.hMin, h1 = Math.max(def.hMax, def.hMin + 1e-3);
      const r0 = def.kind === "cylinder" ? def.radius : h0 * Math.tan(def.halfAngle);
      const r1 = def.kind === "cylinder" ? def.radius : h1 * Math.tan(def.halfAngle);
      const g = new THREE.CylinderGeometry(r1, r0, h1 - h0, 64, 1, true);
      const m = new THREE.Mesh(g, this.fillMat);
      orientToAxis(m, axis);
      const mid = (h0 + h1) / 2;
      m.position.set(base[0] + axis[0] * mid, base[1] + axis[1] * mid, base[2] + axis[2] * mid);
      this.group.add(m);
      const u = anyPerpendicular(axis), v = cross(axis, u);
      const ring = (h: number, r: number) => {
        const pts: Vec3[] = [];
        for (let i = 0; i < 64; i++) {
          const a = (i / 64) * Math.PI * 2, c = Math.cos(a) * r, s = Math.sin(a) * r;
          pts.push([base[0] + axis[0] * h + u[0] * c + v[0] * s, base[1] + axis[1] * h + u[1] * c + v[1] * s, base[2] + axis[2] * h + u[2] * c + v[2] * s]);
        }
        return pts;
      };
      edges.push({ points: ring(h0, r0), closed: true }, { points: ring(h1, r1), closed: true });
      const ext = (h1 - h0) * 0.15;
      edges.push({
        points: [
          [base[0] + axis[0] * (h0 - ext), base[1] + axis[1] * (h0 - ext), base[2] + axis[2] * (h0 - ext)],
          [base[0] + axis[0] * (h1 + ext), base[1] + axis[1] * (h1 + ext), base[2] + axis[2] * (h1 + ext)],
        ],
      });
    } else if (def.kind === "sphere") {
      const g = new THREE.SphereGeometry(def.radius, 48, 32);
      const m = new THREE.Mesh(g, this.fillMat);
      m.position.set(...def.center);
      this.group.add(m);
      for (const ax of [[1, 0, 0], [0, 1, 0], [0, 0, 1]] as Vec3[]) {
        const u = anyPerpendicular(ax), v = cross(ax, u);
        const pts: Vec3[] = [];
        for (let i = 0; i < 72; i++) {
          const a = (i / 72) * Math.PI * 2;
          pts.push([
            def.center[0] + def.radius * (Math.cos(a) * u[0] + Math.sin(a) * v[0]),
            def.center[1] + def.radius * (Math.cos(a) * u[1] + Math.sin(a) * v[1]),
            def.center[2] + def.radius * (Math.cos(a) * u[2] + Math.sin(a) * v[2]),
          ]);
        }
        edges.push({ points: pts, closed: true });
      }
    }
    const lines = fatSegments(polylinesToPairs(edges), this.edgeMat);
    lines.userData.noPick = true;
    this.group.add(lines);
    this.group.visible = e.visible;
    this.viewer.markSceneChanged();
  }

  private applyColor() {
    const c = this.highlighted ? COLORS.highlight : this.baseColor;
    this.fillMat.color.set(c);
    this.fillMat.opacity = this.highlighted ? 0.42 : 0.28;
    this.edgeMat.color.set(c);
  }

  setHighlight(on: boolean): void {
    this.highlighted = on;
    this.applyColor();
    this.viewer.requestRender();
  }

  setVisible(on: boolean): void {
    this.group.visible = on;
    this.viewer.markSceneChanged();
  }

  dispose(): void {
    this.removePick();
    this.viewer.layers.references.remove(this.group);
    disposeObject(this.group);
    this.fillMat.dispose();
    this.edgeMat.dispose();
    this.viewer.markSceneChanged();
  }
}

// ---------------------------------------------------------------------------
// Reference line & point
// ---------------------------------------------------------------------------

export class RefLineView implements EntityView {
  readonly group = new THREE.Group();
  private mat = lineMaterial({ color: COLORS.refline, width: 2.5, dashed: true, dashSize: 4 });
  private removePick: () => void;
  constructor(private viewer: Viewer, e: RefLineEntity) {
    viewer.layers.references.add(this.group);
    this.removePick = viewer.addPickTarget(objectPick(e.id, this.group, 2));
    this.update(e);
  }
  update(e: RefLineEntity): void {
    this.group.clear();
    const { origin: o, direction: d } = e.def;
    const a: Vec3 = [o[0] + d[0] * e.tMin, o[1] + d[1] * e.tMin, o[2] + d[2] * e.tMin];
    const b: Vec3 = [o[0] + d[0] * e.tMax, o[1] + d[1] * e.tMax, o[2] + d[2] * e.tMax];
    this.mat.color.set(e.color ?? COLORS.refline);
    this.group.add(fatSegments([...a, ...b], this.mat));
    this.group.visible = e.visible;
    this.viewer.markSceneChanged();
  }
  setHighlight(on: boolean): void {
    this.mat.color.set(on ? COLORS.highlight : COLORS.refline);
    this.mat.linewidth = on ? 4 : 2.5;
    this.viewer.requestRender();
  }
  setVisible(on: boolean): void {
    this.group.visible = on;
    this.viewer.markSceneChanged();
  }
  dispose(): void {
    this.removePick();
    this.viewer.layers.references.remove(this.group);
    disposeObject(this.group);
    this.viewer.markSceneChanged();
  }
}

export class RefPointView implements EntityView {
  readonly group = new THREE.Group();
  private removePick: () => void;
  private pts: THREE.Points | null = null;
  constructor(private viewer: Viewer, e: RefPointEntity) {
    viewer.layers.references.add(this.group);
    this.removePick = viewer.addPickTarget(objectPick(e.id, this.group, 3));
    this.update(e);
  }
  update(e: RefPointEntity): void {
    if (this.pts) disposeObject(this.pts);
    this.group.clear();
    this.pts = pointSprites(e.def.position, e.color ?? COLORS.refpoint, 11, { depthTest: false });
    this.pts.renderOrder = 5;
    this.group.add(this.pts);
    this.group.visible = e.visible;
    this.viewer.markSceneChanged();
  }
  setHighlight(on: boolean): void {
    if (this.pts) (this.pts.material as THREE.PointsMaterial).color.set(on ? COLORS.highlight : COLORS.refpoint);
    this.viewer.requestRender();
  }
  setVisible(on: boolean): void {
    this.group.visible = on;
    this.viewer.markSceneChanged();
  }
  dispose(): void {
    this.removePick();
    this.viewer.layers.references.remove(this.group);
    disposeObject(this.group);
    this.viewer.markSceneChanged();
  }
}

// ---------------------------------------------------------------------------
// Point cloud
// ---------------------------------------------------------------------------

export class PointsView implements EntityView {
  readonly group = new THREE.Group();
  constructor(private viewer: Viewer, e: PointsEntity) {
    viewer.layers.meshes.add(this.group);
    this.update(e);
  }
  update(e: PointsEntity): void {
    disposeObject(this.group);
    this.group.clear();
    this.group.add(pointCloudObject(e.points.positions, e.color ?? "#c8d0da", 2, e.points.colors));
    this.group.visible = e.visible;
    this.viewer.markSceneChanged();
  }
  setHighlight(): void {}
  setVisible(on: boolean): void {
    this.group.visible = on;
    this.viewer.markSceneChanged();
  }
  dispose(): void {
    this.viewer.layers.meshes.remove(this.group);
    disposeObject(this.group);
    this.viewer.markSceneChanged();
  }
}

// ---------------------------------------------------------------------------
// 3D sketch curves
// ---------------------------------------------------------------------------

export class Sketch3DView implements EntityView {
  readonly group = new THREE.Group();
  private mat = lineMaterial({ color: COLORS.sketch3d, width: 3, depthTest: true });
  private removePick: () => void;
  constructor(private viewer: Viewer, e: Sketch3DEntity) {
    viewer.layers.sketches.add(this.group);
    this.removePick = viewer.addPickTarget(objectPick(e.id, this.group, 2));
    this.update(e);
  }
  update(e: Sketch3DEntity): void {
    for (const c of [...this.group.children]) {
      this.group.remove(c);
      (c as THREE.Mesh).geometry?.dispose();
    }
    this.mat.color.set(e.color ?? COLORS.sketch3d);
    if (e.curves.length) this.group.add(fatSegments(polylinesToPairs(e.curves), this.mat));
    this.group.visible = e.visible;
    this.viewer.markSceneChanged();
  }
  setHighlight(on: boolean): void {
    this.mat.color.set(on ? COLORS.highlight : COLORS.sketch3d);
    this.viewer.requestRender();
  }
  setVisible(on: boolean): void {
    this.group.visible = on;
    this.viewer.markSceneChanged();
  }
  dispose(): void {
    this.removePick();
    this.viewer.layers.sketches.remove(this.group);
    disposeObject(this.group);
    this.viewer.markSceneChanged();
  }
}

// ---------------------------------------------------------------------------
// CAD body (tessellated B-rep)
// ---------------------------------------------------------------------------

export interface BodyTessellation {
  positions: Float32Array;
  normals: Float32Array;
  indices: Uint32Array;
  faceTriangleRanges: Uint32Array;
  edges: { id: number; points: Float32Array }[];
}

export class BodyView implements EntityView {
  readonly group = new THREE.Group();
  private faceMat: THREE.MeshStandardMaterial;
  private edgeMat = lineMaterial({ color: COLORS.bodyEdge, width: 1.4 });
  private removePick: () => void;
  private meshObj: THREE.Mesh | null = null;
  private faceOfTri: Uint32Array = new Uint32Array(0);
  private baseColor: string;

  constructor(private viewer: Viewer, e: BodyEntity) {
    this.baseColor = e.color ?? COLORS.body;
    this.faceMat = new THREE.MeshStandardMaterial({
      color: this.baseColor,
      roughness: 0.45,
      metalness: 0.15,
      side: THREE.DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1,
    });
    viewer.layers.bodies.add(this.group);
    const rc = new THREE.Raycaster();
    rc.firstHitOnly = true;
    this.removePick = viewer.addPickTarget({
      entityId: e.id,
      priority: 1,
      enabled: () => this.group.visible && !!this.meshObj,
      raycast: (ray) => {
        rc.ray.copy(ray);
        const hits = rc.intersectObject(this.meshObj!, false);
        if (!hits.length) return null;
        const h = hits[0];
        const n = h.face!.normal;
        return {
          entityId: e.id,
          point: [h.point.x, h.point.y, h.point.z],
          normal: [n.x, n.y, n.z],
          distance: h.distance,
          faceIndex: h.faceIndex ?? undefined,
          sub: h.faceIndex != null ? this.faceOfTri[h.faceIndex] : undefined,
        };
      },
    });
    this.update(e);
  }

  update(e: BodyEntity): void {
    this.baseColor = e.color ?? COLORS.body;
    this.faceMat.color.set(this.baseColor);
    this.group.visible = e.visible;
  }

  /** last tessellation shown (used for edge/face picking by tools) */
  tess: BodyTessellation | null = null;

  setTessellation(t: BodyTessellation | null): void {
    this.tess = t;
    // materials are shared across tessellations; only geometries are disposed here
    for (const c of [...this.group.children]) {
      this.group.remove(c);
      (c as THREE.Mesh).geometry?.dispose();
    }
    this.meshObj = null;
    if (!t) {
      this.viewer.markSceneChanged();
      return;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(t.positions, 3));
    g.setAttribute("normal", new THREE.BufferAttribute(t.normals, 3));
    g.setIndex(new THREE.BufferAttribute(t.indices, 1));
    g.computeBoundingBox();
    g.computeBoundingSphere();
    (g as unknown as { boundsTree: MeshBVH }).boundsTree = new MeshBVH(g);
    const m = new THREE.Mesh(g, this.faceMat);
    m.raycast = acceleratedRaycast;
    this.meshObj = m;
    this.group.add(m);
    // triangle -> face lookup for face picking
    const triCount = t.indices.length / 3;
    this.faceOfTri = new Uint32Array(triCount);
    for (let f = 0; f < t.faceTriangleRanges.length / 2; f++) {
      const s = t.faceTriangleRanges[2 * f], c = t.faceTriangleRanges[2 * f + 1];
      this.faceOfTri.fill(f, s, s + c);
    }
    let n = 0;
    for (const ed of t.edges) n += Math.max(0, ed.points.length / 3 - 1);
    const pairs = new Float32Array(n * 6);
    let k = 0;
    for (const ed of t.edges) {
      const p = ed.points;
      for (let i = 0; i + 5 < p.length; i += 3) {
        pairs.set(p.subarray(i, i + 6), k);
        k += 6;
      }
    }
    if (pairs.length) {
      const lines = fatSegments(pairs, this.edgeMat);
      lines.userData.noPick = true;
      this.group.add(lines);
    }
    this.viewer.markSceneChanged();
  }

  setHighlight(on: boolean): void {
    this.faceMat.color.set(on ? COLORS.highlight : this.baseColor);
    this.viewer.requestRender();
  }

  setOpacity(alpha: number): void {
    this.faceMat.transparent = alpha < 1;
    this.faceMat.opacity = alpha;
    this.faceMat.depthWrite = alpha >= 1;
    this.viewer.requestRender();
  }

  /** Show per-vertex colours (e.g. a deviation map on the CAD surface). */
  setVertexColors(colors: Uint8Array | null): void {
    if (!this.meshObj) return;
    if (colors) {
      this.meshObj.geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3, true));
      this.faceMat.vertexColors = true;
      this.faceMat.color.set("#ffffff");
    } else {
      this.meshObj.geometry.deleteAttribute("color");
      this.faceMat.vertexColors = false;
      this.faceMat.color.set(this.baseColor);
    }
    this.faceMat.needsUpdate = true;
    this.viewer.requestRender();
  }

  setVisible(on: boolean): void {
    this.group.visible = on;
    this.viewer.markSceneChanged();
  }

  dispose(): void {
    this.removePick();
    this.viewer.layers.bodies.remove(this.group);
    disposeObject(this.group);
    this.faceMat.dispose();
    this.edgeMat.dispose();
    this.viewer.markSceneChanged();
  }
}
