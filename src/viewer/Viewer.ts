/**
 * The 3D viewport: renderer, camera, navigation, layers, on-demand rendering, picking and
 * HTML labels. Entity-specific drawing lives in the *View classes managed by SceneSync.
 */
import * as THREE from "three";
import { CameraControls } from "./CameraControls";
import { Emitter } from "../core/emitter";
import type { Vec3 } from "../core/types";

export interface PickHit {
  entityId: string;
  point: Vec3;
  normal?: Vec3;
  distance: number;
  faceIndex?: number;
  /** sub-element (edge id, sketch entity id, ...) when the view provides one */
  sub?: string | number;
}

/** Side of the compare split view: the scan mesh (left) or the CAD bodies (right). */
export type CompareSide = "stl" | "cad";

/** Anything that can be picked with a world ray. Views register themselves as pick targets. */
export interface PickTarget {
  entityId: string;
  /** larger = picked preferentially when distances are similar (e.g. lines over meshes) */
  priority?: number;
  /**
   * Entity layer the target draws into. In compare (split) view "mesh" targets are pickable only
   * left of the divider and "body" targets only right of it; untagged targets are always pickable.
   */
  layer?: "mesh" | "body";
  enabled(): boolean;
  raycast(ray: THREE.Ray, ctx: { pixelSize: (p: THREE.Vector3) => number; camera: THREE.Camera }): PickHit | null;
}

export interface ViewerEvents {
  /** camera moved (fires during navigation) */
  camera: void;
  /** navigation started/ended — large meshes switch to LOD proxies while interacting */
  interaction: boolean;
  resize: { width: number; height: number };
  beforeRender: void;
  [key: string]: unknown;
}

export type StandardView = "front" | "back" | "left" | "right" | "top" | "bottom" | "iso";

const VIEW_DIRS: Record<StandardView, { dir: [number, number, number]; up: [number, number, number] }> = {
  front: { dir: [0, -1, 0], up: [0, 0, 1] },
  back: { dir: [0, 1, 0], up: [0, 0, 1] },
  left: { dir: [-1, 0, 0], up: [0, 0, 1] },
  right: { dir: [1, 0, 0], up: [0, 0, 1] },
  top: { dir: [0, 0, 1], up: [0, 1, 0] },
  bottom: { dir: [0, 0, -1], up: [0, -1, 0] },
  iso: { dir: [1, -1, 0.8], up: [0, 0, 1] },
};

export interface Label {
  el: HTMLElement;
  position: THREE.Vector3;
  visible: boolean;
}

export class Viewer extends Emitter<ViewerEvents> {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  camera: THREE.PerspectiveCamera;
  readonly controls: CameraControls;
  /** layers for entity views */
  readonly layers = {
    meshes: new THREE.Group(),
    references: new THREE.Group(),
    sketches: new THREE.Group(),
    bodies: new THREE.Group(),
    /** transient tool graphics */
    overlay: new THREE.Group(),
    helpers: new THREE.Group(),
  };
  readonly headlight = new THREE.DirectionalLight(0xffffff, 1.6);
  readonly grid: THREE.Group = new THREE.Group();
  private gridSize = 0;
  private pickTargets = new Set<PickTarget>();
  private renderQueued = false;
  private width = 1;
  private height = 1;
  private resizeObserver: ResizeObserver;
  readonly labelLayer: HTMLDivElement;
  private labels = new Set<Label>();
  interacting = false;
  /** small scene for the orientation triad in the lower-left corner */
  private triadScene = new THREE.Scene();
  private triadCamera = new THREE.OrthographicCamera(-1.6, 1.6, 1.6, -1.6, 0.1, 10);
  showGrid = true;
  /** compare split view: divider position from the left (0..1), or null when off */
  private compareFraction: number | null = null;

  constructor(readonly container: HTMLElement) {
    super();
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.autoClear = false;
    this.renderer.localClippingEnabled = true;
    this.renderer.domElement.className = "viewer-canvas";
    container.appendChild(this.renderer.domElement);

    this.labelLayer = document.createElement("div");
    this.labelLayer.className = "viewer-labels";
    container.appendChild(this.labelLayer);

    this.camera = new THREE.PerspectiveCamera(35, 1, 0.1, 100000);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(300, -400, 250);
    this.camera.lookAt(0, 0, 0);

    for (const g of Object.values(this.layers)) this.scene.add(g);
    this.layers.overlay.renderOrder = 10;

    // Lighting: a headlight that follows the camera + soft hemisphere fill.
    this.scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x3a3530, 0.9));
    this.scene.add(this.camera);
    this.camera.add(this.headlight);
    this.headlight.position.set(0.3, 0.5, 1);
    this.headlight.target.position.set(0, 0, -1);
    this.camera.add(this.headlight.target);

    this.layers.helpers.add(this.grid);
    this.buildTriad();

    this.controls = new CameraControls(this.camera, this.renderer.domElement, {
      pick: (ndc) => {
        const hit = this.pick(ndc);
        return hit ? new THREE.Vector3(...hit.point) : null;
      },
      onChange: () => {
        this.emit("camera", undefined);
        this.requestRender();
      },
      onInteractionStart: () => {
        this.interacting = true;
        this.emit("interaction", true);
      },
      onInteractionEnd: () => {
        this.interacting = false;
        this.emit("interaction", false);
        this.requestRender();
      },
    });

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.resize();
  }

  dispose(): void {
    this.resizeObserver.disconnect();
    this.controls.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
    this.labelLayer.remove();
  }

  get domElement(): HTMLCanvasElement {
    return this.renderer.domElement;
  }

  get size(): { width: number; height: number } {
    return { width: this.width, height: this.height };
  }

  private resize(): void {
    const w = Math.max(1, this.container.clientWidth);
    const h = Math.max(1, this.container.clientHeight);
    this.width = w;
    this.height = h;
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = `${w}px`;
    this.renderer.domElement.style.height = `${h}px`;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.emit("resize", { width: w, height: h });
    this.renderNow();
  }

  // -------------------------------------------------------------------------
  // Rendering
  // -------------------------------------------------------------------------

  requestRender = (): void => {
    if (this.renderQueued) return;
    this.renderQueued = true;
    requestAnimationFrame(() => {
      this.renderQueued = false;
      this.renderNow();
    });
  };

  renderNow(): void {
    this.updateClipping();
    this.emit("beforeRender", undefined);
    const r = this.renderer;
    r.setViewport(0, 0, this.width, this.height);
    r.setScissorTest(false);
    r.clear(true, true, true);
    this.grid.visible = this.showGrid;
    if (this.compareFraction === null) r.render(this.scene, this.camera);
    else this.renderCompare(this.compareFraction);
    this.renderTriad();
    this.updateLabels();
  }

  // -------------------------------------------------------------------------
  // Compare (split) view
  // -------------------------------------------------------------------------

  /**
   * Split view comparing the scan with the CAD model: left of the divider the scene is drawn
   * without bodies, right of it without meshes; both halves share the camera. `fraction` is the
   * divider position from the left edge (0..1); null turns the split off (normal single pass).
   */
  setCompare(fraction: number | null): void {
    const f = fraction === null || !Number.isFinite(fraction) ? null : THREE.MathUtils.clamp(fraction, 0, 1);
    if (f === this.compareFraction) return;
    this.compareFraction = f;
    this.requestRender();
  }

  /** Divider position (0..1 from the left) while compare view is on, else null. */
  get compare(): number | null {
    return this.compareFraction;
  }

  /** Divider x in CSS pixels, snapped to whole device pixels so the two scissor rects tile exactly. */
  private compareSplitX(fraction: number): number {
    const pr = this.renderer.getPixelRatio();
    return Math.round(this.width * fraction * pr) / pr;
  }

  /** Which side of the divider an NDC x coordinate falls on (null when compare view is off). */
  compareSideAt(ndcX: number): CompareSide | null {
    if (this.compareFraction === null) return null;
    const x = ((ndcX + 1) / 2) * this.width;
    return x < this.compareSplitX(this.compareFraction) ? "stl" : "cad";
  }

  /**
   * Two scissored passes over the same, already cleared colour/depth buffer (the rects are
   * disjoint, so nothing is cleared in between). setScissor takes CSS pixels and three scales
   * them by the pixel ratio; the split is snapped to device pixels so the halves meet exactly.
   */
  private renderCompare(fraction: number): void {
    const r = this.renderer;
    const { meshes, bodies, overlay } = this.layers;
    const split = this.compareSplitX(fraction);
    const meshesVisible = meshes.visible;
    const bodiesVisible = bodies.visible;
    // overlay objects may opt into one side with userData.compareSide = "stl" | "cad"
    const tagged: THREE.Object3D[] = [];
    overlay.traverse((o) => {
      const side = o.userData.compareSide;
      if (side === "stl" || side === "cad") tagged.push(o);
    });
    const taggedVisible = tagged.map((o) => o.visible);
    const showSide = (side: CompareSide) => {
      for (let i = 0; i < tagged.length; i++) tagged[i].visible = taggedVisible[i] && tagged[i].userData.compareSide === side;
    };
    r.setScissorTest(true);
    try {
      if (split > 0) {
        r.setScissor(0, 0, split, this.height);
        bodies.visible = false;
        showSide("stl");
        r.render(this.scene, this.camera);
        bodies.visible = bodiesVisible;
      }
      if (split < this.width) {
        r.setScissor(split, 0, this.width - split, this.height);
        meshes.visible = false;
        showSide("cad");
        r.render(this.scene, this.camera);
      }
    } finally {
      meshes.visible = meshesVisible;
      bodies.visible = bodiesVisible;
      for (let i = 0; i < tagged.length; i++) tagged[i].visible = taggedVisible[i];
      r.setScissor(0, 0, this.width, this.height);
      r.setScissorTest(false);
    }
  }

  /** keep near/far tight around the scene to preserve depth precision */
  private updateClipping(): void {
    if (!this.cachedBounds) this.cachedBounds = this.sceneBounds(true);
    const box = this.cachedBounds;
    if (box.isEmpty()) return;
    const sphere = box.getBoundingSphere(new THREE.Sphere());
    const dist = this.camera.position.distanceTo(sphere.center);
    const far = dist + sphere.radius * 3 + 10;
    const near = Math.max(far / 10000, 0.005);
    if (Math.abs(this.camera.far - far) > 1e-6 || Math.abs(this.camera.near - near) > 1e-6) {
      this.camera.near = near;
      this.camera.far = far;
      this.camera.updateProjectionMatrix();
    }
  }

  private cachedBounds: THREE.Box3 | null = null;

  /** Call after adding/removing/changing scene content so cached bounds are recomputed. */
  markSceneChanged(): void {
    this.cachedBounds = null;
    this.requestRender();
  }

  private buildTriad(): void {
    const mk = (dir: THREE.Vector3, color: number, label: string) => {
      const arrow = new THREE.ArrowHelper(dir, new THREE.Vector3(), 1, color, 0.28, 0.14);
      (arrow.line.material as THREE.LineBasicMaterial).linewidth = 2;
      this.triadScene.add(arrow);
      const canvas = document.createElement("canvas");
      canvas.width = canvas.height = 64;
      const ctx = canvas.getContext("2d")!;
      ctx.fillStyle = `#${color.toString(16).padStart(6, "0")}`;
      ctx.font = "bold 44px Segoe UI, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(label, 32, 34);
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas), depthTest: false }));
      sprite.position.copy(dir.clone().multiplyScalar(1.3));
      sprite.scale.setScalar(0.45);
      this.triadScene.add(sprite);
    };
    mk(new THREE.Vector3(1, 0, 0), 0xf05454, "X");
    mk(new THREE.Vector3(0, 1, 0), 0x5bd16b, "Y");
    mk(new THREE.Vector3(0, 0, 1), 0x4f8ff7, "Z");
  }

  private renderTriad(): void {
    const s = 96;
    const dir = new THREE.Vector3().subVectors(this.camera.position, this.controls.target).normalize();
    this.triadCamera.position.copy(dir.multiplyScalar(4));
    this.triadCamera.up.copy(this.camera.up);
    this.triadCamera.quaternion.copy(this.camera.quaternion);
    this.triadCamera.updateMatrixWorld();
    const r = this.renderer;
    r.setViewport(8, 8, s, s);
    r.setScissor(8, 8, s, s);
    r.setScissorTest(true);
    r.clearDepth();
    r.render(this.triadScene, this.triadCamera);
    r.setScissorTest(false);
    r.setViewport(0, 0, this.width, this.height);
  }

  /** Rebuild the ground grid to suit the scene extent. */
  updateGrid(): void {
    const box = this.sceneBounds(false);
    const size = box.isEmpty() ? 200 : Math.max(box.max.x - box.min.x, box.max.y - box.min.y) * 1.6;
    const step = Math.pow(10, Math.floor(Math.log10(Math.max(size, 1) / 10)));
    const half = Math.ceil(size / 2 / step) * step;
    if (Math.abs(half - this.gridSize) < 1e-9) return;
    this.gridSize = half;
    this.grid.clear();
    const divisions = Math.round((2 * half) / step);
    const minor = new THREE.GridHelper(2 * half, divisions, 0x3a4250, 0x2c323c);
    minor.rotation.x = Math.PI / 2;
    (minor.material as THREE.Material).transparent = true;
    (minor.material as THREE.Material).opacity = 0.55;
    (minor.material as THREE.Material).depthWrite = false;
    this.grid.add(minor);
    const z = box.isEmpty() ? 0 : Math.min(0, box.min.z);
    this.grid.position.z = z;
    const axes = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-half, 0, 0), new THREE.Vector3(half, 0, 0),
      new THREE.Vector3(0, -half, 0), new THREE.Vector3(0, half, 0),
    ]);
    axes.setAttribute("color", new THREE.Float32BufferAttribute([0.75, 0.3, 0.3, 0.75, 0.3, 0.3, 0.3, 0.65, 0.35, 0.3, 0.65, 0.35], 3));
    const axisLines = new THREE.LineSegments(axes, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.8, depthWrite: false }));
    this.grid.add(axisLines);
  }

  /** Bounding box of visible entity layers. */
  sceneBounds(includeOverlay: boolean): THREE.Box3 {
    const box = new THREE.Box3();
    const layers = [this.layers.meshes, this.layers.references, this.layers.sketches, this.layers.bodies];
    if (includeOverlay) layers.push(this.layers.overlay);
    for (const g of layers) {
      g.traverseVisible((o) => {
        const geom = (o as THREE.Mesh).geometry as THREE.BufferGeometry | undefined;
        if (!geom || o.userData.noBounds) return;
        if (!geom.boundingBox) geom.computeBoundingBox();
        if (!geom.boundingBox || geom.boundingBox.isEmpty()) return;
        o.updateWorldMatrix(true, false);
        box.union(geom.boundingBox.clone().applyMatrix4(o.matrixWorld));
      });
    }
    return box;
  }

  // -------------------------------------------------------------------------
  // Views
  // -------------------------------------------------------------------------

  fitAll(direction?: StandardView): void {
    const box = this.sceneBounds(false);
    const sphere = box.isEmpty() ? new THREE.Sphere(new THREE.Vector3(), 100) : box.getBoundingSphere(new THREE.Sphere());
    if (direction) {
      const v = VIEW_DIRS[direction];
      this.controls.fitSphere(sphere.center, sphere.radius, new THREE.Vector3(...v.dir).normalize(), new THREE.Vector3(...v.up));
    } else {
      this.controls.fitSphere(sphere.center, sphere.radius);
    }
    this.updateGrid();
    this.requestRender();
  }

  /** Look perpendicular onto a plane (sketch mode). */
  lookAlong(normal: Vec3, up: Vec3, center?: Vec3, radius?: number): void {
    const box = this.sceneBounds(false);
    const sphere = box.isEmpty() ? new THREE.Sphere(new THREE.Vector3(), 100) : box.getBoundingSphere(new THREE.Sphere());
    const c = center ? new THREE.Vector3(...center) : sphere.center;
    this.controls.fitSphere(c, radius ?? sphere.radius, new THREE.Vector3(...normal), new THREE.Vector3(...up));
    this.requestRender();
  }

  // -------------------------------------------------------------------------
  // Coordinates & picking
  // -------------------------------------------------------------------------

  clientToNdc(clientX: number, clientY: number): THREE.Vector2 {
    const r = this.renderer.domElement.getBoundingClientRect();
    return new THREE.Vector2(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
  }

  clientToLocal(clientX: number, clientY: number): { x: number; y: number } {
    const r = this.renderer.domElement.getBoundingClientRect();
    return { x: clientX - r.left, y: clientY - r.top };
  }

  rayFromNdc(ndc: THREE.Vector2): THREE.Ray {
    const rc = new THREE.Raycaster();
    rc.setFromCamera(ndc, this.camera);
    return rc.ray.clone();
  }

  /** World -> CSS pixel coordinates relative to the canvas; z is NDC depth. */
  worldToScreen(p: Vec3 | THREE.Vector3): { x: number; y: number; z: number } {
    const v = p instanceof THREE.Vector3 ? p.clone() : new THREE.Vector3(p[0], p[1], p[2]);
    v.project(this.camera);
    return { x: ((v.x + 1) / 2) * this.width, y: ((1 - v.y) / 2) * this.height, z: v.z };
  }

  /** World size of one screen pixel at point p. */
  pixelSize = (p: THREE.Vector3): number => {
    const cam = this.camera;
    const dist = cam.position.distanceTo(p);
    const fov = THREE.MathUtils.degToRad(cam.fov);
    return (2 * Math.tan(fov / 2) * dist) / this.height;
  };

  addPickTarget(t: PickTarget): () => void {
    this.pickTargets.add(t);
    return () => this.pickTargets.delete(t);
  }

  /** Pick the closest target under an NDC coordinate. `filter` restricts by entity id. */
  pick(ndc: THREE.Vector2, filter?: (entityId: string) => boolean): PickHit | null {
    const ray = this.rayFromNdc(ndc);
    return this.pickRay(ray, filter, ndc.x);
  }

  /**
   * Pick along a camera ray. In compare view the side of the divider decides whether mesh or
   * body targets are eligible; the side comes from `ndcX` when given, else from the ray itself.
   */
  pickRay(ray: THREE.Ray, filter?: (entityId: string) => boolean, ndcX?: number): PickHit | null {
    let side: CompareSide | null = null;
    if (this.compareFraction !== null) {
      // any point along a camera ray projects back onto the screen position it was cast through
      const x = ndcX ?? ray.at(Math.max(this.camera.near * 2, 1e-3), new THREE.Vector3()).project(this.camera).x;
      side = this.compareSideAt(x);
    }
    let best: PickHit | null = null;
    let bestScore = Infinity;
    for (const t of this.pickTargets) {
      if (!t.enabled()) continue;
      if (filter && !filter(t.entityId)) continue;
      if ((side === "stl" && t.layer === "body") || (side === "cad" && t.layer === "mesh")) continue;
      let hit: PickHit | null = null;
      try {
        hit = t.raycast(ray, { pixelSize: this.pixelSize, camera: this.camera });
      } catch (err) {
        console.warn("pick failed", err);
      }
      if (!hit) continue;
      // priority acts as a small depth bias so curves/points win over the surface they lie on
      const score = hit.distance - (t.priority ?? 0) * this.pixelSize(new THREE.Vector3(...hit.point)) * 6;
      if (score < bestScore) {
        bestScore = score;
        best = hit;
      }
    }
    return best;
  }

  // -------------------------------------------------------------------------
  // HTML labels
  // -------------------------------------------------------------------------

  addLabel(html: string, position: Vec3, className = "viewer-label"): Label {
    const el = document.createElement("div");
    el.className = className;
    el.innerHTML = html;
    this.labelLayer.appendChild(el);
    const label: Label = { el, position: new THREE.Vector3(...position), visible: true };
    this.labels.add(label);
    this.requestRender();
    return label;
  }

  removeLabel(label: Label): void {
    label.el.remove();
    this.labels.delete(label);
  }

  private updateLabels(): void {
    for (const l of this.labels) {
      if (!l.visible) {
        l.el.style.display = "none";
        continue;
      }
      const s = this.worldToScreen(l.position);
      if (s.z > 1 || s.z < -1) {
        l.el.style.display = "none";
        continue;
      }
      l.el.style.display = "";
      l.el.style.transform = `translate(${s.x.toFixed(1)}px, ${s.y.toFixed(1)}px) translate(-50%, -50%)`;
    }
  }
}
