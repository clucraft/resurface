/**
 * Mesh triangle selection: brush painting, rectangle, lasso, smart (region growing) and
 * plain entity picking. Rectangle/lasso only take visible triangles by testing projected
 * centroids against a depth capture of the current view.
 */
import * as THREE from "three";
import type { App } from "./App";
import type { ToolPointerEvent } from "../tools/Tool";
import type { MeshRuntime } from "./MeshRuntime";
import { smartRegion } from "./regionGrow";

type Mode = "add" | "remove" | "replace";

const modeFor = (e: { shift: boolean; ctrl: boolean }, plain: Mode): Mode => (e.ctrl ? "remove" : e.shift ? "add" : plain);

export class SelectionController {
  private dragging = false;
  private start = { x: 0, y: 0 };
  private lasso: { x: number; y: number }[] = [];
  private brushMode: Mode = "add";
  private lastBrush: { x: number; y: number } | null = null;
  /** overlay elements (owned by the Viewport component) */
  overlay: { rect?: HTMLDivElement | null; lasso?: SVGPolylineElement | null; brush?: HTMLDivElement | null } = {};

  constructor(private app: App) {}

  private runtime(e?: ToolPointerEvent): MeshRuntime | undefined {
    const app = this.app;
    let id = app.ui.get().activeMeshId;
    if (e) {
      // the mesh under the cursor becomes the active one
      const hit = e.pick((eid) => app.doc.get(eid)?.kind === "mesh");
      if (hit) id = hit.entityId;
    }
    if (!id) {
      const meshes = app.doc.ofKind("mesh").filter((m) => m.visible);
      if (meshes.length === 1) id = meshes[0].id;
    }
    if (id && app.ui.get().activeMeshId !== id) app.ui.patch({ activeMeshId: id });
    return app.sync?.meshRuntime(id);
  }

  updateBrushCursor(x: number, y: number, show: boolean): void {
    const b = this.overlay.brush;
    if (!b) return;
    const r = this.app.ui.get().brushRadius;
    b.style.display = show ? "block" : "none";
    b.style.width = b.style.height = `${2 * r}px`;
    b.style.transform = `translate(${x - r}px, ${y - r}px)`;
  }

  onPointerDown(e: ToolPointerEvent): boolean {
    if (e.button !== 0) return false;
    const mode = this.app.ui.get().selectionMode;
    this.start = { x: e.x, y: e.y };
    switch (mode) {
      case "pick": {
        const hit = e.pick();
        this.app.setActiveEntity(hit?.entityId ?? null);
        return true;
      }
      case "brush": {
        const rt = this.runtime(e);
        if (!rt) return false;
        this.dragging = true;
        this.brushMode = e.ctrl ? "remove" : "add";
        if (!e.shift && !e.ctrl && this.app.ui.get().selectedTriangles > 0 && e.alt) rt.clearSelection();
        this.lastBrush = null;
        this.paint(e, rt);
        return true;
      }
      case "rect":
      case "lasso": {
        this.dragging = true;
        this.lasso = [{ x: e.x, y: e.y }];
        this.drawMarquee(e.x, e.y);
        return true;
      }
      case "smart": {
        const rt = this.runtime(e);
        const hit = e.pick((id) => id === rt?.entityId);
        if (!rt || !hit || hit.faceIndex === undefined) return true;
        const res = smartRegion(rt.mesh, rt.neighbors, hit.faceIndex, this.app.ui.get().smartAngle);
        rt.applyTriangles(toTriangleList(res.mask), modeFor(e, "replace"));
        this.app.setStatus(`Smart select: ${res.kind === "freeform" ? "smooth freeform" : res.kind} region, ${res.count.toLocaleString()} triangles`);
        return true;
      }
    }
    return false;
  }

  onPointerMove(e: ToolPointerEvent): boolean {
    const mode = this.app.ui.get().selectionMode;
    if (mode === "brush") this.updateBrushCursor(e.x, e.y, true);
    if (!this.dragging) return false;
    if (mode === "brush") {
      const rt = this.runtime();
      if (rt) this.paint(e, rt);
      return true;
    }
    if (mode === "rect" || mode === "lasso") {
      if (mode === "lasso") {
        const last = this.lasso[this.lasso.length - 1];
        if (Math.hypot(e.x - last.x, e.y - last.y) > 3) this.lasso.push({ x: e.x, y: e.y });
      }
      this.drawMarquee(e.x, e.y);
      return true;
    }
    return false;
  }

  onPointerUp(e: ToolPointerEvent): boolean {
    if (!this.dragging) return false;
    this.dragging = false;
    const mode = this.app.ui.get().selectionMode;
    if (mode === "rect" || mode === "lasso") {
      this.hideMarquee();
      const rt = this.runtime();
      if (!rt) return true;
      let poly: { x: number; y: number }[];
      if (mode === "rect") {
        const x0 = Math.min(this.start.x, e.x), x1 = Math.max(this.start.x, e.x);
        const y0 = Math.min(this.start.y, e.y), y1 = Math.max(this.start.y, e.y);
        if (x1 - x0 < 3 && y1 - y0 < 3) {
          if (!e.shift && !e.ctrl) rt.clearSelection();
          return true;
        }
        poly = [{ x: x0, y: y0 }, { x: x1, y: y0 }, { x: x1, y: y1 }, { x: x0, y: y1 }];
      } else {
        poly = this.lasso;
        if (poly.length < 3) return true;
      }
      const tris = this.trianglesInPolygon(rt, poly, this.app.ui.get().selectVisibleOnly);
      rt.applyTriangles(tris, modeFor(e, "replace"));
    }
    return true;
  }

  // -------------------------------------------------------------------------

  private paint(e: ToolPointerEvent, rt: MeshRuntime): void {
    const viewer = this.app.viewer!;
    const r = this.app.ui.get().brushRadius;
    // interpolate along the stroke so fast moves leave no gaps
    const pts: { x: number; y: number }[] = [];
    if (this.lastBrush) {
      const d = Math.hypot(e.x - this.lastBrush.x, e.y - this.lastBrush.y);
      const steps = Math.min(32, Math.floor(d / Math.max(2, r * 0.5)));
      for (let i = 1; i <= steps; i++) {
        const t = i / (steps + 1);
        pts.push({ x: this.lastBrush.x + (e.x - this.lastBrush.x) * t, y: this.lastBrush.y + (e.y - this.lastBrush.y) * t });
      }
    }
    pts.push({ x: e.x, y: e.y });
    this.lastBrush = { x: e.x, y: e.y };
    const camPos = viewer.camera.position;
    const pos = rt.mesh.positions;
    const idx = rt.mesh.indices;
    for (const p of pts) {
      const rect = viewer.domElement.getBoundingClientRect();
      const ndc = viewer.clientToNdc(p.x + rect.left, p.y + rect.top);
      const hit = viewer.pick(ndc, (id) => id === rt.entityId);
      if (!hit) continue;
      const center = new THREE.Vector3(...hit.point);
      const radius = viewer.pixelSize(center) * r;
      const tris = rt.spatial.trianglesInSphere(hit.point, radius);
      // keep triangles facing the camera (avoid painting through thin walls)
      const keep: number[] = [];
      for (let i = 0; i < tris.length; i++) {
        const t = tris[i];
        const a = idx[3 * t], b = idx[3 * t + 1], c = idx[3 * t + 2];
        const ax = pos[3 * a], ay = pos[3 * a + 1], az = pos[3 * a + 2];
        const e1x = pos[3 * b] - ax, e1y = pos[3 * b + 1] - ay, e1z = pos[3 * b + 2] - az;
        const e2x = pos[3 * c] - ax, e2y = pos[3 * c + 1] - ay, e2z = pos[3 * c + 2] - az;
        const nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
        const vx = ax - camPos.x, vy = ay - camPos.y, vz = az - camPos.z;
        const facing = nx * vx + ny * vy + nz * vz;
        // the hit normal tells us which side the user is painting on
        const hn = hit.normal ?? [0, 0, 0];
        const sameSide = nx * hn[0] + ny * hn[1] + nz * hn[2] > 0;
        if (facing < 0 || sameSide) keep.push(t);
      }
      rt.applyTriangles(keep, this.brushMode);
    }
  }

  private drawMarquee(x: number, y: number): void {
    const mode = this.app.ui.get().selectionMode;
    if (mode === "rect" && this.overlay.rect) {
      const el = this.overlay.rect;
      el.style.display = "block";
      el.style.left = `${Math.min(this.start.x, x)}px`;
      el.style.top = `${Math.min(this.start.y, y)}px`;
      el.style.width = `${Math.abs(x - this.start.x)}px`;
      el.style.height = `${Math.abs(y - this.start.y)}px`;
    } else if (mode === "lasso" && this.overlay.lasso) {
      const pts = [...this.lasso, { x, y }];
      this.overlay.lasso.setAttribute("points", pts.map((p) => `${p.x},${p.y}`).join(" "));
      (this.overlay.lasso.parentNode as SVGElement).style.display = "block";
    }
  }

  private hideMarquee(): void {
    if (this.overlay.rect) this.overlay.rect.style.display = "none";
    if (this.overlay.lasso) {
      this.overlay.lasso.setAttribute("points", "");
      (this.overlay.lasso.parentNode as SVGElement).style.display = "none";
    }
  }

  /** Triangles whose centroid projects inside a screen polygon (optionally only visible ones). */
  trianglesInPolygon(rt: MeshRuntime, poly: { x: number; y: number }[], visibleOnly: boolean): Uint32Array {
    const viewer = this.app.viewer!;
    const cam = viewer.camera;
    cam.updateMatrixWorld();
    const { width, height } = viewer.size;
    const m = new THREE.Matrix4().multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse).elements;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const p of poly) {
      minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
      minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
    }
    const depth = visibleOnly ? captureDepth(this.app) : null;
    const near = cam.near, far = cam.far;
    const linear = (zNdc: number) => (2 * near * far) / (far + near - zNdc * (far - near));
    const pos = rt.mesh.positions, idx = rt.mesh.indices;
    const camPos = cam.position;
    const out: number[] = [];
    const frustum = new THREE.Frustum().setFromProjectionMatrix(new THREE.Matrix4().multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse));
    for (const chunk of rt.chunks) {
      const box = new THREE.Box3(new THREE.Vector3(...chunk.bounds.min), new THREE.Vector3(...chunk.bounds.max));
      if (!frustum.intersectsBox(box)) continue;
      for (let t = chunk.triStart, end = chunk.triStart + chunk.triCount; t < end; t++) {
        const a = idx[3 * t], b = idx[3 * t + 1], c = idx[3 * t + 2];
        const cx = (pos[3 * a] + pos[3 * b] + pos[3 * c]) / 3;
        const cy = (pos[3 * a + 1] + pos[3 * b + 1] + pos[3 * c + 1]) / 3;
        const cz = (pos[3 * a + 2] + pos[3 * b + 2] + pos[3 * c + 2]) / 3;
        const w = m[3] * cx + m[7] * cy + m[11] * cz + m[15];
        if (w <= 0) continue;
        const sx = ((m[0] * cx + m[4] * cy + m[8] * cz + m[12]) / w + 1) * 0.5 * width;
        const sy = (1 - (m[1] * cx + m[5] * cy + m[9] * cz + m[13]) / w) * 0.5 * height;
        if (sx < minX || sx > maxX || sy < minY || sy > maxY) continue;
        if (poly.length > 4 && !pointInPolygon(sx, sy, poly)) continue;
        if (visibleOnly) {
          // back-facing triangles are never "visible"
          const e1x = pos[3 * b] - pos[3 * a], e1y = pos[3 * b + 1] - pos[3 * a + 1], e1z = pos[3 * b + 2] - pos[3 * a + 2];
          const e2x = pos[3 * c] - pos[3 * a], e2y = pos[3 * c + 1] - pos[3 * a + 1], e2z = pos[3 * c + 2] - pos[3 * a + 2];
          const nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
          if (nx * (cx - camPos.x) + ny * (cy - camPos.y) + nz * (cz - camPos.z) > 0) continue;
          if (depth) {
            const px = Math.min(depth.width - 1, Math.max(0, Math.floor((sx / width) * depth.width)));
            const py = Math.min(depth.height - 1, Math.max(0, Math.floor(((height - sy) / height) * depth.height)));
            const d = depth.data[py * depth.width + px];
            if (d < 1) {
              const zNdc = (m[2] * cx + m[6] * cy + m[10] * cz + m[14]) / w;
              const triDist = linear(zNdc);
              const bufDist = linear(d * 2 - 1);
              if (triDist > bufDist * 1.004 + viewer.pixelSize(new THREE.Vector3(cx, cy, cz)) * 2) continue;
            }
          }
        }
        out.push(t);
      }
    }
    return Uint32Array.from(out);
  }
}

function toTriangleList(res: unknown): Uint32Array {
  if (res instanceof Uint32Array) return res;
  if (res instanceof Uint8Array) {
    let n = 0;
    for (let i = 0; i < res.length; i++) n += res[i] ? 1 : 0;
    const out = new Uint32Array(n);
    let k = 0;
    for (let i = 0; i < res.length; i++) if (res[i]) out[k++] = i;
    return out;
  }
  if (Array.isArray(res)) return Uint32Array.from(res as number[]);
  const r = res as { mask?: Uint8Array; triangles?: ArrayLike<number> };
  if (r?.mask) return toTriangleList(r.mask);
  if (r?.triangles) return Uint32Array.from(r.triangles as ArrayLike<number>);
  return new Uint32Array(0);
}

function pointInPolygon(x: number, y: number, poly: { x: number; y: number }[]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i].x, yi = poly[i].y, xj = poly[j].x, yj = poly[j].y;
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/** Render window-space depth (0..1) of meshes and bodies into a float buffer. */
function captureDepth(app: App): { width: number; height: number; data: Float32Array } | null {
  const viewer = app.viewer!;
  const renderer = viewer.renderer;
  const gl = renderer.getContext() as WebGL2RenderingContext;
  if (!gl.getExtension("EXT_color_buffer_float")) return null;
  const scale = 0.5; // half resolution is plenty and 4x cheaper
  const w = Math.max(1, Math.floor(viewer.size.width * scale));
  const h = Math.max(1, Math.floor(viewer.size.height * scale));
  const target = new THREE.WebGLRenderTarget(w, h, { type: THREE.FloatType, format: THREE.RGBAFormat, depthBuffer: true });
  const mat = new THREE.ShaderMaterial({
    vertexShader: `void main(){ gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
    fragmentShader: `void main(){ gl_FragColor = vec4(gl_FragCoord.z, 0.0, 0.0, 1.0); }`,
    side: THREE.DoubleSide,
  });
  const hidden: THREE.Object3D[] = [];
  for (const g of [viewer.layers.references, viewer.layers.sketches, viewer.layers.overlay, viewer.layers.helpers]) {
    if (g.visible) {
      hidden.push(g);
      g.visible = false;
    }
  }
  const prevOverride = viewer.scene.overrideMaterial;
  viewer.scene.overrideMaterial = mat;
  const prevTarget = renderer.getRenderTarget();
  renderer.setRenderTarget(target);
  renderer.setClearColor(0xffffff, 1);
  renderer.clear(true, true, true);
  renderer.render(viewer.scene, viewer.camera);
  const rgba = new Float32Array(w * h * 4);
  renderer.readRenderTargetPixels(target, 0, 0, w, h, rgba);
  renderer.setRenderTarget(prevTarget);
  renderer.setClearColor(0x000000, 0);
  viewer.scene.overrideMaterial = prevOverride;
  for (const g of hidden) g.visible = true;
  target.dispose();
  mat.dispose();
  const data = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) data[i] = rgba[4 * i];
  viewer.requestRender();
  return { width: w, height: h, data };
}
