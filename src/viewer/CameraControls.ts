/**
 * CAD-style camera navigation (Z-up turntable):
 *   right drag  = orbit around the point under the cursor
 *   middle drag = pan (also shift + right drag)
 *   wheel       = zoom towards the cursor
 * Left mouse is reserved for tools/selection. Works for perspective and orthographic cameras.
 */
import * as THREE from "three";

export interface CameraControlsOptions {
  /** returns the world point under the given NDC coordinate, or null */
  pick: (ndc: THREE.Vector2) => THREE.Vector3 | null;
  onChange: () => void;
  onInteractionStart?: () => void;
  onInteractionEnd?: () => void;
}

type DragMode = "none" | "orbit" | "pan";

export class CameraControls {
  /** orbit/zoom centre used when nothing is under the cursor */
  target = new THREE.Vector3();
  enabled = true;
  rotateSpeed = 0.006;
  zoomSpeed = 0.0015;
  private mode: DragMode = "none";
  private pivot = new THREE.Vector3();
  private last = new THREE.Vector2();
  private panPlaneDepth = 1;
  private interacting = false;
  private wheelTimer: number | undefined;
  /** turntable keeps Z up; when false the orbit is a free trackball */
  turntable = true;

  constructor(
    public camera: THREE.PerspectiveCamera | THREE.OrthographicCamera,
    private dom: HTMLElement,
    private opts: CameraControlsOptions,
  ) {
    dom.addEventListener("pointerdown", this.onPointerDown);
    dom.addEventListener("wheel", this.onWheel, { passive: false });
    dom.addEventListener("contextmenu", this.onContextMenu);
  }

  dispose(): void {
    this.dom.removeEventListener("pointerdown", this.onPointerDown);
    this.dom.removeEventListener("wheel", this.onWheel);
    this.dom.removeEventListener("contextmenu", this.onContextMenu);
    window.removeEventListener("pointermove", this.onPointerMove);
    window.removeEventListener("pointerup", this.onPointerUp);
  }

  private ndc(e: { clientX: number; clientY: number }): THREE.Vector2 {
    const r = this.dom.getBoundingClientRect();
    return new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  }

  private contextMenuSuppressed = false;
  private onContextMenu = (e: Event) => {
    if (this.contextMenuSuppressed) {
      e.preventDefault();
      this.contextMenuSuppressed = false;
    } else {
      e.preventDefault();
    }
  };

  private startInteraction() {
    if (!this.interacting) {
      this.interacting = true;
      this.opts.onInteractionStart?.();
    }
  }

  private endInteraction() {
    if (this.interacting) {
      this.interacting = false;
      this.opts.onInteractionEnd?.();
    }
  }

  private onPointerDown = (e: PointerEvent) => {
    if (!this.enabled) return;
    let mode: DragMode = "none";
    if (e.button === 2 && !e.shiftKey) mode = "orbit";
    else if (e.button === 1 || (e.button === 2 && e.shiftKey)) mode = "pan";
    if (mode === "none") return;
    e.preventDefault();
    this.mode = mode;
    this.last.set(e.clientX, e.clientY);
    const hit = this.opts.pick(this.ndc(e));
    if (mode === "orbit") {
      this.pivot.copy(hit ?? this.target);
    } else {
      const p = hit ?? this.target;
      this.panPlaneDepth = p.clone().project(this.camera).z;
    }
    window.addEventListener("pointermove", this.onPointerMove);
    window.addEventListener("pointerup", this.onPointerUp);
  };

  private onPointerMove = (e: PointerEvent) => {
    const dx = e.clientX - this.last.x;
    const dy = e.clientY - this.last.y;
    if (dx === 0 && dy === 0) return;
    this.contextMenuSuppressed = true;
    this.startInteraction();
    this.last.set(e.clientX, e.clientY);
    if (this.mode === "orbit") this.orbit(dx, dy);
    else if (this.mode === "pan") this.pan(e, dx, dy);
    this.opts.onChange();
  };

  private onPointerUp = () => {
    this.mode = "none";
    window.removeEventListener("pointermove", this.onPointerMove);
    window.removeEventListener("pointerup", this.onPointerUp);
    this.endInteraction();
  };

  private orbit(dx: number, dy: number) {
    const cam = this.camera;
    const offset = cam.position.clone().sub(this.pivot);
    const targetOffset = this.target.clone().sub(this.pivot);
    let q: THREE.Quaternion;
    if (this.turntable) {
      const qz = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), -dx * this.rotateSpeed);
      const right = new THREE.Vector3().setFromMatrixColumn(cam.matrixWorld, 0).applyQuaternion(qz).normalize();
      // keep away from the poles so the up vector stays well-defined
      const forward = cam.getWorldDirection(new THREE.Vector3()).applyQuaternion(qz);
      const pitch = Math.asin(THREE.MathUtils.clamp(forward.z, -1, 1));
      let dPitch = -dy * this.rotateSpeed;
      const limit = Math.PI / 2 - 0.01;
      if (pitch - dPitch > limit) dPitch = pitch - limit;
      if (pitch - dPitch < -limit) dPitch = pitch + limit;
      const qx = new THREE.Quaternion().setFromAxisAngle(right, dPitch);
      q = qx.multiply(qz);
      cam.up.set(0, 0, 1);
    } else {
      const up = new THREE.Vector3().setFromMatrixColumn(cam.matrixWorld, 1);
      const right = new THREE.Vector3().setFromMatrixColumn(cam.matrixWorld, 0);
      const qa = new THREE.Quaternion().setFromAxisAngle(up, -dx * this.rotateSpeed);
      const qb = new THREE.Quaternion().setFromAxisAngle(right, -dy * this.rotateSpeed);
      q = qa.multiply(qb);
      cam.up.applyQuaternion(q);
    }
    offset.applyQuaternion(q);
    targetOffset.applyQuaternion(q);
    cam.position.copy(this.pivot).add(offset);
    this.target.copy(this.pivot).add(targetOffset);
    cam.quaternion.premultiply(q);
    cam.updateMatrixWorld();
  }

  private pan(e: PointerEvent, dx: number, dy: number) {
    const cam = this.camera;
    const ndcNow = this.ndc(e);
    const r = this.dom.getBoundingClientRect();
    const dNdc = new THREE.Vector2((2 * dx) / r.width, (-2 * dy) / r.height);
    const prev = new THREE.Vector3(ndcNow.x - dNdc.x, ndcNow.y - dNdc.y, this.panPlaneDepth).unproject(cam);
    const now = new THREE.Vector3(ndcNow.x, ndcNow.y, this.panPlaneDepth).unproject(cam);
    const delta = prev.sub(now);
    cam.position.add(delta);
    this.target.add(delta);
    cam.updateMatrixWorld();
  }

  private onWheel = (e: WheelEvent) => {
    if (!this.enabled) return;
    e.preventDefault();
    this.startInteraction();
    window.clearTimeout(this.wheelTimer);
    this.wheelTimer = window.setTimeout(() => this.endInteraction(), 250);
    const factor = Math.exp(e.deltaY * this.zoomSpeed * (e.deltaMode === 1 ? 30 : 1));
    const ndc = this.ndc(e);
    const hit = this.opts.pick(ndc);
    this.zoomAt(factor, ndc, hit);
    this.opts.onChange();
  };

  zoomAt(factor: number, ndc: THREE.Vector2, hit: THREE.Vector3 | null) {
    const cam = this.camera;
    if (cam instanceof THREE.OrthographicCamera) {
      const before = new THREE.Vector3(ndc.x, ndc.y, 0).unproject(cam);
      cam.zoom = THREE.MathUtils.clamp(cam.zoom / factor, 1e-4, 1e6);
      cam.updateProjectionMatrix();
      const after = new THREE.Vector3(ndc.x, ndc.y, 0).unproject(cam);
      const d = before.sub(after);
      cam.position.add(d);
      this.target.add(d);
    } else {
      let anchor = hit;
      if (!anchor) {
        // zoom towards the point on the ray at the target's depth
        const dist = cam.position.distanceTo(this.target);
        const ray = new THREE.Raycaster();
        ray.setFromCamera(ndc, cam);
        anchor = ray.ray.at(dist, new THREE.Vector3());
      }
      const toCam = cam.position.clone().sub(anchor);
      const newPos = anchor.clone().add(toCam.multiplyScalar(factor));
      const delta = newPos.clone().sub(cam.position);
      cam.position.copy(newPos);
      // move the target along so orbit distance shrinks/grows consistently
      const tToAnchor = this.target.clone().sub(anchor);
      this.target.copy(anchor).add(tToAnchor.multiplyScalar(factor));
      void delta;
    }
    cam.updateMatrixWorld();
  }

  /** Look at a bounding sphere from a direction (unit vector from target towards the camera). */
  fitSphere(center: THREE.Vector3, radius: number, viewDir?: THREE.Vector3, up?: THREE.Vector3) {
    const cam = this.camera;
    const dir = (viewDir ?? cam.position.clone().sub(this.target).normalize()).clone().normalize();
    if (up) cam.up.copy(up);
    else cam.up.set(0, 0, 1);
    if (Math.abs(dir.dot(cam.up)) > 0.999) cam.up.set(0, 1, 0);
    const r = Math.max(radius, 1e-3);
    if (cam instanceof THREE.PerspectiveCamera) {
      const fov = THREE.MathUtils.degToRad(cam.fov);
      const fovMin = Math.min(fov, 2 * Math.atan(Math.tan(fov / 2) * cam.aspect));
      const dist = (r / Math.sin(fovMin / 2)) * 1.05;
      cam.position.copy(center).addScaledVector(dir, dist);
      cam.near = Math.max(dist / 1000, 0.01);
      cam.far = dist * 100 + r * 10;
      cam.updateProjectionMatrix();
    } else {
      cam.position.copy(center).addScaledVector(dir, r * 4);
      const w = cam.right - cam.left, h = cam.top - cam.bottom;
      cam.zoom = Math.min(w, h) / (2.1 * r);
      cam.near = -r * 100;
      cam.far = r * 100;
      cam.updateProjectionMatrix();
    }
    this.target.copy(center);
    cam.lookAt(center);
    cam.updateMatrixWorld();
    this.opts.onChange();
  }
}
