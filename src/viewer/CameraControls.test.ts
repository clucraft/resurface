import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { CameraControls } from "./CameraControls";

const dom = {
  addEventListener() {},
  removeEventListener() {},
  getBoundingClientRect: () => ({ left: 0, top: 0, width: 800, height: 600 }),
} as unknown as HTMLElement;

function setup(dir = new THREE.Vector3(1, -1, 0.8), up?: THREE.Vector3) {
  const cam = new THREE.PerspectiveCamera(45, 800 / 600, 0.1, 1000);
  const c = new CameraControls(cam, dom, { pick: () => null, onChange: () => {} });
  c.fitSphere(new THREE.Vector3(), 10, dir, up);
  const orbit = (dx: number, dy: number) => (c as unknown as { orbit(dx: number, dy: number): void }).orbit(dx, dy);
  const upZ = () => new THREE.Vector3().setFromMatrixColumn(cam.matrixWorld, 1).z;
  const pitch = () => Math.asin(cam.getWorldDirection(new THREE.Vector3()).z);
  return { cam, orbit, upZ, pitch };
}

describe("CameraControls turntable orbit", () => {
  it("stops at the poles instead of flipping over, and comes back", () => {
    const { orbit, upZ, pitch } = setup();
    // drag down far past looking straight down…
    for (let i = 0; i < 100; i++) {
      orbit(3, 25);
      expect(upZ()).toBeGreaterThan(0);
    }
    expect(pitch()).toBeCloseTo(-(Math.PI / 2 - 0.01), 3);
    // …then all the way up past looking straight up, and back to level
    for (let i = 0; i < 100; i++) {
      orbit(-2, -25);
      expect(upZ()).toBeGreaterThan(0);
    }
    expect(pitch()).toBeCloseTo(Math.PI / 2 - 0.01, 3);
    for (let i = 0; i < 10; i++) orbit(0, 15);
    expect(Math.abs(pitch() - (Math.PI / 2 - 0.01 - 150 * 0.006))).toBeLessThan(1e-6);
  });

  it("levels a camera left upside down (e.g. by a sketch view) on the next drag", () => {
    const { cam, orbit, upZ } = setup();
    cam.up.set(0, 0, -1);
    cam.lookAt(0, 0, 0);
    cam.updateMatrixWorld();
    expect(upZ()).toBeLessThan(0);
    orbit(5, 5);
    expect(upZ()).toBeGreaterThan(0);
  });

  it("orbits smoothly out of the top and bottom standard views", () => {
    for (const [dir, up] of [
      [new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 1, 0)],
      [new THREE.Vector3(0, 0, -1), new THREE.Vector3(0, -1, 0)],
    ]) {
      const { cam, orbit } = setup(dir, up);
      const screenUp = new THREE.Vector3().setFromMatrixColumn(cam.matrixWorld, 1);
      orbit(4, 0);
      // no sudden spin: the screen's up direction barely moves on a small drag
      expect(new THREE.Vector3().setFromMatrixColumn(cam.matrixWorld, 1).angleTo(screenUp)).toBeLessThan(0.1);
    }
  });
});
