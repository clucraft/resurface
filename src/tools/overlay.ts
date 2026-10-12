/** Transient tool graphics in the viewer's overlay layer. */
import * as THREE from "three";
import type { App } from "../app/App";
import type { Label } from "../viewer/Viewer";
import { disposeObject } from "../viewer/draw";

export class Overlay {
  readonly group = new THREE.Group();
  private labels: Label[] = [];

  constructor(private app: App) {
    app.viewer?.layers.overlay.add(this.group);
  }

  add<T extends THREE.Object3D>(o: T): T {
    this.group.add(o);
    this.app.viewer?.requestRender();
    return o;
  }

  label(html: string, pos: [number, number, number], cls?: string): Label | null {
    const l = this.app.viewer?.addLabel(html, pos, cls) ?? null;
    if (l) this.labels.push(l);
    return l;
  }

  clear(): void {
    for (const c of [...this.group.children]) {
      this.group.remove(c);
      disposeObject(c);
    }
    for (const l of this.labels) this.app.viewer?.removeLabel(l);
    this.labels = [];
    this.app.viewer?.requestRender();
  }

  dispose(): void {
    this.clear();
    this.app.viewer?.layers.overlay.remove(this.group);
  }
}
