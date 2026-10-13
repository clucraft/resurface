/**
 * Development-only helpers for driving the UI from scripts/automation (loaded in dev builds only):
 *   rsTest.click("Plane")          — click a button by its visible text
 *   rsTest.clickWorld([x,y,z])     — synthetic left click on the canvas at a world point
 *   rsTest.state()                 — compact summary of the document and UI state
 */
import type { Vec3 } from "../core/types";
import { app } from "../app/App";
import { compute } from "../app/compute";
import { markPrepared } from "../app/MeshRuntime";
import type { MeshData } from "../core/types";

function buttonsByText(text: string): HTMLElement[] {
  const all = [...document.querySelectorAll<HTMLElement>("button, .item")];
  const norm = (s: string) => s.replace(/\s+/g, " ").trim().toLowerCase();
  const t = norm(text);
  const exact = all.filter((b) => norm(b.innerText) === t);
  return exact.length ? exact : all.filter((b) => norm(b.innerText).includes(t));
}

function pointer(type: string, x: number, y: number, opts: { button?: number; shift?: boolean; ctrl?: boolean; alt?: boolean; buttons?: number } = {}) {
  const canvas = app.viewer!.domElement;
  const r = canvas.getBoundingClientRect();
  canvas.dispatchEvent(
    new PointerEvent(type, {
      clientX: r.left + x,
      clientY: r.top + y,
      button: opts.button ?? 0,
      buttons: opts.buttons ?? (type === "pointerup" ? 0 : 1),
      shiftKey: !!opts.shift,
      ctrlKey: !!opts.ctrl,
      altKey: !!opts.alt,
      bubbles: true,
      pointerId: 1,
      pointerType: "mouse",
    }),
  );
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export const rsTest = {
  app,
  sleep,
  click(text: string, index = 0): boolean {
    const b = buttonsByText(text)[index];
    if (!b) return false;
    b.click();
    return true;
  },
  /** screen position (canvas px) of a world point */
  toScreen(p: Vec3) {
    return app.viewer!.worldToScreen(p);
  },
  async clickScreen(x: number, y: number, opts: { shift?: boolean; ctrl?: boolean; alt?: boolean } = {}) {
    pointer("pointermove", x, y, { buttons: 0, alt: opts.alt });
    pointer("pointerdown", x, y, opts);
    await sleep(10);
    pointer("pointerup", x, y, opts);
  },
  async clickWorld(p: Vec3, opts: { shift?: boolean; ctrl?: boolean; alt?: boolean } = {}) {
    app.viewer!.renderNow();
    const s = app.viewer!.worldToScreen(p);
    await rsTest.clickScreen(s.x, s.y, opts);
    return s;
  },
  async drag(from: [number, number], to: [number, number], steps = 8, opts: { shift?: boolean; ctrl?: boolean } = {}) {
    pointer("pointerdown", from[0], from[1], opts);
    for (let i = 1; i <= steps; i++) {
      const t = i / steps;
      pointer("pointermove", from[0] + (to[0] - from[0]) * t, from[1] + (to[1] - from[1]) * t, opts);
      await sleep(5);
    }
    pointer("pointerup", to[0], to[1], opts);
  },
  key(key: string, opts: { ctrl?: boolean; shift?: boolean } = {}) {
    window.dispatchEvent(new KeyboardEvent("keydown", { key, ctrlKey: !!opts.ctrl, shiftKey: !!opts.shift, bubbles: true }));
  },
  async waitIdle(timeoutMs = 60000) {
    const t0 = performance.now();
    await sleep(50);
    while (app.ui.get().busy && performance.now() - t0 < timeoutMs) await sleep(50);
  },
  /** Generate a large synthetic scan and report timings of the main interactive paths. */
  async bigMesh(triangles: number) {
    const t: Record<string, number> = {};
    let t0 = performance.now();
    const res = await compute.run<{ mesh: MeshData; timings: Record<string, number> }>("bigTest", { triangles });
    t.workerTotal = performance.now() - t0;
    Object.assign(t, res.timings);
    markPrepared(res.mesh);
    t0 = performance.now();
    const e = app.addMesh(`Big ${Math.round(triangles / 1e6)}M`, res.mesh, "perf");
    t.addEntity = performance.now() - t0;
    app.viewer!.fitAll("iso");
    t0 = performance.now();
    app.viewer!.renderNow();
    (app.viewer!.renderer.getContext() as WebGL2RenderingContext).finish();
    t.firstRender = performance.now() - t0;
    t0 = performance.now();
    app.viewer!.renderNow();
    (app.viewer!.renderer.getContext() as WebGL2RenderingContext).finish();
    t.secondRender = performance.now() - t0;
    const rt = app.sync!.meshRuntime(e.id)!;
    t0 = performance.now();
    await rt.indexed;
    t.backgroundIndexing = performance.now() - t0;
    t0 = performance.now();
    void rt.spatial;
    t.bvhOnUiThread = performance.now() - t0;
    t0 = performance.now();
    const hit = rt.raycast([0, -300, 0], [0, 1, 0]);
    t.raycast = performance.now() - t0;
    t0 = performance.now();
    const tris = rt.spatial.trianglesInSphere(hit!.point, 5);
    t.sphereQuery = performance.now() - t0;
    t0 = performance.now();
    rt.applyTriangles(tris, "add");
    rt.flush();
    t.selectionUpdate = performance.now() - t0;
    t0 = performance.now();
    void rt.neighbors;
    t.neighbors = performance.now() - t0;
    return { triangles: rt.triangleCount, chunks: rt.chunks.length, selected: tris.length, ms: Object.fromEntries(Object.entries(t).map(([k, v]) => [k, Math.round(v)])) };
  },
  state() {
    const ui = app.ui.get();
    return {
      entities: app.doc.all().map((e) => `${e.kind}:${e.name}${e.visible ? "" : " (hidden)"}`),
      tool: ui.toolId,
      selected: ui.selectedTriangles,
      status: ui.status,
      toasts: ui.toasts.map((t) => `${t.kind}: ${t.text}`),
      busy: ui.busy?.message ?? null,
      errors: app.bodyErrors.get(),
    };
  },
};

(globalThis as { rsTest?: typeof rsTest }).rsTest = rsTest;
