/**
 * 3D sketch: draw smooth curves directly on the scan. Clicked points are interpolated with a
 * cubic spline whose samples are snapped onto the mesh, so the curve follows the surface.
 */
import type { App } from "../app/App";
import { newId } from "../core/document";
import type { Curve3D, MeshEntity, Sketch3DEntity } from "../core/entities";
import type { Vec3 } from "../core/types";
import { evalCurve, fitBSplineCurve, interpolateBSplineCurve } from "../fit/bspline";
import { Tool, type ToolPointerEvent } from "./Tool";
import { Overlay } from "./overlay";
import { Section, Check, Stats, int } from "../ui/widgets";
import { fatSegments, lineMaterial, pointSprites, polylinesToPairs } from "../viewer/draw";

interface DraftCurve {
  ctrl: Vec3[];
  closed: boolean;
}

interface S3State {
  curves: DraftCurve[];
  current: DraftCurve | null;
  snap: boolean;
}

class Sketch3DTool extends Tool<S3State> {
  id = "sketch3d";
  label = "3D Sketch";
  hint = "Click points on the scan to draw a curve. Click the first point to close it, double-click or press Enter to finish a curve. Backspace removes the last point.";
  okLabel = "Finish";
  private overlay = new Overlay(this.app);
  private matCurve = lineMaterial({ color: "#ff4fa3", width: 3, depthTest: false });
  private matDraft = lineMaterial({ color: "#ffd1e8", width: 2, dashed: true, dashSize: 2, depthTest: false });
  private meshId: string;
  private lastClick = 0;
  private hover: Vec3 | null = null;

  constructor(app: App, private editId?: string) {
    super(app, { curves: [], current: null, snap: true });
    const m = app.activeMesh();
    if (!m) throw new Error("Import a mesh first");
    this.meshId = m.id;
    if (editId) {
      const e = app.doc.get<Sketch3DEntity>(editId);
      if (e) {
        this.meshId = e.meshId ?? m.id;
        // reconstruct editable control points by resampling each stored curve
        this.set({
          curves: e.curves.map((c) => ({ ctrl: resample(c.points, Math.min(12, Math.max(3, Math.round(c.points.length / 20)))), closed: c.closed })),
        });
      }
    }
  }

  activate() {
    if (this.editId) this.app.sync?.view<{ setVisible(v: boolean): void }>(this.editId)?.setVisible(false);
    this.redraw();
  }

  private pickOnMesh(e: ToolPointerEvent): Vec3 | null {
    const hit = e.pick((id) => id === this.meshId || this.app.doc.get(id)?.kind === "mesh");
    return hit ? hit.point : null;
  }

  onPointerDown(e: ToolPointerEvent): boolean {
    if (e.button !== 0) return false;
    const p = this.pickOnMesh(e);
    const now = performance.now();
    const dbl = now - this.lastClick < 320;
    this.lastClick = now;
    if (dbl) {
      this.finishCurrent(false);
      return true;
    }
    if (!p) return true;
    const cur = this.s.current;
    if (cur && cur.ctrl.length >= 3) {
      // close when clicking near the first point
      const first = cur.ctrl[0];
      const px = this.app.viewer!.worldToScreen(first);
      if (Math.hypot(px.x - e.x, px.y - e.y) < 10) {
        this.finishCurrent(true);
        return true;
      }
    }
    this.set({ current: { ctrl: [...(cur?.ctrl ?? []), p], closed: false } });
    this.redraw();
    return true;
  }

  onPointerMove(e: ToolPointerEvent): boolean {
    if (!this.s.current) return false;
    this.hover = this.pickOnMesh(e);
    this.redraw();
    return false;
  }

  onKeyDown(e: KeyboardEvent): boolean {
    if (e.key === "Enter" && this.s.current) {
      this.finishCurrent(false);
      return true;
    }
    if (e.key === "Backspace" && this.s.current) {
      const ctrl = this.s.current.ctrl.slice(0, -1);
      this.set({ current: ctrl.length ? { ...this.s.current, ctrl } : null });
      this.redraw();
      return true;
    }
    if (e.key === "Escape" && this.s.current) {
      this.set({ current: null });
      this.redraw();
      return true;
    }
    return false;
  }

  private finishCurrent(closed: boolean) {
    const cur = this.s.current;
    if (cur && cur.ctrl.length >= 2) this.set({ curves: [...this.s.curves, { ctrl: cur.ctrl, closed }], current: null });
    else this.set({ current: null });
    this.redraw();
  }

  /** Dense on-surface polyline for a set of control points. */
  private build(c: DraftCurve): Curve3D | null {
    if (c.ctrl.length < 2) return null;
    const ctrl = c.closed ? [...c.ctrl, c.ctrl[0]] : c.ctrl;
    const deg = Math.min(3, ctrl.length - 1);
    const spline = interpolateBSplineCurve(ctrl, deg);
    const t0 = spline.knots[deg], t1 = spline.knots[spline.knots.length - deg - 1];
    let len = 0;
    for (let i = 1; i < ctrl.length; i++) len += Math.hypot(ctrl[i][0] - ctrl[i - 1][0], ctrl[i][1] - ctrl[i - 1][1], ctrl[i][2] - ctrl[i - 1][2]);
    const n = Math.max(16, Math.min(2000, Math.round(len / 0.5)));
    const rt = this.app.sync?.meshRuntime(this.meshId);
    const pts: Vec3[] = [];
    for (let i = 0; i <= n; i++) {
      let p = evalCurve(spline, t0 + ((t1 - t0) * i) / n);
      if (this.s.snap && rt) {
        const hit = rt.spatial.closestPoint(p, len * 0.25 + 1);
        if (hit) p = hit.point;
      }
      pts.push(p);
    }
    if (c.closed) pts.pop();
    let bspline;
    try {
      bspline = fitBSplineCurve(c.closed ? [...pts, pts[0]] : pts, { nCtrl: Math.max(4, Math.min(60, Math.round(n / 8))), degree: 3 }).curve;
    } catch {
      bspline = undefined;
    }
    return { points: pts, closed: c.closed, bspline };
  }

  private redraw() {
    this.overlay.clear();
    const done = this.s.curves.map((c) => this.build(c)).filter(Boolean) as Curve3D[];
    if (done.length) this.overlay.add(fatSegments(polylinesToPairs(done), this.matCurve)).renderOrder = 30;
    const cur = this.s.current;
    if (cur) {
      const draft = { ctrl: this.hover ? [...cur.ctrl, this.hover] : cur.ctrl, closed: false };
      const b = this.build(draft);
      if (b) this.overlay.add(fatSegments(polylinesToPairs([b]), this.matDraft)).renderOrder = 31;
    }
    const ctrl = [...this.s.curves.flatMap((c) => c.ctrl), ...(cur?.ctrl ?? [])];
    if (ctrl.length) this.overlay.add(pointSprites(ctrl.flat(), "#ffffff", 7, { depthTest: false })).renderOrder = 32;
  }

  renderPanel() {
    const s = this.s;
    return (
      <Section title="Curves">
        <Stats rows={[["Finished curves", int(s.curves.length)], ["Current points", int(s.current?.ctrl.length ?? 0)]]} />
        <div style={{ height: 8 }} />
        <Check checked={s.snap} onChange={(snap) => { this.set({ snap }); this.redraw(); }}>
          Snap curves onto the scan
        </Check>
        <div className="btn-row" style={{ marginTop: 6 }}>
          <button className="btn small" disabled={!s.current} onClick={() => this.finishCurrent(false)}>
            Finish curve
          </button>
          <button className="btn small" disabled={!s.curves.length} onClick={() => { this.set({ curves: s.curves.slice(0, -1) }); this.redraw(); }}>
            Remove last curve
          </button>
        </div>
      </Section>
    );
  }

  async commit() {
    if (this.s.current) this.finishCurrent(false);
    const curves = this.s.curves.map((c) => this.build(c)).filter(Boolean) as Curve3D[];
    if (!curves.length) {
      this.app.toast("error", "Draw at least one curve");
      return false;
    }
    if (this.editId) {
      this.app.doc.transact("Edit 3D sketch", () => this.app.doc.update<Sketch3DEntity>(this.editId!, { curves }));
      return;
    }
    this.app.doc.add<Sketch3DEntity>({ id: newId("k"), kind: "sketch3d", name: this.app.doc.uniqueName("3D sketch"), visible: true, curves, meshId: this.meshId });
  }

  deactivate() {
    if (this.editId) this.app.sync?.view<{ setVisible(v: boolean): void }>(this.editId)?.setVisible(this.app.doc.get(this.editId)?.visible ?? true);
    this.overlay.dispose();
    this.matCurve.dispose();
    this.matDraft.dispose();
  }
}

function resample(points: Vec3[], n: number): Vec3[] {
  if (points.length <= n) return points.slice();
  const out: Vec3[] = [];
  for (let i = 0; i < n; i++) out.push(points[Math.round((i / (n - 1)) * (points.length - 1))]);
  return out;
}

export function startSketch3D(app: App, editId?: string): void {
  app.startTool(new Sketch3DTool(app, editId));
}

void ({} as MeshEntity);
