/** Analysis tools: CAD-vs-scan deviation map, curvature map and measurements. */
import type { App } from "../app/App";
import type { MeshEntity, PrimitiveEntity, RefLineEntity } from "../core/entities";
import type { LineDef, MeshData, PlaneDef, Vec3 } from "../core/types";
import { compute } from "../app/compute";
import { Tool, type ToolClass, type ToolPointerEvent } from "./Tool";
import { Overlay } from "./overlay";
import { Section, Field, NumberInput, Segmented, Stats, Check, Select, Slider, mm, int } from "../ui/widgets";
import { deviationColors, robustRange, sequentialColors } from "../analysis/colormap";
import type { MeshView } from "../viewer/MeshView";
import type { BodyView } from "../viewer/views";
import { fatSegments, lineMaterial, pointSprites } from "../viewer/draw";
import { angleBetweenLines, angleBetweenPlanes, axisOf, distancePointLine, distancePointPlane } from "../fit/reference";

function resetMeshColors(app: App, meshId: string) {
  const mv = app.sync?.view<MeshView>(meshId);
  const e = app.doc.get<MeshEntity>(meshId);
  if (mv && e) {
    mv.setColorMode("solid", null);
    app.applyMeshDisplay(mv, e);
  }
}

/** Merge the tessellations of all visible bodies into one triangle mesh. */
function bodiesMesh(app: App): MeshData | null {
  const parts: { positions: Float32Array; indices: Uint32Array; normals: Float32Array }[] = [];
  for (const b of app.doc.ofKind("body")) {
    if (!b.visible) continue;
    const t = app.sync?.view<BodyView>(b.id)?.tess;
    if (t) parts.push(t);
  }
  if (!parts.length) return null;
  let nv = 0, nt = 0;
  for (const p of parts) {
    nv += p.positions.length / 3;
    nt += p.indices.length / 3;
  }
  const positions = new Float32Array(nv * 3), normals = new Float32Array(nv * 3), indices = new Uint32Array(nt * 3);
  let vo = 0, to = 0;
  for (const p of parts) {
    positions.set(p.positions, vo * 3);
    normals.set(p.normals, vo * 3);
    for (let i = 0; i < p.indices.length; i++) indices[to * 3 + i] = p.indices[i] + vo;
    vo += p.positions.length / 3;
    to += p.indices.length / 3;
  }
  return { positions, indices, normals };
}

// ---------------------------------------------------------------------------
// Deviation
// ---------------------------------------------------------------------------

interface DeviationState {
  target: "scan" | "cad";
  range: number;
  autoRange: boolean;
  tolerance: number;
  maxDistance: number;
  stats: { rms: number; mean: number; max: number; min: number; within: number; count: number; total: number } | null;
  running: boolean;
}

class DeviationTool extends Tool<DeviationState> {
  id = "deviation";
  label = "Deviation Analysis";
  hint = "Distance between the CAD bodies and the scan. Green = within tolerance, red = material above the CAD surface, blue = below.";
  okLabel: string | null = null;
  private meshId: string;
  private values: Float32Array | null = null;
  private bodyColored: string[] = [];

  constructor(app: App) {
    const size = app.sceneSize();
    super(app, { target: "scan", range: 0.5, autoRange: true, tolerance: 0.1, maxDistance: Math.max(1, Math.round(size * 0.02)), stats: null, running: false });
    const m = app.activeMesh();
    if (!m) throw new Error("Import a mesh first");
    this.meshId = m.id;
    if (!app.doc.ofKind("body").some((b) => b.visible)) throw new Error("Create (and show) at least one CAD body first");
  }

  activate() {
    void this.run();
  }

  async run() {
    const mesh = this.app.doc.get<MeshEntity>(this.meshId)?.mesh;
    const cad = bodiesMesh(this.app);
    if (!mesh || !cad) {
      this.app.toast("error", "No tessellated bodies to compare against yet — wait for them to finish building");
      return;
    }
    this.set({ running: true });
    this.clearColors();
    try {
      let values: Float32Array;
      if (this.s.target === "scan") {
        // split scan vertices across the worker pool
        const nv = mesh.positions.length / 3;
        const parts = Math.max(1, Math.min(compute.size, Math.ceil(nv / 50000)));
        const step = Math.ceil(nv / parts);
        const jobs: Promise<Float32Array>[] = [];
        for (let i = 0; i < parts; i++) {
          const start = i * step, end = Math.min(nv, start + step);
          jobs.push(compute.run<Float32Array>("deviation", { points: mesh.positions, start, end, target: cad, maxDistance: this.s.maxDistance }));
        }
        const res = await this.app.busy("Computing deviation", () => Promise.all(jobs));
        if (!res) return;
        values = new Float32Array(nv);
        let o = 0;
        for (const r of res) {
          values.set(r, o);
          o += r.length;
        }
      } else {
        const rt = this.app.sync!.meshRuntime(this.meshId)!;
        const n = cad.positions.length / 3;
        values = new Float32Array(n);
        const q = rt.spatial;
        for (let i = 0; i < n; i++) {
          const p: Vec3 = [cad.positions[3 * i], cad.positions[3 * i + 1], cad.positions[3 * i + 2]];
          const hit = q.closestPoint(p, this.s.maxDistance);
          if (!hit) {
            values[i] = NaN;
            continue;
          }
          // sign: scan above the CAD surface (along the CAD normal) is positive
          const n3: Vec3 = [cad.normals![3 * i], cad.normals![3 * i + 1], cad.normals![3 * i + 2]];
          const s = (hit.point[0] - p[0]) * n3[0] + (hit.point[1] - p[1]) * n3[1] + (hit.point[2] - p[2]) * n3[2];
          values[i] = s >= 0 ? hit.distance : -hit.distance;
        }
      }
      this.values = values;
      this.computeStats();
      if (this.s.autoRange) this.set({ range: Math.max(this.s.tolerance * 2, Number(robustRange(values, 0.99).toPrecision(2))) });
      this.applyColors();
    } finally {
      this.set({ running: false });
    }
  }

  private computeStats() {
    const v = this.values!;
    let n = 0, sum = 0, sq = 0, max = -Infinity, min = Infinity, within = 0;
    for (let i = 0; i < v.length; i++) {
      const x = v[i];
      if (!Number.isFinite(x)) continue;
      n++;
      sum += x;
      sq += x * x;
      if (x > max) max = x;
      if (x < min) min = x;
      if (Math.abs(x) <= this.s.tolerance) within++;
    }
    this.set({ stats: n ? { rms: Math.sqrt(sq / n), mean: sum / n, max, min, within: within / n, count: n, total: v.length } : null });
  }

  private applyColors() {
    const v = this.values;
    if (!v) return;
    const s = this.s;
    const colors = deviationColors(v, s.range, s.tolerance);
    if (s.target === "scan") {
      this.app.sync?.view<MeshView>(this.meshId)?.updateMapColors(colors);
    } else {
      // distribute over the bodies in the same order as bodiesMesh()
      let o = 0;
      this.bodyColored = [];
      for (const b of this.app.doc.ofKind("body")) {
        if (!b.visible) continue;
        const view = this.app.sync?.view<BodyView>(b.id);
        const t = view?.tess;
        if (!t || !view) continue;
        const n = t.positions.length / 3;
        view.setVertexColors(colors.slice(o * 3, (o + n) * 3));
        this.bodyColored.push(b.id);
        o += n;
      }
    }
    const st = s.stats;
    this.app.ui.patch({
      legend: {
        title: s.target === "scan" ? "Scan → CAD deviation" : "CAD → scan deviation",
        min: -s.range,
        max: s.range,
        tolerance: s.tolerance,
        stats: st ? [`RMS ${st.rms.toFixed(3)} mm`, `In tolerance ${(st.within * 100).toFixed(1)}%`] : undefined,
      },
    });
  }

  private clearColors() {
    resetMeshColors(this.app, this.meshId);
    for (const id of this.bodyColored) this.app.sync?.view<BodyView>(id)?.setVertexColors(null);
    this.bodyColored = [];
  }

  renderPanel() {
    const s = this.s;
    const st = s.stats;
    const meshes = this.app.doc.ofKind("mesh").map((m) => ({ value: m.id, label: m.name }));
    return (
      <>
        <Section title="Compare">
          <Field label="Scan">
            <Select value={this.meshId} options={meshes} onChange={(id) => { this.clearColors(); this.meshId = id; void this.run(); }} />
          </Field>
          <Field label="Colour">
            <Segmented value={s.target} options={[{ value: "scan", label: "Scan" }, { value: "cad", label: "CAD" }]} onChange={(target) => { this.set({ target }); void this.run(); }} />
          </Field>
          <Field label="Max distance">
            <NumberInput value={s.maxDistance} unit="mm" min={0.01} onChange={(maxDistance) => { this.set({ maxDistance }); void this.run(); }} />
          </Field>
          <div className="note">Points farther than this (unmodelled areas) are shown grey.</div>
        </Section>
        <Section title="Colour scale">
          <Field label="Tolerance ±">
            <NumberInput value={s.tolerance} unit="mm" min={0} step={0.01} onChange={(tolerance) => { this.set({ tolerance }); this.computeStats(); this.applyColors(); }} />
          </Field>
          <Field label="Range ±">
            <NumberInput value={s.range} unit="mm" min={0.001} step={0.05} onChange={(range) => { this.set({ range, autoRange: false }); this.applyColors(); }} />
          </Field>
          <Check checked={s.autoRange} onChange={(autoRange) => this.set({ autoRange })}>
            Auto range on recompute
          </Check>
          <button className="btn small" onClick={() => void this.run()} disabled={s.running}>
            Recompute
          </button>
        </Section>
        <Section title="Statistics">
          {st ? (
            <Stats
              rows={[
                ["Points compared", `${int(st.count)} / ${int(st.total)}`],
                ["Mean", mm(st.mean, 4)],
                ["RMS", mm(st.rms, 4), st.rms <= s.tolerance ? "good" : "warn"],
                ["Max +", mm(st.max, 4)],
                ["Max −", mm(st.min, 4)],
                ["Within ±tol", `${(st.within * 100).toFixed(1)} %`, st.within > 0.95 ? "good" : st.within > 0.8 ? "warn" : "bad"],
              ]}
            />
          ) : (
            <div className="note">{s.running ? "Computing…" : "No data"}</div>
          )}
        </Section>
      </>
    );
  }

  deactivate() {
    this.clearColors();
    this.app.ui.patch({ legend: null });
  }
}

// ---------------------------------------------------------------------------
// Curvature map
// ---------------------------------------------------------------------------

class CurvatureTool extends Tool<{ kind: "mean" | "signedMean"; scale: number; max: number }> {
  id = "curvatureMap";
  label = "Curvature Map";
  hint = "Colours the scan by curvature — useful to see fillets, feature lines and where to split regions.";
  okLabel: string | null = null;
  private meshId: string;
  private values: Float32Array | null = null;

  constructor(app: App) {
    super(app, { kind: "mean", scale: 1, max: 0 });
    const m = app.activeMesh();
    if (!m) throw new Error("Import a mesh first");
    this.meshId = m.id;
  }

  activate() {
    void this.run();
  }

  private async run() {
    const mesh = this.app.doc.get<MeshEntity>(this.meshId)?.mesh;
    if (!mesh) return;
    const v = await this.app.busy("Computing curvature", () => compute.run<Float32Array>("curvature", { mesh, kind: this.s.kind }));
    if (!v) return;
    this.values = v;
    this.set({ max: robustRange(v, 0.95) });
    this.apply();
  }

  private apply() {
    const v = this.values;
    if (!v) return;
    const m = this.s.max * this.s.scale;
    const colors = this.s.kind === "mean" ? sequentialColors(v, m) : deviationColors(v, m, 0);
    this.app.sync?.view<MeshView>(this.meshId)?.updateMapColors(colors);
    this.app.ui.patch({
      legend: {
        title: this.s.kind === "mean" ? "Curvature |H| (1/mm)" : "Mean curvature (1/mm)",
        min: this.s.kind === "mean" ? 0 : -m,
        max: m,
        stats: [`Radius at max ≈ ${(1 / Math.max(m, 1e-9)).toFixed(2)} mm`],
      },
    });
  }

  renderPanel() {
    const s = this.s;
    return (
      <Section title="Curvature">
        <Field label="Type">
          <Segmented value={s.kind} options={[{ value: "mean", label: "Magnitude" }, { value: "signedMean", label: "Signed" }]} onChange={(kind) => { this.set({ kind }); void this.run(); }} />
        </Field>
        <Field label="Contrast">
          <Slider value={s.scale} min={0.1} max={3} step={0.05} onChange={(scale) => { this.set({ scale }); this.apply(); }} format={(v) => `${v.toFixed(2)}×`} />
        </Field>
      </Section>
    );
  }

  deactivate() {
    resetMeshColors(this.app, this.meshId);
    this.app.ui.patch({ legend: null });
  }
}

// ---------------------------------------------------------------------------
// Measure
// ---------------------------------------------------------------------------

interface Measurement {
  label: string;
  value: string;
}

class MeasureTool extends Tool<{ mode: "points" | "refs"; picks: Vec3[]; a: string; b: string; results: Measurement[] }> {
  id = "measure";
  label = "Measure";
  hint = "Point mode: click two points on the model. Reference mode: choose two planes/axes/points to get distances and angles.";
  okLabel: string | null = null;
  private overlay = new Overlay(this.app);
  private mat = lineMaterial({ color: "#ffd23f", width: 2, depthTest: false });

  constructor(app: App) {
    super(app, { mode: "points", picks: [], a: "", b: "", results: [] });
  }

  onPointerDown(e: ToolPointerEvent): boolean {
    if (this.s.mode !== "points" || e.button !== 0) return false;
    const hit = e.pick();
    if (!hit) return true;
    const picks = this.s.picks.length >= 2 ? [hit.point] : [...this.s.picks, hit.point];
    this.set({ picks });
    this.draw();
    return true;
  }

  private draw() {
    this.overlay.clear();
    const p = this.s.picks;
    if (p.length) this.overlay.add(pointSprites(p.flat(), "#ffd23f", 9, { depthTest: false })).renderOrder = 40;
    if (p.length === 2) {
      this.overlay.add(fatSegments([...p[0], ...p[1]], this.mat)).renderOrder = 39;
      const d = Math.hypot(p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]);
      const mid: Vec3 = [(p[0][0] + p[1][0]) / 2, (p[0][1] + p[1][1]) / 2, (p[0][2] + p[1][2]) / 2];
      this.overlay.label(`${d.toFixed(3)} mm`, mid);
      this.set({
        results: [
          { label: "Distance", value: mm(d) },
          { label: "ΔX", value: mm(p[1][0] - p[0][0]) },
          { label: "ΔY", value: mm(p[1][1] - p[0][1]) },
          { label: "ΔZ", value: mm(p[1][2] - p[0][2]) },
        ],
      });
    }
  }

  private refGeom(id: string): { plane?: PlaneDef; line?: LineDef; point?: Vec3 } | null {
    const e = this.app.doc.get(id);
    if (!e) return null;
    if (e.kind === "primitive") {
      const d = (e as PrimitiveEntity).def;
      if (d.kind === "plane") return { plane: d };
      if (d.kind === "cylinder" || d.kind === "cone") return { line: axisOf(d) };
      if (d.kind === "sphere") return { point: d.center };
    }
    if (e.kind === "refline") return { line: (e as RefLineEntity).def };
    if (e.kind === "refpoint") return { point: e.def.position };
    return null;
  }

  private measureRefs() {
    const A = this.refGeom(this.s.a), B = this.refGeom(this.s.b);
    const out: Measurement[] = [];
    if (A && B) {
      const deg = (r: number) => `${((r * 180) / Math.PI).toFixed(3)}°`;
      if (A.plane && B.plane) {
        const ang = angleBetweenPlanes(A.plane, B.plane);
        out.push({ label: "Angle", value: deg(ang) });
        if (Math.abs(Math.sin(ang)) < 1e-3) out.push({ label: "Distance", value: mm(Math.abs(distancePointPlane(B.plane.origin, A.plane))) });
      } else if (A.line && B.line) {
        out.push({ label: "Angle", value: deg(angleBetweenLines(A.line, B.line)) });
        out.push({ label: "Distance (B origin → A)", value: mm(distancePointLine(B.line.origin, A.line)) });
      } else if ((A.plane && B.line) || (A.line && B.plane)) {
        const pl = (A.plane ?? B.plane)!, ln = (A.line ?? B.line)!;
        const c = Math.abs(ln.direction[0] * pl.normal[0] + ln.direction[1] * pl.normal[1] + ln.direction[2] * pl.normal[2]);
        out.push({ label: "Angle to plane", value: deg(Math.asin(Math.min(1, c))) });
        out.push({ label: "Distance (axis origin)", value: mm(Math.abs(distancePointPlane(ln.origin, pl))) });
      } else {
        const pt = (A.point ?? B.point)!;
        const other = A.point ? B : A;
        if (other.point) out.push({ label: "Distance", value: mm(Math.hypot(pt[0] - other.point[0], pt[1] - other.point[1], pt[2] - other.point[2])) });
        if (other.plane) out.push({ label: "Distance to plane", value: mm(Math.abs(distancePointPlane(pt, other.plane))) });
        if (other.line) out.push({ label: "Distance to axis", value: mm(distancePointLine(pt, other.line)) });
      }
    }
    this.set({ results: out });
  }

  renderPanel() {
    const s = this.s;
    const refs = this.app.doc.all().filter((e) => e.kind === "primitive" || e.kind === "refline" || e.kind === "refpoint").map((e) => ({ value: e.id, label: e.name }));
    return (
      <>
        <Section title="Mode">
          <Segmented value={s.mode} options={[{ value: "points", label: "Two points" }, { value: "refs", label: "References" }]} onChange={(mode) => { this.set({ mode, results: [], picks: [] }); this.draw(); }} />
          <div style={{ height: 8 }} />
          {s.mode === "refs" && (
            <>
              <Field label="A">
                <Select value={s.a} options={refs} placeholder="Choose…" onChange={(a) => { this.set({ a }); this.measureRefs(); }} />
              </Field>
              <Field label="B">
                <Select value={s.b} options={refs} placeholder="Choose…" onChange={(b) => { this.set({ b }); this.measureRefs(); }} />
              </Field>
            </>
          )}
          {s.mode === "points" && <div className="note">Click two points ({s.picks.length}/2).</div>}
        </Section>
        <Section title="Result">
          {s.results.length ? <Stats rows={s.results.map((r) => [r.label, r.value])} /> : <div className="note">—</div>}
        </Section>
      </>
    );
  }

  deactivate() {
    this.overlay.dispose();
    this.mat.dispose();
  }
}

export function startAnalysisTool(app: App, id: string): void {
  const map: Record<string, ToolClass> = {
    deviation: DeviationTool,
    curvatureMap: CurvatureTool,
    measure: MeasureTool,
  };
  const T = map[id];
  if (T) app.startTool(new T(app));
}

