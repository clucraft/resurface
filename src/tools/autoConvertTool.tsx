/**
 * Auto Convert: one-click STL → CAD. The scan/STL is segmented into planes, cylinders, cones,
 * spheres and tori; the kernel builds a solid from them (regions it cannot build validly fall
 * back to facets). The preview is coloured by face type and shown against the mesh in the
 * compare split view; "Create body" adds it as a CAD body ready for STEP export.
 */
import * as THREE from "three";
import type { App } from "../app/App";
import { compute } from "../app/compute";
import { newId } from "../core/document";
import type { BodyEntity, MeshEntity } from "../core/entities";
import type { ConvertResult, FaceKind } from "../convert";
import { defaultTolerance, regionRepairs } from "../convert";
import { conversionMapColors, FACE_KIND_COLOR, FACE_KIND_ORDER, FACE_KIND_PLURAL } from "../convert/colors";
import type { PlanBuildResult, Tessellation } from "../cad/kernelTypes";
import { Tool } from "./Tool";
import { Overlay } from "./overlay";
import { Check, Field, NumberInput, Section, Stats, int, mm } from "../ui/widgets";
import { fatSegments, lineMaterial } from "../viewer/draw";

/** beyond this many unrecognised triangles the faceted B-rep gets too heavy to be useful */
const MAX_FACETS = 30_000;
/** …and beyond this many faces in total */
const MAX_FACES = 20_000;

interface ConvertSummary {
  faces: number;
  kinds: Partial<Record<FaceKind, number>>;
  facetedTriangles: number;
  triangles: number;
  coverage: number;
  maxDeviation: number;
  solid: boolean;
  valid: boolean;
  closed: boolean;
  volume?: number;
  demoted: number;
  seconds: number;
}

interface AutoConvertState {
  tolerance: number;
  autoTolerance: number;
  running: boolean;
  result: ConvertSummary | null;
  error: string | null;
  showMap: boolean;
}

export class AutoConvertTool extends Tool<AutoConvertState> {
  id = "autoConvert";
  label = "Auto Convert";
  hint = "Turns the mesh into a CAD solid of planes, cylinders, cones, spheres and tori. Drag the divider to compare STL and CAD.";
  okLabel = "Create body";
  private overlay = new Overlay(this.app);
  private faceMat = new THREE.MeshStandardMaterial({
    color: "#c9d3e0",
    roughness: 0.45,
    metalness: 0.12,
    side: THREE.DoubleSide,
    polygonOffset: true,
    polygonOffsetFactor: -1,
    polygonOffsetUnits: -2,
  });
  private edgeMat = lineMaterial({ color: "#1d2733", width: 1.3 });
  private meshId: string;
  private shapeId: number | null = null;
  private faceKinds: FaceKind[] = [];
  private tess: Tessellation | null = null;
  private runToken = 0;

  constructor(app: App) {
    const m = app.activeMesh();
    if (!m) throw new Error("Import an STL or scan first");
    const tol = defaultTolerance(m.mesh);
    super(app, { tolerance: tol, autoTolerance: tol, running: false, result: null, error: null, showMap: true });
    this.meshId = m.id;
  }

  activate() {
    void this.run();
  }

  private async releaseShape() {
    if (this.shapeId === null) return;
    const id = this.shapeId;
    this.shapeId = null;
    void (await this.app.getKernel()).release(id);
  }

  private async run() {
    const ent = this.app.doc.get<MeshEntity>(this.meshId);
    if (!ent || ent.kind !== "mesh") return;
    const token = ++this.runToken;
    this.set({ running: true, error: null });
    await this.releaseShape();
    this.overlay.clear();
    const t0 = performance.now();
    const tolerance = this.s.tolerance;
    let failure: string | null = null;
    const pipeline = async () => {
      const k = await this.app.getKernel();
      let res = await compute.run<ConvertResult>("convert", { mesh: ent.mesh, tolerance }, { onProgress: (f, m) => this.app.setProgress(0.8 * f, m) });
      if (res.stats.facetedTriangles > MAX_FACETS || res.stats.planFaces > MAX_FACES)
        throw new Error(
          `This mesh breaks into ${res.stats.planFaces.toLocaleString()} pieces (${res.stats.facetedTriangles.toLocaleString()} triangles match no plane, cylinder, cone, sphere or torus) — too fragmented for a useful CAD model. ` +
            "If this is a scan, raise the fit tolerance (try 0.05–0.1 mm); organic shapes convert better with Freeform → Autosurface.",
        );
      this.app.setProgress(0.82, "Building the CAD model");
      let built: PlanBuildResult = await k.brepFromPlan(res.plan);
      let demoted = 0;
      // regions the kernel could not build validly: big curved ones are retried in two halves,
      // the rest fall back to facets
      for (let pass = 0; pass < 4; pass++) {
        const { split, demote } = regionRepairs(res.plan, res.regions, built);
        if (!split.length && !demote.length) break;
        demoted += demote.length;
        void k.release(built.id);
        this.app.setProgress(0.85, split.length ? "Rebuilding difficult areas" : "Rebuilding difficult areas as facets");
        res = await compute.run<ConvertResult>("convert", { mesh: ent.mesh, prev: res, demote, split });
        built = await k.brepFromPlan(res.plan);
      }
      this.app.setProgress(0.92, "Checking the solid");
      const info = await k.info(built.id);
      const size = Math.max(1e-6, Math.hypot(...info.bbox.max.map((v, i) => v - info.bbox.min[i])));
      let tess: Tessellation;
      try {
        tess = await k.tessellate(built.id, { linearDeflection: size * 4e-4, angularDeflectionDeg: 10 });
      } catch {
        void k.release(built.id);
        throw new Error(
          `The converted model (${info.faces.toLocaleString()} faces) is too fragmented to display. ` +
            "If this is a scan, raise the fit tolerance (try 0.05–0.1 mm); organic shapes convert better with Freeform → Autosurface.",
        );
      }
      return { res, built, info, tess, demoted };
    };
    try {
      const out = await this.app.busy("Converting to CAD", async () => {
        try {
          return await pipeline();
        } catch (e) {
          failure = e instanceof Error ? e.message : String(e);
          throw e;
        }
      });
      if (token !== this.runToken) {
        if (out) void (await this.app.getKernel()).release(out.built.id);
        return;
      }
      if (!out) {
        this.set({ running: false, error: failure ?? "The conversion failed." });
        return;
      }
      const { res, built, info, tess, demoted } = out;
      this.shapeId = built.id;
      this.faceKinds = built.faceKinds;
      this.tess = tess;
      const kinds: Partial<Record<FaceKind, number>> = {};
      for (const fk of built.faceKinds) kinds[fk] = (kinds[fk] ?? 0) + 1;
      const result: ConvertSummary = {
        faces: info.faces,
        kinds,
        facetedTriangles: res.stats.facetedTriangles,
        triangles: res.stats.triangles,
        coverage: res.stats.coverage,
        maxDeviation: res.stats.maxDeviation,
        solid: info.solids > 0,
        valid: info.valid,
        closed: info.closed,
        volume: info.volume,
        demoted,
        seconds: (performance.now() - t0) / 1000,
      };
      this.set({ result, running: false });
      this.redraw();
      this.app.setCompare(true);
      const shape = result.solid && result.valid ? "a valid solid" : result.closed ? "a closed shell" : "an open shell";
      this.app.setStatus(`Auto Convert: ${result.faces} faces, ${shape}, in ${result.seconds.toFixed(1)} s — drag the divider to compare`);
    } catch (e) {
      if (token === this.runToken) this.set({ running: false, error: e instanceof Error ? e.message : String(e) });
    } finally {
      if (token === this.runToken && this.s.running) this.set({ running: false });
    }
  }

  private redraw() {
    this.overlay.clear();
    const t = this.tess;
    if (!t) return;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(t.positions, 3));
    g.setAttribute("normal", new THREE.BufferAttribute(t.normals, 3));
    g.setIndex(new THREE.BufferAttribute(t.indices, 1));
    if (this.s.showMap) {
      g.setAttribute("color", new THREE.BufferAttribute(conversionMapColors(t, this.faceKinds), 3, true));
      this.faceMat.vertexColors = true;
      this.faceMat.color.set("#ffffff");
    } else {
      this.faceMat.vertexColors = false;
      this.faceMat.color.set("#c9d3e0");
    }
    this.faceMat.needsUpdate = true;
    const group = new THREE.Group();
    group.userData.compareSide = "cad";
    group.add(new THREE.Mesh(g, this.faceMat));
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
    if (pairs.length) group.add(fatSegments(pairs, this.edgeMat));
    this.overlay.add(group);
  }

  renderPanel() {
    const s = this.s;
    const r = s.result;
    return (
      <>
        <Section title="Conversion">
          <Field label="Fit tolerance">
            <NumberInput value={s.tolerance} unit="mm" min={1e-5} step={s.autoTolerance / 4} digits={4} onChange={(tolerance) => this.set({ tolerance })} />
          </Field>
          <div className="note">How far a surface may stray from the mesh. The default follows the model size; raise it for noisy scans.</div>
          <div className="btn-row" style={{ marginTop: 6 }}>
            <button className="btn small" disabled={s.running} onClick={() => void this.run()}>
              Convert again
            </button>
            {Math.abs(s.tolerance - s.autoTolerance) > 1e-12 && (
              <button className="btn small" disabled={s.running} onClick={() => this.set({ tolerance: s.autoTolerance })}>
                Reset tolerance
              </button>
            )}
          </div>
        </Section>
        {s.error && (
          <Section title="Problem">
            <div className="note">{s.error}</div>
          </Section>
        )}
        <Section title="Result">
          {r ? (
            <>
              <div className="convert-legend">
                {FACE_KIND_ORDER.filter((k) => r.kinds[k]).map((k) => (
                  <div key={k} className="convert-legend-row">
                    <span className="swatch" style={{ background: FACE_KIND_COLOR[k] }} />
                    <span>{FACE_KIND_PLURAL[k]}</span>
                    <span className="count">{int(r.kinds[k]!)}</span>
                  </div>
                ))}
              </div>
              <Stats
                rows={[
                  ["CAD faces", int(r.faces)],
                  ["Shape", r.solid && r.valid ? "valid solid" : r.closed ? "closed shell" : "open shell", r.solid && r.valid ? "good" : "warn"],
                  ["Recognised area", `${(r.coverage * 100).toFixed(1)} %`, r.coverage > 0.99 ? "good" : r.coverage > 0.9 ? "warn" : "bad"],
                  ["Faceted triangles", r.facetedTriangles ? int(r.facetedTriangles) : "none", r.facetedTriangles ? "warn" : "good"],
                  ["Largest deviation", mm(r.maxDeviation, 4)],
                  ...(r.volume !== undefined ? ([["Volume", `${r.volume.toFixed(1)} mm³`]] as [string, string][]) : []),
                  ["Time", `${r.seconds.toFixed(1)} s`],
                ]}
              />
              {r.demoted > 0 && <div className="note">{r.demoted} difficult area(s) were kept as facets so the solid closes.</div>}
              {r.coverage < 0.9 && (
                <div className="note">
                  Much of this mesh did not match planes, cylinders, cones, spheres or tori. If it is a scan, raise the fit tolerance (try 0.05–0.1 mm). Organic shapes
                  convert better with Freeform → Autosurface.
                </div>
              )}
            </>
          ) : (
            <div className="note">{s.running ? "Converting…" : "—"}</div>
          )}
        </Section>
        <Section title="View">
          <Check
            checked={s.showMap}
            onChange={(showMap) => {
              this.set({ showMap });
              this.redraw();
            }}
          >
            Conversion map (colour by face type)
          </Check>
          <Check checked={this.app.ui.get().compare} onChange={(on) => this.app.setCompare(on)}>
            Compare STL / CAD (split view)
          </Check>
        </Section>
      </>
    );
  }

  async commit() {
    if (this.shapeId === null || !this.s.result) return false;
    const k = await this.app.getKernel();
    const brep = await k.serialize(this.shapeId);
    const mesh = this.app.doc.get<MeshEntity>(this.meshId);
    const r = this.s.result;
    const e = this.app.doc.add<BodyEntity>({
      id: newId("b"),
      kind: "body",
      name: this.app.doc.uniqueName(`${mesh?.name ?? "Mesh"} CAD`),
      visible: true,
      feature: {
        type: "converted",
        brep,
        meshId: this.meshId,
        faceKinds: this.faceKinds,
        stats: { faces: r.faces, facetedTriangles: r.facetedTriangles, tolerance: this.s.tolerance, maxDeviation: r.maxDeviation },
      },
    });
    this.app.setActiveEntity(e.id);
    this.app.toast("success", `Created ${e.name}: ${r.faces} faces. Export it from the Export tab (STEP).`);
  }

  deactivate() {
    this.runToken++;
    this.overlay.dispose();
    this.faceMat.dispose();
    this.edgeMat.dispose();
    void this.releaseShape();
  }
}

export function startAutoConvert(app: App): void {
  try {
    app.startTool(new AutoConvertTool(app));
  } catch (e) {
    app.toast("error", e instanceof Error ? e.message : String(e));
  }
}
