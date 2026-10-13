/**
 * Turns body feature definitions into kernel calls. Each body owns at most one live kernel
 * shape; re-evaluation releases the previous one. Bodies can depend on sketches and on other
 * bodies (booleans, fillets, mirror).
 */
import type { BodyEntity, FeatureDef, SketchEntity } from "../core/entities";
import type { KernelClient } from "../cad/KernelClient";
import type { BodyTessellation } from "../viewer/views";
import type { App } from "./App";
import { lazy } from "./lazy";

export function featureDependencies(f: FeatureDef): string[] {
  switch (f.type) {
    case "extrude":
    case "revolve":
      return [f.sketchId];
    case "loft":
      return f.sketchIds;
    case "boolean":
      return [f.targetId, f.toolId];
    case "fillet":
    case "chamfer":
    case "mirror":
      return [f.baseId];
    default:
      return [];
  }
}

export class FeatureEvaluator {
  /** body id -> kernel shape id */
  private shapes = new Map<string, number>();
  private pending = new Map<string, Promise<number>>();

  constructor(private app: App) {}

  shapeOf(bodyId: string): number | undefined {
    return this.shapes.get(bodyId);
  }

  /** Kernel shape for a body, evaluating if needed (used by dependants and export). */
  async ensureShape(bodyId: string): Promise<number> {
    const existing = this.shapes.get(bodyId);
    if (existing !== undefined) return existing;
    const p = this.pending.get(bodyId);
    if (p) return p;
    const body = this.app.doc.get<BodyEntity>(bodyId);
    if (!body || body.kind !== "body") throw new Error("Missing body dependency");
    return this.build(body);
  }

  private async kernel(): Promise<KernelClient> {
    return this.app.getKernel();
  }

  private sketch(id: string): SketchEntity {
    const s = this.app.doc.get<SketchEntity>(id);
    if (!s || s.kind !== "sketch") throw new Error("The sketch used by this feature was deleted");
    return s;
  }

  private async build(body: BodyEntity): Promise<number> {
    const run = (async () => {
      const k = await this.kernel();
      const { extractProfiles } = await lazy<typeof import("../sketch/profiles")>("../sketch/profiles.ts");
      const f = body.feature;
      let id: number;
      switch (f.type) {
        case "extrude": {
          const sk = this.sketch(f.sketchId);
          const { profile, openWires } = extractProfiles(sk.sketch);
          if (profile.loops.length && f.solid) {
            id = await k.extrude({ frame: sk.frame, profile, distance: f.distance, distance2: f.distance2, solid: true, draftDeg: f.draftDeg });
          } else if (profile.loops.length) {
            id = await k.extrude({ frame: sk.frame, profile, distance: f.distance, distance2: f.distance2, solid: false });
          } else if (openWires.length) {
            id = await k.extrudeWire({ frame: sk.frame, wire: openWires[0], distance: f.distance, distance2: f.distance2 });
          } else throw new Error("Sketch has no geometry to extrude");
          break;
        }
        case "revolve": {
          const sk = this.sketch(f.sketchId);
          const { profile, openWires } = extractProfiles(sk.sketch);
          if (profile.loops.length) {
            id = await k.revolve({ frame: sk.frame, profile, axisOrigin: f.axisOrigin, axisDir: f.axisDir, angleDeg: f.angleDeg, solid: f.solid });
          } else if (openWires.length) {
            id = await k.revolve({ frame: sk.frame, wire: openWires[0], axisOrigin: f.axisOrigin, axisDir: f.axisDir, angleDeg: f.angleDeg, solid: false });
          } else throw new Error("Sketch has no geometry to revolve");
          break;
        }
        case "fitSurface":
          id = await k.bsplineSurfaceFace({ surface: f.surface });
          break;
        case "freeform": {
          const { mergeRegularRegions, toBicubicPatches } = await lazy<typeof import("../freeform/patches")>("../freeform/patches.ts");
          // fewer, larger NURBS faces (regular grid regions merged) are friendlier to downstream CAD
          let patches;
          try {
            patches = mergeRegularRegions(f.cage).surfaces;
          } catch {
            patches = toBicubicPatches(f.cage);
          }
          id = await k.bsplinePatchesToShape({ surfaces: patches, sewTolerance: 1e-3, makeSolid: f.solid });
          break;
        }
        case "primitive":
          id = await k.primitiveFace(f.def, { size: f.size });
          break;
        case "loft": {
          const sections = f.sketchIds.map((sid) => {
            const sk = this.sketch(sid);
            return { frame: sk.frame, profile: extractProfiles(sk.sketch).profile };
          });
          id = await k.loft({ sections, solid: f.solid });
          break;
        }
        case "boolean": {
          const a = await this.ensureShape(f.targetId);
          const b = await this.ensureShape(f.toolId);
          id = await k.boolean(f.op, a, b);
          break;
        }
        case "fillet":
          id = await k.fillet(await this.ensureShape(f.baseId), f.edges, f.radius);
          break;
        case "chamfer":
          id = await k.chamfer(await this.ensureShape(f.baseId), f.edges, f.distance);
          break;
        case "mirror":
          id = await k.mirror(await this.ensureShape(f.baseId), { origin: f.planeOrigin, normal: f.planeNormal });
          break;
      }
      return id;
    })();
    this.pending.set(body.id, run);
    try {
      const id = await run;
      const prev = this.shapes.get(body.id);
      this.shapes.set(body.id, id);
      if (prev !== undefined && prev !== id) void (await this.kernel()).release(prev);
      return id;
    } finally {
      this.pending.delete(body.id);
    }
  }

  /** Evaluate (rebuild) a body and return its tessellation for display. */
  async evaluate(body: BodyEntity): Promise<BodyTessellation> {
    // drop the cached shape so the body is rebuilt from its (possibly changed) inputs
    const prev = this.shapes.get(body.id);
    this.shapes.delete(body.id);
    try {
      const id = await this.build(body);
      const k = await this.kernel();
      if (prev !== undefined && prev !== id) void k.release(prev);
      return await k.tessellate(id, { linearDeflection: 0.05, angularDeflectionDeg: 12 });
    } catch (e) {
      if (prev !== undefined) void (await this.kernel()).release(prev);
      throw e;
    }
  }

  release(bodyId: string): void {
    const id = this.shapes.get(bodyId);
    this.shapes.delete(bodyId);
    if (id !== undefined) void this.kernel().then((k) => k.release(id));
  }
}
