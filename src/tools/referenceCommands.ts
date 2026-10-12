/** Primitive extraction from the current triangle selection (QuickSurface "Primitives"). */
import type { App } from "../app/App";
import { newId } from "../core/document";
import type { PrimitiveEntity } from "../core/entities";
import type { FitResult, PrimitiveDef } from "../core/types";
import { detectBestPrimitive, fitPrimitive, ransacPrimitive, type PrimitiveKind } from "../fit/primitives";

const KIND_FOR: Record<string, PrimitiveKind | "auto"> = {
  fitPlane: "plane",
  fitCylinder: "cylinder",
  fitCone: "cone",
  fitSphere: "sphere",
  fitAuto: "auto",
};

const LABEL: Record<PrimitiveKind, string> = { plane: "Plane", cylinder: "Cylinder", cone: "Cone", sphere: "Sphere" };

/** Half-size for drawing a fitted plane: extent of the selection projected on the plane. */
function planeDisplaySize(def: PrimitiveDef, pos: Float32Array, verts: Uint32Array): number {
  if (def.kind !== "plane") return 50;
  let r = 0;
  const step = Math.max(1, Math.floor(verts.length / 20000));
  for (let i = 0; i < verts.length; i += step) {
    const v = verts[i];
    r = Math.max(r, Math.hypot(pos[3 * v] - def.origin[0], pos[3 * v + 1] - def.origin[1], pos[3 * v + 2] - def.origin[2]));
  }
  return Math.max(5, r * 0.75);
}

export async function fitFromSelection(app: App, cmd: string): Promise<void> {
  const e = app.activeMesh();
  const rt = e ? app.sync?.meshRuntime(e.id) : undefined;
  if (!e || !rt || !rt.selectionCount) throw new Error("Select triangles on the mesh first (brush, rectangle, lasso or smart select)");
  const kind = KIND_FOR[cmd];
  await app.busy(kind === "auto" ? "Detecting primitive" : `Fitting ${kind}`, () => {
    const verts = rt.selectedVertices();
    const pos = rt.mesh.positions;
    const normals = rt.mesh.normals;
    let result: FitResult<PrimitiveDef>;
    let k: PrimitiveKind;
    if (kind === "auto") {
      const ranked = detectBestPrimitive(pos, { indices: verts, normals, robust: true });
      if (!ranked.length) throw new Error("Could not fit any primitive to the selection");
      result = ranked[0].result;
      k = ranked[0].kind;
    } else {
      k = kind;
      // robust LS fit; if many outliers (e.g. sloppy brush across a fillet) fall back to RANSAC
      result = fitPrimitive(k, pos, { indices: verts, normals, robust: true });
      const inlierRatio = result.inliers ? countOnes(result.inliers) / result.inliers.length : 1;
      if (inlierRatio < 0.7 && normals) {
        try {
          const r2 = ransacPrimitive(k, pos, normals, { indices: verts });
          if (r2.stats.rms < result.stats.rms) result = r2;
        } catch {
          /* keep LS result */
        }
      }
    }
    const def = result.def;
    const entity = app.doc.add<PrimitiveEntity>({
      id: newId("r"),
      kind: "primitive",
      name: app.doc.uniqueName(LABEL[k]),
      visible: true,
      def,
      stats: result.stats,
      meshId: e.id,
      displaySize: planeDisplaySize(def, pos, verts),
    });
    rt.clearSelection();
    app.setActiveEntity(entity.id);
    const extra =
      def.kind === "cylinder" ? ` Ø${(2 * def.radius).toFixed(3)}` : def.kind === "sphere" ? ` Ø${(2 * def.radius).toFixed(3)}` : def.kind === "cone" ? ` ${((def.halfAngle * 360) / Math.PI).toFixed(2)}°` : "";
    app.toast("success", `${entity.name}${extra} — RMS ${result.stats.rms.toFixed(4)} mm (${result.stats.count.toLocaleString()} pts)`);
  });
}

function countOnes(m: Uint8Array): number {
  let n = 0;
  for (let i = 0; i < m.length; i++) n += m[i] ? 1 : 0;
  return n;
}
