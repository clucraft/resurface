/** Instant mesh commands that act on the active mesh and its triangle selection. */
import type { App } from "../app/App";
import type { MeshData } from "../core/types";
import { compute } from "../app/compute";
import { markPrepared } from "../app/MeshRuntime";
import { deleteTriangles, flipNormals } from "../mesh/repair";
import { growSelection, invertSelection, selectConnected, shrinkSelection } from "../mesh/select";

function requireMesh(app: App) {
  const e = app.activeMesh();
  const rt = e ? app.sync?.meshRuntime(e.id) : undefined;
  if (!e || !rt) throw new Error("Import or select a mesh first");
  return { e, rt };
}

export async function deleteSelectedTriangles(app: App): Promise<void> {
  const { e, rt } = requireMesh(app);
  if (!rt.selectionCount) return;
  const n = rt.selectionCount;
  if (n === rt.triangleCount) {
    app.deleteEntities([e.id]);
    return;
  }
  await app.busy("Deleting triangles", () => {
    const mesh = deleteTriangles(rt.mesh, rt.selection);
    // compaction keeps the relative (spatial) triangle order, so no re-sorting is needed
    app.replaceMesh(e.id, mesh, `Delete ${n.toLocaleString()} triangles`);
  });
}

export async function runMeshCommand(app: App, id: string): Promise<void> {
  const { e, rt } = requireMesh(app);
  switch (id) {
    case "deleteSel":
      return deleteSelectedTriangles(app);
    case "flip":
      await app.busy("Flipping normals", () => app.replaceMesh(e.id, flipNormals(rt.mesh), "Flip normals"));
      return;
    case "defeature": {
      if (!rt.selectionCount) throw new Error("Select the feature to remove first");
      const mask = rt.selection.slice();
      await app.busy("Removing feature", async () => {
        const mesh = await compute.run<MeshData>("defeature", { mesh: rt.mesh, mask, prepare: "reorder" });
        markPrepared(mesh);
        app.replaceMesh(e.id, mesh, "De-feature", true);
      });
      app.toast("success", "Feature removed and filled");
      return;
    }
    case "growSel":
      rt.setMask(growSelection(rt.mesh, rt.selection, 1));
      return;
    case "shrinkSel":
      rt.setMask(shrinkSelection(rt.mesh, rt.selection, 1));
      return;
    case "invertSel":
      rt.setMask(invertSelection(rt.selection));
      return;
    case "connectedSel": {
      const seeds = rt.selectedTriangles();
      rt.setMask(selectConnected(rt.mesh, seeds));
      return;
    }
  }
}
