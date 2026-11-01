/**
 * Command dispatch for the ribbon/menus. Tool modules are imported lazily so the initial
 * bundle stays small.
 */
import type { App } from "../app/App";
import type { MeshData, PointCloudData } from "../core/types";
import { compute } from "../app/compute";
import { markPrepared } from "../app/MeshRuntime";
import { SAMPLE_INFO, type SampleId } from "../samples/samples";
import { pickFiles, baseName } from "./files";
import { newId } from "../core/document";
import type { PointsEntity } from "../core/entities";

export const SAMPLES = SAMPLE_INFO;

export async function importFiles(app: App, files?: File[]): Promise<void> {
  const list = files ?? (await pickFiles(".stl,.obj,.ply,.ptx"));
  if (!list.length) return;
  const firstImport = app.doc.ofKind("mesh").length === 0;
  for (const file of list) {
    const ext = file.name.toLowerCase().split(".").pop() ?? "";
    if (!["stl", "obj", "ply", "ptx"].includes(ext)) {
      app.toast("error", `Unsupported file type: ${file.name}`);
      continue;
    }
    await app.busy(`Importing ${file.name}`, async () => {
      const t0 = performance.now();
      const data = await file.arrayBuffer();
      const res = await compute.run<{ kind: "mesh"; mesh: MeshData } | { kind: "points"; points: PointCloudData }>(
        "import",
        { name: file.name, data },
        { transfer: [data], onProgress: (f, m) => app.setProgress(f, m ? `${m} — ${file.name}` : undefined) },
      );
      if (res.kind === "mesh") {
        markPrepared(res.mesh);
        const tris = res.mesh.indices.length / 3;
        if (!tris) throw new Error(`${file.name} contains no triangles`);
        app.addMesh(baseName(file.name), res.mesh, file.name);
        app.toast("success", `Imported ${file.name}: ${tris.toLocaleString()} triangles in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
      } else {
        app.doc.add<PointsEntity>({ id: newId("p"), kind: "points", name: app.doc.uniqueName(baseName(file.name)), visible: true, points: res.points, source: file.name });
        app.toast("info", `Imported point cloud with ${(res.points.positions.length / 3).toLocaleString()} points (no triangles).`);
      }
    });
  }
  if (firstImport) app.viewer?.fitAll("iso");
}

export async function loadSample(app: App, id: SampleId): Promise<void> {
  const info = SAMPLE_INFO.find((s) => s.id === id)!;
  const first = app.doc.all().length === 0;
  await app.busy(`Generating sample: ${info.label}`, async () => {
    let mesh: MeshData;
    if (id === "cadPlate") {
      // a real CAD part, tessellated and imported like any STL file
      const { buildCadSampleStl } = await import("../samples/cadSample");
      const stl = await buildCadSampleStl(await app.getKernel());
      const data = stl.slice().buffer;
      const res = await compute.run<{ kind: "mesh"; mesh: MeshData }>("import", { name: "mounting-plate.stl", data }, { transfer: [data] });
      mesh = res.mesh;
    } else mesh = await compute.run<MeshData>("sample", { id });
    markPrepared(mesh);
    app.addMesh(info.label, mesh, "sample");
  });
  if (first) app.viewer?.fitAll("iso");
  app.setStatus(info.description);
}

/** Small floating menu used by the welcome screen. */
export function openSampleMenu(app: App, anchor: HTMLElement): void {
  const menu = document.createElement("div");
  menu.className = "dropdown-menu";
  const r = anchor.getBoundingClientRect();
  Object.assign(menu.style, { position: "fixed", left: `${r.left}px`, top: `${r.bottom + 4}px` });
  for (const s of SAMPLE_INFO) {
    const b = document.createElement("button");
    b.className = "item";
    b.innerHTML = `<div><div>${s.label}</div><div style="color:var(--text-3);font-size:11px;margin-top:2px">${s.description}</div></div>`;
    b.onclick = () => {
      menu.remove();
      void loadSample(app, s.id);
    };
    menu.appendChild(b);
  }
  document.body.appendChild(menu);
  const close = (e: MouseEvent) => {
    if (!menu.contains(e.target as Node)) {
      menu.remove();
      window.removeEventListener("mousedown", close);
    }
  };
  setTimeout(() => window.addEventListener("mousedown", close), 0);
}

export async function executeCommand(app: App, id: string): Promise<void> {
  try {
    switch (id) {
      case "import":
        return importFiles(app);
      // ---------------------------------------------------------------- mesh
      case "reduce":
      case "smooth":
      case "fillHoles":
      case "clean":
      case "clip": {
        const m = await import("./meshTools");
        return m.startMeshTool(app, id);
      }
      case "deleteSel":
      case "flip":
      case "defeature":
      case "growSel":
      case "shrinkSel":
      case "invertSel":
      case "connectedSel": {
        const m = await import("./meshCommands");
        return m.runMeshCommand(app, id);
      }
      // ----------------------------------------------------------- reference
      case "fitPlane":
      case "fitCylinder":
      case "fitCone":
      case "fitSphere":
      case "fitAuto": {
        const m = await import("./referenceCommands");
        return m.fitFromSelection(app, id);
      }
      case "refLine":
      case "refPoint":
      case "refPlane":
      case "symmetry": {
        const m = await import("./referenceTools");
        return m.startReferenceTool(app, id);
      }
      // --------------------------------------------------------------- align
      case "objectAlign":
      case "transform":
      case "nPointAlign":
      case "fineAlign": {
        const m = await import("./alignTools");
        return m.startAlignTool(app, id);
      }
      // -------------------------------------------------------------- sketch
      case "sketch2d":
      case "multiSection": {
        const m = await import("./sketchTool");
        return m.startSectionTool(app, id === "multiSection");
      }
      case "sketch3d": {
        const m = await import("./sketch3dTool");
        return m.startSketch3D(app);
      }
      // --------------------------------------------------------------- model
      case "autoConvert": {
        const m = await import("./autoConvertTool");
        return m.startAutoConvert(app);
      }
      case "extrude":
      case "revolve":
      case "loft":
      case "primitiveBody":
      case "boolean":
      case "fillet":
      case "mirror": {
        const m = await import("./modelTools");
        return m.startModelTool(app, id);
      }
      case "fitSurface": {
        const m = await import("./fitSurfaceTool");
        return m.startFitSurface(app);
      }
      case "quadSurface":
      case "autoSurface": {
        const m = await import("./freeformTools");
        return m.startFreeformTool(app, id);
      }
      // ------------------------------------------------------------- analyze
      case "deviation":
      case "curvatureMap":
      case "measure": {
        const m = await import("./analysisTools");
        return m.startAnalysisTool(app, id);
      }
      // -------------------------------------------------------------- export
      case "exportStep":
      case "exportIges":
      case "exportStlCad":
      case "exportStlMesh": {
        const m = await import("./exportCommands");
        return m.runExport(app, id);
      }
      case "saveProject":
      case "openProject": {
        const m = await import("./project");
        return id === "saveProject" ? m.saveProject(app) : m.openProject(app);
      }
      default:
        app.toast("info", `"${id}" is not available yet.`);
    }
  } catch (err) {
    console.error(err);
    app.toast("error", err instanceof Error ? err.message : String(err));
  }
}

/** Re-open an entity for editing (sketch → sketch editor, body → its feature tool). */
export async function editEntity(app: App, id: string): Promise<void> {
  const e = app.doc.get(id);
  if (!e) return;
  if (e.kind === "sketch") {
    const m = await import("./sketchTool");
    return m.editSketch(app, id);
  }
  if (e.kind === "body") {
    const m = await import("./modelTools");
    return m.editBody(app, id);
  }
  if (e.kind === "sketch3d") {
    const m = await import("./sketch3dTool");
    return m.startSketch3D(app, id);
  }
}
