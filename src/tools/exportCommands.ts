/** STEP / IGES / STL export. */
import type { App } from "../app/App";
import { writeSTLBinary } from "../mesh/io/stl";
import { saveFile } from "./files";

function projectName(app: App): string {
  const m = app.doc.ofKind("mesh")[0];
  return (m?.name ?? "resurface").replace(/[^\w\-. ]+/g, "_");
}

async function bodyShapes(app: App, visibleOnly = true) {
  const bodies = app.doc.ofKind("body").filter((b) => !visibleOnly || b.visible);
  if (!bodies.length) throw new Error("There are no visible CAD bodies to export");
  const shapes: { id: number; name: string }[] = [];
  for (const b of bodies) {
    try {
      shapes.push({ id: await app.features.ensureShape(b.id), name: b.name });
    } catch (e) {
      throw new Error(`${b.name} could not be built: ${e instanceof Error ? e.message : e}`);
    }
  }
  return shapes;
}

export async function runExport(app: App, id: string): Promise<void> {
  switch (id) {
    case "exportStlMesh": {
      const m = app.activeMesh();
      if (!m) throw new Error("No mesh to export");
      const data = await app.busy("Writing STL", () => writeSTLBinary(m.mesh, `Resurface export: ${m.name}`));
      if (data && (await saveFile(data, `${m.name}.stl`, "model/stl"))) app.toast("success", `Saved ${m.name}.stl`);
      return;
    }
    case "exportStep": {
      const text = await app.busy("Exporting STEP", async () => {
        const k = await app.getKernel();
        return k.exportSTEP(await bodyShapes(app));
      });
      if (text && (await saveFile(text, `${projectName(app)}.step`, "model/step"))) app.toast("success", "STEP file saved");
      return;
    }
    case "exportIges": {
      const text = await app.busy("Exporting IGES", async () => {
        const k = await app.getKernel();
        return k.exportIGES(await bodyShapes(app));
      });
      if (text && (await saveFile(text, `${projectName(app)}.igs`, "model/iges"))) app.toast("success", "IGES file saved");
      return;
    }
    case "exportStlCad": {
      const data = await app.busy("Exporting STL", async () => {
        const k = await app.getKernel();
        return k.exportSTL(await bodyShapes(app), { linearDeflection: 0.02 });
      });
      if (data && (await saveFile(data as BlobPart, `${projectName(app)}-cad.stl`, "model/stl"))) app.toast("success", "STL saved");
      return;
    }
  }
}
