import { useEffect, useState } from "react";
import { app } from "../app/App";
import { TopBar } from "./TopBar";
import { Ribbon } from "./Ribbon";
import { ModelTree } from "./ModelTree";
import { Viewport } from "./Viewport";
import { PropertyPanel } from "./PropertyPanel";
import { StatusBar, Toasts } from "./StatusBar";
import { executeCommand, importFiles, loadSample } from "../tools/registry";
import { desktop, toFile } from "../app/desktop";
import type { SampleId } from "../samples/samples";

export function AppShell() {
  const [about, setAbout] = useState(false);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT")) return;
      const ctrl = e.ctrlKey || e.metaKey;
      if (ctrl && e.key.toLowerCase() === "s") {
        e.preventDefault();
        void executeCommand(app, "saveProject");
        return;
      }
      if (ctrl && e.key.toLowerCase() === "o") {
        e.preventDefault();
        void executeCommand(app, "openProject");
        return;
      }
      if (ctrl && e.key.toLowerCase() === "i") {
        e.preventDefault();
        void importFiles(app);
        return;
      }
      if (app.handleKey(e)) e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Desktop shell: native menu commands and files opened from Explorer
  useEffect(() => {
    if (!desktop) return;
    const offMenu = desktop.onMenu((cmd) => {
      if (cmd === "undo") app.doc.undo();
      else if (cmd === "redo") app.doc.redo();
      else if (cmd === "about") setAbout(true);
      else if (cmd === "newProject") {
        if (!app.doc.all().length || confirm("Discard the current project?")) {
          void app.endTool(false);
          app.doc.reset([]);
        }
      } else if (cmd.startsWith("sample:")) void loadSample(app, cmd.slice(7) as SampleId);
      else if (cmd.startsWith("view:")) {
        const v = cmd.slice(5);
        if (v === "fit") app.viewer?.fitAll();
        else app.viewer?.fitAll(v as "iso" | "top" | "front" | "right");
      } else void executeCommand(app, cmd);
    });
    const offOpen = desktop.onOpenFiles((files) => {
      const project = files.find((f) => f.name.toLowerCase().endsWith(".rsf") && f.data);
      if (project) void import("../tools/project").then((m) => m.openProjectFile(app, toFile(project)));
      const scans = files.filter((f) => !f.name.toLowerCase().endsWith(".rsf") && f.data).map(toFile);
      if (scans.length) void importFiles(app, scans);
    });
    return () => {
      offMenu();
      offOpen();
    };
  }, []);

  return (
    <div className="shell">
      <TopBar onAbout={() => setAbout(true)} />
      <Ribbon />
      <ModelTree />
      <Viewport />
      <PropertyPanel />
      <StatusBar />
      <Toasts />
      {about && (
        <div className="modal-backdrop" onClick={() => setAbout(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h2>Resurface 0.1</h2>
            <div className="modal-body">
              Scan-to-CAD reverse engineering: turn 3D scans into editable CAD geometry and export STEP / IGES.
              <br />
              <br />
              Geometry kernel: Open CASCADE Technology (LGPL-2.1) compiled to WebAssembly. Rendering: three.js.
            </div>
            <div className="modal-actions">
              <button className="btn primary" onClick={() => setAbout(false)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
