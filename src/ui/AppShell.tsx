import { useEffect, useState } from "react";
import { app } from "../app/App";
import { TopBar } from "./TopBar";
import { Ribbon } from "./Ribbon";
import { ModelTree } from "./ModelTree";
import { Viewport } from "./Viewport";
import { PropertyPanel } from "./PropertyPanel";
import { StatusBar, Toasts } from "./StatusBar";
import { executeCommand, importFiles } from "../tools/registry";

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
