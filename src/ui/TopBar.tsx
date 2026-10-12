import { useEffect, useRef, useState } from "react";
import { Undo2, Redo2, FilePlus, FolderOpen, Save, Upload, Sparkles, FileDown, ChevronDown, Info } from "lucide-react";
import { app } from "../app/App";
import { useHistory } from "./hooks";
import { executeCommand, importFiles, loadSample, SAMPLES } from "../tools/registry";

export function TopBar({ onAbout }: { onAbout: () => void }) {
  const { canUndo, canRedo } = useHistory();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const close = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener("mousedown", close);
    return () => window.removeEventListener("mousedown", close);
  }, []);

  const item = (icon: React.ReactNode, label: string, fn: () => void, kbd?: string) => (
    <button
      className="item"
      onClick={() => {
        setOpen(false);
        fn();
      }}
    >
      {icon}
      {label}
      {kbd && <span className="kbd">{kbd}</span>}
    </button>
  );

  return (
    <div className="topbar">
      <div className="brand">
        <img src="/favicon.svg" alt="" />
        Resurface
        <span className="sub">scan → CAD</span>
      </div>
      <div className="dropdown" ref={ref}>
        <button className="menu-btn" onClick={() => setOpen(!open)}>
          File <ChevronDown size={13} />
        </button>
        {open && (
          <div className="dropdown-menu">
            {item(<FilePlus size={15} />, "New project", () => {
              if (!app.doc.all().length || confirm("Discard the current project?")) {
                void app.endTool(false);
                app.doc.reset([]);
              }
            })}
            {item(<FolderOpen size={15} />, "Open project…", () => void executeCommand(app, "openProject"), "Ctrl+O")}
            {item(<Save size={15} />, "Save project…", () => void executeCommand(app, "saveProject"), "Ctrl+S")}
            <div className="sep" />
            {item(<Upload size={15} />, "Import scan…", () => void importFiles(app), "Ctrl+I")}
            <div className="label">Samples</div>
            {SAMPLES.map((s) => (
              <div key={s.id}>{item(<Sparkles size={15} />, s.label, () => void loadSample(app, s.id))}</div>
            ))}
            <div className="sep" />
            {item(<FileDown size={15} />, "Export STEP…", () => void executeCommand(app, "exportStep"))}
            {item(<FileDown size={15} />, "Export IGES…", () => void executeCommand(app, "exportIges"))}
            <div className="sep" />
            {item(<Info size={15} />, "About Resurface", onAbout)}
          </div>
        )}
      </div>
      <button className="icon-btn" title="Undo (Ctrl+Z)" disabled={!canUndo} onClick={() => app.doc.undo()}>
        <Undo2 size={16} />
      </button>
      <button className="icon-btn" title="Redo (Ctrl+Y)" disabled={!canRedo} onClick={() => app.doc.redo()}>
        <Redo2 size={16} />
      </button>
      <div className="spacer" />
    </div>
  );
}
