import { CircleAlert, CircleCheck, Info } from "lucide-react";
import { useUi, useEntities } from "./hooks";
import { int } from "./widgets";

export function StatusBar() {
  const status = useUi((s) => s.status);
  const sel = useUi((s) => s.selectedTriangles);
  const cursor = useUi((s) => s.cursor);
  const entities = useEntities();
  const tris = entities.reduce((n, e) => n + (e.kind === "mesh" ? e.mesh.indices.length / 3 : 0), 0);
  return (
    <div className="statusbar">
      <span className="hint">{status}</span>
      {sel > 0 && <span className="cell sel">{int(sel)} selected</span>}
      {tris > 0 && <span className="cell">{int(tris)} ▲</span>}
      <span className="cell" style={{ minWidth: 210, textAlign: "right" }}>
        {cursor ? `X ${cursor[0].toFixed(2)}  Y ${cursor[1].toFixed(2)}  Z ${cursor[2].toFixed(2)}` : ""}
      </span>
      <span className="cell">mm</span>
    </div>
  );
}

export function Toasts() {
  const toasts = useUi((s) => s.toasts);
  return (
    <div className="toasts">
      {toasts.map((t) => (
        <div key={t.id} className={`toast ${t.kind}`}>
          {t.kind === "error" ? <CircleAlert size={16} /> : t.kind === "success" ? <CircleCheck size={16} /> : <Info size={16} />}
          <span>{t.text}</span>
        </div>
      ))}
    </div>
  );
}
