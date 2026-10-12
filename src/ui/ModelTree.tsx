import { useState } from "react";
import {
  Eye, EyeOff, Trash2, ChevronDown, ChevronRight, Triangle, Square, Cylinder, Cone, Circle, Slash, Dot,
  PenTool, Spline, Box, CircleAlert, ScanLine,
} from "lucide-react";
import { app } from "../app/App";
import type { Entity } from "../core/entities";
import { useEntities, useStore, useUi } from "./hooks";
import { int } from "./widgets";

const GROUPS: { id: string; label: string; kinds: Entity["kind"][] }[] = [
  { id: "scan", label: "Scan data", kinds: ["mesh", "points"] },
  { id: "ref", label: "Reference geometry", kinds: ["primitive", "refline", "refpoint"] },
  { id: "sketch", label: "Sketches", kinds: ["sketch", "sketch3d"] },
  { id: "body", label: "CAD bodies", kinds: ["body"] },
];

function iconFor(e: Entity) {
  const p = { size: 15, strokeWidth: 1.7, className: "tree-icon" };
  switch (e.kind) {
    case "mesh":
      return <Triangle {...p} color="#b9c2cc" />;
    case "points":
      return <ScanLine {...p} color="#b9c2cc" />;
    case "primitive": {
      const c = e.symmetry ? "#b46cff" : "#3d8bfd";
      if (e.def.kind === "plane") return <Square {...p} color={c} />;
      if (e.def.kind === "cylinder") return <Cylinder {...p} color={c} />;
      if (e.def.kind === "cone") return <Cone {...p} color={c} />;
      return <Circle {...p} color={c} />;
    }
    case "refline":
      return <Slash {...p} color="#00c2a8" />;
    case "refpoint":
      return <Dot {...p} color="#ffd23f" />;
    case "sketch":
      return <PenTool {...p} color="#2ee6a6" />;
    case "sketch3d":
      return <Spline {...p} color="#ff4fa3" />;
    case "body":
      return <Box {...p} color="#9fb7d6" />;
  }
}

function metaFor(e: Entity): string {
  switch (e.kind) {
    case "mesh":
      return `${int(e.mesh.indices.length / 3)} ▲`;
    case "points":
      return `${int(e.points.positions.length / 3)} pts`;
    case "primitive":
      return e.stats ? `±${e.stats.rms.toFixed(3)}` : "";
    case "body":
      return e.feature.type;
    case "sketch":
      return `${e.sketch?.entities?.length ?? 0} ent`;
    default:
      return "";
  }
}

export function ModelTree() {
  const entities = useEntities();
  const active = useUi((s) => s.activeEntityId);
  const activeMesh = useUi((s) => s.activeMeshId);
  const errors = useStore(app.bodyErrors);
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const [renaming, setRenaming] = useState<string | null>(null);
  const consumed = new Set(entities.flatMap((e) => (e.kind === "body" ? (e.consumes ?? []) : [])));

  return (
    <div className="panel panel-left">
      <div className="panel-header">
        Model
        <span className="count">{entities.length} items</span>
      </div>
      <div className="panel-body">
        {entities.length === 0 && (
          <div className="tree-empty">
            Nothing here yet.
            <br />
            Drop a scan file onto the viewport or use <b>Mesh → Import</b>.
          </div>
        )}
        {GROUPS.map((g) => {
          const items = entities.filter((e) => g.kinds.includes(e.kind) && !consumed.has(e.id));
          if (!items.length) return null;
          const isCollapsed = collapsed[g.id];
          return (
            <div className="tree-group" key={g.id}>
              <div className="tree-group-title" onClick={() => setCollapsed({ ...collapsed, [g.id]: !isCollapsed })}>
                {isCollapsed ? <ChevronRight size={13} /> : <ChevronDown size={13} />}
                {g.label}
                <span style={{ marginLeft: "auto" }}>{items.length}</span>
              </div>
              {!isCollapsed &&
                items.map((e) => (
                  <div
                    key={e.id}
                    className={`tree-row${active === e.id || (e.kind === "mesh" && !active && activeMesh === e.id) ? " active" : ""}${e.visible ? "" : " hidden-entity"}`}
                    onClick={() => app.setActiveEntity(e.id)}
                    onDoubleClick={() => setRenaming(e.id)}
                    title={e.name}
                  >
                    {iconFor(e)}
                    {renaming === e.id ? (
                      <input
                        className="tree-rename"
                        autoFocus
                        defaultValue={e.name}
                        onClick={(ev) => ev.stopPropagation()}
                        onKeyDown={(ev) => {
                          ev.stopPropagation();
                          if (ev.key === "Enter") (ev.target as HTMLInputElement).blur();
                          if (ev.key === "Escape") setRenaming(null);
                        }}
                        onBlur={(ev) => {
                          const v = ev.target.value.trim();
                          if (v && v !== e.name) app.rename(e.id, v);
                          setRenaming(null);
                        }}
                      />
                    ) : (
                      <span className="tree-name">{e.name}</span>
                    )}
                    {errors[e.id] && (
                      <span title={errors[e.id]} className="err">
                        <CircleAlert size={14} />
                      </span>
                    )}
                    <span className="tree-meta">{metaFor(e)}</span>
                    <button
                      className={`tree-action${e.visible ? "" : " always"}`}
                      title={e.visible ? "Hide" : "Show"}
                      onClick={(ev) => {
                        ev.stopPropagation();
                        app.setVisible(e.id, !e.visible);
                      }}
                    >
                      {e.visible ? <Eye size={14} /> : <EyeOff size={14} />}
                    </button>
                    <button
                      className="tree-action"
                      title="Delete"
                      onClick={(ev) => {
                        ev.stopPropagation();
                        app.deleteEntities([e.id]);
                      }}
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}
