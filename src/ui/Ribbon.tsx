import { app } from "../app/App";
import { COMMANDS, RIBBON_TABS, commandEnabled } from "../app/commands";
import { executeCommand } from "../tools/registry";
import { useEntities, useUi } from "./hooks";

export function Ribbon() {
  const tab = useUi((s) => s.ribbonTab);
  const toolId = useUi((s) => s.toolId);
  useUi((s) => s.selectedTriangles); // re-evaluate enabled state on selection changes
  useEntities();
  const cmds = COMMANDS.filter((c) => c.tab === tab);
  const groups: string[] = [];
  for (const c of cmds) if (!groups.includes(c.group)) groups.push(c.group);

  return (
    <div className="ribbon">
      <div className="ribbon-tabs">
        {RIBBON_TABS.map((t) => (
          <button key={t.id} className={`ribbon-tab${t.id === tab ? " active" : ""}`} onClick={() => app.ui.patch({ ribbonTab: t.id })}>
            {t.label}
          </button>
        ))}
      </div>
      <div className="ribbon-body">
        {groups.map((g) => (
          <div className="ribbon-group" key={g}>
            <div className="ribbon-group-items">
              {cmds
                .filter((c) => c.group === g)
                .map((c) => {
                  const Icon = c.icon;
                  const enabled = commandEnabled(app, c);
                  return (
                    <button
                      key={c.id}
                      className={`rbtn${toolId === c.id ? " active" : ""}`}
                      disabled={!enabled}
                      title={c.tooltip + (enabled ? "" : requirementHint(c.requires))}
                      onClick={() => void executeCommand(app, c.id)}
                    >
                      <Icon size={22} strokeWidth={1.6} />
                      <span>{c.label}</span>
                    </button>
                  );
                })}
            </div>
            <div className="ribbon-group-title">{g}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function requirementHint(req?: string[]): string {
  if (!req?.length) return "";
  const map: Record<string, string> = {
    mesh: "import a mesh",
    selection: "select triangles on the mesh",
    sketch: "create a sketch",
    body: "create a CAD body",
    twoMeshes: "import at least two meshes",
    twoBodies: "create at least two bodies",
  };
  return `\n(Requires: ${req.map((r) => map[r] ?? r).join(", ")})`;
}
