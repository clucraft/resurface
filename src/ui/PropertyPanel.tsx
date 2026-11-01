import { Check as CheckIcon, X } from "lucide-react";
import { app } from "../app/App";
import { COMMANDS } from "../app/commands";
import { useEntity, useStore, useUi } from "./hooks";
import { EntityInspector } from "./EntityInspector";

export function PropertyPanel() {
  const tool = useStore(app.toolStore);
  useStore((tool?.state ?? app.ui) as typeof app.ui); // re-render when the tool's state changes
  const activeId = useUi((s) => s.activeEntityId);
  const active = useEntity(activeId);

  if (tool) {
    const cmd = COMMANDS.find((c) => c.id === tool.id);
    const Icon = cmd?.icon;
    return (
      <div className="panel panel-right">
        <div className="tool-title">
          {Icon && <Icon size={18} strokeWidth={1.7} color="var(--accent)" />}
          {tool.label}
        </div>
        {tool.hint && <div className="tool-hint">{tool.hint}</div>}
        <div className="panel-body" style={{ padding: 0 }}>
          {tool.renderPanel()}
        </div>
        <div className="tool-actions">
          {tool.okLabel && (
            <button className="btn primary" style={{ flex: 1 }} onClick={() => void app.endTool(true)}>
              <CheckIcon size={15} /> {tool.okLabel}
            </button>
          )}
          <button className="btn" style={{ flex: tool.okLabel ? undefined : 1 }} onClick={() => void app.endTool(false)}>
            <X size={15} /> {tool.okLabel ? tool.cancelLabel : "Close"}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="panel panel-right">
      <div className="panel-header">Properties</div>
      <div className="panel-body" style={{ padding: 0 }}>
        {active ? <EntityInspector entity={active} /> : <QuickHelp />}
      </div>
    </div>
  );
}

function QuickHelp() {
  return (
    <div className="empty-props">
      <h3>Workflow</h3>
      <div className="note">
        1. <b>Import</b> a scan and clean it up (Mesh tab).
        <br />
        2. <b>Select</b> regions and fit planes, cylinders… (Reference).
        <br />
        3. <b>Align</b> the scan to the world coordinate system.
        <br />
        4. <b>Sketch</b> on sections, then <b>Extrude / Revolve</b>.
        <br />
        5. Use <b>Fit Surface</b> or <b>Freeform</b> for organic areas.
        <br />
        6. Check <b>Deviation</b> and <b>Export</b> STEP / IGES.
      </div>
      <h3 style={{ marginTop: 18 }}>Navigation</h3>
      <div className="kbd-list">
        <kbd>Right drag</kbd>
        <span>Rotate around cursor</span>
        <kbd>Middle drag</kbd>
        <span>Pan (or Shift + right drag)</span>
        <kbd>Wheel</kbd>
        <span>Zoom at cursor</span>
        <kbd>F</kbd>
        <span>Fit all</span>
        <kbd>1 – 5</kbd>
        <span>Pick / Brush / Rect / Lasso / Smart</span>
        <kbd>Shift / Ctrl</kbd>
        <span>Add to / remove from selection</span>
        <kbd>[ ]</kbd>
        <span>Brush size</span>
        <kbd>Esc</kbd>
        <span>Cancel / exit compare / deselect</span>
        <kbd>Ctrl Z / Y</kbd>
        <span>Undo / redo</span>
      </div>
    </div>
  );
}
