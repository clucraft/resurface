import { useEffect, useRef, useState } from "react";
import {
  MousePointer2, Brush, SquareDashed, Lasso, WandSparkles, Maximize, Box, ArrowUp, ArrowRight, ArrowDown,
  Grid3x3, Eye, LoaderCircle, Upload, Sparkles, Palette, X, SquareSplitHorizontal,
} from "lucide-react";
import { app } from "../app/App";
import { CompareDivider } from "./CompareDivider";
import type { SelectionMode } from "../app/ui-state";
import { useEntities, useUi } from "./hooks";
import { importFiles, openSampleMenu } from "../tools/registry";
import { colormapCss } from "../analysis/colormap";

const SEL_MODES: { mode: SelectionMode; icon: typeof Brush; title: string }[] = [
  { mode: "pick", icon: MousePointer2, title: "Pick entities (1)" },
  { mode: "brush", icon: Brush, title: "Brush select (2) — Shift add, Ctrl remove, [ ] size" },
  { mode: "rect", icon: SquareDashed, title: "Rectangle select (3) — Shift add, Ctrl remove" },
  { mode: "lasso", icon: Lasso, title: "Lasso select (4) — Shift add, Ctrl remove" },
  { mode: "smart", icon: WandSparkles, title: "Smart select (5) — click a smooth region" },
];

export function Viewport() {
  const host = useRef<HTMLDivElement>(null);
  const rectRef = useRef<HTMLDivElement>(null);
  const lassoRef = useRef<SVGPolylineElement>(null);
  const brushRef = useRef<HTMLDivElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const mode = useUi((s) => s.selectionMode);
  const brushRadius = useUi((s) => s.brushRadius);
  const smartAngle = useUi((s) => s.smartAngle);
  const busy = useUi((s) => s.busy);
  const legend = useUi((s) => s.legend);
  const toolId = useUi((s) => s.toolId);
  const meshDisplay = useUi((s) => s.meshDisplay);
  const showGrid = useUi((s) => s.showGrid);
  const compare = useUi((s) => s.compare);
  const entities = useEntities();

  useEffect(() => {
    const el = host.current!;
    const viewer = app.attachViewer(el);
    app.selection.overlay = { rect: rectRef.current, lasso: lassoRef.current, brush: brushRef.current };
    const canvas = viewer.domElement;
    const down = (e: PointerEvent) => {
      canvas.focus();
      try {
        if (e.button === 0) canvas.setPointerCapture(e.pointerId);
      } catch {
        /* synthetic events have no active pointer */
      }
      app.handlePointer("down", e);
    };
    const move = (e: PointerEvent) => app.handlePointer("move", e);
    const up = (e: PointerEvent) => {
      try {
        if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
      } catch {
        /* ignore */
      }
      app.handlePointer("up", e);
    };
    const leave = () => app.selection.updateBrushCursor(0, 0, false);
    canvas.tabIndex = 0;
    canvas.addEventListener("pointerdown", down);
    canvas.addEventListener("pointermove", move);
    canvas.addEventListener("pointerup", up);
    canvas.addEventListener("pointerleave", leave);
    return () => {
      canvas.removeEventListener("pointerdown", down);
      canvas.removeEventListener("pointermove", move);
      canvas.removeEventListener("pointerup", up);
      canvas.removeEventListener("pointerleave", leave);
      app.detachViewer();
    };
  }, []);

  useEffect(() => {
    if (mode !== "brush") app.selection.updateBrushCursor(0, 0, false);
  }, [mode]);

  const hasData = entities.length > 0;
  const toolAllowsSelection = !toolId || app.activeTool?.allowMeshSelection;
  const canCompare =
    entities.some((e) => (e.kind === "mesh" || e.kind === "points") && e.visible) && entities.some((e) => e.kind === "body" && e.visible);

  return (
    <div
      className="viewport"
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        if (e.dataTransfer.files.length) void importFiles(app, [...e.dataTransfer.files]);
      }}
    >
      <div ref={host} style={{ position: "absolute", inset: 0 }} />
      <CompareDivider />

      {hasData && toolAllowsSelection && (
        <div className="vp-toolbar top">
          {SEL_MODES.map(({ mode: m, icon: Icon, title }) => (
            <button key={m} className={`icon-btn${mode === m ? " active" : ""}`} title={title} onClick={() => app.ui.patch({ selectionMode: m })}>
              <Icon size={16} />
            </button>
          ))}
          {mode === "brush" && (
            <>
              <div className="sep" />
              <div className="range" title="Brush radius ( [ / ] )">
                Size
                <input type="range" min={4} max={200} value={brushRadius} onChange={(e) => app.ui.patch({ brushRadius: Number(e.target.value) })} />
              </div>
            </>
          )}
          {mode === "smart" && (
            <>
              <div className="sep" />
              <div className="range" title="Angle tolerance for smart selection">
                Tolerance
                <input type="range" min={2} max={45} value={smartAngle} onChange={(e) => app.ui.patch({ smartAngle: Number(e.target.value) })} />
                <span style={{ width: 26 }}>{smartAngle}°</span>
              </div>
            </>
          )}
          <div className="sep" />
          <button className="icon-btn" title="Clear selection (Esc)" onClick={() => app.activeMeshRuntime()?.clearSelection()}>
            <X size={16} />
          </button>
        </div>
      )}

      <div className="vp-toolbar right">
        <button className="icon-btn" title="Fit all (F)" onClick={() => app.viewer?.fitAll()}>
          <Maximize size={16} />
        </button>
        <button className="icon-btn" title="Isometric view" onClick={() => app.viewer?.fitAll("iso")}>
          <Box size={16} />
        </button>
        <button className="icon-btn" title="Top view" onClick={() => app.viewer?.fitAll("top")}>
          <ArrowDown size={16} />
        </button>
        <button className="icon-btn" title="Front view" onClick={() => app.viewer?.fitAll("front")}>
          <ArrowUp size={16} />
        </button>
        <button className="icon-btn" title="Right view" onClick={() => app.viewer?.fitAll("right")}>
          <ArrowRight size={16} />
        </button>
        <div className="sep" />
        <button
          className={`icon-btn${showGrid ? " active" : ""}`}
          title="Grid"
          onClick={() => {
            app.ui.patch({ showGrid: !showGrid });
            if (app.viewer) {
              app.viewer.showGrid = !showGrid;
              app.viewer.requestRender();
            }
          }}
        >
          <Grid3x3 size={16} />
        </button>
        <button
          className={`icon-btn${meshDisplay === "wireframe" ? " active" : ""}`}
          title="Wireframe overlay (small meshes)"
          onClick={() => app.setMeshDisplay(meshDisplay === "wireframe" ? "shaded" : "wireframe")}
        >
          <Eye size={16} />
        </button>
        <button
          className={`icon-btn${meshDisplay === "scan-colors" ? " active" : ""}`}
          title="Show scan colours"
          onClick={() => app.setMeshDisplay(meshDisplay === "scan-colors" ? "shaded" : "scan-colors")}
        >
          <Palette size={16} />
        </button>
        <div className="sep" />
        <button
          className={`icon-btn${compare ? " active" : ""}`}
          title={canCompare || compare ? "Compare STL / CAD (split view)" : "Compare STL / CAD (split view) — needs a visible mesh and a CAD body"}
          disabled={!compare && !canCompare}
          onClick={() => app.setCompare(!compare)}
        >
          <SquareSplitHorizontal size={16} />
        </button>
      </div>

      {legend && (
        <div className="legend">
          <div className="title">{legend.title}</div>
          <div className="bar-wrap">
            <div className="bar" style={{ background: colormapCss(legend.tolerance !== undefined ? legend.tolerance / Math.max(Math.abs(legend.min), Math.abs(legend.max)) : undefined) }} />
            <div className="ticks">
              {[1, 0.75, 0.5, 0.25, 0].map((t) => (
                <span key={t}>{(legend.min + (legend.max - legend.min) * t).toFixed(3)}</span>
              ))}
            </div>
          </div>
          {legend.stats && (
            <div className="stats-lines">
              {legend.stats.map((s, i) => (
                <div key={i}>{s}</div>
              ))}
            </div>
          )}
        </div>
      )}

      {!hasData && (
        <div className="welcome">
          <img src="/favicon.svg" width={56} height={56} alt="" />
          <h2>Resurface</h2>
          <p>Turn 3D scans into CAD. Drop an STL, OBJ, PLY or PTX file here, or start from a sample scan.</p>
          <div className="actions">
            <button className="btn primary" onClick={() => void importFiles(app)}>
              <Upload size={15} /> Import scan
            </button>
            <button className="btn" onClick={(e) => openSampleMenu(app, e.currentTarget)}>
              <Sparkles size={15} /> Load sample
            </button>
          </div>
        </div>
      )}

      <div ref={rectRef} className="marquee" />
      <svg className="lasso-svg">
        <polyline ref={lassoRef} points="" />
      </svg>
      <div ref={brushRef} className="brush-cursor" />

      {dragOver && <div className="drop-hint">Drop scan files to import</div>}

      {busy && (
        <div className="busy-overlay">
          <div className="busy-card">
            <div className="msg">
              <LoaderCircle size={18} className="spin" />
              {busy.message}
            </div>
            <div className={`progress${busy.progress === undefined ? " indeterminate" : ""}`}>
              <div style={{ width: `${Math.round((busy.progress ?? 0) * 100)}%` }} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
