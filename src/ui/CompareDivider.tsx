import { useRef } from "react";
import type { KeyboardEvent, PointerEvent } from "react";
import { ChevronsLeftRight } from "lucide-react";
import { app } from "../app/App";
import { useUi } from "./hooks";

/**
 * Split-view divider over the viewport (compare view): the scan with its triangle edges is drawn
 * left of it, the CAD bodies right of it. Drag the line or the handle to move it; the viewer
 * re-renders on every move. Arrow keys nudge it when the handle has focus.
 */
export function CompareDivider() {
  const on = useUi((s) => s.compare);
  const fraction = useUi((s) => s.compareFraction);
  const root = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  if (!on) return null;

  const moveTo = (clientX: number) => {
    const area = root.current?.parentElement;
    if (!area) return;
    const r = area.getBoundingClientRect();
    if (r.width > 0) app.setCompareFraction((clientX - r.left) / r.width);
  };

  const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    // keep the camera from orbiting and the canvas from starting a selection
    e.preventDefault();
    e.stopPropagation();
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {
      /* synthetic events have no active pointer */
    }
    dragging.current = true;
    root.current?.classList.add("dragging");
    root.current?.querySelector<HTMLElement>(".compare-handle")?.focus({ preventScroll: true });
    moveTo(e.clientX);
  };

  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    if (!dragging.current) return;
    e.stopPropagation();
    moveTo(e.clientX);
  };

  const endDrag = (e: PointerEvent<HTMLDivElement>) => {
    if (!dragging.current) return;
    dragging.current = false;
    root.current?.classList.remove("dragging");
    try {
      if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {
      /* ignore */
    }
  };

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const step = e.shiftKey ? 0.1 : 0.02;
    const current = app.ui.get().compareFraction;
    let next: number;
    if (e.key === "ArrowLeft") next = current - step;
    else if (e.key === "ArrowRight") next = current + step;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = 1;
    else return;
    e.preventDefault();
    e.stopPropagation();
    app.setCompareFraction(next);
  };

  return (
    <div
      ref={root}
      className="compare-divider"
      style={{ left: `${fraction * 100}%` }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
      onLostPointerCapture={endDrag}
    >
      <div className="compare-line" />
      <span className="compare-tag stl">STL</span>
      <span className="compare-tag cad">CAD</span>
      <div
        className="compare-handle"
        role="slider"
        tabIndex={0}
        aria-label="Compare divider: scan on the left, CAD on the right"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(fraction * 100)}
        title="Drag to compare the scan (left) with the CAD model (right)"
        onKeyDown={onKeyDown}
      >
        <ChevronsLeftRight size={16} strokeWidth={2.25} />
      </div>
    </div>
  );
}
