/**
 * Interactive command framework. A Tool owns viewport interaction while active and renders a
 * property panel. Tools that work on the current mesh selection set `allowMeshSelection` so
 * the selection brushes stay usable while their panel is open.
 */
import type * as THREE from "three";
import type { ReactNode } from "react";
import { Store } from "../core/emitter";
import type { PickHit } from "../viewer/Viewer";
import type { App } from "../app/App";

export interface ToolPointerEvent {
  native: PointerEvent;
  button: number;
  shift: boolean;
  ctrl: boolean;
  alt: boolean;
  /** CSS px relative to the canvas */
  x: number;
  y: number;
  ndc: THREE.Vector2;
  ray: THREE.Ray;
  /** picks lazily (cached per event); optional entity filter */
  pick: (filter?: (entityId: string) => boolean) => PickHit | null;
}

export abstract class Tool<S extends object = Record<string, never>> {
  abstract readonly id: string;
  abstract readonly label: string;
  /** one-line instruction shown in the status bar while active */
  hint = "";
  /** keep mesh selection brushes active while this tool runs */
  allowMeshSelection = false;
  readonly state: Store<S>;

  constructor(
    protected app: App,
    initial: S,
  ) {
    this.state = new Store<S>(initial);
  }

  get s(): S {
    return this.state.get();
  }

  set(patch: Partial<S>): void {
    this.state.patch(patch);
  }

  /** called when the tool becomes active */
  activate(): void {}
  /** called when the tool is closed (after commit or cancel) — remove previews here */
  deactivate(): void {}

  onPointerDown(_e: ToolPointerEvent): boolean {
    return false;
  }
  onPointerMove(_e: ToolPointerEvent): boolean {
    return false;
  }
  onPointerUp(_e: ToolPointerEvent): boolean {
    return false;
  }
  onKeyDown(_e: KeyboardEvent): boolean {
    return false;
  }
  /** document changed while active (e.g. selection, undo) */
  onDocumentChanged(_ids: Set<string>): void {}
  /** mesh triangle selection changed while active */
  onSelectionChanged(): void {}

  /** property panel content */
  abstract renderPanel(): ReactNode;

  /** OK button: apply and close. Return false to keep the tool open (e.g. validation failed). */
  async commit(): Promise<boolean | void> {}
  /** Cancel button / Escape */
  cancel(): void {}

  /** text for the OK button; null hides it */
  okLabel: string | null = "OK";
  cancelLabel = "Cancel";
}

export type ToolFactory = (app: App) => Tool<object>;
