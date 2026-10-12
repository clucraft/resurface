import { useSyncExternalStore, useCallback, useRef } from "react";
import type { Store } from "../core/emitter";
import { app } from "../app/App";
import type { Entity } from "../core/entities";
import type { UiState } from "../app/ui-state";

/** Subscribe to a Store; optional selector to limit re-renders. */
export function useStore<T, R = T>(store: Store<T>, selector?: (v: T) => R): R {
  const sel = selector ?? ((v: T) => v as unknown as R);
  const last = useRef<{ v: T; r: R } | null>(null);
  const get = useCallback(() => {
    const v = store.get();
    if (last.current && Object.is(last.current.v, v)) return last.current.r;
    const r = sel(v);
    if (last.current && shallowEqual(last.current.r, r)) {
      last.current = { v, r: last.current.r };
      return last.current.r;
    }
    last.current = { v, r };
    return r;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [store]);
  return useSyncExternalStore(store.subscribe, get, get);
}

function shallowEqual(a: unknown, b: unknown): boolean {
  if (Object.is(a, b)) return true;
  if (typeof a !== "object" || typeof b !== "object" || !a || !b) return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  const ka = Object.keys(a as object), kb = Object.keys(b as object);
  if (ka.length !== kb.length) return false;
  for (const k of ka) if (!Object.is((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k])) return false;
  return true;
}

export function useUi<R>(selector: (s: UiState) => R): R {
  return useStore(app.ui, selector);
}

const docSubscribe = (fn: () => void) => app.doc.on("changed", fn);
const docVersion = () => app.doc.version;

/** Re-render on any document change; returns the current entity list. */
export function useEntities(): Entity[] {
  useSyncExternalStore(docSubscribe, docVersion, docVersion);
  return app.doc.all();
}

export function useEntity<T extends Entity = Entity>(id: string | null | undefined): T | undefined {
  useSyncExternalStore(docSubscribe, docVersion, docVersion);
  return app.doc.get<T>(id);
}

const histSubscribe = (fn: () => void) => app.doc.on("history", fn);
export function useHistory(): { canUndo: boolean; canRedo: boolean } {
  const snap = useSyncExternalStore(
    histSubscribe,
    () => (app.doc.canUndo ? 1 : 0) + (app.doc.canRedo ? 2 : 0),
    () => 0,
  );
  return { canUndo: (snap & 1) !== 0, canRedo: (snap & 2) !== 0 };
}
