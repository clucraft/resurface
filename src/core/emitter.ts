/** Minimal typed event emitter. */
export class Emitter<Events extends Record<string, unknown>> {
  private handlers = new Map<keyof Events, Set<(payload: never) => void>>();

  on<K extends keyof Events>(event: K, fn: (payload: Events[K]) => void): () => void {
    let set = this.handlers.get(event);
    if (!set) {
      set = new Set();
      this.handlers.set(event, set);
    }
    set.add(fn as (payload: never) => void);
    return () => this.off(event, fn);
  }

  off<K extends keyof Events>(event: K, fn: (payload: Events[K]) => void): void {
    this.handlers.get(event)?.delete(fn as (payload: never) => void);
  }

  emit<K extends keyof Events>(event: K, payload: Events[K]): void {
    const set = this.handlers.get(event);
    if (!set) return;
    for (const fn of [...set]) (fn as (p: Events[K]) => void)(payload);
  }
}

/**
 * A tiny observable value store compatible with React's useSyncExternalStore.
 * `version` increases on every change so snapshots are cheap to compare.
 */
export class Store<T> {
  private listeners = new Set<() => void>();
  constructor(private value: T) {}

  get = (): T => this.value;

  set = (next: T | ((prev: T) => T)): void => {
    const v = typeof next === "function" ? (next as (prev: T) => T)(this.value) : next;
    if (Object.is(v, this.value)) return;
    this.value = v;
    for (const l of [...this.listeners]) l();
  };

  patch = (partial: Partial<T>): void => {
    this.set({ ...this.value, ...partial });
  };

  subscribe = (fn: () => void): (() => void) => {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  };
}
