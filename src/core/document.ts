/**
 * The Resurface document: an ordered collection of immutable entities with transactional
 * undo/redo. Every mutation happens inside a transaction (implicitly if needed); a transaction
 * records (before, after) snapshots per touched entity, so undo simply restores snapshots.
 */
import { Emitter } from "./emitter";
import type { Entity, EntityKind, EntityOf } from "./entities";

type Snapshot = { before: Entity | undefined; after: Entity | undefined; beforeIndex: number; afterIndex: number };

interface Transaction {
  label: string;
  changes: Map<string, Snapshot>;
  /** approximate retained bytes (for the undo memory budget) */
  bytes: number;
}

export interface DocumentEvents {
  /** fired after any change (once per transaction / undo / redo) with the ids that changed */
  changed: { ids: Set<string>; reason: "edit" | "undo" | "redo" | "load" };
  history: { canUndo: boolean; canRedo: boolean; undoLabel?: string; redoLabel?: string };
  [key: string]: unknown;
}

let idCounter = 0;
export const newId = (prefix = "e"): string => `${prefix}${Date.now().toString(36)}${(idCounter++).toString(36)}`;

/** Rough size of typed arrays held by an entity (used to cap undo memory). */
function entityBytes(e: Entity | undefined): number {
  if (!e) return 0;
  let bytes = 256;
  const visit = (o: unknown, depth: number) => {
    if (!o || typeof o !== "object" || depth > 3) return;
    if (ArrayBuffer.isView(o)) {
      bytes += (o as ArrayBufferView).byteLength;
      return;
    }
    for (const v of Object.values(o as Record<string, unknown>)) visit(v, depth + 1);
  };
  visit(e, 0);
  return bytes;
}

export class Document extends Emitter<DocumentEvents> {
  private entities = new Map<string, Entity>();
  private order: string[] = [];
  private undoStack: Transaction[] = [];
  private redoStack: Transaction[] = [];
  private current: Transaction | null = null;
  private depth = 0;
  /** max memory retained by undo history */
  undoBudgetBytes = 3 * 1024 * 1024 * 1024;
  /** bumps on every change; handy for React snapshots */
  version = 0;

  // -------------------------------------------------------------------------
  // Queries
  // -------------------------------------------------------------------------

  get<T extends Entity = Entity>(id: string | undefined | null): T | undefined {
    return id ? (this.entities.get(id) as T | undefined) : undefined;
  }

  has(id: string): boolean {
    return this.entities.has(id);
  }

  all(): Entity[] {
    return this.order.map((id) => this.entities.get(id)!);
  }

  ofKind<K extends EntityKind>(kind: K): EntityOf<K>[] {
    return this.all().filter((e) => e.kind === kind) as EntityOf<K>[];
  }

  indexOf(id: string): number {
    return this.order.indexOf(id);
  }

  /** Generate a unique display name like "Plane 3". */
  uniqueName(base: string): string {
    const names = new Set(this.all().map((e) => e.name));
    if (!names.has(base)) return base;
    for (let i = 2; ; i++) if (!names.has(`${base} ${i}`)) return `${base} ${i}`;
  }

  // -------------------------------------------------------------------------
  // Transactions
  // -------------------------------------------------------------------------

  /** Run `fn` as one undoable step. Nested calls merge into the outermost transaction. */
  transact<R>(label: string, fn: () => R): R {
    const outer = this.depth === 0;
    if (outer) this.current = { label, changes: new Map(), bytes: 0 };
    this.depth++;
    try {
      return fn();
    } catch (err) {
      if (outer) this.rollback();
      throw err;
    } finally {
      this.depth--;
      if (outer && this.current) this.commit();
    }
  }

  private record(id: string, after: Entity | undefined, afterIndex: number): void {
    const tx = this.current!;
    const prev = tx.changes.get(id);
    if (prev) {
      prev.after = after;
      prev.afterIndex = afterIndex;
    } else {
      tx.changes.set(id, {
        before: this.entities.get(id),
        after,
        beforeIndex: this.order.indexOf(id),
        afterIndex,
      });
    }
  }

  private ensureTx<R>(label: string, fn: () => R): R {
    return this.depth > 0 ? fn() : this.transact(label, fn);
  }

  add<T extends Entity>(entity: Omit<T, "rev"> & { rev?: number }, index?: number): T {
    return this.ensureTx(`Add ${entity.name}`, () => {
      const e = { ...entity, rev: entity.rev ?? 1 } as T;
      if (this.entities.has(e.id)) throw new Error(`Duplicate entity id ${e.id}`);
      const at = index === undefined ? this.order.length : Math.max(0, Math.min(index, this.order.length));
      this.record(e.id, e, at);
      this.entities.set(e.id, e);
      this.order.splice(at, 0, e.id);
      return e;
    });
  }

  update<T extends Entity>(id: string, patch: Partial<T> | ((e: T) => Partial<T>)): T {
    return this.ensureTx("Edit", () => {
      const cur = this.entities.get(id) as T | undefined;
      if (!cur) throw new Error(`No entity ${id}`);
      const p = typeof patch === "function" ? patch(cur) : patch;
      const next = { ...cur, ...p, id: cur.id, kind: cur.kind, rev: cur.rev + 1 } as T;
      this.record(id, next, this.order.indexOf(id));
      this.entities.set(id, next);
      return next;
    });
  }

  remove(id: string): void {
    this.ensureTx("Delete", () => {
      if (!this.entities.has(id)) return;
      this.record(id, undefined, -1);
      this.entities.delete(id);
      this.order.splice(this.order.indexOf(id), 1);
    });
  }

  move(id: string, toIndex: number): void {
    this.ensureTx("Reorder", () => {
      const from = this.order.indexOf(id);
      if (from < 0) return;
      this.order.splice(from, 1);
      const at = Math.max(0, Math.min(toIndex, this.order.length));
      this.order.splice(at, 0, id);
      this.record(id, this.entities.get(id), at);
    });
  }

  private commit(): void {
    const tx = this.current!;
    this.current = null;
    if (tx.changes.size === 0) return;
    for (const s of tx.changes.values()) tx.bytes += entityBytes(s.before);
    this.undoStack.push(tx);
    this.redoStack = [];
    this.trimHistory();
    this.version++;
    this.emit("changed", { ids: new Set(tx.changes.keys()), reason: "edit" });
    this.emitHistory();
  }

  private rollback(): void {
    const tx = this.current;
    this.current = null;
    if (!tx) return;
    this.applySnapshots(tx, "before");
  }

  private trimHistory(): void {
    let total = this.undoStack.reduce((s, t) => s + t.bytes, 0);
    while (this.undoStack.length > 1 && (total > this.undoBudgetBytes || this.undoStack.length > 200)) {
      total -= this.undoStack.shift()!.bytes;
    }
  }

  private applySnapshots(tx: Transaction, which: "before" | "after"): void {
    // Remove everything touched, then re-insert in index order to restore ordering exactly.
    const entries = [...tx.changes.entries()];
    for (const [id] of entries) {
      const i = this.order.indexOf(id);
      if (i >= 0) this.order.splice(i, 1);
      this.entities.delete(id);
    }
    const inserts = entries
      .map(([id, s]) => ({ id, e: which === "before" ? s.before : s.after, idx: which === "before" ? s.beforeIndex : s.afterIndex }))
      .filter((x) => x.e)
      .sort((a, b) => a.idx - b.idx);
    for (const { id, e, idx } of inserts) {
      this.entities.set(id, e!);
      this.order.splice(Math.min(Math.max(idx, 0), this.order.length), 0, id);
    }
  }

  get canUndo(): boolean {
    return this.undoStack.length > 0;
  }
  get canRedo(): boolean {
    return this.redoStack.length > 0;
  }

  undo(): void {
    const tx = this.undoStack.pop();
    if (!tx) return;
    this.applySnapshots(tx, "before");
    this.redoStack.push(tx);
    this.version++;
    this.emit("changed", { ids: new Set(tx.changes.keys()), reason: "undo" });
    this.emitHistory();
  }

  redo(): void {
    const tx = this.redoStack.pop();
    if (!tx) return;
    this.applySnapshots(tx, "after");
    this.undoStack.push(tx);
    this.version++;
    this.emit("changed", { ids: new Set(tx.changes.keys()), reason: "redo" });
    this.emitHistory();
  }

  private emitHistory(): void {
    this.emit("history", {
      canUndo: this.canUndo,
      canRedo: this.canRedo,
      undoLabel: this.undoStack.at(-1)?.label,
      redoLabel: this.redoStack.at(-1)?.label,
    });
  }

  /** Replace the whole document (project load / new). Clears history. */
  reset(entities: Entity[] = []): void {
    const ids = new Set([...this.entities.keys(), ...entities.map((e) => e.id)]);
    this.entities.clear();
    this.order = [];
    for (const e of entities) {
      this.entities.set(e.id, e);
      this.order.push(e.id);
    }
    this.undoStack = [];
    this.redoStack = [];
    this.version++;
    this.emit("changed", { ids, reason: "load" });
    this.emitHistory();
  }
}
