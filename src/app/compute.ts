/**
 * Pool of compute workers for heavy mesh / fitting jobs. Jobs are referenced by name (see
 * compute.worker.ts). Typed arrays backed by SharedArrayBuffer are passed without copying.
 */
import type { ProgressFn } from "../core/types";

interface Pending {
  resolve: (v: unknown) => void;
  reject: (e: Error) => void;
  onProgress?: ProgressFn;
}

interface QueuedJob {
  job: string;
  args: unknown;
  transfer: Transferable[];
  pending: Pending;
}

let nextId = 1;

class ComputePool {
  private workers: { w: Worker; busy: boolean; current?: number }[] = [];
  private pending = new Map<number, Pending>();
  private queue: (QueuedJob & { id: number })[] = [];
  readonly size: number;

  constructor() {
    const hc = typeof navigator !== "undefined" ? navigator.hardwareConcurrency || 4 : 4;
    this.size = Math.max(1, Math.min(6, hc - 2));
  }

  private spawn() {
    const w = new Worker(new URL("../workers/compute.worker.ts", import.meta.url), { type: "module", name: "resurface-compute" });
    const slot = { w, busy: false as boolean, current: undefined as number | undefined };
    w.onmessage = (ev: MessageEvent) => {
      const { id, type, result, error, fraction, message } = ev.data;
      const p = this.pending.get(id);
      if (type === "progress") {
        p?.onProgress?.(fraction, message);
        return;
      }
      this.pending.delete(id);
      slot.busy = false;
      slot.current = undefined;
      if (error) p?.reject(new Error(error));
      else p?.resolve(result);
      this.pump();
    };
    w.onerror = (ev) => {
      const id = slot.current;
      if (id !== undefined) {
        this.pending.get(id)?.reject(new Error(ev.message || "Worker crashed"));
        this.pending.delete(id);
      }
      slot.busy = false;
      slot.current = undefined;
      this.pump();
    };
    this.workers.push(slot);
    return slot;
  }

  private pump() {
    while (this.queue.length) {
      let slot = this.workers.find((s) => !s.busy);
      if (!slot && this.workers.length < this.size) slot = this.spawn();
      if (!slot) return;
      const q = this.queue.shift()!;
      slot.busy = true;
      slot.current = q.id;
      this.pending.set(q.id, q.pending);
      slot.w.postMessage({ id: q.id, job: q.job, args: q.args }, q.transfer);
    }
  }

  run<T>(job: string, args: unknown, opts: { onProgress?: ProgressFn; transfer?: Transferable[] } = {}): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      this.queue.push({
        id: nextId++,
        job,
        args,
        transfer: opts.transfer ?? [],
        pending: { resolve: resolve as (v: unknown) => void, reject, onProgress: opts.onProgress },
      });
      this.pump();
    });
  }
}

export const compute = new ComputePool();
