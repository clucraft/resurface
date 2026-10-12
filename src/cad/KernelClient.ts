/**
 * Main-thread facade of the CAD kernel. Spawns `kernel.worker.ts` (module worker running
 * OcctKernel on OpenCASCADE wasm) and exposes one async method per kernel method:
 *
 *   const k = new KernelClient();
 *   await k.ready();
 *   const id = await k.extrude({ frame, profile, distance: 10 });
 *   const mesh = await k.tessellate(id, { linearDeflection: 0.05, angularDeflectionDeg: 12 });
 *
 * `call(method, ...args)` is the generic entry point; failures reject with an Error carrying the
 * worker's message (e.g. "Extrude failed: profile loop 0 is not closed ...").
 */
import type { KernelApi, KernelMethod } from "./kernelTypes";
import { KERNEL_METHODS, type AsyncKernelApi, type MessagePortLike, type RpcRequest, type WorkerMessage } from "./kernelRpc";

export type { AsyncKernelApi } from "./kernelRpc";
export * from "./kernelTypes";

/** Anything that behaves like a Worker (a Worker, a MessagePort, a test double). */
export interface KernelTransport extends MessagePortLike {
  onerror?: ((ev: ErrorEvent | Event) => unknown) | null;
  terminate?(): void;
  close?(): void;
}

interface Pending {
  method: string;
  resolve: (v: unknown) => void;
  reject: (e: Error) => void;
}

// Declaration merging: the class gets one typed async method per kernel method (installed below).
export interface KernelClient extends AsyncKernelApi {}

export class KernelClient {
  private readonly transport: KernelTransport;
  private seq = 0;
  private pending = new Map<number, Pending>();
  private readonly readyPromise: Promise<void>;
  private settleReady!: (err?: Error) => void;
  private failure: Error | null = null;

  /** @param transport defaults to a new module worker running kernel.worker.ts */
  constructor(transport?: KernelTransport) {
    this.transport = transport ?? (new Worker(new URL("./kernel.worker.ts", import.meta.url), { type: "module" }) as unknown as KernelTransport);
    this.readyPromise = new Promise<void>((resolve, reject) => {
      this.settleReady = (err) => (err ? reject(err) : resolve());
    });
    this.readyPromise.catch(() => {}); // avoid unhandled rejections when nobody awaits ready()
    this.transport.onmessage = (ev: MessageEvent) => this.onMessage(ev.data as WorkerMessage);
    this.transport.onerror = (ev: ErrorEvent | Event) => {
      const msg = (ev as ErrorEvent).message || "kernel worker failed to load";
      this.fail(new Error(`CAD kernel worker error: ${msg}`));
    };
  }

  /** Resolves once OpenCASCADE is loaded in the worker; rejects if initialisation failed. */
  ready(): Promise<void> {
    return this.readyPromise;
  }

  /** Generic RPC call. */
  call<M extends KernelMethod>(method: M, ...args: Parameters<KernelApi[M]>): Promise<ReturnType<KernelApi[M]>> {
    if (this.failure) return Promise.reject(this.failure);
    const id = ++this.seq;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { method, resolve: resolve as (v: unknown) => void, reject });
      const req: RpcRequest = { id, method, args };
      try {
        this.transport.postMessage(req);
      } catch (e) {
        this.pending.delete(id);
        reject(new Error(`${method}: cannot send request (${e instanceof Error ? e.message : String(e)})`));
      }
    });
  }

  /** Number of requests awaiting a response. */
  get pendingCount(): number {
    return this.pending.size;
  }

  /** Stops the worker; pending calls reject. */
  terminate(): void {
    this.fail(new Error("CAD kernel was terminated"));
    this.transport.terminate?.();
    this.transport.close?.();
  }

  private fail(err: Error): void {
    if (!this.failure) this.failure = err;
    this.settleReady(err);
    for (const p of this.pending.values()) p.reject(err);
    this.pending.clear();
  }

  private onMessage(msg: WorkerMessage): void {
    if (!msg || typeof msg !== "object") return;
    switch (msg.kind) {
      case "ready":
        this.settleReady();
        return;
      case "init-error":
        this.fail(new Error(msg.error));
        return;
      case "response": {
        const p = this.pending.get(msg.id);
        if (!p) return;
        this.pending.delete(msg.id);
        if (msg.ok) p.resolve(msg.result);
        else {
          const err = new Error(msg.error);
          if (msg.stack) err.stack = `${msg.error}\n    [kernel worker] ${msg.stack.split("\n").slice(1).join("\n")}`;
          p.reject(err);
        }
      }
    }
  }
}

for (const method of KERNEL_METHODS) {
  Object.defineProperty(KernelClient.prototype, method, {
    configurable: true,
    writable: true,
    value: function (this: KernelClient, ...args: unknown[]) {
      return (this.call as (m: KernelMethod, ...a: unknown[]) => Promise<unknown>)(method, ...args);
    },
  });
}
