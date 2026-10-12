/**
 * Generic, typed RPC between KernelClient (main thread) and the kernel worker.
 *
 * Protocol (structured-clone messages):
 *   main -> worker : { id, method, args }
 *   worker -> main : { kind: "response", id, ok: true, result } | { kind: "response", id, ok: false, error, stack? }
 *                    { kind: "ready" } | { kind: "init-error", error }
 * Typed arrays in results are transferred (zero-copy) when they own their buffer.
 *
 * To add a kernel method: declare it in `KernelApi` (kernelTypes.ts), implement it in
 * OcctKernel and append its name to `KERNEL_METHODS` (the compiler checks completeness).
 */
import type { KernelApi, KernelMethod } from "./kernelTypes";

export const KERNEL_METHODS = [
  "extrude",
  "extrudeWire",
  "revolve",
  "bsplineSurfaceFace",
  "bsplinePatchesToShape",
  "primitiveFace",
  "loft",
  "sweep",
  "boolean",
  "split",
  "fillet",
  "chamfer",
  "shell",
  "thicken",
  "transform",
  "mirror",
  "linearPattern",
  "circularPattern",
  "compound",
  "tessellate",
  "info",
  "exportSTEP",
  "exportIGES",
  "exportSTL",
  "serialize",
  "deserialize",
  "release",
  "releaseAll",
  "count",
] as const satisfies readonly KernelMethod[];

// Compile-time check: every KernelApi method is listed above.
type MissingMethods = Exclude<KernelMethod, (typeof KERNEL_METHODS)[number]>;
const allMethodsListed: [MissingMethods] extends [never] ? true : MissingMethods = true;
void allMethodsListed;

export interface RpcRequest {
  id: number;
  method: KernelMethod;
  args: unknown[];
}

export type RpcResponse =
  | { kind: "response"; id: number; ok: true; result: unknown }
  | { kind: "response"; id: number; ok: false; error: string; stack?: string };

export type WorkerMessage = RpcResponse | { kind: "ready" } | { kind: "init-error"; error: string };

/** The subset of a Worker / MessagePort / DedicatedWorkerGlobalScope we rely on. */
export interface MessagePortLike {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  onmessage: ((ev: MessageEvent) => unknown) | null;
}

/** ArrayBuffers of typed arrays reachable from `value` that can be transferred safely. */
export function collectTransferables(value: unknown, out: Transferable[] = [], seen = new Set<unknown>(), depth = 0): Transferable[] {
  if (value === null || typeof value !== "object" || depth > 6 || seen.has(value)) return out;
  seen.add(value);
  if (ArrayBuffer.isView(value)) {
    const buf = value.buffer;
    if (buf instanceof ArrayBuffer && value.byteOffset === 0 && value.byteLength === buf.byteLength && !seen.has(buf)) {
      seen.add(buf);
      out.push(buf);
    }
    return out;
  }
  if (Array.isArray(value)) for (const v of value) collectTransferables(v, out, seen, depth + 1);
  else for (const v of Object.values(value)) collectTransferables(v, out, seen, depth + 1);
  return out;
}

const errorText = (e: unknown) => (e instanceof Error ? e.message : String(e));

/**
 * Serve a kernel over a message port (inside the worker: `serveKernel(self, OcctKernel.create(...))`).
 * Requests are processed one at a time, in order.
 */
export function serveKernel(port: MessagePortLike, kernel: Promise<KernelApi>): void {
  const allowed = new Set<string>(KERNEL_METHODS);
  const post = (msg: WorkerMessage, transfer: Transferable[] = []) => port.postMessage(msg, transfer);
  kernel.then(
    () => post({ kind: "ready" }),
    (e) => post({ kind: "init-error", error: `CAD kernel initialisation failed: ${errorText(e)}` }),
  );
  let queue: Promise<void> = Promise.resolve();
  port.onmessage = (ev: MessageEvent) => {
    const req = ev.data as RpcRequest;
    if (!req || typeof req.id !== "number" || typeof req.method !== "string") return;
    queue = queue.then(async () => {
      try {
        const k = await kernel;
        if (!allowed.has(req.method)) throw new Error(`unknown kernel method '${req.method}'`);
        const fn = (k as unknown as Record<string, (...a: unknown[]) => unknown>)[req.method];
        const result = await fn.apply(k, Array.isArray(req.args) ? req.args : []);
        let transfer: Transferable[] = [];
        try {
          transfer = collectTransferables(result);
        } catch {
          transfer = [];
        }
        post({ kind: "response", id: req.id, ok: true, result }, transfer);
      } catch (e) {
        post({ kind: "response", id: req.id, ok: false, error: errorText(e), stack: e instanceof Error ? e.stack : undefined });
      }
    });
  };
}

export type AsyncKernelApi = {
  [K in KernelMethod]: (...args: Parameters<KernelApi[K]>) => Promise<ReturnType<KernelApi[K]>>;
};
