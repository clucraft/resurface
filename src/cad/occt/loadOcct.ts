/**
 * Loads (once) the OpenCASCADE 8 WebAssembly build shipped by `replicad-opencascadejs`.
 *
 * - Node (vitest): `await loadOcct()` — the Emscripten glue locates the .wasm next to itself.
 * - Browser module worker: pass the URL of the wasm asset, e.g.
 *     import wasmUrl from "replicad-opencascadejs/wasm?url";
 *     const oc = await loadOcct({ wasmUrl });
 *
 * OCCT prints transfer statistics on stdout; by default that output is swallowed
 * (pass `verbose: true` or custom `print`/`printErr` to see it).
 */
import init from "replicad-opencascadejs";
import type { OpenCascadeInstance } from "replicad-opencascadejs";

export type OC = OpenCascadeInstance;

export interface LoadOcctOptions {
  /** URL of `replicad_single.wasm` (required in bundled browser builds). */
  wasmUrl?: string;
  verbose?: boolean;
  print?: (line: string) => void;
  printErr?: (line: string) => void;
}

let cached: Promise<OC> | null = null;

export function loadOcct(opts: LoadOcctOptions = {}): Promise<OC> {
  if (!cached) {
    const noop = () => {};
    const moduleOpts: Record<string, unknown> = {
      print: opts.print ?? (opts.verbose ? (s: string) => console.log(s) : noop),
      printErr: opts.printErr ?? (opts.verbose ? (s: string) => console.warn(s) : noop),
    };
    if (opts.wasmUrl) {
      const url = opts.wasmUrl;
      moduleOpts.locateFile = (path: string, prefix: string) => (path.endsWith(".wasm") ? url : prefix + path);
    }
    cached = init(moduleOpts).catch((e: unknown) => {
      cached = null;
      throw e instanceof Error ? e : new Error(`Failed to load OpenCASCADE: ${String(e)}`);
    });
  }
  return cached;
}

/** True once `loadOcct` has been called (the promise may still be pending). */
export const isOcctLoading = (): boolean => cached !== null;
