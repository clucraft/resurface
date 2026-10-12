/// <reference lib="webworker" />
/**
 * Module Web Worker hosting the OpenCASCADE kernel. Spawned by KernelClient:
 *   new Worker(new URL("./kernel.worker.ts", import.meta.url), { type: "module" })
 * The wasm binary is emitted by Vite as an asset and located through its URL.
 */
import wasmUrl from "replicad-opencascadejs/wasm?url";
import { serveKernel, type MessagePortLike } from "./kernelRpc";
import { OcctKernel } from "./OcctKernel";

serveKernel(self as unknown as MessagePortLike, OcctKernel.create({ wasmUrl }));
