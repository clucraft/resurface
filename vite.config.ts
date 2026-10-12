/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Cross-origin isolation enables SharedArrayBuffer, which lets compute workers read large
// meshes without copying them.
const isolationHeaders = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
  base: "./",
  plugins: [react()],
  worker: {
    format: "es",
  },
  optimizeDeps: {
    // The OCCT wasm bundle is loaded by the kernel worker; keep Vite from pre-bundling it.
    exclude: ["replicad-opencascadejs"],
  },
  server: {
    port: 5173,
    headers: isolationHeaders,
  },
  preview: {
    headers: isolationHeaders,
  },
  build: {
    target: "es2022",
    chunkSizeWarningLimit: 4000,
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
    testTimeout: 60000,
  },
});
