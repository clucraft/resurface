import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
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
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
    testTimeout: 60000,
  },
});
