/**
 * Lazily loaded optional modules. Using import.meta.glob keeps the build working while a module
 * is still being developed (a missing file simply reports "not available").
 */
const modules = import.meta.glob(["../sketch/profiles.ts", "../cad/KernelClient.ts", "../freeform/patches.ts", "../sketch/ops.ts", "../sketch/fit2d.ts"]);

export async function lazy<T>(path: string): Promise<T> {
  const loader = modules[path];
  if (!loader) throw new Error(`Module ${path.replace(/^\.\.\//, "")} is not available in this build yet`);
  return (await loader()) as T;
}
