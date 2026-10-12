/**
 * Helpers for SharedArrayBuffer-backed typed arrays. Meshes are immutable snapshots in the
 * document, so when the page is cross-origin isolated we keep their buffers in shared memory and
 * hand them to compute workers without copying.
 */
export const sharedMemoryAvailable = (): boolean =>
  typeof SharedArrayBuffer !== "undefined" && (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === true;

type TA = Float32Array | Float64Array | Uint32Array | Int32Array | Uint8Array | Uint16Array;

/** Return a copy of `a` backed by a SharedArrayBuffer (or `a` itself if already shared / unavailable). */
export function toShared<T extends TA>(a: T): T {
  if (!sharedMemoryAvailable()) return a;
  if (a.buffer instanceof SharedArrayBuffer) return a;
  const Ctor = a.constructor as new (buf: SharedArrayBuffer) => T;
  const sab = new SharedArrayBuffer(a.byteLength);
  const out = new Ctor(sab);
  out.set(a as never);
  return out;
}

/** Allocate a typed array in shared memory when available. */
export function allocShared<T extends TA>(Ctor: { new (n: number): T; new (b: SharedArrayBuffer): T; BYTES_PER_ELEMENT: number }, length: number): T {
  if (!sharedMemoryAvailable()) return new Ctor(length);
  return new Ctor(new SharedArrayBuffer(length * Ctor.BYTES_PER_ELEMENT));
}
