/**
 * Open-addressing hash map from an unsigned 32-bit pair (a, b) to a non-negative int32 value,
 * stored entirely in typed arrays (no Map, no per-entry objects). Used for edge -> new vertex
 * maps in clipping/sections etc. Grows automatically (load factor <= 0.5).
 *
 * The pair is ordered: (a, b) and (b, a) are different keys — callers normalise (min, max)
 * for undirected edges.
 */
import { fmix32 } from "./weld-hash";

export class PairHashMap {
  private keys: Uint32Array;
  private vals: Int32Array;
  private mask: number;
  private n = 0;

  constructor(expected = 64) {
    let cap = 16;
    while (cap < expected * 2) cap *= 2;
    this.keys = new Uint32Array(cap * 2);
    this.vals = new Int32Array(cap).fill(-1);
    this.mask = cap - 1;
  }

  get size(): number {
    return this.n;
  }

  private static h(a: number, b: number): number {
    return fmix32(Math.imul(a, 0x9e3779b1) ^ fmix32(b + 0x7f4a7c15));
  }

  /** Value for (a,b) or -1 if absent. */
  get(a: number, b: number): number {
    const keys = this.keys, vals = this.vals, mask = this.mask;
    let s = PairHashMap.h(a, b) & mask;
    for (;;) {
      const v = vals[s];
      if (v < 0) return -1;
      if (keys[2 * s] === a && keys[2 * s + 1] === b) return v;
      s = (s + 1) & mask;
    }
  }

  /**
   * Insert (a,b) -> value if absent. Returns the existing value when the key was already
   * present (the map is then unchanged), or -1 when the value was inserted.
   */
  insert(a: number, b: number, value: number): number {
    const keys = this.keys, vals = this.vals, mask = this.mask;
    let s = PairHashMap.h(a, b) & mask;
    for (;;) {
      const v = vals[s];
      if (v < 0) break;
      if (keys[2 * s] === a && keys[2 * s + 1] === b) return v;
      s = (s + 1) & mask;
    }
    keys[2 * s] = a;
    keys[2 * s + 1] = b;
    vals[s] = value;
    this.n++;
    if (this.n * 2 > vals.length) this.grow();
    return -1;
  }

  /** Insert or overwrite. */
  set(a: number, b: number, value: number): void {
    const keys = this.keys, vals = this.vals, mask = this.mask;
    let s = PairHashMap.h(a, b) & mask;
    for (;;) {
      const v = vals[s];
      if (v < 0) break;
      if (keys[2 * s] === a && keys[2 * s + 1] === b) {
        vals[s] = value;
        return;
      }
      s = (s + 1) & mask;
    }
    keys[2 * s] = a;
    keys[2 * s + 1] = b;
    vals[s] = value;
    this.n++;
    if (this.n * 2 > vals.length) this.grow();
  }

  private grow(): void {
    const oldK = this.keys, oldV = this.vals;
    const cap = oldV.length * 2;
    const keys = new Uint32Array(cap * 2);
    const vals = new Int32Array(cap).fill(-1);
    const mask = cap - 1;
    for (let s = 0; s < oldV.length; s++) {
      const v = oldV[s];
      if (v < 0) continue;
      const a = oldK[2 * s], b = oldK[2 * s + 1];
      let t = PairHashMap.h(a, b) & mask;
      while (vals[t] >= 0) t = (t + 1) & mask;
      keys[2 * t] = a;
      keys[2 * t + 1] = b;
      vals[t] = v;
    }
    this.keys = keys;
    this.vals = vals;
    this.mask = mask;
  }
}
