/**
 * Open-addressing hash map uint32 key → float64 value in typed arrays (no Map). Used for lazy
 * memoisation over a region of a large mesh where a dense per-element array would be wasteful.
 */
import { fmix32 } from "./weld-hash";

const EMPTY = 0xffffffff;

export class IntFloatMap {
  private keys: Uint32Array;
  private vals: Float64Array;
  private mask: number;
  private n = 0;

  constructor(expected = 64) {
    let cap = 16;
    while (cap < expected * 2) cap *= 2;
    this.keys = new Uint32Array(cap).fill(EMPTY);
    this.vals = new Float64Array(cap);
    this.mask = cap - 1;
  }

  get size(): number {
    return this.n;
  }

  /** Value for key, or NaN when absent. Keys must be < 0xffffffff. */
  get(key: number): number {
    const keys = this.keys;
    let s = fmix32(key) & this.mask;
    for (;;) {
      const k = keys[s];
      if (k === EMPTY) return NaN;
      if (k === key) return this.vals[s];
      s = (s + 1) & this.mask;
    }
  }

  has(key: number): boolean {
    const keys = this.keys;
    let s = fmix32(key) & this.mask;
    for (;;) {
      const k = keys[s];
      if (k === EMPTY) return false;
      if (k === key) return true;
      s = (s + 1) & this.mask;
    }
  }

  set(key: number, value: number): void {
    const keys = this.keys;
    let s = fmix32(key) & this.mask;
    for (;;) {
      const k = keys[s];
      if (k === EMPTY) break;
      if (k === key) {
        this.vals[s] = value;
        return;
      }
      s = (s + 1) & this.mask;
    }
    keys[s] = key;
    this.vals[s] = value;
    this.n++;
    if (this.n * 2 > keys.length) this.grow();
  }

  private grow(): void {
    const oldK = this.keys, oldV = this.vals;
    const cap = oldK.length * 2;
    const keys = new Uint32Array(cap).fill(EMPTY);
    const vals = new Float64Array(cap);
    const mask = cap - 1;
    for (let s = 0; s < oldK.length; s++) {
      const k = oldK[s];
      if (k === EMPTY) continue;
      let t = fmix32(k) & mask;
      while (keys[t] !== EMPTY) t = (t + 1) & mask;
      keys[t] = k;
      vals[t] = oldV[s];
    }
    this.keys = keys;
    this.vals = vals;
    this.mask = mask;
  }
}
