/**
 * Binary min-heap of (key: float64, id: uint32) pairs stored in typed arrays.
 * No decrease-key: callers use lazy re-evaluation (pop, check freshness, re-push).
 */
export class MinHeap {
  private keys: Float64Array;
  private ids: Uint32Array;
  size = 0;

  constructor(capacity = 1024) {
    this.keys = new Float64Array(Math.max(16, capacity));
    this.ids = new Uint32Array(Math.max(16, capacity));
  }

  private growTo(cap: number): void {
    const k = new Float64Array(cap);
    k.set(this.keys.subarray(0, this.size));
    const i = new Uint32Array(cap);
    i.set(this.ids.subarray(0, this.size));
    this.keys = k;
    this.ids = i;
  }

  push(key: number, id: number): void {
    if (this.size >= this.keys.length) this.growTo(this.keys.length * 2);
    const keys = this.keys, ids = this.ids;
    let i = this.size++;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (keys[p] <= key) break;
      keys[i] = keys[p];
      ids[i] = ids[p];
      i = p;
    }
    keys[i] = key;
    ids[i] = id;
  }

  /** Key of the minimum (Infinity when empty). */
  peekKey(): number {
    return this.size > 0 ? this.keys[0] : Infinity;
  }

  /** Id of the minimum (undefined behaviour when empty). */
  peekId(): number {
    return this.ids[0];
  }

  /** Remove the minimum; returns its id (or -1 when empty). Use peekKey() first to read its key. */
  pop(): number {
    if (this.size === 0) return -1;
    const keys = this.keys, ids = this.ids;
    const top = ids[0];
    const n = --this.size;
    if (n > 0) {
      const key = keys[n], id = ids[n];
      let i = 0;
      for (;;) {
        let c = 2 * i + 1;
        if (c >= n) break;
        if (c + 1 < n && keys[c + 1] < keys[c]) c++;
        if (keys[c] >= key) break;
        keys[i] = keys[c];
        ids[i] = ids[c];
        i = c;
      }
      keys[i] = key;
      ids[i] = id;
    }
    return top;
  }

  /**
   * Build the heap from arrays in O(n) (Floyd). Replaces current content.
   */
  static from(keys: Float64Array, ids: Uint32Array, n = keys.length): MinHeap {
    const h = new MinHeap(n);
    h.keys.set(keys.subarray(0, n));
    h.ids.set(ids.subarray(0, n));
    h.size = n;
    for (let i = (n >> 1) - 1; i >= 0; i--) h.siftDown(i);
    return h;
  }

  private siftDown(i: number): void {
    const keys = this.keys, ids = this.ids, n = this.size;
    const key = keys[i], id = ids[i];
    for (;;) {
      let c = 2 * i + 1;
      if (c >= n) break;
      if (c + 1 < n && keys[c + 1] < keys[c]) c++;
      if (keys[c] >= key) break;
      keys[i] = keys[c];
      ids[i] = ids[c];
      i = c;
    }
    keys[i] = key;
    ids[i] = id;
  }
}
