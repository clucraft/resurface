/**
 * Minimal growable typed-array buffers (amortised O(1) push, doubling growth).
 *
 * Used by parsers and algorithms that do not know their output size up front. They never
 * create one JS object per element: storage is a single typed array that is reallocated
 * (copied) when full. `toArray()` returns a trimmed copy; `view()` returns a zero-copy
 * subarray of the current storage (invalidated by the next push that grows the buffer).
 */

type TypedCtor<T> = { new (n: number): T };

abstract class GrowableBase<T extends Float32Array | Float64Array | Uint32Array | Int32Array | Uint8Array> {
  data: T;
  length = 0;
  protected abstract ctor(): TypedCtor<T>;

  constructor(data: T) {
    this.data = data;
  }

  /** Make sure at least `extra` more elements fit without reallocating. */
  reserve(extra: number): void {
    const need = this.length + extra;
    if (need <= this.data.length) return;
    let cap = Math.max(16, this.data.length);
    while (cap < need) cap = cap < 1 << 26 ? cap * 2 : Math.ceil(cap * 1.5);
    const C = this.ctor();
    const next = new C(cap);
    next.set(this.data.subarray(0, this.length) as never);
    this.data = next;
  }

  push(v: number): void {
    if (this.length >= this.data.length) this.reserve(1);
    this.data[this.length++] = v;
  }

  push2(a: number, b: number): void {
    if (this.length + 2 > this.data.length) this.reserve(2);
    const d = this.data, l = this.length;
    d[l] = a;
    d[l + 1] = b;
    this.length = l + 2;
  }

  push3(a: number, b: number, c: number): void {
    if (this.length + 3 > this.data.length) this.reserve(3);
    const d = this.data, l = this.length;
    d[l] = a;
    d[l + 1] = b;
    d[l + 2] = c;
    this.length = l + 3;
  }

  /** Remove all elements (keeps capacity). */
  clear(): void {
    this.length = 0;
  }

  /** Trimmed copy of the contents. */
  toArray(): T {
    return this.data.slice(0, this.length) as T;
  }

  /**
   * Trimmed contents without copying when the storage is exactly full; otherwise a copy.
   * After calling this the builder should not be used any more.
   */
  finish(): T {
    if (this.data.length === this.length) return this.data;
    return this.toArray();
  }

  /** Zero-copy view of the current contents (invalidated by growth). */
  view(): T {
    return this.data.subarray(0, this.length) as T;
  }
}

export class GrowableF32 extends GrowableBase<Float32Array> {
  constructor(initialCapacity = 1024) {
    super(new Float32Array(Math.max(16, initialCapacity)));
  }
  protected ctor() {
    return Float32Array;
  }
}

export class GrowableF64 extends GrowableBase<Float64Array> {
  constructor(initialCapacity = 1024) {
    super(new Float64Array(Math.max(16, initialCapacity)));
  }
  protected ctor() {
    return Float64Array;
  }
}

export class GrowableU32 extends GrowableBase<Uint32Array> {
  constructor(initialCapacity = 1024) {
    super(new Uint32Array(Math.max(16, initialCapacity)));
  }
  protected ctor() {
    return Uint32Array;
  }
}

export class GrowableI32 extends GrowableBase<Int32Array> {
  constructor(initialCapacity = 1024) {
    super(new Int32Array(Math.max(16, initialCapacity)));
  }
  protected ctor() {
    return Int32Array;
  }
}

export class GrowableU8 extends GrowableBase<Uint8Array> {
  constructor(initialCapacity = 1024) {
    super(new Uint8Array(Math.max(16, initialCapacity)));
  }
  protected ctor() {
    return Uint8Array;
  }
}
