/**
 * Exact vertex welding with a typed-array open-addressing hash table keyed on the float bits
 * of the (Float32-rounded) position. No Map, no per-vertex JS objects.
 *
 * Memory: unique positions (12 B/vertex, grown by doubling) + hash table of Int32 slots
 * sized to keep the load factor <= 0.5 (≈ 8–16 B per unique vertex).
 *
 * -0 and +0 are treated as equal (they are normalised to +0). NaN coordinates never compare
 * equal, so every NaN vertex becomes its own vertex.
 */
import { GrowableF32 } from "./growable";

/** 32-bit finaliser from MurmurHash3 — good avalanche for float bit patterns. */
export function fmix32(h: number): number {
  h ^= h >>> 16;
  h = Math.imul(h, 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35);
  h ^= h >>> 16;
  return h >>> 0;
}

/** Hash three uint32 words. */
export function hash3u(a: number, b: number, c: number): number {
  let h = fmix32(a ^ 0x9e3779b9);
  h = fmix32(h ^ Math.imul(b, 0x85ebca6b) ^ 0x27d4eb2f);
  h = fmix32(h ^ Math.imul(c, 0xc2b2ae35) ^ 0x165667b1);
  return h;
}

export class ExactVertexWelder {
  /** unique positions, xyz packed */
  private readonly pos: GrowableF32;
  private table: Int32Array;
  private mask: number;
  private n = 0;
  // scratch for float -> bits conversion (per instance; not shared global state)
  private readonly f32 = new Float32Array(3);
  private readonly u32 = new Uint32Array(this.f32.buffer);

  constructor(expectedVertices = 1024) {
    const ev = Math.max(16, expectedVertices | 0);
    this.pos = new GrowableF32(ev * 3);
    let cap = 32;
    while (cap < ev * 2) cap *= 2;
    this.table = new Int32Array(cap).fill(-1);
    this.mask = cap - 1;
  }

  /** Number of unique vertices so far. */
  get count(): number {
    return this.n;
  }

  /**
   * Return the index of the vertex at (x,y,z) (rounded to float32), inserting it if new.
   */
  add(x: number, y: number, z: number): number {
    const f = this.f32, u = this.u32;
    // +0 normalises -0 to +0
    f[0] = x + 0;
    f[1] = y + 0;
    f[2] = z + 0;
    const fx = f[0], fy = f[1], fz = f[2];
    let slot = hash3u(u[0], u[1], u[2]) & this.mask;
    const table = this.table;
    const p = this.pos.data;
    for (;;) {
      const id = table[slot];
      if (id < 0) break;
      const o = id * 3;
      if (p[o] === fx && p[o + 1] === fy && p[o + 2] === fz) return id;
      slot = (slot + 1) & this.mask;
    }
    const id = this.n++;
    this.pos.push3(fx, fy, fz);
    table[slot] = id;
    if (this.n * 2 > table.length) this.grow();
    return id;
  }

  private grow(): void {
    const cap = this.table.length * 2;
    const table = new Int32Array(cap).fill(-1);
    const mask = cap - 1;
    const p = this.pos.data;
    const f = this.f32, u = this.u32;
    for (let id = 0; id < this.n; id++) {
      const o = id * 3;
      f[0] = p[o];
      f[1] = p[o + 1];
      f[2] = p[o + 2];
      let slot = hash3u(u[0], u[1], u[2]) & mask;
      while (table[slot] >= 0) slot = (slot + 1) & mask;
      table[slot] = id;
    }
    this.table = table;
    this.mask = mask;
  }

  /** Trimmed copy of the unique positions. */
  positions(): Float32Array {
    return this.pos.toArray();
  }

  /** Release the hash table (positions remain available). */
  releaseTable(): void {
    this.table = new Int32Array(0);
    this.mask = 0;
  }
}

/**
 * Exact weld of an indexed vertex array: returns the unique positions and a remap
 * (old vertex index -> new vertex index). New vertices are numbered in order of first
 * occurrence.
 */
export function weldExactPositions(positions: Float32Array): { positions: Float32Array; remap: Uint32Array } {
  const nv = positions.length / 3;
  const welder = new ExactVertexWelder(nv);
  const remap = new Uint32Array(nv);
  for (let i = 0; i < nv; i++) {
    remap[i] = welder.add(positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]);
  }
  return { positions: welder.positions(), remap };
}
