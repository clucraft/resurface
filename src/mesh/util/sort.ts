/**
 * Linear-time sorting helpers on typed arrays.
 */

/**
 * Stable LSD radix arg-sort of unsigned 32-bit keys (11-bit digits, ≤ 3 passes).
 * Returns the permutation `perm` such that keys[perm[0]] <= keys[perm[1]] <= ...
 *
 * `bits` (≤ 32) bounds the key width so narrower keys need fewer passes; digit passes in which
 * every key falls into one bucket are skipped. Memory: 4 temporary arrays of n uint32.
 */
export function radixArgsortU32(keys: ArrayLike<number>, bits = 32, n = keys.length): Uint32Array {
  let perm = new Uint32Array(n);
  let kA = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    perm[i] = i;
    kA[i] = keys[i];
  }
  if (n < 2) return perm;
  let perm2 = new Uint32Array(n);
  let kB = new Uint32Array(n);
  const RADIX = 11;
  const B = 1 << RADIX;
  const count = new Uint32Array(B);
  for (let shift = 0; shift < bits; shift += RADIX) {
    count.fill(0);
    for (let i = 0; i < n; i++) count[(kA[i] >>> shift) & (B - 1)]++;
    // skip pass if all keys share this digit
    let trivial = false;
    for (let d = 0; d < B; d++) {
      if (count[d] === n) {
        trivial = true;
        break;
      }
      if (count[d] !== 0) break;
    }
    if (trivial) continue;
    let sum = 0;
    for (let d = 0; d < B; d++) {
      const c = count[d];
      count[d] = sum;
      sum += c;
    }
    for (let i = 0; i < n; i++) {
      const k = kA[i];
      const pos = count[(k >>> shift) & (B - 1)]++;
      kB[pos] = k;
      perm2[pos] = perm[i];
    }
    let t = kA;
    kA = kB;
    kB = t;
    let tp = perm;
    perm = perm2;
    perm2 = tp;
  }
  return perm;
}

/**
 * Counting sort of items into buckets: given `bucketOf[i]` in [0, bucketCount), returns CSR
 * `{offsets, items}` where items[offsets[b] .. offsets[b+1]) lists the i with bucketOf[i] === b
 * in increasing i order. Entries with bucketOf[i] >= bucketCount are dropped.
 */
export function bucketize(bucketOf: ArrayLike<number>, bucketCount: number, n = bucketOf.length): { offsets: Uint32Array; items: Uint32Array } {
  const offsets = new Uint32Array(bucketCount + 1);
  for (let i = 0; i < n; i++) {
    const b = bucketOf[i];
    if (b < bucketCount) offsets[b + 1]++;
  }
  for (let b = 0; b < bucketCount; b++) offsets[b + 1] += offsets[b];
  const items = new Uint32Array(offsets[bucketCount]);
  const fill = offsets.slice(0, bucketCount);
  for (let i = 0; i < n; i++) {
    const b = bucketOf[i];
    if (b < bucketCount) items[fill[b]++] = i;
  }
  return { offsets, items };
}

/**
 * k-th smallest value of a numeric typed array sample (copies; O(n log n) sort of the copy).
 * Intended for medians of bounded-size samples.
 */
export function quantileOfSample(values: Float32Array | Float64Array, q: number): number {
  const n = values.length;
  if (n === 0) return 0;
  const copy = Float64Array.from(values);
  copy.sort();
  const idx = Math.min(n - 1, Math.max(0, Math.floor(q * (n - 1) + 0.5)));
  return copy[idx];
}
