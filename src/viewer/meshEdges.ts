/**
 * Unique undirected edges of an indexed triangle mesh, as vertex index pairs (a < b), for
 * drawing the triangle edges of a scan ("faceted STL" look) with an indexed LineSegments.
 *
 * Half-edges are bucketed by their lower vertex (CSR layout), then each bucket is de-duplicated:
 * linear in the triangle count with tiny buckets, and O(k log k) for the rare high-valence vertex.
 */
export function uniqueTriangleEdges(indices: ArrayLike<number>, vertexCount: number): Uint32Array {
  const n = indices.length - (indices.length % 3);
  // bucket sizes (upper bounds: degenerate and duplicate edges are dropped later)
  const start = new Uint32Array(vertexCount + 1);
  for (let i = 0; i < n; i += 3) {
    const a = indices[i], b = indices[i + 1], c = indices[i + 2];
    start[(a < b ? a : b) + 1]++;
    start[(b < c ? b : c) + 1]++;
    start[(c < a ? c : a) + 1]++;
  }
  for (let v = 0; v < vertexCount; v++) start[v + 1] += start[v];
  const fill = start.slice(0, vertexCount);
  const hiOf = new Uint32Array(Math.max(n, 1));
  for (let i = 0; i < n; i += 3) {
    for (let k = 0; k < 3; k++) {
      const p = indices[i + k], q = indices[i + (k === 2 ? 0 : k + 1)];
      if (p === q) continue;
      const lo = p < q ? p : q;
      hiOf[fill[lo]++] = p < q ? q : p;
    }
  }
  // de-duplicate each bucket, compacting in place (the write cursor never passes the read cursor)
  let w = 0;
  for (let v = 0; v < vertexCount; v++) {
    const s = start[v], e = fill[v];
    start[v] = w;
    if (e - s > 24) {
      hiOf.subarray(s, e).sort();
      let prev = -1;
      for (let j = s; j < e; j++) {
        const h = hiOf[j];
        if (h !== prev) hiOf[w++] = h;
        prev = h;
      }
    } else {
      const w0 = w;
      for (let j = s; j < e; j++) {
        const h = hiOf[j];
        let dup = false;
        for (let t = w0; t < w; t++) {
          if (hiOf[t] === h) {
            dup = true;
            break;
          }
        }
        if (!dup) hiOf[w++] = h;
      }
    }
    fill[v] = w;
  }
  const out = new Uint32Array(w * 2);
  let k = 0;
  for (let v = 0; v < vertexCount; v++) {
    for (let j = start[v], e = fill[v]; j < e; j++) {
      out[k++] = v;
      out[k++] = hiOf[j];
    }
  }
  return out;
}
