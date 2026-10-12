/**
 * Sparse symmetric helpers for the constraint solver: reverse Cuthill–McKee ordering and an envelope
 * (skyline / profile) Cholesky factorisation that drops numerically dependent rows. Dropping makes the
 * factorisation rank revealing (in elimination order), which the solver uses for redundancy / DOF.
 */

/** Reverse Cuthill–McKee ordering. Returns perm with perm[k] = original index placed at position k. */
export function rcmOrder(n: number, adj: ArrayLike<number>[]): Int32Array {
  const perm = new Int32Array(n);
  const visited = new Uint8Array(n);
  const deg = new Int32Array(n);
  for (let i = 0; i < n; i++) deg[i] = adj[i].length;
  let k = 0;
  const bfsLevels = (start: number): { order: number[]; last: number } => {
    // plain BFS used to find a pseudo-peripheral node (does not mark global visited)
    const seen = new Map<number, number>();
    seen.set(start, 0);
    const q = [start];
    let last = start;
    for (let h = 0; h < q.length; h++) {
      const v = q[h];
      last = v;
      for (let j = 0; j < adj[v].length; j++) {
        const w = adj[v][j];
        if (!seen.has(w)) {
          seen.set(w, seen.get(v)! + 1);
          q.push(w);
        }
      }
    }
    return { order: q, last };
  };
  for (let s = 0; s < n; s++) {
    if (visited[s]) continue;
    // pick a low-degree pseudo-peripheral start in this component
    let start = s;
    const comp = bfsLevels(s).order;
    for (const v of comp) if (deg[v] < deg[start]) start = v;
    start = bfsLevels(start).last;
    const q = [start];
    visited[start] = 1;
    for (let h = 0; h < q.length; h++) {
      const v = q[h];
      perm[k++] = v;
      const nb: number[] = [];
      for (let j = 0; j < adj[v].length; j++) {
        const w = adj[v][j];
        if (!visited[w]) {
          visited[w] = 1;
          nb.push(w);
        }
      }
      nb.sort((a, b) => deg[a] - deg[b]);
      for (const w of nb) q.push(w);
    }
  }
  perm.reverse();
  return perm;
}

/**
 * Symmetric matrix in envelope storage: row i holds columns fc[i]..i (lower triangle incl. diagonal)
 * at vals[rowStart[i] + (j - fc[i])].
 */
export class EnvelopeMatrix {
  readonly n: number;
  readonly fc: Int32Array;
  readonly rowStart: Int32Array;
  readonly vals: Float64Array;
  /** diagonal before factorisation (for relative pivot tests) */
  readonly diag0: Float64Array;
  /** 1 = row dropped as numerically dependent during factorisation */
  readonly dropped: Uint8Array;

  constructor(n: number, fc: Int32Array) {
    this.n = n;
    this.fc = fc;
    this.rowStart = new Int32Array(n + 1);
    for (let i = 0; i < n; i++) this.rowStart[i + 1] = this.rowStart[i] + (i - fc[i] + 1);
    this.vals = new Float64Array(this.rowStart[n]);
    this.diag0 = new Float64Array(n);
    this.dropped = new Uint8Array(n);
  }

  /** flat index of (i, j) with j <= i, j >= fc[i] */
  index(i: number, j: number): number {
    return this.rowStart[i] + (j - this.fc[i]);
  }

  clear(): void {
    this.vals.fill(0);
    this.dropped.fill(0);
  }

  get size(): number {
    return this.vals.length;
  }

  /**
   * In-place Cholesky A = L L^T. A row whose remaining pivot is <= relTol * (original diagonal) is
   * treated as dependent: it is removed from the system (its L row is zeroed, unit diagonal).
   * Returns the number of kept (independent) rows = numerical rank.
   */
  factor(relTol = 1e-10, absTol = 1e-300): number {
    const { n, fc, rowStart, vals, diag0, dropped } = this;
    for (let i = 0; i < n; i++) diag0[i] = vals[rowStart[i] + i - fc[i]];
    let rank = 0;
    for (let i = 0; i < n; i++) {
      const fi = fc[i];
      const ri = rowStart[i] - fi; // vals[ri + k] = L[i][k]
      for (let j = fi; j < i; j++) {
        if (dropped[j]) {
          vals[ri + j] = 0;
          continue;
        }
        const fj = fc[j];
        const rj = rowStart[j] - fj;
        const k0 = fi > fj ? fi : fj;
        let s = vals[ri + j];
        for (let k = k0; k < j; k++) s -= vals[ri + k] * vals[rj + k];
        vals[ri + j] = s / vals[rj + j];
      }
      let d = vals[ri + i];
      for (let k = fi; k < i; k++) d -= vals[ri + k] * vals[ri + k];
      if (!(d > relTol * diag0[i]) || !(d > absTol)) {
        dropped[i] = 1;
        for (let k = fi; k < i; k++) vals[ri + k] = 0;
        vals[ri + i] = 1;
      } else {
        vals[ri + i] = Math.sqrt(d);
        rank++;
      }
    }
    return rank;
  }

  /** Solve L L^T x = b after factor(); dropped rows get x = 0. */
  solve(b: Float64Array, out = new Float64Array(this.n)): Float64Array {
    const { n, fc, rowStart, vals, dropped } = this;
    const y = out;
    for (let i = 0; i < n; i++) {
      if (dropped[i]) {
        y[i] = 0;
        continue;
      }
      const fi = fc[i];
      const ri = rowStart[i] - fi;
      let s = b[i];
      for (let k = fi; k < i; k++) s -= vals[ri + k] * y[k];
      y[i] = s / vals[ri + i];
    }
    for (let i = n - 1; i >= 0; i--) {
      if (dropped[i]) {
        y[i] = 0;
        continue;
      }
      const fi = fc[i];
      const ri = rowStart[i] - fi;
      const xi = y[i] / vals[ri + i];
      y[i] = xi;
      for (let k = fi; k < i; k++) y[k] -= vals[ri + k] * xi;
    }
    return y;
  }
}
