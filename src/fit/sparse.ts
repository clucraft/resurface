/**
 * Banded linear solvers for the B-spline fitting normal equations (and collocation systems).
 */

/**
 * Symmetric positive-definite band matrix (lower band of half-bandwidth b) with in-place
 * Cholesky factorisation. Cost O(n b²), memory O(n b).
 */
export class BandSPD {
  readonly n: number;
  readonly b: number;
  /** lower band: entry (i, j), i-b <= j <= i, at i*(b+1) + (j - i + b) */
  readonly a: Float64Array;
  private factored = false;

  constructor(n: number, b: number) {
    this.n = n;
    this.b = Math.min(b, Math.max(0, n - 1));
    this.a = new Float64Array(n * (this.b + 1));
  }

  /** Add v to entry (i, j); requires i-b <= j <= i (lower triangle). */
  add(i: number, j: number, v: number): void {
    this.a[i * (this.b + 1) + (j - i + this.b)] += v;
  }

  get(i: number, j: number): number {
    if (j > i) [i, j] = [j, i];
    if (i - j > this.b) return 0;
    return this.a[i * (this.b + 1) + (j - i + this.b)];
  }

  /** Mean of the diagonal (for relative ridges). */
  meanDiagonal(): number {
    let s = 0;
    for (let i = 0; i < this.n; i++) s += this.a[i * (this.b + 1) + this.b];
    return s / Math.max(1, this.n);
  }

  addDiagonal(v: number): void {
    for (let i = 0; i < this.n; i++) this.a[i * (this.b + 1) + this.b] += v;
  }

  /** In-place Cholesky. Returns false if the matrix is not (numerically) positive definite. */
  factor(): boolean {
    const { n, b, a } = this;
    const w = b + 1;
    for (let i = 0; i < n; i++) {
      const j0 = Math.max(0, i - b);
      for (let j = j0; j <= i; j++) {
        let s = a[i * w + (j - i + b)];
        const k0 = Math.max(j0, j - b);
        const ri = i * w - i + b, rj = j * w - j + b;
        for (let k = k0; k < j; k++) s -= a[ri + k] * a[rj + k];
        if (i === j) {
          if (!(s > 0)) return false;
          a[i * w + b] = Math.sqrt(s);
        } else {
          a[i * w + (j - i + b)] = s / a[j * w + b];
        }
      }
    }
    this.factored = true;
    return true;
  }

  /** Solve in place (after factor()). */
  solve(x: Float64Array): void {
    if (!this.factored) throw new Error("BandSPD.solve: matrix not factored");
    const { n, b, a } = this;
    const w = b + 1;
    for (let i = 0; i < n; i++) {
      let s = x[i];
      const ri = i * w - i + b;
      for (let k = Math.max(0, i - b); k < i; k++) s -= a[ri + k] * x[k];
      x[i] = s / a[i * w + b];
    }
    for (let i = n - 1; i >= 0; i--) {
      let s = x[i];
      for (let k = i + 1; k <= Math.min(n - 1, i + b); k++) s -= a[k * w - k + b + i] * x[k];
      x[i] = s / a[i * w + b];
    }
  }
}

/**
 * General band matrix (kl sub-, ku super-diagonals) solved by Gaussian elimination WITHOUT
 * pivoting — stable for B-spline collocation matrices (totally positive). Multiple right-hand
 * sides are given as separate columns and overwritten with the solution.
 */
export class BandGeneral {
  readonly n: number;
  readonly kl: number;
  readonly ku: number;
  /** (i, j) at i*(kl+ku+1) + (j - i + kl) */
  readonly a: Float64Array;

  constructor(n: number, kl: number, ku: number) {
    this.n = n;
    this.kl = kl;
    this.ku = ku;
    this.a = new Float64Array(n * (kl + ku + 1));
  }

  set(i: number, j: number, v: number): void {
    const off = j - i + this.kl;
    if (off < 0 || off > this.kl + this.ku) throw new Error("BandGeneral.set: entry outside the band");
    this.a[i * (this.kl + this.ku + 1) + off] = v;
  }

  /** Solve A X = B in place for the given right-hand-side columns. Throws on a zero pivot. */
  solve(cols: Float64Array[]): void {
    const { n, kl, ku, a } = this;
    const w = kl + ku + 1;
    const at = (i: number, j: number) => i * w + (j - i + kl);
    for (let k = 0; k < n; k++) {
      const piv = a[at(k, k)];
      if (Math.abs(piv) < 1e-300) throw new Error("BandGeneral.solve: zero pivot (singular collocation matrix — duplicate parameters?)");
      for (let i = k + 1; i <= Math.min(n - 1, k + kl); i++) {
        const f = a[at(i, k)] / piv;
        if (f === 0) continue;
        a[at(i, k)] = 0;
        for (let j = k + 1; j <= Math.min(n - 1, k + ku); j++) a[at(i, j)] -= f * a[at(k, j)];
        for (const c of cols) c[i] -= f * c[k];
      }
    }
    for (const c of cols) {
      for (let i = n - 1; i >= 0; i--) {
        let s = c[i];
        for (let j = i + 1; j <= Math.min(n - 1, i + ku); j++) s -= a[at(i, j)] * c[j];
        c[i] = s / a[at(i, i)];
      }
    }
  }
}
