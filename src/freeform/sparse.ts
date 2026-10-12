/**
 * Small sparse linear-algebra toolkit used by the freeform module.
 *
 *  - `CsrMatrix`  compressed-sparse-row matrix (Float64 values)
 *  - `CsrBuilder` row-by-row assembly that merges duplicate column entries
 *  - sparse x sparse product (Gustavson), transpose, row selection, mat-vec on xyz data
 *  - `Bsr3Matrix` block-CSR matrix with 3x3 blocks and a block-Jacobi preconditioned
 *    conjugate-gradient solver (used for the coupled point-to-plane fitting system)
 *
 * Pure TypeScript, no dependencies; deterministic.
 */
import type { Vec3 } from "../core/types";

export interface CsrMatrix {
  nRows: number;
  nCols: number;
  /** length nRows + 1 */
  rowPtr: Int32Array;
  /** column index per non-zero (sorted ascending within each row) */
  colIdx: Int32Array;
  values: Float64Array;
}

/**
 * Incremental CSR assembly. Call `add(col, w)` any number of times (duplicates are summed),
 * then `endRow()`; finally `build()`.
 */
export class CsrBuilder {
  readonly nCols: number;
  private cols: Int32Array;
  private vals: Float64Array;
  private nnz = 0;
  private rowPtr: number[] = [0];
  /** marker[col] = position of col in the current row (valid if >= rowStart) */
  private marker: Int32Array;
  private rowStart = 0;

  constructor(nCols: number, capacity = 64) {
    this.nCols = nCols;
    const cap = Math.max(16, capacity | 0);
    this.cols = new Int32Array(cap);
    this.vals = new Float64Array(cap);
    this.marker = new Int32Array(Math.max(1, nCols)).fill(-1);
  }

  get rowCount(): number {
    return this.rowPtr.length - 1;
  }

  add(col: number, w: number): void {
    const m = this.marker[col];
    if (m >= this.rowStart) {
      this.vals[m] += w;
      return;
    }
    if (this.nnz === this.cols.length) this.grow();
    this.marker[col] = this.nnz;
    this.cols[this.nnz] = col;
    this.vals[this.nnz] = w;
    this.nnz++;
  }

  endRow(): void {
    const s = this.rowStart, e = this.nnz;
    const cols = this.cols, vals = this.vals;
    // insertion sort by column (rows are short)
    for (let i = s + 1; i < e; i++) {
      const c = cols[i], v = vals[i];
      let j = i - 1;
      while (j >= s && cols[j] > c) {
        cols[j + 1] = cols[j];
        vals[j + 1] = vals[j];
        j--;
      }
      cols[j + 1] = c;
      vals[j + 1] = v;
    }
    this.rowPtr.push(e);
    this.rowStart = e;
  }

  build(): CsrMatrix {
    if (this.rowStart !== this.nnz) this.endRow();
    return {
      nRows: this.rowPtr.length - 1,
      nCols: this.nCols,
      rowPtr: Int32Array.from(this.rowPtr),
      colIdx: this.cols.slice(0, this.nnz),
      values: this.vals.slice(0, this.nnz),
    };
  }

  private grow(): void {
    const cap = this.cols.length * 2;
    const c = new Int32Array(cap);
    c.set(this.cols);
    const v = new Float64Array(cap);
    v.set(this.vals);
    this.cols = c;
    this.vals = v;
  }
}

export function csrIdentity(n: number): CsrMatrix {
  const rowPtr = new Int32Array(n + 1);
  const colIdx = new Int32Array(n);
  const values = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    rowPtr[i + 1] = i + 1;
    colIdx[i] = i;
    values[i] = 1;
  }
  return { nRows: n, nCols: n, rowPtr, colIdx, values };
}

/** C = A * B (Gustavson). */
export function csrMultiply(A: CsrMatrix, B: CsrMatrix): CsrMatrix {
  if (A.nCols !== B.nRows) throw new Error(`csrMultiply: dimension mismatch ${A.nCols} vs ${B.nRows}`);
  const b = new CsrBuilder(B.nCols, Math.max(64, A.colIdx.length * 4));
  for (let i = 0; i < A.nRows; i++) {
    for (let p = A.rowPtr[i]; p < A.rowPtr[i + 1]; p++) {
      const k = A.colIdx[p];
      const a = A.values[p];
      if (a === 0) continue;
      for (let q = B.rowPtr[k]; q < B.rowPtr[k + 1]; q++) b.add(B.colIdx[q], a * B.values[q]);
    }
    b.endRow();
  }
  return b.build();
}

export function csrTranspose(A: CsrMatrix): CsrMatrix {
  const nnz = A.colIdx.length;
  const rowPtr = new Int32Array(A.nCols + 1);
  for (let p = 0; p < nnz; p++) rowPtr[A.colIdx[p] + 1]++;
  for (let i = 0; i < A.nCols; i++) rowPtr[i + 1] += rowPtr[i];
  const fill = rowPtr.slice(0, A.nCols);
  const colIdx = new Int32Array(nnz);
  const values = new Float64Array(nnz);
  for (let i = 0; i < A.nRows; i++) {
    for (let p = A.rowPtr[i]; p < A.rowPtr[i + 1]; p++) {
      const c = A.colIdx[p];
      const dst = fill[c]++;
      colIdx[dst] = i;
      values[dst] = A.values[p];
    }
  }
  return { nRows: A.nCols, nCols: A.nRows, rowPtr, colIdx, values };
}

/** Matrix made of the given rows of A (in the given order). */
export function csrSelectRows(A: CsrMatrix, rows: ArrayLike<number>): CsrMatrix {
  let nnz = 0;
  for (let k = 0; k < rows.length; k++) nnz += A.rowPtr[rows[k] + 1] - A.rowPtr[rows[k]];
  const rowPtr = new Int32Array(rows.length + 1);
  const colIdx = new Int32Array(nnz);
  const values = new Float64Array(nnz);
  let o = 0;
  for (let k = 0; k < rows.length; k++) {
    const r = rows[k];
    for (let p = A.rowPtr[r]; p < A.rowPtr[r + 1]; p++) {
      colIdx[o] = A.colIdx[p];
      values[o] = A.values[p];
      o++;
    }
    rowPtr[k + 1] = o;
  }
  return { nRows: rows.length, nCols: A.nCols, rowPtr, colIdx, values };
}

/** Apply A to a list of points (each column of the point matrix independently). */
export function csrMulVec3(A: CsrMatrix, P: ArrayLike<Vec3>): Vec3[] {
  const out: Vec3[] = new Array(A.nRows);
  for (let i = 0; i < A.nRows; i++) {
    let x = 0, y = 0, z = 0;
    for (let p = A.rowPtr[i]; p < A.rowPtr[i + 1]; p++) {
      const w = A.values[p];
      const q = P[A.colIdx[p]];
      x += w * q[0];
      y += w * q[1];
      z += w * q[2];
    }
    out[i] = [x, y, z];
  }
  return out;
}

/** Apply A to packed xyz data (x.length = 3 * nCols). */
export function csrMulPacked3(A: CsrMatrix, x: ArrayLike<number>, out?: Float64Array): Float64Array {
  const y = out ?? new Float64Array(3 * A.nRows);
  for (let i = 0; i < A.nRows; i++) {
    let sx = 0, sy = 0, sz = 0;
    for (let p = A.rowPtr[i]; p < A.rowPtr[i + 1]; p++) {
      const w = A.values[p];
      const c = 3 * A.colIdx[p];
      sx += w * x[c];
      sy += w * x[c + 1];
      sz += w * x[c + 2];
    }
    y[3 * i] = sx;
    y[3 * i + 1] = sy;
    y[3 * i + 2] = sz;
  }
  return y;
}

/** Scalar mat-vec y = A x. */
export function csrMulVec(A: CsrMatrix, x: ArrayLike<number>, out?: Float64Array): Float64Array {
  const y = out ?? new Float64Array(A.nRows);
  for (let i = 0; i < A.nRows; i++) {
    let s = 0;
    for (let p = A.rowPtr[i]; p < A.rowPtr[i + 1]; p++) s += A.values[p] * x[A.colIdx[p]];
    y[i] = s;
  }
  return y;
}

/** Dense copy (tests / debugging only). */
export function csrToDense(A: CsrMatrix): Float64Array[] {
  const rows: Float64Array[] = [];
  for (let i = 0; i < A.nRows; i++) {
    const r = new Float64Array(A.nCols);
    for (let p = A.rowPtr[i]; p < A.rowPtr[i + 1]; p++) r[A.colIdx[p]] += A.values[p];
    rows.push(r);
  }
  return rows;
}

// ---------------------------------------------------------------------------
// Block CSR with 3x3 blocks
// ---------------------------------------------------------------------------

export interface Bsr3Matrix {
  /** number of block rows / columns */
  n: number;
  rowPtr: Int32Array;
  colIdx: Int32Array;
  /** 9 values per block, row-major 3x3 */
  values: Float64Array;
  /** block index of the diagonal block of each row */
  diagPos: Int32Array;
}

/** Build an all-zero BSR matrix from per-row column lists (need not be sorted/unique; the diagonal is always added). */
export function bsr3FromRowColumns(n: number, rowCols: ArrayLike<number>[]): Bsr3Matrix {
  const rowPtr = new Int32Array(n + 1);
  const lists: Int32Array[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const src = rowCols[i] ?? [];
    const tmp = new Int32Array(src.length + 1);
    for (let k = 0; k < src.length; k++) tmp[k] = src[k];
    tmp[src.length] = i;
    tmp.sort();
    let u = 0;
    for (let k = 0; k < tmp.length; k++) if (k === 0 || tmp[k] !== tmp[k - 1]) tmp[u++] = tmp[k];
    lists[i] = tmp.subarray(0, u);
    rowPtr[i + 1] = rowPtr[i] + u;
  }
  const colIdx = new Int32Array(rowPtr[n]);
  const diagPos = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    colIdx.set(lists[i], rowPtr[i]);
    for (let p = rowPtr[i]; p < rowPtr[i + 1]; p++) if (colIdx[p] === i) diagPos[i] = p;
  }
  return { n, rowPtr, colIdx, values: new Float64Array(9 * rowPtr[n]), diagPos };
}

/** Block index of (i, j) or -1. */
export function bsr3BlockIndex(A: Bsr3Matrix, i: number, j: number): number {
  let lo = A.rowPtr[i], hi = A.rowPtr[i + 1] - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const c = A.colIdx[mid];
    if (c === j) return mid;
    if (c < j) lo = mid + 1;
    else hi = mid - 1;
  }
  return -1;
}

export function bsr3MulVec(A: Bsr3Matrix, x: Float64Array, y: Float64Array): void {
  const v = A.values;
  for (let i = 0; i < A.n; i++) {
    let s0 = 0, s1 = 0, s2 = 0;
    for (let p = A.rowPtr[i]; p < A.rowPtr[i + 1]; p++) {
      const c = 3 * A.colIdx[p];
      const o = 9 * p;
      const x0 = x[c], x1 = x[c + 1], x2 = x[c + 2];
      s0 += v[o] * x0 + v[o + 1] * x1 + v[o + 2] * x2;
      s1 += v[o + 3] * x0 + v[o + 4] * x1 + v[o + 5] * x2;
      s2 += v[o + 6] * x0 + v[o + 7] * x1 + v[o + 8] * x2;
    }
    y[3 * i] = s0;
    y[3 * i + 1] = s1;
    y[3 * i + 2] = s2;
  }
}

export interface PcgResult {
  iterations: number;
  /** final ||b - A x|| / ||b|| */
  relResidual: number;
  converged: boolean;
}

/** Invert a 3x3 (row-major) matrix into out; returns false if (nearly) singular. */
function invert3(m: Float64Array, o: number, out: Float64Array, oo: number): boolean {
  const a = m[o], b = m[o + 1], c = m[o + 2];
  const d = m[o + 3], e = m[o + 4], f = m[o + 5];
  const g = m[o + 6], h = m[o + 7], i = m[o + 8];
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  const scaleRef = Math.abs(a) + Math.abs(e) + Math.abs(i);
  if (!(Math.abs(det) > 1e-14 * scaleRef * scaleRef * scaleRef) || !Number.isFinite(det)) return false;
  const inv = 1 / det;
  out[oo] = A * inv;
  out[oo + 1] = -(b * i - c * h) * inv;
  out[oo + 2] = (b * f - c * e) * inv;
  out[oo + 3] = B * inv;
  out[oo + 4] = (a * i - c * g) * inv;
  out[oo + 5] = -(a * f - c * d) * inv;
  out[oo + 6] = C * inv;
  out[oo + 7] = -(a * h - b * g) * inv;
  out[oo + 8] = (a * e - b * d) * inv;
  return true;
}

/**
 * Solve A x = b for symmetric positive definite A with block-Jacobi preconditioned CG.
 * `x` holds the initial guess and receives the solution.
 */
export function pcgBsr3(A: Bsr3Matrix, b: Float64Array, x: Float64Array, tol = 1e-10, maxIter = 2000): PcgResult {
  const n3 = 3 * A.n;
  const Minv = new Float64Array(9 * A.n);
  for (let i = 0; i < A.n; i++) {
    if (!invert3(A.values, 9 * A.diagPos[i], Minv, 9 * i)) {
      const o = 9 * A.diagPos[i];
      const tr = (Math.abs(A.values[o]) + Math.abs(A.values[o + 4]) + Math.abs(A.values[o + 8])) / 3;
      const s = tr > 1e-300 ? 1 / tr : 1;
      Minv.fill(0, 9 * i, 9 * i + 9);
      Minv[9 * i] = Minv[9 * i + 4] = Minv[9 * i + 8] = s;
    }
  }
  const r = new Float64Array(n3);
  const z = new Float64Array(n3);
  const p = new Float64Array(n3);
  const Ap = new Float64Array(n3);
  const applyM = (src: Float64Array, dst: Float64Array) => {
    for (let i = 0; i < A.n; i++) {
      const o = 9 * i, c = 3 * i;
      const s0 = src[c], s1 = src[c + 1], s2 = src[c + 2];
      dst[c] = Minv[o] * s0 + Minv[o + 1] * s1 + Minv[o + 2] * s2;
      dst[c + 1] = Minv[o + 3] * s0 + Minv[o + 4] * s1 + Minv[o + 5] * s2;
      dst[c + 2] = Minv[o + 6] * s0 + Minv[o + 7] * s1 + Minv[o + 8] * s2;
    }
  };
  let bNorm = 0;
  for (let i = 0; i < n3; i++) bNorm += b[i] * b[i];
  bNorm = Math.sqrt(bNorm);
  bsr3MulVec(A, x, Ap);
  let rNorm = 0;
  for (let i = 0; i < n3; i++) {
    r[i] = b[i] - Ap[i];
    rNorm += r[i] * r[i];
  }
  rNorm = Math.sqrt(rNorm);
  if (bNorm === 0) {
    if (rNorm === 0) return { iterations: 0, relResidual: 0, converged: true };
    bNorm = 1;
  }
  if (rNorm / bNorm <= tol) return { iterations: 0, relResidual: rNorm / bNorm, converged: true };
  applyM(r, z);
  p.set(z);
  let rz = 0;
  for (let i = 0; i < n3; i++) rz += r[i] * z[i];
  let it = 0;
  for (; it < maxIter; it++) {
    bsr3MulVec(A, p, Ap);
    let pAp = 0;
    for (let i = 0; i < n3; i++) pAp += p[i] * Ap[i];
    if (!(pAp > 0)) break;
    const alpha = rz / pAp;
    rNorm = 0;
    for (let i = 0; i < n3; i++) {
      x[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
      rNorm += r[i] * r[i];
    }
    rNorm = Math.sqrt(rNorm);
    if (rNorm / bNorm <= tol) {
      it++;
      break;
    }
    applyM(r, z);
    let rzNew = 0;
    for (let i = 0; i < n3; i++) rzNew += r[i] * z[i];
    const beta = rzNew / rz;
    rz = rzNew;
    for (let i = 0; i < n3; i++) p[i] = z[i] + beta * p[i];
  }
  return { iterations: it, relResidual: rNorm / bNorm, converged: rNorm / bNorm <= tol };
}
