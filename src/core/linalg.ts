/**
 * Small dense linear-algebra kernels (row-major Float64Array matrices).
 * Good for the small/medium systems that show up in fitting (n up to a few thousand).
 */
import type { Vec3 } from "./types";

/** Solve A x = b in place via Gaussian elimination with partial pivoting. A is n*n row-major. Returns null if singular. */
export function solveLinear(A: Float64Array | number[], b: Float64Array | number[], n: number): Float64Array | null {
  const M = Float64Array.from(A as ArrayLike<number>);
  const x = Float64Array.from(b as ArrayLike<number>);
  for (let k = 0; k < n; k++) {
    let piv = k;
    let max = Math.abs(M[k * n + k]);
    for (let i = k + 1; i < n; i++) {
      const v = Math.abs(M[i * n + k]);
      if (v > max) {
        max = v;
        piv = i;
      }
    }
    if (max < 1e-300) return null;
    if (piv !== k) {
      for (let j = 0; j < n; j++) {
        const t = M[k * n + j];
        M[k * n + j] = M[piv * n + j];
        M[piv * n + j] = t;
      }
      const t = x[k];
      x[k] = x[piv];
      x[piv] = t;
    }
    const inv = 1 / M[k * n + k];
    for (let i = k + 1; i < n; i++) {
      const f = M[i * n + k] * inv;
      if (f === 0) continue;
      for (let j = k; j < n; j++) M[i * n + j] -= f * M[k * n + j];
      x[i] -= f * x[k];
    }
  }
  for (let i = n - 1; i >= 0; i--) {
    let s = x[i];
    for (let j = i + 1; j < n; j++) s -= M[i * n + j] * x[j];
    x[i] = s / M[i * n + i];
  }
  return x;
}

/**
 * Cholesky solve for symmetric positive (semi)definite A (n*n row-major).
 * Adds `ridge` to the diagonal for regularisation. Returns null on failure.
 */
export function choleskySolve(A: Float64Array | number[], b: Float64Array | number[], n: number, ridge = 0): Float64Array | null {
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i * n + j] + (i === j ? ridge : 0);
      for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
      if (i === j) {
        if (s <= 0) return null;
        L[i * n + i] = Math.sqrt(s);
      } else {
        L[i * n + j] = s / L[j * n + j];
      }
    }
  }
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i * n + k] * y[k];
    y[i] = s / L[i * n + i];
  }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k];
    x[i] = s / L[i * n + i];
  }
  return x;
}

/**
 * Linear least squares min ||J x - r|| via normal equations with Levenberg-style damping.
 * J is m*n row-major. Returns x (length n) or null.
 */
export function leastSquares(J: Float64Array, r: Float64Array, m: number, n: number, damping = 0): Float64Array | null {
  const JtJ = new Float64Array(n * n);
  const Jtr = new Float64Array(n);
  for (let i = 0; i < m; i++) {
    const row = i * n;
    const ri = r[i];
    for (let a = 0; a < n; a++) {
      const ja = J[row + a];
      if (ja === 0) continue;
      Jtr[a] += ja * ri;
      for (let b = a; b < n; b++) JtJ[a * n + b] += ja * J[row + b];
    }
  }
  for (let a = 0; a < n; a++) for (let b = 0; b < a; b++) JtJ[a * n + b] = JtJ[b * n + a];
  if (damping > 0) for (let a = 0; a < n; a++) JtJ[a * n + a] *= 1 + damping;
  return choleskySolve(JtJ, Jtr, n, 1e-12) ?? solveLinear(JtJ, Jtr, n);
}

/**
 * Eigen-decomposition of a symmetric 3x3 matrix (row-major, 9 numbers) with the cyclic Jacobi method.
 * Returns eigenvalues ascending and the matching unit eigenvectors.
 */
export function eigenSym3(m: ArrayLike<number>): { values: [number, number, number]; vectors: [Vec3, Vec3, Vec3] } {
  const a = [
    [m[0], m[1], m[2]],
    [m[3], m[4], m[5]],
    [m[6], m[7], m[8]],
  ];
  const v = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  for (let sweep = 0; sweep < 50; sweep++) {
    const off = Math.abs(a[0][1]) + Math.abs(a[0][2]) + Math.abs(a[1][2]);
    if (off < 1e-30) break;
    for (let p = 0; p < 2; p++)
      for (let q = p + 1; q < 3; q++) {
        const apq = a[p][q];
        if (Math.abs(apq) < 1e-300) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < 3; k++) {
          const akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < 3; k++) {
          const apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < 3; k++) {
          const vkp = v[k][p], vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
  }
  const idx = [0, 1, 2].sort((i, j) => a[i][i] - a[j][j]);
  const values = idx.map((i) => a[i][i]) as [number, number, number];
  const vectors = idx.map((i) => {
    const vec: Vec3 = [v[0][i], v[1][i], v[2][i]];
    const l = Math.hypot(vec[0], vec[1], vec[2]) || 1;
    return [vec[0] / l, vec[1] / l, vec[2] / l] as Vec3;
  }) as [Vec3, Vec3, Vec3];
  return { values, vectors };
}

/** Centroid and covariance (row-major 3x3) of a packed point array, optionally restricted to `indices`. */
export function covariance3(
  positions: ArrayLike<number>,
  indices?: ArrayLike<number>,
): { centroid: Vec3; cov: Float64Array; count: number } {
  const count = indices ? indices.length : positions.length / 3;
  let cx = 0, cy = 0, cz = 0;
  for (let k = 0; k < count; k++) {
    const i = indices ? indices[k] : k;
    cx += positions[3 * i];
    cy += positions[3 * i + 1];
    cz += positions[3 * i + 2];
  }
  cx /= count || 1;
  cy /= count || 1;
  cz /= count || 1;
  const cov = new Float64Array(9);
  for (let k = 0; k < count; k++) {
    const i = indices ? indices[k] : k;
    const x = positions[3 * i] - cx, y = positions[3 * i + 1] - cy, z = positions[3 * i + 2] - cz;
    cov[0] += x * x;
    cov[1] += x * y;
    cov[2] += x * z;
    cov[4] += y * y;
    cov[5] += y * z;
    cov[8] += z * z;
  }
  cov[3] = cov[1];
  cov[6] = cov[2];
  cov[7] = cov[5];
  for (let i = 0; i < 9; i++) cov[i] /= count || 1;
  return { centroid: [cx, cy, cz], cov, count };
}
