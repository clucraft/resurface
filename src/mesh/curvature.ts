/**
 * Discrete curvature estimates (per vertex), linear time, ~44 B/vertex scratch.
 *
 * Method: for every mesh edge (i, j) the normal curvature along the edge direction is estimated
 * from the vertex normals,  κ_ij = (n_i − n_j)·(p_i − p_j) / |p_i − p_j|²  (positive where the
 * surface is convex w.r.t. outward normals). At each vertex the second fundamental form
 * II = [[a, b], [b, c]] in a tangent frame (u, v) is fitted by least squares to
 * κ_ij ≈ a·du² + 2b·du·dv + c·dv² over its edges (du, dv = unit edge direction projected to the
 * tangent plane) — this is unbiased by the distribution of edge directions (unlike averaging
 * κ_ij). Mean curvature H = (a + c)/2, Gaussian K = ac − b², principal k1,2 = H ± √(H² − K).
 * Vertices with fewer than three independent edge directions fall back to the mean of κ_ij.
 * Accumulation runs per triangle edge (no adjacency structure needed).
 */
import type { MeshData } from "../core/types";
import { computeVertexNormals, triangleCountOf, vertexCountOf } from "./topology";

export type CurvatureKind = "mean" | "signedMean" | "gaussian" | "rms" | "max";

export interface CurvatureOptions {
  /** Unit vertex normals; mesh.normals or area-weighted normals when omitted. */
  normals?: Float32Array;
  /**
   * 'mean' (default): |H|; 'signedMean': H (convex > 0); 'gaussian': K;
   * 'rms': √((k1² + k2²)/2); 'max': max(|k1|, |k2|).
   */
  kind?: CurvatureKind;
  /** Extra one-ring averaging passes applied to the result (noise reduction). Default 0. */
  smoothIterations?: number;
}

/** Per-vertex curvature (units 1/length) — see CurvatureOptions.kind. */
export function vertexCurvature(mesh: MeshData, opts: CurvatureOptions = {}): Float32Array {
  const P = mesh.positions, I = mesh.indices;
  const nv = vertexCountOf(mesh);
  const nt = triangleCountOf(mesh);
  const N = opts.normals ?? mesh.normals ?? computeVertexNormals(mesh);
  const kind = opts.kind ?? "mean";

  // tangent frame u per vertex (v = n × u computed on the fly)
  const U = new Float32Array(nv * 3);
  for (let v = 0; v < nv; v++) {
    const nx = N[3 * v], ny = N[3 * v + 1], nz = N[3 * v + 2];
    const ax = Math.abs(nx), ay = Math.abs(ny), az = Math.abs(nz);
    // u = normalize(n × e_k) with e_k the axis least aligned with n
    let ux: number, uy: number, uz: number;
    if (ax <= ay && ax <= az) {
      ux = 0; uy = nz; uz = -ny;
    } else if (ay <= az) {
      ux = -nz; uy = 0; uz = nx;
    } else {
      ux = ny; uy = -nx; uz = 0;
    }
    const l = Math.hypot(ux, uy, uz) || 1;
    U[3 * v] = ux / l;
    U[3 * v + 1] = uy / l;
    U[3 * v + 2] = uz / l;
  }
  // normal-equation accumulators per vertex: [Σu⁴, Σu³v, Σu²v², Σuv³, Σv⁴, Σκu², Σκuv, Σκv²] + Σκ, count
  const A = new Float32Array(nv * 8);
  const S = new Float32Array(nv);
  const cnt = new Uint16Array(nv);
  const accumulate = (i: number, dx: number, dy: number, dz: number, k: number) => {
    const nx = N[3 * i], ny = N[3 * i + 1], nz = N[3 * i + 2];
    const ux = U[3 * i], uy = U[3 * i + 1], uz = U[3 * i + 2];
    const vx = ny * uz - nz * uy, vy = nz * ux - nx * uz, vz = nx * uy - ny * ux;
    let du = dx * ux + dy * uy + dz * uz, dv = dx * vx + dy * vy + dz * vz;
    const l = Math.sqrt(du * du + dv * dv);
    if (!(l > 0)) return;
    du /= l;
    dv /= l;
    const u2 = du * du, v2 = dv * dv, uv = du * dv;
    const o = 8 * i;
    A[o] += u2 * u2;
    A[o + 1] += u2 * uv;
    A[o + 2] += u2 * v2;
    A[o + 3] += uv * v2;
    A[o + 4] += v2 * v2;
    A[o + 5] += k * u2;
    A[o + 6] += k * uv;
    A[o + 7] += k * v2;
    S[i] += k;
    if (cnt[i] < 65535) cnt[i]++;
  };
  for (let t = 0; t < nt; t++) {
    for (let e = 0; e < 3; e++) {
      const i = I[3 * t + e], j = I[3 * t + (e === 2 ? 0 : e + 1)];
      if (i === j || i >= nv || j >= nv) continue;
      const dx = P[3 * j] - P[3 * i], dy = P[3 * j + 1] - P[3 * i + 1], dz = P[3 * j + 2] - P[3 * i + 2];
      const l2 = dx * dx + dy * dy + dz * dz;
      if (!(l2 > 0)) continue;
      // κ_ij = (n_j − n_i)·(p_j − p_i)/|p_j − p_i|²  (= (n_i − n_j)·(p_i − p_j)/…, convex > 0)
      const k = ((N[3 * j] - N[3 * i]) * dx + (N[3 * j + 1] - N[3 * i + 1]) * dy + (N[3 * j + 2] - N[3 * i + 2]) * dz) / l2;
      accumulate(i, dx, dy, dz, k);
      accumulate(j, -dx, -dy, -dz, k);
    }
  }
  const out = new Float32Array(nv);
  for (let v = 0; v < nv; v++) {
    const o = 8 * v;
    // unknowns x = (a, b', c) with κ ≈ a·u² + b'·uv + c·v² (b' = 2b); normal equations Σ r rᵀ x = Σ κ r
    const m11 = A[o], m12 = A[o + 1], m13 = A[o + 2];
    const m22 = A[o + 2], m23 = A[o + 3], m33 = A[o + 4];
    const r1 = A[o + 5], r2 = A[o + 6], r3 = A[o + 7];
    const c11 = m22 * m33 - m23 * m23, c12 = m13 * m23 - m12 * m33, c13 = m12 * m23 - m13 * m22;
    const det = m11 * c11 + m12 * c12 + m13 * c13;
    let H: number, K: number;
    const tr = m11 + m22 + m33;
    if (cnt[v] >= 3 && Math.abs(det) > 1e-6 * tr * tr * tr) {
      const c22 = m11 * m33 - m13 * m13, c23 = m12 * m13 - m11 * m23, c33 = m11 * m22 - m12 * m12;
      const a = (c11 * r1 + c12 * r2 + c13 * r3) / det;
      const bp = (c12 * r1 + c22 * r2 + c23 * r3) / det;
      const c = (c13 * r1 + c23 * r2 + c33 * r3) / det;
      const b = bp / 2;
      H = (a + c) / 2;
      K = a * c - b * b;
    } else {
      H = cnt[v] ? S[v] / cnt[v] : 0;
      K = H * H;
    }
    let x: number;
    switch (kind) {
      case "signedMean":
        x = H;
        break;
      case "gaussian":
        x = K;
        break;
      case "rms":
        // (k1² + k2²)/2 = 2H² − K
        x = Math.sqrt(Math.max(0, 2 * H * H - K));
        break;
      case "max": {
        const d = Math.sqrt(Math.max(0, H * H - K));
        x = Math.max(Math.abs(H + d), Math.abs(H - d));
        break;
      }
      default:
        x = Math.abs(H);
    }
    out[v] = x;
  }
  const iters = opts.smoothIterations ?? 0;
  return iters > 0 ? smoothVertexScalar(mesh, out, iters) : out;
}

/**
 * One-ring averaging of a per-vertex scalar (each pass: value ← mean over incident triangle
 * corners, which includes the vertex itself). Returns a new array.
 */
export function smoothVertexScalar(mesh: MeshData, values: Float32Array, iterations: number): Float32Array {
  const I = mesh.indices;
  const nv = vertexCountOf(mesh);
  const nt = triangleCountOf(mesh);
  const cur = Float32Array.from(values);
  const sum = new Float32Array(nv);
  const cnt = new Float32Array(nv);
  for (let c = 0; c < nt * 3; c++) cnt[I[c]] += 3;
  for (let it = 0; it < iterations; it++) {
    sum.fill(0);
    for (let t = 0; t < nt; t++) {
      const a = I[3 * t], b = I[3 * t + 1], c = I[3 * t + 2];
      const s = cur[a] + cur[b] + cur[c];
      sum[a] += s;
      sum[b] += s;
      sum[c] += s;
    }
    for (let v = 0; v < nv; v++) if (cnt[v] > 0) cur[v] = sum[v] / cnt[v];
  }
  return cur;
}

/** Per-triangle value = mean of its three vertex values. */
export function vertexToTriangleScalar(mesh: MeshData, values: Float32Array): Float32Array {
  const I = mesh.indices;
  const nt = triangleCountOf(mesh);
  const out = new Float32Array(nt);
  for (let t = 0; t < nt; t++) out[t] = (values[I[3 * t]] + values[I[3 * t + 1]] + values[I[3 * t + 2]]) / 3;
  return out;
}
