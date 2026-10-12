/** Large synthetic scans for performance testing (dev only). */
import type { MeshData } from "../core/types";

/** Bumpy closed sphere (UV parametrisation) with roughly `triangles` triangles and scan noise. */
export function bumpySphere(triangles: number, radius = 60): MeshData {
  const nu = Math.max(8, Math.round(Math.sqrt(triangles / 2) * 1.414));
  const nv = Math.max(4, Math.round(triangles / 2 / nu) + 1);
  const nVerts = nu * (nv - 1) + 2;
  const positions = new Float32Array(nVerts * 3);
  let seed = 12345;
  const rnd = () => {
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    return ((seed >>> 0) + 0.5) / 4294967296;
  };
  const r = (th: number, ph: number) => radius * (1 + 0.08 * Math.sin(5 * th) * Math.sin(4 * ph) + 0.03 * Math.cos(11 * th + 3 * ph));
  let k = 0;
  // poles
  positions.set([0, 0, r(0, 0)], 0);
  k = 1;
  for (let j = 1; j < nv; j++) {
    const ph = (Math.PI * j) / nv;
    for (let i = 0; i < nu; i++) {
      const th = (2 * Math.PI * i) / nu;
      const rr = r(th, ph) + (rnd() - 0.5) * 0.02;
      positions[3 * k] = rr * Math.sin(ph) * Math.cos(th);
      positions[3 * k + 1] = rr * Math.sin(ph) * Math.sin(th);
      positions[3 * k + 2] = rr * Math.cos(ph);
      k++;
    }
  }
  const south = k;
  positions.set([0, 0, -r(0, Math.PI)], 3 * south);
  const ring = (j: number, i: number) => 1 + (j - 1) * nu + (i % nu);
  const nt = 2 * nu * (nv - 2) + 2 * nu;
  const indices = new Uint32Array(nt * 3);
  let t = 0;
  for (let i = 0; i < nu; i++) {
    indices.set([0, ring(1, i), ring(1, i + 1)], 3 * t++);
  }
  for (let j = 1; j < nv - 1; j++)
    for (let i = 0; i < nu; i++) {
      const a = ring(j, i), b = ring(j + 1, i), c = ring(j + 1, i + 1), d = ring(j, i + 1);
      indices.set([a, b, c], 3 * t++);
      indices.set([a, c, d], 3 * t++);
    }
  for (let i = 0; i < nu; i++) indices.set([south, ring(nv - 1, i + 1), ring(nv - 1, i)], 3 * t++);
  return { positions, indices };
}
