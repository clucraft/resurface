/**
 * Plane clipping of a triangle mesh.
 */
import type { MeshData, PlaneDef } from "../core/types";
import { averageEdgeLength, compactMesh, triangleCountOf, vertexCountOf } from "./topology";
import { GrowableF32 } from "./util/growable";
import { PairHashMap } from "./util/pair-hash";

export interface ClipOptions {
  /**
   * Vertices with |signed distance| <= epsilon count as lying on the plane (they are neither
   * cut nor moved). Default: 1e-6 × average edge length.
   */
  epsilon?: number;
}

/**
 * Keep the part of the mesh on one side of `plane` (`'positive'` = the side the normal points
 * to). With `cut = true`, triangles straddling the plane are split exactly at the plane; the new
 * vertices on an edge are shared by both adjacent triangles, so the cut stays watertight
 * (normals/colours are interpolated). With `cut = false`, straddling triangles are kept or
 * dropped whole depending on which side their centroid lies.
 *
 * Triangles lying in the plane are kept. Unreferenced vertices are removed.
 */
export function clipByPlane(mesh: MeshData, plane: PlaneDef, keep: "positive" | "negative", cut: boolean, opts: ClipOptions = {}): MeshData {
  const P = mesh.positions, I = mesh.indices;
  const nt = triangleCountOf(mesh);
  const nv = vertexCountOf(mesh);
  const sgn = keep === "positive" ? 1 : -1;
  const nx = plane.normal[0] * sgn, ny = plane.normal[1] * sgn, nz = plane.normal[2] * sgn;
  const ox = plane.origin[0], oy = plane.origin[1], oz = plane.origin[2];
  const eps = opts.epsilon ?? 1e-6 * averageEdgeLength(mesh, 100_000);
  const dist = (v: number) => (P[3 * v] - ox) * nx + (P[3 * v + 1] - oy) * ny + (P[3 * v + 2] - oz) * nz;
  const side = (d: number) => (d > eps ? 1 : d < -eps ? -1 : 0);

  // pass 1: classify triangles & count output
  // state: 0 = drop, 1 = keep whole, 2 = cut
  const st = new Uint8Array(nt);
  let outTris = 0;
  for (let t = 0; t < nt; t++) {
    const a = I[3 * t], b = I[3 * t + 1], c = I[3 * t + 2];
    if (a >= nv || b >= nv || c >= nv) continue;
    const da = dist(a), db = dist(b), dc = dist(c);
    const sa = side(da), sb = side(db), sc = side(dc);
    const mx = Math.max(sa, sb, sc), mn = Math.min(sa, sb, sc);
    if (mn >= 0) {
      // all on kept side or on the plane (coplanar triangles included)
      st[t] = 1;
      outTris++;
    } else if (mx <= 0) {
      st[t] = 0;
    } else if (!cut) {
      if (da + db + dc > 0) {
        st[t] = 1;
        outTris++;
      }
    } else {
      st[t] = 2;
      // polygon size = (#vertices with side >= 0) + (#strictly crossing edges)
      let n = 0;
      if (sa >= 0) n++;
      if (sb >= 0) n++;
      if (sc >= 0) n++;
      if (sa * sb < 0) n++;
      if (sb * sc < 0) n++;
      if (sc * sa < 0) n++;
      outTris += n - 2;
    }
  }

  // pass 2: emit
  const indices = new Uint32Array(outTris * 3);
  const extraPos = new GrowableF32(1024);
  const extraNrm = mesh.normals ? new GrowableF32(1024) : null;
  const extraCol = mesh.colors ? new GrowableF32(1024) : null;
  const edgeMap = new PairHashMap(1024);
  const N = mesh.normals, C = mesh.colors;
  let nNew = 0;
  const cutVertex = (u: number, v: number, du: number, dv: number): number => {
    // canonical orientation lo -> hi so both neighbours compute the identical point
    let lo = u, hi = v, dlo = du, dhi = dv;
    if (u > v) {
      lo = v;
      hi = u;
      dlo = dv;
      dhi = du;
    }
    const existing = edgeMap.get(lo, hi);
    if (existing >= 0) return existing;
    const s = dlo / (dlo - dhi);
    for (let k = 0; k < 3; k++) extraPos.push(P[3 * lo + k] + s * (P[3 * hi + k] - P[3 * lo + k]));
    if (extraNrm && N) {
      let x = N[3 * lo] + s * (N[3 * hi] - N[3 * lo]);
      let y = N[3 * lo + 1] + s * (N[3 * hi + 1] - N[3 * lo + 1]);
      let z = N[3 * lo + 2] + s * (N[3 * hi + 2] - N[3 * lo + 2]);
      const l = Math.hypot(x, y, z) || 1;
      extraNrm.push3(x / l, y / l, z / l);
    }
    if (extraCol && C) for (let k = 0; k < 3; k++) extraCol.push(C[3 * lo + k] + s * (C[3 * hi + k] - C[3 * lo + k]));
    const id = nv + nNew++;
    edgeMap.set(lo, hi, id);
    return id;
  };
  const poly = new Uint32Array(4);
  const vs = new Uint32Array(3);
  const ds = new Float64Array(3);
  const ss = new Int8Array(3);
  let o = 0;
  for (let t = 0; t < nt; t++) {
    const s = st[t];
    if (s === 0) continue;
    if (s === 1) {
      indices[o++] = I[3 * t];
      indices[o++] = I[3 * t + 1];
      indices[o++] = I[3 * t + 2];
      continue;
    }
    for (let k = 0; k < 3; k++) {
      vs[k] = I[3 * t + k];
      ds[k] = dist(vs[k]);
      ss[k] = side(ds[k]);
    }
    let n = 0;
    for (let k = 0; k < 3; k++) {
      const k1 = k === 2 ? 0 : k + 1;
      if (ss[k] >= 0) poly[n++] = vs[k];
      if (ss[k] * ss[k1] < 0) poly[n++] = cutVertex(vs[k], vs[k1], ds[k], ds[k1]);
    }
    for (let k = 1; k + 1 < n; k++) {
      indices[o++] = poly[0];
      indices[o++] = poly[k];
      indices[o++] = poly[k + 1];
    }
  }

  // assemble positions (+ attributes) and drop unreferenced vertices
  const positions = new Float32Array((nv + nNew) * 3);
  positions.set(P.subarray(0, nv * 3));
  positions.set(extraPos.view(), nv * 3);
  const out: MeshData = { positions, indices: o === indices.length ? indices : indices.slice(0, o) };
  if (N && extraNrm) {
    const normals = new Float32Array((nv + nNew) * 3);
    normals.set(N.subarray(0, nv * 3));
    normals.set(extraNrm.view(), nv * 3);
    out.normals = normals;
  }
  if (C && extraCol) {
    const colors = new Uint8Array((nv + nNew) * 3);
    colors.set(C.subarray(0, nv * 3));
    const ec = extraCol.view();
    for (let i = 0; i < ec.length; i++) colors[nv * 3 + i] = Math.max(0, Math.min(255, Math.round(ec[i])));
    out.colors = colors;
  }
  return compactMesh(out);
}
