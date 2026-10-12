/**
 * Wavefront OBJ reader (geometry only), parsed directly from bytes.
 *
 * Supported records:
 *  - `v x y z [w]` and `v x y z r g b` (vertex colours; floats 0..1, or 0..255 when any
 *    component exceeds 1.01) → per-vertex colours (vertices without a colour get white).
 *  - `vn x y z`
 *  - `f` with `v`, `v/vt`, `v//vn`, `v/vt/vn`, negative (relative) indices, polygons
 *    (fan-triangulated). Faces referencing an out-of-range vertex (including forward
 *    references to vertices declared later) are dropped. Degenerate faces are kept.
 *  - everything else (vt, g, o, s, usemtl, mtllib, l, p, comments, `\` continuations) is ignored.
 *
 * Vertex identity is the `v` index: vertices are never split per normal / uv (scans need
 * welded topology).
 *
 * Normals: when faces reference `vn`, the output normal of a position vertex is the normalised
 * sum of all `vn` referenced by its face corners. `normals` is only output when EVERY vertex
 * used by a face received at least one normal (vertices not used by any face get (0,0,0));
 * otherwise normals are omitted. When no face references `vn` but the file has exactly as many
 * `vn` as `v` records (typical for point clouds), normals are taken by index.
 *
 * A file without faces yields `indices.length === 0` (use {@link objToImportResult} or the
 * importer to get a point cloud).
 */
import type { MeshData, PointCloudData, ProgressFn } from "../../core/types";
import { GrowableF32, GrowableI32, GrowableU32 } from "../util/growable";
import { ByteScanner, ProgressTicker, isSpaceByte, toBytes } from "./text";

const PROGRESS_BYTES = 4 << 20;
/** Float colours above this are taken to be 0..255. */
const COLOR_255_THRESHOLD = 1.01;

function isHBlank(c: number): boolean {
  return c === 32 || c === 9 || c === 13;
}

/** Convert raw colour floats (NaN = missing) to 0..255 bytes. */
export function colorsToBytes(raw: Float32Array, maxComponent: number, missing = 255): Uint8Array {
  const out = new Uint8Array(raw.length);
  const scale = maxComponent > COLOR_255_THRESHOLD ? 1 : 255;
  for (let i = 0; i < raw.length; i++) {
    const v = raw[i];
    if (v !== v) {
      out[i] = missing;
      continue;
    }
    const s = Math.round(v * scale);
    out[i] = s <= 0 ? 0 : s >= 255 ? 255 : s;
  }
  return out;
}

/** Parse an OBJ file into an indexed triangle mesh (see module docs). */
export function parseOBJ(data: string | ArrayBuffer | Uint8Array, onProgress?: ProgressFn): MeshData {
  const bytes = toBytes(data);
  const len = bytes.length;
  // skip a UTF-8 byte order mark
  const s = new ByteScanner(bytes, len >= 3 && bytes[0] === 0xef && bytes[1] === 0xbb && bytes[2] === 0xbf ? 3 : 0);
  const ticker = new ProgressTicker(onProgress, len, PROGRESS_BYTES, "Reading OBJ");

  // rough capacity guesses (grow as needed)
  const estV = Math.min(1 << 22, Math.max(64, Math.floor(len / 64)));
  const pos = new GrowableF32(estV * 3);
  const vn = new GrowableF32(64);
  const idx = new GrowableU32(estV * 6);
  let colRaw: GrowableF32 | null = null;
  let colMax = 0;

  // per-vertex normal accumulation (allocated on first face corner with a valid vn)
  let nAcc: Float32Array | null = null;
  let nFlag: Uint8Array | null = null;
  let anyFaceNormal = false;

  const faceV = new GrowableI32(16);
  const faceN = new GrowableI32(16);

  while (s.pos < len) {
    s.skipBlanks();
    const p = s.pos;
    if (p >= len) break;
    const c = bytes[p];
    const c1 = p + 1 < len ? bytes[p + 1] : 10;
    if (c === 118 /* v */) {
      if (isHBlank(c1)) {
        // vertex
        s.pos = p + 1;
        let x = s.readFloat();
        if (!s.ok) x = 0;
        let y = s.readFloat();
        if (!s.ok) y = 0;
        let z = s.readFloat();
        if (!s.ok) z = 0;
        const vi = pos.length / 3;
        pos.push3(x, y, z);
        // optional extras: w, or r g b
        let nExtra = 0;
        let e0 = 0, e1 = 0, e2 = 0;
        e0 = s.readFloat();
        if (s.ok) {
          nExtra = 1;
          e1 = s.readFloat();
          if (s.ok) {
            nExtra = 2;
            e2 = s.readFloat();
            if (s.ok) nExtra = 3;
          }
        }
        if (nExtra >= 3) {
          if (!colRaw) {
            colRaw = new GrowableF32(Math.max(pos.data.length, 48));
            colRaw.reserve(vi * 3);
            colRaw.data.fill(NaN, 0, vi * 3);
            colRaw.length = vi * 3;
          }
          colRaw.push3(e0, e1, e2);
          if (e0 > colMax) colMax = e0;
          if (e1 > colMax) colMax = e1;
          if (e2 > colMax) colMax = e2;
        } else if (colRaw) {
          colRaw.push3(NaN, NaN, NaN);
        }
      } else if (c1 === 110 /* n */ && (p + 2 >= len || isSpaceByte(bytes[p + 2]))) {
        s.pos = p + 2;
        let x = s.readFloat();
        if (!s.ok) x = 0;
        let y = s.readFloat();
        if (!s.ok) y = 0;
        let z = s.readFloat();
        if (!s.ok) z = 0;
        vn.push3(x, y, z);
      }
    } else if (c === 102 /* f */ && isHBlank(c1)) {
      s.pos = p + 1;
      const nV = pos.length / 3;
      const nN = vn.length / 3;
      faceV.clear();
      faceN.clear();
      let bad = false;
      for (;;) {
        s.skipBlanks();
        if (s.pos >= len) break;
        const t = bytes[s.pos];
        if (t === 10 || t === 35 /* # */ || t === 92 /* \ */) break;
        let vi = s.readIntRaw();
        if (!s.ok) {
          s.skipToken();
          bad = true;
          continue;
        }
        vi = vi > 0 ? vi - 1 : vi < 0 ? nV + vi : -1;
        if (vi < 0 || vi >= nV) bad = true;
        let ni = -1;
        if (s.pos < len && bytes[s.pos] === 47 /* / */) {
          s.pos++;
          if (s.pos < len && bytes[s.pos] !== 47) s.readIntRaw(); // vt (ignored)
          if (s.pos < len && bytes[s.pos] === 47) {
            s.pos++;
            const n = s.readIntRaw();
            if (s.ok) {
              const r = n > 0 ? n - 1 : n < 0 ? nN + n : -1;
              if (r >= 0 && r < nN) ni = r;
            }
          }
        }
        s.skipToken(); // anything glued to the token
        faceV.push(vi);
        faceN.push(ni);
      }
      const k = faceV.length;
      if (!bad && k >= 3) {
        const fv = faceV.data;
        const v0 = fv[0];
        idx.reserve((k - 2) * 3);
        for (let i = 1; i + 1 < k; i++) idx.push3(v0, fv[i], fv[i + 1]);
        const fn = faceN.data;
        let hasN = false;
        for (let i = 0; i < k; i++) {
          if (fn[i] >= 0) {
            hasN = true;
            break;
          }
        }
        if (hasN) {
          anyFaceNormal = true;
          if (!nAcc || !nFlag || nFlag.length < nV) {
            const cap = Math.max(nV, nFlag ? nFlag.length * 2 : 0, 1024);
            const a = new Float32Array(cap * 3);
            const f = new Uint8Array(cap);
            if (nAcc && nFlag) {
              a.set(nAcc);
              f.set(nFlag);
            }
            nAcc = a;
            nFlag = f;
          }
          const acc = nAcc, flag = nFlag, vnd = vn.data;
          for (let i = 0; i < k; i++) {
            const n = fn[i];
            if (n < 0) continue;
            const v = fv[i];
            acc[3 * v] += vnd[3 * n];
            acc[3 * v + 1] += vnd[3 * n + 1];
            acc[3 * v + 2] += vnd[3 * n + 2];
            flag[v] = 1;
          }
        }
      }
    }
    s.nextLine();
    ticker.tick(s.pos);
  }

  const nV = pos.length / 3;
  const mesh: MeshData = { positions: pos.finish(), indices: idx.finish() };

  if (anyFaceNormal && nAcc && nFlag) {
    const flag: Uint8Array = nFlag;
    const ind = mesh.indices;
    let complete = true;
    for (let i = 0; i < ind.length; i++) {
      if (!flag[ind[i]]) {
        complete = false;
        break;
      }
    }
    if (complete) {
      const acc: Float32Array = nAcc;
      const normals = new Float32Array(nV * 3);
      for (let v = 0; v < nV; v++) {
        if (!flag[v]) continue;
        const x = acc[3 * v], y = acc[3 * v + 1], z = acc[3 * v + 2];
        const l = Math.sqrt(x * x + y * y + z * z);
        if (l > 0) {
          normals[3 * v] = x / l;
          normals[3 * v + 1] = y / l;
          normals[3 * v + 2] = z / l;
        }
      }
      mesh.normals = normals;
    }
  } else if (!anyFaceNormal && nV > 0 && vn.length === nV * 3) {
    const src = vn.data;
    const normals = new Float32Array(nV * 3);
    for (let v = 0; v < nV; v++) {
      const x = src[3 * v], y = src[3 * v + 1], z = src[3 * v + 2];
      const l = Math.sqrt(x * x + y * y + z * z);
      if (l > 0) {
        normals[3 * v] = x / l;
        normals[3 * v + 1] = y / l;
        normals[3 * v + 2] = z / l;
      }
    }
    mesh.normals = normals;
  }

  if (colRaw) mesh.colors = colorsToBytes(colRaw.view(), colMax);
  onProgress?.(1, "Reading OBJ");
  return mesh;
}

/** Mesh when the OBJ has faces, otherwise its vertices as a point cloud. */
export function objToImportResult(
  mesh: MeshData,
): { kind: "mesh"; mesh: MeshData } | { kind: "points"; points: PointCloudData } {
  if (mesh.indices.length > 0) return { kind: "mesh", mesh };
  const points: PointCloudData = { positions: mesh.positions };
  if (mesh.normals) points.normals = mesh.normals;
  if (mesh.colors) points.colors = mesh.colors;
  return { kind: "points", points };
}
