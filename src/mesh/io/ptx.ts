/**
 * Leica PTX (gridded terrestrial scan) reader with grid triangulation.
 *
 * File layout (one or more scans concatenated):
 * ```
 * cols
 * rows
 * sx sy sz            scanner registered position
 * ax ay az            scanner registered axis X
 * bx by bz            scanner registered axis Y
 * cx cy cz            scanner registered axis Z
 * m00 m01 m02 m03     4x4 transformation matrix, one ROW per line
 * m10 m11 m12 m13
 * m20 m21 m22 m23
 * m30 m31 m32 m33     (translation in the 4th line)
 * x y z intensity [r g b]   × cols·rows, column by column: point (c, r) is line c·rows + r
 * ```
 * Conventions assumed:
 *  - Points are row vectors: world_j = Σ_i local_i · M[i][j] + M[3][j] (i, j in 0..2).
 *  - A point with x = y = z = 0 is missing (no return). Only valid points become vertices.
 *  - The scanner sits at the local origin. Exception: when the matrix is the identity and the
 *    header position is not the origin, the points are taken to be stored in registered
 *    coordinates already and the scanner is at the header position.
 *
 * Triangulation: each grid cell gives two triangles (diagonal = the shorter one, or the other
 * diagonal when that keeps more triangles across a discontinuity) or one triangle when exactly
 * three corners are valid. A triangle is kept when every edge satisfies
 * `edgeLength / range < k · median`, where range is the distance of the nearer endpoint from the
 * scanner (local coordinates), the median is taken over a sample of up to ~200k horizontal and
 * vertical grid edges of that scan, and k = `maxEdgeFactor` (default 4). If all ranges are ~0
 * the plain edge length is used instead of the ratio. Triangle winding is chosen per scan so
 * that the normals face the scanner (majority vote over a sample of the scan's triangles).
 *
 * Colours: RGB when present (scans without RGB in a mixed file get intensity grey or mid-grey);
 * otherwise, with `intensityToGray`, intensity mapped to grey using the observed min..max range.
 */
import type { MeshData, ProgressFn } from "../../core/types";
import { GrowableF32, GrowableU32, GrowableU8 } from "../util/growable";
import { ByteScanner, ProgressTicker, toBytes } from "./text";

export interface PTXOptions {
  /** k in the edge test `edge/range < k · median`; default 4 */
  maxEdgeFactor?: number;
  onProgress?: ProgressFn;
  /** map intensity to grey vertex colours when a scan has no RGB (default false) */
  intensityToGray?: boolean;
}

const PROGRESS_BYTES = 4 << 20;
const MEDIAN_SAMPLE_CELLS = 100000;
const VOTE_SAMPLE_TRIS = 20000;
const MAX_GRID_POINTS = 0x7fffffff;

interface ScanInfo {
  vStart: number;
  vEnd: number;
  hasRGB: boolean;
}

function isGridCount(v: number): boolean {
  return Number.isFinite(v) && v >= 0 && v === Math.floor(v);
}

/** Parse a PTX file (one or more scans) into one triangle mesh. */
export function parsePTX(data: string | ArrayBuffer | Uint8Array, opts: PTXOptions = {}): MeshData {
  const bytes = toBytes(data);
  const len = bytes.length;
  // skip a UTF-8 byte order mark
  const s = new ByteScanner(bytes, len >= 3 && bytes[0] === 0xef && bytes[1] === 0xbb && bytes[2] === 0xbf ? 3 : 0);
  const k = opts.maxEdgeFactor ?? 4;
  const onProgress = opts.onProgress;
  const ticker = new ProgressTicker(onProgress, len, PROGRESS_BYTES, "Reading PTX");

  // ~40 bytes per point line
  const est = Math.min(1 << 22, Math.max(64, Math.floor(len / 40)));
  const pos = new GrowableF32(est * 3);
  const idx = new GrowableU32(est * 6);
  let rgb: GrowableU8 | null = null;
  const inten = opts.intensityToGray ? new GrowableF32(est) : null;
  let iMin = Infinity, iMax = -Infinity;
  const scans: ScanInfo[] = [];
  const hdr = new Float64Array(28);

  for (;;) {
    s.skipWhitespace();
    if (s.pos >= len) break;
    const cols = s.readFloatAny();
    const rows = s.readFloatAny();
    if (!s.ok || !isGridCount(cols) || !isGridCount(rows)) {
      if (scans.length === 0) throw new Error("PTX: invalid scan header (expected column and row counts)");
      break; // trailing junk
    }
    let headerOk = true;
    for (let i = 0; i < 28; i++) {
      hdr[i] = s.readFloatAny();
      if (!s.ok) {
        headerOk = false;
        break;
      }
    }
    if (!headerOk) {
      if (scans.length === 0) throw new Error("PTX: truncated scan header");
      break;
    }
    const nPts = cols * rows;
    if (nPts > MAX_GRID_POINTS) throw new Error(`PTX: scan grid too large (${cols} x ${rows})`);
    // matrix rows: hdr[12 + 4i + j] = M[i][j]
    const m = hdr.subarray(12, 28);
    const m00 = m[0], m01 = m[1], m02 = m[2];
    const m10 = m[4], m11 = m[5], m12 = m[6];
    const m20 = m[8], m21 = m[9], m22 = m[10];
    const m30 = m[12], m31 = m[13], m32 = m[14];
    // scanner position in local coordinates
    let slx = 0, sly = 0, slz = 0;
    let identity = true;
    for (let i = 0; i < 16; i++) {
      const want = i % 5 === 0 ? 1 : 0;
      if (Math.abs(m[i] - want) > 1e-9) {
        identity = false;
        break;
      }
    }
    if (identity && (hdr[0] !== 0 || hdr[1] !== 0 || hdr[2] !== 0)) {
      slx = hdr[0];
      sly = hdr[1];
      slz = hdr[2];
    }
    const scanWX = slx * m00 + sly * m10 + slz * m20 + m30;
    const scanWY = slx * m01 + sly * m11 + slz * m21 + m31;
    const scanWZ = slx * m02 + sly * m12 + slz * m22 + m32;

    // a point line takes at least 6 bytes: never allocate more grid than the data can hold
    const maxPts = Math.min(nPts, Math.floor((len - s.pos) / 6) + 1);
    const vidx = new Int32Array(maxPts).fill(-1);
    const rng = new Float32Array(maxPts);
    const vStart = pos.length / 3;
    let hasRGB = false;
    let maxRange = 0;
    let read = 0;
    for (; read < maxPts; read++) {
      s.skipWhitespace();
      if (s.pos >= len) break;
      let x = s.readFloat();
      let y = s.readFloat();
      if (!s.ok) y = 0;
      let z = s.readFloat();
      if (!s.ok) z = 0;
      const it = s.readFloat();
      const hasI = s.ok;
      let r = 0, g = 0, b = 0;
      let hasC = false;
      if (hasI) {
        r = s.readFloat();
        if (s.ok) {
          g = s.readFloat();
          b = s.readFloat();
          hasC = s.ok;
        }
      }
      s.nextLine();
      ticker.tick(s.pos);
      if ((x === 0 && y === 0 && z === 0) || x !== x || y !== y || z !== z) continue;
      const vi = pos.length / 3;
      pos.push3(
        x * m00 + y * m10 + z * m20 + m30,
        x * m01 + y * m11 + z * m21 + m31,
        x * m02 + y * m12 + z * m22 + m32,
      );
      vidx[read] = vi;
      x -= slx;
      y -= sly;
      z -= slz;
      const range = Math.sqrt(x * x + y * y + z * z);
      rng[read] = range;
      if (range > maxRange) maxRange = range;
      if (hasC) {
        if (!rgb) {
          rgb = new GrowableU8(Math.max(48, pos.data.length));
          rgb.reserve(vi * 3);
          rgb.data.fill(0, 0, vi * 3);
          rgb.length = vi * 3;
        }
        rgb.push3(r <= 0 ? 0 : r >= 255 ? 255 : r, g <= 0 ? 0 : g >= 255 ? 255 : g, b <= 0 ? 0 : b >= 255 ? 255 : b);
        hasRGB = true;
      } else if (rgb) {
        rgb.push3(0, 0, 0);
      }
      if (inten) {
        if (hasI && Number.isFinite(it)) {
          inten.push(it);
          if (it < iMin) iMin = it;
          if (it > iMax) iMax = it;
        } else inten.push(NaN);
      }
    }
    const vEnd = pos.length / 3;
    scans.push({ vStart, vEnd, hasRGB });

    if (cols >= 2 && rows >= 2 && vEnd - vStart >= 3) {
      const tStart = idx.length / 3;
      triangulateGrid(cols, rows, vidx, rng, pos.data, maxRange > 1e-6, k, idx);
      orientTowards(idx, tStart, idx.length / 3, pos.data, scanWX, scanWY, scanWZ);
    }
    onProgress?.(Math.min(1, s.pos / Math.max(1, len)), "Reading PTX");
    if (read < nPts) break; // truncated file
  }

  const nV = pos.length / 3;
  const mesh: MeshData = { positions: pos.finish(), indices: idx.finish() };
  const gray = (i: number): number => {
    if (!(i === i) || !(iMax > iMin)) return 128;
    const v = Math.round(((i - iMin) / (iMax - iMin)) * 255);
    return v <= 0 ? 0 : v >= 255 ? 255 : v;
  };
  if (rgb) {
    const colors = rgb.finish();
    for (const sc of scans) {
      if (sc.hasRGB) continue;
      for (let v = sc.vStart; v < sc.vEnd; v++) {
        const gv = inten ? gray(inten.data[v]) : 128;
        colors[3 * v] = gv;
        colors[3 * v + 1] = gv;
        colors[3 * v + 2] = gv;
      }
    }
    mesh.colors = colors;
  } else if (inten && iMax >= iMin) {
    const colors = new Uint8Array(nV * 3);
    const id = inten.data;
    for (let v = 0; v < nV; v++) {
      const gv = gray(id[v]);
      colors[3 * v] = gv;
      colors[3 * v + 1] = gv;
      colors[3 * v + 2] = gv;
    }
    mesh.colors = colors;
  }
  onProgress?.(1, "Reading PTX");
  return mesh;
}

/** Deterministic xorshift32 PRNG (no global state). */
function makeRng(seed: number): () => number {
  let x = seed >>> 0 || 0x9e3779b9;
  return () => {
    x ^= x << 13;
    x >>>= 0;
    x ^= x >>> 17;
    x ^= x << 5;
    x >>>= 0;
    return x;
  };
}

/**
 * Triangulate one scan grid. `vidx[g]` = vertex index of grid point g (-1 = missing),
 * `rng[g]` = its range from the scanner, `P` = vertex positions (world).
 */
export function triangulateGrid(
  cols: number,
  rows: number,
  vidx: Int32Array,
  rng: Float32Array,
  P: Float32Array,
  useRatio: boolean,
  k: number,
  out: GrowableU32,
): void {
  // only complete columns present in the (possibly truncated) grid are triangulated
  cols = Math.min(cols, Math.floor(vidx.length / rows));
  const nCells = cols * rows;
  if (cols < 2) return;

  const edgeLen = (va: number, vb: number): number => {
    const dx = P[3 * va] - P[3 * vb], dy = P[3 * va + 1] - P[3 * vb + 1], dz = P[3 * va + 2] - P[3 * vb + 2];
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  };

  // ---- median edge ratio from a sample of horizontal / vertical grid edges ----
  const sample = new Float32Array(2 * MEDIAN_SAMPLE_CELLS);
  let cnt = 0;
  const sampleEdge = (ga: number, gb: number) => {
    const va = vidx[ga], vb = vidx[gb];
    if (va < 0 || vb < 0) return;
    const l = edgeLen(va, vb);
    if (!(l > 0)) return;
    if (useRatio) {
      const rmin = Math.min(rng[ga], rng[gb]);
      if (!(rmin > 0)) return;
      sample[cnt++] = l / rmin;
    } else sample[cnt++] = l;
  };
  const sampleCell = (g: number) => {
    const c = Math.floor(g / rows), r = g - c * rows;
    if (c + 1 < cols) sampleEdge(g, g + rows);
    if (r + 1 < rows) sampleEdge(g, g + 1);
  };
  if (nCells <= MEDIAN_SAMPLE_CELLS) {
    for (let g = 0; g < nCells; g++) sampleCell(g);
  } else {
    const rand = makeRng(nCells ^ 0x5bd1e995);
    for (let i = 0; i < MEDIAN_SAMPLE_CELLS; i++) sampleCell(rand() % nCells);
  }
  if (cnt === 0) return;
  const sorted = sample.subarray(0, cnt).sort();
  const thr = k * sorted[cnt >> 1];
  if (!(thr > 0)) return;

  /** Edge acceptance test between grid points ga, gb (both valid). */
  const ok = (ga: number, gb: number): boolean => {
    const l = edgeLen(vidx[ga], vidx[gb]);
    if (useRatio) return l < thr * Math.min(rng[ga], rng[gb]);
    return l < thr;
  };

  for (let c = 0; c + 1 < cols; c++) {
    const base = c * rows;
    for (let r = 0; r + 1 < rows; r++) {
      const g00 = base + r, g01 = g00 + 1, g10 = g00 + rows, g11 = g10 + 1;
      const a = vidx[g00], b = vidx[g10], cc = vidx[g01], d = vidx[g11];
      const nValid = (a >= 0 ? 1 : 0) + (b >= 0 ? 1 : 0) + (cc >= 0 ? 1 : 0) + (d >= 0 ? 1 : 0);
      if (nValid < 3) continue;
      // Canonical winding: counter-clockwise in (column, row) parameter space.
      if (nValid === 4) {
        const h0 = ok(g00, g10), h1 = ok(g01, g11), v0 = ok(g00, g01), v1 = ok(g10, g11);
        const dA = edgeLen(a, d), dB = edgeLen(b, cc);
        // option A: diagonal 00-11 → (00,10,11) + (00,11,01); option B: diagonal 10-01 → (00,10,01) + (10,11,01)
        const okA = ok(g00, g11), okB = ok(g10, g01);
        const a1 = okA && h0 && v1, a2 = okA && h1 && v0;
        const b1 = okB && h0 && v0, b2 = okB && v1 && h1;
        const nA = (a1 ? 1 : 0) + (a2 ? 1 : 0), nB = (b1 ? 1 : 0) + (b2 ? 1 : 0);
        const useA = dA <= dB ? nA >= nB : nA > nB;
        if (useA) {
          if (a1) out.push3(a, b, d);
          if (a2) out.push3(a, d, cc);
        } else {
          if (b1) out.push3(a, b, cc);
          if (b2) out.push3(b, d, cc);
        }
      } else if (a < 0) {
        if (ok(g10, g11) && ok(g11, g01) && ok(g01, g10)) out.push3(b, d, cc);
      } else if (b < 0) {
        if (ok(g00, g11) && ok(g11, g01) && ok(g01, g00)) out.push3(a, d, cc);
      } else if (cc < 0) {
        if (ok(g00, g10) && ok(g10, g11) && ok(g11, g00)) out.push3(a, b, d);
      } else {
        if (ok(g00, g10) && ok(g10, g01) && ok(g01, g00)) out.push3(a, b, cc);
      }
    }
  }
}

/**
 * Flip the winding of triangles [t0, t1) when the majority of a sample faces away from the
 * point (sx, sy, sz).
 */
function orientTowards(
  idx: GrowableU32,
  t0: number,
  t1: number,
  P: Float32Array,
  sx: number,
  sy: number,
  sz: number,
): void {
  const nT = t1 - t0;
  if (nT <= 0) return;
  const I = idx.data;
  const step = Math.max(1, Math.floor(nT / VOTE_SAMPLE_TRIS));
  let votes = 0;
  for (let t = t0; t < t1; t += step) {
    const a = 3 * I[3 * t], b = 3 * I[3 * t + 1], c = 3 * I[3 * t + 2];
    const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2];
    const vx = P[c] - P[a], vy = P[c + 1] - P[a + 1], vz = P[c + 2] - P[a + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const cx = (P[a] + P[b] + P[c]) / 3, cy = (P[a + 1] + P[b + 1] + P[c + 1]) / 3, cz = (P[a + 2] + P[b + 2] + P[c + 2]) / 3;
    const dt = nx * (sx - cx) + ny * (sy - cy) + nz * (sz - cz);
    if (dt > 0) votes++;
    else if (dt < 0) votes--;
  }
  if (votes >= 0) return;
  for (let t = t0; t < t1; t++) {
    const tmp = I[3 * t + 1];
    I[3 * t + 1] = I[3 * t + 2];
    I[3 * t + 2] = tmp;
  }
}
