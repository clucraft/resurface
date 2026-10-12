/**
 * STL reader (ASCII + binary, in-memory and streaming) and binary STL writer.
 *
 * Vertices are welded on exact (float32) position equality with {@link ExactVertexWelder}
 * (typed-array open addressing — no Map, no per-vertex objects). File winding is kept, the
 * per-facet normals stored in the file are ignored.
 *
 * Format detection ({@link detectSTLFormat}):
 *  1. if the total size is known and equals 84 + 50·n (n = uint32 LE at offset 80) → binary;
 *  2. else, if the data starts (after whitespace / UTF-8 BOM) with "solid" (or "facet"), the
 *     first KB contains no NUL / control bytes, and "facet" or "endsolid" occurs within the
 *     first 4 KB → ASCII;
 *  3. else binary.
 *
 * Binary triangle count: the header count is honoured when it is > 0 and fits in the data;
 * when it is 0 or larger than the data, all complete 50-byte records are read (truncated
 * files parse what is complete; trailing bytes after the declared records are ignored).
 */
import type { MeshData, ProgressFn } from "../../core/types";
import { GrowableU32 } from "../util/growable";
import { ExactVertexWelder } from "../util/weld-hash";
import { ByteScanner, ProgressTicker, dataViewOf, isSpaceByte, toBytes } from "./text";

const HEADER_BYTES = 84;
const RECORD_BYTES = 50;
/** How many leading bytes the ASCII/binary heuristic looks at. */
export const STL_DETECT_BYTES = 4096;
const NUL_CHECK_BYTES = 1024;
/** Report progress every this many binary triangles. */
const BIN_PROGRESS_TRIS = 1 << 20;
/** Report progress every this many bytes of text. */
const TEXT_PROGRESS_BYTES = 4 << 20;

function emptyMesh(): MeshData {
  return { positions: new Float32Array(0), indices: new Uint32Array(0) };
}

function startsWithCI(b: Uint8Array, p: number, word: string): boolean {
  if (p + word.length > b.length) return false;
  for (let i = 0; i < word.length; i++) {
    let c = b[p + i];
    if (c >= 65 && c <= 90) c += 32;
    if (c !== word.charCodeAt(i)) return false;
  }
  return true;
}

function containsCI(b: Uint8Array, end: number, word: string): boolean {
  const w0 = word.charCodeAt(0);
  const last = end - word.length;
  for (let p = 0; p <= last; p++) {
    let c = b[p];
    if (c >= 65 && c <= 90) c += 32;
    if (c === w0 && startsWithCI(b, p, word)) return true;
  }
  return false;
}

/**
 * Decide whether STL data is ASCII or binary.
 * @param head the first bytes of the file (at least min(fileSize, {@link STL_DETECT_BYTES}) for a
 *   reliable answer)
 * @param totalLength total file size when known (enables the exact binary size test)
 */
export function detectSTLFormat(head: Uint8Array, totalLength?: number): "ascii" | "binary" {
  const n = head.length;
  if (totalLength !== undefined && n >= HEADER_BYTES) {
    const count = dataViewOf(head).getUint32(80, true);
    if (HEADER_BYTES + RECORD_BYTES * count === totalLength) return "binary";
  }
  let p = 0;
  if (n >= 3 && head[0] === 0xef && head[1] === 0xbb && head[2] === 0xbf) p = 3;
  while (p < n && isSpaceByte(head[p])) p++;
  if (!startsWithCI(head, p, "solid") && !startsWithCI(head, p, "facet")) return "binary";
  const nulEnd = Math.min(n, NUL_CHECK_BYTES);
  for (let i = 0; i < nulEnd; i++) {
    const c = head[i];
    // NUL and control characters other than \t \n \v \f \r mean binary
    if (c < 9 || (c > 13 && c < 32) || c === 127) return "binary";
  }
  const searchEnd = Math.min(n, STL_DETECT_BYTES);
  if (containsCI(head, searchEnd, "facet") || containsCI(head, searchEnd, "endsolid")) return "ascii";
  return "binary";
}

/**
 * Append `count` binary records starting at byte `off` of `dv` (vertex positions at record
 * offsets 12/24/36; the stored normal and the attribute word are ignored).
 */
function readBinaryRecords(
  dv: DataView,
  off: number,
  count: number,
  welder: ExactVertexWelder,
  idx: GrowableU32,
): void {
  idx.reserve(count * 3);
  const d = idx.data;
  let l = idx.length;
  for (let t = 0; t < count; t++, off += RECORD_BYTES) {
    const a = welder.add(dv.getFloat32(off + 12, true), dv.getFloat32(off + 16, true), dv.getFloat32(off + 20, true));
    const b = welder.add(dv.getFloat32(off + 24, true), dv.getFloat32(off + 28, true), dv.getFloat32(off + 32, true));
    const c = welder.add(dv.getFloat32(off + 36, true), dv.getFloat32(off + 40, true), dv.getFloat32(off + 44, true));
    d[l] = a;
    d[l + 1] = b;
    d[l + 2] = c;
    l += 3;
  }
  idx.length = l;
}

/** Number of binary records to read given the header count and the available bytes. */
function binaryTriangleCount(headerCount: number, available: number): number {
  return headerCount > 0 && headerCount <= available ? headerCount : available;
}

function parseBinarySTL(bytes: Uint8Array, onProgress?: ProgressFn): MeshData {
  const len = bytes.length;
  if (len < HEADER_BYTES) return emptyMesh();
  const dv = dataViewOf(bytes);
  const available = Math.floor((len - HEADER_BYTES) / RECORD_BYTES);
  const n = binaryTriangleCount(dv.getUint32(80, true), available);
  if (n === 0) return emptyMesh();
  const welder = new ExactVertexWelder(Math.ceil(n / 2) + 16);
  const idx = new GrowableU32(n * 3);
  let off = HEADER_BYTES;
  for (let t = 0; t < n; t += BIN_PROGRESS_TRIS) {
    const m = Math.min(BIN_PROGRESS_TRIS, n - t);
    readBinaryRecords(dv, off, m, welder, idx);
    off += m * RECORD_BYTES;
    onProgress?.((t + m) / n, "Reading binary STL");
  }
  return { positions: welder.positions(), indices: idx.finish() };
}

/** ASCII STL: token-driven (robust to odd line breaks, CRLF, upper case keywords, polygons). */
function parseAsciiSTL(bytes: Uint8Array, onProgress?: ProgressFn): MeshData {
  const len = bytes.length;
  const s = new ByteScanner(bytes);
  // ~250 bytes per facet in typical ASCII STL
  const estTris = Math.max(16, Math.floor(len / 250));
  const welder = new ExactVertexWelder(Math.ceil(estTris / 2) + 16);
  const idx = new GrowableU32(estTris * 3);
  const facet = new GrowableU32(8);
  const ticker = new ProgressTicker(onProgress, len, TEXT_PROGRESS_BYTES, "Reading ASCII STL");

  const flush = () => {
    const n = facet.length;
    if (n >= 3) {
      const f = facet.data;
      for (let i = 1; i + 1 < n; i++) idx.push3(f[0], f[i], f[i + 1]);
    }
    facet.clear();
  };

  for (;;) {
    s.skipWhitespace();
    if (s.pos >= len) break;
    let c = bytes[s.pos];
    if (c >= 65 && c <= 90) c += 32;
    if (c === 118 /* v */ && s.matchWord("vertex")) {
      const x = s.readFloat();
      const okx = s.ok;
      const y = s.readFloat();
      const oky = s.ok;
      const z = s.readFloat();
      if (okx && oky && s.ok) facet.push(welder.add(x, y, z));
    } else if (c === 102 /* f */ && s.matchWord("facet")) {
      flush();
    } else if (c === 101 /* e */ && s.matchWord("endfacet")) {
      flush();
    } else if (c === 101 /* e */ && s.matchWord("endsolid")) {
      flush();
      s.nextLine(); // the solid name may contain keywords
    } else if (c === 115 /* s */ && s.matchWord("solid")) {
      flush();
      s.nextLine();
    } else {
      s.skipToken();
    }
    ticker.tick(s.pos);
  }
  flush();
  onProgress?.(1, "Reading ASCII STL");
  return { positions: welder.positions(), indices: idx.finish() };
}

/**
 * Parse an STL file (ASCII or binary, auto-detected). Exactly coincident vertices are welded.
 * Empty / zero-triangle input gives an empty mesh. Never outputs normals.
 */
export function parseSTL(data: ArrayBuffer | Uint8Array, onProgress?: ProgressFn): MeshData {
  const bytes = toBytes(data);
  if (bytes.length === 0) return emptyMesh();
  const kind = detectSTLFormat(bytes.subarray(0, STL_DETECT_BYTES), bytes.length);
  return kind === "ascii" ? parseAsciiSTL(bytes, onProgress) : parseBinarySTL(bytes, onProgress);
}

export interface STLStreamParser {
  /** Feed the next chunk of the file. Chunks are not retained by reference (copied if needed). */
  push(chunk: Uint8Array): void;
  /** Finish parsing and return the mesh. The parser cannot be used afterwards. */
  finish(): MeshData;
}

export interface STLStreamOptions {
  onProgress?: ProgressFn;
  /** Total file size, when known: enables the exact binary size test, capacity and progress. */
  totalBytes?: number;
}

/**
 * Incremental STL parser for very large files: binary data is parsed record-by-record as
 * chunks arrive (chunks may split the header and records anywhere; only a ≤50-byte carry-over
 * is kept). The header triangle count is only used as a capacity hint and as an upper bound
 * (same rule as {@link parseSTL}). If the start of the data looks like ASCII STL, chunks are
 * accumulated and parsed as ASCII by `finish()`.
 */
export function createSTLStreamParser(opts: STLStreamOptions = {}): STLStreamParser {
  const { onProgress, totalBytes } = opts;
  const MODE_UNDECIDED = 0, MODE_BINARY = 1, MODE_ASCII = 2;
  let mode = MODE_UNDECIDED;
  let finished = false;

  // undecided / ascii: retained (copied) chunks
  let pending: Uint8Array[] = [];
  let pendingLen = 0;

  // binary state
  const header = new Uint8Array(HEADER_BYTES);
  let headerLen = 0;
  const carry = new Uint8Array(RECORD_BYTES);
  const carryView = new DataView(carry.buffer);
  let carryLen = 0;
  let maxRecords = Infinity;
  let records = 0;
  let welder: ExactVertexWelder | null = null;
  let idx: GrowableU32 | null = null;

  let bytesSeen = 0;
  const progressStep = 32 << 20;
  let nextProgress = progressStep;

  const report = () => {
    if (!onProgress || bytesSeen < nextProgress) return;
    nextProgress = bytesSeen + progressStep;
    if (totalBytes && totalBytes > 0) onProgress(Math.min(1, bytesSeen / totalBytes), "Reading STL");
    else onProgress(0, `Reading STL (${Math.round(bytesSeen / (1 << 20))} MB)`);
  };

  const onHeader = () => {
    const count = new DataView(header.buffer).getUint32(80, true);
    maxRecords = count > 0 ? count : Infinity;
    let hint: number;
    if (totalBytes !== undefined && totalBytes >= HEADER_BYTES) {
      const available = Math.floor((totalBytes - HEADER_BYTES) / RECORD_BYTES);
      hint = binaryTriangleCount(count, available);
    } else {
      hint = count > 0 ? Math.min(count, 1 << 22) : 1 << 16;
    }
    hint = Math.max(16, hint);
    welder = new ExactVertexWelder(Math.ceil(hint / 2) + 16);
    idx = new GrowableU32(hint * 3);
  };

  const feedBinary = (chunk: Uint8Array) => {
    const n = chunk.length;
    let p = 0;
    if (headerLen < HEADER_BYTES) {
      const take = Math.min(HEADER_BYTES - headerLen, n);
      header.set(chunk.subarray(0, take), headerLen);
      headerLen += take;
      p = take;
      if (headerLen < HEADER_BYTES) return;
      onHeader();
    }
    if (records >= maxRecords) return;
    const w = welder!, ix = idx!;
    if (carryLen > 0) {
      const take = Math.min(RECORD_BYTES - carryLen, n - p);
      carry.set(chunk.subarray(p, p + take), carryLen);
      carryLen += take;
      p += take;
      if (carryLen < RECORD_BYTES) return;
      readBinaryRecords(carryView, 0, 1, w, ix);
      records++;
      carryLen = 0;
      if (records >= maxRecords) return;
    }
    const full = Math.min(Math.floor((n - p) / RECORD_BYTES), maxRecords - records);
    if (full > 0) {
      const dv = new DataView(chunk.buffer, chunk.byteOffset + p, full * RECORD_BYTES);
      readBinaryRecords(dv, 0, full, w, ix);
      p += full * RECORD_BYTES;
      records += full;
    }
    if (records < maxRecords && p < n) {
      carry.set(chunk.subarray(p), 0);
      carryLen = n - p;
    }
  };

  const concatPending = (): Uint8Array => {
    if (pending.length === 1) return pending[0];
    const all = new Uint8Array(pendingLen);
    let o = 0;
    for (const c of pending) {
      all.set(c, o);
      o += c.length;
    }
    return all;
  };

  /** Build the detection prefix from the pending chunks. */
  const pendingHead = (): Uint8Array => {
    const n = Math.min(pendingLen, STL_DETECT_BYTES);
    if (pending.length > 0 && pending[0].length >= n) return pending[0].subarray(0, n);
    const head = new Uint8Array(n);
    let o = 0;
    for (const c of pending) {
      if (o >= n) break;
      const take = Math.min(c.length, n - o);
      head.set(c.subarray(0, take), o);
      o += take;
    }
    return head;
  };

  /** Whether `head` (the first bytes of the file) is enough to decide ASCII vs binary. */
  const canDecide = (head: Uint8Array): boolean => {
    const n = head.length;
    if (n >= STL_DETECT_BYTES) return true;
    if (totalBytes === undefined) return false;
    if (n >= totalBytes) return true;
    return n >= HEADER_BYTES && HEADER_BYTES + RECORD_BYTES * dataViewOf(head).getUint32(80, true) === totalBytes;
  };

  const decide = (atFinish: boolean) => {
    const head = pendingHead();
    const knownTotal = totalBytes !== undefined ? totalBytes : atFinish ? pendingLen : undefined;
    const kind = detectSTLFormat(head, knownTotal);
    if (kind === "ascii") {
      mode = MODE_ASCII;
      return;
    }
    mode = MODE_BINARY;
    const chunks = pending;
    pending = [];
    pendingLen = 0;
    for (const c of chunks) feedBinary(c);
  };

  return {
    push(chunk: Uint8Array): void {
      if (finished) throw new Error("STL stream parser: push() after finish()");
      if (chunk.length === 0) return;
      bytesSeen += chunk.length;
      if (mode === MODE_UNDECIDED && pendingLen === 0 && canDecide(chunk)) {
        // fast path: the first chunk alone is enough to decide — no copy for binary data
        if (detectSTLFormat(chunk.subarray(0, STL_DETECT_BYTES), totalBytes) === "binary") {
          mode = MODE_BINARY;
        } else {
          mode = MODE_ASCII;
        }
      }
      if (mode === MODE_BINARY) {
        feedBinary(chunk);
      } else {
        pending.push(chunk.slice());
        pendingLen += chunk.length;
        if (mode === MODE_UNDECIDED && canDecide(pendingHead())) decide(false);
      }
      report();
    },

    finish(): MeshData {
      if (finished) throw new Error("STL stream parser: finish() called twice");
      finished = true;
      if (mode === MODE_UNDECIDED) {
        if (pendingLen === 0) return emptyMesh();
        decide(true);
      }
      if (mode === MODE_ASCII) {
        const all = concatPending();
        pending = [];
        return parseAsciiSTL(all, onProgress);
      }
      onProgress?.(1, "Reading STL");
      if (!welder || !idx || records === 0) return emptyMesh();
      return { positions: welder.positions(), indices: idx.finish() };
    },
  };
}

/**
 * Write a little-endian binary STL. Facet normals are computed from the vertices (unit length,
 * zero for degenerate triangles); the attribute word is 0.
 * The 80-byte header never starts with "solid" (readers would otherwise mistake the file for
 * ASCII): a header that does is prefixed with "Resurface ".
 */
export function writeSTLBinary(mesh: MeshData, header?: string): Uint8Array {
  const pos = mesh.positions;
  const ind = mesh.indices;
  const nt = Math.floor(ind.length / 3);
  const out = new Uint8Array(HEADER_BYTES + RECORD_BYTES * nt);
  const dv = new DataView(out.buffer);

  let h = header ?? "Resurface binary STL";
  if (/^\s*solid/i.test(h)) h = "Resurface " + h;
  for (let i = 0; i < 80; i++) {
    if (i < h.length) {
      const c = h.charCodeAt(i);
      out[i] = c >= 32 && c < 127 ? c : c === 9 || c === 10 || c === 13 ? 32 : 63; /* '?' */
    } else {
      out[i] = 32;
    }
  }
  dv.setUint32(80, nt, true);

  let o = HEADER_BYTES;
  for (let t = 0; t < nt; t++, o += RECORD_BYTES) {
    const a = ind[3 * t] * 3, b = ind[3 * t + 1] * 3, c = ind[3 * t + 2] * 3;
    const ax = pos[a], ay = pos[a + 1], az = pos[a + 2];
    const bx = pos[b], by = pos[b + 1], bz = pos[b + 2];
    const cx = pos[c], cy = pos[c + 1], cz = pos[c + 2];
    const ux = bx - ax, uy = by - ay, uz = bz - az;
    const vx = cx - ax, vy = cy - ay, vz = cz - az;
    let nx = uy * vz - uz * vy;
    let ny = uz * vx - ux * vz;
    let nz = ux * vy - uy * vx;
    const l = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (l > 0 && Number.isFinite(l)) {
      nx /= l;
      ny /= l;
      nz /= l;
    } else {
      nx = ny = nz = 0;
    }
    dv.setFloat32(o, nx, true);
    dv.setFloat32(o + 4, ny, true);
    dv.setFloat32(o + 8, nz, true);
    dv.setFloat32(o + 12, ax, true);
    dv.setFloat32(o + 16, ay, true);
    dv.setFloat32(o + 20, az, true);
    dv.setFloat32(o + 24, bx, true);
    dv.setFloat32(o + 28, by, true);
    dv.setFloat32(o + 32, bz, true);
    dv.setFloat32(o + 36, cx, true);
    dv.setFloat32(o + 40, cy, true);
    dv.setFloat32(o + 44, cz, true);
    dv.setUint16(o + 48, 0, true);
  }
  return out;
}
