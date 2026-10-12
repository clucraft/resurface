/**
 * Stanford PLY reader: ascii, binary_little_endian and binary_big_endian.
 *
 * - Any number of elements in any order; unknown elements are skipped (binary lists are
 *   parsed to advance correctly).
 * - Scalar types char/int8, uchar/uint8, short/int16, ushort/uint16, int/int32, uint/uint32,
 *   float/float32, double/float64; list properties with any count / item type.
 * - `vertex` element: x,y,z (any numeric type), optional nx,ny,nz (or normal_x/y/z), colour
 *   red/green/blue (also r/g/b, diffuse_red/green/blue). uchar colours are used as-is, other
 *   integer types are clamped to 0..255, float colours are taken as 0..1 unless some value
 *   exceeds 1.01 (then 0..255). Extra properties (incl. lists) are skipped.
 * - `face` element: list `vertex_indices` / `vertex_index` (else the first list property),
 *   polygons fan-triangulated, faces with out-of-range indices dropped, extra props skipped.
 * - `tristrips` element: list `vertex_indices` with -1 restarts (alternating winding;
 *   degenerate stitching triangles are dropped).
 * - Truncated files: everything complete is kept.
 *
 * Without faces the result is a point cloud.
 */
import type { MeshData, PointCloudData, ProgressFn } from "../../core/types";
import { GrowableU32 } from "../util/growable";
import { ByteScanner, dataViewOf, toBytes } from "./text";

export const PLY_INT8 = 0;
export const PLY_UINT8 = 1;
export const PLY_INT16 = 2;
export const PLY_UINT16 = 3;
export const PLY_INT32 = 4;
export const PLY_UINT32 = 5;
export const PLY_FLOAT32 = 6;
export const PLY_FLOAT64 = 7;

const TYPE_SIZE = [1, 1, 2, 2, 4, 4, 4, 8];
const TYPE_BY_NAME: Record<string, number> = {
  char: PLY_INT8,
  int8: PLY_INT8,
  uchar: PLY_UINT8,
  uint8: PLY_UINT8,
  short: PLY_INT16,
  int16: PLY_INT16,
  ushort: PLY_UINT16,
  uint16: PLY_UINT16,
  int: PLY_INT32,
  int32: PLY_INT32,
  uint: PLY_UINT32,
  uint32: PLY_UINT32,
  float: PLY_FLOAT32,
  float32: PLY_FLOAT32,
  double: PLY_FLOAT64,
  float64: PLY_FLOAT64,
};

export interface PLYProperty {
  name: string;
  /** scalar type (PLY_* constant); for lists the item type */
  type: number;
  isList: boolean;
  /** list count type (lists only) */
  countType: number;
}

export interface PLYElement {
  name: string;
  count: number;
  properties: PLYProperty[];
}

export type PLYFormat = "ascii" | "binary_little_endian" | "binary_big_endian";

export interface PLYHeader {
  format: PLYFormat;
  elements: PLYElement[];
  /** byte offset of the first data byte (just after the end_header line) */
  dataOffset: number;
}

export type PLYResult = { kind: "mesh"; mesh: MeshData } | { kind: "points"; points: PointCloudData };

const MAX_HEADER_LINE = 1 << 16;
const COLOR_255_THRESHOLD = 1.01;
const PROGRESS_RECORDS = 1 << 20;

function plyType(name: string): number {
  const t = TYPE_BY_NAME[name.toLowerCase()];
  if (t === undefined) throw new Error(`PLY: unsupported property type "${name}"`);
  return t;
}

/** Parse the PLY header. Throws on a malformed header. */
export function parsePLYHeader(bytes: Uint8Array): PLYHeader {
  const s = new ByteScanner(bytes);
  if (bytes.length >= 3 && bytes[0] === 0xef && bytes[1] === 0xbb && bytes[2] === 0xbf) s.pos = 3;
  const first = s.readLineString().trim().toLowerCase();
  if (first !== "ply") throw new Error("PLY: missing 'ply' magic line");
  let format: PLYFormat | null = null;
  const elements: PLYElement[] = [];
  for (;;) {
    if (s.eof()) throw new Error("PLY: header has no end_header line");
    const nl = bytes.indexOf(10, s.pos);
    if ((nl < 0 ? bytes.length : nl) - s.pos > MAX_HEADER_LINE) throw new Error("PLY: header line too long");
    const line = s.readLineString().trim();
    if (line.length === 0) continue;
    const parts = line.split(/\s+/);
    const kw = parts[0].toLowerCase();
    if (kw === "end_header") break;
    if (kw === "comment" || kw === "obj_info") continue;
    if (kw === "format") {
      const f = (parts[1] ?? "").toLowerCase();
      if (f !== "ascii" && f !== "binary_little_endian" && f !== "binary_big_endian") {
        throw new Error(`PLY: unsupported format "${parts[1] ?? ""}"`);
      }
      format = f;
    } else if (kw === "element") {
      const count = Number(parts[2]);
      if (!Number.isFinite(count) || count < 0) throw new Error(`PLY: bad element count in "${line}"`);
      elements.push({ name: parts[1] ?? "", count: Math.floor(count), properties: [] });
    } else if (kw === "property") {
      const el = elements[elements.length - 1];
      if (!el) throw new Error("PLY: property before any element");
      if ((parts[1] ?? "").toLowerCase() === "list") {
        if (parts.length < 5) throw new Error(`PLY: bad list property "${line}"`);
        el.properties.push({ name: parts[4], type: plyType(parts[3]), isList: true, countType: plyType(parts[2]) });
      } else {
        if (parts.length < 3) throw new Error(`PLY: bad property "${line}"`);
        el.properties.push({ name: parts[2], type: plyType(parts[1]), isList: false, countType: 0 });
      }
    }
    // unknown header keywords are ignored
  }
  if (!format) throw new Error("PLY: missing format line");
  return { format, elements, dataOffset: s.pos };
}

/** Read one binary scalar. */
function readScalar(dv: DataView, off: number, t: number, le: boolean): number {
  switch (t) {
    case PLY_INT8:
      return dv.getInt8(off);
    case PLY_UINT8:
      return dv.getUint8(off);
    case PLY_INT16:
      return dv.getInt16(off, le);
    case PLY_UINT16:
      return dv.getUint16(off, le);
    case PLY_INT32:
      return dv.getInt32(off, le);
    case PLY_UINT32:
      return dv.getUint32(off, le);
    case PLY_FLOAT32:
      return dv.getFloat32(off, le);
    default:
      return dv.getFloat64(off, le);
  }
}

// vertex property roles
const R_SKIP = 0, R_X = 1, R_Y = 2, R_Z = 3, R_NX = 4, R_NY = 5, R_NZ = 6, R_R = 7, R_G = 8, R_B = 9;

function vertexRole(name: string): number {
  switch (name.toLowerCase()) {
    case "x":
      return R_X;
    case "y":
      return R_Y;
    case "z":
      return R_Z;
    case "nx":
    case "normal_x":
      return R_NX;
    case "ny":
    case "normal_y":
      return R_NY;
    case "nz":
    case "normal_z":
      return R_NZ;
    case "red":
    case "r":
    case "diffuse_red":
      return R_R;
    case "green":
    case "g":
    case "diffuse_green":
      return R_G;
    case "blue":
    case "b":
    case "diffuse_blue":
      return R_B;
    default:
      return R_SKIP;
  }
}

/** Flattened property metadata in typed arrays (no per-record object lookups). */
interface PropTable {
  n: number;
  type: Int32Array;
  isList: Uint8Array;
  countType: Int32Array;
  size: Int32Array;
  /** bytes of the record that are always present (scalars + list counts) */
  minRecordBytes: number;
  hasList: boolean;
}

function propTable(el: PLYElement): PropTable {
  const n = el.properties.length;
  const t: PropTable = {
    n,
    type: new Int32Array(n),
    isList: new Uint8Array(n),
    countType: new Int32Array(n),
    size: new Int32Array(n),
    minRecordBytes: 0,
    hasList: false,
  };
  for (let j = 0; j < n; j++) {
    const p = el.properties[j];
    t.type[j] = p.type;
    t.isList[j] = p.isList ? 1 : 0;
    t.countType[j] = p.countType;
    t.size[j] = TYPE_SIZE[p.type];
    t.minRecordBytes += p.isList ? TYPE_SIZE[p.countType] : TYPE_SIZE[p.type];
    if (p.isList) t.hasList = true;
  }
  return t;
}

function clampByte(v: number): number {
  return v >= 255 ? 255 : v > 0 ? Math.round(v) : 0;
}

const EMPTY_F32 = new Float32Array(0);
const EMPTY_U8 = new Uint8Array(0);

function isFloatType(t: number): boolean {
  return t === PLY_FLOAT32 || t === PLY_FLOAT64;
}

/** Destination of vertex property values (absent attributes use empty arrays, never null). */
class VertexWriter {
  count = 0;
  readonly positions: Float32Array;
  readonly normals: Float32Array;
  readonly colors: Uint8Array;
  /** raw float colours (when the colour properties are float/double) */
  readonly colorRaw: Float32Array;
  colorMax = 0;
  readonly hasNormals: boolean;
  readonly hasColors: boolean;
  readonly floatColors: boolean;
  /** role per property (R_*) */
  readonly roles: Int32Array;

  constructor(el: PLYElement, count: number) {
    const roles = new Int32Array(el.properties.length);
    let hasN = 0, hasC = 0, floatC = false;
    for (let j = 0; j < el.properties.length; j++) {
      const p = el.properties[j];
      const r = p.isList ? R_SKIP : vertexRole(p.name);
      roles[j] = r;
      if (r >= R_NX && r <= R_NZ) hasN |= 1 << (r - R_NX);
      if (r >= R_R && r <= R_B) {
        hasC |= 1 << (r - R_R);
        if (isFloatType(p.type)) floatC = true;
      }
    }
    // incomplete normal / colour triplets are ignored
    for (let j = 0; j < roles.length; j++) {
      const r = roles[j];
      if (hasN !== 7 && r >= R_NX && r <= R_NZ) roles[j] = R_SKIP;
      if (hasC !== 7 && r >= R_R && r <= R_B) roles[j] = R_SKIP;
    }
    this.roles = roles;
    this.hasNormals = hasN === 7;
    this.hasColors = hasC === 7;
    this.floatColors = this.hasColors && floatC;
    this.positions = new Float32Array(count * 3);
    this.normals = this.hasNormals ? new Float32Array(count * 3) : EMPTY_F32;
    this.colors = this.hasColors && !floatC ? new Uint8Array(count * 3) : EMPTY_U8;
    this.colorRaw = this.floatColors ? new Float32Array(count * 3) : EMPTY_F32;
  }

  set(i: number, role: number, v: number): void {
    switch (role) {
      case R_X:
        this.positions[3 * i] = v;
        break;
      case R_Y:
        this.positions[3 * i + 1] = v;
        break;
      case R_Z:
        this.positions[3 * i + 2] = v;
        break;
      case R_NX:
        this.normals[3 * i] = v;
        break;
      case R_NY:
        this.normals[3 * i + 1] = v;
        break;
      case R_NZ:
        this.normals[3 * i + 2] = v;
        break;
      case R_R:
      case R_G:
      case R_B:
        if (this.floatColors) {
          this.colorRaw[3 * i + role - R_R] = v;
          if (v > this.colorMax) this.colorMax = v;
        } else {
          this.colors[3 * i + role - R_R] = clampByte(v); // uchar values pass through unchanged
        }
        break;
    }
  }
}

/** Shared face / tristrip accumulation. */
class FaceSink {
  readonly idx: GrowableU32;
  private scratch = new Float64Array(64);
  n = 0;
  constructor(cap: number) {
    this.idx = new GrowableU32(cap);
  }
  begin(count: number): void {
    if (count > this.scratch.length) {
      let c = this.scratch.length;
      while (c < count) c *= 2;
      this.scratch = new Float64Array(c);
    }
    this.n = 0;
  }
  add(v: number): void {
    this.scratch[this.n++] = v;
  }
  /** Fan-triangulate the collected polygon (dropped when any index is invalid). */
  endPolygon(nv: number): void {
    const n = this.n, sc = this.scratch;
    if (n < 3) return;
    for (let i = 0; i < n; i++) {
      const v = sc[i];
      if (!(v >= 0 && v < nv) || v !== Math.floor(v)) return;
    }
    const v0 = sc[0];
    this.idx.reserve((n - 2) * 3);
    for (let i = 1; i + 1 < n; i++) this.idx.push3(v0, sc[i], sc[i + 1]);
  }
  /** Triangle strip(s) with -1 restarts. */
  endStrip(nv: number): void {
    const n = this.n, sc = this.scratch;
    let start = 0;
    for (let i = 0; i <= n; i++) {
      if (i < n && sc[i] >= 0) continue;
      // strip [start, i)
      for (let k = start + 2; k < i; k++) {
        const a = sc[k - 2], b = sc[k - 1], c = sc[k];
        if (a === b || b === c || a === c) continue;
        if (!(a < nv && b < nv && c < nv)) continue;
        if (a !== Math.floor(a) || b !== Math.floor(b) || c !== Math.floor(c)) continue;
        if ((k - start) % 2 === 0) this.idx.push3(a, b, c);
        else this.idx.push3(b, a, c);
      }
      start = i + 1;
    }
  }
}

function faceListIndex(el: PLYElement): number {
  let first = -1;
  for (let j = 0; j < el.properties.length; j++) {
    const p = el.properties[j];
    if (!p.isList) continue;
    const nm = p.name.toLowerCase();
    if (nm === "vertex_indices" || nm === "vertex_index") return j;
    if (first < 0) first = j;
  }
  return first;
}

/**
 * Element-level readers over the data section. Kept as small methods (rather than one big
 * function) so the JIT can optimise each hot loop separately.
 */
class PlyBody {
  readonly s: ByteScanner;
  readonly dv: DataView;
  readonly len: number;
  readonly le: boolean;
  readonly ascii: boolean;
  /** binary read offset */
  off: number;
  private readonly onProgress: ProgressFn | undefined;

  constructor(bytes: Uint8Array, header: PLYHeader, onProgress: ProgressFn | undefined) {
    this.s = new ByteScanner(bytes, header.dataOffset);
    this.dv = dataViewOf(bytes);
    this.len = bytes.length;
    this.le = header.format === "binary_little_endian";
    this.ascii = header.format === "ascii";
    this.off = header.dataOffset;
    this.onProgress = onProgress;
  }

  report(): void {
    this.onProgress?.(Math.min(1, (this.ascii ? this.s.pos : this.off) / Math.max(1, this.len)), "Reading PLY");
  }

  /** Upper bound of records of this element that can fit in the remaining data. */
  maxRecords(pt: PropTable, declared: number): number {
    const remaining = this.len - (this.ascii ? this.s.pos : this.off);
    if (this.ascii) return Math.min(declared, Math.floor(remaining / Math.max(1, 2 * pt.n)) + 1);
    return pt.minRecordBytes > 0 ? Math.min(declared, Math.floor(remaining / pt.minRecordBytes)) : declared;
  }

  readVerticesAscii(pt: PropTable, w: VertexWriter, count: number): number {
    const s = this.s, len = this.len, n = pt.n, isList = pt.isList, roles = w.roles;
    let i = 0;
    for (; i < count; i++) {
      s.skipWhitespace();
      if (s.pos >= len) break;
      for (let j = 0; j < n; j++) {
        if (isList[j]) {
          const c = s.readFloat();
          for (let k = 0; k < c; k++) s.readFloat();
        } else {
          let v = s.readFloat();
          if (!s.ok) v = 0;
          const r = roles[j];
          if (r !== R_SKIP) w.set(i, r, v);
        }
      }
      s.nextLine();
      if ((i & (PROGRESS_RECORDS - 1)) === PROGRESS_RECORDS - 1) this.report();
    }
    return i;
  }

  readVerticesBinary(pt: PropTable, w: VertexWriter, count: number): number {
    const dv = this.dv, len = this.len, le = this.le, n = pt.n, roles = w.roles;
    const isList = pt.isList, type = pt.type, size = pt.size, countType = pt.countType, minRec = pt.minRecordBytes;
    let off = this.off;
    let i = 0;
    outer: for (; i < count; i++) {
      if (off + minRec > len) break;
      const recStart = off;
      for (let j = 0; j < n; j++) {
        if (isList[j]) {
          const ct = countType[j];
          const c = readScalar(dv, off, ct, le);
          off += TYPE_SIZE[ct];
          const skip = c * size[j];
          if (!(c >= 0) || off + skip > len) {
            off = recStart;
            break outer;
          }
          off += skip;
        } else {
          const r = roles[j];
          if (r !== R_SKIP) w.set(i, r, readScalar(dv, off, type[j], le));
          off += size[j];
        }
      }
      if ((i & (PROGRESS_RECORDS - 1)) === PROGRESS_RECORDS - 1) {
        this.off = off;
        this.report();
      }
    }
    this.off = off;
    return i;
  }

  readFacesAscii(pt: PropTable, li: number, strip: boolean, sink: FaceSink, nv: number, count: number): number {
    const s = this.s, len = this.len, n = pt.n, isList = pt.isList;
    let i = 0;
    for (; i < count; i++) {
      s.skipWhitespace();
      if (s.pos >= len) break;
      for (let j = 0; j < n; j++) {
        if (isList[j]) {
          let c = s.readFloat();
          if (!s.ok || !(c >= 0)) c = 0;
          if (j === li) {
            sink.begin(c);
            for (let k = 0; k < c; k++) {
              const v = s.readFloat();
              sink.add(s.ok ? v : -1);
            }
            if (strip) sink.endStrip(nv);
            else sink.endPolygon(nv);
          } else {
            for (let k = 0; k < c; k++) s.readFloat();
          }
        } else {
          s.readFloat();
        }
      }
      s.nextLine();
      if ((i & (PROGRESS_RECORDS - 1)) === PROGRESS_RECORDS - 1) this.report();
    }
    return i;
  }

  readFacesBinary(pt: PropTable, li: number, strip: boolean, sink: FaceSink, nv: number, count: number): number {
    const dv = this.dv, len = this.len, le = this.le, n = pt.n;
    const isList = pt.isList, type = pt.type, size = pt.size, countType = pt.countType, minRec = pt.minRecordBytes;
    let off = this.off;
    let i = 0;
    outer: for (; i < count; i++) {
      if (off + minRec > len) break;
      const recStart = off;
      for (let j = 0; j < n; j++) {
        if (isList[j]) {
          const ct = countType[j];
          const c = readScalar(dv, off, ct, le);
          off += TYPE_SIZE[ct];
          const isz = size[j];
          if (!(c >= 0) || off + c * isz > len) {
            off = recStart;
            break outer;
          }
          if (j === li) {
            const it = type[j];
            sink.begin(c);
            for (let k = 0; k < c; k++, off += isz) sink.add(readScalar(dv, off, it, le));
            if (strip) sink.endStrip(nv);
            else sink.endPolygon(nv);
          } else {
            off += c * isz;
          }
        } else {
          off += size[j];
        }
      }
      if ((i & (PROGRESS_RECORDS - 1)) === PROGRESS_RECORDS - 1) {
        this.off = off;
        this.report();
      }
    }
    this.off = off;
    return i;
  }

  /** Skip `count` records of an element; returns the number of complete records skipped. */
  skipElement(pt: PropTable, count: number): number {
    if (this.ascii) {
      const s = this.s;
      let i = 0;
      for (; i < count; i++) {
        s.skipWhitespace();
        if (s.pos >= this.len) break;
        s.nextLine();
      }
      return i;
    }
    if (!pt.hasList) {
      const fit = pt.minRecordBytes > 0 ? Math.min(count, Math.floor((this.len - this.off) / pt.minRecordBytes)) : count;
      this.off += fit * pt.minRecordBytes;
      return fit;
    }
    const dv = this.dv, len = this.len, le = this.le, n = pt.n;
    let off = this.off;
    let i = 0;
    outer: for (; i < count; i++) {
      if (off + pt.minRecordBytes > len) break;
      const recStart = off;
      for (let j = 0; j < n; j++) {
        if (pt.isList[j]) {
          const ct = pt.countType[j];
          const c = readScalar(dv, off, ct, le);
          off += TYPE_SIZE[ct];
          const skip = c * pt.size[j];
          if (!(c >= 0) || off + skip > len) {
            off = recStart;
            break outer;
          }
          off += skip;
        } else off += pt.size[j];
      }
    }
    this.off = off;
    return i;
  }
}

function trimTo<T extends Float32Array | Uint8Array>(a: T, n: number): T {
  return a.length === n ? a : (a.slice(0, n) as T);
}

/**
 * Parse a PLY file. Returns a mesh when at least one triangle was read, otherwise the vertices
 * as a point cloud.
 */
export function parsePLY(data: ArrayBuffer | Uint8Array, onProgress?: ProgressFn): PLYResult {
  const bytes = toBytes(data);
  const header = parsePLYHeader(bytes);
  const body = new PlyBody(bytes, header, onProgress);

  const vertexEl = header.elements.find((e) => {
    const n = e.name.toLowerCase();
    return n === "vertex" || n === "vertices";
  });
  const declaredVertices = vertexEl ? vertexEl.count : 0;
  let vw: VertexWriter | null = null;
  let faces: FaceSink | null = null;

  for (const el of header.elements) {
    const pt = propTable(el);
    const name = el.name.toLowerCase();
    // never trust a count that cannot fit in the remaining data
    const count = body.maxRecords(pt, el.count);
    let done: number;
    if (el === vertexEl) {
      const w = new VertexWriter(el, count);
      vw = w;
      done = body.ascii ? body.readVerticesAscii(pt, w, count) : body.readVerticesBinary(pt, w, count);
      w.count = done;
    } else if (name === "face" || name === "faces" || name === "tristrips") {
      if (!faces) faces = new FaceSink(Math.max(16, Math.min(count, 1 << 24) * 3));
      const li = faceListIndex(el);
      const strip = name === "tristrips";
      done = body.ascii
        ? body.readFacesAscii(pt, li, strip, faces, declaredVertices, count)
        : body.readFacesBinary(pt, li, strip, faces, declaredVertices, count);
    } else {
      done = body.skipElement(pt, count);
    }
    body.report();
    if (done < el.count) break; // truncated: nothing after this point is complete
  }

  // ---- assemble ----------------------------------------------------------------------
  const nv = vw ? vw.count : 0;
  const positions = vw ? trimTo(vw.positions, nv * 3) : new Float32Array(0);
  let normals: Float32Array | undefined;
  if (vw && vw.hasNormals) {
    const n = trimTo(vw.normals, nv * 3);
    for (let i = 0; i < nv; i++) {
      const x = n[3 * i], y = n[3 * i + 1], z = n[3 * i + 2];
      const l = Math.sqrt(x * x + y * y + z * z);
      if (l > 0 && l !== 1) {
        n[3 * i] = x / l;
        n[3 * i + 1] = y / l;
        n[3 * i + 2] = z / l;
      }
    }
    normals = n;
  }
  let colors: Uint8Array | undefined;
  if (vw && vw.hasColors) {
    if (vw.floatColors) {
      const raw = vw.colorRaw;
      const scale = vw.colorMax > COLOR_255_THRESHOLD ? 1 : 255;
      colors = new Uint8Array(nv * 3);
      for (let k = 0; k < nv * 3; k++) colors[k] = clampByte(raw[k] * scale);
    } else colors = trimTo(vw.colors, nv * 3);
  }

  let indices: Uint32Array = faces ? faces.idx.finish() : new Uint32Array(0);
  if (indices.length > 0 && nv < declaredVertices) {
    // the vertex element was truncated: drop triangles referencing missing vertices
    let w = 0;
    for (let t = 0; t < indices.length; t += 3) {
      const a = indices[t], b = indices[t + 1], c = indices[t + 2];
      if (a < nv && b < nv && c < nv) {
        indices[w++] = a;
        indices[w++] = b;
        indices[w++] = c;
      }
    }
    indices = indices.slice(0, w);
  }
  onProgress?.(1, "Reading PLY");

  if (indices.length === 0) {
    const points: PointCloudData = { positions };
    if (normals) points.normals = normals;
    if (colors) points.colors = colors;
    return { kind: "points", points };
  }
  const mesh: MeshData = { positions, indices };
  if (normals) mesh.normals = normals;
  if (colors) mesh.colors = colors;
  return { kind: "mesh", mesh };
}
