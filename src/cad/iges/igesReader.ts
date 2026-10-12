/**
 * Minimal pure-JS IGES reader: parses the fixed-format sections, validates record structure
 * (80 columns, sequence numbers, P-section back-pointers, Terminate counts) and decodes the
 * NURBS entities (126 / 128) into the shared B-spline definitions. Used to sanity-check our own
 * IGES output in tests; also a starting point for IGES import.
 */
import type { BSplineCurve2Def, BSplineCurveDef, BSplineSurfaceDef, Vec3 } from "../../core/types";

export type IgesParam = number | string | null;

export interface IgesEntity {
  /** DE pointer (sequence number of the first directory line) */
  de: number;
  type: number;
  form: number;
  status: string;
  label: string;
  subscript: number;
  transform: number;
  paramPointer: number;
  paramLineCount: number;
  /** parameters after the entity type number */
  params: IgesParam[];
}

export interface IgesFile {
  start: string;
  global: IgesParam[];
  entities: IgesEntity[];
  /** lines per section as found */
  counts: { S: number; G: number; D: number; P: number };
  /** counts declared in the Terminate record */
  terminate: { S: number; G: number; D: number; P: number };
}

/** Split free-format parameter text into values, honouring Hollerith strings. */
export function tokenizeParams(text: string, pDelim = ",", rDelim = ";"): IgesParam[] {
  const out: IgesParam[] = [];
  let i = 0;
  let cur = "";
  let isString: string | null = null;
  const flush = () => {
    if (isString !== null) out.push(isString);
    else {
      const t = cur.trim();
      if (!t.length) out.push(null);
      else {
        const v = Number(t.replace(/[dD]/, "E"));
        if (Number.isNaN(v)) throw new Error(`IGES: cannot parse parameter '${t}'`);
        out.push(v);
      }
    }
    cur = "";
    isString = null;
  };
  while (i < text.length) {
    const ch = text[i];
    if (ch === pDelim) { flush(); i++; continue; }
    if (ch === rDelim) { flush(); return out; }
    const m = /^\s*(\d+)H/.exec(text.slice(i, i + 12));
    if (!cur.trim() && isString === null && m) {
      const len = Number(m[1]);
      const s0 = i + m[0].length;
      isString = text.slice(s0, s0 + len);
      i = s0 + len;
      continue;
    }
    cur += ch;
    i++;
  }
  flush();
  return out;
}

export function parseIGES(text: string): IgesFile {
  const raw = text.split(/\r?\n/);
  while (raw.length && raw[raw.length - 1] === "") raw.pop();
  const sections: Record<string, string[]> = { S: [], G: [], D: [], P: [], T: [] };
  const order = "SGDPT";
  let lastSection = 0;
  raw.forEach((line, n) => {
    if (line.length !== 80) throw new Error(`IGES: line ${n + 1} has ${line.length} columns (expected 80)`);
    const letter = line[72];
    const idx = order.indexOf(letter);
    if (idx < 0) throw new Error(`IGES: line ${n + 1} has unknown section letter '${letter}'`);
    if (idx < lastSection) throw new Error(`IGES: section '${letter}' out of order at line ${n + 1}`);
    lastSection = idx;
    const seq = Number(line.slice(73));
    if (seq !== sections[letter].length + 1) throw new Error(`IGES: bad sequence number ${seq} at line ${n + 1}`);
    sections[letter].push(line);
  });
  if (sections.T.length !== 1) throw new Error("IGES: missing or duplicated Terminate record");
  const t = sections.T[0];
  const terminate = {
    S: Number(t.slice(1, 8)),
    G: Number(t.slice(9, 16)),
    D: Number(t.slice(17, 24)),
    P: Number(t.slice(25, 32)),
  };
  const counts = { S: sections.S.length, G: sections.G.length, D: sections.D.length, P: sections.P.length };
  for (const k of ["S", "G", "D", "P"] as const)
    if (terminate[k] !== counts[k]) throw new Error(`IGES: Terminate record says ${terminate[k]} ${k} lines, found ${counts[k]}`);

  const gText = sections.G.map((l) => l.slice(0, 72)).join("");
  let pDelim = ",", rDelim = ";";
  if (gText.startsWith("1H")) pDelim = gText[2];
  const global = tokenizeParams(gText, pDelim, "\u0000");
  // second global parameter is the record delimiter
  if (typeof global[1] === "string" && global[1].length === 1) rDelim = global[1];
  const globalTrimmed = tokenizeParams(gText, pDelim, rDelim);

  if (sections.D.length % 2) throw new Error("IGES: odd number of directory entry lines");
  const field = (line: string, k: number) => line.slice(8 * k, 8 * k + 8);
  const entities: IgesEntity[] = [];
  for (let i = 0; i < sections.D.length; i += 2) {
    const l1 = sections.D[i], l2 = sections.D[i + 1];
    const type = Number(field(l1, 0));
    if (Number(field(l2, 0)) !== type) throw new Error(`IGES: DE ${i + 1} entity type mismatch between its two lines`);
    const paramPointer = Number(field(l1, 1));
    const paramLineCount = Number(field(l2, 3));
    const de = i + 1;
    const pl = sections.P.slice(paramPointer - 1, paramPointer - 1 + paramLineCount);
    if (pl.length !== paramLineCount) throw new Error(`IGES: DE ${de} parameter lines out of range`);
    for (const l of pl)
      if (Number(l.slice(64, 72)) !== de) throw new Error(`IGES: P line '${l.slice(73)}' does not point back to DE ${de}`);
    const ptext = pl.map((l) => l.slice(0, 64)).join("");
    const all = tokenizeParams(ptext, pDelim, rDelim);
    if (all[0] !== type) throw new Error(`IGES: DE ${de} parameter data starts with ${all[0]}, expected ${type}`);
    entities.push({
      de,
      type,
      form: Number(field(l2, 4)) || 0,
      status: field(l1, 8),
      label: field(l2, 7).trim(),
      subscript: Number(field(l2, 8)) || 0,
      transform: Number(field(l1, 6)) || 0,
      paramPointer,
      paramLineCount,
      params: all.slice(1),
    });
  }
  return {
    start: sections.S.map((l) => l.slice(0, 72)).join("").trimEnd(),
    global: globalTrimmed,
    entities,
    counts,
    terminate,
  };
}

const num = (p: IgesParam[], i: number, what: string): number => {
  const v = p[i];
  if (typeof v !== "number") throw new Error(`IGES: expected number for ${what} (param ${i})`);
  return v;
};

/** Decode entity 126. 2D curves (parameter space) are returned with `is2d` and z dropped. */
export function decodeBSplineCurve(e: IgesEntity): { curve: BSplineCurveDef; range: [number, number]; planar: boolean; closed: boolean } {
  if (e.type !== 126) throw new Error(`IGES: entity ${e.de} is ${e.type}, not 126`);
  const p = e.params;
  const K = num(p, 0, "K"), M = num(p, 1, "M");
  const n = K + 1;
  let k = 6;
  const knots: number[] = [];
  for (let i = 0; i < n + M + 1; i++) knots.push(num(p, k++, "knot"));
  const weights: number[] = [];
  for (let i = 0; i < n; i++) weights.push(num(p, k++, "weight"));
  const cps: Vec3[] = [];
  for (let i = 0; i < n; i++) cps.push([num(p, k++, "x"), num(p, k++, "y"), num(p, k++, "z")]);
  const range: [number, number] = [num(p, k++, "V0"), num(p, k++, "V1")];
  const polynomial = num(p, 4, "PROP3") === 1;
  return {
    curve: { degree: M, knots, controlPoints: cps, weights: polynomial ? undefined : weights },
    range,
    planar: num(p, 2, "PROP1") === 1,
    closed: num(p, 3, "PROP2") === 1,
  };
}

export function decodeBSplineCurve2(e: IgesEntity): BSplineCurve2Def {
  const { curve } = decodeBSplineCurve(e);
  return { degree: curve.degree, knots: curve.knots, controlPoints: curve.controlPoints.map((c) => [c[0], c[1]]), weights: curve.weights };
}

/** Decode entity 128. */
export function decodeBSplineSurface(e: IgesEntity): { surface: BSplineSurfaceDef; uRange: [number, number]; vRange: [number, number] } {
  if (e.type !== 128) throw new Error(`IGES: entity ${e.de} is ${e.type}, not 128`);
  const p = e.params;
  const K1 = num(p, 0, "K1"), K2 = num(p, 1, "K2"), M1 = num(p, 2, "M1"), M2 = num(p, 3, "M2");
  const nu = K1 + 1, nv = K2 + 1;
  let k = 9;
  const knotsU: number[] = [], knotsV: number[] = [];
  for (let i = 0; i < nu + M1 + 1; i++) knotsU.push(num(p, k++, "knotU"));
  for (let i = 0; i < nv + M2 + 1; i++) knotsV.push(num(p, k++, "knotV"));
  const w = new Array<number>(nu * nv);
  for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) w[i * nv + j] = num(p, k++, "weight");
  const cps: Vec3[] = new Array(nu * nv);
  for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) cps[i * nv + j] = [num(p, k++, "x"), num(p, k++, "y"), num(p, k++, "z")];
  const uRange: [number, number] = [num(p, k++, "U0"), num(p, k++, "U1")];
  const vRange: [number, number] = [num(p, k++, "V0"), num(p, k++, "V1")];
  const polynomial = num(p, 6, "PROP3") === 1;
  return {
    surface: { degreeU: M1, degreeV: M2, knotsU, knotsV, nu, nv, controlPoints: cps, weights: polynomial ? undefined : w },
    uRange,
    vRange,
  };
}

/** Resolve a DE pointer to its entity. */
export function entityByDE(file: IgesFile, de: number): IgesEntity {
  const e = file.entities[(de - 1) / 2];
  if (!e || e.de !== de) throw new Error(`IGES: dangling DE pointer ${de}`);
  return e;
}

/** Component entities of a curve pointer (expands 102 composite curves). */
export function curveComponents(file: IgesFile, de: number): IgesEntity[] {
  const e = entityByDE(file, de);
  if (e.type !== 102) return [e];
  const n = e.params[0] as number;
  return e.params.slice(1, 1 + n).map((d) => entityByDE(file, d as number));
}
