/**
 * Standalone IGES 5.3 writer for NURBS geometry (no OCCT dependency).
 *
 * Emitted entities:
 *   128  Rational B-spline surface
 *   126  Rational B-spline curve (3D model-space curves, and 2D parameter-space curves with z = 0)
 *   102  Composite curve (chains the curves of one trimming loop)
 *   142  Curve on a parametric surface (one per trimming loop)
 *   144  Trimmed parametric surface
 *
 * Fixed 80-column records: Start (S), Global (G), Directory Entry (D, two lines per entity),
 * Parameter Data (P) and Terminate (T). Parameter delimiter `,`, record delimiter `;`,
 * strings as Hollerith constants. Units: millimetres.
 */
import type { BSplineCurve2Def, BSplineCurveDef, BSplineSurfaceDef, Vec3 } from "../../core/types";

/** One trimming curve: a model-space curve and/or its parameter-space (u, v) image. */
export interface IgesTrimCurve {
  curve3d?: BSplineCurveDef;
  curve2d?: BSplineCurve2Def;
}

/** Ordered, head-to-tail chain of trimming curves forming a closed loop. */
export type TrimLoop = IgesTrimCurve[];

export interface IgesSurfaceItem {
  surface: BSplineSurfaceDef;
  /** When omitted the surface is written as a plain (untrimmed) entity 128. */
  trims?: { outer: TrimLoop; inner: TrimLoop[] };
  /** Optional short label (first 8 characters go into the directory entry label). */
  name?: string;
}

export interface IgesModel {
  surfaces: IgesSurfaceItem[];
  /** Free-standing model-space curves (entity 126). */
  curves?: BSplineCurveDef[];
  units: "mm";
  author?: string;
  organization?: string;
  fileName?: string;
  productId?: string;
  /** Minimum user-intended resolution written to the Global section (default 1e-3 mm). */
  tolerance?: number;
  /** Timestamp for the Global section (default: now). */
  date?: Date;
  /** Text for the Start section. */
  description?: string;
}

// ---------------------------------------------------------------------------
// Parameter tokens
// ---------------------------------------------------------------------------

type Token = { k: "i"; v: number } | { k: "r"; v: number } | { k: "s"; v: string } | { k: "ref"; v: number } | { k: "def" };

const I = (v: number): Token => ({ k: "i", v: Math.round(v) });
const R = (v: number): Token => ({ k: "r", v });
const S = (v: string): Token => (v ? { k: "s", v } : { k: "def" });
const REF = (entityIndex: number): Token => ({ k: "ref", v: entityIndex });

/** IGES real: shortest round-trip digits, always with a decimal point and an `E` exponent. */
export function formatReal(x: number): string {
  if (!Number.isFinite(x) || x === 0) return "0.0";
  const [m0, e] = String(x).split("e");
  const m = m0.includes(".") ? m0 : `${m0}.0`;
  return e ? `${m}E${e.replace("+", "")}` : m;
}

const hollerith = (s: string) => `${s.length}H${s}`;

function tokenText(t: Token, dePointer: (idx: number) => number): string {
  switch (t.k) {
    case "i": return String(t.v);
    case "r": return formatReal(t.v);
    case "s": return hollerith(t.v);
    case "ref": return String(dePointer(t.v));
    case "def": return "";
  }
}

/** Pack tokens into lines of at most `width` characters without splitting a token. */
function packTokens(texts: string[], width: number, pDelim = ",", rDelim = ";"): string[] {
  const lines: string[] = [];
  let cur = "";
  texts.forEach((t, i) => {
    const piece = t + (i === texts.length - 1 ? rDelim : pDelim);
    if (cur.length + piece.length > width && cur.length > 0) {
      lines.push(cur);
      cur = "";
    }
    if (piece.length > width) {
      // only possible for very long strings: hard-split (legal for Hollerith data)
      let rest = piece;
      while (rest.length > width) {
        lines.push(rest.slice(0, width));
        rest = rest.slice(width);
      }
      cur = rest;
    } else cur += piece;
  });
  if (cur.length) lines.push(cur);
  return lines;
}

// ---------------------------------------------------------------------------
// Entities
// ---------------------------------------------------------------------------

interface Entity {
  type: number;
  form: number;
  /** 8-digit status number: blank(2) subordinate(2) use(2) hierarchy(2) */
  status: string;
  params: Token[];
  label?: string;
  subscript?: number;
}

const STATUS_INDEPENDENT = "00000000";
const STATUS_DEPENDENT = "00010000";
const STATUS_DEPENDENT_PARAM2D = "00010500";

function isClosed3(a: readonly number[], b: readonly number[], tol: number): boolean {
  let d = 0;
  for (let i = 0; i < a.length; i++) d += (a[i] - b[i]) ** 2;
  return Math.sqrt(d) <= tol;
}

function isPolynomial(w: readonly number[] | undefined): boolean {
  if (!w || !w.length) return true;
  return w.every((x) => Math.abs(x - w[0]) <= 1e-12 * Math.max(1, Math.abs(w[0])));
}

/** Plane normal if all points are coplanar (within tol), else null. */
function planeNormal(pts: readonly Vec3[], tol: number): Vec3 | null {
  if (pts.length < 3) return null;
  const o = pts[0];
  let best: Vec3 | null = null, bestLen = 0;
  // Pick the most robust normal from triples (o, a, b)
  for (let i = 1; i < pts.length; i++) {
    for (let j = i + 1; j < pts.length; j++) {
      const a = [pts[i][0] - o[0], pts[i][1] - o[1], pts[i][2] - o[2]];
      const b = [pts[j][0] - o[0], pts[j][1] - o[1], pts[j][2] - o[2]];
      const n: Vec3 = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
      const l = Math.hypot(n[0], n[1], n[2]);
      if (l > bestLen) { bestLen = l; best = n; }
      if (pts.length > 24 && j > i + 4) break; // keep it cheap for long control polygons
    }
  }
  if (!best || bestLen < 1e-20) return null;
  const n: Vec3 = [best[0] / bestLen, best[1] / bestLen, best[2] / bestLen];
  for (const p of pts) if (Math.abs((p[0] - o[0]) * n[0] + (p[1] - o[1]) * n[1] + (p[2] - o[2]) * n[2]) > tol) return null;
  return n;
}

function curveEntity(
  c: { degree: number; knots: readonly number[]; controlPoints: readonly (readonly number[])[]; weights?: readonly number[] },
  is2d: boolean,
  status: string,
  tol: number,
): Entity {
  const n = c.controlPoints.length;
  if (n < c.degree + 1 || c.knots.length !== n + c.degree + 1)
    throw new Error(`IGES: invalid B-spline curve (poles=${n}, degree=${c.degree}, knots=${c.knots.length})`);
  const pts: Vec3[] = c.controlPoints.map((p) => [p[0], p[1], is2d ? 0 : (p[2] ?? 0)]);
  const normal: Vec3 | null = is2d ? [0, 0, 1] : planeNormal(pts, tol);
  const params: Token[] = [
    I(n - 1),
    I(c.degree),
    I(normal ? 1 : 0),
    I(isClosed3(pts[0], pts[n - 1], tol) ? 1 : 0),
    I(isPolynomial(c.weights) ? 1 : 0),
    I(0),
  ];
  for (const k of c.knots) params.push(R(k));
  for (let i = 0; i < n; i++) params.push(R(c.weights ? c.weights[i] : 1));
  for (const p of pts) params.push(R(p[0]), R(p[1]), R(p[2]));
  params.push(R(c.knots[c.degree]), R(c.knots[n]));
  const nn = normal ?? [0, 0, 0];
  params.push(R(nn[0]), R(nn[1]), R(nn[2]));
  return { type: 126, form: 0, status, params };
}

function surfaceEntity(s: BSplineSurfaceDef, status: string, tol: number): Entity {
  const { nu, nv, degreeU: p, degreeV: q } = s;
  if (s.controlPoints.length !== nu * nv || s.knotsU.length !== nu + p + 1 || s.knotsV.length !== nv + q + 1)
    throw new Error("IGES: invalid B-spline surface definition");
  const P = s.controlPoints;
  let closedU = true, closedV = true;
  for (let j = 0; j < nv && closedU; j++) closedU = isClosed3(P[j], P[(nu - 1) * nv + j], tol);
  for (let i = 0; i < nu && closedV; i++) closedV = isClosed3(P[i * nv], P[i * nv + nv - 1], tol);
  const params: Token[] = [
    I(nu - 1), I(nv - 1), I(p), I(q),
    I(closedU ? 1 : 0), I(closedV ? 1 : 0),
    I(isPolynomial(s.weights) ? 1 : 0),
    I(0), I(0),
  ];
  for (const k of s.knotsU) params.push(R(k));
  for (const k of s.knotsV) params.push(R(k));
  // weights and control points: first (U) index varies fastest
  for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) params.push(R(s.weights ? s.weights[i * nv + j] : 1));
  for (let j = 0; j < nv; j++)
    for (let i = 0; i < nu; i++) {
      const c = P[i * nv + j];
      params.push(R(c[0]), R(c[1]), R(c[2]));
    }
  params.push(R(s.knotsU[p]), R(s.knotsU[nu]), R(s.knotsV[q]), R(s.knotsV[nv]));
  return { type: 128, form: 0, status, params };
}

function timestamp(d: Date): string {
  const p2 = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${p2(d.getMonth() + 1)}${p2(d.getDate())}.${p2(d.getHours())}${p2(d.getMinutes())}${p2(d.getSeconds())}`;
}

const seqField = (letter: string, n: number) => `${letter}${String(n).padStart(7, " ")}`;
const int8 = (n: number | string) => String(n).padStart(8, " ");

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

export function writeIGES(model: IgesModel): string {
  if (model.units !== "mm") throw new Error("IGES writer: only 'mm' units are supported");
  const tol = model.tolerance ?? 1e-3;
  const entities: Entity[] = [];
  const add = (e: Entity): number => entities.push(e) - 1;
  let maxCoord = 0;
  const track = (pts: readonly (readonly number[])[]) => {
    for (const p of pts) for (const c of p) maxCoord = Math.max(maxCoord, Math.abs(c));
  };

  const addLoop = (loop: TrimLoop, surfIdx: number, what: string): number => {
    if (!loop.length) throw new Error(`IGES: empty trimming loop (${what})`);
    const has2d = loop.every((c) => c.curve2d);
    const c3 = loop.filter((c) => c.curve3d).map((c) => c.curve3d!);
    if (!has2d && !c3.length) throw new Error(`IGES: trimming loop without curves (${what})`);
    const composite = (ids: number[], status: string) =>
      ids.length === 1 ? ids[0] : add({ type: 102, form: 0, status, params: [I(ids.length), ...ids.map(REF)] });
    let bptr = -1, cptr = -1;
    if (has2d) {
      const ids = loop.map((c) => add(curveEntity(c.curve2d!, true, STATUS_DEPENDENT_PARAM2D, tol)));
      bptr = composite(ids, STATUS_DEPENDENT_PARAM2D);
    }
    if (c3.length) {
      for (const c of c3) track(c.controlPoints);
      const ids = c3.map((c) => add(curveEntity(c, false, STATUS_DEPENDENT, tol)));
      cptr = composite(ids, STATUS_DEPENDENT);
    }
    const pref = has2d ? 1 : 2;
    return add({
      type: 142,
      form: 0,
      status: STATUS_DEPENDENT,
      params: [I(1), REF(surfIdx), bptr >= 0 ? REF(bptr) : I(0), cptr >= 0 ? REF(cptr) : I(0), I(pref)],
    });
  };

  model.surfaces.forEach((item, si) => {
    track(item.surface.controlPoints);
    const label = item.name ? item.name.replace(/[^\x20-\x7e]/g, "").slice(0, 8) : undefined;
    if (!item.trims) {
      const e = surfaceEntity(item.surface, STATUS_INDEPENDENT, tol);
      add({ ...e, label, subscript: si + 1 });
      return;
    }
    const sIdx = add(surfaceEntity(item.surface, STATUS_DEPENDENT, tol));
    const outer = addLoop(item.trims.outer, sIdx, `surface ${si} outer`);
    const inner = item.trims.inner.map((l, k) => addLoop(l, sIdx, `surface ${si} inner ${k}`));
    add({
      type: 144,
      form: 0,
      status: STATUS_INDEPENDENT,
      params: [REF(sIdx), I(1), I(inner.length), REF(outer), ...inner.map(REF)],
      label,
      subscript: si + 1,
    });
  });
  for (const c of model.curves ?? []) {
    track(c.controlPoints);
    add(curveEntity(c, false, STATUS_INDEPENDENT, tol));
  }

  // --- Start section
  const startText = model.description ?? "Resurface scan-to-CAD export (IGES 5.3, NURBS trimmed surfaces).";
  const startLines: string[] = [];
  for (let i = 0; i < startText.length || i === 0; i += 72) startLines.push(startText.slice(i, i + 72));

  // --- Global section
  const date = timestamp(model.date ?? new Date());
  const clean = (s: string | undefined, max = 60) => (s ?? "").replace(/[^\x20-\x7e]/g, "").slice(0, max);
  const productId = clean(model.productId) || "Resurface model";
  const globalTokens: string[] = [
    hollerith(","),
    hollerith(";"),
    hollerith(productId),
    hollerith(clean(model.fileName) || "model.igs"),
    hollerith("Resurface"),
    hollerith("Resurface IGES writer 1.0"),
    "32", "38", "6", "308", "15",
    hollerith(productId),
    formatReal(1),
    "2",
    hollerith("MM"),
    "1",
    formatReal(1),
    hollerith(date),
    formatReal(tol),
    formatReal(Math.max(1, maxCoord)),
    clean(model.author) ? hollerith(clean(model.author)) : "",
    clean(model.organization) ? hollerith(clean(model.organization)) : "",
    "11",
    "0",
    hollerith(date),
  ];
  const globalLines = packTokens(globalTokens, 72);

  // --- Parameter data (P) — computed before D so that pointers/line counts are known
  const dePointer = (idx: number) => 2 * idx + 1;
  const pLines: string[] = [];
  const pStart: number[] = [];
  const pCount: number[] = [];
  entities.forEach((e, idx) => {
    const texts = [String(e.type), ...e.params.map((t) => tokenText(t, dePointer))];
    const lines = packTokens(texts, 64);
    pStart.push(pLines.length + 1);
    pCount.push(lines.length);
    for (const l of lines) {
      const seq = pLines.length + 1;
      pLines.push(`${l.padEnd(64, " ")} ${String(dePointer(idx)).padStart(7, " ")}${seqField("P", seq)}`);
    }
  });

  // --- Directory entries
  const dLines: string[] = [];
  entities.forEach((e, idx) => {
    const seq1 = 2 * idx + 1;
    const l1 = [int8(e.type), int8(pStart[idx]), int8(0), int8(0), int8(0), int8(0), int8(0), int8(0), e.status].join("");
    const l2 = [
      int8(e.type), int8(0), int8(0), int8(pCount[idx]), int8(e.form),
      " ".repeat(8), " ".repeat(8),
      (e.label ?? "").padStart(8, " "),
      int8(e.subscript ?? 0),
    ].join("");
    dLines.push(`${l1}${seqField("D", seq1)}`, `${l2}${seqField("D", seq1 + 1)}`);
  });

  const out: string[] = [];
  startLines.forEach((l, i) => out.push(`${l.padEnd(72, " ")}${seqField("S", i + 1)}`));
  globalLines.forEach((l, i) => out.push(`${l.padEnd(72, " ")}${seqField("G", i + 1)}`));
  out.push(...dLines, ...pLines);
  const term = `${seqField("S", startLines.length)}${seqField("G", globalLines.length)}${seqField("D", dLines.length)}${seqField("P", pLines.length)}`;
  out.push(`${term.padEnd(72, " ")}${seqField("T", 1)}`);
  return out.join("\n") + "\n";
}
