import { describe, expect, it } from "vitest";
import type { MeshData } from "../../core/types";
import { parsePTX } from "./ptx";

interface ScanSpec {
  cols: number;
  rows: number;
  /** scanner registered position (header line 3) */
  pos?: [number, number, number];
  /** 4 rows of 4 (row-vector convention, translation in the last row) */
  matrix?: number[][];
  /** local point for (c, r); null = missing */
  point: (c: number, r: number) => [number, number, number] | null;
  intensity?: (c: number, r: number) => number;
  rgb?: (c: number, r: number) => [number, number, number];
}

function ptxScan(s: ScanSpec, nl = "\n"): string {
  const m = s.matrix ?? [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ];
  const p = s.pos ?? [m[3][0], m[3][1], m[3][2]];
  const lines: string[] = [String(s.cols), String(s.rows), p.join(" ")];
  for (let i = 0; i < 3; i++) lines.push(m[i].slice(0, 3).join(" "));
  for (let i = 0; i < 4; i++) lines.push(m[i].join(" "));
  for (let c = 0; c < s.cols; c++)
    for (let r = 0; r < s.rows; r++) {
      const q = s.point(c, r);
      const it = s.intensity ? s.intensity(c, r) : 0.5;
      const vals = q ? [...q.map((x) => x.toFixed(6)), String(it)] : ["0", "0", "0", "0"];
      if (s.rgb) vals.push(...(q ? s.rgb(c, r) : [0, 0, 0]).map(String));
      lines.push(vals.join(" "));
    }
  return lines.join(nl) + nl;
}

/** Triangle normal · (scanner − centroid) for every triangle of the given range. */
function facing(m: MeshData, t0: number, t1: number, s: [number, number, number]): number[] {
  const P = m.positions, I = m.indices;
  const out: number[] = [];
  for (let t = t0; t < t1; t++) {
    const a = 3 * I[3 * t], b = 3 * I[3 * t + 1], c = 3 * I[3 * t + 2];
    const u = [P[b] - P[a], P[b + 1] - P[a + 1], P[b + 2] - P[a + 2]];
    const v = [P[c] - P[a], P[c + 1] - P[a + 1], P[c + 2] - P[a + 2]];
    const n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    const cen = [0, 1, 2].map((k) => (P[a + k] + P[b + k] + P[c + k]) / 3);
    out.push(n[0] * (s[0] - cen[0]) + n[1] * (s[1] - cen[1]) + n[2] * (s[2] - cen[2]));
  }
  return out;
}

// Scan A: plane x = 10 in front of the scanner (identity transform), intensity only, one
// missing interior point.
const scanA: ScanSpec = {
  cols: 6,
  rows: 5,
  point: (c, r) => (c === 2 && r === 2 ? null : [10, c * 0.1, r * 0.1]),
  intensity: (c, r) => -2048 + 100 * (c * 5 + r),
};

// Scan B: depth step between columns 2 and 3 (x = 10 → x = 20, same angular spacing),
// rotated 90° about Z and translated, with RGB; corner (0,0) missing.
const rotZ90T = [
  [0, 1, 0, 0],
  [-1, 0, 0, 0],
  [0, 0, 1, 0],
  [100, 0, 5, 1],
];
const scanB: ScanSpec = {
  cols: 6,
  rows: 5,
  matrix: rotZ90T,
  point: (c, r) => (c === 0 && r === 0 ? null : c < 3 ? [10, c * 0.1, r * 0.1] : [20, c * 0.2, r * 0.2]),
  intensity: () => 0,
  rgb: (c, r) => [c * 40, r * 50, 7],
};

describe("parsePTX", () => {
  it("parses two concatenated scans with transform, missing points, RGB and a depth cut", () => {
    const text = ptxScan(scanA) + ptxScan(scanB, "\r\n");
    const m = parsePTX(text, { intensityToGray: true });
    // vertices: 29 + 29 valid points
    expect(m.positions.length / 3).toBe(58);
    // scan A: 16 full cells × 2 + 4 three-corner cells; scan B: cell column 2 is cut,
    // 4 cell columns × 4 rows × 2 − 1 (three-corner cell at the missing corner)
    expect(m.indices.length / 3).toBe(36 + 31);

    // first vertex of scan A is (0,0) untransformed
    expect(Array.from(m.positions.subarray(0, 3))).toEqual([10, 0, 0]);
    // first vertex of scan B is local (10, 0, 0.1) → world (100 - 0, 10, 0.1 + 5)
    const b0 = Array.from(m.positions.subarray(29 * 3, 29 * 3 + 3));
    expect(b0[0]).toBeCloseTo(100, 5);
    expect(b0[1]).toBeCloseTo(10, 5);
    expect(b0[2]).toBeCloseTo(5.1, 5);

    // the depth discontinuity is not bridged: in world space the near plane is y = 10, far y = 20
    for (let t = 36; t < 67; t++) {
      const ys = [0, 1, 2].map((k) => m.positions[3 * m.indices[3 * t + k] + 1]);
      const near = ys.filter((y) => Math.abs(y - 10) < 1e-3).length;
      expect(near === 0 || near === 3).toBe(true);
    }
    // normals face the scanner in both scans
    expect(facing(m, 0, 36, [0, 0, 0]).every((d) => d > 0)).toBe(true);
    expect(facing(m, 36, 67, [100, 0, 5]).every((d) => d > 0)).toBe(true);

    // colours: scan B RGB as given, scan A grey from intensity
    const C = m.colors!;
    expect(C.length).toBe(58 * 3);
    expect(Array.from(C.subarray(29 * 3, 29 * 3 + 3))).toEqual([0, 50, 7]); // B (0,1)
    expect(Array.from(C.subarray(3 * 57, 3 * 58))).toEqual([200, 200, 7]); // B (5,4)
    for (let v = 0; v < 29; v++) {
      expect(C[3 * v]).toBe(C[3 * v + 1]);
      expect(C[3 * v]).toBe(C[3 * v + 2]);
    }
    expect(C[0]).toBe(0); // lowest intensity
    expect(C[3 * 28]).toBe(255); // highest intensity
  });

  it("bridges the discontinuity only with a very large edge factor", () => {
    const m = parsePTX(ptxScan(scanB), { maxEdgeFactor: 1000 });
    expect(m.indices.length / 3).toBe(39);
    const strict = parsePTX(ptxScan(scanB));
    expect(strict.indices.length / 3).toBe(31);
    expect(strict.colors!.length).toBe(29 * 3);
  });

  it("maps intensity to grey only when asked, and handles a points-only scan", () => {
    const scan: ScanSpec = { cols: 2, rows: 2, point: (c, r) => [5, c, r], intensity: (c, r) => (c * 2 + r) / 3 };
    const plain = parsePTX(ptxScan(scan));
    expect(plain.colors).toBeUndefined();
    const grey = parsePTX(ptxScan(scan), { intensityToGray: true });
    expect(Array.from(grey.colors!)).toEqual([0, 0, 0, 85, 85, 85, 170, 170, 170, 255, 255, 255]);
    // single row: no cells → no triangles, but the points are there
    const line = parsePTX(ptxScan({ cols: 4, rows: 1, point: (c) => [5, c, 0] }));
    expect(line.positions.length / 3).toBe(4);
    expect(line.indices.length).toBe(0);
  });

  it("uses the header position as scanner when the matrix is identity", () => {
    // points stored in registered coordinates on plane x = 40, scanner at x = 50
    const scan: ScanSpec = { cols: 4, rows: 4, pos: [50, 0, 0], point: (c, r) => [40, c * 0.1, r * 0.1] };
    const m = parsePTX(ptxScan(scan));
    expect(m.indices.length / 3).toBe(18);
    expect(facing(m, 0, 18, [50, 0, 0]).every((d) => d > 0)).toBe(true);
  });

  it("keeps the complete part of a truncated file and rejects garbage", () => {
    const text = ptxScan(scanA);
    const cut = text.slice(0, text.length - 40);
    const m = parsePTX(new TextEncoder().encode(cut));
    expect(m.positions.length / 3).toBeLessThan(29);
    expect(m.positions.length / 3).toBeGreaterThan(20);
    expect(m.indices.length / 3).toBeGreaterThan(20);
    expect(() => parsePTX("hello world")).toThrow();
    // absurd grid sizes: huge → error; large but short data → no giant allocation
    expect(() => parsePTX("100000\n100000\n0 0 0\n")).toThrow();
    const big = parsePTX("40000\n40000\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n1 2 3 0.5\n");
    expect(big.positions.length).toBe(3);
    expect(parsePTX("").positions.length).toBe(0);
  });
});
