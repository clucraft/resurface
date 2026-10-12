import { describe, expect, it } from "vitest";
import { ByteScanner, parseNumberString, toBytes } from "./text";

const enc = (s: string) => new TextEncoder().encode(s);

describe("byte number parser", () => {
  it("parses common forms", () => {
    expect(parseNumberString("1")).toBe(1);
    expect(parseNumberString("-1.5")).toBe(-1.5);
    expect(parseNumberString("+.5")).toBe(0.5);
    expect(parseNumberString("5.")).toBe(5);
    expect(parseNumberString("1e3")).toBe(1000);
    expect(parseNumberString("1.5E-3")).toBe(0.0015);
    expect(parseNumberString("2.5e+2")).toBe(250);
    expect(parseNumberString("0.1")).toBe(0.1);
    expect(parseNumberString("3.14159265358979")).toBe(3.14159265358979);
    expect(parseNumberString("-1.234567e+05")).toBe(-123456.7);
    expect(parseNumberString("000123.4500")).toBe(123.45);
    expect(Object.is(parseNumberString("-0"), -0)).toBe(true);
    expect(Object.is(parseNumberString("-0.0e5"), -0)).toBe(true);
    expect(parseNumberString("1e")).toBe(1);
  });

  it("handles special and extreme values", () => {
    expect(parseNumberString("nan")).toBeNaN();
    expect(parseNumberString("NaN")).toBeNaN();
    expect(parseNumberString("-nan(ind)")).toBeNaN();
    expect(parseNumberString("inf")).toBe(Infinity);
    expect(parseNumberString("-Infinity")).toBe(-Infinity);
    expect(parseNumberString("1.#INF")).toBe(Infinity);
    expect(parseNumberString("-1.#INF")).toBe(-Infinity);
    expect(parseNumberString("-1.#IND")).toBeNaN();
    expect(parseNumberString("1.#QNAN")).toBeNaN();
    expect(parseNumberString("1e400")).toBe(Infinity);
    expect(parseNumberString("1e-400")).toBe(0);
    expect(parseNumberString("4.9e-324")).toBe(5e-324);
    const tiny = parseNumberString("2.2250738585072014e-308");
    expect(Math.abs(tiny - 2.2250738585072014e-308)).toBeLessThan(1e-322);
  });

  it("is exact (== parseFloat) for <=15 significant digits with small exponents", () => {
    let seed = 12345;
    const rnd = () => {
      seed = (Math.imul(seed, 1103515245) + 12345) >>> 0;
      return seed / 4294967296;
    };
    for (let i = 0; i < 20000; i++) {
      const mag = Math.pow(10, Math.floor(rnd() * 10) - 5);
      const v = (rnd() - 0.5) * mag;
      const p = 1 + Math.floor(rnd() * 15);
      const s = v.toPrecision(p);
      expect(parseNumberString(s)).toBe(parseFloat(s));
    }
  });

  it("is within a few ulp (and float32-identical) for long mantissas and large exponents", () => {
    let seed = 777;
    const rnd = () => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return seed / 4294967296;
    };
    for (let i = 0; i < 20000; i++) {
      // full 53-bit mantissas (dyadic test values would sit on float32 rounding midpoints)
      const v = Math.sin(i * 12.9898 + rnd()) * Math.pow(10, Math.floor(rnd() * 120) - 60);
      for (const s of [v.toPrecision(17), v.toExponential(19), String(v)]) {
        const ref = parseFloat(s);
        const got = parseNumberString(s);
        // mantissas beyond ~16 digits are rounded: a few ulp at most
        expect(Math.abs(got - ref)).toBeLessThanOrEqual(Math.abs(ref) * 1.2e-15);
        // what the parsers store (float32) agrees
        expect(Math.fround(got)).toBe(Math.fround(ref));
      }
    }
    expect(parseNumberString("123456789012345678901234567890")).toBeCloseTo(1.2345678901234568e29, -15);
  });
});

describe("ByteScanner", () => {
  it("reads line-oriented numbers and reports line ends", () => {
    const s = new ByteScanner(enc("  1 2.5\t-3  \r\n4 x5 6\n\n7"));
    expect(s.readFloat()).toBe(1);
    expect(s.readFloat()).toBe(2.5);
    expect(s.readFloat()).toBe(-3);
    s.readFloat();
    expect(s.ok).toBe(false);
    expect(s.atLineEnd()).toBe(true);
    s.nextLine();
    expect(s.readFloat()).toBe(4);
    expect(s.readFloat()).toBeNaN(); // "x5"
    expect(s.ok).toBe(true);
    expect(s.readFloat()).toBe(6);
    s.nextLine();
    expect(s.readFloatAny()).toBe(7);
    expect(s.eof()).toBe(true);
    expect(s.readFloatAny()).toBeNaN();
    expect(s.ok).toBe(false);
  });

  it("reads OBJ-style ints and keywords", () => {
    const s = new ByteScanner(enc("FACET normal\n-12/7//3"));
    expect(s.matchWord("facet")).toBe(true);
    s.skipBlanks();
    expect(s.matchWord("norm")).toBe(false);
    expect(s.matchWord("normal")).toBe(true);
    s.skipWhitespace();
    expect(s.readIntRaw()).toBe(-12);
    expect(s.peek()).toBe(47);
    s.pos++;
    expect(s.readIntRaw()).toBe(7);
    s.pos++;
    s.readIntRaw();
    expect(s.ok).toBe(false);
    s.pos++;
    expect(s.readIntRaw()).toBe(3);
  });

  it("toBytes respects views and strings", () => {
    const big = new Uint8Array([9, 9, 1, 2, 3, 9]);
    const view = new DataView(big.buffer, 2, 3);
    expect(Array.from(toBytes(view))).toEqual([1, 2, 3]);
    expect(toBytes(big.subarray(1, 3)).byteOffset).toBe(1);
    expect(Array.from(toBytes("é"))).toEqual([0xc3, 0xa9]);
  });
});
