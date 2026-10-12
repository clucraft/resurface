import { describe, expect, it } from "vitest";
import type { MeshData } from "../../core/types";
import { createSTLStreamParser, detectSTLFormat, parseSTL, writeSTLBinary } from "./stl";

const enc = (s: string) => new TextEncoder().encode(s);

/** Unit cube: 8 vertices, 12 outward CCW triangles. */
function cube(): MeshData {
  const positions = new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1]);
  const indices = new Uint32Array([
    0, 2, 1, 0, 3, 2, // bottom (z=0, normal -z)
    4, 5, 6, 4, 6, 7, // top
    0, 1, 5, 0, 5, 4, // y=0
    1, 2, 6, 1, 6, 5, // x=1
    2, 3, 7, 2, 7, 6, // y=1
    3, 0, 4, 3, 4, 7, // x=0
  ]);
  return { positions, indices };
}

/** Triangle corner positions as nested arrays, for winding-preserving comparison. */
function triangleCorners(m: MeshData): number[][] {
  const out: number[][] = [];
  for (let t = 0; t < m.indices.length / 3; t++) {
    const tri: number[] = [];
    for (let k = 0; k < 3; k++) {
      const v = m.indices[3 * t + k];
      tri.push(m.positions[3 * v], m.positions[3 * v + 1], m.positions[3 * v + 2]);
    }
    out.push(tri);
  }
  return out;
}

function headerString(bytes: Uint8Array): string {
  return String.fromCharCode(...bytes.subarray(0, 80));
}

/** Build a raw binary STL (no normals) with the given header text and triangles. */
function rawBinarySTL(header: string, tris: number[][], countOverride?: number, trailing = 0): Uint8Array {
  const out = new Uint8Array(84 + 50 * tris.length + trailing);
  for (let i = 0; i < Math.min(80, header.length); i++) out[i] = header.charCodeAt(i);
  const dv = new DataView(out.buffer);
  dv.setUint32(80, countOverride ?? tris.length, true);
  tris.forEach((t, i) => {
    for (let k = 0; k < 9; k++) dv.setFloat32(84 + 50 * i + 12 + 4 * k, t[k], true);
  });
  return out;
}

function expectSameMesh(a: MeshData, b: MeshData) {
  expect(Array.from(a.positions)).toEqual(Array.from(b.positions));
  expect(Array.from(a.indices)).toEqual(Array.from(b.indices));
}

describe("binary STL", () => {
  it("roundtrips a cube through writeSTLBinary/parseSTL with welding", () => {
    const c = cube();
    const bytes = writeSTLBinary(c);
    expect(bytes.length).toBe(84 + 50 * 12);
    expect(headerString(bytes).startsWith("Resurface")).toBe(true);
    expect(detectSTLFormat(bytes, bytes.length)).toBe("binary");
    const m = parseSTL(bytes);
    expect(m.positions.length / 3).toBe(8);
    expect(m.indices.length / 3).toBe(12);
    expect(m.normals).toBeUndefined();
    expect(triangleCorners(m)).toEqual(triangleCorners(c));
    // also from an ArrayBuffer
    const m2 = parseSTL(bytes.slice().buffer);
    expectSameMesh(m, m2);
  });

  it("writes unit facet normals (zero for degenerate triangles)", () => {
    const mesh: MeshData = {
      positions: new Float32Array([0, 0, 0, 2, 0, 0, 0, 3, 0, 4, 0, 0]),
      indices: new Uint32Array([0, 1, 2, 0, 1, 3]),
    };
    const bytes = writeSTLBinary(mesh);
    const dv = new DataView(bytes.buffer);
    expect([dv.getFloat32(84, true), dv.getFloat32(88, true), dv.getFloat32(92, true)]).toEqual([0, 0, 1]);
    expect([dv.getFloat32(134, true), dv.getFloat32(138, true), dv.getFloat32(142, true)]).toEqual([0, 0, 0]);
    expect(dv.getUint16(84 + 48, true)).toBe(0);
  });

  it("never writes a header starting with 'solid'", () => {
    const bytes = writeSTLBinary(cube(), "solid part exported");
    const h = headerString(bytes);
    expect(/^\s*solid/i.test(h)).toBe(false);
    expect(h).toContain("solid part exported");
    expect(parseSTL(bytes).indices.length).toBe(36);
    const long = writeSTLBinary(cube(), "x".repeat(200));
    expect(long.length).toBe(84 + 600);
  });

  it("reads a binary file whose header starts with 'solid'", () => {
    const c = cube();
    const tris = triangleCorners(c);
    const bytes = rawBinarySTL("solid cube exported by some CAD tool", tris);
    expect(detectSTLFormat(bytes.subarray(0, 4096), bytes.length)).toBe("binary");
    const m = parseSTL(bytes);
    expect(m.positions.length / 3).toBe(8);
    expect(triangleCorners(m)).toEqual(tris);
    // size test fails (trailing bytes) → content heuristic must still say binary
    const withTrailing = rawBinarySTL("solid cube", tris, undefined, 17);
    expect(detectSTLFormat(withTrailing.subarray(0, 4096))).toBe("binary");
    expect(triangleCorners(parseSTL(withTrailing))).toEqual(tris);
  });

  it("parses what is complete in a truncated file and handles bad counts", () => {
    const full = writeSTLBinary(cube());
    const cut = full.slice(0, 84 + 50 * 5 + 23);
    const m = parseSTL(cut);
    expect(m.indices.length / 3).toBe(5);
    expect(triangleCorners(m)).toEqual(triangleCorners(cube()).slice(0, 5));
    // header count 0 → all complete records
    const zero = full.slice();
    new DataView(zero.buffer).setUint32(80, 0, true);
    expect(parseSTL(zero).indices.length / 3).toBe(12);
    // header count smaller than the data → honoured
    const small = full.slice();
    new DataView(small.buffer).setUint32(80, 3, true);
    expect(parseSTL(small).indices.length / 3).toBe(3);
    // garbage huge count → clamped to the data
    const huge = full.slice();
    new DataView(huge.buffer).setUint32(80, 0xffffffff, true);
    expect(parseSTL(huge).indices.length / 3).toBe(12);
  });

  it("handles empty input and header-only files", () => {
    for (const b of [new Uint8Array(0), new Uint8Array(10), rawBinarySTL("hdr", [])]) {
      const m = parseSTL(b);
      expect(m.positions.length).toBe(0);
      expect(m.indices.length).toBe(0);
    }
    expect(parseSTL(enc("solid empty\nendsolid empty\n")).indices.length).toBe(0);
  });

  it("accepts unaligned views and shared buffers", () => {
    const bytes = writeSTLBinary(cube());
    const ref = parseSTL(bytes);
    const big = new Uint8Array(bytes.length + 7);
    big.set(bytes, 3);
    expectSameMesh(parseSTL(big.subarray(3, 3 + bytes.length)), ref);
    const sab = new SharedArrayBuffer(bytes.length + 5);
    const shared = new Uint8Array(sab, 5);
    shared.set(bytes);
    expectSameMesh(parseSTL(shared), ref);
  });
});

describe("ASCII STL", () => {
  const ascii =
    "solid test part\r\n" +
    "  facet normal 0 0 -1\r\n" +
    "    outer loop\r\n" +
    "      vertex 0 0 0\r\n" +
    "      vertex 1.0e+00 0 0\r\n" +
    "      vertex -0.0 1E0 -0\r\n" +
    "    endloop\r\n" +
    "  endfacet\r\n" +
    "  FACET NORMAL 0 0 1\r\n" +
    "    OUTER LOOP\r\n" +
    "      VERTEX 1 0 0   \r\n" +
    "      VERTEX -2.5e-1 -1.25E+1 3.0\r\n" +
    "      VERTEX 0.0 1.0 0.0\r\n" +
    "    ENDLOOP\r\n" +
    "  ENDFACET\r\n" +
    "endsolid test part"; // no final newline

  it("parses CRLF, scientific notation, negatives and welds -0 with 0", () => {
    const m = parseSTL(enc(ascii));
    expect(m.indices.length / 3).toBe(2);
    expect(m.positions.length / 3).toBe(4);
    expect(Array.from(m.positions)).toEqual([0, 0, 0, 1, 0, 0, 0, 1, 0, -0.25, -12.5, 3]);
    expect(Array.from(m.indices)).toEqual([0, 1, 2, 1, 3, 2]);
    expect(m.normals).toBeUndefined();
  });

  it("handles multiple solids, a solid name containing keywords, polygon facets and a BOM", () => {
    const text =
      "﻿  solid vertex facet name\n" +
      "facet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 1 1 0\nvertex 0 1 0\nendloop\nendfacet\n" +
      "endsolid vertex facet name\n" +
      "solid second\nfacet normal 0 0 1 outer loop vertex 0 0 1 vertex 1 0 1 vertex 1 1 1 endloop endfacet\nendsolid\n";
    const m = parseSTL(enc(text));
    expect(m.indices.length / 3).toBe(3);
    expect(m.positions.length / 3).toBe(7);
  });

  it("detects ASCII only when the content looks textual", () => {
    expect(detectSTLFormat(enc(ascii), enc(ascii).length)).toBe("ascii");
    expect(detectSTLFormat(enc("solid x\nfoo bar"), 15)).toBe("binary");
    const junk = enc("solid x\nfacet\n");
    const withNul = new Uint8Array(junk.length + 1);
    withNul.set(junk);
    expect(detectSTLFormat(withNul, withNul.length)).toBe("binary");
  });
});

describe("STL stream parser", () => {
  const feed = (bytes: Uint8Array, chunk: number, totalBytes?: number) => {
    const p = createSTLStreamParser({ totalBytes });
    for (let o = 0; o < bytes.length; o += chunk) p.push(bytes.subarray(o, Math.min(bytes.length, o + chunk)));
    return p.finish();
  };

  /** A bumpy grid big enough that chunk boundaries fall everywhere. */
  function grid(n: number): MeshData {
    const positions = new Float32Array((n + 1) * (n + 1) * 3);
    for (let j = 0, k = 0; j <= n; j++)
      for (let i = 0; i <= n; i++, k += 3) {
        positions[k] = i * 0.37;
        positions[k + 1] = j * 0.41;
        positions[k + 2] = Math.sin(i * 0.3) * Math.cos(j * 0.2);
      }
    const indices = new Uint32Array(n * n * 6);
    for (let j = 0, k = 0; j < n; j++)
      for (let i = 0; i < n; i++, k += 6) {
        const a = j * (n + 1) + i, b = a + 1, c = a + n + 1, d = c + 1;
        indices.set([a, b, d, a, d, c], k);
      }
    return { positions, indices };
  }

  it("gives identical results to parseSTL for odd chunk sizes", () => {
    for (const bytes of [writeSTLBinary(grid(12)), rawBinarySTL("solid but binary", triangleCorners(grid(5)))]) {
      const ref = parseSTL(bytes);
      for (const chunk of [1, 7, 49, 50, 51, 1000]) {
        expectSameMesh(feed(bytes, chunk), ref);
        expectSameMesh(feed(bytes, chunk, bytes.length), ref);
      }
    }
  });

  it("does not retain caller chunks (buffers may be reused)", () => {
    const bytes = writeSTLBinary(grid(6));
    const ref = parseSTL(bytes);
    const p = createSTLStreamParser();
    const scratch = new Uint8Array(37);
    for (let o = 0; o < bytes.length; o += 37) {
      const n = Math.min(37, bytes.length - o);
      scratch.set(bytes.subarray(o, o + n));
      p.push(scratch.subarray(0, n));
      scratch.fill(0xab);
    }
    expectSameMesh(p.finish(), ref);
  });

  it("handles truncated input, empty input and ASCII fallback", () => {
    const bytes = writeSTLBinary(grid(4));
    const cut = bytes.subarray(0, bytes.length - 30);
    expectSameMesh(feed(cut, 13), parseSTL(cut));
    expect(feed(cut, 13).indices.length / 3).toBe(31);
    expect(createSTLStreamParser().finish().indices.length).toBe(0);
    expect(feed(bytes.subarray(0, 50), 7).indices.length).toBe(0);

    const text = enc(
      "solid a\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid a\n",
    );
    const ref = parseSTL(text);
    expect(ref.indices.length).toBe(3);
    for (const chunk of [1, 5, 4096]) {
      expectSameMesh(feed(text, chunk), ref);
      expectSameMesh(feed(text, chunk, text.length), ref);
    }
  });

  it("reports progress when the total size is known", () => {
    const bytes = writeSTLBinary(grid(10));
    const fractions: number[] = [];
    const p = createSTLStreamParser({ totalBytes: bytes.length, onProgress: (f) => fractions.push(f) });
    p.push(bytes);
    p.finish();
    expect(fractions.length).toBeGreaterThan(0);
    expect(fractions[fractions.length - 1]).toBe(1);
    expect(() => p.push(bytes)).toThrow();
  });
});
