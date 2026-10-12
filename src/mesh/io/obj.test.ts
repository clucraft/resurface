import { describe, expect, it } from "vitest";
import { objToImportResult, parseOBJ } from "./obj";

const enc = (s: string) => new TextEncoder().encode(s);

describe("parseOBJ", () => {
  it("parses all face forms, negative indices and polygons", () => {
    const text = [
      "# comment line",
      "mtllib foo.mtl",
      "o object",
      "v 0 0 0",
      "v 1 0 0",
      "v 1 1 0",
      "v 0 1 0",
      "v 0.5 1.5 0",
      "vt 0 0",
      "vt 1 0",
      "vt 1 1",
      "vn 0 0 1",
      "g group",
      "usemtl mat",
      "s 1",
      "f 1 2 3",
      "f 1/1 3/3 4/2",
      "f 1//1 2//1 3//1",
      "f 1/1/1 2/2/1 3/3/1 # trailing comment",
      "f -5 -4 -3 -2", // quad with relative indices = 1 2 3 4
      "f 1 2 3 5 4", // pentagon → 3 triangles
      "l 1 2",
      "p 3",
      "f 1 2", // too few vertices → ignored
    ].join("\n");
    const m = parseOBJ(text);
    expect(m.positions.length / 3).toBe(5);
    expect(Array.from(m.indices)).toEqual([
      0, 1, 2, //
      0, 2, 3, //
      0, 1, 2, //
      0, 1, 2, //
      0, 1, 2, 0, 2, 3, //
      0, 1, 2, 0, 2, 4, 0, 4, 3,
    ]);
  });

  it("handles CRLF, tabs, missing final newline, w component and junk", () => {
    const text = "v 1 2 3\r\nv\t4 5 6 1.0\r\n  v -1e1 +2.5E-1 .5\r\nvp 1 2\r\n\r\nf 1 2 3\r\nf 1 2 3 \\\r\nf 1/2/3/4 2 3";
    const m = parseOBJ(enc(text));
    expect(Array.from(m.positions)).toEqual([1, 2, 3, 4, 5, 6, -10, 0.25, 0.5]);
    expect(m.colors).toBeUndefined();
    expect(Array.from(m.indices)).toEqual([0, 1, 2, 0, 1, 2, 0, 1, 2]);
  });

  it("drops faces with out-of-range indices but keeps degenerate ones", () => {
    const text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 4\nf 0 1 2\nf -4 1 2\nf 1 1 2\nf 1 2 3\n";
    const m = parseOBJ(text);
    expect(Array.from(m.indices)).toEqual([0, 0, 1, 0, 1, 2]);
  });

  it("reads vertex colours in 0..1 and 0..255", () => {
    const unit = parseOBJ("v 0 0 0 1 0 0.5\nv 1 0 0 0 1 0\nv 0 1 0 0.2 0.4 1\nf 1 2 3\n");
    expect(Array.from(unit.colors!)).toEqual([255, 0, 128, 0, 255, 0, 51, 102, 255]);
    const bytes255 = parseOBJ("v 0 0 0 255 0 128\nv 1 0 0 0 1 0\nv 0 1 0 10 20 30\nf 1 2 3\n");
    expect(Array.from(bytes255.colors!)).toEqual([255, 0, 128, 0, 1, 0, 10, 20, 30]);
    // only some vertices coloured: the others get white
    const partial = parseOBJ("v 0 0 0\nv 1 0 0 1 0 0\nv 0 1 0\nf 1 2 3\n");
    expect(Array.from(partial.colors!)).toEqual([255, 255, 255, 255, 0, 0, 255, 255, 255]);
  });

  it("averages referenced vertex normals per position vertex", () => {
    const text = [
      "v 0 0 0",
      "v 1 0 0",
      "v 0 1 0",
      "v 1 1 0",
      "vn 0 0 1",
      "vn 1 0 0",
      "vn 0 0 2",
      "f 1//1 2//1 3//1",
      "f 2//2 4//3 3//2",
    ].join("\n");
    const m = parseOBJ(text);
    expect(m.positions.length / 3).toBe(4); // not split per normal
    const n = Array.from(m.normals!).map((x) => Math.round(x * 1e6) / 1e6);
    const s = Math.SQRT1_2;
    const r = (x: number) => Math.round(x * 1e6) / 1e6;
    expect(n).toEqual([0, 0, 1, r(s), 0, r(s), r(s), 0, r(s), 0, 0, 1]);
    // a face vertex without any normal → normals omitted
    const partial = parseOBJ("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nvn 0 0 1\nf 1//1 2//1 3//1\nf 2 4 3\n");
    expect(partial.normals).toBeUndefined();
    // unreferenced vertex gets (0,0,0) but does not prevent normals
    const unref = parseOBJ("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 5 5 5\nvn 0 0 1\nf 1//1 2//1 3//1\n");
    expect(Array.from(unref.normals!)).toEqual([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]);
  });

  it("returns a point cloud (no indices) for OBJ without faces, with per-index normals", () => {
    const m = parseOBJ("v 0 0 0\nv 1 0 0\nvn 0 0 3\nvn 0 1 0\n");
    expect(m.indices.length).toBe(0);
    expect(Array.from(m.normals!)).toEqual([0, 0, 1, 0, 1, 0]);
    const r = objToImportResult(m);
    expect(r.kind).toBe("points");
    if (r.kind === "points") {
      expect(r.points.positions.length).toBe(6);
      expect(r.points.normals!.length).toBe(6);
    }
  });

  it("accepts string, ArrayBuffer and Uint8Array views identically", () => {
    const text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
    const a = parseOBJ(text);
    const big = new Uint8Array(text.length + 4);
    big.set(enc(text), 2);
    const b = parseOBJ(big.subarray(2, 2 + text.length));
    const c = parseOBJ(enc(text).buffer as ArrayBuffer);
    for (const m of [b, c]) {
      expect(Array.from(m.positions)).toEqual(Array.from(a.positions));
      expect(Array.from(m.indices)).toEqual(Array.from(a.indices));
    }
  });

  it("skips a UTF-8 BOM", () => {
    const m = parseOBJ("﻿v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n");
    expect(m.positions.length / 3).toBe(3);
    expect(m.indices.length).toBe(3);
  });

  it("handles empty input", () => {
    const m = parseOBJ("");
    expect(m.positions.length).toBe(0);
    expect(m.indices.length).toBe(0);
  });
});
