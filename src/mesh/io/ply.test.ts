import { describe, expect, it } from "vitest";
import { parsePLY, parsePLYHeader, type PLYResult } from "./ply";

type Fmt = "ascii" | "binary_little_endian" | "binary_big_endian";
interface PropSpec {
  name: string;
  type: string;
  /** list count type (makes this a list property) */
  list?: string;
}
interface ElemSpec {
  name: string;
  props: PropSpec[];
  /** one row per record; list values are given as arrays */
  rows: (number | number[])[][];
}

const SIZE: Record<string, number> = {
  char: 1, int8: 1, uchar: 1, uint8: 1, short: 2, int16: 2, ushort: 2, uint16: 2,
  int: 4, int32: 4, uint: 4, uint32: 4, float: 4, float32: 4, double: 8, float64: 8,
};

function writeScalar(dv: DataView, off: number, type: string, v: number, le: boolean): number {
  switch (type) {
    case "char": case "int8": dv.setInt8(off, v); break;
    case "uchar": case "uint8": dv.setUint8(off, v); break;
    case "short": case "int16": dv.setInt16(off, v, le); break;
    case "ushort": case "uint16": dv.setUint16(off, v, le); break;
    case "int": case "int32": dv.setInt32(off, v, le); break;
    case "uint": case "uint32": dv.setUint32(off, v, le); break;
    case "float": case "float32": dv.setFloat32(off, v, le); break;
    case "double": case "float64": dv.setFloat64(off, v, le); break;
    default: throw new Error(type);
  }
  return off + SIZE[type];
}

/** Minimal PLY writer for tests (all formats, any element/property layout). */
function buildPLY(fmt: Fmt, elements: ElemSpec[], opts: { crlf?: boolean; comments?: boolean } = {}): Uint8Array {
  const nl = opts.crlf ? "\r\n" : "\n";
  let h = `ply${nl}format ${fmt} 1.0${nl}`;
  if (opts.comments) h += `comment made by a test${nl}obj_info generated${nl}`;
  for (const e of elements) {
    h += `element ${e.name} ${e.rows.length}${nl}`;
    for (const p of e.props) h += p.list ? `property list ${p.list} ${p.type} ${p.name}${nl}` : `property ${p.type} ${p.name}${nl}`;
  }
  h += `end_header${nl}`;
  const head = new TextEncoder().encode(h);
  if (fmt === "ascii") {
    let body = "";
    for (const e of elements)
      for (const r of e.rows)
        body += r.map((v) => (Array.isArray(v) ? [v.length, ...v].join(" ") : String(v))).join(" ") + nl;
    const b = new TextEncoder().encode(body);
    const out = new Uint8Array(head.length + b.length);
    out.set(head);
    out.set(b, head.length);
    return out;
  }
  const le = fmt === "binary_little_endian";
  let size = 0;
  for (const e of elements)
    for (const r of e.rows)
      e.props.forEach((p, j) => {
        const v = r[j];
        size += p.list ? SIZE[p.list] + (v as number[]).length * SIZE[p.type] : SIZE[p.type];
      });
  const out = new Uint8Array(head.length + size);
  out.set(head);
  const dv = new DataView(out.buffer);
  let off = head.length;
  for (const e of elements)
    for (const r of e.rows)
      e.props.forEach((p, j) => {
        const v = r[j];
        if (p.list) {
          const arr = v as number[];
          off = writeScalar(dv, off, p.list, arr.length, le);
          for (const x of arr) off = writeScalar(dv, off, p.type, x, le);
        } else off = writeScalar(dv, off, p.type, v as number, le);
      });
  return out;
}

const FORMATS: Fmt[] = ["ascii", "binary_little_endian", "binary_big_endian"];

/** Square (2 triangles) + extra props, lists on vertices, an edge element and a material element. */
function richElements(): ElemSpec[] {
  return [
    {
      name: "vertex",
      props: [
        { name: "x", type: "double" },
        { name: "y", type: "double" },
        { name: "z", type: "double" },
        { name: "quality", type: "float" },
        { name: "nx", type: "float" },
        { name: "ny", type: "float" },
        { name: "nz", type: "float" },
        { name: "tags", type: "int", list: "uchar" },
        { name: "red", type: "uchar" },
        { name: "green", type: "uchar" },
        { name: "blue", type: "uchar" },
        { name: "alpha", type: "uchar" },
      ],
      rows: [
        [0, 0, 0.125, 1.5, 0, 0, 2, [1, 2, 3], 255, 0, 0, 255],
        [1, 0, 0.125, 2.5, 0, 0, 1, [], 0, 255, 0, 255],
        [1, 1, 0.125, 3.5, 0, 0, 1, [7], 0, 0, 255, 255],
        [0, 1, 0.125, 4.5, 0, 0, 1, [8, 9], 10, 20, 30, 255],
      ],
    },
    {
      name: "edge",
      props: [
        { name: "vertex1", type: "int" },
        { name: "vertex2", type: "int" },
      ],
      rows: [
        [0, 1],
        [1, 2],
      ],
    },
    {
      name: "face",
      props: [
        { name: "flags", type: "int" },
        { name: "vertex_indices", type: "int", list: "uchar" },
        { name: "texcoord", type: "float", list: "uchar" },
      ],
      rows: [
        [7, [0, 1, 2, 3], [0, 0, 1, 0, 1, 1, 0, 1]],
        [8, [0, 2, 9], []], // out of range → dropped
      ],
    },
    {
      name: "material",
      props: [
        { name: "name_chars", type: "uchar", list: "int" },
        { name: "shininess", type: "float" },
      ],
      rows: [[[65, 66], 0.5]],
    },
  ];
}

function mesh(r: PLYResult) {
  expect(r.kind).toBe("mesh");
  if (r.kind !== "mesh") throw new Error("not a mesh");
  return r.mesh;
}

describe("parsePLY", () => {
  it("parses the same rich file identically in ascii, binary LE and binary BE", () => {
    for (const fmt of FORMATS) {
      for (const crlf of [false, true]) {
        const m = mesh(parsePLY(buildPLY(fmt, richElements(), { crlf, comments: true })));
        expect(Array.from(m.positions)).toEqual([0, 0, 0.125, 1, 0, 0.125, 1, 1, 0.125, 0, 1, 0.125]);
        expect(Array.from(m.indices)).toEqual([0, 1, 2, 0, 2, 3]);
        expect(Array.from(m.normals!)).toEqual([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]);
        expect(Array.from(m.colors!)).toEqual([255, 0, 0, 0, 255, 0, 0, 0, 255, 10, 20, 30]);
      }
    }
  });

  it("parses the header", () => {
    const h = parsePLYHeader(buildPLY("binary_big_endian", richElements(), { crlf: true, comments: true }));
    expect(h.format).toBe("binary_big_endian");
    expect(h.elements.map((e) => [e.name, e.count, e.properties.length])).toEqual([
      ["vertex", 4, 12],
      ["edge", 2, 2],
      ["face", 2, 3],
      ["material", 1, 2],
    ]);
    expect(h.elements[0].properties[7]).toMatchObject({ name: "tags", isList: true });
    expect(() => parsePLYHeader(new TextEncoder().encode("ply\nformat ascii 1.0\nelement vertex 1\n"))).toThrow();
    expect(() => parsePLYHeader(new TextEncoder().encode("plx\n"))).toThrow();
    expect(() => parsePLYHeader(new TextEncoder().encode("ply\nformat ascii 1.0\nproperty foo x\nend_header\n"))).toThrow();
  });

  it("handles float colours (0..1 and 0..255), int16 colours and r/g/b names", () => {
    const pts = (type: string, names: string[], vals: number[][]): ElemSpec[] => [
      {
        name: "vertex",
        props: [
          { name: "x", type: "float" },
          { name: "y", type: "float" },
          { name: "z", type: "float" },
          ...names.map((n) => ({ name: n, type })),
        ],
        rows: vals.map((v, i) => [i, 0, 0, ...v]),
      },
    ];
    for (const fmt of FORMATS) {
      const unit = parsePLY(buildPLY(fmt, pts("float", ["red", "green", "blue"], [[1, 0, 0.5], [0.2, 0.4, 1]])));
      expect(unit.kind).toBe("points");
      if (unit.kind === "points") expect(Array.from(unit.points.colors!)).toEqual([255, 0, 128, 51, 102, 255]);
      const full = parsePLY(buildPLY(fmt, pts("double", ["diffuse_red", "diffuse_green", "diffuse_blue"], [[255, 0, 128], [1, 2, 3]])));
      if (full.kind === "points") expect(Array.from(full.points.colors!)).toEqual([255, 0, 128, 1, 2, 3]);
      const shorts = parsePLY(buildPLY(fmt, pts("short", ["r", "g", "b"], [[300, -5, 128], [1, 2, 3]])));
      if (shorts.kind === "points") expect(Array.from(shorts.points.colors!)).toEqual([255, 0, 128, 1, 2, 3]);
    }
  });

  it("supports uint8/int32/uint32 list counts and indices, quads and faces before vertices", () => {
    for (const fmt of FORMATS) {
      for (const [ct, it] of [["uint8", "uint32"], ["int32", "int32"], ["uchar", "ushort"], ["int", "uchar"]]) {
        const els: ElemSpec[] = [
          {
            name: "face",
            props: [{ name: "vertex_index", type: it, list: ct }],
            rows: [[[0, 1, 2, 3]], [[1, 4, 2]]],
          },
          {
            name: "vertex",
            props: [
              { name: "x", type: "int" },
              { name: "y", type: "short" },
              { name: "z", type: "uchar" },
            ],
            rows: [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0], [20, 5, 1]],
          },
        ];
        const m = mesh(parsePLY(buildPLY(fmt, els)));
        expect(m.positions.length / 3).toBe(5);
        expect(Array.from(m.indices)).toEqual([0, 1, 2, 0, 2, 3, 1, 4, 2]);
        expect(m.positions[12]).toBe(20);
        expect(m.normals).toBeUndefined();
        expect(m.colors).toBeUndefined();
      }
    }
  });

  it("returns a point cloud when there are no faces", () => {
    for (const fmt of FORMATS) {
      const r = parsePLY(
        buildPLY(fmt, [
          {
            name: "vertex",
            props: [
              { name: "x", type: "float" },
              { name: "y", type: "float" },
              { name: "z", type: "float" },
              { name: "nx", type: "float" },
              { name: "ny", type: "float" },
              { name: "nz", type: "float" },
            ],
            rows: [[1, 2, 3, 0, 3, 0], [4, 5, 6, 1, 0, 0]],
          },
          { name: "face", props: [{ name: "vertex_indices", type: "int", list: "uchar" }], rows: [] },
        ]),
      );
      expect(r.kind).toBe("points");
      if (r.kind === "points") {
        expect(Array.from(r.points.positions)).toEqual([1, 2, 3, 4, 5, 6]);
        expect(Array.from(r.points.normals!)).toEqual([0, 1, 0, 1, 0, 0]);
      }
    }
  });

  it("reads triangle strips", () => {
    for (const fmt of FORMATS) {
      const m = mesh(
        parsePLY(
          buildPLY(fmt, [
            {
              name: "vertex",
              props: ["x", "y", "z"].map((name) => ({ name, type: "float" })),
              rows: [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [2, 0, 0], [2, 1, 0]],
            },
            { name: "tristrips", props: [{ name: "vertex_indices", type: "int", list: "int" }], rows: [[[0, 1, 2, 3, -1, 2, 3, 4, 5]]] },
          ]),
        ),
      );
      expect(Array.from(m.indices)).toEqual([0, 1, 2, 2, 1, 3, 2, 3, 4, 4, 3, 5]);
    }
  });

  it("keeps what is complete in truncated binary files", () => {
    const els: ElemSpec[] = [
      { name: "vertex", props: ["x", "y", "z"].map((name) => ({ name, type: "float" })), rows: [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]] },
      { name: "face", props: [{ name: "vertex_indices", type: "int", list: "uchar" }], rows: [[[0, 1, 2]], [[0, 2, 3]], [[1, 2, 3]]] },
    ];
    for (const fmt of ["binary_little_endian", "binary_big_endian"] as Fmt[]) {
      const full = buildPLY(fmt, els);
      const m = mesh(parsePLY(full.subarray(0, full.length - 5)));
      expect(Array.from(m.indices)).toEqual([0, 1, 2, 0, 2, 3]);
      // truncated inside the vertex element → point cloud of the complete vertices
      const headLen = parsePLYHeader(full).dataOffset;
      const r = parsePLY(full.subarray(0, headLen + 12 * 2 + 5));
      expect(r.kind).toBe("points");
      if (r.kind === "points") expect(r.points.positions.length).toBe(6);
    }
    const ascii = buildPLY("ascii", els);
    const cut = mesh(parsePLY(ascii.subarray(0, ascii.length - 8)));
    expect(Array.from(cut.indices)).toEqual([0, 1, 2, 0, 2, 3]);
  });

  it("accepts unaligned and shared views", () => {
    const bytes = buildPLY("binary_big_endian", richElements());
    const sab = new SharedArrayBuffer(bytes.length + 3);
    const view = new Uint8Array(sab, 3);
    view.set(bytes);
    const m = mesh(parsePLY(view));
    expect(Array.from(m.indices)).toEqual([0, 1, 2, 0, 2, 3]);
  });
});
