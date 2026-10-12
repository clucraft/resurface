/**
 * Scale smoke test: a ~2M-triangle binary STL parsed in memory and streamed in 1 MB chunks.
 * Timings are logged so they can be extrapolated to 20M+ triangle scans.
 */
import { describe, expect, it } from "vitest";
import type { MeshData } from "../../core/types";
import { parseOBJ } from "./obj";
import { parsePLY } from "./ply";
import { createSTLStreamParser, parseSTL, writeSTLBinary } from "./stl";

function gridMesh(n: number): MeshData {
  const nv = (n + 1) * (n + 1);
  const positions = new Float32Array(nv * 3);
  for (let j = 0, k = 0; j <= n; j++)
    for (let i = 0; i <= n; i++, k += 3) {
      positions[k] = i * 0.25;
      positions[k + 1] = j * 0.25;
      positions[k + 2] = Math.sin(i * 0.05) * Math.cos(j * 0.07) * 3;
    }
  const indices = new Uint32Array(n * n * 6);
  for (let j = 0, k = 0; j < n; j++)
    for (let i = 0; i < n; i++, k += 6) {
      const a = j * (n + 1) + i, b = a + 1, c = a + n + 1, d = c + 1;
      indices[k] = a;
      indices[k + 1] = b;
      indices[k + 2] = d;
      indices[k + 3] = a;
      indices[k + 4] = d;
      indices[k + 5] = c;
    }
  return { positions, indices };
}

const ms = (t0: number) => Math.round(performance.now() - t0);

describe("scale smoke test", () => {
  it("parses a ~2M triangle binary STL (in memory and streamed)", () => {
    const n = 999; // 1000 x 1000 vertices, 1,996,002 triangles
    const mesh = gridMesh(n);
    const nt = n * n * 2;

    let t0 = performance.now();
    const bytes = writeSTLBinary(mesh);
    const tWrite = ms(t0);
    expect(bytes.length).toBe(84 + 50 * nt);

    t0 = performance.now();
    const m = parseSTL(bytes);
    const tParse = ms(t0);
    expect(m.indices.length / 3).toBe(nt);
    expect(m.positions.length / 3).toBe((n + 1) * (n + 1));

    t0 = performance.now();
    const sp = createSTLStreamParser({ totalBytes: bytes.length });
    const CH = 1 << 20;
    for (let o = 0; o < bytes.length; o += CH) sp.push(bytes.subarray(o, Math.min(bytes.length, o + CH)));
    const ms2 = sp.finish();
    const tStream = ms(t0);
    expect(ms2.indices.length).toBe(m.indices.length);
    expect(ms2.positions.length).toBe(m.positions.length);
    expect(ms2.indices[ms2.indices.length - 1]).toBe(m.indices[m.indices.length - 1]);

    const per = (t: number) => `${t} ms (~${Math.round((t * 20e6) / nt / 100) / 10} s per 20M tris)`;
    console.log(
      `[scale] binary STL ${nt} tris / ${(bytes.length / 1048576).toFixed(0)} MB: write ${per(tWrite)}, ` +
        `parseSTL ${per(tParse)}, stream(1MB chunks) ${per(tStream)}`,
    );
  });

  it("parses ~2M triangle OBJ and binary PLY", () => {
    const n = 999;
    const mesh = gridMesh(n);
    const nv = mesh.positions.length / 3, nt = mesh.indices.length / 3;

    // OBJ text (generated in pieces to keep the generator itself reasonable)
    const parts: string[] = [];
    const P = mesh.positions, I = mesh.indices;
    for (let v = 0; v < nv; v++) parts.push(`v ${P[3 * v].toFixed(4)} ${P[3 * v + 1].toFixed(4)} ${P[3 * v + 2].toFixed(6)}`);
    for (let t = 0; t < nt; t++) parts.push(`f ${I[3 * t] + 1} ${I[3 * t + 1] + 1} ${I[3 * t + 2] + 1}`);
    const objBytes = new TextEncoder().encode(parts.join("\n"));
    parts.length = 0;
    let t0 = performance.now();
    const om = parseOBJ(objBytes);
    const tObj = ms(t0);
    expect(om.indices.length / 3).toBe(nt);
    expect(om.positions.length / 3).toBe(nv);

    // binary little-endian PLY: float xyz + uchar/int face lists
    const header = new TextEncoder().encode(
      `ply\nformat binary_little_endian 1.0\nelement vertex ${nv}\nproperty float x\nproperty float y\nproperty float z\n` +
        `element face ${nt}\nproperty list uchar int vertex_indices\nend_header\n`,
    );
    const ply = new Uint8Array(header.length + nv * 12 + nt * 13);
    ply.set(header);
    const dv = new DataView(ply.buffer);
    let o = header.length;
    for (let k = 0; k < nv * 3; k++, o += 4) dv.setFloat32(o, P[k], true);
    for (let t = 0; t < nt; t++) {
      dv.setUint8(o, 3);
      dv.setInt32(o + 1, I[3 * t], true);
      dv.setInt32(o + 5, I[3 * t + 1], true);
      dv.setInt32(o + 9, I[3 * t + 2], true);
      o += 13;
    }
    t0 = performance.now();
    const pr = parsePLY(ply);
    const tPly = ms(t0);
    expect(pr.kind).toBe("mesh");
    if (pr.kind === "mesh") expect(pr.mesh.indices.length / 3).toBe(nt);

    const per = (t: number) => `${t} ms (~${Math.round((t * 20e6) / nt / 100) / 10} s per 20M tris)`;
    console.log(
      `[scale] OBJ ${(objBytes.length / 1048576).toFixed(0)} MB: ${per(tObj)}; ` +
        `binary PLY ${(ply.length / 1048576).toFixed(0)} MB: ${per(tPly)}`,
    );
  });
});
