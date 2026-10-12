import { describe, expect, it } from "vitest";
import { fileExtension, importMeshFile } from "./import";
import { writeSTLBinary } from "./stl";

const buf = (s: string): ArrayBuffer => new TextEncoder().encode(s).buffer as ArrayBuffer;

describe("importMeshFile", () => {
  it("dispatches on the lower-cased extension", () => {
    const stl = writeSTLBinary({
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      indices: new Uint32Array([0, 1, 2]),
    });
    const r1 = importMeshFile("C:\\scans\\Part.STL", stl.buffer as ArrayBuffer);
    expect(r1.kind).toBe("mesh");
    if (r1.kind === "mesh") expect(r1.mesh.indices.length).toBe(3);

    const r2 = importMeshFile("a.obj", buf("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"));
    expect(r2.kind).toBe("mesh");
    const r3 = importMeshFile("cloud.Obj", buf("v 0 0 0\nv 1 0 0\n"));
    expect(r3.kind).toBe("points");
    if (r3.kind === "points") expect(r3.points.positions.length).toBe(6);

    const r4 = importMeshFile("x.ply", buf("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n1 2 3\n"));
    expect(r4.kind).toBe("points");

    const ptx = "2\n2\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n5 0 0 0.1\n5 0 1 0.2\n5 1 0 0.3\n5 1 1 0.4\n";
    const r5 = importMeshFile("scan.ptx", buf(ptx));
    expect(r5.kind).toBe("mesh");
    if (r5.kind === "mesh") {
      expect(r5.mesh.indices.length).toBe(6);
      expect(r5.mesh.colors!.length).toBe(12); // intensity → grey
    }
    const r6 = importMeshFile("line.PTX", buf("3\n1\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n5 0 0 0.1\n5 1 0 0.2\n5 2 0 0.3\n"));
    expect(r6.kind).toBe("points");
  });

  it("throws a helpful error for unknown extensions", () => {
    expect(() => importMeshFile("model.3mf", new ArrayBuffer(4))).toThrow(/unsupported file type "\.3mf".*\.stl/);
    expect(() => importMeshFile("README", new ArrayBuffer(4))).toThrow(/no extension/);
  });

  it("extracts extensions", () => {
    expect(fileExtension("a/b.c/Scan.PLY")).toBe(".ply");
    expect(fileExtension("noext")).toBe("");
    expect(fileExtension("dir.v2\\file")).toBe("");
  });
});
