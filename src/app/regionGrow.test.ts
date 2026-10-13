import { describe, expect, it } from "vitest";
import { makeSample } from "../samples/samples";
import { MeshSpatialIndex } from "../mesh/spatial";
import { buildTriangleNeighbors, computeFaceNormals } from "../mesh/topology";
import { smartRegion } from "./regionGrow";
import type { MeshData, Vec3 } from "../core/types";

function region(mesh: MeshData, origin: Vec3, dir: Vec3) {
  const index = new MeshSpatialIndex(mesh);
  const hit = index.raycast(origin, dir)!;
  expect(hit).toBeTruthy();
  const nbr = buildTriangleNeighbors(mesh);
  const res = smartRegion(mesh, nbr, hit.faceIndex, 12);
  const fn = computeFaceNormals(mesh);
  // fraction of selected triangles facing clearly away (> ~37°) from the seed: spill onto other faces
  let off = 0;
  const s = hit.faceIndex;
  for (let t = 0; t < res.mask.length; t++) {
    if (!res.mask[t]) continue;
    const d = fn[3 * t] * fn[3 * s] + fn[3 * t + 1] * fn[3 * s + 1] + fn[3 * t + 2] * fn[3 * s + 2];
    if (d < 0.8) off++;
  }
  return { ...res, spill: off / res.count };
}

describe("smart select (primitive region growing)", () => {
  it("selects the bracket top face as a plane without crossing the fillets", () => {
    const m = makeSample("bracket", 0.8);
    const r = region(m, [40, 10, 200], [0, 0, -1]);
    expect(r.kind).toBe("plane");
    expect(r.spill).toBeLessThan(0.002);
    expect(r.count).toBeGreaterThan(5000);
  });

  it("stays on a narrow, gently curved grille slat front face", () => {
    const m = makeSample("grille", 0.8);
    // undo nothing: aim along the tilted scan's depth axis at the middle of a slat (scan coordinates)
    const ax = (8 * Math.PI) / 180, az = (-6 * Math.PI) / 180;
    const toScan = (x: number, y: number, z: number): Vec3 => {
      const y1 = y * Math.cos(ax) - z * Math.sin(ax), z1 = y * Math.sin(ax) + z * Math.cos(ax);
      return [x * Math.cos(az) - y1 * Math.sin(az) + 12, x * Math.sin(az) + y1 * Math.cos(az) - 30, z1 + 8];
    };
    const o = toScan(30, -60, 63), p = toScan(30, 0, 63);
    const r = region(m, o, [p[0] - o[0], p[1] - o[1], p[2] - o[2]]);
    expect(r.kind === "plane" || r.kind === "cylinder").toBe(true);
    // never spills onto the slat top/bottom faces (90° away)
    expect(r.spill).toBeLessThan(0.002);
    expect(r.count).toBeGreaterThan(300);
  });
});
