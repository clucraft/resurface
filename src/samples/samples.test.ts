import { describe, expect, it } from "vitest";
import { surfaceNets } from "./surfaceNets";
import { sphere } from "./sdf";
import { makeSample } from "./samples";

function signedVolume(pos: Float32Array, idx: Uint32Array): number {
  let v = 0;
  for (let t = 0; t < idx.length; t += 3) {
    const a = idx[t] * 3, b = idx[t + 1] * 3, c = idx[t + 2] * 3;
    v +=
      (pos[a] * (pos[b + 1] * pos[c + 2] - pos[b + 2] * pos[c + 1]) -
        pos[a + 1] * (pos[b] * pos[c + 2] - pos[b + 2] * pos[c]) +
        pos[a + 2] * (pos[b] * pos[c + 1] - pos[b + 1] * pos[c])) /
      6;
  }
  return v;
}

describe("surface nets", () => {
  it("meshes a sphere with outward orientation and correct volume", () => {
    const r = 10;
    const m = surfaceNets(sphere(0, 0, 0, r), [-r, -r, -r], [r, r, r], 0.5);
    const vol = signedVolume(m.positions, m.indices);
    const exact = (4 / 3) * Math.PI * r ** 3;
    expect(vol).toBeGreaterThan(0);
    expect(Math.abs(vol - exact) / exact).toBeLessThan(0.01);
    // vertices lie on the sphere
    let maxErr = 0;
    for (let i = 0; i < m.positions.length; i += 3) maxErr = Math.max(maxErr, Math.abs(Math.hypot(m.positions[i], m.positions[i + 1], m.positions[i + 2]) - r));
    expect(maxErr).toBeLessThan(0.01);
  });

  it("generates samples", () => {
    for (const id of ["bracket", "turned", "organic"] as const) {
      const t0 = performance.now();
      const m = makeSample(id, 0.5);
      expect(m.indices.length).toBeGreaterThan(3000);
      expect(signedVolume(m.positions, m.indices)).toBeGreaterThan(0);
      console.log(id, m.indices.length / 3, "tris", (performance.now() - t0).toFixed(0), "ms");
    }
  });
});
