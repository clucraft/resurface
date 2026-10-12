import { describe, expect, it } from "vitest";
import { smoothLaplacian, smoothTaubin } from "./smooth";
import { makeGrid, makeIcosphere, seededRandom } from "./util/generate";

function radialStats(P: Float32Array, R: number) {
  let sum = 0, sum2 = 0;
  const n = P.length / 3;
  for (let v = 0; v < n; v++) {
    const r = Math.hypot(P[3 * v], P[3 * v + 1], P[3 * v + 2]);
    sum += r - R;
    sum2 += (r - R) ** 2;
  }
  return { mean: sum / n, rms: Math.sqrt(sum2 / n) };
}

describe("smoothTaubin", () => {
  it("removes noise on a sphere without shrinking it", () => {
    const s = makeIcosphere(10, 4);
    const rnd = seededRandom(11);
    const n = s.positions.length / 3;
    for (let v = 0; v < n; v++) {
      const f = 1 + (rnd() * 2 - 1) * 0.01;
      for (let k = 0; k < 3; k++) s.positions[3 * v + k] *= f;
    }
    const before = radialStats(s.positions, 10);
    const out = smoothTaubin(s, { iterations: 10 });
    const after = radialStats(out.positions, 10);
    expect(after.rms).toBeLessThan(before.rms * 0.4);
    expect(Math.abs(after.mean)).toBeLessThan(0.02); // < 0.2 % shrinkage
    // plain Laplacian shrinks much more
    const lap = radialStats(smoothLaplacian(s, { iterations: 10 }).positions, 10);
    expect(lap.mean).toBeLessThan(after.mean);
  });

  it("respects the vertex mask and keeps the boundary fixed", () => {
    const g = makeGrid(10, 10, 10, 10, (x, y) => ((x * 7 + y * 3) % 2) * 0.1);
    const mask = new Uint8Array(121);
    for (let v = 0; v < 60; v++) mask[v] = 1;
    const out = smoothTaubin(g, { iterations: 5, mask });
    for (let v = 60; v < 121; v++) for (let k = 0; k < 3; k++) expect(out.positions[3 * v + k]).toBe(g.positions[3 * v + k]);
    // boundary row j = 0 is fixed although masked
    for (let i = 0; i <= 10; i++) expect(out.positions[3 * i + 2]).toBe(g.positions[3 * i + 2]);
    // something moved
    let moved = 0;
    for (let v = 0; v < 60; v++) if (out.positions[3 * v + 2] !== g.positions[3 * v + 2]) moved++;
    expect(moved).toBeGreaterThan(10);
  });
});
