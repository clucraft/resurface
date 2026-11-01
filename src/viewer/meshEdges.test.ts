import { describe, expect, it } from "vitest";
import { uniqueTriangleEdges } from "./meshEdges";

function edgeSet(pairs: Uint32Array): Set<string> {
  const s = new Set<string>();
  for (let i = 0; i < pairs.length; i += 2) {
    expect(pairs[i]).toBeLessThan(pairs[i + 1]);
    s.add(`${pairs[i]}-${pairs[i + 1]}`);
  }
  expect(s.size).toBe(pairs.length / 2);
  return s;
}

describe("uniqueTriangleEdges", () => {
  it("returns each shared edge once (quad = 2 triangles -> 5 edges)", () => {
    const e = edgeSet(uniqueTriangleEdges(new Uint32Array([0, 1, 2, 0, 2, 3]), 4));
    expect([...e].sort()).toEqual(["0-1", "0-2", "0-3", "1-2", "2-3"]);
  });

  it("handles a closed tetrahedron and inconsistent winding", () => {
    // faces with mixed orientation still yield the 6 edges exactly once
    const tet = new Uint32Array([0, 1, 2, 0, 3, 1, 1, 3, 2, 2, 0, 3]);
    expect(edgeSet(uniqueTriangleEdges(tet, 4)).size).toBe(6);
  });

  it("drops degenerate edges and keeps boundary edges", () => {
    const e = edgeSet(uniqueTriangleEdges(new Uint32Array([0, 1, 1, 1, 2, 3]), 4));
    expect([...e].sort()).toEqual(["0-1", "1-2", "1-3", "2-3"]);
  });

  it("de-duplicates high-valence fans (sorted bucket path)", () => {
    const rim = 100;
    const idx: number[] = [];
    for (let i = 0; i < rim; i++) idx.push(0, 1 + i, 1 + ((i + 1) % rim));
    const e = edgeSet(uniqueTriangleEdges(new Uint32Array(idx), rim + 1));
    expect(e.size).toBe(2 * rim); // spokes + rim
  });
});
