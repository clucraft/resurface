import { describe, expect, it } from "vitest";
import type { MeshData } from "../core/types";
import { vertexCurvature } from "./curvature";
import { countSelected, growSelection, invertSelection, selectByNormal, selectConnected, shrinkSelection, smartSelect } from "./select";
import { buildTriangleNeighbors } from "./topology";
import { addNoise, makeCylinder, makeExtrudedProfile, makeGrid, makeIcosphere } from "./util/generate";

function merge(a: MeshData, b: MeshData): MeshData {
  const nva = a.positions.length / 3;
  const positions = new Float32Array(a.positions.length + b.positions.length);
  positions.set(a.positions);
  positions.set(b.positions, a.positions.length);
  const indices = new Uint32Array(a.indices.length + b.indices.length);
  indices.set(a.indices);
  for (let i = 0; i < b.indices.length; i++) indices[a.indices.length + i] = b.indices[i] + nva;
  return { positions, indices };
}

/** plane (x in [-20,0], z=0) → concave fillet r=5 → wall (x=5, z in [5,25]); swept 20 along Y. */
function filletPart(): MeshData {
  const prof: [number, number][] = [];
  for (let i = 0; i <= 40; i++) prof.push([-20 + i * 0.5, 0]);
  for (let i = 1; i <= 16; i++) {
    const th = -Math.PI / 2 + (i / 16) * (Math.PI / 2);
    prof.push([5 * Math.cos(th), 5 + 5 * Math.sin(th)]);
  }
  for (let i = 1; i <= 40; i++) prof.push([5, 5 + i * 0.5]);
  return makeExtrudedProfile(prof, 20, 40);
}

function region(m: MeshData, t: number): "plane" | "fillet" | "wall" {
  const P = m.positions, I = m.indices;
  let x = 0, z = 0;
  for (let k = 0; k < 3; k++) {
    x += P[3 * I[3 * t + k]] / 3;
    z += P[3 * I[3 * t + k] + 2] / 3;
  }
  if (z < 1e-6) return "plane";
  if (x > 5 - 1e-6) return "wall";
  return "fillet";
}

function tally(m: MeshData, mask: Uint8Array) {
  const sel = { plane: 0, fillet: 0, wall: 0 };
  const tot = { plane: 0, fillet: 0, wall: 0 };
  for (let t = 0; t < mask.length; t++) {
    const r = region(m, t);
    tot[r]++;
    if (mask[t]) sel[r]++;
  }
  return { sel, tot };
}

function findTri(m: MeshData, x: number, y: number, z: number): number {
  let best = 0, bd = Infinity;
  const P = m.positions, I = m.indices;
  for (let t = 0; t < I.length / 3; t++) {
    let cx = 0, cy = 0, cz = 0;
    for (let k = 0; k < 3; k++) {
      cx += P[3 * I[3 * t + k]] / 3;
      cy += P[3 * I[3 * t + k] + 1] / 3;
      cz += P[3 * I[3 * t + k] + 2] / 3;
    }
    const d = (cx - x) ** 2 + (cy - y) ** 2 + (cz - z) ** 2;
    if (d < bd) {
      bd = d;
      best = t;
    }
  }
  return best;
}

describe("basic selection ops", () => {
  it("grow / shrink / invert", () => {
    const g = makeGrid(10, 10);
    const mask = new Uint8Array(200);
    const t0 = 2 * (5 * 10 + 5); // a cell in the middle
    mask[t0] = 1;
    const g1 = growSelection(g, mask, 1);
    expect(countSelected(g1)).toBe(13); // triangles touching the 3 vertices of t0
    const g2 = growSelection(g, mask, 2);
    expect(countSelected(g2)).toBeGreaterThan(countSelected(g1));
    expect(countSelected(shrinkSelection(g, g2, 2))).toBeLessThanOrEqual(countSelected(g1));
    expect(countSelected(shrinkSelection(g, g1, 1))).toBe(1);
    expect(countSelected(invertSelection(g1))).toBe(200 - 13);
  });

  it("selectConnected floods one shell (edge and vertex modes)", () => {
    const m = merge(makeIcosphere(1, 2), makeGrid(3, 3));
    const a = selectConnected(m, [0]);
    expect(countSelected(a)).toBe(320);
    const b = selectConnected(m, [330], { via: "vertex" });
    expect(countSelected(b)).toBe(18);
    const both = selectConnected(m, [0, 330], { neighbors: buildTriangleNeighbors(m) });
    expect(countSelected(both)).toBe(338);
  });

  it("selectByNormal", () => {
    const c = makeCylinder(5, 10, 36, 4);
    const sel = selectByNormal(c, [1, 0, 0], 20);
    const n = countSelected(sel);
    // ~ 40/360 of the triangles
    expect(n).toBeGreaterThan(20);
    expect(n).toBeLessThan(40);
  });
});

describe("curvature", () => {
  it("sphere ≈ 1/R, cylinder mean ≈ 1/(2R), plane = 0", () => {
    const s = makeIcosphere(8, 4);
    const ks = vertexCurvature(s);
    let avg = 0;
    for (const k of ks) avg += k / ks.length;
    expect(avg).toBeCloseTo(1 / 8, 2);
    const c = makeCylinder(4, 20, 64, 40);
    const kc = vertexCurvature(c, { kind: "signedMean" });
    // interior vertices only (boundary rows have one-sided rings)
    let sum = 0, n = 0;
    for (let v = 64; v < kc.length - 64; v++) {
      sum += kc[v];
      n++;
    }
    expect(sum / n).toBeCloseTo(1 / 8, 2);
    const g = makeGrid(5, 5);
    for (const k of vertexCurvature(g, { kind: "rms" })) expect(k).toBe(0);
  });
});

describe("smartSelect", () => {
  it("stays on the plane / fillet / wall of a clean part", () => {
    const m = filletPart();
    const nbr = buildTriangleNeighbors(m);
    const onPlane = smartSelect(m, findTri(m, -10, 10, 0), { neighbors: nbr });
    const tp = tally(m, onPlane);
    console.log("clean plane seed", tp);
    expect(tp.sel.plane).toBe(tp.tot.plane);
    expect(tp.sel.fillet).toBeLessThanOrEqual(2 * 40); // at most one transition ring
    expect(tp.sel.wall).toBe(0);

    const onFillet = smartSelect(m, findTri(m, 5 * Math.cos(-Math.PI / 4), 10, 5 + 5 * Math.sin(-Math.PI / 4)), { neighbors: nbr });
    const tf = tally(m, onFillet);
    expect(tf.sel.fillet).toBeGreaterThan(0.85 * tf.tot.fillet);
    expect(tf.sel.plane).toBe(0);
    expect(tf.sel.wall).toBe(0);

    const onWall = smartSelect(m, findTri(m, 5, 10, 15), { neighbors: nbr });
    const tw = tally(m, onWall);
    expect(tw.sel.wall).toBe(tw.tot.wall);
    expect(tw.sel.plane).toBe(0);
    expect(tw.sel.fillet).toBeLessThanOrEqual(2 * 40);
  });

  it("is robust to scan noise", () => {
    const clean = filletPart();
    const m = addNoise(filletPart(), 0.01, 5);
    const nbr = buildTriangleNeighbors(m);
    const onPlane = smartSelect(m, findTri(clean, -10, 10, 0), { neighbors: nbr });
    const tp = tally(clean, onPlane);
    expect(tp.sel.plane).toBeGreaterThan(0.95 * tp.tot.plane);
    expect(tp.sel.fillet).toBeLessThan(0.25 * tp.tot.fillet);
    expect(tp.sel.wall).toBe(0);
    const onFillet = smartSelect(m, findTri(clean, 5 * Math.cos(-Math.PI / 4), 10, 5 + 5 * Math.sin(-Math.PI / 4)), { neighbors: nbr });
    const tf = tally(clean, onFillet);
    console.log("noisy plane seed", tp, "fillet seed", tf);
    expect(tf.sel.fillet).toBeGreaterThan(0.7 * tf.tot.fillet);
    expect(tf.sel.plane).toBeLessThan(0.05 * tp.tot.plane);
    expect(tf.sel.wall).toBeLessThan(0.05 * tf.tot.wall);
  });

  it("grows over a whole cylinder", () => {
    const c = makeCylinder(10, 30, 72, 30);
    const sel = smartSelect(c, 100);
    expect(countSelected(sel)).toBe(c.indices.length / 3);
  });
});
