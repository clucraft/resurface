import { describe, expect, it } from "vitest";
import type { Vec3 } from "../core/types";
import { cross, dot, length, normalize, sub } from "../core/vec";
import {
  limitMesh,
  limitMeshWithMap,
  limitNormals,
  limitPoints,
  subdivide,
  subdivideLevels,
  subdivisionStencils,
} from "./catmullClark";
import { buildTopology, createBoxCage, createCylinderCage, createGridCage, setCrease, validate, vertexRing, type QuadMesh } from "./quadMesh";
import { csrMulVec3 } from "./sparse";

const dist = (a: Vec3, b: Vec3) => length(sub(a, b));

/** A "bumpy" open grid: z displaced so the surface is genuinely curved. */
function bumpyGrid(n = 4): QuadMesh {
  const g = createGridCage(n, n, { origin: [0, 0, 0], uAxis: [40, 0, 0], vAxis: [0, 40, 0] });
  g.vertices = g.vertices.map(([x, y]) => [x + 0.7 * Math.sin(y), y, 6 * Math.sin(x / 9) * Math.cos(y / 11)] as Vec3);
  return g;
}

/** Cage with a triangle and a pentagon. */
function mixedCage(): QuadMesh {
  const g = createGridCage(2, 1, { uAxis: [20, 0, 0], vAxis: [0, 10, 0] });
  const apex = g.vertices.length;
  g.vertices.push([5, 18, 3]);
  const p1 = g.vertices.length;
  g.vertices.push([16, 20, -2], [22, 14, 1]);
  g.faces.push([3, 4, apex]); // triangle on the left quad's top edge (which runs 4 -> 3)
  g.faces.push([4, 5, p1 + 1, p1, apex]); // pentagon
  return g;
}

describe("Catmull–Clark subdivision", () => {
  it("one level on a box: V+E+F vertices, 4F quads, still a valid closed manifold", () => {
    const box = createBoxCage([0, 0, 0], [10, 10, 10], 2);
    const topo = buildTopology(box);
    const s = subdivide(box);
    expect(s.vertices.length).toBe(topo.nv + topo.ne + topo.nf);
    expect(s.faces.length).toBe(4 * box.faces.length);
    const r = validate(s);
    expect(r.ok && r.closed && r.eulerCharacteristic === 2).toBe(true);
  });

  it("supports triangles and n-gons (all-quad output, valid)", () => {
    const m = mixedCage();
    expect(validate(m).ok).toBe(true);
    const s = subdivide(m);
    expect(s.faces.every((f) => f.length === 4)).toBe(true);
    expect(s.faces.length).toBe(4 + 4 + 3 + 5);
    expect(validate(s).ok).toBe(true);
  });

  it("boundary rules: boundary curve = cubic B-spline of the boundary polygon, corners interpolated", () => {
    const g = bumpyGrid(4);
    const L = limitPoints(g);
    const topo = buildTopology(g);
    // corner vertex 0 is interpolated
    expect(dist(L[0], g.vertices[0])).toBeLessThan(1e-12);
    // boundary vertex 2 on the bottom row: (P1 + 4 P2 + P3) / 6
    const exp: Vec3 = [0, 1, 2].map((c) => (g.vertices[1][c] + 4 * g.vertices[2][c] + g.vertices[3][c]) / 6) as Vec3;
    expect(dist(L[2], exp)).toBeLessThan(1e-12);
    // subdividing never moves the boundary curve off the B-spline (vertex points of the bottom row)
    const s = subdivide(g);
    const expV: Vec3 = [0, 1, 2].map((c) => (g.vertices[1][c] + 6 * g.vertices[2][c] + g.vertices[3][c]) / 8) as Vec3;
    expect(dist(s.vertices[2], expV)).toBeLessThan(1e-12);
    expect(topo.vBoundary[2]).toBe(1);
  });

  it("closed-form limit positions match repeated subdivision (smooth, extraordinary, boundary, crease, n-gon)", () => {
    const cases: QuadMesh[] = [];
    const box = createBoxCage([-10, -10, -10], [10, 10, 10], 1); // 8 valence-3 vertices
    box.vertices = box.vertices.map((p, i) => [p[0] + 0.3 * i, p[1] - 0.2 * (i % 3), p[2]] as Vec3);
    cases.push(box);
    cases.push(bumpyGrid(3));
    const creased = createBoxCage([-10, -10, -10], [10, 10, 10], 2);
    const topo = buildTopology(creased);
    const ce: [number, number][] = [];
    for (let e = 0; e < topo.ne; e++) {
      const a = creased.vertices[topo.edgeV[2 * e]], b = creased.vertices[topo.edgeV[2 * e + 1]];
      if (a[2] === 10 && b[2] === 10 && (Math.abs(a[0]) === 10 || Math.abs(a[1]) === 10) && (Math.abs(b[0]) === 10 || Math.abs(b[1]) === 10))
        ce.push([topo.edgeV[2 * e], topo.edgeV[2 * e + 1]]);
    }
    setCrease(creased, ce, true);
    cases.push(creased);
    cases.push(mixedCage());
    for (const qm of cases) {
      const L = limitPoints(qm);
      const fine = subdivideLevels(qm, 7);
      let err = 0, size = 0;
      for (let i = 0; i < qm.vertices.length; i++) {
        err = Math.max(err, dist(L[i], fine.vertices[i]));
        size = Math.max(size, length(qm.vertices[i]));
      }
      expect(err / size).toBeLessThan(2e-4);
    }
  });

  it("stencil rows are affine (sum to 1) and reproduce limitMesh positions", () => {
    for (const qm of [createBoxCage([0, 0, 0], [5, 6, 7], [2, 1, 3]), mixedCage(), bumpyGrid(3)]) {
      for (const levels of [0, 1, 2]) {
        const st = subdivisionStencils(qm, levels);
        for (let r = 0; r < st.rows.nRows; r++) {
          let s = 0;
          for (let p = st.rows.rowPtr[r]; p < st.rows.rowPtr[r + 1]; p++) s += st.rows.values[p];
          expect(Math.abs(s - 1)).toBeLessThan(1e-12);
        }
        const Q = csrMulVec3(st.rows, qm.vertices);
        const lm = limitMeshWithMap(qm, levels, { splitCreases: false });
        expect(lm.mesh.positions.length / 3).toBe(Q.length);
        let err = 0;
        for (let i = 0; i < Q.length; i++)
          err = Math.max(err, dist(Q[i], [lm.mesh.positions[3 * i], lm.mesh.positions[3 * i + 1], lm.mesh.positions[3 * i + 2]]));
        expect(err).toBeLessThan(1e-4); // Float32 output
        // samples 0..nv-1 are the cage-vertex limit points
        const L = limitPoints(qm);
        for (let i = 0; i < qm.vertices.length; i++) expect(dist(L[i], Q[i])).toBeLessThan(1e-9);
        // every sample has a face and uv in [0,1]
        for (let i = 0; i < Q.length; i++) {
          expect(st.faceOfSample[i]).toBeGreaterThanOrEqual(0);
          expect(st.uvOfSample[2 * i]).toBeGreaterThanOrEqual(0);
          expect(st.uvOfSample[2 * i]).toBeLessThanOrEqual(1);
        }
      }
    }
  });

  it("sample uv: corner samples of quad faces sit at the expected parameters", () => {
    const g = createGridCage(2, 2);
    const st = subdivisionStencils(g, 2);
    // cage vertex 0 belongs to face 0 at (0,0); vertex 4 (centre) to face 0 at (1,1)
    expect(st.faceOfSample[0]).toBe(0);
    expect([st.uvOfSample[0], st.uvOfSample[1]]).toEqual([0, 0]);
    expect(st.faceOfSample[4]).toBe(0);
    expect([st.uvOfSample[8], st.uvOfSample[9]]).toEqual([1, 1]);
  });

  it("limit normals: planar grid -> +z (incl. boundary/corners); box -> outward; cylinder -> radial", () => {
    const g = createGridCage(3, 3);
    for (const n of limitNormals(g)) expect(dot(n, [0, 0, 1])).toBeGreaterThan(1 - 1e-12);
    const box = createBoxCage([-1, -1, -1], [1, 1, 1], 2);
    const N = limitNormals(box);
    const L = limitPoints(box);
    N.forEach((n, i) => expect(dot(n, normalize(L[i]))).toBeGreaterThan(0.5));
    // symmetric vertices: face centre of +x side -> +x
    const ix = box.vertices.findIndex((p) => p[0] === 1 && p[1] === 0 && p[2] === 0);
    expect(dot(N[ix], [1, 0, 0])).toBeGreaterThan(1 - 1e-12);
    const cyl = createCylinderCage(10, 30, 8, 3);
    const Nc = limitNormals(cyl);
    cyl.vertices.forEach((p, i) => {
      const radial = normalize([p[0], p[1], 0]);
      expect(dot(Nc[i], radial)).toBeGreaterThan(1 - 1e-9);
    });
  });

  it("limit normals agree with the geometry of the refined limit surface (EVs, boundary)", () => {
    const cases = [createBoxCage([-10, -10, -10], [10, 10, 10], 1), bumpyGrid(3), mixedCage()];
    for (const qm of cases) {
      const N = limitNormals(qm);
      const levels = 5;
      const st = subdivisionStencils(qm, levels);
      const Q = csrMulVec3(st.rows, qm.vertices);
      // ring of the vertex at the fine level -> chords of the limit surface around it
      const fine = subdivideLevels(qm, levels);
      const topo = buildTopology(fine);
      for (let v = 0; v < qm.vertices.length; v++) {
        const ring = vertexRing(topo, v);
        if (ring.nbr.length < 2) continue;
        let n: Vec3 = [0, 0, 0];
        const m = ring.closed ? ring.nbr.length : ring.nbr.length - 1;
        for (let i = 0; i < m; i++) {
          const a = sub(Q[ring.nbr[i]], Q[v]), b = sub(Q[ring.nbr[(i + 1) % ring.nbr.length]], Q[v]);
          n = [n[0] + cross(a, b)[0], n[1] + cross(a, b)[1], n[2] + cross(a, b)[2]];
        }
        const ang = Math.acos(Math.min(1, dot(normalize(n), N[v])));
        expect(ang).toBeLessThan((2 * Math.PI) / 180);
      }
    }
  });

  it("limitMesh: triangles, unit normals, crease vertices split into sectors", () => {
    const box = createBoxCage([0, 0, 0], [10, 10, 10], 1);
    const topo = buildTopology(box);
    const all: [number, number][] = [];
    for (let e = 0; e < topo.ne; e++) all.push([topo.edgeV[2 * e], topo.edgeV[2 * e + 1]]);
    setCrease(box, all, true); // fully creased cube -> limit surface is the cube itself
    const res = limitMeshWithMap(box, 3);
    const m = res.mesh;
    expect(m.indices.length % 3).toBe(0);
    expect(res.faceOfTriangle.length).toBe(m.indices.length / 3);
    const nv = m.positions.length / 3;
    for (let i = 0; i < nv; i++) {
      const n: Vec3 = [m.normals![3 * i], m.normals![3 * i + 1], m.normals![3 * i + 2]];
      expect(Math.abs(length(n) - 1)).toBeLessThan(1e-5);
      // every normal is axis aligned (flat faces of the cube)
      expect(Math.max(...n.map(Math.abs))).toBeGreaterThan(1 - 1e-5);
      for (let c = 0; c < 3; c++) {
        expect(m.positions[3 * i + c]).toBeGreaterThanOrEqual(-1e-5);
        expect(m.positions[3 * i + c]).toBeLessThanOrEqual(10 + 1e-5);
      }
    }
    expect(nv).toBeGreaterThan(subdivideLevels(box, 3).vertices.length); // duplicates along creases
    const smooth = limitMesh(createBoxCage([0, 0, 0], [10, 10, 10], 1), 2);
    expect(smooth.positions.length / 3).toBe(subdivideLevels(createBoxCage([0, 0, 0], [10, 10, 10], 1), 2).vertices.length);
  });
});
