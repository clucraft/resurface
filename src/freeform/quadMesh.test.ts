import { describe, expect, it } from "vitest";
import type { Vec3 } from "../core/types";
import {
  addQuad,
  addVertex,
  boundaryLoops,
  buildTopology,
  cloneQuadMesh,
  createBoxCage,
  createCylinderCage,
  createGridCage,
  createTorusCage,
  extrudeBoundaryEdges,
  faceAdjacency,
  mergeVertices,
  orient,
  removeFace,
  ringSectors,
  setCrease,
  signedVolume,
  splitEdgeLoop,
  validate,
  vertexRing,
  vertexValence,
  VertexKind,
  weldVertices,
  type QuadMesh,
} from "./quadMesh";

describe("QuadMesh generators and validation", () => {
  it("box cage is a closed, outward-oriented manifold with Euler characteristic 2", () => {
    const qm = createBoxCage([-1, -1, -1], [1, 1, 1], [2, 3, 4]);
    const r = validate(qm);
    expect(r.ok).toBe(true);
    expect(r.closed).toBe(true);
    expect(r.eulerCharacteristic).toBe(2);
    expect(r.components).toBe(1);
    expect(signedVolume(qm)).toBeCloseTo(8, 10);
    const topo = buildTopology(qm);
    let corners = 0;
    for (let v = 0; v < topo.nv; v++) if (topo.vEdgeStart[v + 1] - topo.vEdgeStart[v] === 3) corners++;
    expect(corners).toBe(8);
  });

  it("grid / cylinder / torus cages", () => {
    const g = createGridCage(3, 2);
    const rg = validate(g);
    expect(rg.ok).toBe(true);
    expect(rg.boundaryLoopCount).toBe(1);
    expect(rg.eulerCharacteristic).toBe(1);
    const c = createCylinderCage(10, 20, 8, 3);
    const rc = validate(c);
    expect(rc.ok && rc.boundaryLoopCount === 2 && rc.eulerCharacteristic === 0).toBe(true);
    const t = createTorusCage(10, 3, 8, 6);
    const rt = validate(t);
    expect(rt.ok && rt.closed && rt.eulerCharacteristic === 0).toBe(true);
    expect([...vertexValence(t)].every((v) => v === 4)).toBe(true);
  });

  it("classifies vertices (smooth / crease / corner) and builds ordered rings", () => {
    const g = createGridCage(2, 2);
    const topo = buildTopology(g);
    expect(topo.vKind[4]).toBe(VertexKind.Smooth); // centre
    expect(topo.vKind[1]).toBe(VertexKind.Crease); // boundary, 2 faces
    expect(topo.vKind[0]).toBe(VertexKind.Corner); // boundary, 1 face
    const ring = vertexRing(topo, 4);
    expect(ring.closed).toBe(true);
    expect(ring.nbr.length).toBe(4);
    // CCW around +z: successive neighbours turn left
    const c = g.vertices[4];
    for (let i = 0; i < 4; i++) {
      const a = g.vertices[ring.nbr[i]], b = g.vertices[ring.nbr[(i + 1) % 4]];
      const z = (a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0]);
      expect(z).toBeGreaterThan(0);
    }
    const br = vertexRing(topo, 1);
    expect(br.closed).toBe(false);
    expect(br.nbr.length).toBe(3);
    expect(ringSectors(topo, br).length).toBe(1);
    // crease through the middle row
    setCrease(g, [[3, 4], [4, 5]], true);
    const t2 = buildTopology(g);
    expect(t2.vKind[4]).toBe(VertexKind.Crease);
    expect(ringSectors(t2, vertexRing(t2, 4)).map((s) => s.faceIdx.length)).toEqual([2, 2]);
    expect(t2.vKind[3]).toBe(VertexKind.Corner); // boundary + crease = 3 sharp edges
  });

  it("detects non-manifold edges and orientation flips", () => {
    const g = createGridCage(2, 1);
    g.faces[1] = g.faces[1].slice().reverse();
    const r = validate(g);
    expect(r.ok).toBe(false);
    expect(r.inconsistentEdges.length).toBe(1);
    const nm = createGridCage(1, 1);
    const top = addVertex(nm, [0, 0, 50]);
    const top2 = addVertex(nm, [100, 0, 50]);
    nm.faces.push([0, 1, top2, top]);
    const extra = addVertex(nm, [50, -50, 0]);
    nm.faces.push([0, extra, 1]);
    const r2 = validate(nm);
    expect(r2.nonManifoldEdges.length).toBe(1);
  });

  it("orient() repairs flipped faces and orients closed meshes outward", () => {
    const qm = createBoxCage([0, 0, 0], [1, 1, 1], 2);
    const flips = [0, 3, 7, 11, 20];
    for (const f of flips) qm.faces[f] = qm.faces[f].slice().reverse();
    expect(validate(qm).ok).toBe(false);
    const res = orient(qm);
    expect(res.orientable).toBe(true);
    expect(res.flipped).toBe(flips.length);
    expect(validate(qm).ok).toBe(true);
    expect(signedVolume(qm)).toBeGreaterThan(0);
    // fully inverted closed mesh gets flipped back
    const inv = createBoxCage([0, 0, 0], [1, 1, 1], 1);
    inv.faces = inv.faces.map((f) => f.slice().reverse());
    orient(inv);
    expect(signedVolume(inv)).toBeCloseTo(1, 12);
  });
});

describe("QuadMesh editing", () => {
  it("addQuad auto-orients against neighbours; removeFace", () => {
    const g = createGridCage(1, 1);
    const a = addVertex(g, [200, 0, 0]);
    const b = addVertex(g, [200, 100, 0]);
    // given in the "wrong" order (would duplicate the direction of edge 1->3)
    const f = addQuad(g, [1, 3, b, a]);
    expect(validate(g).ok).toBe(true);
    expect(f).toBe(1);
    expect(faceAdjacency(g)[0]).toContain(1);
    removeFace(g, 1);
    expect(g.faces.length).toBe(1);
  });

  it("splitEdgeLoop inserts a full ring on a closed box and a strip on a grid", () => {
    const box = createBoxCage([0, 0, 0], [1, 1, 1], 2);
    const nf = box.faces.length, nv = box.vertices.length;
    // an edge parallel to x on the bottom face
    const topo = buildTopology(box);
    let edge: [number, number] | null = null;
    for (let e = 0; e < topo.ne && !edge; e++) {
      const p = box.vertices[topo.edgeV[2 * e]], q = box.vertices[topo.edgeV[2 * e + 1]];
      if (Math.abs(p[1] - q[1]) < 1e-12 && Math.abs(p[2] - q[2]) < 1e-12 && p[2] === 0 && p[1] === 0.5) edge = [topo.edgeV[2 * e], topo.edgeV[2 * e + 1]];
    }
    expect(edge).not.toBeNull();
    const r = splitEdgeLoop(box, edge!, 0.25);
    expect(r.closed).toBe(true);
    expect(r.splitFaces.length).toBe(8); // ring around the box (4 sides x 2)
    expect(box.faces.length).toBe(nf + 8);
    expect(box.vertices.length).toBe(nv + 8);
    const v = validate(box);
    expect(v.ok).toBe(true);
    expect(v.eulerCharacteristic).toBe(2);
    expect(box.faces.every((f) => f.length === 4)).toBe(true);
    // all inserted vertices lie on one plane x = const (parameter measured from the a-side)
    const xs = r.newVertices.map((i) => box.vertices[i][0]);
    const x0 = box.vertices[edge![0]][0], x1 = box.vertices[edge![1]][0];
    for (const x of xs) expect(x).toBeCloseTo(x0 + 0.25 * (x1 - x0), 12);

    const g = createGridCage(3, 3);
    const r2 = splitEdgeLoop(g, [0, 1]); // bottom edge, ring goes along v
    expect(r2.closed).toBe(false);
    expect(r2.splitFaces.length).toBe(3);
    expect(g.faces.length).toBe(12);
    expect(validate(g).ok).toBe(true);
  });

  it("splitEdgeLoop stops at a triangle and inserts the vertex into it", () => {
    const g = createGridCage(2, 1);
    const apex = addVertex(g, [100, 150, 0]);
    g.faces.push([4, 5, apex]); // triangle on top of the right quad (its top edge runs 5 -> 4)
    expect(validate(g).ok).toBe(true);
    splitEdgeLoop(g, [1, 2]);
    expect(validate(g).ok).toBe(true);
    expect(g.faces.some((f) => f.length === 4 && f.includes(apex))).toBe(true);
  });

  it("extrudeBoundaryEdges adds consistently oriented quads", () => {
    const g = createGridCage(2, 2);
    const loops = boundaryLoops(g);
    expect(loops.length).toBe(1);
    const loop = loops[0];
    const edges: [number, number][] = [];
    for (let i = 0; i < loop.length; i++) edges.push([loop[i], loop[(i + 1) % loop.length]]);
    const before = g.faces.length;
    const r = extrudeBoundaryEdges(g, edges, [0, 0, 10]);
    expect(r.skipped.length).toBe(0);
    expect(g.faces.length).toBe(before + edges.length);
    const v = validate(g);
    expect(v.ok).toBe(true);
    expect(v.boundaryLoopCount).toBe(1);
    // extruding an interior edge is skipped
    const r2 = extrudeBoundaryEdges(g, [[4, 5]], [0, 0, 1]);
    expect(r2.skipped.length).toBe(1);
  });

  it("mergeVertices collapses a face edge; weldVertices fuses duplicates", () => {
    const g = createGridCage(2, 2);
    const remap = mergeVertices(g, [1, 4]);
    expect(g.vertices.length).toBe(8);
    expect(g.faces.length).toBe(4);
    expect(g.faces.filter((f) => f.length === 3).length).toBe(2);
    expect(remap[4]).toBe(remap[1]);
    expect(validate(g).ok).toBe(true);
    // collapsing a "bridge" edge of a 1-quad-wide strip pinches it into a bowtie
    const strip = createGridCage(2, 1);
    mergeVertices(strip, [1, 4]);
    expect(validate(strip).nonManifoldVertices.length).toBe(1);

    // two separate quads sharing geometric (but not topological) vertices
    const a: QuadMesh = createGridCage(1, 1);
    const b = createGridCage(1, 1, { origin: [100, 0, 0] });
    const off = a.vertices.length;
    const both: QuadMesh = { vertices: [...a.vertices, ...b.vertices], faces: [...a.faces, ...b.faces.map((f) => f.map((v) => v + off))] };
    expect(validate(both).components).toBe(2);
    weldVertices(both, 1e-6);
    expect(both.vertices.length).toBe(6);
    const vb = validate(both);
    expect(vb.ok).toBe(true);
    expect(vb.components).toBe(1);
  });

  it("cloneQuadMesh is deep", () => {
    const g = createGridCage(1, 1);
    g.creaseEdges = [[0, 1]];
    g.fixed = [true, false, false, false];
    const c = cloneQuadMesh(g);
    c.vertices[0][0] = 999;
    c.faces[0][0] = 3;
    c.creaseEdges![0][0] = 2;
    c.fixed![0] = false;
    expect(g.vertices[0][0]).toBe(0);
    expect(g.faces[0][0]).toBe(0);
    expect(g.creaseEdges[0][0]).toBe(0);
    expect(g.fixed[0]).toBe(true);
    const p: Vec3 = g.vertices[0];
    expect(p).toEqual([0, 0, 0]);
  });
});
