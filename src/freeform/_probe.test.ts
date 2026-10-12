import { it } from "vitest";
import type { Vec3 } from "../core/types";
import { length, sub } from "../core/vec";
import { subdivisionStencils } from "./catmullClark";
import { evaluateSurface, toBicubicPatchesEx } from "./patches";
import { cageSize, createGridCage, removeFaces, removeUnusedVertices, type QuadMesh } from "./quadMesh";
import { csrMulVec3 } from "./sparse";
function fan(nf: number): QuadMesh {
  const vertices: Vec3[] = [[0, 0, 0]];
  for (let i = 0; i <= nf; i++) { const a = (i * Math.PI) / nf; vertices.push([10 * Math.cos(a), 10 * Math.sin(a), 2 * Math.sin(2 * a)]); }
  for (let i = 0; i < nf; i++) { const a = ((i + 0.5) * Math.PI) / nf; vertices.push([14 * Math.cos(a), 14 * Math.sin(a), 3 * Math.cos(a)]); }
  return { vertices, faces: Array.from({ length: nf }, (_, i) => [0, 1 + i, nf + 2 + i, 2 + i]) };
}
function lshape(): QuadMesh {
  const g = createGridCage(6, 6, { uAxis: [60, 0, 0], vAxis: [0, 60, 0] });
  const del: number[] = [];
  g.faces.forEach((f, i) => { const c = g.vertices[f[0]]; if (c[0] >= 30 && c[1] >= 30) del.push(i); });
  removeFaces(g, del); removeUnusedVertices(g);
  g.vertices = g.vertices.map(([x, y]) => [x, y, 4 * Math.sin(x / 12) * Math.cos(y / 15)] as Vec3);
  return g;
}
function dev(qm: QuadMesh, g1: boolean) {
  const ps = toBicubicPatchesEx(qm, { g1Seams: g1 });
  const st = subdivisionStencils(ps.mesh, 4); const Q = csrMulVec3(st.rows, ps.mesh.vertices);
  let e = 0, s2 = 0;
  for (let s = 0; s < Q.length; s++) { const d = length(sub(evaluateSurface(ps.patches[st.faceOfSample[s]], st.uvOfSample[2 * s], st.uvOfSample[2 * s + 1]), Q[s])); e = Math.max(e, d); s2 += d * d; }
  return `${(e / cageSize(qm)).toExponential(2)}/${(Math.sqrt(s2 / Q.length) / cageSize(qm)).toExponential(2)}`;
}
it("probe", () => {
  const out: string[] = [`ACC: fan3 ${dev(fan(3), false)} fan5 ${dev(fan(5), false)} L ${dev(lshape(), false)}`];
  for (const s of [1]) { out.push(`W*${s}: fan3 ${dev(fan(3), true)} fan5 ${dev(fan(5), true)} L ${dev(lshape(), true)}`); }
  console.log(out.join("\n"));
});
