/** Conversion-map colours per face kind (shared by the Auto Convert preview and converted bodies). */
import type { FaceKind } from "./brepPlan";

export const FACE_KIND_COLOR: Record<FaceKind, string> = {
  plane: "#a9b8cc",
  cylinder: "#3d8bfd",
  cone: "#20c9a6",
  sphere: "#b07cf7",
  torus: "#c3e35a",
  faceted: "#f06a6a",
};

export const FACE_KIND_ORDER: FaceKind[] = ["plane", "cylinder", "cone", "sphere", "torus", "faceted"];

export const FACE_KIND_PLURAL: Record<FaceKind, string> = {
  plane: "Planes",
  cylinder: "Cylinders",
  cone: "Cones",
  sphere: "Spheres",
  torus: "Tori (fillets)",
  faceted: "Faceted",
};

const rgb = (hex: string): [number, number, number] => [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)];

/**
 * Per-vertex RGB colours for a body tessellation (vertices are per face) from face kinds.
 * `faceTriangleRanges` = [firstTriangle, count] per face, `indices` the triangle indices.
 */
export function conversionMapColors(
  tess: { positions: Float32Array; indices: Uint32Array; faceTriangleRanges: Uint32Array },
  faceKinds: readonly FaceKind[],
): Uint8Array {
  const colors = new Uint8Array(tess.positions.length);
  const fallback = rgb(FACE_KIND_COLOR.plane);
  colors.fill(200);
  const nf = tess.faceTriangleRanges.length / 2;
  for (let f = 0; f < nf; f++) {
    const c = faceKinds[f] ? rgb(FACE_KIND_COLOR[faceKinds[f]]) : fallback;
    const first = tess.faceTriangleRanges[2 * f], n = tess.faceTriangleRanges[2 * f + 1];
    for (let t = first; t < first + n; t++)
      for (let k = 0; k < 3; k++) {
        const v = tess.indices[3 * t + k];
        colors[3 * v] = c[0];
        colors[3 * v + 1] = c[1];
        colors[3 * v + 2] = c[2];
      }
  }
  return colors;
}
