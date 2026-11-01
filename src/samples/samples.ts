/** Procedural "scans" used as built-in samples (and in tests). */
import type { MeshData } from "../core/types";
import { addScanNoise, surfaceNets } from "./surfaceNets";
import { box, cylinder, polygon2D, revolve, smoothUnion, sphere, subtract, torusZ, union } from "./sdf";

export type SampleId = "bracket" | "turned" | "organic" | "grille" | "cadPlate";

export const SAMPLE_INFO: { id: SampleId; label: string; description: string }[] = [
  { id: "bracket", label: "Mechanical bracket", description: "Prismatic part with holes, boss and fillet — primitives, sketches, extrudes" },
  { id: "turned", label: "Turned shaft", description: "Rotationally symmetric part — sections and revolve" },
  { id: "organic", label: "Organic handle", description: "Freeform shape — fit surface, quad surfacing, autosurface" },
  { id: "grille", label: "Bumper grille", description: "Curved slatted grille, scanned at a tilt — align, section, design a clip-on bracket" },
  { id: "cadPlate", label: "Mounting plate (CAD STL)", description: "STL exported from CAD — try Model → Auto Convert to turn it back into a STEP-ready solid" },
];

export function makeSample(id: SampleId, resolution = 1): MeshData {
  switch (id) {
    case "cadPlate":
      throw new Error("the CAD sample is built by the kernel (see cadSample.ts)");
    case "bracket": {
      const base = box(0, 0, 6, 60, 40, 6, 0.6);
      const wall = box(-50, 0, 38, 10, 40, 38, 0.6);
      // R8 fillet between the base top (z=12) and the wall face (x=-40), built explicitly
      const filletR = 8;
      const filletStrip = subtract(box(-40 + filletR / 2, 0, 12 + filletR / 2, filletR / 2, 40, filletR / 2), cylinder("y", -40 + filletR, 0, 12 + filletR, filletR, 41));
      const boss = cylinder("z", 20, 0, 14, 15, 14);
      // R2.5 ring fillet around the boss foot
      const ringR = 2.5;
      const ring = subtract(cylinder("z", 20, 0, 12 + ringR / 2, 15 + ringR, ringR / 2), torusZ(20, 0, 12 + ringR, 15 + ringR, ringR));
      const solid = union(base, wall, filletStrip, boss, ring);
      const cut = subtract(
        solid,
        cylinder("z", 20, 0, 14, 7.5, 30),
        cylinder("z", 38, 26, 6, 5, 20),
        cylinder("z", 38, -26, 6, 5, 20),
        cylinder("x", -50, 0, 50, 11, 20),
        box(-50, 0, 76, 20, 12, 8, 0.5),
      );
      const mesh = surfaceNets(cut, [-61, -41, -1], [61, 41, 77], 0.55 / resolution);
      addScanNoise(mesh, 0.012, 7);
      return mesh;
    }
    case "turned": {
      const profile = polygon2D([
        [4, 0], [12, 0], [13, 1], [13, 18], [30, 18], [32, 20], [32, 30], [30, 32], [16, 32],
        [16, 40], [13, 42], [13, 52], [22, 62], [22, 70], [20, 72], [8, 72], [8, 80], [4, 80],
      ]);
      const part = revolve(profile);
      const mesh = surfaceNets(part, [-33, -33, -1], [33, 33, 81], 0.4 / resolution);
      addScanNoise(mesh, 0.01, 11);
      return mesh;
    }
    case "grille": {
      // Curved grille insert: 300 x 140 mm frame, horizontal slats (8 mm tall, 14 mm gaps, 15 mm deep),
      // three vertical dividers. The front face follows the bumper curvature (radius 1200 mm across X).
      // Car-style axes: X = left/right, Y = depth (front face at u = 0, slats extend to +Y), Z = up.
      const R = 1200;
      const depth = 15;
      const curved = (f: (x: number, u: number, z: number) => number) => (x: number, y: number, z: number) => f(x, y - (x * x) / (2 * R), z);
      const slab = (cx: number, cz: number, hx: number, hz: number, r: number) =>
        curved((x, u, z) => box(cx, depth / 2, cz, hx, depth / 2, hz, r)(x, u, z));
      const outer = slab(0, 70, 150, 70, 1.2);
      const opening = curved((x, u, z) => box(0, depth / 2, 70, 140, depth, 60, 2)(x, u, z));
      const frame = subtract(outer, opening);
      const parts = [frame];
      for (let zc = 19; zc <= 121; zc += 22) parts.push(slab(0, zc, 141, 4, 1.2));
      for (const xc of [-70, 0, 70]) parts.push(slab(xc, 70, 1.6, 61, 0.6));
      const grille = union(...parts);
      const mesh = surfaceNets(grille, [-152, -2, -2], [152, depth + 40, 142], 0.6 / resolution);
      // as scanned: slightly tilted and offset (use Align → Object Align to straighten it)
      const ax = (8 * Math.PI) / 180, az = (-6 * Math.PI) / 180;
      const P = mesh.positions;
      for (let i = 0; i < P.length; i += 3) {
        const x = P[i], y = P[i + 1], z = P[i + 2];
        const y1 = y * Math.cos(ax) - z * Math.sin(ax), z1 = y * Math.sin(ax) + z * Math.cos(ax);
        P[i] = x * Math.cos(az) - y1 * Math.sin(az) + 12;
        P[i + 1] = x * Math.sin(az) + y1 * Math.cos(az) - 30;
        P[i + 2] = z1 + 8;
      }
      addScanNoise(mesh, 0.02, 21);
      return mesh;
    }
    case "organic": {
      const blobs = [
        sphere(-38, 0, 10, 16),
        sphere(-18, 4, 16, 19),
        sphere(4, 2, 20, 21),
        sphere(26, -2, 16, 17),
        sphere(44, -4, 10, 13),
        sphere(8, 14, 10, 12),
        sphere(-8, -14, 9, 11),
      ];
      const handle = smoothUnion(14, ...blobs);
      // flatten the bottom like a resting object
      const flat = subtract(handle, box(0, 0, -20, 80, 60, 20));
      const shape = union(flat);
      const mesh = surfaceNets(shape, [-56, -36, -1], [62, 36, 44], 0.6 / resolution);
      addScanNoise(mesh, 0.015, 3);
      return mesh;
    }
  }
}
