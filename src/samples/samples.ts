/** Procedural "scans" used as built-in samples (and in tests). */
import type { MeshData } from "../core/types";
import { addScanNoise, surfaceNets } from "./surfaceNets";
import { box, cylinder, polygon2D, revolve, smoothUnion, sphere, subtract, union } from "./sdf";

export type SampleId = "bracket" | "turned" | "organic";

export const SAMPLE_INFO: { id: SampleId; label: string; description: string }[] = [
  { id: "bracket", label: "Mechanical bracket", description: "Prismatic part with holes, boss and fillet — primitives, sketches, extrudes" },
  { id: "turned", label: "Turned shaft", description: "Rotationally symmetric part — sections and revolve" },
  { id: "organic", label: "Organic handle", description: "Freeform shape — fit surface, quad surfacing, autosurface" },
];

export function makeSample(id: SampleId, resolution = 1): MeshData {
  switch (id) {
    case "bracket": {
      const base = box(0, 0, 6, 60, 40, 6, 0.6);
      const wall = box(-50, 0, 38, 10, 40, 38, 0.6);
      const body = smoothUnion(10, base, wall);
      const boss = cylinder("z", 20, 0, 14, 15, 14);
      const solid = smoothUnion(3, body, boss);
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
