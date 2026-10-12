/**
 * Resurface mesh library — public API.
 *
 * Pure TypeScript over typed arrays (MeshData from core/types), linear or near-linear
 * algorithms designed for scans of tens of millions of triangles. Only `spatial` depends on
 * three / three-mesh-bvh; import the sub-modules directly to keep worker bundles lean.
 */
export * from "./io";
export * from "./topology";
export * from "./repair";
export * from "./clip";
export * from "./smooth";
export * from "./holes";
export * from "./defeature";
export * from "./decimate";
export * from "./section";
export * from "./curvature";
export * from "./select";
export * from "./spatial";
export { makeGrid, makeUVSphere, makeIcosphere, makeCylinder, makeExtrudedProfile } from "./util/generate";
