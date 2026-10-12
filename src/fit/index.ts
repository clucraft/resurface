/**
 * Fitting & alignment library (pure TypeScript, worker/Node safe).
 *
 *  - primitives: plane / sphere / cylinder / cone fitting, RANSAC, residuals, constrained refits
 *  - reference:  reference-geometry constructions (intersections, mid planes, projections …)
 *  - align:      Kabsch/Horn, ICP, global multi-scan alignment, object align, N-point align
 *  - symmetry:   symmetry plane estimation + deviation map
 *  - bspline:    NURBS basis, curve/surface evaluation, curve & surface fitting
 *  - fit2d:      2D line/circle/arc fitting, 3D line fitting
 *  - transform:  rigid-transform helpers
 *  - pointQuery: grid spatial index, normal estimation, point-cloud closest-point oracle
 */
export * from "./primitives";
export * from "./reference";
export * from "./align";
export * from "./symmetry";
export * from "./bspline";
export * from "./fit2d";
export * from "./transform";
export * from "./pointQuery";
export { BandSPD, BandGeneral } from "./sparse";
export { DEFAULT_MAX_POINTS, mulberry32, jacobiEigen, levenbergMarquardt } from "./common";
export type { PointArray, LMProblem, LMOptions, LMResult } from "./common";
