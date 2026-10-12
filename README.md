# Resurface

**Scan-to-CAD reverse engineering.** Resurface turns 3D scans (STL / OBJ / PLY / PTX) into
editable CAD geometry — reference primitives, sketches, extrusions, revolves, fitted and
freeform NURBS surfaces — and exports STEP and IGES for any CAD/CAM package.

> Status: early development.

## Feature set (target)

| Area | Features |
| --- | --- |
| Import | STL (ASCII/binary), OBJ, PLY (ASCII/binary, colours), PTX point clouds |
| Mesh | Reduce (decimate), smooth, fill holes, remove outliers / small shells, clip, delete selection, de-feature |
| Selection | Brush, rectangle, lasso, smart (curvature-aware) region growing, grow / shrink / invert |
| Reference geometry | Fit plane, cylinder, cone, sphere; reference lines & points; symmetry plane with analyser |
| Alignment | Object align to world (from primitives), N-point alignment, global fine alignment (ICP) |
| Sketching | Interactive 2D sections, constrained 2D sketches with line/arc fitting, 3D sketches on the mesh |
| Modelling | Extrude, revolve, fit surface, freeform quad surfaces (snap-to-mesh), one-click auto-surface |
| Analysis | Real-time deviation colour map (CAD vs scan) |
| Export | STEP, IGES, STL |

## Development

```bash
npm install
npm run dev        # http://localhost:5173
npm test           # unit tests (vitest)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the module layout.
