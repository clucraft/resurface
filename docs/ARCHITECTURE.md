# Resurface — Architecture

Resurface is a scan-to-CAD reverse-engineering application (inspired by the QuickSurface
feature set): import 3D scans, clean them up, extract reference geometry, align, sketch on
sections, build CAD features (extrude / revolve / fitted & freeform surfaces) and export
STEP / IGES.

Stack: TypeScript + Vite, React UI, Three.js viewport, three-mesh-bvh for spatial queries,
OpenCASCADE 8 (WebAssembly, `replicad-opencascadejs`) as the B-rep kernel. Packaged as a
Windows desktop app with Electron (the same build runs in a browser).

## Conventions

* Units: millimetres. Angles: radians unless suffixed `Deg`.
* World is **Z-up**, right-handed.
* Shared data contracts live in `src/core/types.ts` (plain arrays / typed arrays — no three.js
  types) so that algorithm modules run unchanged in Web Workers and in Node unit tests.
* Vector helpers: `src/core/vec.ts`; dense linear algebra: `src/core/linalg.ts`.
* Algorithm modules are **pure**: no DOM, no React, no global state. They may import
  `three` / `three-mesh-bvh` only inside `src/mesh/spatial.ts`.
* Unit tests sit next to the code as `*.test.ts` and run with `npx vitest run <path>` (Node).
* Do not add npm dependencies without coordinating — `package.json` is owned by the app shell.

## Source layout

| Folder | Responsibility |
| --- | --- |
| `src/core/` | Shared types, vector math, linear algebra, document model, undo/redo, events |
| `src/mesh/` | Mesh I/O (STL/OBJ/PLY/PTX), topology, repair & editing, sections, curvature, spatial index |
| `src/fit/` | Primitive fitting (plane/sphere/cylinder/cone), RANSAC, alignment (Kabsch/ICP/N-point), symmetry plane, 2D line/arc fitting, B-spline curve & surface fitting |
| `src/cad/` | OCCT kernel (runs in a Web Worker): features → B-rep, tessellation, STEP export; IGES writer |
| `src/freeform/` | Quad control-cage freeform modelling (Catmull–Clark), snap-to-mesh fitting, bicubic patch export, auto-surface |
| `src/sketch/` | 2D sketch model, geometric constraint solver, profile/loop extraction, auto-segmentation of sections into lines/arcs |
| `src/viewer/` | Three.js viewport, camera control, picking, overlays, gizmos |
| `src/tools/` | Interactive commands (one per tool), each with a property panel |
| `src/ui/` | React app shell: ribbon, model tree, property panel, status bar, dialogs |
| `electron/` | Desktop packaging |

## Runtime architecture

```
React UI (ribbon, tree, panels)
      │  subscribes to
      ▼
Document (entities, history) ──► SceneSync ──► Viewer (three.js)
      │                                           ▲
      ▼                                           │ picking / pointer events
Tools (active command) ───────────────────────────┘
      │
      ├── algorithm modules (mesh / fit / sketch / freeform) — sync or in compute worker
      └── KernelClient ──RPC──► kernel.worker (OCCT wasm) ──► STEP/IGES/tessellation
```
