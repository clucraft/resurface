# Resurface

**Scan-to-CAD reverse engineering.** Resurface turns 3D scans (STL / OBJ / PLY / PTX) into
editable CAD geometry — reference primitives, constrained sketches, extrusions, revolves, fitted
and freeform NURBS surfaces — and exports STEP and IGES for any CAD/CAM package.

It runs as a Windows desktop application (Electron) and in any modern Chromium browser. The
geometry kernel is Open CASCADE 8 compiled to WebAssembly; rendering uses three.js.

> Status: 0.1 — feature-complete core workflow, under active development.

## Features

| Area | What you can do |
| --- | --- |
| **Import** | STL (ASCII/binary, streamed), OBJ, PLY (ASCII/binary, colours, point clouds), PTX gridded scans. Drag & drop or *Mesh → Import*. Scans up to ~20M triangles stay interactive (LOD while orbiting, background spatial indexing). |
| **Mesh** | Reduce (quadric decimation + clustering for huge scans), Taubin smoothing (whole mesh or selection), fill holes with curvature-matched patches, clean small shells/degenerates, clip with a plane, delete, flip, **de-feature** (remove stamps/markers and fill smoothly). |
| **Selection** | Brush, rectangle, lasso (visible triangles only), and **smart select** — primitive-aware region growing that selects a whole plane / cylinder / cone / sphere face and stops at fillet boundaries. Grow, shrink, invert, connected. |
| **Reference geometry** | Fit plane, cylinder, cone, sphere or *auto-detect* from the selection (robust least squares + RANSAC). Reference lines (axes, plane intersections, 2 points), points (picked, centres, 3-plane intersections), constructed planes (offset, mid-plane, 3 points, through axis…). **Symmetry plane** finder with mirror-deviation colour map and fine-tuning. |
| **Alignment** | **Object Align** to the world coordinate system from primitives, numeric move/rotate, **N-point** registration of two scans, **fine align** (point-to-plane ICP). |
| **2D sketching** | Interactive sections (world/reference planes, perpendicular to or through an axis, parallel stacks). Sketch over the section points: lines, 3-point arcs, circles, rectangles, splines; **fit line/arc/circle** to section points; **auto-fit** whole section curves into tangent lines & arcs. Geometric constraints (H, V, coincident, parallel, perpendicular, tangent, equal, concentric, collinear, midpoint, fix) and driving dimensions with a real-time solver. Trim and fillet. |
| **3D sketching** | Draw smooth curves directly on the scan (spline snapped to the surface). |
| **CAD features** | Extrude (one side / symmetric / two sides, draft), revolve, loft through sections, surfaces from primitives, booleans (unite / subtract / intersect), fillet & chamfer with edge picking, mirror. All features stay parametric — edit a sketch or dimension and dependent bodies rebuild. |
| **Surfacing** | **Fit Surface**: NURBS fitted to a selected region with live deviation map. **Quad Surface**: build a quad cage on the scan (or generate it from a selection); the Catmull–Clark limit surface snaps to the scan. **Autosurface**: one-click NURBS patch layout for organic shapes — G2 everywhere, G1 at extraordinary points, watertight solid when closed. |
| **Analysis** | Real-time **deviation** colour map (scan → CAD or CAD → scan) with tolerance band and statistics; curvature map; measurements (points, planes, axes, angles). |
| **Export** | STEP (AP214, named bodies), IGES 5.3 (trimmed NURBS surfaces), STL of CAD bodies or of the scan. Projects save as `.rsf`. |

## Typical workflow

1. **Import** a scan and clean it (*Mesh* tab: Clean, Fill Holes, Reduce).
2. **Smart-select** faces (key `5`) and fit planes / cylinders (*Reference* tab).
3. **Align** the scan to the world axes (*Align → Object Align*: base plane → Z, side plane → X).
4. **Sketch** on a section (*Sketch → 2D Sketch*), click section curves with **Auto fit**, add
   constraints/dimensions, then **Extrude** or **Revolve** (*Model* tab).
5. For organic areas use **Fit Surface**, **Quad Surface** or **Autosurface** (*Freeform* tab).
6. Check **Deviation** (*Analyze* tab) and **Export** STEP / IGES.

Built-in samples (*File → Samples*): a mechanical bracket, a turned shaft and an organic handle.

### Navigation

| Input | Action |
| --- | --- |
| Right drag | Rotate around the point under the cursor |
| Middle drag (or Shift + right drag) | Pan |
| Wheel | Zoom at cursor |
| `F` | Fit all |
| `1`–`5` | Pick / Brush / Rectangle / Lasso / Smart select |
| Shift / Ctrl + click | Add to / remove from selection |
| `[` `]` | Brush size |
| `Esc` | Clear selection / cancel tool |
| Ctrl+Z / Ctrl+Y | Undo / redo |

In the sketch editor: `L` line, `A` arc, `C` circle, `R` rectangle, `S` spline, `D` dimension,
`T` trim, `N` normal view, `Del` delete.

## Installing (Windows)

Build the installer (see below) and run `release/Resurface-Setup-<version>.exe`. The installer is
currently unsigned, so Windows SmartScreen may ask for confirmation ("More info → Run anyway").
Scan files (`.stl .obj .ply .ptx`) and projects (`.rsf`) can be opened with Resurface from
Explorer.

## Development

Requirements: Node.js 22+ (developed on Node 24).

```bash
npm install
npm run dev          # browser dev server with hot reload → http://localhost:5173
npm run dev:stable   # no hot reload (stable session while files change) → http://localhost:5174
npm test             # unit tests (vitest, ~350 tests)
npm run typecheck
npm run desktop      # build and run the Electron desktop app
npm run dist:win     # build the Windows installer into release/
```

`RESURFACE_SMOKE=<dir> npx electron .` runs an automated smoke test of the desktop build (loads a
sample, exercises the CAD kernel, writes a screenshot and a JSON report).

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the module layout.

## Licences

Resurface's own code is private. Third-party components: Open CASCADE Technology (LGPL-2.1, via
`replicad-opencascadejs`), three.js and three-mesh-bvh (MIT), React (MIT), lucide icons (ISC),
Electron (MIT).
