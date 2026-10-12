/**
 * Declarative command catalogue shown in the ribbon. Execution lives in tools/registry.ts so
 * tool modules can be loaded lazily.
 */
import type { LucideIcon } from "lucide-react";
import {
  Upload, Shrink, Waves, Disc, Eraser, Scissors, Trash2, Sparkles, FlipVertical, Download,
  Square, Cylinder, Cone, Circle, WandSparkles, Slash, Dot, FlipHorizontal2, Move3d, Crosshair, Magnet,
  PenTool, Spline, Layers, ArrowUpFromLine, Rotate3d, Grid3x3, Shapes, Box, Combine, Squircle,
  FlipHorizontal, Activity, Ruler, Palette, FileDown, FileOutput, Save, FolderOpen, Expand, Minimize2,
  ArrowLeftRight, CircleDot, Orbit, Blend, Shell, Copy,
} from "lucide-react";
import type { App } from "./App";

export type Requirement = "mesh" | "selection" | "sketch" | "body" | "twoMeshes" | "twoBodies";

export interface CommandDef {
  id: string;
  label: string;
  icon: LucideIcon;
  tab: string;
  group: string;
  tooltip: string;
  requires?: Requirement[];
  /** professional-tier feature marker */
  pro?: boolean;
}

export const RIBBON_TABS: { id: string; label: string }[] = [
  { id: "mesh", label: "Mesh" },
  { id: "reference", label: "Reference" },
  { id: "align", label: "Align" },
  { id: "sketch", label: "Sketch" },
  { id: "model", label: "Model" },
  { id: "freeform", label: "Freeform" },
  { id: "analyze", label: "Analyze" },
  { id: "export", label: "Export" },
];

export const COMMANDS: CommandDef[] = [
  // Mesh
  { id: "import", label: "Import", icon: Upload, tab: "mesh", group: "File", tooltip: "Import STL, OBJ, PLY or PTX scan data" },
  { id: "reduce", label: "Reduce", icon: Shrink, tab: "mesh", group: "Optimize", tooltip: "Reduce the triangle count (quadric decimation)", requires: ["mesh"] },
  { id: "smooth", label: "Smooth", icon: Waves, tab: "mesh", group: "Optimize", tooltip: "Smooth noise (Taubin) — whole mesh or selection", requires: ["mesh"] },
  { id: "fillHoles", label: "Fill Holes", icon: Disc, tab: "mesh", group: "Repair", tooltip: "Find and fill holes with curvature-matched patches", requires: ["mesh"] },
  { id: "clean", label: "Clean", icon: Sparkles, tab: "mesh", group: "Repair", tooltip: "Remove small disconnected pieces and degenerate triangles", requires: ["mesh"] },
  { id: "defeature", label: "De-feature", icon: Eraser, tab: "mesh", group: "Repair", tooltip: "Remove the selected stamp/marker and fill it smoothly", requires: ["selection"] },
  { id: "clip", label: "Clip", icon: Scissors, tab: "mesh", group: "Edit", tooltip: "Cut the mesh with a plane", requires: ["mesh"] },
  { id: "deleteSel", label: "Delete", icon: Trash2, tab: "mesh", group: "Edit", tooltip: "Delete selected triangles (Del)", requires: ["selection"] },
  { id: "flip", label: "Flip", icon: FlipVertical, tab: "mesh", group: "Edit", tooltip: "Flip triangle orientation", requires: ["mesh"] },
  { id: "growSel", label: "Grow", icon: Expand, tab: "mesh", group: "Selection", tooltip: "Grow the selection by one ring", requires: ["selection"] },
  { id: "shrinkSel", label: "Shrink", icon: Minimize2, tab: "mesh", group: "Selection", tooltip: "Shrink the selection by one ring", requires: ["selection"] },
  { id: "invertSel", label: "Invert", icon: ArrowLeftRight, tab: "mesh", group: "Selection", tooltip: "Invert the selection", requires: ["mesh"] },
  { id: "connectedSel", label: "Connected", icon: Blend, tab: "mesh", group: "Selection", tooltip: "Extend the selection to whole connected pieces", requires: ["selection"] },

  // Reference geometry
  { id: "fitPlane", label: "Plane", icon: Square, tab: "reference", group: "Primitives", tooltip: "Fit a plane to the selected triangles", requires: ["selection"] },
  { id: "fitCylinder", label: "Cylinder", icon: Cylinder, tab: "reference", group: "Primitives", tooltip: "Fit a cylinder to the selected triangles", requires: ["selection"] },
  { id: "fitCone", label: "Cone", icon: Cone, tab: "reference", group: "Primitives", tooltip: "Fit a cone to the selected triangles", requires: ["selection"] },
  { id: "fitSphere", label: "Sphere", icon: Circle, tab: "reference", group: "Primitives", tooltip: "Fit a sphere to the selected triangles", requires: ["selection"] },
  { id: "fitAuto", label: "Auto", icon: WandSparkles, tab: "reference", group: "Primitives", tooltip: "Detect the best primitive for the selection", requires: ["selection"] },
  { id: "refLine", label: "Line", icon: Slash, tab: "reference", group: "Reference", tooltip: "Reference line: axis of a cylinder/cone or intersection of two planes" },
  { id: "refPoint", label: "Point", icon: Dot, tab: "reference", group: "Reference", tooltip: "Reference point: picked on the mesh, sphere centre or 3-plane intersection" },
  { id: "refPlane", label: "Plane+", icon: Copy, tab: "reference", group: "Reference", tooltip: "Construct a plane: offset, 3 points, mid-plane, world plane" },
  { id: "symmetry", label: "Symmetry", icon: FlipHorizontal2, tab: "reference", group: "Analysis", tooltip: "Find the symmetry plane of the scan", requires: ["mesh"] },

  // Align
  { id: "objectAlign", label: "Object Align", icon: Move3d, tab: "align", group: "Coordinate system", tooltip: "Align the scan to the world coordinate system using primitives", requires: ["mesh"] },
  { id: "transform", label: "Move", icon: Orbit, tab: "align", group: "Coordinate system", tooltip: "Move/rotate the scan numerically", requires: ["mesh"] },
  { id: "nPointAlign", label: "N-Points", icon: Crosshair, tab: "align", group: "Registration", tooltip: "Align two scans by picking corresponding points", requires: ["twoMeshes"] },
  { id: "fineAlign", label: "Fine Align", icon: Magnet, tab: "align", group: "Registration", tooltip: "Refine overlapping scans with ICP", requires: ["twoMeshes"] },

  // Sketch
  { id: "sketch2d", label: "2D Sketch", icon: PenTool, tab: "sketch", group: "Sketch", tooltip: "Section the scan with a plane and sketch over the section" },
  { id: "multiSection", label: "Sections", icon: Layers, tab: "sketch", group: "Sketch", tooltip: "Create several parallel section sketches (for lofts)", requires: ["mesh"] },
  { id: "sketch3d", label: "3D Sketch", icon: Spline, tab: "sketch", group: "Sketch", tooltip: "Draw curves directly on the scan", requires: ["mesh"] },

  // Model
  { id: "extrude", label: "Extrude", icon: ArrowUpFromLine, tab: "model", group: "Features", tooltip: "Extrude a sketch into a solid or surface", requires: ["sketch"] },
  { id: "revolve", label: "Revolve", icon: Rotate3d, tab: "model", group: "Features", tooltip: "Revolve a sketch around an axis", requires: ["sketch"] },
  { id: "loft", label: "Loft", icon: Shapes, tab: "model", group: "Features", tooltip: "Loft through several section sketches", requires: ["sketch"], pro: true },
  { id: "fitSurface", label: "Fit Surface", icon: Grid3x3, tab: "model", group: "Surfaces", tooltip: "Fit a NURBS surface to the selected region", requires: ["selection"] },
  { id: "primitiveBody", label: "From Primitive", icon: CircleDot, tab: "model", group: "Surfaces", tooltip: "Turn a reference primitive into a CAD surface" },
  { id: "boolean", label: "Boolean", icon: Combine, tab: "model", group: "Modify", tooltip: "Unite, subtract or intersect bodies", requires: ["twoBodies"], pro: true },
  { id: "fillet", label: "Fillet", icon: Squircle, tab: "model", group: "Modify", tooltip: "Round edges of a body", requires: ["body"], pro: true },
  { id: "mirror", label: "Mirror", icon: FlipHorizontal, tab: "model", group: "Modify", tooltip: "Mirror a body about a plane", requires: ["body"], pro: true },

  // Freeform
  { id: "quadSurface", label: "Quad Surface", icon: Box, tab: "freeform", group: "Freeform", tooltip: "Build a freeform surface from a quad cage that snaps to the scan", requires: ["mesh"] },
  { id: "autoSurface", label: "Autosurface", icon: Shell, tab: "freeform", group: "Freeform", tooltip: "One-click NURBS surfacing of organic shapes", requires: ["mesh"] },

  // Analyze
  { id: "deviation", label: "Deviation", icon: Activity, tab: "analyze", group: "Analysis", tooltip: "Colour map of distances between the CAD model and the scan", requires: ["mesh"] },
  { id: "curvatureMap", label: "Curvature", icon: Palette, tab: "analyze", group: "Analysis", tooltip: "Colour the scan by curvature", requires: ["mesh"] },
  { id: "measure", label: "Measure", icon: Ruler, tab: "analyze", group: "Analysis", tooltip: "Measure distances and angles" },

  // Export
  { id: "exportStep", label: "STEP", icon: FileDown, tab: "export", group: "CAD", tooltip: "Export bodies to STEP (AP214)", requires: ["body"] },
  { id: "exportIges", label: "IGES", icon: FileOutput, tab: "export", group: "CAD", tooltip: "Export bodies to IGES 5.3", requires: ["body"] },
  { id: "exportStlCad", label: "STL (CAD)", icon: Download, tab: "export", group: "Mesh", tooltip: "Export tessellated bodies as STL", requires: ["body"] },
  { id: "exportStlMesh", label: "STL (Scan)", icon: Download, tab: "export", group: "Mesh", tooltip: "Export the active scan mesh as STL", requires: ["mesh"] },
  { id: "saveProject", label: "Save", icon: Save, tab: "export", group: "Project", tooltip: "Save the project (.rsf)" },
  { id: "openProject", label: "Open", icon: FolderOpen, tab: "export", group: "Project", tooltip: "Open a project (.rsf)" },
];

export function commandEnabled(app: App, c: CommandDef): boolean {
  if (!c.requires) return true;
  const doc = app.doc;
  for (const r of c.requires) {
    switch (r) {
      case "mesh":
        if (!doc.ofKind("mesh").length) return false;
        break;
      case "selection":
        if (app.ui.get().selectedTriangles === 0) return false;
        break;
      case "sketch":
        if (!doc.ofKind("sketch").length) return false;
        break;
      case "body":
        if (!doc.ofKind("body").length) return false;
        break;
      case "twoMeshes":
        if (doc.ofKind("mesh").length < 2) return false;
        break;
      case "twoBodies":
        if (doc.ofKind("body").length < 2) return false;
        break;
    }
  }
  return true;
}
