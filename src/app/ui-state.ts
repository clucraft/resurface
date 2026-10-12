/** Global UI state (not part of the document / undo history). */
export type SelectionMode = "pick" | "brush" | "rect" | "lasso" | "smart";

export interface Toast {
  id: number;
  kind: "info" | "success" | "error";
  text: string;
}

export interface ColorLegend {
  title: string;
  /** min/max values of the scale (mm) */
  min: number;
  max: number;
  /** tolerance band drawn in green */
  tolerance?: number;
  /** summary lines shown under the legend */
  stats?: string[];
}

export interface UiState {
  ribbonTab: string;
  /** entity highlighted in tree / viewport */
  activeEntityId: string | null;
  /** mesh that receives triangle selections */
  activeMeshId: string | null;
  status: string;
  busy: { message: string; progress?: number } | null;
  toasts: Toast[];
  selectionMode: SelectionMode;
  /** brush radius in screen pixels */
  brushRadius: number;
  /** smart-select angle tolerance (deg) */
  smartAngle: number;
  /** select only triangles facing the camera for rect/lasso */
  selectVisibleOnly: boolean;
  selectedTriangles: number;
  legend: ColorLegend | null;
  /** id of the active command tool, if any */
  toolId: string | null;
  meshDisplay: "shaded" | "wireframe" | "scan-colors";
  showGrid: boolean;
  /** cursor position on the model, shown in the status bar */
  cursor: [number, number, number] | null;
}

export const initialUiState: UiState = {
  ribbonTab: "mesh",
  activeEntityId: null,
  activeMeshId: null,
  status: "Import a scan (STL, OBJ, PLY, PTX) to get started — or load a sample from the File menu.",
  busy: null,
  toasts: [],
  selectionMode: "brush",
  brushRadius: 30,
  smartAngle: 12,
  selectVisibleOnly: true,
  selectedTriangles: 0,
  legend: null,
  toolId: null,
  meshDisplay: "shaded",
  showGrid: true,
  cursor: null,
};
