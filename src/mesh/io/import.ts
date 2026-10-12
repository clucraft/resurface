/**
 * File-type dispatch for mesh / point-cloud import.
 */
import type { MeshData, PointCloudData, ProgressFn } from "../../core/types";
import { objToImportResult, parseOBJ } from "./obj";
import { parsePLY } from "./ply";
import { parsePTX } from "./ptx";
import { parseSTL } from "./stl";

export type MeshImportResult = { kind: "mesh"; mesh: MeshData } | { kind: "points"; points: PointCloudData };

/** Lower-case file extensions (with dot) understood by {@link importMeshFile}. */
export const SUPPORTED_MESH_EXTENSIONS: readonly string[] = [".stl", ".obj", ".ply", ".ptx"];

/** Lower-cased extension including the dot ("" when there is none). */
export function fileExtension(fileName: string): string {
  const base = fileName.replace(/^.*[\\/]/, "");
  const dot = base.lastIndexOf(".");
  return dot < 0 ? "" : base.slice(dot).toLowerCase();
}

/**
 * Import a mesh or point cloud, dispatching on the file extension (.stl .obj .ply .ptx).
 * OBJ / PLY without faces and PTX without any triangle give a point cloud. PTX intensity is
 * mapped to grey colours when the scan has no RGB.
 * @throws Error for unsupported extensions or malformed headers.
 */
export function importMeshFile(fileName: string, data: ArrayBuffer, onProgress?: ProgressFn): MeshImportResult {
  const ext = fileExtension(fileName);
  switch (ext) {
    case ".stl":
      return { kind: "mesh", mesh: parseSTL(data, onProgress) };
    case ".obj":
      return objToImportResult(parseOBJ(data, onProgress));
    case ".ply":
      return parsePLY(data, onProgress);
    case ".ptx": {
      const mesh = parsePTX(data, { onProgress, intensityToGray: true });
      if (mesh.indices.length > 0) return { kind: "mesh", mesh };
      const points: PointCloudData = { positions: mesh.positions };
      if (mesh.colors) points.colors = mesh.colors;
      return { kind: "points", points };
    }
    default:
      throw new Error(
        `Cannot import "${fileName}": unsupported file type ${ext ? `"${ext}"` : "(no extension)"}. ` +
          `Supported formats: ${SUPPORTED_MESH_EXTENSIONS.join(", ")}.`,
      );
  }
}
