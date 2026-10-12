/** Mesh / point-cloud file I/O: STL, OBJ, PLY, PTX readers, binary STL writer, import dispatch. */
export {
  parseSTL,
  createSTLStreamParser,
  writeSTLBinary,
  detectSTLFormat,
  STL_DETECT_BYTES,
  type STLStreamParser,
  type STLStreamOptions,
} from "./stl";
export { parseOBJ, objToImportResult } from "./obj";
export {
  parsePLY,
  parsePLYHeader,
  type PLYResult,
  type PLYHeader,
  type PLYElement,
  type PLYProperty,
  type PLYFormat,
} from "./ply";
export { parsePTX, triangulateGrid, type PTXOptions } from "./ptx";
export { importMeshFile, fileExtension, SUPPORTED_MESH_EXTENSIONS, type MeshImportResult } from "./import";
export { ByteScanner, parseNumberString, toBytes } from "./text";
