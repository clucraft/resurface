/**
 * Project files (.rsf): an 8-byte magic, a little-endian u32 JSON length, the JSON manifest
 * and then the raw bytes of every typed array referenced from the manifest (8-byte aligned).
 */
import type { App } from "../app/App";
import type { Entity, MeshEntity } from "../core/entities";
import { markPrepared } from "../app/MeshRuntime";
import { toShared } from "../app/shared";
import { pickFiles, saveFile, baseName } from "./files";

const MAGIC = "RSURF001";
const TYPES = { Float32Array, Float64Array, Uint32Array, Int32Array, Uint8Array, Uint16Array } as const;
type TAName = keyof typeof TYPES;

export function serializeProject(entities: Entity[]): Uint8Array {
  const blobs: ArrayBufferView[] = [];
  let offset = 0;
  const replacer = (_k: string, v: unknown) => {
    if (ArrayBuffer.isView(v)) {
      const name = v.constructor.name as TAName;
      if (!(name in TYPES)) return undefined;
      const ref = { $ta: name, off: offset, len: (v as Float32Array).length };
      blobs.push(v);
      offset += Math.ceil(v.byteLength / 8) * 8;
      return ref;
    }
    return v;
  };
  const json = JSON.stringify({ version: 1, app: "Resurface", entities }, replacer);
  const jsonBytes = new TextEncoder().encode(json);
  const headerLen = Math.ceil((12 + jsonBytes.length) / 8) * 8;
  const out = new Uint8Array(headerLen + offset);
  out.set(new TextEncoder().encode(MAGIC), 0);
  new DataView(out.buffer).setUint32(8, jsonBytes.length, true);
  out.set(jsonBytes, 12);
  let o = headerLen;
  for (const b of blobs) {
    out.set(new Uint8Array(b.buffer, b.byteOffset, b.byteLength), o);
    o += Math.ceil(b.byteLength / 8) * 8;
  }
  return out;
}

export function deserializeProject(bytes: Uint8Array): Entity[] {
  const magic = new TextDecoder().decode(bytes.subarray(0, 8));
  if (magic !== MAGIC) throw new Error("Not a Resurface project file");
  const len = new DataView(bytes.buffer, bytes.byteOffset).getUint32(8, true);
  const json = new TextDecoder().decode(bytes.subarray(12, 12 + len));
  const headerLen = Math.ceil((12 + len) / 8) * 8;
  const reviver = (_k: string, v: unknown) => {
    if (v && typeof v === "object" && "$ta" in (v as object)) {
      const { $ta, off, len: n } = v as { $ta: TAName; off: number; len: number };
      const Ctor = TYPES[$ta];
      const start = bytes.byteOffset + headerLen + off;
      // copy out (also guarantees alignment)
      const copy = new Ctor(n);
      new Uint8Array(copy.buffer).set(new Uint8Array(bytes.buffer, start, n * Ctor.BYTES_PER_ELEMENT));
      return toShared(copy);
    }
    return v;
  };
  const data = JSON.parse(json, reviver) as { entities: Entity[] };
  return data.entities;
}

let lastName = "project";

export async function saveProject(app: App): Promise<void> {
  void app.endTool(false);
  const bytes = await app.busy("Saving project", () => serializeProject(app.doc.all()));
  if (bytes && (await saveFile(bytes, `${lastName}.rsf`, "application/octet-stream"))) app.toast("success", "Project saved");
}

export async function openProject(app: App): Promise<void> {
  const [file] = await pickFiles(".rsf", false);
  if (!file) return;
  await openProjectFile(app, file);
}

export async function openProjectFile(app: App, file: File): Promise<void> {
  await app.busy(`Opening ${file.name}`, async () => {
    const entities = deserializeProject(new Uint8Array(await file.arrayBuffer()));
    for (const e of entities) if (e.kind === "mesh") markPrepared((e as MeshEntity).mesh);
    await app.endTool(false);
    app.doc.reset(entities);
    lastName = baseName(file.name);
  });
  app.viewer?.fitAll("iso");
}
