/** File dialogs and downloads (native dialogs in the desktop app, DOM fallbacks in browsers). */
import { desktop, toFile } from "../app/desktop";

const FILTER_NAMES: Record<string, string> = {
  stl: "3D scans",
  obj: "3D scans",
  ply: "3D scans",
  ptx: "3D scans",
  rsf: "Resurface project",
  step: "STEP",
  igs: "IGES",
};

function filtersFor(accept: string) {
  const exts = accept.split(",").map((e) => e.trim().replace(/^\./, "")).filter(Boolean);
  const name = FILTER_NAMES[exts[0]] ?? "Files";
  return [{ name, extensions: exts }];
}

export async function pickFiles(accept: string, multiple = true): Promise<File[]> {
  if (desktop) {
    const files = await desktop.openFiles({ filters: filtersFor(accept), multiple });
    return files.filter((f) => f.data).map(toFile);
  }
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = accept;
    input.multiple = multiple;
    input.style.display = "none";
    let settled = false;
    input.onchange = () => {
      settled = true;
      resolve(input.files ? [...input.files] : []);
      input.remove();
    };
    // resolve empty if the dialog is cancelled
    window.addEventListener(
      "focus",
      () =>
        setTimeout(() => {
          if (!settled) {
            resolve([]);
            input.remove();
          }
        }, 800),
      { once: true },
    );
    document.body.appendChild(input);
    input.click();
  });
}

/** Save data to disk. Uses the File System Access API when available, else a download. */
export async function saveFile(data: BlobPart | Uint8Array<ArrayBufferLike>, suggestedName: string, mime = "application/octet-stream"): Promise<boolean> {
  // Blobs cannot be built from SharedArrayBuffer views: copy those first
  const part = data instanceof Uint8Array && !(data.buffer instanceof ArrayBuffer) ? new Uint8Array(data) : data;
  if (desktop) {
    const ext = suggestedName.includes(".") ? suggestedName.slice(suggestedName.lastIndexOf(".") + 1) : "";
    const payload = typeof part === "string" ? part : part instanceof Uint8Array ? part : new Uint8Array(await new Blob([part as BlobPart]).arrayBuffer());
    return desktop.saveFile({ suggestedName, data: payload, filters: ext ? [{ name: `${ext.toUpperCase()} file`, extensions: [ext] }] : undefined });
  }
  const blob = part instanceof Blob ? part : new Blob([part as BlobPart], { type: mime });
  const w = window as unknown as {
    showSaveFilePicker?: (o: unknown) => Promise<{ createWritable: () => Promise<{ write: (b: Blob) => Promise<void>; close: () => Promise<void> }> }>;
  };
  if (w.showSaveFilePicker) {
    try {
      const ext = suggestedName.includes(".") ? suggestedName.slice(suggestedName.lastIndexOf(".")) : "";
      const handle = await w.showSaveFilePicker({
        suggestedName,
        types: ext ? [{ description: `${ext.slice(1).toUpperCase()} file`, accept: { [mime]: [ext] } }] : undefined,
      });
      const writable = await handle.createWritable();
      await writable.write(blob);
      await writable.close();
      return true;
    } catch (e) {
      if ((e as Error).name === "AbortError") return false;
      // fall through to download
    }
  }
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = suggestedName;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
  return true;
}

export const baseName = (name: string) => name.replace(/\.[^.]+$/, "");
