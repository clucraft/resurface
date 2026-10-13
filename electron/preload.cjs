/**
 * Preload bridge: exposes a small, explicit API for native dialogs and menu commands to the
 * renderer (window.resurfaceDesktop). No Node APIs are exposed directly.
 */
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("resurfaceDesktop", {
  isDesktop: true,
  platform: process.platform,
  /** opts: { title?, filters?: {name, extensions[]}[], multiple? } → [{ name, path, data: Uint8Array }] */
  openFiles: (opts) => ipcRenderer.invoke("dialog:open", opts),
  /** { suggestedName, data: string | Uint8Array, filters? } → boolean (saved) */
  saveFile: (opts) => ipcRenderer.invoke("dialog:save", opts),
  setTitle: (title) => ipcRenderer.send("window:title", title),
  onMenu: (cb) => {
    const handler = (_e, command) => cb(command);
    ipcRenderer.on("menu", handler);
    return () => ipcRenderer.off("menu", handler);
  },
  onOpenFiles: (cb) => {
    const handler = (_e, files) => cb(files);
    ipcRenderer.on("open-files", handler);
    return () => ipcRenderer.off("open-files", handler);
  },
});

// Documentation tours only (RESURFACE_TOUR_DIR set): screenshot hook for the tour script.
if (process.env.RESURFACE_TOUR_DIR) {
  contextBridge.exposeInMainWorld("resurfaceTour", {
    capture: (name) => ipcRenderer.invoke("tour:capture", name),
  });
}
