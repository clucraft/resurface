/** Typed access to the Electron preload bridge (undefined when running in a browser). */
export interface DesktopFile {
  name: string;
  path: string;
  data?: Uint8Array;
  error?: string;
}

export interface DesktopBridge {
  isDesktop: true;
  platform: string;
  openFiles(opts?: { title?: string; filters?: { name: string; extensions: string[] }[]; multiple?: boolean }): Promise<DesktopFile[]>;
  saveFile(opts: { suggestedName: string; data: string | Uint8Array; filters?: { name: string; extensions: string[] }[] }): Promise<boolean>;
  setTitle(title: string): void;
  onMenu(cb: (command: string) => void): () => void;
  onOpenFiles(cb: (files: DesktopFile[]) => void): () => void;
}

export const desktop: DesktopBridge | undefined = (globalThis as { resurfaceDesktop?: DesktopBridge }).resurfaceDesktop;

/** Convert a file delivered by the desktop shell into a DOM File. */
export function toFile(f: DesktopFile): File {
  return new File([f.data as BlobPart], f.name);
}
