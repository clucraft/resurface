/**
 * Resurface desktop shell (Electron main process).
 *
 * The built web app (dist/) is served from a privileged custom protocol (app://resurface/) so we
 * can send the cross-origin isolation headers that enable SharedArrayBuffer for the compute
 * workers. Native menus and file dialogs are bridged to the renderer through preload.cjs.
 */
import { app, BrowserWindow, Menu, dialog, ipcMain, net, protocol, shell } from "electron";
import path from "node:path";
import fs from "node:fs/promises";
import { fileURLToPath, pathToFileURL } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DIST = path.join(__dirname, "..", "dist");
const DEV_URL = process.env.RESURFACE_DEV_URL; // e.g. http://localhost:5173 for live development
const SCAN_EXTENSIONS = ["stl", "obj", "ply", "ptx"];

// Big scans: give V8 room (typed arrays live outside the heap, but parsing creates garbage).
app.commandLine.appendSwitch("js-flags", "--max-old-space-size=8192");

protocol.registerSchemesAsPrivileged([
  {
    scheme: "app",
    privileges: { standard: true, secure: true, supportFetchAPI: true, stream: true, corsEnabled: true, codeCache: true },
  },
]);

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".wasm": "application/wasm",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".json": "application/json",
};

function registerAppProtocol() {
  protocol.handle("app", async (request) => {
    const url = new URL(request.url);
    let rel = decodeURIComponent(url.pathname);
    if (!rel || rel === "/") rel = "/index.html";
    const file = path.normalize(path.join(DIST, rel));
    if (!file.startsWith(DIST)) return new Response("Forbidden", { status: 403 });
    const res = await net.fetch(pathToFileURL(file).toString());
    const headers = new Headers(res.headers);
    headers.set("Cross-Origin-Opener-Policy", "same-origin");
    headers.set("Cross-Origin-Embedder-Policy", "require-corp");
    headers.set("Cross-Origin-Resource-Policy", "same-origin");
    const type = MIME[path.extname(file).toLowerCase()];
    if (type) headers.set("Content-Type", type);
    return new Response(res.body, { status: res.status, headers });
  });
}

/** @type {BrowserWindow | null} */
let mainWindow = null;
/** files passed on the command line / via "Open with" before the window was ready */
let pendingOpen = [];

function filesFromArgv(argv) {
  return argv.slice(1).filter((a) => {
    const ext = path.extname(a).slice(1).toLowerCase();
    return (SCAN_EXTENSIONS.includes(ext) || ext === "rsf") && !a.startsWith("-");
  });
}

async function readForRenderer(paths) {
  const out = [];
  for (const p of paths) {
    try {
      const data = await fs.readFile(p);
      out.push({ name: path.basename(p), path: p, data: new Uint8Array(data.buffer, data.byteOffset, data.byteLength) });
    } catch (e) {
      out.push({ name: path.basename(p), path: p, error: String(e) });
    }
  }
  return out;
}

function sendMenu(command) {
  mainWindow?.webContents.send("menu", command);
}

function buildMenu() {
  const item = (label, command, accelerator) => ({ label, accelerator, registerAccelerator: false, click: () => sendMenu(command) });
  /** @type {Electron.MenuItemConstructorOptions[]} */
  const template = [
    {
      label: "&File",
      submenu: [
        item("New Project", "newProject"),
        item("Open Project…", "openProject", "Ctrl+O"),
        item("Save Project…", "saveProject", "Ctrl+S"),
        { type: "separator" },
        item("Import Scan…", "import", "Ctrl+I"),
        {
          label: "Samples",
          submenu: [item("Mechanical bracket", "sample:bracket"), item("Turned shaft", "sample:turned"), item("Organic handle", "sample:organic"), item("Bumper grille", "sample:grille"), item("Mounting plate (CAD STL)", "sample:cadPlate")],
        },
        { type: "separator" },
        item("Export STEP…", "exportStep"),
        item("Export IGES…", "exportIges"),
        item("Export STL (CAD)…", "exportStlCad"),
        item("Export STL (Scan)…", "exportStlMesh"),
        { type: "separator" },
        { role: "quit", label: "Exit" },
      ],
    },
    {
      label: "&Edit",
      submenu: [item("Undo", "undo", "Ctrl+Z"), item("Redo", "redo", "Ctrl+Y"), { type: "separator" }, { role: "copy" }, { role: "paste" }],
    },
    {
      label: "&View",
      submenu: [
        item("Fit All", "view:fit", "F"),
        item("Isometric", "view:iso"),
        item("Top", "view:top"),
        item("Front", "view:front"),
        item("Right", "view:right"),
        { type: "separator" },
        { role: "togglefullscreen" },
        { role: "resetZoom" },
        { role: "zoomIn" },
        { role: "zoomOut" },
        { type: "separator" },
        { role: "reload" },
        { role: "toggleDevTools" },
      ],
    },
    {
      label: "&Help",
      submenu: [
        item("About Resurface", "about"),
        { label: "Open Log Folder", click: () => shell.openPath(app.getPath("logs")) },
      ],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

const TOUR_DIR = process.env.RESURFACE_TOUR_DIR;
const TOUR_SCRIPT = process.env.RESURFACE_TOUR_SCRIPT;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1600,
    height: 1000,
    useContentSize: !!TOUR_DIR,
    minWidth: 1100,
    minHeight: 700,
    backgroundColor: "#0d1117",
    title: "Resurface",
    icon: path.join(__dirname, "..", "build", "icon.png"),
    show: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      sandbox: false,
      nodeIntegration: false,
    },
  });
  mainWindow.once("ready-to-show", () => {
    if (!TOUR_DIR) mainWindow?.maximize();
    mainWindow?.show();
  });
  mainWindow.webContents.on("did-finish-load", async () => {
    if (pendingOpen.length) {
      mainWindow?.webContents.send("open-files", await readForRenderer(pendingOpen));
      pendingOpen = [];
    }
  });
  // external links open in the default browser
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:/.test(url)) void shell.openExternal(url);
    return { action: "deny" };
  });
  void mainWindow.loadURL(DEV_URL ?? "app://resurface/index.html");
  mainWindow.on("closed", () => (mainWindow = null));
  if (process.env.RESURFACE_SMOKE) void smokeTest(mainWindow, process.env.RESURFACE_SMOKE);
  if (TOUR_DIR && TOUR_SCRIPT) void runTour(mainWindow, TOUR_DIR, TOUR_SCRIPT);
}

/**
 * Documentation tour (RESURFACE_TOUR_DIR + RESURFACE_TOUR_SCRIPT): runs a renderer script that
 * drives the UI and calls window.resurfaceTour.capture(name) to save screenshots, then quits.
 */
async function runTour(win, outDir, scriptPath) {
  await fs.mkdir(outDir, { recursive: true });
  ipcMain.handle("tour:capture", async (_e, name) => {
    const img = await win.webContents.capturePage();
    await fs.writeFile(path.join(outDir, `${name}.png`), img.toPNG());
    return true;
  });
  const log = [];
  try {
    await new Promise((r) => win.webContents.once("did-finish-load", r));
    await new Promise((r) => setTimeout(r, 3000));
    const script = await fs.readFile(scriptPath, "utf8");
    const result = await win.webContents.executeJavaScript(script);
    log.push({ result });
  } catch (e) {
    log.push({ error: String(e) });
  }
  await fs.writeFile(path.join(outDir, "tour.json"), JSON.stringify(log, null, 2));
  app.quit();
}

/**
 * Automated smoke test (RESURFACE_SMOKE=<output dir>): wait for the UI, load a sample, start the
 * CAD kernel, capture a screenshot and a JSON report, then quit.
 */
async function smokeTest(win, outDir) {
  const report = {};
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  try {
    await new Promise((r) => win.webContents.once("did-finish-load", r));
    await wait(2500);
    report.env = await win.webContents.executeJavaScript(
      `({ isolated: crossOriginIsolated, sab: typeof SharedArrayBuffer !== "undefined", desktop: !!window.resurfaceDesktop, title: document.title })`,
    );
    report.flow = await win.webContents.executeJavaScript(`(async () => {
      const buttons = () => [...document.querySelectorAll("button, .item")];
      const click = (t) => { const b = buttons().find((x) => x.innerText.trim() === t); if (b) b.click(); return !!b; };
      const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
      const out = {};
      out.sampleMenu = click("Load sample");
      await sleep(200);
      const item = buttons().find((x) => x.innerText.includes("Mechanical bracket"));
      if (item) item.click();
      for (let i = 0; i < 100 && !document.querySelector(".tree-row"); i++) await sleep(100);
      out.treeRows = document.querySelectorAll(".tree-row").length;
      try {
        const k = await globalThis.resurface.getKernel();
        const frame = { origin: [0, 0, 0], xAxis: [1, 0, 0], yAxis: [0, 1, 0], zAxis: [0, 0, 1] };
        const id = await k.extrude({ frame, profile: { loops: [[{ type: "circle", center: [0, 0], radius: 5 }]] }, distance: 10 });
        const info = await k.info(id);
        const step = await k.exportSTEP([{ id }]);
        out.kernel = { volume: info.volume, valid: info.valid, stepBytes: step.length };
      } catch (e) {
        out.kernelError = String(e);
      }
      return out;
    })()`);
    await wait(1500);
    const img = await win.webContents.capturePage();
    await fs.mkdir(outDir, { recursive: true });
    await fs.writeFile(path.join(outDir, "smoke.png"), img.toPNG());
  } catch (e) {
    report.error = String(e);
  }
  await fs.mkdir(outDir, { recursive: true });
  await fs.writeFile(path.join(outDir, "smoke.json"), JSON.stringify(report, null, 2));
  app.quit();
}

// ---------------------------------------------------------------------------
// IPC: native dialogs and file IO
// ---------------------------------------------------------------------------

ipcMain.handle("dialog:open", async (_e, opts = {}) => {
  const res = await dialog.showOpenDialog(mainWindow, {
    title: opts.title ?? "Open",
    filters: opts.filters ?? [{ name: "3D scans", extensions: SCAN_EXTENSIONS }],
    properties: opts.multiple === false ? ["openFile"] : ["openFile", "multiSelections"],
  });
  if (res.canceled) return [];
  return readForRenderer(res.filePaths);
});

ipcMain.handle("dialog:save", async (_e, { suggestedName, data, filters }) => {
  const res = await dialog.showSaveDialog(mainWindow, { defaultPath: suggestedName, filters });
  if (res.canceled || !res.filePath) return false;
  await fs.writeFile(res.filePath, typeof data === "string" ? data : Buffer.from(data));
  return true;
});

ipcMain.on("window:title", (_e, title) => mainWindow?.setTitle(title));

// ---------------------------------------------------------------------------
// App lifecycle
// ---------------------------------------------------------------------------

if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on("second-instance", async (_e, argv) => {
    const files = filesFromArgv(argv);
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
      if (files.length) mainWindow.webContents.send("open-files", await readForRenderer(files));
    }
  });
  pendingOpen = filesFromArgv(process.argv);
  app.whenReady().then(() => {
    registerAppProtocol();
    buildMenu();
    createWindow();
  });
  app.on("window-all-closed", () => app.quit());
}
