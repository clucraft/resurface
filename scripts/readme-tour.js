/*
 * README screenshots: Auto Convert on a machined housing, then the scan workflow on the built-in
 * samples. Executed inside the desktop app's renderer by
 *   RESURFACE_DEV_URL=http://localhost:5174 RESURFACE_TOUR_SIZE=1100x700 RESURFACE_TOUR_DIR=docs/screenshots RESURFACE_TOUR_SCRIPT=scripts/readme-tour.js npx electron .
 * (needs the dev server, which loads the rsTest automation kit). Evaluates to a JSON log.
 */
(async () => {
  const T = window.rsTest;
  const a = T.app;
  const log = [];
  const sleep = T.sleep;
  const V = a.viewer.camera.position.constructor;
  const reg = await import("/src/tools/registry.ts");
  const settle = async (ms = 500) => {
    await sleep(ms);
    a.ui.patch({ toasts: [] });
    a.viewer.renderNow();
    await sleep(150);
  };
  /** whole window, or just the 3D viewport */
  const shot = async (name, ms = 500, viewportOnly = false) => {
    await settle(ms);
    let rect;
    if (viewportOnly) {
      const r = a.viewer.domElement.getBoundingClientRect();
      rect = { x: Math.round(r.left), y: Math.round(r.top), width: Math.round(r.width), height: Math.round(r.height) };
    }
    await window.resurfaceTour.capture(name, rect);
    log.push(`captured ${name}`);
  };
  const view = (center, radius, dir, up = [0, 0, 1]) => {
    a.viewer.controls.fitSphere(new V(...center), radius, new V(...dir).normalize(), new V(...up));
  };
  const step = (msg) => log.push(msg);
  const fresh = async () => {
    await a.endTool(false);
    a.setCompare(false);
    a.doc.reset([]);
    await sleep(300);
  };
  const divider = (visible) => {
    const d = document.querySelector(".compare-divider");
    if (d) d.style.visibility = visible ? "" : "hidden";
  };

  /** A machined housing (planes, cylinders, countersink cones, a spherical dome and a fillet torus),
   * built with the CAD kernel and exported as an STL, the kind of file Auto Convert is for. */
  const housingStl = async () => {
    const k = await a.getKernel();
    const made = [];
    const keep = async (p) => {
      const id = await p;
      made.push(id);
      return id;
    };
    const WORLD = { origin: [0, 0, 0], xAxis: [1, 0, 0], yAxis: [0, 1, 0], zAxis: [0, 0, 1] };
    const rect = (x0, y0, x1, y1) => {
      const p = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
      return p.map((q, i) => ({ type: "line", a: q, b: p[(i + 1) % 4] }));
    };
    const circle = (cx, cy, r) => [{ type: "circle", center: [cx, cy], radius: r }];
    const edgesWhere = async (id, pred) => {
      const t = await k.tessellate(id, { linearDeflection: 0.1, angularDeflectionDeg: 20 });
      return t.edges
        .filter((e) => {
          const pts = [];
          for (let i = 0; i < e.points.length; i += 3) pts.push([e.points[i], e.points[i + 1], e.points[i + 2]]);
          return pts.length > 1 && pred(pts);
        })
        .map((e) => e.id);
    };
    try {
      let part = await keep(k.extrude({ frame: WORLD, profile: { loops: [rect(-45, -30, 45, 30)] }, distance: 10, solid: true }));
      const corners = await edgesWhere(part, (p) => Math.abs(p[0][0] - p.at(-1)[0]) < 1e-6 && Math.abs(p[0][1] - p.at(-1)[1]) < 1e-6);
      part = await keep(k.fillet(part, corners, 10));
      // turned boss: cylinder, cone and a spherical dome (R20 centred 16 mm up)
      const XZ = { origin: [0, 0, 0], xAxis: [1, 0, 0], yAxis: [0, 0, 1], zAxis: [0, -1, 0] };
      const half = Math.atan2(16, 12) / 2;
      const profile = [
        { type: "line", a: [0, 5], b: [22, 5] },
        { type: "line", a: [22, 5], b: [22, 20] },
        { type: "line", a: [22, 20], b: [16, 28] },
        { type: "arc", start: [16, 28], mid: [20 * Math.sin(half), 16 + 20 * Math.cos(half)], end: [0, 36] },
        { type: "line", a: [0, 36], b: [0, 5] },
      ];
      const boss = await keep(k.revolve({ frame: XZ, profile: { loops: [profile] }, axisOrigin: [0, 0, 0], axisDir: [0, 0, 1], angleDeg: 360, solid: true }));
      part = await keep(k.boolean("union", part, boss, { simplify: true }));
      const foot = await edgesWhere(part, (p) => p.every((q) => Math.abs(q[2] - 10) < 1e-6 && Math.abs(Math.hypot(q[0], q[1]) - 22) < 1e-4));
      part = await keep(k.fillet(part, foot, 4));
      // four countersunk holes, a bore and a cross hole
      const holes = [[-33, -18], [33, -18], [-33, 18], [33, 18]];
      const drills = await keep(k.extrude({ frame: { ...WORLD, origin: [0, 0, -1] }, profile: { loops: holes.map(([x, y]) => circle(x, y, 3.5)) }, distance: 30, solid: true }));
      part = await keep(k.boolean("cut", part, drills));
      for (const [x, y] of holes) {
        const frame = { origin: [x, y, 0], xAxis: [1, 0, 0], yAxis: [0, 0, 1], zAxis: [0, -1, 0] };
        const tool = [
          { type: "line", a: [0, 6.5], b: [3.5, 6.5] },
          { type: "line", a: [3.5, 6.5], b: [7.5, 10.5] },
          { type: "line", a: [7.5, 10.5], b: [0, 10.5] },
          { type: "line", a: [0, 10.5], b: [0, 6.5] },
        ];
        const sink = await keep(k.revolve({ frame, profile: { loops: [tool] }, axisOrigin: [x, y, 0], axisDir: [0, 0, 1], angleDeg: 360, solid: true }));
        part = await keep(k.boolean("cut", part, sink));
      }
      const bore = await keep(k.extrude({ frame: { ...WORLD, origin: [0, 0, -1] }, profile: { loops: [circle(0, 0, 6)] }, distance: 60, solid: true }));
      part = await keep(k.boolean("cut", part, bore));
      const YZ = { origin: [-40, 0, 14], xAxis: [0, 1, 0], yAxis: [0, 0, 1], zAxis: [1, 0, 0] };
      const cross = await keep(k.extrude({ frame: YZ, profile: { loops: [circle(0, 0, 3)] }, distance: 80, solid: true }));
      part = await keep(k.boolean("cut", part, cross));
      return await k.exportSTL([{ id: part, name: "Housing" }], { linearDeflection: 0.03, angularDeflectionDeg: 15 });
    } finally {
      for (const id of made) void k.release(id);
    }
  };

  try {
    // ================= Auto Convert
    await fresh();
    const stl = await housingStl();
    await reg.importFiles(a, [new File([stl], "housing.stl")]);
    await T.waitIdle();
    await reg.executeCommand(a, "autoConvert");
    await sleep(500);
    await T.waitIdle(120000);
    for (let i = 0; i < 1200 && a.activeTool?.s?.running; i++) await sleep(100);
    const conv = a.activeTool;
    step(`auto convert: ${JSON.stringify(conv.s.result)}`);
    view([0, 0, 16], 56, [1, -1.25, 0.95]);
    a.setCompare(true, 0.5);
    await shot("auto-convert", 1200);

    // before / after from the same camera: the STL with its triangles, the solid by face type
    view([0, 0, 16], 54, [-0.9, -1.2, 0.85]);
    a.setCompare(true, 1);
    divider(false);
    await shot("auto-convert-stl", 800, true);
    a.setCompare(true, 0);
    await shot("auto-convert-cad", 800, true);
    divider(true);
    a.setCompare(false);
    T.click("Create body");
    await sleep(1500);
    const meshId = a.doc.ofKind("mesh")[0].id;
    a.setVisible(meshId, false);
    T.click("Export");
    await sleep(200);
    a.setActiveEntity(a.doc.ofKind("body").at(-1).id);
    view([0, 0, 16], 56, [1, -1.25, 0.95]);
    await shot("auto-convert-export", 800);

    // ================= Mechanical bracket: smart select + fitted primitives
    await fresh();
    T.click("Mesh");
    await reg.loadSample(a, "bracket");
    await T.waitIdle();
    view([0, 0, 30], 80, [1, -1.2, 0.9]);
    await settle(300);
    T.key("5");
    const fitAt = async (point, command, name) => {
      await T.clickWorld(point);
      await sleep(600);
      T.click("Reference");
      await sleep(100);
      T.click(command);
      await T.waitIdle();
      await sleep(300);
      const p = a.doc.ofKind("primitive").at(-1);
      a.rename(p.id, name);
      step(`${name}: RMS ${p.stats?.rms}`);
      return p;
    };
    const bossPoint = [20 + 15 * Math.cos(-1), 15 * Math.sin(-1), 22];
    const boss = await fitAt(bossPoint, "Cylinder", "Boss");
    await fitAt([45, 10, 12], "Plane", "Base");
    await fitAt([-40, -20, 40], "Plane", "Wall");
    await fitAt([20 - 5.3, 5.3, 24], "Cylinder", "Bore");
    view([0, 0, 30], 72, [1, -1.2, 0.9]);
    await settle(300);
    await T.clickWorld(bossPoint);
    await sleep(600);
    a.setActiveEntity(boss.id);
    await shot("smart-select-fit", 800);

    // ================= Turned shaft: section sketch with auto fit, revolve
    await fresh();
    T.click("Mesh");
    await reg.loadSample(a, "turned");
    await T.waitIdle();
    T.click("Sketch");
    await sleep(100);
    T.click("2D Sketch");
    await sleep(500);
    a.activeTool.set({ source: "world:ZX", offset: 0 });
    a.activeTool.activate();
    await sleep(300);
    T.click("Start sketch");
    await sleep(1500);
    const ed = a.activeTool;
    const sk = a.doc.ofKind("sketch").at(-1);
    a.viewer.lookAlong(sk.frame.zAxis, sk.frame.yAxis, [0, 0, 40], 58);
    await settle(300);
    ed.setMode("autoFit");
    await sleep(200);
    const p0 = T.toScreen([-40, 0, -8]), p1 = T.toScreen([40, 0, 90]);
    await T.drag([p0.x, p0.y], [p1.x, p1.y], 10);
    await sleep(1500);
    ed.setMode("select");
    step(a.ui.get().status);
    view([0, 0, 40], 50, [0.25, -1, 0.3]);
    await shot("sketch-auto-fit", 800);
    T.click("Finish sketch");
    await sleep(500);
    T.click("Model");
    await sleep(150);
    T.click("Revolve");
    await sleep(1500);
    await T.waitIdle();
    const shaftId = a.doc.ofKind("mesh")[0].id;
    a.setVisible(shaftId, false);
    view([0, 0, 40], 52, [1, -1.3, 0.7]);
    await shot("revolve", 1500);
    a.setVisible(shaftId, true);
    T.click("OK");
    await sleep(2500);
    await T.waitIdle();
    for (const s of a.doc.ofKind("sketch")) a.setVisible(s.id, false);
    a.setActiveEntity(null);
    T.click("Analyze");
    await sleep(150);
    T.click("Deviation");
    await sleep(800);
    await T.waitIdle();
    const dv = a.activeTool;
    dv.set({ maxDistance: 0.5, tolerance: 0.02 });
    await dv.run();
    step(`shaft deviation: ${JSON.stringify(dv.s.stats)}`);
    view([0, 0, 40], 52, [1, -1.3, 0.7]);
    await shot("deviation", 1200);
    await a.endTool(false);

    // ================= Bumper grille (scanned at a tilt): object align from a slat top and the side
    await fresh();
    T.click("Mesh");
    await reg.loadSample(a, "grille");
    await T.waitIdle();
    // the sample is generated tilted; these map design coordinates to the raw scan
    const ax = (8 * Math.PI) / 180, az = (-6 * Math.PI) / 180;
    const toScan = (x, y, z) => {
      const y1 = y * Math.cos(ax) - z * Math.sin(ax), z1 = y * Math.sin(ax) + z * Math.cos(ax);
      return [x * Math.cos(az) - y1 * Math.sin(az) + 12, x * Math.sin(az) + y1 * Math.cos(az) - 30, z1 + 8];
    };
    const dirScan = (x, y, z) => {
      const o = toScan(0, 0, 0), p = toScan(x, y, z);
      return [p[0] - o[0], p[1] - o[1], p[2] - o[2]];
    };
    T.key("5");
    view(toScan(30, 7, 63), 55, dirScan(0.15, -0.55, 0.82), dirScan(0, 0, 1));
    await settle(300);
    await T.clickWorld(toScan(30, 4.5, 67));
    await sleep(500);
    T.click("Reference");
    await sleep(100);
    T.click("Plane");
    await T.waitIdle();
    const top = a.doc.ofKind("primitive").at(-1);
    a.rename(top.id, "Slat top");
    // the frame's end face (the front is one curved surface, no good for a plane)
    view(toScan(150, 7, 70), 60, dirScan(1, -0.3, 0.2), dirScan(0, 0, 1));
    await settle(300);
    await T.clickWorld(toScan(150, 7.5 + (150 * 150) / 2400, 70));
    await sleep(500);
    T.click("Plane");
    await T.waitIdle();
    const side = a.doc.ofKind("primitive").at(-1);
    a.rename(side.id, "Side");
    step(`grille planes: ${a.doc.ofKind("primitive").map((p) => `${p.name} ${p.stats?.rms}`).join(", ")}`);
    a.setActiveEntity(null);
    T.click("Align");
    await sleep(100);
    T.click("Object Align");
    await sleep(400);
    const oa = a.activeTool;
    oa.set({ primary: top.id, primaryTarget: "Z", flipPrimary: false, secondary: side.id, secondaryTarget: "X", flipSecondary: false });
    oa.activate();
    view(toScan(0, 7, 70), 175, dirScan(0.55, -1, 0.6), dirScan(0, 0, 1));
    await shot("object-align", 900);
    await a.endTool(false);

    // ================= Organic handle: Autosurface
    await fresh();
    T.click("Mesh");
    await reg.loadSample(a, "organic");
    await T.waitIdle();
    a.viewer.fitAll("iso");
    T.click("Freeform");
    await sleep(150);
    T.click("Autosurface");
    await sleep(800);
    await T.waitIdle(120000);
    await sleep(500);
    const at = a.activeTool;
    step(`autosurface: ${JSON.stringify({ patches: at.s.result?.patches, rms: at.s.result?.rms, max: at.s.result?.maxDev })}`);
    await shot("autosurface", 1000);
    await a.endTool(false);

    return { ok: true, log };
  } catch (e) {
    return { ok: false, error: String(e && e.stack ? e.stack : e), log };
  }
})();
