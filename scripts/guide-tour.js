/*
 * Documentation tour: runs the clip-on license plate bracket workflow on the "Bumper grille"
 * sample and captures a screenshot per step. Executed inside the desktop app's renderer by
 *   RESURFACE_DEV_URL=http://localhost:5174 RESURFACE_TOUR_DIR=<out> RESURFACE_TOUR_SCRIPT=scripts/guide-tour.js npx electron .
 * (needs the dev server, which loads the rsTest automation kit). Evaluates to a JSON log.
 */
(async () => {
  const T = window.rsTest;
  const a = T.app;
  const log = [];
  const sleep = T.sleep;
  const V = a.viewer.camera.position.constructor;
  const settle = async (ms = 500) => {
    await sleep(ms);
    a.viewer.renderNow();
    await sleep(120);
  };
  const shot = async (name, ms = 500) => {
    await settle(ms);
    await window.resurfaceTour.capture(name);
    log.push(`captured ${name}`);
  };
  const view = (center, radius, dir, up) => {
    a.viewer.controls.fitSphere(new V(...center), radius, new V(...dir).normalize(), up ? new V(...up) : undefined);
  };
  const step = (msg) => log.push(msg);

  // the sample is generated tilted; these map design coordinates to the raw scan (before alignment)
  const ax = (8 * Math.PI) / 180, az = (-6 * Math.PI) / 180;
  const toScan = (x, y, z) => {
    const y1 = y * Math.cos(ax) - z * Math.sin(ax), z1 = y * Math.sin(ax) + z * Math.cos(ax);
    return [x * Math.cos(az) - y1 * Math.sin(az) + 12, x * Math.sin(az) + y1 * Math.cos(az) - 30, z1 + 8];
  };
  const dirScan = (x, y, z) => {
    const o = toScan(0, 0, 0), p = toScan(x, y, z);
    return [p[0] - o[0], p[1] - o[1], p[2] - o[2]];
  };

  try {
    // 1 — start screen
    await shot("01-start", 800);

    // 2 — load the sample
    T.click("Load sample");
    await sleep(200);
    T.click("Bumper grille");
    await sleep(500);
    await T.waitIdle();
    await shot("02-grille-loaded", 800);

    // 3 — smart select the top face of a slat
    T.key("5");
    view(toScan(30, 7, 63), 55, dirScan(0.15, -0.55, 0.82), dirScan(0, 0, 1));
    await settle(300);
    await T.clickWorld(toScan(30, 4.5, 67));
    await sleep(300);
    step(`slat top selected: ${a.ui.get().selectedTriangles}`);
    await shot("03-select-slat-top");

    // 4 — fit a plane
    T.click("Reference");
    await sleep(100);
    T.click("Plane");
    await T.waitIdle();
    const top = a.doc.ofKind("primitive").at(-1);
    a.rename(top.id, "Slat top");
    await shot("04-plane-fitted", 700);

    // 5 — smart select the front of a slat (front view)
    view(toScan(30, 0, 63), 70, dirScan(0.1, -1, 0.15), dirScan(0, 0, 1));
    await settle(300);
    await T.clickWorld(toScan(30, 0, 63));
    await sleep(400);
    step(`front selected: ${a.ui.get().selectedTriangles}`);
    await shot("05-select-front");
    T.click("Plane");
    await T.waitIdle();
    const front = a.doc.ofKind("primitive").at(-1);
    a.rename(front.id, "Front");
    a.setActiveEntity(null);

    // 6 — object align
    T.click("Align");
    await sleep(100);
    T.click("Object Align");
    await sleep(400);
    const oa = a.activeTool;
    oa.set({ primary: top.id, primaryTarget: "Z", flipPrimary: false, secondary: front.id, secondaryTarget: "Y", flipSecondary: true });
    oa.activate();
    a.viewer.fitAll("iso");
    await shot("06-object-align", 700);
    T.click("Align", 1);
    await sleep(1500);
    // the planes did their job; hide them (eye icon in the Model list) for a clearer view
    a.setVisible(top.id, false);
    a.setVisible(front.id, false);
    a.setActiveEntity(null);
    a.viewer.fitAll("front");
    await shot("07-aligned-front-view", 800);

    // geometry after alignment
    const { meshBounds } = await import("/src/mesh/topology.ts");
    const bb = meshBounds(a.doc.ofKind("mesh")[0].mesh);
    const cx = (bb.min[0] + bb.max[0]) / 2, cz = (bb.min[2] + bb.max[2]) / 2;
    const xSec = cx + 30;
    step(`grille centre x=${cx.toFixed(1)} z=${cz.toFixed(1)}, section at x=${xSec.toFixed(1)}`);

    // 7 — section plane
    T.click("Sketch");
    await sleep(100);
    T.click("2D Sketch");
    await sleep(400);
    const st = a.activeTool;
    st.set({ source: "world:YZ", offset: xSec });
    st.activate();
    view([xSec, 0, cz], 120, [0.75, -0.45, 0.45], [0, 0, 1]);
    await shot("08-section-plane", 900);
    T.click("Start sketch");
    await sleep(1200);
    const ed = a.activeTool;
    const sk = a.doc.ofKind("sketch").at(-1);
    await shot("09-sketch-editor", 700);

    // find the middle slat in the section (closed loops ~8 mm tall)
    const slats = sk.sections
      .filter((s) => s.closed)
      .map((s) => {
        const us = s.points.map((p) => p[0]), vs = s.points.map((p) => p[1]);
        return { u0: Math.min(...us), u1: Math.max(...us), v0: Math.min(...vs), v1: Math.max(...vs) };
      })
      .filter((r) => r.v1 - r.v0 > 6 && r.v1 - r.v0 < 10)
      .sort((p, q) => p.v0 - q.v0);
    const s = slats[Math.floor(slats.length / 2)];
    const yf = s.u0, yb = s.u1, zt = s.v1;
    step(`slat: front ${yf.toFixed(2)} back ${yb.toFixed(2)} top ${zt.toFixed(2)}`);

    // 8 — draw the J-hook with the Line tool (Alt = no snapping, leaves the clearance gaps)
    ed.setMode("line");
    const f = sk.frame;
    const W = (u, v) => [0, 1, 2].map((i) => f.origin[i] + f.xAxis[i] * u + f.yAxis[i] * v);
    a.viewer.lookAlong(f.zAxis, f.yAxis, W(yf, zt - 10), 32);
    await settle(300);
    const hook = [
      [yf - 5, zt - 20],
      [yf - 1, zt - 20],
      [yf - 1, zt + 0.5],
      [yb + 0.5, zt + 0.5],
      [yb + 0.5, zt - 5],
      [yb + 3.5, zt - 5],
      [yb + 3.5, zt + 3.5],
      [yf - 5, zt + 3.5],
    ];
    for (const [u, v] of hook) {
      await T.clickWorld(W(u, v), { alt: true });
      await sleep(80);
    }
    await T.clickWorld(W(hook[0][0], hook[0][1]));
    await sleep(300);
    ed.setMode("select");
    await shot("10-hook-profile", 600);
    T.click("Finish sketch");
    await sleep(400);

    // 9 — extrude the hook
    T.click("Model");
    await sleep(100);
    T.click("Extrude");
    await sleep(800);
    const ex = a.activeTool;
    ex.set({ distance: 10, mode: "symmetric" });
    ex.refresh();
    view([xSec, (yf + yb) / 2, zt - 8], 45, [0.8, -0.5, 0.4], [0, 0, 1]);
    await shot("11-extrude-hook", 2200);
    T.click("OK");
    await sleep(2500);
    const hookBody = a.doc.ofKind("body").at(-1);
    a.rename(hookBody.id, "Hook");

    // 10 — symmetry plane and mirror the hook
    T.click("Reference");
    await sleep(100);
    T.click("Symmetry");
    await sleep(600);
    await T.waitIdle(60000);
    a.viewer.fitAll("front");
    await shot("13-symmetry", 2500);
    T.click("Create plane");
    await sleep(500);
    const sym = a.doc.ofKind("primitive").find((p) => p.symmetry);
    T.click("Model");
    await sleep(100);
    T.click("Mirror");
    await sleep(600);
    a.activeTool.set({ baseId: hookBody.id, plane: sym.id });
    a.activeTool.refresh();
    await sleep(1500);
    T.click("OK");
    await sleep(2500);
    const mirrored = a.doc.ofKind("body").at(-1);
    a.rename(mirrored.id, "Hook (mirrored)");


    // 11 — plate sketch on the Front plane, at the front face of the hook
    T.click("Sketch");
    await sleep(100);
    T.click("2D Sketch");
    await sleep(400);
    const st2 = a.activeTool;
    const { resolvePlaneSource } = await import("/src/tools/planes.ts");
    const base = resolvePlaneSource(a, `entity:${front.id}`);
    const off = (yf - 5 - base.origin[1]) * base.zAxis[1];
    st2.set({ source: `entity:${front.id}`, offset: off });
    T.click("Start sketch");
    await sleep(1200);
    const ed2 = a.activeTool;
    const sk2 = a.doc.ofKind("sketch").at(-1);
    const f2 = sk2.frame;
    const W2 = (x, y) => [0, 1, 2].map((i) => f2.origin[i] + f2.xAxis[i] * x + f2.yAxis[i] * y);
    const px = (X) => X - f2.origin[0], py = (Z) => Z - f2.origin[2];
    ed2.setMode("rect");
    await T.clickWorld(W2(px(cx - 150), py(cz - 60)), { alt: true });
    await sleep(400);
    await T.clickWorld(W2(px(cx + 150), py(cz + 60)), { alt: true });
    await sleep(400);
    ed2.setMode("circle");
    for (const dx of [-88.9, 88.9]) {
      await T.clickWorld(W2(px(cx + dx), py(cz + 30)), { alt: true });
      await sleep(400);
      await T.clickWorld(W2(px(cx + dx + 3.5), py(cz + 30)), { alt: true });
      await sleep(400);
    }
    ed2.setMode("select");
    await shot("12-plate-sketch", 600);
    T.click("Finish sketch");
    await sleep(400);
    T.click("Model");
    await sleep(100);
    T.click("Extrude");
    await sleep(800);
    a.activeTool.set({ sketchId: sk2.id, distance: 4, mode: "one" });
    T.click("OK");
    await sleep(2500);
    const plate = a.doc.ofKind("body").at(-1);
    a.rename(plate.id, "Plate");

    // 12 — unite everything
    T.click("Boolean");
    await sleep(600);
    a.activeTool.set({ op: "union", targetId: plate.id, toolId: hookBody.id });
    a.activeTool.refresh();
    view([cx, 0, cz + 10], 110, [0.45, 0.8, 0.45], [0, 0, 1]);
    await shot("14-boolean-unite", 2000);
    T.click("OK");
    await sleep(2500);
    const u1 = a.doc.ofKind("body").at(-1);
    T.click("Boolean");
    await sleep(600);
    a.activeTool.set({ op: "union", targetId: u1.id, toolId: mirrored.id });
    T.click("OK");
    await sleep(2500);
    const finalBody = a.doc.ofKind("body").at(-1);
    a.rename(finalBody.id, "License plate bracket");

    // 13 — deviation check
    for (const p of a.doc.ofKind("primitive")) a.setVisible(p.id, false);
    for (const k of a.doc.ofKind("sketch")) a.setVisible(k.id, false);
    a.setActiveEntity(null);
    T.click("Analyze");
    await sleep(100);
    T.click("Deviation");
    await sleep(800);
    await T.waitIdle();
    const dv = a.activeTool;
    dv.set({ maxDistance: 2, tolerance: 0.25 });
    await dv.run();
    view([cx, 4, zt - 6], 48, [0.35, 0.75, 0.55], [0, 0, 1]);
    step(`deviation: ${JSON.stringify(dv.s.stats)}`);
    await shot("15-deviation", 1200);
    await a.endTool(false);

    // 14 — final bracket (scan see-through, from behind)
    const meshId = a.doc.ofKind("mesh")[0].id;
    a.sync.view(meshId).setOpacity(0.35);
    view([cx, 0, cz + 10], 85, [0.45, 0.8, 0.4], [0, 0, 1]);
    await shot("16-final-bracket", 800);
    view([cx, 0, cz], 190, [0.35, -0.75, 0.3], [0, 0, 1]);
    await shot("17-final-front", 800);

    // 15 — export tab
    T.click("Export");
    await sleep(200);
    a.setActiveEntity(finalBody.id);
    await shot("18-export", 600);

    // ================= Example 2 — scan to a smooth solid body (Autosurface → STEP)
    await a.endTool(false);
    a.doc.reset([]);
    await sleep(300);
    T.click("Mesh");
    await sleep(100);
    T.click("File");
    await sleep(200);
    T.click("Organic handle");
    await sleep(500);
    await T.waitIdle();
    a.viewer.fitAll("iso");
    await shot("e2-01-scan", 800);
    T.click("Freeform");
    await sleep(150);
    T.click("Autosurface");
    await sleep(800);
    await T.waitIdle(120000);
    await sleep(500);
    const at = a.activeTool;
    step(`autosurface: ${JSON.stringify({ patches: at.s.result?.patches, rms: at.s.result?.rms, max: at.s.result?.maxDev })}`);
    await shot("e2-02-autosurface", 800);
    T.click("Create surface");
    await sleep(3000);
    const smooth = a.doc.ofKind("body").at(-1);
    a.rename(smooth.id, "Handle");
    const scanId = a.doc.ofKind("mesh")[0].id;
    a.setVisible(scanId, false);
    a.setActiveEntity(null);
    a.viewer.fitAll("iso");
    await shot("e2-03-smooth-body", 1500);
    a.setVisible(scanId, true);
    T.click("Analyze");
    await sleep(100);
    T.click("Deviation");
    await sleep(800);
    await T.waitIdle();
    const dv2 = a.activeTool;
    dv2.set({ target: "cad", tolerance: 0.05, maxDistance: 1 });
    await dv2.run();
    a.setVisible(scanId, false);
    step(`deviation 2: ${JSON.stringify(dv2.s.stats)}`);
    await shot("e2-04-deviation", 1200);
    await a.endTool(false);
    a.setVisible(scanId, true);
    const k = await a.getKernel();
    const sid = await a.features.ensureShape(smooth.id);
    const info = await k.info(sid);
    const stepText = await k.exportSTEP([{ id: sid, name: "Handle" }]);
    step(`smooth body: ${JSON.stringify({ type: info.type, faces: info.faces, closed: info.closed, valid: info.valid, volume: info.volume, stepKB: Math.round(stepText.length / 1024) })}`);
    T.click("Export");
    await sleep(200);
    a.setActiveEntity(smooth.id);
    await shot("e2-05-export-step", 600);

    return { ok: true, log };
  } catch (e) {
    return { ok: false, error: String(e && e.stack ? e.stack : e), log };
  }
})();
