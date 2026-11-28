// o2xygen WebUI bootstrap.
import { api, errDetail } from "./api.js";
import { OrbitScene, teffColor } from "./scene3d.js";
import { buildCatalog } from "./catalog.js";
import { initPanels, showBody, clearBody, onTimeChanged, setPanelMode } from "./panels.js";
import { initTimeline } from "./timeline.js";
import { initBodyBuilder } from "./add_body.js";
import { initLiveMode } from "./live.js";
import { SurfaceSky } from "./surface_sky.js";
import { initBodyEditor } from "./body_editor.js";

const $ = (s) => document.querySelector(s);

async function boot() {
  const system = await api.system();
  $("#ws-name").textContent = system.name;
  document.title = `O₂xygen · ${system.name}`;
  $("#app-version").textContent = system.version ?? "dev";   // docs/version.txt
  if (system.engine) {
    const dtD = system.engine.dt_s / 86400;
    $("#engine-label").childNodes[0].textContent =
      `N-BODY · ${system.engine.integrator.toUpperCase()} · dt ${dtD.toFixed(2)}d · O₂xygen v`;
  }

  // ---- world selector (.oxygen archives, ADR-0016) ----
  const worldSel = $("#world-select");
  async function refreshWorlds() {
    const w = await api.worlds();
    worldSel.textContent = "";
    for (const n of w.worlds) {
      const opt = document.createElement("option");
      opt.value = n;
      opt.textContent = n;      // world names come from user-named files
      if (n === w.current) opt.selected = true;
      worldSel.appendChild(opt);
    }
  }
  worldSel.addEventListener("change", async () => {
    try {
      await api.worldLoad(worldSel.value);
      location.reload();
    } catch (e) {
      // e.g. 409 while an evolution is running: keep the UI on the real world
      alert(e.message);
      worldSel.value = system.world;
    }
  });
  $("#btn-world-save").addEventListener("click", async () => {
    const name = prompt("另存为世界存档(.oxygen):", system.name);
    if (!name) return;
    try {
      const r = await api.worldSave(name.trim());
      await refreshWorlds();
      worldSel.value = r.world;
      $("#ws-name").textContent = r.world;
    } catch (e) { alert(e.message); }
  });
  await refreshWorlds();

  // ---- server-driven rendering (ADR-0021) ----
  // The server is the ONLY computer. An SSE stream pushes authoritative
  // poses (~60 Hz while the clock moves); the client only retains those poses
  // as a finite trail history. It integrates/extrapolates nothing and has no
  // render clock of its own - currentDays IS the server's sim time. The poll
  // remains for meta (star display props) and as the degraded fallback if
  // the stream dies.
  let selected = null;
  let scene = null;             // set below; timeline callbacks fire later
  let currentDays = 0;          // latest server sim time (authoritative)
  let evolving = false;
  let streamLive = false;
  let appMode = "view";         // default on every boot; never persisted
  let resourcesReady = false;   // atomic canvas gate: no half-loaded scene

  // Time ONLY moves via 推演 (user spec 2026-11-15): in idle mode the
  // timeline is inert (NaN ticks, dead play button); during evolution the
  // same bar shows the time scale + progress and the play button toggles
  // pause/resume.
  const timeline = initTimeline({
    onEvolveToggle: () => { toggleEvolutionPause(); },
    onEvolveSpeed: async (v) => {
      try { await api.evolveSpeed(v); }
      catch (e) { alert(e.message); }
    },
  });

  scene = new OrbitScene($("#gl"), system, {
    onSelect: (name) => select(name, true),
  });
  window.__scene = scene;      // debug/test hook
  const surface = new SurfaceSky(scene.renderer, $("#gl"), system);
  window.__surface = surface;  // observer-local renderer, never a physics clock
  // branch 2: observation points shown on non-stellar surfaces (schema v3);
  // refreshed whenever (re)entering view/edit after live-mode edits
  scene.setObservations(system.observations);
  let live = null;
  let atmosphereSerial = 0;
  const bodyEditor = initBodyEditor({
    system,
    onCommitted: (body) => {
      scene.refreshBodyAppearance(body.name);
      catalog.render();
      $("#grid-status").textContent = "✓ 天体参数已保存";
    },
  });

  const es = new EventSource("/api/stream");
  window.__stream = es;        // debug/test hook
  es.onopen = () => {
    streamLive = true;
    scene.clearTrails();          // a new stream is a new history session
  };
  es.addEventListener("pose", (e) => {
    const m = JSON.parse(e.data);
    streamLive = true;
    currentDays = m.t_days;
    scene.setPose(m.p, m.t_days);
  });
  es.onerror = () => { streamLive = false; };   // poll fallback covers poses/meta

  // ---- meta poll (800 ms): star display props, fallback pose/trails, and
  // the evolution watchdog (a job started elsewhere must attach this UI) ----
  let lastEvolveWatchdog = 0;
  async function pollState() {
    try {
      if (appMode === "view" && !evolving && performance.now() - lastEvolveWatchdog > 3000) {
        lastEvolveWatchdog = performance.now();
        try {
          const st = await api.evolveStatus();
          if (st.state === "running" || st.state === "paused") beginEvolutionUI(st);
        } catch { /* status probe is best-effort */ }
      }
      const snap = await api.state(null);
      if (!streamLive) {
        currentDays = snap.t_days;          // degraded mode: poll owns data
        scene.setSnapshot(snap, true);       //   (records fallback poses too)
      } else {
        scene.setSnapshot(snap);            // meta only: stars/radius/Teff
      }
    } catch { /* transient server busy: keep last snapshot */ }
  }
  await pollState();
  setInterval(pollState, 800);

  const panelCtx = {
    system,
    getTimeDays: () => currentDays,
    getSpanDays: () => timeline.spanDays,
  };
  initPanels(panelCtx);

  const catalog = buildCatalog(system, (name) => select(name, true, "catalog"));
  scene.cb.onStarProps = () => catalog.render();

  async function select(name, fromUser, source) {
    if (appMode === "live") { await live?.select(name); return; }
    // branch 2: re-clicking the ALREADY-SELECTED entry in the left catalog
    // toggles the selection OFF and recenters the view on the archive origin.
    // Only the catalog does this — canvas picks always (re)select and never
    // deselect (user ruling, branch 2 feature list).
    if (appMode === "view" && fromUser && source === "catalog" && selected === name) {
      deselect();
      return;
    }
    selected = name;
    catalog.select(name);
    scene.select(name);
    refreshSky();
    if (appMode === "edit" &&
        system.bodies.find((b) => b.name === name)?.kind === "star") {
      bodyEditor.close();               // only one editor overlay may be visible
      openStarEditor(name);
    } else {
      closeStarEditor();
      if (appMode === "edit") bodyEditor.open(name);
    }
    await showBody(name);
  }

  /** branch 2: no selection — clear catalog highlight, scene tag and panel,
   *  then fly the view back to the fixed archive origin (ADR-0035). */
  function deselect() {
    selected = null;
    catalog.clear();
    scene.clearSelection();       // also flies the view to the archive origin
    clearBody();
  }

  // Real starfield: server computes observer-centric Hipparcos directions.
  // Selection triggers an immediate update; while evolving, a 2 s refresh
  // advances proper motion and the body's parallax without doing any of it
  // on the client.
  let skyInFlight = false;
  async function refreshSky() {
    if (!selected || skyInFlight) return;
    const body = selected;
    skyInFlight = true;
    try {
      const data = await api.sky(body);
      if (body === selected) scene.setSky(data);
    } catch { /* keep the previous sky during a transient failure */ }
    finally { skyInFlight = false; }
  }
  setInterval(() => { if (evolving) refreshSky(); }, 2000);

  // view controls
  document.querySelectorAll("#view-toggle .pill").forEach((pill) => {
    pill.addEventListener("click", () => {
      document.querySelectorAll("#view-toggle .pill").forEach((p) => p.classList.remove("active"));
      pill.classList.add("active");
      scene.set2D(pill.dataset.mode === "2d");
    });
  });
  $("#btn-focus").addEventListener("click", () => {
    if (appMode === "live") surface.home(); else scene.focusSelected();
  });
  $("#btn-grid").addEventListener("click", () => {
    if (appMode === "live") return;   // hidden there anyway
    $("#btn-grid").classList.toggle("on", scene.toggleSurfaceOverlay());
  });
  $("#btn-labels").addEventListener("click", () => {
    if (appMode === "live") return;   // hidden there anyway
    $("#btn-labels").classList.toggle("on", scene.toggleLabels());
  });
  $("#btn-stars").addEventListener("click", () => {
    if (appMode === "live") surface.toggleStars(); else scene.toggleStars();
  });

  // render loop: draw exactly what the server sent - no clock, no physics.
  // A 250 ms ticker drives the clock label and time-followed panels (their
  // own throttles apply), instead of touching the DOM at pose rate.
  setInterval(() => {
    timeline.setClock(currentDays);
    onTimeChanged();
  }, 250);
  (function frame() {
    if (!resourcesReady) {
      requestAnimationFrame(frame);
      return;                    // loading overlay is the ONLY visible canvas
    }
    if (appMode === "live") {
      surface.render();       // latest authoritative observation; no time advance
    } else {
      scene.update(currentDays);
      scene.render();
    }
    requestAnimationFrame(frame);
  })();

  // ---------------- evolution (推演: irreversible, autosaves the world) ----------------
  // Control flow (user spec 2026-11-15):
  //   idle    — 推演 button is orange and opens the setup modal; the
  //             bottom-left play button is greyed and inert; timeline
  //             ticks are all "NaN".
  //   running — 推演 button greyed/unclickable; play button is orange and
  //             toggles pause/resume; the timeline shows time scale+progress.
  //   paused  — 推演 button turns RED "取消": click (with confirm) cancels
  //             the evolution outright; resuming turns it back to grey.
  //   done / cancelled — back to idle.
  const evModal = $("#evolve-modal");
  const evYears = $("#evolve-years"), evSpeed = $("#evolve-speed");
  const evStartBtn = $("#evolve-start"), evCloseBtn = $("#evolve-close");
  const evErr = $("#evolve-err");
  const evBtn = $("#btn-evolve");
  let evTimer = null;

  function setEvolveButton(state) {
    evBtn.classList.toggle("danger", state === "paused");
    evBtn.textContent = state === "paused" ? "✕ 取消推演" : "⏩ 推演";
    evBtn.disabled = state === "running" || appMode !== "view";
  }

  function lockWorldUI(on) {
    for (const id of ["world-select", "btn-world-save"]) {
      const el = document.getElementById(id);
      if (el) el.disabled = on;
    }
    const align = $("#btn-align");
    align.disabled = on || appMode !== "edit";
    align.classList.toggle("accent", appMode === "edit");
    const add = $("#btn-add-body");
    const canAdd = appMode === "edit" || appMode === "live";
    add.disabled = on || !canAdd;
    add.classList.toggle("accent", canAdd);
  }

  evBtn.addEventListener("click", async () => {
    if (appMode !== "view") return;
    if (evolving) {
      // only reachable while PAUSED (running state greys the button out)
      if (!confirm("彻底取消推演?\n\n将回退到上次自动存档(约 0.5 秒内),此后无法恢复。")) return;
      try { await api.evolveCancel(); }
      catch (e) { alert(e.message); }
      return;
    }
    $("#evolve-world").textContent = system.world ?? system.name;
    evErr.textContent = "";
    evModal.hidden = false;
  });
  evCloseBtn.addEventListener("click", () => { evModal.hidden = true; });

  evStartBtn.addEventListener("click", async () => {
    const years = parseFloat(evYears.value), dps = parseFloat(evSpeed.value);
    if (!(years > 0) || !(dps > 0)) { evErr.textContent = "时长与速度必须为正"; return; }
    evErr.textContent = "";
    try {
      const st = await api.evolveStart(years * 365.25, dps);
      evModal.hidden = true;                      // 开始推演即退出弹窗
      beginEvolutionUI(st);
    } catch (e) { evErr.textContent = e.message; }
  });

  function beginEvolutionUI(st) {
    if (appMode !== "view") return;
    evolving = true;
    lockWorldUI(true);
    timeline.setEvolve(st.end_days - st.start_days, st.speed);
    timeline.setPlaying(st.state === "running");
    setEvolveButton(st.state === "paused" ? "paused" : "running");
    clearInterval(evTimer);
    evTimer = setInterval(tickEvolution, 500);
  }

  async function toggleEvolutionPause() {
    let st;
    try { st = await api.evolveStatus(); } catch { return; }
    if (st.state === "running") st = await api.evolvePause();
    else if (st.state === "paused") st = await api.evolveResume();
    applyEvolutionState(st);
  }

  function applyEvolutionState(st) {
    timeline.setPlaying(st.state === "running");
    setEvolveButton(st.state === "paused" ? "paused" : "running");
  }

  async function tickEvolution() {
    let st;
    try { st = await api.evolveStatus(); } catch { return; }
    timeline.setProgress(st.progress || 0);
    applyEvolutionState(st);
    if (["done", "cancelled", "error"].includes(st.state)) {
      clearInterval(evTimer);
      evolving = false;
      timeline.setIdle();            // also resets play button + inputs
      setEvolveButton("idle");
      lockWorldUI(false);
      // the save file was rewritten mid-run: reload so every panel,
      // catalog entry and star property re-reads the new epoch
      if (st.state === "done") location.reload();
      else if (st.state === "error") alert(`推演出错:${st.error || "未知原因"}`);
    }
  }


  api.starEnsure().then((s) => {
    const hint = $("#grid-status");
    if (s.ready) hint.textContent = "✓ MIST 网格已就绪";
    else if (s.loading) {
      hint.textContent = "MIST 网格加载中…";
      let fails = 0;
      const poll = setInterval(async () => {
        try {
          const st = await api.starStatus();
          fails = 0;
          if (st.ready) { hint.textContent = "✓ MIST 网格已就绪"; clearInterval(poll); }
          else if (st.error) { hint.textContent = "MIST 网格加载失败"; clearInterval(poll); }
        } catch {
          // server hiccup: stop polling after 3 consecutive failures instead
          // of an unhandled rejection every 10 s
          if (++fails >= 3) {
            hint.textContent = "MIST 网格状态查询失败";
            clearInterval(poll);
          }
        }
      }, 10_000);
    }
  }).catch(() => {});

  // ---------------- ICRS / J2000 world position (no starfield yet) ----------------
  const afModal = $("#astro-frame-modal");
  const afStatus = $("#astro-frame-status");
  let afState = null;
  $("#btn-astro-frame").addEventListener("click", async () => {
    if (appMode !== "edit") return;
    afStatus.textContent = "";
    try {
      const [frame, ev] = await Promise.all([api.astroFrame(), api.evolveStatus()]);
      afState = frame;
      // viewing astrometry is always allowed; only saving is blocked while
      // an evolution job is active (the server enforces the same rule)
      $("#astro-frame-save").disabled = ["running", "paused"].includes(ev.state);
      const sp = afState.position_radec ?? { ra_deg: 0, dec_deg: 0, distance_pc: 0 };
      $("#af-ra").value = sp.ra_deg;
      $("#af-dec").value = sp.dec_deg;
      $("#af-distance").value = sp.distance_pc;
      $("#af-epoch").value = afState.epoch_jd_tt;
      afModal.hidden = false;
    } catch (e) { alert(e.message); }
  });
  $("#astro-frame-close").addEventListener("click", () => { afModal.hidden = true; });
  $("#astro-frame-save").addEventListener("click", async () => {
    if (appMode !== "edit") return;
    try {
      let ra = parseFloat($("#af-ra").value);
      const dec = parseFloat($("#af-dec").value);
      const dist = parseFloat($("#af-distance").value);
      const epoch = parseFloat($("#af-epoch").value);
      if (![ra, dec, dist, epoch].every(Number.isFinite) ||
          dec < -90 || dec > 90 || dist < 0) {
        afStatus.textContent = "请输入有限坐标；赤纬须为 −90…90°，距离非负";
        return;
      }
      ra = ((ra % 360) + 360) % 360;
      const a = ra * Math.PI / 180, d = dec * Math.PI / 180;
      const position = [dist * Math.cos(d) * Math.cos(a),
                        dist * Math.cos(d) * Math.sin(a), dist * Math.sin(d)];
      afState = await api.astroFrameSave({
        ...afState, position_j2000_pc: position, epoch_jd_tt: epoch,
      });
      system.astro_frame = afState;
      refreshSky();
      afModal.hidden = true;
    } catch (e) { afStatus.textContent = e.message; }
  });

  // ---------------- view / edit mode ----------------
  const modeButtons = [...document.querySelectorAll("#mode-toggle .pill")];
  async function setAppMode(next) {
    if (!resourcesReady || next === appMode) return;
    modeButtons.forEach((b) => { b.disabled = true; });
    try {
      // A mode switch ALWAYS cancels an active job, even if it was started
      // in another tab. Server rule: pause first, then cancel + rollback.
      const st = await api.evolveStatus();
      if (st.state === "running") await api.evolvePause();
      if (st.state === "running" || st.state === "paused") {
        await api.evolveCancel();
      }
      if (!["view", "edit", "live"].includes(next)) return;
      clearInterval(evTimer);
      evolving = false;
      timeline.setIdle();
      if (appMode === "live") {
        atmosphereSerial++;
        live?.exit();
        surface.exit();
      }
      appMode = next;
      scene.controls.enabled = next !== "live";
      scene.pickEnabled = next !== "live";
      document.body.classList.toggle("mode-view", next === "view");
      document.body.classList.toggle("mode-edit", next === "edit");
      document.body.classList.toggle("mode-live", next === "live");
      modeButtons.forEach((b) => b.classList.toggle("active", b.dataset.appMode === next));
      $("#btn-astro-frame").hidden = next !== "edit";
      afModal.hidden = true;
      evModal.hidden = true;
      $("#align-modal").hidden = true;
      bodyBuilder.close();
      closeStarEditor();
      bodyEditor.close();
      setPanelMode(next);
      setEvolveButton("idle");       // only view mode can evolve
      if (next === "live") {
        surface.enter();
        await live?.enter();
        $("#btn-focus").title = "回到观察点天顶 (Home)";
      } else {
        $("#btn-focus").title = "聚焦选中天体";
        // points may have been added/renamed in live mode: refresh the
        // surface overlay before the user sees view/edit again
        try {
          const r = await api.observations();
          system.observations = r.points;
          scene.setObservations(r.points);
        } catch { /* keep the boot-time points on a transient failure */ }
        if (selected) await select(selected, false);
      }
      lockWorldUI(false);            // add point is enabled ONLY in live
    } catch (e) {
      alert(`切换模式失败:${e.message}`);
    } finally {
      modeButtons.forEach((b) => { b.disabled = false; });
    }
  }
  modeButtons.forEach((b) => b.addEventListener("click", () => setAppMode(b.dataset.appMode)));

  // ---------------- stellar editor (edit mode, draft until save) ----------------
  const starEditor = $("#star-editor");
  const starEditStatus = $("#star-edit-status");
  const starSaveBtn = $("#star-edit-save");
  const starPreviewBtn = $("#star-edit-preview");
  let starDraftName = null;
  let previewSerial = 0, previewTimer = null;

  function closeStarEditor() {
    previewSerial++;                         // invalidate in-flight previews
    clearTimeout(previewTimer);
    starEditor.hidden = true;
    starDraftName = null;
    starEditStatus.textContent = "";
  }
  $("#star-edit-close").addEventListener("click", closeStarEditor);
  function updateWDCoolEditability() {
    const on = $("#sx-use-wdcool").checked;
    const derived = window.__wdcoolInput != null;
    for (const id of ["sx-wd-mass", "sx-wd-formation", "sx-wd-envelope"]) {
      document.getElementById(id).disabled = !on || derived;
    }
  }
  $("#sx-use-wdcool").addEventListener("change", () => {
    updateWDCoolEditability();
    schedulePreview();
  });

  const AGE_LOG_MIN = -3.0, AGE_LOG_MAX = Math.log10(200.0);
  function updateAgeSlider() {
    const age = parseFloat($("#sx-age").value);
    if (!Number.isFinite(age) || age <= 0) return;
    const frac = Math.max(0, Math.min(1,
      (Math.log10(age) - AGE_LOG_MIN) / (AGE_LOG_MAX - AGE_LOG_MIN)));
    $("#sx-age-slider").value = Math.round(frac * 1000);
  }
  function schedulePreview() {
    updateAgeSlider();
    clearTimeout(previewTimer);
    previewTimer = setTimeout(() => previewStar(), 80);
  }
  $("#sx-age").addEventListener("input", schedulePreview);
  for (const id of ["sx-mini", "sx-feh"]) {
    document.getElementById(id).addEventListener("input", schedulePreview);
  }
  $("#sx-age-slider").addEventListener("input", () => {
    const v = Number($("#sx-age-slider").value) / 1000;
    $("#sx-age").value = 10 ** (AGE_LOG_MIN + v * (AGE_LOG_MAX - AGE_LOG_MIN));
    schedulePreview();
  });

  function setLifecycleTrack(r) {
    const slider = $("#sx-age-slider");
    const stops = [];
    for (const seg of r.phase_segments ?? []) {
      // MIST phase segments use log10(YEARS); this control uses log10(GYR)
      const a = Math.max(0, Math.min(100,
        (seg.start_logage - 9.0 - AGE_LOG_MIN) / (AGE_LOG_MAX - AGE_LOG_MIN) * 100));
      const b = Math.max(0, Math.min(100,
        (seg.end_logage - 9.0 - AGE_LOG_MIN) / (AGE_LOG_MAX - AGE_LOG_MIN) * 100));
      if (b > a && b > 0 && a < 100) stops.push(`${seg.color} ${a}% ${b}%`);
    }
    if (stops.length) slider.style.setProperty("--life-gradient",
      `linear-gradient(90deg, ${stops.join(", ")})`);
    slider.style.setProperty("--star-color", `#${teffColor(r.teff_k).getHexString()}`);
    $("#sx-phase").textContent = r.phase?.label ?? "—";
  }

  async function openStarEditor(name) {
    starDraftName = name;
    starEditor.hidden = false;
    starEditStatus.textContent = "正在读取恒星数据…";
    starSaveBtn.disabled = true;
    try {
      const d = await api.body(name);
      if (appMode !== "edit" || starDraftName !== name) return;
      const b = d.body;
      $("#star-edit-title").textContent = b.name;
      $("#sx-mini").value = b.extra?.mini_msun ?? b.mass_kg / 1.98847e30;
      $("#sx-mass").value = b.mass_kg / 1.98847e30;
      $("#sx-feh").value = b.extra?.feh ?? 0.0;
      $("#sx-age").value = b.extra?.age0_gyr ?? 1.0;
      $("#sx-radius").value = b.radius_m / 6.957e8;
      $("#sx-lum").value = b.luminosity_w / 3.828e26;
      $("#sx-teff").value = b.extra?.teff_k ?? 5772;
      $("#sx-period").value = (b.rotation_period_s ?? 0) / 3600;
      const coolInfo = d.wdcool_input ?? null;
      window.__wdcoolInput = coolInfo;
      const useCool = b.extra?.wd_mass_msun != null ||
        (coolInfo && !coolInfo.covered_by_mist3 && coolInfo.active);
      $("#sx-use-wdcool").checked = useCool;
      $("#sx-wd-mass").value = coolInfo?.mass_msun ??
        b.extra?.wd_mass_msun ?? b.mass_kg / 1.98847e30;
      $("#sx-wd-formation").value = coolInfo?.formation_age_gyr ??
        b.extra?.wd_formation_age_gyr ??
        Math.max(0, (b.extra?.age0_gyr ?? 1.0) - 0.1);
      $("#sx-wd-envelope").value = b.extra?.wd_envelope ?? "thick";
      updateWDCoolEditability();
      updateAgeSlider();
      const axis = b.spin_axis ?? [0, 0, 1];
      $("#sx-axis-x").value = axis[0];
      $("#sx-axis-y").value = axis[1];
      $("#sx-axis-z").value = axis[2];
      starEditStatus.textContent = "";
      starSaveBtn.disabled = false;
      await previewStar();
    } catch (e) {
      starEditStatus.textContent = e.message;
    }
  }

  function draftPayload() {
    const num = (id) => parseFloat(document.getElementById(id).value);
    const payload = {
      name: starDraftName,
      mini_msun: num("sx-mini"), feh: num("sx-feh"), age0_gyr: num("sx-age"),
      spin_period_h: num("sx-period"),
      spin_axis: [num("sx-axis-x"), num("sx-axis-y"), num("sx-axis-z")],
      use_wdcool: $("#sx-use-wdcool").checked,
    };
    if (payload.use_wdcool) {
      payload.wd_mass_msun = num("sx-wd-mass");
      payload.wd_formation_age_gyr = num("sx-wd-formation");
      payload.wd_envelope = $("#sx-wd-envelope").value;
    }
    return payload;
  }

  async function previewStar() {
    const name = starDraftName;
    if (!name) return;
    const payload = draftPayload();
    const values = [...Object.values(payload).filter((v) => typeof v === "number"),
                    ...payload.spin_axis];
    if (!values.every(Number.isFinite)) {
      starEditStatus.textContent = "所有参数必须为有效数值";
      return;
    }
    starPreviewBtn.disabled = true;
    starEditStatus.textContent = "正在预览（不保存）…";
    const serial = ++previewSerial;
    try {
      const r = await api.starPreview(payload);
      window.__lastStarPreview = r;       // debug/test hook, not saved state
      if (serial !== previewSerial || starDraftName !== name) return;
      // preview is a draft-only readback; never update the saved catalog here
      $("#sx-mass").value = r.mass_msun;
      $("#sx-radius").value = r.radius_rsun;
      $("#sx-lum").value = r.luminosity_lsun;
      $("#sx-teff").value = r.teff_k;
      $("#sx-mdot").textContent = `${r.mdot_msun_yr.toExponential(3)} M☉/年`;
      $("#sx-loss").textContent = `${r.cumulative_loss_msun.toExponential(4)} M☉`;
      $("#sx-cooling").textContent = r.cooling_age_gyr == null ? "—" :
        `${r.cooling_age_gyr.toExponential(4)} Gyr`;
      const modelText = { mist: "MIST 轨道", mist3_wd: "MIST III · WD 冷却",
                          wdcool: "WDCool · WD 冷却" };
      $("#star-edit-model").textContent =
        `${modelText[r.model] ?? r.model} · ${r.phase?.label ?? "—"}`;
      $("#star-edit-swatch").style.background = `#${teffColor(r.teff_k).getHexString()}`;
      setLifecycleTrack(r);
      if (r.wd_mass_msun != null) $("#sx-wd-mass").value = r.wd_mass_msun;
      if (r.formation_age_gyr != null) $("#sx-wd-formation").value = r.formation_age_gyr;
      window.__wdcoolInput = {
        mass_msun: r.wd_mass_msun, formation_age_gyr: r.formation_age_gyr,
        covered_by_mist3: r.covered_by_mist3, active: r.model === "wdcool",
      };
      updateWDCoolEditability();
      updateAgeSlider();
      starEditStatus.textContent = "";
    } catch (e) {
      starEditStatus.textContent = e.message;
    } finally {
      starPreviewBtn.disabled = false;
    }
  }
  starPreviewBtn.addEventListener("click", previewStar);

  starSaveBtn.addEventListener("click", async () => {
    const name = starDraftName;
    if (!name || appMode !== "edit") return;
    previewSerial++;                         // no preview may win over a save
    clearTimeout(previewTimer);
    const payload = draftPayload();
    const values = [...Object.values(payload).filter((v) => typeof v === "number"),
                    ...payload.spin_axis];
    if (!values.every(Number.isFinite)) {
      starEditStatus.textContent = "所有参数必须为有效数值";
      return;
    }
    starSaveBtn.disabled = true;
    starEditStatus.textContent = "正在保存…";
    try {
      const r = await api.starEdit(payload);
      const local = system.bodies.find((b) => b.name === name);
      if (local) Object.assign(local, r.body);
      closeStarEditor();              // only here is the draft committed
    } catch (e) {
      starEditStatus.textContent = e.message;
      starSaveBtn.disabled = false;
    }
  });

  // ---------------- barycenter alignment (edit mode only) ----------------
  const alignModal = $("#align-modal");
  const alignConfirm = $("#align-confirm");
  const alignStatus = $("#align-status");
  let alignBusy = false;

  function setAlignBusy(on) {
    alignBusy = on;
    alignConfirm.disabled = on;
    $("#align-cancel").disabled = on;
    $("#align-close").disabled = on;
    $("#btn-align").disabled = on || appMode !== "edit";
  }
  $("#btn-align").addEventListener("click", () => {
    if (appMode !== "edit" || evolving || alignBusy) return;
    $("#align-world").textContent = system.world ?? system.name;
    alignStatus.textContent = "";
    setAlignBusy(false);
    alignModal.hidden = false;
  });
  for (const id of ["#align-close", "#align-cancel"]) {
    $(id).addEventListener("click", () => {
      if (!alignBusy) alignModal.hidden = true;   // no request, no save
    });
  }
  alignConfirm.addEventListener("click", async () => {
    if (appMode !== "edit" || evolving || alignBusy) return;
    setAlignBusy(true);
    alignStatus.textContent = "正在对齐并保存…";
    try {
      await api.worldAlign();
      alignModal.hidden = true;
      location.reload();               // re-read the saved aligned epoch
    } catch (e) {
      alignStatus.textContent = e.message;
      setAlignBusy(false);
    }
  });

  // ---------------- full-screen body builder (edit mode only) ----------------
  const bodyBuilder = initBodyBuilder({
    system,
    onCommitted: () => location.reload(),
  });
  window.__bodyBuilder = bodyBuilder;   // debug/test hook; preview is read-only
  $("#btn-add-body").addEventListener("click", () => {
    if (appMode === "edit" && !evolving) bodyBuilder.open();
  });

  async function loadLiveAtmosphere(body, pointName, serial) {
    try {
      let model = await api.atmosphereStart(body);
      while (model.status === "running") {
        await new Promise((resolve) => setTimeout(resolve, 500));
        if (serial !== atmosphereSerial || appMode !== "live" || live?.selectedName !== pointName) return;
        model = await api.atmosphereStatus(body);
      }
      if (serial !== atmosphereSerial || appMode !== "live" || live?.selectedName !== pointName) return;
      if (!model.ready) throw new Error(model.error || "大气预计算未完成");
      surface.setAtmosphere(model);
      surface.render();
    } catch (e) {
      if (serial === atmosphereSerial && appMode === "live") {
        surface.title.textContent = `${pointName} · 大气不可用: ${e.message}`;
      }
    }
  }
  live = initLiveMode({
    system,
    onEdit: (point) => live.openEditor(point),
    onSelect: async (pointName, data) => {
      const serial = ++atmosphereSerial;
      surface.setData(data);
      surface.render();
      if (data.atmosphere) await loadLiveAtmosphere(data.point.body, pointName, serial);
    },
  });
  window.__live = live;
  // The server remains the only physics clock. These are read-only observer
  // queries while another tab finishes stellar-grid settlement, never local
  // evolution or extrapolation.
  setInterval(() => { if (appMode === "live") void live.refresh(); }, 3000);

  // initial selection: the moon he had selected in the screenshots
  await select(system.bodies.some((b) => b.name === "Αύρα") ? "Αύρα" : system.bodies[0].name);

  // Atomic resource gate (0.4.3): MIST + MIST III + WDCool, every star's
  // first authoritative properties, and the real Hipparcos starfield must
  // ALL be present before the normal canvas is rendered or mode switching
  // is accepted. Until then the opaque "加载中" layer is the only view.
  const loadingDetail = $("#resource-loading-detail");
  let resourceTimer = null;
  let resourceCheckInFlight = false;
  async function checkResourceReadiness() {
    if (resourcesReady || resourceCheckInFlight) return;
    resourceCheckInFlight = true;
    try {
      const status = await api.starStatus();
      const gridsReady = status.ready && status.wd_ready && status.wdcool_ready;
      const propsReady = system.bodies.every((b) => b.kind !== "star" ||
        scene.snap?.stars?.[b.name]?.properties_ready === true);
      const skyReady = !!scene.skyPoints;
      const pending = [];
      if (!status.ready) pending.push("MIST 演化网格");
      if (!status.wd_ready) pending.push("MIST III 白矮星网格");
      if (!status.wdcool_ready) pending.push("WDCool 冷却网格");
      if (!propsReady) pending.push("恒星首轮属性");
      if (!skyReady) pending.push("Hipparcos 真实星场");
      if (gridsReady && propsReady && skyReady) {
        resourcesReady = true;
        if (resourceTimer !== null) clearInterval(resourceTimer);
        resourceTimer = null;
        $("#resource-loading").hidden = true;
        document.body.classList.add("resources-ready");
        scene.render();           // first normal frame, fully initialized
      } else {
        loadingDetail.textContent = `正在准备：${pending.join(" · ")}`;
        if (status.error) loadingDetail.textContent += `（${status.error}）`;
        if (gridsReady && propsReady && !skyReady) await refreshSky();
      }
    } catch {
      loadingDetail.textContent = "正在等待服务器资源状态…";
    } finally {
      resourceCheckInFlight = false;
    }
  }
  resourceTimer = setInterval(() => { void checkResourceReadiness(); }, 500);
  await checkResourceReadiness();
}

boot().catch((e) => {
  document.body.innerHTML = `<pre style="color:#e87a7a;padding:40px">启动失败: ${e.stack}</pre>`;
});
