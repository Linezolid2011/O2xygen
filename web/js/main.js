// o2xygen WebUI bootstrap.
import { api, errDetail } from "./api.js";
import { OrbitScene } from "./scene3d.js";
import { buildCatalog } from "./catalog.js";
import { initPanels, showBody, onTimeChanged, setPanelMode } from "./panels.js";
import { initTimeline } from "./timeline.js";

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
  // poses (~60 Hz while the clock moves) and server-integrated trails; the
  // client integrates nothing, extrapolates nothing, and has no render clock
  // of its own - currentDays IS the server's sim time. The 800 ms poll
  // remains for meta (star display props) and as the degraded fallback if
  // the stream dies.
  let selected = null;
  let scene = null;             // set below; timeline callbacks fire later
  let currentDays = 0;          // latest server sim time (authoritative)
  let evolving = false;
  let streamLive = false;
  let appMode = "view";         // default on every boot; never persisted

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

  const es = new EventSource("/api/stream");
  window.__stream = es;        // debug/test hook
  es.addEventListener("pose", (e) => {
    const m = JSON.parse(e.data);
    streamLive = true;
    currentDays = m.t_days;
    scene.setPose(m.p, m.t_days, m.bcm);
  });
  es.addEventListener("trails", (e) => {
    scene.updateTrails(JSON.parse(e.data));
  });
  es.onerror = () => { streamLive = false; };   // poll fallback covers meta

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
        scene.setSnapshot(snap);            //   (sets fallback pose too)
        scene.updateTrails(snap);
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

  const catalog = buildCatalog(system, (name) => select(name, true));

  async function select(name, fromUser) {
    selected = name;
    catalog.select(name);
    scene.select(name);
    refreshSky();
    if (appMode === "edit" &&
        system.bodies.find((b) => b.name === name)?.kind === "star") {
      openStarEditor(name);
    } else {
      closeStarEditor();
    }
    await showBody(name);
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
  $("#btn-focus").addEventListener("click", () => scene.focusSelected());
  $("#btn-grid").addEventListener("click", () => scene.toggleGrid());
  $("#btn-stars").addEventListener("click", () => scene.toggleStars());

  // render loop: draw exactly what the server sent - no clock, no physics.
  // A 250 ms ticker drives the clock label and time-followed panels (their
  // own throttles apply), instead of touching the DOM at pose rate.
  setInterval(() => {
    timeline.setClock(currentDays);
    onTimeChanged();
  }, 250);
  (function frame() {
    scene.update(currentDays);
    scene.drawTrails();
    scene.render();
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
    for (const id of ["world-select", "btn-world-save", "btn-pipeline"]) {
      const el = document.getElementById(id);
      if (el) el.disabled = on;
    }
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
    if (next === appMode) return;
    modeButtons.forEach((b) => { b.disabled = true; });
    try {
      // A mode switch ALWAYS cancels an active job, even if it was started
      // in another tab. Server rule: pause first, then cancel + rollback.
      const st = await api.evolveStatus();
      if (st.state === "running") await api.evolvePause();
      if (st.state === "running" || st.state === "paused") {
        await api.evolveCancel();
      }
      clearInterval(evTimer);
      evolving = false;
      timeline.setIdle();
      lockWorldUI(false);
      appMode = next;
      document.body.classList.toggle("mode-view", next === "view");
      document.body.classList.toggle("mode-edit", next === "edit");
      modeButtons.forEach((b) => b.classList.toggle("active", b.dataset.appMode === next));
      $("#btn-astro-frame").hidden = next !== "edit";
      afModal.hidden = true;
      evModal.hidden = true;
      closeStarEditor();
      setPanelMode(next);
      setEvolveButton("idle");       // edit mode: disabled by appMode
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
  let starDraftName = null;

  function closeStarEditor() {
    starEditor.hidden = true;
    starDraftName = null;
    starEditStatus.textContent = "";
  }
  $("#star-edit-close").addEventListener("click", closeStarEditor);

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
      const axis = b.spin_axis ?? [0, 0, 1];
      $("#sx-axis-x").value = axis[0];
      $("#sx-axis-y").value = axis[1];
      $("#sx-axis-z").value = axis[2];
      starEditStatus.textContent = "";
      starSaveBtn.disabled = false;
    } catch (e) {
      starEditStatus.textContent = e.message;
    }
  }

  starSaveBtn.addEventListener("click", async () => {
    const name = starDraftName;
    if (!name || appMode !== "edit") return;
    const num = (id) => parseFloat(document.getElementById(id).value);
    const payload = {
      name,
      mini_msun: num("sx-mini"), feh: num("sx-feh"), age0_gyr: num("sx-age"),
      spin_period_h: num("sx-period"),
      spin_axis: [num("sx-axis-x"), num("sx-axis-y"), num("sx-axis-z")],
    };
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

  // ---------------- pipeline modal ----------------
  const modal = $("#pipeline-modal");
  $("#btn-pipeline").addEventListener("click", () => {
    if (!selected) { alert("请先在左侧目录选择一个天体"); return; }
    runPipeline(selected);
  });
  $("#pipeline-close").addEventListener("click", () => { modal.hidden = true; });
  $("#pipeline-cancel").addEventListener("click", () => { modal.hidden = true; });

  async function runPipeline(bodyName) {
    modal.hidden = false;
    const ul = $("#pipeline-stages");
    const fill = $("#pipeline-fill");
    const status = $("#pipeline-status");
    ul.innerHTML = "";
    fill.style.width = "0%";
    const stageDefs = ["结构与尺寸", "大气与光学厚度", "温度表"];
    const li = stageDefs.map((n) => {
      const el = document.createElement("li");
      el.innerHTML = `<span>${n}</span><span class="st">等待中…</span>`;
      ul.appendChild(el);
      return el;
    });
    status.textContent = `求解对象:${bodyName}`;
    try {
      // animate stage transitions while the real request runs
      li[0].classList.add("running");
      li[0].querySelector(".st").textContent = "正在求解…";
      const t0 = performance.now();
      const res = await api.pipeline({ body: bodyName, span_days: timeline.spanDays, n_samples: 13 });
      const minMs = 900;                      // let the progress read, like his dialog
      const wait = Math.max(0, minMs - (performance.now() - t0));
      await new Promise((r) => setTimeout(r, wait));
      li.forEach((el, i) => {
        el.classList.remove("running");
        el.classList.add("done");
        el.querySelector(".st").textContent = "已完成";
        fill.style.width = `${((i + 1) / 3) * 100}%`;
      });
      status.textContent = `完成,用时 ${res.elapsed_s.toFixed(2)} s`;
      await showBody(bodyName);               // refresh tabs with pipeline results
    } catch (e) {
      status.innerHTML = `<span class="err">失败:${e.message}</span>`;
    }
  }

  // ---------------- add-body modal ----------------
  // TEMP-DISABLED(2026-11-15): 添加天体挡路,后续加回。连同 index.html 的按钮与本块一起取消注释。
  // const addModal = $("#add-modal");
  // const parentSel = $("#add-parent");
  // function refreshParents() {
  //   parentSel.innerHTML = system.bodies
  //     .map((b) => `<option value="${b.name}">${b.name}</option>`).join("");
  // }
  // $("#btn-add-body").addEventListener("click", () => { refreshParents(); addModal.hidden = false; });
  // $("#add-close").addEventListener("click", () => { addModal.hidden = true; });
  // $("#add-cancel").addEventListener("click", () => { addModal.hidden = true; });
  // $("#add-confirm").addEventListener("click", async () => {
  //   const name = $("#add-name").value.trim();
  //   if (!name) { $("#add-error").textContent = "请输入名称"; return; }
  //   if (system.bodies.some((b) => b.name === name)) { $("#add-error").textContent = "名称已存在"; return; }
  //   const a = parseFloat($("#add-a").value), e = parseFloat($("#add-e").value),
  //         inc = parseFloat($("#add-inc").value), mass = parseFloat($("#add-mass").value);
  //   if (![a, e, inc, mass].every(isFinite)) {
  //     $("#add-error").textContent = "请输入有效的轨道根数与质量数值";
  //     return;
  //   }
  //   const r = await fetch("/api/add_body", {
  //     method: "POST", headers: { "Content-Type": "application/json" },
  //     body: JSON.stringify({
  //       name, around: parentSel.value,
  //       a_au: a, e, inc_deg: inc, mass_earth: mass,
  //     }),
  //   });
  //   if (!r.ok) { $("#add-error").textContent = await errDetail(r); return; }
  //   location.reload();
  // });

  // initial selection: the moon he had selected in the screenshots
  await select(system.bodies.some((b) => b.name === "Αύρα") ? "Αύρα" : system.bodies[0].name);
  scene.render();
}

boot().catch((e) => {
  document.body.innerHTML = `<pre style="color:#e87a7a;padding:40px">启动失败: ${e.stack}</pre>`;
});
