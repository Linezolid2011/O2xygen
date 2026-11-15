// o2xygen WebUI bootstrap.
import { api, errDetail } from "./api.js";
import { OrbitScene } from "./scene3d.js";
import { buildCatalog } from "./catalog.js";
import { initPanels, showBody, onTimeChanged } from "./panels.js";
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
    await api.worldLoad(worldSel.value);
    location.reload();
  });
  $("#btn-world-save").addEventListener("click", async () => {
    const name = prompt("另存为世界存档(.oxygen):", system.name);
    if (!name) return;
    const r = await api.worldSave(name.trim());
    await refreshWorlds();
    worldSel.value = r.world;
    $("#ws-name").textContent = r.world;
  });
  await refreshWorlds();

  let selected = null;

  let scene = null;             // set below; timeline callbacks fire later
  let currentDays = 0;          // latest server sim time (snapshot)
  let evolving = false;

  // Time ONLY moves via 推演 (user spec 2026-11-15): in idle mode the
  // timeline is inert (NaN ticks, dead play button); during evolution the
  // same bar shows the time scale + progress and the play button toggles
  // pause/resume.
  const timeline = initTimeline({
    onEvolveToggle: () => { toggleEvolutionPause(); },
  });

  scene = new OrbitScene($("#gl"), system, {
    onSelect: (name) => select(name, true),
  });
  window.__scene = scene;      // debug/test hook

  // ---- live N-body snapshots (ADR-0015) at the CURRENT sim time. A
  // render clock advances with wall time at the evolution speed between
  // snapshots (server snapshots resync it), so slow evolution is SMOOTH
  // instead of jumping once per poll - the "0.5 s per frame" stutter.
  let renderDays = 0, evoSpeedDps = 0;
  let lastSnapDays = 0, lastSnapWall = 0;
  let lastEvolveWatchdog = 0;
  async function pollState() {
    try {
      // watchdog: a job started elsewhere (other tab / page refresh mid-
      // evolution) must attach this UI to it, or rendering falls back to
      // one jump per poll
      if (!evolving && performance.now() - lastEvolveWatchdog > 3000) {
        lastEvolveWatchdog = performance.now();
        try {
          const st = await api.evolveStatus();
          if (st.state === "running" || st.state === "paused") beginEvolutionUI(st);
        } catch { /* status probe is best-effort */ }
      }
      const snap = await api.state(null);
      currentDays = snap.t_days;
      if (evolving) {
        // adaptive rate: EMA of the server's actual advance per wall second
        // (chunk quantisation makes the configured speed inexact); soft
        // position correction on top. Together these bound the tracking
        // error to well under one chunk quantum - no visible hitches.
        const wall = performance.now();
        if (lastSnapWall) {
          const inst = (snap.t_days - lastSnapDays) / ((wall - lastSnapWall) / 1000);
          if (inst >= 0) evoSpeedDps = evoSpeedDps * 0.5 + inst * 0.5;
        }
        lastSnapDays = snap.t_days;
        lastSnapWall = wall;
        renderDays += (snap.t_days - renderDays) * 0.25;
      } else {
        renderDays = snap.t_days;
        evoSpeedDps = 0;
        lastSnapWall = 0;
      }
      timeline.setClock(currentDays);
      scene.setSnapshot(snap);
      scene.updateTrails();
      onTimeChanged();          // irradiation/chart marker follow the clock
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
    await showBody(name);
  }

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

  // render loop: the render clock advances every frame during evolution
  let lastFrameT = performance.now();
  (function frame(now) {
    const dt = Math.min(((now || performance.now()) - lastFrameT) / 1000, 0.25);
    lastFrameT = now || performance.now();
    if (evolving && evoSpeedDps > 0) renderDays += evoSpeedDps * dt;
    scene.update(renderDays);
    scene.render();
    requestAnimationFrame(frame);
  })(performance.now());

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
    evBtn.disabled = state === "running";
  }

  function lockWorldUI(on) {
    for (const id of ["world-select", "btn-world-save", "btn-pipeline"]) {
      const el = document.getElementById(id);
      if (el) el.disabled = on;
    }
  }

  evBtn.addEventListener("click", async () => {
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
    evolving = true;
    evoSpeedDps = st.speed ?? 0;          // until the estimator takes over
    lastSnapWall = 0;
    lockWorldUI(true);
    timeline.setEvolve(st.end_days - st.start_days);
    timeline.setPlaying(true);
    setEvolveButton("running");
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
    if (st.state !== "running") {
      evoSpeedDps = 0;                      // freeze exactly
      lastSnapWall = 0;
      renderDays = st.done_days;
    }
    applyEvolutionState(st);
    if (["done", "cancelled", "error"].includes(st.state)) {
      clearInterval(evTimer);
      evolving = false;
      timeline.setIdle();
      timeline.setPlaying(false);
      setEvolveButton("idle");
      lockWorldUI(false);
      if (st.state === "done") await refreshWorlds();
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
