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
  const timeline = initTimeline({
    onTimeChanged: () => {
      scene?.update(timeline.days);
      onTimeChanged();
    },
    onWrap: () => { pollState(true); },   // timeline looped: re-seed client
  });

  scene = new OrbitScene($("#gl"), system, {
    onSelect: (name) => select(name, true),
  });
  window.__scene = scene;      // debug/test hook

  // ---- live N-body snapshots (ADR-0015): server integrates, client
  // extrapolates p + v·Δt between polls; trails refresh per snapshot.
  let snapReqT = null;
  let evolving = false;
  async function pollState(force = false) {
    if (evolving) return;               // the evolution job drives the clock
    const t = timeline.days;
    if (!force && snapReqT !== null && Math.abs(t - snapReqT) < 0.5) return;
    snapReqT = t;
    try {
      const snap = await api.state(t);
      scene.setSnapshot(snap);
      scene.updateTrails();
    } catch { /* transient server busy: keep last snapshot */ }
  }
  await pollState(true);
  setInterval(pollState, 800);

  scene.update(timeline.days); // place bodies BEFORE any user click can land

  const panelCtx = {
    system,
    getTimeDays: () => timeline.days,
    getSpanDays: () => timeline.span,
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

  // render loop (scene-side; timeline runs its own rAF for state)
  (function frame() {
    scene.render();
    requestAnimationFrame(frame);
  })();

  // ---------------- evolution modal (推演: irreversible, autosaves the world) ----------------
  const evModal = $("#evolve-modal");
  const evYears = $("#evolve-years"), evSpeed = $("#evolve-speed");
  const evStartBtn = $("#evolve-start"), evPauseBtn = $("#evolve-pause"),
        evCancelBtn = $("#evolve-cancel"), evCloseBtn = $("#evolve-close");
  const evFill = $("#evolve-fill"), evStatus = $("#evolve-status"), evErr = $("#evolve-err");
  let evTimer = null;

  $("#btn-evolve").addEventListener("click", async () => {
    $("#evolve-world").textContent = system.world ?? system.name;
    evModal.hidden = false;
    evErr.textContent = "";
    try {
      const st = await api.evolveStatus();
      if (st.state === "running" || st.state === "paused") attachEvolution();
    } catch { /* server down: modal still opens for the error on start */ }
  });
  evCloseBtn.addEventListener("click", () => {
    if (evolving) return;                       // 推演进行中不可关闭
    evModal.hidden = true;
  });

  function lockTimeline(on) {
    for (const id of ["tl-slider", "btn-play", "speed", "span",
                      "world-select", "btn-world-save", "btn-pipeline"]) {
      const el = document.getElementById(id);
      if (el) el.disabled = on;
    }
    // 进度条不可拖动,只显示已推演占比
    const slider = document.getElementById("tl-slider");
    if (on) { slider.max = 1000; slider.value = 0; }
    else { slider.max = timeline.span; slider.value = timeline.days; }
  }

  evStartBtn.addEventListener("click", async () => {
    const years = parseFloat(evYears.value), dps = parseFloat(evSpeed.value);
    if (!(years > 0) || !(dps > 0)) { evErr.textContent = "时长与速度必须为正"; return; }
    evErr.textContent = "";
    try {
      await api.evolveStart(years * 365.25, dps);
      attachEvolution();
    } catch (e) { evErr.textContent = e.message; }
  });

  evPauseBtn.addEventListener("click", async () => {
    const st = await api.evolveStatus();
    if (st.state === "running") await api.evolvePause();
    else if (st.state === "paused") await api.evolveResume();
    refreshEvolveControls(await api.evolveStatus());
  });
  evCancelBtn.addEventListener("click", async () => {
    try { refreshEvolveControls(await api.evolveCancel()); }
    catch (e) { evErr.textContent = e.message; }
  });
  evSpeed.addEventListener("change", async () => {
    const st = await api.evolveStatus();
    if (st.state !== "paused") return;
    try { await api.evolveSpeed(parseFloat(evSpeed.value)); }
    catch (e) { evErr.textContent = e.message; }
  });

  function attachEvolution() {
    evolving = true;
    lockTimeline(true);
    evStartBtn.disabled = true;
    evYears.disabled = true;                      // 最大推演时长开始后不可修改
    evCloseBtn.disabled = true;
    refreshEvolveControls({ state: "running" });
    clearInterval(evTimer);
    evTimer = setInterval(tickEvolution, 500);
  }

  function refreshEvolveControls(st) {
    const paused = st.state === "paused";
    evPauseBtn.disabled = !(st.state === "running" || paused);
    evPauseBtn.textContent = paused ? "继续" : "暂停";
    evCancelBtn.disabled = !paused;               // 仅暂停后可取消
    evSpeed.disabled = !paused;                   // 仅暂停后可改速度
  }

  async function tickEvolution() {
    let st;
    try { st = await api.evolveStatus(); }
    catch { return; }
    const pct = Math.round((st.progress || 0) * 1000) / 10;
    evFill.style.width = `${st.progress * 100}%`;
    document.getElementById("tl-slider").value = (st.progress || 0) * 1000;
    const days = Math.round(st.done_days - st.start_days);
    evStatus.textContent = { running: `推演中 ${pct}% (+${days} 天)`,
                             paused: `已暂停 ${pct}% (+${days} 天)`,
                             done: `推演完成(已推演 ${days} 天,存档已更新)`,
                             cancelled: `已取消(保留至上次自动存档,约 0.5 秒内)`,
                             error: `推演出错:${st.error}` }[st.state] || st.state;
    refreshEvolveControls(st);
    // scene follows the job's clock
    try {
      const snap = await api.state(null);
      scene.setSnapshot(snap);
      scene.updateTrails();
      scene.update(snap.t_days);
      onTimeChanged();
    } catch { /* transient */ }
    if (["done", "cancelled", "error"].includes(st.state)) {
      clearInterval(evTimer);
      evolving = false;
      lockTimeline(false);
      evStartBtn.disabled = false;
      evYears.disabled = false;
      evSpeed.disabled = false;
      evCloseBtn.disabled = false;
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
      const res = await api.pipeline({ body: bodyName, span_days: timeline.span, n_samples: 13 });
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
