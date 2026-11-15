// o2xygen WebUI bootstrap.
import { api } from "./api.js";
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
    worldSel.innerHTML = w.worlds
      .map((n) => `<option value="${n}" ${n === w.current ? "selected" : ""}>${n}</option>`)
      .join("");
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
  async function pollState(force = false) {
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

  // warm the MIST grid in the background so star estimates are instant later
  api.starEnsure().then((s) => {
    const hint = $("#grid-status");
    if (s.ready) hint.textContent = "✓ MIST 网格已就绪";
    else if (s.loading) {
      hint.textContent = "MIST 网格加载中…";
      const poll = setInterval(async () => {
        const st = await api.starStatus();
        if (st.ready) { hint.textContent = "✓ MIST 网格已就绪"; clearInterval(poll); }
        else if (st.error) { hint.textContent = "MIST 网格加载失败"; clearInterval(poll); }
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
  const addModal = $("#add-modal");
  const parentSel = $("#add-parent");
  function refreshParents() {
    parentSel.innerHTML = system.bodies
      .map((b) => `<option value="${b.name}">${b.name}</option>`).join("");
  }
  $("#btn-add-body").addEventListener("click", () => { refreshParents(); addModal.hidden = false; });
  $("#add-close").addEventListener("click", () => { addModal.hidden = true; });
  $("#add-cancel").addEventListener("click", () => { addModal.hidden = true; });
  $("#add-confirm").addEventListener("click", async () => {
    const name = $("#add-name").value.trim();
    if (!name) { $("#add-error").textContent = "请输入名称"; return; }
    if (system.bodies.some((b) => b.name === name)) { $("#add-error").textContent = "名称已存在"; return; }
    const r = await fetch("/api/add_body", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name, around: parentSel.value,
        a_au: parseFloat($("#add-a").value), e: parseFloat($("#add-e").value),
        inc_deg: parseFloat($("#add-inc").value),
        mass_earth: parseFloat($("#add-mass").value),
      }),
    });
    if (!r.ok) { $("#add-error").textContent = (await r.json()).detail; return; }
    location.reload();
  });

  // initial selection: the moon he had selected in the screenshots
  await select(system.bodies.some((b) => b.name === "Αύρα") ? "Αύρα" : system.bodies[0].name);
  scene.render();
}

boot().catch((e) => {
  document.body.innerHTML = `<pre style="color:#e87a7a;padding:40px">启动失败: ${e.stack}</pre>`;
});
