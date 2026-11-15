// Right-hand panel: 基础参数 / 结构 / 大气 / 温度表 / 介绍.
import { api } from "./api.js";
import { tempChart } from "./charts.js";
import { teffColor } from "./scene3d.js";
import { AU, DAY } from "./kepler.js";

const $ = (sel) => document.querySelector(sel);
const fmt = (x, digits = 4) =>
  x == null || !isFinite(x) ? "—" : Number(x).toLocaleString("en-US", { maximumFractionDigits: digits });

let ctx = null;                 // {system, getTimeDays}
let current = null;             // current body name
let chart = null;               // temperature chart handle
let lastTempData = null;
let lastBodyData = null;        // latest /api/body response (osculating etc.)

export function initPanels(context) {
  ctx = context;
  document.querySelectorAll(".ptab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".ptab").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-page").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      $(`#tab-${btn.dataset.tab}`).classList.add("active");
      if (btn.dataset.tab === "temperature" && chart) chart.setSize();
    });
  });
}

export async function showBody(name) {
  current = name;
  lastBodyData = null;
  const b = ctx.system.bodies.find((x) => x.name === name);
  if (!b) return;
  $("#body-name").textContent = b.name;
  $("#body-kind").textContent = b.kind === "star" ? "恒星" : "天体";
  $("#body-swatch").style.background = b.kind === "star"
    ? `#${teffColor(b.extra?.teff_k ?? 5772).getHexString()}` : "#7aa2e8";

  // prefetch osculating data so renderParams can prefer live elements
  try { lastBodyData = await api.body(name, ctx.getTimeDays()); }
  catch { lastBodyData = null; }
  renderParams(b);
  renderStructure(b);
  renderAtmosphere(b);
  await renderTemperature(b);
  await refreshIrradiation();     // fills the 辐照 DOM (element now exists)
  renderIntro(b);
}

/* ============================== 基础参数 ============================== */
async function renderParams(b) {
  const el = $("#tab-params");
  const isOrigin = ctx.system.origin === b.name;
  // osculating elements vs the world origin (ADR-0016) - there is no
  // parent anymore, and no static fallback: state is the only truth
  const osc = lastBodyData?.osculating ?? null;
  const T = (lastBodyData && lastBodyData.period_days) ?? null;
  const el2 = osc ? { a_m: osc.a_m, e: osc.e, inc: osc.inc } : null;
  const kindLabel = b.kind === "star" ? "恒星" : "天体";
  const originName = ctx.system.origin ?? "";

  el.innerHTML = `
    <div class="sec">
      <div class="sec-title">轨道${isOrigin ? "" : ` (相对 ${originName} · 瞬时)`}</div>
      ${isOrigin
        ? `<div class="note">原点天体 · 世界参考点</div>`
        : el2 ? `
      <div class="kv"><span class="k">轨道半长轴</span><span class="v">${fmt(el2.a_m / AU, 3)} AU</span></div>
      <div class="kv"><span class="k">偏心率</span><span class="v">${fmt(el2.e, 4)}</span></div>
      <div class="kv"><span class="k">倾角</span><span class="v">${fmt((el2.inc * 180) / Math.PI, 2)}°</span></div>
      <div class="kv"><span class="k">周期</span><span class="v accent">${T > 1200 ? fmt(T / 365.25, 2) + " 年" : fmt(T, 1) + " 天"}</span></div>`
        : `<div class="note">等待快照…</div>`}
    </div>
    <div class="sec">
      <div class="sec-title">物理</div>
      <div class="kv"><span class="k">质量</span><span class="v">${b.kind === "star" ? fmt(b.mass_kg / 1.98847e30, 4) + " M☉" : fmt(b.mass_kg / 5.9722e24, 3) + " M⊕"}</span></div>
      <div class="kv"><span class="k">半径</span><span class="v">${b.kind === "star" ? fmt(b.radius_m / 6.957e8, 4) + " R☉" : fmt(b.radius_m / 6.371e6, 4) + " R⊕"}</span></div>
      ${b.rotation_period_s ? `<div class="kv"><span class="k">自转周期</span><span class="v">${fmt(b.rotation_period_s / 3600, 1)} h</span></div>` : ""}
      ${b.kind !== "star" ? `<div class="kv"><span class="k">反照率</span><span class="v">${fmt(b.albedo, 2)}</span></div>` : ""}
      ${b.extra?.teff_k ? `<div class="kv"><span class="k">有效温度</span><span class="v accent">${fmt(b.extra.teff_k, 0)} K</span></div>` : ""}
    </div>
    <div class="sec">
      <div class="sec-title">辐照 (随轨道变化)</div>
      <div id="irradiation-live"></div>
      <div class="note">大气顶入射的各恒星垂直光束通量之和；不含本星自辐射、
      遮挡与大气衰减，也非地面或全球平均值。S⊕ = 1361 W/m²。</div>
    </div>
    ${b.kind === "star" ? starEstimateSection(b) : ""}
  `;
  if (b.kind === "star") wireStarEstimate(b);
}

async function refreshIrradiation() {
  // stores lastBodyData (osculating period, flux) for renderParams
  if (!current) return;
  const el = $("#irradiation-live");
  if (!el) return;
  const d = await api.body(current, ctx.getTimeDays());
  lastBodyData = d;
  const rows = Object.entries(d.illumination)
    .filter(([k]) => k !== "total")
    .map(([k, v]) => `<div class="kv"><span class="k">${k}</span><span class="v">${fmt(v, 5)} W/m²</span></div>`)
    .join("");
  el.innerHTML = `
    ${rows}
    <div class="kv"><span class="k">总和</span><span class="v accent">${fmt(d.flux_wm2, 5)} W/m²</span></div>
    ${d.t_eq_k ? `<div class="kv"><span class="k">平衡温度 (昼侧)</span><span class="v">${fmt(d.t_eq_k, 2)} K</span></div>` : ""}`;
}

/* -------- star estimation (MIST) -------- */
function starEstimateSection(b) {
  return `
    <div class="sec">
      <div class="sec-title">参数估算 · MIST v1.2 / MESA r7503</div>
      <div class="row-flex" style="margin-bottom:8px">
        <label class="field grow">初始质量 (M☉)<input id="se-mass" type="number" step="0.01" value="${(b.mass_kg / 1.98847e30).toFixed(3)}"></label>
        <label class="field grow">[Fe/H]<input id="se-feh" type="number" step="0.05" value="${b.extra?.feh ?? 0.0}"></label>
      </div>
      <label class="field">年龄 (Gyr)<input id="se-age" type="number" step="0.1" value="${b.extra?.age_gyr ?? 1.26}"></label>
      <button class="btn ghost btn-sm" id="se-run">估算演化参数</button>
      <div id="se-result" style="margin-top:8px"></div>
      <div class="note">非旋转、太阳比例元素丰度演化轨道。按等效演化点对初始质量与
      [Fe/H] 插值，再按年龄求值；不包含双星传质、超新星爆发或中子星/黑洞演化。</div>
    </div>`;
}

function wireStarEstimate(b) {
  $("#se-run").addEventListener("click", async () => {
    const out = $("#se-result");
    out.innerHTML = `<div class="note">正在查询 MIST 网格… (首次使用需加载数百条轨道, 约 1–2 分钟)</div>`;
    try {
      const r = await api.starEstimate({
        mass_msun: parseFloat($("#se-mass").value),
        feh: parseFloat($("#se-feh").value),
        age_gyr: parseFloat($("#se-age").value),
      });
      if (r._unavailable) {
        out.innerHTML = `<div class="note">网格后台加载中，请 30 秒后再试。</div>`;
        return;
      }
      out.innerHTML = `
        <div class="kv"><span class="k">演化后当前质量</span><span class="v">${fmt(r.mass_msun, 5)} M☉</span></div>
        <div class="kv"><span class="k">模型球形半径</span><span class="v">${fmt(r.radius_km, 0)} km</span></div>
        <div class="kv"><span class="k">光度</span><span class="v accent">${fmt(r.luminosity_lsun, 4)} L☉</span></div>
        <div class="kv"><span class="k">有效温度</span><span class="v">${fmt(r.teff_k, 0)} K</span></div>
        <div class="kv"><span class="k">当前损失率 |star_mdot|</span><span class="v">${r.mdot_msun_yr.toExponential(3)} M☉/年</span></div>
        <div class="kv"><span class="k">累计损失</span><span class="v">${fmt(r.cumulative_loss_msun, 5)} M☉</span></div>`;
    } catch (e) {
      out.innerHTML = `<div class="err">${e.message}</div>`;
    }
  });
}

/* ============================== 结构 ============================== */
const LAYER_PALETTE = ["#d98e63", "#c47a52", "#a96643", "#8e5436"];

function renderStructure(b) {
  const el = $("#tab-structure");
  const mEarth = b.mass_kg / 5.9722e24;
  if (b.kind === "star") {
    el.innerHTML = `<div class="note">恒星结构由 MIST 演化轨道给出（见基础参数 · 参数估算）。</div>`;
    return;
  }
  const cmf = b.extra?.cmf ?? 0.3;
  const rotH = (b.rotation_period_s || 24 * 3600) / 3600;
  el.innerHTML = `
    <div class="sec">
      <div class="sec-title">参数估算</div>
      <label class="field">主要成分
        <select id="st-model">
          <option value="prem">PREM 半经验 (1–8 M⊕, CMF 0–0.4)</option>
          <option value="zeng2019">高压 EOS 表 · Zeng (2019) 近似</option>
        </select>
      </label>
      <label class="field">核心质量分数 CMF <span id="st-cmf-v" class="v mono">${cmf.toFixed(2)}</span>
        <input id="st-cmf" type="range" min="0" max="0.6" step="0.01" value="${cmf}">
      </label>
      <label class="field">自转周期 (h)<input id="st-rot" type="number" step="0.1" value="${rotH.toFixed(1)}"></label>
      <div class="note">PREM: R/R⊕ = (1.07 − 0.21·CMF)·(M/M⊕)^(1/3.7) — Zeng, Sasselov
      &amp; Jacobsen 2016。EOS 表选项为同族近似，正式版接入高压 EOS 网格。</div>
    </div>
    <div class="sec">
      <div class="sec-title">估算结果</div>
      <div id="st-result"></div>
    </div>`;
  $("#st-model").value = mEarth <= 8 ? "prem" : "zeng2019";
  $("#st-cmf").addEventListener("input", (e) => {
    $("#st-cmf-v").textContent = parseFloat(e.target.value).toFixed(2);
    runStructure();
  });
  $("#st-rot").addEventListener("change", runStructure);
  $("#st-model").addEventListener("change", runStructure);
  runStructure();
}

async function runStructure() {
  if (!current) return;
  const body = current;
  const cmfEl = $("#st-cmf");
  if (!cmfEl) return;
  const out = $("#st-result");
  try {
    const r = await api.structure(body, {
      cmf: parseFloat(cmfEl.value),
      model: $("#st-model").value,
      rotation_h: parseFloat($("#st-rot").value),
    });
    if (body !== current) return;
    out.innerHTML = structureHTML(r);
  } catch (e) {
    if (body === current && document.getElementById("st-result")) {
      out.innerHTML = `<div class="err">${e.message}</div>`;
    }
  }
}

function structureHTML(r) {
  const circles = r.layers.map((l, i) => {
    const rad = 62 * l.r1;
    return `<circle cx="70" cy="70" r="${rad}" fill="${LAYER_PALETTE[i]}" stroke="#171a22" stroke-width="1.5"/>`;
  }).reverse().join("");
  const rows = r.layers.map((l, i) => `
    <tr><td><span class="swatch-cell" style="background:${LAYER_PALETTE[i]}"></span>${l.layer}</td>
    <td>${(l.r0 * 100).toFixed(1)}–${(l.r1 * 100).toFixed(1)}%</td>
    <td>${fmt(l.rho_g_cm3, 1)}</td></tr>`).join("");
  return `
    <div class="kv"><span class="k">半径</span><span class="v accent">${fmt(r.radius_earth_r, 5)} R⊕</span></div>
    <div class="kv"><span class="k">平均密度</span><span class="v">${fmt(r.mean_density_g_cm3, 3)} g/cm³</span></div>
    <div class="kv"><span class="k">赤道有效重力</span><span class="v">${fmt(r.g_equator, 3)} m/s²</span></div>
    <div class="kv"><span class="k">两极重力</span><span class="v">${fmt(r.g_pole, 3)} m/s²</span></div>
    <div class="kv"><span class="k">扁率 (慢转近似)</span><span class="v">${(r.flattening_slow * 100).toFixed(3)}%</span></div>
    <div class="layer-svg-wrap"><svg width="140" height="140">${circles}</svg></div>
    <div class="layer-caption">中心 → 表面 · 深色密度较高</div>
    <table class="layers"><tr><th>层</th><th>半径范围</th><th>平均密度 (g/cm³)</th></tr>${rows}</table>
    <div class="note">引力为 GM/r² 单极近似,赤道有效重力另减自转离心项;圈层为 n=1
    多方球剖面的分层预设,密度整体归一化至总质量。</div>`;
}

/* ============================== 大气 ============================== */
const GAS_PRESETS = {
  "氮氧示例": [["氮气 N₂", 78.0], ["氧气 O₂", 21.0], ["氩气 Ar", 1.0]],
  "原始 CO₂": [["二氧化碳 CO₂", 91.3], ["氮气 N₂", 8.7]],
};

function renderAtmosphere(b) {
  const el = $("#tab-atmosphere");
  if (b.kind === "star") {
    el.innerHTML = `<div class="note">恒星无大气模块。</div>`;
    return;
  }
  const atm = b.extra?.atm ?? { gases: [["氮气 N₂", 78.0], ["氧气 O₂", 21.0]], tau0: 0.32, p0_bar: 1.0, n: 2.0, S: 1.0, nabla: 0.0 };
  el.innerHTML = `
    <div class="sec">
      <div class="sec-title">光谱数据与大气剖面</div>
      <div class="row-flex" style="margin-bottom:8px">
        <select id="at-preset" class="grow">
          <option value="">— 预设 —</option>
          ${Object.keys(GAS_PRESETS).map((k) => `<option>${k}</option>`).join("")}
        </select>
        <button class="btn ghost btn-sm" id="at-add">添加气体</button>
      </div>
      <div id="at-gases"></div>
      <div class="gas-sum" id="at-sum"></div>
      <div class="note">Planck 平均吸收: ExoMol CO₂ (H₂/He 展宽, 100–3400 K,
      10⁻⁵–100 bar) — 正式版接入完整线表。当前版本为气柱幂律灰大气估算。</div>
    </div>
    <div class="sec">
      <div class="sec-title">气柱模型参数 (通用气柱幂律)</div>
      <div class="row-flex">
        <label class="field grow">参考光学厚度 τ₀<input id="at-tau0" type="number" step="0.01" value="${atm.tau0}"></label>
        <label class="field grow">参考气压 P₀ (bar)<input id="at-p0" type="number" step="0.01" value="${atm.p0_bar}"></label>
      </div>
      <div class="row-flex">
        <label class="field grow">参考重力 g₀ (m/s²)<input id="at-g0" type="number" step="0.1" value="9.8"></label>
        <label class="field grow">压力指数 n<input id="at-n" type="number" step="0.1" value="${atm.n}"></label>
      </div>
      <div class="row-flex">
        <label class="field grow">相对吸收强度 S<input id="at-S" type="number" step="0.05" value="${atm.S}"></label>
        <label class="field grow">温压剖面指数 ∇<input id="at-nabla" type="number" step="0.02" value="${atm.nabla}"></label>
      </div>
      <button class="btn ghost btn-sm" id="at-estimate">估算大气性质</button>
      <div id="at-result" style="margin-top:8px"></div>
    </div>`;

  const gasesBox = $("#at-gases");
  function drawGases(gases) {
    gasesBox.innerHTML = "";
    gases.forEach(([name, pct], i) => {
      const row = document.createElement("div");
      row.className = "gas-row";
      row.innerHTML = `<span class="grow">${name}</span>
        <input class="pct" type="number" step="0.1" min="0" max="100" value="${pct}">
        <span style="color:var(--text-faint)">%</span>
        <button class="icon-btn" style="width:24px;height:24px" title="移除">×</button>`;
      row.querySelector("button").addEventListener("click", () => { row.remove(); updateSum(); });
      row.querySelector("input").addEventListener("input", updateSum);
      gasesBox.appendChild(row);
    });
    updateSum();
  }
  function gasRows() {
    return [...gasesBox.querySelectorAll(".gas-row")].map((row) => [
      row.querySelector(".grow").textContent,
      parseFloat(row.querySelector("input").value) || 0,
    ]);
  }
  function updateSum() {
    const sum = gasRows().reduce((a, g) => a + g[1], 0);
    const s = $("#at-sum");
    s.textContent = `合计 ${sum.toFixed(1)}%`;
    s.classList.toggle("bad", Math.abs(sum - 100) > 0.5);
  }
  drawGases(atm.gases);
  $("#at-add").addEventListener("click", () => {
    const row = document.createElement("div");
    row.className = "gas-row";
    row.innerHTML = `<input class="grow" placeholder="气体名称" value="甲烷 CH₄">
      <input class="pct" type="number" step="0.1" value="0"><span style="color:var(--text-faint)">%</span>
      <button class="icon-btn" style="width:24px;height:24px">×</button>`;
    row.querySelector("button").addEventListener("click", () => { row.remove(); updateSum(); });
    row.querySelector(".pct").addEventListener("input", updateSum);
    gasesBox.appendChild(row);
  });
  $("#at-preset").addEventListener("change", (e) => {
    if (e.target.value) drawGases(GAS_PRESETS[e.target.value]);
  });
  $("#at-estimate").addEventListener("click", async () => {
    const d = await api.body(current, ctx.getTimeDays());
    const t0 = d.t_eq_k;
    const tau0 = parseFloat($("#at-tau0").value);
    const nabla = parseFloat($("#at-nabla").value);
    // grey-atmosphere estimate: T_surf^4 = T_eq^4 (1 + 3τ/4)
    const tSurf = t0 * (1 + 0.75 * tau0) ** 0.25;
    const p0 = parseFloat($("#at-p0").value);
    const steps = [1e-4, 1e-2, 0.1, 1, 10];
    const rows = steps.map((p) => `<tr><td>${p}</td><td>${(tSurf * (p / p0) ** nabla).toFixed(1)}</td></tr>`).join("");
    $("#at-result").innerHTML = `
      <div class="kv"><span class="k">平衡温度 (昼侧)</span><span class="v">${fmt(t0, 2)} K</span></div>
      <div class="kv"><span class="k">灰大气表面温度 (估算)</span><span class="v accent">${fmt(tSurf, 1)} K</span></div>
      <table class="layers" style="margin-top:6px"><tr><th>P (bar)</th><th>T(P) (K)</th></tr>${rows}</table>
      <div class="note">T(P) = T_ref · (P/P_ref)^∇;τ₀ 与 S 的气柱自洽求解待
      Clima 求解器接入。</div>`;
  });
}

/* ============================== 温度表 ============================== */
async function renderTemperature(b) {
  const el = $("#tab-temperature");
  if (b.kind === "star") {
    el.innerHTML = `<div class="note">恒星无温度表。</div>`;
    chart = null;
    return;
  }
  el.innerHTML = `
    <div class="sec">
      <div class="row-flex">
        <select id="tp-regime" class="grow">
          <option value="fast_rotator">全球平均 (T⁴=S/4σ)</option>
          <option value="dayside" selected>昼侧平均 (T⁴=S/2σ)</option>
          <option value="substellar">星下点 (T⁴=S/σ)</option>
        </select>
        <label class="field" style="margin:0">反照率<input id="tp-albedo" type="number" step="0.01" min="0" max="0.9" value="${b.albedo}" style="width:64px"></label>
      </div>
      <div class="row-flex" style="margin-top:8px">
        <label class="field grow">采样点数<input id="tp-n" type="number" step="1" min="2" max="200" value="13"></label>
        <button class="btn accent btn-sm" id="tp-run">计算温度表</button>
      </div>
      <label class="field" style="margin-top:6px">
        <span class="row-flex"><input type="checkbox" id="tp-follow" checked> 跟随当前时刻与温度</span>
      </label>
    </div>
    <div class="sec"><div class="sec-title">温度曲线</div><div class="chart-box" id="tp-chart"></div></div>
    <div class="sec"><div class="sec-title">温度表</div><div id="tp-table"></div></div>`;

  $("#tp-run").addEventListener("click", computeTemp);
  $("#tp-follow").addEventListener("change", () => chart?.setFollow($("#tp-follow").checked));
  await computeTemp();
}

async function computeTemp() {
  if (!current) return;
  const body = current;
  const albedoEl = $("#tp-albedo");
  if (!albedoEl) return;                        // tab re-rendered meanwhile
  const req = {
    albedo: parseFloat(albedoEl.value),
    regime: $("#tp-regime").value,
    span_days: ctx.getSpanDays(),
    n: parseInt($("#tp-n").value),
  };
  const r = await api.temperature(body, req);
  if (body !== current) return;                 // selection changed while in flight
  const box = $("#tp-chart");
  if (!box) return;
  lastTempData = r;
  box.innerHTML = "";
  chart = tempChart(box, r.dense, { markerDays: ctx.getTimeDays(), follow: $("#tp-follow").checked });
  renderTempTable(r);
}

function renderTempTable(r) {
  const tDays = ctx.getTimeDays();
  const cur = r.rows.reduce((best, row) =>
    Math.abs(row.d - tDays) < Math.abs(best.d - tDays) ? row : best, r.rows[0]);
  $("#tp-table").innerHTML = `
    <table class="temp-table">
      <tr><th>距初始时刻 (天)</th><th>总辐照 (W/m²)</th><th>温度 (K)</th></tr>
      ${r.rows.map((row) => `
        <tr class="${row === cur ? "temp-current" : ""}">
          <td>${fmt(row.d, 3)}</td><td>${fmt(row.flux, 3)}</td><td>${fmt(row.T, 3)}</td>
        </tr>`).join("")}
    </table>
    <div class="note">当前时间 +${fmt(tDays, 3)} 天 · 表中温度 ${fmt(cur.T, 3)} K</div>`;
}

/* -------- live update hook called by timeline -------- */
export function onTimeChanged() {
  refreshIrradiation();
  if (chart) chart.setMarker(ctx.getTimeDays());
  if (lastTempData) renderTempTable(lastTempData);
}

/* ============================== 介绍 ============================== */
function renderIntro(b) {
  const el = $("#tab-intro");
  const kindLabel = b.kind === "star" ? "恒星" : "天体";
  const isOrigin = ctx.system.origin === b.name;
  const osc = lastBodyData?.osculating ?? null;
  const T = (lastBodyData && lastBodyData.period_days) ?? null;
  el.innerHTML = `
    <div class="intro-md">
      <h4>${b.name}</h4>
      <p>${kindLabel}${isOrigin ? ",世界原点天体" : `,瞬时轨道相对 ${ctx.system.origin ?? "原点"}`}。</p>
      ${osc && T ? `<p>瞬时半长轴约 ${fmt(osc.a_m / AU, 2)} AU,周期约 ${fmt(T / 365.25, 2)} 年(N-body 实时状态,非固定轨道)。</p>` : ""}
      <p>质量约 ${b.kind === "star" ? fmt(b.mass_kg / 1.98847e30, 2) + " M☉" : fmt(b.mass_kg / 5.9722e24, 2) + " M⊕"},
      半径约 ${b.kind === "star" ? fmt(b.radius_m / 6.957e8, 3) + " R☉" : fmt(b.radius_m / 6.371e6, 4) + " R⊕"}。</p>
      ${b.extra?.teff_k ? `<p>有效温度约 ${fmt(b.extra.teff_k, 0)} K。</p>` : ""}
      <h4>说明</h4>
      <p>本文档由引擎参数自动生成;接入 LLM 叙事层后,将在此生成完整的世界观
      设定文稿(地貌、矿产、生态等)。</p>
    </div>`;
}
