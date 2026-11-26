// Draft-only body builder. The server owns every physical computation: this
// module submits inputs and draws only the absolute coordinates it returns.
import * as THREE from "three";
import { OrbitControls } from "/vendor/OrbitControls.js";
import { api } from "./api.js";
import { AU } from "./kepler.js";
import { teffColor, bodyDisplayColor } from "./scene3d.js";

const PREVIEW_DELAY_MS = 120;
const SCIENTIFIC_NUMBER_FIELDS = new Set(["mass_value", "radius_value", "a_value"]);
const MODEL_LABELS = {
  mist: "MIST 轨道",
  mist3_wd: "MIST III · WD 冷却",
  wdcool: "WDCool · WD 冷却",
};

function fmt(value, digits = 4) {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  return Number(value).toLocaleString("en-US", { maximumFractionDigits: digits });
}

function sci(value) {
  return value == null || !Number.isFinite(Number(value))
    ? "—" : Number(value).toExponential(3);
}

function validVec(v) {
  return Array.isArray(v) && v.length >= 3 &&
    Number.isFinite(v[0]) && Number.isFinite(v[1]) && Number.isFinite(v[2]);
}

let stellarGlowMap = null;
function stellarGlowTexture() {
  if (stellarGlowMap) return stellarGlowMap;
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d");
  const gradient = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  gradient.addColorStop(0, "rgba(255,255,255,0.95)");
  gradient.addColorStop(0.22, "rgba(255,244,224,0.55)");
  gradient.addColorStop(0.55, "rgba(255,224,176,0.18)");
  gradient.addColorStop(1, "rgba(255,220,160,0)");
  g.fillStyle = gradient;
  g.fillRect(0, 0, 128, 128);
  stellarGlowMap = new THREE.CanvasTexture(c);
  return stellarGlowMap;
}

function makeStellarGlow() {
  const glow = new THREE.Sprite(new THREE.SpriteMaterial({
    map: stellarGlowTexture(), transparent: true, opacity: 0.85,
    blending: THREE.AdditiveBlending, depthWrite: false,
  }));
  glow.visible = false;
  return glow;
}

function updateStellarGlow(glow, radius_m, visible, teff_k) {
  glow.visible = visible;
  if (!visible) return;
  glow.material.color.copy(teffColor(teff_k ?? 5772.0));
  // Local scale is multiplied by the sphere's TRUE radius.  The small floor
  // is only a glow/visibility aid, never a change to the physical sphere.
  const scale = Math.max(2.6, AU * 0.0006 / radius_m);
  glow.scale.set(scale, scale, 1.0);
}

export function initBodyBuilder({ system, onCommitted }) {
  const root = document.getElementById("body-builder");
  if (!root) throw new Error("缺少 #body-builder 容器");

  root.innerHTML = `
    <div class="bb-editor">
      <header class="bb-head">
        <div>
          <div class="bb-eyebrow">O₂XYGEN WORKSPACE</div>
          <h2>添加天体</h2>
          <p>新建草稿 · 预览不会保存</p>
        </div>
        <button type="button" class="icon-btn bb-close" data-action="close"
                title="不保存退出" aria-label="不保存退出">×</button>
      </header>
      <div class="bb-form" aria-label="新天体参数">
        <section class="bb-card">
          <div class="bb-section-title">基本设置</div>
          <div class="bb-setting-row">
            <span class="bb-label">类型</span>
            <div class="mode-switch bb-switch" data-switch="kind" aria-label="天体类型">
              <button type="button" class="pill" data-value="star">恒星</button>
              <button type="button" class="pill" data-value="body">非恒星</button>
            </div>
          </div>
          <label class="bb-field">名称
            <input type="text" data-field="name" maxlength="120"
                   placeholder="例如：Elpis" autocomplete="off">
          </label>
          <div class="bb-setting-row">
            <span class="bb-label">放置方式</span>
            <div class="mode-switch bb-switch" data-switch="placement" aria-label="放置方式">
              <button type="button" class="pill" data-value="direct">直接坐标</button>
              <button type="button" class="pill" data-value="orbit">轨道根数</button>
            </div>
          </div>
        </section>

        <section class="bb-card" data-kind-section="star" hidden>
          <div class="bb-section-title">恒星演化输入</div>
          <div class="bb-grid">
            <label class="bb-field">初始质量 (M☉)
              <input type="number" data-field="mini_msun" min="0" step="any">
            </label>
            <label class="bb-field">[Fe/H]
              <input type="number" data-field="feh" step="any">
            </label>
            <label class="bb-field">当前总年龄 (Gyr)
              <input type="number" data-field="age0_gyr" min="0" step="any">
            </label>
            <label class="bb-field">自转周期 (h；0 = 不旋转)
              <input type="number" data-field="spin_period_h" min="0" step="any">
            </label>
          </div>
          <div class="bb-label">自转轴（自动由服务器归一化）</div>
          <div class="bb-grid bb-grid-3">
            <label class="bb-field">X
              <input type="number" data-field="spin_axis" data-index="0" step="any">
            </label>
            <label class="bb-field">Y
              <input type="number" data-field="spin_axis" data-index="1" step="any">
            </label>
            <label class="bb-field">Z
              <input type="number" data-field="spin_axis" data-index="2" step="any">
            </label>
          </div>
          <label class="bb-check">
            <input type="checkbox" data-field="use_wdcool"> 启用 WDCool 后备
          </label>
          <div class="bb-note">仅在 WD 质量超出 MIST III 覆盖时使用后备冷却模型。</div>
          <div class="bb-grid bb-wd-fields">
            <label class="bb-field">WD 质量 (M☉)
              <input type="number" data-field="wd_mass_msun" min="0" step="any">
            </label>
            <label class="bb-field">WD 形成年龄 (Gyr)
              <input type="number" data-field="wd_formation_age_gyr" min="0" step="any">
            </label>
            <label class="bb-field">氢包层
              <select data-field="wd_envelope">
                <option value="thick">厚氢包层</option>
                <option value="thin">薄氢包层</option>
              </select>
            </label>
          </div>
        </section>

        <section class="bb-card bb-readout" data-kind-section="star" hidden>
          <div class="bb-section-title">恒星物理输出 · 只读</div>
          <div class="bb-kv"><span>当前质量</span><output data-out="star-mass">—</output></div>
          <div class="bb-kv"><span>半径</span><output data-out="star-radius">—</output></div>
          <div class="bb-kv"><span>光度</span><output data-out="star-luminosity">—</output></div>
          <div class="bb-kv"><span>有效温度</span><output data-out="star-teff">—</output></div>
          <div class="bb-kv"><span>当前质量损失率</span><output data-out="star-mdot">—</output></div>
          <div class="bb-kv"><span>累计质量损失</span><output data-out="star-loss">—</output></div>
          <div class="bb-kv"><span>演化模型</span><output data-out="star-model">—</output></div>
        </section>

        <section class="bb-card" data-kind-section="body">
          <div class="bb-section-title">非恒星物理输入</div>
          <div class="bb-label">质量</div>
          <div class="bb-unit-row">
            <input type="text" inputmode="decimal" data-field="mass_value" aria-label="质量数值">
            <div class="mode-switch bb-switch" data-switch="mass_unit" aria-label="质量单位">
              <button type="button" class="pill" data-value="earth">M⊕</button>
              <button type="button" class="pill" data-value="jupiter">M♃</button>
              <button type="button" class="pill" data-value="kg">kg</button>
            </div>
          </div>
          <div class="bb-label">半径</div>
          <div class="bb-unit-row">
            <input type="text" inputmode="decimal" data-field="radius_value" aria-label="半径数值">
            <div class="mode-switch bb-switch" data-switch="radius_unit" aria-label="半径单位">
              <button type="button" class="pill" data-value="earth">R⊕</button>
              <button type="button" class="pill" data-value="km">km</button>
            </div>
          </div>
          <label class="bb-field">反照率
            <input type="number" data-field="albedo" min="0" max="1" step="any">
          </label>
          <div class="bb-grid">
            <label class="bb-field">自转周期 (h；0 = 不转)
              <input type="number" data-field="spin_period_h" min="0" step="any">
            </label>
            <label class="bb-field">初始自转相位 (°)
              <input type="number" data-field="spin_phase_deg" step="any">
            </label>
          </div>
          <div class="bb-label">自转轴（北极方向；服务器归一化）</div>
          <div class="bb-grid bb-grid-3">
            <label class="bb-field">X<input type="number" data-field="spin_axis" data-index="0" step="any"></label>
            <label class="bb-field">Y<input type="number" data-field="spin_axis" data-index="1" step="any"></label>
            <label class="bb-field">Z<input type="number" data-field="spin_axis" data-index="2" step="any"></label>
          </div>
        </section>

        <section class="bb-card bb-readout" data-kind-section="body">
          <div class="bb-section-title">非恒星物理输出 · 只读</div>
          <div class="bb-kv"><span>平均密度</span><output data-out="body-density">—</output></div>
          <div class="bb-kv"><span>表面重力</span><output data-out="body-gravity">—</output></div>
          <div class="bb-kv"><span>逃逸速度</span><output data-out="body-escape">—</output></div>
        </section>

        <section class="bb-card" data-placement-section="direct" hidden>
          <div class="bb-section-title">绝对世界坐标与速度</div>
          <div class="bb-grid bb-grid-3">
            <label class="bb-field">X (AU)
              <input type="number" data-field="x_au" step="any">
            </label>
            <label class="bb-field">Y (AU)
              <input type="number" data-field="y_au" step="any">
            </label>
            <label class="bb-field">Z (AU)
              <input type="number" data-field="z_au" step="any">
            </label>
          </div>
          <div class="bb-grid">
            <label class="bb-field">速率 (km/s)
              <input type="number" data-field="speed_km_s" min="0" step="any">
            </label>
            <label class="bb-field">速度方位角 (°)
              <input type="number" data-field="speed_azimuth_deg" step="any">
            </label>
            <label class="bb-field">速度仰角 (°)
              <input type="number" data-field="speed_elevation_deg" step="any">
            </label>
          </div>
          <div class="bb-note">坐标固定于绝对世界原点；速度方向由方位角和仰角描述。</div>
        </section>

        <section class="bb-card" data-placement-section="orbit">
          <div class="bb-section-title">参考天体 · 单选</div>
          <div class="bb-parents" role="radiogroup" aria-label="已有参考天体"></div>
          <div class="bb-grid">
            <div>
              <div class="bb-label">半长轴</div>
              <div class="bb-unit-row">
                <input type="text" inputmode="decimal" data-field="a_value"
                       aria-label="半长轴数值">
                <div class="mode-switch bb-switch" data-switch="a_unit" aria-label="半长轴单位">
                  <button type="button" class="pill" data-value="au">AU</button>
                  <button type="button" class="pill" data-value="km">km</button>
                </div>
              </div>
            </div>
            <label class="bb-field">偏心率
              <input type="number" data-field="e" min="0" step="any">
            </label>
            <label class="bb-field">倾角 (°)
              <input type="number" data-field="inc_deg" step="any">
            </label>
            <label class="bb-field">升交点经度 (°)
              <input type="number" data-field="raan_deg" step="any">
            </label>
            <label class="bb-field">近心点幅角 (°)
              <input type="number" data-field="argp_deg" step="any">
            </label>
            <label class="bb-field">轨道相位 (%)
              <input type="number" data-field="phase_pct" min="0" max="100" step="any">
            </label>
          </div>
          <div class="bb-note">轨道根数只用于放置；位置、速度和轨道采样均由服务器计算。</div>
        </section>

        <section class="bb-card bb-readout" data-placement-section="orbit">
          <div class="bb-section-title">轨道输出 · 只读</div>
          <div class="bb-kv"><span>周期</span><output data-out="orbit-period">—</output></div>
          <div class="bb-kv"><span>近日点</span><output data-out="orbit-periapsis">—</output></div>
          <div class="bb-kv"><span>远日点</span><output data-out="orbit-apoapsis">—</output></div>
        </section>
      </div>
      <footer class="bb-statusbar">
        <span class="bb-status" role="status" aria-live="polite"></span>
        <span class="bb-draft-tag">未保存</span>
      </footer>
    </div>

    <div class="bb-preview">
      <header class="bb-head">
        <div>
          <div class="bb-eyebrow">SERVER PREVIEW</div>
          <h2>空间预览</h2>
          <p>绝对世界坐标 · 仅显示 API 响应</p>
        </div>
        <div class="bb-head-actions">
          <button type="button" class="icon-btn bb-fit" title="适配预览" aria-label="适配预览">⌖</button>
          <button type="button" class="btn accent bb-commit" data-action="commit">确认创建</button>
        </div>
      </header>
      <div class="bb-viewport">
        <canvas class="bb-canvas" aria-label="新天体三维预览"></canvas>
        <div class="bb-preview-empty">等待服务器预览…</div>
        <div class="bb-legend" hidden></div>
        <div class="bb-body-label" data-label="candidate" hidden></div>
        <div class="bb-body-label" data-label="parent" hidden></div>
        <div class="bb-view-hint">左键旋转 · 右键平移 · 滚轮缩放</div>
      </div>
    </div>`;

  const viewport = root.querySelector(".bb-viewport");

  const form = root.querySelector(".bb-form");
  const status = root.querySelector(".bb-status");
  const commitButton = root.querySelector(".bb-commit");
  const closeButton = root.querySelector(".bb-close");
  const parentsBox = root.querySelector(".bb-parents");
  const empty = root.querySelector(".bb-preview-empty");
  const legend = root.querySelector(".bb-legend");
  const candidateLabel = root.querySelector('[data-label="candidate"]');
  const parentLabel = root.querySelector('[data-label="parent"]');
  const out = (key) => root.querySelector(`[data-out="${key}"]`);

  let opened = false;
  let committing = false;
  let session = 0;
  let previewSerial = 0;
  let previewTimer = null;
  let draft = null;
  let lastPreview = null;
  let lastPlacement = null;

  function defaultParent() {
    const bodies = Array.isArray(system?.bodies) ? system.bodies : [];
    return (bodies.find((b) => b.kind === "star") ?? bodies[0])?.name ?? "";
  }

  function newDraft() {
    return {
      kind: "body",
      placement: "orbit",
      name: "新天体",
      mini_msun: 1.0,
      feh: 0.0,
      age0_gyr: 1.0,
      spin_period_h: 24.0,
      spin_phase_deg: 0.0,
      spin_axis: [0, 0, 1],
      use_wdcool: false,
      wd_mass_msun: 0.6,
      wd_formation_age_gyr: 0.9,
      wd_envelope: "thick",
      mass_value: 1.0,
      mass_unit: "earth",
      albedo: 0.1,
      radius_value: 1.0,
      radius_unit: "earth",
      x_au: 1.0,
      y_au: 0.0,
      z_au: 0.0,
      speed_km_s: 29.78,
      speed_azimuth_deg: 90.0,
      speed_elevation_deg: 0.0,
      parent: defaultParent(),
      a_value: 1.0,
      a_unit: "au",
      e: 0.0,
      inc_deg: 0.0,
      raan_deg: 0.0,
      argp_deg: 0.0,
      phase_pct: 0.0,
    };
  }

  function renderParents() {
    parentsBox.textContent = "";
    const bodies = Array.isArray(system?.bodies) ? system.bodies : [];
    if (!bodies.length) {
      const note = document.createElement("div");
      note.className = "bb-note";
      note.textContent = "当前世界没有可作为参考的已有天体。";
      parentsBox.appendChild(note);
      return;
    }
    for (const body of bodies) {
      const label = document.createElement("label");
      label.className = "bb-parent-option";
      const radio = document.createElement("input");
      radio.type = "radio";
      radio.name = "bb-parent";
      radio.value = body.name;
      radio.checked = body.name === draft.parent;
      radio.addEventListener("change", () => {
        if (!radio.checked) return;
        draft.parent = body.name;
        schedulePreview();
      });
      const text = document.createElement("span");
      text.textContent = body.name;
      const kind = document.createElement("small");
      kind.textContent = body.kind === "star" ? "恒星" : "天体";
      label.append(radio, text, kind);
      parentsBox.appendChild(label);
    }
    if (!bodies.some((b) => b.name === draft.parent)) {
      draft.parent = "";
    }
  }

  function syncControls() {
    for (const el of root.querySelectorAll("[data-field]")) {
      const key = el.dataset.field;
      if (key === "spin_axis") {
        el.value = draft.spin_axis[Number(el.dataset.index)];
      } else if (el.type === "checkbox") {
        el.checked = Boolean(draft[key]);
      } else {
        el.value = draft[key];
      }
    }
    for (const group of root.querySelectorAll("[data-switch]")) {
      const value = draft[group.dataset.switch];
      for (const button of group.querySelectorAll(".pill")) {
        const active = button.dataset.value === value;
        button.classList.toggle("active", active);
        button.setAttribute("aria-pressed", String(active));
      }
    }
    for (const card of root.querySelectorAll("[data-kind-section]")) {
      card.hidden = card.dataset.kindSection !== draft.kind;
    }
    for (const card of root.querySelectorAll("[data-placement-section]")) {
      card.hidden = card.dataset.placementSection !== draft.placement;
    }
    for (const el of root.querySelectorAll(".bb-wd-fields [data-field]")) {
      el.disabled = !draft.use_wdcool;
    }
  }

  function setStatus(message, isError = false) {
    status.textContent = message;
    status.classList.toggle("err", isError);
  }

  function buildPayload() {
    const p = {
      name: draft.name.trim(),
      kind: draft.kind,
      placement: draft.placement,
    };
    if (draft.kind === "star") {
      Object.assign(p, {
        mini_msun: draft.mini_msun,
        feh: draft.feh,
        age0_gyr: draft.age0_gyr,
        spin_period_h: draft.spin_period_h,
        spin_axis: [...draft.spin_axis],
        use_wdcool: draft.use_wdcool,
        wd_mass_msun: draft.wd_mass_msun,
        wd_formation_age_gyr: draft.wd_formation_age_gyr,
        wd_envelope: draft.wd_envelope,
      });
    } else {
      Object.assign(p, {
        mass_value: draft.mass_value,
        mass_unit: draft.mass_unit,
        albedo: draft.albedo,
        radius_value: draft.radius_value,
        radius_unit: draft.radius_unit,
        spin_period_h: draft.spin_period_h,
        spin_phase_deg: draft.spin_phase_deg,
        spin_axis: [...draft.spin_axis],
      });
    }
    if (draft.placement === "direct") {
      Object.assign(p, {
        x_au: draft.x_au,
        y_au: draft.y_au,
        z_au: draft.z_au,
        speed_km_s: draft.speed_km_s,
        speed_azimuth_deg: draft.speed_azimuth_deg,
        speed_elevation_deg: draft.speed_elevation_deg,
      });
    } else {
      Object.assign(p, {
        parent: draft.parent,
        a_value: draft.a_value,
        a_unit: draft.a_unit,
        e: draft.e,
        inc_deg: draft.inc_deg,
        raan_deg: draft.raan_deg,
        argp_deg: draft.argp_deg,
        phase_pct: draft.phase_pct,
      });
    }
    return p;
  }

  function validatePayload(p) {
    if (!p.name) return "请输入名称";
    if (p.placement === "orbit" && !p.parent) return "请选择一个已有参考天体";
    const numbers = Object.values(p).filter((v) => typeof v === "number");
    if (p.spin_axis) numbers.push(...p.spin_axis);
    return numbers.every(Number.isFinite) ? "" : "所有数值参数必须为有效数值";
  }

  function updateReadouts(r) {
    const star = r.star;
    out("star-mass").textContent = star ? `${fmt(star.mass_msun, 6)} M☉` : "—";
    out("star-radius").textContent = star ? `${fmt(star.radius_rsun, 6)} R☉` : "—";
    out("star-luminosity").textContent = star ? `${fmt(star.luminosity_lsun, 6)} L☉` : "—";
    out("star-teff").textContent = star ? `${fmt(star.teff_k, 2)} K` : "—";
    out("star-mdot").textContent = star ? `${sci(star.mdot_msun_yr)} M☉/年` : "—";
    out("star-loss").textContent = star ? `${sci(star.cumulative_loss_msun)} M☉` : "—";
    out("star-model").textContent = star
      ? (MODEL_LABELS[star.model] ?? star.model ?? "—") : "—";

    const body = r.nonstar;
    out("body-density").textContent = body ? `${fmt(body.mean_density_g_cm3, 5)} g/cm³` : "—";
    out("body-gravity").textContent = body ? `${fmt(body.surface_gravity_m_s2, 5)} m/s²` : "—";
    out("body-escape").textContent = body ? `${fmt(body.escape_velocity_m_s, 5)} m/s` : "—";

    out("orbit-period").textContent = r.period_days == null ? "—" : `${fmt(r.period_days, 5)} 天`;
    out("orbit-periapsis").textContent = r.periapsis_m == null ? "—" : `${fmt(r.periapsis_m / AU, 6)} AU`;
    out("orbit-apoapsis").textContent = r.apoapsis_m == null ? "—" : `${fmt(r.apoapsis_m / AU, 6)} AU`;
  }

  function clearReadouts() {
    for (const el of root.querySelectorAll("[data-out]")) el.textContent = "—";
  }

  function schedulePreview() {
    if (!opened || committing) return;
    const serial = ++previewSerial;       // invalidate immediately, not only on send
    const token = session;
    clearTimeout(previewTimer);
    previewTimer = setTimeout(() => runPreview(serial, token), PREVIEW_DELAY_MS);
  }

  async function runPreview(serial, token) {
    if (!opened || committing || serial !== previewSerial || token !== session) return;
    const payload = buildPayload();
    const invalid = validatePayload(payload);
    if (invalid) {
      setStatus(invalid, true);
      return;
    }
    setStatus("正在预览（不保存）…");
    try {
      const response = await api.addBodyPreview(payload);
      if (!opened || serial !== previewSerial || token !== session) return;
      if (response?._unavailable) throw new Error(response.detail || "预览服务暂不可用");
      lastPreview = response;
      if (lastPlacement !== payload.placement) frameNeeded = true;
      lastPlacement = payload.placement;
      updateReadouts(response);
      drawPreview(response, payload.placement);
      empty.hidden = true;
      legend.hidden = false;
      legend.textContent = payload.placement === "orbit" && response.parent
        ? `候选：${response.name || payload.name}　参考：${response.parent}`
        : `候选：${response.name || payload.name}　绝对世界坐标`;
      const rocheMessages = response.roche?.messages ?? [];
      if (rocheMessages.length) {
        setStatus(`洛希极限问题：${rocheMessages.join("；")}`, true);
      } else {
        setStatus("预览已更新 · 尚未保存");
      }
    } catch (error) {
      if (!opened || serial !== previewSerial || token !== session) return;
      setStatus(error.message || "预览失败", true);
      if (!lastPreview) {
        empty.hidden = false;
        empty.textContent = "预览不可用，请检查左侧参数";
      }
    }
  }

  for (const group of root.querySelectorAll("[data-switch]")) {
    group.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-value]");
      if (!button || committing) return;
      const key = group.dataset.switch;
      if (draft[key] === button.dataset.value) return;
      draft[key] = button.dataset.value;
      syncControls();
      schedulePreview();
    });
  }

  form.addEventListener("input", (event) => {
    const el = event.target.closest("[data-field]");
    if (!el || committing) return;
    const key = el.dataset.field;
    if (key === "spin_axis") {
      draft.spin_axis[Number(el.dataset.index)] = el.valueAsNumber;
    } else if (el.type === "checkbox") {
      draft[key] = el.checked;
      syncControls();
    } else if (el.type === "number" || SCIENTIFIC_NUMBER_FIELDS.has(key)) {
      draft[key] = el.value.trim() === "" ? NaN : Number(el.value);
    } else {
      draft[key] = el.value;
    }
    schedulePreview();
  });

  // ---------------- Three.js: response geometry only ----------------
  const canvas = root.querySelector(".bb-canvas");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x090b10, 1);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(48, 1, AU * 1e-6, AU * 5000);
  camera.up.set(0, 0, 1);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.09;

  const grid = new THREE.GridHelper(AU * 20, 20, 0x343b4d, 0x202634);
  grid.rotation.x = Math.PI / 2;             // ecliptic/reference plane = XY
  grid.material.transparent = true;
  grid.material.opacity = 0.28;
  scene.add(grid);
  const axes = new THREE.AxesHelper(AU * 0.8);
  scene.add(axes);

  const markerGeometry = new THREE.SphereGeometry(1, 20, 12);
  const candidate = new THREE.Mesh(markerGeometry, new THREE.MeshBasicMaterial({ color: bodyDisplayColor({ kind: "body" }) }));
  const parentMarker = new THREE.Mesh(markerGeometry, new THREE.MeshBasicMaterial({ color: 0xe8b27d }));
  const candidateGlow = makeStellarGlow();
  const parentGlow = makeStellarGlow();
  candidate.add(candidateGlow);
  parentMarker.add(parentGlow);
  candidate.visible = parentMarker.visible = false;
  scene.add(candidate, parentMarker);

  const velocityArrow = new THREE.ArrowHelper(
    new THREE.Vector3(1, 0, 0), new THREE.Vector3(), AU * 0.2,
    0x8fd0a0, AU * 0.05, AU * 0.025,
  );
  velocityArrow.visible = false;
  scene.add(velocityArrow);

  let orbitCapacity = 4096;
  let orbitArray = new Float32Array(orbitCapacity * 3);
  const orbitGeometry = new THREE.BufferGeometry();
  orbitGeometry.setAttribute("position", new THREE.BufferAttribute(orbitArray, 3).setUsage(THREE.DynamicDrawUsage));
  orbitGeometry.setDrawRange(0, 0);
  const orbitLine = new THREE.LineLoop(orbitGeometry, new THREE.LineBasicMaterial({ color: 0x7aa2e8 }));
  orbitLine.frustumCulled = false;
  orbitLine.visible = false;
  scene.add(orbitLine);

  const majorGeometry = new THREE.BufferGeometry();
  majorGeometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(6), 3).setUsage(THREE.DynamicDrawUsage));
  majorGeometry.setDrawRange(0, 0);
  const majorLine = new THREE.Line(majorGeometry, new THREE.LineBasicMaterial({ color: 0xe8967a }));
  majorLine.frustumCulled = false;
  majorLine.visible = false;
  scene.add(majorLine);

  let raf = null;
  let frameNeeded = true;
  let latestBounds = null;
  const cameraDirection = new THREE.Vector3(1, -1.5, 1.1).normalize();

  function resize() {
    const width = viewport.clientWidth;
    const height = viewport.clientHeight;
    if (!width || !height) return;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(viewport);

  function boundsFor(response, placement) {
    // The focus is semantic, not the ellipse bounding-box midpoint: parent for
    // an orbital draft, the new body itself for a direct draft (user spec).
    const focus = placement === "orbit" && validVec(response.parent_p)
      ? new THREE.Vector3(...response.parent_p)
      : new THREE.Vector3(...response.p);
    let maxDistance = 0;
    const add = (p) => {
      if (!validVec(p)) return;
      maxDistance = Math.max(maxDistance, focus.distanceTo(new THREE.Vector3(...p)));
    };
    add(response.p);
    if (placement === "orbit") {
      for (const p of response.orbit_points ?? []) add(p);
      for (const p of response.major_axis ?? []) add(p);
    }
    const radius = placement === "direct"
      ? Math.max(response.radius_m * 8.0, AU * 1e-7)
      : Math.max(maxDistance * 1.1, AU * 0.01);
    return { focus, radius };
  }

  function fitPreview() {
    if (!latestBounds) return;
    const { focus, radius } = latestBounds;
    const aspectPenalty = Math.max(1, 1 / Math.max(camera.aspect, 0.1));
    const distance = radius / Math.sin(THREE.MathUtils.degToRad(camera.fov * 0.5)) * 1.3 * aspectPenalty;
    controls.target.copy(focus);
    camera.position.copy(focus).addScaledVector(cameraDirection, distance);
    camera.near = Math.max(AU * 1e-7, radius / 2000);
    camera.far = Math.max(AU * 1000, radius * 100);
    camera.updateProjectionMatrix();
    controls.update();
    frameNeeded = false;
  }

  function drawPreview(response, placement) {
    latestBounds = boundsFor(response, placement);
    const visualScale = latestBounds.radius;
    // TRUE physical radii in metres, never a scene-dependent pseudo radius.
    candidate.scale.setScalar(response.radius_m);
    parentMarker.scale.setScalar(response.parent_radius_m ?? 1.0);

    candidate.visible = validVec(response.p);
    if (candidate.visible) {
      candidate.position.set(response.p[0], response.p[1], response.p[2]);
      candidate.material.color.copy(bodyDisplayColor({
        kind: response.kind, extra: { teff_k: response.star?.teff_k },
      }));
    }

    const direct = placement === "direct";
    velocityArrow.visible = direct && validVec(response.p) && validVec(response.v) &&
      (response.v[0] !== 0 || response.v[1] !== 0 || response.v[2] !== 0);
    if (velocityArrow.visible) {
      const direction = new THREE.Vector3(response.v[0], response.v[1], response.v[2]).normalize();
      const length = visualScale * 0.32;
      velocityArrow.position.copy(candidate.position);
      velocityArrow.setDirection(direction);
      velocityArrow.setLength(length, length * 0.22, length * 0.1);
    }

    parentMarker.visible = !direct && validVec(response.parent_p);
    if (parentMarker.visible) {
      parentMarker.position.set(response.parent_p[0], response.parent_p[1], response.parent_p[2]);
    }
    candidateLabel.textContent = response.name || "未命名";
    parentLabel.textContent = response.parent || "";
    updateStellarGlow(candidateGlow, response.radius_m,
      candidate.visible && response.kind === "star", response.star?.teff_k);
    updateStellarGlow(parentGlow, response.parent_radius_m ?? 1.0,
      parentMarker.visible && response.parent_kind === "star",
      response.parent_teff_k);

    const points = !direct && Array.isArray(response.orbit_points)
      ? response.orbit_points.filter(validVec) : [];
    orbitLine.visible = points.length >= 2;
    if (orbitLine.visible) {
      if (points.length > orbitCapacity) {
        orbitCapacity = 2 ** Math.ceil(Math.log2(points.length));
        orbitArray = new Float32Array(orbitCapacity * 3);
        orbitGeometry.setAttribute("position", new THREE.BufferAttribute(orbitArray, 3).setUsage(THREE.DynamicDrawUsage));
      }
      for (let i = 0; i < points.length; i++) {
        orbitArray[i * 3] = points[i][0];
        orbitArray[i * 3 + 1] = points[i][1];
        orbitArray[i * 3 + 2] = points[i][2];
      }
      orbitGeometry.setDrawRange(0, points.length);
      orbitGeometry.attributes.position.needsUpdate = true;
    } else {
      orbitGeometry.setDrawRange(0, 0);
    }

    const major = !direct && Array.isArray(response.major_axis) ? response.major_axis : [];
    majorLine.visible = major.length >= 2 && validVec(major[0]) && validVec(major[1]);
    if (majorLine.visible) {
      const positions = majorGeometry.attributes.position.array;
      for (let i = 0; i < 2; i++) {
        positions[i * 3] = major[i][0];
        positions[i * 3 + 1] = major[i][1];
        positions[i * 3 + 2] = major[i][2];
      }
      majorGeometry.setDrawRange(0, 2);
      majorGeometry.attributes.position.needsUpdate = true;
    } else {
      majorGeometry.setDrawRange(0, 0);
    }
    fitPreview();                     // automatic semantic center on every draft
  }

  function clearPreviewDrawing() {
    candidate.visible = false;
    parentMarker.visible = false;
    candidateGlow.visible = false;
    parentGlow.visible = false;
    velocityArrow.visible = false;
    orbitLine.visible = false;
    majorLine.visible = false;
    orbitGeometry.setDrawRange(0, 0);
    majorGeometry.setDrawRange(0, 0);
    latestBounds = null;
    lastPreview = null;
    lastPlacement = null;
    frameNeeded = true;
    candidateLabel.hidden = true;
    parentLabel.hidden = true;
    legend.hidden = true;
    legend.textContent = "";
    empty.hidden = false;
    empty.textContent = "等待服务器预览…";
  }

  const projectedLabelPosition = new THREE.Vector3();
  function positionBodyLabel(object, label) {
    if (!object.visible || !label.textContent) {
      label.hidden = true;
      return;
    }
    projectedLabelPosition.copy(object.position).project(camera);
    const x = (projectedLabelPosition.x * 0.5 + 0.5) * viewport.clientWidth;
    const y = (-projectedLabelPosition.y * 0.5 + 0.5) * viewport.clientHeight;
    const visible = projectedLabelPosition.z >= -1 && projectedLabelPosition.z <= 1 &&
      x >= 0 && x <= viewport.clientWidth && y >= 0 && y <= viewport.clientHeight;
    label.hidden = !visible;
    if (visible) {
      label.style.left = `${x}px`;
      label.style.top = `${y - 8}px`;
    }
  }

  function renderFrame() {
    if (!opened) return;
    controls.update();
    renderer.render(scene, camera);
    positionBodyLabel(candidate, candidateLabel);
    positionBodyLabel(parentMarker, parentLabel);
    raf = requestAnimationFrame(renderFrame);
  }

  root.querySelector(".bb-fit").addEventListener("click", () => {
    frameNeeded = true;
    fitPreview();
  });

  function setBusy(busy) {
    committing = busy;
    commitButton.disabled = busy;
    closeButton.disabled = busy;
    commitButton.textContent = busy ? "正在创建…" : "确认创建";
    root.querySelector(".bb-editor").setAttribute("aria-busy", String(busy));
  }

  async function commit() {
    if (!opened || committing) return;
    const payload = buildPayload();
    const invalid = validatePayload(payload);
    if (invalid) {
      setStatus(invalid, true);
      return;
    }
    clearTimeout(previewTimer);
    previewSerial++;                       // no preview may overwrite a commit
    const token = session;
    setBusy(true);
    setStatus("正在创建并保存…");
    try {
      const result = await api.addBody({ ...payload, confirm: true });
      if (result?._unavailable) throw new Error(result.detail || "保存服务暂不可用");
    } catch (error) {
      if (opened && token === session) {
        setBusy(false);
        setStatus(error.message || "创建失败；草稿已保留", true);
      }
      return;                              // keep every draft field untouched
    }
    if (opened && token === session) close();
    try {
      await onCommitted?.();
    } catch (error) {
      console.error("天体已创建，但提交回调失败", error);
    }
  }

  closeButton.addEventListener("click", close);
  commitButton.addEventListener("click", commit);
  root.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && opened && !committing) close();
  });

  function open() {
    session++;
    previewSerial++;
    clearTimeout(previewTimer);
    opened = true;
    committing = false;
    draft = newDraft();
    clearReadouts();
    clearPreviewDrawing();
    renderParents();
    syncControls();
    setBusy(false);
    setStatus("正在准备预览…");
    root.hidden = false;
    controls.enabled = true;
    resize();
    if (raf == null) raf = requestAnimationFrame(renderFrame);
    schedulePreview();
  }

  function close() {
    session++;
    previewSerial++;                       // reject all in-flight readbacks
    clearTimeout(previewTimer);
    opened = false;
    committing = false;
    root.hidden = true;
    controls.enabled = false;
    if (raf != null) cancelAnimationFrame(raf);
    raf = null;
  }

  return {
    open, close,
    get preview() { return lastPreview; },  // read-only diagnostics/testing
    get scene() { return scene; },
    get controls() { return controls; },
    get candidate() { return candidate; },
    get parentMarker() { return parentMarker; },
    get candidateGlow() { return candidateGlow; },
    get parentGlow() { return parentGlow; },
    get velocityArrow() { return velocityArrow; },
    get orbitLine() { return orbitLine; },
  };
}
