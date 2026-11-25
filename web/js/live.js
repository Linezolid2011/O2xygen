// Live mode sidebar: observation-point catalog, local conditions and editor.
// The central scene is deliberately owned by main.js; this module only returns
// authoritative observation data through onSelect(name, data).
import { api } from "./api.js";

const AU_M = 149_597_870_700;

function node(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

function finite(value) {
  if (value == null || (typeof value === "string" && !value.trim())) return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function fmt(value, digits = 2) {
  const n = finite(value);
  if (n == null) return "—";
  if (n !== 0 && (Math.abs(n) < 0.001 || Math.abs(n) >= 1e7)) {
    return n.toExponential(3);
  }
  return (Object.is(n, -0) ? 0 : n).toLocaleString("zh-CN", {
    maximumFractionDigits: digits,
  });
}

function coordinate(value, latitude) {
  const n = finite(value);
  if (n == null) return "—";
  const hemisphere = latitude ? (n < 0 ? "S" : "N") : (n < 0 ? "W" : "E");
  return `${fmt(Math.abs(n), 3)}° ${hemisphere}`;
}

function distance(value) {
  const n = finite(value);
  if (n == null) return "—";
  if (Math.abs(n) >= 0.08 * AU_M) return `${fmt(n / AU_M, 4)} AU`;
  if (Math.abs(n) >= 1000) return `${fmt(n / 1000, 2)} km`;
  return `${fmt(n, 1)} m`;
}

function kindLabel(kind) {
  if (kind === "star") return "恒星";
  if (kind === "body") return "天体";
  return kind || "天体";
}

function vectorText(value, exponential = false) {
  if (!Array.isArray(value) || value.length !== 3 || value.some((x) => finite(x) == null)) {
    return "—";
  }
  return `[${value.map((x) => exponential ? Number(x).toExponential(3) : fmt(x, 4)).join(", ")}]`;
}

function pointFrom(raw) {
  if (!raw || typeof raw !== "object") return null;
  const name = typeof raw.name === "string" ? raw.name : String(raw.name ?? "");
  const body = typeof raw.body === "string" ? raw.body : String(raw.body ?? "");
  if (!name || !body) return null;
  return {
    name,
    body,
    latitude_deg: finite(raw.latitude_deg),
    longitude_deg: finite(raw.longitude_deg),
  };
}

function errorMessage(error) {
  if (error instanceof Error && error.message) return error.message;
  return String(error || "未知错误");
}

async function request(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: { Accept: "application/json", ...(options.headers || {}) },
  });
  const text = await response.text();
  let data = null;
  if (text) {
    try { data = JSON.parse(text); }
    catch (parseError) {
      if (response.ok) throw new Error("服务返回了无效的 JSON 数据");
    }
  }
  if (!response.ok) {
    let detail = data?.detail ?? data?.message ?? text.trim();
    if (typeof detail !== "string") detail = JSON.stringify(detail);
    throw new Error(detail || `${response.status} ${response.statusText}`.trim());
  }
  return data ?? {};
}

function kv(parent, key, value, accent = false) {
  const row = node("div", "kv");
  row.append(node("span", "k", key));
  const val = node("span", accent ? "v accent" : "v", value);
  row.append(val);
  parent.append(row);
  return row;
}

function section(title) {
  const sec = node("section", "sec live-section");
  if (title) sec.append(node("div", "sec-title", title));
  return sec;
}

export function initLiveMode({ system, onSelect, onEdit } = {}) {
  if (!system || !Array.isArray(system.bodies)) {
    throw new Error("实况模式需要有效的 system.bodies");
  }

  const catalog = document.getElementById("catalog");
  const count = document.getElementById("catalog-count");
  const filter = document.getElementById("catalog-filter");
  const addButton = document.getElementById("btn-add-body");
  const panel = document.getElementById("panel");
  const hint = document.getElementById("view-hint");
  if (!catalog || !count || !filter || !addButton || !panel) {
    throw new Error("实况模式找不到侧栏所需的 DOM 元素");
  }

  let active = false;
  let points = [];
  let selectedName = null;
  let selectedPoint = null;
  let selectedData = null;
  let saved = null;
  let savedCatalogNodes = [];
  let listSerial = 0;
  let observationSerial = 0;
  let editorSerial = 0;
  let deleteSerial = 0;
  let listController = null;
  let observationController = null;
  let listLoading = false;
  let observationLoading = false;
  let listError = "";
  let deletingName = null;

  let livePanel = null;
  let titleEl = null;
  let subtitleEl = null;
  let refreshButton = null;
  let editButton = null;
  let deleteButton = null;
  let statusEl = null;
  let errorEl = null;
  let contentEl = null;

  let editorMask = null;
  let editorTitle = null;
  let editorForm = null;
  let nameInput = null;
  let bodySelect = null;
  let latitudeInput = null;
  let longitudeInput = null;
  let editorError = null;
  let editorStatus = null;
  let editorSave = null;
  let editorCancel = null;
  let editorClose = null;
  let editorOriginalName = null;
  let editorSaving = false;
  let editorReturnFocus = null;

  function isCurrentObservation(serial, name) {
    return active && serial === observationSerial && selectedName === name;
  }

  function updateControls() {
    if (refreshButton) refreshButton.disabled = !active || listLoading || observationLoading;
    if (editButton) editButton.disabled = !active || !selectedPoint || deletingName != null;
    if (deleteButton) {
      deleteButton.disabled = !active || !selectedPoint || deletingName != null;
      deleteButton.textContent = deletingName != null ? "删除中…" : "删除观察点";
    }
  }

  function clearError() {
    if (!errorEl) return;
    errorEl.textContent = "";
    errorEl.hidden = true;
  }

  function showError(message) {
    if (!errorEl || !active) return;
    errorEl.textContent = message;
    errorEl.hidden = false;
  }

  function setObservationLoading(loading) {
    observationLoading = loading;
    if (statusEl) {
      statusEl.hidden = !loading;
      statusEl.textContent = loading ? "正在读取观察点实况…" : "";
    }
    if (livePanel) livePanel.setAttribute("aria-busy", String(loading));
    updateControls();
  }

  function updateHeader(point = selectedPoint) {
    if (!titleEl) return;
    titleEl.textContent = point?.name || selectedName || "—";
    subtitleEl.textContent = point
      ? `${point.body} · ${coordinate(point.latitude_deg, true)} · ${coordinate(point.longitude_deg, false)}`
      : "从左侧选择观察点";
  }

  function renderEmpty(message = "从左侧选择观察点，或新增一个观察点。") {
    if (!contentEl) return;
    updateHeader(null);
    const empty = node("div", "live-empty");
    empty.append(node("div", "live-empty-mark", "⌖"));
    empty.append(node("p", "", message));
    contentEl.replaceChildren(empty);
    updateControls();
  }

  function renderCatalog() {
    if (!active) return;
    count.textContent = String(points.length);
    const query = filter.value.trim().toLowerCase();
    const rows = [];
    if (listError) {
      const errorRow = node("li", "live-catalog-error");
      errorRow.append(node("span", "err", `观察点列表读取失败：${listError}`));
      rows.push(errorRow);
    }
    const visible = points.filter((point) => !query ||
      point.name.toLowerCase().includes(query) || point.body.toLowerCase().includes(query));
    for (const point of visible) {
      const item = node("li", "live-obs-item");
      if (point.name === selectedName) item.classList.add("selected");
      item.dataset.name = point.name;
      const button = node("button", "live-obs-select");
      button.type = "button";
      button.title = `${point.name} · ${point.body}`;
      button.setAttribute("aria-label", `选择观察点 ${point.name}`);
      if (point.name === selectedName) button.setAttribute("aria-current", "true");

      const top = node("span", "live-obs-top");
      top.append(node("span", "live-obs-name", point.name));
      top.append(node("span", "live-obs-body", point.body));
      const coords = node("span", "live-obs-coords",
        `${coordinate(point.latitude_deg, true)} · ${coordinate(point.longitude_deg, false)}`);
      button.append(top, coords);
      button.addEventListener("click", () => { void select(point.name); });
      item.append(button);
      rows.push(item);
    }
    if (!visible.length && !listError) {
      const empty = node("li", "live-catalog-empty",
        query ? "没有匹配的观察点" : "暂无观察点");
      rows.push(empty);
    }
    catalog.replaceChildren(...rows);
  }

  function renderPointPreview() {
    if (!contentEl) return;
    updateHeader();
    if (!selectedPoint) {
      renderEmpty(selectedName ? "正在读取观察点…" : undefined);
      return;
    }
    const base = section("观察点");
    kv(base, "所属天体", selectedPoint.body);
    kv(base, "纬度", coordinate(selectedPoint.latitude_deg, true));
    kv(base, "经度", coordinate(selectedPoint.longitude_deg, false));
    contentEl.replaceChildren(base);
    updateControls();
  }

  function bodyAngles(body) {
    let altitude = finite(body.altitude_deg);
    let azimuth = finite(body.azimuth_deg);
    const enu = body.direction_enu;
    if ((altitude == null || azimuth == null) && Array.isArray(enu) && enu.length === 3) {
      const east = finite(enu[0]);
      const north = finite(enu[1]);
      const up = finite(enu[2]);
      if (east != null && north != null && up != null) {
        const horizontal = Math.hypot(east, north);
        altitude = Math.atan2(up, horizontal) * 180 / Math.PI;
        azimuth = (Math.atan2(east, north) * 180 / Math.PI + 360) % 360;
      }
    }
    return { altitude, azimuth };
  }

  function renderBodies(dataBodies) {
    const sec = section("系统天体 · 本地地平坐标");
    const table = node("table", "live-body-table");
    const head = node("thead");
    const headerRow = node("tr");
    for (const label of ["天体", "高度", "方位", "遮挡"]) {
      headerRow.append(node("th", "", label));
    }
    head.append(headerRow);
    const tbody = node("tbody");

    const byName = new Map((Array.isArray(dataBodies) ? dataBodies : [])
      .filter((b) => b && b.name != null).map((b) => [String(b.name), b]));
    const merged = system.bodies.filter((known) => byName.has(known.name)).map((known) => ({
      ...byName.get(known.name),
      name: known.name,
      kind: byName.get(known.name)?.kind ?? known.kind,
    }));
    for (const [name, body] of byName) {
      if (!system.bodies.some((known) => known.name === name)) merged.push(body);
    }
    if (!merged.length) {
      sec.append(node("div", "note", "暂无其他天体的方向数据。"));
      return sec;
    }

    for (const body of merged) {
      const { altitude, azimuth } = bodyAngles(body);
      const above = typeof body.above_horizon === "boolean"
        ? body.above_horizon : altitude != null && altitude >= 0;
      const occluders = Array.isArray(body.occluded_by) ? body.occluded_by : [];
      const names = occluders.map((item) => typeof item === "object" ? item?.name : item)
        .filter((item) => item != null).map(String);
      const row = node("tr");
      if (!above) row.classList.add("below-horizon");
      const nameCell = node("td");
      const nameLine = node("div", "live-body-name");
      nameLine.append(node("span", "", body.name ?? "—"));
      nameLine.append(node("small", "", kindLabel(body.kind)));
      const meta = [];
      if (finite(body.distance_m) != null) meta.push(distance(body.distance_m));
      if (finite(body.angular_radius_rad) != null) {
        meta.push(`视半径 ${fmt(Number(body.angular_radius_rad) * 180 / Math.PI, 3)}°`);
      }
      if (body.kind === "star" && finite(body.teff_k) != null) {
        meta.push(`${fmt(body.teff_k, 0)} K`);
      }
      nameCell.append(nameLine);
      if (meta.length) nameCell.append(node("div", "live-body-meta", meta.join(" · ")));
      row.append(nameCell);
      row.append(node("td", "live-number", altitude == null ? "—" : `${fmt(altitude, 2)}°`));
      row.append(node("td", "live-number", azimuth == null ? "—" : `${fmt(azimuth, 2)}°`));
      let visibility = "无数据";
      let statusClass = "muted";
      if (altitude != null || typeof body.above_horizon === "boolean") {
        if (!above) visibility = "地面以下";
        else if (names.length) {
          visibility = `被 ${names.join("、")} 遮挡`;
          statusClass = "blocked";
        } else {
          visibility = "无遮挡";
          statusClass = "clear";
        }
      }
      const status = node("td", `live-visibility ${statusClass}`, visibility);
      if (names.length) status.title = names.join("、");
      row.append(status);
      tbody.append(row);
    }
    table.append(head, tbody);
    const wrap = node("div", "live-table-wrap");
    wrap.append(table);
    sec.append(wrap);
    sec.append(node("div", "note", "方位角自北向东；地面以下的天体不参与遮挡判定。"));
    return sec;
  }

  function illuminationValue(value) {
    const direct = finite(value);
    if (direct != null) return direct;
    if (value && typeof value === "object") {
      for (const key of ["total", "illumination_wm2", "value", "flux_wm2"]) {
        const candidate = finite(value[key]);
        if (candidate != null) return candidate;
      }
    }
    return null;
  }

  function renderIllumination(data) {
    const sec = section("多星照明");
    const illumination = data.illumination && typeof data.illumination === "object"
      ? data.illumination : {};
    sec.append(node("div", "live-subsection-title", "大气顶入射"));
    const seen = new Set();
    let sum = 0;
    for (const [name, raw] of Object.entries(illumination)) {
      if (name === "total") continue;
      const value = illuminationValue(raw);
      if (value == null) continue;
      seen.add(name);
      sum += value;
      kv(sec, name, `${fmt(value, 5)} W/m²`);
    }
    for (const body of Array.isArray(data.bodies) ? data.bodies : []) {
      if (body?.kind !== "star" || seen.has(String(body.name))) continue;
      const value = finite(body.illumination_wm2);
      if (value == null) continue;
      seen.add(String(body.name));
      sum += value;
      kv(sec, String(body.name), `${fmt(value, 5)} W/m²`);
    }
    const total = finite(illumination.total) ?? (seen.size ? sum : null);
    if (total != null) kv(sec, "大气顶总辐照", `${fmt(total, 5)} W/m²`, true);
    if (total == null && !seen.size) sec.append(node("div", "note", "暂无恒星照明数据。"));

    const surface = data.surface_irradiance_wm2;
    if (surface && typeof surface === "object") {
      const surfaceBox = node("div", "live-illumination-surface");
      surfaceBox.append(node("div", "live-subsection-title", "地表直射 · 几何、遮挡与大气衰减后"));
      let rows = 0;
      for (const [name, raw] of Object.entries(surface)) {
        if (name === "total") continue;
        const value = illuminationValue(raw);
        if (value == null) continue;
        kv(surfaceBox, name, `${fmt(value, 5)} W/m²`);
        rows++;
      }
      const surfaceTotal = illuminationValue(surface.total);
      if (surfaceTotal != null) {
        kv(surfaceBox, "地表直射合计", `${fmt(surfaceTotal, 5)} W/m²`, true);
        rows++;
      }
      if (rows) sec.append(surfaceBox);
    }
    return sec;
  }

  const ATMOSPHERE_FIELDS = [
    ["surface_pressure_pa", "地表气压"],
    ["pressure_pa", "气压"],
    ["p0_bar", "参考气压"],
    ["surface_temperature_k", "参考地表温度"],
    ["temperature_k", "大气温度"],
    ["scale_height_m", "标高"],
    ["height_m", "大气顶高度"],
    ["mean_molecular_weight", "平均分子量"],
    ["aerosol_scale_height_m", "气溶胶标高"],
    ["vertical_optical_depth", "垂直光学厚度"],
    ["optical_depth", "光学厚度"],
    ["infrared_optical_depth", "红外光学厚度"],
    ["aerosol_optical_depth", "气溶胶光学厚度"],
    ["aerosol_anisotropy", "气溶胶各向异性"],
    ["transmittance", "天顶透过率"],
    ["zenith_transmittance", "天顶透过率"],
  ];

  function atmosphereValue(key, value) {
    if (key === "mean_molecular_weight") return `${fmt(value, 3)} g/mol`;
    if (key.endsWith("_pa")) return `${fmt(value, 4)} Pa`;
    if (key.endsWith("_bar")) return `${fmt(value, 5)} bar`;
    if (key.endsWith("_m")) return distance(value);
    if (key.endsWith("_k")) return `${fmt(value, 2)} K`;
    if (key.includes("transmittance")) return `${fmt(Number(value) * 100, 2)}%`;
    return fmt(value, 5);
  }

  function renderAtmosphere(atmosphere) {
    const sec = section("大气摘要");
    if (atmosphere == null) {
      sec.append(node("div", "note", "无大气数据；该天体可能未配置大气。"));
      return sec;
    }
    if (typeof atmosphere !== "object") {
      sec.append(node("div", "note", String(atmosphere)));
      return sec;
    }
    let rows = 0;
    const used = new Set();
    for (const [key, label] of ATMOSPHERE_FIELDS) {
      const value = finite(atmosphere[key]);
      if (value == null) continue;
      kv(sec, label, atmosphereValue(key, value));
      used.add(key);
      rows++;
    }
    // Small schema additions remain visible without requiring a UI update.
    for (const [key, raw] of Object.entries(atmosphere)) {
      if (rows >= 8 || used.has(key) || ["summary", "reason"].includes(key)) continue;
      if (finite(raw) == null) continue;
      kv(sec, key.replaceAll("_", " "), atmosphereValue(key, raw));
      rows++;
    }
    if (atmosphere.composition && typeof atmosphere.composition === "object") {
      const composition = Object.entries(atmosphere.composition)
        .map(([gas, fraction]) => [gas, finite(fraction)])
        .filter(([, fraction]) => fraction != null)
        .slice(0, 6)
        .map(([gas, fraction]) => `${gas} ${fmt(fraction * 100, 3)}%`);
      if (composition.length) sec.append(node("div", "note", `组成：${composition.join(" · ")}`));
    }
    if (typeof atmosphere.summary === "string") {
      sec.append(node("div", "note", atmosphere.summary));
    } else if (typeof atmosphere.reason === "string") {
      sec.append(node("div", "note", atmosphere.reason));
    } else if (!rows) {
      sec.append(node("div", "note", "已配置大气，但暂无可显示的摘要字段。"));
    }
    return sec;
  }

  function renderObserverDetails(observer, worldPosition) {
    if (!observer || typeof observer !== "object") return null;
    const details = node("details", "live-observer-details");
    details.append(node("summary", "", "观测位置与参考系"));
    const box = node("div", "live-observer-box");
    const position = observer.position_m ?? worldPosition ?? observer.surface_m;
    if (Array.isArray(position)) kv(box, "世界位置 (m)", vectorText(position, true));
    if (Array.isArray(observer.zenith_world)) kv(box, "天顶方向", vectorText(observer.zenith_world));
    if (Array.isArray(observer.east_world)) kv(box, "东向基矢", vectorText(observer.east_world));
    if (Array.isArray(observer.north_world)) kv(box, "北向基矢", vectorText(observer.north_world));
    const phase = finite(observer.phase_rad);
    if (phase != null) {
      const rawDegrees = phase * 180 / Math.PI;
      const degrees = (rawDegrees % 360 + 360) % 360;
      kv(box, "自转相位", `${fmt(phase, 4)} rad · ${fmt(degrees, 2)}°`);
    }
    if (!box.childNodes.length) return null;
    details.append(box);
    return details;
  }

  function renderObservation(data) {
    if (!contentEl) return;
    updateHeader();
    const base = section("观察点");
    const point = selectedPoint || {};
    kv(base, "所属天体", point.body || "—");
    kv(base, "纬度", coordinate(point.latitude_deg, true));
    kv(base, "经度", coordinate(point.longitude_deg, false));
    const time = finite(data.t_days);
    kv(base, "世界时刻", time == null ? "—" : `${fmt(time, 4)} 天`);

    const temp = section("本地温度");
    const temperature = data.temperature && typeof data.temperature === "object"
      ? data.temperature : {};
    kv(temp, "平衡温度", finite(temperature.equilibrium_k) == null ? "—" : `${fmt(temperature.equilibrium_k, 2)} K`);
    kv(temp, "本地昼侧温度", finite(temperature.local_dayside_k) == null ? "—" : `${fmt(temperature.local_dayside_k, 2)} K`, true);
    kv(temp, "参考表面温度", finite(temperature.reference_surface_k) == null ? "—" : `${fmt(temperature.reference_surface_k, 2)} K`);
    if (finite(temperature.estimated_surface_k) != null) {
      kv(temp, "估算地表温度", `${fmt(temperature.estimated_surface_k, 2)} K`);
    }

    const sections = [base, temp, renderBodies(data.bodies), renderIllumination(data),
      renderAtmosphere(data.atmosphere)];
    const observer = renderObserverDetails(data.observer, data.position_m);
    if (observer) sections.push(observer);
    contentEl.replaceChildren(...sections);
    updateControls();
  }

  function cancelObservation() {
    observationSerial++;
    observationController?.abort();
    observationController = null;
    if (observationLoading) setObservationLoading(false);
  }

  function loadObservation(name, { clear = false } = {}) {
    if (!active || !name) return Promise.resolve(null);
    const serial = ++observationSerial;
    observationController?.abort();
    const controller = new AbortController();
    observationController = controller;
    if (clear) {
      selectedData = null;
      clearError();
      renderPointPreview();
    }
    setObservationLoading(true);
    return request(`/api/observation?name=${encodeURIComponent(name)}`, { signal: controller.signal })
      .then(async (data) => {
        if (!isCurrentObservation(serial, name)) return null;
        const responsePoint = pointFrom(data?.point);
        if (responsePoint) {
          selectedPoint = responsePoint;
          const index = points.findIndex((point) => point.name === name);
          if (index >= 0) points[index] = responsePoint;
          renderCatalog();
        } else if (!selectedPoint) {
          throw new Error("服务未返回有效的观察点信息");
        }
        selectedData = data;
        clearError();
        renderObservation(data);
        // main.js owns OrbitScene.setSky(data.sky) and all camera behavior.
        if (typeof onSelect === "function") {
          try { await onSelect(name, data); }
          catch (error) {
            if (isCurrentObservation(serial, name)) {
              showError(`实况数据已更新，但天空接入失败：${errorMessage(error)}`);
            }
          }
        }
        return data;
      })
      .catch((error) => {
        if (error?.name === "AbortError" || !isCurrentObservation(serial, name)) return null;
        showError(`实况读取失败：${errorMessage(error)}`);
        return null;
      })
      .finally(() => {
        if (serial === observationSerial) {
          observationController = null;
          setObservationLoading(false);
        }
      });
  }

  function clearSelection() {
    cancelObservation();
    selectedName = null;
    selectedPoint = null;
    selectedData = null;
    clearError();
    renderCatalog();
    renderEmpty();
  }

  function select(name) {
    if (name == null || name === "") {
      if (active) clearSelection();
      else {
        selectedName = null;
        selectedPoint = null;
        selectedData = null;
      }
      return Promise.resolve(null);
    }
    selectedName = String(name);
    selectedPoint = points.find((point) => point.name === selectedName) ||
      (selectedPoint?.name === selectedName ? selectedPoint : null);
    if (!active) return Promise.resolve(null);
    renderCatalog();
    return loadObservation(selectedName, { clear: true });
  }

  function refresh() {
    if (!active) return Promise.resolve(null);
    const serial = ++listSerial;
    listController?.abort();
    const controller = new AbortController();
    listController = controller;
    listLoading = true;
    listError = "";
    catalog.setAttribute("aria-busy", "true");
    updateControls();

    let loaded = false;
    return request("/api/observations", { signal: controller.signal })
      .then((data) => {
        if (!active || serial !== listSerial) return;
        if (!Array.isArray(data?.points)) throw new Error("服务未返回 points 数组");
        points = data.points.map(pointFrom).filter(Boolean);
        loaded = true;
        renderCatalog();
      })
      .catch((error) => {
        if (error?.name === "AbortError" || !active || serial !== listSerial) return;
        listError = errorMessage(error);
        renderCatalog();
      })
      .finally(() => {
        if (serial === listSerial) {
          listController = null;
          listLoading = false;
          catalog.setAttribute("aria-busy", "false");
          updateControls();
        }
      })
      .then(() => {
        if (!active || serial !== listSerial) return null;
        const selectedExists = selectedName && points.some((point) => point.name === selectedName);
        if (loaded && !selectedExists) {
          if (points.length) return select(points[0].name);
          clearSelection();
          return null;
        }
        if (!selectedName && points.length) return select(points[0].name);
        if (!selectedName) {
          clearSelection();
          return null;
        }
        selectedPoint = points.find((point) => point.name === selectedName) || selectedPoint;
        return loadObservation(selectedName, { clear: !selectedData });
      });
  }

  function triggerEditor(point) {
    clearError();
    if (typeof onEdit !== "function") {
      openEditor(point);
      return;
    }
    try {
      const result = onEdit(point || null, point ? selectedData : null);
      if (result && typeof result.then === "function") {
        result.catch((error) => showError(`无法打开观察点编辑器：${errorMessage(error)}`));
      }
    } catch (error) {
      showError(`无法打开观察点编辑器：${errorMessage(error)}`);
    }
  }

  async function removeObservation(name) {
    if (typeof api.observationDelete === "function") return api.observationDelete(name);
    // Standalone fallback for the server's confirmed-mutation route. The
    // integration wrapper uses this same POST contract.
    return request("/api/observations/delete?confirm=true", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirm: true, name }),
    });
  }

  function deleteSelected() {
    if (!active || !selectedPoint || deletingName != null) return Promise.resolve(false);
    const name = selectedPoint.name;
    if (!window.confirm(`确定删除观察点“${name}”吗？\n\n此操作会立即保存，无法撤销。`)) {
      return Promise.resolve(false);
    }
    const serial = ++deleteSerial;
    deletingName = name;
    clearError();
    updateControls();
    return removeObservation(name)
      .then(async () => {
        if (!active || serial !== deleteSerial) return false;
        points = points.filter((point) => point.name !== name);
        if (selectedName === name) {
          cancelObservation();
          selectedName = null;
          selectedPoint = null;
          selectedData = null;
        }
        deletingName = null;
        await refresh();
        return true;
      })
      .catch((error) => {
        if (active && serial === deleteSerial) showError(`删除观察点失败：${errorMessage(error)}`);
        return false;
      })
      .finally(() => {
        if (serial === deleteSerial) {
          deletingName = null;
          updateControls();
        }
      });
  }

  function ensurePanel() {
    if (livePanel) return;
    livePanel = node("section", "live-panel");
    livePanel.setAttribute("aria-label", "观察点实况");
    const head = node("header", "live-head");
    const titleBox = node("div", "live-title-box");
    const titleRow = node("div", "live-title-row");
    titleEl = node("h2", "live-title", "—");
    titleRow.append(titleEl, node("span", "kind-chip", "实况"));
    subtitleEl = node("div", "live-subtitle", "从左侧选择观察点");
    titleBox.append(titleRow, subtitleEl);
    refreshButton = node("button", "icon-btn", "↻");
    refreshButton.type = "button";
    refreshButton.title = "刷新观察点与实况";
    refreshButton.setAttribute("aria-label", "刷新实况");
    refreshButton.addEventListener("click", () => { void refresh(); });
    head.append(titleBox, refreshButton);

    const actions = node("div", "live-actions");
    editButton = node("button", "btn ghost btn-sm", "编辑观察点");
    editButton.type = "button";
    editButton.addEventListener("click", () => triggerEditor(selectedPoint));
    deleteButton = node("button", "btn danger btn-sm", "删除观察点");
    deleteButton.type = "button";
    deleteButton.addEventListener("click", () => { void deleteSelected(); });
    actions.append(editButton, deleteButton);

    statusEl = node("div", "note live-status", "正在读取观察点实况…");
    statusEl.hidden = true;
    statusEl.setAttribute("role", "status");
    errorEl = node("div", "err live-error");
    errorEl.hidden = true;
    errorEl.setAttribute("role", "alert");
    contentEl = node("div", "live-content");
    livePanel.append(head, actions, statusEl, errorEl, contentEl);
  }

  function editorField(labelText, input) {
    const label = node("label", "live-editor-field");
    label.append(node("span", "", labelText), input);
    return label;
  }

  function ensureEditor() {
    if (editorMask) return;
    editorMask = node("div", "modal-mask live-editor-mask");
    editorMask.hidden = true;
    const modal = node("div", "modal live-editor");
    modal.setAttribute("role", "dialog");
    modal.setAttribute("aria-modal", "true");
    modal.setAttribute("aria-labelledby", "live-editor-title");
    const head = node("div", "modal-head");
    const heading = node("div");
    heading.append(node("div", "modal-eyebrow", "O2XYGEN LIVE OBSERVATION"));
    editorTitle = node("h3", "", "新增观察点");
    editorTitle.id = "live-editor-title";
    heading.append(editorTitle);
    editorClose = node("button", "icon-btn", "✕");
    editorClose.type = "button";
    editorClose.title = "取消";
    editorClose.setAttribute("aria-label", "取消");
    editorClose.addEventListener("click", () => closeEditor());
    head.append(heading, editorClose);

    editorForm = node("form", "live-editor-form");
    editorForm.id = "live-editor-form";
    editorForm.noValidate = true;
    nameInput = node("input");
    nameInput.type = "text";
    nameInput.required = true;
    nameInput.maxLength = 120;
    nameInput.autocomplete = "off";
    bodySelect = node("select");
    bodySelect.required = true;
    latitudeInput = node("input");
    latitudeInput.type = "number";
    latitudeInput.min = "-90";
    latitudeInput.max = "90";
    latitudeInput.step = "any";
    latitudeInput.required = true;
    latitudeInput.inputMode = "decimal";
    longitudeInput = node("input");
    longitudeInput.type = "number";
    longitudeInput.min = "-180";
    longitudeInput.max = "180";
    longitudeInput.step = "any";
    longitudeInput.required = true;
    longitudeInput.inputMode = "decimal";
    editorForm.append(
      editorField("名称", nameInput),
      editorField("所属天体", bodySelect),
    );
    const coordinates = node("div", "live-editor-coordinates");
    coordinates.append(
      editorField("纬度（−90…90°）", latitudeInput),
      editorField("经度（−180…180°，东经为正）", longitudeInput),
    );
    editorForm.append(coordinates);
    editorForm.append(node("p", "modal-note", "位置固定于天体表面；经度向东增加，0° E 会作为精确的零值保存。"));
    editorForm.addEventListener("submit", saveEditor);

    const foot = node("div", "modal-foot live-editor-foot");
    editorStatus = node("span", "live-editor-status");
    editorStatus.setAttribute("role", "status");
    editorError = node("span", "err live-editor-error");
    editorError.setAttribute("role", "alert");
    editorCancel = node("button", "btn ghost", "取消");
    editorCancel.type = "button";
    editorCancel.addEventListener("click", () => closeEditor());
    editorSave = node("button", "btn accent", "保存观察点");
    editorSave.type = "submit";
    // Native association keeps the button in the footer, outside the form.
    editorSave.setAttribute("form", editorForm.id);
    foot.append(editorStatus, editorError, editorCancel, editorSave);
    modal.append(head, editorForm, foot);
    editorMask.append(modal);
    editorMask.addEventListener("click", (event) => {
      if (event.target === editorMask) closeEditor();
    });
    editorMask.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        closeEditor();
      }
    });
    document.body.append(editorMask);
  }

  function populateBodies(preferred) {
    bodySelect.replaceChildren();
    const bodies = system.bodies.filter((body) => body.kind !== "star");
    for (const body of bodies) {
      const option = node("option", "", `${body.name} · ${kindLabel(body.kind)}`);
      option.value = body.name;
      bodySelect.append(option);
    }
    if (preferred && !bodies.some((body) => body.name === preferred)) {
      const option = node("option", "", `${preferred} · 当前所属天体`);
      option.value = preferred;
      bodySelect.append(option);
    }
    if (!bodySelect.options.length) {
      const option = node("option", "", "暂无可用天体");
      option.value = "";
      bodySelect.append(option);
    }
    const fallback = preferred || selectedPoint?.body;
    bodySelect.value = fallback && [...bodySelect.options].some((option) => option.value === fallback)
      ? fallback : bodySelect.options[0]?.value || "";
  }

  function openEditor(point = null) {
    if (!active) return;
    ensureEditor();
    editorSerial++;
    editorOriginalName = point?.name ?? null;
    editorSaving = false;
    editorReturnFocus = document.activeElement;
    editorTitle.textContent = point ? "编辑观察点" : "新增观察点";
    nameInput.value = point?.name ?? "";
    populateBodies(point?.body);
    // Nullish coalescing, rather than ||, preserves an exact 0° E.
    latitudeInput.value = String(finite(point?.latitude_deg) ?? 0);
    longitudeInput.value = String(finite(point?.longitude_deg) ?? 0);
    editorError.textContent = "";
    editorStatus.textContent = "";
    setEditorBusy(false);
    editorMask.hidden = false;
    requestAnimationFrame(() => {
      if (!editorMask.hidden) {
        nameInput.focus();
        if (point) nameInput.select();
      }
    });
  }

  function closeEditor(force = false) {
    if (!editorMask || (editorSaving && !force)) return;
    editorSerial++;
    editorSaving = false;
    editorMask.hidden = true;
    editorError.textContent = "";
    editorStatus.textContent = "";
    if (editorReturnFocus?.isConnected && active) editorReturnFocus.focus();
    editorReturnFocus = null;
  }

  function coordinateInput(input, label, minimum, maximum) {
    if (!input.value.trim()) throw new Error(`请输入${label}`);
    const value = Number(input.value);
    if (!Number.isFinite(value) || value < minimum || value > maximum) {
      throw new Error(`${label}须为 ${minimum}…${maximum} 之间的有限数值`);
    }
    return Object.is(value, -0) ? 0 : value;
  }

  function setEditorBusy(busy) {
    editorSaving = busy;
    editorSave.disabled = busy;
    editorCancel.disabled = busy;
    editorClose.disabled = busy;
    editorSave.textContent = busy ? "正在保存…" : "保存观察点";
    for (const input of [nameInput, bodySelect, latitudeInput, longitudeInput]) {
      input.disabled = busy;
    }
  }

  async function saveEditor(event) {
    event.preventDefault();
    if (!active || editorSaving) return;
    editorError.textContent = "";
    editorStatus.textContent = "";
    let payload;
    try {
      const name = nameInput.value.trim();
      if (!name) throw new Error("请输入观察点名称");
      if (!bodySelect.value) throw new Error("请选择所属天体");
      payload = {
        confirm: true,
        name,
        body: bodySelect.value,
        latitude_deg: coordinateInput(latitudeInput, "纬度", -90, 90),
        longitude_deg: coordinateInput(longitudeInput, "经度", -180, 180),
      };
      if (editorOriginalName != null) payload.original_name = editorOriginalName;
    } catch (error) {
      editorError.textContent = errorMessage(error);
      return;
    }

    const serial = ++editorSerial;
    setEditorBusy(true);
    editorStatus.textContent = "正在保存…";
    try {
      await request("/api/observations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!active || serial !== editorSerial) return;
      editorMask.hidden = true;
      editorSaving = false;
      setEditorBusy(false);
      editorReturnFocus = null;
      cancelObservation();
      selectedName = payload.name;
      selectedPoint = pointFrom(payload);
      selectedData = null;
      await refresh();
    } catch (error) {
      if (!active || serial !== editorSerial) return;
      editorStatus.textContent = "";
      editorError.textContent = `保存失败：${errorMessage(error)}`;
      setEditorBusy(false);
    }
  }

  function onFilterInput(event) {
    if (!active) return;
    // catalog.js owns the same input in body modes. Do not let its renderer
    // replace the live rows before this capture-phase handler finishes.
    event.stopImmediatePropagation();
    renderCatalog();
  }

  function onAddClick() {
    if (!active || addButton.disabled) return;
    triggerEditor(null);
  }

  addButton.addEventListener("click", onAddClick);

  function enter() {
    if (active) return refresh();
    active = true;
    ensurePanel();
    const title = document.getElementById("catalog-head-title") ||
      document.querySelector(".catalog-head > span:first-child");
    const originals = [".panel-head", "#panel-tabs", "#panel-body"]
      .map((selector) => panel.querySelector(selector)).filter(Boolean);
    saved = {
      title,
      titleText: title?.textContent,
      countText: count.textContent,
      placeholder: filter.placeholder,
      filterValue: filter.value,
      addText: addButton.textContent,
      addTitle: addButton.getAttribute("title"),
      addAccent: addButton.classList.contains("accent"),
      addDisabled: addButton.disabled,
      hintText: hint?.textContent,
      originals: originals.map((el) => ({
        el,
        hidden: el.hidden,
        display: el.style.display || "",
      })),
    };
    savedCatalogNodes = [...catalog.childNodes];
    catalog.replaceChildren();
    if (title) title.textContent = "观察点";
    count.textContent = String(points.length);
    filter.value = "";
    filter.placeholder = "搜索观察点…";
    filter.addEventListener("input", onFilterInput, true);
    addButton.textContent = "＋ 观察点";
    addButton.title = "新增观察点";
    addButton.classList.add("accent");
    // main.js enables this after its body mode is fully switched to live.
    addButton.disabled = true;
    if (hint) hint.textContent = "拖动环视 · 滚轮缩放 · Home 回到天顶";
    for (const { el } of saved.originals) {
      el.hidden = true;
      el.style.display = "none";
    }
    panel.append(livePanel);
    clearError();
    updateHeader();
    renderCatalog();
    if (selectedPoint) renderPointPreview();
    else renderEmpty();
    updateControls();
    return refresh();
  }

  function exit() {
    if (!active) return;
    active = false;
    listSerial++;
    observationSerial++;
    editorSerial++;
    deleteSerial++;
    listController?.abort();
    observationController?.abort();
    listController = null;
    observationController = null;
    listLoading = false;
    observationLoading = false;
    deletingName = null;
    filter.removeEventListener("input", onFilterInput, true);
    closeEditor(true);
    livePanel?.remove();
    catalog.replaceChildren(...savedCatalogNodes);
    savedCatalogNodes = [];
    if (saved) {
      if (saved.title) saved.title.textContent = saved.titleText ?? "天体目录";
      count.textContent = saved.countText;
      filter.placeholder = saved.placeholder;
      filter.value = saved.filterValue;
      addButton.textContent = saved.addText;
      if (saved.addTitle == null) addButton.removeAttribute("title");
      else addButton.setAttribute("title", saved.addTitle);
      addButton.classList.toggle("accent", saved.addAccent);
      addButton.disabled = saved.addDisabled;
      if (hint) hint.textContent = saved.hintText ?? "";
      for (const original of saved.originals) {
        original.el.hidden = original.hidden;
        original.el.style.display = original.display;
      }
    }
    saved = null;
    catalog.setAttribute("aria-busy", "false");
  }

  return {
    enter,
    exit,
    select,
    refresh,
    openEditor,
    get selectedName() { return selectedName; },
    get selectedPoint() { return selectedPoint; },
  };
}
