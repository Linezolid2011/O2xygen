// Bottom timeline: play/pause, typable speed (天/秒), typable span (年),
// scrub, dynamic tick labels, "yyyy-mm-dd From origin" clock (ADR-0016:
// time = seconds since the world origin; the calendar is a uniform display
// convention - 365.25 d/yr, 12 equal months).
const YEAR_D = 365.25;

export function fmtFromOrigin(days) {
  const MONTH = YEAR_D / 12;
  const neg = days < 0;
  const d = Math.abs(days);
  const y = Math.floor(d / YEAR_D);
  const rem = d - y * YEAR_D;
  const mo = Math.min(12, Math.floor(rem / MONTH) + 1);
  const dy = Math.floor(rem - (mo - 1) * MONTH) + 1;
  const pad = (n) => String(n).padStart(2, "0");
  return `${neg ? "-" : ""}${y}-${pad(mo)}-${pad(dy)} From origin`;
}

function fmtTick(d) {
  const trim = (x) => {
    let s = x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2);
    if (s.includes(".")) s = s.replace(/\.?0+$/, "");   // only strip DECIMAL zeros
    return s;
  };
  if (d <= 0) return "+0";
  if (d >= YEAR_D) return `+${trim(d / YEAR_D)} 年`;
  if (d >= 2) return `+${trim(d)} 天`;
  if (d >= 1 / 24) return `+${trim(d * 24)} 小时`;
  return `+${trim(d * 86400)} 秒`;
}

export function initTimeline(callbacks) {
  const slider = document.getElementById("tl-slider");
  const playBtn = document.getElementById("btn-play");
  const speedIn = document.getElementById("speed");
  const spanIn = document.getElementById("span");
  const ticksEl = document.getElementById("tl-ticks");
  const clock = document.getElementById("clock");

  const state = {
    days: 0,
    span: parseFloat(spanIn.value) * YEAR_D,     // stored in days
    playing: false,
    speed: parseFloat(speedIn.value),            // days per second
  };

  function renderTicks() {
    ticksEl.innerHTML = "";
    for (let i = 0; i <= 5; i++) {
      const s = document.createElement("span");
      s.textContent = fmtTick((state.span * i) / 5);
      ticksEl.appendChild(s);
    }
  }

  function applySpan() {
    const years = parseFloat(spanIn.value);
    if (!isFinite(years) || years <= 0) { spanIn.value = (state.span / YEAR_D); return; }
    state.span = years * YEAR_D;
    slider.max = state.span;
    slider.step = Math.max(state.span / 2000, 0.001);   // keep scrubbing smooth
    state.days = Math.min(state.days, state.span);
    slider.value = state.days;
    renderTicks();
    callbacks.onTimeChanged(state.days);
    updateClock();
  }

  function applySpeed() {
    const v = parseFloat(speedIn.value);
    state.speed = isFinite(v) && v > 0 ? v : 1;
    speedIn.value = state.speed;
  }

  slider.addEventListener("input", () => {
    state.days = parseFloat(slider.value);
    callbacks.onTimeChanged(state.days);
    updateClock();
  });
  speedIn.addEventListener("change", applySpeed);
  speedIn.addEventListener("keydown", (e) => { if (e.key === "Enter") speedIn.blur(); });
  spanIn.addEventListener("change", applySpan);
  spanIn.addEventListener("keydown", (e) => { if (e.key === "Enter") spanIn.blur(); });
  playBtn.addEventListener("click", toggle);
  window.addEventListener("keydown", (e) => {
    if (e.code === "Space" && !/INPUT|SELECT|TEXTAREA|BUTTON|A/.test(document.activeElement.tagName)) {
      e.preventDefault();
      toggle();
    }
  });

  function toggle() {
    state.playing = !state.playing;
    playBtn.textContent = state.playing ? "❚❚" : "▶";
  }

  function updateClock() {
    clock.textContent = fmtFromOrigin(state.days);
  }

  renderTicks();
  updateClock();

  // main-loop advance hook
  let last = performance.now();
  function tick(now) {
    const dt = Math.min((now - last) / 1000, 0.25);
    last = now;
    if (state.playing) {
      state.days += state.speed * dt;
      if (state.days > state.span) {
        state.days = 0;                                // loop the timeline
        callbacks.onWrap?.();                          // force a fresh snapshot
      }
      slider.value = state.days;
      callbacks.onTimeChanged(state.days);
      updateClock();
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);

  return {
    get days() { return state.days; },
    get span() { return state.span; },
  };
}
