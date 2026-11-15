// Bottom timeline. Two modes (user spec, 2026-11-15):
//   idle     — the bar is INERT: tick labels are all "NaN", the play button
//              is greyed and clicks do nothing. Time only moves via 推演.
//   evolve   — during evolution the bar shows the real time scale and the
//              progress (evolved portion); the play button is orange and
//              toggles evolution pause/resume.
// The "[time] From origin" clock always shows the current sim time.
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

  let mode = "idle";
  let totalDays = 0;

  function renderTicks(labels) {
    ticksEl.innerHTML = "";
    for (const t of labels) {
      const s = document.createElement("span");
      s.textContent = t;
      ticksEl.appendChild(s);
    }
  }

  function setIdle() {
    mode = "idle";
    slider.disabled = true;
    slider.max = 1000;
    slider.value = 0;
    playBtn.disabled = true;
    playBtn.textContent = "▶";
    speedIn.disabled = true;
    renderTicks(["NaN", "NaN", "NaN", "NaN", "NaN", "NaN"]);
  }

  function setEvolve(durationDays) {
    mode = "evolve";
    totalDays = durationDays;
    slider.disabled = true;                 // 进度条不可拖动,只显示占比
    slider.max = 1000;
    slider.value = 0;
    playBtn.disabled = false;
    playBtn.textContent = "❚❚";
    speedIn.disabled = true;
    const labels = [];
    for (let i = 0; i <= 5; i++) labels.push(fmtTick((totalDays * i) / 5));
    renderTicks(labels);
  }

  playBtn.addEventListener("click", () => {
    if (mode === "evolve") callbacks.onEvolveToggle?.();
  });

  setIdle();

  return {
    setIdle,
    setEvolve,
    setProgress(p) { slider.value = Math.max(0, Math.min(1, p)) * 1000; },
    setPlaying(on) { playBtn.textContent = on ? "❚❚" : "▶"; },
    setClock(days) { clock.textContent = fmtFromOrigin(days); },
    get spanDays() { return (parseFloat(spanIn.value) || 10) * YEAR_D; },
  };
}
