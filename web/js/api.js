// Thin fetch wrappers around the FastAPI backend.
export async function errDetail(r) {
  // non-OK bodies may be plain text ("Internal Server Error") - never let
  // JSON.parse of that kill the app with a confusing SyntaxError
  try {
    const d = await r.json();
    return d.detail?.toString() || r.statusText;
  } catch {
    return `${r.status} ${r.statusText}`;
  }
}
async function jpost(url, payload) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (r.status === 503) return { ...(await r.json()), _unavailable: true };
  if (!r.ok) throw new Error(await errDetail(r));
  return r.json();
}
async function jget(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(await errDetail(r));
  return r.json();
}

export const api = {
  system: () => jget("/api/system"),
  state: (tDays) => tDays == null ? jget("/api/state")
                                  : jget(`/api/state?t_days=${tDays}`),
  // tDays=null -> 'current time' resolved INSIDE the server lock, so a
  // slow poll never scrubs the live clock back to its stale timestamp
  body: (name, tDays) => tDays == null
    ? jget(`/api/body/${encodeURIComponent(name)}`)
    : jget(`/api/body/${encodeURIComponent(name)}?t_days=${tDays}`),
  structure: (name, req) => jpost(`/api/body/${encodeURIComponent(name)}/structure`, req),
  temperature: (name, req) => jpost(`/api/body/${encodeURIComponent(name)}/temperature`, req),
  starStatus: () => jget("/api/star/status"),
  starEnsure: () => jget("/api/star/ensure"),
  starEstimate: (req) => jpost("/api/star/estimate", req),
  starEdit: (req) => jpost("/api/star/edit", req),
  starPreview: (req) => jpost("/api/star/preview", req),
  pipeline: (req) => jpost("/api/pipeline", req),
  worlds: () => jget("/api/worlds"),
  worldLoad: (name) => jpost("/api/world/load", { name }),
  worldSave: (name) => jpost("/api/world/save", { name }),
  astroFrame: () => jget("/api/astroframe"),
  astroFrameSave: (req) => jpost("/api/astroframe", req),
  sky: (body, mlim = 8.0) =>
    jget(`/api/sky?body=${encodeURIComponent(body)}&mlim=${mlim}`),
  evolveStart: (durationDays, speedDps) =>
    jpost("/api/evolve/start", { duration_days: durationDays, speed_days_per_s: speedDps }),
  evolvePause: () => jpost("/api/evolve/pause", {}),
  evolveResume: () => jpost("/api/evolve/resume", {}),
  evolveCancel: () => jpost("/api/evolve/cancel", {}),
  evolveSpeed: (speedDps) => jpost("/api/evolve/speed", { speed_days_per_s: speedDps }),
  evolveStatus: () => jget("/api/evolve/status"),
};
