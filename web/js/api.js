// Thin fetch wrappers around the FastAPI backend.
async function jpost(url, payload) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (r.status === 503) return { ...(await r.json()), _unavailable: true };
  if (!r.ok) throw new Error((await r.json()).detail?.toString() || r.statusText);
  return r.json();
}
async function jget(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(r.statusText);
  return r.json();
}

export const api = {
  system: () => jget("/api/system"),
  state: (tDays) => jget(`/api/state?t_days=${tDays}`),
  body: (name, tDays) => jget(`/api/body/${encodeURIComponent(name)}?t_days=${tDays}`),
  structure: (name, req) => jpost(`/api/body/${encodeURIComponent(name)}/structure`, req),
  temperature: (name, req) => jpost(`/api/body/${encodeURIComponent(name)}/temperature`, req),
  starStatus: () => jget("/api/star/status"),
  starEnsure: () => jget("/api/star/ensure"),
  starEstimate: (req) => jpost("/api/star/estimate", req),
  pipeline: (req) => jpost("/api/pipeline", req),
  worlds: () => jget("/api/worlds"),
  worldLoad: (name) => jpost("/api/world/load", { name }),
  worldSave: (name) => jpost("/api/world/save", { name }),
};
