"""o2xygen WebUI backend — FastAPI over the o2xygen physics engine.

Worlds are files (ADR-0016): [worldname].oxygen in WORLDS_DIR. Each world
is a flat N-body system with Cartesian initial conditions; time is seconds
since the world's time origin (displayed as yyyy-mm-dd From origin).
"""
import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "server"))

from o2xygen.constants import AU, DAY, L_SUN, M_EARTH, M_SUN, R_EARTH_MEAN, R_SUN, gm, G
from o2xygen.kepler import Elements, elements_to_state
from o2xygen.system import System, Body, STAR, BODY
from o2xygen.temperature import equilibrium_temperature
from o2xygen.nbody import IntegrateBudgetExceeded
from o2xygen import mass_radius, structure as struct_mod
from o2xygen import world as world_mod

WEB_DIR = ROOT / "web"
DOCS_DIR = ROOT / "docs"
WORLDS_DIR = Path(r"D:\test3\data\worlds")

VERSION = (DOCS_DIR / "version.txt").read_text(encoding="utf-8").strip()

log = logging.getLogger("o2xygen")
app = FastAPI(title="O2xygen", version=VERSION)


# ------------------------------------------------------------- MIST grid -----
_GRID = {"grid": None, "loading": False, "error": None, "n_tracks": 0}


def _load_grid():
    try:
        import pickle
        from mist_data import ensure_extracted, grid_dirs
        from o2xygen.stars import MistGrid
        # bundled data lives inside the repo: .data/MIST-compact (chunked
        # volumes, in git) -> .data/MIST (extracted on first boot, cached)
        extracted = ensure_extracted(ROOT)
        dirs = {feh: d for feh, d in grid_dirs(extracted).items() if d.is_dir()}
        if not dirs:
            raise FileNotFoundError(f"no MIST grids under {extracted}")
        cache = extracted / "mist_grid_cache.pkl"
        src_mtime = max(d.stat().st_mtime for d in dirs.values())
        if cache.exists() and cache.stat().st_mtime > src_mtime:
            with open(cache, "rb") as f:
                _GRID["grid"] = pickle.load(f)
        else:
            grid = MistGrid(dirs)
            with open(cache, "wb") as f:
                pickle.dump(grid, f, protocol=5)
            _GRID["grid"] = grid
        _GRID["n_tracks"] = sum(len(v[0]) for v in _GRID["grid"]._tracks.values())
    except Exception as e:                       # pragma: no cover
        _GRID["error"] = str(e)
        log.exception("MIST grid load failed")


def _grid_getter():
    if _GRID["grid"] is None and not _GRID["loading"] and _GRID["error"] is None:
        _GRID["loading"] = True
        threading.Thread(target=_load_grid, daemon=True).start()
    return _GRID["grid"]


# ------------------------------------------------------------- workspace -----
from workspace import Workspace          # noqa: E402


def _worlds_dir() -> Path:
    WORLDS_DIR.mkdir(parents=True, exist_ok=True)
    return WORLDS_DIR


def _check_world_name(name: str) -> str:
    """World names become file names — refuse anything that could escape
    WORLDS_DIR (path separators, drive letters, '.'-only names)."""
    name = (name or "").strip()
    if not name:
        raise HTTPException(400, "empty world name")
    if name in (".", "..") or "/" in name or "\\" in name or ":" in name:
        raise HTTPException(400, f"illegal world name {name!r}")
    return name


def _bootstrap_worlds():
    """First run: materialise the built-in 白玫瑰 world as a real file."""
    if not list(_worlds_dir().glob("*.oxygen")):
        from preset import build_preset
        world_mod.save_world(build_preset(), _worlds_dir() / "白玫瑰")


def _load_system(name: str) -> System:
    path = _worlds_dir() / _check_world_name(name)
    if not path.with_suffix(".oxygen").exists():
        raise HTTPException(404, f"no world '{name}'")
    return world_mod.load_world(path)


_bootstrap_worlds()
SYSTEM = _load_system("白玫瑰")
WS = Workspace(SYSTEM, grid_getter=_grid_getter)
CURRENT_WORLD = "白玫瑰"


def _rebuild(system: System, name: str) -> None:
    global SYSTEM, WS, CURRENT_WORLD
    SYSTEM = system
    WS = Workspace(system, grid_getter=_grid_getter)
    CURRENT_WORLD = name


def body_json(b: Body) -> dict:
    return {
        "name": b.name, "kind": b.kind,
        "kind_label": "恒星" if b.kind == STAR else "天体",
        "mass_kg": b.mass_kg, "radius_m": b.radius_m,
        "albedo": b.albedo,
        "rotation_period_s": b.rotation_period_s,
        "luminosity_w": b.luminosity_w,
        "extra": b.extra,
    }


# ------------------------------------------------------------------ API ----
@app.get("/api/system")
def get_system():
    return {"name": SYSTEM.name, "world": CURRENT_WORLD,
            "origin": SYSTEM.origin,
            "version": VERSION,
            "engine": {"integrator": WS.sim.integrator, "dt_s": WS.sim.dt_s},
            "bodies": [body_json(b) for b in SYSTEM.bodies.values()]}


@app.get("/api/state")
def get_state(t_days: float | None = None):
    """Live N-body state: positions, velocities, masses, luminosities,
    osculating elements (vs origin), trails. t_days=None -> current sim
    time (used by the evolution view, where the job drives the clock)."""
    if t_days is None:
        t_days = WS.sim.t_s / DAY
    try:
        return WS.state_at(t_days)
    except IntegrateBudgetExceeded as e:
        raise HTTPException(413, str(e))


@app.get("/api/body/{name}")
def get_body(name: str, t_days: float | None = None):
    b = SYSTEM.bodies.get(name)
    if b is None:
        raise HTTPException(404, f"no body '{name}'")
    try:
        got = WS.flux_at(name, t_days)
    except IntegrateBudgetExceeded as e:
        raise HTTPException(413, str(e))
    ill = got["illumination"]
    osc = got["snapshot"]["elements"].get(name)
    flux = ill["total"]
    T_eq = equilibrium_temperature(flux, albedo=b.albedo, regime="dayside")
    return {
        "body": body_json(b),
        "illumination": ill,
        "flux_wm2": flux,
        "t_eq_k": T_eq,
        "period_days": None if osc is None else osc["period_days"],
        "osculating": osc,
    }


class StructureReq(BaseModel):
    mass_earth: float | None = None        # None -> use catalog mass
    cmf: float = 0.3
    model: str = "prem"                    # prem | zeng2019
    rotation_h: float = 24.0


@app.post("/api/body/{name}/structure")
def structure(name: str, req: StructureReq):
    b = SYSTEM.bodies.get(name)
    if b is None:
        raise HTTPException(404, f"no body '{name}'")
    m_earth = req.mass_earth if req.mass_earth else b.mass_kg / M_EARTH
    model_used = req.model
    try:
        r_earth = mass_radius.radius_earth_r(m_earth, req.cmf, model=req.model)
    except ValueError:
        # small bodies (moons) sit below the PREM fit window but inside the
        # wider Zeng(2019)-class window - serve them with the extended model
        # instead of erroring out of the one-click pipeline
        if req.model == "zeng2019":
            raise
        model_used = "zeng2019"
        try:
            r_earth = mass_radius.radius_earth_r(m_earth, req.cmf,
                                                 model=model_used)
        except ValueError as e:
            raise HTTPException(400, str(e))
    r_m = r_earth * R_EARTH_MEAN
    m_kg = m_earth * M_EARTH
    rho = m_kg / (4 / 3 * np.pi * r_m**3)
    omega = 2 * np.pi / (req.rotation_h * 3600.0)
    mu = gm(m_kg)
    f_slow = struct_mod.flattening_slow_rotation(mu, r_m, omega)
    g_eq, g_pole = struct_mod.gravity_pair(mu, r_m, omega)
    layers = struct_mod.layered_density_table(m_kg, r_m, n_layers=4)
    return {
        "radius_earth_r": r_earth, "radius_m": r_m,
        "mean_density_g_cm3": rho / 1000.0,
        "flattening_slow": f_slow,
        "g_equator": g_eq, "g_pole": g_pole,
        "layers": [{"layer": l["layer"],
                    "r0": l["radius_range_frac"][0],
                    "r1": l["radius_range_frac"][1],
                    "rho_g_cm3": l["mean_density_kg_m3"] / 1000.0}
                   for l in layers],
        "model": model_used,
    }


class TempReq(BaseModel):
    albedo: float = 0.09
    regime: str = "dayside"                # fast_rotator | dayside | substellar
    span_days: float = 3652.5
    n: int = 13


@app.post("/api/body/{name}/temperature")
def temperature(name: str, req: TempReq):
    """Temperature table sampled ALONG the live integration path (ADR-0015):
    the integrator advances to each sample day (stellar evolution applied),
    flux is computed from the integrated state with current luminosities."""
    b = SYSTEM.bodies.get(name)
    if b is None:
        raise HTTPException(404, f"no body '{name}'")

    # NON-DESTRUCTIVE forward sampling from the current sim time, on a
    # restored copy (flux_series_forward). INCIDENT 2026-11-15: sampling on
    # the live sim moved the shared clock - during an evolution this killed
    # the job (sim.t overshot end_days -> 'done') while physics kept moving,
    # and interleaved with any new evolution. Rows carry absolute sim days.
    rows_raw = WS.flux_series_forward(name, req.span_days, req.n)
    rows = [{"d": r["d"], "flux": r["flux"],
             "T": equilibrium_temperature(r["flux"], albedo=req.albedo,
                                          regime=req.regime)} for r in rows_raw]
    dense_raw = WS.flux_series_forward(name, req.span_days, 400)
    dense = [{"d": r["d"], "flux": r["flux"],
              "T": equilibrium_temperature(r["flux"], albedo=req.albedo,
                                           regime=req.regime)} for r in dense_raw]
    return {"rows": rows, "dense": dense, "albedo": req.albedo, "regime": req.regime}


# ------------------------------------------------------------- MIST stars ---
@app.get("/api/star/status")
def star_status():
    return {"ready": _GRID["grid"] is not None,
            "loading": _GRID["loading"], "error": _GRID["error"],
            "n_tracks": _GRID["n_tracks"]}


class StarReq(BaseModel):
    mass_msun: float = 1.29
    feh: float = 0.10
    age_gyr: float = 1.26


@app.post("/api/star/estimate")
def star_estimate(req: StarReq):
    if _GRID["grid"] is None:
        if not _GRID["loading"] and _GRID["error"] is None:
            _GRID["loading"] = True
            threading.Thread(target=_load_grid, daemon=True).start()
        return JSONResponse({"ready": False, "loading": True},
                            status_code=503)
    p = _GRID["grid"].evaluate(req.mass_msun, req.feh, req.age_gyr * 1e9)
    return {"ready": True,
            "mass_msun": p.mass_msun,
            "radius_km": p.radius_m / 1000.0,
            "radius_rsun": p.radius_m / R_SUN,
            "luminosity_lsun": p.luminosity_w / L_SUN,
            "teff_k": p.teff_k,
            "mdot_msun_yr": p.mdot_msun_per_yr,
            "cumulative_loss_msun": req.mass_msun - p.mass_msun}


@app.get("/api/star/ensure")
def star_ensure():
    _grid_getter()
    return star_status()


# ------------------------------------------------------------- pipeline -----
class PipelineReq(BaseModel):
    body: str
    span_days: float = 3652.5
    n_samples: int = 13


@app.post("/api/pipeline")
def pipeline(req: PipelineReq):
    """One-click solve: 结构 -> 大气/光学厚度 -> 温度表 (synchronous v1)."""
    b = SYSTEM.bodies.get(req.body)
    if b is None:
        raise HTTPException(404, f"no body '{req.body}'")
    stages = []
    t0 = time.time()
    cmf = float(b.extra.get("cmf", 0.3))
    sreq = StructureReq(mass_earth=b.mass_kg / M_EARTH, cmf=cmf,
                        rotation_h=(b.rotation_period_s or 24 * 3600) / 3600.0)
    stages.append({"name": "结构与尺寸", "status": "done",
                   "result": structure(req.body, sreq)})
    stages.append({"name": "大气与光学厚度", "status": "done",
                   "result": {"tau0": b.extra.get("atm", {}).get("tau0", 0.32),
                              "note": "气柱幂律 (ExoMol-class 光学厚度待接入)"}})
    treq = TempReq(albedo=b.albedo, span_days=req.span_days, n=req.n_samples)
    stages.append({"name": "温度表", "status": "done",
                   "result": temperature(req.body, treq)})
    return {"body": req.body, "stages": stages,
            "elapsed_s": time.time() - t0}


# ------------------------------------------------------------- add body -----
class AddBodyReq(BaseModel):
    name: str
    mass_earth: float = 1.0
    a_au: float = 1.0
    e: float = 0.0
    inc_deg: float = 0.0
    around: str | None = None              # default: world origin


@app.post("/api/add_body")
def add_body(req: AddBodyReq):
    name = req.name.strip()
    if not name:
        raise HTTPException(400, "请输入名称")
    if name in SYSTEM.bodies:
        raise HTTPException(400, "名称已存在")
    ref_name = req.around or SYSTEM.origin
    if ref_name not in SYSTEM.bodies:
        raise HTTPException(400, f"参考天体 '{ref_name}' 不存在")
    mass = req.mass_earth * M_EARTH
    with WS.lock:
        # place relative to the reference body's CURRENT state (mid-flight
        # placement): the catalog pos0/vel0 become "state at placement time",
        # which is what the save file stores for this body.
        states = WS.sim.live_state()
        r_o, v_o = states[ref_name]
        mu = G * (WS.sim.sim.particles[WS.sim._index[ref_name]].m + mass)
        r_rel, v_rel = elements_to_state(
            Elements(a_m=req.a_au * AU, e=req.e, inc=np.radians(req.inc_deg)),
            mu, 0.0)
        pos = r_o + r_rel
        vel = v_o + v_rel
        SYSTEM.add(Body(name, BODY, mass_kg=mass, pos0=pos.copy(), vel0=vel.copy(),
                        radius_m=R_EARTH_MEAN, albedo=0.1,
                        rotation_period_s=24 * 3600.0,
                        extra={"placed_at_s": WS.sim.t_s}))
        WS.sim.sim.add(m=mass, x=pos[0], y=pos[1], z=pos[2],
                       vx=vel[0], vy=vel[1], vz=vel[2])
        WS.sim._order.append(name)
        WS.sim._index[name] = len(WS.sim._order) - 1
    return {"ok": True, "name": name}


# ------------------------------------------------------------- evolution ----
from evolution import EvolutionJob          # noqa: E402

_EVOLUTION: dict = {"job": None}


def _evolution_active() -> bool:
    j = _EVOLUTION["job"]
    return j is not None and j.active


class EvolveStartReq(BaseModel):
    duration_days: float                     # how long to evolve (sim days)
    speed_days_per_s: float                  # sim days per wall second


@app.post("/api/evolve/start")
def evolve_start(req: EvolveStartReq):
    if _evolution_active():
        raise HTTPException(409, "已有推演在进行中")
    if req.duration_days <= 0 or req.speed_days_per_s <= 0:
        raise HTTPException(400, "推演时长与速度必须为正")
    job = EvolutionJob(WS, _worlds_dir() / CURRENT_WORLD,
                       req.duration_days, req.speed_days_per_s)
    _EVOLUTION["job"] = job
    job.start()
    return job.status()


def _job_or_404() -> EvolutionJob:
    j = _EVOLUTION["job"]
    if j is None or not j.active:
        raise HTTPException(404, "no active evolution")
    return j


@app.post("/api/evolve/pause")
def evolve_pause():
    _job_or_404().pause()
    return _job_or_404().status()


@app.post("/api/evolve/resume")
def evolve_resume():
    _job_or_404().resume()
    return _job_or_404().status()


@app.post("/api/evolve/cancel")
def evolve_cancel():
    # per spec: only a PAUSED evolution may be cancelled outright
    j = _job_or_404()
    if j.state != "paused":
        raise HTTPException(409, "必须先暂停才能取消推演")
    j.cancel()
    return j.status()


class EvolveSpeedReq(BaseModel):
    speed_days_per_s: float


@app.post("/api/evolve/speed")
def evolve_speed(req: EvolveSpeedReq):
    # per spec: speed may change ONLY while paused
    j = _job_or_404()
    if j.state != "paused":
        raise HTTPException(409, "必须先暂停才能修改推演速度")
    j.set_speed(req.speed_days_per_s)
    return j.status()


@app.get("/api/evolve/status")
def evolve_status():
    j = _EVOLUTION["job"]
    return j.status() if j is not None else {"state": "idle"}


@app.get("/api/debug/metrics")
def debug_metrics():
    """Temporary diagnostics: checkpoint archive size, process RSS, lock feel."""
    import os
    try:
        import psutil
        rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1e6
    except Exception:
        rss_mb = -1
    return {"checkpoints": len(WS.sim._cp_times),
            "sim_t_days": WS.sim.t_s / DAY,
            "checkpoint_span_days": (WS.sim._cp_times[-1] - WS.sim._cp_times[0]) / DAY if WS.sim._cp_times else 0,
            "rss_mb": rss_mb}


# ------------------------------------------------------------- worlds -------
@app.get("/api/worlds")
def list_worlds():
    files = sorted(p.stem for p in _worlds_dir().glob("*.oxygen"))
    return {"worlds": files, "current": CURRENT_WORLD}


class WorldReq(BaseModel):
    name: str


@app.post("/api/world/load")
def load_world(req: WorldReq):
    if _evolution_active():
        raise HTTPException(409, "推演进行中,不能切换世界")
    _rebuild(_load_system(req.name), req.name)
    return {"ok": True, "world": req.name}


@app.post("/api/world/save")
def save_world(req: WorldReq):
    if _evolution_active():
        raise HTTPException(409, "推演进行中,不能另存为(推演正在自动覆盖当前存档)")
    name = _check_world_name(req.name) if req.name.strip() else CURRENT_WORLD
    # 另存为 = 备份当前物理状态:pos0/vel0 取实时状态,恒星 age0 相应推进,
    # 使读回的世界与新 epoch 自洽(与推演自动存档同一语义)
    path = world_mod.save_world(WS.snapshot_for_save(), _worlds_dir() / name)
    return {"ok": True, "world": path.stem}


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8501, log_level="warning")
