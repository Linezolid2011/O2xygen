"""o2xygen WebUI backend — stdlib HTTP server over the o2xygen physics engine.

No web framework (ADR-0020): Python's http.server (ThreadingHTTPServer) is
sufficient — the API is a handful of JSON endpoints plus static hosting for
web/, and the workspace lock already serialises the physics.

Worlds are files (ADR-0016): [worldname].oxygen in WORLDS_DIR. Each world
is a flat N-body system with Cartesian initial conditions; time is seconds
since the world's time origin (displayed as yyyy-mm-dd From origin).

API contract (unchanged from the FastAPI era — the client depends on it):
  GET  /api/system                       world + engine + body catalog
  GET  /api/state[?t_days=]              snapshot (None = live clock)
  GET  /api/body/{name}[?t_days=]        irradiation + osculating elements
  POST /api/body/{name}/structure        mass-radius/gravity/layering
  POST /api/body/{name}/temperature      forward temperature table
  GET  /api/star/status | /api/star/ensure
  POST /api/star/estimate                MIST interpolation (503 while loading)
  POST /api/pipeline                     one-click solve
  POST /api/add_body                     (409 during evolution)
  POST /api/evolve/start|pause|resume|cancel|speed, GET /api/evolve/status
  GET  /api/debug/metrics
  GET  /api/worlds, POST /api/world/load, POST /api/world/save
Errors are JSON {"detail": msg} with 400/404/409/413/503 as before.
"""
import json
import logging
import mimetypes
import os
import socket
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import numpy as np

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
from o2xygen.astroframe import (normalize_astro_frame, position_radec,
                                world_to_icrs_pc, julian_year, quat_rotate)

WEB_DIR = ROOT / "web"
DOCS_DIR = ROOT / "docs"
# worlds are user content, kept with the app's other mutable data under
# .data/ (gitignored); overridable for tests/deployments
WORLDS_DIR = Path(os.environ.get("O2YGEN_WORLDS_DIR", ROOT / ".data" / "worlds"))

VERSION = (DOCS_DIR / "version.txt").read_text(encoding="utf-8").strip()

log = logging.getLogger("o2xygen")


class HttpError(Exception):
    """Endpoint-level error -> JSON {"detail": ...} with an HTTP status."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


# ------------------------------------------------------------- MIST grid -----
_GRID = {"grid": None, "loading": False, "error": None, "n_tracks": 0,
         "wd_grid": None, "wd_error": None, "n_wd_tracks": 0}


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
        try:
            from o2xygen.wd_mist3 import Mist3WDGrid
            wd_path = ROOT / ".data" / "mist3_wd.npz"
            if not wd_path.exists():
                raise FileNotFoundError(f"no MIST III WD grid at {wd_path}")
            _GRID["wd_grid"] = Mist3WDGrid(wd_path)
            _GRID["n_wd_tracks"] = _GRID["wd_grid"].n_tracks
        except Exception as e:
            _GRID["wd_error"] = str(e)
            log.exception("MIST III WD grid load failed")
    except Exception as e:                       # pragma: no cover
        _GRID["error"] = str(e)
        log.exception("MIST grid load failed")
    finally:
        # without this, /api/star/status reported "loading" forever after
        # the first attempt (success OR failure), and no retry could start
        _GRID["loading"] = False


def _grid_getter():
    if _GRID["grid"] is None and not _GRID["loading"] and _GRID["error"] is None:
        _GRID["loading"] = True
        threading.Thread(target=_load_grid, daemon=True).start()
    return _GRID["grid"]


def _evaluate_stellar(mini_msun: float, feh: float, age_yr: float):
    """Unified lifecycle: pre-WD MIST, then MIST III WD cooling."""
    wd = _GRID["wd_grid"]
    if wd is not None and wd.is_wd_age(mini_msun, feh, age_yr):
        return wd.evaluate(mini_msun, feh, age_yr)
    grid = _grid_getter()
    if grid is None:
        raise HttpError(503, "MIST 网格尚未就绪")
    return grid.evaluate(mini_msun, feh, age_yr)


# ------------------------------------------------------------- workspace -----
from workspace import Workspace          # noqa: E402


def _worlds_dir() -> Path:
    WORLDS_DIR.mkdir(parents=True, exist_ok=True)
    return WORLDS_DIR


def _check_world_name(name: str) -> str:
    """World names become file names — refuse anything that could escape
    WORLDS_DIR or corrupt the .oxygen suffix mapping (path separators,
    drive letters, dots, '.'-only names)."""
    name = (name or "").strip()
    if not name:
        raise HttpError(400, "empty world name")
    if name in (".", "..") or any(c in name for c in '/\\:.'):
        raise HttpError(400, f"illegal world name {name!r}")
    return name


def _bootstrap_worlds():
    """First run: materialise the built-in 白玫瑰 world as a real file."""
    if not list(_worlds_dir().glob("*.oxygen")):
        from preset import build_preset
        world_mod.save_world(build_preset(), _worlds_dir() / "白玫瑰")


def _load_system(name: str) -> System:
    path = _worlds_dir() / _check_world_name(name)
    if not path.with_suffix(".oxygen").exists():
        raise HttpError(404, f"no world '{name}'")
    return world_mod.load_world(path)


_bootstrap_worlds()
SYSTEM = _load_system("白玫瑰")
WS = Workspace(SYSTEM, grid_getter=_grid_getter,
               wd_grid_getter=lambda: _GRID["wd_grid"])
CURRENT_WORLD = "白玫瑰"


def _rebuild(system: System, name: str) -> None:
    global SYSTEM, WS, CURRENT_WORLD
    SYSTEM = system
    WS = Workspace(system, grid_getter=_grid_getter,
                   wd_grid_getter=lambda: _GRID["wd_grid"])
    CURRENT_WORLD = name


def body_json(b: Body) -> dict:
    return {
        "name": b.name, "kind": b.kind,
        "kind_label": "恒星" if b.kind == STAR else "天体",
        "mass_kg": b.mass_kg, "radius_m": b.radius_m,
        "albedo": b.albedo,
        "rotation_period_s": b.rotation_period_s,
        "spin_axis": [float(x) for x in b.spin_axis],
        "spin_rate_rad_s": float(b.spin_rate_rad_s),
        "spin_phase_rad": float(b.spin_phase_rad),
        "luminosity_w": b.luminosity_w,
        "extra": b.extra,
    }


# ------------------------------------------------------------- evolution ----
from evolution import EvolutionJob          # noqa: E402

_EVOLUTION: dict = {"job": None}


def _evolution_active() -> bool:
    j = _EVOLUTION["job"]
    return j is not None and j.active


def _job_or_404() -> EvolutionJob:
    j = _EVOLUTION["job"]
    if j is None or not j.active:
        raise HttpError(404, "no active evolution")
    return j


# --------------------------------------------------------- request helpers ---
def _body_float(body: dict, key: str, default=None, required: bool = False) -> float:
    """float coercion for a JSON field (pydantic's old role, minus the magic)."""
    v = body.get(key, None)
    if v is None:
        if required:
            raise HttpError(400, f"missing required field '{key}'")
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        raise HttpError(400, f"field '{key}' must be a number")


def _body_str(body: dict, key: str, default=None, required: bool = False) -> str:
    v = body.get(key, None)
    if v is None:
        if required:
            raise HttpError(400, f"missing required field '{key}'")
        return default
    if not isinstance(v, str):
        raise HttpError(400, f"field '{key}' must be a string")
    return v


def _query_float(query: dict, key: str) -> float | None:
    v = query.get(key)
    if not v:
        return None
    try:
        return float(v[0])
    except (TypeError, ValueError):
        raise HttpError(400, f"query parameter '{key}' must be a number")


# ------------------------------------------------------------------ API ------
def api_get_system():
    return {"name": SYSTEM.name, "world": CURRENT_WORLD,
            "origin": SYSTEM.origin,
            "version": VERSION,
            "engine": {"integrator": WS.sim.integrator, "dt_s": WS.sim.dt_s},
            "bodies": [body_json(b) for b in SYSTEM.bodies.values()]}


def api_get_state(query):
    """Live N-body state. NON-DESTRUCTIVE (ADR-0018): t_days=None -> current
    sim time; any explicit t_days is answered from a checkpoint-restored copy
    and never moves the shared clock."""
    try:
        return WS.state_at(_query_float(query, "t_days"))
    except IntegrateBudgetExceeded as e:
        raise HttpError(413, str(e))


def api_get_body(name, query):
    b = SYSTEM.bodies.get(name)
    if b is None:
        raise HttpError(404, f"no body '{name}'")
    try:
        got = WS.flux_at(name, _query_float(query, "t_days"))
    except IntegrateBudgetExceeded as e:
        raise HttpError(413, str(e))
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


def api_structure(name, body):
    b = SYSTEM.bodies.get(name)
    if b is None:
        raise HttpError(404, f"no body '{name}'")
    mass_earth = _body_float(body, "mass_earth")
    cmf = _body_float(body, "cmf", 0.3)
    model = _body_str(body, "model", "prem")
    rotation_h = _body_float(body, "rotation_h", 24.0)
    m_earth = mass_earth if mass_earth else b.mass_kg / M_EARTH
    model_used = model
    try:
        r_earth = mass_radius.radius_earth_r(m_earth, cmf, model=model)
    except ValueError:
        # small bodies (moons) sit below the PREM fit window but inside the
        # wider Zeng(2019)-class window - serve them with the extended model
        # instead of erroring out of the one-click pipeline
        if model == "zeng2019":
            raise
        model_used = "zeng2019"
        try:
            r_earth = mass_radius.radius_earth_r(m_earth, cmf, model=model_used)
        except ValueError as e:
            raise HttpError(400, str(e))
    r_m = r_earth * R_EARTH_MEAN
    m_kg = m_earth * M_EARTH
    rho = m_kg / (4 / 3 * np.pi * r_m**3)
    omega = 2 * np.pi / (rotation_h * 3600.0)
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


def api_temperature(name, body):
    """Temperature table sampled ALONG the integration path (ADR-0015), on a
    restored copy - never moves the live clock (2026-11-15 desync incident).
    One dense pass serves both the chart and the table."""
    b = SYSTEM.bodies.get(name)
    if b is None:
        raise HttpError(404, f"no body '{name}'")
    albedo = _body_float(body, "albedo", 0.09)
    regime = _body_str(body, "regime", "dayside")
    span_days = _body_float(body, "span_days", 3652.5)
    n_req = _body_float(body, "n", 13)

    dense_raw = WS.flux_series_forward(name, span_days, 400)
    def to_row(r):
        return {"d": r["d"], "flux": r["flux"],
                "T": equilibrium_temperature(r["flux"], albedo=albedo, regime=regime)}
    dense = [to_row(r) for r in dense_raw]
    n = max(2, min(int(n_req), len(dense_raw)))
    idx = np.linspace(0, len(dense_raw) - 1, n).round().astype(int)
    rows = [dense[i] for i in idx]
    return {"rows": rows, "dense": dense, "albedo": albedo, "regime": regime}


# ------------------------------------------------------------- MIST stars ---
def api_star_status():
    return {"ready": _GRID["grid"] is not None,
            "loading": _GRID["loading"], "error": _GRID["error"],
            "n_tracks": _GRID["n_tracks"],
            "wd_ready": _GRID["wd_grid"] is not None,
            "wd_error": _GRID["wd_error"],
            "n_wd_tracks": _GRID["n_wd_tracks"]}


def api_star_estimate(body):
    """Returns (payload, status): 503 with a loading notice while the grid is
    still being built (the client maps 503 to a retry hint)."""
    if _GRID["grid"] is None:
        if not _GRID["loading"] and _GRID["error"] is None:
            _GRID["loading"] = True
            threading.Thread(target=_load_grid, daemon=True).start()
        return {"ready": False, "loading": True}, 503
    mass_msun = _body_float(body, "mass_msun", 1.29)
    feh = _body_float(body, "feh", 0.10)
    age_gyr = _body_float(body, "age_gyr", 1.26)
    try:
        p = _evaluate_stellar(mass_msun, feh, age_gyr * 1e9)
    except ValueError as e:
        # out-of-grid mass/feh/age is a client error, not a server crash
        raise HttpError(400, str(e))
    return {"ready": True,
            "mass_msun": p.mass_msun,
            "radius_km": p.radius_m / 1000.0,
            "radius_rsun": p.radius_m / R_SUN,
            "luminosity_lsun": p.luminosity_w / L_SUN,
            "teff_k": p.teff_k,
            "mdot_msun_yr": p.mdot_msun_per_yr,
            "cumulative_loss_msun": mass_msun - p.mass_msun}, 200


# ------------------------------------------------------------- star editor ---
def api_star_edit(body):
    """Save a stellar draft from edit mode. Only MIST inputs (initial mass,
    Fe/H, epoch age) and spin are editable. Current mass/radius/luminosity/
    Teff are MIST OUTPUTS and are never accepted as editable inputs. Off-grid
    inputs are clamped to the nearest covered point by MistGrid.
    Position/velocity remain immutable."""
    if _evolution_active():
        raise HttpError(409, "推演进行中,不能编辑恒星")
    name = _body_str(body, "name", required=True)
    b = SYSTEM.bodies.get(name)
    if b is None:
        raise HttpError(404, f"no body '{name}'")
    if b.kind != STAR:
        raise HttpError(400, "只能编辑恒星")
    forbidden = {"mass_msun", "radius_rsun", "luminosity_lsun", "teff_k",
                 "use_mist"} & set(body)
    if forbidden:
        raise HttpError(400, "当前质量/半径/光度/有效温度是 MIST 输出,只读")
    mini = _body_float(body, "mini_msun", b.extra.get("mini_msun", b.mass_kg / M_SUN))
    feh = _body_float(body, "feh", b.extra.get("feh", 0.0))
    age0 = _body_float(body, "age0_gyr", b.extra.get("age0_gyr", 1.0))
    period_h = _body_float(body, "spin_period_h",
                           (b.rotation_period_s or 0.0) / 3600.0)
    if not all(np.isfinite([mini, feh, age0, period_h])):
        raise HttpError(400, "恒星参数必须为有限数值")
    if mini <= 0 or age0 < 0 or period_h < 0:
        raise HttpError(400, "初始质量须为正,年龄/自转周期不得为负")
    axis = np.asarray(body.get("spin_axis", list(b.spin_axis)), dtype=float)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)) \
            or np.linalg.norm(axis) < 1e-15:
        raise HttpError(400, "自转轴必须为三个有限且非零的数值")
    axis = axis / np.linalg.norm(axis)
    rate = 2.0 * np.pi / (period_h * 3600.0) if period_h > 0.0 else 0.0

    with WS.lock:
        b.extra["mini_msun"] = mini
        b.extra["feh"] = feh
        b.extra["age0_gyr"] = age0
        grid = _grid_getter()
        if grid is None:
            raise HttpError(503, "MIST 网格尚未就绪")
        try:
            p = _evaluate_stellar(mini, feh,
                                  age0 * 1e9 + WS.sim.t_s / (365.25 * DAY))
        except ValueError as e:
            raise HttpError(400, str(e))
        # current properties are always computed, never user-supplied
        b.mass_kg = p.mass_msun * M_SUN
        b.radius_m = p.radius_m
        b.luminosity_w = p.luminosity_w
        b.extra["teff_k"] = p.teff_k
        b.extra["luminosity_w"] = b.luminosity_w
        b.set_spin_vector(axis * rate, b.spin_phase_rad)
        WS.sim.update_masses({name: b.mass_kg})
        # archived snapshots contain the OLD mass; never restore them
        WS.sim._cp_times = [WS.sim.t_s]
        WS.sim._cp_snaps = [WS.sim._snapshot()]
        WS._trail_cache = None
    world_mod.save_world(WS.snapshot_for_save(), _worlds_dir() / CURRENT_WORLD)
    return {"ok": True, "body": body_json(b)}


# ------------------------------------------------------------- pipeline -----
def api_pipeline(body):
    """One-click solve: 结构 -> 大气/光学厚度 -> 温度表 (synchronous v1)."""
    name = _body_str(body, "body", required=True)
    span_days = _body_float(body, "span_days", 3652.5)
    n_samples = _body_float(body, "n_samples", 13)
    b = SYSTEM.bodies.get(name)
    if b is None:
        raise HttpError(404, f"no body '{name}'")
    stages = []
    t0 = time.time()
    cmf = float(b.extra.get("cmf", 0.3))
    sreq = {"mass_earth": b.mass_kg / M_EARTH, "cmf": cmf,
            "rotation_h": (b.rotation_period_s or 24 * 3600) / 3600.0}
    stages.append({"name": "结构与尺寸", "status": "done",
                   "result": api_structure(name, sreq)})
    stages.append({"name": "大气与光学厚度", "status": "done",
                   "result": {"tau0": b.extra.get("atm", {}).get("tau0", 0.32),
                              "note": "气柱幂律 (ExoMol-class 光学厚度待接入)"}})
    treq = {"albedo": b.albedo, "span_days": span_days, "n": n_samples}
    stages.append({"name": "温度表", "status": "done",
                   "result": api_temperature(name, treq)})
    return {"body": name, "stages": stages,
            "elapsed_s": time.time() - t0}


# ------------------------------------------------------------- add body -----
def api_add_body(body):
    if _evolution_active():
        # adding a particle mid-flight would desync the job's checkpoints
        # and invalidate its world file (fixed body count per archive)
        raise HttpError(409, "推演进行中,不能添加天体")
    name = _body_str(body, "name", required=True).strip()
    if not name:
        raise HttpError(400, "请输入名称")
    if name in SYSTEM.bodies:
        raise HttpError(400, "名称已存在")
    # no privileged root body anymore (ADR-0022): default placement
    # reference is the most massive body in the world
    ref_name = _body_str(body, "around") \
        or max(SYSTEM.bodies.values(), key=lambda b: b.mass_kg).name
    if ref_name not in SYSTEM.bodies:
        raise HttpError(400, f"参考天体 '{ref_name}' 不存在")
    mass_earth = _body_float(body, "mass_earth", 1.0)
    a_au = _body_float(body, "a_au", 1.0)
    e = _body_float(body, "e", 0.0)
    inc_deg = _body_float(body, "inc_deg", 0.0)
    mass = mass_earth * M_EARTH
    with WS.lock:
        # place relative to the reference body's CURRENT state (mid-flight
        # placement): the catalog pos0/vel0 become "state at placement time",
        # which is what the save file stores for this body.
        states = WS.sim.live_state()
        r_o, v_o = states[ref_name]
        mu = G * (WS.sim.sim.particles[WS.sim._index[ref_name]].m + mass)
        r_rel, v_rel = elements_to_state(
            Elements(a_m=a_au * AU, e=e, inc=np.radians(inc_deg)), mu, 0.0)
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
        # every archived checkpoint predates this body (wrong particle
        # count) - the archive must restart from the enlarged system or a
        # later restore would index past the old snapshots
        WS.sim._cp_times = [WS.sim.t_s]
        WS.sim._cp_snaps = [WS.sim._snapshot()]
        WS._trail_cache = None
    return {"ok": True, "name": name}


# ------------------------------------------------------------- evolution ----
def api_evolve_start(body):
    if _evolution_active():
        raise HttpError(409, "已有推演在进行中")
    duration_days = _body_float(body, "duration_days", required=True)
    speed_days_per_s = _body_float(body, "speed_days_per_s", required=True)
    if duration_days <= 0 or speed_days_per_s <= 0:
        raise HttpError(400, "推演时长与速度必须为正")
    job = EvolutionJob(WS, _worlds_dir() / CURRENT_WORLD,
                       duration_days, speed_days_per_s)
    _EVOLUTION["job"] = job
    job.start()
    return job.status()


def api_evolve_pause():
    _job_or_404().pause()
    return _job_or_404().status()


def api_evolve_resume():
    _job_or_404().resume()
    return _job_or_404().status()


def api_evolve_cancel():
    # per spec: only a PAUSED evolution may be cancelled outright; cancelling
    # ROLLS BACK to the last autosave - the live workspace is rebuilt from
    # the save file so memory and disk can never diverge
    j = _job_or_404()
    if j.state != "paused":
        raise HttpError(409, "必须先暂停才能取消推演")
    j.cancel()
    j.wait(timeout=30)
    status = j.status()
    _rebuild(_load_system(CURRENT_WORLD), CURRENT_WORLD)
    return status


def api_evolve_speed(body):
    # per spec: speed may change ONLY while paused
    j = _job_or_404()
    if j.state != "paused":
        raise HttpError(409, "必须先暂停才能修改推演速度")
    j.set_speed(_body_float(body, "speed_days_per_s", required=True))
    return j.status()


def api_evolve_status():
    j = _EVOLUTION["job"]
    return j.status() if j is not None else {"state": "idle"}


def api_debug_metrics():
    """Diagnostics: checkpoint archive size, process RSS, sim clock."""
    return {"checkpoints": len(WS.sim._cp_times),
            "sim_t_days": WS.sim.t_s / DAY,
            "checkpoint_span_days": (WS.sim._cp_times[-1] - WS.sim._cp_times[0]) / DAY if WS.sim._cp_times else 0,
            "rss_mb": _rss_mb()}


def _rss_mb() -> float:
    """Process RSS in MB without psutil: psapi on Windows, resource on POSIX."""
    try:                                        # Windows: psapi
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = _PMC()
        counters.cb = ctypes.sizeof(_PMC)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        # NB: ctypes defaults truncate/overflow the 64-bit pseudo-handle
        # (ERROR_INVALID_HANDLE / OverflowError) - declare both sides
        get_mem = ctypes.windll.psapi.GetProcessMemoryInfo
        get_mem.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC),
                            wintypes.DWORD]
        get_mem.restype = wintypes.BOOL
        if get_mem(k32.GetCurrentProcess(), ctypes.byref(counters),
                   counters.cb):
            return counters.WorkingSetSize / 1e6
    except Exception:
        pass
    try:                                        # POSIX fallback
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3
    except Exception:
        return -1.0


# ------------------------------------------------------------- worlds -------
def api_list_worlds():
    files = sorted(p.stem for p in _worlds_dir().glob("*.oxygen"))
    return {"worlds": files, "current": CURRENT_WORLD}


def api_world_load(body):
    if _evolution_active():
        raise HttpError(409, "推演进行中,不能切换世界")
    name = _body_str(body, "name", required=True)
    _rebuild(_load_system(name), name)
    return {"ok": True, "world": name}


def api_world_save(body):
    if _evolution_active():
        raise HttpError(409, "推演进行中,不能另存为(推演正在自动覆盖当前存档)")
    raw = _body_str(body, "name", required=True)
    name = _check_world_name(raw) if raw.strip() else CURRENT_WORLD
    # 另存为 = 备份当前物理状态:pos0/vel0 取实时状态,恒星 age0 相应推进,
    # 使读回的世界与新 epoch 自洽(与推演自动存档同一语义)
    path = world_mod.save_world(WS.snapshot_for_save(), _worlds_dir() / name)
    return {"ok": True, "world": path.stem}


# ------------------------------------------------------------- astro frame ---
def api_get_astroframe():
    f = normalize_astro_frame(SYSTEM.astro_frame)
    return {**f,
            "current_jd_tt": f["epoch_jd_tt"] + WS.sim.t_s / DAY,
            "position_radec": position_radec(f)}


def api_save_astroframe(body):
    if _evolution_active():
        raise HttpError(409, "推演进行中,不能修改世界天文坐标")
    try:
        f = normalize_astro_frame(body)
    except (TypeError, ValueError) as e:
        raise HttpError(400, str(e))
    SYSTEM.astro_frame = f
    WS.system.astro_frame = f
    # metadata edits are persisted with the same epoch-consistent snapshot
    # semantics as 另存为/autosave; this also advances epoch_jd_tt correctly
    world_mod.save_world(WS.snapshot_for_save(), _worlds_dir() / CURRENT_WORLD)
    return api_get_astroframe()


# ------------------------------------------------------------- sky catalog ---
_SKY = {"catalog": None}


def _sky_catalog():
    if _SKY["catalog"] is None:
        from o2xygen.sky import HipparcosSky
        path = ROOT / ".data" / "hipparcos.npz"
        if not path.exists():
            raise HttpError(503, "Hipparcos catalog missing; run scripts/import_hipparcos.py")
        _SKY["catalog"] = HipparcosSky(path)
    return _SKY["catalog"]


def api_get_sky(query):
    """Real starfield for the selected body at the CURRENT sim time.
    The catalog is advanced from J1991.25 to the world's absolute epoch,
    reprojected for the observer's heliocentric ICRS position, then rotated
    back into world axes for Three.js. The synthetic Sun obeys the same
    magnitude cutoff as every catalog star."""
    body = (query.get("body") or [None])[0]
    if body is not None and body not in SYSTEM.bodies:
        raise HttpError(404, f"no body '{body}'")
    mlim = _query_float(query, "mlim")
    mlim = 8.0 if mlim is None else max(-5.0, min(12.0, mlim))
    with WS.lock:                             # snapshot only; no integration
        ps = WS.sim.sim.particles
        idx = WS.sim._index
        names = list(idx)
        ms = np.array([ps[idx[n]].m for n in names])
        pos = np.array([ps[idx[n]].xyz for n in names])
        bcm = (ms[:, None] * pos).sum(axis=0) / ms.sum()
        r_rel = (pos[names.index(body)] - bcm) if body is not None else np.zeros(3)
        t_s = WS.sim.t_s
        frame = normalize_astro_frame(SYSTEM.astro_frame)
    observer = world_to_icrs_pc(frame, r_rel, t_s)
    year = julian_year(frame, t_s)
    sky = _sky_catalog().observe(observer, year, mlim)
    dirs = quat_rotate(frame["world_to_icrs_quat_wxyz"], sky["dir_icrs"],
                       inverse=True)
    return {
        "body": body, "t_days": t_s / DAY, "year": year,
        "observer_helio_pc": [float(x) for x in observer],
        "mlim": mlim,
        "hip": [int(x) for x in sky["hip"]],
        "dir_x": [float(x) for x in dirs[:, 0]],
        "dir_y": [float(x) for x in dirs[:, 1]],
        "dir_z": [float(x) for x in dirs[:, 2]],
        "mag": [float(x) for x in sky["mag"]],
        "dist_pc": [float(x) for x in sky["dist_pc"]],
        "bv": [None if not np.isfinite(x) else float(x) for x in sky["bv"]],
        "sun_index": sky["sun_index"],
    }


# ---------------------------------------------------------------- router ----
# ------------------------------------------------------------- SSE stream ---
# Server-driven rendering (ADR-0021): the server is the ONLY computer. The
# client integrates nothing - it draws what this stream emits.
#   event: pose    ~60 Hz, only while the sim clock moves (idle = silent)
#   event: trails  ~5.5 Hz while moving, only when the cached paths changed
#   comment ": hb" every 5 s, liveness + proxy keep-alive
# Meta (elements, star props) stays on the classic 800 ms /api/state poll -
# it was never the problem.
POSE_INTERVAL_S = 1.0 / 60.0
TRAIL_INTERVAL_S = 0.18
HEARTBEAT_S = 5.0


def _sse_frame(event: str, payload: dict) -> bytes:
    return (f"event: {event}\ndata: "
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            + "\n\n").encode("utf-8")


def _sse_loop(h: "Handler") -> None:
    """Emit pose/trail frames until the client goes away. Runs on the
    connection's own thread (ThreadingHTTPServer). Reads the global WS every
    tick, so world switches / cancel-rollbacks re-anchor automatically."""
    last_pose_t = None
    last_trail_check = 0.0
    last_trails_obj = None
    last_hb = time.time()
    while True:
        now = time.time()
        pose = None
        try:
            ws = WS
            t_s = None
            if ws is not None:
                with ws.lock:                    # microsecond hold: array copy
                    t_s = ws.sim.t_s
                    if t_s != last_pose_t:
                        ps = ws.sim.sim.particles
                        mm = np.array([q.m for q in ps])
                        arr = np.array([q.xyz for q in ps])
                        bcm = (mm[:, None] * arr).sum(axis=0) / mm.sum()
                        pose = {"t_days": t_s / DAY,
                                "bcm": [float(x) for x in bcm],
                                "p": {n: [float(ps[i].x), float(ps[i].y), float(ps[i].z)]
                                      for n, i in ws.sim._index.items()}}
                if pose is not None:
                    h.wfile.write(_sse_frame("pose", pose))
                    last_pose_t = t_s
                # trails: drive the server-side (cached) backward integration
                # only while the clock moves; emit only when it recomputed
                if t_s is not None and now - last_trail_check >= TRAIL_INTERVAL_S:
                    last_trail_check = now
                    snap = ws.state_at(None)
                    trails = snap.get("trails")
                    if trails is not None and trails is not last_trails_obj:
                        h.wfile.write(_sse_frame("trails", {"t_days": snap["t_days"],
                                                            "trails": trails}))
                        last_trails_obj = trails
            if now - last_hb >= HEARTBEAT_S:
                h.wfile.write(b": hb\n\n")
                last_hb = now
            h.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError,
                OSError):
            return                                # client went away
        except Exception:
            log.error("SSE emitter error\n%s", traceback.format_exc())
            return
        time.sleep(POSE_INTERVAL_S)


def route_api(method: str, segs: list[str], query: dict, body: dict):
    """Dispatch one /api request -> (payload, status). Raises HttpError."""
    tail = segs[1:]                                   # after "api"
    if method == "GET" and tail == ["system"]:
        return api_get_system(), 200
    if method == "GET" and tail == ["state"]:
        return api_get_state(query), 200
    if len(tail) >= 2 and tail[0] == "body":
        name = tail[1]
        if method == "GET" and len(tail) == 2:
            return api_get_body(name, query), 200
        if method == "POST" and len(tail) == 3 and tail[2] == "structure":
            return api_structure(name, body), 200
        if method == "POST" and len(tail) == 3 and tail[2] == "temperature":
            return api_temperature(name, body), 200
    if tail[:1] == ["star"]:
        if method == "GET" and tail == ["star", "status"]:
            return api_star_status(), 200
        if method == "GET" and tail == ["star", "ensure"]:
            _grid_getter()
            return api_star_status(), 200
        if method == "POST" and tail == ["star", "estimate"]:
            return api_star_estimate(body)
    if method == "POST" and tail == ["star", "edit"]:
        return api_star_edit(body), 200
    if method == "POST" and tail == ["pipeline"]:
        return api_pipeline(body), 200
    if method == "POST" and tail == ["add_body"]:
        return api_add_body(body), 200
    if tail[:1] == ["evolve"]:
        if method == "POST" and tail == ["evolve", "start"]:
            return api_evolve_start(body), 200
        if method == "POST" and tail == ["evolve", "pause"]:
            return api_evolve_pause(), 200
        if method == "POST" and tail == ["evolve", "resume"]:
            return api_evolve_resume(), 200
        if method == "POST" and tail == ["evolve", "cancel"]:
            return api_evolve_cancel(), 200
        if method == "POST" and tail == ["evolve", "speed"]:
            return api_evolve_speed(body), 200
        if method == "GET" and tail == ["evolve", "status"]:
            return api_evolve_status(), 200
    if method == "GET" and tail == ["debug", "metrics"]:
        return api_debug_metrics(), 200
    if method == "GET" and tail == ["worlds"]:
        return api_list_worlds(), 200
    if method == "POST" and tail == ["world", "load"]:
        return api_world_load(body), 200
    if method == "POST" and tail == ["world", "save"]:
        return api_world_save(body), 200
    if method == "GET" and tail == ["astroframe"]:
        return api_get_astroframe(), 200
    if method == "POST" and tail == ["astroframe"]:
        return api_save_astroframe(body), 200
    if method == "GET" and tail == ["sky"]:
        return api_get_sky(query), 200
    raise HttpError(404, "not found")


# ------------------------------------------------------------ HTTP layer ----
_MIME_OVERRIDES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",      # ES modules need a JS type
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".webm": "video/webm",
    ".ico": "image/x-icon",
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"                 # keep-alive for the polling UI
    server_version = f"O2xygen/{VERSION}"

    def log_message(self, fmt, *args):            # uvicorn-quiet, not silent
        log.debug("%s - %s", self.address_string(), fmt % args)

    # -- response helpers --
    def _send_bytes(self, data: bytes, status: int, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _send_json(self, obj, status: int = 200):
        # compact separators, matching the old FastAPI byte shape (and ~15%
        # smaller dense temperature payloads)
        data = json.dumps(obj, ensure_ascii=False,
                          separators=(",", ":")).encode("utf-8")
        self._send_bytes(data, status, "application/json; charset=utf-8")

    # -- request handling --
    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw.strip():
            return {}
        try:
            data = json.loads(raw.decode("utf-8-sig"))   # tolerate a stray BOM
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise HttpError(400, "request body must be valid JSON")
        if not isinstance(data, dict):
            raise HttpError(400, "request body must be a JSON object")
        return data

    def _serve_static(self, path: str):
        rel = path.lstrip("/")
        if rel == "" or rel.endswith("/"):
            rel += "index.html"
        target = (WEB_DIR / rel).resolve()
        if WEB_DIR.resolve() not in target.parents and target != WEB_DIR.resolve():
            raise HttpError(404, "not found")     # ../ escape attempt
        if not target.is_file():
            raise HttpError(404, "not found")
        ctype = _MIME_OVERRIDES.get(target.suffix.lower())
        if ctype is None:
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self._send_bytes(target.read_bytes(), 200, ctype)

    def _serve_sse(self):
        """GET /api/stream: server-sent events, the render pipeline
        (ADR-0021). No Content-Length: the body is 'until disconnect'."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")     # defeat proxy buffering
        self.end_headers()
        self.close_connection = True
        try:
            self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        _sse_loop(self)

    def _dispatch(self, method: str):
        try:
            u = urlsplit(self.path)
            path = unquote(u.path)
            if method == "GET" and path == "/api/stream":
                self._serve_sse()                   # never returns normally
                return
            if path.startswith("/api/"):
                segs = [s for s in path.split("/") if s != ""]
                query = parse_qs(u.query)
                body = self._read_json_body() if method == "POST" else {}
                payload, status = route_api(method, segs, query, body)
                self._send_json(payload, status)
            elif method in ("GET", "HEAD"):
                self._serve_static(path)
            else:
                raise HttpError(404, "not found")
        except HttpError as e:
            self._send_json({"detail": e.detail}, e.status)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass                                    # client went away mid-poll
        except Exception:
            log.error("unhandled error in %s %s\n%s", method, self.path,
                      traceback.format_exc())
            try:
                self._send_json({"detail": "Internal Server Error"}, 500)
            except Exception:
                pass

    def do_GET(self):
        self._dispatch("GET")

    def do_HEAD(self):
        self._dispatch("HEAD")

    def do_POST(self):
        self._dispatch("POST")


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        # Browsers abort keep-alive connections all the time (tab close,
        # reload, poll cancellation) - socketserver would otherwise dump a
        # ConnectionAbortedError traceback per abort into stderr
        log.debug("connection from %s dropped", client_address)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    server = Server(("127.0.0.1", 8501), Handler)
    print(f"O2xygen {VERSION} serving on http://127.0.0.1:8501", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
