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

from o2xygen.constants import AU, L_SUN, M_EARTH, M_SUN, R_EARTH_MEAN, R_SUN, gm, G
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
        from o2xygen.stars import MistGrid
        dirs = {}
        base = Path(r"D:\test3\data\mist")
        for tag, feh in (("feh_p0.00", 0.0), ("feh_p0.25", 0.25)):
            if (base / tag).is_dir():
                dirs[feh] = base / tag
        if not dirs:
            raise FileNotFoundError(f"no MIST grids under {base}")
        cache = base / "mist_grid_cache.pkl"
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


def _bootstrap_worlds():
    """First run: materialise the built-in 白玫瑰 world as a real file."""
    if not list(_worlds_dir().glob("*.oxygen")):
        from preset import build_preset
        world_mod.save_world(build_preset(), _worlds_dir() / "白玫瑰")


def _load_system(name: str) -> System:
    path = _worlds_dir() / name
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
def get_state(t_days: float = 0.0):
    """Live N-body state at t_days: positions, velocities, masses,
    luminosities, osculating elements (vs origin), trails."""
    try:
        return WS.state_at(t_days)
    except IntegrateBudgetExceeded as e:
        raise HTTPException(413, str(e))


@app.get("/api/body/{name}")
def get_body(name: str, t_days: float = 0.0):
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
    try:
        r_earth = mass_radius.radius_earth_r(m_earth, req.cmf, model=req.model)
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
        "model": req.model,
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

    def sample(d):
        try:
            ill = WS.flux_at(name, d)["illumination"]
        except IntegrateBudgetExceeded as e:
            raise HTTPException(413, str(e))
        return {"d": d, "flux": ill["total"],
                "T": equilibrium_temperature(ill["total"], albedo=req.albedo,
                                             regime=req.regime)}

    rows = [sample(req.span_days * i / max(req.n - 1, 1)) for i in range(req.n)]
    n_dense = 400
    dense = [sample(req.span_days * i / (n_dense - 1)) for i in range(n_dense)]
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


# ------------------------------------------------------------- worlds -------
@app.get("/api/worlds")
def list_worlds():
    files = sorted(p.stem for p in _worlds_dir().glob("*.oxygen"))
    return {"worlds": files, "current": CURRENT_WORLD}


class WorldReq(BaseModel):
    name: str


@app.post("/api/world/load")
def load_world(req: WorldReq):
    _rebuild(_load_system(req.name), req.name)
    return {"ok": True, "world": req.name}


@app.post("/api/world/save")
def save_world(req: WorldReq):
    name = req.name.strip() or CURRENT_WORLD
    path = world_mod.save_world(SYSTEM, _worlds_dir() / name)
    return {"ok": True, "world": path.stem}


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8501, log_level="warning")
