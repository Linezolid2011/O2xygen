"""o2xygen WebUI backend — FastAPI over the o2xygen physics engine.

Serves the static frontend (web/) plus a JSON API for everything the
right-hand panel needs: irradiation, structure estimates, temperature
tables, MIST star estimation, and the one-click solve pipeline.
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

from o2xygen.constants import (AU, L_SUN, M_EARTH, M_SUN, R_EARTH_MEAN,
                                 R_SUN, gm)
from o2xygen.kepler import Elements
from o2xygen.system import System, Body, STAR, PLANET, MOON
from o2xygen.temperature import equilibrium_temperature
from o2xygen import mass_radius, structure as struct_mod

WEB_DIR = ROOT / "web"
DOCS_DIR = ROOT / "docs"
EPOCH_UTC = "2026-11-18T19:00:25Z"          # his "初始时刻", from screenshots

# Single source of truth for the version: docs/version.txt
VERSION = (DOCS_DIR / "version.txt").read_text(encoding="utf-8").strip()

log = logging.getLogger("o2xygen")
app = FastAPI(title="O2xygen", version=VERSION)


# ---------------------------------------------------------------- preset ---
def build_preset() -> System:
    """The '白玫瑰' workspace, rebuilt from screenshot fingerprints."""
    sys = System("白玫瑰")
    # his solved star (see tests/test_stars.py): 1.29 Msun, [Fe/H]~0.10, 1.26 Gyr
    sys.add(Body("Αστραία", STAR, mass_kg=1.29 * M_SUN, radius_m=926278.85 * 1000,
                 luminosity_w=2.8055 * L_SUN,
                 extra={"teff_k": 6474.11, "mini_msun": 1.29, "feh": 0.10,
                        "age0_gyr": 1.26}))
    # Vesper: cool dwarf companion (wide binary)
    sys.add(Body("Vesper", STAR, parent="Αστραία",
                 elements=Elements(a_m=60 * AU, e=0.30, inc=np.radians(12),
                                   raan=0.4, argp=1.2, m0=3.5),
                 mass_kg=0.85 * M_SUN, radius_m=0.83 * R_SUN,
                 luminosity_w=0.45 * L_SUN,
                 extra={"teff_k": 5000.0, "mini_msun": 0.85, "feh": 0.0,
                        "age0_gyr": 1.26}))
    # gas giant with rings (screenshot: Saturn-like)
    sys.add(Body("Νεφέλη", PLANET, parent="Αστραία",
                 elements=Elements(a_m=1.6 * AU, e=0.08, inc=np.radians(1.2),
                                   raan=0.1, argp=0.5, m0=0.0),
                 mass_kg=20 * M_EARTH, radius_m=4.0 * R_EARTH_MEAN,
                 albedo=0.34, rotation_period_s=9.9 * 3600,
                 extra={"kind_label": "行星", "ring": True, "cmf": 0.10,
                        "composition_preset": "巨行星"}))
    # its moon (his selected body in screenshot 1)
    sys.add(Body("Αύρα", MOON, parent="Νεφέλη",
                 elements=Elements(a_m=0.01 * AU, e=0.15, inc=np.radians(24.0),
                                   raan=0.9, argp=2.1, m0=1.0),
                 mass_kg=7.3e22, radius_m=0.35 * R_EARTH_MEAN,
                 albedo=0.09, rotation_period_s=24 * 3600,
                 extra={"kind_label": "卫星"}))
    # his terrestrial showcase planet, full stat sheet from screenshots
    sys.add(Body("Múspellsheimr", PLANET, parent="Αστραία",
                 elements=Elements(a_m=7.83 * AU, e=0.1924, inc=np.radians(25.12),
                                   raan=2.2, argp=4.0, m0=2.4),
                 mass_kg=3.15 * M_EARTH, radius_m=1.35032 * R_EARTH_MEAN,
                 albedo=0.20, rotation_period_s=17.1 * 3600,
                 extra={"kind_label": "行星", "ring": True, "cmf": 0.38,
                        "composition_preset": "硅酸盐 + 铁核",
                        "atm": {"gases": [("二氧化碳 CO₂", 91.3), ("氮气 N₂", 8.7)],
                                "tau0": 0.32, "p0_bar": 0.32, "n": 2.0, "S": 0.2,
                                "nabla": 0.0}}))
    return sys


SYSTEM = build_preset()


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
sys.path.insert(0, str(ROOT / "server"))
from workspace import Workspace          # noqa: E402

WS = Workspace(SYSTEM, grid_getter=_grid_getter)


def body_json(b: Body) -> dict:
    return {
        "name": b.name, "kind": b.kind, "parent": b.parent,
        "mass_kg": b.mass_kg, "radius_m": b.radius_m,
        "albedo": b.albedo,
        "rotation_period_s": b.rotation_period_s,
        "luminosity_w": b.luminosity_w,
        "elements": None if b.elements is None else {
            "a_m": b.elements.a_m, "e": b.elements.e,
            "inc": b.elements.inc, "raan": b.elements.raan,
            "argp": b.elements.argp, "m0": b.elements.m0,
        },
        "extra": b.extra,
    }


# ------------------------------------------------------------------ API ----
@app.get("/api/system")
def get_system():
    return {"name": SYSTEM.name, "epoch_utc": EPOCH_UTC, "version": VERSION,
            "engine": {"integrator": WS.sim.integrator, "dt_s": WS.sim.dt_s},
            "bodies": [body_json(b) for b in SYSTEM.bodies.values()]}


@app.get("/api/state")
def get_state(t_days: float = 0.0):
    """Live N-body state at t_days: positions, velocities, masses,
    luminosities, osculating elements. Advances the workspace integrator."""
    return WS.state_at(t_days)


@app.get("/api/body/{name}")
def get_body(name: str, t_days: float = 0.0):
    b = SYSTEM.bodies.get(name)
    if b is None:
        raise HTTPException(404, f"no body '{name}'")
    got = WS.flux_at(name, t_days)
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
        ill = WS.flux_at(name, d)["illumination"]
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
    if _GRID["grid"] is None and not _GRID["loading"] and _GRID["error"] is None:
        _GRID["loading"] = True
        threading.Thread(target=_load_grid, daemon=True).start()
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


class AddBodyReq(BaseModel):
    name: str
    kind: str                              # planet | moon
    parent: str
    a_au: float
    e: float = 0.0
    mass_earth: float = 1.0


@app.post("/api/add_body")
def add_body(req: AddBodyReq):
    if req.name in SYSTEM.bodies:
        raise HTTPException(400, "名称已存在")
    if req.parent not in SYSTEM.bodies:
        raise HTTPException(400, f"环绕天体 '{req.parent}' 不存在")
    if req.kind not in (PLANET, MOON):
        raise HTTPException(400, "类型必须是行星或卫星")
    SYSTEM.add(Body(req.name, req.kind, parent=req.parent,
                    elements=Elements(a_m=req.a_au * AU, e=req.e),
                    mass_kg=req.mass_earth * M_EARTH,
                    albedo=0.1, rotation_period_s=24 * 3600,
                    extra={"kind_label": "行星" if req.kind == PLANET else "卫星"}))
    global WS
    WS = Workspace(SYSTEM, grid_getter=_grid_getter)   # fresh integrator
    return {"ok": True, "name": req.name}


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8501, log_level="warning")
