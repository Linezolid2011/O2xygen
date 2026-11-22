"""Rescue Múspellsheimr onto a wide circumbinary orbit and add Σκιά.

Default: build .data/candidates/白玫瑰-rescue.oxygen only.
--apply: back up the live world, then replace it with the candidate.
The candidate uses a fresh manual-edit conservation epoch (wind ledger reset).
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import pickle
import shutil
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "server"))

from o2xygen.constants import AU, DAY, G, M_EARTH, M_SUN, YEAR
from o2xygen.nbody import rv_to_elements
from o2xygen.stars import MistGrid
from o2xygen.system import Body, BODY
from o2xygen import world as world_mod
from o2xygen.wd_mist3 import Mist3WDGrid
from o2xygen.wdcool import WDCoolGrid
from mist_data import ensure_extracted, grid_dirs
from workspace import Workspace

A_MUSPELL = 300.0 * AU
A_SKIA = 0.01 * AU
R_SKIA = 2_440_000.0
RHO_SKIA = 5_500.0
NAME_SKIA = "Σκιά"


def grids():
    extracted = ensure_extracted(ROOT)
    dirs = {f: d for f, d in grid_dirs(extracted).items() if d.is_dir()}
    cache = extracted / "mist_grid_cache.pkl"
    mtime = max(d.stat().st_mtime for d in dirs.values())
    mist = None
    if cache.exists() and cache.stat().st_mtime > mtime:
        with cache.open("rb") as f:
            candidate = pickle.load(f)
        if getattr(candidate, "cache_version", 0) >= 2:
            mist = candidate
    if mist is None:
        mist = MistGrid(dirs)
    return (mist, Mist3WDGrid(ROOT / ".data" / "mist3_wd.npz"),
            WDCoolGrid(ROOT / ".data" / "wdcool.npz"))


def plane_basis(r, v):
    z = np.cross(r, v)
    z /= np.linalg.norm(z)
    x = np.asarray(r, dtype=float) - z * np.dot(r, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return x, y, z


def main(apply=False):
    mist, wd, cool = grids()
    live_path = ROOT / ".data" / "worlds" / "白玫瑰.oxygen"
    system = world_mod.load_world(live_path)
    ws = Workspace(system, grid_getter=lambda: mist,
                   wd_grid_getter=lambda: wd, cool_grid_getter=lambda: cool)
    ws.state_at(None, with_trails=False)
    candidate = ws.snapshot_for_save()

    names = ["Αστραία", "Vesper", "Εἴδωλον"]
    states = {n: (candidate.bodies[n].pos0.copy(),
                  candidate.bodies[n].vel0.copy()) for n in candidate.bodies}
    masses = {n: candidate.bodies[n].mass_kg for n in names}
    m_triple = sum(masses.values())
    r_triple = sum((masses[n] * states[n][0] for n in names), np.zeros(3)) / m_triple
    v_triple = sum((masses[n] * states[n][1] for n in names), np.zeros(3)) / m_triple
    mv, mw = masses["Vesper"], masses["Εἴδωλον"]
    r_pair = (mv * states["Vesper"][0] + mw * states["Εἴδωλον"][0]) / (mv + mw)
    v_pair = (mv * states["Vesper"][1] + mw * states["Εἴδωλον"][1]) / (mv + mw)
    r_outer = r_pair - states["Αστραία"][0]
    v_outer = v_pair - states["Αστραία"][1]
    x, y, z = plane_basis(r_outer, v_outer)

    # Múspellsheimr: prograde circular orbit about the THREE-STAR barycenter.
    muspell = candidate.bodies["Múspellsheimr"]
    mu_muspell = G * (m_triple + muspell.mass_kg)
    n_muspell = np.sqrt(mu_muspell / A_MUSPELL**3)
    muspell.pos0 = r_triple + A_MUSPELL * x
    muspell.vel0 = v_triple + n_muspell * A_MUSPELL * y

    # Σκιά: mass from the requested radius and mean density, not a guessed value.
    volume = 4.0 * np.pi * R_SKIA**3 / 3.0
    m_skia = volume * RHO_SKIA
    r_wd, v_wd = states["Εἴδωλον"]
    r_inner = r_wd - states["Vesper"][0]
    v_inner = v_wd - states["Vesper"][1]
    xp, yp, zp = plane_basis(r_inner, v_inner)
    mu_skia = G * (mw + m_skia)
    n_skia = np.sqrt(mu_skia / A_SKIA**3)
    period_skia = 2.0 * np.pi / n_skia
    skia = Body(NAME_SKIA, BODY, mass_kg=m_skia,
                pos0=r_wd + A_SKIA * xp,
                vel0=v_wd + n_skia * A_SKIA * yp,
                radius_m=R_SKIA, albedo=0.20,
                spin_axis=zp, spin_rate_rad_s=n_skia, spin_phase_rad=0.0,
                extra={"cmf": 0.35, "composition_preset": "硅酸盐 + 铁核",
                       "love_k2": 0.30, "tidal_dt_s": 100.0,
                       "inertia_factor": 0.26138193209731553,
                       "placement": "initially tidally locked to Εἴδωλον"})
    candidate.add(skia)

    candidate_ws = Workspace(candidate, grid_getter=lambda: mist,
                             wd_grid_getter=lambda: wd,
                             cool_grid_getter=lambda: cool)
    candidate_ws.state_at(None, with_trails=False)
    # Manual orbital surgery = new conservation baseline.  Uniform translation
    # and boost preserve all relative states; no old wind history is invented.
    candidate_ws.sim.reset_conservation_epoch(recenter=True)
    candidate_ws._trail_cache = None

    candidate_dir = ROOT / ".data" / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = candidate_dir / "白玫瑰-rescue.oxygen"
    world_mod.save_world(candidate_ws.snapshot_for_save(), candidate_path)

    # Independent two-body diagnostics, using the actual candidate save.
    check = world_mod.load_world(candidate_path)
    m_wd = mw
    roche_rigid = R_SKIA * (2.0 * m_wd / m_skia) ** (1.0 / 3.0)
    roche_fluid = 2.44 * R_SKIA * (m_wd / m_skia) ** (1.0 / 3.0)
    hill = 3.0 * AU * (m_wd / (3.0 * (mw + mv))) ** (1.0 / 3.0)
    result = {
        "candidate": str(candidate_path),
        "muspell": {
            "a_AU": A_MUSPELL / AU,
            "period_yr": 2.0 * np.pi * np.sqrt(A_MUSPELL**3 / mu_muspell) / YEAR,
            "periapsis_AU": A_MUSPELL / AU,
            "reference": "Αστραία+Vesper+Εἴδωλον barycenter",
        },
        "skia": {
            "name": NAME_SKIA,
            "mass_kg": m_skia,
            "mass_mearth": m_skia / M_EARTH,
            "radius_km": R_SKIA / 1000.0,
            "mean_density_g_cm3": m_skia / volume / 1000.0,
            "a_AU": A_SKIA / AU,
            "period_h": period_skia / 3600.0,
            "spin_period_h": 2.0 * np.pi / skia.spin_rate_rad_s / 3600.0,
            "rigid_roche_limit_AU": roche_rigid / AU,
            "fluid_roche_limit_AU": roche_fluid / AU,
            "a_over_fluid_roche": A_SKIA / roche_fluid,
            "wd_hill_radius_AU": hill / AU,
        },
        "wind_mass_kg": check.wind.mass_kg,
        "body_count": len(check.bodies),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if apply:
        backup_dir = ROOT / ".data" / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = backup_dir / f"白玫瑰-pre-rescue-{stamp}.oxygen"
        shutil.copy2(live_path, backup)
        shutil.copy2(candidate_path, live_path)
        print(json.dumps({"applied": True, "backup": str(backup),
                          "live": str(live_path)}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"applied": False,
                          "note": "candidate only; live world unchanged"},
                         ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    main(args.apply)
