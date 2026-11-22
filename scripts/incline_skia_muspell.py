"""Move Σκιά to 0.05 AU and give it and Múspellsheimr a 2° inclination.

Default: build .data/candidates/白玫瑰-inclined.oxygen only.
--apply: back up and replace the live world after the candidate is rebuilt.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "scripts"))
from rescue_muspell_add_skia import grids, plane_basis
from o2xygen import world as world_mod
from o2xygen.constants import AU, G, YEAR
from o2xygen.nbody import rv_to_elements
from workspace import Workspace

A_MUSPELL = 300.0 * AU
A_SKIA = 0.05 * AU
INCLINATION = np.radians(2.0)


def main(apply=False):
    mist, wd, cool = grids()
    live_path = ROOT / ".data" / "worlds" / "白玫瑰.oxygen"
    ws = Workspace(world_mod.load_world(live_path), grid_getter=lambda: mist,
                   wd_grid_getter=lambda: wd, cool_grid_getter=lambda: cool)
    ws.state_at(None, with_trails=False)
    dt_before = ws.sim.dt_s
    candidate = ws.snapshot_for_save()
    states = {n: (b.pos0.copy(), b.vel0.copy()) for n, b in candidate.bodies.items()}
    masses = {n: candidate.bodies[n].mass_kg
              for n in ("Αστραία", "Vesper", "Εἴδωλον")}

    # Outer triple plane and barycenter for Múspellsheimr.
    m_triple = sum(masses.values())
    r_triple = sum((masses[n] * states[n][0] for n in masses), np.zeros(3)) / m_triple
    v_triple = sum((masses[n] * states[n][1] for n in masses), np.zeros(3)) / m_triple
    mv, mw = masses["Vesper"], masses["Εἴδωλον"]
    r_pair = (mv * states["Vesper"][0] + mw * states["Εἴδωλον"][0]) / (mv + mw)
    v_pair = (mv * states["Vesper"][1] + mw * states["Εἴδωλον"][1]) / (mv + mw)
    r_outer, v_outer = r_pair - states["Αστραία"][0], v_pair - states["Αστραία"][1]
    x, y, z = plane_basis(r_outer, v_outer)
    muspell = candidate.bodies["Múspellsheimr"]
    n_m = np.sqrt(G * (m_triple + muspell.mass_kg) / A_MUSPELL**3)
    muspell.pos0 = r_triple + A_MUSPELL * x
    muspell.vel0 = v_triple + n_m * A_MUSPELL * (
        np.cos(INCLINATION) * y + np.sin(INCLINATION) * z)

    # Inner-binary plane for Σκιά.  The spin follows the NEW orbital h so the
    # initial tide-locked state remains synchronous and prograde after tilting.
    skia = candidate.bodies["Σκιά"]
    r_inner = states["Εἴδωλον"][0] - states["Vesper"][0]
    v_inner = states["Εἴδωλον"][1] - states["Vesper"][1]
    xp, yp, zp = plane_basis(r_inner, v_inner)
    n_s = np.sqrt(G * (mw + skia.mass_kg) / A_SKIA**3)
    skia.pos0 = states["Εἴδωλον"][0] + A_SKIA * xp
    skia.vel0 = states["Εἴδωλον"][1] + n_s * A_SKIA * (
        np.cos(INCLINATION) * yp + np.sin(INCLINATION) * zp)
    spin_axis = np.cross(A_SKIA * xp, skia.vel0 - states["Εἴδωλον"][1])
    spin_axis /= np.linalg.norm(spin_axis)
    skia.set_spin_vector(spin_axis * n_s, phase_rad=0.0)
    skia.extra["placement"] = "0.05 AU, 2 deg inclined, initially tidally locked to Εἴδωλον"

    candidate_ws = Workspace(candidate, grid_getter=lambda: mist,
                             wd_grid_getter=lambda: wd, cool_grid_getter=lambda: cool)
    candidate_ws.state_at(None, with_trails=False)
    candidate_ws.sim.reset_conservation_epoch(recenter=True)
    candidate_ws._trail_cache = None
    candidate_dir = ROOT / ".data" / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = candidate_dir / "白玫瑰-inclined.oxygen"
    world_mod.save_world(candidate_ws.snapshot_for_save(), candidate_path)

    # Report the step that an actual save/load will use (stars initially carry
    # MIST seeds, then settle).  The in-memory pre-save dt can be different.
    reload_ws = Workspace(world_mod.load_world(candidate_path),
                          grid_getter=lambda: mist,
                          wd_grid_getter=lambda: wd, cool_grid_getter=lambda: cool)
    reload_ws.state_at(None, with_trails=False)
    dt_reloaded = reload_ws.sim.dt_s
    period_s = 2.0 * np.pi / n_s
    result = {
        "candidate": str(candidate_path),
        "inclination_deg": {"Σκιά": 2.0, "Múspellsheimr": 2.0},
        "muspell_a_AU": A_MUSPELL / AU,
        "muspell_period_yr": 2.0 * np.pi / n_m / YEAR,
        "skia_a_AU": A_SKIA / AU,
        "skia_period_h": period_s / 3600.0,
        "skia_spin_period_h": skia.rotation_period_s / 3600.0,
        "skia_period_days": period_s / 86400.0,
        "skia_fluid_roche_ratio": A_SKIA / (
            2.44 * skia.radius_m * (mw / skia.mass_kg) ** (1.0 / 3.0)),
        "dt_days_before": dt_before / 86400.0,
        "dt_days_candidate": dt_reloaded / 86400.0,
        "dt_speed_factor": dt_reloaded / dt_before,
        "body_count": len(candidate_ws.system.bodies),
        "wind_mass_kg": candidate_ws.sim.wind.mass_kg,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if apply:
        backup_dir = ROOT / ".data" / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = backup_dir / f"白玫瑰-pre-incline-{stamp}.oxygen"
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
    main(parser.parse_args().apply)
