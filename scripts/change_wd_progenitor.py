"""Change Εἴδωλον's MIST III progenitor to 6.5 M_sun and preserve the
CURRENT Vesper–WD osculating orbit (a, e, orientation) despite the new mu.

Positions stay fixed. Relative speed is scaled by sqrt(mu_new/mu_old), which
preserves a and e for the same position and velocity direction. The pair's
linear momentum is kept fixed, so the whole world's total momentum is too.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "server"))

import app
from o2xygen.constants import AU, G, M_SUN, DAY
from o2xygen.nbody import rv_to_elements


def main():
    app._load_grid()
    if app._GRID["grid"] is None or app._GRID["wd_grid"] is None:
        raise RuntimeError("MIST grids failed to load")
    ws = app.WS
    ws.state_at(None)       # settle current stellar masses
    b = ws.system.bodies["Εἴδωλον"]
    with ws.lock:
        ps = ws.sim.sim.particles
        iv, iw = ws.sim._index["Vesper"], ws.sim._index["Εἴδωλον"]
        mv, mw = float(ps[iv].m), float(ps[iw].m)
        rv, rw = np.array(ps[iv].xyz), np.array(ps[iw].xyz)
        vv, vw = np.array(ps[iv].vxyz), np.array(ps[iw].vxyz)
        r = rw - rv
        v = vw - vv
        old_el = rv_to_elements(r, v, G * (mv + mw))
        p_before = sum(p.m * np.array(p.vxyz) for p in ps)

        age_yr = b.extra["age0_gyr"] * 1e9 + ws.sim.t_s / (365.25 * DAY)
        props = app._GRID["wd_grid"].evaluate(6.5, b.extra.get("feh", 0.0), age_yr)
        mw_new = props.mass_msun * M_SUN
        scale = float(np.sqrt((mv + mw_new) / (mv + mw)))
        v_new = v * scale
        pair_p = mv * vv + mw * vw                  # keep this exactly
        cm_new = pair_p / (mv + mw_new)
        vv_new = cm_new - mw_new / (mv + mw_new) * v_new
        vw_new = cm_new + mv / (mv + mw_new) * v_new

        b.extra["mini_msun"] = 6.5
        b.mass_kg = mw_new
        b.radius_m = props.radius_m
        b.luminosity_w = props.luminosity_w
        b.extra["teff_k"] = props.teff_k
        b.extra["luminosity_w"] = props.luminosity_w
        ws.sim.update_masses({b.name: mw_new}, record_wind=False)
        for p, vel in ((ps[iv], vv_new), (ps[iw], vw_new)):
            p.vx, p.vy, p.vz = [float(x) for x in vel]
        new_el = rv_to_elements(r, v_new, G * (mv + mw_new))
        p_after = sum(p.m * np.array(p.vxyz) for p in ps)
        ws.sim.reset_conservation_epoch(recenter=True)

    path = app.world_mod.save_world(ws.snapshot_for_save(),
                                    app._world_path(app.CURRENT_WORLD))
    print(f"WD mass: {mw / M_SUN:.9f} -> {mw_new / M_SUN:.9f} M_sun")
    print(f"relative speed scale: {scale:.12f}")
    print(f"a: {old_el.a_m / AU:.12f} -> {new_el.a_m / AU:.12f} AU")
    print(f"e: {old_el.e:.12f} -> {new_el.e:.12f}")
    print(f"inc: {old_el.inc:.12f} -> {new_el.inc:.12f}")
    print("world momentum relative change:",
          np.linalg.norm(p_after - p_before) / max(1.0, np.linalg.norm(p_before)))
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
