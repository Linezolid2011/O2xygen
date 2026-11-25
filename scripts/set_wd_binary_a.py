"""Set the Vesper–Εἴδωλον relative semi-major axis to 0.5 AU while
preserving eccentricity, orientation, true anomaly, pair barycenter and
linear momentum. Also evaluate Eggleton Roche-lobe clearance at periastron.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "server"))

import app
from o2xygen.constants import AU, G, DAY, R_SUN
from o2xygen.nbody import rv_to_elements

TARGET_A_AU = 0.5


def eggleton(ratio: float) -> float:
    q23 = ratio ** (2.0 / 3.0)
    return 0.49 * q23 / (0.6 * q23 + np.log1p(ratio ** (1.0 / 3.0)))


def main():
    app._load_grid()
    ws = app.WS
    ws.state_at(None)
    with ws.lock:
        ps = ws.sim.sim.particles
        iv, iw = ws.sim._index["Vesper"], ws.sim._index["Εἴδωλον"]
        mv, mw = float(ps[iv].m), float(ps[iw].m)
        rv, rw = np.array(ps[iv].xyz), np.array(ps[iw].xyz)
        vv, vw = np.array(ps[iv].vxyz), np.array(ps[iw].vxyz)
        M = mv + mw
        Rcm = (mv * rv + mw * rw) / M
        Vcm = (mv * vv + mw * vw) / M
        r, v = rw - rv, vw - vv
        old = rv_to_elements(r, v, G * M)
        scale = TARGET_A_AU * AU / old.a_m
        r_new = r * scale
        v_new = v / np.sqrt(scale)
        rv_new = Rcm - mw / M * r_new
        rw_new = Rcm + mv / M * r_new
        vv_new = Vcm - mw / M * v_new
        vw_new = Vcm + mv / M * v_new
        for p, pos, vel in ((ps[iv], rv_new, vv_new), (ps[iw], rw_new, vw_new)):
            p.x, p.y, p.z = [float(x) for x in pos]
            p.vx, p.vy, p.vz = [float(x) for x in vel]
        new = rv_to_elements(r_new, v_new, G * M)
        # synchronize the catalog save epoch with the edited live state
        for body, p in ((ws.system.bodies["Vesper"], ps[iv]),
                        (ws.system.bodies["Εἴδωλον"], ps[iw])):
            body.pos0 = np.array(p.xyz)
            body.vel0 = np.array(p.vxyz)
        ws.sim._cp_times = [ws.sim.t_s]
        ws.sim._cp_snaps = [ws.sim._snapshot()]

    path = app.world_mod.save_world(ws.snapshot_for_save(),
                                    app._world_path(app.CURRENT_WORLD))
    peri = TARGET_A_AU * (1.0 - new.e)
    rl_v = peri * eggleton(mv / mw)
    rl_w = peri * eggleton(mw / mv)
    print(f"a: {old.a_m / AU:.12f} -> {new.a_m / AU:.12f} AU")
    print(f"e: {old.e:.12f} -> {new.e:.12f}")
    print(f"periastron: {peri:.6f} AU")
    print(f"Vesper Roche lobe ~{rl_v:.6f} AU; radius "
          f"{ws.system.bodies['Vesper'].radius_m / AU:.8f} AU")
    print(f"WD Roche lobe ~{rl_w:.6f} AU; radius "
          f"{ws.system.bodies['Εἴδωλον'].radius_m / AU:.8f} AU")
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
