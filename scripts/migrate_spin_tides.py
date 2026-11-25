"""Migrate the live 白玫瑰 save to spin + CTL tides (2026-11-17).

The save keeps its existing rotation periods (converted to angular speeds),
adds unit spin axes and phase, and records the tidal constants explicitly so
future runs are reproducible. Ringed bodies keep their former 0.35 rad ring
tilt as an equatorial spin-axis tilt.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "server"))

from o2xygen import world as world_mod
from o2xygen.system import STAR

INERTIA_N1 = 2.0 / 3.0 - 4.0 / np.pi**2


def main(path: Path) -> None:
    w = world_mod.load_world(path)
    for b in w.bodies.values():
        axis = b.spin_axis.copy()
        if b.extra.get("ring"):
            axis = np.array([0.0, np.sin(0.35), np.cos(0.35)])
        b.set_spin_vector(axis * b.spin_rate_rad_s, b.spin_phase_rad)
        b.extra.setdefault("love_k2", 0.02 if b.kind == STAR else 0.30)
        b.extra.setdefault("tidal_dt_s", 100.0)
        b.extra.setdefault("inertia_factor", INERTIA_N1)
        if b.name == "Εἴδωλον":
            # centrally condensed degenerate remnant, not an n=1 main-sequence
            # star: weak tidal response and a smaller moment of inertia
            b.extra["love_k2"] = 1.0e-4
            b.extra["inertia_factor"] = 0.08
    out = world_mod.save_world(w, path)
    print(f"spin + CTL tides saved -> {out}")
    for b in w.bodies.values():
        print(f"  {b.name}: axis={np.round(b.spin_axis, 4).tolist()}, "
              f"omega={b.spin_rate_rad_s:.6e} rad/s, "
              f"k2={b.extra['love_k2']}, lag={b.extra['tidal_dt_s']} s")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1
         else ROOT / "data" / "worlds" / "白玫瑰" / "白玫瑰.oxygen")
