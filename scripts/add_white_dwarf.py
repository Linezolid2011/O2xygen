"""Add a white-dwarf companion to Vesper (flavour save edit, 2026-11-17).

Εἴδωλον ("apparition") - a 1.2 M_sun white dwarf on a 3 AU, e=0.05 orbit
around Vesper, placed barycentrically (Vesper recoils, total momentum is
exactly conserved).

Off-grid star: MIST has no WD tracks, so age0_gyr = 15 Gyr (> AGE_MAX) keeps
the engine's MIST coupling permanently in its designed out-of-coverage
fallback - the catalog radius/luminosity/Teff stay static forever.

Hand-computed physics:
  * Nauenberg (1972) mass-radius:
      R = 0.0112 R_sun * sqrt((Mch/M)^(2/3) - (M/Mch)^(2/3)),  Mch = 1.44
      M = 1.2 M_sun  ->  R = 0.00553 R_sun = 3.846e6 m (~0.60 R_earth)
  * blackbody at Teff = 30 000 K (young hot WD):
      L = 4 pi R^2 sigma T^4 = 8.54e24 W = 0.0223 L_sun
  * Vesper Hill radius at 60 AU from a 1.29 M_sun primary: ~25 AU,
    so 3 AU = 0.12 r_H - comfortably stable (prograde, near-circular)
  * orbital period around the ~2.05 M_sun pair: ~3.6 yr
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "server"))

from o2xygen.constants import AU, G, M_SUN, L_SUN, R_SUN, SIGMA
from o2xygen.kepler import Elements, elements_to_state
from o2xygen.system import Body, STAR
from o2xygen import world as world_mod

NAME = "Εἴδωλον"
M_WD = 1.2 * M_SUN
R_WD = 0.0112 * R_SUN * float(np.sqrt((1.44 / 1.2) ** (2 / 3)
                                      - (1.2 / 1.44) ** (2 / 3)))
TEFF = 30000.0
L_WD = 4.0 * np.pi * R_WD ** 2 * SIGMA * TEFF ** 4


def main(path: Path) -> None:
    w = world_mod.load_world(path)
    if NAME in w.bodies:
        raise SystemExit(f"{NAME} already present in {path}")
    vesper = w.bodies["Vesper"]
    m_tot = vesper.mass_kg + M_WD
    mu = G * m_tot
    el = Elements(a_m=3.0 * AU, e=0.05, inc=np.radians(28.0),
                  raan=0.8, argp=2.2, m0=4.0)
    r_rel, v_rel = elements_to_state(el, mu, 0.0)
    w.add(Body(NAME, STAR, mass_kg=M_WD,
               pos0=vesper.pos0 + r_rel * (vesper.mass_kg / m_tot),
               vel0=vesper.vel0 + v_rel * (vesper.mass_kg / m_tot),
               radius_m=R_WD, luminosity_w=L_WD,
               extra={"teff_k": TEFF, "mini_msun": 1.2, "feh": 0.0,
                      "age0_gyr": 15.0, "luminosity_w": L_WD,
                      "love_k2": 1.0e-4, "tidal_dt_s": 100.0,
                      "inertia_factor": 0.08,
                      "note": "off-grid inputs are approximated at the nearest MIST point"}))
    # barycentric recoil: Vesper's motion around the pair barycenter
    vesper.pos0 = vesper.pos0 - r_rel * (M_WD / m_tot)
    vesper.vel0 = vesper.vel0 - v_rel * (M_WD / m_tot)
    out = world_mod.save_world(w, path)
    period_yr = 2 * np.pi * np.sqrt(el.a_m ** 3 / mu) / 86400 / 365.25
    print(f"{NAME}: R = {R_WD:.4e} m ({R_WD / R_SUN:.5f} R_sun), "
          f"L = {L_WD:.3e} W ({L_WD / L_SUN:.4f} L_sun), "
          f"a = 3 AU, e = 0.05, P ~ {period_yr:.2f} yr")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1
         else ROOT / ".data" / "worlds" / "白玫瑰.oxygen")
