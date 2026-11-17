"""The built-in 白玫瑰 world, in the flat ADR-0016 format.

Bodies are placed with kepler elements as a TOOL: each placement computes
Cartesian (pos0, vel0) around an already-placed reference body and then
FORGETS the elements. The saved world contains only state vectors.
"""
import numpy as np

from o2xygen.constants import G, AU, R_SUN, R_EARTH_MEAN, M_SUN, M_EARTH, L_SUN
from o2xygen.kepler import Elements, elements_to_state
from o2xygen.system import System, Body, STAR, BODY


def _place(sys: System, name: str, kind: str, around: str,
           el: Elements, mass_kg: float, radius_m: float,
           albedo: float = 0.0, rotation_h: float | None = None,
           extra: dict | None = None) -> Body:
    """Place a body on an osculating two-body orbit around `around`
    (which must already be placed). mu = G(M+m) so osculating elements
    round-trip at the epoch."""
    ref = sys.bodies[around]
    mu = G * (ref.mass_kg + mass_kg)
    r_rel, v_rel = elements_to_state(el, mu, 0.0)
    b = Body(name, kind, mass_kg=mass_kg,
             pos0=ref.pos0 + r_rel, vel0=ref.vel0 + v_rel,
             radius_m=radius_m, albedo=albedo,
             rotation_period_s=rotation_h * 3600.0 if rotation_h else None,
             extra=extra or {})
    sys.add(b)
    return b


def build_preset() -> System:
    sys = System("白玫瑰", origin=None)   # ADR-0022: barycentric frame, no root body

    # origin star — his solved star (see tests/test_stars.py)
    sys.add(Body("Αστραία", STAR, mass_kg=1.29 * M_SUN,
                 pos0=np.zeros(3), vel0=np.zeros(3),
                 radius_m=926278.85 * 1000,
                 luminosity_w=2.8055 * L_SUN,
                 extra={"teff_k": 6474.11, "mini_msun": 1.29,
                        "feh": 0.10, "age0_gyr": 1.26}))

    # wide-binary companion
    _place(sys, "Vesper", STAR, "Αστραία",
           Elements(a_m=60 * AU, e=0.30, inc=np.radians(12), raan=0.4,
                    argp=1.2, m0=3.5),
           mass_kg=0.85 * M_SUN, radius_m=0.83 * R_SUN,
           extra={"teff_k": 5000.0, "mini_msun": 0.85,
                  "feh": 0.0, "age0_gyr": 1.26})

    # gas giant with rings
    _place(sys, "Νεφέλη", BODY, "Αστραία",
           Elements(a_m=1.6 * AU, e=0.08, inc=np.radians(1.2), raan=0.1,
                    argp=0.5, m0=0.0),
           mass_kg=20 * M_EARTH, radius_m=4.0 * R_EARTH_MEAN, albedo=0.34,
           rotation_h=9.9,
           extra={"ring": True, "cmf": 0.10, "composition_preset": "巨行星"})

    # its moon — now just a small body on a close orbit around the giant.
    # a = 0.01 AU = 0.27 r_Hill(Νεφέλη) — comfortably stable (Domingos+2006
    # prograde limit ≈ 0.49 r_H ≈ 0.018 AU) and 45x the sum of the two
    # PHYSICAL radii. The "orbit inside the planet" visual was a DISPLAY
    # baseline bug (fixed in scene3d styleRadius, 2026-11-15).
    _place(sys, "Αύρα", BODY, "Νεφέλη",
           Elements(a_m=0.01 * AU, e=0.15, inc=np.radians(24.0), raan=0.9,
                    argp=2.1, m0=1.0),
           mass_kg=7.3e22, radius_m=0.35 * R_EARTH_MEAN, albedo=0.09,
           rotation_h=24.0)

    # the terrestrial showcase, full stat sheet from screenshots
    _place(sys, "Múspellsheimr", BODY, "Αστραία",
           Elements(a_m=7.83 * AU, e=0.1924, inc=np.radians(25.12), raan=2.2,
                    argp=4.0, m0=2.4),
           mass_kg=3.15 * M_EARTH, radius_m=1.35032 * R_EARTH_MEAN,
           albedo=0.20, rotation_h=17.1,
           extra={"ring": True, "cmf": 0.38,
                  "composition_preset": "硅酸盐 + 铁核",
                  "atm": {"gases": [("二氧化碳 CO₂", 91.3), ("氮气 N₂", 8.7)],
                          "tau0": 0.32, "p0_bar": 0.32, "n": 2.0, "S": 0.2,
                          "nabla": 0.0}})
    zero_barycentric_velocity(sys)
    return sys


def zero_barycentric_velocity(sys: System) -> None:
    """Subtract the barycentric velocity from every body.

    Placement assigns the full relative velocity to the NEW body, so a raw
    built world carries nonzero total momentum and drifts linearly forever
    (the pre-fix 白玫瑰 save had drifted ~93 AU off-origin at ~2.8 km/s).
    A uniform velocity shift leaves every relative state - and therefore
    every osculating element - untouched."""
    p = np.zeros(3)
    m_tot = 0.0
    for b in sys.bodies.values():
        p += b.mass_kg * b.vel0
        m_tot += b.mass_kg
    if m_tot <= 0:
        return
    v_cm = p / m_tot
    for b in sys.bodies.values():
        b.vel0 = b.vel0 - v_cm
    return sys
