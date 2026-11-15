"""Physical constants and unit helpers.

Constants chosen to match the values fingerprinted in the ORBIT screenshots
(S_EARTH = 1361 W/m^2 is quoted verbatim in his irradiation footnote).
Where his displayed numbers let us infer a convention (e.g. which R_earth),
tests/calibration documents record the finding.
"""
import numpy as np

# --- SI constants -----------------------------------------------------------
G = 6.67430e-11            # m^3 kg^-1 s^-2
SIGMA = 5.670374419e-8     # Stefan-Boltzmann, W m^-2 K^-4
AU = 1.495978707e11        # m
DAY = 86400.0              # s
YEAR = 365.25 * DAY        # s
L_SUN = 3.828e26           # W (nominal IAU 2015)
M_SUN = 1.98847e30         # kg
R_SUN = 6.957e8            # m (nominal IAU 2015)
M_EARTH = 5.9722e24        # kg
# R_EARTH candidates; 6371 km (mean) reproduces his density display best.
R_EARTH_MEAN = 6.371e6     # m
R_EARTH_EQ = 6.378137e6    # m
S_EARTH = 1361.0           # W/m^2, per his footnote: "S⊕ = 1361 W/m²"

# Derived: sanity check that nominal L_SUN / AU^2 reproduces S_EARTH.
# Explicit RuntimeError rather than assert: asserts vanish under python -O.
if abs(L_SUN / (4 * np.pi * AU**2) - S_EARTH) >= 2.0:
    raise RuntimeError(
        f"nominal L_SUN/(4 pi AU^2) = {L_SUN / (4 * np.pi * AU**2):.3f} "
        f"inconsistent with the S_EARTH fingerprint {S_EARTH}")


def gm(mass_kg: float) -> float:
    """Gravitational parameter mu = G*M."""
    return G * mass_kg


def orbital_period_s(a_m: float, mu: float) -> float:
    return 2.0 * np.pi * np.sqrt(a_m**3 / mu)
