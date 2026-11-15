"""Equilibrium temperature models + screenshot calibration.

His footnote: temperature is derived from the summed stellar flux and varies
with orbital position ("随轨道变化"), excluding self-radiation, occultation
and atmospheric attenuation. The T-P/greenhouse refinement lives in the
atmosphere/Clima modules; this is the bare flux->T layer used for the
温度曲线/温度表.
"""
import numpy as np

from .constants import SIGMA

# Temperature regimes T^4 = (1 - A) * S / (k * sigma)
FAST_ROTATOR = 4.0    # planet-averaged:  S/4      (most common convention)
DAYSIDE = 2.0         # dayside average of cos(theta):  S/2
SUBSTELLAR = 1.0      # substellar point: S

_REGIMES = {"fast_rotator": FAST_ROTATOR, "dayside": DAYSIDE, "substellar": SUBSTELLAR}


def equilibrium_temperature(flux_wm2: float, albedo: float = 0.09,
                            regime: str = "dayside",
                            greenhouse: float = 0.0) -> float:
    """T_eq such that T^4 = (1 - A)(1 + greenhouse) * S / (k sigma).

    greenhouse > 0 models a gray-atmosphere correction multiplicatively on T^4.

    Defaults follow ADR-0009: dayside-average regime, albedo 0.09 (his
    screenshot calibration: 861.46 W/m^2 <-> 288.238 K). Pass the parameters
    explicitly when a different convention is wanted.
    """
    k = _REGIMES[regime]
    return ((1.0 - albedo) * (1.0 + greenhouse) * flux_wm2 / (k * SIGMA)) ** 0.25


def infer_flux_to_t4_factor(flux_wm2: float, temperature_k: float) -> float:
    """Effective (1-A)(1+gh) given S and T in fast-rotator units.

    Recon-by-numbers on the screenshot: S = 861.460022 W/m^2 <-> T = 288.238 K
    gives a factor of ~1.82, which a bare fast-rotator with albedo cannot
    produce ((1-A) <= 1). Consistent readings:
      * dayside regime with albedo ~0.09  (1.82/2)
      * fast rotator + gray greenhouse gh ~0.82
      * a moon-specific correction we have not yet identified
    Until more data appears, treat the factor as a free per-body parameter.
    """
    t4 = temperature_k**4
    return t4 * FAST_ROTATOR * SIGMA / flux_wm2


def temperature_series(flux_series, albedo: float = 0.09, regime: str = "dayside",
                       greenhouse: float = 0.0):
    """Vectorized version for the 温度曲线 (temperature vs days-from-epoch).

    Defaults follow ADR-0009 (same as equilibrium_temperature).
    """
    f = np.asarray(flux_series, dtype=float)
    return ((1.0 - albedo) * (1.0 + greenhouse) * f /
            (_REGIMES[regime] * SIGMA)) ** 0.25
