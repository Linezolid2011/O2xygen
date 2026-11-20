"""Bédard et al. (2020) WDCool sequences: mass + cooling age + envelope.

Used ONLY for WDs outside MIST III's WD-mass coverage (ADR-0028). Age alone
never selects this fallback. R is stored in cm and L in erg/s in the source;
conversion to SI happens here.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .constants import R_SUN, L_SUN
from .stars import StarProperties


class WDCoolGrid:
    def __init__(self, path: str | Path):
        self._tracks = {}
        with np.load(path, allow_pickle=False) as d:
            for envelope in ("thick", "thin"):
                masses = d[f"masses_{envelope}"]
                table = []
                for m in masses:
                    a = d[f"{envelope}_{int(round(m * 100)):03d}"]
                    table.append((float(m), np.log10(np.maximum(a[:, 3], 1.0)),
                                  a[:, 0].copy(), a[:, 2].copy(), a[:, 4].copy()))
                self._tracks[envelope] = table
        self.n_sequences = sum(len(v) for v in self._tracks.values())

    def _at_mass(self, envelope: str, mass: float, logage: float) -> np.ndarray:
        table = self._tracks[envelope]
        lo = max([t for t in table if t[0] <= mass], key=lambda t: t[0])
        hi = min([t for t in table if t[0] >= mass], key=lambda t: t[0])
        if lo[0] == hi[0]:
            use = [(lo, 0.0)]
        else:
            w = (mass - lo[0]) / (hi[0] - lo[0])
            use = [(lo, 1.0 - w), (hi, w)]
        out = np.zeros(3)
        for (m, ages, teff, radius, lum), weight in use:
            out += weight * np.array([
                np.interp(logage, ages, np.log10(teff)),
                np.interp(logage, ages, np.log10(radius)),
                np.interp(logage, ages, np.log10(lum)),
            ])
        return out

    def evaluate(self, mass_msun: float, cooling_age_yr: float,
                 envelope: str = "thick") -> StarProperties:
        if envelope not in self._tracks:
            raise ValueError(f"unknown WD envelope {envelope!r}")
        if not np.all(np.isfinite([mass_msun, cooling_age_yr])):
            raise ValueError("WDCool inputs must be finite")
        table = self._tracks[envelope]
        mass = float(np.clip(mass_msun, table[0][0], table[-1][0]))
        # cooling age 0 is the first model point; the log-axis floor is 1 yr
        logage = float(np.log10(max(cooling_age_yr, 1.0)))
        logTeff, logR_cm, logL_erg = self._at_mass(envelope, mass, logage)
        return StarProperties(mass_msun=mass,
                              radius_m=10.0 ** logR_cm * 0.01,
                              luminosity_w=10.0 ** logL_erg * 1e-7,
                              teff_k=10.0 ** logTeff,
                              mdot_msun_per_yr=0.0)
