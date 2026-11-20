"""MIST III white-dwarf cooling tracks, indexed by TOTAL stellar age.

The source Tracks/*.data files contain M_in, M_WD, log_cool_age and
log_tot_age. We therefore evaluate (initial mass, Fe/H, total age) directly,
without a separate cooling-age field and without static WD property copies.

Interpolation is exact at samples: within each total-age layer we interpolate
across initial mass, then blend the two adjacent age layers and Fe/H grids.
Off-grid inputs are clamped to the nearest covered track/endpoint.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .constants import R_SUN, L_SUN
from .stars import StarProperties


class Mist3WDGrid:
    def __init__(self, path: str | Path):
        with np.load(path, allow_pickle=False) as d:
            self.fehs = [float(x) for x in d["fehs"]]
            self._tracks = {}
            for fi, feh in enumerate(self.fehs):
                n = int(d[f"f{fi}_n"])
                table = []
                for j in range(n):
                    a = d[f"f{fi}_t{j:03d}"]
                    table.append((float(d[f"f{fi}_minis"][j]),
                                  float(d[f"f{fi}_mwds"][j]),
                                  a[:, 1].copy(), a[:, 2:5].copy()))
                self._tracks[feh] = table
        self.n_tracks = sum(len(v) for v in self._tracks.values())

    def _at_feh(self, feh: float, mini: float, logage: float) -> np.ndarray:
        table = self._tracks[feh]
        use = [(mi, mwd, ages, fields) for mi, mwd, ages, fields in table
               if ages[0] <= logage <= ages[-1]]
        if not use:
            # nearest initial-mass track, then nearest total-age endpoint
            nearest = min(table, key=lambda t: (abs(t[0] - mini),
                              min(abs(logage - t[2][0]), abs(logage - t[2][-1]))))
            logage = float(np.clip(logage, nearest[2][0], nearest[2][-1]))
            return self._at_feh(feh, mini, logage)
        mini_eval = float(np.clip(mini, min(t[0] for t in use),
                                  max(t[0] for t in use)))
        rows = []
        for mi, mwd, ages, fields in use:
            values = [np.interp(logage, ages, fields[:, k]) for k in range(3)]
            rows.append((mi, mwd, values))
        rows.sort(key=lambda r: r[0])
        out = [np.interp(mini_eval, [r[0] for r in rows],
                         [r[2][k] for r in rows]) for k in range(3)]
        mass = float(np.interp(mini_eval, [r[0] for r in rows],
                               [r[1] for r in rows]))
        return np.array([out[0], out[1], out[2], np.log10(mass)])

    def formation_age_yr(self, mini_msun: float, feh: float) -> float:
        """Interpolated TOTAL age at the first WD-cooling sample."""
        if not np.all(np.isfinite([mini_msun, feh])):
            raise ValueError("MIST III WD inputs must be finite")
        feh = float(np.clip(feh, self.fehs[0], self.fehs[-1]))
        lo = max(f for f in self.fehs if f <= feh)
        hi = min(f for f in self.fehs if f >= feh)

        def at_feh(f):
            table = self._tracks[f]
            minis = [t[0] for t in table]
            starts = [t[2][0] for t in table]
            m = float(np.clip(mini_msun, min(minis), max(minis)))
            return float(np.interp(m, minis, starts))

        logage = at_feh(lo) if lo == hi else \
            (1.0 - (feh - lo) / (hi - lo)) * at_feh(lo) \
            + ((feh - lo) / (hi - lo)) * at_feh(hi)
        return 10.0 ** logage

    def is_wd_age(self, mini_msun: float, feh: float, age_yr: float) -> bool:
        """Whether the full stellar lifecycle has reached its WD track."""
        minis = [t[0] for t in self._tracks[self.fehs[0]]]
        if not (min(minis) <= mini_msun <= max(minis)) or age_yr <= 0.0:
            return False
        return age_yr >= self.formation_age_yr(mini_msun, feh)

    def initial_mass_covered(self, mini_msun: float) -> bool:
        minis = [t[0] for t in self._tracks[self.fehs[0]]]
        return min(minis) <= mini_msun <= max(minis)

    def wd_mass_covered(self, mass_msun: float, feh: float) -> bool:
        """Whether the WD's CURRENT mass lies in MIST III's WD-mass range.
        Only this mass criterion may select the WDCool fallback (ADR-0028)."""
        feh = float(np.clip(feh, self.fehs[0], self.fehs[-1]))
        lo = max(f for f in self.fehs if f <= feh)
        hi = min(f for f in self.fehs if f >= feh)
        ranges = []
        for f in (lo, hi):
            masses = [t[1] for t in self._tracks[f]]
            ranges.append((min(masses), max(masses)))
        w = 0.0 if lo == hi else (feh - lo) / (hi - lo)
        m_lo = (1 - w) * ranges[0][0] + w * ranges[1][0]
        m_hi = (1 - w) * ranges[0][1] + w * ranges[1][1]
        return m_lo <= mass_msun <= m_hi

    def evaluate(self, mini_msun: float, feh: float, age_yr: float) -> StarProperties:
        if not np.all(np.isfinite([mini_msun, feh, age_yr])):
            raise ValueError("MIST III WD inputs must be finite")
        mini = float(np.clip(mini_msun,
                             min(t[0] for t in self._tracks[self.fehs[0]]),
                             max(t[0] for t in self._tracks[self.fehs[-1]])))
        feh = float(np.clip(feh, self.fehs[0], self.fehs[-1]))
        logage = float(np.log10(max(age_yr, 1.0)))
        lo = max(f for f in self.fehs if f <= feh)
        hi = min(f for f in self.fehs if f >= feh)
        if lo == hi:
            v = self._at_feh(lo, mini, logage)
        else:
            w = (feh - lo) / (hi - lo)
            v = (1.0 - w) * self._at_feh(lo, mini, logage) \
                + w * self._at_feh(hi, mini, logage)
        logL, logTeff, logR, logm = v
        return StarProperties(mass_msun=10.0 ** logm,
                              radius_m=10.0 ** logR * R_SUN,
                              luminosity_w=10.0 ** logL * L_SUN,
                              teff_k=10.0 ** logTeff,
                              mdot_msun_per_yr=0.0)
