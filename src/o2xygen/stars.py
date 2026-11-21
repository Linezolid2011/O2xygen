"""Stellar evolution via MIST v1.2 EEP tracks (MESA r7503, vvcrit0.4).

Reconstructs his 参数估算 modal: "按等效演化点对初始质量与 [Fe/H] 插值，
再按年龄求值" -> interpolate tracks in initial mass and metallicity, then
read off age. We keep it simple and robust: per metallicity, a linear
scattered interpolator over (M_initial, log10 age); metallicities blended
linearly. Age is monotonic along each track, so the mapping is single-valued.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .constants import R_SUN, L_SUN

AGE_MIN, AGE_MAX = 1e6, 1.4e10          # yr, MIST coverage

# Broad, deliberately coarse EEP stages for display. MIST's EEP numbering is
# used as the data source; these are not fractional "progress" estimates.
_PHASE_STEPS = (
    (202.0, "pms", "前主序", "#6b8de8"),
    (353.0, "ms", "主序", "#f0d58a"),
    (454.0, "rgb", "红巨星", "#e89b63"),
    (631.0, "helium", "氦燃烧", "#d9795f"),
    (808.0, "agb", "AGB", "#c77a9a"),
    (1409.0, "post_agb", "后AGB", "#b59ade"),
    (float("inf"), "wd", "白矮星", "#c9d6ff"),
)


def phase_from_eep(eep: float) -> dict:
    for limit, key, label, color in _PHASE_STEPS:
        if eep < limit:
            return {"key": key, "label": label, "color": color}
    return {"key": "wd", "label": "白矮星", "color": "#c9d6ff"}


def _find_column_row(lines: list[str]) -> int:
    """Locate the header row naming columns inside a MIST .eep file."""
    for i, ln in enumerate(lines[:120]):
        s = ln.lstrip("# ").strip()
        if "star_mass" in s and "log_R" in s:
            return i
    raise ValueError("no column header found in track file")


def load_track(path: Path) -> np.ndarray:
    """Parse one .track.eep into a structured float array (age, Mini, logR, logL, logTeff, mdot)."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    hdr = _find_column_row(lines)
    cols = lines[hdr].lstrip("# ").split()
    # data lines: numeric rows below the header
    rows = []
    for ln in lines[hdr + 1:]:
        parts = ln.split()
        if len(parts) == len(cols):
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"no data rows parsed from {path}")
    data = np.array(rows, dtype=float)
    idx = {c: i for i, c in enumerate(cols)}
    age_col = "star_age" if "star_age" in idx else ("age" if "age" in idx else "log10_age")
    out = np.empty(len(data), dtype=[("age_yr", "f8"), ("mini", "f8"),
                                     ("logR", "f8"), ("logL", "f8"),
                                     ("logTeff", "f8"), ("mdot", "f8"), ("eep", "f8")])
    if age_col == "log10_age":
        out["age_yr"] = 10.0 ** data[:, idx[age_col]]
    else:
        out["age_yr"] = data[:, idx[age_col]]
    out["mini"] = data[:, idx["initial_mass"]] if "initial_mass" in idx else data[:, idx["star_mass"]]
    out["logR"] = data[:, idx["log_R"]]
    out["logL"] = data[:, idx["log_L"]]
    out["logTeff"] = data[:, idx["log_Teff"]]
    out["mdot"] = data[:, idx["star_mdot"]] if "star_mdot" in idx else 0.0
    # MIST EEP files are chronological rows; the EEP number is the original
    # row index (the EEPs header lists the milestone rows). Preserve it so
    # evolutionary stages can be read from the track rather than guessed.
    out["eep"] = np.arange(1, len(out) + 1, dtype=float)
    # keep only in-range ages; drop duplicated/zero-age artefacts
    keep = (out["age_yr"] >= AGE_MIN * 0.5) & (out["age_yr"] <= AGE_MAX)
    return out[keep]


@dataclass
class StarProperties:
    mass_msun: float        # current (post-wind) mass
    radius_m: float         # spherical evolutionary radius
    luminosity_w: float
    teff_k: float
    mdot_msun_per_yr: float # wind loss rate, positive = losing mass
                            # (MESA star_mdot is dM/dt, negative on the MS)


class MistGrid:
    """Interpolated MIST tracks for a set of metallicities.

    Interpolation scheme: bilinear in track space. For each track, properties
    are 1-D-interpolated in log10(age); the per-track values are then
    interpolated across initial mass. This is exact at track nodes (a 2-D
    Delaunay scatter interpolation is NOT - its triangles span wildly
    different ages between neighbouring tracks and corrupts R/L badly).
    """
    cache_version = 2       # EEP-carrying parser; invalidates old pickles

    _FIELDS = ("logR", "logL", "logTeff", "logm", "logmdot")

    def __init__(self, track_dirs: dict[float, Path]):
        """track_dirs: {feh_value: directory containing .track.eep files}."""
        self.fehs = sorted(track_dirs)
        # per feh: (minis_sorted, list per track of (logages, {field: values}))
        self._tracks: dict[float, tuple[np.ndarray, list]] = {}
        self._mini_ranges: dict[float, tuple[float, float]] = {}
        self._phase_cache = {}
        for feh, d in track_dirs.items():
            minis, table = [], []
            for f in sorted(Path(d).rglob("*.track.eep")):
                tr = load_track(f)
                if len(tr) < 8:
                    continue
                order = np.argsort(tr["age_yr"])
                logages = np.log10(tr["age_yr"][order])
                entry = {
                    "logR": tr["logR"][order],
                    "logL": tr["logL"][order],
                    "logTeff": tr["logTeff"][order],
                    "logm": np.log10(tr["mini"][order]),
                    # MESA reports dM/dt: negative during wind mass loss.
                    # Loss rate is quoted positive -> interpolate |mdot|.
                    "logmdot": np.log10(np.maximum(np.abs(tr["mdot"][order]), 1e-30)),
                    "eep": tr["eep"][order],
                }
                minis.append(float(tr["mini"][0]))
                table.append((logages, entry))
            if not table:
                raise ValueError(f"no tracks parsed under {d}")
            order = np.argsort(minis)
            self._tracks[feh] = (np.array(minis)[order], [table[i] for i in order])
            self._mini_ranges[feh] = (min(minis), max(minis))

    # -- interpolation ---------------------------------------------------------
    def _at_feh(self, feh: float, mini: float, logage: float) -> np.ndarray:
        minis, table = self._tracks[feh]
        use_m, use_v = [], {k: [] for k in self._FIELDS}
        for mi, (logages, entry) in zip(minis, table):
            if logages[0] <= logage <= logages[-1]:
                use_m.append(mi)
                for k in self._FIELDS:
                    use_v[k].append(np.interp(logage, logages, entry[k]))
        if not use_m:
            # Off-grid policy (ADR-0026): approximate at the nearest covered
            # track endpoint instead of storing fabricated static properties.
            # Pick the closest initial-mass track, then clamp age to its range.
            k = min(range(len(table)),
                    key=lambda j: (abs(minis[j] - mini),
                                   min(abs(logage - table[j][0][0]),
                                       abs(logage - table[j][0][-1]))))
            logages = table[k][0]
            logage = float(np.clip(logage, logages[0], logages[-1]))
            return self._at_feh(feh, mini, logage)
        # At ages where only a subset of tracks survives, clamp mass to the
        # covered range (nearest grid point) rather than extrapolating.
        mini_eval = float(np.clip(mini, use_m[0], use_m[-1]))
        if len(use_m) == 1:
            return np.array([use_v[k][0] for k in self._FIELDS])
        # exact at nodes: np.interp returns node values when x hits a node
        return np.array([np.interp(mini_eval, use_m, use_v[k]) for k in self._FIELDS])

    def clamp_inputs(self, mini_msun: float, feh: float,
                     age_yr: float) -> tuple[float, float, float]:
        """Clamp global mass/FeH/age to the loaded grid. Per-track age
        coverage is handled by _at_feh. Non-finite inputs remain errors."""
        if not np.all(np.isfinite([mini_msun, feh, age_yr])):
            raise ValueError("MIST inputs must be finite")
        feh = float(np.clip(feh, self.fehs[0], self.fehs[-1]))
        lo = max([f for f in self.fehs if f <= feh])
        hi = min([f for f in self.fehs if f >= feh])
        m_lo = min(self._mini_ranges[lo][0], self._mini_ranges[hi][0])
        m_hi = max(self._mini_ranges[lo][1], self._mini_ranges[hi][1])
        mini = float(np.clip(mini_msun, m_lo, m_hi))
        age = float(np.clip(age_yr, AGE_MIN, AGE_MAX))
        return mini, feh, age

    def evaluate(self, mini_msun: float, feh: float, age_yr: float) -> StarProperties:
        mini_msun, feh, age_yr = self.clamp_inputs(mini_msun, feh, age_yr)
        lo = max([f for f in self.fehs if f <= feh])
        hi = min([f for f in self.fehs if f >= feh])
        logage = float(np.log10(age_yr))
        if lo == hi:
            v = self._at_feh(lo, mini_msun, logage)
        else:
            w = (feh - lo) / (hi - lo)
            v = (1 - w) * self._at_feh(lo, mini_msun, logage) + w * self._at_feh(hi, mini_msun, logage)
        logR, logL, logTeff, logm, logmdot = v
        return StarProperties(
            mass_msun=10.0 ** logm,
            radius_m=10.0 ** logR * R_SUN,
            luminosity_w=10.0 ** logL * L_SUN,
            teff_k=10.0 ** logTeff,
            mdot_msun_per_yr=10.0 ** logmdot,
        )

    def final_point(self, mini_msun: float, feh: float) -> dict:
        """Interpolated LAST point of the MIST grid (the IFMR handoff).

        Returns the progenitor's final mass and the TOTAL age at that point.
        For WDs outside MIST III's coverage this is the scientific input to
        WDCool, not an independently editable WD mass (ADR-0028).
        """
        if not np.all(np.isfinite([mini_msun, feh])):
            raise ValueError("MIST inputs must be finite")
        feh = float(np.clip(feh, self.fehs[0], self.fehs[-1]))
        lo = max(f for f in self.fehs if f <= feh)
        hi = min(f for f in self.fehs if f >= feh)

        def at_feh(f):
            minis, table = self._tracks[f]
            last_ages = np.array([t[0][-1] for t in table])
            last_masses = np.array([t[1]["logm"][-1] for t in table])
            m = float(np.clip(mini_msun, minis[0], minis[-1]))
            return float(np.interp(m, minis, last_ages)), \
                float(np.interp(m, minis, last_masses))

        if lo == hi:
            logage, logm = at_feh(lo)
        else:
            w = (feh - lo) / (hi - lo)
            a0, m0 = at_feh(lo)
            a1, m1 = at_feh(hi)
            logage = (1 - w) * a0 + w * a1
            logm = (1 - w) * m0 + w * m1
        return {"age_yr": 10.0 ** logage, "mass_msun": 10.0 ** logm}

    def _phase_at_feh(self, feh: float, mini: float, logage: float) -> float:
        """Interpolate the EEP number itself at a given mass and age."""
        minis, table = self._tracks[feh]
        use = [(mi, np.interp(logage, logages,
                              entry.get("eep", np.arange(1, len(logages) + 1))))
               for mi, (logages, entry) in zip(minis, table)
               if logages[0] <= logage <= logages[-1]]
        if not use:
            k = min(range(len(table)),
                    key=lambda j: (abs(minis[j] - mini),
                                   min(abs(logage - table[j][0][0]),
                                       abs(logage - table[j][0][-1]))))
            ages = table[k][0]
            return self._phase_at_feh(feh, mini,
                                      float(np.clip(logage, ages[0], ages[-1])))
        mini_eval = float(np.clip(mini, min(x[0] for x in use),
                                  max(x[0] for x in use)))
        if len(use) == 1:
            return float(use[0][1])
        return float(np.interp(mini_eval, [x[0] for x in use],
                               [x[1] for x in use]))

    def phase_at(self, mini_msun: float, feh: float, age_yr: float) -> dict:
        mini, feh, age_yr = self.clamp_inputs(mini_msun, feh, age_yr)
        logage = float(np.log10(age_yr))
        lo = max(f for f in self.fehs if f <= feh)
        hi = min(f for f in self.fehs if f >= feh)
        eep = self._phase_at_feh(lo, mini, logage) if lo == hi else \
            (1.0 - (feh - lo) / (hi - lo)) * self._phase_at_feh(lo, mini, logage) \
            + ((feh - lo) / (hi - lo)) * self._phase_at_feh(hi, mini, logage)
        out = phase_from_eep(eep)
        out["eep"] = float(eep)
        return out

    def phase_segments(self, mini_msun: float, feh: float) -> list[dict]:
        """Contiguous broad EEP stages over the MIST age range, cached per
        initial mass/metallicity so dragging the age control is cheap."""
        key = (round(float(mini_msun), 4), round(float(feh), 4))
        if not hasattr(self, "_phase_cache"):
            self._phase_cache = {}       # old pickled grids remain loadable
        if key in self._phase_cache:
            return self._phase_cache[key]
        xs = np.linspace(np.log10(AGE_MIN), np.log10(AGE_MAX), 180)
        stages = [self.phase_at(mini_msun, feh, float(10.0 ** x)) for x in xs]
        out = []
        for i, stage in enumerate(stages):
            if out and out[-1]["key"] == stage["key"]:
                out[-1]["end_logage"] = float(xs[i])
            else:
                out.append({"start_logage": float(xs[i]),
                            "end_logage": float(xs[i]), **stage})
        self._phase_cache[key] = out
        return out

    def solve_age(self, mini_msun: float, feh: float,
                  luminosity_lsun: float, teff_k: float,
                  age_guess_yr: float = 2e9) -> float:
        """Inverse problem: age matching a given (L, Teff) point on the track.
        Used to validate against his displayed star (age itself is not visible
        in the screenshots, so we recover it and check consistency of R)."""
        from .optimize import minimize_scalar_bounded
        target = np.array([np.log10(luminosity_lsun), np.log10(teff_k)])

        def loss(logage):
            try:
                props = self.evaluate(mini_msun, feh, 10.0 ** logage)
            except ValueError:
                # optimizer probe landed outside this mass's track coverage
                # (evaluate is strict now): heavy penalty, keep the search alive
                return 1.0e6
            got = np.array([np.log10(props.luminosity_w / L_SUN),
                            np.log10(props.teff_k)])
            return float(np.sum((got - target) ** 2))

        x = minimize_scalar_bounded(loss, np.log10(AGE_MIN), np.log10(AGE_MAX),
                                    xatol=1e-4)
        return 10.0 ** x
