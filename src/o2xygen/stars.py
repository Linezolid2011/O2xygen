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
                                     ("logTeff", "f8"), ("mdot", "f8")])
    if age_col == "log10_age":
        out["age_yr"] = 10.0 ** data[:, idx[age_col]]
    else:
        out["age_yr"] = data[:, idx[age_col]]
    out["mini"] = data[:, idx["initial_mass"]] if "initial_mass" in idx else data[:, idx["star_mass"]]
    out["logR"] = data[:, idx["log_R"]]
    out["logL"] = data[:, idx["log_L"]]
    out["logTeff"] = data[:, idx["log_Teff"]]
    out["mdot"] = data[:, idx["star_mdot"]] if "star_mdot" in idx else 0.0
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

    _FIELDS = ("logR", "logL", "logTeff", "logm", "logmdot")

    def __init__(self, track_dirs: dict[float, Path]):
        """track_dirs: {feh_value: directory containing .track.eep files}."""
        self.fehs = sorted(track_dirs)
        # per feh: (minis_sorted, list per track of (logages, {field: values}))
        self._tracks: dict[float, tuple[np.ndarray, list]] = {}
        self._mini_ranges: dict[float, tuple[float, float]] = {}
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
            raise ValueError(f"age {10**logage:.3g} yr outside track coverage at [Fe/H]={feh}")
        if len(use_m) == 1:
            return np.array([use_v[k][0] for k in self._FIELDS])
        # exact at nodes: np.interp returns node values when x hits a node
        return np.array([np.interp(mini, use_m, use_v[k]) for k in self._FIELDS])

    def evaluate(self, mini_msun: float, feh: float, age_yr: float) -> StarProperties:
        if not (self.fehs[0] <= feh <= self.fehs[-1]):
            raise ValueError(f"[Fe/H]={feh} outside loaded grid {self.fehs[0]}..{self.fehs[-1]}")
        logage = float(np.log10(np.clip(age_yr, AGE_MIN, AGE_MAX)))
        lo = max([f for f in self.fehs if f <= feh])
        hi = min([f for f in self.fehs if f >= feh])
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

    def solve_age(self, mini_msun: float, feh: float,
                  luminosity_lsun: float, teff_k: float,
                  age_guess_yr: float = 2e9) -> float:
        """Inverse problem: age matching a given (L, Teff) point on the track.
        Used to validate against his displayed star (age itself is not visible
        in the screenshots, so we recover it and check consistency of R)."""
        from scipy.optimize import minimize_scalar
        target = np.array([np.log10(luminosity_lsun), np.log10(teff_k)])

        def loss(logage):
            props = self.evaluate(mini_msun, feh, 10.0 ** logage)
            got = np.array([np.log10(props.luminosity_w / L_SUN),
                            np.log10(props.teff_k)])
            return float(np.sum((got - target) ** 2))

        res = minimize_scalar(loss, bounds=(np.log10(1e7), np.log10(AGE_MAX)),
                              method="bounded", options={"xatol": 1e-4})
        return 10.0 ** res.x
