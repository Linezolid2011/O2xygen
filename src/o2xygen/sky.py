"""Hipparcos starfield observations from arbitrary heliocentric ICRS points.

Adapted from the user's skycalc prototype, reduced to the numerical core. Raw
positions are J1991.25; proper motion is linearly propagated with RV=0. The
Sun is not a Hipparcos entry, so a synthetic Sun is always appended for any
observer not coincident with it (even when fainter than the chosen cutoff).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

MAS2RAD = np.pi / (180.0 * 3600.0 * 1000.0)
DEG2RAD = np.pi / 180.0
SUN_ABS_VMAG = 4.83
SUN_BV = 0.656


class HipparcosSky:
    def __init__(self, path: str | Path):
        with np.load(path, allow_pickle=False) as d:
            self.hip = d["hip"]
            self.ra = d["ra_deg"]
            self.dec = d["dec_deg"]
            self.plx = d["plx_mas"]
            self.pmra = d["pmra_mas_yr"]
            self.pmde = d["pmde_mas_yr"]
            self.vmag = d["vmag"]
            self.bv = d["bv"]
            self.epoch = float(d["epoch_jyear"])
        self.dist0 = 1000.0 / self.plx
        self.abs_mag = self.vmag + 5.0 * np.log10(self.plx) - 10.0
        ra = self.ra * DEG2RAD
        dec = self.dec * DEG2RAD
        self.r0 = np.stack([
            self.dist0 * np.cos(dec) * np.cos(ra),
            self.dist0 * np.cos(dec) * np.sin(ra),
            self.dist0 * np.sin(dec)], axis=1)
        ea = np.stack([-np.sin(ra), np.cos(ra), np.zeros_like(ra)], axis=1)
        ed = np.stack([-np.sin(dec) * np.cos(ra),
                       -np.sin(dec) * np.sin(ra), np.cos(dec)], axis=1)
        self.velocity = (self.dist0 * self.pmra * MAS2RAD)[:, None] * ea \
            + (self.dist0 * self.pmde * MAS2RAD)[:, None] * ed

    def observe(self, observer_pc, year: float, mlim: float = 8.0) -> dict:
        """Return filtered stars: ICRS unit directions, apparent V mag,
        current distance, B-V and HIP. Directions are observer-centric."""
        observer = np.asarray(observer_pc, dtype=float).reshape(3)
        stars = self.r0 + self.velocity * (float(year) - self.epoch)
        rel = stars - observer
        dist = np.linalg.norm(rel, axis=1)
        mag = self.abs_mag + 5.0 * np.log10(np.maximum(dist, 1e-30) / 10.0)
        keep = (mag <= float(mlim)) & np.isfinite(mag) & (dist > 0.0)
        idx = np.flatnonzero(keep)
        dirs = rel[idx] / dist[idx, None]
        out = {"hip": self.hip[idx].copy(), "dir_icrs": dirs,
               "mag": mag[idx].copy(), "dist_pc": dist[idx].copy(),
               "bv": self.bv[idx].copy(), "sun_index": None}
        # The Sun is scientifically necessary but absent from Hipparcos.
        # Keep it even beyond mlim so the world can always locate Sol.
        d_sun = float(np.linalg.norm(observer))
        if d_sun > 1e-12:
            out["hip"] = np.append(out["hip"], 0)
            out["dir_icrs"] = np.vstack([out["dir_icrs"], -observer / d_sun])
            out["mag"] = np.append(out["mag"],
                                   SUN_ABS_VMAG + 5.0 * np.log10(d_sun / 10.0))
            out["dist_pc"] = np.append(out["dist_pc"], d_sun)
            out["bv"] = np.append(out["bv"], SUN_BV)
            out["sun_index"] = len(out["hip"]) - 1
        return out
