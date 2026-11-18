"""World placement relative to the Solar System (J2000), without a sky renderer.

Canonical frame: heliocentric ICRS. The raw Hipparcos catalog copied under
.data/hip2000 is J1991.25, while its generated JS tables are J2000; this module
only defines the WORLD'S placement and does not load or advance star catalogs.

The world has arbitrary inertial Cartesian axes in metres. `astro_frame` pins
its barycenter to a heliocentric ICRS position and its axes to ICRS through a
fixed quaternion. It also maps simulation t=0 to an absolute Julian date so
future sky rendering can propagate catalogs without losing time across saves.
"""
from __future__ import annotations

import numpy as np

from .constants import DAY

J2000_JD_TT = 2451545.0
JULIAN_YEAR_DAYS = 365.25
PC_M = 3.0856775814913673e16


def default_astro_frame() -> dict:
    return {
        "frame": "ICRS",
        "origin": "Sun",
        "position_j2000_pc": [0.0, 0.0, 0.0],
        "epoch_jd_tt": J2000_JD_TT,
        # active right-handed rotation, world vector -> ICRS, [w,x,y,z]
        "world_to_icrs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
    }


def _vec(value, n: int, name: str) -> list[float]:
    a = np.asarray(value if value is not None else [0.0] * n, dtype=float)
    if a.shape != (n,) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must contain {n} finite numbers")
    return [float(x) for x in a]


def normalize_astro_frame(value: dict | None) -> dict:
    """Validate/copy an astro_frame, filling defaults for omitted fields."""
    out = default_astro_frame()
    if value:
        for k in out:
            if k in value:
                out[k] = value[k]
    out["frame"] = "ICRS"                 # fixed until a second frame exists
    out["origin"] = "Sun"                 # heliocentric, matching Hipparcos
    out["position_j2000_pc"] = _vec(out["position_j2000_pc"], 3,
                                    "position_j2000_pc")
    out["epoch_jd_tt"] = float(out["epoch_jd_tt"])
    if not np.isfinite(out["epoch_jd_tt"]):
        raise ValueError("epoch_jd_tt must be finite")
    q = np.asarray(_vec(out["world_to_icrs_quat_wxyz"], 4,
                        "world_to_icrs_quat_wxyz"))
    norm = float(np.linalg.norm(q))
    if norm < 1e-15:
        raise ValueError("world_to_icrs_quat_wxyz must be nonzero")
    q /= norm
    if q[0] < 0.0:                        # q and -q are the same rotation
        q = -q
    out["world_to_icrs_quat_wxyz"] = [float(x) for x in q]
    return out


def quat_rotate(q_wxyz, v, inverse: bool = False) -> np.ndarray:
    """Rotate a 3-vector by a [w,x,y,z] quaternion (no scipy)."""
    q = np.asarray(q_wxyz, dtype=float)
    q = q / np.linalg.norm(q)
    if inverse:
        q[1:] *= -1.0
    w = q[0]
    u = q[1:]
    v = np.asarray(v, dtype=float).reshape(3)
    return 2.0 * np.dot(u, v) * u + (w * w - np.dot(u, u)) * v \
        + 2.0 * w * np.cross(u, v)


def epoch_years_since_j2000(frame: dict, t_s: float = 0.0) -> float:
    frame = normalize_astro_frame(frame)
    return (frame["epoch_jd_tt"] - J2000_JD_TT) / JULIAN_YEAR_DAYS \
        + float(t_s) / (DAY * JULIAN_YEAR_DAYS)


def julian_year(frame: dict, t_s: float = 0.0) -> float:
    return 2000.0 + epoch_years_since_j2000(frame, t_s)


def position_at_epoch_pc(frame: dict, t_s: float = 0.0) -> np.ndarray:
    """World barycenter's fixed heliocentric ICRS position.

    t_s is accepted for a uniform transform API; space velocity is deliberately
    out of scope, so the translation does not change with time."""
    frame = normalize_astro_frame(frame)
    return np.asarray(frame["position_j2000_pc"], dtype=float)


def world_to_icrs_pc(frame: dict, r_world_m, t_s: float = 0.0) -> np.ndarray:
    """Absolute world position [m] -> heliocentric ICRS [pc]."""
    frame = normalize_astro_frame(frame)
    rotated = quat_rotate(frame["world_to_icrs_quat_wxyz"],
                          np.asarray(r_world_m, dtype=float) / PC_M)
    return position_at_epoch_pc(frame, t_s) + rotated


def icrs_to_world_m(frame: dict, r_icrs_pc, t_s: float = 0.0) -> np.ndarray:
    """Heliocentric ICRS [pc] -> absolute world position [m]."""
    frame = normalize_astro_frame(frame)
    rel = np.asarray(r_icrs_pc, dtype=float) - position_at_epoch_pc(frame, t_s)
    return quat_rotate(frame["world_to_icrs_quat_wxyz"], rel, inverse=True) * PC_M


def advance_epoch(frame: dict, elapsed_s: float) -> dict:
    """Rebase a save to a new t=0 without changing its absolute epoch."""
    out = normalize_astro_frame(frame)
    out["epoch_jd_tt"] += float(elapsed_s) / DAY
    return out


def position_radec(frame: dict) -> dict | None:
    """Derived UI display only; Cartesian position remains the save truth."""
    p = np.asarray(normalize_astro_frame(frame)["position_j2000_pc"])
    d = float(np.linalg.norm(p))
    if d <= 0.0:
        return None
    return {"ra_deg": float(np.arctan2(p[1], p[0]) % (2.0 * np.pi)
                            * 180.0 / np.pi),
            "dec_deg": float(np.arcsin(p[2] / d) * 180.0 / np.pi),
            "distance_pc": d}
