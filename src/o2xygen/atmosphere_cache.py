"""Versioned, rebuildable atmospheric LUTs beside a world's .oxygen save.

The world file never stores these derived values. The SHA-256 input key binds
planet radius, optical inputs, the numerical kernel and quality parameters.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

from .atmosphere import SphericalAtmosphere, make_optical_table, multiple_scattering, OPTICS_VERSION

CACHE_FORMAT = "o2xygen-atmosphere-lut"
CACHE_VERSION = 1
QUALITY = {"sun_count": 13, "view_count": 17, "azimuth_count": 8,
           "samples_per_channel": 96, "orders": 4,
           "mu_count": 129, "height_count": 48, "quadrature": 48}


def optics_key(radius_m: float, atmosphere: dict, quality=None) -> str:
    payload = {"kernel": OPTICS_VERSION, "cache": CACHE_VERSION,
               "radius_m": float(radius_m), "atmosphere": atmosphere,
               "quality": quality or QUALITY}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _directions(mu_s, mu_v, azimuth_count):
    sun = np.array([np.sqrt(max(0.0, 1-mu_s*mu_s)), 0.0, mu_s])
    if np.linalg.norm(sun - mu_s*np.array([0.0, 0.0, 1.0])) > 1e-12:
        sx = sun - mu_s*np.array([0.0, 0.0, 1.0])
        sx /= np.linalg.norm(sx)
    else:
        sx = np.array([1.0, 0.0, 0.0])
    sy = np.cross(np.array([0.0, 0.0, 1.0]), sx)
    result = []
    for az in range(azimuth_count):
        # Uniform relative azimuth; this is the explicitly documented angular
        # smoothing of orders >=2. First order retains its full phase later.
        phi = 2*np.pi*(az+0.5)/azimuth_count
        v = mu_v*np.array([0.0, 0.0, 1.0]) + np.sqrt(max(0.0, 1-mu_v*mu_v)) * (
            np.cos(phi)*sx + np.sin(phi)*sy)
        result.append((v, sun))
    return result


def build_atmosphere_lut(radius_m: float, atmosphere: dict, quality=None) -> dict:
    q = dict(QUALITY if quality is None else quality)
    model = SphericalAtmosphere(radius_m, atmosphere,
                                integration_points=int(q["quadrature"]))
    table = make_optical_table(model, int(q["mu_count"]), int(q["height_count"]))
    pairs = []
    sun_cos = np.linspace(-1.0, 1.0, int(q["sun_count"]))
    # Orders >=2 need only the upward viewing hemisphere. The ground pass has
    # its own finite-sphere intersection and direct/diffuse surface lighting.
    view_cos = np.linspace(0.0, 1.0, int(q["view_count"]))
    for mu_s in sun_cos:
        for mu_v in view_cos:
            pairs.extend(_directions(float(mu_s), float(mu_v), int(q["azimuth_count"])))
    result = multiple_scattering(model, np.array([v for v, s in pairs]),
                                 np.array([s for v, s in pairs]),
                                 samples=int(q["samples_per_channel"]),
                                 orders=int(q["orders"]))
    multi = result["multiple"].reshape(len(sun_cos), len(view_cos),
                                       int(q["azimuth_count"]), 3).mean(axis=2)
    # Low-order spatial smoothing is the finite LUT reconstruction, not an
    # invented radiance source. Sum and positivity remain unchanged in sign.
    padded = np.pad(multi, ((1, 1), (1, 1), (0, 0)), mode="edge")
    multi = (4*padded[1:-1, 1:-1]
             + padded[:-2, 1:-1] + padded[2:, 1:-1]
             + padded[1:-1, :-2] + padded[1:-1, 2:]) / 8.0
    if not np.all(np.isfinite(multi)) or np.min(multi) < -1e-12:
        raise ValueError("atmospheric precomputation produced invalid radiance")
    multi = np.maximum(multi, 0.0)
    return {"format": CACHE_FORMAT, "version": CACHE_VERSION,
            "kernel": OPTICS_VERSION, "fingerprint": optics_key(radius_m, atmosphere, q),
            "radius_m": float(radius_m), "atmosphere": atmosphere, "quality": q,
            "mu": table["mu"].astype(np.float64),
            "height_m": table["height_m"].astype(np.float64),
            "tau": table["tau"].astype(np.float32),
            "sun_cos": sun_cos, "view_cos": view_cos,
            "multiple": multi.astype(np.float32),
            "orders": np.asarray(result["volume_orders"]).mean(axis=2).tolist(),
            "ground_order_totals": np.asarray(result["ground_orders"]).mean(axis=2).tolist()}


def load_or_build_atmosphere(path: str | Path, radius_m: float, atmosphere: dict) -> dict:
    path = Path(path)
    key = optics_key(radius_m, atmosphere)
    file = path / f"{key}.npz"
    if file.exists():
        with np.load(file, allow_pickle=False) as z:
            if str(z["format"]) != CACHE_FORMAT or int(z["version"]) != CACHE_VERSION:
                raise ValueError("unsupported atmospheric cache version")
            out = {k: z[k] for k in ("mu", "height_m", "tau", "sun_cos", "view_cos", "multiple")}
            out.update({"format": CACHE_FORMAT, "version": CACHE_VERSION,
                        "fingerprint": key, "cached": True})
            return out
    model = build_atmosphere_lut(radius_m, atmosphere)
    path.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in model.items()
              if k in ("mu", "height_m", "tau", "sun_cos", "view_cos", "multiple")}
    arrays["format"] = np.asarray(CACHE_FORMAT)
    arrays["version"] = np.asarray(CACHE_VERSION)
    arrays["fingerprint"] = np.asarray(key)
    tmp = path / f".{key}.tmp.npz"
    np.savez(tmp, **arrays)
    tmp.replace(file)
    out = {k: arrays[k] for k in ("mu", "height_m", "tau", "sun_cos", "view_cos", "multiple")}
    out.update({"format": CACHE_FORMAT, "version": CACHE_VERSION,
                "fingerprint": key, "cached": False})
    return out


def spectrum_rgb(teff_k: float) -> list[float]:
    """Normalized visible blackbody approximation; total flux is separate.

    Representative RGB bands at 610/550/465 nm. This is not a MIST spectral
    energy distribution, and the UI labels it as such.
    """
    lam = np.array([610e-9, 550e-9, 465e-9])
    h, c, k = 6.62607015e-34, 299792458.0, 1.380649e-23
    t = np.clip(float(teff_k), 500.0, 100000.0)
    b = 1.0 / (lam**5 * np.expm1(h*c/(lam*k*t)))
    b /= float(np.dot(b, [0.25, 0.5, 0.25]))
    return [float(x) for x in b]
