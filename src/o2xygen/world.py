"""World save format (ADR-0016): [worldname].oxygen — JSON, versioned.

A world is a FLAT N-body system. No hierarchy, no moons: bodies are either
stars or non-stars, each with Cartesian initial conditions. Keplerian
elements are a placement tool, never stored.

Schema v2 (v1 remains readable; its missing wind ledger means a zero baseline):
{
  "format": "o2xygen-world",
  "version": 2,
  "name": "...",
  "origin": "<body name> or null",       # metadata only (ADR-0022)
  "wind": {                             # escaped-wind history (ADR-0030)
    "time_s": 0.0, "mass_kg": ...,
    "momentum_kg_m_s": [x,y,z],
    "angular_momentum_kg_m2_s": [x,y,z],
    "first_moment_kg_m": [x,y,z]
  },
  "astro_frame": {                       # fixed heliocentric ICRS placement
    "position_j2000_pc": [x,y,z],
    "epoch_jd_tt": 2451545.0,             # absolute date of world t=0
    "world_to_icrs_quat_wxyz": [w,x,y,z]
  },
  "bodies": [
    {"name", "kind": "star"|"body",
     "mass_kg" (non-star only),
     "pos0_m": [x,y,z], "vel0_m": [x,y,z],
     "spin_axis": [x,y,z],               # unit vector, inertial frame
     "spin_rate_rad_s": |omega|,         # angular speed at the epoch
     "spin_phase_rad": phase,            # rotation phase at the epoch
     # star only (MIST inputs, nothing derived):
     "mini_msun": ..., "feh": ..., "age0_gyr": ...,
     # non-star only:
     "radius_m": ..., "albedo": ...,
     # both kinds:
     "extra": {...}}                     # free-form metadata (cmf, ring, atm,
                                         # love_k2, tidal_dt_s, ...)
  ]
}
Time origin: t = 0 by definition; displays convert seconds to a uniform
calendar (365.25 d/yr, 12 equal months) as "yyyy-mm-dd From origin".
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path

from .constants import M_SUN
from .system import System, Body, STAR, BODY
from .astroframe import normalize_astro_frame
from .wind import WindLedger

FORMAT = "o2xygen-world"
VERSION = 2


def _json_safe(x):
    """Coerce extra payloads to plain JSON types (numpy scalars/arrays would
    otherwise crash json.dumps at autosave time)."""
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_json_safe(v) for v in x.tolist()]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def system_to_dict(system: System) -> dict:
    bodies = []
    for b in system.bodies.values():
        # stars: the save stores the INITIAL mass (MIST seed), never the
        # wind-evolved current mass
        mass = b.extra.get("mini_msun", b.mass_kg / M_SUN) * M_SUN \
            if b.kind == STAR else b.mass_kg
        d = {
            "name": b.name,
            "kind": b.kind,
            "mass_kg": float(mass),
            "pos0_m": [float(x) for x in b.pos0],
            "vel0_m": [float(x) for x in b.vel0],
            "radius_m": float(b.radius_m),
            # spin state is authoritative; rotation_period_s is only a
            # derived display value and is deliberately NOT saved
            "spin_axis": [float(x) for x in b.spin_axis],
            "spin_rate_rad_s": float(b.spin_rate_rad_s),
            "spin_phase_rad": float(b.spin_phase_rad),
        }
        if b.kind == STAR:
            # ONLY MIST inputs are persisted. Current mass/radius/luminosity/
            # Teff are runtime outputs and must never become a second truth.
            d.pop("mass_kg", None)
            d.pop("radius_m", None)
            d["mini_msun"] = float(b.extra.get("mini_msun", b.mass_kg / M_SUN))
            d["feh"] = float(b.extra.get("feh", 0.0))
            d["age0_gyr"] = float(b.extra.get("age0_gyr", 1.0))
        else:
            d["albedo"] = float(b.albedo)
        if b.extra:
            # free-form NON-MIST metadata only (tides, rings, atm, notes)
            extra = b.extra
            if b.kind == STAR:
                derived = {"mini_msun", "feh", "age0_gyr", "teff_k",
                           "luminosity_w", "evolution_model", "manual_override",
                           "manual_mass_kg"}
                extra = {k: v for k, v in b.extra.items() if k not in derived}
            if extra:
                d["extra"] = _json_safe(extra)
            else:
                d.pop("extra", None)
        bodies.append(d)
    return {"format": FORMAT, "version": VERSION, "name": system.name,
            "origin": system.origin, "bodies": bodies,
            "wind": system.wind.to_dict(),
            "astro_frame": normalize_astro_frame(system.astro_frame)}


def save_world(system: System, path: str | Path) -> Path:
    path = Path(path)
    if path.suffix != ".oxygen":
        path = path.with_suffix(".oxygen")
    path.write_text(json.dumps(system_to_dict(system), ensure_ascii=False,
                               indent=2), encoding="utf-8")
    return path


def dict_to_system(d: dict) -> System:
    if d.get("format") != FORMAT:
        raise ValueError(f"not an o2xygen world (format={d.get('format')!r})")
    if int(d.get("version", 0)) > VERSION:
        raise ValueError(f"world version {d['version']} > supported {VERSION}")
    system = System(d["name"], origin=d.get("origin"))
    system.astro_frame = normalize_astro_frame(d.get("astro_frame"))
    # v1 has no wind field: it starts a new zero-wind conservation baseline.
    system.wind = WindLedger.from_dict(d.get("wind"))
    for bd in d["bodies"]:
        kind = bd["kind"]
        if kind not in (STAR, BODY):
            raise ValueError(f"body {bd['name']!r}: bad kind {kind!r}")
        # free-form metadata first; the authoritative schema fields below
        # always win over any stale duplicates inside extra
        extra = dict(bd.get("extra") or {})
        albedo = 0.0
        if kind == STAR:
            extra["feh"] = float(bd.get("feh", 0.0))
            extra["age0_gyr"] = float(bd.get("age0_gyr", 1.0))
            # the only persisted mass is the MIST INITIAL-mass seed
            mini = float(bd.get("mini_msun",
                                float(bd.get("mass_kg", M_SUN)) / M_SUN))
            extra["mini_msun"] = mini
            mass_kg = mini * M_SUN
            radius_m = 0.0                    # computed lazily from MIST
            luminosity_w = 0.0                # never loaded as a second truth
        else:
            albedo = float(bd.get("albedo", 0.0))
            mass_kg = float(bd["mass_kg"])
            radius_m = float(bd.get("radius_m", 0.0))
            luminosity_w = 0.0
        system.add(Body(
            bd["name"], kind,
            mass_kg=mass_kg,
            pos0=np.array(bd["pos0_m"], dtype=float),
            vel0=np.array(bd["vel0_m"], dtype=float),
            radius_m=radius_m,
            albedo=albedo,
            rotation_period_s=bd.get("rotation_period_s"),  # legacy fallback
            spin_axis=np.array(bd.get("spin_axis", [0.0, 0.0, 1.0]), dtype=float),
            spin_rate_rad_s=bd.get("spin_rate_rad_s"),
            spin_phase_rad=float(bd.get("spin_phase_rad", 0.0)),
            luminosity_w=luminosity_w,
            extra=extra,
        ))
    return system


def load_world(path: str | Path) -> System:
    path = Path(path)
    if path.suffix != ".oxygen":
        path = path.with_suffix(".oxygen")
    return dict_to_system(json.loads(path.read_text(encoding="utf-8")))
