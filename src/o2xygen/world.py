"""World save format (ADR-0016): [worldname].oxygen — JSON, versioned.

A world is a FLAT N-body system. No hierarchy, no moons: bodies are either
stars or non-stars, each with Cartesian initial conditions. Keplerian
elements are a placement tool, never stored.

Schema v1:
{
  "format": "o2xygen-world",
  "version": 1,
  "name": "...",
  "origin": "<body name>",              # display/frame reference body
  "bodies": [
    {"name", "kind": "star"|"body",
     "mass_kg",                          # star: INITIAL mass (MIST seed)
     "pos0_m": [x,y,z], "vel0_m": [x,y,z],
     # star only:
     "radius_m": <initial radius>, "feh": ..., "age0_gyr": ...,
     # non-star only:
     "radius_m": ..., "albedo": ...,
     "rotation_period_s": optional,
     # both kinds:
     "extra": {...}}                     # free-form metadata (cmf, ring, atm,
                                         # composition_preset, teff_k, ...)
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

FORMAT = "o2xygen-world"
VERSION = 1


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
        }
        if b.rotation_period_s is not None:
            d["rotation_period_s"] = float(b.rotation_period_s)
        if b.kind == STAR:
            d["feh"] = float(b.extra.get("feh", 0.0))
            d["age0_gyr"] = float(b.extra.get("age0_gyr", 1.0))
        else:
            d["albedo"] = float(b.albedo)
        if b.extra:
            # free-form metadata (cmf, ring, atmosphere, placement stamps);
            # losing these on save once silently wiped a world's pipeline
            # inputs and ring flags (2026-11-16 audit)
            d["extra"] = _json_safe(b.extra)
        bodies.append(d)
    return {"format": FORMAT, "version": VERSION, "name": system.name,
            "origin": system.origin, "bodies": bodies}


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
            # the stored mass IS the initial mass: pin the MIST seed so later
            # wind evolution can never corrupt it across save/load cycles
            extra["mini_msun"] = float(bd["mass_kg"]) / M_SUN
        else:
            albedo = float(bd.get("albedo", 0.0))
        system.add(Body(
            bd["name"], kind,
            mass_kg=float(bd["mass_kg"]),
            pos0=np.array(bd["pos0_m"], dtype=float),
            vel0=np.array(bd["vel0_m"], dtype=float),
            radius_m=float(bd.get("radius_m", 0.0)),
            albedo=albedo,
            rotation_period_s=bd.get("rotation_period_s"),
            # off-grid stars (e.g. white dwarfs: no MIST tracks) carry a
            # hand-specified luminosity in extra - honour it on load or a
            # reload would freeze them dark (MIST stars are re-settled at
            # boot anyway, so this only ever matters off-grid)
            luminosity_w=float(extra.get("luminosity_w", 0.0)) if kind == STAR else 0.0,
            extra=extra,
        ))
    return system


def load_world(path: str | Path) -> System:
    path = Path(path)
    if path.suffix != ".oxygen":
        path = path.with_suffix(".oxygen")
    return dict_to_system(json.loads(path.read_text(encoding="utf-8")))
