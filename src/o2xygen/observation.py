"""Planet-fixed observation points and their local east/north/zenith frames.

Only (name, body, latitude, longitude) are persisted.  The observer's world
position and the local basis are derived from the authoritative body state and
its spin phase.  Longitude increases eastward; the zero meridian is the body's
+X material direction projected onto its equatorial plane at zero phase, and
its north pole is the spin axis. The projection is essential for tilted axes.
"""
from __future__ import annotations

import math
import numpy as np


def normalize_point(value: dict) -> dict:
    if not isinstance(value, dict):
        raise ValueError("observation point must be an object")
    name, body = value.get("name"), value.get("body")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("observation point name must not be empty")
    if not isinstance(body, str) or not body:
        raise ValueError("observation point body must not be empty")
    if len(name.strip()) > 120 or len(body) > 120:
        raise ValueError("observation point name/body is too long")
    try:
        lat = float(value["latitude_deg"])
        lon = float(value["longitude_deg"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError("latitude_deg/longitude_deg must be finite numbers") from e
    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise ValueError("latitude_deg/longitude_deg must be finite numbers")
    if not -90.0 <= lat <= 90.0:
        raise ValueError("latitude must be within -90..90 degrees")
    # [-180, 180) is a canonical representation; preserve +0.0 for 0°E.
    lon = (lon + 180.0) % 360.0 - 180.0
    if lon == 0.0:
        lon = 0.0
    return {"name": name.strip(), "body": body,
            "latitude_deg": lat, "longitude_deg": lon}


def validate_points(values, bodies) -> list[dict]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValueError("observations must be a list")
    result, names = [], set()
    for raw in values:
        p = normalize_point(raw)
        if p["name"] in names:
            raise ValueError(f"duplicate observation point name {p['name']!r}")
        b = bodies.get(p["body"])
        if b is None:
            raise ValueError(f"observation point {p['name']!r}: unknown body {p['body']!r}")
        if b.kind != "body":
            raise ValueError("observation points are supported only on non-stellar bodies")
        if not (math.isfinite(b.radius_m) and b.radius_m > 0.0):
            raise ValueError(f"observation host {b.name!r} must have a positive radius")
        names.add(p["name"])
        result.append(p)
    return result


def _rotate_about_axis(v, axis, angle):
    """Rodrigues active rotation, with no axis-angle library dependency."""
    v, axis = np.asarray(v, dtype=float), np.asarray(axis, dtype=float)
    return (v * math.cos(angle) + np.cross(axis, v) * math.sin(angle)
            + axis * np.dot(axis, v) * (1.0 - math.cos(angle)))


def _unit(v, fallback=None):
    norm = float(np.linalg.norm(v))
    if norm > 1e-15 and math.isfinite(norm):
        return np.asarray(v, dtype=float) / norm
    if fallback is None:
        raise ValueError("undefined local frame direction")
    return np.asarray(fallback, dtype=float).copy()


def local_frame(body: dict, spin: dict, point: dict) -> dict:
    """Return surface offset and an orthonormal east/north/up basis.

    body: {"p": [m,m,m], "radius_m": m}
    spin: {"axis": [..], "rate_rad_s": .., "phase_rad": ..}
    The surface is the zero-altitude reference surface, including for giants.
    """
    p = normalize_point(point)
    axis = _unit(spin.get("axis", [0.0, 0.0, 1.0]))
    phase = float(spin.get("phase_rad", 0.0))
    if not math.isfinite(phase):
        raise ValueError("spin phase must be finite")
    lat, lon = math.radians(p["latitude_deg"]), math.radians(p["longitude_deg"])
    # A tilted spin axis is generally NOT orthogonal to body +X/+Y. Using
    # those raw axes produces non-orthogonal longitudes and a wrong zenith.
    # Project +X onto the equatorial plane, then construct a true right-handed
    # equatorial basis. If +X is the pole, +Y is the well-defined fallback.
    reference = np.array([1.0, 0.0, 0.0]) - axis * axis[0]
    if np.linalg.norm(reference) < 1e-12:
        reference = np.array([0.0, 1.0, 0.0]) - axis * axis[1]
    reference = _unit(reference)
    meridian = _rotate_about_axis(reference, axis, phase)
    prime_vertical = np.cross(axis, meridian)
    local = (math.cos(lat) * (math.cos(lon) * meridian
                              + math.sin(lon) * prime_vertical)
             + math.sin(lat) * axis)
    up = _unit(local)
    east = _unit(np.cross(axis, up),
                 _rotate_about_axis(prime_vertical, axis, lon))
    north = _unit(np.cross(up, east))
    # Guard against small loss of orthogonality from arbitrary input axes.
    east = _unit(np.cross(north, up))
    radius = float(body["radius_m"])
    center = np.asarray(body["p"], dtype=float)
    surface = center + radius * up
    return {"center_m": center, "surface_m": surface,
            "up": up, "east": east, "north": north,
            "zenith_world": up.copy(), "east_world": east.copy(),
            "north_world": north.copy(), "phase_rad": phase,
            "spin_axis": axis.copy(), "point": p}


def enu_angles(direction_enu) -> dict:
    v = np.asarray(direction_enu, dtype=float)
    if v.shape != (3,):
        raise ValueError("direction_enu must have three components")
    v = _unit(v)
    return {"azimuth_deg": math.degrees(math.atan2(v[0], v[1])) % 360.0,
            "altitude_deg": math.degrees(math.asin(np.clip(v[2], -1.0, 1.0)))}
