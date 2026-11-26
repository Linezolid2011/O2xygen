"""Backup-first migration of Αύρα to an Earth-mass/radius/atmosphere body.

The world remains a flat collection of absolute Cartesian states.  The local
host--satellite orbit and the pair-barycentre orbit around Αστραία are each
rescaled as a two-body orbit.  A COMMON translation of only the host and
satellite keeps their NEW mass-weighted barycentre at the OLD barycentre;
no global recentring or velocity zeroing is performed.

Dry run (prints one JSON report, writes nothing):
    python scripts/migrate_aura.py path/to/world.oxygen

Write a candidate (the backup is an exact, exclusive copy made first):
    python scripts/migrate_aura.py world.oxygen --output candidate.oxygen \
        --backup backup.oxygen

In-place application is deliberately separate and requires --apply as well as
--backup.  The delegated tool-development workflow uses dry runs/copies only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from o2xygen.atmosphere import default_atmosphere
from o2xygen.constants import G, M_EARTH, R_EARTH_MEAN, YEAR
from o2xygen.nbody import OsculatingElements, rv_to_elements
from o2xygen.system import BODY, STAR, System
from o2xygen.world import load_world, save_world, system_to_dict

AURA = "Αύρα"
HOST = "Νεφέλη"
PRIMARY = "Αστραία"
REPORT_SCHEMA = "o2xygen-aura-migration-v1"


def _vec(value) -> list[float]:
    return [float(x) for x in np.asarray(value, dtype=float).reshape(3)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _angle_delta(new: float, old: float) -> float:
    return float((new - old + np.pi) % (2.0 * np.pi) - np.pi)


def elements_dict(el: OsculatingElements, mu: float | None = None) -> dict[str, float]:
    """JSON-safe osculating elements, including useful derived quantities."""
    out = {
        "a_m": float(el.a_m),
        "e": float(el.e),
        "inc_rad": float(el.inc),
        "raan_rad": float(el.raan),
        "argp_rad": float(el.argp),
        "true_anomaly_rad": float(el.nu),
        "inc_deg": float(np.degrees(el.inc)),
        "periapsis_m": float(el.a_m * (1.0 - el.e)),
        "apoapsis_m": float(el.a_m * (1.0 + el.e)),
    }
    if mu is not None and el.a_m > 0.0:
        out["period_yr"] = float(2.0 * np.pi * np.sqrt(el.a_m**3 / mu) / YEAR)
    return out


def element_differences(new: OsculatingElements,
                        old: OsculatingElements) -> dict[str, float]:
    return {
        "a_relative": float((new.a_m - old.a_m) / old.a_m),
        "e_absolute": float(new.e - old.e),
        "inc_rad": _angle_delta(new.inc, old.inc),
        "raan_rad": _angle_delta(new.raan, old.raan),
        "argp_rad": _angle_delta(new.argp, old.argp),
        "true_anomaly_rad": _angle_delta(new.nu, old.nu),
    }


def pair_barycentre(host, satellite) -> tuple[np.ndarray, np.ndarray]:
    """Barycentre evaluated relative to the host to avoid AU-scale cancellation."""
    fraction = satellite.mass_kg / (host.mass_kg + satellite.mass_kg)
    position = host.pos0 + fraction * (satellite.pos0 - host.pos0)
    velocity = host.vel0 + fraction * (satellite.vel0 - host.vel0)
    return position, velocity


def _position_tolerance(*vectors) -> float:
    """A few representable ULPs at the largest absolute coordinate, at least 1 μm."""
    largest = max((float(np.max(np.abs(vector))) for vector in vectors),
                  default=1.0)
    return max(1e-6, 16.0 * float(np.spacing(max(largest, 1.0))))


def mean_density_kg_m3(mass_kg: float, radius_m: float) -> float:
    return float(mass_kg / (4.0 * np.pi * radius_m**3 / 3.0))


def fluid_roche_limit_m(host_mass_kg: float, host_radius_m: float,
                        satellite_mass_kg: float,
                        satellite_radius_m: float) -> float:
    """Rigid-host / fluid-satellite Roche limit, coefficient 2.44."""
    rho_host = mean_density_kg_m3(host_mass_kg, host_radius_m)
    rho_satellite = mean_density_kg_m3(satellite_mass_kg, satellite_radius_m)
    return float(2.44 * host_radius_m * (rho_host / rho_satellite)**(1.0 / 3.0))


def hill_radii_m(outer: OsculatingElements, host_mass_kg: float,
                 primary_mass_kg: float) -> dict[str, Any]:
    factor = (host_mass_kg / (3.0 * primary_mass_kg))**(1.0 / 3.0)
    return {
        "semimajor_m": float(outer.a_m * factor),
        "periapsis_m": float(outer.a_m * (1.0 - outer.e) * factor),
        "instantaneous_note": "semimajor and outer-periapsis Hill radii; not a stability boundary",
    }


def _state_dict(body) -> dict[str, Any]:
    return {
        "position_m": _vec(body.pos0),
        "velocity_m_s": _vec(body.vel0),
        "mass_kg": float(body.mass_kg),
        "radius_m": float(body.radius_m),
    }


def _validate_names(system: System, satellite: str, host: str,
                    primary: str) -> None:
    if len({satellite, host, primary}) != 3:
        raise ValueError("satellite, host and primary must be distinct")
    for name in (satellite, host, primary):
        if name not in system.bodies:
            raise ValueError(f"required body {name!r} is absent")
    if system.bodies[satellite].kind != BODY or system.bodies[host].kind != BODY:
        raise ValueError("host and satellite must both be non-stellar bodies")
    if system.bodies[primary].kind != STAR:
        raise ValueError("primary must be a star")
    if system.bodies[host].mass_kg <= 0.0 or system.bodies[primary].mass_kg <= 0.0:
        raise ValueError("host and primary masses must be positive")


def migrate_system(system: System, satellite: str = AURA, host: str = HOST,
                   primary: str = PRIMARY) -> tuple[System, dict[str, Any]]:
    """Return a migrated deep copy and a detailed invariant/safety report.

    Internal relative velocity scales by sqrt(mu_inner_new/mu_inner_old).
    The pair-barycentre velocity relative to the primary scales by the analogous
    outer factor.  The common position translation is derived, not guessed:

        delta = Rcm_old - (M*r_host + m_new*r_satellite)/(M+m_new).

    Applying delta to BOTH bodies preserves r_rel and puts their new physical
    barycentre exactly at Rcm_old.  All other bodies remain bit-for-bit fixed.
    """
    _validate_names(system, satellite, host, primary)
    old = copy.deepcopy(system)
    new = copy.deepcopy(system)
    oh, om, op = old.bodies[host], old.bodies[satellite], old.bodies[primary]
    nh, nm = new.bodies[host], new.bodies[satellite]

    m_host = float(oh.mass_kg)
    m_old = float(om.mass_kg)
    m_new = float(M_EARTH)
    if not (m_old > 0.0 and om.radius_m > 0.0 and oh.radius_m > 0.0):
        raise ValueError("old host/satellite masses and radii must be positive")

    r_rel_old = om.pos0 - oh.pos0
    v_rel_old = om.vel0 - oh.vel0
    r_cm_old, v_cm_old = pair_barycentre(oh, om)
    r_outer_old = r_cm_old - op.pos0
    v_outer_old = v_cm_old - op.vel0

    mu_inner_old = G * (m_host + m_old)
    mu_inner_new = G * (m_host + m_new)
    mu_outer_old = G * (op.mass_kg + m_host + m_old)
    mu_outer_new = G * (op.mass_kg + m_host + m_new)
    inner_scale = float(np.sqrt(mu_inner_new / mu_inner_old))
    outer_scale = float(np.sqrt(mu_outer_new / mu_outer_old))

    inner_old_el = rv_to_elements(r_rel_old, v_rel_old, mu_inner_old)
    outer_old_el = rv_to_elements(r_outer_old, v_outer_old, mu_outer_old)

    # Change only the requested physical parameters and atmosphere.  Spin,
    # albedo, rings, tidal metadata and unrelated atmospheric metadata remain.
    nm.mass_kg = m_new
    nm.radius_m = float(R_EARTH_MEAN)
    nm.extra = dict(nm.extra)
    nm.extra["atmosphere"] = default_atmosphere()

    # Preserve the physical NEW pair barycentre at the OLD reference position.
    # The fractional form avoids subtracting two independently rounded AU-scale
    # barycentres.  Both bodies receive exactly the same translation vector.
    old_satellite_fraction = m_old / (m_host + m_old)
    new_satellite_fraction = m_new / (m_host + m_new)
    r_cm_new_unshifted = oh.pos0 + new_satellite_fraction * r_rel_old
    common_translation = (old_satellite_fraction - new_satellite_fraction) * r_rel_old
    nh.pos0 = oh.pos0 + common_translation
    nm.pos0 = om.pos0 + common_translation

    v_rel_new = inner_scale * v_rel_old
    v_cm_new = op.vel0 + outer_scale * v_outer_old
    nh.vel0 = v_cm_new - (m_new / (m_host + m_new)) * v_rel_new
    nm.vel0 = v_cm_new + (m_host / (m_host + m_new)) * v_rel_new

    # Derived checks use the actual new states and actual new mass weighting.
    r_cm_new, v_cm_check = pair_barycentre(nh, nm)
    r_rel_new = nm.pos0 - nh.pos0
    v_rel_check = nm.vel0 - nh.vel0
    r_outer_new = r_cm_new - op.pos0
    v_outer_new = v_cm_check - op.vel0
    inner_new_el = rv_to_elements(r_rel_new, v_rel_check, mu_inner_new)
    outer_new_el = rv_to_elements(r_outer_new, v_outer_new, mu_outer_new)

    old_roche = fluid_roche_limit_m(oh.mass_kg, oh.radius_m,
                                    om.mass_kg, om.radius_m)
    new_roche = fluid_roche_limit_m(nh.mass_kg, nh.radius_m,
                                    nm.mass_kg, nm.radius_m)
    inner_q = float(inner_new_el.a_m * (1.0 - inner_new_el.e))
    physical_radii_sum = float(nh.radius_m + nm.radius_m)
    old_hill = hill_radii_m(outer_old_el, m_host, op.mass_kg)
    new_hill = hill_radii_m(outer_new_el, m_host, op.mass_kg)

    inner_diff = element_differences(inner_new_el, inner_old_el)
    outer_diff = element_differences(outer_new_el, outer_old_el)
    unaffected_ok = True
    for name in old.bodies:
        if name in (host, satellite):
            continue
        before, after = old.bodies[name], new.bodies[name]
        unaffected_ok &= bool(np.array_equal(before.pos0, after.pos0)
                              and np.array_equal(before.vel0, after.vel0)
                              and before.mass_kg == after.mass_kg)

    barycentre_position_tolerance = _position_tolerance(
        oh.pos0, om.pos0, nh.pos0, nm.pos0, r_cm_old, r_cm_new)
    relative_position_tolerance = _position_tolerance(oh.pos0, om.pos0)
    checks = {
        "inner_bound": bool(0.0 <= inner_new_el.e < 1.0 and inner_new_el.a_m > 0.0),
        "outer_bound": bool(0.0 <= outer_new_el.e < 1.0 and outer_new_el.a_m > 0.0),
        "periapsis_above_physical_radii": bool(inner_q > physical_radii_sum),
        "periapsis_above_fluid_roche_limit": bool(inner_q > new_roche),
        "new_pair_barycentre_position_closed": bool(
            np.linalg.norm(r_cm_new - r_cm_old) <= barycentre_position_tolerance),
        "relative_position_closed": bool(
            np.linalg.norm(r_rel_new - r_rel_old) <= relative_position_tolerance),
        "new_pair_barycentre_velocity_closed": bool(
            np.linalg.norm(v_cm_check - v_cm_new) <= 1e-8),
        "unrelated_absolute_states_unchanged": bool(unaffected_ok),
        "inner_elements_preserved_numerically": bool(
            abs(inner_diff["a_relative"]) < 1e-10
            and all(abs(inner_diff[key]) < 1e-10 for key in
                    ("e_absolute", "inc_rad", "raan_rad", "argp_rad",
                     "true_anomaly_rad"))),
        "outer_elements_preserved_numerically": bool(
            abs(outer_diff["a_relative"]) < 1e-10
            and all(abs(outer_diff[key]) < 1e-10 for key in
                    ("e_absolute", "inc_rad", "raan_rad", "argp_rad",
                     "true_anomaly_rad"))),
    }

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "bodies": {"satellite": satellite, "host": host, "primary": primary},
        "method": {
            "coordinates": "flat absolute Cartesian p/v; no parent hierarchy",
            "inner_velocity_scale": inner_scale,
            "outer_velocity_scale": outer_scale,
            "mu_inner_old_m3_s2": float(mu_inner_old),
            "mu_inner_new_m3_s2": float(mu_inner_new),
            "mu_outer_old_m3_s2": float(mu_outer_old),
            "mu_outer_new_m3_s2": float(mu_outer_new),
            "common_translation_m": _vec(common_translation),
            "common_translation_norm_m": float(np.linalg.norm(common_translation)),
            "translation_scope": [host, satellite],
            "global_recentre_performed": False,
            "barycentric_velocity_zeroed": False,
            "wind_ledger_modified": False,
            "stellar_evolution_or_tides_performed": False,
        },
        "physical_parameters": {
            "old": {
                "satellite_mass_kg": m_old,
                "satellite_mass_mearth": m_old / M_EARTH,
                "satellite_radius_m": float(om.radius_m),
                "satellite_radius_rearth_mean": om.radius_m / R_EARTH_MEAN,
                "host_mass_kg": m_host,
                "host_mass_mearth": m_host / M_EARTH,
                "host_radius_m": float(oh.radius_m),
            },
            "new": {
                "satellite_mass_kg": m_new,
                "satellite_mass_mearth": m_new / M_EARTH,
                "satellite_radius_m": float(nm.radius_m),
                "satellite_radius_rearth_mean": nm.radius_m / R_EARTH_MEAN,
                "host_mass_kg": float(nh.mass_kg),
                "host_mass_mearth": nh.mass_kg / M_EARTH,
                "host_radius_m": float(nh.radius_m),
            },
            "atmosphere": "o2xygen.atmosphere.default_atmosphere()",
        },
        "inner_orbit": {
            "old": elements_dict(inner_old_el, mu_inner_old),
            "new": elements_dict(inner_new_el, mu_inner_new),
            "difference": inner_diff,
            "old_r_rel_m": _vec(r_rel_old),
            "new_r_rel_m": _vec(r_rel_new),
            "old_v_rel_m_s": _vec(v_rel_old),
            "new_v_rel_m_s": _vec(v_rel_new),
            "old_speed_m_s": float(np.linalg.norm(v_rel_old)),
            "new_speed_m_s": float(np.linalg.norm(v_rel_new)),
        },
        "outer_orbit": {
            "old": elements_dict(outer_old_el, mu_outer_old),
            "new": elements_dict(outer_new_el, mu_outer_new),
            "difference": outer_diff,
            "old_r_rel_m": _vec(r_outer_old),
            "new_r_rel_m": _vec(r_outer_new),
            "old_v_rel_m_s": _vec(v_outer_old),
            "new_v_rel_m_s": _vec(v_outer_new),
        },
        "pair_barycentre": {
            "old_position_m": _vec(r_cm_old),
            "new_position_before_common_translation_m": _vec(r_cm_new_unshifted),
            "new_position_m": _vec(r_cm_new),
            "position_closure_error_m": float(np.linalg.norm(r_cm_new - r_cm_old)),
            "position_closure_ulp_tolerance_m": barycentre_position_tolerance,
            "relative_position_ulp_tolerance_m": relative_position_tolerance,
            "old_velocity_m_s": _vec(v_cm_old),
            "new_velocity_m_s": _vec(v_cm_new),
            "velocity_closure_error_m_s": float(np.linalg.norm(v_cm_check - v_cm_new)),
        },
        "absolute_states": {
            "old": {host: _state_dict(oh), satellite: _state_dict(om),
                    primary: _state_dict(op)},
            "new": {host: _state_dict(nh), satellite: _state_dict(nm),
                    primary: _state_dict(op)},
        },
        "hill_and_roche": {
            "old": {"host_hill_radius": old_hill,
                    "host_on_satellite_fluid_roche_limit_m": old_roche},
            "new": {"host_hill_radius": new_hill,
                    "host_on_satellite_fluid_roche_limit_m": new_roche},
            "inner_a_over_new_hill_semimajor": float(inner_new_el.a_m /
                                                       new_hill["semimajor_m"]),
            "periapsis_over_new_roche": float(inner_q / new_roche),
            "periapsis_clearance_above_roche_m": float(inner_q - new_roche),
            "periapsis_clearance_above_radii_m": float(inner_q - physical_radii_sum),
        },
        "checks": checks,
        "passed": bool(all(checks.values())),
        "approximations": [
            "Osculating elements are local two-body diagnostics, not stored parent links.",
            "Hill radius uses the primary/host/pair masses and outer osculating elements.",
            "Fluid Roche limit uses the 2.44 uniform-density fluid-satellite expression.",
            "Mass change is a manual migration, not a stellar-wind event; the wind ledger is untouched.",
        ],
    }
    if not report["passed"]:
        failed = [name for name, ok in checks.items() if not ok]
        raise ValueError(f"Αύρα migration safety/invariant checks failed: {failed}")
    return new, report


def _json_dump(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)


def write_migration(source: Path, output: Path | None, backup: Path | None,
                    apply: bool = False) -> dict[str, Any]:
    """Backup first, then atomically write a candidate or an explicit in-place apply."""
    source = source.resolve()
    destination = (output or source).resolve() if apply else output.resolve() if output else None
    if destination is None:
        original = load_world(source)
        _, report = migrate_system(original)
        report["write"] = {"dry_run": True, "source": str(source),
                           "source_sha256": _sha256(source), "files_written": []}
        return report
    if source.suffix != ".oxygen" or destination.suffix != ".oxygen":
        raise ValueError("source and destination must use the .oxygen suffix")
    if backup is None:
        raise ValueError("a non-existing --backup path is mandatory before any write")
    backup = backup.resolve()
    if backup in (source, destination):
        raise ValueError("backup must differ from source and destination")
    if backup.exists():
        raise FileExistsError(f"refusing to overwrite backup: {backup}")
    if apply and destination != source:
        raise ValueError("--apply is only valid for in-place migration")
    if destination == source and not apply:
        raise ValueError("in-place migration requires --apply")
    if destination.exists() and destination != source:
        raise FileExistsError(f"refusing to overwrite candidate: {destination}")

    original = load_world(source)
    migrated, report = migrate_system(original)
    source_hash = _sha256(source)

    # Exclusive creation protects an existing backup.  Verify the exact bytes
    # before creating any candidate/temporary output.
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup_created = False
    try:
        with source.open("rb") as src, backup.open("xb") as dst:
            backup_created = True
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
        if _sha256(backup) != source_hash:
            raise IOError("backup SHA-256 differs from source")
    except Exception:
        # Never remove somebody else's pre-existing/racing backup.
        if backup_created and backup.exists() and _sha256(backup) != source_hash:
            backup.unlink()
        raise

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".aura-", suffix=".oxygen",
                                         dir=destination.parent, delete=False) as tmp:
            temporary_name = tmp.name
        save_world(migrated, temporary_name)
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)

    if destination != source and _sha256(source) != source_hash:
        raise IOError("source changed during candidate migration")
    # A schema round-trip proves that only flat absolute states were saved.
    round_trip = system_to_dict(load_world(destination))
    for body in round_trip["bodies"]:
        forbidden = {"parent", "parent_name", "orbit", "children"}.intersection(body)
        if forbidden:
            raise IOError(f"hierarchical keys unexpectedly saved: {sorted(forbidden)}")
    report["write"] = {
        "dry_run": False,
        "apply": bool(apply),
        "source": str(source),
        "source_sha256": source_hash,
        "backup": str(backup),
        "backup_sha256": _sha256(backup),
        "output": str(destination),
        "output_sha256": _sha256(destination),
        "files_written": [str(backup), str(destination)],
    }
    return report


def main(argv: list[str] | None = None) -> int:
    # Greek body names must not crash under a legacy Windows console code page.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, help="new candidate .oxygen path")
    parser.add_argument("--backup", type=Path, help="mandatory new exact backup for writes")
    parser.add_argument("--apply", action="store_true",
                        help="explicitly permit in-place replacement of source")
    parser.add_argument("--report", type=Path,
                        help="also write the JSON report here (not needed for stdout)")
    args = parser.parse_args(argv)
    try:
        if args.report:
            report_path = args.report.resolve()
            protected = {args.source.resolve()}
            if args.output:
                protected.add(args.output.resolve())
            if args.backup:
                protected.add(args.backup.resolve())
            if report_path in protected:
                raise ValueError("--report must not alias source, output or backup")
            if report_path.exists():
                raise FileExistsError(f"refusing to overwrite report: {report_path}")
        report = write_migration(args.source, args.output, args.backup, args.apply)
        rendered = _json_dump(report)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("x", encoding="utf-8") as stream:
                stream.write(rendered + "\n")
        print(rendered)
        return 0
    except Exception as exc:
        print(json.dumps({"schema": REPORT_SCHEMA, "passed": False,
                          "error": f"{type(exc).__name__}: {exc}"},
                         ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
