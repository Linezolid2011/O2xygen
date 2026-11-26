"""Homothetically move the outer groups without changing their internal states.

Groups (flat absolute bodies, never a saved hierarchy):
  Vesper + Εἴδωλον + Σκιά       (default scale 4; 2 was also compared)
  Múspellsheimr + Gino + Geo    (default scale 5; 4 leaves a small q/Q overlap)

For each group CM relative to Αστραία, r -> scale*r and
v -> v/sqrt(scale).  Thus the two-body a -> scale*a while e, orientation and
phase are preserved.  Every member receives one common translation/boost, so
all intra-group relative p/v remain unchanged.  A final Galilean boost zeroes
total barycentric momentum (no position recentre) and clears the wind ledger as
a new manual-edit conservation baseline.

Use --migrate-aura to perform the Earth migration in memory BEFORE rebalancing.
Dry-run writes nothing.  Candidate/in-place writes require a new exact backup;
in-place replacement additionally requires --apply.
"""
from __future__ import annotations

import argparse
import copy
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
sys.path.insert(0, str(ROOT / "scripts"))

from migrate_aura import (AURA, HOST, PRIMARY, _angle_delta, _sha256,
                          element_differences, elements_dict, migrate_system)
from o2xygen.constants import G, M_EARTH, R_EARTH_MEAN
from o2xygen.nbody import rv_to_elements
from o2xygen.system import System
from o2xygen.world import load_world, save_world, system_to_dict

VESPER_GROUP = ("Vesper", "Εἴδωλον", "Σκιά")
MUSPELL_GROUP = ("Múspellsheimr", "Gino", "Geo")
REPORT_SCHEMA = "o2xygen-outer-rebalance-v1"


def _vec(value) -> list[float]:
    return [float(x) for x in np.asarray(value, dtype=float).reshape(3)]


def _group_cm(system: System, names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, float]:
    bodies = [system.bodies[name] for name in names]
    total = float(sum(body.mass_kg for body in bodies))
    if total <= 0.0:
        raise ValueError(f"group {names!r} has non-positive mass")
    reference = bodies[0]
    position = reference.pos0 + sum(
        (body.mass_kg * (body.pos0 - reference.pos0) for body in bodies[1:]),
        start=np.zeros(3)) / total
    velocity = reference.vel0 + sum(
        (body.mass_kg * (body.vel0 - reference.vel0) for body in bodies[1:]),
        start=np.zeros(3)) / total
    return position, velocity, total


def _system_velocity(system: System) -> float:
    bodies = list(system.bodies.values())
    total = float(sum(body.mass_kg for body in bodies))
    reference = bodies[0]
    velocity = reference.vel0 + sum(
        (body.mass_kg * (body.vel0 - reference.vel0) for body in bodies[1:]),
        start=np.zeros(3)) / total
    return float(np.linalg.norm(velocity))


def _group_report(old: System, new: System, names: tuple[str, ...],
                  scale: float, actual_velocity_scale: float,
                  rescue_speed_fraction: float) -> dict[str, Any]:
    old_cm, old_v, mass = _group_cm(old, names)
    new_cm, new_v, _ = _group_cm(new, names)
    primary = old.bodies[PRIMARY]
    r_old = old_cm - primary.pos0
    v_old = old_v - primary.vel0
    mu = G * (primary.mass_kg + mass)
    old_el = rv_to_elements(r_old, v_old, mu)
    # Before the optional Galilean baseline, primary is unchanged.  Relative
    # states after a uniform boost remain exactly the same.
    new_primary = new.bodies[PRIMARY]
    r_new = new_cm - new_primary.pos0
    v_new = new_v - new_primary.vel0
    new_el = rv_to_elements(r_new, v_new, mu)
    translation = (scale - 1.0) * r_old
    boost = (actual_velocity_scale - 1.0) * v_old
    max_r_error = 0.0
    max_v_error = 0.0
    for name in names:
        for other in names:
            if name >= other:
                continue
            dr_old = old.bodies[name].pos0 - old.bodies[other].pos0
            dr_new = new.bodies[name].pos0 - new.bodies[other].pos0
            dv_old = old.bodies[name].vel0 - old.bodies[other].vel0
            dv_new = new.bodies[name].vel0 - new.bodies[other].vel0
            max_r_error = max(max_r_error, float(np.linalg.norm(dr_new - dr_old)))
            max_v_error = max(max_v_error, float(np.linalg.norm(dv_new - dv_old)))
    largest_position = max(float(np.max(np.abs(body.pos0))) for body in
                           list(old.bodies.values()) + list(new.bodies.values()))
    position_tolerance = max(1e-6, 16.0 * float(np.spacing(largest_position)))
    return {
        "members": list(names),
        "mass_kg": mass,
        "scale": scale,
        "homothetic_velocity_scale": float(1.0 / np.sqrt(scale)),
        "actual_velocity_scale": float(actual_velocity_scale),
        "bound_rescue_applied": bool(actual_velocity_scale < 1.0 / np.sqrt(scale) - 1e-15),
        "rescue_target_speed_over_escape": rescue_speed_fraction,
        "new_speed_over_escape": float(np.linalg.norm(v_new) /
                                        np.sqrt(2.0 * mu / np.linalg.norm(r_new))),
        "outer_two_body_shape_preserved": bool(
            old_el.a_m > 0.0 and abs(new_el.a_m / old_el.a_m - scale) < 1e-10
            and abs(new_el.e - old_el.e) < 1e-10),
        "common_translation_m": _vec(translation),
        "common_boost_m_s": _vec(boost),
        "old_cm_m": _vec(old_cm), "new_cm_m": _vec(new_cm),
        "old_cm_velocity_m_s": _vec(old_v), "new_cm_velocity_m_s": _vec(new_v),
        "old_outer": elements_dict(old_el, mu),
        "new_outer": elements_dict(new_el, mu),
        "expected_new_a_m": float(scale * old_el.a_m),
        "difference": element_differences(new_el, old_el),
        "a_over_old_a": float(new_el.a_m / old_el.a_m),
        "max_internal_relative_position_error_m": max_r_error,
        "max_internal_relative_velocity_error_m_s": max_v_error,
        "internal_position_ulp_tolerance_m": position_tolerance,
        "internal_velocity_tolerance_m_s": 1e-7,
    }


def rebalance_system(system: System, vesper_scale: float = 4.0,
                     muspellsheimr_scale: float = 5.0,
                     establish_baseline: bool = True,
                     rescue_speed_fraction: float = 0.75) -> tuple[System, dict[str, Any]]:
    """Deep-copy and homothetically move both outer groups around the primary."""
    if vesper_scale < 1.0 or muspellsheimr_scale < 1.0:
        raise ValueError("outer-group scales must be >= 1 (outward or unchanged)")
    if not 0.0 < rescue_speed_fraction < 1.0:
        raise ValueError("rescue_speed_fraction must be between zero and one")
    old = copy.deepcopy(system)
    new = copy.deepcopy(system)
    for names in (VESPER_GROUP, MUSPELL_GROUP):
        for name in names:
            if name not in old.bodies:
                raise ValueError(f"required group member {name!r} is absent")
    primary = old.bodies[PRIMARY]

    # Use original states for both groups; they are disjoint and the primary is
    # fixed until the final uniform baseline boost.
    actual_velocity_scales = {}
    for names, scale in ((VESPER_GROUP, vesper_scale),
                         (MUSPELL_GROUP, muspellsheimr_scale)):
        cm, velocity, group_mass = _group_cm(old, names)
        r = cm - primary.pos0
        v = velocity - primary.vel0
        mu = G * (primary.mass_kg + group_mass)
        old_el = rv_to_elements(r, v, mu)
        velocity_scale = float(1.0 / np.sqrt(scale))
        if old_el.a_m < 0.0 or old_el.e >= 1.0:
            # Homothety cannot change the sign of specific orbital energy.
            # Make the initially unbound group bound at its NEW radius, while
            # keeping the CM velocity direction and every internal relative v.
            escape_at_new_radius = float(np.sqrt(2.0 * mu / (scale * np.linalg.norm(r))))
            rescue_scale = rescue_speed_fraction * escape_at_new_radius / np.linalg.norm(v)
            velocity_scale = min(velocity_scale, float(rescue_scale))
        actual_velocity_scales[names] = velocity_scale
        translation = (scale - 1.0) * r
        boost = (velocity_scale - 1.0) * v
        for name in names:
            new.bodies[name].pos0 = old.bodies[name].pos0 + translation
            new.bodies[name].vel0 = old.bodies[name].vel0 + boost

    baseline_boost = np.zeros(3)
    old_system_v = _system_velocity(new)
    if establish_baseline:
        bodies = list(new.bodies.values())
        total = float(sum(body.mass_kg for body in bodies))
        reference = bodies[0]
        v_cm = reference.vel0 + sum(
            (body.mass_kg * (body.vel0 - reference.vel0) for body in bodies[1:]),
            start=np.zeros(3)) / total
        baseline_boost = -v_cm
        for body in bodies:
            body.vel0 = body.vel0 + baseline_boost
        new.wind.clear(0.0)

    groups = {
        "vesper": _group_report(old, new, VESPER_GROUP, vesper_scale,
                                actual_velocity_scales[VESPER_GROUP], rescue_speed_fraction),
        "muspellsheimr": _group_report(old, new, MUSPELL_GROUP, muspellsheimr_scale,
                                       actual_velocity_scales[MUSPELL_GROUP], rescue_speed_fraction),
    }
    pair_cm, pair_v, pair_mass = _group_cm(new, (HOST, AURA))
    new_primary = new.bodies[PRIMARY]
    neph_el = rv_to_elements(pair_cm - new_primary.pos0, pair_v - new_primary.vel0,
                             G * (new_primary.mass_kg + pair_mass))
    vesper_el = rv_to_elements(
        np.array(groups["vesper"]["new_cm_m"]) - new_primary.pos0,
        np.array(groups["vesper"]["new_cm_velocity_m_s"]) - new_primary.vel0,
        G * (new_primary.mass_kg + groups["vesper"]["mass_kg"]))
    muspell_el = rv_to_elements(
        np.array(groups["muspellsheimr"]["new_cm_m"]) - new_primary.pos0,
        np.array(groups["muspellsheimr"]["new_cm_velocity_m_s"]) - new_primary.vel0,
        G * (new_primary.mass_kg + groups["muspellsheimr"]["mass_kg"]))
    neph_apo = float(neph_el.a_m * (1.0 + neph_el.e))
    vesper_q = float(vesper_el.a_m * (1.0 - vesper_el.e))
    vesper_apo = float(vesper_el.a_m * (1.0 + vesper_el.e))
    muspell_q = float(muspell_el.a_m * (1.0 - muspell_el.e))
    muspell_apo = float(muspell_el.a_m * (1.0 + muspell_el.e))
    internal_ok = all(
        group["max_internal_relative_position_error_m"] <= group["internal_position_ulp_tolerance_m"]
        and group["max_internal_relative_velocity_error_m_s"] <= group["internal_velocity_tolerance_m_s"]
        for group in groups.values())
    checks = {
        "all_group_outer_orbits_bound": all(
            group["new_outer"]["a_m"] > 0.0 and group["new_outer"]["e"] < 1.0
            for group in groups.values()),
        "group_internal_relative_states_preserved": internal_ok,
        "bound_groups_preserve_two_body_shape": all(
            group["outer_two_body_shape_preserved"] for group in groups.values()
            if not group["bound_rescue_applied"]),
        "initially_unbound_groups_rescued_to_bound": all(
            group["new_outer"]["a_m"] > 0.0 and group["new_outer"]["e"] < 1.0
            for group in groups.values() if group["bound_rescue_applied"]),
        "vesper_periapsis_beyond_neph_apoapsis": vesper_q > neph_apo,
        "muspell_periapsis_beyond_vesper_apoapsis": muspell_q > vesper_apo,
        "muspell_periapsis_beyond_neph_apoapsis": muspell_q > neph_apo,
        "new_barycentric_momentum_zeroed": (not establish_baseline
                                             or _system_velocity(new) < 1e-7),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "method": {
            "group_transform": "r_CM' = scale*r_CM; bound groups use v_CM' = v_CM/sqrt(scale); initially hyperbolic groups use explicit <=0.75*new local escape speed bound-rescue",
            "internal_relative_p_v_changed": False,
            "global_position_recentre": False,
            "galilean_baseline_boost_m_s": _vec(baseline_boost),
            "barycentric_speed_before_baseline_m_s": old_system_v,
            "barycentric_speed_after_baseline_m_s": _system_velocity(new),
            "wind_ledger_cleared_for_new_manual_baseline": bool(establish_baseline),
            "parent_hierarchy_saved": False,
        },
        "groups": groups,
        "layout_clearances": {
            "neph_pair_apoapsis_m": neph_apo,
            "vesper_group_periapsis_m": vesper_q,
            "vesper_group_apoapsis_m": vesper_apo,
            "muspell_group_periapsis_m": muspell_q,
            "muspell_group_apoapsis_m": muspell_apo,
            "muspell_minus_vesper_apoapsis_m": muspell_q - vesper_apo,
            "muspell_minus_neph_apoapsis_m": muspell_q - neph_apo,
        },
        "checks": checks,
        "passed": bool(all(checks.values())),
        "scale_reason": "Vesper 4x raises q to about 6.1 AU (2x leaves a marginal 3.05 AU). Múspellsheimr 4x rescue leaves q=263.36 AU slightly inside Vesper Q=265.38 AU, so 5x is selected: q=329.20 AU with about 63.82 AU clearance. Its initially hyperbolic CM receives explicit 0.75-local-escape-speed bound-rescue. Internal relative states are always preserved.",
        "approximations": [
            "Separated osculating periapsis/apoapsis ranges reduce close encounters but do not prove long-term stability.",
            "Homothety cannot bind an initially hyperbolic group; bound-rescue deliberately changes that group's outer a/e while preserving CM velocity direction and all internal relative states.",
            "The final zero-momentum state is a uniform Galilean boost, not a position recentre; every relative orbit is unchanged.",
        ],
    }
    if not report["passed"]:
        failed = [name for name, ok in checks.items() if not ok]
        raise ValueError(f"outer layout checks failed: {failed}")
    return new, report


def apply_current_masses(system: System, state: dict[str, Any]) -> dict[str, float]:
    """Use /api/state current masses for orbital design only.

    Star masses in the save remain MIST initial seeds; system_to_dict ignores
    this runtime override, so no derived stellar output leaks into the file.
    """
    bodies = state.get("bodies")
    if not isinstance(bodies, dict):
        raise ValueError("state JSON is missing bodies")
    applied: dict[str, float] = {}
    for name, body in system.bodies.items():
        record = bodies.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"state JSON is missing body {name!r}")
        mass = float(record.get("m", 0.0))
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError(f"state mass for {name!r} must be positive and finite")
        body.mass_kg = mass
        applied[name] = mass
    return applied


def prepare_system(source: System, migrate_aura: bool = True,
                   vesper_scale: float = 4.0, muspellsheimr_scale: float = 5.0,
                   establish_baseline: bool = True,
                   current_state: dict[str, Any] | None = None
                   ) -> tuple[System, dict[str, Any]]:
    """Apply current masses, Aura, then outer transforms and new baseline."""
    working = copy.deepcopy(source)
    applied = apply_current_masses(working, current_state) if current_state else None
    if migrate_aura:
        migrated, aura_report = migrate_system(working)
    else:
        migrated = copy.deepcopy(working)
        moon = migrated.bodies[AURA]
        if moon.mass_kg != M_EARTH or moon.radius_m != R_EARTH_MEAN:
            raise ValueError("input is not Aura-migrated; use --migrate-aura")
        aura_report = None
    result, layout = rebalance_system(migrated, vesper_scale, muspellsheimr_scale,
                                      establish_baseline)
    return result, {"aura_migration": aura_report, "outer_layout": layout,
                    "current_masses": applied,
                    "star_masses_are_runtime_only": True,
                    "passed": bool(layout["passed"] and
                                   (aura_report is None or aura_report["passed"]))}


def write_rebalanced(source: Path, output: Path | None, backup: Path | None,
                     apply: bool = False, migrate_aura: bool = True,
                     vesper_scale: float = 4.0, muspellsheimr_scale: float = 5.0,
                     establish_baseline: bool = True,
                     state_path: Path | None = None) -> dict[str, Any]:
    source = source.resolve()
    current_state = (json.loads(state_path.read_text(encoding="utf-8"))
                     if state_path is not None else None)
    destination = (output or source).resolve() if apply else output.resolve() if output else None
    if destination is None:
        original = load_world(source)
        _, report = prepare_system(original, migrate_aura, vesper_scale,
                                   muspellsheimr_scale, establish_baseline,
                                   current_state)
        report["write"] = {"dry_run": True, "source": str(source),
                           "source_sha256": _sha256(source), "files_written": []}
        return report
    if backup is None:
        raise ValueError("a new --backup is mandatory before any write")
    backup = backup.resolve()
    if backup in (source, destination):
        raise ValueError("backup must differ from source and destination")
    if backup.exists():
        raise FileExistsError(f"refusing to overwrite backup: {backup}")
    if apply and destination != source:
        raise ValueError("--apply is only valid for in-place replacement")
    if destination == source and not apply:
        raise ValueError("in-place replacement requires --apply")
    if destination.exists() and destination != source:
        raise FileExistsError(f"refusing to overwrite candidate: {destination}")
    original = load_world(source)
    migrated, report = prepare_system(original, migrate_aura, vesper_scale,
                                      muspellsheimr_scale, establish_baseline,
                                      current_state)
    source_hash = _sha256(source)
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup_created = False
    try:
        with source.open("rb") as src, backup.open("xb") as dst:
            backup_created = True
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
        if _sha256(backup) != source_hash:
            raise IOError("backup hash mismatch")
    except Exception:
        if backup_created and backup.exists() and _sha256(backup) != source_hash:
            backup.unlink()
        raise
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".outer-", suffix=".oxygen",
                                         dir=destination.parent, delete=False) as stream:
            temporary = stream.name
        save_world(migrated, temporary)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    for body in system_to_dict(load_world(destination))["bodies"]:
        if {"parent", "parent_name", "children", "orbit"}.intersection(body):
            raise IOError("unexpected hierarchy in saved candidate")
    report["write"] = {"dry_run": False, "apply": bool(apply),
                       "source": str(source), "source_sha256": source_hash,
                       "backup": str(backup), "backup_sha256": _sha256(backup),
                       "output": str(destination), "output_sha256": _sha256(destination),
                       "files_written": [str(backup), str(destination)],
                       "candidate_status": "NOT_ADOPTED_PENDING_REVIEW"}
    return report


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--migrate-aura", action="store_true")
    parser.add_argument("--vesper-scale", type=float, default=4.0)
    parser.add_argument("--muspellsheimr-scale", type=float, default=5.0)
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--state", type=Path,
                        help="server /api/state JSON for current stellar masses")
    args = parser.parse_args(argv)
    try:
        report = write_rebalanced(
            args.source, args.output, args.backup, args.apply, args.migrate_aura,
            args.vesper_scale, args.muspellsheimr_scale, not args.no_baseline,
            args.state)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"schema": REPORT_SCHEMA, "passed": False,
                          "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
