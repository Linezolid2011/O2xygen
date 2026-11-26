"""Pure-REBOUND IAS15 stability check for the migrated Αύρα system.

No Workspace server, stellar evolution, stellar wind, mass update, tides, or
other additional force is used.  The input is loaded and deep-copied; its save
is never written.  --migrate performs the migration only in that memory copy.

Full-world check (default):
    python scripts/check_aura_stability.py migrated.oxygen --years 30000 \
        --output stability.json

Read an old save, migrate in memory, and integrate only the three-body
Αστραία--Νεφέλη--Αύρα subsystem (an explicit, reported approximation):
    python scripts/check_aura_stability.py world.oxygen --migrate \
        --scope aura --years 30000 --output stability.json

Stdout is always one machine-readable JSON report.  --output additionally saves
it.  Progress, when requested, goes only to stderr.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import rebound

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from migrate_aura import (AURA, HOST, PRIMARY, _sha256, fluid_roche_limit_m,
                          hill_radii_m, migrate_system, pair_barycentre)
from o2xygen.atmosphere import normalize_atmosphere
from o2xygen.constants import G, M_EARTH, R_EARTH_MEAN, YEAR
from o2xygen.nbody import rv_to_elements
from o2xygen.system import System
from o2xygen.world import load_world
from rebalance_outer_systems import VESPER_GROUP, MUSPELL_GROUP

REPORT_SCHEMA = "o2xygen-aura-stability-v1"


def _total_energy(sim: rebound.Simulation) -> float:
    particles = sim.particles
    positions = np.array([p.xyz for p in particles], dtype=float)
    velocities = np.array([p.vxyz for p in particles], dtype=float)
    masses = np.array([p.m for p in particles], dtype=float)
    energy = float(np.sum(0.5 * masses * np.sum(velocities**2, axis=1)))
    for i in range(len(particles)):
        for j in range(i + 1, len(particles)):
            distance = float(np.linalg.norm(positions[i] - positions[j]))
            energy -= float(sim.G * masses[i] * masses[j] / distance)
    return energy


def _selected_system(source: System, scope: str) -> tuple[System, list[str]]:
    selected = copy.deepcopy(source)
    if scope == "full":
        return selected, []
    keep = {PRIMARY, HOST, AURA}
    excluded = [name for name in selected.bodies if name not in keep]
    selected.bodies = {name: body for name, body in selected.bodies.items()
                       if name in keep}
    selected.observation_points = [point for point in selected.observation_points
                                   if point.get("body") in keep]
    # No recentering: every retained absolute p/v is exactly the migrated one.
    return selected, excluded


def _element_sample(r: np.ndarray, v: np.ndarray,
                    mu: float) -> tuple[Any, dict[str, float]]:
    el = rv_to_elements(r, v, mu)
    r_norm = float(np.linalg.norm(r))
    v_norm = float(np.linalg.norm(v))
    escape_speed = float(np.sqrt(2.0 * mu / r_norm))
    return el, {
        "a_m": float(el.a_m),
        "e": float(el.e),
        "periapsis_m": float(el.a_m * (1.0 - el.e)),
        "inc_rad": float(el.inc),
        "raan_rad": float(el.raan),
        "argp_rad": float(el.argp),
        "true_anomaly_rad": float(el.nu),
        "r_m": r_norm,
        "radial_speed_m_s": float(np.dot(r, v) / r_norm),
        "speed_m_s": v_norm,
        "escape_speed_m_s": escape_speed,
        "speed_over_escape": v_norm / escape_speed,
        "specific_energy_j_kg": float(v_norm**2 / 2.0 - mu / r_norm),
    }


def _series_summary(series: list[dict[str, Any]], orbit: str) -> dict[str, Any]:
    if not series:
        return {}
    first, last = series[0], series[-1]
    out: dict[str, Any] = {"initial": first[orbit], "final": last[orbit]}
    for key in ("a_m", "e", "periapsis_m"):
        values = [float(row[orbit][key]) for row in series]
        out[f"{key}_min"] = min(values)
        out[f"{key}_max"] = max(values)
        out[f"{key}_peak_to_peak"] = max(values) - min(values)
    return out


def run_stability(source: System, years: float = 30_000.0,
                  sample_years: float = 10.0, scope: str = "full",
                  include_series: bool = True, progress: bool = False,
                  source_path: Path | None = None,
                  migrate: bool = True,
                  continue_on_unbound: bool = False) -> dict[str, Any]:
    """Integrate a deep copy and return a JSON-safe stability report.

    ``scope='full'`` retains every body.  ``scope='aura'`` deliberately retains
    only primary/host/satellite to avoid letting an unrelated, very short-period
    binary dictate IAS15's global adaptive timestep for 30 millennia.
    """
    if not math.isfinite(years) or years <= 0.0:
        raise ValueError("years must be positive and finite")
    if not math.isfinite(sample_years) or sample_years <= 0.0:
        raise ValueError("sample_years must be positive and finite")
    if scope not in ("full", "aura"):
        raise ValueError("scope must be 'full' or 'aura'")

    started = time.perf_counter()
    if migrate:
        migrated, migration_report = migrate_system(source)
    else:
        migrated, migration_report = copy.deepcopy(source), None
    system, excluded = _selected_system(migrated, scope)
    names = list(system.bodies)
    index = {name: i for i, name in enumerate(names)}
    host = system.bodies[HOST]
    moon = system.bodies[AURA]
    primary = system.bodies[PRIMARY]
    if moon.mass_kg != M_EARTH or moon.radius_m != R_EARTH_MEAN:
        raise ValueError("input does not have exact Earth mass/radius; use --migrate")
    normalize_atmosphere(moon.extra.get("atmosphere"), allow_none=False)

    sim = rebound.Simulation()
    sim.G = G
    sim.integrator = "ias15"
    # REBOUND 5.x exposes integrator fields through sim.integrator.
    sim.integrator.epsilon = 1e-9
    # IAS15 is adaptive; this is only its initial trial step.
    initial_r = moon.pos0 - host.pos0
    initial_v = moon.vel0 - host.vel0
    inner_mu = G * (host.mass_kg + moon.mass_kg)
    inner_initial = rv_to_elements(initial_r, initial_v, inner_mu)
    inner_period = 2.0 * np.pi * np.sqrt(inner_initial.a_m**3 / inner_mu)
    sim.dt = float(inner_period / 100.0)
    for body in system.bodies.values():
        sim.add(m=body.mass_kg, x=body.pos0[0], y=body.pos0[1], z=body.pos0[2],
                vx=body.vel0[0], vy=body.vel0[1], vz=body.vel0[2],
                r=max(0.0, body.radius_m))
    sim.t = 0.0
    # Physical non-star radii are loaded from the flat save.  Star radii are
    # MIST-derived and absent from the save, hence zero here; Αύρα--Νεφέλη and
    # every other non-star pair are nevertheless checked at every step.
    sim.collision = "direct"
    sim.collision_resolve = "halt"

    duration_s = float(years * YEAR)
    cadence_s = float(sample_years * YEAR)
    roche_m = fluid_roche_limit_m(host.mass_kg, host.radius_m,
                                  moon.mass_kg, moon.radius_m)
    radii_sum_m = float(host.radius_m + moon.radius_m)
    outer_mu = G * (primary.mass_kg + host.mass_kg + moon.mass_kg)
    rcm0, vcm0 = pair_barycentre(host, moon)
    outer_initial = rv_to_elements(rcm0 - primary.pos0, vcm0 - primary.vel0,
                                   outer_mu)
    hill = hill_radii_m(outer_initial, host.mass_kg, primary.mass_kg)

    series: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    energy_initial = _total_energy(sim)
    energy_max_relative_drift = 0.0
    sampled_all_bound = True
    sampled_periapsis_safe = True
    sampled_inside_hill = True
    fatal = False
    unexpected_error = None
    previous_inner_bound = True
    previous_outer_bound = True
    previous_inside_hill = True
    available_groups = [(label, names) for label, names in
                        (("vesper_group", VESPER_GROUP),
                         ("muspellsheimr_group", MUSPELL_GROUP))
                        if all(name in index for name in names)]
    previous_group_bound = {label: True for label, _ in available_groups}
    sampled_group_bound = {label: True for label, _ in available_groups}
    minimum_encounters = {}

    def group_particle_cm(names) -> tuple[np.ndarray, np.ndarray, float]:
        particles = sim.particles
        total = float(sum(system.bodies[name].mass_kg for name in names))
        reference = particles[index[names[0]]]
        ref_r = np.array(reference.xyz, dtype=float)
        ref_v = np.array(reference.vxyz, dtype=float)
        cm_r = ref_r + sum((system.bodies[name].mass_kg *
                            (np.array(particles[index[name]].xyz) - ref_r)
                            for name in names[1:]), start=np.zeros(3)) / total
        cm_v = ref_v + sum((system.bodies[name].mass_kg *
                            (np.array(particles[index[name]].vxyz) - ref_v)
                            for name in names[1:]), start=np.zeros(3)) / total
        return cm_r, cm_v, total

    def record_sample() -> None:
        nonlocal energy_max_relative_drift, sampled_all_bound
        nonlocal sampled_periapsis_safe, sampled_inside_hill, fatal
        nonlocal previous_inner_bound, previous_outer_bound, previous_inside_hill
        particles = sim.particles
        hp, mp, pp = (particles[index[HOST]], particles[index[AURA]],
                      particles[index[PRIMARY]])
        hr = np.array(hp.xyz, dtype=float)
        mr = np.array(mp.xyz, dtype=float)
        pr = np.array(pp.xyz, dtype=float)
        hv = np.array(hp.vxyz, dtype=float)
        mv = np.array(mp.vxyz, dtype=float)
        pv = np.array(pp.vxyz, dtype=float)
        if not all(np.all(np.isfinite(x)) for x in (hr, mr, pr, hv, mv, pv)):
            raise FloatingPointError("non-finite primary/host/satellite state")
        ri = mr - hr
        vi = mv - hv
        total_pair_mass = host.mass_kg + moon.mass_kg
        satellite_fraction = moon.mass_kg / total_pair_mass
        rcm = hr + satellite_fraction * (mr - hr)
        vcm = hv + satellite_fraction * (mv - hv)
        ro = rcm - pr
        vo = vcm - pv
        inner_el, inner = _element_sample(ri, vi, inner_mu)
        outer_el, outer = _element_sample(ro, vo, outer_mu)
        separation = float(np.linalg.norm(ri))
        row = {
            "time_yr": float(sim.t / YEAR),
            "inner": inner,
            "outer": outer,
            "inner_separation_m": separation,
            "inner_specific_energy_j_kg": float(np.dot(vi, vi) / 2.0
                                                   - inner_mu / separation),
            "total_energy_j": _total_energy(sim),
        }
        series.append(row)
        group_positions = {}
        for label, names in available_groups:
            cm, cv, group_mass = group_particle_cm(names)
            group_positions[label] = cm
            group_mu = G * (primary.mass_kg + group_mass)
            group_el, group_sample = _element_sample(cm - pr, cv - pv, group_mu)
            row[label] = group_sample
            group_bound = bool(group_el.a_m > 0.0 and 0.0 <= group_el.e < 1.0)
            sampled_group_bound[label] &= group_bound
            if not group_bound and previous_group_bound[label]:
                events.append({"type": label + "_unbound", "time_yr": row["time_yr"],
                               "a_m": group_el.a_m, "e": group_el.e})
            elif group_bound and not previous_group_bound[label]:
                events.append({"type": label + "_rebound", "time_yr": row["time_yr"],
                               "a_m": group_el.a_m, "e": group_el.e})
            previous_group_bound[label] = group_bound
        def update_minimum(key, distance):
            distance = float(distance)
            if key not in minimum_encounters or distance < minimum_encounters[key]["distance_m"]:
                minimum_encounters[key] = {"distance_m": distance,
                                           "time_yr": row["time_yr"]}
        if "Vesper" in index:
            update_minimum("vesper_star__neph_host_m",
                           np.linalg.norm(hr - np.array(particles[index["Vesper"]].xyz)))
        if "vesper_group" in group_positions:
            update_minimum("vesper_group_cm__neph_pair_cm_m",
                           np.linalg.norm(rcm - group_positions["vesper_group"]))
        if "muspellsheimr_group" in group_positions:
            update_minimum("muspell_group_cm__neph_pair_cm_m",
                           np.linalg.norm(rcm - group_positions["muspellsheimr_group"]))
        if "vesper_group" in group_positions and "muspellsheimr_group" in group_positions:
            update_minimum("vesper_group_cm__muspell_group_cm_m",
                           np.linalg.norm(group_positions["vesper_group"]
                                          - group_positions["muspellsheimr_group"]))
        if abs(energy_initial) > 0.0:
            drift = abs((row["total_energy_j"] - energy_initial) / energy_initial)
            energy_max_relative_drift = max(energy_max_relative_drift, drift)
        inner_bound = bool(inner_el.a_m > 0.0 and 0.0 <= inner_el.e < 1.0)
        outer_bound = bool(outer_el.a_m > 0.0 and 0.0 <= outer_el.e < 1.0)
        sampled_all_bound &= inner_bound and outer_bound
        sampled_periapsis_safe &= bool(inner["periapsis_m"] > roche_m
                                       and inner["periapsis_m"] > radii_sum_m)
        sampled_inside_hill &= bool(separation < hill["semimajor_m"])
        if not inner_bound and previous_inner_bound:
            events.append({"type": "inner_unbound" if continue_on_unbound else "inner_escape",
                           "time_yr": row["time_yr"], "a_m": inner_el.a_m, "e": inner_el.e})
            if not continue_on_unbound:
                fatal = True
        elif inner_bound and not previous_inner_bound:
            events.append({"type": "inner_rebound", "time_yr": row["time_yr"],
                           "a_m": inner_el.a_m, "e": inner_el.e})
        if not outer_bound and previous_outer_bound:
            events.append({"type": "outer_unbound" if continue_on_unbound else "outer_escape",
                           "time_yr": row["time_yr"], "a_m": outer_el.a_m, "e": outer_el.e})
            if not continue_on_unbound:
                fatal = True
        elif outer_bound and not previous_outer_bound:
            events.append({"type": "outer_rebound", "time_yr": row["time_yr"],
                           "a_m": outer_el.a_m, "e": outer_el.e})
        inside_hill = separation < hill["semimajor_m"]
        if not inside_hill and previous_inside_hill:
            events.append({"type": "inner_hill_exit", "time_yr": row["time_yr"],
                           "separation_m": separation,
                           "hill_radius_m": hill["semimajor_m"]})
            if not continue_on_unbound:
                fatal = True
        elif inside_hill and not previous_inside_hill:
            events.append({"type": "inner_hill_return", "time_yr": row["time_yr"],
                           "separation_m": separation})
        previous_inner_bound = inner_bound
        previous_outer_bound = outer_bound
        previous_inside_hill = inside_hill

    record_sample()
    next_target = min(cadence_s, duration_s)
    next_progress = 0.1
    while sim.t < duration_s and not fatal:
        try:
            sim.integrate(float(next_target), exact_finish_time=1)
            record_sample()
        except rebound.Collision as exc:
            p1 = int(getattr(exc, "p1", -1))
            p2 = int(getattr(exc, "p2", -1))
            # REBOUND 5.2's halt exception does not expose p1/p2.  At the halt
            # state the overlapping pair is the minimum physical clearance.
            best = None
            for i in range(sim.N):
                for j in range(i + 1, sim.N):
                    pi, pj = sim.particles[i], sim.particles[j]
                    radii = float(pi.r + pj.r)
                    if radii <= 0.0:
                        continue
                    distance = float(np.linalg.norm(np.array(pi.xyz) - np.array(pj.xyz)))
                    clearance = distance - radii
                    if best is None or clearance < best[0]:
                        best = (clearance, i, j, distance, radii)
            if best is not None and (p1 < 0 or p2 < 0):
                _, p1, p2, _, _ = best
            events.append({
                "type": "collision", "time_yr": float(sim.t / YEAR),
                "particle_indices": [p1, p2],
                "bodies": [names[p1] if 0 <= p1 < len(names) else "unknown",
                           names[p2] if 0 <= p2 < len(names) else "unknown"],
                "distance_m": best[3] if best is not None else None,
                "radii_sum_m": best[4] if best is not None else None,
                "surface_clearance_m": best[0] if best is not None else None,
            })
            fatal = True
        except Exception as exc:
            unexpected_error = f"{type(exc).__name__}: {exc}"
            events.append({"type": "integration_error", "time_yr": float(sim.t / YEAR),
                           "error": unexpected_error})
            fatal = True
        if progress and sim.t / duration_s >= next_progress:
            print(f"stability: {sim.t / YEAR:.0f}/{years:.0f} yr, "
                  f"events={len(events)}", file=sys.stderr, flush=True)
            next_progress += 0.1
        if fatal:
            break
        next_target = min(next_target + cadence_s, duration_s)
        # Guard against a floating-point non-advancing target at long epochs.
        if next_target <= sim.t and sim.t < duration_s:
            next_target = float(np.nextafter(sim.t, duration_s))

    wall_s = time.perf_counter() - started
    completed = bool(not fatal and abs(sim.t - duration_s) <= max(1e-6, 1e-12 * duration_s))
    collision_or_escape = [event for event in events if event["type"] in
                           ("collision", "inner_escape", "outer_escape",
                            "inner_unbound", "outer_unbound", "inner_hill_exit",
                            "vesper_group_unbound", "muspellsheimr_group_unbound",
                            "integration_error")]
    energy_ok = bool(energy_max_relative_drift < 1e-6)
    checks = {
        "requested_duration_completed": completed,
        "no_collision_or_escape": not collision_or_escape,
        "sampled_inner_and_outer_orbits_bound": bool(sampled_all_bound),
        "sampled_inner_periapsis_safe": bool(sampled_periapsis_safe),
        "sampled_inner_separation_inside_hill": bool(sampled_inside_hill),
        "maximum_sampled_energy_drift_below_1e-6": energy_ok,
    }
    for label, _ in available_groups:
        checks[f"sampled_{label}_bound"] = bool(sampled_group_bound[label])
    stable = bool(all(checks.values()))
    outer_radii = [row["outer"]["r_m"] for row in series]
    bound_outer_a = [row["outer"]["a_m"] for row in series
                     if row["outer"]["a_m"] > 0.0 and row["outer"]["e"] < 1.0]
    trailing = series[-min(100, len(series)):]
    final_outer_bound = bool(series[-1]["outer"]["a_m"] > 0.0
                             and series[-1]["outer"]["e"] < 1.0)
    trailing_unbound = [not (row["outer"]["a_m"] > 0.0 and row["outer"]["e"] < 1.0)
                        for row in trailing]
    outward = [row["outer"]["radial_speed_m_s"] > 0.0 for row in trailing]
    radius_changes = np.diff([row["outer"]["r_m"] for row in trailing])
    sustained_outward = bool(len(trailing) > 1 and all(trailing_unbound)
                             and np.mean(outward) >= 0.95
                             and np.mean(radius_changes > 0.0) >= 0.90
                             and trailing[-1]["outer"]["r_m"] > trailing[0]["outer"]["r_m"])
    if sustained_outward:
        escape_class = "sustained_outward_unbound_likely_escape"
    elif final_outer_bound:
        escape_class = "bound_at_final_not_permanently_escaped"
    else:
        escape_class = "unbound_at_final_but_sustained_escape_not_established"
    escape_assessment = {
        "classification": escape_class,
        "final_outer_bound": final_outer_bound,
        "trailing_sample_count": len(trailing),
        "trailing_time_yr": [trailing[0]["time_yr"], trailing[-1]["time_yr"]],
        "trailing_unbound_fraction": float(np.mean(trailing_unbound)),
        "trailing_outward_radial_speed_fraction": float(np.mean(outward)),
        "trailing_positive_radius_change_fraction": float(np.mean(radius_changes > 0.0)) if len(radius_changes) else None,
        "criterion": "likely escape requires all trailing samples unbound, >=95% outward radial speed, >=90% positive radius changes, and net radius growth",
        "limitation": "sampled osculating/radial evidence, not an analytic proof of asymptotic escape",
    }
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "configuration": {
            "integrator": "REBOUND IAS15",
            "rebound_version": rebound.__version__,
            "ias15_epsilon": 1e-9,
            "initial_trial_dt_s": float(inner_period / 100.0),
            "gravitational_constant_si": G,
            "year_s": YEAR,
            "years_requested": float(years),
            "years_reached": float(sim.t / YEAR),
            "sample_interval_years": float(sample_years),
            "scope": scope,
            "integrated_bodies": names,
            "excluded_bodies": excluded,
            "particle_count": len(names),
            "masses_fixed": True,
            "stellar_wind_loss": False,
            "stellar_evolution": False,
            "tides_or_additional_forces": False,
            "input_deep_copied": True,
            "global_recentre_or_velocity_zeroing": False,
            "collision_detection": "REBOUND direct, stop on collision, checked each step",
            "collision_radius_limitation": "star radii are zero because only MIST seeds are saved; all non-star radii are physical",
            "escape_definition": "sampled local two-body e>=1/a<0, or sampled inner separation >= initial semimajor Hill radius",
            "continue_on_unbound": bool(continue_on_unbound),
        },
        "initial_migration": ({
            "performed_in_memory": True,
            "inner_velocity_scale": migration_report["method"]["inner_velocity_scale"],
            "outer_velocity_scale": migration_report["method"]["outer_velocity_scale"],
            "common_translation_m": migration_report["method"]["common_translation_m"],
            "migration_checks_passed": migration_report["passed"],
        } if migration_report is not None else {
            "performed_in_memory": False,
            "input_already_migrated": True,
        }),
        "safety_thresholds": {
            "host_on_satellite_fluid_roche_limit_m": roche_m,
            "physical_radii_sum_m": radii_sum_m,
            "host_hill_radius": hill,
        },
        "events": events,
        "unexpected_error": unexpected_error,
        "summary": {
            "sample_count": len(series),
            "inner": _series_summary(series, "inner"),
            "outer": _series_summary(series, "outer"),
            "energy_initial_j": energy_initial,
            "energy_final_j": series[-1]["total_energy_j"] if series else None,
            "energy_max_relative_drift_at_samples": energy_max_relative_drift,
            "outer_radius_m_min": min(outer_radii),
            "outer_radius_m_max": max(outer_radii),
            "outer_bound_a_m_min": min(bound_outer_a) if bound_outer_a else None,
            "outer_bound_a_m_max": max(bound_outer_a) if bound_outer_a else None,
            "extrema_are_sampled_not_continuous": True,
        },
        "escape_assessment": escape_assessment,
        "checks": checks,
        "stable": stable,
        "performance": {
            "wall_time_s": wall_s,
            "simulated_years_per_wall_second": float(sim.t / YEAR / wall_s) if wall_s > 0 else None,
            "rebound_steps": int(sim.steps_done),
        },
        "approximations": [
            "All orbital elements are osculating two-body diagnostics of absolute N-body states.",
            "a/e/periapsis extrema and bound/escape checks are evaluated at output samples, not at every IAS15 substep.",
            "Non-star collisions use direct detection every step; MIST-derived stellar radii are unavailable and set to zero.",
            "Fluid Roche and Hill radii are initial analytic diagnostics, not time-dependent disruption criteria.",
            "Unbound osculating events stop the run unless --continue-on-unbound is set; collisions always stop. Transient unbound states are not proof of permanent escape.",
        ],
        "source_verification": {},
        "series": series if include_series else [],
        "series_omitted_by_option": not include_series,
    }
    for label, _ in available_groups:
        report["summary"][label] = _series_summary(series, label)
    report["encounters"] = minimum_encounters
    if scope == "aura":
        report["approximations"].append(
            "Three-body subsystem: excluded bodies do not perturb this run; absolute retained states were not recentred.")
    if source_path is not None:
        report["source_verification"] = {
            "path": str(source_path.resolve()),
            "sha256_before": _sha256(source_path),
            "sha256_after": _sha256(source_path),
            "read_only": True,
        }
    return report


def _json_dump(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--years", type=float, default=30_000.0)
    parser.add_argument("--sample-years", type=float, default=10.0)
    parser.add_argument("--scope", choices=("full", "aura"), default="full")
    parser.add_argument("--migrate", action="store_true",
                        help="migrate an old input only in the memory copy")
    parser.add_argument("--output", type=Path, help="also save this JSON report")
    parser.add_argument("--continue-on-unbound", action="store_true",
                        help="continue after sampled transient unbound states and assess the final state")
    parser.add_argument("--no-series", action="store_true",
                        help="omit per-sample arrays from the report")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    source_hash = _sha256(args.source)
    try:
        if args.output:
            if args.output.resolve() == args.source.resolve():
                raise ValueError("--output must not overwrite the input world")
            if args.output.exists():
                raise FileExistsError(f"refusing to overwrite report: {args.output}")
        source = load_world(args.source)
        report = run_stability(source, years=args.years,
                               sample_years=args.sample_years, scope=args.scope,
                               include_series=not args.no_series,
                               progress=args.progress, source_path=args.source,
                               migrate=args.migrate,
                               continue_on_unbound=args.continue_on_unbound)
        report["source_verification"]["sha256_before"] = source_hash
        report["source_verification"]["unchanged"] = _sha256(args.source) == source_hash
        rendered = _json_dump(report)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(rendered + "\n")
        print(rendered)
        return 0 if report["stable"] else 1
    except Exception as exc:
        failure = {"schema": REPORT_SCHEMA, "stable": False,
                   "error": f"{type(exc).__name__}: {exc}",
                   "source_sha256_before": source_hash,
                   "source_sha256_after": _sha256(args.source)}
        print(_json_dump(failure))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
