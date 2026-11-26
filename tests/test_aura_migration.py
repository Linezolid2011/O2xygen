"""Αύρα migration invariants and short pure-IAS15 stability smoke test.

All file-writing tests use pytest's temporary directory.  The production world
is neither a fixture nor an output target.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from migrate_aura import (AURA, HOST, PRIMARY, fluid_roche_limit_m,
                          migrate_system, pair_barycentre, write_migration)
from check_aura_stability import run_stability
from o2xygen.atmosphere import default_atmosphere, normalize_atmosphere
from o2xygen.constants import (AU, G, M_EARTH, M_SUN, R_EARTH_MEAN)
from o2xygen.kepler import Elements, elements_to_state
from o2xygen.nbody import rv_to_elements
from o2xygen.system import BODY, STAR, Body, System
from o2xygen.world import load_world, save_world


def _angle_close(a: float, b: float) -> bool:
    return abs((a - b + np.pi) % (2.0 * np.pi) - np.pi) < 2e-11


def _assert_same_elements(old, new) -> None:
    assert new.a_m == pytest.approx(old.a_m, rel=2e-12)
    assert new.e == pytest.approx(old.e, abs=2e-12)
    assert _angle_close(new.inc, old.inc)
    assert _angle_close(new.raan, old.raan)
    assert _angle_close(new.argp, old.argp)
    assert _angle_close(new.nu, old.nu)


def make_aura_system() -> System:
    """Exact old pair-barycentric placement plus an unrelated absolute body."""
    system = System("Αύρα migration test", origin=None)
    m_primary = 1.29 * M_SUN
    m_host = 20.0 * M_EARTH
    m_moon = 7.3e22
    star = Body(PRIMARY, STAR, mass_kg=m_primary,
                pos0=np.array([3.1e11, -2.2e11, 8.0e10]),
                vel0=np.array([120.0, -340.0, 75.0]),
                extra={"mini_msun": 1.29, "feh": 0.1, "age0_gyr": 1.26})
    system.add(star)

    outer_r, outer_v = elements_to_state(
        Elements(a_m=1.6 * AU, e=0.08, inc=0.021, raan=0.1,
                 argp=0.5, m0=0.0),
        G * (m_primary + m_host + m_moon), 0.0)
    inner_r, inner_v = elements_to_state(
        Elements(a_m=0.01 * AU, e=0.15, inc=np.radians(24.0),
                 raan=0.9, argp=2.1, m0=1.0),
        G * (m_host + m_moon), 0.0)
    pair_r = star.pos0 + outer_r
    pair_v = star.vel0 + outer_v
    host = Body(HOST, BODY, mass_kg=m_host,
                pos0=pair_r - (m_moon / (m_host + m_moon)) * inner_r,
                vel0=pair_v - (m_moon / (m_host + m_moon)) * inner_v,
                radius_m=4.0 * R_EARTH_MEAN, albedo=0.34,
                extra={"ring": True, "cmf": 0.1})
    moon = Body(AURA, BODY, mass_kg=m_moon,
                pos0=pair_r + (m_host / (m_host + m_moon)) * inner_r,
                vel0=pair_v + (m_host / (m_host + m_moon)) * inner_v,
                radius_m=0.35 * R_EARTH_MEAN, albedo=0.09,
                extra={"love_k2": 0.3})
    system.add(host)
    system.add(moon)
    # A state sentinel: migration must not globally recenter or boost it.
    system.add(Body("unrelated", BODY, mass_kg=1.23e20,
                    pos0=np.array([-9.0e11, 4.0e11, -2.0e11]),
                    vel0=np.array([-1234.0, 5678.0, 91.0]),
                    radius_m=1.0e5, extra={"sentinel": "untouched"}))
    return system


def _inner_elements(system: System):
    host, moon = system.bodies[HOST], system.bodies[AURA]
    return rv_to_elements(moon.pos0 - host.pos0, moon.vel0 - host.vel0,
                          G * (host.mass_kg + moon.mass_kg))


def _outer_elements(system: System):
    host, moon, star = (system.bodies[HOST], system.bodies[AURA],
                        system.bodies[PRIMARY])
    rcm, vcm = pair_barycentre(host, moon)
    return rv_to_elements(rcm - star.pos0, vcm - star.vel0,
                          G * (star.mass_kg + host.mass_kg + moon.mass_kg))


def test_mass_radius_atmosphere_and_source_immutability():
    old = make_aura_system()
    before = copy.deepcopy(old)
    new, report = migrate_system(old)
    moon = new.bodies[AURA]
    assert moon.mass_kg == M_EARTH
    assert moon.mass_kg / M_EARTH == 1.0
    assert moon.radius_m == R_EARTH_MEAN
    assert moon.radius_m / R_EARTH_MEAN == 1.0
    assert moon.extra["atmosphere"] == default_atmosphere()
    assert moon.albedo == before.bodies[AURA].albedo
    assert moon.spin_axis.tolist() == before.bodies[AURA].spin_axis.tolist()
    assert moon.spin_rate_rad_s == before.bodies[AURA].spin_rate_rad_s
    assert new.bodies[HOST].mass_kg == 20.0 * M_EARTH
    assert new.bodies[HOST].radius_m == before.bodies[HOST].radius_m
    assert report["passed"] is True
    # The pure migration function must not mutate its input.
    for name in before.bodies:
        assert old.bodies[name].mass_kg == before.bodies[name].mass_kg
        assert np.array_equal(old.bodies[name].pos0, before.bodies[name].pos0)
        assert np.array_equal(old.bodies[name].vel0, before.bodies[name].vel0)
    assert "atmosphere" not in old.bodies[AURA].extra


def test_inner_two_body_elements_direction_phase_and_speed_are_preserved():
    old = make_aura_system()
    new, report = migrate_system(old)
    oh, om = old.bodies[HOST], old.bodies[AURA]
    nh, nm = new.bodies[HOST], new.bodies[AURA]
    old_r, old_v = om.pos0 - oh.pos0, om.vel0 - oh.vel0
    new_r, new_v = nm.pos0 - nh.pos0, nm.vel0 - nh.vel0
    scale = np.sqrt((oh.mass_kg + M_EARTH) / (oh.mass_kg + om.mass_kg))
    assert np.allclose(new_r, old_r, rtol=0.0, atol=1e-3)
    assert np.allclose(new_v, scale * old_v, rtol=2e-13, atol=2e-11)
    assert np.linalg.norm(new_v) == pytest.approx(
        scale * np.linalg.norm(old_v), rel=2e-13)
    # Parallel relative velocity: direction and phase do not change.
    assert np.linalg.norm(np.cross(old_v, new_v)) / (
        np.linalg.norm(old_v) * np.linalg.norm(new_v)) < 2e-13
    _assert_same_elements(_inner_elements(old), _inner_elements(new))
    assert abs(report["inner_orbit"]["difference"]["a_relative"]) < 2e-12


def test_outer_pair_barycentre_elements_and_common_translation_are_exact():
    old = make_aura_system()
    new, report = migrate_system(old)
    oh, om = old.bodies[HOST], old.bodies[AURA]
    nh, nm = new.bodies[HOST], new.bodies[AURA]
    old_rcm, old_vcm = pair_barycentre(oh, om)
    new_rcm, new_vcm = pair_barycentre(nh, nm)
    delta = (om.mass_kg / (oh.mass_kg + om.mass_kg)
             - M_EARTH / (oh.mass_kg + M_EARTH)) * (om.pos0 - oh.pos0)
    assert np.array_equal(nh.pos0, oh.pos0 + delta)
    assert np.array_equal(nm.pos0, om.pos0 + delta)
    assert np.allclose(new_rcm, old_rcm, rtol=0.0, atol=1e-3)
    assert np.allclose(report["method"]["common_translation_m"], delta,
                       rtol=0.0, atol=1e-8)
    assert np.linalg.norm(delta) > 0.0  # local translation, not a no-op

    outer_scale = np.sqrt(
        (old.bodies[PRIMARY].mass_kg + oh.mass_kg + M_EARTH)
        / (old.bodies[PRIMARY].mass_kg + oh.mass_kg + om.mass_kg))
    star = old.bodies[PRIMARY]
    expected_vcm = star.vel0 + outer_scale * (old_vcm - star.vel0)
    assert np.allclose(new_vcm, expected_vcm, rtol=2e-13, atol=2e-11)
    _assert_same_elements(_outer_elements(old), _outer_elements(new))
    assert report["pair_barycentre"]["position_closure_error_m"] < 1e-3


def test_absolute_velocity_solution_and_no_global_recenter():
    old = make_aura_system()
    new, _ = migrate_system(old)
    oh, om, star = old.bodies[HOST], old.bodies[AURA], old.bodies[PRIMARY]
    nh, nm = new.bodies[HOST], new.bodies[AURA]
    old_rcm, old_vcm = pair_barycentre(oh, om)
    inner_scale = np.sqrt((oh.mass_kg + M_EARTH) / (oh.mass_kg + om.mass_kg))
    outer_scale = np.sqrt((star.mass_kg + oh.mass_kg + M_EARTH)
                          / (star.mass_kg + oh.mass_kg + om.mass_kg))
    v_rel = inner_scale * (om.vel0 - oh.vel0)
    v_cm = star.vel0 + outer_scale * (old_vcm - star.vel0)
    expected_host_v = v_cm - M_EARTH / (oh.mass_kg + M_EARTH) * v_rel
    expected_moon_v = v_cm + oh.mass_kg / (oh.mass_kg + M_EARTH) * v_rel
    assert np.allclose(nh.vel0, expected_host_v, rtol=2e-13, atol=2e-11)
    assert np.allclose(nm.vel0, expected_moon_v, rtol=2e-13, atol=2e-11)
    # Every non-participating absolute state is bit-for-bit unchanged.
    for name in (PRIMARY, "unrelated"):
        assert np.array_equal(new.bodies[name].pos0, old.bodies[name].pos0)
        assert np.array_equal(new.bodies[name].vel0, old.bodies[name].vel0)
    # The total-system barycentre was not reset to zero or otherwise aligned.
    assert new.origin is None
    assert new.bodies["unrelated"].extra["sentinel"] == "untouched"


def test_hill_roche_bound_and_periapsis_safety_report():
    old = make_aura_system()
    new, report = migrate_system(old)
    host, moon = new.bodies[HOST], new.bodies[AURA]
    rho_host = host.mass_kg / (4.0 * np.pi * host.radius_m**3 / 3.0)
    rho_moon = moon.mass_kg / (4.0 * np.pi * moon.radius_m**3 / 3.0)
    expected_roche = 2.44 * host.radius_m * (rho_host / rho_moon)**(1.0 / 3.0)
    actual_roche = report["hill_and_roche"]["new"][
        "host_on_satellite_fluid_roche_limit_m"]
    assert actual_roche == pytest.approx(expected_roche, rel=1e-14)
    assert actual_roche == fluid_roche_limit_m(
        host.mass_kg, host.radius_m, moon.mass_kg, moon.radius_m)
    inner = _inner_elements(new)
    assert 0.0 <= inner.e < 1.0 and inner.a_m > 0.0
    assert inner.a_m * (1.0 - inner.e) > expected_roche
    assert inner.a_m * (1.0 - inner.e) > host.radius_m + moon.radius_m
    assert report["hill_and_roche"]["inner_a_over_new_hill_semimajor"] < 0.5
    assert report["checks"]["periapsis_above_fluid_roche_limit"]


def test_save_round_trip_is_flat_absolute_state_and_dry_run_writes_nothing(tmp_path):
    source = tmp_path / "source.oxygen"
    save_world(make_aura_system(), source)
    before_names = {path.name for path in tmp_path.iterdir()}
    source_bytes = source.read_bytes()
    dry = write_migration(source, None, None)
    assert dry["write"]["dry_run"] is True
    assert dry["write"]["files_written"] == []
    assert source.read_bytes() == source_bytes
    assert {path.name for path in tmp_path.iterdir()} == before_names

    candidate = tmp_path / "candidate.oxygen"
    backup = tmp_path / "backup.oxygen"
    written = write_migration(source, candidate, backup)
    assert source.read_bytes() == source_bytes
    assert backup.read_bytes() == source_bytes
    assert written["write"]["backup_sha256"] == written["write"]["source_sha256"]
    assert candidate.exists()
    raw = json.loads(candidate.read_text(encoding="utf-8"))
    assert raw["format"] == "o2xygen-world"
    for body in raw["bodies"]:
        assert "pos0_m" in body and "vel0_m" in body
        assert not {"parent", "parent_name", "children", "orbit"}.intersection(body)
    loaded = load_world(candidate)
    moon = loaded.bodies[AURA]
    assert moon.mass_kg == M_EARTH
    assert moon.radius_m == R_EARTH_MEAN
    assert moon.extra["atmosphere"] == normalize_atmosphere(default_atmosphere())


def test_in_place_requires_apply_and_backup_on_temporary_copy(tmp_path):
    source = tmp_path / "copy.oxygen"
    save_world(make_aura_system(), source)
    original_bytes = source.read_bytes()
    with pytest.raises(ValueError, match="--apply"):
        write_migration(source, source, tmp_path / "unused-backup.oxygen")
    assert source.read_bytes() == original_bytes
    assert not (tmp_path / "unused-backup.oxygen").exists()
    with pytest.raises(ValueError, match="backup"):
        write_migration(source, source, None, apply=True)
    assert source.read_bytes() == original_bytes

    backup = tmp_path / "pre-apply.oxygen"
    report = write_migration(source, source, backup, apply=True)
    assert backup.read_bytes() == original_bytes
    assert report["write"]["apply"] is True
    assert load_world(source).bodies[AURA].mass_kg == M_EARTH


def test_short_three_body_ias15_stability_report_is_machine_readable():
    old = make_aura_system()
    report = run_stability(old, years=0.05, sample_years=0.025,
                           scope="aura", include_series=True)
    assert report["configuration"]["integrator"] == "REBOUND IAS15"
    assert report["configuration"]["masses_fixed"] is True
    assert report["configuration"]["stellar_wind_loss"] is False
    assert report["configuration"]["tides_or_additional_forces"] is False
    assert report["configuration"]["excluded_bodies"] == ["unrelated"]
    assert report["configuration"]["years_reached"] == pytest.approx(0.05)
    assert report["events"] == []
    assert report["stable"] is True
    assert len(report["series"]) == 3
    for row in report["series"]:
        assert set(("a_m", "e", "periapsis_m")).issubset(row["inner"])
        assert set(("a_m", "e", "periapsis_m")).issubset(row["outer"])
    assert report["summary"]["energy_max_relative_drift_at_samples"] < 1e-6
    # This catches NaN/Infinity and numpy scalars that are not valid JSON.
    json.dumps(report, allow_nan=False)
    assert old.bodies[AURA].mass_kg == 7.3e22


def test_stability_accepts_already_migrated_copy_without_remigrating():
    old = make_aura_system()
    migrated, _ = migrate_system(old)
    report = run_stability(migrated, years=0.02, sample_years=0.01,
                           scope="aura", include_series=False, migrate=False)
    assert report["stable"] is True
    assert report["initial_migration"] == {
        "performed_in_memory": False, "input_already_migrated": True}
    assert report["series"] == []
    with pytest.raises(ValueError, match="--migrate"):
        run_stability(old, years=0.01, sample_years=0.01,
                      scope="aura", migrate=False)
