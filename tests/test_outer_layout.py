"""Outer-group homothety, explicit hyperbolic rescue and backup tests."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from test_aura_migration import make_aura_system
from rebalance_outer_systems import (MUSPELL_GROUP, VESPER_GROUP, _group_cm,
                                     apply_current_masses, prepare_system,
                                     rebalance_system, write_rebalanced)
from migrate_aura import AURA, PRIMARY
from check_aura_stability import run_stability
from o2xygen.constants import AU, G, M_EARTH, M_SUN, R_EARTH_MEAN
from o2xygen.kepler import Elements, elements_to_state
from o2xygen.nbody import rv_to_elements
from o2xygen.system import BODY, STAR, Body
from o2xygen.world import load_world, save_world, system_to_dict


def _add_group(system, names, masses, kinds, outer, raw_offsets, raw_velocities):
    primary = system.bodies[PRIMARY]
    total = float(sum(masses))
    cm_offset = sum((m * np.asarray(r, dtype=float) for m, r in zip(masses, raw_offsets)),
                    start=np.zeros(3)) / total
    cm_v_offset = sum((m * np.asarray(v, dtype=float) for m, v in zip(masses, raw_velocities)),
                      start=np.zeros(3)) / total
    r_cm, v_cm = elements_to_state(outer, G * (primary.mass_kg + total), 0.0)
    cm = primary.pos0 + r_cm
    velocity = primary.vel0 + v_cm
    for name, mass, kind, raw_r, raw_v in zip(names, masses, kinds, raw_offsets, raw_velocities):
        extra = {"mini_msun": mass / M_SUN} if kind == STAR else {}
        system.add(Body(name, kind, mass_kg=mass,
                        pos0=cm + np.asarray(raw_r, dtype=float) - cm_offset,
                        vel0=velocity + np.asarray(raw_v, dtype=float) - cm_v_offset,
                        radius_m=0.0,  # placement fixture, not a collision model
                        extra=extra))


def make_outer_system():
    system = make_aura_system()
    del system.bodies["unrelated"]
    _add_group(
        system, VESPER_GROUP, (0.85 * M_SUN, 6.0 * M_SUN, 3.0e23),
        (STAR, STAR, BODY),
        Elements(a_m=33.93508459856977 * AU, e=0.9550198045889542,
                 inc=0.2, raan=0.4, argp=1.2, m0=3.5),
        (np.zeros(3), np.array([0.05 * AU, 0.0, 0.0]),
         np.array([0.051 * AU, 0.0102 * AU, 0.0])),
        (np.zeros(3),
         -3.0e23 / (6.0 * M_SUN + 3.0e23)
           * np.array([0.0, np.sqrt(G * (6.0 * M_SUN + 3.0e23) / (0.05 * AU)), 0.0])
         + np.array([0.0, 5000.0, 100.0]),
         6.0 * M_SUN / (6.0 * M_SUN + 3.0e23)
           * np.array([0.0, np.sqrt(G * (6.0 * M_SUN + 3.0e23) / (0.05 * AU)), 0.0])
         + np.array([0.0, 5000.0, 100.0])))
    gino_r = np.array([1.0e7, 0.0, 0.0])
    geo_r = np.array([-2.0e7, 1.0e7, 0.0])
    gino_v = np.array([0.0, np.sqrt(G * 3.15 * M_EARTH / np.linalg.norm(gino_r)), 0.0])
    geo_hat = np.array([-geo_r[1], geo_r[0], 0.0]) / np.linalg.norm(geo_r)
    geo_v = geo_hat * np.sqrt(G * 3.15 * M_EARTH / np.linalg.norm(geo_r))
    _add_group(
        system, MUSPELL_GROUP, (3.15 * M_EARTH, 3.84e19, 6.17e20),
        (BODY, BODY, BODY),
        Elements(a_m=80.0 * AU, e=0.20, inc=0.3, raan=2.2, argp=4.0, m0=2.4),
        (np.zeros(3), gino_r, geo_r),
        (np.zeros(3), gino_v, geo_v))
    return system


def _outer(system, names):
    cm, velocity, mass = _group_cm(system, names)
    primary = system.bodies[PRIMARY]
    return rv_to_elements(cm - primary.pos0, velocity - primary.vel0,
                          G * (primary.mass_kg + mass))


def _assert_internal_states_unchanged(old, new, names):
    for name in names:
        for other in names:
            if name >= other:
                continue
            assert np.allclose(new.bodies[name].pos0 - new.bodies[other].pos0,
                               old.bodies[name].pos0 - old.bodies[other].pos0,
                               rtol=0.0, atol=1e-3)
            assert np.allclose(new.bodies[name].vel0 - new.bodies[other].vel0,
                               old.bodies[name].vel0 - old.bodies[other].vel0,
                               rtol=0.0, atol=1e-8)


def test_bound_groups_homothetic_elements_internal_states_and_baseline():
    old = make_outer_system()
    new, report = rebalance_system(old, vesper_scale=4.0,
                                   muspellsheimr_scale=5.0,
                                   establish_baseline=True)
    assert report["passed"] is True
    for key, names, scale in (("vesper", VESPER_GROUP, 4.0),
                              ("muspellsheimr", MUSPELL_GROUP, 5.0)):
        group = report["groups"][key]
        old_el, new_el = _outer(old, names), _outer(new, names)
        assert new_el.a_m == pytest.approx(scale * old_el.a_m, rel=2e-11)
        assert new_el.e == pytest.approx(old_el.e, abs=2e-11)
        assert new_el.inc == pytest.approx(old_el.inc, abs=2e-11)
        assert new_el.raan == pytest.approx(old_el.raan, abs=2e-11)
        assert new_el.argp == pytest.approx(old_el.argp, abs=2e-11)
        assert new_el.nu == pytest.approx(old_el.nu, abs=2e-11)
        assert group["outer_two_body_shape_preserved"] is True
        assert group["bound_rescue_applied"] is False
        assert group["actual_velocity_scale"] == pytest.approx(1.0 / np.sqrt(scale))
        _assert_internal_states_unchanged(old, new, names)
        # CM absolute transform before/after the Galilean baseline is still
        # exact in primary-relative coordinates.
        old_cm, old_v, _ = _group_cm(old, names)
        new_cm, new_v, _ = _group_cm(new, names)
        assert np.allclose(new_cm - new.bodies[PRIMARY].pos0,
                           scale * (old_cm - old.bodies[PRIMARY].pos0),
                           rtol=2e-12, atol=1e-3)
        assert np.allclose(new_v - new.bodies[PRIMARY].vel0,
                           (old_v - old.bodies[PRIMARY].vel0) / np.sqrt(scale),
                           rtol=2e-11, atol=1e-8)
    assert report["method"]["barycentric_speed_after_baseline_m_s"] < 1e-7
    assert np.array_equal(new.bodies[PRIMARY].pos0, old.bodies[PRIMARY].pos0)
    assert new.wind.mass_kg == 0.0


def test_combined_aura_then_outer_layout_is_flat_and_mass_preserving():
    old = make_outer_system()
    new, report = prepare_system(old, migrate_aura=True)
    assert report["passed"] is True
    assert new.bodies[AURA].mass_kg == M_EARTH
    assert new.bodies[AURA].radius_m == R_EARTH_MEAN
    for name in old.bodies:
        if name != AURA:
            assert new.bodies[name].mass_kg == old.bodies[name].mass_kg
    for body in system_to_dict(new)["bodies"]:
        assert "pos0_m" in body and "vel0_m" in body
        assert not {"parent", "parent_name", "children", "orbit"}.intersection(body)
    assert report["aura_migration"]["passed"] is True


def test_initially_hyperbolic_group_requires_and_receives_explicit_bound_rescue():
    old = make_outer_system()
    names = MUSPELL_GROUP
    cm, velocity, mass = _group_cm(old, names)
    primary = old.bodies[PRIMARY]
    r = cm - primary.pos0
    v = velocity - primary.vel0
    escape = np.sqrt(2.0 * G * (primary.mass_kg + mass) / np.linalg.norm(r))
    boost = (1.30 * escape / np.linalg.norm(v) - 1.0) * v
    for name in names:
        old.bodies[name].vel0 = old.bodies[name].vel0 + boost
    assert _outer(old, names).e > 1.0
    new, report = rebalance_system(old, vesper_scale=4.0,
                                   muspellsheimr_scale=20.0,
                                   establish_baseline=True)
    group = report["groups"]["muspellsheimr"]
    assert group["bound_rescue_applied"] is True
    assert group["outer_two_body_shape_preserved"] is False
    assert group["new_speed_over_escape"] == pytest.approx(0.75, rel=1e-11)
    assert _outer(new, names).a_m > 0.0
    assert _outer(new, names).e < 1.0
    _assert_internal_states_unchanged(old, new, names)


def test_current_masses_are_runtime_only_and_do_not_replace_mist_seeds():
    system = make_outer_system()
    old_seed = system.bodies[PRIMARY].extra["mini_msun"]
    state = {"bodies": {name: {"m": body.mass_kg * 0.9}
                        for name, body in system.bodies.items()}}
    new, report = prepare_system(system, migrate_aura=True,
                                 vesper_scale=4.0,
                                 muspellsheimr_scale=5.0,
                                 current_state=state)
    assert report["star_masses_are_runtime_only"] is True
    assert report["current_masses"][PRIMARY] == pytest.approx(
        system.bodies[PRIMARY].mass_kg * 0.9)
    raw = system_to_dict(new)
    primary = next(b for b in raw["bodies"] if b["name"] == PRIMARY)
    assert primary["mini_msun"] == pytest.approx(old_seed)
    assert "mass_kg" not in primary


def test_writer_dry_run_backup_candidate_and_not_adopted_status(tmp_path):
    source = tmp_path / "source.oxygen"
    save_world(make_outer_system(), source)
    original_bytes = source.read_bytes()
    dry = write_rebalanced(source, None, None, migrate_aura=True)
    assert dry["write"]["files_written"] == []
    assert source.read_bytes() == original_bytes
    assert len(list(tmp_path.iterdir())) == 1

    candidate = tmp_path / "candidate.oxygen"
    backup = tmp_path / "backup.oxygen"
    report = write_rebalanced(source, candidate, backup, migrate_aura=True)
    assert backup.read_bytes() == original_bytes
    assert source.read_bytes() == original_bytes
    assert report["write"]["candidate_status"] == "NOT_ADOPTED_PENDING_REVIEW"
    loaded = load_world(candidate)
    assert loaded.bodies[AURA].mass_kg == M_EARTH
    raw = json.loads(candidate.read_text(encoding="utf-8"))
    assert all("parent" not in body and "orbit" not in body for body in raw["bodies"])


def test_stability_report_includes_outer_groups_and_encounter_distances():
    migrated, _ = prepare_system(make_outer_system(), migrate_aura=True)
    report = run_stability(migrated, years=0.02, sample_years=0.01,
                           scope="full", include_series=False,
                           migrate=False, continue_on_unbound=True)
    assert report["configuration"]["years_reached"] == pytest.approx(0.02)
    assert report["summary"]["vesper_group"]
    assert report["summary"]["muspellsheimr_group"]
    assert "vesper_star__neph_host_m" in report["encounters"]
    assert "vesper_group_cm__muspell_group_cm_m" in report["encounters"]
    assert "sampled_vesper_group_bound" in report["checks"]
    assert "sampled_muspellsheimr_group_bound" in report["checks"]
    json.dumps(report, allow_nan=False)
