"""ADR-0036: pure draft placement vs confirmed flat-world insertion."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from o2xygen.constants import AU, DAY, G, M_EARTH, R_EARTH_MEAN, R_SUN
from o2xygen import world as world_mod
from o2xygen.kepler import solve_kepler
from preset import build_preset
from workspace import Workspace


def _direct(**overrides):
    data = {"name": "Probe", "kind": "body", "placement": "direct",
            "mass_value": 2.0, "mass_unit": "earth",
            "radius_value": 6371.0, "radius_unit": "km", "albedo": 0.25,
            "x_au": 2.0, "y_au": -1.0, "z_au": 0.5,
            "speed_km_s": 10.0, "speed_azimuth_deg": 0.0,
            "speed_elevation_deg": 90.0}
    data.update(overrides)
    return data


def test_preview_is_non_destructive_and_computes_direct_state(tmp_path,
                                                               monkeypatch):
    import app

    world = build_preset()
    ws = Workspace(world, grid_getter=None)
    monkeypatch.setattr(app, "SYSTEM", world)
    monkeypatch.setattr(app, "WS", ws)
    monkeypatch.setattr(app, "WORLDS_DIR", tmp_path)
    mtime = tmp_path / "白玫瑰.oxygen"
    world_mod.save_world(world, mtime)
    before = world_mod.system_to_dict(world)
    got = app.api_add_body_preview(_direct())
    np.testing.assert_allclose(got["p"], [2 * AU, -AU, 0.5 * AU])
    np.testing.assert_allclose(got["v"], [0, 0, 10000.0], atol=1e-9)
    assert got["mass_kg"] == pytest.approx(2 * M_EARTH)
    assert got["radius_m"] == pytest.approx(R_EARTH_MEAN)
    assert got["nonstar"]["mean_density_kg_m3"] > 0
    assert got["nonstar"]["surface_gravity_m_s2"] > 0
    assert world_mod.system_to_dict(world) == before
    assert len(ws.sim.sim.particles) == len(before["bodies"])
    assert not list(tmp_path.glob("*Probe*"))
    assert world_mod.system_to_dict(world_mod.load_world(mtime)) == before


def test_orbit_preview_uses_only_parent_and_candidate(monkeypatch):
    import app

    world = build_preset()
    ws = Workspace(world, grid_getter=None)
    monkeypatch.setattr(app, "SYSTEM", world)
    monkeypatch.setattr(app, "WS", ws)
    parent = "Vesper"
    mp = ws.sim.sim.particles[ws.sim._index[parent]].m
    mass = 2 * M_EARTH
    data = _direct(placement="orbit", parent=parent, a_au=3.0, e=0.2,
                   inc_deg=30.0, raan_deg=40.0, argp_deg=50.0, phase_pct=25.0)
    got = app.api_add_body_preview(data)
    rp, vp = ws.sim.live_state()[parent]
    eccentric_anomaly = solve_kepler(np.pi / 2, 0.2)
    np.testing.assert_allclose(np.linalg.norm(np.array(got["p"]) - rp),
                               3 * AU * (1 - 0.2 * np.cos(eccentric_anomaly)))
    assert got["period_days"] == pytest.approx(
        2 * np.pi * np.sqrt((3 * AU)**3 / (G * (mp + mass))) / DAY)
    assert got["periapsis_m"] == pytest.approx(2.4 * AU)
    assert got["apoapsis_m"] == pytest.approx(3.6 * AU)
    assert got["parent_radius_m"] == pytest.approx(
        world.bodies[parent].radius_m)
    assert got["parent_kind"] == "star"
    assert got["parent_teff_k"] > 0.0
    assert len(got["orbit_points"]) == 129
    assert len(got["major_axis"]) == 2
    # Every preview ellipse point uses the same parent origin, not the barycenter.
    np.testing.assert_allclose(np.array(got["orbit_points"][0]) - rp,
                               np.array(got["major_axis"][0]) - rp)


def test_semimajor_km_unit_matches_equivalent_au(monkeypatch):
    import app
    world = build_preset()
    ws = Workspace(world, grid_getter=None)
    monkeypatch.setattr(app, "SYSTEM", world)
    monkeypatch.setattr(app, "WS", ws)
    base = _direct(placement="orbit", parent="Vesper", a_au=3.0, e=0.2)
    au = app.api_add_body_preview(base)
    km = app.api_add_body_preview(_direct(
        placement="orbit", parent="Vesper", a_value=3.0 * AU / 1000.0,
        a_unit="km", e=0.2))
    assert km["a_m"] == pytest.approx(3.0 * AU)
    assert km["period_days"] == pytest.approx(au["period_days"])
    np.testing.assert_allclose(km["p"], au["p"], rtol=0, atol=1e-3)


def test_kg_mass_and_scientific_notation_for_all_lengths(monkeypatch):
    import app
    world = build_preset()
    ws = Workspace(world, grid_getter=None)
    monkeypatch.setattr(app, "SYSTEM", world)
    monkeypatch.setattr(app, "WS", ws)
    got = app.api_add_body_preview(_direct(
        mass_value="1.23e4", mass_unit="kg",
        radius_value="6.371e3", radius_unit="km",
        placement="orbit", parent="Vesper",
        a_value="1.495978707e8", a_unit="km"))
    assert got["mass_kg"] == pytest.approx(12300.0)
    assert got["radius_m"] == pytest.approx(6371000.0)
    assert got["a_m"] == pytest.approx(AU)


def test_roche_warns_candidate_periapsis_but_not_safe_orbit(monkeypatch):
    import app
    world = build_preset()
    ws = Workspace(world, grid_getter=None)
    monkeypatch.setattr(app, "SYSTEM", world)
    monkeypatch.setattr(app, "WS", ws)
    got = app.api_add_body_preview(_direct(
        placement="orbit", parent="Vesper", a_au=0.0001, e=0.0))
    assert got["roche"]["candidate_violation"] is True
    assert got["roche"]["parent_violation"] is False
    assert "洛希极限" in got["roche"]["messages"][0]
    safe = app.api_add_body_preview(_direct(
        placement="orbit", parent="Vesper", a_au=10.0, e=0.0))
    assert safe["roche"]["candidate_violation"] is False
    assert safe["roche"]["parent_violation"] is False
    assert safe["roche"]["messages"] == []


def test_roche_warns_when_parent_is_inside_candidate_limit(monkeypatch):
    import app
    world = build_preset()
    ws = Workspace(world, grid_getter=None)
    monkeypatch.setattr(app, "SYSTEM", world)
    monkeypatch.setattr(app, "WS", ws)
    monkeypatch.setattr(app, "_grid_getter", lambda: object())
    class P:
        mass_msun = 100.0
        radius_m = R_SUN
        luminosity_w = 3.828e26
        teff_k = 5772.0
        mdot_msun_per_yr = 1e-13
    monkeypatch.setattr(app, "_evaluate_stellar", lambda *args, **kwargs: P())
    got = app.api_add_body_preview(_direct(
        kind="star", placement="orbit", parent="Vesper", a_au=0.01,
        mini_msun=100.0, feh=0.0, age0_gyr=1.0))
    assert got["roche"]["candidate_violation"] is False
    assert got["roche"]["parent_violation"] is True
    assert "母星将位于" in got["roche"]["messages"][0]


def test_stellar_orbit_roche_response_is_json_serializable(monkeypatch):
    import app
    world = build_preset()
    ws = Workspace(world, grid_getter=None)
    monkeypatch.setattr(app, "SYSTEM", world)
    monkeypatch.setattr(app, "WS", ws)
    monkeypatch.setattr(app, "_grid_getter", lambda: object())
    class P:
        mass_msun = np.float64(1.0)
        radius_m = np.float64(R_SUN)
        luminosity_w = np.float64(3.828e26)
        teff_k = np.float64(5772.0)
        mdot_msun_per_yr = np.float64(1e-13)
    monkeypatch.setattr(app, "_evaluate_stellar", lambda *args, **kwargs: P())
    got = app.api_add_body_preview(_direct(
        kind="star", placement="orbit", parent="Vesper", a_au=0.01,
        mini_msun=1.0, feh=0.0, age0_gyr=1.0))
    assert isinstance(got["roche"]["candidate_violation"], bool)
    assert isinstance(got["roche"]["parent_violation"], bool)
    import json
    json.dumps(got, allow_nan=False)  # regression: np.bool_ caused HTTP 500


def test_confirm_preserves_exact_draft_state_and_reloads_save(tmp_path,
                                                               monkeypatch):
    import app

    world = build_preset()
    ws = Workspace(world, grid_getter=None)
    monkeypatch.setattr(app, "SYSTEM", world)
    monkeypatch.setattr(app, "WS", ws)
    monkeypatch.setattr(app, "WORLDS_DIR", tmp_path)
    world_mod.save_world(world, tmp_path / "白玫瑰")
    monkeypatch.setattr(app, "CURRENT_WORLD", "白玫瑰")
    monkeypatch.setattr(app, "_evolution_active", lambda: False)
    before_states = ws.sim.live_state()
    preview = app.api_add_body_preview(_direct())
    got = app.api_add_body(_direct(confirm=True))
    assert got["ok"] is True
    saved = world_mod.load_world(tmp_path / "白玫瑰.oxygen")
    assert len(saved.bodies) == len(before_states) + 1
    added = saved.bodies["Probe"]
    np.testing.assert_allclose(added.pos0, preview["p"], rtol=0, atol=1e-3)
    np.testing.assert_allclose(added.vel0, preview["v"], rtol=0, atol=1e-9)
    for name, (_, vel) in before_states.items():
        np.testing.assert_array_equal(saved.bodies[name].vel0, vel)
    assert app.WS.sim.wind.mass_kg == 0.0
    assert "Probe" in app.SYSTEM.bodies
    assert "Probe" in app.WS.sim._index
    assert len(app.WS.sim._index) == len(before_states) + 1


def test_confirm_requires_confirmation_and_rejects_duplicates(monkeypatch):
    import app
    world = build_preset()
    ws = Workspace(world, grid_getter=None)
    monkeypatch.setattr(app, "SYSTEM", world)
    monkeypatch.setattr(app, "WS", ws)
    monkeypatch.setattr(app, "_evolution_active", lambda: False)
    with pytest.raises(app.HttpError) as exc:
        app.api_add_body(_direct())
    assert exc.value.status == 400
    with pytest.raises(app.HttpError) as exc:
        app.api_add_body_preview(_direct(name="Vesper"))
    assert exc.value.status == 400


def test_star_preview_uses_mist_inputs_without_derived_user_outputs(monkeypatch):
    import app
    world = build_preset()
    ws = Workspace(world, grid_getter=None)
    monkeypatch.setattr(app, "SYSTEM", world)
    monkeypatch.setattr(app, "WS", ws)
    monkeypatch.setattr(app, "_grid_getter", lambda: object())
    class P:
        mass_msun = 1.0
        radius_m = 6.957e8
        luminosity_w = 3.828e26
        teff_k = 5772.0
        mdot_msun_per_yr = 1e-13
    monkeypatch.setattr(app, "_evaluate_stellar", lambda *args, **kwargs: P())
    got = app.api_add_body_preview(_direct(kind="star", mini_msun=1.0,
                                          feh=0.0, age0_gyr=1.0))
    assert got["star"]["mass_msun"] == 1.0
    assert got["star"]["radius_rsun"] == 1.0
    assert got["star"]["luminosity_lsun"] == 1.0
    assert got["nonstar"] is None
