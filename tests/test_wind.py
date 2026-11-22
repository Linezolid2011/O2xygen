"""ADR-0030: bodies are open; bodies + escaping wind are closed."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from o2xygen.constants import AU, DAY, G, M_SUN, R_SUN, YEAR
from o2xygen.system import System, Body, STAR, BODY
from o2xygen.nbody import NBodySimulation
from o2xygen.wind import WindLedger
from o2xygen import world as world_mod
from workspace import Workspace


def _body_integrals(m, r, v):
    m = np.asarray(m, dtype=float)
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    return {
        "M": float(m.sum()),
        "P": (m[:, None] * v).sum(axis=0),
        "L": (m[:, None] * np.cross(r, v)).sum(axis=0),
        "Q": (m[:, None] * r).sum(axis=0),
    }


def _sim_integrals(nb, t_s=None):
    ps = nb.sim.particles
    t_s = nb.sim.t if t_s is None else t_s
    b = _body_integrals([p.m for p in ps],
                        [p.xyz for p in ps], [p.vxyz for p in ps])
    w = nb.wind.at(t_s)
    return {key: b[key] + getattr(w, {
        "M": "mass_kg", "P": "momentum_kg_m_s",
        "L": "angular_momentum_kg_m2_s",
        "Q": "first_moment_kg_m"}[key]) for key in b}


def _assert_vector_close(got, want, scale, rtol=2e-9):
    np.testing.assert_allclose(got, want, rtol=rtol,
                               atol=max(1e-30, scale * rtol))


def test_mass_event_conserves_fixed_origin_totals_but_not_open_bodies():
    ledger = WindLedger(
        time_s=2.0, mass_kg=0.4,
        momentum_kg_m_s=[0.2, -0.1, 0.3],
        angular_momentum_kg_m2_s=[1.0, 2.0, -1.0],
        first_moment_kg_m=[3.0, -2.0, 0.5])
    old = np.array([2.0, 1.0])
    new = np.array([1.8, 0.7])
    r = np.array([[4.0, -1.0, 2.0], [-3.0, 5.0, 1.0]])
    v = np.array([[0.7, 0.2, -0.4], [-0.3, 0.8, 0.1]])
    t = 10.0
    before_b = _body_integrals(old, r, v)
    before_w = ledger.at(t)
    before = {
        "M": before_b["M"] + before_w.mass_kg,
        "P": before_b["P"] + before_w.momentum_kg_m_s,
        "L": before_b["L"] + before_w.angular_momentum_kg_m2_s,
        "Q": before_b["Q"] + before_w.first_moment_kg_m,
    }
    ledger.record(t, old, r, v, new)
    after_b = _body_integrals(new, r, v)
    after = {
        "M": after_b["M"] + ledger.mass_kg,
        "P": after_b["P"] + ledger.momentum_kg_m_s,
        "L": after_b["L"] + ledger.angular_momentum_kg_m2_s,
        "Q": after_b["Q"] + ledger.first_moment_kg_m,
    }
    assert after_b["M"] < before_b["M"]
    assert not np.allclose(after_b["P"], before_b["P"])
    assert not np.allclose(after_b["L"], before_b["L"])
    for key in before:
        np.testing.assert_allclose(after[key], before[key], rtol=2e-15,
                                   atol=2e-14)

    # Ballistic interval, then a second pulse: advance old wind BEFORE adding.
    moved = ledger.at(17.0)
    assert moved.mass_kg == ledger.mass_kg
    np.testing.assert_allclose(moved.momentum_kg_m_s,
                               ledger.momentum_kg_m_s)
    np.testing.assert_allclose(moved.angular_momentum_kg_m2_s,
                               ledger.angular_momentum_kg_m2_s)
    np.testing.assert_allclose(
        moved.first_moment_kg_m,
        ledger.first_moment_kg_m + 7.0 * ledger.momentum_kg_m_s)
    before_second = moved.to_dict()
    ledger.record(17.0, new, r, v, np.array([1.7, 0.65]))
    assert ledger.mass_kg > before_second["mass_kg"]


def _binary(kind=BODY, love_k2=0.0):
    m1, m2, a = 1.4 * M_SUN, 0.9 * M_SUN, 5.0 * AU
    mt = m1 + m2
    n = np.sqrt(G * mt / a**3)
    sys = System("wind-binary", origin=None)
    extra1 = {"mini_msun": 1.4, "feh": 0.0, "age0_gyr": 0.0,
              "love_k2": love_k2, "tidal_dt_s": 100.0}
    extra2 = {"mini_msun": 0.9, "feh": 0.0, "age0_gyr": 0.0,
              "love_k2": love_k2, "tidal_dt_s": 100.0}
    sys.add(Body("A", kind, mass_kg=m1,
                 pos0=[-a * m2 / mt, 0, 0], vel0=[0, -n * a * m2 / mt, 0],
                 radius_m=R_SUN, spin_axis=[0, 0, 1], spin_rate_rad_s=2e-5,
                 extra=extra1))
    sys.add(Body("B", kind, mass_kg=m2,
                 pos0=[a * m1 / mt, 0, 0], vel0=[0, n * a * m1 / mt, 0],
                 radius_m=0.8 * R_SUN, spin_axis=[0, 0, 1],
                 spin_rate_rad_s=-3e-5, extra=extra2))
    return sys


def test_differential_wind_conserves_full_binary_through_integration():
    nb = NBodySimulation(_binary())
    initial = _sim_integrals(nb)
    p_scale = sum(p.m * np.linalg.norm(p.vxyz) for p in nb.sim.particles)
    l_scale = sum(p.m * np.linalg.norm(np.cross(p.xyz, p.vxyz))
                  for p in nb.sim.particles)
    q_scale = sum(p.m * np.linalg.norm(p.xyz) for p in nb.sim.particles)
    span = 40.0 * YEAR
    for k in range(1, 9):
        t = span * k / 8
        nb.integrate_to(t, record_checkpoints=False)
        f = t / span
        nb.update_masses({"A": (1.4 - 0.35 * f) * M_SUN,
                          "B": (0.9 - 0.05 * f) * M_SUN})
        got = _sim_integrals(nb)
        assert got["M"] == pytest.approx(initial["M"], rel=2e-15)
        _assert_vector_close(got["P"], initial["P"], p_scale)
        _assert_vector_close(got["L"], initial["L"], l_scale)
        _assert_vector_close(got["Q"], initial["Q"], q_scale)
    assert nb.wind.mass_kg > 0.0


class _EvolvingGrid:
    """Two different deterministic winds plus structural expansion."""
    @staticmethod
    def evaluate(mini, feh, age_yr):
        class P:
            pass
        p = P()
        rate = 0.0015 if mini > 1.0 else 0.0004
        p.mass_msun = mini * (1.0 - rate * age_yr)
        p.radius_m = R_SUN * mini * (1.0 + 0.002 * age_yr)
        p.luminosity_w = 3.828e26 * mini**3
        p.teff_k = 5800.0 * mini
        p.mdot_msun_per_yr = mini * rate
        return p


def _snapshot_totals(snap):
    total = snap["conservation"]["total"]
    return {key: np.asarray(total[key], dtype=float) for key in (
        "momentum_kg_m_s", "angular_momentum_kg_m2_s",
        "first_moment_kg_m", "barycenter_m")}


def test_future_query_save_reload_and_structural_spin_conserve():
    ws = Workspace(_binary(STAR), grid_getter=lambda: _EvolvingGrid())
    start = ws.state_at(None)
    future_days = 20.0 * YEAR / DAY
    predicted = ws.state_at(future_days)
    assert ws.sim.t_s == 0.0
    assert ws.sim.wind.mass_kg == 0.0       # query used a private ledger

    ws.advance_to(future_days)
    live = ws.state_at(None)
    assert live["conservation"]["wind"]["mass_kg"] > 0.0
    before = _snapshot_totals(start)
    after = _snapshot_totals(live)
    p_scale = np.sum([b["m"] * np.linalg.norm(b["v"])
                      for b in start["bodies"].values()])
    l_scale = np.sum([b["m"] * np.linalg.norm(
        np.cross(b["p"], b["v"])) for b in start["bodies"].values()])
    q_scale = np.sum([b["m"] * np.linalg.norm(b["p"])
                      for b in start["bodies"].values()])
    query_scales = {"momentum_kg_m_s": p_scale,
                    "angular_momentum_kg_m2_s": l_scale,
                    "first_moment_kg_m": q_scale,
                    "barycenter_m": 5.0 * AU}
    for key in _snapshot_totals(live):
        np.testing.assert_allclose(_snapshot_totals(live)[key],
                                   _snapshot_totals(predicted)[key],
                                   rtol=2e-9,
                                   atol=2e-9 * query_scales[key])
    assert live["conservation"]["total"]["mass_kg"] == pytest.approx(
        start["conservation"]["total"]["mass_kg"], rel=2e-15)
    a = ws.system.bodies["A"]
    old_inertia = 1.4 * M_SUN * (1.4 * R_SUN)**2
    new_inertia = a.mass_kg * a.radius_m**2
    assert a.spin_rate_rad_s == pytest.approx(
        2e-5 * old_inertia / new_inertia, rel=2e-12)
    _assert_vector_close(after["momentum_kg_m_s"],
                         before["momentum_kg_m_s"], p_scale)
    _assert_vector_close(after["angular_momentum_kg_m2_s"],
                         before["angular_momentum_kg_m2_s"], l_scale)
    _assert_vector_close(after["first_moment_kg_m"],
                         before["first_moment_kg_m"], q_scale)

    # Rebase twice: no duplicate mass, no second Q += P*T, no spin reset.
    previous = live
    for _ in range(2):
        saved = ws.snapshot_for_save()
        assert saved.wind.time_s == 0.0
        assert saved.wind.mass_kg > 0.0
        back = world_mod.dict_to_system(world_mod.system_to_dict(saved))
        ws = Workspace(back, grid_getter=lambda: _EvolvingGrid())
        reloaded = ws.state_at(None)
        assert reloaded["conservation"]["wind"]["mass_kg"] == pytest.approx(
            previous["conservation"]["wind"]["mass_kg"], rel=2e-15)
        for key in _snapshot_totals(reloaded):
            np.testing.assert_allclose(_snapshot_totals(reloaded)[key],
                                       _snapshot_totals(previous)[key],
                                       rtol=2e-12,
                                       atol=2e-12 * query_scales[key])
        previous = reloaded


def test_total_angular_momentum_includes_tides_and_structure():
    ws = Workspace(_binary(STAR, love_k2=0.02),
                   grid_getter=lambda: _EvolvingGrid())
    before = ws.state_at(None)["conservation"]["total"]
    ws.advance_to(20.0)
    after = ws.state_at(None)["conservation"]["total"]
    scale = np.linalg.norm(before["angular_momentum_kg_m2_s"])
    np.testing.assert_allclose(after["angular_momentum_kg_m2_s"],
                               before["angular_momentum_kg_m2_s"],
                               rtol=2e-9, atol=max(1.0, 2e-9 * scale))
    # Structural I*omega closure is active, not a hard-coded constant spin.
    assert ws.system.bodies["A"].spin_rate_rad_s != pytest.approx(2e-5)


def test_manual_mass_change_resets_epoch_without_negative_wind():
    nb = NBodySimulation(_binary())
    nb.update_masses({"A": 1.3 * M_SUN, "B": 0.85 * M_SUN})
    assert nb.wind.mass_kg > 0.0
    nb.update_masses({"A": 1.5 * M_SUN}, record_wind=False)  # user may increase
    nb.reset_conservation_epoch(recenter=True)
    ps = nb.sim.particles
    m = np.array([p.m for p in ps])
    q = (m[:, None] * np.array([p.xyz for p in ps])).sum(axis=0)
    p = (m[:, None] * np.array([p.vxyz for p in ps])).sum(axis=0)
    np.testing.assert_allclose(q, 0.0, rtol=0, atol=1e28)
    np.testing.assert_allclose(p, 0.0, rtol=0, atol=1e20)
    assert nb.wind.mass_kg == 0.0
    assert np.all(nb.wind.momentum_kg_m_s == 0.0)
    assert nb._cp_snaps[-1][3].mass_kg == 0.0


def test_star_editor_resets_wind_and_preserves_relative_state(tmp_path,
                                                               monkeypatch):
    import app

    ws = Workspace(_binary(STAR), grid_getter=lambda: _EvolvingGrid())
    ws.advance_to(YEAR / DAY)
    assert ws.sim.wind.mass_kg > 0.0
    monkeypatch.setattr(app, "SYSTEM", ws.system)
    monkeypatch.setattr(app, "WS", ws)
    monkeypatch.setattr(app, "CURRENT_WORLD", "编辑测试")
    monkeypatch.setattr(app, "WORLDS_DIR", tmp_path)
    monkeypatch.setattr(app, "_evolution_active", lambda: False)
    monkeypatch.setattr(app, "_grid_getter", lambda: object())
    monkeypatch.setattr(app, "_evaluate_stellar",
                        lambda mini, feh, age, body=None:
                        _EvolvingGrid.evaluate(mini, feh, age))
    states = ws.sim.live_state()
    r_before = states["B"][0] - states["A"][0]
    v_before = states["B"][1] - states["A"][1]

    # Increasing the initial mass is legal for an editor, impossible for wind.
    got = app.api_star_edit({"name": "A", "mini_msun": 1.5,
                             "feh": 0.0, "age0_gyr": 0.0})
    assert got["ok"] is True
    assert ws.sim.wind.mass_kg == 0.0
    states = ws.sim.live_state()
    np.testing.assert_allclose(states["B"][0] - states["A"][0], r_before,
                               rtol=0, atol=1e-3)
    np.testing.assert_allclose(states["B"][1] - states["A"][1], v_before,
                               rtol=0, atol=1e-9)
    snap = ws.state_at(None)
    np.testing.assert_allclose(snap["conservation"]["bodies"]["first_moment_kg_m"],
                               0.0, rtol=0, atol=1e28)
    np.testing.assert_allclose(snap["conservation"]["bodies"]["momentum_kg_m_s"],
                               0.0, rtol=0, atol=1e20)
    saved = world_mod.load_world(tmp_path / "编辑测试.oxygen")
    assert saved.wind.mass_kg == 0.0
    assert np.all(saved.wind.momentum_kg_m_s == 0.0)
