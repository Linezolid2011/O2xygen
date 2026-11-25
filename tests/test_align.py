"""ADR-0031: confirmed pure-translation barycenter alignment."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from o2xygen.constants import DAY, M_SUN, R_SUN
from o2xygen.system import System, Body, BODY
from o2xygen.nbody import NBodySimulation
from o2xygen import world as world_mod
from workspace import Workspace


def _offset_world():
    w = System("对齐", origin=None)
    w.add(Body("A", BODY, mass_kg=1.4 * M_SUN,
               pos0=np.array([3.0e11, -2.0e11, 1.0e11]),
               vel0=np.array([1200.0, -700.0, 300.0]),
               radius_m=R_SUN, spin_axis=[0.2, 0.3, 0.932],
               spin_rate_rad_s=2e-5, spin_phase_rad=0.4,
               extra={"marker": "keep-A"}))
    w.add(Body("B", BODY, mass_kg=0.9 * M_SUN,
               pos0=np.array([-4.0e11, 1.0e11, 2.0e11]),
               vel0=np.array([-500.0, 900.0, -200.0]),
               radius_m=0.8 * R_SUN, spin_axis=[-0.1, 0.4, 0.9],
               spin_rate_rad_s=3e-5, spin_phase_rad=1.2,
               extra={"marker": "keep-B"}))
    return w


def _state(nb):
    ps = nb.sim.particles
    return (np.array([p.m for p in ps]),
            np.array([p.xyz for p in ps]),
            np.array([p.vxyz for p in ps]))


def test_alignment_is_pure_translation_including_wind_coordinates():
    nb = NBodySimulation(_offset_world())
    nb.update_masses({"A": 1.25 * M_SUN, "B": 0.82 * M_SUN})
    nb.integrate_to(5.0 * DAY, record_checkpoints=False)
    m0, r0, v0 = _state(nb)
    t0 = nb.t_s
    wind0 = nb.wind.at(t0)
    bcm0 = (m0[:, None] * r0).sum(axis=0) / m0.sum()
    p_body0 = (m0[:, None] * v0).sum(axis=0)
    spins0 = {n: (b.spin_axis.copy(), b.spin_rate_rad_s, b.spin_phase_rad,
                  b.extra.copy())
              for n, b in nb.system.bodies.items()}

    shift, before, after = nb.align_barycenter_to_origin()
    m1, r1, v1 = _state(nb)
    np.testing.assert_allclose(shift, -bcm0, rtol=0, atol=1e-3)
    np.testing.assert_allclose(before, bcm0, rtol=0, atol=1e-3)
    np.testing.assert_allclose(after, np.zeros(3), rtol=0, atol=1e-3)
    np.testing.assert_array_equal(m1, m0)
    np.testing.assert_array_equal(v1, v0)       # no Galilean boost
    np.testing.assert_allclose(r1, r0 + shift, rtol=0, atol=1e-3)
    np.testing.assert_array_equal((m1[:, None] * v1).sum(axis=0), p_body0)
    assert nb.t_s == t0
    for n, (axis, rate, phase, extra) in spins0.items():
        b = nb.system.bodies[n]
        np.testing.assert_array_equal(b.spin_axis, axis)
        assert b.spin_rate_rad_s == rate
        assert b.spin_phase_rad == phase
        assert b.extra == extra

    wind1 = nb.wind.at(t0)
    assert wind1.mass_kg == wind0.mass_kg > 0.0  # no ledger reset
    np.testing.assert_array_equal(wind1.momentum_kg_m_s,
                                  wind0.momentum_kg_m_s)
    np.testing.assert_allclose(
        wind1.first_moment_kg_m,
        wind0.first_moment_kg_m + wind0.mass_kg * shift,
        rtol=0, atol=1e18)
    np.testing.assert_allclose(
        wind1.angular_momentum_kg_m2_s,
        wind0.angular_momentum_kg_m2_s + np.cross(
            shift, wind0.momentum_kg_m_s), rtol=0, atol=1e20)
    assert nb._cp_times == [t0]
    np.testing.assert_allclose(nb._cp_snaps[0][3].first_moment_kg_m,
                               wind1.first_moment_kg_m, rtol=0, atol=1e18)


def test_confirmed_alignment_saves_only_translated_state(tmp_path, monkeypatch):
    import app

    ws = Workspace(_offset_world(), grid_getter=None)
    ws.sim.update_masses({"A": 1.25 * M_SUN, "B": 0.82 * M_SUN})
    # Keep the non-star catalog in sync just as normal placement/edit paths do.
    ws.system.bodies["A"].mass_kg = 1.25 * M_SUN
    ws.system.bodies["B"].mass_kg = 0.82 * M_SUN
    m0, r0, v0 = _state(ws.sim)
    bcm0 = (m0[:, None] * r0).sum(axis=0) / m0.sum()
    wind0 = ws.sim.wind.at(ws.sim.t_s).to_dict()
    monkeypatch.setattr(app, "SYSTEM", ws.system)
    monkeypatch.setattr(app, "WS", ws)
    monkeypatch.setattr(app, "CURRENT_WORLD", "对齐")
    monkeypatch.setattr(app, "WORLDS_DIR", tmp_path)
    monkeypatch.setattr(app, "_evolution_active", lambda: False)

    with pytest.raises(app.HttpError) as exc:
        app.api_world_align({})
    assert exc.value.status == 400
    assert not (app._world_path("对齐")).exists()

    got = app.api_world_align({"confirm": True})
    assert got["ok"] is True
    np.testing.assert_allclose(got["translation_m"], -bcm0,
                               rtol=0, atol=1e-3)
    saved = world_mod.load_world(app._world_path("对齐"))
    names = list(saved.bodies)
    masses = np.array([saved.bodies[n].mass_kg for n in names])
    positions = np.array([saved.bodies[n].pos0 for n in names])
    velocities = np.array([saved.bodies[n].vel0 for n in names])
    np.testing.assert_allclose(
        (masses[:, None] * positions).sum(axis=0) / masses.sum(),
        np.zeros(3), rtol=0, atol=1e-3)
    np.testing.assert_allclose(positions, r0 - bcm0, rtol=0, atol=1e-3)
    np.testing.assert_array_equal(velocities, v0)
    np.testing.assert_array_equal(masses, m0)
    for n in names:
        assert saved.bodies[n].extra["marker"] == f"keep-{n}"
        assert saved.bodies[n].spin_rate_rad_s == ws.system.bodies[n].spin_rate_rad_s
        np.testing.assert_allclose(saved.bodies[n].spin_axis,
                                   ws.system.bodies[n].spin_axis,
                                   rtol=0, atol=1e-15)
        assert saved.bodies[n].spin_phase_rad == ws.system.bodies[n].spin_phase_rad
    assert saved.wind.mass_kg == wind0["mass_kg"]
    np.testing.assert_array_equal(saved.wind.momentum_kg_m_s,
                                  wind0["momentum_kg_m_s"])

    # Active evolution still rejects the operation after confirmation.
    monkeypatch.setattr(app, "_evolution_active", lambda: True)
    with pytest.raises(app.HttpError) as exc:
        app.api_world_align({"confirm": True})
    assert exc.value.status == 409
