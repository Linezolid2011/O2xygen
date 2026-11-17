"""Spin persistence + constant-time-lag tidal coupling (ADR-0023)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from o2xygen.constants import AU, DAY, G, M_EARTH, M_SUN, R_EARTH_MEAN, R_SUN
from o2xygen.system import System, Body, STAR, BODY
from o2xygen import world as world_mod


def test_spin_axis_rate_phase_round_trip(tmp_path):
    sys = System("自转", origin=None)
    b = Body("P", BODY, mass_kg=M_EARTH, radius_m=R_EARTH_MEAN)
    b.set_spin_vector([0.0, 1.0e-5, 2.0e-5], phase_rad=1.25)
    sys.add(b)
    back = world_mod.load_world(world_mod.save_world(sys, tmp_path / "spin"))
    rb = back.bodies["P"]
    assert np.allclose(rb.spin_axis, [0.0, 1 / np.sqrt(5), 2 / np.sqrt(5)],
                       rtol=0, atol=1e-15)
    assert rb.spin_rate_rad_s == pytest.approx(np.sqrt(5) * 1e-5, rel=1e-15)
    assert rb.spin_phase_rad == pytest.approx(1.25)
    assert rb.rotation_period_s == pytest.approx(2 * np.pi / rb.spin_rate_rad_s)
    # only one authoritative spin representation is stored
    d = world_mod.system_to_dict(sys)
    assert "rotation_period_s" not in d["bodies"][0]


def _tidal_system():
    """Circular two-body system; the planet spins at 4x its orbital rate."""
    a = 0.01 * AU
    m_star, m_planet = M_SUN, M_EARTH
    m_tot = m_star + m_planet
    n = np.sqrt(G * m_tot / a**3)
    sys = System("潮汐", origin=None)
    sys.add(Body("S", STAR, mass_kg=m_star, radius_m=R_SUN,
                 pos0=np.array([-a * m_planet / m_tot, 0, 0]),
                 vel0=np.array([0, -n * a * m_planet / m_tot, 0])))
    sys.add(Body("P", BODY, mass_kg=m_planet, radius_m=R_EARTH_MEAN,
                 pos0=np.array([a * m_star / m_tot, 0, 0]),
                 vel0=np.array([0, n * a * m_star / m_tot, 0]),
                 spin_axis=np.array([0.0, 0.0, 1.0]),
                 spin_rate_rad_s=4.0 * n,
                 extra={"love_k2": 0.3, "tidal_dt_s": 100.0,
                        "inertia_factor": 0.26}))
    return sys


def _total_angular_momentum(ws):
    ps = ws.sim.sim.particles
    out = np.zeros(3)
    for p in ps:
        out += p.m * np.cross(np.array(p.xyz), np.array(p.vxyz))
    for b in ws.system.bodies.values():
        I = float(b.extra.get("inertia_factor", 0.26)) * b.mass_kg * b.radius_m**2
        out += I * b.spin_vector()
    return out


def test_tides_spin_down_expand_orbit_and_conserve_angular_momentum():
    from workspace import Workspace
    ws = Workspace(_tidal_system(), grid_getter=None)
    p = ws.sim.sim.particles[ws.sim._index["P"]]
    s = ws.sim.sim.particles[ws.sim._index["S"]]
    r = np.array(p.xyz) - np.array(s.xyz)
    v = np.array(p.vxyz) - np.array(s.vxyz)
    h0 = float(np.linalg.norm(np.cross(r, v)))
    spin0 = ws.system.bodies["P"].spin_rate_rad_s
    L0 = _total_angular_momentum(ws)

    ws.advance_to(5.0)

    b = ws.system.bodies["P"]
    r = np.array(p.xyz) - np.array(s.xyz)
    v = np.array(p.vxyz) - np.array(s.vxyz)
    h1 = float(np.linalg.norm(np.cross(r, v)))
    assert b.spin_rate_rad_s < spin0, "fast spin must be tidally braked"
    assert h1 > h0, "spin angular momentum must feed the orbit"
    L1 = _total_angular_momentum(ws)
    assert np.allclose(L1, L0, rtol=1e-9, atol=0.0)
    # barycentric kicks: no net linear momentum appeared
    px = sum(q.m * q.vx for q in ws.sim.sim.particles)
    py = sum(q.m * q.vy for q in ws.sim.sim.particles)
    pz = sum(q.m * q.vz for q in ws.sim.sim.particles)
    p_scale = sum(q.m * np.linalg.norm(q.vxyz) for q in ws.sim.sim.particles)
    assert np.linalg.norm([px, py, pz]) < 1e-12 * p_scale
    assert np.isfinite(b.spin_phase_rad)
