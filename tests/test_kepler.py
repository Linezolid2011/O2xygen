"""Kepler propagation correctness: circularity, conservation, hierarchy."""
import numpy as np
import pytest

from o2xygen.constants import gm, DAY, M_EARTH, M_SUN
from o2xygen.kepler import Elements, elements_to_state, solve_kepler
from o2xygen.system import System, Body, STAR, PLANET, MOON


def test_circular_orbit_radius_constant():
    mu = gm(M_SUN)
    a = 1.0e11
    el = Elements(a_m=a)
    for t in np.linspace(0, 3 * DAY, 17):
        r, v = elements_to_state(el, mu, t)
        assert np.linalg.norm(r) == pytest.approx(a, rel=1e-9)


def test_energy_and_angular_momentum_conserved():
    mu = gm(M_SUN)
    el = Elements(a_m=2e11, e=0.4, inc=0.3, raan=1.0, argp=2.0, m0=0.7)
    r0, v0 = elements_to_state(el, mu, 0.0)
    E0 = 0.5 * np.dot(v0, v0) - mu / np.linalg.norm(r0)
    L0 = np.cross(r0, v0)
    r1, v1 = elements_to_state(el, mu, 12345.6)
    E1 = 0.5 * np.dot(v1, v1) - mu / np.linalg.norm(r1)
    L1 = np.cross(r1, v1)
    assert E1 == pytest.approx(E0, rel=1e-10)
    assert np.allclose(L1, L0, rtol=1e-10)


def test_kepler_equation_zero_eccentricity():
    for M in np.linspace(-7, 7, 15):
        assert solve_kepler(M, 0.0) == pytest.approx(np.mod(M, 2 * np.pi), abs=1e-12)


def test_hierarchical_positions_add():
    """Moon orbits planet orbits star: absolute position = sum of local offsets."""
    sys = System("test")
    sys.add(Body("star", STAR, mass_kg=M_SUN))
    sys.add(Body("planet", PLANET, parent="star",
                 elements=Elements(a_m=1e11, e=0.2, m0=1.0), mass_kg=M_EARTH))
    sys.add(Body("moon", MOON, parent="planet",
                 elements=Elements(a_m=3.8e8, e=0.05, m0=2.0), mass_kg=7.3e22))
    for t in (0.0, DAY, 17.3 * DAY):
        states = sys.state_at(t)
        rp = states["planet"][0]
        rm = states["moon"][0]
        rl, _ = elements_to_state(sys.bodies["moon"].elements, gm(M_EARTH), t)
        assert np.allclose(rm, rp + rl, rtol=1e-9, atol=1e-3)


def test_orbit_period_matches_kepler_third_law():
    sys = System("t")
    sys.add(Body("s", STAR, mass_kg=M_SUN))
    sys.add(Body("p", PLANET, parent="s", elements=Elements(a_m=2.5e11), mass_kg=M_EARTH))
    el = sys.bodies["p"].elements
    T = 2 * np.pi * np.sqrt(el.a_m**3 / gm(M_SUN))
    r0, _ = elements_to_state(el, gm(M_SUN), 0.0)
    r1, _ = elements_to_state(el, gm(M_SUN), T)
    assert np.allclose(r0, r1, rtol=1e-8)
