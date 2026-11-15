"""Kepler module correctness (placement tool, ADR-0016): circularity,
conservation, and placement-vs-elements equivalence."""
import numpy as np
import pytest

from o2xygen.constants import gm, DAY, G, M_EARTH, M_SUN
from o2xygen.kepler import Elements, elements_to_state, solve_kepler
from o2xygen.system import System, Body, STAR, BODY


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


def test_elements_reject_nonpositive_semi_major_axis():
    """Audit finding B6: negative a_m used to flow through as NaN state
    vectors with only RuntimeWarnings."""
    with pytest.raises(ValueError):
        Elements(a_m=-1.0e11)
    with pytest.raises(ValueError):
        Elements(a_m=0.0)


def test_placement_matches_elements():
    """ADR-0016 placement tool: a body placed on elements around a reference
    body must carry exactly the elements' relative state (plus the ref's)."""
    sys = System("test", origin="star")
    sys.add(Body("star", STAR, mass_kg=M_SUN))
    el = Elements(a_m=1e11, e=0.2, m0=1.0)
    mu = G * (M_SUN + M_EARTH)
    r_rel, v_rel = elements_to_state(el, mu, 0.0)
    sys.add(Body("planet", BODY, mass_kg=M_EARTH, pos0=r_rel, vel0=v_rel))
    st = sys.initial_states()
    assert np.allclose(st["planet"][0], r_rel, rtol=1e-12)
    assert np.allclose(st["planet"][1], v_rel, rtol=1e-12)


def test_orbit_period_matches_kepler_third_law():
    el = Elements(a_m=2.5e11)
    T = 2 * np.pi * np.sqrt(el.a_m**3 / gm(M_SUN))
    r0, _ = elements_to_state(el, gm(M_SUN), 0.0)
    r1, _ = elements_to_state(el, gm(M_SUN), T)
    assert np.allclose(r0, r1, rtol=1e-8)
