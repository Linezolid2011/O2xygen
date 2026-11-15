"""ADR-0015 + ADR-0016: real-time N-body integration over flat worlds.

Standards: two-body convergence against analytic Kepler, bounded energy
drift, adiabatic mass-loss orbital inflation, checkpoint scrubbing
determinism (incl. backward), world-origin osculating elements, stability
+ performance. Worlds are built with the placement tool (elements ->
Cartesian), exactly like saved .oxygen worlds are produced.
"""
import time
import numpy as np
import pytest

from o2xygen.constants import AU, DAY, YEAR, M_SUN, M_EARTH, L_SUN, G
from o2xygen.kepler import Elements, elements_to_state
from o2xygen.system import System, Body, STAR, BODY
from o2xygen.nbody import NBodySimulation


def place(sys, name, kind, around, a_m, e=0.0, inc=0.0, mass=0.0, m0=0.0,
          radius_m=0.0, albedo=0.0):
    """ADR-0016 placement tool: elements around an already-placed body ->
    Cartesian initial conditions."""
    ref = sys.bodies[around]
    mu = G * (ref.mass_kg + mass)
    r_rel, v_rel = elements_to_state(Elements(a_m=a_m, e=e, inc=inc, m0=m0),
                                     mu, 0.0)
    b = Body(name, kind, mass_kg=mass,
             pos0=ref.pos0 + r_rel, vel0=ref.vel0 + v_rel,
             radius_m=radius_m, albedo=albedo)
    sys.add(b)
    return b


def two_body_system(e=0.3, massless=False):
    sys = System("two-body", origin="star")
    sys.add(Body("star", STAR, mass_kg=M_SUN, luminosity_w=L_SUN))
    mass = 0.0 if massless else M_EARTH
    el = Elements(a_m=AU, e=e, inc=0.5, raan=1.0, argp=2.0, m0=0.7)
    r_rel, v_rel = elements_to_state(el, G * (M_SUN + mass), 0.0)
    sys.add(Body("planet", BODY, mass_kg=mass, pos0=r_rel, vel0=v_rel))
    return sys


def five_body_system():
    """白玫瑰-like: origin star + 2 planets + moon, all flat (ADR-0016)."""
    sys = System("test", origin="S")
    sys.add(Body("S", STAR, mass_kg=M_SUN, luminosity_w=L_SUN))
    place(sys, "P1", BODY, "S", a_m=1.2 * AU, e=0.05, mass=20 * M_EARTH)
    place(sys, "M1", BODY, "P1", a_m=0.01 * AU, e=0.15, inc=0.4, mass=7.3e22, m0=1.0)
    place(sys, "P2", BODY, "S", a_m=2.5 * AU, e=0.1, mass=8 * M_EARTH, m0=2.0)
    return sys


# --- 1. two-body convergence -------------------------------------------------
@pytest.mark.parametrize("ecc", [0.0, 0.3, 0.7])
def test_two_body_matches_analytic_kepler(ecc):
    """Massless test particle: integrated state must track the closed form."""
    sys = two_body_system(e=ecc, massless=True)
    nb = NBodySimulation(sys)
    t = 50.0 * YEAR
    nb.integrate_to(t, record_checkpoints=False)
    got = nb.live_state()["planet"][0]
    want, _ = elements_to_state(sys.bodies["planet"].elements if False else
                                Elements(a_m=AU, e=ecc, inc=0.5, raan=1.0,
                                         argp=2.0, m0=0.7),
                                G * M_SUN, t)
    assert np.allclose(got, want, rtol=1e-5, atol=1e4)


def test_two_body_massive_planet_uses_total_mu():
    """With a massive planet the effective mu is G(M+m). Elements are
    relative quantities, so compare the planet-star vector (the absolute
    frame additionally drifts with the barycenter)."""
    sys = two_body_system(e=0.3, massless=False)
    nb = NBodySimulation(sys)
    t = 20.0 * YEAR
    nb.integrate_to(t, record_checkpoints=False)
    st = nb.live_state()
    got = st["planet"][0] - st["star"][0]
    el = Elements(a_m=AU, e=0.3, inc=0.5, raan=1.0, argp=2.0, m0=0.7)
    want, _ = elements_to_state(el, G * (M_SUN + M_EARTH), t)
    rel = np.linalg.norm(got - want) / AU
    assert rel < 1e-5


# --- 2. energy behaviour ------------------------------------------------------
def test_energy_drift_bounded_over_1000_orbits():
    sys = two_body_system(e=0.3, massless=False)
    nb = NBodySimulation(sys)

    def orbital_energy():
        ps = nb.sim.particles
        r = np.array(ps[1].xyz) - np.array(ps[0].xyz)
        v = np.array(ps[1].vxyz) - np.array(ps[0].vxyz)
        mu = G * (M_SUN + M_EARTH)
        return 0.5 * np.dot(v, v) - mu / np.linalg.norm(r)

    nb.integrate_to(0.0, record_checkpoints=False)
    E0 = orbital_energy()
    nb.integrate_to(1000.0 * YEAR, record_checkpoints=False)
    E1 = orbital_energy()
    assert abs((E1 - E0) / E0) < 1e-6     # symplectic: bounded, not linear drift


# --- 3. adiabatic stellar mass loss ------------------------------------------
def test_adiabatic_mass_loss_inflates_orbit():
    """M_star: 1.0 -> 0.5 linearly over 5000 yr. Slow isotropic mass loss
    must inflate a circular orbit to a ~ a0 * M0/Mf = 2 AU, e staying small.
    This is the physics Keplerian propagation cannot express."""
    sys = two_body_system(e=0.0, massless=True)
    nb = NBodySimulation(sys)
    M0, Mf, span = M_SUN, 0.5 * M_SUN, 5000.0 * YEAR
    cadence = 10.0 * YEAR
    t = 0.0
    while t < span:
        t = min(t + cadence, span)
        nb.integrate_to(t, record_checkpoints=False)
        frac = t / span
        nb.update_masses({"star": M0 + (Mf - M0) * frac})
    el = nb.osculating_elements("planet")     # vs origin (the star)
    assert el.a_m == pytest.approx(2.0 * AU, rel=0.03)
    assert el.e < 0.1


# --- 4. checkpoint scrubbing determinism --------------------------------------
def test_state_at_matches_continuous_integration():
    """state_at (checkpoint restore + remainder) vs live integrate_to must agree."""
    sys = five_body_system()
    nb = NBodySimulation(sys)
    nb.integrate_to(200.0 * DAY)
    ref = nb.live_state()["M1"][0].copy()

    fresh = NBodySimulation(sys)
    got = fresh.state_at(200.0 * DAY)["M1"][0]
    assert np.allclose(got, ref, rtol=0, atol=1e-6 * AU)


def test_repeat_query_is_bitwise_stable():
    """Same scrub path twice must give identical answers (UI consistency)."""
    sys = five_body_system()
    nb = NBodySimulation(sys)
    nb.integrate_to(100.0 * DAY)
    a = nb.state_at(73.0 * DAY)["M1"][0]
    b = nb.state_at(73.0 * DAY)["M1"][0]
    assert np.array_equal(a, b)


def test_scrub_backward_and_repeat_determinism():
    """Rewind must land within a visually/physically irrelevant tolerance."""
    sys = five_body_system()
    nb = NBodySimulation(sys)
    nb.integrate_to(300.0 * DAY)
    nb.integrate_to(-50.0 * DAY)                     # rewind past epoch
    back = nb.live_state()["P1"][0].copy()
    assert nb.t_s == pytest.approx(-50.0 * DAY, abs=DAY)
    again = NBodySimulation(sys).state_at(-50.0 * DAY)["P1"][0]
    # path consistency, not path identity (see ADR-0015 amendment)
    assert np.allclose(again, back, rtol=0, atol=1e-2 * AU)

    # epoch recovery after the round trip (m0=0 for P1 -> |r| = a(1-e))
    nb.integrate_to(0.0)
    r0 = nb.live_state()["P1"][0]
    el0 = nb.osculating_elements("P1")
    assert np.linalg.norm(r0) == pytest.approx(1.2 * AU * (1 - 0.05), rel=1e-4)
    assert el0.a_m == pytest.approx(1.2 * AU, rel=1e-4)
    assert el0.e == pytest.approx(0.05, abs=1e-3)


# --- 5. world-origin osculating elements (ADR-0016) ---------------------------
def test_osculating_elements_roundtrip_vs_origin():
    """Elements -> placement -> osculating vs origin must round-trip at epoch."""
    sys = five_body_system()
    nb = NBodySimulation(sys)
    el = nb.osculating_elements("P1")                # default ref = origin
    assert el.a_m == pytest.approx(1.2 * AU, rel=1e-6)
    assert el.e == pytest.approx(0.05, abs=1e-6)

    # the moon, flattened: elements w.r.t. the STAR show a planet-scale orbit
    moon = nb.osculating_elements("M1")
    assert moon.a_m > 1.0 * AU        # not 0.01 AU - there is no parent anymore
    assert moon.e < 0.5               # and the eccentricity oscillates with time

    with pytest.raises(ValueError):
        nb.osculating_elements("S")   # origin itself has no elements


def test_osculating_vs_explicit_ref():
    sys = five_body_system()
    nb = NBodySimulation(sys)
    a = nb.osculating_elements("M1", ref="S")
    b = nb.osculating_elements("M1")                 # default = origin = S
    assert a.a_m == pytest.approx(b.a_m, rel=1e-12)


# --- 6. hierarchical stability + performance ----------------------------------
def test_five_body_year_integration_stable_and_fast():
    sys = five_body_system()
    nb = NBodySimulation(sys)
    t0 = time.perf_counter()
    nb.integrate_to(1.0 * YEAR)
    wall = time.perf_counter() - t0
    states = nb.live_state()
    for name, (r, v) in states.items():
        assert np.all(np.isfinite(r)) and np.all(np.isfinite(v))
    # M1 stays near P1 (bound in the N-body sense: within 2x initial separation)
    d = np.linalg.norm(states["M1"][0] - states["P1"][0])
    assert d < 2.0 * 0.01 * AU
    assert wall < 20.0, f"1 simulated year took {wall:.1f}s wall"


def test_illumination_from_live_state():
    sys = five_body_system()
    nb = NBodySimulation(sys)
    ill = nb.illumination("M1")
    st = nb.live_state()
    d = np.linalg.norm(st["M1"][0] - st["S"][0])   # exact geometry, no guessing
    assert ill["total"] == pytest.approx(L_SUN / (4 * np.pi * d * d), rel=1e-9)
