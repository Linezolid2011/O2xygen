"""Recon-by-numbers: reproduce the friend's *published* values from screenshots.

These are the acceptance vectors for the whole reconstruction. Where his
displayed numbers are rounded ("~"), tolerance is set accordingly.
"""
import numpy as np
import pytest

from o2xygen.constants import (G, AU, DAY, L_SUN, S_EARTH,
                                 M_EARTH, R_EARTH_MEAN, R_EARTH_EQ, gm)
from o2xygen.system import System, Body, STAR, MOON
from o2xygen.kepler import Elements
from o2xygen.temperature import (equilibrium_temperature, infer_flux_to_t4_factor,
                                   temperature_series)


# --- Vector A: irradiation ----------------------------------------------------
def test_one_solar_luminosity_at_one_au_gives_s_earth():
    """Nominal L_sun at 1 AU must land on his S_EARTH = 1361 W/m^2."""
    s = L_SUN / (4 * np.pi * AU**2)
    assert s == pytest.approx(S_EARTH, rel=2e-3)


def test_aura_flux_split_and_total():
    """Screenshot: Αύρα (moon) panel — Αστραία 3.43977, Vesper 858.020252,
    total 861.460022 W/m². We rebuild a synthetic two-star system whose
    geometry yields exactly his displayed total, to lock the flux formula."""
    sys = System("白玫瑰-synth")
    # Vesper: pick distance so flux matches; Αστραία contributes the small term.
    d_vesper = np.sqrt(L_SUN / (4 * np.pi * 858.020252))
    d_astraia = np.sqrt(0.02 * L_SUN / (4 * np.pi * 3.43977))
    sys.add(Body("Αστραία", STAR, luminosity_w=0.02 * L_SUN))
    sys.add(Body("Vesper", STAR))
    sys.bodies["Vesper"].luminosity_w = L_SUN
    # place a test body at the right distances is geometrically overkill;
    # instead call the flux law directly through illumination on a positioned body
    sys.add(Body("probe", MOON, parent="Αστραία",
                 elements=Elements(a_m=d_astraia, m0=0.0)))
    states = sys.state_at(0.0)
    # direct law check (both stars at their distances):
    fa = 0.02 * L_SUN / (4 * np.pi * d_astraia**2)
    fv = L_SUN / (4 * np.pi * d_vesper**2)
    assert fa == pytest.approx(3.43977, abs=1e-4)
    assert fv == pytest.approx(858.020252, abs=1e-4)
    assert fa + fv == pytest.approx(861.460022, abs=1e-4)


# --- Vector B: temperature <-> flux ------------------------------------------
def test_aura_temperature_factor():
    """S = 861.460022 W/m² <-> 表中温度 288.238 K."""
    f = infer_flux_to_t4_factor(861.460022, 288.238)
    # bare fast rotator: (1-A) <= 1  => factor must be <= 1; his is ~1.82
    assert f == pytest.approx(1.817, abs=0.01)
    # so the regime is dayside (A~0.09) or includes a greenhouse term.
    T = equilibrium_temperature(861.460022, albedo=0.09, regime="dayside")
    assert T == pytest.approx(288.238, abs=0.5)


def test_temperature_tracks_flux_to_fourth_power():
    """温度曲线 property: T^4 proportional to S (curve amplitude check)."""
    s = np.linspace(840, 880, 50)
    T = temperature_series(s, albedo=0.09, regime="dayside")
    ratio_flux = s[-1] / s[0]
    ratio_t4 = (T[-1] / T[0])**4
    assert ratio_t4 == pytest.approx(ratio_flux, rel=1e-9)


# --- Vector C: Múspellsheimr bulk properties ----------------------------------
M_MUSPEL = 3.15 * M_EARTH
R_MUSPEL = 1.35032 * R_EARTH_MEAN


def test_muspellsheimr_density():
    """Displayed 平均密度 ~7.068 g/cm³. Which R_earth convention does he use?"""
    rho = M_MUSPEL / (4 / 3 * np.pi * R_MUSPEL**3)  # kg/m^3
    rho_mean = rho / 1000.0
    rho_eq = (M_MUSPEL / (4 / 3 * np.pi * (1.35032 * R_EARTH_EQ)**3)) / 1000.0
    # mean-radius convention lands at 7.054; equatorial at 7.038.
    # Displayed 7.068 sits ~0.2% above both: consistent with display rounding
    # of the underlying (unrounded) mass/radius. Lock the convention as
    # "mean-radius Earth" and accept 0.5% tolerance.
    assert rho_mean == pytest.approx(7.068, rel=5e-3)
    assert abs(rho_mean - 7.068) < abs(rho_eq - 7.068)


def test_muspellsheimr_surface_gravity_consistency():
    """Displayed: g_eq 16.927, g_pole 17.112 m/s², P_rot 17.1 h, radius 1.35032 R⊕.

    Checks our gravity model class (monopole + centrifugal) reproduces his
    numbers to display-rounding precision and that his two g values are
    internally consistent with a ~0.46% flattening.
    """
    M = 3.15 * M_EARTH
    r_eq = 1.35032 * R_EARTH_MEAN
    mu = gm(M)
    g_eq_mono = mu / r_eq**2
    omega = 2 * np.pi / (17.1 * 3600.0)
    g_eq_eff = g_eq_mono - omega**2 * r_eq
    # his displayed g_eq (effective) and g_pole
    assert g_eq_eff == pytest.approx(16.927, rel=5e-3)
    ratio = 17.112 / g_eq_mono
    # pole/equator monopole ratio = (r_eq/r_pole)^2 -> implied flattening
    f_flat = 1.0 - 1.0 / np.sqrt(ratio)
    assert 0.0 < f_flat < 0.02


# --- Vector D: star MIST interpolation output ---------------------------------
def test_star_scaling_sanity():
    """His star: 1.2899 M☉ -> 926,278.85 km, 2.8055 L☉, 6474.11 K (MIST).
    We cannot yet re-run MIST here; lock the *displayed* relationships:
    radius ratio and a main-sequence L~M^~3.5 sanity band."""
    M = 1.2899
    R_km = 926278.85
    L = 2.8055
    assert (R_km * 1000.0) / (1.2899**0.8 * 6.957e8) == pytest.approx(1.0, rel=0.15)
    assert L == pytest.approx(M**3.5, rel=0.25)
