"""P3 structure module: MR relations, gravity, flattening, polytropes."""
import numpy as np
import pytest

from o2xygen.constants import M_EARTH, R_EARTH_MEAN, gm, DAY
from o2xygen import mass_radius, structure


# --- PREM MR (Zeng, Sasselov & Jacobsen 2016, ApJ 819:127) --------------------
def test_prem_recovers_earth():
    # Earth: CMF = 0.325 -> coefficient 1.07-0.21*0.325 = 1.00175 ~ 1
    r = mass_radius.radius_earth_r(1.0, 0.325, model="prem")
    assert r == pytest.approx(1.0, abs=3e-3)


def test_prem_cmf_window_enforced():
    with pytest.raises(ValueError):
        mass_radius.radius_earth_r(3.0, 0.5, model="prem")


def test_muspellsheimr_cmf_inference():
    """RECON FINDING: his Múspellsheimr radius is EXACTLY the Zeng+2016 PREM
    fit at CMF=0.38 (his UI's core-fraction field reads 0.38):
        R = (1.07 - 0.21*0.38) * 3.15^(1/3.7) = 1.3503 R_E  (his: 1.35032)
    So his displayed radius came from the PREM model, not the 2019 grid."""
    cmf_prem = mass_radius.infer_core_fraction(3.15, 1.35032, model="prem")
    assert cmf_prem == pytest.approx(0.38, abs=0.02)
    r_check = mass_radius.radius_earth_r(3.15, 0.38, model="prem")
    assert r_check == pytest.approx(1.35032, abs=2e-4)


# --- gravity -------------------------------------------------------------------
def test_gravity_rules_match_his_footnote():
    """Monopole + centrifugal; pole has none; equator subtracts omega^2 r."""
    mu = gm(M_EARTH)
    r = R_EARTH_MEAN
    omega = 2 * np.pi / DAY
    g0 = mu / r**2
    g_eq = structure.surface_gravity(mu, r, omega, 0.0)
    g_pole = structure.surface_gravity(mu, r, omega, 90.0)
    g_mid = structure.surface_gravity(mu, r, omega, 45.0)
    assert g_eq == pytest.approx(g0 - omega**2 * r, rel=1e-12)
    assert g_pole == pytest.approx(g0, rel=1e-12)
    assert g_eq < g_mid < g_pole


def test_muspellsheimr_gravity_pair_via_structure():
    """Reproduce his displayed (16.837, 17.112)-ish pair through the module."""
    mu = gm(3.15 * M_EARTH)
    r_eq = 1.35032 * R_EARTH_MEAN
    omega = 2 * np.pi / (17.1 * 3600.0)
    g_eq, g_pole = structure.gravity_pair(mu, r_eq, omega)
    assert g_eq == pytest.approx(16.927, rel=5e-3)
    assert g_pole == pytest.approx(17.112, rel=5e-3)


# --- flattening -----------------------------------------------------------------
def test_maclaurin_q_inversion_consistency():
    """Solve e from q, then q(e) must reproduce q."""
    mu = gm(0.3 * M_EARTH)             # dense small planet, strong rotation
    r = 0.9 * R_EARTH_MEAN
    omega = 2 * np.pi / (10 * 3600.0)  # keep q < 0.2247 (Maclaurin bifurcation)
    f = structure.flattening_maclaurin(mu, r, omega)
    rho = 3 * mu / (4 * np.pi * 6.67430e-11 * r**3)
    q = omega**2 / (np.pi * 6.67430e-11 * rho)
    e = np.sqrt(1 - (1 - f)**2)
    assert structure.maclaurin_q_of_e(e) == pytest.approx(q, rel=1e-8)


def test_slow_rotation_is_maclaurin_limit():
    """For slow rotation the first-order formula must approach Maclaurin."""
    mu = gm(M_EARTH)
    r = R_EARTH_MEAN
    omega = 2 * np.pi / (24 * 3600.0)
    f_slow = structure.flattening_slow_rotation(mu, r, omega)
    f_mac = structure.flattening_maclaurin(mu, r, omega)
    assert f_slow == pytest.approx(f_mac, rel=0.15)


def test_flattening_earth_like_sanity():
    mu = gm(M_EARTH)
    f = structure.flattening_maclaurin(mu, R_EARTH_MEAN, 2 * np.pi / (24 * 3600.0))
    assert 0.002 < f < 0.006           # Earth's f = 0.00335


# --- polytropes ------------------------------------------------------------------
def test_polytrope_n1_mass_conservation():
    m = 3.15 * M_EARTH
    R = 1.35032 * R_EARTH_MEAN
    r, rho = structure.polytrope_n1_profile(m, R)
    m_int = np.trapezoid(4 * np.pi * r**2 * rho, r)
    assert m_int == pytest.approx(m, rel=1e-3)


def test_layered_table_shells_partition_radius():
    rows = structure.layered_density_table(M_EARTH, R_EARTH_MEAN, n_layers=4)
    assert len(rows) == 4
    assert rows[0]["radius_range_frac"] == (0.0, 0.25)
    assert rows[0]["mean_density_kg_m3"] > rows[-1]["mean_density_kg_m3"]


def test_tof4_cms_notimplemented_yet():
    with pytest.raises(NotImplementedError):
        structure.flattening_from_rotation(gm(M_EARTH), R_EARTH_MEAN,
                                           2 * np.pi / DAY, model="tof4")
