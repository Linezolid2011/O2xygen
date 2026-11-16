"""Planetary structure: polytrope presets, gravity, rotation flattening.

Reconstructs his 结构 tab: "密度按扁球体积计算"; "引力为 GM/r^2 单极近似，
两极离心加速度为零，赤道有效重力另加自转离心项"; 扁率模型 options
慢转近似 / Maclaurin / ToF4 / CMS.
"""
import numpy as np

from .constants import G
from .optimize import brentq

# ---------------------------------------------------------------- gravity ---

def surface_gravity(mu: float, r_m: float, omega: float = 0.0,
                    latitude_deg: float = 0.0) -> float:
    """Monopole gravity minus centrifugal term.

    mu = G*M [m^3/s^2]; at poles the centrifugal term vanishes (his rule).
    """
    lat = np.radians(latitude_deg)
    g = mu / r_m**2
    return g - omega**2 * r_m * np.cos(lat)**2


def gravity_pair(mu: float, r_eq_m: float, omega: float) -> tuple[float, float]:
    """(equatorial effective, polar) surface gravity, his two displayed values."""
    g_eq = surface_gravity(mu, r_eq_m, omega, 0.0)
    f = flattening_from_rotation(mu, r_eq_m, omega, model="slow")
    r_pole = r_eq_m * (1.0 - f)
    g_pole = mu / r_pole**2
    return g_eq, g_pole

# -------------------------------------------------------------- flattening ---

def flattening_slow_rotation(mu: float, r_eq_m: float, omega: float) -> float:
    """First-order (uniform-body) flattening: f = 5 omega^2 R^3 / (4 G M)."""
    return 5.0 * omega**2 * r_eq_m**3 / (4.0 * mu)


def maclaurin_q_of_e(e: float) -> float:
    """Rotation parameter q = omega^2/(pi G rho) of a Maclaurin spheroid.

    q(e) = 2(1-e^2)^{1/2}(3-2e^2)arcsin(e)/e^3 - 6(1-e^2)/e^2

    For e < 0.03 the closed form is the difference of two huge terms
    (catastrophic cancellation - at e=1e-9 it returns O(1) garbage with a
    sign that depends on the rounding path), so there we use the Taylor
    series in u = e^2 instead:

        q = (8/15)u + (8/105)u^2 - (64/3465)u^4 - (1024/45045)u^5
            - (1024/45045)u^6 - (16384/765765)u^7 - (8192/415701)u^8 + ...

    (coefficients exact; the u^3 term vanishes). Matched against the closed
    form to ~1e-10 relative at the e = 0.03 crossover.
    """
    e = float(e)
    if e <= 0.0:
        return 0.0
    if e < 0.03:
        u = e * e
        return u * (8.0 / 15.0 + u * (8.0 / 105.0 + u * u * (
            -64.0 / 3465.0 + u * (-1024.0 / 45045.0 + u * (
                -1024.0 / 45045.0 + u * (-16384.0 / 765765.0
                                        + u * -8192.0 / 415701.0))))))
    s = np.arcsin(e)
    return (2.0 * np.sqrt(1.0 - e**2) * (3.0 - 2.0 * e**2) * s / e**3
            - 6.0 * (1.0 - e**2) / e**2)


def flattening_maclaurin(mu: float, r_eq_m: float, omega: float) -> float:
    """Exact Maclaurin spheroid solution: invert q(e) for the eccentricity.

    q uses the TRUE mean density, i.e. that of the volume-equivalent sphere
    (radius r_vol = r_eq * (1-f)^{1/3}), not the equatorial-radius density;
    f is iterated to consistency with that density. Returns f = 1-b/a.
    """
    if omega == 0.0 or mu <= 0.0 or r_eq_m <= 0.0:
        return 0.0
    q_max = 0.2247                     # bifurcation point of Maclaurin sequence
    f = 0.0
    for _ in range(64):
        rho_mean = 3.0 * mu / (4.0 * np.pi * G * (r_eq_m**3 * (1.0 - f)))
        q = min(omega**2 / (np.pi * G * rho_mean), q_max)
        # lower bracket safely below the root even for tiny q (the series
        # branch keeps q_of_e accurate there)
        e_lo = min(1e-3, 0.5 * np.sqrt(15.0 * q / 8.0))
        e = brentq(lambda x: maclaurin_q_of_e(x) - q, e_lo, 0.95, xtol=1e-14)
        f_new = 1.0 - np.sqrt(1.0 - e**2)
        if abs(f_new - f) < 1e-14:
            f = float(f_new)
            break
        f = float(f_new)
    return f


def flattening_from_rotation(mu: float, r_eq_m: float, omega: float,
                             model: str = "slow") -> float:
    if model == "slow":
        return flattening_slow_rotation(mu, r_eq_m, omega)
    if model == "maclaurin":
        return flattening_maclaurin(mu, r_eq_m, omega)
    if model in ("tof4", "cms"):
        raise NotImplementedError(
            f"flattening model '{model}' not yet implemented (ToF4: Zharkov & "
            "Trubitsyn 1978; CMS: Hubbard 2013 - layered figures, needed for "
            "giant planets)")
    raise ValueError(model)

# -------------------------------------------------------------- polytropes ---

def polytrope_n1_profile(mass_kg: float, radius_m: float, r_grid=None):
    """Lane-Emden n=1 density profile: rho(r) = rho_c * sin(pi r/R) / (pi r/R).

    Analytic solution: M = 4 rho_c R^3 / pi  =>  rho_c = pi M / (4 R^3).
    This is his "n=1 是球形多方密度剖面的分层预设".

    np.sinc is exactly sin(pi x)/(pi x) with sinc(0) = 1, so r = 0 is handled
    without the 0/0 NaN (the old sin(x)/x form forced callers to skip r = 0).
    """
    if r_grid is None:
        r_grid = np.linspace(0.0, radius_m, 512)
    r_grid = np.asarray(r_grid, dtype=float)
    rho_c = np.pi * mass_kg / (4.0 * radius_m**3)
    shape = np.sinc(r_grid / radius_m)
    return r_grid, rho_c * shape


def layered_density_table(mass_kg: float, radius_m: float, n_layers: int = 4,
                          profile: str = "n1") -> list[dict]:
    """His 层/半径范围/相对密度 table: mean density of each radial shell."""
    if profile != "n1":
        raise NotImplementedError(profile)
    edges = np.linspace(0.0, radius_m, n_layers + 1)
    rows = []
    for i in range(n_layers):
        r0, r1 = edges[i], edges[i + 1]
        # integrate over the FULL shell (from r0, including 0 for layer 0) so
        # the shell mass matches the shell volume convention below
        rmid = np.linspace(r0, r1, 200)
        _, rho = polytrope_n1_profile(mass_kg, radius_m, rmid)
        shell_m = np.trapezoid(4.0 * np.pi * rmid**2 * rho, rmid)
        shell_v = 4.0 / 3.0 * np.pi * (r1**3 - r0**3)
        rows.append({
            "layer": i + 1,
            "radius_range_frac": (r0 / radius_m, r1 / radius_m),
            "mean_density_kg_m3": shell_m / shell_v,
        })
    return rows
