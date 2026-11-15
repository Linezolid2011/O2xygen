"""Classical Keplerian elements -> Cartesian state, hierarchical propagation.

ORBIT is branded "KEPLERIAN SIMULATION": every body orbits its parent on
fixed two-body elements; no N-body integration is visible anywhere in the
UI (elements 半长轴/偏心率/倾角/周期, per-parent orbits for moons).
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Elements:
    """Keplerian elements around a parent.

    a_m    : semi-major axis [m]
    e      : eccentricity
    inc    : inclination [rad]
    raan   : longitude of ascending node [rad]
    argp   : argument of periapsis [rad]
    m0     : mean anomaly at the system epoch [rad]
    """
    a_m: float
    e: float = 0.0
    inc: float = 0.0
    raan: float = 0.0
    argp: float = 0.0
    m0: float = 0.0

    def __post_init__(self):
        if not (0.0 <= self.e < 1.0):
            raise ValueError("only bound elliptic orbits supported (0 <= e < 1)")


def solve_kepler(M: float, e: float, tol: float = 1e-13, max_iter: int = 50) -> float:
    """Eccentric anomaly from mean anomaly (Newton-Raphson, standard)."""
    M = np.mod(M, 2.0 * np.pi)
    E = M if e < 0.8 else np.pi  # standard starter
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        dE = -f / fp
        E += dE
        if abs(dE) < tol:
            break
    return E


def elements_to_state(el: Elements, mu: float, t_s: float):
    """Position/velocity (m, m/s) at time t_s after the system epoch.

    mu is the parent's gravitational parameter.
    """
    n = np.sqrt(mu / el.a_m**3)                      # mean motion
    M = el.m0 + n * t_s
    E = solve_kepler(M, el.e)

    # Perifocal (PQW) coordinates
    cosE, sinE = np.cos(E), np.sin(E)
    r = el.a_m * (1.0 - el.e * cosE)
    x_p = el.a_m * (cosE - el.e)
    y_p = el.a_m * np.sqrt(1.0 - el.e**2) * sinE
    xdot_p = -np.sqrt(mu * el.a_m) * sinE / r
    ydot_p = np.sqrt(mu * el.a_m) * np.sqrt(1.0 - el.e**2) * cosE / r

    # Rotation PQW -> inertial
    ci, si = np.cos(el.inc), np.sin(el.inc)
    co, so = np.cos(el.raan), np.sin(el.raan)
    cw, sw = np.cos(el.argp), np.sin(el.argp)
    R = np.array([
        [co * cw - so * sw * ci, -co * sw - so * cw * ci, so * si],
        [so * cw + co * sw * ci, -so * sw + co * cw * ci, -co * si],
        [sw * si, cw * si, ci],
    ])
    r_vec = R @ np.array([x_p, y_p, 0.0])
    v_vec = R @ np.array([xdot_p, ydot_p, 0.0])
    return r_vec, v_vec


def true_anomaly(el: Elements, mu: float, t_s: float) -> float:
    """True anomaly (for irradiation geometry / display)."""
    n = np.sqrt(mu / el.a_m**3)
    E = solve_kepler(el.m0 + n * t_s, el.e)
    return 2.0 * np.arctan2(np.sqrt(1 + el.e) * np.sin(E / 2),
                            np.sqrt(1 - el.e) * np.cos(E / 2))
