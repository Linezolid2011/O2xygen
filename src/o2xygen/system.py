"""Flat body catalog (ADR-0016): a world is a bag of bodies, no hierarchy.

Bodies are either stars or non-stars. Internal state = Cartesian initial
conditions (pos0/vel0, SI, inertial frame). The body's identity as a
"moon" or "planet" no longer exists - orbiting relationships EMERGE from
the dynamics and are described at query time by osculating elements
relative to the rest-of-system barycenter (ADR-0022).
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

STAR, BODY = "star", "body"


@dataclass
class Body:
    name: str
    kind: str                                  # STAR | BODY
    mass_kg: float = 0.0                       # star: INITIAL mass (MIST seed)
    pos0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    radius_m: float = 0.0                      # initial radius (stars evolve it)
    albedo: float = 0.0                        # non-stars
    rotation_period_s: Optional[float] = None  # display cache, derived from spin
    spin_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    spin_rate_rad_s: Optional[float] = None    # |omega|; axis carries the sense
    spin_phase_rad: float = 0.0                # phase at the world epoch
    luminosity_w: float = 0.0                  # stars: epoch value (settled)
    extra: dict = field(default_factory=dict)  # stars: feh, age0_gyr, mini_msun, teff_k

    def __post_init__(self):
        self.pos0 = np.asarray(self.pos0, dtype=float)
        self.vel0 = np.asarray(self.vel0, dtype=float)
        axis = np.asarray(self.spin_axis, dtype=float).reshape(3)
        norm = float(np.linalg.norm(axis))
        self.spin_axis = axis / norm if np.isfinite(norm) and norm > 1e-15 \
            else np.array([0.0, 0.0, 1.0])
        if self.spin_rate_rad_s is None:
            # legacy placement/save field: period -> angular speed once
            self.spin_rate_rad_s = (2.0 * np.pi / self.rotation_period_s
                                    if self.rotation_period_s
                                    and self.rotation_period_s > 0.0 else 0.0)
        self.spin_rate_rad_s = abs(float(self.spin_rate_rad_s))
        self.rotation_period_s = (2.0 * np.pi / self.spin_rate_rad_s
                                  if self.spin_rate_rad_s > 0.0 else None)
        self.spin_phase_rad = float(self.spin_phase_rad) % (2.0 * np.pi)

    def spin_vector(self) -> np.ndarray:
        """Angular velocity vector omega = spin_axis * spin_rate_rad_s."""
        return self.spin_axis * self.spin_rate_rad_s

    def set_spin_vector(self, omega, phase_rad: float | None = None) -> None:
        """Update spin from a vector, keeping axis/rate/period in sync.
        A zero spin keeps the last well-defined axis rather than writing NaN."""
        omega = np.asarray(omega, dtype=float).reshape(3)
        rate = float(np.linalg.norm(omega))
        if np.isfinite(rate) and rate > 1e-30:
            self.spin_axis = omega / rate
            self.spin_rate_rad_s = rate
            self.rotation_period_s = 2.0 * np.pi / rate
        else:
            self.spin_rate_rad_s = 0.0
            self.rotation_period_s = None
        if phase_rad is not None:
            self.spin_phase_rad = float(phase_rad) % (2.0 * np.pi)


class System:
    """A named world. t = seconds since the time origin (epoch_s = 0)."""

    def __init__(self, name: str = "", origin: Optional[str] = None):
        self.name = name
        self.origin = origin                     # display/frame reference body
        self.bodies: dict[str, Body] = {}
        self.epoch_s = 0.0

    def add(self, body: Body) -> None:
        if body.name in self.bodies:
            raise ValueError(f"duplicate body name {body.name!r}")
        self.bodies[body.name] = body

    def stars(self) -> list[Body]:
        return [b for b in self.bodies.values() if b.kind == STAR]

    def initial_states(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {n: (b.pos0.copy(), b.vel0.copy())
                for n, b in self.bodies.items()}


def illumination_from_states(system: System, states: dict,
                             body_name: str,
                             luminosity_w: Optional[dict] = None) -> dict[str, float]:
    """Per-star top-of-atmosphere beam flux [W/m^2] at body_name + 'total'.
    Kepler-layer equivalent of the N-body live computation - kept as the
    cheap analytic path for tests and tools."""
    pos = states[body_name][0]
    out = {}
    for s in system.stars():
        lum = (luminosity_w or {}).get(s.name, s.luminosity_w)
        d = np.linalg.norm(pos - states[s.name][0])
        if d <= 0.0:
            raise ValueError(
                f"irradiance of '{body_name}' undefined: coincident with "
                f"star '{s.name}' (self-irradiation is excluded)")
        out[s.name] = lum / (4.0 * np.pi * d * d)
    out["total"] = float(sum(out.values()))
    return out
