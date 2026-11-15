"""Flat body catalog (ADR-0016): a world is a bag of bodies, no hierarchy.

Bodies are either stars or non-stars. Internal state = Cartesian initial
conditions (pos0/vel0, SI, inertial frame). The body's identity as a
"moon" or "planet" no longer exists - orbiting relationships EMERGE from
the dynamics and are described at query time by osculating elements
relative to the world's origin body.
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
    rotation_period_s: Optional[float] = None
    luminosity_w: float = 0.0                  # stars: epoch value (settled)
    extra: dict = field(default_factory=dict)  # stars: feh, age0_gyr, mini_msun, teff_k

    def __post_init__(self):
        self.pos0 = np.asarray(self.pos0, dtype=float)
        self.vel0 = np.asarray(self.vel0, dtype=float)


class System:
    """A named world. t = seconds since the time origin (epoch_s = 0)."""

    def __init__(self, name: str = "", origin: Optional[str] = None):
        self.name = name
        self.origin = origin                     # display/frame reference body
        self.bodies: dict[str, Body] = {}
        self.epoch_s = 0.0

    def add(self, body: Body) -> None:
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
        out[s.name] = lum / (4.0 * np.pi * d * d)
    out["total"] = float(sum(out.values()))
    return out
