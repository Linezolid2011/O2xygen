"""Aggregate escaping-wind state for closed-system conservation (ADR-0030).

The N-body particles are an OPEN subsystem: isotropic stellar winds remove
mass without kicking the star.  This ledger retains the escaped mass and its
translational integrals so bodies + wind remain a closed system.  Wind is not
a gravitational particle and is deliberately excluded from rest-barycenter
orbital elements and rendering.

All angular moments use the FIXED world-coordinate origin, never the moving
barycenter.  Between ejection events the non-interacting wind is ballistic:
Q = sum(m*r) advances by P*dt, while M, P and L stay constant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


def _vec(value) -> np.ndarray:
    out = np.asarray(value if value is not None else np.zeros(3),
                     dtype=float).reshape(3).copy()
    if not np.all(np.isfinite(out)):
        raise ValueError("wind ledger vectors must be finite")
    return out


@dataclass
class WindLedger:
    """Escaped-wind integrals valid at ``time_s`` (SI throughout)."""

    time_s: float = 0.0
    mass_kg: float = 0.0
    momentum_kg_m_s: np.ndarray = field(
        default_factory=lambda: np.zeros(3))
    angular_momentum_kg_m2_s: np.ndarray = field(
        default_factory=lambda: np.zeros(3))
    first_moment_kg_m: np.ndarray = field(
        default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        self.time_s = float(self.time_s)
        self.mass_kg = float(self.mass_kg)
        if not np.isfinite(self.time_s) or not np.isfinite(self.mass_kg) \
                or self.mass_kg < 0.0:
            raise ValueError("wind ledger time must be finite and mass non-negative")
        self.momentum_kg_m_s = _vec(self.momentum_kg_m_s)
        self.angular_momentum_kg_m2_s = _vec(self.angular_momentum_kg_m2_s)
        self.first_moment_kg_m = _vec(self.first_moment_kg_m)

    def copy(self) -> "WindLedger":
        return WindLedger(self.time_s, self.mass_kg,
                          self.momentum_kg_m_s,
                          self.angular_momentum_kg_m2_s,
                          self.first_moment_kg_m)

    def at(self, t_s: float) -> "WindLedger":
        """A ballistic copy at ``t_s``; never mutates this ledger."""
        t_s = float(t_s)
        if not np.isfinite(t_s):
            raise ValueError("wind ledger query time must be finite")
        out = self.copy()
        out.first_moment_kg_m += out.momentum_kg_m_s * (t_s - out.time_s)
        out.time_s = t_s
        return out

    def clear(self, t_s: float = 0.0) -> None:
        """Start a new conservation epoch (manual edits, not a wind event)."""
        self.time_s = float(t_s)
        self.mass_kg = 0.0
        self.momentum_kg_m_s.fill(0.0)
        self.angular_momentum_kg_m2_s.fill(0.0)
        self.first_moment_kg_m.fill(0.0)

    def record(self, t_s: float, masses_kg, positions_m, velocities_m_s,
               new_masses_kg) -> None:
        """Eject the mass removed by one simultaneous stellar update.

        Positions and velocities are sampled immediately BEFORE the particle
        masses change.  Mass increases are not winds and are rejected loudly;
        initialization and manual editing must use their non-recording path.
        """
        old = np.asarray(masses_kg, dtype=float).reshape(-1)
        new = np.asarray(new_masses_kg, dtype=float).reshape(-1)
        pos = np.asarray(positions_m, dtype=float).reshape(-1, 3)
        vel = np.asarray(velocities_m_s, dtype=float).reshape(-1, 3)
        if not (len(old) == len(new) == len(pos) == len(vel)):
            raise ValueError("inconsistent wind mass-update arrays")
        if not all(np.all(np.isfinite(a)) for a in (old, new, pos, vel)):
            raise ValueError("wind mass-update values must be finite")
        delta = new - old
        if np.any(delta > 0.0):
            raise ValueError("physical wind updates cannot increase body mass")

        # Existing wind first moves to the ejection epoch; only then is the
        # new shell added.  This ordering is essential for multiple pulses.
        advanced = self.at(t_s)
        self.time_s = advanced.time_s
        self.mass_kg = advanced.mass_kg
        self.momentum_kg_m_s = advanced.momentum_kg_m_s
        self.angular_momentum_kg_m2_s = advanced.angular_momentum_kg_m2_s
        self.first_moment_kg_m = advanced.first_moment_kg_m
        for dm, r, v in zip(-delta, pos, vel):
            if dm == 0.0:
                continue
            self.mass_kg += dm
            self.momentum_kg_m_s += dm * v
            self.angular_momentum_kg_m2_s += dm * np.cross(r, v)
            self.first_moment_kg_m += dm * r

    def to_dict(self) -> dict:
        return {
            "time_s": float(self.time_s),
            "mass_kg": float(self.mass_kg),
            "momentum_kg_m_s": [float(x) for x in self.momentum_kg_m_s],
            "angular_momentum_kg_m2_s": [
                float(x) for x in self.angular_momentum_kg_m2_s],
            "first_moment_kg_m": [float(x) for x in self.first_moment_kg_m],
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "WindLedger":
        data = data or {}
        return cls(
            time_s=data.get("time_s", 0.0),
            mass_kg=data.get("mass_kg", 0.0),
            momentum_kg_m_s=data.get("momentum_kg_m_s", np.zeros(3)),
            angular_momentum_kg_m2_s=data.get(
                "angular_momentum_kg_m2_s", np.zeros(3)),
            first_moment_kg_m=data.get("first_moment_kg_m", np.zeros(3)),
        )
