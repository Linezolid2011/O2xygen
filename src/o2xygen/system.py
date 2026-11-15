"""Hierarchical body system: catalog, state propagation, illumination.

Mirrors the ORBIT body catalog (天体目录: 恒星/行星/卫星 tree) and the
bottom-timeline simulation clock (UTC epoch + speed in 天/秒).
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .constants import gm, DAY
from .kepler import Elements, elements_to_state


STAR, PLANET, MOON = "star", "planet", "moon"


@dataclass
class Body:
    name: str
    kind: str                                  # STAR | PLANET | MOON
    parent: Optional[str] = None               # None => orbits system barycenter/standstill
    elements: Optional[Elements] = None        # orbit around parent
    mass_kg: float = 0.0
    radius_m: float = 0.0                      # equatorial radius
    rotation_period_s: Optional[float] = None
    albedo: float = 0.0
    luminosity_w: float = 0.0                  # stars
    # evolutionary / structural extras get attached later (MIST hook etc.)
    extra: dict = field(default_factory=dict)


class System:
    """A named workspace (工作空间) of bodies, propagated Keplerian-hierarchically."""

    def __init__(self, name: str = ""):
        self.name = name
        self.bodies: dict[str, Body] = {}
        self.epoch_s = 0.0                       # t=0 of the elements (his "初始时刻")

    def add(self, body: Body) -> None:
        if body.elements is None and body.kind != STAR and body.parent is not None:
            raise ValueError(f"{body.name}: non-star needs elements around parent")
        self.bodies[body.name] = body

    def mu_of(self, body_name: str) -> float:
        b = self.bodies[body_name]
        return gm(b.mass_kg) if b.mass_kg > 0 else 0.0

    def state_at(self, t_s: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Absolute position/velocity of every body at t_s after epoch."""
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        # topological order: parents first (catalogs are small; simple loop suffices)
        pending = dict(self.bodies)
        for _ in range(len(pending) + 1):
            progressed = False
            for name, b in list(pending.items()):
                if b.parent is None:
                    out[name] = (np.zeros(3), np.zeros(3))
                    del pending[name]
                    progressed = True
                elif b.parent in out:
                    rp, vp = out[b.parent]
                    rl, vl = elements_to_state(b.elements, self.mu_of(b.parent), t_s)
                    out[name] = (rp + rl, vp + vl)
                    del pending[name]
                    progressed = True
            if not pending:
                break
            if not progressed:
                raise ValueError("parent cycle or missing parent in catalog")
        return out

    # -- irradiation: "辐照为大气顶入射的各恒星垂直光束通量之和" --------------
    def stars(self) -> list[Body]:
        return [b for b in self.bodies.values() if b.kind == STAR]

    def illumination(self, body_name: str, t_s: float) -> dict[str, float]:
        """Per-star top-of-atmosphere beam flux [W/m^2] + 'total'."""
        states = self.state_at(t_s)
        pos = states[body_name][0]
        out = {}
        for s in self.stars():
            d = np.linalg.norm(pos - states[s.name][0])
            if s.name == body_name or d < 1.0:     # 不含本星自辐射
                continue
            out[s.name] = s.luminosity_w / (4.0 * np.pi * d * d)
        out["total"] = float(sum(out.values()))
        return out

    def days_since_epoch(self, t_s: float) -> float:
        return t_s / DAY
