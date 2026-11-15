"""Real-time N-body propagation (ADR-0015).

Replaces analytic Kepler as the propagation truth: the system state is
integrated live with rebound (WHFast by default; IAS15 for close encounters).
`kepler.py` is retained for initial-state construction and two-body
calibration truth.

Key properties:
  * deterministic: fixed timestep, checkpointed; scrubbing backward and
    forward reproduces the same state (time-symmetric queries);
  * stellar evolution coupling: masses updated between steps (adiabatic) so
    long timelines show genuine orbital inflation from wind mass loss;
  * osculating elements: the UI's a/e/i/period become derived outputs of the
    live state, no longer independent inputs after t0.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import numpy as np
import rebound

from .constants import G, DAY, gm
from .kepler import Elements, elements_to_state
from .system import System


@dataclass
class OsculatingElements:
    """Orbital elements of a body relative to its parent, from live state."""
    a_m: float
    e: float
    inc: float
    raan: float
    argp: float
    nu: float


def rv_to_elements(r_vec: np.ndarray, v_vec: np.ndarray, mu: float) -> OsculatingElements:
    """Standard Vallado COE conversion (prograde, non-singular cases)."""
    r = np.linalg.norm(r_vec)
    v2 = float(np.dot(v_vec, v_vec))
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    k_vec = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(k_vec, h_vec)
    n = np.linalg.norm(n_vec)
    e_vec = ((v2 - mu / r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / mu
    e = float(np.linalg.norm(e_vec))
    energy = v2 / 2.0 - mu / r
    if abs(energy) < 1e-15:
        raise ValueError("parabolic orbit: no semi-major axis")
    a = -mu / (2.0 * energy)
    inc = float(np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0)))
    raan = float(np.arctan2(n_vec[1], n_vec[0])) % (2 * np.pi) if n > 1e-12 else 0.0
    if e > 1e-12 and n > 1e-12:
        argp = float(np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * e), -1.0, 1.0)))
        if e_vec[2] < 0:
            argp = 2 * np.pi - argp
    else:
        argp = 0.0
    if e > 1e-12:
        nu = float(np.arccos(np.clip(np.dot(e_vec, r_vec) / (e * r), -1.0, 1.0)))
        if np.dot(r_vec, v_vec) < 0:
            nu = 2 * np.pi - nu
    else:
        nu = float(np.arctan2(np.dot(np.cross(n_vec / n if n > 1e-12 else k_vec, r_vec), h_vec / h),
                              np.dot((n_vec / n if n > 1e-12 else k_vec), r_vec))) % (2 * np.pi)
    return OsculatingElements(a, e, inc, raan, argp, nu)


class NBodySimulation:
    """Integrates a System live. Server-side truth for all state queries.

    integrator: "ias15" (default in the app layer; adaptive, excellent
    time-reversibility for timeline scrubbing, clean mass changes) or
    "whfast" (symplectic; kept for long-haul studies, see ADR-0015).
    """

    def __init__(self, system: System, integrator: str = "ias15",
                 dt_s: float | None = None,
                 checkpoint_every_s: float = 50.0 * DAY):
        self.system = system
        self.integrator = integrator
        self._order = self._particle_order(system)
        self._index = {name: i for i, name in enumerate(self._order)}

        states = self._initial_states(system)
        if dt_s is None:
            dt_s = self._default_dt(system)
        self.dt_s = float(dt_s)
        # checkpoint cadence aligned to the fixed step grid
        steps = max(1, int(round(checkpoint_every_s / self.dt_s)))
        self.checkpoint_every_s = steps * self.dt_s

        self.sim = self._fresh_sim()
        for name in self._order:
            b = system.bodies[name]
            r, v = states[name]
            self.sim.add(m=b.mass_kg, x=r[0], y=r[1], z=r[2],
                         vx=v[0], vy=v[1], vz=v[2])
        self.sim.t = 0.0

        # checkpoint archive: (times list, snapshot list); epoch always first
        self._cp_times: list[float] = [0.0]
        self._cp_snaps: list[tuple] = [self._snapshot()]

    # ----------------------------------------------------------- construction
    @staticmethod
    def _particle_order(system: System) -> list[str]:
        """WHFast needs Jacobi ordering: hierarchy level, then mass desc.
        Massless bodies end up last within their level automatically."""
        level = {}

        def depth(name):
            if name not in level:
                b = system.bodies[name]
                level[name] = 0 if b.parent is None else depth(b.parent) + 1
            return level[name]

        for name in system.bodies:
            depth(name)
        return sorted(system.bodies, key=lambda n: (level[n], -system.bodies[n].mass_kg))

    def _initial_states(self, system: System) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Elements at epoch -> absolute Cartesian state (kepler.py's job)."""
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in self._order:
            b = system.bodies[name]
            if b.parent is None:
                out[name] = (np.zeros(3), np.zeros(3))
            else:
                rp, vp = out[b.parent]
                # exact two-body relative motion: mu = G * (M + m), so that
                # osculating elements round-trip at the epoch. (kepler.py's
                # parent-only mu remains the display/Kepler-layer convention.)
                mu = G * (system.bodies[b.parent].mass_kg + b.mass_kg)
                rl, vl = elements_to_state(b.elements, mu, 0.0)
                out[name] = (rp + rl, vp + vl)
        return out

    def _default_dt(self, system: System) -> float:
        periods = []
        for b in system.bodies.values():
            if b.parent is not None and b.elements is not None:
                mu = gm(system.bodies[b.parent].mass_kg)
                if mu > 0:
                    periods.append(2.0 * np.pi * np.sqrt(b.elements.a_m**3 / mu))
        if not periods:
            return DAY
        return min(periods) / 40.0

    def _fresh_sim(self) -> rebound.Simulation:
        sim = rebound.Simulation()
        sim.G = G
        if self.integrator == "whfast":
            # rebound 5.x: "whc" = WHFast + 17th-order corrector; integrator
            # params live on sim.integrator (field-descriptor struct)
            sim.integrator = "whc"
        else:
            sim.integrator = self.integrator    # "ias15" (default epsilon 1e-9)
        sim.dt = self.dt_s
        return sim

    # ------------------------------------------------------------- checkpoints
    def _snapshot(self) -> tuple:
        ps = self.sim.particles
        return (np.array([p.m for p in ps]),
                np.array([p.xyz for p in ps]),
                np.array([p.vxyz for p in ps]))

    def _restore(self, snap: tuple, t_s: float) -> rebound.Simulation:
        sim = self._fresh_sim()
        masses, pos, vel = snap
        for m, r, v in zip(masses, pos, vel):
            sim.add(m=m, x=r[0], y=r[1], z=r[2], vx=v[0], vy=v[1], vz=v[2])
        sim.t = float(t_s)
        return sim

    # --------------------------------------------------------------- querying
    @property
    def t_s(self) -> float:
        return float(self.sim.t)

    def live_state(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        ps = self.sim.particles
        return {name: (np.array(ps[i].xyz), np.array(ps[i].vxyz))
                for name, i in self._index.items()}

    def state_at(self, t_s: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Non-destructive: restore nearest checkpoint, integrate the rest."""
        t_s = float(t_s)
        if t_s == self.sim.t:
            return self.live_state()
        cp = self._cp_times
        if t_s >= 0:
            i = bisect_right(cp, t_s) - 1
        else:
            i = 0 if cp[0] >= t_s else (bisect_right(cp, t_s) - 1)
        sim = self._restore(self._cp_snaps[i], cp[i])
        direction = 1.0 if t_s >= cp[i] else -1.0
        sim.dt = self.dt_s * direction
        sim.integrate(t_s, exact_finish_time=1)
        ps = sim.particles
        return {name: (np.array(ps[j].xyz), np.array(ps[j].vxyz))
                for name, j in self._index.items()}

    # ------------------------------------------------------------- integration
    def integrate_to(self, t_s: float, record_checkpoints: bool = True) -> None:
        """Advance (or rewind) the live simulation, archiving checkpoints."""
        t_s = float(t_s)
        direction = 1.0 if t_s >= self.sim.t else -1.0
        self.sim.dt = self.dt_s * direction
        while (t_s - self.sim.t) * direction > 0.5 * self.dt_s:
            target = t_s
            if record_checkpoints:
                if direction > 0:
                    nxt = self._cp_times[-1] + self.checkpoint_every_s
                    if nxt < t_s:
                        target = nxt
                else:
                    prv = self._cp_times[0] - self.checkpoint_every_s
                    if prv > t_s:
                        target = prv
            self.sim.integrate(target, exact_finish_time=1)
            if record_checkpoints and target != t_s:
                snap = self._snapshot()
                if direction > 0:
                    self._cp_times.append(float(self.sim.t))
                    self._cp_snaps.append(snap)
                else:
                    self._cp_times.insert(0, float(self.sim.t))
                    self._cp_snaps.insert(0, snap)

    # --------------------------------------------------------- physics hooks
    def update_masses(self, mass_kg: dict[str, float]) -> None:
        """Adiabatic mass change (stellar wind from MIST): positions and
        velocities untouched; a slow M(t) inflates orbits via dynamics alone."""
        for name, m in mass_kg.items():
            self.sim.particles[self._index[name]].m = float(m)
        if self.integrator == "whfast":
            try:
                self.sim.integrator.recalculate_coordinates_this_timestep = 1
            except Exception:
                pass   # field name differs across rebound versions

    def osculating_elements(self, body_name: str) -> OsculatingElements:
        b = self.system.bodies[body_name]
        if b.parent is None:
            raise ValueError("root body has no osculating elements")
        i, j = self._index[body_name], self._index[b.parent]
        ps = self.sim.particles
        r = np.array(ps[i].xyz) - np.array(ps[j].xyz)
        v = np.array(ps[i].vxyz) - np.array(ps[j].vxyz)
        mu = G * (ps[i].m + ps[j].m)
        return rv_to_elements(r, v, mu)

    def illumination(self, body_name: str, luminosity_w: dict[str, float] | None = None
                     ) -> dict[str, float]:
        """Same semantics as System.illumination, from the live state."""
        states = self.live_state()
        pos = states[body_name][0]
        out = {}
        for s in self.system.stars():
            lum = (luminosity_w or {}).get(s.name, s.luminosity_w)
            d = np.linalg.norm(pos - states[s.name][0])
            out[s.name] = lum / (4.0 * np.pi * d * d)
        out["total"] = float(sum(out.values()))
        return out
