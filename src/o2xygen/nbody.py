"""Real-time N-body propagation (ADR-0015), over flat worlds (ADR-0016).

The propagation truth: the world state (Cartesian, SI) integrated live with
rebound. `kepler.py` is retained as a PLACEMENT tool (elements -> state
vectors at world-build time) and two-body calibration truth.

Key properties:
  * deterministic: fixed timestep, checkpointed; scrubbing backward and
    forward reproduces the same state (time-symmetric queries);
  * stellar evolution coupling: masses updated between steps (adiabatic) so
    long timelines show genuine orbital inflation; an aggregate escaping-wind
    ledger closes M/P/L/Q for bodies + wind without adding wind particles;
  * osculating elements: derived outputs of the live state RELATIVE TO THE
    WORLD'S ORIGIN BODY (no parents exist anymore; see ADR-0016).
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import numpy as np
import rebound

from .constants import G, DAY
from .system import System
from .wind import WindLedger


class IntegrateBudgetExceeded(RuntimeError):
    """A requested time leap exceeds the per-call integration step budget."""


MAX_INTEGRATE_STEPS = 20_000_000      # ~1-2 min wall at IAS15 speeds
MAX_CHECKPOINTS = 4_000               # archive thins beyond this (keeps epoch)


def _thin_checkpoints(times: list, snaps: list):
    """Halve archive density, always keeping the epoch snapshot. Bounds
    memory of long sessions (unthinned growth measured at ~5000/s during
    high-speed evolution)."""
    keep = [0] + list(range(2, len(times), 2))
    if keep[-1] != len(times) - 1:
        keep.append(len(times) - 1)
    return [times[i] for i in keep], [snaps[i] for i in keep]


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
    # near-parabolic test relative to the local potential scale mu/r (an
    # absolute threshold misclassifies legitimate very wide orbits)
    if abs(energy) < 1e-12 * mu / r:
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


def update_sim_masses(reb_sim: rebound.Simulation, index: dict[str, int],
                      mass_kg: dict[str, float], wind: WindLedger | None,
                      record_wind: bool = True) -> None:
    """One mass-event implementation for live and restored simulations.

    Physical evolution records the removed mass as escaping wind BEFORE
    changing particle masses.  Epoch initialization and manual editing pass
    ``record_wind=False`` because neither is a physical ejection event.
    Positions and velocities are deliberately untouched (adiabatic wind).
    """
    names = list(mass_kg)
    if record_wind:
        if wind is None:
            raise ValueError("physical wind updates require a wind ledger")
        old, pos, vel, new = [], [], [], []
        for name in names:
            p = reb_sim.particles[index[name]]
            old.append(float(p.m))
            pos.append(np.array(p.xyz))
            vel.append(np.array(p.vxyz))
            new.append(float(mass_kg[name]))
        wind.record(float(reb_sim.t), old, pos, vel, new)
    for name in names:
        reb_sim.particles[index[name]].m = float(mass_kg[name])
    if getattr(reb_sim, "integrator", "") in ("whfast", "whc"):
        # rebound 5.x: refresh Jacobi coordinates after a mass change.
        try:
            reb_sim.ri_whfast.recalculate_coordinates_this_timestep = 1
        except Exception:
            pass   # field layout differs across rebound versions


class NBodySimulation:
    """Integrates a System live. Server-side truth for all state queries.

    integrator: "ias15" (default; adaptive, excellent time-reversibility
    for timeline scrubbing, clean mass changes) or "whfast" (symplectic,
    kept for long-haul studies, see ADR-0015).
    """

    def __init__(self, system: System, integrator: str = "ias15",
                 dt_s: float | None = None,
                 checkpoint_every_s: float = 50.0 * DAY):
        self.system = system
        self.integrator = integrator
        self._order = self._particle_order(system)
        self._index = {name: i for i, name in enumerate(self._order)}

        states = system.initial_states()
        if dt_s is None:
            dt_s = self._default_dt(system, states)
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
        self.wind = system.wind.at(0.0)

        # checkpoint archive: (times list, snapshot list); epoch always first.
        # Each snapshot carries its same-time wind ledger, so restoring a past
        # query never applies the current aggregate wind to an earlier state.
        self._cp_times: list[float] = [0.0]
        self._cp_snaps: list[tuple] = [self._snapshot()]

    # ----------------------------------------------------------- construction
    @staticmethod
    def _particle_order(system: System) -> list[str]:
        """Mass-descending order (stars first). The old hierarchy-based
        Jacobi ordering is gone with the hierarchy itself (ADR-0016)."""
        return sorted(system.bodies, key=lambda n: -system.bodies[n].mass_kg)

    def _default_dt(self, system: System,
                    states: dict) -> float:
        """Smallest pairwise dynamical timescale / 40, estimated from
        initial separations: t_dyn ~ sqrt(d^3 / (G (m1+m2)))."""
        names = list(states)
        best = None
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                d = np.linalg.norm(states[names[i]][0] - states[names[j]][0])
                if d <= 0:
                    continue
                m = system.bodies[names[i]].mass_kg + system.bodies[names[j]].mass_kg
                if m <= 0:
                    continue
                t = np.sqrt(d**3 / (G * m))
                best = t if best is None else min(best, t)
        if best is None or not np.isfinite(best):
            return DAY
        return max(best / 40.0, 1.0)

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
                np.array([p.vxyz for p in ps]),
                self.wind.at(float(self.sim.t)))

    def _restore(self, snap: tuple, t_s: float) -> rebound.Simulation:
        sim = self._fresh_sim()
        masses, pos, vel = snap[:3]       # raw copies carry wind separately
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
        # same hang protection as integrate_to, measured from the checkpoint
        # we would actually restore from
        need = int(abs(t_s - cp[i]) / self.dt_s) + 1
        if need > MAX_INTEGRATE_STEPS:
            raise IntegrateBudgetExceeded(
                f"query at {(t_s - cp[i])/DAY:.3g} d from the nearest "
                f"checkpoint needs ~{need} steps > budget "
                f"{MAX_INTEGRATE_STEPS} (dt={self.dt_s/DAY:.3g} d): "
                "shrink the span or raise dt")
        sim = self._restore(self._cp_snaps[i], cp[i])
        direction = 1.0 if t_s >= cp[i] else -1.0
        sim.dt = self.dt_s * direction
        sim.integrate(t_s, exact_finish_time=1)
        ps = sim.particles
        return {name: (np.array(ps[j].xyz), np.array(ps[j].vxyz))
                for name, j in self._index.items()}

    # ------------------------------------------------------------- integration
    def integrate_to(self, t_s: float, record_checkpoints: bool = True) -> None:
        """Advance (or rewind) the live simulation, archiving checkpoints.
        Raises IntegrateBudgetExceeded if the leap needs more than
        MAX_INTEGRATE_STEPS fixed steps - unbounded leaps would hang the
        shared workspace lock (a user-typed span of 1e6 yr at dt=0.17 d is
        billions of steps)."""
        t_s = float(t_s)
        direction = 1.0 if t_s >= self.sim.t else -1.0
        # integration is now monotone in the requested direction (checkpoint
        # targets are anchored to sim.t, never to the archive - see below),
        # so this step count bounds the actual work done
        need = int(abs(t_s - self.sim.t) / self.dt_s) + 1
        if need > MAX_INTEGRATE_STEPS:
            raise IntegrateBudgetExceeded(
                f"leap of {(t_s - self.sim.t)/DAY:.3g} d needs ~{need} steps "
                f"> budget {MAX_INTEGRATE_STEPS} (dt={self.dt_s/DAY:.3g} d): "
                "shrink the span or raise dt")
        self.sim.dt = self.dt_s * direction
        cad = self.checkpoint_every_s
        while (t_s - self.sim.t) * direction > 0.5 * self.dt_s:
            target = t_s
            if record_checkpoints:
                # next boundary strictly AHEAD of the current time in the
                # integration direction. Anchoring to sim.t (not to the
                # archive ends) guarantees we never hop backwards to a stale
                # checkpoint after an intervening record_checkpoints=False leap
                if direction > 0:
                    nxt = np.floor(self.sim.t / cad) * cad + cad
                    if nxt <= self.sim.t:          # ulp undershoot guard
                        nxt += cad
                    if nxt < t_s:
                        target = nxt
                else:
                    prv = np.ceil(self.sim.t / cad) * cad - cad
                    if prv >= self.sim.t:          # ulp overshoot guard
                        prv -= cad
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
                if len(self._cp_times) > MAX_CHECKPOINTS:
                    self._cp_times, self._cp_snaps = _thin_checkpoints(
                        self._cp_times, self._cp_snaps)
        # final polish: the loop may exit up to 0.5 dt short of t_s - land
        # exactly on the requested time
        if self.sim.t != t_s:
            self.sim.integrate(t_s, exact_finish_time=1)

    # --------------------------------------------------------- physics hooks
    def update_masses(self, mass_kg: dict[str, float],
                      record_wind: bool = True) -> None:
        """Adiabatic mass change with closed-system wind accounting.

        Positions and velocities stay untouched, so a slow M(t) still inflates
        orbits via dynamics alone.  The remaining bodies are open; their lost
        mass, momentum and moments are retained in ``self.wind``.
        """
        update_sim_masses(self.sim, self._index, mass_kg, self.wind,
                          record_wind=record_wind)

    def reset_conservation_epoch(self, recenter: bool = True) -> None:
        """Establish a new manual-edit baseline without inventing wind.

        A uniform translation and Galilean boost preserve every relative
        position/velocity while setting Q=P=0.  The wind ledger is cleared
        because an arbitrary user mass edit is a new world, not an ejection
        history.  Checkpoints must be rebuilt after the coordinate change.
        """
        ps = self.sim.particles
        if len(ps) == 0:
            self.wind.clear(float(self.sim.t))
            return
        ms = np.array([p.m for p in ps])
        pos = np.array([p.xyz for p in ps])
        vel = np.array([p.vxyz for p in ps])
        m_tot = float(ms.sum())
        if not np.isfinite(m_tot) or m_tot <= 0.0:
            raise ValueError("conservation epoch needs positive total mass")
        vcm = (ms[:, None] * vel).sum(axis=0) / m_tot
        rcm = (ms[:, None] * pos).sum(axis=0) / m_tot
        for p in ps:
            p.vx, p.vy, p.vz = [float(x) for x in
                                (np.array(p.vxyz) - vcm)]
            if recenter:
                p.x, p.y, p.z = [float(x) for x in
                                 (np.array(p.xyz) - rcm)]
        self.wind.clear(float(self.sim.t))
        self._cp_times = [float(self.sim.t)]
        self._cp_snaps = [self._snapshot()]

    def osculating_elements(self, body_name: str,
                            ref: str | None = None) -> OsculatingElements:
        """Elements of body_name relative to an explicit `ref`, or to the
        rest-of-system barycenter when no ref/origin exists (ADR-0022).
        With no hierarchy this is a VIEW, not an identity: a moon's elements
        w.r.t. the system look like a perturbed planet's."""
        i = self._index[body_name]
        ps = self.sim.particles
        ref = ref or self.system.origin
        if ref is None:
            names = list(self._index)
            ms = np.array([ps[self._index[n]].m for n in names])
            pos = np.array([ps[self._index[n]].xyz for n in names])
            vel = np.array([ps[self._index[n]].vxyz for n in names])
            m_tot = float(ms.sum())
            m_rest = m_tot - ms[names.index(body_name)]
            if m_rest <= 0.0:
                raise ValueError("osculating elements need a nonzero rest mass")
            r_rest = ((ms[:, None] * pos).sum(axis=0)
                      - ms[names.index(body_name)] * pos[names.index(body_name)]) / m_rest
            v_rest = ((ms[:, None] * vel).sum(axis=0)
                      - ms[names.index(body_name)] * vel[names.index(body_name)]) / m_rest
            r = pos[names.index(body_name)] - r_rest
            v = vel[names.index(body_name)] - v_rest
            return rv_to_elements(r, v, G * m_tot)
        if ref == body_name:
            raise ValueError("osculating elements need a distinct reference body")
        j = self._index[ref]
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
            if d <= 0.0:
                raise ValueError(
                    f"irradiance of '{body_name}' undefined: coincident with "
                    f"star '{s.name}' (self-irradiation is excluded)")
            out[s.name] = lum / (4.0 * np.pi * d * d)
        out["total"] = float(sum(out.values()))
        return out
