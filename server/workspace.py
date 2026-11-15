"""Workspace: the live N-body integration behind the API (ADR-0015).

Holds the System catalog and its NBodySimulation. Every state query
advances (or rewinds) the integrator to the requested time; stellar
evolution (MIST) is applied adiabatically at a fixed cadence while
integrating forward, so star masses/luminosities - and therefore orbits -
genuinely evolve along the timeline.
"""
from __future__ import annotations

import threading
import numpy as np

from o2xygen.constants import DAY, M_SUN
from o2xygen.nbody import NBodySimulation
from o2xygen.system import System

EVOLVE_CADENCE_S = 1.0 * 365.25 * DAY   # stellar evolution update cadence
SECONDS_PER_YEAR = 365.25 * DAY


class Workspace:
    def __init__(self, system: System, grid_getter=None,
                 integrator: str = "ias15"):
        self.system = system
        self._grid_getter = grid_getter
        self.lock = threading.Lock()
        self.sim = NBodySimulation(system, integrator=integrator)
        # epoch consistency: catalog star masses are INITIAL masses; the
        # displayed system lives at age0, so stars must sit on their MIST
        # tracks (current mass) before the first query answers. The grid may
        # still be loading at startup -> settle lazily on first contact.
        self._settled = False
        try:
            self._settled = self._apply_evolution_at(0.0)
        except Exception:
            pass

    # ------------------------------------------------------ internal helpers
    def _star_props_at(self, t_s: float) -> dict[str, dict]:
        """MIST (mass, luminosity) for every evolved star at sim time t_s."""
        grid = self._grid_getter() if self._grid_getter else None
        if grid is None:
            return {}
        out = {}
        for s in self.system.stars():
            mini, feh = s.extra.get("mini_msun"), s.extra.get("feh")
            age0 = s.extra.get("age0_gyr")
            if mini is None or age0 is None:
                continue
            age_yr = age0 * 1e9 + t_s / SECONDS_PER_YEAR
            try:
                p = grid.evaluate(mini, feh or 0.0, age_yr)
            except Exception:
                continue
            out[s.name] = {"mass_kg": p.mass_msun * M_SUN,
                           "luminosity_w": p.luminosity_w}
        return out

    def _apply_evolution_at(self, t_s: float) -> bool:
        """Apply MIST track masses/luminosities at sim time t_s.
        Returns True if the grid was available (settled)."""
        props = self._star_props_at(t_s)
        if not props:
            return False
        self.sim.update_masses({n: v["mass_kg"] for n, v in props.items()})
        for n, v in props.items():
            self.system.bodies[n].luminosity_w = v["luminosity_w"]
            self.system.bodies[n].mass_kg = v["mass_kg"]
        return True

    def _evolve_between(self, t_from: float, t_to: float) -> None:
        """Apply adiabatic stellar evolution on cadence stops in (t_from, t_to]."""
        if t_to <= t_from:
            return
        cad = EVOLVE_CADENCE_S
        tau = np.ceil(t_from / cad) * cad
        while tau <= t_to + 0.5:
            self.sim.integrate_to(float(min(tau, t_to)))
            self._apply_evolution_at(float(min(tau, t_to)))
            tau += cad

    # -------------------------------------------------------------- querying
    def state_at(self, t_days: float, with_trails: bool = True) -> dict:
        """Integrate to t_days; return positions, velocities, masses,
        luminosities and osculating elements. This is the ONLY state truth.
        Trails (the smoke) are the expensive part - internal pipeline callers
        (flux sampling) pass with_trails=False."""
        with self.lock:
            t_s = float(t_days) * DAY
            if not self._settled:
                self._settled = self._apply_evolution_at(self.sim.t_s)
            self._evolve_between(self.sim.t_s, t_s)
            self.sim.integrate_to(t_s)
            states = self.sim.live_state()
            ps = self.sim.sim.particles
            idx = self.sim._index
            bodies = {}
            for name, i in idx.items():
                r, v = states[name]
                bodies[name] = {
                    "p": [float(x) for x in r],
                    "v": [float(x) for x in v],
                    "m": float(ps[i].m),
                }
            elements = {}
            for name, b in self.system.bodies.items():
                if b.parent is None:
                    continue
                try:
                    el = self.sim.osculating_elements(name)
                except Exception:
                    continue
                from o2xygen.constants import G
                mu = G * (ps[idx[name]].m + ps[idx[b.parent]].m)
                T = 2 * np.pi * np.sqrt(el.a_m**3 / mu)
                elements[name] = {
                    "a_m": float(el.a_m), "e": float(el.e),
                    "inc": float(el.inc), "raan": float(el.raan),
                    "argp": float(el.argp), "nu": float(el.nu),
                    "period_days": float(T / DAY),
                }
            return {
                "t_days": float(t_days),
                "dt_s": float(self.sim.dt_s),
                "bodies": bodies,
                "elements": elements,
                "trails": self._trails(states, elements, t_s) if with_trails else {},
                "luminosity_w": {s.name: float(s.luminosity_w)
                                 for s in self.system.stars()},
            }

    # -------------------------------------------------------------- trails
    def _trails(self, states, elements, t_s: float) -> dict:
        """Actual integrated paths over a trailing window (the "smoke").

        Per body the window is a fraction of its osculating period, clamped
        to [20 d, 400 d]. Sampled by ONE backward pass on a restored copy of
        the live state - the integrator itself is never disturbed. Scrub
        backwards and the smoke correctly trails into negative time.
        """
        names = [n for n, el in elements.items()
                 if self.system.bodies[n].parent is not None]
        if not names:
            return {}
        # windows in SECONDS (t_s is seconds - do not mix units!)
        windows = {n: float(np.clip(0.4 * elements[n]["period_days"], 20.0, 400.0)) * DAY
                   for n in names}
        K = 90                                     # samples per body window
        step = min(windows.values()) / K           # shared sample grid
        t_end = t_s - max(windows.values())

        sim = self.sim._restore(self.sim._snapshot(), t_s)
        sim.dt = -abs(self.sim.dt_s)
        idx = self.sim._index

        per_body_n = {n: int(round(windows[n] / step)) + 1 for n in names}
        pts = {n: [[float(x) for x in states[n][0]]] for n in names}
        counts = {n: 1 for n in names}

        t = t_s
        while t > t_end and any(counts[n] < per_body_n[n] for n in names):
            t = t - step
            sim.integrate(t, exact_finish_time=1)
            ps = sim.particles
            for n in names:
                if counts[n] < per_body_n[n]:
                    # sample only while inside this body's window
                    if t >= t_s - windows[n] - step * 0.5:
                        pts[n].append([float(x) for x in ps[idx[n]].xyz])
                        counts[n] += 1
        # decimate to <= ~90 points per body (the shared grid over-samples
        # long-window bodies when one short-period body sets the step)
        out = {}
        for n in names:
            p = pts[n]
            stride = max(1, len(p) // 90)
            out[n] = {"window_days": windows[n] / DAY,
                      "points": p[::stride] if len(p) > 90 else p}
        return out

    def flux_at(self, body_name: str, t_days: float) -> dict:
        """Per-star irradiation at body_name at t_days, from the live state."""
        snap = self.state_at(t_days, with_trails=False)
        p = snap["bodies"][body_name]["p"]
        out = {}
        for s in self.system.stars():
            sp = snap["bodies"][s.name]["p"]
            d = float(np.linalg.norm(np.array(p) - np.array(sp)))
            out[s.name] = snap["luminosity_w"][s.name] / (4 * np.pi * d * d)
        out["total"] = float(sum(out.values()))
        return {"illumination": out, "snapshot": snap}
