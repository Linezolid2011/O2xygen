"""Workspace: the live N-body integration behind the API (ADR-0015).

Holds the System catalog and its NBodySimulation. ALL queries are
non-destructive (ADR-0018): a query for a time other than "now" runs on a
copy restored from the nearest checkpoint and never moves the live clock.
The live clock moves only through the evolution job (or advance_to, the
explicit destructive mutator used by tests/tools). Stellar evolution (MIST)
is applied adiabatically at a fixed cadence while integrating forward, so
star masses/luminosities - and therefore orbits - genuinely evolve along
the timeline.
"""
from __future__ import annotations

from bisect import bisect_right
import threading
import time
import numpy as np

from o2xygen.constants import DAY, G, M_SUN
from o2xygen.nbody import (NBodySimulation, IntegrateBudgetExceeded,
                           MAX_INTEGRATE_STEPS, rv_to_elements)
from o2xygen.system import System, Body, STAR

EVOLVE_CADENCE_S = 1.0 * 365.25 * DAY   # stellar evolution update cadence
SECONDS_PER_YEAR = 365.25 * DAY
TRAIL_WALL_SPAN_S = 4.0                 # smoke covers ~this many WALL seconds
                                        # of recent history at speed (ADR-0021)
# Constant-time-lag tide defaults (ADR-0023). Per-body extra overrides:
# love_k2, tidal_dt_s, inertia_factor (I = inertia_factor * M * R^2).
DEFAULT_LOVE_K2 = {STAR: 0.02, "body": 0.30}
DEFAULT_TIDAL_LAG_S = 100.0
# n=1 polytrope: I/(MR^2) = 2/3 - 4/pi^2 ~= 0.261
DEFAULT_INERTIA_FACTOR = 2.0 / 3.0 - 4.0 / np.pi**2


class Workspace:
    def __init__(self, system: System, grid_getter=None,
                 integrator: str = "ias15", wd_grid_getter=None):
        self.system = system
        self._grid_getter = grid_getter
        self._wd_grid_getter = wd_grid_getter
        self.lock = threading.Lock()
        self.sim = NBodySimulation(system, integrator=integrator)
        # epoch consistency: catalog star masses are INITIAL masses; the
        # displayed system lives at age0, so stars must sit on their MIST
        # tracks (current mass) before the first query answers. The grid may
        # still be loading at startup -> settle lazily on first contact.
        self._settled = False
        self._trail_cache = None            # (t_s, min_window_s, result, windows)
        self._rate_hist = []                # (wall_s, t_s) for clock-rate estimate
        self._spin_epoch_s = 0.0            # catalog spin phases valid at t=0
        try:
            self._settled = self._apply_evolution_at(0.0)
        except Exception:
            pass

    # ------------------------------------------------------ internal helpers
    def _star_props_at(self, t_s: float) -> dict[str, dict]:
        """Stellar properties at sim time t_s, computed from MIST inputs.
        Off-grid inputs are approximated at the nearest covered grid point;
        no static/per-star output copy is stored or consulted."""
        out = {}
        grid = self._grid_getter() if self._grid_getter else None
        for s in self.system.stars():
            # ADR-0016 schema: the stored mass IS the initial mass (MIST seed)
            mini = s.extra.get("mini_msun", s.mass_kg / M_SUN)
            feh, age0 = s.extra.get("feh"), s.extra.get("age0_gyr")
            if age0 is None:
                continue
            age_yr = age0 * 1e9 + t_s / SECONDS_PER_YEAR
            try:
                if s.extra.get("evolution_model") == "mist3_wd":
                    wd = self._wd_grid_getter() if self._wd_grid_getter else None
                    if wd is None:
                        continue
                    p = wd.evaluate(mini, feh or 0.0, age_yr)
                else:
                    if grid is None:
                        continue
                    p = grid.evaluate(mini, feh or 0.0, age_yr)
            except Exception:
                continue
            out[s.name] = {"mass_kg": p.mass_msun * M_SUN,
                           "luminosity_w": p.luminosity_w,
                           "radius_m": p.radius_m,
                           "teff_k": p.teff_k}
        return out

    def _apply_star_props(self, props: dict) -> None:
        """Write MIST properties onto the catalog bodies (display + save)."""
        for n, v in props.items():
            b = self.system.bodies[n]
            b.luminosity_w = v["luminosity_w"]
            b.mass_kg = v["mass_kg"]
            b.radius_m = v["radius_m"]            # star visibly swells near turnoff
            b.extra["teff_k"] = v["teff_k"]       # and reddens

    def _apply_evolution_at(self, t_s: float) -> bool:
        """Apply MIST track masses/luminosities at sim time t_s.
        Returns True if the grid was available (settled)."""
        props = self._star_props_at(t_s)
        if not props:
            return False
        self.sim.update_masses({n: v["mass_kg"] for n, v in props.items()})
        self._apply_star_props(props)
        return True

    # ---------------------------------------------------------- spin & tides
    def _spin_map(self, t_s: float | None = None) -> dict[str, dict]:
        """Spin vectors/phases for every body. Between tidal cadence stops
        the rate/axis are constant, so the phase is read at t_s by linear
        phase advance (display-only; no orbital extrapolation)."""
        if t_s is None:
            t_s = self.sim.t_s
        dt = max(0.0, t_s - self._spin_epoch_s)
        return {n: {"w": b.spin_vector(),
                    "phase": (b.spin_phase_rad + b.spin_rate_rad_s * dt)
                             % (2.0 * np.pi)}
                for n, b in self.system.bodies.items()}

    def _write_live_spins(self, spins: dict, t_s: float) -> None:
        for n, sp in spins.items():
            self.system.bodies[n].set_spin_vector(sp["w"], sp["phase"])
        self._spin_epoch_s = t_s

    def _apply_tides(self, sim, t_from: float, t_to: float,
                     spins: dict) -> None:
        """Constant-time-lag equilibrium tides (ADR-0023), applied as a
        secular operator split at evolution cadence stops.

        For host i and point-mass perturber j, with r from i to j:

            w      = (r_hat x v_rel) / r       (orbital angular velocity)
            C      = 3 G m_j^2 k2_i R_i^5 / r^6
            dOmega = C * lag_i / I_i * (w - Omega_perp) * dt

        The perpendicular spin component relaxes EXPONENTIALLY toward w
        (exact at fixed geometry, stable for arbitrary dt). The spin angular
        momentum change is applied with the opposite sign to the pair's
        orbital angular momentum via a tangential velocity kick, so total
        spin + orbital angular momentum is conserved and total linear
        momentum is unchanged. No per-step Python callback, no Roche/rupture
        model (explicitly out of scope)."""
        dt = t_to - t_from
        if dt <= 0.0:
            return
        is_live = sim is self.sim
        reb = sim.sim if is_live else sim
        ps = reb.particles
        idx = self.sim._index
        names = list(idx)
        kicked = False
        for host in names:
            b = self.system.bodies[host]
            sp = spins.get(host)
            if sp is None:
                sp = spins[host] = {"w": b.spin_vector(),
                                    "phase": b.spin_phase_rad}
            omega0 = sp["w"].copy()
            k2 = float(b.extra.get("love_k2", DEFAULT_LOVE_K2[b.kind]))
            lag = float(b.extra.get("tidal_dt_s", DEFAULT_TIDAL_LAG_S))
            inertia_factor = float(b.extra.get("inertia_factor",
                                               DEFAULT_INERTIA_FACTOR))
            i = idx[host]
            m_i = float(ps[i].m)
            R = float(b.radius_m)
            I = inertia_factor * m_i * R * R
            if m_i > 0.0 and R > 0.0 and I > 0.0 and k2 > 0.0 and lag > 0.0:
                for other in names:
                    if other == host:
                        continue
                    j = idx[other]
                    m_j = float(ps[j].m)
                    if m_j <= 0.0:
                        continue
                    rvec = np.array(ps[j].xyz) - np.array(ps[i].xyz)
                    r2 = float(np.dot(rvec, rvec))
                    if r2 <= 1e-12:
                        continue
                    r = float(np.sqrt(r2))
                    vrel = np.array(ps[j].vxyz) - np.array(ps[i].vxyz)
                    rhat = rvec / r
                    w = np.cross(rhat, vrel) / r
                    omega = sp["w"]
                    omega_par = rhat * float(np.dot(omega, rhat))
                    omega_perp = omega - omega_par
                    C = 3.0 * G * m_j * m_j * k2 * R**5 / r**6
                    decay = float(np.exp(-min(700.0, C * lag * dt / I)))
                    sp["w"] = omega_par + w + (omega_perp - w) * decay
                    # spin angular momentum lost/gained -> opposite orbital
                    # angular momentum kick about the pair barycenter
                    dL = -I * (sp["w"] - omega)
                    reduced = m_i * m_j / (m_i + m_j)
                    dv = np.cross(dL / reduced, rvec) / r2
                    pi, pj = ps[i], ps[j]
                    pi.vx -= m_j / (m_i + m_j) * dv[0]
                    pi.vy -= m_j / (m_i + m_j) * dv[1]
                    pi.vz -= m_j / (m_i + m_j) * dv[2]
                    pj.vx += m_i / (m_i + m_j) * dv[0]
                    pj.vy += m_i / (m_i + m_j) * dv[1]
                    pj.vz += m_i / (m_i + m_j) * dv[2]
                    kicked = True
            rate0 = float(np.linalg.norm(omega0))
            rate1 = float(np.linalg.norm(sp["w"]))
            sp["phase"] = (sp["phase"] + 0.5 * (rate0 + rate1) * dt) \
                % (2.0 * np.pi)
        if kicked and getattr(reb, "integrator", "") in ("whfast", "whc"):
            try:
                reb.ri_whfast.recalculate_coordinates_this_timestep = 1
            except Exception:
                pass
        if is_live:
            self._write_live_spins(spins, t_to)

    def _evolve_between(self, t_from: float, t_to: float,
                        record: bool = True) -> None:
        """Apply adiabatic stellar evolution on cadence stops in (t_from, t_to].
        Cadence is adaptive: at most ~200 evolution updates per leap, so a
        user-typed million-year span can never degenerate into a million
        one-year integrate calls. `record` propagates to the integrator
        (evolution passes False: irreversible leaps archive no checkpoints)."""
        self._evolve_sim_between(self.sim, t_from, t_to, record=record)

    def _evolve_sim_between(self, sim, t_from: float, t_to: float,
                            record: bool = False, spins: dict | None = None) -> None:
        """Stellar-evolution + tidal cadence applied to an ARBITRARY
        simulation object (the live NBodySimulation, or a raw rebound copy
        used by non-destructive queries). Masses AND the catalog's displayed
        luminosity/radius/Teff/spin are updated on the live path, so a
        running evolution visibly ages and tidally evolves its bodies."""
        if t_to <= t_from:
            return
        is_live = sim is self.sim
        particles = sim.sim.particles if is_live else sim.particles
        idx = self.sim._index                # copies share the live ordering
        if spins is None:
            spins = self._spin_map(t_from)
        span = t_to - t_from
        cad = max(EVOLVE_CADENCE_S, span / 200.0)
        # first cadence stop STRICTLY after t_from, and always include t_to:
        # the old ceil() loop skipped the final endpoint for short leaps, so
        # a 5-day advance applied no spin/tide update at all
        tau = np.floor(t_from / cad) * cad + cad
        prev = t_from
        while True:
            t_stop = float(min(tau, t_to))
            self._integrate(sim, t_stop, record)
            props = self._star_props_at(t_stop)
            if props:
                if is_live:
                    sim.update_masses({k: v["mass_kg"] for k, v in props.items()})
                    self._apply_star_props(props)
                else:
                    for k, v in props.items():
                        particles[idx[k]].m = v["mass_kg"]
            self._apply_tides(sim, prev, t_stop, spins)
            prev = t_stop
            if t_stop >= t_to:
                break
            tau += cad

    @staticmethod
    def _integrate(sim, t_s: float, record: bool):
        """Duck-typed step: NBodySimulation (live) or a raw rebound copy."""
        if hasattr(sim, "integrate_to"):
            sim.integrate_to(t_s, record_checkpoints=record)
        else:
            sim.integrate(t_s, exact_finish_time=1)

    # -------------------------------------------------------------- querying
    def advance_to(self, t_days: float) -> None:
        """DESTRUCTIVE: move the live clock to t_days (tests/tools only -
        in production time moves ONLY through the evolution job, ADR-0018)."""
        with self.lock:
            t_s = float(t_days) * DAY
            need = int(abs(t_s - self.sim.t_s) / self.sim.dt_s) + 1
            if need > MAX_INTEGRATE_STEPS:
                raise IntegrateBudgetExceeded(
                    f"leap of {t_days:.3g} d needs ~{need} steps > budget "
                    f"{MAX_INTEGRATE_STEPS} (dt={self.sim.dt_s/DAY:.3g} d): "
                    "shrink the span or raise dt")
            if not self._settled:
                self._settled = self._apply_evolution_at(self.sim.t_s)
            self._evolve_between(self.sim.t_s, t_s)
            self.sim.integrate_to(t_s)
            self._trail_cache = None

    def _copy_at(self, t_s: float):
        """(raw rebound sim, spin map) at t_s, never touching the live
        integrator. Future queries start from the LIVE snapshot (current
        spin state included); past queries restore the nearest checkpoint.
        Spin is not archived historically, so a past view carries the
        current spin metadata - only the dynamics are rewound."""
        now = self.sim.t_s
        if t_s >= now:
            sim = self.sim._restore(self.sim._snapshot(), now)
            spins = self._spin_map(now)
            self._evolve_sim_between(sim, now, t_s, spins=spins)
            return sim, spins
        cp = self.sim._cp_times
        i = max(0, bisect_right(cp, t_s) - 1)
        need = int(abs(t_s - cp[i]) / self.sim.dt_s) + 1
        if need > MAX_INTEGRATE_STEPS:
            raise IntegrateBudgetExceeded(
                f"query at {(t_s - cp[i])/DAY:.3g} d from the nearest "
                f"checkpoint needs ~{need} steps > budget "
                f"{MAX_INTEGRATE_STEPS} (dt={self.sim.dt_s/DAY:.3g} d): "
                "shrink the span or raise dt")
        sim = self.sim._restore(self.sim._cp_snaps[i], cp[i])
        spins = self._spin_map(now)
        if t_s >= cp[i]:
            self._evolve_sim_between(sim, cp[i], t_s, spins=spins)
        else:                                   # before the epoch: plain rewind
            sim.dt = -abs(self.sim.dt_s)
            sim.integrate(t_s, exact_finish_time=1)
        return sim, spins

    def state_at(self, t_days: float | None, with_trails: bool = True) -> dict:
        """State snapshot at t_days (positions, velocities, masses,
        luminosities, osculating elements, trails). NON-DESTRUCTIVE
        (ADR-0018): t_days=None reads the live sim as-is; any other time is
        answered from a checkpoint-restored copy, so no query - however
        pathological - can move the shared clock (2026-11-15 desync class)."""
        with self.lock:
            if not self._settled:
                self._settled = self._apply_evolution_at(self.sim.t_s)
            if t_days is None or abs(float(t_days) * DAY - self.sim.t_s) \
                    <= 0.5 * self.sim.dt_s:
                t_s = self.sim.t_s
                sim = self.sim
                spins = self._spin_map(t_s)
            else:
                t_s = float(t_days) * DAY
                sim, spins = self._copy_at(t_s)
            return self._snapshot_dict(sim, t_s, with_trails, spins)

    def _snapshot_dict(self, sim, t_s: float, with_trails: bool,
                       spins: dict | None = None) -> dict:
        """Serialise a sim object (live NBodySimulation or raw rebound copy)
        sitting at t_s into the /api/state payload."""
        live = sim is self.sim
        ps = sim.sim.particles if live else sim.particles
        idx = self.sim._index
        bodies = {}
        states = {}
        for name, i in idx.items():
            r = np.array(ps[i].xyz)
            v = np.array(ps[i].vxyz)
            states[name] = (r, v)
            bodies[name] = {"p": [float(x) for x in r],
                            "v": [float(x) for x in v],
                            "m": float(ps[i].m)}
        origin = self.system.origin      # metadata only (ADR-0022)
        # ADR-0022: osculating elements are computed against the barycenter
        # of everything EXCEPT the body itself, with mu = G*M_tot. For a
        # two-body system this is EXACTLY the relative orbit; for a
        # star-dominated system it is indistinguishable from "vs the star"
        # at display precision; for near-equal binaries it is the only
        # non-arbitrary choice. No privileged root body exists.
        elements = {}
        periods = {}                               # body -> period_days | None
        names_order = list(idx.keys())
        ms = np.array([ps[idx[n]].m for n in names_order])
        pos_arr = np.array([ps[idx[n]].xyz for n in names_order])
        vel_arr = np.array([ps[idx[n]].vxyz for n in names_order])
        m_tot = float(ms.sum())
        tot_p = (ms[:, None] * pos_arr).sum(axis=0)
        tot_v = (ms[:, None] * vel_arr).sum(axis=0)
        mu_tot = G * m_tot
        for k, name in enumerate(names_order):
            m_rest = m_tot - ms[k]
            if m_rest <= 0.0:
                continue
            r_rest = (tot_p - ms[k] * pos_arr[k]) / m_rest
            v_rest = (tot_v - ms[k] * vel_arr[k]) / m_rest
            try:
                el = rv_to_elements(pos_arr[k] - r_rest,
                                    vel_arr[k] - v_rest, mu_tot)
            except Exception:
                continue
            if not (np.isfinite(el.a_m) and np.isfinite(el.e)
                    and el.a_m > 0.0 and el.e < 1.0):
                # unbound vs the rest of the system - legitimate for
                # members of a tight inner binary (bound to their
                # companion, not to the system frame). Not displayable as
                # Keplerian elements; and NaN in JSON would break every
                # client-side parse (2026-11-17, Εἴδωλον/Vesper).
                continue
            T = 2 * np.pi * np.sqrt(el.a_m**3 / mu_tot)
            elements[name] = {
                "a_m": float(el.a_m), "e": float(el.e),
                "inc": float(el.inc), "raan": float(el.raan),
                "argp": float(el.argp), "nu": float(el.nu),
                "period_days": float(T / DAY),
            }
            periods[name] = float(T / DAY)
        # display star props: fresh MIST values at t_s where the grid has
        # them (covers copies, whose catalog bodies are NOT updated), else
        # the catalog values the live cadence maintains
        if spins is None:
            spins = self._spin_map(t_s)
        spin_json = {}
        for n, b in self.system.bodies.items():
            sp = spins.get(n, {"w": b.spin_vector(), "phase": b.spin_phase_rad})
            w = np.asarray(sp["w"], dtype=float)
            rate = float(np.linalg.norm(w))
            axis = w / rate if rate > 1e-30 else b.spin_axis
            spin_json[n] = {"axis": [float(x) for x in axis],
                            "rate_rad_s": rate,
                            "phase_rad": float(sp["phase"])}
        props = self._star_props_at(t_s)
        stars = {}
        for s in self.system.stars():
            p = props.get(s.name)
            stars[s.name] = {
                "teff_k": float(p["teff_k"]) if p else float(s.extra.get("teff_k", 5772.0)),
                "radius_m": float(p["radius_m"]) if p else float(s.radius_m),
                "luminosity_w": float(p["luminosity_w"]) if p else float(s.luminosity_w),
            }
        return {
            "t_days": float(t_s / DAY),
            "dt_s": float(self.sim.dt_s),
            "bodies": bodies,
            "elements": elements,
            "stars": stars,
            "spins": spin_json,
            "trails": self._trails(sim, states, periods, t_s) if with_trails else {},
            "luminosity_w": {s.name: stars[s.name]["luminosity_w"]
                             for s in self.system.stars()},
        }

    # -------------------------------------------------------------- trails
    def _clock_rate(self, t_s: float) -> float:
        """Recent sim-clock rate [sim s / wall s], from samples taken at
        trail-query time (the SSE emitter and the poll both drive this).
        0 when the clock is frozen - windows then stay purely physical."""
        now = time.time()
        hist = self._rate_hist
        hist.append((now, t_s))
        while len(hist) > 2 and now - hist[0][0] > 2.0:
            hist.pop(0)
        if len(hist) < 2 or hist[-1][0] - hist[0][0] < 0.2:
            return 0.0
        return abs(t_s - hist[0][1]) / (hist[-1][0] - hist[0][0])

    def _trails(self, sim, states, periods: dict, t_s: float) -> dict:
        """Actual integrated paths over a trailing window (the "smoke").

        Per body the window is max(physics, rate x TRAIL_WALL_SPAN_S): a
        fraction of the osculating period clamped to [20 d, 400 d] at low
        speed, and "the last few WALL seconds" at speed - so the recompute
        gap (interval x speed) is always a small fraction of the window and
        no long straight head-chord can appear (2026-11-17 report). Bodies
        without a bound osculating orbit vs the rest of the system
        (tight-binary members) still get smoke: their physical window falls
        back to the 400 d cap. Sampled by ONE backward pass on a copy - the
        live integrator is never disturbed. Results are cached and recomputed
        once the clock advances by one sample stride (window/90), not a fixed
        wall interval: slow playback gets smooth sub-second updates, not
        multi-second freezes.
        """
        names = list(self.sim._index)      # ADR-0022: EVERY body gets smoke
        if not names:
            return {}
        rate = self._clock_rate(t_s)           # sim seconds per wall second
        # windows in SECONDS (t_s is seconds - do not mix units!). A missing
        # or non-finite period (hyperbolic vs origin) uses the 400 d cap.
        def _phys(n):
            p = periods.get(n)
            if p is None or not np.isfinite(p) or p <= 0:
                return 400.0 * DAY
            return float(np.clip(0.4 * p, 20.0, 400.0)) * DAY
        windows = {n: max(_phys(n), rate * TRAIL_WALL_SPAN_S) for n in names}
        K = 90                                     # samples per body window
        step = min(windows.values()) / K           # shared sample grid
        # work safety valve: never more than ~2000 integrate calls per pass
        # (only bites on pathological window spreads)
        step = max(step, max(windows.values()) / 2000.0)
        t_end = t_s - max(windows.values())

        cache = self._trail_cache
        if cache is not None and set(cache[2]) == set(names) \
                and abs(t_s - cache[0]) < cache[1] / K \
                and all(abs(windows[n] - cache[3][n]) <= 0.1 * cache[3][n]
                        for n in names):
            return cache[2]

        if sim is self.sim:
            sim = self.sim._restore(self.sim._snapshot(), t_s)
        sim.dt = -abs(self.sim.dt_s)
        idx = self.sim._index

        # trails are BARYCENTRIC (ADR-0022): subtract the system barycenter
        # at every sample. Every body gets smoke - a primary star visibly
        # weaving about the common center of mass is the point.
        ps = sim.particles
        m0 = np.array([p.m for p in ps])           # masses at t_s (head)
        m0_tot = float(m0.sum())

        def rel(i):
            arr = np.array([q.xyz for q in ps])
            mm = np.array([q.m for q in ps])
            bcm = (mm[:, None] * arr).sum(axis=0) / mm.sum()
            return [float(x) for x in (np.array(ps[i].xyz) - bcm)]

        per_body_n = {n: int(round(windows[n] / step)) + 1 for n in names}
        pts = {n: [None] for n in names}          # head filled below
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
                        pts[n].append(rel(idx[n]))
                        counts[n] += 1
        # heads at the REQUESTED time t_s (the loop ended at t_end; `states`
        # holds the absolute t_s states) - barycenter from the same masses
        pos0 = np.array([states[n][0] for n in idx])
        o_head = (m0[:, None] * pos0).sum(axis=0) / m0_tot
        for n in names:
            pts[n][0] = [float(x) for x in (np.asarray(states[n][0]) - o_head)]
        # decimate to <= ~90 points per body (the shared grid over-samples
        # long-window bodies when one short-period body sets the step)
        out = {}
        for n in names:
            p = pts[n]
            stride = max(1, len(p) // 90)
            out[n] = {"window_days": windows[n] / DAY,
                      "points": p[::stride] if len(p) > 90 else p}
        self._trail_cache = (t_s, min(windows.values()), out, windows)
        return out

    # -------------------------------------------------------------- saving
    def snapshot_for_save(self) -> System:
        """A System whose initial conditions ARE the current live state:
        pos0/vel0 = live state, stellar age0 advanced by elapsed sim-years.
        Saving THIS makes a reloaded world reproduce the same physical
        system at its new epoch (推演 autosaves and 另存为 both use it).
        Takes the workspace lock: autosaves must never read a half-integrated
        state while an API query holds the integrator mid-leap."""
        from o2xygen.constants import YEAR as _YEAR
        with self.lock:
            states = self.sim.live_state()
            spins = self._spin_map(self.sim.t_s)
            out = System(self.system.name, origin=self.system.origin)
            # the save becomes a new t=0: advance the absolute astronomical
            # epoch too, or a reload would silently reset any future sky catalog
            from o2xygen.astroframe import advance_epoch
            out.astro_frame = advance_epoch(self.system.astro_frame, self.sim.t_s)
            age_shift_gyr = self.sim.t_s / _YEAR / 1e9        # YEARS -> GYRS
            for n, b in self.system.bodies.items():
                extra = dict(b.extra)
                if b.kind == STAR:
                    extra["age0_gyr"] = float(b.extra.get("age0_gyr", 1.0)) + age_shift_gyr
                w = spins[n]["w"]
                rate = float(np.linalg.norm(w))
                axis = w / rate if rate > 1e-30 else b.spin_axis
                out.add(Body(n, b.kind, mass_kg=b.mass_kg,
                             pos0=states[n][0].copy(), vel0=states[n][1].copy(),
                             radius_m=b.radius_m, albedo=b.albedo,
                             luminosity_w=b.luminosity_w,
                             spin_axis=axis.copy(), spin_rate_rad_s=rate,
                             spin_phase_rad=spins[n]["phase"], extra=extra))
            return out

    def flux_series_forward(self, body_name: str, span_days: float, n: int) -> list:
        """NON-DESTRUCTIVE forward flux sampling for the temperature table.

        Runs on a restored copy of the live state: queries must never move
        the shared integrator clock. INCIDENT 2026-11-15: the temperature
        endpoint sampled by advancing the LIVE sim; during an evolution
        this (a) pushed sim.t past the job's end_days so the job reported
        'done' while physics kept moving, and (b) interleaved with the job
        when a new evolution was started - the 'terrible thing'.

        Returns rows of absolute sim days + total flux [W/m^2]."""
        # The integration runs on a private copy, so the workspace lock is
        # needed ONLY to take a consistent snapshot. Holding it for the whole
        # 400-sample integration starved poses/evolution for seconds whenever
        # the user switched between non-star bodies (2026-11-17 report).
        with self.lock:
            t0 = self.sim.t_s
            sim = self.sim._restore(self.sim._snapshot(), t0)
            spins = self._spin_map(t0)
            idx = self.sim._index
            star_names = [s.name for s in self.system.stars()]
            # fallback luminosities when the MIST grid is unavailable
            base_lum = {s.name: s.luminosity_w for s in self.system.stars()}
        rows = []
        prev = t0
        for i in range(n):
            t = t0 + span_days * DAY * i / max(n - 1, 1)
            self._evolve_sim_between(sim, prev, t, spins=spins)
            sim.integrate(t, exact_finish_time=1)   # raw copy: no archive
            prev = t
            ps = sim.particles
            p = np.array(ps[idx[body_name]].xyz)
            props = self._star_props_at(t)
            lum = {k: v["luminosity_w"] for k, v in props.items()} or base_lum
            flux = 0.0
            for sname in star_names:
                sp = np.array(ps[idx[sname]].xyz)
                d = float(np.linalg.norm(p - sp))
                flux += lum.get(sname, base_lum.get(sname, 0.0)) / (4.0 * np.pi * d * d)
            rows.append({"d": t / DAY, "flux": flux})
        return rows

    def flux_at(self, body_name: str, t_days: float | None) -> dict:
        """Per-star irradiation at body_name. t_days=None -> current time.
        A star's own flux is excluded (self-irradiation is meaningless and
        divides by zero)."""
        snap = self.state_at(t_days, with_trails=False)
        p = snap["bodies"][body_name]["p"]
        out = {}
        for s in self.system.stars():
            if s.name == body_name:
                continue                        # no self-irradiation
            sp = snap["bodies"][s.name]["p"]
            d = float(np.linalg.norm(np.array(p) - np.array(sp)))
            out[s.name] = snap["luminosity_w"][s.name] / (4 * np.pi * d * d)
        out["total"] = float(sum(out.values()))
        return {"illumination": out, "snapshot": snap}
