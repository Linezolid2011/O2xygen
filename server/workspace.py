"""Workspace: the live N-body integration behind the API (ADR-0015).

Holds the System catalog and its NBodySimulation. ALL queries are
non-destructive (ADR-0018): a query for a time other than "now" runs on a
copy restored from the nearest checkpoint and never moves the live clock.
The live clock moves only through the evolution job (or advance_to, the
explicit destructive mutator used by tests/tools). Stellar evolution (MIST)
is applied adiabatically at a fixed cadence while integrating forward, so
star masses/luminosities - and therefore orbits - genuinely evolve along
the timeline. Escaping-wind integrals travel with every live/copy/checkpoint
state so bodies + wind form the closed conservation system (ADR-0030).
"""
from __future__ import annotations

from bisect import bisect_right
import threading
import numpy as np

from o2xygen.constants import DAY, G, M_SUN
from o2xygen.nbody import (NBodySimulation, IntegrateBudgetExceeded,
                           MAX_INTEGRATE_STEPS, rv_to_elements,
                           update_sim_masses)
from o2xygen.system import System, Body, STAR

EVOLVE_CADENCE_S = 1.0 * 365.25 * DAY   # stellar evolution update cadence
SECONDS_PER_YEAR = 365.25 * DAY
# Constant-time-lag tide defaults (ADR-0023). Per-body extra overrides:
# love_k2, tidal_dt_s, inertia_factor (I = inertia_factor * M * R^2).
DEFAULT_LOVE_K2 = {STAR: 0.02, "body": 0.30}
DEFAULT_TIDAL_LAG_S = 100.0
# n=1 polytrope: I/(MR^2) = 2/3 - 4/pi^2 ~= 0.261
DEFAULT_INERTIA_FACTOR = 2.0 / 3.0 - 4.0 / np.pi**2


class Workspace:
    def __init__(self, system: System, grid_getter=None,
                 integrator: str = "ias15", wd_grid_getter=None,
                 cool_grid_getter=None):
        self.system = system
        self._grid_getter = grid_getter
        self._wd_grid_getter = wd_grid_getter
        self._cool_grid_getter = cool_grid_getter
        self.lock = threading.Lock()
        self.sim = NBodySimulation(system, integrator=integrator)
        # epoch consistency: catalog star masses are INITIAL masses; the
        # displayed system lives at age0, so stars must sit on their MIST
        # tracks (current mass) before the first query answers. The grid may
        # still be loading at startup -> settle lazily on first contact.
        self._settled = False
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
                from o2xygen.lifecycle import evaluate_stellar
                wd = self._wd_grid_getter() if self._wd_grid_getter else None
                cool = self._cool_grid_getter() if self._cool_grid_getter else None
                p = evaluate_stellar(mini, feh or 0.0, age_yr,
                                     grid, wd, cool, s.extra)
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
        required = [s for s in self.system.stars()
                    if s.extra.get("age0_gyr") is not None]
        if not props or any(s.name not in props for s in required):
            # Grids load in stages. A partial result must NOT mark the epoch
            # settled: doing so strands unavailable WDs at their MIST seed
            # mass forever (the Eidolon cold-start race, 0.4.3).
            return False
        # Settling MIST seed masses onto the epoch is initialization, not a
        # new wind pulse.  In particular, a save's existing wind ledger must
        # never receive the same historical loss a second time on reload.
        self.sim.update_masses({n: v["mass_kg"] for n, v in props.items()},
                               record_wind=False)
        self._apply_star_props(props)
        self.sim._cp_times = [float(self.sim.t_s)]
        self.sim._cp_snaps = [self.sim._snapshot()]
        return True

    # ---------------------------------------------------------- spin & tides
    def _conserve_structural_spin(self, props: dict, particles, idx: dict,
                                  spins: dict, radii: dict) -> None:
        """Keep each star's I*omega continuous across a structure update.

        MIST supplies M(t), R(t), not rotational evolution.  ADR-0030 adopts
        the explicit zero-torque closure: isotropic wind carries translational
        angular momentum but no additional spin torque.  Thus contraction
        spins up and expansion spins down, and tides remain the only exchange
        between stellar spin and orbit.  Initial MIST settlement does NOT call
        this helper: a loaded spin already belongs to the settled epoch.
        """
        for name, v in props.items():
            b = self.system.bodies[name]
            alpha = float(b.extra.get("inertia_factor", DEFAULT_INERTIA_FACTOR))
            old_m = float(particles[idx[name]].m)
            old_r = float(radii.get(name, b.radius_m))
            old_i = alpha * old_m * old_r * old_r
            new_i = alpha * float(v["mass_kg"]) * float(v["radius_m"]) ** 2
            sp = spins.get(name)
            if sp is not None and old_i > 0.0 and new_i > 0.0:
                sp["w"] = np.asarray(sp["w"], dtype=float) * (old_i / new_i)

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
                     spins: dict, radii: dict | None = None) -> None:
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
            R = float((radii or {}).get(host, b.radius_m))
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
                            record: bool = False, spins: dict | None = None,
                            wind=None, radii: dict | None = None) -> None:
        """Evolve a live simulation or a private raw rebound copy.

        Every path carries its own same-time wind ledger and radius map.  Raw
        copies must never mutate the live ledger/catalog, but otherwise use
        the exact same mass-event, structural-spin and tidal operators.
        """
        if t_to <= t_from:
            return
        is_live = sim is self.sim
        particles = sim.sim.particles if is_live else sim.particles
        idx = self.sim._index                # copies share the live ordering
        if spins is None:
            spins = self._spin_map(t_from)
        if wind is None:
            if not is_live:
                raise ValueError("raw evolution requires a private wind ledger")
            wind = sim.wind
        if radii is None:
            radii = {n: float(b.radius_m)
                     for n, b in self.system.bodies.items()}
            if not is_live:
                # A checkpoint may be older than the live catalog radius.
                for n, v in self._star_props_at(t_from).items():
                    radii[n] = float(v["radius_m"])
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
                self._conserve_structural_spin(props, particles, idx,
                                               spins, radii)
                masses = {k: v["mass_kg"] for k, v in props.items()}
                if is_live:
                    sim.update_masses(masses, record_wind=True)
                    self._apply_star_props(props)
                else:
                    update_sim_masses(sim, idx, masses, wind,
                                      record_wind=True)
                radii.update({k: float(v["radius_m"])
                              for k, v in props.items()})
            self._apply_tides(sim, prev, t_stop, spins, radii=radii)
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

    def _copy_at(self, t_s: float):
        """(raw rebound sim, spin map, wind ledger) at t_s.

        Future queries start from the LIVE snapshot; past queries restore the
        nearest checkpoint's same-time ledger.  Spin is not archived
        historically, so a past view still carries current spin metadata.
        """
        now = self.sim.t_s
        if t_s >= now:
            snap = self.sim._snapshot()
            sim = self.sim._restore(snap, now)
            spins = self._spin_map(now)
            wind = snap[3].copy()
            self._evolve_sim_between(sim, now, t_s, spins=spins, wind=wind)
            return sim, spins, wind
        cp = self.sim._cp_times
        i = max(0, bisect_right(cp, t_s) - 1)
        need = int(abs(t_s - cp[i]) / self.sim.dt_s) + 1
        if need > MAX_INTEGRATE_STEPS:
            raise IntegrateBudgetExceeded(
                f"query at {(t_s - cp[i])/DAY:.3g} d from the nearest "
                f"checkpoint needs ~{need} steps > budget "
                f"{MAX_INTEGRATE_STEPS} (dt={self.sim.dt_s/DAY:.3g} d): "
                "shrink the span or raise dt")
        snap = self.sim._cp_snaps[i]
        sim = self.sim._restore(snap, cp[i])
        spins = self._spin_map(now)
        wind = snap[3].copy()
        if t_s >= cp[i]:
            self._evolve_sim_between(sim, cp[i], t_s, spins=spins, wind=wind)
        else:                                   # before the epoch: plain rewind
            sim.dt = -abs(self.sim.dt_s)
            sim.integrate(t_s, exact_finish_time=1)
        return sim, spins, wind

    def state_at(self, t_days: float | None) -> dict:
        """State snapshot at t_days (positions, velocities, masses,
        luminosities, osculating elements). NON-DESTRUCTIVE
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
                wind = self.sim.wind
            else:
                t_s = float(t_days) * DAY
                sim, spins, wind = self._copy_at(t_s)
            return self._snapshot_dict(sim, t_s, spins, wind)

    def _conservation_dict(self, names: list[str], ms: np.ndarray,
                           pos: np.ndarray, vel: np.ndarray, spins: dict,
                           star_props: dict, wind, t_s: float) -> dict:
        """Open-body and closed bodies+wind integrals about the fixed origin."""
        def vec(a):
            return [float(x) for x in np.asarray(a).reshape(3)]

        mass_b = float(ms.sum())
        p_b = (ms[:, None] * vel).sum(axis=0)
        q_b = (ms[:, None] * pos).sum(axis=0)
        l_orbit_b = (ms[:, None] * np.cross(pos, vel)).sum(axis=0)
        l_spin_b = np.zeros(3)
        for k, name in enumerate(names):
            b = self.system.bodies[name]
            alpha = float(b.extra.get("inertia_factor", DEFAULT_INERTIA_FACTOR))
            radius = float(star_props.get(name, {}).get("radius_m", b.radius_m))
            inertia = alpha * float(ms[k]) * radius * radius
            l_spin_b += inertia * np.asarray(spins[name]["w"], dtype=float)
        w = (wind if wind is not None else self.sim.wind).at(t_s)
        mass_tot = mass_b + w.mass_kg
        p_tot = p_b + w.momentum_kg_m_s
        q_tot = q_b + w.first_moment_kg_m
        l_orbit_tot = l_orbit_b + w.angular_momentum_kg_m2_s
        l_tot = l_orbit_tot + l_spin_b
        return {
            "bodies": {
                "mass_kg": mass_b,
                "momentum_kg_m_s": vec(p_b),
                "orbital_angular_momentum_kg_m2_s": vec(l_orbit_b),
                "spin_angular_momentum_kg_m2_s": vec(l_spin_b),
                "first_moment_kg_m": vec(q_b),
                "barycenter_m": vec(q_b / mass_b),
            },
            "wind": w.to_dict(),
            "total": {
                "mass_kg": mass_tot,
                "momentum_kg_m_s": vec(p_tot),
                "orbital_angular_momentum_kg_m2_s": vec(l_orbit_tot),
                "spin_angular_momentum_kg_m2_s": vec(l_spin_b),
                "angular_momentum_kg_m2_s": vec(l_tot),
                "first_moment_kg_m": vec(q_tot),
                "barycenter_m": vec(q_tot / mass_tot),
            },
        }

    def _snapshot_dict(self, sim, t_s: float,
                       spins: dict | None = None, wind=None) -> dict:
        """Serialise a sim object (live NBodySimulation or raw rebound copy)
        sitting at t_s into the /api/state payload."""
        live = sim is self.sim
        ps = sim.sim.particles if live else sim.particles
        idx = self.sim._index
        bodies = {}
        for name, i in idx.items():
            r = np.array(ps[i].xyz)
            v = np.array(ps[i].vxyz)
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
                # Seed mass/radius are inputs/placeholders, not current stellar
                # outputs. Tell the client when a staged grid is still pending
                # instead of presenting a giant progenitor as a settled WD.
                "properties_ready": p is not None,
                "mass_kg": float(p["mass_kg"]) if p else None,
                "teff_k": float(p["teff_k"]) if p else float(s.extra.get("teff_k", 5772.0)),
                "radius_m": float(p["radius_m"]) if p else float(s.radius_m),
                "luminosity_w": float(p["luminosity_w"]) if p else float(s.luminosity_w),
            }
        conservation = self._conservation_dict(
            names_order, ms, pos_arr, vel_arr, spins, props, wind, t_s)
        return {
            "t_days": float(t_s / DAY),
            "dt_s": float(self.sim.dt_s),
            "bodies": bodies,
            "elements": elements,
            "stars": stars,
            "spins": spin_json,
            "conservation": conservation,
            "luminosity_w": {s.name: stars[s.name]["luminosity_w"]
                             for s in self.system.stars()},
        }

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
            out.observation_points = [dict(p) for p in self.system.observation_points]
            # Rebase only the ledger clock.  Q(T), P, L and M are copied at
            # the SAME live epoch; adding P*T again would double-propagate it.
            out.wind = self.sim.wind.at(self.sim.t_s)
            out.wind.time_s = 0.0
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
            snap = self.sim._snapshot()
            sim = self.sim._restore(snap, t0)
            spins = self._spin_map(t0)
            wind = snap[3].copy()
            idx = self.sim._index
            star_names = [s.name for s in self.system.stars()]
            # fallback luminosities when the MIST grid is unavailable
            base_lum = {s.name: s.luminosity_w for s in self.system.stars()}
        rows = []
        prev = t0
        for i in range(n):
            t = t0 + span_days * DAY * i / max(n - 1, 1)
            self._evolve_sim_between(sim, prev, t, spins=spins, wind=wind)
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
        snap = self.state_at(t_days)
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
