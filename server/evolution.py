"""Evolution job (推演): irreversible forward simulation that rewrites the
world save file.

Rationale (user, 2026-11-15): with worlds as files, the correct backup
logic is "save-as BEFORE evolving", not "export after". So evolution
mutates the world's own .oxygen file, autosaved every 0.5 s of wall time.

Semantics: the job advances the live integrator; each autosave snapshots
the CURRENT state as the world's new initial conditions (pos0/vel0), with
stellar age0 advanced by elapsed sim-years, so a reloaded world reproduces
the same physical system at its new epoch. Cancelling keeps the last
autosave - everything after it is discarded.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from o2xygen.constants import DAY
from o2xygen.nbody import MAX_INTEGRATE_STEPS
from o2xygen import world as world_mod

WALL_CHUNK_S = 0.4          # target wall time per outer iteration
AUTOSAVE_INTERVAL_S = 0.5   # rewrite the save at most every 0.5 s wall
# Per-LOCK-HOLD step budget: one integration chunk must never hold the
# workspace lock long enough to starve interactive requests (measured
# 2026-11-15: a ~2.4M-step chunk locked the API out for ~7 s at high speed).
# Speeds above ~CHUNK/dt are CPU-bound and simply run at this rate.
CHUNK_STEP_BUDGET = 200_000


class EvolutionJob:
    """One evolution session. Control rules (user spec):
      - duration (最大推演时长) is FIXED once started;
      - pausing is always allowed;
      - cancel and speed change are allowed ONLY while paused."""

    def __init__(self, ws, world_path, duration_days: float,
                 speed_days_per_s: float):
        self.ws = ws
        self.world_path = Path(world_path)
        t0_days = ws.sim.t_s / DAY
        self.start_days = t0_days
        self.end_days = t0_days + float(duration_days)
        self.done_days = t0_days
        self.speed = float(speed_days_per_s)
        self._expected_days = t0_days   # absolute clock the job expects;
        self.state = "running"          # running | paused | done | cancelled | error
        self.error: str | None = None
        self._ctl = threading.Lock()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------- control
    def pause(self):
        with self._ctl:
            if self.state == "running":
                self.state = "paused"

    def resume(self):
        with self._ctl:
            if self.state == "paused":
                self.state = "running"

    def cancel(self):
        # only a PAUSED evolution may be cancelled outright (user spec)
        with self._ctl:
            if self.state == "paused":
                self.state = "cancelled"

    def set_speed(self, speed_days_per_s: float):
        with self._ctl:
            if self.state == "paused" and speed_days_per_s > 0:
                self.speed = float(speed_days_per_s)

    @property
    def active(self) -> bool:
        return self.state in ("running", "paused")

    @property
    def progress(self) -> float:
        span = self.end_days - self.start_days
        return 0.0 if span <= 0 else min(1.0, (self.done_days - self.start_days) / span)

    def status(self) -> dict:
        return {"state": self.state, "progress": self.progress,
                "start_days": self.start_days, "done_days": self.done_days,
                "end_days": self.end_days, "speed": self.speed,
                "world": self.world_path.stem, "error": self.error}

    # --------------------------------------------------------------- run
    def start(self) -> threading.Thread:
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="o2xygen-evolution")
        self._thread.start()
        return self._thread

    def wait(self, timeout: float | None = None) -> bool:
        if self._thread:
            self._thread.join(timeout)
            return not self._thread.is_alive()
        return True

    def _run(self):
        ws = self.ws
        chunk_days = CHUNK_STEP_BUDGET * ws.sim.dt_s / DAY
        next_save = time.time() + AUTOSAVE_INTERVAL_S
        last_wall = time.time()
        self._debt_days = 0.0
        try:
            while True:
                with self._ctl:
                    st = self.state
                now = time.time()
                wall_dt = now - last_wall
                last_wall = now
                if st == "cancelled":
                    break
                if st == "paused":
                    time.sleep(0.05)
                    continue
                sim_now = ws.sim.t_s / DAY
                if sim_now >= self.end_days:
                    break
                # DEBT-BASED pacing: accrue sim-time debt at exactly the
                # configured speed per REAL wall second, then integrate what
                # is owed (bounded by the step budget per lock hold). The
                # old fixed-0.4s-cycle design ran at owe/work_time, which
                # measured HALF the configured speed once autosaves started
                # costing real wall time - a guaranteed client/server rate
                # mismatch.
                self._debt_days += self.speed * wall_dt
                owe_days = min(self._debt_days, chunk_days,
                               self.end_days - sim_now)
                if owe_days <= 1e-9:
                    time.sleep(0.01)
                    continue
                with ws.lock:
                    # external-clock-move detector: another caller holding
                    # the lock between our chunks can advance or rewind the
                    # shared integrator. Abort LOUDLY, skip the final save.
                    actual_days = ws.sim.t_s / DAY
                    drift = actual_days - self._expected_days
                    if abs(drift) > max(1.0, owe_days):
                        raise RuntimeError(
                            f"sim clock moved externally by {drift:.3g} d "
                            "(a query is mutating the live integrator); "
                            "evolution aborted, last autosave kept")
                    t_from = ws.sim.t_s
                    t_to = t_from + owe_days * DAY
                    ws._evolve_between(t_from, t_to, record=False)
                    ws.sim.integrate_to(t_to, record_checkpoints=False)
                    self._expected_days = t_to / DAY
                    self._debt_days -= owe_days
                self.done_days = ws.sim.t_s / DAY
                if time.time() >= next_save:
                    self._autosave()
                    next_save = time.time() + AUTOSAVE_INTERVAL_S
            with self._ctl:
                if self.state == "running":
                    self._autosave()          # final save on completion
                    self.state = "done"
        except Exception as e:                # pragma: no cover
            self.error = str(e)
            with self._ctl:
                self.state = "error"

    def _autosave(self):
        snap = self.ws.snapshot_for_save()
        world_mod.save_world(snap, self.world_path)
