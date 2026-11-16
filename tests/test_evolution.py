"""Evolution (推演): irreversible forward simulation rewriting the world save.

Semantics under test:
  * snapshot_for_save: pos0/vel0 = live state, stellar age0 advanced by
    elapsed sim-years -> a reloaded world reproduces the same physical
    system at its new epoch;
  * EvolutionJob: integrates the live workspace, autosaves mid-run,
    finishes with state "done"; a reloaded world at t=0 matches the
    pre-save live state;
  * control rules (user spec): duration fixed at start; pausing always
    allowed; cancel and speed change ONLY while paused.
"""
import sys
import time
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from o2xygen.constants import DAY
from o2xygen import world as world_mod
from workspace import Workspace
from evolution import EvolutionJob
from preset import build_preset


def test_snapshot_for_save_shifts_epoch():
    ws = Workspace(build_preset(), grid_getter=None)
    ws.advance_to(365.25)                     # advance one year
    snap = ws.snapshot_for_save()
    states = ws.sim.live_state()
    for n, b in snap.bodies.items():
        assert np.allclose(b.pos0, states[n][0], rtol=0, atol=1e-3)
        assert np.allclose(b.vel0, states[n][1], rtol=0, atol=1e-9)
    # stellar age0 advanced by exactly the elapsed sim-years
    star = snap.bodies["Αστραία"]
    assert star.extra["age0_gyr"] == pytest.approx(
        1.26 + ws.sim.t_s / (365.25 * DAY) / 1e9, abs=1e-12)


def test_evolution_rewrites_world_and_reload_matches(tmp_path):
    ws = Workspace(build_preset(), grid_getter=None)
    path = world_mod.save_world(build_preset(), tmp_path / "演化测试")
    job = EvolutionJob(ws, path, duration_days=60.0, speed_days_per_s=1e6)
    job.start()
    assert job.wait(timeout=60)

    assert job.state == "done"
    assert job.progress == pytest.approx(1.0)
    # the save file was rewritten: initial conditions now hold the evolved state
    reloaded = world_mod.load_world(path)
    end_states = ws.sim.live_state()
    for n in reloaded.bodies:
        assert np.allclose(reloaded.bodies[n].pos0, end_states[n][0],
                           rtol=0, atol=1e-3), n
    # age0 advanced by the evolved span (~60 d)
    assert reloaded.bodies["Αστραία"].extra["age0_gyr"] == pytest.approx(
        1.26 + 60.0 / 365.25 / 1e9, abs=1e-9)

    # reloading the mutated world and asking t=0 reproduces the same layout
    ws2 = Workspace(reloaded, grid_getter=None)
    s0 = ws2.state_at(0.0)
    for n in reloaded.bodies:
        assert np.allclose(s0["bodies"][n]["p"], end_states[n][0],
                           rtol=0, atol=1e-3), n


def _fake_grid():
    """Canned MIST stand-in so the mass-update branch (props present) runs
    in tests - production-only AttributeErrors hid there before (the
    'NBodySimulation has no particles' job crash, 2026-11-15)."""
    class P:
        def __init__(self):
            self.mass_msun = 1.29
            self.radius_m = 9.26e8
            self.luminosity_w = 2.8e26
            self.teff_k = 6474.0
    class G:
        def evaluate(self, m, feh, age):
            return P()
    return G()


def test_job_runs_with_stellar_props_present(tmp_path):
    """The evolution cadence must update masses on BOTH the live sim and
    restored copies without crashing (duck-typed particles access)."""
    ws = Workspace(build_preset(), grid_getter=_fake_grid)
    path = world_mod.save_world(build_preset(), tmp_path / "props")
    job = EvolutionJob(ws, path, duration_days=40.0, speed_days_per_s=1e5)
    job.start()
    assert job.wait(timeout=60)
    assert job.state == "done", job.error
    # non-destructive sampling with props present must also leave t intact
    t0 = ws.sim.t_s
    ws.flux_series_forward("Αύρα", 365.25, 5)
    assert ws.sim.t_s == t0


def test_autosave_happens_during_run(tmp_path):
    ws = Workspace(build_preset(), grid_getter=None)
    path = world_mod.save_world(build_preset(), tmp_path / "autosave")
    mtime0 = path.stat().st_mtime
    # slow evolution: plenty of wall time, autosave must fire within ~1 s
    job = EvolutionJob(ws, path, duration_days=2000.0, speed_days_per_s=2.0)
    job.start()
    time.sleep(1.2)
    assert path.stat().st_mtime > mtime0, "autosave did not fire within ~1 s"
    job.pause()
    job.cancel()
    assert job.wait(timeout=30)
    assert job.state == "cancelled"


def test_flux_series_forward_is_non_destructive():
    """INCIDENT 2026-11-15 (client-server desync): the temperature query
    advanced the LIVE integrator, killing a running evolution job (sim.t
    overshot end_days -> 'done') while physics kept moving. Temperature
    sampling must run on a restored copy and leave sim.t bit-exact."""
    ws = Workspace(build_preset(), grid_getter=None)
    ws.advance_to(10.0)
    t0 = ws.sim.t_s
    rows = ws.flux_series_forward("Múspellsheimr", 3652.5, 13)
    assert ws.sim.t_s == t0, "temperature sampling moved the live clock!"
    assert len(rows) == 13
    assert rows[0]["d"] == pytest.approx(t0 / (3600 * 24), rel=1e-9)
    # first sample matches the live illumination at the current time
    live = ws.flux_at("Múspellsheimr", t0 / (3600 * 24))["illumination"]["total"]
    assert rows[0]["flux"] == pytest.approx(live, rel=1e-6)


def test_state_at_is_non_destructive():
    """ADR-0018: NO query may move the live clock - not even state_at with
    an explicit t_days (it once scrubbed the shared integrator on every
    poll). Explicit-time queries run on checkpoint-restored copies."""
    ws = Workspace(build_preset(), grid_getter=None)
    ws.advance_to(10.0)
    t0 = ws.sim.t_s
    snap = ws.state_at(500.0)                  # explicit future time
    assert ws.sim.t_s == t0, "state_at(t_days) moved the live clock!"
    assert snap["t_days"] == pytest.approx(500.0)
    snap_now = ws.state_at(None)               # current time: no 500, no crash
    assert snap_now["t_days"] == pytest.approx(t0 / DAY)
    past = ws.state_at(5.0)                    # explicit past time: also safe
    assert ws.sim.t_s == t0
    assert past["t_days"] == pytest.approx(5.0)


def test_flux_at_excludes_self_irradiation():
    """Querying a STAR's irradiation must not divide by zero on its own
    flux (returned Infinity/NaN before the 2026-11-16 audit)."""
    ws = Workspace(build_preset(), grid_getter=None)
    got = ws.flux_at("Αστραία", None)
    assert "Αστραία" not in got["illumination"]
    assert np.isfinite(got["illumination"]["total"])


def test_job_aborts_loudly_on_external_clock_move(tmp_path):
    """If anything moves the shared clock between chunks, the job must
    abort with a LOUD error and skip the final save - never report a
    phantom 'done' and never keep integrating into a corrupted timeline."""
    ws = Workspace(build_preset(), grid_getter=None)
    path = world_mod.save_world(build_preset(), tmp_path / "desync")
    job = EvolutionJob(ws, path, duration_days=1e5, speed_days_per_s=2.0)
    job.start()
    time.sleep(1.0)
    # external interference: a query mutating the live integrator (+50 d).
    # NB: advance_to acquires ws.lock itself - do NOT wrap it (non-reentrant).
    ws.advance_to(ws.sim.t_s / (3600 * 24) + 50.0)
    assert job.wait(timeout=30)
    assert job.state == "error"
    assert "moved externally" in job.error
    # the abort must skip the final save: no writes after the abort settles
    mtime = path.stat().st_mtime
    time.sleep(1.2)
    assert path.stat().st_mtime == mtime, "final save was written after abort"


def test_control_rules_pause_cancel_speed(tmp_path):
    ws = Workspace(build_preset(), grid_getter=None)
    path = world_mod.save_world(build_preset(), tmp_path / "ctl")
    job = EvolutionJob(ws, path, duration_days=1e6, speed_days_per_s=5.0)
    job.start()
    time.sleep(0.3)

    # running: cancel and speed change must NOT take effect (API layer
    # refuses with 409; the job itself ignores them outside "paused")
    job.cancel()
    assert job.state == "running"
    job.set_speed(99.0)
    assert job.speed == 5.0

    job.pause()
    time.sleep(0.2)
    assert job.state == "paused"
    job.set_speed(10.0)                          # allowed while paused
    assert job.speed == 10.0
    job.cancel()                                 # allowed while paused
    assert job.state == "cancelled"
    assert job.wait(timeout=30)
