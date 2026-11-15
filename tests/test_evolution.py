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
    ws.state_at(365.25)                       # advance one year
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
