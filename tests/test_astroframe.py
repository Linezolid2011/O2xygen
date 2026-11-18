"""Heliocentric ICRS/J2000 world placement and absolute epoch continuity."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from o2xygen.astroframe import (
    J2000_JD_TT, PC_M, advance_epoch, default_astro_frame,
    icrs_to_world_m, julian_year, normalize_astro_frame, world_to_icrs_pc,
)
from o2xygen.system import System
from o2xygen import world as world_mod


def test_identity_and_zero_position_defaults():
    f = default_astro_frame()
    assert f["epoch_jd_tt"] == J2000_JD_TT
    assert julian_year(f) == 2000.0
    assert np.allclose(world_to_icrs_pc(f, [PC_M, 2 * PC_M, -PC_M]),
                       [1.0, 2.0, -1.0])


def test_translation_rotation_and_inverse():
    f = default_astro_frame()
    f["position_j2000_pc"] = [10.0, 0.0, 0.0]
    f["world_to_icrs_quat_wxyz"] = [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
    got = world_to_icrs_pc(f, [PC_M, 0.0, 0.0])
    assert np.allclose(got, [10.0, 1.0, 0.0], rtol=0, atol=1e-14)
    assert np.allclose(icrs_to_world_m(f, got), [PC_M, 0.0, 0.0],
                       rtol=0, atol=10.0)       # double precision at 1 pc


def test_epoch_advances_while_position_stays_fixed():
    f = default_astro_frame()
    f["epoch_jd_tt"] = J2000_JD_TT + 365.25       # world t=0 at 2001.0
    f["position_j2000_pc"] = [3.0, 4.0, 5.0]
    one_year = 365.25 * 86400.0
    assert julian_year(f, one_year) == pytest.approx(2002.0)
    assert np.allclose(world_to_icrs_pc(f, [0.0, 0.0, 0.0], one_year),
                       [3.0, 4.0, 5.0])
    rebased = advance_epoch(f, one_year)
    assert rebased["epoch_jd_tt"] == pytest.approx(J2000_JD_TT + 2 * 365.25)
    assert julian_year(rebased) == pytest.approx(2002.0)


def test_astro_frame_save_round_trip(tmp_path):
    w = System("坐标", origin=None)
    f = default_astro_frame()
    f["position_j2000_pc"] = [1.0, -2.0, 3.0]
    f["epoch_jd_tt"] = 2460000.5
    f["world_to_icrs_quat_wxyz"] = [0.0, 1.0, 0.0, 0.0]
    w.astro_frame = normalize_astro_frame(f)
    back = world_mod.load_world(world_mod.save_world(w, tmp_path / "frame"))
    assert back.astro_frame["position_j2000_pc"] == [1.0, -2.0, 3.0]
    assert back.astro_frame["epoch_jd_tt"] == 2460000.5
    assert back.astro_frame["world_to_icrs_quat_wxyz"] == [0.0, 1.0, 0.0, 0.0]


def test_autosave_rebases_absolute_epoch():
    from workspace import Workspace
    from preset import build_preset
    ws = Workspace(build_preset(), grid_getter=None)
    ws.advance_to(365.25)
    snap = ws.snapshot_for_save()
    assert snap.astro_frame["epoch_jd_tt"] == pytest.approx(
        J2000_JD_TT + 365.25)
    assert julian_year(snap.astro_frame) == pytest.approx(2001.0)
