"""MIST III WD cooling: real track anchors + Workspace/save integration."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from o2xygen.constants import L_SUN, R_SUN
from o2xygen import world as world_mod
from preset import build_preset
from workspace import Workspace

DATA = Path(__file__).resolve().parents[1] / ".data" / "mist3_wd.npz"


@pytest.fixture(scope="module")
def grid():
    if not DATA.exists():
        pytest.skip("run scripts/import_mist3_wd.py first")
    from o2xygen.wd_mist3 import Mist3WDGrid
    return Mist3WDGrid(DATA)


def test_wd_track_anchor_at_1p2_msun_15_gyr(grid):
    p = grid.evaluate(1.2, 0.0, 15.0e9)
    assert p.mass_msun == pytest.approx(0.554515, abs=1e-12)
    assert p.teff_k == pytest.approx(4275.518904362204, rel=1e-12)
    assert p.radius_m / R_SUN == pytest.approx(0.012800990054372784, rel=1e-12)
    assert p.luminosity_w / L_SUN == pytest.approx(4.933266657931625e-05, rel=1e-12)
    assert p.mdot_msun_per_yr == 0.0


def test_wd_cools_while_mass_stays_constant(grid):
    young = grid.evaluate(1.2, 0.0, 12.0e9)
    old = grid.evaluate(1.2, 0.0, 15.0e9)
    assert old.mass_msun == young.mass_msun
    assert old.luminosity_w < young.luminosity_w
    assert old.teff_k < young.teff_k


def test_wd_workspace_and_save_use_model_inputs_only(grid, tmp_path):
    w = build_preset()
    b = w.bodies["Vesper"]
    b.extra.update({"evolution_model": "mist3_wd", "mini_msun": 1.2,
                    "feh": 0.0, "age0_gyr": 15.0})
    ws = Workspace(w, grid_getter=None, wd_grid_getter=lambda: grid)
    got = ws.state_at(None, with_trails=False)["stars"]["Vesper"]
    assert got["teff_k"] == pytest.approx(4275.518904362204)
    ws.advance_to(365.25)
    later = ws.state_at(None, with_trails=False)["stars"]["Vesper"]
    assert later["luminosity_w"] < got["luminosity_w"]
    assert later["teff_k"] < got["teff_k"]
    saved = world_mod.system_to_dict(ws.snapshot_for_save())
    bd = next(x for x in saved["bodies"] if x["name"] == "Vesper")
    assert bd["extra"]["evolution_model"] == "mist3_wd"
    assert "mass_kg" not in bd and "radius_m" not in bd
    for key in ("teff_k", "luminosity_w"):
        assert key not in bd["extra"]
    back = world_mod.dict_to_system(saved)
    ws2 = Workspace(back, grid_getter=None, wd_grid_getter=lambda: grid)
    again = ws2.state_at(None, with_trails=False)["stars"]["Vesper"]
    for key in ("teff_k", "radius_m", "luminosity_w"):
        assert again[key] == pytest.approx(later[key], rel=1e-12)
