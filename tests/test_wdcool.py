"""WDCool fallback is selected ONLY by WD mass outside MIST III coverage."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from o2xygen.constants import L_SUN, R_SUN
from o2xygen.system import System, Body, STAR
from o2xygen import world as world_mod

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def grids():
    from o2xygen.wdcool import WDCoolGrid
    from o2xygen.wd_mist3 import Mist3WDGrid
    return Mist3WDGrid(ROOT / ".data" / "mist3_wd.npz"), \
        WDCoolGrid(ROOT / ".data" / "wdcool.npz")


def test_wdcool_anchor_1p2_msun_100_myr(grids):
    _, cool = grids
    p = cool.evaluate(1.2, 100.0e6, "thick")
    assert p.mass_msun == 1.2
    assert p.teff_k == pytest.approx(34615.34449122387, rel=1e-12)
    assert p.radius_m / R_SUN == pytest.approx(0.00586598403862318, rel=1e-12)
    assert p.luminosity_w / L_SUN == pytest.approx(0.04451030334980103, rel=1e-12)
    assert p.mdot_msun_per_yr == 0.0


def test_wdcool_only_for_wd_mass_outside_mist3(grids):
    from o2xygen.lifecycle import evaluate_stellar
    wd, cool = grids
    # In-range WD mass: MIST III remains authoritative even if the optional
    # WDCool inputs happen to be present.
    inside = evaluate_stellar(1.2, 0.0, 15.0e9, None, wd, cool,
                              {"wd_mass_msun": 1.0})
    assert inside.teff_k == pytest.approx(wd.evaluate(1.2, 0.0, 15.0e9).teff_k)
    # Age overrun alone NEVER invokes WDCool.
    old = evaluate_stellar(1.2, 0.0, 1.0e20, None, wd, cool,
                           {"wd_mass_msun": 1.0})
    assert old.teff_k == pytest.approx(wd.evaluate(1.2, 0.0, 1.0e20).teff_k)
    # 1.2 M_sun is beyond MIST III's maximum WD mass -> WDCool.
    outside = evaluate_stellar(1.2, 0.0, 15.0e9, None, wd, cool,
                               {"wd_mass_msun": 1.2, "wd_envelope": "thick"})
    formation = wd.formation_age_yr(1.2, 0.0)
    expected = cool.evaluate(1.2, 15.0e9 - formation, "thick")
    assert outside.mass_msun == 1.2
    assert outside.teff_k == pytest.approx(expected.teff_k)
    assert outside.luminosity_w == pytest.approx(expected.luminosity_w)


def test_wdcool_handles_initial_mass_outside_mist3(grids):
    from o2xygen.lifecycle import evaluate_stellar
    wd, cool = grids
    got = evaluate_stellar(7.0, 0.0, 100.0e9, None, wd, cool,
                           {"wd_mass_msun": 1.2,
                            "wd_formation_age_gyr": 1.0})
    expected = cool.evaluate(1.2, 99.0e9, "thick")
    assert got.mass_msun == 1.2
    assert got.teff_k == pytest.approx(expected.teff_k)
    assert got.radius_m == pytest.approx(expected.radius_m)


def test_wdcool_inputs_survive_save_without_derived_outputs():
    w = System("WDCool", origin=None)
    w.add(Body("WD", STAR, mass_kg=1.2 * 1.98847e30,
               extra={"wd_mass_msun": 1.2, "wd_formation_age_gyr": 1.0,
                      "wd_envelope": "thin", "teff_k": 99999.0,
                      "luminosity_w": 123.0}))
    d = world_mod.system_to_dict(w)
    bd = d["bodies"][0]
    assert bd["extra"]["wd_mass_msun"] == 1.2
    assert bd["extra"]["wd_formation_age_gyr"] == 1.0
    assert bd["extra"]["wd_envelope"] == "thin"
    assert "mass_kg" not in bd and "radius_m" not in bd
    assert "teff_k" not in bd["extra"] and "luminosity_w" not in bd["extra"]
    back = world_mod.dict_to_system(d)
    assert back.bodies["WD"].extra["wd_mass_msun"] == 1.2
    assert back.bodies["WD"].extra["wd_envelope"] == "thin"
