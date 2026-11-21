"""Stellar save schema: MIST inputs in, derived outputs never persisted."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from o2xygen.constants import M_SUN
from o2xygen import world as world_mod
from preset import build_preset


def test_star_save_contains_only_mist_inputs_and_dynamical_state():
    w = build_preset()
    d = world_mod.system_to_dict(w)
    for bd in d["bodies"]:
        if bd["kind"] != "star":
            continue
        assert {"mini_msun", "feh", "age0_gyr"} <= set(bd)
        assert "mass_kg" not in bd
        assert "radius_m" not in bd
        extra = bd.get("extra", {})
        for key in ("teff_k", "luminosity_w", "evolution_model",
                    "manual_override", "manual_mass_kg"):
            assert key not in extra
    back = world_mod.dict_to_system(d)
    for b in back.stars():
        assert b.mass_kg == pytest.approx(b.extra["mini_msun"] * M_SUN)
        assert b.radius_m == 0.0                 # settled lazily by MIST
        assert b.luminosity_w == 0.0


def test_current_stellar_outputs_are_not_editable_inputs():
    import app
    for field in ("mass_msun", "radius_rsun", "luminosity_lsun", "teff_k"):
        with pytest.raises(app.HttpError) as exc:
            app.api_star_edit({"name": "Vesper", field: 1.0})
        assert exc.value.status == 400


def test_preview_is_non_destructive_and_reports_lifecycle():
    import app
    app._load_grid()
    before = app.SYSTEM.bodies["Vesper"].extra["age0_gyr"]
    t0 = app.WS.sim.t_s
    path = app._worlds_dir() / f"{app.CURRENT_WORLD}.oxygen"
    mtime = path.stat().st_mtime
    got = app.api_star_preview({"name": "Vesper", "age0_gyr": 100.0})
    assert got["model"] in ("mist3_wd", "wdcool")
    assert got["radius_rsun"] < 0.1
    assert app.SYSTEM.bodies["Vesper"].extra["age0_gyr"] == before
    assert app.WS.sim.t_s == t0
    assert path.stat().st_mtime == mtime
