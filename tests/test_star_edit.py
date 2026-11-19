"""Manual stellar overrides from the edit-mode star editor."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from o2xygen.constants import L_SUN, M_SUN, R_SUN
from o2xygen import world as world_mod
from preset import build_preset
from workspace import Workspace


def test_static_off_grid_star_properties_survive_save_load_and_evolution(tmp_path):
    w = build_preset()
    b = w.bodies["Vesper"]
    b.mass_kg = 1.10 * M_SUN
    b.radius_m = 1.20 * R_SUN
    b.luminosity_w = 0.90 * L_SUN
    b.extra.update({"evolution_model": "static",   # off-grid, not user-editable MIST output
                    "mini_msun": 1.10,             # static objects have one fixed mass
                    "luminosity_w": b.luminosity_w,
                    "teff_k": 6000.0})
    back = world_mod.load_world(world_mod.save_world(w, tmp_path / "manual"))
    assert back.bodies["Vesper"].extra["luminosity_w"] == pytest.approx(0.90 * L_SUN)

    ws = Workspace(back, grid_getter=None)
    snap = ws.state_at(None, with_trails=False)
    # autosave/save-as rebuilds Body objects; luminosity must survive that too
    assert ws.snapshot_for_save().bodies["Vesper"].luminosity_w == \
        pytest.approx(0.90 * L_SUN)
    st = snap["stars"]["Vesper"]
    assert st["radius_m"] == pytest.approx(1.20 * R_SUN)
    assert st["luminosity_w"] == pytest.approx(0.90 * L_SUN)
    assert st["teff_k"] == 6000.0
    assert snap["bodies"]["Vesper"]["m"] == pytest.approx(1.10 * M_SUN)
    ws.advance_to(30.0)
    again = ws.state_at(None, with_trails=False)["stars"]["Vesper"]
    assert again == st                       # static off-grid stars never MIST-overwrite


def test_current_stellar_outputs_are_not_editable_inputs():
    import app
    for field in ("mass_msun", "radius_rsun", "luminosity_lsun", "teff_k"):
        with pytest.raises(app.HttpError) as exc:
            app.api_star_edit({"name": "Vesper", field: 1.0})
        assert exc.value.status == 400
