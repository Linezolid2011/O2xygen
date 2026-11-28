"""WD lifecycle must never substitute a progenitor during grid loading."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from o2xygen.constants import M_SUN, R_SUN, L_SUN
from o2xygen.lifecycle import resolve_stellar, evaluate_stellar
from o2xygen.stars import StarProperties
from o2xygen.system import System, Body, STAR
from workspace import Workspace


class FakeMist:
    def final_point(self, mini, feh):
        if mini >= 6.0:
            return {"age_yr": 76.9612105e6, "mass_msun": 1.2032585952}
        return {"age_yr": 100.0e9, "mass_msun": 0.55}

    def evaluate(self, mini, feh, age):
        return StarProperties(mass_msun=mini, radius_m=R_SUN,
                              luminosity_w=L_SUN, teff_k=5772.0,
                              mdot_msun_per_yr=0.0)


class FakeWD:
    def __init__(self, covered=False):
        self.covered = covered

    def wd_mass_covered(self, mass, feh):
        return self.covered

    def is_wd_age(self, mini, feh, age):
        return age >= (76.9612105e6 if mini >= 6.0 else 100.0e9)


def test_missing_wd_grids_are_unavailable_not_giant_progenitor():
    args = (6.0, 0.0, 77.0246214e6, FakeMist(), None, None)
    assert resolve_stellar(*args)["model"] == "unavailable"
    with pytest.raises(ValueError, match="not ready"):
        evaluate_stellar(*args)
    args = (6.0, 0.0, 77.0246214e6, FakeMist(), FakeWD(False), None)
    assert resolve_stellar(*args)["model"] == "unavailable"
    with pytest.raises(ValueError, match="not ready"):
        evaluate_stellar(*args)


def test_wd_models_and_preformation_fallback():
    cool = SimpleNamespace(evaluate=lambda *args: "wdcool-result")
    assert evaluate_stellar(6.0, 0.0, 77.0246214e6,
                            FakeMist(), None, cool) == "wdcool-result"
    assert resolve_stellar(6.0, 0.0, 77.0246214e6,
                           FakeMist(), FakeWD(True), None)["model"] == "mist3_wd"
    assert resolve_stellar(6.0, 0.0, 70e6,
                           FakeMist(), None, None)["model"] == "mist"


def test_workspace_settles_only_when_every_star_is_ready():
    world = System("staged-grids")
    world.add(Body("E", STAR, mass_kg=6.0 * M_SUN,
                   extra={"mini_msun": 6.0, "feh": 0.0, "age0_gyr": 0.077}))
    world.add(Body("A", STAR, mass_kg=M_SUN,
                   extra={"mini_msun": 1.0, "feh": 0.0, "age0_gyr": 1.0}))
    cool = SimpleNamespace(evaluate=lambda mass, age, envelope: StarProperties(
        mass_msun=mass, radius_m=0.006 * R_SUN, luminosity_w=0.1 * L_SUN,
        teff_k=48000.0, mdot_msun_per_yr=0.0))
    grids = {"cool": None}
    ws = Workspace(world, grid_getter=lambda: FakeMist(),
                   wd_grid_getter=lambda: FakeWD(False),
                   cool_grid_getter=lambda: grids["cool"])
    # A is evaluable, E is not: partial output must not settle the workspace
    # or leave E's 6 M_sun MIST seed installed as its current dynamical mass.
    assert not ws._settled
    e_particle = ws.sim.sim.particles[ws.sim._index["E"]]
    assert e_particle.m == pytest.approx(6.0 * M_SUN)
    grids["cool"] = cool
    stars = ws.state_at(None)["stars"]
    assert ws._settled
    assert e_particle.m == pytest.approx(1.2032585952 * M_SUN)
    assert stars["E"]["radius_m"] == pytest.approx(0.006 * R_SUN)
    assert stars["E"]["properties_ready"] is True


def test_body_json_masks_seed_outputs_until_properties_ready():
    from app import body_json
    b = Body("E", STAR, mass_kg=6 * M_SUN, radius_m=1.7e9,
             luminosity_w=8.8e29, extra={"teff_k": 100000, "mini_msun": 6})
    d = body_json(b, properties_ready=False)
    assert d["properties_ready"] is False
    assert d["mass_kg"] is None and d["radius_m"] is None
    assert d["luminosity_w"] is None and "teff_k" not in d["extra"]
    assert d["extra"]["mini_msun"] == 6
    assert b.mass_kg == 6 * M_SUN
