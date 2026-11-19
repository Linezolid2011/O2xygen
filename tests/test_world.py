"""ADR-0016: [worldname].oxygen save format — round-trip fidelity."""
import numpy as np
import pytest

from o2xygen.constants import AU, M_SUN, M_EARTH
from o2xygen.system import System, Body, STAR, BODY
from o2xygen import world as world_mod


def sample_world() -> System:
    sys = System("测试世界", origin="S1")
    sys.add(Body("S1", STAR, mass_kg=1.29 * M_SUN,
                 pos0=np.zeros(3), vel0=np.zeros(3), radius_m=9.26e8,
                 extra={"feh": 0.10, "age0_gyr": 1.26}))
    sys.add(Body("b1", BODY, mass_kg=3.15 * M_EARTH,
                 pos0=np.array([7.83 * AU, 1e8, -2e8]),
                 vel0=np.array([1e3, 1.7e4, 300.0]),
                 radius_m=8.6e6, albedo=0.20))
    return sys


def test_round_trip(tmp_path):
    sys = sample_world()
    path = world_mod.save_world(sys, tmp_path / "测试世界")
    assert path.suffix == ".oxygen"
    back = world_mod.load_world(path)

    assert back.name == sys.name
    assert back.origin == sys.origin
    assert set(back.bodies) == set(sys.bodies)
    for n, b in sys.bodies.items():
        rb = back.bodies[n]
        assert rb.kind == b.kind
        assert rb.mass_kg == pytest.approx(b.mass_kg, rel=1e-12)
        assert np.allclose(rb.pos0, b.pos0, rtol=0, atol=1e-3)
        assert np.allclose(rb.vel0, b.vel0, rtol=0, atol=1e-9)
        if b.kind == STAR:
            # stars persist MIST inputs only; radius is a runtime output
            assert rb.radius_m == 0.0
            assert rb.extra["feh"] == pytest.approx(b.extra["feh"])
            assert rb.extra["age0_gyr"] == pytest.approx(b.extra["age0_gyr"])
        else:
            assert rb.radius_m == pytest.approx(b.radius_m, rel=1e-12)
            assert rb.albedo == pytest.approx(b.albedo)


def test_star_seed_survives_mass_update_round_trip(tmp_path):
    """Audit finding A1: apply_to_body(update_mass=True) changed mass_kg but
    never pinned extra['mini_msun']; a fresh in-session world then saved the
    wind-evolved mass as the MIST seed. The seed must round-trip intact."""
    from o2xygen.star_evolution import StarEstimator, StarEstimate
    from o2xygen.stars import StarProperties
    sys = sample_world()
    est = StarEstimate(1.29, 1.26e9, 0.10,
                       StarProperties(mass_msun=1.28, radius_m=9.3e8,
                                      luminosity_w=2.8e26, teff_k=6474.0,
                                      mdot_msun_per_yr=1e-13))
    StarEstimator(grid=None).apply_to_body(sys, "S1", est, update_mass=True)
    assert abs(sys.bodies["S1"].mass_kg - 1.28 * M_SUN) < 1e-3 * M_SUN
    path = world_mod.save_world(sys, tmp_path / "evolved")
    back = world_mod.load_world(path)
    # stored mass = INITIAL seed, never the evolved current mass
    assert back.bodies["S1"].extra["mini_msun"] == pytest.approx(1.29)
    assert back.bodies["S1"].mass_kg == pytest.approx(1.29 * M_SUN, rel=1e-12)


def test_duplicate_body_name_rejected():
    from o2xygen.system import System, Body, STAR
    sys = System("dup", origin="S")
    sys.add(Body("S", STAR, mass_kg=M_SUN))
    with pytest.raises(ValueError):
        sys.add(Body("S", STAR, mass_kg=M_SUN))


def test_rejects_foreign_format(tmp_path):
    p = tmp_path / "bad.oxygen"
    p.write_text('{"format": "something-else", "version": 1}', encoding="utf-8")
    with pytest.raises(ValueError):
        world_mod.load_world(p)


def test_suffix_normalised(tmp_path):
    sys = sample_world()
    path = world_mod.save_world(sys, tmp_path / "noext")
    assert path.name == "noext.oxygen"


def test_extra_metadata_round_trips(tmp_path):
    """Audit finding (2026-11-16): save/load silently dropped every 'extra'
    key - cmf, ring flags, atmosphere presets, composition notes - so a
    reloaded world lost its pipeline inputs and visuals."""
    sys = sample_world()
    sys.bodies["b1"].extra.update({
        "ring": True, "cmf": 0.38, "composition_preset": "硅酸盐 + 铁核",
        "atm": {"gases": [["二氧化碳 CO₂", 91.3], ["氮气 N₂", 8.7]],
                "tau0": 0.32, "p0_bar": 0.32, "n": 2.0, "S": 0.2, "nabla": 0.0},
    })
    sys.bodies["S1"].extra["teff_k"] = 6474.11
    back = world_mod.load_world(world_mod.save_world(sys, tmp_path / "extras"))
    b1 = back.bodies["b1"]
    assert b1.extra["cmf"] == pytest.approx(0.38)
    assert b1.extra["ring"] is True
    assert b1.extra["composition_preset"] == "硅酸盐 + 铁核"
    assert b1.extra["atm"]["gases"][0] == ["二氧化碳 CO₂", 91.3]
    assert b1.extra["atm"]["tau0"] == pytest.approx(0.32)
    # MIST outputs are not duplicated in the save's extra payload
    assert "teff_k" not in back.bodies["S1"].extra
    assert back.bodies["S1"].extra["age0_gyr"] == pytest.approx(1.26)
