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
        assert rb.radius_m == pytest.approx(b.radius_m, rel=1e-12)
        if b.kind == STAR:
            assert rb.extra["feh"] == pytest.approx(b.extra["feh"])
            assert rb.extra["age0_gyr"] == pytest.approx(b.extra["age0_gyr"])
        else:
            assert rb.albedo == pytest.approx(b.albedo)


def test_rejects_foreign_format(tmp_path):
    p = tmp_path / "bad.oxygen"
    p.write_text('{"format": "something-else", "version": 1}', encoding="utf-8")
    with pytest.raises(ValueError):
        world_mod.load_world(p)


def test_suffix_normalised(tmp_path):
    sys = sample_world()
    path = world_mod.save_world(sys, tmp_path / "noext")
    assert path.name == "noext.oxygen"
