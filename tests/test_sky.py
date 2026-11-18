"""Real-sky numerical anchors (J2000 Hipparcos + synthetic Sun)."""
from pathlib import Path

import numpy as np
import pytest

from o2xygen.sky import HipparcosSky, SUN_ABS_VMAG

CATALOG = Path(__file__).resolve().parents[1] / ".data" / "hipparcos.npz"


@pytest.fixture(scope="module")
def sky():
    if not CATALOG.exists():
        pytest.skip("run scripts/import_hipparcos.py first")
    return HipparcosSky(CATALOG)


def test_sirius_j2000_anchor(sky):
    got = sky.observe((0.0, 0.0, 0.0), 2000.0, mlim=6.5)
    i = int(np.flatnonzero(got["hip"] == 32349)[0])
    x, y, z = got["dir_icrs"][i]
    ra = np.arctan2(y, x) * 180.0 / np.pi % 360.0
    dec = np.arcsin(z) * 180.0 / np.pi
    assert ra == pytest.approx(101.28715539, abs=1e-8)
    assert dec == pytest.approx(-16.71611582, abs=1e-8)
    assert got["mag"][i] == pytest.approx(-1.46, abs=0.1)


def test_sun_obeys_the_same_magnitude_cutoff(sky):
    # At 100 pc the Sun is V=9.83: hidden at mlim=9, visible at mlim=10.
    hidden = sky.observe((100.0, 0.0, 0.0), 2000.0, mlim=9.0)
    assert hidden["sun_index"] is None
    got = sky.observe((100.0, 0.0, 0.0), 2000.0, mlim=10.0)
    i = got["sun_index"]
    assert i is not None and got["hip"][i] == 0
    assert np.allclose(got["dir_icrs"][i], [-1.0, 0.0, 0.0], atol=1e-15)
    assert got["dist_pc"][i] == pytest.approx(100.0)
    assert got["mag"][i] == pytest.approx(SUN_ABS_VMAG + 5.0)


def test_sirius_reprojects_from_100pc(sky):
    u = np.array([np.cos(np.deg2rad(-16.71611582)) * np.cos(np.deg2rad(101.28715539)),
                  np.cos(np.deg2rad(-16.71611582)) * np.sin(np.deg2rad(101.28715539)),
                  np.sin(np.deg2rad(-16.71611582))])
    got = sky.observe(u * 100.0, 2000.0, mlim=12.0)
    i = int(np.flatnonzero(got["hip"] == 32349)[0])
    assert np.dot(got["dir_icrs"][i], u) == pytest.approx(-1.0, abs=1e-12)
    # Hipparcos parallax ~379 mas -> Sirius is 2.64 pc out along the same ray
    assert got["dist_pc"][i] == pytest.approx(97.36, abs=0.2)
