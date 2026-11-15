"""P2 star module: validate MIST interpolation against his displayed star.

Screenshot vector (参数估算结果 modal):
  当前质量 1.2899 M☉, 初始质量 1.29 M☉, 累计损失 0.00013 M☉ (0.01%),
  当前损失率 1.146e-13 M☉/yr,
  模型球形半径 926,278.85 km, 光度 2.8055 L☉, 有效温度 6,474.11 K.
The star's AGE is not visible -> recover it from (L, Teff), then check that
radius and current mass fall out consistently. That triangulates his whole
star pipeline (grid choice, interpolation, evolution application).
"""
from pathlib import Path
import pytest

MIST_DIR = Path(r"D:\test3\data\mist")
HAVE_DATA = (MIST_DIR / "feh_p0.00").is_dir()

pytestmark = pytest.mark.skipif(not HAVE_DATA, reason="MIST grids not downloaded yet")


@pytest.fixture(scope="module")
def grid():
    from o2xygen.stars import MistGrid
    dirs = {0.0: MIST_DIR / "feh_p0.00"}
    p25 = MIST_DIR / "feh_p0.25"
    if p25.is_dir():
        dirs[0.25] = p25
    return MistGrid(dirs)


def test_solve_age_then_check_radius_and_mass(grid):
    """BRACKET CLOSED with the p0.25 grid: his modal lists nodes
    "1.28/1.3 M☉ x [Fe/H] 0/0.25", and indeed the solution sits INSIDE the
    metallicity bracket: at [Fe/H] ~ 0.10, M = 1.29 M☉, age ~ 1.26 Gyr his
    whole triplet (L, Teff, R) is reproduced (log-residual ~ 6e-4)."""
    from o2xygen.constants import L_SUN, R_SUN
    import numpy as np

    tgt = np.array([np.log10(2.8055), np.log10(6474.11),
                    np.log10(926278.85 * 1000 / R_SUN)])
    results = {}
    for feh in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25):
        age = grid.solve_age(1.29, feh, luminosity_lsun=2.8055, teff_k=6474.11)
        p = grid.evaluate(1.29, feh, age)
        got = np.array([np.log10(p.luminosity_w / L_SUN), np.log10(p.teff_k),
                        np.log10(p.radius_m / R_SUN)])
        results[feh] = (float(np.sqrt(np.sum((got - tgt) ** 2))), p, age)

    best_feh = min(results, key=lambda f: results[f][0])
    assert best_feh in (0.05, 0.10, 0.15)      # interior of the bracket
    resid, p, age = results[best_feh]
    assert resid < 2e-3
    assert p.radius_m / 1000.0 == pytest.approx(926278.85, rel=0.01)
    assert p.luminosity_w / L_SUN == pytest.approx(2.8055, rel=5e-3)
    assert p.teff_k == pytest.approx(6474.11, rel=5e-3)
    # current mass after wind: his 演化后当前质量 1.28987
    assert p.mass_msun == pytest.approx(1.28987, abs=1e-3)


def test_mass_loss_scale(grid):
    """His 当前损失率 1.146e-13 M☉/yr, from |star_mdot| at the solution.
    MESA reports dM/dt (negative on the MS); raw track at the bracket gives
    ~9.5e-14, same order."""
    p = grid.evaluate(1.29, 0.10, 1.256e9)
    assert p.mdot_msun_per_yr == pytest.approx(1.146e-13, rel=0.35)
    assert p.mdot_msun_per_yr > 0


def test_solar_age_sun(grid):
    """Sanity: 1.0 M☉, [Fe/H]=0 at 4.56 Gyr should look like the Sun."""
    p = grid.evaluate(1.0, 0.0, 4.56e9)
    assert p.teff_k == pytest.approx(5772.0, rel=0.03)
    assert p.radius_m == pytest.approx(6.957e8, rel=0.05)


def test_evaluate_rejects_off_grid_inputs(grid):
    """Audit findings A3/B8: absurd initial mass silently returned degenerate
    track-end values; negative age was silently clipped to 1 Myr. Both must
    raise now, mirroring the [Fe/H] check."""
    with pytest.raises(ValueError):
        grid.evaluate(1.0e6, 0.0, 4.0e9)      # mass far above grid max
    with pytest.raises(ValueError):
        grid.evaluate(0.01, 0.0, 4.0e9)       # mass below grid minimum (~0.1)
    with pytest.raises(ValueError):
        grid.evaluate(1.0, 0.0, -5.0)         # negative age (was clipped)
    with pytest.raises(ValueError):
        grid.evaluate(1.0, 0.0, 1.0e12)       # far above AGE_MAX (was clipped)
