"""Mass-radius relations for rocky/icy bodies.

His structure tab offers two models:
  * "高压 EOS 表 · Zeng (2019)": 0.01-100 M_E, core fraction 0-1 (full EOS grid)
  * "PREM": 1-8 M_E, core fraction 0-0.4  -> this is exactly the semi-empirical
    PREM-based fit of Zeng, Sasselov & Jacobsen 2016 (ApJ 819:127):
        R/R_E = (1.07 - 0.21*CMF) * (M/M_E)^(1/3.7)

The 2019 EOS grid proper needs Zeng's high-pressure EOS tables (to be wired in
later); meanwhile the same CMF-interpolation form, extended, serves as the
Zeng(2019)-class approximation. Reference-level accuracy, not bit-exact.
"""
from .constants import M_EARTH, R_EARTH_MEAN

# PREM fit validity window (his UI: 1-8 M_E, CMF 0-0.4)
PREM_M_MIN, PREM_M_MAX = 1.0, 8.0
PREM_CMF_MIN, PREM_CMF_MAX = 0.0, 0.4

# 2019-class window (his UI)
ZENG19_M_MIN, ZENG19_M_MAX = 0.01, 100.0
ZENG19_CMF_MIN, ZENG19_CMF_MAX = 0.0, 1.0

_CMFCOEFF_A = 1.07
_CMFCOEFF_B = 0.21
_EXP_PREM = 1.0 / 3.7
# extended-regime exponent flattens toward 1/4 at high mass (Zeng+2019 grid
# behaves this way); used only outside the PREM window
_EXP_ZENG19 = 0.25


def _model_windows(model: str) -> tuple[float, float, float, float, float]:
    """(mass_exp, m_min, m_max, cmf_min, cmf_max) for a model name."""
    if model == "prem":
        return _EXP_PREM, PREM_M_MIN, PREM_M_MAX, PREM_CMF_MIN, PREM_CMF_MAX
    if model == "zeng2019":
        return _EXP_ZENG19, ZENG19_M_MIN, ZENG19_M_MAX, ZENG19_CMF_MIN, ZENG19_CMF_MAX
    raise ValueError(model)


def radius_earth_r(mass_earth: float, core_fraction: float,
                   model: str = "prem") -> float:
    """Radius in Earth radii (mean-radius convention)."""
    exp, m_min, m_max, c_min, c_max = _model_windows(model)
    # negative mass with a fractional exponent would silently produce a
    # complex radius; reject non-positive mass outright
    if mass_earth <= 0.0:
        raise ValueError(f"mass {mass_earth} must be positive")
    if not (m_min <= mass_earth <= m_max):
        raise ValueError(f"mass {mass_earth} outside {model} range "
                         f"[{m_min}, {m_max}] M_E")
    if not (c_min <= core_fraction <= c_max):
        raise ValueError(f"CMF {core_fraction} outside {model} range "
                         f"[{c_min}, {c_max}]")
    return (_CMFCOEFF_A - _CMFCOEFF_B * core_fraction) * mass_earth ** exp


def infer_core_fraction(mass_earth: float, radius_earth_r: float,
                        model: str = "prem") -> float:
    """Invert the relation: CMF from (M, R)."""
    exp, m_min, m_max, _, _ = _model_windows(model)   # unknown model raises
    if mass_earth <= 0.0:
        raise ValueError(f"mass {mass_earth} must be positive")
    if not (m_min <= mass_earth <= m_max):
        raise ValueError(f"mass {mass_earth} outside {model} range "
                         f"[{m_min}, {m_max}] M_E")
    cmf = (_CMFCOEFF_A - radius_earth_r / mass_earth ** exp) / _CMFCOEFF_B
    return cmf


def radius_m(mass_kg: float, core_fraction: float, model: str = "prem") -> float:
    return radius_earth_r(mass_kg / M_EARTH, core_fraction, model) * R_EARTH_MEAN
