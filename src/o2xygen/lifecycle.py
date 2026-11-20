"""Unified stellar lifecycle: MIST -> MIST III WD -> WDCool only by mass.

WDCool is selected ONLY when the WD's current mass is outside MIST III's
WD-mass coverage. Age overrun alone never selects WDCool (ADR-0028).
"""
from __future__ import annotations


def evaluate_stellar(mini_msun: float, feh: float, age_yr: float,
                     mist, wd, cool, extra: dict | None = None):
    extra = extra or {}
    # The IFMR handoff comes from the LAST point of the MIST grid, not from
    # an independently editable WD mass. Explicit extra fields are only a
    # last-resort fallback when the ordinary MIST grid is unavailable.
    final = mist.final_point(mini_msun, feh) \
        if mist is not None and hasattr(mist, "final_point") else None
    wd_mass = final["mass_msun"] if final is not None \
        else extra.get("wd_mass_msun")
    formation = final["age_yr"] if final is not None else None
    if formation is None and wd is not None and wd.initial_mass_covered(mini_msun):
        formation = wd.formation_age_yr(mini_msun, feh)
    if formation is None and "wd_formation_age_gyr" in extra:
        formation = float(extra["wd_formation_age_gyr"]) * 1e9
    if wd is not None and wd.is_wd_age(mini_msun, feh, age_yr):
        if cool is not None and wd_mass is not None and formation is not None \
                and not wd.wd_mass_covered(wd_mass, feh):
            return cool.evaluate(wd_mass, max(0.0, age_yr - formation),
                                 extra.get("wd_envelope", "thick"))
        return wd.evaluate(mini_msun, feh, age_yr)
    # Initial mass itself may be off MIST III. Use the MIST final point when
    # available, or explicit off-grid inputs if that is all we have.
    if cool is not None and wd_mass is not None and formation is not None \
            and not wd.wd_mass_covered(wd_mass, feh) and age_yr >= formation:
        return cool.evaluate(wd_mass, age_yr - formation,
                             extra.get("wd_envelope", "thick"))
    if mist is None:
        raise ValueError("MIST grid not ready")
    return mist.evaluate(mini_msun, feh, age_yr)
