"""Unified stellar lifecycle: MIST -> MIST III WD -> WDCool only by mass.

WDCool is selected ONLY when the WD's current mass is outside MIST III's
WD-mass coverage. Age overrun alone never selects WDCool (ADR-0028).
"""
from __future__ import annotations


def evaluate_stellar(mini_msun: float, feh: float, age_yr: float,
                     mist, wd, cool, extra: dict | None = None):
    extra = extra or {}
    wd_mass = extra.get("wd_mass_msun")
    if wd is not None and wd.is_wd_age(mini_msun, feh, age_yr):
        if cool is not None and wd_mass is not None and \
                not wd.wd_mass_covered(wd_mass, feh):
            # cooling age is derived from the single total-age clock
            formation = wd.formation_age_yr(mini_msun, feh) \
                if wd.initial_mass_covered(mini_msun) \
                else float(extra.get("wd_formation_age_gyr", 0.0)) * 1e9
            return cool.evaluate(wd_mass, max(0.0, age_yr - formation),
                                 extra.get("wd_envelope", "thick"))
        return wd.evaluate(mini_msun, feh, age_yr)
    # Initial mass itself may be off MIST III. In that case an explicit
    # WDCool mass + formation age identifies an off-grid WD; otherwise the
    # caller's ordinary MIST track remains in use.
    if cool is not None and wd_mass is not None \
            and "wd_formation_age_gyr" in extra:
        formation = float(extra["wd_formation_age_gyr"]) * 1e9
        if age_yr >= formation:
            return cool.evaluate(wd_mass, age_yr - formation,
                                 extra.get("wd_envelope", "thick"))
    if mist is None:
        raise ValueError("MIST grid not ready")
    return mist.evaluate(mini_msun, feh, age_yr)
