"""Unified stellar lifecycle: MIST -> MIST III WD -> WDCool only by mass.

WDCool is selected ONLY when the WD's current mass is outside MIST III's
WD-mass coverage. Age overrun alone never selects WDCool (ADR-0028).
"""
from __future__ import annotations


def resolve_stellar(mini_msun: float, feh: float, age_yr: float,
                    mist, wd, cool, extra: dict | None = None) -> dict:
    """Return the model choice and its derived time/mass inputs, without
    evaluating any track. Used identically by live evolution and preview."""
    extra = extra or {}
    # IFMR handoff comes from the LAST MIST point, not an independently
    # editable WD mass. Explicit extras are only for a missing MIST grid.
    final = mist.final_point(mini_msun, feh) \
        if mist is not None and hasattr(mist, "final_point") else None
    wd_mass = final["mass_msun"] if final is not None \
        else extra.get("wd_mass_msun")
    formation = final["age_yr"] if final is not None else None
    if formation is None and wd is not None and wd.initial_mass_covered(mini_msun):
        formation = wd.formation_age_yr(mini_msun, feh)
    if formation is None and "wd_formation_age_gyr" in extra:
        formation = float(extra["wd_formation_age_gyr"]) * 1e9
    covered = wd is not None and wd.wd_mass_covered(wd_mass, feh) \
        if wd_mass is not None else False
    use_cool = cool is not None and wd_mass is not None \
        and formation is not None and not covered
    model = "mist"
    if wd is not None and wd.is_wd_age(mini_msun, feh, age_yr):
        model = "wdcool" if use_cool else "mist3_wd"
    elif use_cool and age_yr >= formation:
        model = "wdcool"
    return {"model": model, "wd_mass_msun": wd_mass,
            "formation_age_yr": formation, "covered_by_mist3": covered,
            "cooling_age_yr": max(0.0, age_yr - formation)
            if formation is not None else None}


def segments_with_wd_stage(segments: list[dict], formation_age_yr: float | None) -> list[dict]:
    """Append/truncate the display track with a WD stage after formation.
    This is a view of the lifecycle, never a second age input."""
    if formation_age_yr is None or not segments:
        return segments
    import numpy as np
    logform = float(np.log10(formation_age_yr))
    end = max(max(s["end_logage"] for s in segments),
              np.log10(200.0 * 1e9))  # editor's 200 Gyr display horizon
    if logform >= end:
        return segments
    out = []
    for s in segments:
        if s["end_logage"] <= logform:
            out.append(dict(s))
        elif s["start_logage"] < logform:
            out.append({**s, "end_logage": logform})
    out.append({"start_logage": logform, "end_logage": end,
                "key": "wd", "label": "白矮星", "color": "#c9d6ff"})
    return out


def evaluate_stellar(mini_msun: float, feh: float, age_yr: float,
                     mist, wd, cool, extra: dict | None = None):
    r = resolve_stellar(mini_msun, feh, age_yr, mist, wd, cool, extra)
    if r["model"] == "wdcool":
        return cool.evaluate(r["wd_mass_msun"], r["cooling_age_yr"],
                             (extra or {}).get("wd_envelope", "thick"))
    if r["model"] == "mist3_wd":
        return wd.evaluate(mini_msun, feh, age_yr)
    if mist is None:
        raise ValueError("MIST grid not ready")
    return mist.evaluate(mini_msun, feh, age_yr)
