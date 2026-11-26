"""Non-stellar display colour is a saved input, not a physics output.

Normalized sRGB triplets are persisted in Body.extra["color_rgb"]. Stellar
colours are never duplicated there: they are derived from current Teff.
"""
from __future__ import annotations

import math

DEFAULT_BODY_RGB = [0.42, 0.55, 0.72]


def normalize_color_rgb(value) -> list[float]:
    if value is None:
        return list(DEFAULT_BODY_RGB)
    try:
        rgb = [float(x) for x in value]
    except (TypeError, ValueError) as e:
        raise ValueError("color_rgb must contain three finite numbers in 0..1") from e
    if len(rgb) != 3 or not all(math.isfinite(x) and 0.0 <= x <= 1.0 for x in rgb):
        raise ValueError("color_rgb must contain three finite numbers in 0..1")
    return rgb
