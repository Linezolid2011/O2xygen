"""Verify which hip2000 files are J1991.25 and which are J2000.

The repository contains BOTH: hip_main.dat is the raw I/239 catalog at its
observation epoch J1991.25; the generated hipparcos_*.js files have been run
through NOVAS transform_hip to J2000. This script proves the distinction by
advancing the raw positions by 8.75 Julian years and comparing the converted
JS, without importing the old skycalc project.
"""
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / ".data" / "hip2000"
MAS2RAD = math.pi / (180.0 * 3600.0 * 1000.0)
DEG = math.pi / 180.0


def load_raw():
    rows = []
    with open(DATA / "hip_main.dat", encoding="ascii", errors="replace") as f:
        for line in f:
            p = line.split("|")
            if len(p) < 14:
                continue
            try:
                rows.append((int(p[1]), float(p[8]), float(p[9]),
                             float(p[11]), float(p[12]), float(p[13])))
            except ValueError:
                pass
    return {r[0]: np.array(r[1:], dtype=float) for r in rows}


def advance(row, years=8.75):
    ra0, dec0, plx, pmra, pmde = row
    dist = 1000.0 / plx
    ra, dec = ra0 * DEG, dec0 * DEG
    u = np.array([math.cos(dec) * math.cos(ra),
                  math.cos(dec) * math.sin(ra), math.sin(dec)])
    ea = np.array([-math.sin(ra), math.cos(ra), 0.0])
    ed = np.array([-math.sin(dec) * math.cos(ra),
                   -math.sin(dec) * math.sin(ra), math.cos(dec)])
    vel = dist * MAS2RAD * (pmra * ea + pmde * ed)   # pc/yr, RV=0
    r = dist * u + vel * years
    ra1 = math.atan2(r[1], r[0]) / DEG % 360.0
    dec1 = math.asin(r[2] / np.linalg.norm(r)) / DEG
    return ra1, dec1


def sep_arcsec(ra1, dec1, ra2, dec2):
    dra = (ra1 - ra2 + 180.0) % 360.0 - 180.0
    return math.hypot(dra * math.cos((dec1 + dec2) * 0.5 * DEG),
                      dec1 - dec2) * 3600.0


def main():
    raw = load_raw()
    text = (DATA / "hipparcos_full_concise.js").read_text(encoding="ascii")
    payload = text[text.find("["):text.rfind("]") + 1]
    payload = re.sub(r",\s*\]", "]", payload)  # JS generator's trailing commas
    converted = json.loads(payload)
    js = {int(r[0]): (float(r[2]), float(r[3])) for r in converted}
    direct, advanced = [], []
    for hip, (ra_js, dec_js) in js.items():
        if hip not in raw or raw[hip][2] <= 0.2:
            continue
        ra0, dec0 = raw[hip][:2]
        ra1, dec1 = advance(raw[hip])
        direct.append(sep_arcsec(ra0, dec0, ra_js, dec_js))
        advanced.append(sep_arcsec(ra1, dec1, ra_js, dec_js))
    print(f"common stars checked: {len(advanced)}")
    print(f"raw -> JS directly:       median {np.median(direct):9.4f}\"")
    print(f"raw + 8.75 yr -> JS/J2000: median {np.median(advanced):9.4f}\", "
          f"max {max(advanced):9.4f}\"")
    for name, hip in [("Sirius", 32349), ("Vega", 91262),
                      ("Polaris", 11767), ("Barnard's Star", 87937)]:
        if hip not in raw or hip not in js:
            continue
        ra0, dec0 = raw[hip][:2]
        ra1, dec1 = advance(raw[hip])
        print(f"{name:15s} raw=({ra0:12.8f},{dec0:12.8f})  "
              f"advanced=({ra1:12.8f},{dec1:12.8f})  "
              f"JS=({js[hip][0]:12.8f},{js[hip][1]:12.8f})  "
              f"residual={sep_arcsec(ra1, dec1, *js[hip]):.4f}\"")
    assert np.median(advanced) < 0.02, "raw epoch advance does not match JS"
    assert np.median(direct) > 20 * np.median(advanced), "no epoch distinction"
    print("VERIFIED: hip_main.dat = J1991.25; generated JS = J2000.")


if __name__ == "__main__":
    main()
