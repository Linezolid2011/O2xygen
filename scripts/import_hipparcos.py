"""Import the copied hip2000 raw catalog into a compact runtime NPZ.

Input:  .data/hip2000/hip_main.dat  (Vizier I/239 layout, J1991.25)
Output: .data/hipparcos.npz         (only fields needed by the sky service)

The generated hipparcos_*.js tables are J2000; the raw DAT is J1991.25. We keep
the raw observation epoch plus proper motions so arbitrary epochs remain
possible. Radial velocity is unavailable and is treated as zero.
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / ".data" / "hip2000" / "hip_main.dat"
OUT = ROOT / ".data" / "hipparcos.npz"


def main():
    rows = []
    with open(SRC, encoding="ascii", errors="replace") as f:
        for line in f:
            p = line.split("|")
            if len(p) < 40:
                continue
            try:
                plx = float(p[11])
                if plx <= 0.2:          # nonpositive/noisy >5 kpc entries
                    continue
                rows.append((int(p[1]), float(p[8]), float(p[9]), plx,
                             float(p[12]), float(p[13]), float(p[5]),
                             float(p[37]) if p[37].strip() else np.nan))
            except ValueError:
                pass
    a = np.asarray(rows, dtype=float)
    np.savez_compressed(
        OUT,
        hip=a[:, 0].astype(np.int64), ra_deg=a[:, 1], dec_deg=a[:, 2],
        plx_mas=a[:, 3], pmra_mas_yr=a[:, 4], pmde_mas_yr=a[:, 5],
        vmag=a[:, 6], bv=a[:, 7], epoch_jyear=np.array(1991.25),
        source=np.array("Hipparcos I/239 via gmiller123456/hip2000; RV=0"),
    )
    print(f"{len(rows)} stars -> {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
