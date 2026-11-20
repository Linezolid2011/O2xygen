"""Import Bédard et al. (2020) WDCool sequences.tar into a compact NPZ.

Each seq_NNN_{thick,thin}.txt file is one fixed-mass cooling sequence with
three-line records: Teff/logg/R/Age/L on the first line. R is in cm and L in
erg/s, age in years. Runtime evaluation uses mass + cooling age + envelope.
"""
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT.parent / "sequences.tar"
OUT = ROOT / ".data" / "wdcool.npz"


def parse_sequence(raw: bytes) -> np.ndarray:
    numeric = []
    for line in raw.decode("ascii", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or set(s) <= set("= "):
            continue
        parts = s.split()
        try:
            numeric.append([float(x) for x in parts])
        except ValueError:
            pass
    rows = []
    for i in range(0, len(numeric) - 2, 3):
        a = numeric[i]
        rows.append([a[1], a[2], a[3], a[4], a[5]])  # Teff, logg, R, Age, L
    return np.asarray(rows, dtype=float)


def main():
    out = {}
    with tarfile.open(SRC, "r:") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            stem = Path(m.name).stem             # seq_120_thick
            _, mass_s, envelope = stem.split("_")
            mass = int(mass_s) / 100.0
            table = parse_sequence(tf.extractfile(m).read())
            table = table[np.argsort(table[:, 3])]
            out[f"{envelope}_{int(mass_s):03d}"] = table
        for envelope in ("thick", "thin"):
            out[f"masses_{envelope}"] = np.arange(0.20, 1.31, 0.05)
    np.savez_compressed(OUT, **out)
    print(f"46 sequences -> {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
