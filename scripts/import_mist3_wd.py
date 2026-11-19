"""Import MIST III WD cooling tracks from default_grids_full.tgz.

We use Tracks/*.data rather than the contour files: each track carries M_in,
M_WD, log_cool_age AND log_tot_age, so O₂xygen can keep its existing scientific
inputs (initial mass, Fe/H, TOTAL age) without a second cooling-age clock.

Only the alpha=0, Fe/H=0/+0.25 grids are packed, matching the existing MIST
interpolation bracket. The 879 MB source archive is not needed at runtime.
"""
import re
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / ".data" / "MIST-III" / "default_grids_full.tgz"
OUT = ROOT / ".data" / "mist3_wd.npz"
TABLES = {"feh_p000_afe_p0_wds_vvcrit0.4": 0.0,
          "feh_p025_afe_p0_wds_vvcrit0.4": 0.25}


def main():
    out = {"fehs": np.array(list(TABLES.values()))}
    with tarfile.open(SRC, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        for fi, (tag, feh) in enumerate(TABLES.items()):
            prefix = f"default_grids/{tag}/Tracks/"
            tracks = [m for m in members if m.name.startswith(prefix)
                      and m.name.endswith(".data")]
            minis, mwds, tables = [], [], []
            for j, m in enumerate(tracks):
                raw = tf.extractfile(m).read()
                head = raw[:120].decode("ascii", errors="replace")
                mini = float(re.search(r"M_in\s*=\s*([0-9.]+)", head).group(1))
                mwd = float(re.search(r"M_WD\s*=\s*([0-9.]+)", head).group(1))
                data = np.loadtxt(raw.splitlines())
                # log_cool_age, log_tot_age, log_L, log_Teff, log_R
                table = data[:, :5]
                table = table[np.argsort(table[:, 1])]
                tables.append(table)
                minis.append(mini)
                mwds.append(mwd)
            order = np.argsort(minis)
            # Reorder metadata and tables into initial-mass order.
            for new_j, old_j in enumerate(order):
                out[f"f{fi}_t{new_j:03d}"] = tables[old_j]
            out[f"f{fi}_minis"] = np.asarray(minis)[order]
            out[f"f{fi}_mwds"] = np.asarray(mwds)[order]
            out[f"f{fi}_n"] = np.array(len(tracks))
            print(f"{tag}: {len(tracks)} tracks")
    np.savez_compressed(OUT, **out)
    print(f"-> {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
