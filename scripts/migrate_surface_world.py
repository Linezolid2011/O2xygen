"""One-time, backup-first v3 surface migration for the 白玫瑰 world.

Dry-run by default. Use --apply for the explicitly requested Νεφέλη inputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from o2xygen.atmosphere import default_atmosphere
from o2xygen.world import load_world, save_world


def nephele_atmosphere():
    a = default_atmosphere()
    a.update({
        "surface_model": "cloud_deck", "height_m": 180_000.0,
        "scale_height_m": 28_000.0, "surface_pressure_pa": 101_325.0,
        "surface_temperature_k": 280.0, "mean_molecular_weight": 2.32,
        "composition": {"H2": 0.90, "He": 0.10},
        "rayleigh_scattering_m_inv": [3.5e-6, 8.0e-6, 18.0e-6],
        "aerosol_scattering_m_inv": [2.2e-6, 2.0e-6, 1.8e-6],
        "absorption_m_inv": [2.0e-7, 8.0e-7, 3.0e-8],
        "aerosol_scale_height_m": 12_000.0, "aerosol_anisotropy": 0.45,
        "ground_albedo": [0.22, 0.27, 0.30],
        "infrared_optical_depth": 1.6, "exposure": 1.0,
    })
    return a


def apply_migration(path: Path, apply=False):
    sys_world = load_world(path)
    b = sys_world.bodies["Νεφέλη"]
    b.extra["atmosphere"] = nephele_atmosphere()
    point = {"name": "Νεφέλη 赤道", "body": "Νεφέλη",
             "latitude_deg": 0.0, "longitude_deg": 0.0}
    sys_world.observation_points = [
        p for p in sys_world.observation_points if p["name"] != point["name"]]
    sys_world.observation_points.append(point)
    if not apply:
        print(f"DRY-RUN {path}: Νεφέλη atmosphere + {point['name']} (0°N, 0°E)")
        return
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = ROOT / ".data" / "backups" / f"surface-v3-{stamp}" / path.name
    backup.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(path, backup)
    save_world(sys_world, path)
    check = load_world(path)
    assert check.bodies["Νεφέλη"].extra["atmosphere"]["surface_model"] == "cloud_deck"
    assert check.observation_points[-1]["longitude_deg"] == 0.0
    print(f"saved {path}; backup {backup}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=Path,
                        default=ROOT / "data" / "worlds" / "白玫瑰" / "白玫瑰.oxygen")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    apply_migration(args.world, args.apply)
