"""Migrate legacy .data/worlds/<name>.oxygen to data/worlds/<name>/<name.oxygen.

Default is a dry run.  --apply backs up each source, moves it, then verifies
SHA-256.  Existing different destination files are never overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".data" / "worlds"
DEST = ROOT / "data" / "worlds"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main(apply=False):
    sources = sorted(SOURCE.glob("*.oxygen")) if SOURCE.is_dir() else []
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = ROOT / ".data" / "backups" / f"layout-migration-{stamp}"
    moved = []
    for src in sources:
        name = src.stem
        dst = DEST / name / src.name
        if dst.exists() and sha256(src) != sha256(dst):
            raise FileExistsError(f"destination differs; refusing overwrite: {dst}")
        moved.append({"source": str(src), "destination": str(dst),
                      "backup": str(backup / src.name), "apply": apply})
        if apply:
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, backup / src.name)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                src.unlink()                 # identical content already verified
            else:
                shutil.move(str(src), str(dst))
            assert sha256(backup / src.name) == sha256(dst)
    print(json.dumps({"legacy_root": str(SOURCE), "new_root": str(DEST),
                      "worlds": moved, "count": len(moved)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="back up, migrate and verify (default: dry run)")
    main(parser.parse_args().apply)
