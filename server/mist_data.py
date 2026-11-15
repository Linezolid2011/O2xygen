"""MIST data bootstrap (user scheme, 2026-11-15):

  .data/MIST-compact/   chunked tarballs tracked in git (each volume < 64 MB,
                        under GitHub's 100 MB per-file limit)
  .data/MIST/           extracted tracks + parse cache (gitignored, created
                        on first boot, kept across runs; compact volumes are
                        NOT deleted after extraction)

Extraction is pure Python (tarfile + xz), since Windows' built-in tar
lacks an xz filter. Reassembly streams parts in name order into a temp
file inside .data/MIST, extracts, then removes the temp.
"""
from __future__ import annotations

import shutil
import tarfile
from pathlib import Path

CHUNK_SIZE = 64 * 1024 * 1024        # 64 MB per volume
_TRACKS = ("feh_p0.00", "feh_p0.25") # extracted dirs the app needs


def write_parts(src: Path, dest_dir: Path,
                chunk_size: int = CHUNK_SIZE) -> list[Path]:
    """Split src into dest_dir/<name>.partNN volumes (runtime-free helper,
    used once when bundling the data; also exercised by tests)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    with open(src, "rb") as f:
        i = 1
        while True:
            blob = f.read(chunk_size)
            if not blob:
                break
            p = dest_dir / f"{src.name}.part{i:02d}"
            p.write_bytes(blob)
            parts.append(p)
            i += 1
    return parts


def _archives_for(compact: Path, tag: str) -> list[Path]:
    return sorted(p for p in compact.glob("*.part*") if tag in p.name)


def _reassemble(parts: list[Path], dest: Path) -> None:
    with open(dest, "wb") as out:
        for part in parts:
            with open(part, "rb") as f:
                shutil.copyfileobj(f, out, 1024 * 1024)


def ensure_extracted(repo_root: Path, log=print) -> Path:
    """Make sure .data/MIST holds the extracted tracks; extract from the
    compact volumes if any required set is missing."""
    compact = repo_root / ".data" / "MIST-compact"
    target = repo_root / ".data" / "MIST"
    have = {d.name for d in target.glob("feh_p*") if d.is_dir()} \
        if target.exists() else set()
    missing = [t for t in _TRACKS if t not in have]
    if not missing:
        return target
    target.mkdir(parents=True, exist_ok=True)
    for tag in missing:
        parts = _archives_for(compact, tag)
        if not parts:
            raise FileNotFoundError(
                f"MIST tracks for {tag} not found under {compact} "
                f"(expected *.partNN volumes)")
        tmp = target / f".reassembling-{tag}"
        log(f"[mist] reassembling {tag} from {len(parts)} volumes ...")
        _reassemble(parts, tmp)
        log(f"[mist] extracting {tag} ...")
        with tarfile.open(tmp, "r:xz") as tf:
            tf.extractall(target, filter="fully_trusted")
        tmp.unlink()
        # the tarball wraps tracks in its own top-level directory
        # (MIST_v1.2_feh_p0.00_..._EEPS) - normalise to <target>/<tag>
        inner = next(target.glob(f"MIST_v1.2_{tag}*"), None)
        if inner is not None and inner.is_dir():
            dest = target / tag
            if dest.exists():
                shutil.rmtree(dest)
            inner.rename(dest)
        log(f"[mist] {tag} ready (compact volumes retained)")
    return target


def grid_dirs(extracted_root: Path) -> dict[float, Path]:
    return {0.0: extracted_root / "feh_p0.00",
            0.25: extracted_root / "feh_p0.25"}
