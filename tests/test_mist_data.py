"""MIST bundling: chunked volumes in .data/MIST-compact -> extracted
tracks in .data/MIST on first boot (user scheme, 2026-11-15)."""
import io
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from mist_data import write_parts, ensure_extracted, grid_dirs


def _make_txz(path: Path, member_name: str, payload: bytes) -> Path:
    with tarfile.open(path, "w:xz") as tf:
        info = tarfile.TarInfo(member_name)
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    return path


def test_write_and_reassemble_roundtrip(tmp_path):
    import os
    src = _make_txz(tmp_path / "demo.txz", "demo/track.eep",
                    os.urandom(200_000))       # incompressible -> real chunking
    parts = write_parts(src, tmp_path / "compact", chunk_size=2048)
    assert len(parts) >= 2
    assert all(p.stat().st_size <= 2048 for p in parts)

    from mist_data import _reassemble, _archives_for
    found = _archives_for(tmp_path / "compact", "demo")
    assert found == parts
    out = tmp_path / "demo.reassembled.txz"
    _reassemble(found, out)
    assert out.read_bytes() == src.read_bytes()


def test_ensure_extracted_extracts_missing_tracks(tmp_path):
    compact = tmp_path / ".data" / "MIST-compact"
    compact.mkdir(parents=True)
    for tag in ("feh_p0.00", "feh_p0.25"):
        src = _make_txz(tmp_path / f"{tag}.txz",
                        f"{tag}/MIST_x/00100M.track.eep", b"track-bytes")
        write_parts(src, compact, chunk_size=1024)

    target = ensure_extracted(tmp_path, log=lambda *a: None)
    for tag in ("feh_p0.00", "feh_p0.25"):
        track = next((target / tag).rglob("*.track.eep"))
        assert track.read_bytes() == b"track-bytes"
    # compact volumes are RETAINED, not deleted
    assert list(compact.glob("*.part*"))
    # extraction is idempotent: second call finds the dirs, reuses them
    target2 = ensure_extracted(tmp_path, log=lambda *a: None)
    assert target2 == target


def test_ensure_extracted_raises_without_volumes(tmp_path):
    (tmp_path / ".data" / "MIST-compact").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        ensure_extracted(tmp_path, log=lambda *a: None)


def test_grid_dirs_maps_feh():
    dirs = grid_dirs(Path("/x"))
    assert dirs[0.0] == Path("/x/feh_p0.00")
    assert dirs[0.25] == Path("/x/feh_p0.25")
