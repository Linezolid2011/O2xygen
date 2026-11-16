"""One-off repair of the 白玫瑰 save (2026-11-16 audit).

Two damages repaired:
  1. DATA LOSS: pre-fix world.py dropped every `extra` key on save, so the
     live save lost cmf / ring / atmosphere / composition metadata. Restored
     from the built-in preset (the values' original source).
  2. DRIFT: the pre-fix preset never zeroed barycentric momentum; the world
     had drifted ~93 AU off-origin. A uniform velocity shift is a Galilean
     transform - relative states and all dynamics are untouched.

Usage: .venv/Scripts/python scripts/repair_world.py <src.oxygen> <dst.oxygen>
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "server"))

from o2xygen import world as world_mod
from preset import build_preset, zero_barycentric_velocity


def main(src: Path, dst: Path) -> None:
    world = world_mod.load_world(src)
    preset = build_preset()
    restored = []
    for name, b in world.bodies.items():
        pb = preset.bodies.get(name)
        if pb is None:
            continue
        for k, v in pb.extra.items():
            if k not in b.extra:
                b.extra[k] = v
                restored.append(f"{name}.{k}")
    zero_barycentric_velocity(world)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out = world_mod.save_world(world, dst)
    print(f"restored extras: {', '.join(restored) or '(none missing)'}")
    print(f"barycentric velocity zeroed; saved -> {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
