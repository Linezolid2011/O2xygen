"""ADR-0037: data/worlds/<name>/<name.oxygen layout and save isolation."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from o2xygen import world as world_mod
from workspace import Workspace


def test_bootstrap_list_save_and_load_nested_world(tmp_path, monkeypatch):
    import app
    monkeypatch.setattr(app, "WORLDS_DIR", tmp_path)
    app._bootstrap_worlds()
    path = app._world_path("白玫瑰")
    assert path == tmp_path / "白玫瑰" / "白玫瑰.oxygen"
    assert path.is_file()
    assert app._world_names() == ["白玫瑰"]
    world = app._load_system("白玫瑰")
    monkeypatch.setattr(app, "SYSTEM", world)
    monkeypatch.setattr(app, "WS", Workspace(world, grid_getter=None))
    monkeypatch.setattr(app, "CURRENT_WORLD", "白玫瑰")
    monkeypatch.setattr(app, "_evolution_active", lambda: False)

    got = app.api_world_save({"name": "第二世界"})
    assert got["ok"] is True
    second = app._world_path("第二世界")
    assert second.is_file()
    assert world_mod.load_world(second).name == world.name
    assert app.api_list_worlds() == {"worlds": sorted(["第二世界", "白玫瑰"]),
                                     "current": "白玫瑰"}


def test_flat_legacy_file_is_not_a_second_world_truth(tmp_path, monkeypatch):
    import app
    from preset import build_preset
    monkeypatch.setattr(app, "WORLDS_DIR", tmp_path)
    world_mod.save_world(build_preset(), tmp_path / "旧世界")
    assert (tmp_path / "旧世界.oxygen").is_file()
    assert app._world_names() == []       # flat files are not listed as worlds
    app._bootstrap_worlds()
    assert app._world_names() == ["白玫瑰"]
    assert app._world_path("白玫瑰").is_file()


def test_world_path_rejects_traversal(tmp_path, monkeypatch):
    import app
    monkeypatch.setattr(app, "WORLDS_DIR", tmp_path)
    with pytest.raises(app.HttpError) as exc:
        app._world_path("../escape")
    assert exc.value.status == 400
