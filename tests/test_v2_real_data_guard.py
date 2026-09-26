"""Under pytest, NEO refuses to touch the user's real ~/.neo — even from a harness that skipped
the isolation fixture."""

from pathlib import Path

import pytest


def test_settings_refuse_the_real_data_dir_under_pytest(monkeypatch):
    from neo import config

    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setenv("NEO_DATA_DIR", str(Path.home() / ".neo"))
    with pytest.raises(RuntimeError, match="refusing to use the real"):
        config.settings()
    monkeypatch.setenv("NEO_DATA_DIR", str(Path.home() / ".neo" / "sub"))
    with pytest.raises(RuntimeError, match="refusing to use the real"):
        config.settings()
    monkeypatch.setattr(config, "_settings", None)


def test_an_isolated_dir_is_fine(tmp_path, monkeypatch):
    from neo import config

    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setenv("NEO_DATA_DIR", str(tmp_path / "d"))
    assert config.settings().data_dir == (tmp_path / "d").resolve()
