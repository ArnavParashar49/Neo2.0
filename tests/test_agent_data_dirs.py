"""An embedded agent's folders are its own: never NEO.app's ~/.neo however it's spelled, never a
folder that holds ~/.neo (the home folder, "/"), never another live agent's, and never a folder
full of someone else's files."""

from __future__ import annotations

import os
import pwd
from pathlib import Path

import pytest
from neo_lib import make_runtime
from pydantic import ValidationError

from neo import config
from neo.config import LibrarySettings

_DATA = Path("/System/Volumes/Data")
NEVER = "never uses NEO.app's own data"


def _via_data_volume(p: Path) -> Path:
    """The same folder reached through the APFS data volume (/Users is a firmlink to it)."""
    alias = Path(str(_DATA) + str(p.resolve()))
    if not alias.parent.exists():
        pytest.skip("no APFS data volume here")
    return alias


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    (h / ".neo").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(h))
    return h


def test_neo_home_through_the_data_volume_is_refused(home):
    with pytest.raises(ValidationError, match=NEVER):
        LibrarySettings(data_dir=home / ".neo" / "agent")
    alias = _via_data_volume(home / ".neo") / "agent"
    with pytest.raises(ValidationError, match=NEVER):
        LibrarySettings(data_dir=alias)
    assert not (home / ".neo" / "agent").exists()


def test_neo_home_through_the_data_volume_before_it_exists(home):
    (home / ".neo").rmdir()
    alias = _via_data_volume(home) / ".neo" / "agent"
    with pytest.raises(ValidationError, match=NEVER):
        LibrarySettings(data_dir=alias)
    with pytest.raises(ValidationError, match=NEVER):
        LibrarySettings(data_dir=_via_data_volume(home) / ".NEO")
    assert not (home / ".neo").exists()


def test_neo_home_through_a_symlink_is_refused(home, tmp_path):
    (tmp_path / "innocent").symlink_to(home / ".neo")
    with pytest.raises(ValidationError, match=NEVER):
        LibrarySettings(data_dir=tmp_path / "innocent" / "agent")


def test_a_folder_holding_neo_home_is_refused(home):
    for d in (home, home.parent, Path("/"), "~", "~/..", _via_data_volume(home)):
        with pytest.raises(ValidationError, match=NEVER):
            LibrarySettings(data_dir=d)


def test_the_login_home_counts_even_when_home_is_moved(tmp_path, monkeypatch):
    """A host that points HOME elsewhere still never reaches the user's own ~/.neo."""
    real = tmp_path / "login-home"
    (real / ".neo").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path / "sandbox"))
    me = pwd.getpwuid(os.getuid())
    monkeypatch.setattr(pwd, "getpwuid", lambda uid: type(me)((*me[:5], str(real), *me[6:])))
    for d in (real / ".neo" / "agent", real):
        with pytest.raises(ValidationError, match=NEVER):
            LibrarySettings(data_dir=d)
    with pytest.raises(RuntimeError, match="refusing to use the real"):
        config._refuse_real_data_without_effects(real / ".neo")


def test_models_never_go_into_neo_home(home, tmp_path):
    for m in (home / ".neo" / "models", _via_data_volume(home / ".neo") / "models", home):
        with pytest.raises(ValidationError, match=NEVER):
            LibrarySettings(data_dir=tmp_path / "a", models_path=m, embeddings="local")
    assert not (home / ".neo" / "models").exists()


def test_the_pytest_guard_sees_the_data_volume_too(home):
    with pytest.raises(RuntimeError, match="refusing to use the real"):
        config._refuse_real_data_without_effects(_via_data_volume(home / ".neo"))
    with pytest.raises(RuntimeError, match="refusing to use the real"):
        config._refuse_real_data_without_effects(home)  # a data dir holding ~/.neo


def test_a_folder_of_someone_elses_files_is_refused(tmp_path):
    docs = tmp_path / "Documents"
    docs.mkdir()
    (docs / "taxes.pdf").write_text("x")
    with pytest.raises(ValidationError, match="its own folder"):
        LibrarySettings(data_dir=docs)
    # missing, or empty (Finder's .DS_Store aside): NEO's from then on
    for d in (tmp_path / "new", tmp_path / "empty"):
        if d.name == "empty":
            d.mkdir()
            (d / ".DS_Store").write_text("")
        s = LibrarySettings(data_dir=d)
        s.ensure_dirs()
        (d / "memory.sqlite3").write_text("")
        assert LibrarySettings(data_dir=d).data_dir == d.resolve()  # its own again next time


def test_two_live_agents_never_share_a_data_dir(tmp_path):
    a = make_runtime(tmp_path / "shared", "a")
    with pytest.raises(ValueError, match="another agent"):
        make_runtime(tmp_path / "shared", "b")
    with pytest.raises(ValueError, match="another agent"):
        make_runtime(tmp_path / "shared" / "inner", "c")
    with pytest.raises(ValueError, match="another agent"):
        make_runtime(_via_data_volume(tmp_path / "shared"), "d")
    b = make_runtime(tmp_path / "other", "b")
    assert b.settings.data_dir != a.settings.data_dir
    a.close()
    assert make_runtime(tmp_path / "shared", "e").settings.audit_log == a.settings.audit_log  # a is gone
