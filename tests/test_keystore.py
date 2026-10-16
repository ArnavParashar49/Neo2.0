"""NEO's own key store (~/.neo/api_keys.json) and saved brains (~/.neo/brains.json): private files,
written whole or not at all, one writer at a time; a file NEO can't read is set aside only when the
user next saves; the real ones are never touched while effects are off. Every key is a fake, and
every file is the test's own."""

from __future__ import annotations

import json
import os
import stat
import threading
from pathlib import Path

import pytest

from neo2.core import config, keystore
from neo2.core.keystore import FileKeyStore, SavedConfig, StoredKey

KEY = "sk-ant-TEST-0000-abcd"
GEM = "AIzaTEST-gemini-0001"


def _mode(p: Path) -> int:
    return stat.S_IMODE(os.stat(p).st_mode)


def test_a_key_round_trips_and_the_file_is_private(tmp_path):
    folder = tmp_path / "neo"
    store = FileKeyStore(folder / "api_keys.json")
    assert store.list() == {} and store.state().ok and not store.state().exists  # nothing made by reading
    assert not folder.exists()
    store.put("anthropic", KEY)
    store.put("gemini", GEM, extra=["AIzaTEST-gemini-0002", GEM, "AIzaTEST-gemini-0002"])
    got = store.list()
    assert got["anthropic"].key == KEY and got["anthropic"].extra == ()
    assert got["gemini"].key == GEM and got["gemini"].extra == ("AIzaTEST-gemini-0002",)  # (once, never the key itself)
    assert got["anthropic"].saved.endswith("Z") and "T" in got["anthropic"].saved
    assert _mode(folder / "api_keys.json") == 0o600 and _mode(folder) == 0o700
    data = json.loads((folder / "api_keys.json").read_text())
    assert data["version"] == 1 and set(data["keys"]) == {"anthropic", "gemini"}
    assert data["keys"]["anthropic"] == {"key": KEY, "saved": got["anthropic"].saved}
    # Replaced, then removed; the others stay.
    store.put("anthropic", "sk-ant-TEST-1111-efgh")
    assert store.get("anthropic").key == "sk-ant-TEST-1111-efgh" and store.get("gemini").key == GEM
    assert store.remove("anthropic") is True and store.remove("anthropic") is False
    assert set(store.list()) == {"gemini"}
    # No temporary file is left behind, and the lock file is private too.
    names = sorted(p.name for p in folder.iterdir())
    assert names == [".store.lock", "api_keys.json"] and _mode(folder / ".store.lock") == 0o600


def test_custom_endpoints_keep_their_keys_by_id(tmp_path):
    store = FileKeyStore(tmp_path / "api_keys.json")
    store.put("custom:deepseek", "sk-TEST-deepseek-0000")
    assert store.get("custom:deepseek").key == "sk-TEST-deepseek-0000"
    for bad in ("custom:Bad Slug", "", "../x", "custom:", "a" * 40):
        with pytest.raises(ValueError):
            store.put(bad, KEY)


def test_a_pasted_key_is_cleaned_or_refused_without_saying_it():
    assert keystore.clean_key('  "sk-ant-TEST-0000"  ') == "sk-ant-TEST-0000"
    assert keystore.clean_key("'AIzaTEST-1'\n") == "AIzaTEST-1"
    for bad in ("", "   ", "sk-ant-TEST\n0000", "sk-ant-TEST 0000", "sk-ant-TEST\x00", "k" * 513):
        with pytest.raises(ValueError) as e:
            keystore.clean_key(bad)
        assert "TEST" not in str(e.value) and "kkkk" not in str(e.value)
    assert len(keystore.clean_key("k" * 512)) == 512


def test_only_the_last_four_ever_show():
    assert keystore.last4("sk-ant-TEST-x9Qa") == "x9Qa" and keystore.mask("sk-ant-TEST-x9Qa") == "••••x9Qa"
    assert keystore.last4("short-key") is None and keystore.mask("short-key") == "••••"  # under 12: nothing of it
    assert keystore.mask("") == "" and keystore.last4(None) is None
    k = StoredKey(KEY, ("AIzaTEST-extra-0000",), "2026-10-16T09:12:00Z")
    assert KEY not in repr(k) and "TEST" not in repr(k) and "abcd" in repr(k)
    assert "TEST" not in repr(FileKeyStore(Path("/x/api_keys.json")))


def test_a_file_it_cant_read_is_empty_and_set_aside_only_on_save(tmp_path):
    path = tmp_path / "api_keys.json"
    junk = b'{"version": 1, "keys": {"anthropic": {"key": 12}}}'
    path.write_bytes(junk)
    store = FileKeyStore(path)
    assert store.list() == {} and store.state().problem == "corrupt"
    assert store.remove("anthropic") is False and path.read_bytes() == junk  # removing never rewrites it
    assert path.read_bytes() == junk  # reading never touched it
    store.put("openai", "sk-proj-TEST-0000")
    bad = [p for p in tmp_path.iterdir() if p.name.startswith("api_keys.json.bad-")]
    assert len(bad) == 1 and bad[0].read_bytes() == junk  # the old file, kept as it was
    assert set(store.list()) == {"openai"} and store.state().ok
    # Not JSON at all, a newer NEO's file, a file this user can't read.
    for body, problem in ((b"\x00\xffnot json", "corrupt"), (b'{"version": 2, "keys": {}}', "newer"), (b"[1]", "corrupt")):
        path.write_bytes(body)
        assert store.state().problem == problem and store.list() == {}
    path.write_text('{"version": 1, "keys": {}}')
    os.chmod(path, 0)
    try:
        if os.geteuid() != 0:
            assert store.state().problem == "unreadable" and not store.state().ok
    finally:
        os.chmod(path, 0o600)


def test_a_write_that_fails_leaves_the_old_file_whole(tmp_path, monkeypatch):
    path = tmp_path / "api_keys.json"
    store = FileKeyStore(path)
    store.put("anthropic", KEY)
    before = path.read_bytes()

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(keystore.os, "replace", boom)
    with pytest.raises(OSError):
        store.put("openai", "sk-proj-TEST-0000")
    monkeypatch.undo()
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == [".store.lock", "api_keys.json"]  # no temp file left


def test_one_writer_at_a_time(tmp_path, monkeypatch):
    monkeypatch.setattr(keystore, "_LOCK_S", 0.2)
    store = FileKeyStore(tmp_path / "api_keys.json")
    errors: list[BaseException] = []
    with keystore.locked(tmp_path):

        def save():
            try:
                store.put("anthropic", KEY)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        t = threading.Thread(target=save)
        t.start()
        t.join(5)
    assert len(errors) == 1 and "busy" in str(errors[0]) and not (tmp_path / "api_keys.json").exists()
    store.put("anthropic", KEY)  # free again
    assert store.get("anthropic").key == KEY


def test_the_real_files_are_never_touched_while_effects_are_off(monkeypatch):
    """Not read, not written, not even opened: refused by their name before anything on disk."""
    monkeypatch.setenv("NEO_NO_EFFECTS", "1")

    def touched(*a, **kw):
        raise AssertionError("the real store was touched")

    monkeypatch.setattr(keystore, "_read_json", touched)
    monkeypatch.setattr(keystore, "atomic_write", touched)
    monkeypatch.setattr(keystore, "locked", touched)
    for name in ("api_keys.json", "brains.json"):
        for spelt in (Path.home() / ".neo" / name, Path("~/.neo") / name, Path.home() / ".NEO" / name):
            assert keystore.is_real(spelt) and keystore.refused(spelt)
            store, saved = FileKeyStore(spelt), SavedConfig(spelt)
            for call in (store.list, store.state, lambda: store.put("anthropic", KEY), lambda: store.remove("x_y"),
                         saved.read, lambda: saved.update({"brain": "claude"})):  # fmt: skip
                with pytest.raises(RuntimeError, match="effects are off"):
                    call()
    # …and the settings never read them either.
    monkeypatch.setattr(config, "KEY_STORE_PATH", Path.home() / ".neo" / "api_keys.json")
    monkeypatch.setattr(config, "SAVED_PATH", Path.home() / ".neo" / "brains.json")
    assert config._store_sources(config.Settings) == ()
    assert config.Settings().anthropic_api_key == ""


def test_the_suite_never_reads_the_real_files_even_as_neo_app(monkeypatch):
    """A test that runs as NEO.app (effects on for its thread) still never gets the real store."""
    from neo2.core import effects

    monkeypatch.delenv("NEO_NO_EFFECTS", raising=False)
    token = effects._as_app.set(True)
    try:
        assert effects.why_off() == ""
        assert keystore.refused(Path.home() / ".neo" / "api_keys.json")
    finally:
        effects._as_app.reset(token)


def test_saved_brains_round_trip_and_keep_only_what_they_know(tmp_path):
    path = tmp_path / "brains.json"
    saved = SavedConfig(path)
    assert saved.read() == ({}, saved.state())
    now = saved.update({"brain": "claude@claude-opus-5-5", "backup_brains": None, "daily_budgets": {"anthropic": 2.0}})
    assert now == {"brain": "claude@claude-opus-5-5", "backup_brains": None, "daily_budgets": {"anthropic": 2.0}}
    assert _mode(path) == 0o600
    saved.update({"voice": "local"}, remove=["daily_budgets"])
    assert saved.read()[0] == {"brain": "claude@claude-opus-5-5", "backup_brains": None, "voice": "local"}
    assert json.loads(path.read_text())["version"] == 1
    with pytest.raises(ValueError):
        saved.update({"gemini_api_key": KEY})  # never a key, never a setting it doesn't keep
    with pytest.raises(ValueError):
        saved.update({"voice": "loud"})
    # A value of the wrong kind in the file (an older or edited one) is ignored, field by field.
    path.write_text(json.dumps({"version": 1, "brain": 3, "voice": "loud", "cloud_eyes": "off", "secret": "x"}))
    assert saved.read()[0] == {"cloud_eyes": "off"}
    # A file it can't read is set aside on the next save.
    path.write_text("{oops")
    assert saved.read()[0] == {} and saved.state().problem == "corrupt"
    saved.update({"brain": "gemini"})
    assert saved.read()[0] == {"brain": "gemini"}
    assert any(p.name.startswith("brains.json.bad-") for p in tmp_path.iterdir())
