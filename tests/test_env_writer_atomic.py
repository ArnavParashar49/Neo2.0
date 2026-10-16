"""The settings NEO writes into ~/.neo/.env itself (NEO_UI_HOST, NEO_WAKE_WORD,
NEO_WELLBEING_CHECKINS) are written whole or not at all: a temporary file renamed over the old one,
so a crash or a full disk never leaves the user's keys half-written. A file that is there but can't
be read stops the write (it would otherwise be replaced by the one line). The byte-exact and 0600
assertions of tests/test_wake_word_setting.py and test_v2_ws_security.py still hold as they are."""

from __future__ import annotations

import os
import stat

import pytest

from neo2.core import config, keystore

THEIRS = "# my keys\nGEMINI_API_KEY=AIzaTEST-keep-me\n\nNEO_LISTEN_TIMEOUT_S = 7\n"


def _mode(p) -> int:
    return stat.S_IMODE(os.stat(p).st_mode)


def test_a_write_replaces_the_file_whole(tmp_path):
    (tmp_path / "home").mkdir()
    env = tmp_path / "home" / ".env"
    env.write_text(THEIRS)
    os.chmod(env, 0o644)
    before = os.stat(env).st_ino
    config.set_wake_word(False, env_path=env)
    assert env.read_text() == THEIRS + "\nNEO_WAKE_WORD=false\n"
    assert os.stat(env).st_ino != before  # a new file, renamed over the old one
    assert _mode(env) == 0o600
    config.set_wellbeing_checkins(True, env_path=env)
    config.set_ui_host("notch", env_path=env)
    assert env.read_text() == THEIRS + "\nNEO_WAKE_WORD=false\n\nNEO_WELLBEING_CHECKINS=true\n\nNEO_UI_HOST=notch\n"
    assert sorted(p.name for p in env.parent.iterdir()) == [".env"]  # no temporary file left


def test_a_failed_write_leaves_the_users_file_as_it_was(tmp_path, monkeypatch):
    (tmp_path / "home").mkdir()
    env = tmp_path / "home" / ".env"
    env.write_text(THEIRS)

    def full(*a, **kw):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(keystore.os, "replace", full)
    with pytest.raises(OSError):
        config.set_wake_word(False, env_path=env)
    assert env.read_text() == THEIRS and sorted(p.name for p in env.parent.iterdir()) == [".env"]


def test_a_file_that_cant_be_read_stops_the_write(tmp_path):
    env = tmp_path / ".env"
    env.write_text(THEIRS)
    os.chmod(env, 0)
    try:
        if os.geteuid() != 0:
            with pytest.raises(PermissionError):
                config.set_wake_word(False, env_path=env)
    finally:
        os.chmod(env, 0o600)
    assert env.read_text() == THEIRS  # never replaced by the one line
    folder = tmp_path / "dir.env"
    folder.mkdir()
    with pytest.raises(IsADirectoryError):
        config.set_ui_host("notch", env_path=folder)
    # No file yet is no lines: it is made, private.
    fresh = tmp_path / "new" / ".env"
    config.set_wellbeing_checkins(False, env_path=fresh)
    assert fresh.read_text() == "NEO_WELLBEING_CHECKINS=false\n" and _mode(fresh) == 0o600


def test_a_linked_env_file_stays_linked(tmp_path):
    real = tmp_path / "dotfiles" / "neo.env"
    real.parent.mkdir()
    real.write_text(THEIRS)
    link = tmp_path / ".env"
    link.symlink_to(real)
    config.set_wake_word(True, env_path=link)
    assert link.is_symlink() and os.readlink(link) == str(real)
    assert real.read_text() == THEIRS + "\nNEO_WAKE_WORD=true\n"


def test_the_real_file_is_still_refused_while_effects_are_off(monkeypatch):
    monkeypatch.setenv("NEO_NO_EFFECTS", "1")
    for set_it, value in ((config.set_wake_word, True), (config.set_wellbeing_checkins, False), (config.set_ui_host, "notch")):
        with pytest.raises(RuntimeError):
            set_it(value)
