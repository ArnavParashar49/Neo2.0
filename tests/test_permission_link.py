"""Setup's microphone and camera rows (neo2/core/permissions.py, the shell's socket): NEO.app's shell asks
the core over their own socket — `{"t":"permission","what":…,"op":"status"|"ask"}` — and the core,
which is the process that uses them, says what macOS says or asks macOS (its prompt is NEO.app's)
and answers on the same socket. Nothing else can be asked about. AVFoundation is faked: no prompt."""

from __future__ import annotations

import asyncio

import pytest

from neo2.core import permissions
from tests.test_v2_ws_security import _Shell, _with_server


class _Device:
    """AVCaptureDevice as the test wants it: a status per media type, and the user's answer."""

    status = {"soun": 0, "vide": 3}
    answer: bool | None = True
    asked: list[str] = []

    @classmethod
    def authorizationStatusForMediaType_(cls, media):
        return cls.status[media]

    @classmethod
    def requestAccessForMediaType_completionHandler_(cls, media, done):
        cls.asked.append(media)
        if cls.answer is not None:
            done(cls.answer)


@pytest.fixture(autouse=True)
def _device(monkeypatch):
    _Device.status = {"soun": 0, "vide": 3}
    _Device.answer = True
    _Device.asked = []
    monkeypatch.setattr(permissions, "_capture_device", lambda: _Device)
    monkeypatch.setattr(permissions.sys, "platform", "darwin")


def test_what_macos_says_in_the_shells_words(monkeypatch):
    assert permissions.media_status("microphone") == "not_determined"
    assert permissions.media_status("camera") == "granted"
    _Device.status = {"soun": 2, "vide": 1}
    assert (permissions.media_status("microphone"), permissions.media_status("camera")) == ("denied", "restricted")
    _Device.status = {"soun": 9, "vide": 3}
    assert permissions.media_status("microphone") == "unknown"
    with pytest.raises(ValueError):
        permissions.media_status("screen")
    monkeypatch.setattr(permissions.sys, "platform", "linux")
    assert permissions.media_status("camera") == "unknown"
    assert set(permissions.STATUSES) == {"granted", "denied", "restricted", "not_determined", "unknown"}


def test_asking_prompts_only_when_macos_never_asked():
    assert asyncio.run(permissions.media("microphone")) == "not_determined" and _Device.asked == []  # status only
    assert asyncio.run(permissions.media("microphone", ask=True)) == "granted" and _Device.asked == ["soun"]
    _Device.answer = False
    assert asyncio.run(permissions.media("microphone", ask=True)) == "denied"
    assert asyncio.run(permissions.media("camera", ask=True)) == "granted" and _Device.asked == ["soun", "soun"]
    _Device.answer = None  # the user hasn't answered: the shell hears what macOS says then
    assert asyncio.run(permissions.media("microphone", ask=True, timeout_s=0.05)) == "not_determined"


def test_the_shell_asks_and_hears_the_answer_on_its_own_socket(monkeypatch):
    async def body(srv, port, seen):
        async with _Shell(srv, monkeypatch) as shell:
            await shell.say(t="permission", what="screen", op="ask")  # nothing but the two is asked about
            await shell.say(t="permission", what="microphone", op="nonsense")
            await shell.say(t="permission", what="camera", op="status")
            first = await shell.read()
            await shell.say(t="permission", what="microphone", op="ask")
            second = await shell.read()
            await shell.say(t="permission", what="microphone")  # (no op: the status)
            third = await shell.read()
            return first, second, third

    first, second, third = _with_server(monkeypatch, body)
    assert first == {"t": "permission", "what": "camera", "status": "granted"}
    assert second == {"t": "permission", "what": "microphone", "status": "granted"} and _Device.asked == ["soun"]
    assert third == {"t": "permission", "what": "microphone", "status": "not_determined"}  # (the fake keeps saying 0)


def test_the_link_says_it_both_ways():
    from pathlib import Path

    doc = (Path(__file__).resolve().parents[1] / "neo2" / "core" / "server" / "shell_link.py").read_text()
    assert '{"t":"permission","what":"microphone",' in doc and '"op":"status"|"ask"}' in doc
    rust = (Path(__file__).resolve().parents[1] / "ui" / "src-tauri" / "src" / "core.rs").read_text()
    assert 'PERMISSION_WHATS: [&str; 2] = ["microphone", "camera"]' in rust
    for status in permissions.STATUSES:
        assert f'"{status}"' in rust.split("const PERMISSION_STATUSES")[1].split("\n")[0], status
