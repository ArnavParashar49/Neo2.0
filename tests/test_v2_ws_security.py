"""The core's websocket admits only NEO's overlay (token + origin) — not other web pages."""

from __future__ import annotations

import asyncio
import json
import os
import socket

import websockets

from neo2.core.server import ws as ws_mod


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _server(monkeypatch):
    from neo2.core import config

    port = _free_port()
    monkeypatch.setenv("NEO_WS_PORT", str(port))
    monkeypatch.setattr(config, "_settings", None)
    seen = []

    async def on_command(cmd):
        seen.append(cmd)

    srv = ws_mod.UIServer(on_command)
    await srv.start()
    return srv, port, seen


async def _try(port, *, token=None, origin=None) -> str:
    q = "" if token is None else f"/?token={token}"
    headers = {"Origin": origin} if origin else None
    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}{q}", additional_headers=headers, open_timeout=3) as c:
            await asyncio.wait_for(c.recv(), 3)
            return "open"
    except websockets.exceptions.InvalidStatus as e:
        return str(e.response.status_code)


def test_only_the_overlay_gets_in(monkeypatch):
    async def run():
        srv, port, _ = await _server(monkeypatch)
        tok = ws_mod.ws_token()
        try:
            return {
                "no token": await _try(port),
                "wrong token": await _try(port, token="x" * 43),
                "a web page": await _try(port, token=tok, origin="https://evil.example"),
                "the overlay": await _try(port, token=tok, origin="tauri://localhost"),
                "the dev overlay": await _try(port, token=tok, origin="http://localhost:1420"),
                "the notch": await _try(port, token=tok, origin="neo://app"),  # NotchHub's WKWebView, neo://app/
                "a notch lookalike": await _try(port, token=tok, origin="neo://app.evil"),
                "another neo:// page": await _try(port, token=tok, origin="neo://evil"),
                "the notch with a port": await _try(port, token=tok, origin="neo://app:1"),
                "a local tool": await _try(port, token=tok),  # no Origin: a local process that can read the file
            }
        finally:
            if srv._server:
                srv._server.close()

    got = asyncio.run(run())
    assert got == {
        "no token": "401",
        "wrong token": "401",
        "a web page": "403",
        "the overlay": "open",
        "the dev overlay": "open",
        "the notch": "open",
        "a notch lookalike": "403",
        "another neo:// page": "403",
        "the notch with a port": "403",
        "a local tool": "open",
    }


def test_ui_host_is_a_specific_addition():
    # One origin for the notch: the scheme alone admits nothing.
    assert "neo://app" in ws_mod._ALLOWED_ORIGINS
    assert not any(o.startswith("neo://") and o != "neo://app" for o in ws_mod._ALLOWED_ORIGINS)


def test_ui_host_is_kept_in_the_env_file_and_told_to_clients(monkeypatch, tmp_path):
    """{"cmd":"ui_host"} from NotchHub: NEO_UI_HOST in ~/.neo/.env (the user's keys untouched), the
    live settings, the hello of later clients, and a ui_host event for the ones connected now."""
    from neo2.core import config

    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=keep-me\nNEO_UI_HOST=window\n")
    config.set_ui_host("notch", env_path=env)
    assert env.read_text() == "GEMINI_API_KEY=keep-me\n\nNEO_UI_HOST=notch\n"
    assert oct(os.stat(env).st_mode & 0o777) == "0o600"
    assert config.settings().ui_host == "notch"
    config.set_ui_host("window", env_path=env)
    assert env.read_text().count("NEO_UI_HOST=") == 1 and "NEO_UI_HOST=window" in env.read_text()
    assert config.settings().ui_host == "window"
    # Only the two hosts; and never the user's real file while effects are off.
    import pytest

    with pytest.raises(ValueError):
        config.set_ui_host("desk", env_path=env)
    with pytest.raises(RuntimeError):
        config.set_ui_host("notch")

    async def run():
        srv, port, seen = await _server(monkeypatch)
        tok = ws_mod.ws_token()
        try:
            async with websockets.connect(f"ws://127.0.0.1:{port}/?token={tok}", additional_headers={"Origin": "neo://app"}, open_timeout=3) as c:
                hello = json.loads(await asyncio.wait_for(c.recv(), 3))
                assert hello["type"] == "hello" and hello["data"]["ui_host"] == "window"
                await c.send(json.dumps({"cmd": "ui_host", "host": "notch"}))
                await asyncio.sleep(0.2)
                return seen
        finally:
            if srv._server:
                srv._server.close()

    seen = asyncio.run(run())
    assert seen == [{"cmd": "ui_host", "host": "notch"}]  # reaches the core's command handler


def test_a_wake_by_hand_is_told_to_every_client():
    """{"cmd":"wake"} (the hotkey, the menu-bar item, the orb) publishes `woke` before the voice
    session starts listening: the wake word never does, so NotchHub can open its panel for a
    summon by hand alone. With no voice session the event still goes out."""
    from neo2.core.app import App
    from neo2.core.events import bus

    seen: list = []
    unsub = bus().subscribe(lambda ev: seen.append((ev.type, dict(ev.data))))
    try:
        app = App.__new__(App)
        app.voice = None
        asyncio.run(app.on_command({"cmd": "wake"}))
        assert seen == [("woke", {"source": "ui"})]

        class V:
            woke: list = []

            async def wake(self, source="ui"):
                self.woke.append((source, list(seen)))

        app.voice = V()
        asyncio.run(app.on_command({"cmd": "wake"}))
        # The voice session is woken after the event left, so a client that opens for it sees
        # `listening` arrive second.
        assert V.woke == [("ui", [("woke", {"source": "ui"})] * 2)]
    finally:
        unsub()


def test_token_file_is_private_and_stable():
    a, b = ws_mod.ws_token(), ws_mod.ws_token()
    from neo2.core.config import settings

    path = settings().data_dir / "ws-token"
    assert a == b and len(a) >= 32 and oct(os.stat(path).st_mode & 0o777) == "0o600"
