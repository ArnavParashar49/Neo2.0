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


async def _try(port, *, token=None, origin=None, role=None) -> str:
    q = "" if token is None else f"/?token={token}"
    if role is not None:
        q += f"&role={role}"
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
                # A host (NotchHub's native listener) says so, and has no Origin; a page never is one.
                "a host": await _try(port, token=tok, role="host"),
                "a host without the token": await _try(port, token="x" * 43, role="host"),
                "NEO's page as a host": await _try(port, token=tok, origin="tauri://localhost", role="host"),
                "the notch's page as a host": await _try(port, token=tok, origin="neo://app", role="host"),
                "a role nobody has": await _try(port, token=tok, role="admin"),
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
        "a host": "open",
        "a host without the token": "401",
        "NEO's page as a host": "403",
        "the notch's page as a host": "403",
        "a role nobody has": "403",
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


# ---- NEO.app's menu, for a host ({"cmd":"shell"}) and what it shows ({"cmd":"shell_status"}) --------

APP, PAGE = "tauri://localhost", "neo://app"
STATUS = {"accessibility": True, "screen_recording": False, "open_at_login": True, "version": "2.1.0", "hotkey": "⌥⌘"}


class _Peer:
    """One client of the core: NEO.app's page (origin APP), the notch's page (PAGE), a host
    (role="host", no Origin) or a local tool (neither)."""

    def __init__(self, port, *, origin=None, role=None):
        q = f"/?token={ws_mod.ws_token()}" + (f"&role={role}" if role else "")
        self._url, self._headers = f"ws://127.0.0.1:{port}{q}", ({"Origin": origin} if origin else None)

    async def __aenter__(self):
        self.c = await websockets.connect(self._url, additional_headers=self._headers, open_timeout=3)
        self.hello = (await self.next("hello"))["data"]
        return self

    async def __aexit__(self, *exc):
        await self.c.close()

    async def send(self, **cmd):
        await self.c.send(json.dumps(cmd))

    async def next(self, type_: str) -> dict:
        """The next message of this type (others before it are dropped)."""
        while True:
            msg = json.loads(await asyncio.wait_for(self.c.recv(), 3))
            if msg["type"] == type_:
                return msg

    async def quiet(self) -> list[str]:
        """What arrived by the time the core answered a `state` request: the types, in order."""
        await self.send(cmd="state")
        got = []
        while True:
            msg = json.loads(await asyncio.wait_for(self.c.recv(), 3))
            if msg["type"] == "state":
                return got
            got.append(msg["type"])


def _with_server(monkeypatch, body):
    async def run():
        srv, port, seen = await _server(monkeypatch)
        try:
            return await body(srv, port, seen)
        finally:
            if srv._server:
                srv._server.close()

    return asyncio.run(run())


def test_who_a_client_is():
    kind = ws_mod.client_kind
    assert kind("tauri://localhost", "") == "app" and kind("http://localhost:1420", "") == "app"
    assert kind("neo://app", "") == "page"
    assert kind(None, "host") == "host" and kind(None, "") == "tool"
    # A page is never a host, whichever page; and no role but that one exists.
    assert kind("tauri://localhost", "host") is None and kind("neo://app", "host") is None
    assert kind(None, "admin") is None and kind("https://evil.example", "") is None
    assert ws_mod._ALLOWED_ORIGINS == ws_mod._APP_ORIGINS | {"neo://app"} and "neo://app" not in ws_mod._APP_ORIGINS


def test_a_host_reaches_the_menu_through_neo_apps_own_page(monkeypatch):
    async def body(srv, port, seen):
        async with _Peer(port, origin=APP) as app, _Peer(port, role="host") as host, _Peer(port, origin=PAGE) as page:
            await app.send(cmd="shell_status", **STATUS)
            await host.next("shell_status")  # NEO.app's shell is known: there is someone to do it
            relayed = []
            for cmd in (
                {"action": "show"},
                {"action": "settings"},
                {"action": "open_logs"},
                {"action": "restart"},
                {"action": "erase_memory"},
                {"action": "open_at_login", "on": False},
                {"action": "quit", "on": True, "path": "/etc/passwd"},  # nothing rides along
            ):
                await host.send(cmd="shell", **cmd)
                assert (await host.next("shell_result"))["data"] == {"action": cmd["action"], "ok": True}
                relayed.append((await app.next("shell"))["data"])
            # Only NEO.app's page hears it: not the page in the notch, not the bus, not the replay.
            assert "shell" not in await page.quiet()
            async with _Peer(port, origin=APP) as late:
                assert all(r["type"] != "shell" for r in late.hello["recent"])
            return relayed, list(seen)

    relayed, seen = _with_server(monkeypatch, body)
    assert relayed == [
        {"action": "show"},
        {"action": "settings"},
        {"action": "open_logs"},
        {"action": "restart"},
        {"action": "erase_memory"},
        {"action": "open_at_login", "on": False},
        {"action": "quit"},
    ]
    # The core itself did nothing: erase_memory is NEO.app's question to ask, never a forget_all here.
    assert seen == []


def test_the_menu_is_the_hosts_alone(monkeypatch):
    """quit, restart, erase_memory, open_at_login — and the rest of `shell` — from NEO's own page,
    from the page in the notch, or from a tool that did not say it is a host: refused, and NEO.app's
    page hears nothing."""

    async def body(srv, port, seen):
        refused = {}
        async with _Peer(port, origin=APP) as app:
            await app.send(cmd="shell_status", **STATUS)
            await app.next("shell_status")
            for who, kw in (("app", {"origin": APP}), ("page", {"origin": PAGE}), ("tool", {})):
                async with _Peer(port, **kw) as peer:
                    for action in ws_mod.SHELL_ACTIONS:
                        await peer.send(cmd="shell", action=action, on=True)
                        refused[who, action] = (await peer.next("shell_result"))["data"]
            # NEO.app's page asking itself is refused too (a page that was talked into it).
            await app.send(cmd="shell", action="quit")
            mine = (await app.next("shell_result"))["data"]
            return refused, mine, await app.quiet(), list(seen)

    refused, mine, heard, seen = _with_server(monkeypatch, body)
    assert len(refused) == 3 * len(ws_mod.SHELL_ACTIONS)
    for (_who, action), data in refused.items():
        assert data == {"action": action, "ok": False, "error": "forbidden"}
    assert mine == {"action": "quit", "ok": False, "error": "forbidden"}
    assert "shell" not in heard and seen == []


def test_a_host_asks_only_for_what_the_menu_has(monkeypatch):
    async def body(srv, port, seen):
        async with _Peer(port, role="host") as host:
            out = {}
            # No NEO.app (a core run by hand): there is nobody to do it.
            assert host.hello["shell"] == {"running": True, "app": False, "accessibility": None, "screen_recording": None, "open_at_login": None, "version": None, "hotkey": None}
            await host.send(cmd="shell", action="restart")
            out["no app"] = (await host.next("shell_result"))["data"]
            async with _Peer(port, origin=APP) as app:
                await app.send(cmd="shell_status", **STATUS)
                await host.next("shell_status")
                for name, cmd in (
                    ("unknown", {"action": "rm_rf"}),
                    ("a pane of System Settings", {"action": "allow_ax"}),  # the menu's own; a host opens it itself
                    ("no action", {}),
                    ("login without a choice", {"action": "open_at_login"}),
                    ("login with a word", {"action": "open_at_login", "on": "yes"}),
                ):
                    await host.send(cmd="shell", **cmd)
                    out[name] = (await host.next("shell_result"))["data"]
                out["heard"] = await app.quiet()
            # NEO.app's page left: again nobody.
            await host.next("shell_status")
            await host.send(cmd="shell", action="quit")
            out["app gone"] = (await host.next("shell_result"))["data"]
            return out, list(seen)

    out, seen = _with_server(monkeypatch, body)
    assert out["no app"] == {"action": "restart", "ok": False, "error": "no_app"}
    assert out["unknown"] == {"action": "rm_rf", "ok": False, "error": "unknown"}
    assert out["a pane of System Settings"]["error"] == "unknown" and out["no action"]["error"] == "unknown"
    assert out["login without a choice"]["error"] == "unknown" and out["login with a word"]["error"] == "unknown"
    assert "shell" not in out["heard"]
    assert out["app gone"] == {"action": "quit", "ok": False, "error": "no_app"}
    assert seen == []


def test_only_neo_app_says_what_its_menu_shows(monkeypatch):
    """`shell_status` from NEO.app's own page is kept, told to every client and put in the hello;
    from the page in the notch, a host or a tool it is dropped — a host's Settings never show a
    permission NEO.app did not report."""
    unknown = {"running": True, "app": False, "accessibility": None, "screen_recording": None, "open_at_login": None, "version": None, "hotkey": None}

    async def body(srv, port, seen):
        out = {}
        async with _Peer(port, role="host") as host:
            for who, kw in (("page", {"origin": PAGE}), ("host", {"role": "host"}), ("tool", {})):
                async with _Peer(port, **kw) as peer:
                    await peer.send(cmd="shell_status", **{**STATUS, "screen_recording": True})
                    out[who] = (await peer.quiet(), srv.shell_snapshot())
            async with _Peer(port, origin=APP) as app:
                await app.send(cmd="shell_status", **STATUS)
                out["told"] = (await host.next("shell_status"))["data"]
                async with _Peer(port, origin=PAGE) as late:
                    out["hello"] = late.hello["shell"]
                await app.send(cmd="shell_status", **STATUS)  # the same again: nothing to tell
                await app.send(cmd="shell_status", accessibility="yes", screen_recording=True, open_at_login=True)  # not a status
                out["unchanged"] = (await host.quiet(), srv.shell_snapshot())
                await app.send(cmd="shell_status", **{**STATUS, "screen_recording": True, "version": "9" * 99})
                out["changed"] = (await host.next("shell_status"))["data"]
            out["app left"] = (await host.next("shell_status"))["data"]
        return out, list(seen)

    out, seen = _with_server(monkeypatch, body)
    for who in ("page", "host", "tool"):
        heard, snapshot = out[who]
        assert "shell_status" not in heard and snapshot == unknown, who
    known = {"running": True, "app": True, **STATUS}
    assert out["told"] == known and out["hello"] == known
    assert out["unchanged"] == ([], known)
    assert out["changed"] == {**known, "screen_recording": True, "version": "9" * 32}
    assert out["app left"] == unknown
    assert seen == []  # never the core's command handler


def test_neo_app_knows_whether_a_host_is_there(monkeypatch):
    """NEO.app hides its menu-bar item only while a host is connected: the hello says so, and a
    `host_attached` event follows the host coming and going (a page in the notch is not one)."""

    async def body(srv, port, seen):
        async with _Peer(port, origin=APP) as app:
            out = {"alone": app.hello["host_attached"]}
            async with _Peer(port, origin=PAGE) as page:
                out["with the notch's page"] = (page.hello["host_attached"], "host_attached" in await app.quiet())
                async with _Peer(port, role="host") as host:
                    out["host's own hello"] = host.hello["host_attached"]
                    out["came"] = (await app.next("host_attached"))["data"]
                    async with _Peer(port, role="host") as second:
                        out["a second host"] = (second.hello["host_attached"], "host_attached" in await app.quiet())
                    while sum(cl.kind == "host" for cl in srv._clients.values()) > 1:
                        await asyncio.sleep(0.01)  # the core has seen the second one go
                    out["one of two left"] = "host_attached" in await app.quiet()
                    async with _Peer(port, origin=APP) as late:
                        out["late hello"] = late.hello["host_attached"]
                out["went"] = (await app.next("host_attached"))["data"]
            return out

    out = _with_server(monkeypatch, body)
    assert out == {
        "alone": False,
        "with the notch's page": (False, False),
        "host's own hello": True,
        "came": {"attached": True},
        "a second host": (True, False),
        "one of two left": False,
        "late hello": True,
        "went": {"attached": False},
    }
