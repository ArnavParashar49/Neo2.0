"""Brains & Keys on the core's websocket: only NEO.app's own page with the shell's ticket is the `keys`
client (one at a time, revoked when the window closes); every other kind is refused its commands; a
result reaches only the window; every client hears the `brains` summary — never a key or its last 4;
`open_keys` reaches NEO.app's shell (one every 5 s), `brains_reload` re-reads NEO's files. The
server, the shell and the files are this test's own; provider checks are faked; keys are fakes."""

from __future__ import annotations

import asyncio
import json

import pytest
import websockets

from neo2.core import brains, config, events, health, logredact
from neo2.core import providers as prov
from neo2.core.providers import keycheck
from neo2.core.server import ws as ws_mod
from tests.test_v2_ws_security import APP, PAGE, _Peer, _Shell, _with_server

TICKET = "ab" * 32
KEY = "sk-ant-TEST-0000-abcdefghijklmnop-x9Qa"
_VARS = ("GEMINI_API_KEY", "GEMINI_API_KEYS", "GROQ_API_KEY", "ANTHROPIC_API_KEY", "CEREBRAS_API_KEY", "OPENAI_API_KEY",
         "OPENROUTER_API_KEY", "NEO_BRAIN", "NEO_FAST_BRAIN", "NEO_EYES_BRAIN", "NEO_CUSTOM_PROVIDERS")  # fmt: skip


@pytest.fixture(autouse=True)
def _own_files(monkeypatch, tmp_path):
    for k in _VARS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(config, "KEY_STORE_PATH", tmp_path / "home" / "api_keys.json")
    monkeypatch.setattr(config, "SAVED_PATH", tmp_path / "home" / "brains.json")
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(prov, "_cache", {})
    monkeypatch.setattr(prov, "_demoted", {})
    monkeypatch.setattr(health, "_live", {})
    monkeypatch.setattr(brains, "_probes", {})
    monkeypatch.setattr(brains, "_started", None)
    monkeypatch.setattr(logredact, "_remembered", set())
    monkeypatch.setattr(events, "_bus", events.EventBus())

    async def check(provider, key=None, endpoint=None):
        return keycheck.KeyCheck(True, "ok", 200, "The key works (api.anthropic.com).", 3)

    monkeypatch.setattr(keycheck, "check", check)


class _Keys(_Peer):
    """The Brains & Keys window: NEO.app's page (origin APP) with the shell's ticket."""

    def __init__(self, port, ticket=TICKET, *, origin=APP, role=None):
        super().__init__(port, origin=origin, role=role)
        self._url += f"&ticket={ticket}"
        self.heard: list[dict] = []  # what came while it waited for an answer

    async def ask(self, **cmd) -> dict:
        await self.send(**cmd)
        while True:
            msg = json.loads(await asyncio.wait_for(self.c.recv(), 5))
            if msg["type"] in (brains.RESULTS.get(cmd["cmd"]), "keys_refused"):
                return msg
            self.heard.append(msg)


async def _status(port, ticket, *, origin=APP, role=None) -> str:
    try:
        async with _Keys(port, ticket, origin=origin, role=role):
            return "open"
    except websockets.exceptions.InvalidStatus as e:
        return str(e.response.status_code)


def test_only_neo_apps_page_with_the_shells_ticket_is_brains_and_keys(monkeypatch, capsys):
    async def body(srv, port, seen):
        got = {"before the shell made one": await _status(port, TICKET)}
        srv.shell_said({"t": "keys_ticket", "ticket": "not-hex"})  # (not one the shell makes: ignored)
        assert srv._keys_ticket is None
        srv.shell_said({"t": "keys_ticket", "ticket": TICKET})
        got |= {
            "a wrong ticket": await _status(port, "cd" * 32),
            "a short ticket": await _status(port, TICKET[:-2]),
            "the notch's page": await _status(port, TICKET, origin=PAGE),
            "a local tool": await _status(port, TICKET, origin=None),
            "a host": await _status(port, TICKET, origin=None, role="host"),
            "NEO's page as a host": await _status(port, TICKET, role="host"),
            "the window": await _status(port, TICKET),
        }
        async with _Keys(port) as window:
            got["a second window"] = await _status(port, TICKET)
            kinds = [cl.kind for cl in srv._clients.values()]
            hello = window.hello
        async with _Peer(port, origin=APP) as app:  # (NEO.app's overlay: no ticket, still the overlay)
            got["the overlay"] = [cl.kind for cl in srv._clients.values()]
            app_hello = app.hello
        return got, kinds, hello, app_hello

    got, kinds, hello, app_hello = _with_server(monkeypatch, body)
    assert got == {
        "before the shell made one": "403",
        "a wrong ticket": "403",
        "a short ticket": "403",
        "the notch's page": "403",
        "a local tool": "403",
        "a host": "403",
        "NEO's page as a host": "403",
        "the window": "open",
        "a second window": "409",
        "the overlay": ["app"],
    }
    assert kinds == ["keys"]
    assert sorted(hello) == ["brains", "problems", "shell", "state", "ui_host"]  # (no transcript, no cards)
    assert "brains" in app_hello and hello["brains"] == app_hello["brains"]
    out = capsys.readouterr().out
    assert TICKET not in out and "cd" * 32 not in out and "refused: its ticket isn't the shell's" in out


def test_every_other_kind_is_refused_and_the_value_goes_nowhere(monkeypatch, capsys):
    published: list = []

    async def body(srv, port, seen):
        events.bus().subscribe(lambda ev: published.append(json.dumps(ev.to_dict(), default=str)))
        out = {}
        for who, kw in (("app", {"origin": APP}), ("page", {"origin": PAGE}), ("host", {"role": "host"}), ("tool", {})):
            async with _Peer(port, **kw) as peer:
                for cmd in brains.COMMANDS:
                    await peer.send(cmd=cmd, id="r1", provider="anthropic", key=KEY)
                    out[who, cmd] = (await peer.next("keys_refused"))["data"]
        return out, list(seen)

    out, seen = _with_server(monkeypatch, body)
    assert len(out) == 4 * len(brains.COMMANDS)
    for (_who, cmd), data in out.items():
        assert data == {"id": "r1", "cmd": cmd, "error": "forbidden"}
    assert seen == []  # (never the core's command handler)
    assert not (config.KEY_STORE_PATH.exists())
    assert KEY not in " ".join(published) and KEY not in capsys.readouterr().out


def test_a_key_saved_is_answered_to_the_window_alone(monkeypatch, capsys):
    async def body(srv, port, seen):
        srv.shell_said({"t": "keys_ticket", "ticket": TICKET})
        async with _Peer(port, origin=PAGE) as page, _Keys(port) as window:
            saved = await window.ask(cmd="key_save", id="r3", provider="anthropic", key=KEY)
            told = (await page.next("brains"))["data"]
            window_told = next(m["data"] for m in window.heard if m["type"] == "brains")  # (before its answer)
            detail = await window.ask(cmd="brains_get", id="r1")
            # The page heard nothing of the result; what it heard before `state` answers:
            heard = await page.quiet()
            set_roles = await window.ask(cmd="brains_set", id="r6", brain="claude@claude-opus-5-5",
                                         fast_brain="claude@claude-haiku-5-5")  # fmt: skip
            after = (await page.next("brains"))["data"]
        return saved, told, window_told, detail, heard, set_roles, after

    saved, told, window_told, detail, heard, set_roles, after = _with_server(monkeypatch, body)
    assert saved["type"] == "key_save_result" and saved["data"]["ok"] is True and saved["data"]["id"] == "r3"
    assert saved["data"]["key"] == {"set": True, "source": "saved", "last4": "x9Qa"}
    assert told == window_told and told["providers"]["anthropic"] is True
    assert "key_save_result" not in heard and "brains_detail" not in heard
    assert detail["data"]["keys"]["anthropic"]["last4"] == "x9Qa" and detail["data"]["id"] == "r1"
    assert set_roles["data"]["ok"] is True
    assert after["brain"]["label"] == "Claude Opus 5.5" and after["ready"] is True
    for d in (told, after):
        assert "x9Qa" not in json.dumps(d) and KEY not in json.dumps(d)
    out = capsys.readouterr().out
    assert KEY not in out and "[ui] keys: key_save ok" in out


def test_the_window_sends_only_its_own_commands(monkeypatch):
    async def body(srv, port, seen):
        srv.shell_said({"t": "keys_ticket", "ticket": TICKET})
        async with _Keys(port) as window:
            for cmd in ({"cmd": "text", "text": "hi"}, {"cmd": "set", "name": "wake_word", "value": False},
                        {"cmd": "shell", "action": "quit"}, {"cmd": "forget_all"}, {"cmd": "ui_host", "host": "notch"},
                        {"cmd": "confirm", "yes": True, "action_id": "x"}, {"cmd": "browser_hello"}):  # fmt: skip
                await window.send(**cmd)
            heard = await window.quiet()
        await asyncio.sleep(0.1)
        return heard, list(seen)

    heard, seen = _with_server(monkeypatch, body)
    assert seen == [] and "set_result" not in heard and "shell_result" not in heard


def test_the_window_closing_revokes_its_ticket(monkeypatch):
    async def body(srv, port, seen):
        srv.shell_said({"t": "keys_ticket", "ticket": TICKET})
        async with _Keys(port) as window:
            srv.shell_said({"t": "keys_closed"})
            with pytest.raises(websockets.ConnectionClosed):
                while True:
                    await asyncio.wait_for(window.c.recv(), 3)
        return await _status(port, TICKET), srv._keys_ticket

    assert _with_server(monkeypatch, body) == ("403", None)


def test_open_keys_reaches_neo_apps_shell_one_every_five_seconds(monkeypatch):
    async def body(srv, port, seen):
        out = {}
        async with _Peer(port, origin=PAGE) as page, _Peer(port) as tool:
            await page.send(cmd="open_keys")
            out["no shell"] = (await page.next("open_keys_result"))["data"]
            srv._open_keys_at = None
            async with _Shell(srv, monkeypatch) as shell:
                await page.send(cmd="open_keys")
                out["the notch's page"] = (await page.next("open_keys_result"))["data"]
                out["the shell heard"] = await shell.read()
                await page.send(cmd="open_keys")
                out["again at once"] = (await page.next("open_keys_result"))["data"]
                await tool.send(cmd="open_keys")
                out["a tool"] = (await tool.next("open_keys_result"))["data"]
        return out

    assert _with_server(monkeypatch, body) == {
        "no shell": {"ok": False, "error": "no_app"},
        "the notch's page": {"ok": True},
        "the shell heard": {"t": "shell", "action": "settings"},
        "again at once": {"ok": False, "error": "busy"},
        "a tool": {"ok": False, "error": "forbidden"},
    }


def test_brains_reload_rereads_the_files_for_anyone(monkeypatch, tmp_path):
    from neo2.core import keystore

    async def body(srv, port, seen):
        async with _Peer(port) as tool, _Peer(port, origin=PAGE) as page:
            assert page.hello["brains"]["providers"]["anthropic"] is False
            keystore.FileKeyStore(config.KEY_STORE_PATH).put("anthropic", KEY)  # (as `neo2 keys set` does)
            await tool.send(cmd="brains_reload")
            done = (await tool.next("brains_reload_result"))["data"]
            told = (await page.next("brains"))["data"]
        return done, told

    done, told = _with_server(monkeypatch, body)
    assert done == {"ok": True} and told["providers"]["anthropic"] is True
    assert config.settings().anthropic_api_key == KEY


def test_brains_restart_asks_the_shell(monkeypatch):
    async def body(srv, port, seen):
        srv.shell_said({"t": "keys_ticket", "ticket": TICKET})
        async with _Keys(port) as window:
            no_app = (await window.ask(cmd="brains_restart", id="r9"))["data"]
            async with _Shell(srv, monkeypatch) as shell:
                went = (await window.ask(cmd="brains_restart", id="r9"))["data"]
                heard = await shell.read()
        return no_app, went, heard

    no_app, went, heard = _with_server(monkeypatch, body)
    assert no_app == {"id": "r9", "ok": False, "error": "no_app"}
    assert went == {"id": "r9", "ok": True} and heard == {"t": "shell", "action": "restart"}


def test_a_failing_command_tells_every_client_without_the_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    monkeypatch.setattr(config, "_settings", None)
    config.settings()  # (loaded: the redaction knows it)

    async def failing(cmd):
        raise RuntimeError(f"401 invalid x-api-key {KEY}")

    async def body(srv, port, seen):
        async with _Peer(port, origin=PAGE) as page:
            await page.send(cmd="wake")
            return (await page.next("error"))["data"]

    data = _with_server(monkeypatch, body, on_command=failing)
    assert KEY not in data["message"] and "••••x9Qa" in data["message"]


def test_the_ticket_rides_the_shells_socket(monkeypatch):
    async def body(srv, port, seen):
        async with _Shell(srv, monkeypatch) as shell:
            await shell.say(t="keys_ticket", ticket=TICKET)
            await asyncio.sleep(0.1)
            got = srv._keys_ticket
            await shell.say(t="keys_closed")
            await asyncio.sleep(0.1)
            return got, srv._keys_ticket

    assert _with_server(monkeypatch, body) == (TICKET, None)
    assert ws_mod.client_kind("tauri://localhost", "", keys=True) == "keys"
    assert ws_mod.client_kind("neo://app", "", keys=True) is None and ws_mod.client_kind(None, "", keys=True) is None
