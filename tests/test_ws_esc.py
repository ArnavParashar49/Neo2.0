"""Esc pressed elsewhere (neo2/core/server/ws.py): NEO's page in the notch alone asks for it
(`esc_watch`), NEO.app's shell hears on its socket whether any page wants it, and the shell's `esc`
goes to the pages that asked — nobody else, never the generic handler. The server and the shell are
this test's own."""

from __future__ import annotations

import asyncio
import json

from tests.test_v2_ws_security import APP, PAGE, _Peer, _Shell, _with_server


async def _silent(shell, wait=0.3) -> bool:
    """The shell hears nothing more (within `wait` seconds)."""
    try:
        line = await asyncio.wait_for(shell.r.readline(), wait)
    except TimeoutError:
        return True
    raise AssertionError(f"the shell heard {json.loads(line)}")


async def _heard(peer, type_: str) -> bool:
    """Whether `type_` reached this client by the time it answered a `state` request."""
    return type_ in await peer.quiet()


def test_only_the_notchs_page_asks_and_the_esc_reaches_it_alone(monkeypatch):
    async def body(srv, port, seen):
        out = {}
        async with _Shell(srv, monkeypatch) as shell, _Peer(port, role="host") as host:
            assert await shell.read() == {"t": "host_attached", "attached": True}
            async with _Peer(port, origin=PAGE) as page, _Peer(port, origin=APP) as app, _Peer(port) as tool:
                for who in (host, app, tool):
                    await who.send(cmd="esc_watch", on=True)
                    await who.quiet()  # (the core has read it)
                out["refused"] = await _silent(shell)
                await shell.say(t="esc")
                out["nobody asked"] = [await _heard(p, "esc") for p in (page, host, app, tool)]
                await page.send(cmd="esc_watch", on=True)
                out["asked"] = await shell.read()
                await page.send(cmd="esc_watch", on=True)  # said again: nothing new for the shell
                await page.quiet()
                out["again"] = await _silent(shell)
                await shell.say(t="esc")
                out["esc"] = (await page.next("esc"))["data"]
                out["the others"] = [await _heard(p, "esc") for p in (host, app, tool)]
                for bad in ("yes", 1, None):
                    await page.send(cmd="esc_watch", on=bad)
                await page.quiet()
                out["not a bool"] = await _silent(shell)
                await page.send(cmd="esc_watch", on=False)
                out["off"] = await shell.read()
                await shell.say(t="esc")
                out["after off"] = await _heard(page, "esc")
        out["generic handler"] = [c for c in seen if c.get("cmd") == "esc_watch"]
        return out

    assert _with_server(monkeypatch, body) == {
        "refused": True,
        "nobody asked": [False, False, False, False],
        "asked": {"t": "esc_watch", "on": True},
        "again": True,
        "esc": {},
        "the others": [False, False, False],
        "not a bool": True,
        "off": {"t": "esc_watch", "on": False},
        "after off": False,
        "generic handler": [],
    }


def test_a_page_that_goes_takes_its_ask_with_it(monkeypatch):
    async def body(srv, port, seen):
        out = {}
        async with _Shell(srv, monkeypatch) as shell, _Peer(port, origin=PAGE) as first:
            await first.send(cmd="esc_watch", on=True)
            out["first"] = await shell.read()
            async with _Peer(port, origin=PAGE) as second:
                await second.send(cmd="esc_watch", on=True)
                await second.quiet()
                out["a second page"] = await _silent(shell)
            while sum(cl.kind == "page" for cl in srv._clients.values()) > 1:
                await asyncio.sleep(0.01)  # the core has seen the second one go
            out["one of two left"] = await _silent(shell)
            await shell.say(t="esc")
            out["the first still hears it"] = (await first.next("esc"))["data"]
        return out, srv

    out, srv = _with_server(monkeypatch, body)
    assert out == {
        "first": {"t": "esc_watch", "on": True},
        "a second page": True,
        "one of two left": True,
        "the first still hears it": {},
    }

    async def gone(srv, port, seen):
        async with _Shell(srv, monkeypatch) as shell:
            async with _Peer(port, origin=PAGE) as page:
                await page.send(cmd="esc_watch", on=True)
                asked = await shell.read()
            return asked, await shell.read()  # the last page that asked went: off

    assert _with_server(monkeypatch, gone) == ({"t": "esc_watch", "on": True}, {"t": "esc_watch", "on": False})


def test_a_shell_that_comes_later_hears_it_after_its_hello(monkeypatch):
    async def body(srv, port, seen):
        async with _Peer(port, origin=PAGE) as page:
            await page.send(cmd="esc_watch", on=True)
            await page.quiet()  # no shell yet: nobody to tell
            async with _Shell(srv, monkeypatch) as shell:
                hello = shell.hello["t"]
                after = await shell.read()
                await shell.say(t="esc")
                esc = (await page.next("esc"))["data"]
            # The shell went and a new one came (NEO.app's shell is the core's parent: in tests only).
            async with _Shell(srv, monkeypatch) as again:
                said_again = await again.read()
        return hello, after, esc, said_again

    assert _with_server(monkeypatch, body) == (
        "hello",
        {"t": "esc_watch", "on": True},
        {},
        {"t": "esc_watch", "on": True},
    )
