"""The core's websocket admits only NEO's overlay (token + origin) — not other web pages."""

from __future__ import annotations

import asyncio
import json
import os
import socket

import pytest
import websockets

from neo.server import ws as ws_mod


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _server(monkeypatch):
    from neo import config

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
        "a local tool": "open",
    }


def test_token_file_is_private_and_stable():
    a, b = ws_mod.ws_token(), ws_mod.ws_token()
    from neo.config import settings

    path = settings().data_dir / "ws-token"
    assert a == b and len(a) >= 32 and oct(os.stat(path).st_mode & 0o777) == "0o600"
