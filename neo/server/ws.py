"""WebSocket bridge between the Python core and the Tauri/React overlay.

Server → UI  : every EventBus event as JSON {type, data, ts}
UI → Server  : {"cmd": "text", "text": "..."}      send a typed request
               {"cmd": "confirm", "yes": true, "action_id": "…"}   answer a pending confirmation
                   (+ "scope": "task" — yes for this task, when the card's "scopes" offer it)
               {"cmd": "ptt", "on": true}          push-to-talk toggle
               {"cmd": "stop"}                     interrupt speech / cancel
               {"cmd": "wake"}                     arm the voice session
               {"cmd": "state"}                    ask for a snapshot
               {"cmd": "browser_*", …}             NEO's browser card (neo/tools/browser_live.py)

NEO's browser card has its own lane: its commands go straight to the browser (never through the
event bus, never logged — they carry what the user types), its JSON messages go out beside the
bus's, and its live page frames go out as binary websocket messages, one latest-frame slot per
client that never queues: pending JSON always goes first, and a flood of frames can't get a
client dropped.

The relay never blocks the event bus: each client has its own outbound queue drained by its
own task, and a client that stops reading is dropped. (An inline `await send()` to a half-dead
overlay once stalled the Gemini Live receive loop long enough to kill the voice session.)
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import os
import secrets
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import parse_qs, urlparse

import websockets
from websockets.asyncio.server import ServerConnection, serve

from neo import health
from neo.agent import confirm
from neo.config import settings
from neo.events import bus

# A port probe or a half-open socket isn't a websocket client; don't dump a traceback for it.
logging.getLogger("websockets.server").setLevel(logging.CRITICAL)

# Where NEO's overlay runs from: the bundled app, and the dev server during `tauri dev`.
_ALLOWED_ORIGINS = {
    "tauri://localhost",
    "http://tauri.localhost",
    "https://tauri.localhost",
    "http://localhost:1420",
    "http://127.0.0.1:1420",
}


def ws_token() -> str:
    """The shared secret in ~/.neo/ws-token (only this user can read it). NEO.app creates it
    before starting the core; either side creates it if missing."""
    path = settings().data_dir / "ws-token"
    try:
        tok = path.read_text().strip()
        if len(tok) >= 32:
            return tok
    except OSError:
        pass
    tok = secrets.token_urlsafe(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(tok)
    return tok


CommandHandler = Callable[[dict[str, Any]], Awaitable[None]]
_QUEUE_MAX = 500
_POSTER_KEEP_S = 600.0  # a closed card's poster stays in the replay this long


class _Client:
    """One overlay connection: its JSON queue and its one latest-frame slot."""

    def __init__(self, conn: ServerConnection) -> None:
        self.conn = conn
        self.q: deque[str] = deque()
        self.frame: bytes | str | None = None
        self.wake = asyncio.Event()
        self.browser = False  # said browser_hello: gets the browser card's messages

    def send_json(self, text: str) -> bool:
        if len(self.q) >= _QUEUE_MAX:
            return False
        self.q.append(text)
        self.wake.set()
        return True

    def offer_frame(self, payload: bytes | str) -> bool:
        """The newest page frame replaces one not yet sent (True when it did)."""
        replaced = self.frame is not None
        self.frame = payload
        self.wake.set()
        return replaced


class UIServer:
    def __init__(self, on_command: CommandHandler) -> None:
        self._clients: dict[ServerConnection, _Client] = {}
        self._on_command = on_command
        self._recent: list[dict[str, Any]] = []  # replayed to late-joining clients

    async def start(self) -> None:
        s = settings()
        bus().subscribe(self._relay)
        from neo.tools.browser_live import live

        live.channel = self
        self._token = ws_token()
        # No per-message deflate: the socket is local, and page frames are JPEGs, which don't
        # compress — zlib on every frame was most of the core's cost of the live view.
        self._server = await serve(
            self._handle, s.ws_host, s.ws_port, max_size=2**22, process_request=self._admit, compression=None
        )
        print(f"[ui] websocket on ws://{s.ws_host}:{s.ws_port} (token required)")

    def _admit(self, connection, request):
        """Only NEO's own overlay (and local tools that can read ~/.neo/ws-token) may connect.
        Browsers let any web page open ws://127.0.0.1 — without this, a page you visit could
        tell NEO to run commands. Rejected before the websocket even opens."""
        origin = request.headers.get("Origin")
        if origin is not None and origin not in _ALLOWED_ORIGINS:
            return connection.respond(403, "Forbidden origin\n")
        query = parse_qs(urlparse(request.path).query)
        given = (query.get("token") or [""])[0]
        if not hmac.compare_digest(given.encode(), self._token.encode()):
            return connection.respond(401, "Missing or wrong token\n")
        return None

    def _relay(self, ev) -> None:
        msg = ev.to_dict()
        if ev.type == "erased":
            self._recent.clear()  # late-joining overlays must not get the old transcript back
        if ev.type in ("transcript", "state", "tool_start", "tool_end", "turn"):
            last = self._recent[-1] if self._recent else None
            if (
                ev.type == "transcript"
                and last
                and last["type"] == "transcript"
                and not last["data"].get("final")
                and last["data"].get("role") == ev.data.get("role")
            ):
                self._recent[-1] = msg  # a streamed reply is one entry, not one per token
            else:
                self._recent.append(msg)
            self._recent = self._recent[-60:]
        if not self._clients:
            return
        self._send_all(json.dumps(msg, default=str), lambda cl: True)

    def _send_all(self, data: str, who: Callable[[_Client], bool]) -> None:
        for c, cl in list(self._clients.items()):
            if who(cl) and not cl.send_json(data):
                self._gone(c)  # not reading → gone

    def _gone(self, c: ServerConnection) -> None:
        if (cl := self._clients.pop(c, None)) is not None:
            from neo.tools.browser_live import live

            live.viewer_gone(cl)
            # A client that stopped reading is cut off for good: close its socket, so the overlay
            # notices, reconnects and gets a fresh hello (an open socket that hears nothing never does).
            transport = getattr(c, "transport", None)
            if transport is not None:
                with contextlib.suppress(Exception):
                    transport.abort()

    def broadcast(
        self, type_: str, data: dict[str, Any], *, recent: bool = False, replay: dict[str, Any] | None = None
    ) -> None:
        """A browser card message (neo/tools/browser_live.py). The card's opening and closing are
        part of the transcript (replayed — as `replay` when given: what late-joining overlays may
        see of it); the rest goes only to clients that show the card."""
        msg = {"type": type_, "data": data, "ts": time.time()}
        if recent:
            kept = replay if replay is not None else data
            self._recent.append({**msg, "data": kept})
            self._recent = self._recent[-60:]
            if kept.get("poster"):
                # In memory only, and not for long: the poster leaves the replay after ten minutes.
                asyncio.get_running_loop().call_later(_POSTER_KEEP_S, lambda: kept.update(poster=None))
        if self._clients:
            self._send_all(json.dumps(msg, default=str), lambda cl: recent or cl.browser)

    async def _drain(self, cl: _Client) -> None:
        c = cl.conn
        try:
            while True:
                await cl.wake.wait()
                cl.wake.clear()
                while cl.q or cl.frame is not None:
                    if cl.q:  # control and state beat pixels
                        data: str | bytes = cl.q.popleft()
                    else:
                        data, cl.frame = cl.frame, None  # type: ignore[assignment]
                    await asyncio.wait_for(c.send(data), 5.0)
        except (websockets.ConnectionClosed, TimeoutError, asyncio.CancelledError):
            pass
        finally:
            self._gone(c)

    async def _handle(self, c: ServerConnection) -> None:
        from neo.tools.browser_live import live

        cl = _Client(c)
        # The hello goes first, before anything the bus relays to this client.
        cl.send_json(
            json.dumps(
                {
                    "type": "hello",
                    "data": {
                        "state": bus().state.value,
                        "recent": self._recent,
                        "confirm": confirm.card(confirm.peek()),  # a question still waiting
                        "problems": health.problems(),  # missing key, blocked mic, voice down…
                        "browser": live.hello_snapshot(),  # NEO's browser card, if one is live
                    },
                },
                default=str,
            )
        )
        self._clients[c] = cl
        drain = asyncio.create_task(self._drain(cl))
        try:
            async for raw in c:
                if not isinstance(raw, str):
                    continue  # nothing binary comes up from the overlay
                try:
                    cmd = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(cmd, dict):
                    continue
                kind = str(cmd.get("cmd") or "")
                if kind.startswith("browser_"):
                    # Straight to the browser: never the bus, never a log (it carries what the user types).
                    if kind == "browser_hello":
                        cl.browser = True
                    try:
                        live.command(cl, cmd, len(raw))
                    except Exception as e:  # noqa: BLE001
                        print(f"[ui] browser command failed: {type(e).__name__}")
                    continue
                if kind == "state":
                    cl.send_json(json.dumps({"type": "state", "data": {"state": bus().state.value}}))
                    continue
                asyncio.create_task(self._safe(cmd))
        finally:
            self._gone(c)
            drain.cancel()

    async def _safe(self, cmd: dict[str, Any]) -> None:
        try:
            await self._on_command(cmd)
        except Exception as e:  # noqa: BLE001
            await bus().publish("error", message=str(e))
