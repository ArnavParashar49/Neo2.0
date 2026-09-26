"""WebSocket bridge between the Python core and the Tauri/React overlay.

Server → UI  : every EventBus event as JSON {type, data, ts}
UI → Server  : {"cmd": "text", "text": "..."}      send a typed request
               {"cmd": "confirm", "yes": true}     answer a pending confirmation
               {"cmd": "ptt", "on": true}          push-to-talk toggle
               {"cmd": "stop"}                     interrupt speech / cancel
               {"cmd": "wake"}                     arm the voice session
               {"cmd": "state"}                    ask for a snapshot

The relay never blocks the event bus: each client has its own outbound queue drained by its
own task, and a client that stops reading is dropped. (An inline `await send()` to a half-dead
overlay once stalled the Gemini Live receive loop long enough to kill the voice session.)
"""

from __future__ import annotations

import logging

# A port probe or a half-open socket isn't a websocket client; don't dump a traceback for it.
logging.getLogger("websockets.server").setLevel(logging.CRITICAL)

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

import websockets
from websockets.asyncio.server import ServerConnection, serve

from neo.config import settings
from neo.events import bus

CommandHandler = Callable[[dict[str, Any]], Awaitable[None]]
_QUEUE_MAX = 500


class UIServer:
    def __init__(self, on_command: CommandHandler) -> None:
        self._clients: dict[ServerConnection, asyncio.Queue[str]] = {}
        self._on_command = on_command
        self._recent: list[dict[str, Any]] = []  # replayed to late-joining clients

    async def start(self) -> None:
        s = settings()
        bus().subscribe(self._relay)
        self._server = await serve(self._handle, s.ws_host, s.ws_port, max_size=2**22)
        print(f"[ui] websocket on ws://{s.ws_host}:{s.ws_port}")

    def _relay(self, ev) -> None:
        msg = ev.to_dict()
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
        data = json.dumps(msg, default=str)
        for c, q in list(self._clients.items()):
            try:
                q.put_nowait(data)
            except asyncio.QueueFull:
                self._clients.pop(c, None)  # not reading → gone

    async def _drain(self, c: ServerConnection, q: asyncio.Queue[str]) -> None:
        try:
            while True:
                data = await q.get()
                await asyncio.wait_for(c.send(data), 5.0)
        except (websockets.ConnectionClosed, TimeoutError, asyncio.CancelledError):
            pass
        finally:
            self._clients.pop(c, None)

    async def _handle(self, c: ServerConnection) -> None:
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=_QUEUE_MAX)
        self._clients[c] = q
        drain = asyncio.create_task(self._drain(c, q))
        try:
            await c.send(
                json.dumps({"type": "hello", "data": {"state": bus().state.value, "recent": self._recent}})
            )
            async for raw in c:
                try:
                    cmd = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if cmd.get("cmd") == "state":
                    q.put_nowait(json.dumps({"type": "state", "data": {"state": bus().state.value}}))
                    continue
                asyncio.create_task(self._safe(cmd))
        finally:
            self._clients.pop(c, None)
            drain.cancel()

    async def _safe(self, cmd: dict[str, Any]) -> None:
        try:
            await self._on_command(cmd)
        except Exception as e:  # noqa: BLE001
            await bus().publish("error", message=str(e))
