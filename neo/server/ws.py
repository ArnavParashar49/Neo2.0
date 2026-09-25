"""WebSocket bridge between the Python core and the Tauri/Vue overlay.

Server → UI  : every EventBus event as JSON {type, data, ts}
UI → Server  : {"cmd": "text", "text": "..."}      send a typed request
               {"cmd": "confirm", "yes": true}     answer a pending confirmation
               {"cmd": "ptt", "on": true}          push-to-talk toggle
               {"cmd": "stop"}                     interrupt speech / cancel
               {"cmd": "state"}                    ask for a snapshot
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

import websockets
from websockets.asyncio.server import ServerConnection, serve

from neo.config import settings
from neo.events import bus

CommandHandler = Callable[[dict[str, Any]], Awaitable[None]]


class UIServer:
    def __init__(self, on_command: CommandHandler) -> None:
        self._clients: set[ServerConnection] = set()
        self._on_command = on_command
        self._recent: list[dict[str, Any]] = []  # replayed to late-joining clients

    async def start(self) -> None:
        s = settings()
        bus().subscribe(self._relay)
        self._server = await serve(self._handle, s.ws_host, s.ws_port, max_size=2**22)
        print(f"[ui] websocket on ws://{s.ws_host}:{s.ws_port}")

    async def _relay(self, ev) -> None:
        msg = ev.to_dict()
        if ev.type in ("transcript", "state", "tool_start", "tool_end"):
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
        await asyncio.gather(*(self._send(c, data) for c in list(self._clients)), return_exceptions=True)

    async def _send(self, c: ServerConnection, data: str) -> None:
        try:
            await c.send(data)
        except websockets.ConnectionClosed:
            self._clients.discard(c)

    async def _handle(self, c: ServerConnection) -> None:
        self._clients.add(c)
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
                    await c.send(json.dumps({"type": "state", "data": {"state": bus().state.value}}))
                    continue
                asyncio.create_task(self._safe(cmd))
        finally:
            self._clients.discard(c)

    async def _safe(self, cmd: dict[str, Any]) -> None:
        try:
            await self._on_command(cmd)
        except Exception as e:  # noqa: BLE001
            await bus().publish("error", message=str(e))
