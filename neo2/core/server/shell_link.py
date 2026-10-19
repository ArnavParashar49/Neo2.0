"""NEO.app's shell and the core, talking to each other directly.

NEO.app starts the core. Before it does, it makes an anonymous socket pair, keeps one end and
lets the core inherit the other (its number is in NEO_APP_FD). Over it go JSON lines:

core → shell   {"t":"hello","ui_host":"notch","host_attached":false}   the websocket is up
               {"t":"host_attached","attached":true}    a host came or went: the menu-bar item follows
               {"t":"ui_host","host":"window"}          where NEO's UI lives now: the window follows
               {"t":"shell","action":"restart"}         a host asked for one of the menu's actions
                                                        (+ "on": bool for open_at_login); "settings"
                                                        also for `open_keys` and a request that
                                                        needs a model NEO hasn't got (Brains & Keys),
                                                        "restart" for Brains & Keys' brains_restart
               {"t":"permission","what":"microphone",   what macOS says of the microphone or the
                "status":"granted"}                     camera: granted | denied | restricted |
                                                        not_determined | unknown — the answer to
                                                        the shell's `permission` below
shell → core   {"t":"status","accessibility":true,…}    what the menu shows (after the hello, and
                                                        whenever it changes)
               {"t":"wake"}                             the summon keys, while the notch shows NEO
               {"t":"keys_ticket","ticket":"<64 hex>"}  the Brains & Keys window is open: the ticket
                                                        it connects with (again after every hello
                                                        while it is open; neo2/core/server/ws.py)
               {"t":"keys_closed"}                      it closed: the ticket is no good any more
               {"t":"permission","what":"camera",       Setup's checklist: what macOS says of the
                "op":"status"|"ask"}                    microphone or the camera, or ("ask") its
                                                        prompt — the core uses them, so the core asks
                                                        (neo2.core.permissions.media); the answer
                                                        comes back as `permission` above

Why not NEO.app's own page, which used to carry all of this over its websocket: with NEO in the
notch that page's window is never on screen, and macOS suspends the process of a web page nobody
can see (runningboardd: "Suspending task", seconds after the window is hidden) — the socket stays
open, the core's messages are taken, and no script runs to read them. The menu's actions, the
summon keys and the status went nowhere. The shell needs no page for any of them.

Who can be at the other end: the process that started this core, and nobody else. The pair has no
name, no port and no token to know; it cannot be connected to, only inherited. The core takes it
out of its environment and marks it not-inheritable at once, so nothing the core starts (a tool, a
browser, a script) ever holds it. A core started by hand has no such socket and no shell: its
`shell` answers no_app and its hello's "shell" says app: false.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
import stat
from typing import Any

ENV = "NEO_APP_FD"
_MAX_LINE = 4096  # a status line is ~150 bytes
_MAX_PENDING = 65536  # bytes waiting for a shell that is not reading


def inherited() -> socket.socket | None:
    """The socket NEO.app handed this core, claimed for the core alone — or None (run by hand).
    Call before anything starts a child process."""
    raw = os.environ.pop(ENV, "")  # no child of the core learns the number
    if not raw.isdigit():
        return None
    fd = int(raw)
    try:
        if fd < 3 or not stat.S_ISSOCK(os.fstat(fd).st_mode):
            return None
        os.set_inheritable(fd, False)  # …or gets the socket itself
        sock = socket.socket(fileno=fd)
        if sock.family != socket.AF_UNIX:  # a pair is a unix socket: not something that listens on a port
            sock.detach()
            return None
        sock.setblocking(False)
        return sock
    except OSError:
        return None


class ShellLink:
    """The core's end of the pair. `ui` (the UIServer) is told three things: `shell_hello(link)` —
    which returns the hello to send — `shell_said(msg)` for each message of the shell's, and
    `shell_left(link)` once, when the shell is gone."""

    def __init__(self, sock: socket.socket, ui: Any) -> None:
        self._sock, self._ui = sock, ui
        self._writer: asyncio.StreamWriter | None = None

    def send(self, msg: dict[str, Any]) -> bool:
        w = self._writer
        if w is None or w.is_closing():
            return False
        if w.transport.get_write_buffer_size() > _MAX_PENDING:
            return False  # a shell that stopped reading: nothing piles up here
        w.write((json.dumps(msg, ensure_ascii=False) + "\n").encode())
        return True

    async def run(self) -> None:
        """Says hello, then reads the shell's lines until it goes away."""
        try:
            reader, self._writer = await asyncio.open_connection(sock=self._sock, limit=_MAX_LINE)
        except OSError:
            return
        try:
            self.send(self._ui.shell_hello(self))
            while True:
                try:
                    line = await reader.readline()
                except ValueError:  # longer than anything the shell says: dropped, and on to the next
                    continue
                if not line:
                    break
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if isinstance(msg, dict):
                    try:
                        self._ui.shell_said(msg)
                    except Exception as e:  # noqa: BLE001
                        print(f"[app] a message of the shell's failed: {type(e).__name__}")
            print("[app] NEO.app's shell is gone from its socket")
        except OSError:
            print("[app] the socket to NEO.app's shell broke")
        finally:  # also when the core itself is stopping (the task is cancelled)
            w, self._writer = self._writer, None
            if w is not None:
                with contextlib.suppress(Exception):
                    w.close()
            self._ui.shell_left(self)


def start(ui: Any, sock: socket.socket | None) -> asyncio.Task | None:
    """Starts the link on the socket `inherited()` returned (None: no shell, nothing to start).
    The caller keeps the task: the event loop alone would not."""
    if sock is None:
        return None
    print("[app] NEO.app's shell is on its socket")
    return asyncio.create_task(ShellLink(sock, ui).run())
