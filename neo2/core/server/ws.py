"""WebSocket bridge between the Python core and the Tauri/React overlay.

Server → UI  : every EventBus event as JSON {type, data, ts}
UI → Server  : {"cmd": "text", "text": "..."}      send a typed request
               {"cmd": "confirm", "yes": true, "action_id": "…"}   answer a pending confirmation
                   (+ "scope": "task" — yes for this task, when the card's "scopes" offer it)
               {"cmd": "ptt", "on": true}          push-to-talk toggle
               {"cmd": "stop"}                     interrupt speech / cancel
               {"cmd": "wake"}                     arm the voice session
               {"cmd": "state"}                    ask for a snapshot
               {"cmd": "ui_host", "host": "notch"} where NEO's UI lives: NEO.app's window, or the
                                                   notch (NotchHub hosts the page; the app hides its window)
               {"cmd": "shell", "action": "restart"}   a host asks for one of NEO.app's menu actions
                   (host only): show | settings | restart | open_logs | erase_memory |
                   open_at_login (+ "on": bool) | quit
               {"cmd": "set", "name": "wake_word", "value": false}   switch a setting while NEO
                   runs (a host, or NEO.app's own page): see "Settings a client may switch"
               {"cmd": "browser_*", …}             NEO's browser card (neo2/core/tools/browser_live.py)
               {"cmd": "camera_*", …}              the camera's card (neo2/core/tools/camera/live.py)
               {"cmd": "song_*", …}                the song card (neo2/core/tools/music/live.py)
               {"cmd": "open_keys"}                open the Brains & Keys window (app, page, host)
               {"cmd": "brains_reload"}            re-read NEO's key and brain files (anyone)
               {"cmd": "key_save", "id": …, …}     Brains & Keys' own commands: see "Brains & Keys"

NEO's browser card has its own lane: its commands go straight to the browser (never through the
event bus, never logged — they carry what the user types), its JSON messages go out beside the
bus's, and its live page frames go out as binary websocket messages, one latest-frame slot per
client that never queues: pending JSON always goes first, and a flood of frames can't get a
client dropped. The camera's card works the same way, with a latest-frame slot of its own (its
preview frames are NBF1 kind 3); the camera also goes off on the stop button. The song card's JSON
takes the camera's lane (no frames: only how loud what NEO hears is), and the stop button stops its
listening too.

NEO.app's menu, for a host. In the notch NEO has no menu-bar item; the host's Settings carry the
menu instead. The host sends `shell`; the core hands it to NEO.app's shell — over the socket the
shell and the core share (neo2/core/server/shell_link.py), never a websocket, the bus or the
replay — and the shell runs the very action its menu item runs. `erase_memory` erases nothing by
itself: it makes NEO.app put up its own "Erase everything?" question, and only that dialog's
button erases. The sender gets a `shell_result` {action, ok, error?} (error: forbidden | unknown |
no_app). The other way, the shell reports what the menu shows, over the same socket; every client
reads it in the hello's "shell" and in a `shell_status` event when it changes:
{running: true, app, accessibility, screen_recording, open_at_login, version, hotkey} — `app` says
whether NEO.app's shell is there at all (a core run by hand has none; the rest is then null).
The hello's "host_attached" (and a `host_attached` event) says whether a host is connected:
NEO.app hides its menu-bar item only while one is (the shell hears it over its socket too, with
where the UI lives: `ui_host`). None of this goes through NEO.app's page: with NEO in the notch
its window is never on screen, and macOS suspends the process of a page nobody can see — it
answered nothing, and the menu, the summon keys and the status stopped with it.

Settings a client may switch. `wake_word` — "Listen for 'Neo'". true: the microphone
stays open so the wake word can be heard. false: the microphone is closed while NEO is idle (macOS
shows no microphone indicator) and opens only when NEO is summoned by hand — `wake` from the summon
keys, the orb or a host, or push-to-talk — until NEO is idle again. `set` keeps the value in
~/.neo/.env (NEO_WAKE_WORD), applies it to the running voice at once. `wellbeing_checkins` — gentle
check-ins, opt-in (NEO_WELLBEING_CHECKINS): detection stays on-device (Apple's sound classifier hears
only frames NEO's microphone captures while it is open anyway; it never opens or holds it, and saves
neither audio nor detection history); with Gemini Live, the check-in line and ~5 s of reply audio go
to Gemini.
Both settings return `set_result` {name, ok, error?} (error: forbidden | unknown | invalid | failed),
and are reported in the hello and `settings` events.

NEO's timers, for a host that shows a countdown (NotchHub's island). The hello's "timers" lists the
ones counting down, soonest first: [{id, title, ends_at}]. Each one that starts (set, or restored
after a restart) or ends is an `activity` event: {op: "present" | "end", id: "timer:<its id>", kind:
"timer", title, ends_at, done} — `ends_at` in seconds since the epoch (the Mac's own clock), `done`
true when it went off and false when it was cancelled (always false with "present"). The client
counts down from `ends_at` itself: nothing is sent per second. Not replayed: the hello says what runs
now. (`kind` says what an activity is: a client shows the kinds it knows and ignores the rest.)

Who may say what. Everyone here knows the token (~/.neo/ws-token, the user's alone to read). The
core then tells its clients apart by the one thing a web page cannot choose: the Origin header a
browser engine puts on every websocket a page opens, which the page can neither set nor leave out.
    app    Origin is NEO.app's own page (tauri://localhost, or the dev server's). A page like
           the next: it does not speak for NEO.app's shell, which has its own socket.
    page   Origin is neo://app: NEO's page shown by a host (NotchHub's WKWebView).
    host   no Origin, and `role=host` in the query beside the token: a native client (NotchHub's
           NeoListener). `role=host` with any Origin is refused at the door (403), so no page —
           not even NEO's own — can ever be a host.
    tool   no Origin, no role: a local program that read the token.
    keys   NEO.app's own page with the shell's ticket: the Brains & Keys window (see "Brains & Keys").
  - `shell` is the host's alone — every action of it, quit, restart, erase_memory and
    open_at_login among them. NEO's pages show what models and websites wrote; a page that was
    talked into something must not be able to quit NEO, restart it, or change what starts at
    login. A tool that wants the menu has to say so at connect (`role=host`), not slip it in.
  - `set` is a host's and NEO.app's own page's (app): the two places the user has a switch for it —
    the host's Settings, NEO.app's own window. Not NEO's page in the notch (page), which has no such
    switch and shows what models and websites wrote, and not a tool: whether the microphone stays
    open is the user's to decide in a UI of NEO's, not something a script flips in passing. Only the
    names in SETTABLE, each with a value of its own type; nothing else of the settings can be
    written from a socket.
  - What the menu shows is the shell's alone to say, and the menu's actions the shell's alone to
    hear — and the shell is not a websocket client at all: it is whoever started this core and
    kept the other end of the socket pair (shell_link.py), which nothing can connect to. So what a
    host's Settings show about NEO.app's permissions is what NEO.app itself said, and `quit` goes
    to NEO.app and nobody else. A `shell_status` on the websocket is refused, whoever sends it.
  - `confirm` — the answer to a pending confirmation — is anyone's who was admitted and names the
    card: NEO.app's own page, NEO's page in the notch, and a host (NotchHub's native Allow /
    Cancel buttons under the notch send it over the host's connection). It must carry the exact
    `action_id` of the card it answers, which only a client that was sent the card knows, so a
    stale or guessed answer approves nothing. No web page the user merely visits can send it: a
    foreign Origin is refused at the door. A local tool can — it is the user's own program (see
    below), and could as well type "yes".
  What this does not claim: a program running as the user can read the token and send any header
  it likes — it is the user, and could as well kill NEO or edit its files. It can be a host, and
  ask for the menu. The roles keep web content out of the menu and every honest client in its
  lane; they are not a wall between the user's own programs. (Speaking for the shell is no longer
  among the things such a program can do: there is no token for that, only the inherited socket.)

Brains & Keys. NEO.app's Brains & Keys window is the one client that may see or change NEO's keys
and brains — kind `keys`. It connects with the token and a ticket beside it (`ticket=<64 hex>`),
which NEO.app's shell made (32 random bytes) and handed the core over its own socket (shell_link.py:
`keys_ticket`, again with each core hello while the window is open; `keys_closed` revokes it). It
is a `keys` client only from NEO.app's own origin, with no role, and with that exact ticket
(compared in constant time); a wrong ticket is refused at the door (403; the log says so, never the
value), and there is one such client at a time (a second: 409). The path, which carries the
ticket, is never logged.
  - Its commands — brains_get, key_test, key_save, key_remove, models_list, brains_set, custom_save,
    custom_remove, brains_restart (neo2/core/brains.py) — each carry an "id" (up to 32 characters),
    echoed in the reply, which goes to that client alone: {"type": <result>, "data": {…}}. They are
    routed here, by name, before anything else: never the generic handler (whose failures go to
    every client), never the event bus, never the replay, never printed — a `[ui] keys:` line names
    the command and how it went, nothing it carried. A result is fixed words and status codes,
    never an exception's text.
  - The same commands from anyone else get {"type": "keys_refused", "data": {"cmd", "error":
    "forbidden"}}, to that client alone. NEO.app's overlay and the notch's page show what models and
    websites wrote; a host's contract is a closed list; a tool is any program — none sets a key.
  - The `keys` client sends nothing else: no request, no card's buttons. It hears the hello (state,
    problems, brains, shell, ui_host) and of the bus only `brains`, `problems` and `state`.
  - `open_keys`, from NEO.app's overlay, the notch's page or a host (one every 5 s): the core asks
    NEO.app's shell for its "settings" action, which opens the window. {"type": "open_keys_result",
    "data": {"ok", "error"?: no_app | busy | forbidden}}.
  - `brains_reload`, from anyone: NEO re-reads its own files (the CLI sends it after `keys set`). It
    carries no value; the sender hears {"type": "brains_reload_result", "data": {"ok"}}. Coalesced
    (brains.reload): one reading at a time, at most one a second, each request answered by a
    reading that began after it.
  - Every client hears the `brains` event (and the hello's "brains"): whether NEO has a model to
    ask, each role's label, the voice, and per provider whether a key is set — never a key, never
    its last 4 (those go to the `keys` client alone, in brains_detail).

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
import re
import secrets
import sys
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import parse_qs, urlparse

import websockets
from websockets.asyncio.server import ServerConnection, serve

from neo2.core import brains, health
from neo2.core.agent import confirm
from neo2.core.config import settings
from neo2.core.events import bus

# A port probe or a half-open socket isn't a websocket client; don't dump a traceback for it.
logging.getLogger("websockets.server").setLevel(logging.CRITICAL)

# Where NEO's overlay runs from: the bundled app, the dev server during `tauri dev`, and the
# notch — NotchHub serves the same built page from its own `neo://app/` scheme (a WKWebView, which
# sends that as its Origin). Exactly that origin, nothing else with the scheme.
_APP_ORIGINS = {
    "tauri://localhost",
    "http://tauri.localhost",
    "https://tauri.localhost",
    "http://localhost:1420",
    "http://127.0.0.1:1420",
}
_NOTCH_ORIGIN = "neo://app"
_ALLOWED_ORIGINS = _APP_ORIGINS | {_NOTCH_ORIGIN}

# NEO.app's menu actions a host may ask for (ui/src-tauri/src/lib.rs runs them: `act`).
SHELL_ACTIONS = ("show", "settings", "restart", "open_logs", "erase_memory", "open_at_login", "quit")
# The settings a client may switch while NEO runs (`set`), each with the type its value must have…
SETTABLE: dict[str, type] = {"wake_word": bool, "wellbeing_checkins": bool}
_MAY_SET = ("host", "app")  # …and who may: a host, and NEO.app's own page
# Brains & Keys (see the module's doc): the ticket NEO.app's shell makes, who may ask for the window,
# how often, and what of the bus the window hears.
_TICKET = re.compile(r"[0-9a-f]{64}")
_MAY_OPEN_KEYS = ("app", "page", "host")
_OPEN_KEYS_S = 5.0
_KEYS_HEARS = frozenset({"brains", "problems", "state"})
_LOGGABLE_CMD = re.compile(r"[a-z_]{1,32}")


def client_kind(origin: str | None, role: str, *, keys: bool = False) -> str | None:
    """Which kind of client a connection is — app | page | host | tool | keys — or None for one that
    is not admitted at all (see the module's "Who may say what"). `keys`: it brought the shell's
    ticket (checked by the caller) — only NEO.app's own page, with no role, is then `keys`."""
    if role not in ("", "host"):
        return None
    if keys:
        return "keys" if origin in _APP_ORIGINS and not role else None
    if origin is None:
        return "host" if role == "host" else "tool"
    if role:
        return None  # a page is never a host
    if origin in _APP_ORIGINS:
        return "app"
    if origin == _NOTCH_ORIGIN:
        return "page"
    return None


def _cmd_name(kind: str) -> str:
    """A command's name as a log line may carry it (a client chooses it: never anything else)."""
    return kind if _LOGGABLE_CMD.fullmatch(kind) else "a command"


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


_serving: UIServer | None = None  # the server started in this process (NEO.app's core has one)


def _timers() -> list[dict[str, Any]]:
    """NEO's running timers, for the hello (see the module's "NEO's timers"). They live in the timer
    tool's module: when nothing has loaded it, no timer can be running — and the hello never loads a
    tool module itself."""
    mod = sys.modules.get("neo2.core.tools.system")
    if mod is None:
        return []
    try:
        return mod.timers_snapshot()
    except Exception as e:  # noqa: BLE001 — a hello without the timers beats no hello
        print(f"[ui] timers not in the hello: {type(e).__name__}")
        return []


def card_on_screen() -> bool:
    """Whether a UI that shows a pending confirmation's buttons is connected now: NEO.app's page,
    NEO's page in the notch, or a host (NotchHub's Allow / Cancel under the notch). A local tool
    shows nothing. The voice asks aloud once and leaves the rest to those buttons."""
    s = _serving
    return s is not None and any(cl.kind in ("app", "page", "host") for cl in list(s._clients.values()))


CommandHandler = Callable[[dict[str, Any]], Awaitable[None]]
_QUEUE_MAX = 500
_POSTER_KEEP_S = 600.0  # a closed card's poster stays in the replay this long


class _Client:
    """One overlay connection: its JSON queue and its one latest-frame slot."""

    def __init__(self, conn: ServerConnection) -> None:
        self.conn = conn
        self.q: deque[str] = deque()
        self.frame: bytes | str | None = None
        self.cam: bytes | None = None  # the camera card's latest preview frame
        self.wake = asyncio.Event()
        self.browser = False  # said browser_hello: gets the browser card's messages
        self.kind = "tool"  # app | page | host | tool | keys: who this is (client_kind)

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

    def offer_camera(self, payload: bytes) -> None:
        """The camera's newest preview frame replaces one not yet sent."""
        self.cam = payload
        self.wake.set()


class UIServer:
    def __init__(self, on_command: CommandHandler) -> None:
        self._clients: dict[ServerConnection, _Client] = {}
        self._on_command = on_command
        self._recent: list[dict[str, Any]] = []  # replayed to late-joining clients
        self._shell: dict[str, Any] | None = None  # what NEO.app's menu shows, as its shell reported
        self._link: Any = None  # the shell's socket (shell_link.ShellLink) while NEO.app is at its end
        self.shell_task: asyncio.Task | None = None  # …and the task reading it (kept: the loop holds it weakly)
        self._keys_ticket: str | None = None  # the Brains & Keys window's ticket, as the shell made it
        self._open_keys_at: float | None = None  # monotonic: the last `open_keys` a client asked for

    async def start(self) -> None:
        s = settings()
        bus().subscribe(self._relay)
        from neo2.core.tools.browser_live import live

        live.channel = self
        from neo2.core.tools.camera.live import camera_live

        camera_live.channel = self
        from neo2.core.tools.music.live import song_live

        song_live.channel = self
        self._token = ws_token()
        global _serving
        _serving = self
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
        role = (query.get("role") or [""])[0]
        ticket = (query.get("ticket") or [None])[0]
        if ticket is not None:  # the Brains & Keys window, or something that says it is
            if not self._ticket_ok(ticket):
                print("[ui] a Brains & Keys connection was refused: its ticket isn't the shell's")
                return connection.respond(403, "Forbidden ticket\n")
            if client_kind(origin, role, keys=True) != "keys":
                print("[ui] a Brains & Keys connection was refused: it isn't NEO.app's own page")
                return connection.respond(403, "Forbidden\n")
            if self._keys_client() is not None:
                print("[ui] a second Brains & Keys connection was refused: one at a time")
                return connection.respond(409, "Busy\n")
            return None
        if client_kind(origin, role) is None:
            return connection.respond(403, "Forbidden role\n")  # a page asking to be a host
        return None

    # ---- Brains & Keys ---------------------------------------------------------------------------

    def _ticket_ok(self, given: Any) -> bool:
        """`given` is the ticket the shell made for the Brains & Keys window (constant-time)."""
        want = self._keys_ticket
        if want is None or not isinstance(given, str) or not _TICKET.fullmatch(given):
            return False
        return hmac.compare_digest(given.encode(), want.encode())

    def _keys_client(self) -> _Client | None:
        return next((cl for cl in self._clients.values() if cl.kind == "keys"), None)

    def has_shell(self) -> bool:
        """NEO.app's shell is at the other end of its socket (a core started by hand has none)."""
        return self._link is not None

    def open_keys(self) -> bool:
        """Ask NEO.app's shell to open the Brains & Keys window (its "settings" action)."""
        went = self._to_shell({"t": "shell", "action": "settings"})
        if went:
            print("[ui] shell: settings (Brains & Keys)")
        return went

    def restart(self) -> bool:
        """Ask NEO.app's shell to restart NEO (brains_restart: a new Gemini key for the voice)."""
        went = self._to_shell({"t": "shell", "action": "restart"})
        if went:
            print("[ui] shell: restart (Brains & Keys)")
        return went

    def _keys_ticket_said(self, msg: dict[str, Any]) -> None:
        ticket = msg.get("ticket")
        if not isinstance(ticket, str) or not _TICKET.fullmatch(ticket):
            print("[ui] a Brains & Keys ticket was ignored: not one the shell makes")
            return
        if ticket != self._keys_ticket:
            self._keys_ticket = ticket
            print("[ui] Brains & Keys is open: its ticket came from NEO.app's shell")

    def _keys_closed(self) -> None:
        """The window is gone: its ticket is no good any more, and its connection ends."""
        if self._keys_ticket is not None:
            print("[ui] Brains & Keys closed: its ticket is revoked")
        self._keys_ticket = None
        for c, cl in list(self._clients.items()):
            if cl.kind == "keys":
                self._gone(c)

    def _send_to(self, cl: _Client, type_: str, data: dict[str, Any]) -> None:
        cl.send_json(json.dumps({"type": type_, "data": data, "ts": time.time()}, default=str))

    async def _keys_ask(self, cl: _Client, cmd: dict[str, Any]) -> None:
        """One of Brains & Keys' commands, from the window: its answer goes to it alone."""
        name = str(cmd.get("cmd") or "")
        type_, data = await brains.command(cmd, restart=self.restart)
        self._send_to(cl, type_, data)
        print(f"[ui] keys: {_cmd_name(name)} {brains.outcome(data)}")

    def _keys_refused(self, cl: _Client, cmd: dict[str, Any]) -> None:
        name = _cmd_name(str(cmd.get("cmd") or ""))
        self._send_to(cl, "keys_refused", {"id": brains.request_id(cmd), "cmd": name, "error": "forbidden"})
        print(f"[ui] keys: {name} refused: from a {cl.kind}, not Brains & Keys")

    def _open_keys_ask(self, cl: _Client) -> None:
        """`open_keys` from NEO.app's overlay, the notch's page or a host: one every 5 seconds."""
        now = time.monotonic()
        if cl.kind not in _MAY_OPEN_KEYS:
            error = "forbidden"
        elif self._open_keys_at is not None and now - self._open_keys_at < _OPEN_KEYS_S:
            error = "busy"
        else:
            self._open_keys_at = now
            error = "" if self.open_keys() else "no_app"
        self._send_to(cl, "open_keys_result", {"ok": not error, **({"error": error} if error else {})})
        if error:
            print(f"[ui] open_keys: {error} (from a {cl.kind})")

    async def _reload_ask(self, cl: _Client) -> None:
        """`brains_reload`: NEO's own files read again (a key the CLI just saved) — coalesced, so a
        client asking again and again costs one reading at a time, at most one a second."""
        try:
            await brains.reload()
            ok = True
        except Exception as e:  # noqa: BLE001
            print(f"[ui] brains_reload failed: {type(e).__name__}")
            ok = False
        self._send_to(cl, "brains_reload_result", {"ok": ok})
        print(f"[ui] brains_reload: {'ok' if ok else 'failed'} (from a {cl.kind})")

    # ---- NEO.app's menu, for a host ---------------------------------------------------------

    def _host_attached(self) -> bool:
        return any(cl.kind == "host" for cl in self._clients.values())

    def shell_snapshot(self) -> dict[str, Any]:
        """What a host's Settings show about NEO: the core is running (it is answering), and what
        NEO.app's shell last reported — null where no shell has."""
        st = self._shell or {}
        keys = ("accessibility", "screen_recording", "open_at_login", "version", "hotkey")
        return {"running": True, "app": self._shell is not None, **{k: st.get(k) for k in keys}}

    def _tell(self, type_: str, data: dict[str, Any]) -> None:
        """To every client connected now — not the bus, not the replay."""
        self._send_all(json.dumps({"type": type_, "data": data, "ts": time.time()}), lambda cl: True)

    # NEO.app's shell, at the other end of its socket (shell_link.py): the three calls it makes.

    def shell_hello(self, link: Any) -> dict[str, Any]:
        """The shell's socket is open: what the shell needs to know first."""
        self._link = link
        return {"t": "hello", "ui_host": settings().ui_host, "host_attached": self._host_attached()}

    def shell_said(self, msg: dict[str, Any]) -> None:
        """One message of the shell's: what its menu shows, or the summon keys."""
        kind = msg.get("t")
        if kind == "status":
            self._shell_report(msg)
        elif kind == "wake":
            # Summoned by hand while the notch shows NEO: the same command the orb's page sends.
            asyncio.create_task(self._safe({"cmd": "wake"}))
        elif kind == "keys_ticket":
            self._keys_ticket_said(msg)  # the Brains & Keys window is open (see "Brains & Keys")
        elif kind == "keys_closed":
            self._keys_closed()

    def shell_left(self, link: Any) -> None:
        """NEO.app is gone from its socket: nobody speaks for the shell, nobody runs the menu."""
        if self._link is not link:
            return
        self._link = None
        if self._shell is not None:
            self._shell = None
            self._tell("shell_status", self.shell_snapshot())

    def _to_shell(self, msg: dict[str, Any]) -> bool:
        link = self._link
        return link is not None and bool(link.send(msg))

    def _shell_report(self, msg: dict[str, Any]) -> None:
        """What the menu shows, from the shell: kept, and told to everyone when it changed."""
        flags = {k: msg.get(k) for k in ("accessibility", "screen_recording", "open_at_login")}
        if not all(isinstance(v, bool) for v in flags.values()):
            return
        version, hotkey = str(msg.get("version") or "")[:32], str(msg.get("hotkey") or "")[:16]
        status = {**flags, "version": version, "hotkey": hotkey}
        if status != self._shell:
            self._shell = status
            self._tell("shell_status", self.shell_snapshot())

    def _host_changed(self, attached: bool) -> None:
        """The first host came, or the last one went: NEO.app's menu-bar item follows."""
        self._tell("host_attached", {"attached": attached})
        self._to_shell({"t": "host_attached", "attached": attached})

    def _shell_ask(self, cl: _Client, cmd: dict[str, Any]) -> None:
        """`shell` from a host: one of the menu's actions, handed to NEO.app's shell. The sender
        hears how it went (`shell_result`)."""
        action = str(cmd.get("action") or "")[:32]
        error = self._shell_relay(cl, action, cmd.get("on"))
        data = {"action": action, "ok": not error, **({"error": error} if error else {})}
        cl.send_json(json.dumps({"type": "shell_result", "data": data, "ts": time.time()}))

    def _shell_relay(self, cl: _Client, action: str, on: Any) -> str:
        """Hands the action to NEO.app's shell; returns why not ("" when it went)."""
        if cl.kind != "host":
            print(f"[ui] shell refused: from a {cl.kind}, not a host")
            return "forbidden"
        if action not in SHELL_ACTIONS:
            return "unknown"
        data: dict[str, Any] = {"action": action}  # the action and its switch: nothing else rides along
        if action == "open_at_login":
            if not isinstance(on, bool):
                return "unknown"
            data["on"] = on
        if not self._to_shell({"t": "shell", **data}):
            return "no_app"  # a core run by hand: there is no NEO.app to do it
        print(f"[ui] shell: {action}")
        return ""

    async def _set_ask(self, cl: _Client, cmd: dict[str, Any]) -> None:
        """`set` from a host or NEO.app's own page: one of the settings in SETTABLE, handed to the
        core's command handler (it keeps the value, applies it and tells every client). The sender
        hears how it went (`set_result`)."""
        name, value = str(cmd.get("name") or "")[:32], cmd.get("value")
        error = self._set_refused(cl, name, value)
        if not error:
            try:
                await self._on_command({"cmd": "set", "name": name, "value": value})
                print(f"[ui] set: {name}={value}")
            except Exception as e:  # noqa: BLE001
                print(f"[ui] set {name} failed: {type(e).__name__}: {str(e)[:120]}")
                error = "failed"
        data = {"name": name, "ok": not error, **({"error": error} if error else {})}
        cl.send_json(json.dumps({"type": "set_result", "data": data, "ts": time.time()}))

    def _set_refused(self, cl: _Client, name: str, value: Any) -> str:
        """Why a `set` isn't done ("" when it is)."""
        if cl.kind not in _MAY_SET:
            print(f"[ui] set refused: from a {cl.kind}, not a host or NEO.app's page")
            return "forbidden"
        if name not in SETTABLE:
            return "unknown"
        if type(value) is not SETTABLE[name]:  # (exactly: 1 is not true)
            return "invalid"
        return ""

    def _relay(self, ev) -> None:
        msg = ev.to_dict()
        if ev.type == "erased":
            self._recent.clear()  # late-joining overlays must not get the old transcript back
        if ev.type == "ui_host":
            self._to_shell({"t": "ui_host", "host": ev.data.get("host")})  # NEO.app's window follows
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
        # (the Brains & Keys window hears what it shows: never a transcript)
        self._send_all(json.dumps(msg, default=str), lambda cl: cl.kind != "keys" or ev.type in _KEYS_HEARS)

    def _send_all(self, data: str, who: Callable[[_Client], bool]) -> None:
        for c, cl in list(self._clients.items()):
            if who(cl) and not cl.send_json(data):
                self._gone(c)  # not reading → gone

    def _gone(self, c: ServerConnection) -> None:
        if (cl := self._clients.pop(c, None)) is not None:
            if cl.kind == "host" and not self._host_attached():
                self._host_changed(False)  # NEO.app's menu-bar item returns
            from neo2.core.tools.browser_live import live

            live.viewer_gone(cl)
            from neo2.core.tools.camera.live import camera_live

            camera_live.viewer_gone(cl)
            # A client that stopped reading is cut off for good: close its socket, so the overlay
            # notices, reconnects and gets a fresh hello (an open socket that hears nothing never does).
            transport = getattr(c, "transport", None)
            if transport is not None:
                with contextlib.suppress(Exception):
                    transport.abort()

    def broadcast(
        self, type_: str, data: dict[str, Any], *, recent: bool = False, replay: dict[str, Any] | None = None
    ) -> None:
        """A browser card message (neo2/core/tools/browser_live.py). The card's opening and closing are
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
            self._send_all(json.dumps(msg, default=str), lambda cl: cl.kind != "keys" and (recent or cl.browser))

    def camera_send(self, type_: str, data: dict[str, Any], *, recent: bool = False) -> None:
        """A camera card message (neo2/core/tools/camera/live.py), to every overlay: its opening and
        closing are part of the transcript (replayed), its state goes to the clients connected now."""
        msg = {"type": type_, "data": data, "ts": time.time()}
        if recent:
            self._recent.append(msg)
            self._recent = self._recent[-60:]
        if self._clients:
            self._send_all(json.dumps(msg, default=str), lambda cl: cl.kind != "keys")

    def song_send(self, type_: str, data: dict[str, Any], *, recent: bool = False) -> None:
        """A song card message (neo2/core/tools/music/live.py): the camera card's lane. Its opening, its
        last state and its closing are part of the transcript (replayed); the rest goes to the clients
        connected now."""
        self.camera_send(type_, data, recent=recent)

    def viewers(self) -> int:
        """Overlays connected now (the camera turns on only where its card can be seen: never in
        Brains & Keys)."""
        return sum(1 for cl in self._clients.values() if cl.kind != "keys")

    def camera_frame(self, payload: bytes, who: Callable[[_Client], bool]) -> None:
        """One camera preview frame, into the latest-frame slot of each client showing the card."""
        for cl in list(self._clients.values()):
            if who(cl):
                cl.offer_camera(payload)

    async def _drain(self, cl: _Client) -> None:
        c = cl.conn
        try:
            while True:
                await cl.wake.wait()
                cl.wake.clear()
                while cl.q or cl.frame is not None or cl.cam is not None:
                    if cl.q:  # control and state beat pixels
                        data: str | bytes = cl.q.popleft()
                    elif cl.frame is not None:
                        data, cl.frame = cl.frame, None
                    else:
                        data, cl.cam = cl.cam, None  # type: ignore[assignment]
                    await asyncio.wait_for(c.send(data), 5.0)
        except (websockets.ConnectionClosed, TimeoutError, asyncio.CancelledError):
            pass
        finally:
            self._gone(c)

    def _hello(self, cl: _Client) -> dict[str, Any]:
        """What a client hears first. The Brains & Keys window: only what it shows."""
        try:
            summary: dict[str, Any] | None = brains.summary()  # NEO's brains: never a key, never its last 4
        except Exception as e:  # noqa: BLE001 — a hello without the brains beats no hello
            print(f"[ui] brains not in the hello: {type(e).__name__}")
            summary = None
        if cl.kind == "keys":
            return {
                "state": bus().state.value,
                "problems": health.problems(),
                "brains": summary,
                "shell": self.shell_snapshot(),
                "ui_host": settings().ui_host,
            }
        from neo2.core.tools.browser_live import live
        from neo2.core.tools.camera.live import camera_live

        return {
            "state": bus().state.value,
            "recent": self._recent,
            "confirm": confirm.card(confirm.peek()),  # a question still waiting
            "problems": health.problems(),  # no model yet, blocked mic, voice down…
            "browser": live.hello_snapshot(),  # NEO's browser card, if one is live
            "camera": camera_live.hello_snapshot(),  # the camera's card, while it is on
            "ui_host": settings().ui_host,  # NEO.app's window, or the notch
            "wake_word": settings().wake_word,  # "Listen for 'Neo'" (see `set`)
            "wellbeing_checkins": settings().wellbeing_checkins,
            "host_attached": cl.kind == "host" or self._host_attached(),
            "shell": self.shell_snapshot(),  # what NEO.app's menu shows
            "timers": _timers(),  # NEO's running timers (then `activity` events)
            "brains": summary,  # NEO's brains (then `brains` events)
        }

    async def _handle(self, c: ServerConnection) -> None:
        from neo2.core.tools.browser_live import live
        from neo2.core.tools.camera.live import camera_live
        from neo2.core.tools.music.live import song_live

        cl = _Client(c)
        request = getattr(c, "request", None)
        if request is not None:  # admitted, so it is one of the five kinds
            query = parse_qs(urlparse(request.path).query)
            role, ticket = (query.get("role") or [""])[0], (query.get("ticket") or [None])[0]
            origin = request.headers.get("Origin")
            if ticket is not None:
                # (checked at the door; checked again: the window may have closed since, or another
                # one got in first)
                if not (self._ticket_ok(ticket) and client_kind(origin, role, keys=True) == "keys") or self._keys_client():
                    print("[ui] a Brains & Keys connection was closed: its ticket is no good now")
                    with contextlib.suppress(Exception):
                        await c.close(1008, "forbidden")
                    return
                cl.kind = "keys"
            else:
                cl.kind = client_kind(origin, role) or "tool"
        first_host = cl.kind == "host" and not self._host_attached()
        # The hello goes first, before anything the bus relays to this client.
        cl.send_json(json.dumps({"type": "hello", "data": self._hello(cl)}, default=str))
        if first_host:
            self._host_changed(True)  # NEO.app puts its menu-bar item away
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
                # Brains & Keys first, by name: its commands never reach anything below (see the doc).
                if cl.kind == "keys":
                    if kind in brains.COMMANDS:
                        asyncio.create_task(self._keys_ask(cl, cmd))
                    elif kind == "brains_reload":
                        asyncio.create_task(self._reload_ask(cl))
                    elif kind == "state":
                        cl.send_json(json.dumps({"type": "state", "data": {"state": bus().state.value}}))
                    else:
                        print(f"[ui] {_cmd_name(kind)} refused: Brains & Keys sends only its own commands")
                    continue
                if kind in brains.COMMANDS:
                    self._keys_refused(cl, cmd)  # to that client alone; never the bus, never a value
                    continue
                if kind == "open_keys":
                    self._open_keys_ask(cl)
                    continue
                if kind == "brains_reload":
                    asyncio.create_task(self._reload_ask(cl))
                    continue
                if kind.startswith("browser_"):
                    # Straight to the browser: never the bus, never a log (it carries what the user types).
                    if kind == "browser_hello":
                        cl.browser = True
                    try:
                        live.command(cl, cmd, len(raw))
                    except Exception as e:  # noqa: BLE001
                        print(f"[ui] browser command failed: {type(e).__name__}")
                    continue
                if kind.startswith("camera_"):
                    try:
                        camera_live.command(cl, cmd)
                    except Exception as e:  # noqa: BLE001
                        print(f"[ui] camera command failed: {type(e).__name__}")
                    continue
                if kind.startswith("song_"):
                    try:
                        song_live.command(cl, cmd)
                    except Exception as e:  # noqa: BLE001
                        print(f"[ui] song command failed: {type(e).__name__}")
                    continue
                if kind == "shell":
                    self._shell_ask(cl, cmd)  # a host's alone; never the bus
                    continue
                if kind == "set":
                    asyncio.create_task(self._set_ask(cl, cmd))  # a host's, or NEO.app's page's
                    continue
                if kind == "shell_status":
                    # The shell says it itself, over its own socket: no websocket client speaks for it.
                    print(f"[ui] shell_status refused: from a {cl.kind} (the shell has its own socket)")
                    continue
                if kind == "stop":
                    camera_live.close("stopped")  # the stop button turns the camera off too
                    song_live.stop("stopped")  # …and stops listening for a song
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
            from neo2.core.logredact import redact

            await bus().publish("error", message=redact(str(e)))  # (every client hears it: no key in it)
