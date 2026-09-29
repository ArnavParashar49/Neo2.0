"""The song card in the chat: NEO listening for a song, then what it found.

JSON through NEO's websocket beside the bus's, in the camera card's lane (neo2.core.server.ws):

* ``song_open {sid, source, listen_s}`` — the card appears (kept in the transcript replay). ``source``
  is "mic", "mac" (the Mac's own sound) or "both".
* ``song_state {sid, status, level, elapsed_s, message, result}`` — status is listening | matching |
  found | not_found | error. ``level`` (0..1) is how loud what NEO hears is, at most ten updates a
  second while it listens; ``elapsed_s`` how much it has heard. ``result``, once found: {title,
  artist, album, year, genre, cover, links: {apple_music, shazam}}, the cover a ``data:image/jpeg``
  URL NEO fetched itself (the UI never loads a remote picture). A card's last state is kept in the
  replay too.
* ``song_close {sid, reason}`` — done | stopped | error, or "erased" (an erase wiped the chat: the UI
  drops every song card, as it does the camera's). Kept in the replay.

UI → core: ``song_stop {sid}`` (stop listening now), ``song_link {sid, which}`` (open that card's
"apple_music" or "shazam" link with the Mac's default handler: only a link of that card's own result,
only https), ``song_hello`` (the live card's state again, for a UI that reloaded). Listening also stops
on the stop button, a spoken or typed "stop", Session.cancel_all() and an erase.

What NEO hears never reaches the card, the event bus, the replay, a log or the disk: only how loud it is.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Any

from neo2.core import logger
from neo2.core.events import app_bus
from neo2.core.tools.browser_live import new_sid
from neo2.core.tools.music import capture
from neo2.core.tools.music.recognize import COVER_S, safe_link

log = logger(__name__)

STATUSES = ("listening", "matching", "found", "not_found", "error")
TICK_S = 0.1  # the meter: at most ten states a second
LINKS_KEPT = 8  # the cards whose links still open (a card from long ago: nothing)
_KEEP: Any = object()


class SongLive:
    def __init__(self) -> None:
        self.channel: Any = None  # the websocket server: song_send(type, data, recent=…)
        self.source_factory: Callable[[str], capture.Source] | None = None  # tests: fake sources
        self.opener: Callable[[str], Any] | None = None  # tests: in place of the Mac's default handler
        self.sid = ""
        self.source = ""
        self.listen_s = 0.0
        self.status = ""
        self.level = 0.0
        self.elapsed = 0.0
        self.message: str | None = None
        self.result: dict[str, Any] | None = None
        self.listener: capture.Listener | None = None
        self.stopped: asyncio.Event = asyncio.Event()  # set when this card's listening is stopped or over
        self.starting = False  # a listen on its way: its sources starting, its card not open yet
        self.halted = False  # …and stopped before its card opened (song_identify then never opens it)
        self._sent_at = 0.0
        self._links: OrderedDict[str, dict[str, str]] = OrderedDict()
        self._tasks: set[asyncio.Task] = set()

    # ---- the state ------------------------------------------------------------------------------

    @property
    def listening(self) -> bool:
        """Capturing sound now, or about to (the microphone or the Mac's)."""
        listener = self.listener
        return self.starting or (listener is not None and listener.capturing)

    @property
    def busy(self) -> bool:
        """A listen starting, or a card still listening or matching (a second ask waits for its answer)."""
        return self.starting or (bool(self.sid) and self.status in ("listening", "matching"))

    def begin(self) -> None:
        """A listen is on its way (its sources start before its card opens)."""
        self.starting, self.halted = True, False

    def begun(self) -> bool:
        """Its sources are started (or failed): False when a stop came in the meantime."""
        self.starting = False
        return not self.halted

    def listener_for(self, names: list[str]) -> capture.Listener:
        make = self.source_factory or capture.source
        return capture.Listener([make(n) for n in names])

    def open(self, listener: capture.Listener, source: str, listen_s: float) -> str:
        """The card appears, listening through ``listener`` (already started). Returns its sid."""
        if self.sid:
            self.close("done")  # (one card at a time: a finished one still fetching its cover)
        self.sid = new_sid()[0]
        self.source, self.listen_s = source, float(listen_s)
        self.status, self.level, self.elapsed, self.message, self.result = "", 0.0, 0.0, None, None
        self.listener = listener
        self.stopped = asyncio.Event()
        self._send("song_open", {"sid": self.sid, "source": source, "listen_s": self.listen_s}, recent=True)
        self.update("listening")
        log.info(f"[song] listening ({source}; the user asked)")  # that it listened — never what it heard
        return self.sid

    def update(
        self,
        status: str | None = None,
        *,
        level: float | None = None,
        elapsed: float | None = None,
        message: Any = _KEEP,
        result: Any = _KEEP,
        recent: bool = False,
    ) -> None:
        """The card's new state. A change of level or time alone is sent at most every ``TICK_S``."""
        if not self.sid:
            return
        news = recent
        if status in STATUSES and status != self.status:
            self.status, news = status, True
        if message is not _KEEP and message != self.message:
            self.message, news = (str(message)[:300] if message else None), True
        if result is not _KEEP:
            self.result, news = result, True
        if level is not None:
            self.level = min(1.0, max(0.0, float(level)))
        if elapsed is not None:
            self.elapsed = max(0.0, float(elapsed))
        now = time.monotonic()
        if not news and now - self._sent_at < TICK_S:
            return
        self._sent_at = now
        self._send("song_state", self._state(), recent=recent)

    def _state(self) -> dict[str, Any]:
        return {
            "sid": self.sid, "status": self.status, "level": round(self.level, 3),
            "elapsed_s": round(self.elapsed, 1), "message": self.message, "result": self.result,
        }  # fmt: skip

    def state(self) -> dict[str, Any] | None:
        """The live card's state (None when no card is live)."""
        return self._state() if self.sid else None

    # ---- the end of a card ----------------------------------------------------------------------

    def finish(
        self, status: str, *, message: str | None = None, result: dict | None = None, reason: str = "done"
    ) -> None:
        """The card's last state (kept in the replay), then its close."""
        if not self.sid:
            return
        if self.listener is not None:
            self.listener.stop()
        self._keep_links(result)
        self.update(status, level=0.0, message=message, result=result, recent=True)
        self.close(reason)

    def found(
        self, result: dict, cover_url: str | None, fetch: Callable[[str], Awaitable[str | None]]
    ) -> None:
        """The song, on the card at once (its links work from now on); its cover art follows (at most
        ``COVER_S`` later), then the card closes."""
        sid = self.sid
        if not sid:
            return
        if self.listener is not None:
            self.listener.stop()
        self._keep_links(result)
        # Kept in the replay too: a card stopped while its cover is on its way still shows its song.
        self.update("found", level=0.0, message=None, result={**result, "cover": None}, recent=True)

        async def cover_then_close() -> None:
            pic = None
            if cover_url:
                with contextlib.suppress(Exception):
                    pic = await asyncio.wait_for(fetch(cover_url), COVER_S + 0.5)
            if self.sid == sid:
                self.finish("found", result={**result, "cover": pic})

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # no loop (a test calling it bare): no cover
            self.finish("found", result={**result, "cover": None})
            return
        task = loop.create_task(cover_then_close())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def close(self, reason: str = "done") -> bool:
        """Listening off and the card closed (False: no card was live)."""
        sid = self.sid
        if not sid:
            return False
        listener, self.listener = self.listener, None
        if listener is not None:
            listener.stop()  # the microphone and the Mac's sound off, what they heard wiped
        self.stopped.set()
        self.sid = ""
        self._send("song_close", {"sid": sid, "reason": reason}, recent=True)
        log.info(f"[song] card closed ({reason})")
        self.status, self.level, self.elapsed, self.message, self.result = "", 0.0, 0.0, None, None
        return True

    def stop(self, reason: str = "stopped") -> bool:
        """Stop listening now: the card's stop, the stop button, "stop", an erase (False: nothing was
        live). A listen still starting never opens its card."""
        was_starting = self.starting
        if was_starting:
            self.halted = True
        return self.close(reason) or was_starting

    def _keep_links(self, result: dict | None) -> None:
        links = {k: v for k, v in ((result or {}).get("links") or {}).items() if safe_link(v)}
        if links and self.sid:
            self._links[self.sid] = links
            while len(self._links) > LINKS_KEPT:
                self._links.popitem(last=False)

    def _send(self, type_: str, data: dict, *, recent: bool = False) -> None:
        ch = self.channel
        if ch is not None:
            with contextlib.suppress(Exception):
                ch.song_send(type_, data, recent=recent)

    # ---- the UI's commands (never the bus, never a log) -----------------------------------------

    def command(self, conn: Any, m: dict) -> None:
        kind = str(m.get("cmd") or "")
        sid = str(m.get("sid") or "")
        if kind == "song_stop":
            if not sid or sid == self.sid:  # (a card from before: its listening is over already)
                self.stop("stopped")
        elif kind == "song_link":
            url = (self._links.get(sid) or {}).get(str(m.get("which") or ""))
            if url and safe_link(url):
                self._open(url)
        elif kind == "song_hello" and conn is not None:
            snap = self.state()
            if snap is not None and hasattr(conn, "send_json"):
                conn.send_json(json.dumps({"type": "song_state", "data": snap, "ts": time.time()}))

    def _open(self, url: str) -> None:
        if self.opener is not None:
            self.opener(url)
            return
        with contextlib.suppress(RuntimeError):  # (no loop: nothing to open it with)
            task = asyncio.get_running_loop().create_task(_open_default(url))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    def forget(self) -> None:
        """An erase: the card off (closed as "erased": the UI drops it with the rest of the chat), no
        old card's links open any more, no cover still on its way."""
        self.stop("erased")
        self._links.clear()
        for t in list(self._tasks):
            t.cancel()


async def _open_default(url: str) -> None:
    """``url`` in whatever app the Mac opens it with (macOS's ``open``: the default handler)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "open", url, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
        )
        await asyncio.wait_for(proc.wait(), 10.0)
    except Exception as e:  # noqa: BLE001
        log.warning(f"[song] couldn't open the link ({type(e).__name__})")


song_live = SongLive()


def _on_event(ev) -> None:
    if ev.type == "erased":
        song_live.forget()


app_bus().subscribe(_on_event)


def reset() -> None:
    """The card closed and a fresh one (tests)."""
    song_live.forget()
    channel, factory, opener = song_live.channel, song_live.source_factory, song_live.opener
    song_live.__dict__.update(SongLive().__dict__)
    song_live.channel, song_live.source_factory, song_live.opener = channel, factory, opener
