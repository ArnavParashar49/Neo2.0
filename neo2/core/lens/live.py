"""Search Screen's session in the core: the shell's ticket, the still, the area the user picked, and
the overlay page's commands (see neo2.core.lens for the whole protocol and the privacy rules).

One session at a time, owned by the ticket NEO.app's shell makes when the hold arms (or the menu,
the voice or a host opens Search Screen): ``lens_freeze`` brings a new ticket, which replaces any
earlier one — its still, its areas, its page connection all dropped — and takes the still (NEO's
page in the notch folds its host's panel first: ``lens_opening``, and a moment's wait there);
smart pick's regions are worked out as soon as the still is in (a hold is armed by then: one let go
of early never froze), the lines and paragraphs first, the things after; ``lens_commit`` says the
overlay is open; ``lens_closed`` revokes the ticket. The core closes it itself on Erase, on Stop,
when the shell goes, when the overlay's page has been gone for :data:`PAGE_GONE_S` without coming
back (a page that died leaves nothing to close it from), and 10 minutes after the still was taken
(``lens_close`` to the shell).

The page is the websocket's ``lens`` client: it connects with the ticket, asks ``lens_hello``, gets
the still as one binary NBF1 frame of kind 4 (to it alone), and then sends areas. Each area has a
number (``sel``) that only grows; a newer one calls off everything the older one still had running,
and a command naming an older one is refused ("stale"). The answers go to the page alone, as fixed
fields and fixed error words; the bus hears ``lens_opening`` and ``lens_continue``, with nothing in
them; the log says what ran and how long it took — never a pixel, a word of the area, a question,
an answer or a link.

Where an area went: every brain the picture or its text was sent to (as it was asked, not only the
one that answered), and a search phrase sent to NEO's web search for its links, rides each answer
the page gets as ``sent_to`` — the privacy line names all of them, "Stayed on this Mac" only when
nothing left.

Results in the notch. When NotchHub hosts NEO's UI and both its connection and NEO's page in it
said ``lens_card_hello``, the overlay asks ``lens_handoff`` for an area instead of showing its
sheet: the core makes the area a card with an id of its own (``cid``, 32 random hex), offers it —
the results to NEO's page in the notch, only ``{cid, open, why}`` to NotchHub's connection — and
waits :data:`HANDOFF_WAIT_S` for NotchHub to say it shows it (``lens_card_shown``). Until then
every result goes to the overlay too, so a refusal (NotchHub's camera or Allow / Cancel hold the
island) or no answer leaves the sheet complete. Shown, the overlay flies the area into the notch and
closes (``lens_closed`` why "notch"): its still, regions and ticket go at once, and the area alone
(its crop and results) stays until the card closes — Esc, a click elsewhere, an action that opens
something, Continue in NEO, Stop, Erase, a new Search Screen, :data:`CARD_IDLE_S` without a
command, or nobody left to show it. The overlay closing for anything but "notch" after NotchHub said
it shows the card (Esc a moment before the answer) closes the card too; an overlay that gave up
waiting says so (``lens_handoff {sel, cancel: true}``) and keeps its area; and NEO asking for an OK
gives the panel back to the chat (the card closes as a chat turn), whose confirm card no card hides. The card's own commands (``lens_card_*``, naming the live cid)
drive the same internals as the overlay's; nothing of a card goes on the bus, the replay or a log
line beyond a command's name.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import hmac
import io
import re
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from neo2.core import logger
from neo2.core.events import app_bus
from neo2.core.lens import ask, capture, local, regions, search
from neo2.core.lens.capture import Spot

log = logger(__name__)

V = 1
KIND_LENS = 4  # NBF1 kind of Search Screen's still (the browser's are 1 and 2, the camera's 3)
STILL_QUALITY = 90
STILL_TTL_S = 600.0  # the still is dropped this long after it was taken, open or not
FREEZE_WAIT_S = 2.5
FOLD_S = 0.3  # in the notch: the host's panel folds away before the still is taken
PAGE_GONE_S = 10.0  # the overlay's page gone this long (no reload brought it back): Search Screen closes
LISTEN_S = 12.0  # the mic button's "Listening…" ends by itself after this, if no sentence came
ASK_AFTER_S = 0.35  # the brain is asked once the text is read, or this long after the area, whichever is first
PREFILL_S = 10.0  # "search my screen for X": X waits this long for the overlay to ask for it
MAX_Q, MAX_CONTINUE = 500, 2000
VIAS = ("hold", "menu", "voice", "host")
HOWS = ("drag", "lasso", "pick", "key", "adjust", "all")
# (notch: the area was handed to NotchHub, and flew there)
WHYS = ("esc", "cancel", "summon", "action", "lock", "display", "idle", "gone", "continue", "done", "erased", "stop", "notch")
_TICKET = re.compile(r"[0-9a-f]{64}")
_ACT_ID = re.compile(r"a\d{1,3}")
_CLOSING = ("open", "mail", "call", "image_search", "text_search", "brains")

# Results in the notch (see the module's doc).
HANDOFF_WAIT_S = 0.8  # NotchHub says it shows the card within this, or the sheet shows by the area
HANDOFF_AREA_S = FREEZE_WAIT_S + 1.0  # a hand-off waits this long at most for its area to be cut
CARD_IDLE_S = 600.0  # the card closes after this long without a command of its own
CARD_PAGE_WAIT_S = 8.0  # …or when no page that can show it came within this
# Why a card closed: what NotchHub and NEO's page hear (anything else is "done").
CARD_WHYS = ("esc", "x", "away", "fold", "unshown", "timeout", "action", "continue", "chat", "stop", "erased", "replaced",
             "idle", "gone", "done")  # fmt: skip
_PAGE_CLOSES = ("esc", "x", "chat")  # what NEO's page may say closed it
_HOST_CLOSES = ("fold", "away", "unshown", "esc")  # …and NotchHub
_CID = re.compile(r"[0-9a-f]{32}")
# The messages about an area that a card shows (tagged with its cid for NEO's page in the notch).
_CARD_TYPES = frozenset({"lens_local", "lens_answer", "lens_links", "lens_reply", "lens_translation", "lens_done", "lens_listening"})
MAX_REPLIES = 12


@dataclass
class Hooks:
    """What the rest of NEO does for Search Screen (App sets these in NEO.app; tests their own)."""

    # Continue in NEO: (the user's text, the area's picture, NEO's context about it) → the chat
    continue_: Callable[[str, Any, str], Awaitable[Any]] | None = None
    # Brains & Keys opened (True when it was)
    open_keys: Callable[[], bool] | None = None
    # the voice armed, for a question about the area (as the summon keys arm it)
    listen: Callable[[], Awaitable[Any]] | None = None


@dataclass
class _Area:
    n: int
    rect: tuple[float, float, float, float]
    how: str
    crop: Any  # screen.Frame
    digest: str = ""
    question: str = ""
    part: Any = None  # the picture the brains get (ImagePart), made once
    found: local.Local | None = None
    answer: ask.Answer | None = None
    acts: dict[str, local.Action] = field(default_factory=dict)
    thread: list[tuple[str, str]] = field(default_factory=list)
    shown_local: dict[str, Any] | None = None
    shown_answer: dict[str, Any] | None = None
    shown_links: dict[str, Any] | None = None
    shown_translation: dict[str, Any] | None = None
    # the follow-ups answered so far, as the page was told them (with the question): a card's snapshot
    replies: list[dict[str, Any]] = field(default_factory=list)
    thumb: str = ""  # the area at 96 px (a JPEG data URI), made off the loop with the on-device pass
    tasks: set[asyncio.Task] = field(default_factory=set)
    # where the area (or what it holds) went: (label, what) → {label, local, what}; what: picture | text | query
    went: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)

    def sent(self, what: str) -> Callable[[str, bool], None]:
        """ask's ``sent`` for this area: each brain it goes to, kept."""

        def note(label: str, local: bool) -> None:
            self.went.setdefault((label, what), {"label": str(label)[:60], "local": bool(local), "what": what})

        return note

    def sent_to(self) -> list[dict[str, Any]]:
        return list(self.went.values())[:12]


@dataclass
class _Card:
    """An area handed to the notch: NotchHub shows its results instead of the overlay's sheet."""

    cid: str  # 32 random hex: the card's commands name it
    area: _Area
    ticket: str  # the overlay session it came from
    state: str = "offered"  # offered (NotchHub hasn't said yet) | shown
    shown: asyncio.Event = field(default_factory=asyncio.Event)  # (set too when it closes while offered)
    refused: bool = False
    translate_to: str = ""
    idle: asyncio.TimerHandle | None = None
    page_wait: asyncio.TimerHandle | None = None


def _jpeg(img: Any, quality: int) -> bytes:
    buf = io.BytesIO()
    (img if img.mode == "RGB" else img.convert("RGB")).save(buf, "JPEG", quality=quality, subsampling=0)
    return buf.getvalue()


def _digest(img: Any) -> str:
    """The area's pixels, told apart cheaply (a nearest-neighbour sample, never a copy of all of it)."""
    from PIL import Image

    sample = img.resize((min(img.width, 256), min(img.height, 256)), Image.NEAREST)
    return hashlib.blake2b(sample.tobytes(), digest_size=16).hexdigest() + f"{img.width}x{img.height}"


def _thumb(img: Any) -> str:
    t = img.convert("RGB")
    t.thumbnail((96, 96))
    return "data:image/jpeg;base64," + base64.b64encode(_jpeg(t, 80)).decode("ascii")


def _analyse(crop: Any) -> tuple[local.Local | None, str]:
    """The on-device pass over an area and its 96 px picture (the sheet's and the card's thumbnail),
    in one trip off the event loop. None for a pass that failed (its kind is logged, nothing else)."""
    try:
        thumb = _thumb(crop.image)
    except Exception as e:  # noqa: BLE001 — no thumbnail: an empty tile
        log.info(f"[lens] no thumbnail ({type(e).__name__})")
        thumb = ""
    try:
        return local.analyse(crop), thumb
    except Exception as e:  # noqa: BLE001 — nothing read on this Mac: the brain's answer still comes
        log.info(f"[lens] the on-device pass failed ({type(e).__name__})")
        return None, thumb


# ---- the clipboard (NEO.app's own write: the user pressed Copy) -----------------------------------


def _pasteboard() -> Any:
    from AppKit import NSPasteboard

    from neo2.core import effects

    effects.guard("clipboard", "write to the clipboard (Search Screen's Copy)", via=NSPasteboard)
    return NSPasteboard.generalPasteboard()


def copy_text(text: str) -> bool:
    from neo2.core.decide._objc import pool

    with pool():
        pb = _pasteboard()
        pb.clearContents()
        return bool(pb.setString_forType_(text, "public.utf8-plain-text"))


def copy_image(img: Any) -> bool:
    """The area on the clipboard as PNG and TIFF (what every app pastes)."""
    from Foundation import NSData

    from neo2.core.decide._objc import pool

    png, tiff = io.BytesIO(), io.BytesIO()
    rgb = img if img.mode == "RGB" else img.convert("RGB")
    rgb.save(png, "PNG", compress_level=1)
    rgb.save(tiff, "TIFF")
    with pool():
        pb = _pasteboard()
        pb.declareTypes_owner_(["public.png", "public.tiff"], None)
        a = pb.setData_forType_(NSData.dataWithBytes_length_(png.getvalue(), len(png.getvalue())), "public.png")
        b = pb.setData_forType_(NSData.dataWithBytes_length_(tiff.getvalue(), len(tiff.getvalue())), "public.tiff")
        return bool(a or b)


class LensLive:
    def __init__(self) -> None:
        self.channel: Any = None  # the websocket server (lens_send, lens_frame, lens_drop, lens_close, open_lens…)
        self.hooks = Hooks()
        self.ticket: str | None = None
        self.via = ""
        self.spot: Spot | None = None
        self.frame: Any = None  # the still (screen.Frame), in memory only
        self.frozen = "none"  # none | pending | ok | no
        self.why = ""  # no: no_permission | no_sck | failed
        self.freeze_ms = 0.0
        self.committed = False
        self.regions: dict[str, Any] | None = None
        self.texts = False  # the page asked for each line's words (VoiceOver)
        self.client: Any = None
        self.listening = False  # the mic button: the next sentence is a question about the area
        self._gen = 0
        self._still: bytes | None = None
        self._sid = b""
        self._area: _Area | None = None
        self._last_n = 0
        self._qid = 0
        self._tasks: set[asyncio.Task] = set()
        self._ttl: asyncio.TimerHandle | None = None
        self._freezing: asyncio.Task | None = None
        self._regioning: asyncio.Task | None = None
        self._prefill: tuple[str, float] | None = None  # ("search my screen for X": X, until when)
        self._ask = ""  # …taken by the session it opened: the first area's question
        self._sent_still: Any = None  # the client the still went to
        self._partial: dict[str, Any] | None = None  # smart pick's lines before its things are in
        self._eyes: tuple[dict[str, Any] | None, str] | None = None  # (eyes, eyes_why), worked out off the loop
        self._hello_owed: Any = None  # a client that got its hello before the eyes were known
        self._gone_timer: asyncio.TimerHandle | None = None
        self._listen_timer: asyncio.TimerHandle | None = None
        self._card: _Card | None = None  # the area handed to the notch (see the module's doc)

    # ---- the ticket ----------------------------------------------------------------------------

    @property
    def open(self) -> bool:
        return self.ticket is not None

    def ticket_ok(self, given: Any) -> bool:
        """``given`` is the live ticket the shell made (constant-time)."""
        want = self.ticket
        if want is None or not isinstance(given, str) or not _TICKET.fullmatch(given):
            return False
        return hmac.compare_digest(given.encode(), want.encode())

    def shell_said(self, msg: dict[str, Any]) -> None:
        """lens_freeze / lens_commit / lens_closed, from NEO.app's shell (its own socket only)."""
        kind = msg.get("t")
        ticket = msg.get("ticket")
        if not isinstance(ticket, str) or not _TICKET.fullmatch(ticket):
            log.info(f"[lens] a {kind} was ignored: not a ticket the shell makes")
            return
        if kind == "lens_freeze":
            self._freeze(ticket, msg.get("display"), str(msg.get("via") or ""))
        elif kind == "lens_commit":
            self._commit(ticket)
        elif kind == "lens_closed":
            if ticket == self.ticket:
                why = str(msg.get("why") or "")
                why = why if why in WHYS else "closed"
                card = self._card
                shown = card is not None and card.ticket == ticket and card.state == "shown"
                if shown and why == "notch":
                    # Handed to the notch: the overlay's things go, the area stays with the card —
                    # nothing the overlay does any more can end it.
                    self._release_overlay(why)
                else:
                    if shown:
                        # Closed some other way (Esc, the pill's ×, a right-click just before NotchHub's
                        # answer reached it): the user left — the card goes too, and never opens by
                        # itself after. (One only offered goes with the overlay: _drop.)
                        self._close_card(why if why in CARD_WHYS else "done")
                    self._drop(why)

    def _freeze(self, ticket: str, raw: Any, via: str) -> None:
        sp = capture.spot(raw)
        if sp is None:
            log.info("[lens] a freeze was ignored: the display isn't one NEO can read")
            return
        if ticket == self.ticket:
            return  # (said again: the same session)
        self._close_card("replaced")  # a new Search Screen: the card in the notch folds
        # (also after a hand-off, with no ticket left: the last session's numbers go)
        self._drop("replaced")
        self.ticket, self.spot, self.via = ticket, sp, via if via in VIAS else "hold"
        self.frozen, self.why, self.committed = "pending", "", False
        self._sid = hashlib.blake2b(ticket.encode(), digest_size=16).digest()
        if self._prefill is not None and time.monotonic() < self._prefill[1]:
            self._ask = self._prefill[0]
        self._prefill = None
        gen = self._gen
        log.info(f"[lens] open ({self.via}): {sp.w:g}×{sp.h:g} pt at {sp.scale:g}×")  # (never what it shows)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._ttl = loop.call_later(STILL_TTL_S, self._expire, gen)
        # NEO's page in the notch folds its host's panel away, so it isn't in the still (a hold and the
        # menu too, not only the voice and a host: it carries nothing).
        fold = self._notch()
        with contextlib.suppress(Exception):
            loop.create_task(app_bus().publish("lens_opening"))
        self._freezing = self._spawn(self._take(gen, FOLD_S if fold else 0.0))
        self._spawn(self._know_eyes(gen))

    @staticmethod
    def _notch() -> bool:
        try:
            from neo2.core.config import settings

            return getattr(settings(), "ui_host", "") == "notch"
        except Exception:  # noqa: BLE001
            return False

    async def _know_eyes(self, gen: int) -> None:
        """Which brain an area goes to first, worked out off the event loop (building the chain may
        import a provider's SDK): the hello reads it."""
        try:
            eyes = await asyncio.to_thread(ask.eyes)
            why = "" if eyes else await asyncio.to_thread(ask.no_brain_why)
        except Exception as e:  # noqa: BLE001
            log.info(f"[lens] no brain for Search Screen ({type(e).__name__})")
            eyes, why = None, "no_brain"
        if gen != self._gen:
            return
        self._eyes = (eyes, why)
        cl, self._hello_owed = self._hello_owed, None
        if cl is not None and cl is self.client:
            self._send("lens_hello", self.hello(), cl)  # (the page's first hello didn't know them yet)

    def _commit(self, ticket: str) -> None:
        if ticket != self.ticket or self.committed:
            return
        self.committed = True
        if self.frozen in ("ok", "no"):
            self._start_regions()
        # (still pending: the regions follow the still)

    async def _take(self, gen: int, fold: float = 0.0) -> None:
        sp = self.spot
        if sp is None:
            return
        if fold > 0:
            await asyncio.sleep(fold)
            if gen != self._gen:
                return
        try:
            res = await asyncio.wait_for(asyncio.to_thread(capture.freeze, sp), FREEZE_WAIT_S)
        except TimeoutError:
            res = capture.Freeze(None, "failed")
        except Exception as e:  # noqa: BLE001 — no still: live mode
            log.info(f"[lens] no still ({type(e).__name__}): live mode")
            res = capture.Freeze(None, "failed")
        if gen != self._gen:
            return  # closed (or replaced) while it was taken: dropped unanalysed
        if res.frame is not None:
            try:
                still = await asyncio.to_thread(_jpeg, res.frame.image, STILL_QUALITY)
            except Exception as e:  # noqa: BLE001
                log.info(f"[lens] the still couldn't be encoded ({type(e).__name__}): live mode")
                res = capture.Freeze(None, "failed")
                still = None
            if gen != self._gen:
                return
            if res.frame is not None:
                self.frame, self._still = res.frame, still
        self.frozen = "ok" if self.frame is not None else "no"
        self.why, self.freeze_ms = res.why, res.ms
        log.info(f"[lens] still: {self.frozen}{' (' + self.why + ')' if self.why else ''} in {res.ms:.0f} ms")
        if self.client is not None:
            self._send_frozen(self.client)
        self._start_regions()  # (on this Mac, from the still already here: ready for the first click)

    def _start_regions(self) -> None:
        if self.regions is None and (self._regioning is None or self._regioning.done()):
            self._regioning = self._spawn(self._regions(self._gen))

    async def _regions(self, gen: int) -> None:
        """Smart pick's regions, in two messages: the lines, paragraphs and windows first (``done``
        false), then everything with the things (saliency, the foreground instances) added."""
        if self.regions is not None or self.spot is None:
            return
        frame, sp = self.frame, self.spot
        empty = {"lines": [], "blocks": [], "objects": [], "instances": [], "windows": []}
        try:
            first = await asyncio.to_thread(regions.compute_text, frame, sp, texts=self.texts)
        except Exception as e:  # noqa: BLE001 — no regions: a click picks a box around the point
            log.info(f"[lens] no regions ({type(e).__name__})")
            first = dict(empty)
        if gen != self._gen:
            return
        if frame is not None:
            self._partial = {**first, "done": False}
            if self.client is not None:
                self._send("lens_regions", self._partial)
            try:
                things = await asyncio.to_thread(regions.compute_things, frame, sp)
            except Exception as e:  # noqa: BLE001
                log.info(f"[lens] no things for smart pick ({type(e).__name__})")
                things = {}
            if gen != self._gen:
                return
            first = {**first, **things}
        self.regions = {**empty, **first, "done": True}
        self._partial = None
        if self.client is not None:
            self._send("lens_regions", self.regions)

    def _expire(self, gen: int) -> None:
        if gen == self._gen and self.ticket is not None:
            log.info("[lens] the still is 10 minutes old: Search Screen closes")
            self.close("idle")

    # ---- closing ---------------------------------------------------------------------------------

    def close(self, why: str = "done") -> bool:
        """The core closes Search Screen: the card in the notch (if there is one) closes for NotchHub
        and NEO's page, the shell is told (it orders the overlay out), and everything is dropped
        now. False: nothing was open."""
        had = self._card is not None
        self._close_card(why)
        if self.ticket is None:
            return had
        ch = self.channel
        if ch is not None:
            with contextlib.suppress(Exception):
                ch.lens_close(why if why in WHYS else "done")
        self._drop(why)
        return True

    def _drop(self, why: str) -> None:
        was = self.ticket is not None
        card = self._card
        if card is not None and card.state == "offered" and card.ticket == self.ticket:
            # The overlay went before NotchHub said it shows its area: there is no card after all.
            self._close_card(why if why in CARD_WHYS else "gone")
        self._gen += 1
        for t in list(self._tasks):
            if not t.done():
                t.cancel()
        self._tasks.clear()
        if self._ttl is not None:
            self._ttl.cancel()
        self._ttl = self._freezing = self._regioning = None
        self.ticket, self.spot, self.frame, self._still, self.regions = None, None, None, None, None
        self.frozen, self.why, self.committed, self.listening, self.texts = "none", "", False, False, False
        self._area, self._sent_still, self._ask = None, None, ""
        self._last_n = self._qid = 0
        self._eyes, self._hello_owed, self._partial = None, None, None
        for timer in (self._gone_timer, self._listen_timer):
            if timer is not None:
                timer.cancel()
        self._gone_timer = self._listen_timer = None
        cl, self.client = self.client, None
        ch = self.channel
        if cl is not None and ch is not None:
            with contextlib.suppress(Exception):
                ch.lens_drop(cl)  # its connection ends: the ticket is no good any more
        if was:
            log.info(f"[lens] closed ({why}): the still and the area are gone")

    def stop(self) -> None:
        """The stop button: Search Screen closes too."""
        self.close("stop")

    def erased(self) -> None:
        self._prefill = None
        self.close("erased")

    def shell_gone(self) -> None:
        """NEO.app's shell is gone: nobody shows the overlay any more (and the core is going too)."""
        self._drop("gone")
        self._close_card("gone")

    # ---- the card in the notch -----------------------------------------------------------------

    def _handoff_possible(self) -> bool:
        """Whether NotchHub could show an area's results now (the hello's `notch`: a hint — the
        hand-off itself asks again)."""
        ch = self.channel
        try:
            return bool(ch is not None and ch.can_host_lens_card())
        except Exception:  # noqa: BLE001 — an older channel: no hand-off, the sheet
            return False

    async def _handoff(self, cl: Any, cmd: dict[str, Any]) -> None:
        """`lens_handoff {sel, to}` from the overlay: offer the area to the notch, and say whether
        NotchHub shows it (`ok`, with its island's display) or why not — then the sheet shows."""
        sel = cmd.get("sel")

        def reply(ok: bool, **extra: Any) -> None:
            self._send("lens_handoff", {"sel": sel, "ok": ok, **extra}, cl)

        if isinstance(sel, bool) or not isinstance(sel, int) or sel != self._last_n:
            reply(False, why="stale")
            return
        if cmd.get("cancel") is True:
            self._handoff_cancel(sel)
            return
        gen, loop = self._gen, asyncio.get_running_loop()
        until = loop.time() + HANDOFF_AREA_S
        while self._area is None or self._area.n != sel:  # (the area is still being cut)
            if gen != self._gen or sel != self._last_n or loop.time() >= until:
                reply(False, why="stale")
                return
            await asyncio.sleep(0.02)
        a, ch = self._area, self.channel
        if ch is None or not self._handoff_possible():
            reply(False, why="no_notch")
            return
        self._close_card("replaced")
        to = cmd.get("to")
        card = _Card(secrets.token_hex(16), a, self.ticket or "", translate_to=str(to) if local.valid_code(to) else "")
        self._card = card
        log.info(f"[lens] area {a.n} offered to the notch")
        # NEO's page first (it lays the card out before NotchHub opens for it), then NotchHub's own
        # connection, which hears only the card's id.
        with contextlib.suppress(Exception):
            ch.lens_card_send("lens_card", self._card_snapshot(card))
        with contextlib.suppress(Exception):
            ch.lens_card_host({"cid": card.cid, "open": True})
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(card.shown.wait(), HANDOFF_WAIT_S)
        if self._card is not card:  # refused, replaced or closed meanwhile
            reply(False, why="unshown" if card.refused else "stale")
            return
        if not card.shown.is_set():
            self._close_card("timeout")
            reply(False, why="timeout")
            return
        card.state = "shown"
        display = None
        with contextlib.suppress(Exception):
            d = ch.host_display()
            display = d if isinstance(d, int) and not isinstance(d, bool) and 0 <= d <= 0xFFFFFFFF else None
        log.info(f"[lens] area {a.n} is shown in the notch")
        reply(True, display=display)
        self._card_touch()
        with contextlib.suppress(Exception):
            if not ch.has_lens_card_page():
                self._card_page_wait()  # (NotchHub makes its page as the panel opens)

    def _handoff_cancel(self, sel: int) -> None:
        """The overlay gave up waiting for the answer to its hand-off (it shows the sheet now): a card
        made of that area closes — offered or already shown — and the area's results so far go to the
        overlay again, whatever only the card heard meanwhile. No reply: the overlay isn't waiting."""
        card = self._card
        if card is None or card.ticket != self.ticket or card.area.n != sel:
            return
        was = card.state
        self._close_card("timeout")
        log.info(f"[lens] the overlay gave up on the notch: the card closes ({was})")
        if self._area is card.area:
            self._resend(card.area)

    def chat_needed(self) -> None:
        """NEO asks for an OK (its ``confirm``): the chat must be seen, so a card in the notch gives the
        panel back as a chat turn (NotchHub keeps the panel, as the chat, which shows the confirm card);
        one only offered closes, and the sheet shows by the area."""
        if self._card is not None:
            self._close_card("chat")

    def _release_overlay(self, why: str) -> None:
        """The overlay closed after its area went to the notch: its still, regions, ticket and line
        go now; the area — its crop, its results and the work still running on it — stays with the
        card."""
        card = self._card
        keep = card.area.tasks if card is not None else set()
        self._gen += 1
        for t in list(self._tasks):
            if t not in keep and not t.done():
                t.cancel()
        if self._ttl is not None:
            self._ttl.cancel()
        self._ttl = self._freezing = self._regioning = None
        for timer in (self._gone_timer, self._listen_timer):
            if timer is not None:
                timer.cancel()
        self._gone_timer = self._listen_timer = None
        self.listening = False
        self.ticket, self.spot, self.frame, self._still, self.regions = None, None, None, None, None
        self.frozen, self.why, self.committed, self.texts = "none", "", False, False
        self._sent_still, self._partial, self._hello_owed, self._prefill, self._ask = None, None, None, None, ""
        cl, self.client = self.client, None
        ch = self.channel
        if cl is not None and ch is not None:
            with contextlib.suppress(Exception):
                ch.lens_drop(cl)
        log.info(f"[lens] the overlay closed ({why}): the still is gone, the area is in the notch")

    def _close_card(self, why: str) -> None:
        """The card closes, for NotchHub and NEO's page alike; an area the overlay no longer owns
        goes with it (and whatever still ran on it)."""
        card = self._card
        if card is None:
            return
        self._card = None
        why = why if why in CARD_WHYS else "done"
        for timer in (card.idle, card.page_wait):
            if timer is not None:
                timer.cancel()
        card.idle = card.page_wait = None
        card.shown.set()  # (a hand-off still waiting for NotchHub hears it at once)
        ch = self.channel
        if ch is not None:
            msg = {"cid": card.cid, "open": False, "why": why}
            with contextlib.suppress(Exception):
                ch.lens_card_send("lens_card", msg)
            with contextlib.suppress(Exception):
                ch.lens_card_host(msg)
        if (self.ticket is None or self.ticket != card.ticket) and self._area is card.area:
            for t in list(card.area.tasks):
                t.cancel()
            self._area = None
            self.listening = False
            if self._listen_timer is not None:
                self._listen_timer.cancel()
                self._listen_timer = None
        log.info(f"[lens] the card in the notch closed ({why})")

    def _card_snapshot(self, card: _Card) -> dict[str, Any]:
        """`lens_card` with everything a card shows, for NEO's page in the notch: the area's 96 px
        picture, its results so far and what the footer and the chips need — never the still, the
        crop, a link, a code or a key."""
        from neo2.core.config import settings

        a = card.area
        eyes, why = self._eyes if self._eyes is not None else (None, "")
        hello = {
            "eyes": eyes,
            "eyes_why": "" if eyes else why,
            "ask": bool(settings().lens_ask),
            "search": search.provider(),
            "langs": local.languages(),
        }
        return {
            "cid": card.cid,
            "open": True,
            "card": {
                "sel": a.n,
                "thumb": a.thumb or None,
                "question": a.question or None,
                "translate_to": card.translate_to,
                "hello": hello,
                "local": a.shown_local,
                "answer": a.shown_answer,
                "links": a.shown_links,
                "thread": list(a.replies),
                "translation": a.shown_translation,
                "sent_to": a.sent_to(),
            },
        }

    def _card_touch(self) -> None:
        """A command of the card's own: its idle clock starts again."""
        card = self._card
        if card is None:
            return
        if card.idle is not None:
            card.idle.cancel()
        with contextlib.suppress(RuntimeError):
            card.idle = asyncio.get_running_loop().call_later(CARD_IDLE_S, self._card_idle, card)

    def _card_idle(self, card: _Card) -> None:
        card.idle = None
        if self._card is card:
            log.info("[lens] the card in the notch was left alone for 10 minutes: it closes")
            self._close_card("idle")

    def _card_page_wait(self) -> None:
        card = self._card
        if card is None or card.page_wait is not None:
            return
        with contextlib.suppress(RuntimeError):
            card.page_wait = asyncio.get_running_loop().call_later(CARD_PAGE_WAIT_S, self._card_no_page, card)

    def _card_no_page(self, card: _Card) -> None:
        card.page_wait = None
        ch = self.channel
        has_page = False
        with contextlib.suppress(Exception):
            has_page = bool(ch is not None and ch.has_lens_card_page())
        if self._card is card and not has_page:
            log.info("[lens] no page in the notch to show the card: it closes")
            self._close_card("gone")

    def card_host_gone(self) -> None:
        """No connection of NotchHub's that shows cards is left."""
        self._close_card("gone")

    def card_page_gone(self) -> None:
        """No NEO page in the notch that shows cards is left: one has a moment to come back."""
        if self._card is not None and self._card.state == "shown":
            self._card_page_wait()

    def _card_refuse(self, cl: Any, cmd: str, error: str) -> None:
        ch = self.channel
        if ch is not None:
            with contextlib.suppress(Exception):
                ch.lens_card_reply(cl, "lens_refused", {"cmd": cmd if re.fullmatch(r"[a-z_]{1,32}", cmd) else "a command", "error": error})

    async def card_command(self, cl: Any, cmd: dict[str, Any]) -> None:
        """One of a card's commands (`lens_card_*`), from NEO's page in the notch or NotchHub's
        connection — the websocket has already checked which kind may send which. Every one but the
        hello names the live card's id."""
        name = str(cmd.get("cmd") or "")
        kind = str(getattr(cl, "kind", ""))
        try:
            card = self._card
            if name == "lens_card_hello":
                if kind == "page" and card is not None:
                    if card.page_wait is not None:
                        card.page_wait.cancel()  # (a page that can show it is here)
                        card.page_wait = None
                    ch = self.channel
                    if ch is not None:
                        with contextlib.suppress(Exception):
                            ch.lens_card_reply(cl, "lens_card", self._card_snapshot(card))
                return
            cid = cmd.get("cid")
            if not isinstance(cid, str) or not _CID.fullmatch(cid):
                self._card_refuse(cl, name, "unknown")
                return
            if card is None or not hmac.compare_digest(cid.encode(), card.cid.encode()):
                self._card_refuse(cl, name, "stale")
                return
            if name == "lens_card_shown":
                if card.state == "offered":
                    card.shown.set()
                return
            if name == "lens_card_close":
                why = str(cmd.get("why") or "")
                why = why if why in (_HOST_CLOSES if kind == "host" else _PAGE_CLOSES) else "done"
                if why == "unshown" and card.state == "offered":
                    card.refused = True  # (NotchHub can't show it: the sheet, by the area)
                self._close_card(why)
                return
            a = card.area
            self._card_touch()
            if name == "lens_card_ask":
                q, qid = cmd.get("q", ""), cmd.get("qid", 0)
                if not isinstance(q, str) or isinstance(qid, bool) or not isinstance(qid, int):
                    self._card_refuse(cl, name, "unknown")
                    return
                q = " ".join(q.split())[:MAX_Q]
                self._area_task(a, self._answer(a) if not q else self._follow(a, qid, q))
            elif name == "lens_card_translate":
                to = cmd.get("to")
                if not local.valid_code(to):
                    self._card_refuse(cl, name, "unknown")
                    return
                card.translate_to = str(to)
                self._area_task(a, self._translate(a, str(to)))
            elif name == "lens_card_act":
                aid = cmd.get("id")
                if not isinstance(aid, str) or not _ACT_ID.fullmatch(aid) or aid not in a.acts:
                    self._card_refuse(cl, name, "stale")
                    return
                await self._act(a, aid)
            elif name == "lens_card_continue":
                text = cmd.get("text", "")
                if not isinstance(text, str):
                    self._card_refuse(cl, name, "unknown")
                    return
                await self._continue(a, " ".join(text.split())[:MAX_CONTINUE])
            elif name == "lens_card_listen":
                if self.hooks.listen is None:
                    self._card_refuse(cl, name, "unknown")
                    return
                self._listening(True)
                await self.hooks.listen()
            else:
                self._card_refuse(cl, name, "unknown")
        finally:
            log.info(f"[lens] {name if re.fullmatch(r'[a-z_]{1,32}', name) else 'a command'}")  # (its name only)

    # ---- the page's connection ----------------------------------------------------------------

    def attach(self, cl: Any) -> None:
        self.client = cl
        self._sent_still = None
        if self._gone_timer is not None:
            self._gone_timer.cancel()  # (back: a reload, a WebContent crash the browser recovered)
            self._gone_timer = None

    def client_gone(self, cl: Any) -> None:
        """The overlay's page went. A page that comes back (a reload) carries on; one that doesn't
        within PAGE_GONE_S leaves an overlay nothing can close from inside: Search Screen closes."""
        if self.client is not cl:
            return
        self.client = None
        self.listening = False
        if self.ticket is None:
            return
        gen = self._gen
        with contextlib.suppress(RuntimeError):
            if self._gone_timer is not None:
                self._gone_timer.cancel()
            self._gone_timer = asyncio.get_running_loop().call_later(PAGE_GONE_S, self._page_gone, gen)

    def _page_gone(self, gen: int) -> None:
        self._gone_timer = None
        if gen == self._gen and self.ticket is not None and self.client is None:
            log.info("[lens] the overlay's page is gone and didn't come back: Search Screen closes")
            self.close("gone")

    def _send(self, type_: str, data: dict[str, Any], cl: Any = None) -> None:
        """To `cl` alone; else to the overlay — and, for an area handed to the notch, to NEO's page
        there with the card's id (and to the overlay as well only while NotchHub hasn't said it
        shows it, so a refusal leaves the sheet complete)."""
        ch = self.channel
        if ch is None:
            return
        card = self._card
        if cl is None and card is not None and type_ in _CARD_TYPES and data.get("sel", card.area.n) == card.area.n:
            with contextlib.suppress(Exception):
                ch.lens_card_send(type_, {**data, "cid": card.cid})
            if card.state != "offered":
                return
        cl = cl if cl is not None else self.client
        if cl is None:
            return
        with contextlib.suppress(Exception):
            ch.lens_send(cl, type_, data)

    def _refuse(self, cl: Any, cmd: str, error: str) -> None:
        self._send("lens_refused", {"cmd": cmd if re.fullmatch(r"[a-z_]{1,32}", cmd) else "a command", "error": error}, cl)

    def hello(self) -> dict[str, Any]:
        """What the page hears first (lens_hello), and the websocket hello's ``lens``."""
        from neo2.core.config import settings
        from neo2.core.tools.computer import screen

        s = settings()
        try:
            perms = {"screen": bool(screen.can_capture())}
        except Exception:  # noqa: BLE001
            perms = {"screen": False}
        # (the footer names it even when NEO_LENS_ASK waits for Ask; worked out off the loop at the
        # freeze — not known yet, the hello comes again once it is: _know_eyes)
        eyes, why = self._eyes if self._eyes is not None else (None, "")
        data: dict[str, Any] = {
            "v": V,
            "open": self.ticket is not None,
            "frozen": {"none": "no", "pending": "pending", "ok": "ok", "no": "no"}[self.frozen],
            "display": self.spot.shown() if self.spot else None,
            "via": self.via,
            "perms": perms,
            "eyes": eyes,
            "eyes_why": "" if eyes else why,
            "ask": bool(s.lens_ask),
            "search": search.provider(),
            "langs": local.languages(),
            "prefill": self._ask or None,
            "notch": self._handoff_possible(),  # (NotchHub could show the results: lens_handoff)
        }
        return data

    def _send_frozen(self, cl: Any) -> None:
        if self.frozen == "ok" and self._still is not None and self.frame is not None and self.spot is not None:
            if self._sent_still is not cl:
                from neo2.core.tools.browser_live import pack_frame

                img = self.frame.image
                payload = pack_frame(
                    self._sid, 1, self._still, kind=KIND_LENS, ts_ms=time.time() * 1000, vw=self.spot.w,
                    vh=self.spot.h, iw=img.width, ih=img.height, dpr=self.spot.scale,
                )  # fmt: skip
                ch = self.channel
                if ch is not None:
                    with contextlib.suppress(Exception):
                        ch.lens_frame(cl, payload)  # (to this client alone: never another kind)
                self._sent_still = cl
            self._send("lens_frozen", {"ok": True, "ms": round(self.freeze_ms)}, cl)
        elif self.frozen == "no":
            self._send("lens_frozen", {"ok": False, "why": self.why or "failed"}, cl)

    # ---- opening it from the core (the voice, the agent's tool) -----------------------------------

    def remember_ask(self, question: str) -> None:
        q = " ".join(str(question or "").split())[:200]
        self._prefill = (q, time.monotonic() + PREFILL_S) if q else None

    def open_overlay(self, question: str = "") -> str:
        """Ask NEO.app's shell to open Search Screen: "opened", "off", "no_app" or "busy"."""
        from neo2.core.config import settings

        if not getattr(settings(), "lens", False):
            return "off"
        ch = self.channel
        if ch is None or not getattr(ch, "has_shell", lambda: False)():
            return "no_app"
        got = ch.open_lens()
        if got == "opened":
            self.remember_ask(question)  # (before the shell's freeze, which comes back later)
        return got

    # ---- the voice, while it is open --------------------------------------------------------------

    def takes_voice(self) -> bool:
        """Is what the user says now about the area (open, a page connected, an area picked)? In the
        notch only one sentence, and only after the card's mic button: once the overlay is gone,
        the user's next "Neo, …" is about anything."""
        if self.ticket is not None and self.client is not None and self._area is not None:
            return True
        card = self._card
        return card is not None and card.state == "shown" and self.listening and self._area is card.area

    async def hear(self, text: str) -> str:
        """A sentence the user said while Search Screen is open with an area: "close" closes it,
        "continue in chat" hands it over, anything else is a question about it. What NEO says back
        ("" for nothing): the answer is in the sheet too."""
        from neo2.core.agent import lens_words

        a = self._area
        if a is None:
            return ""
        self._listening(False)
        c = lens_words.control(text)
        if c == "close":
            self.close("done")
            return ""
        if c == "continue":
            await self._continue(a, "")
            return ""
        self._qid += 1
        return await self._follow(a, 100_000 + self._qid, " ".join(str(text).split())[:MAX_Q])

    # ---- the page's commands ------------------------------------------------------------------------

    async def command(self, cl: Any, cmd: dict[str, Any]) -> None:
        """One of the page's lens_* commands (the websocket routes only the lens client's here)."""
        name = str(cmd.get("cmd") or "")
        if cl is not self.client or self.ticket is None:
            self._refuse(cl, name, "stale")
            return
        try:
            if name == "lens_hello":
                self.texts = cmd.get("texts") is True
                self._send("lens_hello", self.hello(), cl)
                if self._eyes is None:
                    self._hello_owed = cl
                self._send_frozen(cl)
                if self.regions is not None:
                    self._send("lens_regions", self.regions, cl)
                elif self._partial is not None:
                    self._send("lens_regions", self._partial, cl)
                elif self.frozen in ("ok", "no"):
                    self._start_regions()
                if (a := self._area) is not None:
                    self._resend(a)
                return
            if name == "lens_select":
                await self._select(cl, cmd)
                return
            if name == "lens_handoff":
                await self._handoff(cl, cmd)  # (it waits for its area itself)
                return
            a = self._area
            sel = cmd.get("sel")
            if a is None or isinstance(sel, bool) or not isinstance(sel, int) or sel != a.n:
                self._refuse(cl, name, "stale")
                return
            if name == "lens_ask":
                q, qid = cmd.get("q", ""), cmd.get("qid", 0)
                if not isinstance(q, str) or isinstance(qid, bool) or not isinstance(qid, int):
                    self._refuse(cl, name, "unknown")
                    return
                q = " ".join(q.split())[:MAX_Q]
                if not q:  # "Ask" (NEO_LENS_ASK off) or "Try again"
                    self._area_task(a, self._answer(a))
                else:
                    self._area_task(a, self._follow(a, qid, q))
            elif name == "lens_translate":
                to = cmd.get("to")
                if not local.valid_code(to):
                    self._refuse(cl, name, "unknown")
                    return
                self._area_task(a, self._translate(a, str(to)))
            elif name == "lens_act":
                aid = cmd.get("id")
                if not isinstance(aid, str) or not _ACT_ID.fullmatch(aid) or aid not in a.acts:
                    self._refuse(cl, name, "stale")
                    return
                await self._act(a, aid)
            elif name == "lens_continue":
                text = cmd.get("text", "")
                if not isinstance(text, str):
                    self._refuse(cl, name, "unknown")
                    return
                await self._continue(a, " ".join(text.split())[:MAX_CONTINUE])
            elif name == "lens_listen":
                if self.hooks.listen is None:
                    self._refuse(cl, name, "unknown")
                    return
                self._listening(True)
                await self.hooks.listen()
            else:
                self._refuse(cl, name, "unknown")
        finally:
            log.info(f"[lens] {name if re.fullmatch(r'[a-z_]{1,32}', name) else 'a command'}")  # (its name only)

    def _listening(self, on: bool) -> None:
        """The mic button: the next sentence is a question about the area — and the page is told,
        to show "Listening…" (it hears nothing of the bus): on when armed, off when the sentence
        comes or LISTEN_S passed."""
        was, self.listening = self.listening, on
        if self._listen_timer is not None:
            self._listen_timer.cancel()
            self._listen_timer = None
        if on:
            gen = self._gen
            with contextlib.suppress(RuntimeError):
                self._listen_timer = asyncio.get_running_loop().call_later(LISTEN_S, self._listen_over, gen)
        if on or was:
            self._send("lens_listening", {"on": on})

    def _listen_over(self, gen: int) -> None:
        self._listen_timer = None
        if gen == self._gen and self.listening:
            self._listening(False)

    def _spawn(self, coro: Awaitable[Any]) -> asyncio.Task:
        t = asyncio.ensure_future(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)
        t.add_done_callback(_quiet)
        return t

    def _area_task(self, a: _Area, coro: Awaitable[Any]) -> asyncio.Task:
        t = self._spawn(coro)
        a.tasks.add(t)
        t.add_done_callback(a.tasks.discard)
        return t

    def _current(self, a: _Area) -> bool:
        """The area is the one being worked on: the overlay's, or the card's in the notch."""
        if self._area is not a:
            return False
        return self.ticket is not None or (self._card is not None and self._card.area is a)

    async def _select(self, cl: Any, cmd: dict[str, Any]) -> None:
        sel, how = cmd.get("sel"), str(cmd.get("how") or "drag")
        if isinstance(sel, bool) or not isinstance(sel, int) or sel <= self._last_n:
            self._refuse(cl, "lens_select", "stale")
            return
        sp = self.spot
        if sp is None:
            self._refuse(cl, "lens_select", "stale")
            return
        how = how if how in HOWS else "drag"
        rect = (0.0, 0.0, sp.w, sp.h) if how == "all" else capture.clamp(sp, cmd.get("rect"))
        if rect is None:
            self._refuse(cl, "lens_select", "unknown")
            return
        self._last_n = sel
        old, gen = self._area, self._gen
        if self._card is not None and self._card.state == "offered":
            self._close_card("replaced")  # (a new area before NotchHub said: that one stays here)
        if old is not None:
            for t in list(old.tasks):
                t.cancel()  # (the older area's answer is no longer wanted)
        if self.frozen == "pending" and self._freezing is not None and not cmd.get("live"):
            with contextlib.suppress(Exception):
                await asyncio.wait_for(asyncio.shield(self._freezing), FREEZE_WAIT_S)
        if gen != self._gen or sel != self._last_n:
            return
        live = self.frame is None
        if live:
            try:
                crop = await asyncio.to_thread(capture.grab_live, sp, rect)
            except PermissionError:
                self._send("lens_captured", {"sel": sel, "ok": False, "error": "no_permission"})
                return
            except Exception as e:  # noqa: BLE001
                log.info(f"[lens] the area wasn't captured ({type(e).__name__})")
                self._send("lens_captured", {"sel": sel, "ok": False, "error": "failed"})
                return
            if gen != self._gen or sel != self._last_n:
                return
            self._send("lens_captured", {"sel": sel, "ok": True})
        else:
            crop = capture.cut(self.frame, sp, rect)
        digest = await asyncio.to_thread(_digest, crop.image)
        if gen != self._gen or sel != self._last_n:
            return
        question, self._ask = self._ask, ""  # "search my screen for X": the first area's question
        a = _Area(sel, rect, how, crop, digest, question)
        self._area = a
        log.info(f"[lens] area {sel} ({how}{', live' if live else ''}): {rect[2]:.0f}×{rect[3]:.0f} pt")
        if old is not None and old.digest == digest and old.found is not None and old.answer is not None and not question:
            self._reuse(old, a)  # the same pixels as the last area: its results again (only once they're in)
            return
        self._area_task(a, self._work(a, live))

    def _reuse(self, old: _Area, a: _Area) -> None:
        a.part, a.found, a.answer, a.acts, a.thread = old.part, old.found, old.answer, old.acts, list(old.thread)
        a.went, a.thumb = dict(old.went), old.thumb
        a.shown_local = {**old.shown_local, "sel": a.n} if old.shown_local else None
        a.shown_answer = {**old.shown_answer, "sel": a.n} if old.shown_answer else None
        a.shown_links = {**old.shown_links, "sel": a.n} if old.shown_links else None
        self._resend(a)

    def _resend(self, a: _Area) -> None:
        for shown, kind in ((a.shown_local, "lens_local"), (a.shown_answer, "lens_answer"), (a.shown_links, "lens_links")):
            if shown is not None:
                self._send(kind, shown)

    def _act_id(self, a: _Area, act: local.Action) -> str:
        aid = f"a{len(a.acts) + 1}"
        a.acts[aid] = act
        return aid

    def _found(self, a: _Area, found: local.Local, live: bool) -> None:
        a.found = found
        provider = search.provider()
        acts = list(found.actions)
        if provider is not None and provider["image"] and not found.dark:
            acts.append(local.Action("image_search", f"Search with {provider['label']}", ""))
        shown_acts = [self._act_id(a, x) for x in acts]
        data: dict[str, Any] = {
            "sel": a.n,
            "text": found.text,
            "lines": found.lines,
            "lang": found.lang,
            "translate_to": local.translate_to(found.lang, found.guess),
            "dark": found.dark,
            "actions": [a.acts[i].shown(i) for i in shown_acts],
        }
        if a.thumb:
            data["thumb"] = a.thumb  # (always: the sheet's and the card's; made with the on-device pass)
        a.shown_local = data
        self._send("lens_local", data)

    async def _work(self, a: _Area, live: bool) -> None:
        """The on-device pass, and the brain asked as soon as its text is read (or ASK_AFTER_S after
        the area, without the text); a dark area is never asked about."""
        from neo2.core.config import settings

        lt = asyncio.ensure_future(asyncio.to_thread(_analyse, a.crop))
        at: asyncio.Future[Any] | None = None
        try:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(asyncio.shield(lt), ASK_AFTER_S)
            if lt.done():
                if not self._local_done(a, lt, live):
                    return
            if bool(settings().lens_ask) or bool(a.question):
                at = asyncio.ensure_future(self._answer(a, text=a.found.text if a.found else ""))
            else:
                self._answered(a, ask.Answer("failed", error="off"))  # (the page shows Ask)
            if not lt.done():
                await asyncio.wait([lt])
                if not self._local_done(a, lt, live):
                    return
            if at is not None:
                await at
        finally:
            for t in (lt, at):
                if t is not None and not t.done():
                    t.cancel()

    def _local_done(self, a: _Area, lt: Any, live: bool) -> bool:
        """The on-device pass is in: shown. False when nothing more is to be done for the area (it
        isn't the current one any more, or it came out black)."""
        found = self._local_result(a, lt)
        if not self._current(a):
            return False
        self._found(a, found, live)
        if found.dark:
            self._answered(a, ask.Answer("failed", error="dark"))
            return False
        return True

    @staticmethod
    def _local_result(a: _Area, t: Any) -> local.Local:
        found = None
        try:
            found, a.thumb = t.result()
        except Exception as e:  # noqa: BLE001 — nothing read on this Mac: the brain's answer still comes
            log.info(f"[lens] the on-device pass failed ({type(e).__name__})")
        return found if found is not None else local.Local(actions=[local.Action("copy_image", "Copy image", "")])

    async def _picture(self, a: _Area) -> Any:
        if a.part is None:
            from neo2.core.tools.computer import screen

            a.part = (await asyncio.to_thread(screen.render, a.crop, "region")).part
        return a.part

    def _answered(self, a: _Area, ans: ask.Answer, extra: list[dict[str, Any]] | None = None) -> None:
        a.answer = ans
        data = {"sel": a.n, **ans.shown(), "actions": extra or [], "sent_to": a.sent_to()}
        a.shown_answer = data
        self._send("lens_answer", data)

    async def _answer(self, a: _Area, text: str | None = None) -> None:
        if a.shown_answer is not None:
            a.shown_answer = {"sel": a.n, "state": "thinking"}  # (asked again: a card's snapshot says so)
        self._send("lens_answer", {"sel": a.n, "state": "thinking"})
        part = await self._picture(a)
        if not self._current(a):
            return
        if text is None:
            text = a.found.text if a.found else ""
        ans = await ask.answer(part, text=text, question=a.question, lang=ask.user_lang(), sent=a.sent("picture"))
        if not self._current(a):
            return
        extra: list[dict[str, Any]] = []
        if ans.state == "done" and ans.query:
            provider = search.provider()
            if provider is not None and provider["text"]:
                label = f"Search “{ans.query}”" + ("" if provider["image"] else f" on {provider['label']}")
                act = local.Action("text_search", label[:90], ans.query, detail=provider["label"])
                extra.append(act.shown(self._act_id(a, act)))
        elif ans.state == "failed" and ans.error in ("no_brain", "local_only", "key"):
            act = local.Action("brains", "Brains & Keys…", "")
            extra.append(act.shown(self._act_id(a, act)))
        self._answered(a, ans, extra)
        if ans.state == "done" and ans.query and (n := search.wanted(ans.kind, ans.shop)):
            if search.may_link():
                a.sent("query")("web search", False)  # (the answer's search phrase, to NEO's web search)
            rows = await asyncio.to_thread(search.links, ans.query, n)
            if not self._current(a) or not rows:
                return
            shown = []
            for r in rows:
                act = local.Action("open", r["title"], r["href"], detail=r["host"])
                shown.append({"id": self._act_id(a, act), "title": r["title"], "host": r["host"]})
            a.shown_links = {"sel": a.n, "rows": shown, "sent_to": a.sent_to()}
            self._send("lens_links", a.shown_links)

    async def _follow(self, a: _Area, qid: int, q: str) -> str:
        self._send("lens_reply", {"sel": a.n, "qid": qid, "state": "thinking"})
        part = await self._picture(a)
        if not self._current(a):
            return ""
        first = a.answer.brain if a.answer is not None else ""
        text, who = await ask.follow_up(
            part, text=a.found.text if a.found else "", history=a.thread, q=q, first=first, sent=a.sent("picture")
        )
        if not self._current(a):
            return ""
        if text:
            a.thread.append((q, text))
            del a.thread[: -ask.HISTORY]
            data = {"sel": a.n, "qid": qid, "state": "done", "text": text, "brain": {"label": who.label, "local": who.local},
                    "sent_to": a.sent_to()}  # fmt: skip
        else:
            data = {"sel": a.n, "qid": qid, "state": "failed", "error": who.error or "failed", "sent_to": a.sent_to()}
        a.replies.append({**{k: v for k, v in data.items() if k != "sent_to"}, "q": q})  # (a card's snapshot)
        del a.replies[:-MAX_REPLIES]
        self._send("lens_reply", data)
        return text

    async def _translate(self, a: _Area, to: str) -> None:
        name = local.language_name(to)
        text = a.found.text if a.found else ""
        if text:
            out, err = await ask.translate_text(text, name, sent=a.sent("text"))
        else:
            out, err = await ask.translate_picture(await self._picture(a), name, sent=a.sent("picture"))
        if not self._current(a):
            return
        log.info(f"[lens] translation: {'done' if out else err}")
        a.shown_translation = {"sel": a.n, "to": to, **({"text": out} if out else {"error": err or "failed"}), "sent_to": a.sent_to()}
        self._send("lens_translation", a.shown_translation)

    async def _act(self, a: _Area, aid: str) -> None:
        act = a.acts[aid]
        note, ok = "failed", False
        try:
            if act.kind == "open":
                from neo2.core.tools.computer import apps

                out = await apps.open_url(act.value)  # (a web link: the browser comes forward)
                ok = not str(out).startswith("Error")
                note = "opened" if ok else "failed"
            elif act.kind in ("mail", "call"):
                ok = await open_handler(act.value)  # (Mail or FaceTime — whatever opens it — never the browser)
                note = "opened" if ok else "failed"
            elif act.kind in ("copy_text", "wifi", "code"):
                ok = await asyncio.to_thread(copy_text, act.value)
                note = "copied" if ok else "failed"
            elif act.kind == "copy_image":
                ok = await asyncio.to_thread(copy_image, a.crop.image)
                note = "copied" if ok else "failed"
            elif act.kind == "image_search":
                note = await search.image_search(a.crop.image)
                ok = note == "opened"
            elif act.kind == "text_search":
                note = await search.text_search(act.value)
                ok = note == "opened"
            elif act.kind == "brains":
                ok = bool(self.hooks.open_keys and self.hooks.open_keys())
                note = "opened" if ok else "failed"
        except Exception as e:  # noqa: BLE001 — the chip says it failed; nothing of it in the log
            log.info(f"[lens] {act.kind} failed ({type(e).__name__})")
            ok, note = False, "failed"
        close = ok and act.kind in _CLOSING
        log.info(f"[lens] {act.kind}: {note}")
        self._send("lens_done", {"sel": a.n, "id": aid, "ok": ok, "note": note, "close": close})
        if close:
            self.close("action")

    def context(self, a: _Area) -> str:
        """NEO's context for the chat about the area: its title and the questions so far, fenced."""
        from neo2.core.agent.registry import fence

        parts = []
        if a.answer is not None and a.answer.state == "done":
            parts.append(f"{a.answer.title}: {a.answer.answer}")
        for q, r in a.thread[-ask.HISTORY :]:
            parts.append(f"Q: {q}\nA: {r}")
        body = "\n".join(parts)[:1200]
        lead = "The user selected this area of their screen with Search Screen (the picture is attached)."
        return lead + ("\n" + fence("search_screen", body) if body else "")

    async def _continue(self, a: _Area, text: str) -> None:
        hook = self.hooks.continue_
        if hook is None:
            self._send("lens_done", {"sel": a.n, "id": "", "ok": False, "note": "failed", "close": False})
            return
        part = await self._picture(a)
        context = self.context(a)
        self.close("continue")  # the overlay goes; the chat takes it from here
        with contextlib.suppress(Exception):
            await app_bus().publish("lens_continue")  # (nothing in it: the chat window comes forward)
        log.info("[lens] continued in NEO's chat")
        task = asyncio.ensure_future(hook(text, part, context))
        task.add_done_callback(_quiet)


async def open_handler(url: str) -> bool:
    """A mailto: or tel: link, opened by the app this Mac opens it with (LaunchServices brings that
    app forward itself): never through the browser, never made NEO's target app."""
    if not re.match(r"^(?:mailto|tel):", url, re.I):
        return False
    from neo2.core.tools.computer import apps

    code, _out, _err = await apps._run(["open", url], timeout=10)
    return code == 0


def first_sentence(text: str, most: int = 240) -> str:
    """The first sentence of an answer, for saying aloud (the sheet shows all of it)."""
    t = " ".join(str(text or "").split())
    for i, ch in enumerate(t):
        if ch in ".!?" and (i + 1 == len(t) or t[i + 1] == " ") and i >= 12:
            return t[: i + 1][:most]
    return t[:most]


def _quiet(t: asyncio.Task) -> None:
    if not t.cancelled() and (e := t.exception()) is not None:
        log.info(f"[lens] a task failed ({type(e).__name__})")


lens_live = LensLive()


def _on_event(ev) -> None:
    if ev.type == "erased":
        lens_live.erased()
    elif ev.type == "confirm":
        lens_live.chat_needed()  # (NEO asks for an OK: no card hides the question)


app_bus().subscribe(_on_event)


def reset() -> None:
    """A fresh session (tests)."""
    channel, hooks = lens_live.channel, lens_live.hooks
    lens_live._close_card("gone")
    lens_live._drop("reset")
    lens_live.__dict__.update(LensLive().__dict__)
    lens_live.channel, lens_live.hooks = channel, hooks


__all__ = ["KIND_LENS", "Hooks", "LensLive", "copy_image", "copy_text", "first_sentence", "lens_live", "open_handler", "reset"]
