"""Search Screen's session in the core: the shell's ticket, the still, the area the user picked, and
the overlay page's commands (see neo2.core.lens for the whole protocol and the privacy rules).

One session at a time, owned by the ticket NEO.app's shell makes when the hold arms (or the menu,
the voice or a host opens Search Screen): ``lens_freeze`` brings a new ticket, which replaces any
earlier one — its still, its areas, its page connection all dropped — and takes the still;
``lens_commit`` says the overlay is open (smart pick's regions are worked out then, never for a
hold let go of early); ``lens_closed`` revokes the ticket. The core closes it itself on Erase, on
Stop, when the shell goes, and 10 minutes after the still was taken (``lens_close`` to the shell).

The page is the websocket's ``lens`` client: it connects with the ticket, asks ``lens_hello``, gets
the still as one binary NBF1 frame of kind 4 (to it alone), and then sends areas. Each area has a
number (``sel``) that only grows; a newer one calls off everything the older one still had running,
and a command naming an older one is refused ("stale"). The answers go to the page alone, as fixed
fields and fixed error words; the bus hears ``lens_opening`` and ``lens_continue``, with nothing in
them; the log says what ran and how long it took — never a pixel, a word of the area, a question,
an answer or a link.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import hmac
import io
import re
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
ASK_AFTER_S = 0.35  # the brain is asked once the text is read, or this long after the area, whichever is first
PREFILL_S = 10.0  # "search my screen for X": X waits this long for the overlay to ask for it
MAX_Q, MAX_CONTINUE = 500, 2000
VIAS = ("hold", "menu", "voice", "host")
HOWS = ("drag", "lasso", "pick", "key", "adjust", "all")
WHYS = ("esc", "cancel", "summon", "action", "lock", "display", "idle", "gone", "continue", "done", "erased", "stop")
_TICKET = re.compile(r"[0-9a-f]{64}")
_ACT_ID = re.compile(r"a\d{1,3}")
_CLOSING = ("open", "mail", "call", "image_search", "text_search", "brains")


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
    tasks: set[asyncio.Task] = field(default_factory=set)


def _jpeg(img: Any, quality: int) -> bytes:
    buf = io.BytesIO()
    (img if img.mode == "RGB" else img.convert("RGB")).save(buf, "JPEG", quality=quality, subsampling=0)
    return buf.getvalue()


def _digest(img: Any) -> str:
    return hashlib.blake2b(img.tobytes(), digest_size=16).hexdigest() + f"{img.width}x{img.height}"


def _thumb(img: Any) -> str:
    t = img.convert("RGB")
    t.thumbnail((96, 96))
    return "data:image/jpeg;base64," + base64.b64encode(_jpeg(t, 80)).decode("ascii")


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
                self._drop(why if why in WHYS else "closed")

    def _freeze(self, ticket: str, raw: Any, via: str) -> None:
        sp = capture.spot(raw)
        if sp is None:
            log.info("[lens] a freeze was ignored: the display isn't one NEO can read")
            return
        if ticket == self.ticket:
            return  # (said again: the same session)
        if self.ticket is not None:
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
        self._freezing = self._spawn(self._take(gen))

    def _commit(self, ticket: str) -> None:
        if ticket != self.ticket or self.committed:
            return
        self.committed = True
        if self.frozen in ("ok", "no"):
            self._start_regions()
        # (still pending: the regions follow the still)

    async def _take(self, gen: int) -> None:
        sp = self.spot
        if sp is None:
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
        if self.committed:
            self._start_regions()

    def _start_regions(self) -> None:
        if self.regions is None and (self._regioning is None or self._regioning.done()):
            self._regioning = self._spawn(self._regions(self._gen))

    async def _regions(self, gen: int) -> None:
        if self.regions is not None or self.spot is None:
            return
        frame, sp = self.frame, self.spot
        try:
            got = await asyncio.to_thread(regions.compute, frame, sp, texts=self.texts)
        except Exception as e:  # noqa: BLE001 — no regions: a click picks a box around the point
            log.info(f"[lens] no regions ({type(e).__name__})")
            got = {"lines": [], "blocks": [], "objects": [], "instances": [], "windows": []}
        if gen != self._gen:
            return
        self.regions = got
        if self.client is not None:
            self._send("lens_regions", got)

    def _expire(self, gen: int) -> None:
        if gen == self._gen and self.ticket is not None:
            log.info("[lens] the still is 10 minutes old: Search Screen closes")
            self.close("idle")

    # ---- closing ---------------------------------------------------------------------------------

    def close(self, why: str = "done") -> bool:
        """The core closes Search Screen: the shell is told (it orders the overlay out), and
        everything is dropped now. False: nothing was open."""
        if self.ticket is None:
            return False
        ch = self.channel
        if ch is not None:
            with contextlib.suppress(Exception):
                ch.lens_close(why if why in WHYS else "done")
        self._drop(why)
        return True

    def _drop(self, why: str) -> None:
        was = self.ticket is not None
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
        """NEO.app's shell is gone: nobody shows the overlay any more."""
        self._drop("gone")

    # ---- the page's connection ----------------------------------------------------------------

    def attach(self, cl: Any) -> None:
        self.client = cl
        self._sent_still = None

    def client_gone(self, cl: Any) -> None:
        if self.client is cl:
            self.client = None

    def _send(self, type_: str, data: dict[str, Any], cl: Any = None) -> None:
        cl = cl if cl is not None else self.client
        ch = self.channel
        if cl is None or ch is None:
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
        eyes = ask.eyes()  # (the footer names it even when NEO_LENS_ASK waits for Ask)
        data: dict[str, Any] = {
            "v": V,
            "open": self.ticket is not None,
            "frozen": {"none": "no", "pending": "pending", "ok": "ok", "no": "no"}[self.frozen],
            "display": self.spot.shown() if self.spot else None,
            "via": self.via,
            "perms": perms,
            "eyes": eyes,
            "eyes_why": "" if eyes else ask.no_brain_why(),
            "ask": bool(s.lens_ask),
            "search": search.provider(),
            "langs": local.languages(),
            "prefill": self._ask or None,
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
        """Is what the user says now about the area (open, a page connected, an area picked)?"""
        return self.ticket is not None and self.client is not None and self._area is not None

    async def hear(self, text: str) -> str:
        """A sentence the user said while Search Screen is open with an area: "close" closes it,
        "continue in chat" hands it over, anything else is a question about it. What NEO says back
        ("" for nothing): the answer is in the sheet too."""
        from neo2.core.agent import lens_words

        a = self._area
        if a is None:
            return ""
        self.listening = False
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
                self._send_frozen(cl)
                if self.regions is not None:
                    self._send("lens_regions", self.regions, cl)
                elif self.committed and self.frozen in ("ok", "no"):
                    self._start_regions()
                if (a := self._area) is not None:
                    self._resend(a)
                return
            if name == "lens_select":
                await self._select(cl, cmd)
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
                self.listening = True
                await self.hooks.listen()
            else:
                self._refuse(cl, name, "unknown")
        finally:
            log.info(f"[lens] {name if re.fullmatch(r'[a-z_]{1,32}', name) else 'a command'}")  # (its name only)

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
        return self._area is a and self.ticket is not None

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
        if old is not None and old.digest == digest and old.found is not None and not question:
            self._reuse(old, a)  # the same pixels as the last area: its results again
            return
        self._area_task(a, self._work(a, live))

    def _reuse(self, old: _Area, a: _Area) -> None:
        a.part, a.found, a.answer, a.acts, a.thread = old.part, old.found, old.answer, old.acts, list(old.thread)
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
            "translate_to": local.translate_to(found.lang),
            "dark": found.dark,
            "actions": [a.acts[i].shown(i) for i in shown_acts],
        }
        if live:
            with contextlib.suppress(Exception):
                data["thumb"] = _thumb(a.crop.image)
        a.shown_local = data
        self._send("lens_local", data)

    async def _work(self, a: _Area, live: bool) -> None:
        """The on-device pass, and the brain asked as soon as its text is read (or ASK_AFTER_S after
        the area, without the text); a dark area is never asked about."""
        from neo2.core.config import settings

        lt = asyncio.ensure_future(asyncio.to_thread(local.analyse, a.crop))
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
        found = self._local_result(lt)
        if not self._current(a):
            return False
        self._found(a, found, live)
        if found.dark:
            self._answered(a, ask.Answer("failed", error="dark"))
            return False
        return True

    @staticmethod
    def _local_result(t: Any) -> local.Local:
        try:
            return t.result()
        except Exception as e:  # noqa: BLE001 — nothing read on this Mac: the brain's answer still comes
            log.info(f"[lens] the on-device pass failed ({type(e).__name__})")
            return local.Local(actions=[local.Action("copy_image", "Copy image", "")])

    async def _picture(self, a: _Area) -> Any:
        if a.part is None:
            from neo2.core.tools.computer import screen

            a.part = (await asyncio.to_thread(screen.render, a.crop, "region")).part
        return a.part

    def _answered(self, a: _Area, ans: ask.Answer, extra: list[dict[str, Any]] | None = None) -> None:
        a.answer = ans
        data = {"sel": a.n, **ans.shown(), "actions": extra or []}
        a.shown_answer = data
        self._send("lens_answer", data)

    async def _answer(self, a: _Area, text: str | None = None) -> None:
        self._send("lens_answer", {"sel": a.n, "state": "thinking"})
        part = await self._picture(a)
        if not self._current(a):
            return
        if text is None:
            text = a.found.text if a.found else ""
        ans = await ask.answer(part, text=text, question=a.question, lang=ask.user_lang())
        if not self._current(a):
            return
        extra: list[dict[str, Any]] = []
        if ans.state == "done" and ans.query:
            provider = search.provider()
            if provider is not None and provider["text"]:
                label = f"Search “{ans.query}”" + ("" if provider["image"] else f" on {provider['label']}")
                act = local.Action("text_search", label[:90], ans.query, detail=provider["label"])
                extra.append(act.shown(self._act_id(a, act)))
        elif ans.state == "failed" and ans.error in ("no_brain", "local_only"):
            act = local.Action("brains", "Brains & Keys…", "")
            extra.append(act.shown(self._act_id(a, act)))
        self._answered(a, ans, extra)
        if ans.state == "done" and ans.query and (n := search.wanted(ans.kind, ans.shop)):
            rows = await asyncio.to_thread(search.links, ans.query, n)
            if not self._current(a) or not rows:
                return
            shown = []
            for r in rows:
                act = local.Action("open", r["title"], r["href"], detail=r["host"])
                shown.append({"id": self._act_id(a, act), "title": r["title"], "host": r["host"]})
            a.shown_links = {"sel": a.n, "rows": shown}
            self._send("lens_links", a.shown_links)

    async def _follow(self, a: _Area, qid: int, q: str) -> str:
        self._send("lens_reply", {"sel": a.n, "qid": qid, "state": "thinking"})
        part = await self._picture(a)
        if not self._current(a):
            return ""
        first = a.answer.brain if a.answer is not None else ""
        text, who = await ask.follow_up(part, text=a.found.text if a.found else "", history=a.thread, q=q, first=first)
        if not self._current(a):
            return ""
        if text:
            a.thread.append((q, text))
            del a.thread[: -ask.HISTORY]
            self._send("lens_reply", {"sel": a.n, "qid": qid, "state": "done", "text": text,
                                      "brain": {"label": who.label, "local": who.local}})  # fmt: skip
        else:
            self._send("lens_reply", {"sel": a.n, "qid": qid, "state": "failed", "error": who.error or "failed"})
        return text

    async def _translate(self, a: _Area, to: str) -> None:
        name = local.language_name(to)
        text = a.found.text if a.found else ""
        if text:
            out, err = await ask.translate_text(text, name)
        else:
            out, err = await ask.translate_picture(await self._picture(a), name)
        if not self._current(a):
            return
        log.info(f"[lens] translation: {'done' if out else err}")
        self._send("lens_translation", {"sel": a.n, "to": to, **({"text": out} if out else {"error": err or "failed"})})

    async def _act(self, a: _Area, aid: str) -> None:
        act = a.acts[aid]
        note, ok = "failed", False
        try:
            if act.kind in ("open", "mail", "call"):
                from neo2.core.tools.computer import apps

                out = await apps.open_url(act.value)
                ok = not str(out).startswith("Error")
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


def _quiet(t: asyncio.Task) -> None:
    if not t.cancelled() and (e := t.exception()) is not None:
        log.info(f"[lens] a task failed ({type(e).__name__})")


lens_live = LensLive()


def _on_event(ev) -> None:
    if ev.type == "erased":
        lens_live.erased()


app_bus().subscribe(_on_event)


def reset() -> None:
    """A fresh session (tests)."""
    channel, hooks = lens_live.channel, lens_live.hooks
    lens_live._drop("reset")
    lens_live.__dict__.update(LensLive().__dict__)
    lens_live.channel, lens_live.hooks = channel, hooks


__all__ = ["KIND_LENS", "Hooks", "LensLive", "copy_image", "copy_text", "lens_live", "reset"]
