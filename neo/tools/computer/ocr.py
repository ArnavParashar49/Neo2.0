"""Reading the screen's words: Apple Vision's on-device text recognition over NEO's own captures.

Accessibility names most controls, but not a canvas, a game, a web view that hides its tree, or a
dialog an app draws itself. There NEO reads the pixels it captured itself, three ways:

* **text targets** — :func:`find_text` (click_text's fallback): the words asked for, found in the
  front window at full Retina detail, an open sheet's words first. The point then goes through the
  same probe → gate → click path as every other click (:func:`click_text_fallback`); the gate judges
  every version of the words the reads saw there.
* **text marks** — t1…tn for words Accessibility doesn't cover, listed with a picture
  (:func:`marks_block`): ids and words only for a brain that can't see (NEO read the pixels, so it may
  act on them by id), plus where they are for one that can. A mark whose words commit something is
  never clicked by id: click_text asks the user first. A mark is clicked only while its words are
  still where they were read, on the same app's window with nothing over them (checked at click
  time), and for about one look (MARK_TTL_S).

Whose words: the window server's windows over the area, front to back — a line belongs to the topmost
window at its middle. A protected app's or NEO's own words are never read, and another app's window
over the target (a banner, a panel) is never read as the target's words.
* **checking effects** — the words around a click before and after (:func:`sample_change`,
  :func:`words_changed`): “Start a New Game?” is gone, “Start” → “Stop”.

Speed and accuracy by need (synthetic Retina frames, macOS 26, recognition revision 3; measured by
tests/test_eyes_ocr_live.py):

* fast mode: about 6 ms for a 500×170 pt dialog, 10 ms for a window, 23 ms for a whole Retina display,
  six Latin-script languages. It needs about two pixels per point — on a picture shrunk to one pixel
  per point it read 2 of 9 labels — so a picture with fewer is scaled up 2× first (8 of 9);
* accurate mode: about 32 ms for the dialog, 60–80 ms for a window, 30 languages. Whole, a Retina
  display drops small stacked labels (2 of 6), so a big accurate read is cut into overlapping pieces
  (6 of 6, about 170 ms);
* either mode now and then drops or garbles a word (2–3 in 30 framings of the same window), so a find
  is not one read's word: a fast find is taken when a second, differently framed fast read agrees on
  the place and reads as many lines (about 2 × 6–14 ms); otherwise an accurate read decides over the
  union of all of them — where the reads disagree about the words at a place, a place most of them
  read otherwise is no find. On 150 finds over 30 framings: one fast read, 140 right, 5 wrong, 5
  missed; find_in, 150 right (median 22 ms). Its limit: a second copy so small that fast mode drops
  it in both framings (rather than garbling it) isn't seen by a fast find, which then takes the copy
  both read;
* change checks read fast; marks read accurately plus fast (the union); scripts fast can't read,
  accurately in their own languages;
* the picture reaches Vision as raw grey pixels (about 1.5 ms for a Retina display), never through a
  PNG (about 31 ms).

Languages: whatever this Mac's Vision reads, asked once. Hindi is not among them on macOS 26: words in
Devanagari are never searched for (a Hindi target relies on Accessibility's labels), and the answer
says so rather than "not found".

Every word read here came from the screen: it is fenced wherever it goes to a model and taints the
request, like Accessibility's labels. Library mode starts nothing in the background: Vision loads on
first use (0.2–0.4 s once), or ahead of time through :func:`warm` from NEO.app.

With pyobjc-framework-Vision installed its metadata is used; without it the system framework is loaded
through objc.loadBundle (the calls used here need no extra metadata). With neither — or off macOS —
:func:`available` is False and every caller keeps its own answer.
"""

from __future__ import annotations

import asyncio
import difflib
import itertools
import logging
import math
import re
import sys
import threading
import time
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from neo import effects

log = logging.getLogger("neo.ocr")

Box = tuple[float, float, float, float]  # x, y, width, height — screen points, origin top left

_FRAMEWORK = "/System/Library/Frameworks/Vision.framework"
FAST, ACCURATE = "fast", "accurate"
_FAST_PX_PER_PT = 1.8  # fewer pixels per point than this: a fast read garbles — scale up first
_UPSCALE_TO = 2.0  # …to this many pixels per point
_MAX_EDGE = 4096  # never scale a picture past this long edge
# An accurate read of a picture longer than this (pixels) drops or merges small stacked labels (measured
# on a Retina display: whole 1512×982 pt it found 2 of 6 buttons; cut at ≤2424 px, 6 of 6): cut it.
_ACCURATE_MAX_PX = 2400
_FAST_MAX_PX = 4096
_OVERLAP_PT = 160  # the pieces overlap this much (a label shorter than this is whole in one)
_PAD_PX = 24  # the second framing: this much border
_MIN_CONF = 0.4  # Vision's confidences are coarse (fast 0.5, accurate 1.0; noise 0.3): below this, noise
_RUN_WAIT_S = 5.0  # one recognition at a time (Vision queues concurrent ones anyway)
_LABEL_MAX = 60  # words quoted in a result, at most this long


class Failed(Exception):
    """The recognition couldn't run (Vision missing, busy, or it reported an error)."""


# ---- the engine: Vision, or a test's fake ---------------------------------------------------------------


@dataclass(frozen=True)
class Raw:
    """One line as an engine reports it, in Vision's own convention: ``box`` normalised to 0–1 of the
    picture with the origin at the BOTTOM left; ``parts``: (start, end, box) for the character ranges
    asked for (``ranges``), same convention."""

    text: str
    conf: float
    box: Box
    parts: tuple[tuple[int, int, Box], ...] = ()


class Engine(Protocol):
    def languages(self, fast: bool) -> list[str]: ...

    def recognize(
        self,
        image: Image.Image,
        *,
        fast: bool,
        languages: Sequence[str],
        auto: bool,
        ranges: Callable[[str], list[tuple[int, int]]] | None,
    ) -> list[Raw]: ...


def _cgimage(img: Image.Image) -> tuple[Any, bytes]:
    """A grey CGImage over the picture's own bytes (1.5 ms for 3024×1964; a PNG round trip is 35).
    The bytes are returned too: they must outlive the request."""
    import Quartz as Q

    grey = img if img.mode == "L" else img.convert("L")
    data = grey.tobytes()
    provider = Q.CGDataProviderCreateWithData(None, data, len(data), None)
    cg = Q.CGImageCreate(
        grey.width, grey.height, 8, 8, grey.width, Q.CGColorSpaceCreateDeviceGray(), Q.kCGImageAlphaNone,
        provider, None, False, Q.kCGRenderingIntentDefault,
    )  # fmt: skip
    if cg is None:
        raise Failed("couldn't make a picture Vision can read")
    return cg, data


def _utf16(s: str, i: int) -> int:
    """Python's character index ``i`` of ``s`` as an NSString (UTF-16) index."""
    return len(s[:i].encode("utf-16-le")) // 2


class _Vision:
    """Apple's VNRecognizeTextRequest. Loaded on first use; one recognition at a time."""

    def __init__(self) -> None:
        self._ns: dict[str, Any] | None = None
        self._langs: dict[bool, list[str]] = {}
        self._load_lock = threading.Lock()
        self._run_lock = threading.Lock()

    def _classes(self) -> dict[str, Any]:
        effects.guard("model", "load Apple Vision's text recognition")
        with self._load_lock:
            if self._ns is None:
                try:
                    import Vision  # pyobjc-framework-Vision: its metadata

                    self._ns = {k: getattr(Vision, k) for k in ("VNRecognizeTextRequest", "VNImageRequestHandler")}
                except ImportError:
                    import objc

                    ns: dict[str, Any] = {}
                    objc.loadBundle("Vision", ns, bundle_path=_FRAMEWORK)
                    self._ns = ns
            return self._ns

    def languages(self, fast: bool) -> list[str]:
        if fast not in self._langs:
            from neo.decide._objc import pool

            with pool():
                req = self._classes()["VNRecognizeTextRequest"].alloc().init()
                req.setRecognitionLevel_(1 if fast else 0)
                got = req.supportedRecognitionLanguagesAndReturnError_(None)
                got = got[0] if isinstance(got, tuple) else got
                self._langs[fast] = [str(x) for x in (got or [])]
        return list(self._langs[fast])

    def recognize(self, image, *, fast, languages, auto, ranges) -> list[Raw]:
        effects.guard("model", "run Apple Vision's on-device text recognition")
        from neo.decide._objc import pool

        ns = self._classes()
        if not self._run_lock.acquire(timeout=_RUN_WAIT_S):
            raise Failed("text recognition is busy")
        try:
            with pool():
                cg, data = _cgimage(image)
                req = ns["VNRecognizeTextRequest"].alloc().init()
                req.setRecognitionLevel_(1 if fast else 0)
                req.setUsesLanguageCorrection_(False)  # UI words, not prose: faster, and no "fixing" of labels
                if languages:
                    req.setRecognitionLanguages_(list(languages))
                if auto and not fast and hasattr(req, "setAutomaticallyDetectsLanguage_"):
                    req.setAutomaticallyDetectsLanguage_(True)
                handler = ns["VNImageRequestHandler"].alloc().initWithCGImage_options_(cg, {})
                ok = handler.performRequests_error_([req], None)
                if isinstance(ok, tuple):
                    ok, err = ok[0], ok[1] if len(ok) > 1 else None
                else:
                    err = None
                if not ok:
                    raise Failed(f"Vision couldn't read the picture ({err})")
                out = [self._line(o, ranges) for o in (req.results() or [])]
                del data  # (the bytes the CGImage reads from lived until here)
                return [r for r in out if r is not None]
        finally:
            self._run_lock.release()

    @staticmethod
    def _line(o, ranges) -> Raw | None:
        cands = o.topCandidates_(1)
        if not cands:
            return None
        c = cands[0]
        s = str(c.string())
        bb = o.boundingBox()
        parts = []
        for start, end in ranges(s) if ranges else []:
            try:
                got = c.boundingBoxForRange_error_((_utf16(s, start), _utf16(s, end) - _utf16(s, start)), None)
                ob = got[0] if isinstance(got, tuple) else got
                if ob is not None:
                    b = ob.boundingBox()
                    parts.append((start, end, (float(b.origin.x), float(b.origin.y), float(b.size.width), float(b.size.height))))
            except Exception:  # noqa: BLE001 - no box for the words: the line's share of them stands in
                continue
        box = (float(bb.origin.x), float(bb.origin.y), float(bb.size.width), float(bb.size.height))
        return Raw(s, float(c.confidence()), box, tuple(parts))


_engine: Engine | None = None  # a test's or a benchmark's engine (set_engine)
_vision: _Vision | None = None
_have_vision: bool | None = None


def set_engine(e: Engine | None) -> None:
    """Use ``e`` instead of Apple Vision (tests, benchmarks); None: Vision again."""
    global _engine
    _engine = e


def _vision_here() -> bool:
    global _have_vision
    if _have_vision is None:
        try:
            import objc  # noqa: F401

            _have_vision = sys.platform == "darwin" and Path(_FRAMEWORK).exists()
        except ImportError:
            _have_vision = False
    return _have_vision


def _may_load() -> bool:
    """May Vision be loaded in this process (real effects on, or "model" allowed)?"""
    try:
        effects.guard("model", "load Apple Vision's text recognition")
    except effects.EffectRefused:
        return False
    return True


def available() -> bool:
    """Can NEO read text off a picture here (without loading anything to find out)? Never with real
    effects off (tests, scratch scripts) unless an engine was set or "model" is allowed."""
    return _engine is not None or (_vision_here() and _may_load())


def engine() -> Engine | None:
    global _vision
    if _engine is not None:
        return _engine
    if not _vision_here() or not _may_load():
        return None
    if _vision is None:
        _vision = _Vision()
    return _vision


_warmed = threading.Event()


def warm() -> bool:
    """Load Vision in the background so the first read doesn't pay for it (0.2–0.4 s): NEO.app only
    — never in library mode (no background work there), never with real effects off. True when a
    warm-up started."""
    from neo import runtime

    if _warmed.is_set() or runtime.library_bound() or not effects.on() or _engine is not None or not _vision_here():
        return False
    _warmed.set()

    def run() -> None:
        try:
            e = engine()
            blank = Image.new("L", (64, 32), 255)
            for fast in (True, False):  # (each mode loads its own model: the accurate one 0.4–0.6 s)
                if e is not None:
                    e.recognize(blank, fast=fast, languages=["en-US"], auto=False, ranges=None)
        except BaseException as ex:  # noqa: BLE001 - a warm-up that failed costs the first read its load, nothing else
            log.info("text recognition warm-up failed: %s", ex)

    threading.Thread(target=run, name="neo-ocr-warm", daemon=True).start()
    return True


# ---- scripts and languages ------------------------------------------------------------------------------

# Which writing systems each language Vision may list is written in (by its code's first part).
_LANG_SCRIPTS: dict[str, frozenset[str]] = {
    **{k: frozenset({"LATIN"}) for k in (
        "en", "fr", "it", "de", "es", "pt", "vi", "tr", "id", "cs", "da", "nl", "no", "nn", "nb", "ms",
        "pl", "ro", "sv", "fi", "hu", "hr", "sk", "sl", "ca", "et", "lv", "lt", "is", "ga", "tl", "sq",
    )},  # fmt: skip
    **{k: frozenset({"CYRILLIC"}) for k in ("ru", "uk", "bg", "sr", "be", "kk", "mk")},
    "zh": frozenset({"CJK"}),
    "yue": frozenset({"CJK"}),
    "ja": frozenset({"CJK", "HIRAGANA", "KATAKANA"}),
    "ko": frozenset({"HANGUL", "CJK"}),
    "th": frozenset({"THAI"}),
    "ar": frozenset({"ARABIC"}),
    "ars": frozenset({"ARABIC"}),
    "fa": frozenset({"ARABIC"}),
    "ur": frozenset({"ARABIC"}),
    "he": frozenset({"HEBREW"}),
    "el": frozenset({"GREEK"}),
    "hi": frozenset({"DEVANAGARI"}),
    "mr": frozenset({"DEVANAGARI"}),
    "ne": frozenset({"DEVANAGARI"}),
    "bn": frozenset({"BENGALI"}),
    "ta": frozenset({"TAMIL"}),
    "te": frozenset({"TELUGU"}),
    "gu": frozenset({"GUJARATI"}),
    "pa": frozenset({"GURMUKHI"}),
    "kn": frozenset({"KANNADA"}),
    "ml": frozenset({"MALAYALAM"}),
}
# Words that start a character's Unicode name without naming its writing system.
_NOT_SCRIPTS = frozenset({
    "MODIFIER", "COMBINING", "FULLWIDTH", "HALFWIDTH", "MATHEMATICAL", "SUPERSCRIPT", "SUBSCRIPT", "CIRCLED",
    "PARENTHESIZED", "SQUARED", "NEGATIVE", "REGIONAL", "DOUBLE", "SMALL", "VARIATION", "ZERO", "LEFT", "RIGHT",
})  # fmt: skip


def scripts(text: str) -> set[str]:
    """The writing systems of the letters in ``text`` ("LATIN", "DEVANAGARI", "CJK"…, by their Unicode
    names; digits, spaces and punctuation belong to none)."""
    out: set[str] = set()
    for ch in unicodedata.normalize("NFKC", text):
        if not ch.isalpha() and unicodedata.category(ch) not in ("Mn", "Mc"):
            continue
        name = unicodedata.name(ch, "")
        head = name.split()[0].split("-")[0] if name else ""
        if head in ("CJK", "IDEOGRAPHIC"):
            head = "CJK"
        if head and head not in _NOT_SCRIPTS:
            out.add(head)
    return out


def _lang_scripts(code: str) -> frozenset[str]:
    return _LANG_SCRIPTS.get(code.split("-")[0].split("_")[0].lower(), frozenset())


def _supported(e: Engine, fast: bool) -> list[str]:
    try:
        return e.languages(fast)
    except BaseException as ex:  # noqa: BLE001 - (effects off, Vision missing): nothing is known
        log.info("text recognition languages unknown: %s", ex)
        return []


def unreadable(text: str) -> str:
    """The name of the first writing system in ``text`` this Mac's text recognition can't read
    ("Devanagari"); "" when it reads them all — or when there is no recognition to ask."""
    e = engine()
    langs = _supported(e, False) if e is not None else []
    if not langs:
        return ""
    known = set().union(*(_lang_scripts(c) for c in langs))
    for s in sorted(scripts(text)):
        if s not in known:
            return s.title()
    return ""


_preferred: list[str] | None = None


def _preferred_languages() -> list[str]:
    """The Mac's preferred languages (the apps' UI follows them), read once."""
    global _preferred
    if _preferred is None:
        try:
            from Foundation import NSLocale

            _preferred = [str(x) for x in (NSLocale.preferredLanguages() or [])]
        except Exception:  # noqa: BLE001 - (no Foundation): English only
            _preferred = []
    return _preferred


def _pick(supported: list[str], wanted: Iterable[str]) -> list[str]:
    """The supported codes matching ``wanted`` codes by language (en-IN → en-US), in order."""
    out: list[str] = []
    for w in wanted:
        base = w.split("-")[0].split("_")[0].lower()
        for s in supported:
            if s.split("-")[0].lower() == base and s not in out:
                out.append(s)
                break
    return out


@dataclass(frozen=True)
class Plan:
    """How to read: fast or accurate, in which languages, with Vision guessing the language."""

    fast: bool
    languages: tuple[str, ...]
    auto: bool = False


def plan_for(text: str = "", *, want_fast: bool = True) -> Plan | None:
    """How to read the screen to find ``text`` ("" = to read whatever is there), by its writing system:
    fast when fast reads it (and ``want_fast``), else accurate in the languages of its script. None
    when this Mac can't read it at all."""
    e = engine()
    if e is None:
        return None
    acc, fst = _supported(e, False), _supported(e, True)
    if not acc and not fst:
        return None
    if text:
        need = scripts(text)
        if not need:  # digits and signs only: any language reads them
            need = {"LATIN"}
        if unreadable(text):
            return None
        pool = [c for c in fst if need <= _lang_scripts(c)] if want_fast else []
        if pool:
            chosen = _pick(pool, [*_preferred_languages(), "en"]) or pool[:1]
            return Plan(True, tuple(chosen[:3]))
        # Accurate, in languages written in the words' scripts (the Mac's own first): one per script at
        # least, so words mixing two scripts are read in both.
        ranked = _pick(acc, [*_preferred_languages(), "en"]) + [c for c in acc]
        chosen: list[str] = []
        for s in sorted(need):
            chosen += [c for c in ranked if s in _lang_scripts(c) and c not in chosen][:1]
        chosen += [c for c in ranked if _lang_scripts(c) & need and c not in chosen]
        covered = set().union(*(_lang_scripts(c) for c in chosen)) if chosen else set()
        return Plan(False, tuple(chosen[: max(4, len(need))])) if need <= covered else None
    # Whatever is there: the languages this Mac's apps show (its preferred ones), English too.
    mine = _pick(acc, [*_preferred_languages(), "en"])
    latin_only = all(_lang_scripts(c) == frozenset({"LATIN"}) for c in mine)
    if want_fast and latin_only:
        fast_ones = _pick(fst, mine) or fst[:1]
        return Plan(True, tuple(fast_ones[:3]))
    return Plan(False, tuple(mine[:4] or acc[:1]), auto=not latin_only)


# ---- reading a picture ----------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Pixels:
    """A picture and where its pixels are: pixel (0, 0) at screen point (x0, y0), ``ppp`` screen points
    per pixel (0.5 on Retina). The capture pipeline's Frame has the same fields and is taken as is."""

    image: Image.Image
    x0: float
    y0: float
    ppp: float
    t: float = field(default_factory=time.monotonic)


def pixels(frame: Any) -> Pixels:
    """Any frame-like object (image, x0, y0, ppp — the capture pipeline's Frame) as Pixels."""
    if isinstance(frame, Pixels):
        return frame
    return Pixels(frame.image, float(frame.x0), float(frame.y0), float(frame.ppp), float(getattr(frame, "t", time.monotonic())))


@dataclass(frozen=True)
class Line:
    """One line of words read off the screen (the screen's words: fence them). ``box``: where, in
    screen points from the top left; ``parts``: (start, end, box) of words searched for in it."""

    text: str
    conf: float
    box: Box
    parts: tuple[tuple[int, int, Box], ...] = ()

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.box
        return x + w / 2, y + h / 2

    def part_box(self, start: int, end: int) -> Box:
        """Where characters start…end of the line are: Vision's own box for them when it gave one,
        else their share of the line's width."""
        for s, e, b in self.parts:
            if s == start and e == end:
                return b
        x, y, w, h = self.box
        n = max(1, len(self.text))
        return x + w * start / n, y, w * (end - start) / n, h


@dataclass(frozen=True)
class Reading:
    """What one read found: its lines, the screen area read (points), how, and how long it took."""

    lines: tuple[Line, ...]
    area: Box
    level: str
    ms: float
    t: float  # when the picture was taken (time.monotonic)
    px_per_pt: float  # the picture's detail as read (after any scaling up)

    def text(self) -> str:
        return "\n".join(ln.text for ln in self.lines)


def _to_screen(b: Box, x0: float, y0: float, w_pt: float, h_pt: float) -> Box:
    """A box from Vision's convention (0–1, origin bottom left) to screen points (origin top left)."""
    bx, by, bw, bh = b
    return x0 + bx * w_pt, y0 + (1.0 - by - bh) * h_pt, bw * w_pt, bh * h_pt


def _crop(px: Pixels, area: Box | None) -> tuple[Image.Image, float, float]:
    """(picture, x0, y0) of ``area`` (screen points) at the frame's own resolution, clipped to it."""
    if area is None:
        return px.image, px.x0, px.y0
    k = 1.0 / px.ppp
    left = max(0, int((area[0] - px.x0) * k))
    top = max(0, int((area[1] - px.y0) * k))
    right = min(px.image.width, int(round((area[0] + area[2] - px.x0) * k)))
    bottom = min(px.image.height, int(round((area[1] + area[3] - px.y0) * k)))
    if right - left < 2 or bottom - top < 2:
        raise ValueError("that area isn't in the picture")
    if (left, top, right, bottom) == (0, 0, px.image.width, px.image.height):
        return px.image, px.x0, px.y0
    return px.image.crop((left, top, right, bottom)), px.x0 + left * px.ppp, px.y0 + top * px.ppp


def _cuts(total: int, limit: int, overlap: int) -> list[tuple[int, int, int, int]]:
    """How to cut ``total`` pixels into pieces of at most ``limit`` that overlap by ``overlap``: (start,
    end, core start, core end) each — a piece keeps only what is centred in its core (the cores tile
    the whole, so nothing read twice is kept twice, and a line up to ``overlap`` long that crosses a
    cut is whole in the piece that keeps it)."""
    if total <= limit:
        return [(0, total, 0, total)]
    overlap = min(overlap, limit // 2)
    n = math.ceil((total - overlap) / (limit - overlap))
    size = min(limit, math.ceil((total + (n - 1) * overlap) / n))
    spans = [(min(i * (size - overlap), total - size), min(i * (size - overlap), total - size) + size) for i in range(n)]
    out = []
    for i, (s, e) in enumerate(spans):
        cs = 0 if i == 0 else (spans[i - 1][1] + s) // 2
        ce = total if i == n - 1 else (e + spans[i + 1][0]) // 2
        out.append((s, e, cs, ce))
    return out


def _pad(img: Image.Image, pad: int) -> Image.Image:
    """``img`` framed by ``pad`` pixels of its own corners' shade (another framing of the same words)."""
    from PIL import ImageOps

    w, h = img.size
    shade = round(sum(int(img.getpixel(p)) for p in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1))) / 4)
    return ImageOps.expand(img, border=pad, fill=shade)


def read(
    frame: Any,
    area: Box | None = None,
    *,
    plan: Plan | None = None,
    ranges: Callable[[str], list[tuple[int, int]]] | None = None,
    pad: int = 0,
) -> Reading:
    """Read the words in ``area`` of a frame (all of it when None) at the frame's full resolution:
    scaled up 2× first when a fast read would get too few pixels per point; an accurate read of more
    than _ACCURATE_MAX_PX on a side cut into overlapping pieces (whole, it drops or merges small
    stacked labels). ``pad``: frame the picture with this many pixels of its own shade first (a
    second framing: a read that dropped words may not drop them again). Raises Failed when there is no
    engine or it fails."""
    e = engine()
    if e is None:
        raise Failed("no text recognition on this Mac")
    plan = plan or plan_for("")
    if plan is None:
        raise Failed("text recognition knows no language here")
    px = pixels(frame)
    t0 = time.perf_counter()
    img, x0, y0 = _crop(px, area)
    w_pt, h_pt = img.width * px.ppp, img.height * px.ppp
    per_pt = 1.0 / px.ppp
    grey = img.convert("L") if img.mode != "L" else img
    if plan.fast and per_pt < _FAST_PX_PER_PT:
        k = min(_UPSCALE_TO / per_pt, _MAX_EDGE / max(grey.width, grey.height))
        if k > 1.05:
            grey = grey.resize((max(1, round(grey.width * k)), max(1, round(grey.height * k))), Image.BICUBIC)
            per_pt *= k
    kx, ky = w_pt / grey.width, h_pt / grey.height  # screen points per pixel of what is read
    limit = _FAST_MAX_PX if plan.fast else _ACCURATE_MAX_PX
    overlap = round(_OVERLAP_PT * per_pt)
    lines = []
    rows, cols = _cuts(grey.height, limit, overlap), _cuts(grey.width, limit, overlap)
    for ri, (ys, ye, ycs, yce) in enumerate(rows):
        for ci, (xs, xe, xcs, xce) in enumerate(cols):
            piece = grey if (xs, ys, xe, ye) == (0, 0, grey.width, grey.height) else grey.crop((xs, ys, xe, ye))
            if pad:
                piece = _pad(piece, pad)
            pw, ph = piece.size

            def in_pixels(b: Box, xs=xs, ys=ys, pw=pw, ph=ph) -> Box:
                bx, by, bw, bh = b  # (0–1 of the piece, from its bottom left)
                return xs - pad + bx * pw, ys - pad + (1.0 - by - bh) * ph, bw * pw, bh * ph

            def to_screen(b: Box) -> Box:
                x, y, w, h = in_pixels(b)
                return x0 + x * kx, y0 + y * ky, w * kx, h * ky

            try:
                raws = e.recognize(piece, fast=plan.fast, languages=list(plan.languages), auto=plan.auto, ranges=ranges)
            except (Failed, effects.EffectRefused):
                raise
            except Exception as ex:  # noqa: BLE001 - an engine error is a failed read, never a crash of the tool
                raise Failed(f"text recognition failed ({type(ex).__name__})") from ex
            # What is centred in this piece's core (the first and last pieces reach past the edges).
            left, right = (-math.inf if ci == 0 else xcs), (math.inf if ci == len(cols) - 1 else xce)
            top, bottom = (-math.inf if ri == 0 else ycs), (math.inf if ri == len(rows) - 1 else yce)
            for r in raws:
                text = " ".join(str(r.text).split())
                if not text:
                    continue
                bx, by, bw, bh = in_pixels(r.box)
                if not (left <= bx + bw / 2 < right and top <= by + bh / 2 < bottom):
                    continue
                parts = tuple((s_, e_, to_screen(b_)) for s_, e_, b_ in r.parts)
                lines.append(Line(text, float(r.conf), to_screen(r.box), parts))
    lines.sort(key=lambda ln: (round(ln.box[1] / 6), ln.box[0]))
    ms = (time.perf_counter() - t0) * 1000
    return Reading(tuple(lines), (x0, y0, w_pt, h_pt), FAST if plan.fast else ACCURATE, ms, px.t, per_pt)


def without(reading: Reading, rects: Iterable[Box]) -> Reading:
    """The reading without lines whose middle is in any of ``rects``."""
    rects = list(rects)
    if not rects:
        return reading
    keep = tuple(ln for ln in reading.lines if not any(_inside(ln.center, r) for r in rects))
    return replace(reading, lines=keep)


# ---- whose words: the windows over the screen ------------------------------------------------------------


@dataclass(frozen=True)
class Win:
    """One on-screen window as the window server lists it (``box``: screen points)."""

    box: Box
    pid: int
    app: str
    off_limits: bool = False  # NEO's own window or a protected app's: its words are never read


@dataclass(frozen=True)
class Windows:
    """The on-screen windows over an area, front to back (the window server's order). The words at a
    point belong to the topmost window there: a window behind another shows nothing at that point,
    and one in front of the target (a banner, another app's panel) shows its own app's words."""

    wins: tuple[Win, ...] = ()

    def at(self, p: tuple[float, float]) -> Win | None:
        for w in self.wins:
            x, y, ww, hh = w.box
            if x <= p[0] < x + ww and y <= p[1] < y + hh:
                return w
        return None


def windows_over(area: Box) -> Windows | None:
    """The on-screen windows over ``area``, front to back, each marked when it is NEO's own or a
    protected app's. None when the window server can't say (fail closed: nothing is read)."""
    from neo.decide._objc import pool

    try:
        import Quartz as Q

        from neo.tools import computer as _c
        from neo.tools.computer import ax

        with pool():
            return _windows(Q, _c, ax, area)
    except Exception as ex:  # noqa: BLE001 - can't tell whose windows are there: nothing is read
        log.info("the window server couldn't say what's on screen: %s", ex)
        return None


def _windows(Q: Any, _c: Any, ax: Any, area: Box) -> Windows | None:
    listed = Q.CGWindowListCopyWindowInfo(Q.kCGWindowListOptionOnScreenOnly | Q.kCGWindowListExcludeDesktopElements, Q.kCGNullWindowID)
    if listed is None:
        return None
    out: list[Win] = []
    judged: dict[tuple[int, str], bool] = {}
    for w in listed:  # front to back
        b = w.get("kCGWindowBounds") or {}
        r = tuple(float(b.get(k, 0) or 0) for k in ("X", "Y", "Width", "Height"))
        if r[2] <= 0 or r[3] <= 0 or float(w.get("kCGWindowAlpha", 1) or 0) <= 0 or not _overlap(r, area):  # type: ignore[arg-type]
            continue  # (an invisible window shows nothing: ax.window_owner_at skips it too)
        pid, name = int(w.get("kCGWindowOwnerPID", 0) or 0), str(w.get("kCGWindowOwnerName", "") or "")
        if (pid, name) not in judged:
            judged[(pid, name)] = bool(pid and ax.is_neo(pid, name)) or bool(_c._protected(name, pid))
        out.append(Win(r, pid, name, judged[(pid, name)]))  # type: ignore[arg-type]
    return Windows(tuple(out))


def _readable(w: Win | None, pid: int) -> bool:
    return w is not None and not w.off_limits and (not pid or w.pid == pid)


def owned(reading: Reading, wins: Windows, pid: int = 0) -> Reading:
    """The reading with only the lines whose middle is on a window NEO may read: the topmost window
    there is ``pid``'s — with 0, any app's but NEO's own or a protected one's. Words on no window
    (the desktop) and words a window in front covers never count."""
    keep = tuple(ln for ln in reading.lines if _readable(wins.at(ln.center), pid))
    return replace(reading, lines=keep)


def merge(*readings: Reading, keep_all_words: bool = False) -> Reading:
    """One reading of a place from several reads of it (pass the most trusted read first).

    Any read may drop a word now and then, or garble one — measured on 30 framings of a window: 2–3
    drops in either mode, and the accurate mode reading a bordered "Start" as "StartR". The union
    rarely misses what is there. By default a later read adds only lines where no kept line sits (a
    listing never shows the same place twice); ``keep_all_words`` adds every line whose words aren't
    already kept there (a find: the garbled "StartR" can't hide the "Start" another read got right)."""
    kept: list[Line] = []
    for r in readings:
        for ln in r.lines:
            k = key(ln.text)
            if not any(
                (_inside(ln.center, o.box) or _inside(o.center, ln.box)) and (not keep_all_words or key(o.text) == k)
                for o in kept
            ):
                kept.append(ln)
    kept.sort(key=lambda ln: (round(ln.box[1] / 6), ln.box[0]))
    return replace(readings[0], lines=tuple(kept), ms=sum(r.ms for r in readings))


# ---- matching words -------------------------------------------------------------------------------------

_QUOTES = str.maketrans({"“": '"', "”": '"', "„": '"', "‘": "'", "’": "'", "‚": "'", "\u00a0": " "})
_EDGE_PUNCT = " .:?!,;\"'()[]{}<>*•·-–—…"


def fold(s: str) -> tuple[str, list[int]]:
    """``s`` folded for comparing (compatibility forms, lower case, straight quotes, "…" as "...",
    single spaces), and for each folded character the index of the character it came from."""
    out: list[str] = []
    idx: list[int] = []
    space = True
    for i, ch in enumerate(str(s)):
        for p in unicodedata.normalize("NFKC", ch.translate(_QUOTES)).lower():
            if p.isspace():
                if space:
                    continue
                p, space = " ", True
            else:
                space = False
            out.append(p)
            idx.append(i)
    while out and out[-1] == " ":
        out.pop()
        idx.pop()
    return "".join(out), idx


def key(s: str) -> str:
    """``s`` as compared for an exact match: folded, without punctuation at its ends."""
    return fold(s)[0].strip(_EDGE_PUNCT)


def _same_words(a: str, b: str) -> bool:
    ka, kb = key(a), key(b)
    return bool(ka) and (ka == kb or ka.replace(" ", "") == kb.replace(" ", ""))


def _spans(line: str, label: str) -> list[tuple[int, int]]:
    """Where ``label``'s words are in ``line``, as whole words: (start, end) indices of ``line``."""
    want = key(label)
    if not want:
        return []
    folded, idx = fold(line)
    out = []
    for m in re.finditer(rf"(?<!\w){re.escape(want)}(?!\w)", folded):
        if m.end() > m.start():
            out.append((idx[m.start()], idx[m.end() - 1] + 1))
    return out


def _ranges_for(label: str) -> Callable[[str], list[tuple[int, int]]]:
    return lambda text: [] if _same_words(text, label) else _spans(text, label)


@dataclass(frozen=True)
class Found:
    """Words found on the screen: ``text`` the words as read, ``line`` the whole line they are in (both
    the screen's words), ``point`` the middle of the words (screen points), ``exact`` the whole line is
    those words (a button's own label), ``where`` "sheet" when in the open sheet or dialog."""

    label: str
    text: str
    line: str
    point: tuple[float, float]
    box: Box
    exact: bool
    where: str = ""
    level: str = ""
    context: str = ""  # the words read around them (the screen's): the gate's dialog text
    variants: tuple[str, ...] = ()  # other words a read saw at the same place (the gate judges them too)
    line_box: Box | None = None  # where the whole line is (``box`` is the words' part of it)


@dataclass(frozen=True)
class Ambiguous:
    """The words asked for are in several places, none better than the others."""

    label: str
    options: tuple[Found, ...]


@dataclass(frozen=True)
class Missing:
    """Not found; ``why``: NEO's own words (never the screen's)."""

    label: str
    why: str


Result = Found | Ambiguous | Missing


def _inside(p: tuple[float, float], r: Box) -> bool:
    return r[0] <= p[0] <= r[0] + r[2] and r[1] <= p[1] <= r[1] + r[3]


def _overlap(a: Box, b: Box) -> bool:
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def _context(reading: Reading, sheet: Box | None, where: str) -> str:
    """The words around a find, for the gate (what a dialog's "OK" does): the sheet's when the words
    are in it, else all that was read."""
    lines = [ln.text for ln in reading.lines if not where or (sheet is not None and _inside(ln.center, sheet))]
    return " · ".join(lines)[:2000]


def match(reading: Reading, label: str, *, sheet: Box | None = None) -> Result:
    """The place on the screen showing ``label`` (pure: tested without Vision).

    Best first: in the open sheet or dialog (it blocks the rest); a line that is exactly those words
    (a button's label) before those words inside a longer line; a short line before a sentence (a
    title that mentions them). One best place: Found. Several equally good: Ambiguous — never a guess."""
    want = key(label)
    if not want:
        return Missing(label, "there are no words to look for")
    cands: list[tuple[tuple[int, int], Found]] = []
    for ln in reading.lines:
        if ln.conf < _MIN_CONF:
            continue
        where = "sheet" if sheet is not None and _inside(ln.center, sheet) else ""
        if _same_words(ln.text, label):
            f = Found(label, ln.text, ln.text, ln.center, ln.box, True, where, reading.level, line_box=ln.box)
            cands.append(((0 if where or sheet is None else 1, 0), f))
            continue
        extra = max(0, len(key(ln.text).split()) - len(want.split()))
        for start, end in _spans(ln.text, label):
            b = ln.part_box(start, end)
            p = (b[0] + b[2] / 2, b[1] + b[3] / 2)
            w = "sheet" if sheet is not None and _inside(p, sheet) else ""
            f = Found(label, ln.text[start:end], ln.text, p, b, False, w, reading.level, line_box=ln.box)
            cands.append(((0 if w or sheet is None else 1, 1 if extra <= 2 else 2), f))
    if not cands:
        return Missing(label, "no such words on the screen")
    best = min(r for r, _ in cands)
    places: list[Found] = []
    for f in (f for r, f in cands if r == best):
        if not any(_overlap(f.box, p.box) for p in places):
            places.append(replace(f, context=_context(reading, sheet, f.where)))
    if len(places) == 1:
        return places[0]
    places.sort(key=lambda f: (round(f.box[1] / 6), f.box[0]))
    return Ambiguous(label, tuple(places))


def _lookalikes(reading: Reading, label: str, found: Found) -> bool:
    """Is there, away from ``found``, a line with words that look like ``label`` without being them
    ("Stsrt" for "Start": a fast read garbling a second copy)? Then a closer read decides."""
    want = key(label)
    n = max(1, len(want.split()))
    for ln in reading.lines:
        if _overlap(ln.box, found.box):
            continue
        words = key(ln.text).split()
        for i in range(max(1, len(words) - n + 1)):
            chunk = " ".join(words[i : i + n])
            if chunk and chunk != want and difflib.SequenceMatcher(None, chunk, want).ratio() >= 0.75:
                return True
    return False


def _sure(got: Result, reading: Reading, label: str, sheet: Box | None) -> bool:
    """Is a fast find one to act on without a closer look: exactly the words, in one place, in the open
    sheet when there is one, and nothing elsewhere that merely looks like them?"""
    return (
        isinstance(got, Found)
        and got.exact
        and (sheet is None or bool(got.where))
        and not _lookalikes(reading, label, got)
    )


def _at_place(ln: Line, box: Box) -> bool:
    """Is ``ln`` at the place ``box`` is (the one holds the other's middle)?"""
    return _inside((box[0] + box[2] / 2, box[1] + box[3] / 2), ln.box) or _inside(ln.center, box)


def _says(text: str, label: str) -> bool:
    return _same_words(text, label) or bool(_spans(text, label))


def _read_there(box: Box, readings: Sequence[Reading]) -> list[list[Line]]:
    """For each reading, its lines at the place ``box`` is."""
    return [[ln for ln in r.lines if ln.conf >= _MIN_CONF and _at_place(ln, box)] for r in readings]


def _variants(words: str, box: Box, readings: Sequence[Reading]) -> tuple[str, ...]:
    """Every other version of the words any read saw at ``box``'s place (the screen's words)."""
    out: list[str] = []
    for there in _read_there(box, readings):
        for ln in there:
            k = key(ln.text)
            if k and k != key(words) and all(k != key(v) for v in out):
                out.append(ln.text)
    return tuple(out)


def _checked(f: Found, readings: Sequence[Reading]) -> Found | None:
    """``f`` with every other version of the words the reads saw at its place (``variants``: the gate
    judges each) — or None when more reads saw other words there than saw the words asked for: one
    read's misread that happens to be those words ("Details" where the others read "Delete")."""
    agree = disagree = 0
    for there in _read_there(f.box, readings):
        if there:
            if any(_says(ln.text, f.label) for ln in there):
                agree += 1
            else:
                disagree += 1
    if disagree > agree:
        return None
    return replace(f, variants=_variants(f.line, f.box, readings))


def _count(r: Reading) -> int:
    return sum(1 for ln in r.lines if ln.conf >= _MIN_CONF)


def find_in(
    frame: Any,
    label: str,
    *,
    area: Box | None = None,
    sheet: Box | None = None,
    hide: Iterable[Box] = (),
    keep: Callable[[Reading], Reading] | None = None,
) -> Result:
    """Find ``label``'s words in a frame (sync; in a thread).

    Fast first: taken when two fast reads, framed differently, each find exactly the words in one and
    the same place (in the open sheet when there is one) with nothing that merely looks like them
    elsewhere, and read as many lines — about 2 × 6–25 ms. Otherwise, and for any script fast can't
    read, an accurate read decides, over everything read so far (a second copy any read saw makes the
    find ambiguous: a read now and then drops a word, and the copy it dropped may be the one meant) —
    except a place more reads saw other words at than saw these (one read's misread). Every version
    of the words read at the place found comes with it (``variants``), for the gate. Not seen: a
    second copy so small that both fast reads drop it (see the module's notes). ``hide``: screen
    rectangles whose words never count; ``keep``: what of each reading counts (:func:`owned`)."""
    if bad := unreadable(label):
        return Missing(label, f"this Mac's text recognition can't read {bad}")
    fast_plan = plan_for(label, want_fast=True)
    if fast_plan is None:
        raise Failed("this Mac's text recognition can't be asked")
    rng = _ranges_for(label)
    hide = list(hide)
    seen: list[Reading] = []

    def look(plan: Plan, pad: int = 0) -> Reading:
        r = without(read(frame, area, plan=plan, ranges=rng, pad=pad), hide)
        if keep is not None:
            r = keep(r)
        seen.append(r)
        return r

    if fast_plan.fast:
        first = match(look(fast_plan), label, sheet=sheet)
        if _sure(first, seen[-1], label, sheet):
            again = match(look(fast_plan, _PAD_PX), label, sheet=sheet)
            if (
                _sure(again, seen[-1], label, sheet)
                and _overlap(first.box, again.box)  # type: ignore[union-attr]
                and _count(seen[0]) == _count(seen[1])  # (a line one framing dropped: fast is struggling here)
                and (took := _checked(first, seen)) is not None  # type: ignore[arg-type]
            ):
                return took
    closer = look(plan_for(label, want_fast=False) or fast_plan)
    got = match(merge(closer, *seen[:-1], keep_all_words=True), label, sheet=sheet)
    disputed = Missing(label, "the reads of the screen disagree about the words there")
    if isinstance(got, Found):
        return _checked(got, seen) or disputed
    if isinstance(got, Ambiguous):
        options = [f for f in (_checked(o, seen) for o in got.options) if f is not None]
        if not options:
            return disputed
        return options[0] if len(options) == 1 else Ambiguous(label, tuple(options))
    return got


# ---- text marks: t1…tn ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class TextMark:
    """Words NEO read off a capture that a model may act on by id: the middle of ``box`` (screen points),
    on ``app``'s window (``pid``). ``context``: the other words read with it (what a dialog's "OK" does
    is in them); ``also``: other versions of its words a read saw there (each judged like its words);
    ``line_box``: where the whole line was read (``box`` may be the words' part of it)."""

    id: str
    text: str
    box: Box
    where: str = ""
    app: str = ""
    pid: int = 0
    t: float = field(default_factory=time.monotonic)
    context: str = ""
    also: tuple[str, ...] = ()
    line_box: Box | None = None

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.box
        return x + w / 2, y + h / 2


MAX_MARKS = 30
MARK_TTL_S = 30.0  # about one look: a mark older than this is gone ("look again")
_KEEP_MARKS = 300
_RECHECK_PAD_PT = 12.0  # at click time, the words' neighbourhood read again: this much around their line
_marks: dict[str, TextMark] = {}
_mark_ids = itertools.count(1)
_marks_lock = threading.Lock()

# Accessibility roles that hold other things: their title names a page, a pane or a canvas — never the
# words drawn inside them.
_CONTAINERS = frozenset({
    "AXApplication", "AXWindow", "AXSheet", "AXDialog", "AXDrawer", "AXPopover", "AXWebArea", "AXScrollArea",
    "AXGroup", "AXSplitGroup", "AXSplitter", "AXLayoutArea", "AXLayoutItem", "AXUnknown", "AXList", "AXOutline",
    "AXTable", "AXBrowser", "AXGrid", "AXTabGroup", "AXToolbar", "AXRadioGroup",
})  # fmt: skip


def _cover(o: Any) -> tuple[Box, str, str] | None:
    """(frame, label, role) of what Accessibility already names: an element mark (frame, label, role),
    an ax.Element (x, y, w, h, title/value, role) or a (frame, label[, role]) tuple. Role "": unknown."""
    if isinstance(o, tuple) and len(o) in (2, 3):
        return tuple(map(float, o[0])), str(o[1] or ""), str(o[2] or "") if len(o) == 3 else ""  # type: ignore[return-value]
    if hasattr(o, "frame") and hasattr(o, "label"):
        return tuple(map(float, o.frame)), str(o.label or ""), str(getattr(o, "role", "") or "")  # type: ignore[return-value]
    if all(hasattr(o, k) for k in ("x", "y", "w", "h")):
        label = str(getattr(o, "title", "") or getattr(o, "value", "") or "")
        return (float(o.x), float(o.y), float(o.w), float(o.h)), label, str(getattr(o, "role", "") or "")
    return None


def _snug(frame: Box, box: Box) -> bool:
    """Is ``frame`` about the size of words ``box`` (a control holding them, not a pane)?"""
    return frame[2] <= 3 * box[2] + 40 and frame[3] <= 3 * box[3] + 24


def _named_by_ax(ln: Line, covered: list[tuple[Box, str, str]]) -> bool:
    """Does Accessibility already name these words: an element there — not a container (a web area's or
    a pane's title names the page, not what is drawn in it), and about the words' size when its role is
    unknown — whose label is those words, or holds them as whole words, or they hold it?"""
    for frame, label, role in covered:
        if role in _CONTAINERS or not label.strip() or not _inside(ln.center, frame):
            continue
        if not role and not _snug(frame, ln.box):
            continue
        if _same_words(label, ln.text) or _spans(label, ln.text) or _spans(ln.text, label):
            return True
    return False


def pick_marks(
    reading: Reading, *, covered: Iterable[Any] = (), sheet: Box | None = None, limit: int = MAX_MARKS
) -> list[tuple[str, Box, str]]:
    """(words, box, where) worth listing as text marks (pure): lines Accessibility doesn't name, with a
    letter or digit, the open sheet's first, then in reading order, at most ``limit``."""
    cov = [c for c in (_cover(o) for o in covered) if c is not None]
    out: list[tuple[tuple, tuple[str, Box, str]]] = []
    seen: set[tuple] = set()
    for i, ln in enumerate(reading.lines):
        if ln.conf < _MIN_CONF or not any(ch.isalnum() for ch in ln.text) or _named_by_ax(ln, cov):
            continue
        k = (key(ln.text), round(ln.box[0]), round(ln.box[1]))
        if k in seen:
            continue
        seen.add(k)
        where = "sheet" if sheet is not None and _inside(ln.center, sheet) else ""
        out.append(((0 if where else 1, i), (ln.text[:_LABEL_MAX * 2], ln.box, where)))
    out.sort(key=lambda t: t[0])
    return [m for _, m in out[:limit]]


def remember_marks(
    chosen: Iterable[tuple[str, Box, str]],
    *,
    app: str = "",
    pid: int = 0,
    context: str | None = None,
    owners: Sequence[tuple[str, int]] | None = None,
    also: Sequence[Sequence[str]] | None = None,
    line_boxes: Sequence[Box | None] | None = None,
) -> list[TextMark]:
    """Issue ids (t1, t2…: never reused while NEO runs, so an old id can't name newer words) for these
    words, clickable by id until MARK_TTL_S passes while they are still there. ``context``: all the
    words read with them (default: theirs); per mark (same order): ``owners`` (app, pid) whose window
    they are on (default: ``app``, ``pid``), ``also`` other versions of its words a read saw there,
    ``line_boxes`` where its whole line is."""
    chosen = list(chosen)
    if context is None:
        context = " · ".join(text for text, _, _ in chosen)
    context = context[:2000]
    now = time.monotonic()
    out = []
    with _marks_lock:
        for i, (text, box, where) in enumerate(chosen):
            who, whose = owners[i] if owners else (app, pid)
            lb = line_boxes[i] if line_boxes else None
            m = TextMark(
                f"t{next(_mark_ids)}", text, tuple(map(float, box)), where, who, int(whose), now, context,  # type: ignore[arg-type]
                tuple(str(v) for v in (also[i] if also else ())), tuple(map(float, lb)) if lb else None,  # type: ignore[arg-type]
            )  # fmt: skip
            _marks[m.id] = m
            out.append(m)
        while len(_marks) > _KEEP_MARKS:
            _marks.pop(next(iter(_marks)))
    return out


def get_mark(eid: str) -> TextMark | None:
    """The live text mark with this id (None: never issued, or gone)."""
    with _marks_lock:
        m = _marks.get(str(eid).strip())
    if m is None or time.monotonic() - m.t > MARK_TTL_S:
        return None
    return m


def _commits(text: str, context: str = "") -> bool:
    from neo.security import policy

    try:
        return bool(policy.commits(text, context if policy._GENERIC_OK.match(text or "") else ""))
    except Exception:  # noqa: BLE001 - can't tell: treat as committing (the id path refuses, click_text asks)
        return True


class MarkPoint(tuple):
    """A text mark's screen point (x, y) — a plain pair to the pointer's seam — with whose words they are
    (``pid``, ``app``): the seam brings that app forward and refuses a point another window covers."""

    pid: int
    app: str

    def __new__(cls, point: tuple[float, float], pid: int = 0, app: str = "") -> MarkPoint:
        self = super().__new__(cls, (float(point[0]), float(point[1])))
        self.pid, self.app = int(pid), str(app)
        return self


def _no_commit(m: TextMark, words: Iterable[str]) -> None:
    if any(_commits(w, m.context) for w in words):
        raise LookupError(
            f"{m.id}'s words commit something (a send, a delete, a purchase…): use click_text with those words "
            "instead — it asks the user first"
        )


def _grow(b: Box, d: float) -> Box:
    return b[0] - d, b[1] - d, b[2] + 2 * d, b[3] + 2 * d


def _still_there(m: TextMark) -> list[str]:
    """At click time: are ``m``'s words still where they were read, on the same app's window, with no
    other window over them? The window server says whose window is on top there; a fresh capture of
    the words' line and its neighbourhood is read again (fast, a small crop: a few ms; once more framed
    differently, then accurately, before it is called gone). Returns the words read there now; raises
    LookupError (NEO's own words) when they aren't there — or when that can't be told."""
    unsure = f"I can't check that {m.id}'s words are still there — look again (screenshot) before clicking"
    line = m.line_box or m.box
    area = _grow(line, max(_RECHECK_PAD_PT, line[3]))
    wins = windows_over(area)
    if wins is None or not available():
        raise LookupError(unsure)
    top = wins.at(m.center)
    if top is None or top.off_limits or (m.pid and top.pid != m.pid):
        raise LookupError(f"another window is over {m.id}'s words now — look again (screenshot) for fresh ids")
    px = grab(area)
    plan = plan_for(m.text)
    if px is None or plan is None:
        raise LookupError(unsure)
    tries = [(plan, 0), (plan, _PAD_PX)]
    if plan.fast and (closer := plan_for(m.text, want_fast=False)) is not None:
        tries.append((closer, 0))
    names = (m.text, *m.also)
    for p, pad in tries:
        try:
            r = owned(read(px, area, plan=p, pad=pad), wins, top.pid)
        except (Failed, ValueError, effects.EffectRefused) as ex:
            log.info("couldn't check a text mark: %s", ex)
            raise LookupError(unsure) from None
        there = [
            ln.text for ln in r.lines
            if ln.conf >= _MIN_CONF and _inside(m.center, _grow(ln.box, 3.0))
            and any(_similar(ln.text, n) or _spans(ln.text, n) for n in names)
        ]  # fmt: skip
        if there:
            return there
    raise LookupError(f"{m.id}'s words aren't where they were any more — look again (screenshot) for fresh ids")


def resolve_mark(eid: str) -> MarkPoint:
    """The pointer's seam for text marks (``pointer.set_id_resolver("t", resolve_mark, labels=mark_labels)``):
    the screen point of a live mark whose words are still there (:func:`_still_there`), with the app
    whose window they are on. A mark whose words send, delete, buy… (or an "OK" in a dialog that says
    so) — in any version a read saw, then or now — is never clicked by id: click_text with its words
    asks the user first. Sync, and it captures and reads: call it off the event loop. Errors are
    LookupError with NEO's own words (the screen's never go into them, nor any point)."""
    m = get_mark(eid)
    if m is None:
        raise LookupError(f"there's no text mark {str(eid)[:12]} any more — look again (screenshot) for fresh ones")
    _no_commit(m, (m.text, *m.also))
    _no_commit(m, _still_there(m))
    return MarkPoint(m.center, m.pid, m.app)


def _verdict(m: TextMark) -> str:
    """For a plain "OK" whose dialog says it deletes or sends: a label that says so itself — the click
    gate beside the pointer reads a dialog's words through Accessibility, and a drawn one has none."""
    from neo.security import policy

    line = next((p for p in m.context.split(" · ") if policy._CONSEQUENCE.search(p)), "")
    said = f"{m.text} — {line[:_LABEL_MAX]}" if line else ""
    return said if said and policy.commits(said) else f"{m.text} — confirm"


def mark_labels(eid: str) -> list[str]:
    """The words of text mark ``eid`` ([] if gone) — every version a read saw, and for an "OK" in a
    dialog that commits something, a label that says so — for the click gate beside what's under the
    point."""
    m = get_mark(eid)
    if m is None:
        return []
    out = [m.text, *m.also]
    if any(_commits(w, m.context) and not _commits(w) for w in out):
        out.append(_verdict(m))
    return out


def text_marks(
    frame: Any,
    area: Box | None = None,
    *,
    covered: Iterable[Any] = (),
    sheet: Box | None = None,
    app: str = "",
    pid: int = 0,
    limit: int = MAX_MARKS,
) -> list[TextMark]:
    """Text marks for a picture NEO took (sync: in a thread, beside the Accessibility walk): an accurate
    read of ``area`` of ``frame`` (the capture pipeline's full-resolution frame) and a fast one, only
    the words on ``pid``'s windows where nothing covers them (with 0: any app's but NEO's own or a
    protected one's, each mark its window's app's), minus those Accessibility already names
    (``covered``: its marks or elements), issued as t-ids. [] when nothing could be read."""
    if not available():
        return []
    px = pixels(frame)
    where = area or (px.x0, px.y0, px.image.width * px.ppp, px.image.height * px.ppp)
    wins = windows_over(where)
    plan = plan_for("", want_fast=False)
    if wins is None or plan is None:
        return []
    try:
        reads = [owned(read(px, area, plan=plan), wins, pid)]
        quick = plan_for("", want_fast=True)
        if quick is not None and quick.fast:  # (a second read catches most of the words one drops: +6–25 ms)
            reads.append(owned(read(px, area, plan=quick), wins, pid))
    except (Failed, ValueError, effects.EffectRefused) as ex:
        log.info("no text marks: %s", ex)
        return []
    reading = merge(*reads)
    chosen = pick_marks(reading, covered=covered, sheet=sheet, limit=limit)
    owners: list[tuple[str, int]] = []
    for _, box, _ in chosen:
        w = wins.at((box[0] + box[2] / 2, box[1] + box[3] / 2))
        whose = w.pid if w is not None else pid
        owners.append((app if app and pid and whose == pid else (w.app if w is not None else app), whose))
    also = [_variants(text, box, reads) for text, box, _ in chosen]
    return remember_marks(chosen, context=reading.text(), owners=owners, also=also)


async def look_marks(
    frame: Any,
    area: Box | None = None,
    *,
    named: Iterable[Any] = (),
    window: Box | None = None,
    blind: bool = False,
    sheet: Box | None = None,
    app: str = "",
    pid: int = 0,
) -> list[TextMark]:
    """Text marks for a look (a screenshot, observe) when they are worth reading (:func:`worth_reading`:
    always for a brain that can't see; for one that can, when Accessibility names little of the window),
    off the event loop. ``named``: Accessibility's marks or elements for the same area."""
    named = list(named)
    if not available() or not worth_reading(window or area, named, blind=blind):
        return []
    return await asyncio.to_thread(text_marks, frame, area, covered=named, sheet=sheet, app=app, pid=pid)


def _takes(fn: Any, name: str) -> bool:
    """Does ``fn`` take the keyword argument ``name``?"""
    import inspect

    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.kind is p.VAR_KEYWORD or (p.name == name and p.kind is not p.POSITIONAL_ONLY) for p in params)


def install() -> bool:
    """Register the text marks with the pointer's id seam (the Element marks' branch adds it), with
    their words for its click gate. True when registered."""
    from neo.tools.computer import pointer

    setter = getattr(pointer, "set_id_resolver", None)
    if setter is None:
        return False
    if _takes(setter, "labels"):
        setter("t", resolve_mark, labels=mark_labels)
    else:
        setter("t", resolve_mark)
    return True


def _box_2d(box: Box, shot: Any) -> tuple[int, int, int, int]:
    """Where ``box`` is in picture ``shot`` (x0, y0, ppp, w, h): [ymin, xmin, ymax, xmax] on 0–1000."""
    wpt, hpt = shot.w * shot.ppp, shot.h * shot.ppp

    def n(v: float, size: float) -> int:
        return int(round(min(1000.0, max(0.0, v / size * 1000.0)))) if size > 0 else 0

    x, y, w, h = box
    return n(y - shot.y0, hpt), n(x - shot.x0, wpt), n(y + h - shot.y0, hpt), n(x + w - shot.x0, wpt)


def mark_line(m: TextMark, shot: Any = None, *, blind: bool) -> str:
    """One mark as a line of a result: ``t4 Text “Start” [sheet]`` — plus ``box_2d …`` for a brain that
    can see the picture ``shot``; never a number for one that can't (it would only copy it)."""
    words = " ".join(m.text.replace("“", '"').replace("”", '"').split())[:_LABEL_MAX]
    bits = [m.id, "Text", f"“{words}”"]
    if m.where:
        bits.append(f"[{m.where}]")
    if shot is not None and not blind:
        bits.append("box_2d " + ",".join(str(v) for v in _box_2d(m.box, shot)))
    return " ".join(bits)


def marks_block(marks: list[TextMark], shot: Any = None, *, blind: bool) -> str:
    """Text marks as a result's text: NEO's words (unfenced) around the screen's (fenced). "" when none.
    The caller taints the request (the screen's words went into the result)."""
    from neo.agent.registry import fence

    if not marks:
        return ""
    if blind:
        head = "Words NEO read off the screen where Accessibility names nothing (you can't see them, but these ids work: click id=t…):"
    else:
        head = (
            "Words NEO read off this picture where Accessibility names nothing — click id=t… clicks the middle of those "
            "words" + (" (box_2d: where they are in the image, [ymin, xmin, ymax, xmax] on 0–1000)." if shot is not None else ".")
        )
    return "\n" + head + "\n" + fence("screen", "\n".join(mark_line(m, shot, blind=blind) for m in marks))


def worth_reading(window: Box | None, named: Iterable[Any], *, blind: bool, min_share: float = 0.35, min_named: int = 4) -> bool:
    """Should a look list text marks? For a brain that can't see: always (the words are all it gets).
    For one that can: when Accessibility names little of the window — fewer than ``min_named`` labelled
    elements, or their frames cover under ``min_share`` of it (a canvas, a game, a drawn dialog). A
    container's title (a web area's, a pane's) names none of what is drawn inside it: it doesn't count."""
    if blind:
        return True
    cov = [c for c in (_cover(o) for o in named) if c is not None and c[1].strip() and c[2] not in _CONTAINERS]
    if len(cov) < min_named or window is None:
        return True
    wx, wy, ww, wh = window
    if ww <= 0 or wh <= 0:
        return True
    n = 24  # the window as a 24×24 grid of cells: the share of cells some named element covers
    hit = 0
    for i in range(n):
        for j in range(n):
            p = (wx + (i + 0.5) * ww / n, wy + (j + 0.5) * wh / n)
            if any(_inside(p, f) for f, _, _ in cov):
                hit += 1
    return hit / (n * n) < min_share


# ---- what changed: the words before and after -----------------------------------------------------------


@dataclass(frozen=True)
class Change:
    gone: tuple[str, ...] = ()
    new: tuple[str, ...] = ()
    changed: tuple[tuple[str, str], ...] = ()  # (before, after) in the same place

    def __bool__(self) -> bool:
        return bool(self.gone or self.new or self.changed)


def _similar(a: str, b: str) -> bool:
    """The same words, give or take a read's wobble — never when a number in them differs (a score, a
    count: "Score 3" → "Score 4" is the change)."""
    ka, kb = key(a), key(b)
    if ka == kb:
        return True
    if re.findall(r"\d+", ka) != re.findall(r"\d+", kb):
        return False
    return difflib.SequenceMatcher(None, ka, kb).ratio() >= 0.85


def diff(before: Reading, after: Reading) -> Change:
    """How the words differ between two reads of the same place: gone, new, and changed in place. Words
    that only moved, and a read's small wobbles on the same words, are no change."""
    b_lines = [ln for ln in before.lines if ln.conf >= _MIN_CONF]
    a_lines = [ln for ln in after.lines if ln.conf >= _MIN_CONF]
    used: set[int] = set()
    gone, changed = [], []
    for b in b_lines:
        j = next((j for j, a in enumerate(a_lines) if j not in used and _similar(a.text, b.text) and _overlap(a.box, b.box)), None)
        if j is None:
            j = next((j for j, a in enumerate(a_lines) if j not in used and _overlap(a.box, b.box) and (_inside(a.center, b.box) or _inside(b.center, a.box))), None)
            if j is not None and not _similar(a_lines[j].text, b.text):
                changed.append((b.text, a_lines[j].text))
        if j is not None:
            used.add(j)
            continue
        j = next((j for j, a in enumerate(a_lines) if j not in used and _similar(a.text, b.text)), None)
        if j is not None:  # the same words elsewhere: moved, not gone
            used.add(j)
            continue
        gone.append(b.text)
    new = [a.text for j, a in enumerate(a_lines) if j not in used]
    return Change(tuple(gone), tuple(new), tuple(changed))


def _q(s: str) -> str:
    s = " ".join(s.replace("“", '"').replace("”", '"').split())
    return f"“{s[:_LABEL_MAX]}{'…' if len(s) > _LABEL_MAX else ''}”"


def words_changed(before: Reading, after: Reading, limit: int = 3) -> str:
    """What the words did, in a few words — “Start” → “Stop”; “Start a New Game?” is gone; new: “Game
    over” — or "" when they didn't change. The screen's words: the caller fences it."""
    c = diff(before, after)
    bits = [f"{_q(a)} → {_q(b)}" for a, b in c.changed]
    bits += [f"{_q(g)} is gone" for g in c.gone]
    bits += [f"new: {_q(n)}" for n in c.new]
    if not bits:
        return ""
    more = len(bits) - limit
    return "; ".join(bits[:limit]) + (f" (and {more} more)" if more > 0 else "")


def sample_change(before: list | None, after: list | None) -> str:
    """The words that changed between the change samples of one input, from the full-detail crops the
    capture pipeline keeps with each sample (``sample.info["full"]`` and ``["ppp"]``): read fast, only
    where the samples differ. "" when nothing can be said (no crops, no engine, no change in words)."""
    if not before or not after or not available():
        return ""
    from neo.tools.computer import screen

    out: list[str] = []
    for b, a in zip(before, after, strict=False):
        fb, fa = b.info.get("full"), a.info.get("full")
        if fb is None or fa is None or fb.size != fa.size or not screen.differs(b, a):
            continue
        ppp = float(b.info.get("ppp") or a.info.get("ppp") or 0.5)
        try:
            said = compare_pictures(Pixels(fb, 0.0, 0.0, ppp), Pixels(fa, 0.0, 0.0, ppp))
        except (Failed, ValueError, effects.EffectRefused) as ex:
            log.info("no words compared: %s", ex)
            return ""
        if said:
            out.append(said)
    return "; ".join(out)


def compare_pictures(before: Any, after: Any, area: Box | None = None, *, plan: Plan | None = None) -> str:
    """What the words did between two pictures of the same place (frames; ``area`` in screen points):
    words_changed of a read of each — a word only one read missed is no change, so words that seem
    gone (or new) are looked for again in a second framing of the picture that lacks them first."""
    plan = plan or plan_for("")
    if plan is None:
        return ""
    rb, ra = read(before, area, plan=plan), read(after, area, plan=plan)
    c = diff(rb, ra)
    if not c:
        return ""
    if c.gone:
        ra = merge(ra, read(after, area, plan=plan, pad=_PAD_PX))
    if c.new:
        rb = merge(rb, read(before, area, plan=plan, pad=_PAD_PX))
    return words_changed(rb, ra)


# ---- the screen: capture ---------------------------------------------------------------------------------


def _clip_to_display(area: Box) -> tuple[int, int, int, int] | None:
    from neo.tools.computer import screen

    x, y, w, h = area
    home = screen.display_at(x + w / 2, y + h / 2) or screen.display_at(x, y)
    if home is None:
        return None
    left, top = max(x, home.x), max(y, home.y)
    right, bottom = min(x + w, home.x + home.w), min(y + h, home.y + home.h)
    if right - left < 4 or bottom - top < 4:
        return None
    li, ti = int(left), int(top)
    return li, ti, max(1, int(round(right)) - li), max(1, int(round(bottom)) - ti)


def grab(area: Box) -> Pixels | None:
    """What is on screen in ``area`` (screen points) at the display's own resolution — the capture
    pipeline's frame when it has one (a fresh frame it already holds, else its grab), else one
    screencapture. None without Screen Recording (never asks for it) or off every display."""
    from neo.tools.computer import screen

    whole = _clip_to_display(area)
    if whole is None or not screen.can_capture():
        return None
    last = getattr(screen, "last_frame", None)
    if callable(last):
        f = last(0.3)
        if f is not None and not getattr(f, "window_id", 0) and f.covers(*whole):
            return pixels(f)
    fn = getattr(screen, "grab", None)
    if callable(fn):
        return pixels(fn(whole))
    x, y, w, h = whole
    img = screen._grab(["-R", f"{x},{y},{w},{h}"])
    return Pixels(img, float(x), float(y), w / max(1, img.width))


def read_area(area: Box, *, text: str = "", accurate: bool = False, pid: int = 0) -> Reading | None:
    """Capture ``area`` and read its words (sync: in a thread) — only those on ``pid``'s windows where
    nothing covers them (0: any app's but NEO's own or a protected one's). ``text``: words that will be
    looked for (their script picks the languages). None when nothing could be read (no engine, no
    Screen Recording, the window server can't say whose windows are there)."""
    if not available():
        return None
    wins = windows_over(area)
    if wins is None:
        return None
    px = grab(area)
    if px is None:
        return None
    plan = plan_for(text, want_fast=not accurate)
    if plan is None:
        return None
    try:
        return owned(read(px, area, plan=plan), wins, pid)
    except (Failed, ValueError, effects.EffectRefused) as ex:
        log.info("couldn't read the screen: %s", ex)
        return None


def find_text_sync(label: str, window: Box, *, sheet: Box | None = None, pid: int = 0) -> Result | None:
    """:func:`find_in` on a fresh capture of ``window`` (screen points), only the words on ``pid``'s
    windows where no other window covers them (0: any app's but NEO's own or a protected one's). None
    when the screen couldn't be read (no engine, no Screen Recording, the window server can't say whose
    windows are there) — not a "not found": the caller keeps its own answer."""
    if not available():
        return None
    if bad := unreadable(label):
        return Missing(label, f"this Mac's text recognition can't read {bad}")
    if plan_for(label) is None:
        return None
    wins = windows_over(window)
    if wins is None:
        return None
    px = grab(window)
    if px is None:
        return None
    try:
        return find_in(px, label, area=window, sheet=sheet, keep=lambda r: owned(r, wins, pid))
    except (Failed, ValueError, effects.EffectRefused) as ex:
        log.info("couldn't read the screen: %s", ex)
        return None


async def find_text(label: str, window: Box, *, sheet: Box | None = None, pid: int = 0) -> Result | None:
    """Where ``label``'s words are shown in ``window`` (the front window's frame, screen points) on
    ``pid``'s windows: Found, Ambiguous or Missing — or None when the screen couldn't be read. Off the
    event loop."""
    return await asyncio.to_thread(find_text_sync, label, window, sheet=sheet, pid=pid)


def sheet_frame(els: Iterable[Any] | None) -> Box | None:
    """The frame of an open sheet or dialog among a window's elements (ax.window_elements), if any."""
    for e in els or []:
        if getattr(e, "role", "") in ("AXSheet", "AXDialog") and getattr(e, "w", 0) > 0 and getattr(e, "h", 0) > 0:
            return float(e.x), float(e.y), float(e.w), float(e.h)
    return None


# ---- click_text's fallback: the probe → gate → click path, by words read off the screen -----------------


@dataclass
class _Said:
    """Words read off the screen, as an element for the gate's dialog-text check (_dialog_text)."""

    value: str
    role: str = "AXStaticText"
    title: str = ""


def _gates():
    """The package's gate helpers, looked up at call time (a test's monkeypatch and Step 0's frozen
    gates module, re-exported by the package, both reach this path)."""
    from neo.tools import computer as _c

    return _c


def _at(p: tuple[float, float]) -> str:
    return f"({int(round(p[0]))}, {int(round(p[1]))})"


_POINT = re.compile(r"\s*\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)")


def _front_pid(app: str) -> int:
    """The pid of ``app`` while it is the app whose window was read: the one ax.frontmost_app names —
    the answer ax.window_elements went by (a passing banner in front is skipped for the app the user
    is in) — under that same name. 0 when it isn't (it changed since) or can't be told."""
    from neo.decide._objc import pool
    from neo.tools.computer import ax

    try:
        with pool():
            name, pid = ax.frontmost_app()
    except Exception:  # noqa: BLE001 - can't tell: 0 (the fallback steps aside)
        return 0
    same = bool(app) and " ".join(str(name or "").split()).lower() == " ".join(app.split()).lower()
    return int(pid) if pid and same else 0


def _refusal(g: Any, c: Any, r: Any) -> Any:
    """The pointer's refusal as the tool's result — for a brain that can't see, without any point in
    it (bringing an app forward names the spot that changed)."""
    if c.get("blind") and not r.protected:
        from neo.tools.computer import pointer

        r = pointer.Refused(_POINT.sub("", str(r)), screen=r.screen, protected=r.protected)
    return g._refused(c, r)


async def _bring(t: Any, pts: list[tuple[float, float]], pid: int, blind: bool) -> None:
    """pointer.bring for the app whose words they are — told the brain can't see when it takes that."""
    from neo.tools.computer import pointer

    if _takes(pointer.bring, "blind"):
        await pointer.bring(t, pts, want_pid=pid, blind=blind)
    else:
        await pointer.bring(t, pts, want_pid=pid)


async def click_text_fallback(
    a: dict, c: Any, *, app: str, window: Box | None, els: list | None = None, pid: int = 0
) -> Any | None:
    """click_text when Accessibility names nothing with the label: read the front window's words and
    click the place showing them — through the same checks as any click (NEO's own window and protected
    apps refused, a window a sheet blocks refused, the words only where ``app``'s own window shows them
    with nothing over them, the confirmation gate on every version of the words the reads saw AND on
    what is under the point, the words of the dialog as its context; only ``app`` ever brought forward).
    ``pid``: ``app``'s process (the window's owner); when 0 it is asked (ax.frontmost_app, under the
    same name). Returns the tool's result, or None when the screen couldn't be read or whose window it
    is can't be told (the handler keeps its own answer)."""
    from neo.agent.registry import ToolOutput, fence
    from neo.tools.computer import input as inp
    from neo.tools.computer import pointer

    g = _gates()
    label = str(a.get("label") or "")
    blind = bool(c.get("blind"))
    if window is None or not available():
        return None
    if not pid:
        pid = await asyncio.to_thread(_front_pid, app)
    if not pid:
        return None  # whose window was read can't be told: nothing is read, nothing brought forward
    sheet = sheet_frame(els)
    got = await find_text(label, window, sheet=sheet, pid=pid)
    if got is None:
        return None
    if isinstance(got, Missing):
        return ToolOutput(
            f"Error: nothing in the front window of {app or 'the front app'} is called '{label}' — neither "
            f"Accessibility nor the screen's words show it ({got.why}).",
            ok=False,
        )
    if isinstance(got, Ambiguous):
        # Each place's id carries the words around it (a drawn dialog's "OK" is judged by what the
        # dialog says) and every version of its words the reads saw.
        context = " · ".join(dict.fromkeys(f.context for f in got.options if f.context))
        marks = remember_marks(
            [(f.text if f.exact else f.line, f.box, f.where) for f in got.options],
            app=app,
            pid=pid,
            context=context or None,
            also=[tuple(v for v in (f.text, *f.variants) if key(v) != key(f.text if f.exact else f.line)) for f in got.options],
            line_boxes=[f.line_box for f in got.options],
        )
        g._taint(c, "screen")
        text = (
            f"Error: several places on the screen read '{label}' — which one? Click it by its id (click id=t…), "
            "or use more of its words."
        ) + marks_block(marks, None, blind=True)
        return ToolOutput(text, ok=False)
    found: Found = got
    pts = [found.point]
    try:
        t = await pointer.probe(pts)  # NEO's own window, a protected app, a window a sheet blocks: refused
        h = t.hits[0]
        if h.pid != pid:  # (another app's window came over the words since they were read — or no one's)
            raise pointer.Refused(
                f"Error: another window ({h.app or 'another app'}) is over the words '{label}' in {app or 'the front app'} "
                "— look again (screenshot) before clicking.",
            )
        # The gate judges the whole line the words are in (a "Delete" inside "Delete All" is one card, not
        # two) and every other version of it a read saw there, with the dialog's words as the context of
        # a plain "OK"; then what is under the point.
        said = [_Said(found.context or found.line)] + list(els or [])
        for words in dict.fromkeys((found.line, *found.variants)):
            if gate := await g._ask_first("click_text", a, words, app, said):
                return gate
        if gate := await g._gate_click(a, [(h.label, h.app or app)]):
            return gate
        await _bring(t, pts, pid, blind)
        if t.relabeled and (gate := await g._gate_click(a, [(t.hits[0].label, t.hits[0].app or app)])):
            return gate
    except pointer.Refused as r:
        return _refusal(g, c, r)
    h = t.hits[0]
    # (never a number for a brain that can't see: it would only copy it into a click)
    spot = "the words" if blind else _at(pts[0])
    head = f"Clicked the words '{label}'" + ("" if blind else f" at {spot}")
    head += f" in {h.app or app}" if (h.app or app) else ""
    if t.brought:
        head += f" (brought {t.brought} to the front first)"
    head += " — found by reading the screen (Accessibility has no such label)."
    before = await g._watch(c, pid, pts, hover=pts[0])
    await asyncio.to_thread(inp.click, *pts[0])
    g._taint(c, "screen")
    read_there = f"read there: {_q(found.line)}" + (" [in the open sheet]" if found.where else "")
    if before is None:
        return ToolOutput(head + g._input_doubt() + "\n" + fence("screen", read_there))
    out = await pointer.outcome(before, pid, pts, [spot])
    out = pointer.maybe_elsewhere(out, spot)
    text, ok = pointer.report(head, out, [pointer.words(h), read_there], g._input_doubt())
    return _with_effect(ToolOutput(text, ok=ok), out)


def _with_effect(result: Any, outcome: Any) -> Any:
    """The result carrying what changed (Step 0's ToolOutput.effect, through the pointer's helpers once
    they are there); unchanged before."""
    from neo.tools.computer import pointer

    carry, of = getattr(pointer, "with_effect", None), getattr(pointer, "effect_of", None)
    return carry(result, of(outcome)) if callable(carry) and callable(of) else result
