"""Reading the screen's words: Apple Vision's on-device text recognition over NEO's own captures.

Accessibility names most controls, but not a canvas, a game, a web view that hides its tree, or a
dialog an app draws itself. There NEO reads the pixels it captured itself, three ways:

* **text targets** — :func:`find_text` (click_text's fallback): the words asked for, found in the
  front window at full Retina detail, an open sheet's words first. The point then goes through the
  same probe → gate → click path as every other click (:func:`click_text_fallback`).
* **text marks** — t1…tn for words Accessibility doesn't cover, listed with a picture
  (:func:`marks_block`): ids and words only for a brain that can't see (NEO read the pixels, so it may
  act on them by id), plus where they are for one that can. A mark whose words commit something is
  never clicked by id: click_text asks the user first.
* **checking effects** — the words around a click before and after (:func:`sample_change`,
  :func:`words_changed`): “Start a New Game?” is gone, “Start” → “Stop”.

Speed by need (synthetic Retina frames, macOS 26, recognition revision 3; tests/test_eyes_ocr_live.py):

* fast mode: about 6 ms for a 560×200 pt dialog crop, about 23 ms for a whole Retina display. It needs
  about two pixels per point — on a picture shrunk to one pixel per point it garbles (“CarKel”) — so a
  picture with fewer is scaled up 2× before a fast read (1–14 ms);
* accurate mode: about 36 ms for the crop, about 95 ms for the display, 30 languages;
* fast for change checks and the first try at a target (taken only when it reads the words exactly);
  accurate when fast missed or was unsure, for marks, and for any script fast doesn't read.
* The picture reaches Vision as raw grey pixels (about 1.5 ms for a Retina display), never through a
  PNG (about 35 ms).

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
            if e is not None:
                e.recognize(Image.new("L", (64, 32), 255), fast=True, languages=["en-US"], auto=False, ranges=None)
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
        chosen = [c for c in _pick(acc, [*_preferred_languages(), "en"]) if need <= _lang_scripts(c)]
        chosen += [c for c in acc if need <= _lang_scripts(c) and c not in chosen]
        return Plan(False, tuple(chosen[:4])) if chosen else None
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


def read(
    frame: Any,
    area: Box | None = None,
    *,
    plan: Plan | None = None,
    ranges: Callable[[str], list[tuple[int, int]]] | None = None,
) -> Reading:
    """Read the words in ``area`` of a frame (all of it when None), at the frame's full resolution —
    scaled up 2× first when a fast read would get too few pixels per point. Raises Failed when there is
    no engine or it fails."""
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
    try:
        raws = e.recognize(grey, fast=plan.fast, languages=list(plan.languages), auto=plan.auto, ranges=ranges)
    except Failed:
        raise
    except effects.EffectRefused:
        raise
    except Exception as ex:  # noqa: BLE001 - an engine error is a failed read, never a crash of the tool
        raise Failed(f"text recognition failed ({type(ex).__name__})") from ex
    lines = []
    for r in raws:
        text = " ".join(str(r.text).split())
        if not text:
            continue
        parts = tuple((s, en, _to_screen(b, x0, y0, w_pt, h_pt)) for s, en, b in r.parts)
        lines.append(Line(text, float(r.conf), _to_screen(r.box, x0, y0, w_pt, h_pt), parts))
    lines.sort(key=lambda ln: (round(ln.box[1] / 6), ln.box[0]))
    ms = (time.perf_counter() - t0) * 1000
    return Reading(tuple(lines), (x0, y0, w_pt, h_pt), FAST if plan.fast else ACCURATE, ms, px.t, per_pt)


def without(reading: Reading, rects: Iterable[Box]) -> Reading:
    """The reading without lines whose middle is in any of ``rects`` (a protected app's windows)."""
    rects = list(rects)
    if not rects:
        return reading
    keep = tuple(ln for ln in reading.lines if not any(_inside(ln.center, r) for r in rects))
    return replace(reading, lines=keep)


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
            f = Found(label, ln.text, ln.text, ln.center, ln.box, True, where, reading.level)
            cands.append(((0 if where or sheet is None else 1, 0), f))
            continue
        extra = max(0, len(key(ln.text).split()) - len(want.split()))
        for start, end in _spans(ln.text, label):
            b = ln.part_box(start, end)
            p = (b[0] + b[2] / 2, b[1] + b[3] / 2)
            w = "sheet" if sheet is not None and _inside(p, sheet) else ""
            f = Found(label, ln.text[start:end], ln.text, p, b, False, w, reading.level)
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
            if chunk and chunk != want and difflib.SequenceMatcher(None, chunk, want).ratio() >= 0.6:
                return True
    return False


def find_in(frame: Any, label: str, *, area: Box | None = None, sheet: Box | None = None, hide: Iterable[Box] = ()) -> Result:
    """Find ``label``'s words in a frame (sync; in a thread). A fast read first, taken only when it
    reads the words exactly, in one place (in the open sheet when there is one), with nothing that
    merely looks like them elsewhere; otherwise — and for any script fast can't read — an accurate read
    decides. ``hide``: screen rectangles whose words never count (a protected app's windows)."""
    if bad := unreadable(label):
        return Missing(label, f"this Mac's text recognition can't read {bad}")
    first = plan_for(label, want_fast=True)
    if first is None:
        raise Failed("this Mac's text recognition can't be asked")
    rng = _ranges_for(label)
    hide = list(hide)
    if first.fast:
        fast = without(read(frame, area, plan=first, ranges=rng), hide)
        got = match(fast, label, sheet=sheet)
        if isinstance(got, Found) and got.exact and (sheet is None or got.where) and not _lookalikes(fast, label, got):
            return got
    second = plan_for(label, want_fast=False) or first
    return match(without(read(frame, area, plan=second, ranges=rng), hide), label, sheet=sheet)


# ---- text marks: t1…tn ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class TextMark:
    """Words NEO read off a capture that a model may act on by id: the middle of ``box`` (screen points).
    ``context``: the other words read with it (what a dialog's "OK" does is in them)."""

    id: str
    text: str
    box: Box
    where: str = ""
    app: str = ""
    pid: int = 0
    t: float = field(default_factory=time.monotonic)
    context: str = ""

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.box
        return x + w / 2, y + h / 2


MAX_MARKS = 30
MARK_TTL_S = 120.0  # a mark older than this is gone ("look again")
_KEEP_MARKS = 300
_marks: dict[str, TextMark] = {}
_mark_ids = itertools.count(1)
_marks_lock = threading.Lock()


def _cover(o: Any) -> tuple[Box, str] | None:
    """(frame, label) of what Accessibility already names: an element mark (frame, label), an
    ax.Element (x, y, w, h, title/value) or a (frame, label) pair."""
    if isinstance(o, tuple) and len(o) == 2:
        return tuple(map(float, o[0])), str(o[1] or "")  # type: ignore[return-value]
    if hasattr(o, "frame") and hasattr(o, "label"):
        return tuple(map(float, o.frame)), str(o.label or "")  # type: ignore[return-value]
    if all(hasattr(o, k) for k in ("x", "y", "w", "h")):
        return (float(o.x), float(o.y), float(o.w), float(o.h)), str(getattr(o, "title", "") or getattr(o, "value", "") or "")
    return None


def _named_by_ax(ln: Line, covered: list[tuple[Box, str]]) -> bool:
    """Does Accessibility already name these words (an element there whose label is — or holds — them)?"""
    k = key(ln.text)
    for frame, label in covered:
        if not _inside(ln.center, frame):
            continue
        lk = key(label)
        if lk and (lk == k or k in lk or lk in k):
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
    chosen: Iterable[tuple[str, Box, str]], *, app: str = "", pid: int = 0, context: str | None = None
) -> list[TextMark]:
    """Issue ids (t1, t2…: never reused while NEO runs, so an old id can't name newer words) for these
    words, clickable by id until MARK_TTL_S passes. ``context``: all the words read with them (default:
    theirs)."""
    chosen = list(chosen)
    if context is None:
        context = " · ".join(text for text, _, _ in chosen)
    context = context[:2000]
    now = time.monotonic()
    out = []
    with _marks_lock:
        for text, box, where in chosen:
            m = TextMark(f"t{next(_mark_ids)}", text, tuple(map(float, box)), where, app, int(pid), now, context)  # type: ignore[arg-type]
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


def resolve_mark(eid: str) -> tuple[float, float]:
    """The pointer's seam for text marks (``pointer.set_id_resolver("t", resolve_mark)``): the screen
    point of a live mark. A mark whose words send, delete, buy… (or an "OK" in a dialog that says so) is
    never clicked by id — click_text with its words asks the user first. Errors are LookupError with
    NEO's own words (the screen's never go into them)."""
    m = get_mark(eid)
    if m is None:
        raise LookupError(f"there's no text mark {str(eid)[:12]} any more — look again (screenshot) for fresh ones")
    if _commits(m.text, m.context):
        raise LookupError(
            f"{m.id}'s words commit something (a send, a delete, a purchase…): use click_text with those words "
            "instead — it asks the user first"
        )
    return m.center


def mark_labels(eid: str) -> list[str]:
    """The words of text mark ``eid`` ([] if gone), for the click gate alongside what's under the point."""
    m = get_mark(eid)
    return [m.text] if m else []


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
    read of ``area`` of ``frame`` (the capture pipeline's full-resolution frame), minus the words a
    protected app's window shows and those Accessibility already names (``covered``: its marks or
    elements), issued as t-ids. [] when nothing could be read."""
    if not available():
        return []
    px = pixels(frame)
    where = area or (px.x0, px.y0, px.image.width * px.ppp, px.image.height * px.ppp)
    hide = protected_rects(where)
    plan = plan_for("", want_fast=False)
    if hide is None or plan is None:
        return []
    try:
        reading = without(read(px, area, plan=plan), hide)
    except (Failed, ValueError) as ex:
        log.info("no text marks: %s", ex)
        return []
    chosen = pick_marks(reading, covered=covered, sheet=sheet, limit=limit)
    return remember_marks(chosen, app=app, pid=pid, context=reading.text())


def install() -> bool:
    """Register the text marks with the pointer's id seam (the Element marks' branch adds it). True
    when registered."""
    from neo.tools.computer import pointer

    setter = getattr(pointer, "set_id_resolver", None)
    if setter is None:
        return False
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
    elements, or their frames cover under ``min_share`` of it (a canvas, a game, a drawn dialog)."""
    if blind:
        return True
    cov = [c for c in (_cover(o) for o in named) if c is not None and c[1].strip()]
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
            if any(_inside(p, f) for f, _ in cov):
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
    ka, kb = key(a), key(b)
    return ka == kb or difflib.SequenceMatcher(None, ka, kb).ratio() >= 0.85


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
            plan = plan_for("")
            if plan is None:
                return ""
            rb = read(Pixels(fb, 0.0, 0.0, ppp), plan=plan)
            ra = read(Pixels(fa, 0.0, 0.0, ppp), plan=plan)
        except (Failed, ValueError, effects.EffectRefused) as ex:
            log.info("no words compared: %s", ex)
            return ""
        if said := words_changed(rb, ra):
            out.append(said)
    return "; ".join(out)


# ---- the screen: capture, protected windows -------------------------------------------------------------


def protected_rects(area: Box) -> list[Box] | None:
    """The on-screen windows over ``area`` that belong to an app NEO never touches (a host's approval
    dialog, a vault) — their words are never read. None when the window server can't say (fail
    closed: nothing is read)."""
    try:
        import Quartz as Q

        from neo.tools import computer as _c
        from neo.tools.computer import ax

        wins = Q.CGWindowListCopyWindowInfo(Q.kCGWindowListOptionOnScreenOnly | Q.kCGWindowListExcludeDesktopElements, Q.kCGNullWindowID)
    except Exception:  # noqa: BLE001
        return None
    if wins is None:
        return None
    out = []
    for w in wins:
        b = w.get("kCGWindowBounds") or {}
        r = tuple(float(b.get(k, 0) or 0) for k in ("X", "Y", "Width", "Height"))
        if r[2] <= 0 or r[3] <= 0 or not _overlap(r, area):  # type: ignore[arg-type]
            continue
        pid, name = int(w.get("kCGWindowOwnerPID", 0) or 0), str(w.get("kCGWindowOwnerName", "") or "")
        if (pid and ax.is_neo(pid, name)) or _c._protected(name, pid):
            out.append(r)
    return out  # type: ignore[return-value]


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


def read_area(area: Box, *, text: str = "", accurate: bool = False) -> Reading | None:
    """Capture ``area`` and read its words (sync: in a thread) — without any a protected app shows.
    ``text``: words that will be looked for (their script picks the languages). None when nothing
    could be read (no engine, no Screen Recording, the window server can't say what's protected)."""
    if not available():
        return None
    hide = protected_rects(area)
    if hide is None:
        return None
    px = grab(area)
    if px is None:
        return None
    plan = plan_for(text, want_fast=not accurate)
    if plan is None:
        return None
    try:
        return without(read(px, area, plan=plan), hide)
    except (Failed, ValueError) as ex:
        log.info("couldn't read the screen: %s", ex)
        return None


def find_text_sync(label: str, window: Box, *, sheet: Box | None = None) -> Result | None:
    """:func:`find_in` on a fresh capture of ``window`` (screen points), without the words a protected
    app's window shows. None when the screen couldn't be read (no engine, no Screen Recording, the
    window server can't say what's protected) — not a "not found": the caller keeps its own answer."""
    if not available():
        return None
    if bad := unreadable(label):
        return Missing(label, f"this Mac's text recognition can't read {bad}")
    if plan_for(label) is None:
        return None
    hide = protected_rects(window)
    if hide is None:
        return None
    px = grab(window)
    if px is None:
        return None
    try:
        return find_in(px, label, area=window, sheet=sheet, hide=hide)
    except (Failed, ValueError) as ex:
        log.info("couldn't read the screen: %s", ex)
        return None


async def find_text(label: str, window: Box, *, sheet: Box | None = None) -> Result | None:
    """Where ``label``'s words are shown in ``window`` (the front window's frame, screen points): Found,
    Ambiguous or Missing — or None when the screen couldn't be read. Off the event loop."""
    return await asyncio.to_thread(find_text_sync, label, window, sheet=sheet)


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


async def click_text_fallback(
    a: dict, c: Any, *, app: str, window: Box | None, els: list | None = None, pid: int = 0
) -> Any | None:
    """click_text when Accessibility names nothing with the label: read the front window's words and
    click the place showing them — through the same checks as any click (NEO's own window and protected
    apps refused, a window a sheet blocks refused, the owner app checked, the confirmation gate on the
    words read AND on what is under the point, the words of the dialog as its context). Returns the
    tool's result, or None when the screen couldn't be read (the handler keeps its own answer)."""
    from neo.agent.registry import ToolOutput, fence
    from neo.tools.computer import input as inp
    from neo.tools.computer import pointer

    g = _gates()
    label = str(a.get("label") or "")
    if window is None or not available():
        return None
    if not pid:
        from neo.tools.computer.focus import front_app

        front, front_pid = await asyncio.to_thread(front_app)
        pid = front_pid if front and app and front.lower() == app.lower() else 0
    sheet = sheet_frame(els)
    got = await find_text(label, window, sheet=sheet)
    if got is None:
        return None
    if isinstance(got, Missing):
        return ToolOutput(
            f"Error: nothing in the front window of {app or 'the front app'} is called '{label}' — neither "
            f"Accessibility nor the screen's words show it ({got.why}).",
            ok=False,
        )
    if isinstance(got, Ambiguous):
        marks = remember_marks([(f.text if f.exact else f.line, f.box, f.where) for f in got.options], app=app, pid=pid)
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
        if h.pid and pid and h.pid != pid:
            raise pointer.Refused(
                f"Error: another window ({h.app or 'another app'}) is over the words '{label}' in {app or 'the front app'} "
                "— look again (screenshot) before clicking.",
            )
        said = [_Said(found.context or found.line)] + list(els or [])  # (the dialog's words: what an "OK" does)
        if gate := await g._ask_first("click_text", a, found.line if not found.exact else found.text, app, said):
            return gate
        if not found.exact and (gate := await g._ask_first("click_text", a, found.text, app, said)):
            return gate
        if gate := await g._gate_click(a, [(h.label, h.app or app)]):
            return gate
        await pointer.bring(t, pts, want_pid=pid)
        if t.relabeled and (gate := await g._gate_click(a, [(t.hits[0].label, t.hits[0].app or app)])):
            return gate
    except pointer.Refused as r:
        return g._refused(c, r)
    h = t.hits[0]
    head = f"Clicked the words '{label}' at {_at(pts[0])}" + (f" in {h.app or app}" if (h.app or app) else "")
    if t.brought:
        head += f" (brought {t.brought} to the front first)"
    head += " — found by reading the screen (Accessibility has no such label)."
    before = await g._watch(c, pid or h.pid, pts, hover=pts[0])
    await asyncio.to_thread(inp.click, *pts[0])
    g._taint(c, "screen")
    read_there = f"read there: {_q(found.line)}" + (" [in the open sheet]" if found.where else "")
    if before is None:
        return ToolOutput(head + g._input_doubt() + "\n" + fence("screen", read_there))
    out = await pointer.outcome(before, pid or h.pid, pts, [_at(pts[0])])
    out = pointer.maybe_elsewhere(out, _at(pts[0]))
    text, ok = pointer.report(head, out, [pointer.words(h), read_there], g._input_doubt())
    return ToolOutput(text, ok=ok)
