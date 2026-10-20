"""The on-device pass over a Search Screen area: what NEO can tell about it without any brain, in a
few hundred milliseconds, and the actions it offers for it.

* text — Apple Vision's accurate text recognition, guessing the language, with its language
  correction on (ocr.Plan(prose=True): words a person wrote, not a toolbar's labels);
* language — NaturalLanguage's recogniser over that text (at least 8 characters, 3 for Chinese,
  Japanese and Korean; 60% sure), so "Translate to …" can offer the user's own language;
* links, e-mail addresses, phone numbers — Foundation's data detector over the text;
* QR codes and barcodes — Vision's barcode detector over the area itself;
* dark — at least 97% of the area nearly black: protected video, which captures as black. No brain
  is asked about such an area.

Which actions are offered is decided here, never by the page: only http(s) links open (at most 2048
characters; one with a user name or password in it opens without them, and says so; a host in
another script or punycode says so, showing both forms); ``javascript:``, ``file:``, ``data:`` and
any other scheme are offered to copy, never to open. An e-mail address must look like one, a phone
number is digits, spaces and a leading + (at most 20). A Wi-Fi code becomes its network's name and
"Copy Wi-Fi password". Each action's payload stays in the core (neo2.core.lens.live keeps it by id);
the page is shown a label and the full host, and sends back only the id.

Everything here runs on this Mac. Vision and NaturalLanguage are "model" effects (neo2.core.effects):
a test sets its own engine (:func:`set_engine`, and ocr.set_engine for the text).
"""

from __future__ import annotations

import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

from PIL import Image

from neo2.core import effects, logger

log = logger(__name__)

MAX_TEXT = 8000  # characters of the area's text the page gets
MAX_ACTIONS = 12
MAX_URL = 2048
DARK_LUMA, DARK_SHARE = 8, 0.97
LANG_MIN, LANG_MIN_CJK, LANG_CONF = 8, 3, 0.6
_NL = "/System/Library/Frameworks/NaturalLanguage.framework"
_VISION = "/System/Library/Frameworks/Vision.framework"


@dataclass
class Action:
    """Something the user can do with what the area holds. ``kind``: open | mail | call | copy_text |
    copy_image | wifi | code; ``value`` is what it opens or copies (kept in the core);
    ``detail``: what the page shows besides the label (a link's full host); ``warn``: "userinfo" |
    "punycode" | ""."""

    kind: str
    label: str
    value: str
    detail: str = ""
    warn: str = ""

    def shown(self, aid: str) -> dict[str, str]:
        out = {"id": aid, "kind": self.kind, "label": self.label}
        if self.detail:
            out["detail"] = self.detail
        if self.warn:
            out["warn"] = self.warn
        return out


@dataclass
class Local:
    """What the on-device pass found."""

    text: str = ""
    lines: int = 0
    lang: dict[str, Any] | None = None  # {code, name, conf}
    dark: bool = False
    actions: list[Action] = field(default_factory=list)
    ms: float = 0.0
    notes: list[str] = field(default_factory=list)  # parts that couldn't run (their names only)


# ---- the engines: Apple's, or a test's --------------------------------------------------------


class Engine(Protocol):
    def codes(self, img: Image.Image) -> list[tuple[str, str]]: ...  # (payload, symbology)

    def language(self, text: str) -> tuple[str, float] | None: ...  # (BCP-47 code, confidence)

    def detect(self, text: str) -> list[tuple[str, str]]: ...  # ("link" | "phone", value)


class _Apple:
    def __init__(self) -> None:
        self._nl: Any = None
        self._vn: dict[str, Any] | None = None
        self._lock = threading.Lock()

    def codes(self, img: Image.Image) -> list[tuple[str, str]]:
        effects.guard("model", "run Apple Vision's barcode reader for Search Screen")
        from neo2.core.decide._objc import pool
        from neo2.core.vision.apple import _cgimage

        with self._lock:
            if self._vn is None:
                try:
                    import Vision

                    self._vn = {k: getattr(Vision, k) for k in ("VNDetectBarcodesRequest", "VNImageRequestHandler")}
                except ImportError:
                    import objc

                    ns: dict[str, Any] = {}
                    objc.loadBundle("Vision", ns, bundle_path=_VISION)
                    self._vn = ns
        vn = self._vn
        with pool():
            cg, data = _cgimage(img)
            req = vn["VNDetectBarcodesRequest"].alloc().init()
            handler = vn["VNImageRequestHandler"].alloc().initWithCGImage_options_(cg, {})
            ok = handler.performRequests_error_([req], None)
            del data
            if not (ok[0] if isinstance(ok, tuple) else ok):
                return []
            out = []
            for o in req.results() or []:
                payload = o.payloadStringValue()
                if payload:
                    out.append((str(payload), str(o.symbology() or "").removeprefix("VNBarcodeSymbology")))
            return out

    def language(self, text: str) -> tuple[str, float] | None:
        effects.guard("model", "run NaturalLanguage's language recogniser for Search Screen")
        from neo2.core.decide._objc import pool

        with self._lock:
            if self._nl is None:
                import objc

                ns: dict[str, Any] = {}
                objc.loadBundle("NaturalLanguage", ns, bundle_path=_NL)
                self._nl = ns["NLLanguageRecognizer"]
        with pool():
            rec = self._nl.alloc().init()
            rec.processString_(text[:4000])
            hyp = rec.languageHypothesesWithMaximum_(1) or {}
            for code, conf in dict(hyp).items():
                return str(code), float(conf)
            return None

    def detect(self, text: str) -> list[tuple[str, str]]:
        return detect_with_foundation(text)


def detect_with_foundation(text: str) -> list[tuple[str, str]]:
    """Foundation's data detector: links (e-mail addresses come back as mailto: links) and phone
    numbers, in the order they appear. Plain computation on the text: nothing leaves the process."""
    try:
        import Foundation as F
    except ImportError:  # (no Foundation: the plain patterns)
        return detect_plain(text)
    from neo2.core.decide._objc import pool

    with pool():
        dd, _err = F.NSDataDetector.dataDetectorWithTypes_error_(F.NSTextCheckingTypeLink | F.NSTextCheckingTypePhoneNumber, None)
        if dd is None:
            return detect_plain(text)
        out: list[tuple[str, str]] = []
        for m in dd.matchesInString_options_range_(text, 0, (0, len(text))) or []:
            kind = int(m.resultType())
            if kind == int(F.NSTextCheckingTypeLink) and m.URL() is not None:
                out.append(("link", str(m.URL().absoluteString())))
            elif kind == int(F.NSTextCheckingTypePhoneNumber) and m.phoneNumber():
                out.append(("phone", str(m.phoneNumber())))
        return out


_PLAIN_URL = re.compile(r"\bhttps?://[^\s<>\"']{3,}", re.I)
_PLAIN_MAIL = re.compile(r"\b[\w.+-]{1,64}@[\w-]{1,63}(?:\.[\w-]{1,63})+\b")


def detect_plain(text: str) -> list[tuple[str, str]]:
    """The links and addresses a plain pattern finds (no phone numbers: too easy to get wrong)."""
    out = [("link", m.group(0).rstrip(".,;:!?)")) for m in _PLAIN_URL.finditer(text)]
    out += [("link", "mailto:" + m.group(0)) for m in _PLAIN_MAIL.finditer(text)]
    return out


_engine: Engine | None = None
_apple: _Apple | None = None


def set_engine(e: Engine | None) -> None:
    """Use ``e`` instead of Apple's frameworks (tests); None: Apple's again."""
    global _engine
    _engine = e


def engine() -> Engine:
    global _apple
    if _engine is not None:
        return _engine
    if _apple is None:
        _apple = _Apple()
    return _apple


# ---- what may be opened, and how it is shown ----------------------------------------------------

_MAIL = re.compile(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]{1,64}@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+")
_PHONE = re.compile(r"\+?[0-9][0-9 ]{2,19}")
_SCHEME = re.compile(r"^([a-z][a-z0-9+.-]{0,31}):", re.I)


def _scripts(s: str) -> set[str]:
    out = set()
    for ch in s:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            out.add(name.split()[0] if name else "?")
    return out


def _host_forms(host: str) -> tuple[str, str]:
    """(the host as written, its ASCII form): "bücher.de" → ("bücher.de", "xn--bcher-kva.de")."""
    try:
        ascii_ = host.encode("idna").decode("ascii")
    except UnicodeError:
        ascii_ = host
    try:
        shown = ascii_.encode("ascii").decode("idna") if "xn--" in ascii_ else host
    except UnicodeError:
        shown = host
    return shown, ascii_


def link(raw: str) -> Action | None:
    """The action for a link, an address or a code's text (None: nothing to offer)."""
    s = "".join(str(raw or "").split()) if str(raw or "").strip().lower().startswith(("http", "mailto", "tel")) else str(raw or "").strip()
    if not s:
        return None
    if s[:5].upper() == "WIFI:":
        return wifi(s)
    m = _SCHEME.match(s)
    scheme = m.group(1).lower() if m else ""
    if scheme in ("http", "https"):
        return _web(s)
    if scheme == "mailto":
        addr = s[7:].split("?", 1)[0]
        if _MAIL.fullmatch(addr):
            return Action("mail", f"Email {addr}"[:80], f"mailto:{addr}", detail=addr)
        return Action("code", "Copy address", s[:MAX_URL])
    if scheme == "tel":
        num = s[4:]
        if _PHONE.fullmatch(num):
            return Action("call", f"Call {num}"[:60], f"tel:{num.replace(' ', '')}", detail=num)
        return Action("code", "Copy number", s[:64])
    if scheme:  # javascript:, file:, data:, an app's own scheme: copied, never opened
        return Action("code", "Copy link" if "//" in s else "Copy code", s[:MAX_URL], detail=f"{scheme}:")
    return Action("code", "Copy code", s[:MAX_URL])


def _web(s: str) -> Action:
    if len(s) > MAX_URL:
        return Action("code", "Copy link", s[:MAX_URL], detail="too long to open")
    try:
        u = urlsplit(s)
        host = u.hostname or ""
    except ValueError:
        return Action("code", "Copy link", s)
    if not host or not re.fullmatch(r"[\w.-]+", host, re.UNICODE) or ".." in host:
        return Action("code", "Copy link", s)
    warn = ""
    netloc = u.netloc
    if u.username is not None or u.password is not None:
        netloc = netloc.rsplit("@", 1)[1]  # a user name or password in a link is how a fake host hides
        warn = "userinfo"
    shown, ascii_ = _host_forms(host)
    if "xn--" in ascii_ or any(ord(c) > 127 for c in host) or len(_scripts(shown)) > 1:
        warn = "punycode"
        detail = f"{shown} ({ascii_})" if shown != ascii_ else ascii_
        netloc = netloc.replace(host, ascii_) if host != ascii_ else netloc
    else:
        detail = host
    url = urlunsplit((u.scheme.lower(), netloc, u.path, u.query, u.fragment))
    label = f"Open {shown.removeprefix('www.')}"[:60]
    return Action("open", label, url, detail=detail, warn=warn)


def _unescape_wifi(v: str) -> str:
    return re.sub(r"\\(.)", r"\1", v)


def wifi(s: str) -> Action | None:
    """A Wi-Fi code (WIFI:T:WPA;S:<name>;P:<password>;;): "Copy Wi-Fi password" for its network."""
    fields: dict[str, str] = {}
    for part in re.findall(r"([A-Z]):((?:\\.|[^;\\])*)", s[5:], re.I):
        fields.setdefault(part[0].upper(), _unescape_wifi(part[1]))
    name, password = fields.get("S", ""), fields.get("P", "")
    if not password:
        return Action("code", "Copy network name" if name else "Copy code", name or s[:256], detail=name[:64])
    return Action("wifi", "Copy Wi-Fi password", password[:256], detail=name[:64])


def is_dark(img: Image.Image) -> bool:
    """At least DARK_SHARE of the area nearly black (protected video captures as black)."""
    g = img.convert("L")
    if max(g.size) > 256:
        g.thumbnail((256, 256))
    hist = g.histogram()
    total = sum(hist) or 1
    return sum(hist[:DARK_LUMA]) >= DARK_SHARE * total


def _cjk(text: str) -> bool:
    return any(unicodedata.name(c, "").startswith(("CJK", "HIRAGANA", "KATAKANA", "HANGUL")) for c in text)


def language(text: str) -> dict[str, Any] | None:
    """{code, name, conf} of the text's language, when sure enough (None: too short or unsure)."""
    t = " ".join(text.split())
    letters = sum(1 for c in t if c.isalpha())
    if letters < (LANG_MIN_CJK if _cjk(t) else LANG_MIN):
        return None
    got = engine().language(t)
    if not got:
        return None
    code, conf = got
    if conf < LANG_CONF or code in ("und", ""):
        return None
    return {"code": code, "name": language_name(code), "conf": round(conf, 2)}


# ---- languages ---------------------------------------------------------------------------------------

_NAMES = {
    "en": "English", "hi": "Hindi", "es": "Spanish", "fr": "French", "de": "German", "ja": "Japanese",
    "zh-Hans": "Chinese (Simplified)", "zh-Hant": "Chinese (Traditional)", "zh": "Chinese", "ar": "Arabic",
    "pt": "Portuguese", "ru": "Russian", "ko": "Korean", "it": "Italian",
}  # fmt: skip
COMMON = ("en", "hi", "es", "fr", "de", "ja", "zh-Hans", "ar", "pt", "ru", "ko", "it")
_LANG_CODE = re.compile(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8}){0,3}")


def valid_code(code: Any) -> bool:
    return isinstance(code, str) and len(code) <= 16 and bool(_LANG_CODE.fullmatch(code))


def base(code: str) -> str:
    c = str(code or "").replace("_", "-")
    return c if c.startswith(("zh-Hans", "zh-Hant")) else c.split("-")[0].lower()


def language_name(code: str) -> str:
    """A language's name in English ("hi" → "Hindi", "pt-BR" → "Portuguese (Brazil)")."""
    b = base(code)
    if b in _NAMES and (b == code or b.startswith("zh")):
        return _NAMES[b]
    try:
        from Foundation import NSLocale

        name = NSLocale.alloc().initWithLocaleIdentifier_("en").displayNameForKey_value_("kCFLocaleIdentifierKey", code)
        if name:
            return str(name)
    except Exception:  # noqa: BLE001 — no Foundation: the table, else the code
        pass
    return _NAMES.get(b, code)


def preferred() -> list[str]:
    """The Mac's preferred languages (the apps' UI follows them), read once (ocr keeps them)."""
    from neo2.core.tools.computer import ocr

    return [c for c in ocr._preferred_languages() if valid_code(c)]


def translate_to(lang: dict[str, Any] | None) -> str:
    """The language "Translate to …" offers first: the first preferred one whose base language isn't
    the text's ("" when the text is in all of them, or there is no language to go by)."""
    if not lang:
        return ""
    have = base(lang.get("code", ""))
    for c in [*preferred(), "en"]:
        if base(c) != have:
            return base(c) if base(c) in _NAMES else c
    return ""


def languages() -> list[dict[str, str]]:
    """The Translate menu: the preferred languages, then the common ones, each once."""
    seen: list[str] = []
    for c in [*(base(p) if base(p) in _NAMES else p for p in preferred()), *COMMON]:
        if c not in seen:
            seen.append(c)
    return [{"code": c, "name": language_name(c)} for c in seen[:24]]


# ---- the pass -------------------------------------------------------------------------------------


def _read(img_frame: Any) -> tuple[str, int]:
    from neo2.core.tools.computer import ocr

    if not ocr.available():
        return "", 0
    reading = ocr.read(img_frame, plan=ocr.Plan(fast=False, languages=(), auto=True, prose=True))
    lines = [ln.text for ln in reading.lines]
    return "\n".join(lines)[:MAX_TEXT], len(lines)


def analyse(crop: Any) -> Local:
    """The on-device pass over an area (a screen.Frame cut from the still). Parts that can't run are
    left out and named in ``notes``; a refused effect (a test that faked nothing) goes through."""
    t0 = time.perf_counter()
    out = Local()
    img = crop.image
    if is_dark(img):
        out.dark = True
        out.ms = (time.perf_counter() - t0) * 1000
        return out
    acts: list[Action] = []
    try:
        for payload, _sym in engine().codes(img)[:4]:
            a = link(payload)
            if a is not None:
                acts.append(a)
    except Exception as e:  # noqa: BLE001 — no codes read: the rest stands
        out.notes.append("codes")
        log.info(f"[lens] codes not read ({type(e).__name__})")
    try:
        out.text, out.lines = _read(crop)
    except Exception as e:  # noqa: BLE001
        out.notes.append("text")
        log.info(f"[lens] the area's text not read ({type(e).__name__})")
    if out.text:
        try:
            out.lang = language(out.text)
        except Exception as e:  # noqa: BLE001
            out.notes.append("language")
            log.info(f"[lens] the text's language not told ({type(e).__name__})")
        try:
            for kind, value in engine().detect(out.text):
                a = link(value) if kind == "link" else link("tel:" + "".join(c for c in value if c.isdigit() or c == "+"))
                if a is not None and a.kind != "code":
                    acts.append(a)
        except Exception as e:  # noqa: BLE001
            out.notes.append("links")
            log.info(f"[lens] the text's links not found ({type(e).__name__})")
        acts.insert(0, Action("copy_text", "Copy text", out.text))
    acts.append(Action("copy_image", "Copy image", ""))
    seen: set[tuple[str, str]] = set()
    for a in acts:
        if (a.kind, a.value) not in seen:
            seen.add((a.kind, a.value))
            out.actions.append(a)
        if len(out.actions) >= MAX_ACTIONS:
            break
    out.ms = (time.perf_counter() - t0) * 1000
    return out


__all__ = [
    "Action", "Local", "analyse", "detect_plain", "is_dark", "language", "language_name", "languages", "link",
    "preferred", "set_engine", "translate_to", "valid_code", "wifi",
]  # fmt: skip
