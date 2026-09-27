"""The Mac's own UI strings in English and Hindi: which English command a Hindi label is, and back.

macOS ships its interface text in string tables (``.loctable``: a binary plist of language → key →
string). Pairing each table's English entries with its Hindi ones gives an index made from the OS
alone, never from the user's data: "भेजें" is the OS's "Send", "नया गेम" an app's "New Game".

* **Built once per OS build**, at idle and off the event loop (:func:`idle_loop`, started by NEO.app),
  cached as derived OS data under the data dir, keyed by the OS build so an update rebuilds it. A
  table is read lazily - only its English and Hindi parts are decoded, never its other forty
  languages - so a full scan of the OS's ~5,000 tables takes about 3 s of CPU (a plain plist load
  of each takes about 11 s; loading every table twice through the frameworks' links, ~47 s).
  Library mode (NEO embedded in another app) never scans.
* **An app's own tables** (its ``.loctable`` and ``.lproj/*.strings``) are read on first contact
  and kept in RAM only: they aren't the OS's, so never written down. Only near the top of the
  bundle's Resources and within a bound (a big app's Resources hold tens of thousands of files); an
  app with none is remembered, so the next contact costs nothing. Callers ask for them only when
  there are words in another script to read (:func:`for_pid`).
* **The confirmation gate** reads it through ``neo2.core.security.policy``'s back-translation hook, which
  this module registers on import: a Hindi "भेजें" is gated as the "Send" it is. Until the index is
  built, :func:`ready` is False and :func:`readable` says no for a label in a script NEO can't read
  back into English: the paths that press things by themselves fail closed on it (they ask, or hand
  the choice to the model) instead of guessing. A text of several sentences is readable only when
  every one of them is (one known sentence says nothing about the next).

Screen text is only ever looked up here, never obeyed: a lookup returns the OS's strings.
"""

from __future__ import annotations

import asyncio
import json
import mmap
import os
import plistlib
import re
import struct
import sys
import threading
import time
import unicodedata
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from neo2.core import logger
from neo2.core.security import policy

log = logger(__name__)

__all__ = [
    "ROOTS",
    "LANGS",
    "Index",
    "norm",
    "words",
    "content_words",
    "non_latin",
    "ready",
    "english_sources",
    "exact_sources",
    "localized",
    "readable",
    "os_match",
    "for_bundle",
    "for_pid",
    "build",
    "load_or_build",
    "may_scan",
    "idle_loop",
]

# Where the OS keeps its interface text: the frameworks (AppKit's menus and alerts), the private
# frameworks, the apps that ship with macOS and their newer home in the app cryptex, and the core
# services (the desktop's own apps). Paths, never app names.
ROOTS: tuple[str, ...] = (
    "/System/Library/Frameworks",
    "/System/Library/PrivateFrameworks",
    "/System/Applications",
    "/System/Library/CoreServices",
    "/System/Cryptexes/App/System/Applications",
)
LANGS: tuple[str, ...] = ("hi",)
_EN_KEYS = ("en", "English", "en_US", "Base")
MAX_LEN = 100  # UI strings: a button, a menu item, an alert's sentence
IDLE_S = 30.0  # NEO idle this long before the index is loaded or built
_VERSION = 1
_MAX_TABLE = 64 << 20  # a table bigger than this isn't UI text
_APPS_KEPT = 8  # app bundles' tables kept in RAM
_APP_MAX_FILES = 400
_APP_MAX_BYTES = 32 << 20
_APP_MAX_DEPTH = 1  # an app's own UI tables sit in its Resources or one folder down, never deeper
_APP_MAX_ENTRIES = 3000  # files and folders looked at in one bundle's Resources, at most
_EMPTY_KEPT = 256  # app bundles remembered as having no tables to read
_SYSTEM_VERSION = "/System/Library/CoreServices/SystemVersion.plist"


# ---- words ------------------------------------------------------------------------------------------

_ELLIPSIS = re.compile(r"(?:…|\.\.\.)\s*$")
_DEV = "ऀ-ॿ"
_WORD = re.compile(rf"[\w{_DEV}]+")  # Devanagari's vowel signs aren't \w: a word runs through them
_PIECES = re.compile(r"(?<=[.!?।])\s+|\n+")


def norm(text: str) -> str:
    """How two UI strings are compared: NFC, casefolded, one space, no trailing "…" or full stop,
    and Devanagari's spelling variants folded (nukta dropped, chandrabindu as anusvara)."""
    s = unicodedata.normalize("NFC", text or "")
    s = s.replace("़", "").replace("ँ", "ं").replace("‌", "").replace("‍", "")
    s = " ".join(s.casefold().replace("’", "'").split()).strip(" \"“”'‘")
    return _ELLIPSIS.sub("", s).rstrip(" .:!?।").strip()


def words(text: str) -> list[str]:
    return _WORD.findall(norm(text))


# Hindi verb endings, particles and control nouns around a command's name ("नया गेम शुरू करो"
# names "नया गेम"), and English filler: what a request may add without naming anything else.
_HI_AUX = frozenset(
    {
        "करो", "करें", "करे", "कर", "करना", "कीजिए", "कीजिये", "दो", "दें", "दे", "दीजिए", "जाओ", "जाएं", "जाए",
        "लो", "लें", "ले", "है", "हैं", "को", "में", "पर", "से", "का", "की", "के", "यह", "ये", "इस", "वाला", "वाली",
        "वाले", "बटन", "दबाओ", "दबाएं", "दबाइए", "चुनो", "चुनें", "चुनिए", "खोलो", "खोलें", "दिखाओ", "दिखाएं", "ज़रा",
        "जरा", "कृपया", "मेनू", "मेन्यू", "क्लिक",
    }
)  # fmt: skip
_EN_STOP = frozenset({"a", "an", "the", "to", "of", "for", "in", "on", "and", "or", "please", "my", "this", "that", "it"})


def content_words(text: str) -> frozenset[str]:
    return frozenset(w for w in words(text) if w not in _HI_AUX and w not in _EN_STOP)


@lru_cache(maxsize=8192)
def _latin_letter(ch: str) -> bool:
    return unicodedata.name(ch, "").startswith("LATIN")


def non_latin(text: str) -> bool:
    """Does the text have letters of a script other than Latin (Devanagari, say)?"""
    if not text or text.isascii():
        return False
    return any(ch.isalpha() and not _latin_letter(ch) for ch in text)


# ---- the index ----------------------------------------------------------------------------------------


@dataclass
class Index:
    """English ↔ localized UI strings. ``back``: a localized string (normalized) → the English
    strings it translates; ``fwd``: an English string (normalized) → its localized forms."""

    build: str = ""
    langs: tuple[str, ...] = ()
    back: dict[str, tuple[str, ...]] = field(default_factory=dict)
    fwd: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def of(cls, pairs: dict[str, Iterable[str]], build: str = "", langs: Iterable[str] = LANGS) -> Index:
        back: dict[str, dict[str, None]] = {}
        fwd: dict[str, dict[str, None]] = {}
        for en, forms in pairs.items():
            ek = norm(en)
            if not ek:
                continue
            for loc in forms:
                lk = norm(loc)
                if not lk or lk == ek:
                    continue
                back.setdefault(lk, {})[en] = None
                fwd.setdefault(ek, {})[loc] = None
        return cls(
            build,
            tuple(langs),
            {k: tuple(v)[:16] for k, v in back.items()},
            {k: tuple(v)[:16] for k, v in fwd.items()},
        )

    def __len__(self) -> int:
        return len(self.fwd)


_system: Index | None = None  # the OS's index, once loaded or built
_apps: OrderedDict[str, Index] = OrderedDict()  # app bundle path → its own tables (RAM only)
_no_tables: OrderedDict[str, None] = OrderedDict()  # app bundles with nothing to read (not walked again)
_apps_lock = threading.Lock()


def ready() -> bool:
    """Is the OS's index loaded? (Until it is, non-Latin labels can't be read back.)"""
    return _system is not None


def install(index: Index | None) -> None:
    """Make ``index`` the OS's index (None: forget it)."""
    global _system
    _system = index


def _indexes() -> list[Index]:
    with _apps_lock:
        mine = list(reversed(_apps.values()))  # the app contacted last first
    return mine + ([_system] if _system is not None else [])


def exact_sources(text: str) -> tuple[str, ...]:
    """The English strings the OS (or an app contacted lately) translates as ``text``, exactly as it
    is: never pieced together from some of its sentences. () for Latin-script text."""
    if not text or not non_latin(text):
        return ()
    key = norm(text)
    return tuple(dict.fromkeys(s for ix in _indexes() for s in ix.back.get(key, ())))[:24]


def english_sources(text: str) -> tuple[str, ...]:
    """The English strings the OS (or an app contacted lately) translates as ``text`` - a label, or
    a dialog's text, looked up whole and then sentence by sentence. () for Latin-script text."""
    if not text or not non_latin(text):
        return ()
    found: list[str] = []
    ixs = _indexes()
    key = norm(text)
    for ix in ixs:
        found.extend(ix.back.get(key, ()))
    if not found:
        pieces = [p for p in _PIECES.split(text) if p.strip()][:20]
        if len(pieces) > 1:
            for p in pieces:
                k = norm(p)
                for ix in ixs:
                    found.extend(ix.back.get(k, ()))
    return tuple(dict.fromkeys(found))[:24]


def localized(english: str) -> tuple[str, ...]:
    """The OS's (and recently contacted apps') localized forms of an English UI string."""
    key = norm(english)
    out: list[str] = []
    for ix in _indexes():
        out.extend(ix.fwd.get(key, ()))
    return tuple(dict.fromkeys(out))[:24]


def readable(text: str) -> bool:
    """Can NEO read this label or text in English - it's in Latin script, or the OS says what it is
    (the whole of it, or every sentence of it)? A path that presses something by itself asks instead
    when it can't (fail closed)."""
    return not policy.unread(text, lookup=exact_sources)


def os_match(request: str, label: str) -> bool:
    """The OS-strings stage: the label, in the request's other language, with every content word of
    it in the request ("नया गेम शुरू करो" → "New Game"; "save karo" → "सहेजें" on a Hindi UI)."""
    want = content_words(request)
    if not want or non_latin(request) == non_latin(label):
        return False
    forms = localized(label) if non_latin(request) else english_sources(label)
    for f in forms:
        have = content_words(f)
        if have and have <= want:
            return True
    return False


# The gate reads localized labels through this (neo2.core.security.policy.commits): data handed over.
policy.use_back_translations(english_sources, whole=exact_sources)


# ---- reading string tables ---------------------------------------------------------------------------


class _BPlist:
    """Just enough of a binary property list to read part of a big one: an object is decoded only
    when asked for, by its reference, so a table's other languages are never touched."""

    def __init__(self, buf: memoryview) -> None:
        if len(buf) < 40 or bytes(buf[:8]) != b"bplist00":
            raise ValueError("not a binary plist")
        osz, rsz, n, top, table = struct.unpack(">6xBBQQQ", bytes(buf[-32:]))
        end = len(buf) - 32
        if not (1 <= osz <= 8 and 1 <= rsz <= 8) or n == 0 or top >= n or table + n * osz > end:
            raise ValueError("a broken trailer")
        self.b, self.osz, self.rsz, self.n, self.top, self.table, self.end = buf, osz, rsz, n, top, table, end

    def _int(self, pos: int, size: int) -> int:
        if pos < 0 or pos + size > self.end:
            raise ValueError("out of bounds")
        return int.from_bytes(self.b[pos : pos + size], "big")

    def _at(self, ref: int) -> int:
        if not 0 <= ref < self.n:
            raise ValueError("a bad reference")
        return self._int(self.table + ref * self.osz, self.osz)

    def _count(self, pos: int, info: int) -> tuple[int, int]:
        if info != 0xF:
            return pos + 1, info
        marker = self._int(pos + 1, 1)
        if marker >> 4 != 1:
            raise ValueError("a bad length")
        size = 1 << (marker & 0xF)
        return pos + 2 + size, self._int(pos + 2, size)

    def string(self, ref: int) -> str | None:
        pos = self._at(ref)
        m = self._int(pos, 1)
        kind, info = m >> 4, m & 0xF
        if kind not in (5, 6, 7):
            return None
        start, n = self._count(pos, info)
        size = n * 2 if kind == 6 else n
        if start + size > self.end:
            raise ValueError("out of bounds")
        raw = bytes(self.b[start : start + size])
        return raw.decode({5: "ascii", 6: "utf-16-be", 7: "utf-8"}[kind], "replace")

    def dict_refs(self, ref: int) -> list[tuple[int, int]] | None:
        pos = self._at(ref)
        m = self._int(pos, 1)
        if m >> 4 != 0xD:
            return None
        start, n = self._count(pos, m & 0xF)
        if start + 2 * n * self.rsz > self.end:
            raise ValueError("out of bounds")
        r = self.rsz
        return [(self._int(start + i * r, r), self._int(start + (n + i) * r, r)) for i in range(n)]

    def table_at(self, ref: int) -> dict[str, str]:
        """A {key: string} dict (entries that aren't plain strings - plural rules - left out)."""
        out: dict[str, str] = {}
        for k, v in self.dict_refs(ref) or ():
            key = self.string(k)
            if key is None:
                continue
            val = self.string(v)
            if val is not None:
                out[key] = val
        return out


def _table_langs(path: str, langs: Iterable[str]) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """(English, {lang: localized}) of one .loctable, decoding nothing else."""
    with open(path, "rb") as fh:
        size = os.fstat(fh.fileno()).st_size
        if size < 40 or size > _MAX_TABLE:
            return {}, {}
        with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            view = memoryview(mm)
            p: _BPlist | None = None
            try:
                p = _BPlist(view)
                tops = {p.string(k): v for k, v in p.dict_refs(p.top) or ()}
                en_ref = next((tops[k] for k in _EN_KEYS if k in tops), None)
                want = {lang: tops[lang] for lang in langs if lang in tops}
                if en_ref is None or not want:
                    return {}, {}
                return p.table_at(en_ref), {lang: p.table_at(ref) for lang, ref in want.items()}
            finally:
                p = None  # (nothing may hold the view when the map closes)
                try:
                    view.release()
                except BufferError:  # pragma: no cover - a slice still alive: the map's close waits for it
                    pass


def _usable(en: str, loc: str, max_len: int) -> bool:
    return (
        0 < len(en) <= max_len
        and 0 < len(loc) <= 3 * max_len
        and "%" not in en
        and "%" not in loc
        and en != loc
        and any(ch.isalpha() for ch in en)
    )


def _add(pairs: dict[str, set[str]], en_tab: dict[str, str], loc_tabs: dict[str, dict[str, str]], max_len: int) -> None:
    for tab in loc_tabs.values():
        for key, en in en_tab.items():
            loc = tab.get(key)
            if loc is None:
                continue
            en, loc = en.strip(), loc.strip()
            if _usable(en, loc, max_len):
                pairs.setdefault(en, set()).add(loc)


def _tables(
    roots: Iterable[str], suffix: str = ".loctable", *, max_depth: int | None = None, max_entries: int | None = None
) -> Iterator[str]:
    """Every string table under the roots, once each (links are not followed: a framework's
    Versions/Current is the same files again). ``max_depth``: folders below a root walked into;
    ``max_entries``: files and folders looked at in all, after which the walk stops."""
    seen: set[tuple[int, int]] = set()
    looked = 0
    for root in roots:
        base = root.rstrip(os.sep).count(os.sep)
        for d, dirs, files in os.walk(root):
            looked += len(files) + len(dirs)
            if max_entries is not None and looked > max_entries:
                return
            if max_depth is not None and d.rstrip(os.sep).count(os.sep) - base >= max_depth:
                dirs[:] = []  # (pruned: os.walk goes no deeper)
            for f in files:
                if not f.endswith(suffix):
                    continue
                path = os.path.join(d, f)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                if (st.st_dev, st.st_ino) in seen:
                    continue
                seen.add((st.st_dev, st.st_ino))
                yield path


class Stopped(Exception):
    """The build was told to stop (NEO is shutting down)."""


def build(
    roots: Iterable[str] = ROOTS,
    langs: Iterable[str] = LANGS,
    *,
    max_len: int = MAX_LEN,
    stop: threading.Event | None = None,
    busy: Callable[[], bool] | None = None,
) -> dict[str, set[str]]:
    """{English: its localized forms} from every string table under ``roots``. ``busy``: while it
    says yes (NEO is handling a request) the scan waits; ``stop`` ends it (:class:`Stopped`)."""
    langs = tuple(langs)
    pairs: dict[str, set[str]] = {}
    for path in _tables(roots):
        if stop is not None and stop.is_set():
            raise Stopped
        while busy is not None and busy():
            if stop is not None and stop.wait(0.5):
                raise Stopped
            if stop is None:
                time.sleep(0.5)
        try:
            en_tab, loc_tabs = _table_langs(path, langs)
        except (OSError, ValueError, UnicodeError, struct.error, KeyError, BufferError):
            continue
        _add(pairs, en_tab, loc_tabs, max_len)
    return pairs


# ---- an app's own tables (RAM only) -------------------------------------------------------------------

_STRINGS_ENTRY = re.compile(r'"((?:[^"\\]|\\.)*)"\s*=\s*"((?:[^"\\]|\\.)*)"\s*;')
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\"}


def _unescape(s: str) -> str:
    return re.sub(r"\\(.)", lambda m: _ESCAPES.get(m.group(1), m.group(1)), s)


def _read_strings(path: str) -> dict[str, str]:
    """A .strings file: a plist (binary or XML), or the old "key" = "value"; text."""
    with open(path, "rb") as fh:
        raw = fh.read(4 << 20)
    if raw.startswith(b"bplist00") or raw.lstrip().startswith(b"<?xml"):
        try:
            v = plistlib.loads(raw)
        except Exception:  # noqa: BLE001
            return {}
        return {k: s for k, s in v.items() if isinstance(k, str) and isinstance(s, str)} if isinstance(v, dict) else {}
    try:
        utf16 = raw[:2] in (b"\xff\xfe", b"\xfe\xff")
        text = raw.decode("utf-16") if utf16 else raw.decode("utf-8-sig", "replace")
    except UnicodeDecodeError:
        return {}
    return {_unescape(k): _unescape(v) for k, v in _STRINGS_ENTRY.findall(text)}


def _bundle_pairs(path: str, langs: tuple[str, ...], max_len: int = MAX_LEN) -> dict[str, set[str]]:
    res = Path(path) / "Contents" / "Resources"
    pairs: dict[str, set[str]] = {}
    if not res.is_dir():
        return pairs
    files, budget = 0, _APP_MAX_BYTES
    for table in _tables([str(res)], max_depth=_APP_MAX_DEPTH, max_entries=_APP_MAX_ENTRIES):
        files += 1
        if files > _APP_MAX_FILES or budget <= 0:
            return pairs
        try:
            budget -= os.path.getsize(table)
            en_tab, loc_tabs = _table_langs(table, langs)
        except (OSError, ValueError, UnicodeError, struct.error, KeyError, BufferError):
            continue
        _add(pairs, en_tab, loc_tabs, max_len)
    en_dir = next((res / f"{k}.lproj" for k in _EN_KEYS if (res / f"{k}.lproj").is_dir()), None)
    if en_dir is None:
        return pairs
    for lang in langs:
        loc_dir = res / f"{lang}.lproj"
        if not loc_dir.is_dir():
            continue
        for f in sorted(loc_dir.glob("*.strings"))[:_APP_MAX_FILES]:
            src = en_dir / f.name
            if not src.is_file():
                continue
            try:
                _add(pairs, _read_strings(str(src)), {lang: _read_strings(str(f))}, max_len)
            except OSError:
                continue
    return pairs


def for_bundle(path: str) -> Index | None:
    """The app bundle's own UI strings (read once, kept in RAM among the last few apps; a bundle
    with none is remembered as such, so it is never walked again)."""
    if not path:
        return None
    key = os.path.realpath(path)
    with _apps_lock:
        if key in _apps:
            _apps.move_to_end(key)
            return _apps[key]
        if key in _no_tables:
            _no_tables.move_to_end(key)
            return Index(langs=LANGS)
    try:
        ix = Index.of(_bundle_pairs(key, LANGS), langs=LANGS)
    except OSError:
        return None
    with _apps_lock:
        _apps[key] = ix
        while len(_apps) > _APPS_KEPT:
            _apps.popitem(last=False)
        if not len(ix):
            _no_tables[key] = None
            while len(_no_tables) > _EMPTY_KEPT:
                _no_tables.popitem(last=False)
    return ix


def _bundle_path(pid: int) -> str:
    try:
        from AppKit import NSRunningApplication

        app = NSRunningApplication.runningApplicationWithProcessIdentifier_(int(pid))
        url = app.bundleURL() if app is not None else None
        return str(url.path()) if url is not None else ""
    except Exception:  # noqa: BLE001
        return ""


def for_pid(pid: int) -> Index | None:
    """The running app's own UI strings, on first contact (a few ms for most apps, and bounded for
    a big one; later from RAM). Ask for them only when there are words in another script to read."""
    return for_bundle(_bundle_path(pid)) if pid else None


# ---- per OS build: the cache, and building at idle ---------------------------------------------------


def os_build() -> str:
    """This Mac's OS version and build ("26.6.2-25G83"), the key of the cache; "" if unknown."""
    try:
        with open(_SYSTEM_VERSION, "rb") as fh:
            v = plistlib.load(fh)
    except (OSError, ValueError, plistlib.InvalidFileException):
        return ""
    got = "-".join(str(v.get(k) or "") for k in ("ProductVersion", "ProductBuildVersion")).strip("-")
    return re.sub(r"[^\w.-]", "_", got)[:40]


def cache_path(build_id: str) -> Path:
    from neo2.core.config import settings

    return settings().data_dir / "cache" / f"uistrings-{build_id}.json"


def may_scan() -> bool:
    """Only NEO.app scans the OS's tables: never an agent embedded in another app (library mode)."""
    from neo2.core import runtime

    return sys.platform == "darwin" and not runtime.library_mode() and not runtime.library_bound()


def _load(p: Path, build_id: str) -> Index | None:
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("v") != _VERSION or data.get("build") != build_id:
        return None
    if list(data.get("langs") or []) != list(LANGS) or not isinstance(data.get("pairs"), dict):
        return None
    pairs = {k: v for k, v in data["pairs"].items() if isinstance(k, str) and isinstance(v, list)}
    return Index.of(pairs, build_id, LANGS)


def _save(p: Path, build_id: str, pairs: dict[str, set[str]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    body = {"v": _VERSION, "build": build_id, "langs": list(LANGS), "pairs": {k: sorted(v) for k, v in pairs.items()}}
    tmp = p.with_name(p.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(body, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, p)
    for old in p.parent.glob("uistrings-*.json"):  # an earlier OS build's
        if old != p:
            old.unlink(missing_ok=True)


def load_or_build(*, busy: Callable[[], bool] | None = None, stop: threading.Event | None = None) -> bool:
    """This OS build's index: from its cache, else built from the OS's tables and cached. False in
    library mode (nothing is scanned there)."""
    if not may_scan():
        return False
    bid = os_build()
    p = cache_path(bid) if bid else None
    if p is not None and p.exists() and (ix := _load(p, bid)) is not None:
        install(ix)
        return True
    t0 = time.process_time()
    pairs = build(ROOTS, LANGS, busy=busy, stop=stop)
    if p is not None:
        try:
            _save(p, bid, pairs)
        except OSError as e:
            log.warning(f"[uistrings] cache not written ({str(e)[:80]})")
    install(Index.of(pairs, bid, LANGS))
    log.info(f"[uistrings] {len(pairs)} OS strings indexed in {time.process_time() - t0:.1f}s of CPU")
    return True


async def _in_daemon(fn: Callable[[], bool]) -> bool:
    """Run ``fn`` on a daemon thread (a build never holds up NEO's exit), awaiting its answer."""
    loop = asyncio.get_running_loop()
    fut: asyncio.Future[bool] = loop.create_future()

    def run() -> None:
        try:
            res: bool | BaseException = fn()
        except BaseException as e:  # noqa: BLE001 - handed to the awaiting task
            res = e

        def settle() -> None:
            if fut.done():
                return
            if isinstance(res, BaseException):
                fut.set_exception(res)
            else:
                fut.set_result(res)

        try:
            loop.call_soon_threadsafe(settle)
        except RuntimeError:  # the loop is gone (NEO quit meanwhile)
            pass

    threading.Thread(target=run, daemon=True, name="neo-uistrings").start()
    return await fut


async def idle_loop(
    is_idle: Callable[[], bool], *, idle_s: float = IDLE_S, poll_s: float = 15.0, retry_s: float = 3600.0
) -> None:
    """NEO.app's background task: once NEO has been idle ``idle_s``, load this OS build's index
    from its cache, or build it (paused while NEO is busy). Returns at once in library mode."""
    if not may_scan():
        return
    since: float | None = None
    while not ready():
        await asyncio.sleep(poll_s)
        if not may_scan():
            return
        if not is_idle():
            since = None
            continue
        now = time.monotonic()
        since = since or now
        if now - since < idle_s:
            continue
        try:
            ok = await _in_daemon(lambda: load_or_build(busy=lambda: not is_idle()))
        except Exception as e:  # noqa: BLE001 - an index NEO can't build is only a missed check
            log.warning(f"[uistrings] not built ({type(e).__name__}: {str(e)[:80]})")
            ok = False
        if not ok:
            since = None
            await asyncio.sleep(retry_s)


# ---- measuring (python -m neo2.core.decide.uistrings) --------------------------------------------------------


def bench(tables: int = 60, keys: int = 400, langs: int = 40, root: str | None = None) -> dict[str, float]:
    """Hermetic: synthetic string tables (``langs`` languages each, like the OS's) read with the
    lazy reader and with a plain plist load; seconds of CPU for each, and that both agree."""
    import random
    import tempfile

    rnd = random.Random(7)
    codes = ["en", "hi"] + [f"x{i:02d}" for i in range(langs - 2)]
    with tempfile.TemporaryDirectory(dir=root) as d:
        for t in range(tables):
            body = {c: {f"k{t}_{i}": f"{c} string {rnd.random():.6f} number {i}" for i in range(keys)} for c in codes}
            with open(os.path.join(d, f"t{t}.loctable"), "wb") as fh:
                plistlib.dump(body, fh, fmt=plistlib.FMT_BINARY)
        t0 = time.process_time()
        lazy = build([d], ("hi",))
        t_lazy = time.process_time() - t0
        t0 = time.process_time()
        full: dict[str, set[str]] = {}
        for path in _tables([d]):
            with open(path, "rb") as fh:
                tab = plistlib.load(fh)
            _add(full, tab.get("en") or {}, {"hi": tab.get("hi") or {}}, MAX_LEN)
        t_full = time.process_time() - t0
    return {"tables": tables, "lazy_cpu_s": round(t_lazy, 4), "plistlib_cpu_s": round(t_full, 4),
            "speedup": round(t_full / max(t_lazy, 1e-9), 1), "same": float(lazy == full)}  # fmt: skip


if __name__ == "__main__":  # pragma: no cover - a measuring aid
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench", action="store_true", help="synthetic tables: lazy reader vs plistlib")
    ap.add_argument("--scan", action="store_true", help="read this Mac's own tables (read-only; nothing written)")
    args = ap.parse_args()
    if args.bench:
        print(json.dumps(bench(), indent=1))
    if args.scan:
        c0, w0 = time.process_time(), time.time()
        got = build()
        print(json.dumps({"english_strings": len(got), "forms": sum(len(v) for v in got.values()),
                          "cpu_s": round(time.process_time() - c0, 2), "wall_s": round(time.time() - w0, 2)}))  # fmt: skip
