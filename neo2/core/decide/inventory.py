"""The apps on THIS Mac: what each declares it can do, and how much this user uses it.

Read-only and app-agnostic. Everything is derived from
  * what bundles declare about themselves (Info.plist, scripting dictionary, linked frameworks),
  * what LaunchServices knows (default handlers for URL schemes and content types),
  * what Spotlight records about use (kMDItemLastUsedDate / kMDItemUseCount / kMDItemUsedDates),
  * what is running right now (the process table) and what is in the Dock.
Nothing here launches, activates or sends Apple events to any app, and no app name, bundle id or
website appears in the logic.

Costs: the static part of each app (plist, sdef, Mach-O links) is cached in-process keyed by the
bundle path + its Info.plist mtime/size; the app list is re-enumerated when an app folder changes
(or every ``_ENUM_TTL``); Spotlight usage is re-read for running apps on every call and for the
rest every ``_USAGE_TTL``. A warm ``inventory()`` is a few milliseconds.
"""

from __future__ import annotations

import contextlib
import ctypes
import math
import os
import plistlib
import re
import stat
import struct
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property, lru_cache

from neo2.core.decide import evidence, sdef
from neo2.core.decide._objc import pooled

__all__ = [
    "App",
    "inventory",
    "snapshot",
    "app_by_id",
    "app_for_bundle",
    "app_for_pid",
    "named_in",
    "named_span",
    "name_forms",
    "default_for",
    "affinity",
    "own_schemes",
    "STD_SCHEMES",
]

HOME = os.path.expanduser("~")
# Folders walked (one level of sub-folders too) in addition to Spotlight, which misses some apps.
APP_DIRS: tuple[str, ...] = (
    "/Applications",
    "/System/Applications",
    "/System/Applications/Utilities",
    "/System/Volumes/Preboot/Cryptexes/App/System/Applications",
    os.path.join(HOME, "Applications"),  # per-user installs, browser web-app shims
    "/System/Library/CoreServices/Applications",
)
# OS-bundled apps that sit next to agents: only the ones declaring an app category are user-facing.
SERVICES_DIR = "/System/Library/CoreServices"
_DOCK_DOMAIN = "com.apple.dock"  # the Dock's own preferences (which tiles the user pinned)
_WORDS_FILE = "/usr/share/dict/words"  # system word list: tells common words from coined names
_CONNECTIVES_FILE = "/usr/share/dict/connectives"

# Standard transport/document schemes: protocols, not apps.
STD_SCHEMES = frozenset({
    "http", "https", "file", "ftp", "ftps", "sftp", "smb", "afp", "nfs", "mailto", "tel", "sms", "facetime",
    "facetime-audio", "data", "about", "blob", "ws", "wss", "ssh", "vnc", "rtsp", "rtsps", "rtmp", "rtmps",
    "rtmpe", "rtmpt", "mms", "mmsh", "udp", "rtp", "webcal", "feed", "news", "irc", "ircs", "git", "telnet",
    "x-apple-helpviewer",
})  # fmt: skip
# Standard schemes whose default handler is worth knowing (App.handles), in a stable order.
HANDLED_SCHEMES = (
    "https", "http", "mailto", "sms", "tel", "facetime", "facetime-audio", "webcal", "feed", "ftp", "sftp",
    "ssh", "vnc", "smb", "afp", "rtsp", "news", "irc",
)  # fmt: skip
# Tokens of bundle ids / names that say nothing about which app it is.
_GENERIC_TOKENS = frozenset({
    "com", "org", "net", "io", "co", "us", "uk", "de", "ai", "app", "apps", "application", "mac", "macos", "osx",
    "desktop", "client", "the", "inc", "ltd", "llc", "gmbh", "www", "dev", "beta", "helper", "for", "and",
    "pro", "free", "lite", "plus", "x", "id", "launcher",
})  # fmt: skip
_ENUM_TTL = 600.0  # re-run the Spotlight/LaunchServices enumeration at least this often
_USAGE_TTL = 60.0  # Spotlight usage of apps that are not running
_CTX_TTL = 30.0  # Dock + default handlers
_MEDIA_WORDS = {  # generic content nouns (not app names), used only to guess what kind of media
    "audio": ("music", "audio", "song", "songs", "radio", "sound", "fm", "amp"),
    "podcast": ("podcast", "pcast"),
    "video": ("video", "movie", "tv", "film", "tube", "player"),
}


@dataclass(frozen=True)
class App:
    bundle_id: str
    name: str
    path: str
    category: str = ""  # LSApplicationCategoryType (else Spotlight App Store category), e.g. 'music'
    viewer_types: tuple[str, ...] = ()  # UTIs / extensions it opens as Viewer or Shell
    editor_types: tuple[str, ...] = ()
    schemes: tuple[str, ...] = ()  # every URL scheme it declares, lower-case
    own_schemes: tuple[str, ...] = ()  # non-standard schemes spelled like the app (deep links)
    handles: tuple[str, ...] = ()  # standard schemes it is the DEFAULT handler for ('https', 'mailto')
    script: sdef.Dictionary | None = None
    media_evidence: tuple[str, ...] = ()  # human-readable; empty when nothing says "media"
    web_start_url: str = ""  # web-app wrapper start URL
    last_used: float | None = None  # epoch (Spotlight kMDItemLastUsedDate)
    use_count: int = 0  # kMDItemUseCount (0 if unknown)
    days_used_30: int = 0  # distinct days used in the last 30 (kMDItemUsedDates)
    running: bool = False
    in_dock: bool = False
    aliases: tuple[str, ...] = ()  # extra: other names it goes by (bundle name, Finder name)
    # declared by the bundle beyond its dictionary (see neo2.core.decide.evidence)
    intents: tuple[str, ...] = ()  # SiriKit intents its extensions handle
    ai_schemas: tuple[str, ...] = ()  # App Intents: 'schema:<domain>.<Name>', 'protocol:<Name>'
    share_text: bool = False  # a share extension that accepts text
    services: tuple[tuple[str, tuple[str, ...]], ...] = ()  # Services menu: (title, send types)
    usage: frozenset[str] = frozenset()  # NS...UsageDescription keys it declares
    doc_based: bool = False  # its document types name a document class
    kind: str = "app"  # "route": a synthetic destination (an OS store), not an installed app
    note: str = ""  # a route's evidence (counts only)

    @cached_property
    def caps(self) -> frozenset[str]:
        """Generic capability tags from its scripting dictionary (see ``sdef.caps``)."""
        return sdef.caps(self.script)


# ============================================================================== bundle reading
def _read_info(app: str, must_have: bytes = b"") -> tuple[dict, str]:
    """(Info.plist, the directory holding it): a Mac app's, or the wrapped iPhone/iPad app's.
    ``must_have``: skip parsing a plist that doesn't even contain this key (cheap pre-check)."""
    for d in (os.path.join(app, "Contents"), os.path.join(app, "WrappedBundle")):
        try:
            p = os.path.join(d, "Info.plist")
            if not stat.S_ISREG(os.stat(p).st_mode):  # never block on a pipe dressed as a plist
                continue
            with open(p, "rb") as f:
                raw = f.read(4 << 20)
            if must_have and must_have not in raw:
                return {}, d
            info = plistlib.loads(raw)
            if isinstance(info, dict):
                return info, d
        except (OSError, plistlib.InvalidFileException, ValueError, TypeError, OverflowError):
            continue
    return {}, ""


def _truthy(v) -> bool:
    return v in (True, 1, "1", "YES", "Yes", "yes", "true", "True", "TRUE")


_UTTYPE = None


def _uttype():
    global _UTTYPE
    if _UTTYPE is None:
        import objc
        from AppKit import NSWorkspace  # noqa: F401 - loads UniformTypeIdentifiers

        _UTTYPE = objc.lookUpClass("UTType")
    return _UTTYPE


@dataclass(frozen=True)
class _Doc:
    viewer: tuple[str, ...]
    editor: tuple[str, ...]
    av_view: int
    av_edit: int
    owns_project: bool
    html: bool


def _doc_roles(info: dict) -> _Doc:
    UT = _uttype()
    av, html_t = UT.typeWithIdentifier_("public.audiovisual-content"), UT.typeWithIdentifier_("public.html")
    viewer: dict[str, None] = {}
    editor: dict[str, None] = {}
    av_view = av_edit = 0
    owns = html = False
    seen_av: set[tuple[str, str]] = set()
    for d in info.get("CFBundleDocumentTypes") or []:
        if not isinstance(d, dict):
            continue
        role = str(d.get("CFBundleTypeRole") or "Viewer")
        utis = [str(u) for u in d.get("LSItemContentTypes") or [] if isinstance(u, str)]
        exts = [
            str(e).lower() for e in d.get("CFBundleTypeExtensions") or [] if isinstance(e, str) and e != "*"
        ]
        bucket = editor if role == "Editor" else viewer if role in ("Viewer", "Shell") else None
        if bucket is not None:
            bucket.update(dict.fromkeys(utis + exts))
        types = [(u, UT.typeWithIdentifier_(u)) for u in utis] + [
            (e, UT.typeWithFilenameExtension_(e)) for e in exts
        ]
        is_av = False
        for label, t in types:
            if t is None:
                continue
            if t.conformsToType_(html_t):
                html = True
            if t.conformsToType_(av) and (role, label) not in seen_av:
                seen_av.add((role, label))
                is_av = True
                if role == "Editor":
                    av_edit += 1
                elif role in ("Viewer", "Shell"):
                    av_view += 1
        if d.get("LSHandlerRank") == "Owner" and role == "Editor" and not is_av:
            owns = True  # its own project/document format (a creator/editor app)
    return _Doc(tuple(viewer), tuple(editor), av_view, av_edit, owns, html)


def own_schemes(schemes, bundle_id: str, name: str) -> tuple[str, ...]:
    """Schemes that are this app's own deep links: not a standard protocol, spelled like the app's
    bundle id or name ('music'/'musics' for an app called Music, 'com.vendor.tv' for 'TV')."""
    key = re.sub(r"[^a-z0-9]", "", f"{bundle_id}{name}".lower())
    toks = {
        t for t in re.split(r"[^a-z0-9]+", f"{bundle_id} {name}".lower()) if t and t not in _GENERIC_TOKENS
    }
    out = []
    for s in schemes:
        s = s.lower()
        if s in STD_SCHEMES:
            continue
        if s == bundle_id.lower():
            out.append(s)
            continue
        if "." in s:  # reverse-DNS schemes are only the app's own when they ARE its id (not msauth.<id>)
            continue
        stem = re.sub(r"s$", "", s.split("-")[0])
        if stem in toks or (len(stem) >= 3 and stem not in _GENERIC_TOKENS and stem in key):
            out.append(s)
    return tuple(out)


_LC_DYLIB = {0xC, 0x80000018, 0x8000001F, 0x20, 0x80000023}


def _linked_frameworks(exe: str, limit: int = 64 * 1024) -> frozenset[str]:
    """Framework names the main executable links (pure-Python Mach-O read; otool may not exist)."""
    names: set[str] = set()
    try:
        with open(exe, "rb") as f:
            head = f.read(8)
            if len(head) < 8:
                return frozenset()
            magic = struct.unpack(">I", head[:4])[0]
            off = 0
            if magic in (0xCAFEBABE, 0xCAFEBABF):  # fat: prefer the arm64 slice
                n = min(struct.unpack(">I", head[4:8])[0], 8)
                size = 20 if magic == 0xCAFEBABE else 32
                slices = []
                for _ in range(n):
                    a = f.read(size)
                    cpu = struct.unpack(">I", a[:4])[0]
                    o = (
                        struct.unpack(">I", a[8:12])[0]
                        if magic == 0xCAFEBABE
                        else struct.unpack(">Q", a[8:16])[0]
                    )
                    slices.append((cpu, o))
                off = next((o for cpu, o in slices if cpu == 0x0100000C), slices[0][1] if slices else 0)
            f.seek(off)
            hdr = f.read(32)
            m = struct.unpack("<I", hdr[:4])[0]
            if m not in (0xFEEDFACF, 0xFEEDFACE):
                return frozenset()
            ncmds, sizeofcmds = struct.unpack("<II", hdr[16:24])
            f.seek(off + (32 if m == 0xFEEDFACF else 28))
            blob = f.read(min(sizeofcmds, limit))
        i = 0
        for _ in range(ncmds):
            if i + 12 > len(blob):
                break
            cmd, size = struct.unpack("<II", blob[i : i + 8])
            if cmd in _LC_DYLIB:
                nameoff = struct.unpack("<I", blob[i + 8 : i + 12])[0]
                raw = blob[i + nameoff : i + size].split(b"\0")[0].decode(errors="ignore")
                mfw = re.search(r"/([^/]+)\.framework/", raw)
                if mfw:
                    names.add(mfw.group(1))
            i += size or 8
    except (OSError, struct.error, IndexError):
        pass
    return frozenset(names)


_WEB_URL_SKIP = re.compile(
    r"update|feed|support|help|privacy|policy|terms|crash|report|download|sparkle|license|legal|"
    r"appcast|telemetry|analytics|endpoint|server|api|issue|bug|documentation|docs|release",
    re.I,
)


def _web_start_url(info: dict) -> str:
    """Web-app / site-specific-browser shims keep the site they open in their Info.plist."""
    items = list(info.items())
    for _k, v in info.items():  # one level of nesting (a manifest dict)
        if isinstance(v, dict):
            items += list(v.items())
    for k, v in items:
        if (
            isinstance(k, str)
            and isinstance(v, str)
            and v.startswith(("http://", "https://"))
            and "url" in k.lower()
            and not _WEB_URL_SKIP.search(k)
        ):
            return v
    return ""


def _media_words(text: str) -> tuple[str, ...]:
    """Which content kinds the words in ``text`` mention. Short words must be whole tokens
    ('tv', 'fm'); longer ones may be part of a token ('podcasts', 'musicbox')."""
    toks = re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", text).lower())
    hits = []
    for kind, words in _MEDIA_WORDS.items():
        if any((w in toks) if len(w) <= 3 else any(w in t for t in toks) for w in words):
            hits.append(kind)
    return tuple(hits)


@dataclass(frozen=True)
class _Static:
    """Everything about a bundle that only changes when the bundle does."""

    key: tuple
    path: str
    real: str
    bundle_id: str
    plist_name: str
    aliases: tuple[str, ...]
    plist_category: str
    doc: _Doc
    schemes: tuple[str, ...]
    own_schemes: tuple[str, ...]
    script: sdef.Dictionary | None
    tags: frozenset[str]
    web_start_url: str
    signals: frozenset[str]
    ev: evidence.Evidence = evidence.NONE


def _plist_key(app: str) -> tuple | None:
    for p in (os.path.join(app, "Contents", "Info.plist"), os.path.join(app, "WrappedBundle", "Info.plist")):
        try:
            st = os.stat(p)
        except OSError:
            continue
        if stat.S_ISREG(st.st_mode):
            return (p, st.st_mtime_ns, st.st_size)
    return None


def _build_static(app: str, key: tuple, need_category: bool = False) -> _Static | None:
    info, where = _read_info(app, b"LSApplicationCategoryType" if need_category else b"")
    if not info or _truthy(info.get("LSBackgroundOnly")):
        return None
    bid = str(info.get("CFBundleIdentifier") or "").strip()
    if not bid:
        return None
    cat = str(info.get("LSApplicationCategoryType") or "").strip()
    if need_category and not cat:
        return None
    stem = os.path.basename(app.rstrip("/"))
    stem = stem[:-4] if stem.lower().endswith(".app") else stem
    names = [str(info.get(k) or "").strip() for k in ("CFBundleDisplayName", "CFBundleName")]
    plist_name = next((n for n in names if n), stem)
    doc = _doc_roles(info)
    schemes = tuple(
        sorted(
            {
                str(s).lower()
                for u in info.get("CFBundleURLTypes") or []
                if isinstance(u, dict)
                for s in u.get("CFBundleURLSchemes") or []
                if isinstance(s, str) and s
            }
        )
    )
    script = sdef.load(app, info) if where.endswith("Contents") else None
    exe_name = str(info.get("CFBundleExecutable") or "")
    exe = (
        os.path.join(where, "MacOS", exe_name)
        if where.endswith("Contents")
        else os.path.join(where, exe_name)
    )
    fws = _linked_frameworks(exe) if exe_name and os.path.isfile(exe) else frozenset()
    signals = {f"links {f}" for f in fws & {"MediaPlayer", "MusicKit", "MediaRemote"}}
    if "audio" in (info.get("UIBackgroundModes") or []):
        signals.add("background-audio")
    if _truthy(info.get("MPSupportsExternallyPlayableContent")):
        signals.add("externally-playable")
    if info.get("NSAppleMusicUsageDescription"):
        signals.add("media-library-access")
    if info.get("NSMicrophoneUsageDescription"):
        signals.add("mic")
    if os.path.isdir(os.path.join(app, "Contents", "Frameworks", "Electron Framework.framework")):
        signals.add("electron")
    aliases = tuple(dict.fromkeys(a for a in (*names, stem) if a))
    try:
        ev = evidence.read(where, info)
    except Exception:  # noqa: BLE001 - evidence only adds to an app: an unreadable file must never drop it
        ev = evidence.NONE
    return _Static(
        key=key,
        path=app,
        real=os.path.realpath(app),
        bundle_id=bid,
        plist_name=plist_name,
        aliases=aliases,
        plist_category=cat,
        doc=doc,
        schemes=schemes,
        own_schemes=own_schemes(schemes, bid, f"{plist_name} {stem}"),
        script=script,
        tags=sdef.caps(script),
        web_start_url=_web_start_url(info),
        signals=frozenset(signals),
        ev=ev,
    )


# ============================================================================== finding the apps
def _skip_path(p: str) -> bool:
    return (
        not p.lower().endswith(".app")
        or "/." in p  # hidden dirs: caches, test-browser runtimes, the Trash
        or "/node_modules/" in p
        or ".app/" in p  # helper apps nested inside other apps
        or "/Library/" in p  # agents, caches, Application Support, developer tools
        or (p.startswith("/Volumes/") and "/Applications/" not in p)  # mounted installer images
    )


def _in_dir(p: str, d: str) -> bool:
    return p == d or p.startswith(d.rstrip("/") + "/")


def _spotlight_app_paths() -> list[str]:
    try:
        out = subprocess.run(
            ["mdfind", "-0", "kMDItemContentType == 'com.apple.application-bundle'"],
            capture_output=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [x.decode(errors="ignore") for x in out.split(b"\0") if x]


def _launchservices_app_paths() -> list[str]:
    """Apps LaunchServices says open audio/video or web pages (catches apps in odd places)."""
    try:
        from AppKit import NSWorkspace
        from Foundation import NSURL

        UT, ws = _uttype(), NSWorkspace.sharedWorkspace()
        out: list[str] = []
        for uti in ("public.audiovisual-content", "public.audio", "public.movie"):
            t = UT.typeWithIdentifier_(uti)
            out += [
                str(u.path()) for u in (ws.URLsForApplicationsToOpenContentType_(t) or []) if t is not None
            ]
        out += [
            str(u.path()) for u in (ws.URLsForApplicationsToOpenURL_(NSURL.URLWithString_("https:")) or [])
        ]
        return out
    except Exception:  # noqa: BLE001 - LaunchServices is a nicety here
        return []


def _folder_app_paths() -> tuple[list[str], list[str], list[str]]:
    """(apps in the app folders, OS service apps that need a category, sub-folders walked)."""
    apps: list[str] = []
    subdirs: list[str] = []
    for d in APP_DIRS:
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.name.startswith("."):
                        continue
                    if e.name.lower().endswith(".app"):
                        apps.append(e.path)
                    elif e.is_dir(follow_symlinks=False):
                        subdirs.append(e.path)
                        with contextlib.suppress(OSError), os.scandir(e.path) as it2:
                            apps += [
                                e2.path
                                for e2 in it2
                                if e2.name.lower().endswith(".app") and not e2.name.startswith(".")
                            ]
        except OSError:
            continue
    services: list[str] = []
    with contextlib.suppress(OSError), os.scandir(SERVICES_DIR) as it:
        services = [e.path for e in it if e.name.lower().endswith(".app") and not e.name.startswith(".")]
    return apps, services, subdirs


def _dir_rank(p: str) -> int:
    for i, d in enumerate(APP_DIRS):
        if _in_dir(p, d):
            return i
    return len(APP_DIRS) + 1


def _enumerate(pool: ThreadPoolExecutor | None = None) -> tuple[list[tuple[str, bool]], list[str]]:
    """[(path, needs_category)] of candidate bundles (deduped by real path, visible path kept),
    plus the sub-folders walked (their mtimes tell installs/removals cheaply)."""
    spot: Future | None = pool.submit(_spotlight_app_paths) if pool else None  # mdfind runs meanwhile
    folder, services, subdirs = _folder_app_paths()
    found = [(p, False) for p in folder] + [(p, True) for p in services]
    spotted = spot.result() if spot else _spotlight_app_paths()
    for p in spotted + _launchservices_app_paths():
        if _in_dir(p, SERVICES_DIR) and os.path.dirname(p) == SERVICES_DIR:
            found.append((p, True))
        elif not _skip_path(p) or any(_in_dir(p, d) for d in APP_DIRS):
            found.append((p, False))
    by_real: dict[str, tuple[str, bool]] = {}
    for p, need in sorted(found, key=lambda x: (_dir_rank(x[0]), len(x[0]), x[0])):
        if ".app/" in p or "/." in p:
            continue
        rp = os.path.realpath(p)
        if rp in by_real:
            if not need and by_real[rp][1]:
                by_real[rp] = (by_real[rp][0], False)
            continue
        by_real[rp] = (p, need)
    return list(by_real.values()), subdirs


def _dir_sig(subdirs: list[str]) -> tuple:
    sig = []
    for d in (*APP_DIRS, SERVICES_DIR, *subdirs):
        try:
            sig.append((d, os.stat(d).st_mtime_ns))
        except OSError:
            sig.append((d, 0))
    return tuple(sig)


# ============================================================================== this user's use
def _spotlight_usage(paths: list[str]) -> dict[str, tuple[float | None, int, int, str, str]]:
    """{path: (last_used, use_count, distinct days used in 30, App Store category, display name)}."""
    try:
        import objc
        from Foundation import NSURL, NSMetadataItem
    except ImportError:
        return {}
    attrs = [
        "kMDItemLastUsedDate",
        "kMDItemUseCount",
        "kMDItemUsedDates",
        "kMDItemAppStoreCategoryType",
        "kMDItemDisplayName",
    ]
    now = time.time()
    out: dict[str, tuple[float | None, int, int, str, str]] = {}
    with objc.autorelease_pool():
        for p in paths:
            try:
                it = NSMetadataItem.alloc().initWithURL_(NSURL.fileURLWithPath_(p))
                v = it.valuesForAttributes_(attrs) if it is not None else None
            except Exception:  # noqa: BLE001
                v = None
            if not v:
                out[p] = (None, 0, 0, "", "")
                continue
            last = v.get("kMDItemLastUsedDate")
            days = set()
            for d in v.get("kMDItemUsedDates") or []:
                ts = float(d.timeIntervalSince1970())
                if now - ts <= 30 * 86400:
                    days.add(datetime.fromtimestamp(ts).date())
            try:
                count = int(v.get("kMDItemUseCount") or 0)
            except (TypeError, ValueError):
                count = 0
            out[p] = (
                float(last.timeIntervalSince1970()) if last is not None else None,
                count,
                len(days),
                str(v.get("kMDItemAppStoreCategoryType") or ""),
                str(v.get("kMDItemDisplayName") or ""),
            )
    return out


_LIBPROC = None
_MAIN_EXE = re.compile(r"Contents/MacOS/[^/]+|Wrapper/[^/]+\.app/[^/]+|WrappedBundle/[^/]+")


@lru_cache(maxsize=2048)
def _declared_exe(bundle: str) -> str | None:
    """The bundle's own CFBundleExecutable (None if unreadable): a background helper that also
    lives in Contents/MacOS/ runs without the app being open."""
    try:
        with open(os.path.join(bundle, "Contents", "Info.plist"), "rb") as f:
            exe = plistlib.load(f).get("CFBundleExecutable")
        return str(exe) if exe else None
    except (OSError, plistlib.InvalidFileException, ValueError, AttributeError):
        return None


def _is_main_exe(bundle: str, rest: str) -> bool:
    if not _MAIN_EXE.fullmatch(rest):
        return False
    if rest.startswith("Contents/MacOS/"):
        declared = _declared_exe(bundle)
        return declared is None or rest == f"Contents/MacOS/{declared}"
    return True


@pooled
def _running_reals() -> set[str]:
    """Real paths of app bundles with a live process - read fresh from the process table (NSWorkspace's
    list goes stale in a process that doesn't pump a Cocoa run loop)."""
    global _LIBPROC
    try:
        if _LIBPROC is None:
            _LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        lib = _LIBPROC
        n = lib.proc_listallpids(None, 0)
        if n <= 0:
            raise OSError("proc_listallpids")
        arr = (ctypes.c_int * (n + 64))()
        n = lib.proc_listallpids(arr, ctypes.sizeof(arr))
        buf = ctypes.create_string_buffer(4096)
        out: set[str] = set()
        for i in range(max(0, n)):
            pid = arr[i]
            if pid <= 0 or lib.proc_pidpath(pid, buf, 4096) <= 0:
                continue
            exe = buf.value.decode(errors="ignore")
            j = exe.find(".app/")
            # Only the app's own main executable: a widget/extension or helper living inside the
            # bundle (PlugIns/*.appex, Frameworks/*Helper.app) runs without the app being open.
            if j > 0 and _is_main_exe(exe[: j + 4], exe[j + 5 :]):
                out.add(exe[: j + 4])
        return out
    except (OSError, AttributeError):
        try:
            from AppKit import NSWorkspace

            return {
                os.path.realpath(str(a.bundleURL().path()))
                for a in NSWorkspace.sharedWorkspace().runningApplications()
                if a.bundleURL() is not None
            }
        except Exception:  # noqa: BLE001
            return set()


@pooled
def _dock_ids() -> set[str]:
    """Bundle ids of the apps shown in the Dock (public CFPreferences read; ids only)."""
    try:
        from CoreFoundation import CFPreferencesAppSynchronize, CFPreferencesCopyAppValue

        CFPreferencesAppSynchronize(_DOCK_DOMAIN)
        ids = set()
        for key in ("persistent-apps", "recent-apps"):
            for tile in CFPreferencesCopyAppValue(key, _DOCK_DOMAIN) or []:
                bid = (tile.get("tile-data") or {}).get("bundle-identifier")
                if bid:
                    ids.add(str(bid))
        return ids
    except Exception:  # noqa: BLE001
        return set()


@pooled
def _scheme_defaults() -> dict[str, str]:
    """{standard scheme: real path of its default handler}."""
    try:
        from AppKit import NSWorkspace
        from Foundation import NSURL

        ws, out = NSWorkspace.sharedWorkspace(), {}
        for s in HANDLED_SCHEMES:
            u = ws.URLForApplicationToOpenURL_(NSURL.URLWithString_(f"{s}:"))
            if u is not None:
                out[s] = os.path.realpath(str(u.path()))
        return out
    except Exception:  # noqa: BLE001
        return {}


# ============================================================================== assembly + cache
def _media_evidence(s: _Static, cat: str, handles: tuple[str, ...]) -> tuple[str, ...]:
    """Why this could be a media app, strongest first; () when nothing points that way."""
    tags, sig, doc = s.tags, s.signals, s.doc
    player_state = "media.state" in tags and bool(
        tags & {"media.resume", "media.search_play", "media.play_uri"}
    )
    strong = [f"category={cat}"] if cat in ("music", "video") else []
    strong += ["externally-playable"] if "externally-playable" in sig else []
    strong += ["script: player state"] if player_state else []
    support = [
        x
        for x in (
            "background-audio",
            "media-library-access",
            "links MediaPlayer",
            "links MusicKit",
            "links MediaRemote",
        )
        if x in sig
    ]
    support += [
        label
        for tag, label in (
            ("media.search_play", "script: search+play"),
            ("media.play_uri", "script: play by URI"),
        )
        if tag in tags
    ]
    if "media.resume" in tags:
        support.append("script: play")
    support += ["category=entertainment"] if cat == "entertainment" else []
    words = _media_words(
        " ".join([s.bundle_id, s.plist_name, *s.aliases, *s.own_schemes, cat, s.web_start_url])
    )
    electron = "electron" in sig
    door = (
        bool(s.own_schemes)
        or bool(tags & {"media.search_play", "media.play_uri"})
        or player_state
        or bool(sig & {"externally-playable", "background-audio"})
    )
    uncategorised = cat in ("", "entertainment")
    viewer, editor, mic = doc.av_view, doc.av_edit, "mic" in sig
    if ("https" in s.schemes or "https" in handles) and doc.html:
        kind = "web browser"
    elif s.web_start_url:
        kind = "web app"
    elif (
        viewer >= 5
        and viewer > editor
        and not (player_state or "media.search_play" in tags or "externally-playable" in sig)
    ):
        kind = "media file player"  # opens many media files, no catalog/library verbs
    elif (
        strong
        and mic
        and (editor > viewer or doc.owns_project)
        and not (player_state or sig & {"externally-playable", "background-audio"})
    ):
        kind = "audio/video editor"  # records/edits (music-category creator apps)
    elif strong and door:
        kind = "media app"
    elif viewer >= 3:
        kind = "media file player"
    elif uncategorised and (
        (words and (support or electron))
        or ("background-audio" in sig and "links MediaPlayer" in sig)
        or (words and not cat)
    ):
        kind = "possible media app"  # e.g. a wrapper with a media-ish name; weak on its own
    else:
        return ()
    ev = [kind, *strong, *support]
    if viewer:
        ev.append(f"opens {viewer} audio/video types")
    if editor:
        ev.append(f"edits {editor} audio/video types")
    if s.own_schemes and kind not in ("web browser",):
        ev.append("own scheme " + ",".join(s.own_schemes[:3]))
    if words and kind in ("possible media app", "web app", "media app"):
        ev.append("name suggests " + "/".join(words))
    if s.web_start_url:
        host = re.sub(r"^https?://([^/]+).*$", r"\1", s.web_start_url)
        ev.append(f"web app for {host}")
    if electron and kind != "web browser":
        ev.append("electron")
    return tuple(dict.fromkeys(ev))


def _assemble(
    s: _Static,
    use: tuple[float | None, int, int, str, str] | None,
    running: set[str],
    dock: set[str],
    defaults: dict[str, str],
) -> App:
    last, count, days, store_cat, shown = use or (None, 0, 0, "", "")
    cat = (s.plist_category or store_cat).strip()
    cat = cat.removeprefix("public.app-category.")
    shown = shown[:-4] if shown.lower().endswith(".app") else shown
    name = shown or s.plist_name
    handles = tuple(sch for sch in HANDLED_SCHEMES if defaults.get(sch) == s.real)
    return App(
        bundle_id=s.bundle_id,
        name=name,
        path=s.path,
        category=cat,
        viewer_types=s.doc.viewer,
        editor_types=s.doc.editor,
        schemes=s.schemes,
        own_schemes=s.own_schemes,
        handles=handles,
        script=s.script,
        media_evidence=_media_evidence(s, cat, handles),
        web_start_url=s.web_start_url,
        last_used=last,
        use_count=count,
        days_used_30=days,
        running=s.real in running,
        in_dock=s.bundle_id in dock,
        aliases=tuple(dict.fromkeys(a for a in (name, *s.aliases) if a)),
        intents=s.ev.intents,
        ai_schemas=s.ev.ai_schemas,
        share_text=s.ev.share_text,
        services=s.ev.services,
        usage=s.ev.usage,
        doc_based=s.ev.doc_based,
    )


class _Cache:
    def __init__(self) -> None:
        self.paths: list[tuple[str, bool]] | None = None
        self.paths_at = 0.0
        self.sig: tuple = ()
        self.subdirs: list[str] = []
        # (path, needs a category) -> static (None = not a user-facing app), and its plist key
        self.static: dict[tuple[str, bool], _Static | None] = {}
        self.static_key: dict[tuple[str, bool], tuple | None] = {}
        self.usage: dict[str, tuple[float, tuple]] = {}  # path -> (read at, usage)
        self.was_running: set[str] = set()
        self.dock: set[str] = set()
        self.defaults: dict[str, str] = {}
        self.ctx_at = 0.0
        self.apps: list[App] = []
        self.by_real: dict[str, App] = {}
        self.scanned = False  # a scan has completed (snapshot() has something to give)


_cache = _Cache()
_lock = threading.RLock()


def _reset() -> None:
    """Forget everything (tests; or after the user moves apps around)."""
    global _cache
    with _lock:
        _cache = _Cache()


def _static_for(path: str, need_category: bool) -> _Static | None:
    key = _plist_key(path)
    if key is None:
        return None
    # Cached per (path, need_category): an OS service app skipped from the listing for having no
    # category is still a real answer for default_for / app_by_id (e.g. a disk-image handler).
    ck = (path, need_category)
    if _cache.static_key.get(ck) == key and ck in _cache.static:
        return _cache.static[ck]
    try:
        s = _build_static(path, key, need_category)
    except Exception:  # noqa: BLE001 - one odd bundle must never break the inventory
        s = None
    _cache.static[ck], _cache.static_key[ck] = s, key
    return s


@pooled
def inventory(refresh: bool = False) -> list[App]:
    """All real user-facing apps on this Mac (sorted by name). Cached; ``refresh`` re-enumerates
    and re-reads everything. Without it, usage and the running state are still re-read cheaply."""
    with _lock:
        c, now = _cache, time.time()
        reenumerate = (
            refresh or c.paths is None or now - c.paths_at > _ENUM_TTL or _dir_sig(c.subdirs) != c.sig
        )
        cold = refresh or not c.usage
        pool = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="neo-inventory")
            if (reenumerate or cold)
            else None
        )
        try:
            return _inventory(c, now, refresh, reenumerate, pool)
        finally:
            if pool is not None:
                pool.shutdown(wait=True)


def _inventory(
    c: _Cache, now: float, refresh: bool, reenumerate: bool, pool: ThreadPoolExecutor | None
) -> list[App]:
    if reenumerate or c.paths is None:
        c.paths, c.subdirs = _enumerate(pool)
        c.sig, c.paths_at = _dir_sig(c.subdirs), now
    if refresh:  # re-read every bundle too (a dictionary can change without its Info.plist)
        c.static.clear()
        c.static_key.clear()
    prefetch: Future | None = None
    if pool is not None and (refresh or not c.usage):  # cold: Spotlight answers while bundles are parsed
        cand = [p for p, need in c.paths if not need]
        prefetch = pool.submit(_spotlight_usage, cand) if cand else None
    statics: dict[str, _Static] = {}
    for p, need in c.paths:
        s = _static_for(p, need)
        if s is None:
            continue
        prev = statics.get(s.bundle_id)  # one entry per bundle id: app folders first
        if prev is None or (_dir_rank(s.path), len(s.path)) < (_dir_rank(prev.path), len(prev.path)):
            statics[s.bundle_id] = s
    live = list(statics.values())
    running = _running_reals()
    if refresh or now - c.ctx_at > _CTX_TTL:
        c.dock, c.defaults, c.ctx_at = _dock_ids(), _scheme_defaults(), now
    stale = [
        s.path
        for s in live
        if refresh
        or s.path not in c.usage
        or now - c.usage[s.path][0] > _USAGE_TTL
        or s.real in running
        or s.real in c.was_running
    ]
    if stale:
        got = dict(prefetch.result()) if prefetch is not None else {}
        rest = [p for p in stale if p not in got]
        if rest:
            got.update(_spotlight_usage(rest))
        for p in stale:
            c.usage[p] = (now, got.get(p))
    c.was_running = running
    apps = [_assemble(s, (c.usage.get(s.path) or (0, None))[1], running, c.dock, c.defaults) for s in live]
    apps.sort(key=lambda a: (a.name.casefold(), a.bundle_id))
    c.apps, c.by_real = apps, {os.path.realpath(a.path): a for a in apps}
    c.scanned = True
    return list(apps)


def snapshot() -> list[App] | None:
    """The apps as of the last completed scan, without waiting for (or starting) one: for checks
    that must answer in milliseconds. None before the first scan."""
    c = _cache
    return list(c.apps) if c.scanned else None


def app_for_bundle(bundle_id: str) -> App | None:
    """The app with this bundle id in the last completed scan (never waits for one)."""
    want = (bundle_id or "").strip().lower()
    return next((a for a in snapshot() or () if want and a.bundle_id.lower() == want), None)


def app_for_pid(pid: int) -> App | None:
    """The scanned app a process belongs to (the app macOS holds responsible for it)."""
    from neo2.core.decide import media

    try:
        return app_for_bundle(media._bundle_id_of_pid(media.responsible_pid(pid)) or media._bundle_id_of_pid(pid))
    except Exception:  # noqa: BLE001 - no AppKit: nothing to tell
        return None


@pooled
def _app_for_path(path: str) -> App | None:
    """The inventory entry for a bundle path, or one built on the spot for an app elsewhere."""
    rp = os.path.realpath(path)
    with _lock:
        if not _cache.by_real:
            inventory()
        a = _cache.by_real.get(rp)
        if a is not None:
            return a
        s = _static_for(path, False)  # cached like the rest, but not listed in the inventory
        if s is None:
            return None
        use = _spotlight_usage([path]).get(path)
        return _assemble(s, use, _running_reals(), _cache.dock, _cache.defaults)


@pooled
def _ls_path_for_id(bundle_id: str) -> str | None:
    """Where LaunchServices has an app with this bundle id (anywhere on disk)."""
    try:
        from AppKit import NSWorkspace

        u = NSWorkspace.sharedWorkspace().URLForApplicationWithBundleIdentifier_(bundle_id)
    except Exception:  # noqa: BLE001
        return None
    return str(u.path()) if u is not None else None


@pooled
def _ls_default_path(url_or_uti: str) -> str | None:
    """LaunchServices' default handler (bundle path) for a URL/scheme, a UTI or a file extension."""
    try:
        from AppKit import NSWorkspace
        from Foundation import NSURL

        ws = NSWorkspace.sharedWorkspace()
        if re.match(r"^[A-Za-z][A-Za-z0-9+.\-]*:", url_or_uti):
            url = NSURL.URLWithString_(url_or_uti)
            u = ws.URLForApplicationToOpenURL_(url) if url is not None else None
        else:
            UT = _uttype()
            s = url_or_uti
            t = (
                UT.typeWithIdentifier_(s)
                if "." in s.strip(".")
                else UT.typeWithFilenameExtension_(s.lstrip("."))
            )
            u = ws.URLForApplicationToOpenContentType_(t) if t is not None else None
    except Exception:  # noqa: BLE001
        return None
    return str(u.path()) if u is not None else None


@pooled
def app_by_id(bundle_id: str) -> App | None:
    want = (bundle_id or "").strip()
    if not want:
        return None
    for a in inventory():
        if a.bundle_id.lower() == want.lower():
            return a
    p = _ls_path_for_id(want)  # installed somewhere the scan doesn't look
    return _app_for_path(p) if p else None


@pooled
def default_for(url_or_uti: str) -> App | None:
    """The default handler for a URL / scheme ('mailto:', 'https://...') or a content type
    ('public.mp3', or a bare extension 'mp3')."""
    s = (url_or_uti or "").strip()
    p = _ls_default_path(s) if s else None
    return _app_for_path(p) if p else None


# ============================================================================== how much this user uses it
def affinity(app: App, now: float | None = None) -> float:
    """0..1: how much THIS user uses the app (recency, launches, distinct days, running, Dock)."""
    now = time.time() if now is None else now
    s = 0.0
    if app.last_used:
        age_days = max(0.0, (now - app.last_used) / 86400)
        s += 4 * math.exp(-age_days / 14)  # recency (~10 day half-life)
    s += 0.5 * min(max(app.days_used_30, 0), 10)  # distinct days this month
    s += 0.6 * math.log1p(max(app.use_count, 0))  # lifetime launches (a few vs. hundreds)
    s += 3.0 if app.running else 0.0
    s += 1.5 if app.in_dock else 0.0
    s += 0.5 if app.handles else 0.0
    return round(1.0 - math.exp(-s / 6.0), 4)


# ============================================================================== "on spotify", "in notes"
# Words that introduce an app name ("play it ON x", "open x"); a name that is also a common
# English word only counts right after one of these, or when followed by "app".
_VERB_TRIGGER = r"(?:open|launch|start|run|quit|close|activate|use|using|show|switch\s+to|go\s+to|bring\s+up|pull\s+up|focus\s+on|focus)"
_PREP_TRIGGER = r"(?:on|in|with|via|through|inside|into)"
_PHRASAL_ON = re.compile(
    r"\b(turn|put|switch|log|sign|carry|keep|hold|come|get|move|go|based|depends|later)\s+$"
)
_FUNCTION_WORDS = frozenset(
    {"a", "an", "the", "app", "apps", "application", "my", "your", "it", "this", "that"}
)
# 'Neo, ...' / 'hey Neo, ...' (a name before a comma), or 'hey neo play ...' (greeting + a word,
# which is only the assistant's name when it isn't an ordinary word: 'ok play music').
_VOCATIVE = re.compile(
    r"^\s*(?:(?:hey|hi|ok|okay|yo)\s+)?[\w.-]+\s*,\s*"
    r"|^\s*(?P<greet>(?:hey|hi|ok|okay|yo)\s+)(?P<bare>[\w.-]+)\s+"
)
# Generic nouns that name what an app is FOR, not which app: the media content nouns above plus
# everyday words newer than the system word list ('podcast', 'screenshot'). An app called by one
# of these ('Podcasts', 'Screenshot') needs a cue like any other ordinary word.
_GENERIC_NOUNS = frozenset(
    {w for ws in _MEDIA_WORDS.values() for w in ws}
    | {
        "podcast",
        "screenshot",
        "email",
        "playlist",
        "voicemail",
        "website",
        "webpage",
        "online",
        "inbox",
        "wifi",
    }
)

_words_known: dict[str, bool] = {}
_connectives: frozenset[str] | None = None
_words_lock = threading.Lock()


def _variants(w: str) -> list[str]:
    v = [w]
    if w.endswith("ies") and len(w) > 4:
        v.append(w[:-3] + "y")
    if w.endswith("es") and len(w) > 3:
        v.append(w[:-2])
    if w.endswith("s") and len(w) > 3:
        v.append(w[:-1])
    return v


def _load_connectives() -> frozenset[str]:
    global _connectives
    if _connectives is None:
        try:
            with open(_CONNECTIVES_FILE, encoding="utf-8", errors="ignore") as f:
                _connectives = frozenset(w.strip().lower() for w in f if w.strip())
        except OSError:
            _connectives = frozenset()
    return _connectives


def _common_words(words: set[str]) -> dict[str, bool]:
    """Which single words are ordinary dictionary words (plurals included) - one streaming pass over
    the system word list for the words not seen before. Without a word list every word counts as
    common, which only makes matching stricter."""
    with _words_lock:
        todo = {w for w in words if w not in _words_known}
        if todo:
            want: dict[str, set[str]] = {}
            for w in todo:
                for v in _variants(w):
                    want.setdefault(v, set()).add(w)
            hit: set[str] = set()
            try:
                with open(_WORDS_FILE, encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        owners = want.get(line.strip().lower())
                        if owners:
                            hit |= owners
                missing = False
            except OSError:
                missing = True
            for w in todo:
                _words_known[w] = True if missing else (w in hit)
        return {w: _words_known[w] for w in words}


def _alias_forms(app: App) -> list[str]:
    """Lower-case names an app goes by: its names, the names without the vendor token
    ('google chrome' -> 'chrome') or with it ('music' -> 'apple music'), before a dot ('zoom.us'),
    without trailing version/edition words, and its own URL schemes."""
    names = [app.name, *app.aliases, os.path.basename(app.path.rstrip("/"))]
    parts = [p for p in re.split(r"[.\-_]", app.bundle_id.lower()) if p]
    vendor = parts[1] if len(parts) >= 3 and parts[1] not in _GENERIC_TOKENS and len(parts[1]) >= 3 else ""
    out: list[str] = []
    for n in names:
        n = re.sub(r"\.app$", "", n.strip(), flags=re.I).lower()
        n = re.sub(r"\s+", " ", n.replace("’", "'")).strip()
        if not n:
            continue
        cands = [n]
        # trailing version / edition / generic descriptor words: 'quicktime player' -> 'quicktime'
        base = re.sub(r"(\s+(\d[\w.]*|desktop|app|beta|for mac(os)?|mac|player|pro))+$", "", n).strip()
        cands.append(base)
        if "." in base and not base.startswith("."):
            cands.append(base.split(".")[0])
        if " " in base:
            cands.append(base.replace(" ", ""))
        if vendor:
            if base.startswith(vendor + " "):
                cands.append(base[len(vendor) + 1 :])
            elif vendor not in base.split():
                cands.append(f"{vendor} {base}")
        out += cands
    out += [s for s in app.own_schemes if re.fullmatch(r"[a-z][a-z0-9]{2,}", s)]
    conn = _load_connectives()
    return [a for a in dict.fromkeys(out) if len(a) >= 2 and a not in _FUNCTION_WORDS and a not in conn]


_forms_cache: dict[tuple, list[str]] = {}


def _forms_cached(app: App) -> list[str]:
    key = (app.bundle_id, app.name, app.path, app.aliases, app.own_schemes)
    f = _forms_cache.get(key)
    if f is None:
        if len(_forms_cache) > 4096:
            _forms_cache.clear()
        f = _forms_cache[key] = _alias_forms(app)
    return f


def name_forms(app: App) -> list[str]:
    """The lower-case names the app goes by (see :func:`_alias_forms`), cached."""
    return list(_forms_cached(app))


def _alias_rx(alias: str) -> re.Pattern:
    body = r"\s+".join(re.escape(t) for t in alias.split())
    return re.compile(rf"(?<![\w]){body}(?![\w])")


_rx_cache: dict[str, re.Pattern] = {}


def named_in(text: str, apps: list[App] | None = None) -> App | None:
    """The installed app the user named in ``text`` ('play 505 on spotify', 'open whatsapp',
    'with vlc'), or None. Whole words, case-insensitive, longest name wins. A name made only of
    ordinary words ('music', 'notes', 'safari', 'find my', 'podcasts') counts only where it names
    an app: after 'on/in/with/open/use ...', before 'app', or as the whole request. A leading
    'X,' / 'hey X' is the assistant being addressed, not an app."""
    t = re.sub(r"\s+", " ", (text or "").lower().replace("’", "'")).strip()
    if not t.strip(" .!?"):
        return None
    hit = _scan(t, inventory() if apps is None else apps)
    return hit[0] if hit else None


def named_span(text: str, apps: list[App] | None = None) -> tuple[App, int, int] | None:
    """:func:`named_in`'s app and where its name is in ``text`` (``text[start:end]``), so the name
    can be masked before a model reads the request."""
    # lower-cased and one space per whitespace character: positions stay those of ``text``
    t = "".join(c.lower() if len(c.lower()) == 1 else c for c in (text or "")).replace("’", "'")
    t = re.sub(r"\s", " ", t)
    if not t.strip(" .!?"):
        return None
    return _scan(t, inventory() if apps is None else apps)


def _scan(t: str, apps: list[App]) -> tuple[App, int, int] | None:
    """The named app and its span in ``t`` (lower-case, spaces for whitespace)."""
    forms = {a.bundle_id or a.path: _forms_cached(a) for a in apps}
    voc = _VOCATIVE.match(t)
    words = {w for fs in forms.values() for f in fs for w in f.split() if w.isalpha()}
    if voc and voc.group("bare") and voc.group("bare").isalpha():
        words.add(voc.group("bare"))
    common = _common_words(words)
    off = 0
    if voc:  # 'hey neo play ...' drops the name; 'ok play music' keeps 'play music'
        bare = voc.group("bare")
        off = voc.end("greet") if bare and common.get(bare, True) else voc.end()
    t = t[off:]
    off += len(t) - len(t.lstrip(" .!?"))
    t = t.strip(" .!?")
    if not t:
        return None
    conn = _load_connectives()

    def plain(w: str) -> bool:
        return (
            len(w) <= 2
            or w in _FUNCTION_WORDS
            or w in conn
            or common.get(w, False)
            or any(v in _GENERIC_NOUNS for v in _variants(w))
        )

    best: tuple[int, float, App, int, int] | None = None
    for app in apps:
        for alias in forms[app.bundle_id or app.path]:
            rx = _rx_cache.get(alias)
            if rx is None:
                rx = _rx_cache[alias] = _alias_rx(alias)
            # made only of ordinary words ('music', 'find my', 'app store'): needs a cue
            ordinary = all(plain(w) for w in alias.split())
            for m in rx.finditer(t):
                if ordinary and not _names_app(t, m.start(), m.end()):
                    continue
                cand = (len(alias), affinity(app), app, m.start(), m.end())
                if best is None or cand[:2] > best[:2]:
                    best = cand
                break
    return (best[2], off + best[3], off + best[4]) if best else None


def _names_app(t: str, start: int, end: int) -> bool:
    """Is the ordinary word at t[start:end] used as an app name here?"""
    if t[:start].strip() == "" and t[end:].strip() in ("", "app", "application"):
        return True
    if re.match(r"\s+(app|application)\b", t[end:]):
        return True
    before = t[:start]
    if re.search(rf"\b{_VERB_TRIGGER}\s+(?:(?:the|my)\s+)?$", before):
        return True
    m = re.search(rf"\b({_PREP_TRIGGER})\s+$", before)
    return bool(m) and not (m.group(1) == "on" and _PHRASAL_ON.search(before[: m.start()]))
