"""An app's AppleScript dictionary (sdef), read straight from its bundle, and the generic
capability tags it implies.

Nothing here launches or talks to the app: the ``sdef`` CLI needs Xcode and
``OSACopyScriptingDefinition`` can launch the target, so the XML is parsed directly. That covers
the forms seen in the wild: XInclude in both the 2001 and 2003 namespaces (with or without an
``xpointer`` that filters what is pulled in), commented-out includes, and hand-written files with
undefined HTML entities (``&nbsp;``).

Capabilities are inferred from the dictionary alone, by command/class/property NAMES and by
their four-char CODES (the de-facto AppleEvent codes many apps copy: ``hookPlay``-style event ids,
``pPlS`` player state, ``pTrk`` current track, ``corecrel`` make ...). No app names anywhere.
Legacy apps that set ``NSAppleScriptEnabled`` but ship no sdef are treated as having none.
"""

from __future__ import annotations

import os
import plistlib
import re
import stat
import threading
import urllib.parse
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass, field

__all__ = ["Command", "Dictionary", "load", "parse", "caps", "read_caps", "ALL_CAPS", "READ_CAPS"]

XI_TAGS = frozenset({"{http://www.w3.org/2001/XInclude}include", "{http://www.w3.org/2003/XInclude}include"})
_MAX_BYTES = 8 << 20  # no real dictionary is anywhere near this
_MAX_DEPTH = 3
# Anything that looks like an entity but isn't one of XML's five or a character reference.
_BAD_ENTITY = re.compile(rb"&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)[A-Za-z][\w.-]*;")

ALL_CAPS = frozenset(
    {
        "media.search_play",  # search the app's catalog/library for text, then play a hit
        "media.play_uri",  # play an item by an app-specific id/URI (text)
        "media.resume",  # play/resume with no argument, or toggle play/pause
        "media.state",  # player state (playing / paused / stopped) is readable
        "media.now_playing",  # the current item is readable
        "media.item_uri",  # the current item exposes its own URI (replayable via media.play_uri)
        "note.make",
        "event.make",
        "reminder.make",
        "message.send",
        "mail.compose",
    }
)
# What NEO can read or look up through the dictionary (:func:`read_caps`), kept apart from what it
# can make the app do.
READ_CAPS = frozenset(
    {
        "note.read",  # a note class with a readable text property
        "reminder.read",  # a reminder/to-do class with a completed flag
        "event.read",  # an event class with a start date, reachable from the application
        "mail.read",  # a message class with a read status
        "mail.unread_count",  # an unread count property
        "web.search",  # a search for text as such (the engine counts it only for browsers)
    }
)


@dataclass(frozen=True)
class Command:
    """One command. ``direct`` is the type of the direct parameter ('' when it takes none;
    alternatives joined by '|', lists as 'list of X'); ``params`` are (name, code, type)."""

    name: str
    code: str  # 8 chars: event class + event id
    direct: str = ""
    params: tuple[tuple[str, str, str], ...] = ()
    # Extras (not needed to use the contract, handy for building scripts):
    direct_optional: bool = False
    optional: tuple[str, ...] = ()  # names of the optional labelled parameters
    description: str = ""
    direct_description: str = ""
    result: str = ""

    @property
    def event_id(self) -> str:
        return self.code[4:8]

    def param(self, name: str) -> tuple[str, str, str] | None:
        return next((p for p in self.params if p[0].lower() == name.lower()), None)


@dataclass(frozen=True)
class Dictionary:
    """``commands``: lower-cased name -> Command. ``classes``: name -> (code, property names,
    inherited ones included). ``properties``: name -> code over all classes (the application
    class's win). ``enums``: enumeration name -> {enumerator name: code}."""

    commands: dict[str, Command]
    classes: dict[str, tuple[str, tuple[str, ...]]]
    properties: dict[str, str]
    enums: dict[str, dict[str, str]]
    # Extras (optional; ``caps`` works without them):
    class_props: dict[str, dict[str, tuple[str, str]]] = field(
        default_factory=dict
    )  # class -> {prop: (code, type)}
    elements: dict[str, tuple[str, ...]] = field(default_factory=dict)  # class -> element class names
    inherits: dict[str, str] = field(default_factory=dict)  # class -> superclass
    # class -> {prop: (access 'r' | 'w' | 'rw', description)}, inherited ones included
    prop_meta: dict[str, dict[str, tuple[str, str]]] = field(default_factory=dict)

    def __hash__(self) -> int:  # dict fields: hash what identifies the dictionary
        return hash(
            (
                tuple(sorted(self.commands)),
                tuple(sorted(self.classes)),
                tuple(sorted(self.properties.items())),
            )
        )

    def command(self, name: str) -> Command | None:
        return self.commands.get(name.lower())

    def class_code(self, name: str) -> str:
        c = self.classes.get(name)
        return c[0] if c else ""

    def prop_type(self, klass: str, prop: str) -> str:
        return (self.class_props.get(klass, {}).get(prop) or ("", ""))[1]

    def prop_access(self, klass: str, prop: str) -> str:
        """'r', 'w' or 'rw' (the sdef default when a property doesn't say); '' if unknown."""
        meta = self.prop_meta.get(klass, {}).get(prop)
        if meta is not None:
            return meta[0]
        return "rw" if prop in self.class_props.get(klass, {}) else ""

    def prop_desc(self, klass: str, prop: str) -> str:
        return (self.prop_meta.get(klass, {}).get(prop) or ("", ""))[1]

    def container_path(self, target: str) -> list[str] | None:
        """Shortest element path application -> a class that is (or inherits from) ``target``;
        [] when the application itself is one; None when unknown/unreachable."""
        if not self.elements:
            return None

        def is_a(c: str) -> bool:
            return target in _chain(self.inherits, c)

        if is_a("application"):
            return []
        q, seen = deque([("application", [])]), {"application"}
        while q:
            c, path = q.popleft()
            for cc in _chain(self.inherits, c):
                for el in self.elements.get(cc, ()):
                    if is_a(el):
                        return [*path, el]
                    if el not in seen and len(path) < 3:
                        seen.add(el)
                        q.append((el, [*path, el]))
        return None


def _chain(inherits: dict[str, str], name: str) -> list[str]:
    out: list[str] = []
    while name and name not in out:
        out.append(name)
        name = inherits.get(name, "")
    return out


# ------------------------------------------------------------------------------ XML + XInclude
def _parse_file(path: str) -> ET.Element:
    # Regular files only (an include pointing at a device or a pipe must not hang the scan), bounded.
    if not stat.S_ISREG(os.stat(path).st_mode):
        raise ValueError("not a regular file")
    with open(path, "rb") as f:
        raw = f.read(_MAX_BYTES + 1)
    if len(raw) > _MAX_BYTES:
        raise ValueError("sdef too large")
    try:
        return ET.fromstring(raw)
    except ET.ParseError:  # stray HTML entities in hand-written dictionaries
        return ET.fromstring(_BAD_ENTITY.sub(b" ", raw))


def _href_path(href: str, base_dir: str) -> str:
    if href.startswith("file:"):
        return urllib.parse.unquote(urllib.parse.urlparse(href).path)  # file://localhost/x, file:///x
    return os.path.normpath(os.path.join(base_dir, urllib.parse.unquote(href)))


_INCLUDE_CACHE: dict[tuple[str, int], ET.Element] = {}
_INCLUDE_LOCK = threading.Lock()


def _included_root(path: str, depth: int) -> ET.Element | None:
    """A (shared, read-only) parsed include target such as the Cocoa standard suite."""
    try:
        key = (path, os.stat(path).st_mtime_ns)
    except OSError:
        return None
    with _INCLUDE_LOCK:
        root = _INCLUDE_CACHE.get(key)
    if root is None:
        try:
            root = _parse_file(path)
        except (OSError, ValueError, ET.ParseError):
            return None
        _expand(root, os.path.dirname(path), depth + 1)
        with _INCLUDE_LOCK:
            _INCLUDE_CACHE[key] = root
    return root


def _select(root: ET.Element, xpointer: str) -> list[ET.Element]:
    """The handful of xpointer shapes sdefs use: /dictionary/suite, /dictionary/suite[@name='X'],
    and /dictionary/suite/node()[not(self::command and ((@name = 'a') or ...))]."""
    m = re.search(r"xpointer\((.*)\)\s*$", xpointer or "", re.S)
    expr = (m.group(1) if m else "").strip()
    suites = [root] if root.tag == "suite" else root.findall("suite")
    named = re.match(r"/dictionary/suite\[@name\s*=\s*['\"]([^'\"]+)['\"]\]", expr)
    if named:
        suites = [s for s in suites if s.get("name") == named.group(1)]
    if re.match(r"/dictionary/suite(\[[^\]]*\])?/(node|\*)\(?", expr):
        kids = [k for s in suites for k in s]
        excl = re.search(r"not\(\s*self::([\w-]+)\s+and(.*)\)\s*\]", expr, re.S)
        if excl:
            tag, names = excl.group(1), set(re.findall(r"@name\s*=\s*['\"]([^'\"]+)['\"]", excl.group(2)))
            kids = [k for k in kids if not (k.tag == tag and k.get("name") in names)]
        return kids
    return suites


def _expand(root: ET.Element, base_dir: str, depth: int = 0) -> None:
    """Splice xi:include'd nodes in place (comments are already gone: ET drops them)."""
    for parent in list(root.iter()):
        kids = list(parent)
        if not any(k.tag in XI_TAGS for k in kids):
            continue
        new: list[ET.Element] = []
        for k in kids:
            if k.tag not in XI_TAGS:
                new.append(k)
                continue
            inc = (
                _included_root(_href_path(k.get("href", ""), base_dir), depth) if depth < _MAX_DEPTH else None
            )
            if inc is None:
                continue
            nodes = _select(inc, k.get("xpointer", ""))
            if parent.tag == "suite":  # a suite pulling in another's members
                for n in nodes:
                    new.extend(list(n) if n.tag == "suite" else [n])
            elif all(n.tag == "suite" for n in nodes):
                new.extend(nodes)
            else:  # members spliced at dictionary level: give them a suite
                s = ET.Element("suite", {"name": "Included Suite", "code": "????"})
                s.extend(nodes)
                new.append(s)
        for k in kids:
            parent.remove(k)
        parent.extend(new)


# ------------------------------------------------------------------------------ dictionary model
def _type_of(el: ET.Element | None) -> str:
    if el is None:
        return ""
    t = el.get("type")
    if t:
        return f"list of {t}" if el.get("list") == "yes" else t
    alts = [
        (f"list of {x.get('type', '')}" if x.get("list") == "yes" else x.get("type", ""))
        for x in el.findall("type")
        if x.get("type")
    ]
    return "|".join(alts)


def parse(root: ET.Element) -> Dictionary:
    """Build a :class:`Dictionary` from an (include-expanded) sdef root element."""
    cmds: dict[str, tuple[Command, bool]] = {}
    raw: dict[str, dict] = {}
    enums: dict[str, dict[str, str]] = {}
    for suite in root.iter("suite"):
        for c in suite.findall("command"):
            name = (c.get("name") or "").strip()
            if not name:
                continue
            dp = c.find("direct-parameter")
            params, optional = [], []
            for p in c.findall("parameter"):
                params.append((p.get("name", ""), p.get("code", ""), _type_of(p)))
                if p.get("optional") == "yes":
                    optional.append(p.get("name", ""))
            cmd = Command(
                name=name,
                code=c.get("code", ""),
                direct=_type_of(dp),
                params=tuple(params),
                direct_optional=dp is not None and dp.get("optional") == "yes",
                optional=tuple(optional),
                description=c.get("description", ""),
                direct_description=dp.get("description", "") if dp is not None else "",
                result=_type_of(c.find("result")),
            )
            hidden = c.get("hidden") == "yes"
            key = name.lower()
            prev = cmds.get(key)
            # Later suites override the standard ones; a hidden definition never hides a visible one.
            if prev is None or not hidden or prev[1]:
                cmds[key] = (cmd, hidden)
        for tag in ("class", "class-extension"):
            for k in suite.findall(tag):
                name = k.get("name") if tag == "class" else k.get("extends")
                if not name:
                    continue
                r = raw.setdefault(name, {"code": "", "inherits": "", "props": {}, "meta": {}, "elements": []})
                if tag == "class":
                    r["code"] = r["code"] or k.get("code", "")
                    r["inherits"] = r["inherits"] or k.get("inherits", "")
                for p in k.findall("property"):
                    pname = p.get("name", "")
                    if pname and pname not in r["props"]:
                        r["props"][pname] = (p.get("code", ""), _type_of(p))
                        r["meta"][pname] = _meta(p)
                for cont in k.findall("contents"):  # a class's text/contents behaves like a property
                    pname = cont.get("name") or "contents"
                    if pname not in r["props"]:
                        r["props"][pname] = (cont.get("code", ""), _type_of(cont))
                        r["meta"][pname] = _meta(cont)
                r["elements"] += [e.get("type", "") for e in k.findall("element") if e.get("type")]
        for e in suite.findall("enumeration"):
            ename = e.get("name", "")
            if ename:
                enums.setdefault(ename, {}).update(
                    {x.get("name", ""): x.get("code", "") for x in e.findall("enumerator")}
                )
    inherits = {n: r["inherits"] for n, r in raw.items() if r["inherits"]}
    class_props: dict[str, dict[str, tuple[str, str]]] = {}
    prop_meta: dict[str, dict[str, tuple[str, str]]] = {}
    for n in raw:
        flat: dict[str, tuple[str, str]] = {}
        meta: dict[str, tuple[str, str]] = {}
        for c in _chain(inherits, n):
            for pname, v in raw.get(c, {}).get("props", {}).items():
                flat.setdefault(pname, v)
                meta.setdefault(pname, raw[c]["meta"][pname])
        class_props[n] = flat
        prop_meta[n] = meta
    classes = {n: (r["code"], tuple(class_props[n])) for n, r in raw.items()}
    properties: dict[str, str] = {}
    for n in ["application", *[x for x in class_props if x != "application"]]:
        for pname, (code, _t) in class_props.get(n, {}).items():
            properties.setdefault(pname, code)
    return Dictionary(
        commands={k: v[0] for k, v in cmds.items()},
        classes=classes,
        properties=properties,
        enums=enums,
        class_props=class_props,
        elements={n: tuple(dict.fromkeys(r["elements"])) for n, r in raw.items() if r["elements"]},
        inherits=inherits,
        prop_meta=prop_meta,
    )


def _meta(el: ET.Element) -> tuple[str, str]:
    access = (el.get("access") or "rw").strip().lower()
    return (access if access in ("r", "w", "rw") else "rw"), el.get("description", "")


# ------------------------------------------------------------------------------ loading from a bundle
def _read_info(app_path: str) -> dict:
    for p in (
        os.path.join(app_path, "Contents", "Info.plist"),
        os.path.join(app_path, "WrappedBundle", "Info.plist"),
    ):
        try:
            if not stat.S_ISREG(os.stat(p).st_mode):
                continue
            with open(p, "rb") as f:
                d = plistlib.loads(f.read(4 << 20))
            return d if isinstance(d, dict) else {}
        except (OSError, plistlib.InvalidFileException, ValueError, TypeError, OverflowError):
            continue
    return {}


def sdef_path(app_path: str, info: dict | None = None) -> str | None:
    """The dictionary file an app declares (``OSAScriptingDefinition``), resolved in its bundle."""
    info = _read_info(app_path) if info is None else info
    res = os.path.join(app_path, "Contents", "Resources")
    name = info.get("OSAScriptingDefinition")
    if isinstance(name, str) and name.strip():
        name = name.strip()
        cands = [name] if name.lower().endswith(".sdef") else [name, f"{name}.sdef"]
        for c in cands:
            for d in (
                res,
                os.path.join(res, "Base.lproj"),
                os.path.join(res, "en.lproj"),
                os.path.join(res, "English.lproj"),
            ):
                p = os.path.join(d, c)
                if os.path.isfile(p):
                    return p
        return None
    if info.get("NSAppleScriptEnabled") in (True, 1, "1", "YES", "Yes", "yes", "true", "TRUE", "True"):
        try:
            sdefs = sorted(n for n in os.listdir(res) if n.lower().endswith(".sdef"))
        except OSError:
            return None
        stems = {str(info.get(k) or "").lower() for k in ("CFBundleExecutable", "CFBundleName")} - {""}
        pick = next((n for n in sdefs if n[:-5].lower() in stems), None) or (
            sdefs[0] if len(sdefs) == 1 else None
        )
        return os.path.join(res, pick) if pick else None
    return None


_LOAD_CACHE: dict[tuple[str, int, int], Dictionary | None] = {}
_LOAD_LOCK = threading.Lock()


def load_file(path: str) -> Dictionary | None:
    """Parse one sdef file (includes resolved relative to it). None if unreadable or empty."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (path, st.st_mtime_ns, st.st_size)
    with _LOAD_LOCK:
        if key in _LOAD_CACHE:
            return _LOAD_CACHE[key]
    try:
        root = _parse_file(path)
        _expand(root, os.path.dirname(os.path.realpath(path)))
        d: Dictionary | None = parse(root)
        if not d.commands and not d.classes:
            d = None
    except Exception:  # noqa: BLE001 - an odd dictionary means "not scriptable", never a crash
        d = None
    with _LOAD_LOCK:
        if len(_LOAD_CACHE) > 512:
            _LOAD_CACHE.clear()
        _LOAD_CACHE[key] = d
    return d


def load(app_path: str, info: dict | None = None) -> Dictionary | None:
    """The app's scripting dictionary, or None (not scriptable / legacy suite files / unreadable).
    Never launches the app."""
    p = sdef_path(app_path, info)
    return load_file(p) if p else None


# ------------------------------------------------------------------------------ capability tags
_TEXTY = {"text", "string", "unicode text", "styled text", "international text", "utf8 text", "utxt", "ctxt"}
_NOT_ITEM = _TEXTY | {
    "file", "alias", "url", "any", "list of file", "list of alias", "boolean", "integer", "real", "document",
    "number", "date", "record", "type", "location specifier", "list of text",
}  # fmt: skip
_URI_WORDS = re.compile(r"\b(uri|url|urls|id|identifier|link|permalink|path)\b", re.I)
_ITEM_WORDS = r"(track|item|song|episode|media|video|movie|stream|title|content)"


def _alts(t: str) -> list[str]:
    return [x.strip().lower() for x in t.split("|") if x.strip()]


def _is_text(t: str) -> bool:
    return any(a in _TEXTY for a in _alts(t))


def _is_item(t: str) -> bool:
    a = _alts(t)
    return bool(a) and not any(x in _NOT_ITEM for x in a)


def _role(c: Command) -> str | None:
    n, e = re.sub(r"\s+", " ", c.name.lower().strip()), c.event_id.lower()
    if e == "srch" or re.match(r"search\b", n):
        return "search"
    if e == "plps" or re.search(r"play\s*/?\s*pause|toggle\s+(play|playback)", n):
        return "playpause"
    if n == "play" or e == "play":
        return "play"
    if re.match(r"play\s+\w", n):  # play track / play uri / play item ...
        return "play_named"
    if n in ("open location", "open url", "openurl", "geturl", "get url") or c.code.lower() == "gurlgurl":
        return "open_location"
    return None


def _make(d: Dictionary) -> bool:
    return "make" in d.commands or any(c.code.lower() == "corecrel" for c in d.commands.values())


def _named_command(d: Dictionary, noun: str) -> bool:
    rx = re.compile(rf"^(create|make|new|add|compose)\s+(a\s+|an\s+|new\s+)?{noun}\b")
    return any(rx.match(k) for k in d.commands)


def _class_like(d: Dictionary, name_rx: str, codes: set[str]) -> list[str]:
    rx = re.compile(name_rx, re.I)
    return [n for n, (code, _p) in d.classes.items() if rx.match(n.strip()) or code in codes]


def _has_prop(d: Dictionary, klass: str, name_rx: str, codes: set[str] = frozenset()) -> bool:
    rx = re.compile(name_rx, re.I)
    props = d.class_props.get(klass)
    if props:
        return any(rx.match(p) or code in codes for p, (code, _t) in props.items())
    return any(rx.match(p) or d.properties.get(p) in codes for p in d.classes.get(klass, ("", ()))[1])


def _media_caps(d: Dictionary) -> set[str]:
    out: set[str] = set()
    roles: dict[str, list[Command]] = {}
    for c in d.commands.values():
        r = _role(c)
        if r:
            roles.setdefault(r, []).append(c)
    plays = roles.get("play", [])
    # Search + play: a text search whose hits the plain `play` command accepts.
    for s in roles.get("search", []):
        term = any(_is_text(p[2]) for p in s.params) or _is_text(s.direct)
        where = s.direct.split("|")[0].strip()  # the object searched (a library/playlist class) or text
        target_ok = not where or _is_text(where) or not d.elements or d.container_path(where) is not None
        if term and target_ok and any(_is_item(p.direct) for p in plays):
            out.add("media.search_play")
            break
    # Play an item by an app-specific id: a play command taking text that names a URI/URL/id.
    for c in plays + roles.get("play_named", []):
        uri_param = any(_is_text(t) and _URI_WORDS.search(n) for n, _code, t in c.params)
        if (
            _is_text(c.direct) and (_URI_WORDS.search(c.direct_description) or _role(c) == "play_named")
        ) or uri_param:
            out.add("media.play_uri")
            break
    if any(not c.direct or c.direct_optional for c in plays) or roles.get("playpause"):
        out.add("media.resume")
    app_props = d.class_props.get("application") or {
        p: (d.properties.get(p, ""), "") for p in d.classes.get("application", ("", ()))[1]
    }
    all_props = d.class_props or {
        k: {p: (d.properties.get(p, ""), "") for p in v[1]} for k, v in d.classes.items()
    }
    player_state = any(
        re.match(r"^(player|playback|play)\s*state$", p, re.I) or code == "pPlS"
        for p, (code, _t) in app_props.items()
    )
    # ... or a boolean "playing" (on the app or a document) - only in a dictionary that can also play
    playing_flag = bool(plays or roles.get("playpause")) and any(
        re.match(r"^(is\s+)?playing$", p, re.I) and t in ("", "boolean")
        for props in all_props.values()
        for p, (_c, t) in props.items()
    )
    if player_state or playing_flag:
        out.add("media.state")
    cur = next(
        (
            (p, t)
            for p, (code, t) in app_props.items()
            if code == "pTrk" or re.match(rf"^current\s+{_ITEM_WORDS}$", p, re.I)
        ),
        None,
    )
    if cur:
        out.add("media.now_playing")
        item_classes = [c for c in _chain(d.inherits, cur[1]) if c in d.classes] if cur[1] else []
        if not item_classes:  # no types known: any item-ish class
            item_classes = _class_like(d, rf"^{_ITEM_WORDS}$", set())
        uri_rx = r"^(?!.*\b(artwork|image|cover|icon|thumb\w*|album\s+art)\b).*\b(url|uri)$"
        if any(_has_prop(d, c, uri_rx) for c in item_classes):
            out.add("media.item_uri")
    return out


def _make_caps(d: Dictionary) -> set[str]:
    out: set[str] = set()
    make = _make(d)
    if (make and _class_like(d, r"^(\w+\s+)?notes?$", {"note"})) or _named_command(d, "note"):
        out.add("note.make")
    events = _class_like(d, r"^(calendar\s+)?(event|appointment)s?$", {"wrev"})
    calendars = _class_like(d, r"^calendars?$", {"wres"})
    dated = [e for e in events if _has_prop(d, e, r"^start(\s+date|\s+time)?$", {"wr1s"})]
    if (make and events and (calendars or dated)) or _named_command(d, r"(calendar\s+)?(event|appointment)"):
        out.add("event.make")
    todo = _class_like(d, r"^(reminder|to\s?-?\s?do|todo)s?$", {"remi"})
    tasks = [
        t
        for t in _class_like(d, r"^(\w+\s+)?tasks?$", set())
        if _has_prop(d, t, r"^(due|defer|deadline|remind)", set())
    ]
    if (make and (todo or tasks)) or _named_command(d, r"(reminder|to\s?-?\s?do|todo|task)"):
        out.add("reminder.make")
    return out


def _comm_caps(d: Dictionary) -> set[str]:
    out: set[str] = set()
    sends = [c for c in d.commands.values() if c.name.lower() == "send" or c.event_id.lower() == "send"]
    who = re.compile(r"(participant|buddy|chat|contact|recipient|person|handle|conversation|channel)", re.I)
    for c in sends:
        to = any(n.lower() in ("to", "recipient", "recipients") or who.search(t) for n, _code, t in c.params)
        body = _is_text(c.direct) or "file" in _alts(c.direct) or any(_is_text(t) for _n, _c, t in c.params)
        mail_object = bool(re.search(r"(outgoing|draft|mail|email)", c.direct, re.I))
        if to and body and not mail_object:
            out.add("message.send")
            break
    outgoing = _class_like(
        d, r"^(outgoing|new|draft)\s+(e-?mail\s+|mail\s+)?(message|mail|e-?mail)$", {"bcke"}
    )
    recipients = _class_like(d, r"^(to\s+)?recipients?$", {"trcp", "rcpt"})
    if (outgoing and (_make(d) or sends)) or (recipients and sends and _make(d)) or "mailto" in d.commands:
        out.add("mail.compose")
    return out


def _readable(d: Dictionary, klass: str, name_rx: str, codes: set[str] = frozenset()) -> bool:
    """The class has a property matching ``name_rx`` (or one of ``codes``) that can be read."""
    rx = re.compile(name_rx, re.I)
    props = d.class_props.get(klass) or {
        p: (d.properties.get(p, ""), "") for p in d.classes.get(klass, ("", ()))[1]
    }
    return any((rx.match(p) or code in codes) and d.prop_access(klass, p) != "w" for p, (code, _t) in props.items())


def _read_caps(d: Dictionary) -> set[str]:
    out: set[str] = set()
    text_rx = r"^(body|plain\s*text|text|contents?)$"
    if any(_readable(d, n, text_rx, {"body"}) for n in _class_like(d, r"^(\w+\s+)?notes?$", {"note"})):
        out.add("note.read")
    todo = _class_like(d, r"^(reminder|to\s?-?\s?do|todo)s?$", {"remi"})
    if any(_readable(d, t, r"^completed$", {"comb"}) for t in todo):
        out.add("reminder.read")
    events = _class_like(d, r"^(calendar\s+)?(event|appointment)s?$", {"wrev"})
    if any(_readable(d, e, r"^start(\s+date|\s+time)?$", {"wr1s"}) and d.container_path(e) for e in events):
        out.add("event.read")  # reachable through at least one container (a calendar), not only by id
    messages = _class_like(d, r"^(e-?mail\s+)?messages?$", {"mssg"})
    if any(_readable(d, m, r"^read(\s+status)?$", {"isrd"}) for m in messages):
        out.add("mail.read")
    if any(
        re.match(r"^unread\s+count$", p, re.I) or code == "mbuc"
        for props in d.class_props.values()
        for p, (code, _t) in props.items()
    ) or "mbuc" in d.properties.values():
        out.add("mail.unread_count")
    for c in d.commands.values():
        # Searches for text as such: no library/playlist object to search in (a player's search
        # takes one), and nothing but text required.
        term = _is_text(c.direct) or any(_is_text(t) for _n, _code, t in c.params)
        scoped = (c.direct and not _is_text(c.direct) and not c.direct_optional) or any(
            not _is_text(t) and n not in c.optional for n, _code, t in c.params
        )
        if _role(c) == "search" and term and not scoped:
            out.add("web.search")
            break
    return out


_CAPS_CACHE: dict[int, tuple[Dictionary, frozenset[str]]] = {}


def caps(d: Dictionary | None) -> frozenset[str]:
    """Generic capability tags (see :data:`ALL_CAPS`) the dictionary supports."""
    if d is None:
        return frozenset()
    hit = _CAPS_CACHE.get(id(d))
    if hit is not None and hit[0] is d:
        return hit[1]
    out = frozenset(_media_caps(d) | _make_caps(d) | _comm_caps(d))
    if len(_CAPS_CACHE) > 1024:
        _CAPS_CACHE.clear()
    _CAPS_CACHE[id(d)] = (d, out)  # holds d, so its id can't be reused while cached
    return out


_READ_CACHE: dict[int, tuple[Dictionary, frozenset[str]]] = {}


def read_caps(d: Dictionary | None) -> frozenset[str]:
    """Read tags (see :data:`READ_CAPS`) the dictionary supports."""
    if d is None:
        return frozenset()
    hit = _READ_CACHE.get(id(d))
    if hit is not None and hit[0] is d:
        return hit[1]
    out = frozenset(_read_caps(d))
    if len(_READ_CACHE) > 1024:
        _READ_CACHE.clear()
    _READ_CACHE[id(d)] = (d, out)
    return out
