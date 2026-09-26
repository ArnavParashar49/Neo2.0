"""What an app bundle declares beyond its scripting dictionary: evidence of what it is for.

Read from the bundle only, never from the running app:

* ``intents``: SiriKit intents its extensions handle (``IntentsSupported``);
* ``ai_schemas``: App Intents it ships (``Metadata.appintents/extract.actionsdata``), as
  ``schema:<domain>.<Name>`` for an assistant schema and ``protocol:<Name>`` for a system protocol;
* ``share_text``: a share extension whose activation rule accepts text;
* ``services``: its Services-menu entries (title, the pasteboard types it takes);
* ``usage``: the privacy usage keys it declares (``NS...UsageDescription``);
* ``doc_based``: its document types name a document class (a document-based app).

None of it can be invoked from NEO's process: it is evidence for choosing an app, nothing more.
Every file is size-capped and a malformed one is skipped; nothing is launched.
"""

from __future__ import annotations

import json
import os
import plistlib
import re
import stat
from dataclasses import dataclass
from functools import lru_cache
from xml.parsers.expat import ExpatError

__all__ = ["Evidence", "read", "responsible_usage", "NONE"]

MAX_PLIST = 1 << 20  # an extension's Info.plist
MAX_ACTIONS = 4 << 20  # App Intents metadata JSON
MAX_EXTENSIONS = 64  # extensions read per app
MAX_SERVICES = 32
_EXT_DIRS = ("PlugIns", "Extensions")
# a Mac bundle keeps it in Resources/, an iPhone/iPad bundle at its top level
_ACTIONS = (
    os.path.join("Resources", "Metadata.appintents", "extract.actionsdata"),
    os.path.join("Metadata.appintents", "extract.actionsdata"),
)
_USAGE_KEY = re.compile(r"^NS\w+UsageDescription$")
# A predicate-form activation rule that takes text: a text type among what it conforms to, or
# a rule that accepts anything.
_TEXT_RULE = re.compile(
    r"\bTRUEPREDICATE\b|UTI-CONFORMS-TO\s+\"public\.(?:plain-text|utf8-plain-text|text)\"", re.I
)


@dataclass(frozen=True)
class Evidence:
    intents: tuple[str, ...] = ()
    ai_schemas: tuple[str, ...] = ()
    share_text: bool = False
    services: tuple[tuple[str, tuple[str, ...]], ...] = ()
    usage: frozenset[str] = frozenset()
    doc_based: bool = False


NONE = Evidence()


def _list(v: object) -> list:
    return v if isinstance(v, list) else []


def _dict(v: object) -> dict:
    return v if isinstance(v, dict) else {}


def _usage(info: dict) -> frozenset[str]:
    return frozenset(k for k, v in info.items() if isinstance(k, str) and _USAGE_KEY.match(k) and v)


def _regular(path: str, limit: int) -> bytes | None:
    """The file's bytes if it is a regular file no bigger than ``limit``, else None."""
    try:
        st = os.stat(path)
        if not stat.S_ISREG(st.st_mode) or st.st_size > limit:
            return None
        with open(path, "rb") as f:
            raw = f.read(limit + 1)
    except OSError:
        return None
    return raw if len(raw) <= limit else None


def _plist(path: str) -> dict:
    raw = _regular(path, MAX_PLIST)
    if raw is None:
        return {}
    try:
        d = plistlib.loads(raw)
    except (plistlib.InvalidFileException, ExpatError, ValueError, TypeError, OverflowError):
        return {}
    return d if isinstance(d, dict) else {}


def _actions(contents: str) -> list[str]:
    """``schema:<domain>.<Name>`` and ``protocol:<Name>`` entries of one bundle's App Intents."""
    raw = next((r for a in _ACTIONS if (r := _regular(os.path.join(contents, a), MAX_ACTIONS)) is not None), None)
    if raw is None:
        return []
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return []
    acts = data.get("actions") if isinstance(data, dict) else None
    out: list[str] = []
    for a in acts.values() if isinstance(acts, dict) else ():
        if not isinstance(a, dict):
            continue
        for s in _list(a.get("assistantDefinedSchemas")):
            if isinstance(s, dict) and isinstance(s.get("domain"), str) and isinstance(s.get("name"), str):
                out.append(f"schema:{s['domain']}.{s['name']}")
        for p in _list(a.get("systemProtocols")):
            if isinstance(p, str) and p.strip():
                out.append("protocol:" + p.rsplit(".", 1)[-1])  # reverse-DNS: the last part names it
    return out


def _accepts_text(rule: object) -> bool:
    if isinstance(rule, dict):
        v = rule.get("NSExtensionActivationSupportsText")
        return v is True or v == 1 or v in ("1", "YES", "true")
    return isinstance(rule, str) and bool(_TEXT_RULE.search(rule))


def _extensions(contents: str) -> list[str]:
    """Paths of the extension bundles' Contents folders, in a stable order, capped."""
    out: list[str] = []
    for d in _EXT_DIRS:
        try:
            names = sorted(n for n in os.listdir(os.path.join(contents, d)) if n.endswith(".appex"))
        except OSError:
            continue
        out += [os.path.join(contents, d, n, "Contents") for n in names]
    return out[:MAX_EXTENSIONS]


def _services(info: dict) -> tuple[tuple[str, tuple[str, ...]], ...]:
    out = []
    for s in _list(info.get("NSServices")):
        if not isinstance(s, dict):
            continue
        item = s.get("NSMenuItem")
        title = item.get("default") if isinstance(item, dict) else None
        if not isinstance(title, str) or not title.strip():
            continue
        types = tuple(t for t in _list(s.get("NSSendTypes")) if isinstance(t, str))
        out.append((title.strip(), types))
    return tuple(out[:MAX_SERVICES])


def read(contents: str, info: dict) -> Evidence:
    """Evidence for the bundle whose ``Contents`` folder is ``contents`` (its Info.plist already
    parsed as ``info``)."""
    intents: dict[str, None] = {}
    schemas: dict[str, None] = dict.fromkeys(_actions(contents))
    share_text = False
    for ext in _extensions(contents):
        ex = _plist(os.path.join(ext, "Info.plist"))
        ns = _dict(ex.get("NSExtension"))  # an app extension; ExtensionKit ones say it differently
        ekit = _dict(ex.get("EXAppExtensionAttributes"))
        attrs = _dict(ns.get("NSExtensionAttributes"))
        for src in (attrs, ekit):
            intents.update(dict.fromkeys(i for i in _list(src.get("IntentsSupported")) if isinstance(i, str)))
        point = str(ns.get("NSExtensionPointIdentifier") or ekit.get("EXExtensionPointIdentifier") or "")
        if point.endswith(".share-services") and _accepts_text(attrs.get("NSExtensionActivationRule")):
            share_text = True
        schemas.update(dict.fromkeys(_actions(ext)))
    return Evidence(
        intents=tuple(intents),
        ai_schemas=tuple(schemas),
        share_text=share_text,
        services=_services(info),
        usage=_usage(info),
        doc_based=any(_dict(d).get("NSDocumentClass") for d in _list(info.get("CFBundleDocumentTypes"))),
    )


@lru_cache(maxsize=1)
def responsible_usage() -> frozenset[str]:
    """The usage keys declared by the app macOS holds responsible for NEO (NEO.app, or the
    terminal it runs from): macOS asks the user on that app's behalf, and asking for something it
    doesn't declare may end the process, so nothing is requested without the key."""
    from neo.decide import media

    exe = media._proc_path(media.responsible_pid(os.getpid()))
    i = exe.find(".app/")
    if i < 0:
        return frozenset()
    return _usage(_plist(os.path.join(exe[: i + 4], "Contents", "Info.plist")))
