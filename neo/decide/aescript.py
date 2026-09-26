"""AppleScript built from an app's own dictionary: the pieces every kind's script routes share.

Every property and class is addressed by the four-char code the app's own dictionary gives it
(checked by :func:`ok4` / :func:`ok8` before it goes into source), and every value travels only
in argv: nothing the user said is ever part of a script.

Until the helpers move here for good, running a script and asking macOS's Automation question go
through :mod:`neo.decide.media`'s own functions, looked up on each call (so its tests' patches
keep applying and media behaves exactly as before).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from neo.decide.run import Effect, Outcome

if TYPE_CHECKING:
    from neo.decide.inventory import App
    from neo.decide.sdef import Dictionary

__all__ = [
    "Slot",
    "SLOTS",
    "run",
    "control_gate",
    "ok4",
    "ok8",
    "slot_prop",
    "is_html",
    "rows",
    "fail",
    "US",
    "RS",
]

US, RS = "\x1f", "\x1e"  # field and row separators in a script's answer
STATUS = ("OK", "EMPTY", "NO_CONTAINER", "NO_PARTY")  # first field of every answer


def _media():
    from neo.decide import media

    return media


async def run(script: str, argv: list[str], timeout: float) -> tuple[int, str, str]:
    """(exit code, stdout, stderr) of osascript with ``argv`` (killed on timeout: code 124)."""
    return await _media()._osascript(script, argv, timeout)


async def control_gate(app: App) -> Outcome | None:
    """None when NEO may script ``app``; else why not (macOS asks the user once per app, here,
    where NEO waits for the answer)."""
    got = await _media()._control_gate(app)
    return None if got is None else Outcome(False, got.text, reason=got.reason)


def ok4(code: str | None) -> bool:
    return _media()._ok4(code)


def ok8(code: str | None) -> bool:
    return _media()._ok8(code)


# ---- slots: which property of a class holds what ------------------------------------------------


@dataclass(frozen=True)
class Slot:
    """A kind of value, the de-facto codes apps give its property, and a name pattern for apps
    that use their own code. ``write=False``: read back only, never set."""

    kind: Literal["text", "date", "bool", "int"]
    codes: frozenset[str]
    names: str
    write: bool = True


def _slot(kind: Literal["text", "date", "bool", "int"], codes: set[str], names: str, write: bool = True) -> Slot:
    return Slot(kind, frozenset(codes), names, write)


SLOTS: dict[str, Slot] = {
    "title": _slot("text", {"pnam", "wr11"}, r"^(name|title|summary)$"),
    "body": _slot("text", {"body", "wr12"}, r"^(body|notes?|description|comment)$"),
    "plain": _slot("text", {"text"}, r"^plain\s*text$", write=False),
    "start": _slot("date", {"wr1s"}, r"^start(\s+date|\s+time)?$"),
    "end": _slot("date", {"wr5s"}, r"^end(\s+date|\s+time)?$"),
    "all_day": _slot("bool", {"wrad"}, r"^all\s*-?\s*day(\s+event)?$"),
    "location": _slot("text", {"wr14"}, r"^location$"),
    "due": _slot("date", {"dued"}, r"^due(\s+date)?$"),
    "due_day": _slot("date", {"adue"}, r"^all\s*-?\s*day\s+due(\s+date)?$"),
    "alert": _slot("date", {"rmdt"}, r"^(remind\s+me|alert|alarm)(\s+date)?$"),
    "priority": _slot("int", {"prio"}, r"^priority$"),
    "subject": _slot("text", {"subj"}, r"^subject$"),
    "content": _slot("text", {"ctnt"}, r"^content$"),
    "visible": _slot("bool", {"pvis"}, r"^visible$"),
    "address": _slot("text", {"radd"}, r"^(address|e-?mail(\s+address)?)$"),
    "handle": _slot("text", {"hndl"}, r"^handle$"),
    "id": _slot("text", {"ID  "}, r"^(id|uid)$", write=False),
    "created": _slot("date", {"ascd"}, r"^creation\s+date$", write=False),
    "read": _slot("bool", {"isrd"}, r"^read(\s+status)?$", write=False),
    "sender": _slot("text", {"sndr"}, r"^sender$", write=False),
    "received": _slot("date", {"rdrc"}, r"^date\s+received$", write=False),
    "sent_at": _slot("date", {"drcv"}, r"^date\s+sent$", write=False),
    "recur": _slot("text", {"wr15"}, r"^recurrence$", write=False),
    "exdates": _slot("date", {"wr2s"}, r"^excluded\s+dates$", write=False),
    "writable": _slot("bool", {"wr05"}, r"^(writable|editable)$", write=False),
}


def slot_prop(d: Dictionary, klass: str, slot: str, *, write: bool = False) -> tuple[str, str] | None:
    """(property name, code) of ``klass`` that fills ``slot``: a de-facto code first, then the name
    pattern. With ``write``, only a slot meant to be set and a property the dictionary lets NEO
    set (a read-only property is never written)."""
    spec = SLOTS[slot]
    if write and not spec.write:
        return None
    props = d.class_props.get(klass) or {}
    rx = re.compile(spec.names, re.I)
    for match in (lambda p, c: c in spec.codes, lambda p, c: bool(rx.match(p.strip()))):
        for p, (code, _t) in props.items():
            if match(p, code) and ok4(code) and (not write or "w" in d.prop_access(klass, p)):
                return p, code
    return None


def is_html(d: Dictionary, klass: str, prop: str) -> bool:
    """The dictionary says this text property holds HTML (so plain text must be escaped into it)."""
    return bool(re.search(r"\bHTML\b", d.prop_desc(klass, prop)))


def rows(out: str) -> tuple[str, list[list[str]]]:
    """(status token, rows of fields) of a script's answer; ('', []) for anything else."""
    recs = out.split(RS)
    head = recs[0].split(US)
    if head[0].strip() not in STATUS:
        return "", []
    first = head[1:]
    return head[0].strip(), ([first] if first else []) + [r.split(US) for r in recs[1:] if r]


# ---- errors -----------------------------------------------------------------------------------

_AE_CODE = re.compile(r"\((-?\d+)\)\s*$")


def fail(app_name: str, code: int, err: str, *, committed: bool) -> Outcome:
    """What an osascript failure means. Before the commit point nothing happened (NONE, safe to
    try another way); after it only an event that surely never ran is NONE - anything else may
    have had its effect (UNKNOWN: never a second try)."""
    m = _AE_CODE.search(err or "")
    n = int(m.group(1)) if m else 0
    if n in (-1743, -1744):  # errAEEventNotPermitted / would need the user's consent
        return Outcome(False, f"{app_name} needs your OK before NEO can control it", reason="permission")
    if n == -600:  # procNotFound: the event never reached the app
        return Outcome(False, f"{app_name} isn't running")
    if n == -1728:  # errAENoSuchObject: the object never resolved, so nothing was done to it
        return Outcome(False, f"{app_name} couldn't find it", reason="not_found")
    if committed:
        return Outcome(
            False, f"I may have done it in {app_name} but couldn't check", reason="unconfirmed", effect=Effect.UNKNOWN
        )
    if code == 124 or code < 0 or n in (-1712, -609):  # timed out, killed, connection lost
        return Outcome(False, f"{app_name} didn't answer in time", reason="timeout")
    if n in (-10000, -1708):  # the handler failed / doesn't understand: this route is wrong here
        return Outcome(False, f"{app_name} doesn't take that", reason="route_bad")
    return Outcome(False, f"{app_name}'s scripting failed ({n or code})")
