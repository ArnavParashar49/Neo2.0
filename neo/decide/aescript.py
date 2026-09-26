"""AppleScript built from an app's own dictionary: the pieces every kind's script routes share.

Every property and class is addressed by the four-char code the app's own dictionary gives it
(checked by :func:`ok4` / :func:`ok8` before it goes into source), and every value travels only
in argv: nothing the user said is ever part of a script.

Until the helpers move here for good, running a script and asking macOS's Automation question go
through :mod:`neo.decide.media`'s own functions, looked up on each call (so its tests' patches
keep applying and media behaves exactly as before).
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

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
    "Item",
    "CreateRoute",
    "ListRoute",
    "create_route",
    "render_create",
    "readback",
    "search_back",
    "containers",
    "list_route",
    "render_list",
    "iso",
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


# ---- making an item: a note, a reminder, an event ------------------------------------------------
# One script makes the item (the commit point: once it ran the item may exist), then others read
# it back by its id, find it again after a timeout (search-back), or list the containers a user
# may name. Containers are addressed by their id, else by their name - both only ever from argv.

Item = Literal["note", "reminder", "event"]
# the item class: a de-facto code, else a name pattern (the same vocabulary as sdef's make tags)
_ITEMS: dict[str, tuple[frozenset[str], str]] = {
    "note": (frozenset({"note"}), r"^(\w+\s+)?notes?$"),
    "reminder": (frozenset({"remi"}), r"^(reminder|to\s?-?\s?do|todo|(\w+\s+)?task)s?$"),
    "event": (frozenset({"wrev"}), r"^(calendar\s+)?(event|appointment)s?$"),
}
# read back after a make, in this order (whatever the class has)
_READS = ("title", "start", "end", "all_day", "due", "due_day", "alert", "body", "plain")


@dataclass(frozen=True)
class CreateRoute:
    """How to make one kind of item in one app, every code from the app's own dictionary."""

    item: str
    code: str  # the item class
    container: str  # the class holding the items ('' = the application itself)
    container_key: str  # the container's property its argv value is matched against (id, else name)
    container_name: str  # the container's name property (what the user calls it)
    listable: bool  # the application holds every container directly (so they can be listed)
    writable_key: str  # the container's "may add to it" flag ('' = none)
    default: tuple[str, ...]  # property codes from the application down to the default container
    sets: tuple[tuple[str, str, str], ...]  # (slot, code, value kind) NEO may write, in the wanted order
    reads: tuple[tuple[str, str, str], ...]  # (slot, code, value kind) read back
    id_code: str  # the item's id property ('' = it can't be found again by id)
    created_code: str  # its creation date ('' = none kept)
    html: bool  # its body holds HTML


def _cls(code: str) -> str:
    return f"«class {code}»"


def _prop(code: str) -> str:
    return f"«property {code}»"


def _item_class(d: Dictionary, item: str) -> str | None:
    codes, names = _ITEMS[item]
    rx = re.compile(names, re.I)
    by_code = [n for n, (c, _p) in d.classes.items() if c in codes]
    by_name = [n for n, (c, _p) in d.classes.items() if rx.match(n.strip()) and n not in by_code]
    for n in by_code + by_name:
        if ok4(d.class_code(n)) and slot_prop(d, n, "title", write=True):
            return n
    return None


def _lineage(d: Dictionary, name: str) -> list[str]:
    out: list[str] = []
    while name and name not in out:
        out.append(name)
        name = d.inherits.get(name, "")
    return out


def _holds(d: Dictionary, container: str, klass: str) -> bool:
    """``container`` (or a class it inherits from) has elements of ``klass`` (or of its superclass)."""
    kinds = set(_lineage(d, klass))
    return any(el in kinds for c in _lineage(d, container) for el in d.elements.get(c, ()))


def _default_chain(d: Dictionary, klass: str) -> tuple[tuple[str, ...], str] | None:
    """The application's own ``default …`` properties (to depth 2) leading to a class that holds
    the items: (codes from the application down, that class). The most specific wins: a longer
    chain (a default folder of the default account), then a class the others hold."""
    found: list[tuple[tuple[str, ...], str]] = []
    frontier: list[tuple[tuple[str, ...], str]] = [((), "application")]
    for _depth in range(2):
        nxt = []
        for path, cls in frontier:
            for p, (code, typ) in (d.class_props.get(cls) or {}).items():
                if not re.match(r"^default\b", p.strip(), re.I) or typ not in d.classes or not ok4(code):
                    continue
                step = ((*path, code), typ)
                nxt.append(step)
                if _holds(d, typ, klass) and ok4(d.class_code(typ)):
                    found.append(step)
        frontier = nxt
    if not found:
        return None
    finals = {cls for _p, cls in found}
    return max(found, key=lambda f: (len(f[0]), sum(f[1] in d.elements.get(o, ()) for o in finals)))


def create_route(d: Dictionary | None, item: Item, want: tuple[str, ...]) -> CreateRoute | None:
    """How to make ``item`` with this dictionary, setting the slots in ``want`` that its class has
    (and NEO may write); None when the app can't make one."""
    if d is None or not ("make" in d.commands or any(c.code.lower() == "corecrel" for c in d.commands.values())):
        return None
    klass = _item_class(d, item)
    if klass is None:
        return None
    chain = _default_chain(d, klass)
    holders = [c for c in d.classes if c != "application" and _holds(d, c, klass) and ok4(d.class_code(c))]
    if chain is not None:
        container = chain[1]
    elif holders:
        # the most specific holder: the one the others hold (a list inside an account)
        container = max(holders, key=lambda c: sum(c in d.elements.get(o, ()) for o in holders))
    elif _holds(d, "application", klass):
        container = ""
    else:
        return None
    key = name = ""
    if container:
        named = slot_prop(d, container, "title")
        found = slot_prop(d, container, "id") or named
        if found is None:
            return None  # a container NEO couldn't address (or name) is no way in
        key, name = found[1], named[1] if named else ""
    listable = bool(container) and container in d.elements.get("application", ())
    if container and chain is None and not listable:
        return None  # a container that is neither the default nor one NEO can name
    writable = (slot_prop(d, container, "writable") or ("", ""))[1] if container else ""
    sets = tuple(
        (s, hit[1], SLOTS[s].kind) for s in dict.fromkeys(("title", *want)) if (hit := slot_prop(d, klass, s, write=True))
    )
    reads = tuple((s, hit[1], SLOTS[s].kind) for s in _READS if (hit := slot_prop(d, klass, s)))
    body = slot_prop(d, klass, "body", write=True)
    return CreateRoute(
        item=item,
        code=d.class_code(klass),
        container=d.class_code(container) if container else "",
        container_key=key,
        container_name=name,
        listable=listable,
        writable_key=writable,
        default=chain[0] if chain else (),
        sets=sets,
        reads=reads,
        id_code=(slot_prop(d, klass, "id") or ("", ""))[1],
        created_code=(slot_prop(d, klass, "created") or ("", ""))[1],
        html=body is not None and is_html(d, klass, body[0]),
    )


def iso(dt: datetime) -> str:
    """A date as the scripts take and give it (local, to the second)."""
    return dt.replace(microsecond=0, tzinfo=None).isoformat()


# Dates go in as ISO text and are rebuilt field by field (locale-independent), and come back as
# «class isot» text.
_DATE_HANDLER = """
on neoDate(s)
	set d to current date
	set day of d to 1
	set year of d to (text 1 thru 4 of s) as integer
	set month of d to (text 6 thru 7 of s) as integer
	set day of d to (text 9 thru 10 of s) as integer
	set time of d to ((text 12 thru 13 of s) as integer) * 3600 + ((text 15 thru 16 of s) as integer) * 60 + ((text 18 thru 19 of s) as integer)
	return d
end neoDate

on neoText(v, k)
	try
		if v is missing value then return ""
		if k is "date" then return (v as «class isot») as string
		return v as text
	on error
		return ""
	end try
end neoText
"""
_HEAD = (
    "on run argv\n\tset bid to item 1 of argv\n\tset cid to item 2 of argv\n"
    "\tset US to character id 31\n\tset RS to character id 30\n"
)


def _container(r: CreateRoute) -> str:
    """Script lines that set ``c`` to the container: the one whose id (or name) is argv's cid,
    else the default; NO_CONTAINER when there is none."""
    if not r.container:
        return "\t\tset c to it\n"  # the application holds the items itself
    default = " of ".join(_prop(code) for code in reversed(r.default)) if r.default else ""
    lines = "\t\tif cid is \"\" then\n"
    if default:
        lines += f"\t\t\tset c to {default}\n\t\t\tif c is missing value then return \"NO_CONTAINER\"\n"
    else:
        lines += '\t\t\treturn "NO_CONTAINER"\n'
    if r.listable:
        lines += (
            f"\t\telse\n\t\t\tset cs to (every {_cls(r.container)} whose {_prop(r.container_key)} is cid)\n"
            '\t\t\tif cs is {} then return "NO_CONTAINER"\n\t\t\tset c to item 1 of cs\n'
        )
    else:
        lines += "\t\telse\n\t\t\treturn \"NO_CONTAINER\"\n"
    return lines + "\t\tend if\n"


def _value(kind: str, n: int) -> str:
    v = f"(item {n} of argv)"
    return {"date": f"(my neoDate{v})", "bool": f"({v} is \"true\")", "int": f"({v} as integer)"}.get(kind, v)


def _arg(kind: str, v: Any) -> str:
    if kind == "date":
        return iso(v)
    if kind == "bool":
        return "true" if v else "false"
    return str(v)


def render_create(r: CreateRoute, bid: str, values: dict[str, Any], container_id: str = "") -> tuple[str, list[str]]:
    """(script, argv) making the item with ``values`` (slot -> value; the slots the route can
    set) in the container whose id is ``container_id`` ('' = the default). A body that holds
    HTML gets the text escaped, line by line. Answers OK + the new item's id + the container's
    name, each '' when it can't be read: reading them never fails the script, so an error it
    gives is the make's own."""
    argv = [bid, container_id]
    fields = []
    for slot, code, kind in r.sets:
        if slot not in values or values[slot] is None:
            continue
        v = values[slot]
        if slot == "body" and r.html:
            v = note_html(str(v))
        argv.append(_arg(kind, v))
        fields.append(f"{_prop(code)}:{_value(kind, len(argv))}")
    after = '\t\tset newId to ""\n\t\tset cName to ""\n'
    reads = [("newId", r.id_code, "o"), ("cName", r.container_name if r.container else "", "c")]
    for var, code, of in reads:
        if code:
            after += f'\t\ttry\n\t\t\tset {var} to my neoText(get {_prop(code)} of {of}, "text")\n\t\tend try\n'
    script = (
        f"{_HEAD}\ttell application id bid\n{_container(r)}"
        f"\t\ttell c\n\t\t\tset o to make new {_cls(r.code)} with properties {{{', '.join(fields)}}}\n\t\tend tell\n"
        f'{after}\t\treturn "OK" & US & newId & US & cName\n\tend tell\nend run\n{_DATE_HANDLER}'
    )
    return script, argv


def _fields(r: CreateRoute, slots: tuple[tuple[str, str, str], ...]) -> str:
    return "".join(f'\t\tset out to out & US & my neoText(get {_prop(code)} of o, "{kind}")\n' for _s, code, kind in slots)


def readback(r: CreateRoute, bid: str, oid: str, container_id: str = "") -> tuple[str, list[str]] | None:
    """(script, argv) re-finding the item by its id and reading its slots (``r.reads`` order):
    OK + the values, EMPTY when it isn't there. None when the class has no id."""
    if not r.id_code:
        return None
    script = (
        f"{_HEAD}\tset oid to item 3 of argv\n\ttell application id bid\n{_container(r)}"
        f"\t\tset hits to (every {_cls(r.code)} of c whose {_prop(r.id_code)} is oid)\n"
        '\t\tif hits is {} then return "EMPTY"\n\t\tset o to item 1 of hits\n\t\tset out to "OK"\n'
        f"{_fields(r, r.reads)}\t\treturn out\n\tend tell\nend run\n{_DATE_HANDLER}"
    )
    return script, [bid, container_id, oid]


def search_back(
    r: CreateRoute, bid: str, title: str, since: datetime, start: datetime | None = None, container_id: str = ""
) -> tuple[str, list[str]] | None:
    """(script, argv) finding an item made after a make that timed out: this title, created at or
    after ``since`` (or, for a class that keeps no creation date, starting at ``start``). OK + its
    id, else EMPTY. None when neither can tell a new item from an old one."""
    title_code = next((code for s, code, _k in r.sets if s == "title"), "")
    start_code = next((code for s, code, _k in r.sets if s == "start"), "")
    if r.created_code:
        when, test = since, f"(get {_prop(r.created_code)} of o) ≥ t"
    elif start_code and start is not None:
        when, test = start, f"(get {_prop(start_code)} of o) = t"
    else:
        return None
    ident = f'my neoText(get {_prop(r.id_code)} of o, "text")' if r.id_code else '""'
    script = (
        f"{_HEAD}\tset t to my neoDate(item 4 of argv)\n\ttell application id bid\n{_container(r)}"
        f"\t\tset hits to (every {_cls(r.code)} of c whose {_prop(title_code)} is (item 3 of argv))\n"
        "\t\trepeat with k from 1 to count of hits\n\t\t\tset o to item k of hits\n"
        f'\t\t\tif {test} then return "OK" & US & {ident}\n\t\tend repeat\n'
        f'\t\treturn "EMPTY"\n\tend tell\nend run\n{_DATE_HANDLER}'
    )
    return script, [bid, container_id, title, iso(when)]


def containers(r: CreateRoute, bid: str) -> tuple[str, list[str]] | None:
    """(script, argv) listing the containers a user may name: rows of (id or name, name,
    writable). Read-only. None when the application doesn't hold them directly."""
    if not r.listable:
        return None
    every = f"every {_cls(r.container)}"
    writable = f"(get {_prop(r.writable_key)} of {every})" if r.writable_key else "{}"
    names = f"(get {_prop(r.container_name)} of {every})" if r.container_name else "ks"
    script = (
        f"{_HEAD}\ttell application id bid\n"
        f"\t\tset ks to (get {_prop(r.container_key)} of {every})\n"
        f"\t\tset ns to {names}\n"
        f"\t\tset ws to {writable}\n\tend tell\n"
        '\tset out to "OK"\n\trepeat with i from 1 to count of ks\n'
        '\t\tset w to "true"\n\t\tif (count of ws) ≥ i then set w to my neoText(item i of ws, "bool")\n'
        '\t\tset out to out & RS & my neoText(item i of ks, "text") & US & my neoText(item i of ns, "text") & US & w\n'
        f"\tend repeat\n\treturn out\nend run\n{_DATE_HANDLER}"
    )
    return script, [bid, ""]


# ---- listing events in a window -------------------------------------------------------------------

_EVENT_ROW = ("id", "title", "start", "end", "all_day", "recur", "exdates")


@dataclass(frozen=True)
class ListRoute:
    """How to list one calendar's events overlapping a window, by the app's own codes."""

    code: str  # the event class
    container: str  # the calendar class
    container_key: str  # the calendar's id (or name) property
    cols: tuple[tuple[str, str, str], ...]  # (slot, code, value kind) per row field, in _EVENT_ROW order


def list_route(d: Dictionary | None) -> ListRoute | None:
    """How to read the events of each calendar in a window; None when the dictionary can't say
    when an event starts and ends, or its calendars can't be listed."""
    r = create_route(d, "event", ())
    klass = _item_class(d, "event") if d is not None else None
    if r is None or klass is None or not r.listable:
        return None
    cols = []
    for s in _EVENT_ROW:
        hit = slot_prop(d, klass, s)
        if hit is None and s in ("id", "title", "start", "end"):
            return None
        if hit is not None:
            cols.append((s, hit[1], SLOTS[s].kind))
    return ListRoute(r.code, r.container, r.container_key, tuple(cols))


def render_list(lr: ListRoute, bid: str, container_id: str, t0: datetime, t1: datetime) -> tuple[str, list[str]]:
    """(script, argv) reading one calendar's events that overlap [t0, t1), plus the repeating ones
    that began before it (expanded by the caller). Rows follow ``lr.cols``; OK first, NO_CONTAINER
    when the calendar is gone. The date lists (excluded dates) come as ISO text joined by commas."""
    code = dict((s, c) for s, c, _k in lr.cols)
    start, end = _prop(code["start"]), _prop(code["end"])
    every = f"every {_cls(lr.code)} of c"
    queries = [f"({every} whose {start} < t1 and {end} > t0)"]
    if "recur" in code:
        rec = _prop(code["recur"])
        queries.append(f"({every} whose {start} < t0 and {rec} is not missing value and {rec} is not \"\")")
    reads = ""
    for n, q in enumerate(queries):
        gets = "".join(f"\t\t\tset v{n}_{i} to (get {_prop(c)} of {q})\n" for i, (_s, c, _k) in enumerate(lr.cols))
        if n == 0:
            reads += gets.replace("\t\t\t", "\t\t")
        else:  # an app that can't filter on repeats still gives the window's own events
            empty = "".join(f"\t\t\tset v{n}_{i} to {{}}\n" for i in range(len(lr.cols)))
            reads += f"\t\ttry\n{gets}\t\ton error\n{empty}\t\tend try\n"
    rows = ""
    for n in range(len(queries)):
        rows += f"\trepeat with j from 1 to count of v{n}_0\n\t\tset row to \"\"\n"
        for i, (s, _c, kind) in enumerate(lr.cols):
            sep = "" if i == 0 else " & US"
            if s == "exdates":
                rows += (
                    f"\t\tset xs to {{}}\n\t\ttry\n\t\t\trepeat with x in (item j of v{n}_{i})\n"
                    '\t\t\t\tset end of xs to my neoText(contents of x, "date")\n\t\t\tend repeat\n\t\tend try\n'
                    "\t\tset AppleScript's text item delimiters to \",\"\n"
                    f"\t\tset row to row{sep} & (xs as text)\n"
                )
            else:
                rows += f'\t\tset row to row{sep} & my neoText(item j of v{n}_{i}, "{kind}")\n'
        rows += "\t\tset out to out & RS & row\n\tend repeat\n"
    script = (
        f"{_HEAD}\tset t0 to my neoDate(item 3 of argv)\n\tset t1 to my neoDate(item 4 of argv)\n"
        f"\ttell application id bid\n\t\tset cs to (every {_cls(lr.container)} whose {_prop(lr.container_key)} is cid)\n"
        '\t\tif cs is {} then return "NO_CONTAINER"\n\t\tset c to item 1 of cs\n'
        f"{reads}\tend tell\n\tset out to \"OK\"\n{rows}\treturn out\nend run\n{_DATE_HANDLER}"
    )
    return script, [bid, container_id, iso(t0), iso(t1)]


# ---- notes and the inbox (write_note, read_mail) ------------------------------------------------------
# Scripts for making a note and for listing unread mail, built like the rest: codes from the app's
# own dictionary, values in argv only (item 1 is always the bundle id), answers in the status /
# separator protocol above.

__all__ += [
    "NoteRoute",
    "note_route",
    "note_html",
    "render_note",
    "render_note_search",
    "render_folders",
    "UnreadRoute",
    "unread_route",
    "render_unread",
    "may_script",
]


def may_script(bundle_id: str) -> bool:
    """NEO may already script this app and it is running: macOS answers without asking the
    user, and nothing is launched (a check for work nobody is waiting on)."""
    m = _media()
    return m.automation(bundle_id, False) == m.AE_ALLOWED


def _elements(d: Dictionary, klass: str) -> set[str]:
    """Element classes of ``klass``, its superclasses' included."""
    out: set[str] = set()
    seen: set[str] = set()
    while klass and klass not in seen:
        seen.add(klass)
        out.update(d.elements.get(klass, ()))
        klass = d.inherits.get(klass, "")
    return out


def _class_named(d: Dictionary, name_rx: str, code: str) -> str:
    """The class with this de-facto code, else the first whose name matches; '' if none."""
    rx = re.compile(name_rx, re.I)
    hits = [n for n, (c, _p) in d.classes.items() if (c == code or rx.match(n.strip())) and ok4(c)]
    return next((n for n in hits if d.class_code(n) == code), hits[0] if hits else "")


@dataclass(frozen=True)
class NoteRoute:
    """How an app's dictionary makes a note: the make event and its parameters, the note class
    and the properties to set and to read back, where a new note goes, and the folders the user
    can name. Every value is a four- (or eight-) char code checked by :func:`ok4` / :func:`ok8`."""

    klass: str
    make: str
    p_new: str
    p_at: str
    p_props: str
    title: str  # '' when the class has no writable title
    body: str
    html: bool  # the body holds HTML
    ident: str  # '' = a new note can't be re-resolved by id
    text: str  # what is read back: its plain text, else the body itself
    created: str  # '' = found again by title
    chain: tuple[str, ...]  # the default container, outermost property first; () = the application
    folder: str  # a class the user can name notes into, and its name and id properties
    folder_name: str
    folder_id: str
    folders_listed: bool  # the application lists those folders itself


def note_route(d: Dictionary | None) -> NoteRoute | None:
    """The dictionary's way to make a note with a body, or None when it has none."""
    if d is None:
        return None
    make = d.command("make") or next((c for c in d.commands.values() if c.code.lower() == "corecrel"), None)
    klass = _class_named(d, r"^(\w+\s+)?notes?$", "note")
    if make is None or not ok8(make.code) or not klass:
        return None
    new, at, props = (make.param(n) for n in ("new", "at", "with properties"))
    body = slot_prop(d, klass, "body", write=True)
    if not (new and props and body and ok4(new[1]) and ok4(props[1])):
        return None
    chain, holder = _default_chain(d, klass) or ((), "")
    if chain and not (at and ok4(at[1])):
        return None
    if not chain and klass not in _elements(d, "application"):
        return None  # no default place for a new note: the dictionary route can't choose one
    folder = holder or next(
        (c for c in sorted(_elements(d, "application")) if klass in _elements(d, c) and c != klass), ""
    )
    fname, fid = (slot_prop(d, folder, s) if folder else None for s in ("title", "id"))
    title, ident = slot_prop(d, klass, "title", write=True), slot_prop(d, klass, "id")
    text, created = slot_prop(d, klass, "plain") or body, slot_prop(d, klass, "created")
    named = bool(folder and fname and fid and ok4(d.class_code(folder)))
    return NoteRoute(
        klass=d.class_code(klass),
        make=make.code,
        p_new=new[1],
        p_at=at[1] if at and ok4(at[1]) else "",
        p_props=props[1],
        title=title[1] if title else "",
        body=body[1],
        html=is_html(d, klass, body[0]),
        ident=ident[1] if ident else "",
        text=text[1],
        created=created[1] if created else "",
        chain=chain,
        folder=d.class_code(folder) if named else "",
        folder_name=fname[1] if named and fname else "",
        folder_id=fid[1] if named and fid else "",
        folders_listed=named and folder in _elements(d, "application"),
    )


def note_html(text: str) -> str:
    """Plain text for an HTML body: every line escaped and in its own block (an empty line keeps
    its place), so nothing the user said can become markup."""
    return "".join(f"<div>{html.escape(line) or '<br>'}</div>" for line in text.split("\n"))


# the note and inbox scripts' header: argv item 1 is the bundle id, the rest is data
_BID_HEAD = "on run argv\n\tset bid to item 1 of argv\n\tset US to character id 31\n\tset RS to character id 30\n"
# Local handlers: text of a possibly missing value, clipped text, a date read back as ISO text, and
# an ISO local date built in-script (locale-safe: no date string is ever parsed by AppleScript).
_HANDLERS = """
on txt(x)
\tif x is missing value then return ""
\treturn x as text
end txt
on clip(x)
\tset x to my txt(x)
\tif (count x) > 400 then return text 1 thru 400 of x
\treturn x
end clip
on iso(d)
\tif d is missing value then return ""
\treturn (d as «class isot» as string)
end iso
on isoDate(s)
\tset d to current date
\tset day of d to 1
\tset year of d to (text 1 thru 4 of s) as integer
\tset month of d to (text 6 thru 7 of s) as integer
\tset day of d to (text 9 thru 10 of s) as integer
\tset hours of d to (text 12 thru 13 of s) as integer
\tset minutes of d to (text 15 thru 16 of s) as integer
\tset seconds of d to (text 18 thru 19 of s) as integer
\treturn d
end isoDate
"""


def _where(r: NoteRoute, folder_arg: int | None) -> str:
    """Lines (inside the app's tell block) that set ``c`` to where the note goes: the folder whose
    id is argv item ``folder_arg`` when it still exists, else the default container (missing
    value = the application itself)."""
    chain = " of ".join(f"«property {p}»" for p in reversed(r.chain))
    default = f"set c to {chain}" if r.chain else "set c to missing value"
    if folder_arg is None or not r.folder:
        return f"\t\t{default}\n"
    return (
        f"\t\ttry\n\t\t\tset c to «class {r.folder}» id (item {folder_arg} of argv)\n"
        f"\t\t\tget «property {r.folder_name}» of c\n\t\ton error\n\t\t\t{default}\n\t\tend try\n"
    )


def render_note(
    r: NoteRoute, bundle_id: str, title: str, body: str, folder_id: str = ""
) -> tuple[str, list[str]]:
    """(script, argv) that makes the note and reads it back through the app's own id for it.
    Answers OK, id, title, text (first 400 chars), container name; just OK when the note was made
    but couldn't be read back (the caller then looks for it rather than making another)."""
    argv = [bundle_id, title, note_html(body) if r.html else body]
    if folder_id and r.folder:
        argv.append(folder_id)
    props = ", ".join(
        f"«class {code}»:(item {i} of argv)" for code, i in ((r.title, 2), (r.body, 3)) if code
    )
    make = f"«event {r.make}» given «class {r.p_new}»:«class {r.klass}»"
    tail = f", «class {r.p_props}»:{{{props}}}"
    if r.chain:
        do_make = f"\t\tset o to {make}, «class {r.p_at}»:c{tail}\n"
    else:
        at = f"\t\t\tset o to {make}, «class {r.p_at}»:c{tail}\n" if r.p_at else f"\t\t\tset o to {make}{tail}\n"
        do_make = f"\t\tif c is missing value then\n\t\t\tset o to {make}{tail}\n\t\telse\n{at}\t\tend if\n"
    read = ['set p to ""', 'set n to ""', 'set cn to ""', 'set oid to ""']
    if r.ident:
        read += [
            f"set oid to («property {r.ident}» of o) as text",
            f"set o2 to «class {r.klass}» id oid",  # found again by the app itself: a real read-back
            f"set p to «property {r.text}» of o2",
        ]
        if r.title:
            read.append(f"set n to «property {r.title}» of o2")
    if r.folder_name:
        read.append(f"if c is not missing value then set cn to «property {r.folder_name}» of c")
    lines = "".join(f"\t\t\t{x}\n" for x in read)
    script = (
        f"{_BID_HEAD}\ttell application id bid\n{_where(r, 4 if len(argv) > 3 else None)}{do_make}\tend tell\n"
        # the note exists from here on: nothing below may fail the script (a failure would read as
        # "nothing happened" and a second note would follow)
        f"\ttry\n\t\ttell application id bid\n{lines}\t\tend tell\n"
        "\t\treturn \"OK\" & US & oid & US & my clip(n) & US & my clip(p) & US & my txt(cn)\n"
        "\ton error\n\t\treturn \"OK\"\n\tend try\nend run\n" + _HANDLERS
    )
    return script, argv


def render_note_search(
    r: NoteRoute, bundle_id: str, since: datetime, title: str, folder_id: str = ""
) -> tuple[str, list[str]] | None:
    """(script, argv) listing the notes made at or after ``since`` (by title when the dictionary
    has no creation date) where a new note would have gone: id, title, text per row. None when
    the dictionary gives no way to find one again."""
    # the value a `whose` test compares with is bound first: evaluated here, not inside the app
    if r.created:
        bind, test = "\tset t0 to my isoDate(item 2 of argv)\n", f"«property {r.created}» ≥ t0"
    elif r.title:
        bind, test = "\tset wanted to item 3 of argv\n", f"«property {r.title}» is wanted"
    else:
        return None
    argv = [bundle_id, since.strftime("%Y-%m-%dT%H:%M:%S"), title]
    if folder_id and r.folder:
        argv.append(folder_id)
    ident = f"(«property {r.ident}» of o) as text" if r.ident else '""'
    name = f"«property {r.title}» of o" if r.title else '""'
    script = (
        f"{_BID_HEAD}{bind}\ttell application id bid\n{_where(r, 4 if len(argv) > 3 else None)}"
        f"\t\tif c is missing value then\n\t\t\tset hits to every «class {r.klass}» whose {test}\n"
        f"\t\telse\n\t\t\tset hits to every «class {r.klass}» of c whose {test}\n\t\tend if\n\tend tell\n"
        "\tif (count hits) > 20 then set hits to items 1 thru 20 of hits\n"
        '\tset out to "OK"\n\trepeat with o in hits\n\t\ttell application id bid\n'
        f"\t\t\tset oid to {ident}\n\t\t\tset n to {name}\n\t\t\tset p to «property {r.text}» of o\n\t\tend tell\n"
        "\t\tset out to out & RS & oid & US & my clip(n) & US & my clip(p)\n\tend repeat\n\treturn out\nend run\n"
        + _HANDLERS
    )
    return script, argv


def render_folders(r: NoteRoute, bundle_id: str) -> tuple[str, list[str]] | None:
    """(script, argv) listing the folders the user can name (name, id per row); None when the
    application doesn't list them itself."""
    if not r.folders_listed:
        return None
    script = (
        f"{_BID_HEAD}\ttell application id bid\n"
        f"\t\tset ns to «property {r.folder_name}» of every «class {r.folder}»\n"
        f"\t\tset ids to «property {r.folder_id}» of every «class {r.folder}»\n\tend tell\n"
        '\tset out to "OK"\n\trepeat with i from 1 to count ns\n'
        "\t\tset out to out & RS & my txt(item i of ns) & US & my txt(item i of ids)\n\tend repeat\n"
        "\treturn out\nend run\n" + _HANDLERS
    )
    return script, [bundle_id]


@dataclass(frozen=True)
class UnreadRoute:
    """How an app's dictionary lists unread mail: the application's inbox property, its unread
    count, the message class and the properties read from each message ('' = not in it)."""

    inbox: str
    count: str
    message: str
    read: str
    sender: str
    subject: str
    received: str


def unread_route(d: Dictionary | None) -> UnreadRoute | None:
    """The dictionary's inbox, unread count and message read status, or None without them."""
    if d is None:
        return None
    app_props = d.class_props.get("application") or {}
    inbox = next(((c, t) for p, (c, t) in app_props.items() if c == "inmb" or re.match(r"^inbox$", p.strip(), re.I)), None)
    if inbox is None or inbox[1] not in d.classes or not ok4(inbox[0]):
        return None
    box = inbox[1]
    box_props = (d.class_props.get(box) or {}).items()
    count = next((c for p, (c, _t) in box_props if c == "mbuc" or re.match(r"^unread\s+count$", p.strip(), re.I)), "")
    message = _class_named(d, r"^(e-?mail\s+)?messages?$", "mssg")
    if not message or message not in _elements(d, box) or not ok4(count):
        return None
    read = slot_prop(d, message, "read")
    if read is None:
        return None
    got = {s: slot_prop(d, message, s) for s in ("sender", "subject", "received")}
    if got["sender"] is None and got["subject"] is None:
        return None
    return UnreadRoute(
        inbox=inbox[0],
        count=count,
        message=d.class_code(message),
        read=read[1],
        **{s: (p[1] if p else "") for s, p in got.items()},
    )


def render_unread(
    r: UnreadRoute, bundle_id: str, limit: int, scan: int, *, only_running: bool = False
) -> tuple[str, list[str]]:
    """(script, argv) reading the inbox's unread count, then the newest ``scan`` messages by
    index, keeping up to ``limit`` unread ones: sender, subject, date received (ISO). Never a
    `whose` filter (it drops the connection on big inboxes) and never a property set (nothing is
    marked read). ``only_running``: answer EMPTY rather than start the app."""
    box = f"«property {r.inbox}»"
    get = "".join(
        f"\t\t\t\tset {v} to «property {code}» of m\n" if code else f"\t\t\t\tset {v} to missing value\n"
        for v, code in (("s", r.sender), ("t", r.subject), ("d", r.received))
    )
    guard = '\tif application id bid is not running then return "EMPTY"\n' if only_running else ""
    script = (
        f"{_BID_HEAD}\tset lim to (item 2 of argv) as integer\n\tset scan to (item 3 of argv) as integer\n{guard}"
        f"\ttell application id bid\n\t\tset total to «property {r.count}» of {box}\n"
        f"\t\tset n to count «class {r.message}» of {box}\n\tend tell\n"
        '\tif n > scan then set n to scan\n\tset out to "OK" & US & my txt(total)\n\tset k to 0\n'
        "\trepeat with i from 1 to n\n\t\tif k ≥ lim then exit repeat\n\t\ttell application id bid\n"
        f"\t\t\tset m to «class {r.message}» i of {box}\n\t\t\tset seen to «property {r.read}» of m\n"
        f"\t\t\tif seen is false then\n{get}\t\t\tend if\n\t\tend tell\n"
        "\t\tif seen is false then\n\t\t\tset out to out & RS & my txt(s) & US & my txt(t) & US & my iso(d)\n"
        "\t\t\tset k to k + 1\n\t\tend if\n\tend repeat\n\treturn out\nend run\n" + _HANDLERS
    )
    return script, [bundle_id, str(int(limit)), str(int(scan))]
