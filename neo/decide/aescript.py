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
    HTML gets the text escaped, line by line. Answers OK + the new item's id."""
    argv = [bid, container_id]
    fields = []
    for slot, code, kind in r.sets:
        if slot not in values or values[slot] is None:
            continue
        v = values[slot]
        if slot == "body" and r.html:
            v = "".join(f"<div>{html.escape(line) or '<br>'}</div>" for line in str(v).split("\n"))
        argv.append(_arg(kind, v))
        fields.append(f"{_prop(code)}:{_value(kind, len(argv))}")
    ident = f'(my neoText(get {_prop(r.id_code)} of o, "text"))' if r.id_code else '""'
    script = (
        f"{_HEAD}\ttell application id bid\n{_container(r)}"
        f"\t\ttell c\n\t\t\tset o to make new {_cls(r.code)} with properties {{{', '.join(fields)}}}\n\t\tend tell\n"
        f'\t\treturn "OK" & US & {ident}\n\tend tell\nend run\n{_DATE_HANDLER}'
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
