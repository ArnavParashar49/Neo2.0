"""Task tools: notes, reminders, the calendar and mail.

Each tool switches on `mode(kind)`, the user's `decide_modes` setting (neo.config):
- `legacy` runs today's scripts (neo/tools/computer/legacy.py);
- `shadow` runs them too, and the kind's driver works out in the background what it would have
  done (neo.decide.shadow): it opens, asks and learns nothing;
- `on` hands the request to the kind's driver (neo/decide/kinds).

A kind's new surface (the tool's description and parameters, its new fast paths, Live's direct
tools) shows only while the kind is on: in shadow nothing a model or the user sees changes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from neo.agent.registry import ToolContext, registry, tool
from neo.config import DecideMode, decide_mode
from neo.decide import shadow
from neo.tools.computer import legacy

_OBJ = {"type": "object", "properties": {}}
_S = {"type": "string"}


def _p(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or []}


def mode(kind: str) -> DecideMode:
    """How a kind ("add_event", "write_note"…) runs now (see the module docstring)."""
    return decide_mode(kind)


def when_on(kind: str, check: Callable[[dict], bool] | None = None) -> Callable[[dict], bool]:
    """A fast-path check for a pattern that claims words only while ``kind`` is on (and then
    only when ``check``, the driver's own test on cached data, agrees)."""
    return lambda filled: mode(kind) == "on" and (check is None or check(filled))


def _schedule_check(kind: str) -> Callable[[dict], bool]:
    def check(filled: dict) -> bool:
        from neo.decide.kinds import schedule

        return schedule.check(kind, filled)

    return when_on(kind, check)


# ---- each tool's surface while its kind is on -------------------------------------------------------
# tool -> (kind, the RegisteredTool fields it has while the kind is on). Legacy and shadow keep
# the fields the tool was registered with.
_WHEN_ON: dict[str, tuple[str, dict[str, Any]]] = {
    "notes_create": (
        "write_note",
        {
            "description": "Save a note. NEO saves it in the notes app this user uses (or the app they named: "
            "pass it as `app`) and checks it was saved. `title` is optional.",
            "parameters": _p({"body": _S, "title": _S, "folder": _S, "app": _S}, ["body"]),
        },
    ),
    "reminder_add": (
        "add_reminder",
        {
            "description": "Add a reminder, a to-do or an item to a list. `due` is ISO local 'YYYY-MM-DDTHH:MM' "
            "(a date alone = that day); `list` only if the user named one; `app` only if the user named an app. "
            "NEO sets an alert at the due time.",
            "parameters": _p({"title": _S, "due": _S, "list": _S, "notes": _S, "app": _S}, ["title"]),
            "payload": True,  # its fast paths capture free text to the end of the sentence
        },
    ),
    "calendar_add": (
        "add_event",
        {
            "description": "Put an event on the user's calendar (their default one unless they named another). "
            "Times are ISO local 'YYYY-MM-DDTHH:MM'; a date alone 'YYYY-MM-DD' is an all-day event, and an "
            "all-day `end` date is its last day. `app` only if the user named an app. Never add invitees.",
            "parameters": _p(
                {
                    "title": _S,
                    "start": _S,
                    "end": _S,
                    "all_day": {"type": "boolean"},
                    "location": _S,
                    "notes": _S,
                    "calendar": _S,
                    "app": _S,
                },
                ["title", "start"],
            ),
        },
    ),
    "calendar_today": (
        "read_calendar",
        {
            "description": "What's on the user's calendar: today by default, or `day` ('tomorrow', 'friday', "
            "'next week') or a `start`–`end` range (ISO; an end date alone is the last day).",
            "parameters": _p({"day": _S, "start": _S, "end": _S}),
        },
    ),
    # the driver's own budget, plus macOS's Automation question, waited for outside it
    "mail_unread": (
        "read_mail",
        {"parameters": _p({"limit": {"type": "integer"}, "app": _S}), "timeout": 110.0},
    ),
    "search_site": (
        "web_search",
        {
            "description": "Open a search for `query` on the website the user named, in their browser. `site` is "
            "the name or domain the user said; `browser` only if they named one. To PLAY music or video, use "
            "play_music. The query must be the actual thing, never 'it'.",
            "parameters": _p({"site": _S, "query": _S, "browser": _S}, ["site", "query"]),
        },
    ),
    # page_read reads whichever browser is in front; until web_search is on, the model keeps the old tool
    "page_read": ("web_search", {"hidden": False}),
    "safari_read": ("web_search", {"hidden": True}),
}
_AS_REGISTERED: dict[str, dict[str, Any]] = {}


def apply_modes() -> None:
    """Show each tool as its kind's mode wants. The modes are read once, at start (load_all)."""
    for name, (kind, on) in _WHEN_ON.items():
        t = registry().get(name)
        if t is None:
            continue
        registered = _AS_REGISTERED.setdefault(name, {k: getattr(t, k) for k in on})
        for k, v in (on if mode(kind) == "on" else registered).items():
            setattr(t, k, v)


def _said(c: ToolContext, a: dict) -> str:
    """The user's own words for the request ('' when a model's arguments stand in for them)."""
    return "" if c.user_text == str(a) else c.user_text


def _legacy_ok(out: str) -> bool:
    return not str(out).startswith("Error")


# ---- mail -----------------------------------------------------------------------------------------


@tool(
    "mail_send",
    "Compose an email in Mail. send=false opens a draft for the user; send=true sends it.",
    _p(
        {
            "to": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string"},
            "send": {"type": "boolean"},
        },
        ["to", "subject", "body"],
    ),
    risk="destructive",
    category="apps",
    slow=True,
)
async def mail_send(a: dict, c: ToolContext) -> str:
    from neo.agent import confirm

    if confirm.as_bool(a.get("send")):
        gate = confirm.gated(
            "mail_send",
            a,
            f"send an email to {a['to']} — “{a['subject']}”",
            detail=f"To: {a['to']}\nSubject: {a['subject']}\n\n{a['body']}",
        )
        if isinstance(gate, str):
            return gate
    return await legacy.mail_send(a["to"], a["subject"], a["body"], bool(a.get("send")))


@tool(
    "mail_unread",
    "List unread inbox messages (sender | subject | date).",
    _p({"limit": {"type": "integer"}}),
    parallel_safe=True,
    category="apps",
    slow=True,
    timeout=50,
    fast_path=[
        (
            r"^\s*(?:(?:check|read|show(?:\s+me)?|list|get|open)\s+(?:my\s+)?(?:new\s+|unread\s+|latest\s+)?"
            r"(?:e-?mails?|mail|inbox)|(?:do\s+i\s+have\s+)?(?:any\s+)?(?:new|unread)\s+(?:e-?mails?|mail|messages)|"
            r"what(?:'s| is)\s+(?:new\s+)?in\s+my\s+inbox|any\s+(?:new\s+)?(?:e-?mails?|mail))\s*[?.!]*\s*$",
            {},
        ),
        (  # "mera mail check karo", "koi naya email aaya?"
            r"^\s*(?:(?:mera|meri|mere)\s+)?(?:e-?mails?|mail|inbox)\s+(?:check|dekh|dikha)\s*(?:karo|kar\s+do|lo|do|na)?"
            r"\s*[?.!]*\s*$|^\s*(?:kya\s+)?(?:koi\s+)?(?:naya|nayi|naye)\s+(?:e-?mails?|mail)\s+(?:aaya|aayi|aaye|hai)"
            r"(?:\s+(?:hai|kya))?\s*[?.!]*\s*$",
            {},
            when_on("read_mail"),
        ),
    ],
    untrusted=True,
)
async def mail_unread(a: dict, c: ToolContext):
    m = mode("read_mail")
    if m == "on":
        from neo.decide import run
        from neo.decide.kinds import mail

        return run.to_output(await mail.unread(a, _said(c, a)))
    out = await legacy.mail_unread(int(a.get("limit") or 10))
    if m == "shadow":
        with shadow.starting("read_mail"):
            from neo.decide.kinds import mail

            shadow.later("read_mail", mail.preview(a, _said(c, a), legacy=out), legacy_ok=_legacy_ok(out))
    return out


# ---- calendar and reminders -------------------------------------------------------------------------

# a clock time said the Hinglish way, bounded so it never runs into the title ("kal shaam 5 baje")
_BAJE = (
    r"(?:(?:aaj|kal|parso|parson)\s+)?(?:(?:subah|dopahar|shaam|sham|raat)\s+)?(?:(?:saadhe|sawa|paune)\s+)?"
    r"\d{1,2}(?:[:.]\d{2})?\s+baje"
)
_WEEKDAY = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"


async def _schedule(kind: str, a: dict, c: ToolContext, run_legacy: Callable[[], Any], **extra: Any):
    """One schedule kind's request in its mode (see the module docstring)."""
    m = mode(kind)
    if m == "on":
        from neo.decide import run
        from neo.decide.kinds import schedule

        return run.to_output(await schedule.carry_out(kind, a, _said(c, a), ask_access=True))
    out = await run_legacy()
    if m == "shadow":
        with shadow.starting(kind):
            from neo.decide.kinds import schedule

            facts = {"legacy_ok": _legacy_ok(out)}
            if kind == "read_calendar":  # the events legacy listed ("Nothing on the calendar today." is none)
                listed = str(out).strip().splitlines()
                facts["legacy_items"] = 0 if str(out).startswith("Nothing") else len(listed)
            shadow.later(kind, schedule.would_choose(kind, a, _said(c, a)), **facts)
    return out


@tool(
    "calendar_today",
    "List today's calendar events.",
    _OBJ,
    parallel_safe=True,
    category="apps",
    fast_path=[
        (
            r"^\s*(?:what(?:'s| is)\s+on\s+(?:my\s+)?(?:calendar|schedule|agenda)(?:\s+(?:for\s+)?today)?|"
            r"(?:do\s+i\s+have\s+)?(?:any\s+)?(?:meetings?|events?|appointments?)\s+today|"
            r"(?:show|check|read|open)\s+(?:me\s+)?(?:my\s+)?(?:calendar|schedule|agenda)(?:\s+(?:for\s+)?today)?|"
            r"what(?:'s| is)\s+my\s+(?:schedule|agenda|day)(?:\s+like)?(?:\s+today)?|what\s+do\s+i\s+have\s+(?:on\s+)?today)\s*[?.!]*\s*$",
            {},
        ),
        (  # "what's on my calendar tomorrow", "... for next friday", "... this week"
            r"^\s*what(?:'s| is)\s+on\s+(?:my\s+)?(?:calendar|schedule|agenda)\s+(?:for\s+|on\s+)?"
            rf"(?P<day>today|tonight|tomorrow|(?:this\s+|next\s+)?{_WEEKDAY}|(?:this|next)\s+week)\s*[?.!]*\s*$",
            {"day": "<day>"},
            _schedule_check("read_calendar"),
        ),
        (  # "kal ka calendar kya hai", "aaj mera schedule mein kya hai"
            r"^\s*(?P<day>aaj|kal|parso|parson)\s+(?:ka\s+)?(?:(?:mera|meri)\s+)?(?:calendar|schedule)\s+"
            r"(?:mein\s+|me\s+)?kya\s+(?:hai|h)\s*\??\s*$",
            {"day": "<day>"},
            _schedule_check("read_calendar"),
        ),
        (  # "kal mera calendar kya tha": kal, said in the past, is yesterday
            r"^\s*kal\s+(?:(?:mera|meri)\s+)?(?:calendar|schedule)\s+(?:mein\s+|me\s+)?kya\s+(?:tha|thi|the)\s*\??\s*$",
            {"day": "yesterday"},
            _schedule_check("read_calendar"),
        ),
        (
            r"^\s*(?:mera|meri)\s+(?:schedule|calendar)\s+kya\s+(?:hai|h)\s*\??\s*$",
            {},
            _schedule_check("read_calendar"),
        ),
        (
            r"^\s*(?P<day>आज|कल|परसों)\s+(?:का\s+)?(?:कैलेंडर|शेड्यूल)\s+(?:में\s+)?क्या\s+है\s*\??\s*$",
            {"day": "<day>"},
            _schedule_check("read_calendar"),
        ),
    ],
    untrusted=True,
)
async def calendar_today(a: dict, c: ToolContext):
    return await _schedule("read_calendar", a, c, legacy.calendar_today)


@tool(
    "calendar_add",
    "Add a calendar event. Times are ISO local 'YYYY-MM-DDTHH:MM'.",
    _p(
        {
            "title": {"type": "string"},
            "start": {"type": "string"},
            "end": {"type": "string"},
            "calendar": {"type": "string"},
        },
        ["title", "start", "end"],
    ),
    risk="reversible",
    category="apps",
    quiet=True,
    fast_path=[
        (  # "put lunch with sam on my calendar tomorrow at 1" ("in calendar" alone names the app)
            r"^\s*(?:please\s+)?(?:add|put|schedule)\s+(?P<title>.+?)\s+(?:on|to|in)\s+my\s+calendar\s+"
            r"(?P<when>.+?)\s*[.!]*\s*$",
            {"title": "<title>", "start": "<when>"},
            _schedule_check("add_event"),
        ),
        (  # "kal shaam 5 baje dentist calendar mein daal do"
            rf"^\s*(?P<when>{_BAJE})\s+(?P<title>.+?)\s+calendar\s+(?:mein|me)\s+(?:daal|dal|add\s+kar)\s*do\s*$",
            {"title": "<title>", "start": "<when>"},
            _schedule_check("add_event"),
        ),
    ],
)
async def calendar_add(a: dict, c: ToolContext):
    def run_legacy():
        return legacy.calendar_add(a["title"], a["start"], a["end"], a.get("calendar", ""))

    return await _schedule("add_event", a, c, run_legacy)


# ---- notes ----------------------------------------------------------------------------------------


@tool(
    "notes_create",
    "Create a note in Apple Notes.",
    _p(
        {"title": {"type": "string"}, "body": {"type": "string"}, "folder": {"type": "string"}},
        ["title", "body"],
    ),
    risk="reversible",
    category="apps",
    quiet=True,
    chain=False,  # only as the whole utterance: a later clause means 'in that app'
    fast_path=[
        (
            r"^\s*(?:please\s+)?(?:make|create|add|start)\s+(?:a\s+)?(?:new\s+)?note\s+(?:called|titled|named)\s+"
            r"[\"“']?(?P<title>.+?)[\"”']?(?:\s+(?:saying|that\s+says|with(?:\s+the\s+text)?|containing)\s+"
            r"[\"“']?(?P<body>.+?)[\"”']?)?\s*[.!]*\s*$",
            {"title": "<title>", "body": "<body>"},
        ),
        (
            r"^\s*(?:please\s+)?(?:make|create|add|start)\s+(?:a\s+)?(?:new\s+)?note\s+(?:saying|that\s+says)\s+"
            r"[\"“']?(?P<body>.+?)[\"”']?\s*[.!]*\s*$",
            {"title": "", "body": "<body>"},
        ),
        (  # "note kar lo ki kal meeting hai", "likh lo ki ..."
            r"^\s*(?:please\s+)?(?:(?:ek\s+)?note\s+(?:kar|bana|likh)|likh)\s+(?:lo|lena|do|de|le|dena)\s+(?:ki|ke)\s+"
            r"(?P<body>.+?)\s*[.!।]*\s*$",
            {"title": "", "body": "<body>"},
            when_on("write_note"),
        ),
        (
            r"^\s*(?:नोट\s+(?:कर|बना)|लिख)\s+(?:लो|लें|लेना|दो|दें|देना)\s+कि\s+(?P<body>.+?)\s*[।.!]*\s*$",
            {"title": "", "body": "<body>"},
            when_on("write_note"),
        ),
    ],
    payload=True,
)
async def notes_create(a: dict, c: ToolContext):
    m = mode("write_note")
    if m == "on":
        from neo.decide import run
        from neo.decide.kinds import note

        return run.to_output(await note.write(a, _said(c, a)))
    title, body = (a.get("title") or "").strip(), (a.get("body") or "").strip()
    if not title:  # "make a note saying buy milk": the first words become the title
        title = " ".join(body.split()[:6]) or "New note"
    out = await legacy.notes_create(title, body, a.get("folder", ""))
    if m == "shadow":
        with shadow.starting("write_note"):
            from neo.decide.kinds import note

            shadow.later("write_note", note.preview(a, _said(c, a)), legacy_ok=_legacy_ok(out))
    return out


@tool(
    "reminder_add",
    "Add a reminder, optionally due at ISO local 'YYYY-MM-DDTHH:MM'.",
    _p({"title": {"type": "string"}, "due": {"type": "string"}}, ["title"]),
    risk="reversible",
    category="apps",
    quiet=True,
    fast_path=[
        (  # "remind me tomorrow at 9 to call mom" (before the next one: its title is lazy)
            r"^\s*(?:please\s+)?remind\s+me\s+(?P<when>.+?)\s+to\s+(?P<title>.+?)\s*[.!]*\s*$",
            {"title": "<title>", "due": "<when>"},
            _schedule_check("add_reminder"),
        ),
        (  # "remind me to call mom at 5"
            r"^\s*(?:please\s+)?remind\s+me\s+(?:to\s+)?(?P<title>.+?)\s+"
            r"(?P<when>(?:at|on|by|tomorrow|tonight|today|in|next|this|kal|aaj)\b.*?)\s*[.!]*\s*$",
            {"title": "<title>", "due": "<when>"},
            _schedule_check("add_reminder"),
        ),
        (  # "add milk to my shopping list" (a list NEO has seen in the store)
            r"^\s*(?:please\s+)?add\s+(?P<title>.+?)\s+to\s+(?:my|the)\s+(?P<list>[\w ]{1,20}?)\s+list\s*[.!]*\s*$",
            {"title": "<title>", "list": "<list>"},
            _schedule_check("add_reminder"),
        ),
        (  # "kal subah 9 baje doctor ko call ka reminder lagao", "... yaad dilana"
            rf"^\s*(?P<when>{_BAJE})\s+(?P<title>.+?)\s+(?:ka\s+reminder\s+(?:lagao|laga\s+do|set\s+karo)|"
            r"yaad\s+dila(?:na|\s+dena))\s*$",
            {"title": "<title>", "due": "<when>"},
            _schedule_check("add_reminder"),
        ),
        (
            r"^\s*(?P<when>(?:(?:आज|कल|परसों)\s+)?(?:(?:सुबह|दोपहर|शाम|रात)\s+)?\d{1,2}\s+बजे)\s+(?P<title>.+?)\s+"
            r"(?:याद\s+दिलाना|याद\s+दिला\s+देना)\s*$",
            {"title": "<title>", "due": "<when>"},
            _schedule_check("add_reminder"),
        ),
    ],
)
async def reminder_add(a: dict, c: ToolContext):
    return await _schedule("add_reminder", a, c, lambda: legacy.reminder_add(a["title"], a.get("due", "")))
