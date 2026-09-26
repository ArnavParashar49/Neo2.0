"""Task tools: notes, reminders, the calendar and mail.

Today each runs the scripts in neo/tools/computer/legacy.py. As a task kind gets its
evidence-chosen driver (neo/decide), its tool switches on `mode(kind)`, the user's
`decide_modes` setting: `legacy` keeps these scripts, `shadow` runs them while the driver only
reports what it would have chosen (`shadow()`), `on` hands the kind to the driver.
"""

from __future__ import annotations

from neo.agent.registry import ToolContext, tool
from neo.config import DecideMode, settings
from neo.tools.computer import legacy

_OBJ = {"type": "object", "properties": {}}


def _p(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or []}


def mode(kind: str) -> DecideMode:
    """The mode the user set for a kind ("add_event", "write_note"…); legacy by default."""
    return settings().decide_modes.get(kind, "legacy")


def shadow(kind: str, **facts: str | int | float | bool) -> None:
    """Where a kind in `shadow` mode reports what its new path would have chosen: app keys,
    methods, counts and booleans, never a title, a name or a message. Nothing is recorded
    until a kind has a driver to compare with legacy."""


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
        )
    ],
    untrusted=True,
)
async def mail_unread(a: dict, c: ToolContext) -> str:
    return await legacy.mail_unread(int(a.get("limit") or 10))


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
        )
    ],
    untrusted=True,
)
async def calendar_today(a: dict, c: ToolContext) -> str:
    return await legacy.calendar_today()


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
)
async def calendar_add(a: dict, c: ToolContext) -> str:
    return await legacy.calendar_add(a["title"], a["start"], a["end"], a.get("calendar", ""))


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
        )
    ],
    payload=True,
)
async def notes_create(a: dict, c: ToolContext) -> str:
    title, body = (a.get("title") or "").strip(), (a.get("body") or "").strip()
    if not title:  # "make a note saying buy milk": the first words become the title
        title = " ".join(body.split()[:6]) or "New note"
    return await legacy.notes_create(title, body, a.get("folder", ""))


@tool(
    "reminder_add",
    "Add a reminder, optionally due at ISO local 'YYYY-MM-DDTHH:MM'.",
    _p({"title": {"type": "string"}, "due": {"type": "string"}}, ["title"]),
    risk="reversible",
    category="apps",
    quiet=True,
)
async def reminder_add(a: dict, c: ToolContext) -> str:
    return await legacy.reminder_add(a["title"], a.get("due", ""))
