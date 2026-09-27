"""The task kinds' drivers (one module per kind, arriving with each kind) and what they share:
which tools carry out a kind, the everyday nouns for a kind of app, and a correction's cue words.

Every table here is a language table (words people say), never a list of apps or sites.
"""

from __future__ import annotations

import importlib
import re
from types import ModuleType

__all__ = ["KIND_TOOLS", "NOUNS", "HANDLER_NOUNS", "CORRECTION", "noun_kind", "is_correction", "driver"]

# kind -> the tools that carry it out (the model is offered these when Laya routes a request)
KIND_TOOLS: dict[str, tuple[str, ...]] = {
    "play_media": ("play_music",),
    "write_note": ("notes_create",),
    "add_reminder": ("reminder_add",),
    "add_event": ("calendar_add",),
    "read_calendar": ("calendar_today",),
    "send_message": ("message_send",),
    "send_email": ("mail_send",),
    "read_mail": ("mail_unread",),
    "open_app": ("open_app",),
    "web_search": ("web_search", "search_site"),
}

# "open <noun>": the kind whose most-used app is meant. An installed app's own name always wins
# first, so this only answers when no app is called that.
NOUNS: dict[str, str] = {
    **dict.fromkeys(("email", "emails", "mail", "inbox", "e-mail"), "read_mail"),
    **dict.fromkeys(("calendar", "schedule", "agenda"), "read_calendar"),
    **dict.fromkeys(("notes", "note", "notepad"), "write_note"),
    **dict.fromkeys(("reminders", "reminder", "to-dos", "todos", "to-do list", "todo list"), "add_reminder"),
    **dict.fromkeys(("messages", "chats", "texts"), "send_message"),
}
# nouns for "whatever opens these links": the default handler of the scheme
HANDLER_NOUNS: dict[str, str] = dict.fromkeys(("browser", "internet", "web browser"), "https")

# "no, in <app>" / "not there, on <app>" / "nahi, <app> pe": the user redirects what NEO just did
CORRECTION = re.compile(
    r"(?<![\wऀ-ॿ])(?:no|nope|not|instead|actually|rather|nahi|nahin|balki|नहीं|नही|बल्कि)(?![\wऀ-ॿ])",
    re.I,
)
_WORD = re.compile(r"[\wऀ-ॿ]+")
# a word between the app's name and a closing cue ("<app> pe nahin")
_POSTPOSITIONS = frozenset({"on", "in", "pe", "par", "mein", "me", "पे", "पर", "में"})


def noun_kind(word: str) -> str | None:
    """The kind an everyday noun for a kind of app means ("my email" -> read_mail)."""
    w = " ".join(re.sub(r"^(?:my|the)\s+", "", (word or "").strip().casefold()).split())
    return NOUNS.get(w)


def is_correction(text: str, app: tuple[int, int] | None = None) -> bool:
    """Does the request redirect what NEO just did? Only the command's own words count, never what
    is being written or sent ("note in <app>: do not forget the milk" is a new note): a cue that
    opens the request ("no, in <app>", "actually use <app>"), or, given where ``text`` names the
    app, one that ends it right after the name ("<app> instead", "<app> pe nahin")."""
    words = _WORD.findall(text or "")
    if words and CORRECTION.fullmatch(words[0]):
        return True
    if app is None:
        return False
    after = _WORD.findall(text[app[1] :])
    if after and after[0].casefold() in _POSTPOSITIONS:
        after = after[1:]
    return len(after) == 1 and bool(CORRECTION.fullmatch(after[0]))


def driver(kind: str) -> ModuleType | None:
    """The kind's driver module (engine.DRIVERS), None until that kind has one."""
    from neo2.core.decide.engine import DRIVERS

    name = DRIVERS.get(kind)
    if not name:
        return None
    try:
        return importlib.import_module(name)
    except ImportError:
        return None
