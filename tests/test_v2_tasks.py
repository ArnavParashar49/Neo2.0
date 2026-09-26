"""The task tools (notes, reminders, calendar, mail) behave exactly as before their move to
neo/tools/tasks.py; today's app- and site-specific code lives only in legacy.py; the new modules
name no app, bundle id or website; the per-kind settings parse forgivingly."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from neo.agent import confirm
from neo.agent.registry import ToolContext, registry
from neo.events import EventBus

ROOT = Path(__file__).resolve().parent.parent
_PIM = ("notes_create", "reminder_add", "calendar_add", "calendar_today", "mail_send", "mail_unread")


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    from neo import config, events
    from neo.tools import load_all

    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setenv("NEO_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(events, "_bus", EventBus())
    confirm.cancel("test")
    load_all()
    yield
    confirm.cancel("test")


@pytest.fixture
def calls(monkeypatch):
    """Every legacy script call, with its arguments (nothing reaches an app)."""
    from neo.tools import tasks
    from neo.tools.computer import legacy

    for name in _PIM:  # other suites swap handlers on the shared registry
        monkeypatch.setattr(registry().get(name), "handler", getattr(tasks, name))
    log: list = []
    for name in _PIM:
        async def fake(*a, _n=name, **k):
            log.append((_n, a, k))
            return f"{_n} done"

        monkeypatch.setattr(legacy, name, fake)
    return log


def _run(name, args, ctx=None):
    return asyncio.run(registry().invoke(name, args, ctx or ToolContext()))


def test_the_task_tools_are_registered_from_tasks_with_their_flags():
    import neo.tools.computer as comp
    from neo.tools import tasks

    for name in _PIM:
        assert registry().get(name) is not None and callable(getattr(tasks, name)), name
        assert not hasattr(comp, name)  # no second registration left behind
    flags = {n: (registry().get(n).risk, registry().get(n).quiet, registry().get(n).chain) for n in _PIM}
    assert flags == {
        "notes_create": ("reversible", True, False),
        "reminder_add": ("reversible", True, True),
        "calendar_add": ("reversible", True, True),
        "calendar_today": ("safe", False, True),
        "mail_send": ("destructive", False, True),
        "mail_unread": ("safe", False, True),
    }
    assert registry().get("mail_unread").untrusted and registry().get("calendar_today").untrusted
    assert registry().get("notes_create").payload and registry().get("mail_send").slow


def test_the_task_tools_keep_their_place_in_the_registry():
    """The tool picker's keyword fallback breaks ties, and Live declares its tools, in
    registration order: the move changes neither."""
    names = [t.name for t in registry().all()]
    i = names.index("safari_read")
    assert names[i + 1 : i + 8] == [
        "mail_send", "mail_unread", "calendar_today", "calendar_add", "notes_create", "reminder_add", "finder_reveal"
    ]


def test_the_task_tools_call_the_legacy_scripts_as_before(calls):
    assert _run("notes_create", {"title": "", "body": "buy milk and eggs for the party tonight"}).ok
    _run("reminder_add", {"title": "call mom"})
    _run("reminder_add", {"title": "call mom", "due": "2026-09-27T17:00"})
    _run("calendar_add", {"title": "Standup", "start": "2026-09-27T10:00", "end": "2026-09-27T10:15"})
    _run("calendar_today", {})
    _run("mail_unread", {})
    _run("mail_unread", {"limit": 3})
    _run("mail_send", {"to": "a@b.c", "subject": "Hi", "body": "Hello"})
    assert calls == [
        ("notes_create", ("buy milk and eggs for the", "buy milk and eggs for the party tonight", ""), {}),
        ("reminder_add", ("call mom", ""), {}),
        ("reminder_add", ("call mom", "2026-09-27T17:00"), {}),
        ("calendar_add", ("Standup", "2026-09-27T10:00", "2026-09-27T10:15", ""), {}),
        ("calendar_today", (), {}),
        ("mail_unread", (10,), {}),
        ("mail_unread", (3,), {}),
        ("mail_send", ("a@b.c", "Hi", "Hello", False), {}),
    ]


def test_sending_mail_still_asks_first(calls):
    out = _run("mail_send", {"to": "a@b.c", "subject": "Hi", "body": "Hello", "send": True})
    assert out.halt and out.text.startswith(confirm.NEEDS_CONFIRM) and calls == []
    with confirm.approved(confirm.peek().action_id):
        out = _run("mail_send", {"to": "a@b.c", "subject": "Hi", "body": "Hello", "send": True})
    assert out.ok and calls == [("mail_send", ("a@b.c", "Hi", "Hello", True), {})]


def test_the_web_tools_still_use_the_moved_site_table():
    from neo.tools import web
    from neo.tools.computer import legacy

    assert web.site_search_url is legacy.site_search_url
    src = (ROOT / "neo/tools/web.py").read_text()
    assert not re.search(r"^(?:_SITES|_AMAZON|SITE_NAMES|_GENERIC_SITES)\s*[:=]|^def site_search_url", src, re.M)
    apps = (ROOT / "neo/tools/computer/apps.py").read_text()
    assert not re.search(r"^async def (?:safari_page_text|mail_send|calendar_add|notes_create)\b", apps, re.M)


# ---- per-kind modes ---------------------------------------------------------------------------------


def test_every_kind_is_legacy_unless_the_user_set_it(monkeypatch):
    from neo import config
    from neo.tools import tasks

    assert tasks.mode("add_event") == "legacy"
    monkeypatch.setenv("NEO_DECIDE_MODES", "add_event=on, read_calendar = Shadow; write_note=maybe, junk")
    monkeypatch.setattr(config, "_settings", None)
    assert config.settings().decide_modes == {"add_event": "on", "read_calendar": "shadow"}
    assert (tasks.mode("add_event"), tasks.mode("write_note")) == ("on", "legacy")
    assert tasks.shadow("add_event", candidates=2, route=True) is None


@pytest.mark.parametrize(
    "raw, want",
    [('{"send_message": "OFF"}', {"send_message": "off"}), ("{broken", {}), ("", {}), ("=on", {})],
)
def test_decide_modes_never_stop_neo_from_starting(monkeypatch, raw, want):
    from neo import config

    monkeypatch.setenv("NEO_DECIDE_MODES", raw)
    monkeypatch.setattr(config, "_settings", None)
    s = config.settings()
    assert s.decide_modes == want and s.decide_kind_routing is False
    assert s.notes_dir == Path.home() / "Documents" / "NEO Notes"


# ---- no app names in the new modules (§12.1 of the stage-2 plan) ------------------------------------

_FORBIDDEN = re.compile(
    r"\b(?:spotify|itunes|apple music|youtube|safari|chrome|firefox|whatsapp|telegram|slack|discord|gmail|"
    r"outlook|facetime|amazon|flipkart|ebay|google|obsidian|notion|fantastical)\b"
    r"|org\.whispersystems|signal-desktop|com\.apple\.|tell application \"|\bwww\.[a-z]"
)
_GUARDED = ["neo/tools/tasks.py"]


@pytest.mark.parametrize("path", _GUARDED)
def test_new_modules_name_no_app_or_site(path):
    for n, line in enumerate((ROOT / path).read_text().lower().splitlines(), 1):
        assert not _FORBIDDEN.search(line), f"{path}:{n}: {line.strip()}"
