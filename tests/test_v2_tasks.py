"""The task tools (notes, reminders, calendar, mail, and open_app / search_site / page_read) per
kind mode: in legacy and shadow (the release default) they behave exactly as before and show the
same surface, shadow logging the new path's dry run; on hands each kind to its driver and shows
its new surface. Today's app- and site-specific code lives only in legacy.py; the new modules name
no app, bundle id or website; the per-kind settings parse forgivingly."""

from __future__ import annotations

import asyncio
import json
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


_DRIVEN = ("write_note", "add_reminder", "add_event", "read_calendar", "read_mail", "open_app", "web_search")


def test_the_kinds_with_drivers_ship_in_shadow_and_the_user_switches_them(monkeypatch):
    from neo import config
    from neo.tools import tasks

    assert {k: tasks.mode(k) for k in _DRIVEN} == dict.fromkeys(_DRIVEN, "shadow")
    assert (tasks.mode("send_message"), tasks.mode("send_email")) == ("legacy", "legacy")  # R4/R5
    monkeypatch.setenv("NEO_DECIDE_MODES", "add_event=on, read_calendar = Legacy; write_note=maybe, junk")
    monkeypatch.setattr(config, "_settings", None)
    assert config.settings().decide_modes == {"add_event": "on", "read_calendar": "legacy"}
    assert (tasks.mode("add_event"), tasks.mode("read_calendar"), tasks.mode("write_note")) == ("on", "legacy", "shadow")


# ---- shadow: legacy answers, the new path's dry run is logged ------------------------------------------


@pytest.fixture
def dry_runs(monkeypatch, shadow_runs):
    """Each kind's dry run faked: what it was asked, and the facts it reports."""
    from neo.decide.kinds import mail, note, schedule

    asked: list = []

    async def would_choose(kind, a, text=""):
        asked.append((kind, dict(a), text))
        return {"kind": kind, "first": "route:eventkit.events", "candidates": "route:eventkit.events,test.cal"}

    async def note_preview(a, said=""):
        asked.append(("write_note", dict(a), said))
        return {"app": "test.notes", "method": "script_make", "title": object()}  # not plain: dropped

    async def mail_preview(a, said="", legacy=""):
        asked.append(("read_mail", dict(a), said, legacy))
        return {"app": "test.mail", "unread": 3, "same": True}

    monkeypatch.setattr(schedule, "would_choose", would_choose)
    monkeypatch.setattr(note, "preview", note_preview)
    monkeypatch.setattr(mail, "preview", mail_preview)
    return asked


def _settled(name, args, ctx=None):
    """Invoke, then let the background dry runs finish (the loop they started on ends here)."""
    from neo.decide import shadow

    async def go():
        out = await registry().invoke(name, args, ctx or ToolContext(user_text="the words"))
        await asyncio.gather(*shadow._tasks)
        return out

    return asyncio.run(go())


def _log() -> list[dict]:
    import json

    from neo.decide import shadow

    p = shadow.path()
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def test_in_shadow_legacy_answers_and_the_dry_run_is_logged(calls, dry_runs, monkeypatch):
    from neo.tools.computer import legacy

    async def today():
        calls.append(("calendar_today", (), {}))
        return "9:00 Standup\n13:00 Lunch\n"

    monkeypatch.setattr(legacy, "calendar_today", today)
    assert _settled("calendar_add", {"title": "Standup", "start": "2026-09-27T10:00", "end": "2026-09-27T10:15"}).ok
    assert _settled("calendar_today", {}).text == "9:00 Standup\n13:00 Lunch"
    _settled("reminder_add", {"title": "call mom"})
    _settled("notes_create", {"title": "", "body": "buy milk"})
    _settled("mail_unread", {"limit": 3})
    assert [c[0] for c in calls] == ["calendar_add", "calendar_today", "reminder_add", "notes_create", "mail_unread"]
    assert [a[0] for a in dry_runs] == ["add_event", "read_calendar", "add_reminder", "write_note", "read_mail"]
    assert dry_runs[0][2] == "the words" and dry_runs[-1][3] == "mail_unread done"  # the user's words; legacy's answer
    log = _log()
    assert [r["kind"] for r in log] == ["add_event", "read_calendar", "add_reminder", "write_note", "read_mail"]
    assert all(r["legacy_ok"] for r in log) and log[1]["legacy_items"] == 2
    assert "title" not in log[3] and log[4]["unread"] == 3
    assert "Standup" not in json.dumps(log) and "milk" not in json.dumps(log)


def test_legacy_runs_no_dry_run_and_a_failed_one_is_logged_by_name(calls, dry_runs, decide_modes, monkeypatch):
    from neo.decide.kinds import schedule

    decide_modes(add_event="legacy")
    _settled("calendar_add", {"title": "x", "start": "2026-09-27T10:00", "end": "2026-09-27T10:15"})
    assert calls and dry_runs == [] and _log() == []

    async def broken(kind, a, text=""):
        raise LookupError("no store")

    monkeypatch.setattr(schedule, "would_choose", broken)
    _settled("reminder_add", {"title": "x"})
    assert _log()[0]["error"] == "LookupError" and _log()[0]["kind"] == "add_reminder"


# ---- on: the kind's driver answers ---------------------------------------------------------------------------


def test_on_hands_each_kind_to_its_driver(calls, decide_modes, monkeypatch):
    from neo.decide import run
    from neo.decide.kinds import mail, note, schedule

    decide_modes(**dict.fromkeys(_DRIVEN, "on"))
    seen: list = []

    async def carry_out(kind, a, text="", *, ask_access=False):
        seen.append((kind, dict(a), text, ask_access))
        return run.Result(f"{kind} done", True, say=kind == "add_reminder")

    async def write(a, said=""):
        seen.append(("write_note", dict(a), said))
        return run.Result("Saved it as a file", True, say=True)

    async def unread(a, said=""):
        seen.append(("read_mail", dict(a), said))
        return run.Result("Error: the inbox didn't answer", False)

    monkeypatch.setattr(schedule, "carry_out", carry_out)
    monkeypatch.setattr(note, "write", write)
    monkeypatch.setattr(mail, "unread", unread)
    ctx = ToolContext(user_text="remind me at 5 to call mom")
    out = _run("reminder_add", {"title": "call mom", "due": "at 5", "list": "Errands"}, ctx)
    assert out.ok and out.say is True and out.text == "add_reminder done"
    assert _run("calendar_add", {"title": "Lunch", "start": "friday at 1"}).ok
    assert _run("calendar_today", {"day": "tomorrow"}).text == "read_calendar done"
    assert _run("notes_create", {"body": "buy milk", "app": "Jotter"}).say is True
    assert not _run("mail_unread", {"app": "Postbox"}).ok
    assert calls == []  # no legacy script ran
    assert seen[0] == ("add_reminder", {"title": "call mom", "due": "at 5", "list": "Errands"}, ctx.user_text, True)
    assert [s[0] for s in seen[1:]] == ["add_event", "read_calendar", "write_note", "read_mail"]
    assert seen[3][1] == {"body": "buy milk", "app": "Jotter"}


def test_a_kind_shows_its_new_surface_only_while_on(decide_modes):
    def props(name):
        return set(registry().get(name).parameters["properties"])

    def shown(name):
        return not registry().get(name).hidden

    assert props("calendar_today") == set() and props("reminder_add") == {"title", "due"}
    assert not registry().get("reminder_add").payload
    assert registry().get("calendar_add").parameters["required"] == ["title", "start", "end"]
    assert registry().get("mail_unread").timeout == 50 and shown("page_read") and not shown("safari_read")
    decide_modes(**dict.fromkeys(_DRIVEN, "on"))
    assert props("calendar_today") == {"day", "start", "end"}
    assert props("reminder_add") == {"title", "due", "list", "notes", "app"} and registry().get("reminder_add").payload
    assert props("notes_create") == {"body", "title", "folder", "app"}
    assert registry().get("notes_create").parameters["required"] == ["body"]
    assert registry().get("calendar_add").parameters["required"] == ["title", "start"]
    assert registry().get("mail_unread").timeout == 110 and "app" in props("mail_unread")
    assert "browser" in props("search_site") and shown("page_read") and not shown("safari_read")
    decide_modes(write_note="on")  # one kind at a time
    assert "app" in props("notes_create") and props("calendar_today") == set() and "browser" not in props("search_site")
    assert shown("page_read") and not shown("safari_read")  # page_read reads any browser, in every mode


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


# ---- the other tools of these kinds: open_app, search_site, page_read, "new document" ----------------------


def test_open_app_in_shadow_answers_with_legacy_and_logs_the_dry_run(monkeypatch, shadow_runs, real_handlers):
    from neo.decide.kinds import launch
    from neo.tools.computer import apps

    async def opened(target):
        return f"Opened {target}"

    async def launch_shadow(target, legacy=""):
        return {"how": "app", "key": "test.surfer", "agrees": legacy == "Opened surfer"}

    monkeypatch.setattr(apps, "open_app", opened)
    monkeypatch.setattr(launch, "shadow", launch_shadow)
    assert _settled("open_app", {"target": "surfer"}).text == "Opened surfer"
    (log,) = _log()
    assert log == {**log, "kind": "open_app", "key": "test.surfer", "agrees": True, "legacy_ok": True}


@pytest.mark.parametrize("mode", ["legacy", "shadow", "on"])
def test_page_read_reads_the_browser_in_front_in_every_mode(decide_modes, monkeypatch, real_handlers, mode):
    """A model's call gets the page itself; the user's own words (a fast path) get NEO's answer from
    it. safari_read is the old name, hidden."""
    from neo.decide import run
    from neo.decide.kinds import site

    decide_modes(web_search=mode)
    asked: list = []

    async def page_read(limit=site.PAGE_CHARS, ask=""):
        asked.append(ask)
        return run.Result(f"answer to {ask}" if ask else "URL: https://example.test/\n\nwords", True)

    monkeypatch.setattr(site, "page_read", page_read)
    assert _run("page_read", {}).text.startswith("URL: https://example.test/")
    assert _run("safari_read", {}).text.startswith("URL: https://example.test/")
    said = ToolContext(user_text="  summarize   this page ", origin="user")
    assert _run("page_read", {}, said).text == "answer to summarize this page"
    live = ToolContext(user_text="summarize this page", origin="live")
    assert _run("page_read", {}, live).text.startswith("URL:")
    assert asked == ["", "", "summarize this page", ""]
    t = registry().get("page_read")
    assert t.untrusted and not t.hidden and not t.chain and registry().get("safari_read").hidden


def test_with_web_search_on_a_search_carrying_data_asks(decide_modes, monkeypatch, real_handlers):
    from neo.decide import run
    from neo.decide.kinds import site

    decide_modes(web_search="on")
    searched: list = []

    async def search(s, q, *, browser=""):
        searched.append((s, q, browser))
        return run.Result("Searched it", True)

    monkeypatch.setattr(site, "search", search)
    assert _run("search_site", {"site": "shopkart.in", "query": "2tb ssd", "browser": "Surfer"}).ok
    assert searched == [("shopkart.in", "2tb ssd", "Surfer")]
    secret = "x" * 80
    out = _run("search_site", {"site": "shopkart.in", "query": secret}, ToolContext(user_text="", taint=["mail_unread"]))
    assert out.halt and out.text.startswith(confirm.NEEDS_CONFIRM) and len(searched) == 1


def test_a_new_document_goes_by_legacys_names_until_write_note_is_on(decide_modes, monkeypatch):
    from neo.decide import inventory
    from neo.decide.inventory import App
    from neo.decide.kinds import note
    from neo.tools import computer

    editor = App(bundle_id="test.pad", name="Padwise", path="/Applications/Padwise.app", editor_types=("public.rtf",))
    monkeypatch.setattr(inventory, "app_for_pid", lambda pid: editor if pid == 7 else None)
    assert computer._doc_like("Notes", 1) and not computer._doc_like("Padwise", 7)
    decide_modes(write_note="on")
    assert note.doc_like(editor) and computer._doc_like("Padwise", 7) and not computer._doc_like("Notes", 1)


def test_the_users_notes_app_takes_new_note_only_while_write_note_is_on(decide_modes, monkeypatch, real_handlers):
    """A call naming "note" in any other mode, or with other keys (a model's, a playbook saved
    before a rollback), goes the legacy way: no Accessibility press, no launch."""
    from neo.tools import computer

    went: list[str] = []

    async def new_note(c):
        went.append("new note")
        return "New note in Jotter."

    async def legacy_guard(want=""):
        went.append(f"legacy {want}")
        return "Error: stopped here"

    monkeypatch.setattr(computer, "_new_note", new_note)
    monkeypatch.setattr(computer, "_guard_front", legacy_guard)
    for mode in ("shadow", "legacy"):
        decide_modes(write_note=mode)
        _run("hotkey", {"keys": "cmd+n", "app": "note"})
    decide_modes(write_note="on")
    _run("hotkey", {"keys": "cmd+s", "app": "note"})
    assert went == ["legacy note"] * 3
    assert _run("hotkey", {"keys": "Command + N", "app": "note"}).text == "New note in Jotter."


def test_live_and_the_prompts_change_only_for_kinds_switched_on(decide_modes):
    from neo.agent import prompt
    from neo.voice import live

    assert live._direct_tools() == live._DIRECT_TOOLS and "calendar_add" not in live._DIRECT_TOOLS
    extra = live._live_extra()
    assert '"Opening Google"' in extra and '"open it on Amazon"' in extra and "<site>" not in extra
    assert "Notes, reminders and calendar" not in extra and '"find X on Amazon"' in prompt.persona()
    decide_modes(write_note="on")
    assert "NEO picks the app the user uses" in live._live_extra() and "<site>" not in live._live_extra()
    decide_modes(add_event="on", web_search="on")
    extra = live._live_extra()
    assert "calendar_add" in live._direct_tools() and '"Opening it"' in extra and "Amazon" not in extra
    assert '"open it on <site>"' in extra and "Notes, reminders and calendar: NEO picks" in extra
    assert '"find X on <site>"' in prompt.persona() and "Amazon" not in prompt.persona()
