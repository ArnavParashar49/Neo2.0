"""neo.decide.shadow: the log of what a kind's new path would have done while legacy ran. Plain
facts only, the user's own file, 30 days at most, gone with the privacy erase, and nothing from a
dry run that outlived an erase."""

from __future__ import annotations

import asyncio
import json
import stat
import time

import pytest

from neo.decide import shadow


@pytest.fixture
def calls_legacy(monkeypatch, real_handlers):
    from neo.tools.computer import legacy

    async def notes_create(*a):
        return "notes_create done"

    monkeypatch.setattr(legacy, "notes_create", notes_create)


def _lines() -> list[dict]:
    return [json.loads(x) for x in shadow.path().read_text().splitlines()]


def _later(kind, coro, **extra):
    async def go():
        shadow.later(kind, coro, **extra)
        await asyncio.gather(*shadow._tasks)

    asyncio.run(go())


def test_a_line_holds_plain_facts_only_in_the_users_own_file():
    shadow.record(
        "add_event",
        {
            "kind": "forged",
            "first": "route:eventkit.events",
            "items": 3,
            "ok": True,
            "title": ["Dentist"],
            "when": object(),
        },
    )
    (line,) = _lines()
    assert line["kind"] == "add_event" and line["first"] == "route:eventkit.events" and line["items"] == 3
    assert "title" not in line and "when" not in line and time.time() - line["t"] < 5
    assert stat.S_IMODE(shadow.path().stat().st_mode) == 0o600


def test_lines_older_than_thirty_days_go_on_the_next_write():
    old = time.time() - shadow.KEEP_S - 60
    shadow.path().write_text(json.dumps({"t": old, "kind": "open_app"}) + "\nnot json\n")
    shadow.record("web_search", {"how": "learned"})
    assert [r["kind"] for r in _lines()] == ["web_search"]
    assert stat.S_IMODE(shadow.path().stat().st_mode) == 0o600
    shadow.record("web_search", {"how": "domain"})
    assert len(_lines()) == 2


def test_a_dry_run_is_logged_with_what_legacy_did_and_a_failure_by_its_name(shadow_runs):
    async def facts():
        return {"app": "test.notes"}

    async def broken():
        raise KeyError("x")

    _later("write_note", facts(), legacy_ok=False)
    _later("read_mail", broken())
    assert _lines()[0] == {**_lines()[0], "kind": "write_note", "app": "test.notes", "legacy_ok": False}
    assert _lines()[1]["error"] == "KeyError"


def test_a_slow_dry_run_is_logged_as_timed_out(shadow_runs, monkeypatch):
    monkeypatch.setattr(shadow, "TIMEOUT_S", 0.01)

    async def slow():
        await asyncio.sleep(5)
        return {}

    _later("web_search", slow())
    assert _lines()[0]["error"] == "timeout"


def test_nothing_is_logged_from_a_dry_run_that_outlived_an_erase_and_the_erase_deletes_the_log(shadow_runs):
    from neo import privacy

    shadow.record("open_app", {"how": "app"})

    async def erased_meanwhile():
        await privacy.forget_everything()
        return {"how": "site"}

    _later("open_app", erased_meanwhile())
    assert not shadow.path().exists()
    shadow.record("open_app", {"how": "app"})
    assert asyncio.run(privacy.forget_everything())["task dry runs"] == 1 and not shadow.path().exists()


def test_in_tests_a_dry_run_never_starts_unless_asked_for():
    ran = []

    async def facts():
        ran.append(1)
        return {}

    _later("open_app", facts())
    assert ran == [] and not shadow.path().exists()


def test_a_dry_run_that_cant_start_never_touches_legacys_answer(calls_legacy, monkeypatch):
    from neo.agent.registry import ToolContext, registry

    def broken(kind, facts, **extra):
        facts.close()
        raise ImportError("the driver won't load")

    monkeypatch.setattr(shadow, "later", broken)
    out = asyncio.run(
        registry().invoke("notes_create", {"title": "t", "body": "b"}, ToolContext(user_text="x"))
    )
    assert out.ok and out.text == "notes_create done"
