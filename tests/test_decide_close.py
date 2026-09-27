"""'close YouTube', 'quit Spotify': close what's named — an app, or a site's tabs — without a card.

Fakes only: no app is quit, no browser scripted."""

from __future__ import annotations

import asyncio

import pytest

from neo.decide import close, inventory, media
from neo.decide.inventory import App
from neo.decide.sdef import Command, Dictionary
from neo.tools.computer import apps as mac


def browser_dict(tab="CrTb", url="URL ", title="pnam", title_name="title") -> Dictionary:
    return Dictionary(
        commands={"close": Command("close", "coreclos", "specifier", ())},
        classes={"window": ("cwin", ("name",)), "tab": (tab, ("id", title_name, "URL"))},
        properties={"URL": url, title_name: title, "name": "pnam"},
        enums={},
    )


BROWSER = App(bundle_id="com.fixture.browser", name="Fixture Browser", path="/Applications/Fixture Browser.app",
              handles=("https",), script=browser_dict(), running=True)
PLAYER = App(bundle_id="com.fixture.player", name="Tunewave", path="/Applications/Tunewave.app", running=True)
IDLE = App(bundle_id="com.fixture.idle", name="Sketchpad", path="/Applications/Sketchpad.app", running=False)


@pytest.fixture
def mac_fakes(monkeypatch):
    calls = {"quit": [], "scripts": []}
    monkeypatch.setattr(close, "_running_apps", lambda: [BROWSER, PLAYER])
    monkeypatch.setattr(inventory, "named_in", lambda text, apps=None: next(
        (a for a in (apps if apps is not None else [BROWSER, PLAYER, IDLE]) if a.name.lower() in text.lower()), None))

    async def quit_app(name):
        calls["quit"].append(name)
        return f"Quit {name}"

    async def gate(app):
        return None

    async def osa(script, args, timeout):
        calls["scripts"].append((script, args))
        return 0, calls.get("closed", "2"), ""

    monkeypatch.setattr(mac, "quit_app", quit_app)
    monkeypatch.setattr(media, "_control_gate", gate)
    monkeypatch.setattr(media, "_osascript", osa)
    return calls


def test_a_running_app_is_quit(mac_fakes):
    assert asyncio.run(close.close_target("Tunewave")) == "Quit Tunewave"
    assert mac_fakes["quit"] == ["Tunewave"] and mac_fakes["scripts"] == []


def test_a_sites_tabs_close_with_the_browsers_own_codes_and_the_words_as_data(mac_fakes):
    out = asyncio.run(close.close_target("the YouTube tab"))
    assert out == "Closed 2 YouTube tabs in Fixture Browser"
    (script, args), = mac_fakes["scripts"]
    assert args == ["com.fixture.browser", "YouTube"]  # argv, never script source
    assert "«class CrTb»" in script and "«property URL »" in script and "«event coreclos»" in script
    assert "YouTube" not in script and "Fixture" not in script


def test_nothing_open_by_that_name_is_said(mac_fakes):
    mac_fakes["closed"] = "0"
    assert asyncio.run(close.close_target("netflix")) == "Error: nothing called 'netflix' is open"
    assert asyncio.run(close.close_target("Sketchpad")) == "Error: Sketchpad isn't open"
    assert asyncio.run(close.close_target("the tab")) == "Error: close what?"


def test_a_browser_that_does_not_script_its_tabs_is_skipped():
    plain = App(bundle_id="x", name="Plain", path="/Applications/Plain.app", script=Dictionary({}, {}, {}, {}))
    assert close._tab_codes(plain) is None
    bad = App(bundle_id="y", name="Odd", path="/Applications/Odd.app", script=browser_dict(tab='Cr"b'))
    assert close._tab_codes(bad) is None  # a code that isn't four plain characters never reaches a script


def test_close_needs_no_card_and_is_what_the_fast_path_runs(monkeypatch):
    from neo.agent import fastpath
    from neo.agent.registry import registry
    from neo.tools import load_all

    load_all()
    t = registry().get("close")
    assert t.risk == "safe" and t.quiet and t.idempotent
    assert fastpath.plan("close youtube") == [("close", {"target": "youtube"})]
    assert fastpath.plan("quit Spotify") == [("close", {"target": "Spotify"})]
    assert fastpath.plan("close the netflix tab") == [("close", {"target": "the netflix tab"})]
    assert fastpath.plan("close this tab") == [("hotkey", {"keys": "cmd+w"})]
    assert fastpath.plan("close the window") == [("hotkey", {"keys": "cmd+w"})]
