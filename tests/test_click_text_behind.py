"""click_text while NEO's own window is in front (the user just talked to it) works in the app behind
it (ax.working_app). Accessibility's press needs no app in front; a click with the mouse does — a
click on a window in the background may only bring it forward — so an element with no press, or
words read off the screen, are clicked only once that app has come to the front, and never while
NEO's own window still is. Hermetic: every Accessibility read, window and event is faked."""

from __future__ import annotations

import asyncio
import dataclasses
from types import SimpleNamespace

import pytest

pytest.importorskip("Quartz")

# The OCR fallback's fake Mac (its fixtures, used here by these names).
from test_eyes_ocr import APP, PID, _fresh_ocr, _show  # noqa: E402, F401
from test_eyes_ocr import desk as ocr_desk  # noqa: E402, F401

from neo2.core.agent.registry import ToolContext, ToolOutput, registry  # noqa: E402
from neo2.core.tools import load_all  # noqa: E402
from neo2.core.tools.computer import apps, ax, focus, gates, ocr  # noqa: E402
from neo2.core.tools.computer import input as inp  # noqa: E402

NEO_PID = 900
MAIL, MAIL_PID = "Mail", 777


@pytest.fixture
def mac(monkeypatch):
    """NEO's window in front, Mail's window behind it with an "Inbox" row Accessibility can't press."""
    load_all()
    m = SimpleNamespace(front=("NEO", NEO_PID), clicks=[], brought=[], regular=True, comes=True, layer=0)
    m.els = [ax.Element("e1", "AXStaticText", "Inbox", "", 100.0, 100.0, 60.0, 20.0, actions=[], ref=object())]
    monkeypatch.setattr(apps, "_target", None)
    monkeypatch.setattr(focus, "front_app", lambda: m.front)
    monkeypatch.setattr(apps, "front_name", lambda: m.front[0])
    monkeypatch.setattr(ax, "is_neo", lambda pid, name="": pid == NEO_PID or name == "NEO")
    monkeypatch.setattr(ax, "working_app", lambda: (MAIL, MAIL_PID))
    monkeypatch.setattr(ax, "window_elements", lambda app=None, **k: (app or m.front[0], (0.0, 0.0, 800.0, 600.0), list(m.els)))
    monkeypatch.setattr(ax, "pid_of", lambda ref: MAIL_PID)
    monkeypatch.setattr(ax, "_running", lambda pid: SimpleNamespace(bundleIdentifier=lambda: "test.mail"))
    monkeypatch.setattr(ax, "app_names", lambda pid: set())
    monkeypatch.setattr(ax, "regular", lambda pid: m.regular)
    monkeypatch.setattr(gates, "_outside_system_menu", lambda els: list(els))

    def hit(x, y):
        return ax.Hit(pid=MAIL_PID, app=MAIL, label="Inbox", role="AXStaticText", regular=m.regular)

    monkeypatch.setattr(ax, "hit", hit)
    # The window server: Mail's window at the point, a normal app window (layer 0) unless a test says
    # otherwise — the one thing a click brings an app forward for (DECISIONS 59).
    monkeypatch.setattr(ax, "windows_at", lambda x, y: [ax.Win(MAIL_PID, MAIL, m.layer, (0.0, 0.0, 800.0, 600.0))])

    async def bring(pid):  # (async, like apps.bring_pid_front: it comes forward, or it doesn't)
        m.brought.append(pid)
        if m.comes:
            m.front = (MAIL, pid)
        return m.comes, MAIL

    monkeypatch.setattr(apps, "bring_pid_front", bring)
    monkeypatch.setattr(inp, "click", lambda x, y, **k: m.clicks.append(((x, y), m.front[0])))
    return m


def _click(label="Inbox", origin="user"):
    out = asyncio.run(registry().invoke("click_text", {"label": label}, ToolContext(user_text=f"click {label}", origin=origin)))
    return out.text if isinstance(out, ToolOutput) else str(out)


def test_an_element_with_no_press_behind_neos_window_is_clicked_once_its_app_is_in_front(mac):
    text = _click()
    assert mac.brought == [MAIL_PID]  # brought forward first…
    assert mac.clicks == [((130.0, 110.0), MAIL)]  # …so the click lands in Mail, not on a window behind NEO's
    assert text.startswith("Clicked 'Inbox'") and "brought Mail to the front first" in text


def test_an_app_that_wont_come_forward_is_never_clicked_behind_neos_window(mac):
    mac.comes = False
    text = _click()
    assert mac.clicks == [] and text.startswith("Error") and "didn't come to the front" in text


@pytest.mark.parametrize("regular", [True, False])
def test_an_app_bring_passes_over_is_never_clicked_while_neos_window_is_in_front(mac, regular):
    """What is at the spot isn't the app's own normal window (a panel, an overlay: never brought
    forward, DECISIONS 59), so the bring passes it over: NEO's own window is still in front, and a
    click would land on a window behind it — nothing is clicked."""
    mac.regular, mac.layer = regular, 3
    text = _click()
    assert mac.clicks == [] and mac.front == ("NEO", NEO_PID)
    assert text.startswith("Error") and "didn't come to the front" in text and "Mail" in text


def test_with_the_app_already_in_front_nothing_is_brought(mac):
    mac.front = (MAIL, MAIL_PID)
    text = _click()
    assert mac.brought == [] and mac.clicks == [((130.0, 110.0), MAIL)] and "brought" not in text


def test_the_screens_words_are_read_in_the_app_behind_neos_window(mac, monkeypatch):
    """Nothing Accessibility names so: the OCR fallback reads the app behind NEO's window — told its
    pid (the front app is NEO's, so it can't be asked) and that NEO's window is in front."""
    seen = []

    async def fallback(a, c, **k):
        seen.append(k)
        return ToolOutput("Clicked the words")

    monkeypatch.setattr(ocr, "click_text_fallback", fallback)
    assert _click("Drafts") == "Clicked the words"
    assert seen and seen[0]["app"] == MAIL and seen[0]["pid"] == MAIL_PID and seen[0]["behind"] is True
    mac.front = (MAIL, MAIL_PID)  # (Mail itself in front: asked as before)
    seen.clear()
    _click("Drafts")
    assert seen and not seen[0].get("pid") and not seen[0].get("behind")


def test_the_ocr_fallback_behind_neos_window_clicks_only_once_the_app_is_in_front(request, monkeypatch):
    desk = request.getfixturevalue("ocr_desk")
    _show(monkeypatch, [("Hint", (300.0, 40.0, 40.0, 16.0))])
    desk.front = ("NEO", NEO_PID)  # (NEO's window in front; the app comes forward when brought)
    monkeypatch.setattr(ax, "is_neo", lambda pid, name="": pid == NEO_PID)
    ctx = ToolContext(user_text="click hint", origin="user")

    def fallback():
        return asyncio.run(
            ocr.click_text_fallback({"label": "hint"}, ctx, app=APP, window=(0.0, 0.0, 400.0, 300.0), pid=PID, behind=True)
        )

    out = fallback()  # (brought forward: bring_pid_front says it came)
    assert out.ok and desk.brought == [PID] and len(desk.clicks) == 1
    real_windows = desk.windows_at
    monkeypatch.setattr(ax, "windows_at", lambda x, y: [dataclasses.replace(w, layer=3) for w in real_windows(x, y)])
    desk.front = ("NEO", NEO_PID)
    desk.brought.clear()
    desk.clicks.clear()
    out = fallback()  # not the app's own normal window there: the bring passes it over, NEO's stays in front
    assert not out.ok and desk.clicks == [] and "didn't come to the front" in out.text
