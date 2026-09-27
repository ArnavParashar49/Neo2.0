"""The tools group's wiring after the split: the eyes (capture, marks, OCR, board) and the commands
(resolver, menus) reach the handlers in tools_*.py through the frozen gates. Hermetic: every capture,
Accessibility read and event is faked."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("Quartz")

from neo2.core.agent.registry import ToolContext  # noqa: E402
from neo2.core.providers.base import ImagePart  # noqa: E402
from neo2.core.tools.computer import apps, ax, focus, gates, pointer, screen, tools_pointer  # noqa: E402


def _cap(kind="full", *, clickable=True, x0=0.0, y0=0.0, ppp=1.0, w=100, h=80):
    return SimpleNamespace(
        part=ImagePart(b"x", "image/jpeg"), x0=x0, y0=y0, ppp=ppp, w=w, h=h, kind=kind, clickable=clickable,
        covered=(), frame=None, window_id=0, pid=0,
    )  # fmt: skip


class _App:
    def __init__(self, pid, name):
        self._pid, self._name = pid, name

    def processIdentifier(self):  # noqa: N802 - AppKit's name
        return self._pid

    def localizedName(self):  # noqa: N802
        return self._name


@pytest.fixture
def shots(monkeypatch):
    took: list[tuple] = []
    monkeypatch.setattr(screen, "shot", lambda region=None, display=None, **k: took.append(("shot", region)) or _cap())
    monkeypatch.setattr(screen, "zoom", lambda x, y, w, h, **k: took.append(("zoom", x, y, w, h)) or _cap("zoom", ppp=0.5))
    monkeypatch.setattr(screen, "window_shot", lambda wid, *a, **k: took.append(("window", wid)) or _cap("window", x0=10, y0=20))
    monkeypatch.setattr(apps, "_running", lambda name: _App(4321, "Pages") if name.lower() == "pages" else None)
    monkeypatch.setattr(focus, "front_window_id", lambda pid: 77 if pid == 4321 else 0)
    monkeypatch.setattr(ax, "is_neo", lambda pid, name="": False)
    return took


def _run(a: dict):
    return asyncio.run(tools_pointer.screenshot(a, ToolContext(origin="agent")))


def test_a_screenshot_takes_the_working_apps_front_window(shots, monkeypatch):
    monkeypatch.setattr(apps, "target", lambda: "Pages")
    out = _run({})
    assert shots == [("window", 77)] and "Pages's front window" in out.text and out.images
    assert pointer.get_shot(out.text.split()[1]).x0 == 10  # clicks map through the window's own origin


def test_the_whole_display_when_asked_or_when_no_app_is_known(shots, monkeypatch):
    monkeypatch.setattr(apps, "target", lambda: None)
    _run({})
    monkeypatch.setattr(apps, "target", lambda: "Pages")
    _run({"display": 1})
    _run({"x": 0, "y": 0, "w": 50, "h": 50})
    assert [s[0] for s in shots] == ["shot", "shot", "shot"]


def test_never_a_protected_apps_or_neos_window(shots, monkeypatch):
    monkeypatch.setattr(apps, "target", lambda: "Pages")
    monkeypatch.setattr(gates, "_protected", lambda name, pid=0: name == "Pages")
    _run({})
    monkeypatch.setattr(gates, "_protected", lambda name, pid=0: False)
    monkeypatch.setattr(ax, "is_neo", lambda pid, name="": True)
    _run({"app": "Pages"})
    assert [s[0] for s in shots] == ["shot", "shot"]


def test_a_window_picture_that_cannot_be_mapped_is_for_looking_only(shots, monkeypatch):
    monkeypatch.setattr(screen, "window_shot", lambda wid, *a, **k: _cap("window", clickable=False))
    out = _run({"app": "Pages"})
    assert "for looking only" in out.text and 'shot="' not in out.text


def test_a_zoom_around_a_point(shots):
    out = _run({"zoom": True, "x": 300, "y": 200})
    assert shots == [("zoom", 300.0, 200.0, float(screen.ZOOM_W), float(screen.ZOOM_H))] and "zoom" in out.text
    assert not _run({"zoom": True, "x": 300}).ok  # a zoom needs its point


def test_a_click_on_a_part_of_a_window_picture_another_window_covered_is_refused():
    s = pointer.remember(100.0, 100.0, 1.0, 400, 300, covered=((150.0, 150.0, 50.0, 50.0),))
    with pytest.raises(pointer.Refused, match="under another window"):
        pointer.numeric({"x": 60, "y": 60, "shot": s.id}, ToolContext(), "click")
    assert pointer.numeric({"x": 10, "y": 10, "shot": s.id}, ToolContext(), "click").point == (110.0, 110.0)


def test_a_screenshot_lists_the_controls_in_it(shots, monkeypatch):
    from neo2.core.tools.computer import marks

    seen = []

    async def collect(area, ctx=None, *, pid=0, deadline_s=0.8):
        seen.append((area, pid))
        return marks.Collected(app="Pages", pid=pid)

    monkeypatch.setattr(marks, "collect", collect)
    monkeypatch.setattr(marks, "for_shot", lambda col, s, c: "\n(marks of " + col.app + ")")
    monkeypatch.setattr(apps, "target", lambda: "Pages")
    out = _run({})
    assert seen == [(None, 4321)] and out.text.endswith("(marks of Pages)")  # the window's own app, walked alongside
    _run({"x": 0, "y": 0, "w": 50, "h": 40})
    assert seen[-1] == ((0.0, 0.0, 50.0, 40.0), 0)


def test_a_blind_brains_screenshot_says_no_numbers(shots, monkeypatch):
    import re

    monkeypatch.setattr(apps, "target", lambda: None)
    for a in ({}, {"x": 10, "y": 20, "w": 300, "h": 200}, {"zoom": True, "x": 300, "y": 200}):
        out = asyncio.run(tools_pointer.screenshot(a, ToolContext(origin="agent", extras={"blind": True})))
        assert out.ok and not re.search(r"\d", out.text), out.text


# ---- eyes-ocr -------------------------------------------------------------------------------------------


def test_click_text_reads_the_screen_when_accessibility_names_nothing(monkeypatch):
    from neo2.core.agent.registry import ToolOutput
    from neo2.core.tools.computer import ocr, tools_label

    async def no_guard(want=""):
        return None

    monkeypatch.setattr(gates, "_guard_front", no_guard)
    monkeypatch.setattr(ax, "window_elements", lambda app=None: ("Game", (0.0, 0.0, 800.0, 600.0), []))
    asked = []

    async def fallback(a, c, *, app, window, els=None, pid=0):
        asked.append((a["label"], app, window))
        return ToolOutput("Clicked the words 'Hint'")

    monkeypatch.setattr(ocr, "click_text_fallback", fallback)
    out = asyncio.run(tools_label.click_text({"label": "Hint"}, ToolContext(origin="agent")))
    assert asked == [("Hint", "Game", (0.0, 0.0, 800.0, 600.0))] and out.text.startswith("Clicked the words")

    async def unread(a, c, **k):
        return None  # the screen couldn't be read: the tool's own answer

    monkeypatch.setattr(ocr, "click_text_fallback", unread)
    assert "is called 'Hint'" in asyncio.run(tools_label.click_text({"label": "Hint"}, ToolContext(origin="agent")))


def test_an_outcome_says_which_words_changed(monkeypatch):
    from neo2.core.tools.computer import ocr

    monkeypatch.setattr(ocr, "sample_change", lambda b, a: "“Start” → “Stop”")
    before = pointer.Look(ax.Scene(), ["b"])
    out = pointer.Outcome("Changed: the screen changed around (1, 2).", "", True)
    got = asyncio.run(pointer._words(out, before, pointer.Look(ax.Scene(), ["a"])))
    assert got.details == "“Start” → “Stop”"
    same = pointer.Outcome("Nothing visibly changed.", "", False)
    assert asyncio.run(pointer._words(same, before, pointer.Look(ax.Scene(), ["a"]))).details == ""


def test_text_marks_are_clicked_by_id_through_the_pointer_with_their_words_gated():
    import inspect

    from neo2.core.tools.computer import ocr

    assert "ocr.install()" in inspect.getsource(tools_pointer)  # (at import: text marks work from the start)
    assert ocr.install()
    assert pointer._id_resolvers["t"] is ocr.resolve_mark and pointer._id_labels["t"] is ocr.mark_labels
