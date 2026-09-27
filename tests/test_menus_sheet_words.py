"""The sheet a menu command opens, as the real AX core reads it (on test_ax_core's fake AX layer):
its words kept whole and one by one, and a walk that was cut short is a sheet NEO didn't read -
never answered by itself (DECISIONS 46: ask whenever its words carry loss or consequence wording;
words NEO didn't read could). The menu bar and the presses are test_menus' fakes."""

from __future__ import annotations

import test_ax_core
import test_menus
from test_ax_core import El, app_root, window
from test_ax_core import button as ax_button
from test_menus import APP, PID, user

from neo2.core.tools.computer import ax, menus

fake = test_ax_core.fake  # the fake AX layer (a fixture)
world = test_menus.world  # the fake menu bar (a fixture)

_REAL_BLOCKERS = ax.blockers  # (the fake menu bar puts its own in its place)
_SETUP = "Choose the board, the players and the clock for each side before the first move. " * 6
_WARNING = "The current game will be lost."


def _sheet(fake, monkeypatch, *kids) -> ax.Blocker:
    """A sheet on the app's window with these children, read by the real AX core."""
    sheet = El("AXSheet", frame=(500, 60, 400, 400), children=list(kids))
    fake.add_app(APP, PID, app_root(window("Game", El("AXGroup", frame=(0, 60, 1512, 900)), sheet)))
    monkeypatch.setattr(ax, "window_owner_at", lambda x, y: ("", 0))  # no other app's window over it
    got = _REAL_BLOCKERS(PID)
    assert [b.kind for b in got] == ["sheet"]
    return got[0]


def _text(value: str, row: int) -> El:
    return El("AXStaticText", value=value, frame=(520, 80 + 24 * row, 360, 20))


def _buttons() -> list[El]:
    return [ax_button("Start", frame=(800, 420, 60, 24)), ax_button("Cancel", frame=(720, 420, 60, 24))]


async def test_a_warning_past_the_first_400_characters_is_still_read(fake, world, monkeypatch):
    """Probe F1: the sheet's warning comes after 450 characters of setup text."""
    assert len(_SETUP) > 400
    world.sheets["New Game"] = _sheet(fake, monkeypatch, _text("Start a New Game?", 0), _text(_SETUP, 1), _text(_WARNING, 2), *_buttons())
    out = await menus.by_utterance("start a new game", user("start a new game"))
    assert world.pressed == ["Game › New Game"] and out.halt and out.text.startswith("NEEDS_USER")


async def test_a_sheet_whose_walk_stopped_early_is_never_answered_by_itself(fake, world, monkeypatch):
    """The app stops answering after the sheet's buttons were read, before its warning: what wasn't
    read could say anything."""
    stuck = [El("AXGroup", frame=(520, 300, 100, 20)) for _ in range(ax._HUNG_AFTER)]
    for s in stuck:
        s.fails = True  # quick refusals in a row: the walk ends ("hung")
    blk = _sheet(fake, monkeypatch, _text("Start a New Game?", 0), *_buttons(), *stuck, _text(_WARNING, 2))
    assert [b.title for b in blk.buttons] == ["Start", "Cancel"] and _WARNING not in blk.texts
    world.sheets["New Game"] = blk
    out = await menus.by_utterance("start a new game", user("start a new game"))
    assert world.pressed == ["Game › New Game"] and out.halt and out.text.startswith("NEEDS_USER")


async def test_a_sheet_read_whole_is_still_answered_by_the_users_word(fake, world, monkeypatch):
    world.sheets["New Game"] = _sheet(fake, monkeypatch, _text("Start a New Game?", 0), _text(_SETUP, 1), *_buttons())
    out = await menus.by_utterance("start a new game", user("start a new game"))
    assert world.pressed == ["Game › New Game", "Start"] and out.ok


def test_the_ax_core_keeps_a_sheets_texts_whole_apart_and_says_when_it_read_them_all(fake, monkeypatch):
    blk = _sheet(fake, monkeypatch, _text("Start a New Game?", 0), _text(_SETUP, 1), _text(_WARNING, 2), *_buttons())
    assert blk.complete and blk.texts == ["Start a New Game?", " ".join(_SETUP.split()), _WARNING]
    assert len(blk.text) <= 400  # (the short form, as before)

    def within(*texts: str) -> ax._Within:
        els = [ax.Element(f"t{i}", "AXStaticText", "", t, 0, 0, 10, 10) for i, t in enumerate(texts)]
        return ax._Within("sheet", object(), "", (0, 0, 400, 200), els)

    assert ax._as_blocker(within("a", "b"), PID, True).complete
    assert not ax._as_blocker(within("a", "b"), PID).complete  # a walk nobody vouched for: not whole
    many = ax._as_blocker(within(*(f"line {i}" for i in range(ax._BLOCKER_TEXTS + 1))), PID, True)
    assert not many.complete and len(many.texts) == ax._BLOCKER_TEXTS
    long = ax._as_blocker(within("x" * (ax._BLOCKER_CHARS + 1)), PID, True)
    assert not long.complete and long.texts == []
    spaced = ax._as_blocker(within("a" + " " * ax._BLOCKER_CHARS + "lost"), PID, True)  # (cut before it shrank)
    assert not spaced.complete
