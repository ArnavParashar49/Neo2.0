"""Hard bounds on the work and memory of NEO's screen readers, and the system menu told by its place
alone.

- Accessibility: an element's text is kept short (a terminal's scrollback is never copied whole into
  every look), the id registry is bounded by size as well as count, and every look that isn't a
  walk (the scene a press is watched with, what is under a point, a sheet check, a web tree
  settling, an element's on-screen part) has a deadline, reads a bounded number of elements and
  stops at an app that stopped answering.
- The system menu: a menu bar's first item is the system's however many items the bar has.
- The flow memory learns no wording and no recipe from a request that read untrusted content before
  the model chose (DECISIONS: a path learned while reading a page could have been steered by it).
- Reading the screen's words: linear in the lines read (a spreadsheet drawn on a canvas reads
  thousands), the places a find lists are capped, and a look never waits on text marks for long.
- A flat board's state read holds a few rows of sums, not a float copy of the whole picture.

Hermetic: the ax-core suite's fake Accessibility layer and fake clock, a fake Vision, synthetic
pictures."""
# ruff: noqa: F811 - the ax-core suite's fixtures (fake, clock), imported by name

from __future__ import annotations

import asyncio
import gc
import random
import time
import tracemalloc

import numpy as np
import pytest
from PIL import Image

from neo2.core.tools.computer import ax, marks, ocr, wait
from tests.test_ax_core import (  # noqa: F401 - fixtures
    _REAL_HIT,
    _REAL_SCENE,
    El,
    app_root,
    button,
    calls,
    clock,
    fake,
    window,
)

# ---- Accessibility: an element's text ------------------------------------------------------------------------


def _log_app(fake, chars: int, pid: int = 7):
    area = El("AXTextArea", frame=(0, 60, 1500, 900))
    kids = [button(f"B{i}", frame=(10 + i * 30, 30, 28, 20)) for i in range(40)]
    win = window("Log", area, *kids)
    fake.add_app("Term", pid, app_root(win))
    base = "x" * chars
    reads = []

    def value():  # (PyObjC makes a new str on every read, as the real IPC does)
        reads.append(1)
        return base[:-1] + "y"

    fake.values[(id(area), "AXValue")] = value
    area.attrs["AXNumberOfCharacters"] = chars

    def in_part(el, name, span, _):  # AXStringForRange: a read that can't be longer than asked
        if err := fake._cost(el, "range", name):
            return err, None
        if el is not area or name != "AXStringForRange":
            return -25205, None
        start, n = span
        return 0, base[start : start + n]

    fake.AXUIElementCopyParameterizedAttributeValue = in_part
    fake.AXValueCreate = lambda kind, span: span
    fake.kAXValueCFRangeType = 4
    return area, reads


def _held() -> int:
    els = {id(e): e for e in [*ax._issued.values(), *ax._last_tree.values()]}.values()
    return sum(len(e.value) + len(e.title) for e in els)


def test_a_long_text_is_kept_short_in_every_look(fake, monkeypatch):
    area, _reads = _log_app(fake, 4 << 20)  # a 4 MB build log
    monkeypatch.setattr(marks, "_off_limits", lambda name, pid: False)
    for _ in range(12):
        ax.snapshot("Term")
        marks.collect_sync(None, pid=7, name="Term")
    got = [e for e in ax._issued.values() if e.ref is area]
    assert got and all(len(e.value) <= ax._TEXT_MAX for e in got)
    assert got[-1].value.startswith("xxx")  # (its first words: what a line shows)
    assert _held() <= 4 * ax._TEXT_MAX * 12 + 200_000  # never 12 copies of 4 MB


def test_a_text_known_to_be_long_is_not_read_whole_again(fake):
    area, reads = _log_app(fake, 1 << 20)
    ax.snapshot("Term")
    first = len(reads)
    ax.forget()
    fake.calls.clear()
    ax.snapshot("Term")
    asked = [c for c in fake.calls if c[1] is area and c[0] in ("multi", "get")]
    assert first >= 1 and len(reads) == first  # (the second look asked for no whole value)
    assert asked and all("AXValue" not in (c[2] if isinstance(c[2], tuple) else (c[2],)) for c in asked)
    e = next(e for e in ax._last_tree.values() if e.ref is area)
    assert len(e.value) <= ax._TEXT_MAX


def test_the_id_registry_is_bounded_by_size_too(monkeypatch):
    monkeypatch.setattr(ax, "_issued", {})
    monkeypatch.setattr(ax, "_ISSUED_CHARS", 10_000)
    els = [ax.Element(f"z{i}", "AXStaticText", "", "w" * 1000, 0, 0, 10, 10) for i in range(50)]
    ax._issue(els)
    assert sum(len(e.value) for e in ax._issued.values()) <= 10_000
    assert "z49" in ax._issued and "z0" not in ax._issued  # (the oldest go first)


def test_kept_looks_of_apps_not_looked_at_lately_are_let_go(fake, monkeypatch):
    monkeypatch.setattr(ax, "_watch_gen", lambda pid: 1)
    fake.add_app("A", 1, app_root(window("A", button("One"))))
    fake.add_app("B", 2, app_root(window("B", button("Two"))))
    ax.snapshot("A")
    assert 1 in ax._cache
    kept = list(ax._cache[1])
    kept[4] -= ax._CACHE_MAX_S + 1  # (A's look is older than any look that could be used again)
    ax._cache[1] = tuple(kept)
    ax.snapshot("B")
    assert 1 not in ax._cache and 2 in ax._cache


# ---- Accessibility: every look is bounded ----------------------------------------------------------------------


def _groups(n):
    return [El("AXGroup", frame=(10 + (i % 50) * 20, 60 + (i // 50) * 20, 18, 18)) for i in range(n)]


@pytest.mark.parametrize("n", [20, 5000])
def test_the_scene_reads_a_bounded_number_of_elements(fake, n):
    fake.add_app("App", 7, app_root(window("W", *_groups(n))))
    got, sc = calls(fake, lambda: _REAL_SCENE(7, front=("App", 7)))
    assert got <= 40 and sc.known and sc.title == "W"


@pytest.mark.parametrize("n", [20, 5000])
def test_whats_under_a_point_reads_a_bounded_number_of_elements(fake, n):
    kids = _groups(n)
    fake.add_app("App", 7, app_root(window("W", *kids)))
    fake.at_point = kids[0]
    got, h = calls(fake, lambda: _REAL_HIT(20, 70))
    assert got <= 40 and h.pid == 7 and not h.sheet


def test_a_sheet_added_last_is_still_found_on_a_window_with_many_children(fake):
    sheet = El("AXSheet", frame=(500, 60, 400, 200), children=[button("Start", frame=(800, 220, 60, 24))])
    kids = _groups(3000)
    fake.add_app("App", 7, app_root(window("W", *kids, sheet)))
    fake.at_point = kids[0]
    assert "a sheet" in _REAL_SCENE(7, front=("App", 7)).sheet
    assert "Start" in _REAL_HIT(20, 70).sheet


def _busy(fake, clock, n=30):
    kids = _groups(n)
    win = window("W", *kids)
    fake.add_app("App", 7, app_root(win))
    fake.clock = clock
    ax._ensure_timeout()
    for k in kids:
        k.hung = True  # the app went busy after answering the window: each call waits out the timeout
    return win, kids


def test_a_scene_of_an_app_that_went_busy_costs_about_one_timeout(fake, clock):
    _busy(fake, clock)
    t0 = wait.now()
    sc = _REAL_SCENE(7, front=("App", 7))
    assert wait.now() - t0 <= 2 * ax._AX_TIMEOUT_S and sc.title == "W"


def test_whats_under_a_point_of_an_app_that_went_busy_costs_about_one_timeout(fake, clock):
    _win, kids = _busy(fake, clock)
    fake.at_point = El("AXButton", "Go", frame=(10, 60, 18, 18))
    kids[0].add(fake.at_point)
    fake.at_point.hung = False
    t0 = wait.now()
    _REAL_HIT(20, 70)
    assert wait.now() - t0 <= 2 * ax._AX_TIMEOUT_S


def test_a_sheet_check_keeps_to_its_deadline(fake, clock):
    win, _kids = _busy(fake, clock, 20)
    t0 = wait.now()
    assert ax._sheet_of(win) is None
    assert wait.now() - t0 <= 2 * ax._AX_TIMEOUT_S


def test_the_screenshots_sheet_check_keeps_to_the_looks_deadline(fake, clock, monkeypatch):
    _busy(fake, clock, 20)
    monkeypatch.setattr(marks.time, "monotonic", wait.now)  # (the walk's deadline on the same fake clock)
    t0 = wait.now()
    marks.walk_window(7, wait.now() + 0.8)
    assert wait.now() - t0 <= 0.8 + 2 * ax._AX_TIMEOUT_S


def test_a_web_tree_settling_keeps_to_its_budget(fake, clock):
    win, _kids = _busy(fake, clock, 12)
    t0 = wait.now()
    ax._web_settle(win)
    assert wait.now() - t0 <= ax._WEB_SETTLE_S + ax._AX_TIMEOUT_S + 0.1


def test_an_elements_on_screen_part_stops_at_a_busy_app(fake, clock):
    inner = El("AXButton", "Go", frame=(10, 60, 18, 18))
    chain = inner
    for _ in range(12):
        chain = El("AXScrollArea", frame=(0, 0, 900, 900), children=[chain])
    fake.add_app("App", 7, app_root(window("W", chain)))
    fake.clock = clock
    ax._ensure_timeout()
    p = inner.parent
    while p is not None:
        p.hung = True
        p = p.parent
    t0 = wait.now()
    ax.visible_frame(inner)
    assert wait.now() - t0 <= 2 * ax._AX_TIMEOUT_S


# ---- the system menu ------------------------------------------------------------------------------------------


def _apple_only(fake, pid=61):
    sleep = El("AXMenuItem", "Sleep", actions=("AXPress",))
    restart = El("AXMenuItem", "Restart…", actions=("AXPress",))
    apple = El("AXMenuBarItem", "Apple", frame=(10, 0, 30, 24), actions=("AXPress",),
               children=[El("AXMenu", frame=(0, 0, 0, 0), children=[sleep, restart])])  # fmt: skip
    bar = El("AXMenuBar", frame=(0, 0, 1512, 24), children=[apple])
    fake.add_app("Lone", pid, app_root(window("Lone"), bar, AXMenuBar=bar))
    return apple, sleep


def test_a_menu_bar_with_one_item_has_only_the_system_menu(fake):
    apple, sleep = _apple_only(fake)
    assert ax.menu_items(61) == []
    assert ax.in_system_menu(sleep) and ax.in_system_menu(apple)
    e = ax.Element("e1", "AXMenuItem", "Sleep", "", 0, 30, 100, 20, actions=["AXPress"], ref=sleep)
    assert ax.press_el(e) == ax._SYSTEM_MENU and fake.performed == []
    assert "Apple" not in ax.snapshot("Lone")  # (the menu bar line leaves it out too)


# ---- the flow memory learns nothing a page could have steered -------------------------------------------------


class _Ctx:
    def __init__(self, origin="agent", user_text="", taint=None):
        self.origin, self.user_text, self.extras, self.taint = origin, user_text, {}, list(taint or [])

    def get(self, k, default=None):
        return self.extras.get(k, default)


def _handed_off(text: str, ctx):
    from neo2.core.tools.computer import resolve as rs

    cands = [rs.Candidate(t, "AXButton", box=(0, 40.0 * i, 80, 20)) for i, t in enumerate(("Start Game", "Start Over", "Hint"))]
    amb = rs.resolve(text, cands, index=False, embed=False, choose=False, alias=False, log_it=False)
    assert isinstance(amb, rs.Ambiguous) and amb.shortlist
    rs.handoff(amb, text, ctx=ctx, register=False)
    # the model, in the same request, calls click_text(label="Start Over")
    return rs.resolve("Start Over", cands, origin="agent", index=False, embed=False, choose=False, alias=False, log_it=False)


def test_a_pick_made_after_reading_a_page_is_marked_untrusted():
    from neo2.core.tools.computer import resolve as rs

    ctx = _Ctx(user_text="click the start button", taint=["browse"])  # a page was read first
    pick = _handed_off("click the start button", ctx)
    sig = rs.after_press(pick, verified=True, bundle="org.example.mail", ctx=ctx)
    assert sig["alias"] is True and sig["untrusted"] is True


def test_a_pick_made_after_the_screen_was_read_before_the_shortlist_is_untrusted():
    from neo2.core.tools.computer import resolve as rs

    ctx = _Ctx(user_text="click start now", taint=["screen"])  # (a screenshot of the page first)
    pick = _handed_off("click start now", ctx)
    assert rs.after_press(pick, verified=True, bundle="b", ctx=ctx)["untrusted"] is True


def test_a_pick_from_the_shortlist_alone_is_trusted():
    from neo2.core.tools.computer import resolve as rs

    ctx = _Ctx(user_text="click start")
    pick = _handed_off("click start", ctx)
    assert ctx.taint == [rs.TAINT]  # (the shortlist's own labels)
    sig = rs.after_press(pick, verified=True, bundle="b", ctx=ctx)
    assert sig["alias"] is True and sig["untrusted"] is False


def test_a_screen_read_between_the_shortlist_and_the_pick_makes_it_untrusted():
    from neo2.core.agent import grants
    from neo2.core.tools.computer import resolve as rs

    ctx = _Ctx(user_text="press start")
    with grants.task(grants.new_task()):
        pick = _handed_off("press start", ctx)
        grants.read_untrusted("screen")  # the model took a screenshot of the page before it chose
        asyncio.run(rs.ground({"label": "Start Over"}, ctx, []))  # (its click_text call starts)
        assert rs.after_press(pick, verified=True, bundle="b", ctx=ctx)["untrusted"] is True


def test_a_models_own_press_after_reading_the_screen_is_untrusted_the_users_words_are_not():
    from neo2.core.tools.computer import resolve as rs

    cands = [rs.Candidate("Start", "AXButton", box=(0, 0, 80, 20))]
    model = rs.resolve("Start", cands, origin="agent", index=False, embed=False, choose=False, alias=False, log_it=False)
    assert rs.after_press(model, verified=True, bundle="b", ctx=_Ctx("agent", "go", ["screen"]))["untrusted"] is True
    user = rs.resolve("click start", cands, origin="user", index=False, embed=False, choose=False, alias=False, log_it=False)
    assert rs.after_press(user, verified=True, bundle="b", ctx=_Ctx("user", "click start", ["screen"]))["untrusted"] is False
    assert rs.after_press(user, verified=True, bundle="b", ctx=_Ctx("user", "click start", ["read_email"]))["untrusted"] is True


def test_a_request_that_read_untrusted_content_teaches_no_alias_and_no_recipe(monkeypatch, tmp_path):
    from neo2.core.reflex import uimemory

    mem = uimemory.UIMemory(path=lambda: tmp_path / "ui_memory.json", clock=lambda: 1_800_000_000.0)
    monkeypatch.setattr(uimemory, "_memory", mem)
    pick = {"kind": "press", "role": "AXButton", "label": "Start Over", "bundle": "org.example.mail",
            "expect": {}, "changed": True, "alias": True, "untrusted": True}  # fmt: skip
    assert uimemory.settled("click the start button", [pick]) is None
    assert mem.alias("click the start button", "org.example.mail") is None and len(mem) == 0
    uimemory.settled("click the start button", [{**pick, "untrusted": False}])
    assert mem.alias("click the start button", "org.example.mail") == "Start Over"  # (a clean pick is learned)


# ---- the screen's words: linear, capped, never waited on long ------------------------------------------------


class _Dense:
    """Reads a grid of ``n`` cells over the picture, like Vision on a spreadsheet drawn on a canvas."""

    def __init__(self, n: int, word=None) -> None:
        self.n, self.word, self.calls = n, word, 0

    def languages(self, fast):
        return ["en-US"]

    def recognize(self, image, *, fast, languages, auto, ranges):
        self.calls += 1
        cols = 20
        rows = max(1, (self.n + cols - 1) // cols)
        out = []
        for i in range(self.n):
            r, c = divmod(i, cols)
            text = self.word or f"w{(i * 7) % 50} w{(i * 13) % 50}"
            x = c / cols + (0.003 if fast else 0.0)
            out.append(ocr.Raw(text, 0.5 if fast else 1.0, (x, 1.0 - (r + 1) / rows, 0.8 / cols, 0.6 / rows)))
        return out


@pytest.fixture
def dense(monkeypatch):
    monkeypatch.setattr(ocr, "_engine", None)
    monkeypatch.setattr(ocr, "_marks", {})
    monkeypatch.setattr(ocr, "_preferred", ["en-US"])
    monkeypatch.setattr(ocr, "_windows", lambda Q, _c, ax_, area: ocr.Windows((ocr.Win((-1e5, -1e5, 2e5, 2e5), 7, "App"),)))
    return ocr.Pixels(Image.new("RGB", (3024, 1964), (255, 255, 255)), 0.0, 0.0, 0.5)


def test_text_marks_on_a_dense_canvas_are_quick(dense):
    ocr.set_engine(_Dense(3000))
    ocr.text_marks(dense, pid=7)  # (warm)
    t0 = time.perf_counter()
    got = ocr.text_marks(dense, pid=7)
    assert time.perf_counter() - t0 < 0.6 and 0 < len(got) <= ocr.MAX_MARKS


def test_a_find_of_missing_words_on_a_dense_canvas_is_quick(dense):
    ocr.set_engine(_Dense(3000))
    t0 = time.perf_counter()
    got = ocr.find_in(dense, "w999 w998")
    assert isinstance(got, ocr.Missing) and time.perf_counter() - t0 < 1.0


def test_words_shown_in_a_thousand_places_are_listed_in_a_few_quickly(dense):
    ocr.set_engine(_Dense(1000, word="Item"))
    t0 = time.perf_counter()
    got = ocr.find_in(dense, "Item")
    assert isinstance(got, ocr.Ambiguous) and 2 <= len(got.options) <= ocr._MAX_PLACES
    assert time.perf_counter() - t0 < 1.0
    ys = [(round(f.box[1] / 6), f.box[0]) for f in got.options]
    assert ys == sorted(ys)  # (the first places in reading order)


def _random_reading(rng: random.Random, n: int, jitter: float) -> ocr.Reading:
    lines = []
    for _ in range(n):
        x, y = rng.uniform(0, 900), rng.uniform(0, 600)
        w, h = rng.choice((8.0, 30.0, 120.0, 400.0)), rng.choice((6.0, 11.0, 24.0))
        text = rng.choice(("Start", "Stop", "Item 1", "Item 2", "OK", "w1 w2", "Cancel"))
        lines.append(ocr.Line(text, 1.0, (x + rng.uniform(-jitter, jitter), y, w, h)))
    lines.sort(key=lambda ln: (round(ln.box[1] / 6), ln.box[0]))
    return ocr.Reading(tuple(lines), (0, 0, 1000, 700), "accurate", 1.0, 0.0, 2.0)


def _merge_by_pairs(*readings, keep_all_words=False):
    kept = []
    for r in readings:
        for ln in r.lines:
            k = ocr.key(ln.text)
            if not any((ocr._inside(ln.center, o.box) or ocr._inside(o.center, ln.box)) and (not keep_all_words or ocr.key(o.text) == k) for o in kept):
                kept.append(ln)
    kept.sort(key=lambda ln: (round(ln.box[1] / 6), ln.box[0]))
    return kept


@pytest.mark.parametrize("seed", range(6))
def test_the_quick_merge_keeps_exactly_what_comparing_every_pair_kept(seed):
    rng = random.Random(seed)
    a, b, c = (_random_reading(rng, 250, 3.0) for _ in range(3))
    for keep_all in (False, True):
        assert list(ocr.merge(a, b, c, keep_all_words=keep_all).lines) == _merge_by_pairs(a, b, c, keep_all_words=keep_all)


def _diff_by_pairs(before, after):
    """ocr.diff as it was: every before line against every after line."""
    b_lines = [ln for ln in before.lines if ln.conf >= ocr._MIN_CONF]
    a_lines = [ln for ln in after.lines if ln.conf >= ocr._MIN_CONF]
    used, gone, changed = set(), [], []
    for b in b_lines:
        j = next((j for j, a in enumerate(a_lines) if j not in used and ocr._similar(a.text, b.text) and ocr._overlap(a.box, b.box)), None)
        if j is None:
            j = next((j for j, a in enumerate(a_lines) if j not in used and ocr._overlap(a.box, b.box) and (ocr._inside(a.center, b.box) or ocr._inside(b.center, a.box))), None)
            if j is not None and not ocr._similar(a_lines[j].text, b.text):
                changed.append((b.text, a_lines[j].text))
        if j is not None:
            used.add(j)
            continue
        j = next((j for j, a in enumerate(a_lines) if j not in used and ocr._similar(a.text, b.text)), None)
        if j is not None:
            used.add(j)
            continue
        gone.append(b.text)
    return tuple(gone), tuple(a.text for j, a in enumerate(a_lines) if j not in used), tuple(changed)


@pytest.mark.parametrize("seed", range(8))
def test_the_quick_diff_says_exactly_what_comparing_every_pair_said(seed):
    rng = random.Random(100 + seed)
    a, b = _random_reading(rng, 120, 2.0), _random_reading(rng, 120, 2.0)
    got = ocr.diff(a, b)
    assert (got.gone, got.new, got.changed) == _diff_by_pairs(a, b)


def test_merging_two_dense_readings_is_linear():
    rng = random.Random(9)
    a, b = _random_reading(rng, 4000, 1.0), _random_reading(rng, 4000, 1.0)
    t0 = time.perf_counter()
    ocr.merge(a, b, keep_all_words=True)
    assert time.perf_counter() - t0 < 0.8


def test_the_words_that_changed_on_a_dense_screen_are_told_quickly():
    rng = random.Random(3)
    a, b = _random_reading(rng, 3000, 0.0), _random_reading(rng, 3000, 0.0)
    t0 = time.perf_counter()
    ocr.diff(a, b)
    assert time.perf_counter() - t0 < 1.0


def test_a_look_never_waits_long_for_its_text_marks(monkeypatch, dense):
    ocr.set_engine(_Dense(10))

    def slow(*a, **kw):
        time.sleep(0.6)
        return ["late"]

    monkeypatch.setattr(ocr, "text_marks", slow)
    monkeypatch.setattr(ocr, "LOOK_BUDGET_S", 0.1)

    async def look():
        t0 = time.perf_counter()
        got = await ocr.look_marks(dense, blind=True)
        return got, time.perf_counter() - t0

    got, took = asyncio.run(look())  # (the read itself is left to finish on its own)
    assert got == [] and took < 0.4


# ---- a flat board's state ----------------------------------------------------------------------------------------


def test_the_integral_at_row_and_column_boundaries_is_exact():
    from neo2.core.tools.computer import board as B

    rng = np.random.default_rng(1)
    crop = rng.integers(0, 256, (97, 131, 3), dtype=np.uint8)
    rows = np.unique(np.r_[0, rng.integers(0, 98, 20), 97])
    cols = np.unique(np.r_[rng.integers(0, 132, 25), 131])
    got = B._integral(crop, rows, cols)
    want = np.zeros((len(rows), len(cols), 3))
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            want[i, j] = crop[:r, :c].reshape(-1, 3).sum(axis=0)
    assert np.array_equal(got, want)
    pts = B._integral_at(crop, np.array([0.0, 10.5, 96.9]), np.array([130.2, 0.0, 44.25]), grid=False)
    grid = B._integral_at(crop, np.array([0.0, 10.5, 96.9]), np.array([130.2, 0.0, 44.25]))
    assert np.allclose(pts, grid[[0, 1, 2], [0, 1, 2]])


def test_a_flat_boards_state_holds_little_memory_on_a_big_display():
    from neo2.core.tools.computer import board as B

    w, h, n = 3008, 1692, 24
    rng = np.random.default_rng(0)
    pic = B.Picture.of(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
    b = B.Board(np.array([[w / n, 0, 0], [0, h / n, 0], [0, 0, 1.0]]), n, n, "neo", flat=True)
    gc.collect()
    tracemalloc.start()
    try:
        B._take(pic, b)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 60 * 2**20, f"{peak / 2**20:.0f} MB"  # (it was 230 MB: a float copy of the picture and more)
