"""Reading the screen's words (neo2.core.tools.computer.ocr): where they are in screen points, which read
(fast or accurate) by need, finding a label (an open sheet's first, never a guess between equals),
text marks a brain that can't see may click by id (never a committing one), the words that changed
after a click, and click_text's fallback through the same probe → gate → click path as any click.

Hermetic: Apple Vision is replaced by a fake engine that reads synthetic pictures the way Vision
reports (boxes normalised to the picture it is handed, origin bottom left), so NEO's own cropping,
scaling up and flipping are what is tested. Nothing is captured, clicked or loaded."""

from __future__ import annotations

import asyncio
import re
import time
import types

import numpy as np
import pytest
from PIL import Image, ImageDraw

from neo2.core import effects, runtime
from neo2.core.agent import confirm
from neo2.core.agent.registry import ToolContext, ToolOutput
from neo2.core.tools import computer
from neo2.core.tools.computer import apps, ax, focus, ocr, pointer
from neo2.core.tools.computer import input as inp

# What Vision on this Mac (macOS 26, revision 3) says it reads: no Hindi.
VISION_ACCURATE = [
    "en-US", "fr-FR", "it-IT", "de-DE", "es-ES", "pt-BR", "zh-Hans", "zh-Hant", "yue-Hans", "yue-Hant", "ko-KR",
    "ja-JP", "ru-RU", "uk-UA", "th-TH", "vi-VT", "ar-SA", "ars-SA", "tr-TR", "id-ID", "cs-CZ", "da-DK", "nl-NL",
    "no-NO", "nn-NO", "nb-NO", "ms-MY", "pl-PL", "ro-RO", "sv-SE",
]  # fmt: skip
VISION_FAST = ["en-US", "fr-FR", "it-IT", "de-DE", "es-ES", "pt-BR"]
APP, PID, NEO, OTHER = "Board", 4242, 900, 55


# ---- a fake Vision -------------------------------------------------------------------------------------


class FakeVision:
    """Reads a synthetic picture: each line of words is a flat block of its own grey level. The engine
    finds each block in the picture it is handed (after NEO's crop and scaling) and reports it as Vision
    does — a box normalised to that picture with the origin at the bottom left. Like Vision, a fast read
    of words fewer than ``fast_min_px`` pixels tall garbles them, and words in a script none of the
    languages asked for is written in come back as nothing; with ``fast_drop_px``, a fast read drops
    words fewer pixels tall than that altogether."""

    def __init__(
        self, words: dict[int, str], *, fast_min_px: int = 16, fast_drop_px: int = 0, acc=VISION_ACCURATE, fast=VISION_FAST, drop=None
    ):
        self.words = words
        self.fast_min_px = fast_min_px
        self.fast_drop_px = fast_drop_px  # a fast read drops words fewer pixels tall than this (Vision does)
        self.drop = drop  # drop(words, fast, n-th call) → True: this read misses them (Vision does, now and then)
        self.acc, self.fst = list(acc), list(fast)
        self.calls: list[dict] = []

    def languages(self, fast: bool) -> list[str]:
        return list(self.fst if fast else self.acc)

    @staticmethod
    def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
        rows, cols = mask.sum(axis=1), mask.sum(axis=0)
        if rows.max(initial=0) == 0:
            return None
        ys = np.nonzero(rows >= 0.3 * rows.max())[0]  # (a resize's stray edge pixels don't count)
        xs = np.nonzero(cols >= 0.3 * cols.max())[0]
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    def recognize(self, image, *, fast, languages, auto, ranges):
        self.calls.append({"fast": fast, "size": image.size, "languages": list(languages), "auto": auto, "mode": image.mode})
        arr = np.asarray(image.convert("L"))
        H, W = arr.shape
        out = []
        readable = set().union(*(ocr._lang_scripts(c) for c in languages)) if not auto else {"LATIN", "CJK", "CYRILLIC"}
        for level, text in self.words.items():
            b = self._bbox(arr == level)
            if b is None:
                continue
            x0, y0, x1, y1 = b
            if not ocr.scripts(text) <= readable | {"LATIN"} or "DEVANAGARI" in ocr.scripts(text):
                continue  # (Vision returns nothing for a script it can't read)
            if self.drop and self.drop(text, fast, len(self.calls)):
                continue
            if fast and y1 - y0 < self.fast_drop_px:
                continue
            said = text
            if fast and y1 - y0 < self.fast_min_px:
                said = text.replace("a", "s").replace("n", "rK")  # "Start" → "Stsrt", "Cancel" → "CarKcel"
            box = (x0 / W, 1 - y1 / H, (x1 - x0) / W, (y1 - y0) / H)
            n = max(1, len(said))
            parts = tuple(
                (s, e, (box[0] + box[2] * s / n, box[1], box[2] * (e - s) / n, box[3])) for s, e in (ranges(said) if ranges else [])
            )
            out.append(ocr.Raw(said, 0.5 if fast else 1.0, box, parts))
        return out


def scene(items, *, size_pt=(400.0, 300.0), ppp=0.5, x0=0.0, y0=0.0):
    """(frame, engine): ``items`` are (words, (x, y, w, h) in screen points) drawn as blocks."""
    W, H = round(size_pt[0] / ppp), round(size_pt[1] / ppp)
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)
    words = {}
    for i, (text, (x, y, w, h)) in enumerate(items):
        level = 10 + 12 * i
        d.rectangle([round((x - x0) / ppp), round((y - y0) / ppp), round((x + w - x0) / ppp) - 1, round((y + h - y0) / ppp) - 1], fill=level)
        words[level] = text
    return ocr.Pixels(img, x0, y0, ppp), FakeVision(words)


@pytest.fixture(autouse=True)
def _fresh_ocr(monkeypatch):
    monkeypatch.setattr(ocr, "_engine", None)
    monkeypatch.setattr(ocr, "_marks", {})
    monkeypatch.setattr(ocr, "_preferred", ["en-IN", "hi-IN"])  # (this Mac's user: English and Hindi)
    yield


def use(engine):
    ocr.set_engine(engine)
    return engine


BOARD_WIN = ocr.Win((0.0, 0.0, 400.0, 300.0), PID, APP)


def windows(monkeypatch, *wins):
    """The window server's windows over any area, front to back (Board's window alone by default)."""
    listed = ocr.Windows(tuple(wins) or (BOARD_WIN,))
    monkeypatch.setattr(ocr, "windows_over", lambda area: listed)
    return listed


def world(monkeypatch, items, *wins, **kw):
    """A screen showing ``items`` (see :func:`scene`) that NEO may capture and read, on these windows."""
    frame, eng = scene(items, **kw)
    use(eng)
    monkeypatch.setattr(ocr, "grab", lambda area: frame)
    windows(monkeypatch, *wins)
    return frame, eng


def near(a, b, tol=1.5):
    return all(abs(u - v) <= tol for u, v in zip(a, b, strict=True))


# ---- where the words are ---------------------------------------------------------------------------------


def test_boxes_come_back_in_screen_points_from_the_top_left():
    # A Retina window at (100, 50): Vision's boxes are 0–1 of the picture, from the bottom left.
    frame, eng = scene([("Start", (220.0, 90.0, 40.0, 14.0))], x0=100.0, y0=50.0)
    use(eng)
    r = ocr.read(frame)
    assert [ln.text for ln in r.lines] == ["Start"]
    assert near(r.lines[0].box, (220.0, 90.0, 40.0, 14.0), 1.0)
    assert r.area == (100.0, 50.0, 400.0, 300.0) and r.level == ocr.FAST


def test_an_area_is_read_at_the_frames_own_detail_and_mapped_back():
    frame, eng = scene([("Cancel", (60.0, 200.0, 50.0, 14.0)), ("Start", (300.0, 40.0, 40.0, 14.0))])
    use(eng)
    r = ocr.read(frame, (250.0, 20.0, 120.0, 60.0))
    assert [ln.text for ln in r.lines] == ["Start"] and near(r.lines[0].box, (300.0, 40.0, 40.0, 14.0), 1.0)
    assert eng.calls[-1]["size"] == (240, 120)  # 2 pixels per point, as captured: not shrunk
    assert eng.calls[-1]["mode"] == "L"  # handed over as grey pixels (no PNG on the way)


def test_a_fast_read_of_a_picture_with_one_pixel_per_point_is_scaled_up_first():
    # 12-pt words on a 1-pixel-per-point picture: fast mode would garble them ("Stsrt").
    frame, eng = scene([("Start", (100.0, 100.0, 40.0, 12.0))], ppp=1.0)
    use(eng)
    r = ocr.read(frame)
    assert eng.calls[-1]["size"] == (800, 600) and r.px_per_pt == pytest.approx(2.0)
    assert [ln.text for ln in r.lines] == ["Start"] and near(r.lines[0].center, (120.0, 106.0), 0.5)
    # The accurate read needs no scaling; a Retina picture never gets any.
    ocr.read(frame, plan=ocr.Plan(False, ("en-US",)))
    assert eng.calls[-1]["size"] == (400, 300)
    frame2, eng2 = scene([("Start", (100.0, 100.0, 40.0, 12.0))])
    use(eng2)
    ocr.read(frame2)
    assert eng2.calls[-1]["size"] == (800, 600)


def test_the_capture_pipelines_frame_is_taken_as_it_is():
    frame, eng = scene([("Start", (120.0, 60.0, 40.0, 14.0))], x0=30.0, y0=40.0)
    use(eng)
    like = types.SimpleNamespace(image=frame.image, x0=30.0, y0=40.0, ppp=0.5, t=time.monotonic(), window_id=0)
    assert near(ocr.read(like).lines[0].box, (120.0, 60.0, 40.0, 14.0), 1.0)


def test_protected_windows_words_never_count():
    frame, eng = scene([("Approve", (40.0, 40.0, 60.0, 14.0)), ("Start", (200.0, 40.0, 40.0, 14.0))])
    use(eng)
    r = ocr.without(ocr.read(frame), [(0.0, 0.0, 150.0, 300.0)])
    assert [ln.text for ln in r.lines] == ["Start"]


# ---- fast or accurate, which languages -----------------------------------------------------------------


def test_latin_words_are_read_fast_other_scripts_accurately_and_devanagari_not_at_all():
    use(FakeVision({}))
    p = ocr.plan_for("Start")
    assert p.fast and p.languages[0] == "en-US"
    p = ocr.plan_for("Сохранить")  # Cyrillic: only the accurate mode reads it
    assert not p.fast and set(p.languages) <= {"ru-RU", "uk-UA"}
    p = ocr.plan_for("キャンセル")
    assert not p.fast and p.languages == ("ja-JP",)
    assert ocr.plan_for("नया खेल") is None and ocr.unreadable("नया खेल") == "Devanagari"
    assert ocr.unreadable("Start 2") == "" and ocr.plan_for("42").fast  # digits: any language reads them
    assert ocr.plan_for("Start", want_fast=False).fast is False
    p = ocr.plan_for("Save Сохранить")  # two scripts: accurate, in a language of each
    assert not p.fast and "en-US" in p.languages and {"ru-RU", "uk-UA"} & set(p.languages)


def test_reading_whatever_is_there_follows_the_macs_languages(monkeypatch):
    use(FakeVision({}))
    assert ocr.plan_for("").fast  # English (and Hindi, which Vision can't read): fast
    monkeypatch.setattr(ocr, "_preferred", ["ja-JP", "en-US"])
    p = ocr.plan_for("")
    assert not p.fast and p.auto and "ja-JP" in p.languages  # a Japanese UI: accurate, language guessed


def test_scripts_are_told_by_the_letters_not_by_signs_or_marks():
    assert ocr.scripts("Start 42 …!") == {"LATIN"}
    assert ocr.scripts("नया खेल") == {"DEVANAGARI"}
    assert ocr.scripts("コーヒー") == {"KATAKANA"}  # (the long-vowel mark is kana too)
    assert ocr.scripts("日々") == {"CJK"}
    assert ocr.scripts("ｓｔａｒｔ") == {"LATIN"}  # full-width forms are Latin letters


# ---- finding a label ---------------------------------------------------------------------------------------


def _reading(lines, level=ocr.ACCURATE):
    return ocr.Reading(tuple(ocr.Line(t, c, b) for t, c, b in lines), (0, 0, 1000, 800), level, 1.0, 0.0, 2.0)


def test_a_buttons_own_label_beats_a_title_that_mentions_it():
    r = _reading([
        ("Start a New Game?", 1.0, (580.0, 110.0, 150.0, 16.0)),
        ("Cancel", 1.0, (835.0, 215.0, 50.0, 14.0)),
        ("Start", 1.0, (955.0, 215.0, 40.0, 14.0)),
    ])  # fmt: skip
    got = ocr.match(r, "start")
    assert isinstance(got, ocr.Found) and got.exact and got.point == (975.0, 222.0)
    assert "Cancel" in got.context  # the dialog's words go to the gate with it


def test_words_inside_a_longer_line_are_found_where_those_words_are():
    r = _reading([("Cancel Start", 1.0, (100.0, 100.0, 120.0, 14.0))])  # two buttons read as one line
    got = ocr.match(r, "Start")
    assert isinstance(got, ocr.Found) and not got.exact and got.text == "Start" and got.line == "Cancel Start"
    assert near(got.point, (195.0, 107.0), 0.5)  # the words' share of the line (Vision's own box when given)
    parts = ((7, 12, (170.0, 100.0, 45.0, 14.0)),)
    r2 = ocr.Reading((ocr.Line("Cancel Start", 1.0, (100.0, 100.0, 120.0, 14.0), parts),), (0, 0, 1, 1), "fast", 1, 0, 2)
    assert ocr.match(r2, "Start").point == (192.5, 107.0)


def test_the_open_sheet_comes_first():
    r = _reading([("Start", 1.0, (100.0, 300.0, 40.0, 14.0)), ("Start", 1.0, (600.0, 200.0, 40.0, 14.0))])
    assert isinstance(ocr.match(r, "Start"), ocr.Ambiguous)  # two equal places: never a guess
    got = ocr.match(r, "Start", sheet=(500.0, 100.0, 300.0, 200.0))
    assert isinstance(got, ocr.Found) and got.where == "sheet" and got.point == (620.0, 207.0)


def test_equal_places_are_ambiguous_and_listed_in_reading_order():
    r = _reading([("Delete", 1.0, (400.0, 300.0, 40.0, 14.0)), ("Delete", 1.0, (100.0, 100.0, 40.0, 14.0))])
    got = ocr.match(r, "Delete")
    assert isinstance(got, ocr.Ambiguous) and [o.point for o in got.options] == [(120.0, 107.0), (420.0, 307.0)]


@pytest.mark.parametrize(
    "shown, asked",
    [("Share…", "share"), ("“Save As…”", "save as"), ("New  Game", "new game"), ("NewGame", "New Game"), ("OK", "ok."), ("Ｏｋ", "ok")],
)
def test_labels_match_across_case_quotes_ellipses_and_spacing(shown, asked):
    got = ocr.match(_reading([(shown, 1.0, (10.0, 10.0, 60.0, 14.0))]), asked)
    assert isinstance(got, ocr.Found) and got.exact


def test_part_of_a_word_and_noise_are_not_matches():
    r = _reading([("Restart", 1.0, (10.0, 10.0, 60.0, 14.0)), ("Start", 0.3, (10.0, 40.0, 60.0, 14.0))])
    assert isinstance(ocr.match(r, "Start"), ocr.Missing)  # "Restart" isn't "Start"; 0.3 is Vision's noise


def test_a_fast_exact_find_is_taken_without_an_accurate_read():
    frame, eng = scene([("Start a New Game?", (40.0, 20.0, 160.0, 16.0)), ("Start", (300.0, 200.0, 40.0, 16.0))])
    use(eng)
    got = ocr.find_in(frame, "Start")
    assert isinstance(got, ocr.Found) and got.exact and got.level == ocr.FAST
    # Two fast reads, framed differently (the second with a border), agreeing: no accurate read.
    assert [c["fast"] for c in eng.calls] == [True, True] and eng.calls[1]["size"] == (848, 648)
    assert near(got.point, (320.0, 208.0), 1.0)


def test_a_copy_one_read_dropped_still_makes_the_find_ambiguous():
    # Two "Start" buttons; the first fast read drops one, the other reads disagree on which: the union
    # of what was read has both — ambiguous, never the one a lucky read happened to keep.
    frame, eng = scene([("Start", (40.0, 40.0, 40.0, 16.0)), ("Start", (300.0, 200.0, 40.0, 16.0))])
    order = iter([True, False, False, True, False, False])
    eng.drop = lambda text, fast, n: n == 1 and next(order)  # (call 1: the first read, drops the first copy)
    use(eng)
    got = ocr.find_in(frame, "Start")
    assert isinstance(got, ocr.Ambiguous) and len(got.options) == 2
    frame, eng = scene([("Start", (40.0, 40.0, 40.0, 16.0)), ("Start", (300.0, 200.0, 40.0, 16.0))])
    seen = []

    def only_first_read_misses(text, fast, n):
        seen.append(n)
        return n == 1 and len([m for m in seen if m == 1]) == 1  # the first read drops the first copy only

    eng.drop = only_first_read_misses
    use(eng)
    got = ocr.find_in(frame, "Start")
    assert isinstance(got, ocr.Ambiguous) and [c["fast"] for c in eng.calls] == [True, True, False]


def test_a_fast_miss_or_a_lookalike_is_settled_by_an_accurate_read():
    # 6-pt words: fast garbles them even at full Retina detail ("Cancel" → "CarKcel").
    frame, eng = scene([("Cancel", (40.0, 40.0, 50.0, 6.0)), ("Start", (300.0, 200.0, 40.0, 16.0))])
    eng.fast_min_px = 14
    use(eng)
    got = ocr.find_in(frame, "Cancel")
    assert isinstance(got, ocr.Found) and got.level == ocr.ACCURATE and [c["fast"] for c in eng.calls] == [True, False]
    # A second "Start" that fast garbles ("Stsrt") makes the fast find unsure: accurate says there are two.
    frame, eng = scene([("Start", (300.0, 200.0, 40.0, 16.0)), ("Start", (40.0, 40.0, 40.0, 6.0))])
    eng.fast_min_px = 14
    use(eng)
    got = ocr.find_in(frame, "Start")
    assert isinstance(got, ocr.Ambiguous) and [c["fast"] for c in eng.calls] == [True, False]


def test_with_a_sheet_open_a_fast_find_outside_it_is_checked_accurately():
    frame, eng = scene([("Start", (40.0, 40.0, 40.0, 16.0)), ("Start", (300.0, 200.0, 40.0, 6.0))])
    eng.fast_min_px = 14
    use(eng)
    got = ocr.find_in(frame, "Start", sheet=(250.0, 150.0, 140.0, 100.0))
    assert isinstance(got, ocr.Found) and got.where == "sheet" and near(got.point, (320.0, 203.0), 1.5)


def test_hindi_words_are_never_searched_for_and_the_answer_says_why():
    frame, eng = scene([("नया खेल", (40.0, 40.0, 60.0, 16.0))])
    use(eng)
    got = ocr.find_in(frame, "नया खेल")
    assert isinstance(got, ocr.Missing) and "Devanagari" in got.why and eng.calls == []


def test_pieces_cover_everything_and_their_cores_meet():
    for total, limit, ov in ((3024, 2400, 320), (5000, 2400, 320), (2400, 2400, 320), (100, 2400, 320)):
        cuts = ocr._cuts(total, limit, ov)
        assert cuts[0][0] == 0 and cuts[-1][1] == total and all(e - s <= limit for s, e, _, _ in cuts)
        assert cuts[0][2] == 0 and cuts[-1][3] == total and all(a[3] == b[2] for a, b in zip(cuts, cuts[1:], strict=False))
        assert all(s <= cs < ce <= e for s, e, cs, ce in cuts)


def test_a_big_accurate_read_is_cut_into_pieces_and_nothing_is_read_twice():
    # A whole Retina display (3024 px wide): read accurately in two overlapping pieces, each ≤ 2400 px.
    frame, eng = scene(
        [("Left", (40.0, 40.0, 60.0, 14.0)), ("Across the cut", (700.0, 300.0, 120.0, 14.0)), ("Right", (1400.0, 900.0, 60.0, 14.0))],
        size_pt=(1512.0, 982.0),
    )
    use(eng)
    r = ocr.read(frame, plan=ocr.Plan(False, ("en-US",)))
    assert len(eng.calls) == 2 and all(c["size"][0] <= ocr._ACCURATE_MAX_PX for c in eng.calls)
    assert [ln.text for ln in r.lines] == ["Left", "Across the cut", "Right"]
    assert near(r.lines[1].box, (700.0, 300.0, 120.0, 14.0), 1.0)  # whole, from the piece that holds it
    eng.calls.clear()
    ocr.read(frame, plan=ocr.Plan(True, ("en-US",)))
    assert len(eng.calls) == 1  # a fast read takes the display whole


def test_a_second_framing_maps_back_to_the_same_places():
    frame, eng = scene([("Start", (220.0, 90.0, 40.0, 14.0))], x0=100.0, y0=50.0)
    use(eng)
    plain, padded = ocr.read(frame), ocr.read(frame, pad=ocr._PAD_PX)
    assert eng.calls[1]["size"] == (800 + 48, 600 + 48)
    assert near(plain.lines[0].box, padded.lines[0].box, 0.01)


def test_merge_keeps_the_trusted_reads_lines_and_adds_what_it_missed():
    acc = _reading([("Start", 1.0, (100.0, 100.0, 40.0, 14.0))])
    fast = _reading([("Stsrt", 0.5, (101.0, 100.0, 40.0, 14.0)), ("Hint", 0.5, (300.0, 100.0, 40.0, 14.0))], ocr.FAST)
    m = ocr.merge(acc, fast)
    assert [ln.text for ln in m.lines] == ["Start", "Hint"] and m.level == ocr.ACCURATE


def test_a_garbled_accurate_read_cant_hide_what_a_fast_read_got_right():
    # Measured: the accurate mode read a bordered "Start" as "StartR" in some framings, while fast read
    # both copies right. For a find, every read's words count.
    acc = _reading([("StartR", 1.0, (953.0, 217.0, 31.0, 13.0)), ("Start", 1.0, (1111.0, 211.0, 30.0, 14.0))])
    fast = _reading([("Start", 0.5, (955.0, 218.0, 26.0, 10.0)), ("Start", 0.5, (1112.0, 212.0, 26.0, 10.0))], ocr.FAST)
    assert isinstance(ocr.match(ocr.merge(acc, fast, keep_all_words=True), "Start"), ocr.Ambiguous)
    got = ocr.match(ocr.merge(acc, fast, keep_all_words=True), "Start", sheet=(560.0, 90.0, 500.0, 170.0))
    assert isinstance(got, ocr.Found) and got.exact and near(got.point, (968.0, 223.0), 0.5)
    assert isinstance(ocr.match(ocr.merge(acc, fast), "Start"), ocr.Found)  # (a listing's merge would have hidden it)


def test_other_words_that_share_letters_are_no_lookalikes():
    r = _reading([("Start", 0.5, (100.0, 100.0, 40.0, 14.0)), ("Share…", 0.5, (100.0, 200.0, 40.0, 14.0)), ("Stsrt", 0.5, (300.0, 300.0, 40.0, 14.0))])
    found = ocr.match(_reading([("Start", 0.5, (100.0, 100.0, 40.0, 14.0))]), "Start")
    assert ocr._lookalikes(r, "Start", found)  # "Stsrt": a garbled copy
    r2 = _reading([("Start", 0.5, (100.0, 100.0, 40.0, 14.0)), ("Share…", 0.5, (100.0, 200.0, 40.0, 14.0)), ("Settings", 0.5, (100.0, 240.0, 40.0, 14.0))])
    assert not ocr._lookalikes(r2, "Start", found)


# ---- text marks ----------------------------------------------------------------------------------------------


def test_marks_list_what_accessibility_doesnt_name_the_sheets_first():
    r = _reading([
        ("Score: 12", 1.0, (20.0, 20.0, 80.0, 14.0)),
        ("New Game", 1.0, (820.0, 20.0, 70.0, 14.0)),  # Accessibility names this button
        ("Start a New Game?", 1.0, (400.0, 300.0, 150.0, 16.0)),
        ("—", 1.0, (10.0, 700.0, 10.0, 14.0)),  # no letter or digit: nothing to act on
        ("Start", 1.0, (500.0, 360.0, 40.0, 14.0)),
    ])  # fmt: skip
    covered = [((810.0, 15.0, 90.0, 24.0), "New Game")]
    got = ocr.pick_marks(r, covered=covered, sheet=(380.0, 280.0, 300.0, 120.0))
    assert [(t, w) for t, _, w in got] == [("Start a New Game?", "sheet"), ("Start", "sheet"), ("Score: 12", "")]
    assert len(ocr.pick_marks(r, limit=2)) == 2


def test_accessibilitys_own_elements_and_marks_both_cover_words():
    r = _reading([("Save", 1.0, (100.0, 100.0, 40.0, 14.0))])
    el = ax.Element("e3", "AXButton", "Save", "", 90.0, 95.0, 60.0, 24.0)
    mark = types.SimpleNamespace(frame=(90.0, 95.0, 60.0, 24.0), label="Save")
    assert ocr.pick_marks(r, covered=[el]) == [] and ocr.pick_marks(r, covered=[mark]) == []
    assert ocr.pick_marks(r, covered=[ax.Element("e4", "AXGroup", "", "", 0.0, 0.0, 1000.0, 800.0)])  # a bare group names nothing


def test_marks_get_fresh_ids_and_resolve_to_the_middle_of_their_words(monkeypatch):
    world(monkeypatch, [("Hint", (100.0, 100.0, 40.0, 14.0)), ("Hint", (300.0, 100.0, 40.0, 14.0))])
    a = ocr.remember_marks([("Hint", (100.0, 100.0, 40.0, 14.0), "")], app=APP, pid=PID)
    b = ocr.remember_marks([("Hint", (300.0, 100.0, 40.0, 14.0), "")], app=APP, pid=PID)
    assert a[0].id != b[0].id  # never reused: an old id can't name newer words
    got = ocr.resolve_mark(a[0].id)
    assert got == (120.0, 107.0) and (got.pid, got.app) == (PID, APP) and ocr.mark_labels(b[0].id) == ["Hint"]
    assert ocr.MARK_TTL_S <= 30  # about one look
    monkeypatch.setattr(ocr, "MARK_TTL_S", 0.0)
    with pytest.raises(LookupError, match="no text mark"):
        ocr.resolve_mark(a[0].id)
    assert ocr.mark_labels(a[0].id) == []
    with pytest.raises(LookupError, match="no text mark t999"):
        ocr.resolve_mark("t999")


@pytest.mark.parametrize("words, context", [("Delete", ""), ("Send", ""), ("Buy Now", ""), ("OK", "Delete all files? · Cancel · OK")])
def test_a_mark_whose_words_commit_is_never_clicked_by_id(words, context):
    m = ocr.remember_marks([(words, (100.0, 100.0, 40.0, 14.0), "sheet")], context=context or None)[0]
    with pytest.raises(LookupError) as e:
        ocr.resolve_mark(m.id)
    assert "click_text" in str(e.value) and words not in str(e.value)  # NEO's words only, never the screen's


def test_a_plain_ok_is_clicked_by_id(monkeypatch):
    world(monkeypatch, [("Game over", (100.0, 40.0, 80.0, 14.0)), ("OK", (100.0, 100.0, 40.0, 14.0))])
    m = ocr.remember_marks([("OK", (100.0, 100.0, 40.0, 14.0), "")], context="Game over · OK", app=APP, pid=PID)[0]
    assert ocr.resolve_mark(m.id) == (120.0, 107.0) and ocr.mark_labels(m.id) == ["OK"]


def test_a_blind_brain_gets_ids_and_words_never_numbers():
    marks = ocr.remember_marks([("Start a New Game?", (400.0, 300.0, 150.0, 16.0), "sheet"), ("Hint", (20.0, 20.0, 40.0, 14.0), "")])
    shot = types.SimpleNamespace(x0=0.0, y0=0.0, ppp=1.0, w=1000, h=800)
    blind = ocr.marks_block(marks, shot, blind=True)
    body = blind.split('<untrusted source="screen">')[1]
    assert "box_2d" not in blind and all(not ch.isdigit() for ch in body.replace(marks[0].id, "").replace(marks[1].id, ""))
    seeing = ocr.marks_block(marks, shot, blind=False)
    assert f"{marks[0].id} Text “Start a New Game?” [sheet] box_2d 375,400,395,550" in seeing
    assert seeing.index("click id=t") < seeing.index('<untrusted source="screen">')  # NEO's words outside the fence


def test_install_registers_the_marks_with_the_pointers_id_seam(monkeypatch):
    got = {}
    monkeypatch.setattr(pointer, "set_id_resolver", lambda prefix, fn: got.update({prefix: fn}), raising=False)
    assert ocr.install() and got == {"t": ocr.resolve_mark}

    def with_labels(prefix, fn, *, labels=None):  # (the Element marks' seam: the mark's words for its gate)
        got[prefix] = (fn, labels)

    monkeypatch.setattr(pointer, "set_id_resolver", with_labels, raising=False)
    assert ocr.install() and got["t"] == (ocr.resolve_mark, ocr.mark_labels)
    monkeypatch.delattr(pointer, "set_id_resolver")
    assert not ocr.install()


def test_text_marks_read_accurately_and_leave_out_protected_windows(monkeypatch):
    frame, eng = scene([("Score 12", (20.0, 20.0, 80.0, 14.0)), ("Approve", (250.0, 20.0, 60.0, 14.0)), ("Save", (20.0, 200.0, 40.0, 14.0))])
    use(eng)
    windows(monkeypatch, ocr.Win((240.0, 0.0, 160.0, 100.0), 99, "Vault", True), BOARD_WIN)  # a protected app's, in front
    marks = ocr.text_marks(frame, covered=[((10.0, 190.0, 60.0, 30.0), "Save")], app=APP, pid=PID)
    assert [m.text for m in marks] == ["Score 12"] and [c["fast"] for c in eng.calls] == [False, True]
    assert "Approve" not in marks[0].context and (marks[0].app, marks[0].pid) == (APP, PID)
    monkeypatch.setattr(ocr, "windows_over", lambda area: None)  # the window server can't say: nothing read
    assert ocr.text_marks(frame) == []


def test_a_look_lists_text_marks_for_a_blind_brain_or_a_canvas_only(monkeypatch):
    frame, eng = scene([("Score 12", (20.0, 20.0, 80.0, 14.0)), ("Hint", (300.0, 200.0, 40.0, 14.0))])
    use(eng)
    windows(monkeypatch)
    win = (0.0, 0.0, 400.0, 300.0)
    panes = [((0.0, 0.0, 200.0, 300.0), "Sidebar"), ((200.0, 0.0, 200.0, 300.0), "Editor")] + [((x, 0.0, 10.0, 10.0), f"B{x}") for x in (0.0, 20.0, 40.0)]
    assert asyncio.run(ocr.look_marks(frame, named=panes, window=win)) == [] and eng.calls == []
    got = asyncio.run(ocr.look_marks(frame, named=panes, window=win, blind=True))
    assert [m.text for m in got] == ["Score 12", "Hint"]
    assert [m.text for m in asyncio.run(ocr.look_marks(frame, window=win))] == ["Score 12", "Hint"]  # a canvas


def test_worth_reading_when_accessibility_names_little():
    win = (0.0, 0.0, 1000.0, 800.0)
    buttons = [((x, 10.0, 80.0, 24.0), f"B{x}") for x in (10.0, 100.0, 200.0, 300.0)]
    assert ocr.worth_reading(win, buttons, blind=True)  # a blind brain: the words are all it gets
    assert ocr.worth_reading(win, buttons, blind=False)  # four small buttons over a canvas
    panes = [((0.0, 0.0, 500.0, 800.0), "Sidebar"), ((500.0, 0.0, 500.0, 800.0), "Editor"), *buttons]
    assert not ocr.worth_reading(win, panes, blind=False)


# ---- what changed --------------------------------------------------------------------------------------------


def test_words_changed_says_gone_new_and_changed_in_place():
    before = _reading([
        ("Start a New Game?", 1.0, (400.0, 300.0, 150.0, 16.0)),
        ("Start", 1.0, (500.0, 360.0, 40.0, 14.0)),
        ("Moves: 3", 1.0, (20.0, 20.0, 60.0, 14.0)),
    ])  # fmt: skip
    after = _reading([
        ("Stop", 1.0, (502.0, 360.0, 36.0, 14.0)),
        ("Moves: 3", 1.0, (20.0, 60.0, 60.0, 14.0)),  # moved, not gone
        ("Game over", 1.0, (100.0, 500.0, 80.0, 14.0)),
    ])  # fmt: skip
    said = ocr.words_changed(before, after)
    assert said == "“Start” → “Stop”; “Start a New Game?” is gone; new: “Game over”"
    assert ocr.words_changed(before, before) == ""
    wobble = _reading([("Start a New Game ?", 1.0, (400.0, 300.0, 150.0, 16.0)), ("Start", 1.0, (500.0, 360.0, 40.0, 14.0)), ("Moves: 3", 1.0, (20.0, 20.0, 60.0, 14.0))])
    assert ocr.words_changed(before, wobble) == ""  # a read's small wobble is no change


def test_sample_change_reads_the_full_detail_crops_only_where_the_samples_differ():
    # The full-detail crops around the point, before and after (the same words block, level 10).
    f1, _ = scene([("Start", (40.0, 50.0, 40.0, 14.0))], size_pt=(120.0, 120.0))
    f2, _ = scene([("Stop", (40.0, 50.0, 40.0, 14.0))], size_pt=(120.0, 120.0))
    engine = FakeVision({10: "Start"})

    def sample(img, full):
        s = Image.new("L", (24, 24), img)
        s.info.update({"full": full, "ppp": 0.5})
        return s

    use(engine)
    same = ocr.sample_change([sample(40, f1.image)], [sample(40, f1.image)])
    assert same == "" and engine.calls == []  # the samples don't differ: nothing read

    class Two(FakeVision):  # the second read sees "Stop"
        def recognize(self, image, **kw):
            self.words = {10: "Stop"} if self.calls else {10: "Start"}
            return super().recognize(image, **kw)

    use(Two({10: "Start"}))
    got = ocr.sample_change([sample(40, f1.image)], [sample(160, f2.image)])
    assert got == "“Start” → “Stop”"
    plain = Image.new("L", (24, 24), 40)
    assert ocr.sample_change([plain], [Image.new("L", (24, 24), 160)]) == ""  # no full-detail crops kept


def test_a_word_one_read_missed_is_not_reported_gone():
    before, _ = scene([("Start", (40.0, 50.0, 40.0, 14.0)), ("Score 3", (40.0, 90.0, 60.0, 14.0))], size_pt=(160.0, 160.0))
    after, _ = scene([("Start", (40.0, 50.0, 40.0, 14.0)), ("Score 4", (40.0, 90.0, 60.0, 14.0))], size_pt=(160.0, 160.0))
    eng = FakeVision({10: "Start", 22: "Score 3"})
    reads = []

    def recognize(image, **kw):
        reads.append(image.size)
        eng.words = {10: "Start", 22: "Score 3" if len(reads) == 1 else "Score 4"}
        return FakeVision.recognize(eng, image, **kw)

    eng.recognize = recognize
    eng.drop = lambda text, fast, n: text == "Start" and n == 2  # the after read (unframed) misses "Start"
    use(eng)
    said = ocr.compare_pictures(before, after)
    assert said == "“Score 3” → “Score 4”" and len(reads) == 3  # looked for "Start" again: still there


# ---- never in the suite, never in the background -----------------------------------------------------------


def test_vision_is_never_loaded_with_real_effects_off():
    assert not ocr.available() and ocr.engine() is None  # (the suite: effects off, no engine set)
    with pytest.raises(effects.EffectRefused):
        ocr._Vision()._classes()
    assert ocr.find_text_sync("Start", (0.0, 0.0, 100.0, 100.0)) is None  # can't read: the caller's answer stands
    assert ocr.sample_change([Image.new("L", (4, 4))], [Image.new("L", (4, 4))]) == ""


def test_warm_up_never_runs_in_library_mode_or_with_effects_off(monkeypatch):
    assert not ocr.warm()
    monkeypatch.setattr(effects, "on", lambda: True)
    monkeypatch.setattr(runtime, "library_bound", lambda: True)
    assert not ocr.warm()


# ---- click_text's fallback ---------------------------------------------------------------------------------------


class Desk:
    """The fake Mac for the fallback: what is under each point, who is in front, what was clicked."""

    def __init__(self):
        self.regions = [(0.0, 0.0, 400.0, 300.0, PID, APP, "", "AXGroup", "")]  # a canvas: no labels
        self.clicks: list = []
        self.moves: list = []
        self.brought: list = []
        self.front = (APP, PID)  # the app in front (active): an app brought forward is

    def hit(self, x, y):
        for rx, ry, rw, rh, pid, app, label, role, sheet in self.regions:
            if rx <= x < rx + rw and ry <= y < ry + rh:
                return ax.Hit(pid=pid, app=app, label=label, role=role, regular=True, sheet=sheet)
        return ax.Hit()

    def windows_at(self, x, y):  # the window server: each region is an app's normal window (layer 0)
        return [
            ax.Win(pid, app, 0, (rx, ry, rw, rh))
            for rx, ry, rw, rh, pid, app, *_ in self.regions
            if rx <= x < rx + rw and ry <= y < ry + rh
        ]


@pytest.fixture
def desk(monkeypatch):
    d = Desk()
    monkeypatch.setattr(ax, "hit", d.hit)
    monkeypatch.setattr(ax, "is_neo", lambda pid, name="": pid == NEO)
    monkeypatch.setattr(ax, "app_names", lambda pid: set())
    monkeypatch.setattr(focus, "front_app", lambda: d.front)
    monkeypatch.setattr(ax, "frontmost_app", lambda: (APP, PID))
    monkeypatch.setattr(ax, "windows_at", d.windows_at)

    async def bring(pid):  # (async, like apps.bring_pid_front)
        d.brought.append(pid)
        d.front = (APP, pid)
        return True, APP

    monkeypatch.setattr(apps, "bring_pid_front", bring)
    monkeypatch.setattr(ax, "regular", lambda pid: True)
    monkeypatch.setattr(inp, "click", lambda x, y, **k: d.clicks.append((round(x, 1), round(y, 1))))
    monkeypatch.setattr(inp, "move", lambda x, y: d.moves.append((x, y)))
    windows(monkeypatch)
    monkeypatch.setattr(pointer, "HOVER_S", 0)
    monkeypatch.setattr(pointer, "LOOK_S", 0)
    monkeypatch.setattr(pointer, "LOOK_AGAIN_S", 0)
    return d


def _show(monkeypatch, items):
    frame, eng = scene(items)
    use(eng)
    monkeypatch.setattr(ocr, "grab", lambda area: frame)
    return eng


def _fallback(label, ctx=None, els=None, window=(0.0, 0.0, 400.0, 300.0)):
    ctx = ctx or ToolContext(user_text=f"click {label}", origin="user")
    return asyncio.run(ocr.click_text_fallback({"label": label}, ctx, app=APP, window=window, els=els)), ctx


def test_words_accessibility_doesnt_name_are_clicked_where_the_screen_shows_them(desk, monkeypatch):
    _show(monkeypatch, [("Hint", (300.0, 40.0, 40.0, 16.0)), ("Score 12", (20.0, 20.0, 80.0, 14.0))])
    out, ctx = _fallback("hint")
    assert isinstance(out, ToolOutput) and out.ok and len(desk.clicks) == 1 and near(desk.clicks[0], (320.0, 48.0), 1.5)
    assert "found by reading the screen" in out.text and "screen" in ctx.taint
    head, fenced = out.text.split('<untrusted source="screen">')
    assert "“Hint”" in fenced and "“Hint”" not in head  # the screen's words are fenced


def test_the_gate_asks_before_words_that_commit(desk, monkeypatch):
    _show(monkeypatch, [("Delete All", (300.0, 40.0, 70.0, 16.0))])
    out, _ = _fallback("delete all")
    assert str(out).startswith(confirm.NEEDS_CONFIRM) and "Delete All" in str(out) and desk.clicks == []
    confirm.cancel("t")


def test_the_gate_judges_the_whole_line_when_the_words_are_part_of_it(desk, monkeypatch):
    _show(monkeypatch, [("All Delete", (300.0, 40.0, 70.0, 16.0))])  # "All" asked; the line deletes
    out, _ = _fallback("all")
    assert str(out).startswith(confirm.NEEDS_CONFIRM) and desk.clicks == []
    confirm.cancel("t")


def test_a_drawn_dialogs_ok_is_judged_by_the_dialogs_words(desk, monkeypatch):
    _show(monkeypatch, [("Erase the whole drawing?", (100.0, 100.0, 180.0, 16.0)), ("OK", (250.0, 200.0, 30.0, 16.0))])
    out, _ = _fallback("OK")
    assert str(out).startswith(confirm.NEEDS_CONFIRM) and desk.clicks == []
    confirm.cancel("t")


def test_whats_under_the_point_is_gated_too(desk, monkeypatch):
    _show(monkeypatch, [("Hint", (300.0, 40.0, 40.0, 16.0))])
    desk.regions.insert(0, (290.0, 30.0, 60.0, 40.0, PID, APP, "Send", "AXButton", ""))
    out, _ = _fallback("hint")
    assert str(out).startswith(confirm.NEEDS_CONFIRM) and "Send" in str(out) and desk.clicks == []
    confirm.cancel("t")


def test_neos_own_window_and_protected_apps_are_refused(desk, monkeypatch):
    _show(monkeypatch, [("Hint", (300.0, 40.0, 40.0, 16.0))])
    desk.regions.insert(0, (250.0, 0.0, 150.0, 150.0, NEO, "NEO", "Stop", "AXButton", ""))
    out, _ = _fallback("hint")
    assert not out.ok and "NEO's own window" in out.text and desk.clicks == [] and desk.brought == []


def test_another_apps_window_over_the_words_is_refused(desk, monkeypatch):
    _show(monkeypatch, [("Hint", (300.0, 40.0, 40.0, 16.0))])
    desk.regions.insert(0, (250.0, 0.0, 150.0, 150.0, OTHER, "Mail", "Inbox", "AXRow", ""))
    out, _ = _fallback("hint")
    assert not out.ok and "another window" in out.text and desk.clicks == [] and desk.brought == []


def test_a_window_a_sheet_blocks_is_refused_and_the_sheet_named(desk, monkeypatch):
    _show(monkeypatch, [("Hint", (300.0, 40.0, 40.0, 16.0))])
    desk.regions[0] = (0.0, 0.0, 400.0, 300.0, PID, APP, "", "AXGroup", "“Save changes?” (buttons: Save, Cancel)")
    out, ctx = _fallback("hint")
    assert not out.ok and "open sheet or dialog" in out.text and desk.clicks == [] and "screen" in ctx.taint


def test_words_in_several_places_are_listed_by_id_not_guessed(desk, monkeypatch):
    _show(monkeypatch, [("Hint", (300.0, 40.0, 40.0, 16.0)), ("Hint", (30.0, 240.0, 40.0, 16.0))])
    out, ctx = _fallback("hint")
    assert not out.ok and "which one" in out.text and desk.clicks == [] and "screen" in ctx.taint
    ids = [m.id for m in ocr._marks.values()]
    assert len(ids) == 2 and all(i in out.text for i in ids) and "box_2d" not in out.text
    assert near(ocr.resolve_mark(ids[0]), (320.0, 48.0), 1.5)  # the first in reading order


def test_not_found_says_both_ways_were_checked_and_devanagari_says_why(desk, monkeypatch):
    _show(monkeypatch, [("Hint", (300.0, 40.0, 40.0, 16.0))])
    out, _ = _fallback("resign")
    assert not out.ok and "nothing in the front window of Board is called 'resign'" in out.text and desk.clicks == []
    out, _ = _fallback("नया खेल")
    assert not out.ok and "can't read Devanagari" in out.text


def test_without_text_recognition_the_handlers_answer_stands(desk, monkeypatch):
    monkeypatch.setattr(ocr, "grab", lambda area: None)  # no Screen Recording
    use(FakeVision({}))
    assert _fallback("hint")[0] is None
    ocr.set_engine(None)
    assert _fallback("hint")[0] is None  # no engine at all
    use(FakeVision({}))
    monkeypatch.setattr(ocr, "windows_over", lambda area: None)  # the window server can't say
    assert _fallback("hint")[0] is None
    assert _fallback("hint", window=None)[0] is None


def test_a_models_click_looks_before_and_after(desk, monkeypatch):
    _show(monkeypatch, [("Hint", (300.0, 40.0, 40.0, 16.0))])
    looks = []
    monkeypatch.setattr(pointer, "look", lambda pid, pts: looks.append(pid) or asyncio.sleep(0, pointer.Look(ax.Scene(), None)))
    out, _ = _fallback("hint", ctx=ToolContext(user_text="play", origin="agent"))
    assert len(desk.clicks) == 1 and desk.moves and looks and looks[0] == PID  # the pointer rested there first
    assert "couldn't" in out.text  # (no picture, no Accessibility answer: it says so, never "ok")


def test_a_blind_brain_is_told_what_was_clicked_never_where(desk, monkeypatch):
    _show(monkeypatch, [("Hint", (300.0, 40.0, 40.0, 16.0))])
    monkeypatch.setattr(pointer, "look", lambda pid, pts: asyncio.sleep(0, pointer.Look(ax.Scene(), None)))
    ctx = ToolContext(user_text="play", origin="agent", extras={"blind": True})
    out, _ = _fallback("hint", ctx=ctx)
    assert len(desk.clicks) == 1 and "Clicked the words 'hint' in Board" in out.text
    assert not any(ch.isdigit() for ch in out.text)  # no coordinates in any form


def test_the_gate_helpers_are_the_packages_own():
    # (looked up at call time: Step 0's frozen gates module, re-exported by the package, reaches this path)
    assert ocr._gates() is computer


# ---- whose words: the window server's windows, front to back ---------------------------------------------


def _cg(monkeypatch, listed):
    import Quartz as Q

    monkeypatch.setattr(Q, "CGWindowListCopyWindowInfo", lambda *a: listed)


def cgwin(x, y, w, h, pid, name, alpha=1.0):
    return {"kCGWindowBounds": {"X": x, "Y": y, "Width": w, "Height": h}, "kCGWindowOwnerPID": pid, "kCGWindowOwnerName": name, "kCGWindowAlpha": alpha}


def _protect(monkeypatch, *names):
    """settings.protected_apps, as the user sets it (the package's own _protected judges)."""
    from neo2.core import config

    monkeypatch.setenv("NEO_PROTECTED_APPS", "[" + ", ".join(f'"{n.lower()}"' for n in names) + "]")
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(ax, "app_names", lambda pid: {n.lower() for n in names} if pid == 99 else set())


def test_the_words_are_the_topmost_windows_there(monkeypatch):
    # Front to back: a banner over Board's top right, NEO's own window, Board, a protected app's window
    # behind Board, an invisible window, and one elsewhere.
    monkeypatch.setattr(ax, "is_neo", lambda pid, name="": pid == NEO)
    _protect(monkeypatch, "Vault")
    _cg(monkeypatch, [
        cgwin(250, 0, 150, 60, OTHER, "Banner"),
        cgwin(0, 250, 60, 50, NEO, "NEO"),
        cgwin(0, 0, 400, 300, PID, APP),
        cgwin(0, 0, 800, 600, 99, "Vault"),
        cgwin(0, 0, 400, 300, 77, "Ghost", alpha=0.0),
        cgwin(900, 900, 50, 50, 78, "Far"),
    ])  # fmt: skip
    wins = ocr.windows_over((0.0, 0.0, 400.0, 300.0))
    assert [(w.app, w.off_limits) for w in wins.wins] == [("Banner", False), ("NEO", True), (APP, False), ("Vault", True)]
    r = _reading([
        ("Score 12", 1.0, (20.0, 20.0, 80.0, 14.0)),
        ("New mail", 1.0, (280.0, 20.0, 80.0, 14.0)),  # the banner's
        ("Hint", 1.0, (300.0, 200.0, 40.0, 14.0)),
        ("Orb", 1.0, (10.0, 260.0, 30.0, 14.0)),  # NEO's
        ("Secret", 1.0, (500.0, 500.0, 60.0, 14.0)),  # the vault's, beside Board
    ])  # fmt: skip
    # Board's words only: the protected window behind it hides none of them, the banner's aren't Board's.
    assert [ln.text for ln in ocr.owned(r, wins, PID).lines] == ["Score 12", "Hint"]
    assert [ln.text for ln in ocr.owned(r, wins).lines] == ["Score 12", "New mail", "Hint"]  # (any app's but NEO's or a protected one's)


def test_text_marks_are_tagged_with_the_app_whose_window_shows_them(monkeypatch):
    frame, eng = scene([("Score 12", (20.0, 20.0, 80.0, 14.0)), ("New mail", (280.0, 20.0, 80.0, 14.0)), ("Hint", (300.0, 200.0, 40.0, 14.0))])
    use(eng)
    windows(monkeypatch, ocr.Win((250.0, 0.0, 150.0, 60.0), OTHER, "Banner"), BOARD_WIN)
    got = ocr.text_marks(frame)  # (a look at the whole screen: no one app)
    assert [(m.text, m.app, m.pid) for m in got] == [("Score 12", APP, PID), ("New mail", "Banner", OTHER), ("Hint", APP, PID)]
    assert [m.text for m in ocr.text_marks(frame, app=APP, pid=PID)] == ["Score 12", "Hint"]  # a banner's words are never Board's


def test_without_the_window_servers_answer_nothing_is_read(monkeypatch):
    _cg(monkeypatch, None)
    assert ocr.windows_over((0.0, 0.0, 400.0, 300.0)) is None

    def broken(*a):
        raise RuntimeError("no window server")

    _cg(monkeypatch, [])
    import Quartz as Q

    monkeypatch.setattr(Q, "CGWindowListCopyWindowInfo", broken)
    assert ocr.windows_over((0.0, 0.0, 400.0, 300.0)) is None


def test_a_protected_window_behind_the_front_one_hides_none_of_its_words(desk, monkeypatch):
    _show(monkeypatch, [("Hint", (300.0, 40.0, 40.0, 16.0))])
    windows(monkeypatch, BOARD_WIN, ocr.Win((0.0, 0.0, 800.0, 600.0), 99, "Vault", True))
    out, _ = _fallback("hint")
    assert out.ok and len(desk.clicks) == 1


def test_words_another_window_covers_are_never_the_targets(desk, monkeypatch):
    # The window server: another app's window over Board's words — they aren't read as Board's at all.
    _show(monkeypatch, [("Hint", (300.0, 40.0, 40.0, 16.0))])
    windows(monkeypatch, ocr.Win((250.0, 0.0, 150.0, 150.0), OTHER, "Mail"), BOARD_WIN)
    desk.regions.insert(0, (250.0, 0.0, 150.0, 150.0, OTHER, "Mail", "Inbox", "AXRow", ""))
    out, _ = _fallback("hint")
    assert not out.ok and "is called 'hint'" in out.text and desk.clicks == [] and desk.brought == []


# ---- the app acted on: the one whose window was read, and only it --------------------------------------------


def test_a_banner_in_front_doesnt_lose_the_app_whose_window_was_read(desk, monkeypatch):
    # LaunchServices names the banner's app; ax.frontmost_app — what the handler's window_elements went
    # by — names the app the user is in. Its pid is the one used, and only it is brought forward.
    _show(monkeypatch, [("Hint", (300.0, 40.0, 40.0, 16.0))])
    desk.front = ("NotificationCenter", 77)
    out, _ = _fallback("hint")
    assert out.ok and desk.brought == [PID] and len(desk.clicks) == 1 and f"brought {APP} to the front first" in out.text


def test_an_app_that_isnt_the_one_read_is_never_guessed(desk, monkeypatch):
    _show(monkeypatch, [("Hint", (300.0, 40.0, 40.0, 16.0))])
    monkeypatch.setattr(ax, "frontmost_app", lambda: ("Board Helper", PID))  # (another name, or it changed since)
    assert _fallback("hint")[0] is None and desk.clicks == [] and desk.brought == []
    monkeypatch.setattr(ax, "frontmost_app", lambda: (APP, 0))
    assert _fallback("hint")[0] is None
    # The handler's own pid is taken as it is.
    ctx = ToolContext(user_text="click hint", origin="user")
    out = asyncio.run(ocr.click_text_fallback({"label": "hint"}, ctx, app=APP, window=(0.0, 0.0, 400.0, 300.0), pid=PID))
    assert out.ok and len(desk.clicks) == 1


def test_an_app_not_in_front_is_brought_forward_first(desk, monkeypatch):
    _show(monkeypatch, [("Hint", (300.0, 40.0, 40.0, 16.0))])
    desk.front = ("Finder", 11)
    out, _ = _fallback("hint")
    assert out.ok and desk.brought == [PID] and len(desk.clicks) == 1 and f"(brought {APP} to the front first)" in out.text


def test_whats_there_once_the_app_is_in_front_is_gated_again(desk, monkeypatch):
    _show(monkeypatch, [("Hint", (300.0, 40.0, 40.0, 16.0))])
    desk.front = ("Finder", 11)
    first = desk.hit

    def hit(x, y):  # once Board is in front, its "Send" button is what's under the point
        return ax.Hit(pid=PID, app=APP, label="Send", role="AXButton", regular=True) if desk.brought else first(x, y)

    monkeypatch.setattr(ax, "hit", hit)
    out, _ = _fallback("hint")
    assert str(out).startswith(confirm.NEEDS_CONFIRM) and "Send" in str(out) and desk.brought == [PID] and desk.clicks == []
    confirm.cancel("t")


def test_a_protected_apps_window_over_the_words_is_refused_and_never_brought_forward(desk, monkeypatch):
    _show(monkeypatch, [("Hint", (300.0, 40.0, 40.0, 16.0))])
    _protect(monkeypatch, "MyApprover")
    desk.regions.insert(0, (250.0, 0.0, 150.0, 150.0, 99, "MyApprover", "Approve", "AXButton", ""))
    desk.front = ("Finder", 11)
    out, _ = _fallback("hint")
    assert not out.ok and desk.clicks == [] and desk.brought == []


# ---- a brain that can't see: never a point, whatever refuses ---------------------------------------------------


@pytest.mark.parametrize("case", ["spot changed", "covered", "stays behind", "another window", "sheet", "protected"])
def test_a_blind_brain_is_never_told_a_point_when_a_click_is_refused(desk, monkeypatch, case):
    _show(monkeypatch, [("Hint", (300.0, 40.0, 40.0, 16.0))])
    desk.front = ("Finder", 11)  # (Board isn't in front: it is brought)
    if case == "spot changed":  # (the pointer names the spot it looked at again)

        async def bring(t, pts, **kw):
            raise pointer.Refused(
                "Error: bringing Board forward changed what's at that spot (320, 48) — it's now "
                + '<untrusted source="screen">Board: Button “Other”</untrusted>',
                screen=True,
            )

        monkeypatch.setattr(pointer, "bring", bring)
    elif case == "covered":
        first = desk.hit
        monkeypatch.setattr(ax, "hit", lambda x, y: ax.Hit(pid=OTHER, app="Mail", label="Inbox", regular=True) if desk.brought else first(x, y))
    elif case == "stays behind":

        async def stays(pid):
            desk.brought.append(pid)
            return False, APP

        monkeypatch.setattr(apps, "bring_pid_front", stays)
    elif case == "another window":
        desk.regions.insert(0, (250.0, 0.0, 150.0, 150.0, OTHER, "Mail", "Inbox", "AXRow", ""))
    elif case == "sheet":
        desk.regions[0] = (0.0, 0.0, 400.0, 300.0, PID, APP, "", "AXGroup", "“Save changes?” (buttons: Save, Cancel)")
    else:
        desk.regions.insert(0, (250.0, 0.0, 150.0, 150.0, NEO, "NEO", "Stop", "AXButton", ""))
    ctx = ToolContext(user_text="play", origin="agent", extras={"blind": True})
    out, _ = _fallback("hint", ctx=ctx)
    assert not out.ok and desk.clicks == []
    assert not re.search(r"\(\s*\d+\s*,\s*\d+\s*\)", out.text)  # no point in any form
    assert not any(ch.isdigit() for ch in out.text.split("<untrusted")[0])  # (NEO's own words: no numbers at all)


# ---- one read's misread never decides, and every version is judged ---------------------------------------------


class Scripted(FakeVision):
    """Reads as FakeVision does, except read number n (from 0) says other words: ``misread[n]``."""

    def __init__(self, words, misread, **kw):
        super().__init__(words, **kw)
        self.misread = misread

    def recognize(self, image, **kw):
        swap = self.misread.get(len(self.calls), {})
        return [ocr.Raw(swap.get(r.text, r.text), r.conf, r.box, r.parts) for r in super().recognize(image, **kw)]


def test_one_reads_misread_never_outvotes_two_reads(desk, monkeypatch):
    # The first fast read says "Details" where the button says "Delete"; the padded fast read and the
    # accurate read both say "Delete": asked for "Details", there is none — and nothing is clicked.
    frame, eng = scene([("Delete", (300.0, 40.0, 50.0, 16.0))])
    eng = use(Scripted(eng.words, {0: {"Delete": "Details"}}))
    got = ocr.find_in(frame, "Details")
    assert isinstance(got, ocr.Missing) and "disagree" in got.why and [c["fast"] for c in eng.calls] == [True, True, False]
    monkeypatch.setattr(ocr, "grab", lambda area: frame)
    eng.calls.clear()
    out, _ = _fallback("details")
    assert not out.ok and desk.clicks == [] and not str(out).startswith(confirm.NEEDS_CONFIRM)


def test_the_gate_judges_every_version_of_the_words_a_read_saw(desk, monkeypatch):
    # One read each way ("Detail" elsewhere makes the first fast read unsure, so the accurate one
    # decides): the find stands — and the gate asks, as it would before "Delete".
    frame, eng = scene([("Delete", (300.0, 40.0, 50.0, 16.0)), ("Detail", (40.0, 200.0, 50.0, 16.0))])
    eng = use(Scripted(eng.words, {0: {"Delete": "Details"}}))
    got = ocr.find_in(frame, "Details")
    assert isinstance(got, ocr.Found) and got.line == "Details" and got.variants == ("Delete",)
    monkeypatch.setattr(ocr, "grab", lambda area: frame)
    eng.calls.clear()
    out, _ = _fallback("details")
    assert str(out).startswith(confirm.NEEDS_CONFIRM) and "Delete" in str(out) and desk.clicks == []
    confirm.cancel("t")


def test_a_mark_two_reads_disagree_on_is_judged_as_both(monkeypatch):
    # The accurate read says "Details", the fast one "Delete": the mark shows the first, and is judged as
    # both — never clicked by id, and the pointer's gate gets both.
    frame, eng = world(monkeypatch, [("Delete", (300.0, 40.0, 50.0, 16.0))])
    use(Scripted(eng.words, {0: {"Delete": "Details"}}))
    marks = ocr.text_marks(frame, app=APP, pid=PID)
    assert [(m.text, m.also) for m in marks] == [("Details", ("Delete",))]
    with pytest.raises(LookupError, match="commit"):
        ocr.resolve_mark(marks[0].id)
    assert ocr.mark_labels(marks[0].id) == ["Details", "Delete"]


def test_an_ok_among_several_in_a_drawn_dialog_is_never_clicked_by_id(desk, monkeypatch):
    _show(monkeypatch, [("Delete all files?", (100.0, 100.0, 180.0, 16.0)), ("OK", (150.0, 200.0, 30.0, 16.0)), ("OK", (250.0, 200.0, 30.0, 16.0))])
    out, _ = _fallback("OK")
    assert not out.ok and "which one" in out.text and desk.clicks == []
    ids = list(ocr._marks)
    assert len(ids) == 2 and all("Delete all files?" in ocr._marks[i].context for i in ids)  # the dialog's words go with each id
    for i in ids:
        with pytest.raises(LookupError, match="commit"):
            ocr.resolve_mark(i)
    # And the pointer's gate, handed the mark's words, asks — though Accessibility sees no dialog there.
    monkeypatch.setattr(ax, "window_elements", lambda app=None: (APP, (0.0, 0.0, 400.0, 300.0), []))
    labels = ocr.mark_labels(ids[0])
    assert labels[0] == "OK" and len(labels) == 2
    gate = asyncio.run(computer._gate_click({"id": ids[0]}, [(w, APP) for w in labels]))
    assert str(gate).startswith(confirm.NEEDS_CONFIRM) and "Delete all files?" in str(gate)
    confirm.cancel("t")


# ---- a mark is clicked only while its words are still there ---------------------------------------------------


def test_a_mark_is_clicked_only_while_its_words_are_still_there(monkeypatch):
    hint = [("Hint", (300.0, 40.0, 40.0, 16.0))]
    frame, eng = world(monkeypatch, hint)
    m = ocr.text_marks(frame, app=APP, pid=PID)[0]
    eng.calls.clear()
    assert near(ocr.resolve_mark(m.id), (320.0, 48.0), 0.01)
    # Checked with one fast read of the words' neighbourhood (about 144×96 px at Retina detail).
    assert [c["fast"] for c in eng.calls] == [True] and near(eng.calls[0]["size"], (144, 96), 1)

    def refused(match, *wins, items=hint, **kw):
        world(monkeypatch, items, *wins, **kw)
        with pytest.raises(LookupError, match=match) as e:
            ocr.resolve_mark(m.id)
        assert not any(ch.isdigit() for ch in str(e.value).replace(m.id, ""))  # NEO's words: no point, none of the screen's

    refused("aren't where they were", items=[("Hint", (300.0, 140.0, 40.0, 16.0))])  # the window moved or scrolled
    refused("aren't where they were", items=[("Resign", (300.0, 40.0, 50.0, 16.0))])  # other words there now
    refused("commit", items=[("Hint: Delete save", (300.0, 40.0, 140.0, 16.0))])  # the words there now delete
    refused("another window", ocr.Win((250.0, 0.0, 150.0, 150.0), OTHER, "Mail"), BOARD_WIN)  # another app's over it
    refused("another window", ocr.Win((250.0, 0.0, 150.0, 150.0), 99, "Vault", True), BOARD_WIN)  # a protected one's
    world(monkeypatch, hint)
    monkeypatch.setattr(ocr, "windows_over", lambda area: None)
    with pytest.raises(LookupError, match="can't check"):
        ocr.resolve_mark(m.id)
    world(monkeypatch, hint)
    monkeypatch.setattr(ocr, "grab", lambda area: None)  # (no Screen Recording)
    with pytest.raises(LookupError, match="can't check"):
        ocr.resolve_mark(m.id)
    world(monkeypatch, hint)
    ocr.set_engine(None)
    with pytest.raises(LookupError, match="can't check"):
        ocr.resolve_mark(m.id)


def test_a_read_that_drops_the_words_once_is_read_again_before_a_mark_is_called_gone(monkeypatch):
    frame, eng = world(monkeypatch, [("Hint", (300.0, 40.0, 40.0, 16.0))])
    m = ocr.remember_marks([("Hint", (300.0, 40.0, 40.0, 16.0), "")], app=APP, pid=PID)[0]
    eng.drop = lambda text, fast, n: n == 1  # the first read of the check misses them
    assert ocr.resolve_mark(m.id) == (320.0, 48.0) and len(eng.calls) == 2


# ---- what Accessibility names --------------------------------------------------------------------------------


def test_a_containers_title_names_none_of_the_words_drawn_in_it():
    r = _reading([("Start", 1.0, (100.0, 100.0, 40.0, 14.0)), ("me", 1.0, (300.0, 100.0, 20.0, 14.0))])
    web = ax.Element("e9", "AXWebArea", "Start a New Game - Home", "", 0.0, 0.0, 1000.0, 800.0)
    assert [t for t, _, _ in ocr.pick_marks(r, covered=[web])] == ["Start", "me"]
    tabs = [ax.Element(f"e{i}", "AXButton", f"Tab {i}", "", 10.0 * i, 0.0, 10.0, 10.0) for i in range(4)]
    assert ocr.worth_reading((0.0, 0.0, 1000.0, 800.0), [web, *tabs], blind=False)  # a canvas game in a web area: read
    # Whole words only: a "Home" button doesn't name "me".
    home = ax.Element("e1", "AXButton", "Home", "", 290.0, 95.0, 40.0, 24.0)
    assert [t for t, _, _ in ocr.pick_marks(r, covered=[home])] == ["Start", "me"]
    # An element whose role isn't known names words only when it is about their size.
    assert [t for t, _, _ in ocr.pick_marks(r, covered=[((0.0, 0.0, 1000.0, 800.0), "Start a New Game")])] == ["Start", "me"]
    assert [t for t, _, _ in ocr.pick_marks(r, covered=[((90.0, 95.0, 60.0, 24.0), "Start a New Game")])] == ["me"]


# ---- the known limit of a fast find ---------------------------------------------


def test_a_copy_both_fast_framings_drop_is_a_fast_finds_known_limit():
    # (The module's notes say so: fast mode drops, rather than garbles, words too small for it in both
    # framings, and the find is taken from the copy both read. An accurate read would list two.)
    frame, eng = scene([("Start", (40.0, 40.0, 40.0, 16.0)), ("Start", (300.0, 200.0, 40.0, 7.0))])
    eng.fast_drop_px, eng.fast_min_px = 15, 4
    use(eng)
    got = ocr.find_in(frame, "Start")
    assert isinstance(got, ocr.Found) and [c["fast"] for c in eng.calls] == [True, True]


# ---- the capture ------------------------------------------------------------------------------------------------


def test_grab_takes_the_displays_own_detail_and_never_asks_for_screen_recording(monkeypatch):
    from neo2.core.tools.computer import screen

    monkeypatch.setattr(screen, "can_capture", lambda: False)
    assert ocr.grab((0.0, 0.0, 100.0, 100.0)) is None  # no Screen Recording: nothing (and nothing asked)
    monkeypatch.setattr(screen, "can_capture", lambda: True)
    shots = []

    def screencapture(args):
        shots.append(args)
        _, _, w, h = (int(v) for v in args[1].split(","))
        return Image.new("RGB", (w * 2, h * 2))

    monkeypatch.setattr(screen, "_grab", screencapture)
    px = ocr.grab((1400.0, 900.0, 200.0, 100.0))  # it runs off the 1512×982 display: clipped to it
    assert shots == [["-R", "1400,900,112,82"]] and (px.x0, px.y0) == (1400.0, 900.0) and px.ppp == pytest.approx(0.5)
    assert ocr.grab((2000.0, 2000.0, 50.0, 50.0)) is None  # off every display
    # The capture pipeline's fresh frame of the whole screen is taken as it is; a window's own isn't.
    fresh = types.SimpleNamespace(image=Image.new("L", (3024, 1964)), x0=0.0, y0=0.0, ppp=0.5, t=time.monotonic(), window_id=0, covers=lambda *r: True)
    monkeypatch.setattr(screen, "last_frame", lambda age: fresh, raising=False)
    assert ocr.grab((10.0, 10.0, 100.0, 100.0)).image is fresh.image and len(shots) == 1
    fresh.window_id = 7
    assert ocr.grab((10.0, 10.0, 100.0, 100.0)).image is not fresh.image and len(shots) == 2
