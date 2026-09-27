"""Apple Vision itself on synthetic Retina screens: the real engine behind neo2.core.tools.computer.ocr — the
picture handed over as raw grey pixels, boxes flipped to screen points, word boxes by UTF-16 range —
and the numbers behind its speed and accuracy claims (run with -s to see them).

Opt-in (NEO_TEST_OCR=1): it loads Vision's text recognition into this process (allowed for this
module only, as the "model" effect). Synthetic pictures only: nothing is captured, clicked or sent."""

from __future__ import annotations

import io
import os
import random
import statistics
import time

import pytest
from PIL import Image, ImageDraw, ImageFont

from neo2.core import effects
from neo2.core.tools.computer import ocr

pytestmark = pytest.mark.skipif(os.environ.get("NEO_TEST_OCR") != "1", reason="loads Apple Vision: NEO_TEST_OCR=1")

W_PT, H_PT = 1512, 982
SF = "/System/Library/Fonts/SFNS.ttf"
DEVA = "/System/Library/Fonts/Kohinoor.ttc"
BUTTONS = ["New Game", "Start", "Undo Move", "Hint", "Resign", "Settings", "Share…", "नया खेल"]


def _font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def synthetic(scale: int = 2, sheet: bool = True):
    """(picture, truth): a macOS-like screen drawn at ``scale`` pixels per point — a menu bar, a window
    with a grid, a column of buttons (one Hindi), a list, and optionally a sheet asking a question.
    truth: each button's frame and the sheet's, in points."""
    img = Image.new("RGB", (W_PT * scale, H_PT * scale), (236, 236, 238))
    d = ImageDraw.Draw(img)
    f, fb, fh = _font(SF, 13 * scale), _font(SF, 15 * scale), _font(DEVA, 14 * scale)
    d.rectangle([0, 0, W_PT * scale, 24 * scale], fill=(246, 246, 246))
    x = 40
    for word in ["Game", "Edit", "View", "Window", "Help"]:
        d.text((x * scale, 4 * scale), word, fill=(20, 20, 20), font=fb)
        x += len(word) * 9 + 22
    d.rounded_rectangle([360 * scale, 90 * scale, 1260 * scale, 890 * scale], 12 * scale, fill=(250, 250, 250), outline=(200, 200, 200))
    d.text((740 * scale, 98 * scale), "Untitled — Game 1", fill=(60, 60, 60), font=f)
    for r in range(8):
        for c in range(8):
            x0, y0 = (420 + c * 76) * scale, (150 + r * 76) * scale
            d.rectangle([x0, y0, x0 + 76 * scale - 1, y0 + 76 * scale - 1], fill=(238, 238, 210) if (r + c) % 2 == 0 else (118, 150, 86))
    truth: dict = {"buttons": {}, "sheet": None}
    y = 160
    for lab in BUTTONS:
        bx, bw, bh = 1100, 130, 28
        d.rounded_rectangle([bx * scale, y * scale, (bx + bw) * scale, (y + bh) * scale], 6 * scale, fill=(255, 255, 255), outline=(180, 180, 180))
        d.text(((bx + 12) * scale, (y + 5) * scale), lab, fill=(0, 0, 0), font=f if lab.isascii() or "…" in lab else fh)
        truth["buttons"][lab] = (bx, y, bw, bh)
        y += 44
    for i in range(18):
        d.text((20 * scale, (60 + i * 26) * scale), f"Item {i + 1}: saved game", fill=(40, 40, 40), font=f)
    if sheet:
        sx, sy, sw, sh = 560, 90, 500, 170
        d.rectangle([sx * scale, sy * scale, (sx + sw) * scale, (sy + sh) * scale], fill=(245, 245, 245), outline=(150, 150, 150))
        d.text(((sx + 20) * scale, (sy + 20) * scale), "Start a New Game?", fill=(0, 0, 0), font=fb)
        for lab, bx in (("Cancel", sx + 250), ("Start", sx + 370)):
            d.rounded_rectangle([bx * scale, (sy + 120) * scale, (bx + 100) * scale, (sy + 148) * scale], 6 * scale, fill=(255, 255, 255), outline=(160, 160, 160))
            d.text(((bx + 25) * scale, (sy + 125) * scale), lab, fill=(0, 0, 0), font=f)
            truth["buttons"]["sheet " + lab] = (bx, sy + 120, 100, 28)
        truth["sheet"] = (sx, sy, sw, sh)
    return img, truth


@pytest.fixture(scope="module")
def vision():
    with effects.allow("model"):
        ocr.set_engine(None)
        assert ocr.available(), "Apple Vision isn't here"
        e = ocr.engine()
        t = time.perf_counter()
        e.languages(False)
        load_ms = (time.perf_counter() - t) * 1000
        print(f"\n[ocr] Vision loaded in {load_ms:.0f} ms; accurate reads {len(e.languages(False))} languages, fast {len(e.languages(True))}")
        yield e


def _ms(fn, n=5):
    ts = []
    out = None
    for _ in range(n):
        t = time.perf_counter()
        out = fn()
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts), out


def _inside(p, r):
    return r[0] <= p[0] <= r[0] + r[2] and r[1] <= p[1] <= r[1] + r[3]


def test_handing_the_picture_over_raw_instead_of_as_a_png(vision):
    import Quartz as Q

    retina, _ = synthetic()

    def png():
        buf = io.BytesIO()
        retina.save(buf, format="PNG", compress_level=1)
        b = buf.getvalue()
        src = Q.CGImageSourceCreateWithData(Q.CFDataCreate(None, b, len(b)), None)
        return Q.CGImageSourceCreateImageAtIndex(src, 0, None)

    before, _ = _ms(png, 3)
    after, _ = _ms(lambda: ocr._cgimage(retina), 5)
    print(f"[ocr] picture → Vision, 3024×1964: PNG round trip {before:.1f} ms → raw grey {after:.1f} ms")
    assert after < before


def test_fast_reads_need_retina_detail_and_get_it(vision):
    retina, truth = synthetic()
    small = retina.resize((W_PT, H_PT), Image.LANCZOS)  # what a 1-pixel-per-point screenshot holds
    wanted = [b for b in BUTTONS if b.isascii() or "…" in b] + ["Start a New Game?", "Cancel"]
    plan = ocr.Plan(True, ("en-US",))
    with effects.allow("model"):
        raw_ms, raws = _ms(lambda: vision.recognize(small.convert("L"), fast=True, languages=["en-US"], auto=False, ranges=None), 3)
        read_ms, reading = _ms(lambda: ocr.read(ocr.Pixels(small, 0.0, 0.0, 1.0), plan=plan), 3)
        native_ms, native = _ms(lambda: ocr.read(ocr.Pixels(retina, 0.0, 0.0, 0.5), plan=plan), 3)
        acc_ms, acc = _ms(lambda: ocr.read(ocr.Pixels(retina, 0.0, 0.0, 0.5), plan=ocr.Plan(False, ("en-US",))), 3)

    def hits(texts):
        return sum(any(ocr.key(w) in ocr.key(t) for t in texts) for w in wanted)

    before = hits([r.text for r in raws])
    after = hits([ln.text for ln in reading.lines])
    print(
        f"[ocr] fast read, whole display: 1 px/pt as is {before}/{len(wanted)} labels ({raw_ms:.0f} ms) → scaled up "
        f"{after}/{len(wanted)} ({read_ms:.0f} ms); Retina native fast {hits([ln.text for ln in native.lines])}/{len(wanted)} "
        f"({native_ms:.0f} ms), accurate {hits([ln.text for ln in acc.lines])}/{len(wanted)} ({acc_ms:.0f} ms)"
    )
    assert after >= len(wanted) - 1 and after > before
    assert hits([ln.text for ln in native.lines]) >= len(wanted) - 1


def test_labels_on_a_canvas_are_found_inside_their_buttons(vision):
    retina, truth = synthetic()
    frame = ocr.Pixels(retina, 0.0, 0.0, 0.5)
    window = (360.0, 90.0, 900.0, 800.0)
    rows = []
    with effects.allow("model"):
        for lab, box in truth["buttons"].items():
            if not (lab.isascii() or "…" in lab):
                continue
            asked = lab.removeprefix("sheet ")
            area = (1090.0, 150.0, 170.0, 360.0) if not lab.startswith("sheet") else window
            ms, got = _ms(lambda asked=asked, area=area, lab=lab: ocr.find_in(frame, asked, area=area, sheet=truth["sheet"] if lab.startswith("sheet") else None), 3)
            ok = isinstance(got, ocr.Found) and _inside(got.point, box)
            rows.append((lab, ok, getattr(got, "level", type(got).__name__), ms))
        hindi = ocr.find_in(frame, "नया खेल", area=(1090.0, 150.0, 170.0, 360.0))
    for lab, ok, level, ms in rows:
        print(f"[ocr] find {lab!r:22} {'inside its button' if ok else 'MISSED'} ({level}, {ms:.0f} ms)")
    print(f"[ocr] find 'नया खेल': {type(hindi).__name__} ({getattr(hindi, 'why', '')})")
    assert all(ok for _, ok, _, _ in rows), rows
    assert isinstance(hindi, ocr.Missing) and "Devanagari" in hindi.why


def test_word_boxes_by_range_and_the_dialog_crop(vision):
    retina, truth = synthetic()
    frame = ocr.Pixels(retina, 0.0, 0.0, 0.5)
    sheet = truth["sheet"]
    with effects.allow("model"):
        fast_ms, fast = _ms(lambda: ocr.read(frame, sheet, plan=ocr.Plan(True, ("en-US",))), 5)
        acc_ms, acc = _ms(lambda: ocr.read(frame, sheet, plan=ocr.Plan(False, ("en-US",))), 5)
        title = ocr.read(frame, sheet, plan=ocr.Plan(False, ("en-US",)), ranges=lambda s: ocr._spans(s, "New Game"))
    print(f"[ocr] dialog crop 500×170 pt @2×: fast {fast_ms:.1f} ms {[ln.text for ln in fast.lines]}, accurate {acc_ms:.1f} ms")
    line = next(ln for ln in title.lines if "New Game" in ln.text)
    assert line.parts, "Vision gave no box for the words"
    x, y, w, h = line.parts[0][2]
    assert line.box[0] < x and x + w <= line.box[0] + line.box[2] + 1 and w < line.box[2] * 0.8


def test_text_marks_and_word_changes_timing(vision, monkeypatch):
    before_img, truth = synthetic(sheet=True)
    after_img, _ = synthetic(sheet=False)
    monkeypatch.setattr(ocr, "windows_over", lambda area: ocr.Windows((ocr.Win((0.0, 0.0, 1512.0, 982.0), 1, "Board"),)))
    window = (360.0, 90.0, 900.0, 800.0)
    with effects.allow("model"):
        ms, marks = _ms(lambda: ocr.text_marks(ocr.Pixels(before_img, 0.0, 0.0, 0.5), window), 3)
        crop = (560.0, 90.0, 500.0, 170.0)
        plan = ocr.plan_for("")
        b = ocr.read(ocr.Pixels(before_img, 0.0, 0.0, 0.5), crop, plan=plan)
        a = ocr.read(ocr.Pixels(after_img, 0.0, 0.0, 0.5), crop, plan=plan)
    said = ocr.words_changed(b, a)
    print(f"[ocr] text marks for a 900×800 pt window: {len(marks)} in {ms:.0f} ms; after the sheet closed: {said}")
    assert any(m.text == "Start a New Game?" for m in marks)
    assert "“Start a New Game?” is gone" in said


def test_a_whole_display_read_accurately_in_pieces_keeps_every_label(vision):
    retina, truth = synthetic()
    frame = ocr.Pixels(retina, 0.0, 0.0, 0.5)
    six = ["New Game", "Undo Move", "Hint", "Resign", "Settings", "Share…"]
    plan = ocr.Plan(False, ("en-US",))
    with effects.allow("model"):
        e = ocr.engine()
        whole_ms, raws = _ms(lambda: e.recognize(retina.convert("L"), fast=False, languages=["en-US"], auto=False, ranges=None), 3)
        cut_ms, reading = _ms(lambda: ocr.read(frame, plan=plan), 3)

    def hits(texts):
        return sum(any(ocr.key(w) == ocr.key(t) for t in texts) for w in six)

    before, after = hits([r.text for r in raws]), hits([ln.text for ln in reading.lines])
    print(f"[ocr] accurate, whole Retina display: in one piece {before}/6 sidebar labels ({whole_ms:.0f} ms) → in pieces {after}/6 ({cut_ms:.0f} ms)")
    assert after == 6 and after >= before


def test_finds_across_many_framings_never_pick_the_wrong_copy(vision):
    """30 framings of the window (as a window's frame lands on the screen): a single fast read taken as
    it is (before) against find_in (after) — wrong: a point outside the button meant; the sheet's "Start"
    is asked with the sheet known, the sidebar's without (two copies: ambiguous is the right answer)."""
    retina, truth = synthetic()
    frame = ocr.Pixels(retina, 0.0, 0.0, 0.5)
    rnd = random.Random(7)
    asks = [("Hint", truth["buttons"]["Hint"], None), ("Resign", truth["buttons"]["Resign"], None),
            ("Start", truth["buttons"]["sheet Start"], truth["sheet"]), ("Cancel", truth["buttons"]["sheet Cancel"], truth["sheet"]),
            ("Start", None, None)]  # fmt: skip
    tally = {"before": {"right": 0, "wrong": 0, "missed": 0, "ms": []}, "after": {"right": 0, "wrong": 0, "missed": 0, "ambiguous": 0, "ms": []}}
    fast = ocr.Plan(True, ("en-US",))
    with effects.allow("model"):
        for _ in range(30):
            x0, y0 = 540 + rnd.uniform(-60, 20), 70 + rnd.uniform(-40, 20)
            area = (x0, y0, 1260 - x0 + rnd.uniform(-20, 60), 520 - y0 + rnd.uniform(-20, 60))
            for label, box, sheet in asks:
                t = time.perf_counter()
                one = ocr.match(ocr.read(frame, area, plan=fast), label, sheet=sheet)
                tally["before"]["ms"].append((time.perf_counter() - t) * 1000)
                t = time.perf_counter()
                got = ocr.find_in(frame, label, area=area, sheet=sheet)
                tally["after"]["ms"].append((time.perf_counter() - t) * 1000)
                for k, res in (("before", one), ("after", got)):
                    if box is None:  # two copies, no sheet known: only "ambiguous" is right
                        ok = isinstance(res, ocr.Ambiguous)
                        tally[k]["right" if ok else ("missed" if isinstance(res, ocr.Missing) else "wrong")] += 1
                    elif isinstance(res, ocr.Found):
                        tally[k]["right" if _inside(res.point, box) else "wrong"] += 1
                    elif isinstance(res, ocr.Ambiguous):
                        tally[k]["ambiguous" if k == "after" else "missed"] += 1
                    else:
                        tally[k]["missed"] += 1
    for k in ("before", "after"):
        ms = tally[k].pop("ms")
        print(f"[ocr] {k:6} 150 finds: {tally[k]}, median {statistics.median(ms):.0f} ms, p90 {sorted(ms)[int(len(ms) * 0.9)]:.0f} ms")
    assert tally["after"]["wrong"] == 0
    assert tally["after"]["right"] >= tally["before"]["right"]


def test_the_words_around_a_click_before_and_after(vision):
    """A click's change samples (120×120 pt around the point, kept at full detail): the words that
    changed, read only where the samples differ."""
    from neo2.core.tools.computer import screen

    def crop(word, fill=(255, 255, 255)):
        img = Image.new("RGB", (240, 240), (250, 250, 250))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([60, 100, 180, 150], 12, fill=fill, outline=(170, 170, 170))
        d.text((88, 110), word, fill=(0, 0, 0), font=_font(SF, 26))
        s = img.convert("L").resize((24, 24), Image.BILINEAR)
        s.info.update({"full": img, "ppp": 0.5})
        return s

    before, after = crop("Start"), crop("Stop", fill=(170, 200, 250))  # (the button lit up as it changed)
    assert screen.differs(before, after)
    with effects.allow("model"):
        ms, said = _ms(lambda: ocr.sample_change([before], [after]), 5)
        same_ms, same = _ms(lambda: ocr.sample_change([before], [crop("Start")]), 5)
    print(f"[ocr] words around a click: {said} in {ms:.1f} ms; unchanged samples: {same!r} in {same_ms:.2f} ms")
    assert said == "“Start” → “Stop”" and same == ""
