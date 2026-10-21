"""Search Screen's on-device pass (neo2.core.lens.local): what may be opened and how it is shown (only
http(s), a user name in a link and punycode hosts flagged, other schemes copy only), e-mail and phone
checks, Wi-Fi codes, the black check (protected video), the language and "Translate to", OCR read
as prose — the engine told only then — and the shape of a pass with fake engines. Smart pick's
regions (neo2.core.lens.regions) from fakes: blocks, the instance boxes from a mask, windows."""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from neo2.core.lens import capture, local, regions
from neo2.core.tools.computer import ocr, screen


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(ocr, "_engine", None)
    monkeypatch.setattr(ocr, "_preferred", ["en-IN", "hi-IN"])
    local.set_engine(None)
    regions.set_engine(None)
    yield
    local.set_engine(None)
    regions.set_engine(None)


# ---- links --------------------------------------------------------------------------------------------


def test_only_web_links_open_and_their_full_host_shows():
    a = local.link("https://shop.example.com/p/123?q=1")
    assert (a.kind, a.label, a.detail, a.value, a.warn) == (
        "open", "Open shop.example.com", "shop.example.com", "https://shop.example.com/p/123?q=1", "",
    )  # fmt: skip
    assert local.link("http://www.example.com").label == "Open example.com"
    for raw in ("javascript:alert(1)", "file:///etc/passwd", "data:text/html,<b>x</b>", "slack://open?team=1"):
        a = local.link(raw)
        assert a.kind == "code" and a.value == raw, raw  # copied, never opened
    assert local.link("https://" + "a" * 2100 + ".com").kind == "code"  # too long to open


def test_a_user_name_in_a_link_is_dropped_and_flagged():
    a = local.link("https://paypal.com@evil.example/login")
    assert (a.kind, a.detail, a.warn) == ("open", "evil.example", "userinfo")
    assert a.value == "https://evil.example/login" and "paypal" not in a.value


def test_punycode_and_mixed_scripts_show_both_forms():
    a = local.link("https://xn--pypal-4ve.com/")  # "pаypal" with a Cyrillic а
    assert a.warn == "punycode" and "xn--pypal-4ve.com" in a.detail and "pаypal.com" in a.detail
    b = local.link("https://bücher.de/x")
    assert b.warn == "punycode" and "xn--bcher-kva.de" in b.detail and b.value == "https://xn--bcher-kva.de/x"


def test_mail_and_phone_are_checked():
    assert local.link("mailto:anna@example.org").kind == "mail"
    assert local.link("mailto:anna@example.org").value == "mailto:anna@example.org"
    assert local.link("mailto:not an address").kind == "code"
    assert local.link("tel:+91 98765 43210").kind == "call"
    assert local.link("tel:+91 98765 43210").value == "tel:+919876543210"
    assert local.link("tel:12;rm -rf").kind == "code"


@pytest.mark.parametrize(
    "raw,want",
    [
        ("WIFI:T:WPA;S:Home Net;P:pa\\;ss;;", ("wifi", "Copy Wi-Fi password", "pa;ss", "Home Net")),
        ("WIFI:S:Cafe;T:nopass;P:;;", ("code", "Copy network name", "Cafe", "Cafe")),
        ("wifi:T:WEP;P:s3cret;S:Lab\\:2;;", ("wifi", "Copy Wi-Fi password", "s3cret", "Lab:2")),
    ],
)
def test_wifi_codes(raw, want):
    a = local.link(raw)
    assert (a.kind, a.label, a.value, a.detail) == want


def test_the_plain_detector_when_foundation_cant():
    got = local.detect_plain("see https://x.org/a), or mail me@ex.com.")
    assert ("link", "https://x.org/a") in got and ("link", "mailto:me@ex.com") in got


def test_foundations_detector_finds_links_mail_and_phones():
    got = local.detect_with_foundation("Call +1 415 555 0100 or mail a@b.com, see https://x.org/p")
    kinds = {k for k, _ in got}
    assert kinds == {"link", "phone"} and ("link", "mailto:a@b.com") in got and ("link", "https://x.org/p") in got


# ---- the black check, the language ---------------------------------------------------------------------


def test_a_black_area_is_protected_video():
    assert local.is_dark(Image.new("RGB", (800, 450), (3, 3, 3)))
    almost = Image.new("RGB", (800, 450), (2, 2, 2))
    ImageDraw.Draw(almost).rectangle((0, 0, 799, 30), fill=(255, 255, 255))  # (subtitles: 7% of it)
    assert not local.is_dark(almost)


def _terminal(size, lines):
    img = Image.new("RGB", size, (0, 0, 0))
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        d.text((16, 16 + 22 * i), line, fill=(230, 230, 230))
    return img


def test_text_on_black_is_never_protected_video():
    # A black terminal's error (iTerm2's default, Terminal's "Pro"): a few lines in a big black area,
    # about 97-98% of it black — the user's "explain this error", never "came out black".
    assert not local.is_dark(_terminal((1200, 400), ["Traceback (most recent call last):", "KeyError: 'id'"]))
    assert not local.is_dark(_terminal((1000, 300), ["zsh: command not found: pyhton"]))
    assert not local.is_dark(_terminal((900, 500), ["def main():", "    return 0", "", "main()"]))


class FakeEngine:
    def __init__(self, *, codes=(), lang=("fr", 0.98), detect=None):
        self._codes, self._lang, self._detect = list(codes), lang, detect
        self.langs: list[str] = []

    def codes(self, img):
        return list(self._codes)

    def language(self, text):
        self.langs.append(text)
        return self._lang

    def detect(self, text):
        return self._detect(text) if self._detect else local.detect_plain(text)


def test_the_language_needs_enough_letters_and_a_sure_guess():
    local.set_engine(FakeEngine(lang=("fr", 0.98)))
    assert local.language("Bonjour mon ami") == {"code": "fr", "name": "French", "conf": 0.98}
    assert local.language("Oui") is None  # too short to tell
    local.set_engine(FakeEngine(lang=("ja", 0.9)))
    assert local.language("こんにちは")["name"] == "Japanese"  # (3 letters are enough in Japanese)
    local.set_engine(FakeEngine(lang=("de", 0.4)))
    assert local.language("Das ist vielleicht Deutsch") is None  # not sure enough


def test_translate_to_the_users_first_language_that_isnt_the_texts():
    assert local.translate_to({"code": "fr"}) == "en"
    assert local.translate_to({"code": "en-GB"}) == "hi"  # (English and Hindi on this Mac)
    assert local.translate_to(None) == ""


def test_a_short_foreign_word_still_gets_translate():
    local.set_engine(FakeEngine(lang=("fr", 0.35)))
    assert local.language("Merci") is None and local.guess("Merci") == "fr"  # too short to be sure
    assert local.translate_to(None, "fr") == "en"
    assert local.translate_to(None, "zh-Hans") == "en"  # 出口 on a sign
    assert local.translate_to(None, "en") == "" and local.translate_to(None, "hi") == ""  # the user's own ("Cancel")
    assert local.translate_to(None, "") == ""
    assert local.guess("1234 5678") == ""  # no letters, no guess asked for
    local.set_engine(FakeEngine(lang=None))
    assert local.guess("出口") == ""
    local.set_engine(FakeEngine(lang=("und", 0.2)))
    assert local.guess("xq") == ""
    # Through the pass: no sure language, the guess kept for Translate.
    ocr.set_engine(Words("Merci"))
    local.set_engine(FakeEngine(lang=("fr", 0.35)))
    got = local.analyse(_crop())
    assert got.lang is None and got.guess == "fr"
    menu = [x["code"] for x in local.languages()]
    assert menu[:2] == ["en", "hi"] and "ja" in menu and len(menu) == len(set(menu))


# ---- the pass, and OCR read as prose ---------------------------------------------------------------------------


class Words:
    """A fake Vision that reads one line, and remembers how it was asked."""

    def __init__(self, text):
        self.text, self.calls = text, []

    def languages(self, fast):
        return ["en-US", "fr-FR"]

    def recognize(self, image, *, fast, languages, auto, ranges, **kw):
        self.calls.append({"fast": fast, "languages": list(languages), "auto": auto, **kw})
        return [ocr.Raw(self.text, 1.0, (0.1, 0.4, 0.8, 0.2))]


def _crop(color=(250, 250, 250), size=(400, 200)):
    return screen.Frame(Image.new("RGB", size, color), 100.0, 50.0, 0.5)


def test_the_area_is_read_as_prose_and_only_it(monkeypatch):
    eng = Words("Prix : 3 499 € — voir https://boutique.example.fr/sac")
    ocr.set_engine(eng)
    local.set_engine(FakeEngine(codes=[("WIFI:T:WPA;S:Shop;P:hunter2;;", "QR")]))
    got = local.analyse(_crop())
    assert eng.calls == [{"fast": False, "languages": [], "auto": True, "prose": True}]
    assert got.text.startswith("Prix") and got.lines == 1 and got.lang["code"] == "fr" and not got.dark
    kinds = [a.kind for a in got.actions]
    assert kinds == ["copy_text", "wifi", "open", "copy_image"]
    assert got.actions[2].detail == "boutique.example.fr"
    # Everyone else reads as before: the engine never hears of prose.
    eng.calls.clear()
    ocr.read(_crop(), plan=ocr.Plan(fast=False, languages=("en-US",)))
    assert "prose" not in eng.calls[0]


class NoWords(Words):
    def __init__(self):
        super().__init__("")

    def recognize(self, image, **kw):
        self.calls.append(kw)
        return []


def test_an_area_is_dark_only_when_nothing_was_found_on_it(monkeypatch):
    ocr.set_engine(NoWords())
    local.set_engine(FakeEngine())
    got = local.analyse(_crop((0, 0, 0)))
    assert got.dark and got.actions == [] and got.text == ""  # (protected video: nothing to copy)
    # One short error in a big black terminal: read like any other area (dark is the last verdict).
    eng = Words("zsh: command not found: pyhton")
    ocr.set_engine(eng)
    got = local.analyse(screen.Frame(_terminal((1600, 900), ["zsh: command not found: pyhton"]), 0.0, 0.0, 0.5))
    assert not got.dark and got.text.startswith("zsh") and len(eng.calls) == 1
    assert [a.kind for a in got.actions] == ["copy_text", "copy_image"]
    # A code on black (a QR on a dark page) is never dark either.
    ocr.set_engine(NoWords())
    local.set_engine(FakeEngine(codes=[("https://example.com/x", "QR")]))
    got = local.analyse(_crop((0, 0, 0)))
    assert not got.dark and [a.kind for a in got.actions] == ["open", "copy_image"]


def test_a_part_that_cant_run_is_left_out(monkeypatch):
    class Broken(FakeEngine):
        def codes(self, img):
            raise RuntimeError("no barcodes here")

    ocr.set_engine(Words("hello world"))
    local.set_engine(Broken())
    got = local.analyse(_crop())
    assert got.notes == ["codes"] and got.text == "hello world" and got.actions[0].kind == "copy_text"


# ---- smart pick's regions -------------------------------------------------------------------------------


def test_lines_make_blocks_by_column_and_gap():
    boxes = [(10, 10, 200, 14), (10, 28, 180, 14), (10, 46, 150, 14), (300, 10, 100, 14), (10, 120, 90, 14)]
    of, blocks = regions.group_blocks(boxes)
    assert of == [0, 0, 0, 1, 2]
    assert blocks[0] == [10.0, 10.0, 200.0, 50.0]


def test_each_things_box_comes_from_its_mask():
    m = Image.new("L", (100, 50), 0)
    d = ImageDraw.Draw(m)
    d.rectangle((10, 5, 29, 24), fill=1)
    d.rectangle((60, 20, 89, 44), fill=2)
    assert regions.instance_boxes(m, 2) == [(0.1, 0.1, 0.2, 0.4), (0.6, 0.4, 0.3, 0.5)]


class Things:
    def objects(self, img):
        return [(0.1, 0.1, 0.2, 0.2), (0.0, 0.0, 0.0, 0.1)]

    def instances(self, img):
        return [(0.5, 0.5, 0.25, 0.25), (0.0, 0.0, 0.9, 0.9)]  # (the second: nearly all of it — no pick)


def test_regions_from_the_still_in_display_points(monkeypatch):
    sp = capture.Spot(1, 1512.0, -200.0, 1000.0, 600.0, 2.0)
    frame = screen.Frame(Image.new("RGB", (2000, 1200), (240, 240, 240)), 1512.0, -200.0, 0.5)
    ocr.set_engine(Words("Add to cart"))
    regions.set_engine(Things())
    monkeypatch.setattr(regions, "windows", lambda sp: [[0.0, 0.0, 800.0, 500.0]])
    got = regions.compute(frame, sp, texts=True)
    x, y, w, h, block = got["lines"][0]
    assert (round(x), round(y), round(w), round(h), block) == (100, 240, 800, 120, 0)  # (display-local)
    assert got["texts"] == ["Add to cart"]
    assert got["objects"] == [[100.0, 60.0, 200.0, 120.0]]
    assert got["instances"] == [[500.0, 300.0, 250.0, 150.0]]
    assert got["windows"] == [[0.0, 0.0, 800.0, 500.0]]
    assert "texts" not in regions.compute(frame, sp)
    live = regions.compute(None, sp)  # live mode: windows only
    assert live["lines"] == [] and live["objects"] == [] and live["windows"]
