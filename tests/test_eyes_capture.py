"""Eyes, capture: every screenshot is grabbed once at the display's own resolution and kept as a frame,
the model gets a small JPEG (or PNG, the A/B switch) that maps exactly onto the screen, zooms keep
Retina detail, window shots can be clicked through, and ScreenCaptureKit captures in process when
switched on — falling back to the screencapture CLI whenever it can't.

Hermetic: screencapture, the window server and ScreenCaptureKit are fakes (synthetic pictures, a
fake framework whose completion handlers run on their own thread, CGImages made in memory);
nothing is captured and nothing reaches the real Mac.

``python tests/test_eyes_capture.py`` prints the before/after benchmark (synthetic 3024×1964 screens).
"""

from __future__ import annotations

import functools
import io
import os
import random
import statistics
import subprocess
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import Quartz as Q
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from neo import effects
from neo.tools.computer import sck, screen

_REAL_SAMPLE = screen.sample_points  # (the suite's conftest makes it "no picture" in every test)
_REAL_STACK = getattr(screen, "_window_stack", None)  # (faked in every test here: the window server)
W2, H2 = 3024, 1964  # a 14" MacBook Pro's Retina frame (1512×982 pt)


# ---- synthetic screens -------------------------------------------------------------------------------


def _font(size: int) -> ImageFont.ImageFont:
    for path in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/SFNS.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


_WORDS = (
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "Settings", "General", "Display",
    "Battery", "Network", "Sound", "Focus", "Privacy", "Security", "Desktop", "Dock", "Notes", "Safari",
    "Mail", "Calendar", "Reminders", "Photos", "Messages", "Music", "Maps", "Finder", "Downloads",
    "Documents", "Applications", "Recents", "Shared", "AirDrop", "Tags",
)


def _wallpaper(w: int, h: int) -> Image.Image:
    """A smooth gradient with soft noise, like a macOS wallpaper (photo-like: hard for PNG)."""
    base = Image.linear_gradient("L").resize((w, h))
    img = Image.merge(
        "RGB",
        (base.point(lambda v: 40 + v // 3), base.rotate(90).resize((w, h)).point(lambda v: 60 + v // 2), Image.new("L", (w, h), 150)),
    )
    noise = Image.effect_noise((max(1, w // 4), max(1, h // 4)), 40).resize((w, h), Image.BILINEAR)
    return Image.blend(img, Image.merge("RGB", (noise, noise, noise)), 0.25)


@functools.cache
def synthetic_screen(kind: str = "ui", size: tuple[int, int] = (W2, H2), seed: int = 1) -> Image.Image:
    """A screen-like Retina picture (the design-speed benchmark's): "ui" (windows of text over a flat
    desktop), "mixed" (a wallpaper, and a photo in the window), "board" (a textured 8×8 board)."""
    rng = random.Random(seed)
    w, h = size
    s = w / W2 * 2  # pixels per point
    img = _wallpaper(w, h) if kind in ("mixed", "board") else Image.new("RGB", size, (236, 236, 238))
    d = ImageDraw.Draw(img)
    f13, f20 = _font(max(8, round(13 * s))), _font(max(8, round(20 * s)))
    d.rectangle((0, 0, w, 24 * s), fill=(246, 246, 246))
    for i, word in enumerate(("Finder", "File", "Edit", "View", "Go", "Window", "Help")):
        d.text(((40 + 70 * i) * s, 5 * s), word, font=f13, fill=(20, 20, 20))
    wx, wy, ww, wh = 120 * s, 70 * s, 1100 * s, 800 * s
    d.rounded_rectangle((wx, wy, wx + ww, wy + wh), 12 * s, fill=(255, 255, 255), outline=(200, 200, 200))
    d.rectangle((wx, wy + 52 * s, wx + 220 * s, wy + wh), fill=(240, 240, 244))
    for i in range(30):
        words = " ".join(rng.choice(_WORDS) for _ in range(2))
        d.text((wx + 30 * s, wy + (64 + 22 * i) * s), words.title(), font=f13, fill=(30, 30, 30))
    cx, cy = wx + 240 * s, wy + 70 * s
    if kind == "board":
        sq = max(2, round(80 * s))
        for r in range(8):
            for c in range(8):
                col = (240, 217, 181) if (r + c) % 2 == 0 else (181, 136, 99)
                tile = Image.blend(Image.new("RGB", (sq, sq), col), Image.effect_noise((sq, sq), 18).convert("RGB"), 0.12)
                img.paste(tile, (round(cx + c * sq), round(cy + r * sq)))
                if r in (0, 1, 6, 7):
                    x0, y0 = cx + c * sq, cy + r * sq
                    d.ellipse((x0 + 18 * s, y0 + 14 * s, x0 + 62 * s, y0 + 66 * s), fill=(250, 250, 250) if r > 4 else (30, 30, 30))
    else:
        d.text((cx, cy), " ".join(rng.choice(_WORDS) for _ in range(6)).title(), font=f20, fill=(0, 0, 0))
        for i in range(30):
            d.text((cx, cy + (40 + 20 * i) * s), " ".join(rng.choice(_WORDS) for _ in range(16)), font=f13, fill=(40, 40, 40))
        for i in range(5):
            bx = cx + i * 130 * s
            d.rounded_rectangle((bx, cy + 680 * s, bx + 110 * s, cy + 706 * s), 6 * s, fill=(10, 132, 255))
        if kind == "mixed":
            pw, ph = max(1, round(360 * s)), max(1, round(240 * s))
            photo = _wallpaper(pw, ph).filter(ImageFilter.DETAIL)
            photo = Image.blend(photo, Image.effect_noise((pw, ph), 60).convert("RGB"), 0.3)
            img.paste(photo, (round(cx + 470 * s), round(cy + 60 * s)))
    return img


# ---- fixtures ----------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_capture_state(monkeypatch):
    monkeypatch.setattr(screen, "_last", None)
    monkeypatch.setattr(screen, "_cli_format", screen._CLI_FORMAT)
    monkeypatch.setattr(sck, "_SC", None)
    monkeypatch.setattr(sck, "_content", None)
    monkeypatch.setattr(sck, "_misses", 0)
    monkeypatch.setattr(sck, "_off", "")
    # Hermetic: the window server is never asked (a test that needs a window fakes these itself).
    monkeypatch.setattr(screen, "_window_bounds", lambda wid: None)
    monkeypatch.setattr(screen, "_window_stack", lambda wid: None, raising=False)
    monkeypatch.setattr(sck, "_on_screen_owners", lambda: set(), raising=False)
    monkeypatch.setattr(screen.Q, "CGWindowListCopyWindowInfo", _no_window_server)


class _ReachedTheWindowServer(BaseException):  # (a BaseException: no "can't tell" handler swallows it)
    pass


def _no_window_server(*args):
    raise _ReachedTheWindowServer("a test asked the real window server")


def _settings(monkeypatch, **values):
    real = screen._setting
    monkeypatch.setattr(screen, "_setting", lambda name, default: values.get(name, real(name, default)))


@pytest.fixture
def cli(monkeypatch):
    """screencapture faked at NEO's own seam (_grab): every run's arguments, and the picture it gives."""
    runs: list[list[str]] = []

    def retina(extra):  # a region's pixels at 2× (a Retina display), or a whole 14" screen
        if extra[:1] == ["-R"]:
            w, h = (int(v) for v in extra[1].split(",")[2:])
            return synthetic_screen("ui", (2 * w, 2 * h))
        return synthetic_screen("ui")

    world = SimpleNamespace(runs=runs, picture=retina)

    def grab(extra):
        runs.append(list(extra))
        return world.picture(extra)

    monkeypatch.setattr(screen, "can_capture", lambda: True)
    monkeypatch.setattr(screen, "_grab", grab)
    return world


def _decode(part) -> Image.Image:
    return Image.open(io.BytesIO(part.data))


# ---- the pipeline ------------------------------------------------------------------------------------


def test_a_retina_display_becomes_a_points_sized_jpeg_and_its_frame_is_kept(cli):
    c = screen.shot()
    img = _decode(c.part)
    assert cli.runs == [[]]  # (the main display: screencapture's default)
    assert img.format == "JPEG" and c.part.mime == "image/jpeg" and img.size == (1512, 982)
    assert (c.w, c.h, c.ppp, c.x0, c.y0) == (1512, 982, 1.0, 0.0, 0.0) and c.kind == "full" and c.clickable
    assert getattr(c.part, "detail", None) == "high"
    f = screen.last_frame()
    assert f is c.frame and f.image.size == (W2, H2) and f.ppp == 0.5 and f.backend == "cli"
    assert screen.last_frame(max_age_s=60) is f and screen.capture()[1] == 1.0
    assert c.part.data[:2] == b"\xff\xd8" and len(c.part.data) < 400_000


def test_the_png_switch_is_lossless_and_never_optimized(cli, monkeypatch):
    _settings(monkeypatch, shot_format="png")
    saved: list[dict] = []
    real_save = Image.Image.save

    def save(self, fp, format=None, **kw):
        saved.append({"format": format, **kw})
        return real_save(self, fp, format=format, **kw)

    monkeypatch.setattr(Image.Image, "save", save)
    c = screen.shot()
    assert c.part.mime == "image/png" and _decode(c.part).format == "PNG"
    assert saved == [{"format": "PNG", "compress_level": 1}]  # never optimize=True (100–290 ms)
    small = c.frame.image.reduce(2)
    assert np.array_equal(np.asarray(_decode(c.part).convert("RGB")), np.asarray(small))  # lossless
    _settings(monkeypatch, shot_format="bogus")  # a typo is the default, never an error
    assert screen.shot().part.mime == "image/jpeg"


def test_a_16_inch_display_is_reduced_then_capped_and_a_1x_display_is_scaled(cli, monkeypatch):
    big = [screen.Display(1, 0.0, 0.0, 1728.0, 1117.0)]
    monkeypatch.setattr(screen, "displays", lambda: big)
    cli.picture = lambda extra: synthetic_screen("ui", (3456, 2234))
    c = screen.shot()
    assert (c.w, c.h) == (1600, 1034) and c.ppp == pytest.approx(1.08)
    two = [screen.Display(1, 0.0, 0.0, 1512.0, 982.0), screen.Display(2, 1512.0, -200.0, 1920.0, 1080.0)]
    monkeypatch.setattr(screen, "displays", lambda: two)
    cli.picture = lambda extra: Image.new("RGB", (1920, 1080), (90, 90, 90))
    c = screen.shot(display=two[1])
    assert cli.runs[-1] == ["-D", "2"] and (c.w, c.h) == (1600, 900) and c.ppp == pytest.approx(1.2)
    assert c.to_screen(100, 100) == pytest.approx((1512 + 120, -200 + 120))


def test_the_brains_image_limit_sizes_the_picture(cli, monkeypatch):
    from neo import providers

    monkeypatch.setattr(providers, "image_limit", lambda: 1024, raising=False)
    c = screen.shot()
    assert (c.w, c.h) == (1024, 665) and c.ppp == pytest.approx(1512 / 1024)
    assert c.to_screen(1024, 0)[0] == pytest.approx(1512.0)  # the image the model sees is the one NEO mapped
    monkeypatch.setattr(providers, "image_limit", lambda: (800, 400), raising=False)
    c = screen.shot()
    assert c.w <= 800 and c.h <= 400 and c.ppp == pytest.approx(1512 / c.w)
    monkeypatch.setattr(providers, "image_limit", lambda: None, raising=False)
    assert screen.shot().w == 1512
    monkeypatch.setattr(providers, "image_limit", lambda: "weird", raising=False)
    assert screen.shot().w == 1512  # an answer it can't read: NEO's own caps


# ---- regions and zooms keep Retina detail ------------------------------------------------------------


def test_a_region_keeps_the_displays_own_pixels_and_maps_from_its_origin(cli):
    cli.picture = lambda extra: synthetic_screen("ui", (800, 500))
    c = screen.shot((300, 200, 400, 250))
    assert cli.runs == [["-R", "300,200,400,250"]]
    assert (c.w, c.h, c.ppp) == (800, 500, 0.5) and c.kind == "region"
    assert c.to_screen(400, 250) == (500.0, 325.0)  # the centre of the region
    part, ppp = screen.capture((300, 200, 400, 250))
    assert ppp == 0.5 and _decode(part).size == (800, 500)


def test_a_big_region_is_points_and_a_small_one_is_scaled_up(cli):
    cli.picture = lambda extra: Image.new("RGB", (2000, 1200), (200, 200, 200))
    c = screen.shot((0, 0, 1000, 600))
    assert (c.w, c.h, c.ppp) == (1000, 600, 1.0)  # past _ZOOM_MAX natively: points (an exact halving)
    cli.picture = lambda extra: Image.new("RGB", (200, 120), (200, 200, 200))
    c = screen.shot((10, 10, 100, 60))
    assert (c.w, c.h) == (768, 461) and c.ppp == pytest.approx(100 / 768)  # small text, read big
    assert c.to_screen(768, 461) == pytest.approx((110.0, 70.0), abs=0.2)


def test_a_zoom_is_centred_on_the_point_and_stays_on_its_display(cli):
    c = screen.zoom(700.0, 500.0)
    assert cli.runs[-1] == ["-R", "540,400,320,200"] and c.kind == "zoom" and c.part.detail == "high"
    assert (c.w, c.h, c.ppp) == (768, 480, pytest.approx(320 / 768))  # 2× Retina, then up to _ZOOM_MIN
    assert c.to_screen(384, 240) == pytest.approx((700.0, 500.0))
    screen.zoom(5.0, 5.0)
    assert cli.runs[-1] == ["-R", "0,0,320,200"]  # near the corner: moved onto the screen, not cut
    screen.zoom(1510.0, 980.0, 100, 50)
    assert cli.runs[-1] == ["-R", "1412,932,100,50"]
    assert screen.last_frame().rect == pytest.approx((1412.0, 932.0, 100.0, 50.0))


# ---- windows -----------------------------------------------------------------------------------------


def test_a_window_shot_maps_through_the_window_servers_bounds(cli, monkeypatch):
    cli.picture = lambda extra: synthetic_screen("ui", (2400, 1600))
    monkeypatch.setattr(screen, "_window_bounds", lambda wid: (100.0, 50.0, 1200.0, 800.0))
    monkeypatch.setattr(screen, "_window_stack", lambda wid: (4321, []), raising=False)  # on screen, nothing over it
    c = screen.window_shot(77)
    assert cli.runs == [["-o", "-l", "77"]]  # its own pixels, no shadow
    assert c.clickable and c.kind == "window" and (c.w, c.h, c.ppp) == (1200, 800, 1.0)
    assert c.to_screen(10, 10) == (110.0, 60.0) and c.frame.window_id == 77
    cut = screen.window_shot(77, (100.0, 130.0, 1200.0, 720.0))  # without the tab strip
    assert (cut.x0, cut.y0, cut.w, cut.h) == (100.0, 130.0, 1200, 720) and cut.to_screen(0, 0) == (100.0, 130.0)
    cli.picture = lambda extra: synthetic_screen("ui", (1200, 800))  # a window that fits: its own pixels
    monkeypatch.setattr(screen, "_window_bounds", lambda wid: (0.0, 0.0, 600.0, 400.0))
    small = screen.window_shot(77)
    assert (small.w, small.h, small.ppp) == (1200, 800, 0.5)


def test_a_window_picture_that_doesnt_match_its_bounds_is_for_looking_only(cli, monkeypatch):
    cli.picture = lambda extra: Image.new("RGB", (2600, 1600), "white")  # (a shadow, or another window)
    monkeypatch.setattr(screen, "_window_bounds", lambda wid: (100.0, 50.0, 1200.0, 800.0))
    assert not screen.window_shot(77).clickable
    moving = iter([(100.0, 50.0, 1300.0, 800.0), (140.0, 50.0, 1300.0, 800.0)])  # dragged while captured
    monkeypatch.setattr(screen, "_window_bounds", lambda wid: next(moving))
    assert not screen.window_shot(77).clickable
    monkeypatch.setattr(screen, "_window_bounds", lambda wid: None)
    c = screen.window_shot(77)
    assert not c.clickable and max(c.w, c.h) <= 1600
    assert isinstance(screen.capture_window(77), type(c.part))


# ---- one capture for all samples ---------------------------------------------------------------------


def test_an_after_look_and_its_change_samples_come_from_one_capture(cli):
    look, samples = screen.after_look((200.0, 100.0, 400.0, 300.0), [(250.0, 150.0), (900.0, 500.0)])
    assert len(cli.runs) == 1 and cli.runs[0] == ["-R", "190,90,770,470"]  # the look and both points
    assert look.kind == "after" and look.part.detail == "medium"
    assert (look.x0, look.y0, look.w, look.h, look.ppp) == (200.0, 100.0, 800, 600, 0.5)
    assert [s.size for s in samples] == [(24, 24), (24, 24)] and all(s.mode == "L" for s in samples)
    assert screen.last_frame() is look.frame


def test_a_sample_before_and_an_after_look_after_agree_on_an_unchanged_screen(cli):
    """The pair a pointer action compares: sample_points before, after_look after. Cut the same way,
    so an unchanged screen is no change and a moved piece is one."""
    screen_now = {"img": synthetic_screen("board")}

    def region(extra):
        x, y, w, h = (int(v) for v in extra[1].split(","))
        return screen_now["img"].crop((2 * x, 2 * y, 2 * (x + w), 2 * (y + h)))

    cli.picture = region
    pts = [(403.7, 301.2), (651.5, 455.9)]  # (fractional points: a shot's pixels mapped to the screen)
    before = _REAL_SAMPLE(pts)
    _, after = screen.after_look((300.0, 200.0, 500.0, 400.0), pts, like=before)
    assert [np.asarray(a).tolist() for a in after] == [np.asarray(b).tolist() for b in before]
    moved = screen_now["img"].copy()
    ImageDraw.Draw(moved).ellipse((2 * 630, 2 * 435, 2 * 675, 2 * 480), fill=(20, 20, 20))
    screen_now["img"] = moved
    _, after = screen.after_look((300.0, 200.0, 500.0, 400.0), pts, like=before)
    assert [screen.differs(b, a) for b, a in zip(before, after, strict=True)] == [False, True]


def test_change_samples_are_cut_from_a_fresh_frame_without_capturing(cli):
    frame = screen.Frame(synthetic_screen("ui"), 0.0, 0.0, 0.5)
    got = _REAL_SAMPLE([(300.0, 300.0), (700.0, 500.0)], frame=frame)
    assert cli.runs == [] and [g.size for g in got] == [(24, 24), (24, 24)]
    again = _REAL_SAMPLE([(300.0, 300.0), (700.0, 500.0)])  # no frame: one capture for both points
    assert cli.runs == [["-R", "240,240,520,320"]] and len(again) == 2
    part = screen.Frame(Image.new("RGB", (200, 200)), 0.0, 0.0, 0.5)  # a frame that doesn't show them
    _REAL_SAMPLE([(300.0, 300.0)], frame=part)
    assert len(cli.runs) == 2


# ---- the CLI -----------------------------------------------------------------------------------------


def test_the_cli_writes_an_uncompressed_tiff_and_falls_back_to_png_once(monkeypatch):
    runs: list[list[str]] = []

    def run(cmd, check, timeout):
        runs.append(cmd)
        if cmd[3] == "tiff" and len(runs) == 1:
            raise subprocess.CalledProcessError(1, cmd)
        Image.new("RGB", (64, 40), (10, 20, 30)).save(cmd[-1], format="TIFF" if cmd[3] == "tiff" else "PNG")

    monkeypatch.setattr(screen.subprocess, "run", run)
    img = screen._grab([])  # (a whole display that fails as TIFF and not as PNG)
    assert [c[:4] for c in runs] == [["screencapture", "-x", "-t", "tiff"], ["screencapture", "-x", "-t", "png"]]
    assert img.mode == "RGB" and img.getpixel((0, 0)) == (10, 20, 30)
    screen._grab([])
    assert runs[-1][3] == "png" and not any(os.path.exists(c[-1]) for c in runs)  # the files are gone
    monkeypatch.setattr(screen, "_cli_format", "tiff")
    runs.clear()
    assert screen._grab([]).size == (64, 40) and runs[0][3] == "tiff" and runs[0][-1].endswith(".tiff")


def test_captures_never_overlap(monkeypatch):
    live, most = [0], [0]

    def run(cmd, check, timeout):
        live[0] += 1
        most[0] = max(most[0], live[0])
        time.sleep(0.02)
        Image.new("RGB", (8, 8)).save(cmd[-1], format="TIFF")
        live[0] -= 1

    monkeypatch.setattr(screen.subprocess, "run", run)
    threads = [threading.Thread(target=screen._grab, args=([],)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert most[0] == 1


def test_a_real_capture_is_an_effect_and_needs_the_permission(monkeypatch):
    monkeypatch.setattr(screen, "can_capture", lambda: True)
    with pytest.raises(effects.EffectRefused):
        screen.shot()  # (screencapture itself: refused while effects are off)
    monkeypatch.setattr(screen, "can_capture", lambda: False)
    monkeypatch.setattr(screen, "_request_access", lambda: None)
    for call in (screen.shot, lambda: screen.zoom(10, 10), lambda: screen.window_shot(1)):
        with pytest.raises(PermissionError):
            call()
    assert _REAL_SAMPLE([(10.0, 10.0)]) is None  # samples never ask: unknown


# ---- ScreenCaptureKit (a fake framework) -------------------------------------------------------------


def cgimage(img: Image.Image, layout: str = "bgra", pad: int = 0):
    """A CGImage made in memory from a PIL picture, in one of the pixel layouts CoreGraphics uses.
    Returns (image, the bytes behind it: keep them alive)."""
    w, h = img.size
    order, info, bpc, bpp = {
        "bgra": ([2, 1, 0, 3], Q.kCGImageAlphaPremultipliedFirst | Q.kCGBitmapByteOrder32Little, 8, 32),
        "rgba": ([0, 1, 2, 3], Q.kCGImageAlphaPremultipliedLast | Q.kCGBitmapByteOrder32Big, 8, 32),
        "argb": ([3, 0, 1, 2], Q.kCGImageAlphaNoneSkipFirst | Q.kCGBitmapByteOrderDefault, 8, 32),
        "abgr": ([3, 2, 1, 0], Q.kCGImageAlphaNoneSkipLast | Q.kCGBitmapByteOrder32Little, 8, 32),
        "rgba16": ([0, 1, 2, 3], Q.kCGImageAlphaPremultipliedLast | Q.kCGBitmapByteOrder16Little, 16, 64),
    }[layout]
    arr = np.asarray(img.convert("RGBA"))[..., order]
    if bpc == 16:
        rows, row = (arr.astype(np.uint16) * 257).astype("<u2").tobytes(), w * 8
    else:
        rows, row = np.ascontiguousarray(arr).tobytes(), w * 4
    data = b"".join(rows[i * row : (i + 1) * row] + b"\0" * pad for i in range(h))
    provider = Q.CGDataProviderCreateWithData(None, data, len(data), None)
    space = Q.CGColorSpaceCreateWithName(Q.kCGColorSpaceSRGB)
    cg = Q.CGImageCreate(w, h, bpc, bpp, row + pad, space, info, provider, None, False, Q.kCGRenderingIntentDefault)
    return cg, data


@pytest.mark.parametrize("layout", ["bgra", "rgba", "argb", "abgr"])
def test_a_cgimage_becomes_the_same_rgb_picture(layout):
    src = synthetic_screen("board", (320, 208))
    cg, _keep = cgimage(src, layout, pad=64 if layout == "bgra" else 0)  # (rows padded as CoreGraphics does)
    assert np.array_equal(np.asarray(sck.to_pil(cg)), np.asarray(src))


def test_a_16_bit_cgimage_is_drawn_into_8_bits_first():
    src = Image.new("RGB", (40, 30), (200, 30, 90))
    cg, _keep = cgimage(src, "rgba16")
    got = np.asarray(sck.to_pil(cg)).astype(int)
    assert got.shape == (30, 40, 3) and np.abs(got - np.array([200, 30, 90])).max() <= 2


class _Rect:
    def __init__(self, x, y, w, h):
        self.origin, self.size = SimpleNamespace(x=x, y=y), SimpleNamespace(width=w, height=h)


class FakeSCK:
    """ScreenCaptureKit as NEO calls it: shareable content, filters, configurations, and a one-shot
    capture whose completion handler runs on another thread (as on ScreenCaptureKit's own queue)."""

    def __init__(self, scale: float = 2.0):
        fake = self
        self.scale = scale
        self.shots: list[tuple] = []
        self.content_calls = 0
        self.answer = True  # False: the completion handler never comes
        self.error: str | None = None
        self.displays = [SimpleNamespace(displayID=lambda: 1, frame=lambda: _Rect(0, 0, 1512, 982))]
        self.apps = [
            SimpleNamespace(processID=lambda: 999, applicationName=lambda: "NEO"),
            SimpleNamespace(processID=lambda: 42, applicationName=lambda: "Some App"),
        ]
        self.windows = [SimpleNamespace(windowID=lambda: 77, frame=lambda: _Rect(100, 50, 600, 400))]
        self._keep: list = []

        def later(handler, *args):
            if fake.answer:
                threading.Thread(target=handler, args=args).start()

        class SCShareableContent:
            @staticmethod
            def getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler_(ex, on, handler):
                fake.content_calls += 1
                shown, apps, windows = list(fake.displays), list(fake.apps), list(fake.windows)  # (as of now)
                content = SimpleNamespace(displays=lambda: shown, applications=lambda: apps, windows=lambda: windows)
                threading.Thread(target=handler, args=(content, None)).start()

            @staticmethod
            def infoForFilter_(flt):
                return SimpleNamespace(pointPixelScale=lambda: fake.scale)

        class SCContentFilter:
            @classmethod
            def alloc(cls):
                return cls()

            def initWithDisplay_excludingApplications_exceptingWindows_(self, display, apps, windows):
                self.display, self.excluded, self.window = display, list(apps), None
                return self

            def initWithDesktopIndependentWindow_(self, window):
                self.display, self.excluded, self.window = None, [], window
                return self

        class SCStreamConfiguration:
            @classmethod
            def alloc(cls):
                return cls()

            def init(self):
                self.set = {}
                return self

            def __getattr__(self, name):
                if name.startswith("set") and name.endswith("_"):
                    return lambda v: self.set.__setitem__(name[3:-1], v)
                raise AttributeError(name)

        class SCScreenshotManager:
            @staticmethod
            def captureImageWithFilter_configuration_completionHandler_(flt, cfg, handler):
                fake.shots.append((flt, cfg.set))
                if fake.error:
                    return later(handler, None, SimpleNamespace(localizedDescription=lambda: fake.error))
                img = Image.new("RGB", (int(cfg.set["Width"]), int(cfg.set["Height"])), (12, 34, 56))
                cg, data = _make_cg(img)
                fake._keep.append(data)
                later(handler, cg, None)

        self.SCShareableContent = SCShareableContent
        self.SCContentFilter = SCContentFilter
        self.SCStreamConfiguration = SCStreamConfiguration
        self.SCScreenshotManager = SCScreenshotManager


def _make_cg(img):
    return cgimage(img, "bgra")


@pytest.fixture
def fake_sck(monkeypatch):
    from neo.tools.computer import ax

    fake = FakeSCK()
    monkeypatch.setattr(sck, "_SC", fake)
    monkeypatch.setattr(ax, "is_neo", lambda pid, name="": pid == 999)
    # the window server's view now: every app the fake lists has a window on screen
    monkeypatch.setattr(sck, "_on_screen_owners", lambda: {(a.processID(), a.applicationName()) for a in fake.apps}, raising=False)
    return fake


def test_screencapturekit_is_off_unless_switched_on_and_installed(fake_sck, monkeypatch):
    assert not screen._sck_on()  # the default: the CLI until the self-test has checked it
    _settings(monkeypatch, sck_capture=True)
    assert screen._sck_on() and sck.available()
    monkeypatch.setattr(sck, "_SC", None)
    monkeypatch.setattr(sck, "_framework", lambda: (_ for _ in ()).throw(sck.Failed("no framework")))
    assert not sck.available() and not screen._sck_on()


def test_a_rect_is_captured_in_process_at_retina_with_neo_left_out(fake_sck):
    d = screen.Display(1, 0.0, 0.0, 1512.0, 982.0, 1, 2.0)
    img = sck.capture_rect(300, 200, 400, 250, d)
    assert img.size == (800, 500) and img.getpixel((0, 0)) == (12, 34, 56)
    flt, cfg = fake_sck.shots[-1]
    assert [a.processID() for a in flt.excluded] == [999]  # NEO's own windows never in the picture
    assert cfg["SourceRect"] == ((300, 200), (400, 250)) and (cfg["Width"], cfg["Height"]) == (800, 500)
    assert cfg["ShowsCursor"] is False and cfg["PixelFormat"] == 0x42475241 and cfg["ColorSpaceName"] == Q.kCGColorSpaceSRGB
    second = screen.Display(2, 1512.0, -200.0, 1920.0, 1080.0, 2, 1.0)
    fake_sck.displays.append(SimpleNamespace(displayID=lambda: 2, frame=lambda: _Rect(1512, -200, 1920, 1080)))
    sck.capture_rect(1600, -100, 200, 100, second)
    _, cfg = fake_sck.shots[-1]
    assert cfg["SourceRect"] == ((88, 100), (200, 100)) and (cfg["Width"], cfg["Height"]) == (200, 100)  # display-local
    sck.capture_rect(10, 10, 100, 60, d, scale=1.0)  # (a change sample: points are enough)
    assert (fake_sck.shots[-1][1]["Width"], fake_sck.shots[-1][1]["Height"]) == (100, 60)
    assert fake_sck.content_calls == 2  # reused, but fetched again once for the display it didn't list


def test_a_window_is_captured_in_process_without_its_shadow(fake_sck):
    img = sck.capture_window(77)
    flt, cfg = fake_sck.shots[-1]
    assert img.size == (1200, 800) and flt.window.windowID() == 77 and cfg["IgnoreShadowsSingleWindow"] is True
    assert "SourceRect" not in cfg and cfg["ShowsCursor"] is False
    fake_sck.windows.append(SimpleNamespace(windowID=lambda: 88, frame=lambda: _Rect(0, 0, 100, 50)))
    assert sck.capture_window(88).size == (200, 100) and fake_sck.content_calls == 2  # new: looked again once
    with pytest.raises(sck.Failed):
        sck.capture_window(99)


def test_no_answer_is_a_failure_and_two_switch_it_off_for_the_run(fake_sck):
    d = screen.Display(1, 0.0, 0.0, 1512.0, 982.0, 1, 2.0)
    sck.capture_rect(0, 0, 10, 10, d)
    fake_sck.answer = False
    t = time.perf_counter()
    with pytest.raises(sck.Failed, match="no answer"):
        sck.capture_rect(0, 0, 10, 10, d, timeout=0.05)
    assert time.perf_counter() - t < 1.0 and sck.enabled()
    with pytest.raises(sck.Failed):
        sck.capture_rect(0, 0, 10, 10, d, timeout=0.05)
    assert not sck.enabled() and "no answer" in sck._off
    with pytest.raises(sck.Failed):
        sck.capture_rect(0, 0, 10, 10, d)  # (off: not even tried)


def test_an_error_from_screencapturekit_is_a_failure_that_keeps_it_on(fake_sck):
    fake_sck.error = "The user declined TCCs for application, window, display capture"
    d = screen.Display(1, 0.0, 0.0, 1512.0, 982.0, 1, 2.0)
    with pytest.raises(sck.Failed, match="declined"):
        sck.capture_rect(0, 0, 10, 10, d)
    assert sck.enabled()
    with pytest.raises(sck.Failed, match="isn't on the display"):
        sck.capture_rect(5000, 0, 10, 10, d)


def test_screen_uses_screencapturekit_when_on_and_the_cli_when_it_fails(fake_sck, cli, monkeypatch):
    _settings(monkeypatch, sck_capture=True)
    c = screen.shot()
    assert c.frame.backend == "sck" and cli.runs == [] and (c.w, c.h) == (1512, 982)
    assert fake_sck.shots[-1][1]["SourceRect"] == ((0, 0), (1512, 982))
    z = screen.zoom(700, 500)
    assert z.frame.backend == "sck" and (z.w, z.h) == (768, 480)
    assert len(_REAL_SAMPLE([(700.0, 500.0)])) == 1 and fake_sck.shots[-1][1]["Width"] == 240  # one rect, Retina
    fake_sck.error = "stopped"
    c = screen.shot()
    assert c.frame.backend == "cli" and cli.runs == [[]]  # the CLI took over, same picture for the model
    assert (c.w, c.h) == (1512, 982)


def test_a_before_and_after_pair_of_change_samples_never_mixes_capture_paths(fake_sck, cli, monkeypatch):
    """The in-process capture renders colours in sRGB, the CLI's file in the display's own space: a
    sample of each, of an unchanged screen, could look like a change. So a pair is one path or none."""
    _settings(monkeypatch, sck_capture=True)
    before = _REAL_SAMPLE([(700.0, 500.0)])
    assert before[0].info["via"] == "sck" and cli.runs == []
    fake_sck.error = "stopped"
    assert _REAL_SAMPLE([(700.0, 500.0)], like=before) is None and cli.runs == []  # unknown, never the CLI's
    assert _REAL_SAMPLE([(700.0, 500.0)]) is None and cli.runs == []  # (no fallback for samples at all)
    fake_sck.error = None
    cli_before = screen.samples_from(screen.Frame(synthetic_screen("ui"), 0.0, 0.0, 0.5), [(700.0, 500.0)])
    assert cli_before[0].info["via"] == "cli"
    after = _REAL_SAMPLE([(700.0, 500.0)], like=cli_before)  # the pair began on the CLI: it ends there
    assert cli.runs == [["-R", "640,440,120,120"]] and after[0].info["via"] == "cli"
    shots = len(fake_sck.shots)
    look, samples = screen.after_look((600.0, 400.0, 200.0, 200.0), [(700.0, 500.0)], like=cli_before)
    assert len(fake_sck.shots) == shots + 1 and look.frame.backend == "sck" and samples is None
    look, samples = screen.after_look((600.0, 400.0, 200.0, 200.0), [(700.0, 500.0)], like=before)
    assert samples is not None and samples[0].info["via"] == "sck"
    fresh = screen.last_frame()
    assert _REAL_SAMPLE([(700.0, 500.0)], frame=fresh, like=cli_before) is None
    assert _REAL_SAMPLE([(700.0, 500.0)], frame=fresh, like=before)[0].info["via"] == "sck"


def test_an_in_process_capture_is_a_guarded_effect(fake_sck, monkeypatch):
    kinds: list[str] = []
    real_guard = effects.guard
    monkeypatch.setattr(effects, "guard", lambda kind, what, via=None: (kinds.append(kind), real_guard(kind, what, via=via)))
    sck.capture_rect(0, 0, 10, 10, screen.Display(1, 0.0, 0.0, 1512.0, 982.0, 1, 2.0))
    assert kinds == ["capture", "capture"]  # (the window list, the capture: a fake framework goes through)
    # the real framework (a PyObjC function where the capture belongs) is refused while effects are off,
    # and the refusal is never taken for a failure to fall back from
    fake_sck.SCScreenshotManager.captureImageWithFilter_configuration_completionHandler_ = Q.CGMainDisplayID
    with pytest.raises(effects.EffectRefused):
        sck.capture_rect(0, 0, 10, 10, screen.Display(1, 0.0, 0.0, 1512.0, 982.0, 1, 2.0))


# ---- windows on the screen, frames that map, one deadline -------------------------------------------


def test_a_window_shot_says_whose_it_is_and_where_other_windows_covered_it(cli, monkeypatch):
    """A window shot shows the window's own pixels, even where another window (another app's, or
    NEO's overlay) is over it on the screen; a click there would land on that other window. So the
    shot carries its window, its owner and those covered parts, and no spot in them is clear."""
    cli.picture = lambda extra: synthetic_screen("ui", (2400, 1600))
    monkeypatch.setattr(screen, "_window_bounds", lambda wid: (100.0, 50.0, 1200.0, 800.0))
    above = [(1000.0, 700.0, 400.0, 300.0), (1400.0, 0.0, 100.0, 100.0)]  # (the overlay; one off the window)
    monkeypatch.setattr(screen, "_window_stack", lambda wid: (4321, above))
    c = screen.window_shot(77)
    assert c.clickable and (c.window_id, c.pid) == (77, 4321)
    assert c.covered == ((1000.0, 700.0, 300.0, 150.0),)  # (cut to the window)
    assert c.clear_at(200.0, 200.0) and not c.clear_at(1100.0, 800.0)
    assert not c.clear_at(50.0, 60.0)  # (outside the picture: nothing to say about it)
    assert c.frame.covered == c.covered and not c.frame.shows(1100.0, 800.0) and c.frame.shows(200.0, 200.0)
    assert screen.covered_at(c.covered, 1100.0, 800.0) and not screen.covered_at(c.covered, 200.0, 200.0)
    page = screen.window_shot(77, (100.0, 130.0, 600.0, 400.0))  # a part the overlay isn't over
    assert page.covered == () and page.clear_at(300.0, 300.0)
    # the window server can't say what is over it (or it isn't on screen): for looking only
    monkeypatch.setattr(screen, "_window_stack", lambda wid: None)
    c = screen.window_shot(77)
    assert not c.clickable and not c.clear_at(200.0, 200.0)
    whole = screen.shot()  # a picture of the screen shows what is on it everywhere
    assert whole.window_id == 0 and whole.covered == () and whole.clear_at(1100.0, 800.0)


def test_the_window_stack_is_read_from_the_window_server(monkeypatch):
    info = [
        {"kCGWindowNumber": 9, "kCGWindowOwnerPID": 999, "kCGWindowAlpha": 1.0, "kCGWindowIsOnscreen": True,
         "kCGWindowBounds": {"X": 1000, "Y": 700, "Width": 400, "Height": 300}},
        {"kCGWindowNumber": 8, "kCGWindowOwnerPID": 5, "kCGWindowAlpha": 0.0, "kCGWindowIsOnscreen": True,
         "kCGWindowBounds": {"X": 0, "Y": 0, "Width": 1512, "Height": 982}},  # (invisible: covers nothing)
        {"kCGWindowNumber": 77, "kCGWindowOwnerPID": 4321, "kCGWindowAlpha": 1.0, "kCGWindowIsOnscreen": True,
         "kCGWindowBounds": {"X": 100, "Y": 50, "Width": 1200, "Height": 800}},
    ]
    asked: list = []
    monkeypatch.setattr(screen.Q, "CGWindowListCopyWindowInfo", lambda opts, wid: (asked.append((opts, wid)), info)[1])
    monkeypatch.setattr(screen, "_window_stack", _REAL_STACK)
    assert screen._window_stack(77) == (4321, [(1000.0, 700.0, 400.0, 300.0)])
    assert asked == [(screen.Q.kCGWindowListOptionOnScreenAboveWindow | screen.Q.kCGWindowListOptionIncludingWindow, 77)]
    info[2]["kCGWindowIsOnscreen"] = False  # minimized, or on another Space
    assert screen._window_stack(77) is None
    assert screen._window_stack(78) is None  # (not listed at all)


def test_only_a_frame_whose_place_on_the_screen_is_known_is_kept(cli, monkeypatch):
    """last_frame() is what the board finder, OCR and change checks map through: a window picture
    with no mapping (bounds unknown, moved during the capture, not on screen) is never it."""
    screen.shot()
    assert screen.last_frame() is not None
    cli.picture = lambda extra: Image.new("RGB", (2400, 1600), "white")
    monkeypatch.setattr(screen, "_window_stack", lambda wid: (4321, []))
    c = screen.window_shot(77)  # (no bounds)
    assert not c.clickable and not c.frame.mapped and screen.last_frame() is None
    moving = iter([(100.0, 50.0, 1200.0, 800.0), (400.0, 300.0, 1200.0, 800.0)])
    monkeypatch.setattr(screen, "_window_bounds", lambda wid: next(moving))
    c = screen.window_shot(77)
    assert not c.frame.mapped and screen.last_frame() is None
    monkeypatch.setattr(screen, "_window_bounds", lambda wid: (100.0, 50.0, 1200.0, 800.0))
    c = screen.window_shot(77)
    kept = screen.last_frame()
    assert kept is c.frame and kept.mapped and kept.window_id == 77 and kept.pid == 4321


def test_change_samples_are_never_cut_from_a_window_picture(cli, monkeypatch):
    """A window's own pixels aren't the screen (another window may be over them): a sample cut from
    them, compared with one of the screen, could report a change that never happened."""
    win = screen.Frame(Image.new("RGB", (2400, 1600), "white"), 100.0, 50.0, 0.5, window_id=77)
    with pytest.raises(ValueError):
        screen.samples_from(win, [(300.0, 300.0)])
    got = _REAL_SAMPLE([(300.0, 300.0)], frame=win)  # not from the window: a capture of the screen
    assert cli.runs == [["-R", "240,240,120,120"]] and got is not None
    guess = screen.Frame(Image.new("RGB", (3024, 1964), "white"), 0.0, 0.0, 0.5, mapped=False)
    with pytest.raises(ValueError):
        screen.samples_from(guess, [(300.0, 300.0)])
    _REAL_SAMPLE([(300.0, 300.0)], frame=guess)
    assert len(cli.runs) == 2


def test_a_region_past_its_displays_edge_maps_the_same_either_way(fake_sck, cli, monkeypatch):
    """Only what is on the display is ever in the picture: the region is cut to it before either
    path captures, so the frame maps from what was really captured."""
    d = screen.Display(1, 0.0, 0.0, 1512.0, 982.0, 1, 2.0)
    monkeypatch.setattr(screen, "displays", lambda: [d])
    f = screen.grab((-100, 100, 400, 200))  # 100 pt of it off the left edge (the CLI)
    assert cli.runs == [["-R", "0,100,300,200"]] and (f.x0, f.y0, f.ppp) == (0.0, 100.0, 0.5)
    _settings(monkeypatch, sck_capture=True)
    f = screen.grab((-100, 100, 400, 200))  # (in process)
    assert f.backend == "sck" and f.image.size == (600, 400) and (f.x0, f.y0, f.ppp) == (0.0, 100.0, 0.5)
    assert fake_sck.shots[-1][1]["SourceRect"] == ((0, 100), (300, 200))
    assert f.to_screen(600, 400) == (300.0, 300.0)
    f = screen.grab((1400, 900, 300, 300))  # past the bottom right corner
    assert f.rect == (1400.0, 900.0, 112.0, 82.0)


def test_a_frame_is_stamped_when_its_capture_began(cli, monkeypatch):
    def slow(extra):
        time.sleep(0.2)  # screencapture and the decode
        return Image.new("RGB", (3024, 1964))

    cli.picture = slow
    t0 = time.monotonic()
    f = screen.grab()
    assert f.t - t0 < 0.05 and f.age() >= 0.2
    monkeypatch.setattr(screen, "_window_bounds", lambda wid: (0.0, 0.0, 1512.0, 982.0))
    monkeypatch.setattr(screen, "_window_stack", lambda wid: (4321, []))
    t0 = time.monotonic()
    assert screen.window_shot(77).frame.t - t0 < 0.05


# ---- the CLI's format ----------------------------------------------------------------------------


def test_a_failed_window_or_region_capture_runs_once_and_keeps_the_tiff(monkeypatch):
    """A window that closed or a region off the screen fails as PNG too: one screencapture run, and
    the fast format stays for every later capture."""
    runs: list[list[str]] = []

    def run(cmd, check, timeout):
        runs.append(cmd)
        if "-l" in cmd or "-R" in cmd:
            raise subprocess.CalledProcessError(1, cmd)
        Image.new("RGB", (64, 40)).save(cmd[-1], format="TIFF" if cmd[3] == "tiff" else "PNG")

    monkeypatch.setattr(screen.subprocess, "run", run)
    for extra in (["-o", "-l", "4242"], ["-R", "5000,5000,10,10"]):
        with pytest.raises(subprocess.CalledProcessError):
            screen._grab(extra)
    assert [c[3] for c in runs] == ["tiff", "tiff"] and screen._cli_format == "tiff"
    runs.clear()
    screen._grab([])
    assert [c[3] for c in runs] == ["tiff"]


def test_a_whole_display_failure_switches_to_png_only_when_png_works(monkeypatch):
    runs: list[list[str]] = []
    works = {"png": False}

    def run(cmd, check, timeout):
        runs.append(cmd)
        if cmd[3] == "tiff" or not works["png"]:
            raise subprocess.CalledProcessError(1, cmd)
        Image.new("RGB", (64, 40)).save(cmd[-1], format="PNG")

    monkeypatch.setattr(screen.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        screen._grab([])  # (both fail: nothing to do with the TIFF)
    assert [c[3] for c in runs] == ["tiff", "png"] and screen._cli_format == "tiff"
    works["png"] = True
    assert screen._grab([]).size == (64, 40) and screen._cli_format == "png"  # PNG works where TIFF doesn't


def test_a_tiff_that_cant_be_read_switches_to_png(monkeypatch):
    runs: list[list[str]] = []

    def run(cmd, check, timeout):
        runs.append(cmd)
        if cmd[3] == "tiff":
            with open(cmd[-1], "wb") as f:
                f.write(b"not a picture")
        else:
            Image.new("RGB", (64, 40), (10, 20, 30)).save(cmd[-1], format="PNG")

    monkeypatch.setattr(screen.subprocess, "run", run)
    img = screen._grab(["-R", "0,0,32,20"])
    assert [c[3] for c in runs] == ["tiff", "png"] and img.getpixel((0, 0)) == (10, 20, 30)
    assert screen._cli_format == "png" and not any(os.path.exists(c[-1]) for c in runs)


# ---- ScreenCaptureKit: switching off, one deadline, NEO left out -----------------------------------


def test_a_capture_that_never_answers_is_switched_off_even_when_the_list_answers(fake_sck, monkeypatch):
    """Screenshots more than CONTENT_TTL_S apart fetch the window list again each time; its answer
    is not an answered capture."""
    monkeypatch.setattr(sck, "CONTENT_TTL_S", 0.0)
    fake_sck.answer = False
    d = screen.Display(1, 0.0, 0.0, 1512.0, 982.0, 1, 2.0)
    for _ in range(2):
        with pytest.raises(sck.Failed, match="no answer"):
            sck.capture_rect(0, 0, 10, 10, d, timeout=0.05)
    assert not sck.enabled() and "no answer" in sck._off and fake_sck.content_calls == 2


def test_an_answered_capture_starts_the_count_again(fake_sck):
    d = screen.Display(1, 0.0, 0.0, 1512.0, 982.0, 1, 2.0)
    fake_sck.answer = False
    with pytest.raises(sck.Failed):
        sck.capture_rect(0, 0, 10, 10, d, timeout=0.05)
    fake_sck.answer = True
    sck.capture_rect(0, 0, 10, 10, d)
    fake_sck.answer = False
    with pytest.raises(sck.Failed):
        sck.capture_rect(0, 0, 10, 10, d, timeout=0.05)
    assert sck.enabled()


def test_one_capture_waits_one_timeout_in_all(fake_sck):
    """The lock, the window list, the list again for a new window and the capture share one
    deadline: a slow list and a capture that never answers cost the timeout once, then the CLI."""
    timeout, lists = 0.2, [0]

    def slow_list(ex, on, handler):  # answers just inside the timeout; the window is on the second list
        lists[0] += 1
        wins = [SimpleNamespace(windowID=lambda: 88, frame=lambda: _Rect(0, 0, 100, 50))] if lists[0] > 1 else []
        content = SimpleNamespace(displays=lambda: fake_sck.displays, applications=lambda: [], windows=lambda: wins)
        threading.Timer(0.9 * timeout, handler, args=(content, None)).start()

    fake_sck.SCShareableContent.getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler_ = staticmethod(slow_list)
    fake_sck.answer = False
    t = time.perf_counter()
    with pytest.raises(sck.Failed, match="no answer"):
        sck.capture_window(88, timeout=timeout)
    assert time.perf_counter() - t < 1.5 * timeout


def test_a_neo_window_newer_than_the_cached_list_is_still_left_out(fake_sck):
    """NEO's overlay (re)started after the list was fetched: the list is fetched again before the
    filter is made, so the new NEO process is left out too."""
    fake_sck.apps = [SimpleNamespace(processID=lambda: 42, applicationName=lambda: "Some App")]
    d = screen.Display(1, 0.0, 0.0, 1512.0, 982.0, 1, 2.0)
    sck.capture_rect(0, 0, 10, 10, d)
    sck.capture_rect(0, 0, 10, 10, d)
    assert fake_sck.content_calls == 1  # (nothing new: the list is reused)
    fake_sck.apps.append(SimpleNamespace(processID=lambda: 999, applicationName=lambda: "NEO"))
    sck.capture_rect(0, 0, 10, 10, d)
    flt, _ = fake_sck.shots[-1]
    assert [a.processID() for a in flt.excluded] == [999] and fake_sck.content_calls == 2
    sck.capture_rect(0, 0, 10, 10, d)
    assert fake_sck.content_calls == 2  # (and reused again)


# ---- the upload budget, for every kind -----------------------------------------------------------


def _jpeg_is_q85_444(data: bytes) -> bool:
    from PIL import JpegImagePlugin

    img = Image.open(io.BytesIO(data))
    ref = io.BytesIO()
    Image.new("RGB", (16, 16)).save(ref, format="JPEG", quality=screen.JPEG_QUALITY, subsampling=0)
    return JpegImagePlugin.get_sampling(img) == 0 and img.quantization == Image.open(ref).quantization


@pytest.mark.parametrize("kind", ["ui", "mixed"])
def test_windows_and_regions_fit_the_upload_at_q85(kind):
    """Windows, regions and zooms keep Retina pixels only while they're about a whole display's
    worth; a picture still past the budget is made smaller, never blurrier (q85 4:4:4 stays)."""
    f = screen.Frame(synthetic_screen(kind), 0.0, 0.0, 0.5)
    for what, rect in (("window", (100.0, 60.0, 700.0, 700.0)), ("window", (100.0, 60.0, 780.0, 780.0)),
                       ("region", (300.0, 100.0, 760.0, 490.0)), ("zoom", (300.0, 200.0, 320.0, 200.0))):
        c = screen.render(f, what, rect=rect, fmt="jpeg")
        assert len(c.part.data) <= 400_000 and _jpeg_is_q85_444(c.part.data), (what, rect, len(c.part.data))
        assert c.w * c.h <= 1_600_000 and c.to_screen(c.w, c.h) == pytest.approx((rect[0] + rect[2], rect[1] + rect[3]), abs=1.0)


def test_a_picture_past_the_budget_is_scaled_down_and_still_maps():
    noise = Image.effect_noise((1512, 982), 90).convert("RGB")  # (the worst case for a JPEG)
    c = screen.render(screen.Frame(noise, 0.0, 0.0, 1.0), "full", fmt="jpeg")
    assert len(c.part.data) <= 400_000 and _jpeg_is_q85_444(c.part.data)
    assert c.w < 1512 and _decode(c.part).size == (c.w, c.h) and c.to_screen(c.w, 0)[0] == pytest.approx(1512.0, abs=1.5)


# ---- speed (hermetic: synthetic frames) --------------------------------------------------------------


def _med(fn, n: int = 7) -> float:
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)


def _old_post(png: bytes) -> bytes:
    """What screen.capture did after screencapture until now: PNG decode, LANCZOS, PNG optimize."""
    img = Image.open(io.BytesIO(png)).convert("RGB")
    img = img.resize((1512, 982), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _new_post(tiff: bytes) -> screen.Capture:
    """What it does now: TIFF decode (the CLI's uncompressed file), then render (reduce + JPEG)."""
    img = Image.open(io.BytesIO(tiff)).convert("RGB")
    return screen.render(screen.Frame(img, 0.0, 0.0, 0.5), "full", fmt="jpeg")


def _files(kind: str) -> tuple[bytes, bytes]:
    big = synthetic_screen(kind)
    png, tiff = io.BytesIO(), io.BytesIO()
    big.save(png, format="PNG")
    big.convert("RGBA").save(tiff, format="TIFF")  # (screencapture's TIFF carries alpha)
    return png.getvalue(), tiff.getvalue()


def test_post_processing_is_fast_and_the_upload_small():
    """Targets: ≤15 ms of Python after the capture (the old path: 165–390 ms) and ≤0.4 MB. The bound
    here has slack for a busy machine; the benchmark below prints the real numbers."""
    for kind in ("ui", "mixed", "board"):
        _, tiff = _files(kind)
        cap = _new_post(tiff)
        assert (cap.w, cap.h) == (1512, 982) and len(cap.part.data) <= 400_000, (kind, len(cap.part.data))
        assert _med(lambda tiff=tiff: _new_post(tiff), 5) <= 45.0
    png, tiff = _files("board")
    assert 3 * _med(lambda: _new_post(tiff), 3) < _med(lambda: _old_post(png), 1)


def bench() -> None:  # pragma: no cover - run by hand: python tests/test_eyes_capture.py
    print(f"{'screen':7} | {'before: PNG decode+LANCZOS+PNG optimize':40} | {'after: TIFF decode+reduce+JPEG q85':36} | render only")
    for kind in ("ui", "mixed", "board"):
        png, tiff = _files(kind)
        before = _med(lambda png=png: _old_post(png), 5)
        size_before = len(_old_post(png))
        after = _med(lambda tiff=tiff: _new_post(tiff), 15)
        cap = _new_post(tiff)
        frame = screen.Frame(Image.open(io.BytesIO(tiff)).convert("RGB"), 0.0, 0.0, 0.5)
        render = _med(lambda frame=frame: screen.render(frame, "full", fmt="jpeg"), 15)
        print(
            f"{kind:7} | {before:7.1f} ms, {size_before / 1e6:5.2f} MB{'':20} | {after:6.1f} ms, {len(cap.part.data) / 1e6:5.2f} MB"
            f"{'':15} | {render:5.1f} ms"
        )
    frame = screen.Frame(synthetic_screen("ui"), 0.0, 0.0, 0.5)
    rect = (300.0, 200.0, 320.0, 200.0)
    zoom = _med(lambda: screen.render(frame, "zoom", rect=rect, fmt="jpeg"), 15)
    after = _med(lambda: screen.render(frame, "after", rect=(100.0, 100.0, 800.0, 600.0), fmt="jpeg"), 15)
    samples = _med(lambda: screen.samples_from(frame, [(400.0, 300.0), (900.0, 500.0)]), 30)
    print(f"zoom 320×200 pt → 768×480: {zoom:.1f} ms · after-look 800×600 pt: {after:.1f} ms · 2 change samples: {samples:.2f} ms")
    _bench_cli_file()
    _bench_sck()


def _imageio_write(cg, uti: str, path: str) -> None:
    """What screencapture does with its picture: ImageIO writes it to the file (the CLI's own cost)."""
    from Foundation import NSURL

    dest = Q.CGImageDestinationCreateWithURL(NSURL.fileURLWithPath_(path), uti, 1, None)
    Q.CGImageDestinationAddImage(dest, cg, None)
    assert Q.CGImageDestinationFinalize(dest)


def _bench_cli_file() -> None:  # pragma: no cover
    """screencapture -t png vs -t tiff, hermetically: ImageIO's encode + the file + PIL's decode."""
    import tempfile

    tmp = tempfile.mkdtemp()
    print("CLI file (ImageIO encode + write + PIL decode to RGB), 3024×1964:")
    for kind in ("ui", "mixed", "board"):
        cg, _keep = cgimage(synthetic_screen(kind), "bgra")
        row = []
        for fmt, uti in (("png", "public.png"), ("tiff", "public.tiff")):
            path = os.path.join(tmp, f"shot.{fmt}")
            enc = _med(lambda cg=cg, uti=uti, path=path: _imageio_write(cg, uti, path), 5)
            dec = _med(lambda path=path: Image.open(path).convert("RGB"), 5)
            row.append(f"-t {fmt}: {enc:5.1f} + {dec:5.1f} = {enc + dec:6.1f} ms ({os.path.getsize(path) / 1e6:5.2f} MB)")
        print(f"  {kind:6} " + " | ".join(row))


def _bench_sck() -> None:  # pragma: no cover
    """The in-process path's Python work: a BGRA CGImage → RGB → a points-sized JPEG."""
    for kind in ("ui", "mixed"):
        cg, _keep = cgimage(synthetic_screen(kind), "bgra")
        conv = _med(lambda cg=cg: sck.to_pil(cg), 9)
        img = sck.to_pil(cg)
        total = _med(lambda cg=cg: screen.render(screen.Frame(sck.to_pil(cg), 0.0, 0.0, 0.5), "full", fmt="jpeg"), 9)
        small, _k = cgimage(img.reduce(2), "bgra")
        at_points = _med(lambda small=small: screen.render(screen.Frame(sck.to_pil(small), 0.0, 0.0, 1.0), "full", fmt="jpeg"), 9)
        print(
            f"ScreenCaptureKit {kind:5}: CGImage→RGB {conv:4.1f} ms; + reduce + JPEG {total:4.1f} ms at Retina, "
            f"{at_points:4.1f} ms when captured at points"
        )


if __name__ == "__main__":  # pragma: no cover
    os.environ.setdefault("NEO_NO_EFFECTS", "1")
    bench()
