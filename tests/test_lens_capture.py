"""Search Screen's pictures (neo2.core.lens.capture): the still comes from ScreenCaptureKit with NEO's
own windows left out — even with sck_capture off (that switch is the computer tools') — at the
display's own pixels, never through the CLI; without Screen Recording nothing is captured and macOS
is never asked; without ScreenCaptureKit it is live mode; areas are clamped and cut in points;
live mode grabs only the area. ScreenCaptureKit is tests/test_eyes_capture.py's fake."""

from __future__ import annotations

import pytest

from neo2.core.lens import capture
from neo2.core.tools.computer import screen, sck
from tests.test_eyes_capture import FakeSCK


@pytest.fixture
def fake_sck(monkeypatch):
    from neo2.core.tools.computer import ax

    fake = FakeSCK()
    monkeypatch.setattr(sck, "_SC", fake)
    monkeypatch.setattr(sck, "_content", None)
    monkeypatch.setattr(sck, "_misses", 0)
    monkeypatch.setattr(sck, "_off", "")
    monkeypatch.setattr(ax, "is_neo", lambda pid, name="": pid == 999)
    monkeypatch.setattr(sck, "_on_screen_owners", lambda: {(a.processID(), a.applicationName()) for a in fake.apps}, raising=False)
    monkeypatch.setattr(screen, "can_capture", lambda: True)
    return fake


MAIN = {"id": 1, "x": 0, "y": 0, "w": 1512, "h": 982, "scale": 2}


def test_the_shells_display_is_read_strictly():
    sp = capture.spot(MAIN)
    assert sp == capture.Spot(1, 0.0, 0.0, 1512.0, 982.0, 2.0) and sp.shown() == {"w": 1512.0, "h": 982.0, "scale": 2.0}
    for bad in (None, [], {**MAIN, "w": 0}, {**MAIN, "scale": 0}, {**MAIN, "scale": 99}, {**MAIN, "x": float("nan")},
                {**MAIN, "id": True}, {**MAIN, "id": -1}, {k: v for k, v in MAIN.items() if k != "h"}, {**MAIN, "w": "wide"}):  # fmt: skip
        assert capture.spot(bad) is None, bad


def test_the_still_is_screencapturekits_without_neo_even_with_sck_capture_off(fake_sck, monkeypatch):
    assert not screen._sck_on()  # (the computer tools' switch is off: the default)
    monkeypatch.setattr(screen, "_grab", lambda *a, **k: pytest.fail("never the screencapture CLI"))
    got = capture.freeze(capture.spot(MAIN))
    assert got.why == "" and got.frame is not None and got.frame.backend == "sck"
    assert got.frame.image.size == (3024, 1964) and got.frame.ppp == 0.5 and (got.frame.x0, got.frame.y0) == (0, 0)
    flt, cfg = fake_sck.shots[-1]
    assert [a.processID() for a in flt.excluded] == [999]  # NEO's own windows — the overlay — never in it
    assert cfg["SourceRect"] == ((0, 0), (1512, 982)) and cfg["ShowsCursor"] is False


def test_a_second_display_at_its_own_scale(fake_sck, monkeypatch):
    from types import SimpleNamespace

    from tests.test_eyes_capture import _Rect

    second = screen.Display(2, 1512.0, -200.0, 1920.0, 1080.0, 7, 1.0)
    monkeypatch.setattr(screen, "displays", lambda: [screen.Display(1, 0.0, 0.0, 1512.0, 982.0, 1, 2.0), second])
    fake_sck.displays.append(SimpleNamespace(displayID=lambda: 7, frame=lambda: _Rect(1512, -200, 1920, 1080)))
    got = capture.freeze(capture.spot({"id": 7, "x": 1512, "y": -200, "w": 1920, "h": 1080, "scale": 1}))
    assert got.frame.image.size == (1920, 1080) and got.frame.ppp == 1.0 and (got.frame.x0, got.frame.y0) == (1512, -200)
    # A display the core doesn't see where the shell says it is: no still (live mode), never a guess.
    assert capture.freeze(capture.spot({"id": 9, "x": 5000, "y": 0, "w": 800, "h": 600, "scale": 1})).why == "failed"


def test_without_screen_recording_nothing_is_captured_or_asked(fake_sck, monkeypatch):
    monkeypatch.setattr(screen, "can_capture", lambda: False)
    monkeypatch.setattr(screen, "_request_access", lambda: pytest.fail("never macOS's prompt from Search Screen"))
    got = capture.freeze(capture.spot(MAIN))
    assert (got.frame, got.why) == (None, "no_permission") and not fake_sck.shots
    with pytest.raises(PermissionError):
        capture.grab_live(capture.spot(MAIN), (10, 10, 100, 50))


def test_without_screencapturekit_it_is_live_mode(monkeypatch):
    monkeypatch.setattr(screen, "can_capture", lambda: True)
    monkeypatch.setattr(sck, "_SC", None)
    monkeypatch.setattr(sck, "_framework", lambda: (_ for _ in ()).throw(sck.Failed("no framework")))
    got = capture.freeze(capture.spot(MAIN))
    assert (got.frame, got.why) == (None, "no_sck")


def test_a_capture_that_fails_is_live_mode(fake_sck):
    fake_sck.error = "the display went to sleep"
    got = capture.freeze(capture.spot(MAIN))
    assert (got.frame, got.why) == (None, "failed")


def test_areas_are_clamped_to_the_display_and_cut_in_points(fake_sck):
    sp = capture.spot(MAIN)
    assert capture.clamp(sp, [100, 50, 300, 200]) == (100.0, 50.0, 300.0, 200.0)
    assert capture.clamp(sp, [1400, 900, 300, 300]) == (1400.0, 900.0, 112.0, 82.0)
    assert capture.clamp(sp, [400, 300, -100, -50]) == (300.0, 250.0, 100.0, 50.0)  # (dragged up and left)
    for bad in (None, [1, 2, 3], [0, 0, float("inf"), 5], ["a", 0, 1, 1], [True, 0, 10, 10], [5000, 5000, 10, 10]):
        assert capture.clamp(sp, bad) is None, bad
    frame = capture.freeze(sp).frame
    crop = capture.cut(frame, sp, (100.0, 50.0, 300.0, 200.0))
    assert crop.image.size == (600, 400) and (crop.x0, crop.y0, crop.ppp) == (100.0, 50.0, 0.5)


def test_live_mode_grabs_just_the_area(monkeypatch):
    monkeypatch.setattr(screen, "can_capture", lambda: True)
    asked = []
    monkeypatch.setattr(screen, "grab", lambda region=None, display=None: asked.append(region) or "frame")
    sp = capture.spot({"id": 1, "x": 0, "y": 0, "w": 1512, "h": 982, "scale": 2})
    assert capture.grab_live(sp, (10.5, 20.25, 100.0, 50.0)) == "frame"
    assert asked == [(10, 20, 101, 51)]  # (whole points that cover it, global)
