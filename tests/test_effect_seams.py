"""The Step-0 seams are in, so the stand-ins that waited for them are gone: one Effect type (the
registry's) wherever an input reports what it did, and ImagePart.detail set directly."""

from __future__ import annotations

from types import SimpleNamespace

from neo2.core.agent.registry import Effect, ToolOutput
from neo2.core.providers.base import ImagePart
from neo2.core.tools.computer import ocr, pointer, screen


def test_the_pointer_reports_the_registrys_own_effect():
    assert pointer.Effect is Effect  # (no second class to drift from it)
    out = pointer.with_effect(ToolOutput("Clicked"), pointer.effect_of(None, looked=True))
    assert type(out.effect) is Effect and out.effect == Effect(changed=None, looked=True)
    seen = SimpleNamespace(changed=True, scene_changed=True, sheet="Save?")
    got = ocr._with_effect(ToolOutput("Clicked the words"), seen)
    assert type(got.effect) is Effect and got.effect == Effect(changed=True, scene_changed=True, sheet="Save?")


def test_a_pictures_detail_is_the_image_parts_own_field():
    assert screen._part(b"x", "image/jpeg", "low") == ImagePart(b"x", "image/jpeg", detail="low")
    assert pointer.detail(ImagePart(b"x"), "medium") == ImagePart(b"x", detail="medium")
