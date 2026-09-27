"""The SDK's tests: embedded agents only, built with fake brains.

Besides tests/conftest.py's fixtures (an isolated NEO_DATA_DIR, library mode and the live agents
reset after each test, no real embeddings, no dry runs), every test here gets its own $HOME, and
fails if anything created a ~/.neo in it: an embedded agent must never reach for NEO.app's data.
"""

from __future__ import annotations

import pytest

_KEYS = ("GEMINI_API_KEY", "GEMINI_API_KEYS", "GROQ_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)
    yield h
    assert not (h / ".neo").exists(), "an embedded agent created ~/.neo"


@pytest.fixture
def brain():
    from neo2.testing import ScriptedBrain

    return ScriptedBrain([])
