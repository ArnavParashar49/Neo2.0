"""GEMINI_API_KEYS on its own is a Gemini key (it used to leave Gemini unavailable unless
GEMINI_API_KEY was set too). Keys are fakes; no client here is ever asked anything."""

from __future__ import annotations

import pytest

pytest.importorskip("google.genai")

from neo2.core import config  # noqa: E402
from neo2.core import providers as prov  # noqa: E402
from neo2.core.providers import catalog  # noqa: E402
from neo2.core.providers.base import Unavailable  # noqa: E402
from neo2.core.providers.gemini import GeminiProvider  # noqa: E402

A, B, C = "AIzaTEST-a-0000", "AIzaTEST-b-1111", "AIzaTEST-c-2222"


def _env(monkeypatch, **env: str) -> None:
    for k in ("GEMINI_API_KEY", "GEMINI_API_KEYS", "GROQ_API_KEY", "ANTHROPIC_API_KEY", "NEO_BRAIN", "NEO_FAST_BRAIN"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(prov, "_cache", {})
    monkeypatch.setattr(prov, "_demoted", {})


def test_extra_keys_alone_build_gemini(monkeypatch):
    _env(monkeypatch, GEMINI_API_KEYS=f" {A} ,{B},,{A}")
    g = GeminiProvider()
    assert len(g._clients) == 2 and len(g._key_ids) == 2
    assert all(k not in "".join(g._key_ids) for k in (A, B))  # tracked by a hash, never the key
    assert catalog.configured("gemini") and catalog.any_key()
    assert [p.name for p in prov.brain("agent")._providers()][0] == "gemini"


def test_the_main_key_still_comes_first(monkeypatch):
    _env(monkeypatch, GEMINI_API_KEY=C, GEMINI_API_KEYS=f"{A},{C}")
    assert config.settings().gemini_keys == [C, A]
    assert len(GeminiProvider()._clients) == 2


def test_no_key_at_all_is_still_unavailable(monkeypatch):
    _env(monkeypatch)
    with pytest.raises(Unavailable, match="GEMINI_API_KEY is not set"):
        GeminiProvider()
    _env(monkeypatch, GEMINI_API_KEYS=" , ")
    with pytest.raises(Unavailable):
        GeminiProvider()
