"""Search Screen's words (neo2.core.agent.lens_words) and its settings: which sentences open it, the
question "search my screen for X" carries, what never opens it, the words that close it or hand
it to the chat, and NEO_LENS_* read forgivingly (a typo never stops NEO, an embedded agent has none)."""

from __future__ import annotations

import pytest

from neo2.core import config
from neo2.core.agent import lens_words

OPENS = [
    "search my screen",
    "Search my screen.",
    "search the screen",
    "search on my screen",
    "search in my screen",
    "search what's on my screen",
    "search what is on the screen",
    "search this on my screen",
    "search something on my screen",
    "screen search",
    "search screen",
    "circle to search",
    "open search screen",
    "start circle to search",
    "Neo, search my screen please",
    "hey neo can you search my screen",
    "could you please search the screen for me",
    "I want to search my screen",
    "let me circle something on my screen",
    "screen pe search karo",
    "screen par search kar do",
    "meri screen pe kuch search karo",
    "screen search karo",
    "स्क्रीन पर सर्च करो",
    "स्क्रीन सर्च करो",
]
NEVER = [
    "what's on my screen",
    "what is on my screen",
    "read my screen",
    "look at my screen",
    "search the web for screen protectors",
    "search for screen protectors",
    "screen recording settings",
    "search shoes on google",
    "search cats",
    "remind me to search my screen later",
    "take a screenshot",
    "my screen is too dim",
    "",
]


@pytest.mark.parametrize("said", OPENS)
def test_these_words_open_search_screen(said):
    assert lens_words.wants_lens(said) == (True, ""), said


@pytest.mark.parametrize("said", NEVER)
def test_these_words_never_do(said):
    assert lens_words.wants_lens(said) == (False, ""), said


def test_search_my_screen_for_x_carries_the_question_as_said():
    assert lens_words.wants_lens("search my screen for the Price of this") == (True, "the Price of this")
    assert lens_words.wants_lens("Neo, search the screen for what these shoes cost, please?") == (
        True,
        "what these shoes cost",
    )
    assert lens_words.wants_lens("search my screen for") == (False, "")  # (nothing after "for")
    long = "search my screen for " + "x" * 400
    assert lens_words.wants_lens(long) == (False, "")  # a sentence that long isn't a command


@pytest.mark.parametrize(
    "said,want",
    [
        ("close", "close"),
        ("never mind", "close"),
        ("cancel that", "close"),
        ("Neo, close it", "close"),
        ("band karo", "close"),
        ("बंद करो", "close"),
        ("continue in chat", "continue"),
        ("open it in NEO", "continue"),
        ("take this to the chat", "continue"),
        ("what brand is this", None),
        ("how much does it cost", None),
        ("close the door in the picture and tell me what it is", None),
    ],
)
def test_while_open_the_control_words(said, want):
    assert lens_words.control(said) == want


def _settings(monkeypatch, **env):
    for k in ("NEO_LENS", "NEO_LENS_HOLD_MS", "NEO_LENS_SEARCH", "NEO_LENS_ASK"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(config, "_settings", None)
    return config.settings()


def test_the_settings_and_their_defaults(monkeypatch):
    s = _settings(monkeypatch)
    assert (s.lens, s.lens_hold_ms, s.lens_search, s.lens_ask) == (True, 1000, "google", True)
    s = _settings(monkeypatch, NEO_LENS="off", NEO_LENS_HOLD_MS="1500", NEO_LENS_SEARCH="Bing", NEO_LENS_ASK="off")
    assert (s.lens, s.lens_hold_ms, s.lens_search, s.lens_ask) == (False, 1500, "bing", False)


@pytest.mark.parametrize(
    "raw,want", [("100", 600), ("9999", 3000), ("600", 600), ("800ms", 800), ("soon", 1000), ("", 1000), ("1.2e3", 1200)]
)
def test_the_hold_is_clamped_and_a_typo_is_the_default(monkeypatch, raw, want):
    assert _settings(monkeypatch, NEO_LENS_HOLD_MS=raw).lens_hold_ms == want


@pytest.mark.parametrize(
    "raw,want",
    [("google", "google"), ("", "google"), ("on", "google"), ("off", "off"), ("none", "off"), ("duckduckgo", "duckduckgo"),
     ("kagi.com", "kagi.com"), ("<script>", "google"), ("x" * 41, "google")],
)  # fmt: skip
def test_the_search_provider_is_a_site_word_or_off(monkeypatch, raw, want):
    assert _settings(monkeypatch, NEO_LENS_SEARCH=raw).lens_search == want


def test_a_typo_never_turns_search_screen_off_or_stops_neo(monkeypatch):
    s = _settings(monkeypatch, NEO_LENS="maybe", NEO_LENS_ASK="sure")
    assert s.lens is True and s.lens_ask is True
    for off in ("0", "false", "no", "OFF", "never"):
        assert _settings(monkeypatch, NEO_LENS=off).lens is False


def test_an_embedded_agent_has_no_search_screen(tmp_path):
    lib = config.LibrarySettings(data_dir=tmp_path / "agent")
    assert lib.lens is False
