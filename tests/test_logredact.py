"""No loaded API key reaches NEO's log whole: the core's stdout and stderr (core.log) go through
neo2.core.logredact, and so do the bus's "error" messages and the health list; NEO's memory masks the
loaded keys and the xAI and OpenRouter shapes too. Every key here is an obvious fake."""

from __future__ import annotations

import io
import logging
import sys

import pytest

from neo2.core import config, health, logredact
from neo2.core.memory import redact as memory_redact

ANTHROPIC = "sk-ant-TEST-0000-abcdefghijklmnop-x9Qa"
MISTRAL = "TESTmistral0000abcdefghijklmnopqrst"  # (no shape of its own: only the exact key is known)
EXTRA = "AIzaTEST-extra-0000-abcdefghijklmnopqr"


@pytest.fixture
def loaded(monkeypatch):
    for k in ("GEMINI_API_KEY", "GEMINI_API_KEYS", "ANTHROPIC_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", ANTHROPIC)
    monkeypatch.setenv("GEMINI_API_KEYS", EXTRA)
    monkeypatch.setenv("GROQ_API_KEY", "short")  # (under 8 characters: never looked for)
    monkeypatch.setattr(config, "_settings", None)
    s = config.settings()
    s.custom_keys = {"custom:mistral": MISTRAL}
    monkeypatch.setattr(logredact, "_remembered", set())
    return s


def test_every_loaded_key_is_masked_to_its_last_4(loaded):
    text = f"401: invalid x-api-key {ANTHROPIC}; also {EXTRA} and {MISTRAL}; a short word stays"
    out = logredact.redact(text)
    assert ANTHROPIC not in out and EXTRA not in out and MISTRAL not in out
    assert "••••x9Qa" in out and "••••" + MISTRAL[-4:] in out
    assert "short" in out  # (a key under 8 characters is never searched for: it would hit ordinary words)
    assert logredact.redact("nothing here") == "nothing here"


def test_a_key_handed_in_to_test_is_masked_before_it_is_saved(loaded):
    new = "sk-proj-TEST-0000-handed-in-zz99"
    assert new in logredact.redact(f"error {new}")
    logredact.remember(new, "tiny")
    assert logredact.redact(f"error {new}") == "error ••••zz99"
    assert "tiny" not in logredact._remembered


def test_nothing_is_built_or_read_to_find_the_keys(monkeypatch):
    """Before the settings exist (`python -m neo2 --help`), the redactor reads nothing to find keys."""
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(logredact, "_remembered", set())
    assert logredact.redact("hello") == "hello"
    assert config._settings is None


def test_the_stream_masks_a_key_split_over_two_writes(loaded):
    sink = io.StringIO()
    s = logredact.RedactingStream(sink)
    s.write("[brain] claude failed (401: " + ANTHROPIC[:10])
    assert sink.getvalue() == ""  # (held until the line ends)
    s.write(ANTHROPIC[10:] + ")\n")
    s.write("no newline yet")
    assert ANTHROPIC not in sink.getvalue() and "••••x9Qa" in sink.getvalue()
    assert "no newline yet" not in sink.getvalue()
    s.flush()
    assert sink.getvalue().endswith("no newline yet")
    s.write("a progress bar\r")  # (a carriage return goes out at once)
    assert sink.getvalue().endswith("a progress bar\r")


def test_install_wraps_stdout_and_stderr_once_and_neos_log_handler_follows(loaded, monkeypatch):
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    lg = logging.getLogger("neo")
    h = logging.StreamHandler(out)
    lg.addHandler(h)
    try:
        assert logredact.install() is True
        assert logredact.install() is False  # (once)
        assert isinstance(sys.stdout, logredact.RedactingStream) and sys.stdout.inner is out
        assert h.stream is sys.stdout
        print(f"[voice] unavailable: bad key {ANTHROPIC}")
        print(f"traceback … {MISTRAL}", file=sys.stderr)
        lg.warning(f"[brain] claude failed ({ANTHROPIC})")
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        lg.removeHandler(h)
    assert ANTHROPIC not in out.getvalue() and MISTRAL not in err.getvalue()
    assert out.getvalue().count("••••x9Qa") == 2 and "••••" + MISTRAL[-4:] in err.getvalue()


def test_the_health_list_never_carries_a_key(loaded, monkeypatch):
    monkeypatch.setattr(health, "_live", {})
    health.report("voice", f"Voice is off (Live refused {ANTHROPIC}). You can type; NEO keeps trying.")
    assert ANTHROPIC not in " ".join(health.problems()) and "••••x9Qa" in health._live["voice"]


def test_memory_masks_the_loaded_keys_and_more_shapes(loaded):
    for text in (
        f"my mistral key is fine: {MISTRAL} works",
        "the xai one: xai-TESTabcdefghijklmnopqrstuvwxyz0123",
        "openrouter sk-or-v1-TEST0000abcdefghijklmnopqrstuvwxyz",
    ):
        cleaned = memory_redact.clean(text)
        assert memory_redact.MASK in cleaned, text
        assert MISTRAL not in cleaned and "xai-TEST" not in cleaned and "sk-or-v1-TEST" not in cleaned
    assert memory_redact.clean("the wifi password is on the fridge") == "the wifi password is on the fridge"
