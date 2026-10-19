"""The health list knows NEO's brains: "no AI model yet" instead of "needs a free Gemini key", a key
set that no role uses, a refused key said by where it came from, a brain that can't see with no Eyes
— in words that fit NEO.app's window and the notch alike. And the `brains` summary every client
hears: labels and booleans, never a key or its last 4. Every key is an obvious fake; nothing here
asks the network (what answers on this Mac is faked)."""

from __future__ import annotations

import asyncio
import json

import pytest

from neo2.core import brains, config, health
from neo2.core import providers as prov

ANTHROPIC = "sk-ant-TEST-0000-abcdefghijklmnop"
GEMINI = "AIzaTEST-gemini-0000-abcdefghijklmn"
GROQ = "gsk_TEST-groq-0000-abcdefghijklmnop"
DEEPSEEK = {"id": "custom:deepseek", "label": "DeepSeek", "template": "deepseek", "base_url": "https://api.deepseek.com",
            "model": "deepseek-v4-pro", "context": 131072, "env": "DEEPSEEK_API_KEY"}  # fmt: skip
OLLAMA = {"id": "custom:ollama", "label": "Ollama", "template": "ollama", "base_url": "http://localhost:11434/v1",
          "model": "qwen3:14b"}  # fmt: skip
_VARS = (
    "GEMINI_API_KEY", "GEMINI_API_KEYS", "GROQ_API_KEY", "ANTHROPIC_API_KEY", "CEREBRAS_API_KEY", "OPENAI_API_KEY",
    "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "NEO_BRAIN", "NEO_FAST_BRAIN", "NEO_OFFLINE_BRAIN", "NEO_EYES_BRAIN",
    "NEO_BACKUP_BRAINS", "NEO_LIGHT_BRAIN", "NEO_CLOUD_EYES", "NEO_CUSTOM_PROVIDERS", "NEO_VOICE",
)  # fmt: skip


def _env(monkeypatch, **env):
    for k in _VARS:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(prov, "_cache", {})
    monkeypatch.setattr(prov, "_demoted", {})
    monkeypatch.setattr(health, "_live", {})
    monkeypatch.setattr(brains, "_probes", {})
    monkeypatch.setattr(brains, "_started", None)
    monkeypatch.setattr(brains, "voice_now", None)
    return config.settings()


def _up(monkeypatch, *urls):
    """What answers on this Mac: these URLs, nothing else."""

    async def ask(url):
        return url in urls

    monkeypatch.setattr(brains, "_ask", ask)
    asyncio.run(brains.refresh_probes(force=True))


def test_no_key_says_no_model_yet_in_words_for_any_host(monkeypatch):
    _env(monkeypatch)
    said = health.problems()
    assert said[0] == health.NO_MODEL
    assert not any("Gemini API key" in p or "Settings & API Keys" in p for p in said)
    assert "menu-bar" not in said[0]
    assert brains.summary()["ready"] is False


def test_gemini_extra_keys_alone_count(monkeypatch):
    _env(monkeypatch, GEMINI_API_KEYS=GEMINI)
    assert brains.ready() and health.NO_MODEL not in health.problems()


def test_a_key_no_role_uses_says_so(monkeypatch):
    _env(monkeypatch, ANTHROPIC_API_KEY=ANTHROPIC)  # (the roles are still Gemini, Groq, the local model)
    assert health.problems()[0] == health.NO_ROLE
    _env(monkeypatch, ANTHROPIC_API_KEY=ANTHROPIC, NEO_BRAIN="claude@claude-opus-5-5")
    assert not any(p in (health.NO_ROLE, health.NO_MODEL) for p in health.problems())


def test_the_local_model_counts_only_while_it_answers(monkeypatch):
    _env(monkeypatch, NEO_BRAIN="local")
    assert health.NO_MODEL in health.problems()
    _up(monkeypatch, brains.LOCAL_MODELS_URL)
    assert brains.ready() and health.NO_MODEL not in health.problems()
    # It can't see: with nothing else that sees, a task that needs a look stops — said once.
    assert "The local model can't see the screen and no Eyes brain is set: tasks that need a look will stop." in health.problems()


def test_a_text_only_brain_with_nothing_that_sees(monkeypatch):
    _env(monkeypatch, NEO_CUSTOM_PROVIDERS=json.dumps([DEEPSEEK]), DEEPSEEK_API_KEY="sk-TEST-deepseek-0000-abcdefghijkl",
         NEO_BRAIN="custom:deepseek", NEO_FAST_BRAIN="custom:deepseek")  # fmt: skip
    assert "DeepSeek can't see the screen and no Eyes brain is set: tasks that need a look will stop." in health.problems()
    # A Gemini key: Gemini looks (the seeing fallbacks), and nothing is said.
    _env(monkeypatch, NEO_CUSTOM_PROVIDERS=json.dumps([DEEPSEEK]), DEEPSEEK_API_KEY="sk-TEST-deepseek-0000-abcdefghijkl",
         NEO_BRAIN="custom:deepseek", GEMINI_API_KEY=GEMINI)  # fmt: skip
    assert not any("can't see the screen" in p for p in health.problems())
    # Groq alone: its vision model looks.
    _env(monkeypatch, GROQ_API_KEY=GROQ)
    assert not any("can't see the screen" in p for p in health.problems()) and brains.ready()


def test_a_refused_key_is_said_by_where_it_came_from(monkeypatch, tmp_path):
    from neo2.core import keystore

    _env(monkeypatch)
    store = tmp_path / "keys" / "api_keys.json"
    monkeypatch.setattr(config, "KEY_STORE_PATH", store)
    keystore.FileKeyStore(store).put("anthropic", ANTHROPIC)
    monkeypatch.setattr(config, "_settings", None)
    prov._key_health("claude@claude-opus-5-5", refused=True)
    assert health._live["key:anthropic"] == "Claude refused the API key saved in NEO. Open Brains & Keys to replace it."
    _env(monkeypatch, ANTHROPIC_API_KEY=ANTHROPIC)
    monkeypatch.setattr(config, "KEY_STORE_PATH", None)
    prov._key_health("claude", refused=True)
    assert "environment variable ANTHROPIC_API_KEY" in health._live["key:anthropic"]
    assert ANTHROPIC not in json.dumps(health._live)
    prov._key_health("claude", refused=False)
    assert "key:anthropic" not in health._live


def test_the_summary_carries_labels_and_booleans_never_a_key(monkeypatch):
    s = _env(monkeypatch, ANTHROPIC_API_KEY=ANTHROPIC, GEMINI_API_KEY=GEMINI, NEO_BRAIN="claude@claude-opus-5-5",
             NEO_FAST_BRAIN="claude@claude-haiku-5-5", NEO_EYES_BRAIN="gemini",
             NEO_CUSTOM_PROVIDERS=json.dumps([OLLAMA]))  # fmt: skip
    got = brains.summary()
    assert got["ready"] is True
    assert got["brain"] == {"ref": "claude@claude-opus-5-5", "provider": "anthropic", "label": "Claude Opus 5.5", "ok": True}
    assert got["quick"]["label"] == "Claude Haiku 5.5" and got["eyes"]["label"] == "Gemini 3.8 Flash"
    assert got["providers"] == {"gemini": True, "anthropic": True, "openai": False, "groq": False, "cerebras": False,
                                "openrouter": False, "custom:ollama": True}  # fmt: skip
    assert got["voice"] == "live" and got["restart_needed"] is False
    text = json.dumps(got)
    assert ANTHROPIC not in text and GEMINI not in text and ANTHROPIC[-4:] not in text and GEMINI[-4:] not in text
    assert sorted(got) == ["brain", "eyes", "local_voice", "providers", "quick", "ready", "restart_needed", "voice"]
    assert got["local_voice"]["state"] == "missing"  # (downloaded only when the user asks: neo2.core.fetch)
    del s


def test_a_brain_pasted_as_a_key_is_never_in_the_summary(monkeypatch):
    _env(monkeypatch, NEO_BRAIN=ANTHROPIC, GROQ_API_KEY=GROQ)
    got = brains.summary()
    assert got["brain"] == {"ref": "", "provider": "", "label": "Not a brain NEO knows", "ok": False}
    assert "TEST" not in json.dumps(got)


def test_a_changed_gemini_key_needs_a_restart_for_the_voice(monkeypatch):
    s = _env(monkeypatch, GEMINI_API_KEY=GEMINI)
    brains.started(s)
    assert brains.restart_needed() is False
    s.gemini_api_key = "AIzaTEST-another-0000-abcdefghijklm"
    assert brains.restart_needed() is True
    s.gemini_api_key = GEMINI
    s.voice = "local"
    assert brains.restart_needed() is True


def test_the_voice_is_gemini_live_only_with_a_gemini_key(monkeypatch):
    _env(monkeypatch, ANTHROPIC_API_KEY=ANTHROPIC, NEO_BRAIN="claude@claude-opus-5-5")
    assert brains.voice() == "local"
    _env(monkeypatch, ANTHROPIC_API_KEY=ANTHROPIC, NEO_BRAIN="claude@claude-opus-5-5", GEMINI_API_KEY=GEMINI)
    assert brains.voice() == "live"  # (a Claude brain keeps Gemini Live: decision 5)
    _env(monkeypatch, GEMINI_API_KEY=GEMINI, NEO_VOICE="off")
    assert brains.voice() == "off"
    monkeypatch.setattr(brains, "voice_now", lambda: "local")  # (Live is down: the on-device voice runs)
    _env(monkeypatch, GEMINI_API_KEY=GEMINI)
    monkeypatch.setattr(brains, "voice_now", lambda: "local")
    assert brains.voice() == "local"


def test_watch_tells_every_client_when_the_brains_change(monkeypatch):
    from neo2.core import events

    _env(monkeypatch)
    bus = events.EventBus()
    monkeypatch.setattr(events, "_bus", bus)
    seen: list[tuple[str, dict]] = []
    bus.subscribe(lambda ev: seen.append((ev.type, dict(ev.data))))

    async def nothing(*a, **k):
        return {}

    monkeypatch.setattr(brains, "refresh_probes", nothing)
    ticks = 0

    async def tick(_s):
        nonlocal ticks
        ticks += 1
        if ticks == 1:
            monkeypatch.setenv("GEMINI_API_KEY", GEMINI)
            config.settings().gemini_api_key = GEMINI  # (as brains.apply copies it)
        else:
            raise asyncio.CancelledError

    monkeypatch.setattr(health.asyncio, "sleep", tick)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(health.watch(every_s=0))
    told = [d for t, d in seen if t == "brains"]
    assert len(told) == 1 and told[0]["ready"] is True
    assert [t for t, _ in seen].count("problems") == 1
