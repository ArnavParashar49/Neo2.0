"""Brains & Keys' commands in the core (neo2.core.brains): a key tested, saved (with the $2 a day cap
for a paid one saved here), removed; the roles' rules; an endpoint added and removed; and a change
applied while NEO runs — each provider whose key changed is forgotten, nothing else. NEO's files are
this test's own (tmp_path); provider checks are faked; every key is an obvious fake."""

from __future__ import annotations

import asyncio
import json

import pytest

from neo2.core import brains, config, health, logredact
from neo2.core import providers as prov
from neo2.core.providers import keycheck

ANTHROPIC = "sk-ant-TEST-0000-abcdefghijklmnop-x9Qa"
GEMINI = "AIzaTEST-gemini-0000-abcdefghijk-g3m1"
DEEPSEEK = {"label": "DeepSeek", "template": "deepseek", "base_url": "https://api.deepseek.com", "model": "deepseek-v4-pro"}
_VARS = (
    "GEMINI_API_KEY", "GEMINI_API_KEYS", "GROQ_API_KEY", "ANTHROPIC_API_KEY", "CEREBRAS_API_KEY", "OPENAI_API_KEY",
    "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "NEO_BRAIN", "NEO_FAST_BRAIN", "NEO_OFFLINE_BRAIN", "NEO_EYES_BRAIN",
    "NEO_BACKUP_BRAINS", "NEO_LIGHT_BRAIN", "NEO_CLOUD_EYES", "NEO_CUSTOM_PROVIDERS", "NEO_VOICE", "NEO_DAILY_BUDGETS",
)  # fmt: skip


@pytest.fixture
def files(monkeypatch, tmp_path):
    for k in _VARS:
        monkeypatch.delenv(k, raising=False)
    home = tmp_path / "neo-home"
    monkeypatch.setattr(config, "KEY_STORE_PATH", home / "api_keys.json")
    monkeypatch.setattr(config, "SAVED_PATH", home / "brains.json")
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(prov, "_cache", {})
    monkeypatch.setattr(prov, "_demoted", {})
    monkeypatch.setattr(health, "_live", {})
    monkeypatch.setattr(brains, "_probes", {})
    monkeypatch.setattr(brains, "_started", None)
    monkeypatch.setattr(logredact, "_remembered", set())
    checked: list[tuple[str, bool]] = []

    async def check(provider, key=None, endpoint=None):
        checked.append((provider, key is not None))
        if key == "sk-ant-TEST-refused-0000000000":
            return keycheck.KeyCheck(False, "bad_key", 401, "api.anthropic.com didn't accept this key.")
        return keycheck.KeyCheck(True, "ok", 200, "The key works (the server).", 3)

    monkeypatch.setattr(keycheck, "check", check)
    return home, checked


def run(**cmd):
    return asyncio.run(brains.command(cmd, restart=lambda: False))


def test_a_key_saved_is_kept_masked_capped_and_used_at_once(files, capsys):
    home, checked = files
    prov._cache["claude"] = object()  # (built with no key: forgotten when one is saved)
    prov._cache["gemini"] = object()
    t, data = run(cmd="key_save", id="r3", provider="claude", key=f"  '{ANTHROPIC}'  ")
    assert t == "key_save_result" and data["ok"] is True and data["id"] == "r3" and data["provider"] == "anthropic"
    assert data["key"] == {"set": True, "source": "saved", "last4": "x9Qa"} and data["shadowed_by"] is None
    assert data["daily_budget"] == 2.0  # (a paid key saved in NEO: the $2 a day cap, decision 4)
    assert checked == [("anthropic", True)]
    stored = json.loads((home / "api_keys.json").read_text())
    assert stored["keys"]["anthropic"]["key"] == ANTHROPIC
    assert json.loads((home / "brains.json").read_text())["daily_budgets"] == {"anthropic": 2.0}
    s = config.settings()
    assert s.anthropic_api_key == ANTHROPIC and s.daily_budgets == {"anthropic": 2.0}
    assert "claude" not in prov._cache and "gemini" in prov._cache  # (only its own provider forgotten)
    assert ANTHROPIC not in json.dumps(data) and ANTHROPIC not in capsys.readouterr().out
    assert logredact.redact(f"x {ANTHROPIC}") == "x ••••x9Qa"


def test_a_key_the_provider_refuses_isnt_saved(files):
    home, _ = files
    t, data = run(cmd="key_save", id="r", provider="anthropic", key="sk-ant-TEST-refused-0000000000")
    assert data == {"id": "r", "provider": "anthropic", "ok": False, "shadowed_by": None, "restart_needed": False,
                    "error": "bad_key", "message": "api.anthropic.com didn't accept this key."}  # fmt: skip
    assert not (home / "api_keys.json").exists()
    _, data = run(cmd="key_save", id="r", provider="anthropic", key="sk-ant-TEST-refused-0000000000", test=False)
    assert data["ok"] is True  # (saved without the test, when asked)


@pytest.mark.parametrize(
    "cmd",
    [
        {"provider": "nobody", "key": ANTHROPIC},
        {"provider": "anthropic", "key": "two\nlines"},
        {"provider": "anthropic", "key": ""},
        {"provider": "anthropic", "key": ANTHROPIC, "extra": [GEMINI]},  # (extra is Gemini's alone)
        {"provider": "local", "key": ANTHROPIC},  # (no key to keep)
        {"provider": "custom:gone", "key": ANTHROPIC},
    ],
)
def test_what_isnt_a_key_save_is_invalid(files, cmd):
    _, data = run(cmd="key_save", id="x", **cmd)
    assert data["ok"] is False and data["error"] == "invalid"
    assert "two" not in json.dumps(data) and ANTHROPIC not in json.dumps(data)


def test_a_key_from_the_environment_shadows_one_saved_and_isnt_capped(files, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-TEST-env-0000000000000000")
    monkeypatch.setattr(config, "_settings", None)
    _, data = run(cmd="key_save", id="r", provider="anthropic", key=ANTHROPIC)
    assert data["ok"] is True and data["shadowed_by"] == "env"
    assert config.settings().anthropic_api_key == "sk-ant-TEST-env-0000000000000000"  # (R1: the environment wins)


def test_gemini_extras_and_a_restart_for_the_voice(files):
    brains.started(config.settings())
    _, data = run(cmd="key_save", id="g", provider="google", key=GEMINI, extra=["AIzaTEST-extra-0000-abcdefghijklmn"])
    assert data["ok"] is True and data["restart_needed"] is True and data["daily_budget"] is None  # (free tier: no cap)
    assert config.settings().gemini_keys == [GEMINI, "AIzaTEST-extra-0000-abcdefghijklmn"]


def test_removing_a_saved_key_says_what_is_still_set(files, monkeypatch):
    run(cmd="key_save", id="a", provider="anthropic", key=ANTHROPIC)
    _, data = run(cmd="key_remove", id="b", provider="anthropic")
    assert data == {"id": "b", "provider": "anthropic", "ok": True, "removed": True, "still_set": None}
    assert config.settings().anthropic_api_key == ""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-TEST-env-0000000000000000")
    _, data = run(cmd="key_remove", id="c", provider="anthropic")
    assert data["removed"] is False and data["still_set"] == "env"


def test_the_detail_carries_last4_and_sources_never_a_key(files):
    run(cmd="key_save", id="a", provider="anthropic", key=ANTHROPIC)
    _, d = run(cmd="brains_get", id="r1")
    assert d["id"] == "r1"
    assert d["keys"]["anthropic"]["source"] == "saved" and d["keys"]["anthropic"]["last4"] == "x9Qa"
    assert d["keys"]["gemini"]["set"] is False
    assert d["roles"]["brain"] == {"ref": "gemini", "valid": True, "source": "default"}
    assert d["daily_budgets"]["value"] == {"anthropic": 2.0} and d["daily_budgets"]["source"] == "saved"
    assert {c["id"] for c in d["catalog"]} >= {"gemini", "anthropic", "openai", "groq", "cerebras", "openrouter", "custom"}
    assert d["store"]["ok"] is True and d["voice"]["live_possible"] is False
    assert ANTHROPIC not in json.dumps(d)


def test_the_roles_rules(files):
    _, data = run(cmd="brains_set", id="s", brain="claude@claude-opus-5-5")
    assert data == {"id": "s", "ok": False, "error": "not_configured", "field": "brain", "warnings": [],
                    "restart_needed": False, "shadowed": []}  # fmt: skip
    run(cmd="key_save", id="k", provider="anthropic", key=ANTHROPIC)
    for changes, error, field in (
        ({"brain": "claud"}, "invalid", "brain"),
        ({"brain": ""}, "invalid", "brain"),
        ({"brain": "claude", "eyes_brain": "groq"}, "not_configured", "eyes_brain"),
        ({"cloud_eyes": "sometimes"}, "invalid", "cloud_eyes"),
        ({"backup_brains": ["claude"] * 2 + ["groq", "gemini", "local"]}, "invalid", "backup_brains"),
        ({"daily_budgets": {"anthropic": "lots"}}, "invalid", "daily_budgets"),
        ({"daily_budgets": {"anthropic": 5000}}, "invalid", "daily_budgets"),
    ):
        _, data = run(cmd="brains_set", id="s", **changes)
        assert (data["ok"], data.get("error"), data.get("field")) == (False, error, field), changes
    _, data = run(cmd="brains_set", id="s", brain="claude@claude-opus-5-5", fast_brain="claude@claude-haiku-5-5",
                  eyes_brain="claude@claude-opus-5-5", light_brain="", cloud_eyes="off", backup_brains=None)  # fmt: skip
    assert data["ok"] is True and data["warnings"] == ["Quick replies on a paid model costs money on every summary"]
    s = config.settings()
    assert (s.brain, s.fast_brain, s.eyes_brain, s.light_brain, s.cloud_eyes) == (
        "claude@claude-opus-5-5", "claude@claude-haiku-5-5", "claude@claude-opus-5-5", "", "off")  # fmt: skip
    assert s.voice == "auto"  # (a Claude brain doesn't switch the voice: decision 5)
    saved = json.loads(config.SAVED_PATH.read_text())
    assert saved["brain"] == "claude@claude-opus-5-5" and "anthropic" not in json.dumps(saved).replace('"anthropic": 2.0', "")


def test_eyes_must_see_and_the_brain_must_call_tools(files, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_TEST-groq-0000-abcdefghijklmnop")
    monkeypatch.setenv("NEO_CUSTOM_PROVIDERS", json.dumps([
        {"id": "custom:notools", "label": "No tools", "base_url": "https://example.test/v1", "model": "m", "tools": False}]))  # fmt: skip
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    monkeypatch.setattr(config, "_settings", None)
    config.settings().custom_keys = {"custom:notools": "sk-TEST-notools-0000"}
    _, data = run(cmd="brains_set", id="s", eyes_brain="groq")
    assert data["error"] == "cant_see" and data["field"] == "eyes_brain"
    _, data = run(cmd="brains_set", id="s", brain="custom:notools")
    assert data["error"] == "no_tools"
    _, data = run(cmd="brains_set", id="s", brain="groq", fast_brain="groq")
    assert data["ok"] and data["warnings"] == ["Groq can't see your screen and no Eyes brain is set"]


def test_something_on_this_mac_counts_only_while_it_answers(files, monkeypatch):
    async def down(url):
        return False

    monkeypatch.setattr(brains, "_ask", down)
    _, data = run(cmd="brains_set", id="s", brain="local")
    assert data["error"] == "not_configured"

    async def up(url):
        return url == brains.LOCAL_MODELS_URL

    monkeypatch.setattr(brains, "_ask", up)
    _, data = run(cmd="brains_set", id="s", brain="local", fast_brain="local")
    assert data["ok"] is True and config.settings().brain == "local"
    assert brains.summary()["ready"] is True


def test_an_environment_variable_shadows_a_saved_role(files, monkeypatch):
    monkeypatch.setenv("NEO_BRAIN", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI)
    monkeypatch.setattr(config, "_settings", None)
    _, data = run(cmd="brains_set", id="s", brain="gemini_lite")
    assert data["ok"] and data["shadowed"] == ["brain"] and config.settings().brain == "gemini"


def test_an_endpoint_is_added_used_and_removed(files, monkeypatch):
    t, data = run(cmd="custom_save", id="c", endpoint={**DEEPSEEK, "id": None}, key="sk-TEST-deepseek-0000-abcdefgh")
    assert (t, data) == ("custom_save_result", {"id": "c", "ok": True, "provider": "custom:deepseek"})
    s = config.settings()
    (cp,) = s.custom_providers
    assert cp.env == "DEEPSEEK_API_KEY" and s.custom_keys == {"custom:deepseek": "sk-TEST-deepseek-0000-abcdefgh"}
    _, data = run(cmd="brains_set", id="s", brain="custom:deepseek")
    assert data["ok"] and data["warnings"] == ["DeepSeek can't see your screen and no Eyes brain is set"]
    _, data = run(cmd="custom_remove", id="r", provider="custom:deepseek")
    assert data == {"id": "r", "ok": False, "provider": "custom:deepseek", "error": "in_use"}
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI)
    monkeypatch.setattr(config, "_settings", None)
    assert run(cmd="brains_set", id="s", brain="gemini")[1]["ok"]
    _, data = run(cmd="custom_remove", id="r", provider="custom:deepseek")
    assert data["ok"] is True and config.settings().custom_providers == [] and config.settings().custom_keys == {}
    _, data = run(cmd="custom_save", id="c", endpoint={**DEEPSEEK, "base_url": "http://api.example.test/v1"})
    assert data == {"id": "c", "ok": False, "provider": None, "error": "invalid"}  # (http only on this Mac)


def test_a_failure_is_fixed_words_and_restart_needs_neo_app(files, monkeypatch):
    async def boom(changes):
        raise RuntimeError(f"it broke near {ANTHROPIC}")

    monkeypatch.setattr(brains, "set_roles", boom)
    t, data = run(cmd="brains_set", id="s", brain="claude")
    assert (t, data) == ("brains_set_result", {"id": "s", "ok": False, "error": "failed"})
    assert run(cmd="brains_restart", id="z") == ("brains_restart_result", {"id": "z", "ok": False, "error": "no_app"})
    assert asyncio.run(brains.command({"cmd": "brains_restart", "id": "z"}, restart=lambda: True))[1] == {"id": "z", "ok": True}
    assert brains.request_id({"id": "x" * 40}) == "" and brains.request_id({"id": "r 1"}) == ""


def test_apply_forgets_only_what_changed(files):
    s = config.settings()
    for n in ("claude", "gemini", "groq", "custom:deepseek"):
        prov._cache[n] = object()
    run(cmd="custom_save", id="c", endpoint=DEEPSEEK)
    assert "custom:deepseek" not in prov._cache and {"claude", "gemini", "groq"} <= set(prov._cache)
    s.claude_model = "claude-opus-5-5"  # (the file still says otherwise: applying puts the file's back)
    asyncio.run(brains.apply(tell=False))
    assert config.settings().claude_model == "claude-opus-5" and "claude" not in prov._cache and "gemini" in prov._cache
