"""Each setting on its own, first match wins: the environment, then what was saved in NEO (its key
store, its saved brains), then a checkout's .env, then ~/.neo/.env, then the default. Every file is
the test's own (a ~/.neo/.env stand-in and a checkout .env), every key a fake."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neo2.core import config, effects, keystore
from neo2.core.keystore import FileKeyStore, KeyStoreSource, SavedConfig, SavedSource

ENV_KEY, SAVED_KEY, CHECKOUT_KEY, USER_KEY = (f"AIzaTEST-{w}-0000" for w in ("env", "saved", "checkout", "user"))


@pytest.fixture
def neo(tmp_path, monkeypatch):
    """A fresh store and saved-brains file, and both .env files, all in tmp_path."""
    for k in ("GEMINI_API_KEY", "GEMINI_API_KEYS", "GROQ_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
              "OPENROUTER_API_KEY", "XAI_API_KEY", "NEO_BRAIN", "NEO_FAST_BRAIN", "NEO_EYES_BRAIN",
              "NEO_CUSTOM_PROVIDERS", "NEO_VOICE", "NEO_CLOUD_EYES", "NEO_DAILY_BUDGETS"):  # fmt: skip
        monkeypatch.delenv(k, raising=False)
    store_path, saved_path = tmp_path / "neo" / "api_keys.json", tmp_path / "neo" / "brains.json"
    monkeypatch.setattr(config, "KEY_STORE_PATH", store_path)
    monkeypatch.setattr(config, "SAVED_PATH", saved_path)

    class Neo:
        store = FileKeyStore(store_path)
        saved = SavedConfig(saved_path)
        user_env = tmp_path / "user.env"
        checkout_env = tmp_path / "checkout.env"

        def settings(self, *, files: bool = True) -> config.Settings:
            """What a new NEO.app process reads (effects on for this thread only, as in NEO.app)."""
            if not files:
                return config.Settings()
            monkeypatch.delenv("NEO_NO_EFFECTS", raising=False)
            token = effects._as_app.set(True)
            try:
                return config.Settings(_env_file=(self.user_env, self.checkout_env))
            finally:
                effects._as_app.reset(token)

    return Neo()


def test_env_beats_saved_beats_checkout_beats_user_env(neo, monkeypatch):
    neo.user_env.write_text(f"GEMINI_API_KEY={USER_KEY}\n")
    assert neo.settings().gemini_api_key == USER_KEY
    neo.checkout_env.write_text(f"GEMINI_API_KEY={CHECKOUT_KEY}\n")
    assert neo.settings().gemini_api_key == CHECKOUT_KEY
    neo.store.put("gemini", SAVED_KEY)
    assert neo.settings().gemini_api_key == SAVED_KEY
    monkeypatch.setenv("GEMINI_API_KEY", ENV_KEY)
    assert neo.settings().gemini_api_key == ENV_KEY
    # And back down, one at a time.
    monkeypatch.delenv("GEMINI_API_KEY")
    assert neo.store.remove("gemini")
    assert neo.settings().gemini_api_key == CHECKOUT_KEY


def test_each_key_is_decided_on_its_own(neo, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_TEST-env-0000")
    neo.store.put("anthropic", "sk-ant-TEST-saved-0000")
    neo.store.put("openai", "sk-proj-TEST-saved-0000")
    neo.user_env.write_text("GEMINI_API_KEY=AIzaTEST-user-0000\nGROQ_API_KEY=gsk_TEST-user-0000\nOPENROUTER_API_KEY=sk-or-TEST-0000\n")
    s = neo.settings()
    assert (s.groq_api_key, s.anthropic_api_key, s.gemini_api_key) == (
        "gsk_TEST-env-0000", "sk-ant-TEST-saved-0000", "AIzaTEST-user-0000",
    )  # fmt: skip
    assert (s.openai_api_key, s.openrouter_api_key) == ("sk-proj-TEST-saved-0000", "sk-or-TEST-0000")


def test_geminis_extra_keys(neo, monkeypatch):
    neo.store.put("gemini", SAVED_KEY, extra=["AIzaTEST-extra-1111", "AIzaTEST-extra-2222"])
    s = neo.settings()
    assert s.gemini_keys == [SAVED_KEY, "AIzaTEST-extra-1111", "AIzaTEST-extra-2222"]
    monkeypatch.setenv("GEMINI_API_KEYS", "AIzaTEST-env-extra-3333")  # the environment's extras win, on their own
    assert neo.settings().gemini_keys == [SAVED_KEY, "AIzaTEST-env-extra-3333"]


def test_the_store_answers_by_the_keys_own_names_and_the_saved_brains_by_field(neo):
    """Pinned: pydantic-settings resolves an aliased field by its alias, a plain one by its name."""
    neo.store.put("gemini", SAVED_KEY, extra=["AIzaTEST-extra-1111"])
    neo.store.put("anthropic", "sk-ant-TEST-saved-0000")
    neo.store.put("custom:deepseek", "sk-TEST-deepseek-0000")
    neo.saved.update({"brain": "claude@claude-opus-5-5", "voice": "local"})
    assert KeyStoreSource(config.Settings, config.KEY_STORE_PATH)() == {
        "GEMINI_API_KEY": SAVED_KEY,
        "GEMINI_API_KEYS": "AIzaTEST-extra-1111",
        "ANTHROPIC_API_KEY": "sk-ant-TEST-saved-0000",
        "custom_keys": {"custom:deepseek": "sk-TEST-deepseek-0000"},
    }
    assert SavedSource(config.Settings, config.SAVED_PATH)() == {"brain": "claude@claude-opus-5-5", "voice": "local"}
    s = neo.settings(files=False)  # (with effects off too: these are the test's own files)
    assert s.brain == "claude@claude-opus-5-5" and s.voice == "local" and s.custom_keys == {"custom:deepseek": "sk-TEST-deepseek-0000"}


def test_saved_brains_sit_between_the_environment_and_the_env_files(neo, monkeypatch):
    neo.user_env.write_text("NEO_BRAIN=groq\nNEO_FAST_BRAIN=local\nNEO_VOICE=live\n")
    assert neo.settings().brain == "groq"
    neo.saved.update({"brain": "claude@claude-opus-5-5", "eyes_brain": "claude@claude-opus-5-5", "voice": "local"})
    s = neo.settings()
    assert (s.brain, s.fast_brain, s.eyes_brain, s.voice) == (
        "claude@claude-opus-5-5", "local", "claude@claude-opus-5-5", "local",
    )  # fmt: skip
    monkeypatch.setenv("NEO_BRAIN", "gemini")
    assert neo.settings().brain == "gemini"


def test_an_endpoints_own_variable_wins_over_its_saved_key_which_wins_over_a_env_line(neo, monkeypatch):
    xai = {"id": "custom:xai", "label": "xAI", "template": "xai", "base_url": "https://api.x.ai/v1", "model": "grok-4.7",
           "env": "XAI_API_KEY"}  # fmt: skip
    neo.saved.update({"custom_providers": [xai]})
    neo.user_env.write_text("XAI_API_KEY=xai-TEST-file-0000\n")
    assert neo.settings().custom_keys == {"custom:xai": "xai-TEST-file-0000"}
    neo.store.put("custom:xai", "xai-TEST-saved-0000")
    assert neo.settings().custom_keys == {"custom:xai": "xai-TEST-saved-0000"}
    monkeypatch.setenv("XAI_API_KEY", "xai-TEST-env-0000")
    assert neo.settings().custom_keys == {"custom:xai": "xai-TEST-env-0000"}
    # An endpoint that names no variable gets none from the environment.
    neo.saved.update({"custom_providers": [{**xai, "env": None}]})
    assert neo.settings().custom_keys == {"custom:xai": "xai-TEST-saved-0000"}


def test_sources_say_where_each_key_comes_from_and_only_its_last_four(neo, monkeypatch):
    from neo2.core.brainref import CustomProvider

    neo.user_env.write_text("GEMINI_API_KEY=AIzaTEST-user-0000\nGROQ_API_KEY=gsk_TEST-user-gq12\n")
    neo.checkout_env.write_text("GEMINI_API_KEY=AIzaTEST-checkout-wxyz\n")
    neo.store.put("anthropic", "sk-ant-TEST-saved-x9Qa")
    neo.store.put("groq", "gsk_TEST-saved-sv34")
    xai = CustomProvider(id="custom:xai", label="xAI", base_url="https://api.x.ai/v1", model="grok-4.7", env="XAI_API_KEY")
    env = {"OPENAI_API_KEY": "sk-proj-TEST-env-op56", "XAI_API_KEY": "short"}
    files = [("dotenv", neo.user_env), ("checkout_env", neo.checkout_env)]
    got = keystore.sources(environ=env, store=neo.store, files=files, custom=[xai])
    assert got["gemini"] == {"set": True, "source": "checkout_env", "last4": "wxyz", "extra": 0, "saved": None, "also": ["dotenv"]}
    assert got["anthropic"]["source"] == "saved" and got["anthropic"]["last4"] == "x9Qa" and got["anthropic"]["saved"]
    assert got["groq"]["source"] == "saved" and got["groq"]["also"] == ["dotenv"]  # (the .env line is still there)
    assert got["openai"]["source"] == "env" and got["openai"]["last4"] == "op56"
    assert got["openrouter"] == {"set": False, "source": "none", "last4": None, "extra": 0, "saved": None, "also": []}
    assert got["custom:xai"]["source"] == "env" and got["custom:xai"]["last4"] is None  # (short: nothing of it)
    text = json.dumps(got)
    for whole in ("AIzaTEST", "sk-ant-TEST", "gsk_TEST", "sk-proj-TEST", "short"):
        assert whole not in text
    # Under the suite, by default neither the real store nor the real .env files.
    assert keystore.sources(environ={})["gemini"]["source"] == "none"


def test_library_settings_never_read_the_store(neo, tmp_path):
    neo.store.put("gemini", SAVED_KEY)
    neo.saved.update({"brain": "claude"})
    lib = config.LibrarySettings(data_dir=tmp_path / "agent")
    assert lib.gemini_api_key == "" and lib.brain == "gemini" and lib.custom_keys == {}
    sources = config.LibrarySettings.settings_customise_sources(config.LibrarySettings, "init", "env", "dotenv", "secrets")
    assert sources == ("init",)


def test_a_corrupt_saved_file_never_stops_neo_from_starting(neo):
    Path(config.SAVED_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(config.SAVED_PATH).write_text('{"version": 1, "brain": "claud", "voice": "loud", "custom_providers": [{"id": 1}]}')
    Path(config.KEY_STORE_PATH).write_text("not json")
    s = neo.settings()
    assert s.brain == "claud" and s.voice == "auto" and s.custom_providers == []  # (reported in health)
    assert s.gemini_api_key == ""


def test_a_store_that_fails_to_read_never_stops_neo_from_starting(neo, monkeypatch, capsys):
    neo.store.put("anthropic", "sk-ant-TEST-saved-0000")
    neo.saved.update({"brain": "claude"})

    def broken(self):
        raise OSError("I/O error")

    monkeypatch.setattr(FileKeyStore, "list", broken)
    monkeypatch.setattr(SavedConfig, "read", broken)
    s = neo.settings(files=False)
    assert s.anthropic_api_key == "" and s.brain == "gemini"  # (as if nothing were saved)
    assert "TEST" not in capsys.readouterr().out
