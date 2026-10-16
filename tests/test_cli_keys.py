"""`python -m neo2 keys …` and `python -m neo2 brain …`: a key typed without an echo (never on the
command line, never printed but its last 4), written to NEO's store directly, a running NEO told to
read it; the roles' rules as in the window; --doctor's new rows. The store is this test's own; getpass,
the provider check and the running core are faked; every key is an obvious fake."""

from __future__ import annotations

import json
import sys

import pytest

from neo2.core import brains, cli_keys, config, health, keystore, logredact
from neo2.core import providers as prov
from neo2.core.providers import keycheck

KEY = "sk-ant-TEST-0000-abcdefghijklmnop-x9Qa"
GEMINI = "AIzaTEST-gemini-0000-abcdefghijk-g3m1"
_VARS = ("GEMINI_API_KEY", "GEMINI_API_KEYS", "GROQ_API_KEY", "ANTHROPIC_API_KEY", "CEREBRAS_API_KEY", "OPENAI_API_KEY",
         "OPENROUTER_API_KEY", "NEO_BRAIN", "NEO_FAST_BRAIN", "NEO_EYES_BRAIN", "NEO_CUSTOM_PROVIDERS")  # fmt: skip


@pytest.fixture
def cli(monkeypatch, tmp_path):
    for k in _VARS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(config, "KEY_STORE_PATH", tmp_path / "home" / "api_keys.json")
    monkeypatch.setattr(config, "SAVED_PATH", tmp_path / "home" / "brains.json")
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(prov, "_cache", {})
    monkeypatch.setattr(health, "_live", {})
    monkeypatch.setattr(brains, "_probes", {})
    monkeypatch.setattr(logredact, "_remembered", set())
    typed: list[str] = []
    prompts: list[str] = []

    def getpass(prompt=""):
        prompts.append(prompt)
        return typed.pop(0) if typed else ""

    monkeypatch.setattr(cli_keys.getpass, "getpass", getpass)
    told: list[bool] = []

    async def tell():
        told.append(True)
        return False  # (no NEO running)

    monkeypatch.setattr(cli_keys, "_tell_core", tell)
    verdict = {"status": "ok"}

    async def check(provider, key=None, endpoint=None):
        st = verdict["status"]
        return keycheck.KeyCheck(st == "ok", st, 200 if st == "ok" else 401, f"[{st}] api.anthropic.com")

    monkeypatch.setattr(keycheck, "check", check)

    class CLI:
        def __init__(self):
            self.typed, self.prompts, self.told, self.verdict = typed, prompts, told, verdict

        def run(self, *argv):
            monkeypatch.setattr(config, "_settings", None)
            monkeypatch.setattr(sys, "argv", ["neo2", *argv])
            return cli_keys.main(list(argv))

    return CLI()


def test_keys_set_reads_the_key_without_an_echo_and_never_prints_it(cli, capsys):
    cli.typed.append(f"  {KEY}\n")
    assert cli.run("keys", "set", "claude") == 0
    assert cli.prompts == ["Anthropic (Claude) API key (it isn't shown as you type): "]
    assert KEY not in " ".join(sys.argv)
    out = capsys.readouterr()
    assert KEY not in out.out + out.err and "••••x9Qa" in out.out
    assert "NEO stops using Claude for the day at $2.00" in out.out  # (decision 4: a paid key saved in NEO)
    assert "NEO isn't running here" in out.out and cli.told == [True]
    stored = keystore.FileKeyStore(config.KEY_STORE_PATH).get("anthropic")
    assert stored is not None and stored.key == KEY
    assert json.loads(config.SAVED_PATH.read_text())["daily_budgets"] == {"anthropic": 2.0}


def test_a_key_the_provider_refuses_isnt_saved_unless_asked(cli, capsys):
    cli.verdict["status"] = "bad_key"
    cli.typed.append(KEY)
    assert cli.run("keys", "set", "anthropic") == 1
    assert not config.KEY_STORE_PATH.exists() and "Not saved" in capsys.readouterr().out
    cli.typed.append(KEY)
    assert cli.run("keys", "set", "anthropic", "--no-test") == 0
    assert keystore.FileKeyStore(config.KEY_STORE_PATH).get("anthropic") is not None


def test_gemini_extras_and_a_key_for_another_provider_refused(cli, capsys):
    cli.typed += [KEY]  # (an Anthropic key typed for Gemini: never saved for it, nor sent to Google)
    assert cli.run("keys", "set", "gemini", "--no-test") == 1
    out = capsys.readouterr().out
    assert "That looks like an Anthropic key, so NEO didn't send it to Gemini." in out and "Nothing saved." in out
    assert KEY not in out and keystore.FileKeyStore(config.KEY_STORE_PATH).get("gemini") is None
    cli.typed += ["nope-TEST-0000-abcdefghijklmnop"]  # (a shape NEO doesn't know: a soft hint only)
    assert cli.run("keys", "set", "gemini", "--no-test") == 0
    assert "Gemini keys usually start with AIza." in capsys.readouterr().out
    cli.typed += [GEMINI, "AIzaTEST-extra-0000-abcdefghijklmn", ""]
    assert cli.run("keys", "set", "gemini", "--extra") == 0
    got = keystore.FileKeyStore(config.KEY_STORE_PATH).get("gemini")
    assert got.key == GEMINI and got.extra == ("AIzaTEST-extra-0000-abcdefghijklmn",)
    assert "daily" not in capsys.readouterr().out  # (a free tier: no cap)


def test_keys_list_and_remove(cli, capsys, monkeypatch):
    cli.typed.append(KEY)
    cli.run("keys", "set", "anthropic", "--no-test")
    capsys.readouterr()
    assert cli.run("keys", "list") == 0
    out = capsys.readouterr().out
    assert "✓ Anthropic (Claude) (anthropic) — saved in NEO ••••x9Qa" in out and "· OpenAI (openai) — not set" in out
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-TEST-env-000000000000-envv")
    assert cli.run("keys", "remove", "anthropic") == 0
    out = capsys.readouterr().out
    assert "Removed the Anthropic (Claude) key saved in NEO." in out
    assert "A key is still set from the environment (ANTHROPIC_API_KEY) ••••envv." in out
    assert KEY not in out


def test_a_key_typed_where_the_provider_goes_is_never_printed(cli, capsys):
    assert cli.run("keys", "set", KEY) == 2
    out = capsys.readouterr().out
    assert KEY not in out and "That isn't a provider NEO keeps a key for" in out


def test_keys_test_says_how_it_went(cli, capsys):
    cli.verdict["status"] = "rate_limited"
    assert cli.run("keys", "test", "openai") == 1  # (KeyCheck.ok is the check's: faked False here)
    assert "[rate_limited]" in capsys.readouterr().out


def test_brain_set_follows_the_windows_rules(cli, capsys):
    assert cli.run("brain", "set", "--brain", "claude@claude-opus-5-5") == 1
    assert "brain isn't set up" in capsys.readouterr().out
    cli.typed.append(KEY)
    cli.run("keys", "set", "anthropic", "--no-test")
    capsys.readouterr()
    assert cli.run("brain", "set", "--brain", "claude@claude-opus-5-5", "--quick", "claude@claude-haiku-5-5",
                   "--budget", "claude=5") == 0  # fmt: skip
    out = capsys.readouterr().out
    assert "Note: Quick replies on a paid model costs money on every summary." in out and "Saved." in out
    saved = json.loads(config.SAVED_PATH.read_text())
    assert saved["brain"] == "claude@claude-opus-5-5" and saved["daily_budgets"] == {"anthropic": 5.0}
    assert cli.run("brain", "set", "--brain", KEY) == 1
    assert KEY not in capsys.readouterr().out
    assert cli.run("brain", "show") == 0
    out = capsys.readouterr().out
    assert "✓ Brain: Claude Opus 5.5 (claude@claude-opus-5-5) — saved in NEO" in out and "has a model to ask" in out


def test_python_m_neo2_hands_keys_and_brain_to_the_cli(cli, monkeypatch):
    from neo2.core import main as main_mod

    got: list = []
    monkeypatch.setattr(cli_keys, "main", lambda argv: got.append(list(argv)) or 0)
    import neo2.core

    monkeypatch.setattr(logredact, "install", lambda: False)  # (process-wide: not in a test)
    monkeypatch.setattr(neo2.core, "log_to_stdout", lambda: None)
    monkeypatch.setattr(sys, "argv", ["neo2", "keys", "list"])
    with pytest.raises(SystemExit) as e:
        main_mod.main()
    assert e.value.code == 0 and got == [["keys", "list"]]


def test_the_doctor_shows_sources_and_roles_and_no_longer_fails_for_groq(cli, capsys, monkeypatch):
    from neo2.core import main as main_mod

    monkeypatch.setenv("GEMINI_API_KEY", GEMINI)
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(main_mod.sys, "platform", "linux")  # (no macOS permission rows here)
    import types

    for mod in ("laya", "mlx_audio"):  # (the local models' packages: not loaded for this)
        monkeypatch.setitem(sys.modules, mod, types.ModuleType(mod))
    code = main_mod.doctor()
    out = capsys.readouterr().out
    assert "✓ Google Gemini — from the environment (GEMINI_API_KEY) ••••g3m1" in out
    assert "· Groq — not set" in out and "✓ Brain: Gemini 3.8 Flash" in out and "✓ NEO has a model to ask" in out
    assert GEMINI not in out
    assert code == 0  # (no Groq key is no failure any more)
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setattr(config, "_settings", None)
    config.KEY_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    (config.KEY_STORE_PATH.parent / "api_keys.json.bad-20261017T000000").write_text("{}")  # (a copy set aside)
    assert main_mod.doctor() == 1
    out = capsys.readouterr().out
    assert "✗ NEO has a model to ask — connect one in Brains & Keys" in out
    assert "· 1 older copy of api_keys.json (…bad-<date>) still in" in out and "they may hold keys" in out
