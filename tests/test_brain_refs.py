"""Brain references, "<brain>[@<model>]": the grammar, its aliases, and what becomes of one NEO can't
read — a placeholder the chains skip and the health list names, never a crash and never another
provider."""

from __future__ import annotations

import pytest

from neo2.core import brainref, config, health
from neo2.core.brainref import CustomProvider, parse


@pytest.mark.parametrize(
    "typed,text,brain,model",
    [
        ("gemini", "gemini", "gemini", None),
        ("  Claude ", "claude", "claude", None),
        ("anthropic", "claude", "claude", None),
        ("google@gemini-3.7-flash", "gemini@gemini-3.7-flash", "gemini", "gemini-3.7-flash"),
        ("claude@claude-haiku-5-5", "claude@claude-haiku-5-5", "claude", "claude-haiku-5-5"),
        ("openai@gpt-6-luna", "openai@gpt-6-luna", "openai", "gpt-6-luna"),
        ("openai", "openai", "openai", None),
        ("openrouter@deepseek/deepseek-v4-pro", "openrouter@deepseek/deepseek-v4-pro", "openrouter", "deepseek/deepseek-v4-pro"),
        ("custom:ollama@qwen3:14b", "custom:ollama@qwen3:14b", "custom:ollama", "qwen3:14b"),
        ("CUSTOM:My-Box", "custom:my-box", "custom:my-box", None),
        ("groq_vision@qwen/qwen3.8-27b", "groq_vision@qwen/qwen3.8-27b", "groq_vision", "qwen/qwen3.8-27b"),
        ("local@mlx-community/gemma+x.1", "local@mlx-community/gemma+x.1", "local", "mlx-community/gemma+x.1"),
    ],
)
def test_the_grammar(typed, text, brain, model):
    r = parse(typed)
    assert r.valid and (r.text, r.brain, r.model) == (text, brain, model)
    assert parse(r.text) == r  # (its normal form reads back the same)


def test_bare_legacy_names_are_their_old_selves():
    for n in brainref.LEGACY:
        assert parse(n).legacy
    assert not parse("claude@claude-opus-5").legacy and not parse("openai").legacy and parse("custom:x").custom


@pytest.mark.parametrize(
    "typed",
    ["claud", "openai@", "@gpt", "custom:", "custom:Bad Slug", "custom:" + "a" * 25, "gemini@has space",
     "gemini@what?", "gem\x00ini", "x" * 300, "claude@" + "m" * 161, "anthropic/claude", "gemini lite"],
)  # fmt: skip
def test_what_it_cant_read_stays_what_was_typed(typed):
    r = parse(typed)
    assert not r.valid and r.brain == "" and r.text
    assert not parse(r.text).valid  # a placeholder never turns into a brain
    assert len(r.text) <= brainref.MAX_TEXT and r.text.isprintable()
    assert parse(None).text == parse("  ").text == "" and not parse("").valid


def test_a_bad_brain_name_no_longer_stops_neo_from_starting(monkeypatch):
    """NEO_BRAIN=openai used to fail Settings() (NEO.app then crash-looped); NEO_BRAIN=claud is kept
    as typed, skipped and reported — never swapped for a different provider."""
    monkeypatch.setenv("NEO_BRAIN", "openai")
    monkeypatch.setenv("NEO_FAST_BRAIN", "anthropic")
    monkeypatch.setattr(config, "_settings", None)
    s = config.settings()
    assert (s.brain, s.fast_brain) == ("openai", "claude")
    monkeypatch.setenv("NEO_BRAIN", "claud")
    monkeypatch.setenv("NEO_OFFLINE_BRAIN", "  ")
    monkeypatch.setattr(config, "_settings", None)
    s = config.settings()
    assert s.brain == "claud" and s.offline_brain == ""
    said = health.problems()
    assert "NEO_BRAIN=claud isn't a brain NEO knows. Pick one in Brains & Keys." in said
    assert not any("NEO_FAST_BRAIN" in p or "NEO_OFFLINE_BRAIN" in p for p in said)


def test_a_key_pasted_as_a_brain_is_never_shown_back(monkeypatch):
    for typed in ("sk-ant-TEST-0000-abcdefghijkl", "AIzaTEST0000abcdefghijklmnopqrst", "gsk_TEST", "x" * 60):
        monkeypatch.setenv("NEO_EYES_BRAIN", typed)
        monkeypatch.setattr(config, "_settings", None)
        said = [p for p in health.problems() if "NEO_EYES_BRAIN" in p]
        assert said == ["NEO_EYES_BRAIN isn't set to a brain NEO knows. Pick one in Brains & Keys."]
        assert typed not in " ".join(health.problems())


def test_backups_light_and_eyes(monkeypatch):
    s = config.Settings()
    assert (s.eyes_brain, s.backup_brains, s.light_brain) == ("", None, "gemini_lite")  # today's chains
    assert brainref.backups("auto") is None and brainref.backups("") is None and brainref.backups(None) is None
    assert brainref.backups("none") == [] and brainref.backups("[]") == []
    assert brainref.backups("claude, Google ,claude,groq,local,openai") == ["claude", "gemini", "groq", "local"]  # 4 at most
    assert brainref.backups('["openai@gpt-6-luna", "claud"]') == ["openai@gpt-6-luna", "claud"]  # (a placeholder stays)
    assert brainref.backups("[oops") is None and brainref.backups(7) is None
    monkeypatch.setenv("NEO_BACKUP_BRAINS", "claude,groq")
    monkeypatch.setenv("NEO_LIGHT_BRAIN", "")
    monkeypatch.setenv("NEO_EYES_BRAIN", "anthropic@claude-opus-5-5")
    s = config.Settings()
    assert (s.backup_brains, s.light_brain, s.eyes_brain) == (["claude", "groq"], "", "claude@claude-opus-5-5")


def test_cloud_eyes_gains_off_and_typos_stay_auto():
    for typed, want in (("off", "off"), ("Never", "off"), ("on", "on"), ("always", "on"), ("", "auto"),
                        ("auto", "auto"), ("sometimes", "auto"), ("false", "auto")):  # fmt: skip
        assert config.Settings(cloud_eyes=typed).cloud_eyes == want


def test_daily_budgets():
    assert brainref.budgets({"claude": 2, "openai": "$1.50", "bad key!": 1, "groq": -1, "x_y": "nan"}) == {
        "anthropic": 2.0,
        "openai": 1.5,
    }
    assert brainref.budgets('{"custom:deepseek": 0.5}') == {"custom:deepseek": 0.5}
    assert brainref.budgets("anthropic=2; openai=1") == {"anthropic": 2.0, "openai": 1.0}
    assert brainref.budgets("{oops") == {} and brainref.budgets(None) == {}


OLLAMA = {"id": "custom:ollama", "label": "Ollama", "template": "ollama", "base_url": "http://localhost:11434/v1",
          "model": "qwen3:14b"}  # fmt: skip


def test_an_endpoint_and_its_rules():
    cp = CustomProvider.model_validate(OLLAMA)
    assert cp.slug == "ollama" and cp.on_device and cp.tools and not cp.vision and cp.context == 32768
    assert CustomProvider.model_validate({**OLLAMA, "base_url": "https://api.deepseek.com/"}).base_url == "https://api.deepseek.com"
    assert not CustomProvider.model_validate({**OLLAMA, "base_url": "https://api.deepseek.com"}).on_device
    for host in ("http://127.0.0.1:1234/v1", "http://[::1]:8080/v1", "https://localhost/v1"):
        assert CustomProvider.model_validate({**OLLAMA, "base_url": host}).on_device
    for bad in (
        {"base_url": "http://api.example.com/v1"},  # http only on this Mac
        {"base_url": "https://user:pw@api.example.com/v1"},  # no password in it
        {"base_url": "https://api.example.com/v1?key=x"},  # no query
        {"base_url": "https://api.example.com/v1#x"},
        {"base_url": "ftp://api.example.com"},
        {"base_url": "https://" + "a" * 300},
        {"base_url": "https://api.example.com:notaport/v1"},
        {"id": "custom:Bad Slug"},
        {"id": "openai"},
        {"label": ""},
        {"label": "x" * 41},
        {"context": 1000},
        {"model": "has space"},
        {"env": "NEO_BRAIN"},
        {"env": "lower"},
        {"template": "nope"},
        {"secret": "x"},  # extra=forbid
    ):
        with pytest.raises(ValueError):
            CustomProvider.model_validate({**OLLAMA, **bad})
    with pytest.raises(ValueError) as e:
        CustomProvider.model_validate({**OLLAMA, "base_url": "https://user:hunter2@api.example.com/v1"})
    assert "hunter2" not in str(e.value)  # a URL with a password in it is a secret


def test_endpoints_are_read_tolerantly(monkeypatch):
    good = [OLLAMA, {**OLLAMA, "id": "custom:lm", "base_url": "http://localhost:1234/v1"}]
    junk = [{"id": 1}, "nope", {**OLLAMA, "base_url": "http://evil.example/v1"}, OLLAMA]  # (a second ollama: dropped)
    assert [c.id for c in brainref.custom_list(good + junk)] == ["custom:ollama", "custom:lm"]
    assert brainref.custom_list("[oops") == [] and brainref.custom_list(None) == []
    many = [{**OLLAMA, "id": f"custom:e{i}"} for i in range(14)]
    assert len(brainref.custom_list(many)) == brainref.MAX_CUSTOM
    import json

    monkeypatch.setenv("NEO_CUSTOM_PROVIDERS", json.dumps(good))
    assert [c.id for c in config.Settings().custom_providers] == ["custom:ollama", "custom:lm"]
    monkeypatch.setenv("NEO_CUSTOM_PROVIDERS", "{not json")
    assert config.Settings().custom_providers == []  # never a reason not to start


def test_new_keys_never_show_in_repr(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-TEST-K3Y")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-TEST-K3Y")
    s = config.Settings(custom_keys={"custom:x": "sk-TEST-K3Y"})
    assert s.openai_api_key == "sk-proj-TEST-K3Y" and s.openrouter_api_key == "sk-or-TEST-K3Y"
    assert "K3Y" not in repr(s) and "K3Y" not in str(s)
