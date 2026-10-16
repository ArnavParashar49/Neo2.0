"""Daily caps for the paid providers (decision 4): settings.daily_budgets, dollars a day per provider.

A capped provider's answers are counted at the catalogue's prices (a model with no price: that
provider's dearest; a provider with none listed: the catalogue's dearest), 80% is said once, and once
it's spent the brain is skipped until midnight (QuotaExhausted, in words the user hears). Only a paid
key saved in the Brains & Keys window gets a cap filled in ($2); a key from .env or the environment
stays uncapped; Cerebras keeps its own budget. Ledgers live in the test's data dir; keys are fakes."""

from __future__ import annotations

import json
from typing import Any

import pytest

from neo2.core import config
from neo2.core import providers as prov
from neo2.core.providers import spend
from neo2.core.providers.base import Message, QuotaExhausted, Turn, failure_words
from neo2.core.providers.claude import ClaudeProvider

KEY = "sk-ant-TEST-0000-abcdefghijklmnop"


class Usage:
    def __init__(self, i: int, o: int) -> None:
        self.input_tokens, self.output_tokens = i, o


class Block:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)

    def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
        return dict(self.__dict__)


class Resp:
    def __init__(self, usage: Usage) -> None:
        self.content, self.stop_reason, self.usage = [Block(type="text", text="ok")], "end_turn", usage
        self.input_transformations = []


class Fake:
    def __init__(self, usage: Usage) -> None:
        self.asked = 0
        me = self

        class Messages:
            async def create(self, **kw):
                me.asked += 1
                return Resp(usage)

        class Beta:
            messages = Messages()

        self.beta = Beta()


@pytest.fixture
def notes(monkeypatch):
    said: list[str] = []

    class Bus:
        async def note(self, text):
            said.append(text)

    from neo2.core import events

    monkeypatch.setattr(events, "bus", lambda: Bus())
    return said


def _settings(monkeypatch, **env: str) -> config.Settings:
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "NEO_DAILY_BUDGETS", "CEREBRAS_API_KEY",
              "NEO_CEREBRAS_DAILY_BUDGET"):  # fmt: skip
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(config, "_settings", None)
    return config.settings()


def test_only_a_paid_key_saved_in_the_window_gets_a_cap_filled_in():
    assert spend.prefill("anthropic", "saved") == spend.DEFAULT_CAP == 2.0
    assert spend.prefill("openai", "saved") == 2.0 and spend.prefill("openrouter", "saved") == 2.0
    assert spend.prefill("custom:deepseek", "saved") == 2.0
    for pid, source in (("anthropic", "dotenv"), ("anthropic", "env"), ("anthropic", "checkout_env"),
                        ("gemini", "saved"), ("groq", "saved"), ("local", "saved"), ("cerebras", "saved")):  # fmt: skip
        assert spend.prefill(pid, source) is None, (pid, source)
    assert spend.prefill("custom:ollama", "saved", on_device=True) is None


def test_no_entry_no_cap(monkeypatch):
    s = _settings(monkeypatch, ANTHROPIC_API_KEY=KEY)  # a key from the environment: uncapped
    assert spend.cap_for("anthropic", s) is None
    assert ClaudeProvider()._cap is None
    s = _settings(monkeypatch, ANTHROPIC_API_KEY=KEY, NEO_DAILY_BUDGETS='{"claude": 2}')
    cap = spend.cap_for("anthropic", s)
    assert cap is not None and cap.ledger.daily_budget == 2.0 and cap.label == "Claude"
    assert cap.ledger.path == s.data_dir / "spend-anthropic.json"
    assert spend.cap_for("gemini", s) is None  # (only paid providers are capped here)


async def test_a_capped_brain_says_80_percent_then_is_skipped_until_midnight(monkeypatch, notes):
    s = _settings(monkeypatch, ANTHROPIC_API_KEY=KEY, NEO_DAILY_BUDGETS="anthropic=0.01")
    p = ClaudeProvider("claude-opus-5-5")
    p._client = fake = Fake(Usage(300, 100))  # 300 × $4 + 100 × $20 per M = $0.0032 an answer
    for _ in range(2):
        await p.complete([Message.user("hi")])
    assert notes == []
    await p.complete([Message.user("hi")])  # $0.0096: past 80%
    assert notes == ["Claude has used $0.0096 of today's $0.01 budget"]
    await p.complete([Message.user("hi")])  # $0.0128: spent
    assert notes[-1] == "Claude's daily budget of $0.01 is used up; I'll skip it until tomorrow"
    with pytest.raises(QuotaExhausted) as e:
        await p.complete([Message.user("hi")])
    assert fake.asked == 4  # not asked once it's spent
    assert failure_words(p.name, e.value) == "Claude's daily budget of $0.01 is used up"
    assert e.value.until == p._cap.ledger.midnight()
    book = json.loads((s.data_dir / "spend-anthropic.json").read_text())
    assert book["tokens"]["claude-opus-5-5"] == [1200, 400] and KEY not in json.dumps(book) and "hi" not in json.dumps(book)
    assert notes.count("Claude's daily budget of $0.01 is used up; I'll skip it until tomorrow") == 1  # once a day


async def test_the_chain_goes_on_to_the_next_brain_once_one_is_spent(monkeypatch, notes):
    _settings(monkeypatch, ANTHROPIC_API_KEY=KEY, NEO_DAILY_BUDGETS='{"anthropic": 0}')
    c = ClaudeProvider("claude-opus-5-5")
    c._client = Fake(Usage(1, 1))

    class Other:
        name, model, supports_vision, supports_tools = "groq", "m", False, True

        async def complete(self, messages, **kw):
            return Turn(text="from groq")

    monkeypatch.setattr(prov, "_cache", {"claude": c, "groq": Other()})
    monkeypatch.setattr(prov, "_demoted", {})
    turn = await prov.Chain(["claude", "groq"]).complete([Message.user("hi")])
    assert turn.text == "from groq" and c._client.asked == 0
    assert prov._demoted["claude"] >= c._cap.ledger.midnight()  # benched until the day is over


def test_prices_come_from_the_catalogue(tmp_path):
    cap = spend.Cap("anthropic", 1.0, path=tmp_path / "a.json", label="Claude")
    assert cap.ledger.cost("claude-haiku-5-5", 1_000_000, 1_000_000) == pytest.approx(0.60)
    assert cap.ledger.cost("claude-opus-9", 1_000_000, 0) == pytest.approx(10.0)  # unknown: Claude's dearest (Fable)
    o = spend.Cap("openrouter", 1.0, path=None, label="OpenRouter")
    assert o.ledger.cost("deepseek/deepseek-v4-pro", 0, 1_000_000) == pytest.approx(50.0)  # none listed: the dearest known


def test_a_host_s_own_claude_is_never_capped(monkeypatch):
    _settings(monkeypatch, NEO_DAILY_BUDGETS='{"anthropic": 0}')
    assert ClaudeProvider("claude-opus-5-5", api_key=KEY)._cap is None


def test_cerebras_keeps_its_own_budget_unless_one_was_set_in_brains_and_keys(monkeypatch):
    from neo2.core.providers import cerebras, cerebras_hooks

    s = _settings(monkeypatch, CEREBRAS_API_KEY="csk-TEST-0000")
    assert cerebras.conf(s).daily_budget == 0.5 and cerebras_hooks.enabled(s)
    s = _settings(monkeypatch, CEREBRAS_API_KEY="csk-TEST-0000", NEO_CEREBRAS_DAILY_BUDGET="1.25")
    assert cerebras.conf(s).daily_budget == 1.25
    s = _settings(monkeypatch, CEREBRAS_API_KEY="csk-TEST-0000", NEO_CEREBRAS_DAILY_BUDGET="1.25", NEO_DAILY_BUDGETS="cerebras=0")
    assert cerebras.conf(s).daily_budget == 0.0 and not cerebras_hooks.enabled(s)
