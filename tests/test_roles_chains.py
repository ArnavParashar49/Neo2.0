"""The roles order the chains: Brain, Quick replies, Eyes, Backups and light_brain; cloud_eyes "off"
keeps every seeing fallback out; a provider added for "Brains & Keys" (OpenAI, OpenRouter, an
endpoint) never joins a chain on its own — only where the user gave it a role. Capabilities come from
the catalogue (describe), and the default settings rebuild today's chains (tests/golden pins those).
Every key is a fake; no brain here is ever asked over the network."""

from __future__ import annotations

import asyncio
import json
import sys

import pytest

from neo2.core import config
from neo2.core import providers as prov
from neo2.core.providers import catalog
from neo2.core.providers.base import (
    AllBrainsFailed,
    ImagePart,
    KeyRefused,
    Message,
    NoBrain,
    ProviderError,
    ToolCall,
    ToolResult,
    Turn,
    Unavailable,
    failure_words,
    spoken_name,
)

OLLAMA = {"id": "custom:ollama", "label": "Ollama", "template": "ollama", "base_url": "http://localhost:11434/v1",
          "model": "qwen3:14b"}  # fmt: skip
DEEPSEEK = {"id": "custom:deepseek", "label": "DeepSeek", "template": "deepseek", "base_url": "https://api.deepseek.com",
            "model": "deepseek-v4-pro", "context": 131072, "env": "DEEPSEEK_API_KEY"}  # fmt: skip


class Brain:
    def __init__(self, name, vision=True):
        self.name, self.model = name, name
        self.supports_vision, self.supports_tools = vision, True
        self.asked = 0

    def cooling_until(self):
        return 0.0

    async def complete(self, messages, **kw):
        self.asked += 1
        return Turn(text=self.name, sees=self.supports_vision)

    async def stream(self, messages, **kw):
        yield self.name


def _env(monkeypatch, **env):
    for k in ("GEMINI_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY", "CEREBRAS_API_KEY", "OPENAI_API_KEY",
              "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "NEO_BRAIN", "NEO_FAST_BRAIN", "NEO_OFFLINE_BRAIN",
              "NEO_EYES_BRAIN", "NEO_BACKUP_BRAINS", "NEO_LIGHT_BRAIN", "NEO_CLOUD_EYES", "NEO_CUSTOM_PROVIDERS",
              "NEO_OPENROUTER_MODEL"):  # fmt: skip
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(prov, "_cache", {})
    monkeypatch.setattr(prov, "_demoted", {})
    monkeypatch.setattr(prov, "_said_unavailable", set())


def _fakes(monkeypatch, *names, blind=()):
    made = {n: Brain(n, n not in blind) for n in names}

    def make(n):
        if n in made:
            return made[n]
        raise Unavailable(f"no {n}")

    monkeypatch.setattr(prov, "make", make)
    return made


def _picture_step() -> list[Message]:
    return [
        Message.user("play e4"),
        Message.assistant("", [ToolCall("c1", "screenshot", {})]),
        Message.tool([ToolResult("c1", "screenshot", "Screenshot attached.", images=[ImagePart(b"\x89PNG", "image/png")])]),
    ]


# ---- what a brain can do: from the catalogue ---------------------------------------------------------


def test_the_old_names_say_what_the_old_lists_said(monkeypatch):
    _env(monkeypatch)
    names = ("gemini", "gemini_lite", "groq", "local", "claude", "groq_vision", "cerebras", "cerebras_vision")
    assert {n for n in names if not catalog.describe(n).sees} == {"groq", "local", "cerebras"}  # _TEXT_ONLY
    assert {n for n in names if catalog.describe(n).eyes_only} == {"groq_vision"}  # _EYES_ONLY
    assert {n for n in names if catalog.describe(n).on_device} == {"local"}  # _ON_THIS_MAC
    assert prov.EYES == ("gemini", "gemini_lite", "claude", "groq_vision")
    assert catalog.describe("claude").model == "claude-opus-5"  # (Settings' default stays Opus 5)
    assert catalog.describe("claude@claude-haiku-5-5").model == "claude-haiku-5-5"


def test_new_brains_say_what_they_can_do(monkeypatch):
    _env(monkeypatch, NEO_CUSTOM_PROVIDERS=json.dumps([OLLAMA, {**DEEPSEEK, "tools": True, "vision": False}]))
    o = catalog.describe("custom:ollama")
    assert (o.provider, o.model, o.sees, o.tools, o.on_device, o.paid) == ("custom:ollama", "qwen3:14b", False, True, True, False)
    d = catalog.describe("custom:deepseek@deepseek-flash")
    assert (d.model, d.on_device, d.paid, d.known) == ("deepseek-flash", False, True, True)
    assert not catalog.describe("custom:gone").known and not catalog.describe("custom:gone").tools
    assert catalog.describe("openai").model == "gpt-6.1-sol" and catalog.describe("openai").sees
    assert not catalog.describe("openrouter").known  # (no model picked yet)
    assert catalog.describe("openrouter@anthropic/claude-opus-5.5").sees
    assert not catalog.describe("openrouter@deepseek/deepseek-v4-pro").sees  # (unknown: text, its Eyes see)
    bad = catalog.describe("claud")
    assert not bad.valid and not bad.sees and not bad.tools
    assert catalog.label_for("claude@claude-opus-5-5") == "Claude Opus 5.5"
    assert catalog.label_for("groq_vision") == "Qwen 3.8 27B on Groq"
    assert catalog.label_for("custom:deepseek") == "DeepSeek · deepseek-v4-pro"
    assert spoken_name("claude@claude-haiku-5-5") == "Claude Haiku 5.5" and spoken_name("openai@gpt-x") == "OpenAI"
    assert spoken_name("custom:ollama") == "Ollama" and spoken_name("gemini_lite") == "Gemini Lite"  # (the old ones as before)


# ---- building a reference -------------------------------------------------------------------------------


def test_each_reference_is_its_own_brain(monkeypatch):
    _env(monkeypatch, ANTHROPIC_API_KEY="sk-ant-TEST-0000", GEMINI_API_KEY="AIzaTEST-0000", GROQ_API_KEY="gsk_TEST")
    opus, haiku = prov.make("claude"), prov.make("claude@claude-haiku-5-5")
    assert opus is not haiku and opus.name == "claude" and haiku.name == "claude@claude-haiku-5-5"
    assert (opus.model, haiku.model) == ("claude-opus-5", "claude-haiku-5-5")
    assert prov.make("claude@claude-haiku-5-5") is haiku  # (its own cache entry)
    g = prov.make("gemini@gemini-3.7-flash")
    assert g.name == "gemini@gemini-3.7-flash" and g.model == "gemini-3.7-flash"
    q = prov.make("groq@llama-3.3-70b-versatile")
    assert q.name == "groq@llama-3.3-70b-versatile" and q.model == "llama-3.3-70b-versatile" and not q.supports_vision
    assert prov.make("groq_vision@qwen/other").model == "qwen/other"
    assert prov.make("gemini").name == "gemini" and prov.make("gemini_lite").name == "gemini_lite"


def test_a_reference_it_cant_build_is_skipped_never_swapped(monkeypatch):
    _env(monkeypatch, GEMINI_API_KEY="AIzaTEST-0000")
    for ref, words in (("claud", "isn't a brain NEO knows"), ("custom:gone", "custom:gone isn't set up"),
                       ("openai", "OPENAI_API_KEY is not set"), ("openrouter@x/y", "OPENROUTER_API_KEY is not set"),
                       ("", "isn't a brain")):  # fmt: skip
        with pytest.raises(Unavailable, match=words):
            prov._build(ref)
    with pytest.raises(Unavailable) as e:
        prov._build("sk-ant-TEST-0000-abcdefghijklmnop")  # a key pasted as a brain is never repeated
    assert "TEST" not in str(e.value)
    # A chain with a placeholder skips it (as a brain with no key) and goes on.
    _env(monkeypatch, GEMINI_API_KEY="AIzaTEST-0000", NEO_BRAIN="claud")
    c = prov.brain("agent")
    assert c.names[0] == "claud" and [p.name for p in c._providers()][0] != "claud"
    assert "claud" in [n for n, _ in c._missing]


def test_openai_waits_for_its_provider(monkeypatch):
    _env(monkeypatch, OPENAI_API_KEY="sk-proj-TEST-0000")
    monkeypatch.setitem(sys.modules, "neo2.core.providers.openai_responses", None)  # (not in this build)
    with pytest.raises(Unavailable, match="isn't in this build"):
        prov._build("openai@gpt-6-luna")


def test_openrouter_and_endpoints_build_as_compatible_brains(monkeypatch):
    _env(monkeypatch, OPENROUTER_API_KEY="sk-or-TEST-0000", NEO_CUSTOM_PROVIDERS=json.dumps([OLLAMA, DEEPSEEK]))
    with pytest.raises(Unavailable, match="no OpenRouter model"):
        prov._build("openrouter")
    r = prov._build("openrouter@anthropic/claude-opus-5.5")
    assert (r.name, r.model, r.supports_vision, r.on_device) == ("openrouter@anthropic/claude-opus-5.5", "anthropic/claude-opus-5.5", True, False)
    assert r._client.default_headers.get("X-Title") == "NEO" and "openrouter.ai" in str(r._client.base_url)
    o = prov._build("custom:ollama@llama3.3")
    assert (o.name, o.model, o.on_device, o.token_budget) == ("custom:ollama@llama3.3", "llama3.3", True, 32768)
    with pytest.raises(Unavailable, match="has no API key"):
        prov._build("custom:deepseek")  # a cloud endpoint needs its key
    _env(monkeypatch, DEEPSEEK_API_KEY="sk-TEST-deepseek-0000", NEO_CUSTOM_PROVIDERS=json.dumps([OLLAMA, DEEPSEEK]))
    d = prov._build("custom:deepseek")
    assert (d.name, d.model, d.token_budget, d.on_device) == ("custom:deepseek", "deepseek-v4-pro", 131072, False)
    no_tools = {**OLLAMA, "id": "custom:plain", "tools": False}
    _env(monkeypatch, NEO_CUSTOM_PROVIDERS=json.dumps([no_tools]))
    assert prov._build("custom:plain").supports_tools is False


# ---- the roles --------------------------------------------------------------------------------------------


def test_default_roles_are_todays_chains(monkeypatch):
    _env(monkeypatch)
    assert prov.brain("agent").names == ["gemini", "groq", "local"]
    assert prov.brain("fast").names == ["groq", "gemini", "local"]
    assert prov.brain("light").names == ["gemini_lite", "groq", "gemini", "local"]
    assert prov.brain("vision").names == ["gemini", "gemini_lite", "claude", "groq", "local"]
    assert prov.brain("agent").lead == [] and prov.brain("agent").eyes == ["gemini_lite", "claude", "groq_vision"]


def test_backups(monkeypatch):
    _env(monkeypatch, NEO_BRAIN="claude@claude-opus-5-5", NEO_FAST_BRAIN="claude@claude-haiku-5-5",
         NEO_BACKUP_BRAINS="gemini, groq")  # fmt: skip
    assert prov.brain("agent").names == ["claude@claude-opus-5-5", "gemini", "groq", "local"]
    assert prov.brain("fast").names == ["claude@claude-haiku-5-5", "claude@claude-opus-5-5", "gemini", "groq", "local"]
    assert prov.brain("light").names == ["gemini_lite", "claude@claude-haiku-5-5", "claude@claude-opus-5-5", "gemini", "groq", "local"]
    _env(monkeypatch, NEO_BRAIN="claude", NEO_BACKUP_BRAINS="none")
    assert prov.brain("agent").names == ["claude", "local"]  # no backups at all: not even Quick replies
    assert prov.brain("fast").names == ["groq", "claude", "local"]


def test_light_brain_empty_is_quick_replies(monkeypatch):
    _env(monkeypatch, NEO_BRAIN="claude", NEO_FAST_BRAIN="claude@claude-haiku-5-5", NEO_LIGHT_BRAIN="")
    assert prov.brain("light").names == ["claude@claude-haiku-5-5", "claude", "local"]  # never Gemini Lite quietly


def test_eyes_are_asked_first_for_a_picture_and_only_for_one(monkeypatch):
    _env(monkeypatch, NEO_EYES_BRAIN="claude@claude-opus-5-5")
    b = _fakes(monkeypatch, "gemini", "groq", "local", "gemini_lite", "claude", "groq_vision", "claude@claude-opus-5-5")
    agent = prov.brain("agent")
    assert agent.lead == ["claude@claude-opus-5-5"] and "claude@claude-opus-5-5" not in agent.names
    assert asyncio.run(agent.complete(_picture_step())).text == "claude@claude-opus-5-5"
    assert asyncio.run(agent.complete([Message.user("what's 2+2")])).text == "gemini"  # no picture: the Brain
    assert asyncio.run(prov.brain("fast").complete(_picture_step())).text == "claude@claude-opus-5-5"
    # The vision chain: Eyes first, for every step (they see and call tools).
    assert prov.brain("vision").names[:2] == ["claude@claude-opus-5-5", "gemini"]
    assert b["claude@claude-opus-5-5"].asked == 2


def test_an_eyes_only_brain_as_eyes_answers_only_pictures(monkeypatch):
    _env(monkeypatch, NEO_EYES_BRAIN="groq_vision", NEO_BRAIN="claude")
    _fakes(monkeypatch, "claude", "groq", "local", "gemini", "gemini_lite", "groq_vision")
    vision = prov.brain("vision")
    assert "groq_vision" not in vision.names and vision.lead == ["groq_vision"]
    assert asyncio.run(vision.complete([Message.user("list the windows")])).text == "claude"
    assert asyncio.run(vision.complete(_picture_step())).text == "groq_vision"


def test_eyes_that_dont_see_are_left_out(monkeypatch):
    _env(monkeypatch, NEO_EYES_BRAIN="groq")
    b = _fakes(monkeypatch, "gemini", "groq", "local", "gemini_lite", "claude", "groq_vision", blind=("groq", "local"))
    assert asyncio.run(prov.brain("agent").complete(_picture_step())).text == "gemini"
    assert b["groq"].asked == 0


def test_cloud_eyes_off_keeps_every_fallback_out(monkeypatch):
    _env(monkeypatch, GEMINI_API_KEY="AIzaTEST", ANTHROPIC_API_KEY="sk-ant-TEST", CEREBRAS_API_KEY="csk-TEST",
         GROQ_API_KEY="gsk_TEST", NEO_CLOUD_EYES="off")  # fmt: skip
    for purpose in ("agent", "fast", "light", "vision"):
        c = prov.brain(purpose)  # type: ignore[arg-type]
        assert c.eyes == [] and "claude" not in c.names and "groq_vision" not in c.names and "cerebras_vision" not in c.names
    assert prov.brain("vision").names[0] == "gemini"  # (the Brain the user picked still sees)
    assert "gemini_lite" not in prov.brain("vision").names
    assert prov.camera_order() == ["gemini"]  # only the roles' brains for the camera too


def test_new_providers_never_join_a_chain_on_their_own(monkeypatch):
    _env(monkeypatch, GEMINI_API_KEY="AIzaTEST", OPENAI_API_KEY="sk-proj-TEST", OPENROUTER_API_KEY="sk-or-TEST",
         NEO_OPENROUTER_MODEL="openai/gpt-6-luna", DEEPSEEK_API_KEY="sk-TEST",
         NEO_CUSTOM_PROVIDERS=json.dumps([OLLAMA, DEEPSEEK]), NEO_CLOUD_EYES="on")  # fmt: skip
    for purpose in ("agent", "fast", "light", "vision", "offline"):
        c = prov.brain(purpose)  # type: ignore[arg-type]
        every = [*c.names, *c.eyes, *c.lead]
        assert not [n for n in every if n.startswith(("openai", "openrouter", "custom:"))], purpose
    assert all(not n.startswith(("openai", "openrouter", "custom:")) for n in prov.camera_order())


def test_cerebras_finds_groq_by_its_provider(monkeypatch):
    _env(monkeypatch, CEREBRAS_API_KEY="csk-TEST", NEO_FAST_BRAIN="groq@llama-3.3-70b-versatile")
    assert prov.brain("fast").names == ["cerebras", "groq@llama-3.3-70b-versatile", "gemini", "local"]


def test_an_endpoint_on_this_mac_counts_as_local_first(monkeypatch):
    """custom:ollama as the Brain, only a Groq key: as NEO_BRAIN=local, no picture leaves the Mac."""
    _env(monkeypatch, GROQ_API_KEY="gsk_TEST", NEO_BRAIN="custom:ollama", NEO_CUSTOM_PROVIDERS=json.dumps([OLLAMA]))
    assert prov.brain("agent").eyes == []
    _env(monkeypatch, GROQ_API_KEY="gsk_TEST", NEO_BRAIN="custom:ollama", NEO_CUSTOM_PROVIDERS=json.dumps([OLLAMA]),
         NEO_EYES_BRAIN="claude")  # fmt: skip
    assert prov.brain("agent").eyes == ["gemini", "gemini_lite", "groq_vision"]  # named a cloud brain that sees


def test_camera_order(monkeypatch):
    _env(monkeypatch)
    assert prov.camera_order() == ["gemini", "cerebras_vision"]  # today's
    _env(monkeypatch, NEO_BRAIN="claude")
    assert prov.camera_order() == ["claude", "gemini", "cerebras_vision"]
    _env(monkeypatch, NEO_FAST_BRAIN="gemini_lite", NEO_EYES_BRAIN="openai@gpt-6-luna", NEO_BRAIN="groq")
    assert prov.camera_order() == ["gemini_lite", "openai@gpt-6-luna", "gemini", "cerebras_vision"]
    from neo2.core.vision import answer

    assert answer.brains() == ["gemini_lite", "openai@gpt-6-luna", "gemini"]  # (Cerebras: no key)
    assert answer.effort_for("gemini") == "minimal" and answer.effort_for("cerebras_vision") == "low"
    assert answer.effort_for("claude@claude-haiku-5-5") == "low" and answer.effort_for("openai@gpt-6-luna") == "low"


# ---- what a chain says, and forgets -------------------------------------------------------------------


def test_a_refused_key_rests_its_brain_for_an_hour(monkeypatch):
    import time

    _env(monkeypatch)
    c = prov.Chain(["claude", "gemini"])
    c._demote("claude", KeyRefused("claude: 401"))
    assert prov._demoted["claude"] - time.time() > 3500
    assert failure_words("claude", KeyRefused("x")) == "Claude refused its API key"
    assert NoBrain("x").kind == "no_brain" and issubclass(KeyRefused, Unavailable)


def test_any_key_counts_for_the_no_key_message(monkeypatch):
    _env(monkeypatch, OPENAI_API_KEY="sk-proj-TEST-0000", NEO_BRAIN="claude")
    _fakes(monkeypatch)
    with pytest.raises(AllBrainsFailed):
        asyncio.run(prov.brain("agent").complete([Message.user("hi")]))
    _env(monkeypatch)
    _fakes(monkeypatch)
    with pytest.raises(ProviderError, match="no API key set"):
        asyncio.run(prov.brain("agent").complete([Message.user("hi")]))


def test_forget_drops_one_providers_brains(monkeypatch):
    from neo2.core.providers import cerebras_hooks
    from neo2.core.vision import answer

    _env(monkeypatch)
    for n in ("claude", "claude@claude-haiku-5-5", "gemini", "groq", "cerebras"):
        prov._cache[n] = Brain(n)
        prov._demoted[n] = 1e12
        prov._said_unavailable.add((n, "Unavailable"))
    monkeypatch.setattr(answer, "_late_until", {"claude": 1e12, "gemini": 1e12})
    monkeypatch.setattr(answer, "_late_seen", {"claude": 1.0})
    assert sorted(prov.forget("anthropic")) == ["claude", "claude@claude-haiku-5-5"]
    assert sorted(prov._cache) == ["cerebras", "gemini", "groq"] and sorted(prov._demoted) == ["cerebras", "gemini", "groq"]
    assert {k[0] for k in prov._said_unavailable} == {"cerebras", "gemini", "groq"}
    assert answer._late_until == {"gemini": 1e12} and answer._late_seen == {}
    monkeypatch.setitem(cerebras_hooks._accounts, prov._app_pool, object())
    prov.forget("cerebras")
    assert prov._app_pool not in cerebras_hooks._accounts and "cerebras" not in prov._cache
    prov.forget()
    assert prov._cache == {} and prov._demoted == {} and not prov._said_unavailable


def test_the_teacher_takes_any_set_up_reference(monkeypatch):
    from neo2.core.reflex import teacher

    _env(monkeypatch, ANTHROPIC_API_KEY="sk-ant-TEST")
    assert teacher.provider_of({"brain": "claude@claude-haiku-5-5"}) == "claude@claude-haiku-5-5"
    assert teacher.provider_of({"brain": "openai@gpt-6-luna"}) is None  # no OpenAI key: never sent anywhere
    assert teacher.provider_of({"brain": "gemini_lite"}) == "gemini" and teacher.provider_of({"brain": "cerebras"}) is None
    assert teacher.provider_of({"brain": "claud"}) is None


def test_a_key_pasted_as_a_brain_is_never_said_or_logged(monkeypatch, caplog):
    import logging

    key = "sk-ant-TEST-0000-abcdefghijklmnop"
    # (an OpenAI key, so the stop names each brain rather than "no API key set")
    _env(monkeypatch, NEO_BRAIN=key, NEO_FAST_BRAIN="claude", NEO_OFFLINE_BRAIN="", OPENAI_API_KEY="sk-proj-X")
    caplog.set_level(logging.INFO)
    c = prov.brain("agent")
    with pytest.raises(AllBrainsFailed) as stop:
        asyncio.run(c.complete([Message.user("hi")]))
    assert "doesn't know" in str(stop.value) and "ANTHROPIC_API_KEY is not set" in str(stop.value)
    said = [str(stop.value), getattr(stop.value, "spoken", lambda: "")(), c.name, caplog.text]
    assert all("TEST" not in t for t in said)
    assert spoken_name(key) == "a brain NEO doesn't know" and spoken_name("hosts") == "Hosts"
