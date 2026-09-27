"""AgentConfig: only what the host passes (or, opted in, its environment), checked up front, and
turned into LibrarySettings with an embedded agent's pins. Plus the provider factories and the
small public helpers."""

from __future__ import annotations

import sys

import pytest
from pydantic import ValidationError

import neo2
from neo import runtime
from neo2.sdk import agent as agent_mod
from neo2.testing import ScriptedBrain


def test_defaults_are_the_quiet_ones():
    c = neo2.AgentConfig()
    assert (c.tools, c.reflex, c.embeddings, c.memory, c.browser_headless, c.include_username) == (
        "web", "off", "off", False, True, False,
    )  # fmt: skip
    assert c.approval_timeout_s == 110.0 and c.max_concurrent_tasks == 2
    with pytest.raises(ValidationError):
        c.tools = "all"  # type: ignore[misc]  # frozen
    with pytest.raises(ValidationError):
        neo2.AgentConfig(confirm_destructive=False)  # type: ignore[call-arg]  # no such switch


def test_keys_never_show():
    c = neo2.AgentConfig(gemini_api_keys=("AIza-K3Y-1",), groq_api_key="gsk-K3Y-2", anthropic_api_key="sk-ant-K3Y-3")
    for text in (repr(c), str(c), c.model_dump_json()):
        assert "K3Y" not in text
    with pytest.raises(ValidationError) as e:
        neo2.AgentConfig(groq_api_key="gsk-K3Y-2", max_steps="many")  # type: ignore[arg-type]
    assert "K3Y" not in str(e.value)


@pytest.mark.parametrize(
    "kw,match",
    [
        ({"learn": True}, "learn isn't available"),
        ({"learn": False}, "learn isn't available"),
        ({"reflex": "laya"}, "reflex=\"laya\""),
        ({"reflex": "lite"}, "needs gemini_api_keys"),
        ({"approval_timeout_s": 0}, "approval_timeout_s"),
        ({"approval_timeout_s": 120}, "approval_timeout_s"),
        ({"memory": True}, "memory=True keeps it in data_dir"),
        ({"max_concurrent_tasks": 0}, "greater than or equal to 1"),
        ({"name": "  "}, "name must be"),
    ],
)
def test_bad_configs_fail_up_front(kw, match):
    with pytest.raises(ValidationError, match=match):
        neo2.AgentConfig(**kw)


def test_the_settings_an_agent_runs_with(tmp_path, home):
    c = neo2.AgentConfig(
        name="acme",
        tools=("web", "browser_*"),
        gemini_api_keys=("k1", " ", "k2", "k3"),
        groq_api_key="g",
        memory=neo2.MemoryConfig(path=tmp_path / "m.sqlite3", cloud="full", playbooks=False, summaries=True),
        persona="You are Shopbot.",
        instructions="Answer in French.",
        protected_apps=("MyApp",),
        extra_secret_paths=(tmp_path / "secrets",),
        embeddings="local",
        max_steps=5,
    )
    s = c.library_settings(tmp_path / "data")
    assert s.gemini_keys == ["k1", "k2", "k3"] and s.groq_api_key == "g"
    assert (s.memory_enabled, s.cloud_memory, s.playbooks, s.summaries) == (True, "full", False, True)
    assert s.memory_db == tmp_path / "m.sqlite3"
    assert s.persona == "You are Shopbot." and s.instructions == "Answer in French." and s.max_steps == 5
    assert s.protected_apps == ["MyApp"] and s.extra_secret_paths == [tmp_path / "secrets"]
    assert s.models_dir.is_relative_to(home) and "neo2" in s.models_dir.parts  # the user's cache folder
    assert s.confirm_destructive and not s.learn and s.learn_teacher == "off"
    assert set(s.decide_modes.values()) == {"legacy"} and s.notes_dir == tmp_path / "data" / "notes"
    plain = neo2.AgentConfig(data_dir=tmp_path / "d", memory=True).library_settings(tmp_path / "d")
    assert plain.memory_db == tmp_path / "d" / "memory.sqlite3" and plain.memory_path is None
    assert (plain.cloud_memory, plain.playbooks, plain.summaries) == ("profile", True, False)


def test_from_env_is_an_explicit_opt_in(tmp_path, monkeypatch):
    env = tmp_path / "neo.env"
    env.write_text("GEMINI_API_KEY=file-key\nNEO_BRAIN=claude\nNEO_MAX_STEPS=7\nANTHROPIC_API_KEY=ant\n")
    later = tmp_path / "later.env"
    later.write_text("NEO_MAX_STEPS=9\n")
    monkeypatch.setenv("GEMINI_API_KEYS", "env-a,env-b")
    monkeypatch.setenv("NEO_TOOLS", "web,browser_read")
    monkeypatch.setenv("NEO_CONFIRM_DESTRUCTIVE", "false")
    monkeypatch.setenv("NEO_INSTRUCTIONS", "Be brief.")
    assert neo2.AgentConfig().gemini_api_keys == ()  # built directly: nothing from the environment
    c = neo2.AgentConfig.from_env(env_files=[env, later], name="from-env")
    assert [k.get_secret_value() for k in c.gemini_api_keys] == ["file-key", "env-a", "env-b"]
    assert c.brain == "claude" and c.max_steps == 9 and c.tools == ("web", "browser_read")
    assert c.anthropic_api_key.get_secret_value() == "ant" and c.instructions == "Be brief."
    assert c.name == "from-env"
    assert c.library_settings(tmp_path / "d").confirm_destructive  # nothing turns confirmation off
    monkeypatch.setenv("NEO_MAX_STEPS", "3")
    assert neo2.AgentConfig.from_env(max_steps=4).max_steps == 4  # overrides win over the environment
    assert neo2.AgentConfig.from_env().max_steps == 3


def test_from_env_never_reads_neo_apps_settings_file(home):
    with pytest.raises(ValueError, match="NEO.app's own data"):
        neo2.AgentConfig.from_env(env_files=[home / ".neo" / ".env"])


def test_an_agent_needs_a_brain():
    with pytest.raises(ValueError, match="no brain"):
        neo2.AsyncAgent(neo2.AgentConfig())
    with pytest.raises(TypeError, match="isn't a provider"):
        neo2.AsyncAgent(neo2.AgentConfig(), provider=object())
    with pytest.raises(ValueError, match="approver"):
        neo2.AsyncAgent(neo2.AgentConfig(), provider=ScriptedBrain(), approver="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        neo2.AsyncAgent({"tools": "web"}, provider=ScriptedBrain())  # type: ignore[arg-type]


async def test_named_brains_come_from_the_configs_keys_only(monkeypatch):
    from neo.providers import gemini

    keys: list[str] = []

    class Client:
        def __init__(self, *, api_key, http_options=None):
            keys.append(api_key)

    monkeypatch.setattr(gemini.genai, "Client", Client)
    monkeypatch.setenv("GEMINI_API_KEY", "host-env-key")
    async with neo2.AsyncAgent(neo2.AgentConfig(gemini_api_keys=("cfg-key", "cfg-key-2"))) as agent:
        with runtime.bind(agent._rt.scope()):
            from neo.providers import brain, make

            p = make("gemini")
            assert isinstance(p, gemini.GeminiProvider) and keys == ["cfg-key", "cfg-key-2"]
            assert brain("agent").names[0] == "gemini"


async def test_host_providers_serve_every_purpose():
    main, fast = ScriptedBrain(name="main"), ScriptedBrain(name="fast")
    async with neo2.AsyncAgent(neo2.AgentConfig(), provider=[main], fast_provider=fast) as agent:
        with runtime.bind(agent._rt.scope()):
            from neo.providers import brain

            assert brain("agent").names == [main] and brain("vision").names == [main]
            assert brain("light").names == [main] and brain("fast").names == [fast]
    async with neo2.AsyncAgent(neo2.AgentConfig(), provider=main) as agent:
        with runtime.bind(agent._rt.scope()):
            from neo.providers import brain

            assert brain("fast").names == [main]  # fast defaults to the agent's provider


async def test_tool_selection(monkeypatch):
    with pytest.raises(ValueError, match="play_music"):
        neo2.AsyncAgent(neo2.AgentConfig(tools=("web", "play_music")), provider=ScriptedBrain())
    with pytest.raises(ValueError, match="unknown tool"):
        neo2.AsyncAgent(neo2.AgentConfig(tools="webz"), provider=ScriptedBrain())
    with pytest.raises(ValueError, match="memory on"):
        neo2.AsyncAgent(neo2.AgentConfig(tools=("web", "memory")), provider=ScriptedBrain())
    async with neo2.AsyncAgent(neo2.AgentConfig(tools="web", exclude_tools=("web_*",)), provider=ScriptedBrain()) as a:
        assert {t.name for t in a.tools} == {"weather", "clock"}
    monkeypatch.setattr(agent_mod, "_platform", lambda: "linux")
    with pytest.raises(neo2.UnsupportedPlatform):
        neo2.AsyncAgent(neo2.AgentConfig(tools="mac"), provider=ScriptedBrain())
    with pytest.raises(neo2.UnsupportedPlatform):
        neo2.AsyncAgent(neo2.AgentConfig(tools=("web", "shell")), provider=ScriptedBrain())


async def test_all_off_macos_is_the_portable_tools(monkeypatch, tmp_path):
    monkeypatch.setattr(agent_mod, "_platform", lambda: "linux")
    async with neo2.AsyncAgent(neo2.AgentConfig(tools="all", data_dir=tmp_path / "a", memory=True), provider=ScriptedBrain()) as a:
        names = {t.name for t in a.tools}
    assert {"web_search", "web_fetch", "weather", "clock", "memory"} <= names
    assert not names & {"shell", "click", "screenshot", "volume", "search_site", "mail_send"}
    assert all("macos" not in t.requires for t in a.tools)


async def test_the_browser_needs_playwright(monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a: None if n == "playwright" else real(n, *a))
    with pytest.raises(neo2.UnsupportedPlatform, match="playwright"):
        neo2.AsyncAgent(neo2.AgentConfig(tools="browser"), provider=ScriptedBrain())
    async with neo2.AsyncAgent(neo2.AgentConfig(tools="all", data_dir=None), provider=ScriptedBrain()) as a:
        assert not any(t.name.startswith("browser_") for t in a.tools)  # "all": left out, not refused


def test_provider_factories_read_no_settings(monkeypatch):
    runtime.set_library_mode()  # any settings() call with no agent bound now raises
    from neo2 import providers

    o = providers.OpenAICompatible(model="m", base_url="http://127.0.0.1:9/v1", api_key="k", vision=True)
    assert o.name == "openai" and o.model == "m" and o.supports_vision and not o._is_local
    g = providers.Groq("gk", model="x")
    assert g.name == "groq" and g.model == "x"
    gem = providers.Gemini(["a", "b"], model="gemini-x", fallback_models=["gemini-y"])
    assert gem.models == ["gemini-x", "gemini-y"] and len(gem._clients) == 2
    c = providers.Claude("ak", model="claude-x")
    assert c.name == "claude" and c.model == "claude-x"
    f = providers.Failover(o, g)
    assert f.name == "openai+groq"
    with pytest.raises(providers.ProviderError):
        providers.Gemini("")
    with pytest.raises(providers.ProviderError):
        providers.Groq("")
    with pytest.raises(ValueError):
        providers.Failover()


def test_a_missing_vendor_sdk_names_the_extra(monkeypatch):
    from neo2 import providers

    real = providers.importlib.import_module

    def no_openai(name, *a):
        if name in ("openai", "neo.providers.openai_compat"):
            raise ImportError("No module named 'openai'")
        return real(name, *a)

    monkeypatch.setattr(providers.importlib, "import_module", no_openai)
    with pytest.raises(providers.ProviderError, match=r"neo2\[openai\]"):
        providers.Groq("k")


def test_small_helpers():
    assert neo2.parse_yes_no("yes please") is True and neo2.parse_yes_no("no, wait") is False
    assert neo2.parse_yes_no("what's the weather") is None
    assert "Only the\n  user can approve" in neo2.SAFETY_RULES
    assert neo2.ToolContext is __import__("neo.agent.registry", fromlist=["x"]).ToolContext
    assert set(neo2.__all__) >= {"AsyncAgent", "Agent", "Approval", "host_tool", "events", "providers"}
    with pytest.raises(AttributeError):
        neo2.nope  # noqa: B018

def test_macos_helpers_never_prompt(monkeypatch):
    from neo import permissions
    from neo2 import macos

    monkeypatch.setattr(permissions, "missing", lambda: ["Screen Recording"])
    if sys.platform == "darwin":
        assert macos.permission_status() == {"accessibility": True, "screen_recording": False}
    else:
        assert not macos.available() and macos.permission_status() == {"accessibility": False, "screen_recording": False}


def test_request_permissions_asks_only_for_whats_missing(monkeypatch):
    from neo import permissions
    from neo2 import macos

    monkeypatch.setattr(permissions, "missing", lambda: [])
    assert macos.request_permissions() == []


async def test_a_persona_keeps_the_safety_rules():
    brain = ScriptedBrain(["Bonjour."])
    cfg = neo2.AgentConfig(tools="web", persona="You are Shopbot, Acme's assistant.", instructions="Réponds en français.")
    async with neo2.AsyncAgent(cfg, provider=brain) as agent:
        await agent.ask("hello")
    system = brain.systems[0]
    assert system.startswith("You are Shopbot, Acme's assistant.")
    assert "personal AI agent for their Mac" not in system  # NEO.app's identity paragraph is replaced
    for line in neo2.SAFETY_RULES.splitlines():
        assert line in system
    assert "Réponds en français." in system and "user " not in system.split("[CONTEXT]")[1]
