"""Each agent has its own brains: built from its own keys, demoted on its own, or the host's."""

from __future__ import annotations

import asyncio

import pytest
from neo_lib import make_runtime

from neo import providers as prov
from neo import runtime
from neo.providers.base import ProviderError, RateLimited, Turn


class Fake:
    supports_vision = False
    supports_tools = True

    def __init__(self, name: str, *, fail: type[Exception] | None = None, model: str = "m1") -> None:
        self.name, self.fail, self.model = name, fail, model
        self.calls = 0

    async def complete(self, messages, **kw):
        self.calls += 1
        if self.fail:
            raise self.fail(f"{self.name} down")
        return Turn(text=f"from {self.name}")

    async def stream(self, messages, **kw):
        self.calls += 1
        if self.fail:
            raise self.fail(f"{self.name} down")
        yield f"from {self.name}"


@pytest.fixture
def genai_clients(monkeypatch):
    gem = pytest.importorskip("neo.providers.gemini")
    made: list[str] = []

    class Client:
        def __init__(self, api_key, http_options=None):
            made.append(api_key)
            self.api_key = api_key

    monkeypatch.setattr(gem.genai, "Client", Client)
    return made


def test_each_agent_builds_its_brains_from_its_own_keys(tmp_path, genai_clients, monkeypatch):
    monkeypatch.setattr(prov, "_cache", {})
    a = make_runtime(tmp_path / "a", "a", gemini_api_key="key-a")
    b = make_runtime(tmp_path / "b", "b", gemini_api_key="key-b", gemini_api_keys="key-b2")
    got = {}
    for rt in (a, b):
        with runtime.bind(rt.scope()):
            got[rt.name] = prov.make("gemini")
            assert prov.make("gemini") is got[rt.name]  # cached, in the agent's pool
    assert got["a"] is not got["b"]
    assert [c.api_key for c in got["a"]._clients] == ["key-a"]
    assert [c.api_key for c in got["b"]._clients] == ["key-b", "key-b2"]
    assert prov._cache == {}  # NEO.app's pool never saw them


def test_an_agent_without_keys_has_no_named_brains(tmp_path):
    a = make_runtime(tmp_path / "a", "a")
    with runtime.bind(a.scope()):
        with pytest.raises(ProviderError, match="GEMINI_API_KEY"):
            prov.make("gemini")
        with pytest.raises(ProviderError, match="no API key"):
            asyncio.run(prov.brain("agent").complete([]))


def test_demotion_is_per_agent(tmp_path, monkeypatch):
    fakes: dict[str, dict[str, Fake]] = {}

    def build(name):
        rt = runtime.current_runtime()
        f = Fake(name, fail=ProviderError if (rt.name, name) == ("a", "gemini") else None)
        fakes.setdefault(rt.name, {})[name] = f
        return f

    monkeypatch.setattr(prov, "_build", build)
    monkeypatch.setattr(prov, "_demoted", {})
    a = make_runtime(tmp_path / "a", "a")
    b = make_runtime(tmp_path / "b", "b")
    with runtime.bind(a.scope()):
        assert asyncio.run(prov.Chain(["gemini", "groq"]).complete([])).text == "from groq"
        assert prov.Chain(["gemini", "groq"])._order() == ["groq", "gemini"]  # gemini went to the back
    with runtime.bind(b.scope()):
        assert prov.Chain(["gemini", "groq"])._order() == ["gemini", "groq"]  # B's order is its own
        assert asyncio.run(prov.Chain(["gemini", "groq"]).complete([])).text == "from gemini"
    assert prov._demoted == {}  # nor NEO.app's


def test_the_hosts_providers_stand_in_for_the_named_brains(tmp_path):
    main, quick = Fake("host-main", model="gpt-x"), Fake("host-quick")
    a = make_runtime(tmp_path / "a", "a")
    a.brains = prov.BrainPool(overrides={"agent": [main], "vision": [main], "light": [main], "fast": [quick]})
    with runtime.bind(a.scope()):
        chain = prov.brain("agent")
        assert chain.names == [main] and chain.name == "host-main"
        assert asyncio.run(chain.complete([])).text == "from host-main"
        assert (chain.last_used, chain.last_model) == ("host-main", "gpt-x")
        assert prov.brain("light").names == [main] and prov.brain("vision").names == [main]
        assert not prov.brain("vision").sticky
        assert asyncio.run(prov.brain("fast").complete([])).text == "from host-quick"

        async def chat():
            return [c async for c in prov.brain("fast").stream([])]

        assert asyncio.run(chat()) == ["from host-quick"]
        # a purpose the host left out falls back to the agent's own (keyless) named brains
        assert prov.brain("offline").names == ["local"]


def test_a_chain_of_host_providers_fails_over(tmp_path):
    limited, ok = Fake("limited", fail=RateLimited), Fake("ok")
    a = make_runtime(tmp_path / "a", "a")
    a.brains = prov.BrainPool(overrides={"agent": [limited, ok]})
    with runtime.bind(a.scope()):
        assert asyncio.run(prov.brain("agent").complete([])).text == "from ok"
        assert prov.brain("agent")._order() == [ok, limited]  # demoted by name, in A's pool
        asyncio.run(prov.brain("agent").complete([]))
    assert limited.calls == 1 and ok.calls == 2


def test_when_every_host_provider_fails_it_says_so(tmp_path):
    a = make_runtime(tmp_path / "a", "a")
    a.brains = prov.BrainPool(overrides={"agent": [Fake("x", fail=ProviderError)]})
    with runtime.bind(a.scope()), pytest.raises(ProviderError, match="all brains failed"):
        asyncio.run(prov.brain("agent").complete([]))


def test_constructors_given_everything_read_no_settings(genai_clients):
    """Hosts build providers before any agent exists: that must not reach NEO.app's settings."""
    from neo.providers.claude import ClaudeProvider
    from neo.providers.gemini import GeminiProvider
    from neo.providers.openai_compat import OpenAICompatProvider

    runtime.set_library_mode()  # an unbound settings() would raise NoRuntimeError
    g = GeminiProvider("gemini-x", keys=["k1", "k1", "k2"], fallback_models=["gemini-y"])
    assert genai_clients == ["k1", "k2"] and g.models == ["gemini-x", "gemini-y"]
    c = ClaudeProvider("claude-x", api_key="sk")
    assert c.model == "claude-x"

    class Client:
        base_url = "http://localhost:11434/v1/"

    o = OpenAICompatProvider(name="ollama", model="llama", client=Client())  # type: ignore[arg-type]
    assert o._client.__class__ is Client and not o._is_local  # a host's client: no NEO preflight
    with pytest.raises(runtime.NoRuntimeError):
        GeminiProvider()  # nothing passed: it would have to read settings


def test_gemini_quota_memory_goes_to_the_agents_data_dir(tmp_path, genai_clients):
    from neo.providers import gemini as gem

    a = make_runtime(tmp_path / "a", "a")
    make_runtime(tmp_path / "b", "b")
    p = gem.GeminiProvider("gemini-x", keys=["k"], fallback_models=[])
    p._exhausted[("gemini-x", 0)] = 9e18
    with runtime.bind(a.scope()):
        p._save_cooldowns()
    assert (tmp_path / "a" / gem._COOLDOWN_FILE).exists() and not (tmp_path / "b" / gem._COOLDOWN_FILE).exists()


def test_neo_apps_pool_is_the_module_cache(monkeypatch, genai_clients):
    monkeypatch.setattr(prov, "_cache", {})
    monkeypatch.setenv("GEMINI_API_KEY", "app-key")
    from neo import config

    monkeypatch.setattr(config, "_settings", None)
    p = prov.make("gemini")
    assert prov._cache == {"gemini": p} and prov.make("gemini") is p
    assert prov.pool() is prov._app_pool
