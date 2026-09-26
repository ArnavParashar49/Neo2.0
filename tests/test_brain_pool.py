"""Each agent has its own brains: built from its own keys, demoted on its own, or the host's."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

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
        # a purpose the host left out gets the host's agent providers, never a named brain
        assert prov.brain("offline").names == [main]


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


# ---- no brain the host didn't give ------------------------------------------------------------------


def test_an_agent_never_falls_back_to_a_local_server(tmp_path):
    """Whatever answers on 127.0.0.1:8080 of the host's machine is nobody's brain for an agent."""
    from neo.config import Settings

    a = make_runtime(tmp_path / "a", "a", gemini_api_key="k-a")
    with runtime.bind(a.scope()):
        for purpose in ("agent", "fast", "vision", "light", "offline"):
            assert "local" not in prov.brain(purpose).names, purpose
        assert prov.brain("offline").names == [] and prov.brain("agent").names == ["gemini", "groq"]
    b = make_runtime(tmp_path / "b", "b", offline_brain="local")  # the host asked for it
    with runtime.bind(b.scope()):
        assert prov.brain("agent").names == ["gemini", "groq", "local"]
    runtime.set_library_mode(False)
    assert Settings().offline_brain == "local"  # NEO.app: as before
    assert prov.brain("fast").names == ["groq", "gemini", "local"]


def test_the_hosts_providers_stand_in_for_every_purpose(tmp_path):
    a = make_runtime(tmp_path / "a", "a", gemini_api_key="k-a")
    host, fast = Fake("host"), Fake("fast")
    a.brains = prov.BrainPool(overrides={"agent": [host]})
    with runtime.bind(a.scope()):
        for purpose in ("agent", "fast", "vision", "light", "offline"):
            assert prov.brain(purpose).names == [host], purpose
        assert asyncio.run(prov.brain("fast").complete([])).text == "from host"
    a.brains = prov.BrainPool(overrides={"agent": [host], "fast": [fast]})
    with runtime.bind(a.scope()):
        assert prov.brain("fast").names == [fast] and prov.brain("light").names == [host]


def test_a_hosts_provider_built_before_any_agent_reads_no_settings(tmp_path):
    """Built first, with library mode still off: ClaudeProvider(api_key=...) used to build
    NEO.app's settings for its model — reading ~/.neo/.env and creating ~/.neo."""
    import os
    import subprocess
    import sys
    import textwrap

    home = tmp_path / "home"
    home.mkdir()
    env = {k: v for k, v in os.environ.items() if k not in ("NEO_DATA_DIR", "PYTEST_CURRENT_TEST")}
    env["HOME"] = str(home)
    code = textwrap.dedent(
        """
        import json, os
        from neo import config
        from neo.config import Settings
        from neo.providers.claude import ClaudeProvider
        from neo.providers.gemini import GeminiProvider

        c, g = ClaudeProvider(api_key="sk-host"), GeminiProvider(api_key="k-host")
        d = Settings.model_fields
        print(json.dumps([c.model == d["claude_model"].default, g.models[0] == d["gemini_model"].default,
                          g.models[1:] == d["gemini_fallback_models"].default, config._settings is None,
                          os.path.exists(os.path.expanduser("~/.neo"))]))
        """
    )
    root = Path(__file__).resolve().parents[1]
    out = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert json.loads(out.stdout.strip().splitlines()[-1]) == [True, True, True, True, False]
    assert not (home / ".neo").exists()


def test_the_teacher_still_uses_the_apps_gemini_model(monkeypatch, genai_clients):
    from neo import config
    from neo.reflex import teacher

    monkeypatch.setenv("GEMINI_API_KEY", "app-key")
    monkeypatch.setenv("NEO_GEMINI_MODEL", "gemini-custom")
    monkeypatch.setenv("NEO_GEMINI_FALLBACK_MODELS", '["gemini-fb"]')
    monkeypatch.setattr(config, "_settings", None)
    p = asyncio.run(teacher._provider("gemini"))
    assert p.models == ["gemini-custom", "gemini-fb"] and genai_clients == ["app-key"]
