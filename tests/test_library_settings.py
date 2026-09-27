"""An embedded agent's settings come only from its host, and every accessor answers per agent."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest
from neo_lib import make_runtime
from pydantic import ValidationError

from neo import effects, runtime
from neo.config import DECIDE_DEFAULTS, LibrarySettings, Settings

ROOT = Path(__file__).resolve().parents[1]


# ---- where settings come from ----------------------------------------------------------------------


def test_the_environment_and_env_files_never_reach_an_embedded_agent(tmp_path):
    """A host process's own keys and NEO_* variables, and ~/.neo/.env, are NEO.app's — not an
    embedded agent's. Checked in a fresh interpreter whose HOME holds a real-looking ~/.neo/.env."""
    home = tmp_path / "home"
    (home / ".neo").mkdir(parents=True)
    (home / ".neo" / ".env").write_text("GEMINI_API_KEY=from-dotenv\nNEO_CONFIRM_DESTRUCTIVE=false\nNEO_BRAIN=claude\n")
    # NEO.app's own process: effects on (off, its Settings would read no .env file at all)
    env = {
        **{k: v for k, v in os.environ.items() if k not in ("NEO_NO_EFFECTS", "PYTEST_CURRENT_TEST")},
        "HOME": str(home),
        "NEO_DATA_DIR": str(tmp_path / "app_data"),
        "GEMINI_API_KEY": "from-env",
        "GROQ_API_KEY": "groq-env",
        "NEO_CONFIRM_DESTRUCTIVE": "false",
        "NEO_DECIDE_MODES": "web_search=on,add_event=on",
        "NEO_MEMORY_ENABLED": "true",
    }
    code = textwrap.dedent(
        f"""
        import json
        from neo.config import LibrarySettings, Settings
        app = Settings()
        lib = LibrarySettings(data_dir={str(tmp_path / "agent")!r})
        print(json.dumps({{
            "app": [app.gemini_api_key, app.confirm_destructive, app.brain, app.decide_modes.get("web_search")],
            "lib": [lib.gemini_api_key, lib.groq_api_key, lib.confirm_destructive, lib.brain,
                    lib.decide_modes.get("web_search"), lib.memory_enabled],
        }}))
        """
    )
    with effects.allow("process"):  # a Python child: refused unless meant, as here
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["app"] == ["from-env", False, "claude", "on"]  # NEO.app reads them, as always
    assert got["lib"] == ["", "", True, "gemini", "legacy", False]


def test_env_files_even_when_passed_are_ignored(tmp_path, monkeypatch):
    f = tmp_path / "x.env"
    f.write_text("GEMINI_API_KEY=from-file\nNEO_CONFIRM_DESTRUCTIVE=false\n")
    assert Settings(_env_file=f).gemini_api_key == ""  # effects off (the suite): no .env file, ever
    # NEO.app's mode (effects on, this thread only): NEO.app reads it, an embedded agent still doesn't
    monkeypatch.delenv("NEO_NO_EFFECTS", raising=False)
    token = effects._as_app.set(True)
    try:
        assert effects.on() and Settings(_env_file=f).gemini_api_key == "from-file"
        lib = LibrarySettings(data_dir=tmp_path / "a", _env_file=f)
    finally:
        effects._as_app.reset(token)
    assert lib.gemini_api_key == "" and lib.confirm_destructive


def test_a_checkout_reads_its_own_env_file_an_installed_package_does_not(tmp_path):
    from neo import config

    assert config._env_files(ROOT) == (config.USER_ENV, ROOT / ".env")
    assert config._env_files(tmp_path) == (config.USER_ENV,)  # site-packages has no pyproject.toml


# ---- what is pinned -----------------------------------------------------------------------------------


def test_confirmation_is_always_on(tmp_path):
    assert LibrarySettings(data_dir=tmp_path / "agent", confirm_destructive=False).confirm_destructive is True


def test_task_kinds_run_their_legacy_scripts_and_learning_is_off(tmp_path):
    s = LibrarySettings(
        data_dir=tmp_path / "a",
        decide_modes={"web_search": "on", "add_event": "shadow", "open_app": "on"},
        decide_kind_routing=True,
        learn=True,
        learn_teacher="cloud",
        notes_dir=tmp_path / "elsewhere",
    )
    assert set(DECIDE_DEFAULTS) <= set(s.decide_modes) and set(s.decide_modes.values()) == {"legacy"}
    assert s.decide_kind_routing is False and s.learn is False and s.learn_teacher == "off"
    assert s.notes_dir == (tmp_path / "a" / "notes").resolve()


def test_the_quiet_defaults(tmp_path):
    s = LibrarySettings(data_dir=tmp_path / "agent")
    assert (s.reflex, s.embeddings, s.memory_enabled, s.voice) == ("off", "off", False, "off")
    assert s.browser_headless and not s.include_username
    app = Settings()
    assert (app.reflex, app.embeddings, app.memory_enabled, app.browser_headless) == ("off", "local", True, False)
    assert app.include_username and app.playbooks and app.summaries and app.fast_paths
    assert app.persona is None and app.instructions == "" and app.extra_secret_paths == []


def test_data_dir_is_required_and_never_neo_home(tmp_path, monkeypatch):
    with pytest.raises(ValidationError, match="data_dir"):
        LibrarySettings()
    monkeypatch.setenv("HOME", str(tmp_path))
    for d in (tmp_path / ".neo", tmp_path / ".neo" / "agents" / "a", "~/.neo", "~/.NEO/a", "~/.Neo"):
        with pytest.raises(ValidationError, match="never uses NEO.app's own data"):
            LibrarySettings(data_dir=d)
    assert LibrarySettings(data_dir="~/myapp/neo").data_dir == (tmp_path / "myapp" / "neo").resolve()
    assert LibrarySettings(data_dir="~/.neo-agent").data_dir == (tmp_path / ".neo-agent").resolve()  # not inside


def test_typos_and_unknown_reflex_modes_are_refused(tmp_path):
    with pytest.raises(ValidationError, match="extra"):
        LibrarySettings(data_dir=tmp_path / "agent", confirm_destrutive=False)
    with pytest.raises(ValidationError):
        LibrarySettings(data_dir=tmp_path / "agent", reflex="laya")  # one process-wide model: NEO.app's
    with pytest.raises(ValidationError, match="Gemini key"):
        LibrarySettings(data_dir=tmp_path / "agent", reflex="lite")
    assert LibrarySettings(data_dir=tmp_path / "agent", reflex="lite", gemini_api_key="k").reflex == "lite"


def test_keys_never_show_in_repr(tmp_path):
    s = LibrarySettings(data_dir=tmp_path / "agent", gemini_api_key="sk-GEM", groq_api_key="sk-GROQ", anthropic_api_key="sk-ANT")
    s2 = LibrarySettings(data_dir=tmp_path / "agent", GEMINI_API_KEYS="sk-EXTRA")  # the env-style name works too
    for text in (repr(s), str(s), repr(s2)):
        assert "sk-" not in text
    assert s.gemini_keys == ["sk-GEM"] and s2.gemini_keys == ["sk-EXTRA"]


# ---- folders ------------------------------------------------------------------------------------------


def test_a_hosts_existing_folder_keeps_its_permissions(tmp_path):
    host = tmp_path / "host"
    host.mkdir(mode=0o755)
    host.chmod(0o755)
    s = LibrarySettings(data_dir=host)
    s.ensure_dirs()
    assert stat.S_IMODE(host.stat().st_mode) == 0o755
    assert not (host / "models").exists()  # no local embeddings: no model folder


def test_folders_it_creates_are_private(tmp_path):
    old = os.umask(0o022)
    try:
        s = LibrarySettings(data_dir=tmp_path / "a" / "b", embeddings="local", models_path=tmp_path / "m" / "n")
        s.ensure_dirs()
    finally:
        os.umask(old)
    for p in (tmp_path / "a", tmp_path / "a" / "b", tmp_path / "m", tmp_path / "m" / "n"):
        assert stat.S_IMODE(p.stat().st_mode) == 0o700, p
    assert s.models_dir == (tmp_path / "m" / "n").resolve()


# ---- per-agent accessors -------------------------------------------------------------------------


@pytest.fixture
def two(tmp_path):
    a = make_runtime(tmp_path / "a", "a", ["web_search", "clock"], memory_enabled=True, gemini_api_key="ka")
    b = make_runtime(tmp_path / "b", "b", ["web_search", "memory"], memory_enabled=True, gemini_api_key="kb")
    return a, b


def test_each_agent_gets_its_own_everything(two):
    from neo import privacy
    from neo.agent.early import early
    from neo.agent.registry import registry
    from neo.agent.toolselect import selector
    from neo.config import settings
    from neo.events import bus
    from neo.memory.store import store
    from neo.tools.browser import _browser

    seen = {}
    for rt in two:
        with runtime.bind(rt.scope()):
            seen[rt.name] = {
                "settings": settings(),
                "registry": registry(),
                "bus": bus(),
                "store": store(),
                "selector": selector(),
                "early": early(),
                "browser": _browser(),
            }
            assert seen[rt.name]["settings"] is rt.settings
            assert store().path == rt.settings.data_dir / "memory.sqlite3"
            assert selector().reg is rt.registry
            assert privacy.generation() == rt.generation
            # the same objects every time within one agent
            assert (bus(), store(), selector(), early(), _browser()) == tuple(
                seen[rt.name][k] for k in ("bus", "store", "selector", "early", "browser")
            )
    for k, v in seen["a"].items():
        assert v is not seen["b"][k], k
    two[0].generation += 1
    with runtime.bind(two[1].scope()):
        assert privacy.generation() == 0


def test_each_conversation_has_its_own_bus(two):
    from neo.events import bus

    a = two[0]
    with runtime.bind(a.scope("one")):
        b1 = bus()
    with runtime.bind(a.scope("two")):
        b2 = bus()
    assert b1 is not b2


def test_memory_writes_stay_in_each_agents_file(two):
    from neo.memory.store import store

    a, b = two
    with runtime.bind(a.scope()):
        store().remember("a's secret", "note")
    with runtime.bind(b.scope()):
        assert store().recall("secret") == []
        store().remember("b's thing", "note")
    with runtime.bind(a.scope()):
        assert [m.text for m in store().recall("thing")] == []
        assert [m.text for m in store().recall("secret")] == ["a's secret"]


def test_memory_off_means_no_store(tmp_path):
    from neo.memory.store import MemoryDisabled, store

    rt = make_runtime(tmp_path / "a")
    with runtime.bind(rt.scope()), pytest.raises(MemoryDisabled):
        store()
    assert not (tmp_path / "a" / "memory.sqlite3").exists()


def test_the_memory_file_is_private_whatever_the_umask(tmp_path):
    from neo.memory.store import store

    rt = make_runtime(tmp_path / "a", memory_enabled=True)
    old = os.umask(0o022)
    try:
        with runtime.bind(rt.scope()):
            store().remember("x", "note")
    finally:
        os.umask(old)
    assert stat.S_IMODE((tmp_path / "a" / "memory.sqlite3").stat().st_mode) == 0o600


def test_unbound_accessors_fail_closed_in_library_mode(two):
    from neo import privacy
    from neo.agent.early import early
    from neo.agent.registry import registry
    from neo.agent.toolselect import selector
    from neo.config import decide_mode, settings
    from neo.events import bus
    from neo.memory.store import store
    from neo.tools.browser import _browser

    for fn in (settings, registry, bus, store, selector, early, _browser, privacy.generation, lambda: decide_mode("x")):
        with pytest.raises(runtime.NoRuntimeError):
            fn()


def test_app_mode_still_uses_the_module_globals(monkeypatch):
    from neo import config, events, privacy
    from neo.agent import early as early_mod
    from neo.agent import toolselect
    from neo.memory import store as store_mod

    monkeypatch.setattr(store_mod, "_store", None)  # restored afterwards: no store leaks out
    for mod, name, factory in (
        (events, "_bus", events.EventBus),
        (early_mod, "_actor", early_mod.EarlyActor),
        (toolselect, "_selector", toolselect.ToolSelector),
    ):
        mine = factory()
        monkeypatch.setattr(mod, name, mine)
        assert getattr(mod, name.lstrip("_") if name != "_actor" else "early")() is mine
    monkeypatch.setattr(privacy, "_generation", 7)
    assert privacy.generation() == 7
    assert config.settings() is config._settings
    s = store_mod.store()
    assert s.path == config.settings().memory_db and store_mod.store() is s


def test_inside_an_agent_every_task_kind_is_legacy(two, monkeypatch):
    from neo import config
    from neo.agent import prompt

    monkeypatch.setenv("NEO_DECIDE_MODES", "web_search=on")
    monkeypatch.setattr(config, "_settings", None)
    runtime.set_library_mode(False)
    assert config.decide_mode("web_search") == "on"  # NEO.app, told to
    app_persona = prompt.persona()
    with runtime.bind(two[0].scope()):
        for kind in DECIDE_DEFAULTS:
            assert config.decide_mode(kind) == "legacy"
        assert prompt.persona() != app_persona and '"find X on Amazon"' in prompt.persona()


def test_an_agents_requests_never_reach_the_learning_ledger(two):
    from neo.reflex import learn, outcomes

    with runtime.bind(two[0].scope()):
        outcomes.begin("j1", "open safari")
        outcomes.count("tool_lane")
        outcomes.stop()
    assert len(outcomes.ledger()) == 0 and not learn._counts


def test_the_apps_erase_is_not_an_agents(two):
    from neo import privacy

    with runtime.bind(two[0].scope()), pytest.raises(RuntimeError, match="app's erase"):
        asyncio.run(privacy.forget_everything())


def test_the_browser_follows_the_agents_settings(two, monkeypatch):
    launched = []

    class _Page:
        url = "about:blank"

        def is_closed(self):
            return False

        def on(self, *a):
            pass

    class _Ctx:
        pages: list = []

        def set_default_timeout(self, ms):
            pass

        def on(self, *a):
            pass

        async def route(self, pattern, handler):  # an agent's browser: the web only (test_browser_web_only)
            pass

        async def new_page(self):
            return _Page()

        async def close(self):
            pass

    class _Chromium:
        async def launch_persistent_context(self, path, **kw):
            launched.append((path, kw["headless"]))
            return _Ctx()

    class _PW:
        chromium = _Chromium()

        async def start(self):
            return self

        async def stop(self):
            pass

    fake = types.ModuleType("playwright.async_api")
    fake.async_playwright = lambda: _PW()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "playwright.async_api", fake)
    from neo import effects
    from neo.tools.browser import _browser

    monkeypatch.setattr(effects, "_allowed", {"browser"})  # the Playwright above is the test's own fake

    a, _ = two
    headed = make_runtime(a.settings.data_dir.parent / "c", "c", browser_headless=False)
    for rt in (a, headed):
        with runtime.bind(rt.scope()):
            assert isinstance(asyncio.run(_browser().page()), _Page)
    assert launched == [
        (str(a.settings.data_dir / "browser"), True),
        (str(headed.settings.data_dir / "browser"), False),
    ]
    assert stat.S_IMODE((a.settings.data_dir / "browser").stat().st_mode) == 0o700


def test_weather_never_guesses_here_from_the_network_for_an_agent(two, monkeypatch):
    from neo.tools import weather

    calls = []

    class _Client:
        def get(self, url, **kw):
            calls.append(url)
            raise AssertionError("no network in tests")

    monkeypatch.setattr(weather, "_here", (1.0, 2.0, "Server City", 9e18))  # a cached app lookup
    with runtime.bind(two[0].scope()):
        assert weather._home(_Client()) is None
    assert calls == []
    runtime.set_library_mode(False)
    assert weather._home(_Client()) == (1.0, 2.0, "Server City")  # NEO.app: as before


# ---- the pins hold after construction ----------------------------------------------------------------


def test_the_pins_survive_assignment_and_copies(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    s = LibrarySettings(data_dir=tmp_path / "a")
    for field, value in (("confirm_destructive", False), ("decide_modes", {"web_search": "on"}), ("data_dir", tmp_path)):
        with pytest.raises(ValidationError):
            setattr(s, field, value)
    assert s.confirm_destructive and s.data_dir == (tmp_path / "a").resolve()
    c = s.model_copy(update={"persona": "Ada", "confirm_destructive": False, "decide_modes": {"web_search": "on"}})
    assert isinstance(c, LibrarySettings) and c.persona == "Ada"
    assert c.confirm_destructive and c.decide_modes["web_search"] == "legacy"  # a copy is validated too
    with pytest.raises(ValidationError, match="never uses NEO.app's own data"):
        s.model_copy(update={"data_dir": tmp_path / "home" / ".neo"})


def test_a_library_runtime_takes_only_pinned_library_settings(tmp_path, monkeypatch):
    from neo.agent.registry import Registry

    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    with pytest.raises(TypeError, match="LibrarySettings"):
        runtime.Runtime("app", Settings(), Registry())  # NEO.app's settings read the environment and ~/.neo
    for bad in (
        {"confirm_destructive": False},
        {"decide_modes": {"web_search": "on"}},
        {"learn": True},
        {"data_dir": Path.home()},
    ):
        s = LibrarySettings.model_construct(**{"data_dir": tmp_path / "a", **bad})  # skips every validator
        with pytest.raises(ValueError):
            runtime.Runtime("x", s, Registry())
    assert runtime.live_runtimes() == []


def test_a_library_runtime_turns_library_mode_on(tmp_path):
    import threading

    from neo import config
    from neo.agent import confirm
    from neo.agent.registry import Registry

    assert not runtime.library_mode()
    rt = runtime.Runtime("a", LibrarySettings(data_dir=tmp_path / "a"), Registry())
    assert runtime.library_mode()
    out: dict[str, str] = {}

    def lost_its_context():  # a raw thread: it carries no agent
        for what, fn in (("settings", config.settings), ("gate", confirm._gate)):
            try:
                fn()
                out[what] = "fell back to NEO.app's"
            except runtime.NoRuntimeError:
                out[what] = "closed"

    t = threading.Thread(target=lost_its_context)
    t.start()
    t.join()
    assert out == {"settings": "closed", "gate": "closed"} and config._settings is None
    rt.close()


def test_confirmation_cant_be_switched_off_inside_an_agent(tmp_path):
    from neo.agent import confirm

    rt = make_runtime(tmp_path / "a")
    object.__setattr__(rt.settings, "confirm_destructive", False)  # past every guard above
    with runtime.bind(rt.scope()):
        out = confirm.gated("shell", {"command": "rm -rf ~/x"}, "run rm")
        assert isinstance(out, str) and out.startswith(confirm.NEEDS_CONFIRM) and confirm.peek() is not None


def test_a_rejected_key_never_shows_in_the_error(tmp_path, monkeypatch):
    key = "AIzaSyTESTSECRET0000000000000000000000"
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    errors = []
    for kw in (
        {"data_dir": tmp_path / "k", "gemini_key": key},  # a typo: extra="forbid"
        {"data_dir": tmp_path / "home" / ".neo" / "x", "GEMINI_API_KEY": key},  # the model validator
        {"data_dir": tmp_path / "k", "reflex": "laya", "gemini_api_key": key},
    ):
        with pytest.raises(ValidationError) as e:
            LibrarySettings(**kw)
        errors.append(e.value)
    for e in errors:
        assert key[:10] not in str(e) and key[:10] not in repr(e) and key[:10] not in repr(e.errors()) and key[:10] not in e.json()
    assert "gemini_key" in str(errors[0]) and "never uses NEO.app's own data" in str(errors[1])


def test_an_unbound_erase_changes_nothing_before_it_fails(tmp_path):
    from neo import privacy

    rt = make_runtime(tmp_path / "a")
    before = privacy._generation
    with pytest.raises(runtime.NoRuntimeError):
        asyncio.run(privacy.forget_everything())
    assert privacy._generation == before
    rt.close()
