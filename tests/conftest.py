"""Test-wide defaults: no model downloads, no GPU, no network, no real effects on the Mac (neo/effects.py
refuses every one under pytest: a test fakes the step it needs instead)."""

import os

os.environ.setdefault("NEO_REFLEX", "off")  # never load Laya (1.7 GB) inside the test suite
os.environ.setdefault("NEO_VOICE", "off")

from pathlib import Path

import pytest

from neo import effects
from neo.config import Settings

# A scratch conftest doing `from tests.conftest import *` must get the underscore fixtures too —
# without them a test runs against the user's real ~/.neo.
__all__ = [
    "_isolated_data_dir",
    "_never_the_real_data_dir",
    "_no_real_embeddings",
    "_fresh_learning_state",
    "_no_shadow_dry_runs",
    "_app_mode",
    "_accessibility_on",
    "_no_task_grants",
]

# Hermetic: no test may read the developer's real keys or settings (~/.neo/.env, ./.env).
Settings.model_config["env_file"] = None
for _k in ("GEMINI_API_KEY", "GEMINI_API_KEYS", "GROQ_API_KEY", "ANTHROPIC_API_KEY", "NEO_DECIDE_MODES"):
    os.environ.pop(_k, None)

if os.environ.get("NEO_TEST_RENDER") == "1":  # the one real check: a headless Chromium reading a public page
    effects.allow("browser", "network")

from neo.decide import shadow as _shadow  # noqa: E402

_REAL_LATER = _shadow.later


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    """Every test gets its own ~/.neo (and notes folder): nothing a test does may touch the user's real
    data (memory, audit log, Laya's training examples, cooldowns, notes)."""
    from neo import config

    monkeypatch.setenv("NEO_DATA_DIR", str(tmp_path / "neo_data"))
    monkeypatch.setenv("NEO_NOTES_DIR", str(tmp_path / "notes"))  # never the user's ~/Documents
    monkeypatch.setattr(config, "_settings", None)
    yield
    monkeypatch.setattr(config, "_settings", None)


@pytest.fixture(autouse=True)
def _never_the_real_data_dir(_isolated_data_dir):
    """The last line of defence: nothing runs (erase, retrain, record…) against the real ~/.neo.
    Read from the environment, not settings(): building those would already touch the dir."""
    data_dir = os.environ.get("NEO_DATA_DIR", "")
    if not data_dir or Path(data_dir).expanduser().resolve() == (Path.home() / ".neo").resolve():
        pytest.exit("refusing to run: the data dir is the real ~/.neo", returncode=3)
    yield


@pytest.fixture(autouse=True)
def _no_real_embeddings(monkeypatch):
    """Keyword-only memory by default; tests that need vectors install their own fake embedder."""
    from neo.memory import embed

    monkeypatch.setattr(embed, "embed_texts", lambda texts: None)


@pytest.fixture(autouse=True)
def _fresh_learning_state():
    """Requests still settling in RAM, the correction memory and unsaved counters are
    process-wide: none of them may carry from one test into the next."""
    from neo.reflex import learn, memory, outcomes

    outcomes.forget()
    memory.forget()
    learn._counts.clear()
    yield
    outcomes.forget()
    memory.forget()
    learn._counts.clear()


@pytest.fixture(autouse=True)
def _no_shadow_dry_runs(monkeypatch):
    """The task kinds ship in shadow mode: after a tool's legacy answer, the kind's driver works
    out in the background what it would have done, looking at the real apps, stores and web. In
    tests that never happens: the dry run is dropped unstarted. A test that faked everything a
    dry run touches asks for the `shadow_runs` fixture."""

    def drop(kind, facts, **extra):
        facts.close()

    monkeypatch.setattr(_shadow, "later", drop)


@pytest.fixture(autouse=True)
def _app_mode(monkeypatch):
    """Every test starts as NEO.app does: not embedded, no agent bound, no live agents. A test
    that embeds agents (library mode) never leaves that on for the next one."""
    import weakref

    from neo import runtime

    monkeypatch.setattr(runtime, "_library", False)
    monkeypatch.setattr(runtime, "_live", weakref.WeakSet())
    monkeypatch.setattr(runtime, "_retired", {})
    yield
    monkeypatch.setattr(runtime, "_library", False)


@pytest.fixture(autouse=True)
def _accessibility_on(monkeypatch):
    """The suite never depends on whether this Mac granted the test runner Accessibility: it is
    on, and a test that needs it off says so (monkeypatching ax.is_trusted)."""
    try:
        from neo.tools.computer import ax
    except ImportError:  # no macOS frameworks (the core install's portable run): nothing to fake
        return
    monkeypatch.setattr(ax, "is_trusted", lambda: True)


@pytest.fixture(autouse=True)
def _no_task_grants():
    """A "yes for this task" never outlives the test that gave it."""
    from neo.agent import grants

    grants.reset()
    yield
    grants.reset()


@pytest.fixture
def shadow_runs(monkeypatch):
    """Dry runs of kinds in shadow mode run (their drivers faked by the test)."""
    monkeypatch.setattr(_shadow, "later", _REAL_LATER)


@pytest.fixture
def real_handlers(monkeypatch):
    """The task kinds' tools with their own handlers (another test may have swapped one)."""
    from neo.agent.registry import registry
    from neo.tools import load_all

    load_all()
    from neo.tools import computer, tasks, web

    for mod, names in (
        (tasks, ("notes_create", "reminder_add", "calendar_add", "calendar_today", "mail_unread")),
        (computer, ("open_app", "hotkey", "page_read", "safari_read")),
        (web, ("web_search", "search_site")),
    ):
        for name in names:
            monkeypatch.setattr(registry().get(name), "handler", getattr(mod, name))


@pytest.fixture
def decide_modes(monkeypatch):
    """Set the task kinds' modes for one test (`decide_modes(add_event="on")`), with the tools shown
    as those modes want; the release defaults come back afterwards."""
    from neo import config
    from neo.tools import load_all

    def use(**modes: str) -> None:
        monkeypatch.setenv("NEO_DECIDE_MODES", ",".join(f"{k}={m}" for k, m in modes.items()))
        monkeypatch.setattr(config, "_settings", None)
        load_all()

    yield use
    monkeypatch.delenv("NEO_DECIDE_MODES", raising=False)
    monkeypatch.setattr(config, "_settings", None)
    load_all()
