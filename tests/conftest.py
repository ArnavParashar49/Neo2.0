"""Test-wide defaults: no model downloads, no GPU, no network, no real effects on the Mac (neo2/core/effects.py
refuses every one in NEO's own test runs: a test fakes the step it needs instead)."""

import os

# Never Laya (1.7 GB) or the on-device voice inside the suite, whatever the environment says: a test
# that wants one sets it for itself (monkeypatch.setenv) and fakes the load.
os.environ["NEO_REFLEX"] = "off"
os.environ["NEO_VOICE"] = "off"

from pathlib import Path

import pytest

from neo2.core import effects

effects.for_neo_tests()  # first: nothing real happens in this process, whatever a test sets or embeds

from neo2.core.config import Settings  # noqa: E402

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
    "_no_real_screen",
    "_no_task_grants",
]

# Hermetic: no test may read the developer's real keys or settings (~/.neo/.env, ./.env).
Settings.model_config["env_file"] = None
for _k in ("GEMINI_API_KEY", "GEMINI_API_KEYS", "GROQ_API_KEY", "ANTHROPIC_API_KEY", "NEO_DECIDE_MODES"):
    os.environ.pop(_k, None)

from neo2.core.decide import shadow as _shadow  # noqa: E402

_REAL_LATER = _shadow.later


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    """Every test gets its own ~/.neo (and notes folder): nothing a test does may touch the user's real
    data (memory, audit log, Laya's training examples, cooldowns, notes)."""
    from neo2.core import config

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
    from neo2.core.memory import embed

    monkeypatch.setattr(embed, "embed_texts", lambda texts: None)


@pytest.fixture(autouse=True)
def _fresh_learning_state():
    """Requests still settling in RAM, the correction memory and unsaved counters are
    process-wide: none of them may carry from one test into the next."""
    from neo2.core.reflex import learn, memory, outcomes

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

    from neo2.core import runtime

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
        from neo2.core.tools.computer import ax
    except ImportError:  # no macOS frameworks (the core install's portable run): nothing to fake
        return
    monkeypatch.setattr(ax, "is_trusted", lambda: True)


@pytest.fixture(autouse=True)
def _no_real_screen(monkeypatch):
    """The input tools look at the real screen around every click (what's under the point, a
    picture before and after) and ask macOS whether events may be posted: in the suite nothing is
    under any point, no picture can be taken, one 1512×982 display, and posting is allowed. A test
    that needs a world fakes these itself."""
    from neo2.core.tools.computer import ax, pointer, screen
    from neo2.core.tools.computer import input as inp

    monkeypatch.setattr(inp, "can_post", lambda: True)
    monkeypatch.setattr(ax, "hit", lambda x, y: ax.Hit())
    monkeypatch.setattr(ax, "window_owner_at", lambda x, y: ("", 0))  # (the window server: no window there)
    monkeypatch.setattr(ax, "scene", lambda pid: ax.Scene())
    monkeypatch.setattr(ax, "raise_window", lambda h: False)
    monkeypatch.setattr(ax, "focused_element_text", lambda: "")
    monkeypatch.setattr(screen, "sample_points", lambda pts, r=60: None)
    monkeypatch.setattr(screen, "displays", lambda: [screen.Display(1, 0.0, 0.0, 1512.0, 982.0)])
    monkeypatch.setattr(pointer, "LOOK_S", 0.0)
    monkeypatch.setattr(pointer, "LOOK_AGAIN_S", 0.0)
    monkeypatch.setattr(pointer, "SETTLE_S", 0.0)
    # The app NEO works in is per test (bringing a real one forward is refused: neo2/core/effects.py), and
    # no app runs a shell unless a test says so.
    from neo2.core.tools.computer import apps, focus

    monkeypatch.setattr(apps, "_target", None)
    monkeypatch.setattr(focus, "hosts_shell", lambda pid, depth=3: False)
    # The wait after an app opens reads the front app and its focus: in the suite, it is ready at once.
    from neo2.core.agent import settle

    monkeypatch.setattr(settle, "probe", lambda app, typing: settle.Ready(True, True, True))


@pytest.fixture(autouse=True)
def _no_task_grants():
    """A "yes for this task" never outlives the test that gave it."""
    from neo2.core.agent import grants

    grants.reset()
    yield
    grants.reset()


class FakeClock:
    """Time that passes only when something waits: ``now`` for the clock, ``sleep`` for the wait
    (every one recorded in ``sleeps``), ``advance`` for time that passes meanwhile."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += max(0.0, seconds)

    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def fake_clock(monkeypatch):
    """neo2.core.tools.computer.wait on a fake clock: ``until`` polls and waits in fake time (counted in
    ``fake_clock.sleeps``), and a test says when things happen with ``fake_clock.advance``."""
    wait = pytest.importorskip("neo2.core.tools.computer.wait")  # (the computer package needs macOS)
    clock = FakeClock()
    monkeypatch.setattr(wait, "_now", clock.now)
    monkeypatch.setattr(wait, "_sleep", clock.sleep)
    return clock


@pytest.fixture
def real_browser():
    """The one real check (NEO_TEST_RENDER=1, which skips the tests asking for this otherwise): a
    headless Chromium and the public page it reads, for that test only."""
    with effects.allow("browser", "network"):
        yield


@pytest.fixture
def shadow_runs(monkeypatch):
    """Dry runs of kinds in shadow mode run (their drivers faked by the test)."""
    monkeypatch.setattr(_shadow, "later", _REAL_LATER)


@pytest.fixture
def real_handlers(monkeypatch):
    """The task kinds' tools with their own handlers (another test may have swapped one)."""
    from neo2.core.agent.registry import registry
    from neo2.core.tools import load_all

    load_all()
    from neo2.core.tools import computer, tasks, web

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
    from neo2.core import config
    from neo2.core.tools import load_all

    def use(**modes: str) -> None:
        monkeypatch.setenv("NEO_DECIDE_MODES", ",".join(f"{k}={m}" for k, m in modes.items()))
        monkeypatch.setattr(config, "_settings", None)
        load_all()

    yield use
    monkeypatch.delenv("NEO_DECIDE_MODES", raising=False)
    monkeypatch.setattr(config, "_settings", None)
    load_all()
