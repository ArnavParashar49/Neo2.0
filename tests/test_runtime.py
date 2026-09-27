"""The per-agent runtime: binding, how it travels with spawned work, and failing closed."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from neo2.core import runtime
from neo2.core.agent.registry import Registry


def _rt(name: str, data_dir: str = "/tmp/x", **kw) -> runtime.Runtime:
    """The bare scaffolding (binding, spawned work, lazy members): not a library agent, whose
    settings and data dir are checked (test_library_settings, test_agent_data_dirs)."""
    s = SimpleNamespace(data_dir=Path(data_dir), extra_secret_paths=[])
    return runtime.Runtime(name, s, Registry(), **{"library": False, **kw})  # type: ignore[arg-type]


def test_binding_nests_and_unbinds():
    a, b = _rt("a"), _rt("b")
    assert runtime.current_scope() is None and runtime.current_runtime() is None
    with runtime.bind(a.scope()) as sa:
        assert runtime.current_runtime() is a and runtime.current_scope() is sa
        with runtime.bind(b.scope("thread-2")) as sb:
            assert runtime.current_runtime() is b and sb.conversation == "thread-2"
        assert runtime.current_runtime() is a
    assert runtime.current_scope() is None


def test_a_conversation_keeps_its_scope():
    a = _rt("a")
    assert a.scope("x") is a.scope("x") and a.scope("x") is not a.scope("y")
    assert a.scope().runtime is a and a.scope().conversation == "default"


def test_unbinds_even_when_the_body_raises():
    a = _rt("a")
    with pytest.raises(ValueError), runtime.bind(a.scope()):
        raise ValueError
    assert runtime.current_scope() is None


async def test_the_scope_travels_with_every_way_neo_spawns_work():
    a = _rt("a")
    seen: dict[str, object] = {}

    async def child(tag: str) -> None:
        seen[tag] = runtime.current_runtime()

    with runtime.bind(a.scope()):
        await asyncio.create_task(child("create_task"))  # session._spawn, session._adopt
        await asyncio.gather(child("gather1"), child("gather2"))  # loop._run_calls
        seen["to_thread"] = await asyncio.to_thread(runtime.current_runtime)  # session's store calls
        fired = asyncio.Event()

        def later() -> None:
            seen["call_later"] = runtime.current_runtime()
            fired.set()

        asyncio.get_running_loop().call_later(0.01, later)  # confirm.stage's expiry check
    await asyncio.wait_for(fired.wait(), 2)
    assert all(v is a for v in seen.values()) and len(seen) == 5


async def test_concurrent_tasks_see_their_own_agent():
    a, b = _rt("a"), _rt("b")

    async def who(rt: runtime.Runtime) -> list:
        with runtime.bind(rt.scope()):
            out = []
            for _ in range(5):
                await asyncio.sleep(0)
                out.append(runtime.current_runtime())
            return out

    ra, rb = await asyncio.gather(who(a), who(b))
    assert ra == [a] * 5 and rb == [b] * 5


def test_a_plain_thread_does_not_carry_the_agent():
    a = _rt("a")
    seen = []
    with runtime.bind(a.scope()):
        t = threading.Thread(target=lambda: seen.append(runtime.current_runtime()))
        t.start()
        t.join()
    assert seen == [None]


def test_library_mode_fails_closed_only_when_unbound(monkeypatch):
    monkeypatch.setattr(runtime, "_library", False)
    runtime.check_unbound("settings()")  # the app: falls back, as always
    runtime.set_library_mode()
    assert runtime.library_mode()
    with pytest.raises(runtime.NoRuntimeError, match="settings"):
        runtime.check_unbound("settings()")
    with runtime.bind(_rt("a").scope()):
        runtime.check_unbound("settings()")  # bound: fine


def test_live_runtimes_and_their_secret_roots(tmp_path, monkeypatch):
    from neo2.core import config

    monkeypatch.setattr(config, "_settings", None)
    a = _rt("a", str(tmp_path / "a"), secret_paths=(tmp_path / "keys",))
    a.settings.extra_secret_paths = [tmp_path / "vault"]
    b = _rt("b", str(tmp_path / "b"))
    assert {a, b} <= set(runtime.live_runtimes())
    roots = runtime.secret_roots()
    for p in (tmp_path / "a", tmp_path / "b", tmp_path / "keys", tmp_path / "vault"):
        assert p in roots
    assert config._settings is None  # never builds the app's settings to find its data dir
    b.close()
    assert tmp_path / "b" not in runtime.secret_roots()


def test_the_apps_data_dir_is_a_root_once_built(tmp_path, monkeypatch):
    from neo2.core import config

    monkeypatch.setenv("NEO_DATA_DIR", str(tmp_path / "custom"))
    monkeypatch.setattr(config, "_settings", None)
    assert (tmp_path / "custom").resolve() not in runtime.secret_roots()
    config.settings()
    assert (tmp_path / "custom").resolve() in runtime.secret_roots()


def test_lazy_members_are_built_once():
    a = _rt("a")
    made = []
    for _ in range(3):
        a.get_or_create("early", lambda: made.append(1) or object())
    assert len(made) == 1 and a.early is not None
    sc = a.scope()
    bus = sc.get_or_create("bus", object)
    assert sc.get_or_create("bus", object) is bus
