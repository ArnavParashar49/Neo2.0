"""Where an embedded agent keeps things: with no data_dir, only a private temp folder that goes
with aclose() (audit records to on_audit and the 'neo.audit' logger); with one, only its own files
there. Never ~/.neo, and nothing from the host's environment (keys, NEO_CONFIRM_DESTRUCTIVE)."""

from __future__ import annotations

import json
import logging
import os
import stat
from pathlib import Path

import pytest

import neo2
from neo2.testing import ScriptedBrain, ToolCall, Turn


def _wire(calls: list):
    @neo2.host_tool(risk="destructive")
    def wire_money(to: str) -> str:
        """Send money."""
        calls.append(to)
        return "wired"

    return wire_money


PAY = Turn(tool_calls=[ToolCall("1", "wire_money", {"to": "mallory"})])


async def test_no_data_dir_is_a_private_temp_dir_that_goes_with_close(home, monkeypatch, caplog):
    monkeypatch.setenv("NEO_CONFIRM_DESTRUCTIVE", "false")
    monkeypatch.setenv("GEMINI_API_KEY", "a-key-from-the-host-env")
    monkeypatch.setenv("NEO_DECIDE_MODES", "web_search=on,open_app=on")
    monkeypatch.setenv("NEO_MEMORY_ENABLED", "true")
    calls: list = []
    audit: list = []
    caplog.set_level(logging.INFO, logger="neo.audit")
    brain = ScriptedBrain([PAY, "(no more)"])
    async with neo2.AsyncAgent(neo2.AgentConfig(tools="web"), provider=brain, host_tools=[_wire(calls)], on_audit=audit.append) as agent:
        dd = agent.data_dir
        assert dd.is_dir() and stat.S_IMODE(dd.stat().st_mode) == 0o700
        assert not dd.is_relative_to(home)  # the system temp dir, not the user's folders
        s = agent._rt.settings
        assert s.gemini_api_key == "" and s.confirm_destructive and not s.memory_enabled
        assert set(s.decide_modes.values()) == {"legacy"} and not s.decide_kind_routing
        r = await agent.ask("pay mallory")
        assert r.status == "needs_approval" and calls == []  # confirmation stayed on
        await agent.deny(r.approval.action_id)
        assert sorted(p.name for p in dd.iterdir()) == [".neo-agent"]  # no memory, no audit file
    assert not dd.exists()
    assert [a["status"] for a in audit] == ["staged", "cancelled: user cancelled"]
    logged = [json.loads(r.getMessage()) for r in caplog.records if r.name == "neo.audit"]
    assert [x["status"] for x in logged] == ["staged", "cancelled: user cancelled"]
    assert not (home / ".neo").exists()
    assert not Path(os.environ["NEO_DATA_DIR"]).exists()  # NEO.app's settings were never built


async def test_a_data_dir_holds_only_neo_files_private_to_the_user(tmp_path):
    dd = tmp_path / "agents" / "u42"
    calls: list = []
    brain = ScriptedBrain(
        [PAY, "Sent.", Turn(tool_calls=[ToolCall("2", "memory", {"action": "remember", "text": "likes tea"})]), "Noted."]
    )
    async with neo2.AsyncAgent(neo2.AgentConfig(tools="web", data_dir=dd, memory=True), provider=brain, host_tools=[_wire(calls)]) as agent:
        r = await agent.ask("pay mallory")
        await agent.approve(r.approval.action_id, via=neo2.Approval("t", "u"))
        await agent.ask("remember I like tea")
    names = sorted(p.name for p in dd.iterdir())
    assert names == [".neo-agent", "audit.log", "memory.sqlite3"]
    for n in names:
        assert stat.S_IMODE((dd / n).stat().st_mode) == 0o600
    assert stat.S_IMODE(dd.stat().st_mode) == 0o700
    assert calls == ["mallory"]
    assert dd.exists()  # the host's folder stays after close


async def test_memory_off_means_no_memory_file_and_no_memory_tool(tmp_path):
    dd = tmp_path / "agent"
    brain = ScriptedBrain([Turn(tool_calls=[ToolCall("1", "memory", {"action": "remember", "text": "x"})]), "ok"])
    async with neo2.AsyncAgent(neo2.AgentConfig(tools="web", data_dir=dd), provider=brain) as agent:
        assert "memory" not in {t.name for t in agent.tools}
        r = await agent.ask("remember x")
    assert r.steps[0].summary == "Unknown tool: memory"
    assert not (dd / "memory.sqlite3").exists()


async def test_memory_in_a_file_of_the_hosts_choosing(tmp_path):
    mem = tmp_path / "db" / "u42.sqlite3"
    mem.parent.mkdir()
    brain = ScriptedBrain(
        [Turn(tool_calls=[ToolCall("1", "memory", {"action": "remember", "text": "likes tea", "kind": "profile"})]), "Noted.", "You like tea."]
    )
    cfg = neo2.AgentConfig(tools="none", memory=neo2.MemoryConfig(path=mem, cloud="profile"))
    async with neo2.AsyncAgent(cfg, provider=brain) as agent:
        await agent.ask("please store that I enjoy tea")
        assert mem.exists() and not (agent.data_dir / "memory.sqlite3").exists()
        from neo.security import policy

        assert policy.is_secret(mem) and policy.is_secret(str(mem) + "-wal")  # NEO's tools never read it
        await agent.ask("what do I like?")
        assert "likes tea" in brain.systems[-1]  # the profile rides along to the model
    assert mem.exists()  # the host's file stays


async def test_memory_is_never_a_database_neo_didnt_make(tmp_path):
    """A host's own SQLite file (NEO would add its tables to it, and an erase would delete it), a
    file that isn't SQLite, NEO.app's own memory: refused, untouched. A new or empty file is fine."""
    import sqlite3

    app_db = tmp_path / "app.db"
    db = sqlite3.connect(app_db)
    db.execute("CREATE TABLE customers (id INTEGER, name TEXT)")
    db.execute("INSERT INTO customers VALUES (1, 'Ada')")
    db.commit()
    db.close()
    notes = tmp_path / "notes.bin"
    notes.write_bytes(b"not a database")
    (tmp_path / "notes.bin-journal").write_bytes(b"x")
    neo_apps = tmp_path / "neo-app-memory.sqlite3"  # NEO.app's own memory: a NEO database, not an agent's
    from neo.memory.store import Store

    Store(neo_apps).close()
    (tmp_path / "a-folder").mkdir()
    before = sorted(p.name for p in tmp_path.iterdir())
    for path in (app_db, notes, neo_apps, tmp_path / "a-folder"):
        with pytest.raises(ValueError, match="didn't make"):
            neo2.AsyncAgent(neo2.AgentConfig(tools="none", memory=neo2.MemoryConfig(path=path)), provider=ScriptedBrain())
    assert sorted(p.name for p in tmp_path.iterdir()) == before  # not even a -wal left next to them
    assert sqlite3.connect(app_db).execute("SELECT name FROM customers").fetchall() == [("Ada",)]
    empty = tmp_path / "empty.sqlite3"
    empty.write_bytes(b"")
    async with neo2.AsyncAgent(neo2.AgentConfig(tools="none", memory=neo2.MemoryConfig(path=empty)), provider=ScriptedBrain()):
        pass


async def test_the_agents_folders_are_never_neo_apps(home):
    for bad in (home / ".neo", home / ".neo" / "agents" / "x", home / ".NEO"):
        with pytest.raises(ValueError, match="NEO.app's own data"):
            neo2.AgentConfig(data_dir=bad)
    with pytest.raises(ValueError, match="NEO.app's own data"):
        neo2.AgentConfig(memory=neo2.MemoryConfig(path=home / ".neo" / "memory.sqlite3"))
    with pytest.raises(ValueError, match="NEO.app's own data"):
        neo2.AgentConfig(models_dir=home / ".neo" / "models")


async def test_neo_apps_data_where_neo_data_dir_moved_it_is_never_an_agents(tmp_path, monkeypatch):
    """NEO.app's data moved with NEO_DATA_DIR (or its memory with NEO_MEMORY_PATH): never an
    agent's folder or memory, and secret to every agent's tools (read from the environment only —
    an embedded agent never opens ~/.neo/.env to find out)."""
    from neo.security import policy

    app = tmp_path / "moved-neo"
    app.mkdir()
    (app / "memory.sqlite3").write_text("NEO.app's memory")
    app_mem = tmp_path / "elsewhere" / "neo-memory.sqlite3"
    monkeypatch.setenv("NEO_DATA_DIR", str(app))
    monkeypatch.setenv("NEO_MEMORY_PATH", str(app_mem))
    for bad in (
        {"data_dir": app},
        {"data_dir": app / "sub"},  # inside it
        {"data_dir": tmp_path},  # holding it
        {"memory": neo2.MemoryConfig(path=app / "memory.sqlite3")},
        {"memory": neo2.MemoryConfig(path=app_mem)},
        {"data_dir": app_mem.parent},  # holding NEO.app's memory
    ):
        with pytest.raises(ValueError, match="NEO.app's own data"):
            neo2.AgentConfig(**bad)
    from neo.config import LibrarySettings

    with pytest.raises(ValueError, match="NEO.app's own data"):
        LibrarySettings(data_dir=app / "sub")  # the settings an agent runs with check it too
    assert not policy.is_secret(app / "memory.sqlite3")  # NEO.app's own tools: as before
    async with neo2.AsyncAgent(neo2.AgentConfig(tools="none", data_dir=tmp_path / "agent"), provider=ScriptedBrain()):
        for p in (app / "memory.sqlite3", app / "audit.log", app_mem, Path(str(app_mem) + "-wal")):
            assert policy.is_secret(p), p
        assert policy.covers_secret(tmp_path) and not policy.shell_is_read_only(f"cat {app}/audit.log")
    assert (app / "memory.sqlite3").read_text() == "NEO.app's memory"


async def test_a_folder_of_the_hosts_own_files_is_refused(tmp_path):
    d = tmp_path / "project"
    d.mkdir()
    (d / "notes.txt").write_text("mine")
    with pytest.raises(ValueError, match="already holds other files"):
        neo2.AsyncAgent(neo2.AgentConfig(data_dir=d), provider=ScriptedBrain())
    assert sorted(p.name for p in d.iterdir()) == ["notes.txt"]


async def test_two_live_agents_never_share_a_folder(tmp_path):
    dd = tmp_path / "shared"
    async with neo2.AsyncAgent(neo2.AgentConfig(data_dir=dd), provider=ScriptedBrain()):
        with pytest.raises(ValueError, match="another agent's"):
            neo2.AsyncAgent(neo2.AgentConfig(data_dir=dd / "inner"), provider=ScriptedBrain())
    async with neo2.AsyncAgent(neo2.AgentConfig(data_dir=dd), provider=ScriptedBrain()):
        pass  # closed agents give their folder back


async def test_a_closed_agents_data_stays_secret_to_the_others(tmp_path):
    """One agent per tenant, idle ones closed: a live one's tools never read what a closed one
    kept — its folder (memory, audit log, browser profile), its memory file, its secret paths."""
    from neo import runtime
    from neo.security import policy

    a_dir, a_mem, vault = tmp_path / "tenants" / "a", tmp_path / "dbs" / "a.sqlite3", tmp_path / "vault"
    cfg = neo2.AgentConfig(tools="none", data_dir=a_dir, memory=neo2.MemoryConfig(path=a_mem), extra_secret_paths=(vault,))
    async with neo2.AsyncAgent(neo2.AgentConfig(tools="none", data_dir=tmp_path / "b"), provider=ScriptedBrain()):
        a = neo2.AsyncAgent(cfg, provider=ScriptedBrain())
        await a.ask("remember that my locker is 42")
        (a_dir / "browser").mkdir()
        (a_dir / "browser" / "Cookies").write_text("session=1")
        await a.aclose()
        for p in (a_dir / "audit.log", a_dir / "browser" / "Cookies", a_mem, Path(str(a_mem) + "-wal"), vault / "k"):
            assert policy.is_secret(p), p
        assert policy.covers_secret(tmp_path / "tenants") and policy.covers_secret(tmp_path / "dbs")
        assert not policy.shell_is_read_only(f"cat {a_mem}") and not policy.shell_is_read_only(f"grep -r x {tmp_path / 'tenants'}")
        t = neo2.AsyncAgent(neo2.AgentConfig(tools="none"), provider=ScriptedBrain())
        td = t.data_dir
        await t.aclose()  # a temp folder goes with its agent: nothing of it is kept
        assert not td.exists() and td not in runtime.secret_roots()
    async with neo2.AsyncAgent(cfg, provider=ScriptedBrain()):
        pass  # and the tenant's agent opens again on its own folder and memory


async def test_a_failed_construction_leaves_no_temp_dir(tmp_path, monkeypatch):
    import tempfile

    made: list[str] = []
    real = tempfile.mkdtemp

    def mkdtemp(*a, **kw):
        made.append(real(*a, **kw))
        return made[-1]

    monkeypatch.setattr(tempfile, "mkdtemp", mkdtemp)
    with pytest.raises(ValueError):
        neo2.AsyncAgent(neo2.AgentConfig(extra_secret_paths=("relative/but/fine",), tools="nonsense"), provider=ScriptedBrain())
    assert made == []  # refused before any folder was made
    from neo import config

    monkeypatch.setattr(config.LibrarySettings, "ensure_dirs", lambda self: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        neo2.AsyncAgent(neo2.AgentConfig(), provider=ScriptedBrain())
    assert made and not Path(made[-1]).exists()


async def test_two_live_agents_never_share_a_memory(tmp_path):
    mem = tmp_path / "shared" / "mem.sqlite3"
    first = neo2.AgentConfig(tools="none", memory=neo2.MemoryConfig(path=mem))
    async with neo2.AsyncAgent(first, provider=ScriptedBrain()):
        assert mem.parent.is_dir()  # made for it, private
        assert stat.S_IMODE(mem.parent.stat().st_mode) == 0o700
        with pytest.raises(ValueError, match="another agent's"):
            neo2.AsyncAgent(neo2.AgentConfig(tools="none", memory=neo2.MemoryConfig(path=mem)), provider=ScriptedBrain())
        with pytest.raises(ValueError, match="another agent's"):  # a folder holding the other's memory
            neo2.AsyncAgent(neo2.AgentConfig(tools="none", data_dir=mem.parent), provider=ScriptedBrain())
        assert list(mem.parent.iterdir()) == []  # a refused agent made nothing there
        async with neo2.AsyncAgent(neo2.AgentConfig(tools="none", data_dir=tmp_path / "own"), provider=ScriptedBrain()) as b:
            with pytest.raises(ValueError, match="another agent's"):  # a memory file in another's folder
                neo2.AsyncAgent(
                    neo2.AgentConfig(tools="none", memory=neo2.MemoryConfig(path=b.data_dir / "x.sqlite3")),
                    provider=ScriptedBrain(),
                )


async def test_a_rejected_host_tool_leaves_nothing_behind():
    from neo import runtime

    before = set(map(id, runtime.live_runtimes()))

    def shell() -> str:
        return ""

    with pytest.raises(ValueError, match="NEO's own tools"):
        neo2.AsyncAgent(neo2.AgentConfig(tools="none"), provider=ScriptedBrain(), host_tools=[neo2.host_tool(lambda: "", name="fine"), shell])
    assert set(map(id, runtime.live_runtimes())) == before
