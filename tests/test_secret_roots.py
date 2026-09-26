"""NEO's own data is secret wherever it lives; an embedded agent's shell sees only a basic environment."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from neo_lib import make_runtime

from neo import runtime
from neo.tools.computer import policy, shell


@pytest.fixture
def agent(tmp_path):
    vault = tmp_path / "host" / "vault"
    vault.mkdir(parents=True)
    rt = make_runtime(tmp_path / "host" / "neo-data", "a", extra_secret_paths=[vault])
    (rt.settings.data_dir / "audit.log").write_text("x\n")
    (vault / "keys.txt").write_text("sk-1\n")
    (tmp_path / "host" / "notes.txt").write_text("hello\n")
    return rt, vault


def test_an_agents_data_dir_and_extra_paths_are_secret(agent, tmp_path):
    rt, vault = agent
    dd = rt.settings.data_dir
    for p in (dd, dd / "audit.log", dd / "memory.sqlite3", dd / "browser" / "Cookies", vault / "keys.txt"):
        assert policy.is_secret(p), p
    assert not policy.is_secret(tmp_path / "host" / "notes.txt")
    assert policy.covers_secret(tmp_path / "host") and policy.covers_secret(tmp_path)
    assert not policy.covers_secret(tmp_path / "elsewhere")


def test_the_shell_and_file_tools_refuse_them(agent, tmp_path):
    rt, vault = agent
    dd = rt.settings.data_dir
    assert not policy.shell_is_read_only(f"cat {dd}/audit.log")
    assert not policy.shell_is_read_only(f"grep -r sk- {tmp_path / 'host'}")  # a recursive read reaching it
    assert not policy.shell_is_read_only(f"head {vault}/keys.txt")
    assert policy.shell_is_read_only(f"cat {tmp_path / 'host' / 'notes.txt'}")
    for p in (dd / "audit.log", vault / "keys.txt"):
        out = shell.read_file(str(p))
        assert out.startswith("Error:") and "doesn't read it" in out
    assert policy.write_risk(dd / "new.txt") == "it holds secrets"
    assert shell.write_file(str(dd / "planted.txt"), "x", {}).startswith("Error:")
    assert not (dd / "planted.txt").exists()


def test_through_a_symlink_too(agent, tmp_path):
    rt, _ = agent
    link = tmp_path / "innocent"
    link.symlink_to(rt.settings.data_dir)
    assert policy.is_secret(link / "audit.log")


def test_a_runtimes_own_secret_paths(tmp_path):
    rt = make_runtime(tmp_path / "d", "d")
    rt.secret_paths = (tmp_path / "tenant-keys",)
    assert policy.is_secret(tmp_path / "tenant-keys" / "a")
    rt.close()  # gone: its folders are ordinary folders again
    assert not policy.is_secret(tmp_path / "tenant-keys" / "a") and not policy.is_secret(tmp_path / "d")


def test_every_live_agent_counts_even_unbound(tmp_path):
    a = make_runtime(tmp_path / "a", "a")
    b = make_runtime(tmp_path / "b", "b")
    with runtime.bind(a.scope()):
        assert policy.is_secret(b.settings.data_dir / "memory.sqlite3")  # B's data, seen from A


def test_neo_apps_custom_data_dir_is_secret_once_its_settings_exist(tmp_path, monkeypatch):
    from neo import config

    custom = tmp_path / "custom-neo"
    monkeypatch.setenv("NEO_DATA_DIR", str(custom))
    monkeypatch.setattr(config, "_settings", None)
    assert not policy.is_secret(custom / "audit.log")  # never builds settings to find out
    config.settings()
    assert policy.is_secret(custom / "audit.log") and policy.is_secret(custom)
    assert not policy.shell_is_read_only(f"cat {custom}/audit.log")
    assert policy.is_secret(policy.HOME / ".neo" / "memory.sqlite3")  # the default, as always


_HOST_ENV = {
    "DATABASE_URL": "postgres://u:p@db/prod",
    "SENTRY_DSN": "https://k@sentry.io/1",
    "STRIPE_SK": "sk_live_x",
    "GEMINI_API_KEY": "g",
    "LANG": "en_US.UTF-8",
    "LC_ALL": "en_US.UTF-8",
    "TZ": "Asia/Dubai",
    "EDITOR": "vim",
}


def test_an_agents_shell_gets_only_the_basics(tmp_path, monkeypatch):
    for k, v in _HOST_ENV.items():
        monkeypatch.setenv(k, v)
    rt = make_runtime(tmp_path / "a", "a")
    with runtime.bind(rt.scope()):
        env = shell._env()
    for k in ("DATABASE_URL", "SENTRY_DSN", "STRIPE_SK", "GEMINI_API_KEY", "EDITOR"):
        assert k not in env, k
    assert env["LANG"] == env["LC_ALL"] == "en_US.UTF-8" and env["TZ"] == "Asia/Dubai"
    assert env["HOME"] == os.environ["HOME"] and "/opt/homebrew/bin" in env["PATH"]
    for k, v in policy.GIT_SAFE_ENV.items():
        assert env[k] == v


def test_neo_apps_shell_environment_is_unchanged(monkeypatch):
    for k, v in _HOST_ENV.items():
        monkeypatch.setenv(k, v)
    env = shell._env()
    assert "GEMINI_API_KEY" not in env  # the credential-looking names, as before
    for k in ("DATABASE_URL", "SENTRY_DSN", "STRIPE_SK", "EDITOR", "LANG"):
        assert env[k] == _HOST_ENV[k]


def test_a_data_dir_named_through_the_data_volume(tmp_path):
    """The host named the folder through /System/Volumes/Data; a tool names its files the usual way."""
    real = tmp_path.resolve() / "agent"
    alias = Path("/System/Volumes/Data" + str(real))
    if not alias.parent.exists():
        pytest.skip("no APFS data volume here")
    rt = make_runtime(alias, "a")  # live while referenced
    for p in (real / "memory.sqlite3", alias / "memory.sqlite3", real / "browser" / "Cookies"):
        assert policy.is_secret(p), p
    assert policy.covers_secret(real.parent) and not policy.shell_is_read_only(f"cat {real}/audit.log")
    rt.close()


async def test_applescript_never_sees_the_hosts_environment(monkeypatch, tmp_path):
    """`system attribute "X"` reads osascript's environment: it asks first, and an agent's
    osascript gets the shell tool's basic environment, never the host process's."""
    from neo.agent.registry import ToolContext, registry
    from neo.tools.computer import apps

    monkeypatch.setenv("HOST_DATABASE_URL", "postgres://svc:hunter2@db/prod")
    envs: list = []

    class _Proc:
        returncode = 0

        async def communicate(self):
            return b"ok", b""

    async def fake_exec(*cmd, **kw):
        envs.append(kw.get("env"))
        return _Proc()

    monkeypatch.setattr(apps.asyncio, "create_subprocess_exec", fake_exec)
    for script in ('system attribute "HOST_DATABASE_URL"', "get system attribute", "system info"):
        assert not policy.script_is_read_only(script), script
    assert policy.script_is_read_only('name of application "Finder"')
    rt = make_runtime(tmp_path / "a", "a", ["applescript"])
    with runtime.bind(rt.scope()):
        out = await registry().invoke("applescript", {"script": 'system attribute "HOST_DATABASE_URL"'}, ToolContext())
        assert out.halt and envs == []  # asks first
        await registry().invoke("applescript", {"script": 'name of application "Finder"'}, ToolContext())
        basic = shell._env()
    assert envs and envs[-1] == basic and "HOST_DATABASE_URL" not in envs[-1]
    runtime.set_library_mode(False)
    await apps.osascript('name of application "Finder"')
    assert envs[-1] is None  # NEO.app's osascript: its own environment, as always
