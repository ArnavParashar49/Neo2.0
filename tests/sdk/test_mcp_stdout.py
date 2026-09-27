"""`python -m neo2.mcp` in a real process: stdout carries JSON-RPC and nothing else. A stray print,
a raw write to fd 1 or a child process's output lands on stderr. Nothing touches ~/.neo, and the
agent's temp folder is gone after the client hangs up. (The only processes started are the dev
interpreter itself.)"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from neo import effects

pytest.importorskip("mcp")

pytestmark = pytest.mark.portable  # Linux or macOS, any install (what needs an extra skips): CI's sdk-core job

ROOT = Path(__file__).resolve().parents[2]
NOISY = """
import os, subprocess, sys
import neo2.mcp.server as srv

real = srv.NeoMCP._list_tools

async def noisy(self, ctx, params):
    print("stray python print")
    os.write(1, b"stray fd1 write\\n")
    subprocess.run([sys.executable, "-c", "print('stray child output')"], check=True)
    return await real(self, ctx, params)

srv.NeoMCP._list_tools = noisy
from neo2.mcp import main
sys.exit(main(["--tools", "none", "--approve", "deny"]))
"""


class Server:
    """The server process, spoken to line by line; every stdout line is kept."""

    def __init__(self, argv: list[str], tmp_path: Path) -> None:
        self.home = tmp_path / "home"
        self.tmp = tmp_path / "tmp"
        self.home.mkdir(exist_ok=True)
        self.tmp.mkdir(exist_ok=True)
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("NEO_", "GEMINI", "GROQ", "ANTHROPIC", "OPENAI", "PYTEST"))
        }
        env.update(HOME=str(self.home), TMPDIR=str(self.tmp), PYTHONPATH=str(ROOT))
        self.stderr_path = tmp_path / "stderr.txt"
        self._err = open(self.stderr_path, "wb")  # noqa: SIM115 — closed in close()
        with effects.allow("process"):  # a Python child: refused unless meant, as here
            self.proc = subprocess.Popen(
                argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err, cwd=str(tmp_path), env=env
            )
        self.lines: list[str] = []
        self._q: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for raw in self.proc.stdout:
            self._q.put(raw.decode("utf-8", "replace"))
        self._q.put(None)

    def send(self, msg: dict) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write((json.dumps(msg) + "\n").encode())
        self.proc.stdin.flush()

    def reply(self, id_: int, timeout: float = 60) -> dict:
        while True:
            try:
                line = self._q.get(timeout=timeout)
            except queue.Empty:
                self.proc.kill()
                pytest.fail(f"no reply to request {id_}; stderr: {self.stderr()[-2000:]}")
            if line is None:
                pytest.fail(f"the server closed stdout; stderr: {self.stderr()[-2000:]}")
            self.lines.append(line)
            msg = json.loads(line)
            if msg.get("id") == id_:
                return msg

    def close(self) -> int:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        try:
            code = self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            pytest.fail("the server didn't exit after stdin closed")
        while (line := self._q.get(timeout=10)) is not None:
            self.lines.append(line)
        self._err.close()
        return code

    def stderr(self) -> str:
        if not self._err.closed:
            self._err.flush()
        return self.stderr_path.read_text("utf-8", "replace")


def _talk(srv: Server) -> list[str]:
    init = {
        "protocolVersion": "2025-11-25",
        "capabilities": {},
        "clientInfo": {"name": "stdout-test", "version": "1"},
    }
    srv.send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": init})
    r = srv.reply(1)
    assert r["result"]["serverInfo"]["name"] == "neo"
    srv.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    srv.send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    names = [t["name"] for t in srv.reply(2)["result"]["tools"]]
    call = {"name": "neo_task", "arguments": {"goal": "what time is it?"}}
    srv.send({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": call})
    r = srv.reply(3)["result"]
    assert r["isError"] is True and "--env" in r["content"][0]["text"]  # no brain without --env
    return names


def _only_json_rpc(srv: Server) -> None:
    assert srv.lines, "nothing on stdout"
    for line in srv.lines:
        msg = json.loads(line)  # every line parses…
        assert msg.get("jsonrpc") == "2.0", line  # …as JSON-RPC


def test_stdout_is_only_json_rpc(tmp_path):
    srv = Server([sys.executable, "-m", "neo2.mcp", "--tools", "none", "--approve", "deny"], tmp_path)
    try:
        names = _talk(srv)
    finally:
        code = srv.close()
    assert code == 0, srv.stderr()[-2000:]
    assert names == ["neo_task"]
    _only_json_rpc(srv)
    assert "no model to think with" in srv.stderr()  # said once, on stderr
    assert not (srv.home / ".neo").exists()
    assert list(srv.tmp.glob("neo-agent-*")) == []  # the agent's temp folder went with it


def test_stray_output_goes_to_stderr(tmp_path):
    srv = Server([sys.executable, "-c", NOISY], tmp_path)
    try:
        assert _talk(srv) == ["neo_task"]
    finally:
        code = srv.close()
    assert code == 0, srv.stderr()[-2000:]
    _only_json_rpc(srv)
    err = srv.stderr()
    for stray in ("stray python print", "stray fd1 write", "stray child output"):
        assert stray in err
        assert not any(stray in line for line in srv.lines)


def test_sigterm_stops_it_cleanly(tmp_path):
    """A client that terminates the server (SIGTERM) still gets its cleanup: the agent's temp
    folder — which may hold a browser profile — doesn't outlive it."""
    import signal

    srv = Server([sys.executable, "-m", "neo2.mcp", "--tools", "none"], tmp_path)
    try:
        assert _talk(srv) == ["neo_task"]
        assert len(list(srv.tmp.glob("neo-agent-*"))) == 1  # the agent's folder, while it runs
        srv.proc.send_signal(signal.SIGTERM)
        code = srv.proc.wait(timeout=60)
    finally:
        if srv.proc.poll() is None:
            srv.proc.kill()
        code = srv.close()
    assert code == 0, srv.stderr()[-2000:]
    assert list(srv.tmp.glob("neo-agent-*")) == []
    _only_json_rpc(srv)


def test_the_entry_point_imports_only_the_standard_library(tmp_path):
    """`python -m neo2.mcp` runs neo2/__init__ and neo2/mcp/__init__ before main() claims fd 1:
    neither may import anything that could write to stdout first."""
    code = (
        "import json, sys\n"
        "before = set(sys.modules)\n"
        "import neo2.mcp\n"
        "new = {m.split('.')[0] for m in set(sys.modules) - before}\n"
        "print(json.dumps(sorted(m for m in new if m not in sys.stdlib_module_names)))\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NEO_", "PYTEST"))}
    env.update(HOME=str(tmp_path), PYTHONPATH=str(ROOT))
    with effects.allow("process"):  # a Python child: refused unless meant, as here
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(tmp_path), env=env, timeout=120
        )
    assert out.returncode == 0, out.stderr[-2000:]
    assert json.loads(out.stdout.strip().splitlines()[-1]) == ["neo2"]
