"""What importing and first use cost a host: `import neo2` is the standard library only, and
`from neo2 import AsyncAgent` plus a first web-only request load no model, no vendor SDK, no macOS
framework and no browser. Each check runs in a fresh interpreter."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from neo import effects

pytestmark = pytest.mark.portable  # no macOS or extra needed: CI's sdk-core job runs it on Linux too

ROOT = Path(__file__).resolve().parents[2]
HEAVY = (
    "torch",
    "sentence_transformers",
    "transformers",
    "google.genai",
    "openai",
    "anthropic",
    "playwright",
    "AppKit",
    "Quartz",
    "ApplicationServices",
    "objc",
    "laya",
    "mlx",
    "neo.decide",
    "neo.tools.computer",
    "neo.voice",
)


def _run(code: str, tmp_path: Path) -> dict:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("NEO_", "GEMINI", "GROQ", "ANTHROPIC", "OPENAI", "PYTEST"))
    }
    env.update(HOME=str(home), PYTHONPATH=str(ROOT), NEO_DATA_DIR=str(tmp_path / "app_data"))
    with effects.allow("process"):  # a Python child: refused unless meant, as here
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(tmp_path), env=env, timeout=120
        )
    assert out.returncode == 0, out.stderr[-3000:]
    assert not (home / ".neo").exists() and not (tmp_path / "app_data").exists()
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_import_neo2_loads_only_the_standard_library(tmp_path):
    got = _run(
        "import sys, json\n"
        "before = set(sys.modules)\n"
        "import neo2\n"
        "new = sorted({m.split('.')[0] for m in set(sys.modules) - before})\n"
        "print(json.dumps({'new': new, 'std': sorted(sys.stdlib_module_names)}))\n",
        tmp_path,
    )
    outside = [m for m in got["new"] if m not in set(got["std"]) and m != "neo2"]
    assert outside == [], outside


def test_from_neo2_import_asyncagent_is_light(tmp_path):
    got = _run(
        "import sys, json, time\n"
        "t = time.perf_counter()\n"
        "from neo2 import AsyncAgent, AgentConfig, Approval, Result, host_tool\n"
        "dt = time.perf_counter() - t\n"
        f"heavy = [m for m in {HEAVY!r} if m in sys.modules]\n"
        "print(json.dumps({'s': dt, 'heavy': heavy, 'n': len(sys.modules)}))\n",
        tmp_path,
    )
    assert got["heavy"] == []
    # The target is 150 ms on a warm Mac (measured ~0.1 s); the bound here only catches a regression
    # that pulls in something big, without flaking on a slow CI machine.
    assert got["s"] < 2.0, got


def test_a_first_web_request_never_loads_a_model_or_the_mac_frameworks(tmp_path):
    got = _run(
        "import sys, json, asyncio\n"
        "import neo2\n"
        "from neo2.testing import ScriptedBrain, ToolCall, Turn\n"
        "async def main():\n"
        "    brain = ScriptedBrain([Turn(tool_calls=[ToolCall('1', 'clock', {})]), 'noon'])\n"
        "    async with neo2.AsyncAgent(neo2.AgentConfig(tools='web'), provider=brain) as a:\n"
        "        r1 = await a.ask('what time is it')\n"
        "        r2 = await a.ask('and the time again, via a model', conversation='c2')\n"
        "    return [r1.status, r2.status]\n"
        "st = asyncio.run(main())\n"
        f"heavy = [m for m in {HEAVY!r} if m in sys.modules]\n"
        "print(json.dumps({'status': st, 'heavy': heavy}))\n",
        tmp_path,
    )
    assert got["status"] == ["done", "done"]
    assert got["heavy"] == []


def test_the_adapters_and_the_mcp_server_load_on_first_use_without_their_sdks(tmp_path):
    got = _run(
        "import sys, json\n"
        "import neo2\n"
        "a, m = neo2.adapters, neo2.mcp\n"
        "import neo2.adapters.openai, neo2.adapters.anthropic, neo2.adapters.gemini, neo2.adapters.langchain\n"
        f"heavy = [x for x in {(*HEAVY, 'mcp', 'mcp_types', 'langchain_core')!r} if x in sys.modules]\n"
        "print(json.dumps({'names': [a.__name__, m.__name__], 'heavy': heavy, 'main': callable(m.main)}))\n",
        tmp_path,
    )
    assert got == {"names": ["neo2.adapters", "neo2.mcp"], "heavy": [], "main": True}
