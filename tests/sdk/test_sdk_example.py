"""examples/embed_quickstart.py stays runnable: a host's first contact with the SDK."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from neo import effects

pytestmark = pytest.mark.portable  # Linux or macOS, any install (what needs an extra skips): CI's sdk-core job

ROOT = Path(__file__).resolve().parents[2]


def test_the_quickstart_runs(tmp_path, home):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NEO_", "GEMINI", "GROQ", "ANTHROPIC", "OPENAI", "PYTEST"))}
    env.update(HOME=str(home), NEO_DATA_DIR=str(tmp_path / "app_data"))
    with effects.allow("process"):  # a Python child: refused unless meant, as here
        out = subprocess.run(
            [sys.executable, str(ROOT / "examples" / "embed_quickstart.py")],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            env=env,
            timeout=120,
        )
    assert out.returncode == 0, out.stderr[-3000:]
    lines = out.stdout.splitlines()
    assert lines[0].startswith("1. needs_approval: refund 49.5 on order 812")
    assert lines[1] == "2. done: Refunded 49.50 on order 812.  (steps: ['refund'])"
    assert lines[2] == "   audit: [('staged', None), ('confirmed', 'user:42')]"
    assert lines[3].startswith("3. started") and lines[3].endswith("completed")
    assert lines[4] == "4. done: Sent."
    assert out.stderr == ""
    assert not (home / ".neo").exists() and not (tmp_path / "app_data").exists()
