"""NEO's own package lives in neo2/core since 2.1: what finds its way around the checkout from there
still does, and `python -m neo2` is NEO.app's core without making `import neo2` any heavier."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from neo2.core import config, effects

ROOT = Path(__file__).resolve().parents[1]


def test_the_checkout_is_found_from_neo2_core():
    # the checkout's .env on top of ~/.neo/.env when running from source; the suite's own switch
    assert config.PROJECT_ROOT == ROOT and (ROOT / "pyproject.toml").is_file()
    assert config._env_files(config.PROJECT_ROOT) == (config.USER_ENV, ROOT / ".env")
    assert effects._source_tree() is True


def test_python_m_neo2_imports_nothing_until_it_runs(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NEO_", "PYTEST"))}
    env.update(HOME=str(tmp_path), PYTHONPATH=str(ROOT), NEO_NO_EFFECTS="1")
    code = (
        "import json, sys\n"
        "import neo2.__main__\n"  # imported, not run: what `python -m neo2` would start stays unloaded
        "print(json.dumps(sorted(m for m in sys.modules if m.startswith('neo2'))))\n"
    )
    with effects.allow("process"):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(tmp_path), env=env, timeout=60)
        run = subprocess.run(
            [sys.executable, "-m", "neo2", "--help"], capture_output=True, text=True, cwd=str(tmp_path), env=env, timeout=120
        )
    assert out.returncode == 0, out.stderr[-2000:]
    assert json.loads(out.stdout.strip().splitlines()[-1]) == ["neo2", "neo2.__main__"]
    assert run.returncode == 0 and run.stdout.startswith("usage: neo2 "), run.stderr[-2000:]
    for flag in ("--text", "--doctor", "--local", "--forget-all"):  # every entry the old name had
        assert flag in run.stdout
    assert not (tmp_path / ".neo").exists()
