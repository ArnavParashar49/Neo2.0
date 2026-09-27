"""The repo-only `neo` compatibility shim: NEO's package moved to neo2.core, and a checkout still
answers to the old name — the same modules, never a second copy, with one DeprecationWarning.
neo2-codemod: skip-file (this file names the old package on purpose)

Each check runs in a fresh interpreter: the shim changes the import system of its process."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from neo2.core import effects

ROOT = Path(__file__).resolve().parents[1]


def _python(args: list[str], tmp_path: Path) -> subprocess.CompletedProcess:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NEO_", "GEMINI", "GROQ", "ANTHROPIC", "PYTEST"))}
    env.update(HOME=str(home), PYTHONPATH=str(ROOT), NEO_NO_EFFECTS="1", NEO_DATA_DIR=str(tmp_path / "data"))
    with effects.allow("process"):  # a Python child: refused unless meant, as here
        out = subprocess.run(
            [sys.executable, *args], capture_output=True, text=True, cwd=str(tmp_path), env=env, timeout=120
        )
    assert not (home / ".neo").exists()
    return out


def _run(code: str, tmp_path: Path) -> dict:
    out = _python(["-c", code], tmp_path)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_old_imports_give_the_very_same_modules(tmp_path):
    got = _run(
        "import json, sys\n"
        "import neo.tools.web\n"
        "import neo2.core, neo2.core.tools.web\n"
        "import neo\n"
        "from neo import runtime, log_to_stdout\n"
        "from neo.agent.registry import registry\n"
        "import neo.agent.session as session\n"
        "import neo2.core.agent.registry as registry_mod, neo2.core.agent.session\n"
        "mods = {id(m) for m in list(sys.modules.values()) if (getattr(m, '__file__', '') or '').endswith('tools/web.py')}\n"
        "print(json.dumps({\n"
        "  'neo is core': neo is neo2.core,\n"
        "  'web': neo.tools.web is neo2.core.tools.web and sys.modules['neo.tools.web'] is neo2.core.tools.web,\n"
        "  'from': runtime is neo2.core.runtime and log_to_stdout is neo2.core.log_to_stdout\n"
        "          and registry is registry_mod.registry and session is neo2.core.agent.session,\n"
        "  'copies of web.py': len(mods),\n"
        "  'spec names': [neo2.core.tools.web.__spec__.name, neo2.core.tools.__spec__.name, session.__name__],\n"
        "}))\n",
        tmp_path,
    )
    assert got == {
        "neo is core": True,
        "web": True,
        "from": True,
        "copies of web.py": 1,
        "spec names": ["neo2.core.tools.web", "neo2.core.tools", "neo2.core.agent.session"],
    }


def test_a_patch_through_the_old_name_is_the_new_modules(tmp_path):
    got = _run(
        "import json, sys, types\n"
        "import neo.reflex\n"
        "import neo2.core.reflex\n"
        "neo.reflex.get_laya = lambda: 'patched'\n"  # what monkeypatch.setattr(neo.reflex, …) does
        "stub = types.ModuleType('stub')\n"
        "sys.modules['neo2.core.decide.inventory'] = stub\n"
        "import neo.decide.inventory as inv\n"
        "sys.modules['neo2.core.reflex.outcomes'] = None\n"  # blocked: an ImportError, never a copy
        "try:\n"
        "    import neo.reflex.outcomes\n"
        "    blocked = False\n"
        "except ImportError:\n"
        "    blocked = True\n"
        "print(json.dumps({'patched': neo2.core.reflex.get_laya(), 'stub': inv is stub, 'blocked': blocked,\n"
        "                  'copy': 'neo.reflex.outcomes' in sys.modules}))\n",
        tmp_path,
    )
    assert got == {"patched": "patched", "stub": True, "blocked": True, "copy": False}


def test_the_first_import_warns_once(tmp_path):
    got = _run(
        "import json, warnings\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    import neo\n"
        "    import neo.tools\n"
        "    from neo import runtime\n"
        "    import neo.agent.registry\n"
        "ours = [w for w in caught if issubclass(w.category, DeprecationWarning) and 'neo2.core' in str(w.message)]\n"
        "print(json.dumps({'n': len(ours), 'at': [w.filename for w in ours]}))\n",
        tmp_path,
    )
    assert got == {"n": 1, "at": ["<string>"]}  # once, pointing at the line that imported it


def test_neo2_never_loads_the_shim(tmp_path):
    got = _run(
        "import json, sys\n"
        "import neo2\n"
        "from neo2 import AsyncAgent\n"
        "import neo2.core.tools.web\n"
        "print(json.dumps({'neo': 'neo' in sys.modules, 'aliases': [m for m in sys.modules if m.startswith('neo.')]}))\n",
        tmp_path,
    )
    assert got == {"neo": False, "aliases": []}


def test_python_m_neo_still_starts_the_core_and_says_where_it_went(tmp_path):
    old = _python(["-m", "neo", "--help"], tmp_path)
    new = _python(["-m", "neo2", "--help"], tmp_path)
    assert old.returncode == 0 and new.returncode == 0, old.stderr[-2000:] + new.stderr[-2000:]
    assert old.stdout == new.stdout and new.stdout.startswith("usage: neo2 ")
    assert old.stderr.count("DeprecationWarning") == 1 and "python -m neo2" in old.stderr
    assert "DeprecationWarning" not in new.stderr


def test_python_m_neo_module_runs_the_moved_module(tmp_path):
    old = _python(["-m", "neo.reflex.hf_data", "--help"], tmp_path)
    new = _python(["-m", "neo2.core.reflex.hf_data", "--help"], tmp_path)
    assert old.returncode == 0 and new.returncode == 0, old.stderr[-2000:] + new.stderr[-2000:]
    assert old.stdout == new.stdout and "--only" in new.stdout
    missing = _python(["-m", "neo.reflex.not_a_module"], tmp_path)
    assert missing.returncode != 0 and "No module named neo.reflex.not_a_module" in missing.stderr
