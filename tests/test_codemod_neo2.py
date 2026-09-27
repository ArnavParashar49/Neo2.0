"""scripts/codemod_neo2.py: the old `neo` package's paths become neo2.core's, and nothing else moves.
neo2-codemod: skip-file (the samples below are the old paths on purpose)

It runs again after every merge of a branch that still has the old paths, so each rewrite is
checked to be a fixed point, and the repo itself is checked to need nothing more."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from neo2.core import effects

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("codemod_neo2", ROOT / "scripts" / "codemod_neo2.py")
cm = importlib.util.module_from_spec(_spec)
sys.modules["codemod_neo2"] = cm  # dataclasses look their module up
_spec.loader.exec_module(cm)

SUBS = {"agent", "app", "config", "decide", "effects", "events", "main", "providers", "reflex", "runtime", "tools",
        "privacy", "__main__", "log_to_stdout", "logger"}  # fmt: skip
CTX = cm.Ctx(frozenset(SUBS), frozenset({"web", "mcp", "openai", "gemini", "claude", "brains", "app", "all"}))


def py(src: str):
    out = cm.rewrite_python(src, CTX)
    assert cm.rewrite_python(out.text, CTX).text == out.text  # a second run changes nothing
    return out


def text(src: str, name: str = "README.md"):
    out = cm.rewrite_file(name, src, CTX)
    assert cm.rewrite_file(name, out.text, CTX).text == out.text
    return out


def test_python_imports():
    src = (
        "from neo import runtime, effects\n"
        "from neo.agent.registry import tool as t\n"
        "import neo.tools.web as web\n"
        "import neo as core, os\n"
        "from . import sibling\n"
        "from .neo import x\n"
        "from neo2.sdk import AsyncAgent\n"
        "from neo2.core.config import settings\n"
        "def f():\n"
        "    from neo.config import (\n"
        "        settings,\n"
        "    )\n"
        "    if True: from neo.reflex import zeroshot\n"
        "    return settings\n"
    )
    out = py(src)
    assert out.text == (
        "from neo2.core import runtime, effects\n"
        "from neo2.core.agent.registry import tool as t\n"
        "import neo2.core.tools.web as web\n"
        "import neo2.core as core, os\n"
        "from . import sibling\n"
        "from .neo import x\n"
        "from neo2.sdk import AsyncAgent\n"
        "from neo2.core.config import settings\n"
        "def f():\n"
        "    from neo2.core.config import (\n"
        "        settings,\n"
        "    )\n"
        "    if True: from neo2.core.reflex import zeroshot\n"
        "    return settings\n"
    )
    assert out.counts == {"import": 6} and out.refused == []


def test_a_plain_import_neo_rewrites_its_references_in_that_file():
    src = (
        "import neo.reflex\n"
        "def test(mp):\n"
        "    import neo\n"
        "    mp.setattr(neo.reflex, 'get_laya', None)\n"
        "    obj.neo = neo.log_to_stdout\n"
        "    return f'{neo.reflex}'\n"
    )
    out = py(src)
    assert out.text == (
        "import neo2.core.reflex\n"
        "def test(mp):\n"
        "    import neo2.core\n"
        "    mp.setattr(neo2.core.reflex, 'get_laya', None)\n"
        "    obj.neo = neo2.core.log_to_stdout\n"
        "    return f'{neo2.core.reflex}'\n"
    )
    assert out.refused == []


def test_neo_imported_and_also_a_plain_name_is_refused():
    src = "import neo.reflex\n\ndef f(neo):\n    return neo.ask()\n"
    out = py(src)
    assert [line for line, _ in out.refused] == [3] and "plain name" in out.refused[0][1]


def test_strings_that_name_the_package():
    src = (
        'mock.patch("neo.tools.web.fetch")\n'
        'monkeypatch.setattr("neo.decide.axkit._pid_of", None)\n'
        'importlib.import_module("neo.decide.kinds.note")\n'
        'importlib.import_module("neo")\n'
        'pytest.importorskip("neo.providers.gemini")\n'
        'sys.modules["neo.decide.inventory"] = stub\n'
        'monkeypatch.setitem(sys.modules, "neo", stub)\n'
        'subprocess.run([sys.executable, "-m", "neo"])\n'
        "subprocess.run([sys.executable, '-m', 'neo.reflex.hf_data', '--help'])\n"
        'p = ROOT / "neo" / "tools" / "web.py"\n'
        'p = ROOT / "neo" / "effects.py"\n'
        'q = ROOT / "neo/tools/web.py"\n'
        'MODULES = ["neo.tools.system", "neo.tools.web"]\n'
        "# python -m neo --text; see neo/reflex/sources.py and neo.agent.grants\n"
        'x = f"run it inside `with neo.effects.allow({kind!r}):`"\n'
        '"""`python -m neo.reflex.<module>` and ``from neo import runtime``"""\n'
    )
    out = py(src)
    assert out.text == (
        'mock.patch("neo2.core.tools.web.fetch")\n'
        'monkeypatch.setattr("neo2.core.decide.axkit._pid_of", None)\n'
        'importlib.import_module("neo2.core.decide.kinds.note")\n'
        'importlib.import_module("neo2.core")\n'
        'pytest.importorskip("neo2.core.providers.gemini")\n'
        'sys.modules["neo2.core.decide.inventory"] = stub\n'
        'monkeypatch.setitem(sys.modules, "neo2.core", stub)\n'
        'subprocess.run([sys.executable, "-m", "neo2"])\n'
        "subprocess.run([sys.executable, '-m', 'neo2.core.reflex.hf_data', '--help'])\n"
        'p = ROOT / "neo2" / "core" / "tools" / "web.py"\n'
        'p = ROOT / "neo2" / "core" / "effects.py"\n'
        'q = ROOT / "neo2/core/tools/web.py"\n'
        'MODULES = ["neo2.core.tools.system", "neo2.core.tools.web"]\n'
        "# python -m neo2 --text; see neo2/core/reflex/sources.py and neo2.core.agent.grants\n"
        'x = f"run it inside `with neo2.core.effects.allow({kind!r}):`"\n'
        '"""`python -m neo2.core.reflex.<module>` and ``from neo2.core import runtime``"""\n'
    )
    assert out.refused == []


def test_what_is_never_touched():
    src = (
        'USER_ENV = "~/.neo/.env"\n'
        'DATA = Path.home() / ".neo" / "logs"\n'
        'os.environ["NEO_DATA_DIR"] = "$HOME/.neo/x"\n'
        'BUNDLE = "com.arnav.neo"\n'
        'URL = "https://github.com/ArnavParashar49/Neo2.0"\n'
        'OLD = "https://github.com/ArnavParashar49/NEO"\n'
        "from neo2.sdk import AsyncAgent\n"
        "import neo2.core.tools\n"
        'S = "neon lights, neo-ui, neo_lib, neo2.mcp, neo2-mcp, python -m neo2, pip install neo2"\n'
        'logging.getLogger("neo").info("x")\n'
        'logging.getLogger("neo.audit").info("x")\n'
        'app.emit("neo://wake")\n'
        'r = neo.ask("hi")  # neo is an agent here: no import of the package in this file\n'
        'tools = neo.tools  # the same\n'
        '"""NEO prose, NEO.app, and neo2.core.runtime already moved."""\n'
        'tmp_path / "neo"\n'
        'uid = f"{uuid}@neo.local"\n'
        'controller = "neo"\n'
        're.compile(r"^(?:hey\\s+)?neo[\\s,!.:-]*")\n'
        'SELF = re.compile(r"^neo2\\[([a-z0-9_,-]+)\\]$")\n'
        'log = logging.getLogger("neo.providers")  # NEO still logs as neo.<module>\n'
        'caplog.set_level(logging.INFO, logger="neo.agent.session")\n'
        'with caplog.at_level(20, logger = "neo.tools.web"): pass\n'
    )
    out = py(src)
    assert out.text == src and out.counts == {} and out.refused == []
    md = 'Quiet it with `logging.getLogger("neo.providers").setLevel(...)` or `caplog.set_level(20, logger="neo.agent")`.\n'
    assert text(md).text == md


def test_docs_config_and_scripts():
    md = (
        ".venv/bin/python -m neo --doctor\n"
        ".venv/bin/python -m neo.reflex.finetune   # train\n"
        'pip install "neo[web,mcp] @ git+https://github.com/ArnavParashar49/Neo2.0"\n'
        "pip install neo\n"
        "Extras: `neo[...]`. See `neo/tools/web.py`, `./neo/decide/` and `neo.agent.registry`.\n"
        "Data lives in `~/.neo/`, set by NEO_DATA_DIR; NEO.app is com.arnav.neo.\n"
        "from neo.agent.registry import tool\n"
        "from neo import runtime\n"
        "The agent: `r = await neo.ask(...)`, `neo.approve(...)`.\n"
    )
    out = text(md)
    assert out.text == (
        ".venv/bin/python -m neo2 --doctor\n"
        ".venv/bin/python -m neo2.core.reflex.finetune   # train\n"
        'pip install "neo2[web,mcp] @ git+https://github.com/ArnavParashar49/Neo2.0"\n'
        "pip install neo2\n"
        "Extras: `neo2[...]`. See `neo2/core/tools/web.py`, `./neo2/core/decide/` and `neo2.core.agent.registry`.\n"
        "Data lives in `~/.neo/`, set by NEO_DATA_DIR; NEO.app is com.arnav.neo.\n"
        "from neo2.core.agent.registry import tool\n"
        "from neo2.core import runtime\n"
        "The agent: `r = await neo.ask(...)`, `neo.approve(...)`.\n"
    )
    assert out.refused == []


def test_rust_and_typescript():
    rs = (
        "//! A core started some other way (a developer's `python -m neo`) is used as-is.\n"
        '        .args(["-u", "-m", "neo"])\n'
        '    home().join(".neo").join("core.pid")\n'
        '    let _ = app.emit("neo://wake", ());\n'
        '    TrayIconBuilder::with_id("neo")\n'
    )
    out = text(rs, "ui/src-tauri/src/core.rs")
    assert out.text == rs.replace("python -m neo`", "python -m neo2`").replace('"-m", "neo"]', '"-m", "neo2"]')
    ts = (
        'const s = neo.connected ? "ok" : "core offline — run  python -m neo";\n'
        "if (neo.tools.length && neo.config) listen(\"neo://erase\");\n"
    )
    out = text(ts, "ui/src/App.tsx")  # `neo` is a variable there: only command lines and paths
    assert out.text == ts.replace("python -m neo\"", "python -m neo2\"")


def test_what_it_refuses():
    out = text(
        "run `-m neo` yourself\n"
        "import neo\n"
        "neo[a,b] is a placeholder\n"
        "keep `-m neo` here  neo2-codemod: keep\n"
    )
    assert [line for line, _ in out.refused] == [1, 2, 3] and out.counts == {}
    out = py('if name.startswith("neo."):\n    pass\n')
    assert [line for line, _ in out.refused] == [1] and out.text.startswith('if name.startswith("neo.")')
    out = py('for pkg in ("neo2", "neo"):\n    pass\nassert "/site/neo/" in f\nURL = "https://x.org/neo/y"\n')
    assert [line for line, _ in out.refused] == [1, 3]
    out = py("ROOT = Path(effects.__file__).resolve().parent.parent\n")  # was the checkout, now neo2/
    assert [line for line, _ in out.refused] == [1]
    assert cm.rewrite_python("ROOT = Path(__file__).resolve().parent.parent\n", CTX, in_core=True).refused
    assert not py("ROOT = Path(__file__).resolve().parents[1]\n").refused  # a test's own file
    assert not py("ROOT = Path(effects.__file__).resolve().parents[2]\n").refused
    three_up = "root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n"
    assert not cm.rewrite_python(three_up, CTX, in_core=True).refused
    assert cm.rewrite_python(three_up.replace("os.path.dirname(os.path.abspath", "os.path.abspath"), CTX, in_core=True).refused
    out = py("def f(:\n")  # not Python: left alone, and said so
    assert out.refused and out.text == "def f(:\n"


def test_keep_and_skip_pragmas():
    src = (
        'OLD = "neo.agent.session"  # a logger name  neo2-codemod: keep\n'
        'NEW = "neo.agent.session"\n'
    )
    assert py(src).text == src.replace('NEW = "neo.', 'NEW = "neo2.core.')
    skipped = '"""A file about the old name.\nneo2-codemod: skip-file\n"""\nfrom neo import runtime\n'
    assert py(skipped).text == skipped


def test_pyproject():
    toml = (
        '[project]\nname = "neo"\nversion = "2.1.0"\n'
        '[project.optional-dependencies]\nweb = ["ddgs"]\napp = ["neo[web]"]\n'
        '[project.scripts]\nneo2-mcp = "neo2.mcp:main"\n'
        '[tool.setuptools.packages.find]\ninclude = ["neo*"]\n'
    )
    ctx = cm.Ctx(CTX.subs, frozenset({"web", "app"}))
    out = cm.rewrite_pyproject(toml, ctx)
    assert out.text == toml.replace('name = "neo"', 'name = "neo2"').replace('"neo[web]"', '"neo2[web]"').replace(
        '"neo*"', '"neo2*"'
    )
    assert out.refused == []
    out = cm.rewrite_pyproject(toml.replace('neo2-mcp = "neo2.mcp:main"', 'neo = "neo.main:main"'), ctx)
    assert any("`neo` command" in why for _, why in out.refused)



def test_the_package_folder_by_name_is_refused():
    """After the move neo/ holds only the shim: a path or a tool pointed at it checks nothing."""
    out = py(
        '(ROOT / "neo").rglob("*.py")\n'
        'p = Path(__file__).resolve().parents[1] / "neo"\n'
        'q = ROOT / "neo" / "data"\n'
        'r = os.path.join(ROOT, "neo", "tools")\n'
        's = Path("neo")\n'
        'subprocess.run([sys.executable, "-m", "mypy", "neo"])\n'
        'subprocess.run(["ruff", "check", "neo"])\n'
        'CMD = "pytest --cov=neo tests"\n'
    )
    assert [line for line, _ in out.refused] == [1, 2, 3, 4, 5, 6, 7, 8], out.refused
    assert all("only the compatibility shim" in why for _, why in out.refused) and out.counts == {}
    fine = (
        'd = tmp_path / "neo"\n'
        'e = tmp_path / "myapp" / "neo"\n'
        'f = os.path.join(tmp, "neo")\n'
        'print(", ".join(["neo", "x"]))\n'
        'controller = "neo"\n'
        'p = ROOT / "neo" / "tools"\n'  # rewritten, not refused
    )
    out = py(fine)
    assert out.refused == [] and out.counts == {"path": 1}
    out = text(
        "ruff check neo tests\n"
        "python -m compileall -q neo\n"
        "mypy neo2 tests\n"  # neo2: fine
        "pytest --cov=neo\n"
        '[tool.coverage.run]\nsource = ["neo"]\n'
        "ruff check neo neo2  # neo2-codemod: keep\n",
        "ci.yml",
    )
    assert [line for line, _ in out.refused] == [1, 2, 4, 6], out.refused
    rs = 'let tools = root.join("neo").join("tools");\nhome().join(".neo").join("core.pid")\n'
    assert [line for line, _ in text(rs, "ui/src-tauri/src/core.rs").refused] == [1]


def core(src: str):
    out = cm.rewrite_python(src, CTX, in_core=True)
    assert cm.rewrite_python(out.text, CTX, in_core=True).text == out.text
    return out


def test_neos_modules_keep_logging_as_neo_dot_module():
    """A merged branch's new module in NEO's package with `logging.getLogger(__name__)` would log as
    neo2.core.…, outside the `neo` logger NEO.app prints to core.log: it gets neo2.core's logger()."""
    out = core(
        '"""A module."""\n\nfrom __future__ import annotations\n\nimport asyncio\nimport logging\n\n'
        "from neo import runtime\nfrom neo.agent import session\n\nlog = logging.getLogger(__name__)\n"
    )
    assert out.text == (
        '"""A module."""\n\nfrom __future__ import annotations\n\nimport asyncio\n\n'
        "from neo2.core import logger, runtime\nfrom neo2.core.agent import session\n\nlog = logger(__name__)\n"
    )
    assert out.refused == []
    # its own line, where isort puts it; `import logging` stays while something else uses it
    out = core("import logging\n\nfrom neo.config import settings\n\nlog = logging.getLogger(__name__)\nlogging.basicConfig()\n")
    assert out.text == (
        "import logging\n\nfrom neo2.core import logger\nfrom neo2.core.config import settings\n\n"
        "log = logger(__name__)\nlogging.basicConfig()\n"
    )
    out = core("import logging\n\nlog = logging.getLogger(__name__)\n")
    assert out.text == "from neo2.core import logger\n\nlog = logger(__name__)\n"
    out = core("import logging\n\nfrom neo2.core import logger, runtime\n\nlog = logging.getLogger(__name__)\n")
    assert out.text == "\nfrom neo2.core import logger, runtime\n\nlog = logger(__name__)\n"[1:]
    # outside NEO's package a module's own name is right: the SDK logs as neo2.sdk.…
    assert py("import logging\n\nlog = logging.getLogger(__name__)\n").counts == {}
    # what it can't decide
    out = core("import logging\n\nlogger = logging.getLogger(__name__)\n")
    assert [line for line, _ in out.refused] == [3] and "another name" in out.refused[0][1]
    out = core("from logging import getLogger\n\nlog = getLogger(__name__)\n")
    assert [line for line, _ in out.refused] == [3] and "logger(__name__)" in out.refused[0][1]


def test_the_rewritten_core_logs_under_neo():
    """The rewrite runs: the module's logger is neo.<module>, a child of the `neo` logger."""
    out = core("import logging\n\nfrom neo import runtime\n\nlog = logging.getLogger(__name__)\n")
    ns = {"__name__": "neo2.core.tools.computer.newmod"}
    exec(compile(out.text, "newmod.py", "exec"), ns)
    assert ns["log"].name == "neo.tools.computer.newmod" and ns["log"].parent.name.startswith("neo")


def test_relocate_moves_what_a_merge_left_in_the_old_folder(tmp_path):
    core, old = tmp_path / "neo2" / "core", tmp_path / "neo"
    (core / "tools").mkdir(parents=True)
    old.mkdir()
    (core / "__init__.py").write_text("")
    (core / "tools" / "web.py").write_text("A = 1\n")
    (old / "__init__.py").write_text('"""neo2-codemod: shim"""\n')
    (old / "tools").mkdir()
    (old / "tools" / "new.py").write_text("NEW = 1\n")  # a branch added it
    (old / "tools" / "web.py").write_text("A = 1\n")  # the same as the moved one
    with effects.allow("process"):  # `git ls-files`: not a repo here, so plain moves
        moved, refused = cm.relocate(tmp_path, dry=False)
    assert refused == [] and len(moved) == 2
    assert (core / "tools" / "new.py").read_text() == "NEW = 1\n" and not (old / "tools" / "new.py").exists()
    assert not (old / "tools" / "web.py").exists() and (old / "__init__.py").exists()

    (old / "tools" / "web.py").write_text("A = 2\n")  # changed on a branch: a person merges it
    (old / "__main__.py").write_text("from neo.main import main\n")  # the shim replaced by a merge
    with effects.allow("process"):
        moved, refused = cm.relocate(tmp_path, dry=False)
    assert moved == [] and len(refused) == 2 and (old / "tools" / "web.py").exists()


def test_relocate_before_the_rename_moves_the_package(tmp_path):
    (tmp_path / "neo" / "tools").mkdir(parents=True)
    (tmp_path / "neo" / "__init__.py").write_text("")
    (tmp_path / "neo" / "tools" / "web.py").write_text("")
    with effects.allow("process"):
        moved, refused = cm.relocate(tmp_path, dry=False)
    assert refused == [] and moved and (tmp_path / "neo2" / "core" / "tools" / "web.py").exists()


def _pre_move_tree(root: Path) -> None:
    (root / "neo" / "tools").mkdir(parents=True)
    (root / "neo" / "__init__.py").write_text("")
    (root / "neo" / "app.py").write_text("from neo import tools\n")
    (root / "neo" / "tools" / "__init__.py").write_text("")
    (root / "neo" / "tools" / "web.py").write_text("")
    # a branch switch from the moved tree leaves its git-ignored bytecode (and Finder's .DS_Store)
    for d in ("neo2/core/__pycache__", "neo2/core/tools/__pycache__", "neo2/core/agent/__pycache__"):
        (root / d).mkdir(parents=True)
        (root / d / "x.cpython-312.pyc").write_bytes(b"\0")
    (root / "neo2" / "core" / ".DS_Store").write_bytes(b"\0")


@pytest.mark.parametrize("in_git", [False, True])
def test_relocate_before_the_rename_over_a_stale_neo2_core(tmp_path, in_git):
    _pre_move_tree(tmp_path)
    with effects.allow("process"):
        if in_git:  # `git mv neo neo2/core` onto an existing folder would have made neo2/core/neo/
            cm.git(tmp_path, "init", "-q")
            cm.git(tmp_path, "add", "neo")
        moved, refused = cm.relocate(tmp_path, dry=False)
    core = tmp_path / "neo2" / "core"
    assert refused == [] and moved == ["neo/ → neo2/core/ (the whole package)"]
    assert (core / "__init__.py").exists() and (core / "tools" / "web.py").exists()
    assert not (core / "neo").exists() and not (tmp_path / "neo").exists()
    assert cm.package_ctx(tmp_path).subs >= {"app", "tools"}  # what the rewrite then knows
    if in_git:
        with effects.allow("process"):
            staged = cm.git(tmp_path, "ls-files").stdout.split()
        assert staged == [f"neo2/core/{f}" for f in ("__init__.py", "app.py", "tools/__init__.py", "tools/web.py")]


def test_relocate_refuses_a_neo2_core_that_holds_more_than_caches(tmp_path):
    _pre_move_tree(tmp_path)
    (tmp_path / "neo2" / "core" / "stray.py").write_text("")
    with effects.allow("process"):
        moved, refused = cm.relocate(tmp_path, dry=False)
    assert moved == [] and len(refused) == 1 and "stray.py" in refused[0]
    assert (tmp_path / "neo" / "app.py").exists() and not (tmp_path / "neo2" / "core" / "neo").exists()


def test_main_says_to_fix_the_import_order(tmp_path, capsys):
    (tmp_path / "neo" / "tools").mkdir(parents=True)
    (tmp_path / "neo" / "__init__.py").write_text("")
    (tmp_path / "neo" / "tools" / "web.py").write_text("from neo import app\nfrom neo2 import x\n")
    (tmp_path / "neo" / "app.py").write_text("")
    with effects.allow("process"):
        assert cm.main(["--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "then: ruff check --fix" in out
    assert (tmp_path / "neo2" / "core" / "tools" / "web.py").read_text() == "from neo2.core import app\nfrom neo2 import x\n"
    with effects.allow("process"):
        assert cm.main(["--root", str(tmp_path)]) == 0
    assert "then:" not in capsys.readouterr().out  # nothing changed: nothing to fix


def test_a_leftover_that_differs_says_how_to_merge(tmp_path):
    core, old = tmp_path / "neo2" / "core", tmp_path / "neo"
    core.mkdir(parents=True)
    old.mkdir()
    (core / "__init__.py").write_text("")
    (core / "config.py").write_text("A = 1\n")
    (old / "__init__.py").write_text('"""neo2-codemod: shim"""\n')
    (old / "config.py").write_text("A = 2\n")  # modify/delete: the branch's whole file
    with effects.allow("process"):
        _, refused = cm.relocate(tmp_path, dry=False)
    assert len(refused) == 1 and "merge c475799" in refused[0]


def test_a_file_git_cant_move_yet_is_listed_not_a_crash(tmp_path):
    """A merge conflict under neo/ (both sides added the file): `git mv` refuses it until it's resolved."""
    def g(*args: str) -> None:
        cm.git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", *args)

    (tmp_path / "neo").mkdir()
    (tmp_path / "neo" / "__init__.py").write_text("A = 1\n")
    with effects.allow("process"):
        g("init", "-q", "-b", "main")
        g("add", "-A")
        g("commit", "-qm", "base")
        g("checkout", "-qb", "branch")
        (tmp_path / "neo" / "new.py").write_text("X = 1\n")
        g("add", "-A")
        g("commit", "-qm", "branch")
        g("checkout", "-q", "main")
        (tmp_path / "neo" / "new.py").write_text("X = 2\n")
        g("add", "-A")
        g("commit", "-qm", "main")
        cm.git(tmp_path, "merge", "branch", check=False)  # add/add conflict in neo/new.py
        (tmp_path / "neo2" / "core").mkdir(parents=True)
        (tmp_path / "neo2" / "core" / "__init__.py").write_text("")
        (tmp_path / "neo" / "__init__.py").write_text('"""neo2-codemod: shim"""\n')
        moved, refused = cm.relocate(tmp_path, dry=False)
    assert moved == [] and len(refused) == 1 and refused[0].startswith("neo/new.py:") and "git add" in refused[0]

@pytest.mark.skipif(not (ROOT / ".git").exists(), reason="not a git checkout")
def test_the_repo_needs_nothing_more(capsys):
    with effects.allow("process"):  # git ls-files
        code = cm.main(["--check", "--root", str(ROOT)])
    out = capsys.readouterr().out
    assert code == 0, out  # run `python scripts/codemod_neo2.py` (after a merge) and review its diff
    assert "would rewrite 0 references in 0 files" in out
