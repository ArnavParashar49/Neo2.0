#!/usr/bin/env python3
"""Move references to NEO's old top-level package `neo` over to `neo2.core` (and the distribution
`neo` over to `neo2`). neo2-codemod: skip-file

    python scripts/codemod_neo2.py              # rewrite in place and print what changed
    python scripts/codemod_neo2.py --check      # change nothing: exit 1 if anything is left to do
    python scripts/codemod_neo2.py -v PATH…     # only these files or folders, listing each file

Safe to run again, on any branch, as often as needed: what it already rewrote doesn't match any
more. A branch that still has the old paths:

    git merge c475799                  # first main just before the move, if the branch is older
    git merge main                     # then the move: git follows neo/ → neo2/core/ (ordinary conflicts)
    python scripts/codemod_neo2.py     # then `git add` what it moved and any "file location" conflict
    ruff check --fix neo neo2 tests scripts   # a rewritten import can sort differently

Merged straight into an older branch, the move loses git's rename detection for files the branch
changed a lot: they come back whole at neo/… (modify/delete), which this lists to port by hand.

1. Files under `neo/` that aren't the repo-only compatibility shim (a merged branch added or changed
   them there) move to `neo2/core/` (`git mv`). Before the rename, the whole package moves (over a
   neo2/core/ holding only __pycache__ left by a branch switch; anything else there is refused).
2. Python, by tokens: `import neo`, `import neo.x [as y]`, `from neo[.x] import …` (relative imports
   are left alone); after a plain `import neo[.x]`, the `neo.x…` references in that file; and in
   strings and comments the text rules below, plus "neo" as a module argument
   (`import_module("neo")`, `sys.modules["neo"]`, `"-m", "neo"`) and `ROOT / "neo" / "tools"`.
3. Everything else (docs, pyproject, CI, scripts, Rust; TypeScript only for command lines and paths):
   - module paths `neo.<a module of the package>` (mock.patch targets, import_module, -m neo.x.y…)
   - `python -m neo` and `"-m", "neo"` → neo2
   - `neo/…` paths to the package folder (never `~/.neo/…`, `…/neo/…` or URLs)
   - NEO's own extras `neo[web,mcp]` → `neo2[…]`, and `pip install neo` → `neo2`
   - pyproject.toml: `name = "neo"`, `include = ["neo*"]`
4. In NEO's own package, `logging.getLogger(__name__)` → `logger(__name__)` from neo2.core: its
   modules log as `neo.<module>` (the `neo` logger NEO.app prints), and __name__ is neo2.core.… now.

Never touched: ~/.neo, NEO_* names, NEO in prose, com.arnav.neo, URLs, neo2.*, neon, neo-ui, neo_lib,
logger names (`neo`, `neo.audit`, `getLogger("neo.x")`, `logger="neo.x"`: NEO still logs as neo.*),
and `neo` as a variable (`neo.ask(…)`).

Anything it can't rewrite safely is refused and listed (exit 1): a bare `import neo` in prose, `-m neo`
outside a Python command line, `"neo."` used as a prefix, `neo[…]` naming unknown extras, `neo` both
imported and used as a plain name in one file, files that differ in neo/ and neo2/core/, pyproject
entries only a person can rename, the folder by name where only the shim is left (`ROOT / "neo"`,
`os.path.join(ROOT, "neo")`, `Path("neo")`, `.join("neo")`, `mypy neo`, `--cov=neo`,
`packages = ["neo"]`), and in NEO's package a getLogger(__name__) it can't make logger(__name__).
A line that must keep an old name carries `neo2-codemod: keep`; a file that must be skipped says
`neo2-codemod: skip-file` in its first lines.
"""

from __future__ import annotations

import argparse
import ast
import filecmp
import io
import os
import re
import shutil
import subprocess
import sys
import tokenize
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

OLD = "neo"
NEW = "neo2.core"
NEW_DIR = "neo2/core"
DIST = "neo2"
KEEP = "neo2-codemod: keep"
SKIP_FILE = "neo2-codemod: skip-file"
SHIM = "neo2-codemod: shim"  # marks the compatibility shim's own files (neo/__init__.py, neo/__main__.py)
FIX_ORDER = "ruff check --fix neo neo2 tests scripts   (import order)"
PRE_MOVE = "c475799"  # main just before the move: a branch older than that merges it first
MERGE_ORDER = (f"merge {PRE_MOVE} (main just before the move) into the branch first, then the move, "
               "then run this: git follows the move, and only ordinary conflicts are left")  # fmt: skip

TEXT_SUFFIXES = {".py", ".pyi", ".md", ".rst", ".txt", ".toml", ".cfg", ".ini", ".yml", ".yaml", ".sh",
                 ".bash", ".zsh", ".rs", ".ts", ".tsx", ".js", ".mjs", ".cjs", ".json", ".html", ".example"}  # fmt: skip
SKIP_NAMES = {"package-lock.json", "requirements.lock", "Cargo.lock", "uv.lock", "poetry.lock"}
SKIP_PREFIXES = ("tests/fixtures/", "ui/node_modules/", "ui/src-tauri/target/", "ui/src-tauri/gen/", "ui/dist/",
                 "build/", "dist/", ".venv/", ".git/")  # fmt: skip
LIGHT = {
    ".ts",
    ".tsx",
    ".js",
    ".mjs",
    ".cjs",
}  # `neo` is a common variable there: command lines and paths only

# `neo` on its own: not part of a longer name, a dotted path (x.neo, ~/.neo), $VAR or a path (a/neo)
_ALONE = r"(?<![\w.~$/-])"
R_MODULE = re.compile(_ALONE + r"neo\.(?P<sub>[A-Za-z_]\w*)(?![\w-])")
R_PATH = re.compile(r"(?:(?<=\./)|" + _ALONE + r")neo/")
R_SUBDIR = re.compile(r"(?<=/)(?<!\./)neo/")  # …/neo/…: site-packages/neo/? (a URL's path is left alone)
R_M_PY = re.compile(r"(?P<pre>\bpython[\w.]*\b[^\n]*?(?<![\w-])-m\s+)neo(?![\w.-])")
R_M_ARGS = re.compile(r"(?P<pre>(?P<q1>[\"'])-m(?P=q1)\s*,\s*)(?P<q2>[\"'])neo(?P=q2)")
R_M_BARE = re.compile(r"(?<![\w-])-m\s+neo(?![\w.-])")
R_EXTRAS = re.compile(_ALONE + r"neo\[(?P<x>[^\]\n]{1,80})\]")
R_EXTRA_LIST = re.compile(r"^\s*[a-z0-9_-]+(\s*,\s*[a-z0-9_-]+)*\s*$")
R_PIP = re.compile(r"\b(?:pip3?|uv\s+pip|pipx)\s+install\b[^\n]*")
R_PIP_NEO = re.compile(_ALONE + r"neo(?=\s*@|[\s\"'`]|$)")
R_FROM = re.compile(r"(?<![\w.])from\s+neo\s+import\b")
R_IMPORT_BARE = re.compile(r"(?<![\w.])import\s+neo(?![\w.])")
R_TOML_NAME = re.compile(r'(?m)^(?P<pre>name\s*=\s*)"neo"(?P<post>\s*(#.*)?)$')
R_TOML_INCLUDE = re.compile(r'(?m)^(?P<pre>include\s*=\s*\[[^\]\n]*?)"neo\*"')

# two folders up from a module's file: the checkout from neo/x.py, only neo2/ from neo2/core/x.py
R_UP_TWO = re.compile(
    r"(?<!\.parent)\.parent\.parent\b(?!\.parent)|\.parents\[1\]"
    r"|(?<!dirname\(os\.path\.)dirname\(\s*os\.path\.dirname\((?!\s*os\.path\.dirname)"
)
R_MODULE_FILE = re.compile(r"\w\.__file__\b")
MODULE_FUNCS = {"import_module", "importorskip", "find_spec", "__import__", "reload", "run_module"}

# A logger name, not a module path: NEO's modules still log as neo.<module> (see neo2/core/__init__.py)
R_LOGGER_ARG = re.compile(r"""(?:\bgetLogger\(\s*|\blogger\s*=\s*)["']$""")
# The package folder by name, as a path or a tool's argument: after the move it holds only the shim
TOOLS = {"ruff", "mypy", "pyright", "pytest", "compileall", "coverage", "pylint", "flake8", "pyflakes", "black",
         "isort", "vulture", "bandit", "pdoc", "pydoc"}  # fmt: skip
# a tool and its arguments: words up to the end of the line, a comment or a sentence's punctuation
R_TOOL_LINE = re.compile(r"(?<![\w.-])(?:" + "|".join(sorted(TOOLS)) + r")(?![\w.-])(?:[ \t]+(?!#)[^\s;()#]+)*")
R_BARE_NEO = re.compile(r"""(?<=[\s"'`=,\[])neo(?=[\s"'`,\]]|$)""")
R_COV = re.compile(r"--(?:cov|source)(?:=|\s+)neo(?![\w./-])")
R_TOML_LIST = re.compile(
    r"""(?m)^\s*(?:source|packages|src|include|known[-_]first[-_]party|py[-_]modules)\s*=\s*\[[^\]\n]*?["']neo["']"""
)
R_JOIN_NEO = re.compile(r"""\bjoin\([^()\n]*?["']neo["']""")
R_PATH_NEO = re.compile(r"""\b(?:Path|PurePath|PosixPath|PathBuf::from)\(\s*["']neo["']""")
FOLDER = "only the compatibility shim is in neo/ now (neo2/core, neo2): rewrite by hand or mark it"
REPO_NAMES = {"root", "repo", "repo_root", "here", "base", "base_dir", "project", "project_root", "project_dir",
              "checkout", "src", "top", "pkg_root", "__file__", "parent", "parents", "cwd"}  # fmt: skip


@dataclass
class Ctx:
    subs: frozenset[str]  # the package's top-level modules and public names: agent, tools, log_to_stdout…
    extras: frozenset[str]  # the distribution's extras (pyproject's optional-dependencies)


@dataclass
class Edit:
    start: int
    end: int
    new: str
    rule: str


@dataclass
class Outcome:
    text: str
    counts: Counter = field(default_factory=Counter)
    refused: list[tuple[int, str]] = field(default_factory=list)  # (line, why)


# ---- text rules ------------------------------------------------------------------------------------


def text_edits(
    text: str, ctx: Ctx, base: int = 0, *, light: bool = False
) -> tuple[list[Edit], list[tuple[int, str]]]:
    """Edits (offsets + base) and refusals (offsets + base) for one stretch of prose, config or code
    that isn't Python code: a whole doc, or one Python string or comment."""
    edits: list[Edit] = []
    refused: list[tuple[int, str]] = []
    taken: list[tuple[int, int]] = []

    def free(a: int, b: int) -> bool:
        return all(b <= x or a >= y for x, y in taken)

    def add(a: int, b: int, new: str, rule: str) -> None:
        if free(a, b):
            taken.append((a, b))
            edits.append(Edit(base + a, base + b, new, rule))

    for m in R_M_PY.finditer(text):
        add(m.end("pre"), m.end(), DIST, "-m neo")
    for m in R_M_ARGS.finditer(text):
        a = m.start("q2") + 1
        add(a, a + 3, DIST, "-m neo")
    for m in R_M_BARE.finditer(text):
        if free(m.start(), m.end()):
            refused.append(
                (base + m.start(), "`-m neo` outside a Python command line: rewrite it by hand or mark it")
            )
    for m in R_PIP.finditer(text):
        for n in R_PIP_NEO.finditer(m.group()):
            add(m.start() + n.start(), m.start() + n.end(), DIST, "distribution")
    for m in R_EXTRAS.finditer(text):
        inner = m.group("x").strip()
        if inner in ("...", "…"):
            add(m.start(), m.start() + 3, DIST, "extras")
        elif R_EXTRA_LIST.match(inner):
            unknown = [e.strip() for e in inner.split(",") if e.strip() not in ctx.extras]
            if unknown:
                refused.append((base + m.start(), f"`neo[{inner}]` names no extra of NEO's: {unknown}"))
            else:
                add(m.start(), m.start() + 3, DIST, "extras")
    if not light:
        for m in R_FROM.finditer(text):
            s = m.group().rfind("neo")
            add(m.start() + s, m.start() + s + 3, NEW, "import (text)")
        for m in R_IMPORT_BARE.finditer(text):
            if free(m.start(), m.end()):
                refused.append((base + m.start(), "a bare `import neo` in text: rewrite it by hand"))
        for m in R_MODULE.finditer(text):
            if m.group("sub") in ctx.subs and not R_LOGGER_ARG.search(text, max(0, m.start() - 40), m.start()):
                add(m.start(), m.start() + 3, NEW, "module path")
        for m in R_TOOL_LINE.finditer(text):
            for n in R_BARE_NEO.finditer(m.group()):
                if free(m.start() + n.start(), m.start() + n.end()):
                    refused.append((base + m.start() + n.start(), f"`neo` as {m.group().split()[0]}'s argument: {FOLDER}"))
        for rx, what in ((R_COV, "--cov=neo"), (R_TOML_LIST, 'a list with "neo"')):
            for m in rx.finditer(text):
                if free(m.start(), m.end()) and not any(base + m.start() <= o < base + m.end() for o, _ in refused):
                    refused.append((base + m.start(), f"{what}: {FOLDER}"))
    for rx, what in ((R_JOIN_NEO, 'join(…, "neo")'), (R_PATH_NEO, 'Path("neo")')):
        for m in rx.finditer(text):
            refused.append((base + m.start(), f"{what} (the package folder?): {FOLDER}"))
    for m in R_PATH.finditer(text):
        add(m.start(), m.end(), NEW_DIR + "/", "path")
    if not light:
        for m in R_SUBDIR.finditer(text):
            if "://" not in text[text.rfind("\n", 0, m.start()) + 1 : m.start()]:
                refused.append(
                    (
                        base + m.start(),
                        "a path through a folder named neo (site-packages/neo/?): rewrite by hand or mark it",
                    )
                )
    return edits, refused


def apply(text: str, edits: list[Edit], keep_lines: set[int]) -> tuple[str, Counter]:
    counts: Counter = Counter()
    out, pos = [], 0
    for e in sorted(edits, key=lambda e: e.start):
        if e.start < pos or (keep_lines and line_of(text, e.start) in keep_lines):
            continue
        out.append(text[pos : e.start])
        out.append(e.new)
        pos = e.end
        counts[e.rule] += 1
    out.append(text[pos:])
    return "".join(out), counts


def line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def keep_lines(text: str) -> set[int]:
    return {i for i, line in enumerate(text.split("\n"), 1) if KEEP in line}


def skipped(text: str) -> bool:
    head = "\n".join(text.split("\n", 12)[:12])
    return SKIP_FILE in head or SHIM in head


def rewrite_text(text: str, ctx: Ctx, *, light: bool = False) -> Outcome:
    if skipped(text):
        return Outcome(text)
    edits, refused = text_edits(text, ctx, light=light)
    keep = keep_lines(text)
    new, counts = apply(text, edits, keep)
    return Outcome(
        new, counts, sorted((line_of(text, o), why) for o, why in refused if line_of(text, o) not in keep)
    )


# ---- Python --------------------------------------------------------------------------------------

_SKIP_TOKENS = (tokenize.NL, tokenize.COMMENT, tokenize.ENCODING)
_FSTART = getattr(tokenize, "FSTRING_START", -1)
_FEND = getattr(tokenize, "FSTRING_END", -1)


def _literal(tok: tokenize.TokenInfo) -> object:
    try:
        return ast.literal_eval(tok.string)
    except Exception:  # noqa: BLE001 — not a plain literal (a byte string with escapes, …)
        return None


def _opener(sig: list[tokenize.TokenInfo], k: int) -> int | None:
    """The bracket sig[k] is inside (its index), or None at the top of a statement."""
    depth = 0
    for j in range(k - 1, -1, -1):
        t = sig[j]
        if t.type in (tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
            return None
        if t.type == tokenize.OP and t.string in ")]}":
            depth += 1
        elif t.type == tokenize.OP and t.string in "([{":
            if depth == 0:
                return j
            depth -= 1
    return None

_EXPR_START = {"return", "in", "for", "yield", "assert", "if", "else", "and", "or", "not", "lambda", "await"}


def _operand_names(sig: list[tokenize.TokenInfo], k: int) -> set[str]:
    """The names in the expression left of sig[k] (`Path(__file__).parents[1] / "neo"`), back to
    the start of its statement, bracket or argument."""
    names: set[str] = set()
    depth = 0
    for j in range(k - 1, -1, -1):
        t = sig[j]
        if t.type in (tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
            break
        if t.type == tokenize.OP and t.string in ")]}":
            depth += 1
        elif t.type == tokenize.OP and t.string in "([{":
            if depth == 0:
                break
            depth -= 1
        elif depth == 0 and (
            (t.type == tokenize.OP and t.string not in ("/", ".", "//"))
            or (t.type == tokenize.NAME and t.string in _EXPR_START)
        ):
            break
        if t.type == tokenize.NAME:
            names.add(t.string)
    return names


def _names_the_folder(sig: list[tokenize.TokenInfo], k: int) -> str | None:
    """Why the string "neo" at sig[k] looks like the package's folder (None if it doesn't): the
    checkout's `ROOT / "neo"`, `os.path.join(ROOT, "neo")`, `Path("neo")`, or a tool's argument
    (`["mypy", "neo"]`). Never `tmp_path / "neo"` or a name like `controller = "neo"`."""
    p1 = sig[k - 1] if k else None
    n1 = sig[k + 1] if k + 1 < len(sig) else None

    def repoish(names: Iterable[str]) -> bool:
        return bool({n.lower() for n in names} & REPO_NAMES)

    if (p1 is not None and p1.string == "/") or (n1 is not None and n1.string == "/"):
        if p1 is not None and p1.string == "/" and repoish(_operand_names(sig, k - 1)):
            return '`… / "neo"` (the package folder?)'
        return None
    ob = _opener(sig, k)
    if ob is None:
        return None
    fn = sig[ob - 1] if ob and sig[ob - 1].type == tokenize.NAME else None
    if sig[ob].string == "(" and fn is not None:
        if fn.string in ("Path", "PurePath", "PosixPath", "PurePosixPath") and ob == k - 1:
            return 'Path("neo") (the package folder?)'
        if fn.string == "join" and ob >= 3 and sig[ob - 2].string == "." and sig[ob - 3].string in ("path", "os", "posixpath"):
            if repoish(t.string for t in sig[ob + 1 : k] if t.type == tokenize.NAME):
                return 'os.path.join(…, "neo") (the package folder?)'
            return None
    args, depth = [], 0
    for t in sig[ob + 1 : k]:
        if t.type == tokenize.OP and t.string in "([{":
            depth += 1
        elif t.type == tokenize.OP and t.string in ")]}":
            depth -= 1
        elif depth == 0 and t.type == tokenize.STRING:
            v = _literal(t)
            if isinstance(v, str):
                args.append(v.rsplit("/", 1)[-1])
    tool = next((a for a in args if a in TOOLS), None)
    return f"`neo` as {tool}'s argument" if tool else None


def rewrite_python(src: str, ctx: Ctx, *, in_core: bool = False) -> Outcome:
    if skipped(src):
        return Outcome(src)
    starts = [0] + [m.end() for m in re.finditer("\n", src)]

    def at(pos: tuple[int, int]) -> int:
        return starts[pos[0] - 1] + pos[1]

    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
        tree0 = ast.parse(src)
    except (tokenize.TokenError, SyntaxError, ValueError) as e:
        return Outcome(src, refused=[(getattr(e, "lineno", 1) or 1, f"not valid Python ({e}): left as is")])

    edits: list[Edit] = []
    refused: list[tuple[int, str]] = []
    sig = [t for t in toks if t.type not in _SKIP_TOKENS]
    import_ids: set[int] = set()  # indexes in sig of tokens inside import statements
    bare_import = False
    binds_neo: list[int] = []  # lines where an import binds the name neo to something else

    def dotted(j: int) -> tuple[list[int], int]:
        names = [j]
        j += 1
        while j + 1 < len(sig) and sig[j].string == "." and sig[j + 1].type == tokenize.NAME:
            names.append(j + 1)
            j += 2
        return names, j

    for i, t in enumerate(sig):
        if t.type != tokenize.NAME or t.string not in ("import", "from"):
            continue
        prev = sig[i - 1] if i else None
        if prev is not None and not (
            prev.type in (tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)
            or (prev.type == tokenize.OP and prev.string in (";", ":"))
        ):
            continue  # `yield from`, `raise … from`: not an import statement
        j = i + 1
        end = j
        while (
            end < len(sig)
            and sig[end].type not in (tokenize.NEWLINE, tokenize.ENDMARKER)
            and not (sig[end].type == tokenize.OP and sig[end].string == ";")
        ):
            end += 1
        import_ids.update(range(i, end))
        if t.string == "import":
            while j < end and sig[j].type == tokenize.NAME:
                names, j = dotted(j)
                first = sig[names[0]]
                alias = None
                if j < end and sig[j].string == "as":
                    alias = sig[j + 1].string
                    j += 2
                if first.string == OLD:
                    edits.append(Edit(at(first.start), at(first.end), NEW, "import"))
                    bare_import |= alias is None
                if alias == OLD:
                    binds_neo.append(first.start[0])
                if j < end and sig[j].string == ",":
                    j += 1
                    continue
                break
        else:  # from
            if j < end and sig[j].type == tokenize.OP and sig[j].string in (".", "..."):
                continue  # relative: left alone
            if j < end and sig[j].type == tokenize.NAME:
                names, k = dotted(j)
                first = sig[names[0]]
                if first.string == OLD:
                    edits.append(Edit(at(first.start), at(first.end), NEW, "import"))
                # names it binds
                k += 1  # "import"
                while k < end:
                    if sig[k].type == tokenize.NAME and sig[k].string not in ("as",):
                        bound = (
                            sig[k + 2].string if k + 2 < end and sig[k + 1].string == "as" else sig[k].string
                        )
                        if bound == OLD:
                            binds_neo.append(sig[k].start[0])
                        k += 3 if k + 1 < end and sig[k + 1].string == "as" else 1
                    else:
                        k += 1

    # f-strings: one stretch of text each (3.12 splits them into tokens)
    fspans: list[tuple[int, int]] = []
    depth, fstart = 0, 0
    for t in toks:
        if t.type == _FSTART:
            if depth == 0:
                fstart = at(t.start)
            depth += 1
        elif t.type == _FEND:
            depth -= 1
            if depth == 0:
                fspans.append((fstart, at(t.end)))

    def in_fstring(off: int) -> bool:
        return any(a <= off < b for a, b in fspans)

    if bare_import:
        if binds_neo:
            refused.append(
                (
                    binds_neo[0],
                    "`import neo…` here, and `neo` is also bound to something else: rewrite by hand",
                )
            )
        for k, t in enumerate(sig):
            if t.type != tokenize.NAME or t.string != OLD or k in import_ids or in_fstring(at(t.start)):
                continue
            if k and sig[k - 1].type == tokenize.OP and sig[k - 1].string == ".":
                continue  # an attribute: obj.neo
            if k + 1 < len(sig) and sig[k + 1].type == tokenize.OP and sig[k + 1].string == ".":
                edits.append(Edit(at(t.start), at(t.end), NEW, "name"))
            else:
                refused.append(
                    (t.start[0], "`import neo…` here, and `neo` is also a plain name: rewrite by hand")
                )

    # strings in a context that names a module
    as_is: set[int] = set()  # strings refused: no text rule either
    for k, t in enumerate(sig):
        if t.type != tokenize.STRING:
            continue
        val = _literal(t)
        p1 = sig[k - 1] if k >= 1 else None
        p2 = sig[k - 2] if k >= 2 else None
        n1 = sig[k + 1] if k + 1 < len(sig) else None
        n2 = sig[k + 2] if k + 2 < len(sig) else None
        if p1 is not None and p2 is not None and (
            (p1.string == "(" and p2.string == "getLogger")
            or (p1.string == "=" and p2.string == "logger" and k >= 3 and sig[k - 3].string in ("(", ","))
        ):
            as_is.add(at(t.start))  # a logger name (getLogger("neo.x"), logger="neo.x"): those didn't move
            continue
        if val in ("neo.", "neo/"):
            refused.append(
                (t.start[0], f"{t.string} as a prefix: is it the package? rewrite by hand or mark it")
            )
            as_is.add(at(t.start))
            continue
        if val != OLD:
            continue
        near = [sig[j] for j in (k - 2, k + 2) if 0 <= j < len(sig) and sig[j].type == tokenize.STRING]
        if any(_literal(s) == DIST for s in near):
            refused.append((t.start[0], '"neo" next to "neo2": the old package? rewrite by hand or mark it'))
            continue
        q = t.string[-1]
        if p1 is None or p2 is None:
            continue
        names_module = (p1.string == "(" and p2.string in MODULE_FUNCS) or (
            p1.string in ("[", ",") and p2.string == "modules"
        )
        if names_module and p2.type == tokenize.NAME:  # import_module("neo"), sys.modules["neo"]
            edits.append(Edit(at(t.start), at(t.end), f"{q}{NEW}{q}", "module path"))
            continue
        if p1.string == "," and p2.type == tokenize.STRING and _literal(p2) == "-m":
            edits.append(Edit(at(t.start), at(t.end), f"{q}{DIST}{q}", "-m neo"))
            continue
        if p1.string == "/" and n1 is not None and n1.string == "/" and n2 is not None and n2.type == tokenize.STRING:
            nxt = _literal(n2)
            if isinstance(nxt, str) and (
                nxt in ctx.subs or nxt.removesuffix(".py") in ctx.subs | {"__init__"}
            ):
                edits.append(Edit(at(t.start), at(t.end), f"{q}neo2{q} / {q}core{q}", "path"))
                continue
        why = _names_the_folder(sig, k)
        if why:
            refused.append((t.start[0], f"{why}: {FOLDER}"))

    # strings and comments: the text rules
    for t in toks:
        if (
            t.type in (tokenize.STRING, tokenize.COMMENT)
            and not in_fstring(at(t.start))
            and at(t.start) not in as_is
        ):
            a = at(t.start)
            e, r = text_edits(src[a : at(t.end)], ctx, a)
            edits += e
            refused += [(line_of(src, o), why) for o, why in r]
    for a, b in fspans:
        e, r = text_edits(src[a:b], ctx, a)
        edits += e
        refused += [(line_of(src, o), why) for o, why in r]

    if in_core:
        e, r = _core_loggers(src, sig, import_ids, tree0, starts)
        edits += e
        refused += r

    # A path worked out from a module's own file, two folders up: the package is a level deeper now
    for n, line in enumerate(src.split("\n"), 1):
        if "__file__" in line and R_UP_TWO.search(line) and (in_core or R_MODULE_FILE.search(line)):
            refused.append(
                (n, "two folders up from a module's file: neo2/core is one level deeper than neo/ was")
            )

    keep = keep_lines(src)
    new, counts = apply(src, edits, keep)
    refused = [(n, why) for n, why in refused if n not in keep]
    try:
        tree = ast.parse(new)
    except SyntaxError as e:
        return Outcome(src, refused=[(e.lineno or 1, f"the rewrite wouldn't parse ({e.msg}): left as is")])
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        if any(n == OLD or n.startswith(OLD + ".") for n in names) and node.lineno not in keep:
            refused.append((node.lineno, "an import of neo is left: rewrite by hand"))
    return Outcome(new, counts, sorted(set(refused)))


def _core_loggers(
    src: str, sig: list[tokenize.TokenInfo], import_ids: set[int], tree: ast.Module, starts: list[int]
) -> tuple[list[Edit], list[tuple[int, str]]]:
    """In NEO's own package, `logging.getLogger(__name__)` → `logger(__name__)` from neo2.core: its
    modules log under the `neo` logger as neo.<module>, which NEO.app (log_to_stdout, core.log) and
    hosts configure, and __name__ says neo2.core.<module> now. Adds the import (into an existing
    one-line `from neo2.core import …` when there is one) and drops a plain `import logging` nothing
    else uses."""
    edits: list[Edit] = []
    refused: list[tuple[int, str]] = []

    def off(line: int, col: int) -> int:  # ast columns count UTF-8 bytes
        text = src[starts[line - 1] : starts[line] if line < len(starts) else len(src)]
        return starts[line - 1] + len(text.encode()[:col].decode(errors="ignore"))

    calls: list[int] = []  # sig index of `logging` in logging.getLogger(__name__)
    for k, t in enumerate(sig):
        if not (t.type == tokenize.NAME and t.string == "getLogger" and k + 3 < len(sig)):
            continue
        if [x.string for x in sig[k + 1 : k + 4]] != ["(", "__name__", ")"]:
            continue
        if k >= 2 and sig[k - 1].string == "." and sig[k - 2].string == "logging" and sig[k - 3].string != ".":
            calls.append(k - 2)
        else:
            refused.append((t.start[0], "getLogger(__name__) in NEO's package logs as neo2.core.…, outside the "
                                        "`neo` logger: use `logger(__name__)` from neo2.core"))  # fmt: skip
    if not calls:
        return edits, refused

    ours, other = False, []  # `from neo2.core import logger` at the top; anything else named logger
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                if (a.asname or a.name.split(".")[0]) != "logger":
                    continue
                if (isinstance(node, ast.ImportFrom) and node.level == 0 and node.module in (OLD, NEW)
                        and a.asname is None and node in tree.body):  # fmt: skip
                    ours = True
                else:
                    other.append(node.lineno)
        elif "logger" in _binds(node):
            other.append(getattr(node, "lineno", 1))
    first = min(sig[k].start[0] for k in calls)
    if other:
        return edits, refused + [(sig[k].start[0], f"`logger` is another name here (line {other[0]}): use "
                                                  "neo2.core's logger(__name__) by hand") for k in calls]  # fmt: skip

    imp = None
    if not ours:
        imp = _logger_import(src, tree, starts, off, first)
        if imp is None:
            return edits, refused + [(first, "no imports before this: add `from neo2.core import logger` and "
                                             "use logger(__name__) by hand")]  # fmt: skip
    for k in calls:
        edits.append(Edit(_at(starts, sig[k].start), _at(starts, sig[k + 2].end), "logger", "logger"))

    # `import logging`, if that was all it was for (and the blank line after it, if it was a section alone)
    rest = [
        k for k, t in enumerate(sig)
        if t.type == tokenize.NAME and t.string == "logging" and k not in import_ids and k not in calls
        and not (k and sig[k - 1].string == ".")
    ]  # fmt: skip
    in_text = any(t.type == tokenize.STRING and re.search(r"\blogging\.", t.string) for t in sig)
    lines = src.split("\n")
    for node in tree.body if not rest and not in_text else []:
        if not (isinstance(node, ast.Import) and [(a.name, a.asname) for a in node.names] == [("logging", None)]):
            continue
        n = node.lineno
        if node.end_lineno != n or not re.fullmatch(r"import logging[ \t]*(#.*)?", lines[n - 1]):
            continue
        alone = (n == 1 or not lines[n - 2].strip()) and n < len(lines) - 1 and not lines[n].strip()
        end = starts[n + 1] if alone else starts[n] if n < len(starts) else len(src)
        if imp is not None and starts[n - 1] <= imp.start < end:  # the new import goes where this one was
            new = imp.new.strip("\n") + "\n" + ("\n" if alone else "")
            edits.append(Edit(starts[n - 1], end, new, "logger import"))
            imp = None
        else:
            edits.append(Edit(starts[n - 1], end, "", "logger import"))
    if imp is not None:
        edits.append(imp)
    return edits, refused


def _binds(node: ast.AST) -> set[str]:
    """The names a node (other than an import) binds: a def or class, an assignment, a parameter."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
        return {node.id}
    if isinstance(node, ast.arg):
        return {node.arg}
    if isinstance(node, (ast.Global, ast.Nonlocal)):
        return set(node.names)
    if isinstance(node, ast.ExceptHandler) and node.name:
        return {node.name}
    return set()


def _at(starts: list[int], pos: tuple[int, int]) -> int:
    return starts[pos[0] - 1] + pos[1]


def _logger_import(
    src: str, tree: ast.Module, starts: list[int], off: Callable[[int, int], int], first: int
) -> Edit | None:
    """Where `from neo2.core import logger` goes: into a one-line `from neo2.core import …` above the
    first use, else a line of its own where isort would put it (among NEO's own imports)."""

    def new_name(m: str) -> str:
        return NEW + m[len(OLD) :] if m == OLD or m.startswith(OLD + ".") else m

    def ours(m: str) -> bool:
        return m == DIST or m.startswith(DIST + ".")

    def line_start(n: int) -> tuple[int, str]:
        """The offset of line n (1-based), and a newline to put first when the file's last line has none."""
        at = starts[n - 1] if n - 1 < len(starts) else len(src)
        return at, "\n" if at == len(src) and src and not src.endswith("\n") else ""

    header = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)) and n.end_lineno < first]
    froms = [n for n in header if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module and ours(new_name(n.module))]
    core = [n for n in froms if new_name(n.module) == NEW and all(a.name != "*" for a in n.names)]
    if core and core[0].lineno == core[0].end_lineno:
        names = core[0].names

        def rank(name: str) -> tuple[int, str]:  # isort's order-by-type: CONSTANTS, Classes, the rest
            return (0 if name.isupper() and len(name) > 1 else 1 if name[:1].isupper() else 2, name.lower())

        after = next((a for a in names if rank(a.name) > rank("logger")), None)
        if after is not None:
            at = off(after.lineno, after.col_offset)
            return Edit(at, at, "logger, ", "logger import")
        at = off(names[-1].end_lineno, names[-1].end_col_offset)
        return Edit(at, at, ", logger", "logger import")
    line = "from neo2.core import logger\n"
    if core:  # a multi-line import: a line of its own just above (ruff --fix joins them)
        at, _ = line_start(core[0].lineno)
        return Edit(at, at, line, "logger import")
    later = [n for n in froms if tuple(new_name(n.module).split(".")) > tuple(NEW.split("."))]
    if later:
        at, _ = line_start(later[0].lineno)
        return Edit(at, at, line, "logger import")
    plain = [n for n in header if isinstance(n, ast.Import) and any(ours(new_name(a.name)) for a in n.names)]
    mine = froms + plain
    if mine:
        at, nl = line_start(max(n.end_lineno for n in mine) + 1)
        return Edit(at, at, nl + line, "logger import")
    relative = [n for n in header if isinstance(n, ast.ImportFrom) and n.level > 0]
    if relative:
        at, _ = line_start(relative[0].lineno)
        return Edit(at, at, line + "\n", "logger import")
    if header:
        at, nl = line_start(header[-1].end_lineno + 1)
        return Edit(at, at, nl + "\n" + line, "logger import")
    return None


# ---- pyproject -----------------------------------------------------------------------------------


def rewrite_pyproject(text: str, ctx: Ctx) -> Outcome:
    out = rewrite_text(text, ctx)
    new = out.text
    new, n = R_TOML_NAME.subn(lambda m: f'{m["pre"]}"{DIST}"{m["post"]}', new)
    out.counts["distribution"] += n
    new, n = R_TOML_INCLUDE.subn(lambda m: f'{m["pre"]}"neo2*"', new)
    out.counts["packages"] += n
    out.text = new
    out.counts = +out.counts
    try:
        import tomllib

        data = tomllib.loads(new)
    except Exception as e:  # noqa: BLE001
        out.refused.append((1, f"pyproject.toml doesn't parse after the rewrite: {e}"))
        return out
    proj = data.get("project", {})
    if proj.get("name") not in (DIST, None):
        out.refused.append((1, f"[project] name is {proj.get('name')!r}, not {DIST!r}"))
    if OLD in proj.get("scripts", {}):
        out.refused.append((1, "[project.scripts] still has a `neo` command: rename it by hand (neo2)"))
    for group in ("scripts", "gui-scripts"):
        for k, v in proj.get(group, {}).items():
            if str(v).split(":")[0].split(".")[0] == OLD:
                out.refused.append((1, f"[project.{group}] {k} = {v!r} still points into neo"))
    setuptools = data.get("tool", {}).get("setuptools", {})
    find = (
        setuptools.get("packages", {}).get("find", {}) if isinstance(setuptools.get("packages"), dict) else {}
    )
    import fnmatch

    for pat in find.get("include", []):
        if fnmatch.fnmatchcase(OLD, pat) or fnmatch.fnmatchcase(OLD + ".tools", pat):
            out.refused.append((1, f"packages.find include {pat!r} would ship the repo-only neo/ shim"))
    if OLD in setuptools.get("package-data", {}):
        out.refused.append((1, "[tool.setuptools.package-data] still names the neo package"))
    return out


# ---- the files -----------------------------------------------------------------------------------


def git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=check)


def tracked(root: Path, rel: str) -> bool:
    return git(root, "ls-files", "--error-unmatch", rel, check=False).returncode == 0


def is_shim(p: Path) -> bool:
    try:
        return SHIM in p.read_text(encoding="utf-8", errors="replace")[:2000]
    except OSError:
        return False


def _move(root: Path, src: str, dst: str) -> str | None:
    """`git mv` a tracked path (a plain rename otherwise); why it failed, or None."""
    if tracked(root, src):
        r = git(root, "mv", src, dst, check=False)
        if r.returncode:
            return (r.stderr.strip() or f"git mv exited {r.returncode}") + " — resolve it (git add?) and re-run"
    else:
        os.rename(root / src, root / dst)
    return None


def _not_just_caches(d: Path) -> list[str]:
    """What a folder holds besides what nothing needs: __pycache__ (bytecode) and .DS_Store."""
    left: list[str] = []
    if d.exists() and not d.is_dir():
        return [d.name]
    for dirpath, dirnames, filenames in os.walk(d):
        dirnames[:] = [x for x in dirnames if x != "__pycache__"]
        left += [Path(dirpath, f).relative_to(d).as_posix() for f in filenames
                 if f != ".DS_Store" and not f.endswith((".pyc", ".pyo"))]  # fmt: skip
    return sorted(left)


def relocate(root: Path, dry: bool) -> tuple[list[str], list[str]]:
    """Put NEO's package under neo2/core: (what moved, what was refused)."""
    moved: list[str] = []
    refused: list[str] = []
    old, new = root / OLD, root / NEW_DIR
    if not old.is_dir():
        return moved, refused
    init = old / "__init__.py"
    if not (new / "__init__.py").exists():
        if init.exists() and not is_shim(init):
            # A neo2/core/ without an __init__.py: usually the renamed tree's git-ignored __pycache__
            # folders, left by a branch switch. Moving onto it would put the package in neo2/core/neo/.
            left = _not_just_caches(new) if new.exists() else []
            if left:
                refused.append(f"{NEW_DIR}/ exists without __init__.py ({', '.join(left[:3])}"
                               f"{', …' if len(left) > 3 else ''}): move that away and re-run")  # fmt: skip
                return moved, refused
            moved.append(f"{OLD}/ → {NEW_DIR}/ (the whole package)")
            if not dry:
                if new.exists():
                    shutil.rmtree(new)  # only bytecode and .DS_Store: nothing git tracks or needs
                new.parent.mkdir(parents=True, exist_ok=True)
                why = _move(root, OLD, NEW_DIR)
                if why or not (new / "__init__.py").exists():
                    moved.pop()
                    refused.append(f"{OLD}/ didn't move to {NEW_DIR}/: {why or 'its __init__.py is not there'}")
        return moved, refused
    for name in ("__init__.py", "__main__.py"):
        f = old / name
        if f.exists() and not is_shim(f):
            refused.append(f"{OLD}/{name}: not the compatibility shim any more (a merge changed it?) — port the change "
                           f"to {NEW_DIR}/{name} by hand and put the shim back (git checkout the shim's version)")  # fmt: skip
    for dirpath, dirnames, filenames in os.walk(old):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in sorted(filenames):
            src = Path(dirpath) / fn
            rel = src.relative_to(old).as_posix()
            if rel in ("__init__.py", "__main__.py") or fn.endswith((".pyc", ".pyo")) or fn == ".DS_Store":
                continue
            dst = new / rel
            if not dst.exists():
                moved.append(f"{OLD}/{rel} → {NEW_DIR}/{rel}")
                if not dry:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    why = _move(root, f"{OLD}/{rel}", f"{NEW_DIR}/{rel}")
                    if why:
                        moved.pop()
                        refused.append(f"{OLD}/{rel}: {why}")
            elif filecmp.cmp(src, dst, shallow=False):
                moved.append(f"{OLD}/{rel} removed (the same as {NEW_DIR}/{rel})")
                if not dry:
                    if tracked(root, f"{OLD}/{rel}"):
                        git(root, "rm", "-q", "-f", f"{OLD}/{rel}")
                    else:
                        src.unlink()
            else:
                refused.append(f"{OLD}/{rel} and {NEW_DIR}/{rel} both exist and differ: merge them by hand "
                               f"(a branch from before the move? {MERGE_ORDER})")  # fmt: skip
    return moved, refused


def package_ctx(root: Path) -> Ctx:
    pkg = root / NEW_DIR if (root / NEW_DIR / "__init__.py").exists() else root / OLD
    subs = {"__main__"}
    if pkg.is_dir():
        for p in pkg.iterdir():
            if p.suffix == ".py" and p.stem != "__init__":
                subs.add(p.stem)
            elif p.is_dir() and (p / "__init__.py").exists():
                subs.add(p.name)
        init = pkg / "__init__.py"
        if init.exists() and not is_shim(init):
            try:
                for node in ast.parse(init.read_text()).body:
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
                        subs.add(node.name)
            except SyntaxError:
                pass
    extras: set[str] = set()
    try:
        import tomllib

        data = tomllib.loads((root / "pyproject.toml").read_text())
        extras = set(data.get("project", {}).get("optional-dependencies", {}))
    except Exception:  # noqa: BLE001 — no pyproject: every neo[…] is then refused, never guessed
        pass
    return Ctx(frozenset(subs), frozenset(extras))


def wanted(rel: str) -> bool:
    p = Path(rel)
    if p.name in SKIP_NAMES or any(rel.startswith(s) for s in SKIP_PREFIXES) or "__pycache__" in p.parts:
        return False
    if rel in (f"{OLD}/__init__.py", f"{OLD}/__main__.py"):
        return False  # the shim: it names the old package on purpose
    return p.suffix in TEXT_SUFFIXES


def list_files(root: Path, only: list[str]) -> list[str]:
    try:
        out = git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard").stdout
        rels = sorted({r for r in out.split("\0") if r and (root / r).is_file()})
    except (OSError, subprocess.CalledProcessError):  # not a git checkout: every file under root
        rels = []
        for d, dirs, files in os.walk(root):
            dirs[:] = [x for x in dirs if x not in (".git", "node_modules", "target", "__pycache__", ".venv")]
            rels += [Path(d, f).relative_to(root).as_posix() for f in files]
        rels.sort()
    if only:
        sel = [Path(o).resolve() for o in only]
        rels = [
            r
            for r in rels
            if any((root / r).resolve() == s or s in (root / r).resolve().parents for s in sel)
        ]
    return [r for r in rels if wanted(r)]


def rewrite_file(rel: str, text: str, ctx: Ctx) -> Outcome:
    suffix = Path(rel).suffix
    if suffix in (".py", ".pyi"):
        # neo/…: a merged branch's module, on its way to neo2/core (--check doesn't move it)
        return rewrite_python(text, ctx, in_core=rel.startswith((NEW_DIR + "/", OLD + "/")))
    if rel == "pyproject.toml":
        return rewrite_pyproject(text, ctx)
    return rewrite_text(text, ctx, light=suffix in LIGHT)


def main(argv: list[str] | None = None) -> int:
    if sys.version_info < (3, 12):  # noqa: UP036 — NEO is 3.12 syntax (PEP 695): an older ast can't read it
        print("codemod_neo2: run it with Python 3.12 (the repo's .venv/bin/python)", file=sys.stderr)
        return 2
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("paths", nargs="*", help="only these files or folders (default: the whole repo)")
    ap.add_argument("--root", default=None, help="the repo (default: the one this script is in)")
    ap.add_argument("--check", action="store_true", help="change nothing; exit 1 if anything would change")
    ap.add_argument("-v", "--verbose", action="store_true", help="list every file changed")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve() if a.root else Path(__file__).resolve().parents[1]

    moved, refused_moves = relocate(root, dry=a.check)
    ctx = package_ctx(root)
    totals: Counter = Counter()
    files_by_rule: dict[str, set[str]] = {}
    changed: list[tuple[str, Counter]] = []
    refused: list[str] = list(refused_moves)
    for rel in list_files(root, a.paths):
        path = root / rel
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        out = rewrite_file(rel, text, ctx)
        refused += [f"{rel}:{line}: {why}" for line, why in out.refused]
        if out.text != text:
            changed.append((rel, out.counts))
            totals.update(out.counts)
            for rule in out.counts:
                files_by_rule.setdefault(rule, set()).add(rel)
            if not a.check:
                with open(path, "w", encoding="utf-8", newline="") as fh:
                    fh.write(out.text)

    verb = "would rewrite" if a.check else "rewrote"
    print(f"codemod_neo2: {verb} {sum(totals.values())} references in {len(changed)} files"
          + (f"; {'would move' if a.check else 'moved'} {len(moved)}" if moved else ""))  # fmt: skip
    for m in moved:
        print(f"  move  {m}")
    for rule, n in totals.most_common():
        print(f"  {rule:<14} {n:>5}  in {len(files_by_rule[rule])} files")
    if a.verbose:
        for rel, counts in changed:
            print(f"    {rel}: " + ", ".join(f"{r} {n}" for r, n in counts.most_common()))
    if (changed or moved) and not a.check:
        # `from neo import x` above `from neo2 import y` sorts after it once rewritten (ruff's I001)
        print(f"then: {FIX_ORDER}, and review the diff")
    if refused:
        print(f"refused {len(refused)} (left as they are — fix by hand, or mark a line `{KEEP}`):")
        for r in refused:
            print(f"  {r}")
        return 1
    if a.check and (changed or moved):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
