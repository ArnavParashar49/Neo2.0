#!/usr/bin/env python3
"""Move references to NEO's old top-level package `neo` over to `neo2.core` (and the distribution
`neo` over to `neo2`). neo2-codemod: skip-file

    python scripts/codemod_neo2.py              # rewrite in place and print what changed
    python scripts/codemod_neo2.py --check      # change nothing: exit 1 if anything is left to do
    python scripts/codemod_neo2.py -v PATH…     # only these files or folders, listing each file

Safe to run again, on any branch, as often as needed: what it already rewrote doesn't match any
more. After merging a branch that still has the old paths, run it and review the diff.

1. Files under `neo/` that aren't the repo-only compatibility shim (a merged branch added or changed
   them there) move to `neo2/core/` (`git mv`). Before the rename, the whole package moves.
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
   Never touched: ~/.neo, NEO_* names, NEO in prose, com.arnav.neo, URLs, neo2.*, neon, neo-ui,
   neo_lib, the `neo` and `neo.audit` logger names, and `neo` as a variable (`neo.ask(…)`).

Anything it can't rewrite safely is refused and listed (exit 1): a bare `import neo` in prose, `-m neo`
outside a Python command line, `"neo."` used as a prefix, `neo[…]` naming unknown extras, `neo` both
imported and used as a plain name in one file, files that differ in neo/ and neo2/core/, and pyproject
entries only a person can rename. A line that must keep an old name carries `neo2-codemod: keep`;
a file that must be skipped says `neo2-codemod: skip-file` in its first lines.
"""

from __future__ import annotations

import argparse
import ast
import filecmp
import io
import os
import re
import subprocess
import sys
import tokenize
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

OLD = "neo"
NEW = "neo2.core"
NEW_DIR = "neo2/core"
DIST = "neo2"
KEEP = "neo2-codemod: keep"
SKIP_FILE = "neo2-codemod: skip-file"
SHIM = "neo2-codemod: shim"  # marks the compatibility shim's own files (neo/__init__.py, neo/__main__.py)

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

MODULE_FUNCS = {"import_module", "importorskip", "find_spec", "__import__", "reload", "run_module"}


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
            if m.group("sub") in ctx.subs:
                add(m.start(), m.start() + 3, NEW, "module path")
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


def rewrite_python(src: str, ctx: Ctx) -> Outcome:
    if skipped(src):
        return Outcome(src)
    starts = [0] + [m.end() for m in re.finditer("\n", src)]

    def at(pos: tuple[int, int]) -> int:
        return starts[pos[0] - 1] + pos[1]

    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
        ast.parse(src)
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
        p1 = sig[k - 1] if k >= 1 else None
        p2 = sig[k - 2] if k >= 2 else None
        n1 = sig[k + 1] if k + 1 < len(sig) else None
        n2 = sig[k + 2] if k + 2 < len(sig) else None
        q = t.string[-1]
        if p1 is None or p2 is None:
            continue
        names_module = (p1.string == "(" and p2.string in MODULE_FUNCS) or (
            p1.string in ("[", ",") and p2.string == "modules"
        )
        if names_module and p2.type == tokenize.NAME:  # import_module("neo"), sys.modules["neo"]
            edits.append(Edit(at(t.start), at(t.end), f"{q}{NEW}{q}", "module path"))
        elif p1.string == "," and p2.type == tokenize.STRING and _literal(p2) == "-m":
            edits.append(Edit(at(t.start), at(t.end), f"{q}{DIST}{q}", "-m neo"))
        elif (
            p1.string == "/"
            and n1 is not None
            and n1.string == "/"
            and n2 is not None
            and n2.type == tokenize.STRING
        ):
            nxt = _literal(n2)
            if isinstance(nxt, str) and (
                nxt in ctx.subs or nxt.removesuffix(".py") in ctx.subs | {"__init__"}
            ):
                edits.append(Edit(at(t.start), at(t.end), f"{q}neo2{q} / {q}core{q}", "path"))

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
            moved.append(f"{OLD}/ → {NEW_DIR}/ (the whole package)")
            if not dry:
                new.parent.mkdir(parents=True, exist_ok=True)
                if tracked(root, f"{OLD}/__init__.py"):
                    git(root, "mv", OLD, NEW_DIR)
                else:
                    os.rename(old, new)
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
                    if tracked(root, f"{OLD}/{rel}"):
                        git(root, "mv", f"{OLD}/{rel}", f"{NEW_DIR}/{rel}")
                    else:
                        os.rename(src, dst)
            elif filecmp.cmp(src, dst, shallow=False):
                moved.append(f"{OLD}/{rel} removed (the same as {NEW_DIR}/{rel})")
                if not dry:
                    if tracked(root, f"{OLD}/{rel}"):
                        git(root, "rm", "-q", "-f", f"{OLD}/{rel}")
                    else:
                        src.unlink()
            else:
                refused.append(f"{OLD}/{rel} and {NEW_DIR}/{rel} both exist and differ: merge them by hand")
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
        return rewrite_python(text, ctx)
    if rel == "pyproject.toml":
        return rewrite_pyproject(text, ctx)
    return rewrite_text(text, ctx, light=suffix in LIGHT)


def main(argv: list[str] | None = None) -> int:
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
