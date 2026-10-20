#!/usr/bin/env python3
"""NEO's version, in every file that carries it.

    scripts/version.py              print each file's version
    scripts/version.py check        every file has the same version (exit 1 and say which differ)
    scripts/version.py check 2.2.0  …and it is 2.2.0 (scripts/build_release.sh, before a release)
    scripts/version.py bump 2.2.0   write 2.2.0 into every file

The files: pyproject.toml, neo2/__init__.py, ui/src-tauri/tauri.conf.json, ui/src-tauri/Cargo.toml,
NEO's own entry in ui/src-tauri/Cargo.lock, ui/package.json and ui/package-lock.json. A version is
X.Y.Z (what PEP 440, Cargo, npm and macOS's CFBundleShortVersionString all read the same way).
Each file is edited in place by one pattern, so its formatting stays as it was. Standard library
only: CI runs it with any python3.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = re.compile(r"^\d+\.\d+\.\d+$")


@dataclass(frozen=True)
class Spot:
    """One place a version is written: a file and a pattern whose `v` group is the version."""

    path: str
    pattern: str
    count: int = 1  # how many times the pattern must match in the file

    def regex(self) -> re.Pattern[str]:
        return re.compile(self.pattern, re.MULTILINE | re.DOTALL)


SPOTS = (
    # [project]'s version: the first `version = "…"` line after the [project] header
    Spot("pyproject.toml", r'^\[project\]\n(?:(?!^\[).)*?^version = "(?P<v>[^"]*)"'),
    Spot("neo2/__init__.py", r'^__version__ = "(?P<v>[^"]*)"'),
    # the top-level "version" key (two spaces in: the file's own indentation)
    Spot("ui/src-tauri/tauri.conf.json", r'\A\{\n(?:(?!^  "version").)*?^  "version": "(?P<v>[^"]*)"'),
    Spot("ui/src-tauri/Cargo.toml", r'^\[package\]\n(?:(?!^\[).)*?^version = "(?P<v>[^"]*)"'),
    Spot("ui/src-tauri/Cargo.lock", r'^\[\[package\]\]\nname = "neo-ui"\nversion = "(?P<v>[^"]*)"'),
    Spot("ui/package.json", r'\A\{\n(?:(?!^  "version").)*?^  "version": "(?P<v>[^"]*)"'),
    # the lock's own version, and its root package's ("packages": {"": {…}})
    Spot("ui/package-lock.json", r'\A\{\n(?:(?!^  "version").)*?^  "version": "(?P<v>[^"]*)"'),
    Spot("ui/package-lock.json", r'^    "": \{\n(?:(?!^    \}).)*?^      "version": "(?P<v>[^"]*)"'),
)


def read(root: Path) -> list[tuple[Spot, str | None]]:
    """Each spot's version, or None when the file or the pattern isn't there."""
    out: list[tuple[Spot, str | None]] = []
    for spot in SPOTS:
        f = root / spot.path
        text = f.read_text(encoding="utf-8") if f.is_file() else ""
        found = [m.group("v") for m in spot.regex().finditer(text)]
        out.append((spot, found[0] if len(found) == spot.count else None))
    return out


def check(root: Path, want: str | None = None) -> list[str]:
    """What's wrong ([] when every file agrees, and equals `want` when given)."""
    found = read(root)
    problems = [f"{s.path}: no version found" for s, v in found if v is None]
    versions = {v for _, v in found if v is not None}
    if want is not None:
        problems += [f"{s.path}: {v}, not {want}" for s, v in found if v is not None and v != want]
    elif len(versions) > 1:
        problems += [f"{s.path}: {v}" for s, v in found if v is not None]
        problems.insert(0, f"the files disagree ({', '.join(sorted(versions))}):")
    return problems


def bump(root: Path, new: str) -> list[str]:
    """Write `new` into every spot; returns the files changed. Nothing is written if a spot is missing."""
    if not VERSION.match(new):
        raise ValueError(f"not an X.Y.Z version: {new!r}")
    missing = [s.path for s, v in read(root) if v is None]
    if missing:
        raise LookupError("no version found in " + ", ".join(missing))
    texts: dict[str, str] = {}
    for spot in SPOTS:
        text = texts.get(spot.path) or (root / spot.path).read_text(encoding="utf-8")
        m = spot.regex().search(text)
        assert m is not None
        texts[spot.path] = text[: m.start("v")] + new + text[m.end("v") :]
    changed = []
    for path, text in texts.items():
        f = root / path
        if f.read_text(encoding="utf-8") != text:
            f.write_text(text, encoding="utf-8")
            changed.append(path)
    return changed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)  # the tests' copy
    sub = ap.add_subparsers(dest="cmd")
    c = sub.add_parser("check", help="every file has the same version (and this one, if given)")
    c.add_argument("version", nargs="?")
    b = sub.add_parser("bump", help="write this version into every file")
    b.add_argument("version")
    args = ap.parse_args(argv)
    root = args.root.resolve()

    if args.cmd == "check":
        if args.version is not None and not VERSION.match(args.version):
            print(f"not an X.Y.Z version: {args.version!r}", file=sys.stderr)
            return 2
        problems = check(root, args.version)
        for p in problems:
            print(p, file=sys.stderr)
        if not problems:
            print(read(root)[0][1])
        return 1 if problems else 0
    if args.cmd == "bump":
        try:
            changed = bump(root, args.version)
        except (ValueError, LookupError) as e:
            print(e, file=sys.stderr)
            return 2
        print(f"{args.version}: " + (", ".join(changed) if changed else "already everywhere"))
        return 0
    for spot, v in read(root):
        print(f"{v or '?':10s} {spot.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
