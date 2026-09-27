"""``python -m neo2.core.selftest controls`` — NEO's controls self-test from a developer's own Terminal.

The usual way is to ask NEO: say "Neo, check your computer controls" (or type it in the overlay).
Inside NEO only those words start it, never a model: so this command refuses to run inside NEO.app's
process tree or from NEO's own shell tool (which runs a model's commands, behind one generic "run a
command" card — and whose time limit would kill the test mid-drag, leaving nothing put back), and
without an interactive terminal (a script or a tool, not a person at the keyboard).

From a Terminal window it clicks and types with that terminal's permissions (Accessibility; Screen
Recording for the screenshot checks), not NEO.app's: it tests NEO's tools, the way NEO.app would
use them.

Exit status: 0 every check passed, 1 something failed or the run stopped early, 2 it didn't run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys

_APP_EXEC = "/NEO.app/Contents/MacOS/"
_HOW = 'Say "Neo, check your computer controls" (or type it in NEO\'s overlay) and NEO runs it itself.'
SHELL_MARK = "NEO_SHELL"  # set by NEO's shell tool on every command it runs (neo2.core.tools.computer.shell)


def process_table() -> dict[int, tuple[int, str]]:
    """pid → (parent pid, executable path) for every process (ps; nothing is changed)."""
    try:
        out = subprocess.run(["ps", "-A", "-ww", "-o", "pid=,ppid=,comm="], capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    table: dict[int, tuple[int, str]] = {}
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            table[int(parts[0])] = (int(parts[1]), parts[2])
    return table


def neo_app_ancestor(pid: int | None = None, table: dict[int, tuple[int, str]] | None = None) -> str:
    """The NEO.app executable this process descends from ("" when it doesn't)."""
    table = process_table() if table is None else table
    pid = pid or os.getpid()
    seen: set[int] = set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        entry = table.get(pid)
        if entry is None:
            return ""
        ppid, path = entry
        if _APP_EXEC in path:
            return path
        pid = ppid
    return ""


def _interactive() -> bool:
    try:
        return bool(sys.stdin.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m neo2.core.selftest", description="NEO's self-tests.")
    p.add_argument("test", choices=["controls"], help="controls: NEO's mouse, keyboard and screen tools, in a test window")
    p.add_argument("--json", action="store_true", help="print the report as JSON")
    args = p.parse_args(argv)
    if os.environ.get(SHELL_MARK) or neo_app_ancestor():
        print(
            "Inside NEO the controls check starts only from your own request, never from a command NEO runs "
            f"(a model could have asked for it). {_HOW}",
            file=sys.stderr,
        )
        return 2
    if not _interactive():
        print(
            "The controls check drives the mouse and keyboard, so this command runs only in an interactive "
            f"terminal, with you at it. {_HOW}",
            file=sys.stderr,
        )
        return 2
    from neo2.core.selftest import controls
    from neo2.core.tools import computer  # noqa: F401 - registers NEO's own tools, which the test drives

    if computer._needs_ax() is not None:
        print("Accessibility is off for this terminal: turn it on in System Settings › Privacy & Security.", file=sys.stderr)
        return 2
    report = asyncio.run(controls.run(notify=lambda msg: print(msg, file=sys.stderr)))
    saved = controls.save(report)
    if args.json:
        print(json.dumps(report.to_json(), ensure_ascii=False, indent=1))
    else:
        print(report.text(), end="")
        if saved is not None:
            print(f"(saved to {saved})")
    if report.refused:
        return 2
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
