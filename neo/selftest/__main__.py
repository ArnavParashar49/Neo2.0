"""``python -m neo.selftest controls`` — NEO's controls self-test from a developer's own Terminal.

    python -m neo.selftest controls [--bench [N]] [--ax-walk [APP]] [--save-shot] [--json]
    python -m neo.selftest agent                     # 3 real requests through the brain chain
    python -m neo.selftest ab [--runs 10] [--arms effort,batch]
    python -m neo.selftest grounding --grounder pkg.module:function [--shot PATH]

``--bench``: each primitive's check N times (default 5) for p50/p90 timings, within 90 s more.
``--ax-walk``: first a read-only timing walk of an app's Accessibility tree (the one in front, or
APP). ``--save-shot``: keep the test window's picture and targets (for ``grounding``). ``agent`` and
``ab`` use your model quota (neo/selftest/agent.py); ``ab`` is the measurement the thinking-level and
batching defaults change on.

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
SHELL_MARK = "NEO_SHELL"  # set by NEO's shell tool on every command it runs (neo.tools.computer.shell)


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
    p = argparse.ArgumentParser(prog="python -m neo.selftest", description="NEO's self-tests.")
    p.add_argument(
        "test", choices=["controls", "agent", "ab", "grounding"],
        help="controls: NEO's mouse, keyboard and screen tools, in a test window; agent / ab: real requests "
        "through the brain chain there (your model quota); grounding: score a local grounder on a saved picture",
    )  # fmt: skip
    p.add_argument("--json", action="store_true", help="print the report as JSON")
    p.add_argument("--bench", nargs="?", type=int, const=5, default=None, metavar="N", help="repeat the primitives N times (5)")
    p.add_argument("--ax-walk", nargs="?", const="front", default=None, metavar="APP", help="time an app's Accessibility reads first")
    p.add_argument("--save-shot", action="store_true", help="keep the test window's picture and targets")
    p.add_argument("--runs", type=int, default=10, help="ab: runs per arm (10)")
    p.add_argument("--arms", default="effort,batch", help="ab: effort, batch or both")
    p.add_argument("--grounder", default="", help="grounding: module:function")
    p.add_argument("--shot", default="", help="grounding: a saved window picture (default: the latest)")
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
    if args.test == "grounding":
        from pathlib import Path

        from neo.selftest import grounding

        if not args.grounder:
            print("grounding needs --grounder module:function", file=sys.stderr)
            return 2
        out = grounding.run(args.grounder, Path(args.shot) if args.shot else None)
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return 0 if "error" not in out else 1
    from neo.selftest import controls
    from neo.tools import computer  # noqa: F401 - registers NEO's own tools, which the test drives

    if computer._needs_ax() is not None:
        print("Accessibility is off for this terminal: turn it on in System Settings › Privacy & Security.", file=sys.stderr)
        return 2
    if args.test in ("agent", "ab"):
        from neo.selftest import agent

        arms = [a for k in args.arms.split(",") for a in agent.ARMS.get(k.strip(), ())] if args.test == "ab" else None
        runs = max(1, args.runs) if args.test == "ab" else len(agent.REQUESTS)
        result = asyncio.run(agent.run(arms=arms, runs=runs, notify=lambda msg: print(msg, file=sys.stderr)))
        saved = agent.save(result, args.test)
        print(json.dumps(result.to_json(), ensure_ascii=False, indent=1) if args.json else result.text(), end="")
        if saved is not None:
            print(f"(saved to {saved})")
        return 2 if result.report.refused else 0
    report = asyncio.run(
        controls.run(
            notify=lambda msg: print(msg, file=sys.stderr), reps=args.bench or 1, ax_walk=args.ax_walk, save_shot=args.save_shot,
        )
    )  # fmt: skip
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
