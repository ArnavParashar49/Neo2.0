"""NEO as an MCP server over stdio, for Claude Desktop, Claude Code, Cursor, the OpenAI Agents SDK
(``MCPServerStdio``), langchain-mcp-adapters or any other MCP client:

    python -m neo2.mcp --tools web --expose task --approve deny

``--tools``    NEO's tools the agent gets: presets, globs and names, comma-separated
               ("web", "web,browser", "mac", "none"). Default: "web".
``--expose``   ``task``: one tool, ``neo_task(goal, context?, conversation?)``, that hands the whole
               task to NEO (its own prompt, safety rules and confirmation gate); ``tools``: NEO's
               tools one by one; ``both``.
``--approve``  Who says yes when NEO wants to do something that needs the user's OK — chosen by
               whoever starts the server, never by a tool or a model:
               ``deny`` (default): nothing gated runs; the model is told the user must do it.
               ``dialog`` (macOS): a native dialog of this process asks the user.
               ``elicit``: the MCP client asks its user (MCP elicitation), and only for client
               names given with ``--trust-client NAME``. This trusts the client: a client may
               answer elicitations by itself.
``--env``      Read keys (GEMINI_API_KEY, GROQ_API_KEY, ANTHROPIC_API_KEY…) and NEO_MCP_* settings
               from the environment (and ``--env-file`` files). Without it nothing is read from
               the environment, and ``neo_task`` has no model to think with.

No tool, argument or result can approve anything, and no action id ever reaches the client. stdio
only: no network listener. stdout carries nothing but the protocol: the server moves fd 1 to
stderr before it imports anything, so a stray print (NEO's, a library's or a child process's)
lands on stderr.

This module is standard library only; the server itself (``neo2.mcp.server``) needs the ``mcp``
package (``mcp>=2,<3``).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import logging
import os
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neo2.mcp.approvers import DialogApprover as DialogApprover
    from neo2.mcp.server import NeoMCP as NeoMCP
    from neo2.mcp.server import ServerOptions as ServerOptions
    from neo2.mcp.server import run_stdio as run_stdio

__all__ = ["DialogApprover", "NeoMCP", "ServerOptions", "claim_stdout", "main", "release_stdout", "run_stdio"]

_LAZY = {
    "NeoMCP": "neo2.mcp.server",
    "ServerOptions": "neo2.mcp.server",
    "run_stdio": "neo2.mcp.server",
    "DialogApprover": "neo2.mcp.approvers",
}

_saved_stdout: list[Any] = []  # the sys.stdout claim_stdout() replaced, for release_stdout()


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def claim_stdout() -> int:
    """Keep stdout for the protocol alone: returns a private duplicate of fd 1 (the wire, not
    inherited by child processes) and points fd 1 — and ``sys.stdout`` — at stderr, so anything
    else written to stdout, by Python code or by a child process, lands on stderr."""
    with contextlib.suppress(Exception):
        sys.stdout.flush()
    wire = os.dup(1)  # non-inheritable (PEP 446): a child never gets the wire
    try:
        os.dup2(2, 1)
    except OSError:  # no usable stderr: stray output goes nowhere rather than onto the wire
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.close(devnull)
    _saved_stdout.append(sys.stdout)
    sys.stdout = sys.stderr
    return wire


def release_stdout(wire: int) -> None:
    """Undo :func:`claim_stdout`: fd 1 is the wire again, and the duplicate is closed."""
    with contextlib.suppress(Exception):
        sys.stderr.flush()
    with contextlib.suppress(OSError):
        os.dup2(wire, 1)
    with contextlib.suppress(OSError):
        os.close(wire)
    if _saved_stdout:
        sys.stdout = _saved_stdout.pop()


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m neo2.mcp",
        description="NEO as an MCP server (stdio). Gated actions are denied unless --approve says who may allow them.",
    )
    p.add_argument(
        "--tools",
        default=None,
        help='NEO\'s tools: presets, globs and names, comma-separated (e.g. "web", "web,browser", "none"). Default: web',
    )
    p.add_argument(
        "--expose",
        choices=("task", "tools", "both"),
        default="task",
        help="what the client sees (default: task)",
    )
    p.add_argument(
        "--approve",
        choices=("deny", "dialog", "elicit"),
        default="deny",
        help="who may say yes to a gated action (default: deny — nobody)",
    )
    p.add_argument(
        "--trust-client",
        action="append",
        default=[],
        metavar="NAME",
        help="with --approve elicit: a client name (its clientInfo.name) whose elicitation answers count; repeatable",
    )
    p.add_argument("--env", action="store_true", help="read keys and NEO_MCP_* settings from the environment")
    p.add_argument(
        "--env-file",
        action="append",
        default=[],
        metavar="PATH",
        help="also read this .env file (implies --env); repeatable",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="NEO's own log lines on stderr")
    return p


def _tools_arg(raw: str | None) -> str | tuple[str, ...] | None:
    if raw is None:
        return None
    items = tuple(x.strip() for x in raw.split(",") if x.strip())
    if not items:
        return "none"
    return items[0] if len(items) == 1 else items


def _log_setup(verbose: bool) -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO if verbose else logging.WARNING,
        format="neo2.mcp %(levelname)s %(name)s: %(message)s",
    )
    # Every staged, approved, denied or expired action, redacted as NEO redacts its audit log: on
    # stderr, which MCP clients keep in their server logs.
    logging.getLogger("neo.audit").setLevel(logging.INFO)
    if verbose:
        logging.getLogger("neo").setLevel(logging.INFO)


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m neo2.mcp`` (and the ``neo2-mcp`` console script)."""
    wire = claim_stdout()  # first of all: nothing imported below may write onto the wire
    try:
        args = _parser().parse_args(argv)
        _log_setup(args.verbose)
        try:
            import asyncio

            from neo2.mcp.server import ServerOptions, run_stdio
        except ImportError as e:
            print(
                f"neo2.mcp: the MCP server needs the mcp package (pip install 'mcp>=2,<3'): {e}",
                file=sys.stderr,
            )
            return 2
        opts = ServerOptions(
            tools=_tools_arg(args.tools),
            expose=args.expose,
            approve=args.approve,
            trust_clients=tuple(args.trust_client),
            env=bool(args.env or args.env_file),
            env_files=tuple(args.env_file),
        )
        try:
            asyncio.run(run_stdio(opts, wire_fd=wire))
        except KeyboardInterrupt:
            return 130
        except Exception as e:  # noqa: BLE001 — a configuration error: one line, not a traceback
            if args.verbose:
                logging.getLogger("neo2.mcp").exception("the server stopped")
            print(f"neo2.mcp: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
        return 0
    finally:
        release_stdout(wire)
