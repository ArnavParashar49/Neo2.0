"""Who may say yes for NEO's MCP server — chosen by whoever starts it (``--approve``), never by a
tool, an argument or a model.

- ``dialog`` (:class:`DialogApprover`, macOS): a native dialog owned by the server process. The
  summary and the exact detail reach it only as ``osascript`` arguments (``on run argv``), never
  inside the script's source, so nothing in them can become AppleScript. Buttons "Cancel" and
  "Allow once", Cancel is the default (Return declines), and it gives up 10 s before the card's
  lifetime ends. The server also puts "osascript" in the agent's ``protected_apps``, so NEO's own
  click tools can never press "Allow once".
- ``elicit`` (:func:`elicit_request`, :func:`elicit_decision`): the MCP client asks its user.
  Only an ``accept`` whose ``approve`` is exactly ``true`` approves; the server asks only clients
  on its ``--trust-client`` list. It trusts the client, which may answer elicitations itself.

Anything else — a decline, a cancel, a timeout, an error, a missing or odd answer — declines.
"""

from __future__ import annotations

import asyncio
import contextlib
import getpass
import logging
import os
import re
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from contextvars import ContextVar
from typing import Any

from neo2.sdk.approval import Approval, ApprovalRequest

__all__ = [
    "DIALOG_CHANNEL",
    "ELICIT_CHANNEL",
    "ELICIT_KEY",
    "ELICIT_SCHEMA",
    "DialogApprover",
    "approval_text",
    "current_client",
    "dialog_argv",
    "elicit_decision",
    "elicit_request",
]

log = logging.getLogger("neo2.mcp")

DIALOG_CHANNEL = "mcp-dialog"
ELICIT_CHANNEL = "mcp-elicit"
OSASCRIPT = "/usr/bin/osascript"  # by path: never whatever "osascript" PATH finds
DIALOG_TITLE = "NEO: allow this action?"
DIALOG_MAX_CHARS = 3000  # a dialog can't show more; a longer action is declined, never half-shown
GIVE_UP_EARLY_S = 10  # the dialog closes itself this long before the card would lapse

# The MCP client this request came from (its clientInfo.name, as it names itself), for the text
# a human is shown and for the audit line. Set by the server around each tool call.
current_client: ContextVar[str | None] = ContextVar("neo2_mcp_client", default=None)

# The dialog. Its text arrives as arguments only: each one starts with _ARG (so none can look like
# an osascript option), which unwrap() drops again.
_ARG = "|"
_SCRIPT = (
    "on run argv",
    "set theMessage to my unwrap(item 1 of argv)",
    "set theTitle to my unwrap(item 2 of argv)",
    "set theWait to (my unwrap(item 3 of argv)) as integer",
    "try",
    'set theReply to display dialog theMessage with title theTitle buttons {"Cancel", "Allow once"} '
    'default button "Cancel" cancel button "Cancel" with icon caution giving up after theWait',
    "on error number -128",
    'return "cancel"',
    "end try",
    'if gave up of theReply then return "timeout"',
    'if button returned of theReply is "Allow once" then return "allow"',
    'return "cancel"',
    "end run",
    "on unwrap(s)",
    "if (length of s) > 1 then return text 2 thru -1 of s",
    'return ""',
    "end unwrap",
)

ELICIT_KEY = "neo_approval"  # the input request's key in a 2026-07-28 InputRequiredResult
ELICIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "approve": {
            "type": "boolean",
            "title": "Allow this once",
            "description": "Let NEO do exactly what is shown above, once. Leave it off to refuse.",
            "default": False,
        }
    },
    "required": ["approve"],
}

_SOURCES = {
    "caller": "the task itself (another model wrote it)",
    "context": "the context passed with the task",
}
_BIDI = re.compile("[‎‏‪-‮⁦-⁩]")
_CONTROL = re.compile("[\x00-\x08\x0b-\x1f\x7f]")


def clean(text: str) -> str:
    """What a human is shown: control and text-direction characters made visible, so nothing in
    the action can hide or reorder what the rest says."""
    text = _BIDI.sub(lambda m: f"<U+{ord(m.group()):04X}>", str(text))
    return _CONTROL.sub(lambda m: f"<0x{ord(m.group()):02X}>", text)


def approval_text(req: ApprovalRequest, *, client: str | None = None) -> str:
    """The question for the human: the card's summary, exactly what will run, and what untrusted
    text NEO had read before asking."""
    detail = clean(req.detail).strip()
    lines = [
        f"NEO wants to: {clean(req.summary)}",
        "",
        "Exactly what will run:",
        detail or "(nothing more than the line above)",
    ]
    read = [_SOURCES.get(s, s) for s in req.after_reading]
    if read:
        lines += ["", "Before asking, it read text nobody vetted: " + "; ".join(clean(s) for s in read) + "."]
    who = f" (client: {clean(client)})" if client else ""
    lines += ["", f"Asked through NEO's MCP server{who}. Allow it once?"]
    return "\n".join(lines)


def dialog_argv(message: str, title: str, wait_s: int) -> list[str]:
    """osascript's command line: the fixed script, then the text as arguments only."""
    argv = [OSASCRIPT]
    for line in _SCRIPT:
        argv += ["-e", line]
    argv += [_ARG + message, _ARG + title, _ARG + str(int(wait_s))]
    return argv


def _dialog_env() -> dict[str, str]:
    env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}
    for k in ("HOME", "USER", "LOGNAME", "LANG", "TMPDIR"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


async def _spawn(argv: Sequence[str], timeout: float) -> str:
    """Run the dialog (no shell: argv as given) and return what it printed ("" on failure)."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        env=_dialog_env(),
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    finally:
        if proc.returncode is None:  # timed out or cancelled: the dialog goes too
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
    return out.decode("utf-8", "replace").strip() if proc.returncode == 0 else ""


class DialogApprover:
    """``--approve dialog``: asks the person at this Mac with a native dialog. Pass
    ``approver=DialogApprover().ask`` to an agent (the MCP server does)."""

    channel = DIALOG_CHANNEL

    def __init__(
        self,
        *,
        user: str | None = None,
        spawn: Callable[[Sequence[str], float], Awaitable[str]] | None = None,
    ) -> None:
        if user is None:
            try:
                user = getpass.getuser()
            except Exception:  # noqa: BLE001
                user = "local-user"
        self.user = user
        self._spawn = spawn  # tests: a fake osascript

    async def ask(self, request: ApprovalRequest) -> Approval | None:
        if sys.platform != "darwin":
            return None
        client = current_client.get()
        text = approval_text(request, client=client)
        if len(text) > DIALOG_MAX_CHARS:
            log.warning(
                "declined without asking: the action is too long to show in a dialog (%d characters)",
                len(text),
            )
            return None
        wait = int(request.expires_at - time.time() - GIVE_UP_EARLY_S)
        if wait < 1:
            return None
        spawn = self._spawn or _spawn
        try:
            said = await spawn(dialog_argv(text, DIALOG_TITLE, wait), wait + 5)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — a dialog that fails declines
            log.warning("the approval dialog failed (%s): declined", type(e).__name__)
            return None
        if str(said).strip() != "allow":
            return None
        return Approval(DIALOG_CHANNEL, self.user, note=f"client: {client}" if client else "")


def elicit_request(req: ApprovalRequest, *, client: str | None = None) -> Any:
    """The elicitation (form mode) that asks the client's user about this card."""
    import mcp_types as types

    return types.ElicitRequest(
        params=types.ElicitRequestFormParams(
            message=approval_text(req, client=client), requested_schema=ELICIT_SCHEMA
        )
    )


def elicit_decision(answer: Any, client: str | None) -> Approval | None:
    """Only ``accept`` with ``approve`` exactly ``true`` is a yes."""
    import mcp_types as types

    if not isinstance(answer, types.ElicitResult) or answer.action != "accept":
        return None
    content = answer.content if isinstance(answer.content, dict) else {}
    if content.get("approve") is not True:
        return None
    return Approval(ELICIT_CHANNEL, f"client:{client or 'unnamed'}", note="elicitation: trusts the client")
