"""What kind of failure a tool's answer is, and one agent run's memory of the actions that failed.

The loop's thrash guard (neo2.core.agent.loop) catches the same action in consecutive turns only:
a turn of looks in between starts it over. So a click by a label the window doesn't have failed
again and again with a search or a page read between the tries (core.log 4080-4101), and nothing
stopped it. Keeping the guard's count across looks would stop "press Next, look, press Next" too,
so this is a memory of *failures* instead:

* :func:`kind_of` — which kind a failure is, from NEO's **own** error words (the ``"Error: …"``
  answers of neo2/core/tools/computer/*, and the registry's ``took longer than`` / ``failed:``).
  Unknown words are ``other``. It never looks for an app's or a site's name: its only uses are a
  log enum, the daily counters and which hint the model gets;
* :func:`target_key` — what makes two calls "the same step": the tool and its arguments, without
  the ones that only carry words about it. Of the action tools' schemas, that is ``utterance``
  (click_text's "the user's whole request"); NEO's own ``_`` arguments go too (the registry drops
  them outside an approval anyway). A screenshot id or an element id stays: a new look gives new
  ones, and a call with them is a new step;
* :class:`RunFailures` — per run, how often each step failed, by kind. The 2nd identical failure
  gets one hint (:func:`advice`, by kind: it names tools, never apps); a 3rd identical call isn't
  run (:func:`skipped`, the ``skipped: `` that batch and history read as "no result"). An action
  that worked clears it all, since the screen moved on (a scroll, then the same click, is a fair
  try); a look that worked clears only the stale-id failures, whose own words ask for that look.

Only failures count: a call that worked is never skipped. NEO_STUCK_BREAKER=false turns the hint and
the skip off; the kinds are still logged and counted (numbers and fixed names, never content).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Literal, get_args

from neo2.core import runtime
from neo2.core.providers.base import ToolCall

Kind = Literal[
    "not_found", "offscreen", "stale_id", "not_frontmost", "disabled", "secure_field", "timeout", "refused", "other"
]  # fmt: skip
KINDS: tuple[str, ...] = get_args(Kind)  # the only words a failure's kind is ever logged or counted as

# NEO's own error words → kind, first match wins (so the order matters: "Safari didn't come to the
# front (NEO's own window is still there)" is about what is in front, not a refusal). Lower case.
_TABLE: tuple[tuple[Kind, re.Pattern[str]], ...] = tuple(
    (k, re.compile(p))
    for k, p in (
        ("secure_field", r"password field"),
        ("timeout", r"took longer than|timed out|busy or hung|isn't answering|isn't responding"),
        ("not_frontmost", r"didn't come to the front"),
        (
            "refused",
            r"neo never|neo's own|i won't|neo doesn't (?:read|write|edit)|accessibility is still off"
            r"|you can't see the screen",
        ),
        (
            "stale_id",
            r"unknown element|ids come from the latest|no screenshot .{0,40}any more|no longer has screenshot"
            r"|isn't open any more|neo last saw that board|since neo found the board|isn't where neo last saw"
            r"|take a screenshot of it first|has no board from screenshot",
        ),
        ("offscreen", r"isn't on screen|is off every screen|is off the screen|is outside screenshot"),
        (
            "not_frontmost",
            r"isn't in front|no app is in front|is not running|has no window in front| covers |is over the words"
            r"|over that spot|under another window",
        ),
        ("disabled", r"greyed out|can't be pressed|can't be chosen|waiting on an open sheet or dialog"),
        (
            "not_found",
            r"is called '|no menu item|no menu command|no field called|isn't a label i can find"
            r"|has no window called|isn't an installed app|does not exist|doesn't exist|old_string not found",
        ),
    )
)
_HEAD = 600  # NEO's own words come first; what follows is often a listing or the app's text


def kind_of(tool: str, text: str) -> Kind:
    """The kind of failure this answer is (``other`` when NEO's words don't say). Where the
    registry names the tool ("Error: <tool> took longer than…", "Error: <tool> failed: …") the name
    is left out: a tool called "password_field_check" isn't a secure field."""
    said = str(text or "")[:_HEAD].replace("’", "'").lower()
    named = f"error: {tool.lower()} " if tool else ""
    if named and said.startswith(named):
        said = "error: " + said[len(named) :]
    for kind, pattern in _TABLE:
        if pattern.search(said):
            return kind
    return "other"


_VOLATILE = frozenset({"utterance"})


def target_key(tool: str, args: dict[str, Any] | None) -> str:
    """The step a call is: its tool and arguments, in a fixed order, without the ones that only
    carry words about it (``utterance``, NEO's own ``_`` arguments)."""
    kept = {k: v for k, v in (args or {}).items() if k not in _VOLATILE and not str(k).startswith("_")}
    return tool + json.dumps(kept, sort_keys=True, default=str)


_HINT = "This exact step failed twice the same way."
_ADVICE: dict[str, str] = {
    "not_found": "Don't repeat it: look again (ax_find / look) or reach the control another way; for a web "
    "page, page_read and the browser tools read the page itself.",
    "offscreen": "Don't repeat it: bring it into view first (scroll), or look again for where it is now.",
    "stale_id": "Don't repeat it: the ids or the picture it used are out of date — take a new look "
    "(look / ax_tree / screenshot) and use what that gives.",
    "not_frontmost": "Don't repeat it: activate the app first (activate_app), then look again before acting.",
    "disabled": "Don't repeat it: that control can't be used right now — look at what is in the way first (an "
    "open sheet or dialog, a field to fill in).",
    "secure_field": "Don't repeat it: NEO never types into password fields — that part is the user's, so tell them.",
    "timeout": "Don't repeat it as it is: it is busy or not answering — try a different approach, or tell the user.",
    "refused": "Don't repeat it: NEO won't do this step — choose a different one, or tell the user.",
    "other": "Don't repeat it: try a different approach.",
}


def advice(kind: str) -> str:
    """The one line added to the 2nd identical failure's result: NEO's own words, by kind only
    (never anything the failure quoted), naming tools, never apps."""
    return f"{_HINT} {_ADVICE.get(kind, _ADVICE['other'])}"


def skipped(kind: str) -> str:
    """A 3rd identical call's result. ``skipped: `` is what batch.halted and the history read as
    "this didn't run": no result of an action, nothing unseen on the screen."""
    return (
        f"skipped: this exact step already failed twice ({kind.replace('_', ' ')}) — choose a different step"
    )


class RunFailures:
    """One agent run's failed steps: how often each (step, kind) failed since the last action that
    worked. Lives as long as the run; nothing of it is kept."""

    def __init__(self) -> None:
        self._n: Counter[tuple[str, str]] = Counter()

    def record(self, call: ToolCall, *, ok: bool, kind: str) -> int:
        """An action's outcome. One that worked clears everything (0); a failure returns how often
        this step has now failed this way."""
        if ok:
            self.reset()
            return 0
        key = (target_key(call.name, call.args), kind or "other")
        self._n[key] += 1
        return self._n[key]

    def seen_twice(self, call: ToolCall) -> str:
        """The kind this exact step already failed twice with ("" = it may run)."""
        key = target_key(call.name, call.args)
        for (k, kind), n in self._n.items():
            if k == key and n >= 2:
                return kind
        return ""

    def looked(self) -> None:
        """A look worked: the ids and pictures a stale-id failure was about are new now."""
        for k in [k for k in self._n if k[1] == "stale_id"]:
            del self._n[k]

    def reset(self) -> None:
        self._n.clear()


def tally(name: str) -> None:
    """One of NEO.app's daily counters (neo2.core.reflex.learn): ``tool_fail``, ``tool_fail_<kind>``,
    ``tool_skip_repeat``, ``agent_end_<why>``. Never an embedded agent's: its runs are its host's."""
    if runtime.library_bound():
        return
    try:
        from neo2.core.reflex import learn

        learn.count(name)
    except Exception:  # noqa: BLE001 — a counter must never break a run
        pass
