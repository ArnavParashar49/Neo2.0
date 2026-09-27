"""How hard the brain thinks, step by step.

Most screen steps are mechanical — press the labelled control, type, click a square already
located — and pay for thinking in waiting: Gemini Flash's published time to its first token is
~10–24 s at 'high' and ~1–2 s at 'low'. So a run gets a level to *start* at, and each step's level
follows how the run is going:

* one level up after a miss: a step that failed, an action after which nothing changed, a sheet
  or dialog that opened, the same action asked for twice in a row (the thrash guard's near-miss);
* one level back down after an action that worked — never below where the run started, never
  above 'high'.

The level is read off the transcript each step (the request being worked on, from its first turn
in the user's words), so a run carried on after a confirmation goes on where it was, and there is
no state to keep in step with the loop.

Where a run starts (:func:`start_level`): today's rule — 'high' for an agent task, 'medium'
otherwise — with the setting ``screen_effort`` in place of 'high' for tasks that work on the screen,
and as the most the one-tool light lane starts at. Its default is 'high', today's level: it
changes only after the user's A/B on the self-test window, ≥10 runs per arm (DECISIONS 47). When
Laya is unsure what the request even is (a low margin between its top two intents), a screen run
starts one level higher than the setting, but never above today's level.

'minimal' is used only where a brain takes it: the chain gives any other brain the next level up
(:func:`for_brain`), and Gemini, told once by a model that it doesn't take it, asks that model at
'low' from then on (neo.providers.gemini).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from typing import Any, cast

from neo.providers.base import Level, Message, ToolResult
from neo.providers.history import OBSERVATIONS, halted, request_start

log = logging.getLogger(__name__)

LEVELS: tuple[Level, ...] = ("minimal", "low", "medium", "high")
# What a brain takes unless it says otherwise (`efforts` on the provider).
USUAL: frozenset[str] = frozenset({"low", "medium", "high"})
# Laya's margin between its top two intents under which it is "unsure what the request is".
UNSURE_MARGIN = 0.2
# Looks: a step made only of these is neither a miss nor a success (the model is still looking).
_LOOKS = OBSERVATIONS | {"look", "observe", "more_tools"}
# The words the tools use for a miss when they report no Effect (neo.tools.computer.pointer).
_MISS_WORDS = ("Nothing visibly changed",)
_SHEET_WORDS = ("a sheet or dialog opened",)


def _i(level: str) -> int:
    return LEVELS.index(cast(Level, level)) if level in LEVELS else LEVELS.index("medium")


def lower(a: Level, b: Level) -> Level:
    return LEVELS[min(_i(a), _i(b))]


def higher(a: Level, b: Level) -> Level:
    return LEVELS[max(_i(a), _i(b))]


def up(level: Level, ceiling: Level = "high") -> Level:
    return LEVELS[min(_i(level) + 1, _i(ceiling))]


def for_brain(p: Any, level: str) -> Level:
    """The level to ask this brain at: `level`, or the nearest one above it that it takes (a
    brain declares `efforts`; any other takes low/medium/high)."""
    takes = getattr(p, "efforts", None) or USUAL
    lv = _i(level)
    for x in LEVELS[lv:]:
        if x in takes:
            return x
    return LEVELS[lv] if LEVELS[lv] in USUAL else "low"


def start_level(d: Any, *, screen: bool, light: bool = False, knob: str | None = None) -> Level:
    """Where a run's thinking starts. `d`: the reflex's Decision (or None). `screen`: the task
    works on the screen (the reflex says it needs the screen, the computer tools were chosen for
    it, or it is the play-a-game route). `light`: the one-tool lane. `knob`: settings().screen_effort."""
    if knob is None:
        from neo.config import settings

        knob = settings().screen_effort
    knob_l: Level = cast(Level, knob if knob in LEVELS else "high")
    agent_task = d is not None and getattr(d, "intent", "") == "agent_task"
    today: Level = "high" if agent_task else "medium"  # the level every run started at before this
    if screen:
        level = knob_l if agent_task else lower(knob_l, "medium")
    elif light:
        level = lower(knob_l, "medium")
    else:
        return today
    if unsure(d):
        level = lower(up(level), today)  # one more, never past what it was before
    return level


def unsure(d: Any) -> bool:
    """Laya isn't sure what the request is: not confident, or its top two intents nearly tied."""
    if d is None:
        return False
    margin = getattr(d, "margin", None)
    confident = getattr(d, "confident", True)
    return (not confident) or (margin is not None and float(margin) < UNSURE_MARGIN)


def trouble(results: Iterable[ToolResult]) -> bool | None:
    """How a step went: True (a miss: something failed, nothing changed, a sheet opened), False
    (an action worked), None (it only looked, or stopped to ask the user)."""
    worked = False
    for r in results:
        if halted(r):
            continue  # stopped to ask the user, or skipped: not the brain's miss
        eff = getattr(r, "effect", None)
        if not r.ok:
            return True
        if eff is not None and (getattr(eff, "changed", None) is False or getattr(eff, "sheet", None)):
            return True
        if eff is None and any(w in r.content for w in (*_MISS_WORDS, *_SHEET_WORDS)):
            return True
        if r.name not in _LOOKS:
            worked = True
    return False if worked else None


def _acts(m: Message) -> str:
    """The side-effect calls of an assistant turn, as one comparable key ('' when it only looks)."""
    return "|".join(
        c.name + json.dumps(c.args, sort_keys=True, default=str) for c in m.tool_calls if c.name not in _LOOKS
    )


class Effort:
    """A run's thinking level, per step: `start`, one up after each miss, one back down after
    each action that worked (never below `start`, never above `ceiling`)."""

    def __init__(self, start: Level, *, ceiling: Level = "high") -> None:
        self.start: Level = cast(Level, start if start in LEVELS else "medium")
        self.ceiling: Level = higher(ceiling, self.start)
        self.last: Level = self.start
        self.why = ""

    def __repr__(self) -> str:
        return f"Effort(start={self.start!r}, last={self.last!r})"

    def at(self, messages: Sequence[Message]) -> Level:
        """This step's level, from the request's transcript so far."""
        start = request_start(list(messages)) if messages else 0
        lv, floor, top = _i(self.start), _i(self.start), _i(self.ceiling)
        prev_acts = ""
        repeated = False
        why = ""
        for m in messages[start:]:
            if m.role == "assistant" and m.tool_calls:
                acts = _acts(m)
                # The same action again: one short of the thrash guard's stop, whatever it returned.
                repeated = bool(acts) and acts == prev_acts
                prev_acts = acts
            elif m.role == "tool":
                t = trouble(m.tool_results)
                if repeated:
                    lv, why = min(lv + 1, top), "the same action twice"
                elif t is True:
                    lv, why = min(lv + 1, top), "a miss"
                elif t is False:
                    lv, why = max(lv - 1, floor), "an action worked"
                repeated = False
        level = LEVELS[lv]
        if level != self.last:
            log.info(f"[brain] thinking {self.last} → {level} ({why})")
        self.last, self.why = level, why
        return level
