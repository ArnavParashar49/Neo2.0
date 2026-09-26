"""Zero-LLM dispatch: regex fast paths, and plans that cover a whole utterance with them.

Tools declare `fast_path=[(regex, args)]`. `match()` finds the tool whose pattern accepts a
text. `plan()` covers a whole utterance with fast-path steps — "open notes and type salt and
pepper" → open_app(notes), type_text("salt and pepper") — or returns None so the utterance
goes to a model. The rules that make it safe:

* A *payload* tool (type_text, dictate, click_text, notes_create, memory, web_search) takes
  the rest of the sentence as its text: "salt and pepper" is never split at "and". Only a
  trailing clause that is itself a plain command ("… and press enter") is split off.
* A `chain=False` tool runs only as the first step: in "open youtube and search for cats" the
  search belongs to YouTube, so the plan fails and the model handles it.
* Anything the regexes can't cover — one unknown clause — makes the whole plan None.
"""

from __future__ import annotations

import re

from neo.agent.registry import RegisteredTool, registry

Step = tuple[str, dict]

SPLIT = re.compile(r"\s*(?:,|;|\band then\b|\bthen\b|\band\b|\bafter that\b)\s*", re.I)
_EDGE = " ,.;!?"
_QUOTED = re.compile(r"""^(["“'])([^"“”']*)(["”'])$""")


def strip_quotes(v):
    """type 'Laptops' → Laptops: one balanced pair of quotes around a payload is punctuation."""
    if isinstance(v, str) and (m := _QUOTED.match(v.strip())):
        return m.group(2)
    return v


def match(text: str) -> Step | None:
    """The first tool whose fast-path regex accepts the text. System tools (volume,
    brightness, timer, clock) come first so a broad pattern can never shadow a specific one."""
    tools = sorted(registry().all(), key=lambda t: 0 if t.category == "system" else 1)
    for t in tools:
        for pattern, args in t.fast_path:
            m = re.search(pattern, text, re.I)
            if not m:
                continue
            # "<name>" values are filled from the named regex group of the same name.
            filled: dict = {}
            for k, v in args.items():
                if isinstance(v, str) and v.startswith("<") and v.endswith(">"):
                    filled[k] = strip_quotes(m.groupdict().get(v[1:-1]))
                else:
                    filled[k] = v
            return t.name, filled
    return None


def _tool(name: str) -> RegisteredTool | None:
    return registry().get(name)


def is_payload(name: str) -> bool:
    t = _tool(name)
    return bool(t and t.payload)


def plan(text: str, *, first_index: int = 0) -> list[Step] | None:
    """Fast-path steps that cover the *whole* text, in order, or None.

    `first_index` > 0 means steps already ran for this utterance (chain rules apply)."""
    text = text.strip(_EDGE)
    if not text:
        return []
    whole = match(text)
    if whole is not None:
        tool, _ = whole
        t = _tool(tool)
        if first_index > 0 and t is not None and not t.chain:
            return None
        if t is not None and t.payload:
            return _split_trailing_commands(text, whole, first_index)
        return [whole]
    steps: list[Step] = []
    pos = 0
    for m in list(SPLIT.finditer(text)) + [None]:
        end = m.start() if m else len(text)
        clause = text[pos:end].strip(_EDGE)
        nxt = m.end() if m else len(text)
        if not clause:
            pos = nxt
            continue
        hit = match(clause)
        if hit is None:
            return None
        t = _tool(hit[0])
        if (first_index + len(steps)) > 0 and t is not None and not t.chain:
            return None
        if t is not None and t.payload and pos > 0:
            # The payload runs to the end of the utterance ("… and type salt and pepper").
            rest = plan(text[pos:], first_index=first_index + len(steps))
            return None if rest is None else steps + rest
        # (A payload clause at the very start whose whole-text match failed — "click play and
        # then press enter" — is just this clause: its regex refused to swallow the rest.)
        steps.append(hit)
        pos = nxt
    return steps


def _split_trailing_commands(text: str, whole: Step, first_index: int) -> list[Step]:
    """ "type hello and press enter" → type_text(hello), hotkey(return). Only a tail that starts
    with a plain (non-payload) command is split off; "salt and pepper" stays one payload."""
    tool = whole[0]
    for m in SPLIT.finditer(text):
        head = text[: m.start()].strip(_EDGE)
        tail = text[m.end() :].strip(_EDGE)
        if not head or not tail:
            continue
        h = match(head)
        if h is None or h[0] != tool:
            continue
        rest = plan(tail, first_index=first_index + 1)
        if rest and not is_payload(rest[0][0]):  # the tail starts with a real command
            return [h, *rest]
    return [whole]
