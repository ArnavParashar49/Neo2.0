"""Fit a request into a brain's token budget before it is sent.

A fallback brain can have far less room than the one a run started on: Groq's free tier takes at
most ~8K tokens per request — the prompt *plus* the answer it may write — where Gemini took 250K.
An agent run that fails over halfway (a dozen steps of UI trees, every tool schema) would die on
a 413. So the request is shrunk, least useful first, and only refused when even the minimum
doesn't fit:

1. pictures: none for a text-only brain (it gets a note); only the newest for one that sees
2. older tool results, cut to head and tail, then to a stub — the latest steps stay whole, and so
   does the newest result of each tool in this request (a halted step asking for approval doesn't
   push the board the model is playing on out of view)
3. the room reserved for the answer, down to a floor
4. tool descriptions, down to their first sentence
5. older conversation text; then earlier requests altogether
6. the tools this run hasn't used — never an essential one (`more_tools`, which gets any other
   back; a fenced run's tools) — then as many of those as still fit come back, in the order given
7. the latest results themselves

Nothing here mutates the caller's messages (they are the session's history).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace

from neo.providers.base import Message, RequestTooLarge, ToolCall, ToolResult, ToolSpec
from neo.providers.history import continues_request, normalize

_BYTES_PER_TOKEN = 3.4  # UTF-8 bytes: ~4 per token for English, less for code, UI trees and JSON
_IMAGE_TOKENS = 1300  # a screenshot as a vision brain bills it (roughly; tiles vary)
_MSG_OVERHEAD = 6
_TOOL_OVERHEAD = 12
_HEADROOM = 0.9  # estimates are estimates: aim under the limit, not at it
_KEEP_WHOLE = 2  # the newest result turns are never trimmed before everything else is
# A step that only asked the user (or waited on that) isn't a result worth keeping whole in its place.
_HALTS = ("NEEDS_CONFIRM", "NEEDS_USER", "skipped: ")
_MIN_OUTPUT = 512
_ENABLED = re.compile(r"^Enabled: (.*?)\. They are available", re.S)


@dataclass
class Fitted:
    messages: list[Message]
    tools: list[ToolSpec] | None
    max_tokens: int
    tokens: int  # estimated prompt tokens, as sent
    trimmed: list[str] = field(default_factory=list)  # what had to go, for the log

    def offered(self) -> list[str] | None:
        """The names of the tools sent (Turn.offered), None when the request had none."""
        return None if self.tools is None else [s.name for s in self.tools]


def _n(text: str) -> float:
    return len(text.encode("utf-8", "ignore")) / _BYTES_PER_TOKEN


def _spec_tokens(s: ToolSpec) -> float:
    return (
        _n(s.name) + _n(s.description) + _n(json.dumps(s.parameters, separators=(",", ":"))) + _TOOL_OVERHEAD
    )


def estimate(
    messages: list[Message], system: str = "", tools: list[ToolSpec] | None = None, *, vision: bool = True
) -> int:
    """Rough prompt tokens for this request (conservative: better to trim a little early than
    to be refused)."""
    return int(_estimate(messages, system, tools, vision))


def _estimate(messages: list[Message], system: str, tools: list[ToolSpec] | None, vision: bool) -> float:
    t = _n(system)
    for s in tools or []:
        t += _spec_tokens(s)
    images = 0
    for m in messages:
        t += _MSG_OVERHEAD + _n(m.text)
        images += len(m.images)
        for c in m.tool_calls:
            t += _MSG_OVERHEAD + _n(c.name) + _n(json.dumps(c.args, ensure_ascii=False, default=str))
        for r in m.tool_results:
            t += _MSG_OVERHEAD + _n(r.name) + _n(r.content)
            images += len(r.images)
    return t + images * (_IMAGE_TOKENS if vision else 8)


def _cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    return (
        text[:head].rstrip()
        + f" …[{len(text) - limit} chars trimmed to fit]… "
        + text[-(limit - head) :].lstrip()
    )


def _tool_turns(msgs: list[Message]) -> list[int]:
    return [i for i, m in enumerate(msgs) if m.role == "tool"]


def _halted(r: ToolResult) -> bool:
    return r.content.lstrip().startswith(_HALTS)


def _latest(msgs: list[Message]) -> set[int]:
    """The result turns kept whole until the very last step: the newest few that hold a real
    result, and the one with the newest result of each tool in the request being worked on."""
    turns = _tool_turns(msgs)
    real = [i for i in turns if not all(_halted(r) for r in msgs[i].tool_results)]
    keep = set(real[-_KEEP_WHOLE:])
    start, seen = _request_start(msgs), set()
    for i in reversed(real):
        if i < start:
            break
        names = {r.name for r in msgs[i].tool_results if not _halted(r)}
        if names - seen:
            keep.add(i)
        seen |= names
    return keep


def _user_turns(msgs: list[Message]) -> list[int]:
    return [i for i, m in enumerate(msgs) if m.role == "user"]


def _shrink_results(msgs: list[Message], limit: int, which: list[int]) -> list[Message]:
    out = list(msgs)
    for i in which:
        m = out[i]
        if any(len(r.content) > limit or r.images for r in m.tool_results):
            out[i] = Message.tool(
                [ToolResult(r.call_id, r.name, _cut(r.content, limit), r.ok) for r in m.tool_results]
            )
    return out


def _drop_images(msgs: list[Message], keep: int) -> list[Message]:
    """All but the newest `keep` pictures of tool results go; so do those of older user turns.
    The request's own picture ("what's in this photo?") stays."""
    out = list(msgs)
    users = _user_turns(out)
    last_user = users[-1] if users else -1
    kept = 0
    for i in range(len(out) - 1, -1, -1):
        m = out[i]
        if m.role == "tool" and any(r.images for r in m.tool_results):
            results = []
            for r in m.tool_results:
                if r.images and kept < keep:
                    kept += len(r.images)
                    results.append(r)
                elif r.images:
                    results.append(
                        ToolResult(r.call_id, r.name, r.content + " [picture dropped to fit]", r.ok)
                    )
                else:
                    results.append(r)
            if any(a is not b for a, b in zip(results, m.tool_results, strict=True)):
                out[i] = Message.tool(results)
        elif m.role == "user" and m.images and i != last_user:
            out[i] = Message.user(m.text + " [picture dropped to fit]")
    return out


def _short_tools(tools: list[ToolSpec]) -> list[ToolSpec]:
    """First sentence of each description; parameter descriptions dropped (names and types stay —
    a parameter that is itself called "description" is a schema, not a string, and stays too)."""

    def strip(schema):
        if isinstance(schema, dict):
            return {k: strip(v) for k, v in schema.items() if not (k == "description" and isinstance(v, str))}
        if isinstance(schema, list):
            return [strip(v) for v in schema]
        return schema

    out = []
    for s in tools:
        first = re.split(r"(?<=[.!?])\s", s.description, maxsplit=1)[0][:200]
        out.append(replace(s, description=first, parameters=strip(s.parameters)))
    return out


def _essential(s: ToolSpec) -> bool:
    return s.essential or s.name == "more_tools"


def _used_tools(msgs: list[Message]) -> set[str]:
    """The tools this conversation called, and those `more_tools` switched on for it."""
    used = {c.name for m in msgs for c in m.tool_calls}
    for m in msgs:
        for r in m.tool_results:
            if r.name == "more_tools" and (g := _ENABLED.match(r.content)):
                used |= {x.strip() for x in g.group(1).split(",")}
    return used


def _shorten_text(msgs: list[Message], before: int, limit: int) -> list[Message]:
    out = list(msgs)
    for i in range(before):
        m = out[i]
        if m.role == "user" and len(m.text) > limit:
            out[i] = Message.user(_cut(m.text, limit), images=m.images)
        elif m.role == "assistant" and (
            len(m.text) > limit or any(len(json.dumps(c.args, default=str)) > limit for c in m.tool_calls)
        ):
            calls = [
                ToolCall(
                    c.id,
                    c.name,
                    {k: (_cut(v, limit // 2) if isinstance(v, str) else v) for k, v in c.args.items()},
                    c.signature,
                    c.source,
                )
                for c in m.tool_calls
            ]
            out[i] = Message.assistant(_cut(m.text, limit), calls)
    return out


def _request_start(msgs: list[Message]) -> int:
    """Where the request being worked on begins: its last user turn in the user's words (NEO's
    own notes — "[User confirmed …]", "[verify] …" — continue a request, they don't start one)."""
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].role == "user" and not continues_request(msgs[i]):
            return i
    users = _user_turns(msgs)
    return users[0] if users else 0


def fit(
    messages: list[Message],
    *,
    system: str = "",
    tools: list[ToolSpec] | None = None,
    budget: int,
    max_tokens: int,
    vision: bool,
    name: str = "brain",
    scale: float = 1.0,
) -> Fitted:
    """The request, trimmed until estimated prompt × `scale` + `max_tokens` fits in `budget`
    (prompt and answer together: what a per-request token limit counts). `scale` corrects the
    estimate once a brain has told us the real count. Raises RequestTooLarge when nothing more
    can go."""
    msgs = normalize(messages)
    specs = list(tools) if tools else tools
    out = max_tokens
    limit = int(budget * _HEADROOM)
    trimmed: list[str] = []
    cut_from: list[ToolSpec] | None = None  # the tool list before any were left out

    def prompt() -> int:
        return int(estimate(msgs, system, specs, vision=vision) * scale)

    def fits() -> bool:
        return prompt() + out <= limit

    if fits():
        return Fitted(msgs, specs, out, prompt())
    before = prompt()

    def refill() -> None:
        """Room left once everything fits goes back to the tools that were left out."""
        nonlocal specs
        if cut_from is None or specs is None:
            return
        keep = {s.name for s in specs}
        t = _estimate(msgs, system, specs, vision)
        for s in cut_from:
            if s.name not in keep and int((t + (c := _spec_tokens(s))) * scale) + out <= limit:
                keep.add(s.name)
                t += c
        was, specs = specs, [s for s in cut_from if s.name in keep]
        if not fits():  # (an estimate that doesn't add up: keep the list that fitted)
            specs = was

    def done(what: str, was: list[Message] | None = None) -> Fitted | None:
        if was is None or len(was) != len(msgs) or any(a is not b for a, b in zip(was, msgs, strict=True)):
            trimmed.append(what)  # (a step that found nothing to cut isn't reported)
        if not fits():
            return None
        refill()
        if cut_from is not None and specs is not None:
            trimmed[trimmed.index("tools not in use")] = (
                f"tools not in use ({len(cut_from) - len(specs)} left out)"
            )
        print(
            f"[brain] {name}: request fitted ~{before}→{prompt()} tokens (+{out} answer) by trimming {', '.join(trimmed)}"
        )
        return Fitted(msgs, specs, out, prompt(), trimmed)

    # 1. pictures
    if any(m.images or any(r.images for r in m.tool_results) for m in msgs):
        was, msgs = msgs, _drop_images(msgs, keep=1 if vision else 0)
        if r := done("pictures", was):
            return r
    # 2. older tool results, newest steps whole
    latest = _latest(msgs)
    older = [i for i in _tool_turns(msgs) if i not in latest]
    for lim in (1500, 400, 160) if older else ():
        was, msgs = msgs, _shrink_results(msgs, lim, older)
        if r := done(f"older results to {lim} chars", was):
            return r
    # 3. the answer's reserve
    floor = min(max_tokens, max(_MIN_OUTPUT, min(2048, limit // 4)))
    if out > floor:
        out = floor
        if r := done(f"answer room to {out}"):
            return r
    # 4. tool descriptions
    if specs:
        specs = _short_tools(specs)
        if r := done("tool descriptions"):
            return r
    # 5. older conversation: its text, then earlier requests altogether
    start = _request_start(msgs)
    was, msgs = msgs, _shorten_text(msgs, start, 300)
    if r := done("older conversation text", was):
        return r
    if start > 0:
        msgs = msgs[start:]
        if r := done("earlier requests"):
            return r
    # 6. tools this run hasn't used (the ones it has, and essential ones, stay); refilled in done()
    if specs:
        used = _used_tools(msgs)
        kept = [s for s in specs if s.name in used or _essential(s)]
        if len(kept) < len(specs):
            cut_from, specs = specs, kept
            if r := done("tools not in use"):
                return r
    # 7. the latest results too
    for lim in (2000, 600):
        was, msgs = msgs, _shrink_results(msgs, lim, _tool_turns(msgs))
        if r := done(f"latest results to {lim} chars", was):
            return r
    if out > _MIN_OUTPUT:
        out = _MIN_OUTPUT
        if r := done(f"answer room to {out}"):
            return r
    raise RequestTooLarge(
        f"{name}: request too large — ~{prompt() + out} tokens even trimmed, limit {budget}"
    )
