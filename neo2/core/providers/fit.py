"""Fit a request into a brain's token budget before it is sent.

A fallback brain can have far less room than the one a run started on: Groq's free tier takes at
most ~8K tokens per request — the prompt *plus* the answer it may write — where Gemini took 250K.
An agent run that fails over halfway (a dozen steps of UI trees, every tool schema) would die on
a 413. So the request is shrunk, least useful first, and only refused when even the minimum
doesn't fit:

1. pictures: none for a text-only brain (it gets a note); only the newest for one that sees
2. older tool results, cut to head and tail, then to a stub. Two kinds stay whole: the newest
   steps, and — behind those — the newest look at each thing observed in this request (the board
   the model is playing on isn't pushed out of view by the approvals and actions after it)
3. the room reserved for the answer, down to a floor
4. tool descriptions, down to their first sentence
5. older conversation text; then earlier requests altogether
6. the tools this run hasn't used — never an essential one (`more_tools`, which gets any other
   back; a fenced run's tools) — then as many of those as still fit come back, in the order given
7. the looks kept whole behind newer steps, then the newest steps themselves: each cut only as
   far as the request must shrink (the room that frees is theirs, it doesn't go back to tools)

A budget that is a per-minute rate (Groq's free tier) is paid back before the next step can go:
a prompt of half of it already means a wait of ~30s per step. For such a brain the tools that
come back in 6 are the first few in the order given (a tenth of the budget; the agent loop lists
the observe/act ones first), and more only while the prompt stays under that half.

A picture costs what the brain it goes to bills for it: Gemini a fixed 280 / 560 / 1,120 / 2,240
tokens by its detail (media resolution); Claude about width × height / 750 once scaled to its limit;
Groq's vision model ~2,048; a text-only brain nothing (it gets a note instead).

Nothing here mutates the caller's messages (they are the session's history).
"""

from __future__ import annotations

import json
import math
import re
import struct
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from neo2.core import logger
from neo2.core.providers.base import BLIND_PICTURE, ImagePart, Message, RequestTooLarge, ToolCall, ToolSpec
from neo2.core.providers.history import OBSERVATIONS, halted, normalize, request_start

_BYTES_PER_TOKEN = 3.4  # UTF-8 bytes: ~4 per token for English, less for code, UI trees and JSON
_IMAGE_TOKENS = 1300  # a picture as a seeing brain bills it, when nothing more is known of that brain
# Gemini 3: a fixed count per media resolution ("auto" is its default for pictures: high).
_GEMINI_IMAGE_TOKENS = {"low": 280, "medium": 560, "high": 1120, "ultra": 2240, "auto": 1120}
_GROQ_IMAGE_TOKENS = 2048  # Groq's vision model, per picture (its docs)
# Claude: ≈ width × height / 750 once the API has scaled it to fit (the larger of its limits,
# Opus 4.7+'s 2576 px / 3.75 MP: an estimate a little high is safer than one too low).
_CLAUDE_PX_PER_TOKEN = 750
_CLAUDE_LONG_EDGE, _CLAUDE_PIXELS = 2576, 3_750_000
_BLIND_IMAGE_TOKENS = 8  # a text-only brain gets a short note in the picture's place
_MSG_OVERHEAD = 6
_TOOL_OVERHEAD = 12
_HEADROOM = 0.9  # estimates are estimates: aim under the limit, not at it
_KEEP_WHOLE = 2  # the newest result turns are never trimmed before everything else is
_MIN_OUTPUT = 512
_MIN_RESULT = 600  # chars a result kept whole is cut to, at the very least, before giving up
_PER_MINUTE_SHARE = 0.5  # a per-minute budget: tools come back while the prompt is under this…
_PER_MINUTE_TOOLS = 0.1  # …or while they add no more than this to it
_ENABLED = re.compile(r"^Enabled: (.*?)\. They are available", re.S)

log = logger(__name__)


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


def image_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) of a PNG, JPEG, GIF or WebP from its header, None when it can't be told."""
    try:
        if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
            w, h = struct.unpack(">II", data[16:24])
            return int(w), int(h)
        if data[:6] in (b"GIF87a", b"GIF89a"):
            w, h = struct.unpack("<HH", data[6:10])
            return int(w), int(h)
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            kind = data[12:16]
            if kind == b"VP8X":
                return 1 + int.from_bytes(data[24:27], "little"), 1 + int.from_bytes(data[27:30], "little")
            if kind == b"VP8 ":
                w, h = struct.unpack("<HH", data[26:30])
                return int(w & 0x3FFF), int(h & 0x3FFF)
            if kind == b"VP8L":
                b = int.from_bytes(data[21:25], "little")
                return 1 + (b & 0x3FFF), 1 + ((b >> 14) & 0x3FFF)
            return None
        if data[:2] == b"\xff\xd8":  # JPEG: walk the segments to a start-of-frame
            i = 2
            while i + 9 < len(data):
                if data[i] != 0xFF:
                    i += 1
                    continue
                marker = data[i + 1]
                if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                    i += 2
                    continue
                size = struct.unpack(">H", data[i + 2 : i + 4])[0]
                if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    h, w = struct.unpack(">HH", data[i + 5 : i + 9])
                    return int(w), int(h)
                i += 2 + size
    except (struct.error, IndexError, ValueError):
        return None
    return None


def _claude_image_tokens(im: ImagePart) -> int:
    size = image_size(im.data)
    if size is None:
        return int(_CLAUDE_PIXELS / _CLAUDE_PX_PER_TOKEN)
    w, h = size
    if w <= 0 or h <= 0:
        return _IMAGE_TOKENS
    scale = min(1.0, _CLAUDE_LONG_EDGE / max(w, h), math.sqrt(_CLAUDE_PIXELS / (w * h)))
    return max(1, math.ceil((w * scale) * (h * scale) / _CLAUDE_PX_PER_TOKEN))


def _gemini_image_tokens(im: ImagePart) -> int:
    return _GEMINI_IMAGE_TOKENS.get(getattr(im, "detail", "auto"), _GEMINI_IMAGE_TOKENS["auto"])


# What one picture costs, by the name of the brain it goes to (its prefix). A seeing brain not
# listed costs _IMAGE_TOKENS; a new one adds its own row.
IMAGE_TOKENS: dict[str, Callable[[ImagePart], int]] = {
    "gemini": _gemini_image_tokens,
    "claude": _claude_image_tokens,
    "groq": lambda im: _GROQ_IMAGE_TOKENS,
}


def image_tokens(im: ImagePart, name: str = "", *, vision: bool = True) -> int:
    """What one picture costs the brain called `name` (a text-only one: only the note)."""
    if not vision:
        return _BLIND_IMAGE_TOKENS
    for prefix, cost in IMAGE_TOKENS.items():
        if name.startswith(prefix):
            return int(cost(im))
    return _IMAGE_TOKENS


def estimate(
    messages: list[Message],
    system: str = "",
    tools: list[ToolSpec] | None = None,
    *,
    vision: bool = True,
    name: str = "",
) -> int:
    """Rough prompt tokens for this request (conservative: better to trim a little early than
    to be refused). `name`: the brain it goes to, for what its pictures cost."""
    return int(_estimate(messages, system, tools, vision, name))


def _estimate(
    messages: list[Message], system: str, tools: list[ToolSpec] | None, vision: bool, name: str = ""
) -> float:
    t = _n(system)
    for s in tools or []:
        t += _spec_tokens(s)
    for m in messages:
        t += _MSG_OVERHEAD + _n(m.text)
        t += sum(image_tokens(im, name, vision=vision) for im in m.images)
        for c in m.tool_calls:
            t += _MSG_OVERHEAD + _n(c.name) + _n(json.dumps(c.args, ensure_ascii=False, default=str))
        for r in m.tool_results:
            t += _MSG_OVERHEAD + _n(r.name) + _n(r.content)
            t += sum(image_tokens(im, name, vision=vision) for im in r.images)
    return t


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


def _protected(msgs: list[Message]) -> tuple[list[int], list[int]]:
    """The result turns kept whole until the last step, most precious first:
    (the newest few that hold a real result — not a halted step,
     behind those, the newest look at each observed thing in the request being worked on)."""
    real = [i for i in _tool_turns(msgs) if not all(halted(r) for r in msgs[i].tool_results)]
    newest = real[-_KEEP_WHOLE:]
    start, seen, spared = request_start(msgs), set(), []
    for i in reversed(real):
        if i < start:
            break
        names = {r.name for r in msgs[i].tool_results if r.name in OBSERVATIONS and not halted(r)}
        if names - seen and i not in newest:
            spared.append(i)
        seen |= names
    return newest, sorted(spared)


def _user_turns(msgs: list[Message]) -> list[int]:
    return [i for i, m in enumerate(msgs) if m.role == "user"]


def _shrink_results(msgs: list[Message], limit: int, which: list[int]) -> list[Message]:
    out = list(msgs)
    for i in which:
        m = out[i]
        if any(len(r.content) > limit or r.images for r in m.tool_results):
            out[i] = Message.tool(
                [replace(r, content=_cut(r.content, limit), images=[]) for r in m.tool_results]
            )
    return out


def _drop_images(msgs: list[Message], keep: int, vision: bool = True) -> list[Message]:
    """All but the newest `keep` pictures of tool results go; so do those of older user turns.
    The request's own picture ("what's in this photo?") stays. A brain without vision is told
    plainly that it can't see the picture, not that it was "dropped to fit"."""
    dropped = " [picture dropped to fit]" if vision else "\n" + BLIND_PICTURE
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
                    results.append(replace(r, content=r.content + dropped, images=[]))
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
    per_minute: bool = False,
) -> Fitted:
    """The request, trimmed until estimated prompt × `scale` + `max_tokens` fits in `budget`
    (prompt and answer together: what a per-request token limit counts). `scale` corrects the
    estimate once a brain has told us the real count. `per_minute`: the budget is a rate the
    brain refills per minute (see the module docstring). Raises RequestTooLarge when nothing
    more can go."""
    msgs = normalize(messages)
    specs = list(tools) if tools else tools
    out = max_tokens
    limit = int(budget * _HEADROOM)
    trimmed: list[str] = []
    cut_from: list[ToolSpec] | None = None  # the tool list before any were left out
    refill_ok = True  # until results kept whole had to be cut: that room is theirs

    def prompt() -> int:
        return int(estimate(msgs, system, specs, vision=vision, name=name) * scale)

    def fits() -> bool:
        return prompt() + out <= limit

    if fits():
        return Fitted(msgs, specs, out, prompt())
    before = prompt()

    def refill() -> None:
        """Room left once everything fits goes back to the tools that were left out."""
        nonlocal specs
        if cut_from is None or specs is None or not refill_ok:
            return
        keep = {s.name for s in specs}
        t = _estimate(msgs, system, specs, vision, name)
        cap = limit
        if per_minute:
            cap = max(int(budget * _PER_MINUTE_SHARE), int(t * scale) + int(budget * _PER_MINUTE_TOOLS))
        for s in cut_from:
            if (
                s.name not in keep
                and (p := int((t + (c := _spec_tokens(s))) * scale)) + out <= limit
                and p <= cap
            ):
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
        return finish()

    def finish() -> Fitted:
        refill()
        if cut_from is not None and specs is not None:
            trimmed[trimmed.index("tools not in use")] = (
                f"tools not in use ({len(cut_from) - len(specs)} left out)"
            )
        log.info(
            f"[brain] {name}: request fitted ~{before}→{prompt()} tokens (+{out} answer) by trimming {', '.join(trimmed)}"
        )
        return Fitted(msgs, specs, out, prompt(), trimmed)

    def squeeze(which: list[int]) -> int | None:
        """Cut the results in `which` only as far as the request must shrink: the longest length
        (≥ _MIN_RESULT chars) that fits, or None — left at the minimum — when even that doesn't."""
        nonlocal msgs
        base = msgs
        top = max((len(r.content) for i in which for r in base[i].tool_results), default=0)
        msgs = _shrink_results(base, top, which)  # (their pictures go first)
        if fits():
            return top
        lo = min(top, _MIN_RESULT)
        msgs = _shrink_results(base, lo, which)
        if not fits():
            return None
        hi = top
        while hi - lo > 32:
            mid = (lo + hi) // 2
            msgs = _shrink_results(base, mid, which)
            if fits():
                lo = mid
            else:
                hi = mid
        msgs = _shrink_results(base, lo, which)
        return lo

    # 1. pictures
    if any(m.images or any(r.images for r in m.tool_results) for m in msgs):
        was, msgs = msgs, _drop_images(msgs, keep=1 if vision else 0, vision=vision)
        if r := done("pictures", was):
            return r
    # 2. older tool results; the newest steps, and the newest look at each thing, whole
    newest, spared = _protected(msgs)
    older = [i for i in _tool_turns(msgs) if i not in newest and i not in spared]
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
    start = request_start(msgs)
    was, msgs = msgs, _shorten_text(msgs, start, 300)
    if r := done("older conversation text", was):
        return r
    if start > 0:
        msgs = msgs[start:]
        if r := done("earlier requests"):
            return r
    # 6. tools this run hasn't used (the ones it has, and essential ones, stay); refilled in finish()
    if specs:
        used = _used_tools(msgs)
        kept = [s for s in specs if s.name in used or _essential(s)]
        if len(kept) < len(specs):
            cut_from, specs = specs, kept
            if r := done("tools not in use"):
                return r
    # 7. what was kept whole: the looks behind newer steps first, then the newest steps
    refill_ok = False
    newest, spared = _protected(msgs)  # (positions moved with the earlier requests)
    for what, which in (("observations behind newer steps", spared), ("latest results", newest)):
        if not which:
            continue
        was = msgs
        lim = squeeze(which)
        if any(a is not b for a, b in zip(was, msgs, strict=True)):
            trimmed.append(f"{what} to {lim if lim is not None else _MIN_RESULT} chars")
        if lim is not None:
            return finish()
    if out > _MIN_OUTPUT:
        out = _MIN_OUTPUT
        if r := done(f"answer room to {out}"):
            return r
    raise RequestTooLarge(
        f"{name}: request too large — ~{prompt() + out} tokens even trimmed, limit {budget}"
    )
