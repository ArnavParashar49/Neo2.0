"""The context diet: what an agent's request carries of the task so far, for every brain.

Tokens are money on a paid brain and time on every brain. The chess log (NEO_BRAIN=cerebras_vision):
one task of 18 tool calls sent 466K input tokens to qwen and spent Cerebras' $0.50 day in four
minutes. The agent loop and the session already shorten a look a newer one replaced within the
request being worked on; the rest — an earlier request's looks and its last picture, every older
step's result at full length — went to a large brain as the history held it: fit
(neo2.core.providers.fit) only ever shrank a request to fit a *small* one. Now every agent request
(one that offers tools) keeps whole what the model needs:

* every word the user and NEO said: the user's request, every note carrying it on ("[User
  confirmed …]"), a text the user pasted, a draft NEO wrote;
* the task being worked on (:func:`task_start`: the request, and the one it answers when its
  first words answer a question a step asked — "which one?" → "the second one"): every result
  it got — the emails, files and pages it read, a listing it works through — but a look a newer
  look at the same thing replaced;
* the newest look at each thing: the latest screenshot of a display (or region), the latest
  ax_tree of each app, the latest read of each page or file — and, in the task, of each search,
  article or inbox read;
* the last two steps, every question a step asked the user (a halted step) and every error;
* the user's latest picture (until they send a newer one) and the request's own, and the newest
  picture of each view: the desktop's and NEO's browser's.

and leaves out the rest:

1. pictures: each older one becomes a one-line note;
2. a look at something a newer look at the same thing replaced (the board before the last move):
   a one-line note that names it;
3. an earlier request's other results: their first and last few hundred characters.

fit then holds an agent request to a soft budget (:func:`soft_budget`) by cutting an earlier
request's results shorter still (:func:`cap`) — never the task's, never anything anyone said.

Pure (never mutates the caller's messages: they are the session's history) and idempotent (a
dieted transcript goes through unchanged: a provider's retry fits the same messages again).
"""

from __future__ import annotations

import json
import re
from dataclasses import replace

from neo2.core.providers.base import Message, ToolCall, ToolResult
from neo2.core.providers.history import OBSERVATIONS, halted, request_start

KEEP_STEPS = 2  # the newest steps (tool turns with a real result) stay whole
OLD_RESULT_CHARS = 400  # what an earlier request's result keeps (head and tail)
SCREEN_BUDGET = 12_000  # prompt tokens an agent request working on the screen aims under
TASK_BUDGET = 24_000  # …and any other agent request (pages and files read whole)
# Tools that look at the screen: a request that used one is a screen task.
_SCREEN = frozenset({"ax_tree", "ax_find", "screenshot", "apps_running", "browser_screenshot"})
# Reads of what the task is about (a search, an article, the inbox): within the task, the newest
# read of each thing is a look too — a newer read of the same thing replaces it.
_READS = frozenset({"web_search", "web_fetch", "mail_unread", "calendar_today", "weather", "clock", "memory"})
# A step that stopped to ask the user: the user's next words answer it (see task_start).
_ASKS = ("NEEDS_USER", "NEEDS_CONFIRM")

_OUT = "left out of this request"  # in every note and cut the diet writes (so it never writes twice)
_CUT = "chars of an older step " + _OUT
_PICTURE = f" [older picture {_OUT}: a newer one follows]"
_USER_PICTURE = f" [older picture {_OUT}: the user sent a newer one]"
_CUT_AT = re.compile(r" …\[(\d+) " + re.escape(_CUT) + r"\]… ")
_SHOT = re.compile(r"^Screenshot (\S+) attached")
_APP = re.compile(r'^(?:<untrusted[^>\n]*>\s*)?App: (.+?)(?: — window "|\s*$)', re.M)
_URL = re.compile(r"^URL: (\S+)$", re.M)
_FIND = re.compile(r"^\(only what mentions “(.*)”\)$", re.M)


def task_start(msgs: list[Message]) -> int:
    """Where the task being worked on begins: the request being worked on (history.request_start),
    or — when that request's first words answer a question a step stopped to ask (the result just
    before them is a NEEDS_USER or NEEDS_CONFIRM: "which one?" → "the second one") — the request
    that asked, and so on back."""
    start = request_start(msgs)
    while start > 0:
        j = start - 1
        while j >= 0 and msgs[j].role == "assistant" and not msgs[j].tool_calls:
            j -= 1  # (the question, had it been written out as a turn of its own)
        if j < 0 or msgs[j].role != "tool" or not any(r.content.lstrip().startswith(_ASKS) for r in msgs[j].tool_results):
            break
        start = request_start(msgs[:j])
    return start


def _error(r: ToolResult) -> bool:
    return not r.ok or r.content.lstrip().startswith(("Error", "error:", "Failed", "failed:"))


def _mine(r: ToolResult) -> bool:
    """A look the diet already turned into its note, or cut: no longer a look at anything."""
    c = r.content
    return (c.startswith("[older ") and c.endswith("follows]") and _OUT in c) or _CUT in c


def _look(r: ToolResult, ours: bool = False) -> bool:
    """A look at something as it is now (see history.OBSERVATIONS) that worked — or, in the task
    being worked on (`ours`), a read of what it is about."""
    return (r.name in OBSERVATIONS or (ours and r.name in _READS)) and not halted(r) and not _error(r)


def view(r: ToolResult, call: ToolCall | None) -> str:
    """What a look is a look at: two looks with the same view are of the same thing, and only the
    newer one is still true. The app an ax_tree read (its first line names it), the display or
    region a screenshot took, the page a browser read (its URL), else the call's own arguments."""
    args = call.args if call is not None else {}
    head = r.content[:2000]
    if r.name == "ax_tree":
        m = _APP.search(head[:400])
        app = m.group(1) if m else str(args.get("app") or "")
        return f"ax_tree|{app.strip().casefold()}"
    if r.name == "screenshot":
        if all(args.get(k) not in (None, "") for k in ("x", "y", "w", "h")):
            return "screenshot|region " + ",".join(str(args[k]) for k in ("x", "y", "w", "h"))
        return f"screenshot|display {args.get('display') or 1}"
    if (u := _URL.search(head)) is not None:
        f = _FIND.search(head)
        kind = "page-picture" if r.name == "browser_screenshot" else "page"
        return f"{kind}|{u.group(1)}" + (f"|{f.group(1)}" if f else "")
    return f"{r.name}|{json.dumps(args, sort_keys=True, default=str)}"


def _calls(msgs: list[Message]) -> dict[tuple[int, int], ToolCall]:
    """(tool turn, result) → the call it answers: the call turn just before, by id, else by place."""
    out: dict[tuple[int, int], ToolCall] = {}
    calls: list[ToolCall] = []
    for i, m in enumerate(msgs):
        if m.role == "assistant":
            calls = list(m.tool_calls)
        elif m.role == "tool":
            by_id = {c.id: c for c in calls}
            for j, r in enumerate(m.tool_results):
                c = by_id.get(r.call_id) or (calls[j] if j < len(calls) else None)
                if c is not None:
                    out[(i, j)] = c
        else:
            calls = []
    return out


def _recent(msgs: list[Message]) -> set[int]:
    real = [i for i, m in enumerate(msgs) if m.role == "tool" and not all(halted(r) for r in m.tool_results)]
    return set(real[-KEEP_STEPS:])


def _sorted(msgs: list[Message], start: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """(the results kept whole wherever they are, the looks a newer look at the same thing
    replaced), as (tool turn, result); `start` is the task's (see task_start). Kept whole: the last
    steps', the newest look at each thing, every question to the user and every error. Replaced:
    any other look (the last steps' stay whole all the same)."""
    recent, calls = _recent(msgs), _calls(msgs)
    keep: set[tuple[int, int]] = set()
    stale: set[tuple[int, int]] = set()
    seen: set[str] = set()
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if m.role != "tool":
            continue
        for j in range(len(m.tool_results) - 1, -1, -1):
            r = m.tool_results[j]
            if i in recent or halted(r) or _error(r):
                keep.add((i, j))
            if _look(r, i > start) and not _mine(r):
                v = view(r, calls.get((i, j)))
                if v not in seen:
                    keep.add((i, j))
                elif i not in recent:
                    stale.add((i, j))
                seen.add(v)
    return keep, stale


def _pictures(msgs: list[Message], stale: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """The results whose picture stays: the newest of each view, of a look still true — the
    desktop's (a screenshot) and NEO's browser's. At most two."""
    out: set[tuple[int, int]] = set()
    seen: set[bool] = set()
    for i in range(len(msgs) - 1, -1, -1):
        for j in range(len(msgs[i].tool_results) - 1, -1, -1):
            r = msgs[i].tool_results[j]
            if r.images and (i, j) not in stale and (browser := r.name.startswith("browser_")) not in seen:
                out.add((i, j))
                seen.add(browser)
    return out


def _cut(text: str, limit: int) -> str:
    """The head and the tail of `text` (a fenced result keeps its closing tag), `limit` chars. A
    text cut before is cut again only when `limit` is shorter (its count says all that went)."""
    m = _CUT_AT.search(text)
    gone, body = (int(m.group(1)), text[: m.start()] + " " + text[m.end() :]) if m else (0, text)
    if len(body) <= limit + (1 if m else 0):
        return text
    head = int(limit * 0.7)
    return f"{body[:head].rstrip()} …[{gone + len(body) - limit} {_CUT}]… {body[-(limit - head):].lstrip()}"


def _stale(r: ToolResult) -> str:
    """The one line a look a newer one replaced becomes (its shot id stays: calls name it)."""
    m = _SHOT.match(r.content)
    what = f"screenshot {m.group(1)}" if m else r.name
    return f"[older {what}: {_OUT} — a newer look at the same thing follows]"


def diet(messages: list[Message]) -> list[Message]:
    """`messages` as an agent request sends them (see the module docstring). Never mutates them."""
    msgs = list(messages)
    turns = [i for i, m in enumerate(msgs) if m.role == "tool"]
    if not turns and not any(m.images for m in msgs):
        return msgs
    start = task_start(msgs)
    keep, stale = _sorted(msgs, start)
    pictures = _pictures(msgs, stale)  # (each keeps its last image)
    # the user's pictures that stay: the request's own, and their latest until they send a newer one
    users = [i for i, m in enumerate(msgs) if m.role == "user" and m.images]
    theirs = {request_start(msgs), *users[-1:]}
    for i, m in enumerate(msgs):
        if m.role == "user" and m.images and i not in theirs:
            msgs[i] = replace(m, text=m.text + _USER_PICTURE, images=[])
        if m.role != "tool":
            continue
        results = list(m.tool_results)
        for j, r in enumerate(results):
            images = r.images[-1:] if (i, j) in pictures else []
            note = _PICTURE if len(images) < len(r.images) else ""
            if (i, j) in stale:
                content = _stale(r)
            elif (i, j) in keep or i > start:  # (everything the task got that no newer look replaced)
                content = r.content + note
            else:
                content = _cut(r.content, OLD_RESULT_CHARS) + note
            if content != r.content or len(images) != len(r.images):
                results[j] = replace(r, content=content, images=images)
        if any(a is not b for a, b in zip(results, m.tool_results, strict=True)):
            msgs[i] = replace(m, tool_results=results)
    return msgs


def cap(msgs: list[Message], limit: int) -> list[Message]:
    """An earlier request's results cut to `limit` chars, its newest looks too (the soft budget's
    shedding: the screen they showed has moved on since). What the task being worked on got, the
    last steps, every question and every error stay whole."""
    start, recent = task_start(msgs), _recent(msgs)
    out = list(msgs)
    for i, m in enumerate(out):
        if m.role != "tool" or i > start or i in recent:
            continue
        results = [
            r if halted(r) or _error(r) or len(r.content) <= limit else replace(r, content=_cut(r.content, limit))
            for r in m.tool_results
        ]
        if any(a is not b for a, b in zip(results, m.tool_results, strict=True)):
            out[i] = replace(m, tool_results=results)
    return out


def soft_budget(msgs: list[Message]) -> int:
    """Prompt tokens an agent request aims under, whatever the brain's own room: less for a task on
    the screen (its looks are big and go stale fast) than for one reading pages and files."""
    start = task_start(msgs) if msgs else 0
    screen = any(
        m.images or any(r.images or r.name in _SCREEN for r in m.tool_results) for m in msgs[start:]
    )
    return SCREEN_BUDGET if screen else TASK_BUDGET
