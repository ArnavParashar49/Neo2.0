"""A transcript every brain accepts.

Gemini, the OpenAI-compatible servers and Claude all insist on the same shape around tools: the
conversation opens with a user turn, an assistant turn that calls tools is answered at once by one
turn holding exactly those calls' results, and no result stands without its call. Gemini says it
as a 400 ("function response turn comes immediately after a function call turn"), Claude as
"tool_result without tool_use", OpenAI as "tool_calls must be followed by tool messages".

NEO's history can break that shape: a window over a long session that begins halfway through an
agent run (its first message a call whose result, or a result whose call, fell outside), a run
cut short between a call and its result, a step answered twice. :func:`normalize` repairs every
case the same way for every provider, deterministically and without touching the caller's
messages:

* whatever comes before the first user turn is dropped (a leading call or result lost its pair);
* each call turn is followed by one result turn, results in the calls' order; several result
  turns in a row are merged into it;
* a call nobody answered gets a stand-in failed result; a result no call asked for (or a second
  one for the same call) is dropped;
* an assistant turn with nothing in it is dropped.
"""

from __future__ import annotations

import json

from neo.providers.base import Message, ToolCall, ToolResult

NO_RESULT = "(no result: this step was cut short)"


def normalize(messages: list[Message]) -> list[Message]:
    start = next((i for i, m in enumerate(messages) if m.role == "user"), len(messages))
    out: list[Message] = []
    pending: Message | None = None  # the call turn whose results are being gathered
    slots: dict[str, list[int]] = {}  # call id → positions in pending.tool_calls (ids can repeat)
    got: dict[int, ToolResult] = {}

    def settle() -> None:
        nonlocal pending, slots, got
        if pending is None:
            return
        results = []
        for i, c in enumerate(pending.tool_calls):
            r = got.get(i)
            if r is None:
                r = ToolResult(c.id, c.name, NO_RESULT, ok=False)
            elif (r.call_id, r.name) != (c.id, c.name):
                r = ToolResult(c.id, c.name, r.content, r.ok, list(r.images))
            results.append(r)
        out.append(Message.tool(results))
        pending, slots, got = None, {}, {}

    for m in messages[start:]:
        if m.role == "tool":
            if pending is None:
                continue  # its call isn't in this transcript: nothing can pair with it
            for r in m.tool_results:
                free = [i for i in slots.get(r.call_id, []) if i not in got]
                if free:
                    got[free[0]] = r
            continue
        settle()
        if m.role == "assistant":
            if not m.text and not m.tool_calls and m.raw is None:
                continue
            if m.tool_calls:
                for i, c in enumerate(m.tool_calls):  # results name the ids the model gave
                    slots.setdefault(c.id, []).append(i)
                m = pending = _unique_ids(m)
        out.append(m)
    settle()
    return out


def _unique_ids(m: Message) -> Message:
    """Calls in one turn must have distinct ids for their results to be told apart (a local server
    may number every call "call_0"). A repeat gets a suffix; its result is matched by order."""
    ids = [c.id for c in m.tool_calls]
    if len(set(ids)) == len(ids) and all(ids):
        return m
    seen: dict[str, int] = {}
    calls = []
    for c in m.tool_calls:
        n = seen[c.id] = seen.get(c.id, 0) + 1
        new_id = c.id if n == 1 and c.id else f"{c.id or 'call'}~{n}"
        calls.append(ToolCall(new_id, c.name, c.args, c.signature, c.source))
    return Message(role="assistant", text=m.text, images=m.images, tool_calls=calls, raw=m.raw)


def continues_request(m: Message) -> bool:
    """A user turn NEO wrote to carry a request on ("[User confirmed. …]", "[verify] …"), not
    one that starts a request in the user's own words."""
    return m.role == "user" and m.text.lstrip().startswith("[")


def flatten(messages: list[Message]) -> list[Message]:
    """The same conversation with no tool turns at all: each call and its result written out as
    text. The last resort for a brain that rejects the tool turns themselves with a 400 nothing
    above foresaw — the task goes on instead of dying on the history's shape."""
    out: list[Message] = []
    for m in normalize(messages):
        if m.role == "assistant" and m.tool_calls:
            calls = "; ".join(
                f"{c.name}({json.dumps(c.args, ensure_ascii=False, default=str)[:400]})" for c in m.tool_calls
            )
            out.append(Message.assistant((m.text + "\n" if m.text else "") + f"[I called: {calls}]"))
        elif m.role == "tool":
            body = "\n".join(
                f"[{r.name} {'result' if r.ok else 'failed'}]: {r.content}" for r in m.tool_results
            )
            out.append(Message.user(body, images=[im for r in m.tool_results for im in r.images]))
        elif m.role == "assistant":
            out.append(Message.assistant(m.text))
        else:
            out.append(m)
    return out
