"""The model's memory tool: remember / recall / forget, backed by the local SQLite store."""

from __future__ import annotations

import asyncio

from neo.agent.registry import ToolContext, tool
from neo.memory.store import store


@tool(
    "memory",
    "Long-term memory. action=remember stores a fact (kind: profile for stable facts about the user, "
    "lesson for feedback on how to behave, note for anything else; key optional e.g. 'city'). "
    "action=recall searches. action=forget deletes by id. Save preferences and corrections silently.",
    {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["remember", "recall", "forget"]},
            "text": {"type": "string"},
            "kind": {"type": "string", "enum": ["profile", "lesson", "note"]},
            "key": {"type": "string"},
            "query": {"type": "string"},
            "id": {"type": "integer"},
        },
        "required": ["action"],
    },
    parallel_safe=True,
    category="memory",
    chain=False,  # only as the whole utterance: a later clause means 'in that app'
    fast_path=[
        (
            # "remember that I park on level 3" — a real fact of 2+ words; "note that down",
            # "remember that" alone, and "remember to …" (a reminder) are not facts.
            r"^\s*(?:please\s+)?(?:remember|keep\s+in\s+mind|note)\s+(?!to\b)(?:that\s+|this:?\s+)?"
            r"(?!(?:down|it|this|that|them)\b)(?!.*\bdown\s*[.!]*\s*$)"
            r"(?P<text>\S+(?:\s+\S+)+?)\s*[.!]*\s*$",
            {"action": "remember", "text": "<text>", "kind": "note"},
        ),
        (
            r"^\s*(?:what\s+did\s+i\s+(?:say|tell\s+you)\s+about|do\s+you\s+remember(?:\s+anything\s+about)?|"
            r"what\s+do\s+you\s+(?:know|remember)\s+about)\s+(?P<query>.{2,}?)\s*[?.!]*\s*$",
            {"action": "recall", "query": "<query>"},
        ),
    ],
    payload=True,
)
async def memory(a: dict, c: ToolContext) -> str:
    st = store()
    act = a.get("action")
    if act in ("remember", "forget") and c.tainted():
        # What NEO remembers shapes every later answer. After reading a page, an email or the
        # screen, a memory could be planted by that content rather than the user — so they OK it.
        from neo.agent import confirm

        what = f"remember “{str(a.get('text', ''))[:80]}”" if act == "remember" else f"forget memory #{a.get('id')}"
        gate = confirm.gated("memory", a, f"{what} (after reading {c.tainted()})", detail=str(a.get("text", "")))
        if isinstance(gate, str):
            return gate
    # Off the event loop: an embedding and a database write can take a moment.
    if act == "remember":
        if not a.get("text"):
            return "Error: text required"
        mid = await asyncio.to_thread(st.remember, a["text"], a.get("kind", "note"), a.get("key", ""))
        return f"Remembered (#{mid})."
    if act == "recall":
        hits = await asyncio.to_thread(st.recall, a.get("query") or c.user_text, 8)
        return (
            "\n".join(f"#{m.id} [{m.kind}{'/' + m.key if m.key else ''}] {m.text}" for m in hits)
            or "Nothing relevant in memory."
        )
    if act == "forget":
        gone = await asyncio.to_thread(st.forget, int(a.get("id") or 0))
        return "Forgotten." if gone else "Error: no such memory."
    return "Error: unknown action"
