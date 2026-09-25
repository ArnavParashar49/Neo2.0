"""Session summaries: a few lines about what happened, written at shutdown (and on demand).

Lets the next session answer "what did we do yesterday" and carry over unfinished business.
"""

from __future__ import annotations

import time

from neo.memory.store import store
from neo.providers import brain
from neo.providers.base import Message

_PROMPT = (
    "Summarise this conversation between a user and their Mac assistant NEO in 2-4 short lines: "
    "what the user asked for, what was done, anything unfinished, and any preference the user expressed. "
    "Plain text, no headings, no bullets, past tense, under 80 words."
)


def _transcript(history: list[Message], max_chars: int = 12000) -> str:
    lines = []
    for m in history:
        if m.role == "user" and m.text and not m.text.startswith("["):
            lines.append(f"User: {m.text}")
        elif m.role == "assistant" and m.text:
            lines.append(f"NEO: {m.text}")
        elif m.role == "assistant" and m.tool_calls:
            lines.append("NEO used: " + ", ".join(c.name for c in m.tool_calls))
    text = "\n".join(lines)
    return text[-max_chars:]


async def summarize_session(history: list[Message], started: float) -> str:
    """Write a summary for this session into the store; returns it (empty if nothing to say)."""
    text = _transcript(history)
    if text.count("User:") < 1:
        return ""
    try:
        turn = await brain("fast").complete(
            [Message.user(text)], system=_PROMPT, effort="low", max_tokens=200
        )
        summary = turn.text.strip()
    except Exception as e:  # noqa: BLE001 — no brain at shutdown? keep a crude fallback
        firsts = [m.text for m in history if m.role == "user" and m.text][:4]
        summary = "Asked about: " + "; ".join(f[:60] for f in firsts) if firsts else ""
        print(f"[memory] summary fallback ({str(e)[:60]})")
    if summary:
        store().add_session_summary(summary, started, time.time())
    return summary
