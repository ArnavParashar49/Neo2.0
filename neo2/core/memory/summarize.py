"""Conversation summaries: a few lines about what happened, written from the journal.

One summary per conversation (neo2.core.memory.store: a run of journal lines less than ten minutes
apart — a few "Neo…" in a row, a typed exchange), written in the background once the conversation
has gone quiet, and written again if it went on afterwards. They are what "what did we do
yesterday" is answered from, and they are searchable like everything else in memory.

Because they are written from the journal — which is on disk within a quarter of a second of
each line — nothing depends on NEO shutting down cleanly: a conversation that has no summary yet
(NEO was killed, the Mac lost power) gets one at the next start (:func:`catch_up`). Shutdown
itself only makes a short best effort.

What the model sees: the user's lines, NEO's own lines (short), and which tools ran — never a line
of NEO's that quoted a page, a mail or a file. With ``cloud_memory = off`` no model is asked: the
summary is the crude local one ("Asked about: …").

:func:`summarize_session` is the older way, from a Session's in-RAM history: an agent that keeps
no journal (an embedded one, by default) still uses it.
"""

from __future__ import annotations

import asyncio
import time

from neo2.core import logger
from neo2.core.config import settings
from neo2.core.memory.store import Turn, store
from neo2.core.providers import brain
from neo2.core.providers.base import Message

log = logger(__name__)

IDLE_S = 180.0  # a conversation quiet for this long is summarised (again, if it goes on later)
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


def _from_journal(turns: list[Turn], max_chars: int = 12000) -> str:
    lines = []
    for t in turns:
        if t.role == "user":
            lines.append(f"User: {t.text}")
        elif t.source == "tool":
            lines.append("NEO used: " + t.text.removeprefix("(ran: ").rstrip(")"))
        elif t.tainted:
            lines.append("NEO: (read out what a page, a mail or a file said)")
        else:
            lines.append(f"NEO: {t.text[:400]}")
    return "\n".join(lines)[-max_chars:]


def _crude(asked: list[str]) -> str:
    return "Asked about: " + "; ".join(a[:60] for a in asked[:4]) if asked else ""


def _use_model() -> bool:
    """A summary's transcript goes to a model only when memory may go to one at all."""
    return settings().cloud_memory != "off"


async def _summary_of(text: str, asked: list[str], budget_s: float, *, model: bool = True) -> str:
    if not model or not _use_model():
        return _crude(asked)
    try:
        turn = await asyncio.wait_for(
            brain("fast").complete([Message.user(text)], system=_PROMPT, effort="low", max_tokens=200), budget_s
        )
        return turn.text.strip()
    except Exception as e:  # noqa: BLE001 — no brain right now? keep a crude fallback
        log.warning(f"[memory] summary fallback ({str(e)[:60]})")
        return _crude(asked)


async def summarize_conversation(conv: int, *, budget_s: float = 8.0, model: bool = True) -> str:
    """Write (or rewrite) the summary of one journal conversation; returns it ("" when there was
    nothing to say). An erase while the model was writing drops it."""
    from neo2.core import privacy

    gen, began = privacy.generation(), time.monotonic()
    st = store()
    turns = await asyncio.to_thread(st.turns_of, conv)
    if not turns:
        return ""
    asked = [t.text for t in turns if t.role == "user"]
    upto = max(t.id for t in turns)
    summary = await _summary_of(_from_journal(turns), asked, budget_s, model=model) if asked else ""
    if privacy.generation() != gen:
        return ""  # the user erased NEO's memory while this was being written: it must not come back
    await asyncio.to_thread(lambda: st.set_summary(conv, summary, upto, since=began))
    return summary


async def summarize_due(*, idle_s: float = IDLE_S, budget_s: float = 8.0, limit: int = 4, model: bool = True) -> int:
    """Summarise the conversations that have gone quiet for `idle_s` and have lines their
    summary doesn't cover. Returns how many were written."""
    s = settings()
    if not (s.memory_enabled and s.summaries and s.memory_journal):
        return 0
    st = store()
    due = await asyncio.to_thread(st.unsummarized, idle_s=idle_s)
    n = 0
    for conv, _started, _last in due[:limit]:
        try:
            n += bool(await summarize_conversation(conv, budget_s=budget_s, model=model))
        except Exception as e:  # noqa: BLE001 — a summary is a nicety, never a failure
            log.warning(f"[memory] summary skipped: {str(e)[:80]}")
    return n


async def catch_up(*, budget_s: float = 8.0) -> int:
    """At start: every conversation an earlier run left without a summary (it was killed, or
    quit mid-conversation) gets one now. They are over — nothing waits for them to go quiet."""
    try:
        return await summarize_due(idle_s=0.0, budget_s=budget_s, limit=12)
    except Exception as e:  # noqa: BLE001
        log.warning(f"[memory] catch-up skipped: {str(e)[:80]}")
        return 0


async def summarize_session(history: list[Message], started: float, *, budget_s: float = 5.0) -> str:
    """Write a summary for this session into the store; returns it (empty if nothing to say).
    The model gets `budget_s`; slower than that (shutdown can't wait) → the crude summary.
    (From a Session's own history: for an agent that keeps no journal.)"""
    from neo2.core import privacy

    gen, began = privacy.generation(), time.monotonic()
    text = _transcript(history)
    if text.count("User:") < 1:
        return ""
    firsts = [m.text for m in history if m.role == "user" and m.text][:4]
    summary = await _summary_of(text, firsts, budget_s)
    if summary and privacy.generation() != gen:
        return ""  # the user erased NEO's memory while this was being written: it must not come back
    if summary:
        store().add_session_summary(summary, started, time.time(), since=began)
    return summary
