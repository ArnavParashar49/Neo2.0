"""Cerebras as the rest of NEO calls on it (decision 49): whether it takes part, where it sits in a
chain, what it has to tell the user with a reply, and warming its connection up.

Apart from neo2.core.providers.cerebras (the provider itself, which imports the openai SDK) so that
building a chain, finishing a reply or hearing the wake word never imports a vendor SDK: without a
key nothing here touches it, and an embedded agent's first request stays light (the SDK's promise).
"""

from __future__ import annotations

import asyncio
import threading
import weakref
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from neo2.core import logger
from neo2.core.providers import spend

log = logger(__name__)

TEXT_BRAIN = "cerebras"
VISION_BRAIN = "cerebras_vision"
DEFAULT_BUDGET = 0.50  # dollars a day

# Each agent's Cerebras account (neo2.core.providers.cerebras.Account), by its BrainPool: made there on
# first use, read here for what it has to say.
_accounts: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()
_accounts_lock = threading.Lock()


def enabled(s: Any = None) -> bool:
    """Cerebras takes part: a key is set and the daily budget isn't 0 (the user's off switch).
    Otherwise no chain names it, so nothing the user sees or hears changes. `s`: the settings,
    default the bound agent's (none bound in library mode: no)."""
    try:
        if s is None:
            from neo2.core.config import settings

            s = settings()
        key = str(getattr(s, "cerebras_api_key", "") or "").strip()
        budget = spend.budget(getattr(s, "cerebras_daily_budget", DEFAULT_BUDGET), DEFAULT_BUDGET)
    except Exception:  # noqa: BLE001 — no agent bound in library mode: no settings, no Cerebras
        return False
    return bool(key) and budget > 0


# ---- where the brains go in a chain ----------------------------------------------------------

_SEEING = frozenset({"gemini", "gemini_lite", "claude"})


def _before_groq_vision(names: list[Any]) -> int:
    """Where the seeing brain goes: right ahead of Groq's vision brain, else right after the
    leading seeing brains."""
    if "groq_vision" in names:
        return names.index("groq_vision")
    at = 0
    while at < len(names) and names[at] in _SEEING:
        at += 1
    return at


def with_cerebras(order: Sequence[str | None], purpose: str, s: Any = None) -> list[str | None]:
    """A brain() purpose's order with Cerebras placed as decision 49 says: the text brain right
    ahead of Groq (no Groq: ahead of the offline brain, else last); for "vision" also the seeing
    one right ahead of Groq's vision brain (none: right after the seeing brains). "offline" is
    left alone, and so is an order that already names them — and every order when Cerebras
    isn't :func:`enabled` (`s`: the settings, default the bound agent's)."""
    out = list(order)
    if purpose == "offline" or not enabled(s):
        return out
    if purpose == "vision" and VISION_BRAIN not in out:
        out.insert(_before_groq_vision(out), VISION_BRAIN)
    if TEXT_BRAIN not in out:
        if "groq" in out:
            out.insert(out.index("groq"), TEXT_BRAIN)
        elif "local" in out:
            out.insert(out.index("local"), TEXT_BRAIN)
        else:
            out.append(TEXT_BRAIN)
    return out


def with_cerebras_eyes(eyes: Sequence[str], s: Any = None) -> tuple[str, ...]:
    """The seeing fallbacks a step working from a picture may ask (an agent chain's `eyes`), with
    Cerebras' seeing brain right ahead of Groq's vision brain (decision 49: when Gemini is
    overloaded) — when Cerebras is :func:`enabled`; else `eyes` as they are."""
    out = list(eyes)
    if VISION_BRAIN not in out and enabled(s):
        out.insert(_before_groq_vision(out), VISION_BRAIN)
    return tuple(out)


# ---- what the user hears ---------------------------------------------------------------------


def spoken_notes() -> str:
    """What Cerebras has to say out loud since the last call — the 80% warning, the skip once the
    budget is spent, a refused key or account (each once a day) — as sentences, or "". Drained:
    the bound agent's own account, never another's; nothing is made to answer."""
    try:
        from neo2.core.providers import pool

        owner = pool()
    except Exception:  # noqa: BLE001 — no agent bound in library mode: nobody to tell
        return ""
    with _accounts_lock:
        a = _accounts.get(owner)
    if a is None:
        return ""
    return " ".join(t if t.endswith((".", "!", "?")) else t + "." for t in a.take_heard())


def with_spoken_notes(reply: str, notes: str, *, question: bool = False) -> str:
    """`reply` with `notes` (:func:`spoken_notes`) said too: after it, or before it when the reply
    is a question (it stays the last thing the user hears)."""
    if not notes:
        return reply
    if not reply:
        return notes
    return f"{notes} {reply}" if question else f"{reply} {notes}"


def add_spoken_notes(reply: Any) -> Any:
    """The session's Reply (neo2.core.agent.session: text, route, silent) with :func:`spoken_notes` in
    what the user hears: after its words, before them when it asks a question ("confirm"); a
    quiet reply (nothing to say) says just the notes. The same Reply when there are none."""
    notes = spoken_notes()
    if not notes:
        return reply
    silent = bool(getattr(reply, "silent", False))
    text = with_spoken_notes(
        "" if silent else reply.text, notes, question=getattr(reply, "route", "") == "confirm"
    )
    return replace(reply, text=text, silent=False)


# ---- warming up ------------------------------------------------------------------------------

_warming: set[asyncio.Task] = set()


def prewarm_soon() -> None:
    """Open the connection to Cerebras in the background (neo2.core.providers.cerebras.prewarm): the
    caller — the wake word, the start of the user's speech or typing — goes on at once. Nothing
    when Cerebras isn't :func:`enabled` or there's no running event loop; never raises."""
    try:
        if not enabled():
            return
        loop = asyncio.get_running_loop()
    except Exception:  # noqa: BLE001 — no loop, no agent bound (library mode): nothing to warm
        return
    t = loop.create_task(_prewarm())
    _warming.add(t)  # (a task nobody holds can be collected mid-flight)
    t.add_done_callback(_warming.discard)


async def _prewarm() -> bool:
    from neo2.core.providers.cerebras import prewarm

    return await prewarm()
