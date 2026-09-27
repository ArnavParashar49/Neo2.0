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
# Cloud brains that see, as a user may name them (NEO_BRAIN, NEO_FAST_BRAIN, NEO_OFFLINE_BRAIN).
_SEES_IN_CLOUD = _SEEING | {"groq_vision", VISION_BRAIN}
# Brains on this Mac: what the user says to one, and what it sees, stays here.
_ON_THIS_MAC = frozenset({"local"})


def _before_groq_vision(names: list[Any]) -> int:
    """Where the seeing brain goes: right ahead of Groq's vision brain, else right after the
    leading seeing brains."""
    if "groq_vision" in names:
        return names.index("groq_vision")
    at = 0
    while at < len(names) and names[at] in _SEEING:
        at += 1
    return at


def _named(s: Any) -> tuple[Any, ...]:
    """The brains the user named: NEO_BRAIN, NEO_FAST_BRAIN, NEO_OFFLINE_BRAIN."""
    return tuple(getattr(s, k, None) for k in ("brain", "fast_brain", "offline_brain"))


def _text_brain_at(names: list[Any], s: Any) -> int | None:
    """Where the text brain goes: right ahead of Groq. No Groq: right ahead of the first brain on
    this Mac that comes after a cloud brain the user named (their last resort, as for the seeing
    fallbacks), else last. None when the user named only brains on this Mac: nothing they said
    ever left it, and neither a key nor an order's own cloud brains ("light"'s Flash-Lite,
    "vision"'s seeing ones) change that — NEO_BRAIN or NEO_FAST_BRAIN=cerebras does."""
    if "groq" in names:
        return names.index("groq")
    theirs = {n for n in _named(s) if n is not None and n not in _ON_THIS_MAC}
    if not theirs:
        return None
    after = False
    for i, n in enumerate(names):
        if n in theirs:
            after = True
        elif after and n in _ON_THIS_MAC:
            return i
    return len(names)


def _cloud_may_see(s: Any) -> bool:
    """May a cloud fallback see this user's pictures? Yes — except for a user who put a brain on
    this Mac first and named no cloud brain that sees: on main no picture of theirs ever left the
    Mac (a pixel step with no seeing brain stops and says so, DECISIONS 39), unless they opt in
    (NEO_CLOUD_EYES=on, brain-effort's setting). The same rule as brain-effort's _fallback_eyes."""
    if str(getattr(s, "cloud_eyes", "auto") or "auto") == "on":
        return True
    named = _named(s)
    return named[0] not in _ON_THIS_MAC or any(n in _SEES_IN_CLOUD for n in named)


def _bound(s: Any) -> Any:
    """`s`, else the bound agent's settings (None in library mode with no agent bound)."""
    if s is not None:
        return s
    try:
        from neo2.core.config import settings

        return settings()
    except Exception:  # noqa: BLE001 — no agent bound in library mode: no settings, no Cerebras
        return None


def _theirs(s: Any) -> bool:
    """The user named Cerebras' text brain their brain (NEO_BRAIN=cerebras: gpt-oss): its seeing
    model is theirs too — the first a step working from a picture asks, ahead of the free seeing
    fallbacks, as a brain the user named that sees would be."""
    return getattr(s, "brain", None) == TEXT_BRAIN


def with_cerebras(order: Sequence[str | None], purpose: str, s: Any = None) -> list[str | None]:
    """A brain() purpose's order with Cerebras placed as decision 49 says: the text brain right
    ahead of Groq, and never ahead of a brain on this Mac the user put first (:func:`_text_brain_at`);
    for "vision" also the seeing one right ahead of Groq's vision brain (none: right after the
    seeing brains) — where a cloud brain may see this user's pictures (:func:`_cloud_may_see`).
    With NEO_BRAIN=cerebras (:func:`_theirs`) the "vision" order starts with the two: a screen task's
    text steps go to the text brain, its pictures to the seeing one (the chain asks a brain that
    sees first for a picture). "offline" is left alone, and so is an order that already names them
    — and every order when Cerebras isn't :func:`enabled` (`s`: the settings, default the bound
    agent's)."""
    out = list(order)
    s = _bound(s)
    if purpose == "offline" or s is None or not enabled(s):
        return out
    if purpose == "vision" and VISION_BRAIN not in out and _cloud_may_see(s):
        out.insert(_before_groq_vision(out), VISION_BRAIN)
    if TEXT_BRAIN not in out and (at := _text_brain_at(out, s)) is not None:
        out.insert(at, TEXT_BRAIN)
    if purpose == "vision" and _theirs(s):
        head = [n for n in (TEXT_BRAIN, VISION_BRAIN) if n in out]
        out = head + [n for n in out if n not in head]
    return out


def with_cerebras_eyes(eyes: Sequence[str], s: Any = None) -> tuple[str, ...]:
    """The seeing fallbacks a step working from a picture may ask (an agent chain's `eyes`), with
    Cerebras' seeing brain right ahead of Groq's vision brain (decision 49: when Gemini is
    overloaded) — when Cerebras is :func:`enabled` and a cloud brain may see this user's pictures
    (:func:`_cloud_may_see`); else `eyes` as they are. With NEO_BRAIN=cerebras (:func:`_theirs`)
    it is the first of them."""
    out = list(eyes)
    s = _bound(s)
    if VISION_BRAIN not in out and s is not None and enabled(s) and _cloud_may_see(s):
        out.insert(0 if _theirs(s) else _before_groq_vision(out), VISION_BRAIN)
    return tuple(out)


# ---- what the user hears ---------------------------------------------------------------------


def _heard() -> list[str]:
    """The lines :func:`spoken_notes` says, drained from the bound agent's own account."""
    try:
        from neo2.core.providers import pool

        owner = pool()
    except Exception:  # noqa: BLE001 — no agent bound in library mode: nobody to tell
        return []
    with _accounts_lock:
        a = _accounts.get(owner)
    return [] if a is None else list(a.take_heard())


def _sentences(lines: Sequence[str]) -> str:
    return " ".join(t if t.endswith((".", "!", "?")) else t + "." for t in lines)


def spoken_notes() -> str:
    """What Cerebras has to say out loud since the last call — the 80% warning, the skip once the
    budget is spent, a refused key or account (each once a day) — as sentences, or "". Drained:
    the bound agent's own account, never another's; nothing is made to answer."""
    return _sentences(_heard())


def _already_said(line: str, reply: str) -> bool:
    """Does `reply` already say what `line` says? A stop message says a refusal in its own words
    (base.failure_words: "Cerebras' daily budget of $0.50 is used up"), the note the same words
    and then what NEO does about it ("; skipping it until tomorrow")."""
    news = line.split(";", 1)[0].strip().rstrip(".!?").strip().lower()
    return bool(news) and news in reply.lower()


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
    quiet reply (nothing to say) says just the notes. The same Reply when there are none — or
    when the reply already says each of them (a stop message naming the spent budget): drained
    all the same, so the user hears it once."""
    silent = bool(getattr(reply, "silent", False))
    said = "" if silent else str(getattr(reply, "text", "") or "")
    notes = _sentences([t for t in _heard() if not _already_said(t, said)])
    if not notes:
        return reply
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
