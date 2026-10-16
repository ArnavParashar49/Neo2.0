"""When a request needs a model and NEO has none: what NEO says, and opening Brains & Keys.

The agent's run ends on NoBrain (neo2.core.providers: no key for any provider NEO may ask, no model on
this Mac answering). The session then asks :func:`no_brain` for its words — said through whichever
voice runs (the on-device one when there's no Gemini key), the same words on screen — and with them
NEO.app's core opens the Brains & Keys window: only for a request the user just made, and at most
once every 10 minutes (the App's opener keeps that: neo2.core.app). With no NEO.app (a core started
by hand, the text REPL) the screen shows the terminal command, and the voice says to look at it.

Every client also hears a ``setup_needed`` event ({shell, opened, command}): the chat page and the
notch show their "Give NEO a brain" card from it as from the hello's brains.ready.
"""

from __future__ import annotations

from collections.abc import Callable

from neo2.core import logger

log = logger(__name__)

COMMAND = "python -m neo2 keys set gemini"
_WHAT = "a free Gemini key takes about a minute, or you can use Claude, OpenAI or another provider."
OPENED = f"I need an AI model for that. I've opened Brains and Keys — {_WHAT}"
OPEN_IT = f"I need an AI model for that. Open Brains and Keys to connect one — {_WHAT}"
TERMINAL = f"I need an AI model for that. In a terminal, run: {COMMAND}"
TERMINAL_SPOKEN = "I need an AI model for that — see the message on screen."

KEY_TYPED = "That looks like an API key, so I didn't send it anywhere. Save it in Brains & Keys."

# The App's: `fn(open_now)` opens the window for a request that needs a model and says how it went —
# "opened", "busy" (not now: opened in the last 10 minutes, or `open_now` False) or "no_app" (no
# NEO.app at the other end). None: nothing can open it.
_opener: Callable[[bool], str] | None = None


def set_opener(fn: Callable[[bool], str] | None) -> None:
    global _opener
    _opener = fn


def _open(user: bool) -> str:
    """Ask the App to open the window ("opened" | "busy" | "no_app"). Not the user's own request: never
    opened, only whether a window could be."""
    fn = _opener
    if fn is None:
        return "no_app"
    try:
        got = fn(user)
    except Exception as e:  # noqa: BLE001
        log.info(f"[keys] Brains & Keys didn't open: {type(e).__name__}")
        return "no_app"
    return got if got in ("opened", "busy", "no_app") else "no_app"


async def no_brain(*, user: bool) -> tuple[str, str]:
    """(the words on screen, the words said) for a request no brain could answer because none is set
    up. `user`: the user's own request, just made (only then may the window open)."""
    from neo2.core.events import bus

    got = _open(user)
    if got == "opened":
        shown, said = OPENED, OPENED
    elif got == "busy":
        shown, said = OPEN_IT, OPEN_IT
    else:
        shown, said = TERMINAL, TERMINAL_SPOKEN
    try:
        await bus().publish("setup_needed", shell=got != "no_app", opened=got == "opened", command=COMMAND)
    except Exception:  # noqa: BLE001 — the words are what matters
        pass
    return shown, said
