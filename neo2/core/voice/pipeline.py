"""Picks the voice backend: Gemini Live when the key + quota allow it, else the local cascade.

A failed check at startup is usually the network not being up yet (NEO starts at login) — that
doesn't turn Live off: it connects on the next "Neo". Only a definite "this key can't use
Live" falls back to the on-device voice for good.

The on-device voice is the user's to ask for: its models are a 2.9 GB download (Parakeet and Kokoro),
which NEO never starts by itself. With no Gemini key and those models not on this Mac, there is no
voice yet (:class:`VoiceNotSetUp`): NEO says so and types meanwhile, and Brains & Keys and the chat's
card offer the download (neo2.core.fetch; ``python -m neo2 fetch --local-voice``).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from neo2.core.config import settings

TextHandler = Callable[[str], Awaitable[None]]
_PROBE_TIMEOUT_S = 15.0
_REFUSED = ("not found", "404", "permission", "403", "not supported", "api key not valid", "invalid api key", "unauthenticated", "401")


def live_refused(e: BaseException) -> bool:
    """A definite no from the Live API (as opposed to a network blip or today's quota)."""
    msg = str(e).lower()
    return any(k in msg for k in _REFUSED)


class VoiceNotSetUp(RuntimeError):
    """No voice can start yet, and NEO won't download one by itself: no Gemini key (or one Live
    refused, or the voice is set to "On this Mac") and the on-device voice isn't downloaded."""

    def __init__(self, why: str, problem: str) -> None:
        super().__init__(why)
        self.problem = problem  # what the overlay's problem list says


def local_ready() -> bool:
    """Are the on-device speech models already downloaded, whole, at their pinned revisions? (Never
    start a 2.9 GB download unasked: not at a first start, not as a fallback mid-conversation.)"""
    try:
        from neo2.core import models

        s = settings()
        return all(models.here(m) for m in (s.stt_model, s.tts_model))
    except Exception:  # noqa: BLE001
        return False


def _size() -> str:
    from neo2.core.fetch import LOCAL_VOICE_BYTES

    return f"{LOCAL_VOICE_BYTES / 1e9:.1f} GB"


def _not_set_up(why: str) -> VoiceNotSetUp:
    s = settings()
    if s.voice == "local":
        problem = (f"The on-device voice isn't downloaded yet ({_size()}). Download it in Brains & Keys, or pick "
                   "Gemini Live there. Typing works meanwhile.")  # fmt: skip
    elif s.gemini_api_key:
        problem = (f"Gemini Live refused this key, and the on-device voice isn't downloaded ({_size()}). Open Brains "
                   "& Keys to download it or use another Gemini key. Typing works meanwhile.")  # fmt: skip
    else:
        problem = (f"NEO's voice needs a Gemini key or the on-device voice (a {_size()} download). Open Brains & "
                   "Keys for either. Typing works meanwhile.")  # fmt: skip
    return VoiceNotSetUp(why, problem)


async def make_local(on_text: TextHandler):
    from neo2.core.voice.local import LocalVoice

    s = settings()
    v = LocalVoice(on_text)
    await v.start()
    print(f"[voice] local cascade ({s.stt_model.split('/')[-1]} → {s.tts_model.split('/')[-1]})")
    return v


async def make_voice(on_text: TextHandler, agent_session=None, *, on_speech: TextHandler | None = None):
    """`on_text`: what handles typed words (Gemini Live hands them over when it is down);
    `on_speech`: what handles the on-device voice's transcribed speech (default: the same)."""
    s = settings()
    mode = s.voice
    if mode in ("auto", "live") and s.gemini_api_key:
        from neo2.core.voice.live import LiveVoice

        v = LiveVoice(on_text, agent_session)
        try:
            await asyncio.wait_for(v.probe(), _PROBE_TIMEOUT_S)
            refused = False
        except Exception as e:  # noqa: BLE001
            why = "timed out" if isinstance(e, TimeoutError) else str(e)[:120]
            refused = live_refused(e)
            if refused and mode == "auto":
                print(f"[voice] Gemini Live refused this key ({why}); using the on-device voice")
            elif refused:
                raise
            else:
                print(f"[voice] Gemini Live not reachable yet ({why}); it will connect on “Neo”")
        if not refused:
            await v.start()
            print(f"[voice] Gemini Live ({s.gemini_live_model})")
            return v
    if not local_ready():  # (its download is the user's to start: VoiceNotSetUp says how)
        if mode == "local":
            raise _not_set_up("the on-device voice isn't downloaded")
        if not s.gemini_api_key:
            raise _not_set_up("no Gemini key, and the on-device voice isn't downloaded")
        raise _not_set_up("Gemini Live refused the key, and the on-device voice isn't downloaded")
    return await make_local(on_speech or on_text)
