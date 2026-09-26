"""Picks the voice backend: Gemini Live when the key + quota allow it, else the local cascade.

A failed check at startup is usually the network not being up yet (NEO starts at login) — that
doesn't turn Live off: it connects on the next "Neo". Only a definite "this key can't use
Live" falls back to the on-device voice for good.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from neo.config import settings

TextHandler = Callable[[str], Awaitable[None]]
_PROBE_TIMEOUT_S = 15.0
_REFUSED = ("not found", "404", "permission", "403", "not supported", "api key not valid", "invalid api key", "unauthenticated", "401")


def live_refused(e: BaseException) -> bool:
    """A definite no from the Live API (as opposed to a network blip or today's quota)."""
    msg = str(e).lower()
    return any(k in msg for k in _REFUSED)


def local_ready() -> bool:
    """Are the on-device speech models already downloaded? (Never start a 2.6 GB download as a
    fallback in the middle of a conversation.)"""
    try:
        from huggingface_hub import try_to_load_from_cache

        s = settings()
        return all(isinstance(try_to_load_from_cache(m, "config.json"), str) for m in (s.stt_model, s.tts_model))
    except Exception:  # noqa: BLE001
        return False


async def make_local(on_text: TextHandler):
    from neo.voice.local import LocalVoice

    s = settings()
    v = LocalVoice(on_text)
    await v.start()
    print(f"[voice] local cascade ({s.stt_model.split('/')[-1]} → {s.tts_model.split('/')[-1]})")
    return v


async def make_voice(on_text: TextHandler, agent_session=None):
    s = settings()
    mode = s.voice
    if mode in ("auto", "live") and s.gemini_api_key:
        from neo.voice.live import LiveVoice

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
    return await make_local(on_text)
