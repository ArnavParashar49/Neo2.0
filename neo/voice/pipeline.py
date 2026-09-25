"""Picks the voice backend: Gemini Live when the key + quota allow it, else the local cascade."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from neo.config import settings

TextHandler = Callable[[str], Awaitable[None]]


async def make_voice(on_text: TextHandler, agent_session=None):
    s = settings()
    mode = s.voice
    if mode in ("auto", "live") and s.gemini_api_key:
        try:
            from neo.voice.live import LiveVoice

            v = LiveVoice(on_text, agent_session)
            await v.probe()
            await v.start()
            print(f"[voice] Gemini Live ({s.gemini_live_model})")
            return v
        except Exception as e:  # noqa: BLE001
            print(f"[voice] Live unavailable ({str(e)[:120]}); using local cascade")
            if mode == "live":
                raise
    from neo.voice.local import LocalVoice

    v = LocalVoice(on_text)
    await v.start()
    print(f"[voice] local cascade ({s.stt_model.split('/')[-1]} → {s.tts_model.split('/')[-1]})")
    return v
