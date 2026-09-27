"""What stops NEO from working fully right now, in words the overlay can show.

The overlay gets this list when it connects (the websocket hello) and again whenever it changes,
so "no API key", "microphone blocked" or "voice is down" are visible instead of silent.
"""

from __future__ import annotations

import asyncio

from neo.config import settings

_live: dict[str, str] = {}  # problems reported while running (voice down, …)


def problems() -> list[str]:
    s = settings()
    out: list[str] = []
    if not (s.gemini_api_key or s.gemini_api_keys):
        out.append(
            "NEO needs a free Gemini API key to understand you and to talk. Click the menu-bar icon → "
            "Settings & API Keys…, paste the key, save, then choose Restart NEO."
        )
    try:
        from neo.voice import devices

        if devices.mic_status() in (devices.DENIED, devices.RESTRICTED):
            out.append(
                "Microphone access is off, so NEO can't hear you — allow NEO in System Settings → "
                "Privacy & Security → Microphone, then restart NEO. Typing still works."
            )
    except Exception:  # noqa: BLE001
        pass
    try:
        from neo.permissions import missing

        if "Accessibility" in missing():
            out.append("NEO can't click or type in other apps until you allow Accessibility (menu-bar icon → Allow Accessibility…).")
    except Exception:  # noqa: BLE001
        pass
    out.extend(_live.values())
    return out


def report(key: str, message: str | None) -> None:
    """Set (or clear, with None) a problem found while running, and tell the overlay."""
    if message is None:
        if _live.pop(key, None) is None:
            return
    elif _live.get(key) == message:
        return
    else:
        _live[key] = message
    try:
        from neo.events import bus

        asyncio.get_running_loop().create_task(bus().publish("problems", items=problems()))
    except RuntimeError:
        pass


async def watch(every_s: float = 10.0) -> None:
    """Re-check now and then: permissions granted in System Settings, a key added to the settings
    file — the overlay's problem list follows without a restart."""
    from neo.events import bus

    last: list[str] | None = None
    while True:
        try:
            items = await asyncio.to_thread(problems)
            if last is not None and items != last:
                await bus().publish("problems", items=items)
            last = items
        except Exception as e:  # noqa: BLE001
            print(f"[health] check failed: {str(e)[:80]}")
        await asyncio.sleep(every_s)
