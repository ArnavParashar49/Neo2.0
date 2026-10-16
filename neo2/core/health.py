"""What stops NEO from working fully right now, in words the overlay can show.

The overlay gets this list when it connects (the websocket hello) and again whenever it changes,
so "no API key", "microphone blocked" or "voice is down" are visible instead of silent.
"""

from __future__ import annotations

import asyncio
from typing import Any

from neo2.core.config import settings

_live: dict[str, str] = {}  # problems reported while running (voice down, …)


def problems() -> list[str]:
    s = settings()
    out: list[str] = []
    if not (s.gemini_api_key or s.gemini_api_keys):
        out.append(
            "NEO needs a free Gemini API key to understand you and to talk. Click the menu-bar icon → "
            "Settings & API Keys…, paste the key, save, then choose Restart NEO."
        )
    out.extend(brain_problems(s))
    try:
        from neo2.core.voice import devices

        if devices.mic_status() in (devices.DENIED, devices.RESTRICTED):
            out.append(
                "Microphone access is off, so NEO can't hear you — allow NEO in System Settings → "
                "Privacy & Security → Microphone, then restart NEO. Typing still works."
            )
    except Exception:  # noqa: BLE001
        pass
    try:
        from neo2.core.permissions import missing

        gaps = missing()
        if "Accessibility" in gaps:
            out.append("NEO can't click or type in other apps until you allow Accessibility (menu-bar icon → Allow Accessibility…).")
        elif "Input events" in gaps:
            out.append(
                "macOS isn't letting NEO send clicks or keystrokes — in System Settings → Privacy & Security → "
                "Accessibility, turn NEO off and on again, then restart NEO."
            )
    except Exception:  # noqa: BLE001
        pass
    out.extend(_live.values())
    return out


# A role's setting, as the user would have typed it.
_ROLE_VARS = {
    "brain": "NEO_BRAIN",
    "fast_brain": "NEO_FAST_BRAIN",
    "offline_brain": "NEO_OFFLINE_BRAIN",
    "eyes_brain": "NEO_EYES_BRAIN",
    "light_brain": "NEO_LIGHT_BRAIN",
    "backup_brains": "NEO_BACKUP_BRAINS",
}
def brain_problems(s: Any) -> list[str]:
    """A role set to something NEO can't read as a brain (it is skipped, never swapped for another),
    and NEO's own saved files when they can't be read."""
    from neo2.core import brainref

    out: list[str] = []
    for f, var in _ROLE_VARS.items():
        v = getattr(s, f, None)
        for one in v if isinstance(v, list) else [v]:
            if not isinstance(one, str) or not one or brainref.parse(one).valid:
                continue
            shown = brainref.shown(one)
            out.append(
                f"{var}={shown} isn't a brain NEO knows. Pick one in Brains & Keys."
                if shown
                else f"{var} isn't set to a brain NEO knows. Pick one in Brains & Keys."
            )
    out.extend(_store_problems())
    return out


def _store_problems() -> list[str]:
    from neo2.core import config, keystore

    out: list[str] = []
    for path, cls in ((config.KEY_STORE_PATH, keystore.FileKeyStore), (config.SAVED_PATH, keystore.SavedConfig)):
        if path is None or keystore.refused(path):
            continue
        try:
            st = cls(path).state()
        except Exception:  # noqa: BLE001 — health never fails for this
            continue
        if not st.ok:
            what = "was made by a newer NEO" if st.problem == "newer" else "can't be read"
            out.append(
                f"NEO's {path.name} {what}, so what was saved there isn't used. Saving in Brains & Keys "
                f"sets it aside as {path.name}.bad-… and starts a new one."
            )
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
        from neo2.core.events import bus

        asyncio.get_running_loop().create_task(bus().publish("problems", items=problems()))
    except RuntimeError:
        pass


def clear(prefix: str) -> None:
    """Clear every problem whose key starts with `prefix` ("key:anthropic", "model:anthropic:"): a
    key just replaced or removed (neo2.core.providers.forget)."""
    for key in [k for k in _live if k.startswith(prefix)]:
        report(key, None)


async def watch(every_s: float = 10.0) -> None:
    """Re-check now and then: permissions granted in System Settings, a key added to the settings
    file — the overlay's problem list follows without a restart."""
    from neo2.core.events import bus

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
