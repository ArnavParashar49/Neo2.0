"""The macOS permissions NEO needs: Accessibility and Screen Recording (the microphone prompt
appears by itself the first time the voice layer opens the mic).

Started by NEO.app, the core never prompts: macOS attributes the permissions to the app, and the
app asks (once per version) and shows what's missing in its menu. Run from a terminal, the core
asks at most once a day — the system prompt is a nag, not a status.
"""

from __future__ import annotations

import os
import sys
import time

from neo2.core import effects

_PROMPT_EVERY_S = 24 * 3600


def missing() -> list[str]:
    if sys.platform != "darwin":
        return []
    out: list[str] = []
    try:
        import ApplicationServices as AS

        if not AS.AXIsProcessTrusted():
            out.append("Accessibility")
        else:
            # Trusted to read the UI, but macOS may still drop the clicks and keys NEO posts (its
            # separate "control the computer" check): nothing would arrive, silently.
            from neo2.core.tools.computer.input import can_post

            if not can_post():
                out.append("Input events")
    except Exception:  # noqa: BLE001
        pass
    try:
        import Quartz as Q

        if not Q.CGPreflightScreenCaptureAccess():
            out.append("Screen Recording")
    except Exception:  # noqa: BLE001
        pass
    return out


def request_missing() -> list[str]:
    """Returns what's missing; prompts only when that's ours to do and not done recently."""
    gaps = missing()
    if not gaps or os.environ.get("NEO_LAUNCHED_BY_APP"):
        return gaps
    from neo2.core.config import settings

    stamp = settings().data_dir / "permissions-prompted"
    try:
        if time.time() - stamp.stat().st_mtime < _PROMPT_EVERY_S:
            return gaps
    except OSError:
        pass
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.touch()
    try:
        if "Accessibility" in gaps:
            import ApplicationServices as AS

            effects.guard("permission", "ask macOS for Accessibility", via=AS.AXIsProcessTrustedWithOptions)
            AS.AXIsProcessTrustedWithOptions({AS.kAXTrustedCheckOptionPrompt: True})
        if "Screen Recording" in gaps:
            import Quartz as Q

            effects.guard("permission", "ask macOS for Screen Recording", via=Q.CGRequestScreenCaptureAccess)
            Q.CGRequestScreenCaptureAccess()
        if "Input events" in gaps:
            import Quartz as Q

            if hasattr(Q, "CGRequestPostEventAccess"):
                effects.guard("permission", "ask macOS to let NEO post input events", via=Q.CGRequestPostEventAccess)
                Q.CGRequestPostEventAccess()
    except Exception:  # noqa: BLE001
        pass
    return gaps
