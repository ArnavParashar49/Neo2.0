"""Ask macOS for the permissions NEO needs, once, at startup.

When NEO.app launches the core, macOS attributes the core's requests to NEO.app, so these
prompts say "NEO". Already granted (or running from a terminal that has them) → no prompt.
The microphone prompt appears by itself the first time the voice layer opens the mic.
"""

from __future__ import annotations

import sys


def request_missing() -> list[str]:
    """Prompt for anything missing; returns the names of what is still missing."""
    if sys.platform != "darwin":
        return []
    missing: list[str] = []
    try:
        import ApplicationServices as AS

        if not AS.AXIsProcessTrusted():
            AS.AXIsProcessTrustedWithOptions({AS.kAXTrustedCheckOptionPrompt: True})
            missing.append("Accessibility")
    except Exception:  # noqa: BLE001
        pass
    try:
        import Quartz as Q

        if not Q.CGPreflightScreenCaptureAccess():
            Q.CGRequestScreenCaptureAccess()
            missing.append("Screen Recording")
    except Exception:  # noqa: BLE001
        pass
    return missing
