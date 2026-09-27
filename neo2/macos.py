"""macOS helpers for hosts that give an agent the Mac tools (``tools="mac"`` / ``"mac_read"``).

Nothing here runs by itself: an embedded agent never prompts for permissions and never pumps
Cocoa's run loop on its own. When the Mac tools are on, NEO clicks and types on the real desktop:
keep your own approval UI off it, or list your app in ``AgentConfig.protected_apps``.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys

__all__ = ["available", "cocoa_pump", "permission_status", "request_permissions"]


def available() -> bool:
    """This is macOS and the Mac tools' frameworks (pyobjc) are installed."""
    if sys.platform != "darwin":
        return False
    return all(importlib.util.find_spec(m) is not None for m in ("AppKit", "Quartz", "ApplicationServices"))


def permission_status() -> dict[str, bool]:
    """{"accessibility": …, "screen_recording": …} for this process. Never prompts."""
    if sys.platform != "darwin":
        return {"accessibility": False, "screen_recording": False}
    from neo import permissions

    gaps = permissions.missing()
    return {"accessibility": "Accessibility" not in gaps, "screen_recording": "Screen Recording" not in gaps}


def request_permissions() -> list[str]:
    """Ask macOS for what's missing (its system prompts) — only when your app decides to, e.g.
    from a settings screen. Returns what was missing ([] = nothing to ask)."""
    if sys.platform != "darwin":
        return []
    from neo import permissions

    gaps = permissions.missing()
    try:
        if "Accessibility" in gaps:
            import ApplicationServices as AS

            AS.AXIsProcessTrustedWithOptions({AS.kAXTrustedCheckOptionPrompt: True})
        if "Screen Recording" in gaps:
            import Quartz as Q

            Q.CGRequestScreenCaptureAccess()
    except Exception:  # noqa: BLE001 — a prompt that can't be shown: still report the gaps
        pass
    return gaps


async def cocoa_pump(interval_s: float = 0.2) -> None:
    """For a host running asyncio on the main thread: let Cocoa deliver its notifications, so the
    Mac tools see apps launched after the process started and which one is in front. Run it as a
    task (``asyncio.create_task(neo2.macos.cocoa_pump())``); it runs until cancelled."""
    try:
        from AppKit import NSDate, NSRunLoop
    except ImportError:  # not macOS, or no pyobjc
        return
    while True:
        NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.002))
        await asyncio.sleep(interval_s)
