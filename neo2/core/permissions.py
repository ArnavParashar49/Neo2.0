"""The macOS permissions NEO needs: Accessibility and Screen Recording (the microphone prompt
appears by itself the first time the voice layer opens the mic).

Started by NEO.app, the core never prompts: macOS attributes the permissions to the app, and the
app asks (once per version) and shows what's missing in its menu. Run from a terminal, the core
asks at most once a day — the system prompt is a nag, not a status.

The microphone and the camera are asked for by the core itself (it is the process that uses them;
macOS credits the prompt to NEO.app, its responsible app, with NEO.app's own words for each). NEO.app's
Setup checklist shows them and has a button for each: the shell asks over its socket
(neo2.core.server.shell_link: `permission`), and :func:`media` answers — what macOS says now, or,
asked to, macOS's prompt and the user's answer.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Any

from neo2.core import effects

_PROMPT_EVERY_S = 24 * 3600
# What the shell may ask about, by AVFoundation's media type (AVMediaTypeAudio, AVMediaTypeVideo).
MEDIA = {"microphone": "soun", "camera": "vide"}
# AVAuthorizationStatus, in the words the shell reads.
_STATUS = {0: "not_determined", 1: "restricted", 2: "denied", 3: "granted"}
STATUSES = (*_STATUS.values(), "unknown")
_ASK_TIMEOUT_S = 120.0  # the prompt waits for the user; a Setup window waits this long for the answer


def _capture_device() -> Any:
    """AVFoundation's AVCaptureDevice (a test sets its own)."""
    import AVFoundation

    return AVFoundation.AVCaptureDevice


def media_status(what: str) -> str:
    """What macOS says NEO may do with the microphone or the camera: granted | denied | restricted |
    not_determined (never asked) | unknown (not a Mac, or AVFoundation can't say)."""
    if what not in MEDIA:
        raise ValueError(f"not a permission NEO asks about: {what!r}")
    if sys.platform != "darwin":
        return "unknown"
    try:
        return _STATUS.get(int(_capture_device().authorizationStatusForMediaType_(MEDIA[what])), "unknown")
    except Exception:  # noqa: BLE001
        return "unknown"


async def media(what: str, *, ask: bool = False, timeout_s: float = _ASK_TIMEOUT_S) -> str:
    """:func:`media_status`; with `ask`, when macOS hasn't asked yet, its prompt now and the user's
    answer ("granted" or "denied"). An answer that doesn't come in time: what macOS says then."""
    status = media_status(what)
    if not ask or status != "not_determined":
        return status
    device = _capture_device()
    effects.guard("permission", f"ask macOS for the {what}", via=device)
    loop = asyncio.get_running_loop()
    answer: asyncio.Future[bool] = loop.create_future()

    def done(granted: bool) -> None:  # (on one of AVFoundation's threads)
        loop.call_soon_threadsafe(lambda: answer.done() or answer.set_result(bool(granted)))

    device.requestAccessForMediaType_completionHandler_(MEDIA[what], done)
    try:
        return "granted" if await asyncio.wait_for(answer, timeout_s) else "denied"
    except TimeoutError:
        return media_status(what)


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
