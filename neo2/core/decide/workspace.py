"""Which app macOS itself would use: LaunchServices' default handlers, by URL scheme or by file
extension. Read-only: nothing here opens an app, a URL or a file."""

from __future__ import annotations

from neo2.core.decide._objc import pooled

__all__ = ["default_handler", "handlers", "default_for_extension"]


@pooled
def _bundle_id(path: str) -> str | None:
    try:
        from Foundation import NSBundle

        b = NSBundle.bundleWithPath_(path)
        bid = b.bundleIdentifier() if b is not None else None
    except Exception:  # noqa: BLE001
        return None
    return str(bid) if bid else None


@pooled
def _handler_paths(scheme: str) -> list[str]:
    try:
        from AppKit import NSWorkspace
        from Foundation import NSURL

        url = NSURL.URLWithString_(f"{scheme}:")
        found = NSWorkspace.sharedWorkspace().URLsForApplicationsToOpenURL_(url) if url is not None else None
    except Exception:  # noqa: BLE001
        return []
    return [str(u.path()) for u in found or []]


def _scheme(s: str) -> str:
    return (s or "").strip().lower().rstrip(":")


def default_handler(scheme: str) -> str | None:
    """The bundle id of the app that opens ``scheme:`` links ('mailto', 'sms', 'https')."""
    from neo2.core.decide import inventory

    s = _scheme(scheme)
    path = inventory._ls_default_path(f"{s}:") if s else None
    return _bundle_id(path) if path else None


def handlers(scheme: str) -> list[str]:
    """Bundle ids of every app that can open ``scheme:`` links."""
    s = _scheme(scheme)
    ids = [b for p in (_handler_paths(s) if s else []) if (b := _bundle_id(p))]
    return list(dict.fromkeys(ids))


def default_for_extension(ext: str) -> str | None:
    """The bundle id of the app that opens files with this extension ('ics', 'md')."""
    from neo2.core.decide import inventory

    e = (ext or "").strip().lstrip(".").lower()
    path = inventory._ls_default_path(e) if e and "." not in e and "/" not in e else None
    return _bundle_id(path) if path else None
