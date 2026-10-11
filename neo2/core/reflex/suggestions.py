"""Quiet, local counters for repeated successful app/site setup workflows.

Only a hash of the ordered app names / website hosts, a count and timestamps are persisted. No
request text, page content, titles, paths or query strings are kept here.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from neo2.core import runtime
from neo2.core.config import settings

_FILE = "workflow_patterns.json"
_LOCK = threading.RLock()
_MAX = 500
_KEEP_S = 90 * 86400
_SUGGEST_AT = 3


def _path() -> Path:
    return settings().data_dir / _FILE


def _signature(steps: list) -> tuple[str, int] | None:
    actions = []
    for step in steps:
        if isinstance(step, dict):
            name = str(step.get("tool") or "")
            args = step.get("args") or {}
        else:
            name = str(getattr(step, "tool", "") or "")
            args = getattr(step, "args", {}) or {}
        if name == "open_app":
            target = " ".join(str(args.get("target") or "").casefold().split())
            if not target or "/" in target or "://" in target:
                return None
            actions.append(("app", target[:100]))
        elif name == "open_url":
            try:
                target = urlsplit(str(args.get("url") or ""))
                host = (target.hostname or "").lower()
            except ValueError:
                return None
            if target.scheme not in ("http", "https") or not host or target.username or target.password:
                return None
            actions.append(("site", host[:120]))
        else:
            return None
    if len(actions) < 2 or len(actions) > 8:
        return None
    packed = json.dumps(actions, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()[:24], len(actions)


def _read() -> dict:
    try:
        value = json.loads(_path().read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def record_success(steps: list) -> str:
    """Record a verified launch-only workflow and suggest a shortcut on its third success."""
    if runtime.library_bound() or not settings().learn or not settings().workflow_suggestions:
        return ""
    sig = _signature(steps)
    if sig is None:
        return ""
    key, actions = sig
    try:
        from neo2.core.tools.routines import has_actions

        routine_actions = []
        for step in steps:
            if isinstance(step, dict):
                name, args = step.get("tool"), step.get("args") or {}
            else:
                name, args = getattr(step, "tool", ""), getattr(step, "args", {}) or {}
            routine_actions.append({"kind": name, "target": args.get("target") if name == "open_app" else args.get("url")})
        if has_actions(routine_actions):
            return ""
    except Exception:  # noqa: BLE001 — suggestions never interrupt a task
        pass
    now = time.time()
    path = _path()
    with _LOCK:
        data = _read()
        patterns = data.get("patterns") if isinstance(data.get("patterns"), dict) else {}
        patterns = {
            k: v for k, v in patterns.items()
            if isinstance(v, dict) and now - float(v.get("last", 0)) <= _KEEP_S
        }
        old = patterns.get(key, {})
        count = int(old.get("count", 0)) + 1
        patterns[key] = {"count": count, "last": now, "actions": actions}
        # Keep recent patterns only; the insertion order is from the most recent successful task.
        patterns = dict(list(patterns.items())[-_MAX:])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"patterns": patterns}, f, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    if count != _SUGGEST_AT:
        return ""
    return "You’ve repeated this app and website setup a few times. Want to give it a spoken shortcut? Tell me the phrase you’d like to say and I’ll save it as a routine."


def forget() -> int:
    path = _path()
    with _LOCK:
        try:
            value = _read()
            count = len(value.get("patterns") or {})
        except Exception:  # noqa: BLE001
            count = 0
        path.unlink(missing_ok=True)
        return count
