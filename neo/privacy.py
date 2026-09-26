"""Erase everything NEO has kept about the user, in one step.

What goes: memories (profile, lessons, notes), playbooks, session summaries, the requests Laya
learned from and their cached features, the audit trail, the core's logs, and NEO's own browser
profile (its cookies and logins). What stays: settings and keys (~/.neo/.env), timers that are
still running, and downloaded models — none of them is about the user.

Reached only from the overlay (after its own confirmation) or `python -m neo --forget-all`: never
a tool a model can call, so a web page or an email can't wipe anything.
"""

from __future__ import annotations

import shutil

from neo.config import settings


async def forget_everything() -> dict[str, int]:
    from neo.memory.store import store
    from neo.reflex import finetune, learn

    s = settings()
    counts = store().forget_all()
    counts["learned requests"] = learn.forget_examples()
    finetune.feature_cache().unlink(missing_ok=True)  # rebuilt from the public datasets on the next retrain
    for p in (s.audit_log, s.audit_log.with_name(s.audit_log.name + ".1")):
        p.unlink(missing_ok=True)
    logs = s.data_dir / "logs"
    for f in logs.glob("core.log*"):
        if f.name == "core.log":
            f.open("r+").truncate(0)  # NEO.app appends to it: empty it in place
        else:
            f.unlink(missing_ok=True)
    try:
        from neo.tools.browser import shutdown as close_browser

        await close_browser()
    except Exception:  # noqa: BLE001 — not open
        pass
    shutil.rmtree(s.data_dir / "browser", ignore_errors=True)
    return counts
