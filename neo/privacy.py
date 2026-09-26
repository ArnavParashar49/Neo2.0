"""Erase everything NEO has kept about the user, in one step.

What goes: memories (profile, lessons, notes), playbooks, session summaries, the requests Laya
learned from (on disk, still settling in RAM, and the correction memory) and their cached
features, which apps worked for what (and the songs/videos, sites, lists and contacts NEO
learned), the audit trail, the core's logs, and NEO's own browser profile (its cookies and logins). What stays: settings and keys (~/.neo/.env), timers that are still running, and
downloaded models — none of them is about the user.

Reached only from the overlay (after its own confirmation) or `python -m neo --forget-all`: never
a tool a model can call, so a web page or an email can't wipe anything.
"""

from __future__ import annotations

import asyncio
import shutil

from neo.config import settings

_generation = 0  # bumped by every erase: work that started before it must not write afterwards


def generation() -> int:
    return _generation


async def forget_everything() -> dict[str, int]:
    global _generation
    from neo.memory.store import store
    from neo.reflex import finetune, learn, memory, outcomes

    _generation += 1
    s = settings()
    counts = store().forget_all()
    outcomes.forget()  # requests still settling in RAM: nothing more is learned from them
    counts["learned requests"] = learn.forget_examples()
    memory.forget()  # the corrections' embeddings (RAM only)
    finetune.feature_cache().unlink(missing_ok=True)  # rebuilt from the public datasets on the next retrain
    from neo.decide import engine as decide
    from neo.decide import eventkit, learned, media, run, shadow, sites

    counts["app choices"] = decide.forget()
    counts["learned media"] = media.forget()
    counts["learned places & people"] = learned.forget()
    counts["task dry runs"] = shadow.forget()  # the shadow log
    run.forget()  # receipts and last actions (memory only)
    eventkit.forget()
    sites.forget()  # site words searched for lately (memory only)
    from neo.decide.kinds import schedule

    schedule.forget()  # calendar files handed to a calendar app (an event's title and notes)
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
    if learn._training:
        # A retrain in flight notices the erase before it saves anything; if it was already
        # saving, remove what it wrote once it's done.
        asyncio.get_running_loop().create_task(_unlink_cache_after_retrain())
    return counts


async def _unlink_cache_after_retrain(limit_s: float = 1800.0) -> None:
    from neo.reflex import finetune, learn

    waited = 0.0
    while learn._training and waited < limit_s:
        await asyncio.sleep(2.0)
        waited += 2.0
    finetune.feature_cache().unlink(missing_ok=True)
