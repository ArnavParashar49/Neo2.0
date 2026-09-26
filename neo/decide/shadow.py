"""The shadow log: what a task kind's new path would have done while its legacy code ran.

A kind in "shadow" mode (neo.config ``decide_modes``) answers with legacy, then its driver's dry
run works out in the background what the new path would have chosen. One JSON line per request
goes to ``data_dir/decide_shadow.log`` (0600): the kind and facts that are app keys, hosts, method
names, ids, counts and booleans. Never a title, a name, a query or a message: the drivers' dry runs
report none, and a value that isn't a plain str/number/bool is dropped here. Lines older than 30
days go on each write; the privacy erase deletes the file; a dry run that ends after an erase
writes nothing.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from collections.abc import Awaitable, Iterator, Mapping
from pathlib import Path
from typing import Any

__all__ = ["KEEP_S", "TIMEOUT_S", "path", "record", "later", "starting", "forget"]

KEEP_S = 30 * 86400.0
TIMEOUT_S = 60.0  # a dry run that takes longer (a slow network, a busy app) is logged as timed out
_tasks: set[asyncio.Task] = set()  # the loop keeps only weak references to tasks


def path() -> Path:
    from neo.config import settings

    return settings().data_dir / "decide_shadow.log"


def _plain(facts: Mapping[str, Any]) -> dict[str, str | int | float | bool]:
    return {str(k): v for k, v in facts.items() if isinstance(v, (str, int, float, bool))}


def _stamp(line: str) -> float:
    try:
        return float(json.loads(line).get("t", 0))
    except (ValueError, AttributeError):
        return 0.0


def record(kind: str, facts: Mapping[str, Any]) -> None:
    """Append one request's facts, dropping the lines that are too old."""
    p = path()
    now = time.time()
    line = json.dumps({"t": round(now, 3), **_plain(facts), "kind": kind}, ensure_ascii=False)
    old = p.read_text(encoding="utf-8").splitlines() if p.exists() else []
    kept = [x for x in old if _stamp(x) >= now - KEEP_S]
    if len(kept) == len(old):
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        return
    tmp = p.with_name(p.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("".join(x + "\n" for x in [*kept, line]))
    os.replace(tmp, p)


def later(kind: str, facts: Awaitable[Mapping[str, Any]], **extra: Any) -> None:
    """Run a dry run in the background, after the legacy answer went out, and log its facts
    plus ``extra`` (about legacy's own result). It never raises into the request: a failure is
    logged as its exception's name."""
    from neo import privacy
    from neo.decide import run

    started = privacy.generation()

    async def go() -> None:
        run.begin_request()  # anything the dry run would learn is dropped after an erase, too
        try:
            got = dict(await asyncio.wait_for(facts, TIMEOUT_S))
        except TimeoutError:
            got = {"error": "timeout"}
        except Exception as e:  # noqa: BLE001 - a dry run must never disturb the real one
            got = {"error": type(e).__name__}
        if privacy.generation() != started:
            return
        try:
            record(kind, {**got, **extra})
        except OSError as e:
            print(f"[decide] shadow log for {kind} not written ({str(e)[:60]})")

    t = asyncio.get_running_loop().create_task(go())
    _tasks.add(t)
    t.add_done_callback(_tasks.discard)


@contextlib.contextmanager
def starting(kind: str) -> Iterator[None]:
    """Around starting a dry run, after legacy answered: nothing that goes wrong there (a driver
    that won't import, say) may turn legacy's answer into an error."""
    try:
        yield
    except Exception as e:  # noqa: BLE001
        print(f"[decide] shadow {kind} not started ({str(e)[:60]})")


def forget() -> int:
    """The privacy erase: delete the log. Returns how many requests it held."""
    p = path()
    try:
        n = len(p.read_text(encoding="utf-8").splitlines())
    except OSError:
        return 0
    p.unlink(missing_ok=True)
    return n
