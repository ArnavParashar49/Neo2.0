"""Small things NEO learned about where things go on this Mac, one store per namespace.

* ``sites`` - a site word (and country) -> the host a confirmed visit landed on;
* ``templates`` - a host -> its search URL template (the terms slot only, nothing else);
* ``bindings`` - a hashed phrase -> the contact it meant and which of their handles (by label);
* ``containers`` - (kind, app) -> the folder / list / calendar id the user chose;
* ``store_views`` - kind:app -> the app writes into an OS store route;
* ``ax_recipes`` - (kind, app, version) -> the UI fields and keys that worked.

Never message, note or event text, never a number or address, never a search query. Kept in
``<data_dir>/decide_learned/<ns>.json`` (folder 0700, files 0600, replaced atomically), capped per
namespace, and gone with the privacy erase. A write from a request that began before an erase is
dropped.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from neo.config import settings

__all__ = ["NAMESPACES", "get", "put", "drop", "forget"]

NAMESPACES = ("sites", "templates", "bindings", "containers", "store_views", "ax_recipes")
CAP = 500

_lock = threading.RLock()
_cache: dict[str, tuple[int, dict[str, dict]]] = {}  # path -> (mtime_ns, {key: {"v": value, "t": when}})


def _dir() -> Path:
    return settings().data_dir / "decide_learned"


def _path(ns: str) -> Path:
    if ns not in NAMESPACES:
        raise ValueError(f"no learned store called {ns!r}")
    return _dir() / f"{ns}.json"


def _load(ns: str) -> dict[str, dict]:
    p = _path(ns)
    with _lock:
        try:
            mtime = p.stat().st_mtime_ns
        except OSError:
            return {}
        hit = _cache.get(str(p))
        if hit and hit[0] == mtime:
            return hit[1]
        try:
            raw = json.loads(p.read_text())
        except (OSError, ValueError) as e:
            print(f"[decide] {p.name} was unreadable ({str(e)[:60]}); starting it fresh")
            raw = {}
        data = {
            str(k): {"v": v["v"], "t": float(v.get("t", 0.0))}
            for k, v in (raw.items() if isinstance(raw, dict) else ())
            if isinstance(v, dict) and isinstance(v.get("v"), dict)
        }
        _cache[str(p)] = (mtime, data)
        return data


def _save(ns: str, data: dict[str, dict]) -> None:
    d = _dir()
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    p = _path(ns)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=f".{ns}.", suffix=".tmp")  # created 0600
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, separators=(",", ":"), ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    with contextlib.suppress(OSError):
        _cache[str(p)] = (p.stat().st_mtime_ns, data)


def get(ns: str, key: str) -> dict | None:
    hit = _load(ns).get(key)
    return dict(hit["v"]) if hit else None


def put(ns: str, key: str, value: dict[str, Any], *, cap: int = CAP) -> None:
    """Learn ``key`` -> ``value``; the least recently learned entries go past ``cap``."""
    from neo.decide import run

    if run.erased():
        return
    with _lock:
        data = dict(_load(ns))
        data[key] = {"v": dict(value), "t": time.time()}
        for old in sorted(data, key=lambda k: data[k]["t"])[: max(0, len(data) - cap)]:
            del data[old]
        try:
            _save(ns, data)
        except OSError as e:
            print(f"[decide] couldn't save what was learned ({str(e)[:60]})")


def drop(ns: str, key: str) -> None:
    with _lock:
        data = dict(_load(ns))
        if data.pop(key, None) is not None:
            try:
                _save(ns, data)
            except OSError as e:
                print(f"[decide] couldn't update what was learned ({str(e)[:60]})")


def forget() -> int:
    """The privacy erase: every learned store goes. Returns how many entries there were."""
    with _lock:
        n = sum(len(_load(ns)) for ns in NAMESPACES)
        _cache.clear()
        d = _dir()
        for f in d.glob("*") if d.is_dir() else ():
            f.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            d.rmdir()
        return n
