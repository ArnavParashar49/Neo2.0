"""Idle-time labels for the requests Laya was least sure about — off unless NEO_LEARN_TEACHER is set.

A request that no outcome settled carries only the handling model's guess. While NEO is idle, the
unsettled rows with the smallest margin (least-margin picks beat random ones by 1-4 points for the
same number of labels, research_learning 6) are labelled again, blind, with generate.py's labeller.
As in generate.py only rows both models agree on are kept: a row the labeller confirms stays at
full weight, one it disputes is dropped from training (w 0). No label is ever changed. At most 50
rows a day.

Who may read a request's text — never anyone who hadn't already:

* ``local``: only the local model (the mlx-lm server, and only while it is already running).
* ``cloud``: only the provider that handled that request. Gemini Live's rows go to Gemini on the
  same key; a row that names its ``brain`` goes to that brain. A row whose provider can't be known
  is never sent anywhere.
"""

from __future__ import annotations

import asyncio
import json
import time

from neo.config import settings
from neo.providers.base import ProviderError
from neo.reflex import learn
from neo.reflex.generate import _LABEL_SYS, _ask, _parse_list, label_prompt, safe_error
from neo.reflex.schema import INTENT_CRITERIA

STATE = "reflex_teacher.json"  # {"day", "sent"}: counts only, never text
DAILY_MAX = 50
_CHUNK = 25
_GAP_S = 3600.0  # at most one round an hour: new unsure rows arrive through the day
_AGREED_W, _DISPUTED_W = 1.0, 0.0
_BRAINS = ("gemini", "gemini_lite", "groq", "claude", "local")


def mode() -> str:
    s = settings()
    return s.learn_teacher if s.learn else "off"


def provider_of(row: dict) -> str | None:
    """The brain that already read this request's text, or None when that can't be known."""
    brain = row.get("brain")
    if brain in _BRAINS:
        return "gemini" if brain == "gemini_lite" else brain
    if row.get("source") == "live":  # Gemini Live heard it
        return "gemini"
    return None


def labeller_for(row: dict, how: str) -> str | None:
    if how == "local":
        return "local"
    return provider_of(row) if how == "cloud" else None


def pick(rows: list[dict], n: int, how: str) -> list[dict]:
    """Up to `n` rows that no outcome settled, no teacher has seen and `how` may send: least sure first."""
    cands = [
        r
        for r in rows
        if not r.get("verified")
        and not r.get("wrong")
        and "teacher" not in r
        and r.get("intent") in INTENT_CRITERIA
        and isinstance(r.get("margin"), int | float)
        and labeller_for(r, how)
    ]
    return sorted(cands, key=lambda r: r["margin"])[: max(0, n)]


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def budget_left() -> int:
    try:
        st = json.loads((settings().data_dir / STATE).read_text())
    except Exception:  # noqa: BLE001 — none yet, or unreadable: nothing spent
        st = {}
    return max(0, DAILY_MAX - int(st.get("sent", 0))) if st.get("day") == _today() else DAILY_MAX


def _spend(n: int) -> None:
    p = settings().data_dir / STATE
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"day": _today(), "sent": DAILY_MAX - budget_left() + n}))
    tmp.replace(p)


async def _provider(name: str):
    """Gemini on the primary key only (the one Live uses); the local model only if it's serving."""
    if name == "gemini":
        from neo.providers.gemini import GeminiProvider

        return GeminiProvider(api_key=settings().gemini_api_key or None)
    if name == "local":
        from neo.providers.local_server import is_up

        if not await is_up():
            raise ProviderError("the local model server isn't running")
    from neo.providers import make

    return make(name)


async def verdicts(rows: list[dict], how: str, *, provider=None) -> list[tuple[dict, dict]]:
    """(row, what to store for it) for every row a labeller answered: the row with ``w`` 1 where
    the labeller agrees with its label, 0 where it doesn't, and who labelled it."""
    provider = provider or _provider
    groups: dict[str, list[dict]] = {}
    for r in rows:
        if name := labeller_for(r, how):
            groups.setdefault(name, []).append(r)
    out: list[tuple[dict, dict]] = []
    for name, rs in groups.items():
        try:
            p = await provider(name)
        except Exception as e:  # noqa: BLE001 — no key, server down: try again next round
            print(f"[reflex] teacher: {name} unavailable ({safe_error(e)})")
            continue
        for i in range(0, len(rs), _CHUNK):
            chunk = rs[i : i + _CHUNK]
            _spend(len(chunk))  # before sending: once asked, the text has gone either way
            try:
                reply = await _ask(p, _LABEL_SYS, label_prompt([r["text"] for r in chunk]), max_tokens=600)
            except Exception as e:  # noqa: BLE001
                print(f"[reflex] teacher: {name} failed ({safe_error(e)})")
                break
            labels = _parse_list(reply)
            if len(labels) != len(chunk):  # can't tell which label is whose: a wrong match costs more
                continue
            for r, lab in zip(chunk, labels, strict=True):
                if lab in INTENT_CRITERIA:
                    ok = lab == r["intent"]
                    out.append((r, {**r, "w": _AGREED_W if ok else _DISPUTED_W, "teacher": name, "agreed": ok}))
    return out


def store(pairs: list[tuple[dict, dict]], erase_gen: int) -> int:
    """Append each verdict to the learning file if its row is still the latest word on that
    sentence — nothing newer recorded, nothing erased since it was read. Keeps the row's own
    timestamp, so its retention clock doesn't restart. Returns how many were written."""
    path = learn._path(learn.EXTRA)
    with learn._lock:
        if learn._erase_gen != erase_gen:
            return 0
        latest = {r["text"].lower(): r for r in learn._read(path)}
        new = [v for r, v in pairs if latest.get(r["text"].lower()) == r]
        if new:
            with path.open("a", encoding="utf-8") as f:
                f.writelines(json.dumps(v, ensure_ascii=False) + "\n" for v in new)
    return len(new)


async def run_once() -> dict:
    """One round: pick, label, store. Returns counts, never text."""
    how = mode()
    if how == "off" or learn._training:
        return {"labelled": 0, "agreed": 0, "stored": 0}
    erase_gen = learn._erase_gen
    rows = pick(learn._read(learn._path(learn.EXTRA)), budget_left(), how)
    pairs = await verdicts(rows, how) if rows else []
    stored = store(pairs, erase_gen) if pairs else 0
    agreed = sum(v["agreed"] for _, v in pairs)
    if pairs:
        print(f"[reflex] teacher ({how}): {len(pairs)} re-labelled, {agreed} agreed, {len(pairs) - agreed} dropped")
    return {"labelled": len(pairs), "agreed": agreed, "stored": stored}


async def teacher_loop(is_idle) -> None:
    """Background task in the core: a round once NEO has been idle a while, at most hourly."""
    if mode() == "off":
        return
    idle_since: float | None = None
    next_round = 0.0
    while True:
        await asyncio.sleep(60)
        try:
            if not is_idle():
                idle_since = None
                continue
            now = time.time()
            idle_since = idle_since or now
            if now - idle_since >= learn._IDLE_S and now >= next_round and budget_left():
                next_round = now + _GAP_S
                await run_once()
        except Exception as e:  # noqa: BLE001 — labelling must never take NEO down
            print(f"[reflex] teacher failed: {safe_error(e)}")
