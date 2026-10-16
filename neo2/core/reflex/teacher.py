"""Idle-time labels for the requests Laya was least sure about — off unless NEO_LEARN_TEACHER is set.

A request that no outcome settled carries only the handling model's guess. While NEO is idle, the
unsettled rows with the smallest margin (least-margin picks beat random ones by 1-4 points for the
same number of labels, research_learning 6) are labelled again, blind, with generate.py's labeller.
As in generate.py only rows both models agree on are kept: a row the labeller disputes is stored
with ``w`` 0, which drops it wherever training weights learned rows by ``w`` (until training does,
a verdict changes nothing). One it confirms keeps its weight: the labeller may be the very
provider whose decision made the label, so its agreement verifies nothing. No label is ever
changed. At most 50 rows a day.

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

from neo2.core.config import settings
from neo2.core.providers.base import ProviderError
from neo2.core.reflex import learn
from neo2.core.reflex.generate import _LABEL_SYS, _ask, _parse_list, label_prompt, safe_error
from neo2.core.reflex.schema import INTENT_CRITERIA

STATE = "reflex_teacher.json"  # {"day", "sent"}: counts only, never text
DAILY_MAX = 50
_CHUNK = 25
_GAP_S = 3600.0  # at most one round an hour: new unsure rows arrive through the day
_DISPUTED_W = 0.0
_BRAINS = ("gemini", "gemini_lite", "groq", "claude", "local")


def mode() -> str:
    s = settings()
    return s.learn_teacher if s.learn else "off"


def provider_of(row: dict) -> str | None:
    """The brain that already read this request's text, or None when that can't be known. A bare
    old name: one of _BRAINS, as before; another brain reference ("claude@claude-haiku-5-5",
    "custom:deepseek") when its provider is set up (its key is still there)."""
    brain = row.get("brain")
    if brain in _BRAINS:
        return "gemini" if brain == "gemini_lite" else brain
    if isinstance(brain, str) and brain:
        from neo2.core.brainref import parse
        from neo2.core.providers import catalog

        r = parse(brain)
        if r.valid and not r.legacy and r.text == brain and catalog.configured(catalog.provider_of(brain)):
            return brain
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
        from neo2.core.providers.gemini import GeminiProvider

        s = settings()
        return GeminiProvider(s.gemini_model, s.gemini_api_key or None, fallback_models=s.gemini_fallback_models)
    if name == "local":
        from neo2.core.providers.local_server import is_up

        if not await is_up():
            raise ProviderError("the local model server isn't running")
    from neo2.core.providers import make

    return make(name)


async def verdicts(rows: list[dict], how: str, *, provider=None) -> list[tuple[dict, dict]]:
    """(row, what to store for it) for every row a labeller answered: the row with who labelled it
    and whether they agreed, and ``w`` 0 where they didn't."""
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
                    v = {**r, "teacher": name, "agreed": lab == r["intent"]}
                    if not v["agreed"]:
                        v["w"] = _DISPUTED_W
                    out.append((r, v))
    return out


def _key(text: str) -> str:
    return " ".join(text.lower().split())


def store(pairs: list[tuple[dict, dict]], erase_gen: int) -> int:
    """Write each verdict over its row in the learning file if that row is still the latest word
    on its sentence — nothing newer recorded, nothing erased since it was read. In its place, with
    the row's own timestamp: rows are kept newest by position and by time, so a verdict mustn't
    look newer than the rows recorded after it. Returns how many were written."""
    path = learn._path(learn.EXTRA)
    with learn._lock:
        if learn._erase_gen != erase_gen or not path.exists():
            return 0
        lines = path.read_text(encoding="utf-8").splitlines()
        rows: dict[int, dict] = {}
        last: dict[str, int] = {}  # sentence → the line with its latest word
        for i, line in enumerate(lines):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(r, dict) and isinstance(r.get("text"), str):
                rows[i] = r
                last[_key(r["text"])] = i
        n = 0
        for r, v in pairs:
            i = last.get(_key(r["text"]))
            if i is not None and rows[i] == r:
                lines[i] = json.dumps(v, ensure_ascii=False)
                n += 1
        if n:
            tmp = path.with_suffix(".tmp")
            tmp.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
            tmp.replace(path)
    return n


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
