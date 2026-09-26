"""Laya learns from use.

Every request the user makes ends up handled somehow, and *how it went* is a label for what they
said. The outcome ledger (``neo/reflex/outcomes.py``) watches each request in RAM — what ran,
whether each step worked, whether the user stopped it or said "no, I meant…" — and only once it
has settled hands a row to :func:`record`:

* verified (``w`` 1.0): what actually ran and worked, with nothing contradicting it;
* teacher-only (``w`` 0.5): what Gemini Live or NEO's brain chose, with no outcome to check;
* corrected: the words of a request the user corrected, with the label of the fix and the
  original in ``wrong`` (a hard negative).

Those rows go to ``~/.neo/reflex_extra.jsonl`` (the file ``dataset.build`` already reads — on
this Mac only): the request's words (only the command of a message), labels, weight, flags, the
reflex's confidence, the script and the *day* — never tool arguments, results, replies or
embeddings. What is never stored at all: dictated text, notes, memories and mail (the words are
content, not a request), and anything that looks like a password, code or card number. Verified
rows are kept 180 days, teacher-only ones 30; at most 2,000, the least useful dropped first.
``reflex_learn.json`` holds the training state and daily counters without any text.
``NEO_LEARN=off`` turns all of it off and ``forget_examples()`` wipes it. When enough new rows
(or a few corrections) have piled up and NEO has been idle for a while, the heads are retrained
inside the running core, reusing the Laya model already in memory, and swapped in — only if they
score at least as well as the current ones.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from pathlib import Path

from neo.config import settings

EXTRA = "reflex_extra.jsonl"
STATE = "reflex_learn.json"
_MIN_NEW = 30  # new examples before a retrain is worth it
_MIN_FIXES = 5  # …or this many new corrections: the user said the reflex was wrong
_MIN_GAP_S = 6 * 3600  # at most this often…
_BURST = 150  # …unless this many have piled up
_IDLE_S = 180.0  # NEO must be idle this long before training starts
_lock = threading.Lock()
_training = False
_erase_gen = 0  # bumped by forget_examples(): a retrain that started before it throws its work away

_KEEP_ROWS = 2000
_KEEP_DAYS = 180  # verified outcomes
_KEEP_GUESS_DAYS = 30  # teacher-only rows: a guess nobody checked goes stale sooner
_LOW_CONF = 0.6  # a verified row the reflex was unsure about teaches the most (after corrections)
_COMPACT_BYTES = 1_000_000
_COUNT_DAYS = 90
# The request's *text* is the content here — not something to keep.
_CONTENT_TOOLS = {
    "type_text", "dictate", "memory", "notes_create", "mail_send", "write_file", "edit_file",
    "reminder_add", "calendar_add", "ax_set_value", "browser_type", "shell", "applescript",
}  # fmt: skip
# "text Priya that I'm running late" → "text Priya": the words after that/saying/: (or straight
# after "tell <someone>") are a message.
_PAYLOAD = re.compile(
    r"^(?P<cmd>.{0,60}?\b(?:text|message|msg|tell|reply|email|mail|write\s+to|ping|dm|whatsapp|send)\b.{0,40}?)"
    r"(?:\s+(?:that|saying|to\s+say|and\s+say)|\s*:)\s+\S+(?:\s+\S+)+.*$"
    r"|^(?P<cmd2>(?:text|message|msg|tell|whatsapp|dm|ping)\s+(?!(?:me|us|a|an|the|about)\b)\S+)\s+\S+(?:\s+\S+){2,}$",
    re.I,
)
_SECRET = re.compile(
    r"\b(?:password|passcode|pass\s*word|pin|otp|one[-\s]?time\s+code|cvv|api\s*key|token|secret|ssn)\b"
    r"|\d[\d\s-]{5,}\d",  # long digit runs: codes, card and phone numbers
    re.I,
)
_SKIP = re.compile(r"^\s*(?:(?:hey\s+)?neo|stop|cancel|never\s*mind|ok(?:ay)?|yes|yeah|no|thanks?(?:\s+you)?)\s*[.!?]*\s*$", re.I)
_DEVANAGARI = re.compile("[ऀ-ॿ]")
_KIND = re.compile(r"[a-z_]{1,32}")
_counts: dict[str, int] = {}  # today's counters, not yet in the state file
_tidied_day: int | None = None  # the day the file on disk was last held to the retention limits


def _path(name: str) -> Path:
    return settings().data_dir / name


def enabled() -> bool:
    return settings().learn


def norm(text: str) -> str:
    """The key a sentence is known by: case and spacing don't make it a different request."""
    return " ".join(str(text).lower().split())


def script(text: str) -> str:
    """A cheap script tag (Laya can't tell Hinglish from English; the script at least splits
    Devanagari from Latin)."""
    return "devanagari" if _DEVANAGARI.search(text) else "latin"


def _day(t: float) -> int:
    return int(t // 86400) * 86400


def skeleton(text: str) -> str | None:
    """The part of a request worth keeping ("text Priya", not what to tell her), or None if it
    isn't one: too short, noise, or it looks like it carries a secret."""
    text = " ".join(str(text).split())
    if len(text.split()) < 2 or len(text) > 300 or _SKIP.match(text) or _SECRET.search(text):
        return None
    if m := _PAYLOAD.match(text):
        text = (m.group("cmd") or m.group("cmd2")).strip(" ,")
        if len(text.split()) < 2:
            return None
    return text


def record(
    text: str,
    intent: str,
    tool: str = "",
    source: str = "",
    tools: tuple[str, ...] = (),
    *,
    w: float | None = None,
    verified: bool | None = None,
    wrong: tuple[str, ...] | list[str] = (),
    conf: float | None = None,
    margin: float | None = None,
    kind: str = "",
    brain: str = "",
) -> bool:
    """Remember what this request turned out to be. Returns whether it was kept.

    ``wrong`` holds labels ("intent" or "intent:tool") it was *not*; ``kind`` is neo/decide's task
    kind, only ever given for a request that decide carried out successfully. ``brain``: the
    provider that already read this request (neo/providers name) — the only one the teacher may
    ever show it to again (neo/reflex/teacher.py)."""
    if not enabled() or intent not in ("chat", "quick_action", "agent_task"):
        return False
    if _CONTENT_TOOLS & ({tool} | set(tools) | {str(x).partition(":")[2] for x in wrong}):
        return False
    if (text := skeleton(text)) is None:
        return False
    row: dict = {"text": text, "intent": intent, "tool": tool if intent == "quick_action" else "", "source": source}
    row["ts"] = _day(time.time())  # the day, not the moment: when you asked isn't training data
    row["script"] = script(text)
    if w is not None:
        row["w"] = round(float(w), 2)
    if verified is not None:
        row["verified"] = bool(verified)
    if wrong:
        row["wrong"] = [str(x)[:48] for x in wrong][:4]
    if conf is not None:
        row["conf"] = round(float(conf), 3)
    if margin is not None:
        row["margin"] = round(float(margin), 3)
    if kind and _KIND.fullmatch(kind):
        row["kind"] = kind
    if brain and _KIND.fullmatch(brain):
        row["brain"] = brain
    with _lock:
        # A counter, not the time, tells what's new since the last retrain.
        st = _state()
        st["seq"] = row["seq"] = int(st.get("seq", 0)) + 1
        _merge_counts(st)
        _save_state(st)
        path = _path(EXTRA)
        with path.open("a", encoding="utf-8") as f:  # append: O(1), no rewrite on the event loop
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if path.stat().st_size > _COMPACT_BYTES:
            _compact(path)
    return True


def _tier(r: dict) -> int:
    """How much a row teaches, for eviction: corrected > verified the reflex was unsure of >
    verified > teacher-only."""
    if r.get("wrong"):
        return 3
    if r.get("verified"):
        return 2 if r.get("conf") is not None and float(r["conf"]) < _LOW_CONF else 1
    return 0


def _select(rows: list[dict], now: float | None = None) -> list[dict]:
    """Latest verdict per sentence within retention; past _KEEP_ROWS the least useful go first,
    oldest first within a tier. A later guess never overrides a verified outcome."""
    now = time.time() if now is None else now
    latest: dict[str, dict] = {}
    for r in rows:
        days = _KEEP_DAYS if r.get("verified") else _KEEP_GUESS_DAYS
        if float(r.get("ts", 0)) < now - days * 86400:
            continue
        k = norm(r["text"])
        old = latest.get(k)
        if old is not None and old.get("verified") and not r.get("verified"):
            continue
        if old is not None and old.get("wrong") and not r.get("wrong") and (old["intent"], old.get("tool")) == (r["intent"], r.get("tool")):
            r = {**r, "wrong": old["wrong"]}  # the same fix confirmed again: keep what it was not
        latest.pop(k, None)
        latest[k] = r
    keep = list(latest.values())
    if len(keep) > _KEEP_ROWS:
        order = sorted(range(len(keep)), key=lambda i: (_tier(keep[i]), i))
        drop = set(order[: len(keep) - _KEEP_ROWS])
        keep = [r for i, r in enumerate(keep) if i not in drop]
    return keep


def _compact(path: Path) -> None:
    """Rewrite the file as training sees it (see _select)."""
    keep = _select(_read_raw(path))
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep), encoding="utf-8")
    tmp.replace(path)


def tidy(now: float | None = None) -> bool:
    """Hold the file on disk to the retention limits once a day — not only at start-up or when
    it outgrows _COMPACT_BYTES: a menu-bar app runs for weeks. Returns whether it ran."""
    global _tidied_day
    day = _day(time.time() if now is None else now)
    if _tidied_day == day:
        return False
    with _lock:
        if _path(EXTRA).exists():
            _compact(_path(EXTRA))
    _tidied_day = day
    return True


def forget_examples() -> int:
    """Delete every stored request, the counters, the correction memory built from them, and
    the cached features of them. Returns how many."""
    global _erase_gen
    with _lock:
        _erase_gen += 1
        rows = _read_raw(_path(EXTRA))
        n = len(_read(_path(EXTRA)))
        for name in (EXTRA, STATE):
            _path(name).unlink(missing_ok=True)
        _counts.clear()
    try:
        from neo.reflex import memory

        memory.forget()
    except Exception as e:  # noqa: BLE001
        print(f"[reflex] couldn't clear the correction memory: {str(e)[:80]}")
    try:
        from neo.reflex.finetune import drop_from_cache

        drop_from_cache({r["text"] for r in rows})
    except Exception as e:  # noqa: BLE001
        print(f"[reflex] couldn't clear the feature cache: {str(e)[:80]}")
    return n


def _read_raw(path: Path) -> list[dict]:
    out = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(r, dict) and r.get("text") and r.get("intent"):
                out.append(r)
    return out


def _read(path: Path) -> list[dict]:
    """The rows as training sees them: latest verdict per sentence, within the retention limits."""
    return _select(_read_raw(path))


def corrections() -> list[dict]:
    """The stored verified corrections, oldest first (what the instant memory is built from)."""
    return [r for r in _read(_path(EXTRA)) if r.get("wrong") and r.get("verified")]


def _state() -> dict:
    try:
        st = json.loads(_path(STATE).read_text())
        if isinstance(st, dict):
            return st
    except Exception:  # noqa: BLE001
        pass
    return {"trained_at": 0.0, "trained_rows": 0}


def _is_new(r: dict, st: dict) -> bool:
    if "seq" in r:  # recorded by this version: its number says whether training has seen it
        return st.get("trained_seq") is None or int(r["seq"]) > int(st["trained_seq"])
    return float(r.get("ts", 0)) > float(st.get("trained_at", 0.0))  # an older row, with a precise time


def _pending() -> tuple[int, int]:
    st = _state()
    new = [r for r in _read(_path(EXTRA)) if _is_new(r, st)]
    return len(new), sum(1 for r in new if r.get("wrong"))


def pending() -> int:
    """Examples recorded since the last retrain (by counter, so it keeps counting once the file
    is at its size cap, and a corrected label for a known sentence counts too)."""
    return _pending()[0]


def due(now: float | None = None) -> bool:
    now = now or time.time()
    n, fixes = _pending()
    since = now - float(_state().get("trained_at", 0.0))
    return n >= _BURST or (n >= _MIN_NEW and since >= _MIN_GAP_S) or fixes >= _MIN_FIXES


def retrain() -> dict | None:
    """Retrain the heads in this process on the already-loaded Laya, then hot-swap them."""
    global _training
    from neo.reflex import finetune, get_laya, laya_lock

    laya = get_laya()
    if laya is None or _training:
        return None
    _training = True
    started = time.time()
    gen = _erase_gen
    try:
        seq = int(_state().get("seq", 0))  # rows after this one arrive while training: still new
        n_rows = len(_read(_path(EXTRA)))
        t0 = time.time()
        try:
            report = finetune.train(
                finetune.defaults(), agent=laya._agent, lock=laya_lock(), cancelled=lambda: _erase_gen != gen
            )
        except Exception as e:  # noqa: BLE001
            # Count it as a try: the next one waits the usual gap instead of re-running (and
            # re-failing) every few idle minutes.
            _write_state({"trained_at": started, "trained_rows": n_rows, "trained_seq": seq, "last_error": str(e)[:300]})
            print(f"[reflex] retrain failed: {str(e)[:160]}")
            return None
        if report.get("cancelled") or _erase_gen != gen:
            print("[reflex] retrain discarded: NEO's memory was erased while it ran")
            return None
        if report.get("saved"):
            with laya_lock():
                laya.reload_head()
        _write_state({"trained_at": started, "trained_rows": n_rows, "trained_seq": seq, "last": report})
        prev = report.get("previous")
        print(
            f"[reflex] learned from {n_rows} of your requests in {time.time() - t0:.0f}s: intent "
            f"{prev if prev is None else f'{prev:.3f}'} → {report['intent']:.3f}"
            + ("" if report.get("saved") else " (kept the old head)")
        )
        return report
    finally:
        _training = False


def _save_state(st: dict) -> None:
    p = _path(STATE)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, default=str))
    tmp.replace(p)


def _write_state(state: dict) -> None:
    """Replace the training state. The request counter and the daily counters carry over; a
    state that says it trained now covers every request so far unless it names its own point."""
    with _lock:
        cur = _state()
        st = dict(state)
        st.setdefault("seq", int(cur.get("seq", 0)))
        if "trained_at" in st:
            st.setdefault("trained_seq", st["seq"])
        st.setdefault("daily", cur.get("daily", {}))
        _save_state(st)


# ---- daily counters (no text) -------------------------------------------------------------------
def count(name: str, n: int = 1) -> None:
    """Bump today's aggregate counter (corrections, barge-ins, tool-lane fires, unsure decisions,
    memory hits…). Held in RAM and written with the next row or the learn loop's tick."""
    if not enabled():
        return
    with _lock:
        _counts[name] = _counts.get(name, 0) + n


def _merge_counts(st: dict) -> None:
    """Fold the pending counters into `st` (the caller holds _lock)."""
    if not _counts:
        return
    daily = st.setdefault("daily", {})
    today = daily.setdefault(time.strftime("%Y-%m-%d"), {})
    for k, v in _counts.items():
        today[k] = today.get(k, 0) + v
    _counts.clear()
    for day in sorted(daily)[:-_COUNT_DAYS]:
        del daily[day]


def flush_counts() -> None:
    if not enabled():
        return
    with _lock:
        if _counts:
            st = _state()
            _merge_counts(st)
            _save_state(st)


def summary() -> str:
    """One line about the latest day with counters: how often the reflex was corrected or cut
    off, how often it acted alone, how often it was unsure, and what the memory fixed."""
    daily = _state().get("daily") or {}
    if not daily:
        return ""
    day = max(daily)
    c = daily[day]
    req, dec, fixes = c.get("requests", 0), c.get("decisions", 0), c.get("corrections", 0)
    parts = [
        f"{req} requests",
        f"{fixes} corrections" + (f" ({100 * fixes / req:.1f}/100)" if req else ""),
        f"{c.get('barge_ins', 0)} stopped within 3 s",
        f"tool lane {c.get('tool_lane', 0)}",
        f"unsure {100 * c.get('unsure', 0) / dec:.0f}%" if dec else "unsure n/a",
        f"memory hits {c.get('memory_hits', 0)}",
    ]
    return f"[reflex] {day}: " + " · ".join(parts)


async def learn_loop(is_idle) -> None:
    """Background task in the core: settle requests, keep the counters, rebuild the correction
    memory and retrain — the heavy parts only when NEO is idle."""
    try:  # the retention limits hold on disk too, not just in what training reads
        tidy()
    except Exception as e:  # noqa: BLE001
        print(f"[reflex] couldn't tidy the learning file: {str(e)[:80]}")
    if enabled() and (line := summary()):
        print(line)
    idle_since: float | None = None
    while True:
        await asyncio.sleep(60)
        try:
            from neo.reflex import memory, outcomes

            outcomes.tick()  # requests nothing followed up on settle by time, not only by the next one
            flush_counts()
            await asyncio.to_thread(tidy)  # rewrites the file only when the day has changed
            if not is_idle():
                idle_since = None
                continue
            idle_since = idle_since or time.time()
            if enabled():
                memory.rebuild_soon(is_idle)
            if enabled() and time.time() - idle_since >= _IDLE_S and due():
                done = asyncio.get_running_loop().create_future()

                def run(done: asyncio.Future = done) -> None:
                    try:
                        retrain()
                    finally:
                        done.get_loop().call_soon_threadsafe(lambda: done.done() or done.set_result(None))

                # A daemon thread, not the default executor: quitting NEO mid-retrain must not
                # wait minutes for it (the head file is written atomically, so a cut-off run
                # leaves the old head in place).
                threading.Thread(target=run, daemon=True, name="laya-retrain").start()
                await done
                idle_since = None
                if line := summary():
                    print(line)
        except Exception as e:  # noqa: BLE001 — learning must never take NEO down
            print(f"[reflex] retrain failed: {str(e)[:120]}")
