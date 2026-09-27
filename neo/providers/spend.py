"""What a paid brain costs, per day, against a daily budget (decision 49).

Every answer's token usage — prompt and completion as the server counted them, reasoning
included — times the model's price is added to today's total. The day is the local calendar day:
the total starts again at midnight. Crossing 80% of the budget is announced once a day; once the
budget is spent the brain is skipped until midnight (the caller says so and fails over). A total
may overshoot the budget by the one request that crossed it: a request is admitted while
anything is left, since its size is only known once it's answered.

A refusal that no retry can fix — an invalid key, an account out of credits — is kept here too,
until midnight, so a restart doesn't pay the same failed round trip again. It is tied to the key
it was for (a short fingerprint, never the key): a new key is tried at once.

The ledger is a small JSON file in the agent's data dir (NEO.app's ~/.neo; an embedded agent's
own folder), readable only by this user, re-read before every change so two NEO processes sharing
a folder add up rather than overwrite each other. With no path (a host's own provider object) it
lives in memory only. It never holds a key, a prompt or an answer: dates, dollars, token counts.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import math
import os
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# US dollars per million tokens (input, output), as inference-docs.cerebras.ai lists them (checked
# 2026-09-27). The setting (NEO_CEREBRAS_PRICES) overrides or adds models.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-oss-120b": (0.35, 0.75),
    "qwen-3.8-27b": (0.99, 1.49),
}
WARN_AT = 0.8  # of the day's budget: said once a day
FILE = "cerebras_spend.json"
_VERSION = 1


def fingerprint(key: str) -> str:
    """A key's short, one-way tag: which key a refusal was for, without keeping the key."""
    return hashlib.sha256(("neo-spend:" + key).encode()).hexdigest()[:10] if key else ""


def prices(extra: Any = None) -> dict[str, tuple[float, float]]:
    """The price table: the published prices, with the setting's entries on top. The setting is a
    mapping or JSON, model → [input, output] or {"input": …, "output": …} in $/M tokens; an entry
    that isn't two non-negative numbers is ignored (a typo never stops NEO from starting)."""
    out = dict(PRICES)
    if isinstance(extra, str):
        try:
            extra = json.loads(extra) if extra.strip() else {}
        except ValueError:
            log.warning("[spend] the price setting isn't JSON: using the published prices")
            extra = {}
    if not isinstance(extra, Mapping):
        return out
    for model, v in extra.items():
        if isinstance(v, Mapping):
            v = (v.get("input"), v.get("output"))
        try:
            pin, pout = (float(x) for x in v)  # type: ignore[union-attr]
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(x) and x >= 0 for x in (pin, pout)):
            out[str(model)] = (pin, pout)
    return out


def budget(v: Any, default: float = 0.5) -> float:
    """The daily budget in dollars from a setting: a number ≥ 0 (0: never spend), else the default."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) and f >= 0 else default


class Ledger:
    """Today's spend on one account's brains, against `daily_budget` dollars."""

    def __init__(
        self,
        path: Path | None,
        *,
        daily_budget: float,
        price_table: Mapping[str, tuple[float, float]] | None = None,
        key: str = "",
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.daily_budget = budget(daily_budget)
        self.prices = dict(price_table if price_table is not None else PRICES)
        self.account = fingerprint(key)
        self.clock = clock
        self._lock = threading.Lock()
        self._state: dict[str, Any] = self._fresh()
        self._unknown: set[str] = set()
        self._warned_unreadable = False
        self._unsaved = False  # the last save failed: this object's state is the newer one

    # ---- the day --------------------------------------------------------------------------
    def day(self) -> str:
        return time.strftime("%Y-%m-%d", time.localtime(self.clock()))

    def midnight(self) -> float:
        """The next local midnight (epoch seconds): when today's total and refusals end."""
        now = _dt.datetime.fromtimestamp(self.clock())
        nxt = (now + _dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return nxt.timestamp()

    def _fresh(self) -> dict[str, Any]:
        return {"version": _VERSION, "day": self.day(), "usd": 0.0, "tokens": {}, "said": [], "refused": {}}

    # ---- the file -------------------------------------------------------------------------
    def _load(self) -> dict[str, Any]:
        """Today's state: from the file when there is one (another process may have added to it),
        else what this object has. Another day's state starts over."""
        st = self._state
        if self.path is not None and not self._unsaved:
            try:
                raw = json.loads(self.path.read_text())
                if isinstance(raw, dict) and raw.get("version") == _VERSION:
                    st = raw
            except FileNotFoundError:
                pass
            except (OSError, ValueError) as e:
                if not self._warned_unreadable:  # (once: it's read before every request)
                    self._warned_unreadable = True
                    log.warning(
                        f"[spend] {self.path.name} unreadable ({type(e).__name__}): today's total starts over"
                    )
        if st.get("day") != self.day():
            st = self._fresh()
        try:
            st["usd"] = max(0.0, float(st.get("usd", 0.0)))
        except (TypeError, ValueError):
            st["usd"] = 0.0
        if not isinstance(st.get("tokens"), dict):
            st["tokens"] = {}
        if not isinstance(st.get("said"), list):
            st["said"] = []
        if not isinstance(st.get("refused"), dict):
            st["refused"] = {}
        self._state = st
        return st

    def _save(self, st: dict[str, Any]) -> None:
        self._state = st
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                json.dump(st, f)
            os.replace(tmp, self.path)
            self._unsaved = False
        except OSError as e:  # a full disk must not fail the answer that was just paid for
            self._unsaved = True
            log.warning(f"[spend] couldn't save {self.path.name}: {type(e).__name__}")

    # ---- spending -------------------------------------------------------------------------
    def price(self, model: str) -> tuple[float, float]:
        """$/M tokens for `model`; one not in the table is charged at the dearest known price."""
        if model in self.prices:
            return self.prices[model]
        if model not in self._unknown:
            self._unknown.add(model)
            log.warning(f"[spend] no price for {model}: counting it at the dearest known price")
        return (
            max((p[0] for p in self.prices.values()), default=1.0),
            max((p[1] for p in self.prices.values()), default=1.0),
        )

    def cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        pin, pout = self.price(model)
        return (max(0, tokens_in) * pin + max(0, tokens_out) * pout) / 1_000_000

    def add(self, model: str, tokens_in: int, tokens_out: int) -> list[str]:
        """Count one answer. Returns what it crossed today, each only once a day: "warn" (80% of
        the budget) and "spent" (all of it)."""
        usd = self.cost(model, tokens_in, tokens_out)
        with self._lock:
            st = self._load()
            before = st["usd"]
            st["usd"] = before + usd
            t = st["tokens"].setdefault(model, [0, 0])
            t[0], t[1] = int(t[0]) + max(0, tokens_in), int(t[1]) + max(0, tokens_out)
            crossed = []
            if self.daily_budget > 0:
                if st["usd"] >= WARN_AT * self.daily_budget and "warn" not in st["said"]:
                    crossed.append("warn")
                if st["usd"] >= self.daily_budget and "spent" not in st["said"]:
                    crossed.append("spent")
            st["said"] = [*st["said"], *crossed]
            self._save(st)
        return crossed

    def today(self) -> float:
        """Dollars spent today."""
        with self._lock:
            return self._load()["usd"]

    def over(self) -> bool:
        """Today's budget is spent (a budget of 0: nothing may be spent)."""
        return self.today() >= self.daily_budget

    def mark_said(self, what: str) -> bool:
        """True the first time today `what` is said (a note shown once a day, not per request)."""
        with self._lock:
            st = self._load()
            if what in st["said"]:
                return False
            st["said"] = [*st["said"], what]
            self._save(st)
            return True

    # ---- refusals -------------------------------------------------------------------------
    def refuse(self, scope: str, words: str) -> float:
        """Remember until midnight that `scope` ("account", or a model) is refused, and why (the
        words the user hears). Returns when it ends."""
        until = self.midnight()
        with self._lock:
            st = self._load()
            refused = dict(st["refused"])
            refused[scope] = {"account": self.account, "words": words, "until": until}
            st["refused"] = refused
            self._save(st)
        return until

    def refusal(self, scope: str) -> tuple[str, float] | None:
        """(words, until) while `scope` is refused today for this key, else None."""
        with self._lock:
            st = self._load()
        r = st["refused"].get(scope)
        if not isinstance(r, dict) or r.get("account") != self.account:
            return None
        try:
            until = float(r.get("until", 0.0))
        except (TypeError, ValueError):
            return None
        return (str(r.get("words", "")), until) if until > self.clock() else None
