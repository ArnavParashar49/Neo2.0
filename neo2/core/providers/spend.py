"""What a paid brain costs, per day, against a daily budget (decision 49).

Every answer's token usage — prompt and completion as the server counted them, reasoning
included — times the model's price is added to today's total. The day is the local calendar day:
the total starts again at midnight. Crossing 80% of the budget is announced once a day; once the
budget is spent the brain is skipped until midnight (the caller says so and fails over). A total
may overshoot the budget by the one request that crossed it: a request is admitted while
anything is left, since its size is only known once it's answered.

Cerebras keeps its own ledger (neo2.core.providers.cerebras). The other paid providers a user may
cap ("Brains & Keys": settings.daily_budgets, dollars a day per provider id) share the same
:class:`Ledger` through a :class:`Cap`, one file per provider, at the catalogue's prices
(neo2.core.providers.catalog); a model with no known price is counted at that provider's dearest.
A provider without an entry in daily_budgets has no NEO cap, as before. The window prefills $2 a
day only for a paid key saved in it (:func:`prefill`); a key from .env or the environment stays
uncapped (decision 4).

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
import math
import os
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from neo2.core import logger

log = logger(__name__)

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


_OFF = frozenset({"off", "false", "no", "none", "disabled", "never"})


def budget(v: Any, default: float = 0.5) -> float:
    """The daily budget in dollars from a setting (NEO_CEREBRAS_DAILY_BUDGET): a number ≥ 0 ("$1.25"
    too; 0: never spend). Only an unset (empty) value is `default`: the cap fails closed, so "off",
    a negative number or a typo is 0 — Cerebras off, and the log says why — never a day of paid
    calls the user didn't mean to allow. A typo still never stops NEO from starting."""
    if v is None or (isinstance(v, str) and not v.strip()):
        return default
    if isinstance(v, str):
        if v.strip().lower() in _OFF:
            log.info("[spend] NEO_CEREBRAS_DAILY_BUDGET is off: Cerebras isn't used")
            return 0.0
        v = v.strip().lstrip("$").strip()
    try:
        f = float(v)
    except (TypeError, ValueError):
        f = math.nan
    if math.isfinite(f) and f >= 0:
        return f
    log.warning("[spend] NEO_CEREBRAS_DAILY_BUDGET isn't a dollar amount of 0 or more: Cerebras stays off")
    return 0.0


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


# ---- caps for the other paid providers ("Brains & Keys", decision 4) ---------------------------------

# Dollars a day the window fills in for a paid key the user saves in it.
DEFAULT_CAP = 2.0
# The providers a daily_budgets entry caps here. Cerebras has its own ledger (its entry replaces
# NEO_CEREBRAS_DAILY_BUDGET: cerebras_budget); Gemini and Groq are free tiers; NEO's local model is free.
CAPPED = ("anthropic", "openai", "openrouter")
# A model NEO knows no price for, at a provider whose catalogue lists none (OpenRouter, an endpoint
# the user added): the dearest price the catalogue knows, so a cap never counts low.
_NO_PRICE = "(unknown model)"


def capped(pid: str) -> bool:
    """`pid`'s spend is counted against its daily_budgets entry here (Cerebras: by its own ledger)."""
    return pid in CAPPED or pid.startswith("custom:")


def cerebras_budget(s: Any, default: float = 0.5) -> float:
    """Cerebras' daily budget: daily_budgets["cerebras"] when the user set one in Brains & Keys,
    else NEO_CEREBRAS_DAILY_BUDGET (as before)."""
    caps = getattr(s, "daily_budgets", None) or {}
    v = caps["cerebras"] if isinstance(caps, Mapping) and "cerebras" in caps else getattr(s, "cerebras_daily_budget", default)
    return budget(v, default)


def prefill(pid: str, source: str, *, on_device: bool = False) -> float | None:
    """The daily cap the Brains & Keys window fills in when a key is saved: $2 (:data:`DEFAULT_CAP`)
    for a paid provider whose key was saved in the window (`source` "saved"); None — no cap — for a
    key from a .env file or the environment, a free provider, one on this Mac, and Cerebras (its
    budget, NEO_CEREBRAS_DAILY_BUDGET, is $0.50 a day already)."""
    if source != "saved" or not capped(pid):
        return None
    if pid.startswith("custom:"):
        return None if on_device else DEFAULT_CAP
    from neo2.core.providers import catalog

    p = catalog.provider(pid)
    return DEFAULT_CAP if p is not None and p.cost == "paid" else None


def provider_prices(pid: str) -> dict[str, tuple[float, float]]:
    """$/M tokens per model of `pid`, from the catalogue. A provider with none listed (OpenRouter, an
    endpoint) counts every model at the dearest price the catalogue knows."""
    from neo2.core.providers import catalog

    p = catalog.provider(pid)
    own = {m.id: m.price for m in (p.models if p else ()) if m.price}
    if own:
        return own
    dearest = [m.price for q in catalog.PROVIDERS.values() for m in q.models if m.price]
    return {_NO_PRICE: (max(x[0] for x in dearest), max(x[1] for x in dearest))} if dearest else {_NO_PRICE: (10.0, 50.0)}


def _file(pid: str) -> str:
    return "spend-" + "".join(c if c.isalnum() or c == "-" else "-" for c in pid) + ".json"


class Cap:
    """One paid provider's daily cap: its ledger, and what is said at 80% and once it's spent."""

    def __init__(
        self,
        provider: str,
        daily_budget: float,
        *,
        path: Path | None,
        label: str,
        price_table: Mapping[str, tuple[float, float]] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.provider = provider
        self.label = label
        self.ledger = Ledger(
            path, daily_budget=daily_budget, price_table=price_table or provider_prices(provider), clock=clock
        )
        self._heard: list[str] = []

    def words(self) -> str:
        b = self.ledger.daily_budget
        return f"{self.label}'s daily budget of {_usd(b)} is used up" if b > 0 else f"{self.label}'s daily budget is $0"

    def refusal(self, name: str) -> Any:
        """QuotaExhausted until midnight once today's budget is spent (None: it may be asked)."""
        if not self.ledger.over():
            return None
        from neo2.core.providers.base import QuotaExhausted

        e = QuotaExhausted(f"{name}: today's budget is spent", until=self.ledger.midnight())
        e.words = self.words()  # type: ignore[attr-defined] — what the user hears (failure_words)
        if self.ledger.daily_budget > 0 and self.ledger.mark_said("spent"):
            self._heard.append(f"{self.words()}; I'll skip it until tomorrow")
        return e

    def add(self, model: str, tokens_in: int, tokens_out: int) -> None:
        crossed = self.ledger.add(model, tokens_in, tokens_out)
        b = self.ledger.daily_budget
        if "warn" in crossed and "spent" not in crossed:
            self._heard.append(f"{self.label} has used {_usd(self.ledger.today())} of today's {_usd(b)} budget")
        if "spent" in crossed:
            self.ledger.mark_said("spent")
            self._heard.append(f"{self.words()}; I'll skip it until tomorrow")

    async def tell(self) -> None:
        """Show what's to be said (80% used, all of it) on the agent's own bus, once each."""
        while self._heard:
            text = self._heard.pop(0)
            log.warning(f"[spend] {text}")
            try:
                from neo2.core.events import bus

                await bus().note(text)
            except Exception as e:  # noqa: BLE001 — a note must never fail the answer
                log.info(f"[spend] couldn't show a note: {type(e).__name__}")


def cap_for(pid: str, s: Any = None) -> Cap | None:
    """`pid`'s cap from the settings' daily_budgets (None: no entry, or not a provider capped here).
    The ledger lives in the settings' data dir, one file per provider."""
    if not capped(pid):
        return None
    if s is None:
        from neo2.core.config import settings

        s = settings()
    caps = getattr(s, "daily_budgets", None) or {}
    if not isinstance(caps, Mapping) or pid not in caps:
        return None
    from neo2.core.providers import catalog

    dd = getattr(s, "data_dir", None)
    label = catalog.provider_label(pid, s)
    return Cap(pid, budget(caps[pid], 0.0), path=Path(dd) / _file(pid) if dd else None, label=label)


def _usd(x: float) -> str:
    """Dollars as said: "$0.42" (a budget under a cent keeps its digits: "$0.0010")."""
    return f"${x:.2f}" if x == 0 or x >= 0.01 else f"${x:.4f}"
