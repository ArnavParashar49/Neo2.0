"""The time an agent run has left, as the brains see it.

A run has a wall-clock budget (settings.max_seconds, 180 s), checked by the agent loop between
steps. Inside a step the chain used to be free to wait: one brain call up to its request timeout, a
rate-limit pause of up to 65 s on top — so a run on Groq's vision model (8K tokens a minute, ~30–40 s
per picture step) could end a minute past its limit. With the run's clock bound here, the chain
(neo.providers.Chain) asks no brain whose wait alone is longer than the time left, cuts a call off
when the time is up, and ends the step with the reason instead (DECISIONS 43: a budget-aware stop).

Time the user holds the run (NEO's browser while they drive it: the loop's `paused()`) doesn't
count, as in the loop's own check. Nothing is bound outside an agent run: a chain used on its own
(a chat, a teacher round) keeps its old behaviour.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterator
from contextvars import ContextVar

_left: ContextVar[Callable[[], float] | None] = ContextVar("neo_run_left", default=None)
_limit: ContextVar[float] = ContextVar("neo_run_limit", default=0.0)


@contextlib.contextmanager
def run(left: Callable[[], float], *, limit_s: float = 0.0) -> Iterator[None]:
    """Bind the run's clock for what runs inside: `left()` is the seconds it has left (≤0: none).
    `limit_s`: the run's whole budget, for the words the user hears."""
    t1, t2 = _left.set(left), _limit.set(limit_s)
    try:
        yield
    finally:
        _left.reset(t1)
        _limit.reset(t2)


@contextlib.contextmanager
def seconds(total: float, *, paused: Callable[[], float] = lambda: 0.0,
            clock: Callable[[], float] = time.monotonic) -> Iterator[None]:  # fmt: skip
    """`run` for a budget of `total` seconds from now (`paused()`: seconds the user held it so far)."""
    t0, p0 = clock(), paused()
    with run(lambda: total - (clock() - t0 - (paused() - p0)), limit_s=total):
        yield


def left() -> float | None:
    """Seconds the bound run has left, None when no run is bound."""
    fn = _left.get()
    if fn is None:
        return None
    try:
        return float(fn())
    except Exception:  # noqa: BLE001 — a broken clock never stops a run
        return None


def limit() -> float:
    """The bound run's whole budget in seconds (0 when unknown)."""
    return _limit.get()
