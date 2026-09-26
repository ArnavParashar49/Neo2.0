"""Carrying out a task kind: the loop every kind shares, and what NEO learns from each attempt.

A driver knows how to do one kind with one app (or route) and method; this module decides the
order, the budgets and when to stop, generalising :func:`neo.decide.media.play_best`:

* :func:`run_best` - reversible and read kinds: the engine's candidates in order, each method
  behind its permission gate, until one works;
* :func:`run_plan` - sends: only the app and methods the user approved on a card.

**At most once.** An attempt that may already have created or sent something (it passed its
commit point, see :func:`commit`, and then timed out or was cancelled, or it says so with
``Effect.UNKNOWN``) ends the loop, and the same add asked again within RECENT_S gets the same
answer: NEO never makes a second copy or sends twice.

**Nothing learned after an erase.** :func:`begin_request` notes the privacy generation; every
write this module (and the engine's preferences and the learned stores) makes is dropped once the
user has erased NEO's memory since.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from neo.agent.registry import ToolOutput
    from neo.decide.engine import Candidate
    from neo.decide.inventory import App

__all__ = [
    "Effect",
    "Outcome",
    "Req",
    "Result",
    "Recipient",
    "Plan",
    "Driver",
    "run_best",
    "run_plan",
    "commit",
    "begin_request",
    "erased",
    "record",
    "remember_last",
    "last",
    "idempotent",
    "to_output",
    "forget",
]

RECENT_S = 120.0  # a repeat of the same add, or a correction, counts within this window
# Reasons that end the loop: NEO asked the user something, or the app says there is nothing to find.
_STOP = frozenset({"ambiguous", "needs_user", "empty"})
# The reason worth saying when nothing worked, most useful first.
_USEFUL = ("permission", "login", "offline", "not_found", "empty", "timeout")


class Effect(Enum):
    NONE = 0  # nothing changed: safe to try the next method or app
    PARTIAL = 1  # a harmless empty artifact (a blank note window): may try the next; said only if all fail
    DONE = 2  # created / sent / read, and verified (or handed off: receipt["handoff"])
    UNKNOWN = 3  # may have been created or sent (past the commit point): stop, never a second copy


@dataclass
class Outcome:
    ok: bool
    text: str  # one short English line (Live relays it in the user's language)
    # '', permission, login, offline, not_found, timeout, unconfirmed, empty, ambiguous, needs_user,
    # declined, route_bad, handoff
    reason: str = ""
    effect: Effect = Effect.NONE
    receipt: dict = field(default_factory=dict)  # id, where, container, verified, handoff
    say: bool | None = None
    data: str = ""  # read kinds: the information itself (the tool fences it as untrusted)


@dataclass
class Req:
    kind: str
    fields: dict[str, Any]  # the tool's parsed args (schema keys only)
    text: str  # the user's words (a named app, a site hint, what Laya reads)
    about: str = ""  # hashed contact id (sends)
    named: str = ""  # the app named through the tool's `app` arg
    sig: tuple = ()  # what makes two adds the same one (idempotency); () = never deduplicated
    also: tuple[tuple, ...] = ()  # more keys for the same add (a time as said, beside sig's resolved one)


@dataclass
class Result:
    text: str
    ok: bool
    say: bool | None = None  # True: speak even a quiet tool's result; False: keep a success silent
    unsure: bool = False  # not ok, yet it may have happened (Effect.UNKNOWN): nobody tries it again


@dataclass(frozen=True)
class Recipient:
    display: str  # what the card and the summary say (a name; never a handle from Contacts)
    source: Literal["contacts", "literal", "spoken"]
    contact_id: str = ""
    label: str = ""
    handle: str = ""  # the phone number / address, only ever shown in the card's detail
    kind: str = ""  # phone | email | im


@dataclass(frozen=True)
class Plan:
    """A send the user is asked to approve, staged by NEO itself (never by a model's arguments):
    the card is a pure function of it, and the approved call runs exactly this."""

    v: int
    kind: str  # send_message | send_email
    app_key: str
    app_name: str
    methods: tuple[str, ...]
    to: tuple[Recipient, ...]
    cc: tuple[Recipient, ...] = ()
    bcc: tuple[Recipient, ...] = ()
    tried: tuple[str, ...] = ()  # apps tried under earlier cards
    about: str = ""  # sha256(contact_id)[:12]
    tainted_by: str = ""  # frozen at staging, so the re-invoke rebuilds the same card
    dup_note: str = ""  # "You sent this exact message to Priya 40 s ago." (frozen)
    header: str = ""  # set only on a header re-card

    def as_param(self) -> dict:
        return asdict(self)

    @classmethod
    def from_param(cls, d: Any) -> Plan:
        """The plan from its staged form; ValueError for anything malformed."""
        if not isinstance(d, dict):
            raise ValueError("plan: not a mapping")

        def text(k: str) -> str:
            v = d.get(k, "")
            if not isinstance(v, str):
                raise ValueError(f"plan: {k} is not text")
            return v

        def people(k: str) -> tuple[Recipient, ...]:
            rows = d.get(k) or []
            if not isinstance(rows, list | tuple):
                raise ValueError(f"plan: {k} is not a list")
            out = []
            for r in rows:
                if not isinstance(r, dict) or r.get("source") not in ("contacts", "literal", "spoken"):
                    raise ValueError(f"plan: bad recipient in {k}")
                vals = {f: r.get(f, "") for f in ("display", "contact_id", "label", "handle", "kind")}
                if not all(isinstance(v, str) for v in vals.values()):
                    raise ValueError(f"plan: bad recipient in {k}")
                out.append(Recipient(source=r["source"], **vals))
            return tuple(out)

        methods, tried = d.get("methods"), d.get("tried") or []
        if not isinstance(methods, list | tuple) or not all(isinstance(m, str) for m in methods):
            raise ValueError("plan: methods")
        if not isinstance(tried, list | tuple) or not all(isinstance(t, str) for t in tried):
            raise ValueError("plan: tried")
        if not isinstance(d.get("v"), int) or not text("app_key") or not methods:
            raise ValueError("plan: incomplete")
        return cls(
            v=d["v"],
            kind=text("kind"),
            app_key=text("app_key"),
            app_name=text("app_name"),
            methods=tuple(methods),
            to=people("to"),
            cc=people("cc"),
            bcc=people("bcc"),
            tried=tuple(tried),
            about=text("about"),
            tainted_by=text("tainted_by"),
            dup_note=text("dup_note"),
            header=text("header"),
        )


class Driver(Protocol):
    KIND: str
    OVERALL_S: float
    BUDGET_S: dict[str, float]  # per method

    def methods_for(self, app: App, kind: str | None = None) -> tuple[str, ...]: ...

    def accepts(self, app: App, method: str, req: Req) -> bool: ...

    def describe(self, app: App) -> str: ...  # capability words for Laya ("can script send")

    # None: nothing to go by (a hint that isn't there), not an attempt
    async def attempt(self, req: Req, cand: Candidate, method: str) -> Outcome | None: ...

    async def fallback(self, req: Req, key: str) -> Outcome | None: ...


# ---- one request's state ---------------------------------------------------------------------

_gen: ContextVar[int | None] = ContextVar("neo_run_generation", default=None)


def begin_request() -> None:
    """Note which memory this request started with (see :func:`erased`)."""
    from neo import privacy

    _gen.set(privacy.generation())


def erased() -> bool:
    """The user erased NEO's memory after this request started: learn nothing from it."""
    from neo import privacy

    started = _gen.get()
    return started is not None and started != privacy.generation()


class _Attempt:
    committed = False


_attempt: ContextVar[_Attempt | None] = ContextVar("neo_run_attempt", default=None)


async def commit[T](aw: Awaitable[T]) -> T:
    """The one action that creates or sends. From here on the attempt may have had its effect, so
    a timeout or a cancel means UNKNOWN (never a second try), and the action itself runs to its
    end even when the attempt is cancelled."""
    a = _attempt.get()
    if a is not None:
        a.committed = True
    return await asyncio.shield(aw)


# ---- what is remembered (in memory; learning goes to the engine) ---------------------------------

_last: dict[str, tuple[str, str, float]] = {}  # kind -> (key, about, when)
_receipts: dict[tuple[str, tuple], tuple[float, dict]] = {}  # (kind, sig) -> (when, receipt)


def record(kind: str, key: str, method: str, out: Outcome) -> None:
    """One attempt, to the engine's track record: a success; UNKNOWN as unconfirmed (no blame);
    anything else blamed unless its reason isn't the app's fault."""
    if erased():
        return
    from neo.decide import engine

    try:
        if out.ok:
            engine.record(kind, key, method, True)
        elif out.effect is Effect.UNKNOWN:
            engine.record(kind, key, method, False, "unconfirmed")
        else:
            engine.record(kind, key, method, False, out.reason)
    except Exception as e:  # noqa: BLE001 - learning must never break the task
        print(f"[decide] couldn't record the outcome ({str(e)[:60]})")


def remember_last(kind: str, key: str, about: str = "") -> None:
    """The app a task of this kind just went to, for a correction that follows ("no, in <app>")."""
    if not erased():
        _last[kind] = (key, about, time.monotonic())


def last(kind: str, within_s: float = RECENT_S) -> tuple[str, str] | None:
    """(key, about) of this kind's last success, if it was within ``within_s``."""
    hit = _last.get(kind)
    return (hit[0], hit[1]) if hit and time.monotonic() - hit[2] <= within_s else None


def idempotent(kind: str, sig: tuple) -> dict | None:
    """The receipt of the same add within the last RECENT_S (a repeated or retried request);
    ``{"unconfirmed": <what was said>}`` when that add may or may not have happened."""
    hit = _receipts.get((kind, sig))
    return hit[1] if hit and time.monotonic() - hit[0] <= RECENT_S else None


def _remember_receipt(kind: str, sigs: list[tuple], receipt: dict) -> None:
    now = time.monotonic()
    for k in [k for k, (t, _r) in _receipts.items() if now - t > RECENT_S]:
        del _receipts[k]
    if not erased():
        for sig in sigs:
            _receipts[(kind, sig)] = (now, dict(receipt))


def forget() -> None:
    """The privacy erase: receipts and last actions go (nothing of them is on disk)."""
    _last.clear()
    _receipts.clear()


# ---- the loops -------------------------------------------------------------------------------------


def _who(cand: Candidate) -> str:
    return cand.app.name if cand.app is not None else cand.key


async def _gate(app: App | None, method: str) -> Outcome | None:
    """None when the method may run: scripting needs macOS's Automation OK for that app (asked
    here, where NEO can wait for the answer), the UI needs Accessibility."""
    if app is None or app.kind == "route":
        return None  # a framework route checks its own access
    if method.startswith("script"):
        from neo.decide import aescript

        return await aescript.control_gate(app)
    if method.startswith("ax"):
        from neo.decide import axkit

        if not axkit.trusted():
            return Outcome(
                False,
                "Accessibility access is off for NEO: System Settings › Privacy & Security › Accessibility",
                reason="permission",
            )
    return None


async def _attempt_once(
    driver: Driver, req: Req, cand: Candidate, method: str, budget: float, unknown: str
) -> tuple[Outcome | None, bool]:
    """One attempt within ``budget`` and whether it passed its commit point; None when a
    fallback doesn't apply. Past the commit point a timeout or a cancel is UNKNOWN."""
    a = _Attempt()
    token = _attempt.set(a)
    try:
        if cand.app is None:
            return await asyncio.wait_for(driver.fallback(req, cand.key), budget), a.committed
        return await asyncio.wait_for(driver.attempt(req, cand, method), budget), a.committed
    except TimeoutError:
        if a.committed:
            return Outcome(False, unknown, reason="unconfirmed", effect=Effect.UNKNOWN, say=True), True
        return Outcome(False, f"{_who(cand)} took too long", reason="timeout"), False
    except asyncio.CancelledError:
        if a.committed:  # the user said stop after it went out: say so rather than nothing
            return Outcome(False, unknown, reason="unconfirmed", effect=Effect.UNKNOWN, say=True), True
        raise
    except Exception as e:
        if not a.committed:
            raise
        # it broke after the action went out (a read-back, say): the item may exist all the same
        print(f"[decide] {_who(cand)} ({method}) broke past its commit ({type(e).__name__}: {str(e)[:60]})")
        return Outcome(False, unknown, reason="unconfirmed", effect=Effect.UNKNOWN, say=True), True
    finally:
        _attempt.reset(token)


def _speak(out: Outcome, first: bool, habitual: bool) -> bool | None:
    """Silent when it went where this kind always goes; one line on first use or after a fallback."""
    if out.say is not None:
        return out.say
    if out.data:
        return None  # read kinds: the information is the answer
    return not (first and habitual)


async def _note(trail: list[str]) -> None:
    """The attempt trail: to the log and the overlay, never to speech."""
    if not trail:
        return
    line = "Tried: " + "; ".join(trail)
    print(f"[decide] {line}")
    try:
        from neo.events import bus

        await bus().note(line[:300])
    except Exception:  # noqa: BLE001 - a note is a nicety
        pass


def _one_reason(outs: list[Outcome]) -> Outcome | None:
    for reason in _USEFUL:
        if hit := next((o for o in outs if o.reason == reason), None):
            return hit
    return outs[-1] if outs else None


async def run_best(driver: Driver, req: Req, *, max_attempts: int = 6) -> Result:
    """Do ``req`` with the best candidate that works, trying the next on a clean failure."""
    from neo.decide import engine

    begin_request()
    sigs = [s for s in (req.sig, *req.also) if s]
    if (seen := next((r for s in sigs if (r := idempotent(req.kind, s)) is not None), None)) is not None:
        if said := seen.get("unconfirmed"):  # a retry of an add that may have happened: not again
            return Result(said, False, say=True, unsure=True)
        return Result("Already added a moment ago.", True, say=True)
    t0 = time.monotonic()
    try:
        cands = list(await engine.choose(req.kind, req.text, about=req.about))
    except Exception as e:  # noqa: BLE001
        print(f"[decide] choosing failed ({str(e)[:80]})")
        cands = []
    outs: list[Outcome] = []
    trail: list[str] = []
    attempts = 0
    for c, m in [(cand, m) for cand in cands for m in cand.methods]:
        remaining = driver.OVERALL_S - (time.monotonic() - t0)
        if remaining < 1.0 or attempts >= max_attempts:
            break
        if c.app is not None and not driver.accepts(c.app, m, req):
            continue
        asked_at = time.monotonic()
        gate = await _gate(c.app, m)
        t0 += time.monotonic() - asked_at  # waiting for macOS's question isn't the task's time
        if gate is not None:
            record(req.kind, c.key, m, gate)
            outs.append(gate)
            trail.append(f"{_who(c)} ({m}): {gate.text}")
            continue
        habitual = engine.habitual(req.kind, c.key)  # before this attempt is learned
        budget = min(driver.BUDGET_S.get(m, remaining), remaining)
        unknown = f"I may have done it in {_who(c)} but couldn't check. Have a look there."
        # past its commit point an attempt's own effect says whether anything happened (a script
        # whose object never resolved did nothing), so only UNKNOWN stops the loop here
        out, _ = await _attempt_once(driver, req, c, m, budget, unknown)
        if out is None:
            continue  # a fallback that doesn't apply, or a hint with nothing to show: not an attempt
        attempts += 1
        record(req.kind, c.key, m, out)
        if out.ok:
            remember_last(req.kind, c.key, req.about)
            _remember_receipt(req.kind, sigs, out.receipt)
            await _note(trail)
            return Result(out.data or out.text, True, say=_speak(out, not outs, habitual))
        outs.append(out)
        trail.append(f"{_who(c)} ({m}): {out.text}")
        if out.effect is Effect.UNKNOWN or out.reason in _STOP:
            unsure = out.effect is Effect.UNKNOWN
            if unsure:
                _remember_receipt(req.kind, sigs, {"unconfirmed": out.text})
            await _note(trail)
            return Result(out.text, False, say=True if out.say is None else out.say, unsure=unsure)
    await _note(trail)
    if (why := _one_reason(outs)) is None:
        return Result("Nothing on this Mac can do that.", False, say=True)
    return Result(why.text, False, say=True)


async def run_plan(driver: Driver, plan: Plan, req: Req) -> Outcome:
    """Carry out an approved send: only the approved app, only its planned methods, in order.
    After a commit there is never another method; a clean failure of every method is returned
    for the caller to offer the next app on a new card."""
    from neo.decide import engine, inventory

    begin_request()
    app = inventory.app_for_bundle(plan.app_key) or await asyncio.to_thread(inventory.app_by_id, plan.app_key)
    if app is None:
        return Outcome(False, f"{plan.app_name} isn't on this Mac any more", reason="not_found")
    cand = engine.Candidate(app, 0.0, 0.0, "approved by the user", plan.methods)
    last_out: Outcome | None = None
    for m in plan.methods:
        if not driver.accepts(app, m, req):
            continue
        gate = await _gate(app, m)
        if gate is not None:
            record(plan.kind, plan.app_key, m, gate)
            last_out = gate
            continue
        budget = driver.BUDGET_S.get(m, driver.OVERALL_S)
        out, committed = await _attempt_once(driver, req, cand, m, budget, "I may already have sent it.")
        if out is None:
            continue
        record(plan.kind, plan.app_key, m, out)
        if out.ok:
            remember_last(plan.kind, plan.app_key, plan.about)
        if out.ok or out.effect is Effect.UNKNOWN or committed or out.reason in _STOP:
            return out
        last_out = out
    return last_out or Outcome(False, f"{plan.app_name} couldn't do it", reason="route_bad")


def to_output(r: Result) -> ToolOutput:
    """A driver's result as the tool's output (a NEEDS_USER / NEEDS_CONFIRM halt stays one). A
    question for the user and an add that may have happened are no failure: nobody (a fast path
    handing over to the agent, a model, Live) should try them again."""
    from neo.agent.confirm import Halt
    from neo.agent.registry import ToolOutput

    halt = isinstance(r.text, Halt)
    out = ToolOutput(str(r.text), ok=r.ok or r.unsure or halt, halt=halt)
    out.say = r.say  # the session's speech policy reads it (ToolOutput.say)
    return out
