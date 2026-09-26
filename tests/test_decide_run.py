"""neo.decide.run: the loop every kind shares. A scripted fake driver stands in for the kinds'
drivers (none exists yet) and the engine's choice is fixed per test; outcomes are learned into
the test's own data dir. Nothing is launched or scripted."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from neo import privacy
from neo.decide import engine, inventory, media, run
from neo.decide.engine import Candidate
from neo.decide.inventory import App
from neo.decide.run import Effect, Outcome, Plan, Recipient, Req


@pytest.fixture(autouse=True)
def _own_request():
    """begin_request() called outside a task sets the test's own context: undo it afterwards.
    Receipts and last actions are process-wide memory: none carries between tests."""
    token = run._gen.set(None)
    run.forget()
    yield
    run.forget()
    run._gen.reset(token)

NOTEBOOK = App(bundle_id="test.notebook", name="Notebook", path="/Applications/Notebook.app")
JOTTER = App(bundle_id="test.jotter", name="Jotter", path="/Applications/Jotter.app")


def cand(app: App | None, *methods: str) -> Candidate:
    return Candidate(app, 0.5, 0.0, "", methods)


class FakeDriver:
    KIND = "write_note"

    def __init__(self, script: dict, *, overall: float = 20.0, budget: float = 2.0, refuse=(), fallback=None):
        self.script, self.calls, self.refuse, self.fb = script, [], set(refuse), fallback
        self.OVERALL_S = overall
        self.BUDGET_S = {m: budget for (_k, m) in script}

    def methods_for(self, app, kind=None):
        return tuple(m for (k, m) in self.script if k == app.bundle_id)

    def accepts(self, app, method, req):
        return (app.bundle_id, method) not in self.refuse

    def describe(self, app):
        return ""

    async def attempt(self, req, c, method):
        self.calls.append((c.key, method))
        got = self.script[(c.key, method)]
        return await got() if callable(got) else got

    async def fallback(self, req, key):
        self.calls.append((key, "fallback"))
        return self.fb


@pytest.fixture
def cands(monkeypatch):
    chosen: list[Candidate] = []
    asked = []

    async def choose(kind, text, apps=None, *, said="", about="", exclude=frozenset()):
        asked.append((kind, text, about))
        return list(chosen)

    async def allowed(app):  # macOS's Automation answer for every app: yes
        return None

    monkeypatch.setattr(engine, "choose", choose)
    monkeypatch.setattr(media, "_control_gate", allowed)
    monkeypatch.setattr(media, "_trusted", lambda: True)
    return chosen, asked


def req(**kw) -> Req:
    return Req(**{"kind": "write_note", "fields": {"body": "milk"}, "text": "note milk", **kw})


OK = Outcome(True, "Saved in Notebook.", effect=Effect.DONE, receipt={"id": "n1"})
NONE = Outcome(False, "Notebook's scripting failed")


def test_falls_through_clean_failures_and_says_where_it_went(cands):
    chosen, asked = cands
    chosen += [cand(NOTEBOOK, "script_make", "ax_new"), cand(JOTTER, "ax_new")]
    ok_j = Outcome(True, "Saved in Jotter.", effect=Effect.DONE)
    drv = FakeDriver({("test.notebook", "script_make"): NONE, ("test.notebook", "ax_new"): NONE, ("test.jotter", "ax_new"): ok_j})
    r = asyncio.run(run.run_best(drv, req(about="h1")))
    assert (r.text, r.ok, r.say) == ("Saved in Jotter.", True, True)  # after a fallback: say where
    assert drv.calls == [("test.notebook", "script_make"), ("test.notebook", "ax_new"), ("test.jotter", "ax_new")]
    assert asked == [("write_note", "note milk", "h1")]
    assert engine.prior("write_note", "test.notebook", "ax_new") < engine.prior("write_note", "x", "ax_new")  # blamed
    assert engine.proven("write_note", "test.jotter") and run.last("write_note") == ("test.jotter", "h1")


def test_silent_in_the_habitual_app_and_one_line_otherwise(cands):
    chosen, _ = cands
    chosen.append(cand(NOTEBOOK, "script_make"))
    drv = FakeDriver({("test.notebook", "script_make"): OK})
    assert asyncio.run(run.run_best(drv, req())).say is True  # the first time in this app
    assert asyncio.run(run.run_best(drv, req())).say is False  # where notes always go now
    said = Outcome(True, "Set for 5 PM.", effect=Effect.DONE, say=True)
    drv.script[("test.notebook", "script_make")] = said
    assert asyncio.run(run.run_best(drv, req())).say is True  # the driver's own say wins (an inferred time)


def test_unknown_stops_the_loop_and_blames_no_one(cands):
    chosen, _ = cands
    chosen += [cand(NOTEBOOK, "script_make"), cand(JOTTER, "ax_new")]
    unknown = Outcome(False, "I may have saved it in Notebook but couldn't check.", effect=Effect.UNKNOWN)
    drv = FakeDriver({("test.notebook", "script_make"): unknown, ("test.jotter", "ax_new"): OK})
    r = asyncio.run(run.run_best(drv, req()))
    assert not r.ok and r.say is True and "may have" in r.text
    assert drv.calls == [("test.notebook", "script_make")]  # never a second copy elsewhere
    assert engine.prior("write_note", "test.notebook", "script_make") == pytest.approx(0.75)  # its seed, untouched


def test_a_timeout_after_the_commit_point_is_unknown_and_the_action_still_finishes(cands):
    chosen, _ = cands
    chosen += [cand(NOTEBOOK, "script_make"), cand(JOTTER, "ax_new")]
    finished = []

    async def make():
        await asyncio.sleep(0.3)
        finished.append(1)
        return "made"

    async def slow_commit():
        await run.commit(make())
        return OK

    drv = FakeDriver({("test.notebook", "script_make"): slow_commit, ("test.jotter", "ax_new"): OK}, budget=0.05)

    async def go():
        r = await run.run_best(drv, req())
        await asyncio.sleep(0.4)  # the shielded action runs to its end
        return r

    r = asyncio.run(go())
    assert not r.ok and "Notebook" in r.text and finished == [1]
    assert drv.calls == [("test.notebook", "script_make")]


def test_a_timeout_before_the_commit_point_is_blamed_and_the_next_is_tried(cands):
    chosen, _ = cands
    chosen += [cand(NOTEBOOK, "script_make"), cand(JOTTER, "ax_new")]

    async def hang():
        await asyncio.sleep(5)

    drv = FakeDriver({("test.notebook", "script_make"): hang, ("test.jotter", "ax_new"): OK}, budget=0.05)
    assert asyncio.run(run.run_best(drv, req())).ok
    assert engine.prior("write_note", "test.notebook", "script_make") < 0.75


@pytest.mark.parametrize("reason", ["ambiguous", "needs_user", "empty"])
def test_a_question_or_an_empty_answer_ends_it(cands, reason):
    chosen, _ = cands
    chosen += [cand(NOTEBOOK, "script_make"), cand(JOTTER, "ax_new")]
    drv = FakeDriver({("test.notebook", "script_make"): Outcome(False, "Which folder?", reason=reason),
                      ("test.jotter", "ax_new"): OK})
    r = asyncio.run(run.run_best(drv, req()))
    assert (r.text, r.ok) == ("Which folder?", False) and len(drv.calls) == 1


def test_permission_gates_are_asked_outside_the_budget_and_named_at_the_end(cands, monkeypatch):
    chosen, _ = cands
    chosen += [cand(NOTEBOOK, "script_make", "ax_new")]
    drv = FakeDriver({("test.notebook", "script_make"): OK, ("test.notebook", "ax_new"): OK}, overall=1.5)

    async def slow_denied(app):
        await asyncio.sleep(0.6)  # the user reading macOS's question
        return SimpleNamespace(ok=False, text="Notebook needs your OK before NEO can control it", reason="permission")

    monkeypatch.setattr(media, "_control_gate", slow_denied)
    monkeypatch.setattr(media, "_trusted", lambda: False)
    r = asyncio.run(run.run_best(drv, req()))
    assert not r.ok and r.text == "Notebook needs your OK before NEO can control it" and drv.calls == []
    assert engine.prior("write_note", "test.notebook", "script_make") == pytest.approx(0.75)  # no blame
    monkeypatch.setattr(media, "_trusted", lambda: True)
    assert asyncio.run(run.run_best(drv, req())).ok and drv.calls == [("test.notebook", "ax_new")]


def test_refused_methods_and_a_fallback_that_does_not_apply_are_not_attempts(cands):
    chosen, _ = cands
    chosen += [cand(NOTEBOOK, "script_make", "ax_new"), cand(None, "file")]
    drv = FakeDriver({("test.notebook", "script_make"): NONE, ("test.notebook", "ax_new"): OK},
                     refuse={("test.notebook", "ax_new")}, fallback=None)
    r = asyncio.run(run.run_best(drv, req()))
    assert not r.ok and r.text == NONE.text
    assert drv.calls == [("test.notebook", "script_make"), ("file", "fallback")]
    drv.fb = Outcome(True, "Saved it as a file in Documents › NEO Notes.", effect=Effect.DONE)
    r = asyncio.run(run.run_best(drv, req()))
    assert r.ok and r.text.startswith("Saved it as a file") and engine.proven("write_note", "file")


def test_read_kinds_answer_with_the_information(cands):
    chosen, _ = cands
    chosen.append(cand(NOTEBOOK, "script_read"))
    drv = FakeDriver({("test.notebook", "script_read"): Outcome(True, "read 2", effect=Effect.DONE, data="a | b\nc | d")})
    r = asyncio.run(run.run_best(drv, req(kind="read_mail")))
    assert (r.text, r.ok, r.say) == ("a | b\nc | d", True, None)


def test_nothing_to_try_and_the_attempt_cap(cands):
    chosen, _ = cands
    r = asyncio.run(run.run_best(FakeDriver({}), req()))
    assert not r.ok and r.say is True and r.text == "Nothing on this Mac can do that."
    chosen += [cand(App(bundle_id=f"test.a{i}", name=f"A{i}", path=""), "ax_new") for i in range(5)]
    drv = FakeDriver({(f"test.a{i}", "ax_new"): Outcome(False, f"no {i}", reason="not_found") for i in range(5)})
    r = asyncio.run(run.run_best(drv, req(), max_attempts=2))
    assert len(drv.calls) == 2 and r.text == "no 0"  # the most useful reason, not the last


def test_the_same_add_twice_within_the_window_is_one_add(cands):
    chosen, asked = cands
    chosen.append(cand(NOTEBOOK, "script_make"))
    drv = FakeDriver({("test.notebook", "script_make"): OK})
    sig = ("milk", "2026-10-01T17:00")
    assert asyncio.run(run.run_best(drv, req(sig=sig))).ok
    again = asyncio.run(run.run_best(drv, req(sig=sig)))
    assert (again.text, again.ok) == ("Already added a moment ago.", True) and len(asked) == 1
    assert run.idempotent("write_note", sig) == {"id": "n1"} and run.idempotent("write_note", ("other",)) is None
    assert asyncio.run(run.run_best(drv, req())).ok and len(asked) == 2  # no signature: never deduplicated
    run.forget()
    assert run.idempotent("write_note", sig) is None and run.last("write_note") is None


def test_nothing_is_learned_from_a_request_that_began_before_an_erase(cands, monkeypatch):
    chosen, _ = cands
    chosen.append(cand(NOTEBOOK, "script_make"))

    async def erased_meanwhile():
        monkeypatch.setattr(privacy, "_generation", privacy.generation() + 1)
        return OK

    drv = FakeDriver({("test.notebook", "script_make"): erased_meanwhile})
    assert asyncio.run(run.run_best(drv, req(sig=("x",)))).ok
    assert not engine.proven("write_note", "test.notebook") and run.last("write_note") is None
    assert run.idempotent("write_note", ("x",)) is None


def test_last_action_expires():
    run.forget()
    run.begin_request()
    run.remember_last("send_message", "test.chat", "h9")
    assert run.last("send_message") == ("test.chat", "h9") and run.last("send_message", within_s=-1) is None
    assert run.last("write_note") is None


# ---- sends ---------------------------------------------------------------------------------------

PRIYA = Recipient("Priya Sharma", "contacts", "c1", "mobile", "+91 98765 43123", "phone")


def plan(**kw) -> Plan:
    return Plan(**{"v": 1, "kind": "send_message", "app_key": "test.chat", "app_name": "Chat",
                   "methods": ("script_send", "ax_chat"), "to": (PRIYA,), "about": "h1", **kw})


@pytest.fixture
def chat_app(monkeypatch):
    app = App(bundle_id="test.chat", name="Chat", path="/Applications/Chat.app")
    monkeypatch.setattr(inventory, "app_for_bundle", lambda bid: app if bid == "test.chat" else None)
    monkeypatch.setattr(inventory, "app_by_id", lambda bid: None)
    monkeypatch.setattr(media, "_trusted", lambda: True)

    async def allowed(a):
        return None

    monkeypatch.setattr(media, "_control_gate", allowed)
    return app


def test_run_plan_runs_only_the_approved_app_and_stops_after_a_commit(chat_app):
    async def committed_then_failed():
        await run.commit(asyncio.sleep(0))
        return Outcome(False, "It didn't go. Press Return in Chat to send it.")

    drv = FakeDriver({("test.chat", "script_send"): committed_then_failed, ("test.chat", "ax_chat"): OK})
    out = asyncio.run(run.run_plan(drv, plan(), req(kind="send_message")))
    assert not out.ok and drv.calls == [("test.chat", "script_send")]  # one approval, one commit


def test_run_plan_tries_the_planned_methods_until_one_commits(chat_app):
    sent = Outcome(True, "Sent.", effect=Effect.DONE)
    drv = FakeDriver({("test.chat", "script_send"): NONE, ("test.chat", "ax_chat"): sent})
    assert asyncio.run(run.run_plan(drv, plan(), req(kind="send_message"))).ok
    assert drv.calls == [("test.chat", "script_send"), ("test.chat", "ax_chat")]
    assert run.last("send_message") == ("test.chat", "h1")
    drv = FakeDriver({("test.chat", "script_send"): NONE, ("test.chat", "ax_chat"): Outcome(False, "no chat", reason="not_found")})
    out = asyncio.run(run.run_plan(drv, plan(), req(kind="send_message")))
    assert out.reason == "not_found"  # every method failed cleanly: the caller offers the next app


def test_cancel_after_the_commit_says_it_may_have_gone(chat_app):
    async def slow_send():
        await run.commit(asyncio.sleep(0.3))
        return OK

    drv = FakeDriver({("test.chat", "script_send"): slow_send})

    async def go():
        t = asyncio.create_task(run.run_plan(drv, plan(methods=("script_send",)), req(kind="send_message")))
        await asyncio.sleep(0.05)
        t.cancel()  # the user said stop
        return await t

    out = asyncio.run(go())
    assert out.effect is Effect.UNKNOWN and out.text == "I may already have sent it."

    async def hang():
        await asyncio.sleep(5)

    drv = FakeDriver({("test.chat", "script_send"): hang})

    async def before_commit():
        t = asyncio.create_task(run.run_plan(drv, plan(methods=("script_send",)), req(kind="send_message")))
        await asyncio.sleep(0.05)
        t.cancel()
        return await t

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(before_commit())


def test_run_plan_for_an_app_that_is_gone(chat_app):
    out = asyncio.run(run.run_plan(FakeDriver({}), plan(app_key="test.gone", app_name="Gone"), req(kind="send_message")))
    assert out.reason == "not_found" and "Gone" in out.text


def test_a_plan_round_trips_and_rejects_anything_malformed():
    p = plan(cc=(Recipient("Sam", "literal", handle="sam@example.test", kind="email"),), tainted_by="mail_unread")
    assert Plan.from_param(p.as_param()) == p
    for bad in (None, [], {**p.as_param(), "methods": "ax_chat"}, {**p.as_param(), "v": "1"},
                {**p.as_param(), "to": [{"display": "x", "source": "guess"}]}, {**p.as_param(), "app_key": ""},
                {**p.as_param(), "to": [{"display": 3, "source": "spoken"}]}, {**p.as_param(), "tainted_by": 1}):
        with pytest.raises(ValueError):
            Plan.from_param(bad)


def test_to_output_keeps_halts_and_the_speech_choice():
    from neo.agent.confirm import ask_user

    out = run.to_output(run.Result("Saved in Notebook.", True, say=False))
    assert (out.text, out.ok, out.halt, out.say) == ("Saved in Notebook.", True, False, False)
    q = run.to_output(run.Result(ask_user("Which Priya?"), False, say=True))
    assert q.halt and q.text.startswith("NEEDS_USER") and q.say is True
