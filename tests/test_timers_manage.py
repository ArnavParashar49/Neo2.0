"""NEO's timers: list them, cancel one, ask how long is left — and the countdown its clients show.

Setting a timer is the `timer` tool as before (neo2/core/tools/system.py); `timers` (neo2/core/tools/
timers.py) reads timers.json and only ever removes entries from it. Every timer that starts or ends
is announced on the bus as an `activity` event, and a client's hello carries the running ones, so
NotchHub draws the countdown from `ends_at` itself — nothing is sent per second. A fake clock and the
test's own data dir throughout; no notification is ever really posted.
"""

from __future__ import annotations

import asyncio
import json
import socket

import pytest
import websockets

from neo2.core import events
from neo2.core.agent import fastpath
from neo2.core.agent.registry import ToolContext, registry
from neo2.core.events import EventBus
from neo2.core.reflex.schema import Decision
from neo2.core.tools import MACOS_ONLY, MODULES_FOR, PRESETS, load_all, modules_for, system
from neo2.core.tools import timers as timers_mod

T0 = 1_800_000_000.0


class Clock:
    """time.time() for the timer modules: stands still until a test moves it."""

    def __init__(self) -> None:
        self.now = T0

    def time(self) -> float:
        return self.now


@pytest.fixture
def env(monkeypatch):
    """The tools loaded, a fresh bus whose events are kept, a fake clock and a fake notifier."""
    load_all()
    bus = EventBus()
    monkeypatch.setattr(events, "_bus", bus)
    seen: list[tuple[str, dict]] = []
    bus.subscribe(lambda ev: seen.append((ev.type, dict(ev.data))))
    clock = Clock()
    monkeypatch.setattr(system, "time", clock)
    monkeypatch.setattr(timers_mod, "time", clock)
    notes: list[str] = []

    async def fake_osa(script, timeout=10):
        notes.append(script)
        return ""

    monkeypatch.setattr(system, "osascript", fake_osa)
    system._timer_tasks.clear()

    class Env:
        pass

    e = Env()
    e.bus, e.seen, e.clock, e.notes = bus, seen, clock, notes
    e.activity = lambda: [d for t, d in seen if t == "activity"]
    e.said = lambda: [d["text"] for t, d in seen if t == "transcript"]
    yield e
    for t in list(system._timer_tasks.values()):
        t.cancel()
    system._timer_tasks.clear()


async def _timers(**a) -> str:
    return (await registry().invoke("timers", a, ToolContext())).text


async def _set(minutes: float, label: str | None = None) -> dict:
    args = {"minutes": minutes, **({"label": label} if label else {})}
    out = await registry().invoke("timer", args, ToolContext())
    assert out.ok and out.text.startswith("Timer set"), out.text
    await asyncio.sleep(0)  # its countdown starts (and says so)
    return system._load_timers()[-1]


async def _settle(n: int = 5) -> None:
    for _ in range(n):
        await asyncio.sleep(0.01)


# ---- setting a timer is as before ---------------------------------------------------------------


def test_setting_a_timer_is_unchanged(env):
    async def run():
        out = await registry().invoke("timer", {"minutes": 5, "label": "Pasta"}, ToolContext())
        assert out.text == "Timer set for 5 minutes."
        saved = system._load_timers()
        assert len(saved) == 1 and set(saved[0]) == {"id", "label", "due"}  # the file's shape: nothing added
        assert saved[0]["label"] == "Pasta" and saved[0]["due"] == T0 + 300

    asyncio.run(run())


# ---- list / left / cancel ---------------------------------------------------------------------


def test_nothing_running(env):
    async def run():
        for a in ({"action": "list"}, {"action": "left"}, {"action": "cancel"}, {"action": "cancel", "all": True}):
            assert await _timers(**a) == "You have no timers running."

    asyncio.run(run())


def test_list_and_time_left(env):
    async def run():
        await _set(4.2, "Pasta")  # 252 s
        await _set(2, "Tea")
        env.clock.now += 0.4  # a moment later: the seconds round up, as a countdown shows them
        assert await _timers(action="list") == "You have 2 timers: Tea, 2 minutes left; Pasta, 4 minutes 12 seconds left."
        assert await _timers(action="left", label="pasta") == "Pasta: 4 minutes 12 seconds left."
        assert await _timers(action="left", label="the pasta timer") == "Pasta: 4 minutes 12 seconds left."
        assert await _timers(action="left") == "Tea: 2 minutes left; Pasta: 4 minutes 12 seconds left."
        assert await _timers(action="left", label="eggs") == (
            "You don't have a timer called eggs. You have 2 timers: Tea, 2 minutes left; Pasta, 4 minutes 12 seconds left."
        )
        await _timers(action="cancel", label="tea")
        assert await _timers(action="list") == "You have one timer: Pasta, 4 minutes 12 seconds left."
        assert await _timers(action="left") == "Pasta: 4 minutes 12 seconds left."

    asyncio.run(run())


@pytest.mark.parametrize(
    ("seconds", "words"),
    [
        (1, "1 second"), (42, "42 seconds"), (60, "1 minute"), (61, "1 minute 1 second"), (252, "4 minutes 12 seconds"),
        (599, "9 minutes 59 seconds"), (600, "10 minutes"), (659, "11 minutes"), (3599, "1 hour"),
        (3900, "1 hour 5 minutes"), (7200, "2 hours"),
    ],
)  # fmt: skip
def test_time_left_reads_like_a_person_says_it(seconds, words):
    assert timers_mod._left_words(seconds) == words


def test_cancel_by_name_removes_it_and_it_never_goes_off(env):
    async def run():
        tea = await _set(10, "Tea")
        # Due in 0.2 s (shorter than the timer tool sets): its countdown looks at the clock every 0.2 s.
        pasta = {"id": "p1", "label": "Pasta", "due": T0 + 0.2}
        system._save_timers(system._load_timers() + [pasta])
        system._start(pasta)
        await asyncio.sleep(0.3)
        assert await _timers(action="cancel", label="pasta") == "Cancelled the Pasta timer."
        assert [t["id"] for t in system._load_timers()] == [tea["id"]]
        assert pasta["id"] not in [t["id"] for t in system._load_timers()]
        assert pasta["id"] not in system._timer_tasks
        env.clock.now += 3600  # long past when it was due: a countdown still running would go off now
        await asyncio.sleep(0.6)
        assert env.notes == [] and not any("time's up" in s for s in env.said())
        assert tea["id"] in system._timer_tasks  # (the other one is untouched)

    asyncio.run(run())


def test_cancel_with_several_and_no_name_asks_which(env):
    async def run():
        await _set(5, "Pasta")
        await _set(2, "Tea")
        out = await _timers(action="cancel")
        assert out == "You have 2 timers: Tea, 2 minutes left; Pasta, 5 minutes left. Which one should I cancel?"
        assert not out.startswith("NEEDS_")  # a plain question, not a halt
        assert len(system._load_timers()) == 2  # nothing cancelled
        # a duration isn't a name: "the 5 minute timer" is any of them, so the question again
        assert (await _timers(action="cancel", label="5 minute")).endswith("Which one should I cancel?")
        assert await _timers(action="cancel", label="tea") == "Cancelled the Tea timer."
        assert await _timers(action="cancel") == "Cancelled the Pasta timer."  # one left: that one

    asyncio.run(run())


def test_timers_with_the_same_name_are_told_apart_by_place(env):
    async def run():
        await _set(9)
        await _set(4)
        out = await _timers(action="cancel")
        assert out == "You have 2 timers: the first, 4 minutes left; the second, 9 minutes left. Which one should I cancel?"
        assert await _timers(action="cancel", label="the second one") == "Cancelled your timer."
        assert await _timers(action="list") == "You have one timer: Timer, 4 minutes left."

    asyncio.run(run())


def test_cancel_all_and_an_unknown_name(env):
    async def run():
        await _set(5, "Pasta")
        await _set(2, "Tea")
        await _set(7, "Laundry")
        assert await _timers(action="cancel", label="eggs") == (
            "You don't have a timer called eggs. You have 3 timers: Tea, 2 minutes left; Pasta, 5 minutes left; "
            "Laundry, 7 minutes left."
        )
        assert len(system._load_timers()) == 3
        assert await _timers(action="cancel", all=True) == "Cancelled all 3 timers."
        assert system._load_timers() == [] and system._timer_tasks == {}
        await _set(1, "Tea")
        assert await _timers(action="cancel", all=True) == "Cancelled the Tea timer."

    asyncio.run(run())


# ---- what the clients are told ------------------------------------------------------------------


def test_activity_events_on_set_cancel_restore_and_fire(env):
    async def run():
        pasta = await _set(5, "Pasta  al\ndente")
        present = {"op": "present", "id": f"timer:{pasta['id']}", "kind": "timer", "title": "Pasta al dente",
                   "ends_at": T0 + 300, "done": False}  # fmt: skip
        assert env.activity() == [present]
        await _timers(action="cancel", label="pasta")
        assert env.activity() == [present, {**present, "op": "end"}]  # done=false: cancelled
        # restored after a restart: told again; overdue, it goes off at once (done=true)
        env.seen.clear()
        late = {"id": "abc123", "label": "Tea", "due": T0 - 30}
        system._save_timers([late])
        assert system.restore_timers() == 1
        await _settle()
        want = {"op": "present", "id": "timer:abc123", "kind": "timer", "title": "Tea", "ends_at": T0 - 30, "done": False}
        assert env.activity() == [want, {**want, "op": "end", "done": True}]
        assert len(env.notes) == 1 and env.said() == ["Tea: time's up."]
        assert system._load_timers() == []

    asyncio.run(run())


def test_a_long_label_is_cut_for_the_island(env):
    async def run():
        await _set(1, "a" * 70)
        assert env.activity()[0]["title"] == "a" * 40

    asyncio.run(run())


def test_the_snapshot_holds_only_running_timers(env):
    async def run():
        assert system.timers_snapshot() == []
        tea = await _set(2, "Tea")
        pasta = await _set(1, "Pasta")
        assert system.timers_snapshot() == [
            {"id": f"timer:{pasta['id']}", "title": "Pasta", "ends_at": T0 + 60},
            {"id": f"timer:{tea['id']}", "title": "Tea", "ends_at": T0 + 120},
        ]
        # in timers.json but not counting down in this process (never restored): not running
        system._save_timers(system._load_timers() + [{"id": "zzz", "label": "Old", "due": T0 + 30}])
        assert [t["title"] for t in system.timers_snapshot()] == ["Pasta", "Tea"]

    asyncio.run(run())


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_the_hello_carries_the_running_timers(env, monkeypatch):
    from neo2.core import config
    from neo2.core.server import ws as ws_mod

    port = _free_port()
    monkeypatch.setenv("NEO_WS_PORT", str(port))
    monkeypatch.setattr(config, "_settings", None)

    async def on_command(cmd):
        pass

    async def hello(c) -> dict:
        msg = json.loads(await asyncio.wait_for(c.recv(), 3))
        assert msg["type"] == "hello"
        return msg["data"]

    async def activity(c) -> dict:
        while (msg := json.loads(await asyncio.wait_for(c.recv(), 3)))["type"] != "activity":
            pass
        return msg["data"]

    async def run():
        srv = ws_mod.UIServer(on_command)
        await srv.start()
        url = f"ws://127.0.0.1:{port}/?token={ws_mod.ws_token()}&role=host"
        try:
            async with websockets.connect(url, open_timeout=3) as c:
                assert (await hello(c))["timers"] == []
                pasta = await _set(5, "Pasta")
                assert (await activity(c))["op"] == "present"
                async with websockets.connect(url, open_timeout=3) as late:
                    assert (await hello(late))["timers"] == [{"id": f"timer:{pasta['id']}", "title": "Pasta", "ends_at": T0 + 300}]
                await _timers(action="cancel")
                assert await activity(c) == {"op": "end", "id": f"timer:{pasta['id']}", "kind": "timer", "title": "Pasta",
                                       "ends_at": T0 + 300, "done": False}  # fmt: skip
            async with websockets.connect(url, open_timeout=3) as again:
                assert (await hello(again))["timers"] == []
        finally:
            srv._server.close()

    asyncio.run(run())


# ---- the words that reach it --------------------------------------------------------------------


def _hit(text: str):
    return fastpath.match(text)


@pytest.mark.parametrize(
    ("text", "args"),
    [
        ("cancel the timer", {"action": "cancel", "label": None}),
        ("stop timer", {"action": "cancel", "label": None}),
        ("cancel timer", {"action": "cancel", "label": None}),
        ("please delete my timer", {"action": "cancel", "label": None}),
        ("stop the pasta timer", {"action": "cancel", "label": "pasta"}),
        ("turn off the 10 minute timer", {"action": "cancel", "label": "10 minute"}),
        ("cancel all my timers", {"action": "cancel", "all": True}),
        ("cancel the timers", {"action": "cancel", "all": True}),
        ("timer band karo", {"action": "cancel", "label": None}),
        ("timer cancel kar do", {"action": "cancel", "label": None}),
        ("pasta wala timer band karo", {"action": "cancel", "label": "pasta"}),
        ("sab timers band karo", {"action": "cancel", "all": True}),
        ("टाइमर बंद करो", {"action": "cancel"}),
        ("how long is left on the pasta timer?", {"action": "left", "label": "pasta"}),
        ("how much time is left on my timer", {"action": "left", "label": None}),
        ("how much time is left?", {"action": "left"}),
        ("how long left", {"action": "left"}),
        ("what's the time left on the timer", {"action": "left", "label": None}),
        ("when will my timer go off?", {"action": "left", "label": None}),
        ("kitna time bacha hai", {"action": "left"}),
        ("timer mein kitna time bacha hai?", {"action": "left", "label": None}),
        ("pasta wale timer mein kitni der baaki hai", {"action": "left", "label": "pasta"}),
        ("कितना समय बचा है", {"action": "left"}),
        ("what timers do I have", {"action": "list"}),
        ("which timers are running?", {"action": "list"}),
        ("list my timers", {"action": "list"}),
        ("show me all my timers", {"action": "list"}),
        ("do I have any timers running?", {"action": "list"}),
        ("mere timers dikhao", {"action": "list"}),
    ],
)  # fmt: skip
def test_the_fast_paths(env, text, args):
    system._save_timers([{"id": "t1", "label": "Pasta", "due": T0 + 60}])  # a timer is set
    assert _hit(text) == ("timers", args)


@pytest.mark.parametrize("text", ["how much time is left?", "how long left", "kitna time bacha hai", "कितना समय बचा है"])
def test_a_bare_how_long_left_is_a_timers_question_only_while_one_runs(env, text):
    hit = _hit(text)
    assert hit is None or hit[0] != "timers"


@pytest.mark.parametrize(
    "text",
    [
        "how long is the movie", "how much time is left in the semester", "cancel my dentist appointment",
        "what time is it", "stop the music", "I love this timer app", "what's left to do today", "stop", "cancel",
        "set an alarm", "stop the timer app from opening at login", "timer", "the timer is broken", "how is the weather",
    ],
)  # fmt: skip
def test_ordinary_sentences_are_not_timer_commands(env, text):
    system._save_timers([{"id": "t1", "label": "Pasta", "due": T0 + 60}])
    hit = _hit(text)
    assert hit is None or hit[0] != "timers", hit


def test_setting_one_still_reaches_the_timer_tool(env):
    assert _hit("set a timer for 5 minutes") == ("timer", {"minutes": "5"})
    assert _hit("set a 10 minute timer") == ("timer", {"minutes": "10"})


# ---- "stop timer" is a command, not a bare stop -------------------------------------------------


class StopReflex:
    """Laya at its most certain that every sentence means "stop"."""

    def __init__(self, intent: str = "stop") -> None:
        self.intent = intent

    async def decide(self, text):
        return Decision(self.intent, 0.95, False, 0.0, False, 0.0, "fake")


@pytest.mark.parametrize("text", ["stop timer", "cancel timer", "timer band karo"])
def test_stop_timer_never_reaches_cancel_all(env, text, monkeypatch):
    from neo2.core.agent.session import Session

    async def run():
        await _set(5, "Pasta")
        s = Session(reflex=StopReflex())
        stopped = []

        async def cancel_all():
            stopped.append(1)
            return 0

        monkeypatch.setattr(s, "cancel_all", cancel_all)
        r = await s.handle(text, from_user=True)
        assert stopped == [] and r.route == "quick" and r.text == "Cancelled the Pasta timer."
        assert system._load_timers() == []

    asyncio.run(run())


def test_a_bare_stop_still_cancels_everything(env, monkeypatch):
    from neo2.core.agent.session import Session

    async def run():
        await _set(5, "Pasta")
        s = Session(reflex=StopReflex())
        stopped = []

        async def cancel_all():
            stopped.append(1)
            return 0

        monkeypatch.setattr(s, "cancel_all", cancel_all)
        for text in ("stop", "never mind", "stop it"):
            r = await s.handle(text, from_user=True)
            assert r.route == "stop"
        assert stopped == [1, 1, 1] and len(system._load_timers()) == 1  # (and a timer isn't a job)

    asyncio.run(run())


def test_a_timer_question_laya_calls_chat_is_still_answered_from_the_timers(env):
    from neo2.core.agent.session import Session

    async def run():
        await _set(4.2, "Pasta")
        r = await Session(reflex=StopReflex("chat")).handle("how long is left on the pasta timer", from_user=True)
        assert r.route == "quick" and r.text == "Pasta: 4 minutes 12 seconds left."

    asyncio.run(run())


# ---- the catalog ------------------------------------------------------------------------------


def test_the_catalog_knows_the_timers_tool(env):
    from neo2.core.voice import live

    t = registry().get("timers")
    assert t is not None and t.requires == {"macos"} and not t.quiet and t.category == "system"
    assert "timers" in PRESETS["mac"] and "timers" in MACOS_ONLY
    assert MODULES_FOR["timers"] == "neo2.core.tools.timers" and modules_for(["timers"]) == ["neo2.core.tools.timers"]
    assert "timers" in live._DIRECT_TOOLS  # Gemini Live answers "how long left" itself
