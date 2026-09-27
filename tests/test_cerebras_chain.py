"""Cerebras in main's brain chain: where it sits with a key and that nothing changes without one,
failing over past it (a 429, a 402, a spent budget) without long waits, the retry when its
thinking used all the room, and the daily budget's spoken notes.

Hermetic: Cerebras' API is an httpx.MockTransport, every other brain a fake provider object — no
key is real, nothing leaves the Mac, ~/.neo is never read (the suite's data dir is a temp dir).
"""

from __future__ import annotations

import json
import logging
import weakref
from typing import Any

import httpx
import pytest

from neo2.core import config, health
from neo2.core import providers as prov
from neo2.core.providers import cerebras as cb
from neo2.core.providers import cerebras_hooks as hooks
from neo2.core.providers import spend
from neo2.core.providers.base import (
    AllBrainsFailed,
    ImagePart,
    Message,
    Overloaded,
    ToolCall,
    ToolResult,
    Turn,
)

KEY = "csk-test-not-a-real-key"


# ---- fakes -------------------------------------------------------------------------------------


def completion(content: str | None = "ok", *, finish: str = "stop", usage=(1000, 200), model=cb.TEXT_MODEL):
    out: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
    }
    if usage:
        out["usage"] = {"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": sum(usage)}
    return out


class FakeCerebras:
    """Cerebras' API: replies queued per test (else the default), every chat request kept."""

    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []
        self.replies: list[tuple[int, Any, dict[str, str]]] = []
        self.default: tuple[int, Any, dict[str, str]] = (200, completion(), {})

    def reply(self, status: int = 200, payload: Any = None, headers: dict[str, str] | None = None) -> None:
        self.replies.append((status, completion() if payload is None else payload, headers or {}))

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == f"Bearer {KEY}"
        if request.url.path.endswith("/models"):
            ids = [cb.TEXT_MODEL, cb.VISION_MODEL]
            return httpx.Response(200, json={"object": "list", "data": [{"id": i, "object": "model"} for i in ids]})
        self.bodies.append(json.loads(request.content))
        status, payload, headers = self.replies.pop(0) if self.replies else self.default
        return httpx.Response(status, json=payload, headers=headers)

    @property
    def models(self) -> list[str]:
        return [b["model"] for b in self.bodies]


class Brain:
    """Any other brain (Gemini, Groq, the local model): answers, or is overloaded; counts its calls."""

    supports_tools = True

    def __init__(self, name: str, *, vision: bool, busy: bool = False, cooling: float = 0.0) -> None:
        self.name, self.model, self.supports_vision = name, f"{name}-model", vision
        self.on_device = name == "local"
        self.busy, self.cooling = busy, cooling
        self.calls = 0

    def cooling_until(self) -> float:
        return self.cooling

    async def complete(self, messages, **kw) -> Turn:
        self.calls += 1
        if self.busy:
            raise Overloaded(f"{self.name}: 503")
        return Turn(text=f"from {self.name}")

    async def stream(self, messages, **kw):
        self.calls += 1
        if self.busy:
            raise Overloaded(f"{self.name}: 503")
        yield f"from {self.name}"


@pytest.fixture
def fake(monkeypatch):
    f = FakeCerebras()
    monkeypatch.setattr(cb, "_transport", httpx.MockTransport(f.handler))
    slept: list[float] = []

    async def sleep(s: float) -> None:  # a Cerebras wait takes no time here, but is counted
        slept.append(s)

    monkeypatch.setattr(cb, "_sleep", sleep)
    f.slept = slept  # type: ignore[attr-defined]
    return f


@pytest.fixture
def mac(monkeypatch):
    """NEO.app's settings (a Cerebras key unless `key=""`) with fake Gemini, Groq and local brains;
    a fresh pool, accounts, cooldowns and problem list. Returns the fakes by name."""

    def make(*, key: str = KEY, busy: tuple[str, ...] = (), cooling: dict[str, float] | None = None, **env):
        monkeypatch.setenv("CEREBRAS_API_KEY", key)
        monkeypatch.setenv("GEMINI_API_KEY", "g-test")
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        for k, v in env.items():
            monkeypatch.setenv(k, str(v))
        s = config.Settings()
        s.ensure_dirs()
        monkeypatch.setattr(config, "_settings", s)
        fakes = {
            n: Brain(n, vision=n in ("gemini", "gemini_lite", "claude"), busy=n in busy,
                     cooling=(cooling or {}).get(n, 0.0))
            for n in ("gemini", "gemini_lite", "groq", "local", "claude")
        }  # fmt: skip
        real = prov._build
        monkeypatch.setattr(prov, "_build", lambda n: fakes[n] if n in fakes else real(n))
        return fakes

    monkeypatch.setattr(hooks, "_accounts", weakref.WeakKeyDictionary())
    monkeypatch.setattr(prov, "_cache", {})
    monkeypatch.setattr(prov, "_demoted", {})
    monkeypatch.setattr(health, "_live", {})
    return make


ASK = [Message.user("press the OK button")]


def _screenshot_step() -> list[Message]:
    return [
        Message.user("what's on the board?"),
        Message.assistant(tool_calls=[ToolCall("c0", "screenshot", {})]),
        Message.tool([ToolResult("c0", "screenshot", "shot", images=[ImagePart(b"x" * 500, "image/jpeg")])]),
    ]


def _spend_it(s) -> None:
    """Today's spend is already past the (tiny) budget."""
    spend.Ledger(s.data_dir / spend.FILE, daily_budget=1.0, key=KEY).add(cb.TEXT_MODEL, 10_000, 0)


# ---- where it sits -----------------------------------------------------------------------------

MAINS = {  # main's chains, which a missing key or a $0 budget must leave exactly as they are
    "agent": ["gemini", "groq", "local"],
    "fast": ["groq", "gemini", "local"],
    "vision": ["gemini", "claude", "groq", "local"],
    "light": ["gemini_lite", "groq", "gemini", "local"],
    "offline": ["local"],
}


def test_with_a_key_the_text_brain_goes_ahead_of_groq_and_gemini_stays_first(mac):
    mac()
    got = {p: (prov.brain(p).names, prov.brain(p).eyes) for p in MAINS}
    assert got == {
        "agent": (["gemini", "cerebras", "groq", "local"], ["cerebras_vision"]),
        "fast": (["cerebras", "groq", "gemini", "local"], ["cerebras_vision"]),
        "vision": (["gemini", "claude", "cerebras_vision", "cerebras", "groq", "local"], []),
        "light": (["gemini_lite", "cerebras", "groq", "gemini", "local"], ["cerebras_vision"]),
        "offline": (["local"], []),  # offline means this Mac: no cloud brain, no cloud eyes
    }


@pytest.mark.parametrize("off", [{"key": ""}, {"NEO_CEREBRAS_DAILY_BUDGET": "0"}])
def test_without_a_key_or_with_a_zero_budget_every_chain_is_main_s(mac, off):
    mac(**off)
    for purpose, names in MAINS.items():
        c = prov.brain(purpose)
        assert (c.names, c.eyes) == (names, []), purpose


async def test_without_a_key_nothing_goes_to_cerebras_and_a_picture_step_is_main_s(mac, fake):
    brains = mac(key="", busy=("gemini",))
    assert (await prov.brain("agent").complete(ASK)).text == "from groq"
    chain = prov.brain("agent")
    turn = await chain.complete(_screenshot_step())
    assert chain.last_used == "groq" and turn.sees is False  # blind, as on main: its coordinates are refused
    assert fake.bodies == []
    brains["groq"].busy = brains["local"].busy = True
    with pytest.raises(AllBrainsFailed) as e:
        await prov.brain("agent").complete(ASK)
    assert "Cerebras" not in e.value.spoken() and fake.bodies == []


# ---- text steps ----------------------------------------------------------------------------------


async def test_a_text_step_goes_to_gemini_first_then_cerebras_ahead_of_groq(mac, fake):
    brains = mac()
    chain = prov.brain("agent")
    assert (await chain.complete(ASK)).text == "from gemini" and fake.bodies == []  # Gemini keeps first place
    brains["gemini"].busy = True
    prov._demoted.clear()
    chain = prov.brain("agent")
    turn = await chain.complete(ASK, effort="low")
    assert turn.text == "ok" and chain.last_used == cb.TEXT_BRAIN and turn.sees is False
    assert fake.models == [cb.TEXT_MODEL] and brains["groq"].calls == 0
    # Gemini benched after its failure: the next step goes straight to Cerebras
    chain = prov.brain("agent")
    await chain.complete(ASK)
    assert chain.last_used == cb.TEXT_BRAIN and brains["gemini"].calls == 2


async def test_chat_goes_to_cerebras_ahead_of_groq(mac, fake):
    brains = mac()
    fake.default = (200, completion("Hi!"), {})
    chain = prov.brain("fast")
    assert (await chain.complete([Message.user("hello")])).text == "Hi!" and chain.last_used == cb.TEXT_BRAIN
    assert brains["groq"].calls == 0


# ---- picture steps ---------------------------------------------------------------------------------


async def test_a_picture_step_goes_to_gemini_while_it_answers(mac, fake):
    brains = mac()
    chain = prov.brain("agent")
    turn = await chain.complete(_screenshot_step())
    assert turn.text == "from gemini" and chain.last_used == "gemini" and fake.bodies == []
    assert brains["gemini"].calls == 1


@pytest.mark.parametrize("state", ["overloaded", "cooling"])
async def test_a_picture_step_goes_to_cerebras_vision_when_gemini_is_out(mac, fake, state):
    brains = (
        mac(busy=("gemini",)) if state == "overloaded" else mac(cooling={"gemini": 4e9}, busy=("gemini",))
    )
    fake.default = (200, completion("A knight moved to f3.", model=cb.VISION_MODEL), {})
    chain = prov.brain("agent")
    turn = await chain.complete(_screenshot_step(), effort="low")
    assert chain.last_used == cb.VISION_BRAIN and turn.sees is True and turn.text == "A knight moved to f3."
    assert fake.models == [cb.VISION_MODEL]  # never the text model with a picture while one can see it
    assert brains["groq"].calls == 0 and brains["local"].calls == 0
    parts = [p for m in fake.bodies[0]["messages"] if isinstance(m["content"], list) for p in m["content"]]
    assert sum(p["type"] == "image_url" for p in parts) == 1  # it got the picture
    # its step after, text only: the text brain, never the seeing one
    chain = prov.brain("agent")
    await chain.complete(ASK)
    assert chain.last_used == cb.TEXT_BRAIN and fake.models == [cb.VISION_MODEL, cb.TEXT_MODEL]


async def test_the_seeing_fallback_never_goes_ahead_of_a_brain_on_this_mac_the_user_put_first(mac, fake):
    brains = mac(NEO_BRAIN="local")
    chain = prov.brain("agent")
    assert chain.names == ["local", "cerebras", "groq"]
    assert (await chain.complete(_screenshot_step())).text == "from local" and fake.bodies == []
    brains["local"].busy = True
    fake.default = (200, completion("a board", model=cb.VISION_MODEL), {})
    chain = prov.brain("agent")
    await chain.complete(_screenshot_step())
    assert chain.last_used == cb.VISION_BRAIN  # the Mac's own brain couldn't: then the one that sees


async def test_a_spent_seeing_brain_leaves_the_step_to_main_s_blind_fallbacks(mac, fake):
    brains = mac(busy=("gemini",), NEO_CEREBRAS_DAILY_BUDGET="0.001")
    _spend_it(config.settings())
    chain = prov.brain("agent")
    turn = await chain.complete(_screenshot_step())
    assert chain.last_used == "groq" and turn.sees is False and fake.bodies == []  # no round trip to Cerebras
    assert brains["local"].calls == 0


# ---- failing over past Cerebras, without long waits ---------------------------------------------------


async def test_a_429_fails_over_to_groq_at_once_and_cerebras_sits_out_its_wait(mac, fake):
    brains = mac(busy=("gemini",))
    fake.reply(429, {"message": "Too many tokens", "code": "too_many_tokens_error"}, {"retry-after": "30"})
    chain = prov.brain("agent")
    assert (await chain.complete(ASK)).text == "from groq" and chain.last_used == "groq"
    assert fake.slept == [] and len(fake.bodies) == 1  # no sleep, no second try: straight on to Groq
    chain = prov.brain("agent")
    assert (await chain.complete(ASK)).text == "from groq"
    assert len(fake.bodies) == 1 and brains["groq"].calls == 2  # still in its wait: not asked again


async def test_a_429_that_clears_in_a_moment_is_waited_for_briefly(mac, fake):
    mac(busy=("gemini",))
    fake.reply(429, {"message": "Too many requests"}, {"retry-after": "1"})
    chain = prov.brain("agent")
    assert (await chain.complete(ASK)).text == "ok" and chain.last_used == cb.TEXT_BRAIN
    assert fake.slept == [1.0] and len(fake.bodies) == 2


async def test_an_account_out_of_credits_fails_over_and_is_skipped_for_the_day(mac, fake):
    brains = mac(busy=("gemini",))
    fake.reply(402, {"message": "Payment required", "code": "payment_required"})
    chain = prov.brain("agent")
    assert (await chain.complete(ASK)).text == "from groq" and len(fake.bodies) == 1
    chain = prov.brain("agent")
    await chain.complete(ASK)
    assert len(fake.bodies) == 1 and brains["groq"].calls == 2  # skipped without a round trip
    assert "out of credits" in cb.spoken_notes()
    assert any("out of credits" in p for p in health.problems())


async def test_a_spent_budget_fails_over_without_a_round_trip_and_the_stop_says_why(mac, fake):
    brains = mac(busy=("gemini",), NEO_CEREBRAS_DAILY_BUDGET="0.001")
    _spend_it(config.settings())
    chain = prov.brain("agent")
    assert (await chain.complete(ASK)).text == "from groq" and fake.bodies == []  # it knew: asked last
    brains["groq"].busy = brains["local"].busy = True
    with pytest.raises(AllBrainsFailed) as e:
        await prov.brain("agent").complete(ASK)
    said = e.value.spoken()
    assert "Cerebras' daily budget of $0.0010 is used up" in said and "free quota" not in said
    assert "skipping it until tomorrow" in cb.spoken_notes()  # said with the stop, once a day
    assert fake.bodies == []


async def test_a_reply_that_thought_away_all_its_room_is_asked_again_then_left_to_groq(mac, fake):
    brains = mac(busy=("gemini",))
    fake.reply(payload=completion(None, finish="length", usage=(500, 1024)))
    fake.reply(payload=completion("Done."))
    chain = prov.brain("agent")
    assert (await chain.complete(ASK, effort="medium", max_tokens=1000)).text == "Done."
    first, second = fake.bodies
    assert second["reasoning_effort"] == "low" and second["max_tokens"] >= 2 * first["max_tokens"]
    # twice out of room: Groq takes this step, and Cerebras isn't benched for the next (request-shaped)
    fake.reply(payload=completion(None, finish="length"))
    fake.reply(payload=completion(None, finish="length"))
    chain = prov.brain("agent")
    assert (await chain.complete(ASK)).text == "from groq" and brains["groq"].calls == 1
    chain = prov.brain("agent")
    await chain.complete(ASK)
    assert chain.last_used == cb.TEXT_BRAIN


# ---- money ---------------------------------------------------------------------------------------------


async def test_spend_is_counted_and_80_percent_is_said_once_with_the_reply(mac, fake):
    mac(busy=("gemini",), NEO_CEREBRAS_DAILY_BUDGET="0.001")
    fake.reply(payload=completion(usage=(2000, 200)))  # $0.00085 of $0.001
    await prov.brain("agent").complete(ASK)
    day = json.loads((config.settings().data_dir / spend.FILE).read_text())
    assert day["tokens"][cb.TEXT_MODEL] == [2000, 200]
    assert "Cerebras has used $0.0008 of today's $0.0010 budget" in cb.spoken_notes()
    await prov.brain("agent").complete(ASK)  # past the budget now: said once more, that it's spent
    assert "used up" in cb.spoken_notes()
    await prov.brain("agent").complete(ASK)  # skipped: Groq
    assert len(fake.bodies) == 2 and cb.spoken_notes() == ""


def test_the_default_budget_is_fifty_cents_a_day(mac):
    mac()
    assert config.settings().cerebras_daily_budget == 0.50 and cb.conf().daily_budget == 0.50
    assert "csk" not in repr(config.settings())


async def test_the_session_says_the_budget_note_with_its_reply(tmp_path, fake, monkeypatch):
    from neo2.core import runtime
    from neo2.core.agent.registry import catalog
    from neo2.core.agent.session import Session
    from tests.neo_lib import scripted

    monkeypatch.setattr(hooks, "_accounts", weakref.WeakKeyDictionary())
    s = config.LibrarySettings(
        data_dir=tmp_path / "agent", cerebras_api_key=KEY, cerebras_daily_budget=0.001, fast_paths=False
    )
    s.ensure_dirs()
    rt = runtime.Runtime("agent", s, catalog().subset(()))
    scripted(rt, ["The brain answered.", "The brain answered."])
    with runtime.bind(rt.scope()):
        cb.account(cb.conf()).spent(cb.TEXT_MODEL, 2000, 200)  # 85% of the day's budget
        r = await Session().handle("what's new")
        assert r.text.startswith("The brain answered.") and "Cerebras has used $0.0008" in r.text
        assert not r.silent
        again = await Session().handle("and now?")
        assert again.text == "The brain answered."  # said once


# ---- the key ---------------------------------------------------------------------------------------------


async def test_the_key_is_never_logged_or_said(mac, fake, caplog):
    brains = mac(busy=("gemini",))
    brains["groq"].busy = brains["local"].busy = True
    fake.reply(400, {"message": f"bad request from key {KEY}"})
    fake.reply(429, {"message": f"slow down, {KEY}"}, {"retry-after": "100"})  # (past the chain's short wait)
    with caplog.at_level(logging.DEBUG):
        for _ in range(2):
            prov._demoted.clear()
            with pytest.raises(AllBrainsFailed) as e:
                await prov.brain("agent").complete(ASK)
            assert KEY not in str(e.value) and KEY not in e.value.spoken()
            assert all(KEY not in str(x) for _, x in e.value.failures)
    assert len(fake.bodies) == 2
    assert not [r for r in caplog.records if KEY in r.getMessage()]
