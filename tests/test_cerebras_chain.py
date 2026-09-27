"""Cerebras in the brain chains (brain-effort's: the seeing fallbacks, NoEyes, no picture to a brain
that can't see it): where it sits with a key and that nothing changes without one, failing over past
it (a 429, a 402, a spent budget) without long waits, the retry when its thinking used all the room,
the daily budget's spoken notes — and the user's own config, NEO_BRAIN=cerebras: gpt-oss for the
text steps, qwen for the pictures.

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
    NoEyes,
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
            n: Brain(n, vision=n in ("gemini", "gemini_lite", "claude", "groq_vision"), busy=n in busy,
                     cooling=(cooling or {}).get(n, 0.0))
            for n in ("gemini", "gemini_lite", "groq", "local", "claude", "groq_vision")
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

MAINS = {  # the chains (names, seeing fallbacks) a missing key or a $0 budget must leave as they are
    "agent": (["gemini", "groq", "local"], ["gemini_lite", "claude", "groq_vision"]),
    "fast": (["groq", "gemini", "local"], ["gemini_lite", "claude", "groq_vision"]),
    "vision": (["gemini", "gemini_lite", "claude", "groq", "local"], ["groq_vision"]),
    "light": (["gemini_lite", "groq", "gemini", "local"], ["claude", "groq_vision"]),
    "offline": (["local"], []),
}


def test_with_a_key_the_text_brain_goes_ahead_of_groq_and_gemini_stays_first(mac):
    mac()
    got = {p: (prov.brain(p).names, prov.brain(p).eyes) for p in MAINS}
    assert got == {
        "agent": (["gemini", "cerebras", "groq", "local"], ["gemini_lite", "claude", "cerebras_vision", "groq_vision"]),
        "fast": (["cerebras", "groq", "gemini", "local"], ["gemini_lite", "claude", "cerebras_vision", "groq_vision"]),
        "vision": (["gemini", "gemini_lite", "claude", "cerebras_vision", "cerebras", "groq", "local"], ["groq_vision"]),
        "light": (["gemini_lite", "cerebras", "groq", "gemini", "local"], ["claude", "cerebras_vision", "groq_vision"]),
        "offline": (["local"], []),  # offline means this Mac: no cloud brain, no cloud eyes
    }  # fmt: skip


@pytest.mark.parametrize("off", [{"key": ""}, {"NEO_CEREBRAS_DAILY_BUDGET": "0"}])
def test_without_a_key_or_with_a_zero_budget_every_chain_is_main_s(mac, off):
    mac(**off)
    for purpose, chain in MAINS.items():
        c = prov.brain(purpose)
        assert (c.names, c.eyes) == chain, purpose


@pytest.mark.parametrize(
    ("env", "want"),
    [
        # only brains on this Mac named: nothing the user said ever left it, and a key doesn't change
        # that (nor do main's own cloud brains in "light" and "vision": Cerebras isn't added to them)
        (
            {"NEO_BRAIN": "local", "NEO_FAST_BRAIN": "local"},
            {"agent": ["local"], "fast": ["local"], "light": ["gemini_lite", "local"],
             "vision": ["gemini", "gemini_lite", "claude", "local"]},
        ),
        # a cloud brain named after the local one: Cerebras joins behind it, never ahead of local
        (
            {"NEO_BRAIN": "local", "NEO_FAST_BRAIN": "gemini"},
            {"agent": ["local", "gemini", "cerebras"], "fast": ["gemini", "cerebras", "local"],
             "light": ["gemini_lite", "gemini", "cerebras", "local"],
             "vision": ["gemini", "gemini_lite", "claude", "cerebras_vision", "cerebras", "local"]},
        ),
        (
            {"NEO_BRAIN": "gemini", "NEO_FAST_BRAIN": "local"},
            {"agent": ["gemini", "cerebras", "local"], "fast": ["local", "gemini", "cerebras"],
             "light": ["gemini_lite", "local", "gemini", "cerebras"],
             "vision": ["gemini", "gemini_lite", "claude", "cerebras_vision", "cerebras", "local"]},
        ),
        (
            {"NEO_BRAIN": "claude", "NEO_FAST_BRAIN": "local"},
            {"agent": ["claude", "cerebras", "local"], "fast": ["local", "claude", "cerebras"],
             "light": ["gemini_lite", "local", "claude", "cerebras"],
             "vision": ["claude", "gemini", "gemini_lite", "cerebras_vision", "cerebras", "local"]},
        ),
        # local first, Groq for chat: ahead of Groq (decision 49), and no seeing Cerebras (no picture
        # of theirs left the Mac: see the next test)
        (
            {"NEO_BRAIN": "local"},
            {"agent": ["local", "cerebras", "groq"], "fast": ["cerebras", "groq", "local"],
             "light": ["gemini_lite", "cerebras", "groq", "local"],
             "vision": ["gemini", "gemini_lite", "claude", "local", "cerebras", "groq"]},
        ),
    ],
)  # fmt: skip
def test_cerebras_never_goes_ahead_of_a_brain_on_this_mac_the_user_put_first(mac, env, want):
    mac(**env)
    assert {p: prov.brain(p).names for p in want} == want


async def test_without_a_key_nothing_goes_to_cerebras_and_a_picture_step_is_main_s(mac, fake):
    brains = mac(key="", busy=("gemini",))
    assert (await prov.brain("agent").complete(ASK)).text == "from groq"
    chain = prov.brain("agent")
    turn = await chain.complete(_screenshot_step())
    # the next brain that sees (Flash-Lite) — never a text brain acting blind, nor the local brain the
    # user put last (Gemini's failure moved it behind that brain: no reason to give it the screen)
    assert chain.last_used == "gemini_lite" and turn.sees is True
    assert brains["local"].calls == 0 and brains["groq"].calls == 1 and fake.bodies == []
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
    """Gemini out (and the free seeing fallbacks asked before it: Flash-Lite, Claude): the picture
    goes to Cerebras' seeing brain, ahead of Groq's vision model."""
    free = ("gemini", "gemini_lite", "claude")
    brains = mac(busy=free) if state == "overloaded" else mac(cooling={"gemini": 4e9}, busy=free)
    fake.default = (200, completion("A knight moved to f3.", model=cb.VISION_MODEL), {})
    chain = prov.brain("agent")
    turn = await chain.complete(_screenshot_step(), effort="low")
    assert chain.last_used == cb.VISION_BRAIN and turn.sees is True and turn.text == "A knight moved to f3."
    assert fake.models == [cb.VISION_MODEL]  # never the text model with a picture while one can see it
    assert brains["groq"].calls == brains["local"].calls == brains["groq_vision"].calls == 0
    parts = [p for m in fake.bodies[0]["messages"] if isinstance(m["content"], list) for p in m["content"]]
    assert sum(p["type"] == "image_url" for p in parts) == 1  # it got the picture
    # its step after, text only: the text brain, never the seeing one
    chain = prov.brain("agent")
    await chain.complete(ASK)
    assert chain.last_used == cb.TEXT_BRAIN and fake.models == [cb.VISION_MODEL, cb.TEXT_MODEL]


async def test_a_local_first_user_s_pictures_never_go_to_cerebras(mac, fake):
    """NEO_BRAIN=local and no cloud brain that sees named: no picture ever leaves the Mac (a pixel
    step with no seeing brain stops and says so, DECISIONS 39). A Cerebras key alone doesn't change
    that: no seeing fallback, and its text brain never gets the picture — nor acts on it blind."""
    brains = mac(NEO_BRAIN="local")
    chain = prov.brain("agent")
    assert (chain.names, chain.eyes, chain.supports_vision) == (["local", "cerebras", "groq"], [], False)
    with pytest.raises(NoEyes) as e:
        await chain.complete(_screenshot_step())
    assert "can see the screen" in e.value.spoken() and fake.bodies == []  # the picture stayed on the Mac
    assert not any(b.calls for b in brains.values())
    assert (await prov.brain("agent").complete(ASK)).text == "from local"  # its words still go to local


def test_a_local_first_user_who_names_a_brain_that_sees_gets_cerebras_eyes_too(mac):
    mac(NEO_BRAIN="local", NEO_FAST_BRAIN="gemini")  # Gemini already sees their screenshots
    assert prov.brain("agent").eyes == ["gemini_lite", "claude", cb.VISION_BRAIN, "groq_vision"]


def test_cerebras_eyes_follow_the_seeing_fallbacks_rule():
    """with_cerebras_eyes mirrors brain-effort's _fallback_eyes: cloud eyes for a local-first user
    only with NEO_CLOUD_EYES=on or a cloud brain that sees among the ones they named."""
    from types import SimpleNamespace as NS

    key = {"cerebras_api_key": KEY, "cerebras_daily_budget": 0.5}
    eyes = ("gemini", "gemini_lite", "claude", "groq_vision")
    local = NS(**key, brain="local", fast_brain="groq", offline_brain="local")
    assert hooks.with_cerebras_eyes((), local) == ()
    opted_in = NS(**key, brain="local", fast_brain="groq", offline_brain="local", cloud_eyes="on")
    assert hooks.with_cerebras_eyes((), opted_in) == (cb.VISION_BRAIN,)
    cloud = NS(**key, brain="gemini", fast_brain="groq", offline_brain="local")
    want = ("gemini", "gemini_lite", "claude", cb.VISION_BRAIN, "groq_vision")
    assert hooks.with_cerebras_eyes(eyes, cloud) == want


async def test_a_spent_seeing_brain_leaves_the_step_to_groq_s_vision_model(mac, fake):
    brains = mac(busy=("gemini", "gemini_lite", "claude"), NEO_CEREBRAS_DAILY_BUDGET="0.001")
    _spend_it(config.settings())
    chain = prov.brain("agent")
    turn = await chain.complete(_screenshot_step())
    assert chain.last_used == "groq_vision" and turn.sees is True and fake.bodies == []  # no round trip to Cerebras
    assert brains["local"].calls == brains["groq"].calls == 0


# ---- the user's config: NEO_BRAIN=cerebras, $1.00 a day ----------------------------------------------


def _pictures(body: dict[str, Any]) -> int:
    return sum(
        p.get("type") == "image_url" for m in body["messages"] if isinstance(m["content"], list) for p in m["content"]
    )


async def test_neo_brain_cerebras_text_steps_on_gpt_oss_and_picture_steps_on_qwen(mac, fake):
    """The user's live config (NEO_BRAIN=cerebras, NEO_CEREBRAS_DAILY_BUDGET=1.00): gpt-oss is the
    brain, qwen sees the screen. With Gemini and Groq keys set too, an agent's text steps go to
    Cerebras' gpt-oss and its picture steps to Cerebras' qwen — in the agent chain and in a screen
    task's (the vision chain) — and no other brain is asked."""
    brains = mac(NEO_BRAIN="cerebras", NEO_CEREBRAS_DAILY_BUDGET="1.00")
    assert config.settings().cerebras_daily_budget == 1.00 and hooks.enabled()
    for purpose in ("agent", "vision"):
        chain = prov.brain(purpose)
        fake.default = (200, completion("text step", model=cb.TEXT_MODEL), {})
        turn = await chain.complete(ASK)
        assert (chain.last_used, chain.last_model, turn.sees) == (cb.TEXT_BRAIN, cb.TEXT_MODEL, False), purpose
        fake.default = (200, completion("a picture step", model=cb.VISION_MODEL), {})
        turn = await chain.complete(_screenshot_step())
        assert (chain.last_used, chain.last_model, turn.sees) == (cb.VISION_BRAIN, cb.VISION_MODEL, True), purpose
    assert fake.models == [cb.TEXT_MODEL, cb.VISION_MODEL] * 2
    assert [_pictures(b) for b in fake.bodies] == [0, 1] * 2  # the picture only ever to qwen
    assert not any(b.calls for b in brains.values())  # Gemini, Flash-Lite, Claude, Groq, local: never asked
    # qwen out for the day: the picture goes to the next brain that sees, never to gpt-oss blind
    fake.default = (402, {"message": "Payment required", "code": "payment_required"}, {})
    chain = prov.brain("agent")
    turn = await chain.complete(_screenshot_step())
    assert chain.last_used == "gemini" and turn.sees is True and fake.models[-1] == cb.VISION_MODEL


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


async def test_a_stop_says_the_spent_budget_once_not_once_per_cerebras_brain(mac, fake):
    from neo2.core.agent.loop import _model_trouble
    from neo2.core.agent.session import Reply

    mac(busy=("gemini", "gemini_lite", "claude", "groq", "local"), NEO_CEREBRAS_DAILY_BUDGET="0.001")
    _spend_it(config.settings())
    with pytest.raises(AllBrainsFailed) as e:
        await prov.brain("vision").complete(ASK)  # a screen task's text step: both Cerebras brains asked, both out
    assert {n for n, _ in e.value.failures} >= {cb.TEXT_BRAIN, cb.VISION_BRAIN}
    budget = "Cerebras' daily budget of $0.0010 is used up"
    assert e.value.spoken().count(budget) == 1
    # the session's reply: the stop, and not the same news again as a note after it
    heard = cb.add_spoken_notes(Reply(text=_model_trouble(e.value, []), route="agent")).text
    assert heard.count(budget) == 1 and heard.startswith("I had to stop:")
    assert cb.spoken_notes() == ""  # (and not saved up for the next reply either)


def test_a_note_the_reply_doesn_t_already_say_is_still_heard(mac):
    from neo2.core.agent.session import Reply

    mac()
    a = cb.account(cb.conf())
    a._tell("Cerebras' daily budget of $0.50 is used up; I'll skip it until tomorrow")
    a._tell("Cerebras refused the API key (CEREBRAS_API_KEY); skipping it for today")
    r = cb.add_spoken_notes(
        Reply(text="I had to stop: Cerebras refused the API key (CEREBRAS_API_KEY).", route="agent")
    )
    assert r.text == (
        "I had to stop: Cerebras refused the API key (CEREBRAS_API_KEY). "
        "Cerebras' daily budget of $0.50 is used up; I'll skip it until tomorrow."
    )


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


@pytest.mark.parametrize(
    "value", ["off", "OFF", "none", "false", "no", "-1", "$-2", "nan", "inf", "lots", "1,25"]
)
def test_a_budget_that_isn_t_a_dollar_amount_turns_cerebras_off_not_on(mac, value, caplog):
    """The cap fails closed: "off" (the README's 0), a negative number or a typo spends nothing."""
    with caplog.at_level(logging.INFO):
        mac(NEO_CEREBRAS_DAILY_BUDGET=value)
    assert config.settings().cerebras_daily_budget == 0.0 and cb.conf().daily_budget == 0.0
    assert not hooks.enabled()
    for purpose, chain in MAINS.items():
        c = prov.brain(purpose)
        assert (c.names, c.eyes) == chain, purpose
    assert "NEO_CEREBRAS_DAILY_BUDGET" in caplog.text  # said why in the log


@pytest.mark.parametrize(
    ("value", "want"), [("", 0.50), ("  ", 0.50), ("$1.25", 1.25), (" 0.2 ", 0.2), ("0", 0.0)]
)
def test_only_an_empty_budget_is_the_default(mac, value, want):
    mac(NEO_CEREBRAS_DAILY_BUDGET=value)
    assert config.settings().cerebras_daily_budget == want
    assert spend.budget(value, 0.5) == want  # (a host's hosted(daily_budget=...) reads it the same way)
    assert spend.budget(None, 0.5) == 0.5


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


# ---- the SDK ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["brain", "fast_brain"])
@pytest.mark.parametrize("name", [cb.TEXT_BRAIN, cb.VISION_BRAIN])
def test_an_embedded_agent_names_a_cerebras_brain_with_its_own_key(field, name):
    from neo2.sdk.config import AgentConfig

    c = AgentConfig(**{field: name}, cerebras_api_key=KEY)
    assert getattr(c, field) == name and KEY not in repr(c)
    assert AgentConfig(brain="groq", fast_brain="gemini").brain == "groq"  # the others as before


def test_erasing_an_embedded_agent_removes_its_cerebras_spend_ledger():
    from neo2.sdk import agent

    assert spend.FILE in agent._OWN_FILES
