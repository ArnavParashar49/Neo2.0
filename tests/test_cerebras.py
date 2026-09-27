"""Cerebras (decision 49): the provider, its spend ledger and where it goes in a chain.

Hermetic: every request goes through the real openai SDK into an httpx.MockTransport (or a
server this test process runs on 127.0.0.1) — no key is real, nothing leaves the Mac, and
~/.neo is never read (the suite's data dir is a temp dir).
"""

from __future__ import annotations

import asyncio
import json
import math
import time
import weakref
from types import SimpleNamespace
from typing import Annotated, Any

import httpx
import pytest
from pydantic import Field, field_validator
from pydantic_settings import NoDecode

from neo2.core import config
from neo2.core import providers as prov
from neo2.core.providers import cerebras as cb
from neo2.core.providers import cerebras_hooks as hooks
from neo2.core.providers import spend
from neo2.core.providers.base import (
    AllBrainsFailed,
    BadRequest,
    ImagePart,
    Message,
    Overloaded,
    RateLimited,
    RequestTooLarge,
    ToolCall,
    ToolResult,
    ToolSpec,
    Turn,
    Unavailable,
)
from neo2.core.providers.openai_compat import OpenAICompatProvider, Truncated

KEY = "csk-test-not-a-real-key"


# ---- the settings this workstream asks the integrator to declare (see the wiring) -------------


def _settings_class():
    """neo2.core.config.Settings once the Cerebras keys are declared there; until then, the same block
    on a subclass (the exact fields the wiring adds)."""
    if "cerebras_api_key" in config.Settings.model_fields:
        return config.Settings

    class S(config.Settings):
        cerebras_api_key: str = Field(default="", validation_alias="CEREBRAS_API_KEY", repr=False)
        cerebras_model: str = "gpt-oss-120b"
        cerebras_vision_model: str = "qwen-3.8-27b"
        cerebras_daily_budget: float = 0.50
        cerebras_prices: Annotated[dict[str, Any] | str, NoDecode] = ""

        @field_validator("cerebras_daily_budget", mode="before")
        @classmethod
        def _cerebras_budget(cls, v):
            try:
                f = float(str(v).strip().lstrip("$"))
            except (TypeError, ValueError):
                return 0.50
            return f if f == f and 0 <= f < float("inf") else 0.50

    return S


@pytest.fixture
def cset(monkeypatch):
    """NEO.app's settings with a (fake) Cerebras key; a fresh pool, accounts and cooldowns."""

    def make(**env):
        monkeypatch.setenv("CEREBRAS_API_KEY", KEY)
        for k, v in env.items():
            monkeypatch.setenv(k, str(v))
        s = _settings_class()()
        s.ensure_dirs()
        monkeypatch.setattr(config, "_settings", s)
        return s

    from neo2.core import health

    monkeypatch.setattr(hooks, "_accounts", weakref.WeakKeyDictionary())
    monkeypatch.setattr(prov, "_cache", {})
    monkeypatch.setattr(prov, "_demoted", {})
    monkeypatch.setattr(health, "_live", {})  # (NEO.app's problem list: process-wide)
    return make


def completion(
    content: str | None = "ok", *, tool=None, finish="stop", usage=(1000, 200), model=cb.TEXT_MODEL
):
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool:
        msg["tool_calls"] = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": tool[0], "arguments": json.dumps(tool[1])},
            }
        ]
    out: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
    }
    if usage:
        out["usage"] = {"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": sum(usage)}
    return out


MODELS = {
    "object": "list",
    "data": [
        {"id": cb.TEXT_MODEL, "object": "model", "created": 0, "owned_by": "Cerebras"},
        {"id": cb.VISION_MODEL, "object": "model", "created": 0, "owned_by": "Cerebras"},
    ],
}


class Fake:
    """Cerebras' API as an httpx.MockTransport: replies queued per test, every request kept."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict[str, Any]] = []
        self.replies: list[tuple[int, Any, dict[str, str]]] = []
        self.default: tuple[int, Any, dict[str, str]] = (200, completion(), {})
        self.models = MODELS
        self.key = KEY

    def reply(self, status: int = 200, payload: Any = None, headers: dict[str, str] | None = None) -> None:
        self.replies.append((status, completion() if payload is None else payload, headers or {}))

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert request.headers["authorization"] == f"Bearer {self.key}"
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json=self.models)
        self.bodies.append(json.loads(request.content))
        status, payload, headers = self.replies.pop(0) if self.replies else self.default
        if isinstance(payload, str):  # a raw stream body
            return httpx.Response(
                status, content=payload.encode(), headers={"content-type": "text/event-stream", **headers}
            )
        return httpx.Response(status, json=payload, headers=headers)

    @property
    def chats(self) -> int:
        return len(self.bodies)


@pytest.fixture
def fake(monkeypatch):
    f = Fake()
    monkeypatch.setattr(cb, "_transport", httpx.MockTransport(f.handler))
    return f


def ledger_file(s) -> dict:
    return json.loads((s.data_dir / spend.FILE).read_text())


TOOLS = [
    ToolSpec("ax_press", "Press an element", {"type": "object", "properties": {"id": {"type": "string"}}})
]
ASK = [Message.user("press the OK button")]


# ---- no key: nothing changes ----------------------------------------------------------------


def test_without_a_key_there_is_no_cerebras_brain(monkeypatch):
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    with pytest.raises(Unavailable, match="CEREBRAS_API_KEY"):
        cb.cerebras()
    with pytest.raises(Unavailable):
        cb.cerebras_vision()
    assert asyncio.run(cb.prewarm()) is False


def test_settings_read_forgivingly(cset):
    s = cset(
        NEO_CEREBRAS_DAILY_BUDGET="$1.25", NEO_CEREBRAS_PRICES='{"gpt-oss-120b": [0.4, 0.8], "x": "bad"}'
    )
    c = cb.conf(s)
    assert c.api_key == KEY and c.model == "gpt-oss-120b" and c.vision_model == "qwen-3.8-27b"
    assert c.daily_budget == 1.25
    assert c.prices["gpt-oss-120b"] == (0.4, 0.8) and "x" not in c.prices
    assert c.prices["qwen-3.8-27b"] == spend.PRICES["qwen-3.8-27b"]
    assert "csk" not in repr(s) and "csk" not in repr(c)  # a printed setting never shows the key
    s2 = cset(NEO_CEREBRAS_DAILY_BUDGET="lots", NEO_CEREBRAS_PRICES="{not json")
    c2 = cb.conf(s2)
    assert c2.daily_budget == 0.5 and c2.prices == spend.PRICES  # a typo never stops NEO starting


# ---- a step, as the agent loop asks for one ---------------------------------------------------


async def test_a_text_step_goes_through_with_low_effort_and_room_to_think(cset, fake):
    s = cset()
    fake.reply(payload=completion(None, tool=("ax_press", {"id": "e12"}), finish="tool_calls"))
    p = cb.cerebras()
    turn = await p.complete(ASK, system="sys", tools=TOOLS, effort="low", max_tokens=64)
    assert [c.name for c in turn.tool_calls] == ["ax_press"] and turn.tool_calls[0].args == {"id": "e12"}
    assert turn.sees is False  # text only: the loop refuses any coordinates from it
    body = fake.bodies[0]
    assert body["model"] == "gpt-oss-120b" and body["reasoning_effort"] == "low"
    assert body["max_tokens"] >= cb.ROOM["low"]  # 64 would all go to thinking
    assert str(fake.requests[0].url) == "https://api.cerebras.ai/v1/chat/completions"
    day = ledger_file(s)
    assert day["tokens"]["gpt-oss-120b"] == [1000, 200]
    assert math.isclose(day["usd"], (1000 * 0.35 + 200 * 0.75) / 1e6)


async def test_a_reply_that_spent_all_its_room_thinking_is_asked_again_once(cset, fake):
    s = cset()
    fake.reply(payload=completion(None, finish="length", usage=(500, 1024)))
    fake.reply(payload=completion("Done."))
    p = cb.cerebras()
    turn = await p.complete(ASK, effort="medium", max_tokens=1000)
    assert turn.text == "Done."
    first, second = fake.bodies
    assert first["reasoning_effort"] == "medium" and second["reasoning_effort"] == "low"
    assert second["max_tokens"] >= 2 * first["max_tokens"]
    assert ledger_file(s)["tokens"]["gpt-oss-120b"] == [1500, 1224]  # the empty reply was paid for too

    fake.reply(payload=completion(None, finish="length"))
    fake.reply(payload=completion(None, finish="length"))
    with pytest.raises(Truncated) as e:
        await p.complete(ASK, max_tokens=1000)
    assert isinstance(e.value, BadRequest) and e.value.kind == "bad_request"  # request-shaped: no bench


async def test_groq_keeps_its_old_behaviour(monkeypatch):
    """The hooks added to the shared OpenAI-compatible class change nothing for Groq or local."""
    p = OpenAICompatProvider(name="groq", base_url="https://api.groq.test/v1", api_key="k", model="m",
                             reasoning_effort=True, token_budget=8000)  # fmt: skip
    seen: list[dict] = []

    async def create(**kw):
        seen.append(kw)
        if kw.get("stream"):

            async def gen():
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hi"))])

            return gen()
        msg = SimpleNamespace(content=None, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="length")], usage=None)

    monkeypatch.setattr(p._client.chat.completions, "create", create)
    turn = await p.complete(ASK, max_tokens=16)
    assert turn.text == "" and not turn.tool_calls  # as before: no Truncated for Groq
    assert [c async for c in p.stream(ASK)] == ["hi"]
    assert "reasoning_effort" not in seen[1]  # its streams as before


# ---- rate limits: what the server's headers say ----------------------------------------------


async def test_rate_headers_stop_a_request_that_would_not_fit(cset, fake, monkeypatch):
    cset()
    fake.reply(headers={
        "x-ratelimit-limit-requests-minute": "1000",
        "x-ratelimit-remaining-requests-minute": "0",
        "x-ratelimit-reset-requests-minute": "30.5",
        "x-ratelimit-remaining-tokens-minute": "499000",
        "x-ratelimit-reset-tokens-minute": "30.5",
    })  # fmt: skip
    p = cb.cerebras()
    await p.complete(ASK)
    with pytest.raises(RateLimited) as e:
        await p.complete(ASK)
    assert fake.chats == 1  # not sent: the server said there was no room
    assert 29 < e.value.retry_after <= 30.5
    assert p.cooling_until() > time.time() + 29  # the chain asks the others first


async def test_a_window_that_reopens_in_a_moment_is_waited_for(cset, fake, monkeypatch):
    cset()
    slept: list[float] = []

    async def sleep(s):
        slept.append(s)

    monkeypatch.setattr(cb, "_sleep", sleep)
    fake.reply(headers={"x-ratelimit-remaining-tokens-minute": "1", "x-ratelimit-reset-tokens-minute": "0.8"})
    p = cb.cerebras()
    await p.complete(ASK)
    await p.complete(ASK)
    assert fake.chats == 2 and slept and 0 < slept[0] <= 0.8


async def test_a_daily_limit_used_up_is_out_until_it_resets(cset, fake):
    cset()
    fake.reply(headers={"x-ratelimit-remaining-tokens-day": "0", "x-ratelimit-reset-tokens-day": "40000"})
    p = cb.cerebras()
    await p.complete(ASK)
    with pytest.raises(cb.Refused) as e:
        await p.complete(ASK)
    assert e.value.kind == "quota" and e.value.until > time.time() + 39000
    assert "limit is used up" in e.value.words


async def test_a_429_takes_its_wait_from_the_headers(cset, fake):
    cset()
    fake.reply(429, {"message": "Requests per minute limit exceeded", "type": "too_many_requests_error"},
               {"x-ratelimit-remaining-requests-minute": "0", "x-ratelimit-reset-requests-minute": "12"})  # fmt: skip
    p = cb.cerebras()
    with pytest.raises(RateLimited) as e:
        await p.complete(ASK)
    assert not isinstance(e.value, cb.Refused) and 11 < e.value.retry_after <= 12


def test_reset_values_in_either_notation():
    assert cb._seconds("11.5") == 11.5 and cb._seconds("1m2.5s") == 62.5 and cb._seconds("250ms") == 0.25
    assert cb._seconds("2h") == 7200 and cb._seconds("") is None and cb._seconds("soon") is None


# ---- refusals: key, credits, models ----------------------------------------------------------


async def test_an_invalid_key_is_refused_for_the_day_and_said_once(cset, fake, monkeypatch):
    s = cset()
    notes: list[str] = []

    async def note(text):
        notes.append(text)

    from neo2.core.events import bus

    monkeypatch.setattr(bus(), "note", note)
    fake.reply(401, {"message": "Wrong API Key", "type": "invalid_request_error", "code": "wrong_api_key"})
    p = cb.cerebras()
    with pytest.raises(cb.Refused) as e:
        await p.complete(ASK)
    assert "refused the API key" in e.value.words and e.value.until > time.time()
    with pytest.raises(cb.Refused):
        await p.complete(ASK)
    assert fake.chats == 1  # the second one never left
    assert len([n for n in notes if "API key" in n]) == 1
    from neo2.core import health

    assert any("refused the API key" in x for x in health.problems())  # the overlay's problem list
    assert KEY not in json.dumps(ledger_file(s))  # the ledger keeps a fingerprint, never the key

    # a restart the same day: still refused, without a round trip
    monkeypatch.setattr(hooks, "_accounts", weakref.WeakKeyDictionary())
    monkeypatch.setattr(prov, "_cache", {})
    with pytest.raises(cb.Refused):
        await cb.cerebras().complete(ASK)
    assert fake.chats == 1
    # a new key is tried at once
    monkeypatch.setenv("CEREBRAS_API_KEY", KEY + "2")
    monkeypatch.setattr(config, "_settings", _settings_class()())
    fake.key = KEY + "2"
    assert (await cb.cerebras().complete(ASK)).text == "ok"
    assert fake.chats == 2


async def test_an_account_out_of_credits_is_refused_for_the_day(cset, fake):
    cset()
    fake.reply(402, {"message": "Payment required: insufficient credits", "type": "payment_required"})
    p = cb.cerebras()
    with pytest.raises(cb.Refused) as e:
        await p.complete(ASK)
    assert "out of credits" in e.value.words
    v = cb.cerebras_vision()  # the whole account: the seeing brain too
    with pytest.raises(cb.Refused):
        await v.complete(ASK)
    assert fake.chats == 1


async def test_billing_words_in_a_429_are_a_refusal_not_a_wait(cset, fake):
    cset()
    fake.reply(429, {"message": "Your account has insufficient_balance; please top up", "type": "billing"})
    with pytest.raises(cb.Refused, match="out of credits"):
        await cb.cerebras().complete(ASK)


async def test_a_model_the_key_may_not_use_takes_out_only_that_brain(cset, fake):
    cset()
    fake.reply(403, {"message": "Model access denied"})
    with pytest.raises(cb.Refused):
        await cb.cerebras_vision().complete(ASK)
    turn = await cb.cerebras().complete(ASK)  # the text brain still answers
    assert turn.text == "ok"


async def test_a_model_not_offered_is_skipped_without_a_round_trip(cset, fake):
    cset()
    fake.models = {"object": "list", "data": [MODELS["data"][0]]}  # no vision model for this key
    assert await cb.prewarm() is True
    with pytest.raises(Unavailable, match="doesn't offer"):
        await cb.cerebras_vision().complete(ASK)
    assert fake.chats == 0


async def test_server_trouble_is_overloaded(cset, fake):
    cset()
    fake.reply(503, {"message": "We're experiencing high traffic right now"})
    with pytest.raises(Overloaded):
        await cb.cerebras().complete(ASK)


# ---- money ------------------------------------------------------------------------------------


def test_prices_and_ledger_arithmetic(tmp_path):
    now = [time.mktime((2026, 9, 27, 23, 0, 0, 0, 0, -1))]
    led = spend.Ledger(tmp_path / "s.json", daily_budget=0.5, key=KEY, clock=lambda: now[0])
    assert math.isclose(led.cost("gpt-oss-120b", 1_000_000, 1_000_000), 1.10)
    assert math.isclose(led.cost("qwen-3.8-27b", 1_000_000, 0), 0.99)
    assert led.cost("some-new-model", 1_000_000, 0) == max(p[0] for p in spend.PRICES.values())  # dearest
    assert led.add("gpt-oss-120b", 1_000_000, 0) == []  # $0.35: 70%
    assert led.add("gpt-oss-120b", 200_000, 0) == ["warn"]  # $0.42: 84%
    assert led.add("gpt-oss-120b", 10, 0) == []  # said once a day
    assert led.add("gpt-oss-120b", 300_000, 0) == ["spent"]
    assert led.over()
    again = spend.Ledger(tmp_path / "s.json", daily_budget=0.5, key=KEY, clock=lambda: now[0])
    assert again.over() and math.isclose(again.today(), 0.5250035)  # kept across a restart
    now[0] += 2 * 3600  # past midnight: a new day
    assert not again.over() and again.today() == 0.0
    assert spend.budget("-1") == 0.5 and spend.budget(0) == 0.0 and spend.budget("nan") == 0.5
    assert oct((tmp_path / "s.json").stat().st_mode & 0o777) == "0o600"


async def test_the_budget_warns_at_80_percent_then_skips_cerebras_until_tomorrow(cset, fake, monkeypatch):
    cset(NEO_CEREBRAS_DAILY_BUDGET="0.001")  # a tenth of a cent
    notes: list[str] = []

    async def note(text):
        notes.append(text)

    from neo2.core.events import bus

    monkeypatch.setattr(bus(), "note", note)
    p = cb.cerebras()
    fake.reply(payload=completion(usage=(2000, 200)))  # $0.00085: 85%
    await p.complete(ASK)
    assert notes == ["Cerebras has used $0.0008 of today's $0.0010 budget"]
    fake.reply(payload=completion(usage=(2000, 200)))  # over
    await p.complete(ASK)
    assert any("used up" in n for n in notes)
    with pytest.raises(cb.Refused) as e:
        await p.complete(ASK)
    assert fake.chats == 2  # skipped without a round trip
    assert "budget" in e.value.words and e.value.until > time.time()
    assert sum("used up" in n for n in notes) == 1  # said once, not per request


async def test_a_chain_fails_over_past_a_spent_budget(cset, fake, monkeypatch):
    s = cset(NEO_CEREBRAS_DAILY_BUDGET="0.001", GEMINI_API_KEY="g-test")  # (a key NEO knows: see the wiring)
    _spend_it(s)

    class Backup:
        name, model, supports_vision, supports_tools = "groq", "m", False, True

        async def complete(self, messages, **kw):
            return Turn(text="from groq")

    real = prov._build
    monkeypatch.setattr(prov, "_build", lambda n: cb.cerebras() if n == cb.TEXT_BRAIN else real(n))
    chain = prov.Chain([cb.TEXT_BRAIN, Backup()])
    turn = await chain.complete(ASK)
    assert turn.text == "from groq" and chain.last_used == "groq" and fake.chats == 0
    assert chain._providers()[-1].name == cb.TEXT_BRAIN  # it knows it's out: asked last, no round trip
    # every brain out: the failure carries Cerebras' own reason, in `words` (what the stop message
    # says once base.failure_words reads it: test_wired_the_stop_message_gives_cerebras_own_reason)
    chain2 = prov.Chain([cb.TEXT_BRAIN])
    with pytest.raises(AllBrainsFailed) as e:
        await chain2.complete(ASK)
    out = e.value.failures[0][1]
    assert out.words == "Cerebras' daily budget of $0.0010 is used up" and out.kind == "quota"
    assert prov._demoted[cb.TEXT_BRAIN] >= out.until  # benched until midnight, not for 5 minutes


async def test_a_streamed_answer_is_counted_even_without_usage(cset, fake):
    s = cset()
    chunk = {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "gpt-oss-120b",
             "choices": [{"index": 0, "delta": {"content": "Hello there"}, "finish_reason": None}]}  # fmt: skip
    fake.reply(payload=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n")
    out = [c async for c in cb.cerebras().stream(ASK, max_tokens=64)]
    assert out == ["Hello there"]
    assert fake.bodies[0]["reasoning_effort"] == "low" and fake.bodies[0]["max_tokens"] >= cb.ROOM["low"]
    tin, tout = ledger_file(s)["tokens"]["gpt-oss-120b"]
    assert tin > 0 and tout >= 4  # estimated: the prompt as fitted, the text plus unseen thinking

    fake.reply(payload="data: [DONE]\n\n")  # thought, said nothing: another brain may say it
    with pytest.raises(Truncated):
        _ = [c async for c in cb.cerebras().stream(ASK)]


# ---- pictures ---------------------------------------------------------------------------------


def _shot(n: int, fill: bytes = b"x") -> ImagePart:
    return ImagePart(fill * n, "image/jpeg")


def _history(*sizes: int) -> list[Message]:
    msgs = [Message.user("what's on the board?")]
    for i, n in enumerate(sizes):
        msgs.append(Message.assistant(tool_calls=[ToolCall(f"c{i}", "screenshot", {})]))
        msgs.append(
            Message.tool([ToolResult(f"c{i}", "screenshot", f"shot {i}", images=[_shot(n, bytes([65 + i]))])])
        )
    return msgs


def test_at_most_two_pictures_newest_first_and_never_blind():
    msgs = _history(100, 100, 100)
    out = cb.cap_pictures(msgs, most=2, max_bytes=10_000)
    kept = [r.images for m in out for r in m.tool_results]
    assert [len(x) for x in kept] == [0, 1, 1]
    assert "left out" in out[2].tool_results[0].content and out[6] is msgs[6]
    assert [len(r.images) for m in msgs for r in m.tool_results] == [1, 1, 1]  # the caller's are untouched
    # bytes: only what fits (base64) goes
    out = cb.cap_pictures(_history(3000, 3000), most=2, max_bytes=4100)
    assert [len(r.images) for m in out for r in m.tool_results] == [0, 1]
    # the newest can't go: refused, not answered blind
    with pytest.raises(RequestTooLarge):
        cb.cap_pictures(_history(100, 9000), most=2, max_bytes=4000)


async def test_the_seeing_brain_sends_at_most_two_pictures(cset, fake):
    cset()
    fake.default = (200, completion("the knight moved", model=cb.VISION_MODEL), {})
    v = cb.cerebras_vision()
    assert v.supports_vision and v.max_images == 2 and v.max_request_bytes == 10 * 1024 * 1024
    turn = await v.complete(_history(2000, 2000, 2000), effort="low")
    assert turn.sees is True and turn.text == "the knight moved"
    body = fake.bodies[0]
    parts = [p for m in body["messages"] if isinstance(m["content"], list) for p in m["content"]]
    assert sum(p["type"] == "image_url" for p in parts) == 2
    assert body["model"] == "qwen-3.8-27b" and body["reasoning_effort"] == "low"  # not its default xhigh
    assert len(fake.requests[0].content) < 10 * 1024 * 1024


# ---- one account per agent, one warm connection per account ----------------------------------


def test_one_account_per_pool_with_its_own_ledger(cset, monkeypatch, tmp_path):
    cset()
    pools = [prov.BrainPool(), prov.BrainPool()]
    which = [0]
    monkeypatch.setattr(prov, "pool", lambda: pools[which[0]])
    a0 = cb.account(cb.Conf(api_key=KEY, data_dir=tmp_path / "a"))
    assert cb.account(cb.Conf(api_key=KEY, data_dir=tmp_path / "a")) is a0
    which[0] = 1
    a1 = cb.account(cb.Conf(api_key=KEY, data_dir=tmp_path / "b"))
    assert a1 is not a0 and a1.ledger.path == tmp_path / "b" / spend.FILE
    a0.ledger.add("gpt-oss-120b", 1000, 0)
    assert a1.ledger.today() == 0.0  # one agent's spend is its own


async def test_both_brains_share_one_connection_pool_per_event_loop(cset, fake):
    cset()
    t, v = cb.cerebras(), cb.cerebras_vision()
    assert t.account is v.account
    assert t.client()._client._transport is v.client()._client._transport  # one pool: one warm connection
    first = t.client()

    def other_loop():
        async def get():
            return t.client()

        return asyncio.run(get())

    assert await asyncio.to_thread(other_loop) is not first  # a new loop: new connections


async def test_announce_never_reaches_the_app_bus_from_an_unbound_library_call(cset, monkeypatch):
    cset()
    from neo2.core import runtime

    monkeypatch.setattr(runtime, "_library", True)
    a = cb.Account(cb.Conf(api_key=KEY))  # a host's own (in memory)
    a._pending.append("hello")
    await a.announce()  # no NoRuntimeError, nothing said to NEO.app's bus
    assert a._pending == [] and a.ledger.path is None


def test_hosted_provider_keeps_everything_in_memory(tmp_path):
    p = cb.hosted(KEY, vision=True, daily_budget=2)
    assert p.name == cb.VISION_BRAIN and p.model == cb.VISION_MODEL and p.account.ledger.path is None
    assert p.account.ledger.daily_budget == 2
    with pytest.raises(Exception, match="needs an API key"):
        cb.hosted("")


# ---- where it goes in a chain -----------------------------------------------------------------


def test_placed_ahead_of_groq_as_decision_49_says(cset):
    cset()  # (a key: without one, no order changes)
    w = cb.with_cerebras
    assert w(["gemini", "groq", "local"], "agent") == ["gemini", "cerebras", "groq", "local"]
    assert w(["groq", "gemini", "local"], "fast") == ["cerebras", "groq", "gemini", "local"]
    assert w(["gemini_lite", "groq", "gemini", "local"], "light")[:3] == ["gemini_lite", "cerebras", "groq"]
    assert w(["gemini", "local"], "fast") == ["gemini", "cerebras", "local"]
    assert w(["local"], "offline") == ["local"]
    # vision: the seeing one after the seeing brains, ahead of Groq's vision brain when there is one
    assert w(["gemini", "claude", "groq", "local"], "vision") == [
        "gemini", "claude", "cerebras_vision", "cerebras", "groq", "local"]  # fmt: skip
    assert w(["gemini", "gemini_lite", "groq_vision", "groq"], "vision") == [
        "gemini", "gemini_lite", "cerebras_vision", "groq_vision", "cerebras", "groq"]  # fmt: skip
    assert w(["cerebras", "groq"], "agent") == ["cerebras", "groq"]  # already named: left alone


# ---- speed: the warm connection (a local server with a slow handshake) ------------------------


class SlowHandshakeServer:
    """A plain-HTTP stand-in on 127.0.0.1 whose every new connection costs `handshake` seconds
    before its first answer (a cold TCP+TLS round trip); requests on a kept-alive one answer at
    once. Counts connections."""

    def __init__(self, handshake: float) -> None:
        self.handshake = handshake
        self.connections = 0
        self.writers: list[asyncio.StreamWriter] = []
        self.server: asyncio.base_events.Server | None = None

    async def start(self) -> str:
        self.server = await asyncio.start_server(self._serve, "127.0.0.1", 0)
        port = self.server.sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{port}/v1"

    async def _serve(self, r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        self.connections += 1
        self.writers.append(w)
        await asyncio.sleep(self.handshake)
        try:
            while True:
                head = await r.readuntil(b"\r\n\r\n")
                lines = head.decode().split("\r\n")
                n = next(
                    (int(x.split(":", 1)[1]) for x in lines if x.lower().startswith("content-length:")), 0
                )
                if n:
                    await r.readexactly(n)
                body = json.dumps(MODELS if " /v1/models " in lines[0] else completion()).encode()
                w.write(b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: "
                        + str(len(body)).encode() + b"\r\n\r\n" + body)  # fmt: skip
                await w.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            w.close()

    async def stop(self) -> None:
        for w in self.writers:
            w.close()
        if self.server:
            self.server.close()
            await self.server.wait_closed()


async def test_prewarm_takes_the_cold_round_trip_off_the_first_step(tmp_path, monkeypatch):
    srv = SlowHandshakeServer(handshake=0.4)
    url = await srv.start()
    try:
        a = cb.Account(cb.Conf(api_key=KEY, data_dir=tmp_path), base_url=url)
        t = cb.CerebrasProvider(a, model=cb.TEXT_MODEL, name=cb.TEXT_BRAIN, vision=False)
        v = cb.CerebrasProvider(a, model=cb.VISION_MODEL, name=cb.VISION_BRAIN, vision=True)
        assert await a.prewarm() is True
        t0 = time.perf_counter()
        await t.complete(ASK, effort="low")
        await v.complete(ASK, effort="low")
        await t.complete(ASK, effort="low")
        warm = time.perf_counter() - t0
        assert srv.connections == 1  # prewarm's connection served all three steps, both brains
        assert await a.prewarm() is False  # still warm: nothing to do
        assert warm < 0.4  # none of the steps paid the handshake
        await a.client(cb.TEXT_MODEL).close()
    finally:
        await srv.stop()


async def test_a_kept_alive_connection_the_server_closed_is_retried_once_but_a_timeout_is_not(cset, fake):
    cset()
    real = fake.handler
    state = {"n": 0}

    def flaky(request):
        state["n"] += 1
        if state["n"] == 1:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
        return real(request)

    cb._transport = httpx.MockTransport(flaky)  # (restored by the fake fixture's monkeypatch)
    assert (await cb.cerebras().complete(ASK)).text == "ok"
    assert state["n"] == 2

    def slow(request):
        raise httpx.ReadTimeout("timed out")

    cb._transport = httpx.MockTransport(slow)
    hooks._accounts.clear()  # a new account: connections over the new transport
    with pytest.raises(Unavailable):
        await cb.cerebras().complete(ASK)


async def test_prewarm_soon_runs_in_the_background_and_only_with_a_key(cset, fake, monkeypatch):
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    cb.prewarm_soon()
    assert not hooks._warming
    cset()
    cb.prewarm_soon()
    assert len(hooks._warming) == 1
    await asyncio.gather(*hooks._warming)
    assert [r.url.path for r in fake.requests] == ["/v1/models"] and not hooks._warming


def test_a_damaged_or_unwritable_ledger_never_fails_an_answer(tmp_path, monkeypatch, caplog):
    f = tmp_path / spend.FILE
    day = time.strftime("%Y-%m-%d")
    f.write_text(json.dumps({"version": 1, "day": day, "usd": "x", "tokens": [], "said": 3, "refused": [1]}))
    odd = spend.Ledger(f, daily_budget=0.5, key=KEY)
    assert odd.today() == 0.0 and odd.refusal("account") is None and odd.refuse("account", "w") > time.time()
    f.write_text("{not json")
    led = spend.Ledger(f, daily_budget=0.5, key=KEY)
    assert led.today() == 0.0 and led.refusal("account") is None
    assert led.add("gpt-oss-120b", 1000, 0) == []
    assert json.loads(f.read_text())["tokens"]["gpt-oss-120b"] == [1000, 0]  # rewritten whole
    assert sum("unreadable" in r.message for r in caplog.records) == 1

    def full(*a, **k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(spend.os, "open", full)
    assert led.add("gpt-oss-120b", 1000, 0) == []  # counted in memory; the answer isn't lost
    assert led.add("gpt-oss-120b", 1000, 0) == []
    assert led._state["tokens"]["gpt-oss-120b"][0] == 3000  # (not the file's stale 1000 again)


async def test_each_embedded_agent_has_its_own_account_ledger_and_notes(tmp_path, fake, monkeypatch):
    """Library mode: an agent's Cerebras spend, refusals and notes are its own (its pool, its data
    dir, its conversation's bus) — never NEO.app's ~/.neo or another agent's."""
    from neo2.core import health, runtime
    from neo2.core.events import bus
    from tests.neo_lib import make_runtime

    monkeypatch.setattr(hooks, "_accounts", weakref.WeakKeyDictionary())
    monkeypatch.setattr(health, "_live", {})
    a, b = make_runtime(tmp_path / "a", "a"), make_runtime(tmp_path / "b", "b")
    heard: dict[str, list[str]] = {"a": [], "b": []}
    fake.reply(401, {"message": "Wrong API Key"})
    for rt in (a, b):
        with runtime.bind(rt.scope()):
            bus().subscribe(
                lambda ev, n=rt.name: heard[n].append(ev.data.get("text")) if ev.type == "note" else None
            )
            c = cb.Conf(
                api_key=KEY, data_dir=rt.settings.data_dir
            )  # (the host's key, as its settings give it)
            p = cb.CerebrasProvider(cb.account(c), model=cb.TEXT_MODEL, name=cb.TEXT_BRAIN, vision=False)
            if rt is a:
                with pytest.raises(cb.Refused):
                    await p.complete(ASK)
            else:
                assert (await p.complete(ASK)).text == "ok"  # b's account isn't refused by a's 401
    assert (tmp_path / "a" / spend.FILE).exists() and (tmp_path / "b" / spend.FILE).exists()
    assert json.loads((tmp_path / "b" / spend.FILE).read_text())["tokens"]["gpt-oss-120b"][0] == 1000
    assert "gpt-oss-120b" not in json.loads((tmp_path / "a" / spend.FILE).read_text())["tokens"]
    assert any("API key" in t for t in heard["a"]) and not any("API key" in t for t in heard["b"])
    assert not any("Cerebras" in x for x in health._live.values())  # never NEO.app's problem list


# ---- a per-minute 429 is a per-minute wait, whatever a day window has left --------------------


async def test_a_minute_429_is_a_minute_wait_even_with_a_day_window_under_a_tenth(cset, fake):
    cset()
    fake.reply(429, {"message": "Requests per minute limit exceeded - too many requests sent.",
                     "type": "too_many_requests_error", "code": "request_quota_exceeded"}, {
        "x-ratelimit-limit-requests-minute": "1000",
        "x-ratelimit-remaining-requests-minute": "0",
        "x-ratelimit-reset-requests-minute": "12",
        "x-ratelimit-limit-tokens-day": "720000000",
        "x-ratelimit-remaining-tokens-day": "60000000",  # 8% of the day left: plenty for a step
        "x-ratelimit-reset-tokens-day": "50000",
    })  # fmt: skip
    p = cb.cerebras()
    with pytest.raises(RateLimited) as e:
        await p.complete(ASK)
    assert not isinstance(e.value, cb.Refused) and 11 < e.value.retry_after <= 12
    assert p.cooling_until() < time.time() + 13  # benched for the minute, not the day


def test_after_a_429_only_a_window_too_small_for_the_request_counts():
    now = 1000.0
    lim = cb.Limits()
    lim.update({"x-ratelimit-remaining-requests-minute": "0", "x-ratelimit-reset-requests-minute": "12",
                "x-ratelimit-remaining-tokens-day": "2000", "x-ratelimit-reset-tokens-day": "50000"}, now)  # fmt: skip
    assert lim.refused_for(now, 500) == 12  # the day still takes a 500-token request
    assert lim.refused_for(now, 3000) == 50000  # it doesn't take a 3000-token one: out for the day


# ---- billing: said by the status or a code field, never by words a request happened to quote ----


@pytest.mark.parametrize(
    "status,body",
    [
        (400, {"message": "Invalid 'messages[1].content': open the desktop update panel"}),
        (422, {"message": "field required: input 'no credit card on file needed'"}),
        (400, {"message": "Invalid 'messages[0].content': why does it say invalid api key?"}),
        (500, {"message": "internal error while billing usage; retry"}),
        (503, {"message": "payment service unavailable: please top up later", "type": "server_error"}),
    ],
)
async def test_words_in_a_request_error_never_refuse_the_account(cset, fake, status, body):
    s = cset()
    fake.reply(status, body)
    with pytest.raises(Exception) as e:  # noqa: PT011 — whatever it is, it isn't a refusal
        await cb.cerebras().complete(ASK)
    assert not isinstance(e.value, cb.Refused)
    assert not (s.data_dir / spend.FILE).exists() or not ledger_file(s)["refused"]
    assert (await cb.cerebras_vision().complete(ASK)).text == "ok"  # the account is fine


async def test_a_billing_code_on_a_403_refuses_the_account(cset, fake):
    cset()
    fake.reply(
        403, {"message": "Access denied", "type": "invalid_request_error", "code": "insufficient_funds"}
    )
    with pytest.raises(cb.Refused, match="out of credits"):
        await cb.cerebras().complete(ASK)
    with pytest.raises(cb.Refused):
        await cb.cerebras_vision().complete(ASK)
    assert fake.chats == 1


# ---- a prewarm learns only what's about the account --------------------------------------------


@pytest.mark.parametrize("status", [403, 404, 429, 500])
async def test_a_models_list_error_benches_nothing(cset, fake, status):
    s = cset()
    real = fake.handler

    def handler(request):
        if request.url.path.endswith("/models"):
            fake.requests.append(request)
            return httpx.Response(status, json={"message": "nope"})
        return real(request)

    cb._transport = httpx.MockTransport(handler)
    assert await cb.prewarm() is False
    assert (await cb.cerebras().complete(ASK)).text == "ok"  # the text brain was never benched
    assert fake.chats == 1
    assert not (s.data_dir / spend.FILE).exists() or not ledger_file(s)["refused"]


async def test_a_prewarm_that_meets_a_refused_key_refuses_it_for_the_day(cset, fake):
    cset()
    real = fake.handler

    def handler(request):
        if request.url.path.endswith("/models"):
            fake.requests.append(request)
            return httpx.Response(401, json={"message": "Wrong API Key", "code": "wrong_api_key"})
        return real(request)

    cb._transport = httpx.MockTransport(handler)
    assert await cb.prewarm() is False
    with pytest.raises(cb.Refused, match="API key"):
        await cb.cerebras().complete(ASK)
    assert fake.chats == 0


# ---- what the user hears -------------------------------------------------------------------------


async def test_the_budget_warning_and_the_skip_are_spoken_with_the_reply(cset, fake):
    cset(NEO_CEREBRAS_DAILY_BUDGET="0.001")
    assert cb.spoken_notes() == ""
    fake.reply(payload=completion(usage=(2000, 200)))  # 85%
    await cb.cerebras().complete(ASK)
    said = cb.spoken_notes()
    assert "Cerebras has used $0.0008 of today's $0.0010 budget" in said
    assert cb.spoken_notes() == ""  # said once
    fake.reply(payload=completion(usage=(2000, 200)))  # over
    await cb.cerebras().complete(ASK)
    assert "used up" in cb.spoken_notes()
    with pytest.raises(cb.Refused):
        await cb.cerebras().complete(ASK)
    assert cb.spoken_notes() == ""  # the skip was already said today


async def test_a_refused_key_is_spoken_once(cset, fake):
    cset()
    fake.reply(401, {"message": "Wrong API Key", "code": "wrong_api_key"})
    with pytest.raises(cb.Refused):
        await cb.cerebras().complete(ASK)
    assert "refused the API key" in cb.spoken_notes()
    with pytest.raises(cb.Refused):
        await cb.cerebras().complete(ASK)
    assert cb.spoken_notes() == ""


def test_spoken_notes_put_a_question_last():
    assert cb.with_spoken_notes("Done.", "Cerebras is out.") == "Done. Cerebras is out."
    assert cb.with_spoken_notes("Send it?", "Cerebras is out.", question=True) == "Cerebras is out. Send it?"
    assert cb.with_spoken_notes("", "Cerebras is out.") == "Cerebras is out."
    assert cb.with_spoken_notes("Done.", "") == "Done."


async def test_the_session_s_reply_carries_them(cset, fake):
    from neo2.core.agent.session import Reply

    cset(NEO_CEREBRAS_DAILY_BUDGET="0.001")
    done = Reply("Done.", "agent")
    assert cb.add_spoken_notes(done) is done  # nothing to say: the same reply
    fake.reply(payload=completion(usage=(2000, 200)))
    await cb.cerebras().complete(ASK)  # 85%
    quiet = cb.add_spoken_notes(Reply("Opened it.", "quick", silent=True))
    assert quiet.text.startswith("Cerebras has used") and not quiet.silent  # said, though the step was quiet
    fake.reply(payload=completion(usage=(2000, 200)))
    await cb.cerebras().complete(ASK)  # spent
    ask = cb.add_spoken_notes(Reply("Delete the file — should I go ahead?", "confirm"))
    assert ask.text.endswith("should I go ahead?") and "used up" in ask.text  # the question stays last
    assert cb.add_spoken_notes(done) is done  # each said once


def test_nothing_is_spoken_without_a_key(monkeypatch):
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    assert cb.spoken_notes() == ""


# ---- without a key (or with a $0 budget) the chain doesn't change ------------------------------


def test_no_key_or_a_zero_budget_leaves_every_order_alone(cset, monkeypatch):
    s = cset()
    agent, vision = ["gemini", "groq", "local"], ["gemini", "groq_vision", "groq"]
    assert cb.with_cerebras(agent, "agent", s) == ["gemini", "cerebras", "groq", "local"]
    assert cb.with_cerebras_eyes(("gemini", "gemini_lite", "claude", "groq_vision"), s) == (
        "gemini", "gemini_lite", "claude", "cerebras_vision", "groq_vision")  # fmt: skip
    assert cb.with_cerebras_eyes(("gemini", "claude"), s) == ("gemini", "claude", "cerebras_vision")
    assert cb.with_cerebras_eyes(("gemini", "cerebras_vision", "groq_vision"), s) == (
        "gemini", "cerebras_vision", "groq_vision")  # fmt: skip
    for off in (cset(CEREBRAS_API_KEY=""), cset(NEO_CEREBRAS_DAILY_BUDGET="0")):
        assert cb.with_cerebras(agent, "agent", off) == agent
        assert cb.with_cerebras(vision, "vision", off) == vision
        eyes = ("gemini", "gemini_lite", "claude", "groq_vision")
        assert cb.with_cerebras_eyes(eyes, off) == eyes


def test_the_hooks_the_chain_and_the_session_call_import_no_vendor_sdk(tmp_path):
    """Placing Cerebras in a chain and finishing a reply run on every request, NEO.app's and an
    embedded agent's alike: they must not import the openai SDK (the SDK's import-cost promise)."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    from neo2.core import effects

    code = (
        "import sys, json, types\n"
        "from neo2.core.providers import cerebras_hooks as h\n"
        "s = types.SimpleNamespace(cerebras_api_key='k', cerebras_daily_budget=0.5)\n"
        "o = h.with_cerebras(['gemini', 'groq', 'local'], 'agent', s)\n"
        "e = h.with_cerebras_eyes(('gemini', 'groq_vision'), s)\n"
        "print(json.dumps({'o': o, 'e': list(e), 'n': h.spoken_notes(), 'openai': 'openai' in sys.modules}))\n"
    )
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "HOME": str(tmp_path), "PYTHONPATH": str(root), "NEO_DATA_DIR": str(tmp_path / "d")}
    env["NEO_NO_EFFECTS"] = "1"
    with effects.allow("process"):  # a Python child: refused unless meant, as here
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120
        )
    assert out.returncode == 0, out.stderr[-2000:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got == {
        "o": ["gemini", "cerebras", "groq", "local"],
        "e": ["gemini", "cerebras_vision", "groq_vision"],
        "n": "",
        "openai": False,
    }


# ---- a step that starts while the prewarm is still connecting joins it ---------------------------


@pytest.mark.parametrize("head_start", [0.02, 0.3])
async def test_a_step_right_after_the_prewarm_starts_rides_its_connection(tmp_path, head_start):
    srv = SlowHandshakeServer(handshake=0.4)
    url = await srv.start()
    try:
        a = cb.Account(cb.Conf(api_key=KEY, data_dir=tmp_path), base_url=url)
        t = cb.CerebrasProvider(a, model=cb.TEXT_MODEL, name=cb.TEXT_BRAIN, vision=False)
        warm = asyncio.ensure_future(a.prewarm())  # prewarm_soon() at the wake word
        await asyncio.sleep(head_start)
        t0 = time.perf_counter()
        await t.complete(ASK, effort="low")
        first = time.perf_counter() - t0
        assert await warm is True
        assert srv.connections == 1  # one connection, not a second cold one next to the prewarm's
        assert first < 0.4 - head_start + 0.15  # the rest of the handshake, not all of it again
        await a.client(cb.TEXT_MODEL).close()
    finally:
        await srv.stop()


# ---- an agent's own settings: its key, its ledger ------------------------------------------------


def _library_settings_class():
    """neo2.core.config.LibrarySettings once the Cerebras fields are declared on Settings (the wiring);
    until then, the same fields on a subclass."""
    if "cerebras_api_key" in config.LibrarySettings.model_fields:
        return config.LibrarySettings

    class L(config.LibrarySettings):
        cerebras_api_key: str = Field(default="", validation_alias="CEREBRAS_API_KEY", repr=False)
        cerebras_model: str = "gpt-oss-120b"
        cerebras_vision_model: str = "qwen-3.8-27b"
        cerebras_daily_budget: float = 0.50
        cerebras_prices: Annotated[dict[str, Any] | str, NoDecode] = ""

    return L


async def test_a_bound_agent_s_cerebras_uses_its_own_settings_and_folder(tmp_path, fake, monkeypatch):
    from neo2.core import runtime
    from neo2.core.agent.registry import catalog

    monkeypatch.setattr(hooks, "_accounts", weakref.WeakKeyDictionary())
    monkeypatch.setenv("CEREBRAS_API_KEY", "csk-the-app-s-own-key")  # never an agent's
    L = _library_settings_class()
    s = L(data_dir=tmp_path / "agent", cerebras_api_key=KEY, cerebras_daily_budget=2.0)
    s.ensure_dirs()
    rt = runtime.Runtime("agent", s, catalog().subset(()))
    with runtime.bind(rt.scope()):
        p = cb.cerebras()
        assert p.account.conf.api_key == KEY and p.account.ledger.daily_budget == 2.0
        assert p.account.ledger.path == tmp_path / "agent" / spend.FILE
        assert (await p.complete(ASK)).text == "ok"
    assert json.loads((tmp_path / "agent" / spend.FILE).read_text())["tokens"]["gpt-oss-120b"] == [1000, 200]


# ---- the wiring, on brain-effort's chain (seeing fallbacks, a run's time budget) ------------------
# These run once this branch sits on brain-effort (the integrator merges it first) and check the
# wiring this workstream hands over: _build makes both brains; brain() places them with
# with_cerebras(order, purpose, s) and eyes=with_cerebras_eyes(EYES, s); "cerebras" is text-only;
# base.failure_words says an error's own `words` and spoken_name knows both names; _too_late
# reports a brain's cooling_reason() instead of "no time left".

on_the_chain = pytest.mark.skipif(
    not hasattr(prov, "EYES"), reason="needs brain-effort's chain: runs once the integrator merges it"
)


class _Busy:
    supports_tools = True

    def __init__(self, name: str, vision: bool = True) -> None:
        self.name, self.model, self.supports_vision = name, name, vision

    async def complete(self, messages, **kw):
        raise Overloaded(f"{self.name}: 503")

    async def stream(self, messages, **kw):
        raise Overloaded(f"{self.name}: 503")
        yield ""  # (an async generator)


def _busy(monkeypatch, *names: str, text_only: tuple[str, ...] = ("local",)) -> None:
    real = prov._build
    busy = {n: _Busy(n, vision=n not in text_only) for n in names}
    monkeypatch.setattr(prov, "_build", lambda n: busy[n] if n in busy else real(n))


def _screenshot_step() -> list[Message]:
    return [
        Message.user("what's in the front window?"),
        Message.assistant(tool_calls=[ToolCall("c0", "screenshot", {})]),
        Message.tool([ToolResult("c0", "screenshot", "shot", images=[_shot(500)])]),
    ]


@on_the_chain
async def test_wired_a_picture_step_goes_to_cerebras_vision_when_gemini_is_overloaded(
    cset, fake, monkeypatch
):
    cset(GEMINI_API_KEY="g-test")
    _busy(monkeypatch, "gemini", "gemini_lite", "local")
    fake.default = (200, completion("A save dialog is open.", model=cb.VISION_MODEL), {})
    chain = prov.brain("agent")
    turn = await chain.complete(_screenshot_step(), effort="low")
    assert chain.last_used == cb.VISION_BRAIN and turn.sees
    assert [b["model"] for b in fake.bodies] == [cb.VISION_MODEL]  # never the text model with a picture
    names = [*chain.names, *chain.eyes]
    assert names.index(cb.VISION_BRAIN) < names.index("groq_vision")  # ahead of Groq's vision brain
    assert prov.brain("vision").names.index(cb.VISION_BRAIN) < prov.brain("vision").names.index("groq_vision")


@on_the_chain
async def test_wired_a_text_step_goes_to_cerebras_ahead_of_groq(cset, fake, monkeypatch):
    cset(GEMINI_API_KEY="g-test", GROQ_API_KEY="gsk-test")
    _busy(monkeypatch, "gemini")
    chain = prov.brain("agent")
    assert chain.names.index(cb.TEXT_BRAIN) < chain.names.index("groq")
    assert (await chain.complete(ASK, effort="low")).text == "ok" and chain.last_used == cb.TEXT_BRAIN
    fast = prov.brain("fast")
    assert fast.names.index(cb.TEXT_BRAIN) < fast.names.index("groq")


@on_the_chain
async def test_wired_without_a_key_no_chain_names_cerebras(monkeypatch):
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    monkeypatch.setattr(config, "_settings", _settings_class()())
    monkeypatch.setattr(prov, "_cache", {})
    monkeypatch.setattr(prov, "_demoted", {})
    _busy(monkeypatch, "gemini", "gemini_lite", "local")
    for purpose in ("agent", "fast", "vision", "light"):
        c = prov.brain(purpose)
        assert not {cb.TEXT_BRAIN, cb.VISION_BRAIN} & {*c.names, *c.eyes}, purpose
    with pytest.raises(AllBrainsFailed) as e:
        await prov.brain("agent").complete(ASK)
    assert "Cerebras" not in e.value.spoken()
    with pytest.raises(AllBrainsFailed) as e:
        await prov.brain("agent").complete(_screenshot_step())
    assert "Cerebras" not in e.value.spoken()


def _spend_it(s) -> None:
    """Today's spend is past the (tiny) budget: the budget was lowered below what was already spent,
    so nothing has said so yet."""
    spend.Ledger(s.data_dir / spend.FILE, daily_budget=1.0, key=KEY).add(cb.TEXT_MODEL, 10_000, 0)


@on_the_chain
async def test_wired_the_stop_message_gives_cerebras_own_reason(cset, fake, monkeypatch):
    s = cset(NEO_CEREBRAS_DAILY_BUDGET="0.001", GEMINI_API_KEY="g-test")
    _spend_it(s)
    _busy(monkeypatch, "gemini", "local")
    with pytest.raises(AllBrainsFailed) as e:
        await prov.brain("agent").complete(ASK)
    said = e.value.spoken()
    assert "Cerebras' daily budget of $0.0010 is used up" in said and "free quota" not in said
    assert fake.chats == 0
    assert "skipping it until tomorrow" in cb.spoken_notes()  # the skip is said, with the reply
    with pytest.raises(AllBrainsFailed):
        await prov.brain("agent").complete(ASK)
    assert cb.spoken_notes() == ""  # once a day
    from neo2.core.providers.base import spoken_name

    assert spoken_name(cb.VISION_BRAIN) == "Cerebras vision" and spoken_name(cb.TEXT_BRAIN) == "Cerebras"


@on_the_chain
async def test_wired_a_spent_budget_inside_a_run_is_the_budget_not_the_time(cset, fake, monkeypatch):
    from neo2.core.providers import budget

    s = cset(NEO_CEREBRAS_DAILY_BUDGET="0.001", GEMINI_API_KEY="g-test")
    _spend_it(s)
    _busy(monkeypatch, "gemini", "local")
    with budget.seconds(180.0), pytest.raises(AllBrainsFailed) as e:
        await prov.brain("agent").complete(ASK)
    assert e.value.kind != "budget"  # not OutOfTime: the run had time; Cerebras had no budget
    said = e.value.spoken()
    assert "daily budget of $0.0010 is used up" in said and "time" not in said
    assert "skipping it until tomorrow" in cb.spoken_notes()


@on_the_chain
async def test_wired_the_session_says_the_notes_with_its_reply(tmp_path, fake, monkeypatch):
    from neo2.core import runtime
    from neo2.core.agent.registry import catalog
    from neo2.core.agent.session import Session
    from tests.neo_lib import scripted

    monkeypatch.setattr(hooks, "_accounts", weakref.WeakKeyDictionary())
    L = _library_settings_class()
    s = L(data_dir=tmp_path / "agent", cerebras_api_key=KEY, cerebras_daily_budget=0.001, fast_paths=False)
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
