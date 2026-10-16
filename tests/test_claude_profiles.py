"""What ClaudeProvider sends, per model (its profile), and what an API error becomes.

claude-opus-5 sends exactly what it always did (tests/golden pins those requests byte for byte).
The 5.5 models and Fable 5.1 add the thinking-binding controls (drop a replayed block that fails the
conversation check instead of a 400), in complete() and stream(); Haiku 5.5 is never sent a
fallback; a model NEO doesn't know gets adaptive thinking and plain effort only. A refused key is
KeyRefused (and the health list says so, without the key); a 429 carries its retry-after.

Every request goes into a fake client, or through the real anthropic SDK into an httpx2
MockTransport: nothing leaves the process, and every key is an obvious fake."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pytest

from neo2.core import config, health
from neo2.core import providers as prov
from neo2.core.providers import claude
from neo2.core.providers.base import (
    BadRequest,
    KeyRefused,
    Message,
    Overloaded,
    QuotaExhausted,
    RateLimited,
    ToolCall,
    ToolResult,
)
from neo2.core.providers.claude import BINDING_BETA, FALLBACK_BETA, ClaudeProvider

KEY = "sk-ant-TEST-0000-abcdefghijklmnop"
GOLDEN = Path(__file__).parent / "golden" / "claude_kwargs.json"
DROP = {"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "drop_block"}}


class Block:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)

    def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not (exclude_none and v is None)}


class Usage:
    def __init__(self, i: int = 10, o: int = 5) -> None:
        self.input_tokens, self.output_tokens = i, o


class Resp:
    def __init__(self, content=None, stop_reason="end_turn", transformations=None, usage=None) -> None:
        self.content = content if content is not None else [Block(type="text", text="ok")]
        self.stop_reason, self.usage = stop_reason, usage or Usage()
        self.input_transformations = transformations


class Stream:
    def __init__(self, final: Any = None) -> None:
        async def gen():
            for t in ("Hel", "lo"):
                yield t

        self.text_stream = gen()
        self.current_message_snapshot = final or Resp()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class Fake:
    """Takes what ClaudeProvider sends; answers `reply` (a Resp) or raises it (an exception)."""

    def __init__(self, reply: Any = None) -> None:
        self.sent: list[dict[str, Any]] = []
        self.reply = reply
        me = self

        class Messages:
            async def create(self, **kw):
                me.sent.append({"call": "create", **kw})
                if isinstance(me.reply, BaseException):
                    raise me.reply
                return me.reply or Resp()

            def stream(self, **kw):
                me.sent.append({"call": "stream", **kw})
                return Stream(me.reply if isinstance(me.reply, Resp) else None)

        class Beta:
            messages = Messages()

        self.beta = Beta()


def _plain(kw: dict[str, Any]) -> dict[str, Any]:
    import anthropic

    return {k: v for k, v in kw.items() if not isinstance(v, anthropic.NotGiven)}


async def _asked(model: str, effort: str = "high") -> tuple[dict[str, Any], dict[str, Any]]:
    fake = Fake()
    p = ClaudeProvider(model, client=fake)
    await p.complete([Message.user("hi")], effort=effort)  # type: ignore[arg-type]
    _ = [t async for t in p.stream([Message.user("hi")])]
    return _plain(fake.sent[0]), _plain(fake.sent[1])


# ---- the profiles ---------------------------------------------------------------------------------


async def test_opus_5_sends_exactly_what_the_golden_pinned():
    golden = json.loads(GOLDEN.read_text())["host requests"]
    create, stream = await _asked("claude-opus-5")
    want_create = {k: v for k, v in golden[2].items() if v != "<NOT_GIVEN>"}  # (effort "high" → xhigh)
    got = json.loads(json.dumps(create, default=str))
    assert got == want_create
    want_stream = {k: v for k, v in golden[-1].items() if v != "<NOT_GIVEN>"}
    assert {k: stream[k] for k in ("betas", "fallbacks", "output_config", "call")} == {
        k: want_stream[k] for k in ("betas", "fallbacks", "output_config", "call")
    }
    assert "thinking" not in stream and "thinking" not in want_stream


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1"])
async def test_the_5_5_models_and_fable_drop_a_stale_thinking_block_instead_of_failing(model):
    create, stream = await _asked(model)
    for kw in (create, stream):
        assert kw["thinking"] == DROP
        assert kw["betas"] == [FALLBACK_BETA, BINDING_BETA] and kw["fallbacks"] == "default"
        assert "tool_choice" not in kw  # (forced tool use is a 400 on these models)
    assert create["output_config"] == {"effort": "high"}  # high is high (opus-5's xhigh is its own)
    assert stream["output_config"] == {"effort": "low"}


async def test_haiku_5_5_is_never_sent_a_fallback():
    create, stream = await _asked("claude-haiku-5-5", effort="low")
    for kw in (create, stream):
        assert "fallbacks" not in kw and kw["betas"] == [BINDING_BETA] and kw["thinking"] == DROP
    assert create["output_config"] == {"effort": "low"}


async def test_a_model_neo_doesnt_know_gets_adaptive_thinking_and_plain_effort_only():
    create, stream = await _asked("claude-sonnet-4-5")
    assert create["thinking"] == {"type": "adaptive"} and create["output_config"] == {"effort": "high"}
    assert not {"betas", "fallbacks"} & (create.keys() | stream.keys())
    assert "thinking" not in stream
    assert not claude.profile("claude-sonnet-4-5").known and claude.profile("claude-opus-5-5").known


def test_every_model_in_the_table_is_one_the_catalogue_lists():
    from neo2.core.providers import catalog

    listed = {m.id for m in catalog.PROVIDERS["anthropic"].models}
    assert set(claude.PROFILES) == listed
    assert not claude.PROFILES["claude-haiku-5-5"].fallbacks
    assert claude.PROFILES["claude-opus-5"].effort["high"] == "xhigh" and not claude.PROFILES["claude-opus-5"].binding


async def test_the_count_of_dropped_blocks_is_logged_never_what_was_in_them(caplog):
    caplog.set_level(logging.INFO, logger="neo")
    secret = "the-reasoning-itself"
    drops = [{"type": "thinking_dropped", "path": "messages.1.content.0", "reason": "prefix_binding_mismatch", "x": secret}] * 2
    p = ClaudeProvider("claude-opus-5-5", client=Fake(Resp(transformations=drops)))
    await p.complete([Message.user("hi")])
    assert "dropped 2 earlier thinking block(s)" in caplog.text and secret not in caplog.text
    caplog.clear()
    p = ClaudeProvider("claude-opus-5-5", client=Fake(Resp(transformations=[])))
    await p.complete([Message.user("hi")])
    assert "dropped" not in caplog.text


# ---- the reply ------------------------------------------------------------------------------------


async def test_after_a_server_side_fallback_only_the_model_that_answered_is_kept():
    content = [
        Block(type="thinking", thinking="declined model's", signature="s-old"),
        Block(type="tool_use", id="toolu_old", name="open_app", input={"name": "X"}),
        Block(type="text", text="Partial. "),
        Block(type="fallback", from_={"model": "claude-opus-5-5"}, to={"model": "claude-opus-5"}),
        Block(type="thinking", thinking="", signature="s-new"),
        Block(type="text", text="Done."),
        Block(type="tool_use", id="toolu_new", name="open_app", input={"name": "Notes"}),
    ]
    turn = await ClaudeProvider("claude-opus-5-5", client=Fake(Resp(content))).complete([Message.user("hi")])
    assert [b["type"] for b in turn.raw] == ["text", "thinking", "text", "tool_use"]
    assert [c.id for c in turn.tool_calls] == ["toolu_new"] and turn.text == "Partial. Done."
    assert all(b.get("signature") != "s-old" for b in turn.raw)


async def test_a_refusal_is_said_plainly():
    turn = await ClaudeProvider("claude-opus-5-5", client=Fake(Resp([], stop_reason="refusal"))).complete([Message.user("x")])
    assert turn.text == "I can't help with that one." and not turn.tool_calls


def test_another_brains_replay_payload_is_never_sent_as_claudes_blocks():
    openai_raw = {"provider": "openai", "model": "gpt-6.1-sol", "items": [{"type": "reasoning", "id": "rs_1"}]}
    msgs = [
        Message.user("open notes"),
        Message.assistant("Opening.", [ToolCall("call_1", "open_app", {"name": "Notes"})], raw=openai_raw),
        Message.tool([ToolResult("call_1", "open_app", "opened")]),
    ]
    out = claude._to_messages(msgs)
    assert out[1]["content"] == [
        {"type": "text", "text": "Opening."},
        {"type": "tool_use", "id": "call_1", "name": "open_app", "input": {"name": "Notes"}},
    ]


# ---- errors, through the real SDK -------------------------------------------------------------------


def _sdk(status: int, body: dict[str, Any], headers: dict[str, str] | None = None) -> Any:
    anthropic = pytest.importorskip("anthropic")
    lib = sys.modules["anthropic._base_client"].httpx2

    def respond(request):
        assert request.headers.get("x-api-key") == KEY
        return lib.Response(status, json=body, headers=headers or {})

    return anthropic.AsyncAnthropic(
        api_key=KEY, max_retries=0, http_client=lib.AsyncClient(transport=lib.MockTransport(respond))
    )


def _err(kind: str, message: str) -> dict[str, Any]:
    return {"type": "error", "error": {"type": kind, "message": message}}


async def test_a_refused_key_is_key_refused_and_never_repeats_the_key():
    for status in (401, 403):
        p = ClaudeProvider("claude-opus-5-5", client=_sdk(status, _err("authentication_error", f"invalid x-api-key {KEY}")))
        with pytest.raises(KeyRefused) as e:
            await p.complete([Message.user("hi")])
        assert e.value.kind == "key" and KEY not in str(e.value) and "TEST" not in str(e.value)


async def test_a_rate_limit_says_how_long_to_wait():
    p = ClaudeProvider("claude-opus-5-5", client=_sdk(429, _err("rate_limit_error", "slow down"), {"retry-after": "17"}))
    with pytest.raises(RateLimited) as e:
        await p.complete([Message.user("hi")])
    assert e.value.retry_after == 17.0 and e.value.kind == "rate"
    p = ClaudeProvider("claude-opus-5-5", client=_sdk(429, _err("rate_limit_error", "slow down")))
    with pytest.raises(RateLimited) as e:
        await p.complete([Message.user("hi")])
    assert e.value.retry_after is None


async def test_overload_no_credit_and_a_bad_request():
    p = ClaudeProvider("claude-opus-5-5", client=_sdk(529, _err("overloaded_error", "busy"), {"retry-after": "3"}))
    with pytest.raises(Overloaded) as e:
        await p.complete([Message.user("hi")])
    assert e.value.retry_after == 3.0
    p = ClaudeProvider("claude-opus-5-5", client=_sdk(402, _err("billing_error", "no credit")))
    with pytest.raises(QuotaExhausted):
        await p.complete([Message.user("hi")])
    p = ClaudeProvider("claude-opus-5-5", client=_sdk(400, _err("invalid_request_error", "messages: bad")))
    with pytest.raises(BadRequest) as e:
        await p.complete([Message.user("hi")])
    assert e.value.kind == "bad_request"


@pytest.fixture
def app_claude(monkeypatch):
    """NEO.app's own Claude brains (built from settings), with the SDK's transport faked."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(health, "_live", {})

    def make(model: str, status: int, body: dict[str, Any]) -> ClaudeProvider:
        p = ClaudeProvider(model)
        p._client = _sdk(status, body)
        return p

    return make


async def test_a_model_the_api_rejects_is_one_health_problem(app_claude):
    p = app_claude("claude-opus-9-9", 404, _err("not_found_error", "model: claude-opus-9-9"))
    with pytest.raises(BadRequest):
        await p.complete([Message.user("hi")])
    said = [v for k, v in health._live.items() if k.startswith("model:anthropic:")]
    assert said == ["Claude rejected model claude-opus-9-9 — pick a current one in Brains & Keys."]
    # A 400 about something else isn't about the model.
    health._live.clear()
    p = app_claude("claude-opus-5-5", 400, _err("invalid_request_error", "messages.0: roles must alternate"))
    with pytest.raises(BadRequest):
        await p.complete([Message.user("hi")])
    assert not health._live
    # A host's own Claude never touches NEO.app's health list.
    p = ClaudeProvider("claude-opus-9-9", client=_sdk(404, _err("not_found_error", "model: claude-opus-9-9")))
    with pytest.raises(BadRequest):
        await p.complete([Message.user("hi")])
    assert not health._live


async def test_a_refused_key_benches_the_brain_for_an_hour_and_says_so_until_the_key_changes(app_claude, monkeypatch):
    import time

    p = app_claude("claude-opus-5-5", 401, _err("authentication_error", "invalid x-api-key"))
    monkeypatch.setattr(prov, "_cache", {"claude": p})
    monkeypatch.setattr(prov, "_demoted", {})
    with pytest.raises(prov.AllBrainsFailed):
        await prov.Chain(["claude"]).complete([Message.user("hi")])
    assert prov._demoted["claude"] - time.time() > 3000
    assert health._live["key:anthropic"] == "Claude refused its API key. Open Brains & Keys to replace it."
    assert KEY not in json.dumps(health._live)
    prov.forget("anthropic")  # (a key saved, replaced or removed)
    assert "key:anthropic" not in health._live and "claude" not in prov._demoted
