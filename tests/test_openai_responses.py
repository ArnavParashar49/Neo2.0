"""OpenAI's own brain on the Responses API (neo2.core.providers.openai_responses).

Through the real openai SDK into an httpx2 MockTransport (nothing leaves the process; every key is a
fake): what each request carries, checked against the SDK's own request types; the model's own
output items replayed verbatim (its reasoning too), another's reply as its text and calls; pictures
inside a tool's result; tool calls read back; every error mapping; streamed deltas."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

openai_sdk = pytest.importorskip("openai")

from neo2.core import config, health, runtime  # noqa: E402
from neo2.core import providers as prov  # noqa: E402
from neo2.core.providers import openai_responses as oa  # noqa: E402
from neo2.core.providers.base import (  # noqa: E402
    BLIND_PICTURE,
    BadRequest,
    ImagePart,
    KeyRefused,
    Message,
    Overloaded,
    QuotaExhausted,
    RateLimited,
    RequestTooLarge,
    ToolCall,
    ToolResult,
    ToolSpec,
    Unavailable,
)

KEY = "sk-proj-TEST-0000-abcdefghijklmnop"
PNG = b"\x89PNG\r\n\x1a\nfake"
TOOLS = [ToolSpec("open_app", "Open an app.", {"type": "object", "properties": {"name": {"type": "string"}}})]
_spec = importlib.util.spec_from_file_location("vendor_shapes", Path(__file__).parent / "sdk" / "vendor_shapes.py")
vendor_shapes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vendor_shapes)  # type: ignore[union-attr]
lib = sys.modules["openai._base_client"].httpx2


def response(output: list[dict[str, Any]], *, status: str = "completed", usage: tuple[int, int] = (11, 7)) -> dict[str, Any]:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "gpt-6.1-sol",
        "status": status,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output": output,
        "usage": {
            "input_tokens": usage[0],
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": usage[1],
            "output_tokens_details": {"reasoning_tokens": 3},
            "total_tokens": sum(usage),
        },
    }


def text(t: str) -> dict[str, Any]:
    return {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": t, "annotations": []}]}  # fmt: skip


REASONING = {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "plan"}],
             "encrypted_content": "gAAAA-opaque"}  # fmt: skip
CALL = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "open_app",
        "arguments": '{"name": "Notes"}', "status": "completed"}  # fmt: skip


class Server:
    """The Responses API as a MockTransport: replies queued; every request body kept."""

    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []
        self.replies: list[Any] = []

    def reply(self, status: int = 200, body: Any = None, headers: dict[str, str] | None = None) -> None:
        self.replies.append((status, body if body is not None else response([text("ok")]), headers or {}))

    def handle(self, request):
        assert request.headers["authorization"] == f"Bearer {KEY}"
        self.bodies.append(json.loads(request.content or b"{}"))
        status, body, headers = self.replies.pop(0) if self.replies else (200, response([text("ok")]), {})
        if isinstance(body, Exception):
            raise body
        if isinstance(body, bytes):
            return lib.Response(status, content=body, headers={"content-type": "text/event-stream", **headers})
        return lib.Response(status, json=body, headers=headers)

    def client(self) -> Any:
        return openai_sdk.AsyncOpenAI(
            api_key=KEY, max_retries=0, http_client=lib.AsyncClient(transport=lib.MockTransport(self.handle))
        )


@pytest.fixture
def server():
    return Server()


def brain(server: Server, model: str = "gpt-6.1-sol", **kw: Any) -> oa.OpenAIResponsesProvider:
    return oa.OpenAIResponsesProvider(model, client=server.client(), **kw)


def conforms_input(items: list[dict[str, Any]]) -> None:
    from openai.types.responses import ResponseInputItemParam

    for it in items:
        vendor_shapes.assert_conforms(it, ResponseInputItemParam)


# ---- the request ----------------------------------------------------------------------------------


async def test_the_request_carries_what_the_responses_api_takes(server):
    from openai.types.responses import FunctionToolParam

    p = brain(server)
    await p.complete([Message.user("hi")], system="You are NEO.", tools=TOOLS, effort="high")
    b = server.bodies[0]
    assert b["model"] == "gpt-6.1-sol" and b["instructions"] == "You are NEO." and b["store"] is False
    assert b["reasoning"] == {"effort": "high"} and b["include"] == ["reasoning.encrypted_content"]
    assert b["max_output_tokens"] >= 16000 and b["parallel_tool_calls"] is True
    assert "tool_choice" not in b
    for t in b["tools"]:
        vendor_shapes.assert_conforms(t, FunctionToolParam)
        assert t["strict"] is False
    conforms_input(b["input"])
    await p.complete([Message.user("hi")], effort="low")  # no tools: no parallel_tool_calls, no instructions
    assert "tools" not in server.bodies[1] and "parallel_tool_calls" not in server.bodies[1]
    assert "instructions" not in server.bodies[1] and server.bodies[1]["reasoning"] == {"effort": "low"}


async def test_pictures_go_as_input_images_and_inside_a_tools_result(server):
    p = brain(server)
    msgs = [
        Message.user("what's this?", [ImagePart(PNG, "image/png", detail="high")]),
        Message.assistant("Looking.", [ToolCall("call_0", "screenshot", {})]),
        Message.tool([ToolResult("call_0", "screenshot", "the desktop", images=[ImagePart(PNG, "image/png")])]),
    ]
    await p.complete(msgs)
    items = server.bodies[0]["input"]
    conforms_input(items)
    assert items[0]["content"][1]["image_url"].startswith("data:image/png;base64,")
    assert items[0]["content"][1]["detail"] == "high"
    assert items[1] == {"role": "assistant", "content": "Looking."}
    assert items[2] == {"type": "function_call", "call_id": "call_0", "name": "screenshot", "arguments": "{}"}
    out = items[3]
    assert out["type"] == "function_call_output" and out["call_id"] == "call_0"
    assert out["output"][0] == {"type": "input_text", "text": "the desktop"}
    assert out["output"][1]["type"] == "input_image"
    # A model that doesn't see gets the words that say so, no picture.
    blind = brain(server, "gpt-6-luna", vision=False)
    await blind.complete(msgs)
    items = server.bodies[1]["input"]
    assert "no vision" in items[0]["content"][0]["text"] and len(items[0]["content"]) == 1
    assert BLIND_PICTURE in items[3]["output"] and isinstance(items[3]["output"], str)


async def test_its_own_reply_is_replayed_verbatim_reasoning_and_all(server):
    server.reply(body=response([REASONING, CALL]))
    p = brain(server)
    turn = await p.complete([Message.user("open notes")], tools=TOOLS)
    assert [(c.id, c.name, c.args) for c in turn.tool_calls] == [("call_1", "open_app", {"name": "Notes"})]
    assert turn.thinking == "plan" and turn.usage.input_tokens == 11 and turn.usage.output_tokens == 7
    msgs = [
        Message.user("open notes"),
        Message.assistant(turn.text, turn.tool_calls, raw=turn.raw),
        Message.tool([ToolResult("call_1", "open_app", "opened")]),
    ]
    await p.complete(msgs, tools=TOOLS)
    items = server.bodies[1]["input"]
    conforms_input(items)
    assert items[1]["type"] == "reasoning" and items[1]["encrypted_content"] == "gAAAA-opaque" and items[1]["id"] == "rs_1"
    assert items[2]["type"] == "function_call" and items[2]["call_id"] == "call_1" and items[2]["id"] == "fc_1"
    assert items[3] == {"type": "function_call_output", "call_id": "call_1", "output": "opened"}


async def test_another_models_reply_goes_as_its_text_and_calls(server):
    p = brain(server, "gpt-6-luna")
    sol = {"provider": "openai_responses", "model": "gpt-6.1-sol", "items": [REASONING, CALL]}
    claude_raw = [{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "text", "text": "Hi."}]
    for raw in (sol, claude_raw):
        msgs = [
            Message.user("open notes"),
            Message.assistant("On it.", [ToolCall("call_1", "open_app", {"name": "Notes"})], raw=raw),
            Message.tool([ToolResult("call_1", "open_app", "opened")]),
        ]
        await p.complete(msgs)
        items = server.bodies[-1]["input"]
        assert not any(i.get("type") in ("reasoning", "thinking") for i in items)
        assert items[1:3] == [
            {"role": "assistant", "content": "On it."},
            {"type": "function_call", "call_id": "call_1", "name": "open_app", "arguments": '{"name": "Notes"}'},
        ]


async def test_a_call_whose_arguments_arent_json_is_kept_raw(server):
    server.reply(body=response([{**CALL, "arguments": "{not json"}]))
    turn = await brain(server).complete([Message.user("x")], tools=TOOLS)
    assert turn.tool_calls[0].args == {"_raw": "{not json"}


async def test_a_reply_with_no_room_left_for_the_answer_is_truncated(server):
    from neo2.core.providers.openai_compat import Truncated

    server.reply(body=response([REASONING], status="incomplete"))
    with pytest.raises(Truncated):
        await brain(server).complete([Message.user("x")])


async def test_a_model_that_refuses_the_reasoning_parameter_is_asked_again_without_it(server):
    server.reply(400, {"error": {"message": "Unsupported parameter: 'reasoning.effort' is not supported with this model.",
                                 "type": "invalid_request_error", "param": "reasoning.effort", "code": None}})  # fmt: skip
    p = brain(server, "gpt-6-luna")
    turn = await p.complete([Message.user("hi")])
    assert turn.text == "ok" and len(server.bodies) == 2
    assert "reasoning" in server.bodies[0] and "reasoning" not in server.bodies[1] and "include" not in server.bodies[1]
    await p.complete([Message.user("hi")])
    assert "reasoning" not in server.bodies[2]  # remembered


# ---- errors -----------------------------------------------------------------------------------------


def _err(message: str, *, type_: str = "invalid_request_error", code: str | None = None, param: str | None = None) -> dict:
    return {"error": {"message": message, "type": type_, "code": code, "param": param}}


async def test_every_error_becomes_what_the_chain_acts_on(server):
    p = brain(server)
    cases = [
        (401, _err(f"Incorrect API key provided: {KEY[:8]}****{KEY[-4:]}", code="invalid_api_key"), {}, KeyRefused),
        (403, _err("Your API key does not have the right permissions", code="invalid_api_key"), {}, KeyRefused),
        (403, _err("Project does not have access to model gpt-6-astra", code="model_not_found"), {}, BadRequest),
        (429, _err("You exceeded your current quota", type_="insufficient_quota", code="insufficient_quota"), {}, QuotaExhausted),
        (429, _err("Rate limit reached", type_="requests", code="rate_limit_exceeded"), {"retry-after": "12"}, RateLimited),
        (400, _err("This model's maximum context length is 400000 tokens", code="context_length_exceeded"), {}, RequestTooLarge),
        (400, _err("Invalid value for 'input'"), {}, BadRequest),
        (500, _err("server error", type_="server_error"), {}, Overloaded),
    ]
    for status, body, headers, kind in cases:
        server.reply(status, body, headers)
        with pytest.raises(kind) as e:
            await p.complete([Message.user("hi")])
        assert type(e.value) is kind, (status, body)
        assert KEY[-4:] not in str(e.value) and "TEST" not in str(e.value)
    assert p.cooling_until() > 0  # the 429 that said how long
    server.reply(body=lib.ConnectError("refused"))
    with pytest.raises(Unavailable):
        await p.complete([Message.user("hi")])


async def test_out_of_quota_rests_for_an_hour(server):
    import time

    server.reply(429, _err("You exceeded your current quota", type_="insufficient_quota", code="insufficient_quota"))
    with pytest.raises(QuotaExhausted) as e:
        await brain(server).complete([Message.user("hi")])
    assert 3500 < e.value.until - time.time() <= 3600
    server.reply(429, _err("slow down", code="rate_limit_exceeded"), {"retry-after": "7"})
    with pytest.raises(RateLimited) as r:
        await brain(server).complete([Message.user("hi")])
    assert r.value.retry_after == 7.0 and not isinstance(r.value, QuotaExhausted)


# ---- streaming ---------------------------------------------------------------------------------------


def _sse(*events: dict[str, Any]) -> bytes:
    return b"".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events)


async def test_a_stream_yields_the_text_as_it_comes(server):
    delta = {"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0, "content_index": 0,
             "logprobs": [], "sequence_number": 1}  # fmt: skip
    server.reply(body=_sse({**delta, "delta": "Hel"}, {**delta, "delta": "lo", "sequence_number": 2},
                           {"type": "response.completed", "sequence_number": 3, "response": response([text("Hello")])}))  # fmt: skip
    p = brain(server)
    got = [t async for t in p.stream([Message.user("tell me a joke")], system="Be brief.")]
    assert "".join(got) == "Hello"
    b = server.bodies[0]
    assert b["stream"] is True and b["reasoning"] == {"effort": "low"} and "tools" not in b and b["instructions"] == "Be brief."


# ---- how NEO builds it --------------------------------------------------------------------------------


def _env(monkeypatch, **env: str) -> None:
    for k in ("OPENAI_API_KEY", "NEO_OPENAI_MODEL", "NEO_DAILY_BUDGETS"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(prov, "_cache", {})


def test_neo_builds_it_only_with_its_key_and_names_it_by_the_reference(monkeypatch):
    _env(monkeypatch)
    with pytest.raises(Unavailable, match="OPENAI_API_KEY is not set"):
        prov._build("openai@gpt-6-luna")
    _env(monkeypatch, OPENAI_API_KEY=KEY, NEO_DAILY_BUDGETS="openai=2")
    p = prov._build("openai@gpt-6-luna")
    assert isinstance(p, oa.OpenAIResponsesProvider)
    assert (p.name, p.model, p.supports_vision, p._app) == ("openai@gpt-6-luna", "gpt-6-luna", True, True)
    assert p._cap is not None and p._cap.ledger.daily_budget == 2.0
    assert prov._build("openai").model == "gpt-6.1-sol"  # settings.openai_model
    assert prov.limit_of(p) == oa.IMAGE_LIMIT


def test_the_sdk_factory_reads_no_settings():
    from neo2 import providers

    runtime.set_library_mode()
    try:
        o = providers.OpenAI("gpt-6-luna", api_key=KEY)
        assert (o.name, o.model, o.supports_vision, o._cap, o._app) == ("openai", "gpt-6-luna", True, None, False)
        assert providers.OpenAI("o-something", client=openai_sdk.AsyncOpenAI(api_key=KEY)).supports_vision is False
    finally:
        runtime.set_library_mode(False)


async def test_a_model_openai_rejects_is_one_health_problem(server, monkeypatch):
    _env(monkeypatch, OPENAI_API_KEY=KEY)
    monkeypatch.setattr(health, "_live", {})
    p = oa.openai("gpt-9")
    p._client = server.client()
    server.reply(404, _err("The model `gpt-9` does not exist or you do not have access to it.", code="model_not_found"))
    with pytest.raises(BadRequest):
        await p.complete([Message.user("hi")])
    assert health._live == {"model:openai:gpt-9": "OpenAI rejected model gpt-9 — pick a current one in Brains & Keys."}
    await p.complete([Message.user("hi")])  # it works again
    assert health._live == {}
