"""The chat-completions brains (Groq, OpenRouter, the endpoints a user adds) against strict servers.

* Every tool reply of one assistant turn comes right after it; the pictures they returned follow in
  one user message (strict servers refuse a user message between tool replies).
* `token_param`: max_tokens or max_completion_tokens; `short_tool_ids`: 9 letters and digits (Mistral),
  the same id for a call and its reply.
* 401, or a 403 about the key → KeyRefused (never the server's words); 402 → QuotaExhausted for an hour.
* An endpoint on this Mac is checked before each call at its own address; a base URL NEO may not use
  is refused before anything is sent.

Through the real openai SDK into an httpx2 MockTransport: nothing leaves the process; keys are fakes."""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import pytest

openai_sdk = pytest.importorskip("openai")

from neo2.core import brainref, config  # noqa: E402
from neo2.core import providers as prov  # noqa: E402
from neo2.core.providers import local_server  # noqa: E402
from neo2.core.providers import openai_compat as oc  # noqa: E402
from neo2.core.providers.base import (  # noqa: E402
    BadRequest,
    ImagePart,
    KeyRefused,
    Message,
    QuotaExhausted,
    ToolCall,
    ToolResult,
    Unavailable,
)

KEY = "gsk_TEST-0000-abcdefghijklmnop"
PNG = b"\x89PNG\r\n\x1a\nfake"
lib = sys.modules["openai._base_client"].httpx2


def completion(text: str = "ok", usage: tuple[int, int] = (20, 5)) -> dict[str, Any]:
    return {
        "id": "c1",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": sum(usage)},
    }


class Server:
    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []
        self.replies: list[tuple[int, Any]] = []

    def handle(self, request):
        self.bodies.append(json.loads(request.content or b"{}"))
        status, body = self.replies.pop(0) if self.replies else (200, completion())
        return lib.Response(status, json=body)

    def client(self) -> Any:
        return openai_sdk.AsyncOpenAI(
            api_key=KEY, base_url="https://api.example.test/v1", max_retries=0,
            http_client=lib.AsyncClient(transport=lib.MockTransport(self.handle)),
        )  # fmt: skip


@pytest.fixture
def server():
    return Server()


def _two_pictures() -> list[Message]:
    return [
        Message.user("compare them"),
        Message.assistant("", [ToolCall("toolu_01abcdefghijklmnop", "screenshot", {}), ToolCall("call_2", "zoom", {"x": 1})]),
        Message.tool([
            ToolResult("toolu_01abcdefghijklmnop", "screenshot", "the desktop", images=[ImagePart(PNG)]),
            ToolResult("call_2", "zoom", "zoomed", images=[ImagePart(PNG), ImagePart(PNG)]),
        ]),
    ]  # fmt: skip


def test_every_tool_reply_follows_its_call_then_one_message_of_pictures():
    out = oc._to_messages(_two_pictures(), "sys", vision=True)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool", "tool", "user"]
    assert [m["tool_call_id"] for m in out[3:5]] == ["toolu_01abcdefghijklmnop", "call_2"]
    pictures = out[5]["content"]
    assert [p["type"] for p in pictures] == ["text", "image_url", "text", "image_url", "image_url"]
    assert pictures[0]["text"] == "Image from screenshot:" and pictures[2]["text"] == "Image from zoom:"
    # A brain that can't see: no pictures message at all.
    assert [m["role"] for m in oc._to_messages(_two_pictures(), "sys", vision=False)] == ["system", "user", "assistant", "tool", "tool"]


def test_short_tool_ids_are_nine_letters_and_digits_the_same_for_call_and_reply():
    out = oc._to_messages(_two_pictures(), "", vision=False, short_ids=True)
    calls = [c["id"] for c in out[1]["tool_calls"]]
    replies = [m["tool_call_id"] for m in out[2:4]]
    assert calls == replies and all(len(i) == 9 and i.isalnum() for i in calls)
    assert calls[0] != calls[1]
    assert oc.short_id("abcDEF123") == "abcDEF123"  # (already one: kept)
    assert oc.short_id("call_2") == oc.short_id("call_2")


async def test_the_answers_room_goes_by_the_name_the_server_takes(server):
    p = oc.OpenAICompatProvider(name="x", model="m", client=server.client(), token_param="max_completion_tokens")
    await p.complete([Message.user("hi")], max_tokens=1234)
    assert server.bodies[0]["max_completion_tokens"] == 1234 and "max_tokens" not in server.bodies[0]
    q = oc.OpenAICompatProvider(name="y", model="m", client=server.client())
    await q.complete([Message.user("hi")], max_tokens=1234)
    assert server.bodies[1]["max_tokens"] == 1234 and "max_completion_tokens" not in server.bodies[1]


def _err(message: str, code: str | None = None) -> dict[str, Any]:
    return {"error": {"message": message, "type": "invalid_request_error", "code": code}}


async def test_a_refused_key_and_no_credit(server):
    p = oc.OpenAICompatProvider(name="groq", model="m", client=server.client())
    for status, body, kind in (
        (401, _err(f"Invalid API Key {KEY}", "invalid_api_key"), KeyRefused),
        (403, _err("Your API key was revoked"), KeyRefused),
        (403, _err("This model is not available in your region"), BadRequest),
        (402, _err("Insufficient credits. Add more using https://openrouter.ai/credits"), QuotaExhausted),
    ):
        server.replies.append((status, body))
        with pytest.raises(kind) as e:
            await p.complete([Message.user("hi")])
        assert type(e.value) is kind and KEY not in str(e.value) and "TEST" not in str(e.value)
        if kind is QuotaExhausted:
            assert 3500 < e.value.until - time.time() <= 3600


# ---- endpoints the user added ------------------------------------------------------------------------

OLLAMA = {"id": "custom:ollama", "label": "Ollama", "template": "ollama", "base_url": "http://localhost:11434/v1", "model": "qwen3:14b"}
MINE = {"id": "custom:mine", "label": "My server", "template": "other", "base_url": "http://[::1]:8000/api/v1", "model": "m"}
MISTRAL = {"id": "custom:mistral", "label": "Mistral", "template": "mistral", "base_url": "https://api.mistral.ai/v1",
           "model": "mistral-large-latest", "short_tool_ids": True, "token_param": "max_completion_tokens"}  # fmt: skip


def _env(monkeypatch, **env: str) -> None:
    for k in ("OPENROUTER_API_KEY", "NEO_CUSTOM_PROVIDERS", "NEO_DAILY_BUDGETS", "MISTRAL_API_KEY", "NEO_OPENROUTER_MODEL"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(prov, "_cache", {})


async def test_an_endpoint_on_this_mac_is_checked_at_its_own_address(monkeypatch):
    asked: list[Any] = []

    async def is_up(port=8080):
        asked.append(("is_up", port))
        return False

    async def reachable(url, timeout=1.5):
        asked.append(("reachable", url))
        return False

    monkeypatch.setattr(local_server, "is_up", is_up)
    monkeypatch.setattr(local_server, "reachable", reachable)
    ollama = oc.custom(brainref.CustomProvider.model_validate(OLLAMA))
    mine = oc.custom(brainref.CustomProvider.model_validate(MINE))
    assert ollama._is_local and mine._is_local and mine.on_device
    with pytest.raises(Unavailable) as e:
        await ollama.complete([Message.user("hi")])
    assert "Ollama isn't answering on this Mac (port 11434)" in str(e.value) and "neo2 --local" not in str(e.value)
    with pytest.raises(Unavailable, match="My server isn't answering"):
        await mine.complete([Message.user("hi")])
    # (an endpoint the user added is asked at its own address, where any answer means it's up — one that
    # wants a key answers 401 without it; NEO's own server's /v1/models must answer 200)
    assert asked == [("reachable", "http://localhost:11434/v1/models"), ("reachable", "http://[::1]:8000/api/v1/models")]
    with pytest.raises(Unavailable, match="neo2 --local"):
        await oc.local().complete([Message.user("hi")])  # NEO's own local server: as before


def test_a_base_url_neo_may_not_use_is_refused_before_anything_is_sent():
    bad = brainref.CustomProvider.model_construct(**{**MISTRAL, "base_url": "http://api.mistral.ai/v1"})  # (unchecked)
    with pytest.raises(Unavailable, match="base URL isn't one NEO may use"):
        oc.custom(bad, "key")
    with pytest.raises(Unavailable, match="no base URL"):
        oc.OpenAICompatProvider(name="x", model="m")


async def test_an_endpoints_own_settings_reach_its_requests(server, monkeypatch):
    cp = brainref.CustomProvider.model_validate(MISTRAL)
    p = oc.custom(cp, "mistral-TEST-key")
    p._client = server.client()
    await p.complete(_two_pictures())
    b = server.bodies[0]
    assert "max_completion_tokens" in b and "max_tokens" not in b
    ids = [c["id"] for c in b["messages"][1]["tool_calls"]]
    assert all(len(i) == 9 and i.isalnum() for i in ids)
    assert p.token_budget == 32768 and not p.on_device


async def test_a_capped_compatible_brain_counts_what_it_spends(server, monkeypatch):
    _env(monkeypatch, OPENROUTER_API_KEY="sk-or-TEST-0000", NEO_DAILY_BUDGETS='{"openrouter": 0.0001}')
    p = prov._build("openrouter@deepseek/deepseek-v4-pro")
    assert p._cap is not None and p._cap.ledger.daily_budget == 0.0001
    p._client = server.client()
    server.replies.append((200, completion(usage=(10, 10))))  # 10 × $10 + 10 × $50 per M (no price known: the dearest)
    await p.complete([Message.user("hi")])
    with pytest.raises(QuotaExhausted):
        await p.complete([Message.user("hi")])
    assert len(server.bodies) == 1
    _env(monkeypatch, OPENROUTER_API_KEY="sk-or-TEST-0000")
    assert prov._build("openrouter@deepseek/deepseek-v4-pro")._cap is None  # no entry: no NEO cap
