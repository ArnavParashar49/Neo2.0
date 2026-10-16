"""The models a key may use (neo2.core.providers.keycheck.models_list): what each can do, the
default per role (dropped when the live list doesn't have it), what is left out, and the 10-minute
memory. Every provider is faked (httpx.MockTransport, or the SDKs into an httpx2 MockTransport);
every key is a fake."""

from __future__ import annotations

import json
import sys
from typing import Any

import httpx
import pytest

from neo2.core import config
from neo2.core.providers import keycheck as kc

KEY = "sk-proj-TEST-0000-abcdefghijkl"


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    from neo2.core.providers import catalog

    monkeypatch.setattr(kc, "_cache", {})
    monkeypatch.setattr(catalog, "_LEARNED", {})
    for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "NEO_CUSTOM_PROVIDERS", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(config, "_settings", None)


class Api:
    def __init__(self, module: str | None) -> None:
        self.lib = sys.modules[f"{module}._base_client"].httpx2 if module else httpx
        self.n = 0
        self.answer: Any = (200, {"object": "list", "data": []})

    def respond(self, request):
        self.n += 1
        status, body = self.answer
        return self.lib.Response(status, json=body)


@pytest.fixture
def openai_api(monkeypatch):
    openai = pytest.importorskip("openai")
    api = Api("openai")
    monkeypatch.setattr(
        kc, "_openai_client",
        lambda base, key: openai.AsyncOpenAI(base_url=base, api_key=key or "none", max_retries=0,
                                             http_client=api.lib.AsyncClient(transport=api.lib.MockTransport(api.respond))),
    )  # fmt: skip
    return api


@pytest.fixture
def anthropic_api(monkeypatch):
    anthropic = pytest.importorskip("anthropic")
    api = Api("anthropic")
    monkeypatch.setattr(
        kc, "_anthropic_client",
        lambda key: anthropic.AsyncAnthropic(api_key=key, max_retries=0,
                                             http_client=api.lib.AsyncClient(transport=api.lib.MockTransport(api.respond))),
    )  # fmt: skip
    return api


@pytest.fixture
def http(monkeypatch):
    api = Api(None)
    monkeypatch.setattr(kc, "_transport", httpx.MockTransport(api.respond))
    return api


def _ids(*ids: str, **extra: Any) -> dict[str, Any]:
    return {"object": "list", "data": [{"id": i, "object": "model", "created": 0, "owned_by": "x", **extra} for i in ids]}


async def test_openai_known_models_say_what_they_do_and_the_rest_are_left_out(openai_api):
    hidden = ["text-embedding-3-large", "whisper-1", "tts-1", "gpt-image-1", "omni-moderation-latest", "gpt-realtime-2.1",
              "gpt-4o-audio-preview", "gpt-4o-mini-transcribe", "dall-e-3"]  # fmt: skip
    openai_api.answer = (200, _ids("gpt-6-astra", "gpt-x-new", "gpt-6-luna", "gpt-6.1-sol", *hidden))
    r = await kc.models_list("openai", KEY)
    assert r["ok"] and r["error"] is None
    by = {m["id"]: m for m in r["models"]}
    assert set(by) == {"gpt-6-astra", "gpt-x-new", "gpt-6-luna", "gpt-6.1-sol"}
    assert [m["id"] for m in r["models"]][:3] == ["gpt-6.1-sol", "gpt-6-luna", "gpt-6-astra"]  # recommended, cheap, premium
    assert by["gpt-6.1-sol"] == {"id": "gpt-6.1-sol", "label": "GPT-6.1 Sol", "sees": True, "tools": True,
                                 "context": 400_000, "tier": "recommended", "price": [2.0, 10.0]}  # fmt: skip
    assert by["gpt-x-new"]["sees"] and by["gpt-x-new"]["label"] == "gpt-x-new" and "tier" not in by["gpt-x-new"]
    assert r["defaults"] == {"brain": "openai@gpt-6.1-sol", "quick": "openai@gpt-6-luna", "eyes": "openai@gpt-6.1-sol"}


async def test_a_default_the_key_cant_use_is_dropped(openai_api):
    openai_api.answer = (200, _ids("gpt-6-luna"))
    r = await kc.models_list("openai", KEY)
    assert r["defaults"] == {"quick": "openai@gpt-6-luna"}


async def test_anthropic_reads_its_own_capabilities(anthropic_api):
    def model(mid: str, images: bool) -> dict[str, Any]:
        return {"id": mid, "display_name": mid.upper(), "created_at": "2026-09-01T00:00:00Z", "type": "model",
                "max_input_tokens": 1_000_000, "max_tokens": 128_000,
                "capabilities": {"image_input": {"supported": images}}}  # fmt: skip

    anthropic_api.answer = (200, {"data": [model("claude-opus-5-5", True), model("claude-haiku-5-5", True),
                                           model("claude-old-1", False)], "has_more": False})  # fmt: skip
    r = await kc.models_list("anthropic", "sk-ant-TEST-0000-abcdefgh")
    by = {m["id"]: m for m in r["models"]}
    assert by["claude-opus-5-5"]["label"] == "CLAUDE-OPUS-5-5" and by["claude-opus-5-5"]["context"] == 1_000_000
    assert by["claude-opus-5-5"]["tier"] == "recommended" and by["claude-haiku-5-5"]["price"] == [0.1, 0.5]
    assert by["claude-old-1"]["sees"] is False and by["claude-old-1"]["note"] == "may not work with NEO"
    assert "note" not in by["claude-opus-5-5"]
    assert r["defaults"] == {"brain": "claude@claude-opus-5-5", "quick": "claude@claude-haiku-5-5", "eyes": "claude@claude-opus-5-5"}


async def test_openrouter_reads_tools_and_pictures_from_its_list(http):
    http.answer = (200, {"data": [
        {"id": "deepseek/deepseek-v4-pro", "name": "DeepSeek V4 Pro", "context_length": 131072,
         "architecture": {"input_modalities": ["text"]}, "supported_parameters": ["tools", "max_tokens"],
         "pricing": {"prompt": "0.0000005", "completion": "0.0000015"}},
        {"id": "anthropic/claude-opus-5.5", "name": "Claude Opus 5.5", "context_length": 1000000,
         "architecture": {"input_modalities": ["text", "image"]}, "supported_parameters": ["tools"]},
        {"id": "some/chat-only", "name": "Chat only", "architecture": {"input_modalities": ["text"]}, "supported_parameters": []},
    ]})  # fmt: skip
    r = await kc.models_list("openrouter", "sk-or-TEST-0000-abcdefgh")
    by = {m["id"]: m for m in r["models"]}
    assert (by["deepseek/deepseek-v4-pro"]["tools"], by["deepseek/deepseek-v4-pro"]["sees"]) == (True, False)
    assert by["deepseek/deepseek-v4-pro"]["price"] == [0.5, 1.5] and by["deepseek/deepseek-v4-pro"]["context"] == 131072
    assert by["anthropic/claude-opus-5.5"]["sees"] and not by["some/chat-only"]["tools"]
    assert r["defaults"] == {}  # (the user picks one)
    # What the list said is what the chains read from then on.
    from neo2.core.providers import catalog

    assert catalog.describe("openrouter@some/chat-only").tools is False
    assert catalog.describe("openrouter@deepseek/deepseek-v4-pro").sees is False
    assert catalog.describe("openrouter@anthropic/claude-opus-5.5").sees is True
    assert catalog.describe("openrouter@never/listed").tools is True  # (unknown: as before)


async def test_gemini_keeps_only_the_models_that_answer(http, monkeypatch):
    http.answer = (200, {"models": [
        {"name": "models/gemini-3.8-flash", "displayName": "Gemini 3.8 Flash", "inputTokenLimit": 1048576,
         "supportedGenerationMethods": ["generateContent", "countTokens"]},
        {"name": "models/gemini-3.5-flash-lite", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-embedding-001", "supportedGenerationMethods": ["embedContent"]},
        {"name": "models/gemini-3.8-live", "supportedGenerationMethods": ["bidiGenerateContent"]},
    ]})  # fmt: skip
    r = await kc.models_list("gemini", "AIzaTEST-0000-abcdefgh")
    assert [m["id"] for m in r["models"]] == ["gemini-3.8-flash", "gemini-3.5-flash-lite"]
    assert r["models"][0]["context"] == 1048576 and r["models"][0]["sees"]
    assert r["defaults"] == {"brain": "gemini", "quick": "gemini_lite", "eyes": "gemini"}


async def test_an_endpoint_s_list_or_its_own_switches(openai_api, monkeypatch):
    mistral = {"id": "custom:mistral", "label": "Mistral", "template": "mistral", "base_url": "https://api.mistral.ai/v1",
               "model": "mistral-large-latest"}  # fmt: skip
    openai_api.answer = (200, _ids("mistral-large-latest", "mistral-embed", max_context_length=131072,
                                   capabilities={"completion_chat": True, "function_calling": True, "vision": True}))  # fmt: skip
    r = await kc.models_list("custom", "mistral-TEST-key-0000", endpoint=mistral)
    assert [m["id"] for m in r["models"]] == ["mistral-large-latest"]
    assert r["models"][0] == {"id": "mistral-large-latest", "label": "mistral-large-latest", "sees": True, "tools": True,
                                   "context": 131072}  # fmt: skip
    assert r["defaults"] == {"brain": "custom:mistral@mistral-large-latest", "quick": "custom:mistral@mistral-large-latest"}
    together = {"id": "custom:together", "label": "Together", "template": "together", "base_url": "https://api.together.xyz/v1",
                "model": "meta/llama-5", "tools": False, "vision": True, "context": 65536}  # fmt: skip
    openai_api.answer = (200, _ids("meta/llama-5"))
    r = await kc.models_list("custom", "tg-TEST-key-0000", endpoint=together)
    assert r["models"][0] == {"id": "meta/llama-5", "label": "meta/llama-5", "sees": True, "tools": False, "context": 65536}
    assert "brain" not in r["defaults"] and r["defaults"]["eyes"] == "custom:together@meta/llama-5"


async def test_deepseek_says_its_context(openai_api):
    ds = {"id": "custom:deepseek", "label": "DeepSeek", "template": "deepseek", "base_url": "https://api.deepseek.com",
          "model": "deepseek-v4-pro"}  # fmt: skip
    openai_api.answer = (200, _ids("deepseek-v4-pro", "deepseek-flash", context_window=131072))
    r = await kc.models_list("custom", "sk-TEST-ds-0000", endpoint=ds)
    assert {m["context"] for m in r["models"]} == {131072}
    assert r["defaults"] == {"brain": "custom:deepseek@deepseek-v4-pro", "quick": "custom:deepseek@deepseek-flash"}


async def test_a_list_is_kept_for_ten_minutes_per_key(openai_api, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(kc, "_now", lambda: clock[0])
    openai_api.answer = (200, _ids("gpt-6-luna"))
    first = await kc.models_list("openai", KEY)
    assert openai_api.n == 1 and (await kc.models_list("openai", KEY)) is first and openai_api.n == 1
    assert KEY not in json.dumps([list(k) for k in kc._cache])  # (a hash of the key, never the key)
    await kc.models_list("openai", "sk-proj-TEST-other-key-0000")
    assert openai_api.n == 2  # another key: its own list
    await kc.models_list("openai", KEY, refresh=True)
    assert openai_api.n == 3
    clock[0] += 599
    await kc.models_list("openai", KEY)
    assert openai_api.n == 3
    clock[0] += 2
    await kc.models_list("openai", KEY)
    assert openai_api.n == 4
    kc.forget("openai")
    assert not kc._cache


async def test_when_the_list_cant_be_had_the_catalogues_stands_in(openai_api):
    openai_api.answer = (401, {"error": {"message": f"Incorrect API key provided: {KEY}"}})
    r = await kc.models_list("openai", KEY)
    assert (r["ok"], r["error"]) == (False, "bad_key") and KEY not in json.dumps(r)
    assert [m["id"] for m in r["models"]] == ["gpt-6.1-sol", "gpt-6-luna", "gpt-6-astra"]
    assert r["defaults"]["brain"] == "openai@gpt-6.1-sol" and not kc._cache  # (a failure isn't kept)


async def test_a_long_list_is_trimmed(openai_api):
    openai_api.answer = (200, _ids(*(f"m-{i}" for i in range(kc.MAX_MODELS + 50))))
    r = await kc.models_list("custom", "k-TEST-0000", endpoint={"id": "custom:big", "label": "Big", "base_url": "https://big.test/v1",
                                                              "model": "m-1"})  # fmt: skip
    assert len(r["models"]) == kc.MAX_MODELS
