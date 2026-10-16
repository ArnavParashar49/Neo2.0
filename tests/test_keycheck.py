"""Testing a key (neo2.core.providers.keycheck.check) and the key-shape hints (neo2.core.keyshape).

Every provider's check, through httpx.MockTransport (Gemini, OpenRouter, DeepSeek) or the real
anthropic / openai SDK into an httpx2 MockTransport: 200/401/402/403/404/429, no connection and a
timeout each come back as a fixed status and fixed words with the host's name — and the key never
appears in the result, a log line or stdout. Every key is a fake; nothing leaves the process."""

from __future__ import annotations

import json
import sys
from typing import Any

import httpx
import pytest

from neo2.core import config, keyshape
from neo2.core.providers import keycheck as kc

KEYS = {
    "gemini": "AIzaTEST-gemini-0000-abcdefgh",
    "anthropic": "sk-ant-TEST-0000-abcdefghijkl",
    "openai": "sk-proj-TEST-0000-abcdefghijkl",
    "groq": "gsk_TEST-groq-0000-abcdefghij",
    "cerebras": "csk-TEST-cerebras-0000-abcdef",
    "openrouter": "sk-or-TEST-0000-abcdefghijklm",
}
DEEPSEEK = {"id": "custom:deepseek", "label": "DeepSeek", "template": "deepseek", "base_url": "https://api.deepseek.com",
            "model": "deepseek-v4-pro"}  # fmt: skip
OLLAMA = {"id": "custom:ollama", "label": "Ollama", "template": "ollama", "base_url": "http://localhost:11434/v1",
          "model": "qwen3:14b"}  # fmt: skip
DS_KEY = "sk-TEST-deepseek-0000-abcdefgh"


class Http:
    """Plain HTTPS (Gemini, OpenRouter, DeepSeek): answers queued per path; every request kept."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.answers: dict[str, Any] = {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        a = self.answers.get(request.url.path, (200, {}))
        if isinstance(a, Exception):
            raise a
        status, body = a
        return httpx.Response(status, json=body)


@pytest.fixture
def http(monkeypatch):
    h = Http()
    monkeypatch.setattr(kc, "_transport", httpx.MockTransport(h.handler))
    return h


class Sdk:
    """The /models of an SDK's server: what it answers next, every request kept."""

    def __init__(self, module: str) -> None:
        self.lib = sys.modules[f"{module}._base_client"].httpx2
        self.requests: list[Any] = []
        self.answer: Any = (200, {"data": [], "has_more": False, "object": "list"})

    def http(self) -> Any:
        def respond(request):
            self.requests.append(request)
            if isinstance(self.answer, Exception):
                raise self.answer
            status, body = self.answer
            return self.lib.Response(status, json=body)

        return self.lib.AsyncClient(transport=self.lib.MockTransport(respond))


@pytest.fixture
def anthropic_api(monkeypatch):
    anthropic = pytest.importorskip("anthropic")
    s = Sdk("anthropic")
    monkeypatch.setattr(kc, "_anthropic_client", lambda key: anthropic.AsyncAnthropic(api_key=key, max_retries=0, http_client=s.http()))
    return s


@pytest.fixture
def openai_api(monkeypatch):
    openai = pytest.importorskip("openai")
    s = Sdk("openai")
    s.bases: list[str] = []

    def client(base, key):
        s.bases.append(base)
        return openai.AsyncOpenAI(base_url=base, api_key=key or "none", max_retries=0, http_client=s.http())

    monkeypatch.setattr(kc, "_openai_client", client)
    return s


def _env(monkeypatch, **env: str) -> None:
    for k in ("GEMINI_API_KEY", "GEMINI_API_KEYS", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "CEREBRAS_API_KEY",
              "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "NEO_CUSTOM_PROVIDERS"):  # fmt: skip
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(config, "_settings", None)


def _secret_free(result: kc.KeyCheck, *keys: str) -> None:
    text = json.dumps(result.as_dict())
    for k in keys:
        assert k not in text and k[-6:] not in text
    assert result.status in kc.STATUSES


ERRORS = [(401, "bad_key", False), (402, "no_credit", False), (403, "no_access", False), (404, "not_found", False),
          (429, "rate_limited", True), (500, "error", False)]  # fmt: skip


# ---- Gemini ---------------------------------------------------------------------------------------------


async def test_gemini_is_checked_with_its_header_never_in_the_url(http):
    key = KEYS["gemini"]
    http.answers["/v1beta/models"] = (200, {"models": [
        {"name": "models/gemini-3.8-flash", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-embedding-001", "supportedGenerationMethods": ["embedContent"]},
    ]})  # fmt: skip
    r = await kc.check("gemini", key)
    assert (r.ok, r.status, r.http, r.models) == (True, "ok", 200, 1)
    assert r.message == "The key works (generativelanguage.googleapis.com)."
    req = http.requests[0]
    assert req.headers["x-goog-api-key"] == key and key not in str(req.url) and "pageSize=1000" in str(req.url)
    _secret_free(r, key)


@pytest.mark.parametrize("code,status,ok", ERRORS)
async def test_gemini_errors(http, code, status, ok):
    http.answers["/v1beta/models"] = (code, {"error": {"message": f"echo {KEYS['gemini']}"}})
    r = await kc.check("gemini", KEYS["gemini"])
    assert (r.ok, r.status, r.http) == (ok, status, code)
    _secret_free(r, KEYS["gemini"])


async def test_googles_400_for_a_bad_key_is_a_bad_key(http):
    http.answers["/v1beta/models"] = (400, {"error": {"code": 400, "message": "API key not valid. Please pass a valid API key.",
                                                      "details": [{"reason": "API_KEY_INVALID"}]}})  # fmt: skip
    r = await kc.check("gemini", KEYS["gemini"])
    assert (r.ok, r.status) == (False, "bad_key")


async def test_no_connection_and_a_timeout(http):
    http.answers["/v1beta/models"] = httpx.ConnectError("nodename nor servname provided")
    r = await kc.check("gemini", KEYS["gemini"])
    assert (r.status, r.http, r.message) == ("unreachable", None, "Couldn't reach generativelanguage.googleapis.com.")
    http.answers["/v1beta/models"] = httpx.ReadTimeout("slow")
    r = await kc.check("gemini", KEYS["gemini"])
    assert r.status == "timeout" and "10 seconds" in r.message


# ---- Anthropic ----------------------------------------------------------------------------------------


def _claude_model(mid: str) -> dict[str, Any]:
    return {"id": mid, "display_name": mid.replace("-", " ").title(), "created_at": "2026-09-01T00:00:00Z", "type": "model",
            "max_input_tokens": 1_000_000, "max_tokens": 128_000}  # fmt: skip


async def test_anthropic_is_checked_through_its_sdk(anthropic_api):
    key = KEYS["anthropic"]
    anthropic_api.answer = (200, {"data": [_claude_model("claude-opus-5-5"), _claude_model("claude-haiku-5-5")],
                                  "has_more": False, "first_id": "a", "last_id": "b"})  # fmt: skip
    r = await kc.check("anthropic", key)
    assert (r.ok, r.status, r.models) == (True, "ok", 2) and r.message == "The key works (api.anthropic.com)."
    req = anthropic_api.requests[0]
    assert req.url.path == "/v1/models" and req.headers["x-api-key"] == key
    _secret_free(r, key)


@pytest.mark.parametrize("code,status,ok", ERRORS)
async def test_anthropic_errors(anthropic_api, code, status, ok):
    anthropic_api.answer = (code, {"type": "error", "error": {"type": "x", "message": f"bad {KEYS['anthropic']}"}})
    r = await kc.check("anthropic", KEYS["anthropic"])
    assert (r.ok, r.status, r.http) == (ok, status, code)
    _secret_free(r, KEYS["anthropic"])


async def test_anthropic_unreachable_and_timeout(anthropic_api):
    lib = anthropic_api.lib
    anthropic_api.answer = lib.ConnectError("refused")
    assert (await kc.check("anthropic", KEYS["anthropic"])).status == "unreachable"
    anthropic_api.answer = lib.ReadTimeout("slow")
    assert (await kc.check("anthropic", KEYS["anthropic"])).status == "timeout"


# ---- OpenAI and the compatible ones -----------------------------------------------------------------------


@pytest.mark.parametrize("pid,host", [("openai", "api.openai.com"), ("groq", "api.groq.com"), ("cerebras", "api.cerebras.ai")])
async def test_openai_shaped_providers_are_checked_with_the_openai_sdk(openai_api, pid, host):
    key = KEYS[pid]
    openai_api.answer = (200, {"object": "list", "data": [{"id": "m1", "object": "model", "created": 0, "owned_by": "x"}]})
    r = await kc.check(pid, key)
    assert (r.ok, r.status, r.models) == (True, "ok", 1) and host in r.message
    req = openai_api.requests[0]
    assert req.headers["authorization"] == f"Bearer {key}" and req.url.path.endswith("/models")
    for code, status, ok in ERRORS:
        openai_api.answer = (code, {"error": {"message": f"Incorrect API key provided: {key}"}})
        r = await kc.check(pid, key)
        assert (r.ok, r.status, r.http) == (ok, status, code)
        _secret_free(r, key)
    openai_api.answer = openai_api.lib.ConnectError("refused")
    assert (await kc.check(pid, key)).status == "unreachable"
    openai_api.answer = openai_api.lib.ReadTimeout("slow")
    assert (await kc.check(pid, key)).status == "timeout"


async def test_an_endpoint_on_this_mac_needs_no_key(openai_api, monkeypatch):
    _env(monkeypatch)
    r = await kc.check("custom", endpoint=OLLAMA)
    assert r.ok and openai_api.bases == ["http://localhost:11434/v1"]
    assert openai_api.requests[0].headers["authorization"] == "Bearer none"


async def test_an_endpoint_form_that_doesnt_validate_is_said_plainly(openai_api):
    r = await kc.check("custom", "k-TEST-0000", endpoint={**DEEPSEEK, "base_url": "http://user:hunter2@api.deepseek.com"})
    assert (r.ok, r.status) == (False, "error") and "hunter2" not in r.message and not openai_api.requests


# ---- OpenRouter and DeepSeek: their own checks (and credit) ---------------------------------------------------


async def test_openrouter_checks_its_key_endpoint_and_says_the_credit(http):
    key = KEYS["openrouter"]
    http.answers["/api/v1/key"] = (200, {"data": {"label": "k", "limit": 10.0, "usage": 1.25, "is_free_tier": False}})
    r = await kc.check("openrouter", key)
    assert (r.ok, r.credit) == (True, {"limit": 10.0, "usage": 1.25, "free_tier": False})
    assert http.requests[0].headers["authorization"] == f"Bearer {key}"
    for code, status, ok in ERRORS:
        http.answers["/api/v1/key"] = (code, {"error": {"message": "x"}})
        r = await kc.check("openrouter", key)
        assert (r.ok, r.status) == (ok, status)
        _secret_free(r, key)


async def test_deepseek_checks_its_balance(http, monkeypatch):
    _env(monkeypatch, NEO_CUSTOM_PROVIDERS=json.dumps([{**DEEPSEEK, "env": "DEEPSEEK_API_KEY"}]), DEEPSEEK_API_KEY=DS_KEY)
    http.answers["/user/balance"] = (200, {"is_available": True, "balance_infos": [{"currency": "USD", "total_balance": "4.20"}]})
    r = await kc.check("custom:deepseek")  # (no key given: the one NEO would use, here its variable)
    assert (r.ok, r.credit) == (True, {"available": True, "balance": 4.2, "currency": "USD"})
    assert http.requests[0].headers["authorization"] == f"Bearer {DS_KEY}"
    http.answers["/user/balance"] = (200, {"is_available": False, "balance_infos": [{"currency": "USD", "total_balance": "0"}]})
    assert (await kc.check("custom:deepseek")).status == "no_credit"
    _secret_free(r, DS_KEY)


# ---- whose key -----------------------------------------------------------------------------------------------


async def test_no_key_given_tests_the_one_neo_would_use(openai_api, monkeypatch):
    _env(monkeypatch, OPENAI_API_KEY=KEYS["openai"])
    r = await kc.check("openai")
    assert r.ok and openai_api.requests[0].headers["authorization"] == f"Bearer {KEYS['openai']}"
    _env(monkeypatch)
    r = await kc.check("openai")
    assert (r.ok, r.status) == (False, "error") and "no key" in r.message


async def test_a_pasted_key_is_cleaned_and_one_with_a_line_break_refused(openai_api):
    r = await kc.check("openai", f'  "{KEYS["openai"]}"  ')
    assert r.ok and openai_api.requests[0].headers["authorization"] == f"Bearer {KEYS['openai']}"
    r = await kc.check("openai", "sk-proj-TEST\n0000")
    assert (r.ok, r.status) == (False, "error") and "TEST" not in r.message


async def test_nothing_is_printed_or_logged(http, capsys, caplog):
    import logging

    caplog.set_level(logging.DEBUG)
    http.answers["/v1beta/models"] = (401, {"error": {"message": f"key {KEYS['gemini']} rejected"}})
    await kc.check("gemini", KEYS["gemini"])
    out = capsys.readouterr()
    for text in (out.out, out.err, caplog.text):
        assert KEYS["gemini"] not in text


async def test_check_tools_asks_once_with_a_dummy_tool(openai_api, monkeypatch):
    _env(monkeypatch)
    reply = {"id": "c", "object": "chat.completion", "created": 0, "model": "qwen3:14b", "choices": [
        {"index": 0, "finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "ping", "arguments": "{}"}}]}}]}  # fmt: skip
    openai_api.answer = (200, reply)
    r = await kc.check_tools(OLLAMA)
    assert r.ok and "calls tools" in r.message
    body = json.loads(openai_api.requests[0].content)
    assert body["tools"][0]["function"]["name"] == "ping" and body["max_tokens"] == 64
    reply["choices"][0]["message"] = {"role": "assistant", "content": "pong"}
    openai_api.answer = (200, reply)
    r = await kc.check_tools(OLLAMA)
    assert not r.ok and "turn Tools off" in r.message


# ---- key shapes -----------------------------------------------------------------------------------------------


def test_shape_hints_never_block_and_name_the_likely_provider():
    assert keyshape.hint("anthropic", KEYS["openai"]) == "This looks like an OpenAI key — did you mean OpenAI?"
    assert keyshape.hint("openai", KEYS["anthropic"]) == "This looks like an Anthropic key — did you mean Anthropic?"
    assert keyshape.hint("groq", KEYS["gemini"]) == "This looks like a Gemini key — did you mean Gemini?"
    assert keyshape.hint("gemini", "nope-0000") == "Gemini keys usually start with AIza."
    assert keyshape.hint("openai", "sk-legacy-0000") is None  # (OpenAI's older shape)
    assert keyshape.hint("custom:deepseek", DS_KEY) is None  # ("sk-" alone names nobody)
    assert keyshape.hint("custom:xai", "xai-TEST-0000") is None and keyshape.hint("anthropic", f'  "{KEYS["anthropic"]}" ') is None
    assert keyshape.guess(KEYS["openrouter"]) == "openrouter" and keyshape.guess("hello") is None


def test_shape_problems_are_line_breaks_spaces_and_length():
    assert keyshape.problem(KEYS["openai"]) is None and keyshape.problem(f"'{KEYS['openai']}'") is None
    assert keyshape.problem("sk-proj-a\nb") and keyshape.problem("sk-proj a") and keyshape.problem("x" * 513)
    assert keyshape.problem("   ") == "the key is empty"
    assert "TEST" not in (keyshape.problem("sk-TEST\x07") or "")


def test_the_composer_guard_is_narrow():
    for text in (KEYS["anthropic"], f"{KEYS['openai']} please", KEYS["gemini"], KEYS["groq"], "xai-" + "a" * 40,
                 "sk-" + "A1b2" * 12):  # fmt: skip
        assert keyshape.looks_like_key(text), text
    for text in ("sk-learn is a library", "AIza is a prefix", "my key is gsk_short", "deadbeef" * 8, "", "skip it",
                 "sk-" + "a" * 20):  # fmt: skip
        assert not keyshape.looks_like_key(text), text
