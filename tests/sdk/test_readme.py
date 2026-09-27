"""The README's "Use NEO from your own app" snippets run as written.

Each labelled block (``<!-- snippet: name -->``) runs against fakes: a scripted brain behind NEO's
provider classes, each vendor SDK's real client over a mock HTTP transport, fake host services
(billing, the approval UI) and fake handlers for NEO's web tools. Fast paths are off and sockets are
blocked, so nothing reaches a network, a model, an app or ~/.neo. A vendor SDK's snippet is skipped
when that SDK isn't installed (they're extras)."""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import re
import shlex
import socket
import sys
import tomllib
from pathlib import Path
from typing import Any

import httpx
import pytest

import neo2
import neo2.providers
from neo2.testing import ScriptedBrain, ToolCall, Turn

pytestmark = pytest.mark.portable  # Linux or macOS, any install (what needs an extra skips): CI's sdk-core job

ROOT = Path(__file__).resolve().parents[2]
SECTION = (ROOT / "README.md").read_text().split("\n## Use NEO from your own app\n", 1)[1].split("\n## ", 1)[0]
BLOCK = re.compile(r"<!-- snippet: ([a-z-]+) -->\n```(\w+)\n(.*?)```", re.S)
SNIPPETS: dict[str, tuple[str, str]] = {name: (lang, code) for name, lang, code in BLOCK.findall(SECTION)}
PYTHON = ("quickstart", "sync", "approvals", "approver", "openai", "anthropic", "gemini", "langchain", "tools")

REFUND = Turn(tool_calls=[ToolCall("1", "refund", {"order_id": "812", "amount": 49.5})])
SUMMARY = "Refund 49.5 on order 812"
CONFIRM = f"NEEDS_CONFIRM: {SUMMARY} — waiting for the user's decision in the app. Don't retry or work around it."
WEB_TOOLS = ("web_search", "web_fetch", "weather", "clock")


class Billing:
    def __init__(self) -> None:
        self.refunds: list[tuple[str, float, str]] = []

    async def refund(self, order_id: str, amount: float, *, tenant: str) -> str:
        self.refunds.append((order_id, amount, tenant))
        return f"Refunded {amount:.2f} on order {order_id}."


class UI:
    """The host's approval UI: records what it was shown, answers `answer`."""

    def __init__(self, answer: bool = True) -> None:
        self.shown: list[tuple[str, str]] = []
        self.answer = answer

    async def confirm(self, summary: str, detail: str) -> bool:
        self.shown.append((summary, detail))
        return self.answer


@pytest.fixture(autouse=True)
def hermetic(monkeypatch):
    """No fast path, NEO's web tools answer from here, and no socket may connect."""
    from neo2.core.agent import fastpath
    from neo2.core.agent.registry import catalog
    from neo2.core.tools import load

    monkeypatch.setattr(fastpath, "plan", lambda text, **kw: None)
    load(list(WEB_TOOLS))
    for name in WEB_TOOLS:  # before any agent copies them
        monkeypatch.setattr(catalog().get(name), "handler", lambda a, c, n=name: f"({n} is faked in this test)")

    def no_network(*a: Any, **kw: Any) -> Any:
        raise AssertionError(f"a README snippet tried to reach the network: {a[:2]}")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", no_network)
    monkeypatch.setattr(asyncio, "create_subprocess_shell", no_network)


PROVIDERS = ("OpenAICompatible", "Gemini", "Claude", "Groq")


def brain_is(monkeypatch, brain: ScriptedBrain) -> None:
    """NEO's provider classes, as a snippet calls them, return `brain` — once the call fits the
    real signature (a README call that doesn't would fail for a reader too)."""
    for name in PROVIDERS:
        monkeypatch.setattr(neo2.providers, name, _checked(getattr(neo2.providers, name), brain))


def _checked(real: Any, brain: ScriptedBrain) -> Any:
    sig = inspect.signature(real)

    def make(*a: Any, **kw: Any) -> ScriptedBrain:
        sig.bind(*a, **kw)  # TypeError, as the real one would raise
        return brain

    return make


async def run(name: str, ns: dict[str, Any]) -> dict[str, Any]:
    """Run one snippet in `ns`: top-level await on this loop, sync code on a thread of its own."""
    lang, code = SNIPPETS[name]
    assert lang == "python"
    compiled = compile(code, f"README.md <{name}>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    ns.setdefault("__name__", f"readme_{name}")
    if compiled.co_flags & inspect.CO_COROUTINE:
        await eval(compiled, ns)
    else:
        await asyncio.to_thread(exec, compiled, ns)
    return ns


def agent_with_refund(turns: list, ran: list, audit: list | None = None) -> neo2.AsyncAgent:
    """The agent the adapter snippets call `neo`: it has the README's refund tool (gated)."""

    @neo2.host_tool(risk="destructive", confirm=lambda a: f"Refund {a['amount']} on order {a['order_id']}")
    async def refund(order_id: str, amount: float) -> str:
        """Refund an order, in part or in full."""
        ran.append(order_id)
        return f"Refunded {amount:.2f} on order {order_id}."

    cfg = neo2.AgentConfig(tools="none", fast_paths=False)
    return neo2.AsyncAgent(cfg, provider=ScriptedBrain(turns), host_tools=[refund], on_audit=None if audit is None else audit.append)


def mock_http(sent: list[dict], replies: list[dict], sdk: Any = None) -> Any:
    """An HTTP client of the httpx package the SDK is built on (openai and anthropic moved to
    httpx2): records each request body and answers from `replies`, never touching a network."""
    lib = getattr(sys.modules.get(f"{sdk.__name__}._base_client") if sdk else None, "httpx2", None) or httpx

    def respond(request: Any) -> Any:
        body = json.loads(request.content or b"{}")
        sent.append(body)
        return lib.Response(200, json=replies[len(sent) - 1])

    return lib.AsyncClient(transport=lib.MockTransport(respond))


# ---- the section itself ------------------------------------------------------------------------


def test_every_code_block_in_the_section_is_checked():
    opening = re.findall(r"^```(\w+)$", SECTION, re.M)
    labelled = [lang for _, lang, _ in BLOCK.findall(SECTION)]
    assert opening.count("python") == labelled.count("python") and opening.count("json") == labelled.count("json")
    assert sorted(n for n, (lang, _) in SNIPPETS.items() if lang == "python") == sorted(PYTHON)
    for word in ("protected_apps", "Approvals: only your code says yes", "What NEO never exposes"):
        assert word in SECTION


# ---- the agent, sync and async -------------------------------------------------------------------


async def test_quickstart(monkeypatch, capsys):
    brain = ScriptedBrain(["It's 24 °C and sunny in Lisbon this afternoon.", "Porto: 21 °C and cloudy."])
    brain_is(monkeypatch, brain)
    ns = await run("quickstart", {})
    assert ns["r"].status == "done" and ns["neo"].closed
    out = capsys.readouterr().out
    assert "done It's 24 °C and sunny in Lisbon" in out and "Porto: 21 °C and cloudy." in out


def test_brain_is_holds_a_snippet_to_the_real_signatures(monkeypatch):
    brain_is(monkeypatch, ScriptedBrain())
    assert neo2.providers.OpenAICompatible(model="m", api_key="k") is not None
    with pytest.raises(TypeError):
        neo2.providers.OpenAICompatible(model="m", key="k")  # no such argument
    with pytest.raises(TypeError):
        neo2.providers.Groq()  # its key is required


def test_the_providers_a_snippet_names_in_passing_take_those_arguments():
    """"# or Gemini(key), Claude(key), Groq(key)": what a reader would type instead."""
    named = [
        (name, arg)
        for lang, code in SNIPPETS.values()
        if lang == "python"
        for name, arg in re.findall(r"\b([A-Z]\w+)\((\w+)\)", " ".join(re.findall(r"#.*", code)))
        if name in PROVIDERS
    ]
    assert {n for n, _ in named} >= {"Gemini", "Claude", "Groq"}
    for name, arg in named:
        inspect.signature(getattr(neo2.providers, name)).bind(arg)  # one positional: the key


async def test_sync_twin(capsys):
    ns = await run("sync", {"neo2": neo2, "brain": ScriptedBrain(["It's 9:41 pm in Tokyo."])})
    assert ns["r"].status == "done" and ns["neo"].closed
    assert "It's 9:41 pm in Tokyo." in capsys.readouterr().out


# ---- approvals -----------------------------------------------------------------------------------


async def test_approvals_deferred(capsys):
    audit: list[dict] = []
    neo = neo2.AsyncAgent(
        neo2.AgentConfig(tools="none"),
        provider=ScriptedBrain([REFUND, "Refunded 49.50 on order 812."]),
        on_audit=audit.append,
    )
    billing, ui = Billing(), UI()
    async with neo:
        ns = await run("approvals", {"neo2": neo2, "neo": neo, "billing": billing, "my_ui": ui})
    assert ui.shown[0][0] == SUMMARY and json.loads(ui.shown[0][1]) == {"order_id": "812", "amount": 49.5}
    assert billing.refunds == [("812", 49.5, "acme")]  # once, with the card's arguments, after the yes
    assert ns["r"].status == "done" and "done Refunded 49.50 on order 812." in capsys.readouterr().out
    assert [a["status"] for a in audit] == ["staged", "confirmed"]
    assert audit[1]["via"] == {"channel": "web-ui", "approver": "user:42"}


async def test_approvals_declined():
    billing = Billing()
    async with neo2.AsyncAgent(neo2.AgentConfig(tools="none"), provider=ScriptedBrain([REFUND])) as neo:
        ns = await run("approvals", {"neo2": neo2, "neo": neo, "billing": billing, "my_ui": UI(answer=False)})
    assert billing.refunds == [] and ns["r"].status == "denied"


async def test_inline_approver():
    ui, ran = UI(), []
    ns = await run("approver", {"neo2": neo2, "brain": ScriptedBrain([REFUND, "Refunded."]), "my_ui": ui})
    neo = ns["neo"]
    try:

        @neo.tool(risk="destructive")
        def refund(order_id: str, amount: float) -> str:
            """Refund an order."""
            ran.append(order_id)
            return "refunded"

        r = await neo.ask("Refund order 812 in full")
        assert r.status == "done" and ran == ["812"] and len(ui.shown) == 1  # asked before ask() returned
    finally:
        await neo.aclose()


# ---- the framework adapters ------------------------------------------------------------------------


async def test_openai_whole_agent_tool(monkeypatch):
    openai_sdk = pytest.importorskip("openai")
    sent: list[dict] = []
    call = {"id": "call_1", "type": "function", "function": {"name": "neo", "arguments": '{"task": "Refund order 812 in full"}'}}
    replies = [{"id": "c1", "object": "chat.completion", "created": 0, "model": "m", "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": [call]}}]}]  # fmt: skip
    http = mock_http(sent, replies, openai_sdk)
    real = openai_sdk.AsyncOpenAI
    monkeypatch.setattr(
        openai_sdk,
        "AsyncOpenAI",
        lambda *a, **kw: real(api_key="sk-test", base_url="http://model.test/v1", max_retries=0, http_client=http),
    )
    ran: list[str] = []
    audit: list[dict] = []
    ui = UI()
    async with agent_with_refund([REFUND, "Refunded 49.50 on order 812."], ran, audit) as neo:
        ns = await run("openai", {"neo2": neo2, "neo": neo, "my_ui": ui})
    await http.aclose()
    msgs = ns["messages"]
    assert sent[0]["tools"] == [ns["spec"]]
    assert msgs[2] == {"role": "tool", "tool_call_id": "call_1", "content": CONFIRM}
    assert ui.shown[0][0] == SUMMARY and ran == ["812"]
    # the refund tool's output is third-party content by default, so the answer NEO wrote after it is fenced
    assert msgs[3] == {"role": "user", "content": '<untrusted source="neo">\nRefunded 49.50 on order 812.\n</untrusted>'}
    card_ids = {a["action"] for a in audit}
    assert card_ids and not any(i in json.dumps(msgs) for i in card_ids)  # no action id reaches the model


async def test_anthropic_whole_agent_tool(monkeypatch):
    anthropic_sdk = pytest.importorskip("anthropic")
    sent: list[dict] = []
    usage = {"input_tokens": 1, "output_tokens": 1}
    use = {"type": "tool_use", "id": "toolu_1", "name": "neo", "input": {"task": "Refund order 812"}}
    replies = [{"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-test", "content": [use], "stop_reason": "tool_use", "stop_sequence": None, "usage": usage}]  # fmt: skip
    http = mock_http(sent, replies, anthropic_sdk)
    real = anthropic_sdk.AsyncAnthropic
    monkeypatch.setattr(
        anthropic_sdk,
        "AsyncAnthropic",
        lambda *a, **kw: real(api_key="sk-ant-test", base_url="http://model.test", max_retries=0, http_client=http),
    )
    ran: list[str] = []
    async with agent_with_refund([REFUND], ran) as neo:
        ns = await run("anthropic", {"neo2": neo2, "neo": neo})
        assert neo.pending().summary == SUMMARY
    await http.aclose()
    assert sent[0]["tools"] == [ns["spec"]]
    assert ns["results"] == [{"type": "tool_result", "tool_use_id": "toolu_1", "content": CONFIRM}] and ran == []


async def test_gemini_whole_agent_tool(monkeypatch):
    genai = pytest.importorskip("google.genai")
    from google.genai import types

    sent: list[dict] = []
    fc = {"functionCall": {"id": "fc_1", "name": "neo", "args": {"task": "Refund order 812"}}}
    replies = [{"candidates": [{"content": {"role": "model", "parts": [fc]}, "finishReason": "STOP"}]}]
    http = mock_http(sent, replies)
    real = genai.Client
    monkeypatch.setattr(
        genai,
        "Client",
        lambda *a, **kw: real(
            api_key="test-key", http_options=types.HttpOptions(base_url="http://model.test", httpx_async_client=http)
        ),
    )
    ran: list[str] = []
    async with agent_with_refund([REFUND], ran) as neo:
        ns = await run("gemini", {"neo2": neo2, "neo": neo})
    await http.aclose()
    assert len(sent) == 1 and ran == []
    (part,) = ns["parts"]
    assert part.function_response.name == "neo" and part.function_response.response == {"output": CONFIRM}


async def test_langchain_whole_agent_tool(capsys):
    pytest.importorskip("langchain_core")
    ran: list[str] = []
    async with agent_with_refund([REFUND], ran) as neo:
        ns = await run("langchain", {"neo2": neo2, "neo": neo})
        assert neo.pending().summary == SUMMARY
    assert ns["msg"].content == CONFIRM and ran == []
    printed = capsys.readouterr().out
    assert printed.startswith(f"NEEDS_CONFIRM: {SUMMARY} — waiting for the user's decision in the app.")


async def test_individual_tools():
    async with neo2.AsyncAgent(neo2.AgentConfig(tools="web"), provider=ScriptedBrain()) as neo:
        ns = await run("tools", {"neo2": neo2, "neo": neo, "messages": []})
    assert {t["function"]["name"] for t in ns["tools"]} == set(WEB_TOOLS)
    assert ns["out"].status == "ok"
    assert ns["messages"] == [{"role": "tool", "tool_call_id": "call_1", "content": "(clock is faked in this test)"}]


# ---- the MCP server's command line and client config ---------------------------------------------------


def _script() -> str:
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    (name,) = [k for k, v in scripts.items() if v == "neo2.mcp:main"]
    return name


def test_mcp_command_line():
    from neo2.mcp import _parser

    lang, code = SNIPPETS["mcp-command"]
    argv = shlex.split(code.split("#", 1)[0])
    assert lang == "bash" and argv[0] == _script()
    a = _parser().parse_args(argv[1:])
    assert (a.tools, a.expose, a.approve, a.env) == ("web", "task", "deny", True)


def test_mcp_client_config(monkeypatch):
    from neo2.mcp import _parser, _tools_arg

    lang, code = SNIPPETS["mcp-config"]
    server = json.loads(code)["mcpServers"]["neo"]
    assert lang == "json" and server["command"] == _script()
    a = _parser().parse_args(server["args"])
    assert (a.tools, a.approve, a.env) == ("web", "deny", True)
    pytest.importorskip("mcp")
    from neo2.mcp.server import ServerOptions, agent_config

    for k, v in server["env"].items():  # what the client puts in the server's environment
        monkeypatch.setenv(k, v)
    cfg = agent_config(ServerOptions(tools=_tools_arg(a.tools), approve=a.approve, env=a.env))
    assert cfg.tools == "web" and [k.get_secret_value() for k in cfg.gemini_api_keys] == ["your-key"]
