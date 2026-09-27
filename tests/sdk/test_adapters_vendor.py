"""The adapters' tool specs and replies are exactly what each installed vendor SDK takes: checked
offline against the SDK's own request types (openai's and anthropic's TypedDicts, google-genai's
extra="forbid" pydantic models, langchain-core's tool and message classes), then sent through each
real client, whose HTTP is an httpx.MockTransport — nothing leaves the process, no key is real.

Every check is skipped when that SDK isn't installed (they're extras)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel
from vendor_shapes import assert_conforms, assert_rejected

import neo2
from neo.providers.base import ImagePart
from neo2.adapters import anthropic as neo_anthropic
from neo2.adapters import gemini as neo_gemini
from neo2.adapters import langchain as neo_lc
from neo2.adapters import openai as neo_openai
from neo2.adapters._common import agent_parameters
from neo2.sdk.approval import ApprovalRequest
from neo2.sdk.results import ToolOutcome
from neo2.testing import ScriptedBrain, ToolCall, Turn

pytestmark = pytest.mark.portable  # no macOS or extra needed: CI's sdk-core job runs it on Linux too

ROOT = Path(__file__).resolve().parents[2]
NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
REFUND = Turn(tool_calls=[ToolCall("1", "refund", {"order_id": "812"})])
CONFIRM = (
    "NEEDS_CONFIRM: refund order 812 — waiting for the user's decision in the app. "
    "Don't retry or work around it."
)


class Item(BaseModel):
    sku: str
    qty: int = 1


def _host_tools(ran: list) -> list:
    @neo2.host_tool(risk="destructive", confirm=lambda a: f"refund order {a['order_id']}")
    async def refund(order_id: str) -> str:
        """Refund an order."""
        ran.append(order_id)
        return f"refunded {order_id}"

    @neo2.host_tool(read_only=True)
    def snap(label: str) -> neo2.ToolOutput:
        """Take a picture."""
        return neo2.ToolOutput(f"picture of {label}", images=[ImagePart(PNG)])

    @neo2.host_tool
    def order(items: list[Item], priority: str = "low", note: str | None = None) -> str:
        """Place an order (a schema with nested objects, defaults and optionals)."""
        return "ordered"

    return [refund, snap, order]


def _agent(turns=(), ran: list | None = None, **cfg):
    config = neo2.AgentConfig(**{"tools": "none", "fast_paths": False, **cfg})
    return neo2.AsyncAgent(
        config, provider=ScriptedBrain(list(turns)), host_tools=_host_tools([] if ran is None else ran)
    )


def _every_tool(tmp_path):
    """Every NEO tool this machine can load (their schemas are what gets exported), plus host
    tools with typed signatures. Nothing is run."""
    return _agent(tools="all", data_dir=tmp_path / "all", memory=True)


def _outcomes() -> list[ToolOutcome]:
    card = ApprovalRequest(
        "0123456789abcdef", "refund", "refund order 812", "{}", {}, (), "default", "r", 0.0, 120.0
    )
    fenced = '<untrusted source="snap">\npicture\n</untrusted>'
    return [
        ToolOutcome(
            "ok",
            fenced,
            (ImagePart(PNG), ImagePart(PNG, "image/jpg"), ImagePart(PNG, "image/tiff")),
            None,
            True,
        ),
        ToolOutcome("ok", "", (), None, False),
        ToolOutcome("error", "Error: no such order", (), None, False),
        ToolOutcome(
            "needs_approval", "NEEDS_CONFIRM: refund order 812 — Should I go ahead?", (), card, False
        ),
        ToolOutcome("denied", "The user's OK is required and was not given.", (), card, False),
        ToolOutcome("needs_user", "Which order?", (), None, False),
        ToolOutcome("stopped", "Stopped.", (), None, False),
    ]


def _mock(sent: list[dict], replies: list[dict], sdk: Any = None) -> Any:
    """An HTTP client for a vendor SDK — of the httpx package that SDK is built on (openai and
    anthropic moved to httpx2) — that records each request's JSON body and answers from
    `replies`, never touching a network."""
    lib = getattr(sys.modules.get(f"{sdk.__name__}._base_client") if sdk else None, "httpx2", None) or httpx

    def respond(request: Any) -> Any:
        body = json.loads(request.content or b"{}")
        body["__path__"] = request.url.path
        sent.append(body)
        return lib.Response(200, json=replies[len(sent) - 1])

    return lib.AsyncClient(transport=lib.MockTransport(respond))


def _field(d: dict, snake: str) -> Any:
    """A field of a google-genai request body, sent in either case."""
    camel = re.sub(r"_([a-z])", lambda m: m.group(1).upper(), snake)
    return d[camel] if camel in d else d[snake]


# ---- specs, against the SDKs' own types ----------------------------------------------------------


async def test_openai_specs_are_what_the_openai_sdk_takes(tmp_path):
    chat_t = pytest.importorskip("openai.types.chat")
    resp_t = pytest.importorskip("openai.types.responses")
    async with _every_tool(tmp_path) as agent:
        chat, _ = neo_openai.agent_tool(agent)
        responses, _ = neo_openai.agent_tool(agent, api="responses")
        chats = [chat, *agent.export_tools("openai_chat")]
        flats = [responses, *agent.export_tools("openai_responses")]
    assert len(chats) > 10
    for spec in chats:
        assert_conforms(spec, chat_t.ChatCompletionFunctionToolParam)
        assert_conforms(spec, chat_t.ChatCompletionToolUnionParam)
        chat_t.ChatCompletionFunctionTool.model_validate(spec)
        assert NAME.match(spec["function"]["name"]) and spec["function"]["parameters"]["type"] == "object"
    for spec in flats:
        assert_conforms(spec, resp_t.FunctionToolParam)
        assert_conforms(spec, resp_t.ToolParam)
        resp_t.FunctionTool.model_validate(spec)
        assert spec["strict"] is False
    # the checker itself: a wrong shape fails
    assert_rejected(
        {"type": "function", "name": "neo", "parameters": {}}, chat_t.ChatCompletionFunctionToolParam
    )
    assert_rejected({k: v for k, v in responses.items() if k != "strict"}, resp_t.FunctionToolParam)


async def test_anthropic_specs_are_what_the_anthropic_sdk_takes(tmp_path):
    at = pytest.importorskip("anthropic.types")
    async with _every_tool(tmp_path) as agent:
        specs = [neo_anthropic.agent_tool(agent)[0], *agent.export_tools("anthropic")]
    for spec in specs:
        assert_conforms(spec, at.ToolParam)
        assert_conforms(spec, at.ToolUnionParam)
        assert NAME.match(spec["name"]) and spec["input_schema"]["type"] == "object"
    assert_rejected({"name": "neo", "description": "x"}, at.ToolParam)  # (any dict is an input_schema to it)


async def test_gemini_declarations_are_what_google_genai_takes(tmp_path):
    types = pytest.importorskip("google.genai.types")
    from pydantic import ValidationError

    async with _every_tool(tmp_path) as agent:
        decls = [neo_gemini.agent_tool(agent)[0], *agent.export_tools("gemini")]
    for d in decls:
        fd = types.FunctionDeclaration.model_validate(d)  # extra="forbid": no key it doesn't know
        assert fd.parameters_json_schema == d["parameters_json_schema"] and fd.parameters is None
        assert types.FunctionDeclaration.model_validate_json(json.dumps(d)) == fd
        assert re.match(r"^[a-zA-Z_][a-zA-Z0-9_.:-]{0,63}$", d["name"])
    types.Tool(function_declarations=decls)
    with pytest.raises(ValidationError):
        types.FunctionDeclaration.model_validate({"name": "neo", "input_schema": {"type": "object"}})


async def test_the_langchain_tool_is_a_langchain_tool():
    pytest.importorskip("langchain_core")
    chat_t = pytest.importorskip("openai.types.chat")
    from langchain_core.tools import BaseTool, StructuredTool
    from langchain_core.utils.function_calling import convert_to_openai_tool

    async with _agent() as agent:
        tool = neo_lc.agent_tool(agent)
        assert isinstance(tool, StructuredTool) and isinstance(tool, BaseTool) and tool.name == "neo"
        as_openai = convert_to_openai_tool(tool)  # what LangChain sends a model
        assert_conforms(as_openai, chat_t.ChatCompletionFunctionToolParam)
        params = as_openai["function"]["parameters"]
        assert params["required"] == ["task"] and set(params["properties"]) == {"task", "context"}
        assert params["properties"]["task"]["type"] == "string"


# ---- replies, against the SDKs' own types --------------------------------------------------------


def test_openai_replies_are_chat_messages_and_responses_items():
    chat_t = pytest.importorskip("openai.types.chat")
    resp_t = pytest.importorskip("openai.types.responses")
    from openai.types.responses.response_input_item_param import FunctionCallOutput

    for i, out in enumerate(_outcomes()):
        msgs = neo_openai.tool_message(out, f"call_{i}")
        assert_conforms(msgs[0], chat_t.ChatCompletionToolMessageParam)
        for m in msgs:
            assert_conforms(m, chat_t.ChatCompletionMessageParam)
        for m in msgs[1:]:
            assert_conforms(m, chat_t.ChatCompletionUserMessageParam)
        item = neo_openai.function_call_output(out, f"call_{i}")
        assert_conforms(item, FunctionCallOutput)
        assert_conforms(item, resp_t.ResponseInputItemParam)
        assert "0123456789abcdef" not in json.dumps([msgs, item])
    first = neo_openai.tool_message(_outcomes()[0], "c")
    assert len(first) == 2 and len(first[1]["content"]) == 3  # a note, png, jpeg (tiff left out)


def test_anthropic_tool_results_are_content_blocks():
    at = pytest.importorskip("anthropic.types")
    blocks = [neo_anthropic.tool_result(out, f"toolu_{i}") for i, out in enumerate(_outcomes())]
    for b in blocks:
        assert_conforms(b, at.ToolResultBlockParam)
        assert_conforms(b, at.ContentBlockParam)
    assert_conforms({"role": "user", "content": blocks}, at.MessageParam)
    assert [b.get("is_error", False) for b in blocks] == [False, False, True, False, False, False, False]
    assert_rejected({"type": "tool_result", "tool_use_id": "t", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/tiff", "data": "AA"}}]}, at.ToolResultBlockParam)  # fmt: skip


def test_gemini_function_responses_are_parts():
    types = pytest.importorskip("google.genai.types")
    for i, out in enumerate(_outcomes()):
        parts = neo_gemini.function_response(out, "snap", call_id=f"fc_{i}")
        validated = [types.Part.model_validate(p) for p in parts]
        assert validated[0].function_response is not None and validated[0].function_response.id == f"fc_{i}"
        types.Content(role="user", parts=parts)
    shots = neo_gemini.function_response(_outcomes()[0], "snap")
    assert [types.Part.model_validate(p).inline_data.mime_type for p in shots[1:]] == [
        "image/png",
        "image/jpeg",
        "image/tiff",
    ]


def test_langchain_tool_messages_use_standard_blocks():
    pytest.importorskip("langchain_core")
    from langchain_core.messages import ToolMessage
    from langchain_core.messages.content import ImageContentBlock, TextContentBlock

    for i, out in enumerate(_outcomes()):
        m = neo_lc.tool_message(out, f"call_{i}")
        assert isinstance(m, ToolMessage) and m.status == ("error" if out.status == "error" else "success")
        if isinstance(m.content, list):
            assert_conforms(m.content[0], TextContentBlock)
            for b in m.content[1:]:
                assert_conforms(b, ImageContentBlock)
            assert [b["type"] for b in m.content_blocks] == ["text", "image", "image"]


# ---- whole loops through the real clients, over a mock transport ---------------------------------


async def test_openai_chat_loop_offline():
    openai_sdk = pytest.importorskip("openai")
    sent: list[dict] = []
    call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "neo", "arguments": '{"task": "refund order 812"}'},
    }
    snap = {
        "id": "call_2",
        "type": "function",
        "function": {"name": "snap", "arguments": '{"label": "desk"}'},
    }
    replies = [
        {"id": "c1", "object": "chat.completion", "created": 0, "model": "m", "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": [call, snap]}}]},
        {"id": "c2", "object": "chat.completion", "created": 0, "model": "m", "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "It's waiting on you."}}]},
    ]  # fmt: skip
    http = _mock(sent, replies, openai_sdk)
    client = openai_sdk.AsyncOpenAI(
        api_key="sk-test", base_url="http://model.test/v1", max_retries=0, http_client=http
    )
    ran: list = []
    async with _agent([REFUND], ran) as agent:
        spec, run = neo_openai.agent_tool(agent)
        tools = [spec, *agent.export_tools("openai_chat")]
        msgs: list[dict] = [{"role": "user", "content": "refund order 812 and show me my desk"}]
        first = await client.chat.completions.create(model="m", messages=msgs, tools=tools)
        reply = first.choices[0].message
        msgs.append(reply.model_dump(exclude_none=True))
        answers: list[dict] = []
        async with agent.turn() as t:
            for tc in reply.tool_calls:
                if tc.function.name == spec["function"]["name"]:
                    answers.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": await run(tc.function.arguments)}
                    )
                else:
                    out = await agent.call_tool(tc.function.name, json.loads(tc.function.arguments), turn=t)
                    answers += neo_openai.tool_message(out, tc.id)
        msgs += sorted(answers, key=lambda m: m["role"] != "tool")  # tool messages first
        second = await client.chat.completions.create(model="m", messages=msgs, tools=tools)
    await http.aclose()
    assert sent[0]["__path__"] == "/v1/chat/completions" and sent[0]["tools"][0] == spec
    assert [m["role"] for m in sent[1]["messages"]] == ["user", "assistant", "tool", "tool", "user"]
    assert sent[1]["messages"][2] == {"role": "tool", "tool_call_id": "call_1", "content": CONFIRM}
    assert sent[1]["messages"][4]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert ran == [] and second.choices[0].message.content == "It's waiting on you."


async def test_openai_responses_loop_offline():
    openai_sdk = pytest.importorskip("openai")
    sent: list[dict] = []
    fc = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "neo",
        "arguments": '{"task": "refund order 812"}',
        "status": "completed",
    }
    fc2 = {
        "type": "function_call",
        "id": "fc_2",
        "call_id": "call_2",
        "name": "snap",
        "arguments": '{"label": "desk"}',
        "status": "completed",
    }
    done = {
        "type": "message",
        "id": "m_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "Waiting on you.", "annotations": []}],
    }
    base = {
        "object": "response",
        "created_at": 0,
        "model": "m",
        "status": "completed",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }
    replies = [{**base, "id": "r1", "output": [fc, fc2]}, {**base, "id": "r2", "output": [done]}]  # fmt: skip
    http = _mock(sent, replies, openai_sdk)
    client = openai_sdk.AsyncOpenAI(
        api_key="sk-test", base_url="http://model.test/v1", max_retries=0, http_client=http
    )
    ran: list = []
    async with _agent([REFUND], ran) as agent:
        spec, run = neo_openai.agent_tool(agent, api="responses")
        tools = [spec, *agent.export_tools("openai_responses")]
        items: list[dict] = [{"role": "user", "content": "refund order 812 and show me my desk"}]
        first = await client.responses.create(model="m", input=items, tools=tools)
        items += [o.model_dump(exclude_none=True) for o in first.output]
        async with agent.turn() as t:
            for o in first.output:
                if o.name == spec["name"]:
                    items.append(
                        {
                            "type": "function_call_output",
                            "call_id": o.call_id,
                            "output": await run(o.arguments),
                        }
                    )
                else:
                    out = await agent.call_tool(o.name, json.loads(o.arguments), turn=t)
                    items.append(neo_openai.function_call_output(out, o.call_id))
        second = await client.responses.create(model="m", input=items, tools=tools)
    await http.aclose()
    assert sent[0]["__path__"] == "/v1/responses" and sent[0]["tools"][0] == spec
    outs = [i for i in sent[1]["input"] if i.get("type") == "function_call_output"]
    assert outs[0] == {"type": "function_call_output", "call_id": "call_1", "output": CONFIRM}
    assert outs[1]["output"][0]["text"].startswith('<untrusted source="snap">')
    assert outs[1]["output"][1]["type"] == "input_image"
    assert ran == [] and second.output_text == "Waiting on you."


async def test_anthropic_loop_offline():
    anthropic_sdk = pytest.importorskip("anthropic")
    sent: list[dict] = []
    usage = {"input_tokens": 1, "output_tokens": 1}
    first_content = [
        {"type": "tool_use", "id": "toolu_1", "name": "neo", "input": {"task": "refund order 812"}},
        {"type": "tool_use", "id": "toolu_2", "name": "snap", "input": {"label": "desk"}},
    ]
    replies = [
        {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-test", "content": first_content, "stop_reason": "tool_use", "stop_sequence": None, "usage": usage},
        {"id": "msg_2", "type": "message", "role": "assistant", "model": "claude-test", "content": [{"type": "text", "text": "Waiting on you."}], "stop_reason": "end_turn", "stop_sequence": None, "usage": usage},
    ]  # fmt: skip
    http = _mock(sent, replies, anthropic_sdk)
    client = anthropic_sdk.AsyncAnthropic(
        api_key="sk-ant-test", base_url="http://model.test", max_retries=0, http_client=http
    )
    ran: list = []
    async with _agent([REFUND], ran) as agent:
        spec, run = neo_anthropic.agent_tool(agent)
        tools = [spec, *agent.export_tools("anthropic")]
        msgs: list[dict] = [{"role": "user", "content": "refund order 812 and show me my desk"}]
        first = await client.messages.create(model="claude-test", max_tokens=256, messages=msgs, tools=tools)
        msgs.append(
            {"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in first.content]}
        )
        results: list[dict] = []
        async with agent.turn() as t:
            for b in first.content:
                if b.name == spec["name"]:
                    results.append(
                        {"type": "tool_result", "tool_use_id": b.id, "content": await run(b.input)}
                    )
                else:
                    results.append(
                        neo_anthropic.tool_result(await agent.call_tool(b.name, b.input, turn=t), b.id)
                    )
        msgs.append({"role": "user", "content": results})
        second = await client.messages.create(model="claude-test", max_tokens=256, messages=msgs, tools=tools)
    await http.aclose()
    assert sent[0]["__path__"] == "/v1/messages" and sent[0]["tools"][0] == spec
    back = sent[1]["messages"][-1]["content"]
    assert back[0] == {"type": "tool_result", "tool_use_id": "toolu_1", "content": CONFIRM}
    assert back[1]["content"][1] == {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": neo_anthropic.tool_result(_outcomes()[0], "x")["content"][1]["source"]["data"]}}  # fmt: skip
    assert "is_error" not in back[1]
    assert ran == [] and second.content[0].text == "Waiting on you."


async def test_gemini_loop_offline():
    genai = pytest.importorskip("google.genai")
    from google.genai import types

    sent: list[dict] = []
    calls = [
        {"functionCall": {"id": "fc_1", "name": "neo", "args": {"task": "refund order 812"}}},
        {"functionCall": {"id": "fc_2", "name": "snap", "args": {"label": "desk"}}},
    ]
    replies = [
        {"candidates": [{"content": {"role": "model", "parts": calls}, "finishReason": "STOP"}]},
        {"candidates": [{"content": {"role": "model", "parts": [{"text": "Waiting on you."}]}, "finishReason": "STOP"}]},
    ]  # fmt: skip
    http = _mock(sent, replies)
    client = genai.Client(
        api_key="test-key",
        http_options=types.HttpOptions(base_url="http://model.test", httpx_async_client=http),
    )
    ran: list = []
    async with _agent([REFUND], ran) as agent:
        decl, run = neo_gemini.agent_tool(agent)
        config = types.GenerateContentConfig(
            tools=[types.Tool(function_declarations=[decl, *agent.export_tools("gemini")])]
        )
        ask = types.Content(role="user", parts=[types.Part(text="refund order 812 and show me my desk")])
        first = await client.aio.models.generate_content(model="gemini-test", contents=[ask], config=config)
        parts: list[dict] = []
        async with agent.turn() as t:
            for fc in first.function_calls:
                if fc.name == decl["name"]:
                    text = await run(fc.args)
                    parts.append(
                        {"function_response": {"id": fc.id, "name": fc.name, "response": {"output": text}}}
                    )
                else:
                    out = await agent.call_tool(fc.name, fc.args, turn=t)
                    parts += neo_gemini.function_response(out, fc.name, call_id=fc.id)
        contents = [ask, first.candidates[0].content, types.Content(role="user", parts=parts)]
        second = await client.aio.models.generate_content(
            model="gemini-test", contents=contents, config=config
        )
    await http.aclose()
    assert sent[0]["__path__"].endswith("/models/gemini-test:generateContent")
    first_decl = _field(sent[0]["tools"][0], "function_declarations")[0]
    assert first_decl["name"] == "neo" and _field(first_decl, "parameters_json_schema") == agent_parameters()
    back = sent[1]["contents"][-1]["parts"]
    assert _field(back[0], "function_response") == {
        "id": "fc_1",
        "name": "neo",
        "response": {"output": CONFIRM},
    }
    assert _field(back[1], "function_response")["response"]["output"].startswith('<untrusted source="snap">')
    assert _field(_field(back[2], "inline_data"), "mime_type") == "image/png"
    assert ran == [] and second.text == "Waiting on you."


async def test_langchain_tool_call_offline():
    pytest.importorskip("langchain_core")
    from langchain_core.messages import ToolMessage

    ran: list = []
    async with _agent([REFUND], ran) as agent:
        tool = neo_lc.agent_tool(agent)
        msg = await tool.ainvoke(
            {"name": "neo", "args": {"task": "refund order 812"}, "id": "call_1", "type": "tool_call"}
        )
        assert agent.pending() is not None
    assert isinstance(msg, ToolMessage) and msg.tool_call_id == "call_1" and msg.content == CONFIRM
    assert ran == []


# ---- what importing an adapter costs --------------------------------------------------------------


def test_the_adapters_import_no_vendor_sdk_and_no_decide(tmp_path):
    home = tmp_path / "sub_home"
    home.mkdir()
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("NEO_", "GEMINI", "GROQ", "ANTHROPIC", "OPENAI", "PYTEST"))
    }
    env.update(HOME=str(home), PYTHONPATH=str(ROOT), NEO_DATA_DIR=str(tmp_path / "app_data"))
    heavy = ("openai", "anthropic", "google.genai", "langchain_core", "neo.decide", "torch", "AppKit")
    code = (
        "import sys, json, asyncio\n"
        "import neo2.adapters, neo2.adapters.openai, neo2.adapters.anthropic, neo2.adapters.gemini\n"
        "import neo2.adapters.langchain\n"
        f"at_import = [m for m in {heavy!r} if m in sys.modules]\n"
        "import neo2\n"
        "from neo2.testing import ScriptedBrain\n"
        "async def main():\n"
        "    cfg = neo2.AgentConfig(tools='none', fast_paths=False)\n"
        "    async with neo2.AsyncAgent(cfg, provider=ScriptedBrain(['hi'])) as a:\n"
        "        spec, run = neo2.adapters.openai.agent_tool(a)\n"
        "        return await run({'task': 'hello'})\n"
        "text = asyncio.run(main())\n"
        f"after_use = [m for m in {heavy!r} if m in sys.modules]\n"
        "print(json.dumps({'import': at_import, 'use': after_use, 'text': text}))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(tmp_path), env=env, timeout=120
    )
    assert out.returncode == 0, out.stderr[-3000:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got == {"import": [], "use": [], "text": "hi"}
    assert not (home / ".neo").exists() and not (tmp_path / "app_data").exists()
