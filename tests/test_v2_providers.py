"""Provider translation tests — no network. Checks the exact wire shapes each SDK needs."""

from __future__ import annotations

import base64
import json

from neo.providers.base import ImagePart, Message, ToolCall, ToolResult, ToolSpec


def _history(sig: bytes | None = None) -> list[Message]:
    return [
        Message.user("what time is it"),
        Message.assistant("", [ToolCall("call_1", "clock", {}, signature=sig)]),
        Message.tool([ToolResult("call_1", "clock", "It's noon.", images=[ImagePart(b"png", "image/png")])]),
    ]


def test_gemini_contents_echo_signature_or_bypass():
    from neo.providers.gemini import _SKIP_SIGNATURE, _to_contents

    c = _to_contents(_history(sig=b"real-sig"))
    assert [x.role for x in c] == ["user", "model", "user"]
    fc = c[1].parts[0]
    assert fc.function_call.name == "clock" and fc.thought_signature == b"real-sig"
    # foreign tool calls (Groq/Claude) carry no signature → documented bypass value
    c2 = _to_contents(_history(sig=None))
    assert c2[1].parts[0].thought_signature == _SKIP_SIGNATURE
    fr = c2[2].parts[0].function_response
    assert fr.name == "clock" and fr.id == "call_1" and fr.response["result"] == "It's noon."
    assert c2[2].parts[1].inline_data.mime_type == "image/png"


def test_gemini_skips_empty_model_turn():
    from neo.providers.gemini import _to_contents

    c = _to_contents([Message.user("hi"), Message.assistant("")])
    assert len(c) == 1  # an empty model turn would 400


def test_gemini_tool_declarations_use_json_schema():
    from neo.providers.gemini import _tools

    t = _tools(
        [ToolSpec("x", "d", {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]})]
    )
    d = t[0].function_declarations[0]
    assert d.name == "x" and d.parameters_json_schema["required"] == ["a"]


def test_openai_messages_round_trip_tool_ids_and_vision_flag():
    from neo.providers.openai_compat import _to_messages

    m = _to_messages(_history(), "sys", vision=False)
    assert m[0] == {"role": "system", "content": "sys"}
    assert m[2]["tool_calls"][0]["id"] == "call_1"
    assert json.loads(m[2]["tool_calls"][0]["function"]["arguments"]) == {}
    assert m[3]["role"] == "tool" and m[3]["tool_call_id"] == "call_1"
    assert "[screenshot attached]" in m[3]["content"]  # no vision → image dropped but flagged
    mv = _to_messages(_history(), "", vision=True)
    img = mv[-1]["content"][1]["image_url"]["url"]
    assert img.startswith("data:image/png;base64,") and base64.b64decode(img.split(",")[1]) == b"png"


def test_claude_messages_shape():
    from neo.providers.claude import _to_messages

    m = _to_messages(_history())
    assert m[1]["content"][0] == {"type": "tool_use", "id": "call_1", "name": "clock", "input": {}}
    tr = m[2]["content"][0]
    assert tr["type"] == "tool_result" and tr["tool_use_id"] == "call_1" and tr["is_error"] is False
    assert tr["content"][1]["type"] == "image"


def test_gemini_keys_are_deduped_and_ordered(monkeypatch):
    from neo import config

    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setenv("GEMINI_API_KEY", "k1")
    monkeypatch.setenv("GEMINI_API_KEYS", " k2 ,k1,,k3")
    assert config.Settings().gemini_keys == ["k1", "k2", "k3"]
    monkeypatch.setattr(config, "_settings", None)
