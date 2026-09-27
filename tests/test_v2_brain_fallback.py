"""The brain fallback has to survive a long agent task (no network, no real apps).

The run that died: "play chess as white" → open Chess, read it, six AppleScripts each approved by
the user, then Gemini ran out of quota key by key, answered a 400 ("function response turn comes
immediately after a function call turn"), Groq a 413 (request too large) and the local model
wasn't running. These tests rebuild each request offline and check it against the rules every
provider enforces on tool turns, the token room of the fallback brain, and what the user hears
when nothing is left.
"""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import pytest

import neo.agent.session as sess_mod
from neo.agent import confirm
from neo.agent.registry import RegisteredTool, registry
from neo.agent.session import Session
from neo.events import EventBus
from neo.providers.base import (
    AllBrainsFailed,
    ImagePart,
    Message,
    ProviderError,
    QuotaExhausted,
    RateLimited,
    RequestTooLarge,
    ToolCall,
    ToolResult,
    ToolSpec,
    Turn,
    Unavailable,
)
from neo.providers.history import NO_RESULT, flatten, normalize
from neo.reflex.schema import Decision

_OBJ = {"type": "object", "properties": {}}


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    from neo import config, events
    from neo.tools import load_all

    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setenv("NEO_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(events, "_bus", EventBus())
    confirm.cancel("test")
    load_all()
    yield
    confirm.cancel("test")


# ---- what each provider accepts around tool turns ------------------------------------------------


def gemini_violations(contents) -> list[str]:
    """Gemini's rules: the conversation opens with a user turn; a function-call turn comes right
    after a user turn or a function-response turn; a function-response turn comes right after a
    function-call turn and answers every call in it."""
    out = []

    def calls(c):
        return [p.function_call for p in c.parts or [] if p.function_call]

    def responses(c):
        return [p.function_response for p in c.parts or [] if p.function_response]

    for i, c in enumerate(contents):
        prev = contents[i - 1] if i else None
        if i == 0 and (c.role != "user" or responses(c)):
            out.append(f"{i}: opens with a {c.role} turn")
        if responses(c):
            if prev is None or prev.role != "model" or not calls(prev):
                out.append(f"{i}: function response turn not right after a function call turn")
            elif sorted((x.id, x.name) for x in calls(prev)) != sorted((x.id, x.name) for x in responses(c)):
                out.append(f"{i}: responses don't answer the calls")
        if c.role == "model" and calls(c):
            if prev is None or prev.role != "user":
                out.append(f"{i}: function call turn not after a user or response turn")
            nxt = contents[i + 1] if i + 1 < len(contents) else None
            if nxt is None or not responses(nxt):
                out.append(f"{i}: function call turn not answered")
    return out


def openai_violations(msgs: list[dict]) -> list[str]:
    out = []
    body = [m for m in msgs if m["role"] != "system"]
    if body and body[0]["role"] != "user":
        out.append(f"opens with {body[0]['role']}")
    i = 0
    while i < len(body):
        m = body[i]
        if m["role"] == "tool":
            out.append(f"{i}: tool message without its call")
        if m["role"] == "assistant" and m.get("tool_calls"):
            ids = [c["id"] for c in m["tool_calls"]]
            if len(set(ids)) != len(ids):
                out.append(f"{i}: repeated call ids")
            j = i + 1
            answered = []
            while j < len(body) and body[j]["role"] == "tool":
                answered.append(body[j]["tool_call_id"])
                j += 1
            if sorted(answered) != sorted(ids):
                out.append(f"{i}: calls {ids} answered by {answered}")
            i = j
            continue
        i += 1
    return out


def claude_violations(msgs: list[dict]) -> list[str]:
    out = []
    if msgs and msgs[0]["role"] != "user":
        out.append("opens with assistant")
    for i, m in enumerate(msgs):
        uses = (
            [b["id"] for b in m["content"] if b.get("type") == "tool_use"] if m["role"] == "assistant" else []
        )
        results = [b["tool_use_id"] for b in m["content"] if b.get("type") == "tool_result"]
        if results:
            prev = msgs[i - 1] if i else None
            prev_uses = [b["id"] for b in prev["content"] if b.get("type") == "tool_use"] if prev else []
            if prev is None or prev["role"] != "assistant" or sorted(prev_uses) != sorted(results):
                out.append(f"{i}: tool_result without its tool_use")
        if uses:
            nxt = msgs[i + 1] if i + 1 < len(msgs) else None
            got = [b["tool_use_id"] for b in nxt["content"] if b.get("type") == "tool_result"] if nxt else []
            if sorted(got) != sorted(uses):
                out.append(f"{i}: tool_use not answered")
    return out


def wire_violations(messages: list[Message]) -> dict[str, list[str]]:
    from neo.providers.claude import _to_messages as claude_msgs
    from neo.providers.gemini import _to_contents
    from neo.providers.openai_compat import _to_messages as openai_msgs

    return {
        k: v
        for k, v in {
            "gemini": gemini_violations(_to_contents(messages, "gemini-3.8-flash")),
            "openai": openai_violations(openai_msgs(messages, "sys", vision=False)),
            "claude": claude_violations(claude_msgs(messages)),
        }.items()
        if v
    }


# ---- (a) the chess run: confirm, yes, approved call, follow-up run — and the window over it -------


def _agent_reflex():
    class R:
        async def decide(self, t):
            return Decision("agent_task", 0.95, False, 0.0, False, 0.0, "laya+head", reply=False, reply_p=0.1)

    return R()


def _chess_tools(monkeypatch, ran: list[str]):
    """Stand-ins for the tools of that run (nothing real is opened, read or scripted)."""

    def add(name, handler, **kw):
        monkeypatch.setitem(registry()._tools, name, RegisteredTool(name, "test tool", _OBJ, handler, **kw))

    add("open_app", lambda a, c: "Opened Chess.")
    add(
        "ax_tree",
        lambda a, c: "Error: Accessibility isn't granted to NEO. " + "row " * 600,
        parallel_safe=True,
    )
    add("hotkey", lambda a, c: "Pressed cmd+n.")

    def zz_script(a, c):
        gate = confirm.gated("zz_script", a, "run an AppleScript", detail=a.get("script", ""))
        if isinstance(gate, str):
            return gate
        ran.append(a["script"])
        return "Error: System Events got an error" if len(ran) % 3 == 0 else "window 1, window 2"

    add("zz_script", zz_script, risk="destructive")


class ChessBrain:
    """Plays the model of that run, and checks every request it is sent against all providers."""

    name = "fake"
    supports_vision = False
    supports_tools = True

    def __init__(self):
        self.k = 0
        self.bad: list[tuple[int, dict]] = []

    async def complete(self, messages, **kw):
        self.k += 1
        if v := wire_violations(messages):
            self.bad.append((self.k, v))
        if not any(m.text.startswith("play chess") for m in messages):
            self.bad.append((self.k, {"lost": ["the request itself fell out of the window"]}))
        goal = max((i for i, m in enumerate(messages) if m.text.startswith("play chess")), default=0)
        done = [c.name for m in messages[goal:] for c in m.tool_calls]
        plan = ["open_app", "ax_tree", "hotkey", "ax_tree"]  # then AppleScripts, each approved
        if len(done) < len(plan):
            return Turn(tool_calls=[ToolCall(f"c{self.k}", plan[len(done)], {"app": "Chess"})])
        return Turn(
            tool_calls=[ToolCall(f"c{self.k}", "zz_script", {"script": f"get name of windows {self.k}"})]
        )

    async def stream(self, messages, **kw):
        yield ""


def _earlier_runs(s: Session, runs: int = 2, steps: int = 5) -> None:
    """Earlier requests in the same session, each a multi-step agent run."""
    for j in range(runs):
        s.history.append(Message.user(f"earlier request {j}"))
        for k in range(steps):
            s.history.append(Message.assistant("", [ToolCall(f"e{j}{k}", "ax_tree", {})]))
            s.history.append(Message.tool([ToolResult(f"e{j}{k}", "ax_tree", "tree " * 50)]))
        s.history.append(Message.assistant("Done."))


def test_the_chess_run_sends_every_brain_a_valid_history_through_many_approvals(monkeypatch):
    ran: list[str] = []
    _chess_tools(monkeypatch, ran)
    b = ChessBrain()
    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": b)
    monkeypatch.setattr(sess_mod.fastpath, "plan", lambda text: None)  # the model plays every step

    async def go():
        s = Session(reflex=_agent_reflex())
        _earlier_runs(s)
        r = await s.handle("play chess as white", from_user=True)
        for _ in range(16):  # far more approvals than the run that died had: the run outgrows the window
            assert r.route == "confirm", r.text
            r = await s.handle("yes", from_user=True)
        return s

    s = asyncio.run(go())
    assert len(ran) == 16 and b.k >= 20 and len(s.history) > sess_mod._MAX_HISTORY + 30
    # The session's 40-message window moved into an agent run after a few approvals; it used to
    # open the transcript with a call turn whose partner lay outside it (Gemini's 400).
    assert not b.bad, b.bad[:2]
    assert s._trimmed()[0].role == "user"


def test_the_window_opens_at_a_user_turn_and_keeps_one_long_turn_whole():
    s = Session(reflex=_agent_reflex())
    _earlier_runs(s, runs=4)  # 48 messages: history[-40] is inside a run
    assert s.history[-sess_mod._MAX_HISTORY].role != "user"
    h = s._trimmed()
    assert h[0].role == "user" and not wire_violations(h)
    assert len(h) <= sess_mod._MAX_HISTORY

    long = Session(reflex=_agent_reflex())
    _earlier_runs(long, runs=1, steps=30)  # one turn, 62 messages
    h = long._trimmed()
    assert h[0].text == "earlier request 0" and len(h) == len(long.history) and not wire_violations(h)


def test_the_follow_up_after_a_yes_pairs_every_call_including_a_halted_parallel_batch():
    """The halted turn had two calls: the one that asked, and one skipped while waiting. The
    user's yes adds a note after the results — never between a call and its result."""
    hist = [
        Message.user("play chess as white"),
        Message.assistant("", [ToolCall("a", "zz_script", {"script": "x"}), ToolCall("b", "ax_tree", {})]),
        Message.tool(
            [
                ToolResult(
                    "a", "zz_script", "NEEDS_CONFIRM: run an AppleScript — Should I go ahead?", ok=True
                ),
                ToolResult("b", "ax_tree", "skipped: waiting for the user's confirmation", ok=False),
            ]
        ),
        Message.user("[User confirmed. zz_script result: window 1] Continue and finish the task."),
    ]
    assert not wire_violations(hist)
    c = __import__("neo.providers.gemini", fromlist=["_to_contents"])._to_contents(hist, "gemini-3.8-flash")
    assert [x.role for x in c] == ["user", "model", "user", "user"]


# ---- (c) any history in → a valid transcript out, deterministically -------------------------------


def _pair(i: str, name: str = "ax_tree", text: str = "tree"):
    return Message.assistant("", [ToolCall(i, name, {})]), Message.tool([ToolResult(i, name, text)])


@pytest.mark.parametrize(
    "case",
    [
        "leading_result",
        "leading_call",
        "unanswered_call",
        "split_results",
        "text_between",
        "unknown_result",
        "repeated_ids",
        "empty_assistant",
    ],
)
def test_normalize_repairs_every_broken_shape(case):
    call, res = _pair("x")
    if case == "leading_result":
        msgs = [res, Message.user("hi"), *_pair("y")]
    elif case == "leading_call":
        msgs = [call, res, Message.user("hi"), *_pair("y")]
    elif case == "unanswered_call":
        msgs = [Message.user("hi"), call, Message.user("[User confirmed.]")]
    elif case == "split_results":
        two = Message.assistant("", [ToolCall("p", "ax_tree", {}), ToolCall("q", "clock", {})])
        msgs = [Message.user("hi"), two, Message.tool([ToolResult("q", "clock", "noon")]),
                Message.tool([ToolResult("p", "ax_tree", "tree")])]  # fmt: skip
    elif case == "text_between":
        msgs = [Message.user("hi"), call, Message.user("note"), res]
    elif case == "unknown_result":
        msgs = [
            Message.user("hi"),
            call,
            Message.tool([ToolResult("x", "ax_tree", "t"), ToolResult("zz", "o", "?")]),
        ]
    elif case == "repeated_ids":
        two = Message.assistant("", [ToolCall("call_0", "ax_tree", {}), ToolCall("call_0", "clock", {})])
        msgs = [Message.user("hi"), two,
                Message.tool([ToolResult("call_0", "ax_tree", "tree"), ToolResult("call_0", "clock", "noon")])]  # fmt: skip
    else:
        msgs = [Message.user("hi"), Message.assistant(""), call, res]
    before = [(m.role, m.text, [c.id for c in m.tool_calls], [(r.call_id, r.content) for r in m.tool_results])
              for m in msgs]  # fmt: skip
    out = normalize(msgs)
    assert not wire_violations(msgs), case  # every provider gets a valid transcript…
    assert normalize(msgs) == out and normalize(out) == out  # …the same one every time
    assert out[0].role == "user"
    # …and the caller's history (the session's own objects) is left as it was.
    assert before == [(m.role, m.text, [c.id for c in m.tool_calls], [(r.call_id, r.content) for r in m.tool_results])
                      for m in msgs]  # fmt: skip
    if case == "unanswered_call" or case == "text_between":
        assert out[2].tool_results[0].content == NO_RESULT and not out[2].tool_results[0].ok
        assert out[3].role == "user" and len(out) == 4  # the late result had lost its place
    if case == "split_results":
        assert [r.call_id for r in out[2].tool_results] == ["p", "q"] and len(out) == 3
    if case == "unknown_result":
        assert [r.call_id for r in out[2].tool_results] == ["x"]
    if case == "repeated_ids":
        ids = [c.id for c in out[1].tool_calls]
        assert len(set(ids)) == 2 and [r.content for r in out[2].tool_results] == ["tree", "noon"]
        assert [r.call_id for r in out[2].tool_results] == ids
    if case == "empty_assistant":
        assert [m.role for m in out] == ["user", "assistant", "tool"]


def test_flatten_writes_tool_turns_as_text():
    call, res = _pair("x", text="the tree")
    flat = flatten([Message.user("hi"), call, res, Message.user("more")])
    assert all(not m.tool_calls and not m.tool_results for m in flat)
    assert "[I called: ax_tree({})]" in flat[1].text and "the tree" in flat[2].text


# ---- (b) Gemini switching model or key halfway through a run ----------------------------------------


def _err(code: int, message: str = "x", details: list | None = None):
    from google.genai import errors as gerrors

    return gerrors.APIError(
        code, {"error": {"code": code, "message": message, "status": "S", "details": details or []}}
    )


def _per_day():
    return [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
         "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "43s"},
    ]  # fmt: skip


def _per_minute(delay="20s"):
    return [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
         "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay},
    ]  # fmt: skip


@pytest.fixture
def gem(monkeypatch):
    """A Gemini provider with two keys and fake clients: `script(model, key) -> response | error`."""
    g = pytest.importorskip("neo.providers.gemini")
    from neo import config

    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setenv("GEMINI_API_KEY", "k1")
    monkeypatch.setenv("GEMINI_API_KEYS", "k2")
    monkeypatch.setattr(g, "_BACKOFF_S", (0.0,))
    monkeypatch.setattr(g, "_SLOW_FAIL_S", -1.0)  # every 5xx counts as real overload

    async def quiet(text):
        pass

    monkeypatch.setattr(g, "_note", quiet)
    p = g.GeminiProvider(model="gemini-3.8-flash")
    sent: list[tuple[str, int, list]] = []
    rules: dict = {}

    def client(ki):
        async def generate_content(*, model, contents, config):
            sent.append((model, ki, contents))
            r = rules.get((model, ki), rules.get(model, rules.get("*")))
            r = r(len(sent)) if callable(r) else r
            if isinstance(r, Exception):
                raise r
            return r

        return SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))

    p._clients = [client(0), client(1)]
    return SimpleNamespace(g=g, p=p, sent=sent, rules=rules)


def _reply(*, call: str = "", sig: bytes | None = None, text: str = ""):
    from google.genai import types as gt

    parts = []
    if call:
        parts.append(gt.Part(function_call=gt.FunctionCall(name=call, args={}), thought_signature=sig))
    if text:
        parts.append(gt.Part.from_text(text=text))
    return gt.GenerateContentResponse(
        candidates=[gt.Candidate(content=gt.Content(role="model", parts=parts))]
    )


def test_a_mid_run_model_switch_sends_a_valid_history_and_only_that_models_signatures(gem):
    g, p = gem.g, gem.p
    gem.rules["gemini-3.8-flash"] = _reply(call="ax_tree", sig=b"sig-3.8")
    msgs = [Message.user("play chess as white")]
    t1 = asyncio.run(p.complete(msgs))
    assert t1.tool_calls[0].source == "gemini-3.8-flash"
    msgs += [
        Message.assistant("", t1.tool_calls),
        Message.tool([ToolResult(t1.tool_calls[0].id, "ax_tree", "tree")]),
    ]

    # Next step: 3.8 overloaded, 3.7 key 1 out of quota for the day, 3.7 key 2 answers.
    gem.rules["gemini-3.8-flash"] = _err(503)
    gem.rules[("gemini-3.7-flash", 0)] = _err(429, "quota", _per_day())
    gem.rules[("gemini-3.7-flash", 1)] = _reply(text="The board is open.")
    gem.sent.clear()
    t2 = asyncio.run(p.complete(msgs))
    assert t2.text == "The board is open."
    assert [(m, k) for m, k, _ in gem.sent] == [
        ("gemini-3.8-flash", 0),
        ("gemini-3.7-flash", 0),
        ("gemini-3.7-flash", 1),
    ]
    for model, _k, contents in gem.sent:
        assert not gemini_violations(contents)
        sig = contents[1].parts[0].thought_signature
        # 3.8's own signature goes back to 3.8; any other model gets the documented bypass.
        assert sig == (b"sig-3.8" if model == "gemini-3.8-flash" else g._SKIP_SIGNATURE)


def test_quota_is_kept_for_its_whole_window(gem):
    g, p = gem.g, gem.p
    gem.rules[("gemini-3.8-flash", 0)] = _err(429, "quota", _per_day())
    gem.rules[("gemini-3.8-flash", 1)] = _err(429, "quota", _per_minute("20s"))
    gem.rules["*"] = _reply(text="ok")
    now = time.time()
    asyncio.run(p.complete([Message.user("hi")]))
    day = p._exhausted[("gemini-3.8-flash", 0)] - now
    assert abs(day - g._to_daily_reset(now)) < 5 and 60 < day <= 25 * 3600  # to midnight Pacific
    assert 20 <= p._exhausted[("gemini-3.8-flash", 1)] - now <= 25  # the server's own delay


def test_every_key_spent_fails_at_once_without_a_round_trip(gem):
    p = gem.p
    gem.rules["*"] = _err(429, "quota", _per_day())
    with pytest.raises(QuotaExhausted) as e1:
        asyncio.run(p.complete([Message.user("hi")]))
    assert len(gem.sent) == 6  # 3 models × 2 keys learned once…
    assert p.cooling_until() > time.time() + 60 and e1.value.until == pytest.approx(p.cooling_until())
    gem.sent.clear()
    with pytest.raises(QuotaExhausted):
        asyncio.run(p.complete([Message.user("the next step")]))
    assert gem.sent == []  # …and not asked again on every step
    assert "429" in str(e1.value)  # (neo.reflex.generate rotates on that)


def test_a_history_gemini_still_rejects_is_retried_once_as_text(gem):
    shape = _err(
        400, "Please ensure that function response turn comes immediately after a function call turn."
    )
    gem.rules["*"] = lambda n: shape if n == 1 else _reply(text="done")
    call, res = _pair("x")
    t = asyncio.run(gem.p.complete([Message.user("hi"), call, res]))
    assert t.text == "done" and len(gem.sent) == 2
    retry = gem.sent[1][2]
    assert not any(pt.function_call or pt.function_response for c in retry for pt in c.parts)
    # Any other 400 is the request's own fault: straight to the next brain.
    gem.sent.clear()
    gem.rules["*"] = _err(400, "Invalid value at 'tools'")
    with pytest.raises(ProviderError) as e:
        asyncio.run(gem.p.complete([Message.user("hi")]))
    assert e.value.kind == "bad_request" and len(gem.sent) == 1


# ---- (2) a fallback brain with little room: fit the request, don't 413 -------------------------------


def _long_run(steps: int = 10, approvals: int = 6) -> list[Message]:
    """The chess run as the follow-up after the last approval sends it: big UI trees and all."""
    msgs = [Message.user("play chess as white")]
    for i in range(steps):
        name = "ax_tree" if i % 2 else "zz_script"
        msgs += [
            Message.assistant(
                "", [ToolCall(f"s{i}", name, {"script": 'tell application "System Events" ' * 8})]
            ),
            Message.tool(
                [
                    ToolResult(
                        f"s{i}",
                        name,
                        f"step {i} " + "AXButton “Board” " * 400,
                        images=[ImagePart(b"png" * 100)] if i == steps - 2 else [],
                    )
                ]
            ),  # fmt: skip
        ]
    for k in range(approvals):
        msgs.append(Message.user(f"[User confirmed. zz_script result: window 1] Continue {k}."))
        msgs += [
            Message.assistant("", [ToolCall(f"f{k}", "zz_script", {"script": "get name of windows"})]),
            Message.tool(
                [ToolResult(f"f{k}", "zz_script", "NEEDS_CONFIRM: run an AppleScript — Should I go ahead?")]
            ),
        ]
    msgs.append(
        Message.user("[User confirmed. zz_script result: window 1, window 2] Continue and finish the task.")
    )
    return msgs


def test_the_follow_up_request_fits_groqs_budget_keeping_the_latest_steps(monkeypatch):
    from neo.providers.fit import estimate, fit

    msgs = _long_run()
    tools = registry().specs()  # the approval follow-up offers every tool
    system = "You are NEO. " * 200
    assert estimate(msgs, system, tools, vision=False) + 4096 > 8000  # what drew the 413
    f = fit(msgs, system=system, tools=tools, budget=8000, max_tokens=4096, vision=False, name="groq")
    assert estimate(f.messages, system, f.tools, vision=False) + f.max_tokens <= 8000 * 0.9
    assert not wire_violations(f.messages)
    results = [r for m in f.messages for r in m.tool_results]
    assert results[-1].content == msgs[-2].tool_results[0].content  # the newest step is whole
    assert all(not r.images for r in results)  # a text-only brain gets no pictures
    assert results[0].content != msgs[2].tool_results[0].content  # older observations are cut
    used = {"zz_script", "ax_tree"}
    assert {s.name for s in f.tools} <= used | {"more_tools"} and len(f.tools) < len(tools)
    assert f.max_tokens < 4096 and f.messages[0].text == "play chess as white"
    # The caller's history is untouched.
    assert (
        msgs[2].tool_results[0].content.startswith("step 0 ") and len(msgs[2].tool_results[0].content) > 4000
    )


def test_a_request_that_cannot_fit_fails_as_too_large():
    from neo.providers.fit import fit

    with pytest.raises(RequestTooLarge):
        fit(
            [Message.user("x " * 4000)],
            system="s " * 4000,
            budget=2000,
            max_tokens=4096,
            vision=False,
            name="groq",
        )


def test_groqs_413_teaches_the_real_size_and_the_request_is_fitted_again(monkeypatch):
    import httpx
    import openai

    from neo.providers.openai_compat import OpenAICompatProvider

    p = OpenAICompatProvider(name="groq", base_url="https://api.groq.test/v1", api_key="k", model="m",
                             reasoning_effort=True, token_budget=8000)  # fmt: skip
    calls: list[dict] = []

    async def create(**kw):
        calls.append(kw)
        if len(calls) == 1:
            req = httpx.Request("POST", "https://api.groq.test/v1/chat/completions")
            raise openai.APIStatusError(
                "Error code: 413 - Request too large for model `m` on tokens per minute (TPM): Limit 6000, Requested 9500",
                response=httpx.Response(413, request=req),
                body=None,
            )
        msg = SimpleNamespace(content="Your move.", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)

    monkeypatch.setattr(p._client.chat.completions, "create", create)
    msgs = _long_run(steps=4, approvals=1)
    turn = asyncio.run(p.complete(msgs, system="sys", tools=registry().specs()[:6], effort="high"))
    assert turn.text == "Your move." and len(calls) == 2
    assert p.token_budget == 6000 and p._scale > 1.0
    size = [len(json.dumps(c["messages"])) + len(json.dumps(c.get("tools") or [])) for c in calls]
    assert size[1] < size[0] and calls[1]["reasoning_effort"] == "low"  # less room: think briefly


# ---- (3) quota everywhere: next brain at once, and a clear reason at the end -------------------------


class FakeBrain:
    supports_vision = False
    supports_tools = True

    def __init__(self, name, error=None, cooling=0.0, answer="ok"):
        self.name, self.model, self.error, self.cooling, self.answer = name, name, error, cooling, answer
        self.asked = 0

    def cooling_until(self):
        return self.cooling

    async def complete(self, messages, **kw):
        self.asked += 1
        if self.error is not None:
            raise self.error
        return Turn(text=self.answer)

    async def stream(self, messages, **kw):
        yield self.answer


@pytest.fixture
def chain(monkeypatch):
    import neo.providers as prov

    monkeypatch.setattr(prov, "_demoted", {})

    def use(**brains):
        monkeypatch.setattr(prov, "make", lambda n: brains[n])
        return prov.Chain(list(brains))

    return use


def test_a_spent_brain_is_asked_last_and_the_next_one_at_once(chain):
    until = time.time() + 3600
    gemini = FakeBrain("gemini", QuotaExhausted("429 out", until=until), cooling=until)
    groq = FakeBrain("groq", answer="from groq")
    c = chain(gemini=gemini, groq=groq)
    assert asyncio.run(c.complete([Message.user("hi")])).text == "from groq"
    assert (gemini.asked, groq.asked) == (0, 1)


def test_a_too_large_request_does_not_bench_the_brain(chain):
    import neo.providers as prov

    groq = FakeBrain("groq", RequestTooLarge("413"))
    local = FakeBrain("local", answer="local")
    asyncio.run(chain(groq=groq, local=local).complete([Message.user("hi")]))
    assert "groq" not in prov._demoted  # the next, smaller request goes to Groq first again


def test_a_short_wait_is_waited_out_when_every_brain_failed(chain):
    groq = FakeBrain("groq", RateLimited("429 try again in 0.01s", retry_after=0.01))
    local = FakeBrain("local", Unavailable("local model server not running"))
    c = chain(groq=groq, local=local)

    async def later():
        await asyncio.sleep(0)
        groq.error = None

    async def go():
        asyncio.get_running_loop().create_task(later())
        return await c.complete([Message.user("hi")])

    assert asyncio.run(go()).text == "ok" and groq.asked == 2


def test_when_every_brain_fails_the_task_ends_saying_which_and_why(chain):
    from neo.agent.loop import run_agent

    until = time.time() + 3 * 3600
    c = chain(
        gemini=FakeBrain("gemini", QuotaExhausted("429 out", until=until), cooling=until),
        groq=FakeBrain("groq", RequestTooLarge("groq: 413 request too large")),
        local=FakeBrain("local", Unavailable("local model server not running")),
    )

    async def go():
        from neo import config

        config.settings().groq_api_key = "k"  # a brain was set up: this isn't the no-key message
        return await run_agent("play chess as white", provider=c, system="", tools=[], max_steps=2)

    res = asyncio.run(go())
    assert res.stopped == "error"
    t = time.localtime(until)
    clock = f"{t.tm_hour % 12 or 12}:{t.tm_min:02d}"
    assert res.text.startswith("I had to stop: ")
    assert f"Gemini is out of free quota until {clock}" in res.text
    assert (
        "Groq couldn't take a request this long" in res.text and "the local model isn't running" in res.text
    )
    assert "problem talking to the model" not in res.text


def test_all_brains_failed_names_brains_that_are_not_set_up(chain, monkeypatch):
    import neo.providers as prov
    from neo import config

    def make(n):
        if n == "claude":
            raise Unavailable("ANTHROPIC_API_KEY is not set")
        return FakeBrain(n, ProviderError(f"{n} down"))

    monkeypatch.setattr(prov, "make", make)
    config.settings().gemini_api_key = "k"
    with pytest.raises(AllBrainsFailed) as e:
        asyncio.run(prov.Chain(["gemini", "claude"]).complete([Message.user("hi")]))
    assert [n for n, _ in e.value.failures] == ["gemini", "claude"]
    assert "Claude isn't available (ANTHROPIC_API_KEY is not set)" in e.value.spoken()


def test_tool_specs_given_to_fit_survive_when_nothing_was_used_yet():
    from neo.providers.fit import fit

    params = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "What it's called. " * 10},
            "description": {"type": "string", "description": "A parameter named description."},
        },
        "required": ["title", "description"],
    }
    tools = [ToolSpec(f"t{i}", "Does a thing. " + "More words. " * 40, params) for i in range(30)]
    f = fit([Message.user("hi")], tools=tools, budget=4000, max_tokens=512, vision=False)
    assert len(f.tools) == 30 and all(len(s.description) < 40 for s in f.tools)  # shortened, not dropped
    props = f.tools[0].parameters["properties"]
    assert props == {"title": {"type": "string"}, "description": {"type": "string"}}  # still valid


# ---- the whole run again, on the real chain: Gemini spent, Groq's 8K, no local model ---------------


def test_the_chess_run_survives_on_groq_when_gemini_is_spent(monkeypatch):
    import httpx
    import openai

    import neo.providers as prov
    from neo import config
    from neo.providers import local_server

    g = pytest.importorskip("neo.providers.gemini")
    monkeypatch.setattr(config, "_settings", None)
    for k, v in (("GEMINI_API_KEY", "k1"), ("GEMINI_API_KEYS", "k2"), ("GROQ_API_KEY", "g1")):
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(prov, "_cache", {})
    monkeypatch.setattr(prov, "_demoted", {})

    async def down(port=8080):
        return False

    async def quiet(text):
        pass

    monkeypatch.setattr(local_server, "is_up", down)
    monkeypatch.setattr(g, "_note", quiet)

    gemini_asked: list[str] = []

    def spent(ki):
        async def generate_content(*, model, contents, config):
            gemini_asked.append(model)
            raise _err(429, "quota", _per_day())

        return SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))

    gem_p = prov.make("gemini")
    gem_p._clients = [spent(0), spent(1)]

    groq_p = prov.make("groq")
    sizes: list[int] = []
    k = {"n": 0}

    async def create(*, model, messages, max_tokens, tools=None, **kw):
        # What Groq counts: the prompt as it tokenizes (~4 chars a token here) plus the answer's room.
        size = (len(json.dumps(messages)) + len(json.dumps(tools or []))) // 4 + max_tokens
        sizes.append(size)
        if size > 8000:
            req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
            raise openai.APIStatusError(
                f"Error code: 413 - Request too large for model `{model}` on tokens per minute (TPM): "
                f"Limit 8000, Requested {size}",
                response=httpx.Response(413, request=req),
                body=None,
            )
        k["n"] += 1
        goal = max(i for i, m in enumerate(messages) if str(m.get("content") or "").startswith("play chess"))
        done = [c["function"]["name"] for m in messages[goal:] for c in m.get("tool_calls") or []]
        plan = ["open_app", "ax_tree", "hotkey", "ax_tree"]
        name = plan[len(done)] if len(done) < len(plan) else "zz_script"
        call = SimpleNamespace(
            id=f"g{k['n']}",
            function=SimpleNamespace(name=name, arguments=json.dumps({"script": f"get {k['n']}"})),
        )
        msg = SimpleNamespace(content="", tool_calls=[call])
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)

    monkeypatch.setattr(groq_p._client.chat.completions, "create", create)
    ran: list[str] = []
    _chess_tools(monkeypatch, ran)
    monkeypatch.setitem(
        registry()._tools, "ax_tree",
        RegisteredTool("ax_tree", "test tool", _OBJ, lambda a, c: "AXWindow “Chess” " * 400, parallel_safe=True),
    )  # fmt: skip
    monkeypatch.setattr(sess_mod.fastpath, "plan", lambda text: None)

    async def go():
        s = Session(reflex=_agent_reflex())
        _earlier_runs(s)
        r = await s.handle("play chess as white", from_user=True)
        for _ in range(8):
            assert r.route == "confirm", r.text
            r = await s.handle("yes", from_user=True)
        return r

    r = asyncio.run(go())
    assert r.route == "confirm" and len(ran) == 8
    # Gemini: each (model, key) learned spent once, then never asked again on a later step.
    assert len(gemini_asked) == 6
    # Groq: every request it answered fitted its 8K (a 413, if any, was learned from and refitted).
    assert k["n"] >= 12 and all(sz <= 8000 for sz in sizes[-5:])
