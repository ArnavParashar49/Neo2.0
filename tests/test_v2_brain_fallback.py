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

import neo2.core.agent.session as sess_mod
from neo2.core.agent import confirm
from neo2.core.agent.registry import RegisteredTool, registry
from neo2.core.agent.session import Session
from neo2.core.events import EventBus
from neo2.core.providers.base import (
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
from neo2.core.providers.history import NO_RESULT, flatten, normalize
from neo2.core.reflex.schema import Decision

_OBJ = {"type": "object", "properties": {}}


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    from neo2.core import config, events
    from neo2.core.tools import load_all

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
    from neo2.core.providers.claude import _to_messages as claude_msgs
    from neo2.core.providers.gemini import _to_contents
    from neo2.core.providers.openai_compat import _to_messages as openai_msgs

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
    monkeypatch.setattr(sess_mod.fastpath, "plan", lambda text, **_: None)  # the model plays every step

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
    c = __import__("neo2.core.providers.gemini", fromlist=["_to_contents"])._to_contents(hist, "gemini-3.8-flash")
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
    g = pytest.importorskip("neo2.core.providers.gemini")
    from neo2.core import config

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
    day = p._until("gemini-3.8-flash", 0) - now
    assert abs(day - g._to_daily_reset(now)) < 5 and 60 < day <= 25 * 3600  # to midnight Pacific
    assert 20 <= p._until("gemini-3.8-flash", 1) - now <= 25  # the server's own delay


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
    assert "429" in str(e1.value)  # (neo2.core.reflex.generate rotates on that)


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
    from neo2.core.providers.fit import estimate, fit

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
    # The tools it used stay; of the rest, only as many as the room left takes (in the order given).
    names = [s.name for s in f.tools]
    assert "ax_tree" in names and len(f.tools) < len(tools)
    assert names == [s.name for s in tools if s.name in names]
    assert f.max_tokens < 4096 and f.messages[0].text == "play chess as white"
    # The caller's history is untouched.
    assert (
        msgs[2].tool_results[0].content.startswith("step 0 ") and len(msgs[2].tool_results[0].content) > 4000
    )


def test_a_request_that_cannot_fit_fails_as_too_large():
    from neo2.core.providers.fit import fit

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

    from neo2.core.providers.openai_compat import OpenAICompatProvider

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
    import neo2.core.providers as prov

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
    import neo2.core.providers as prov

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
    from neo2.core.agent.loop import run_agent

    until = time.time() + 3 * 3600
    c = chain(
        gemini=FakeBrain("gemini", QuotaExhausted("429 out", until=until), cooling=until),
        groq=FakeBrain("groq", RequestTooLarge("groq: 413 request too large")),
        local=FakeBrain("local", Unavailable("local model server not running")),
    )

    async def go():
        from neo2.core import config

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
    import neo2.core.providers as prov
    from neo2.core import config

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
    from neo2.core.providers.fit import fit

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

    import neo2.core.providers as prov
    from neo2.core import config
    from neo2.core.providers import local_server

    g = pytest.importorskip("neo2.core.providers.gemini")
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

    # A game on this Mac to play (the route a real one takes: a screen task, the vision chain), not
    # whatever an earlier scan left behind. Flash-Lite, one of that chain's own brains, is spent too.
    # The run's steps carry no picture, so Groq's text model carries it: Groq's vision model, a
    # fallback only for a step with a picture, is never asked one (its 8K a minute stays for those).
    from neo2.core.decide.inventory import App

    game = App(bundle_id="test.boards.game", name="Board Game", path="/Applications/Board Game.app",
               category="board-games")  # fmt: skip
    monkeypatch.setattr(sess_mod, "_game_to_play", lambda text: game)
    lite_asked: list[int] = []
    vision_asked: list[int] = []

    async def lite_spent(*, model, contents, config):
        lite_asked.append(1)
        raise _err(429, "quota", _per_day())

    prov.make("gemini_lite")._clients = [
        SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=lite_spent)))
    ]

    async def vision_create(**kw):
        vision_asked.append(1)
        req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        raise openai.RateLimitError("rate limited", response=httpx.Response(429, request=req), body=None)

    monkeypatch.setattr(prov.make("groq_vision")._client.chat.completions, "create", vision_create)

    groq_p = prov.make("groq")
    sizes: list[int] = []
    offered: list[set[str]] = []
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
        offered.append({t["function"]["name"] for t in tools or []})
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
    monkeypatch.setattr(sess_mod.fastpath, "plan", lambda text, **_: None)

    first_run: list[int] = []

    async def go():
        s = Session(reflex=_agent_reflex())
        _earlier_runs(s)
        r = await s.handle("play chess as white", from_user=True)
        first_run.append(len(offered))
        for _ in range(8):
            assert r.route == "confirm", r.text
            r = await s.handle("yes", from_user=True)
        return r

    r = asyncio.run(go())
    assert r.route == "confirm" and len(ran) == 8
    # Gemini: each (model, key) learned spent once, then never asked again on a later step.
    assert len(gemini_asked) == 6
    assert len(lite_asked) == 1  # Flash-Lite: learned spent once, like Gemini
    assert vision_asked == []  # no step of the run carried a picture
    # Groq: every request it answered fitted its 8K (a 413, if any, was learned from and refitted).
    assert k["n"] >= 12 and all(sz <= 8000 for sz in sizes[-5:])
    # …and every one could ask for the tools it wasn't shown; every follow-up after a yes (offered
    # all of NEO's tools) could still act on the screen, not only script it.
    assert all("more_tools" in names for names in offered)
    assert all({"click", "type_text"} <= names for names in offered[first_run[0] :])


# ---- review: tools a fitted request keeps, and more_tools to get the rest back ----------------------


def _chess_follow_up(tree_chars: int = 6000, approvals: int = 1) -> list[Message]:
    """open Chess, new game, read it (a full tree), a staged AppleScript, the user's yes."""
    msgs = [
        Message.user("play chess as white"),
        *_pair("a", "open_app", "Opened Chess."),
        *_pair("b", "hotkey", "Pressed cmd+n."),
        *_pair("c", "ax_tree", ("AXButton “Board” " * 400)[:tree_chars]),
    ]
    for k in range(approvals):
        msgs += [
            Message.assistant("", [ToolCall(f"d{k}", "applescript", {"script": "get name of windows"})]),
            Message.tool(
                [ToolResult(f"d{k}", "applescript", "NEEDS_CONFIRM: run an AppleScript — Should I go ahead?")]
            ),
            Message.user(
                f"[User confirmed. applescript result: Chess, New Game {k}] Continue and finish the task."
            ),
        ]
    return msgs


def test_a_fitted_request_keeps_more_tools_and_fills_its_room_with_the_tools_it_dropped():
    from neo2.core.agent.prompt import build
    from neo2.core.agent.toolselect import MORE_TOOLS
    from neo2.core.providers.fit import estimate, fit

    msgs = _chess_follow_up()
    tools = [*registry().specs(), MORE_TOOLS]  # the approval follow-up: every tool, and more_tools
    system = build("")
    f = fit(msgs, system=system, tools=tools, budget=8000, max_tokens=4096, vision=False, name="groq")
    names = [s.name for s in f.tools]
    limit = int(8000 * 0.9)
    total = estimate(f.messages, system, f.tools, vision=False) + f.max_tokens
    assert total <= limit
    assert "more_tools" in names and {"open_app", "hotkey", "ax_tree", "applescript"} <= set(names)
    # Room left after the cut goes back to the tools that were dropped (the observe/act ones first,
    # the order they were given in): the model can click and type, not only script.
    assert {"click", "type_text", "ax_press"} <= set(names)
    dropped = [s for s in tools if s.name not in names]
    assert not dropped or limit - total < min(estimate([], "", [s], vision=False) for s in dropped) + 20
    assert names == [s.name for s in tools if s.name in names]  # in the order given
    tree = [r.content for m in f.messages for r in m.tool_results if r.name == "ax_tree"]
    assert tree == [msgs[6].tool_results[0].content]  # the tree, whole


def test_older_conversation_goes_before_the_tools_the_run_has_not_used():
    from neo2.core.providers.fit import fit

    earlier = []
    for j in range(3):
        earlier += [Message.user(f"earlier request {j} " + "words " * 300), Message.assistant("Done. " * 200)]
    msgs = earlier + _chess_follow_up(tree_chars=2000)
    tools = [ToolSpec(f"t{i}", "Does a thing.", _OBJ) for i in range(30)]
    tools += [ToolSpec(n, "Used.", _OBJ) for n in ("open_app", "hotkey", "ax_tree", "applescript")]
    f = fit(msgs, system="sys", tools=tools, budget=3000, max_tokens=512, vision=False, name="groq")
    assert f.messages[0].text == "play chess as white"  # earlier requests went…
    assert len(f.tools) == len(tools) and not any("tools" in t for t in f.trimmed)  # …every tool stayed


def test_essential_tools_are_never_dropped():
    from neo2.core.providers.fit import fit

    fence = [ToolSpec(f"b{i}", "A browser tool. " * 20, _OBJ, essential=True) for i in range(12)]
    f = fit(_chess_follow_up(), system="sys", tools=fence, budget=4000, max_tokens=512, vision=False)
    assert [s.name for s in f.tools] == [s.name for s in fence]
    with pytest.raises(RequestTooLarge):  # a fence that can't fit is refused, not quietly thinned
        fit(_chess_follow_up(), system="sys", tools=fence * 8, budget=2000, max_tokens=512, vision=False)


class ShownBrain:
    """Answers with what a fitted brain reports: it was shown only some of the tools."""

    name = "fake"
    supports_vision = False
    supports_tools = True

    def __init__(self, turns, hide=()):
        self.turns, self.hide, self.seen = list(turns), set(hide), []

    async def complete(self, messages, *, tools=None, **kw):
        self.seen.append([t.name for t in tools or []])
        t = self.turns.pop(0)
        t.offered = [n for n in self.seen[-1] if n not in self.hide]
        return t

    async def stream(self, messages, **kw):
        yield ""


def test_more_tools_is_offered_in_a_follow_up_and_gives_back_what_fitting_dropped():
    from neo2.core.agent.loop import run_agent

    b = ShownBrain(
        [Turn(tool_calls=[ToolCall("m1", "more_tools", {"query": "click"})]), Turn(text="Your move.")],
        hide={"click"},
    )
    res = asyncio.run(
        run_agent(
            "[User confirmed. applescript result: ok] Continue.",
            provider=b,
            system="",
            ev=EventBus(),
            verify=False,
        )
    )
    assert res.stopped == "done"
    assert "more_tools" in b.seen[0]  # the approval follow-up (tools=None) can ask for more…
    said = res.messages[2].tool_results[0].content
    assert said.startswith("Enabled: ") and "click" in said  # …and gets back what the brain wasn't shown
    assert b.seen[1].count("click") == 1


def test_a_strict_run_marks_its_whole_fence_essential():
    from neo2.core.agent.loop import run_agent

    fence = [ToolSpec("browser_read", "Read.", _OBJ), ToolSpec("browser_click", "Click.", _OBJ)]
    got = {}

    class B(ShownBrain):
        async def complete(self, messages, *, tools=None, **kw):
            got["tools"] = tools
            return Turn(text="done")

    asyncio.run(
        run_agent("x", provider=B([]), system="", tools=fence, ev=EventBus(), strict=True, verify=False)
    )
    assert [t.name for t in got["tools"]] == ["browser_read", "browser_click"]
    assert all(t.essential for t in got["tools"]) and not fence[0].essential  # (the caller's specs untouched)


# ---- review: the lite brain is its own brain --------------------------------------------------------


def test_the_lite_brain_is_demoted_under_its_own_name(monkeypatch):
    import neo2.core.providers as prov
    from neo2.core import config

    g = pytest.importorskip("neo2.core.providers.gemini")
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setenv("GEMINI_API_KEY", "k1")
    monkeypatch.setattr(prov, "_cache", {})
    monkeypatch.setattr(prov, "_demoted", {})

    async def quiet(text):
        pass

    monkeypatch.setattr(g, "_note", quiet)

    async def spent(*, model, contents, config):
        raise _err(429, "quota", _per_day())

    lite = prov.make("gemini_lite")
    lite._clients = [SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=spent)))]
    groq = FakeBrain("groq", answer="from groq")
    monkeypatch.setattr(prov, "make", lambda n: {"gemini_lite": lite, "groq": groq}[n])
    assert asyncio.run(prov.Chain(["gemini_lite", "groq"]).complete([Message.user("hi")])).text == "from groq"
    assert lite.name == "gemini_lite" and set(prov._demoted) == {"gemini_lite"}
    assert prov.brain("agent")._order()[0] == "gemini"  # the main brain still goes first
    from neo2.core.providers.base import failure_words

    assert failure_words(lite.name, QuotaExhausted("x")).startswith("Gemini Lite ")


# ---- review: Gemini quota and bad keys follow the key, not its place in the list ------------------


def test_quota_follows_the_key_not_its_place_in_the_list(gem, monkeypatch):
    from neo2.core import config

    g = gem.g
    gem.rules["*"] = _err(429, "quota", _per_day())
    with pytest.raises(QuotaExhausted):
        asyncio.run(gem.p.complete([Message.user("hi")]))

    def fresh(keys: list[str]):
        monkeypatch.setattr(config, "_settings", None)
        monkeypatch.setenv("GEMINI_API_KEY", keys[0])
        monkeypatch.setenv("GEMINI_API_KEYS", ",".join(keys[1:]))
        p = g.GeminiProvider(model="gemini-3.8-flash")  # a restart after the .env changed
        asked: list[int] = []

        def client(ki):
            async def generate_content(*, model, contents, config):
                asked.append(ki)
                return _reply(text="ok")

            return SimpleNamespace(
                aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
            )

        p._clients = [client(i) for i in range(len(keys))]
        return p, asked

    p, asked = fresh(["NEW-fresh-key"])  # the spent key replaced: the new one is asked at once
    assert asyncio.run(p.complete([Message.user("hi")])).text == "ok" and asked == [0]
    p, asked = fresh(["NEW-fresh-key", "k2"])  # a spent key moved to another place is still spent…
    assert p._blocked("gemini-3.8-flash", 1, time.time()) and not p._blocked(
        "gemini-3.8-flash", 0, time.time()
    )


def test_a_key_gemini_calls_invalid_is_skipped_for_the_next(gem):
    from neo2.core.providers.base import failure_words

    bad = _err(
        400,
        "API key not valid. Please pass a valid API key.",
        [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID"}],
    )
    gem.rules[("gemini-3.8-flash", 0)] = bad
    gem.rules["*"] = _reply(text="ok")
    for _ in range(3):
        assert asyncio.run(gem.p.complete([Message.user("hi")])).text == "ok"
    assert [k for _m, k, _c in gem.sent] == [0, 1, 1, 1]  # the revoked key is asked once, not every step

    gem.sent.clear()
    gem.p._bad_keys.clear()
    gem.rules.clear()
    gem.rules["*"] = _err(403, "Permission denied: consumer has been suspended.")
    with pytest.raises(Unavailable) as e:
        asyncio.run(gem.p.complete([Message.user("hi")]))
    assert len(gem.sent) == 2  # each key once, then no model is asked with a key that can't work
    assert "API key" in failure_words("gemini", e.value)


# ---- review: the request stays in memory through a long approval chain ------------------------------


def test_the_memory_cap_keeps_the_request_a_confirmation_note_continues(monkeypatch):
    async def no_summary(msgs, started, budget_s=8.0):
        return ""

    monkeypatch.setattr(Session, "_summarize", staticmethod(no_summary))

    async def go():
        s = Session(reflex=_agent_reflex())
        s.history.append(Message.user("play chess as white"))
        trims = 0
        for k in range(1, 41):  # 40 approvals of 11 messages: the cap trims several times
            before = len(s.history)
            turns = [
                Message.user(f"[User confirmed. applescript result: moved {k}] Continue and finish the task.")
            ]
            for j in range(5):
                turns += [
                    Message.assistant("", [ToolCall(f"c{k}.{j}", "ax_tree", {})]),
                    Message.tool([ToolResult(f"c{k}.{j}", "ax_tree", "board")]),
                ]
            await s._remember(*turns)
            trims += len(s.history) < before
            assert s.history[0].text == "play chess as white", k
            assert s._trimmed()[0].text == "play chess as white"
            assert not wire_violations(s._trimmed())
        return s, trims

    s, trims = asyncio.run(go())
    assert trims >= 3 and len(s.history) <= sess_mod._HISTORY_CAP


# ---- review: consecutive assistant turns ---------------------------------------------------------------


def test_normalize_merges_consecutive_assistant_turns():
    call, res = _pair("x")
    msgs = [Message.user("hi"), Message.tool([ToolResult("o", "ax_tree", "orphan")]), Message.assistant("ok"),
            Message.tool([ToolResult("o2", "ax_tree", "orphan")]),
            Message.assistant("looking", [ToolCall("x", "ax_tree", {})]), res]  # fmt: skip
    out = normalize(msgs)
    assert not wire_violations(msgs)
    assert [m.role for m in out] == ["user", "assistant", "tool"]
    assert out[1].text == "ok\n\nlooking" and [c.id for c in out[1].tool_calls] == ["x"]
    assert normalize(out) == out and msgs[2].text == "ok"


def test_normalize_keeps_every_random_history_valid():
    import random

    rng = random.Random(7)
    for _ in range(2000):
        msgs: list[Message] = []
        for n in range(1, rng.randint(2, 13)):
            r = rng.random()
            if r < 0.25:
                msgs.append(Message.user(f"u{n}"))
            elif r < 0.5:
                calls = [ToolCall(f"c{n}.{j}", "ax_tree", {}) for j in range(rng.randint(0, 2))]
                msgs.append(Message.assistant(rng.choice(["", "t"]), calls))
            else:
                ids = [f"c{rng.randint(1, n)}.{rng.randint(0, 1)}" for _ in range(rng.randint(1, 2))]
                msgs.append(Message.tool([ToolResult(i, "ax_tree", "r") for i in ids]))
        assert not wire_violations(msgs), msgs
        assert normalize(normalize(msgs)) == normalize(msgs)


# ---- review: the latest observation survives approvals --------------------------------------------------


def test_the_only_observation_of_a_run_is_kept_whole_through_approvals():
    s = Session(reflex=_agent_reflex())
    board = "AXButton “e2” " * 180
    s.history = [
        Message.user("an earlier request"),
        *_pair("e", "ax_tree", "old tree " * 100),
        Message.assistant("Done."),
        Message.user("play chess as white"),
        *_pair("t", "ax_tree", board),
    ]
    for k in range(3):  # three approvals, each note a user turn
        s.history += [
            Message.assistant("", [ToolCall(f"s{k}", "applescript", {"script": "x"})]),
            Message.tool(
                [ToolResult(f"s{k}", "applescript", "NEEDS_CONFIRM: run an AppleScript — go ahead?")]
            ),
            Message.user(f"[User confirmed. applescript result: ok {k}] Continue and finish the task."),
        ]
    h = s._trimmed()
    trees = [r.content for m in h for r in m.tool_results if r.name == "ax_tree"]
    assert trees[-1] == board  # the board the model is playing on, whole
    assert trees[0] == "old tree " * 100  # the earlier request is one of the last two: whole

    # A newer look at the same thing supersedes the older one — in the copy the model gets only.
    s.history += list(_pair("t2", "ax_tree", board + " moved"))
    h = s._trimmed()
    trees = [r.content for m in h for r in m.tool_results if r.name == "ax_tree"]
    assert trees[-1] == board + " moved" and len(trees[-2]) < 300 and trees[0] == "old tree " * 100
    assert s.history[6].tool_results[0].content == board  # the session's own history is untouched


def test_fit_spares_the_newest_observation_behind_the_halted_steps():
    from neo2.core.providers.fit import fit

    msgs = _chess_follow_up(tree_chars=6000, approvals=3)
    msgs = [Message.user("before " * 10), *_pair("z", "ax_tree", "stale tree " * 500)] + msgs
    f = fit(msgs, system="sys", tools=None, budget=6000, max_tokens=2048, vision=False, name="groq")
    trees = [r.content for m in f.messages for r in m.tool_results if r.name == "ax_tree"]
    assert trees[-1] == ("AXButton “Board” " * 400)[:6000]  # the newest tree, whole
    assert len(trees[0]) < 2000  # the stale one from before went first


# ---- review: a learned size can come back down ----------------------------------------------------------


def test_a_learned_size_is_capped_and_comes_back_down(monkeypatch):
    import httpx
    import openai

    from neo2.core.providers import openai_compat as oc

    p = oc.OpenAICompatProvider(name="local", base_url="https://api.local.test/v1", api_key="k", model="m",
                                token_budget=8000)  # fmt: skip
    state = {"refuse": 0, "prompt": None}

    async def create(**kw):
        if state["refuse"]:
            state["refuse"] -= 1
            req = httpx.Request("POST", "https://api.local.test/v1/chat/completions")
            raise openai.APIStatusError(
                "context_length_exceeded", response=httpx.Response(400, request=req), body=None
            )
        msg = SimpleNamespace(content="ok", tool_calls=None)
        usage = (
            SimpleNamespace(prompt_tokens=state["prompt"], completion_tokens=5) if state["prompt"] else None
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)

    monkeypatch.setattr(p._client.chat.completions, "create", create)
    msgs = [Message.user("hello " * 300)]
    for _ in range(6):  # 413s that say no numbers, over and over
        state["refuse"] = 2
        asyncio.run(p.complete(msgs, max_tokens=256))
    assert 1.0 < p._scale <= oc._MAX_SCALE  # not 1.4^12

    # Requests that go through say how big they really were: the estimate walks back to that.
    from neo2.core.providers.fit import estimate

    state["prompt"] = int(estimate(msgs, "", None, vision=False))
    for _ in range(8):
        asyncio.run(p.complete(msgs, max_tokens=256))
    assert p._scale < 1.1


# ---- second review: what a small brain keeps whole, and in which order it goes --------------------


def test_a_small_brain_keeps_the_board_whole_before_stale_results_of_other_tools():
    from neo2.core.agent.toolselect import MORE_TOOLS
    from neo2.core.providers.fit import fit

    board = ("AXButton “Board” " * 400)[:6000]
    msgs = [
        Message.user("play chess as white, look up the opening first"),
        *_pair("a", "open_app", "Opened Chess."),
        *_pair("b", "web_search", ("result about openings " * 300)[:4000]),
        *_pair("c", "web_fetch", ("page text about e4 " * 400)[:6000]),
        *_pair("d", "read_file", ("notes file " * 700)[:6000]),
        *_pair("e", "hotkey", "Pressed cmd+n."),
        *_pair("f", "ax_tree", board),
    ]
    approval = [
        Message.assistant("", [ToolCall("g", "applescript", {"script": "get name of windows"})]),
        Message.tool(
            [ToolResult("g", "applescript", "NEEDS_CONFIRM: run an AppleScript — Should I go ahead?")]
        ),
        Message.user("[User confirmed. applescript result: Chess] Continue and finish the task."),
    ]
    tools = [*registry().specs(), MORE_TOOLS]
    for request in (msgs, msgs + approval):  # a step of the run, and the follow-up after a yes
        f = fit(request, system="You are NEO. " * 200, tools=tools, budget=8000, max_tokens=4096,
                vision=False, name="groq")  # fmt: skip
        got = {r.name: r.content for m in f.messages for r in m.tool_results}
        assert got["ax_tree"] == board  # the board the model is playing on, whole
        # Web results it already read went first; the file (a look at something) only as far as needed.
        assert len(got["web_search"]) < 400 and len(got["web_fetch"]) < 400
        assert len(got["read_file"]) > 4000


def test_cutting_what_was_kept_whole_frees_room_for_it_not_for_tools():
    from neo2.core.agent.toolselect import MORE_TOOLS
    from neo2.core.providers.fit import estimate, fit

    board = "AXButton “Board” " * 1500  # far more than Groq's 8K takes whole
    msgs = [Message.user("play chess as white"), *_pair("a", "open_app", "Opened Chess."),
            *_pair("b", "ax_tree", board)]  # fmt: skip
    tools = [*registry().specs(), MORE_TOOLS]
    system = "You are NEO. " * 200
    f = fit(msgs, system=system, tools=tools, budget=8000, max_tokens=4096, vision=False, name="groq")
    tree = next(r.content for m in f.messages for r in m.tool_results if r.name == "ax_tree")
    assert 2000 < len(tree) < len(board)  # cut only as far as it had to be…
    assert {s.name for s in f.tools} == {"open_app", "ax_tree", "more_tools"}  # …and tools didn't get it back
    total = estimate(f.messages, system, f.tools, vision=False) + f.max_tokens
    assert int(8000 * 0.9) - 60 < total <= int(8000 * 0.9)


def test_a_per_minute_brain_gets_back_the_first_tools_but_not_a_prompt_of_its_whole_budget():
    from neo2.core.agent.loop import all_tools
    from neo2.core.agent.toolselect import MORE_TOOLS, SCREEN
    from neo2.core.providers.fit import estimate, fit

    msgs = _chess_follow_up()
    tools = [*all_tools(registry()), MORE_TOOLS]  # the follow-up after a yes: every tool
    system = "You are NEO. " * 100
    kw = dict(system=system, tools=tools, budget=8000, max_tokens=4096, vision=False, name="groq")
    rate = fit(msgs, per_minute=True, **kw)
    context = fit(msgs, **kw)
    # Groq's 8K is per minute: each step pays its prompt back before the next can go (a 5K prompt
    # is ~40s of waiting per step). The observe/act tools still come back first.
    assert estimate(rate.messages, system, rate.tools, vision=False) <= 8000 * 0.5
    assert {"click", "type_text", "ax_press", "more_tools"} <= {s.name for s in rate.tools}
    # (whichever module registered its tools first: the observe/act ones are what comes back)
    assert {t.name for t in all_tools(registry())[: len(SCREEN)]} == SCREEN
    assert estimate(context.messages, system, context.tools, vision=False) > 8000 * 0.5
    assert len(context.tools) > len(rate.tools)


def test_the_one_brain_left_is_waited_on_through_a_per_minute_limit(chain, monkeypatch):
    # Groq on its 8K a minute, the step after a 5.5K prompt: "try again in 41.3s".
    groq = FakeBrain("groq", RateLimited("429 try again in 41.3s", retry_after=41.3))
    local = FakeBrain("local", Unavailable("local model server not running"))
    c = chain(groq=groq, local=local)
    waits: list[float] = []
    real_sleep = asyncio.sleep

    async def sleep(secs, *a, **kw):  # (no real waiting in a test: the minute passes at once)
        waits.append(secs)
        groq.error = None
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    assert asyncio.run(c.complete([Message.user("hi")])).text == "ok" and groq.asked == 2
    assert len(waits) == 1 and 41.3 <= waits[0] < 45


def test_the_chain_names_brains_the_way_the_user_hears_them(chain, monkeypatch):
    from neo2.core.events import bus

    notes: list[str] = []

    async def note(text):
        notes.append(text)

    monkeypatch.setattr(bus(), "note", note)
    lite = FakeBrain("gemini_lite", RateLimited("429 try again in 0.01s", retry_after=0.01))
    local = FakeBrain("local", Unavailable("local model server not running"))
    c = chain(gemini_lite=lite, local=local)

    async def go():
        async def later():
            await asyncio.sleep(0)
            lite.error = None

        asyncio.get_running_loop().create_task(later())
        return await c.complete([Message.user("hi")])

    assert asyncio.run(go()).text == "ok"
    assert "Gemini Lite unavailable, switching brain" in notes
    assert "The local model unavailable, switching brain" in notes
    assert "Waiting 0s for Gemini Lite's rate limit" in notes


# ---- second review: a yes continues the request it was asked for ------------------------------------


def _fake_every_tool(monkeypatch):
    """Every tool NEO has, replaced by a stand-in: nothing real can run whatever a test reaches."""
    for n, t in list(registry()._tools.items()):
        monkeypatch.setitem(
            registry()._tools, n, RegisteredTool(n, t.description, t.parameters, lambda a, c: "done (fake)")
        )


def test_a_yes_after_another_request_continues_the_request_it_was_asked_for(monkeypatch):
    from neo2.core.providers.fit import fit

    _fake_every_tool(monkeypatch)
    ran: list[str] = []
    _chess_tools(monkeypatch, ran)
    board = ("AXButton “e2” " * 300).strip()
    monkeypatch.setitem(
        registry()._tools, "ax_tree",
        RegisteredTool("ax_tree", "test tool", _OBJ, lambda a, c: board, parallel_safe=True),
    )  # fmt: skip
    monkeypatch.setattr(sess_mod.fastpath, "plan", lambda text, **_: None)

    class Reflex:
        async def decide(self, t):
            if "weather" in t:
                return Decision("chat", 0.95, False, 0.0, False, 0.0, "laya+head", reply=True, reply_p=0.9)
            return Decision("agent_task", 0.95, False, 0.0, False, 0.0, "laya+head", reply=False, reply_p=0.1)

    class Brain(ChessBrain):
        async def complete(self, messages, **kw):
            self.last = list(messages)
            return await super().complete(messages, **kw)

        async def stream(self, messages, **kw):
            yield "Sunny, 24 degrees."

    b = Brain()
    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": b)

    async def go():
        s = Session(reflex=Reflex())
        r = await s.handle("play chess as white", from_user=True)
        assert r.route == "confirm", r.text
        # While the card waits, Gemini Live relays another question of the user's: it's answered,
        # and the card still waits for the user.
        w = await s.handle("what's the weather like", from_user=False, origin="user")
        assert w.route == "chat" and confirm.peek() is not None
        r = await s.answer(True, action_id=confirm.peek().action_id)
        return s, r

    s, r = asyncio.run(go())
    assert r.route == "confirm" and len(ran) == 1 and not b.bad, b.bad[:2]
    msgs = b.last  # the follow-up after the yes
    asks = [m.text for m in msgs if m.role == "user"]
    assert asks[-1].startswith("[User confirmed (request: “play chess as white”).")
    # The other request came before the one being continued, not between its steps.
    assert asks.index("what's the weather like") < asks.index("play chess as white")
    from neo2.core.providers.history import request_start

    assert msgs[request_start(msgs)].text == "play chess as white"
    from neo2.core.agent.toolselect import MORE_TOOLS

    tools = [*registry().specs(), MORE_TOOLS]  # (what the follow-up after a yes offers)
    f = fit(msgs, system="You are NEO. " * 200, tools=tools, budget=8000, max_tokens=4096, vision=False,
            name="groq")  # fmt: skip
    assert "earlier requests" in f.trimmed and f.messages[0].text == "play chess as white"
    assert board in [r.content for m in f.messages for r in m.tool_results]
    # The session's own history keeps what happened in the order it happened.
    hist = [m.text for m in s.history if m.role == "user"]
    assert hist.index("play chess as white") < hist.index("what's the weather like")


def test_the_window_opening_on_a_note_finds_the_request_it_names():
    s = Session(reflex=_agent_reflex())
    s.history = [
        Message.user("play chess as white"),
        *_pair("t", "ax_tree", "board"),
        Message.assistant("", [ToolCall("s", "applescript", {"script": "x"})]),
        Message.tool([ToolResult("s", "applescript", "NEEDS_CONFIRM: run an AppleScript — go ahead?")]),
        Message.user("what's the weather like"),
        Message.assistant("Sunny."),
    ]
    note = "[User confirmed (request: “play chess as white”). applescript result: ok] Continue and finish the task."
    s.history.append(Message.user(note))
    for k in range(20):  # the follow-up run: the window opens on its note
        s.history += list(_pair(f"r{k}", "applescript", "window 1"))
    h = s._trimmed()
    assert h[0].text == "play chess as white" and h[len(h) - 41].text == note
    assert [r.content for m in h for r in m.tool_results if r.name == "ax_tree"] == ["board"]


# ---- second review: the board stays in view through a long approval chain ---------------------------


def _approvals(board: str, n: int) -> list[Message]:
    """A board read once, then n approvals: each a note, a long AppleScript read, a new staged one."""
    msgs = [Message.user("play chess as white"), *_pair("t", "ax_tree", board)]
    for k in range(n + 1):
        if k:
            msgs += [
                Message.user(f"[User confirmed. applescript result: ok {k}] Continue and finish the task."),
                *_pair(f"r{k}", "applescript", "window 1, " * 150),
            ]
        msgs += [
            Message.assistant("", [ToolCall(f"s{k}", "applescript", {"script": "x"})]),
            Message.tool(
                [ToolResult(f"s{k}", "applescript", "NEEDS_CONFIRM: run an AppleScript — go ahead?")]
            ),
        ]
    return msgs


def test_the_only_look_at_the_board_stays_in_view_when_the_window_opens_on_a_note():
    from neo2.core.providers.fit import fit

    board = "AXButton “e2” " * 300
    s = Session(reflex=_agent_reflex())
    for n in range(1, 13):
        s.history = _approvals(board, n)
        h = s._trimmed()
        assert h[0].text == "play chess as white" and not wire_violations(h), n
        assert [r.content for m in h for r in m.tool_results if r.name == "ax_tree"] == [board], n
        f = fit([*h, Message.user("[User confirmed. applescript result: ok] Continue.")], system="sys",
                budget=8000, max_tokens=4096, vision=False, name="groq")  # fmt: skip
        assert board in [r.content for m in f.messages for r in m.tool_results], n
    assert len(s.history) > sess_mod._MAX_HISTORY + 10  # the window did open on a note


def test_the_memory_cap_keeps_the_last_look_at_the_board_with_the_request(monkeypatch):
    async def no_summary(msgs, started, budget_s=8.0):
        return ""

    monkeypatch.setattr(Session, "_summarize", staticmethod(no_summary))
    board = "AXButton “e2” " * 300

    async def go():
        s = Session(reflex=_agent_reflex())
        s.history = _approvals(board, 0)
        for k in range(1, 45):
            await s._remember(*_approvals(board, k)[-5:])
        return s

    s = asyncio.run(go())
    assert len(s.history) <= sess_mod._HISTORY_CAP and s.history[0].text == "play chess as white"
    assert [r.content for m in s.history for r in m.tool_results if r.name == "ax_tree"] == [board]
    assert not wire_violations(s.history) and board in [
        r.content for m in s._trimmed() for r in m.tool_results
    ]


# ---- second review: a 403 about one model isn't a refused key ---------------------------------------


def test_a_403_for_one_model_leaves_that_keys_other_models_to_answer(gem):
    denied = _err(
        403,
        "Permission denied on resource project for model gemini-3.8-flash.",
        [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "IAM_PERMISSION_DENIED",
          "metadata": {"consumer": "projects/1"}}],
    )  # fmt: skip
    gem.rules["gemini-3.8-flash"] = denied
    gem.rules["*"] = _reply(text="ok")
    assert asyncio.run(gem.p.complete([Message.user("hi")])).text == "ok"
    asked = [(m, k) for m, k, _c in gem.sent]
    assert asked == [("gemini-3.8-flash", 0), ("gemini-3.8-flash", 1), ("gemini-3.7-flash", 0)]
    gem.sent.clear()
    assert asyncio.run(gem.p.complete([Message.user("hi")])).text == "ok"
    assert [(m, k) for m, k, _c in gem.sent] == [("gemini-3.7-flash", 0)]  # not asked again

    gem.sent.clear()
    gem.p._denied.clear()
    gem.rules.clear()
    gem.rules["*"] = denied  # every model refused: said as such, not as a bad key
    with pytest.raises(Unavailable) as e:
        asyncio.run(gem.p.complete([Message.user("hi")]))
    assert "permission denied" in str(e.value) and "API key" not in str(e.value)
