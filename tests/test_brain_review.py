"""The brains review's fixes, kept as regression tests (a merge that takes the older side fails here):

- click_text counts as checked through Accessibility only when it pressed its control there and saw
  it change — never when it clicked words read off the screen with the mouse;
- an action that went wrong stops the rest of its reply only on the screen; a failed reminder, mail
  or file call leaves the others to run, as before batching;
- the Cerebras cap fails closed ("off", -1, a typo: Cerebras off), the key never reaches an error's
  text, and a stream that said nothing is paid for all the same;
- a user whose brain is Cerebras's (NEO_BRAIN=cerebras, DECISIONS 65) gets its seeing model first
  for the steps that carry a screenshot.

Hermetic: Cerebras' API is an httpx.MockTransport, the other brains fakes, no model or Mac is touched.
"""

from __future__ import annotations

import asyncio
import json
import weakref
from typing import Any

import httpx
import pytest

from neo2.core import config, health
from neo2.core import providers as prov
from neo2.core.agent import batch
from neo2.core.agent.loop import run_agent
from neo2.core.agent.registry import RegisteredTool, Registry, ToolContext, ToolOutput
from neo2.core.events import EventBus
from neo2.core.providers import cerebras as cb
from neo2.core.providers import cerebras_hooks as hooks
from neo2.core.providers import spend
from neo2.core.providers.base import (
    ImagePart,
    Message,
    Overloaded,
    ProviderError,
    ToolCall,
    ToolResult,
    Turn,
)
from neo2.core.providers.openai_compat import Truncated

KEY = "csk-test-not-a-real-key"
ASK = [Message.user("press the OK button")]


# ---- click_text: Accessibility, or the mouse ----------------------------------------------------


def test_click_text_at_words_read_off_the_screen_is_not_verified():
    """The OCR fallback clicks with the mouse and carries the pointer's outcome: a press highlight is
    a change at the point, not a check through Accessibility — a check turn still follows it, and
    `finish` can't end the run on it (the same Effect on `click` isn't verified either)."""
    from neo2.core.tools.computer import ocr, pointer

    out = ocr._with_effect(
        ToolOutput("Clicked the words 'Start' — found by reading the screen (Accessibility has no such label)."),
        pointer.Outcome("Changed: the screen changed around (40, 60).", "", True),
    )
    for tool in ("click_text", "click"):
        r = ToolResult("c1", tool, out.text)
        r.effect = out.effect
        assert out.effect.changed is True and not batch.verified(tool, r) and not batch.seen(tool, r), tool
        assert not batch.finished([ToolCall("c1", tool, {"label": "Start"})], [r]), tool


def _press_world(monkeypatch, *, changed: bool | None):
    """A front window with one button that takes AXPress; the press and what it changed are fakes."""
    from neo2.core.tools import computer as comp
    from neo2.core.tools import load_all
    from neo2.core.tools.computer import ax

    load_all()
    els = [ax.Element("e1", "AXButton", "Start", "", 10, 10, 60, 24, actions=["AXPress"], ref=object())]
    monkeypatch.setattr(ax, "window_elements", lambda app=None: ("Game", (0, 0, 800, 600), els))
    monkeypatch.setattr(comp.apps, "_target", None)
    monkeypatch.setattr(comp.apps, "front_name", lambda: "Game")
    monkeypatch.setattr(ax, "pid_of", lambda ref: 4242)
    monkeypatch.setattr(ax, "is_neo", lambda pid, name="": False)
    monkeypatch.setattr(ax, "_running", lambda pid: None)
    monkeypatch.setattr(ax, "press_watch", lambda e, pid: "watching")
    pressed: list[str] = []
    monkeypatch.setattr(ax, "press_element", lambda e: pressed.append(e.id) or "Pressed")
    said = "(the button changed)" if changed else "(nothing visibly changed)"
    monkeypatch.setattr(ax, "press_change", lambda w, e, **kw: ax.PressChange(changed, False, "", said, ""))
    return pressed


@pytest.mark.parametrize(("changed", "verified"), [(True, True), (False, False), (None, False)])
def test_click_text_pressed_through_accessibility_says_whether_it_verified(monkeypatch, changed, verified):
    from neo2.core.agent.registry import registry

    pressed = _press_world(monkeypatch, changed=changed)
    out = asyncio.run(registry().invoke("click_text", {"label": "start"}, ToolContext(origin="agent")))
    assert pressed == ["e1"] and out.ok, out.text
    assert out.effect is not None and out.effect.verified is verified
    r = ToolResult("c1", "click_text", out.text)
    r.effect = out.effect
    assert batch.verified("click_text", r) is verified


# ---- a reply's actions after one that went wrong ------------------------------------------------


class _Script:
    name = "script"
    supports_vision = True
    supports_tools = True

    def __init__(self, *turns: Turn) -> None:
        self.turns = list(turns)

    async def complete(self, messages, **kw) -> Turn:
        return self.turns.pop(0)


def _two_calls(tool: str, category: str, first_fails: str) -> tuple[list[str], list[str]]:
    reg = Registry()
    ran: list[str] = []

    async def handler(a, c):
        ran.append(a["text"])
        return f"Error: couldn't do {a['text']}" if a["text"] == first_fails else f"Did {a['text']}"

    params = {"type": "object", "properties": {"text": {"type": "string"}}}
    reg.register(RegisteredTool(tool, "an action", params, handler, category=category))
    brain = _Script(
        Turn(tool_calls=[ToolCall("1", tool, {"text": "milk"}), ToolCall("2", tool, {"text": "eggs"})]),
        Turn(text="done"),
    )
    res = asyncio.run(
        run_agent("x", provider=brain, system="", ev=EventBus(), reg=reg, tools=[reg.get(tool).spec()], verify=False)
    )
    return ran, [s.result for s in res.steps]


def test_a_failed_action_off_the_screen_leaves_the_rest_of_the_reply_to_run():
    ran, results = _two_calls("reminder_add", "apps", first_fails="milk")
    assert ran == ["milk", "eggs"] and results[1] == "Did eggs"


def test_a_failed_action_on_the_screen_still_stops_the_rest_of_the_reply():
    ran, results = _two_calls("type_text", "computer", first_fails="milk")
    assert ran == ["milk"] and results[1].startswith("skipped: an earlier action in this turn failed")


# ---- Cerebras: the cap, the key, the silent stream ----------------------------------------------


def _completion(content: str | None = "ok", *, usage=(1000, 200)) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": cb.TEXT_MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
    }
    if usage:
        out["usage"] = {"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": sum(usage)}
    return out


class _Cerebras:
    """Cerebras' API: replies queued per test (else a plain answer)."""

    def __init__(self) -> None:
        self.replies: list[tuple[int, Any, dict[str, str]]] = []
        self.bodies: list[dict[str, Any]] = []

    def reply(self, status: int, payload: Any, headers: dict[str, str] | None = None) -> None:
        self.replies.append((status, payload, headers or {}))

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            ids = [cb.TEXT_MODEL, cb.VISION_MODEL]
            return httpx.Response(200, json={"object": "list", "data": [{"id": i, "object": "model"} for i in ids]})
        self.bodies.append(json.loads(request.content))
        status, payload, headers = self.replies.pop(0) if self.replies else (200, _completion(), {})
        if isinstance(payload, str):  # a raw stream body
            return httpx.Response(status, content=payload.encode(), headers={"content-type": "text/event-stream"})
        return httpx.Response(status, json=payload, headers=headers)


@pytest.fixture
def mac(monkeypatch):
    """NEO.app's settings with (fake) keys; a fresh pool, accounts, cooldowns and problem list."""

    def make(**env: str) -> config.Settings:
        for k, v in {"CEREBRAS_API_KEY": KEY, "GEMINI_API_KEY": "g-test", "GROQ_API_KEY": "gsk-test", **env}.items():
            monkeypatch.setenv(k, v)
        s = config.Settings()
        s.ensure_dirs()
        monkeypatch.setattr(config, "_settings", s)
        return s

    monkeypatch.setattr(hooks, "_accounts", weakref.WeakKeyDictionary())
    monkeypatch.setattr(prov, "_cache", {})
    monkeypatch.setattr(prov, "_demoted", {})
    monkeypatch.setattr(health, "_live", {})
    return make


@pytest.fixture
def api(monkeypatch):
    f = _Cerebras()
    monkeypatch.setattr(cb, "_transport", httpx.MockTransport(f.handler))
    return f


@pytest.mark.parametrize("said", ["off", "none", "-1", "1,00", "one dollar", "nan", "inf"])
def test_a_cap_nobody_can_read_turns_cerebras_off(mac, said):
    """The user who wrote "off" to stop paying, or mistyped the cap, is never billed the default."""
    s = mac(NEO_CEREBRAS_DAILY_BUDGET=said)
    assert s.cerebras_daily_budget == 0.0 and spend.budget(said) == 0.0
    assert not hooks.enabled(s)
    for purpose in ("agent", "fast", "light", "vision"):
        c = prov.brain(purpose)
        assert not {cb.TEXT_BRAIN, cb.VISION_BRAIN} & {*c.names, *c.eyes}, purpose


def test_a_cap_it_can_read_is_kept_and_only_an_unset_one_is_the_default(mac):
    assert mac(NEO_CEREBRAS_DAILY_BUDGET="$1.25").cerebras_daily_budget == 1.25
    assert mac(NEO_CEREBRAS_DAILY_BUDGET=" 1.00 ").cerebras_daily_budget == 1.0
    assert spend.budget(None) == spend.budget("") == 0.5 and spend.budget("0") == 0.0


@pytest.mark.parametrize(
    ("status", "body", "headers"),
    [
        (429, {"message": f"slow down, {KEY}"}, {"retry-after": "100"}),  # (past the short wait: raised)
        (400, {"message": f"bad request from key {KEY}"}, {}),
        (500, {"message": f"server trouble near {KEY}"}, {}),
    ],
)
async def test_an_error_that_echoes_the_key_never_carries_it(mac, api, status, body, headers):
    mac()
    for _ in range(3):  # (a server error may be asked again)
        api.reply(status, body, headers)
    with pytest.raises(ProviderError) as e:
        await cb.cerebras().complete(ASK, effort="low")
    assert KEY not in str(e.value) and all(KEY not in str(a) for a in e.value.args)


async def test_a_stream_that_said_nothing_is_paid_for_all_the_same(mac, api):
    """Its thinking used the room: the request went through and is billed, so the cap must see it."""
    s = mac()
    api.reply(200, "data: [DONE]\n\n")
    with pytest.raises(Truncated):
        _ = [c async for c in cb.cerebras().stream(ASK)]
    tin, tout = json.loads((s.data_dir / spend.FILE).read_text())["tokens"][cb.TEXT_MODEL]
    assert tin > 0 and tout >= cb.ROOM["low"]


# ---- which brain sees a screenshot first ---------------------------------------------------------


class _Brain:
    """Any brain but Cerebras: answers, or is overloaded; records who was asked."""

    supports_tools = True

    def __init__(self, name: str, asked: list[str], *, busy: bool) -> None:
        self.name, self.model, self.asked, self.busy = name, f"{name}-model", asked, busy
        self.supports_vision = name in ("gemini", "gemini_lite", "claude", "groq_vision")
        self.on_device = name == "local"

    def cooling_until(self) -> float:
        return 0.0

    async def complete(self, messages, **kw) -> Turn:
        self.asked.append(self.name)
        if self.busy:
            raise Overloaded(f"{self.name}: 503")
        return Turn(text=f"from {self.name}")


def _brains(monkeypatch, *, busy: tuple[str, ...] = ()) -> list[str]:
    asked: list[str] = []
    fakes = {n: _Brain(n, asked, busy=n in busy) for n in ("gemini", "gemini_lite", "claude", "groq", "groq_vision", "local")}
    real = prov._build
    monkeypatch.setattr(prov, "_build", lambda n: fakes[n] if n in fakes else real(n))
    return asked


def _screenshot_step() -> list[Message]:
    return [
        Message.user("what's on the board?"),
        Message.assistant(tool_calls=[ToolCall("c0", "screenshot", {})]),
        Message.tool([ToolResult("c0", "screenshot", "shot", images=[ImagePart(b"x" * 500, "image/jpeg")])]),
    ]


def _eyes(purpose: str) -> list[str]:
    """The seeing brains a step with a screenshot asks, in order (set up or not)."""
    c = prov.brain(purpose)
    return [n for n in (*c.names, *c.eyes) if n in ("gemini", "gemini_lite", "claude", "groq_vision", cb.VISION_BRAIN)]


@pytest.mark.parametrize(
    ("env", "agent", "vision"),
    [
        # no Cerebras key: Gemini, then the free seeing fallbacks
        ({"CEREBRAS_API_KEY": ""}, ["gemini", "gemini_lite", "claude", "groq_vision"],
         ["gemini", "gemini_lite", "claude", "groq_vision"]),
        # a key, Gemini the brain: Gemini first, Cerebras' seeing model ahead of Groq's (decisions 49, 61)
        ({}, ["gemini", "gemini_lite", "claude", "cerebras_vision", "groq_vision"],
         ["gemini", "gemini_lite", "claude", "cerebras_vision", "groq_vision"]),
        # NEO_BRAIN=cerebras (the user's choice, DECISIONS 65): qwen answers the screenshot steps; the
        # vision chain asks Gemini, then qwen (vision gemini > cerebras_vision > cerebras > groq > local)
        ({"NEO_BRAIN": "cerebras"}, ["cerebras_vision", "gemini", "gemini_lite", "claude", "groq_vision"],
         ["gemini", "cerebras_vision", "gemini_lite", "claude", "groq_vision"]),
        # NEO_BRAIN=cerebras_vision (DECISIONS 62): its own brain sees, first everywhere
        ({"NEO_BRAIN": "cerebras_vision"}, ["cerebras_vision", "gemini", "gemini_lite", "claude", "groq_vision"],
         ["cerebras_vision", "gemini", "gemini_lite", "claude", "groq_vision"]),
    ],
)  # fmt: skip
def test_the_seeing_brains_a_screenshot_step_asks_in_order(mac, env, agent, vision):
    mac(**env)
    assert _eyes("agent") == agent and _eyes("vision") == vision


async def test_brain_cerebras_sends_the_screenshot_to_qwen_before_gemini(mac, api, monkeypatch):
    mac(NEO_BRAIN="cerebras")
    asked = _brains(monkeypatch)
    api.reply(200, {**_completion("A board.", usage=(900, 30)), "model": cb.VISION_MODEL})
    chain = prov.brain("agent")
    turn = await chain.complete(_screenshot_step(), effort="low")
    assert chain.last_used == cb.VISION_BRAIN and turn.sees and asked == []
    assert [b["model"] for b in api.bodies] == [cb.VISION_MODEL]
    # qwen out: the free seeing brains still answer it, never a text-only one
    api.reply(503, {"message": "busy"})
    api.reply(503, {"message": "busy"})
    turn = await prov.brain("agent").complete(_screenshot_step(), effort="low")
    assert turn.text == "from gemini" and asked == ["gemini"]
