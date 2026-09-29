"""The context diet: an agent's request carries what the model needs from the task, not all of it.

The log (NEO_BRAIN=cerebras_vision): one "play chess as white" — 18 tool calls in 181 s — sent
466,043 input and 32,067 output tokens to qwen-3.8-27b and spent Cerebras' $0.50 daily budget in
about four minutes. Twice a reply spent all its room thinking and was asked again: the whole prompt
paid for twice.

The replica here is that task, hermetic: the brain is a fake Cerebras server (httpx.MockTransport)
that asks for the calls the log shows — open the app, read its tree, start a new game as white,
then look (screenshot) → move (drag) — the tools are fakes that answer the way the real ones do
(full-size screenshots, AX trees of 3-6K characters), and everything between them is NEO's own
code: the real system prompt and tool schemas, the agent loop, the session's history window, the
Cerebras provider, fit, the OpenAI wire format. Every request body the server receives is measured
(text: UTF-8 bytes / 3.4, fit's rule; pictures: one token per 28×28 pixels, Qwen-VL's). The fake
brain also checks, before each step, that what it needs is in front of it — the user's request, the
latest screenshot (the one its move's pixels are of), the latest tree of the app (the one its
element id comes from), the last step whole — and asks for another look when it isn't: the diet
may not change a single action. NEO as it was before the diet is the same code with the diet
switched off (:func:`as_before`), checked against main's own numbers.

No model is called, nothing leaves the process, ~/.neo is never read, nothing is clicked.
"""

from __future__ import annotations

import base64
import dataclasses
import io
import json
import math
import random
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from neo2.core.agent import prompt
from neo2.core.agent.loop import run_agent
from neo2.core.agent.registry import Registry, ToolContext, ToolOutput, catalog
from neo2.core.agent.session import Session
from neo2.core.agent.toolselect import ToolSelector
from neo2.core.decide import games
from neo2.core.decide.inventory import App
from neo2.core.events import EventBus
from neo2.core.providers import cerebras as cb
from neo2.core.providers import diet, spend
from neo2.core.providers.base import ImagePart, Message, ToolCall, ToolResult, ToolSpec
from neo2.core.providers.fit import estimate, fit
from neo2.core.providers.history import naming
from neo2.core.tools import load_all

KEY = "csk-test-not-a-real-key"
GOAL = "play chess as white"
AGAIN = "another game of chess, as white again"
CHESS = App(bundle_id="test.boards.chess", name="Chess", path="/System/Applications/Chess.app", category="board-games")
W, H = 1512, 982  # a 14" MacBook Pro's display in points: the size NEO's screenshot sends
_BYTES_PER_TOKEN = 3.4  # (fit's rule for text)
_PATCH = 28  # Qwen-VL: one token per 28×28 pixels of a picture

# ---- the fake tools: what the real ones answer, in size and shape ---------------------------------


def _png(i: int) -> bytes:
    """A screenshot of the display with Chess on it, as a PNG of the real size (~330 KB): a board
    with pieces moved about, on a window with some texture."""
    from PIL import Image, ImageDraw

    rnd = random.Random(i)
    im = Image.new("RGB", (W, H), (236, 236, 236))
    d = ImageDraw.Draw(im)
    x0, y0, s = 356, 90, 100
    for r in range(8):
        for c in range(8):
            d.rectangle(
                [x0 + c * s, y0 + r * s, x0 + c * s + s - 1, y0 + r * s + s - 1],
                fill=(238, 238, 210) if (r + c) % 2 == 0 else (118, 150, 86),
            )
    for k in range(32):
        r, c = rnd.randrange(8), rnd.randrange(8)
        d.ellipse(
            [x0 + c * s + 20, y0 + r * s + 15, x0 + c * s + 80, y0 + r * s + 85],
            fill=(20, 20, 20) if k % 2 else (250, 250, 250),
            outline=(0, 0, 0),
        )
    px = im.load()
    for _ in range(40000):
        v = rnd.randrange(256)
        px[rnd.randrange(W), rnd.randrange(H)] = (v, v, v)
    b = io.BytesIO()
    im.save(b, format="PNG")
    return b.getvalue()


_ROLES = ["Button", "MenuItem", "StaticText", "Image", "Group", "PopUpButton", "RadioButton", "Toolbar"]
_WORDS = ["New Game", "Undo", "Hint", "Game", "Moves", "White", "Black", "Human", "Computer", "Level", "OK", "Cancel",
          "Board", "Rotate", "Show Moves", "Players", "Vs. Computer", "Strength", "Clock", "Window", "Help"]  # fmt: skip


def _tree(i: int, ids: tuple[str, ...], title: str, app: str = "Chess") -> str:
    """An ax_tree as snapshot() writes it: ~3-6K characters of element lines."""
    rnd = random.Random(1000 + i)
    lines = [f'App: {app} — window "{title}"']
    want = list(ids)
    for k in range(rnd.randrange(55, 95)):
        eid = want.pop(0) if want and k % 9 == 4 else f"e{100 + i * 200 + k}"
        extra = f" = '{rnd.choice(_WORDS)}'" if rnd.random() < 0.3 else ""
        lines.append(
            f'  {eid} {rnd.choice(_ROLES)} "{rnd.choice(_WORDS)}"{extra} @({rnd.randrange(0, 1500)},'
            f"{rnd.randrange(0, 980)} {rnd.randrange(10, 400)}x{rnd.randrange(10, 60)})"
        )
    lines += [f'  {eid} Button "OK" @(900,640 80x24)' for eid in want]
    return "\n".join(lines)


def _note(sid: str) -> str:
    return (
        f"Screenshot {sid} attached (all of display 1 “Built-in Retina Display”, {W}×{H} points, main); the "
        f"image is {W}×{H} px. To act on something in it, give its pixel position in this image with "
        f'shot="{sid}" — click x,y · drag x1,y1,x2,y2 · scroll x,y — and NEO maps it to the screen. '
        "(Here image pixels are also screen points.)"
    )


def _tool_output(n: int, name: str, a: dict, shots: dict[str, bytes]) -> str | ToolOutput:
    """The n-th call of a run, as the real tool would answer it."""
    if name == "open_app":
        return "Opened Chess."
    if name == "ax_tree":
        ids = ("e7",) if n < 3 else ("e31",) if n < 5 else ()
        title = "Chess" if n < 3 else "New Game" if n < 5 else "Chess — White to move"
        return _tree(n, ids, title)
    if name == "click":
        return (
            f"Clicked {a['id']} (MenuItem) at (412, 88) in Chess. The screen changed there: a sheet opened "
            "(“New Game”).\n<untrusted source=\"screen\">\nunder the pointer: MenuItem \"New Game\" in Chess\n"
            "</untrusted>"
        )
    if name == "screenshot":
        sid = f"s{len(shots) + 1}"
        shots[sid] = _png(len(shots) + 1)
        return ToolOutput(_note(sid), images=[ImagePart(shots[sid], "image/png")], fence=False)
    if name == "drag":
        x1, y1, x2, y2 = (int(a[k]) for k in ("x1", "y1", "x2", "y2"))
        return (
            f"Dragged ({x1}, {y1}) → ({x2}, {y2}) (pixels of {a.get('shot')}) in Chess. Something changed at "
            f"the start ({x1}, {y1}) and at the end ({x2}, {y2}).\n"
            '<untrusted source="screen">\nunder the pointer: start Image "board" in Chess; end Image "board" '
            "in Chess; the window's title changed to “Chess — Black to move”\n</untrusted>"
        )
    return f"Error: {name} isn't part of this replica."


# ---- the task as the log ran it --------------------------------------------------------------------


@dataclass
class Step:
    tool: str
    args: dict[str, Any]
    needs: tuple[str, ...] = ()  # what the fake brain must have in front of it to ask for this
    hard: bool = False  # a step it thinks long about (the log: two replies out of room)


def script(first_shot: int = 1) -> list[Step]:
    """The log's 18 calls: open Chess, read it, New Game as white, then look → move, five moves."""
    out = [
        Step("open_app", {"target": "Chess"}, ("request",)),
        Step("ax_tree", {"app": "Chess"}, ("request", "whole:last")),
        Step("click", {"id": "e7"}, ("request", "tree:e7")),  # Game ▸ New Game
        Step("ax_tree", {"app": "Chess"}, ("request", "whole:last")),
        Step("click", {"id": "e31"}, ("request", "tree:e31")),  # the sheet: White, vs. Computer, OK
    ]
    moves = [(1, 6, 1, 4), (6, 7, 5, 5), (3, 6, 3, 4), (2, 7, 5, 4), (5, 7, 2, 4)]  # (col,row)→(col,row)
    for n, (c1, r1, c2, r2) in enumerate(moves, start=1):
        sid = f"s{first_shot + n - 1}"
        out.append(Step("screenshot", {}, ("request", "whole:last")))
        a = {"x1": 406 + c1 * 100, "y1": 140 + r1 * 100, "x2": 406 + c2 * 100, "y2": 140 + r2 * 100, "shot": sid}
        out.append(Step("drag", a, ("request", f"shot:{sid}", "whole:last"), hard=n in (2, 4)))
        if n in (1, 3):  # whose turn is it? (the window's title says)
            out.append(Step("ax_tree", {"app": "Chess"}, ("request", "whole:last")))
    out.append(Step("screenshot", {}, ("request", "whole:last")))
    return out


ANSWER = "We're playing: you're white and I've made the opening moves — it's black's turn."


# ---- the fake Cerebras server: the brain, and the meter ----------------------------------------------


@dataclass
class Request:
    text_tokens: float
    pictures: list[tuple[int, int]]
    effort: str
    max_tokens: int
    truncated: bool = False
    out_tokens: int = 0

    @property
    def picture_tokens(self) -> int:
        return sum(math.ceil(w / _PATCH) * math.ceil(h / _PATCH) for w, h in self.pictures)

    @property
    def tokens(self) -> int:
        return int(self.text_tokens) + self.picture_tokens


@dataclass
class Brain:
    """The chess task's brain: asks for the next step of its script when what it needs is in front
    of it, else for another look (which then shows in the actions)."""

    goal: str = GOAL
    first_shot: int = 1
    shots: dict[str, bytes] = field(default_factory=dict)  # the session's screenshots, by shot id
    k: int = 0
    requests: list[Request] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.script = script(self.first_shot)
        self.answer_needs = ("request", f"shot:s{self.first_shot + 5}", "whole:last")

    def expected(self) -> list[str]:
        return [f"{s.tool} {json.dumps(s.args, sort_keys=True)}" for s in self.script] + ["answer"]

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": []})
        body = json.loads(request.content)
        req = _measure(body)
        self.requests.append(req)
        step = self.script[self.k] if self.k < len(self.script) else None
        lacking = [n for n in (step.needs if step else self.answer_needs) if not self._has(n, body)]
        thinking = _thinking(str(body.get("reasoning_effort")), bool(step and step.hard))
        if thinking + 80 > int(body["max_tokens"]):
            req.truncated, req.out_tokens = True, int(body["max_tokens"])
            return _reply(None, None, "length", req)
        req.out_tokens = thinking + 80
        if lacking:
            self.missing += [f"step {self.k + 1}: {n}" for n in lacking]
            if "request" in lacking:
                self.actions.append("answer: what should I do?")
                return _reply("What would you like me to do?", None, "stop", req)
            look = "screenshot" if any(n.startswith("shot:") for n in lacking) else "ax_tree"
            self.actions.append(f"{look} (again)")
            return _reply(None, (look, {"app": "Chess"} if look == "ax_tree" else {}), "tool_calls", req)
        if step is None:
            self.actions.append("answer")
            return _reply(ANSWER, None, "stop", req)
        self.k += 1
        self.actions.append(f"{step.tool} {json.dumps(step.args, sort_keys=True)}")
        return _reply(None, (step.tool, step.args), "tool_calls", req, cid=f"call_{self.first_shot}_{self.k}")

    def _has(self, need: str, body: dict) -> bool:
        msgs = body["messages"]
        if need == "request":
            return any(m["role"] == "user" and self.goal in _text(m) for m in msgs)
        tools = [m["content"] for m in msgs if m["role"] == "tool"]
        if need == "whole:last":
            return bool(tools) and not _cut(tools[-1])
        if need.startswith("tree:"):
            trees = [t for t in tools if 'source="ax_tree"' in t]
            return bool(trees) and not _cut(trees[-1]) and f"  {need[5:]} " in trees[-1]
        if need.startswith("shot:"):
            sid = need[5:]
            pics = [p for m in msgs if isinstance(m.get("content"), list) for p in m["content"] if p["type"] == "image_url"]
            want = "data:image/png;base64," + base64.b64encode(self.shots.get(sid, b"-")).decode()
            noted = any(f'shot="{sid}"' in t and not _cut(t) for t in tools)
            return bool(pics) and pics[-1]["image_url"]["url"] == want and noted
        raise AssertionError(need)


def _text(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, list):
        return " ".join(p.get("text", "") for p in c if p["type"] == "text")
    return c or ""


def _cut(text: str) -> bool:
    return any(w in text for w in ("trimmed", "left out"))


def _thinking(effort: str, hard: bool) -> int:
    """Tokens the fake qwen reasons for before it answers: a hard step at high effort runs past 4096."""
    table = {"low": 1200, "medium": 2600, "high": 4400} if hard else {"low": 400, "medium": 800, "high": 1300}
    return table.get(effort, table["medium"])


def _measure(body: dict) -> Request:
    from PIL import Image

    text, pics = 0, []
    for m in body["messages"]:
        c = m.get("content")
        if isinstance(c, str):
            text += len(c.encode())
        elif isinstance(c, list):
            for p in c:
                if p["type"] == "text":
                    text += len(p["text"].encode())
                else:
                    raw = base64.b64decode(p["image_url"]["url"].split(",", 1)[1])
                    pics.append(Image.open(io.BytesIO(raw)).size)
        for tc in m.get("tool_calls") or []:
            text += len(tc["function"]["name"]) + len(tc["function"]["arguments"])
    text += len(json.dumps(body.get("tools") or [], ensure_ascii=False).encode())
    return Request(text / _BYTES_PER_TOKEN, pics, str(body.get("reasoning_effort")), int(body["max_tokens"]))


def _reply(content, tool, finish, req: Request, cid: str = "call_x") -> httpx.Response:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool:
        msg["tool_calls"] = [{"id": cid, "type": "function", "function": {"name": tool[0], "arguments": json.dumps(tool[1])}}]
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": cb.VISION_MODEL,
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": req.tokens, "completion_tokens": req.out_tokens,
                      "total_tokens": req.tokens + req.out_tokens},
        },
    )  # fmt: skip


# ---- running it ----------------------------------------------------------------------------------------


@dataclass
class Replay:
    brain: Brain
    answer: str
    stopped: str
    messages: list[Message]

    @property
    def input_tokens(self) -> int:
        return sum(r.tokens for r in self.brain.requests)

    @property
    def output_tokens(self) -> int:
        return sum(r.out_tokens for r in self.brain.requests)

    @property
    def usd(self) -> float:
        cin, cout = spend.PRICES[cb.VISION_MODEL]
        return (self.input_tokens * cin + self.output_tokens * cout) / 1e6

    @property
    def retries(self) -> int:
        return sum(r.truncated for r in self.brain.requests)

    def table(self, title: str) -> str:
        rows = [f"{title}:"] + [
            f"  {i + 1:>2} {r.tokens:>7,} tok ({r.picture_tokens:>5,} in {len(r.pictures)} pictures) "
            f"effort={r.effort:<5} room={r.max_tokens:<5}{' OUT OF ROOM' if r.truncated else ''}"
            for i, r in enumerate(self.brain.requests)
        ]
        rows.append(
            f"  {len(self.brain.requests)} requests: {self.input_tokens:,} input + {self.output_tokens:,} output "
            f"tokens ≈ ${self.usd:.3f}, {self.retries} out of room"
        )
        return "\n".join(rows)


def _registry(names: set[str], shots: dict[str, bytes]) -> Registry:
    """The catalog's own tools (their schemas, flags and fences) with fake handlers."""
    reg = Registry(strict=True)
    n = [0]

    def fake(name: str):
        def handler(a: dict, c: ToolContext):
            n[0] += 1
            return _tool_output(n[0], name, a, shots)

        return handler

    for name in names:
        if (t := catalog().get(name)) is not None:
            reg.register(dataclasses.replace(t, handler=fake(name), fast_path=[]))
    return reg


async def replay(monkeypatch, brain: Brain, history: list[Message] | None = None) -> Replay:
    """One run of the task, as the session starts it: the real prompt, the tools the selector picks
    for it (keyword matching, as without the embedder), effort "high" (an agent task)."""
    load_all()
    monkeypatch.setattr(cb, "_transport", httpx.MockTransport(brain.handler))
    tools = ToolSelector(catalog()).select(brain.goal, needs_screen=True, required=("open_app", "activate_app"))
    specs = tools.specs(catalog())
    reg = _registry({s.name for s in specs} | {s.tool for s in brain.script}, brain.shots)
    system = prompt.build("", games.hint(CHESS))
    p = cb.hosted(KEY, vision=True, daily_budget=100.0)  # the replica measures the whole task
    res = await run_agent(
        brain.goal, provider=p, system=system, tools=specs, reg=reg, ev=EventBus(), effort="high", history=history
    )
    return Replay(brain, res.text, res.stopped, res.messages)


def as_before(monkeypatch) -> None:
    """NEO before the diet (main at 04757ee): no diet, no soft budget, and main's room for the
    caller's effort ("high" for an agent task: 4096 tokens)."""
    monkeypatch.setattr(diet, "diet", lambda messages: list(messages))
    monkeypatch.setattr(diet, "soft_budget", lambda messages: 10**12)
    monkeypatch.setattr(cb, "ROOM", {"low": 1024, "medium": 2048, "high": 4096})


async def _game_after_game(monkeypatch) -> Replay:
    """The task asked again in the same session, right after the first game: the history the session
    hands the second run (its window, as Session._trimmed makes it) holds the first."""
    first = await replay(monkeypatch, Brain())
    history = list(first.messages)  # what the session remembered (nothing older to strip pixels from)
    snapshot = Session._trimmed(SimpleNamespace(history=history))
    return await replay(monkeypatch, Brain(goal=AGAIN, first_shot=7, shots=dict(first.brain.shots)), snapshot)


# ---- the measurement ---------------------------------------------------------------------------------------


async def test_the_task_as_before_is_the_trees_own_numbers(monkeypatch):
    """The switched-off diet is the tree as it was: the same requests, token for token (this replica
    run on main 04757ee's providers: 21 requests of 139,317 input + 34,212 output tokens, two out of
    room; as the second game of a session, 242,370 input — on this tree's own prompt and tool list,
    174,933 and 280,570). The numbers are this tree's: re-pin them when the system prompt or the
    tool list changes."""
    as_before(monkeypatch)
    r = await replay(monkeypatch, Brain())
    assert (len(r.brain.requests), r.input_tokens, r.output_tokens, r.retries) == (21, 174_933, 34_212, 2)
    second = await _game_after_game(monkeypatch)
    assert (len(second.brain.requests), second.input_tokens, second.retries) == (21, 280_570, 2)


async def test_the_chess_task_on_the_diet_makes_the_same_moves_for_less(monkeypatch):
    with monkeypatch.context() as m:
        as_before(m)
        before = await replay(m, Brain())
    after = await replay(monkeypatch, Brain())
    print("\n" + before.table("before (main)") + "\n" + after.table("after (the diet)"))  # (pytest -s)
    for r in (before, after):
        assert r.stopped == "done" and r.answer == ANSWER
        assert not r.brain.missing, r.brain.missing  # it always had what it needed…
        assert r.brain.actions == r.brain.expected()  # …so the same 18 calls, in the same order
    # The effort the task asked for, with room for its thinking: no prompt sent twice.
    assert before.retries == 2 and after.retries == 0
    assert all(r.effort == "high" and r.max_tokens >= 16_384 for r in after.brain.requests)
    assert all(len(r.pictures) <= 1 for r in after.brain.requests)
    assert after.input_tokens < before.input_tokens * 0.9
    assert after.output_tokens < before.output_tokens  # (the room the two retries spent thinking)
    assert after.usd < before.usd * 0.9


async def test_a_second_game_in_the_same_session_no_longer_carries_the_first(monkeypatch):
    """Before, the session handed the second run the first one's last screenshot with its pixels and
    its last tree whole, and every step of the second game sent them again: two pictures a request."""
    with monkeypatch.context() as m:
        as_before(m)
        before = await _game_after_game(m)
    after = await _game_after_game(monkeypatch)
    print("\n" + before.table("second game, before") + "\n" + after.table("second game, after"))
    for r in (before, after):
        assert r.stopped == "done" and not r.brain.missing, r.brain.missing
        assert r.brain.actions == r.brain.expected()
    assert sum(len(r.pictures) == 2 for r in before.brain.requests) >= 10
    assert all(len(r.pictures) <= 1 for r in after.brain.requests)
    assert after.input_tokens < before.input_tokens * 0.7
    assert after.usd < before.usd * 0.75


# ---- the rules, one by one ----------------------------------------------------------------------------------


def _pic(tag: bytes = b"p") -> ImagePart:
    return ImagePart(tag, "image/png")


def _steps(*steps: tuple[str, dict, ToolResult]) -> list[Message]:
    """(call, result) pairs."""
    out = []
    for i, (name, args, r) in enumerate(steps):
        cid = f"c{i}"
        out.append(Message.assistant("", [ToolCall(cid, name, args)]))
        out.append(Message.tool([dataclasses.replace(r, call_id=cid, name=name)]))
    return out


def _run(*steps: tuple[str, dict, ToolResult], ask: str = GOAL, images: list | None = None) -> list[Message]:
    """A transcript: the request, then (call, result) pairs."""
    return [Message.user(ask, images=images), *_steps(*steps)]


def _shot(sid: str) -> tuple[str, dict, ToolResult]:
    return "screenshot", {}, ToolResult("", "screenshot", _note(sid), images=[_pic(sid.encode())])


def _ax(app: str, i: int) -> tuple[str, dict, ToolResult]:
    body = '<untrusted source="ax_tree">\n' + _tree(i, (), app, app) + "\n</untrusted>"
    return "ax_tree", {"app": app}, ToolResult("", "ax_tree", body)


def _act(name: str = "drag", text: str = "Dragged (1, 2) → (3, 4) in Chess. " * 30) -> tuple[str, dict, ToolResult]:
    return name, {"x1": 1}, ToolResult("", name, text)


def _results(msgs: list[Message]) -> list[ToolResult]:
    return [r for m in msgs for r in m.tool_results]


def test_only_the_newest_picture_goes_with_the_requests_own():
    old = [Message.user("what's this?", images=[_pic(b"u0")]), Message.assistant("A cat.")]
    msgs = old + _run(_shot("s1"), _act(), _shot("s2"), _act(), ask="and now play chess", images=[_pic(b"u1")])
    out = diet.diet(msgs)
    rs = _results(out)
    assert [len(r.images) for r in rs] == [0, 0, 1, 0]  # s1's picture is a note; s2's goes
    assert rs[0].content.startswith("[older screenshot s1:") and "\n" not in rs[0].content
    assert out[2].images == msgs[2].images  # the request's own picture stays…
    assert out[0].images == [] and "left out" in out[0].text  # …one it replaced doesn't
    assert out[0].text.startswith("what's this?")  # (its words do)


def test_a_look_that_a_newer_look_replaced_is_one_line_and_each_apps_latest_tree_stays():
    msgs = _run(_ax("Chess", 1), _ax("Mail", 2), _ax("Chess", 3), _act(), _act("click"), _act())
    out = diet.diet(msgs)
    rs = _results(out)
    assert rs[0].content == "[older ax_tree: left out of this request — a newer look at the same thing follows]"
    assert rs[1].content == msgs[4].tool_results[0].content  # Mail's only tree: whole
    assert rs[2].content == msgs[6].tool_results[0].content  # Chess's latest: whole
    assert rs[3].content == msgs[8].tool_results[0].content  # an older action of this request: whole
    assert [r.content for r in rs[4:]] == [m.tool_results[0].content for m in msgs[10:13:2]]  # last two: whole
    # A tree read without naming the app is of the app it names in its first line.
    front = _run(_ax("Chess", 1), ("ax_tree", {}, _ax("Chess", 4)[2]), _act(), _act())
    assert _results(diet.diet(front))[0].content.startswith("[older ax_tree:")


def test_the_last_two_steps_questions_and_errors_stay_whole():
    ask = ToolResult("", "mail_send", "NEEDS_CONFIRM: send the email to Priya — Should I go ahead?")
    err = ToolResult("", "shell", "Error: " + "x" * 2000, ok=False)
    earlier = _run(("mail_send", {}, ask), ("shell", {}, err), _act(), ask="email Priya") + [Message.assistant("Sent.")]
    msgs = earlier + _run(_act(), _ax("Chess", 1), _ax("Chess", 2))
    rs, was = _results(diet.diet(msgs)), _results(msgs)
    assert rs[0].content == ask.content and rs[1].content == err.content  # an earlier request's too
    assert "chars of an older step" in rs[2].content and len(rs[2].content) < 500  # its other results: cut
    assert [r.content for r in rs[3:]] == [r.content for r in was[3:]]


def test_a_page_read_again_leaves_its_older_picture_behind_too():
    page = "URL: https://shop.example/cart\nTitle: Cart\n" + "item " * 400
    shot = ToolResult("", "browser_click", page, images=[_pic(b"cart")])
    msgs = _run(("browser_click", {"target": "3"}, shot), _act(), ("browser_read", {}, ToolResult("", "browser_read", page)), _act(), _act())
    rs = _results(diet.diet(msgs))
    assert rs[0].content.startswith("[older browser_click:") and rs[0].images == []  # nothing newer to show
    assert rs[2].content == page  # the page as it is now: whole


def test_the_reads_a_request_is_about_stay_whole_until_it_is_answered():
    fetch = [("web_fetch", {"url": f"https://air{i}.example"}, ToolResult("", "web_fetch", f"fare {i} " * 500)) for i in range(3)]
    earlier = _run(fetch[0], ask="how much is a flight to Goa?") + [Message.assistant("About 4,000.")]
    msgs = earlier + _run(*fetch, _act("click"), _act("click"), ask="compare these three airlines' fares")
    rs = _results(diet.diet(msgs))
    assert "chars of an older step" in rs[0].content  # the earlier request's read: cut
    assert [r.content for r in rs[1:4]] == [f[2].content for f in fetch]  # this request's: whole
    again = _run(fetch[0], _act("click"), fetch[0], _act("click"), _act("click"))
    assert _results(diet.diet(again))[0].content.startswith("[older web_fetch:")  # the same read, newer


def test_the_diet_is_pure_and_idempotent():
    msgs = _run(_shot("s1"), _ax("Chess", 1), _act(), _shot("s2"), _ax("Chess", 2), _act(), _act(), _shot("s3"))
    snapshot = [(m.text, [(r.content, list(r.images)) for r in m.tool_results]) for m in msgs]
    once = diet.diet(msgs)
    assert [(m.text, [(r.content, list(r.images)) for r in m.tool_results]) for m in msgs] == snapshot
    twice = diet.diet(once)
    assert [[(r.content, r.images) for r in m.tool_results] for m in twice] == [
        [(r.content, r.images) for r in m.tool_results] for m in once
    ]
    assert diet.cap(diet.cap(once, 160), 160) == diet.cap(once, 160)


def test_fit_puts_every_agent_request_on_the_diet_but_leaves_chat_alone():
    msgs = _run(_shot("s1"), _ax("Chess", 1), _act(), _shot("s2"), _act(), _act())
    tools = [ToolSpec("drag", "Drag.", {"type": "object", "properties": {}})]
    big = fit(msgs, tools=tools, budget=1_000_000, max_tokens=4096, vision=True)  # a large brain
    assert big.messages == diet.diet(msgs) and big.max_tokens == 4096 and big.tools == tools
    chat = fit(msgs, budget=1_000_000, max_tokens=4096, vision=True)  # no tools: a chat answer
    assert [r.content for r in _results(chat.messages)] == [r.content for r in _results(msgs)]


async def test_gemini_and_claude_steps_get_the_diet_too(monkeypatch):
    earlier = [
        Message.user("list my files"), Message.assistant("", [ToolCall("e0", "shell", {})]),
        Message.tool([ToolResult("e0", "shell", "file " * 300)]), Message.assistant("Done."),
    ]  # fmt: skip
    msgs = earlier + _run(_shot("s1"), _ax("Chess", 1), _act(), _shot("s2"), _act(), _act())
    tools = [ToolSpec("drag", "Drag.", {"type": "object", "properties": {}})]
    want = [r.content for r in _results(diet.diet(msgs))]
    gem = pytest.importorskip("neo2.core.providers.gemini")
    seen: list[list[Message]] = []
    real = gem._to_contents
    monkeypatch.setattr(gem, "_to_contents", lambda m, model, **kw: seen.append(m) or real(m, model, **kw))

    class Models:
        async def generate_content(self, **kw):
            return SimpleNamespace(candidates=[], usage_metadata=None)

    g = gem.GeminiProvider("gemini-x", keys=["k1"], fallback_models=[])
    monkeypatch.setattr(g, "_call", lambda fn: fn(SimpleNamespace(aio=SimpleNamespace(models=Models())), "gemini-x"))
    await g.complete(msgs, tools=tools)
    assert [r.content for r in _results(seen[-1])] == want

    from neo2.core.providers.claude import ClaudeProvider

    sent: list[dict] = []

    async def create(**kw):
        sent.append(kw)
        return SimpleNamespace(stop_reason="end_turn", content=[], usage=SimpleNamespace(input_tokens=1, output_tokens=1))

    c = ClaudeProvider("claude-x", api_key="sk-test")
    monkeypatch.setattr(c, "_client", SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create))))
    await c.complete(msgs, tools=tools)
    blocks = [b for m in sent[-1]["messages"] if isinstance(m["content"], list) for b in m["content"]]
    shown = [b["content"] for b in blocks if b.get("type") == "tool_result"]
    flat = json.dumps(shown)
    assert "older screenshot s1" in flat and "chars of an older step" in flat


# ---- what the task being worked on read stays whole ------------------------------------------------------


def _read(name: str, args: dict, body: str, ok: bool = True) -> tuple[str, dict, ToolResult]:
    return name, args, ToolResult("", name, body, ok=ok)


def test_everything_the_request_being_worked_on_read_stays_whole():
    """Emails read with applescript, files with `shell cat`, a listing worked through step by step:
    the request acts on them long after (the reply quotes email 1's ask; invoice 45 still has to
    move). Only an earlier request's results are cut short."""
    mails = [_read("applescript", {"script": f"content of message {i}"}, f"From: p{i}\n" + "news " * 300 + f"ASK-{i}")
             for i in range(3)]  # fmt: skip
    files = [_read("shell", {"command": f"cat q{i}.csv"}, "row,amount\n" + "a,1\n" * 400 + f"TOTAL-{i}") for i in range(3)]
    listing = _read("shell", {"command": "ls ~/Downloads"}, "\n".join(f"invoice-{k:02}.pdf" for k in range(60)))
    moves = [_read("shell", {"command": f"mv invoice-{k:02}.pdf Invoices/"}, "") for k in range(4)]
    earlier = _run(mails[0], ask="what did Priya say?") + [Message.assistant("She asked about Tuesday.")]
    msgs = earlier + _run(*mails, *files, listing, *moves, ask="answer these three emails, then file the invoices")
    rs, was = _results(diet.diet(msgs)), _results(msgs)
    assert "chars of an older step" in rs[0].content  # an earlier request's read: cut
    assert [r.content for r in rs[1:]] == [r.content for r in was[1:]]  # this request's: every one whole
    tools = [ToolSpec("shell", "Run it.", {"type": "object", "properties": {}})]
    f = fit(msgs, tools=tools, budget=131_072, max_tokens=8192, vision=True)
    assert [r.content for r in _results(f.messages)[1:]] == [r.content for r in was[1:]]


def test_an_answer_to_a_question_goes_on_with_the_request_that_asked():
    """"Which one?" — "the second one": the pages read before the question are what the answer is
    about. The user's answer starts no new task, and nothing the task read is cut."""
    names = ["Luigi's", "Trattoria Roma", "Pasta Bar"]
    pages = [_read("web_fetch", {"url": f"https://r{k}.example"}, f"{n}\n" + "menu " * 600 + f"\nBOOK: 555-01{k}{k}")
             for k, n in enumerate(names)]  # fmt: skip
    which = _read("ask_user", {}, "NEEDS_USER: ask them which: Luigi's, Trattoria Roma or Pasta Bar?")
    goal = "book a table for two tonight at a good Italian place nearby"
    asked = _run(_read("web_search", {"query": "italian near me"}, "results " * 300), *pages, which, ask=goal)
    later = [_read("open_url", {"url": "https://r1.example/book"}, "Opened it."), _act("click"), _act("click")]
    msgs = asked + _run(*later, ask="the second one")
    want = [p[2].content for p in pages]
    assert [r.content for r in _results(diet.diet(msgs))[1:4]] == want
    said = asked + [Message.assistant("Which one: Luigi's, Trattoria Roma or Pasta Bar?")] + _run(*later, ask="the second one")
    assert [r.content for r in _results(diet.diet(said))[1:4]] == want  # (the question written out)
    # …and so on back: an answer that asked again, answered in turn
    when = _read("ask_user", {}, "NEEDS_USER: ask them what time")
    again = asked + _run(later[0], when, ask="the second one") + _run(*later[1:], _act("click"), ask="8pm")
    assert [r.content for r in _results(diet.diet(again))[1:4]] == want
    # A question to confirm, answered in the user's own words ("no — send it to Dana instead")
    send = _read("mail_send", {"to": "priya@example.com"}, "NEEDS_CONFIRM: send the email to Priya — Should I go ahead?")
    other = _run(*pages, send, ask=goal) + _run(_act("click"), _act("click"), _act("click"), ask="send it to Dana instead")
    assert [r.content for r in _results(diet.diet(other))[:3]] == want
    # A request after a finished one (no question in between) is a new one: what that read, cut
    done = _run(*pages, ask=goal) + [Message.assistant("Booked.")] + _run(*later, ask="the second one")
    assert all("chars of an older step" in r.content for r in _results(diet.diet(done))[:3])


def test_a_confirmed_request_keeps_what_it_read_before_the_question():
    """Read Dana's email → a question (the calendar card) → "[User confirmed …]" → an error → open
    Mail → its tree: the reply still has Dana's proposed time in front of it."""
    goal = "reply to Dana's email and put the meeting in my calendar"
    mail = _read("applescript", {"script": "content of message 1"}, "From: Dana\n" + "hello " * 300 + "How about Thursday 3pm? " + "bye " * 100)
    card = _read("calendar_add", {"title": "Sync with Dana"}, "NEEDS_CONFIRM: add “Sync with Dana” — Should I go ahead?")
    msgs = _run(mail, card, ask=goal) + [Message.user(f"[User confirmed{naming(goal)}. The event was added.]")]
    msgs += _steps(_read("mail_reply", {}, "Error: Mail isn't open", ok=False), _read("open_app", {"target": "Mail"}, "Opened Mail."), _ax("Mail", 1))
    assert _results(diet.diet(msgs))[0].content == mail[2].content


def test_the_soft_budget_never_cuts_what_the_user_or_neo_said():
    """The user pasted an essay, NEO fixed it, then "type the fixed essay into Pages": both are what
    the request acts on, whatever the brain's room."""
    essay = "their going too the store tomorow, " * 260
    fixed = "They're going to the store tomorrow. " * 260
    msgs = [Message.user("fix the grammar:\n" + essay), Message.assistant(fixed)] + _run(
        _read("open_app", {"target": "Pages"}, "Opened Pages."), _ax("Pages", 1), _shot("s1"), _act("click", "Clicked e5 (Button) in Pages."),
        ask="type the fixed essay into a new Pages document",
    )  # fmt: skip
    system = "Rules. " * 3000
    tools = [ToolSpec("type_text", "Type it.", {"type": "object", "properties": {}})]
    assert estimate(diet.diet(msgs), system, tools) > diet.SCREEN_BUDGET
    for budget in (131_072, 1_000_000):  # Cerebras' room, Gemini's
        f = fit(msgs, system=system, tools=tools, budget=budget, max_tokens=8192, vision=True)
        assert f.messages[0].text == msgs[0].text and f.messages[1].text == fixed


def test_over_the_soft_budget_only_an_earlier_requests_results_get_shorter():
    """'compare these three laptops', then chess: every chess step carried the laptop pages and
    Safari's tree whole. Over the soft budget they are cut short; the task being worked on (its
    reads, its looks, its last steps) and what anyone said stay whole."""
    laptops = [
        ("browser_open", {"url": f"https://shop.example/laptop{k}"},
         ToolResult("", "browser_open", f"URL: https://shop.example/laptop{k}\nTitle: Laptop {k}\n" + "spec " * 1100 + f"PRICE-{k}"))
        for k in range(3)
    ]  # fmt: skip
    answer = "Laptop 1 is the cheapest, at $899."
    earlier = _run(*laptops, _ax("Safari", 7), ask="compare the prices of these three laptops") + [Message.assistant(answer)]
    mails = [_read("applescript", {"script": f"content of message {i}"}, "From: coach\n" + "opening " * 950) for i in range(2)]
    msgs = earlier + _run(*mails, _ax("Chess", 1), _shot("s1"), _act(), _act(), ask=GOAL)
    tools = [ToolSpec("drag", "Drag it. " * 50, {"type": "object", "properties": {}})]
    assert estimate(diet.diet(msgs), "", tools) > diet.SCREEN_BUDGET
    f = fit(msgs, tools=tools, budget=131_072, max_tokens=8192, vision=True, name="cerebras_vision")
    rs, was = _results(f.messages), _results(msgs)
    assert all(len(r.content) < 260 for r in rs[:4])  # the laptops' pages and Safari's tree: short
    assert [r.content for r in rs[4:]] == [r.content for r in was[4:]]  # this task's: whole
    assert len(rs[-3].images) == 1 and f.messages[len(earlier) - 1].text == answer
    assert f.tokens <= diet.SCREEN_BUDGET and f.max_tokens == 8192 and f.tools == tools
    # Over it and nothing left to shed: sent as it is, never refused (a brain's own room refuses).
    huge = _run(_ax("Chess", 1), ("read_file", {"path": "a"}, ToolResult("", "read_file", "z" * 90_000)))
    assert fit(huge, tools=tools, budget=131_072, max_tokens=8192, vision=True).tokens > diet.TASK_BUDGET


def test_the_users_latest_picture_and_the_newest_picture_of_each_view_stay():
    """'what lamp is this?' with a photo, then 'find that exact lamp on a shopping site': the photo is
    what the request is about. The desktop's newest picture and NEO's browser's both go."""
    lamp = [Message.user("what lamp is this?", images=[_pic(b"lamp")]), Message.assistant("An Anglepoise Type 75.")]
    page = ToolResult("", "browser_open", "URL: https://shop.example/q\nTitle: Lamps\n" + "lamp " * 200, images=[_pic(b"page")])
    bshot = ToolResult("", "browser_screenshot", "Screenshot of https://shop.example/q (1280×800).", images=[_pic(b"bshot")])
    msgs = lamp + _run(
        _shot("s1"), ("browser_open", {"url": "https://shop.example/q"}, page), ("browser_screenshot", {}, bshot), _act("click"),
        ask="find that exact lamp on a shopping site",
    )  # fmt: skip
    out = diet.diet(msgs)
    assert out[0].images == msgs[0].images  # the user's photo: still in view
    rs = _results(out)
    assert [len(r.images) for r in rs] == [1, 0, 1, 0]  # the desktop's newest, the browser's newest
    assert rs[1].content.endswith("a newer one follows]")  # (the browser's older picture)
    # A newer picture from the user replaces the lamp's
    more = msgs + [Message.assistant("Found it."), Message.user("and this chair?", images=[_pic(b"chair")])]
    out = diet.diet(more)
    assert out[0].images == [] and out[0].text.startswith("what lamp is this?") and out[-1].images == more[-1].images


def test_the_diet_is_idempotent_across_requests_questions_and_pictures():
    lamp = [Message.user("what lamp is this?", images=[_pic(b"lamp")]), Message.assistant("An Anglepoise.")]
    page = ToolResult("", "browser_open", "URL: https://shop.example/q\n" + "lamp " * 400, images=[_pic(b"page")])
    which = _read("ask_user", {}, "NEEDS_USER: ask them which size")
    msgs = lamp + _run(_read("shell", {"command": "ls"}, "file " * 500), ("browser_open", {"url": "q"}, page), _shot("s1"), which, ask="find it")
    msgs += _run(_ax("Safari", 1), _shot("s2"), _ax("Safari", 2), _act(), ask="the large one")
    once = diet.diet(msgs)
    twice = diet.diet(once)
    assert [(m.text, m.images, [(r.content, r.images) for r in m.tool_results]) for m in twice] == [
        (m.text, m.images, [(r.content, r.images) for r in m.tool_results]) for m in once
    ]
    assert diet.cap(diet.cap(once, 160), 160) == diet.cap(once, 160)
