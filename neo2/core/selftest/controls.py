"""The controls self-test: NEO checks its own mouse, keyboard and screen controls on this Mac.

How the user runs it: say "Neo, check your computer controls" (or type it in NEO's overlay). A
developer, from their own Terminal: ``python -m neo2.core.selftest controls`` (neo2/core/selftest/__main__.py,
which never runs as a command NEO itself runs).

What happens, in under a minute:

1. NEO opens its test window (neo2.core.selftest.testwin: a separate app it starts as its own child,
   top-left on the main display) *without* bringing it forward, so the first click lands on a
   window that isn't in front — the way the Chess board is when NEO starts playing.
2. It goes no further unless that window is what's under every point it will use: it never
   clicks, scrolls or types anywhere else (neo2.core.selftest.watch checks each event before it's
   posted), and it stops at once when the user moves the mouse or presses a key.
3. It drives the window only through NEO's own tools, as a model would — click at a point, right
   and double click, screenshot → coordinates → click (a region and the whole screen),
   ax_tree → click by id, ax_press, ax_set_value, type_text (Hindi and emoji, and two lines),
   hotkey (Return, the Test menu's ⌘J), scroll down and up, a drag on a Chess-like board, drag and
   drop, activate_app, a click on a second display when there is one — and compares what the
   window received with what NEO sent and with what the tool said it did.
4. It closes the window, lets go of anything still pressed, puts the pointer back and brings back
   the app that was in front — unless the user took the mouse or keyboard (during the run or as
   it ends: the watch listens until the clean-up is done) or another app came to the front: then
   the pointer and the front app are left as they are.

It doesn't start while another NEO job is at work or a mouse button is down: it needs the mouse
and keyboard to itself.

The report gives each check as passed or failed with what was sent, what NEO's tool answered and
what actually arrived; it is saved in NEO's data folder (``selftest/``).
"""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import io
import json
import re
import secrets
import shutil
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from neo2.core.selftest import layout as L
from neo2.core.selftest.mac import Child, Mac, Rect, contains
from neo2.core.selftest.watch import Stop, Watch

_BUDGET_S = 45.0  # no check starts after this; with one check's limit and the clean-up, < 60 s in all
_CHECK_S = 8.0
_READY_S = 12.0  # a cold Python + AppKit start
_WAIT_S = 1.0  # for a click to arrive
_TEXT_WAIT_S = 2.5
_DRAG_WAIT_S = 1.5  # a drag, and a drop (drag and drop runs its own loop in the app)
_POINT_SLACK = 1.5  # points: a click "lands where it was sent"
_SHOT_SLACK = 3.0  # points: screenshot → coordinates
_ASKED = "check your computer controls"


# ---- results ------------------------------------------------------------------------------------


@dataclass
class Check:
    name: str
    title: str
    status: str = "skip"  # pass | fail | skip | stopped
    sent: str = ""  # what NEO sent
    said: str = ""  # what NEO's tool answered
    arrived: str = ""  # what the test window received
    why: str = ""  # what a failure means
    ms: int = 0

    def ok(self, arrived: str) -> Check:
        self.status, self.arrived = "pass", arrived
        return self

    def fail(self, arrived: str, why: str = "") -> Check:
        self.status, self.arrived, self.why = "fail", arrived, why
        return self

    def skip(self, why: str) -> Check:
        self.status, self.arrived = "skip", why
        return self

    def line(self) -> str:
        mark = {"pass": "PASS", "fail": "FAIL", "skip": "skip", "stopped": "stopped"}[self.status]
        out = [f"[{mark}] {self.title}: {self.arrived}"]
        if self.sent:
            out.append(f"    sent: {self.sent}")
        if self.said:
            out.append(f"    NEO's tool said: {_one_line(self.said)}")
        if self.why:
            out.append(f"    why it matters: {self.why}")
        return "\n".join(out)


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)
    refused: str = ""  # didn't run: why
    stopped: str = ""  # stopped part-way: why
    info: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def passed(self) -> list[Check]:
        return [c for c in self.checks if c.status == "pass"]

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status == "fail"]

    @property
    def ok(self) -> bool:
        return not self.refused and not self.stopped and not self.failed

    def summary(self) -> str:
        if self.refused:
            return f"I didn't run the controls check: {self.refused}"
        ran = [c for c in self.checks if c.status in ("pass", "fail")]
        head = f"Controls check: {len(self.passed)} of {len(ran)} passed"
        if self.stopped:
            head += f" before I stopped ({self.stopped})"
        return head + "."

    def reply(self, saved: Path | None = None) -> str:
        """What NEO says and shows: the summary, what didn't work, what did, where the rest is."""
        parts = [self.summary()]
        if self.failed:
            parts.append("Didn't work:")
            for c in self.failed:
                said = f" NEO's tool said: “{_one_line(c.said)[:90]}”." if c.said and c.name != "feedback" else ""
                parts.append(f"- {c.title}: {c.arrived}.{said}")
        if self.passed:
            parts.append("Worked: " + ", ".join(c.title[0].lower() + c.title[1:] for c in self.passed) + ".")
        skipped = [c for c in self.checks if c.status == "skip"]
        if skipped:
            parts.append("Skipped: " + "; ".join(f"{c.title.lower()} ({c.arrived})" for c in skipped) + ".")
        if saved is not None:
            parts.append(f"Full report: {_home(saved)}")
        return "\n".join(parts)

    def text(self) -> str:
        lines = [self.summary(), ""]
        for k, v in self.info.items():
            lines.append(f"{k}: {v}")
        lines.append(f"took: {self.seconds:.1f} s")
        lines.append("")
        lines += [c.line() for c in self.checks]
        return "\n".join(lines) + "\n"

    def to_json(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "ok": self.ok,
            "refused": self.refused,
            "stopped": self.stopped,
            "seconds": round(self.seconds, 2),
            "info": self.info,
            "checks": [asdict(c) for c in self.checks],
        }


def save(report: Report) -> Path | None:
    """The report in NEO's data folder (selftest/controls-<time>.txt and .json)."""
    try:
        from neo2.core.config import settings

        folder = settings().data_dir / "selftest"
        folder.mkdir(parents=True, exist_ok=True)
        stem = folder / time.strftime("controls-%Y%m%d-%H%M%S")
        stem.with_suffix(".txt").write_text(report.text(), encoding="utf-8")
        stem.with_suffix(".json").write_text(json.dumps(report.to_json(), ensure_ascii=False, indent=1), encoding="utf-8")
        for p in (stem.with_suffix(".txt"), stem.with_suffix(".json")):
            p.chmod(0o600)
        return stem.with_suffix(".txt")
    except OSError:
        return None


def _home(p: Path) -> str:
    try:
        return "~/" + str(p.relative_to(Path.home()))
    except ValueError:
        return str(p)


def _one_line(s: str) -> str:
    return " ".join(str(s).split())[:200]


# ---- the window's log ---------------------------------------------------------------------------


class LogReader:
    """The test window's JSON lines, read as the file grows."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.events: list[dict] = []
        self._pos = 0
        self._tail = b""

    def poll(self) -> None:
        try:
            with open(self.path, "rb") as f:
                f.seek(self._pos)
                data = f.read()
        except OSError:
            return
        self._pos += len(data)
        lines = (self._tail + data).split(b"\n")
        self._tail = lines.pop()
        for raw in lines:
            try:
                ev = json.loads(raw)
            except ValueError:
                continue
            if isinstance(ev, dict):
                self.events.append(ev)

    def last(self, name: str) -> dict | None:
        self.poll()
        return next((e for e in reversed(self.events) if e.get("ev") == name), None)


# ---- one run ------------------------------------------------------------------------------------


class _Run:
    """What the checks share: the window, its log, NEO's tools, the watch."""

    def __init__(self, mac: Any, child: Child, reader: LogReader, ready: dict, watch: Watch, bounds: Rect) -> None:
        self.mac, self.child, self.reader, self.ready, self.watch = mac, child, reader, ready, watch
        self.pid = child.pid
        self.bounds = bounds  # the main window, as the window list has it
        self.targets: dict[str, Rect] = {k: tuple(v) for k, v in (ready.get("targets") or {}).items()}  # type: ignore[misc]
        self.ids: dict[str, str] = {}
        self.taint: list[str] = []
        self.can_capture = False

    # the window
    def center(self, name: str) -> tuple[float, float]:
        x, y, w, h = self.targets[name]
        return x + w / 2, y + h / 2

    def cell(self, col: int, row: int) -> tuple[float, float]:
        x, y, _, _ = self.targets["board"]
        cell = float(self.ready.get("cell", L.CELL))
        return x + (col + 0.5) * cell, y + (row + 0.5) * cell

    def front_pid(self) -> int:
        return int(self.mac.front()[1])

    # the log
    def mark(self) -> int:
        self.reader.poll()
        return len(self.reader.events)

    def since(self, mark: int, name: str | None = None, **match: Any) -> list[dict]:
        self.reader.poll()
        return [
            e
            for e in self.reader.events[mark:]
            if (name is None or e.get("ev") == name) and all(e.get(k) == v for k, v in match.items())
        ]

    def state(self) -> dict:
        return self.reader.last("state") or {}

    async def wait(self, mark: int, pred: Callable[[dict], bool], timeout: float | None = None) -> dict | None:
        end = time.monotonic() + (_WAIT_S if timeout is None else timeout)
        while True:
            self.reader.poll()
            for e in self.reader.events[mark:]:
                if pred(e):
                    return e
            if time.monotonic() >= end or self.watch.stopped or self.watch.refused or not self.child.alive():
                return None
            await asyncio.sleep(0.03)

    async def wait_state(self, pred: Callable[[dict], bool], timeout: float | None = None) -> dict | None:
        end = time.monotonic() + (_TEXT_WAIT_S if timeout is None else timeout)
        while True:
            s = self.state()
            if s and pred(s):
                return s
            if time.monotonic() >= end or self.watch.stopped or self.watch.refused or not self.child.alive():
                return None
            await asyncio.sleep(0.03)

    async def wait_front(self, pred: Callable[[int], bool], timeout: float = 1.5) -> bool:
        end = time.monotonic() + timeout
        while True:
            if pred(await asyncio.to_thread(self.front_pid)):
                return True
            if time.monotonic() >= end:
                return False
            await asyncio.sleep(0.05)

    # NEO's tools, the way the agent calls them
    async def tool(self, tool_name: str, /, **args: Any):
        from neo2.core.agent import confirm
        from neo2.core.agent.registry import ToolContext, ToolOutput, registry

        if tool_name in _KEY_TOOLS:
            # Keys go to the test window only. NEO's key tools bring the app it last acted on back to
            # the front first — here, over an app that came forward meanwhile (the user's ⌘Tab, which
            # a mouse-only tap can't hear): the test stops instead of taking the keyboard back.
            name, pid = await asyncio.to_thread(self.mac.front)
            if pid != self.child.pid:
                self.watch.refused = f"keys would have gone to {name or 'another app'}, which is in front"
                return ToolOutput(f"Error: {self.watch.refused} — not sent.", ok=False)
        out = await registry().invoke(tool_name, args, ToolContext(user_text=_ASKED, taint=self.taint, origin="user"))
        if out.staged and (p := confirm.peek()) is not None and p.action_id == out.staged:
            confirm.cancel("the controls self-test never presses anything that needs a yes")
        return out


# The tools that send keys to whatever is in front (after bringing the app NEO last acted on forward).
_KEY_TOOLS = frozenset({"type_text", "dictate", "hotkey"})


# ---- the checks ---------------------------------------------------------------------------------


def _cell_str(cell: Any) -> str:
    return f"{cell[0]},{cell[1]}" if isinstance(cell, list) and len(cell) == 2 else "(off the board)"


def _off(ev: dict, x: float, y: float) -> tuple[float, float]:
    at = ev.get("at") or [x, y]
    return float(at[0]) - x, float(at[1]) - y


async def _first_click(r: _Run) -> Check:
    c = Check("first_click", "First click on a window that isn't in front")
    col, row = 6, 1
    x, y = r.cell(col, row)
    in_front = r.front_pid() == r.pid
    m = r.mark()
    out = await r.tool("click", x=x, y=y)
    c.sent = f"click at ({x:.0f}, {y:.0f}), board cell {col},{row}" + (" (the window was already in front)" if in_front else "")
    c.said = out.text
    hit = await r.wait(m, lambda e: e.get("ev") == "board_down")
    came = bool(r.since(m, "focus", active=True)) or r.front_pid() == r.pid
    if hit is not None and hit.get("cell") == [col, row]:
        return c.ok(f"the board got the click on cell {col},{row}" + ("; the window came to the front" if came and not in_front else ""))
    if hit is not None:
        return c.fail(f"the click landed on cell {_cell_str(hit.get('cell'))} instead of {col},{row}", "clicks land off target")
    if r.since(m, "raw_ldown"):
        return c.fail(
            "the app got the click, but it only brought the window to the front; the board never saw it",
            "an app that isn't in front has to be brought forward before the click, or the first click is lost "
            "(Chess's board works like this)",
        )
    if came:
        return c.fail("the window came to the front, but no click arrived")
    return c.fail("nothing arrived")


async def _click_point(r: _Run) -> Check:
    c = Check("click_point", "Click at a point")
    col, row = 2, 2
    x, y = r.cell(col, row)
    m = r.mark()
    out = await r.tool("click", x=x, y=y)
    c.sent, c.said = f"click at ({x:.0f}, {y:.0f}), board cell {col},{row}", out.text
    hit = await r.wait(m, lambda e: e.get("ev") == "board_down" and e.get("button") == "left")
    if hit is None:
        return c.fail("nothing arrived")
    dx, dy = _off(hit, x, y)
    where = f"cell {_cell_str(hit.get('cell'))}, {dx:+.1f}, {dy:+.1f} pt from the point"
    if hit.get("cell") != [col, row] or abs(dx) > _POINT_SLACK or abs(dy) > _POINT_SLACK:
        return c.fail(f"it landed on {where}", "clicks land off target")
    return c.ok(f"it landed on {where}")


async def _right_click(r: _Run) -> Check:
    c = Check("right_click", "Right click")
    col, row = 5, 5
    x, y = r.cell(col, row)
    m = r.mark()
    out = await r.tool("click", x=x, y=y, button="right")
    c.sent, c.said = f"right click at ({x:.0f}, {y:.0f})", out.text
    hit = await r.wait(m, lambda e: e.get("ev") == "board_down")
    if hit is None:
        return c.fail("nothing arrived")
    if hit.get("button") != "right":
        return c.fail(f"it arrived as a {hit.get('button')} click")
    return c.ok(f"a right click arrived on cell {_cell_str(hit.get('cell'))}")


async def _double_click(r: _Run) -> Check:
    c = Check("double_click", "Double click")
    x, y = r.center("double")
    m = r.mark()
    out = await r.tool("click", x=x, y=y, count=2)
    c.sent, c.said = f"double click at ({x:.0f}, {y:.0f})", out.text
    hit = await r.wait(m, lambda e: e.get("ev") == "double_down" and e.get("clicks") == 2)
    if hit is not None:
        return c.ok("the box got a double click (click count 2)")
    counts = [e.get("clicks") for e in r.since(m, "double_down")]
    if counts:
        return c.fail(f"it got separate clicks (click counts {counts}), not a double click")
    return c.fail("nothing arrived")


_ID_LINE = re.compile(r"^\s*(e\d+)\s")


async def _ax_tree(r: _Run) -> Check:
    from neo2.core.tools.computer import ax

    c = Check("ax_tree", "Read the window with ax_tree")
    if r.front_pid() != r.pid:
        return c.skip("the test window isn't in front, and ax_tree reads the app in front")
    out = await r.tool("ax_tree")
    c.sent, c.said = "ax_tree (the app in front)", out.text.splitlines()[0] if out.text else ""
    ids: dict[str, str] = {}
    for line in out.text.splitlines():
        m = _ID_LINE.match(line)
        e = ax.get(m.group(1)) if m else None
        if e is None or await asyncio.to_thread(r.mac.element_pid, e.ref) != r.pid:
            continue
        name = (
            "button_a" if e.role == "AXButton" and e.title == L.BUTTON_A
            else "button_b" if e.role == "AXButton" and e.title == L.BUTTON_B
            else "field" if e.role == "AXTextField"
            else "area" if e.role == "AXTextArea"
            else ""
        )  # fmt: skip
        if name and name not in ids:
            ids[name] = e.id
    r.ids = ids
    missing = [n for n in ("button_a", "button_b", "field", "area") if n not in ids]
    if missing:
        return c.fail(f"no ids for {', '.join(missing)} (found {', '.join(ids) or 'nothing'})")
    off = []
    for name in ("button_a", "button_b", "field", "area"):
        e = ax.get(ids[name])
        if e is not None and not contains(r.targets[name], *e.center, 1.0):  # where a click by id goes
            tx, ty, tw, th = r.targets[name]
            off.append(f"{name}'s center ({e.center[0]:.0f},{e.center[1]:.0f}) is outside ({tx:.0f},{ty:.0f} {tw:.0f}×{th:.0f})")
    if off:
        return c.fail("positions differ from the window's own: " + "; ".join(off), "clicks by id would land off target")
    return c.ok("ids for Alpha, Bravo, the text field and the text area, each centered on the real control")


async def _click_id(r: _Run) -> Check:
    c = Check("click_id", "Click a button by its ax_tree id")
    eid = r.ids.get("button_a")
    if not eid:
        return c.skip("ax_tree gave no id for the Alpha button")
    m = r.mark()
    out = await r.tool("click", id=eid)
    c.sent, c.said = f"click id={eid} (Alpha)", out.text
    if await r.wait(m, lambda e: e.get("ev") == "button" and e.get("name") == L.BUTTON_A):
        return c.ok("Alpha was pressed")
    return c.fail("Alpha wasn't pressed")


async def _ax_press(r: _Run) -> Check:
    c = Check("ax_press", "Press a button with ax_press")
    eid = r.ids.get("button_b")
    if not eid:
        return c.skip("ax_tree gave no id for the Bravo button")
    m = r.mark()
    out = await r.tool("ax_press", id=eid)
    c.sent, c.said = f"ax_press id={eid} (Bravo)", out.text
    if await r.wait(m, lambda e: e.get("ev") == "button" and e.get("name") == L.BUTTON_B):
        return c.ok("Bravo was pressed")
    return c.fail("Bravo wasn't pressed")


async def _ax_set_value(r: _Run) -> Check:
    c = Check("ax_set_value", "Set the text area with ax_set_value")
    eid = r.ids.get("area")
    if not eid:
        return c.skip("ax_tree gave no id for the text area")
    out = await r.tool("ax_set_value", id=eid, value=L.SET_TEXT)
    c.sent, c.said = f"ax_set_value id={eid} to {L.SET_TEXT!r}", out.text
    if await r.wait_state(lambda s: s.get("area") == L.SET_TEXT):
        return c.ok("the text area reads exactly that")
    return c.fail(f"the text area reads {r.state().get('area', '')!r}")


def _chunks(sent: list[dict]) -> str:
    """The typed text as NEO sent it: characters and UTF-16 units per key event."""
    texts = [e["text"] for e in sent if e.get("kind") == "key_down" and e.get("text")]
    if not texts:
        return ""
    parts = [f"{len(t)} chars/{len(t.encode('utf-16-le')) // 2} UTF-16 units" for t in texts]
    return f"{len(texts)} key event(s): " + ", ".join(parts)


def _diff(want: str, got: str) -> str:
    sm = difflib.SequenceMatcher(a=want, b=got, autojunk=False)
    missing = "".join(want[i1:i2] for op, i1, i2, _, _ in sm.get_opcodes() if op in ("delete", "replace"))
    extra = "".join(got[j1:j2] for op, _, _, j1, j2 in sm.get_opcodes() if op in ("insert", "replace"))
    bits = []
    if missing:
        bits.append(f"missing {missing!r}")
    if extra:
        bits.append(f"extra {extra!r}")
    return "; ".join(bits)


async def _type_field(r: _Run) -> Check:
    c = Check("type_field", "Type Hindi and emoji into a text field")
    eid = r.ids.get("field")
    if not eid:
        return c.skip("ax_tree gave no id for the text field")
    await r.tool("click", id=eid)  # the caret goes into the field
    await asyncio.sleep(0.15)
    n = len(r.watch.sent)
    out = await r.tool("type_text", text=L.TYPE_TEXT)
    c.said = out.text
    c.sent = f"type_text {L.TYPE_TEXT!r}" + (f" as {_chunks(r.watch.sent[n:])}" if _chunks(r.watch.sent[n:]) else "")
    if await r.wait_state(lambda s: s.get("field") == L.TYPE_TEXT):
        return c.ok("the field reads exactly what was typed")
    got = str(r.state().get("field", ""))
    return c.fail(f"the field reads {got!r} ({_diff(L.TYPE_TEXT, got)})", "text comes out different from what NEO meant to type")


async def _hotkey_return(r: _Run) -> Check:
    c = Check("hotkey_return", "Press Return with hotkey")
    m = r.mark()
    out = await r.tool("hotkey", keys="return")
    c.sent, c.said = "hotkey return (in the text field)", out.text
    if await r.wait(m, lambda e: e.get("ev") == "field_return"):
        return c.ok("the field got its Return")
    if r.since(m, "raw_key_down"):
        return c.fail("a key arrived, but the field didn't take it as Return")
    return c.fail("no key arrived")


async def _type_area(r: _Run) -> Check:
    c = Check("type_area", "Type two lines into a text area")
    eid = r.ids.get("area")
    if not eid:
        return c.skip("ax_tree gave no id for the text area")
    await r.tool("click", id=eid)
    await asyncio.sleep(0.15)
    out = await r.tool("type_text", text=L.AREA_TEXT)
    c.sent, c.said = f"type_text {L.AREA_TEXT!r}", out.text
    if await r.wait_state(lambda s: L.AREA_TEXT in str(s.get("area", ""))):
        return c.ok("both lines arrived, with a real line break")
    return c.fail(f"the text area reads {r.state().get('area', '')!r}")


async def _hotkey_menu(r: _Run) -> Check:
    c = Check("hotkey_menu", "A menu shortcut with hotkey")
    m = r.mark()
    out = await r.tool("hotkey", keys=L.MENU_KEYS)
    c.sent, c.said = f"hotkey {L.MENU_KEYS} (Test › {L.MENU_TITLE})", out.text
    if await r.wait(m, lambda e: e.get("ev") == "menu"):
        return c.ok(f"the menu's {L.MENU_TITLE} ran")
    keys = r.since(m, "raw_key_down")
    if keys:
        k = keys[-1]
        return c.fail(f"a key arrived ({k.get('chars')!r}, modifiers {k.get('mods', 0):#x}) but the menu didn't run")
    return c.fail("no key arrived")


async def _scroll(r: _Run) -> Check:
    c = Check("scroll", "Scroll a list down and back up")
    x, y = r.center("list")
    start = float(r.state().get("list", 0.0))
    m = r.mark()
    out = await r.tool("scroll", x=x, y=y, dy=-5)
    c.sent, c.said = f"scroll at ({x:.0f}, {y:.0f}) dy=-5, then dy=5", out.text
    down = await r.wait_state(lambda s: float(s.get("list", 0)) > start + 1)
    if down is None:
        got = r.since(m, "raw_scroll")
        if got:
            return c.fail(f"the app got a scroll event (delta {got[-1].get('dy')}), but the list didn't move down")
        return c.fail("no scroll arrived")
    lower = float(down["list"])
    back = await r.tool("scroll", x=x, y=y, dy=5)
    c.said += f" / {back.text}"
    up = await r.wait_state(lambda s: float(s.get("list", 0)) < lower - 1)
    if up is None:
        return c.fail(f"it went down {lower - start:.0f} pt but didn't come back up")
    return c.ok(f"down {lower - start:.0f} pt, then up {lower - float(up['list']):.0f} pt")


async def _drag_board(r: _Run) -> Check:
    c = Check("drag_board", "Drag a piece on a board")
    piece = r.state().get("piece") or r.ready.get("piece") or [1, 6]
    end = [4, 3]
    (x1, y1), (x2, y2) = r.cell(*piece), r.cell(*end)
    m = r.mark()
    out = await r.tool("drag", x1=x1, y1=y1, x2=x2, y2=y2)
    c.sent, c.said = f"drag ({x1:.0f}, {y1:.0f}) → ({x2:.0f}, {y2:.0f}), cell {_cell_str(piece)} → {_cell_str(end)}", out.text
    moved = await r.wait(m, lambda e: e.get("ev") == "piece_moved", _DRAG_WAIT_S)
    if moved is not None and moved.get("end") == end:
        return c.ok(f"the piece moved from {_cell_str(piece)} to {_cell_str(end)}")
    if moved is not None:
        return c.fail(f"the piece moved to {_cell_str(moved.get('end'))} instead")
    if not r.since(m, "board_down"):
        return c.fail("the board got no button-down")
    if not r.since(m, "board_drag"):
        return c.fail("the board saw a click, not a drag: no drag events arrived", "games and canvases can't be played by dragging")
    ups = r.since(m, "board_up")
    return c.fail(f"the drag ended on cell {_cell_str(ups[-1].get('cell')) if ups else '(none)'}")


async def _drag_drop(r: _Run) -> Check:
    c = Check("drag_drop", "Drag and drop onto a target")
    (x1, y1), (x2, y2) = r.center("tile"), r.center("well")
    m = r.mark()
    out = await r.tool("drag", x1=x1, y1=y1, x2=x2, y2=y2)
    c.sent, c.said = f"drag ({x1:.0f}, {y1:.0f}) → ({x2:.0f}, {y2:.0f}), tile → well", out.text
    drop = await r.wait(m, lambda e: e.get("ev") == "drop", _DRAG_WAIT_S)
    if drop is not None and drop.get("payload") == L.DRAG_PAYLOAD:
        return c.ok("the well got the drop")
    if drop is not None:
        return c.fail(f"something else was dropped ({drop.get('payload')!r})")
    steps = []
    if r.since(m, "tile_down"):
        steps.append("the tile was pressed")
    if r.since(m, "dnd_begin"):
        steps.append("a drag began")
    if r.since(m, "dnd_enter"):
        steps.append("it reached the well")
    ends = r.since(m, "dnd_end")
    if ends:
        steps.append(f"it ended without a drop (operation {ends[-1].get('operation')})")
    return c.fail(
        (", ".join(steps) or "nothing arrived") + ", but nothing was dropped",
        "drag and drop needs the button held a moment and the pointer moved in steps before a drop target notices",
    )


# screenshots


def find_marker(png: bytes, size_px: float | None = None) -> tuple[float, float, float, float] | None:
    """The test window's magenta square in a screenshot: (center x, center y, width, height) in
    image pixels, or None. `size_px`: its expected width, when known."""
    import numpy as np
    from PIL import Image

    a = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.int16)
    mask = (a[..., 0] > 170) & (a[..., 2] > 170) & (a[..., 1] < 110) & (np.abs(a[..., 0] - a[..., 2]) < 70)
    count = int(mask.sum())
    if count == 0 or count > 250_000:
        return None
    try:
        from scipy import ndimage

        labels, _ = ndimage.label(mask)
        boxes = [(i + 1, s) for i, s in enumerate(ndimage.find_objects(labels)) if s is not None]
    except ImportError:  # pragma: no cover - scipy is locked, but the square is usually alone
        ys, xs = np.nonzero(mask)
        labels = mask.astype(np.int32)
        boxes = [(1, (slice(int(ys.min()), int(ys.max()) + 1), slice(int(xs.min()), int(xs.max()) + 1)))]
    best: tuple[float, tuple[float, float, float, float]] | None = None
    for label, (sy, sx) in boxes:
        h, w = sy.stop - sy.start, sx.stop - sx.start
        if w < 6 or h < 6 or abs(w - h) > max(3, 0.2 * max(w, h)):
            continue
        if float((labels[sy, sx] == label).mean()) < 0.85:
            continue
        if size_px and not (0.6 * size_px <= w <= 1.5 * size_px):
            continue
        score = abs(w - size_px) if size_px else -float(w * h)
        found = (sx.start + w / 2, sy.start + h / 2, float(w), float(h))
        if best is None or score < best[0]:
            best = (score, found)
    return best[1] if best else None


# A screenshot's note (neo2.core.tools.computer.pointer.note): its shot id — a model clicks by pixels of the
# image with shot="…", and NEO maps them — and, for clicks without it, the image→screen mapping.
_NOTE_SHOT = re.compile(r'shot="([^"\s]+)"')
_NUM = r"(-?\d+(?:\.\d+)?)"
_NOTE_MAP = re.compile(rf"screen_x = {_NUM} \+ image_x × {_NUM}, screen_y = {_NUM} \+ image_y × {_NUM}")
_NOTE_SAME = "image pixels are also screen points"


def note_shot(note: str) -> str | None:
    """The shot id a screenshot's note gives (None: it gives none to click by)."""
    m = _NOTE_SHOT.search(note)
    return m.group(1) if m else None


def model_point(note: str, ix: float, iy: float) -> tuple[float, float] | None:
    """The screen point a screenshot's note says image pixel (ix, iy) is (None: it doesn't say)."""
    if m := _NOTE_MAP.search(note):
        return float(m.group(1)) + ix * float(m.group(2)), float(m.group(3)) + iy * float(m.group(4))
    if _NOTE_SAME in note:
        return ix, iy
    return None


def note_scale(note: str) -> float | None:
    """Screen points per image pixel, as the note says."""
    if m := _NOTE_MAP.search(note):
        return float(m.group(2))
    return 1.0 if _NOTE_SAME in note else None


async def _shot(r: _Run, c: Check, region: tuple[int, int, int, int] | None) -> Check:
    from neo2.core.tools.computer import pointer

    if not r.can_capture:
        return c.skip("Screen Recording is off for NEO")
    args = dict(zip(("x", "y", "w", "h"), region, strict=True)) if region else {}
    out = await r.tool("screenshot", **args)
    c.said = out.text
    if not out.ok or not out.images:
        return c.fail(f"no screenshot: {_one_line(out.text)}")
    sid = note_shot(out.text)
    if sid is None:
        return c.fail("the screenshot's note gives no shot id to click by", "a model can't click what a screenshot shows")
    scale = note_scale(out.text)
    found = find_marker(out.images[0].data, (L.MARKER_PT / scale) if scale else None)
    tx, ty = r.center("marker")
    if found is None:
        return c.fail("the magenta marker isn't in the screenshot")
    ix, iy = found[0], found[1]
    shot = pointer.get_shot(sid)
    if shot is None:
        return c.fail(f"NEO no longer knows screenshot {sid}", "a model's click by shot id would be refused")
    px, py = shot.to_screen(ix, iy)  # where NEO puts a click at those pixels of that shot
    said = model_point(out.text, ix, iy)
    c.sent = f"screenshot{' of ' + str(region) if region else ''}; marker at image ({ix:.0f}, {iy:.0f}) → click shot={sid} ({px:.0f}, {py:.0f})"
    if said is not None and (abs(said[0] - px) > 1.0 or abs(said[1] - py) > 1.0):
        return c.fail(
            f"the note says image ({ix:.0f}, {iy:.0f}) is ({said[0]:.0f}, {said[1]:.0f}) on screen, but NEO maps it to ({px:.0f}, {py:.0f})",
            "a model that clicks by the note's numbers misses",
        )
    if abs(px - tx) > _SHOT_SLACK or abs(py - ty) > _SHOT_SLACK:
        return c.fail(
            f"a click on those pixels of {sid} would land at ({px:.0f}, {py:.0f}), but the marker is at ({tx:.0f}, {ty:.0f})"
            + (f" (1 image px = {scale:.2f} pt)" if scale else ""),
            "clicks read off a screenshot miss",
        )
    if not contains(r.bounds, px, py, 1.0):
        return c.fail(f"the point ({px:.0f}, {py:.0f}) isn't in the test window, so I didn't click it")
    m = r.mark()
    click = await r.tool("click", x=round(ix, 1), y=round(iy, 1), shot=sid)  # as a model does: pixels + shot id
    c.said += f" / {click.text}"
    hit = await r.wait(m, lambda e: e.get("ev") == "marker_down")
    if hit is None:
        return c.fail(f"the point was right ({px - tx:+.1f}, {py - ty:+.1f} pt), but the click didn't arrive")
    return c.ok(f"found at image ({ix:.0f}, {iy:.0f}) of {sid} → ({px:.0f}, {py:.0f}), {px - tx:+.1f}, {py - ty:+.1f} pt off; the click landed on it")


async def _shot_region(r: _Run) -> Check:
    c = Check("screenshot_region", "Screenshot of the window, then click what it shows")
    x, y, w, h = r.bounds
    return await _shot(r, c, (int(round(x)), int(round(y)), int(round(w)), int(round(h))))


async def _shot_full(r: _Run) -> Check:
    c = Check("screenshot_full", "Full-screen screenshot, then click what it shows")
    return await _shot(r, c, None)


async def _second_display(r: _Run) -> Check:
    c = Check("second_display", "Click on a second display")
    second = r.ready.get("second")
    if not second:
        return c.skip("there's one display")
    x, y, w, h = second["target"]
    x, y = x + w / 2, y + h / 2
    m = r.mark()
    out = await r.tool("click", x=x, y=y)
    c.sent, c.said = f"click at ({x:.0f}, {y:.0f})", out.text
    if await r.wait(m, lambda e: e.get("ev") == "second_down"):
        return c.ok("it landed on the second display's window (NEO's full-screen screenshot shows only the main display)")
    return c.fail("nothing arrived on the second display", "windows on another display can't be clicked")


async def _activate(r: _Run) -> Check:
    c = Check("activate", "Bring the window forward with activate_app")
    name = str(r.ready.get("name") or "")
    pids = await asyncio.to_thread(r.mac.pids_named, name) if name else []
    if pids != [r.pid]:
        return c.skip(f"macOS doesn't know the test window by the name {name!r} alone")
    r.child.send("deactivate")
    if not await r.wait_front(lambda pid: pid != r.pid, 1.5):
        return c.skip("the test window wouldn't step back")
    m = r.mark()
    out = await r.tool("activate_app", name=name)
    c.sent, c.said = f"activate_app {name!r}", out.text
    if await r.wait_front(lambda pid: pid == r.pid, 1.5):
        return c.ok("the test window came to the front")
    return c.fail("it stayed behind" + (" (the app saw an activation)" if r.since(m, "focus", active=True) else ""))


CHECKS: tuple[Callable[[_Run], Awaitable[Check]], ...] = (
    _first_click,
    _click_point,
    _right_click,
    _double_click,
    _ax_tree,
    _click_id,
    _ax_press,
    _ax_set_value,
    _type_field,
    _hotkey_return,
    _type_area,
    _hotkey_menu,
    _scroll,
    _drag_board,
    _drag_drop,
    _shot_region,
    _shot_full,
    _second_display,
    _activate,
)


def _feedback(checks: list[Check]) -> Check:
    """Did NEO's tools say they worked when they didn't? (The Chess run: every call "ok", the game
    never moved.)"""
    c = Check("feedback", "NEO's tools report failures")
    liars = [ch for ch in checks if ch.status == "fail" and ch.said and not ch.said.lstrip().startswith(("Error", "NEEDS_"))]
    if liars:
        names = ", ".join(ch.title.lower() for ch in liars)
        return c.fail(
            f"{len(liars)} tool answer(s) sounded like success though it didn't work ({names})",
            "a model that reads only that answer thinks it worked and moves on",
        )
    failed = [ch for ch in checks if ch.status == "fail"]
    return c.ok("every step that failed also said so" if failed else "nothing failed")


async def _one(fn: Callable[[_Run], Awaitable[Check]], r: _Run) -> Check:
    t0 = time.monotonic()
    name = fn.__name__.lstrip("_")
    try:
        c = await asyncio.wait_for(fn(r), _CHECK_S)
    except Stop as e:
        c = Check(name, name.replace("_", " ").capitalize(), "stopped", arrived=str(e))
    except TimeoutError:
        c = Check(name, name.replace("_", " ").capitalize()).fail(f"no answer within {_CHECK_S:.0f} s")
    except Exception as e:  # noqa: BLE001 - one broken check must not stop the clean-up
        c = Check(name, name.replace("_", " ").capitalize()).fail(f"the check broke: {e!r}"[:200])
    if (r.watch.stopped or r.watch.refused) and c.status in ("pass", "fail"):
        # Cut short: what arrived may be the watch letting go of NEO's button (a drag's end the
        # tool never sent), not the tool's own doing — neither a pass nor a failure of the tool.
        c.status, c.why = "stopped", ""
        c.arrived = f"stopped: {r.watch.stopped or r.watch.refused}"
    c.ms = int((time.monotonic() - t0) * 1000)
    return c


# ---- the run ------------------------------------------------------------------------------------


async def _wait_ready(reader: LogReader, child: Child, timeout: float) -> tuple[dict | None, str]:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        reader.poll()
        for e in reader.events:
            if e.get("ev") == "ready":
                return e, ""
            if e.get("ev") == "crash":
                return None, f"the test window failed to start ({e.get('error', '')[:120]})"
        if not child.alive():
            return None, "the test window quit before it was ready"
        await asyncio.sleep(0.05)
    return None, f"the test window didn't open within {timeout:.0f} s"


def _identity(mac: Any, child: Child, ready: dict) -> tuple[dict[int, Rect], str]:
    """The test window's numbers and bounds on screen, from the window list (not its own word) —
    or why they can't be trusted."""
    if int(ready.get("pid", 0)) != child.pid:
        return {}, "the window that reported in isn't the one NEO started"
    wins: dict[int, Rect] = {}
    numbers = [int(ready.get("window", 0))]
    if ready.get("second"):
        numbers.append(int(ready["second"].get("window", 0)))
    for n in numbers:
        w = mac.window(n)
        if w is None or w["pid"] != child.pid:
            return {}, "the test window isn't on screen as NEO's own child"
        wins[n] = w["bounds"]
    frame = ready.get("frame") or [0, 0, 0, 0]
    main = wins[numbers[0]]
    if any(abs(a - b) > 2 for a, b in zip(main, frame, strict=False)):
        return {}, f"the test window isn't where it says it is ({main} on screen, {frame} by its own account)"
    return wins, ""


def _points(ready: dict) -> list[tuple[float, float]]:
    """Every point the checks start a click, drag or scroll at: all must be the test window's."""
    t = ready.get("targets") or {}
    pts = [(x + w / 2, y + h / 2) for x, y, w, h in t.values()]
    bx, by, _, _ = t.get("board", (0, 0, 0, 0))
    cell = float(ready.get("cell", L.CELL))
    pts += [(bx + (c + 0.5) * cell, by + (r + 0.5) * cell) for c, r in ((6, 1), (2, 2), (5, 5), (1, 6), (4, 3))]
    if ready.get("second"):
        x, y, w, h = ready["second"]["target"]
        pts.append((x + w / 2, y + h / 2))
    return pts


async def run(
    *,
    mac: Any = None,
    budget_s: float = _BUDGET_S,
    notify: Callable[[str], Any] | None = None,
    busy: Callable[[], str] | None = None,
) -> Report:
    """Run the controls self-test (see the module doc). Never raises for what happens on screen:
    everything ends up in the report. `busy`: why not now ("" = go), asked again right before the
    test takes the mouse (other NEO jobs at work)."""
    from neo2.core.tools import computer
    from neo2.core.tools.computer import apps

    mac = mac or Mac()
    t0 = time.monotonic()
    report = Report()
    if not mac.trusted():
        report.refused = "Accessibility is off for NEO, so it can't click or type anywhere"
        return report
    prev_name, prev_pid = await asyncio.to_thread(mac.front)
    pointer0 = await asyncio.to_thread(mac.pointer)
    saved_target = apps._target
    report.info["in front before"] = f"{prev_name or '?'} (pid {prev_pid})"
    started = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="neo-selftest-"))
    log = tmp / "events.jsonl"
    log.touch()
    child: Child | None = None
    watch: Watch | None = None
    front_end: tuple[str, int] | None = None
    front_armed: tuple[str, int] | None = None  # in front when the test took the mouse
    test_name = ""  # the test window's name, once it is up: a target NEO's tools set to it is the test's
    try:
        child = await asyncio.to_thread(mac.launch, log, secrets.token_hex(2).upper())
        reader = LogReader(log)
        ready, why = await _wait_ready(reader, child, _READY_S)
        if ready is None:
            report.refused = why
            with contextlib.suppress(OSError):  # what the window said on its way out, for the report
                if err := log.with_suffix(".stderr").read_text(errors="replace").strip():
                    report.info["test window error"] = err[-400:]
            return report
        test_name = str(ready.get("name") or "")
        report.info["test window"] = f"{ready.get('name')} (pid {child.pid}), {ready.get('frame')}"
        report.info["displays"] = ready.get("screens")
        windows, why = await asyncio.to_thread(_identity, mac, child, ready)
        if why:
            report.refused = why
            return report
        main = windows[int(ready["window"])]
        if await asyncio.to_thread(lambda: mac.front()[1]) == child.pid:
            child.send("deactivate")  # it starts behind: that is the first thing to check
            await asyncio.sleep(0.4)
        watch = Watch(mac, child.pid, windows)
        for x, y in _points(ready):
            where = await asyncio.to_thread(watch.elsewhere, x, y)
            if where:
                report.refused = (
                    f"at ({x:.0f}, {y:.0f}) the pointer would reach {where}, not the test window, and I never "
                    "click anything but the test window"
                )
                return report
        if busy is not None and (why := busy()):
            report.refused = f"{why}, and the check needs the mouse and keyboard to itself"
            return report
        if await asyncio.to_thread(mac.buttons_down):
            report.refused = "a mouse button is held down (yours, or another NEO job's), and the check needs the mouse to itself"
            return report
        if notify is not None:
            with contextlib.suppress(Exception):
                await _maybe_await(notify("Checking my computer controls in a test window: please don't touch the mouse or keyboard."))
        r = _Run(mac, child, reader, ready, watch, main)
        r.can_capture = bool(await asyncio.to_thread(_can_capture, mac))
        front_armed = await asyncio.to_thread(mac.front)
        try:
            mode = await asyncio.to_thread(watch.arm, r.cell(6, 1))
        except Stop as e:
            report.refused = str(e)
            return report
        report.info["watching"] = _WATCHING.get(mode, mode)
        if watch.stopped:
            report.stopped = f"{watch.stopped}, so I stopped before doing anything"
            return report
        apps.set_target("")  # keys go to the test window only, never an app NEO opened earlier
        with computer.allow_test_window(child.pid):
            for fn in CHECKS:
                if time.monotonic() - t0 > budget_s:
                    report.stopped = "it was taking too long"
                    break
                check = await _one(fn, r)
                report.checks.append(check)
                if watch.stopped:
                    report.stopped = f"{watch.stopped}; nothing else was clicked or typed"
                    break
                if watch.refused:
                    report.stopped = f"{watch.refused}, so I didn't send it"
                    break
                if not child.alive():
                    report.stopped = "the test window closed"
                    break
        if report.checks:
            report.checks.append(_feedback(report.checks))
        return report
    except Exception as e:  # noqa: BLE001 - a broken test is a report too, never a crash of the request
        if report.checks:
            report.stopped = f"the test itself broke ({e!r:.160})"
        else:
            report.refused = f"the test itself broke ({e!r:.160})"
        return report
    finally:
        if child is not None and child.alive():
            with contextlib.suppress(Exception):
                front_end = await asyncio.to_thread(mac.front)
        try:
            if watch is not None:
                await asyncio.to_thread(watch.disarm)  # it still listens: the user may take over now
            # What the test set (none, or the test window its clicks made the target): put back.
            # Another job's target, set meanwhile, stays.
            if apps._target is None or (test_name and apps._target[0] == test_name):
                apps._target = saved_target
            _forget_test_tree(started)
            if child is not None:
                await asyncio.to_thread(child.stop)
            await _clean_up(
                mac, report, watch, (prev_name, prev_pid), child.pid if child else 0, front_end, pointer0, front_armed
            )
        finally:
            if watch is not None:
                await asyncio.to_thread(watch.stop_listening)
            shutil.rmtree(tmp, ignore_errors=True)
            report.seconds = time.monotonic() - t0


async def _clean_up(
    mac: Any,
    report: Report,
    watch: Watch | None,
    prev: tuple[str, int],
    test_pid: int,
    front_end: tuple[str, int] | None,
    pointer0: tuple[float, float] | None,
    front_armed: tuple[str, int] | None = None,
) -> None:
    """The Mac back as the user left it: the app that was in front, and the pointer — unless it is
    theirs now. The user took over (the watch saw them, or the pointer isn't where NEO left it) or
    another app came to the front during the run (not the test window, not theirs, not the one in
    front when the test began): nothing is moved or brought forward over them. Only when the user
    took over while the test window was in front, and has let go since, does their app come back
    (macOS picked some app when the test window closed)."""
    from neo2.core.tools.computer import input as inp

    name, pid = prev
    known = {test_pid, pid} | ({int(front_armed[1])} if front_armed else set())
    elsewhere = front_end is not None and int(front_end[1]) not in known
    was_front = front_end is not None and int(front_end[1]) == test_pid
    took_over = active = False
    with contextlib.suppress(Exception):
        if watch is not None:
            took_over = await asyncio.to_thread(watch.took_over)
            active = await asyncio.to_thread(watch.active_after_run)
    if elsewhere:
        report.info["clean-up"] = f"{front_end[0] or 'another app'} came to the front during the check, so I left it there"
    elif active:
        report.info["clean-up"] = "you were using the mouse or keyboard, so I left the pointer and the front app as they were"
    if pid and pid != test_pid and not elsewhere and not (took_over and (active or not was_front)):
        with contextlib.suppress(Exception):
            if (await asyncio.to_thread(mac.front))[1] == pid:
                report.info["in front after"] = name
            else:
                ok = await asyncio.to_thread(mac.activate, pid)
                report.info["in front after"] = name if ok else f"couldn't bring {name} back"
    if watch is None or not watch.sent or took_over or elsewhere or not pointer0:
        return
    with contextlib.suppress(Exception):
        if not await asyncio.to_thread(watch.took_over):  # a last look: still where NEO left it
            await asyncio.to_thread(inp.move, *pointer0)  # the pointer back where the user left it


_WATCHING = {
    "tap": "your mouse and keyboard (an event tap)",
    "tap-mouse": "your mouse (an event tap; keys you press can't be noticed without Input Monitoring)",
    "pointer": "pointer only (no event tap: keys you press can't be noticed)",
}


def _forget_test_tree(since: float) -> None:
    """The ax_tree ids the test read point into its window, which is about to close: a model that
    clicked one later would hit whatever is there then. Dropped, so it has to look again."""
    from neo2.core.tools.computer import ax

    if ax._last_tree and ax._last_tree_ts >= since:
        ax._last_tree, ax._last_tree_ts, ax._last_tree_app = {}, 0.0, ""


def _can_capture(mac: Any) -> bool:
    try:
        return bool(mac.can_capture())
    except Exception:  # noqa: BLE001
        return False


async def _maybe_await(v: Any) -> None:
    if asyncio.iscoroutine(v) or isinstance(v, asyncio.Future):
        await v
