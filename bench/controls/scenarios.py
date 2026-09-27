"""The bench's fake apps, and its task scenarios: a fake app state, an utterance, and a goal.

Apps (``fakes.App`` trees; names are fake-world data, never NEO's decision logic):

* ``big_app`` — a Finder/Safari-like app of 2,522 elements: ten closed menus of twenty items, a
  toolbar, a forty-row sidebar, a 600-block web page mostly scrolled out of view, a background
  window of 200 texts (the tree design-speed measured 7,879 AX calls on).
* ``board_app`` — a window whose board is one view with no Accessibility children; its Game menu's
  New Game (⌘N) opens a "Start a New Game?" sheet (Players and Side popups, Cancel, Start).
* ``form_app`` — labelled fields, a password field, a search field, a checkbox, a Submit that commits.
* ``hindi_app`` — the form, localized in Hindi (a Send, "भेजें", that commits).
* ``notes_app`` — one text area; ``neo_overlay`` — NEO's own floating window (never to be touched).

A scenario passes only when the goal is reached in the fake world, nothing that commits was pressed
(the bench never says yes to a NEEDS_CONFIRM), a brain that can't see neither had coordinates acted
on nor was handed any (DECISIONS 43), and nothing touched NEO's own window.

The scripted brain follows what it is offered, not what the bench hopes for: it clicks a square by
name only when ``click`` takes a ``cell``, skips its checking screenshot only when the action's
result came with a picture, sends several actions in one turn only when the prompt says it may,
starts without a look only when the request already carries one, and uses ``menu_command`` only
when that tool is offered. So round trips fall when NEO's code offers these, and not before.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from bench.controls import fakes as F
from bench.controls.fakes import (
    App,
    Board,
    FakeEl,
    World,
    app,
    button,
    field_el,
    menu_bar_item,
    menu_item,
    text,
    window,
)

# ---- the apps ---------------------------------------------------------------------------------------

BIG, BOARD, FORM, HINDI, NOTES = 501, 502, 503, 504, 505


def big_app(pid: int = BIG, name: str = "Safari", rows: int = 40, doc_items: int = 600, menus: int = 10,
            menu_items: int = 20, bg_items: int = 200) -> App:  # fmt: skip
    """design-speed's bench_ax.build(), as an App."""
    mbar = [menu_bar_item(f"Menu{m}", [menu_item(f"Item {m}.{i}") for i in range(menu_items)]) for m in range(menus)]
    toolbar = FakeEl("AXToolbar", frame=(120, 70, 1100, 52), children=[
        button(f"Tool {i}", (130 + 40 * i, 80, 32, 28)) for i in range(12)])  # fmt: skip
    side_rows = [
        FakeEl("AXRow", frame=(120, 130 + 22 * i, 220, 22), children=[
            FakeEl("AXCell", frame=(120, 130 + 22 * i, 220, 22), children=[
                FakeEl("AXImage", frame=(126, 132 + 22 * i, 16, 16), AXDescription="folder"),
                FakeEl("AXStaticText", value=f"Folder {i}", frame=(146, 132 + 22 * i, 180, 18))])])
        for i in range(rows)
    ]  # fmt: skip
    sidebar = FakeEl("AXScrollArea", frame=(120, 122, 220, 748), children=[
        FakeEl("AXOutline", frame=(120, 122, 220, 22 * rows), children=side_rows)])  # fmt: skip
    items = []
    for i in range(doc_items):
        y = 130 + 30 * i
        kids = [FakeEl("AXStaticText", value=f"Paragraph {i} text about things", frame=(360, y, 700, 16)),
                FakeEl("AXLink", f"link {i}", frame=(360, y + 16, 80, 14), actions=("AXPress",))]  # fmt: skip
        if i % 5 == 0:
            kids.append(button(f"Action {i}", (460, y + 14, 60, 16)))
        items.append(FakeEl("AXGroup", frame=(350, y, 800, 30), children=kids))
    web = FakeEl("AXWebArea", "Page", frame=(350, 122, 870, 30 * doc_items + 20), children=items)
    content = FakeEl("AXScrollArea", frame=(350, 122, 870, 748), children=[web])
    win = window("Front", (120, 70, 1100, 800), toolbar, FakeEl("AXSplitGroup", frame=(120, 122, 1100, 748), children=[sidebar, content]))
    bg = window("Back", (200, 200, 800, 600), *[FakeEl("AXStaticText", value=f"bg {i}", frame=(210, 210 + 2 * i, 300, 16)) for i in range(bg_items)])
    return app(name, pid, win, bg, menus=mbar, app_menu=False)


def _new_game_sheet(w: World, win: FakeEl, *, hindi: bool = False) -> None:
    """The "Start a New Game?" sheet on ``win`` (Players and Side, Cancel and Start)."""
    if win.find(lambda e: e.role == "AXSheet") is not None:
        return
    x, y, _ww, _hh = win.frame
    sx, sy = x + 200, y + 28

    def close(ww: World, el: FakeEl, start: bool) -> None:
        sheet = el.parent
        if sheet is not None and sheet.parent is not None:
            sheet.parent.remove(sheet)
        if start and ww.board is not None:
            ww.board.reset()

    t = ("नया खेल शुरू करें?", "रद्द करें", "शुरू करें", "खिलाड़ी", "पक्ष") if hindi else (
        "Start a New Game?", "Cancel", "Start", "Players", "Side")  # fmt: skip
    sheet = FakeEl("AXSheet", t[0], frame=(sx, sy, 500, 170), children=[
        text(t[0], (sx + 20, sy + 20, 300, 18)),
        FakeEl("AXPopUpButton", t[3], value="Human vs Computer", frame=(sx + 20, sy + 60, 220, 24), actions=("AXPress", "AXShowMenu")),
        FakeEl("AXPopUpButton", t[4], value="White", frame=(sx + 260, sy + 60, 200, 24), actions=("AXPress", "AXShowMenu")),
        button(t[1], (sx + 250, sy + 120, 100, 28), on_press=lambda ww, el: close(ww, el, False)),
        button(t[2], (sx + 370, sy + 120, 100, 28), on_press=lambda ww, el: close(ww, el, True)),
    ])  # fmt: skip
    win.add(sheet)
    w.version += 1


def board_app(w: World, pid: int = BOARD, name: str = "Chess", *, sheet: bool = False) -> App:
    """A board game: the board is one view (no children), clicks and drags move pieces; Game ›
    New Game (⌘N) opens the new-game sheet. Installs the world's board."""
    board = Board(420, 150, 76)
    w.board = board

    def board_click(ww: World, el: FakeEl, x: float, y: float, button_: str) -> None:
        b = ww.board
        d = ww.last_down or {}
        start = b.square_at(*d.get("at", (x, y))) if b else None
        end = b.square_at(x, y) if b else None
        if b is None or start is None or end is None:
            return
        if d.get("drags") and start != end:
            b.move(start, end)
            b.selected = None
        elif b.selected and end != b.selected and start == end:
            b.move(b.selected, end)
            b.selected = None
        elif end in b.pieces:
            b.selected = end
        ww.acted.append(ww._act("board", el, "click", start=start, end=end))

    board_el = FakeEl("AXGroup", frame=(board.x, board.y, board.sq * 8, board.sq * 8), on_click=board_click, first_mouse=False)
    w.board_el = board_el
    win = window(f"{name} — Game 1", (360, 90, 900, 800), board_el, text("White to move", (1080, 160, 160, 18)))
    game = menu_bar_item("Game", [
        menu_item("New Game", key="N", on_press=lambda ww, el: _new_game_sheet(ww, ww.named(name).windows()[0])),
        menu_item("Resign", commit=True),
    ])  # fmt: skip
    view = menu_bar_item("Board", [menu_item("Show Coordinates", key="J")])
    a = app(name, pid, win, menus=[game, view])
    if sheet:
        _new_game_sheet(w, win)
    return a


def form_app(pid: int = FORM, name: str = "Accounts", *, hindi: bool = False) -> App:
    """A sign-up form. In Hindi: every label localized, and its Send commits."""
    t = (
        ("खाता बनाएँ", "नाम", "ईमेल", "पासवर्ड", "खोजें", "मुझे याद रखें", "रद्द करें", "शुरू करें", "भेजें")
        if hindi
        else ("Create your account", "Name", "Email", "Password", "Search", "Remember me", "Cancel", "Options…", "Submit")
    )

    def toggle(w: World, el: FakeEl) -> None:
        el.attrs["fill"] = (255, 255, 255) if el.attrs.get("fill") == (120, 200, 255) else (120, 200, 255)

    def options(w: World, el: FakeEl) -> None:
        win = el.window()
        if win is None or win.find(lambda e: e.role == "AXSheet") is not None:
            return
        x, y, _, _ = win.frame
        done = button("Done", (x + 500, y + 150, 80, 26), on_press=lambda ww, b: b.parent.parent.remove(b.parent))
        win.add(FakeEl("AXSheet", "Options", frame=(x + 150, y + 28, 460, 160), children=[text("Options", (x + 170, y + 50, 200, 18)), done]))

    remember = FakeEl("AXCheckBox", t[5], value=0, frame=(340, 330, 180, 22), actions=("AXPress",))
    win = window(t[0], (300, 120, 700, 520),
                 text(t[0], (320, 160, 300, 20)),
                 field_el(t[1], (320, 200, 400, 24)),
                 field_el(t[2], (320, 240, 400, 24)),
                 field_el(t[3], (320, 280, 400, 24), secure=True),
                 field_el(t[4], (760, 160, 200, 24), role="AXSearchField"),
                 remember,
                 button(t[6], (520, 580, 100, 28)),
                 button(t[7], (640, 580, 100, 28), on_press=toggle if hindi else options),
                 button(t[8], (760, 580, 100, 28), commit=True))  # fmt: skip
    return app(name, pid, win, menus=[menu_bar_item("File", [menu_item("Close", key="W")])])


def notes_app(pid: int = NOTES, name: str = "Notes", frame: tuple[float, float, float, float] = (700, 110, 700, 700)) -> App:
    x, y, w_, h_ = frame
    area = field_el("", (x + 20, y + 60, w_ - 40, h_ - 100), role="AXTextArea")

    def bold(w: World, el: FakeEl) -> None:
        el.attrs["fill"] = (255, 255, 255) if el.attrs.get("fill") == (255, 220, 120) else (255, 220, 120)

    win = window("New Note", frame, button("Bold", (x + w_ - 80, y + 22, 60, 26), on_press=bold), area)
    a = app(name, pid, win, menus=[menu_bar_item("Format", [menu_item("Bold", key="B")])])
    a.focused = area
    return a


def neo_overlay() -> App:
    win = window("NEO", (1150, 800, 340, 150), text("Should I go ahead?", (1170, 830, 200, 18)),
                 button("Approve", (1170, 900, 90, 28), commit=True))  # fmt: skip
    return app("NEO", F.NEO_PID, win, layer=3)


# ---- worlds -------------------------------------------------------------------------------------------


def world(*make: Callable[[World], App], front: str = "") -> World:
    w = World()
    for m in make:
        w.add(m(w))
    w.add(neo_overlay())
    if front:
        w.activate(w.named(front).pid, by_neo=False)
    return w


def board_world(*, sheet: bool = False, front: str = "Chess") -> World:
    return world(lambda w: big_app(), lambda w: board_app(w, sheet=sheet), front=front)


# ---- reading tool results ------------------------------------------------------------------------------

_LINE = re.compile(r'^\s*(e\d+)\s+(\w+)(?:\s+"([^"]*)")?(?:\s+=\s+(.*?))?(?:\s+\((?:disabled|focused)\))*\s+@\((-?\d+),(-?\d+) (\d+)x(\d+)\)', re.M)
_SHOT = re.compile(r'shot="([^"\s]+)"')
NEEDS = ("NEEDS_CONFIRM", "NEEDS_USER")


def elements(txt: str) -> list[dict]:
    out = []
    for m in _LINE.finditer(txt or ""):
        val = (m.group(4) or "").strip().strip("'\"")
        out.append({"id": m.group(1), "role": m.group(2), "title": m.group(3) or "", "value": val,
                    "frame": tuple(int(m.group(i)) for i in range(5, 9))})  # fmt: skip
    return out


def find_id(txt: str, label: str, role: str | None = None) -> str | None:
    want = label.casefold()
    for e in elements(txt):
        if (e["title"].casefold() == want or e["value"].casefold() == want) and (role is None or e["role"] == role):
            return e["id"]
    return None


def results(messages: list) -> list:
    return [r for m in messages if m.role == "tool" for r in m.tool_results]


def latest(messages: list, *names: str) -> Any | None:
    return next((r for r in reversed(results(messages)) if r.name in names), None)


def looks(messages: list) -> str:
    """Every observation's text so far, newest last (the request's own prefetched look included)."""
    first = next((m for m in messages if m.role == "user"), None)
    parts = [first.text if first else ""]
    parts += [r.content for r in results(messages) if r.name in ("ax_tree", "ax_find", "screenshot", "look", "observe")]
    return "\n".join(parts)


def prefetched(messages: list) -> bool:
    """The request itself already carries a look (a tree or a picture): no first observation."""
    first = next((m for m in messages if m.role == "user"), None)
    return bool(first and (first.images or elements(first.text)))


def verify_asked(messages: list) -> bool:
    return bool(messages) and messages[-1].role == "user" and messages[-1].text.startswith("[verify]")


def params(tools: list, name: str) -> set[str]:
    spec = next((t for t in tools if t.name == name), None)
    return set((spec.parameters or {}).get("properties", {})) if spec is not None else set()


def offered(tools: list, name: str) -> bool:
    return any(t.name == name for t in tools)


# menu_command's argument for the command, by the names the commands workstream may give it (the
# self-test's _menu_sheet asks in the same order).
MENU_ARGS = ("command", "item", "request", "text", "title", "name", "path")


def menu_arg(names: set[str]) -> str:
    """Which of menu_command's parameters takes the command: the first of MENU_ARGS it has, else its
    one other parameter (never whichever sorts first: "confirm" before "item")."""
    got = next((k for k in MENU_ARGS if k in names), None)
    if got is not None:
        return got
    rest = sorted(names - {"app"})
    return rest[0] if len(rest) == 1 else "command"


_BATCH_WORDS = re.compile(r"\b(?:several|multiple|a batch of)\s+(?:actions|tool calls|calls|steps)\s+in\s+one\s+(?:turn|reply|step)", re.I)


def may_batch(system: str, tools: list) -> bool:
    """The prompt tells the model it may send several actions in one turn (and NEO stops at the
    first that fails)."""
    return bool(_BATCH_WORDS.search(system or "")) or any(_BATCH_WORDS.search(t.description or "") for t in tools)


_ids = iter(range(1, 1_000_000))


def call(name: str, **args: Any) -> Any:
    from neo2.core.providers.base import ToolCall

    return ToolCall(id=f"c{next(_ids)}", name=name, args=args)


def turn(*calls: Any, text: str = "") -> Any:
    from neo2.core.providers.base import Turn

    return Turn(text=text, tool_calls=list(calls))


def say(text_: str) -> Any:
    from neo2.core.providers.base import Turn

    return Turn(text=text_)


# ---- policies (the scripted model) ----------------------------------------------------------------------


def press_policy(label: str, *, app_name: str = "", verb: str = "ax_press", done: str = "") -> Callable:
    """Find ``label`` in a tree, press it by id, answer; look again when the loop asks to verify."""

    def policy(b: F.FakeBrain, msgs: list, system: str, tools: list) -> Any:
        st = b.state
        where = {"app": app_name} if app_name else {}
        if verify_asked(msgs):
            st["verified"] = True
            return turn(call("ax_tree", **where))
        if st.get("verified") or st.get("acted"):
            return say(done or f"Pressed {label}.")
        eid = find_id(looks(msgs), label)
        if eid is None:
            if st.get("looked"):
                return say(f"I can't find {label}.")
            st["looked"] = True
            return turn(call("ax_tree", **where))
        st["acted"] = True
        return turn(call(verb, id=eid))

    return policy


def set_policy(label: str, value: str) -> Callable:
    def policy(b: F.FakeBrain, msgs: list, system: str, tools: list) -> Any:
        st = b.state
        if verify_asked(msgs):
            st["verified"] = True
            return turn(call("ax_find", query=label))
        if st.get("verified") or st.get("acted"):
            return say(f"Put {value} in the {label} field.")
        eid = find_id(looks(msgs), label)
        if eid is None:
            if st.get("looked"):
                return say(f"I can't find the {label} field.")
            st["looked"] = True
            return turn(call("ax_tree"))
        st["acted"] = True
        return turn(call("ax_set_value", id=eid, value=value))

    return policy


def scroll_policy(app_name: str) -> Callable:
    """Find a scrollable area in the app's tree and scroll it by id; answer — with NEO's own words
    when the scroll was refused (a model passes an error on)."""

    def policy(b: F.FakeBrain, msgs: list, system: str, tools: list) -> Any:
        st = b.state
        if verify_asked(msgs):
            st["verified"] = True
            return turn(call("ax_tree", app=app_name))
        if st.get("verified") or st.get("acted"):
            refused = latest(msgs, "scroll")
            if refused is not None and refused.content.startswith(("Error", "NEEDS_")):
                return say(refused.content)
            return say(f"Scrolled down in {app_name}.")
        areas = [e for e in elements(looks(msgs)) if e["role"] in ("WebArea", "ScrollArea", "TextArea")]
        if not areas:
            if st.get("looked"):
                return say("There's nothing to scroll.")
            st["looked"] = True
            return turn(call("ax_tree", app=app_name))
        st["acted"] = True
        return turn(call("scroll", id=areas[0]["id"], dy=-8))

    return policy


def _pixel(world: World, sid: str, square: str) -> tuple[float, float] | None:
    from neo2.core.tools.computer import pointer

    s = pointer.get_shot(sid)
    if s is None or world.board is None:
        return None
    x, y = world.board.center(square)
    return round((x - s.x0) / s.ppp, 1), round((y - s.y0) / s.ppp, 1)


def move_policy(world: World, moves: list[tuple[str, str]], *, app_name: str = "Chess", then: Callable | None = None,
                blind_guess: bool = False) -> Callable:  # fmt: skip
    """Make these moves on the board by clicks: look (a tree, then a picture), click the piece,
    look, click its square, look, answer — fewer looks when results carry a picture, both clicks
    in one turn when the prompt allows it, squares by name when ``click`` takes a ``cell``."""

    def clicks_for(frm: str, to: str, sid: str | None, cells: bool) -> list:
        if cells:
            return [call("click", cell=frm), call("click", cell=to)]
        if sid is None:
            return []
        out = []
        for sq in (frm, to):
            px = _pixel(world, sid, sq)
            if px is None:
                return []
            out.append(call("click", x=px[0], y=px[1], shot=sid))
        return out

    def policy(b: F.FakeBrain, msgs: list, system: str, tools: list) -> Any:
        st = b.state
        if then is not None and not st.get("then_done"):
            got = then(b, msgs, system, tools)
            if got is not None:
                return got
            st["then_done"] = True
            st["base"] = len(results(msgs))  # (the moves' own looks start here)
        if blind_guess and not b.sees:
            if not st.get("guessed"):
                st["guessed"] = True
                return turn(call("click", x=506, y=770))  # coordinates made up: the Chess log's failure
            return say("I can't see the board right now, so I can't make that move.")
        cells = "cell" in params(tools, "click")
        batch = may_batch(system, tools)
        todo = st.setdefault("todo", [sq for mv in moves for sq in mv])
        base = st.get("base", 0)
        mine = results(msgs)[base:]
        shots = [r for r in mine if r.name in ("screenshot", "look", "observe") and r.images] + [r for r in mine if r.images]
        if not st.get("looked_tree") and not prefetched(msgs) and not cells and not st.get("then_used_tree"):
            st["looked_tree"] = True
            return turn(call("ax_tree", app=app_name))
        last = mine[-1] if mine else None
        if not todo:
            if last is not None and (last.images or last.name in ("screenshot", "look")) or verify_asked(msgs) and st.get("checked"):
                return say("Moved " + ", ".join(f"{a} to {c}" for a, c in moves) + ".")
            if verify_asked(msgs) or not (last and last.images):
                st["checked"] = True
                return turn(call("screenshot"))
            return say("Moved " + ", ".join(f"{a} to {c}" for a, c in moves) + ".")
        sid = None
        pic = shots[-1] if shots else None
        if pic is not None:
            m = _SHOT.search(pic.content)
            sid = m.group(1) if m else None
        first = next((m for m in msgs if m.role == "user"), None)
        if sid is None and first is not None and (m := _SHOT.search(first.text or "")):
            sid = m.group(1)
        need_look = not cells and (sid is None or (last is not None and last.name == "click" and not last.images))
        if need_look:
            return turn(call("screenshot"))
        if batch:
            frm, to = todo[0], todo[1]
            calls = clicks_for(frm, to, sid, cells)
            del todo[:2]
            return turn(*calls) if calls else say("I can't find the board.")
        sq = todo.pop(0)
        if cells:
            return turn(call("click", cell=sq))
        px = _pixel(world, sid or "", sq)
        return turn(call("click", x=px[0], y=px[1], shot=sid)) if px else say("I can't find the board.")

    return policy


def new_game_policy(start_label: str = "Start", app_name: str = "Chess", *, open_first: bool = False) -> Callable:
    """Open the app if asked, start a new game from its menu (⌘N, or menu_command when offered),
    look at the sheet (unless the result already listed it), press Start. Returns None when done."""

    def step(b: F.FakeBrain, msgs: list, system: str, tools: list) -> Any:
        st = b.state.setdefault("new_game", {})
        if open_first and not st.get("opened"):
            st["opened"] = True
            return turn(call("open_app", target=app_name))
        if not st.get("tree") and not prefetched(msgs):
            st["tree"] = True
            b.state["then_used_tree"] = True
            return turn(call("ax_tree", app=app_name))
        if not st.get("menu"):
            st["menu"] = True
            if offered(tools, "menu_command"):
                return turn(call("menu_command", **{menu_arg(params(tools, "menu_command")): "New Game", "app": app_name}))
            return turn(call("hotkey", keys="cmd+n", app=app_name))
        last = results(msgs)[-1] if results(msgs) else None
        if not st.get("pressed"):
            eid = find_id(last.content if last else "", start_label)
            if eid is None and not st.get("sheet_tree"):
                st["sheet_tree"] = True
                return turn(call("ax_tree", app=app_name))
            eid = eid or find_id(looks(msgs), start_label)
            if eid is None:
                return None  # (the sheet was answered already)
            st["pressed"] = True
            return turn(call("ax_press", id=eid))
        return None

    return step


def menu_policy(start_label: str = "Start", app_name: str = "Chess") -> Callable:
    step = new_game_policy(start_label, app_name)

    def policy(b: F.FakeBrain, msgs: list, system: str, tools: list) -> Any:
        if verify_asked(msgs):
            b.state["verified"] = True
            return turn(call("ax_tree", app=app_name))
        if b.state.get("verified"):
            return say("Started a new game.")
        got = step(b, msgs, system, tools)
        return got if got is not None else say("Started a new game.")

    return policy


def type_policy(txt: str) -> Callable:
    def policy(b: F.FakeBrain, msgs: list, system: str, tools: list) -> Any:
        if b.state.get("typed"):
            return say("Typed it.")
        b.state["typed"] = True
        return turn(call("type_text", text=txt))

    return policy


# ---- scenarios ----------------------------------------------------------------------------------------------


@dataclass
class Scenario:
    name: str
    utterance: str
    build: Callable[[], World]
    policy: Callable[[World], Callable]
    goal: Callable[[World, dict], bool]
    intent: str = "agent_task"
    needs_screen: bool = False
    sees: bool = True
    note: str = ""
    targets: dict[str, Any] = field(default_factory=dict)  # the plan's per-task targets (round trips)


def _moved(*mv: tuple[str, str]) -> Callable[[World, dict], bool]:
    def goal(w: World, run: dict) -> bool:
        b = w.board
        return b is not None and all(to in b.pieces and frm not in b.pieces for frm, to in mv)

    return goal


def _new_game(w: World, run: dict) -> bool:
    b = w.board
    win = w.named("Chess").windows()[0]
    return b is not None and b.games >= 2 and win.find(lambda e: e.role == "AXSheet") is None


def _value(app_name: str, label: str, want: str) -> Callable[[World, dict], bool]:
    def goal(w: World, run: dict) -> bool:
        a = w.named(app_name)
        el = a.root.find(lambda e: e.attrs.get("AXTitle") == label and e.role in F._EDITABLE)
        pw = a.root.find(lambda e: e.attrs.get("AXSubrole") == "AXSecureTextField")
        return el is not None and el.attrs.get("AXValue") == want and not (pw and pw.attrs.get("AXValue"))

    return goal


def _scrolled(app_name: str) -> Callable[[World, dict], bool]:
    """A scroll landed in this app's window (whatever part of it)."""

    def goal(w: World, run: dict) -> bool:
        return any(a["kind"] == "scroll" and a["app"] == app_name for a in w.acted)

    return goal


def _covered_scroll(app_name: str, cover: str) -> Callable[[World, dict], bool]:
    """A scroll meant for ``app_name`` whose window ``cover`` covers in part: it lands on
    ``app_name`` (a part that shows), or NEO says what covers it (the scroll's own answer, or the
    reply) — and never lands on ``cover``."""

    def goal(w: World, run: dict) -> bool:
        if _scrolled(cover)(w, run):
            return False
        said = [run.get("answer") or ""] + [text for tool, text in run.get("results") or [] if tool == "scroll"]
        return _scrolled(app_name)(w, run) or any(cover.casefold() in t.casefold() for t in said)

    return goal


def _typed(app_name: str, want: str) -> Callable[[World, dict], bool]:
    def goal(w: World, run: dict) -> bool:
        a = w.named(app_name)
        area = a.root.find(lambda e: e.role == "AXTextArea")
        return area is not None and want in str(area.attrs.get("AXValue") or "")

    return goal


def _pressed(label: str) -> Callable[[World, dict], bool]:
    return lambda w, run: w.pressed(label)


def _asked_not_pressed(label: str) -> Callable[[World, dict], bool]:
    """A commit label: NEO asked the user (NEEDS_CONFIRM) and pressed nothing."""
    return lambda w, run: bool(run.get("asked")) and not w.pressed(label)


def _stopped_blind(w: World, run: dict) -> bool:
    """A pixel task with no brain that can see: it stops, nothing was clicked on the board."""
    return not (w.board and w.board.moves) and not any(e["type"] in (1, 3) for e in w.posted)


SCENARIOS: list[Scenario] = [
    Scenario(
        "chess_move", "make the first move e4", lambda: board_world(),
        lambda w: move_policy(w, [("e2", "e4")]), _moved(("e2", "e4")),
        needs_screen=True, targets={"round_trips": 2, "round_trips_p0": 3, "seconds": 10.0},
    ),
    Scenario(
        "chess_full", "play chess as white and make the first move e4", lambda: board_world(front="Safari"),
        lambda w: move_policy(w, [("e2", "e4")], then=new_game_policy(open_first=True)),
        lambda w, run: _new_game(w, run) and _moved(("e2", "e4"))(w, run),
        needs_screen=True, targets={"round_trips": 4},
    ),
    Scenario(
        "scroll_named_app", "scroll down in Safari",
        lambda: world(lambda w: big_app(), lambda w: notes_app(frame=(1228, 110, 280, 420)), front="Notes"),
        lambda w: scroll_policy("Safari"), lambda w, run: _scrolled("Safari")(w, run) and not _scrolled("Notes")(w, run),
        intent="quick_action", targets={"round_trips": 0}, note="Safari's page is in view; Notes is in front beside it",
    ),
    Scenario(
        "scroll_covered_app", "scroll down in Safari", lambda: world(lambda w: big_app(), lambda w: notes_app(), front="Notes"),
        lambda w: scroll_policy("Safari"), _covered_scroll("Safari", "Notes"), intent="quick_action",
        note="Notes' window covers the middle of Safari's page: the scroll lands on Safari's page, or NEO says Notes "
             "is in the way — never on Notes",
    ),
    Scenario(
        "fill_field", "fill the name field with Arnav", lambda: world(lambda w: form_app(), front="Accounts"),
        lambda w: set_policy("Name", "Arnav"), _value("Accounts", "Name", "Arnav"), needs_screen=True,
        targets={"round_trips": 0},
    ),
    Scenario(
        "click_fallback", "click the Start button", lambda: _neo_front(board_world(sheet=True)),
        lambda w: press_policy("Start"), _new_game, intent="quick_action", targets={"round_trips": 0},
        note="NEO's own window in front (the user just talked to it); the button is on a sheet behind it",
    ),
    Scenario(
        "reworded_en", "press start in the dialog", lambda: board_world(sheet=True),
        lambda w: press_policy("Start"), _new_game, intent="quick_action", targets={"round_trips": 0},
    ),
    Scenario(
        "reworded_hinglish", "start wala button dabao", lambda: board_world(sheet=True),
        lambda w: press_policy("Start"), _new_game, intent="quick_action", targets={"round_trips": 0},
    ),
    Scenario(
        "reworded_hindi", "शुरू करें वाला बटन दबाओ", lambda: world(lambda w: form_app(HINDI, "Khata", hindi=True), front="Khata"),
        lambda w: press_policy("शुरू करें"), _pressed("शुरू करें"), intent="quick_action", targets={"round_trips": 0},
    ),
    Scenario(
        "hindi_commit_gate", "भेजें दबाओ", lambda: world(lambda w: form_app(HINDI, "Khata", hindi=True), front="Khata"),
        lambda w: press_policy("भेजें"), _asked_not_pressed("भेजें"), intent="quick_action",
        note="a Devanagari commit label: passes only when NEO asks first (0/27 Hindi commit labels were gated at c769cb0)",
    ),
    Scenario(
        "menu_command", "start a new game", lambda: board_world(),
        lambda w: menu_policy(), _new_game, intent="quick_action", targets={"round_trips": 0},
    ),
    Scenario(
        "open_and_type", "open Notes and type hello", lambda: world(lambda w: big_app(), lambda w: notes_app(), front="Safari"),
        lambda w: type_policy("hello"), _typed("Notes", "hello"), intent="quick_action", targets={"round_trips": 0},
    ),
    Scenario(
        "blind_chess_move", "make the first move e4", lambda: board_world(),
        lambda w: move_policy(w, [("e2", "e4")], blind_guess=True), _stopped_blind, needs_screen=True, sees=False,
        note="only a text-only brain left: it stops, no made-up coordinates act",
    ),
    Scenario(
        "blind_press", "press start in the dialog", lambda: board_world(sheet=True),
        lambda w: press_policy("Start"), _new_game, intent="quick_action", sees=False,
        note="a text-only brain works by ids from the tree: it is never handed coordinates (DECISIONS 43)",
    ),
]


def _neo_front(w: World) -> World:
    w.activate(F.NEO_PID, by_neo=False)
    return w


def by_name(name: str) -> Scenario:
    return next(s for s in SCENARIOS if s.name == name)
