"""The gates of NEO's computer tools, in one module: every check that decides whether a click,
key, press, drag, scroll or typed text may happen at all, or must be asked about first.

FROZEN. No workstream edits this file; a change goes through the integrator, is reviewed as a
security change, and may only make a gate stricter. Every actuation path — the pointer tools, AX
presses and value sets, keys and typing, menu presses, snaps, batched actions, board cells — calls
these helpers through this module at call time (``gates._ask_first(...)``, never a name bound by
``from … import``), so the one definition here is the one that runs, and a test that patches it
(here or through the package, which forwards) patches what the tools use.

- Accessibility and input permission: _needs_ax, _needs_input, _input_doubt (and the texts).
- The confirmation gate on what is under the pointer or pressed: _dialog_text, _ask_before_click,
  _ask_first, _gate_click, _gate_drop; on keys: _chord, _command_line, _doc_like,
  _return_runs_or_sends, _ask_before_keys.
- Protected targets — NEO's own windows and the `protected_apps` setting, by name, bundle id or
  pid, failing closed, never brought forward: _NEVER_TYPE_INTO, _protected, _protected_at,
  _protected_hit, _off_limits_el, _refuse_protected, _protected_out, _guard_front; the controls
  self-test's one exception, by pid, while it runs: allow_test_window, _test_window_front.
- The system (Apple) menu, off limits the same way wherever it shows: _in_system_menu,
  _outside_system_menu.
- Reading the screen taints the request: _taint, _read_from_screen, _refused, _watch, _changes.
"""

from __future__ import annotations

import asyncio
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from neo import runtime
from neo.agent import confirm, grants
from neo.agent.registry import ToolContext, ToolOutput, fence
from neo.config import decide_mode, settings
from neo.security import policy
from neo.tools.computer import apps, ax, pointer
from neo.tools.computer import input as inp

# ---- Accessibility ------------------------------------------------------------------------
# Seeing and acting in other apps (the AX tree, clicks, keys, typing, System Events' UI
# scripting) all take the Accessibility permission. Without it every such call fails, and a
# model that keeps getting failures tries its way around them (the Chess run: six AppleScripts,
# each a card). So the first one in a task stops the run and tells the user how to turn it on.
AX_OFF = (
    "NEO needs Accessibility to see and click in apps — turn NEO on in System Settings › Privacy & "
    "Security › Accessibility, then ask again"
)
_AX_STILL_OFF = (
    "Error: Accessibility is still off for NEO, and the user has already been told how to turn it on. "
    "Stop here and tell them; don't try AppleScript or any other way into the app."
)
# What osascript says when System Events may not drive another app's UI.
_AX_DENIED = re.compile(r"assistive access|not allowed to send keystrokes|\(-1719\)|\(-25211\)", re.I)


def _ax_trusted() -> bool:
    try:
        return bool(ax.is_trusted())
    except Exception:  # noqa: BLE001
        return False


def _host_app() -> str:
    """The app bundle this process runs from ("" = a bare executable, a script's interpreter)."""
    try:
        exe = Path(sys.executable).resolve()
    except (OSError, RuntimeError):
        return ""
    return next((p.stem for p in (exe, *exe.parents) if p.suffix == ".app"), "")


def _host_who() -> str:
    """Embedded in another app: that app, the one macOS must trust for this agent's permissions,
    named as its user knows it."""
    app = _host_app()
    return (
        f"{app}, the app this agent runs in"
        if app
        else f"the app this agent runs in ({Path(sys.executable).name}: for a script, the terminal or app that started it)"
    )


def _off_for_host(permission: str) -> str:
    """Embedded: the question when `permission` (a Privacy & Security pane) is off for the host app."""
    return (
        f"{permission} is off for {_host_who()} — turn it on in System Settings › Privacy & Security › "
        f"{permission}, then ask again"
    )


def _ax_off_texts() -> tuple[str, str]:
    """(the question, the error after it) when Accessibility is off. Embedded in another app, it is
    that app's process macOS must trust, not NEO.app: the texts name it."""
    if not runtime.library_bound():
        return AX_OFF, _AX_STILL_OFF
    app = _host_app()
    ask = _off_for_host("Accessibility")
    still = (
        f"Error: Accessibility is still off for {app or 'the app this agent runs in'}, and the user has already "
        "been told how to turn it on. Stop here and tell them; don't try AppleScript or any other way into the app."
    )
    return ask, still


def _ax_off_reply() -> str:
    """NEEDS_USER the first time in a task (the run stops, the user hears how to fix it); after
    that an error that tells the model to stop rather than look for another way in."""
    ask, still = _ax_off_texts()
    return confirm.ask_user(ask) if grants.first("accessibility-off") else still


def _needs_ax() -> str | None:
    """None with Accessibility on, else what the tool answers instead of trying."""
    return None if _ax_trusted() else _ax_off_reply()


# macOS checks separately whether a process may post input events ("control this computer"), and
# drops them silently when it may not. Its answer isn't taken as a hard stop — a wrong "no" would lock
# NEO out of every click and key — but no result claims success past it: a click that changed nothing
# says why, and every other input says macOS may have dropped it.
INPUT_DOUBT = (
    " (macOS reports NEO may not send clicks or keystrokes, so this may not have arrived — if nothing "
    "happened, turn NEO off and on in System Settings › Privacy & Security › Accessibility.)"
)


def _needs_input() -> str | None:
    """None when NEO may read the UI (Accessibility); else what the tool answers."""
    return _needs_ax()


def _input_doubt() -> str:
    """INPUT_DOUBT when macOS says this process may not post events, else ""."""
    try:
        return "" if inp.can_post() else INPUT_DOUBT
    except Exception:  # noqa: BLE001
        return ""


def _dialog_text(els=None, app: str = "") -> str:
    """The words in the app's focused window (a sheet or alert says what its "OK" does)."""
    if els is None:
        try:
            _, _, els = ax.window_elements(app or None)
        except Exception:  # noqa: BLE001
            return ""
    return " ".join((e.value or e.title) for e in els if e.role in ("AXStaticText", "AXTextArea"))[:2000]


def _ask_before_click(tool_name: str, a: dict, label: str, app: str = "", els=None) -> str | None:
    """A button that sends, buys, deletes, grants or publishes is the user's call — and so is a
    plain "OK" / "Continue" in a dialog that says it deletes or sends something."""
    context = _dialog_text(els, app) if policy._GENERIC_OK.match(label or "") else ""
    if not policy.commits(label, context):
        return None
    where = f" in {app}" if app else ""
    gate = confirm.gated(tool_name, a, f"click “{label.strip()[:60]}”{where}")
    return gate if isinstance(gate, str) else None


async def _ask_first(tool_name: str, a: dict, label: str, app: str = "", els=None) -> str | None:
    """:func:`_ask_before_click` from a handler: the dialog's words (a walk of its window, slow on
    a busy app) are read off the event loop; the question itself is staged on it (its card)."""
    if els is None and policy._GENERIC_OK.match(label or ""):
        try:
            _, _, els = await asyncio.to_thread(ax.window_elements, app or None)
        except Exception:  # noqa: BLE001
            els = []
    return _ask_before_click(tool_name, a, label, app, els)


async def _gate_click(a: dict, labels: list[tuple[str, str]], verb: str = "click") -> str | None:
    """Ask before a click on any of these (label, its app): the element meant, and what is really
    under the point."""
    for label, owner in dict.fromkeys((lb, app) for lb, app in labels if lb):
        if gate := await _ask_first(verb, a, label, owner):
            return gate
    return None


async def _gate_drop(a: dict, t: pointer.Target) -> str | None:
    """A drop can delete (onto the Trash) or send (into an AirDrop window): the user's call, as a
    click on such a button is. Judged by where it lands — never by the name of what is dragged."""
    src, dst = t.hits
    around = await asyncio.to_thread(ax.window_words, dst)
    if not policy.drop_commits(dst.label, around):
        return None
    where = f" in {dst.app}" if dst.app else ""
    gate = confirm.gated("drag", a, f"drag “{(src.label or 'that')[:40]}” onto “{(dst.label or 'that spot')[:40]}”{where}")
    return gate if isinstance(gate, str) else None


# ---- protected targets --------------------------------------------------------------

_NEVER_TYPE_INTO = {"neo-ui", "neo", "com.arnav.neo"}  # NEO's own overlay: its names and bundle id


def _protected(owner: str, pid: int = 0) -> bool:
    """An app NEO's input tools never touch: NEO's own overlay (its card's "Yes, do it" is the
    user's to press, never NEO's), and any app in the `protected_apps` setting — a host's own
    approval dialog, for one. Matched on the app's name, bundle id or executable's name (`pid`:
    the app's process, when that's what is known)."""
    if pid and pid == ax._test_window_pid:
        return False  # the controls self-test's own window, while it runs (allow_test_window)
    name = (owner or "").strip().lower()
    if name in _NEVER_TYPE_INTO:
        return True
    listed = {str(p).strip().lower() for p in settings().protected_apps}
    if not pid and not (name and listed):
        return False
    names = {name} | {n.lower() for n in (ax.app_names(pid) if pid else ax.names_of_app(owner))}
    return bool((names - {""}) & (_NEVER_TYPE_INTO | listed))


async def _protected_at(owner: str, x: float, y: float) -> bool:
    """Is the target at (x, y) — `owner`'s, as accessibility named it — one NEO never touches?
    When accessibility can't name the owner, the window server says whose window is there; when
    neither can, the answer is yes: nothing is touched that might be NEO's card."""
    if owner:
        return _protected(owner)
    hit = await asyncio.to_thread(ax.window_owner_at, x, y)
    if hit is None:
        return True
    name, pid = hit
    return bool(name or pid) and _protected(name, pid)


def _refuse_protected() -> str:
    return confirm.ask_user(
        "that's NEO's own window, the system menu or an app I'm not allowed to touch — I won't click or "
        "type there."
    )


def _protected_out() -> ToolOutput:
    """The refusal as a failed result (nothing was done) that still stops the run to tell the user."""
    return ToolOutput(_refuse_protected(), ok=False)


async def _protected_hit(h: ax.Hit, x: float, y: float) -> bool:
    """Is what :func:`ax.hit` found at (x, y) NEO's own window (any of its processes), the system
    menu or a protected app's? When it couldn't name an owner, the window server is asked; when
    nobody can say, yes."""
    if h.pid and ax.is_neo(h.pid, h.app):
        return True
    if h.el is not None and h.role in (*_MENU_ROLES, "") and await asyncio.to_thread(_in_system_menu, h.el, h.role):
        return True
    if h.pid or h.app:
        return _protected(h.app, h.pid)
    return await _protected_at("", x, y)


async def _off_limits_el(e: ax.Element, pid: int, owner: str) -> bool:
    """Is an ax_tree element NEO's own, in the system menu, or a protected app's? By its process
    when that is known, else by its owner's name, else by whose window is at its centre (fail
    closed)."""
    if pid and ax.is_neo(pid, owner):
        return True
    if e.role in _MENU_ROLES and e.ref is not None and await asyncio.to_thread(_in_system_menu, e.ref, e.role):
        return True
    if pid:
        return _protected(owner, pid)
    return await _protected_at(owner, *e.center)


# ---- the system menu --------------------------------------------------------------------
# The first menu of every app's menu bar is the system's (the Apple menu): Restart…, Shut Down…,
# Log Out, Force Quit… — served by other processes, for the whole Mac. Nothing of NEO's ever
# presses, clicks or answers anything in it: it is off limits like a protected app, by its place in
# the menu bar (never by its words, in whatever language they are).
_MENU_ROLES = frozenset({"AXMenuBarItem", "AXMenu", "AXMenuItem"})


def _in_system_menu(el, role: str = "") -> bool:
    """Is this Accessibility element (`role`: its role, when already known) the system menu or in
    it — the first item of a menu bar, or a menu or item under it? Only menus are walked up:
    anything else is in no menu bar. A menu whose way up can't be read might be the system's: yes
    (fail closed)."""
    node = el
    for _ in range(16):
        if node is None:
            return True
        role = role or ax._str(ax._attr(node, "AXRole"))
        if role == "AXMenuBarItem":
            bar = ax._attr(node, "AXParent")
            items = (ax._attr(bar, "AXChildren") or []) if bar is not None else []
            return not items or ax._same(items[0], node)
        if role not in _MENU_ROLES:
            return False
        node, role = ax._attr(node, "AXParent"), ""
    return True  # menus nested deeper than any app's: not told apart, so not touched


def _outside_system_menu(els: list) -> list:
    """The elements that are neither the system menu nor in it (a label match never picks one)."""
    return [e for e in els if not (e.role in _MENU_ROLES and e.ref is not None and _in_system_menu(e.ref, e.role))]


# The controls self-test's own test window (neo.selftest.controls), by its pid, while that test
# runs: a separate app NEO started — NEO's child, so macOS holds NEO responsible for it — never NEO's
# overlay. It is the one exception to the rules above, should they ever see it as NEO's: by pid
# (ax._test_window_pid, which ax.is_neo and _protected honour), so nothing else that shares a name or
# a bundle gets through.


@contextmanager
def allow_test_window(pid: int) -> Iterator[None]:
    """For the length of a controls self-test: its test window (this pid, and only it) may be typed
    into even if a protection of NEO's own windows would take it for one of them."""
    ax._test_window_pid = int(pid)
    try:
        yield
    finally:
        ax._test_window_pid = 0


def _test_window_front() -> bool:
    if not ax._test_window_pid:
        return False
    from neo.tools.computer.focus import front_app

    return front_app()[1] == ax._test_window_pid


async def _guard_front(want: str = "") -> str | None:
    """Keystrokes only ever go to the app NEO is working in.

    `want` (or else the app NEO last opened/activated, if that was recent) must be frontmost —
    brought forward if needed. If macOS won't, nothing is sent and the user is told, instead of
    typing into whatever they happen to be using."""
    goal = want or apps.target()
    if goal and _protected(goal):  # never brought forward, even to refuse it afterwards
        return _refuse_protected()
    front = apps.front_name()
    if goal and front.lower() != goal.lower():
        ok, real = await apps.bring_front(goal, wait=0.3)
        if not ok:
            return confirm.ask_user(
                f"{real} isn't in front ({apps.front_name() or 'another app'} is), "
                f"so I didn't type or press anything — click {real} and say it again."
            )
        front = real
    if _protected(front) and not _test_window_front():
        return confirm.ask_user("click where you want me to type first.")
    return None


# ---- what reading the screen does to the request ------------------------------------


def _taint(c: ToolContext, name: str) -> None:
    """Words read off the screen went into this result (a label, a sheet's text, even in a
    refusal): the request has read untrusted content, and a "yes for this task" given before
    doesn't cover what follows it (grants.read_untrusted)."""
    if name not in c.taint:
        c.taint.append(name)
    grants.read_untrusted(name)


def _read_from_screen(c: ToolContext, text: str) -> str:
    """A reply that quotes the screen (a sheet in the way, fenced) counts as reading it."""
    if '<untrusted source="screen">' in text:
        _taint(c, "screen")
    return text


def _refused(c: ToolContext, r: pointer.Refused) -> ToolOutput:
    if r.protected:
        return _protected_out()
    if r.screen:
        _taint(c, "screen")
    return ToolOutput(str(r), ok=False)


async def _watch(
    c: ToolContext, pid: int, pts: list[tuple[float, float]], hover: tuple[float, float] | None = None
) -> pointer.Look | None:
    """The look before an input a model chose (the user's own commands aren't second-guessed:
    they are watching), with the pointer already at ``hover``, where the input will leave it."""
    return await pointer.before(pid, pts, hover) if c.origin == "agent" else None


async def _changes(before: pointer.Look, pid: int, c: ToolContext) -> str:
    """What visibly followed a key press or an AX press — only what changed (a press may rightly
    change nothing anyone can see)."""
    out = await pointer.scene_change(before, pid)
    if out.changed is not True:
        return ""
    text = " " + out.said
    if out.details:
        _taint(c, "screen")
        text += "\n" + fence("screen", out.details)
    return text


# ---- keys that run or send ------------------------------------------------------------

_DOC_APPS = {"notes", "textedit", "pages", "stickies", "numbers", "keynote", "freeform"}  # legacy's list


def _doc_like(name: str, pid: int | None) -> bool:
    """The front app makes a fresh note or document with ⌘N: by evidence while write_note is on
    (neo.decide.kinds.note.doc_like), else by legacy's list of names."""
    if decide_mode("write_note") != "on":
        return name.lower() in _DOC_APPS
    from neo.decide import inventory
    from neo.decide.kinds import note

    app = inventory.app_for_pid(pid) if pid else None
    return app is not None and note.doc_like(app)


async def _command_line(pid: int) -> bool:
    """Is the keyboard on a command line, where a Return runs what's on it? A terminal app (its own
    processes say so: focus.hosts_shell), or an app's terminal pane while the keyboard is in it."""
    from neo.tools.computer.focus import hosts_shell

    kind = await asyncio.to_thread(hosts_shell, pid)
    if kind == "pane":
        return await asyncio.to_thread(ax.focused_terminal, pid)
    return bool(kind)


def _chord(keys: str) -> set[str]:
    """The modifiers and key of a chord, by name ({"cmd", "shift", "d"}), however it was written."""
    try:
        flags, key = inp.parse_chord(keys)
    except ValueError:
        return {p.strip().lower() for p in keys.replace(" ", "").split("+") if p.strip()}
    return {name for name, f in inp._MODS.items() if flags & f} | {key}


async def _return_runs_or_sends(pid: int, name: str) -> str:
    """What a Return does where the keyboard focus is, by evidence rather than the app's name:
    "runs" on a command line (a terminal app, an app's terminal pane with the keyboard in it),
    "sends" in a multi-line input outside a document editor (a chat box, a web composer), ""
    otherwise (a document, an address or search field)."""
    if await _command_line(pid):
        return "runs"
    role, _ = await asyncio.to_thread(ax.focused_role)
    multi = role == "AXTextArea" or (role not in ax._EDITABLE and role != "" and await asyncio.to_thread(ax.focused_editable))
    if multi and not await asyncio.to_thread(_doc_like, name, pid):
        return "sends"
    return ""


async def _ask_before_keys(a: dict, keys: str, app: str, c: ToolContext, pid: int = 0) -> str | None:
    """Keys that run or send something: Return in a terminal runs whatever is on its command
    line; ⌘⇧D sends the open Mail message; Return in a chat box sends the message — asked first
    once this request has read something untrusted (a page or an email could have asked for it)."""
    ch, low = _chord(keys), app.lower()
    submit = bool(ch & {"return", "enter"}) or ("ctrl" in ch and bool(ch & {"m", "j"}))
    what, detail = "", ""
    does = await _return_runs_or_sends(pid, app) if submit else ""
    if does == "runs":
        text = await asyncio.to_thread(ax.focused_element_text)
        line = next((ln for ln in reversed(text.splitlines()) if ln.strip()), "")
        if len(line) > 2000:
            line = "…" + line[-2000:] + "\n(the start of this line is cut off here)"
        what, detail = f"press Return in {app} — it runs the command on its line", line or "(couldn't read the line)"
    elif low == "mail" and ch == {"cmd", "shift", "d"}:
        what = "send the message that's open in Mail"
        _, win = await asyncio.to_thread(ax.focused_window, app)
        detail = f"Window: {ax._str(ax._attr(win, 'AXTitle', ''))}" if win is not None else ""
    elif does == "sends" and c.tainted():
        what = f"press Return in {app} — it sends the message"
    if not what:
        return None
    gate = confirm.gated("hotkey", a, what, detail=detail)
    return gate if isinstance(gate, str) else None
