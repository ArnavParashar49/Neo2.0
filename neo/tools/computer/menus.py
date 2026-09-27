"""Menu commands by meaning, the sheet a command opens, and typing into a named field.

"start a new game" → the app's own New Game menu item, pressed through Accessibility (never a
guessed shortcut) → the sheet it opened → its Start button, with no model call, but only under
the strict rules below. Anything else is asked (the user's words) or handed to the model with the
choices as ids (a model's words).

* **Never the system (Apple) menu.** The menu bar's first item is left out when the menus are read
  and refused again when an item is pressed; nothing in the Services submenu (other apps'
  commands) or in NEO's own or a protected app's menus.
* **A menu item is gated twice**: like a button, by its title (``policy.commits``, in any language
  the OS string index reads back into English), and like its keyboard shortcut (the key gate).
* **A sheet is answered by itself only when all of these hold** (DECISIONS 46): the request is the
  user's own words (origin "user"); the sheet belongs to the menu's own app (the same pid); exactly
  one of its buttons has its words, as they are, in the user's words; that button doesn't commit;
  the sheet's text carries no loss or consequence wording (checked whatever the button); and the
  request is fully used up by the menu item, that button and filler words. Otherwise NEO asks.
* **Fail closed on words NEO can't read**: a label in a script with no English back-translation
  (Hindi while the OS index isn't built) is never pressed by itself - NEO asks first.
* **Screen text is data**: the sheet's words are fenced for a model and only ever matched against
  the user's own words.
* **Which element is pressed is decided once**: the item (or button) gated is the one pressed.

Matching is lexical and live (an exact title, or the title's words in order in the request). The
OS-strings stage (a Hindi request, an English menu) runs in shadow unless ``settings.resolver`` is
"live" (DECISIONS 40); every grounding is logged to the shadow log without its words.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from neo.agent import confirm
from neo.agent.confirm import Halt
from neo.agent.registry import ToolContext, ToolOutput, fence
from neo.config import settings
from neo.decide import uistrings
from neo.security import policy
from neo.tools.computer import apps, ax

log = logging.getLogger(__name__)

__all__ = [
    "TOOL",
    "FAST_PATHS",
    "FIELD_FAST_PATHS",
    "menu_command",
    "by_utterance",
    "fill_field",
    "parse_field",
]

# A by-meaning stage for Latin paraphrases (the commands resolver's MiniLM+Laya agreement), plugged
# in at integration: (request, candidate titles) -> the index it picks, or None. Consulted only while
# the resolver is promoted to "live"; a command it picks never gets its sheet answered by itself.
meaning: Callable[[str, list[str]], int | None] | None = None

_SETTLE_S = 0.6  # how long a press is watched for a sheet, a new window or a title change
_CLOSE_S = 1.0  # how long an answered sheet is watched for closing
_EVERY_S = 0.03
_MENU_TTL_S = 2.0  # a menu bar read is reused this long (a claim, then the run)
_OPTIONS = 6  # a shortlist handed on is at most this long
SYSTEM_MENU = "Error: that's in the system menu (the Apple menu) — NEO never uses it."

# Clock and sleep, for the tests' fake clock.
_now = time.monotonic
_sleep = asyncio.sleep


class _Gates:
    """The frozen gate helpers (the confirmation gate, the key gate, the protected-app checks), looked
    up by name on every use so a test's patch reaches them: the package's own name first (it
    re-exports them), else computer/gates once the package is split."""

    def __getattr__(self, name: str) -> Any:
        from neo.tools import computer

        found = getattr(computer, name, None)
        if found is None:
            from neo.tools.computer import gates  # type: ignore[attr-defined]

            found = getattr(gates, name)
        return found


_GATES = _Gates()


def _gates() -> Any:
    return _GATES


# ---- words ------------------------------------------------------------------------------------------

_ARTICLES = frozenset({"a", "an", "the"})
# What a request may say around a command's name without naming anything else: politeness, the
# frame verbs, control nouns, and their Hinglish and Hindi forms. Never "ok" / "yes" (a sheet's
# button words) and never "with" (what follows it constrains the command: "… with two players").
_FILLER_WORDS = (
    "a an the please pls can could would will you kindly just now then and also so for me us i to go ahead "
    "click tap press hit select choose pick use run do open start begin launch make create "
    "button menu item option command entry link icon from in on under inside of this that it app window "
    "karo kar do dena kardo kijiye kariye karein dabao daba dabaiye chuno chun lo pe par pr wala wali waala "
    "wale vala ko ek zara jara se mein me main shuru chalu kholo khol "
    "करो करें कर दो दें दीजिए कीजिए दबाओ दबाएं दबाइए चुनो चुनें चुनिए पर को वाला वाली वाले बटन मेनू मेन्यू से में "
    "शुरू चालू खोलो खोलें क्लिक ज़रा कृपया एक"
)
_FILLER = frozenset(uistrings.norm(w) for w in _FILLER_WORDS.split())
_QUESTION = re.compile(
    r"\?|^\s*(?:what|what's|whats|how|why|where|which|who|when|is|are|does|do\s+you|kya|kaise|kyun|kahan|kaun)\b"
    r"|(?:क्या|कैसे|क्यों|कहाँ|कहां|कौन)",
    re.I,
)
_ARROW = re.compile(r"\s*(?:->|>|›|→|»)\s*")
_LEAD_FRAME = re.compile(
    r"^(?:(?:please|pls|now|and|then|so|hey|neo)[\s,]+)*(?:(?:can|could|would)\s+you\s+)?"
    r"(?:(?:click|tap|press|hit|select|choose|pick|use|run|open|go\s+to)\s+(?:on\s+)?)?(?:the\s+)?",
    re.I,
)
_TAIL_FRAME = re.compile(
    r"(?:\s+(?:button|menu\s+item|item|option|command|entry))?"
    r"(?:\s+(?:(?:pe|par|pr)\s+)?(?:click\s+|select\s+)?(?:karo|kar\s+do|kardo|kijiye|dabao|daba\s+do|chuno|chun\s+lo))?"
    r"(?:\s+(?:वाला\s+)?बटन)?"
    r"(?:\s+(?:पर\s+)?(?:क्लिक\s+कर(?:ो|ें|िए|\s+दो)|दबा(?:ओ|एँ|एं|इए|\s+दो)|चुन(?:ो|ें|िए|\s+लो)))?"
    r"\s*[.!।]*$",
    re.I,
)


def _toks(text: str) -> list[str]:
    return list(_toks_of(text or ""))


@lru_cache(maxsize=16384)
def _toks_of(text: str) -> tuple[str, ...]:
    """A text's words as matching sees them (articles dropped) - titles repeat, so kept."""
    return tuple(w for w in uistrings.words(text) if w not in _ARTICLES)


def _span(req: list[str], want: list[str]) -> tuple[int, int] | None:
    """Where ``want``'s words stand in ``req``, in order and next to each other (first place)."""
    n = len(want)
    if not n or n > len(req):
        return None
    for i in range(len(req) - n + 1):
        if req[i : i + n] == want:
            return i, i + n
    return None


def _strip_frame(phrase: str) -> str:
    s = " ".join((phrase or "").split()).strip(" ,.!?।\"“”'‘’")
    for _ in range(3):
        t = _TAIL_FRAME.sub("", _LEAD_FRAME.sub("", s)).strip(" ,.!?।\"“”'‘’")
        if t == s:
            break
        s = t
    return s


def _leftover(req: list[str], used: set[int], extra: frozenset[str]) -> list[str]:
    """The request's words nothing accounts for: not the command's, not the button's, not the app's
    name, not filler."""
    return [w for i, w in enumerate(req) if i not in used and w not in _FILLER and w not in extra]


def _sounds_destructive(text: str) -> bool:
    try:
        from neo.reflex.schema import sounds_destructive

        return bool(text) and sounds_destructive(text)
    except Exception:  # noqa: BLE001 - can't tell: take it as destructive (the picks abstain)
        return True


def _resolver_mode() -> str:
    return str(getattr(settings(), "resolver", "shadow") or "shadow")


def _script(text: str) -> str:
    return "latin" if not uistrings.non_latin(text) else ("mixed" if re.search(r"[A-Za-z]", text) else "other")


async def _log(**facts: Any) -> None:
    """One line in the shadow log per grounding: stages, counts, flags - never a label or a word."""
    if _resolver_mode() == "off":
        return
    try:
        from neo.decide import shadow

        await asyncio.to_thread(shadow.record, "ui_resolve", facts)
    except Exception:  # noqa: BLE001 - a log line is never worth a failed command
        pass


# ---- the menu bar -----------------------------------------------------------------------------------


@dataclass
class Item:
    """A menu command (the AX core's reader gives the same fields): its title, the menu titles down
    to it, whether it can be chosen now, its keyboard shortcut, its AX element and app."""

    title: str
    path: tuple[str, ...]
    enabled: bool = True
    key: str = ""
    mods: int = 0
    vkey: int = 0
    checked: bool = False
    ref: Any = None
    pid: int = 0

    def where(self) -> str:
        return " › ".join(self.path)


_MENU_SHIFT, _MENU_OPTION, _MENU_CONTROL, _MENU_NO_COMMAND = 1, 2, 4, 8


def _chord(item: Any) -> str:
    """The item's shortcut as the key gate reads it ("cmd+shift+d"); "" when it has none."""
    got = getattr(item, "chord", None)
    if isinstance(got, str):
        return got
    key = str(getattr(item, "key", "") or "").strip().lower()
    if not key:
        return ""
    mods = int(getattr(item, "mods", 0) or 0)
    names = [] if mods & _MENU_NO_COMMAND else ["cmd"]
    names += [m for bit, m in ((_MENU_SHIFT, "shift"), (_MENU_OPTION, "alt"), (_MENU_CONTROL, "ctrl")) if mods & bit]
    return "+".join([*names, {" ": "space"}.get(key, key)])


def _where(item: Any) -> str:
    return " › ".join(getattr(item, "path", ()) or (item.title,))


def _walk_menu(menu: Any, path: tuple[str, ...], out: list[Item], pid: int, depth: int, limit: int) -> None:
    for mi in ax._attr(menu, "AXChildren") or []:
        if len(out) >= limit:
            return
        if ax._str(ax._attr(mi, "AXRole")) != "AXMenuItem":
            continue
        title = ax._str(ax._attr(mi, "AXTitle")).strip()
        if not title:
            continue  # a separator
        try:
            mods = int(ax._attr(mi, "AXMenuItemCmdModifiers") or 0)
        except (TypeError, ValueError):
            mods = 0
        en = ax._attr(mi, "AXEnabled")
        item = Item(title, (*path, title), en is None or bool(en), ax._str(ax._attr(mi, "AXMenuItemCmdChar")), mods, ref=mi, pid=pid)
        out.append(item)
        if depth < 4:
            for sub in ax._attr(mi, "AXChildren") or []:
                if ax._str(ax._attr(sub, "AXRole")) == "AXMenu":
                    _walk_menu(sub, item.path, out, pid, depth + 1, limit)


def _read_fallback(pid: int, limit: int = 1500) -> list[Item]:
    """The menu bar's commands by plain attribute reads (before the AX core's batched reader):
    never the bar's first item, the system menu."""
    bar = ax._attr(ax._app_el(pid), "AXMenuBar")
    tops = list(ax._attr(bar, "AXChildren") or []) if bar is not None else []
    out: list[Item] = []
    for i, top in enumerate(tops):
        if i == 0 and len(tops) >= 2:
            continue
        if ax._str(ax._attr(top, "AXRole")) != "AXMenuBarItem":
            continue
        name = ax._str(ax._attr(top, "AXTitle")).strip()
        for m in ax._attr(top, "AXChildren") or []:
            _walk_menu(m, (name,), out, pid, 1, limit)
    return out


def _in_system_menu(item: Any) -> bool:
    """Is the item in the system (Apple) menu - under the menu bar's first item? Told by where it
    sits, never by its words; when it can't be told, yes (nothing is pressed)."""
    ref = getattr(item, "ref", None)
    if ref is None:
        return True
    fn = getattr(ax, "in_system_menu", None)
    try:
        if fn is not None:
            return bool(fn(ref))
        p = ref
        for _ in range(10):
            if p is None:
                return False
            if ax._str(ax._attr(p, "AXRole")) == "AXMenuBarItem":
                bar = ax._attr(p, "AXParent")
                items = list(ax._attr(bar, "AXChildren") or []) if bar is not None else []
                return len(items) >= 2 and ax._same(items[0], p)
            p = ax._attr(p, "AXParent")
    except Exception:  # noqa: BLE001
        return True
    return False


_menus: dict[int, tuple[float, list]] = {}


def _read(pid: int) -> list:
    fn = getattr(ax, "menu_items", None)
    return list(fn(pid)) if fn is not None else _read_fallback(pid)


def _commands(pid: int) -> list:
    """The app's menu commands NEO may choose: its own (this pid), not the system menu, not in the
    Services submenu, and not a submenu's own title (it holds commands, it isn't one)."""
    hit = _menus.get(pid)
    if hit is not None and _now() - hit[0] < _MENU_TTL_S:
        return hit[1]
    items = [i for i in _read(pid) if getattr(i, "title", "") and (not getattr(i, "pid", 0) or i.pid == pid)]
    services = {uistrings.norm("Services")} | {uistrings.norm(f) for f in uistrings.localized("Services")}
    parents = {tuple(uistrings.norm(p) for p in i.path[:-1]) for i in items}
    keep, seen = [], set()
    for i in items:
        path = tuple(uistrings.norm(p) for p in i.path)
        if path in parents or path in seen or any(p in services for p in path[:-1]):
            continue
        seen.add(path)
        keep.append(i)
    _menus[pid] = (_now(), keep)
    return keep


def _forget(pid: int) -> None:
    _menus.pop(pid, None)


# ---- grounding ---------------------------------------------------------------------------------------


@dataclass
class _Ask:
    item: str = ""  # the command's words, when given (a model's argument, an explicit phrase)
    menu: str = ""  # the menu named, if any
    path: tuple[str, ...] = ()  # "File > Export > PDF"
    text: str = ""  # the request grounded by meaning (the whole utterance), when no item is given
    user: str = ""  # the user's own words (origin "user"), which a sheet's answer must come from


@dataclass
class _Pick:
    item: Any = None
    stage: str = ""  # exact | lexical | os
    used: set[int] = field(default_factory=set)  # the user's words the command took
    options: list = field(default_factory=list)  # several matched: the candidates
    shadow_os: bool = False  # the OS-strings stage would have picked (while in shadow)


_FROM_MENU = re.compile(r"^(?P<item>.+?)\s+(?:from|in|under|on|inside)\s+(?:the\s+)?(?P<menu>[^,;]+?)\s+menu$", re.I)
_OPEN_THEN = re.compile(
    r"^(?:open|click(?:\s+on)?|go\s+to)\s+(?:the\s+)?(?P<menu>.+?)\s+menu\s*,?\s+(?:and\s+)?(?:then\s+)?(?P<item>.+)$", re.I
)
_CLICK_THEN = re.compile(
    r"^(?:click|tap|choose|select|open)\s+(?:on\s+)?(?:the\s+)?(?P<menu>[^,]+?)\s*,?\s+(?:and\s+)?then\s+(?P<item>.+)$", re.I
)
_POLITE = re.compile(r"^(?:(?:please|pls|now|and|then|so|hey|neo)[\s,]+)*(?:(?:can|could|would)\s+you\s+)?", re.I)
_MENU_SE = re.compile(r"^(?P<menu>.+?)\s+(?:menu|मेनू|मेन्यू)\s+(?:se|mein|me|main|से|में)\s+(?P<item>.+)$", re.I)


def _explicit(text: str) -> tuple[str, str, tuple[str, ...]] | None:
    """(item, menu, path) when the words name the menu ("New Window from the File menu", "File >
    New Window", "click File then New Window", "File menu se New Window chuno")."""
    s = _POLITE.sub("", " ".join((text or "").split()).strip(" .!?।"))
    if _ARROW.search(s):
        parts = tuple(p.strip(" \"“”'") for p in _ARROW.split(s) if p.strip())
        return (parts[-1], parts[0], parts) if len(parts) >= 2 else None
    for rx in (_FROM_MENU, _OPEN_THEN, _MENU_SE, _CLICK_THEN):
        m = rx.match(s)
        if m:
            return m.group("item"), m.group("menu"), ()
    return None


def _ask_of(a: dict, c: ToolContext) -> _Ask | None:
    user = c.user_text if c.origin == "user" else ""
    path_arg = str(a.get("path") or "").strip()
    item = str(a.get("item") or "").strip()
    menu = str(a.get("menu") or "").strip()
    text = str(a.get("utterance") or "").strip()
    path: tuple[str, ...] = ()
    if path_arg or _ARROW.search(item):
        parts = tuple(p.strip(" \"“”'") for p in _ARROW.split(path_arg or item) if p.strip())
        if len(parts) >= 2:
            item, menu, path = parts[-1], parts[0], parts
        elif parts:
            item = parts[0]
    if not item and text:
        got = _explicit(text)
        if got is not None:
            item, menu, path = got
    item = _strip_frame(item)
    menu = re.sub(r"\s+(?:menu|मेनू|मेन्यू)$", "", _strip_frame(menu), flags=re.I)
    if not item and not text:
        return None
    return _Ask(item, menu, path, "" if item else text, user)


def _in_menu(it: Any, ask: _Ask) -> bool:
    if not ask.menu and len(ask.path) < 3:
        return True
    parents = [uistrings.norm(p) for p in it.path[:-1]]
    want = [uistrings.norm(p) for p in (ask.path[:-1] if ask.path else (ask.menu,))]
    at = 0
    for w in want:  # each named menu, in order, among the item's menus
        w_toks = _toks(w)
        while at < len(parents) and parents[at] != w and _toks(parents[at]) != w_toks:
            at += 1
        if at >= len(parents):
            return False
        at += 1
    return True


def _best(hits: list[tuple[int, Any, set[int]]]) -> tuple[Any, set[int], list]:
    """The longest title that matched, if only one title did (the enabled one of equal titles)."""
    if not hits:
        return None, set(), []
    top = max(h[0] for h in hits)
    best = [h for h in hits if h[0] == top]
    by_title: dict[str, list] = {}
    for h in best:
        by_title.setdefault(uistrings.norm(h[1].title), []).append(h)
    if len(by_title) == 1:
        same = next(iter(by_title.values()))
        live = [h for h in same if getattr(h[1], "enabled", True)]
        if len(live) == 1 or (not live and len(same) == 1):
            return (live or same)[0][1], (live or same)[0][2], []
        return None, set(), [h[1] for h in (live or same)]
    return None, set(), [h[1] for h in best]


def _ground(ask: _Ask, items: list) -> _Pick:
    cands = [i for i in items if _in_menu(i, ask)]
    if ask.item:
        want = _toks(ask.item)
        exact = [(len(want), i, set()) for i in cands if _toks(i.title) == want]
        it, used, opts = _best(exact)
        if it is not None or opts:
            return _Pick(it, "exact", used, opts)
        within = [(len(t), i, set()) for i in cands if (t := _toks(i.title)) and _span(want, t)]
        it, used, opts = _best(within)
        if it is None and not opts:
            around = [(len(want), i, set()) for i in cands if _span(_toks(i.title), want)]
            it, used, opts = _best(around)
        if it is not None or opts:
            return _Pick(it, "lexical", used, opts)
        request = ask.item
    else:
        req = _toks(ask.text)
        hits = [(len(t), i, set()) for i in cands if (t := _toks(i.title)) and _span(req, t)]
        it, used, opts = _best(hits)
        if it is not None or opts:
            return _Pick(it, "lexical", used, opts)
        request = ask.text
    # The OS's own strings: a Hindi request for an English menu, or the other way round.
    os_hits = [i for i in cands if uistrings.os_match(request, i.title)]
    titles = {uistrings.norm(i.title) for i in os_hits}
    if len(titles) == 1:
        if _resolver_mode() == "live":
            return _Pick(os_hits[0], "os")
        return _Pick(shadow_os=True)
    fn = meaning
    if fn is not None and cands and _resolver_mode() == "live" and not uistrings.non_latin(request):
        try:
            k = fn(request, [i.title for i in cands])
        except Exception:  # noqa: BLE001 - a stage that fails picks nothing
            k = None
        if isinstance(k, int) and 0 <= k < len(cands):
            return _Pick(cands[k], "meaning")
    return _Pick()


# ---- acting ------------------------------------------------------------------------------------------


def _with_effect(out: ToolOutput, *, changed: bool | None, scene_changed: bool = False, sheet: str = "") -> ToolOutput:
    """Attach what the command changed (ToolOutput.effect, where this build has it)."""
    try:
        from neo.agent import registry as reg

        cls = getattr(reg, "Effect", None)
        if cls is not None and hasattr(out, "effect"):
            out.effect = cls(changed=changed, scene_changed=scene_changed, sheet=sheet, looked=False)
    except Exception:  # noqa: BLE001 - an Effect this build can't make: the text still says it
        pass
    return out


def _with_handoff(out: ToolOutput, options: list[str]) -> ToolOutput:
    """Attach the multiple choice a model should pick from (ToolOutput.handoff, where it exists)."""
    if options and hasattr(out, "handoff"):
        try:
            out.handoff = tuple(options[:_OPTIONS])
        except Exception:  # noqa: BLE001
            pass
    return out


async def _target(a: dict) -> tuple[str, int] | str | ToolOutput:
    """(name, pid) of the app whose menus are meant: the one named (brought forward, never a
    protected one), else the app in front of NEO's own window - or why not."""
    g = _gates()
    want = str(a.get("app") or "").strip()
    if want:
        if err := await g._guard_front(want):  # refuses a protected app before bringing it forward
            return err
        name, pid = await asyncio.to_thread(ax.frontmost_app)
    else:
        name, pid = await asyncio.to_thread(ax.working_app)
    if not pid:
        return "Error: no app is in front to use the menus of."
    if ax.is_neo(pid, name) or g._protected(name, pid):
        return g._protected_out()
    return name, pid


def _scene(pid: int) -> Any:
    return ax.scene(pid)


def _differs(b: Any, s: Any) -> bool:
    if not (getattr(b, "known", False) and getattr(s, "known", False)):
        return False
    return (s.sheet, s.title, s.front_pid, s.focus) != (b.sheet, b.title, b.front_pid, b.focus)


async def _watch(pid: int, before: Any, timeout: float, until=_differs) -> Any:
    """The scene, looked at until ``until(before, now)`` or ``timeout``."""
    deadline = _now() + timeout
    while True:
        now = await asyncio.to_thread(_scene, pid)
        if until(before, now) or _now() >= deadline:
            return now
        await _sleep(_EVERY_S)


def _press_item(item: Any) -> str:
    """AXPress on the item, through the one door every AX action takes (the watcher, the effects
    switch). The AX core's press_menu_item refuses the system menu and disabled items again."""
    fn = getattr(ax, "press_menu_item", None)
    if fn is not None:
        return fn(item)
    el = ax.Element(id="menu", role="AXMenuItem", title=item.title, value="", x=0, y=0, w=0, h=0, actions=["AXPress"], ref=item.ref)
    out = ax.press_el(el)
    return out if out.startswith("Error") else f"Chose {_where(item)}"


def _blocker(pid: int) -> Any:
    """The sheet or dialog now in the way in this app, with its controls as ids (AX core), or None."""
    fn = getattr(ax, "blockers", None)
    if fn is None:
        return None
    for b in fn(pid) or []:
        if getattr(b, "kind", "") in ("sheet", "modal", "dialog"):
            return b
    return None


async def _gate_item(item: Any, name: str, pid: int, a: dict, c: ToolContext, pick: _Pick, ask: _Ask) -> str | ToolOutput | None:
    g = _gates()
    where = _where(item)
    if not all(uistrings.readable(p) for p in item.path):  # fail closed: words NEO can't read in English
        gate = confirm.gated("menu_command", a, f"choose “{where[:80]}” in {name}")
        if isinstance(gate, str):
            return gate
    # sends, deletes, quits…: the item with its submenus ("Share › AirDrop"), not the top menu's name
    label = " › ".join(item.path[1:]) or item.title
    if gate := await g._ask_first("menu_command", a, label, name, []):
        return gate
    if (chord := _chord(item)) and (gate := await g._ask_before_keys(a, chord, name, c, pid)):
        return gate  # the same question its keyboard shortcut would ask
    if ask.user and pick.stage != "exact" and _sounds_destructive(ask.user) and not policy.commits(label):
        # "remove …" grounded on a title that removes nothing: not what was meant
        return ToolOutput(f"Error: “{where}” doesn't sound like what was asked — which command?", ok=False)
    return None


async def _run(a: dict, c: ToolContext, *, claim: bool) -> ToolOutput | str | None:
    g = _gates()
    ask = _ask_of(a, c)
    if ask is None:
        return None if claim else ToolOutput("Error: which menu command? Give its name (and its menu, if known).", ok=False)
    if claim and (not ask.user or _QUESTION.search(ask.user)):
        return None
    if err := g._needs_input():
        return None if claim else err
    got = await _target(a)
    if not isinstance(got, tuple):
        return None if claim else got
    name, pid = got
    await asyncio.to_thread(uistrings.for_pid, pid)  # the app's own strings, on first contact
    items = await asyncio.to_thread(_commands, pid)
    pick = _ground(ask, items)
    req = _toks(ask.user)
    extra = frozenset(_toks(name)) | frozenset(_toks(ask.menu)) | frozenset(w for p in ask.path for w in _toks(p))
    if pick.item is not None:
        pick.used = _used(req, pick)
    rest = _leftover(req, pick.used, extra) if pick.item is not None and ask.user else []
    await _log(want="menu", stage=pick.stage or "none", acted=pick.item is not None and not (claim and rest),
               os_would=pick.shadow_os, n=len(items), ambiguous=bool(pick.options), script=_script(ask.user or ask.item or ask.text),
               origin=c.origin)  # fmt: skip
    if pick.item is None:
        if claim:
            return None
        return _no_pick(ask, pick, name, items, c)
    if claim and rest:
        return None  # the request says more than this command: the model decides
    item = pick.item
    where = _where(item)
    if not getattr(item, "enabled", True):
        return ToolOutput(f"Error: “{where}” can't be chosen right now (it's greyed out).", ok=False)
    if await asyncio.to_thread(_in_system_menu, item):
        return None if claim else ToolOutput(SYSTEM_MENU, ok=False)
    if gate := await _gate_item(item, name, pid, a, c, pick, ask):
        return gate
    before = await asyncio.to_thread(_scene, pid)
    said = await asyncio.to_thread(_press_item, item)
    _forget(pid)
    if said.startswith("Error"):
        return ToolOutput(said, ok=False)
    if await asyncio.to_thread(ax.regular, pid):
        apps.set_target(name)  # "…, then type": the typing goes to this app
    after = await _watch(pid, before, _SETTLE_S)
    if getattr(after, "sheet", "") and after.sheet != getattr(before, "sheet", ""):
        return await _answer_sheet(item, ask, pick, name, pid, a, c, after, said, req, extra)
    out = ToolOutput(said + ("" if said.endswith(".") else "."))
    changed: bool | None = None
    if _differs(before, after):
        changed = True
        if after.title != before.title and after.title:
            g._taint(c, "screen")
            out.text += "\n" + fence("screen", f"window: “{after.title[:80]}”")
    elif getattr(before, "known", False) and getattr(after, "known", False):
        changed = False  # (a command may rightly change nothing anyone sees: not a failure)
    return _with_effect(out, changed=changed, scene_changed=bool(changed))


def _used(req: list[str], pick: _Pick) -> set[int]:
    """Which of the user's words the chosen command accounts for."""
    title = pick.item.title
    if pick.stage == "meaning":
        return set(range(len(req)))  # the resolver read the whole request as this command
    if pick.stage == "os":
        forms = uistrings.localized(title) + uistrings.english_sources(title) + (title,)
        words = {w for f in forms for w in uistrings.content_words(f)}
        return {k for k, w in enumerate(req) if w in words}
    sp = _span(req, _toks(title))
    return set(range(*sp)) if sp else set()


def _no_pick(ask: _Ask, pick: _Pick, name: str, items: list, c: ToolContext) -> ToolOutput:
    g = _gates()
    said = ask.item or ask.text
    if pick.options:
        g._taint(c, "screen")
        opts = [_where(i) for i in pick.options[:_OPTIONS]]
        out = ToolOutput(
            f"Error: several menu commands in {name} match “{said[:60]}” — which one?\n" + fence("screen", "\n".join(opts)),
            ok=False,
        )
        return _with_handoff(out, opts)
    tops = list(dict.fromkeys(i.path[0] for i in items if getattr(i, "path", ())))[:12]
    listing = ("\n" + fence("screen", "its menus: " + ", ".join(tops))) if tops else ""
    if tops:
        g._taint(c, "screen")
    where = f" in the {ask.menu} menu" if ask.menu else ""
    return ToolOutput(f"Error: no menu command{where} of {name} matches “{said[:60]}”.{listing}", ok=False)


def _button_label(b: Any) -> str:
    return (getattr(b, "title", "") or getattr(b, "value", "") or "").strip()


def _control(b: Any) -> str:
    """A sheet's control for a model: its id, kind and label - never where it is (a brain that can't
    see never gets coordinates; the id is what it presses)."""
    role = str(getattr(b, "role", "") or "").removeprefix("AX") or "Control"
    off = "" if getattr(b, "enabled", True) else " (disabled)"
    return f"{getattr(b, 'id', '?')} {role} “{_button_label(b)[:60]}”{off}"


async def _answer_sheet(
    item: Any, ask: _Ask, pick: _Pick, name: str, pid: int, a: dict, c: ToolContext, after: Any, said: str,
    req: list[str], extra: frozenset[str],
) -> ToolOutput | str:  # fmt: skip
    """The command opened a sheet or dialog: answer it by itself only under the strict rule, else
    ask the user (their words) or give the model its controls as ids."""
    g = _gates()
    blk = await asyncio.to_thread(_blocker, pid)
    g._taint(c, "screen")  # its words were read
    kind = getattr(blk, "kind", "") if blk is not None else "sheet"
    kind = "dialog" if kind in ("modal", "dialog") else "sheet"
    title = (getattr(blk, "title", "") or "").strip() if blk is not None else ""
    text = (getattr(blk, "text", "") or "").strip() if blk is not None else ""
    words = " ".join(x for x in (title, text) if x) or after.sheet
    buttons = [b for b in (getattr(blk, "buttons", None) or []) if _button_label(b)] if blk is not None else []
    pushable = [b for b in buttons if getattr(b, "role", "") == "AXButton" and getattr(b, "enabled", True)]

    why = ""
    btn, used = None, set()
    if c.origin != "user" or not ask.user:
        why = "not the user's own words"
    elif pick.stage not in ("exact", "lexical"):
        why = "the command was picked by meaning, not by its words"
    elif blk is None or not pushable:
        why = "its buttons can't be read"
    elif getattr(blk, "owner_pid", pid) != pid:
        why = "another app's"
    elif (front := getattr(after, "front_pid", 0)) and front != pid and not ax.is_neo(front):
        why = "another app came in front"
    else:
        hits = []
        for b in pushable:
            t = _toks(_button_label(b))
            if t and (sp := _span(req, t)):
                hits.append((len(t), b, set(range(*sp))))
        top = max((h[0] for h in hits), default=0)
        best = [h for h in hits if h[0] == top]
        if len({uistrings.norm(_button_label(h[1])) for h in best}) == 1:
            btn, used = best[0][1], best[0][2]
        if btn is None:
            why = "no single button named in the request" if not best else "several buttons named"
        elif not all(uistrings.readable(x) for x in (_button_label(btn), title, text) if x):
            why = "words NEO can't read in English"
        elif policy.commits(_button_label(btn), words):
            why = "that button commits"
        elif policy.sheet_consequence(words):
            why = "the sheet says something ends or is lost"
        elif _sounds_destructive(ask.user):
            why = "the request sounds destructive"
        elif _leftover(req, pick.used | used, extra):
            why = "the request says more"
        elif g._protected(name, pid):
            return g._protected_out()
    await _log(want="sheet", acted=btn is not None and not why, reason=why or "auto", n=len(buttons), origin=c.origin)
    if btn is not None and not why:
        label = _button_label(btn)
        before = await asyncio.to_thread(_scene, pid)
        out = await asyncio.to_thread(ax.press_el, btn)  # the button gated is the one pressed
        if out.startswith("Error"):
            return ToolOutput(f"{said}, but its {kind}'s “{label}” couldn't be pressed ({out.removeprefix('Error: ')}).", ok=False)
        now = await _watch(pid, before, _CLOSE_S, until=lambda b, s: getattr(s, "known", False) and s.sheet != b.sheet)
        closed = getattr(now, "known", False) and not now.sheet
        text_out = f"{said}, then answered its {kind} with “{label}”" + (
            f" — the {kind} closed." if closed else f", but the {kind} is still open — look before pressing again."
        )
        return _with_effect(ToolOutput(text_out, say=None if closed else True), changed=True, scene_changed=True, sheet="" if closed else now.sheet)
    labels = [_button_label(b) for b in (pushable or buttons)]
    if c.origin == "user":
        choices = " or ".join(f"“{x[:30]}”" for x in labels[:_OPTIONS]) or "its buttons"
        quoted = f" “{words[:120]}”" if words else ""
        return ToolOutput(confirm.ask_user(f"{name} opened a {kind}{quoted} — {choices}? Which one should I press?"), ok=True)
    lines = [f"says: {words[:400]}"] if words else []
    lines += [_control(b) for b in buttons[:12]]
    how = (
        "answer it by pressing one of its controls by id (ax_press) — a button that commits asks the user first"
        if buttons
        else "look at it (ax_tree) and answer it"
    )
    out = ToolOutput(f"{said}. It opened a {kind} in {name} that waits for an answer: {how}.\n" + fence("screen", "\n".join(lines) or after.sheet))
    out = _with_handoff(out, [f"{getattr(b, 'id', '')} “{_button_label(b)}”" for b in buttons])
    return _with_effect(out, changed=True, scene_changed=True, sheet=after.sheet)


# ---- the tool ----------------------------------------------------------------------------------------


def _out(x: ToolOutput | str) -> ToolOutput:
    """One result shape: a question NEO's own code asks (a Halt) stays one."""
    if isinstance(x, ToolOutput):
        if isinstance(x.text, Halt):
            x.halt, x.staged = True, x.staged or x.text.action_id
        return x
    halt = isinstance(x, Halt)
    return ToolOutput(x, ok=halt or not x.startswith(("Error", "NEEDS_")), halt=halt, staged=x.action_id if halt else "")


async def menu_command(a: dict, c: ToolContext) -> ToolOutput:
    """Choose a menu-bar command of the front (or named) app by its name."""
    out = await _run(a, c, claim=False)
    return _out(out) if out is not None else ToolOutput("Error: which menu command?", ok=False)


async def by_utterance(text: str, c: ToolContext) -> ToolOutput | None:
    """The user's own words, when they name one of the working app's menu commands and nothing
    more ("start a new game"): that command, and its sheet under the rule above. None - nothing
    done, nothing pressed - when the words aren't that (the model takes the request)."""
    if c.origin != "user" or not (text or "").strip() or _resolver_mode() == "off":
        return None  # (settings.resolver "off": nothing is pressed from words by itself)
    ctx = c if c.user_text else ToolContext(user_text=text, extras=c.extras, taint=c.taint, origin=c.origin)
    out = await _run({"utterance": text}, ctx, claim=True)
    return None if out is None else _out(out)


_LEAD = r"^\s*(?:and\s+|now\s+|please\s+|then\s+)*(?:can\s+you\s+|could\s+you\s+)?"
_TAIL = r"\s*[.!।]*\s*$"
_VERB = r"(?:choose|select|click(?:\s+on)?|pick|press|hit|use|run|open|go\s+to)"
_NAME = r"[^\s>›→»,;!?\"“”][^>›→»,;!?\"“”]{0,58}?"
_Q = r"[\"“']?"
_QE = r"[\"”']?"
# Only words that name a menu: a bare "click X" is click_text's, "click X then Y" is two steps.
FAST_PATHS: list[tuple[str, dict]] = [
    # "choose New Window from the File menu", "click Save As… in the File menu"
    (
        rf"{_LEAD}{_VERB}\s+(?:the\s+)?{_Q}(?P<item>{_NAME}){_QE}(?:\s+(?:command|item|option|menu\s+item|entry))?"
        rf"\s+(?:from|in|under|on)\s+(?:the\s+)?{_Q}(?P<menu>{_NAME}){_QE}\s+menu{_TAIL}",
        {"item": "<item>", "menu": "<menu>"},
    ),
    # "File > New Window", "choose File › Export › PDF"
    (rf"{_LEAD}(?:{_VERB}\s+)?(?:the\s+)?(?P<path>{_NAME}(?:\s*(?:->|>|›|→|»)\s*{_NAME}){{1,4}}){_TAIL}", {"path": "<path>"}),
    # "open the File menu and choose New Window", "click the Game menu, then New Game"
    (
        rf"{_LEAD}(?:open|click(?:\s+on)?|go\s+to)\s+(?:the\s+)?{_Q}(?P<menu>{_NAME}){_QE}\s+menu\s*,?\s+(?:and\s+)?(?:then\s+)?"
        rf"(?:{_VERB}\s+)?(?:on\s+)?(?:the\s+)?{_Q}(?P<item>{_NAME}){_QE}{_TAIL}",
        {"item": "<item>", "menu": "<menu>"},
    ),
    # Hinglish: "File menu se New Window chuno", "Game menu mein New Game pe click karo"
    (
        rf"{_LEAD}(?P<menu>{_NAME})\s+menu\s+(?:se|mein|me|main)\s+(?P<item>{_NAME})\s+(?:(?:pe|par)\s+)?"
        rf"(?:click\s+kar(?:o|\s+do)|select\s+kar(?:o|\s+do)|chuno|chun\s+lo|dabao|daba\s+do){_TAIL}",
        {"item": "<item>", "menu": "<menu>"},
    ),
    # Hindi: "फ़ाइल मेनू से नई विंडो चुनो", "गेम मेनू में नया गेम पर क्लिक करो"
    (
        rf"^\s*(?P<menu>{_NAME})\s+(?:मेनू|मेन्यू)\s+(?:से|में)\s+(?P<item>{_NAME})\s+"
        rf"(?:(?:पर\s+)?क्लिक\s+कर(?:ो|ें|िए|\s+दो)|चुन(?:ो|ें|िए|\s+लो)|दबा(?:ओ|एँ|एं|इए|\s+दो)){_TAIL}",
        {"item": "<item>", "menu": "<menu>"},
    ),
]

TOOL: dict[str, Any] = {
    "name": "menu_command",
    "description": (
        "Choose a command from the front (or named) app's menu bar by its name: item 'New Window' (menu "
        "'File'), or path 'File > Export > PDF'. Presses the app's own menu item (never the system menu, "
        "never a guessed shortcut); if it opens a sheet or dialog, the result lists its controls with ids "
        "to answer it with ax_press."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "item": {"type": "string", "description": "the command as the menu shows it"},
            "menu": {"type": "string", "description": "the menu it is in (optional)"},
            "path": {"type": "string", "description": "'Menu > Item' or 'Menu > Submenu > Item', instead of item/menu"},
            "app": {"type": "string", "description": "app name; omit for the app in front"},
        },
        "required": [],
    },
    "category": "computer",
    "quiet": True,
    "timeout": 15.0,
    "fast_path": FAST_PATHS,
}


# ---- typing into a named field ----------------------------------------------------------------------

_FIELD_ROLES = frozenset({"AXTextField", "AXSearchField", "AXComboBox", "AXTextArea", "AXSecureTextField"})
_FIELD_NOUN = r"(?:field|box|text\s*box|input(?:\s+box)?|bar|area)"
_NOT_PREP = r"(?:(?!\b(?:in|into|on|inside)\b)[^,;])"
_FIELD_WORDS = frozenset({"field", "box", "textbox", "text", "input", "bar", "area"})
_FIELD_RX = [
    # "type hello in the search field", "enter 42 into the amount box"
    re.compile(
        rf"^(?:type|enter|put|write|input|fill\s+in|add)\s+(?P<text>.+?)\s+(?:in|into|in\s+to|on|inside)\s+(?:the\s+)?"
        rf"(?P<field>{_NOT_PREP}{{1,40}}?)\s+{_FIELD_NOUN}$",
        re.I,
    ),
    # "fill the subject field with Lunch"
    re.compile(rf"^fill\s+(?:in\s+|out\s+)?(?:the\s+)?(?P<field>[^,;]{{1,40}}?)\s+{_FIELD_NOUN}\s+with\s+(?P<text>.+)$", re.I),
    # Hinglish: "search box mein cats likho"
    re.compile(
        r"^(?P<field>[^,;]{1,40}?)\s+(?:field|box|bar)\s+(?:mein|me|main)\s+(?P<text>.+?)\s+"
        r"(?:likho|likh\s+do|type\s+karo|type\s+kar\s+do|daalo|daal\s+do|dalo|bharo|bhar\s+do)$",
        re.I,
    ),
    # Hindi: "सर्च बॉक्स में बिल्ली लिखो"
    re.compile(
        r"^(?P<field>[^,;]{1,40}?)\s+(?:फ़ील्ड|फील्ड|बॉक्स|बार)\s+में\s+(?P<text>.+?)\s+"
        r"(?:लिखो|लिख\s+दो|टाइप\s+करो|टाइप\s+कर\s+दो|डालो|डाल\s+दो|भरो|भर\s+दो)$"
    ),
]
_QUOTED = re.compile(r"""^(["“'])(.*)(["”'])$""", re.S)


def parse_field(text: str) -> tuple[str, str] | None:
    """(what to type, the field's name) when the words name a field: "type hello in the search
    field", "fill the subject field with Lunch", "search box mein cats likho"."""
    s = " ".join((text or "").split()).strip()
    s = re.sub(r"^(?:(?:please|now|and|then)\s+)*(?:(?:can|could)\s+you\s+)?", "", s, flags=re.I).rstrip(" .!।")
    for rx in _FIELD_RX:
        m = rx.match(s)
        if m:
            what = m.group("text").strip()
            if q := _QUOTED.match(what):
                what = q.group(2)
            name = m.group("field").strip(" \"“”'")
            if what and name:
                return what, name
    return None


FIELD_FAST_PATHS: list[tuple[str, dict]] = [
    (
        rf"{_LEAD}(?:type|enter|put|write|input)\s+(?P<text>\"[^\"]+\"|“[^”]+”|'[^']+'|.+?)\s+(?:in|into|in\s+to|on|inside)\s+"
        rf"(?:the\s+)?(?P<field>{_NOT_PREP}{{1,40}}?)\s+{_FIELD_NOUN}{_TAIL}",
        {"text": "<text>", "field": "<field>"},
    ),
    (
        rf"{_LEAD}fill\s+(?:in\s+|out\s+)?(?:the\s+)?(?P<field>[^,;]{{1,40}}?)\s+{_FIELD_NOUN}\s+with\s+(?P<text>.+?){_TAIL}",
        {"text": "<text>", "field": "<field>"},
    ),
]


def _field_label(e: Any) -> str:
    return (getattr(e, "title", "") or "").strip()


def _fields(name: str, pid: int) -> list:
    fn = getattr(ax, "visible_elements", None)
    if fn is not None:
        try:
            els = list(fn(pid, interactive=True, limit=120))
        except TypeError:
            els = list(fn(pid))
    else:
        _, _, els = ax.window_elements(name or None)
    return [e for e in els if e.role in _FIELD_ROLES and e.enabled and e.w > 0 and e.h > 0]


def _ground_field(phrase: str, els: list) -> tuple[Any, list]:
    """(the field, or None; the candidates when several match)."""
    want = [w for w in _toks(phrase) if w not in _FIELD_WORDS] or _toks(phrase)
    if not want:
        return None, []

    def pick(hits: list) -> tuple[Any, list] | None:
        if not hits:
            return None
        return (hits[0], []) if len(hits) == 1 else (None, hits)

    for rule in (
        lambda e: _toks(_field_label(e)) == want,
        lambda e: bool(_toks(_field_label(e))) and bool(_span(want, _toks(_field_label(e)))),
        lambda e: bool(_span(_toks(_field_label(e)), want)),
    ):
        got = pick([e for e in els if rule(e)])
        if got is not None:
            return got
    if want == ["search"]:  # "the search field": the window's one search field, whatever its label says
        got = pick([e for e in els if e.role == "AXSearchField"])
        if got is not None:
            return got
    os_hits = [e for e in els if _field_label(e) and uistrings.os_match(phrase, _field_label(e))]
    if len(os_hits) == 1 and _resolver_mode() == "live":
        return os_hits[0], []
    return None, []


async def fill_field(text: str, field_name: str, c: ToolContext, *, app: str = "") -> ToolOutput:
    """Type ``text`` into the field called ``field_name`` in the front (or named) app's window: that
    field, found by its label, set through Accessibility and read back. Never a password field,
    never NEO's own or a protected app's."""
    return _out(await _fill(text, field_name, c, app))


async def _fill(text: str, field_name: str, c: ToolContext, app: str) -> str | ToolOutput:
    g = _gates()
    if err := g._needs_input():
        return err
    got = await _target({"app": app})
    if not isinstance(got, tuple):
        return got
    name, pid = got
    els = await asyncio.to_thread(_fields, name, pid)
    e, several = _ground_field(field_name, els)
    await _log(want="field", stage="lexical" if e is not None else "none", acted=e is not None, n=len(els),
               ambiguous=bool(several), script=_script(field_name), origin=c.origin)  # fmt: skip
    if e is None:
        if several:
            g._taint(c, "screen")
            opts = [f"{getattr(x, 'id', '')} {x.role.removeprefix('AX')} “{_field_label(x)[:40]}”" for x in several[:_OPTIONS]]
            out = ToolOutput(f"Error: several fields in {name} match “{field_name[:40]}” — which one?\n" + fence("screen", "\n".join(opts)), ok=False)
            return _with_handoff(out, opts)
        return ToolOutput(f"Error: no field called “{field_name[:40]}” in the front window of {name}.", ok=False)
    if e.role == "AXSecureTextField" or (e.ref is not None and await asyncio.to_thread(ax.secure, e.ref)):
        return ToolOutput("Error: that's a password field — NEO never fills those.", ok=False)
    out = await asyncio.to_thread(ax.set_el_value, e, text)
    if out.startswith("Error"):
        return ToolOutput(out, ok=False)
    if await asyncio.to_thread(ax.regular, pid):
        apps.set_target(name)
    label = _field_label(e) or e.role.removeprefix("AX")
    return _with_effect(ToolOutput(f"Typed “{text[:60]}” into “{label[:40]}” in {name}."), changed=True)
