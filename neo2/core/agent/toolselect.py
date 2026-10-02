"""Per-request tool selection ("tool search", the cheap way).

Sending all ~46 tool schemas on every model call is the single biggest token cost in an agent
turn, and it makes small/free-tier brains (Groq's 8K TPM, a local 12B) unusable. So each run
gets a short list instead:

1. semantic match — the request is embedded (MiniLM, ~5 ms) against every tool's one-line
   description; the top hits come in,
2. category expansion — a multi-step task needs the whole family (browser_open alone is useless
   without browser_read/click), so the categories of the top hits are added,
3. rules — screen-related requests always get the observe/act tools; memory (and its search) is
   always present,
4. playbooks — tools a similar past task used are included,
5. `more_tools(query)` — a meta-tool the model can call mid-run to pull in anything else.

A run that works on the screen gets a lean list: the tools to look, act by id or label, type and
move (SCREEN_CORE), not the whole computer family; never NEO's own browser when the request names an
app installed on this Mac (the app is the target, not NEO's Chromium: evidence from the installed
apps' own names, never a list here); never shell, trash or dictate unless the request itself asks
for them. Fewer schemas on every call, and no wrong tool at the top (browser_scroll for "scroll
down in <app>").

Without an embedder (tests, or before the model has loaded) it falls back to keyword overlap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from neo2.core import runtime
from neo2.core.agent.registry import RegisteredTool, Registry
from neo2.core.agent.registry import registry as default_registry
from neo2.core.providers.base import ToolSpec

# Memory is always at hand: the one tool that saves, searches and forgets, and the search by itself
# (what a model reaches for when the user asks about something earlier or about themselves).
ALWAYS = ("memory", "memory_search")
SCREEN = {
    "ax_tree",
    "ax_find",
    "ax_press",
    "ax_set_value",
    "click",
    "type_text",
    "hotkey",
    "screenshot",
    "apps_running",
}
# A run that works on the screen gets these, the ones registered, in this order (the lean screen
# mode): look, act by id or label, type, press keys, scroll and drag. The first few come before the
# request's own best matches, the rest after them.
SCREEN_CORE = (
    "look", "ax_tree", "screenshot", "click", "ax_press", "type_text",
    "click_text", "ax_set_value", "hotkey", "scroll", "drag", "menu_command", "ax_find",
)  # fmt: skip
_SCREEN_FIRST = 6  # SCREEN_CORE[:6]: to look, click, press and type — whatever the request matched
# NEO's own tools a screen run never gets unless the request itself asks for them.
_SCREEN_UNLESS_ASKED = frozenset({"shell", "trash", "dictate"})
_SCREEN_CATEGORY = "computer"
_BROWSER_CATEGORY = "browser"  # NEO's own Chromium
_ASKED_TOP = 2  # "asks for it": among the request's best matches…
_ASKED_SIM = 0.35  # …and a close one
# Categories that only make sense as a set.
FAMILIES = {"browser", "computer"}
_TOP_K = 6
_MIN_SIM = 0.25
_MAX_TOOLS = 19  # (one more than before memory_search became always-there: it crowds nothing out)
_MAX_SCREEN_TOOLS = 17

MORE_TOOLS = ToolSpec(
    "more_tools",
    "You only see a subset of NEO's tools. Call this with a short description of what you need "
    "(e.g. 'send an email', 'read a PDF', 'control the browser') to get more tools enabled.",
    {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    essential=True,  # a request fitted into a small brain keeps it: the way back to what was cut
)

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> set[str]:
    return {t for t in _WORD.findall(s.lower()) if len(t) > 2}


@dataclass
class Selection:
    names: list[str]
    reason: dict[str, str] = field(default_factory=dict)

    def specs(self, reg: Registry) -> list[ToolSpec]:
        out = [t.spec() for n in self.names if (t := reg.get(n)) and not t.hidden]
        return out + [MORE_TOOLS]


class ToolSelector:
    def __init__(self, reg: Registry | None = None) -> None:
        self.reg = reg or default_registry()
        self._vecs: np.ndarray | None = None
        self._vec_names: list[str] = []

    # ---- embeddings (lazy, cached per tool set) --------------------------------------------
    def _tool_matrix(self) -> tuple[list[str], np.ndarray] | None:
        tools = [t for t in self.reg.all() if not t.hidden]
        names = [t.name for t in tools]
        if self._vecs is not None and names == self._vec_names:
            return names, self._vecs
        try:
            from neo2.core.memory.embed import embed_texts

            v = embed_texts([f"{t.name.replace('_', ' ')}: {t.description}" for t in tools])
        except Exception:  # noqa: BLE001
            v = None
        if v is None:
            return None
        self._vec_names, self._vecs = names, v
        return names, v

    def _scores(self, text: str) -> dict[str, float]:
        mat = self._tool_matrix()
        if mat is not None:
            from neo2.core.memory.embed import embed_texts

            q = embed_texts([text])
            if q is not None:
                names, vecs = mat
                sims = vecs @ q[0]
                return {n: float(s) for n, s in zip(names, sims, strict=True)}
        # keyword fallback: Jaccard-ish overlap between the request and name+description
        qt = _tokens(text)
        out = {}
        for t in self.reg.all():
            if t.hidden:
                continue
            tt = _tokens(t.name.replace("_", " ") + " " + t.description)
            out[t.name] = len(qt & tt) / (len(qt) + 1)
        return out

    # ---- public ------------------------------------------------------------------------------
    def select(
        self,
        text: str,
        *,
        needs_screen: bool = False,
        playbook_tools: list[str] | None = None,
        history_tools: list[str] | None = None,
        required: tuple[str, ...] = (),
    ) -> Selection:
        """`required`: tools the caller knows the run needs (a game's run opens the app)."""
        if runtime.library_bound():
            shown = [t.name for t in self.reg.all() if not t.hidden]
            if len(shown) <= _MAX_TOOLS:
                # An embedded agent's few tools, its host's among them: all of them, every time
                # (a keyword match would hide a host tool the request didn't happen to name).
                why = dict.fromkeys(shown, "all")
                for n in required:
                    if n in self.reg:
                        why.setdefault(n, "required")
                return Selection(list(why), why)
        scores = self._scores(text)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        top = [n for n, s in ranked[:_TOP_K] if s >= _MIN_SIM] or [n for n, _ in ranked[:3]]
        screen = needs_screen or any((t := self.reg.get(n)) is not None and t.category == _SCREEN_CATEGORY for n in top)
        asked = {n for n, s in ranked[:_ASKED_TOP] if s >= _ASKED_SIM}
        # A request that names an app on this Mac works in that app, never in NEO's own browser.
        in_an_app = screen and _names_an_app(text)
        pinned = set(required) | set(playbook_tools or [])

        def left_out(n: str) -> bool:
            if not screen or n in pinned:
                return False
            t = self.reg.get(n)
            if t is None:
                return True
            if n in _SCREEN_UNLESS_ASKED and n not in asked:
                return True
            return in_an_app and t.category == _BROWSER_CATEGORY

        chosen: dict[str, str] = {}

        def add(n: str, why: str) -> None:
            if n in self.reg and not left_out(n):
                chosen.setdefault(n, why)

        # Priority order matters: the list is capped, so guaranteed tools go in first and the
        # (large) families fill whatever room is left.
        for n in ALWAYS:
            add(n, "always")
        for n in required:
            add(n, "required")
        if screen:
            for n in SCREEN_CORE[:_SCREEN_FIRST]:
                add(n, "screen")
        for n in top:
            add(n, "match")
        if screen:
            for n in SCREEN_CORE[_SCREEN_FIRST:]:
                add(n, "screen")
        for n in playbook_tools or []:
            add(n, "playbook")
        for n in history_tools or []:
            add(n, "history")
        for n in list(top):
            t = self.reg.get(n)
            if not t or left_out(n):
                continue
            # Tools sharing a name prefix (mail_*, ax_*, calendar_*, safari_*) work as a set.
            if "_" in n:
                prefix = n.split("_", 1)[0] + "_"
                for sib in self.reg.all():
                    if sib.name.startswith(prefix) and not sib.hidden:
                        add(sib.name, f"prefix:{prefix}")
            # The screen's family is its core, above: a screen run doesn't get the rest of it.
            if t.category in FAMILIES and not (screen and t.category == _SCREEN_CATEGORY):
                for sib in self.reg.all(category=t.category):
                    if not sib.hidden:
                        add(sib.name, f"family:{t.category}")
        names = list(chosen)[: _MAX_SCREEN_TOOLS if screen else _MAX_TOOLS]
        return Selection(names, {n: chosen[n] for n in names})

    def search(self, query: str, exclude: set[str] | None = None, k: int = 6) -> list[RegisteredTool]:
        """For `more_tools`: the best matches not already enabled."""
        scores = self._scores(query)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        out = []
        for n, _ in ranked:
            if exclude and n in exclude:
                continue
            t = self.reg.get(n)
            if t and not t.hidden:
                out.append(t)
            if len(out) >= k:
                break
        return out


def _names_an_app(text: str) -> bool:
    """Does the request name an app installed on this Mac (the installed apps' own names, from the
    last completed scan — never starting one)? NEO.app only: an embedded agent has no inventory."""
    if runtime.library_bound():
        return False
    try:
        from neo2.core.decide import inventory
        from neo2.core.tools.computer import ax

        apps = inventory.snapshot()
        hit = inventory.named_span(text, apps) if apps else None
        # "…in NEO's browser": NEO itself is no app the request is about
        return hit is not None and not ax.is_neo(0, str(getattr(hit[0], "name", "")))
    except Exception:  # noqa: BLE001 - can't tell: nothing is left out
        return False


_selector: ToolSelector | None = None


def selector() -> ToolSelector:
    """The bound agent's selector, over its own tools (neo2.core.runtime), else NEO.app's."""
    rt = runtime.current_runtime()
    if rt is not None:
        return rt.get_or_create("selector", lambda: ToolSelector(rt.registry))
    runtime.check_unbound("selector()")
    global _selector
    if _selector is None:
        _selector = ToolSelector()
    return _selector
