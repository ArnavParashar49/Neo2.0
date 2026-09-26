"""Per-request tool selection ("tool search", the cheap way).

Sending all ~46 tool schemas on every model call is the single biggest token cost in an agent
turn, and it makes small/free-tier brains (Groq's 8K TPM, a local 12B) unusable. So each run
gets a short list instead:

1. semantic match — the request is embedded (MiniLM, ~5 ms) against every tool's one-line
   description; the top hits come in,
2. category expansion — a multi-step task needs the whole family (browser_open alone is useless
   without browser_read/click), so the categories of the top hits are added,
3. rules — screen-related requests always get the observe/act tools; memory is always present,
4. playbooks — tools a similar past task used are included,
5. `more_tools(query)` — a meta-tool the model can call mid-run to pull in anything else.

Without an embedder (tests, or before the model has loaded) it falls back to keyword overlap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from neo.agent.registry import RegisteredTool, Registry
from neo.agent.registry import registry as default_registry
from neo.providers.base import ToolSpec

ALWAYS = {"memory"}
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
# Categories that only make sense as a set.
FAMILIES = {"browser", "computer"}
_TOP_K = 6
_MIN_SIM = 0.25
_MAX_TOOLS = 18

MORE_TOOLS = ToolSpec(
    "more_tools",
    "You only see a subset of NEO's tools. Call this with a short description of what you need "
    "(e.g. 'send an email', 'read a PDF', 'control the browser') to get more tools enabled.",
    {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
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
            from neo.memory.embed import embed_texts

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
            from neo.memory.embed import embed_texts

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
    ) -> Selection:
        scores = self._scores(text)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        chosen: dict[str, str] = {}
        # Priority order matters: the list is capped, so guaranteed tools go in first and the
        # (large) families fill whatever room is left.
        for n in ALWAYS:
            if n in self.reg:
                chosen[n] = "always"
        if needs_screen:
            for n in SCREEN:
                if n in self.reg:
                    chosen.setdefault(n, "screen")
        top = [n for n, s in ranked[:_TOP_K] if s >= _MIN_SIM] or [n for n, _ in ranked[:3]]
        for n in top:
            chosen.setdefault(n, "match")
        for n in playbook_tools or []:
            if n in self.reg:
                chosen.setdefault(n, "playbook")
        for n in history_tools or []:
            if n in self.reg:
                chosen.setdefault(n, "history")
        for n in list(top):
            t = self.reg.get(n)
            if not t:
                continue
            # Tools sharing a name prefix (mail_*, ax_*, calendar_*, safari_*) work as a set.
            if "_" in n:
                prefix = n.split("_", 1)[0] + "_"
                for sib in self.reg.all():
                    if sib.name.startswith(prefix) and not sib.hidden:
                        chosen.setdefault(sib.name, f"prefix:{prefix}")
            if t.category in FAMILIES:
                for sib in self.reg.all(category=t.category):
                    chosen.setdefault(sib.name, f"family:{t.category}")
        names = list(chosen)[:_MAX_TOOLS]
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


_selector: ToolSelector | None = None


def selector() -> ToolSelector:
    global _selector
    if _selector is None:
        _selector = ToolSelector()
    return _selector
