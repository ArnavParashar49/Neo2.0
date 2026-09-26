"""Act while the user is still talking.

The voice layer feeds the running transcript of the current utterance in here as it grows.

* Mid-sentence (`tick()`): a clause that is finished — followed by "and"/"then"/a comma, or
  sitting unchanged for a moment at the end — runs at once *if its tool opted in*
  (`early=True`: open_app, volume, brightness — cheap, idempotent, and their patterns refuse
  to match a sentence that continues). "open youtube and …" opens YouTube before the sentence
  is over. Keystrokes, clicks and anything with a free-text payload never run mid-sentence:
  "select all" might be the start of "select all the photos from June".
* At the end (`tick(final=True)`, local cascade): the whole rest of the utterance is planned
  with the fast paths (`neo.agent.fastpath.plan`) and run if it can be covered completely;
  otherwise nothing more runs and `remaining()` hands the utterance to the pipeline.

Everything that ran early leaves a one-shot *claim*: when the language model then asks for the
same thing (Gemini Live hears the whole sentence too), the claim answers it instead of running
it twice. A claim is consumed once, so a command the user repeats later still runs.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass

from neo.agent import fastpath
from neo.agent.registry import ToolContext, registry
from neo.events import bus

_STABLE_S = 0.35  # a clause must sit unchanged this long before it counts as "said"
_CLAIM_S = 15.0  # how long an early run can stand in for the model's own call for it
_FILLERS = {"please", "now", "thanks", "thank you", "ok", "okay"}
_SETTLE_AFTER_APP_S = 0.8  # "open notes and type hi": give the app a moment to come up first
_APP_TOOLS = ("open_app", "activate_app")

_KEY_ALIASES = {
    "enter": "return",
    "esc": "escape",
    "command": "cmd",
    "control": "ctrl",
    "option": "alt",
    "backspace": "delete",
}
_MOD_ORDER = ("cmd", "ctrl", "alt", "shift")


def _canon_keys(keys: str) -> str:
    parts = [_KEY_ALIASES.get(p, p) for p in str(keys).lower().replace(" ", "").split("+") if p]
    mods = sorted((p for p in parts if p in _MOD_ORDER), key=_MOD_ORDER.index)
    return "+".join(mods + [p for p in parts if p not in _MOD_ORDER])


def key(tool: str, args: dict) -> str:
    """Identity of an action, tolerant of how the model spells the same thing."""
    def canon(v) -> str:
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        return str(v).strip().lower()

    norm = {k: canon(v) for k, v in args.items() if v not in (None, "")}
    if tool == "hotkey" and "keys" in norm:
        norm["keys"] = _canon_keys(norm["keys"])
    if tool == "scroll":  # pointer-relative vs absolute coordinates: same intent
        norm = {k: v for k, v in norm.items() if k in ("dy", "dx")}
    return tool + json.dumps(norm, sort_keys=True)


@dataclass
class Ran:
    tool: str
    args: dict
    ok: bool
    text: str


class EarlyActor:
    def __init__(self) -> None:
        self._text = ""
        self._changed = 0.0
        self._consumed = 0  # chars of the current utterance already acted on
        self._utt = 0  # bumps on every new utterance
        self._ran_keys: set[str] = set()  # this utterance: never the same action twice
        self._claims: dict[str, tuple[float, Ran]] = {}  # early runs the model hasn't asked for yet
        self._lock = asyncio.Lock()
        self.executed: list[Ran] = []  # this utterance, in order

    # ---- utterance lifecycle ----------------------------------------------------------------
    def new_utterance(self) -> list[Ran]:
        """Start the next utterance; returns what ran for the one that just ended."""
        done = self.executed
        self._text, self._changed, self._consumed = "", 0.0, 0
        self._utt += 1
        self._ran_keys = set()
        self.executed = []
        return done

    def feed(self, text: str) -> None:
        """The full transcript of the current utterance so far (may be revised)."""
        text = text.strip()
        if text != self._text:
            if not text.startswith(self._text[: self._consumed]):
                self._consumed = 0  # the recogniser revised words we already acted on
            self._text, self._changed = text, time.time()

    def remaining(self) -> str:
        """What the caller should still hand to the normal pipeline.

        Empty when everything ran early. Otherwise the *whole* utterance: a leftover clause
        only makes sense with its context ("open youtube and search for cat videos" is a
        YouTube search), and the agent prompt lists what already ran so it isn't repeated."""
        left = self._text[self._consumed :].strip(" ,.;!?")
        if not left or left.lower() in _FILLERS:
            return ""
        return self._text if self._consumed else left

    # ---- acting ----------------------------------------------------------------------------
    async def tick(self, *, final: bool = False) -> list[Ran]:
        """Run whatever may run now. Returns what ran during this call."""
        if not self._text or (not final and time.time() - self._changed < _STABLE_S):
            return []
        async with self._lock:
            utt, text, start = self._utt, self._text, self._consumed
            pending = text[start:]
            ran: list[Ran] = []
            if final:
                steps = fastpath.plan(pending, first_index=len(self.executed))
                if not steps:
                    return []
                for tool, args in steps:
                    ran.append(await self._run(tool, args, utt))
                    if self._utt != utt:
                        return ran
                if all(r.ok for r in ran):
                    self._consumed = len(text)
                return ran
            pos = 0
            stable = time.time() - self._changed >= _STABLE_S
            for m in list(fastpath.SPLIT.finditer(pending)) + [None]:
                end = m.start() if m else len(pending)
                clause = pending[pos:end].strip(" ,.;!?")
                nxt = m.end() if m else len(pending)
                if not clause:
                    pos = nxt
                    continue
                if m is None and not stable:
                    break  # the last clause is still being spoken
                hit = fastpath.match(clause)
                t = registry().get(hit[0]) if hit else None
                if t is None or not t.early or t.payload:
                    break  # not a command we may run mid-sentence: the rest waits
                if (len(self.executed) + len(ran)) > 0 and not t.chain:
                    break
                r = await self._run(hit[0], hit[1], utt)
                if self._utt != utt or not r.ok:
                    break
                ran.append(r)
                pos = nxt
            if self._utt == utt and pos:
                self._consumed = start + pos
            return ran

    async def _run(self, tool: str, args: dict, utt: int) -> Ran:
        k = key(tool, args)
        if k in self._ran_keys:
            return next((r for r in self.executed if key(r.tool, r.args) == k), Ran(tool, args, True, ""))
        self._ran_keys.add(k)
        if self.executed and self.executed[-1].tool in _APP_TOOLS and tool not in _APP_TOOLS:
            await asyncio.sleep(_SETTLE_AFTER_APP_S)
        await bus().tool_start(tool, args, job="early")
        out = await registry().invoke(tool, args, ToolContext(user_text=self._text))
        await bus().tool_end(tool, out.ok, out.text, job="early")
        r = Ran(tool, args, out.ok, out.text)
        if self._utt == utt:
            self.executed.append(r)
        if out.ok:
            self._claims[k] = (time.time(), r)
        return r

    # ---- claims: the model's own call for what already ran --------------------------------
    def claim(self, tool: str, args: dict) -> str | None:
        """If this ran early and nobody has claimed it yet, claim it and return its result."""
        self._expire()
        rec = self._claims.pop(key(tool, args), None)
        return rec[1].text if rec else None

    def unclaimed(self) -> list[Ran]:
        """Everything that ran early and nobody has accounted for — consumed by the caller
        (the agent prompt says these are done, so the model doesn't redo them)."""
        self._expire()
        out = [r for _, r in sorted(self._claims.values(), key=lambda x: x[0])]
        self._claims.clear()
        return out

    def _expire(self) -> None:
        now = time.time()
        for k in [k for k, (t, _) in self._claims.items() if now - t > _CLAIM_S]:
            del self._claims[k]


_actor: EarlyActor | None = None


def early() -> EarlyActor:
    global _actor
    if _actor is None:
        _actor = EarlyActor()
    return _actor
