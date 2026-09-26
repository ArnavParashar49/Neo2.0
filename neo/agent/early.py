"""Act while the user is still talking.

The voice layer feeds the running transcript of the current utterance in here as it grows.
Whenever the not-yet-handled part has been stable for a moment (the speaker paused, or moved
on to the next clause) and a fast-path regex fully matches it, the matching quick tool runs
immediately — "open youtube and …" opens YouTube before the sentence is over. This is the
Jev-demo behaviour: a fast decision layer over the live transcript, acting on confidence.

Safety comes from *what* may run early: only tools that declare fast-path patterns (open app,
volume, brightness, timer, clock) — cheap, idempotent, nothing destructive. Everything that
ran is remembered for a short while so the language model's own later call for the same thing
is answered "already done" instead of running twice.
"""

from __future__ import annotations

import asyncio
import json
import re
import time

from neo.agent.registry import ToolContext, registry
from neo.events import bus

_STABLE_S = 0.35  # a clause must sit unchanged this long before it counts as "said"
_DEDUPE_S = 30.0  # the model's own call for the same tool+args within this window is a no-op
_FILLERS = {"please", "now", "thanks", "thank you", "ok", "okay"}
_SETTLE_AFTER_APP_S = 0.8  # "open notes and type hi": give the app a moment to come up first
_CLAUSE_SPLIT = re.compile(r"\s*(?:,|;|\band then\b|\bthen\b|\band\b|\bafter that\b)\s*", re.I)


def _key(tool: str, args: dict) -> str:
    return tool + json.dumps({k: str(v).strip().lower() for k, v in args.items()}, sort_keys=True)


class EarlyActor:
    def __init__(self) -> None:
        self._text = ""
        self._changed = 0.0
        self._consumed = 0  # chars of the current utterance already acted on
        self._done: dict[str, tuple[float, str]] = {}  # key → (when, result)
        self._running: set[str] = set()
        self._log: list[tuple[str, dict, float]] = []  # everything that ran early, with when
        self._lock = asyncio.Lock()
        self.executed: list[tuple[str, dict]] = []  # this utterance, in order

    # ---- utterance lifecycle ----------------------------------------------------------------
    def new_utterance(self) -> None:
        self._text, self._changed, self._consumed = "", 0.0, 0
        self.executed = []

    def feed(self, text: str) -> None:
        """The full transcript of the current utterance so far (may be revised)."""
        text = text.strip()
        if text != self._text:
            self._text, self._changed = text, time.time()
            if len(text) < self._consumed:  # the recogniser revised earlier words — start over
                self._consumed = 0

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
    async def tick(self, *, final: bool = False) -> list[str]:
        """Run any fast-path clause that has stabilised. Returns results of what ran."""
        if not self._text or (not final and time.time() - self._changed < _STABLE_S):
            return []
        async with self._lock:
            from neo.agent.session import _match_fast_path

            pending = self._text[self._consumed :]
            results: list[str] = []
            pos = 0
            for m in list(_CLAUSE_SPLIT.finditer(pending)) + [None]:
                end = m.start() if m else len(pending)
                clause = pending[pos:end].strip(" ,.;!?")
                nxt = m.end() if m else len(pending)
                if not clause:
                    pos = nxt
                    continue
                if not final and m is None and time.time() - self._changed < _STABLE_S:
                    break  # the last clause is still being spoken
                hit = _match_fast_path(clause)
                if hit:
                    tool, args = hit
                    t = registry().get(tool)
                    if not final and t is not None and not t.early:
                        break  # "type …": the words are the payload, so wait for the whole sentence
                    if (self._consumed + pos) > 0 and t is not None and not t.chain:
                        break  # "… and search for X": that search belongs to the app just opened
                    if self.executed and self.executed[-1][0] in ("open_app", "activate_app") and tool not in ("open_app", "activate_app"):
                        await asyncio.sleep(_SETTLE_AFTER_APP_S)
                    results.append(await self._run(tool, args))
                    self._consumed += nxt
                    pos = nxt
                else:
                    break  # a non-command clause: leave the rest for the full pipeline
            return results

    async def _run(self, tool: str, args: dict) -> str:
        key = _key(tool, args)
        if key in self._running:
            return ""
        recent = self.recently_done(tool, args)
        if recent is not None:
            return recent
        self._running.add(key)
        try:
            await bus().tool_start(tool, args, job="early")
            out = await registry().invoke(tool, args, ToolContext(user_text=self._text))
            await bus().tool_end(tool, out.ok, out.text, job="early")
            if out.ok:
                self._done[key] = (time.time(), out.text)
                self.executed.append((tool, args))
                self._log.append((tool, args, time.time()))
                del self._log[:-50]
            return out.text
        finally:
            self._running.discard(key)

    def recent(self, within_s: float) -> list[tuple[str, dict]]:
        """Tools that ran early in the last `within_s` seconds (for the agent's context)."""
        now = time.time()
        return [(t, a) for t, a, when in self._log if now - when < within_s]

    def recently_done(self, tool: str, args: dict) -> str | None:
        rec = self._done.get(_key(tool, args))
        if rec and time.time() - rec[0] < _DEDUPE_S:
            return rec[1]
        return None


_actor: EarlyActor | None = None


def early() -> EarlyActor:
    global _actor
    if _actor is None:
        _actor = EarlyActor()
    return _actor
