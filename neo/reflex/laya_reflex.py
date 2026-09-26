"""Laya-backed reflex: local typed decisions on Apple Silicon.

One encoder pass (the multilingual checkpoint by default: it also reads Hindi and Hinglish, and
embeds in ~13 ms on an M3 Pro) feeds the heads trained by ``neo/reflex/finetune.py`` — intent,
destructive, needs_screen, reply, tool and task kind, each a small MLP ensemble. Rules back up the
two signals that must not miss: the words of an action that is hard to undo, and a request about
what is on the screen. Without a head for this encoder, Laya answers zero-shot.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from neo.config import settings
from neo.reflex import features
from neo.reflex.features import head_probs, load_encoder
from neo.reflex.schema import INTENT_CRITERIA, QUESTIONS, Decision, refers_to_screen, sounds_destructive

_INTENTS = list(INTENT_CRITERIA)
_ACTIONS = ("quick_action", "agent_task")


class LayaReflex:
    def __init__(
        self, device: str | None = None, *, encoder: str | None = None, head: Path | None = None, agent=None
    ) -> None:
        s = settings()
        self.encoder = encoder or s.reflex_encoder
        self._agent = agent or load_encoder(self.encoder, device or s.laya_device)  # pinned
        self.dim = int(self._agent.model.encoder.config.hidden_size)
        self._head_path = head or s.data_dir / "reflex_head.json"
        # After each decide() with a head: the utterance's L2-normalised embedding (float32) and the
        # head's mean embedding (same dim) — subtract it before comparing two embeddings, which
        # otherwise all sit at cosine ~0.83 of each other.
        self.last_embedding: np.ndarray | None = None
        self.center: np.ndarray | None = None
        self._head: dict[str, Any] | None = None
        self.reload_head()
        self.decide("warm up")  # first MPS call is ~2 s; take it now

    def reload_head(self) -> None:
        """Re-read the head file (the caller holds the Laya lock): a freshly trained head, the base
        head after an erase, or none — a head for another encoder counts as none until retrained."""
        path = Path(self._head_path)
        self._head = _load_head(path, self.encoder, self.dim)
        if self._head is None and path.exists():
            print("[reflex] the trained head doesn't fit this Laya encoder — ignoring it until retrained")
        self.center = None if self._head is None else self._head["mu"][:-1]

    @property
    def has_head(self) -> bool:
        return self._head is not None

    def decide(self, text: str) -> Decision:
        """The decision for `text`. With a head it also leaves ``last_embedding`` (the text's
        L2-normalised embedding) and ``center`` (the head's mean embedding) for the correction
        memory (neo/reflex/memory.py); otherwise both are None."""
        if self._head is None:
            return self._zero_shot(text)
        t0 = time.perf_counter()
        h = self._head
        x = features.embed(self._agent, [text])[0]
        self.last_embedding, self.center = x, h["mu"][:-1]
        xs = (np.append(x, 1.0).astype(np.float32) - h["mu"]) / h["sd"]
        probs = head_probs(h["intent"], xs)
        top = np.sort(probs)
        # Not sure: the head's pick still stands. Blending in zero-shot intent was measured worse
        # (0.80 vs 0.81 on real phrasings; zero-shot alone 0.37) and cost ~230 ms each time. What
        # actually runs for an unsure request becomes its training label (neo/reflex/learn.py).
        source = "laya+unsure" if float(probs.max()) < h["zs_below"] else "laya+head"
        intent = _INTENTS[int(probs.argmax())]
        d_p = max(_yes(h["destructive"], xs), float(sounds_destructive(text)))
        s_p = _yes(h["needs_screen"], xs)
        if self.encoder == "english" and intent == "agent_task" and 0.3 <= s_p < 0.5:
            # On the fence for an agent task (where it picks the vision brain): the English
            # checkpoint's zero-shot answer is worth ~90 ms. The multilingual one's is not.
            q = {"needs_screen": QUESTIONS["needs_screen"]}
            s_p = max(s_p, float(self._agent.system_one({"text": text}, q)["answers"]["needs_screen"].get("noul", 0.0)))
        s_p = max(s_p, float(refers_to_screen(text)))
        reply, reply_p = True, 1.0
        if h["reply"] is not None and intent not in ("chat", "stop"):
            reply_p = min(float(head_probs(h["reply"], xs)[1]), 0.9999)
            reply = reply_p >= 0.5
        tool, tool_p = "", 0.0
        if intent == "quick_action" and h["tool"] is not None:
            tp = head_probs(h["tool"], xs)
            j = int(tp.argmax())
            if h["tool"]["classes"][j] != "none":
                tool, tool_p = h["tool"]["classes"][j], float(tp[j])
        kind, kind_p = "", 0.0
        if intent in _ACTIONS and h["kind"] is not None:
            kp = head_probs(h["kind"], xs)
            j = int(kp.argmax())
            kind, kind_p = h["kind"]["classes"][j], float(kp[j])
        return Decision(
            intent=intent,  # type: ignore[arg-type]
            intent_confidence=float(probs.max()),
            destructive=d_p >= 0.5,
            destructive_p=d_p,
            needs_screen=s_p >= 0.5,
            needs_screen_p=s_p,
            source=source,
            latency_ms=(time.perf_counter() - t0) * 1000,
            tool=tool,
            tool_p=tool_p,
            reply=reply,
            reply_p=reply_p,
            kind=kind,
            kind_p=kind_p,
            margin=float(top[-1] - top[-2]),
        )

    def _zero_shot(self, text: str) -> Decision:
        """No usable head yet: Laya's own answers (~50-130 ms). The multilingual checkpoint isn't
        asked about the screen: it says yes to ~1 in 5 chat requests (measured), and a yes sends
        conversation to the vision agent."""
        t0 = time.perf_counter()
        self.last_embedding = self.center = None
        q = QUESTIONS if self.encoder == "english" else {k: QUESTIONS[k] for k in ("intent", "destructive")}
        ans = self._agent.system_one({"text": text}, q)["answers"]
        probs = _choice_probs(ans["intent"])
        intent = max(probs, key=probs.get)
        d_p = max(float(ans["destructive"].get("noul", 0.0)), float(sounds_destructive(text)))
        s_p = max(float(ans.get("needs_screen", {}).get("noul", 0.0)), float(refers_to_screen(text)))
        top = sorted(probs.values())
        return Decision(
            intent=intent,  # type: ignore[arg-type]
            intent_confidence=float(probs[intent]),
            destructive=d_p >= 0.5,
            destructive_p=d_p,
            needs_screen=s_p >= 0.5,
            needs_screen_p=s_p,
            source="laya",
            latency_ms=(time.perf_counter() - t0) * 1000,
            margin=top[-1] - top[-2],
        )


def _choice_probs(ans: dict[str, Any]) -> dict[str, float]:
    dist = ans.get("probabilities") or {}
    probs = {k: float(dist.get(k, 0.0)) for k in _INTENTS}
    if not any(probs.values()) and ans.get("choice") in probs:
        probs[ans["choice"]] = 1.0
    z = sum(probs.values()) or 1.0
    return {k: v / z for k, v in probs.items()}


def _yes(block: dict[str, Any] | None, xs: np.ndarray) -> float:
    """P(yes) from a yes/no head; 0 without one (training had a single class): the rule decides."""
    return float(head_probs(block, xs)[1]) if block is not None else 0.0


def _load_head(path: Path, encoder: str | None = None, dim: int | None = None) -> dict[str, Any] | None:
    """The head at `path` as numpy ensembles, if it fits this encoder (and embedding size)."""
    head = features.load_head(path, encoder, dim)
    return head if head is not None and head["intent"] is not None else None
