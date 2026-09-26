"""Laya-backed reflex: local typed decisions on Apple Silicon.

Laya's pip package ships no trainer, so the "fine-tune" is a linear head over Laya's own
features — encoder embedding + a few yes/no sub-questions + zero-shot intent probabilities
(``neo/reflex/finetune.py``). The head file records which sub-questions it needs, so inference
only pays for those. Without a head we fall back to zero-shot intent.
destructive / needs_screen are always zero-shot (Laya is well-calibrated on yes/no).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from neo.config import settings
from neo.reflex.features import features_one, questions_for
from neo.reflex.schema import INTENT_CRITERIA, QUESTIONS, Decision

_INTENTS = list(INTENT_CRITERIA)


class LayaReflex:
    def __init__(self, device: str | None = None) -> None:
        import laya  # heavy import (torch) — keep it lazy

        self._agent = laya.load("convaiinnovations/laya", device=device or settings().laya_device)
        self._head = _load_head(settings().data_dir / "reflex_head.json")
        self._questions = questions_for(self._head["subq"]) if self._head else QUESTIONS
        self.decide("warm up")  # first MPS call is ~2 s; take it now

    @property
    def has_head(self) -> bool:
        return self._head is not None

    def decide(self, text: str) -> Decision:
        if self._head and self._head.get("version") == 2:
            return self._decide_v2(text)
        t0 = time.perf_counter()
        ans = self._agent.system_one({"text": text}, self._questions)["answers"]
        if self._head:
            h = self._head
            x = features_one(self._agent, text, h["subq"], ans)
            logits = h["W"] @ ((x - h["mu"]) / h["sd"])
            e = np.exp(logits - logits.max())
            probs = dict(zip(_INTENTS, (e / e.sum()).tolist(), strict=True))
        else:
            probs = _choice_probs(ans["intent"])
        intent = max(probs, key=probs.get)
        d_p = float(ans["destructive"].get("noul", 0.0))
        s_p = float(ans["needs_screen"].get("noul", 0.0))
        return Decision(
            intent=intent,  # type: ignore[arg-type]
            intent_confidence=float(probs[intent]),
            destructive=d_p >= 0.5,
            destructive_p=d_p,
            needs_screen=s_p >= 0.5,
            needs_screen_p=s_p,
            source="laya+head" if self._head else "laya",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


    # ---- v2: everything from one encoder pass; zero-shot only when the head is unsure ----------
    def _decide_v2(self, text: str) -> Decision:
        t0 = time.perf_counter()
        h = self._head
        from neo.reflex.features import embed

        x = np.append(embed(self._agent, [text])[0], 1.0).astype(np.float32)
        xs = (x - h["mu"]) / h["sd"]
        probs = _softmax(_forward(h["intent"], xs) / h["T"]["intent"])
        source = "laya+head"
        ans = None
        if float(probs.max()) < h["zs_below"]:  # not sure: ask Laya properly and blend
            ans = self._agent.system_one({"text": text}, QUESTIONS)["answers"]
            zs = np.array([_choice_probs(ans["intent"])[k] for k in _INTENTS], np.float32)
            probs = 0.5 * probs + 0.5 * zs
            source = "laya+zs"
        intent = _INTENTS[int(probs.argmax())]
        d_p = float(_softmax(_forward(h["destructive"], xs) / h["T"]["destructive"])[1])
        s_p = float(_softmax(_forward(h["needs_screen"], xs) / h["T"]["needs_screen"])[1])
        if intent == "agent_task" and ans is None and s_p < 0.5:
            # The distilled head under-detects "look at my screen"; for agent tasks (where it
            # picks the vision brain) the zero-shot question is worth its ~45 ms.
            q = {"needs_screen": QUESTIONS["needs_screen"]}
            s_p = max(s_p, float(self._agent.system_one({"text": text}, q)["answers"]["needs_screen"].get("noul", 0.0)))
        elif ans is not None:
            s_p = max(s_p, float(ans.get("needs_screen", {}).get("noul", 0.0)))
        reply, reply_p = True, 1.0
        if h.get("reply") is not None and intent not in ("chat", "stop"):
            reply_p = float(_softmax(_forward(h["reply"], xs) / h["T"]["reply"])[1])
            reply = reply_p >= 0.5
        tool, tool_p = "", 0.0
        if intent == "quick_action" and h.get("tool"):
            tp = _softmax(_forward(h["tool"], xs) / h["T"]["tool"])
            j = int(tp.argmax())
            if h["tool_classes"][j] != "none":
                tool, tool_p = h["tool_classes"][j], float(tp[j])
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
        )


def _forward(layers: list[tuple[np.ndarray, np.ndarray]], x: np.ndarray) -> np.ndarray:
    for i, (W, b) in enumerate(layers):
        x = W @ x + b
        if i < len(layers) - 1:
            x = np.maximum(x, 0.0)
    return x


def _layers(block: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """A head saved as {"layers": [[W, b], ...]} (MLP/linear) or the older {"W": ...} (linear)."""
    if "layers" in block:
        return [(np.array(W, dtype=np.float32), np.array(b, dtype=np.float32)) for W, b in block["layers"]]
    W = np.array(block["W"], dtype=np.float32)
    return [(W, np.zeros(W.shape[0], dtype=np.float32))]


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


def _choice_probs(ans: dict[str, Any]) -> dict[str, float]:
    dist = ans.get("probabilities") or {}
    probs = {k: float(dist.get(k, 0.0)) for k in _INTENTS}
    if not any(probs.values()) and ans.get("choice") in probs:
        probs[ans["choice"]] = 1.0
    z = sum(probs.values()) or 1.0
    return {k: v / z for k, v in probs.items()}


def _load_head(path: Path) -> dict[str, Any] | None:
    try:
        if path.exists():
            h = json.loads(path.read_text())
            if h.get("version") == 2:
                return {
                    "version": 2,
                    "mu": np.array(h["mu"], dtype=np.float32),
                    "sd": np.array(h["sd"], dtype=np.float32),
                    "intent": _layers(h["intent"]),
                    "destructive": _layers(h["destructive"]),
                    "needs_screen": _layers(h["needs_screen"]),
                    "tool": _layers(h["tool"]) if h.get("tool") else None,
                    "reply": _layers(h["reply"]) if h.get("reply") else None,
                    "T": {k: float(h.get(k, {}).get("T", 1.0)) for k in ("intent", "destructive", "needs_screen", "tool", "reply")},
                    "tool_classes": list(h["tool"]["classes"]) if h.get("tool") else [],
                    "zs_below": float(h.get("zs_below", 0.6)),
                    "subq": [],
                }
            if "mu" in h and "subq" in h:
                return {
                    "W": np.array(h["W"], dtype=np.float32),
                    "mu": np.array(h["mu"], dtype=np.float32),
                    "sd": np.array(h["sd"], dtype=np.float32),
                    "subq": list(h["subq"]),
                }
    except Exception:
        pass
    return None
