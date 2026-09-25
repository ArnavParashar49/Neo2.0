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
