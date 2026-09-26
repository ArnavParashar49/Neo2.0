"""Feature extraction shared by training (finetune.py) and inference (laya_reflex.py).

Feature vector = [ encoder mean-pool embedding (1024) | yes/no sub-question probabilities (k) |
                   4 zero-shot intent log-probs | 1 ]
The sub-question set is configurable: every extra question costs ~40 ms per decision on MPS,
so the trained head records which ones it was fit with and inference asks only those.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from neo.reflex.schema import INTENT_CRITERIA, QUESTIONS

_INTENTS = list(INTENT_CRITERIA)

SUBQ: dict[str, dict[str, Any]] = {
    "knowledge": {
        "type": "noul",
        "instructions": "Can `text` be answered from general knowledge or by writing text, without touching the computer?",
    },
    "smalltalk": {
        "type": "noul",
        "instructions": "Is `text` a greeting, thanks, small talk, or a question about the assistant itself?",
    },
    "compose": {
        "type": "noul",
        "instructions": "Does `text` ask to write, compose, draft or summarize something for the user to read?",
    },
    "app_switch": {
        "type": "noul",
        "instructions": "Does `text` ask to open, launch, close, quit or switch to an app or website?",
    },
    "device": {
        "type": "noul",
        "instructions": "Does `text` ask to change volume, brightness, mute, play/pause music, or set a timer or reminder, or ask the time or weather?",
    },
    "workspace": {
        "type": "noul",
        "instructions": "Does `text` involve email, messages, files, documents, folders, a browser page, code, or what is on the screen?",
    },
    "multistep": {
        "type": "noul",
        "instructions": "Would `text` take several actions or steps on the computer to complete?",
    },
    "end": {
        "type": "noul",
        "instructions": "Is `text` telling the assistant to stop, cancel, be quiet, or end the conversation?",
    },
}
ALL_SUBQ = list(SUBQ)
EMB_DIM = 1024


def questions_for(subq: list[str]) -> dict[str, Any]:
    return {**QUESTIONS, **{k: SUBQ[k] for k in subq}}


def dim_for(subq: list[str]) -> int:
    return EMB_DIM + len(subq) + len(_INTENTS) + 1


def answers_to_vector(ans: dict[str, Any], subq: list[str]) -> np.ndarray:
    p = ans["intent"].get("probabilities", {})
    zs = [np.log(max(float(p.get(k, 0.0)), 1e-6)) for k in _INTENTS]
    sub = [float(ans[k].get("noul", 0.0)) for k in subq]
    return np.array(sub + zs, dtype=np.float32)


def embed(agent, texts: list[str]) -> np.ndarray:
    import torch

    enc = agent.tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    enc = {k: v.to(agent.device) for k, v in enc.items()}
    with torch.no_grad():
        h = agent.model.encoder(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
        ).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
    return pooled.cpu().numpy().astype(np.float32)


def extract_all(agent, texts: list[str], batch: int = 32, progress: bool = False) -> dict[str, np.ndarray]:
    """One pass over the texts with EVERY question; returns per-block arrays so heads with any
    sub-question subset can be fit offline without re-running the model."""
    import sys

    embs, subs, zs, base = [], [], [], []
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        embs.append(embed(agent, chunk))
        for t in chunk:
            ans = agent.system_one({"text": t}, questions_for(ALL_SUBQ))["answers"]
            v = answers_to_vector(ans, ALL_SUBQ)
            subs.append(v[: len(ALL_SUBQ)])
            zs.append(v[len(ALL_SUBQ) :])
            # Laya's zero-shot yes/no answers: the teacher for the fast (embedding-only) heads.
            base.append([float(ans["destructive"].get("noul", 0.0)), float(ans["needs_screen"].get("noul", 0.0))])
        if progress:
            print(f"\r  features {min(i + batch, len(texts))}/{len(texts)}", end="", file=sys.stderr)
    if progress:
        print(file=sys.stderr)
    return {"emb": np.vstack(embs), "sub": np.vstack(subs), "zs": np.vstack(zs), "base": np.array(base, np.float32)}


def embed_only(agent, texts: list[str], batch: int = 64, progress: bool = False) -> np.ndarray:
    """Just the encoder embedding (~20 ms per text on MPS): all the fast heads need."""
    import sys

    out = []
    for i in range(0, len(texts), batch):
        out.append(embed(agent, texts[i : i + batch]))
        if progress:
            print(f"\r  embeddings {min(i + batch, len(texts))}/{len(texts)}", end="", file=sys.stderr)
    if progress:
        print(file=sys.stderr)
    return np.vstack(out)


def assemble(blocks: dict[str, np.ndarray], subq: list[str]) -> np.ndarray:
    idx = [ALL_SUBQ.index(k) for k in subq]
    n = blocks["emb"].shape[0]
    return np.hstack(
        [blocks["emb"], blocks["sub"][:, idx], blocks["zs"], np.ones((n, 1), np.float32)]
    ).astype(np.float32)


def features_one(agent, text: str, subq: list[str], ans: dict[str, Any] | None = None) -> np.ndarray:
    ans = ans or agent.system_one({"text": text}, questions_for(subq))["answers"]
    return np.concatenate([embed(agent, [text])[0], answers_to_vector(ans, subq), [1.0]]).astype(np.float32)
