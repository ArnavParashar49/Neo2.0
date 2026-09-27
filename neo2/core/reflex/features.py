"""The reflex's encoder, features and heads, shared by training (finetune.py), inference
(laya_reflex.py) and evaluation (evaluate.py).

Features = [ L2-normalised mean-pooled embedding of one Laya encoder | 1 ] (768-d multilingual,
1024-d English). A head file holds, per decision, an ensemble of small MLPs over the standardised
features and one temperature; a decision's probabilities are its members' mean softmax.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from neo.reflex.schema import INTENT_CRITERIA

HEADS = ("intent", "destructive", "needs_screen", "reply", "tool", "kind")
_CLASSES = {"intent": list(INTENT_CRITERIA), "destructive": ["no", "yes"], "needs_screen": ["no", "yes"], "reply": ["do it", "answer"]}

# The English checkpoint ships a choice:11+ temperature (0.10) that laya clamps to 0.5 and warns
# about on every load; NEO asks no question with that many options.
warnings.filterwarnings("ignore", message="laya: this checkpoint ships invalid temperatures", category=RuntimeWarning)


def load_encoder(encoder: str, device: str):
    """The pinned Laya checkpoint for `encoder` ("multilingual" | "english")."""
    import laya

    from neo import effects, models

    effects.guard("model", f"load Laya's {encoder} checkpoint into memory", via=laya.load)
    # fp16 autocast never beat fp32 on the M3 Pro at any row count (neither did compile or fast=True,
    # which stay off); laya would switch it on at 5 rows.
    os.environ.setdefault("LAYA_MPS_AMP_MIN_ROWS", "1000000")
    agent = laya.load(models.laya_dir(encoder), device=device)
    agent.neo_encoder = encoder  # which heads fit it: a retrain handed this agent trains for it
    return agent


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


def embed_only(agent, texts: list[str], batch: int = 64, progress: bool = False) -> np.ndarray:
    """Just the encoder embedding: all the heads need. Batched, it is 5-7x cheaper per text."""
    import sys

    out = []
    for i in range(0, len(texts), batch):
        out.append(embed(agent, texts[i : i + batch]))
        if progress:
            print(f"\r  embeddings {min(i + batch, len(texts))}/{len(texts)}", end="", file=sys.stderr)
    if progress:
        print(file=sys.stderr)
    return np.vstack(out)


# ---- heads --------------------------------------------------------------------------------------
def encoder_of(h: dict) -> tuple[str, str]:
    """(revision, subfolder) a head file was trained on; heads from before the subfolder was
    recorded were all trained on the English root."""
    from neo import models

    return h.get("encoder") or models.LAYA_REVISION, h.get("subfolder", "")


def parse_head(h: dict, encoder: str | None = None, dim: int | None = None) -> dict[str, Any] | None:
    """A head file's JSON as numpy ensembles, or None when it can't be used with this encoder: its
    outputs would be meaningless on other features (and the confirm gate reads "destructive")."""
    from neo import models

    if h.get("version") != 2:
        return None
    if encoder is not None and encoder_of(h) != (models.LAYA_REVISION, models.LAYA_SUBFOLDERS[encoder]):
        return None
    mu = _array(h["mu"])
    if dim is not None and len(mu) != dim + 1:
        return None
    out: dict[str, Any] = {
        "version": 2,
        "mu": mu,
        "sd": _array(h["sd"]),
        "zs_below": float(h.get("zs_below", 0.6)),
    }
    for name in HEADS:
        blk = h.get(name)  # an old file's "kind" was the model type ("mlp"), not a head
        out[name] = _block(blk, _CLASSES.get(name)) if isinstance(blk, dict) else None
    return out


def load_head(path: Path, encoder: str | None = None, dim: int | None = None) -> dict[str, Any] | None:
    try:
        return parse_head(json.loads(Path(path).read_text()), encoder, dim)
    except (OSError, ValueError, KeyError, TypeError):
        return None  # none yet, or unreadable: no head


def _block(blk: dict, classes: list[str] | None) -> dict[str, Any]:
    """{"members": [layers, ...]} (an ensemble), {"layers": ...} (one net) or {"W": ...} (linear)."""
    if "members" in blk:
        raw = blk["members"]
    elif "layers" in blk:
        raw = [blk["layers"]]
    else:
        raw = [[[blk["W"], np.zeros(len(blk["W"])).tolist()]]]
    members = [[(_array(W), _array(b)) for W, b in m] for m in raw]
    return {"members": members, "T": float(blk.get("T", 1.0)), "classes": list(blk.get("classes") or classes or [])}


def _array(v) -> np.ndarray:
    """A weight array: base64 float32 {"f32", "shape"} (finetune.pack) or a plain JSON list."""
    if isinstance(v, dict):
        import base64

        return np.frombuffer(base64.b64decode(v["f32"]), dtype="<f4").reshape(v["shape"]).astype(np.float32)
    return np.array(v, dtype=np.float32)


def forward(layers: list[tuple[np.ndarray, np.ndarray]], X: np.ndarray) -> np.ndarray:
    """Logits of one net for one standardised vector (d,) or a batch (n, d); ReLU between layers."""
    h = X
    for i, (W, b) in enumerate(layers):
        h = h @ W.T + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0.0)
    return h


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def head_probs(block: dict[str, Any], X: np.ndarray) -> np.ndarray:
    """The ensemble's calibrated probabilities: the mean of its members' softmax at temperature T."""
    return np.mean([softmax(forward(m, X) / block["T"]) for m in block["members"]], axis=0)
