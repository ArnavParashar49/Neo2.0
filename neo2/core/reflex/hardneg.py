"""Hard negatives: the public and generated rows closest to a request the user corrected.

A correction says the head drew a boundary wrong right there, and one row barely moves it. Its
nearest neighbours in the rest of the data, whatever their labels, are added again at weight 2
(``w``: training must weight rows by it), so the retrain looks harder at that region. Only rows
the retrain trains on take part: one held out to judge it, trained on at weight 2, would flatter
the head it judges. Nearness is cosine on *centred* embeddings: Laya's mean-pooled vectors share
one large direction (random pairs score 0.83 raw, 0.02 once the mean is subtracted —
research_learning 8), so raw cosine ranks mostly by how generic a sentence is.

Not a proven gain yet: with real Laya embeddings and 100-400 simulated corrections (MASSIVE rows
the head got wrong; 3 seeds), adding them stayed within about a point of the corrections alone
(45% of a correction's neighbours carry the very label it was wrongly given). Use it only where
the retrain gate shows it helps.

    extra = hard_negatives(rows, emb, train_mask)  # finetune: train on rows + extra (embeddings by text)
"""

from __future__ import annotations

import numpy as np

from neo2.core.reflex.learn import EXTRA

K = 20
WEIGHT = 2.0
_CHUNK = 256  # corrected rows per similarity block: bounds memory at 256 × pool floats


def _unit(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def hard_negatives(
    rows: list[dict],
    emb: np.ndarray,
    trainable: np.ndarray,
    *,
    k: int = K,
    weight: float = WEIGHT,
    mean: np.ndarray | None = None,
) -> list[dict]:
    """Copies of the `k` nearest pool rows to each trainable corrected row (one with a non-empty
    ``wrong``), each returned once, at `weight`, with its own labels.

    ``trainable[i]`` says whether the retrain trains on row i. The pool is every trainable row
    that isn't the user's own and isn't one of the corrected sentences. ``emb[i]`` is row i's
    embedding; `mean` (default: theirs) is what gets subtracted — pass the head's ``mu[:-1]`` to
    centre exactly as the head does."""
    emb = np.asarray(emb, dtype=np.float32)
    trainable = np.asarray(trainable, dtype=bool)
    if not len(rows) == len(emb) == len(trainable):
        raise ValueError(f"{len(rows)} rows, {len(emb)} embeddings, {len(trainable)} trainable flags")
    said = {r["text"].strip().lower() for r in rows if r.get("wrong")}
    corrected = [i for i, r in enumerate(rows) if r.get("wrong") and trainable[i]]
    pool = np.array(
        [
            i
            for i, r in enumerate(rows)
            if trainable[i] and r.get("source") != EXTRA and r["text"].strip().lower() not in said
        ],
        dtype=int,
    )
    if not corrected or not len(pool) or k <= 0:
        return []
    mu = emb.mean(0) if mean is None else np.asarray(mean, dtype=np.float32)
    P = _unit(emb[pool] - mu)
    kk = min(k, len(pool))
    picked: dict[int, None] = {}  # ordered set: first by corrected row, then by pool order
    for s in range(0, len(corrected), _CHUNK):
        sims = _unit(emb[corrected[s : s + _CHUNK]] - mu) @ P.T
        for near in np.argpartition(-sims, kk - 1, axis=1)[:, :kk]:
            picked.update(dict.fromkeys(int(j) for j in pool[np.sort(near)]))
    return [{**rows[j], "w": weight, "hardneg": True} for j in picked]
