"""Local text embeddings for memory and playbook recall.

Uses all-MiniLM-L6-v2 (22M params, 384-d) via sentence-transformers on the CPU: ~5 ms per
query, well-calibrated cosine similarities (related ≈ 0.5-0.9, unrelated ≈ 0-0.3). Laya's
encoder was tried first and rejected — its mean-pooled vectors are anisotropic (everything scores
~0.85), which made semantic recall useless. Returns None if the model can't be loaded, and
callers fall back to keyword search.
"""

from __future__ import annotations

import threading

import numpy as np

from neo.config import settings

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384

_lock = threading.Lock()
_model = None
_failed = False


def _get():
    global _model, _failed
    with _lock:
        if _model is None and not _failed:
            try:
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer(MODEL, device="cpu", cache_folder=str(settings().models_dir))
            except Exception as e:  # noqa: BLE001 — embeddings are an optimisation, never a failure
                print(f"[memory] embedding model unavailable: {str(e)[:100]}")
                _failed = True
        return _model


def embed_texts(texts: list[str]) -> np.ndarray | None:
    if not texts:
        return np.zeros((0, DIM), dtype=np.float32)
    m = _get()
    if m is None:
        return None
    with _lock:
        v = m.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(v, dtype=np.float32)


def warmup() -> None:
    threading.Thread(target=_get, daemon=True).start()


def cosine_top(query: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Indices + similarities of the k most similar rows (rows are unit vectors)."""
    if matrix.shape[0] == 0:
        return []
    sims = matrix @ query
    idx = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in idx]
