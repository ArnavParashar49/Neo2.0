"""Local text embeddings for memory and playbook recall.

Uses all-MiniLM-L6-v2 (22M params, 384-d) via sentence-transformers on the CPU: ~5 ms per
query, well-calibrated cosine similarities (related ≈ 0.5-0.9, unrelated ≈ 0-0.3). Laya's
encoder was tried first and rejected — its mean-pooled vectors are anisotropic (everything scores
~0.85), which made semantic recall useless.

Nothing ever waits for the model:

* :func:`embed_texts` returns None until it is loaded (callers fall back to keyword search), and
  only asks for the load to start in the background (:func:`warmup`) — it never loads it itself.
  A remember or a recall during start-up, or in the middle of a voice turn, costs nothing.
* The load runs outside every lock; :func:`ready` and the ``ready_event`` say when it is done.
  The memory store fills in the vectors it couldn't make meanwhile (Store.backfill).
* Once the model's files are in the models folder it is loaded with ``local_files_only``: no
  request to the Hugging Face Hub at start, so NEO starts the same offline. Only the first run
  ever downloads.
* A load that failed is tried again after a while, not never.

:data:`MODEL_ID` names the model and its pinned revision: it is stored next to every vector, so
vectors made by another model are ignored (and made again) instead of compared.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from neo2.core import logger, models
from neo2.core.config import settings

log = logger(__name__)

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384
MODEL_ID = f"{MODEL}@{models.EMBEDDER_REVISION[:12]}"
_RETRY_S = 600.0  # a failed load is tried again after this long

_state = threading.Lock()  # guards the flags below — never held while the model loads or runs
_encode = threading.Lock()  # one encode at a time (the tokenizer isn't thread-safe)
_model = None
_failed = False  # the last load failed (see _failed_at): no model for now
_failed_at = 0.0
_loading = False
ready_event = threading.Event()


def _cached(cache: Path) -> bool:
    """Are the model's files already in the models folder (the hub's cache layout)?"""
    snap = cache / ("models--" + MODEL.replace("/", "--")) / "snapshots" / models.EMBEDDER_REVISION
    try:
        return snap.is_dir() and any(snap.iterdir())
    except OSError:
        return False


def _load(cache: str) -> None:
    """Load the model (the caller made sure only one load runs). Never raises."""
    global _model, _failed, _failed_at, _loading
    try:
        from sentence_transformers import SentenceTransformer

        from neo2.core import effects

        effects.guard("model", "load the memory's embedding model", via=SentenceTransformer)
        kw = {"device": "cpu", "cache_folder": cache, "revision": models.EMBEDDER_REVISION}
        m = None
        if _cached(Path(cache)):
            try:
                m = SentenceTransformer(MODEL, local_files_only=True, **kw)  # no network at start
            except Exception as e:  # noqa: BLE001 — a half-written cache: fetch what's missing
                log.warning(f"[memory] cached embedding model didn't load ({str(e)[:80]}); checking the Hub")
        if m is None:
            m = SentenceTransformer(MODEL, **kw)
        with _state:
            _model, _failed = m, False
        ready_event.set()
    except Exception as e:  # noqa: BLE001 — embeddings are an optimisation, never a failure
        log.warning(f"[memory] embedding model unavailable: {str(e)[:100]}")
        with _state:
            _failed, _failed_at = True, time.time()
    finally:
        with _state:
            _loading = False


def _claim() -> str | None:
    """Take the load if it is needed and nobody has it: the models folder to load from, else None."""
    global _loading
    if _model is not None or settings().embeddings == "off":
        return None
    with _state:
        if _model is not None or _loading or (_failed and time.time() - _failed_at < _RETRY_S):
            return None
        _loading = True
    return str(settings().models_dir)


def _get():
    """Load the model here and now, unless it is loaded, loading elsewhere or off; returns it (or
    None). For the one caller that loads models in order in its own background thread
    (neo2.core.reflex.warmup: Laya, then this — two threads importing `transformers` at once trip its
    lazy loader). Nothing on a request's path calls this."""
    cache = _claim()
    if cache is not None:
        _load(cache)
    return _model


def _load_quietly(cache: str) -> None:
    from neo2.core import effects

    try:
        _load(cache)
    except effects.EffectRefused:  # effects are off (a test): no model, and nobody's failure
        global _failed, _failed_at
        with _state:
            _failed, _failed_at = True, time.time()


def warmup() -> None:
    """Start loading in the background, if it isn't loaded or loading. Returns at once."""
    cache = _claim()
    if cache is not None:
        threading.Thread(target=_load_quietly, args=(cache,), daemon=True, name="neo-embedder").start()


def hold() -> str | None:
    """Take the load for a caller that will run it itself in a moment (:func:`load_held`), after
    another model that must not be imported at the same time: until then a lazy :func:`warmup`
    does nothing. Returns what to pass to :func:`load_held`, or None when there is nothing to load."""
    return _claim()


def load_held(cache: str | None) -> None:
    """Run the load a :func:`hold` took (None: nothing was held)."""
    if cache is not None:
        _load_quietly(cache)


def ready() -> bool:
    """Whether the model is loaded: a caller that must never wait for (or start) the load asks this."""
    return _model is not None


def embed_texts(texts: list[str]) -> np.ndarray | None:
    """Unit vectors for `texts`, or None while there is no model (off, not loaded yet, failed):
    the caller uses keywords. Never loads the model — it only asks for the load to start."""
    if not texts:
        return np.zeros((0, DIM), dtype=np.float32)
    if settings().embeddings == "off":  # keyword recall only: the model is never loaded
        return None
    m = _model
    if m is None:
        warmup()
        return None
    with _encode:
        v = m.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(v, dtype=np.float32)


def cosine_top(query: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Indices + similarities of the k most similar rows (rows are unit vectors)."""
    if matrix.shape[0] == 0:
        return []
    sims = matrix @ query
    if k < sims.shape[0]:
        idx = np.argpartition(-sims, k)[:k]
        idx = idx[np.argsort(-sims[idx])]
    else:
        idx = np.argsort(-sims)
    return [(int(i), float(sims[i])) for i in idx]

