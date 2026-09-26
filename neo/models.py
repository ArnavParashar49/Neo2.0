"""Every downloaded model, pinned to the exact revision NEO was tested with.

Loading by repo name alone follows whatever the author pushes next: a new Laya encoder would
silently invalidate the head trained on this Mac (including the "destructive" and "needs the
screen" signals the confirm gate reads), and a new speech model could change behaviour
overnight. Revisions here only change when NEO is updated and re-tested.
"""

from __future__ import annotations

LAYA = "convaiinnovations/laya"
LAYA_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
# The files Laya's own loader fetches (the repo also holds sibling checkpoints NEO doesn't use).
LAYA_FILES = ["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]

EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

# The on-device voice (settings.stt_model / tts_model): pinned when they are the defaults.
REVISIONS = {
    "mlx-community/parakeet-tdt-0.6b-v3": "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15",
    "mlx-community/Kokoro-82M-bf16": "a71e4d38b236d968966a2002c4c895dbd12b1c3c",
}


def revision(repo: str) -> str | None:
    """The pinned revision for a repo, or None for one a user chose themselves."""
    return {LAYA: LAYA_REVISION, EMBEDDER: EMBEDDER_REVISION, **REVISIONS}.get(repo)


def laya_dir() -> str:
    """Laya's checkpoint at the pinned revision (downloaded once; afterwards from the cache)."""
    from huggingface_hub import snapshot_download

    return snapshot_download(LAYA, revision=LAYA_REVISION, allow_patterns=LAYA_FILES)


def local_path(repo: str) -> str:
    """A pinned repo as a local snapshot path (loaders accept paths); an unpinned one as given."""
    rev = revision(repo)
    if not rev:
        return repo
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, revision=rev)
