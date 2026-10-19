"""Every downloaded model, pinned to the exact revision NEO was tested with.

Loading by repo name alone follows whatever the author pushes next: a new Laya encoder would
silently invalidate the head trained on this Mac (including the "destructive" and "needs the
screen" signals the confirm gate reads), and a new speech model could change behaviour
overnight. Revisions here only change when NEO is updated and re-tested.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

LAYA = "convaiinnovations/laya"
LAYA_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
# The files of one checkpoint, as Laya's own loader fetches them. The repo holds several: English at
# the root, the others in subfolders (settings.reflex_encoder -> subfolder).
LAYA_FILES = ["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]
LAYA_SUBFOLDERS = {"english": "", "multilingual": "multilingual"}
# What laya.load reads from a checkpoint's folder (a symlink into the hub's blobs counts only if
# its blob is there), besides a non-empty tokenizer/.
_LAYA_REQUIRED = ("rl_agent_config.json", "model.safetensors", "encoder/config.json")

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


def _laya_missing(path: str) -> list[str]:
    """What laya.load needs from a checkpoint's folder that isn't there."""
    missing = [f for f in _LAYA_REQUIRED if not os.path.isfile(os.path.join(path, f))]
    tok = os.path.join(path, "tokenizer")
    if not (os.path.isdir(tok) and any(os.path.isfile(os.path.join(tok, f)) for f in os.listdir(tok))):
        missing.append("tokenizer/*")
    return missing


def _whole(path: str) -> bool:
    """A cached snapshot folder fit to load without asking the Hub: it holds files, and none of its
    links points at a blob that's gone. (The hub checks a cached folder against the file list it
    saved with the download; this covers a folder it has no list for.)"""
    found = False
    for folder, _dirs, files in os.walk(path):
        for f in files:
            if not os.path.isfile(os.path.join(folder, f)):
                return False
            found = True
    return found


def _snapshot(repo: str, rev: str, ok: Callable[[str], bool], **kw: Any) -> str:
    """A pinned repo's snapshot folder, the cache first (local_files_only): the call as it always was
    only when the cache misses or `ok` (what the loader reads is there) says no.

    Asking the Hub first on every start printed "Fetching 5 files" and its unauthenticated-request
    warning each time and made an offline start wait on the network for files already here. Under
    HF_HUB_OFFLINE that second call is the hub's own offline one, as before: it never goes online.
    When the call fails (no network), a cached folder the hub calls incomplete — it lacks some of the
    repo's files, such as a README a loader never fetched — is still used if `ok`: today that start
    failed, and a loader missing a file it needs says so itself."""
    from huggingface_hub import snapshot_download

    partial = None
    try:
        root = snapshot_download(repo, revision=rev, local_files_only=True, **kw)
    except Exception as e:  # noqa: BLE001 — not in the cache, or not all of it: the call below
        partial = getattr(e, "snapshot_path", None)  # (the hub's IncompleteSnapshotError)
    else:
        if ok(root):
            return root
    try:
        return snapshot_download(repo, revision=rev, **kw)  # as it always was
    except Exception:
        if partial and ok(partial):
            return partial  # no network: what the loader reads is here
        raise


def laya_patterns(encoder: str = "english") -> list[str]:
    """The files of one Laya checkpoint in the pinned repo (what :func:`laya_dir` fetches)."""
    sub = LAYA_SUBFOLDERS[encoder]
    return [f"{sub}/{f}" if sub else f for f in LAYA_FILES]


def _cached(repo: str, rev: str | None, ok: Callable[[str], bool], **kw: Any) -> str | None:
    """`repo`'s snapshot folder when the cache alone already holds what its loader reads (`ok`) —
    the folder :func:`_snapshot` would load from with no network — else None. Never the network."""
    from huggingface_hub import snapshot_download

    try:
        root = snapshot_download(repo, revision=rev, local_files_only=True, **kw)
    except Exception as e:  # noqa: BLE001 — not in the cache, or not all of the repo (still fine if `ok`)
        root = getattr(e, "snapshot_path", None)
    return root if root and ok(root) else None


def laya_here(encoder: str = "english") -> bool:
    """One Laya checkpoint is on this Mac, whole: :func:`laya_dir` needs no network for it."""
    sub = LAYA_SUBFOLDERS[encoder]

    def whole(root: str) -> bool:
        return not _laya_missing(os.path.join(root, sub) if sub else root)

    return _cached(LAYA, LAYA_REVISION, whole, allow_patterns=laya_patterns(encoder)) is not None


def _loadable(root: str) -> bool:
    """A model folder its loader can open as it is: whole (:func:`_whole`), with its config and its
    weights. (The hub writes a file under a temporary name and renames it once complete, so a
    weights file that is there is all there.)"""
    if not _whole(root) or not os.path.isfile(os.path.join(root, "config.json")):
        return False
    try:
        names = os.listdir(root)
    except OSError:
        return False
    return any(n.endswith(".safetensors") and os.path.isfile(os.path.join(root, n)) for n in names)


def here(repo: str, **kw: Any) -> bool:
    """A model (a repo at its pinned revision, a repo the user chose, or a folder) is on this Mac fit
    to load with no network: its config and its weights. Never downloads — the question the voice
    asks before it would start a 2.6 GB download (neo2.core.voice.pipeline.local_ready)."""
    if os.path.isdir(repo):
        return _loadable(repo)
    return _cached(repo, revision(repo), _loadable, **kw) is not None


def laya_dir(encoder: str = "english") -> str:
    """One Laya checkpoint at the pinned revision (downloaded once; afterwards from the cache): only
    its own files, so the multilingual reflex never pulls the 1.7 GB English weights. The cache is
    asked first (_snapshot): a start with the checkpoint here needs no network.

    Raises FileNotFoundError when the checkpoint's files aren't all there: offline,
    snapshot_download hands back the cached revision's folder even when this checkpoint was never
    fetched into it, and laya.load would then fail somewhere less clear."""
    sub = LAYA_SUBFOLDERS[encoder]
    patterns = laya_patterns(encoder)

    def checkpoint(root: str) -> str:
        return os.path.join(root, sub) if sub else root

    def whole(root: str) -> bool:
        return not _laya_missing(checkpoint(root))

    path = checkpoint(_snapshot(LAYA, LAYA_REVISION, whole, allow_patterns=patterns))
    if missing := _laya_missing(path):
        raise FileNotFoundError(
            f"Laya's {encoder} checkpoint isn't on this Mac yet ({', '.join(missing)} missing under {path}): "
            "connect to the internet once so it can download"
        )
    return path


def local_path(repo: str) -> str:
    """A pinned repo as a local snapshot path (loaders accept paths); an unpinned one as given.
    The cache first (_snapshot): the Hub only when it misses, or its folder is empty or broken."""
    rev = revision(repo)
    if not rev:
        return repo
    return _snapshot(repo, rev, _whole)
