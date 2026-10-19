"""What NEO downloads, only when asked and with progress: ``python -m neo2 fetch``.

    python -m neo2 fetch                  # --essential: what every start needs (about 1.7 GB)
    python -m neo2 fetch --local-voice    # the on-device voice, Parakeet and Kokoro (about 2.9 GB)
    python -m neo2 fetch --browser        # NEO's browser, Playwright's Chromium (about 360 MB on disk)
    python -m neo2 fetch --vision         # boxes on the camera, D-FINE small (about 40 MB)
    python -m neo2 fetch --status         # what's on this Mac already: nothing is downloaded
    python -m neo2 fetch … --json         # JSON lines on stdout, for NEO's Setup window

The groups combine (``--essential --local-voice``). Every item is what the core itself loads, from
where it loads it, by the loaders it uses: Laya and the on-device voice at the revisions pinned in
neo2.core.models, the memory's embedder (MiniLM, pinned) into NEO's models dir as the memory loads
it, the wake word's zip checked against its pinned SHA-256 (neo2.core.voice.wake), Playwright's
Chromium at the revision the installed playwright package was built for (``playwright install
chromium --no-shell``: NEO never launches the headless shell when the full Chromium is there), and
D-FINE through transformers into NEO's models dir — which also sets vision_download_ok, the user's
yes to it. What is here already is never asked for again. Hugging Face's per-file locks make it safe
to run beside a core that is fetching the same files.

``--json`` writes one object per line to stdout, and nothing else goes there (whatever a library
prints lands on stderr):

    {"event": "plan", "items": [{id, group, label, source, size, disk, here, skip}], "size": N}
        what will happen: each item, about how many bytes it downloads (`size`) and keeps (`disk`),
        whether it is here already, or why it is skipped (a setting turned it off); `size` in all
    {"event": "start", "id"}                      an item's download begins
    {"event": "progress", "id", "done", "total", "all_done", "all_total", "rate", "eta_s"}
        bytes so far (at most four lines a second); `total` is exact once the server has said it
    {"event": "stage", "id", "stage"}             a step with no bytes to count ("unpacking")
    {"event": "item", "id", "state": "here" | "done" | "skipped" | "failed", "code"?, "message"?}
    {"event": "done", "ok", "failed": [ids], "seconds", "cancelled"?}

A failure's `code` is one of offline (no network, or HF_HUB_OFFLINE), disk_full, checksum, not_found
(the pinned file is gone from its server) or failed; `message` says it in words. Exit status: 0 when
everything asked for is here, 1 when something failed, 2 for a bad command line, 130 when stopped
(SIGTERM / SIGINT: Setup's Cancel).

The core runs the same command for a download the user starts from a window (:class:`Downloads`):
Brains & Keys' and the chat's "Download on-device voice", the browser's first page. Nothing here
downloads on its own: the voice never fetches its models unasked (neo2.core.voice.pipeline).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import errno
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

GROUPS = ("essential", "local_voice", "browser", "vision")
_FLAG = {g: "--" + g.replace("_", "-") for g in GROUPS}

# What each item downloads, in bytes, at the pinned revisions (the Hub's own file list for each):
# what Setup shows before anything has asked a server. The download says the exact size as it starts.
LAYA_BYTES = {"multilingual": 678_201_636, "english": 842_610_120}
DECIDE_BYTES = 846_195_716  # Laya's typed-decisions checkpoint (neo2.core.reflex.zeroshot)
EMBEDDER_BYTES = 91_578_415
WAKE_BYTES = 41_000_000  # the zip (about 68 MB unpacked)
STT_BYTES = 2_509_044_141  # Parakeet TDT 0.6B v3
TTS_BYTES = 389_410_807  # Kokoro 82M bf16
VOICE_PACK_BYTES = 522_339
CHROMIUM_BYTES = 160_000_000  # Chrome for Testing's zip, then ffmpeg's (Playwright says the exact size)
CHROMIUM_DISK = 362_000_000
DFINE_BYTES = 42_000_000
LOCAL_VOICE_BYTES = STT_BYTES + TTS_BYTES + VOICE_PACK_BYTES

# What sentence-transformers reads of all-MiniLM-L6-v2 (neo2.core.memory.embed loads it with
# local_files_only once these are in NEO's models dir).
EMBEDDER_FILES = [
    "modules.json", "config_sentence_transformers.json", "README.md", "sentence_bert_config.json", "config.json",
    "model.safetensors", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json", "vocab.txt",
    "1_Pooling/config.json",
]  # fmt: skip
DFINE_FILES = ["config.json", "preprocessor_config.json", "model.safetensors"]
KOKORO_VOICES = "prince-canuma/Kokoro-82M"  # where mlx-audio's Kokoro takes its voice packs from

_ROOM = 256 << 20  # headroom left on the disk after an item, besides what it keeps
_STALE_PARTIAL_S = 1800.0  # a download's temporary file untouched this long was left by one that died
_EMIT_EVERY_S = 0.25
_RATE_WINDOW_S = 8.0


# ---- the items ----------------------------------------------------------------------------------------


@dataclass
class Item:
    id: str
    group: str
    label: str
    source: str  # where it comes from, in words
    size: int  # about how many bytes it downloads
    here: Callable[[], bool]  # on this Mac already (never touches the network)
    get: Callable[[Job], None]  # downloads it (in a worker thread); raises when it can't
    disk: int = 0  # bytes it keeps (0: its size)
    where: Path | None = None  # the folder it lands in (the disk check)
    skip: str = ""  # why it isn't fetched ("" = it is)

    def plan(self, here: bool) -> dict[str, Any]:
        return {"id": self.id, "group": self.group, "label": self.label, "source": self.source, "size": self.size,
                "disk": self.disk or self.size, "here": here, "skip": self.skip or None}  # fmt: skip


class Job:
    """One item's download as the main thread watches it: the bytes so far (reported by the worker,
    or measured on disk), the size, a step with no bytes to count."""

    def __init__(self, item: Item) -> None:
        self.item = item
        self.total = item.size
        self.exact = False
        self.stage = ""
        self.error: BaseException | None = None
        self.ended = threading.Event()
        self.proc: subprocess.Popen | None = None  # a program the item runs (Playwright's installer)
        self._done = 0
        self._measure: Callable[[], int] | None = None

    def sized(self, n: int | None) -> None:
        """The exact size, once a server has said it."""
        if n and n > 0:
            self.total, self.exact = int(n), True

    def at(self, n: int) -> None:
        self._done = max(self._done, int(n))

    def measure(self, fn: Callable[[], int]) -> None:
        """Bytes so far, read from the disk each time the main thread looks."""
        self._measure = fn

    def done(self) -> int:
        n = self._done
        if self._measure is not None:
            try:
                n = max(n, int(self._measure()))
            except Exception:  # noqa: BLE001 — a folder that moved under us: what was said stands
                pass
        if not self.exact and n > self.total:
            self.total = n  # the estimate was low (no server said the size): it grows with the download
        return max(0, min(n, self.total)) if self.total else max(0, n)


def _hf_cache() -> Path:
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


def _repo_blobs(repo: str, cache: Path) -> Path:
    return cache / ("models--" + repo.replace("/", "--")) / "blobs"


def _blob_bytes(blobs: Path) -> tuple[int, int]:
    """(bytes of the finished files, bytes of those still downloading) in a repo's cache folder. A
    finished blob may be a link into the hub's shared store: its target is counted. A file being
    written is counted by the blocks it holds (a download may set its full size first)."""
    done = partial = 0
    try:
        entries = list(os.scandir(blobs))
    except OSError:
        return 0, 0
    for e in entries:
        try:
            if e.name.endswith(".incomplete"):
                st = e.stat(follow_symlinks=False)
                partial += min(st.st_size, st.st_blocks * 512)
            else:
                done += e.stat().st_size
        except OSError:
            continue
    return done, partial


def _sweep(blobs: Path, *, now: float | None = None) -> int:
    """Remove the temporary files of downloads that died long ago (a hub download writes under a
    name of its own and renames it when complete; one that was killed leaves its part behind, never
    used again). A live download writes all the time, so nothing recent is touched."""
    now = time.time() if now is None else now
    gone = 0
    with contextlib.suppress(OSError):
        for e in os.scandir(blobs):
            if e.name.endswith(".incomplete"):
                with contextlib.suppress(OSError):
                    if now - e.stat(follow_symlinks=False).st_mtime > _STALE_PARTIAL_S:
                        os.unlink(e.path)
                        gone += 1
    return gone


def _hf(repo: str, load: Callable[[], Any], *, revision: str | None = None, patterns: list[str] | None = None,
        cache_dir: Path | None = None) -> Callable[[Job], None]:  # fmt: skip
    """A Hugging Face download: the loader the core uses (`load`) fetches it; this measures it in the
    repo's cache folder, against the size the Hub lists for those files (a dry run of the same call)."""

    def get(job: Job) -> None:
        from huggingface_hub import snapshot_download

        cache = cache_dir if cache_dir is not None else _hf_cache()
        blobs = _repo_blobs(repo, cache)
        _sweep(blobs)
        try:
            kw: dict[str, Any] = {"cache_dir": str(cache_dir)} if cache_dir is not None else {}
            infos = snapshot_download(repo, revision=revision, allow_patterns=patterns, dry_run=True, **kw)
            job.sized(sum(int(getattr(i, "file_size", 0) or 0) for i in infos if getattr(i, "will_download", True)))
        except Exception:  # noqa: BLE001 — the estimate stays; the download itself says what's wrong
            pass
        base = _blob_bytes(blobs)[0]

        def measured() -> int:
            done, partial = _blob_bytes(blobs)
            return max(0, done - base) + partial

        job.measure(measured)
        load()

    return get


def _wake(job: Job) -> None:
    from neo2.core.voice import wake

    def progress(done: int, total: int | None) -> None:
        job.sized(total)
        job.at(done)

    wake.ensure_model(progress)


def _embedder(models_dir: Path) -> Callable[[], Any]:
    def load() -> Any:
        from huggingface_hub import snapshot_download

        from neo2.core import models

        return snapshot_download(models.EMBEDDER, revision=models.EMBEDDER_REVISION, cache_dir=str(models_dir),
                                 allow_patterns=EMBEDDER_FILES)  # fmt: skip

    return load


def _embedder_here(models_dir: Path) -> bool:
    from neo2.core import models

    def ok(root: str) -> bool:
        return all(os.path.isfile(os.path.join(root, f)) for f in ("modules.json", "config.json", "model.safetensors"))

    return models._cached(models.EMBEDDER, models.EMBEDDER_REVISION, ok, cache_dir=str(models_dir),
                          allow_patterns=EMBEDDER_FILES) is not None  # fmt: skip


def _voice_pack_here(voice: str) -> bool:
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(KOKORO_VOICES, f"voices/{voice}.safetensors"), str)


def _voice_pack(voice: str) -> Callable[[], Any]:
    def load() -> Any:  # (as mlx-audio fetches it: its main revision, so its own cache lookup finds it)
        from huggingface_hub import snapshot_download

        return snapshot_download(KOKORO_VOICES, allow_patterns=[f"voices/{voice}.safetensors"])

    return load


# ---- Playwright's Chromium ------------------------------------------------------------------------------


def _playwright_dir() -> Path | None:
    """The installed playwright package's folder (not imported: its driver is a 128 MB node)."""
    import importlib.util

    spec = importlib.util.find_spec("playwright")
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def _browser_revision(name: str) -> str | None:
    """The revision of one of Playwright's browsers this playwright package drives (its browsers.json)."""
    pkg = _playwright_dir()
    if pkg is None:
        return None
    try:
        data = json.loads((pkg / "driver" / "package" / "browsers.json").read_text())
    except (OSError, ValueError):
        return None
    for b in data.get("browsers") or []:
        if isinstance(b, dict) and b.get("name") == name and b.get("revision"):
            return str(b["revision"])
    return None


def browsers_dir() -> Path:
    """Where Playwright keeps its browsers (PLAYWRIGHT_BROWSERS_PATH, else its default on macOS)."""
    p = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "").strip()
    if p == "0":
        pkg = _playwright_dir()
        return (pkg / "driver" / "package" / ".local-browsers") if pkg else Path(".local-browsers")
    if p:
        return Path(p).expanduser()
    return Path.home() / "Library" / "Caches" / "ms-playwright"


def chromium_here(*, shell_ok: bool = False) -> bool:
    """NEO's browser can start: Playwright's Chromium for this playwright package is installed (its
    INSTALLATION_COMPLETE marker) — or, `shell_ok`, the headless shell (enough for a headless page,
    which NEO starts on the full Chromium whenever it is there)."""
    root = browsers_dir()
    rev = _browser_revision("chromium")
    if rev and (root / f"chromium-{rev}" / "INSTALLATION_COMPLETE").is_file():
        return True
    if shell_ok:
        srev = _browser_revision("chromium-headless-shell")
        if srev and (root / f"chromium_headless_shell-{srev}" / "INSTALLATION_COMPLETE").is_file():
            return True
    return False


_PERCENT = re.compile(r"\|\s*(\d{1,3})% of ([\d.]+) (KiB|MiB|GiB)")
_UNIT = {"KiB": 1 << 10, "MiB": 1 << 20, "GiB": 1 << 30}


def _python() -> list[str]:
    """This interpreter, the way it runs now: isolated (-I) when this process is (a release engine)."""
    return [sys.executable, *(["-I"] if sys.flags.isolated else [])]


def _chromium(job: Job) -> None:
    """``playwright install chromium --no-shell`` (Chromium, and the ffmpeg it plays media with). Its
    download is measured where Playwright writes it (its temporary folder: one of ours, for this
    run); the size comes from its own progress lines."""
    tmp = Path(tempfile.mkdtemp(prefix="neo-chromium-"))
    env = {**os.environ, "TMPDIR": str(tmp)}
    cmd = [*_python(), "-m", "playwright", "install", "chromium", "--no-shell"]
    before = 0  # bytes of the downloads this run finished before the current one
    current = 0  # the current download's size, as Playwright said it
    tail: deque[str] = deque(maxlen=6)

    def zips() -> int:
        n = 0
        for z in tmp.glob("playwright-download-*/*"):
            with contextlib.suppress(OSError):
                n += z.stat().st_size
        return n

    job.measure(lambda: before + zips())
    try:
        job.proc = proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
        )
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.strip()
            if not line:
                continue
            tail.append(line)
            if line.startswith("Downloading "):
                before += current
                current = 0
                job.stage = ""
            elif m := _PERCENT.search(line):
                size = int(float(m.group(2)) * _UNIT[m.group(3)])
                if not current:
                    current = size
                    job.sized(before + current)
                if int(m.group(1)) >= 100:
                    job.at(before + current)
                    job.stage = "unpacking"
        rc = proc.wait()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if rc != 0:
        said = next((t for t in reversed(tail) if not t.startswith("|")), "") or f"exit status {rc}"
        raise RuntimeError(f"playwright install: {said[:200]}")


# ---- D-FINE --------------------------------------------------------------------------------------------


def _dfine(job: Job, models_dir: Path) -> None:
    from neo2.core import config
    from neo2.core.vision import detect

    src, _ = detect.source()
    _hf(src, detect.download, patterns=DFINE_FILES, cache_dir=models_dir / "hf")(job)
    config.set_vision_download_ok(True)  # the user's yes to it, kept


def items(s: Any = None) -> list[Item]:
    """Everything ``fetch`` knows, by group, as these settings would load it."""
    from neo2.core import models
    from neo2.core.config import settings

    s = s or settings()
    md = Path(s.models_dir)
    hf = _hf_cache()
    enc = str(getattr(s, "reflex_encoder", "multilingual") or "multilingual")
    laya_off = "" if getattr(s, "reflex", "laya") == "laya" else f"NEO_REFLEX is {s.reflex}"
    out: list[Item] = [
        Item("laya", "essential", f"Laya, NEO's reflex ({enc})", f"Hugging Face ({models.LAYA})",
             LAYA_BYTES.get(enc, LAYA_BYTES["multilingual"]), lambda: models.laya_here(enc),
             _hf(models.LAYA, lambda: models.laya_dir(enc), revision=models.LAYA_REVISION,
                 patterns=models.laya_patterns(enc)), where=hf, skip=laya_off),
    ]  # fmt: skip
    from neo2.core.reflex import zeroshot

    out.append(Item(
        "decide", "essential", "Laya for choosing apps (typed-decisions)", f"Hugging Face ({models.LAYA})",
        DECIDE_BYTES, lambda: zeroshot._local() is not None,
        _hf(models.LAYA, zeroshot.checkpoint_dir, revision=models.LAYA_REVISION, patterns=zeroshot.FILES), where=hf,
        skip=laya_off or ("" if sys.platform == "darwin" else "only on macOS"),
    ))  # fmt: skip
    out.append(Item(
        "embedder", "essential", "Memory search (MiniLM)", f"Hugging Face ({models.EMBEDDER})", EMBEDDER_BYTES,
        lambda: _embedder_here(md),
        _hf(models.EMBEDDER, _embedder(md), revision=models.EMBEDDER_REVISION, patterns=EMBEDDER_FILES, cache_dir=md),
        where=md, skip="" if getattr(s, "embeddings", "local") == "local" else "NEO_EMBEDDINGS is off",
    ))  # fmt: skip
    from neo2.core.voice import wake

    out.append(Item("wake", "essential", "The wake word, “Neo” (Vosk)", "alphacephei.com", WAKE_BYTES,
                    wake.model_here, _wake, disk=68_000_000, where=md))  # fmt: skip
    stt, tts = str(s.stt_model), str(s.tts_model)
    out.append(Item("stt", "local_voice", "On-device listening (Parakeet)", f"Hugging Face ({stt})",
                    STT_BYTES if models.revision(stt) else 0, lambda: models.here(stt),
                    _hf(stt, lambda: models.local_path(stt), revision=models.revision(stt)), where=hf))  # fmt: skip
    out.append(Item("tts", "local_voice", "On-device speaking (Kokoro)", f"Hugging Face ({tts})",
                    TTS_BYTES if models.revision(tts) else 0, lambda: models.here(tts),
                    _hf(tts, lambda: models.local_path(tts), revision=models.revision(tts)), where=hf))  # fmt: skip
    voice = str(getattr(s, "tts_voice", "") or "")
    if "kokoro" in tts.lower() and voice and "/" not in voice and not voice.endswith(".safetensors"):
        out.append(Item("voice_pack", "local_voice", f"Kokoro's voice ({voice})", f"Hugging Face ({KOKORO_VOICES})",
                        VOICE_PACK_BYTES, lambda: _voice_pack_here(voice),
                        _hf(KOKORO_VOICES, _voice_pack(voice), patterns=[f"voices/{voice}.safetensors"]),
                        where=hf))  # fmt: skip
    no_pw = "" if _playwright_dir() is not None else "the playwright package isn't installed"
    out.append(Item("chromium", "browser", "NEO's browser (Chromium)", "Playwright's servers (playwright.dev)",
                    CHROMIUM_BYTES, chromium_here, _chromium, disk=CHROMIUM_DISK, where=browsers_dir(),
                    skip=no_pw))  # fmt: skip
    out.append(Item("dfine", "vision", "Boxes on the camera (D-FINE small)", "Hugging Face (ustc-community/dfine-small-coco)",
                    DFINE_BYTES, _dfine_here, lambda job: _dfine(job, md), where=md))  # fmt: skip
    return out


def _dfine_here() -> bool:
    from neo2.core.vision import detect

    return detect.weights_here()


# ---- running them ------------------------------------------------------------------------------------------


def failure(e: BaseException) -> tuple[str, str]:
    """(code, words) for why a download failed — fixed words, never a server's own text, except
    the first line of an error that names no known cause (`failed`)."""
    chain: list[BaseException] = []
    x: BaseException | None = e
    while x is not None and len(chain) < 8 and x not in chain:
        chain.append(x)
        x = x.__cause__ or x.__context__
    for c in chain:
        if isinstance(c, OSError) and c.errno == errno.ENOSPC:
            return "disk_full", "The disk is full."
    names = {type(c).__name__ for c in chain}
    text = " ".join(str(c) for c in chain).lower()
    if "checksum" in text:
        return "checksum", "The download was damaged (it didn't match its checksum)."
    if "OfflineModeIsEnabled" in names or "hf_hub_offline" in text:
        return "offline", "Downloads are switched off (HF_HUB_OFFLINE)."
    if names & {"RepositoryNotFoundError", "RevisionNotFoundError", "EntryNotFoundError", "GatedRepoError"} or re.search(
        r"\b404\b", text
    ):
        return "not_found", "The file isn't on its server any more."
    offline = {
        "ConnectError", "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout", "TimeoutException",
        "LocalEntryNotFoundError", "ConnectionError", "NameResolutionError", "gaierror", "TimeoutError",
        "RemoteProtocolError", "NetworkError",
    }  # fmt: skip
    if names & offline or any(w in text for w in ("nodename nor servname", "network is unreachable", "timed out",
                                                   "getaddrinfo", "connection refused", "connection reset")):  # fmt: skip
        return "offline", "Can't reach the server — check the internet connection."
    first = (str(e).strip().splitlines() or [type(e).__name__])[0]
    return "failed", first[:200]


def _no_room(item: Item) -> str:
    """Why `item` can't fit on its disk ("" = it can)."""
    where = item.where
    while where is not None and not where.exists() and where != where.parent:
        where = where.parent
    if where is None:
        return ""
    try:
        free = shutil.disk_usage(where).free
    except OSError:
        return ""
    need = (item.disk or item.size) + _ROOM
    return "" if free >= need else f"{_gb(need)} free space is needed, {_gb(free)} is free."


def _gb(n: float) -> str:
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


class _Rate:
    """Bytes a second over the last few seconds, for the ETA."""

    def __init__(self, now: float) -> None:
        self.points: deque[tuple[float, int]] = deque([(now, 0)])

    def add(self, now: float, done: int) -> float:
        self.points.append((now, done))
        while len(self.points) > 2 and now - self.points[0][0] > _RATE_WINDOW_S:
            self.points.popleft()
        t0, d0 = self.points[0]
        return (done - d0) / (now - t0) if now > t0 else 0.0


def _work(job: Job) -> None:
    try:
        job.item.get(job)
    except BaseException as e:  # noqa: BLE001 — everything a download can raise is reported, never raised
        job.error = e
    finally:
        job.ended.set()


def run(groups: Iterable[str], emit: Callable[[dict[str, Any]], None], *, status: bool = False,
        stop: threading.Event | None = None, poll_s: float = _EMIT_EVERY_S, known: list[Item] | None = None) -> int:  # fmt: skip
    """Fetch every item of `groups` that isn't here (`status`: only say what's here). `emit` gets each
    event (see the module's doc); `stop` set: give up now (130). Returns the exit status."""
    t0 = time.monotonic()
    want = [g for g in GROUPS if g in set(groups)]
    todo_items = [it for it in (known if known is not None else items()) if it.group in want]
    here = {it.id: (not it.skip and _safe_here(it)) for it in todo_items}
    pending = [it for it in todo_items if not it.skip and not here[it.id]]
    emit({"event": "plan", "groups": want, "items": [it.plan(here[it.id]) for it in todo_items],
          "size": sum(it.size for it in pending)})  # fmt: skip
    if status:
        emit({"event": "done", "ok": not pending, "failed": [], "seconds": round(time.monotonic() - t0, 2)})
        return 0
    failed: list[str] = []
    totals = {it.id: it.size for it in pending}
    finished = 0
    rate = _Rate(time.monotonic())
    for it in todo_items:
        if stop is not None and stop.is_set():
            emit({"event": "done", "ok": False, "failed": failed, "cancelled": True,
                  "seconds": round(time.monotonic() - t0, 2)})  # fmt: skip
            return 130
        if it.skip:
            emit({"event": "item", "id": it.id, "state": "skipped", "message": it.skip})
            continue
        if here[it.id]:
            emit({"event": "item", "id": it.id, "state": "here"})
            continue
        if why := _no_room(it):
            failed.append(it.id)
            emit({"event": "item", "id": it.id, "state": "failed", "code": "disk_full", "message": why})
            finished += totals[it.id]
            continue
        emit({"event": "start", "id": it.id})
        job = Job(it)
        worker = threading.Thread(target=_work, args=(job,), name=f"neo-fetch-{it.id}", daemon=True)
        worker.start()
        last = 0.0
        stage = ""
        while True:
            ended = job.ended.wait(poll_s)
            if stop is not None and stop.is_set():
                if job.proc is not None:
                    with contextlib.suppress(Exception):
                        job.proc.terminate()
                emit({"event": "done", "ok": False, "failed": failed + [it.id], "cancelled": True,
                      "seconds": round(time.monotonic() - t0, 2)})  # fmt: skip
                return 130
            totals[it.id] = job.total
            now = time.monotonic()
            done = job.done()
            if job.stage != stage:
                stage = job.stage
                if stage:
                    emit({"event": "stage", "id": it.id, "stage": stage})
            if ended:
                break
            if now - last >= _EMIT_EVERY_S:
                last = now
                all_done = finished + done
                all_total = sum(totals.values())
                bps = rate.add(now, all_done)
                eta = round((all_total - all_done) / bps, 1) if bps > 0 else None
                emit({"event": "progress", "id": it.id, "done": done, "total": job.total, "all_done": all_done,
                      "all_total": all_total, "rate": int(bps), "eta_s": eta})  # fmt: skip
        finished += job.total
        if job.error is None and not _safe_here(it):
            job.error = RuntimeError(f"{it.label} isn't complete after its download")
        if job.error is not None:
            failed.append(it.id)
            code, words = failure(job.error)
            emit({"event": "item", "id": it.id, "state": "failed", "code": code, "message": words})
        else:
            emit({"event": "progress", "id": it.id, "done": job.total, "total": job.total, "all_done": finished,
                  "all_total": sum(totals.values()), "rate": 0, "eta_s": 0})  # fmt: skip
            emit({"event": "item", "id": it.id, "state": "done"})
    emit({"event": "done", "ok": not failed, "failed": failed, "seconds": round(time.monotonic() - t0, 2)})
    return 1 if failed else 0


def _safe_here(it: Item) -> bool:
    try:
        return bool(it.here())
    except Exception:  # noqa: BLE001 — a cache that can't be read counts as not here
        return False


# ---- the command line ------------------------------------------------------------------------------------


def _human(out: TextIO) -> Callable[[dict[str, Any]], None]:
    """The events as lines a person reads (without --json)."""
    labels: dict[str, str] = {}
    shown: dict[str, int] = {}

    def emit(ev: dict[str, Any]) -> None:
        kind = ev.get("event")
        if kind == "plan":
            for it in ev["items"]:
                labels[it["id"]] = it["label"]
            if ev["size"]:
                print(f"NEO will download about {_gb(ev['size'])}.", file=out)
        elif kind == "start":
            print(f"↓ {labels.get(ev['id'], ev['id'])}…", file=out)
        elif kind == "progress":
            pct = int(100 * ev["done"] / ev["total"]) if ev["total"] else 0
            if pct // 10 > shown.get(ev["id"], -1):
                shown[ev["id"]] = pct // 10
                eta = f", about {int(ev['eta_s'])} s left" if ev.get("eta_s") else ""
                print(f"  {pct:3d}% of {_gb(ev['total'])}{eta}", file=out)
        elif kind == "stage":
            print(f"  {ev['stage']}…", file=out)
        elif kind == "item":
            label = labels.get(ev["id"], ev["id"])
            state = ev["state"]
            if state == "failed":
                print(f"✗ {label}: {ev.get('message', '')}", file=out)
            elif state == "skipped":
                print(f"· {label}: skipped ({ev.get('message', '')})", file=out)
            else:
                print(f"✓ {label}" + (" (already here)" if state == "here" else ""), file=out)
        elif kind == "done":
            if ev.get("cancelled"):
                print("Stopped.", file=out)
            elif not ev["ok"]:
                print(f"Not everything downloaded: {', '.join(labels.get(i, i) for i in ev['failed'])}.", file=out)
        out.flush()

    return emit


def _status_lines(out: TextIO) -> Callable[[dict[str, Any]], None]:
    def emit(ev: dict[str, Any]) -> None:
        if ev.get("event") == "plan":
            for it in ev["items"]:
                mark = "·" if it["skip"] else "✓" if it["here"] else "✗"
                note = f"skipped: {it['skip']}" if it["skip"] else "here" if it["here"] else f"not here ({_gb(it['size'])})"
                print(f"  {mark} {it['label']} — {note}", file=out)
            out.flush()

    return emit


def _json_lines(out: TextIO) -> Callable[[dict[str, Any]], None]:
    lock = threading.Lock()

    def emit(ev: dict[str, Any]) -> None:
        with lock:
            out.write(json.dumps(ev, ensure_ascii=False) + "\n")
            out.flush()

    return emit


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="neo2 fetch", description="Download what NEO needs, with progress.")
    for g in GROUPS:
        ap.add_argument(_FLAG[g], dest=g, action="store_true")
    ap.add_argument("--status", action="store_true", help="say what's here; download nothing")
    ap.add_argument("--json", action="store_true", help="JSON lines on stdout (NEO's Setup window)")
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    groups = [g for g in GROUPS if getattr(args, g)] or ["essential"]
    out: TextIO = sys.stdout
    saved: int | None = None
    was = sys.stdout
    if args.json:
        # Only the events go to stdout: whatever else prints (a library's warning, a progress bar,
        # a C library writing to its stdout) lands on stderr.
        sys.stdout.flush()
        saved = os.dup(1)
        out = os.fdopen(os.dup(saved), "w", buffering=1, encoding="utf-8")
        os.dup2(2, 1)
        sys.stdout = sys.stderr
    try:
        return _main(args, groups, out)
    finally:
        if saved is not None:  # (stdout as it was: for a caller in this process, a test)
            sys.stdout = was
            with contextlib.suppress(Exception):
                sys.stdout.flush()
            os.dup2(saved, 1)
            os.close(saved)
            with contextlib.suppress(Exception):
                out.close()


def _main(args: argparse.Namespace, groups: list[str], out: TextIO) -> int:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    with contextlib.suppress(Exception):
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    stop = threading.Event()
    before: dict[int, Any] = {}
    if threading.current_thread() is threading.main_thread():  # (Setup's Cancel is a SIGTERM)
        for sig in (signal.SIGTERM, signal.SIGINT):
            before[sig] = signal.signal(sig, lambda *_: stop.set())
    emit = _json_lines(out) if args.json else (_status_lines(out) if args.status else _human(out))
    try:
        code = run(groups, emit, status=args.status, stop=stop)
    except Exception as e:  # noqa: BLE001 — the window hears why, in its own words
        c, words = failure(e)
        emit({"event": "done", "ok": False, "failed": [], "code": c, "message": words, "seconds": 0})
        code = 1
    finally:
        for sig, handler in before.items():
            signal.signal(sig, handler)
    out.flush()
    if code == 130:
        os._exit(130)  # a download thread can't be interrupted: leave without waiting for it
    return code


# ---- the core's own downloads --------------------------------------------------------------------------------


@dataclass
class State:
    """One group's download, as every window shows it."""

    state: str = "idle"  # idle | running | done | failed | cancelled
    done: int = 0
    total: int = 0
    eta_s: float | None = None
    code: str = ""  # a failure's code (see failure())
    message: str = ""  # …and its words
    item: str = ""  # the item downloading now
    changed: float = field(default_factory=time.monotonic)

    def as_dict(self) -> dict[str, Any]:
        return {"state": self.state, "done": self.done, "total": self.total, "eta_s": self.eta_s,
                "code": self.code or None, "item": self.item or None}  # fmt: skip


Spawn = Callable[[str], AsyncIterator[dict[str, Any]]]


async def _cli(group: str, procs: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
    """`python -m neo2 fetch --<group> --json` as this interpreter runs, its events one by one."""
    proc = await asyncio.create_subprocess_exec(
        *_python(), "-m", "neo2", "fetch", _FLAG[group], "--json",
        stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,  # (its stderr: the core's log)
    )  # fmt: skip
    procs[group] = proc
    try:
        assert proc.stdout is not None
        async for raw in proc.stdout:
            try:
                ev = json.loads(raw)
            except ValueError:
                continue
            if isinstance(ev, dict):
                yield ev
        await proc.wait()
    finally:
        procs.pop(group, None)
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()


class Downloads:
    """The core's downloads, each asked for by the user in a window: one of each group at a time, in a
    child process (``python -m neo2 fetch --<group> --json``) — never in the core, whose voice and
    websocket keep going — its progress for whoever listens (:meth:`listen`)."""

    def __init__(self, spawn: Spawn | None = None) -> None:
        self._procs: dict[str, Any] = {}
        self._spawn: Spawn = spawn or (lambda g: _cli(g, self._procs))
        self._states: dict[str, State] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._listeners: list[Callable[[str, State], None]] = []
        self._cancelled: set[str] = set()

    def listen(self, fn: Callable[[str, State], None]) -> None:
        self._listeners.append(fn)

    def state(self, group: str) -> State:
        return self._states.get(group) or State()

    def running(self, group: str) -> bool:
        t = self._tasks.get(group)
        return t is not None and not t.done()

    def start(self, group: str) -> bool:
        """Start `group`'s download (False: one runs already). Call on the core's event loop."""
        if group not in GROUPS:
            raise ValueError(f"no such download: {group!r}")
        if self.running(group):
            return False
        self._cancelled.discard(group)
        self._states[group] = State(state="running")
        self._tasks[group] = asyncio.get_running_loop().create_task(self._run(group))
        self._tell(group)
        return True

    async def wait(self, group: str) -> State:
        t = self._tasks.get(group)
        if t is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.shield(t)
        return self.state(group)

    def cancel(self, group: str) -> bool:
        """Stop `group`'s download (False: none runs)."""
        if not self.running(group):
            return False
        self._cancelled.add(group)
        proc = self._procs.get(group)
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
        else:
            self._tasks[group].cancel()
        return True

    def _tell(self, group: str) -> None:
        st = self.state(group)
        st.changed = time.monotonic()
        for fn in list(self._listeners):
            try:
                fn(group, st)
            except Exception as e:  # noqa: BLE001
                print(f"[fetch] a listener failed: {type(e).__name__}")

    async def _run(self, group: str) -> None:
        st = self._states[group]
        ended = False
        try:
            async for ev in self._spawn(group):
                kind = ev.get("event")
                if kind == "start":
                    st.item = str(ev.get("id") or "")
                elif kind == "progress":
                    st.done, st.total = int(ev.get("all_done") or 0), int(ev.get("all_total") or 0)
                    eta = ev.get("eta_s")
                    st.eta_s = float(eta) if isinstance(eta, int | float) else None
                elif kind == "plan":
                    st.total = int(ev.get("size") or 0)
                elif kind == "item" and ev.get("state") == "failed":
                    st.code = str(ev.get("code") or "failed")
                    st.message = str(ev.get("message") or "")[:200]
                elif kind == "done":
                    ended = True
                    st.state = "done" if ev.get("ok") else ("cancelled" if ev.get("cancelled") else "failed")
                    if st.state == "failed" and not st.code:
                        st.code = str(ev.get("code") or "failed")
                self._tell(group)
        except asyncio.CancelledError:
            st.state = "cancelled"
        except Exception as e:  # noqa: BLE001 — NEO keeps running; the window says it failed
            st.state, (st.code, st.message) = "failed", failure(e)
        if group in self._cancelled:
            st.state = "cancelled"
        elif not ended and st.state == "running":
            st.state, st.code = "failed", st.code or "failed"
        st.item, st.eta_s = "", None
        self._tell(group)


downloads = Downloads()  # NEO.app's core


# ---- the on-device voice, as every window shows it --------------------------------------------------------

_ready_at = 0.0
_ready: bool | None = None
_READY_RECHECK_S = 5.0


def local_voice_ready(*, fresh: bool = False) -> bool:
    """The on-device voice's models are on this Mac (neo2.core.voice.pipeline.local_ready), looked at
    again at most every few seconds while they aren't (a client's hello asks often)."""
    global _ready, _ready_at
    now = time.monotonic()
    if fresh or _ready is not True and (_ready is None or now - _ready_at >= _READY_RECHECK_S):
        from neo2.core.voice.pipeline import local_ready

        _ready, _ready_at = local_ready(), now
    return bool(_ready)


def local_voice(*, fresh: bool = False) -> dict[str, Any]:
    """The `brains` summary's "local_voice" and the ``local_voice`` event: {state: ready | missing |
    downloading | failed, size (bytes), done, total, eta_s, code}. Nothing is downloaded until the
    user asks (`voice_download`)."""
    st = downloads.state("local_voice")
    out: dict[str, Any] = {"size": LOCAL_VOICE_BYTES}
    if st.state == "running":
        return {**out, "state": "downloading", "done": st.done, "total": st.total or LOCAL_VOICE_BYTES,
                "eta_s": st.eta_s}  # fmt: skip
    if local_voice_ready(fresh=fresh or st.state == "done"):
        return {**out, "state": "ready"}
    if st.state == "failed":
        return {**out, "state": "failed", "code": st.code or "failed"}
    return {**out, "state": "missing"}


def start_local_voice() -> str:
    """The user asked for the on-device voice: "" when its download started, else why not ("ready":
    it's here; "busy": it is downloading)."""
    if local_voice_ready(fresh=True):
        return "ready"
    return "" if downloads.start("local_voice") else "busy"


__all__ = [
    "GROUPS", "Downloads", "Item", "Job", "LOCAL_VOICE_BYTES", "State", "browsers_dir", "chromium_here", "downloads",
    "failure", "items", "local_voice", "local_voice_ready", "main", "run", "start_local_voice",
]  # fmt: skip
