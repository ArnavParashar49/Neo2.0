"""Local wake-word spotting with a grammar-restricted Vosk recogniser (tiny CPU cost)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

from neo2.core.config import settings

_MODEL = "vosk-model-small-en-us-0.15"
_URL = f"https://alphacephei.com/vosk/models/{_MODEL}.zip"
_SHA256 = "30f26242c4eb449f948e42cb302dd7a686cb29a3423a8367f99ff41780942498"  # pinned: the zip we tested
_MARKERS = ("am/final.mdl", "conf/model.conf")  # a complete model has these
# The wake word is just "Neo" (a greeting in front is fine), and it has to open what's said:
# "Neo, open Notes" wakes NEO, "I need a new phone" doesn't. Measured on 264 synthetic
# utterances from eight voices (US, UK, Irish, Australian, Indian English) through one
# continuously running recogniser, as NEO runs it: false wakes on ordinary speech down from 68 %
# to 5 %, with 92 % of wakes caught (the ⌥⌘ shortcut is always there). "neil" is how
# Indian-English "Neo" often comes out, so it counts too.
_WAKE_RE = re.compile(r"^(?:(?:hey|hi|ok|okay|yo)\s+)?(?:neo|neil|nio)\b", re.I)
_STOP_RE = re.compile(r"\b(stop|cancel)\b", re.I)
_GREETINGS = ["hey", "hi", "okay", "yo"]
# Words that sound like "Neo" but aren't: with them in the grammar, "you know", "Leo", "neon"…
# are recognised as themselves instead of being forced onto "neo". (Not "new", "knee", "near"
# or "need": those swallowed real wakes in some accents.)
_DECOYS = [
    "know", "no", "now", "leo", "neon", "nearly", "me", "nemo", "video", "snow", "theo", "rio",
    "nina", "leon", "oreo", "casino", "renew",
]
_STABLE = 3  # partial results in a row (~60 ms) before a wake counts


def _complete(d: Path) -> bool:
    return all((d / m).is_file() for m in _MARKERS)


def ensure_model() -> Path:
    """The wake-word model, downloaded once. The download is checked against a pinned hash and
    unpacked into a temporary folder that only becomes the model once it's complete — quitting
    halfway leaves nothing half-installed behind."""
    root = settings().models_dir
    d = root / _MODEL
    if _complete(d):
        return d
    if d.exists():
        print("[wake] the wake-word model is incomplete (an interrupted download?) — fetching it again")
        shutil.rmtree(d, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    import httpx

    print("[wake] downloading the wake-word model (~40 MB)…")
    with tempfile.TemporaryDirectory(dir=root, prefix=".download-") as tmp:
        zpath = Path(tmp) / f"{_MODEL}.zip"
        h = hashlib.sha256()
        with httpx.stream("GET", _URL, timeout=httpx.Timeout(30.0, read=60.0), follow_redirects=True) as r:
            r.raise_for_status()
            with zpath.open("wb") as f:
                for chunk in r.iter_bytes(1 << 16):
                    h.update(chunk)
                    f.write(chunk)
        if h.hexdigest() != _SHA256:
            raise RuntimeError("the wake-word model download didn't match its checksum — not using it")
        out = Path(tmp) / "x"
        with zipfile.ZipFile(zpath) as z:
            for name in z.namelist():  # no absolute paths or ../ escapes
                if name.startswith("/") or ".." in Path(name).parts:
                    raise RuntimeError(f"unexpected path in the model archive: {name}")
            z.extractall(out)
        if not _complete(out / _MODEL):
            raise RuntimeError("the wake-word model archive is missing files")
        os.replace(out / _MODEL, d)
    return d


class WakeSpotter:
    def __init__(self, rate: int = 16000) -> None:
        import vosk

        vosk.SetLogLevel(-1)
        self._model = vosk.Model(str(ensure_model()))
        grammar = json.dumps(_GREETINGS + ["neo", "neil"] + _DECOYS + ["stop", "cancel", "[unk]"])
        self._rec = vosk.KaldiRecognizer(self._model, rate, grammar)
        self._streak = 0

    def feed(self, frame: bytes) -> str | None:
        """Returns 'wake' / 'stop' on a (stable) partial hit, 'wake_final' / 'stop_final' once the
        recogniser has committed to it, else None."""
        if self._rec.AcceptWaveform(frame):
            text = json.loads(self._rec.Result()).get("text", "")
            final = True
        else:
            text = json.loads(self._rec.PartialResult()).get("partial", "")
            final = False
        return self._judge(text, final)

    def _judge(self, text: str, final: bool) -> str | None:
        # Raw text: an unrecognised word before "neo" ("[unk] neo" — "I need…") means it wasn't
        # said first, so it doesn't wake NEO.
        if _WAKE_RE.search(text):
            self._streak = 0 if final else self._streak + 1
            if final or self._streak >= _STABLE:
                self._rec.Reset()
                self._streak = 0
                return "wake_final" if final else "wake"
            return None
        self._streak = 0
        if not text:
            return None
        if _STOP_RE.search(text):
            self._rec.Reset()
            return "stop_final" if final else "stop"
        return None

    def reset(self) -> None:
        self._rec.Reset()
        self._streak = 0
