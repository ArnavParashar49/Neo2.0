"""Local wake-word spotting with a grammar-restricted Vosk recogniser (tiny CPU cost)."""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from neo.config import settings

_MODEL = "vosk-model-small-en-us-0.15"
_URL = f"https://alphacephei.com/vosk/models/{_MODEL}.zip"
_WAKE_RE = re.compile(r"\b(?:hey|hi|ok|okay)?\s*(?:neo|nio|neal|neil)\b", re.I)
_STOP_RE = re.compile(r"\b(stop|cancel|shut up|quiet)\b", re.I)


def ensure_model() -> Path:
    d = settings().models_dir / _MODEL
    if d.exists():
        return d
    zpath = settings().models_dir / f"{_MODEL}.zip"
    print("[wake] downloading Vosk small model (~40 MB)…")
    urlretrieve(_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(settings().models_dir)
    zpath.unlink(missing_ok=True)
    return d


class WakeSpotter:
    def __init__(self, rate: int = 16000) -> None:
        import vosk

        vosk.SetLogLevel(-1)
        self._model = vosk.Model(str(ensure_model()))
        grammar = json.dumps(["hey neo", "neo", "stop", "cancel", "[unk]"])
        self._rec = vosk.KaldiRecognizer(self._model, rate, grammar)

    def feed(self, frame: bytes) -> str | None:
        """Returns 'wake' / 'stop' when spotted, else None."""
        if self._rec.AcceptWaveform(frame):
            text = json.loads(self._rec.Result()).get("text", "")
        else:
            text = json.loads(self._rec.PartialResult()).get("partial", "")
        if not text:
            return None
        if _WAKE_RE.search(text):
            self._rec.Reset()
            return "wake"
        if _STOP_RE.search(text):
            self._rec.Reset()
            return "stop"
        return None

    def reset(self) -> None:
        self._rec.Reset()
