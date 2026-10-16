"""No API key in NEO's log: the core's stdout and stderr (NEO.app's core.log, which users share)
go through :func:`redact` on their way out.

Every key NEO has loaded — the settings' keys (environment, saved in NEO, .env), the endpoints'
keys, and a key the Brains & Keys window or the CLI just handed in to test or save
(:func:`remember`) — of 8 characters or more is replaced by "••••" and its last 4 characters
(keystore.mask). That catches a provider's error that quotes the key it refused (OpenAI-style 401
bodies do), whatever line printed it. The bus's "error" messages and the health list pass through
the same function (neo2.core.server.ws, neo2.core.health).

:func:`install` wraps sys.stdout and sys.stderr once, at the core's start (neo2.core.main). It
never builds the settings itself (that would read ~/.neo before `--help` answers): only the ones
already built are read.
"""

from __future__ import annotations

import io
import logging
import sys
import threading
from typing import Any, TextIO

MIN_LEN = 8  # a key shorter than this isn't looked for (it would match ordinary words)
_KEY_FIELDS = (
    "gemini_api_key", "groq_api_key", "anthropic_api_key", "cerebras_api_key", "openai_api_key", "openrouter_api_key",
)  # fmt: skip
_remembered: set[str] = set()  # keys handed in to test or save (this process's lifetime, in memory only)
_lock = threading.Lock()
_busy = threading.local()  # (a redaction that prints must not redact itself again)


def remember(*keys: Any) -> None:
    """Keys handed in by the Brains & Keys window or the CLI (to test, list models or save): masked
    in the log from now on, even before they are saved or if they never are."""
    with _lock:
        for k in keys:
            if isinstance(k, str) and len(k.strip()) >= MIN_LEN:
                _remembered.add(k.strip())


def _from_settings(s: Any) -> list[str]:
    out: list[str] = []
    if s is None:
        return out
    for f in _KEY_FIELDS:
        v = getattr(s, f, "")
        if isinstance(v, str):
            out.append(v)
    extra = getattr(s, "gemini_api_keys", "")
    if isinstance(extra, str):
        out += extra.split(",")
    custom = getattr(s, "custom_keys", None)
    if isinstance(custom, dict):
        out += [v for v in custom.values() if isinstance(v, str)]
    return out


def secrets() -> list[str]:
    """The key values to mask now, longest first (one inside another is masked whole)."""
    found: list[str] = []
    try:
        # (only what is loaded already: nothing is imported, read or built for this)
        config, runtime = sys.modules.get("neo2.core.config"), sys.modules.get("neo2.core.runtime")
        if config is not None:
            found += _from_settings(getattr(config, "_settings", None))  # NEO.app's, once built
        rt = runtime.current_runtime() if runtime is not None else None
        if rt is not None:
            found += _from_settings(getattr(rt, "settings", None))  # an embedded agent's own
    except Exception:  # noqa: BLE001 — the log is never stopped by this
        pass
    with _lock:
        found += list(_remembered)
    keys = {k.strip() for k in found if isinstance(k, str) and len(k.strip()) >= MIN_LEN}
    return sorted(keys, key=len, reverse=True)


def _mask(key: str) -> str:
    from neo2.core.keystore import mask

    return mask(key)


def redact(text: Any) -> str:
    """`text` with every loaded key replaced by ••••<last 4> (see the module's doc)."""
    s = text if isinstance(text, str) else str(text)
    if not s:
        return s
    for k in secrets():
        if k in s:
            s = s.replace(k, _mask(k))
    return s


class RedactingStream(io.TextIOBase):
    """A text stream that redacts what is written to it, a line at a time, and hands it on. A line
    is held until it ends (a key split over two writes is still caught), or until flush() — print's
    flush=True, the logging handler's, and the interpreter's at exit."""

    _MAX_HELD = 16384

    def __init__(self, inner: TextIO) -> None:
        self._inner = inner
        self._held = ""
        self._wlock = threading.RLock()

    @property
    def inner(self) -> TextIO:
        return self._inner

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:  # type: ignore[override]
        if not isinstance(s, str):
            s = str(s)
        with self._wlock:
            self._held += s
            cut = self._held.rfind("\n")
            if cut < 0 and len(self._held) < self._MAX_HELD and "\r" not in self._held:
                return len(s)
            if cut < 0:
                out, self._held = self._held, ""
            else:
                out, self._held = self._held[: cut + 1], self._held[cut + 1 :]
            self._emit(out)
        return len(s)

    def _emit(self, out: str) -> None:
        if not out:
            return
        if getattr(_busy, "on", False):
            self._inner.write(out)
            return
        _busy.on = True
        try:
            self._inner.write(redact(out))
        finally:
            _busy.on = False

    def flush(self) -> None:
        with self._wlock:
            out, self._held = self._held, ""
            self._emit(out)
        try:
            self._inner.flush()
        except (OSError, ValueError):
            pass

    def isatty(self) -> bool:
        try:
            return self._inner.isatty()
        except (OSError, ValueError):
            return False

    def fileno(self) -> int:
        return self._inner.fileno()

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return getattr(self._inner, "encoding", "utf-8")

    @property
    def errors(self) -> str | None:  # type: ignore[override]
        return getattr(self._inner, "errors", None)

    @property
    def buffer(self) -> Any:
        return getattr(self._inner, "buffer", None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def install() -> bool:
    """Wrap sys.stdout and sys.stderr (once; True when it did now), and point NEO's own log handlers
    that write to them at the wrapped ones."""
    done = False
    for name in ("stdout", "stderr"):
        cur = getattr(sys, name)
        if cur is None or isinstance(cur, RedactingStream):
            continue
        wrapped = RedactingStream(cur)
        setattr(sys, name, wrapped)
        done = True
        for lg in (logging.getLogger(), logging.getLogger("neo")):
            for h in lg.handlers:
                if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is cur:
                    h.setStream(wrapped)
    return done
