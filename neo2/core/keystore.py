"""NEO's own store of API keys and brain choices ("Brains & Keys").

Two files, made only when the user saves something (nothing is ever copied from a .env file or the
environment into them):

* ``~/.neo/api_keys.json`` (:class:`FileKeyStore`): the keys saved in NEO, per provider id::

      {"version": 1, "keys": {"anthropic": {"key": "sk-ant-…", "saved": "2026-10-16T09:12:00Z"},
                              "gemini": {"key": "AIza…", "extra": ["AIza…"], "saved": "…"}}}

* ``~/.neo/brains.json`` (:class:`SavedConfig`): which brain each role uses, and the rest of the
  window's choices (no key is ever in it).

Both are mode 0600 in the 0700 ~/.neo, written by NEO's core and its CLI alone: under an flock on
``~/.neo/.store.lock``, through a temporary file (O_EXCL, 0600) that is written, fsynced and renamed
over the old one, and the folder fsynced. A file that can't be read counts as empty and is never
overwritten silently: it is set aside as ``<name>.bad-<time>`` when the user next saves, and its
state says so. "Readable only by your Mac account" — never more is promised.

Settings read them through :class:`KeyStoreSource` and :class:`SavedSource` (neo2.core.config):
after the environment, before the .env files. With effects off (the suite, NEO_NO_EFFECTS=1) the
real files are never read or written — a test passes paths of its own.

A key never leaves this module except into Settings' fields, and as its last 4 characters
(:func:`last4`) for the Brains & Keys window and the CLI.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import secrets
import time
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

from neo2.core import effects

KEY_FILE = "api_keys.json"
SAVED_FILE = "brains.json"
LOCK_FILE = ".store.lock"
VERSION = 1
MAX_KEY = 512  # characters
_MAX_FILE = 1024 * 1024  # a larger file isn't one NEO wrote
_LOCK_S = 5.0
_PROVIDER = re.compile(r"[a-z][a-z0-9_]{1,23}|custom:[a-z0-9-]{1,24}")


# ---- the real files are the user's ---------------------------------------------------------------


def _fold(parts: Sequence[str]) -> list[str]:
    return [unicodedata.normalize("NFC", x).casefold() for x in parts]


def is_real(path: str | os.PathLike) -> bool:
    """`path` lies in the user's real ~/.neo — by name first (nothing on disk is looked at for
    that), then however it's linked or spelt (neo2.core.config's check)."""
    p = Path(os.path.abspath(os.path.expanduser(str(path))))
    homes = [Path.home() / ".neo"]
    try:
        import pwd

        homes.append(Path(pwd.getpwuid(os.getuid()).pw_dir) / ".neo")
    except (ImportError, KeyError, OSError):
        pass
    for h in homes:
        hp = _fold(h.parts)
        if _fold(p.parts)[: len(hp)] == hp:
            return True
    from neo2.core.config import _in_neo_home

    return _in_neo_home(p, moved=False) is not None


def refused(path: str | os.PathLike) -> str:
    """Why `path` may not be read or written now ("" when it may): it is the user's real file and
    real effects are off, or this is one of NEO's own test runs (even one acting as NEO.app)."""
    why = effects.why_off() or ("running under pytest" if effects._under_pytest() else "")
    if why and is_real(path):
        return why
    return ""


def _guard(path: Path, what: str) -> None:
    if why := refused(path):
        raise RuntimeError(f"{what}: real effects are off ({why}), and {path.name} in ~/.neo is the user's")


# ---- writing: one writer at a time, never half a file ------------------------------------------------


@contextlib.contextmanager
def locked(folder: Path) -> Iterator[None]:
    """The store's lock (an flock on <folder>/.store.lock): one writer at a time, the core's or the
    CLI's. Waits up to 5 s for another writer, then raises RuntimeError."""
    _private_dir(folder)
    fd = os.open(folder / LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        deadline = time.monotonic() + _LOCK_S
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise RuntimeError("NEO's key store is busy (another save is under way)") from None
                time.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _private_dir(folder: Path) -> None:
    """The folder exists, and is this user's only (0700): it is NEO's own (~/.neo, or a test's)."""
    if not folder.is_dir():
        folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(folder, 0o700)


def atomic_write(path: Path, data: bytes, *, mode: int = 0o600) -> None:
    """`data` becomes `path` whole or not at all: a temporary file in the same folder (O_EXCL,
    `mode`), written and fsynced, renamed over `path`, then the folder fsynced. A link at `path` is
    followed (the file it points to is the one replaced, and the link stays)."""
    target = Path(os.path.realpath(path)) if os.path.islink(path) else Path(path)
    folder = target.parent
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f".{target.name}.{secrets.token_hex(6)}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        try:
            os.fchmod(fd, mode)  # (the umask can't widen it, but make it exact)
            view = memoryview(data)
            while view:
                n = os.write(fd, view)
                view = view[n:]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    try:
        dfd = os.open(folder, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dfd)
    except OSError:
        pass
    finally:
        os.close(dfd)


def _stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def old_copies(path: str | os.PathLike) -> int:
    """How many copies of `path` NEO set aside (<name>.bad-<time>) are still there: they may hold
    keys a Remove here never touches, so the window and --doctor say so. Never their contents."""
    p = Path(path)
    if refused(p):
        return 0
    try:
        return sum(1 for q in p.parent.glob(f"{p.name}.bad-*") if q.is_file())
    except OSError:
        return 0


def _quarantine(path: Path) -> Path | None:
    """Set an unreadable file aside (<name>.bad-<time>) so a save never overwrites it silently."""
    bad = path.with_name(f"{path.name}.bad-{time.strftime('%Y%m%dT%H%M%S')}")
    n = 1
    while bad.exists():
        n += 1
        bad = path.with_name(f"{path.name}.bad-{time.strftime('%Y%m%dT%H%M%S')}-{n}")
    try:
        os.replace(path, bad)
    except FileNotFoundError:
        return None
    return bad


# ---- what a file holds ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class StoreState:
    """How a store file is: `ok` (readable, or not there yet), `exists`, and when it can't be read,
    `problem` ("unreadable", "corrupt" or "newer": made by a newer NEO)."""

    path: Path
    ok: bool = True
    exists: bool = False
    problem: str = ""


def _read_json(path: Path) -> tuple[dict[str, Any] | None, StoreState]:
    try:
        with open(path, "rb") as f:
            raw = f.read(_MAX_FILE + 1)
    except FileNotFoundError:
        return {}, StoreState(path)
    except OSError:
        return None, StoreState(path, False, True, "unreadable")
    if len(raw) > _MAX_FILE:
        return None, StoreState(path, False, True, "corrupt")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None, StoreState(path, False, True, "corrupt")
    if not isinstance(data, dict):
        return None, StoreState(path, False, True, "corrupt")
    v = data.get("version")
    if isinstance(v, int) and not isinstance(v, bool) and v > VERSION:
        return None, StoreState(path, False, True, "newer")
    if v != VERSION:
        return None, StoreState(path, False, True, "corrupt")
    return data, StoreState(path, True, True)


def clean_key(value: Any) -> str:
    """A pasted key as kept: spaces and quotes around it gone. One with a line break or another
    control character in it, over 512 characters, or empty is refused (ValueError, which never
    says the key)."""
    s = str(value if value is not None else "").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    if not s:
        raise ValueError("the key is empty")
    if len(s) > MAX_KEY:
        raise ValueError(f"the key is longer than {MAX_KEY} characters")
    if any(not c.isprintable() or c.isspace() for c in s):
        raise ValueError("the key has a line break, a space or a control character in it")
    return s


def check_provider(provider: Any) -> str:
    p = str(provider or "").strip().lower()
    if not _PROVIDER.fullmatch(p):
        raise ValueError("not a provider id")
    return p


def last4(key: str | None) -> str | None:
    """The last 4 characters of a key of 12 or more, else None (a short key shows nothing of itself)."""
    k = (key or "").strip()
    return k[-4:] if len(k) >= 12 else None


def mask(key: str | None) -> str:
    """"••••x9Qa" (12 characters or more), "••••" (shorter), "" (none)."""
    if not (key or "").strip():
        return ""
    end = last4(key)
    return "••••" + (end or "")


@dataclass(frozen=True)
class StoredKey:
    key: str = field(repr=False)
    extra: tuple[str, ...] = field(default=(), repr=False)
    saved: str = ""

    def __repr__(self) -> str:
        return f"StoredKey({mask(self.key)!r}, extra={len(self.extra)}, saved={self.saved!r})"


class KeyStore(Protocol):
    """Where keys saved in NEO live (a file now; the Keychain could be another, later)."""

    def get(self, provider: str) -> StoredKey | None: ...

    def put(self, provider: str, key: str, extra: Sequence[str] = ()) -> None: ...

    def remove(self, provider: str) -> bool: ...

    def list(self) -> dict[str, StoredKey]: ...


class FileKeyStore:
    """~/.neo/api_keys.json (or a test's own file): see the module's doc."""

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)

    def __repr__(self) -> str:
        return f"FileKeyStore({str(self.path)!r})"

    def _read(self) -> tuple[dict[str, StoredKey], StoreState]:
        _guard(self.path, "read the key store")
        data, st = _read_json(self.path)
        if data is None:
            return {}, st
        keys = data.get("keys", {})
        if not isinstance(keys, dict):
            return {}, StoreState(self.path, False, True, "corrupt")
        out: dict[str, StoredKey] = {}
        for p, e in keys.items():
            try:
                pid = check_provider(p)
                if not isinstance(e, dict) or not isinstance(e.get("key"), str):
                    raise ValueError
                extra = e.get("extra") or []
                if not isinstance(extra, list) or not all(isinstance(x, str) for x in extra):
                    raise ValueError
                saved = e.get("saved", "")
                out[pid] = StoredKey(
                    clean_key(e["key"]), tuple(clean_key(x) for x in extra), saved if isinstance(saved, str) else ""
                )
            except ValueError:
                # One entry NEO didn't write: the file isn't NEO's to rewrite without asking.
                return {}, StoreState(self.path, False, True, "corrupt")
        return out, st

    def state(self) -> StoreState:
        return self._read()[1]

    def list(self) -> dict[str, StoredKey]:
        return self._read()[0]

    def get(self, provider: str) -> StoredKey | None:
        return self.list().get(check_provider(provider))

    def _write(self, keys: Mapping[str, StoredKey]) -> None:
        body: dict[str, Any] = {}
        for p in sorted(keys):
            k = keys[p]
            e: dict[str, Any] = {"key": k.key}
            if k.extra:
                e["extra"] = list(k.extra)
            e["saved"] = k.saved
            body[p] = e
        data = json.dumps({"version": VERSION, "keys": body}, indent=1, ensure_ascii=False) + "\n"
        atomic_write(self.path, data.encode("utf-8"))

    def put(self, provider: str, key: str, extra: Sequence[str] = ()) -> None:
        """Save `key` (and, for Gemini, its extra keys) for `provider`, replacing any saved before.
        A file that can't be read is set aside first (never overwritten silently)."""
        pid = check_provider(provider)
        k = clean_key(key)
        more = tuple(dict.fromkeys(x for x in (clean_key(x) for x in extra) if x != k))
        _guard(self.path, "save a key")
        with locked(self.path.parent):
            keys, st = self._read()
            if st.exists and not st.ok:
                _quarantine(self.path)
                keys = {}
            keys[pid] = StoredKey(k, more, _stamp())
            self._write(keys)

    def remove(self, provider: str) -> bool:
        """Forget the key saved for `provider`. False: none was saved (or the file can't be read,
        and is left as it is)."""
        pid = check_provider(provider)
        _guard(self.path, "remove a key")
        if not self.path.exists():
            return False
        with locked(self.path.parent):
            keys, st = self._read()
            if not st.ok or pid not in keys:
                return False
            del keys[pid]
            self._write(keys)
            return True


# ---- brains.json --------------------------------------------------------------------------------------

# What brains.json may hold, and the kind of value each takes (anything else in it is ignored).
SAVED_FIELDS: dict[str, tuple[type, ...]] = {
    "brain": (str,),
    "fast_brain": (str,),
    "offline_brain": (str,),
    "eyes_brain": (str,),
    "light_brain": (str,),
    "backup_brains": (list, type(None)),
    "cloud_eyes": (str,),
    "voice": (str,),
    "custom_providers": (list,),
    "daily_budgets": (dict,),
    "known_models": (dict,),
}
_CHOICES = {"cloud_eyes": ("auto", "on", "off"), "voice": ("auto", "live", "local", "off")}


class SavedConfig:
    """~/.neo/brains.json (or a test's own file): the roles and choices saved in NEO. No key."""

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)

    def __repr__(self) -> str:
        return f"SavedConfig({str(self.path)!r})"

    def read(self) -> tuple[dict[str, Any], StoreState]:
        """The saved choices NEO knows (each of the kind it takes), and the file's state."""
        _guard(self.path, "read the saved brains")
        data, st = _read_json(self.path)
        if data is None:
            return {}, st
        out: dict[str, Any] = {}
        for k, kinds in SAVED_FIELDS.items():
            if k not in data:
                continue
            v = data[k]
            if not isinstance(v, kinds) or (k in _CHOICES and v not in _CHOICES[k]):
                continue  # (a value of the wrong kind: that field isn't saved)
            out[k] = v
        return out, st

    def state(self) -> StoreState:
        return self.read()[1]

    def update(
        self, changes: Mapping[str, Any], *, remove: Sequence[str] = (), add_to: Mapping[str, Mapping[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Save `changes` (fields of :data:`SAVED_FIELDS`) over what was saved, and drop the fields
        in `remove` (they fall back to the .env files and the defaults). `add_to`: entries put into
        a map field ("daily_budgets") only where it has none yet — read and written under the
        store's lock, so a save at the same moment is never lost. Returns what is saved now. A file
        that can't be read is set aside first."""
        bad = [k for k in [*changes, *remove, *(add_to or {})] if k not in SAVED_FIELDS]
        if bad:
            raise ValueError(f"not a saved setting: {', '.join(sorted(bad))}")
        for k, v in changes.items():
            if not isinstance(v, SAVED_FIELDS[k]) or (k in _CHOICES and v not in _CHOICES[k]):
                raise ValueError(f"{k}: not a value it takes")
        for k in add_to or {}:
            if dict not in SAVED_FIELDS[k]:
                raise ValueError(f"{k}: not a map")
        _guard(self.path, "save the brains")
        with locked(self.path.parent):
            saved, st = self.read()
            if st.exists and not st.ok:
                _quarantine(self.path)
                saved = {}
            saved.update(changes)
            for k, entries in (add_to or {}).items():
                merged = dict(saved.get(k) or {})
                for name, v in entries.items():
                    merged.setdefault(name, v)
                saved[k] = merged
            for k in remove:
                saved.pop(k, None)
            data = {"version": VERSION, **{k: saved[k] for k in SAVED_FIELDS if k in saved}}
            atomic_write(self.path, (json.dumps(data, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
            return saved


# ---- as settings sources (neo2.core.config.Settings.settings_customise_sources) ---------------------


def key_aliases() -> dict[str, str]:
    """Provider id → the Settings key it fills, by the key's own name (GEMINI_API_KEY…): the
    catalogue's providers that have a key."""
    from neo2.core.config import Settings
    from neo2.core.providers.catalog import PROVIDERS

    out: dict[str, str] = {}
    for p in PROVIDERS.values():
        if p.key_field:
            alias = Settings.model_fields[p.key_field].validation_alias
            out[p.id] = str(alias)
    return out


class KeyStoreSource(PydanticBaseSettingsSource):
    """The keys saved in NEO, for Settings: by the keys' own names (GEMINI_API_KEY, and
    GEMINI_API_KEYS for Gemini's extra keys), as the environment gives them — and the endpoints'
    keys as `custom_keys`. Below the environment, above the .env files."""

    def __init__(self, settings_cls: type[BaseSettings], path: str | os.PathLike) -> None:
        super().__init__(settings_cls)
        self.store = FileKeyStore(path)

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False  # (__call__ answers for all of them at once)

    def __call__(self) -> dict[str, Any]:
        try:
            keys = self.store.list()
        except Exception as e:  # noqa: BLE001 — what was saved never stops NEO from starting
            print(f"[keys] the key store couldn't be read ({type(e).__name__})")
            return {}
        if not keys:
            return {}
        out: dict[str, Any] = {}
        aliases = key_aliases()
        for pid, k in keys.items():
            if pid.startswith("custom:"):
                out.setdefault("custom_keys", {})[pid] = k.key
            elif pid in aliases:
                out[aliases[pid]] = k.key
                if pid == "gemini" and k.extra:
                    out["GEMINI_API_KEYS"] = ",".join(k.extra)
        return out


class SavedSource(PydanticBaseSettingsSource):
    """The roles and choices saved in NEO (brains.json), for Settings, by field name. Below the
    environment, above the .env files."""

    def __init__(self, settings_cls: type[BaseSettings], path: str | os.PathLike) -> None:
        super().__init__(settings_cls)
        self.saved = SavedConfig(path)

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        try:
            data, _ = self.saved.read()
        except Exception as e:  # noqa: BLE001 — what was saved never stops NEO from starting
            print(f"[keys] the saved brains couldn't be read ({type(e).__name__})")
            return {}
        return {k: v for k, v in data.items() if k in self.settings_cls.model_fields}


# ---- where each key comes from ------------------------------------------------------------------------


def _dotenv(path: Path) -> dict[str, str]:
    from dotenv import dotenv_values

    try:
        if not path.is_file():
            return {}
        return {k: v for k, v in dotenv_values(path).items() if isinstance(v, str)}
    except (OSError, UnicodeDecodeError, ValueError):
        return {}


def env_files() -> list[tuple[str, Path]]:
    """The .env files NEO reads, as `sources` names them: ~/.neo/.env ("dotenv"), and a checkout's
    own .env ("checkout_env", which wins) when running from source."""
    from neo2.core import config

    return [("dotenv" if f == config.USER_ENV else "checkout_env", f) for f in config._ENV_FILES]


def sources(
    *,
    environ: Mapping[str, str] | None = None,
    store: FileKeyStore | None = None,
    files: Sequence[tuple[str, Path]] | None = None,
    custom: Sequence[Any] = (),
) -> dict[str, dict[str, Any]]:
    """Where each provider's key comes from, as R1 orders them — the environment, then saved in NEO,
    then a checkout's .env, then ~/.neo/.env — per provider id: {"set", "source" (env | saved |
    checkout_env | dotenv | none), "last4", "extra" (how many extra keys), "saved" (when, for one
    saved in NEO), "also" (the other places that have one, below the one that wins)}. `custom`: the
    endpoints (their keys are saved by id, or come from their template's variable). No value leaves
    this function but its last 4 characters.

    `files`: (name, path) in the order Settings reads them (the later wins). Defaults: os.environ,
    the store at neo2.core.config.KEY_STORE_PATH, and :func:`env_files` — neither the store nor the
    files while effects are off (a test passes its own)."""
    from neo2.core import config

    env = os.environ if environ is None else environ
    if store is None and config.KEY_STORE_PATH is not None and not refused(config.KEY_STORE_PATH):
        store = FileKeyStore(config.KEY_STORE_PATH)
    if files is None:
        files = [] if (effects.why_off() or effects._under_pytest()) else env_files()
    saved = store.list() if store is not None else {}
    read: list[tuple[str, dict[str, str]]] = []
    for name, f in files:
        if not refused(f):
            read.insert(0, (name, _dotenv(Path(f))))  # (highest first: the later file wins)

    def one(vals: Mapping[str, str], primary: str, extra_name: str | None) -> tuple[str, int] | None:
        key = str(vals.get(primary, "") if primary else "").strip()
        more = [x for x in str((vals.get(extra_name, "") if extra_name else "") or "").split(",") if x.strip()]
        if not key and not more:
            return None
        return key or more[0].strip(), len(more)

    def entry(primary: str, extra_name: str | None, pid: str) -> dict[str, Any]:
        found: list[tuple[str, str, int, str]] = []  # (source, key, extras, when saved)
        if got := one(env, primary, extra_name):
            found.append(("env", *got, ""))
        if pid in saved:
            s = saved[pid]
            found.append(("saved", s.key, len(s.extra), s.saved))
        for name, vals in read:
            if got := one(vals, primary, extra_name):
                found.append((name, *got, ""))
        if not found:
            return {"set": False, "source": "none", "last4": None, "extra": 0, "saved": None, "also": []}
        src, key, n, when = found[0]
        return {
            "set": True,
            "source": src,
            "last4": last4(key),
            "extra": n,
            "saved": when or None,
            "also": [f[0] for f in found[1:]],
        }

    out: dict[str, dict[str, Any]] = {}
    for pid, alias in key_aliases().items():
        out[pid] = entry(alias, "GEMINI_API_KEYS" if pid == "gemini" else None, pid)
    for cp in custom:
        pid = str(getattr(cp, "id", "") or "")
        if pid:
            # (its template's variable at its template's host, else only a key saved under its id)
            out[pid] = entry(getattr(cp, "key_env", None) or "", None, pid)
    return out
