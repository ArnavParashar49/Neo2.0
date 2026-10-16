"""Brains & Keys, in the core: what every client may know of NEO's brains, what the Brains & Keys
window may see and change, and applying a change while NEO runs.

* :func:`summary` — the ``brains`` event and the hello's "brains", for every client: whether NEO has
  a model it can ask (`ready`), each role's brain by its label and whether it can answer, the voice,
  and per provider only whether a key is set. Never a key, never its last 4.
* :func:`detail` — ``brains_detail``, for the Brains & Keys window alone: the catalogue, each key's
  source and last 4, where each role's choice comes from, the endpoints, the voice, the caps.
* :func:`command` — the window's commands (key_test, key_save, key_remove, models_list, brains_set,
  custom_save, custom_remove, brains_restart, brains_get). The websocket routes them here for the
  "keys" client only (neo2.core.server.ws); each answer is fixed words and status codes, never an
  error's own text, and a key handed in is masked in the log from then on (neo2.core.logredact).
  A key whose own prefix names another provider is never sent to this one (:func:`wrong_provider`),
  and an endpoint's saved key never follows its base URL to another host (it is asked for again).
* :func:`apply` — re-read NEO's files (a fresh Settings), copy the brain and key fields onto the live
  settings, forget the brains of each provider whose key, endpoint, daily cap or models' notes
  changed (neo2.core.providers.forget), and tell every client (``brains``, ``problems``). Text
  brains, the camera and new tasks switch at once; a task already running keeps its step's brain.
  Gemini Live, Laya's lite mode and the voice probe read their key when they start: a change there
  says ``restart_needed``. One reading at a time (an older one never lands over a newer one); the
  window's changes run one at a time too, and :func:`reload` (brains_reload) is coalesced.
* :func:`set_roles` — brains_set's rules (Brain calls tools and is set up, Eyes sees, …), then saved
  in brains.json and applied.

What runs on this Mac (NEO's own model at 127.0.0.1:8080, an endpoint the user added here) counts as
set up only while something answers there: :func:`refresh_probes` asks (health.watch, every 30 s),
and :func:`reachable` reads what it last found — nothing here waits on the network otherwise.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import time
import weakref
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from neo2.core import brainref, keyshape, keystore, logger
from neo2.core.providers import catalog

log = logger(__name__)

LOCAL_MODELS_URL = "http://127.0.0.1:8080/v1/models"  # NEO's own model (python -m neo2 --local)
PROBE_TTL_S = 30.0
LABEL_MAX = 60
_ID = re.compile(r"[A-Za-z0-9_.:-]{1,32}")

# The settings a change in NEO's files reaches while NEO runs (apply), by what they are.
KEY_FIELDS = (
    "gemini_api_key", "gemini_api_keys", "groq_api_key", "anthropic_api_key", "cerebras_api_key", "openai_api_key",
    "openrouter_api_key", "custom_keys",
)  # fmt: skip
ROLE_FIELDS = (
    "brain", "fast_brain", "offline_brain", "eyes_brain", "light_brain", "backup_brains", "cloud_eyes",
    "custom_providers", "daily_budgets", "voice", "known_models",
)  # fmt: skip
# A model setting, and the provider whose brains it changes.
MODEL_FIELDS = {
    "claude_model": "anthropic",
    "openai_model": "openai",
    "openrouter_model": "openrouter",
    "gemini_model": "gemini",
    "gemini_fallback_models": "gemini",
    "gemini_lite_model": "gemini",
    "groq_model": "groq",
    "groq_vision_model": "groq",
    "cerebras_model": "cerebras",
    "cerebras_vision_model": "cerebras",
    "cerebras_daily_budget": "cerebras",
}
# The roles a client sees, by the settings field each one is.
ROLES = {"brain": "brain", "quick": "fast_brain", "eyes": "eyes_brain"}
# brains_set's fields (every one optional).
SETTABLE_ROLES = (
    "brain", "fast_brain", "eyes_brain", "backup_brains", "light_brain", "offline_brain", "cloud_eyes", "voice",
    "daily_budgets",
)  # fmt: skip
MAX_BUDGET = 1000.0  # dollars a day: more than that is a typo
KEY_CONSOLE = "python -m neo2 keys set gemini"  # what a terminal user runs (no window without NEO.app)
RELOAD_GAP_S = 1.0  # brains_reload: at most one reading of NEO's files this often (asked more: coalesced)
# The window's commands that change NEO's files: one at a time (each reads, then writes).
_CHANGES = frozenset({"key_save", "key_remove", "brains_set", "custom_save", "custom_remove"})


class _PerLoop:
    """One asyncio object per event loop (a test runs many loops; NEO.app runs one)."""

    def __init__(self, make: Callable[[], Any]) -> None:
        self._make = make
        self._by: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, Any] = weakref.WeakKeyDictionary()

    def get(self) -> Any:
        loop = asyncio.get_running_loop()
        got = self._by.get(loop)
        if got is None:
            got = self._by[loop] = self._make()
        return got


_change_lock = _PerLoop(asyncio.Lock)  # the window's changes, one at a time
_apply_lock = _PerLoop(asyncio.Lock)  # apply(): the files read and put to work, one reading at a time


def _settings(s: Any = None) -> Any:
    if s is not None:
        return s
    from neo2.core.config import settings

    return settings()


# ---- what's on this Mac -----------------------------------------------------------------------------

_probes: dict[str, tuple[bool, float]] = {}  # URL → (something answered, when it was asked)
_now = time.monotonic


def _probe_url(ref: str, s: Any) -> str | None:
    """Where a brain on this Mac answers (its /models), or None for a brain in the cloud."""
    info = catalog.describe(ref, s)
    if not info.valid or not info.on_device:
        return None
    if info.provider == "local":
        return LOCAL_MODELS_URL
    cp = catalog.custom_endpoint(info.provider, s)
    return f"{cp.base_url.rstrip('/')}/models" if cp is not None else None


def reachable(url: str | None) -> bool:
    """Something answered at `url` when it was last asked (never asked: no)."""
    hit = _probes.get(url or "")
    return bool(hit and hit[0])


async def _ask(url: str) -> bool:
    from neo2.core import effects
    from neo2.core.providers.local_server import reachable as answers

    if effects.why_off():
        return False  # (a test, NEO_NO_EFFECTS=1: nothing is asked — a test fakes this step)
    try:
        return await answers(url, timeout=1.0)
    except Exception:  # noqa: BLE001 — nothing there (or not allowed to look): not reachable
        return False


async def probe(url: str) -> bool:
    """Ask `url` now (kept for the next :func:`reachable`)."""
    up = await _ask(url)
    _probes[url] = (up, _now())
    return up


def _named(s: Any) -> list[str]:
    """Every brain the user's roles name (and the chains NEO builds from them)."""
    from neo2.core import providers

    out: list[str] = []
    for purpose in ("agent", "fast", "vision"):
        try:
            c = providers.brain(purpose)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            continue
        out += [n for n in (*c.lead, *c.names, *c.eyes) if isinstance(n, str)]
    out += [str(getattr(s, f, "") or "") for f in ("eyes_brain", "light_brain")]
    return [n for n in dict.fromkeys(out) if n]


async def refresh_probes(s: Any = None, *, force: bool = False) -> dict[str, bool]:
    """Ask each brain on this Mac the roles name whether it answers (one asked within the last 30 s
    isn't asked again unless `force`). Returns URL → answered."""
    s = _settings(s)
    urls = {u for n in _named(s) if (u := _probe_url(n, s))}
    now = _now()
    stale = [u for u in urls if force or u not in _probes or now - _probes[u][1] >= PROBE_TTL_S]
    if stale:
        await asyncio.gather(*(probe(u) for u in stale))
    return {u: reachable(u) for u in urls}


# ---- what can answer ------------------------------------------------------------------------------------


def usable(ref: str, s: Any = None) -> bool:
    """`ref` could answer now: a brain NEO knows, its provider set up (a key; Cerebras with a budget
    left above 0), and on this Mac only while something answers there."""
    s = _settings(s)
    info = catalog.describe(ref, s)
    if not info.valid or not info.known or not info.provider:
        return False
    if not catalog.configured(info.provider, s):
        return False
    if info.on_device:
        return reachable(_probe_url(ref, s))
    if info.provider == "cerebras":
        from neo2.core.providers.cerebras_hooks import enabled

        return enabled(s)
    return True


def _text_chain(s: Any) -> list[str]:
    from neo2.core import providers

    out: list[str] = []
    for purpose in ("agent", "fast"):
        try:
            out += [n for n in providers.brain(purpose).names if isinstance(n, str)]  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            continue
    return list(dict.fromkeys(out))


def ready(s: Any = None) -> bool:
    """NEO has a model it can ask: some brain of its text chains (the roles, Backups, the offline
    brain — and Cerebras where it joins) could answer now."""
    s = _settings(s)
    return any(usable(n, s) for n in _text_chain(s))


def answering(s: Any = None) -> str | None:
    """The brain that answers a task's words now: the first of the agent chain that could."""
    from neo2.core import providers

    s = _settings(s)
    try:
        names = [n for n in providers.brain("agent").names if isinstance(n, str)]
    except Exception:  # noqa: BLE001
        return None
    return next((n for n in names if usable(n, s)), None)


def can_see(s: Any = None) -> bool:
    """Some brain that sees could answer a step with a picture (the user's Eyes, the chains' own,
    the seeing fallbacks NEO may ask)."""
    from neo2.core import providers

    s = _settings(s)
    try:
        c = providers.brain("vision")
    except Exception:  # noqa: BLE001
        return False
    names = [n for n in (*c.lead, *c.names, *c.eyes) if isinstance(n, str)]
    return any(catalog.describe(n, s).sees and usable(n, s) for n in names)


def _label(ref: str, s: Any) -> str:
    r = brainref.parse(ref)
    if not r.valid:
        return "Not a brain NEO knows"
    try:
        return catalog.label_for(r.text, s=s)[:LABEL_MAX]
    except Exception:  # noqa: BLE001
        return keyshape.scrub(r.text)[:LABEL_MAX]


def _role(ref: str, role: str, s: Any) -> dict[str, Any]:
    """One role as every client may see it: its brain's reference and label (a reference NEO can't
    read is never echoed: it could be a key pasted into the wrong line), and whether it can answer."""
    r = brainref.parse(ref)
    info = catalog.describe(ref, s)
    ok = usable(ref, s) and (role != "brain" or info.tools) and (role != "eyes" or info.sees)
    return {"ref": r.text if r.valid else "", "provider": info.provider, "label": _label(ref, s), "ok": bool(ok)}


def key_providers(s: Any = None) -> list[str]:
    """The providers that take a key, then the endpoints the user added: the summary's "providers"."""
    s = _settings(s)
    out = [pid for pid, p in catalog.PROVIDERS.items() if p.key_field]
    return out + [cp.id for cp in getattr(s, "custom_providers", None) or ()]


# ---- the voice, and what a change needs a restart for -------------------------------------------------

_started: str | None = None  # what the running voice started with (a fingerprint, never a key)
voice_now: Callable[[], str | None] | None = None  # the App's: which voice runs now ("live" | "local" | None)


def _voice_print(s: Any) -> str:
    return hashlib.sha256(f"{getattr(s, 'gemini_api_key', '')}\x00{getattr(s, 'voice', '')}".encode()).hexdigest()


def started(s: Any = None) -> None:
    """NEO.app's core started its voice with these settings (neo2.core.app.run)."""
    global _started
    _started = _voice_print(_settings(s))


def restart_needed(s: Any = None) -> bool:
    """Gemini Live, Laya's lite mode and the voice probe read the Gemini key (and the voice setting)
    when they start: changed since, NEO has to restart for them."""
    return _started is not None and _voice_print(_settings(s)) != _started


def voice(s: Any = None) -> str:
    """The voice: "live" (Gemini Live), "local" (on this Mac) or "off"."""
    s = _settings(s)
    if voice_now is not None:
        try:
            v = voice_now()
        except Exception:  # noqa: BLE001
            v = None
        if v in ("live", "local"):
            return v
    mode = getattr(s, "voice", "auto")
    if mode == "off":
        return "off"
    return "live" if mode in ("auto", "live") and getattr(s, "gemini_api_key", "") else "local"


# ---- what every client may know -------------------------------------------------------------------------


def summary(s: Any = None) -> dict[str, Any]:
    """The ``brains`` event's data (and the hello's "brains"): no key, no last 4 — see the module's doc."""
    s = _settings(s)
    look = str(getattr(s, "eyes_brain", "") or "")
    return {
        "ready": ready(s),
        "brain": _role(s.brain, "brain", s),
        "quick": _role(s.fast_brain, "quick", s),
        "eyes": _role(look, "eyes", s) if look else None,
        "voice": voice(s),
        "providers": {pid: catalog.configured(pid, s) for pid in key_providers(s)},
        "restart_needed": restart_needed(s),
    }


async def publish(s: Any = None) -> dict[str, Any]:
    """Tell every client the brains as they are now (the ``brains`` event)."""
    from neo2.core.events import bus

    data = summary(s)
    await bus().publish("brains", **data)
    return data


# ---- where each choice comes from ----------------------------------------------------------------------------


def _saved_path() -> Path | None:
    from neo2.core import config

    p = config.SAVED_PATH
    return None if p is None or keystore.refused(p) else p


def _store_path() -> Path | None:
    from neo2.core import config

    p = config.KEY_STORE_PATH
    return None if p is None or keystore.refused(p) else p


def saved() -> dict[str, Any]:
    """What brains.json holds now (nothing: no file, or none NEO may read here)."""
    p = _saved_path()
    if p is None:
        return {}
    try:
        return keystore.SavedConfig(p).read()[0]
    except Exception:  # noqa: BLE001
        return {}


def _env_name(field: str) -> str:
    return f"NEO_{field.upper()}"


def _in_environ(name: str, environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    want = name.upper()
    return any(k.upper() == want and str(v).strip() != "" for k, v in env.items())


def _env_files() -> list[tuple[str, dict[str, str]]]:
    """(name, values) of the .env files NEO reads, the one that wins first. None while effects are off."""
    from neo2.core import effects

    if effects.why_off() or effects._under_pytest():
        return []
    out: list[tuple[str, dict[str, str]]] = []
    for name, f in keystore.env_files():
        if not keystore.refused(f):
            out.insert(0, (name, keystore._dotenv(Path(f))))
    return out


def setting_source(field: str, *, saved_now: Mapping[str, Any] | None = None) -> str:
    """Where a setting's value comes from, in R1's order: env | saved | checkout_env | dotenv | default."""
    name = _env_name(field)
    if _in_environ(name):
        return "env"
    if field in (saved() if saved_now is None else saved_now):
        return "saved"
    for src, vals in _env_files():
        if any(k.upper() == name and str(v).strip() for k, v in vals.items()):
            return src
    return "default"


def _shown_ref(value: Any) -> dict[str, Any]:
    r = brainref.parse(value)
    if r.valid or r.empty:
        return {"ref": r.text, "valid": True}
    return {"ref": None, "valid": False, "typed": brainref.shown(r.text)}  # (a pasted key is never echoed)


def detail(s: Any = None) -> dict[str, Any]:
    """``brains_detail``: for the Brains & Keys window alone (it carries each key's last 4)."""
    from neo2.core.providers import spend

    s = _settings(s)
    now_saved = saved()
    customs = list(getattr(s, "custom_providers", None) or ())
    srcs = keystore.sources(custom=customs)
    keys = {pid: dict(e) for pid, e in srcs.items()}
    roles: dict[str, Any] = {}
    for f in ("brain", "fast_brain", "eyes_brain", "light_brain", "offline_brain"):
        roles[f] = {**_shown_ref(getattr(s, f, "")), "source": setting_source(f, saved_now=now_saved)}
    b = getattr(s, "backup_brains", None)
    roles["backup_brains"] = {
        "refs": None if b is None else [_shown_ref(x) for x in b],
        "source": setting_source("backup_brains", saved_now=now_saved),
    }
    prefill: dict[str, float] = {}
    for pid, e in srcs.items():
        cp = catalog.custom_endpoint(pid, s) if pid.startswith("custom:") else None
        cap = spend.prefill(pid, e.get("source", ""), on_device=bool(cp and cp.on_device))
        if cap is not None and pid not in (getattr(s, "daily_budgets", None) or {}):
            prefill[pid] = cap
    templates = [
        {"id": t.id, "label": t.label, "base_url": t.base_url, "env": t.env, "key_url": t.key_url, "key_hint": t.key_hint,
         "models": dict(t.models), "tools": t.tools, "vision": t.vision, "short_tool_ids": t.short_tool_ids,
         "check": dict(t.check), "needs_key": t.needs_key}
        for t in catalog.TEMPLATES.values()
    ]  # fmt: skip
    cat = [catalog.as_dict(p) for p in catalog.PROVIDERS.values()]
    cat.append({"id": "custom", "label": "Other endpoint (OpenAI-compatible)", "transport": "openai_compat",
                "cost": "paid", "key_url": "", "key_hint": "", "roles": ["brain", "quick", "eyes"], "env": None,
                "check": {"method": "GET", "path": "/models", "auth": "bearer"}, "templates": templates})  # fmt: skip
    store_st = _store_state()
    return {
        "catalog": cat,
        "keys": keys,
        "roles": roles,
        "custom": [
            {**cp.model_dump(mode="json"), "on_device": cp.on_device, "reachable": reachable(_probe_url(cp.id, s)) if cp.on_device else None,
             "saved": any(isinstance(x, dict) and x.get("id") == cp.id for x in now_saved.get("custom_providers") or [])}
            for cp in customs
        ],  # fmt: skip
        "voice": {"value": s.voice, "source": setting_source("voice", saved_now=now_saved),
                  "live_possible": bool(getattr(s, "gemini_api_key", ""))},  # fmt: skip
        "cloud_eyes": {"value": s.cloud_eyes, "source": setting_source("cloud_eyes", saved_now=now_saved)},
        "daily_budgets": {"value": dict(getattr(s, "daily_budgets", None) or {}),
                          "source": setting_source("daily_budgets", saved_now=now_saved), "prefill": prefill},  # fmt: skip
        "store": store_st,
        "summary": summary(s),
    }


def _store_state() -> dict[str, Any]:
    from neo2.core import config

    p = config.KEY_STORE_PATH
    real = Path.home() / ".neo" / keystore.KEY_FILE
    shown = "~/.neo/api_keys.json" if p is None or Path(p) == real else str(p)  # (by name: nothing looked at)
    if p is None or keystore.refused(p):
        return {"path": shown, "ok": False, "problem": "unavailable"}
    old = keystore.old_copies(p)  # (copies set aside may still hold keys: said, never read)
    try:
        st = keystore.FileKeyStore(p).state()
    except Exception:  # noqa: BLE001
        return {"path": shown, "ok": False, "problem": "unreadable", "old_copies": old}
    return {"path": shown, "ok": st.ok, "problem": st.problem or None, "old_copies": old}


# ---- applying a change while NEO runs ---------------------------------------------------------------------


def _changed_providers(old: Any, new: Any) -> list[str]:
    """The providers whose key, model or endpoint differs between two settings."""
    out: list[str] = []
    for pid, p in catalog.PROVIDERS.items():
        if pid == "gemini":
            if list(getattr(old, "gemini_keys", [])) != list(getattr(new, "gemini_keys", [])):
                out.append(pid)
        elif p.key_field and getattr(old, p.key_field, "") != getattr(new, p.key_field, ""):
            out.append(pid)
    for f, pid in MODEL_FIELDS.items():
        if getattr(old, f, None) != getattr(new, f, None) and pid not in out:
            out.append(pid)
    ok_, nk = getattr(old, "custom_keys", None) or {}, getattr(new, "custom_keys", None) or {}
    oc = {cp.id: cp for cp in getattr(old, "custom_providers", None) or ()}
    nc = {cp.id: cp for cp in getattr(new, "custom_providers", None) or ()}
    for cid in sorted(set(oc) | set(nc) | set(ok_) | set(nk)):
        if oc.get(cid) != nc.get(cid) or ok_.get(cid) != nk.get(cid):
            out.append(cid)
    return out


def _recapped(old: Any, new: Any) -> list[str]:
    """The providers whose daily cap differs (each built brain holds its cap: rebuilt with the new
    one, its rest after a spent budget dropped), or whose models' notes do (OpenRouter's: what its
    list said they see and cost)."""
    ob, nb = dict(getattr(old, "daily_budgets", None) or {}), dict(getattr(new, "daily_budgets", None) or {})
    out = [pid for pid in sorted(set(ob) | set(nb)) if ob.get(pid) != nb.get(pid)]
    on, nn = getattr(old, "known_models", None) or {}, getattr(new, "known_models", None) or {}
    out += [pid for pid in sorted(set(on) | set(nn)) if on.get(pid) != nn.get(pid) and pid not in out]
    return out


async def apply(*, tell: bool = True) -> dict[str, Any]:
    """NEO's files re-read and put to work now (see the module's doc). Returns the new summary. One
    reading at a time: two applies never put an older reading over a newer one."""
    async with _apply_lock.get():
        return await _apply(tell=tell)


async def _apply(*, tell: bool) -> dict[str, Any]:
    from neo2.core import config, health, providers

    live = config.settings()
    fresh = await asyncio.to_thread(config.Settings)
    changed = _changed_providers(live, fresh)
    recapped = [pid for pid in _recapped(live, fresh) if pid not in changed]
    for f in (*KEY_FIELDS, *ROLE_FIELDS, *MODEL_FIELDS):
        if f in type(fresh).model_fields:
            setattr(live, f, getattr(fresh, f))
    for pid in changed:
        providers.forget(pid)
    for pid in recapped:
        providers.forget(pid, keys=False)  # (its cap or its models' notes: the brains, not its key's health)
    if changed or recapped:
        log.info(f"[keys] applied: {', '.join([*changed, *recapped])}")
    try:
        await refresh_probes(live, force=True)
    except Exception as e:  # noqa: BLE001
        log.info(f"[keys] couldn't look for brains on this Mac: {type(e).__name__}")
    data = summary(live)
    if tell:
        from neo2.core.events import bus

        await bus().publish("brains", **data)
        await bus().publish("problems", items=await asyncio.to_thread(health.problems))
    return data


# ---- brains_set's rules ---------------------------------------------------------------------------------------


class Refused(Exception):
    """A change the rules refuse: `error` (invalid | not_configured | no_tools | cant_see), `field`,
    and fixed words for it when there are some (`message`)."""

    def __init__(self, error: str, field: str = "", message: str = "") -> None:
        super().__init__(error)
        self.error, self.field, self.message = error, field, message


async def _set_up(ref: str, field: str, s: Any) -> None:
    """`ref` is set up: its provider's key, an endpoint the user added, something answering here."""
    info = catalog.describe(ref, s)
    if not info.known or not catalog.configured(info.provider, s):
        raise Refused("not_configured", field)
    if info.on_device:
        url = _probe_url(ref, s)
        if not url or not await probe(url):
            raise Refused("not_configured", field)


def _ref(value: Any, field: str, *, empty_ok: bool) -> str:
    if not isinstance(value, str):
        raise Refused("invalid", field)
    r = brainref.parse(value)
    if r.empty and empty_ok:
        return ""
    if not r.valid:
        raise Refused("invalid", field, brainref.KEY_IN_A_FIELD if keyshape.keyish(value) else "")
    return r.text


async def check_roles(changes: Mapping[str, Any], s: Any = None) -> tuple[dict[str, Any], list[str]]:
    """brains_set's rules over `changes` (fields of :data:`SETTABLE_ROLES`): what to save, and the
    warnings. Raises :class:`Refused`. Brain calls tools and is set up; Quick replies is set up; Eyes
    ("" for none) sees and is set up; Backups at most 4 brains NEO knows."""
    s = _settings(s)
    out: dict[str, Any] = {}
    for k in changes:
        if k not in SETTABLE_ROLES:
            raise Refused("invalid", k)
    if "brain" in changes:
        ref = _ref(changes["brain"], "brain", empty_ok=False)
        await _set_up(ref, "brain", s)
        if not catalog.describe(ref, s).tools:
            raise Refused("no_tools", "brain")
        out["brain"] = ref
    if "fast_brain" in changes:
        ref = _ref(changes["fast_brain"], "fast_brain", empty_ok=False)
        await _set_up(ref, "fast_brain", s)
        out["fast_brain"] = ref
    if "eyes_brain" in changes:
        ref = _ref(changes["eyes_brain"], "eyes_brain", empty_ok=True)
        if ref:
            await _set_up(ref, "eyes_brain", s)
            if not catalog.describe(ref, s).sees:
                raise Refused("cant_see", "eyes_brain")
        out["eyes_brain"] = ref
    for f in ("light_brain", "offline_brain"):
        if f in changes:
            out[f] = _ref(changes[f], f, empty_ok=True)
    if "backup_brains" in changes:
        b = changes["backup_brains"]
        if b is None:
            out["backup_brains"] = None
        elif isinstance(b, list) and len(b) <= brainref.MAX_BACKUPS:
            out["backup_brains"] = list(dict.fromkeys(_ref(x, "backup_brains", empty_ok=False) for x in b))
            # (a Backup answers the agent's steps too, which offer tools: one that can't call them never is one)
            if any(not catalog.describe(x, s).tools for x in out["backup_brains"]):
                raise Refused("no_tools", "backup_brains")
        else:
            raise Refused("invalid", "backup_brains")
    for f, choices in (("cloud_eyes", ("auto", "on", "off")), ("voice", ("auto", "live", "local", "off"))):
        if f in changes:
            if changes[f] not in choices:
                raise Refused("invalid", f)
            out[f] = changes[f]
    if "daily_budgets" in changes:
        d = changes["daily_budgets"]
        if not isinstance(d, dict):
            raise Refused("invalid", "daily_budgets")
        got = brainref.budgets(d)
        if len(got) != len(d) or any(v > MAX_BUDGET for v in got.values()):
            raise Refused("invalid", "daily_budgets")
        out["daily_budgets"] = got
    return out, _warnings(out, s)


def _warnings(out: Mapping[str, Any], s: Any) -> list[str]:
    said: list[str] = []
    brain = out.get("brain", s.brain)
    eyes = out.get("eyes_brain", getattr(s, "eyes_brain", ""))
    info = catalog.describe(brain, s)
    if info.valid and not info.sees and not eyes:
        said.append(f"{catalog.provider_label(info.provider, s)} can't see your screen and no Eyes brain is set")
    fast = out.get("fast_brain", s.fast_brain)
    if catalog.describe(fast, s).paid:
        said.append("Quick replies on a paid model costs money on every summary")
    return said


def _notes_for(out: Mapping[str, Any], s: Any) -> dict[str, Any] | None:
    """known_models as it should be saved with these roles: what OpenRouter's list said of each
    model a role (or a Backup) names — what it sees, calls and costs — so a restart still knows.
    None: as saved already."""
    refs = [out.get(f, getattr(s, f, "")) for f in ("brain", "fast_brain", "eyes_brain")]
    refs += list(out.get("backup_brains", getattr(s, "backup_brains", None)) or [])
    notes: dict[str, dict[str, Any]] = {}
    for pid in catalog.LEARNS:
        for r in refs:
            if isinstance(r, str) and catalog.provider_of(r) == pid and (m := catalog.describe(r, s).model):
                if (got := catalog.learned(pid, m, s)) is not None:
                    notes.setdefault(pid, {})[m] = got.as_note()
    return notes if notes != (saved().get("known_models") or {}) else None


async def set_roles(changes: Mapping[str, Any]) -> dict[str, Any]:
    """brains_set: the rules, then brains.json, then :func:`apply`. {"ok", "error"?, "field"?,
    "message"?, "warnings", "restart_needed", "shadowed"}."""
    try:
        out, warnings = await check_roles(changes)
    except Refused as r:
        return {"ok": False, "error": r.error, **({"field": r.field} if r.field else {}), **({"message": r.message} if r.message else {}),
                "warnings": [], "restart_needed": False, "shadowed": []}  # fmt: skip
    if out and (notes := _notes_for(out, _settings())) is not None:
        out = {**out, "known_models": notes}
    if out:
        p = _saved_path()
        if p is None:
            return {"ok": False, "error": "store_failed", "warnings": warnings, "restart_needed": False, "shadowed": []}
        try:
            await asyncio.to_thread(keystore.SavedConfig(p).update, out)
        except Exception as e:  # noqa: BLE001 — said in fixed words
            log.info(f"[keys] brains.json not saved: {type(e).__name__}")
            return {"ok": False, "error": "store_failed", "warnings": warnings, "restart_needed": False, "shadowed": []}
    shadowed = [f for f in out if f != "known_models" and _in_environ(_env_name(f))]
    data = await apply()
    return {"ok": True, "warnings": warnings, "restart_needed": data["restart_needed"], "shadowed": shadowed}


# ---- the window's commands ---------------------------------------------------------------------------------------

RESULTS = {
    "brains_get": "brains_detail",
    "key_test": "key_test_result",
    "key_save": "key_save_result",
    "key_remove": "key_remove_result",
    "models_list": "models_result",
    "brains_set": "brains_set_result",
    "custom_save": "custom_save_result",
    "custom_remove": "custom_remove_result",
    "brains_restart": "brains_restart_result",
}
COMMANDS = frozenset(RESULTS)


def request_id(cmd: Mapping[str, Any]) -> str:
    rid = cmd.get("id")
    return rid if isinstance(rid, str) and _ID.fullmatch(rid) else ""


def provider_id(value: Any) -> str:
    """A provider id as the window or the CLI names it ("claude" and "anthropic", "google" and
    "gemini"); "" for one that isn't."""
    p = str(value or "").strip().lower()
    p = {"claude": "anthropic", "google": "gemini"}.get(p, p)
    try:
        return keystore.check_provider(p)
    except ValueError:
        return ""


def _keyed(pid: str, s: Any) -> bool:
    """`pid` is a provider whose key NEO keeps: the catalogue's with a key, or an endpoint added."""
    if pid.startswith("custom:"):
        return catalog.custom_endpoint(pid, s) is not None
    p = catalog.provider(pid)
    return p is not None and bool(p.key_field)


def _endpoint_arg(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, dict) else None


def _optional_key(value: Any) -> str | None:
    """A key handed in (None when none was): cleaned, and masked in the log from now on."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    k = keystore.clean_key(value)
    from neo2.core import logredact

    logredact.remember(k)
    return k


def _key_state(pid: str, s: Any) -> dict[str, Any]:
    customs = list(getattr(s, "custom_providers", None) or ())
    return keystore.sources(custom=customs).get(pid) or {"set": False, "source": "none", "last4": None}


def _host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def wrong_provider(pid: str, key: str | None, endpoint: Any = None) -> str | None:
    """Fixed words when `key`'s own prefix names another provider than the one it would be sent to
    (an Anthropic key in OpenRouter's card, in an endpoint at some other host): it isn't sent at all.
    An endpoint at that provider's own host (Anthropic's OpenAI-compatible address), one from its
    template (xAI's), or one on this Mac (the user's own proxy) may take it. None: nothing wrong."""
    g = keyshape.guess(key or "") if key else None
    if g is None:
        return None
    if pid.startswith("custom:") or pid == "custom":
        cp = endpoint
        if cp is None:
            return None
        host = _host(cp.base_url)
        own = catalog.provider(g)
        own_host = _host(own.base_url) if own is not None else (brainref.TEMPLATE_KEYS.get(g, ("", ""))[1])
        if cp.template == g or cp.on_device or (own_host and host == own_host):
            return None
        target = cp.label
    else:
        if g == pid:
            return None
        target = catalog.provider_label(pid)
    name = keyshape.LABELS.get(g, g)
    a = "an" if g in keyshape._AN else "a"
    return f"That looks like {a} {name} key, so NEO didn't send it to {target}. Paste it where {name} keys go."


def _endpoint_for(pid: str, endpoint: Any) -> Any:
    """The endpoint a typed key would go to (the form's, else the saved one), or None."""
    from neo2.core.providers import keycheck

    if not (pid == "custom" or pid.startswith("custom:")):
        return None
    try:
        return keycheck._endpoint(pid, _endpoint_arg(endpoint), _settings())
    except Exception:  # noqa: BLE001 — a form that doesn't validate: the check itself says so
        return None


async def _key_test(cmd: Mapping[str, Any]) -> dict[str, Any]:
    from neo2.core.providers import keycheck

    raw = str(cmd.get("provider") or "")
    pid = "custom" if raw.strip().lower() == "custom" else provider_id(raw)
    try:
        key = _optional_key(cmd.get("key"))
    except ValueError:
        return {"provider": pid, "ok": False, "status": "error", "http": None, "message": "That key can't be used as it is.",
                "models": None, "credit": None}  # fmt: skip
    if not pid:
        r = keycheck.KeyCheck(False, "error", None, "NEO doesn't know that provider.")
    elif key is not None and (why := wrong_provider(pid, key, _endpoint_for(pid, cmd.get("endpoint")))):
        r = keycheck.KeyCheck(False, "wrong_provider", None, why)
    elif cmd.get("check") == "tools":
        endpoint = _endpoint_arg(cmd.get("endpoint")) or (_custom_dict(pid) if pid.startswith("custom:") else None)
        if endpoint is None:
            r = keycheck.KeyCheck(False, "error", None, "That endpoint isn't set up in NEO.")
        else:
            try:
                ep = keycheck._endpoint(pid, endpoint, _settings())
            except Exception:  # noqa: BLE001
                ep = None
            r = (
                await keycheck.check_tools(ep, key)
                if ep is not None
                else keycheck.KeyCheck(False, "error", None, "That endpoint isn't complete: check its base URL and model.")
            )
    else:
        r = await keycheck.check(pid, key, _endpoint_arg(cmd.get("endpoint")))
    return {"provider": pid, **r.as_dict()}


def _custom_dict(pid: str) -> dict[str, Any] | None:
    cp = catalog.custom_endpoint(pid, _settings())
    return cp.model_dump(mode="json") if cp is not None else None


_REFUSING = ("bad_key", "no_access", "no_credit")


async def _key_save(cmd: Mapping[str, Any]) -> dict[str, Any]:
    from neo2.core.providers import keycheck, spend

    s = _settings()
    pid = provider_id(cmd.get("provider"))
    base = {"provider": pid, "ok": False, "shadowed_by": None, "restart_needed": False}
    if not pid or not _keyed(pid, s):
        return {**base, "error": "invalid"}
    try:
        key = keystore.clean_key(cmd.get("key"))
        extra_in = cmd.get("extra") or []
        if not isinstance(extra_in, list) or (extra_in and pid != "gemini") or len(extra_in) > 10:
            raise ValueError("extra")
        extra = [keystore.clean_key(x) for x in extra_in if str(x or "").strip()]
    except ValueError:
        return {**base, "error": "invalid"}
    from neo2.core import logredact

    logredact.remember(key, *extra)
    if why := next((w for k in (key, *extra) if (w := wrong_provider(pid, k, _endpoint_for(pid, None)))), None):
        return {**base, "error": "wrong_provider", "message": why}
    if cmd.get("test", True) is not False:
        r = await keycheck.check(pid, key)
        if r.status in _REFUSING:
            return {**base, "error": r.status, "message": r.message}
    path = _store_path()
    if path is None:
        return {**base, "error": "store_failed"}
    try:
        await asyncio.to_thread(keystore.FileKeyStore(path).put, pid, key, extra)
    except Exception as e:  # noqa: BLE001 — said in fixed words
        log.info(f"[keys] key not saved: {type(e).__name__}")
        return {**base, "error": "store_failed"}
    cap = await _prefill_cap(pid, spend)
    data = await apply()
    now = _key_state(pid, _settings())
    return {
        **base,
        "ok": True,
        "key": {"set": True, "source": "saved", "last4": keystore.last4(key)},
        "shadowed_by": "env" if now.get("source") == "env" else None,
        "restart_needed": data["restart_needed"],
        "daily_budget": cap,
    }


def auto_cap(pid: str, s: Any = None) -> float | None:
    """The daily cap a paid key just saved in NEO gets on its own (decision 4: $2 a day), or None:
    its provider has a cap already, it isn't paid, it runs on this Mac, it has no price list NEO
    could count by (an endpoint the user added: its $2 is only suggested, in Spending), or the key
    NEO uses for it isn't the one saved here (an environment variable or a .env line wins, and
    those stay uncapped)."""
    from neo2.core.providers import spend

    s = _settings(s)
    if pid in (getattr(s, "daily_budgets", None) or {}) or not spend.has_prices(pid):
        return None
    cap = spend.prefill(pid, "saved")
    if cap is None or _key_state(pid, s).get("source") != "saved":
        return None
    return cap


async def _prefill_cap(pid: str, spend: Any) -> float | None:
    """A paid key just saved in NEO gets its cap (:func:`auto_cap`), put into brains.json under its
    lock (a cap saved at the same moment is kept). Returns the provider's cap now (None: none)."""
    s = _settings()
    have = (getattr(s, "daily_budgets", None) or {}).get(pid)
    if have is not None:
        return have
    cap = await asyncio.to_thread(auto_cap, pid, s)  # (where the key NEO uses comes from: read from the files)
    p = _saved_path()
    if cap is None or p is None:
        return None
    try:
        now = await asyncio.to_thread(keystore.SavedConfig(p).update, {}, add_to={"daily_budgets": {pid: cap}})
    except Exception as e:  # noqa: BLE001
        log.info(f"[keys] the daily cap wasn't saved: {type(e).__name__}")
        return None
    return (now.get("daily_budgets") or {}).get(pid, cap)


async def _key_remove(cmd: Mapping[str, Any]) -> dict[str, Any]:
    pid = provider_id(cmd.get("provider"))
    if not pid:
        return {"provider": "", "ok": False, "error": "invalid", "still_set": None}
    path = _store_path()
    if path is None:
        return {"provider": pid, "ok": False, "error": "store_failed", "still_set": None}
    try:
        removed = await asyncio.to_thread(keystore.FileKeyStore(path).remove, pid)
    except Exception as e:  # noqa: BLE001
        log.info(f"[keys] key not removed: {type(e).__name__}")
        return {"provider": pid, "ok": False, "error": "store_failed", "still_set": None}
    await apply()
    now = _key_state(pid, _settings())
    still = now.get("source") if now.get("set") and now.get("source") != "saved" else None
    return {"provider": pid, "ok": True, "removed": bool(removed), "still_set": still}


async def _models_list(cmd: Mapping[str, Any]) -> dict[str, Any]:
    from neo2.core.providers import keycheck

    raw = str(cmd.get("provider") or "")
    pid = "custom" if raw.strip().lower() == "custom" else provider_id(raw)
    try:
        key = _optional_key(cmd.get("key"))
    except ValueError:
        return {"provider": pid, "ok": False, "models": [], "defaults": {}, "error": "error", "message": "That key can't be used as it is."}
    if not pid:
        return {"provider": "", "ok": False, "models": [], "defaults": {}, "error": "error", "message": "NEO doesn't know that provider."}
    if key is not None and (why := wrong_provider(pid, key, _endpoint_for(pid, cmd.get("endpoint")))):
        return {"provider": pid, "ok": False, "models": [], "defaults": {}, "error": "wrong_provider", "message": why}
    return await keycheck.models_list(pid, key, _endpoint_arg(cmd.get("endpoint")), refresh=cmd.get("refresh") is True)


def _slug(text: str, taken: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:20] or "endpoint"
    slug, n = base, 1
    while f"{brainref.CUSTOM}{slug}" in taken:
        n += 1
        slug = f"{base[: 20 - len(str(n)) - 1]}-{n}"
    return slug


def _in_use(cid: str, s: Any) -> bool:
    refs = [getattr(s, f, "") for f in ("brain", "fast_brain", "eyes_brain", "light_brain", "offline_brain")]
    refs += list(getattr(s, "backup_brains", None) or [])
    return any(brainref.parse(r).brain == cid for r in refs if isinstance(r, str))


async def _custom_save(cmd: Mapping[str, Any]) -> dict[str, Any]:
    s = _settings()
    ep = _endpoint_arg(cmd.get("endpoint"))
    if ep is None:
        return {"ok": False, "provider": None, "error": "invalid"}
    kept = [x for x in saved().get("custom_providers") or [] if isinstance(x, dict)]
    taken = {str(x.get("id")) for x in kept} | {cp.id for cp in getattr(s, "custom_providers", None) or ()}
    data = dict(ep)
    if not data.get("id"):
        data["id"] = brainref.CUSTOM + _slug(str(data.get("template") or "") if data.get("template") not in (None, "", "other") else str(data.get("label") or ""), taken)
    # Its variable is its template's own, whatever the form says (R1: honoured as an override, and
    # only at the template's host — CustomProvider.key_env).
    data["env"] = brainref.TEMPLATE_KEYS.get(str(data.get("template") or "other"), (None,))[0]
    try:
        cp = brainref.CustomProvider.model_validate(data)
    except Exception:  # noqa: BLE001 — never the form's own text back
        return {"ok": False, "provider": None, "error": "invalid"}
    rest = [x for x in kept if x.get("id") != cp.id]
    if len(rest) >= brainref.MAX_CUSTOM:
        return {"ok": False, "provider": cp.id, "error": "too_many"}
    try:
        key = _optional_key(cmd.get("key"))
    except ValueError:
        return {"ok": False, "provider": cp.id, "error": "invalid"}
    if key is not None and (why := wrong_provider(cp.id, key, cp)):
        return {"ok": False, "provider": cp.id, "error": "wrong_provider", "message": why}
    p = _saved_path()
    if p is None:
        return {"ok": False, "provider": cp.id, "error": "store_failed"}
    # A saved key goes only where it was saved for: an endpoint moved to another scheme, host or port
    # takes its key again (to this Mac: the old one is dropped — never sent to whatever listens here).
    from neo2.core.providers import keycheck

    was = next((c for c in getattr(s, "custom_providers", None) or () if c.id == cp.id), None)
    sp = _store_path()
    drop_key = False
    if was is not None and key is None and not keycheck.same_origin(was.base_url, cp.base_url):
        stored = None
        try:
            stored = keystore.FileKeyStore(sp).get(cp.id) if sp is not None else None
        except Exception:  # noqa: BLE001
            stored = None
        if stored is not None:
            if not cp.on_device:
                return {"ok": False, "provider": cp.id, "error": "needs_key"}
            drop_key = True
    try:
        await asyncio.to_thread(keystore.SavedConfig(p).update, {"custom_providers": [*rest, cp.model_dump(mode="json", exclude_none=True)]})
        if key is not None:
            if sp is None:
                raise RuntimeError("no key store")
            await asyncio.to_thread(keystore.FileKeyStore(sp).put, cp.id, key)
        elif drop_key and sp is not None:
            await asyncio.to_thread(keystore.FileKeyStore(sp).remove, cp.id)
    except Exception as e:  # noqa: BLE001
        log.info(f"[keys] endpoint not saved: {type(e).__name__}")
        return {"ok": False, "provider": cp.id, "error": "store_failed"}
    await apply()
    return {"ok": True, "provider": cp.id}


async def _custom_remove(cmd: Mapping[str, Any]) -> dict[str, Any]:
    s = _settings()
    cid = provider_id(cmd.get("provider"))
    if not cid.startswith(brainref.CUSTOM):
        return {"ok": False, "provider": cid, "error": "invalid"}
    if _in_use(cid, s):
        return {"ok": False, "provider": cid, "error": "in_use"}
    kept = [x for x in saved().get("custom_providers") or [] if isinstance(x, dict)]
    if not any(x.get("id") == cid for x in kept):
        return {"ok": False, "provider": cid, "error": "not_found"}
    p = _saved_path()
    if p is None:
        return {"ok": False, "provider": cid, "error": "store_failed"}
    try:
        await asyncio.to_thread(keystore.SavedConfig(p).update, {"custom_providers": [x for x in kept if x.get("id") != cid]})
        sp = _store_path()
        if sp is not None:
            await asyncio.to_thread(keystore.FileKeyStore(sp).remove, cid)
    except Exception as e:  # noqa: BLE001
        log.info(f"[keys] endpoint not removed: {type(e).__name__}")
        return {"ok": False, "provider": cid, "error": "store_failed"}
    await apply()
    return {"ok": True, "provider": cid}


async def command(cmd: Mapping[str, Any], *, restart: Callable[[], bool] | None = None) -> tuple[str, dict[str, Any]]:
    """One of the window's commands: (the result's type, its data). The data carries the command's
    "id" back. Never raises: anything unexpected is {"ok": false, "error": "failed"}. The ones that
    change NEO's files run one at a time (each reads what it changes, then writes it)."""
    kind = str(cmd.get("cmd") or "")
    if kind in _CHANGES:
        async with _change_lock.get():
            return await _command(kind, cmd, restart)
    return await _command(kind, cmd, restart)


async def _command(kind: str, cmd: Mapping[str, Any], restart: Callable[[], bool] | None) -> tuple[str, dict[str, Any]]:
    rid = request_id(cmd)
    out_type = RESULTS.get(kind, "keys_refused")
    try:
        if kind == "brains_get":
            data = detail()
        elif kind == "key_test":
            data = await _key_test(cmd)
        elif kind == "key_save":
            data = await _key_save(cmd)
        elif kind == "key_remove":
            data = await _key_remove(cmd)
        elif kind == "models_list":
            data = await _models_list(cmd)
        elif kind == "brains_set":
            data = await set_roles({k: cmd[k] for k in SETTABLE_ROLES if k in cmd})
        elif kind == "custom_save":
            data = await _custom_save(cmd)
        elif kind == "custom_remove":
            data = await _custom_remove(cmd)
        elif kind == "brains_restart":
            went = bool(restart()) if restart is not None else False
            data = {"ok": went, **({} if went else {"error": "no_app"})}
        else:
            return "keys_refused", {"id": rid, "cmd": kind[:32], "error": "unknown"}
    except Exception as e:  # noqa: BLE001 — fixed words: an error's text may quote what was sent
        log.info(f"[keys] {kind} failed: {type(e).__name__}")
        data = {"ok": False, "error": "failed"}
    return out_type, {"id": rid, **data}


class _Reloads:
    """brains_reload's bookkeeping, per event loop: asked (a count), and how far applied."""

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.asked = 0
        self.covered = 0
        self.last: dict[str, Any] = {}
        self.at = 0.0


_reloads = _PerLoop(_Reloads)


async def reload() -> dict[str, Any]:
    """brains_reload (the CLI after `keys set`; anyone may ask): NEO's files read again and put to
    work. Coalesced: one reading at a time, at most one every :data:`RELOAD_GAP_S`, and a request
    is answered by the first reading that began after it was made — never by an older one (a key
    the CLI saved just before asking is always read)."""
    r = _reloads.get()
    r.asked += 1
    mine = r.asked
    async with r.lock:
        if r.covered >= mine:
            return r.last
        start = r.asked
        wait = RELOAD_GAP_S - (time.monotonic() - r.at)
        if wait > 0:
            await asyncio.sleep(wait)
            start = r.asked  # (asked while it waited: this reading covers those too)
        r.at = time.monotonic()
        r.last = await apply()
        r.covered = start
        return r.last


def outcome(data: Mapping[str, Any]) -> str:
    """A result as one log word: ok, or its error / status (never a value)."""
    if data.get("ok") is True:
        return "ok"
    for k in ("error", "status"):
        v = data.get(k)
        if isinstance(v, str) and re.fullmatch(r"[a-z_]{1,24}", v):
            return v
    return "not ok" if "ok" in data else "sent"
