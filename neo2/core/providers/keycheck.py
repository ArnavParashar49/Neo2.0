"""Test a key, and list the models it may use ("Brains & Keys": key_test and models_list).

:func:`check` asks the provider whether the key works — no tokens spent, 10 seconds at most, no
retries — and says so in fixed words plus the host's name (never the key, never an error's own text):

* gemini: GET …/v1beta/models with the key in the x-goog-api-key header (never ?key=);
* anthropic: the anthropic SDK's models.list();
* openai, groq, cerebras, xAI, Mistral, Together, Ollama, LM Studio, any other endpoint: the openai
  SDK's models.list() at that base URL (no key on this Mac);
* openrouter: GET /api/v1/key (its /models answers without a key), which also says the credit left;
* deepseek: GET /user/balance, which says the balance.

The result's status: ok, bad_key (401), no_credit (402), no_access (403), not_found (404),
rate_limited (429 — the key works), unreachable, timeout, or error.

:func:`models_list` lists the models a key may use, each with what it can do (sees, calls tools,
context, a tier and a price where known) and the default per role — dropped when the live list
doesn't have it (R8). Capabilities come from the provider's own list where it says them (Anthropic's
capabilities, OpenRouter's supported_parameters and input_modalities, Mistral's capabilities,
DeepSeek's context_window), else from the catalogue's known models, else from the endpoint's own
switches. Embedding, speech, image, moderation, realtime and audio models are left out. A list is
kept for 10 minutes (per provider, endpoint and key — the key itself is never kept, only a hash).
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx

from neo2.core import logger

log = logger(__name__)

TIMEOUT_S = 10.0
CACHE_S = 600.0
MAX_MODELS = 500
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta"
OPENROUTER_URL = "https://openrouter.ai/api/v1"
LOCAL_URL = "http://127.0.0.1:8080/v1"  # NEO's own model on this Mac (python -m neo2 --local)
_HIDDEN = re.compile(
    r"(?i)(?:^|[-_/.:])(?:embed(?:dings?)?|tts|whisper|transcribe|image|moderation|realtime|audio|dall-e)(?:$|[-_/.:\d])"
)
STATUSES = ("ok", "bad_key", "no_access", "no_credit", "rate_limited", "not_found", "unreachable", "timeout", "error")

# Tests hand in an httpx.MockTransport here (nothing leaves the process).
_transport: httpx.AsyncBaseTransport | None = None
_now = time.monotonic
_cache: dict[tuple[str, str, str], tuple[float, dict[str, Any]]] = {}


@dataclass(frozen=True)
class KeyCheck:
    """What a key check found. `models`: how many models the key may use (None: the check doesn't
    say). `credit`: what the provider says is left (OpenRouter: limit, usage, free_tier; DeepSeek:
    available, balance, currency)."""

    ok: bool
    status: str
    http: int | None
    message: str
    models: int | None = None
    credit: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class _Failed(Exception):
    def __init__(self, result: KeyCheck) -> None:
        super().__init__(result.status)
        self.result = result


# ---- words -------------------------------------------------------------------------------------------


def _host(url: str) -> str:
    try:
        return urlsplit(url).hostname or "the server"
    except ValueError:
        return "the server"


def _words(status: str, host: str, code: int | None = None) -> str:
    return {
        "ok": f"The key works ({host}).",
        "bad_key": f"{host} didn't accept this key.",
        "no_credit": f"{host} says the account has no credit left.",
        "no_access": f"This key can't use the API from here — region, project or permission ({host}).",
        "not_found": f"{host} answered 404 — check the base URL.",
        "rate_limited": f"The key works; {host} is busy right now.",
        "unreachable": f"Couldn't reach {host}.",
        "timeout": f"{host} didn't answer within {int(TIMEOUT_S)} seconds.",
    }.get(status, f"{host} answered HTTP {code}." if code else f"Something went wrong asking {host}.")


def _by_code(code: int, host: str, *, body: str = "") -> KeyCheck:
    if 200 <= code < 300:
        return KeyCheck(True, "ok", code, _words("ok", host))
    if code == 400 and re.search(r"API key not valid|API_KEY_INVALID", body):  # (Google's bad key)
        return KeyCheck(False, "bad_key", code, _words("bad_key", host))
    status = {401: "bad_key", 402: "no_credit", 403: "no_access", 404: "not_found", 429: "rate_limited"}.get(code, "error")
    return KeyCheck(status == "rate_limited", status, code, _words(status, host, code))


def _failure(e: BaseException, host: str) -> KeyCheck:
    """A request that got no HTTP answer, or an SDK's error: in fixed words."""
    code = getattr(e, "status_code", None)
    if isinstance(code, int):
        body = getattr(e, "body", None)
        return _by_code(code, host, body=str(body)[:400] if body is not None else "")
    name = type(e).__name__
    if isinstance(e, httpx.TimeoutException) or "Timeout" in name:
        return KeyCheck(False, "timeout", None, _words("timeout", host))
    if isinstance(e, httpx.TransportError) or "Connection" in name or "Connect" in name:
        return KeyCheck(False, "unreachable", None, _words("unreachable", host))
    return KeyCheck(False, "error", None, _words("error", host))


# ---- clients (tests replace these) ---------------------------------------------------------------------


def _http() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=TIMEOUT_S, transport=_transport, follow_redirects=False)


def _anthropic_client(key: str) -> Any:
    import anthropic

    return anthropic.AsyncAnthropic(api_key=key, max_retries=0, timeout=TIMEOUT_S)


def _openai_client(base_url: str, key: str) -> Any:
    import openai

    return openai.AsyncOpenAI(base_url=base_url, api_key=key or "none", max_retries=0, timeout=TIMEOUT_S)


async def _get(url: str, headers: dict[str, str]) -> tuple[int, Any]:
    async with _http() as c:
        r = await c.get(url, headers=headers)
    try:
        body = r.json()
    except ValueError:
        body = None
    if not (200 <= r.status_code < 300):
        raise _Failed(_by_code(r.status_code, _host(url), body=r.text[:400]))
    return r.status_code, body


# ---- which provider, which key ---------------------------------------------------------------------------


def _settings() -> Any:
    try:
        from neo2.core.config import settings

        return settings()
    except Exception:  # noqa: BLE001 — none bound: only what the caller passed
        return None


# The id of a form that isn't saved yet: outside the slug grammar, so no endpoint is ever saved
# under it, and no saved key is ever looked up for it.
FORM_ID = "custom:__form__"


def _endpoint(pid: str, endpoint: Any, s: Any) -> Any:
    """The endpoint being checked: one as the window's form has it (`endpoint`, checked as it would
    be saved — a form not saved yet gets :data:`FORM_ID`), or the saved one `pid` names. None:
    neither."""
    from neo2.core import brainref
    from neo2.core.providers import catalog

    if endpoint is not None:
        if isinstance(endpoint, brainref.CustomProvider):
            return endpoint
        data = dict(endpoint)
        data.pop("env", None)  # (the template's own variable, or none: never the form's say)
        given = data.get("id") or (pid if pid.startswith("custom:") else None)
        if given:
            data["id"] = given
            return brainref.CustomProvider.model_validate(data)
        data["id"] = "custom:form"
        return brainref.CustomProvider.model_validate(data).model_copy(update={"id": FORM_ID})
    return catalog.custom_endpoint(pid, s) if pid.startswith("custom:") else None


def origin(url: str) -> tuple[str, str, int] | None:
    """(scheme, host, port) of a base URL: where a key sent to it goes."""
    try:
        u = urlsplit(url)
        return (u.scheme, (u.hostname or "").lower(), u.port or (443 if u.scheme == "https" else 80))
    except ValueError:
        return None


def same_origin(a: str, b: str) -> bool:
    oa = origin(a)
    return oa is not None and oa == origin(b)


def _effective_key(pid: str, cp: Any, s: Any) -> str:
    """The key NEO would use for `pid` now (its settings: environment, saved, .env)."""
    if s is None:
        return ""
    if cp is not None:
        return str((getattr(s, "custom_keys", None) or {}).get(cp.id, "") or "")
    if pid == "gemini":
        keys = getattr(s, "gemini_keys", None) or []
        return keys[0] if keys else ""
    from neo2.core.providers import catalog

    p = catalog.provider(pid)
    return str(getattr(s, p.key_field, "") or "") if p is not None and p.key_field else ""


def _target(provider: str, key: str | None, endpoint: Any) -> tuple[str, Any, str, str]:
    """(provider id, endpoint or None, base URL, key) — or _Failed when there's nothing to ask."""
    from neo2.core import keystore
    from neo2.core.providers import catalog

    pid = str(provider or "").strip().lower()
    s = _settings()
    cp = None
    if pid == "custom" or pid.startswith("custom:"):
        try:
            cp = _endpoint(pid, endpoint, s)
        except Exception:  # noqa: BLE001 — a form that doesn't validate: said plainly, never echoed
            raise _Failed(KeyCheck(False, "error", None, "That endpoint isn't complete: check its base URL and model.")) from None
        if cp is None:
            raise _Failed(KeyCheck(False, "error", None, "That endpoint isn't set up in NEO."))
        pid, base = cp.id, cp.base_url
    else:
        p = catalog.provider(pid)
        if p is None:
            raise _Failed(KeyCheck(False, "error", None, "NEO doesn't know that provider."))
        base = LOCAL_URL if pid == "local" else p.base_url
    if key is not None and str(key).strip():
        try:
            k = keystore.clean_key(key)
        except ValueError as e:
            raise _Failed(KeyCheck(False, "error", None, f"That key can't be used: {e}.")) from None
    elif cp is not None:
        # An endpoint's saved key goes only where it was saved for: the endpoint as saved, or as the
        # form has it at the same scheme, host and port. A form not saved yet, or one whose base URL
        # moved, has no key but the one typed.
        saved = catalog.custom_endpoint(cp.id, s)
        k = _effective_key(pid, saved, s) if saved is not None and same_origin(saved.base_url, cp.base_url) else ""
    else:
        k = _effective_key(pid, cp, s)
    local = pid == "local" or (cp is not None and cp.on_device)
    if not k and not local:
        raise _Failed(KeyCheck(False, "error", None, f"There's no key to test for {_host(base)}."))
    return pid, cp, base, k


# ---- the checks ------------------------------------------------------------------------------------------


async def _gemini_models(key: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    token = ""
    for _ in range(5):  # (1000 a page: one page today)
        url = f"{GEMINI_URL}/models?pageSize=1000" + (f"&pageToken={token}" if token else "")
        _, body = await _get(url, {"x-goog-api-key": key})
        out += [m for m in (body or {}).get("models", []) if isinstance(m, dict)]
        token = str((body or {}).get("nextPageToken") or "")
        if not token:
            break
    return [m for m in out if "generateContent" in (m.get("supportedGenerationMethods") or [])]


async def _anthropic_models(key: str) -> list[dict[str, Any]]:
    client = _anthropic_client(key)
    out: list[dict[str, Any]] = []
    async for m in client.models.list(limit=1000):
        out.append(m.model_dump(mode="json"))
        if len(out) >= 2 * MAX_MODELS:
            break
    return out


async def _openai_models(base: str, key: str) -> list[dict[str, Any]]:
    page = await _openai_client(base, key).models.list()
    data = getattr(page, "data", None)
    items = data if data is not None else [m async for m in page]
    return [m.model_dump(mode="json") if hasattr(m, "model_dump") else dict(m) for m in items]


async def _openrouter_key(key: str) -> dict[str, Any]:
    _, body = await _get(f"{OPENROUTER_URL}/key", {"Authorization": f"Bearer {key}"})
    d = (body or {}).get("data") or {}
    return {"limit": d.get("limit"), "usage": d.get("usage"), "free_tier": d.get("is_free_tier")}


async def _openrouter_models(key: str) -> list[dict[str, Any]]:
    _, body = await _get(f"{OPENROUTER_URL}/models", {"Authorization": f"Bearer {key}"})
    return [m for m in (body or {}).get("data", []) if isinstance(m, dict)]


async def _deepseek_balance(base: str, key: str) -> tuple[bool, dict[str, Any]]:
    _, body = await _get(f"{base.rstrip('/')}/user/balance", {"Authorization": f"Bearer {key}"})
    b = body or {}
    infos = [i for i in b.get("balance_infos") or [] if isinstance(i, dict)]
    first = infos[0] if infos else {}
    try:
        balance = float(first.get("total_balance"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        balance = None
    return bool(b.get("is_available")), {"available": bool(b.get("is_available")), "balance": balance, "currency": first.get("currency")}


async def _models(pid: str, cp: Any, base: str, key: str) -> list[dict[str, Any]]:
    """The provider's own list of models for this key (its raw entries)."""
    if pid == "gemini":
        return await _gemini_models(key)
    if pid == "anthropic":
        return await _anthropic_models(key)
    if pid == "openrouter":
        return await _openrouter_models(key)
    return await _openai_models(base, key)


async def check(provider: str, key: str | None = None, endpoint: Any = None) -> KeyCheck:
    """Does the key work? `key` None: the key NEO would use now. `endpoint`: an endpoint not saved
    yet (provider "custom"), as the window's form has it. Never raises."""
    host = "the server"
    try:
        pid, cp, base, k = _target(provider, key, endpoint)
        host = _host(base)
        if pid == "openrouter":
            credit = await _openrouter_key(k)
            return KeyCheck(True, "ok", 200, _words("ok", host), None, credit)
        if cp is not None and cp.template == "deepseek":
            available, credit = await _deepseek_balance(base, k)
            if not available:
                return KeyCheck(False, "no_credit", 200, _words("no_credit", host), None, credit)
            return KeyCheck(True, "ok", 200, _words("ok", host), None, credit)
        found = await _models(pid, cp, base, k)
        return KeyCheck(True, "ok", 200, _words("ok", host), len(found))
    except _Failed as f:
        return f.result
    except Exception as e:  # noqa: BLE001 — every failure in fixed words: never str(e), which may quote the key
        return _failure(e, host)


async def check_tools(endpoint: Any, key: str | None = None) -> KeyCheck:
    """The optional "Check tools" of an endpoint ("uses a few tokens"): one tiny chat completion with
    a dummy tool, to see that the model really calls tools. Never raises."""
    host = "the server"
    try:
        pid, cp, base, k = _target(getattr(endpoint, "id", None) or "custom", key, endpoint)
        host = _host(base)
        tool = {"type": "function", "function": {"name": "ping", "description": "Answer by calling this.",
                                                  "parameters": {"type": "object", "properties": {}}}}  # fmt: skip
        resp = await _openai_client(base, k).chat.completions.create(
            model=cp.model,
            messages=[{"role": "user", "content": "Call the ping tool."}],
            tools=[tool],
            **{cp.token_param: 64},
        )
        calls = getattr(resp.choices[0].message, "tool_calls", None) if resp.choices else None
        if calls:
            return KeyCheck(True, "ok", 200, f"{cp.model} calls tools ({host}).")
        return KeyCheck(False, "error", 200, f"{cp.model} answered without calling the tool — turn Tools off for it ({host}).")
    except _Failed as f:
        return f.result
    except Exception as e:  # noqa: BLE001
        return _failure(e, host)


# ---- the model lists ----------------------------------------------------------------------------------------


def _fingerprint(key: str) -> str:
    return hashlib.sha256(("neo-models:" + (key or "")).encode()).hexdigest()[:16]


def _num(v: Any) -> int | None:
    try:
        n = int(v)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _price_per_m(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return round(x * 1_000_000, 6) if x >= 0 else None


def _known(pid: str, mid: str) -> Any:
    from neo2.core.providers import catalog

    return catalog.known_model(pid, mid)


def _entry(pid: str, cp: Any, raw: dict[str, Any]) -> dict[str, Any] | None:
    """One model as the window lists it: {id, label, sees, tools, context, tier?, price?, note?}."""
    mid = str(raw.get("id") or raw.get("name") or "").strip()
    if pid == "gemini":
        mid = mid[len("models/") :] if mid.startswith("models/") else mid
    if not mid or _HIDDEN.search(mid):
        return None
    km = _known(pid, mid)
    label = km.label if km else mid
    sees = km.sees if km else False
    tools = km.tools if km else True
    context = km.context if km else None
    if pid == "anthropic":
        label = str(raw.get("display_name") or label)
        caps = raw.get("capabilities") or {}
        img = caps.get("image_input") if isinstance(caps, dict) else None
        sees = bool(img.get("supported")) if isinstance(img, dict) else True
        context = _num(raw.get("max_input_tokens")) or context
    elif pid == "gemini":
        label = str(raw.get("displayName") or label)
        sees, tools = True, True
        context = _num(raw.get("inputTokenLimit")) or context
    elif pid == "openrouter":
        label = str(raw.get("name") or label)
        arch = raw.get("architecture") or {}
        sees = "image" in (arch.get("input_modalities") or [])
        tools = "tools" in (raw.get("supported_parameters") or [])
        context = _num(raw.get("context_length")) or context
    elif pid == "openai" and km is None:
        sees = mid.startswith("gpt-")
    elif cp is not None:
        caps = raw.get("capabilities")
        if isinstance(caps, dict) and ("function_calling" in caps or "vision" in caps):  # (Mistral's)
            tools = bool(caps.get("function_calling", tools))
            sees = bool(caps.get("vision", False))
        else:
            tools, sees = cp.tools, cp.vision
        context = _num(raw.get("max_context_length")) or _num(raw.get("context_window")) or _num(raw.get("context_length")) or cp.context
    out: dict[str, Any] = {"id": mid, "label": label, "sees": bool(sees), "tools": bool(tools), "context": context}
    if km is not None and km.tier in ("recommended", "cheap", "premium"):
        out["tier"] = km.tier
    if km is not None and km.price:
        out["price"] = list(km.price)
    elif pid == "openrouter":
        pr = raw.get("pricing") or {}
        pin, pout = _price_per_m(pr.get("prompt")), _price_per_m(pr.get("completion"))
        if pin is not None and pout is not None:
            out["price"] = [pin, pout]
    if km is not None and km.eyes_only:
        out["eyes_only"] = True
    if pid == "anthropic":
        from neo2.core.providers.claude import profile

        if not profile(mid).known:
            out["note"] = "may not work with NEO"
    return out


def _ref(pid: str, model: str) -> str:
    return f"{pid}@{model}"


def _defaults(pid: str, cp: Any, ids: set[str] | None) -> dict[str, str]:
    """The pre-selected brain per role; one whose model the live list doesn't have is dropped (R8)."""
    from neo2.core.providers import catalog

    if cp is not None:
        t = catalog.TEMPLATES.get(cp.template)
        models = dict(t.models) if t is not None else {}
        models.setdefault("brain", cp.model)
        roles = {"brain": models.get("brain", cp.model), "quick": models.get("quick", cp.model)}
        if cp.vision:
            roles["eyes"] = roles["brain"]
        if not cp.tools:
            roles.pop("brain", None)
        refs = {r: _ref(cp.id, m) for r, m in roles.items() if m}
    else:
        p = catalog.provider(pid)
        refs = dict(p.defaults) if p is not None else {}
    if ids is None:
        return refs
    s = _settings()
    return {r: ref for r, ref in refs.items() if catalog.describe(ref, s).model in ids}


def _fallback_models(pid: str, cp: Any) -> list[dict[str, Any]]:
    """The catalogue's own list, for when the live one can't be had ("type a model id" too)."""
    from neo2.core.providers import catalog

    p = catalog.provider(pid)
    out = [_entry(pid, cp, {"id": m.id}) for m in (p.models if p else ())]
    if cp is not None:
        out = [_entry(pid, cp, {"id": cp.model})]
    return [e for e in out if e]


async def models_list(provider: str, key: str | None = None, endpoint: Any = None, *, refresh: bool = False) -> dict[str, Any]:
    """The models `provider`'s key may use: {"provider", "ok", "models": [...], "defaults": {role:
    ref}, "error": status|None, "message"}. Kept for 10 minutes unless `refresh`. Never raises."""
    host = "the server"
    try:
        pid, cp, base, k = _target(provider, key, endpoint)
    except _Failed as f:
        return {"provider": str(provider or ""), "ok": False, "models": [], "defaults": {}, "error": f.result.status,
                "message": f.result.message}  # fmt: skip
    host = _host(base)
    ck = (pid, base, _fingerprint(k))
    hit = _cache.get(ck)
    if hit is not None and not refresh and _now() - hit[0] < CACHE_S:
        return hit[1]
    try:
        raw = await _models(pid, cp, base, k)
    except Exception as e:  # noqa: BLE001 — in fixed words; the catalogue's list stands in
        r = e.result if isinstance(e, _Failed) else _failure(e, host)
        return {"provider": pid, "ok": False, "models": _fallback_models(pid, cp), "defaults": _defaults(pid, cp, None),
                "error": r.status, "message": r.message}  # fmt: skip
    seen: set[str] = set()
    entries: list[dict[str, Any]] = []
    for item in raw:
        e = _entry(pid, cp, item)
        if e is not None and e["id"] not in seen:
            seen.add(e["id"])
            entries.append(e)
    entries.sort(key=lambda e: ({"recommended": 0, "cheap": 1, "premium": 2}.get(e.get("tier", ""), 3)))
    if pid == "openrouter":  # (what its list says is what the chains and the cap then read)
        from neo2.core.providers import catalog

        changed = [e["id"] for e in entries[:MAX_MODELS] if catalog.learn(pid, e["id"], sees=e["sees"], tools=e["tools"], price=e.get("price"))]
        if changed:
            from neo2.core import providers

            providers.rebuild(pid, changed)  # (a brain built on what was known before is built again)
    ids = {e["id"] for e in entries}
    out = {
        "provider": pid,
        "ok": True,
        "models": entries[:MAX_MODELS],
        "defaults": _defaults(pid, cp, ids),
        "error": None,
        "message": _words("ok", host),
    }
    _cache[ck] = (_now(), out)
    return out


def forget(provider: str | None = None) -> None:
    """Drop the kept lists (of `provider`; None: every one) — a key replaced."""
    for ck in [c for c in _cache if provider is None or c[0] == provider]:
        _cache.pop(ck, None)
