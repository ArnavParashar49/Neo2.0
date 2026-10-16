"""Brain references: which brain a role uses, as one string — ``<brain>[@<model>]``.

``brain`` is one of :data:`BRAINS` (the names NEO had before, plus ``openai`` and ``openrouter``) or
``custom:<slug>``, an OpenAI-compatible endpoint the user added (:class:`CustomProvider`). ``@model``
picks the model; without it the brain's own setting (or the endpoint's model) is used. "@" because
OpenRouter's and Ollama's model ids hold "/" and ":" ("openrouter@deepseek/deepseek-v4-pro",
"custom:ollama@qwen3:14b"). "anthropic" is "claude" and "google" is "gemini".

Each distinct reference is its own brain (its own cache entry, demotion and cooldowns, as
gemini_lite is today); a bare legacy name is exactly the brain it always was.

A reference that can't be read is never swapped for another provider: it stays what was typed (a
placeholder, :attr:`Ref.valid` False) — the chains skip it like a brain with no key, and the health
list says so. Nothing here reads settings, files or the network.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from neo2.core.keyshape import keyish

# The brains NEO knew before brain references (config.BrainName): bare, each resolves exactly as then.
LEGACY: tuple[str, ...] = (
    "gemini", "gemini_lite", "groq", "local", "claude", "groq_vision", "cerebras", "cerebras_vision",
)  # fmt: skip
BRAINS: tuple[str, ...] = (*LEGACY, "openai", "openrouter")
ALIASES = {"anthropic": "claude", "google": "gemini"}
CUSTOM = "custom:"
_SLUG = re.compile(r"[a-z0-9-]{1,24}")
_MODEL = re.compile(r"[A-Za-z0-9._:/+-]{1,160}")
# What a placeholder keeps of what was typed. Longer than any valid reference can be, so a cut one
# can never turn into a valid one.
MAX_TEXT = 200
MAX_BACKUPS = 4
MAX_CUSTOM = 10
TEMPLATES = ("xai", "deepseek", "mistral", "together", "ollama", "lmstudio", "other")
LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1"})
# The one environment variable an endpoint may take its key from — its template's own — and the only
# host that key goes to (neo2.core.providers.catalog.TEMPLATES says the same: a test holds them
# together). XAI_API_KEY is for api.x.ai: an endpoint moved to another host, or one from another
# template, gets no key from the environment. None of these is a key NEO keeps itself
# (ANTHROPIC_API_KEY…), so a saved key can never be read into an endpoint by its name.
TEMPLATE_KEYS: dict[str, tuple[str, str]] = {
    "xai": ("XAI_API_KEY", "api.x.ai"),
    "deepseek": ("DEEPSEEK_API_KEY", "api.deepseek.com"),
    "mistral": ("MISTRAL_API_KEY", "api.mistral.ai"),
    "together": ("TOGETHER_API_KEY", "api.together.xyz"),
}
KEY_IN_A_FIELD = "That looks like an API key — paste it under Providers, not as a brain or a name."


@dataclass(frozen=True)
class Ref:
    """A parsed reference. `text`: its normal form ("claude@claude-haiku-5-5"), or for one that
    can't be read, what was typed (cut, control characters as "?"). "" is no brain at all."""

    text: str
    brain: str = ""
    model: str | None = None
    valid: bool = False

    @property
    def empty(self) -> bool:
        return self.text == ""

    @property
    def legacy(self) -> bool:
        """A bare name NEO had before references: its own code path, exactly as then."""
        return self.valid and self.model is None and self.brain in LEGACY

    @property
    def custom(self) -> bool:
        return self.valid and self.brain.startswith(CUSTOM)

    @property
    def slug(self) -> str:
        return self.brain[len(CUSTOM) :] if self.custom else ""


def _placeholder(s: str) -> Ref:
    shown = "".join(c if c.isprintable() else "?" for c in s)[:MAX_TEXT]
    return Ref(shown)


def parse(value: Any) -> Ref:
    """`value` as a reference; never raises."""
    if value is None:
        return Ref("")
    s = str(value).strip()
    if not s:
        return Ref("")
    if len(s) > MAX_TEXT or not s.isprintable():
        return _placeholder(s)
    head, at, model = s.partition("@")
    head = head.strip().lower()
    head = ALIASES.get(head, head)
    m: str | None = None
    if at:
        m = model.strip()
        if not valid_model(m):  # (a key pasted where a model id goes is never one: keyshape.keyish)
            return _placeholder(s)
    if head.startswith(CUSTOM):
        if not _SLUG.fullmatch(head[len(CUSTOM) :]):
            return _placeholder(s)
    elif head not in BRAINS:
        return _placeholder(s)
    return Ref(head + (f"@{m}" if m is not None else ""), head, m, True)


_SHOWN = re.compile(r"[A-Za-z0-9 _.:@/+-]{1,32}")
_KEYISH = re.compile(r"(?i)^(sk-|aiza|gsk_|csk-|xai-)|[A-Za-z0-9_-]{24,}")


def shown(value: str) -> str | None:
    """What was typed, when it's safe to say it back (the health list, a log line): short, plain,
    and nothing like a key — a key pasted into the wrong line must never reach every client's hello
    or core.log. None: it isn't."""
    v = str(value or "").strip()
    return v if _SHOWN.fullmatch(v) and not _KEYISH.search(v) else None


def normalize(value: Any) -> str:
    """The settings' tolerant validator: a reference's normal form, or the placeholder."""
    return parse(value).text


def valid_model(model: str) -> bool:
    """A model id NEO takes: its characters, and nothing shaped like an API key in it."""
    return bool(_MODEL.fullmatch(model or "")) and not keyish(model)


def backups(value: Any) -> list[str] | None:
    """NEO_BACKUP_BRAINS / brains.json's backup_brains: None (or "", "auto") is Automatic — today's
    chains; "none" or [] is no backups; a list (JSON, or "claude, groq") is at most
    :data:`MAX_BACKUPS` references (a placeholder among them is kept: the chain skips it)."""
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        if s.lower() in ("", "auto", "automatic", "null", "default"):
            return None
        if s.lower() in ("none", "off", "[]"):
            return []
        if s.startswith("["):
            try:
                value = json.loads(s)
            except ValueError:
                return None
        else:
            value = s.split(",")
    if not isinstance(value, list | tuple):
        return None
    out: list[str] = []
    for v in value:
        t = normalize(v) if isinstance(v, str) else ""
        if t and t not in out:
            out.append(t)
    return out[:MAX_BACKUPS]


_PROVIDER_KEY = re.compile(r"[a-z][a-z0-9_]{1,23}|custom:[a-z0-9-]{1,24}")


def budgets(value: Any) -> dict[str, float]:
    """daily_budgets: dollars a day per provider id, from a dict, JSON, or "anthropic=2, openai=1".
    A brain name stands for its provider ("claude" is "anthropic"). An entry that can't be read is
    dropped (no cap for it, as before), never a reason not to start."""
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("{"):
            try:
                value = json.loads(s)
            except ValueError:
                return {}
        else:
            value = dict(p.split("=", 1) for p in s.replace(";", ",").split(",") if "=" in p)
    if not isinstance(value, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in value.items():
        key = str(k).strip().lower()
        key = {"claude": "anthropic", "google": "gemini"}.get(key, key)
        if not _PROVIDER_KEY.fullmatch(key):
            continue
        try:
            x = float(str(v).strip().lstrip("$"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and x >= 0:
            out[key] = x
    return out


MAX_NOTES = 50  # models a provider's notes keep (the roles use a handful)


def model_notes(value: Any) -> dict[str, dict[str, dict[str, Any]]]:
    """known_models (brains.json): what a provider's live list said of the models the roles use —
    {"openrouter": {model id: {"sees": bool, "tools": bool, "price": [in, out] | None}}}. Tolerant:
    an entry that can't be read is left out, never a reason not to start."""
    if isinstance(value, str):
        try:
            value = json.loads(value) if value.strip() else {}
        except ValueError:
            return {}
    if not isinstance(value, dict):
        return {}
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for pid, models in value.items():
        if pid != "openrouter" or not isinstance(models, dict):
            continue
        kept: dict[str, dict[str, Any]] = {}
        for mid, note in list(models.items())[:MAX_NOTES]:
            if not isinstance(mid, str) or not valid_model(mid) or not isinstance(note, dict):
                continue
            price = note.get("price")
            pr: list[float] | None = None
            if isinstance(price, list | tuple) and len(price) == 2:
                try:
                    a, b = float(price[0]), float(price[1])
                    pr = [a, b] if all(math.isfinite(x) and x >= 0 for x in (a, b)) else None
                except (TypeError, ValueError):
                    pr = None
            kept[mid] = {"sees": bool(note.get("sees")), "tools": bool(note.get("tools", True)), "price": pr}
        if kept:
            out[pid] = kept
    return out


def loopback(base_url: str) -> bool:
    """The endpoint is on this Mac (localhost, 127.0.0.1 or [::1])."""
    try:
        return (urlsplit(base_url).hostname or "") in LOOPBACK
    except ValueError:
        return False


def check_base_url(url: str) -> str:
    """An endpoint's base URL as kept: https (http only on this Mac), no user or password, no query
    or fragment, at most 300 characters. Raises ValueError (without echoing it: a URL with a
    password in it is a secret)."""
    s = str(url or "").strip()
    if not s or len(s) > 300 or not s.isprintable() or any(c.isspace() for c in s):
        raise ValueError("base_url must be a URL of at most 300 characters")
    try:
        parts = urlsplit(s)
        port = parts.port  # (a port that isn't a number raises)
    except ValueError:
        raise ValueError("base_url isn't a URL") from None
    host = parts.hostname or ""
    if parts.scheme not in ("https", "http") or not host:
        raise ValueError("base_url must start with https://")
    if parts.scheme == "http" and host not in LOOPBACK:
        raise ValueError("base_url must be https:// (http:// only on this Mac)")
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise ValueError("base_url can't hold a user name or password")
    if parts.query or parts.fragment or "?" in s or "#" in s:
        raise ValueError("base_url can't have a query")
    del port
    return s.rstrip("/")


class CustomProvider(BaseModel):
    """An OpenAI-compatible endpoint the user added (brains.json, or NEO_CUSTOM_PROVIDERS as JSON):
    its brain is ``custom:<slug>``. Its key lives in the key store (or its template's environment
    variable, `env` — only at that template's host: :attr:`key_env`), never here. A name or a model id
    shaped like a key is refused: every client hears them."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    id: str
    label: str = Field(min_length=1, max_length=40)
    template: Literal["xai", "deepseek", "mistral", "together", "ollama", "lmstudio", "other"] = "other"
    base_url: str
    model: str
    tools: bool = True
    vision: bool = False
    context: int = Field(default=32768, ge=4096, le=2_000_000)
    reasoning: bool = False  # send reasoning_effort
    token_param: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    short_tool_ids: bool = False
    # The environment variable its key may come from: only its template's own (XAI_API_KEY for xai,
    # TEMPLATE_KEYS), honoured only at that template's host (:attr:`key_env`). None: none.
    env: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _templates_variable(cls, data: Any) -> Any:
        """`env` names its template's own variable or nothing: never another provider's key
        (ANTHROPIC_API_KEY), nor any other secret the environment holds (GITHUB_TOKEN)."""
        if isinstance(data, dict) and data.get("env") not in (None, ""):
            own = TEMPLATE_KEYS.get(str(data.get("template") or "other"), ("",))[0]
            if str(data.get("env")).strip() != own:
                raise ValueError("env can only be the template's own variable (XAI_API_KEY for xAI…)")
        return data

    @field_validator("id", mode="before")
    @classmethod
    def _id(cls, v: Any) -> str:
        r = parse(v)
        if not r.custom or r.model is not None:
            raise ValueError("id must be custom:<slug> (a-z, 0-9 and -, at most 24)")
        return r.text

    @field_validator("label", mode="before")
    @classmethod
    def _label(cls, v: Any) -> str:
        s = " ".join(str(v or "").split())
        if not s.isprintable():
            raise ValueError("label must be plain text")
        if keyish(s):  # (every client hears an endpoint's name: a key pasted there never is one)
            raise ValueError("label can't be an API key")
        return s

    @field_validator("base_url", mode="before")
    @classmethod
    def _base_url(cls, v: Any) -> str:
        return check_base_url(str(v or ""))

    @field_validator("model", mode="before")
    @classmethod
    def _model(cls, v: Any) -> str:
        s = str(v or "").strip()
        if not valid_model(s):
            raise ValueError("model must be a model id")
        return s

    @field_validator("env", mode="before")
    @classmethod
    def _env(cls, v: Any) -> str | None:
        if v in (None, ""):
            return None
        s = str(v).strip()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{1,63}", s) or s.startswith("NEO_"):
            raise ValueError("env must be an environment variable's name (XAI_API_KEY)")
        return s

    @property
    def slug(self) -> str:
        return self.id[len(CUSTOM) :]

    @property
    def on_device(self) -> bool:
        return loopback(self.base_url)

    @property
    def key_env(self) -> str | None:
        """The environment variable its key comes from now: its template's own (`env`), and only
        while its base URL is that template's host over https — XAI_API_KEY never goes anywhere
        but api.x.ai."""
        if not self.env:
            return None
        own = TEMPLATE_KEYS.get(self.template)
        try:
            u = urlsplit(self.base_url)
        except ValueError:
            return None
        if own is None or own[0] != self.env or u.scheme != "https" or (u.hostname or "") != own[1] or u.port not in (None, 443):
            return None
        return self.env


def custom_list(value: Any) -> list[CustomProvider]:
    """custom_providers, tolerant: a list (or its JSON) of endpoints; one that isn't valid is left out
    (its brain then "isn't set up", which is reported), never a reason not to start. The first of
    an id wins; at most :data:`MAX_CUSTOM`."""
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            value = json.loads(s)
        except ValueError:
            return []
    if not isinstance(value, list | tuple):
        return []
    out: list[CustomProvider] = []
    for item in value:
        try:
            cp = item if isinstance(item, CustomProvider) else CustomProvider.model_validate(item)
        except (ValidationError, ValueError, TypeError):
            continue
        if all(cp.id != o.id for o in out):
            out.append(cp)
    return out[:MAX_CUSTOM]
