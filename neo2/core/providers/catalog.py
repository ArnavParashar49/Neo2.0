"""The providers NEO can use, in one place ("Brains & Keys").

Each :class:`ProviderInfo` says how NEO talks to the provider (its transport), which Settings field
and environment variable hold its key, where to get a key and what one looks like, what it costs,
how a key is checked, its default brain per role (Brain, Quick replies, Eyes) and the brain names
it backs. The OpenAI-compatible endpoints a user adds (``custom:<slug>``) start from a
:data:`TEMPLATES` entry. Default models are pre-selections, not hard-coding: the window's model
pickers read each provider's live list, and drop a default that isn't on it.

Nothing here reads a key's value, a file or the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Cost = Literal["free", "free_tier", "paid", "this_mac"]
Transport = Literal["gemini", "anthropic", "openai_responses", "openai_compat", "cerebras", "local"]
Role = Literal["brain", "quick", "eyes", "camera", "voice"]


@dataclass(frozen=True)
class ModelInfo:
    """What is known of one model before its provider's live list says more."""

    id: str
    label: str
    sees: bool = True
    tools: bool = True
    context: int | None = None
    tier: Literal["recommended", "cheap", "premium", "balanced", "previous", ""] = ""
    price: tuple[float, float] | None = None  # $ per million tokens, in / out
    eyes_only: bool = False
    preview: bool = False


@dataclass(frozen=True)
class ProviderInfo:
    id: str
    label: str
    transport: Transport
    key_field: str | None  # the Settings field its key fills (None: no key, on this Mac)
    env: str | None  # that key's environment variable
    key_url: str
    key_hint: str  # what a key starts with (a soft hint, never a reason to refuse one)
    cost: Cost
    base_url: str
    check: dict[str, str]  # how a key is checked (no tokens spent)
    defaults: dict[str, str]  # role → brain reference
    brains: tuple[str, ...]  # the brain names it backs
    roles: tuple[Role, ...]
    extra_env: str | None = None  # (Gemini: GEMINI_API_KEYS, more keys, comma-separated)
    models: tuple[ModelInfo, ...] = ()
    key_hints: tuple[str, ...] = ()  # every prefix its keys may start with (key_hint first)


PROVIDERS: dict[str, ProviderInfo] = {
    p.id: p
    for p in (
        ProviderInfo(
            id="gemini",
            label="Google Gemini",
            transport="gemini",
            key_field="gemini_api_key",
            env="GEMINI_API_KEY",
            extra_env="GEMINI_API_KEYS",
            key_url="https://aistudio.google.com/apikey",
            key_hint="AIza",
            cost="free_tier",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            check={"method": "GET", "path": "/models?pageSize=1000", "auth": "x-goog-api-key"},
            # Brain: Flash with its own model walk (3.8 → 3.7 → 3.5); Quick: Flash-Lite.
            defaults={"brain": "gemini", "quick": "gemini_lite", "eyes": "gemini"},
            brains=("gemini", "gemini_lite"),
            roles=("brain", "quick", "eyes", "camera", "voice"),
            models=(
                ModelInfo("gemini-3.8-flash", "Gemini 3.8 Flash", tier="recommended"),
                ModelInfo("gemini-3.5-flash-lite", "Gemini 3.5 Flash-Lite", tier="cheap"),
                ModelInfo("gemini-3.7-flash", "Gemini 3.7 Flash"),
                ModelInfo("gemini-3.5-flash", "Gemini 3.5 Flash"),
            ),
        ),
        ProviderInfo(
            id="anthropic",
            label="Anthropic (Claude)",
            transport="anthropic",
            key_field="anthropic_api_key",
            env="ANTHROPIC_API_KEY",
            key_url="https://console.anthropic.com/settings/keys",
            key_hint="sk-ant-",
            cost="paid",
            base_url="https://api.anthropic.com",
            check={"sdk": "anthropic", "call": "models.list"},
            defaults={
                "brain": "claude@claude-opus-5-5",
                "quick": "claude@claude-haiku-5-5",
                "eyes": "claude@claude-opus-5-5",
            },
            brains=("claude",),
            roles=("brain", "quick", "eyes", "camera"),
            models=(
                ModelInfo("claude-opus-5-5", "Claude Opus 5.5", context=1_000_000, tier="recommended", price=(4.0, 20.0)),
                ModelInfo("claude-haiku-5-5", "Claude Haiku 5.5", context=1_000_000, tier="cheap", price=(0.10, 0.50)),
                ModelInfo("claude-sonnet-5-5", "Claude Sonnet 5.5", context=1_000_000, tier="balanced", price=(2.0, 10.0)),
                # (needs 30-day data retention on the account)
                ModelInfo("claude-fable-5-1", "Claude Fable 5.1", context=1_000_000, tier="premium", price=(10.0, 50.0)),
                ModelInfo("claude-opus-5", "Claude Opus 5", context=1_000_000, tier="previous", price=(5.0, 25.0)),
            ),
        ),
        ProviderInfo(
            id="openai",
            label="OpenAI",
            transport="openai_responses",
            key_field="openai_api_key",
            env="OPENAI_API_KEY",
            key_url="https://platform.openai.com/api-keys",
            key_hint="sk-proj-",
            key_hints=("sk-proj-", "sk-"),
            cost="paid",
            base_url="https://api.openai.com/v1",
            check={"sdk": "openai", "call": "models.list"},
            defaults={"brain": "openai@gpt-6.1-sol", "quick": "openai@gpt-6-luna", "eyes": "openai@gpt-6.1-sol"},
            brains=("openai",),
            roles=("brain", "quick", "eyes", "camera"),
            models=(
                ModelInfo("gpt-6.1-sol", "GPT-6.1 Sol", context=400_000, tier="recommended", price=(2.0, 10.0)),
                ModelInfo("gpt-6-luna", "GPT-6 Luna", context=400_000, tier="cheap", price=(0.10, 0.50)),
                ModelInfo("gpt-6-astra", "GPT-6 Astra", context=400_000, tier="premium", price=(10.0, 50.0)),
            ),
        ),
        ProviderInfo(
            id="groq",
            label="Groq",
            transport="openai_compat",
            key_field="groq_api_key",
            env="GROQ_API_KEY",
            key_url="https://console.groq.com/keys",
            key_hint="gsk_",
            cost="free",
            base_url="https://api.groq.com/openai/v1",
            check={"method": "GET", "path": "/models", "auth": "bearer"},
            defaults={"brain": "groq", "quick": "groq", "eyes": "groq_vision"},
            brains=("groq", "groq_vision"),
            roles=("brain", "quick", "eyes"),
            models=(
                ModelInfo("openai/gpt-oss-120b", "GPT-OSS 120B", sees=False, tier="recommended"),
                ModelInfo("qwen/qwen3.8-27b", "Qwen 3.8 27B", eyes_only=True, preview=True),
            ),
        ),
        ProviderInfo(
            id="cerebras",
            label="Cerebras",
            transport="cerebras",
            key_field="cerebras_api_key",
            env="CEREBRAS_API_KEY",
            key_url="https://cloud.cerebras.ai",
            key_hint="csk-",
            cost="paid",  # (capped by NEO_CEREBRAS_DAILY_BUDGET)
            base_url="https://api.cerebras.ai/v1",
            check={"method": "GET", "path": "/models", "auth": "bearer"},
            defaults={"brain": "cerebras", "quick": "cerebras", "eyes": "cerebras_vision"},
            brains=("cerebras", "cerebras_vision"),
            roles=("brain", "quick", "eyes", "camera"),
            models=(
                ModelInfo("gpt-oss-120b", "GPT-OSS 120B", sees=False, tier="recommended"),
                ModelInfo("qwen-3.8-27b", "Qwen 3.8 27B"),
            ),
        ),
        ProviderInfo(
            id="openrouter",
            label="OpenRouter",
            transport="openai_compat",
            key_field="openrouter_api_key",
            env="OPENROUTER_API_KEY",
            key_url="https://openrouter.ai/keys",
            key_hint="sk-or-",
            cost="paid",  # (pay as you go; a few models are free)
            base_url="https://openrouter.ai/api/v1",
            # /models answers without a key: /key is what proves one works (and says its credit)
            check={"method": "GET", "path": "/key", "auth": "bearer"},
            defaults={},  # the user picks a model (its ids are vendor-prefixed)
            brains=("openrouter",),
            roles=("brain", "quick", "eyes"),
        ),
        ProviderInfo(
            id="local",
            label="NEO's model on this Mac (Gemma)",
            transport="local",
            key_field=None,
            env=None,
            key_url="",
            key_hint="",
            cost="this_mac",
            base_url="http://127.0.0.1:8080/v1",  # python -m neo2 --local
            check={"method": "GET", "path": "/models", "auth": "none"},
            defaults={"brain": "local", "quick": "local"},
            brains=("local",),
            roles=("brain", "quick"),
        ),
    )
}


@dataclass(frozen=True)
class Template:
    """What the "Other endpoint (OpenAI-compatible)" form starts from."""

    id: str
    label: str
    base_url: str
    env: str | None = None
    key_url: str = ""
    key_hint: str = ""
    models: dict[str, str] = field(default_factory=dict)  # role → model id ("" or absent: from the live list)
    tools: bool = True
    vision: bool = False
    short_tool_ids: bool = False
    check: dict[str, str] = field(default_factory=lambda: {"method": "GET", "path": "/models", "auth": "bearer"})
    needs_key: bool = True


TEMPLATES: dict[str, Template] = {
    t.id: t
    for t in (
        Template(
            "xai", "xAI (Grok)", "https://api.x.ai/v1", env="XAI_API_KEY", key_url="https://console.x.ai",
            key_hint="xai-", models={"brain": "grok-4.7", "quick": "grok-4.7"},
        ),
        Template(
            "deepseek", "DeepSeek", "https://api.deepseek.com", env="DEEPSEEK_API_KEY",
            key_url="https://platform.deepseek.com/api_keys",
            models={"brain": "deepseek-v4-pro", "quick": "deepseek-flash"},
            check={"method": "GET", "path": "/user/balance", "auth": "bearer"},
        ),
        Template(
            "mistral", "Mistral", "https://api.mistral.ai/v1", env="MISTRAL_API_KEY",
            key_url="https://console.mistral.ai/api-keys", short_tool_ids=True,  # (9-character tool-call ids)
        ),
        Template(
            "together", "Together", "https://api.together.xyz/v1", env="TOGETHER_API_KEY",
            key_url="https://api.together.ai/settings/api-keys",
        ),
        Template(
            "ollama", "Ollama (on this Mac)", "http://localhost:11434/v1", needs_key=False,
            check={"method": "GET", "path": "/models", "auth": "none"},
        ),
        Template(
            "lmstudio", "LM Studio (on this Mac)", "http://localhost:1234/v1", needs_key=False,
            check={"method": "GET", "path": "/models", "auth": "none"},
        ),
        Template("other", "Other endpoint", ""),
    )
}  # fmt: skip


def provider(pid: str) -> ProviderInfo | None:
    return PROVIDERS.get(pid)


def known_model(pid: str, model: str) -> ModelInfo | None:
    p = PROVIDERS.get(pid)
    return next((m for m in p.models if m.id == model), None) if p else None


def as_dict(p: ProviderInfo) -> dict[str, Any]:
    """A provider as the Brains & Keys window lists it (no key, no value of one)."""
    return {
        "id": p.id,
        "label": p.label,
        "transport": p.transport,
        "cost": p.cost,
        "key_url": p.key_url,
        "key_hint": p.key_hint,
        "roles": list(p.roles),
        "env": p.env,
        "check": dict(p.check),
    }
