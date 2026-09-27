"""Brains to hand an agent (``AsyncAgent(provider=...)``): NEO's own provider classes, built from
the host's keys or clients.

The constructors never read NEO's settings, the environment's NEO_* values or ~/.neo; a vendor SDK
is imported only when its provider is built (a missing one raises ProviderError naming the
extra to install). Anything that implements :class:`Provider` works too — a host's own class,
or ``neo2.testing.ScriptedBrain`` in tests.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from neo.providers.base import (
    ImagePart,
    Message,
    Provider,
    ProviderError,
    RateLimited,
    ToolCall,
    ToolResult,
    ToolSpec,
    Turn,
    Usage,
)

if TYPE_CHECKING:
    import anthropic
    import openai

__all__ = [
    "Claude",
    "Failover",
    "Gemini",
    "Groq",
    "ImagePart",
    "Message",
    "OpenAICompatible",
    "Provider",
    "ProviderError",
    "RateLimited",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
    "Turn",
    "Usage",
]

_GEMINI_MODEL = "gemini-3.8-flash"
_GEMINI_FALLBACKS = ("gemini-3.7-flash", "gemini-3.5-flash")


def _need(module: str, extra: str) -> Any:
    """Import a vendor SDK (or NEO's module that wraps it); missing → ProviderError."""
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ProviderError(f"{extra} support isn't installed: install NEO with its [{extra}] extra ({e})") from None


def OpenAICompatible(
    *,
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    client: openai.AsyncOpenAI | None = None,
    vision: bool = False,
    reasoning_effort: bool = False,
    name: str = "openai",
) -> Provider:
    """Any chat-completions server: OpenAI, Groq, Together, vLLM, Ollama, LM Studio… Pass your own
    configured `client`, or `base_url` and `api_key` (unset: the openai SDK's own defaults — its
    OPENAI_API_KEY — and, for a `base_url` server that takes no key, none)."""
    mod = _need("neo.providers.openai_compat", "openai")
    if client is None:
        openai_mod = _need("openai", "openai")
        try:
            client = openai_mod.AsyncOpenAI(base_url=base_url, api_key=api_key)
        except openai_mod.OpenAIError as e:
            if base_url is None or api_key is not None:
                raise ProviderError(f"OpenAICompatible needs an API key: pass api_key= ({e})") from None
            # the host's own server (Ollama, LM Studio, a local vLLM…): it takes no key
            client = openai_mod.AsyncOpenAI(base_url=base_url, api_key="none")
    return mod.OpenAICompatProvider(
        name=name, model=model, supports_vision=vision, reasoning_effort=reasoning_effort, client=client
    )


def Gemini(
    api_key: str | Sequence[str],
    *,
    model: str = _GEMINI_MODEL,
    fallback_models: Sequence[str] = _GEMINI_FALLBACKS,
) -> Provider:
    """Google Gemini. Several keys: tried in order when one runs out of quota."""
    keys = [api_key] if isinstance(api_key, str) else list(api_key)
    if not any(k and k.strip() for k in keys):
        raise ProviderError("Gemini needs an API key")
    mod = _need("neo.providers.gemini", "gemini")
    return mod.GeminiProvider(model, keys=keys, fallback_models=list(fallback_models))


def Groq(api_key: str, *, model: str = "openai/gpt-oss-120b") -> Provider:
    """Groq's free tier (fast, text only)."""
    if not api_key:
        raise ProviderError("Groq needs an API key")
    mod = _need("neo.providers.openai_compat", "openai")
    return mod.OpenAICompatProvider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
        model=model,
        reasoning_effort=True,
    )


def Claude(
    api_key: str | None = None, *, model: str = "claude-opus-5", client: anthropic.AsyncAnthropic | None = None
) -> Provider:
    """Anthropic's Claude. Pass `api_key` or your own `client` (unset: the anthropic SDK's own
    default, ANTHROPIC_API_KEY)."""
    mod = _need("neo.providers.claude", "claude")
    if client is None:
        client = _need("anthropic", "claude").AsyncAnthropic(api_key=api_key)
    return mod.ClaudeProvider(model, client=client)


def Failover(*providers: Provider, sticky: bool = True) -> Provider:
    """Try `providers` in order, moving on when one is rate-limited or unreachable. `sticky`: one
    that just failed goes to the back of the line for a few minutes (per agent)."""
    if not providers:
        raise ValueError("Failover needs at least one provider")
    from neo.providers import Chain

    return Chain(list(providers), sticky=sticky)
