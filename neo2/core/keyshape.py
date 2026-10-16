"""What a pasted API key looks like: soft hints for the Brains & Keys window and the CLI, and the
composer's key guard (ui/src/keys/keyshape.ts is the window's twin of this file).

A hint never stops a save: "This looks like an OpenAI key — did you mean OpenAI?" A key is cleaned
as it is kept (keystore.clean_key: spaces and quotes around it go; a line break, a control character
or more than 512 characters is refused). Nothing here keeps, logs or sends a key.
"""

from __future__ import annotations

import re

# Prefixes that say which provider a key is for, longest first. "sk-" alone is OpenAI's older shape,
# but DeepSeek's and others' keys start with it too: it names no provider by itself.
PREFIXES: tuple[tuple[str, str], ...] = (
    ("sk-ant-", "anthropic"),
    ("sk-or-", "openrouter"),
    ("sk-proj-", "openai"),
    ("AIza", "gemini"),
    ("gsk_", "groq"),
    ("csk-", "cerebras"),
    ("xai-", "xai"),
)
# What a provider's keys start with (the first is the one to suggest).
EXPECTED: dict[str, tuple[str, ...]] = {
    "anthropic": ("sk-ant-",),
    "openrouter": ("sk-or-",),
    "openai": ("sk-proj-", "sk-"),
    "gemini": ("AIza",),
    "groq": ("gsk_",),
    "cerebras": ("csk-",),
    "xai": ("xai-",),
}
LABELS = {
    "anthropic": "Anthropic",
    "openrouter": "OpenRouter",
    "openai": "OpenAI",
    "gemini": "Gemini",
    "groq": "Groq",
    "cerebras": "Cerebras",
    "xai": "xAI",
}
_AN = {"anthropic", "openrouter", "openai", "xai"}  # "an Anthropic key"
# The composer's guard: a message whose first word is a key — a known prefix and then a run of key
# characters — is never sent to a model. Narrow on purpose: "sk-learn" or a long hash is not a key.
_KEY_BODY = re.compile(r"[A-Za-z0-9_\-]{16,}")


def family(provider: str) -> str:
    """The provider a key is for: an endpoint the user added from the xAI template is xAI's."""
    p = (provider or "").strip().lower()
    return p[len("custom:") :] if p.startswith("custom:") else p


def guess(key: str) -> str | None:
    """The provider `key`'s prefix names (None: none does)."""
    k = (key or "").strip().strip("\"'").strip()
    return next((pid for prefix, pid in PREFIXES if k.startswith(prefix)), None)


def hint(provider: str, key: str) -> str | None:
    """A soft hint about `key` for `provider` (None: nothing to say). Never a reason to refuse it."""
    k = (key or "").strip().strip("\"'").strip()
    if not k:
        return None
    fam = family(provider)
    g = guess(k)
    if g is not None and g != fam:
        name = LABELS[g]
        return f"This looks like {'an' if g in _AN else 'a'} {name} key — did you mean {name}?"
    want = EXPECTED.get(fam)
    if want and not k.startswith(want):
        return f"{LABELS[fam]} keys usually start with {want[0]}."
    return None


def problem(key: str) -> str | None:
    """Why `key` can't be kept as it is (a line break, a space, a control character, over 512
    characters, nothing at all), else None. The words never say the key."""
    from neo2.core.keystore import clean_key

    try:
        clean_key(key)
    except ValueError as e:
        return str(e)
    return None


def looks_like_key(text: str) -> bool:
    """Typed text that starts with an API key (the composer's guard: such a message is never sent
    to a model). Only the known prefixes, each followed by at least 16 key characters."""
    first = (text or "").strip().split(maxsplit=1)
    if not first:
        return False
    word = first[0].strip("\"'`")
    for prefix, _pid in PREFIXES:
        if word.startswith(prefix):
            return bool(_KEY_BODY.fullmatch(word[len(prefix) :]))
    return word.startswith("sk-") and bool(re.fullmatch(r"[A-Za-z0-9_\-]{32,}", word[3:]))
