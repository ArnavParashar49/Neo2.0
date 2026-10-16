"""Secrets never reach NEO's memory: what is written to the journal, the facts and the summaries
goes through :func:`redact` first, and a secret's value is replaced by ``[secret removed]``.

What counts as a secret (the value, never the sentence around it):

* a key or token by its shape — provider API keys (``sk-…``, ``sk-or-…``, ``AIza…``, ``gsk_…``,
  ``xai-…``, ``ghp_…``, ``xoxb-…``, ``AKIA…``), JSON web tokens, ``Bearer …``, private-key blocks,
  and any long run of letters and digits that looks generated (40+ characters, mixed);
* the exact keys NEO has loaded (its settings' and the endpoints' — a Mistral key has no shape of
  its own), read from neo2.core.logredact.secrets as the text is cleaned, never kept here;
* what follows a word that names one — "my password is …", "pin 4321", "the OTP is 482913",
  "cvv 123", "api key: …", "mera password … hai" — the value only: "the wifi password is on the
  fridge" has none and is kept as said;
* a payment card number (13–19 digits that pass the Luhn check), with or without spaces.

A phone number, a date, a flat number or a postal code is not a secret: the user asked NEO to
remember those, and it does. (Laya's learning log is stricter — it skips any request that even
mentions a secret word or carries a long digit run: :data:`MENTIONS_SECRET`, which lived in
neo2.core.reflex.learn before and is the one pattern both share.)

Functions over text: no store, nothing kept (the loaded keys are only read).
"""

from __future__ import annotations

import re

MASK = "[secret removed]"

# A request that talks about a secret at all (the learning log skips those whole).
MENTIONS_SECRET = re.compile(
    r"\b(?:password|passcode|pass\s*word|pin|otp|one[-\s]?time\s+code|cvv|api\s*key|token|secret|ssn)\b"
    r"|\d[\d\s-]{5,}\d",  # long digit runs: codes, card and phone numbers
    re.I,
)

# ---- by shape -------------------------------------------------------------------------------
_SHAPES = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)"
    r"|\bsk-(?:ant-|proj-|or-|live-|test-)?[A-Za-z0-9_-]{16,}"  # OpenAI / Anthropic / OpenRouter style
    r"|\bAIza[0-9A-Za-z_-]{30,}"  # Google API keys
    r"|\bgsk_[A-Za-z0-9]{20,}"  # Groq
    r"|\bcsk-[A-Za-z0-9]{20,}"  # Cerebras
    r"|\bxai-[A-Za-z0-9]{20,}"  # xAI
    r"|\bgh[pousr]_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{20,}"
    r"|\bxox[abprs]-[A-Za-z0-9-]{10,}"  # Slack
    r"|\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"  # AWS access key ids
    r"|\bhf_[A-Za-z0-9]{20,}"
    r"|\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{4,}"  # a JSON web token
    r"|\bBearer\s+[A-Za-z0-9._~+/=-]{16,}",
    re.S,
)
# A long generated-looking run: letters and digits mixed, no spaces (a hash, a token).
_LONG = re.compile(r"(?<![A-Za-z0-9_-])[A-Za-z0-9_+/=-]{40,}(?![A-Za-z0-9_-])")

# ---- by the word before it --------------------------------------------------------------------
_NAMES = (
    r"pass\s*words?|passcodes?|pass\s*phrases?|pin(?!\s*code)(?:\s+number)?|otp|cvv|cvc|"
    r"(?:one[-\s]?time|verification|security|access|login|log-in|auth(?:entication)?|2fa|two[-\s]factor|"
    r"door|gate|lock(?:er)?|unlock|recovery|backup)\s+(?:code|pin|key|password)s?|"
    r"api[\s_-]*keys?|secret(?:\s+key)?|(?:(?:access|auth|api|bearer|refresh)[\s_-]*)?tokens?|private\s+key|"
    r"ssn|social\s+security(?:\s+number)?|aadha?ar(?:\s+(?:number|card))?|"
    r"(?:card|account|a/c)\s+(?:number|no\.?)|पासवर्ड|पिन|ओटीपी"
)
# "<name> is|:|=|hai <value>" — and "<name> <value>" when the value carries a digit.
_SEP = r"(?:\s+(?:is|was|=|:|are|hai|h|tha|होगा|है)\b|\s*[:=])"
_VALUE = r"(?:\"[^\"]{1,80}\"|“[^”]{1,80}”|'[^']{1,80}'|\S+)"
_NAMED = re.compile(
    rf"(?P<name>(?<![\w-])(?:{_NAMES})(?![\w-]))"
    rf"(?P<sep>(?:\s+(?:for|of|to|on|ka|ki|ke)\s+(?:\S+\s+){{0,3}}?\S+)?{_SEP}\s*|\s+)"
    rf"(?P<value>{_VALUE})",
    re.I,
)
# Hinglish word order: "mera password hunter2 hai", "पासवर्ड hunter2 है".
_VALUE_FIRST = re.compile(
    rf"(?P<name>(?<![\w-])(?:{_NAMES})(?![\w-]))\s+(?P<value>{_VALUE})\s+(?:hai|h|tha|है)(?![\w])",
    re.I,
)
# Words that follow "password is …" without being one: where it is, how it is, a question.
_NOT_A_VALUE = {
    "on", "in", "at", "under", "inside", "behind", "with", "the", "a", "an", "my", "your", "their", "its", "his", "her",
    "saved", "stored", "written", "kept", "there", "here", "same", "different", "not", "wrong", "incorrect", "correct",
    "right", "expired", "weak", "strong", "required", "needed", "missing", "empty", "blank", "what", "where", "which",
    "who", "when", "how", "why", "too", "very", "also", "still", "now", "that", "this", "it", "something", "anything",
    "nothing", "secret", "safe", "long", "short", "old", "new", "changed", "reset", "for", "from", "to", "and", "or",
    "but", "so", "if", "is", "was", "hai", "kya", "kahan", "nahi", "nahin", "yaad", "bhool", "wala", "wali", "protected",
    "manager", "field", "box", "prompt", "screen", "page", "change", "please", "yet", "again", "only", "galat", "sahi",
    "bata", "batao", "de", "do", "dedo", "kahin", "likha", "likhi", "pata", "malum", "maloom",
}  # fmt: skip
_CARD = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = ord(ch) - 48
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def _generated(s: str) -> bool:
    """A long run that looks made by a machine: letters and digits mixed (never a long word, a
    path or a line of dashes)."""
    return any(c.isdigit() for c in s) and any(c.isalpha() for c in s) and "/" not in s[:1]


def _is_value(v: str, *, bare: bool) -> bool:
    """Is what follows the secret's name its value? After "is" / ":" anything but a word that
    only says where or how it is; with nothing between ("pin 4321"), only something with a digit."""
    core = v.strip("\"'“”.,;!?()[]").strip(_GONE)
    if not core or core.lower() in _NOT_A_VALUE:
        return False
    if bare:
        return any(c.isdigit() for c in core)
    return True


_GONE = "\x00"  # stands in for a mask already in the text while it is scanned again


def _loaded() -> list[str]:
    """The keys NEO has loaded now (neo2.core.logredact), longest first; none when it can't say."""
    try:
        from neo2.core.logredact import secrets

        return secrets()
    except Exception:  # noqa: BLE001 — the shapes still apply
        return []


def _spans(text: str) -> list[tuple[int, int]]:
    text = text.replace(MASK, _GONE * len(MASK))  # what is masked already is no secret (nor its words)
    out: list[tuple[int, int]] = [m.span() for m in _SHAPES.finditer(text)]
    for key in _loaded():
        at = text.find(key)
        while at >= 0:
            out.append((at, at + len(key)))
            at = text.find(key, at + len(key))
    out += [m.span() for m in _LONG.finditer(text) if _generated(m.group())]
    for m in _NAMED.finditer(text):
        if _is_value(m.group("value"), bare=not m.group("sep").strip()):
            out.append(m.span("value"))
    for m in _VALUE_FIRST.finditer(text):
        if _is_value(m.group("value"), bare=False):
            out.append(m.span("value"))
    for m in _CARD.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and _luhn(digits):
            out.append(m.span())
    return out


def redact(text: str) -> tuple[str, int]:
    """(`text` with every secret's value replaced by ``[secret removed]``, how many were)."""
    if not text:
        return text, 0
    spans = sorted(_spans(text))
    if not spans:
        return text, 0
    merged: list[list[int]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out, pos = [], 0
    for a, b in merged:
        # trailing punctuation of the sentence stays outside the mask
        while b > a and text[b - 1] in ".,;!?)":
            b -= 1
        out.append(text[pos:a])
        out.append(MASK)
        pos = b
    out.append(text[pos:])
    return "".join(out), len(merged)


def has_secret(text: str) -> bool:
    """Does `text` carry a secret's value (see the module doc)?"""
    return bool(text) and bool(_spans(text))


def clean(text: str) -> str:
    """`text`, safe to store."""
    return redact(text)[0]
