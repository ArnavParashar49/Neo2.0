"""Small, contextual retry utterances shared by NEO's follow-up lanes."""

from __future__ import annotations

import re

_ADDRESS = re.compile(r"^(?:(?:hey|hi|ok|okay|yo)\s+)?neo\b[\s,.:!-]*", re.I)
_REQUEST = re.compile(
    r"^(?:(?:can|could|would|will)\s+you\s+)?(?:(?:please|pls|just)\s+)*"
    r"(?:try(?:\s+(?:again|it(?: again)?|that(?: again)?|once more|one more time))?"
    r"|retry(?:\s+(?:it|that))?|again|one more time|once more|"
    r"do\s+(?:it|that)\s+again|repeat(?:\s+(?:it|that))?|redo(?:\s+(?:it|that))?|"
    r"(?:look|listen)(?:\s+(?:at|to)\s+(?:it|that))?\s+again|"
    r"take\s+another\s+look|give\s+it\s+another\s+try)"
    r"(?:\s+(?:please|pls|neo))?$",
    re.I,
)
_RETRY_HI = re.compile(
    r"^(?:ek\s+baar\s+(?:aur|phir(?:\s+se)?|dobara)|ek\s+bar\s+(?:aur|phir(?:\s+se)?|dobara)|"
    r"phir\s+se|fir\s+se|dobara|एक\s+बार\s+(?:और|फिर(?:\s+से)?|दोबारा)|फिर\s+से|दोबारा)$",
    re.I,
)
_EDGE = " ,.;:!?।"


def is_retry_request(text: str) -> bool:
    """Recognize a short request to repeat the immediately preceding action.

    Callers must also have fresh context for the action being retried. This deliberately does not
    authorize a camera or microphone by itself.
    """
    value = " ".join(str(text or "").replace("’", "'").split()).strip(_EDGE)
    value = _ADDRESS.sub("", value, count=1).strip(_EDGE)
    if not value or len(value) > 80:
        return False
    return bool(_REQUEST.fullmatch(value) or _RETRY_HI.fullmatch(value))
