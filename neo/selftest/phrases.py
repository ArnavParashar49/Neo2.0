"""The words that ask NEO to check its own computer controls: "Neo, check your computer controls".

The controls self-test moves the pointer and types for about half a minute (in a test window of its
own), so only the user's own request starts it: these patterns are its fast path, and a reflex rule
(neo.reflex._rules) so that such a sentence is never taken for conversation. A model never starts it.

Imperatives only. "Are your computer controls working?" is a question, which NEO answers; it does
not start moving the pointer.
"""

from __future__ import annotations

import re

_LEAD = (
    r"^\s*(?:(?:hey|hi|ok(?:ay)?)\s+)?(?:neo\b[\s,:!.-]*)?"
    r"(?:(?:please|can\s+you|could\s+you|would\s+you|will\s+you)\s+)?(?:(?:go\s+)?(?:and\s+)?)?"
)
_TAIL = r"(?:[\s,]+(?:please|now|for\s+me|again|once|quickly))*\s*[.!?।]*\s*$"
_WHOSE = r"(?:(?:your(?:\s+own)?|neo'?s|the|my|all\s+(?:your|the))\s+)?"
_CONTROLS = r"(?:computer|mac|mouse\s+and\s+keyboard|input)\s+controls?"
_WORKS = r"(?:\s+(?:(?:are|is)\s+)?work(?:s|ing)?(?:\s+(?:properly|correctly|right|fine|ok(?:ay)?))?)?"
_CHECK = r"(?:check|test|verify|diagnose|self[\s-]?test)"

_PATTERNS = (
    # check / test / verify (your) computer controls (are working)
    rf"{_LEAD}{_CHECK}\s+{_WHOSE}{_CONTROLS}{_WORKS}{_TAIL}",
    # check your controls — "your" makes them NEO's own
    rf"{_LEAD}{_CHECK}\s+your(?:\s+own)?\s+controls?{_WORKS}{_TAIL}",
    # check whether your computer controls work
    rf"{_LEAD}(?:check|see|test|find\s+out)\s+(?:if|whether)\s+(?:your|the)\s+{_CONTROLS}{_WORKS}{_TAIL}",
    # run a controls self-test, run the computer controls check
    rf"{_LEAD}(?:run|do|start)\s+(?:a|the|your)\s+(?:computer\s+)?controls?\s+(?:self[\s-]?test|test|check){_TAIL}",
    # Hinglish: "apne computer controls check karo"
    rf"{_LEAD}(?:(?:apne|apna|apni|neo\s+ke)\s+)?(?:computer|mac)\s+controls?\s+(?:check|test)\s+"
    rf"(?:karo|kar\s+lo|kar\s+do|kijiye|kar\s+ke\s+dekho|karke\s+dekho){_TAIL}",
    # Devanagari: "अपने कंप्यूटर कंट्रोल्स चेक करो"
    r"^\s*(?:(?:अपने|अपना)\s+)?(?:कंप्यूटर|कम्प्यूटर)\s+(?:कंट्रोल्स|कंट्रोल)\s+(?:चेक|टेस्ट)\s+"
    r"(?:करो|कर\s+लो|कर\s+दो|कीजिए|कीजिये)\s*[।.!]*\s*$",
)

# One pattern, as the tool's fast path wants it (matched case-insensitively).
PATTERN = "|".join(f"(?:{p})" for p in _PATTERNS)
REQUEST = re.compile(PATTERN, re.I)


def is_request(text: str) -> bool:
    """Is this whole utterance the user asking NEO to check its computer controls?"""
    return bool(REQUEST.match(" ".join(str(text or "").split())))
