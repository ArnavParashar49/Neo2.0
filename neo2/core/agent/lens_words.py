"""Does the user want Search Screen? The words alone decide ("search my screen", "screen pe search
karo", "स्क्रीन पर सर्च करो") — never a model's.

* :func:`wants_lens` — the whole sentence asks NEO to open Search Screen, the overlay where the
  user circles or drags over part of the screen to search it: "search my screen", "search the
  screen", "search on my screen", "screen search", "circle to search", "search what's on my
  screen", "search this on my screen", in English, Hinglish and Hindi. "search my screen for <X>"
  opens it with X as the question for the first area the user picks. Never "what's on my screen"
  (the agent's own look), "search the web for screen protectors", "screen recording settings" or
  "search X on google" — and never a web search for a thing: "search a monitor for gaming", "search
  the monitor for deals", "search a display for my macbook" (only *my*/*this* screen, display or
  monitor, or *the screen*, is the one in front of the user).
* :func:`control` — while Search Screen is open with an area picked, a sentence that closes it
  ("close", "never mind", "band karo") or hands the area to NEO's chat ("continue in chat").

Whole-sentence matches only (after how NEO was addressed, a polite lead and a tail come off):
"remind me to search my screen later" never opens it. Who said the words is fastpath.lens's to
check.
"""

from __future__ import annotations

import re

from neo2.core.agent.retry_words import ADDRESS

_EDGE = " ,.;:!?।"
_LEAD = re.compile(
    r"^(?:(?:please|pls|so|ok|okay|hey|um+|uh+|and|acha|achha|accha|just|now)\s+)*"
    r"(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?|i\s+(?:want|wanna|need|would\s+like|'d\s+like)\s+to\s+"
    r"|let(?:'s|\s+me|\s+us)\s+|(?:go\s+ahead\s+and|quickly|quick)\s+)?",
    re.I,
)
_TAIL = re.compile(r"[\s,]*(?:please|pls|neo|now|right\s+now|for\s+me|quickly|yaar|yar|bhai|na|zara|ji)$", re.I)

_SCREEN = r"(?:screen|display|monitor)"
# The screen in front of the user: "my/this screen|display|monitor", or "the screen" — never "a
# monitor", "the display", "your screen" (a thing to search the web for).
_MINE = rf"(?:(?:my|this)\s+{_SCREEN}|the\s+screen)"
_ENGLISH = [
    rf"(?:search|scan)\s+(?:on\s+|in\s+|across\s+|through\s+)?{_MINE}",
    rf"search\s+(?:what(?:'s|s|\s+is)|whatever(?:'s|\s+is)?|something|anything|stuff|things)\s+on\s+{_MINE}",
    rf"search\s+(?:this|that|it)\s+on\s+{_MINE}",
    rf"{_SCREEN}\s+search",
    r"search\s+screen",
    r"circle\s+(?:to|and)\s+search",
    rf"(?:open|start|use|show|do|launch|turn\s+on)\s+(?:the\s+)?(?:search\s+screen|{_SCREEN}\s+search|circle\s+to\s+search)",
    rf"(?:let\s+me\s+)?circle\s+(?:something|what\s+i\s+want|it|this)\s+on\s+{_MINE}(?:\s+to\s+search)?",
]
# "search my screen for <question>": the question is what comes after "for".
_FOR = re.compile(
    rf"^(?:search|scan)\s+(?:on\s+|in\s+|across\s+)?{_MINE}\s+(?:for|about)\s+(?P<ask>.{{2,200}})$", re.I
)

# Hinglish, as the transcript writes it
_PE = r"(?:pe|par|per|pr|mein|me|main|mai)"
_KARO = r"(?:karo|kar\s+do|kardo|kijiye|kariye|karna(?:\s+(?:hai|h))?|kar)"
_HINGLISH = [
    rf"(?:meri\s+|mere\s+|is\s+|iss\s+)?(?:screen|skreen)\s+{_PE}\s+(?:kuch\s+|ye\s+|yeh\s+)?(?:search|serch|dhoondo|dhundo|khojo)\s*{_KARO}?",
    rf"(?:screen|skreen)\s+(?:search|serch)\s+{_KARO}",
    rf"(?:search|serch)\s+(?:screen|skreen)\s+{_KARO}",
]
# Hindi (Devanagari): no \b — its vowel signs aren't word characters to `re`
_HINDI = [
    r"(?:मेरी\s+|इस\s+)?स्क्रीन\s+(?:पर|पे|में)\s+(?:कुछ\s+)?(?:सर्च|खोज|ढूंढो|ढूँढो)\s*(?:करो|कीजिए|कर\s+दो|करें|करना\s+है)?",
    r"स्क्रीन\s+सर्च\s+(?:करो|कीजिए|कर\s+दो|करें)",
]

_ASKS = re.compile(r"^(?:" + "|".join(f"(?:{p})" for p in _ENGLISH + _HINGLISH) + r")$", re.I)
_ASKS_HI = re.compile(r"^(?:" + "|".join(f"(?:{p})" for p in _HINDI) + r")$")

# While it is open, with an area picked
_CLOSE = re.compile(
    r"^(?:close(?:\s+(?:it|this|that|search\s+screen|the\s+overlay))?|cancel(?:\s+(?:it|this|that))?|never\s*mind|"
    r"nevermind|forget\s+it|stop|done|that'?s\s+(?:all|it)|exit|dismiss|go\s+away|no\s+thanks|"
    r"band\s+(?:karo|kar\s+do|kardo)|rehne\s+do|chhodo|chodo|"
    r"बंद\s+(?:करो|कर\s+दो)|रहने\s+दो|छोड़ो)$",
    re.I,
)
_CONTINUE = re.compile(
    r"^(?:continue(?:\s+(?:in|on|with)\s+(?:the\s+)?(?:chat|neo))?|open\s+(?:it\s+|this\s+)?in\s+(?:the\s+)?(?:chat|neo)|"
    r"(?:take|send|move)\s+(?:it|this)\s+to\s+(?:the\s+)?(?:chat|neo)|(?:tell|show)\s+me\s+more\s+in\s+(?:the\s+)?chat|"
    r"chat\s+(?:mein|me|main)\s+(?:kholo|khol\s+do|le\s+chalo|continue\s+karo))$",
    re.I,
)


def _plain(text: str) -> str:
    """One space, how NEO was addressed and the edge punctuation off — the casing kept (a question
    the sentence carries goes on as the user said it; every pattern here ignores case)."""
    t = " ".join(str(text or "").replace("’", "'").split()).strip(_EDGE)
    return ADDRESS.sub("", t, count=1).strip(_EDGE)


def _untail(t: str) -> str:
    for _ in range(3):  # "…, please, neo?"
        t2 = _TAIL.sub("", t).strip(_EDGE)
        if t2 == t or not t2:
            break
        t = t2
    return t


def _cores(text: str) -> tuple[str, ...]:
    """The sentence as said, without a trailing "please", and without its polite lead too."""
    plain = _plain(text)
    whole = _untail(plain)
    led = _LEAD.sub("", whole, count=1).strip(_EDGE) or whole
    return tuple(dict.fromkeys((plain, whole, led)))


def wants_lens(text: str) -> tuple[bool, str]:
    """(do these words, as a whole sentence, ask for Search Screen; the question they carry — "" for
    none). "search my screen for the price of this" → (True, "the price of this")."""
    raw = " ".join(str(text or "").split())
    if not raw or len(raw) > 260:
        return False, ""
    cores = _cores(raw)
    for c in cores:
        if _ASKS.match(c) or _ASKS_HI.match(c):
            return True, ""
    for c in reversed(cores):  # (the core first: "…, please" is no part of the question)
        if m := _FOR.match(c):
            return True, m.group("ask").strip(_EDGE)[:200]
    return False, ""


def control(text: str) -> str | None:
    """"close" or "continue" for a sentence that closes Search Screen or hands its area to NEO's
    chat; None for anything else (a question about the area)."""
    raw = " ".join(str(text or "").split())
    if not raw or len(raw) > 80:
        return None
    for c in _cores(raw):
        if _CLOSE.match(c):
            return "close"
        if _CONTINUE.match(c):
            return "continue"
    return None


__all__ = ["control", "wants_lens"]
