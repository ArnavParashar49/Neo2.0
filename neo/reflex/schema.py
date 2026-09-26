"""Typed questions the reflex layer answers for every utterance.

Kept deliberately small: few options → Laya scores each well even zero-shot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Intent = Literal["chat", "quick_action", "agent_task", "stop"]

INTENT_CRITERIA: dict[str, str] = {
    "chat": "a question, conversation, opinion, fact, math, or writing help that needs no action "
    "on the computer",
    "quick_action": "one simple device or app command: change volume or brightness, open or close "
    "an app or website, set a timer or reminder, check the weather or time, play or pause",
    "agent_task": "anything needing several steps or looking at the computer: work inside an app, "
    "read the screen, browse or research, edit files, write and send email, organize, install, "
    "fill forms, code, or automate",
    "stop": "cancel, stop, never mind, be quiet, go to sleep, or shut down",
}

QUESTIONS: dict[str, dict] = {
    "intent": {
        "type": "choice",
        "instructions": "What kind of request is `text`?",
        "criteria": INTENT_CRITERIA,
    },
    "destructive": {
        "type": "noul",
        "instructions": "Would carrying out `text` delete, send, pay, purchase, install, uninstall, "
        "overwrite, or otherwise change something that is hard to undo?",
    },
    "needs_screen": {
        "type": "noul",
        "instructions": "Does answering `text` require seeing what is currently on the user's screen "
        "or in an open window?",
    },
}


# Questions and requests for information want an answer; commands just want doing. The reflex
# head learns this from data; this is the fallback (and the labeller for NEO's own rows).
_REPLY_RE = re.compile(
    r"^\s*(?:hey\s+)?(?:neo[,!]?\s+)?(?:please\s+|can\s+you\s+|could\s+you\s+|would\s+you\s+|will\s+you\s+)?"
    r"(?:what|what's|whats|when|where|who|whose|why|how|which|is|are|am|was|were|do|does|did|have|has|"
    r"should|tell\s+me|show\s+me|let\s+me\s+know|check|read|summari[sz]e|find\s+out|look\s+up|"
    r"search|google|any\b|list|give\s+me|explain|describe|compare|count|calculate|convert|translate|"
    r"define|recommend|suggest|remind\s+me\s+what|do\s+i\b|did\s+i\b|am\s+i\b)\b",
    re.I,
)


def wants_reply(text: str) -> bool:
    t = text.strip()
    return t.endswith("?") or bool(_REPLY_RE.match(t))


# Anywhere in the sentence — "open notes and tell me how many there are" asks something even
# though it starts with a command.
_ASKS_RE = re.compile(
    r"\b(?:tell\s+me|let\s+me\s+know|what|what's|whats|how\s+many|how\s+much|which|who|when|where|why|"
    r"read\s+(?:it|them|me|out)|show\s+me|is\s+there|are\s+there|do\s+i\s+have|did\s+i|summari[sz]e)\b",
    re.I,
)


def asks_something(text: str) -> bool:
    """Does the utterance ask for information anywhere in it?"""
    return wants_reply(text) or bool(_ASKS_RE.search(text))


# A request about what is on the screen. Labels NEO's own rows for the needs_screen head, and backs
# the head up at run time (it only sees phrasings like these in its training data anyway).
SCREEN_RE = re.compile(
    r"\b(?:this|that|the|my) (?:screen|window|page|form|error|tab|article|table|dialog|button)s?\b"
    r"|\bwhat am i looking at|\bthat'?s open\b|\bon (?:my|the) screen\b|\bthis (?:one|thing)\b",
    re.I,
)


def refers_to_screen(text: str) -> bool:
    return bool(SCREEN_RE.search(text))


# The words of an action that is hard to undo, in English, Hinglish and Hindi ("delete/install/pay
# karo" are caught by the English verb). A backstop under the destructive head so the confirm signal
# never misses them: the head was measured missing Hinglish sends (recall 0.17-0.33), and its
# training positives are English templates only.
_DV = "\u0900-\u097f"  # Devanagari: its vowel signs aren't word characters, so \b can't bound a word
_DESTRUCTIVE_RE = re.compile(
    r"\b(?P<w>send|e-?mail|message|text|reply|post|delete|remove|erase|format|install|uninstall|pay|buy|"
    r"purchase|transfer|wipe|overwrite)\b"
    r"|(?<!fast )(?<!fast-)\bforward\s+(?:it|this|that|these|those|them|the|my|his|her|all|everything|"
    r"an?\s+(?:e-?mail|mail|message|file|link))\b"
    r"|\bempty\s+(?:the\s+|my\s+)?(?:trash|bin|recycle\s+bin)\b|\bmake\s+(?:a|an|the)\s+(?:payment|transfer|purchase)\b"
    # Hinglish imperatives: bhej / bhejo / bhejna / bhej do / bhej dena / bhej dijiye (not "bheja": a
    # brain, or "sent"), hata / mita alike, kharid lo, khaali kar do (not "kharidari", "khaali hai").
    r"|\b(?:bhej(?:o|na|ne|iye|ie|en|\s*d(?:o|e|ena|en|ijiye|ijie)|\s*l(?:o|e|ena))?"
    r"|(?:hata|mita)(?:o|na|iye|ie|en|\s*d(?:o|e|ena|en|ijiye|ijie))"
    r"|khar(?:i|ee)d(?:o|na|iye|ie|en|\s*l(?:o|e|ena|en|ijiye|ijie)|\s*d(?:o|e|ena))"
    r"|kh(?:aa|a)l[iy]\s+kar(?:o|na|en|\s*d(?:o|e|ena|en|ijiye|ijie)))\b"
    # Hindi: भेजो / भेज दो, हटाओ / हटा दो, मिटाओ, खरीदो / ख़रीद लो, and डिलीट / इंस्टॉल / पे / पेमेंट /
    # भुगतान / ट्रांसफ़र / ईमेल / मैसेज + करो (not भेजा, खरीदारी, मिठाई, पेड़)
    rf"|(?<![{_DV}])(?:भेज(?:ो|ना|िए|िये|ें|\s+(?:दो|दे|देना|दें|दीजिए|दीजिये))?"
    r"|(?:हटा|मिटा)(?:ओ|ना|इए|इये|एं|एँ|\s+(?:दो|दे|देना|दें|दीजिए|दीजिये))"
    r"|(?:ख\u093c?|\u0959)रीद(?:ो|ना|िए|िये|ें|\s+(?:लो|ले|लेना|लें|लीजिए|लीजिये|दो|दे))"
    r"|(?:डिल[ीि]ट|(?:अन)?इ[ंन]्?स्ट[ॉा]ल|पे|पेमेंट|भुगतान|ट्रांस(?:फ\u093c?|\u095e)र|ई-?मेल|मेल|मैसेज)"
    r"\s*(?:करो|करना|करें|कीजिए|कीजिये|कर\s+(?:दो|दे|देना|दें|दीजिए|दीजिये)))"
    rf"(?![{_DV}])",
    re.I,
)
# ...but "my email", "the last message", "some text" name a thing, not an action.
_NOUNS = {"email", "e-mail", "message", "text", "reply", "post", "format", "purchase", "transfer"}
_DETERMINER = re.compile(
    r"\b(?:my|the|a|an|any|some|of|new|unread|last|latest|this|that|his|her|their|your|our|what|which|selected)\s+$",
    re.I,
)


def sounds_destructive(text: str) -> bool:
    for m in _DESTRUCTIVE_RE.finditer(text):
        if (m.group("w") or "").lower() in _NOUNS and _DETERMINER.search(text[: m.start()]):
            continue
        return True
    return False


@dataclass
class Decision:
    intent: Intent
    intent_confidence: float
    destructive: bool
    destructive_p: float
    needs_screen: bool
    needs_screen_p: float
    source: str = "laya"  # laya | lite | rules
    latency_ms: float = 0.0
    tool: str = ""  # the one NEO tool that does this (quick actions), when the reflex is sure
    tool_p: float = 0.0
    reply: bool = True  # does the user want an answer, or just the thing done (silently)?
    reply_p: float = 1.0
    margin: float | None = None  # top intent probability minus the runner-up (Laya only)
    kind: str = ""  # an action's task kind (a neo/decide TASKS key), when the reflex has a kind head
    kind_p: float = 0.0

    @property
    def confident(self) -> bool:
        return self.intent_confidence >= 0.55
