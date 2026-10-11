"""Does the user want NEO to look through the camera? The words alone decide ("what am I holding",
"ye kya hai", "मेरे हाथ में क्या है") — never a model's.

* :func:`wants_camera` — the whole sentence asks about something in front of the camera: what the
  user is holding, "what is this?", "can you see me", "read this label", "how many fingers", "what
  colour is this", in English, Hinglish and Hindi. A sentence about the screen ("what is on my
  screen", "look at this page", "read this email") never is: that is the screen tools' job.
* :func:`follow_up` — "and now?", "what about this one?", "aur ye?" right after a look: the camera
  is still on for a short window, so the next question reuses it (the window closes whenever the
  camera goes off: a follow-up never turns it back on).
* :func:`wants_off` — "turn off the camera", "camera band karo", "कैमरा बंद करो".
* :func:`pointing_only` — a bare "what is this?" / "ye kya hai" with nothing about the camera in it
  (no hand, no "me", no object named): it could be about the screen, so the router asks Laya.

Whole-sentence matches only (after how NEO was addressed and a polite lead come off): "remind me to
ask what this is" never turns the camera on. Who said the words is fastpath.camera's to check.
"""

from __future__ import annotations

import re
import time

from neo2.core.agent import retry_words
from neo2.core.agent.retry_words import ADDRESS, is_retry_request

FOLLOW_UP_S = 25.0  # the camera's follow-up window (≈20 s after the answer, plus the answer itself)
_last_look = 0.0
_last_question = ""

# ---- how the sentence is framed ------------------------------------------------------------------
_ADDRESS = ADDRESS  # (one for every lane: retry_words)
_LEAD = re.compile(
    r"^(?:(?:please|pls|so|ok|okay|hey|um+|uh+|and|acha|achha|accha)\s+)*"
    r"(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?(?:(?:tell|show)\s+me|check|see|say|guess|look\s+and\s+tell\s+me)?\s*"
    r"|(?:do|did)\s+you\s+know\s+|tell\s+me\s+|guess\s+|quick(?:ly)?\s+)?",
    re.I,
)
_TAIL = re.compile(
    r"[\s,]*(?:please|pls|neo|now|right\s+now|for\s+me|quickly|then|yaar|yar|bhai|na|zara|ji|"
    r"using\s+(?:the|your|my)\s+(?:camera|webcam)|(?:with|through|on|from)\s+(?:the|your|my)\s+(?:camera|webcam))$",
    re.I,
)
_EDGE = " ,.;:!?।"

# ---- the screen: never the camera ------------------------------------------------------------------
_SCREEN = re.compile(
    r"\b(?:screen|screenshot|monitor|display|desktop|page|web\s*page|website|site|tab|tabs|window|browser|"
    r"app|email|e-mail|mail|inbox|message|messages|chat|document|doc|docs|pdf|file|folder|slide|sheet|"
    r"spreadsheet|cell|button|menu|link|url|tweet|post|video|photo|picture|image|pic|chart|graph|error|"
    r"code|notification|dialog|popup|pop-up|article|song|music|word|sentence|paragraph|line|text\s+here|font|icon|"
    r"emoji|gif|meme|game|movie|film|episode|table|list|result|results|map|field|form|terminal|setting|settings)\b"
    r"|स्क्रीन|पेज|वेबसाइट|ईमेल",
    re.I,
)

# ---- things in front of the camera ---------------------------------------------------------------
_THING = (
    r"(?:thing|object|item|stuff|product|gadget|device|toy|fruit|vegetable|plant|flower|animal|"
    r"bottle|can|cup|mug|glass|box|jar|packet|package|pack|carton|wrapper|label|tag|sticker|book|pen|"
    r"card|coin|medicine|tablet|pill|shirt|t-?shirt|dress|hat|cap|jacket|shoe|watch|ring|key|keys)"
)
_ME = r"(?:me|my\s+face|my\s+hands?|my\s+outfit|my\s+shirt|us)"
_HAND = r"(?:in\s+(?:my|both\s+my|my\s+(?:left|right))\s+hands?|i(?:'m|\s+am)\s+holding(?:\s+up)?|i\s+have\s+(?:here|in\s+my\s+hands?)|i(?:'m|\s+am)\s+showing\s+you)"

_WHATS = r"what(?:'s|s|\s+is)"
_ENGLISH = [
    # what am I holding / what's in my hand / what do I have in my hand
    r"what(?:'s|s|\s+is|\s+are)?\s+(?:this\s+|that\s+|the\s+thing\s+)?(?:am\s+)?i(?:'m|\s+am)?\s+holding(?:\s+up)?(?:\s+(?:here|right\s+now|now))?",
    r"what\s+am\s+i\s+holding(?:\s+up)?(?:\s+(?:here|in\s+my\s+hands?))?",
    rf"{_WHATS}\s+(?:this\s+|that\s+)?(?:\w+\s+)?{_HAND}",
    rf"{_WHATS}\s+in\s+(?:my|both\s+my)\s+hands?",
    r"what\s+(?:do\s+)?i\s+(?:have|got)\s+in\s+my\s+hands?",
    r"what\s+(?:thing|object|item)\s+(?:is\s+this|am\s+i\s+holding)",
    # what is this (thing)?  what's this called?  what kind of X is this?
    rf"{_WHATS}\s+(?:this|this\s+{_THING}|that\s+{_THING})(?:\s+(?:called|here|right\s+here|thing))?",
    rf"what(?:\s+is|'s|s)?\s*this\s+(?:{_THING}\s+)?called",
    rf"what\s+(?:kind|type|sort|brand|model|make)\s+of\s+(?:\w+\s+)?is\s+(?:this|that)(?:\s+{_THING})?",
    rf"(?:identify|recognise|recognize)\s+(?:this|that|what\s+i(?:'m|\s+am)\s+holding)(?:\s+{_THING})?",
    rf"(?:tell\s+me\s+)?what\s+this\s+(?:{_THING}\s+)?is",
    # look at this / look at me / what do you see
    rf"(?:take\s+a\s+)?look\s+at\s+(?:this|that|me|this\s+{_THING}|what\s+i(?:'m|\s+am)\s+holding)",
    rf"(?:can|could|do)\s+you\s+see\s+(?:me|this|that|{_ME}|this\s+{_THING}|what\s+i(?:'m|\s+am)\s+holding)",
    r"what\s+(?:do|can)\s+you\s+see",
    r"what\s+am\s+i\s+wearing",
    r"how\s+do\s+i\s+look",
    r"do\s+i\s+look\s+(?:ok|okay|good|fine|alright|all\s+right|nice)",
    # read this label
    rf"read\s+(?:this|that|the|my)\s+(?:{_THING})(?:\s+{_HAND})?",
    rf"read\s+(?:{_WHATS}\s+written\s+on\s+)(?:this|that|it)(?:\s+{_THING})?",
    rf"what\s+does\s+(?:this|that|the)\s+(?:{_THING})\s+say",
    r"read\s+what\s+i(?:'m|\s+am)\s+holding",
    # how many fingers
    r"how\s+many\s+fingers(?:\s+(?:am\s+i\s+holding(?:\s+up)?|do\s+you\s+see|can\s+you\s+see|is\s+this|are\s+these|are\s+up|am\s+i\s+showing(?:\s+you)?))?",
    # what colour
    rf"what\s+colou?r\s+(?:is|'s|are)\s+(?:this|that|these|those|my\s+\w+|this\s+{_THING}|that\s+{_THING})",
    rf"what\s+colou?r\s+(?:{_THING}\s+)?(?:am\s+i\s+(?:wearing|holding)|is\s+(?:the\s+)?(?:thing|object)\s+{_HAND})",
    # the camera by name
    r"(?:turn|switch)\s+on\s+(?:the|your|my)\s+(?:camera|webcam)(?:\s+and\s+.+)?",
    r"(?:turn|switch)\s+(?:the|your|my)\s+(?:camera|webcam)\s+on(?:\s+and\s+.+)?",
    r"(?:use|check|open)\s+(?:the|your|my)\s+(?:camera|webcam)\s+(?:and|to)\s+.+",
    r"look\s+(?:through|with|using)\s+(?:the|your|my)\s+(?:camera|webcam).*",
]

# Hinglish, as the transcript writes it
_YE = r"(?:ye|yeh|yah|yeah|ya)"
_KYA = r"(?:kya|kyaa|kia|kiya)"
_HAI = r"(?:hai|he|h|hain|hei|hay)"
_HINGLISH = [
    rf"(?:dekho|dekh|dekhna|dekho\s+na|zara\s+dekho)?\s*{_YE}\s+{_KYA}\s+{_HAI}",
    rf"{_YE}\s+(?:cheez|chiz|chij|saaman|saman)\s+{_KYA}\s+{_HAI}",
    rf"(?:mere|mera|meri|apne)\s+(?:haath|hath|haat|hat|hathon|haathon|hand)\s+(?:mein|me|mai|main|mei)\s+{_KYA}\s+{_HAI}",
    rf"(?:maine|mene|main\s+ne|mai\s+ne)\s+{_KYA}\s+(?:pakda|pakad|pakada|pakdi|pakde|pakad\s+rakha|pakda\s+hua|liya|pakda\s+hai)(?:\s+(?:hai|h|hua\s+hai|rakha\s+hai))?",
    r"(?:kitni|kitne|kitna)\s+(?:ungli|ungliyan|ungliya|ungliyaan|fingers?)(?:\s+(?:hai|hain|h|dikh\s+rahi\s+(?:hai|hain)|dikha\s+raha\s+hoon|utha(?:i|yi)\s+hai))?",
    rf"{_YE}\s+(?:kaunsa|konsa|kaun\s+sa|kis)\s+(?:rang|colou?r)\s+(?:ka\s+)?{_HAI}",
    rf"{_YE}\s+(?:kis|kaunse|konse)\s+(?:rang|colou?r)\s+ka\s+{_HAI}",
    rf"(?:iska|uska|iss\s+ka)\s+(?:rang|colou?r)\s+{_KYA}\s+{_HAI}",
    r"(?:kya\s+)?(?:tum|aap|tu)\s+(?:mujhe|muje|mujhko|ye|yeh|isko)\s+dekh\s+(?:sakte|sakti|sakta)\s+(?:ho|hain|hai|h)",
    rf"(?:mujhe|muje|isko|ise|ye|yeh)\s+dekho(?:\s+(?:aur|or)\s+(?:batao|bolo)(?:\s+{_YE}\s+{_KYA}\s+{_HAI})?)?",
    r"(?:is|iss|ye|yeh)\s+(?:label|tag|sticker|packet|dabba|bottle|box)\s+(?:ko\s+)?(?:padho|padh\s+do|read\s+karo)",
    rf"(?:is|iss|ye|yeh)\s+(?:label|tag|sticker|packet|dabba|bottle|box)\s+(?:pe|par|mein|me)\s+{_KYA}\s+likha\s+{_HAI}",
    r"(?:camera|webcam)\s+(?:on|chalu|shuru)\s+(?:karo|kar\s+do|kijiye)(?:\s+(?:aur|or)\s+.+)?",
    rf"(?:batao|bolo|bata\s+do)\s+{_YE}\s+{_KYA}\s+{_HAI}",
    rf"{_YE}\s+{_KYA}\s+{_HAI}\s+(?:batao|bolo|bata\s+do|dekho|dekh\s+ke\s+batao)",
]

# Hindi (Devanagari): no \b — its vowel signs aren't word characters to `re`
_HINDI = [
    r"(?:देखो\s*)?(?:ये|यह|ये|इसे\s+देखो\s+ये)\s+क्या\s+(?:है|हैं)(?:\s+(?:बताओ|बोलो))?",
    r"(?:मेरे|अपने)\s+(?:हाथ|हाथों)\s+(?:में|मे)\s+क्या\s+(?:है|हैं)",
    r"(?:मैंने|मैने)\s+क्या\s+(?:पकड़ा|पकड़ी|पकड़े|पकडा|पकड़\s+रखा)\s*(?:है|हुआ\s+है|रखा\s+है)?",
    r"(?:कितनी|कितने)\s+(?:उंगलियां|उंगलियाँ|उँगलियाँ|उंगली|उँगली)(?:\s+.+)?",
    r"(?:क्या\s+)?(?:तुम|आप)\s+(?:मुझे|इसे)\s+देख\s+(?:सकते|सकती|सकता)\s+(?:हो|हैं|है)",
    r"(?:ये|यह|ये)\s+(?:किस|कौन\s+से|कौनसे)\s+रंग\s+का\s+(?:है|हैं)",
    r"(?:इसका|इस\s+का)\s+रंग\s+क्या\s+है",
    r"(?:इस|ये|यह)\s+(?:लेबल|पैकेट|बोतल|डिब्बे)\s+(?:पर|पे|में)\s+क्या\s+लिखा\s+है",
    r"कैमरा\s+(?:चालू|ऑन|शुरू)\s+(?:करो|कीजिए|कर\s+दो)(?:\s+.+)?",
]

_ASKS = re.compile(
    r"^(?:" + "|".join(f"(?:{p})" for p in _ENGLISH + _HINGLISH) + r")$",
    re.I,
)
_ASKS_HI = re.compile(r"^(?:" + "|".join(f"(?:{p})" for p in _HINDI) + r")$")

# ---- the camera off ----------------------------------------------------------------------------------
_CAM = r"(?:camera|webcam|cam)"
_OFF = re.compile(
    r"^(?:"
    rf"(?:turn|switch|shut)\s+(?:off|down)\s+(?:the\s+|your\s+|my\s+)?{_CAM}"
    rf"|(?:turn|switch|shut)\s+(?:the\s+|your\s+|my\s+)?{_CAM}\s+(?:off|down)"
    rf"|(?:stop|close|end|disable|kill)\s+(?:the\s+|your\s+|my\s+)?{_CAM}"
    rf"|(?:the\s+)?{_CAM}\s+off|stop\s+looking(?:\s+at\s+me)?|no\s+more\s+{_CAM}"
    rf"|{_CAM}\s+(?:band|bandh|off)(?:\s+(?:karo|kar\s+do|kardo|kijiye|karna|kar))?"
    r"|कैमरा\s+(?:बंद|ऑफ)(?:\s+(?:करो|कीजिए|कर\s+दो|कर))?"
    r")$",
    re.I,
)

# ---- something of the camera's in the words (not just "this") ---------------------------------------
_CUE = re.compile(
    rf"\b(?:hold|holding|held|hands?|fingers?|wear|wearing|camera|webcam|me|my|us|face|outfit|{_THING[3:-1]})s?\b"
    r"|\b(?:haath|hath|haat|hathon|haathon|ungli\w*|pakd\w*|pakad\w*|mujhe|muje|mujhko|mere|mera|meri|"
    r"cheez|chiz|chij|saaman|saman|dabba|label|packet|tag|sticker)\b"
    r"|हाथ|उंगल|उँगल|पकड़|पकडा|मुझे|मेरे|कैमरा|लेबल|पैकेट|बोतल|डिब्बे",
    re.I,
)

_NAMED = re.compile(r"\b(?:camera|webcam)\b|कैमरा", re.I)

_FOLLOW = re.compile(
    r"^(?:(?:and|ok|okay|so|acha|achha)\s+)?(?:"
    r"now|now\s+what|this\s+one|that\s+one|what\s+about\s+now|how\s+about\s+now|what\s+about\s+(?:this|that)(?:\s+one)?|"
    r"how\s+about\s+(?:this|that)(?:\s+one)?|this|now\s+this|what(?:'s|s|\s+is)\s+(?:this|that)\s+one|what(?:'s|s|\s+is)\s+it\s+now|"
    r"ab|ab\s+(?:kya|dekho|batao|bolo)|ab\s+kya\s+(?:hai|h)|aur\s+(?:ye|yeh)(?:\s+(?:kya\s+(?:hai|h)|wala))?|aur\s+ab|"
    r"(?:ye|yeh)\s+wala|और\s+(?:अब|ये|यह)|अब|अब\s+क्या\s+है|और\s+(?:ये|यह)\s+क्या\s+है)$",
    re.I,
)


def _plain(text: str) -> str:
    """Lowercase, one space, how NEO was addressed and the edge punctuation off."""
    t = " ".join(str(text or "").replace("\u2019", "'").split()).strip(_EDGE)
    return _ADDRESS.sub("", t, count=1).strip(_EDGE).lower()


def _untail(t: str) -> str:
    for _ in range(3):  # "…, please, neo?"
        t2 = _TAIL.sub("", t).strip(_EDGE)
        if t2 == t or not t2:
            break
        t = t2
    return t


def _cores(text: str) -> tuple[str, str]:
    """(the sentence without a trailing "please", without its polite lead too): "Neo, can you see
    me?" → ("can you see me", "me") — a pattern may need the lead ("can you see me")."""
    whole = _untail(_plain(text))
    return whole, (_LEAD.sub("", whole, count=1).strip(_EDGE) or whole)


def core(text: str) -> str:
    """The request itself: "Neo, can you tell me what I'm holding, please?" → "what i'm holding"."""
    return _cores(text)[1]


def about_screen(text: str) -> bool:
    """The words are about the screen (a page, a window, an email…): the screen tools' job."""
    return bool(_SCREEN.search(_plain(text)))


def wants_camera(text: str) -> bool:
    """Do these words, as a whole sentence, ask NEO to look at something in front of the camera?
    Never a sentence about the screen (unless it names the camera itself)."""
    plain = _plain(text)
    if not plain or len(plain) > 160:
        return False
    hit = any(_ASKS.match(c) or _ASKS_HI.match(c) for c in (plain, *_cores(text)))
    if not hit:
        return False
    if _NAMED.search(plain):
        return True  # named the camera
    return not _SCREEN.search(plain)


def wants_off(text: str) -> bool:
    """"turn off the camera", "stop the camera", "camera band karo", "कैमरा बंद करो"."""
    plain = _plain(text)
    if not plain or len(plain) > 80:
        return False
    return any(_OFF.match(c) for c in (plain, *_cores(text)))


def pointing_only(text: str) -> bool:
    """A camera ask with nothing of the camera's in it: "what is this?", "look at this", "ye kya
    hai" — the user may be pointing at the screen. ("what am I holding", "can you see me", "read
    this label" name a hand, the user or an object: those are the camera's.)"""
    return wants_camera(text) and not _CUE.search(core(text)) and not _NAMED.search(_plain(text))


def _window() -> float:
    try:
        from neo2.core.config import settings

        return max(FOLLOW_UP_S, float(getattr(settings(), "camera_follow_s", 0.0)) + 5.0)
    except Exception:  # noqa: BLE001
        return FOLLOW_UP_S


def follow_up(text: str, *, now: float | None = None) -> bool:
    """A short follow-up ("and now?", "aur ye?") or a retry ("try again", "phir se dekho") while the
    last look's camera window is still open. A retry is the camera's only while a look is the last
    retryable thing NEO did (retry_words.last_lane): not after another request, nor after a song
    NEO couldn't name since."""
    now = time.monotonic() if now is None else now
    if not _last_look or now - _last_look > _window():
        return False
    if is_retry_request(text, "camera") and retry_words.last_lane() == "camera":
        return True
    c = _plain(text)
    for _ in range(2):
        c = re.sub(r"[\s,]*(?:neo|please|pls)$", "", c).strip(_EDGE)
    return bool(c) and len(c) <= 40 and bool(_FOLLOW.match(c))


def follow_up_question(text: str) -> str:
    """Reuse the original visual question for a short follow-up or retry."""
    return _last_question if follow_up(text) and _last_question else text


def mark_look(now: float | None = None, *, question: str | None = None) -> None:
    """A look was routed (or answered): refresh its window, retain its question, and make it the
    lane a bare retry goes to."""
    global _last_look, _last_question
    _last_look = time.monotonic() if now is None else now
    retry_words.mark("camera")
    if question:
        _last_question = str(question)[:500]


def forget() -> None:
    """The camera went off (or the user stopped, or erased): no follow-up reuses it."""
    global _last_look, _last_question
    _last_look = 0.0
    _last_question = ""


def prewarm() -> bool:
    """Warm the camera answer's cloud brains (neo2.core.vision.answer.prewarm: the Gemini / Cerebras
    connection) while the request is still being routed. Never waits, never raises; False when
    there's nothing to warm (the vision layer isn't installed)."""
    try:
        from neo2.core.vision import answer

        fn = getattr(answer, "prewarm", None)
        if fn is None:
            return False
        fn()
        return True
    except Exception:  # noqa: BLE001 — a warm-up is never worth a failure
        return False
