"""How people point at an on-screen control: the words around its name, in English, Hinglish and
Hindi. A language table, never app names or sites.

* :func:`strip_frame` — the control's name inside a request: "click on the Start button" →
  "Start", "new game pe click karo" → "new game", "नया गेम वाला बटन दबाओ" → "नया गेम". Generous: the
  resolver (neo.tools.computer.resolve) only acts on what then matches a real on-screen label.
* :func:`pointing` — strict: the request *is* "press that control" (a click / tap / press verb, a
  Hinglish "… pe click karo / dabao", a Hindi "… पर क्लिक करो / दबाओ / चुनो"), one clause, not a
  question and not a key ("press enter" is the hotkey tool's). The reflex tool lane
  (neo.agent.fastpath.by_utterance) sends only these to a by-utterance tool on its own.
* :func:`content_words` — a request's or label's words minus the verb endings and particles that
  carry no name (करो / करें / दो / पर / को …, "the", "please" …), for the OS-strings match.

Screen labels are data: nothing here reads them as instructions, they are only compared with the
user's own words.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "DEVANAGARI",
    "content_words",
    "is_key_name",
    "latin_only",
    "norm",
    "pointing",
    "script",
    "strip_frame",
]

DEVANAGARI = re.compile("[ऀ-ॿ]")
# A letter of a word, Devanagari's vowel signs and nukta included (Python's \w stops at them).
_WORD_CH = r"\wऀ-ॿ"
_TOKEN = re.compile(rf"[{_WORD_CH}]+(?:['’&+.-][{_WORD_CH}]+)*")

# ---- English ------------------------------------------------------------------------------------
_POLITE = r"(?:(?:hey\s+)?neo[\s,]+)?(?:(?:please|pls|plz|kindly|now|just|and|then|ok(?:ay)?|so)[\s,]+)*"
_ASK = r"(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?)?"
_POINT_VERB = r"(?:click|double[\s-]?click|tap|press|hit|select|choose|push|pick|toggle)"
_LEAD_EN = re.compile(rf"^\s*{_POLITE}{_ASK}{_POINT_VERB}\s+(?:on\s+|onto\s+|in\s+|into\s+)?(?:the\s+|that\s+|this\s+)?", re.I)
_CONTROL_NOUNS = (
    r"(?:button|btn|link|tab|icon|menu\s+item|menu\s+option|option|checkbox|check\s*box|item|toggle|switch|"
    r"control|entry|radio\s+button|field|text\s+field|box|text\s+box)"
)
_END_FILLER = r"(?:[\s,]+(?:please|pls|plz|now|for\s+me|again|here|there|na|yaar))*\s*[.!।]*\s*$"
_TAIL_EN = re.compile(rf"\s+{_CONTROL_NOUNS}{_END_FILLER}", re.I)
_TAIL_FILLER = re.compile(_END_FILLER, re.I)

# ---- Hinglish -----------------------------------------------------------------------------------
_HING_WALA = r"(?:wala|wali|wale|waala|waali|waale|vala|vali|vaala)"
_HING_NOUN = r"(?:button|btn|option|icon|link|tab|menu|item)"
_HING_ON = r"(?:pe|par|pr|per|pa)"
_HING_ACT = r"(?:click|tap|select|press|touch)"
_HING_DO = (
    r"(?:karo|kar\s+do|kardo|kar\s+de|karde|karna|kijiye|kijie|kariye|karein|karen|kr\s+do|krdo|kro|"
    r"dabao|daba\s+do|dabado|daba\s+de|dabaiye|dabaaiye|dabaye|dabana|chuno|chun\s+lo|chunlo)"
)
_TAIL_HING = re.compile(
    rf"\s+(?:{_HING_WALA}\s+)?(?:{_HING_NOUN}\s+)?(?:{_HING_ON}\s+)?(?:{_HING_ACT}\s+)?{_HING_DO}{_END_FILLER}",
    re.I,
)
# The words that make a Hinglish sentence a click (not just "X karo", which can be any action).
_HING_POINT = re.compile(
    rf"\s+(?:(?:{_HING_WALA}\s+)?{_HING_NOUN}\s+(?:{_HING_ON}\s+)?(?:{_HING_ACT}\s+)?{_HING_DO}"
    rf"|(?:{_HING_WALA}\s+)?(?:{_HING_ON}\s+)?{_HING_ACT}\s+{_HING_DO}"
    rf"|(?:{_HING_WALA}\s+)?(?:{_HING_NOUN}\s+)?(?:dabao|daba\s+do|dabado|daba\s+de|dabaiye|dabaaiye|dabaye|chuno|chun\s+lo|chunlo))"
    rf"{_END_FILLER}",
    re.I,
)

# ---- Hindi (Devanagari) -------------------------------------------------------------------------
_HI_WALA = r"(?:वाला|वाले|वाली)"
_HI_NOUN = r"(?:बटन|विकल्प|आइकन|टैब|लिंक)"
_HI_ON = r"(?:पर|पे)"
_HI_CLICK = r"(?:क्लिक|टैप|सेलेक्ट|सिलेक्ट)"
_HI_DO = r"(?:करो|करें|करे|करिए|कीजिए|कीजिये|कर\s+दो|कर\s+दें|कर\s+दीजिए)"
_HI_PRESS = r"(?:दबाओ|दबाएँ|दबाएं|दबाइए|दबाइये|दबा\s+दो|दबा\s+दें|चुनो|चुनें|चुनिए|चुन\s+लो)"
_HI_STOP = r"\s*[।.!]*\s*$"
_TAIL_HI = re.compile(
    rf"\s*(?:{_HI_WALA}\s+)?(?:{_HI_NOUN}\s+)?(?:{_HI_ON}\s+)?(?:{_HI_CLICK}\s+{_HI_DO}|{_HI_PRESS}|{_HI_DO}){_HI_STOP}"
)
_HI_POINT = re.compile(
    rf"\s*(?:{_HI_WALA}\s+)?(?:{_HI_NOUN}\s+)?(?:{_HI_ON}\s+)?(?:{_HI_CLICK}\s+{_HI_DO}|{_HI_PRESS}){_HI_STOP}"
)
_LEFTOVER = re.compile(rf"\s+(?:{_HING_WALA}|{_HI_WALA})\s*$", re.I)

# ---- what is never a control's name --------------------------------------------------------------
# Keys belong to the hotkey tool ("press enter", "hit cmd+s"), not to a click.
_KEY = (
    r"(?:enter|return|tab|escape|esc|delete|backspace|space(?:\s*bar)?|up|down|left|right|home|end|all|"
    r"command|cmd|control|ctrl|option|opt|alt|shift|fn|caps\s+lock|page\s+(?:up|down)|f\d{1,2}|"
    r"(?:(?:cmd|command|ctrl|control|opt|option|alt|shift|fn)\s*[+-]?\s*)+\S)"
)
_KEY_NAME = re.compile(rf"^(?:the\s+)?{_KEY}(?:\s+(?:key|button))?(?:\s*(?:\+|and)\s*{_KEY})*$", re.I)
_VAGUE = frozenset(
    ["it", "this", "that", "these", "those", "them", "here", "there", "on", "in", "the", "a", "an", "one", "none", "me", "us", "you", "him", "her", "something", "anything", "everything", "nothing", "again", "away", "around", "through", "twice", "once", "text", "all", "yes", "no", "ok", "okay",
     # Hinglish and Hindi deictics: "is button pe click karo" names no control
     "is", "ye", "yeh", "isko", "ise", "wo", "woh", "usko", "yahan", "wahan", "idhar", "udhar",
     "इस", "यह", "ये", "इसे", "इसको", "वह", "वो", "उस", "यहाँ", "यहां", "वहाँ", "वहां"]
)
# A question about a control is conversation. English questions start with a wh-word or an
# auxiliary ("can you click …" is a request); Hinglish and Hindi ones carry kya / kaun / क्या …
# anywhere ("is button pe click karo" is a request: "is" is Hindi's "this").
_EN_QUESTION = re.compile(
    r"^\s*(?:what|what's|whats|which|where|why|how|who|when|is|are|does|do|did|can|could|should|will|would)\b"
    r"(?!\s+you\s+(?:please\s+)?" + _POINT_VERB + r")",
    re.I,
)
_HI_QUESTION = re.compile(
    r"(?<![\w\u0900-\u097f])(?:kya|kaun|kahan|kidhar|kyun|kaise|क्या|कौन|कहाँ|कहां|किधर|क्यों|कैसे)(?![\w\u0900-\u097f])",
    re.I,
)
_CLAUSES = re.compile(r"\s*(?:,|;|\band\s+then\b|\bthen\b|\band\b|\bafter\s+that\b|\bphir\b|\baur\b|और|फिर)\s+", re.I)

# Words that carry no name: verb endings and particles (Hindi), and English function words.
_HI_AUX = frozenset(
    ["करो", "करें", "करे", "कर", "करना", "कीजिए", "कीजिये", "करिए", "दो", "दें", "दे", "दीजिए", "जाओ", "जाएँ", "जाएं", "जाए", "लो", "लें", "ले", "है", "हैं", "को", "में", "पर", "पे", "से", "का", "की", "के", "यह", "ये", "इस", "वाला", "वाली", "वाले", "बटन", "दबाओ", "दबाएँ", "दबाएं", "दबाइए", "दबा", "चुनो", "चुनें", "चुनिए", "चुन", "खोलो", "खोलें", "दिखाओ", "दिखाएँ", "दिखाएं", "ज़रा", "जरा", "कृपया", "प्लीज़", "प्लीज", "अभी"]
)
_EN_FUNCTION = frozenset(
    ["the", "a", "an", "to", "my", "me", "please", "and", "of", "in", "on", "for", "that", "this", "is", "it", "i", "you", "your", "now", "just", "can", "could", "would", "will", "button", "link", "tab", "icon", "click", "tap", "press", "hit", "select", "choose", "pe", "par", "karo", "dabao", "wala", "wali", "kar", "do"]
)


def script(text: str) -> str:
    """'devanagari' when the text has any Devanagari, else 'latin' (neo.reflex.learn.script's tag)."""
    return "devanagari" if DEVANAGARI.search(text or "") else "latin"


def latin_only(text: str) -> bool:
    """Every letter is plain Latin (ASCII): the gate's English rules can read it as it is."""
    return all(ord(ch) < 128 or not ch.isalpha() for ch in unicodedata.normalize("NFC", text or ""))


def norm(text: str) -> str:
    """A label or phrase as compared: NFC, one space, lower case, no trailing ellipsis, colon or
    quotes (a menu's "Print…" is "print")."""
    s = unicodedata.normalize("NFC", str(text or "")).replace("‌", "").replace("‍", "")
    s = " ".join(s.split()).lower()
    s = s.strip(" \"'“”‘’`")
    s = re.sub(r"(?:…|\.\.\.|:)\s*$", "", s).strip(" \"'“”‘’`")
    return s


def strip_frame(text: str) -> str:
    """The control's name inside a request, with the words people wrap it in removed (English
    click verbs and control nouns, Hinglish "… pe click karo / dabao / wala button", Hindi "… पर
    क्लिक करो / दबाओ / चुनो"). The text itself (trimmed) when nothing is left."""
    raw = " ".join(unicodedata.normalize("NFC", str(text or "")).split())
    s = _LEAD_EN.sub("", raw, count=1)
    s = _TAIL_HI.sub("", s, count=1)
    s = _TAIL_HING.sub("", s, count=1)
    s = _TAIL_EN.sub("", s, count=1)
    s = _TAIL_FILLER.sub("", s, count=1)
    s = _LEFTOVER.sub("", s, count=1)
    s = re.sub(r"^(?:the|that|this)\s+", "", s, flags=re.I)
    s = s.strip(" ,.!?।\"'“”‘’")
    return s or raw.strip(" ,.!?।")


def is_key_name(phrase: str) -> bool:
    """"enter", "the escape key", "cmd+s", "command shift d": a key, not a control."""
    return bool(_KEY_NAME.match(" ".join((phrase or "").split())))


def _nameable(core: str) -> bool:
    words = core.lower().split()
    if not words or len(core) < 2 or len(words) > 7:
        return False
    return not all(w in _VAGUE for w in words) and not is_key_name(core)


def pointing(text: str) -> str | None:
    """The control's name when the request is "press that control" and nothing else, else None.

    Strict, unlike :func:`strip_frame`: an explicit pointing verb or frame in the user's words
    (English click/tap/press/hit/select/choose…, Hinglish "… pe click karo" / "… dabao" / "… wala
    button", Hindi "… पर क्लिक करो" / "… दबाओ" / "… चुनो"), one clause, not a question, and not a
    key ("press enter")."""
    raw = " ".join(unicodedata.normalize("NFC", str(text or "")).split())
    if not raw or raw.rstrip().endswith("?") or _HI_QUESTION.search(raw):
        return None
    lead = _LEAD_EN.match(raw)
    tail = _HING_POINT.search(raw) or _HI_POINT.search(raw)
    if not (lead or tail) or (not tail and _EN_QUESTION.match(raw)):
        return None
    core = strip_frame(raw)
    if _CLAUSES.search(core) or not _nameable(core):
        return None
    return core


def content_words(text: str) -> set[str]:
    """The words of a phrase that can name something: lower case, nukta folded (ज़ = ज), without
    Hindi verb endings and particles or English function words."""
    out: set[str] = set()
    for t in _TOKEN.findall(unicodedata.normalize("NFC", str(text or "")).lower()):
        if t in _HI_AUX or t in _EN_FUNCTION:
            continue
        out.add(t.replace("़", ""))
    return out
