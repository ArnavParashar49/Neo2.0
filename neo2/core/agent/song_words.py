"""Does the user want NEO to name the song that's playing? The words alone decide ("what song is
this", "ye kaunsa gaana hai", "यह कौन सा गाना है") — never a model's.

* :func:`wants_song` — the whole sentence asks NEO to listen and name the song playing now, in English,
  Hinglish or Hindi: "what's this song", "which song is playing", "name this song", "shazam this",
  "who sings this", "kaunsa gaana chal raha hai", "इस गाने का नाम क्या है".
* :func:`asks_about_song` — looser, for a model's call (the agent loop): the user's sentence is about
  which song something is at all ("what's the song playing in this video"). :func:`wants_song`
  always counts.
* :func:`wants_stop` — "stop listening", "sunna band karo", while NEO listens.

Never: "play this song", "play a song", "next song", "what songs do you know", "sing a song", and a
question about a song the user names ("who sings Blinding Lights"): that is a web question, not
listening. Whole-sentence matches only (after how NEO was addressed and a polite lead come off):
"remind me to ask what song this is" never listens. Who said the words is fastpath.song's to check.
"""

from __future__ import annotations

import re

_EDGE = " ,.;:!?।"
_ADDRESS = re.compile(r"^(?:(?:hey|hi|ok|okay|yo|arre|arey)\s+)?neo\b[\s,.:!-]*", re.I)
_LEAD = re.compile(
    r"^(?:(?:please|pls|so|ok|okay|hey|um+|uh+|and|acha|achha|accha|quick(?:ly)?)\s+)*"
    r"(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?(?:(?:tell|let)\s+me\s+(?:know\s+)?|check\s+|find\s+out\s+)?"
    r"|(?:tell|let)\s+me\s+(?:know\s+)?|(?:i\s+)?(?:wonder|want\s+to\s+know|wanna\s+know)\s+|guess\s+|(?:any\s+)?idea\s+)?",
    re.I,
)
_TAIL = re.compile(
    r"[\s,]*(?:please|pls|neo|for\s+me|quickly|yaar|yar|bhai|na|zara|ji|batao|bolo|bata\s+do|"
    r"(?:right\s+)?now|here|in\s+the\s+background|on\s+(?:the\s+)?(?:radio|tv|speakers?)|on\s+my\s+(?:mac|computer|laptop)|"
    r"that(?:'s|\s+is)\s+playing|which\s+is\s+playing|playing|on)$",
    re.I,
)

# ---- English -----------------------------------------------------------------------------------------
_SONG = r"(?:song|track|tune|music|melody)"
_THIS = r"(?:this|that|the)"
_IS = r"(?:'s|s|\s+is)"
_ENGLISH = [
    rf"what{_IS}\s+{_THIS}\s+{_SONG}(?:\s+called)?",  # what's this song, what is the song called
    rf"what{_IS}\s+{_THIS}\s+{_SONG}'s\s+name",
    rf"what{_IS}\s+the\s+name\s+of\s+{_THIS}\s+{_SONG}",
    rf"(?:what|which)\s+{_SONG}(?:'s|\s+is|\s+are\s+we\s+listening\s+to)(?:\s+(?:this|that|it))?(?:\s+called)?",  # what song is this ("what songs": never)
    rf"(?:what|which)\s+{_SONG}\s+(?:is\s+)?(?:this|that|it)\s+is",  # (tell me) what song this is
    rf"(?:what|which)\s+{_SONG}\s+(?:am\s+i|are\s+we)\s+(?:listening\s+to|hearing)",
    r"what(?:'s|s|\s+is|\s+am\s+i|\s+are\s+we)\s+(?:listening\s+to|hearing)",  # what am I listening to
    rf"what{_IS}\s+playing",  # (after "right now" / "on the radio" came off)
    rf"name\s+(?:this|that|the)\s+{_SONG}",  # name that tune
    rf"shazam(?:\s+(?:this|that|it)(?:\s+{_SONG})?|\s+the\s+{_SONG})?",  # shazam this
    rf"(?:identify|recogni[sz]e|detect)\s+(?:this|that|the)\s+{_SONG}",
    rf"(?:do|can|could)\s+you\s+(?:know|recogni[sz]e|identify|name|tell)\s+(?:this|that|the)\s+{_SONG}",
    rf"(?:do|can|could)\s+you\s+(?:know|tell|recogni[sz]e)\s+(?:what|which)\s+{_SONG}\s+(?:this|that|it)\s+is",
    rf"who\s+(?:sings|sang|is\s+singing|'s\s+singing|s\s+singing|performs)\s+(?:this|that|it)(?:\s+{_SONG})?",
    rf"who(?:'s|s|\s+is)\s+(?:the\s+)?(?:singer|artist)\s+(?:of|on|for)\s+(?:this|that)(?:\s+{_SONG})?",
    rf"whose\s+{_SONG}\s+is\s+(?:this|that|it)",
    rf"listen(?:\s+to\s+(?:this|that|it|the\s+{_SONG}))?(?:\s+and)?\s+(?:tell\s+me\s+)?(?:what|which)\s+{_SONG}\s+(?:this|it)\s+is",
    rf"listen(?:\s+to\s+(?:this|that|it|the\s+{_SONG}))?(?:\s+and)?\s+(?:identify|name|recogni[sz]e)\s+(?:it|this|the\s+{_SONG})",
]

# ---- Hinglish, as the transcript writes it -----------------------------------------------------------
_KAUN = r"(?:kaun\s*sa|kon\s*sa|kaunsa|konsa|kaun\s*se|kon\s*se|kaunse|konse)"
_GAANA = r"(?:gaana|gana|gaanaa|song|geet|track)"
_HAI = r"(?:hai|he|h|hain)"
_YE = r"(?:ye|yeh|yah|ya|wo|woh|vo)"
_CHAL = r"(?:chal|baj|bajj|play\s+ho)\s+(?:raha|rha|rahi|rhi)"
_HINGLISH = [
    rf"(?:{_YE}\s+)?{_KAUN}\s+{_GAANA}\s+{_HAI}(?:\s+{_YE})?",  # ye kaunsa gaana hai
    rf"(?:{_YE}\s+)?{_GAANA}\s+{_KAUN}\s+{_HAI}",  # ye gaana kaunsa hai
    rf"(?:{_YE}\s+)?{_KAUN}\s+{_GAANA}\s+{_CHAL}\s+{_HAI}",  # kaunsa gaana chal raha hai
    rf"(?:is|iss|{_YE})\s+(?:gaane|gane|gaana|song)\s+ka\s+naam\s+(?:kya\s+{_HAI}|batao|bolo|bata\s+do)",
    rf"(?:{_YE}\s+)?(?:kiska|kis\s+ka)\s+{_GAANA}\s+{_HAI}(?:\s+{_YE})?",  # ye kiska gaana hai
    rf"{_YE}\s+{_GAANA}\s+(?:kiska|kis\s+ka)\s+{_HAI}",
    rf"(?:{_YE}|is|iss)\s+(?:gaana|gaane|gana|gane|song)\s+(?:ko\s+)?(?:pehchano|pehchan\s+lo|pehchaan\s+ke\s+batao|identify\s+karo|shazam\s+karo)",
    r"shazam\s+karo",
]

# ---- Hindi (Devanagari): no \b — its vowel signs aren't word characters to `re` ---------------------
_H_KAUN = r"(?:कौन\s*सा|कौनसा|कौन\s*से|कौनसे)"
_H_GAANA = r"(?:गाना|गीत|सॉन्ग)"
_H_YE = r"(?:यह|ये|वो|वह)"
_HINDI = [
    rf"(?:{_H_YE}\s+)?{_H_KAUN}\s+{_H_GAANA}\s+(?:है|हैं)(?:\s+{_H_YE})?",  # यह कौन सा गाना है
    rf"(?:{_H_YE}\s+)?{_H_GAANA}\s+{_H_KAUN}\s+(?:है|हैं)",
    rf"(?:{_H_YE}\s+)?{_H_KAUN}\s+{_H_GAANA}\s+(?:चल|बज)\s+रहा\s+(?:है|हैं)",
    r"(?:इस|इसी)\s+गाने\s+का\s+नाम\s+(?:क्या\s+है|बताओ|बोलो)",
    rf"(?:{_H_YE}\s+)?(?:किसका|किस\s+का)\s+{_H_GAANA}\s+है",
    rf"{_H_YE}\s+(?:गाना|गाने\s+को)\s+पहचानो",
]

_ASKS = re.compile(r"^(?:" + "|".join(f"(?:{p})" for p in _ENGLISH + _HINGLISH) + r")$", re.I)
_ASKS_HI = re.compile(r"^(?:" + "|".join(f"(?:{p})" for p in _HINDI) + r")$")

# ---- looser: a model's call backed by the user's sentence -------------------------------------------
_SONG_WORD = re.compile(
    r"\b(?:songs?|track|tune|music|melody|gaan[ae]|gana|gane|geet|shazam)\b|गाना|गाने|गीत", re.I
)
_ASK_WORD = re.compile(
    r"\b(?:what|which|whose|who|name|identify|recogni[sz]e|kaun\w*|kon\w*|kis\w*|naam|pehch\w*|shazam)\b"
    r"|कौन|किस|नाम|पहचान",
    re.I,
)
# ...about a song playing now, not songs in general ("what song should I listen to", "the song of the year")
_HERE = re.compile(
    r"\b(?:this|that|it|playing|background|now|here|currently|on\s+(?:the\s+)?(?:radio|tv)|ye|yeh|yah|chal\w*|baj\w*)\b"
    r"|यह|ये|बज|चल",
    re.I,
)
_NOT = re.compile(
    r"^\s*(?:play|sing|next|skip|pause|resume|stop|add|like|save|download|queue|search|find\s+me|recommend|suggest)\b"
    r"|\bplay\s+(?:this|that|the|a|some|me|it)\b|\bsongs\b|\b(?:bajao|sunao|chalao|lagao)\b|बजाओ|सुनाओ|चलाओ",
    re.I,
)

# ---- stop listening ----------------------------------------------------------------------------------
_STOP = re.compile(
    r"^(?:(?:stop|quit|end|cancel)\s+(?:the\s+)?(?:listening|shazam(?:ing)?|recogni[sz]ing(?:\s+the\s+song)?)"
    r"|don'?t\s+listen|no\s+need\s+to\s+listen|sunna\s+band\s+kar(?:o|\s+do)?|mat\s+suno|सुनना\s+बंद\s+करो|मत\s+सुनो)$",
    re.I,
)


def _plain(text: str) -> str:
    """Lowercase, one space, how NEO was addressed and the edge punctuation off."""
    t = " ".join(str(text or "").replace("’", "'").split()).strip(_EDGE)
    return _ADDRESS.sub("", t, count=1).strip(_EDGE).lower()


def _cores(text: str) -> list[str]:
    """The sentence as said, then each time a trailing "right now"/"please"/"playing" comes off, each
    with and without its polite lead ("what's playing right now" → "what's playing" → "what's")."""
    t = _plain(text)
    out: list[str] = []
    for _ in range(4):  # "…playing right now, please?"
        out += [t, _LEAD.sub("", t, count=1).strip(_EDGE) or t]
        t2 = _TAIL.sub("", t).strip(_EDGE)
        if t2 == t or not t2:
            break
        t = t2
    return list(dict.fromkeys(out))


def wants_song(text: str) -> bool:
    """Do these words, as a whole sentence, ask NEO to name the song playing now?"""
    plain = _plain(text)
    if not plain or len(plain) > 160:
        return False
    return any(_ASKS.match(c) or _ASKS_HI.match(c) for c in _cores(text))


def asks_about_song(text: str) -> bool:
    """Is the user's sentence about which song something is (a model calling song_identify for it
    does what they asked)? :func:`wants_song` always is; "play a song", "what songs do you know"
    never are."""
    if wants_song(text):
        return True
    plain = _plain(text)
    if not plain or len(plain) > 200 or _NOT.search(plain):
        return False
    return bool(_SONG_WORD.search(plain) and _ASK_WORD.search(plain) and _HERE.search(plain))


def wants_stop(text: str) -> bool:
    """ "stop listening", "sunna band karo", "सुनना बंद करो"."""
    plain = _plain(text)
    return bool(plain) and len(plain) <= 60 and any(_STOP.match(c) for c in _cores(text))
