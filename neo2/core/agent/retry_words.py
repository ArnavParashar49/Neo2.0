"""Small, contextual retry utterances shared by NEO's follow-up lanes, and which lane one belongs to.

* :func:`is_retry_request` — "try again", "one more time", "phir se try karo", "arre neo dobara" for
  a lane: the camera's ("look again", "phir se dekho") or the song's ("listen again", "phir se
  suno"). Words that could mean anything after any request ("again", "repeat that", "redo", "do it
  again") are the camera's only: a song retry opens the mic, so it takes a clearer ask.
* :func:`mark` / :func:`last_lane` — the last retryable thing NEO did: a camera look, a song listen,
  or something else. A bare retry belongs to that one alone, so "try again" after a look and then
  an unrelated request retries nothing, and with both windows open the most recent one wins.

A retry phrase never authorizes the camera or the microphone by itself: the lane's own window
(camera_words.follow_up, song_words.retry_after_miss) has to be open too.
"""

from __future__ import annotations

import re
from typing import Literal

Lane = Literal["camera", "song", "other"]

# How NEO is addressed, the same for every lane ("arre neo" too): camera_words and song_words use it.
ADDRESS = re.compile(r"^(?:(?:hey|hi|ok|okay|yo|arre|arey)\s+)?neo\b[\s,.:!-]*", re.I)
_FILLER = r"(?:(?:please|pls|just|arre|arey|acha|achha|accha|ok|okay)[\s,]+)*"
_ASK = r"(?:(?:can|could|would|will)\s+you\s+)?" + _FILLER
_TAIL = r"(?:[\s,]+(?:please|pls|neo|yaar|na))*"
_IT = r"(?:\s+(?:it|that))?"

# Either lane: a clear request to do the last thing once more.
_EN_BOTH = (
    r"try\s+(?:again|it\s+again|that\s+again|once\s+more|one\s+more\s+time)|retry" + _IT
    + r"|one\s+more\s+time|once\s+more|once\s+again|give\s+it\s+another\s+(?:try|go|shot)"
)
# The camera's only: looking again, and the bare words a song retry must never take.
_EN_CAMERA = (
    r"try|again|redo" + _IT + r"|do\s+(?:it|that)\s+again|repeat\s+(?:it|that)"
    r"|look(?:\s+at\s+(?:it|that))?\s+(?:again|once\s+more)|take\s+another\s+look|check\s+again"
)
_EN_SONG = r"listen(?:\s+to\s+(?:it|that))?\s+(?:again|once\s+more|one\s+more\s+time)"
_REQUEST = {
    "camera": re.compile(rf"^{_ASK}(?:{_EN_BOTH}|{_EN_CAMERA}){_TAIL}$", re.I),
    "song": re.compile(rf"^{_ASK}(?:{_EN_BOTH}|{_EN_SONG}){_TAIL}$", re.I),
}

# Hinglish and Hindi: "phir se", "dobara", "ek baar aur", with an optional verb ("try karo",
# "karke dekho"); "dekho" is a look, "suno" a listen.
_HI_AGAIN = (
    r"ek\s+ba+r\s+(?:aur|phir(?:\s+se)?|fir(?:\s+se)?|dobara)|ek\s+aur\s+ba+r|phir\s+se|fir\s+se|dobara"
    r"|एक\s+बार\s+(?:और|फिर(?:\s+से)?|दोबारा)|फिर\s+से|दोबारा"
)
_HI_DO = (
    r"try\s+(?:karo|kar\s+do|karke\s+dekho|kar\s+ke\s+dekho)|karo|kar\s+do|karke\s+dekho|kar\s+ke\s+dekho"
    r"|koshish\s+karo|ट्राई\s+(?:करो|कर\s+दो)|करो|कर\s+दो|करके\s+देखो|कोशिश\s+करो"
)
_HI_LOOK = r"dekho|dekh\s+lo|देखो|देख\s+लो"
_HI_LISTEN = r"suno|sun\s+lo|सुनो|सुन\s+लो"
_RETRY_HI = {
    "camera": re.compile(rf"^{_FILLER}(?:{_HI_AGAIN})(?:\s+(?:{_HI_DO}|{_HI_LOOK}))?{_TAIL}$", re.I),
    "song": re.compile(rf"^{_FILLER}(?:{_HI_AGAIN})(?:\s+(?:{_HI_DO}|{_HI_LISTEN}))?{_TAIL}$", re.I),
}
_EDGE = " ,.;:!?।"

_last_lane: Lane | Literal[""] = ""


def is_retry_request(text: str, lane: Literal["camera", "song"]) -> bool:
    """Recognize a short request to repeat the immediately preceding action of ``lane``.

    Callers must also have fresh context for the action being retried (its window, and
    :func:`last_lane`). This deliberately does not authorize a camera or microphone by itself.
    """
    value = " ".join(str(text or "").replace("’", "'").split()).strip(_EDGE)
    value = ADDRESS.sub("", value, count=1).strip(_EDGE)
    if not value or len(value) > 80:
        return False
    return bool(_REQUEST[lane].fullmatch(value) or _RETRY_HI[lane].fullmatch(value))


def mark(lane: Lane) -> None:
    """``lane`` is now the last retryable thing NEO did: a look (camera_words.mark_look), a listen
    (song_words.begin_listen, mark_no_match), or another request of the user's ("other")."""
    global _last_lane
    _last_lane = lane


def last_lane() -> str:
    """"camera", "song", "other", or "" when nothing happened yet."""
    return _last_lane


def forget() -> None:
    """Nothing to retry (a test's fresh start)."""
    global _last_lane
    _last_lane = ""
