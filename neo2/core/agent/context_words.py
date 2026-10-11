"""Do the user's words point at what is in front of them — what they selected, copied or are reading
("summarize what I copied", "translate the selected text", "ye copy kiya hua padho")? The words alone
decide, never a model's.

* :func:`wants` — the sentence asks about the selection, the clipboard, "this" text or the files
  selected in Finder. The one test for every path: the Session's hook (the agent gets ``in_front``),
  Gemini Live's handing over of the user's own sentence, and the tool's own origin rule
  (neo2.core.tools.context_now) — so what one of them reads, the others allow.
* :func:`kinds` / :func:`kind` — where the words say to look: "copied" / "my clipboard" → the
  clipboard, "selected" / "highlighted" → the selection, "these files" → Finder's selection. An
  explicit word wins over a model's choice, and only the user's own word ever opens the clipboard.

Specific and anchored, never a bare deictic: "what is this?" and "ye kya hai" stay the camera's
(fastpath.camera runs first, and Live's camera line says so), "what does this mean" and "read this"
could be about anything in view, "summarize this page" is page_read's, and "send this message" or
"close this document" are commands about an app, not a request to read anything. A "this" counts
only with a verb that reads text (summarize, translate, explain, proofread, rewrite…), naming text
("this paragraph", "this sentence"), or carried into an app ("add this to my calendar", "remind me
about this"). Only "this", never "that": "can you explain that?", "is it correct that…" and "add
that to my calendar" are about the conversation, not about anything in front of the user. Who said
the words is the caller's to check.
"""

from __future__ import annotations

import re

# ---- English ---------------------------------------------------------------------------------------
_I = r"i(?:'ve|\s+have)?\s+(?:just\s+)?"  # I / I've / I have, just
_COPIED = r"cop(?:ied|y)"  # "what did I copy" says it in the present
_PICKED = r"(?:selected|highlighted|select|highlight)"
# Text a user selects or copies, named as a thing.
_BIT = (
    r"(?:text|words?|link|url|address|number|code|line|lines|paragraph|para|sentence|passage|section|quote|"
    r"bit|part|stuff|thing|message|email|e-mail|list|table|names?|files?|items?|folders?|documents?)"
)
# Nouns of text: "this paragraph" is text in front of the user, whatever the verb.
_TEXT_NOUN = r"(?:text|paragraph|para|sentence|passage|word|words|wording|phrase|quote)"
# Verbs that read text: "<verb> this" is about text in front of the user.
_READ_VERB = (
    r"(?:summari[sz]e|translate|explain|proof-?read|rewrite|re-?phrase|re-?word|paraphrase|simplify|shorten|"
    r"tl;?\s*dr|fix\s+the\s+(?:grammar|spelling|typos?)\s+(?:in|of)|check\s+the\s+(?:grammar|spelling)\s+(?:in|of)|"
    r"correct|polish|improve)"
)
# What "this" may not be for a reading verb: the page, a video, a picture, the screen (other tools),
# or a thing held up to the camera.
_NOT_TEXT = (
    r"(?:page|web\s*page|tab|site|website|video|clip|song|music|track|picture|photo|image|pic|screenshot|screen|"
    r"camera|room|thing\s+i(?:'m|\s+am)\s+holding|object|label|sign|product|game|book|menu|receipt|card)"
)
_ENGLISH = [
    rf"\bwhat\s+{_I}(?:{_COPIED}|{_PICKED})\b",  # what I (just) copied / selected
    r"\bwhat\s+did\s+i\s+(?:just\s+)?(?:copy|select|highlight)\b",
    rf"\b{_BIT}\s+(?:that\s+|which\s+)?{_I}(?:copied|selected|highlighted)\b",  # the link I copied
    r"\b(?:my|the)\s+clipboard\b(?!\s+(?:app|manager|history|settings?)\b)",
    rf"\b(?:the\s+|my\s+)?(?:selected|highlighted|copied)\s+{_BIT}\b",  # the selected text
    r"\b(?:the|my|this)\s+selection\b(?!\s+(?:of|committee|process|criteria)\b)",
    rf"\bthis\s+{_TEXT_NOUN}\b",  # this paragraph, is this sentence correct
    rf"\b{_READ_VERB}\s+this\b(?!\s+(?:{_NOT_TEXT})\b)",  # translate this (email)
    r"\badd\s+this\s+to\s+(?:my\s+|the\s+)?(?:calendar|reminders?|notes?|to-?do(?:\s+list)?)\b",
    r"\bremind\s+me\s+(?:about|of)\s+this\b(?!\s+(?:page|video|song)\b)",
    r"\b(?:these|those|the\s+selected|selected)\s+(?:files|folders|items|documents)\b",
]

# ---- Hinglish ---------------------------------------------------------------------------------------
_KIYA = r"(?:kiya|kia|kara|kari|kiye|ki)"
_DO = r"(?:karo|kar\s*do|kardo|karna|kijiye|kariye|karke|kar\s+dijiye)"
_HINGLISH = [
    # ye / jo (maine) copy kiya (hua), maine jo copy kiya
    rf"\b(?:ye|yeh|jo|jo\s+bhi)\s+(?:maine\s+|mene\s+|main\s+ne\s+)?(?:abhi\s+)?(?:copy|select|highlight)\s+{_KIYA}\b",
    rf"\b(?:maine|mene|main\s+ne)\s+(?:jo\s+|abhi\s+)*(?:copy|select|highlight)\s+{_KIYA}\b",
    r"\b(?:copy|select|highlight)\s+(?:kiya|kia|kiye)\s+(?:hua|huwa|hue|huye)\b",
    r"\bclipboard\s+(?:mein|me|mai|main|pe|par|wala|waala|ka|ki)\b",
    rf"\b(?:is|iss|ye|yeh)\s+{_TEXT_NOUN}\s+(?:ko|ka|ki|ke|mein|me)\b",  # is paragraph ko summarize karo
    rf"\b(?:isko|isse|ise|isey|ye|yeh)\s+(?:translate|summari[sz]e|explain|proof-?read|rewrite|simplify)\s+{_DO}\b",
]

_ANY = re.compile("|".join(_ENGLISH + _HINGLISH), re.I)

# Where the words say to look, in this order when they say several ("the files I copied" are the
# clipboard's; "the selected files" are Finder's).
_FILES = re.compile(
    rf"\b(?:these|those|the\s+selected|selected)\s+(?:files|folders|items|documents)\b"
    rf"|\b(?:files?|folders?|items?|documents?)\s+(?:that\s+|which\s+)?{_I}(?:selected|highlighted)\b",
    re.I,
)
_CLIPBOARD = re.compile(
    rf"\bwhat\s+{_I}{_COPIED}\b|\bwhat\s+did\s+i\s+(?:just\s+)?copy\b|\b{_BIT}\s+(?:that\s+|which\s+)?{_I}copied\b"
    r"|\b(?:my|the)\s+clipboard\b|\bclipboard\s+(?:mein|me|mai|main|pe|par|wala|waala|ka|ki)\b"
    r"|\bcopied\s+" + _BIT + r"\b"
    rf"|\bcopy\s+{_KIYA}\b",
    re.I,
)
_SELECTION = re.compile(
    rf"\bwhat\s+{_I}(?:selected|highlighted)\b|\bwhat\s+did\s+i\s+(?:just\s+)?(?:select|highlight)\b"
    rf"|\b{_BIT}\s+(?:that\s+|which\s+)?{_I}(?:selected|highlighted)\b"
    rf"|\b(?:selected|highlighted)\s+{_BIT}\b|\b(?:the|my|this)\s+selection\b"
    rf"|\b(?:select|highlight)\s+{_KIYA}\b",
    re.I,
)


def _norm(text: str) -> str:
    return " ".join(str(text or "").replace("’", "'").split())


def wants(text: str) -> bool:
    """Do these words ask about what the user selected, copied or has in front of them?"""
    t = _norm(text)
    if not t or t.startswith("{"):  # (a call's own arguments, handed over as if said: never a sentence)
        return False
    return bool(_ANY.search(t))


def kinds(text: str) -> tuple[str, ...]:
    """Where the words say to look — "files", "clipboard", "selection" — each it names, in that order
    (empty: they only say "this", and the selection, else the window, is it)."""
    t = _norm(text)
    if not wants(t):
        return ()
    out: list[str] = []
    if _FILES.search(t):
        out.append("files")
    if _CLIPBOARD.search(t):
        out.append("clipboard")
    if _SELECTION.search(t):
        out.append("selection")
    return tuple(out)


def kind(text: str) -> str:
    """The first place the words name ("" = none: "translate this")."""
    k = kinds(text)
    return k[0] if k else ""
