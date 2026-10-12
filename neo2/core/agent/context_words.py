"""Do the user's words point at what is in front of them — what they selected, copied or are reading
("summarize what I copied", "translate the selected text", "what does this say", "ye copy kiya hua
padho", "ये कॉपी किया हुआ पढ़ो")? The words alone decide, never a model's.

* :func:`wants` — the sentence asks about the selection, the clipboard, "this" text or the files
  selected in Finder. The one test for every path: the Session's hook (the agent gets ``in_front``),
  Gemini Live's handing over of the user's own sentence, and the tool's own origin rule
  (neo2.core.tools.context_now) — so what one of them reads, the others allow.
* :func:`kinds` / :func:`kind` — where the words say to look: "copied" / "my clipboard" → the
  clipboard, "selected" / "highlighted" → the selection, "these files" → Finder's selection. An
  explicit word wins over a model's choice, and only the user's own word ever opens the clipboard.
* :func:`names_text` — a fast path's free text that only *names* that text ("search what I copied on
  YouTube" → the query "what I copied"): the words to use are in front of the user, not in the
  sentence (neo2.core.agent.fastpath leaves such a step to a model that can read them).

**A request, anchored.** Each English cue is read from the start of the request — after how NEO was
addressed ("Neo,") and a polite lead ("please", "can you") come off — so the sentence must start as
a request: a verb ("summarize what I copied", "open the files I copied") or a question ("what did I
copy", "is this sentence correct"). A statement never is ("the selection was great", "these files
are too big", "what I copied yesterday was wrong"), and neither is editing the selection in its own
app ("copy the selected text", "paste what I copied", "delete these files"): nothing there for NEO
to read. Hinglish and Hindi say the verb last, so there the request is how the sentence ends
("… padho", "… batao", "… क्या है").

**"This" is narrow.** A "this" counts only with a verb that reads text (summarize, translate,
explain, proofread, rewrite, read…), naming text ("this paragraph", "this sentence"), asked about
("what does this say", "what's this error"), or carried into an app ("add this to my calendar",
"remind me about this") — and never:

* the camera's: a bare "what is this?" / "ye kya hai", or a thing held up ("read this label");
  fastpath.camera runs first, and Live's camera line says so;
* the conversation's: "explain this answer", "summarize this chat", "explain this in simpler words",
  "how can I improve this plan" (and "that" never: "can you explain that?" is about what was said);
* another tool's: "summarize this page" is page_read's, "send this message" and "close this
  document" are commands about an app;
* text the sentence carries itself: "translate this: hola", "add this to my to-do list: milk".

Who said the words is the caller's to check.
"""

from __future__ import annotations

import re
import unicodedata

from neo2.core.agent.retry_words import ADDRESS

# ---- how the request is framed --------------------------------------------------------------------
_EDGE = " ,.;!?।\"'“”"
_LEAD = re.compile(
    r"^(?:(?:please|pls|plz|kindly|so|ok|okay|and|now|also|then|hey|hi|yo|um+|uh+|acha|achha|accha|just|"
    r"quickly|quick)[\s,]+)*"
    r"(?:(?:can|could|would|will)\s+(?:you|u)\s+(?:please\s+|pls\s+|just\s+|quickly\s+|also\s+)*"
    r"|i\s+(?:want|need|would\s+like|'d\s+like)\s+(?:you\s+)?to\s+"
    r"|help\s+me\s+(?:to\s+)?)?",
    re.I,
)
_TAIL = re.compile(r"(?:[\s,]+(?:please|pls|neo|now|for\s+me|quickly|exactly|here|yaar|na))+$", re.I)

# How a statement starts: the sentence tells NEO something ("the selection was great", "I copied the
# wrong thing", "that's correct this time"); it doesn't ask for anything.
_STATEMENT = (
    r"(?:i|i'm|im|i've|i'd|i'll|my|mine|me|we|we're|we've|our|you|you're|youre|your|he|she|they|they're|it|"
    r"it's|its|there|there's|here|here's|the|a|an|this|these|those|that|that's|thats|his|her|their|some|all|"
    r"no|nope|not|yes|yeah|yep|sure|thanks|thank|cool|nice|great|awesome|wow|oh|ah|lol|haha|hmm+|because|"
    r"cause|cuz|but|if|when|while|since|maybe|actually|well|honestly|apparently|love|loved|hate|hated|"
    r"like|liked|what\s+i(?:'ve|\s+have)?)"
)
# Editing the selection (or the files) in its own app: the app does it, NEO reads nothing.
_EDIT = (
    r"(?:copy|cut|paste|delete|remove|erase|clear|bold|italici[sz]e|underline|strike|highlight|select|"
    r"deselect|unselect|move|drag|drop|indent|outdent|align|cent(?:er|re)|resize|undo|redo|close|print|"
    r"trash|duplicate|rename|zip|compress|colou?r|drag)"
)
_REQUEST = re.compile(rf"^(?!{_STATEMENT}\b)(?!{_EDIT}\b)\S", re.I)

# ---- English cues ------------------------------------------------------------------------------------
_I = r"i(?:'ve|\s+have)?\s+(?:just\s+)?"  # I / I've / I have, just
_COPIED = r"cop(?:ied|y)"  # "what did I copy" says it in the present
_PICKED = r"(?:selected|highlighted|select|highlight)"
# Text a user selects or copies, named as a thing.
_BIT = (
    r"(?:text|words?|link|url|address|number|code|line|lines|paragraph|para|sentence|passage|section|quote|"
    r"bit|part|stuff|thing|message|email|e-mail|list|table|names?|files?|items?|folders?|documents?)"
)
# What the user selected, copied or picked in Finder, by name: it is in front of them, whatever the
# verb ("summarize what I copied", "open the files I copied", "what are the selected files").
_OBJECT = [
    rf"\bwhat\s+{_I}(?:{_COPIED}|{_PICKED})\b",  # what I (just) copied / selected
    r"\bwhat\s+did\s+i\s+(?:just\s+)?(?:copy|select|highlight)\b",
    rf"\b{_BIT}\s+(?:that\s+|which\s+)?{_I}(?:copied|selected|highlighted)\b",  # the link I copied
    r"\b(?:my|the)\s+clipboard\b(?!\s+(?:app|manager|history|settings?)\b)",
    rf"\b(?:the\s+|my\s+)?(?:selected|highlighted|copied)\s+{_BIT}\b",  # the selected text
    r"\b(?:the|my|this)\s+selection\b(?!\s+(?:of|committee|process|criteria|round|test)\b)",
    r"\b(?:these|those|the\s+selected|selected)\s+(?:files|folders|items|documents)\b",
]
_OBJECT_RE = re.compile("|".join(_OBJECT), re.I)
# "the selected text, translate it": the thing first, then the request.
_FRONTED = re.compile(
    rf"^(?:what\s+{_I}(?:{_COPIED}|{_PICKED})|(?:the\s+|my\s+)?(?:selected|highlighted|copied)\s+{_BIT}"
    rf"|(?:the\s+)?{_BIT}\s+(?:that\s+|which\s+)?{_I}(?:copied|selected|highlighted))\s*[,;:—–-]+\s*(?P<rest>.+)$",
    re.I,
)

# Verbs that read text: "<verb> this" is about text in front of the user.
_READ_VERB = (
    r"(?:summari[sz]e|translate|explain|proof-?read|rewrite|re-?phrase|re-?word|paraphrase|tl;?\s*dr|"
    r"fix\s+the\s+(?:grammar|spelling|typos?)\s+(?:in|of)|check\s+the\s+(?:grammar|spelling)\s+(?:in|of)|"
    r"read(?:\s+out)?(?:\s+me)?)"
)
# What "this" is not, after a reading verb: the page, a video, a picture, the screen (other tools'),
# a thing held up to the camera — or the conversation itself (NEO's answer, the chat, a plan).
_NOT_TEXT = (
    r"(?:page|web\s*page|tab|site|website|video|clip|song|music|track|picture|photo|image|pic|screenshot|screen|"
    r"camera|room|thing\s+i(?:'m|\s+am)\s+holding|object|label|sign|product|game|book|menu|receipt|card|"
    r"answers?|repl(?:y|ies)|response|conversation|chat|discussion|plan|steps?|ideas?|concepts?|questions?|"
    r"explanation|topic|joke|time)"
)
# Said right after "this": the conversation's ("explain this again", "… in simpler words").
_AGAIN = (
    r"(?:again|once\s+more|in\s+(?:simpler|simple|easier|easy|plain|other|different|fewer|more|layman'?s?)\b|"
    r"more\s+simply|simpl(?:y|er)\b|better\b|differently|step\s+by\s+step|like\s+i(?:'?m|\s+am)\b|"
    r"(?:to|for)\s+(?:a|an)\s+(?:kid|child|beginner|\w+[\s-]+year[\s-]+old)\b)"
)
# Nouns of text: "this paragraph" is text in front of the user, whatever the verb.
_TEXT_NOUN = r"(?:text|paragraph|para|sentence|sentences|passage)\b(?!\s+messages?\b)"
# What a question about "this" asks it to say ("what does this error mean").
_READABLE = (
    r"(?:text|error|warning|alert|message|word|sentence|paragraph|line|note|code|notification|email|mail|"
    r"passage|phrase|quote)"
)
_THIS = [
    # translate this (email), summarize this, read this to me
    re.compile(
        rf"^(?P<verb>{_READ_VERB})\s+this\b(?!\s+(?:{_NOT_TEXT})\b)(?!(?:\s+to\s+me)?\s+{_AGAIN})", re.I
    ),
    # is this sentence correct, what does this paragraph say, fix this text
    re.compile(rf"^(?!{_EDIT}\b)\S.*?\bthis\s+{_TEXT_NOUN}", re.I),
    # what does this say / mean, what does this error mean
    re.compile(rf"^what\s+does\s+this(?:\s+{_READABLE})?\s+(?:say|mean)$", re.I),
    # what's this error, what is this message saying
    re.compile(rf"^what(?:'s|s|\s+is)\s+this\s+{_READABLE}(?:\s+(?:saying|about|say|mean))?$", re.I),
    # into an app
    re.compile(
        r"^(?:add|put|save)\s+this\s+(?:to|in|into|on)\s+(?:my\s+|the\s+)?"
        r"(?:calendar|reminders?|notes?|to-?do(?:\s+list)?)\b",
        re.I,
    ),
    re.compile(r"^remind\s+me\s+(?:about|of)\s+this\b(?!\s+(?:page|video|song)\b)", re.I),
]
# The text to work on comes in the sentence itself: "translate this: hola", "… "hola amigo"".
_SUPPLIED = re.compile(r"[:\"“”]\s*\S")
# "translate this to French hello how are you": a spoken sentence has no colon, but three words or
# more after the language that aren't the request going on ("and send it", "for me") are its text.
_TRANSLATE_REST = re.compile(
    r"^\s+(?:in)?to\s+\S+(?P<more>(?:\s+(?!and\b|then\b|please\b|for\b|so\b|now\b|na\b|yaar\b)\S+){3,})", re.I
)

# ---- Hinglish ----------------------------------------------------------------------------------------
_KIYA = r"(?:kiya|kia|kara|kari|kiye|ki)"
_DO = r"(?:karo|kar\s*do|kardo|karna|kijiye|kariye|karke|kar\s+dijiye)"
_TEXT_HI = r"(?:text|paragraph|para|sentence|passage|line|lines)"
_HINGLISH = [
    # ye / jo (maine) copy kiya (hua), maine jo copy kiya
    rf"\b(?:ye|yeh|jo|jo\s+bhi)\s+(?:maine\s+|mene\s+|main\s+ne\s+)?(?:abhi\s+)?(?:copy|select|highlight)\s+{_KIYA}\b",
    rf"\b(?:maine|mene|main\s+ne)\s+(?:jo\s+|abhi\s+)*(?:copy|select|highlight)\s+{_KIYA}\b",
    r"\b(?:copy|select|highlight)\s+(?:kiya|kia|kiye)\s+(?:hua|huwa|hue|huye)\b",
    r"\bclipboard\s+(?:mein|me|mai|main|pe|par|wala|waala|ka|ki)\b",
    rf"\b(?:is|iss|ye|yeh)\s+{_TEXT_HI}\s+(?:ko|ka|ki|ke|mein|me)\b",  # is paragraph ko summarize karo
    rf"\b(?:isko|isse|ise|isey|ye|yeh)\s+(?:translate|summari[sz]e|explain|proof-?read|rewrite|simplify)\s+{_DO}\b",
    # "ye kya likha hai", "isko padho": what is written here, read it
    r"^(?:ye|yeh|isme|ismein|is\s+mein)\s+(?:kya|kyaa)\s+likha\s+(?:hai|h|hua\s+hai)$",
    r"^(?:ye|yeh|isko|ise|isse)\s+(?:padho|padh\s*do|padh\s*ke\s+sunao|padhke\s+sunao|padh\s+kar\s+sunao)$",
]
_HINGLISH_RE = re.compile("|".join(_HINGLISH), re.I)
# The verb comes last: a request ends with one ("… padho", "… batao", "… kya hai").
_HI_ASKS = re.compile(
    rf"(?:{_DO}|do|dijiye|dena|padho|padh\s*do|sunao|batao|bata\s*do|bataiye|bolo|dikhao|samjhao|samjha\s*do|"
    r"likho|bhejo|(?:kya|kyaa)\s+(?:hai|h|likha\s+(?:hai|h))|kaisa\s+(?:hai|h)|sahi\s+(?:hai|h)|"
    r"likha\s+(?:hai|h)|sunao)(?:\s+(?:na|yaar|please|pls|jaldi|zara))*$",
    re.I,
)


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


# ---- Hindi (Devanagari): no \b — its vowel signs aren't word characters to `re` -------------------------
_COPY_HI = r"(?:कॉपी|कोपी)"
_PICK_HI = r"(?:सेलेक्ट|सिलेक्ट|हाइलाइट)"
_KIYA_HI = r"(?:किया|की|किए|किये)"
_HINDI = [
    rf"(?:ये|यह|इसे|इस|जो)\s+(?:मैंने\s+|मैने\s+)?(?:अभी\s+)?(?:{_COPY_HI}|{_PICK_HI})\s+{_KIYA_HI}",
    rf"(?:मैंने|मैने)\s+(?:जो\s+)?(?:अभी\s+)?(?:{_COPY_HI}|{_PICK_HI})\s+{_KIYA_HI}",
    rf"(?:{_COPY_HI}|{_PICK_HI})\s+(?:किया|की|किए)\s+(?:हुआ|हुई|हुए)",
    r"क्लिपबोर्ड\s+(?:में|मे|का|की|के|पे|पर)",
    r"(?:इसे|इसको|इस\s+(?:पैराग्राफ|टेक्स्ट|वाक्य|लाइन)\s+(?:को|का|की|के))\s+(?:\S+\s+)?"
    r"(?:ट्रांसलेट|अनुवाद|समराइज़|समराइज|सारांश|समझाओ|समझा|पढ़ो|पढ़कर|पढ़)",
    r"^(?:ये|यह|इसमें|इस\s+में)\s+क्या\s+लिखा\s+(?:है|हैं|हुआ\s+है)$",
    r"^(?:ये|यह|इसे|इसको)\s+(?:पढ़ो|पढ़\s+दो|पढ़कर\s+सुनाओ|पढ़\s+के\s+सुनाओ)$",
]
_HINDI_RE = re.compile(_nfc("|".join(_HINDI)))
_HINDI_ASKS = re.compile(
    _nfc(
        r"(?:करो|कर\s+दो|कीजिए|कीजिये|करें|दो|दीजिए|पढ़ो|पढ़\s+दो|सुनाओ|बताओ|बता\s+दो|बताइए|बताइये|दिखाओ|"
        r"समझाओ|समझा\s+दो|लिखो|भेजो|क्या\s+है|क्या\s+हैं|लिखा\s+है)(?:\s+(?:ना|न|यार|जल्दी|ज़रा))*$"
    )
)

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
    rf"|\bcopy\s+{_KIYA}\b" + _nfc(rf"|{_COPY_HI}|क्लिपबोर्ड"),
    re.I,
)
_SELECTION = re.compile(
    rf"\bwhat\s+{_I}(?:selected|highlighted)\b|\bwhat\s+did\s+i\s+(?:just\s+)?(?:select|highlight)\b"
    rf"|\b{_BIT}\s+(?:that\s+|which\s+)?{_I}(?:selected|highlighted)\b"
    rf"|\b(?:selected|highlighted)\s+{_BIT}\b|\b(?:the|my|this)\s+selection\b"
    rf"|\b(?:select|highlight)\s+{_KIYA}\b" + _nfc(rf"|{_PICK_HI}"),
    re.I,
)


def _norm(text: str) -> str:
    return _nfc(" ".join(str(text or "").replace("’", "'").split()))


def _request(text: str) -> str:
    """The request itself: how NEO was addressed, a polite lead and a trailing "please" off."""
    t = ADDRESS.sub("", text.strip(_EDGE), count=1).strip(_EDGE)
    t = _LEAD.sub("", t, count=1).strip(_EDGE)
    return _TAIL.sub("", t).strip(_EDGE)


def _asks_this(body: str) -> bool:
    """An English request about "this" text (the selection, else the window)."""
    for i, pattern in enumerate(_THIS):
        m = pattern.search(body)
        if m is None:
            continue
        rest = body[m.end() :]
        if _SUPPLIED.search(rest):
            return False  # "translate this: hola" — the text is in the sentence
        verb = (m.groupdict().get("verb") or "").lower() if i == 0 else ""
        # "translate this to French hello how are you": the words to translate follow the language
        return not (verb.startswith("translate") and _TRANSLATE_REST.match(rest))
    return False


def _asks_english(body: str) -> bool:
    if _REQUEST.match(body) and (_OBJECT_RE.search(body) or _asks_this(body)):
        return True
    m = _FRONTED.match(body)  # "the selected text, translate it"
    if m is not None:
        rest = _request(m["rest"])
        return bool(rest) and bool(_REQUEST.match(rest))
    return False


def wants(text: str) -> bool:
    """Do these words ask about what the user selected, copied or has in front of them?"""
    t = _norm(text)
    if not t or t.startswith("{"):  # (a call's own arguments, handed over as if said: never a sentence)
        return False
    body = _request(t)
    if not body:
        return False
    if _asks_english(body):
        return True
    plain = body.strip(_EDGE)
    if _HINGLISH_RE.search(plain) and (_HI_ASKS.search(plain) or plain.endswith("?")):
        return True
    return bool(_HINDI_RE.search(plain) and (_HINDI_ASKS.search(plain) or plain.endswith("?")))


def names_text(value: str) -> bool:
    """Is this free text (a fast path's capture: a query, a song, an app) only a name for what the
    user selected or copied ("what I copied", "the selected text", "ye copy kiya hua")? Then the
    words to use aren't in it."""
    v = _norm(value)
    return bool(v) and bool(_OBJECT_RE.search(v) or _HINGLISH_RE.search(v) or _HINDI_RE.search(v))


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
