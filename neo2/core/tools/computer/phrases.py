"""The words of NEO's computer tools' fast paths: the regexes that turn a sentence the user said
into a tool call with no model — scroll, type, keys, click by label. Data only: the tools that use
them (tools_pointer, tools_keys, tools_label) look them up here. Then, "by meaning": how people point
at an on-screen control (the words around its name, in English, Hinglish and Hindi), for the
resolver (resolve.py) and the by-utterance lane (neo2.core.agent.fastpath.by_utterance)."""

from __future__ import annotations

import re
import unicodedata

from neo2.core.config import decide_mode

_TAIL = r"\s*[.!]*\s*$"
_LEAD = r"^\s*(?:and\s+|now\s+|please\s+|then\s+)*(?:can\s+you\s+|could\s+you\s+)?"
# Voice scroll at the pointer. "scroll down", "scroll down a bit", "scroll up some more" — but a
# place ("scroll down in Safari", "scroll to the comments") or a task ("… and tell me") goes to
# the agent, which can scroll the right window.
_SCROLL_CTX = r"(?:\s+(?:a\s+(?:bit|little|lot)|some|more|further|again|a\s+page|please|now))*"
_SCROLL = [
    (rf"{_LEAD}scroll\s+down{_SCROLL_CTX}{_TAIL}", {"at_pointer": True, "dy": -8}),
    (rf"{_LEAD}scroll\s+up{_SCROLL_CTX}{_TAIL}", {"at_pointer": True, "dy": 8}),
    (rf"{_LEAD}page\s+down{_TAIL}", {"at_pointer": True, "dy": -30}),
    (rf"{_LEAD}page\s+up{_TAIL}", {"at_pointer": True, "dy": 30}),
]


# ---- dictation lane -------------------------------------------------------------------------
# "type Laptops in the heading", "new line", "type some points about laptops": the words *are*
# the payload, so no model is needed to understand them. These regexes make typing and keys
# zero-LLM fast paths (the reflex still gates them), and composing a brief is one small chat
# call plus typing — never the multi-step agent loop. Anything ambiguous is left to a model:
# a pronoun ("write that down"), a message to someone ("write an email to Sam"), a thing
# ("write the report"). The model knows what "that" is; a regex doesn't.

# "type" is unambiguous. "write" is dictation only for short or quoted text, or with a place
# ("write buy milk in the note"). "put", "enter", "insert" are task verbs.
_TYPE_VERB = r"type"
_WRITE_VERB = r"(?:jot\s+down|write\s+down|write)"
_NOT_DICTATION = (
    # a message to someone is a task for Mail/Messages
    r"(?!(?:an?\s+|the\s+|new\s+|my\s+)*(?:e-?mails?|mails?|messages?|texts?|letters?|repl(?:y|ies)|responses?|dms?|notes?)\s+to\b)"
    r"(?!(?:an?\s+|new\s+)*(?:e-?mails?|mail|messages?|letters?|dms?)\b)"
    r"(?!to\s+\w+)"
    r"(?!(?:me|us)\b)"  # "write me a poem" wants an answer, not dictation
    # pronouns and particles: "write that down", "type it in", "type up the notes"
    r"(?!(?:it|this|that|these|those|them|what|whatever|everything|up|in|out|down|over|back)\b)"
)
_WRITE_NOT_THING = r"(?!(?:the|my|our|his|her|their|your|a|an)\b)"  # "write the report" is a task
_COUNT_NOUN = (
    r"(?:bullet\s+)?(?:points?|bullets?|reasons?|ideas?|lines?|sentences?|tips?|ways?|things?|items?|"
    r"facts?|examples?|questions?|paragraphs?|options?|steps?|names?|titles?|headlines?|taglines?|"
    r"slogans?|jokes?|thoughts?|notes?|words?|pros?|cons?|benefits?|features?)"
)
# Payloads that are a *brief* to write, not text to type verbatim.
_COMPOSE_CUE = (
    r"(?:(?:some|a\s+few|several|a\s+couple(?:\s+of)?|two|three|four|five|six|seven|eight|nine|ten|\d+)"
    rf"\s+(?:\w+\s+){{0,2}}{_COUNT_NOUN}\b"
    rf"|{_COUNT_NOUN}\s+(?:about|on|for|of|related\s+to|regarding)\b"
    r"|an?\s+(?:short|quick|brief|long|nice|polite|formal|casual|simple|small|little|funny|good)?\s*"
    r"(?:list|paragraph|summary|essay|poem|story|sentence|description|caption|bio|intro|introduction|"
    r"outline|recipe|plan|draft|note|haiku|joke|title|heading|tagline|slogan|toast|limerick)s?\b"
    r"|something\b|anything\b)"
)
# Apps a dictation may name as its place (patterns run case-insensitively, so no capital trick).
_APP = (
    r"(?:the\s+)?(?:notes|textedit|pages|numbers|keynote|word|excel|docs|google\s+docs|sheets|safari|"
    r"chrome|firefox|mail|messages|slack|discord|whatsapp|telegram|terminal|vs\s*code|xcode|notion|"
    r"obsidian|reminders|stickies|freeform)(?:\s+app)?"
)
# UI places only (never "the box", "the page", "the line" — those occur in dictated prose).
_PLACE = (
    r"(?:heading|title|header|subject(?:\s+line)?|search\s+(?:bar|box|field)|address\s+bar|url\s+bar|"
    r"body|(?:text\s+)?field|text\s+box|input(?:\s+box)?|editor|cell|note|document|form|"
    r"first\s+line|comment\s+box|message\s+box|chat\s+box)"
)
# "… in the heading", "… into the search bar", "… as the title of the current note (in Notes)".
_TYPE_TARGET = (
    rf"(?:\s+(?:in|into|as|for|on)\s+(?:the|this|my)\s+(?:current\s+|open\s+|new\s+|same\s+)?{_PLACE}"
    r"(?:\s+(?:of|in)\s+(?:the|this|my)\s+(?:current\s+|open\s+|new\s+)?(?:note|document|page|file|email|message|window))?"
    rf"(?:\s+in\s+{_APP})?"
    rf"|\s+in\s+{_APP})?"
)
# "type in the search bar hello": the place may come first as well as last.
_TYPE_PLACE_FIRST = rf"(?:(?:in|into|on)\s+(?:the|this|my)\s+{_PLACE}\s+)?"
# A quoted payload ends at its closing quote: type 'Laptops' in the heading of the note in Notes.
_PAYLOAD = r"(?P<text>\"[^\"]+\"|“[^”]+”|'[^']+'|.+?)"
_SHORT_PAYLOAD = r"(?P<text>\"[^\"]+\"|“[^”]+”|'[^']+'|\S+(?:\s+\S+){0,5}?)"
_LITERAL_TYPE = (
    rf"{_LEAD}{_TYPE_VERB}\s+(?:the\s+(?:words?|text)\s+)?{_TYPE_PLACE_FIRST}{_NOT_DICTATION}"
    rf"(?!{_COMPOSE_CUE}){_PAYLOAD}{_TYPE_TARGET}{_TAIL}"
)
_LITERAL_WRITE = (
    rf"{_LEAD}{_WRITE_VERB}\s+(?:the\s+(?:words?|text)\s+)?{_TYPE_PLACE_FIRST}{_NOT_DICTATION}"
    rf"{_WRITE_NOT_THING}(?!{_COMPOSE_CUE}){_SHORT_PAYLOAD}{_TYPE_TARGET}{_TAIL}"
)
_COMPOSE_TYPE = (
    rf"{_LEAD}(?:{_TYPE_VERB}|{_WRITE_VERB}|draft|compose)\s+{_TYPE_PLACE_FIRST}{_NOT_DICTATION}"
    rf"(?P<brief>{_COMPOSE_CUE}.*?){_TYPE_TARGET}{_TAIL}"
)


_HOTKEYS: list[tuple[str, dict]] = [
    (
        rf"{_LEAD}(?:press|hit|push)\s+(?:the\s+)?(?P<keys>enter|return|tab|escape|esc|delete|"
        rf"backspace|space|up|down|left|right|home|end)(?:\s+(?:key|arrow|bar))?{_TAIL}",
        {"keys": "<keys>"},
    ),
    (rf"{_LEAD}(?:new|next)\s+line{_TAIL}", {"keys": "return"}),
    (rf"{_LEAD}select\s+all{_TAIL}", {"keys": "cmd+a"}),
    # undo/redo only right after NEO itself typed or pressed something
    (rf"{_LEAD}(?:undo(?:\s+that)?|scratch\s+that){_TAIL}", {"keys": "cmd+z", "after": "input"}),
    (rf"{_LEAD}redo(?:\s+that)?{_TAIL}", {"keys": "cmd+shift+z", "after": "input"}),
    (rf"{_LEAD}copy\s+(?:that|it|this){_TAIL}", {"keys": "cmd+c"}),
    (rf"{_LEAD}paste(?:\s+(?:that|it|this))?{_TAIL}", {"keys": "cmd+v"}),
    (rf"{_LEAD}save(?:\s+(?:that|it|this|the\s+(?:file|note|document)))?{_TAIL}", {"keys": "cmd+s"}),
    # the notes app this user uses (by evidence) while write_note is on; else the legacy guard
    (
        rf"{_LEAD}(?:make\s+|create\s+|start\s+|open\s+)?(?:a\s+)?new\s+note{_TAIL}",
        {"keys": "cmd+n", "app": "note"},
        lambda f: decide_mode("write_note") == "on",
    ),
    (rf"{_LEAD}(?:make\s+|create\s+|start\s+|open\s+)?(?:a\s+)?new\s+note{_TAIL}", {"keys": "cmd+n", "app": "Notes"}),
    (rf"{_LEAD}(?:make\s+|create\s+|start\s+|open\s+)?(?:a\s+)?new\s+document{_TAIL}", {"keys": "cmd+n", "app": "document"}),
    (rf"{_LEAD}(?:open\s+)?(?:a\s+)?new\s+window{_TAIL}", {"keys": "cmd+n"}),
    (rf"{_LEAD}(?:open\s+)?(?:a\s+)?new\s+tab{_TAIL}", {"keys": "cmd+t", "app": "tabs"}),
    (rf"{_LEAD}close\s+(?:this|the|that)\s+(?:window|tab|note|document)(?:\s+please)?{_TAIL}", {"keys": "cmd+w"}),
    (rf"{_LEAD}(?:go\s+to\s+the\s+)?next\s+tab{_TAIL}", {"keys": "ctrl+tab"}),
    (rf"{_LEAD}(?:go\s+to\s+the\s+)?(?:previous|last|prev)\s+tab{_TAIL}", {"keys": "ctrl+shift+tab"}),
    # a browser's own undo for tabs: ⌘⇧T brings back what was closed, most recent first
    (
        rf"{_LEAD}(?:re-?open|bring\s+back|restore|undo\s+clos(?:e|ing))\s+(?:the\s+)?(?:last\s+|previous\s+|recently\s+)?"
        rf"(?:closed\s+)?tab(?:\s+(?:i|you)\s+(?:just\s+)?closed)?{_TAIL}",
        {"keys": "cmd+shift+t", "app": "browser"},
    ),
    *[
        (rf"{_LEAD}(?:go\s+to|switch\s+to|open)\s+(?:the\s+)?(?:{w})\s+tab{_TAIL}", {"keys": f"cmd+{i}", "app": "browser"})
        for i, w in enumerate(
            ("first|1st", "second|2nd", "third|3rd", "fourth|4th", "fifth|5th", "sixth|6th", "seventh|7th", "eighth|8th"), 1
        )
    ],
    *[
        (rf"{_LEAD}(?:go\s+to|switch\s+to)\s+tab\s+(?:number\s+)?{i}{_TAIL}", {"keys": f"cmd+{i}", "app": "browser"})
        for i in range(1, 9)
    ],
    (rf"{_LEAD}(?:open\s+)?(?:an?\s+)?(?:new\s+)?(?:private|incognito)\s+(?:window|tab){_TAIL}", {"keys": "cmd+shift+n", "app": "browser"}),
    (rf"{_LEAD}close\s+(?:all\s+(?:the\s+)?tabs|the\s+whole\s+window|this\s+whole\s+window){_TAIL}", {"keys": "cmd+shift+w", "app": "browser"}),
    (rf"{_LEAD}(?:go\s+to|focus|select|click)\s+(?:on\s+)?(?:the\s+)?(?:address|url|search)\s+bar{_TAIL}", {"keys": "cmd+l", "app": "browser"}),
    (rf"{_LEAD}bookmark\s+(?:this|the)\s+(?:page|tab|site){_TAIL}", {"keys": "cmd+d", "app": "browser"}),
    (
        rf"{_LEAD}(?:hard\s+(?:reload|refresh)|(?:reload|refresh)\s+(?:the\s+page\s+)?without\s+(?:the\s+)?cache){_TAIL}",
        {"keys": "cmd+shift+r", "app": "browser"},
    ),
    # Hinglish
    (rf"{_LEAD}(?:ek\s+)?naya\s+tab\s+(?:kholo|khol\s+do|open\s+karo){_TAIL}", {"keys": "cmd+t", "app": "tabs"}),
    (rf"{_LEAD}(?:ye|yeh|is|iss)\s+tab\s+(?:ko\s+)?(?:band\s+karo|band\s+kar\s+do|close\s+karo|close\s+kar\s+do){_TAIL}", {"keys": "cmd+w"}),
    (
        rf"{_LEAD}(?:(?:band|close)\s+(?:kiya\s+hua|hua)|pichla\s+band|last\s+closed)\s+tab\s+(?:wapas|phir\s+se)\s+"
        rf"(?:kholo|khol\s+do|lao|le\s+aao){_TAIL}",
        {"keys": "cmd+shift+t", "app": "browser"},
    ),
    (rf"{_LEAD}agla\s+tab(?:\s+(?:kholo|dikhao|pe\s+jao))?{_TAIL}", {"keys": "ctrl+tab"}),
    (rf"{_LEAD}pichla\s+tab(?:\s+(?:kholo|dikhao|pe\s+jao))?{_TAIL}", {"keys": "ctrl+shift+tab"}),
    (rf"{_LEAD}(?:page\s+)?(?:refresh|reload)\s+(?:karo|kar\s+do){_TAIL}", {"keys": "cmd+r", "app": "browser"}),
    (rf"{_LEAD}go\s+back(?:\s+a\s+page)?{_TAIL}", {"keys": "cmd+[", "app": "browser|finder"}),
    (rf"{_LEAD}go\s+forward(?:\s+a\s+page)?{_TAIL}", {"keys": "cmd+]", "app": "browser|finder"}),
    (rf"{_LEAD}(?:reload|refresh)(?:\s+(?:the|this)\s+page)?{_TAIL}", {"keys": "cmd+r", "app": "browser"}),
    (rf"{_LEAD}zoom\s+in{_TAIL}", {"keys": "cmd+="}),
    (rf"{_LEAD}zoom\s+out{_TAIL}", {"keys": "cmd+-"}),
    (rf"{_LEAD}(?:reset\s+(?:the\s+)?zoom|actual\s+size){_TAIL}", {"keys": "cmd+0"}),
    (rf"{_LEAD}(?:(?:enter|exit|leave|toggle)\s+)?full\s*screen{_TAIL}", {"keys": "ctrl+cmd+f"}),
    (rf"{_LEAD}minimi[sz]e(?:\s+(?:this|the|that)\s+window)?{_TAIL}", {"keys": "cmd+m"}),
    (rf"{_LEAD}hide\s+(?:this|it|that|the\s+app|this\s+app){_TAIL}", {"keys": "cmd+h"}),
    (rf"{_LEAD}(?:find|search)\s+(?:on|in)\s+(?:this|the)\s+page{_TAIL}", {"keys": "cmd+f"}),
    (rf"{_LEAD}(?:take|grab|capture)\s+(?:a\s+)?screenshot(?:\s+of\s+(?:this|the\s+screen|my\s+screen)?)?{_TAIL}", {"keys": "cmd+shift+3"}),
    (rf"{_LEAD}screenshot(?:\s+(?:this|the\s+screen|my\s+screen))?{_TAIL}", {"keys": "cmd+shift+3"}),
    (rf"{_LEAD}scroll\s+to\s+(?:the\s+)?top{_TAIL}", {"keys": "cmd+up", "app": "browser"}),
    (rf"{_LEAD}scroll\s+to\s+(?:the\s+)?bottom{_TAIL}", {"keys": "cmd+down", "app": "browser"}),
    (rf"{_LEAD}(?:open\s+)?spotlight{_TAIL}", {"keys": "cmd+space"}),
    (rf"{_LEAD}(?:quit|close)\s+(?:this|the|the\s+current)\s+app{_TAIL}", {"keys": "cmd+q"}),
    (rf"{_LEAD}(?:switch|next)\s+app{_TAIL}", {"keys": "cmd+tab"}),
]

# ---- click by label -----------------------------------------------------------------------------
_KEYNAMES = (
    r"(?:enter|return|tab|escape|esc|delete|backspace|space(?:\s*bar)?|up|down|left|right|home|end|all|"
    r"command|cmd|control|ctrl|option|alt|shift|fn|caps\s+lock|page\s+(?:up|down)|f\d+)"
)
_VAGUE_LABELS = {
    "it", "this", "that", "these", "those", "them", "here", "there", "on", "in", "the", "a", "an",
    "one", "none", "me", "us", "you", "him", "her", "something", "anything", "everything",
    "nothing", "wisely", "again", "away", "around", "through", "twice", "once",
}
_CONTROL_NOUN = r"(?:button|link|tab|icon|menu\s+item|menu|option|checkbox|item)"
_NOT_VAGUE = rf"(?!(?:on\s+)?(?:the\s+)?(?:{'|'.join(sorted(_VAGUE_LABELS))})\s*[.!]*\s*$)"
_CLICK_PATHS: list[tuple[str, dict]] = [
    # quoted: click "Sign in"
    (rf"{_LEAD}(?:click|tap|press|hit|select|choose)\s+(?:on\s+)?(?:the\s+)?[\"“'](?P<label>[^\"”']+)[\"”'](?:\s+{_CONTROL_NOUN})?{_TAIL}", {"label": "<label>"}),
    # click/tap <label> [button] — no key names, no "and …" (that's two things)
    (
        rf"{_LEAD}(?:click|tap)\s+{_NOT_VAGUE}(?:on\s+)?(?:the\s+)?(?!(?:the\s+)?{_KEYNAMES}\b)(?!.*\b(?:and|then)\b)"
        rf"(?P<label>[\w'’&.+-]+(?:\s+[\w'’&.+-]+){{0,4}}?)(?:\s+{_CONTROL_NOUN})?{_TAIL}",
        {"label": "<label>"},
    ),
    # press/hit/select/choose only with a control noun: "press the send button"
    (
        rf"{_LEAD}(?:press|hit|select|choose)\s+(?:on\s+)?(?:the\s+)?(?!(?:the\s+)?{_KEYNAMES}\b)(?!.*\b(?:and|then)\b)"
        rf"(?P<label>[\w'’&.+-]+(?:\s+[\w'’&.+-]+){{0,4}}?)\s+{_CONTROL_NOUN}{_TAIL}",
        {"label": "<label>"},
    ),
]


# ---- by meaning -------------------------------------------------------------------------------
# How people point at an on-screen control: the words around its name, in English, Hinglish and
# Hindi. A language table, never app names or sites.
#
# * :func:`strip_frame` — the control's name inside a request: "click on the Start button" →
#   "Start", "new game pe click karo" → "new game", "नया गेम वाला बटन दबाओ" → "नया गेम". Generous: the
#   resolver (neo2.core.tools.computer.resolve) only acts on what then matches a real on-screen label.
# * :func:`frames` — the same name at each depth of stripping, the words as said first: a label can
#   hold a frame word ("New Tab", "Copy Link", "Try Again", "Select All"), so the resolver tries the
#   least stripped form that names a control before the bare name ("click new tab" is "New Tab" when
#   there is one, "New" only when there isn't).
# * :func:`pointing` — strict: the request *is* "press that control" (a click / tap / press verb, a
#   Hinglish "… pe click karo / dabao", a Hindi "… पर क्लिक करो / दबाओ / चुनो"), one clause, not a
#   question and not a key ("press enter" is the hotkey tool's). The reflex tool lane
#   (neo2.core.agent.fastpath.by_utterance) sends only these to a by-utterance tool on its own.
# * :func:`content_words` — a request's or label's words minus the verb endings and particles that
#   carry no name (करो / करें / दो / पर / को …, "the", "please" …), for the OS-strings match.
#
# Screen labels are data: nothing here reads them as instructions, they are only compared with the
# user's own words.

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

# ---- the words around a name that are never part of one (for :func:`frames`) ---------------------
# A trailing click verb (Hinglish, Hindi) with a "wala" and a plain "button" / "बटन" before it, an
# English "button" / "btn", and please-like fillers. Never a control noun that labels hold ("tab",
# "link", "item", "menu"…) or a filler that can end a label ("now", "again", "here").
_ARTICLE = re.compile(r"^(?:the|that|this)\s+", re.I)
_FRAME_FILLER = r"(?:[\s,]+(?:please|pls|plz|for\s+me|na|yaar))*\s*[.!।]*\s*$"
_TAIL_HI_VERB = re.compile(
    rf"\s*(?:{_HI_WALA}\s+)?(?:बटन\s+)?(?:{_HI_ON}\s+)?(?:{_HI_CLICK}\s+{_HI_DO}|{_HI_PRESS}|{_HI_DO}){_HI_STOP}"
)
_TAIL_HING_VERB = re.compile(
    rf"\s+(?:{_HING_WALA}\s+)?(?:(?:button|btn)\s+)?(?:{_HING_ON}\s+)?(?:{_HING_ACT}\s+)?{_HING_DO}{_END_FILLER}", re.I
)
_TAIL_BUTTON = re.compile(rf"\s+(?:button|btn){_FRAME_FILLER}", re.I)
_TAIL_FRAME_FILLER = re.compile(_FRAME_FILLER, re.I)
_PUNCT = " ,.!?।\"'“”‘’"

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
    """'devanagari' when the text has any Devanagari, else 'latin' (neo2.core.reflex.learn.script's tag)."""
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


def nameable(core: str) -> bool:
    """Could these words be a control's name: 1-7 words, not only deictics or fillers ("it",
    "this", "ye"), and not a key."""
    return _nameable(core)


def _nameable(core: str) -> bool:
    words = core.lower().split()
    if not words or len(core) < 2 or len(words) > 7:
        return False
    return not all(w in _VAGUE for w in words) and not is_key_name(core)


def is_question(text: str) -> bool:
    """A question, not a request: a trailing "?", a Hinglish or Hindi question word anywhere (kya,
    kaun, क्या…), or an English wh-word or auxiliary first ("can you click …" is a request, and a
    Hinglish click frame's "is" is "this")."""
    raw = " ".join(unicodedata.normalize("NFC", str(text or "")).split())
    if raw.endswith("?") or _HI_QUESTION.search(raw):
        return True
    return bool(_EN_QUESTION.match(raw)) and not (_HING_POINT.search(raw) or _HI_POINT.search(raw))


def frames(text: str) -> list[tuple[str, bool]]:
    """The control's name at each depth of stripping, the least stripped first, normalised
    (:func:`norm`) and without repeats: the words as said; without the lead click verb; without the
    words that are never a name's (a trailing click verb, "wala", "button", "please"); without a
    filler that can end a label ("now", "again", "here"); and :func:`strip_frame`'s bare name
    (control nouns such as "tab", "link", "item" gone too). With each, whether a partial match (a
    whole word of a longer label) may count on it: not while it still holds a click verb or a
    plain "button"."""
    raw = " ".join(unicodedata.normalize("NFC", str(text or "")).split())
    lead = _ARTICLE.sub("", _LEAD_EN.sub("", raw, count=1).strip(_PUNCT))
    said = _TAIL_HI_VERB.sub("", lead, count=1)
    said = _TAIL_HING_VERB.sub("", said, count=1)
    said = _TAIL_BUTTON.sub("", said, count=1)
    said = _TAIL_FRAME_FILLER.sub("", said, count=1)
    said = _LEFTOVER.sub("", said, count=1).strip(_PUNCT)
    bare = _TAIL_FILLER.sub("", said, count=1).strip(_PUNCT)
    out: dict[str, bool] = {}
    for s, partial in ((raw, False), (lead, False), (said, True), (bare, True), (strip_frame(raw), True)):
        k = norm(s.strip(_PUNCT))
        if k:
            out[k] = out.get(k, False) or partial
    return list(out.items())


def pointing(text: str) -> str | None:
    """The control's name when the request is "press that control" and nothing else, else None.

    Strict, unlike :func:`strip_frame`: an explicit pointing verb or frame in the user's words
    (English click/tap/press/hit/select/choose…, Hinglish "… pe click karo" / "… dabao" / "… wala
    button", Hindi "… पर क्लिक करो" / "… दबाओ" / "… चुनो"), one clause, not a question, and not a
    key ("press enter")."""
    raw = " ".join(unicodedata.normalize("NFC", str(text or "")).split())
    if not raw or is_question(raw):
        return None
    lead = _LEAD_EN.match(raw)
    tail = _HING_POINT.search(raw) or _HI_POINT.search(raw)
    if not (lead or tail):
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
