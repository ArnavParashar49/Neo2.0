"""The words of NEO's computer tools' fast paths: the regexes that turn a sentence the user said
into a tool call with no model — scroll, type, keys, click by label. Data only: the tools that use
them (tools_pointer, tools_keys, tools_label) look them up here."""

from __future__ import annotations

from neo.config import decide_mode

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
