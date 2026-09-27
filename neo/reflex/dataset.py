"""Synthetic labelled utterances for the reflex head — templates × slots, no API needed.

Run ``python -m neo.reflex.finetune`` to build + train. Add real misroutes to
``~/.neo/reflex_extra.jsonl`` (``{"text": ..., "intent": ...}``) and retrain any time.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from neo.config import settings
from neo.reflex.schema import INTENT_CRITERIA

APPS = [
    "Safari",
    "Mail",
    "Notes",
    "Slack",
    "Spotify",
    "Finder",
    "Terminal",
    "VS Code",
    "Calendar",
    "Messages",
    "Chrome",
    "Preview",
    "Xcode",
    "Zoom",
    "Figma",
    "Discord",
]
SITES = ["youtube", "github", "gmail", "twitter", "reddit", "amazon", "netflix", "notion", "linkedin"]
FILES = [
    "the report",
    "my resume",
    "the budget spreadsheet",
    "the screenshots on my desktop",
    "the PDF I downloaded",
    "this folder",
    "the notes from yesterday",
    "main.py",
    "the invoice",
    "my downloads folder",
]
PEOPLE = ["Sam", "my manager", "the landlord", "Priya", "the team", "mom", "Alex", "the recruiter"]
TOPICS = [
    "the capital of Peru",
    "how photosynthesis works",
    "the difference between TCP and UDP",
    "a good name for a cat",
    "what a Roth IRA is",
    "why the sky is blue",
    "who won the 2018 World Cup",
    "how to say thank you in Japanese",
    "the plot of Dune",
    "how compound interest works",
    "what causes jet lag",
]
MATH = [
    "17 times 23",
    "the square root of 289",
    "15% of 240",
    "2 to the power of 10",
    "how many days until Friday",
    "340 divided by 8",
]
WRITE = [
    "a haiku about rain",
    "a short toast for a wedding",
    "a polite way to decline an invite",
    "a tweet about Monday mornings",
    "a two-line bio for me",
    "a limerick about coffee",
]

# (template, may_take_polite_prefix)
CHAT = [
    ("what's {topic}", False),
    ("tell me {topic}", True),
    ("explain {topic}", True),
    ("do you know {topic}?", False),
    ("what is {math}", False),
    ("calculate {math}", True),
    ("what's {math}", False),
    ("write {write}", True),
    ("give me {write}", True),
    ("what do you think about {topic}", False),
    ("how are you", False),
    ("thanks", False),
    ("thank you", False),
    ("good morning", False),
    ("who are you", False),
    ("what can you do", False),
    ("what's your favourite colour", False),
    ("why is {topic} important", False),
    ("tell me a joke", True),
    ("summarize {topic} in one sentence", True),
    ("is it true that {topic}", False),
    ("what does {topic} mean", False),
    ("help me think through {topic}", True),
    ("that's funny", False),
    ("nice", False),
    ("okay cool", False),
    ("how do you spell necessary", False),
    ("what's a synonym for happy", False),
    ("translate good morning to French", True),
    ("recommend a book about {topic}", True),
]
QUICK = [
    ("play {song}", True),
    ("play the song {song}", True),
    ("play {song} by {artist}", True),
    ("play some {genre}", True),
    ("put on some {genre} music", True),
    ("play {song} on youtube", True),
    ("i want to listen to {song}", True),
    ("play the {artist} song {song}", True),
    ("{song} chalao", False),
    ("{song} gaana bajao", False),
    ("open {app}", True),
    ("launch {app}", True),
    ("open {site}", True),
    ("go to {site}", True),
    ("close {app}", True),
    ("quit {app}", True),
    # short ones too: a sure "stop" of two words cancels every running job ("close it" was one)
    ("close it", False),
    ("close the app", True),
    ("close this app", True),
    ("close the window", True),
    ("close this window", True),
    ("quit the app", True),
    ("switch to {app}", True),
    ("bring up {app}", True),
    ("turn the volume up", True),
    ("volume down", False),
    ("set volume to {n}", True),
    ("turn it up a bit", True),
    ("mute", False),
    ("unmute", False),
    ("brightness up", False),
    ("make the screen dimmer", True),
    ("dim the screen", True),
    ("what time is it", False),
    ("what's the time", False),
    ("set a timer for {n} minutes", True),
    ("start a {n} minute timer", True),
    ("remind me to call {person} at 5", True),
    ("add a reminder to buy milk", True),
    ("remind me about the dentist tomorrow", True),
    ("play some music", True),
    ("pause", False),
    ("pause the music", True),
    ("next song", False),
    ("skip this track", True),
    ("next", False),
    ("next one", False),
    ("skip", False),
    ("skip it", False),
    ("previous song", False),
    ("go to the previous track", True),
    ("what's the weather", False),
    ("is it going to rain today", False),
    ("how cold is it outside", False),
    ("take a screenshot", True),
    ("lock the screen", True),
    ("put the display to sleep", True),
    ("what's on my calendar today", False),
    ("do I have any meetings today", False),
    ("show me my unread emails", True),
    ("any new mail", False),
    ("empty the trash", True),
    ("show desktop", True),
    # dictation and keys (zero-LLM fast paths — the tool head should know them too)
    ("type {phrase}", True),
    ("write {phrase} in the heading", False),
    ("type {phrase} in the search bar", True),
    ("write down {phrase}", True),
    ("put {phrase} as the title", False),
    ("type some points about {topic}", True),
    ("write a short paragraph about {topic}", True),
    ("write a few lines about {topic} in the note", False),
    ("press enter", False),
    ("new line", False),
    ("select all", False),
    ("undo that", False),
    ("save it", False),
    ("scratch that", False),
    ("open a new tab", False),
    ("take a screenshot of this", True),
    ("screenshot the screen", True),
    ("what apps are open", False),
    ("which apps are running", False),
    ("what's running right now", False),
    ("list the open apps", True),
    ("grab a screenshot", True),
    ("screenshot this window", True),
    ("press escape", False),
    ("hit tab", False),
    ("redo that", False),
    ("paste it", False),
    ("copy that", False),
    ("new note", False),
    ("save the file", True),
    ("write a few bullet points about {topic}", True),
    ("jot down some ideas about {topic}", True),
    ("remember this: {fact}", False),
    ("make a note that says {phrase}", True),
    # weather (its own tool now) and searching a site
    ("what's the weather in {city}", False),
    ("will it rain tomorrow", False),
    ("how hot is it outside", False),
    ("do I need an umbrella today", False),
    ("weather forecast for the weekend", False),
    ("is it going to be cold in {city} tomorrow", False),
    ("search {thing} on amazon", True),
    ("look up {thing} on youtube", True),
    ("find {thing} on google", True),
    ("search youtube for {thing}", True),
    ("open {thing} on amazon", True),
    # clicking, scrolling and keys (the tool head's weakest classes)
    ("click the {label} button", True),
    ("click {label}", True),
    ("tap {label}", False),
    ("scroll down", True),
    ("scroll up a bit", False),
    ("page down", False),
    ("press enter", False),
    ("hit escape", False),
    ("press tab", False),
    ("go back", False),
    ("reload the page", True),
    ("zoom in", False),
    ("close this tab", True),
    ("next tab", False),
    ("take a screenshot", True),
    ("grab a screenshot of my screen", True),
    ("type {phrase}", True),
    ("type {phrase} in the search bar", True),
    ("remember that {fact}", True),
    ("note that {fact}", False),
    ("make a note called {phrase}", True),
    ("create a note saying {phrase}", True),
]
PHRASES = [
    "hello world",
    "meeting notes",
    "Laptops",
    "buy milk",
    "call the dentist tomorrow",
    "thanks for your help",
    "project ideas",
    "see you at five",
    "Q3 planning",
    "groceries",
]
CITIES = ["Dubai", "London", "Mumbai", "Tokyo", "New York", "Paris", "Delhi", "Berlin", "Toronto", "Sydney"]
THINGS = ["a 2TB portable SSD", "noise cancelling headphones", "a standing desk", "lofi music", "a mechanical keyboard",
          "iPhone cases", "running shoes", "the Crucial X9 Pro", "cat videos", "a USB-C hub"]
LABELS = ["Sign in", "Play", "Next", "Send", "Continue", "Accept", "Download", "Save", "Submit", "Settings"]
FACTS = [
    "my wifi password is on the fridge",
    "Sam's birthday is in June",
    "I park on level 3",
    "the meeting moved to Thursday",
    "I prefer dark mode",
    "my flight is at 6 am on Friday",
]
AGENT = [
    ("read {file} and summarize it", True),
    ("reply to {person}'s email and say I'll be late", True),
    ("email {person} the notes from today", True),
    ("find {file} and move it to a new folder called archive", True),
    ("organize {file}", True),
    ("delete {file}", True),
    ("rename {file} to final version", True),
    ("what does this window say", False),
    ("what's on my screen", False),
    ("what am I looking at", False),
    ("read the article that's open and give me the key points", True),
    ("fill in this form with my details", True),
    ("book a table for two tonight near me", True),
    ("research {topic} and write a summary in Notes", True),
    ("install {app}", True),
    ("uninstall {app}", True),
    ("update all my brew packages", True),
    ("clone the repo and run the tests", True),
    ("fix the failing test in {file}", True),
    ("compare prices for a used MacBook Air and put them in a note", True),
    ("go through my unread emails and flag the urgent ones", True),
    ("download the invoice from the last email and put it in Documents", True),
    ("create a presentation about {topic}", True),
    ("translate the page that's open into Spanish", True),
    ("look at this error and tell me how to fix it", True),
    ("clear my downloads folder", True),
    ("send a message to {person} on Slack saying the build is green", True),
    ("schedule a meeting with {person} tomorrow at 3 and email the invite", True),
    ("convert {file} to PDF", True),
    ("write a script that renames all the photos by date", True),
    ("find the cheapest flight to Tokyo next month", True),
    ("what's the total of the receipts in {file}", False),
    ("open {site} and search for {topic}", True),
    ("log into the wifi portal for me", True),
    ("sort {file} by date and delete the duplicates", True),
    ("check if the website is up and tell me the load time", True),
    ("back up {file} to an external drive", True),
    ("find the last message from {person} and summarize the thread", True),
    ("copy the table on this page into a spreadsheet", True),
    ("set up a python project called scraper", True),
    ("what's in {file}", False),
    ("how many unread emails do I have from {person}", False),
    ("close all the tabs except this one", True),
    ("zip up {file} and email it to {person}", True),
]
STOP = [
    ("stop", False),
    ("cancel", False),
    ("never mind", False),
    ("be quiet", False),
    ("shut up", False),
    ("go to sleep", False),
    ("that's all", False),
    ("quit", False),
    ("exit", False),
    ("stop talking", False),
    ("cancel that", False),
    ("forget it", False),
    ("stop it", False),
    ("that's enough", False),
    ("okay stop", False),
]

# Short commands that aren't a stop, never trained on (split "validation": the calibration slice, which
# the gate judges on): the false cancels it counts include these every time. The held-out data had
# three short close/quit rows, and the multilingual head was sure "close it" meant stop.
STOP_LOOKALIKES = [
    "close this", "close that", "close window", "close WhatsApp", "quit Photos", "next track", "next please",
    "skip this", "skip song", "previous track", "hide it", "minimise it", "agla gaana", "gaana badlo",
]  # fmt: skip

DESTRUCTIVE_HINTS = (
    "delete",
    "remove",
    "uninstall",
    "send",
    "email",
    "reply",
    "clear",
    "rename",
    "overwrite",
    "install",
    "pay",
    "buy",
    "book",
    "log into",
    "message",
    "empty",
)
_PREFIX = ["can you ", "could you ", "please ", "hey neo, ", "neo, ", "hey neo ", "neo "]

# ---- UI commands: the tool head's click_text class, and what isn't one ----------------------------
# "click X" in many shapes is a one-tool quick action the resolver grounds without looking at the
# screen (needs_screen False); a question about a control is conversation; a click on a label that
# sends, deletes or buys stays destructive. Generic control names only, never an app's: when the OS
# string index (neo.decide.uistrings) is built, its short English UI strings are sampled as well.
UI_LABELS = [
    "New Game", "Start", "Cancel", "Done", "Undo", "Redo", "Hint", "Settings", "Search", "Back", "Go Forward",
    "Reload", "New Tab", "Close", "Minimize", "Full Screen", "Compose", "Reply", "Reply All", "Archive", "Junk",
    "Flag", "Inbox", "Drafts", "Downloads", "Show Sidebar", "Bold", "Italic", "Zoom In", "Zoom Out", "Play",
    "Pause", "Next Track", "Previous", "Mute", "Shuffle", "Repeat", "Sign In", "Create Account", "Help", "Print",
    "Save", "Open", "New Folder", "Rename", "Copy", "Paste", "Reject All", "Next Page", "Today", "Add Event",
    "Filter", "Sort By", "More Options", "Edit", "Attach", "Home", "Profile", "Continue", "Skip", "Details",
    "Show All", "Options", "Apply", "Retry", "Got It", "Learn More", "View", "Add",
]  # fmt: skip
COMMIT_LABELS = ["Send", "Delete", "Move to Trash", "Buy Now", "Place Order", "Empty Trash", "Remove", "Pay", "Publish",
                 "Accept All", "Unsubscribe", "Install", "Sign Out"]  # fmt: skip
UI_FRAMES = [
    "click {x}", "click on {x}", "tap {x}", "tap on {x}", "click the {x} button", "press the {x} button",
    "hit the {x} button", "select {x}", "choose {x}", "click the {x} tab", "tap the {x} option",
    "click the {x} link", "press {x}", "hit {x}", "please click {x}", "can you click {x}", "select the {x} option",
]  # fmt: skip
UI_QUESTIONS = [
    "what does the {x} button do", "where is the {x} button", "what happens if i click {x}",
    "should i press {x}", "is the {x} option safe to click", "why is the {x} button grey",
    "what's the {x} option for", "how do i find the {x} button",
]  # fmt: skip


def _ui_label_pool() -> list[str]:
    """UI_LABELS, plus the OS's own short English UI strings when that index is built (never
    starting a build: training reads what is there)."""
    pool = list(UI_LABELS)
    try:
        from neo.decide import uistrings

        ready = getattr(uistrings, "ready", None)
        short = getattr(uistrings, "short_strings", None)
        if callable(ready) and ready() and callable(short):
            extra = [x for x in short() if isinstance(x, str) and 0 < len(x.split()) <= 3 and x.isascii()]
            pool += sorted(set(extra) - set(pool))[:400]
    except Exception:  # noqa: BLE001 - no index: the generic list
        pass
    return pool


def ui_rows(n: int = 160, n_questions: int = 48, seed: int = 13) -> list[dict]:
    """English UI-command rows with explicit labels: click_text quick actions (needs_screen False;
    destructive when the label commits) and questions about controls (chat)."""
    rnd = random.Random(f"ui-rows:{seed}")
    pool = _ui_label_pool()
    out: list[dict] = []
    seen: set[str] = set()

    def add(row: dict) -> None:
        if row["text"].lower() not in seen:
            seen.add(row["text"].lower())
            out.append(row)

    tries = 0
    while len(out) < n and tries < n * 20:
        tries += 1
        commit = rnd.random() < 0.15
        label = rnd.choice(COMMIT_LABELS if commit else pool)
        text = rnd.choice(UI_FRAMES).format(x=label if rnd.random() < 0.6 else label.lower())
        add({"text": text, "intent": "quick_action", "tool": "click_text", "source": "template", "lang": "en",
             "destructive": commit, "screen": False, "reply": False})  # fmt: skip
    base = len(out)
    tries = 0
    while len(out) < base + n_questions and tries < n_questions * 20:
        tries += 1
        text = rnd.choice(UI_QUESTIONS).format(x=rnd.choice(pool).lower())
        add({"text": text, "intent": "chat", "tool": "", "source": "template", "lang": "en", "destructive": False,
             "reply": True})  # fmt: skip
    return out


# Which NEO tool a quick-action template means (the tool head learns these; "" = none/other).
_TOOL_HINTS: list[tuple[str, str]] = [
    ("play|put on|listen to|chalao|bajao", "play_music"),
    ("open|launch|go to|bring up|switch to", "open_app"),
    ("close|quit", "quit_app"),
    ("weather|rain|cold is it|hot is it|forecast|umbrella", "weather"),
    ("screenshot", "screenshot"),
    ("lock the screen|display to sleep|sleep the display", "sleep_display"),
    ("(?:what'?s |anything )?on my calendar|my calendar|meetings today|my schedule|my agenda", "calendar_today"),
    ("unread email|new mail|inbox|my email", "mail_unread"),
    ("remind me|reminder", "reminder_add"),
    ("what time|the time|what's the date|today's date", "clock"),
    ("timer", "timer"),
    ("volume|mute|unmute|turn it up|turn it down|louder|quieter", "volume"),
    ("brightness|dimmer|brighter|dim the screen", "brightness"),
    ("remember that|remember this|note that", "memory"),
    ("note called|note saying|note that says|note titled", "notes_create"),
    ("on amazon|on youtube|on google|on flipkart|search youtube|search amazon", "search_site"),
    ("click|tap", "click_text"),
    ("scroll|page down|page up", "scroll"),
    ("press|hit|go back|reload|zoom|next tab|close this tab|screenshot", "hotkey"),
    ("what apps are open|which apps are running", "apps_running"),
]


# Hugging Face intents that now have a NEO tool (older ~/.neo/reflex_hf.jsonl files carry "").
_HF_TOOLS = {"play_music": "play_music", "play_radio": "play_music", "play_podcasts": "play_music", "play_audiobook": "play_music"}
# Optional per-row labels of reflex_hf.jsonl passed through as-is (see hf_data.py).
_HF_LABELS = ("kind", "screen", "reply", "lang", "group", "split")
# The benchmark's splits (hf_data.py writes them to reflex_eval.jsonl): never training data.
EVAL_SPLITS = ("eval", "test")


def tool_for(text: str) -> str:
    """Best-effort tool label for a quick action: the exact regex fast paths first, then hints."""
    import re

    try:
        from neo.agent.session import _match_fast_path
        from neo.tools import load_all

        load_all()
        hit = _match_fast_path(text)
        if hit:
            return hit[0]
    except Exception:  # noqa: BLE001 — tools unavailable in a bare training env
        pass
    low = text.lower()
    for pat, tool in _TOOL_HINTS:
        if re.search(rf"\b(?:{pat})\b", low):
            return tool
    return ""


SONGS = ["505", "Blinding Lights", "Shape of You", "Kesariya", "Bohemian Rhapsody", "Levitating", "Tum Hi Ho",
         "Believer", "Heat Waves", "Channa Mereya", "Perfect", "Starboy", "Apna Bana Le", "Yellow", "Numb"]
ARTISTS = ["Arctic Monkeys", "The Weeknd", "Ed Sheeran", "Arijit Singh", "Queen", "Dua Lipa", "Coldplay",
           "Imagine Dragons", "Taylor Swift", "AP Dhillon", "Linkin Park"]
GENRES = ["jazz", "lofi", "rock", "bollywood", "classical", "hip hop", "punjabi", "chill"]


def _fill(t: str, rnd: random.Random) -> str:
    return t.format(
        song=rnd.choice(SONGS),
        artist=rnd.choice(ARTISTS),
        genre=rnd.choice(GENRES),
        app=rnd.choice(APPS),
        site=rnd.choice(SITES),
        file=rnd.choice(FILES),
        person=rnd.choice(PEOPLE),
        topic=rnd.choice(TOPICS),
        math=rnd.choice(MATH),
        write=rnd.choice(WRITE),
        n=rnd.choice([5, 10, 20, 30, 50, 75]),
        phrase=rnd.choice(PHRASES),
        fact=rnd.choice(FACTS),
        city=rnd.choice(CITIES),
        thing=rnd.choice(THINGS),
        label=rnd.choice(LABELS),
    )


def build(n_per_class: int = 320, seed: int = 7, *, override: bool = True, own: bool = True) -> list[dict]:
    """Every labelled row. `override` False leaves other sources' rows with your words as they are
    (see _own_outcomes_win): finetune keeps those out of the head itself, and fits the head it falls
    back to after an erase on them, so that head never learns your labels. `own` False: your
    requests (reflex_extra.jsonl) aren't even read — learning is off."""
    rnd = random.Random(seed)
    rows: list[dict] = []
    seen: set[str] = set()
    for intent, templates in (("chat", CHAT), ("quick_action", QUICK), ("agent_task", AGENT), ("stop", STOP)):
        i = 0
        tries = 0
        while i < n_per_class and tries < n_per_class * 6:
            tries += 1
            tpl, polite = templates[(i + tries) % len(templates)]
            t = _fill(tpl, rnd)
            r = rnd.random()
            if r < 0.25:
                t = rnd.choice(_PREFIX if polite else _PREFIX[3:]) + t
            elif r < 0.35:
                t = t.rstrip("?.") + rnd.choice(["?", " please", " now", ""])
            t = t.strip()
            if t.lower() in seen:
                continue
            seen.add(t.lower())
            i += 1
            rows.append(
                {
                    "text": t,
                    "intent": intent,
                    "source": "template",
                    "destructive": intent == "agent_task" and any(h in t for h in DESTRUCTIVE_HINTS),
                }
            )
    for r in rows:
        r.setdefault("tool", tool_for(r["text"]) if r["intent"] == "quick_action" else "")
    # Romanised-Hindi templates for every class, with explicit labels (templates_hi.py).
    from neo.reflex.templates_hi import rows as hinglish_rows

    rows += [r for r in hinglish_rows() if r["text"].lower() not in seen]
    seen |= {r["text"].lower() for r in rows}
    rows += [r for r in ui_rows() if r["text"].lower() not in seen]  # UI commands (EN), labelled
    # LLM-generated + real-world extras + mapped Hugging Face rows live outside the repo
    # (see generate.py, hf_data.py, README). reflex_eval.jsonl is the benchmark: never trained on.
    for name in ("reflex_generated.jsonl", "reflex_extra.jsonl", "reflex_hf.jsonl"):
        if name == "reflex_extra.jsonl" and not own:
            continue
        extra = settings().data_dir / name
        if extra.exists():
            if name == "reflex_extra.jsonl":  # the user's own requests: deduped, within retention
                from neo.reflex.learn import _read as read_examples

                src = read_examples(extra)
            else:
                src = [json.loads(line) for line in extra.read_text().splitlines() if line.strip()]
            for r in src:
                if not r or (name == "reflex_hf.jsonl" and r.get("split", "train") in EVAL_SPLITS):
                    continue  # a validation split comes through: finetune calibrates on it
                hf = name == "reflex_hf.jsonl"
                gen = name == "reflex_generated.jsonl"  # focused generation labels kind/destructive too
                label = r.get("hf_intent") or r.get("source", "").split(":", 1)[-1]
                out = {
                    "text": r["text"],
                    "intent": r["intent"],
                    "source": name,
                    "hf_intent": label if hf else "",
                    # a public or generated row without the label doesn't know (None); the user's
                    # own rows default to False
                    "destructive": r.get("destructive") if hf or gen else r.get("destructive", False),
                    "tool": r.get("tool") or _HF_TOOLS.get(label, "")
                    if hf
                    else r.get("tool", tool_for(r["text"]) if r["intent"] == "quick_action" else ""),
                }
                if hf or gen:
                    out.update({k: r[k] for k in _HF_LABELS if k in r})
                rows.append(out)
    rows = _own_outcomes_win(rows, override) if own else rows
    said = {r["text"].lower() for r in rows}  # judged only as sentences nothing trains on
    rows += [
        {"text": t, "intent": "quick_action", "source": "template", "destructive": False, "tool": tool_for(t), "split": "validation"}
        for t in STOP_LOOKALIKES
        if t.lower() not in said
    ]
    rnd.shuffle(rows)
    return rows


_OWN_FIELDS = ("w", "verified", "wrong", "conf", "margin", "script", "kind")


def _own_outcomes_win(rows: list[dict], override: bool = True) -> list[dict]:
    """What actually happened beats a guess: a sentence the user said and verified overrides every
    other row with the same words (case and spacing aside) from any source. Their rows also carry
    the ledger's weight and flags through (w, verified, wrong…) so training can weight them.

    A template row isn't dropped but takes the user's intent and tool: templates carry the only
    explicit destructive / needs_screen labels, and "delete this folder" must stay destructive.
    `override` False: only the user's own rows get their flags; every other row stays as it was."""
    extra = settings().data_dir / "reflex_extra.jsonl"
    if not extra.exists():
        return rows
    from neo.reflex.learn import _read as read_examples
    from neo.reflex.learn import norm

    own = {norm(r["text"]): r for r in read_examples(extra)}
    verified = {k for k, r in own.items() if r.get("verified")}
    out = []
    for r in rows:
        k = norm(r["text"])
        if r.get("source") == "reflex_extra.jsonl":
            r.update({f: own[k][f] for f in _OWN_FIELDS if k in own and f in own[k]})
        elif k in verified and override:
            if r.get("source") != "template":
                continue
            r = {**r, "intent": own[k]["intent"], "tool": own[k].get("tool", "")}
        out.append(r)
    return out


def benchmark(rows: list[dict] | None = None, path: Path | None = None) -> tuple[list[dict], int]:
    """The benchmark (``reflex_eval.jsonl``, see hf_data.py) minus every sentence NEO trains on.

    hf_data holds the public train rows out of it, but NEO's own rows — the templates, generated
    and your own examples — are made without looking at it ("next song", "ruk jao" are in both), so
    whoever evaluates drops the overlap here. `rows`: what the head was trained on (default: build()).
    `path`: another benchmark file. Returns (benchmark rows, how many were dropped)."""
    from neo.reflex.sources import text_key

    path = path or settings().data_dir / "reflex_eval.jsonl"
    if not path.exists():
        return [], 0
    bench = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # a cut-off line: not a row
        if isinstance(r, dict) and r.get("text") and r.get("intent") in INTENT_CRITERIA:
            bench.append(r)
    trained = {text_key(r["text"]) for r in (build() if rows is None else rows)}
    keep = [r for r in bench if text_key(r["text"]) not in trained]
    return keep, len(bench) - len(keep)


if __name__ == "__main__":
    from neo import log_to_stdout

    log_to_stdout()  # the providers' status lines, as this command always printed them
    rows = build()
    out = Path(settings().data_dir) / "reflex_dataset.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows))
    print(f"{len(rows)} rows → {out}")
