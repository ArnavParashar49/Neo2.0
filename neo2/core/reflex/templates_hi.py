"""Romanised-Hindi (Hinglish) templates for every reflex class — no API, no LLM, deterministic.

The public Hinglish sets are alarms, weather and chit-chat; nothing in them sends an email, pays
or deletes, and Laya's destructive head, trained on English templates, catches only 17–33 % of
"Priya ko email bhejo …" / "ye file delete karo" (measured). These rows are NEO's own data
(dataset.build() adds them as source "template"), with explicit labels: tool, task kind,
destructive, needs the screen, and whether the user wants an answer.

Stop gets its own phrase set (there is no public Hinglish stop data at all): "bas karo",
"ruk jao", "rehne do", "chup ho jao", "bolna band karo" … with the fillers people wrap them in.
A bare "band karo" is not one: it usually means "turn it off" (the music, the alarm), as
neo2/core/reflex/__init__.py's stop rule also decides.
"""

from __future__ import annotations

import random

# ---- slots ---------------------------------------------------------------------------------------
SONGS = ["Kesariya", "Tum Hi Ho", "Channa Mereya", "Apna Bana Le", "Blinding Lights", "Shape of You", "Believer",
         "Raataan Lambiyan", "Kun Faya Kun", "Levitating", "Excuses", "Pasoori"]
ARTISTS = ["Arijit Singh", "AP Dhillon", "Shreya Ghoshal", "Diljit Dosanjh", "Coldplay", "The Weeknd", "Atif Aslam",
           "Badshah", "Taylor Swift", "Kishore Kumar"]
GENRES = ["bollywood", "punjabi", "lofi", "old hindi", "sufi", "romantic", "party", "classical"]
APPS = ["Spotify", "Safari", "Chrome", "Notes", "Mail", "Slack", "WhatsApp", "Finder", "Calendar", "Zoom", "VS Code",
        "Terminal", "Messages", "Photos"]
SITES = ["youtube", "gmail", "amazon", "flipkart", "github", "netflix", "instagram"]
CITIES = ["Delhi", "Mumbai", "Pune", "Bangalore", "Jaipur", "Kolkata", "Lucknow", "Chennai", "Dubai", "London"]
PEOPLE = ["Priya", "Rahul", "Aman", "Neha", "mummy", "papa", "boss", "Sam", "bhaiya", "didi", "Rohan", "manager"]
TEAMS = ["team", "design team", "marketing wale group", "dev channel", "project group"]
MSGS = ["main 10 minute late hoon", "meeting 4 baje shift ho gayi hai", "report ready hai", "kal chutti pe hoon",
        "build green hai", "aaj ghar late aaunga", "call pe aa jao", "payment ho gaya hai"]
FILES = ["ye file", "report wali file", "resume.pdf", "downloads folder ki purani files", "desktop ke screenshots",
         "budget wali sheet", "kal ke notes", "invoice.pdf", "photos wala folder"]
DOCS = ["report wali file", "resume.pdf", "invoice.pdf", "kal ke notes", "budget wali sheet", "contract.pdf",
        "meeting ke notes"]  # something with text in it: you read a document, not a photos folder
SHEETS = ["expenses wali sheet", "budget wali sheet", "is mahine ki receipts wali file", "bills.xlsx"]
THINGS = ["sasta laptop", "noise cancelling headphones", "iPhone 15 ka price", "aaj ka cricket score",
          "paneer butter masala recipe", "nearest petrol pump", "running shoes", "USB-C hub", "Goa ki flights"]
TOPICS = ["photosynthesis", "black hole", "GST", "inflation", "cricket ke rules", "Taj Mahal ka itihas",
          "compound interest", "AI", "global warming", "Roth IRA"]
MATH = ["17 guna 23", "250 ka 15 percent", "340 divided by 8", "2 ki power 10", "144 ka square root"]
WRITE = ["baarish par ek chhoti kavita", "shaadi ke liye ek toast", "invite mana karne ka polite message",
         "Monday morning par ek tweet", "mere liye do line ka bio"]
WORDS = ["serendipity", "nostalgia", "ubiquitous", "jugaad", "resilient"]
TIMES = ["5 baje", "kal subah 7 baje", "shaam 6 baje", "aadhe ghante baad", "raat 10 baje", "kal dopahar 2 baje"]
TODOS = ["doodh lana", "mummy ko call karna", "dawai lena", "bill bharna", "gym jana", "plants ko paani dena"]
EVENTS = ["team meeting", "dentist appointment", "Rahul ki birthday party", "project review", "doctor ka checkup"]
FACTS = ["meri gaadi level 3 pe parked hai", "Priya ka birthday June mein hai", "meeting Thursday ko shift ho gayi",
         "mujhe dark mode pasand hai"]
LABELS = ["Sign in", "Next", "Send", "Continue", "Download", "Save", "Submit", "Play"]
MINUTES = [5, 10, 15, 20, 30, 45]
AMOUNTS = ["500 rupaye", "2000 rupaye", "100 dollar", "1500 rupaye"]
BILLS = ["bijli ka bill", "phone ka bill", "internet ka bill", "credit card ka bill"]
BUYS = ["naya charger", "do movie tickets", "ek phone cover", "groceries", "pizza"]

# (template, tool, kind, reply) for quick actions; the one NEO tool that does it, "" when none.
QUICK: list[tuple[str, str, str, bool]] = [
    ("{song} chalao", "play_music", "play_media", False),
    ("{song} bajao", "play_music", "play_media", False),
    ("{song} gaana laga do", "play_music", "play_media", False),
    ("{artist} ke gaane chalao", "play_music", "play_media", False),
    ("{artist} ka {song} bajao", "play_music", "play_media", False),
    ("kuch {genre} gaane chalao", "play_music", "play_media", False),
    ("spotify pe {song} chalao", "play_music", "play_media", False),
    ("youtube pe {song} laga do", "play_music", "play_media", False),
    ("{app} kholo", "open_app", "open_app", False),
    ("{app} khol do", "open_app", "open_app", False),
    ("{app} open karo", "open_app", "open_app", False),
    ("{site} kholo", "open_app", "open_app", False),
    ("{app} chalu karo", "open_app", "open_app", False),
    ("{app} band karo", "quit_app", "other", False),
    ("{app} band kar do", "quit_app", "other", False),
    ("{app} close karo", "quit_app", "other", False),
    ("isse band karo", "quit_app", "other", False),
    ("ise band kar do", "quit_app", "other", False),
    ("ye app band karo", "quit_app", "other", False),
    ("awaaz badhao", "volume", "other", False),
    ("awaaz thodi kam karo", "volume", "other", False),
    ("volume badha do", "volume", "other", False),
    ("volume {n} pe kar do", "volume", "other", False),
    ("sound mute kar do", "volume", "other", False),
    ("brightness badhao", "brightness", "other", False),
    ("screen ki brightness kam karo", "brightness", "other", False),
    ("{n} minute ka timer lagao", "timer", "other", False),
    ("{n} minute ka timer set karo", "timer", "other", False),
    ("aaj mausam kaisa hai", "weather", "other", True),
    ("{city} mein mausam kaisa hai", "weather", "other", True),
    ("kal baarish hogi kya", "weather", "other", True),
    ("{city} ka temperature kitna hai", "weather", "other", True),
    ("kitne baje hain", "clock", "other", True),
    ("abhi time kya hua hai", "clock", "other", True),
    ("aaj ki date kya hai", "clock", "other", True),
    ("mujhe {time} {todo} yaad dilana", "reminder_add", "add_reminder", False),
    ("{time} {todo} ka reminder lagao", "reminder_add", "add_reminder", False),
    ("{time} ka alarm laga do", "reminder_add", "add_reminder", False),
    ("aaj mera schedule kya hai", "calendar_today", "other", True),
    ("aaj koi meeting hai kya", "calendar_today", "other", True),
    ("calendar mein {time} {event} daal do", "calendar_add", "add_event", False),
    ("koi naya mail aaya kya", "mail_unread", "other", True),
    ("mere unread emails dikhao", "mail_unread", "other", True),
    ("{thing} search karo", "web_search", "web_search", True),
    ("{thing} google karo", "web_search", "web_search", True),
    ("internet pe {thing} dhoondo", "web_search", "web_search", True),
    ("youtube pe {thing} dhoondo", "search_site", "web_search", True),
    ("amazon pe {thing} search karo", "search_site", "web_search", True),
    ("ek note banao ki {fact}", "notes_create", "write_note", False),
    ("{todo} naam ka note bana do", "notes_create", "write_note", False),
    ("yaad rakhna ki {fact}", "memory", "other", False),
    ("screenshot le lo", "screenshot", "other", False),
    ("screen lock kar do", "sleep_display", "other", False),
    ("neeche scroll karo", "scroll", "other", False),
    ("{label} pe click karo", "click_text", "other", False),
    ("enter dabao", "hotkey", "other", False),
    ("agla gaana chalao", "", "other", False),
    ("gaana skip karo", "", "other", False),
    ("next karo", "", "other", False),
    ("gaana pause karo", "", "other", False),
    ("kaun kaun se apps khule hain", "apps_running", "other", True),
]
# (template, kind, destructive, screen, reply)
AGENT: list[tuple[str, str, bool, bool, bool]] = [
    # sends, pays, deletes, installs: the confirm gate has to catch these in Hinglish too
    ("{person} ko email bhejo ki {msg}", "send_email", True, False, False),
    ("{person} ko mail karo ki {msg}", "send_email", True, False, False),
    ("{person} ke email ka reply karo ki {msg}", "send_email", True, False, False),
    ("{person} ko aaj ke notes email kar do", "send_email", True, False, False),
    ("boss ko chutti ke liye email likh ke bhej do", "send_email", True, False, False),
    ("slack pe {team} ko message karo ki {msg}", "send_message", True, False, False),
    ("{person} ko whatsapp pe message bhejo ki {msg}", "send_message", True, False, False),
    ("{person} ko text kar do ki {msg}", "send_message", True, False, False),
    ("{team} ko batao ki {msg}", "send_message", True, False, False),
    ("{file} delete karo", "other", True, False, False),
    ("{file} delete kar do", "other", True, False, False),
    ("{file} hatao", "other", True, False, False),
    ("{file} trash mein daal do", "other", True, False, False),
    ("downloads folder saaf kar do", "other", True, False, False),
    ("{app} install karo", "other", True, False, False),
    ("{app} uninstall kar do", "other", True, False, False),
    ("saare brew packages update karo", "other", True, False, False),
    ("{bill} pay karo", "other", True, False, False),
    ("{person} ko {amount} bhejo", "other", True, False, False),
    ("amazon se {buy} kharido", "other", True, False, False),
    ("{buy} order kar do", "other", True, False, False),
    ("kal ke liye cab book karo", "other", True, False, False),
    ("{doc} ka naam badal ke final kar do", "other", True, False, False),
    ("{person} ke saath kal 3 baje meeting set karo aur invite bhej do", "add_event", True, False, False),
    # several steps, nothing hard to undo
    ("{doc} padho aur summary do", "other", False, False, True),
    ("is page ko padh ke main points batao", "other", False, True, True),
    ("screen pe kya likha hai batao", "other", False, True, True),
    ("is error ko dekho aur fix batao", "other", False, True, True),
    ("ye form meri details se bhar do", "other", False, True, False),
    ("is table ko spreadsheet mein copy karo", "other", False, True, False),
    ("{topic} pe research karke Notes mein likho", "write_note", False, False, False),
    ("{doc} ko PDF mein convert karo", "other", False, False, False),
    ("{file} ko archive folder mein move karo", "other", False, False, False),
    ("mere unread emails dekho aur urgent wale flag karo", "other", False, False, False),
    ("agle mahine Tokyo ki sabse sasti flight dhoondo", "web_search", False, False, True),
    ("{site} kholo aur {thing} search karo", "web_search", False, False, True),
    ("{person} ka last message dhoondo aur summary batao", "other", False, False, True),
    ("scraper naam ka python project setup karo", "other", False, False, False),
    ("{topic} par ek presentation banao", "other", False, False, False),
    ("{sheet} mein receipts ka total kitna hai", "other", False, False, True),
]
# Conversation and knowledge: always wants an answer, never touches the computer.
CHAT: list[str] = [
    "{topic} kya hai",
    "{topic} ke baare mein batao",
    "{topic} samjhao",
    "mujhe {topic} simple mein explain karo",
    "{math} kitna hota hai",
    "{word} ka matlab kya hai",
    "{write} likho",
    "koi joke sunao",
    "tum kaun ho",
    "tumhara naam kya hai",
    "kaise ho",
    "kya haal hai neo",
    "tum kya kya kar sakte ho",
    "shukriya",
    "thank you yaar",
    "bahut badhiya",
    "good morning neo",
    "{topic} pe koi achhi book suggest karo",
    "tumhe {topic} ke baare mein kya lagta hai",
    "{topic} kyun zaroori hai",
    "hindi mein thank you ko kya bolte hain",
    "mujhe {topic} ke baare mein kuch interesting batao",
]
# Greetings and thanks as people say them: never wrapped in "please …" / "… jaldi" ("please shukriya").
CHAT_SOCIAL = frozenset(["tum kaun ho", "tumhara naam kya hai", "kaise ho", "kya haal hai neo", "tum kya kya kar sakte ho",
                         "shukriya", "thank you yaar", "bahut badhiya", "good morning neo"])

# ---- stop phrases ---------------------------------------------------------------------------------
# Not "hatao" ("ye file hatao" deletes) and not a bare "band karo" (usually "turn it off").
STOP_CORE = ["bas karo", "bas kar", "bas bas", "bas", "ruk jao", "ruko", "ruk ja", "rehne do", "rehne de",
             "rehne dijiye", "chup ho jao", "chup ho ja", "chup raho", "chup kar", "chup", "bolna band karo",
             "mat bolo", "chhodo", "chhod do", "jaane do", "cancel karo", "cancel kar do", "stop karo", "stop kar do",
             "kuch nahi", "abhi nahi", "nahi chahiye", "bas itna hi", "so jao", "shaant ho jao", "khamosh"]
# The rude tu-form (or "oye") doesn't take "ji" ("chup kar ji"), and "not now" doesn't take "quickly"
# ("abhi nahi jaldi").
_TU_FORM = frozenset(["bas kar", "ruk ja", "rehne de", "chup ho ja", "chup kar", "chup", "khamosh"])
_NEGATIVE = frozenset(["kuch nahi", "abhi nahi", "nahi chahiye", "bas itna hi"])
STOP_PRE = ["", "arre ", "yaar ", "neo ", "please ", "bhai ", "acha ", "theek hai ", "nahi nahi ", "oye ", "arey yaar "]
STOP_POST = ["", " yaar", " please", " na", " abhi", " bhai", " ab", " ji", " neo", " jaldi"]
STOP_DEVANAGARI = ["बस करो", "बस", "रुको", "रुक जाओ", "रहने दो", "चुप हो जाओ", "चुप रहो", "चुप", "बोलना बंद करो",
                   "छोड़ो", "जाने दो", "कैंसल करो", "मत बोलो", "अभी नहीं", "कुछ नहीं"]
STOP_PRE_HI = ["", "अरे ", "यार ", "प्लीज़ "]
STOP_POST_HI = ["", " यार", " प्लीज़", " ना", " अभी"]

_PREFIX = ["", "", "", "neo ", "please ", "zara ", "yaar ", "neo please ", "bhai "]
_SUFFIX = ["", "", "", " please", " na", " yaar", " jaldi", " abhi"]


def stop_phrases(n: int = 250, seed: int = 0) -> list[str]:
    """`n` distinct romanised stop phrases: every bare core word, then wrapped variants."""
    rnd = random.Random(f"hinglish-stop:{seed}")
    combos = [f"{p}{c}{s}".strip() for c in STOP_CORE for p in STOP_PRE for s in STOP_POST
              if (p or s) and p.strip() != s.strip() and _fits(p, c, s)]
    rnd.shuffle(combos)
    out = list(dict.fromkeys(STOP_CORE + combos))
    return out[: max(n, len(STOP_CORE))]


def _fits(pre: str, core: str, post: str) -> bool:
    if post == " ji" and (core in _TU_FORM or pre == "oye "):
        return False
    return not (post in (" jaldi", " abhi") and core in _NEGATIVE)


def stop_phrases_devanagari(n: int = 50, seed: int = 0) -> list[str]:
    rnd = random.Random(f"hindi-stop:{seed}")
    combos = [f"{p}{c}{s}".strip() for c in STOP_DEVANAGARI for p in STOP_PRE_HI for s in STOP_POST_HI
              if (p or s) and p.strip() != s.strip()]
    rnd.shuffle(combos)
    return list(dict.fromkeys(STOP_DEVANAGARI + combos))[: max(n, len(STOP_DEVANAGARI))]


def _fill(t: str, rnd: random.Random) -> str:
    return t.format(
        song=rnd.choice(SONGS), artist=rnd.choice(ARTISTS), genre=rnd.choice(GENRES), app=rnd.choice(APPS),
        site=rnd.choice(SITES), city=rnd.choice(CITIES), person=rnd.choice(PEOPLE), team=rnd.choice(TEAMS),
        msg=rnd.choice(MSGS), file=rnd.choice(FILES), doc=rnd.choice(DOCS), sheet=rnd.choice(SHEETS),
        thing=rnd.choice(THINGS), topic=rnd.choice(TOPICS),
        math=rnd.choice(MATH), write=rnd.choice(WRITE), word=rnd.choice(WORDS), time=rnd.choice(TIMES),
        todo=rnd.choice(TODOS), event=rnd.choice(EVENTS), fact=rnd.choice(FACTS), label=rnd.choice(LABELS),
        n=rnd.choice(MINUTES), amount=rnd.choice(AMOUNTS), bill=rnd.choice(BILLS), buy=rnd.choice(BUYS),
    )


def _cycle(templates: list, n: int, rnd: random.Random, seen: set[str]):
    """`n` distinct fillings, visiting the templates in turn so every one (every tool) is used."""
    got, tries = 0, 0
    while got < n and tries < n * 20:
        spec = templates[tries % len(templates)]
        tries += 1
        tpl = spec[0] if isinstance(spec, tuple) else spec
        t = _fill(tpl, rnd)
        r = rnd.random()
        if tpl in CHAT_SOCIAL:
            pass
        elif r < 0.3:
            t = rnd.choice(_PREFIX) + t
        elif r < 0.5:
            t = t + rnd.choice(_SUFFIX)
        t = " ".join(t.split())
        if t.lower() in seen:
            continue
        seen.add(t.lower())
        got += 1
        yield t, spec


def rows(n_chat: int = 60, n_quick: int = 120, n_agent: int = 100, n_stop: int = 250, seed: int = 11) -> list[dict]:
    """Labelled Hinglish template rows (source "template"); Devanagari stop phrases come as lang "hi"."""
    rnd = random.Random(f"hinglish-templates:{seed}")
    seen: set[str] = set()
    out: list[dict] = []
    base = {"source": "template", "lang": "hinglish"}
    for t, _ in _cycle(CHAT, n_chat, rnd, seen):
        out.append({**base, "text": t, "intent": "chat", "tool": "", "destructive": False, "screen": False, "reply": True})
    for t, (_, tool, kind, reply) in _cycle(QUICK, n_quick, rnd, seen):
        out.append({**base, "text": t, "intent": "quick_action", "tool": tool, "kind": kind, "destructive": False,
                    "screen": False, "reply": reply})
    for t, (_, kind, destructive, screen, reply) in _cycle(AGENT, n_agent, rnd, seen):
        out.append({**base, "text": t, "intent": "agent_task", "tool": "", "kind": kind, "destructive": destructive,
                    "screen": screen, "reply": reply})
    # A stop wants nothing said back: every label explicit, so nothing downstream has to guess.
    for t in stop_phrases(n_stop, seed):
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append({**base, "text": t, "intent": "stop", "tool": "", "destructive": False, "screen": False,
                        "reply": False})
    for t in stop_phrases_devanagari(seed=seed):
        if t not in seen:
            seen.add(t)
            out.append({**base, "text": t, "intent": "stop", "tool": "", "destructive": False, "screen": False,
                        "reply": False, "lang": "hi"})
    return out
