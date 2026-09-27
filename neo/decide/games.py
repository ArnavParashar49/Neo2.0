"""Games on this Mac: "play <game>" opens that game and plays it, not a song to find, when a game
by that name is installed. Told from evidence, never from a list of games, apps or sites.

An app is a game when its bundle says so: LSApplicationCategoryType ``public.app-category.games``
or one of its ``…-games`` sub-categories (board, card, puzzle, …), else the App Store category
Spotlight records for it. Which words name a game comes from the installed games' own names
(:func:`inventory.name_forms`).

* :func:`is_game` - does the app declare itself a game?
* :func:`game_at_head` - the installed game a play query starts with: "<game>", "<game> as white",
  "a game of <game>", "<game> with me on hard".
* :func:`to_play` - the installed game a whole request asks to play: "play <game> as white",
  "let's play <game>", "can we play a game of <game>", "open <game> and play as black",
  "<game> khelo", "chalo <game> khelte hain".
* :func:`in_play` - what the play_music handler refuses: the rest of a request to play a game
  ("<game>", or "as white" of "open <game> and play as white"), or a query that starts with one
  when the user's words (if any) don't ask for music themselves.
* :func:`hint` / :func:`refusal` - what the agent, or the model that called play_music, is told.

The game's whole name must START the query (after "a game of", "some", "the"): a title that only
contains it ("<word><game>", "the <game> player's song") is music. What follows the name decides
the rest:

* nothing, or words about the game ("as white", "with me", "against the computer", "on hard",
  "later", "on my mac", "in the <game> app") - the game;
* a credit ("<game> by <artist>", "<game> from the musical") - music: the song or album of that
  name. That is the call for "play <game> by <band>": the user named who made it, so they mean
  the recording, and play_music plays it as it does today;
* a media word ("<game> song", "<game> video", "<game> with lyrics") or a playback one ("on
  repeat", "in the background", "on the speaker") - music or a video;
* a place that is another app or a site ("<game> on <a music app>", "<game> on <some-site.com>",
  "on <a name the system word list doesn't know>") - play_music, as today. "On my <x>" is this
  computer only when <x> is an everyday word or one of this Mac's own names;
* any other word right after the name ("<game> face"), "and" or a comma that doesn't start a new
  request ("<game> and the <band>", "<game>, by <artist>") - the name is only the start of a title.

With no name at all, "another game", "a new game as black" or "as black" is the game that is open:
exactly one installed game running (``App.running``), else music as before.

Unless given the apps, only those of the last completed scan are read (:func:`inventory.snapshot`):
nothing waits for a scan, and before the first one there are no games and nothing changes. A word
the system word list hasn't been asked about yet ("<game> on <word>") is one read of that list,
so the session asks :func:`to_play` from a thread.
"""

from __future__ import annotations

import functools
import re
from typing import TYPE_CHECKING

from neo.decide import inventory

if TYPE_CHECKING:
    from neo.decide.inventory import App

__all__ = ["is_game", "games", "game_at_head", "to_play", "in_play", "hint", "refusal"]

_SYSTEM_PREFS = "/Library/Preferences/SystemConfiguration/preferences.plist"

# ---- language tables (words people say, never names) ---------------------------------------------
_DETS = frozenset({"a", "an", "the", "another", "some", "one"})
_ADJS = frozenset({"quick", "new", "little", "short", "fresh", "friendly", "more", "proper", "nice"})
_ROUNDS = frozenset({"game", "games", "match", "matches", "round", "rounds"})  # "a game of <x>"
_NOUNS_AFTER = frozenset({"game", "games", "app", "application"})  # "<x> game as white", "the <x> app"
# Words that end a game's name and go on about playing it ("<x> as white", "<x> with me").
_ABOUT_PLAYING = frozenset({
    "as", "with", "against", "vs", "versus", "on", "in", "at", "for", "now", "again", "please", "pls",
    "online", "together", "today", "tonight", "right", "alone", "solo", "until", "till", "while", "here",
    "but", "so", "instead", "too", "also", "quickly", "playing", "na", "yaar", "bhai",
    "later", "tomorrow", "soon", "after", "afterwards", "sometime", "first", "before",
})  # fmt: skip
_CREDITS = frozenset({"by", "from", "feat", "ft", "featuring"})  # "<x> by <artist>": the recording
_MEDIA = frozenset({
    "song", "songs", "track", "tracks", "music", "musical", "soundtrack", "ost", "theme", "remix", "cover",
    "lyrics", "lyric", "video", "videos", "trailer", "playlist", "album", "podcast", "episode", "audiobook",
    "movie", "film", "radio", "karaoke", "instrumental", "acoustic", "audio",
    "gaana", "gana", "gaane", "gane", "geet", "गाना", "गाने", "गीत",
})  # fmt: skip
# How music plays, and the other devices it plays on: "<x> on repeat", "<x> on the speaker".
_PLAYBACK = frozenset({
    "repeat", "shuffle", "loop", "looped", "background", "concert", "volume", "loud", "loudly", "speaker",
    "speakers", "headphones", "earphones", "earbuds", "stereo", "soundbar", "tv", "television", "phone",
    "tablet",
})  # fmt: skip
# Words that ask for music, whatever follows: "put on <x>", "listen to <x>", "<x> bajao".
_MUSIC_VERB = re.compile(
    r"\b(?:put\s+on|listen(?:ing)?\b|hear|bajao|baja\s+do|bajana|sunao|suna\s+do|sunna|sunte)\b|बजा|सुना|सुन"
)
# The side someone plays, said bare after the game is named: "open <game> and play black".
_SIDES = frozenset({"white", "black", "red", "blue", "green", "yellow", "first", "second"})
# A follow-up that names no game says the side or the opponent: "as black", "against the computer",
# "with me" (not "as it was", "against all odds": titles).
_SIDE_OPENERS = frozenset({"as", "against", "vs", "versus", "with"})
_OPPONENTS = frozenset({
    "computer", "you", "me", "us", "ai", "cpu", "bot", "bots", "machine", "him", "her", "them", "someone",
    "somebody", "anyone", "anybody", "friend", "friends", "player", "players", "human", "mac",
})  # fmt: skip
_WHERE = frozenset({"on", "in", "via", "through", "using"})  # "on hard", "on <an app or a site>"
_OWNERS = frozenset({"my", "this", "your", "our", "mere", "mera", "meri", "apne", "apna", "hamare", "humare"})
# This computer, whatever it's called: "on my laptop" (names of this Mac come from the Mac itself).
_COMPUTER = frozenset({"computer", "laptop", "desktop", "pc", "machine", "system"})
_PRONOUNS = (
    ("it",), ("one",), ("a", "game"), ("the", "game"), ("a", "match"), ("a", "round"), ("some",), ("me",), ("us",),
)  # fmt: skip
_OPENERS = frozenset({"me", "us"})  # "play me a game of <x>"
# Where the first clause ends: in "play <game> and put on some music" only the game is played.
# "Then", "aur" and ";" always end it; "and" and a comma only when a new request follows ("<game>
# and the <band>" is a title).
_CLAUSE_ENDS = frozenset({";", "then", "aur"})
_MAYBE_ENDS = frozenset({",", "and"})
# What starts a new request after "and" or a comma: someone doing something, or a command's verb.
_REQUEST_STARTS = frozenset({
    "i", "i'll", "i'm", "i'd", "im", "we", "we'll", "you", "you'll", "you're", "let's", "lets", "let", "u",
    "then", "also", "after", "now", "please", "pls", "just", "can", "could", "would", "will", "shall",
    "should", "don't", "dont", "do", "what", "what's", "whats", "how", "when", "who", "where", "why",
    "put", "turn", "make", "tell", "set", "show", "give", "start", "open", "close", "stop", "pause", "resume",
    "go", "take", "help", "teach", "explain", "send", "text", "call", "remind", "check", "search", "find",
    "look", "type", "write", "read", "move", "click", "press", "choose", "pick", "select", "use", "keep",
    "try", "be", "get", "bring", "switch", "launch", "run", "save", "mute", "unmute", "lower", "raise", "dim",
    "increase", "decrease", "remember", "watch", "listen", "record", "add", "create", "note", "email",
    "message", "ask", "wait", "hold", "skip", "enable", "disable", "hide", "minimize", "maximize", "quit",
    "exit", "leave", "beat", "see", "say", "change", "clear", "lock", "unlock", "shut", "restart", "reply",
    "share", "copy", "paste", "delete", "undo", "play", "order", "book", "buy", "schedule", "want", "wanna",
    "need", "bajao", "chalao", "lagao", "sunao", "karo", "kar", "batao", "dikhao",
})  # fmt: skip
# Hindi: fillers around a request, what may follow its verb, and what says it's no request at all.
_HI_LEAD = frozenset({
    "chalo", "chal", "aao", "aaja", "mujhe", "muje", "hum", "ham", "humein", "hame", "main", "mai", "ab", "abhi",
    "aaj", "please", "pls", "zara", "jaldi", "yaar", "bhai", "ek", "baar", "phir", "fir", "se", "ok", "okay",
    "let's", "lets", "kya",
})  # fmt: skip
_HI_TAIL = frozenset({
    "hain", "hai", "he", "hei", "ho", "hoon", "hu", "hun", "chalo", "chal", "na", "naa", "yaar", "yar", "bhai",
    "please", "pls", "zara", "abhi", "jaldi", "ji", "do", "de", "dete", "lete", "le", "lo", "sakte", "sakta",
    "sakti", "chahiye", "chahta", "chahti", "chahte", "kya", "main", "mai", "hum", "ab", "phir", "fir", "se",
    "mere", "saath", "sath", "mujhse", "humse", "ek", "baar", "now",
})  # fmt: skip
_HI_NOT = frozenset({
    "nahi", "nahin", "nahiin", "nhi", "nai", "mat", "band", "bas", "ruko", "roko", "rok", "chhodo", "chhod",
    "kaise", "kese", "kaisa", "kyun", "kyu", "kyon", "kab", "kaun", "kisne", "rules", "niyam", "sikhao",
    "sikha", "sikhna", "batao", "bata", "samjhao", "नहीं", "मत", "बंद", "कैसे", "क्यों",
})  # fmt: skip
_HI_WHERE = frozenset({"pe", "par", "pa", "pr", "mein", "me", "on", "in", "में", "पे", "पर"})
_HI_OF = frozenset({"ka", "ki", "ke", "by"})  # "<artist> ka <x>": theirs, the recording

_LEAD = (
    r"(?:(?:please|pls|ok|okay|so|now|hey|come\s+on|chalo|alright)[\s,]+)*"
    r"(?:(?:let'?s|let\s+us|shall\s+we|can\s+we|could\s+we|should\s+we|can\s+you|could\s+you|would\s+you|"
    r"will\s+you|i\s+(?:want|wanna|would\s+like|would\s+love|feel\s+like|need)(?:\s+to)?|"
    r"i'?d\s+(?:really\s+)?(?:like|love)(?:\s+to)?|(?:would|do|did)\s+you\s+(?:like|want|wanna|feel\s+like)(?:\s+to)?|"
    r"(?:you\s+)?wanna|you\s+want\s+to|(?:i'm|i\s+am)\s+(?:going\s+to|gonna)|want\s+to|how\s+about\s+we|"
    r"why\s+don't\s+we)[\s,]+)?"
    r"(?:(?:go|come|just|quickly|now)\s+)?"
)
# "play <x> …", "let's play <x>", "i feel like playing <x>"
_PLAY = re.compile(rf"^{_LEAD}(?:play|playing)\s+(?P<obj>.+)$")
# "open <x> and play as white", "launch <x>, then let's play", "open the <x> app and start a new game"
_OPEN_THEN_PLAY = re.compile(
    rf"^{_LEAD}(?:open|launch|start|run|fire\s+up|bring\s+up|pull\s+up|switch\s+to|go\s+to)\s+(?P<obj>.+?)"
    r"\s*(?:,\s*(?:(?:and|then)\s+)*|\s(?:and|then)\s+(?:then\s+)?)"
    r"(?:(?:let'?s|let\s+us|let\s+me|we'll|i'll|we\s+can|i\s+want\s+to|i\s+wanna)\s+)?"
    r"(?:(?P<play>play)\b|(?:start|begin)\s+(?:a\s+)?(?:new\s+)?(?:game|match)\b)(?P<rest>.*)$"
)
# "start a game of <x>", "how about a quick match of <x>"
_GAME_OF = re.compile(
    rf"^{_LEAD}(?:start|begin|set\s+up|how\s+about|what\s+about)\s+"
    r"(?P<obj>(?:(?:a|an|another|one\s+more)\s+)?(?:(?:new|quick)\s+)?(?:game|match|round)\s+of\s+.+)$"
)
# Hinglish / Hindi, object first: "<x> khelo", "chalo <x> khelte hain", "<x> ka ek game khelo",
# "<x> play karo", "<x> chalao" (the play_music fast path's own verbs), "<x> खेलो".
_HINDI = re.compile(
    r"^(?P<obj>.+?)\s+(?:(?:ka|ki|ke)\s+)?(?:(?:ek|1)\s+)?(?:(?:game|match|round|baazi|bazi)\s+)?"
    r"(?:(?P<khel>khel\S*|खेल\S*)|play\s+kar\S*|chala(?:o|\s+do)|laga(?:o|\s+do))(?P<tail>\s.*)?$"
)
_HI_PLAY = re.compile(r"^(?:khel|खेल)")  # a word of Hindi's verb for playing a game


def is_game(app: App) -> bool:
    """Does the app declare itself a game (its category, or its App Store category)?"""
    seg = (app.category or "").rsplit(".", 1)[-1].strip().lower()
    return seg == "games" or seg.endswith("-games")


def games(apps: list[App] | None = None) -> list[App]:
    """The installed games (of the last completed scan, when ``apps`` isn't given)."""
    return [a for a in _apps(apps) if a.kind == "app" and is_game(a)]


def _apps(apps: list[App] | None) -> list[App]:
    return list(inventory.snapshot() or ()) if apps is None else list(apps)


def _words(s: str) -> list[str]:
    """Lower-case words, and the commas and semicolons between them (where a clause ends)."""
    s = (s or "").casefold().replace("’", "'")
    # Indic scripts' vowel signs aren't \w: "खेलो" is one word, not four letters.
    toks = re.findall(r"[\w\u0900-\u0963\u0966-\u0dff'.+-]+|[,;]", s)
    return [w for t in toks if (w := t if t in ",;" else t.strip("'.+-_"))]


def _clause(words: list[str]) -> list[str]:
    """The words up to the end of their clause ("as white" of "as white and turn it up"). "And" or
    a comma ends it only before a new request; otherwise it's part of a title ("<x> and the <band>")
    and a comma is dropped ("<x>, by <artist>" reads as "<x> by <artist>")."""
    for i, w in enumerate(words):
        if w in _CLAUSE_ENDS or (w == "after" and words[i + 1 : i + 2] == ["that"]):
            return [x for x in words[:i] if x != ","]
        if w in _MAYBE_ENDS and _new_request(words[i + 1 :]):
            return [x for x in words[:i] if x != ","]
    return [x for x in words if x != ","]


def _new_request(rest: list[str]) -> bool:
    """Does a new request start here ("put on some music", "you be white", "make it hard")?"""
    rest = [w for w in rest if w != ","]
    while rest and rest[0] in _MAYBE_ENDS:
        rest = rest[1:]
    return not rest or rest[0] in _REQUEST_STARTS or bool(_HI_PLAY.match(rest[0]))


def _norm(text: str) -> str:
    t = " ".join((text or "").casefold().replace("’", "'").split())
    return t.strip(" .!?।,")


_forms_cache: dict[tuple, tuple[tuple[str, ...], ...]] = {}


def _forms(app: App) -> tuple[tuple[str, ...], ...]:
    """The app's names as words ("<name>", "<vendor> <name>"), longest first."""
    key = (app.bundle_id, app.name, app.path, app.aliases, app.own_schemes)
    f = _forms_cache.get(key)
    if f is None:
        if len(_forms_cache) > 1024:
            _forms_cache.clear()
        ws = dict.fromkeys(w for n in inventory.name_forms(app) if (w := tuple(_words(n))))
        f = _forms_cache[key] = tuple(sorted(ws, key=len, reverse=True))
    return f


def _named_at(words: list[str], candidates: list[App]) -> tuple[App, int] | None:
    """(the game whose name the words start with, how many words it takes). Longest name first,
    then the game this user plays most."""
    best: tuple[int, float, App] | None = None
    for g in candidates:
        for fw in _forms(g):
            n = len(fw)
            if tuple(words[:n]) == fw:
                cand = (n, inventory.affinity(g), g)
                if best is None or cand[:2] > best[:2]:
                    best = cand
                break  # longest form of this game first
    return (best[2], best[0]) if best else None


def _openers(words: list[str]) -> list[int]:
    """Where the name may start: at once, or after "a", "a quick", "a game of", "the game"."""
    starts, i = [0], 0
    if words[:1] and words[0] in _OPENERS:  # "me a game of <x>"
        i = 1
    elif words[:1] == ["with"] and words[1:2] and words[1] in _OPENERS:  # "with me a round of <x>"
        i = 2
    if i:
        starts.append(i)
    if i < len(words) and words[i] in _DETS:
        i += 1
        starts.append(i)
    while i < len(words) and words[i] in _ADJS:
        i += 1
        starts.append(i)
    if i < len(words) - 1 and words[i] in _ROUNDS:
        i += 2 if words[i + 1] == "of" else 1
        starts.append(i)
    return list(dict.fromkeys(s for s in starts if s < len(words)))


def _elsewhere(dest: list[str], game: App, apps: list[App]) -> bool:
    """Does "on/in <dest>" name somewhere else to play it: another app, or a site?"""
    owned = bool(dest) and dest[0] in _OWNERS
    d = dest[1:] if dest and (dest[0] in _DETS or owned) else dest
    if not d or _named_at(d, [game]) is not None:  # "in the <game> app"
        return False
    if any("." in w for w in d):  # a site written as a domain
        return True
    others = [a for a in apps if a.bundle_id != game.bundle_id]
    if inventory.named_in(" ".join(d), others) is not None:  # another app on this Mac
        return True
    from neo.decide import sites

    plain = [w for w in d if not w.isdigit()]
    if not plain:
        return False
    if owned and all(w in _COMPUTER or w in _this_mac() for w in plain):  # "on my laptop": here
        return False
    # "on hard", "in two player mode", "on the computer", "on my mac": everyday words. A name the
    # system word list doesn't know is a place (a site, a service, "my <speaker>"): play_music
    # decides, as it always did.
    return not sites.everyday(" ".join(plain))


@functools.cache
def _machine_names() -> tuple[str, ...]:
    """This Mac's own names (the Computer Name and the local host name set in Sharing), from the
    system's configuration preferences: one small local file read, once."""
    import plistlib

    try:
        with open(_SYSTEM_PREFS, "rb") as f:
            system = plistlib.load(f).get("System") or {}
        hosts = (system.get("Network") or {}).get("HostNames") or {}
        names = ((system.get("System") or {}).get("ComputerName"), hosts.get("LocalHostName"))
        return tuple(str(n) for n in names if n)
    except Exception:  # noqa: BLE001 - no names: only everyday words and _COMPUTER count
        return ()


@functools.lru_cache(maxsize=4)
def _name_words(names: tuple[str, ...]) -> frozenset[str]:
    return frozenset(
        w for n in names for w in _words(re.sub(r"[-_]", " ", n)) if "'" not in w and not w.isdigit()
    )


def _this_mac() -> frozenset[str]:
    """The words of this Mac's own names ("<owner>'s <model>": the model's words)."""
    return _name_words(_machine_names())


def _about_playing(tail: list[str], game: App, apps: list[App]) -> bool:
    """Is what follows the game's name about playing it ("as white", "with me on hard")? Only its
    first clause counts: "<game> and put on some music" plays the game, then the music."""
    tail = _clause(tail)
    if tail and tail[0] in _NOUNS_AFTER:
        tail = tail[1:]
    if not tail:
        return True
    if tail[0] in _CREDITS or tail[0] not in _ABOUT_PLAYING:
        return False  # "<x> by <artist>", or the name goes on as a title
    if any(w in _MEDIA or w in _PLAYBACK for w in tail):
        return False
    for i, w in enumerate(tail):
        if w not in _WHERE:
            continue
        dest: list[str] = []
        for x in tail[i + 1 :]:
            if x in _ABOUT_PLAYING:
                break
            dest.append(x)
        if dest and _elsewhere(dest, game, apps):
            return False
    return True


def _at_head(words: list[str], gs: list[App], apps: list[App]) -> App | None:
    for start in _openers(words):
        hit = _named_at(words[start:], gs)
        if hit is not None:
            game, n = hit
            return game if _about_playing(words[start + n :], game, apps) else None
    return None


def game_at_head(query: str, apps: list[App] | None = None) -> App | None:
    """The installed game a play query starts with, when the rest is about playing it (see the
    module docstring), else None."""
    all_apps = _apps(apps)
    gs = games(all_apps)
    if not gs:
        return None
    return _at_head(_words(_norm(query)), gs, all_apps)


def _whole(words: list[str], gs: list[App]) -> App | None:
    """The game the words name and nothing else ("<game>", "the <game> app")."""
    if words and words[0] in _DETS | _OWNERS:
        words = words[1:]
    if words and words[-1] in _NOUNS_AFTER:
        words = words[:-1]
    hit = _named_at(words, gs)
    return hit[0] if hit is not None and hit[1] == len(words) else None


def _as_rest(rest: list[str], game: App, apps: list[App]) -> bool:
    """Is ``rest`` (what follows "play" once the game is known) about playing that game: nothing,
    "it", "as white", "black", "me as white", "a game of <game>", "another game"? Not "some music"
    or a song's name."""
    rest = _clause(rest)
    if rest and _at_head(rest, [game], apps) is game:  # "a game of <game>", "<game> with me"
        return True
    if _follow_up(rest, game, apps):  # "another game", "a new game as black"
        return True
    for p in _PRONOUNS:
        if tuple(rest[: len(p)]) == p:
            rest = rest[len(p) :]
            break
    if rest and _at_head(rest, [game], apps) is game:
        return True
    if rest and all(w in _SIDES for w in rest):  # "open <game> and play black"
        return True
    return _about_playing(rest, game, apps)


def _follow_up(words: list[str], game: App, apps: list[App]) -> bool:
    """Words about playing that name no game: "another game", "a new game as black", "one more
    round", "as black", "against the computer". Not "it", "another one", "the game" (titles)."""
    w = _clause(words)
    i = 0
    while i < len(w) and w[i] in _DETS:
        i += 1
    said_which = bool(set(w[:i]) - {"the"})
    while i < len(w) and w[i] in _ADJS:
        i += 1
        said_which = True
    if i < len(w) and w[i] in _ROUNDS:
        tail = w[i + 1 :]
        if not tail:
            return said_which  # "another game", "(a) new game"; "the game" alone is a title
    elif i == 0 and _side_or_opponent(w):
        tail = w
    else:
        return False
    if any(x in _CREDITS for x in tail):  # "with me by <band>"
        return False
    return tail[0] in _ABOUT_PLAYING and _about_playing(tail, game, apps)


def _side_or_opponent(w: list[str]) -> bool:
    """"as black", "as the white player", "against the computer", "vs a friend", "with me"."""
    if len(w) < 2 or w[0] not in _SIDE_OPENERS:
        return False
    rest = w[2:] if w[1] in ("the", "a", "an", "my", "your") else w[1:]
    if not rest:
        return False
    if w[0] == "as":
        return rest[0] in _SIDES or rest[0] == "player"
    return rest[0] in _OPPONENTS


def _open_game(gs: list[App]) -> App | None:
    """The one installed game that is running, or None (none, or several)."""
    live = [g for g in gs if g.running]
    return live[0] if len(live) == 1 else None


def to_play(text: str, apps: list[App] | None = None) -> App | None:
    """The installed game the request asks to play, or None (see the module docstring)."""
    all_apps = _apps(apps)
    gs = games(all_apps)
    if not gs:
        return None
    t = _norm(text)
    if m := _PLAY.match(t):
        obj = _words(m["obj"])
        if (game := _at_head(obj, gs, all_apps)) is not None:
            return game
        # "play another game", "play as black": the game that is open, when exactly one is
        game = _open_game(gs)
        return game if game is not None and _follow_up(obj, game, all_apps) else None
    if m := _OPEN_THEN_PLAY.match(t):
        game = _whole(_words(m["obj"]), gs)
        if game is None or not m["play"]:  # "… and start a new game": that game, whatever follows
            return game
        return game if _as_rest(_words(m["rest"]), game, all_apps) else None
    if m := _GAME_OF.match(t):
        return _at_head(_words(m["obj"]), gs, all_apps)
    if m := _HINDI.match(t):
        return _hindi(m, gs, all_apps)
    return None


def _hindi(m: re.Match, gs: list[App], apps: list[App]) -> App | None:
    """"chalo <game> khelte hain", "<game> ka ek game khelo", "<game> play karo". With khelna (to
    play a game) the object ends with the game's name and what comes before it is no place or
    credit ("<site> pe", "<artist> ka"); with chalao / lagao / play karo, which play songs too, the
    object is the game and nothing else. After the verb only fillers or words about playing: never
    "nahi", "band karo", "rules batao"."""
    words = _words(m["obj"])
    if any(w in _HI_NOT or w in _MEDIA for w in words):
        return None
    tail = _words(m["tail"] or "")
    # The verb closed the game's clause: after "aur", "and" or a comma comes the next request.
    tail = tail[: next((i for i, w in enumerate(tail) if w in _MAYBE_ENDS or w in _CLAUSE_ENDS), len(tail))]
    if any(w in _HI_NOT or w in _MEDIA or w in _PLAYBACK for w in tail):
        return None
    if not m["khel"]:
        lead = 0
        while lead < len(words) - 1 and words[lead] in _HI_LEAD:
            lead += 1
        game = _whole(words[lead:], gs)
    else:
        ends = [(len(fw), g) for g in gs for fw in _forms(g) if tuple(words[-len(fw) :]) == fw]
        if not ends:
            return None
        n, game = max(ends, key=lambda e: e[0])
        before = words[:-n]
        if before and before[-1] in _HI_OF:  # "<artist> ka <game>": their recording
            return None
        for i, w in enumerate(before):  # "<site> pe <game>", "<app> mein <game>"
            dest = [x for x in before[:i] if x not in _HI_LEAD and x not in _HI_WHERE]
            if w in _HI_WHERE and dest and _elsewhere(dest, game, apps):
                return None
    if game is None:
        return None
    rest = [w for w in tail if w not in _HI_TAIL and w not in _SIDES]
    return game if not rest or (rest[0] in _ABOUT_PLAYING and _about_playing(rest, game, apps)) else None


def in_play(query: str, said: str = "", apps: list[App] | None = None) -> App | None:
    """The game a play_music call is really about, or None: it's music.

    With the user's own words (``said``) that ask to play an installed game, they decide: the query
    is that game or only the rest of the request ("<game>", or "as white" of "open <game> and play
    as white"). Otherwise the query does - it starts with an installed game ("<game> as white"), or
    it's "another game" with one open - unless the user's words ask for music themselves: a music
    verb ("put on", "listen to", "bajao"), a media or playback word, another app or a site, or after
    the game's name a credit, a place or more of a title ("play <game> by <band>", "open <a music
    app> and play <game>"). Words this reading doesn't know ("time for <game>", another script)
    never make a game music."""
    all_apps = _apps(apps)
    gs = games(all_apps)
    if not gs:
        return None
    q = _words(_norm(query))
    if said.strip():
        game = to_play(said, all_apps)
        if game is not None:
            return game if _as_rest(q, game, all_apps) else None
    game = _at_head(q, gs, all_apps)
    if game is None and (live := _open_game(gs)) is not None and _follow_up(q, live, all_apps):
        game = live
    if game is None or (said.strip() and _asks_music(said, game, all_apps)):
        return None
    return game


def _asks_music(said: str, game: App, apps: list[App]) -> bool:
    """Do the user's words ask for music or a video (see :func:`in_play`)?"""
    t = _norm(said)
    if _MUSIC_VERB.search(t):
        return True
    words = _words(t)
    if any(w in _MEDIA or w in _PLAYBACK for w in words):
        return True
    if any(re.fullmatch(r"[\w-]+(?:\.[\w-]+)*\.[a-z]{2,}", w) for w in words):  # a site: "<x>.com"
        return True
    if inventory.named_in(t, [a for a in apps if a.bundle_id != game.bundle_id]) is not None:
        return True
    for i in range(len(words)):  # what follows the game's name, wherever it's said
        hit = _named_at(words[i:], [game])
        if hit is None:
            continue
        tail = words[i + hit[1] :]
        if tail and _HI_PLAY.match(tail[0]):  # "<game> khelte hain …": playing it
            continue
        if not _about_playing(tail, game, apps):
            return True
    return False


def _kind(game: App) -> str:
    """"board game", "card game", "game": what the app says it is."""
    seg = (game.category or "").rsplit(".", 1)[-1].strip().lower().replace("-", " ")
    return seg[:-1] if seg.endswith("games") else "game"


def hint(game: App) -> str:
    """For the agent's system prompt: the app to play in and what to do there."""
    return (
        f"\n[GAME] {game.name} is installed on this Mac as a {_kind(game)}, and the user wants to play "
        f"it there - not a song or a video: don't use play_music or the web for it. Open {game.name} or "
        "bring it to the front, set up the game the way they asked (their side, the level, who they "
        "play against) through its own New Game menu or dialog, then play on its board."
    )


def refusal(game: App, origin: str = "agent") -> str:
    """play_music's answer for a game: short, and what to do instead."""
    head = f"Error: {game.name} is a {_kind(game)} on this Mac, not a song or a video."
    if origin == "live":
        return f"{head} Call agent_task to open {game.name} and play it there, with the user's words as the goal."
    return f"{head} Open {game.name} (open_app) and play it there; don't look for it on the web."
