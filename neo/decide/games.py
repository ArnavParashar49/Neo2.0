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
  ("<game>", or "as white" of "open <game> and play as white"), or without the user's words a
  query that starts with one.
* :func:`hint` / :func:`refusal` - what the agent, or the model that called play_music, is told.

The game's whole name must START the query (after "a game of", "some", "the"): a title that only
contains it ("<word><game>", "the <game> player's song") is music. What follows the name decides
the rest:

* nothing, or words about the game ("as white", "with me", "against the computer", "on hard",
  "on my mac", "in the <game> app") - the game;
* a credit ("<game> by <artist>", "<game> from the musical") - music: the song or album of that
  name. That is the call for "play <game> by <band>": the user named who made it, so they mean
  the recording, and play_music plays it as it does today;
* a media word ("<game> song", "<game> video", "<game> with lyrics") - music or a video;
* a place that is another app or a site ("<game> on <a music app>", "<game> on <some-site.com>",
  "on <a name the system word list doesn't know>") - play_music, as today;
* any other word right after the name ("<game> face") - the name is only the start of a title.

Only the apps of the last completed scan are read (:func:`inventory.snapshot`), so every function
answers in microseconds; before the first scan there are no games and nothing changes.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from neo.decide import inventory

if TYPE_CHECKING:
    from neo.decide.inventory import App

__all__ = ["is_game", "games", "game_at_head", "to_play", "in_play", "hint", "refusal"]

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
})  # fmt: skip
_CREDITS = frozenset({"by", "from", "feat", "ft", "featuring"})  # "<x> by <artist>": the recording
_MEDIA = frozenset({
    "song", "songs", "track", "tracks", "music", "musical", "soundtrack", "ost", "theme", "remix", "cover",
    "lyrics", "lyric", "video", "videos", "trailer", "playlist", "album", "podcast", "episode", "audiobook",
    "movie", "film", "radio", "karaoke", "instrumental", "acoustic", "audio",
})  # fmt: skip
_WHERE = frozenset({"on", "in", "via", "through", "using"})  # "on hard", "on <an app or a site>"
_OWNERS = frozenset({"my", "this", "your", "our"})
_PRONOUNS = (("it",), ("one",), ("a", "game"), ("the", "game"), ("a", "match"), ("a", "round"), ("some",))
# Where the first clause ends: in "play <game> and put on some music" only the game is played.
_CLAUSE_ENDS = frozenset({",", ";", "and", "then"})

_LEAD = (
    r"(?:(?:please|pls|ok|okay|so|now|hey|come\s+on|chalo|alright)[\s,]+)*"
    r"(?:(?:let'?s|let\s+us|shall\s+we|can\s+we|could\s+we|should\s+we|can\s+you|could\s+you|would\s+you|"
    r"will\s+you|i\s+(?:want|wanna|would\s+like|'d\s+like|would\s+love|'d\s+love|feel\s+like|need)(?:\s+to)?|"
    r"(?:i'm|i\s+am)\s+(?:going\s+to|gonna)|want\s+to|wanna|how\s+about\s+we|why\s+don't\s+we)[\s,]+)?"
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
    r"(?:khel\S*|खेल\S*|play\s+kar\S*|chala(?:o|\s+do)|laga(?:o|\s+do))(?:\s.*)?$"
)


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
    return [w for t in re.findall(r"[\w'.+-]+|[,;]", s) if (w := t if t in ",;" else t.strip("'.+-_"))]


def _clause(words: list[str]) -> list[str]:
    """The words up to the end of their clause ("as white" of "as white and turn it up")."""
    for i, w in enumerate(words):
        if w in _CLAUSE_ENDS or (w == "after" and words[i + 1 : i + 2] == ["that"]):
            return words[:i]
    return words


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
    if owned:  # "on my mac", "on my laptop": this computer
        return False
    from neo.decide import sites

    plain = [w for w in d if not w.isdigit()]
    # "on hard", "in two player mode", "on the computer": everyday words. A name the system word
    # list doesn't know is a place (a site, a service): play_music decides, as it always did.
    return bool(plain) and not sites.everyday(" ".join(plain))


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
    if any(w in _MEDIA for w in tail):
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
    "it", "as white", "<game> with me"? Not "some music" or a song's name."""
    rest = _clause(rest)
    for p in _PRONOUNS:
        if tuple(rest[: len(p)]) == p:
            rest = rest[len(p) :]
            break
    if rest and _at_head(rest, [game], apps) is game:
        return True
    return _about_playing(rest, game, apps)


def to_play(text: str, apps: list[App] | None = None) -> App | None:
    """The installed game the request asks to play, or None (see the module docstring)."""
    all_apps = _apps(apps)
    gs = games(all_apps)
    if not gs:
        return None
    t = _norm(text)
    if m := _PLAY.match(t):
        return _at_head(_words(m["obj"]), gs, all_apps)
    if m := _OPEN_THEN_PLAY.match(t):
        game = _whole(_words(m["obj"]), gs)
        if game is None or not m["play"]:  # "… and start a new game": that game, whatever follows
            return game
        return game if _as_rest(_words(m["rest"]), game, all_apps) else None
    if m := _GAME_OF.match(t):
        return _at_head(_words(m["obj"]), gs, all_apps)
    if m := _HINDI.match(t):  # the object ends with the game's name: "chalo <game> khelte hain"
        words = _words(m["obj"])
        ends = [(len(fw), g) for g in gs for fw in _forms(g) if tuple(words[-len(fw) :]) == fw]
        return max(ends, key=lambda e: e[0])[1] if ends else None
    return None


def in_play(query: str, said: str = "", apps: list[App] | None = None) -> App | None:
    """The game a play_music call is really about, or None: it's music.

    With the user's own words (``said``) they decide: they ask to play an installed game and the
    query is that game or only the rest of the request ("<game>", or "as white" of "open <game> and
    play as white"). Words that ask for no game ("play <game> by <band>", "open <a music app> and
    play <game>") let the query play whatever it is. Without them, the query alone: it starts with
    an installed game ("<game> as white")."""
    all_apps = _apps(apps)
    if said.strip():
        game = to_play(said, all_apps)
        return game if game is not None and _as_rest(_words(query), game, all_apps) else None
    return game_at_head(query, all_apps)


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
