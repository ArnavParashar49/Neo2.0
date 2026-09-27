""""Play chess as white" with a chess game on this Mac: the game, in its app, never a video.

The log: Laya called it quick_action (0.86), play_music's fast path claimed "play X", and NEO
searched the web for a video of "chess as white". Now:

* play_music's fast path doesn't claim a query that starts with an installed game (its category
  says it is one), nor the rest of a request to play one ("as white" of "open chess and play as
  white"); every music and video request plans exactly as before;
* the session hands a request to play an installed game to the agent, whatever the reflex made of
  it (a tool, a chat, a task) and whatever the voice lane had planned, with the app named and the
  tools to open it;
* play_music itself refuses such a query from Gemini Live or a model, saying what to do instead;
* the agent's prompt says how to play one ("play chess as white").

Apps are fixtures (the inventory is never scanned), the word lists are fixed, no handler runs for
real, no model is called and nothing is launched or played.
"""

from __future__ import annotations

import asyncio
import re

import pytest

import neo.agent.session as sess_mod
from neo import events
from neo.agent import fastpath
from neo.agent.registry import Registry, ToolContext, ToolOutput, registry
from neo.agent.session import Reply, Session
from neo.decide import games, inventory, sites
from neo.decide.inventory import App
from neo.events import EventBus
from neo.reflex.schema import Decision
from neo.tools import load_all

CHESS = App(
    bundle_id="test.boards.chess", name="Chess", path="/System/Applications/Chess.app", category="board-games"
)
POKER = App(bundle_id="test.cards.poker", name="Poker", path="/Applications/Poker.app", category="card-games")
QUEST = App(bundle_id="test.quartz.quest", name="Quartz Quest", path="/Applications/Quartz Quest.app", category="games")
TUNES = App(
    bundle_id="test.tunes",
    name="Tunes",
    path="/Applications/Tunes.app",
    category="music",
    media_evidence=("category=music",),
)
NOTEBOOK = App(bundle_id="test.notebook", name="Notebook", path="/Applications/Notebook.app", category="productivity")
APPS = [CHESS, POKER, QUEST, TUNES, NOTEBOOK]
FLORENCE = App(bundle_id="test.story.florence", name="Florence", path="/Applications/Florence.app", category="games")
SIMON = App(bundle_id="test.toys.simon", name="Simon", path="/Applications/Simon.app", category="games")
_COMMON = {"chess", "poker", "notebook", "hard", "computer", "mac", "quest", "florence", "simon"}
# As the system word list has them: playback words and devices are everyday words too.
_EVERYDAY = {
    "hard", "easy", "mode", "two", "player", "computer", "mac", "level", "evening", "repeat", "shuffle", "loop",
    "background", "concert", "speaker", "phone", "dinner",
}  # fmt: skip

# The user's phrasings for a game of chess, in English and Hinglish.
CHESS_PHRASINGS = [
    "play chess",
    "play chess as white",
    "Play chess as white.",
    "play a game of chess",
    "play a quick game of chess against the computer",
    "let's play chess",
    "Let’s play chess as black",
    "can you play chess with me?",
    "i want to play chess on hard",
    "play chess on my mac",
    "play the chess game as white",
    "chess khelo",
    "chess khelte hain",
    "chalo chess khelte hain",
    "chess ka ek game khelo",
    "chess play karo",
    "open chess and play as white",
    "open chess, then let's play",
    "start a game of chess",
]
# Every music and video request plans as it did before any game was installed.
MUSIC = [
    "play 505",
    "play Pray for Me",
    "play despacito on youtube",
    "play the weeknd",
    "play lo-fi beats",
    "play something by queen",
    "play chess by abba",  # a credit: the recording of that name (see neo.decide.games)
    "play chess from the musical",
    "play chess on youtube",  # a place that is no everyday word: a site, as before
    "play chess on tunes",  # another app: music in it
    "play the chess song",
    "play chessboxing anthem",  # contains the name, isn't it
    "play poker face",  # the name only starts a title
    "play poker face by lady gaga",
    "put on chess",  # "put on" / "listen to" are music words, never a game's
    "listen to chess",
    "open tunes and play chess",  # music in the app they named
    "chess bajao",
]


@pytest.fixture
def mac(monkeypatch):
    """This Mac's apps (the last scan's), and fixed word lists. Nothing may scan."""
    installed = list(APPS)

    def use(apps):
        installed[:] = apps

    monkeypatch.setattr(inventory, "snapshot", lambda: list(installed))
    monkeypatch.setattr(inventory, "inventory", lambda refresh=False: pytest.fail("scanned the Mac"))
    monkeypatch.setattr(inventory, "_common_words", lambda ws: {w: w in _COMMON for w in ws})
    monkeypatch.setattr(sites, "everyday", lambda words: all(w in _EVERYDAY for w in re.findall(r"\w+", words.lower())))
    monkeypatch.setattr(games, "_machine_names", lambda: ("Ada’s Workbench", "Adas-Workbench"))
    load_all()
    return use


def _cold(text: str):
    """The plan before the first scan (no games known): today's behaviour."""
    real = inventory.snapshot
    inventory.snapshot = lambda: None
    try:
        return fastpath.plan(text)
    finally:
        inventory.snapshot = real


# ---- what counts as a game, and as asking to play one ---------------------------------------------


def test_a_game_is_what_its_category_says():
    assert games.is_game(CHESS) and games.is_game(POKER) and games.is_game(QUEST)
    assert games.is_game(App(bundle_id="x", name="X", path="/X.app", category="public.app-category.puzzle-games"))
    assert games.is_game(App(bundle_id="x", name="X", path="/X.app", category="public.app-category.games"))
    assert not games.is_game(TUNES) and not games.is_game(NOTEBOOK)
    assert not games.is_game(App(bundle_id="x", name="X", path="/X.app"))


@pytest.mark.parametrize("text", CHESS_PHRASINGS)
def test_the_chess_phrasings_ask_for_the_installed_game(mac, text):
    assert games.to_play(text) is CHESS


@pytest.mark.parametrize("text", MUSIC)
def test_music_asks_for_no_game(mac, text):
    assert games.to_play(text) is None


def test_a_game_whose_name_has_several_words(mac):
    assert games.to_play("play quartz quest") is QUEST
    assert games.to_play("let's play a round of quartz quest") is QUEST
    assert games.to_play("play quartz") is None  # the whole name, never a piece of it


@pytest.mark.parametrize(
    "query",
    ["chess", "chess as white", "game of chess", "chess with me", "chess against the computer", "chess on hard",
     "chess in two player mode", "chess on level 3", "chess on my mac", "chess in the chess app", "chess game as white",
     "chess and put on some lo-fi"],
)  # fmt: skip
def test_words_about_playing_keep_it_the_game(mac, query):
    assert games.game_at_head(query) is CHESS


@pytest.mark.parametrize(
    "query",
    ["chess by abba", "chess from the musical", "chess on youtube", "chess on tunes", "chess on chess.com",
     "chess song", "chess with lyrics", "chessboxing anthem", "chess master flash", "poker face", "the weeknd", "505"],
)  # fmt: skip
def test_a_credit_a_media_word_a_place_or_a_title_keeps_it_music(mac, query):
    assert games.game_at_head(query) is None


def test_how_to_questions_are_not_requests_to_play(mac):
    assert games.to_play("how do i play chess") is None
    assert games.to_play("who taught you to play chess") is None
    assert games.to_play("open chess") is None  # just opening it: open_app, as before
    assert games.to_play("open chess and play some music") is None


# ---- play_music's fast path ------------------------------------------------------------------------


@pytest.mark.parametrize("text", CHESS_PHRASINGS)
def test_play_music_never_claims_a_game(mac, text):
    steps = fastpath.plan(text)
    assert not any(tool == "play_music" for tool, _ in steps or [])
    if text.lower().startswith(("play", "chess play")):
        assert [tool for tool, _ in _cold(text)] == ["play_music"]  # which it used to claim


def test_open_the_game_and_play_is_no_song_called_as_white(mac):
    assert _cold("open chess and play as white") == [
        ("open_app", {"target": "chess"}),
        ("play_music", {"query": "as white"}),
    ]
    assert fastpath.plan("open chess and play as white") is None


@pytest.mark.parametrize("text", MUSIC)
def test_music_plans_exactly_as_before(mac, text):
    assert fastpath.plan(text) == _cold(text)


def test_music_plans_keep_their_queries(mac):
    assert fastpath.plan("play despacito on youtube") == [("play_music", {"query": "despacito on youtube"})]
    assert fastpath.plan("play chess by abba") == [("play_music", {"query": "chess by abba"})]
    assert fastpath.plan("open tunes and play chess") == [
        ("open_app", {"target": "tunes"}),
        ("play_music", {"query": "chess"}),
    ]


def test_a_game_that_is_not_installed_changes_nothing(mac):
    mac([POKER, QUEST, TUNES, NOTEBOOK])  # games, but no chess
    for text in ["play chess", "play chess as white", "play a game of chess", "chess play karo"]:
        assert fastpath.plan(text) == _cold(text) and fastpath.plan(text)[0][0] == "play_music"
        assert games.to_play(text) is None
    assert games.to_play("chess khelo") is None


def test_an_app_by_that_name_that_is_no_game_changes_nothing(mac):
    ledger = App(bundle_id="test.ledger", name="Chess", path="/Applications/Chess.app", category="productivity")
    mac([ledger, TUNES])
    assert fastpath.plan("play chess as white") == [("play_music", {"query": "chess as white"})]
    assert games.to_play("play chess") is None


def test_before_the_first_scan_nothing_waits_and_nothing_changes(monkeypatch):
    load_all()
    monkeypatch.setattr(inventory, "snapshot", lambda: None)
    monkeypatch.setattr(inventory, "inventory", lambda refresh=False: pytest.fail("scanned on the event loop"))
    assert fastpath.plan("play chess as white") == [("play_music", {"query": "chess as white"})]
    assert games.to_play("play chess as white") is None and games.game_at_head("chess") is None


# ---- the session hands it to the agent -------------------------------------------------------------


class _Reflex:
    def __init__(self, d: Decision):
        self.d = d

    async def decide(self, text):
        return self.d


_LANES = {
    # the log: Laya's one-tool lane with play_music
    "tool": Decision("quick_action", 0.86, False, 0.0, False, 0.0, "laya+head", tool="play_music", tool_p=0.95),
    "chat": Decision("chat", 0.7, False, 0.0, False, 0.0, "laya+head"),
    "task": Decision("agent_task", 0.9, False, 0.0, False, 0.0, "laya+head"),
}


@pytest.fixture
def agent(monkeypatch, mac):
    """Session._agent recorded, never run; no model and no tool may be used."""
    handed: list[tuple[str, Decision, dict]] = []

    async def fake_agent(self, text, d, images, job, **kw):
        handed.append((text, d, kw))
        return Reply("Your move.", "agent", d, job=job)

    async def no_tool(self, name, args, ctx):
        pytest.fail(f"ran {name}")

    monkeypatch.setattr(events, "_bus", EventBus())
    monkeypatch.setattr(Session, "_agent", fake_agent)
    monkeypatch.setattr(Registry, "invoke", no_tool)
    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": pytest.fail("chatted"))
    return handed


@pytest.mark.parametrize("lane", sorted(_LANES))
@pytest.mark.parametrize(
    "text", ["play chess as white", "let's play chess", "chess khelte hain", "open chess and play as white"]
)
def test_the_session_gives_a_game_to_the_agent_with_its_app(agent, lane, text):
    r = asyncio.run(Session(reflex=_Reflex(_LANES[lane])).handle(text, from_user=True))
    assert r.route == "agent"
    ((said, d, kw),) = agent
    assert said == text and d.intent == "agent_task" and d.needs_screen and not d.tool
    assert "Chess" in kw["hint"] and "board game" in kw["hint"] and "New Game" in kw["hint"]
    assert {"open_app", "activate_app"} <= set(kw["also"])
    assert not kw.get("only")  # the whole agent, never the one-tool lane


def test_music_still_takes_the_fast_path(agent, monkeypatch):
    played = []

    async def fake_invoke(self, name, args, ctx):
        played.append((name, args))
        return ToolOutput("Playing.", ok=True)

    monkeypatch.setattr(Registry, "invoke", fake_invoke)
    r = asyncio.run(Session(reflex=_Reflex(_LANES["tool"])).handle("play 505", from_user=True))
    assert r.route == "quick" and played == [("play_music", {"query": "505"})] and agent == []


def test_the_voice_lane_hands_open_and_play_to_the_agent(agent):
    from neo.voice import live

    assert live._command_steps("open chess and play as white") is None
    assert live._command_steps("play chess as white") is None
    assert live._command_steps("play 505") == [("play_music", {"query": "505"})]
    # A plan made before the game was known still never runs as music.
    steps = [("open_app", {"target": "chess"}), ("play_music", {"query": "as white"})]
    r = asyncio.run(Session().command("open chess and play as white", steps))
    assert r.route == "agent" and len(agent) == 1 and "Chess" in agent[0][2]["hint"]


def test_opened_mid_sentence_the_rest_is_still_no_song(monkeypatch, mac):
    """The local voice loop: "open chess" runs while the user is still talking; at the end the rest,
    "play as white", is read with the whole sentence, so nothing plays and all of it goes on."""
    from neo.agent.early import EarlyActor

    opened = []

    async def fake_open(a, c):
        opened.append(a["target"])
        return "Opened"

    async def no_music(a, c):
        pytest.fail("played music")

    monkeypatch.setattr(events, "_bus", EventBus())
    monkeypatch.setattr(registry().get("open_app"), "handler", fake_open)
    monkeypatch.setattr(registry().get("play_music"), "handler", no_music)
    ea = EarlyActor()

    async def run():
        ea.feed("open chess and play as white")
        ea._changed -= 1.0
        await ea.tick()
        assert opened == ["chess"]
        assert await ea.tick(final=True) == []
        assert ea.remaining() == "open chess and play as white"  # for the session, whole

    asyncio.run(run())
    assert fastpath.plan("play as white") == [("play_music", {"query": "as white"})]  # alone, it's a title


def test_the_agent_run_gets_the_app_and_the_tools_to_open_it(monkeypatch, mac):
    seen: dict = {}
    purposes: list[str] = []

    class Brain:
        last_used = last_model = "fake"

    async def fake_run_agent(goal, **kw):
        from neo.agent.loop import AgentResult

        seen.update(kw, goal=goal)
        return AgentResult("Your move.", "done", [], [])

    def fake_brain(purpose="agent"):
        purposes.append(purpose)
        return Brain()

    monkeypatch.setattr(events, "_bus", EventBus())
    monkeypatch.setattr(sess_mod, "run_agent", fake_run_agent)
    monkeypatch.setattr(sess_mod, "brain", fake_brain)
    r = asyncio.run(Session(reflex=_Reflex(_LANES["tool"])).handle("play chess as white", from_user=True))
    assert r.route == "agent" and r.text == "Your move."
    assert purposes == ["vision"]  # the board is looked at
    assert "[GAME] Chess is installed on this Mac as a board game" in seen["system"]
    names = {t.name for t in seen["tools"]}
    assert {"open_app", "activate_app", "screenshot", "click", "more_tools"} <= names


# ---- play_music refuses a game itself --------------------------------------------------------------


@pytest.fixture
def player(monkeypatch, mac):
    from neo.decide import media as media_mod
    from neo.tools import media

    asked: list[str] = []

    async def fake_play_best(query, text=""):
        asked.append(query)
        return "Playing."

    monkeypatch.setattr(media_mod, "play_best", fake_play_best)

    def call(query: str, said: str = "", origin="agent"):
        args = {"query": query}
        ctx = ToolContext(user_text=said or str(args), origin=origin)
        return asyncio.run(media.play_music(args, ctx))

    call.asked = asked
    return call


def test_play_music_refuses_the_game_and_says_what_to_do(player):
    out = player("chess")  # no words of the user's: the query decides
    assert out.startswith("Error: Chess is a board game on this Mac, not a song or a video.")
    assert "Open Chess (open_app) and play it there" in out
    live = player("chess as white", origin="live")
    assert live.startswith("Error: Chess is a board game on this Mac") and "agent_task" in live
    assert player("as white", said="open chess and play as white").startswith("Error: Chess is a board game")
    assert player("chess", said="chess khelo", origin="live").startswith("Error: Chess")
    assert player.asked == []


@pytest.mark.parametrize(
    ("query", "said"),
    [
        ("505", ""),
        ("chess by abba", ""),
        ("chess on youtube", ""),
        ("chess", "open tunes and play chess"),
        ("chess", "play chess on youtube"),
        ("lo-fi beats", "play chess and put on some lo-fi"),
        ("poker face", "play poker face"),
    ],
)
def test_play_music_still_plays_music(player, query, said):
    assert player(query, said=said) == "Playing." and player.asked == [query]


def test_the_refusal_needs_the_game_installed(player, mac):
    mac([TUNES])
    assert player("chess as white") == "Playing."


# ---- what the models are told -------------------------------------------------------------------


def test_the_agent_prompt_covers_play_a_game_as_a_side():
    from neo.agent.prompt import persona

    p = persona()
    assert '"Play <game>" (as a side, at a level, against the user)' in p
    assert "New Game" in p and "the side the user plays" in p and "never music or a video" in p
    assert "chess" not in p.lower()  # no game or app is named in the rules: evidence only


def test_live_is_told_a_game_is_no_music():
    from neo.voice import live

    load_all()
    extra = live._live_extra()
    assert "A game installed on this Mac is not music" in extra and "agent_task" in extra
    assert "open that app and play it there" in registry().get("play_music").description


# ---- review: more ways to ask, and more ways music is asked for -------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "I'd like to play chess as white",
        "i’d love to play chess",
        "would you like to play chess?",
        "do you want to play chess?",
        "do you wanna play chess",
        "you wanna play chess",
        "play me a game of chess",
        "play us a quick game of chess",
        "play chess later",
        "play chess tomorrow",
        "play chess after dinner",
        "open chess and play a game of chess",
        "Open the Chess app and play a game of chess as white against the computer",
        "open chess and play me as white",
        "open chess and play black",
        "play chess, please",
        "play chess with me and make it hard",
        "let's play chess, you be white",
    ],
)
def test_more_ways_to_ask_for_the_game(mac, text):
    assert games.to_play(text) is CHESS
    assert not any(tool == "play_music" for tool, _ in fastpath.plan(text) or [])


@pytest.mark.parametrize(
    ("apps", "text"),
    [
        ([FLORENCE, TUNES], "play florence and the machine"),  # a band whose name starts with the game's
        ([SIMON, TUNES], "play simon and garfunkel"),
        (APPS, "play chess, by abba"),  # a credit after a comma the transcriber put in
        (APPS, "play chess, the musical"),
    ],
)
def test_and_or_a_comma_inside_a_title_is_no_new_request(mac, apps, text):
    mac(apps)
    assert games.to_play(text) is None
    assert fastpath.plan(text) == _cold(text) and fastpath.plan(text)[0][0] == "play_music"


@pytest.mark.parametrize(
    "text",
    [
        "tunes pe chess chalao",  # another app
        "youtube pe chess chalao",  # a site
        "abba ka chess chalao",  # a credit
        "abba ka chess khelo",
        "youtube pe chess khelo",
        "chess khelna band karo",  # stop playing chess
        "mujhe chess khelna nahi hai",  # I don't want to play chess
        "chess khelne ke rules batao",  # tell me the rules
        "chess mat khelo",
        "chess kaise khelte hain",  # how is chess played
    ],
)
def test_hindi_that_asks_for_no_game(mac, text):
    assert games.to_play(text) is None


@pytest.mark.parametrize(
    "text",
    ["mujhe chess khelna hai", "chess khelo na", "mac pe chess khelo", "computer ke saath chess khelo",
     "chess khelte hain aur gaana bajao", "chess khelein kya"],
)  # fmt: skip
def test_hindi_that_asks_for_the_game(mac, text):
    assert games.to_play(text) is CHESS


def test_hinglish_aur_starts_the_next_request(mac):
    assert games.to_play("play chess aur gaana bajao") is CHESS  # the game, then a song
    assert games.to_play("play chess aur the machine") is CHESS


def test_hindi_music_on_another_app_stays_music_in_the_session(agent, monkeypatch):
    played = []

    async def fake_invoke(self, name, args, ctx):
        played.append((name, args))
        return ToolOutput("Playing.", ok=True)

    monkeypatch.setattr(Registry, "invoke", fake_invoke)
    r = asyncio.run(Session(reflex=_Reflex(_LANES["tool"])).handle("tunes pe chess chalao", from_user=True))
    assert r.route == "quick" and played == [("play_music", {"query": "tunes pe chess"})] and agent == []


def _running(app: App) -> App:
    from dataclasses import replace

    return replace(app, running=True)


@pytest.mark.parametrize(
    "text",
    ["play another game", "play as black", "play a new game", "play one more round", "let's play another game as white",
     "play against the computer", "play with me"],
)  # fmt: skip
def test_a_follow_up_plays_the_game_that_is_open(mac, text):
    mac([_running(CHESS), POKER, TUNES])
    assert games.to_play(text).bundle_id == CHESS.bundle_id
    assert not any(tool == "play_music" for tool, _ in fastpath.plan(text) or [])


def test_a_follow_up_needs_exactly_one_game_open(mac):
    assert fastpath.plan("play another game") == [("play_music", {"query": "another game"})]  # none open
    mac([_running(CHESS), _running(POKER), TUNES])
    assert games.to_play("play as black") is None
    assert fastpath.plan("play as black") == [("play_music", {"query": "as black"})]
    mac([_running(CHESS), TUNES])
    titles = ["play it", "play another one", "play something", "play the game", "play black", "play as it was",
              "play against all odds", "play with me by the band", "play one more song"]  # fmt: skip
    for text in titles:
        assert games.to_play(text) is None and fastpath.plan(text) == _cold(text)  # titles, as before


@pytest.mark.parametrize(
    "query",
    ["chess on repeat", "chess on shuffle", "chess on loop", "chess in the background", "chess in concert",
     "chess on my homepod", "chess on my speaker", "chess on my phone"],
)  # fmt: skip
def test_playback_words_and_other_devices_keep_it_music(mac, query):
    assert games.game_at_head(query) is None


@pytest.mark.parametrize("query", ["chess on my mac", "chess on my laptop", "chess on my workbench", "chess on my computer"])
def test_this_computer_by_any_of_its_names_is_still_the_game(mac, query):
    assert games.game_at_head(query) is CHESS


@pytest.mark.parametrize(
    ("query", "said"),
    [
        ("chess as white", "I'd like to play chess as white"),
        ("chess", "do you want to play chess?"),
        ("chess", "चेस खेलो"),  # Hindi in Devanagari: the words don't name it the way the query does
        ("chess", "time for chess"),
        ("chess as white", "mood for some chess as white"),
        ("another game", "another game?"),
    ],
)
def test_play_music_refuses_the_game_whatever_the_wording(player, mac, query, said):
    mac([_running(CHESS), POKER, TUNES, NOTEBOOK])
    assert player(query, said=said, origin="live").startswith("Error: Chess is a board game")
    assert player.asked == []


@pytest.mark.parametrize(
    ("query", "said"),
    [
        ("chess", "put on chess"),
        ("chess", "listen to chess"),
        ("chess", "chess bajao"),
        ("chess", "time for the chess soundtrack"),
        ("chess", "time for chess by abba"),
        ("chess", "time for chess on tunes"),
        ("chess", "open chess.com and play chess"),
        ("poker", "time for poker face"),  # more of a title after the name
    ],
)
def test_play_music_plays_what_the_user_asked_for_as_music(player, query, said):
    assert player(query, said=said) == "Playing." and player.asked == [query]


def test_the_logs_lane_never_searches_for_a_video(monkeypatch, mac):
    """Laya's one-tool lane with play_music, and the model calling it: an unfamiliar wording of
    "play chess" still gets the refusal, never play_best."""
    from neo.decide import media as media_mod

    asked, results = [], []

    async def fake_play_best(query, text=""):
        asked.append(query)
        return "Playing."

    class Brain:
        last_used = last_model = "fake"

    async def fake_run_agent(goal, **kw):
        from neo.agent.loop import AgentResult

        out = await registry().get("play_music").handler({"query": "chess as white"}, kw["ctx"])
        results.append(out)
        return AgentResult(str(out), "done", [], [])

    monkeypatch.setattr(events, "_bus", EventBus())
    monkeypatch.setattr(media_mod, "play_best", fake_play_best)
    monkeypatch.setattr(sess_mod, "run_agent", fake_run_agent)
    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": Brain())
    asyncio.run(Session(reflex=_Reflex(_LANES["tool"])).handle("mood for some chess as white", from_user=True))
    assert asked == [] and results and str(results[0]).startswith("Error: Chess is a board game")


def test_play_music_waits_for_the_first_scan_off_the_event_loop(player, monkeypatch):
    import threading

    scans = []

    def fake_inventory(refresh=False):
        scans.append(threading.current_thread() is threading.main_thread())
        return list(APPS)

    monkeypatch.setattr(inventory, "snapshot", lambda: None)
    monkeypatch.setattr(inventory, "inventory", fake_inventory)
    assert player("chess as white", said="play chess as white").startswith("Error: Chess is a board game")
    assert scans == [False] and player.asked == []


def test_the_session_reads_the_request_off_the_event_loop(agent, monkeypatch):
    import threading

    threads = []
    real = games.to_play

    def spy(text, apps=None):
        threads.append(threading.current_thread() is threading.main_thread())
        return real(text, apps)

    monkeypatch.setattr(games, "to_play", spy)
    asyncio.run(Session(reflex=_Reflex(_LANES["tool"])).handle("play chess as white", from_user=True))
    asyncio.run(Session().command("open chess and play as white", [("open_app", {"target": "chess"})]))
    assert threads == [False, False] and len(agent) == 2  # handle, then command: never on the loop


# ---- an embedded agent (neo2): its own tools decide ------------------------------------------------


def test_only_an_agent_that_can_open_the_game_is_sent_to_play_it(mac, tmp_path):
    """The apps are the Mac's (shared); whether a request becomes a game run is the agent's own
    tools': a host's agent with no way to open an app handles "play chess" as any other request."""
    from neo_lib import make_runtime

    from neo import runtime

    host = make_runtime(tmp_path / "host", "host", ["weather"])
    player = make_runtime(tmp_path / "player", "player", ["open_app", "click", "screenshot"])
    with runtime.bind(host.scope()):
        assert sess_mod._game_to_play("play chess as white") is None
    with runtime.bind(player.scope()):
        assert sess_mod._game_to_play("play chess as white") == CHESS


def test_a_game_runs_required_tools_come_from_the_agents_own_selector(tmp_path):
    from neo_lib import make_runtime

    from neo import runtime
    from neo.agent import toolselect

    rt = make_runtime(tmp_path / "a", "a", ["weather", "open_app"])
    with runtime.bind(rt.scope()):
        sel = toolselect.selector().select("play chess as white", required=sess_mod._GAME_TOOLS)
    assert "open_app" in sel.names and "activate_app" not in sel.names  # not the agent's: never added
    assert set(sel.names) <= {"weather", "open_app"}
