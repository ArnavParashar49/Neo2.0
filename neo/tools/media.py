"""play_music: "Neo, play 505" — recognised here, decided by neo.decide.

Which app plays it, and how, is not decided in this file: neo.decide works it out on each Mac
from what the installed apps declare, how this user uses them and what worked before, and Laya
makes the call (see neo/decide/). This tool only recognises the request.

"Play chess" is not music when a game by that name is installed (neo.decide.games): the fast
path leaves it to the session, which hands it to the agent, and a model that calls this tool for
it anyway is told to open the game and play it there.
"""

from __future__ import annotations

from neo.agent.registry import ToolContext, tool

# The check both patterns share: "chess as white" of "play chess as white", or "as white" of "open
# chess and play as white", is an installed game, not music (neo.decide.games, which reads the
# request the plan covers). Cached apps only: before the first scan every query is claimed.


def _not_a_game(f: dict) -> bool:
    from neo.agent import fastpath
    from neo.decide import games

    return games.in_play(str(f.get("query") or ""), fastpath.utterance()) is None


_VAGUE = {"it", "this", "that", "something", "anything", "some music", "music", "a song", "song", "the song"}


@tool(
    "play_music",
    "Play a song, artist, album, playlist, podcast or video by name. NEO decides which app on "
    "this Mac can actually play it (the one the user uses, if it has it) and confirms it started. "
    "Put the app or site in the query only if the user named one ('505 on youtube'). Never open "
    "a streaming website for music yourself: call this. Not for a game installed on this Mac "
    "('play chess' with a chess game app): open that app and play it there.",
    {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    category="apps",
    slow=True,
    quiet=True,
    payload=True,
    fast_path=[
        (
            r"^\s*(?:please\s+)?(?:play|put\s+on|listen\s+to)\s+(?:(?:the|a|some)\s+)?(?:song|track|video|album)?\s*"
            r"(?!(?:it|this|that|something|anything|music|next|previous|again|the\s+next\s+one)\s*[.!?]*\s*$)"
            r"(?P<query>.+?)\s*[.!?]*\s*$",
            {"query": "<query>"},
            _not_a_game,
        ),
        (
            r"^\s*(?P<query>.+?)\s+(?:chalao|chala\s+do|bajao|baja\s+do|lagao|laga\s+do|play\s+karo)\s*[.!?]*\s*$",
            {"query": "<query>"},
            _not_a_game,
        ),
    ],
    idempotent=True,
)
async def play_music(a: dict, c: ToolContext) -> str:
    query = " ".join(str(a.get("query") or "").split()).strip(" .!?\"'")
    if not query or query.lower() in _VAGUE:
        return "Error: play what? (a song, artist or video name)"
    from neo.decide import games
    from neo.decide.media import play_best

    # Gemini Live hands over its arguments as the text while the user's words are still on the
    # way: those aren't what the user said, so play_best decides by the query and reports nothing.
    said = c.user_text if c.user_text.strip() and c.user_text != str(a) else ""
    # Gemini Live and the agent call this directly: "chess", or "as white" of "open chess and play
    # as white", is the game on this Mac, never a search for a video of it.
    if (game := games.in_play(query, said)) is not None:
        return games.refusal(game, c.origin)
    return await play_best(query, text=said)
