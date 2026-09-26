"""play_music: "Neo, play 505" — recognised here, decided by neo.decide.

Which app plays it, and how, is not decided in this file: neo.decide works it out on each Mac
from what the installed apps declare, how this user uses them and what worked before, and Laya
makes the call (see neo/decide/). This tool only recognises the request.
"""

from __future__ import annotations

from neo.agent.registry import ToolContext, tool

_VAGUE = {"it", "this", "that", "something", "anything", "some music", "music", "a song", "song", "the song"}


@tool(
    "play_music",
    "Play a song, artist, album, playlist, podcast or video by name. NEO decides which app on "
    "this Mac can actually play it (the one the user uses, if it has it) and confirms it started. "
    "Put the app or site in the query only if the user named one ('505 on youtube'). Never open "
    "a streaming website for music yourself: call this.",
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
        ),
        (r"^\s*(?P<query>.+?)\s+(?:chalao|chala\s+do|bajao|baja\s+do|lagao|laga\s+do|play\s+karo)\s*[.!?]*\s*$", {"query": "<query>"}),
    ],
)
async def play_music(a: dict, c: ToolContext) -> str:
    query = " ".join(str(a.get("query") or "").split()).strip(" .!?\"'")
    if not query or query.lower() in _VAGUE:
        return "Error: play what? (a song, artist or video name)"
    from neo.decide.media import play_best

    return await play_best(query, text=c.user_text or query)
