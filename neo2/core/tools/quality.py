"""Ask NEO for its local, content-free reliability signals."""

from __future__ import annotations

from neo2.core.agent.registry import ToolContext, tool
from neo2.core.reflex import learn


@tool(
    "quality_report",
    "Show NEO's recent local reliability signals: settled requests, verified outcomes, user corrections, "
    "early interruptions, unresolved outcomes and Laya routing uncertainty. This is not a factual-accuracy score.",
    {"type": "object", "properties": {"days": {"type": "integer", "minimum": 1, "maximum": 30}}},
    category="system",
    parallel_safe=True,
    read_only=True,
    # Only an explicit ask about reliability: "how are you doing?" or "how is Neo?" is a greeting,
    # and a chat answers it (statistics would be a strange reply).
    fast_path=[
        (r"^\s*(?:how(?:'s| is) neo doing|how(?:'s| is) your reliability|"
         r"show (?:me )?(?:neo'?s )?(?:recent )?reliability|show (?:me )?(?:your )?quality report)"
         r"\s*[?.!]*$", {"days": 7}),
    ],
)
async def quality_report(args: dict, ctx: ToolContext) -> str:
    try:
        days = int(args.get("days", 7))
    except (TypeError, ValueError):
        days = 7
    return learn.quality_report(days)
