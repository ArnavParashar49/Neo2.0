"""Typed questions the reflex layer answers for every utterance.

Kept deliberately small: few options → Laya scores each well even zero-shot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Intent = Literal["chat", "quick_action", "agent_task", "stop"]

INTENT_CRITERIA: dict[str, str] = {
    "chat": "a question, conversation, opinion, fact, math, or writing help that needs no action "
    "on the computer",
    "quick_action": "one simple device or app command: change volume or brightness, open or close "
    "an app or website, set a timer or reminder, check the weather or time, play or pause",
    "agent_task": "anything needing several steps or looking at the computer: work inside an app, "
    "read the screen, browse or research, edit files, write and send email, organize, install, "
    "fill forms, code, or automate",
    "stop": "cancel, stop, never mind, be quiet, go to sleep, or shut down",
}

QUESTIONS: dict[str, dict] = {
    "intent": {
        "type": "choice",
        "instructions": "What kind of request is `text`?",
        "criteria": INTENT_CRITERIA,
    },
    "destructive": {
        "type": "noul",
        "instructions": "Would carrying out `text` delete, send, pay, purchase, install, uninstall, "
        "overwrite, or otherwise change something that is hard to undo?",
    },
    "needs_screen": {
        "type": "noul",
        "instructions": "Does answering `text` require seeing what is currently on the user's screen "
        "or in an open window?",
    },
}


@dataclass
class Decision:
    intent: Intent
    intent_confidence: float
    destructive: bool
    destructive_p: float
    needs_screen: bool
    needs_screen_p: float
    source: str = "laya"  # laya | lite | rules
    latency_ms: float = 0.0

    @property
    def confident(self) -> bool:
        return self.intent_confidence >= 0.55
