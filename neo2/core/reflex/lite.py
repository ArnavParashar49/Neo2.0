"""Cloud fallback reflex: Gemini Flash-Lite with a JSON schema. ~300-600 ms, free tier."""

from __future__ import annotations

import json
import time

from google import genai
from google.genai import types as gt

from neo.config import settings
from neo.reflex.schema import INTENT_CRITERIA, Decision

_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": list(INTENT_CRITERIA)},
        "destructive": {"type": "boolean"},
        "needs_screen": {"type": "boolean"},
    },
    "required": ["intent", "destructive", "needs_screen"],
}

_PROMPT = (
    "Classify the user's request.\n"
    + "\n".join(f"- {k}: {v}" for k, v in INTENT_CRITERIA.items())
    + (
        "\ndestructive = would delete, send, pay, install, overwrite or otherwise change something hard to undo."
        "\nneeds_screen = requires seeing what is currently on screen."
    )
)


class LiteReflex:
    def __init__(self) -> None:
        s = settings()
        self._client = genai.Client(api_key=s.gemini_api_key)
        self._model = s.gemini_lite_model

    async def decide(self, text: str) -> Decision:
        t0 = time.perf_counter()
        resp = await self._client.aio.models.generate_content(
            model=self._model,
            contents=text,
            config=gt.GenerateContentConfig(
                system_instruction=_PROMPT,
                response_mime_type="application/json",
                response_json_schema=_SCHEMA,
                thinking_config=gt.ThinkingConfig(thinking_level=gt.ThinkingLevel.MINIMAL),
                max_output_tokens=64,
            ),
        )
        d = json.loads(resp.text or "{}")
        return Decision(
            intent=d.get("intent", "agent_task"),
            intent_confidence=0.9,
            destructive=bool(d.get("destructive")),
            destructive_p=1.0 if d.get("destructive") else 0.0,
            needs_screen=bool(d.get("needs_screen")),
            needs_screen_p=1.0 if d.get("needs_screen") else 0.0,
            source="lite",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
