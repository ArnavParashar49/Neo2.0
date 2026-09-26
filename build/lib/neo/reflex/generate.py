"""Grow the reflex training set with LLM-generated utterances, cross-checked by a second LLM.

Gemini and Groq are both free. One model generates candidates for a class in a given style; the
other labels each candidate blind. Only agreed labels are kept. Output: ``~/.neo/reflex_generated.jsonl``
which ``dataset.build()`` merges into training.

    python -m neo.reflex.generate               # ~3k items, ~10 min with free-tier pacing
    python -m neo.reflex.generate --per-style 40 --styles 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys

from neo.config import settings
from neo.providers.base import Message
from neo.reflex.dataset import build as build_templates
from neo.reflex.schema import INTENT_CRITERIA

_INTENTS = list(INTENT_CRITERIA)

STYLES = [
    "terse, two to five words, lowercase, no punctuation — the way speech-to-text transcribes a quick command",
    "natural spoken sentences, some with 'can you' / 'could you' / 'please', mixed lengths",
    "indirect or conversational phrasing where the intent is implied rather than stated",
    "with realistic speech-recognition errors: dropped words, homophones, run-on sentences, no capitals",
    "specific and detailed, naming real macOS apps, websites, file names, people, times",
    "questions and requests a student or developer would make during a workday",
]

_GEN_SYS = (
    "You write training data for a voice assistant that runs on a Mac. Output ONLY a JSON array of strings, "
    "nothing else. Each string is something a user might say to the assistant. No numbering, no duplicates, "
    "no quotes inside strings, English only."
)


def _gen_prompt(intent: str, style: str, n: int, seed: int) -> str:
    others = "\n".join(f"- NOT {k}: {v}" for k, v in INTENT_CRITERIA.items() if k != intent)
    return (
        f"Generate {n} distinct utterances that are clearly of this kind:\n"
        f"- {intent}: {INTENT_CRITERIA[intent]}\n\n"
        f"They must NOT fit any of these other kinds:\n{others}\n\n"
        f"Style: {style}.\nVariation seed: {seed}. Make them varied in topic, length and wording."
    )


_LABEL_SYS = (
    "Classify each utterance said to a Mac voice assistant into exactly one class. Classes:\n"
    + "\n".join(f"- {k}: {v}" for k, v in INTENT_CRITERIA.items())
    + "\nOutput ONLY a JSON array of class names, one per input, same order."
)


def _parse_list(text: str) -> list:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        v = json.loads(m.group(0))
        return v if isinstance(v, list) else []
    except json.JSONDecodeError:
        return []


async def _ask(provider, system: str, prompt: str, max_tokens: int = 4000) -> str:
    turn = await provider.complete([Message.user(prompt)], system=system, effort="low", max_tokens=max_tokens)
    return turn.text


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-style", type=int, default=50)
    ap.add_argument("--styles", type=int, default=len(STYLES))
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    from neo.providers.gemini import GeminiProvider
    from neo.providers.openai_compat import groq

    grq = groq()

    class _GeminiPool:
        """Free-tier quotas are per model: rotate 3.8 → 3.5 → 3.5-Lite on 429 instead of stalling."""

        def __init__(self) -> None:
            self.models = [
                GeminiProvider(m) for m in ("gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite")
            ]
            self.i = 0

        async def complete(self, *a, **kw):
            for _ in range(len(self.models)):
                try:
                    return await self.models[self.i].complete(*a, **kw)
                except Exception as e:  # noqa: BLE001
                    if "429" not in str(e):
                        raise
                    self.i = (self.i + 1) % len(self.models)
            raise RuntimeError("all gemini models exhausted")

    gem = _GeminiPool()
    rnd = random.Random(args.seed)
    existing = {r["text"].lower() for r in build_templates()}
    out_path = settings().data_dir / "reflex_generated.jsonl"
    seen: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                seen.add(json.loads(line)["text"].lower())

    # ---- generate ------------------------------------------------------------------------
    cands: list[tuple[str, str, str]] = []  # (text, intent, generator)
    jobs = [(i, s) for i in _INTENTS for s in STYLES[: args.styles]]
    rnd.shuffle(jobs)
    for k, (intent, style) in enumerate(jobs):
        gen_name, gen = ("gemini", gem) if k % 2 == 0 else ("groq", grq)
        try:
            text = await _ask(
                gen, _GEN_SYS, _gen_prompt(intent, style, args.per_style, rnd.randint(1, 10_000))
            )
            items = [
                str(x).strip() for x in _parse_list(text) if isinstance(x, str) and 2 <= len(str(x)) <= 160
            ]
        except Exception as e:  # noqa: BLE001 — free tiers hiccup; keep going
            print(f"  gen {gen_name}/{intent} failed: {str(e)[:80]}", file=sys.stderr)
            items = []
        fresh = [t for t in items if t.lower() not in existing and t.lower() not in seen]
        for t in fresh:
            seen.add(t.lower())
            cands.append((t, intent, gen_name))
        print(f"  [{k + 1}/{len(jobs)}] {gen_name:6s} {intent:12s} +{len(fresh)}", file=sys.stderr)
        await asyncio.sleep(6 if gen_name == "gemini" else 2)  # 10 RPM / 30 RPM free tiers
    print(f"{len(cands)} candidates", file=sys.stderr)

    # ---- cross-label with the other model ------------------------------------------------
    kept = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for gen_name in ("gemini", "groq"):
            batch_src = [c for c in cands if c[2] == gen_name]
            labeller_name, labeller = ("groq", grq) if gen_name == "gemini" else ("gemini", gem)
            for i in range(0, len(batch_src), 40):
                chunk = batch_src[i : i + 40]
                prompt = "Utterances:\n" + "\n".join(f"{j + 1}. {t}" for j, (t, _, _) in enumerate(chunk))
                try:
                    labels = _parse_list(await _ask(labeller, _LABEL_SYS, prompt, max_tokens=600))
                except Exception as e:  # noqa: BLE001
                    print(f"  label {labeller_name} failed: {str(e)[:80]}", file=sys.stderr)
                    labels = []
                for (t, intent, g), lab in zip(chunk, labels, strict=False):
                    if lab == intent:
                        f.write(
                            json.dumps({"text": t, "intent": intent, "source": f"{g}>{labeller_name}"}) + "\n"
                        )
                        kept += 1
                print(
                    f"  labelled {min(i + 40, len(batch_src))}/{len(batch_src)} via {labeller_name} — kept {kept}",
                    file=sys.stderr,
                )
                await asyncio.sleep(6 if labeller_name == "gemini" else 2)
    print(f"kept {kept} agreed items → {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
