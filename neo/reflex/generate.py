"""Grow the reflex training set with LLM-generated utterances, cross-checked by a second LLM.

Gemini and Groq are both free. One model generates candidates for a class in a given style; the
other labels each candidate blind. Only agreed labels are kept. Output: ``~/.neo/reflex_generated.jsonl``
which ``dataset.build()`` merges into training.

    python -m neo.reflex.generate               # ~3k items, ~10 min with free-tier pacing
    python -m neo.reflex.generate --per-style 40 --styles 3

Targeted runs write at most 300 rows at one boundary the heads get wrong, both sides of it:

    python -m neo.reflex.generate --focus chat:quick_action          # two intents
    python -m neo.reflex.generate --focus tool:open_app:play_music   # tools; also kind:…, destructive:yes:no
    python -m neo.reflex.generate --from-confusion confusion.json    # the most confused pairs
    python -m neo.reflex.generate --lang hinglish                    # Hinglish stop, chat, destructive tasks

The generators see the two classes and the *shape* of utterances that fell between them (every
word that isn't common in the public data replaced by a placeholder), never a request as said.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

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

_LANGS = {
    "en": "English only.",
    "hinglish": "Hinglish only: Hindi and English mixed the way people in India talk, written in Latin "
    "letters, never Devanagari (like 'volume thoda kam karo', 'kal subah ka weather kya hai').",
}


def _gen_sys(lang: str = "en") -> str:
    return (
        "You write training data for a voice assistant that runs on a Mac. Output ONLY a JSON array, "
        "nothing else. Each item is something a user might say to the assistant. No numbering, no "
        "duplicates, no quotes inside strings. " + _LANGS[lang]
    )


def _gen_prompt(intent: str, style: str, n: int, seed: int) -> str:
    others = "\n".join(f"- NOT {k}: {v}" for k, v in INTENT_CRITERIA.items() if k != intent)
    return (
        f"Generate {n} distinct utterances, as a JSON array of strings, that are clearly of this kind:\n"
        f"- {intent}: {INTENT_CRITERIA[intent]}\n\n"
        f"They must NOT fit any of these other kinds:\n{others}\n\n"
        f"Style: {style}.\nVariation seed: {seed}. Make them varied in topic, length and wording."
    )


_LABEL_SYS = (
    "Classify each utterance said to a Mac voice assistant into exactly one class. Classes:\n"
    + "\n".join(f"- {k}: {v}" for k, v in INTENT_CRITERIA.items())
    + "\nOutput ONLY a JSON array of class names, one per input, same order."
)


def label_prompt(texts: list[str]) -> str:
    return "Utterances:\n" + "\n".join(f"{j + 1}. {t}" for j, t in enumerate(texts))


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


def safe_error(e: Exception) -> str:
    """An error for the log: any API key blanked out, then cut short."""
    s = settings()
    msg = str(e)
    for k in (*s.gemini_keys, s.groq_api_key, s.anthropic_api_key):
        if k:
            msg = msg.replace(k, "***")
    return msg[:100]


# ---- targeted generation: one contrastive pair at a time ---------------------------------------
CYCLE_CAP = 300  # rows per run: enough to move one boundary, too few to swamp the rest of the data
_BATCH = 40  # utterances per generation call (and so per labelling call)
_CALLS_PER_SIDE = 4  # a side whose rows keep being rejected is given up on, not retried forever
_PREFIXES = ("tool", "kind", "destructive")
_DESTRUCTIVE = {
    "yes": "a task that would delete, send, pay, purchase, install, uninstall, overwrite or otherwise "
    "change something that is hard to undo",
    "no": "a task that only reads, finds, checks, opens, drafts or shows things — nothing hard to undo",
}
# Where Hinglish is weakest (research_laya 3-4, research_datasets A): stop is 0.08-0.25 for every
# head, small talk goes to an action 98% of the time, destructive recall is 0.17-0.33.
HINGLISH_WEAK = ("stop:quick_action", "chat:quick_action", "destructive:yes:no")


@dataclass
class Target:
    """One side of a contrastive pair: what its rows mean and the fields they carry."""

    key: str  # chat · tool:open_app · kind:send_email · destructive:yes
    criteria: str
    intent: str | None  # the route its rows take; None = both models must state one (task kinds)
    fields: dict = field(default_factory=dict)


@dataclass
class Pair:
    a: Target
    b: Target
    n: int = 0  # how often the heads confused them
    # class key → texts of that class that were read as the other one. Local only: prompts get
    # their shapes (see `patterns`), never the texts.
    examples: dict[str, list[str]] = field(default_factory=dict)


def _tool_description(name: str) -> str:
    from neo.agent.registry import registry
    from neo.tools import load_all

    load_all()
    t = registry().get(name)
    if t is None:
        raise ValueError(f"unknown tool {name!r}")
    return t.description


def _kind_description(name: str) -> str:
    from neo.decide.engine import TASKS

    if name not in TASKS:
        raise ValueError(f"unknown task kind {name!r}")
    return TASKS[name].description


def target(key: str) -> Target:
    head, _, name = key.partition(":")
    if not name and key in INTENT_CRITERIA:
        return Target(key, INTENT_CRITERIA[key], key)
    if head == "tool" and name:
        desc = _tool_description(name)
        return Target(key, f"one quick command that NEO's `{name}` tool does in one step: {desc}", "quick_action", {"tool": name})
    if head == "kind" and name:
        return Target(key, _kind_description(name), None, {"kind": name})
    if head == "destructive" and name in _DESTRUCTIVE:
        return Target(key, _DESTRUCTIVE[name], "agent_task", {"destructive": name == "yes"})
    raise ValueError(f"unknown class {key!r}: use an intent ({', '.join(_INTENTS)}), tool:X, kind:X or destructive:yes|no")


def parse_focus(spec: str) -> Pair:
    """``chat:quick_action`` · ``tool:open_app:tool:play_music`` (or ``tool:open_app:play_music``:
    a bare second name keeps the first one's prefix) · ``destructive:yes:no``."""
    toks, keys, prefix, i = spec.split(":"), [], "", 0
    while i < len(toks):
        if toks[i] in _PREFIXES and i + 1 < len(toks):
            prefix = toks[i]
            keys.append(f"{prefix}:{toks[i + 1]}")
            i += 2
        else:
            keys.append(toks[i] if toks[i] in INTENT_CRITERIA or not prefix else f"{prefix}:{toks[i]}")
            i += 1
    if len(keys) != 2 or keys[0] == keys[1]:
        raise ValueError(f"--focus wants two different classes A:B, got {spec!r}")
    return Pair(target(keys[0]), target(keys[1]))


def _qualify(head: str, name) -> str:
    name = str(name).strip().lower() if head == "destructive" else str(name).strip()
    if ":" in name or head not in _PREFIXES:
        return name
    if head == "destructive":
        name = "yes" if name in ("yes", "true", "1") else "no" if name in ("no", "false", "0") else name
    return f"{head}:{name}"


def read_confusion(path: Path, top: int = 3) -> list[Pair]:
    """The `top` most confused pairs from a confusion summary.

    ``{"pairs": [[gold, predicted, count], ...]}``, optionally with a ``"head"`` (tool, kind or
    destructive) that bare names belong to, or one such block per head: ``{"tool": {"pairs": …}}``.
    A pair may carry a fourth element, texts of `gold` that were read as `predicted`. A→B and B→A
    are one boundary: their counts add up."""
    data = json.loads(Path(path).read_text())
    if "pairs" in data:
        blocks = [(data.get("head", ""), data["pairs"])]
    else:
        blocks = [(h, v.get("pairs", []) if isinstance(v, dict) else v) for h, v in data.items() if h in ("intent", *_PREFIXES)]
    merged: dict[frozenset, Pair] = {}
    for head, pairs in blocks:
        for p in pairs:
            if not isinstance(p, list) or len(p) < 3:
                continue
            gold, pred = _qualify(head, p[0]), _qualify(head, p[1])
            if gold == pred:
                continue
            try:
                pair = merged.get(frozenset((gold, pred))) or Pair(target(gold), target(pred))
            except ValueError as e:
                print(f"  skipped {gold}→{pred}: {e}", file=sys.stderr)
                continue
            merged[frozenset((gold, pred))] = pair
            pair.n += int(p[2])
            if len(p) > 3 and isinstance(p[3], list):
                pair.examples.setdefault(gold, []).extend(str(t) for t in p[3])
    return sorted(merged.values(), key=lambda p: -p.n)[:top]


def _class_of(row: dict, like: Target) -> str:
    """The row's class in `like`'s vocabulary (an intent, tool:X, kind:X or destructive:yes|no)."""
    head = like.key.partition(":")[0] if ":" in like.key else ""
    if not head:
        return row.get("intent", "")
    if head == "destructive":
        return f"destructive:{'yes' if row['destructive'] else 'no'}" if "destructive" in row else ""
    return f"{head}:{row[head]}" if row.get(head) else ""


def add_corrections(pairs: list[Pair], rows: list[dict]) -> None:
    """Requests the user corrected (a non-empty ``wrong``) are examples of the pair they fell between."""
    for r in rows:
        wrong = {str(w) for w in r.get("wrong") or []}
        for p in pairs:
            for t, o in ((p.a, p.b), (p.b, p.a)):
                if wrong & {o.key, o.key.split(":", 1)[-1]} and _class_of(r, t) == t.key:
                    p.examples.setdefault(t.key, []).append(r["text"])


_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*|\d+")


def public_vocab(rows: list[dict], private: set[str], min_share: float = 0.005) -> set[str]:
    """The only words a prompt may show: those in at least `min_share` of the rows (and 3) that
    aren't the user's own (`private`: their lowercased texts).

    Common is not enough — NEO's templates repeat names and places. At 0.5% of the public rows,
    what survives of the people, places, titles and message slots of TOPv2 utterances (outside
    the vocabulary's data) is pronouns and words like "work" or "tomorrow"; at 3 rows, 54% of
    those values reached the prompt whole."""
    public = [r["text"].lower() for r in rows if r["text"].lower() not in private]
    n: Counter = Counter()
    for t in public:
        n.update(set(_WORD.findall(t)))
    floor = max(3, math.ceil(min_share * len(public)))
    return {w for w, c in n.items() if c >= floor}


def abstract(text: str, vocab: set[str]) -> str:
    """The shape of an utterance: numbers become {n}, words outside `vocab` {x} (a run of them one)."""
    out: list[str] = []
    for w in _WORD.findall(text.lower()):
        tok = "{n}" if w.isdigit() else w if w in vocab else "{x}"
        if not (tok[0] == "{" and out and out[-1] == tok):
            out.append(tok)
    return " ".join(out)


def patterns(texts: list[str], vocab: set[str], limit: int = 12) -> list[str]:
    """Distinct shapes, most frequent first. A shape with no placeholder is the request itself, and
    one that is mostly placeholders says nothing: neither is used."""
    n: Counter = Counter()
    for t in texts:
        shape = abstract(t, vocab)
        words = [w for w in shape.split() if w[0] != "{"]
        if "{" in shape and len(words) >= 2 and 2 * len(words) >= len(shape.split()):
            n[shape] += 1
    return [s for s, _ in n.most_common(limit)]


def _routes() -> str:
    return "\n".join(f"- {k}: {v}" for k, v in INTENT_CRITERIA.items())


def _focus_prompt(t: Target, o: Target, shapes: dict[str, list[str]], style: str, n: int, seed: int) -> str:
    fmt = "a JSON array of strings" if t.intent else "a JSON array of [utterance, route] pairs"
    lines = [
        f"Generate {n} distinct utterances, as {fmt}, that are clearly of this kind:",
        f"- {t.key}: {t.criteria}",
        "",
        "A voice assistant keeps confusing it with this kind:",
        f"- {o.key}: {o.criteria}",
        "",
    ]
    for cls, other in ((t, o), (o, t)):
        if shapes.get(cls.key):
            lines += [
                f"Real {cls.key} requests it took for {other.key} were shaped like this "
                "({x} is any word or name, {n} any number):",
                *(f"  {s}" for s in shapes[cls.key]),
                "",
            ]
    lines.append(
        f"Write utterances close to that boundary — sharing words and structure with {o.key} — that "
        f"are still unambiguously {t.key}. New content each time; don't copy the shapes."
    )
    if not t.intent:
        lines.append(f"The route of each (how the assistant should handle it) is one of:\n{_routes()}")
    lines.append(f"Style: {style}.\nVariation seed: {seed}.")
    return "\n".join(lines)


def _pair_label_sys(p: Pair) -> str:
    return (
        "Classify each utterance said to a Mac voice assistant (English or Hinglish).\nClasses:\n"
        f"- {p.a.key}: {p.a.criteria}\n- {p.b.key}: {p.b.criteria}\n- neither: fits neither class\n"
        f"Routes (how the assistant should handle it):\n{_routes()}\n"
        "Output ONLY a JSON array with one [class, route] pair per input, same order."
    )


def _items(values: list) -> list[tuple[str, str | None]]:
    """JSON items as (first, second): "x" → ("x", None), ["x", "y"] → ("x", "y"). Anything else
    keeps its slot as ("", None), so labels stay aligned with the utterances they label."""
    out: list[tuple[str, str | None]] = []
    for v in values:
        if isinstance(v, str):
            out.append((v.strip(), None))
        elif isinstance(v, list) and v and all(isinstance(x, str) for x in v[:2]):
            out.append((v[0].strip(), v[1].strip() if len(v) > 1 else None))
        else:
            out.append(("", None))
    return out


def agreed(t: Target, gen_route: str | None, label: tuple[str, str | None]) -> str | None:
    """The route of a generated row of `t` when the labeller agrees with the generator, else None."""
    cls, lab_route = label
    route = t.intent or gen_route
    return route if cls == t.key and route in INTENT_CRITERIA and lab_route == route else None


async def _label(labeller, p: Pair, texts: list[str]) -> list[tuple[str, str | None]]:
    """(class, route) per text; [] when the answer doesn't have one label per text — a skipped
    item would shift every later label onto the wrong utterance, and in a batch of one class a
    shifted label still looks like agreement."""
    prompt = label_prompt(texts)
    if p.a.key in INTENT_CRITERIA and p.b.key in INTENT_CRITERIA:  # plain intents: the usual labeller
        labels = [(c, c) for c, _ in _items(_parse_list(await _ask(labeller, _LABEL_SYS, prompt, max_tokens=600)))]
    else:
        bare = {k.split(":", 1)[-1]: k for k in (p.a.key, p.b.key)}  # "open_app" means tool:open_app
        items = _items(_parse_list(await _ask(labeller, _pair_label_sys(p), prompt, max_tokens=1200)))
        labels = [(bare.get(c, c), r) for c, r in items]
    return labels if len(labels) == len(texts) else []


async def focus(
    pairs: list[Pair],
    models: list[tuple[str, object, float]],
    vocab: set[str],
    *,
    cap: int = CYCLE_CAP,
    lang: str = "en",
    seed: int = 1,
    seen: set[str] | None = None,
    write: Callable[[dict], None] | None = None,
    sleep=asyncio.sleep,
) -> list[dict]:
    """Rows for both sides of each pair, at most `cap` (and never over CYCLE_CAP) in all.

    `models` are (name, provider, seconds to wait after a call); each batch is generated by one and
    labelled blind by the other, and a row is kept only when the two agree. Each kept row goes to
    `write` as soon as it is agreed."""
    if len(models) < 2:
        raise ValueError("two-model agreement needs two models")
    cap = max(0, min(cap, CYCLE_CAP))
    rnd = random.Random(seed)
    seen = set() if seen is None else seen
    sides = [(p, t, o) for p in pairs for t, o in ((p.a, p.b), (p.b, p.a))]
    kept: list[dict] = []
    if not sides:
        return kept
    k = 0
    for s, (p, t, o) in enumerate(sides):
        need = cap // len(sides) + (s < cap % len(sides))
        shapes = {key: patterns(texts, vocab) for key, texts in p.examples.items()}
        got = 0
        for _ in range(_CALLS_PER_SIDE):
            if got >= need:
                break
            (gen_name, gen, gen_wait), (lab_name, lab, lab_wait) = models[k % 2], models[(k + 1) % 2]
            k += 1
            n = min(_BATCH, need - got + need // 2 + 4)  # ask for more: disagreements drop some
            prompt = _focus_prompt(t, o, shapes, rnd.choice(STYLES), n, rnd.randint(1, 10_000))
            try:
                items = _items(_parse_list(await _ask(gen, _gen_sys(lang), prompt)))
            except Exception as e:  # noqa: BLE001 — free tiers hiccup; keep going
                print(f"  gen {gen_name}/{t.key} failed: {safe_error(e)}", file=sys.stderr)
                items = []
            await sleep(gen_wait)
            fresh = []
            for text, route in items:
                if 2 <= len(text) <= 160 and text.lower() not in seen:
                    seen.add(text.lower())
                    fresh.append((text, route))
            if not fresh:
                continue
            try:
                labels = await _label(lab, p, [x for x, _ in fresh])
            except Exception as e:  # noqa: BLE001
                print(f"  label {lab_name} failed: {safe_error(e)}", file=sys.stderr)
                labels = []
            await sleep(lab_wait)
            for (text, gen_route), label in zip(fresh, labels, strict=False):  # [] = none
                route = agreed(t, gen_route, label)
                if route and got < need:
                    row = {"text": text, "intent": route, **t.fields, "source": f"{gen_name}>{lab_name}",
                           "focus": f"{p.a.key}|{p.b.key}", "lang": lang}  # fmt: skip
                    kept.append(row)
                    got += 1
                    if write:
                        write(row)
        print(f"  {t.key} (vs {o.key}): +{got}/{need}", file=sys.stderr)
    return kept


class _GeminiPool:
    """Free-tier quotas are per model: rotate 3.8 → 3.5 → 3.5-Lite on 429 instead of stalling."""

    def __init__(self) -> None:
        from neo.providers.gemini import GeminiProvider

        self.models = [GeminiProvider(m) for m in ("gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite")]
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


def _models() -> list[tuple[str, object, float]]:
    """The two free models, each with the wait its free tier needs after a call (10 / 30 RPM)."""
    from neo.providers.openai_compat import groq

    return [("gemini", _GeminiPool(), 6.0), ("groq", groq(), 2.0)]


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-style", type=int, default=50)
    ap.add_argument("--styles", type=int, default=len(STYLES))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--focus", help="a contrastive pair A:B (chat:quick_action, tool:X:Y, kind:X:Y, destructive:yes:no)")
    ap.add_argument("--from-confusion", type=Path, help='confusion summary JSON: {"pairs": [[gold, predicted, count], ...]}')
    ap.add_argument("--top", type=int, default=3, help="pairs to take from --from-confusion")
    ap.add_argument("--lang", choices=list(_LANGS), default="en", help="hinglish alone targets its weakest classes")
    ap.add_argument("--max-rows", type=int, default=CYCLE_CAP, help=f"rows per targeted run (at most {CYCLE_CAP})")
    return ap


async def main() -> None:
    args = _parser().parse_args()
    if args.focus or args.from_confusion or args.lang != "en":
        await _focused(args)
    else:
        await _broad(args)


async def _focused(args: argparse.Namespace) -> None:
    from neo.reflex.learn import EXTRA, _path, _read

    if args.focus:
        pairs = [parse_focus(args.focus)]
    elif args.from_confusion:
        pairs = read_confusion(args.from_confusion, args.top)
    else:
        pairs = [parse_focus(s) for s in HINGLISH_WEAK]
    if not pairs:
        print("no usable pairs", file=sys.stderr)
        return
    rows = build_templates()
    own = _read(_path(EXTRA))
    add_corrections(pairs, own)
    out_path = settings().data_dir / "reflex_generated.jsonl"
    with open(out_path, "a", encoding="utf-8") as f:

        def write(row: dict) -> None:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()  # an interrupted run keeps what it paid for

        kept = await focus(
            pairs, _models(), public_vocab(rows, {r["text"].lower() for r in own}), cap=args.max_rows, lang=args.lang, seed=args.seed,
            seen={r["text"].lower() for r in rows}, write=write,
        )  # fmt: skip
    print(f"kept {len(kept)} agreed items → {out_path}")


async def _broad(args: argparse.Namespace) -> None:
    models = _models()
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
        gen_name, gen, wait = models[k % 2]
        try:
            text = await _ask(
                gen, _gen_sys(), _gen_prompt(intent, style, args.per_style, rnd.randint(1, 10_000))
            )
            items = [
                str(x).strip() for x in _parse_list(text) if isinstance(x, str) and 2 <= len(str(x)) <= 160
            ]
        except Exception as e:  # noqa: BLE001 — free tiers hiccup; keep going
            print(f"  gen {gen_name}/{intent} failed: {safe_error(e)}", file=sys.stderr)
            items = []
        fresh = [t for t in items if t.lower() not in existing and t.lower() not in seen]
        for t in fresh:
            seen.add(t.lower())
            cands.append((t, intent, gen_name))
        print(f"  [{k + 1}/{len(jobs)}] {gen_name:6s} {intent:12s} +{len(fresh)}", file=sys.stderr)
        await asyncio.sleep(wait)
    print(f"{len(cands)} candidates", file=sys.stderr)

    # ---- cross-label with the other model ------------------------------------------------
    kept = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for gi, (gen_name, _, _) in enumerate(models):
            batch_src = [c for c in cands if c[2] == gen_name]
            labeller_name, labeller, wait = models[1 - gi]
            for i in range(0, len(batch_src), 40):
                chunk = batch_src[i : i + 40]
                try:
                    labels = _parse_list(
                        await _ask(labeller, _LABEL_SYS, label_prompt([t for t, _, _ in chunk]), max_tokens=600)
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"  label {labeller_name} failed: {safe_error(e)}", file=sys.stderr)
                    labels = []
                if len(labels) != len(chunk):  # can't tell which label is whose
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
                await asyncio.sleep(wait)
    print(f"kept {kept} agreed items → {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
