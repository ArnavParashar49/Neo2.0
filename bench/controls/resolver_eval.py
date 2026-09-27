"""The resolver eval: how often NEO's UI resolver acts on a request without a model, and how often
it is right when it does.

    python -m bench.controls.resolver_eval [--data tests/data/ground_eval.jsonl] [--resolver module:function]
                                           [--allow-models] [--out PATH]

Opt-in, and apart from pytest's thresholds: the resolver's later stages load local models (MiniLM,
Laya), which NEO_NO_EFFECTS=1 refuses. ``--allow-models`` runs the eval inside
``effects.allow("model")`` — local models from the Hugging Face cache only (HF_HUB_OFFLINE=1), never a
download, never a model API. Without ``--resolver`` it uses NEO's own resolver when the code has one
(``RESOLVERS``), else the built-in exact-label baseline (what acts without a model today).

Rows, one JSON object per line (the names in brackets are accepted too):
  text [q, utterance, request]          the user's words
  elements [screen, candidates]         [{"role", "label", "context", "commit"}] or [role, label, context]
                                        ("commit" is the answer key: scored, never shown to the resolver)
  gold [target, answer]                 the right element: an index, a list of indices, or a label
  n                                     window size (default: len(elements))
  lang                                  en | hinglish | hi (optional; the script is read from the text)

The resolver: ``resolve(text, elements) -> {"index": i | None, "stage": str, "p": float} | int | None``
(None, or index None: it doesn't act — the model gets the request).

Reports coverage (share acted on) and precision when it acts, overall, per script (latin,
devanagari), per window size and per stage; presses on commit labels (it should never pick one on
its own); p50/p90 ms warm, and the first call's ms (cold).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests" / "data" / "ground_eval.jsonl"
RESOLVERS = ("neo2.core.decide.resolver:resolve", "neo2.core.agent.resolver:resolve", "neo2.core.tools.computer.resolver:resolve")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")

Resolver = Callable[[str, list[dict]], Any]


def _el(e: Any) -> dict:
    if isinstance(e, dict):
        return {"role": str(e.get("role", "")), "label": str(e.get("label") or e.get("title") or ""),
                "context": str(e.get("context", "")), "commit": bool(e.get("commit", False))}  # fmt: skip
    role, label, *rest = list(e) + [""]
    return {"role": str(role), "label": str(label), "context": str(rest[0] if rest else ""), "commit": False}


def load_rows(path: Path) -> list[dict]:
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        r = json.loads(line)
        text = str(r.get("text") or r.get("q") or r.get("utterance") or r.get("request") or "")
        els = [_el(e) for e in (r.get("elements") or r.get("screen") or r.get("candidates") or [])]
        gold = r.get("gold", r.get("target", r.get("answer")))
        if isinstance(gold, str):
            gold = [j for j, e in enumerate(els) if e["label"].casefold() == gold.casefold()]
        elif isinstance(gold, int):
            gold = [gold]
        rows.append({
            "i": i, "text": text, "elements": els, "gold": set(gold or []), "n": int(r.get("n") or len(els)),
            "script": "devanagari" if _DEVANAGARI.search(text) else "latin", "lang": str(r.get("lang", "")),
        })  # fmt: skip
    return rows


_FRAME = re.compile(
    r"^\s*(?:please\s+)?(?:click|press|tap|hit|choose|select|open)\s+(?:on\s+)?(?:the\s+)?|\s+(?:button|tab|link|item|wala\s+button|"
    r"dabao|daba\s+do|karo|बटन|दबाओ|दबा\s+दो)\s*[.!।]*\s*$",
    re.I,
)


def lexical(text: str, elements: list[dict]) -> dict | None:
    """The exact-label baseline: the request (its verb and "button" stripped) is one element's
    label, and only one's."""
    want = text.strip()
    for _ in range(3):
        want = _FRAME.sub("", want).strip()
    hits = [i for i, e in enumerate(elements) if e["label"].strip().casefold() == want.casefold()]
    return {"index": hits[0], "stage": "lexical", "p": 1.0} if len(hits) == 1 else None


def find_resolver(spec: str = "") -> tuple[str, Resolver]:
    for s in [spec] if spec else RESOLVERS:
        mod, _, fn = s.partition(":")
        try:
            return s, getattr(importlib.import_module(mod), fn)
        except (ImportError, AttributeError):
            if spec:
                raise
    return "lexical (built-in)", lexical


def _pick(out: Any) -> tuple[int | None, str]:
    if out is None:
        return None, ""
    if isinstance(out, int):
        return out, ""
    if isinstance(out, dict):
        i = out.get("index")
        return (int(i) if i is not None else None), str(out.get("stage", ""))
    return None, ""


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))], 3)


def _shown(elements: list[dict]) -> list[dict]:
    """What the resolver under test is handed: each element without the rows' gold ``commit`` flag
    (the answer key its commit presses are scored against — a real resolver has to tell a commit
    label itself)."""
    return [{k: v for k, v in e.items() if k != "commit"} for e in elements]


def evaluate(rows: list[dict], resolve: Resolver) -> dict[str, Any]:
    times: list[float] = []
    per: list[dict] = []
    for r in rows:
        shown = _shown(r["elements"])
        t = time.perf_counter()
        idx, stage = _pick(resolve(r["text"], shown))
        times.append((time.perf_counter() - t) * 1000)
        acted = idx is not None
        per.append({
            "row": r["i"], "script": r["script"], "n": r["n"], "stage": stage or ("-" if not acted else "?"),
            "acted": acted, "right": acted and idx in r["gold"],
            "commit": acted and 0 <= idx < len(r["elements"]) and r["elements"][idx]["commit"],
        })  # fmt: skip

    def score(xs: list[dict]) -> dict[str, Any]:
        acted = [x for x in xs if x["acted"]]
        return {
            "rows": len(xs), "coverage": round(len(acted) / len(xs), 3) if xs else None,
            "precision": round(sum(x["right"] for x in acted) / len(acted), 3) if acted else None,
            "acted": len(acted), "wrong": sum(1 for x in acted if not x["right"]),
        }  # fmt: skip

    def group(key: str) -> dict[str, Any]:
        keys = sorted({str(x[key]) for x in per})
        return {k: score([x for x in per if str(x[key]) == k]) for k in keys}

    return {
        "overall": score(per),
        "by_script": group("script"),
        "by_window": group("n"),
        "by_stage": group("stage"),
        "commit_presses": sum(1 for x in per if x["commit"]),
        "ms": {"cold_first": round(times[0], 2) if times else None, "p50": _pct(times[1:] or times, 0.5),
               "p90": _pct(times[1:] or times, 0.9)},  # fmt: skip
        "gates": {
            "precision_97": (score(per)["precision"] or 0) >= 0.97,
            "per_script_95": all((v["precision"] or 0) >= 0.95 for v in group("script").values() if v["acted"]),
            "no_commit_presses": not any(x["commit"] for x in per),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bench.controls.resolver_eval", description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--resolver", default="", help="module:function (default: NEO's own, else the exact-label baseline)")
    ap.add_argument("--allow-models", action="store_true", help="let the resolver load local models (effects.allow('model'))")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    os.environ["NEO_NO_EFFECTS"] = "1"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if not args.data.exists():
        print(f"no eval data at {args.data} (the commands workstream writes it): nothing to do", file=sys.stderr)
        return 0
    from bench.controls.run import prepare_process

    prepare_process()
    from neo2.core import effects

    rows = load_rows(args.data)
    name, resolve = find_resolver(args.resolver)
    if args.allow_models:
        with effects.allow("model"):
            out = evaluate(rows, resolve)
    else:
        out = evaluate(rows, resolve)
    out = {"resolver": name, "data": str(args.data), **out}
    text = json.dumps(out, ensure_ascii=False, indent=1)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
