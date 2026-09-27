"""Real human phrasings for the reflex, from open datasets mapped onto NEO's labels.

Our own training data is templates plus LLM-generated sentences — clean, but nobody talks like
that. Public intent, dialogue, function-calling and computer-use sets (neo2/core/reflex/sources.py has
each one's pin, licence and mapping) supply how people actually phrase NEO's routes (chat /
quick_action / agent_task / stop), tools, task kinds, "is it destructive", "does it need the
screen" and "does it want an answer" — in English, Devanagari Hindi and romanised Hinglish.

    python -m neo2.core.reflex.hf_data                  # → <data_dir>/reflex_hf.jsonl + reflex_eval.jsonl
    python -m neo2.core.reflex.hf_data --with leyzer    # also the opt-in sources
    python -m neo2.core.reflex.hf_data --only clinc,massive --scale 0.5

Rows: {text, intent, tool, source, hf_intent, lang, split, group} plus, when the source says,
kind / destructive / screen / reply (left out when unknown, which is not the same as False).
``reflex_hf.jsonl`` holds train rows only; ``reflex_eval.jsonl`` is the benchmark (Hinglish-TOP,
MTOP en/hi, CLINC and MASSIVE en/hi test splits) and is never read by dataset.build(). The
benchmark sources are read whatever --only / --skip say, so the benchmark and its holdout are
always whole. A train row is held out when its sentence (compared by text_key: case, spacing and
punctuation don't count) or its group is the benchmark's, and so are the translations of such a
row (its group), transitively. Duplicates across sources are dropped; each source is capped per
intent (seeded round-robin), so ~50k public rows can't bury NEO's own.
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

from neo2.core.config import settings
from neo2.core.reflex.sources import (  # noqa: F401 — CLINC/MASSIVE/_clean are imported from here by older scripts
    BENCHMARK,
    BY_NAME,
    CLINC,
    MASSIVE,
    SOURCES,
    Fetcher,
    Source,
    clean,
    text_key,
)

_clean = clean


def _sample(rows: list[dict], src: Source, rnd: random.Random, admit, scale: float) -> tuple[list[dict], int]:
    """Round-robin over the source's keys (intent × language by default), a seeded shuffle within
    each, up to `cap` per key and `total` overall. Returns (kept, rows never looked at)."""
    cap = max(1, round(src.cap * scale))
    total = None if src.total is None else max(1, round(src.total * scale))
    buckets: dict[str, list[dict]] = {}
    for r in rows:
        buckets.setdefault(src.key_of(r), []).append(r)
    order = sorted(buckets)
    for k in order:
        rnd.shuffle(buckets[k])
    pos = dict.fromkeys(order, 0)
    taken: Counter = Counter()
    out: list[dict] = []
    active = order
    while active and (total is None or len(out) < total):
        nxt = []
        for k in active:
            b = buckets[k]
            while pos[k] < len(b):
                r = b[pos[k]]
                pos[k] += 1
                if admit(r):
                    out.append(r)
                    taken[k] += 1
                    break
            if pos[k] < len(b) and taken[k] < cap:
                nxt.append(k)
            if total is not None and len(out) >= total:
                break
        active = nxt
    return out, sum(len(b) - pos[k] for k, b in buckets.items())


def holdout(eval_rows: list[dict], rows: list[dict]) -> tuple[set[str], set[str]]:
    """(text keys, groups) no train row may have: the benchmark's sentences and groups, closed over
    the groups of `rows` — a row whose sentence is held brings its group (its translations and
    re-tokenised copies) into the holdout, their sentences bring their groups, and so on."""
    texts: dict[str, set[str]] = {}  # group → its sentences
    groups: dict[str, set[str]] = {}  # sentence → the groups it is in
    for r in rows:
        k = text_key(r["text"])
        texts.setdefault(r["group"], set()).add(k)
        groups.setdefault(k, set()).add(r["group"])
    held_text: set[str] = set()
    held_group: set[str] = set()
    todo_t = [text_key(r["text"]) for r in eval_rows]
    todo_g = [r["group"] for r in eval_rows]
    while todo_t or todo_g:
        while todo_t:
            k = todo_t.pop()
            if k not in held_text:
                held_text.add(k)
                todo_g.extend(groups.get(k, ()))
        while todo_g:
            g = todo_g.pop()
            if g not in held_group:
                held_group.add(g)
                todo_t.extend(texts.get(g, ()))
    return held_text, held_group


def build(
    names: list[str] | None = None,
    *,
    fetcher=None,
    seed: int = 0,
    scale: float = 1.0,
    log=print,
) -> tuple[list[dict], list[dict], dict]:
    """(train rows, eval rows, report). `names`: the sources to use (default: every default source).
    The benchmark sources are loaded either way: the benchmark and its holdout never shrink."""
    want = set(names or [s.name for s in SOURCES if s.default])
    chosen = [s for s in SOURCES if s.name in want]  # registry order is the dedupe priority
    loaded = [s for s in SOURCES if s.name in want or s.name in BENCHMARK]
    fx = fetcher or Fetcher(settings().data_dir / "hf_cache")
    skipped: Counter = Counter()
    raw: dict[str, list[dict]] = {}
    report: dict = {"sources": {}, "skipped_labels": skipped,
                    "not_used": {s.name: s.note for s in SOURCES if s not in loaded}}
    for s in loaded:
        t0 = time.time()
        raw[s.name] = [r for r in s.load(fx, skipped) if r]
        report["sources"][s.name] = {"licence": s.licence, "rows": len(raw[s.name]), "seconds": round(time.time() - t0, 1),
                                     "benchmark_only": s not in chosen}
        log(f"  {s.name:20s} {len(raw[s.name]):7d} mapped rows in {time.time() - t0:5.1f}s"
            + ("  (benchmark only)" if s not in chosen else ""))

    # The benchmark: every eval row, each sentence once per source and language.
    eval_rows, ev_seen = [], set()
    for s in loaded:
        for r in raw[s.name]:
            k = (s.name, r["lang"], text_key(r["text"]))
            if r["split"] == "eval" and k not in ev_seen:
                ev_seen.add(k)
                eval_rows.append(r)
    held_text, held_group = holdout(eval_rows, [r for rs in raw.values() for r in rs if r["split"] == "train"])

    seen_text: set[str] = set()
    seen_group: set[tuple[str, str]] = set()
    train: list[dict] = []
    for s in chosen:
        st = report["sources"][s.name]
        rows = [r for r in raw[s.name] if r["split"] == "train"]
        free = [r for r in rows if text_key(r["text"]) not in held_text and r["group"] not in held_group]
        st.update(eval=sum(r["split"] == "eval" for r in raw[s.name]), held_out=len(rows) - len(free))
        dups = Counter()

        def admit(r: dict, dups=dups) -> bool:
            k, g = text_key(r["text"]), (r["group"], r["lang"])
            if k in seen_text or g in seen_group:  # same sentence, or its copy in the same language
                dups["n"] += 1
                return False
            seen_text.add(k)
            seen_group.add(g)
            return True

        kept, unseen = _sample(free, s, random.Random(f"{seed}:{s.name}"), admit, scale)
        st.update(duplicate=dups["n"], capped=unseen, kept=len(kept))
        train += kept
    for s in loaded:
        if s not in chosen:
            report["sources"][s.name]["eval"] = sum(r["split"] == "eval" for r in raw[s.name])
    random.Random(seed).shuffle(train)
    return train, eval_rows, report


def load_rows(seed: int = 0, **kw) -> list[dict]:
    """Train rows only (the old entry point)."""
    return build(seed=seed, **kw)[0]


def _write(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(path)


def summary(train: list[dict], eval_rows: list[dict], report: dict) -> str:
    def line(title: str, c: Counter) -> str:
        return f"{title}: " + ", ".join(f"{k} {v}" for k, v in c.most_common())

    out = ["", f"{'source':20s} {'mapped':>7s} {'eval':>6s} {'held':>6s} {'dup':>6s} {'capped':>7s} {'kept':>6s}  licence"]
    for name, st in report["sources"].items():
        if st.get("benchmark_only"):
            out.append(f"{name:20s} {st['rows']:7d} {st.get('eval', 0):6d}  benchmark only (not in this build's train rows)")
            continue
        out.append(f"{name:20s} {st['rows']:7d} {st.get('eval', 0):6d} {st.get('held_out', 0):6d} "
                   f"{st.get('duplicate', 0):6d} {st.get('capped', 0):7d} {st.get('kept', 0):6d}  {st['licence']}")
    for name, why in report["not_used"].items():
        if why:
            out.append(f"{name:20s} not used: {why}")
    out.append(f"\ntrain {len(train)} rows")
    for title, key in (("intent", "intent"), ("lang", "lang"), ("kind", "kind")):
        out.append("  " + line(title, Counter(r.get(key, "–") for r in train)))
    out.append("  " + line("tool", Counter(r["tool"] for r in train if r["tool"])))
    for flag in ("destructive", "screen", "reply"):
        c = Counter("unknown" if flag not in r else str(r[flag]).lower() for r in train)
        out.append(f"  {flag}: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
    out.append(f"eval {len(eval_rows)} rows")
    out.append("  " + line("by source × lang", Counter(f"{r['source'].split(':')[0]}/{r['lang']}" for r in eval_rows)))
    top = [(k, v) for k, v in report["skipped_labels"].most_common() if ":unmapped:" not in k][:12]
    if top:
        out.append("filtered (top): " + ", ".join(f"{k} {v}" for k, v in top))
    unmapped = sorted((k, v) for k, v in report["skipped_labels"].items() if ":unmapped:" in k)
    if unmapped:
        out.append("unmapped labels, skipped (map them in sources.py): " + ", ".join(f"{k} {v}" for k, v in unmapped))
    return "\n".join(out)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", default="", help="comma-separated sources (default: all default ones)")
    ap.add_argument("--with", dest="also", default="", help="add opt-in sources, e.g. leyzer")
    ap.add_argument("--skip", default="", help="comma-separated sources to leave out")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every per-source cap")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="train rows (default <data_dir>/reflex_hf.jsonl)")
    ap.add_argument("--eval-out", type=Path, default=None, help="benchmark (default <data_dir>/reflex_eval.jsonl)")
    ap.add_argument("--list", action="store_true", help="list the sources and exit")
    a = ap.parse_args()
    if a.list:
        for s in SOURCES:
            print(f"{s.name:20s} {'default' if s.default else 'opt-in ':8s} cap {s.cap:<5d} {s.licence}")
        return
    split = lambda v: [x for x in v.split(",") if x]  # noqa: E731
    unknown = [n for n in split(a.only) + split(a.also) + split(a.skip) if n not in BY_NAME]
    if unknown:
        sys.exit(f"unknown source(s): {', '.join(unknown)} (see --list)")
    names = split(a.only) or [s.name for s in SOURCES if s.default]
    names = [n for n in dict.fromkeys(names + split(a.also)) if n not in split(a.skip)]
    t0 = time.time()
    train, eval_rows, report = build(names, seed=a.seed, scale=a.scale)
    data = settings().data_dir
    data.mkdir(parents=True, exist_ok=True)
    out, eout = a.out or data / "reflex_hf.jsonl", a.eval_out or data / "reflex_eval.jsonl"
    _write(out, train)
    _write(eout, eval_rows)
    print(summary(train, eval_rows, report))
    print(f"\n{len(train)} train rows → {out}\n{len(eval_rows)} eval rows → {eout}\n({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    from neo2.core import log_to_stdout

    log_to_stdout()  # the providers' status lines, as this command always printed them
    main()
