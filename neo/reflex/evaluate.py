"""How good is a reflex head? Intent accuracy by language and by source, per-class recall, what
NEO acts on for destructive and stop, tool and kind accuracy, calibration, and decision latency.

    python -m neo.reflex.evaluate                                # the current head
    python -m neo.reflex.evaluate --head old.json --head new.json   # side by side (before / after)
    python -m neo.reflex.evaluate --eval bench.jsonl --latency 0

Rows: the benchmark in ``reflex_eval.jsonl`` ({text, intent, tool, kind?, lang, source}) when there
is one, plus the training data's held-out rows (the hash slices finetune never trains on). Each
head is run on its own encoder, so heads for different encoders compare fairly.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from neo.config import settings
from neo.reflex.features import embed_only, head_probs, load_encoder, parse_head
from neo.reflex.finetune import GENERATED, HF, ece, labels, load_cache, rule_signals, slices, training_rows
from neo.reflex.schema import INTENT_CRITERIA

_INTENTS = list(INTENT_CRITERIA)
_STOP = _INTENTS.index("stop")


def _bench(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(r, dict) and r.get("text") and r.get("intent") in INTENT_CRITERIA:
            rows.append({**r, "source": f"bench:{r.get('source', '')}"})
    return rows


def _slices(rows: list[dict], held: dict[str, np.ndarray], lab: dict) -> dict[str, np.ndarray]:
    """Named row groups to report accuracy on (a row can be in several)."""
    n = len(rows)
    bench = np.array([r["source"].startswith("bench:") for r in rows], dtype=bool)
    src = np.array([r.get("source", "") for r in rows], dtype=object)
    out: dict[str, np.ndarray] = {"all": np.ones(n, dtype=bool)}
    if bench.any():
        out["bench"] = bench
        for lang in sorted({str(r.get("lang") or "?") for r, b in zip(rows, bench, strict=True) if b}):
            out[f"bench lang={lang}"] = bench & np.array([str(r.get("lang") or "?") == lang for r in rows])
        dataset = np.array([str(x).split(":")[1] if b else "" for x, b in zip(src, bench, strict=True)], dtype=object)
        for s in sorted(set(dataset[bench])):
            out[f"bench {s or '?'}"] = bench & (dataset == s)
    out["held-out"] = ~bench
    out["held templates"] = ~bench & (src == "template")
    out["held generated"] = ~bench & (src == GENERATED)
    out["held public"] = ~bench & (src == HF)
    out["held own rows"] = ~bench & held["own"]
    for s in ("en", "latin-undecided", "devanagari"):
        out[f"script {s}"] = lab["script"] == s
    return {k: v for k, v in out.items() if v.any()}


def _metrics(h: dict, X: np.ndarray, lab: dict, groups: dict, rules: dict) -> dict:
    Xs = (X - h["mu"]) / h["sd"]
    y = lab["intent"]
    P = head_probs(h["intent"], Xs)
    pred = P.argmax(1)
    ok = pred == y
    m: dict[str, float] = {}
    for g, mask in groups.items():
        m[f"acc {g}"] = float(ok[mask].mean())
    for i, c in enumerate(_INTENTS):
        if (y == i).any():
            m[f"recall {c}"] = float(ok[y == i].mean())
    m["ECE intent"] = ece(P, y)
    stop = y == _STOP
    if stop.any():
        m["stop recall (head)"] = float((pred[stop] == _STOP).mean())
        m["stop recall (+rule)"] = float(((pred == _STOP) | rules["stop"])[stop].mean())
    for name in ("destructive", "needs_screen"):
        trusted = lab[f"{name}_trusted"]
        if h[name] is None or not trusted.any():
            continue
        p = head_probs(h[name], Xs[trusted])[:, 1]
        both = np.maximum(p, rules[name][trusted])
        pos, neg = lab[name][trusted] == 1, lab[name][trusted] == 0
        if pos.any():
            m[f"{name} recall (head)"] = float((p[pos] >= 0.5).mean())
            m[f"{name} recall (+rule)"] = float((both[pos] >= 0.5).mean())
        if neg.any():
            m[f"{name} false alarms (+rule)"] = float((both[neg] >= 0.5).mean())
    have = lab["reply"] >= 0
    if h["reply"] is not None and have.any():
        m["reply acc"] = float((head_probs(h["reply"], Xs[have]).argmax(1) == lab["reply"][have]).mean())
    for name in ("tool", "kind"):
        have = lab[name] != ""
        if h[name] is None or not have.any():
            continue
        classes = np.array(h[name]["classes"], dtype=object)
        P_k = head_probs(h[name], Xs[have])
        m[f"{name} acc"] = float((classes[P_k.argmax(1)] == lab[name][have]).mean())
        m[f"{name} ECE"] = ece(P_k, np.array([list(classes).index(c) if c in classes else -1 for c in lab[name][have]]))
        m[f"{name} n"] = float(have.sum())
    return m


def _gpu_busy() -> int:
    """The GPU's utilisation right now (%), or -1: latency measured beside other GPU work is slower."""
    try:
        out = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return -1
    m = re.search(r'"Device Utilization %"=(\d+)', out)
    return int(m.group(1)) if m else -1


def _latency(agent, encoder: str, path: Path, texts: list[str]) -> dict[str, float]:
    """p50/p90 of a whole decide() (embed + every head + rules), one utterance at a time."""
    from neo.reflex.laya_reflex import LayaReflex

    r = LayaReflex(encoder=encoder, head=path, agent=agent)
    for t in texts[:5]:
        r.decide(t)
    busy = [_gpu_busy()]
    ms = []
    for t in texts:
        t0 = time.perf_counter()
        r.decide(t)
        ms.append((time.perf_counter() - t0) * 1000)
    busy.append(_gpu_busy())
    return {"p50 ms": float(np.median(ms)), "p90 ms": float(np.percentile(ms, 90)), "GPU busy %": float(max(busy))}


def main() -> None:
    from neo import models

    s = settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=Path, action="append", help="a head file (repeat to compare); default: the current one")
    ap.add_argument("--eval", type=Path, default=s.data_dir / "reflex_eval.jsonl", help="benchmark rows")
    ap.add_argument("--per-class", type=int, default=640)
    ap.add_argument("--latency", type=int, default=50, help="utterances to time (0 = skip)")
    args = ap.parse_args()
    paths = args.head or [s.data_dir / "reflex_head.json"]

    rows = training_rows(args.per_class)
    held = slices(rows)
    keep = (held["base"] | held["user"] | held["cal"]) & ~held["shadowed"]
    rows = [r for r, k in zip(rows, keep, strict=True) if k] + _bench(args.eval)
    held = {k: np.concatenate([v[keep], np.zeros(len(rows) - int(keep.sum()), dtype=bool)]) for k, v in held.items()}
    lab = labels(rows)
    groups = _slices(rows, held, lab)
    rules = rule_signals(rows)
    texts = [r["text"] for r in rows]
    print(f"{len(rows)} rows: " + ", ".join(f"{g} {int(m.sum())}" for g, m in groups.items()))

    by_encoder: dict[str, list[Path]] = {}
    for p in paths:
        sub = json.loads(p.read_text()).get("subfolder", "")
        enc = next(e for e, f in models.LAYA_SUBFOLDERS.items() if f == sub)
        by_encoder.setdefault(enc, []).append(p)
    results: dict[str, dict] = {}
    for enc, heads in by_encoder.items():
        agent = load_encoder(enc, s.laya_device)
        cached, _ = load_cache(enc)
        missing = [t for t in dict.fromkeys(texts) if t not in cached]
        cached.update(zip(missing, embed_only(agent, missing) if missing else [], strict=True))
        X = np.hstack([np.vstack([cached[t] for t in texts]), np.ones((len(texts), 1), np.float32)]).astype(np.float32)
        for p in heads:
            h = parse_head(json.loads(p.read_text()), enc, X.shape[1] - 1)
            if h is None:
                print(f"{p}: not a head for the {enc} encoder")
                continue
            m = _metrics(h, X, lab, groups, rules)
            if args.latency:
                m.update(_latency(agent, enc, p, texts[: args.latency]))
            results[f"{p.parent.name}/{p.name} ({enc})"] = m
        del agent
    _print(results)


def _print(results: dict[str, dict]) -> None:
    if not results:
        return
    names = list(results)
    keys = list(dict.fromkeys(k for m in results.values() for k in m))
    width = max(len(k) for k in keys) + 2
    print("\n" + " " * width + "".join(f"{n[-40:]:>42s}" for n in names))
    for k in keys:
        cells = []
        for n in names:
            v = results[n].get(k)
            cells.append(f"{'—' if v is None or v != v else (f'{v:.0f}' if k.endswith(' n') else f'{v:.3f}' if abs(v) < 10 else f'{v:.1f}'):>42s}")
        print(f"{k:{width}s}" + "".join(cells))


if __name__ == "__main__":
    main()
