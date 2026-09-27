"""How good is a reflex head? Intent accuracy by language and by source, per-class recall, what
NEO acts on for destructive, stop and the screen (and how often each goes off by mistake), tool and
kind accuracy, calibration, and decision latency.

    python -m neo.reflex.evaluate                                # the current head
    python -m neo.reflex.evaluate --head old.json --head new.json   # side by side (before / after)
    python -m neo.reflex.evaluate --eval bench.jsonl --latency 0

Rows: the benchmark in ``reflex_eval.jsonl`` ({text, intent, tool?, kind?, lang?, source?}) when
there is one, minus every sentence the training data has (dataset.benchmark), plus the training
data's held-out rows (the hash slices finetune never trains on). Each head is run on its own
encoder, so heads for different encoders compare fairly.
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
from neo.reflex.dataset import benchmark
from neo.reflex.features import embed_only, head_probs, load_encoder, parse_head
from neo.reflex.finetune import (
    GENERATED,
    HF,
    cancels,
    ece,
    labels,
    load_cache,
    rule_signals,
    slices,
    training_rows,
)
from neo.reflex.schema import INTENT_CRITERIA

_INTENTS = list(INTENT_CRITERIA)
_STOP = _INTENTS.index("stop")


def _bench(path: Path, trained: list[dict]) -> list[dict]:
    """The benchmark rows, without any sentence the training data also has (NEO's templates and
    generated rows are made without looking at it): those would only flatter the head."""
    rows, dropped = benchmark(trained, path)
    if rows or dropped:
        print(f"benchmark: {len(rows)} rows ({dropped} left out: the training data has them too)")
    return [{**r, "source": f"bench:{r.get('source') or ''}"} for r in rows]


def _encoder_of(path: Path) -> str | None:
    """The encoder a head file was trained for; None (after saying why) if it isn't a head NEO can
    run."""
    from neo import models

    try:
        h = json.loads(Path(path).read_text())
        sub = h.get("subfolder", "") if isinstance(h, dict) else None
    except (OSError, ValueError) as e:
        print(f"{path}: no head there ({str(e)[:80]}) — train one with python -m neo.reflex.finetune")
        return None
    enc = next((e for e, f in models.LAYA_SUBFOLDERS.items() if f == sub), None)
    if enc is None:
        print(f"{path}: not a reflex head for a Laya checkpoint NEO has ({sub!r})")
    return enc


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
    fired = cancels(P, rules)
    can = rules["stop"] | rules["short"]  # the rows the session could cancel on at all
    if stop.any():
        m["stop recall (head)"] = float((pred[stop] == _STOP).mean())
        m["stop recall (+rule)"] = float(((pred == _STOP) | rules["stop"])[stop].mean())
        if (stop & can).any():
            m["stop cancels (short)"] = float(fired[stop & can].mean())
            m["stop cancels n"] = float((stop & can).sum())
    if (~stop & can).any():  # a short command that isn't a stop, cancelling every running job
        m["stop false cancels (short)"] = float(fired[~stop & can].mean())
        m["stop false cancels n"] = float((~stop & can).sum())
    # Only rows whose destructive / screen label is their own: on a row the rule labelled, head or
    # rule is a hit by construction.
    chat = y == _INTENTS.index("chat")
    for name in ("destructive", "needs_screen"):
        if h[name] is None:
            continue
        p = head_probs(h[name], Xs)[:, 1]
        both = np.maximum(p, rules[name]) >= 0.5
        own, yes = lab[f"{name}_labelled"], lab[name]
        pos, neg = own & (yes == 1), own & (yes == 0)
        if pos.any():
            m[f"{name} recall (head)"] = float((p[pos] >= 0.5).mean())
            m[f"{name} recall (+rule)"] = float(both[pos].mean())
            m[f"{name} recall n"] = float(pos.sum())
        if neg.any():
            m[f"{name} false alarms (+rule)"] = float(both[neg].mean())
        if (chat & (yes == 0)).any():
            m[f"{name} false alarms on chat (+rule)"] = float(both[chat & (yes == 0)].mean())
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
    s = settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=Path, action="append", help="a head file (repeat to compare); default: the current one")
    ap.add_argument("--eval", type=Path, default=s.data_dir / "reflex_eval.jsonl", help="benchmark rows")
    ap.add_argument("--per-class", type=int, default=640)
    ap.add_argument("--latency", type=int, default=50, help="utterances to time (0 = skip)")
    args = ap.parse_args()
    paths = args.head or [s.data_dir / "reflex_head.json"]
    by_encoder: dict[str, list[Path]] = {}
    for p in paths:
        if enc := _encoder_of(p):
            by_encoder.setdefault(enc, []).append(p)
    if not by_encoder:
        raise SystemExit(1)

    rows = training_rows(args.per_class)
    held = slices(rows)
    keep = (held["base"] | held["user"] | held["cal"]) & ~held["shadowed"]
    rows = [r for r, k in zip(rows, keep, strict=True) if k] + _bench(args.eval, rows)
    held = {k: np.concatenate([v[keep], np.zeros(len(rows) - int(keep.sum()), dtype=bool)]) for k, v in held.items()}
    lab = labels(rows)
    groups = _slices(rows, held, lab)
    rules = rule_signals(rows)
    texts = [r["text"] for r in rows]
    print(f"{len(rows)} rows: " + ", ".join(f"{g} {int(m.sum())}" for g, m in groups.items()))

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


# ---- the resolver's benchmark ------------------------------------------------------------------------
# tests/data/ground_eval.jsonl (built by tests/data/build_ground_eval.py): requests in English,
# Hinglish and Hindi against windows of 12/50/200 elements, hand-built dialogs and the self-test
# window. A stage of neo.tools.computer.resolve goes live only once it is precise enough here
# (SHIP_P overall, SHIP_P_SCRIPT per script) and on the user's own requests (shadow_report).

GROUND_EVAL = Path(__file__).resolve().parents[2] / "tests" / "data" / "ground_eval.jsonl"
SHIP_P = 0.97
SHIP_P_SCRIPT = 0.95
SHIP_N = 200  # answers a stage needs on real requests before its shadow numbers count


class EvalStrings:
    """The eval file's own OS strings (macOS's English→Hindi forms of its labels) as the resolver's
    string index."""

    def __init__(self, os_map: dict[str, list[str]]) -> None:
        from neo.tools.computer.phrases import norm

        self._norm = norm
        self._f = {norm(k): list(v) for k, v in os_map.items()}
        self._r: dict[str, list[str]] = {}
        for en, his in os_map.items():
            for h in his:
                for key in {norm(h), norm(h.rstrip("…"))}:
                    self._r.setdefault(key, []).append(en)

    def ready(self) -> bool:
        return True

    def forms(self, label: str) -> list[str]:
        return self._f.get(self._norm(label), [])

    def sources(self, label: str) -> list[str]:
        return self._r.get(self._norm(label), [])


def load_ground(path: Path = GROUND_EVAL) -> tuple[dict, list[dict]]:
    lines = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    head = next(x for x in lines if x.get("kind") == "vocab")
    return head, [x for x in lines if x.get("kind") == "row"]


def ground_windows(head: dict, row: dict, sizes: tuple[str, ...] = ("12", "50", "200")) -> list[tuple[str, list, set[int]]]:
    """(size, candidates, positions of the right element) for each window of a row."""
    from neo.tools.computer.resolve import Candidate

    gold = set(row["gold"])
    out = []
    if "screen" in row:
        cands = [Candidate(lab, role, modal=bool(m), id=f"e{i}") for i, (role, lab, m) in enumerate(row["screen"])]
        out.append(("hand", cands, {i for i, c in enumerate(cands) if c.label in gold}))
        return out
    for n in sizes:
        ids = row["screens"].get(n)
        if ids is None:
            continue
        cands = [Candidate(head["elements"][j][1], head["elements"][j][0], id=f"e{i}") for i, j in enumerate(ids)]
        out.append((n, cands, {i for i, c in enumerate(cands) if c.label in gold}))
    return out


def _rate(right: int, n: int) -> float | None:
    return round(right / n, 4) if n else None


def ground_eval(
    path: Path = GROUND_EVAL,
    *,
    sizes: tuple[str, ...] = ("12", "50", "200"),
    stages: tuple[str, ...] | None = ("lexical",),
    index: bool = True,
    embed=False,
    choose=False,
    rows: list[dict] | None = None,
) -> dict:
    """Run the resolver over the benchmark: how often it acts (coverage), how often it is right when
    it does (precision) — overall, per script, per stage (in cascade order) and per window size —,
    whether every commit label it acted on reached the gate, and its latency. ``stages``: the ones
    that act (None: every stage). ``index``: the eval's own OS strings as the string index (False:
    none, as before it is built). ``embed`` / ``choose``: MiniLM and Laya's choice (False: none)."""
    from neo.tools.computer import resolve as rs

    head, all_rows = load_ground(path)
    rows = all_rows if rows is None else rows
    idx = EvalStrings(head.get("os", {})) if index else False
    acting = tuple(rs.STAGES) if stages is None else tuple(stages)
    tally: dict[str, dict[str, int]] = {}
    ms: dict[str, list[float]] = {}
    commit_seen = commit_unflagged = asked = 0
    wrong: list[dict] = []

    def add(key: str, acted: bool, right: bool) -> None:
        t = tally.setdefault(key, {"n": 0, "acted": 0, "right": 0})
        t["n"] += 1
        t["acted"] += int(acted)
        t["right"] += int(acted and right)

    for row in rows:
        for size, cands, gold in ground_windows(head, row, sizes):
            t0 = time.perf_counter()
            res = rs.resolve(
                row["text"], cands, origin="user", stages=acting, index=idx, embed=embed, choose=choose,
                alias=False, log_it=False,
            )  # fmt: skip
            ms.setdefault(size, []).append((time.perf_counter() - t0) * 1000)
            acted = isinstance(res, rs.Resolved)
            right = acted and cands.index(res.candidate) in gold
            stage = res.stage if acted else ""
            for key in ("all", f"script {row['script']}", f"size {size}", f"source {row['source']}", f"lang {row['lang']}"):
                add(key, acted, right)
            # what the stages in shadow would have pressed (the cascade's later stages, MiniLM alone)
            for sh, lab in (res.shadow.get("picks") or {}).items():
                if lab is not None and sh != stage:
                    add(f"shadow {sh}", True, lab in set(row["gold"]))
                    add(f"shadow {sh} {row['script']}", True, lab in set(row["gold"]))
            if acted:
                add(f"stage {stage}", True, right)
                add(f"stage {stage} {row['script']}", True, right)
                asked += int(bool(res.ask))
                picked_commits = rs.label_commits(res.label, idx or None) or bool(row.get("commit") and right)
                if picked_commits:
                    commit_seen += 1
                    if not (res.commits or res.ask):
                        commit_unflagged += 1
                if not right and len(wrong) < 40:
                    wrong.append({"text": row["text"], "size": size, "stage": stage, "picked": res.label, "gold": row["gold"]})
    rep: dict = {"rows": len(rows), "stages": list(acting), "index": bool(index)}
    for key, t in sorted(tally.items()):
        rep[key] = {"n": t["n"], "acted": t["acted"], "coverage": _rate(t["acted"], t["n"]), "precision": _rate(t["right"], t["acted"])}
    rep["commit labels acted on"] = commit_seen
    rep["commit labels not flagged for the gate"] = commit_unflagged
    rep["asked first (label the gate can't read)"] = asked
    rep["latency ms"] = {
        size: {"p50": round(float(np.median(v)), 3), "p90": round(float(np.percentile(v, 90)), 3)} for size, v in ms.items()
    }
    rep["ship"] = ship(rep)
    rep["wrong"] = wrong
    return rep


def ship(rep: dict) -> dict[str, bool | None]:
    """Per stage: does it clear the bar (SHIP_P overall, SHIP_P_SCRIPT on each script it answered)?
    None when it answered nothing."""
    out: dict[str, bool | None] = {}
    for key, v in rep.items():
        if not key.startswith("stage ") or key.count(" ") != 1 or not isinstance(v, dict):
            continue
        stage = key.split(" ", 1)[1]
        if not v["acted"]:
            out[stage] = None
            continue
        ok = (v["precision"] or 0.0) >= SHIP_P
        for sc in ("latin", "devanagari"):
            s = rep.get(f"stage {stage} {sc}")
            if s and s["acted"]:
                ok = ok and (s["precision"] or 0.0) >= SHIP_P_SCRIPT
        out[stage] = ok
    return out


def shadow_report(path: Path | None = None) -> dict:
    """The resolver's own shadow log (neo.decide.shadow ``ui_resolve`` / ``ui_resolve_outcome``
    lines): per stage, how often its pick was what was finally pressed and verified — overall and
    per script — and whether that clears the bar on at least SHIP_N real answers."""
    from neo.decide import shadow

    p = Path(path) if path is not None else shadow.path()
    lines = []
    if p.exists():
        for x in p.read_text(encoding="utf-8").splitlines():
            try:
                lines.append(json.loads(x))
            except ValueError:
                continue
    script = {x.get("rid"): x.get("script", "") for x in lines if x.get("kind") == "ui_resolve"}
    tally: dict[str, dict[str, int]] = {}
    for x in lines:
        if x.get("kind") != "ui_resolve_outcome" or not x.get("verified"):
            continue
        for k, v in x.items():
            if not k.endswith("_right"):
                continue
            stage = k[: -len("_right")]
            for key in (stage, f"{stage} {script.get(x.get('rid'), '')}".strip()):
                t = tally.setdefault(key, {"n": 0, "right": 0})
                t["n"] += 1
                t["right"] += int(bool(v))
    out: dict = {"resolutions": len(script)}
    for key, t in sorted(tally.items()):
        out[key] = {"n": t["n"], "precision": _rate(t["right"], t["n"])}
    for stage in {k.split(" ")[0] for k in tally}:
        v = out.get(stage, {})
        ok = v.get("n", 0) >= SHIP_N and (v.get("precision") or 0) >= SHIP_P
        for sc in ("latin", "devanagari"):
            s = out.get(f"{stage} {sc}")
            if s:
                ok = ok and (s["precision"] or 0) >= SHIP_P_SCRIPT
        out[f"ship {stage}"] = ok
    return out


def _ground_main(argv: list[str]) -> None:
    """python -m neo.reflex.evaluate --ground [--models] [--all-stages] [--no-index]"""
    ap = argparse.ArgumentParser(prog="python -m neo.reflex.evaluate --ground")
    ap.add_argument("--ground", action="store_true")
    ap.add_argument("--path", type=Path, default=GROUND_EVAL)
    ap.add_argument("--all-stages", action="store_true", help="every stage acts (default: the lexical stage only)")
    ap.add_argument("--no-index", action="store_true", help="as before the OS string index is built")
    ap.add_argument("--models", action="store_true", help="load MiniLM and Laya's typed-decisions checkpoint (local, opt-in)")
    ap.add_argument("--shadow", action="store_true", help="report the shadow log of real requests instead")
    a = ap.parse_args(argv)
    if a.shadow:
        print(json.dumps(shadow_report(), indent=1, ensure_ascii=False))
        return
    embed = choose = False
    if a.models:
        embed, choose = _ground_models()
    rep = ground_eval(a.path, stages=None if a.all_stages else ("lexical",), index=not a.no_index, embed=embed, choose=choose)
    print(json.dumps(rep, indent=1, ensure_ascii=False))


def _ground_models():
    """MiniLM and Laya's typed-decisions checkpoint for the benchmark, from the local cache only
    (HF_HUB_OFFLINE): an opt-in run outside the tests (it loads ~1.8 GB)."""
    import os

    from neo import effects

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    # (tqdm, inside the loaders, starts Python's multiprocessing resource tracker: a process)
    with effects.allow("model", "process"):
        from sentence_transformers import SentenceTransformer

        from neo import models
        from neo.memory import embed as mem
        from neo.reflex import zeroshot

        mini = SentenceTransformer(mem.MODEL, device="cpu", revision=models.EMBEDDER_REVISION)
        path = zeroshot.checkpoint_dir(download=False)
        agent = zeroshot._open(path)
        agent.system_one({"text": "warm up"}, zeroshot._WARM)

    def embed(texts: list[str]):
        return mini.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

    def choose(request: str, options: list[str], wait_s: float):
        from neo.tools.computer import resolve as rs

        keys: list[str] = []
        crit: dict[str, str] = {}
        for i, lab in enumerate(options):
            k = lab if lab not in crit else f"{lab} ({i})"
            keys.append(k)
            crit[k] = f"the on-screen control labelled “{lab}”"
        res = agent.system_one({"request": request}, {"el": {"type": "choice", "instructions": "Which on-screen control does the request mean?", "criteria": crit}})
        probs = rs._probs(res, "el")
        if not probs:
            return None, 0.0
        pick = max(probs, key=probs.__getitem__)
        return (keys.index(pick) if pick in keys else None), float(probs[pick])

    return embed, choose


if __name__ == "__main__":
    import sys

    if "--ground" in sys.argv[1:]:
        _ground_main(sys.argv[1:])
    else:
        main()
