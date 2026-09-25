"""Train the reflex head (multinomial logistic regression over Laya features).

One pass extracts every feature block and caches it in ``~/.neo/reflex_features.npz``; heads
for several sub-question subsets are then fit offline in seconds and compared, so you can pick
the accuracy/latency trade-off (each sub-question costs ~40 ms per decision on MPS). The winner
is written to ``~/.neo/reflex_head.json`` together with the sub-question list it needs.

    python -m neo.reflex.finetune             # extract (or reuse cache), compare variants, save best
    python -m neo.reflex.finetune --refresh   # ignore the feature cache
    python -m neo.reflex.finetune --subq knowledge,device,multistep   # force one variant
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from neo.config import settings
from neo.reflex.dataset import build
from neo.reflex.features import ALL_SUBQ, assemble, dim_for, extract_all
from neo.reflex.schema import INTENT_CRITERIA

_INTENTS = list(INTENT_CRITERIA)
_MS_PER_SUBQ = 42  # measured on M3 Pro: 3 questions ≈ 143 ms, 11 ≈ 485 ms
_MS_BASE = 165  # 3 base questions + embedding

VARIANTS: dict[str, list[str]] = {
    "none": [],
    "core3": ["knowledge", "device", "multistep"],
    "core4": ["knowledge", "app_switch", "device", "multistep"],
    "core5": ["knowledge", "app_switch", "device", "workspace", "multistep"],
    "all": ALL_SUBQ,
}


def train(
    X: np.ndarray, y: np.ndarray, *, epochs: int = 2000, lr: float = 0.1, l2: float = 1e-4
) -> np.ndarray:
    """Full-batch gradient descent with Nesterov momentum on standardised features."""
    n, d = X.shape
    k = len(_INTENTS)
    W = np.zeros((k, d), dtype=np.float32)
    V = np.zeros_like(W)
    Y = np.eye(k, dtype=np.float32)[y]
    for _ in range(epochs):
        Wl = W + 0.9 * V
        logits = X @ Wl.T
        logits -= logits.max(1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(1, keepdims=True)
        grad = (P - Y).T @ X / n + l2 * Wl
        V = 0.9 * V - lr * grad
        W += V
    return W


def standardise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(0)
    sd = X.std(0) + 1e-6
    mu[-1], sd[-1] = 0.0, 1.0  # keep the bias column
    return mu, sd


def apply(W: np.ndarray, mu: np.ndarray, sd: np.ndarray, X: np.ndarray) -> np.ndarray:
    return ((X - mu) / sd) @ W.T


def fit_and_score(blocks, y, subq, tr, te):
    X = assemble(blocks, subq)
    mu, sd = standardise(X[tr])
    W = train((X[tr] - mu) / sd, y[tr])
    pred = apply(W, mu, sd, X[te]).argmax(1)
    acc = float((pred == y[te]).mean())
    recall = {n: float((pred[y[te] == i] == i).mean()) for i, n in enumerate(_INTENTS) if (y[te] == i).any()}
    return W, mu, sd, acc, recall, pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-extract features even if cached")
    ap.add_argument("--subq", type=str, default="", help="comma-separated sub-questions to force")
    ap.add_argument("--per-class", type=int, default=320)
    ap.add_argument("--min-acc", type=float, default=0.94, help="pick the fastest variant at or above this")
    args = ap.parse_args()

    cache = settings().data_dir / "reflex_features.npz"
    rows = build(args.per_class)
    texts = [r["text"] for r in rows]
    y = np.array([_INTENTS.index(r["intent"]) for r in rows])

    if cache.exists() and not args.refresh:
        z = np.load(cache, allow_pickle=True)
        if list(z["texts"]) == texts:
            blocks = {k: z[k] for k in ("emb", "sub", "zs")}
            print(f"using cached features {cache}", file=sys.stderr)
        else:
            blocks = None
    else:
        blocks = None
    if blocks is None:
        import laya

        t0 = time.time()
        agent = laya.load("convaiinnovations/laya", device=settings().laya_device)
        print(f"laya loaded in {time.time() - t0:.1f}s", file=sys.stderr)
        t0 = time.time()
        blocks = extract_all(agent, texts, progress=True)
        print(f"features in {time.time() - t0:.0f}s", file=sys.stderr)
        np.savez(cache, texts=np.array(texts, dtype=object), **blocks)

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(rows))
    cut = int(len(rows) * 0.8)
    tr, te = idx[:cut], idx[cut:]
    print(f"zero-shot accuracy: {(blocks['zs'][te].argmax(1) == y[te]).mean():.3f}   (n={len(te)})")

    variants = {"forced": [s for s in args.subq.split(",") if s]} if args.subq else VARIANTS
    results = []
    for name, subq in variants.items():
        W, mu, sd, acc, recall, pred = fit_and_score(blocks, y, subq, tr, te)
        ms = _MS_BASE + _MS_PER_SUBQ * len(subq)
        results.append((name, subq, acc, ms, W, mu, sd, recall, pred))
        print(
            f"  {name:7s} subq={len(subq)}  ~{ms} ms  held-out {acc:.3f}  "
            + "  ".join(f"{k[:5]} {v:.2f}" for k, v in recall.items())
        )

    ok = [r for r in results if r[2] >= args.min_acc] or [max(results, key=lambda r: r[2])]
    best = min(ok, key=lambda r: r[3])
    name, subq, acc, ms, W, mu, sd, recall, pred = best
    head_path = settings().data_dir / "reflex_head.json"
    head_path.write_text(
        json.dumps(
            {
                "W": W.tolist(),
                "mu": mu.tolist(),
                "sd": sd.tolist(),
                "intents": _INTENTS,
                "subq": subq,
                "dim": dim_for(subq),
            }
        )
    )
    print(f"\nchose {name} (subq={subq}) → held-out {acc:.3f}, ~{ms} ms/decision → {head_path}")
    bad = [(texts[j], _INTENTS[y[j]], _INTENTS[pred[k]]) for k, j in enumerate(te) if pred[k] != y[j]][:12]
    for t, gold, got in bad:
        print(f"  ✗ {t!r}: {gold} → {got}")


if __name__ == "__main__":
    main()
