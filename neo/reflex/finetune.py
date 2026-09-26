"""Train the reflex heads: small linear models over Laya's encoder embedding.

v2 puts *everything* on the embedding (one ~20 ms encoder pass) instead of Laya's zero-shot
question answering (three questions, ~126 ms): the intent head, the destructive and
needs_screen heads — distilled from Laya's own zero-shot answers on a labelled subset — and
a tool head that names the one NEO tool a quick action needs. Zero-shot stays as the slow
path for utterances the intent head isn't sure about.

    python -m neo.reflex.finetune             # extract (or reuse cache), train, evaluate, save
    python -m neo.reflex.finetune --refresh   # ignore the feature cache
    python -m neo.reflex.finetune --teacher 3000   # how many rows get zero-shot teacher labels

Data: templates (dataset.py) + ~/.neo/reflex_generated.jsonl (generate.py)
+ ~/.neo/reflex_hf.jsonl (hf_data.py) + ~/.neo/reflex_extra.jsonl (your own misroutes).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time

import numpy as np

from neo.config import settings
from neo.reflex.dataset import build
from neo.reflex.features import embed_only, questions_for
from neo.reflex.schema import INTENT_CRITERIA

_INTENTS = list(INTENT_CRITERIA)
_SCREEN_RE = re.compile(
    r"\b(?:this|that|the|my) (?:screen|window|page|form|error|tab|article|table|dialog|button)s?\b"
    r"|\bwhat am i looking at|\bthat'?s open\b|\bon (?:my|the) screen\b|\bthis (?:one|thing)\b",
    re.I,
)
_MIN_TOOL_ROWS = 12


def train_linear(
    X: np.ndarray, y: np.ndarray, k: int, *, epochs: int = 2000, lr: float = 0.1, l2: float = 1e-4
) -> np.ndarray:
    """Full-batch gradient descent with Nesterov momentum on standardised features."""
    n, d = X.shape
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


def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    *,
    weights: np.ndarray | None = None,
    hidden: int = 256,
    epochs: int = 40,
    lr: float = 2e-3,
    wd: float = 1e-3,
    dropout: float = 0.2,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """One hidden layer (ReLU) over the standardised embedding. Returns [(W1, b1), (W2, b2)].

    The linear head tops out around 0.83 on real (crowd-sourced) phrasings; a small MLP buys
    several points for ~0.1 ms of extra inference — still nothing next to the 20 ms encoder."""
    import torch

    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    wt = torch.tensor(weights if weights is not None else np.ones(len(y)), dtype=torch.float32)
    d = X.shape[1]
    net = torch.nn.Sequential(
        torch.nn.Linear(d, hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout), torch.nn.Linear(hidden, k)
    )
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    n = len(y)
    bs = 128
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i : i + bs]
            loss = (torch.nn.functional.cross_entropy(net(Xt[b]), yt[b], reduction="none") * wt[b]).sum() / wt[b].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    l1, l2 = net[0], net[3]
    return [
        (l1.weight.detach().numpy().astype(np.float32), l1.bias.detach().numpy().astype(np.float32)),
        (l2.weight.detach().numpy().astype(np.float32), l2.bias.detach().numpy().astype(np.float32)),
    ]


def forward(layers: list[tuple[np.ndarray, np.ndarray]], X: np.ndarray) -> np.ndarray:
    """Logits for a head saved as [(W, b), ...] — one pair is linear, two is the MLP."""
    h = X
    for i, (W, b) in enumerate(layers):
        h = h @ W.T + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0.0)
    return h


def standardise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(0)
    sd = X.std(0) + 1e-6
    mu[-1], sd[-1] = 0.0, 1.0  # bias column stays 1
    return mu, sd


def apply(W: np.ndarray, mu: np.ndarray, sd: np.ndarray, X: np.ndarray) -> np.ndarray:
    return ((X - mu) / sd) @ W.T


def calibrate(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature that minimises held-out NLL: a 40-epoch MLP says 1.00 about everything, and
    the session needs probabilities it can put a threshold on (tool lane, zero-shot fallback)."""
    best_t, best_nll = 1.0, float("inf")
    for t in np.concatenate([np.arange(0.5, 3.0, 0.1), np.arange(3.0, 12.0, 0.5)]):
        z = logits / t
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        nll = float(-np.log(p[np.arange(len(y)), y] + 1e-9).mean())
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def _fit(X, y, k, tr, te, name: str, classes: list[str], *, weights=None, mlp=True):
    """Train a head; returns (layers, held-out accuracy, held-out predictions, temperature)."""
    if mlp:
        layers = train_mlp(X[tr], y[tr], k, weights=None if weights is None else weights[tr])
    else:
        layers = [(train_linear(X[tr], y[tr], k), np.zeros(k, np.float32))]
    logits = forward(layers, X[te]) if len(te) else np.zeros((0, k), np.float32)
    pred = logits.argmax(1) if len(te) else np.zeros(0, int)
    acc = float((pred == y[te]).mean()) if len(te) else float("nan")
    temp = calibrate(logits, y[te]) if len(te) > 20 else 1.0
    z = logits / temp
    z -= z.max(1, keepdims=True) if len(te) else 0
    conf = (np.exp(z) / np.exp(z).sum(1, keepdims=True)).max(1) if len(te) else np.zeros(0)
    sure = conf >= 0.9
    per = {
        c: float((pred[y[te] == i] == i).mean()) for i, c in enumerate(classes) if (y[te] == i).any()
    }
    print(
        f"  {name:12s} held-out {acc:.3f} (n={len(te)})  T={temp:.1f}  "
        f"≥0.9: {sure.mean():.0%} of rows at {float((pred[sure] == y[te][sure]).mean()) if sure.any() else float('nan'):.3f}  "
        + "  ".join(f"{c[:8]} {v:.2f}" for c, v in per.items())
    )
    return layers, acc, pred, temp


_ASKS = re.compile(
    r"query|qa_|status|balance|what_|how_|when|where|who_|which|info|reviews|rate|_time|date|weather|"
    r"traffic|directions|distance|calendar$|calendar_today|reminder$|todo_list$|shopping_list$|"
    r"spending|transactions|expiration|routing|limit$|payday|income|taxes|due|used|pto_balance|"
    r"meeting_schedule|flight_status|order_status|current_location|find_phone|what_song|"
    r"suggestion|recommendation|definition|spelling|translate|calculator|conversion|fun_fact|"
    r"joke|greeting|thank_you|repeat|oos|history|alerts?$|timezone|holiday|vaccines|visa|carry_on|"
    r"plug_type|mpg|gas$|tire_pressure|jump_start|recipe|cook|ingredients|calories|nutrition|"
    r"food_last|exchange_rate|credit_score|apr$|interest_rate|min_payment|international_fees|"
    r"insurance$|w2|next_holiday|how_busy|accept_reservations|confirm_reservation|"
    r"are_you|do_you|what_are|what_is|where_are|who_made|how_old|meaning_of_life|what_can"
)


def _reply_label(r: dict) -> int:
    """1 = wants an answer, 0 = just do it, -1 = unknown (not used for training)."""
    if r["intent"] == "chat":
        return 1
    if r["intent"] == "stop":
        return -1
    src = r.get("source", "")
    if src.startswith("reflex_hf"):
        name = r.get("hf_intent", "")
        if not name:
            return -1
        return 1 if _ASKS.search(name) else 0
    from neo.reflex.schema import wants_reply

    return int(wants_reply(r["text"]))


_SPLIT = "hash5-v1"
_FLOOR = 0.85  # with nothing comparable, a new head must at least clear this


def _held_out(text: str) -> bool:
    import hashlib

    return int(hashlib.md5(text.strip().lower().encode()).hexdigest(), 16) % 5 == 0


def _score_current_head(path, *, Xs_raw: np.ndarray, blocks: dict) -> dict:
    """Accuracy of the head on disk for each block it can be compared on, on the same held-out
    rows as the new one ({} when it was trained on a different split and would be flattered)."""
    try:
        h = json.loads(path.read_text())
    except Exception:  # noqa: BLE001 — no head yet
        return {}
    if h.get("version") != 2 or h.get("split") != _SPLIT or len(h["mu"]) != Xs_raw.shape[1]:
        return {}
    mu, sd = np.array(h["mu"], np.float32), np.array(h["sd"], np.float32)
    out = {}
    for name, (y, te, tool_info) in blocks.items():
        blk = h.get(name)
        if not blk or te is None or not len(te):
            continue
        layers = [(np.array(W, np.float32), np.array(b, np.float32)) for W, b in blk["layers"]] if "layers" in blk else [
            (np.array(blk["W"], np.float32), np.zeros(len(blk["W"]), np.float32))
        ]
        if tool_info is not None:  # the tool head: rows are a subset, and classes must match
            t_rows, classes = tool_info
            if list(blk.get("classes", [])) != list(classes):
                continue
            X = (Xs_raw[t_rows] - mu) / sd
        else:
            X = (Xs_raw - mu) / sd
        pred = forward(layers, X[te]).argmax(1)
        out[name] = float((pred == y[te]).mean())
    return out


def _dump(layers: list[tuple[np.ndarray, np.ndarray]]) -> list[list]:
    return [[W.tolist(), b.tolist()] for W, b in layers]


def main() -> None:
    train(_parser().parse_args())


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-extract features even if cached")
    ap.add_argument("--per-class", type=int, default=640)
    ap.add_argument("--teacher", type=int, default=3000, help="rows to label with Laya's zero-shot answers")
    ap.add_argument("--zs-below", type=float, default=0.6, help="intent confidence under which zero-shot is consulted")
    ap.add_argument("--linear", action="store_true", help="linear heads instead of the small MLP")
    ap.add_argument("--own-weight", type=float, default=3.0, help="loss weight of NEO's own rows vs Hugging Face rows")
    ap.add_argument("--force", action="store_true", help="save the new head even if it scores lower")
    return ap


def defaults() -> argparse.Namespace:
    return _parser().parse_args([])


def train(args: argparse.Namespace, *, agent=None, lock=None) -> dict:
    """Build the data, fill the feature cache, fit every head, save if it doesn't regress.

    `agent`/`lock`: reuse a Laya instance that is already loaded (the running core) instead of
    loading a second copy; every call into it holds `lock` (MPS is not thread-safe)."""
    from contextlib import nullcontext

    lock = lock or nullcontext()
    quiet = agent is not None  # inside the core: no progress bars in the log
    rows = build(args.per_class)
    texts = [r["text"] for r in rows]
    y_intent = np.array([_INTENTS.index(r["intent"]) for r in rows])
    print(f"{len(rows)} rows: " + ", ".join(f"{k} {int((y_intent == i).sum())}" for i, k in enumerate(_INTENTS)))

    # Features are cached *per text*, so editing the data only costs the new rows.
    cache = settings().data_dir / "reflex_features_v2.npz"
    emb_by: dict[str, np.ndarray] = {}
    teach_by: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # text → (destructive/needs_screen, zs)
    if cache.exists() and not args.refresh:
        z = np.load(cache, allow_pickle=True)
        emb_by = dict(zip(z["texts"].tolist(), z["emb"], strict=True))
        if "teacher_texts" in z.files:
            t_texts = z["teacher_texts"].tolist()
        else:  # first-format cache: teacher rows were indexed into `texts`
            t_texts = [z["texts"][i] for i in z["teacher_idx"]]
        teach_by = {t: (d, s) for t, d, s in zip(t_texts, z["teacher"], z["zs"], strict=True)}
        print(f"cache: {len(emb_by)} embeddings, {len(teach_by)} teacher rows", file=sys.stderr)
    # Teacher rows: those already labelled first, so new data doesn't reshuffle the sample and
    # force hundreds of fresh ~130 ms zero-shot calls; the rest topped up at random.
    rng = np.random.default_rng(0)
    per = max(1, args.teacher // len(_INTENTS))
    picked = []
    for i in range(len(_INTENTS)):
        idx = np.where(y_intent == i)[0]
        cached = [j for j in idx if texts[j] in teach_by][:per]
        rest = [j for j in rng.permutation(idx) if texts[j] not in teach_by][: per - len(cached)]
        picked.append(np.array(cached + rest, dtype=int))
    tidx = np.concatenate(picked)
    need_emb = [t for t in dict.fromkeys(texts) if t not in emb_by]
    need_teach = [texts[i] for i in tidx if texts[i] not in teach_by]
    if need_emb or need_teach:
        if agent is None:
            import laya

            t0 = time.time()
            agent = laya.load("convaiinnovations/laya", device=settings().laya_device)
            print(f"laya loaded in {time.time() - t0:.1f}s", file=sys.stderr)
        if need_emb:
            t0 = time.time()
            for i in range(0, len(need_emb), 64):
                chunk = need_emb[i : i + 64]
                with lock:
                    embs = embed_only(agent, chunk, progress=False)
                for t, e in zip(chunk, embs, strict=True):
                    emb_by[t] = e
            print(f"{len(need_emb)} embeddings in {time.time() - t0:.0f}s", file=sys.stderr)
        if need_teach:
            q = questions_for([])
            t0 = time.time()
            for n, t in enumerate(need_teach):
                with lock:
                    ans = agent.system_one({"text": t}, q)["answers"]
                p = ans["intent"].get("probabilities", {})
                teach_by[t] = (
                    np.array([float(ans["destructive"].get("noul", 0.0)), float(ans["needs_screen"].get("noul", 0.0))], np.float32),
                    np.array([float(p.get(k, 0.0)) for k in _INTENTS], np.float32),
                )
                if n % 100 == 0 and not quiet:
                    print(f"\r  teacher {n}/{len(need_teach)}", end="", file=sys.stderr)
            print(f"\n  {len(need_teach)} teacher labels in {time.time() - t0:.0f}s", file=sys.stderr)
        np.savez(
            cache,
            texts=np.array(list(emb_by), dtype=object),
            emb=np.vstack([emb_by[t] for t in emb_by]),
            teacher_texts=np.array(list(teach_by), dtype=object),
            teacher=np.vstack([teach_by[t][0] for t in teach_by]) if teach_by else np.zeros((0, 2), np.float32),
            zs=np.vstack([teach_by[t][1] for t in teach_by]) if teach_by else np.zeros((0, len(_INTENTS)), np.float32),
        )
    blocks = {
        "emb": np.vstack([emb_by[t] for t in texts]),
        "teacher_idx": tidx,
        "teacher": np.vstack([teach_by[texts[i]][0] for i in tidx]),
        "zs": np.vstack([teach_by[texts[i]][1] for i in tidx]),
    }

    X = np.hstack([blocks["emb"], np.ones((len(texts), 1), np.float32)]).astype(np.float32)
    mu, sd = standardise(X)
    Xs = (X - mu) / sd
    # A fixed held-out set: ~20% of sentences by a hash of their text, identical in every run,
    # so the current head and a new one can be compared fairly. The user's own examples are
    # always trained on — learning from them is the point.
    own = np.array([r.get("source") == "reflex_extra.jsonl" for r in rows])
    held = np.array([_held_out(t) for t in texts]) & ~own
    te, tr = np.where(held)[0], np.where(~held)[0]
    tr_set = set(tr.tolist())

    # ---- intent --------------------------------------------------------------------------
    # NEO's own rows (templates, generated, your misroutes) describe what NEO actually hears;
    # the Hugging Face rows add robustness. Weight the loss accordingly.
    weights = np.array([1.0 if r.get("source") == "reflex_hf.jsonl" else args.own_weight for r in rows], np.float32)
    mlp = not args.linear
    L_int, acc_int, pred_te, T_int = _fit(Xs, y_intent, len(_INTENTS), tr, te, "intent", _INTENTS, weights=weights, mlp=mlp)
    by_src: dict[str, list[bool]] = {}
    for k, j in enumerate(te):
        by_src.setdefault(rows[j].get("source", "?"), []).append(bool(pred_te[k] == y_intent[j]))
    print("    by source: " + "  ".join(f"{k} {np.mean(v):.3f} (n={len(v)})" for k, v in by_src.items()))
    tidx = blocks["teacher_idx"]
    te_t = [k for k, i in enumerate(tidx) if i not in tr_set]
    if te_t:
        zs_acc = float((blocks["zs"][te_t].argmax(1) == y_intent[tidx[te_t]]).mean())
        head_acc = float((forward(L_int, Xs[tidx[te_t]]).argmax(1) == y_intent[tidx[te_t]]).mean())
        print(f"  on the teacher subset: zero-shot {zs_acc:.3f} vs head {head_acc:.3f} (n={len(te_t)})")

    # ---- destructive / needs_screen (explicit template labels + Laya's own answers) ---------
    d_lab = np.full(len(rows), -1)
    s_lab = np.full(len(rows), -1)
    for i, r in enumerate(rows):
        if r.get("source") == "template":
            d_lab[i] = int(bool(r.get("destructive")))
            if r["intent"] in ("agent_task", "quick_action"):
                s_lab[i] = int(bool(_SCREEN_RE.search(r["text"])))
    for k, i in enumerate(tidx):
        if d_lab[i] < 0:
            d_lab[i] = int(blocks["teacher"][k, 0] >= 0.5)
        if s_lab[i] < 0:
            s_lab[i] = int(blocks["teacher"][k, 1] >= 0.5)
    heads = {}
    for name, lab in (("destructive", d_lab), ("needs_screen", s_lab)):
        have = np.where(lab >= 0)[0]
        h_tr = np.array([i for i in have if i in tr_set])
        h_te = np.array([i for i in have if i not in tr_set])
        Lb, _, _, Tb = _fit(Xs, lab.clip(0), 2, h_tr, h_te, name, ["no", "yes"], weights=weights, mlp=mlp)
        heads[name] = (Lb, Tb)

    # ---- reply: does the utterance want an answer? ------------------------------------------
    # chat → always; NEO's own rows → the regex fallback's verdict; Hugging Face rows → from the
    # source intent's name (query/qa/status… vs set/send/open…). The head generalises past both.
    r_lab = np.array([_reply_label(r) for r in rows])
    have = np.where(r_lab >= 0)[0]
    r_tr = np.array([i for i in have if i in tr_set])
    r_te = np.array([i for i in have if i not in tr_set])
    L_reply, acc_reply, _, T_reply = _fit(Xs, r_lab.clip(0), 2, r_tr, r_te, "reply", ["do it", "answer"], weights=weights, mlp=mlp)

    # ---- tool (quick actions, plus any row that names a tool) -------------------------------
    tool_rows = [i for i, r in enumerate(rows) if r["intent"] == "quick_action" or r.get("tool")]
    # A real "none" region: without chat/agent rows the head has never seen an utterance that
    # needs no single tool, and names one confidently for anything the intent head misroutes.
    rng_none = np.random.default_rng(1)
    others = [i for i, r in enumerate(rows) if r["intent"] in ("chat", "agent_task") and not r.get("tool")]
    tool_rows += rng_none.permutation(others)[: len(tool_rows) // 2].tolist()
    counts: dict[str, int] = {}
    for i in tool_rows:
        counts[rows[i].get("tool") or "none"] = counts.get(rows[i].get("tool") or "none", 0) + 1
    classes = sorted(c for c, n in counts.items() if n >= _MIN_TOOL_ROWS or c == "none")
    print(f"  tool classes: {', '.join(f'{c}({counts[c]})' for c in classes)}")
    t_rows = [i for i in tool_rows if (rows[i].get("tool") or "none") in classes]
    y_tool = np.array([classes.index(rows[i].get("tool") or "none") for i in t_rows])
    t_tr = np.array([k for k, i in enumerate(t_rows) if i in tr_set])
    t_te = np.array([k for k, i in enumerate(t_rows) if i not in tr_set])
    Xt = Xs[t_rows]
    L_tool, acc_tool, _, T_tool = _fit(Xt, y_tool, len(classes), t_tr, t_te, "tool", classes, weights=weights[t_rows], mlp=mlp)

    head_path = settings().data_dir / "reflex_head.json"
    # Judge the current head on *this* held-out split — scores from different data aren't
    # comparable. (It may have trained on some of these rows, which only flatters it.)
    cur = _score_current_head(head_path, Xs_raw=X, blocks={
        "intent": (y_intent, te, None),
        "reply": (r_lab.clip(0), r_te, None),
        "tool": (y_tool, t_te, (t_rows, classes)),
    })
    prev = cur.get("intent")
    report = {"rows": len(rows), "intent": acc_int, "tool": acc_tool, "reply": acc_reply, "previous": prev, "saved": False}
    new_scores = {"intent": acc_int, "reply": acc_reply, "tool": acc_tool}
    tolerance = {"intent": 0.005, "reply": 0.01, "tool": 0.02}
    worse = [k for k, v in cur.items() if v is not None and new_scores[k] < v - tolerance[k]]
    if cur:
        print("  current head on the same held-out rows: " + "  ".join(f"{k} {v:.3f}→{new_scores[k]:.3f}" for k, v in cur.items()))
    if worse and not args.force:
        print(f"\nkept the current head: the new one is worse at {', '.join(worse)}")
        return report
    if prev is None and acc_int < _FLOOR and not args.force:
        print(f"\nkept the current head: new intent {acc_int:.3f} is below the {_FLOOR} floor")
        return report
    if head_path.exists():
        head_path.with_suffix(".prev.json").write_text(head_path.read_text())  # one step of undo
    report["saved"] = True
    tmp = head_path.with_suffix(".tmp")  # written whole, then swapped in: never a torn head
    tmp.write_text(
        json.dumps(
            {
                "version": 2,
                "split": _SPLIT,
                "mu": mu.tolist(),
                "sd": sd.tolist(),
                "intent": {"layers": _dump(L_int), "classes": _INTENTS, "T": T_int},
                "destructive": {"layers": _dump(heads["destructive"][0]), "T": heads["destructive"][1]},
                "needs_screen": {"layers": _dump(heads["needs_screen"][0]), "T": heads["needs_screen"][1]},
                "tool": {"layers": _dump(L_tool), "classes": classes, "T": T_tool},
                "reply": {"layers": _dump(L_reply), "T": T_reply},
                "kind": "mlp" if mlp else "linear",
                "zs_below": args.zs_below,
                "held_out": {"intent": acc_int, "tool": acc_tool, "reply": acc_reply},
                "rows": len(rows),
            }
        )
    )
    tmp.replace(head_path)
    print(f"\nsaved v2 head → {head_path}  (intent {acc_int:.3f}, tool {acc_tool:.3f}; ~20 ms/decision)")
    pred = pred_te
    bad = [(texts[j], _INTENTS[y_intent[j]], _INTENTS[pred[k]]) for k, j in enumerate(te) if pred[k] != y_intent[j]]
    for t, gold, got in bad[: 0 if quiet else 15]:
        print(f"  ✗ {t!r}: {gold} → {got}")
    return report


if __name__ == "__main__":
    main()
