"""Train the reflex heads: small MLP ensembles over one Laya encoder embedding.

One encoder pass feeds six heads: intent, destructive, needs_screen, reply (wants an answer?),
tool (the one NEO tool a quick action needs) and kind (neo/decide's task kind). Each head is three
seeds averaged; its loss (cross-entropy, or spherical — which holds up under noisy labels) and its
temperature are chosen on rows it never trained on, and a new head replaces the current one only if
it passes the gate (`_gate`). Every run also fits a head without any of your own rows, which the
privacy erase falls back to (`reset_to_base`).

    python -m neo.reflex.finetune             # extract (or reuse cache), train, gate, save
    python -m neo.reflex.finetune --refresh   # ignore the feature cache
    python -m neo.reflex.finetune --teacher 3000   # public rows that may get zero-shot labels

Data: templates (dataset.py) + reflex_generated.jsonl (generate.py) + reflex_hf.jsonl (hf_data.py)
+ reflex_extra.jsonl (your own requests). A row's own kind/destructive/screen/reply labels win over
derived ones; split=="eval" rows are never trained on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from neo.config import settings
from neo.reflex.dataset import build
from neo.reflex.features import embed_only, forward, head_probs, load_encoder, parse_head, softmax
from neo.reflex.schema import INTENT_CRITERIA, QUESTIONS, refers_to_screen, sounds_destructive, wants_reply

_INTENTS = list(INTENT_CRITERIA)
_STOP = _INTENTS.index("stop")
_ACTIONS = ("quick_action", "agent_task")
OWN, HF, GENERATED = "reflex_extra.jsonl", "reflex_hf.jsonl", "reflex_generated.jsonl"
_SEEDS = (0, 1, 2)  # a 3-seed ensemble: +1.1 points, and seed-to-seed spread 2.0 -> 0.35 points
_LOSSES = ("ce", "spherical")
_MIN_CLASS_ROWS = 12
_MIN_SLICE = 50  # fewer rows than this can't choose a temperature or a loss
_SPLIT = "hash5-v1"
_VALIDATION = ("validation", "val", "dev")
_EVAL = ("eval", "test")
# Fields only rows from a labelled public source carry (hf_data.py).
_PUBLIC_LABELS = ("kind", "screen", "reply", "lang", "split", "group")
_TEACHER_Q = {k: QUESTIONS[k] for k in ("destructive", "needs_screen")}
# Laya's zero-shot answer teaches a head only if it agrees with NEO's own labels at least this well
# (balanced accuracy on rows that have one). Measured: destructive misses most sends on both
# checkpoints, and the multilingual needs_screen says yes to ~1 in 5 chat requests.
_TEACHER_MIN = 0.8


# ---- the nets -----------------------------------------------------------------------------------
def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    *,
    weights: np.ndarray | None = None,
    loss: str = "ce",
    hidden: int = 256,
    epochs: int = 40,
    lr: float = 2e-3,
    wd: float = 1e-3,
    dropout: float = 0.2,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """One hidden layer (ReLU) over the standardised embedding. Returns [(W1, b1), (W2, b2)].

    loss "spherical" is laya.proper_reward with only its spherical term: bounded, so a mislabelled
    row can't pull as hard (+5.9 points with 20% label noise, -0.8 on clean labels)."""
    import torch
    from laya.common import proper_reward

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
            z = net(Xt[b])
            if loss == "ce":
                per_row = torch.nn.functional.cross_entropy(z, yt[b], reduction="none")
            else:
                q = torch.softmax(z, -1)
                target = torch.nn.functional.one_hot(yt[b], k).float()
                qtype = torch.zeros(len(b), dtype=torch.long)  # a choice question
                per_row = -proper_reward(q, target, qtype, torch.ones_like(q), w_sph=1.0, log_floor=0.0)
            total = (per_row * wt[b]).sum() / wt[b].sum()
            opt.zero_grad()
            total.backward()
            opt.step()
    net.eval()
    l1, l2 = net[0], net[3]
    return [
        (l1.weight.detach().numpy().astype(np.float32), l1.bias.detach().numpy().astype(np.float32)),
        (l2.weight.detach().numpy().astype(np.float32), l2.bias.detach().numpy().astype(np.float32)),
    ]


def standardise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(0)
    sd = X.std(0) + 1e-6
    mu[-1], sd[-1] = 0.0, 1.0  # bias column stays 1
    return mu, sd


def pack(a: np.ndarray) -> dict:
    """A weight array for the head file: base64 float32. Six 3-seed heads are ~3.5M weights, ~78 MB
    as JSON numbers and ~19 MB like this (and read far faster)."""
    import base64

    a = np.ascontiguousarray(a, dtype="<f4")
    return {"f32": base64.b64encode(a.tobytes()).decode(), "shape": list(a.shape)}


def _ens(members: list, X: np.ndarray, T: float) -> np.ndarray:
    return np.mean([softmax(forward(m, X) / T) for m in members], axis=0)


def _nll(P: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(P[np.arange(len(y)), y] + 1e-9).mean()) if len(y) else float("nan")


def calibrate(member_logits: list[np.ndarray], y: np.ndarray) -> float:
    """Temperature that minimises held-out NLL of the members' mean softmax: a 40-epoch MLP says
    1.00 about everything, and the session puts thresholds on these probabilities. Never below
    0.5 (laya clamps its own there too): sharper only says the held-out rows were too easy."""
    best_t, best_nll = 1.0, float("inf")
    for t in np.geomspace(0.5, 20.0, 48):
        nll = _nll(np.mean([softmax(z / t) for z in member_logits], axis=0), y)
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return round(best_t, 4)


def ece(P: np.ndarray, y: np.ndarray) -> float:
    from laya.common import ece_score

    return ece_score(P.max(1), (P.argmax(1) == y).astype(float)) if len(y) else float("nan")


# ---- labels -------------------------------------------------------------------------------------
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
    if (given := _given(r, "reply")) >= 0:
        return given
    if r["intent"] == "chat":
        return 1
    if r["intent"] == "stop":
        return -1
    if r.get("source", "").startswith("reflex_hf"):
        name = r.get("hf_intent", "")
        if not name:
            return -1
        return 1 if _ASKS.search(name) else 0
    return int(wants_reply(r["text"]))


def _given(r: dict, key: str) -> int:
    """The row's own 0/1 label for `key`, or -1. dataset.build fills `destructive: False` into rows
    that never had one, so a False counts only on a row from a labelled public source."""
    v = r.get(key)
    if v is None:
        return -1
    if key == "destructive" and not v and (
        r.get("source") == OWN or not any(r.get(k) is not None for k in _PUBLIC_LABELS)
    ):
        return -1
    return int(bool(v))


# NEO tool -> the neo/decide task kind it carries out; a row with any other tool is "other".
_TOOL_KIND = {
    "play_music": "play_media", "open_app": "open_app", "reminder_add": "add_reminder",
    "calendar_add": "add_event", "mail_send": "send_email", "notes_create": "write_note",
    "web_search": "web_search", "search_site": "web_search",
}  # fmt: skip
# What a template sentence asks for, when its words say so. A template that matches two kinds (or a
# kind other than its tool's) gets no kind label: only unambiguous ones teach the head.
_KIND_WORDS = {
    k: re.compile(p, re.I)
    for k, p in {
        "play_media": r"\b(?:play|put on|listen to|chalao|bajao)\b",
        "send_message": r"\bsend (?:a |an )?(?:text|message|msg|dm)\b",
        "send_email": r"(?:^|\b(?:and|then|you|please|neo,?)\s+)email\s+(?!me\b)\w+|\breply to\b.*\bemail\b",
        "write_note": r"\b(?:make|create|write|jot)\b.*\bnotes?\b",
        "add_reminder": r"\bremind",
        "add_event": r"\b(?:schedule|book)\b.*\b(?:meeting|appointment|event)\b|\bto (?:my|the) calendar\b",
        "web_search": r"\b(?:search|look up|google)\b",
        "other": r"\b(?:delete|remove|rename|organi[sz]e|install|uninstall|convert|zip|move|clone|fix|back up|clear|sort)\b",
    }.items()
}


def _kind_label(r: dict, kinds: set[str]) -> str:
    """The row's task kind: its own label, else what its tool does (and, for a template, its
    words say), else "" (no label)."""
    if r.get("kind") in kinds:
        return r["kind"]
    found = set()
    if r.get("tool"):
        found.add(_TOOL_KIND.get(r["tool"], "other"))
    if r.get("source") == "template" and r["intent"] in _ACTIONS:
        found |= {k for k, rx in _KIND_WORDS.items() if rx.search(r["text"])}
    return found.pop() if len(found) == 1 and found <= kinds else ""


_DEVANAGARI = re.compile(r"[\u0900-\u097f]")
_ROMAN_HINDI = ("hinglish", "hi-latn", "hi_latn", "hi-en", "latin-undecided", "cs", "hi")


def script_of(r: dict) -> str:
    """The row's script: devanagari, latin-undecided (Laya can't tell the language: Hinglish looks
    like this), en, or another — from the row's own lang/script field when it has one."""
    if _DEVANAGARI.search(r["text"]):
        return "devanagari"
    lang = str(r.get("script") or r.get("lang") or "").lower()
    if lang in _ROMAN_HINDI:
        return "latin-undecided"
    if lang:
        return "en" if lang.startswith("en") else lang
    from laya.lang import analyse

    a = analyse(r["text"])
    if a["script"] == "latin":
        return "latin-undecided" if a["language_undecided"] else str(a["language"] or "latin")
    return str(a["script"])


def _weight(r: dict) -> float:
    """Loss weight: your verified requests 10, your other ones (a teacher's verdict) 3, NEO's
    templates and generated rows 3, public datasets 1. Weight 10 won on clean labels only."""
    src = r.get("source")
    if src == OWN:
        return 10.0 if r.get("verified") else 3.0
    return 3.0 if src in ("template", GENERATED) else 1.0


def labels(rows: list[dict]) -> dict[str, np.ndarray]:
    """Every head's label per row, without the zero-shot teacher: -1 / "" = none. `*_trusted`
    marks a label the row carries or NEO derives (never the teacher's)."""
    from neo.decide.engine import TASKS

    kinds = set(TASKS)
    n = len(rows)
    lab = {
        "intent": np.array([_INTENTS.index(r["intent"]) for r in rows], dtype=int),
        "destructive": np.full(n, -1),
        "needs_screen": np.full(n, -1),
        "reply": np.array([_reply_label(r) for r in rows], dtype=int),
        "tool": np.array([(r.get("tool") or "none") if r["intent"] == "quick_action" or r.get("tool") else "" for r in rows], dtype=object),
        "kind": np.array([_kind_label(r, kinds) for r in rows], dtype=object),
        "script": np.array([script_of(r) for r in rows], dtype=object),
        "weight": np.array([_weight(r) for r in rows], dtype=np.float32),
    }
    for i, r in enumerate(rows):
        # NEO's own derivations where the row has no label. Destructive: a template's hint words, and
        # only an agent task changes anything (conversation, a stop or a one-step device command
        # never sends, deletes or pays — how the templates are labelled). Needs the screen: the words
        # say so. A head trained on templates alone was sure of both on unrelated public phrasings.
        d = _given(r, "destructive")
        if d < 0 and r.get("source") == "template":
            d = int(bool(r.get("destructive")))
        elif d < 0 and r["intent"] != "agent_task":
            d = 0
        s = _given(r, "screen")
        lab["destructive"][i] = d
        lab["needs_screen"][i] = s if s >= 0 else int(refers_to_screen(r["text"]))
    lab["destructive_trusted"] = lab["destructive"] >= 0
    lab["needs_screen_trusted"] = lab["needs_screen"] >= 0
    return lab


# ---- slices -------------------------------------------------------------------------------------
def _hash(salt: str, text: str) -> int:
    return int(hashlib.md5((salt + text.strip().lower()).encode()).hexdigest(), 16)


def _held_out(text: str) -> bool:
    return _hash("", text) % 5 == 0


def slices(rows: list[dict]) -> dict[str, np.ndarray]:
    """Which rows train and which judge, fixed by hashes of the text so every run agrees:

    * base: ~20% of the public/NEO rows (by `group` when a row has one, so a translation of a
      held-out sentence is held out too) — the general skill;
    * user: 20% of your verified requests (another hash) — did it learn *your* phrasing;
    * cal: 10% of your requests plus the public validation rows — fits temperatures, ECE;
    * train: the rest.

    `shadowed`: other rows with the same text as one of yours. Your verdict replaces theirs in the
    head (the same sentence mustn't train on one label and judge on another), but not in the base
    head, which never sees yours and so must not depend on them."""
    own = np.array([r.get("source") == OWN for r in rows], dtype=bool)
    own_texts = {r["text"].strip().lower() for r, o in zip(rows, own, strict=True) if o}
    shadowed = ~own & np.array([r["text"].strip().lower() in own_texts for r in rows], dtype=bool)
    verified = own & np.array([bool(r.get("verified")) for r in rows], dtype=bool)
    user = verified & np.array([_hash("user:", r["text"]) % 5 == 0 for r in rows], dtype=bool)
    cal = (own & ~user & np.array([_hash("cal:", r["text"]) % 10 == 0 for r in rows], dtype=bool)) | (
        ~own & np.array([r.get("split") in _VALIDATION for r in rows], dtype=bool)
    )
    base = ~own & ~cal & np.array([_held_out(str(r.get("group") or r["text"])) for r in rows], dtype=bool)
    return {"own": own, "user": user, "cal": cal, "base": base, "train": ~(user | cal | base), "shadowed": shadowed}


def training_rows(per_class: int) -> list[dict]:
    """The data minus eval rows, in an order fixed by a hash: the head fit without your rows can't
    depend on them even through the shuffle."""
    rows = [r for r in build(per_class) if r.get("split") not in _EVAL]
    rows.sort(key=lambda r: (_hash("order:", r["text"]), str(r.get("source"))))
    return rows


# ---- fitting ------------------------------------------------------------------------------------
def _fit(name, Xs, y, classes, w, masks, candidates) -> dict | None:
    """Train each candidate loss as a seed ensemble; keep the one with the lower NLL on the user
    slice (or the calibration rows), with its temperature fit on the calibration rows."""
    tr_m, cal_m, user_m, base_m = masks
    have = y >= 0
    tr = np.where(tr_m & have)[0]
    cal = np.where(cal_m & have)[0]
    if len(cal) < _MIN_SLICE:  # no calibration rows (no own rows, no public validation split)
        cal = np.where(base_m & have)[0]
    sel = np.where(user_m & have)[0]
    if len(sel) < _MIN_SLICE:
        sel = cal
    if len(tr) < _MIN_CLASS_ROWS or len(np.unique(y[tr])) < 2:
        return None
    k = len(classes)
    tried = {}
    for loss in candidates:
        members = [train_mlp(Xs[tr], y[tr], k, weights=w[tr], loss=loss, seed=s) for s in _SEEDS]
        T = calibrate([forward(m, Xs[cal]) for m in members], y[cal]) if len(cal) >= 20 else 1.0
        tried[loss] = (members, T, _nll(_ens(members, Xs[sel], T), y[sel]))
    loss = min(tried, key=lambda c: np.nan_to_num(tried[c][2], nan=np.inf))
    members, T, _ = tried[loss]
    P = _ens(members, Xs[cal], T)
    acc = float((P.argmax(1) == y[cal]).mean()) if len(cal) else float("nan")
    err = ece(P, y[cal])
    print(
        f"  {name:12s} n={len(tr):5d}  loss {loss:9s} "
        + " ".join(f"{c} {v[2]:.3f}" for c, v in tried.items())
        + f"  T={T:.2f}  held-out {acc:.3f} (n={len(cal)})  ECE {err:.3f}"
    )
    return {"members": [[[pack(W), pack(b)] for W, b in m] for m in members], "classes": classes, "T": T, "loss": loss, "ece": err}


def _fit_heads(rows, X, lab, sl, keep, losses: dict[str, str] | None = None) -> dict:
    """All six heads on the rows in `keep`; `losses` pins each head's loss (else both are tried)."""
    from neo.decide.engine import TASKS

    mu, sd = standardise(X[keep])
    Xs = (X - mu) / sd
    masks = tuple(sl[k] & keep for k in ("train", "cal", "user", "base"))
    w = lab["weight"]

    def cands(name):
        return (losses[name],) if losses and name in losses else _LOSSES

    out = {"mu": mu.tolist(), "sd": sd.tolist()}
    out["intent"] = _fit("intent", Xs, lab["intent"], _INTENTS, w, masks, cands("intent"))
    for name, classes in (("destructive", ["no", "yes"]), ("needs_screen", ["no", "yes"]), ("reply", ["do it", "answer"])):
        out[name] = _fit(name, Xs, lab[name], classes, w, masks, cands(name))

    # tool: quick actions (and any row naming a tool), plus a "none" region of chat/agent rows —
    # without it the head has never seen an utterance that needs no single tool, and names one
    # confidently for anything the intent head misroutes.
    tool_rows = keep & (lab["tool"] != "")
    no_tool = np.isin(lab["intent"], [_INTENTS.index("chat"), _INTENTS.index("agent_task")])
    others = np.where(keep & (lab["tool"] == "") & no_tool)[0]
    none_rows = sorted(others, key=lambda i: _hash("none:", rows[i]["text"]))[: int(tool_rows.sum()) // 2]
    names = lab["tool"].copy()
    names[none_rows] = "none"
    counts = {c: int(((names == c) & keep).sum()) for c in set(names[keep]) - {""}}
    classes = sorted(c for c, n in counts.items() if n >= _MIN_CLASS_ROWS or c == "none")
    y_tool = np.array([classes.index(t) if t in classes else -1 for t in names])
    out["tool"] = _fit("tool", Xs, y_tool, classes, w, masks, cands("tool"))

    k_counts = {c: int(((lab["kind"] == c) & keep).sum()) for c in TASKS}
    k_classes = [c for c in TASKS if k_counts[c] >= _MIN_CLASS_ROWS]
    y_kind = np.array([k_classes.index(c) if c in k_classes else -1 for c in lab["kind"]])
    out["kind"] = _fit("kind", Xs, y_kind, k_classes, w, masks, cands("kind"))
    return out


# ---- judging a head -----------------------------------------------------------------------------
def score(h: dict, X: np.ndarray, lab: dict, sl: dict, rules: dict) -> dict:
    """A parsed head's numbers on the rows no head trains on. destructive / needs_screen / stop
    count what NEO acts on — the head or the rule behind it."""
    held = (sl["base"] | sl["user"] | sl["cal"]) & ~sl["shadowed"]
    idx = np.where(held)[0]
    Xs = (X[idx] - h["mu"]) / h["sd"]
    y = lab["intent"][idx]
    P = head_probs(h["intent"], Xs)
    ok = P.argmax(1) == y

    def acc(mask):
        return float(ok[mask].mean()) if mask.any() else float("nan")

    m: dict = {"base": acc(sl["base"][idx]), "user": acc(sl["user"][idx]), "user_ok": ok[sl["user"][idx]]}
    cal = sl["cal"][idx] if sl["cal"][idx].sum() >= _MIN_SLICE else sl["base"][idx]
    m["ece"] = ece(P[cal], y[cal])
    for s in ("latin-undecided", "devanagari"):
        m[s] = acc(lab["script"][idx] == s)
    stop = y == _STOP
    m["stop"] = float(((P.argmax(1) == _STOP) | rules["stop"][idx])[stop].mean()) if stop.any() else float("nan")
    for name in ("destructive", "needs_screen"):
        pos = lab[f"{name}_trusted"][idx] & (lab[name][idx] == 1)
        if h[name] is None or not pos.any():
            m[name] = float("nan")
            continue
        p = head_probs(h[name], Xs[pos])[:, 1]
        m[name] = float((np.maximum(p, rules[name][idx][pos]) >= 0.5).mean())
        m[f"{name}_head"] = float((p >= 0.5).mean())
    have = lab["reply"][idx] >= 0
    m["reply"] = float((head_probs(h["reply"], Xs[have]).argmax(1) == lab["reply"][idx][have]).mean()) if h["reply"] is not None and have.any() else float("nan")
    for name in ("tool", "kind"):
        have = lab[name][idx] != ""
        if h[name] is None or not have.any():
            m[name] = float("nan")
            continue
        pred = np.array(h[name]["classes"], dtype=object)[head_probs(h[name], Xs[have]).argmax(1)]
        m[name] = float((pred == lab[name][idx][have]).mean())
    return m


def rule_signals(rows: list[dict]) -> dict[str, np.ndarray]:
    """What the rules behind the heads say for each row: a bare stop, destructive words, the screen."""
    from neo.reflex import is_bare_stop

    return {
        "stop": np.array([is_bare_stop(r["text"]) for r in rows], dtype=bool),
        "destructive": np.array([float(sounds_destructive(r["text"])) for r in rows], dtype=np.float32),
        "needs_screen": np.array([float(refers_to_screen(r["text"])) for r in rows], dtype=np.float32),
    }


def mcnemar_worse(cur_ok: np.ndarray, new_ok: np.ndarray) -> float:
    """One-sided exact McNemar p-value that the new head is worse on the same rows."""
    b = int((cur_ok & ~new_ok).sum())  # only the current head right
    c = int((~cur_ok & new_ok).sum())  # only the new one right
    n = b + c
    if b <= c:
        return 1.0
    return sum(math.comb(n, i) for i in range(b, n + 1)) / 2**n


_TOL = {"reply": 0.01, "tool": 0.02, "kind": 0.02}
# With no head to compare to (the first one, or a new encoder), a new head must clear these: well
# under what the public data gives either encoder (base 0.905-0.922, destructive with the rule
# 0.99-1.0, stop 0.83-0.89 — its held-out rows include loose generated "stop"s — ECE ~0.02).
_FLOORS = {"base": 0.85, "destructive": 0.95, "stop": 0.8, "ece": 0.05}


def _gate(new: dict, cur: dict | None, anchor: float) -> tuple[bool, list[str]]:
    """Accept only if: your requests aren't worse (paired McNemar p<0.1, and >= current - 0.5 pt);
    the general skill stays >= the first head's - 1 pt (a fixed anchor, so small losses can't pile
    up over retrains); what NEO acts on for destructive, stop and needs_screen misses nothing more;
    calibration ECE <= max(0.05, current + 0.01); reply/tool/kind within a small tolerance."""
    lines: list[str] = []
    ok = True

    def check(name: str, passed: bool, text: str) -> None:
        nonlocal ok
        ok &= passed
        lines.append(f"  {'ok  ' if passed else 'FAIL'} {name:13s} {text}")

    def num(v) -> bool:
        return v == v  # not NaN: the slice has rows

    if cur is None:
        for name, floor in _FLOORS.items():
            v = new[name]
            if num(v):
                passed = v <= floor if name == "ece" else v >= floor
                check(name, passed, f"{v:.3f} ({'<=' if name == 'ece' else '>='} floor {floor})")
        return ok, lines
    if len(new["user_ok"]):
        p = mcnemar_worse(cur["user_ok"], new["user_ok"])
        check("user slice", p >= 0.1 and new["user"] >= cur["user"] - 0.005,
              f"{cur['user']:.3f} -> {new['user']:.3f} (n={len(new['user_ok'])}, McNemar p={p:.2f})")  # fmt: skip
    check("base slice", new["base"] >= anchor - 0.01, f"{cur['base']:.3f} -> {new['base']:.3f} (>= anchor {anchor:.3f} - 0.01)")
    for name in ("destructive", "stop", "needs_screen"):
        if num(new[name]) and num(cur[name]):
            check(name, new[name] >= cur[name], f"recall {cur[name]:.3f} -> {new[name]:.3f}")
    if num(new["ece"]) and num(cur["ece"]):
        limit = max(0.05, cur["ece"] + 0.01)
        check("ECE", new["ece"] <= limit, f"{cur['ece']:.3f} -> {new['ece']:.3f} (<= {limit:.3f})")
    for name, tol in _TOL.items():
        if num(new[name]) and num(cur[name]):
            check(name, new[name] >= cur[name] - tol, f"{cur[name]:.3f} -> {new[name]:.3f}")
    return ok, lines


def _fingerprint(rows: list[dict], mask: np.ndarray) -> str:
    """Identifies the base slice's contents: the anchor only means something on the same rows."""
    keys = sorted(f"{r['text']}\t{r['intent']}" for r, m in zip(rows, mask, strict=True) if m)
    return hashlib.md5("\n".join(keys).encode()).hexdigest()[:16]


# ---- training -----------------------------------------------------------------------------------
def main() -> None:
    train(_parser().parse_args())


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-extract features even if cached")
    ap.add_argument("--per-class", type=int, default=640)
    ap.add_argument("--teacher", type=int, default=3000, help="public rows that may get Laya's zero-shot labels")
    ap.add_argument("--zs-below", type=float, default=0.6, help="intent confidence under which a decision is marked unsure")
    ap.add_argument("--force", action="store_true", help="save the new head even if it fails the gate")
    return ap


def defaults() -> argparse.Namespace:
    return _parser().parse_args([])


def train(args: argparse.Namespace, *, agent=None, lock=None, cancelled=None) -> dict:
    """Build the data, fill the feature cache, fit every head, save if it passes the gate.

    `cancelled()` true (the user erased NEO's memory while this ran): nothing is written — the
    features and the head were computed from sentences that must no longer exist.

    `agent`/`lock`: reuse a Laya instance that is already loaded (the running core) instead of
    loading a second copy; every call into it holds `lock` (MPS is not thread-safe)."""
    from neo import models

    lock = lock or nullcontext()
    quiet = agent is not None  # inside the core: no progress bars in the log
    s = settings()
    encoder = s.reflex_encoder
    rows = training_rows(args.per_class)
    texts = [r["text"] for r in rows]
    lab = labels(rows)
    sl = slices(rows)
    print(
        f"{len(rows)} rows ({encoder} encoder): "
        + ", ".join(f"{k} {int((lab['intent'] == i).sum())}" for i, k in enumerate(_INTENTS))
        + f"; own {int(sl['own'].sum())} (user slice {int(sl['user'].sum())}), calibration {int(sl['cal'].sum())}, base {int(sl['base'].sum())}"
    )

    got = _features(rows, lab, sl["own"], args, encoder, agent=agent, lock=lock, quiet=quiet, cancelled=cancelled)
    if got is None:
        return {"saved": False, "cancelled": True, "intent": 0.0}
    X, agree = got
    dim = X.shape[1] - 1
    t0 = time.time()
    fitted = _fit_heads(rows, X, lab, sl, ~sl["shadowed"])
    new_json = _head_json(fitted, args, encoder, dim, len(rows))
    rules = rule_signals(rows)
    new = score(parse_head(new_json), X, lab, sl, rules)

    head_path = s.data_dir / "reflex_head.json"
    cur = None
    try:
        cur_json = json.loads(head_path.read_text())
        cur_h = parse_head(cur_json, encoder, dim) if cur_json.get("split") == _SPLIT else None
    except (OSError, ValueError):
        cur_json, cur_h = {}, None
    if cur_h is not None:  # judged on *this* data's held-out rows: scores from other data don't compare
        cur = score(cur_h, X, lab, sl, rules)
    # The anchor is the first head's base score on these same rows (re-set when the public data
    # changes, since the score then means something else).
    fp = _fingerprint(rows, sl["base"])
    old = cur_json.get("anchor") or {}
    anchor = new["base"] if cur is None else old["base"] if old.get("data") == fp else cur["base"]
    new_json["anchor"] = {"base": anchor, "data": fp}
    new_json["teacher"] = agree
    new_json["held_out"] = {k: None if v != v else v for k, v in new.items() if k != "user_ok"}
    accepted, lines = _gate(new, cur, anchor)
    print(f"\ntrained in {time.time() - t0:.0f}s; gate " + ("against the current head" if cur else "(no comparable head: absolute floors)"))
    print("\n".join(lines))
    report = {
        "rows": len(rows), "encoder": encoder, "intent": new["base"], "tool": new["tool"], "reply": new["reply"],
        "kind": new["kind"], "ece": new["ece"], "previous": cur["base"] if cur else None, "saved": False, "gate": lines,
    }  # fmt: skip
    base_path = base_head_path()
    if not accepted and not args.force:
        print("\nkept the current head")
        if not base_path.exists() and not (cancelled and cancelled()):
            _write(base_path, _base_json(rows, X, lab, sl, fitted, new_json, args, encoder, dim))
        return report
    base_json = _base_json(rows, X, lab, sl, fitted, new_json, args, encoder, dim)
    if cancelled and cancelled():
        return {**report, "cancelled": True}
    if head_path.exists():
        head_path.with_suffix(".prev.json").write_text(head_path.read_text())  # one step of undo
    _write(head_path, new_json)
    _write(base_path, base_json)
    report["saved"] = True
    print(f"\nsaved head → {head_path}  (base {new['base']:.3f}, tool {new['tool']:.3f}, kind {new['kind']:.3f}, {models.LAYA_SUBFOLDERS[encoder] or 'english'} {dim}-d)")
    if not quiet:
        _show_mistakes(parse_head(new_json), X, lab, sl, texts)
    return report


def _features(rows, lab, own, args, encoder, *, agent, lock, quiet, cancelled):
    """(features, teacher agreement): embeddings from the cache or the encoder, and zero-shot
    labels where the teacher is good enough and nothing else labelled the row (filled into `lab`).
    None if the user erased NEO's memory meanwhile."""
    s = settings()
    texts = [r["text"] for r in rows]
    # Features are cached *per text* (and per encoder), so editing the data only costs new rows.
    emb_by: dict[str, np.ndarray] = {}
    teach_by: dict[str, np.ndarray] = {}  # text → zero-shot (destructive, needs_screen)
    if not args.refresh:
        emb_by, teach_by = load_cache(encoder)
        if emb_by:
            print(f"cache: {len(emb_by)} embeddings, {len(teach_by)} teacher rows", file=sys.stderr)
    cached_teach = len(teach_by)
    probe = _teacher_probe(rows, lab, own)
    need_emb = [t for t in dict.fromkeys(texts) if t not in emb_by]
    if need_emb or any(texts[i] not in teach_by for i in probe):
        if agent is None:
            t0 = time.time()
            agent = load_encoder(encoder, s.laya_device)
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
        _ask_teacher(agent, lock, [texts[i] for i in probe], teach_by, quiet)
    agree = _teacher_agreement(rows, lab, probe, teach_by)
    usable = [q for q, v in agree.items() if v >= _TEACHER_MIN]
    print("zero-shot teacher vs NEO's labels (balanced accuracy): " + ", ".join(f"{q} {v:.2f}" for q, v in agree.items())
          + f" → teaches {', '.join(usable) or 'nothing'}")  # fmt: skip
    tidx = _teacher_rows(rows, lab, own, teach_by, args.teacher, usable)
    if len(tidx) and any(texts[i] not in teach_by for i in tidx):
        agent = agent or load_encoder(encoder, s.laya_device)
        _ask_teacher(agent, lock, [texts[i] for i in tidx], teach_by, quiet)
    # Only what today's data needs: sentences dropped from the data (compaction, a forget) leave
    # the cache too, so it can't grow without limit or keep what the user asked NEO to forget.
    live = set(texts)
    stale = len(emb_by) + len(teach_by)
    emb_by = {t: e for t, e in emb_by.items() if t in live}
    teach_by = {t: v for t, v in teach_by.items() if t in live}
    if cancelled and cancelled():
        return None
    if stale != len(emb_by) + len(teach_by) or need_emb or len(teach_by) != cached_teach:
        save_cache(emb_by, teach_by, encoder)
    # The teacher only fills labels nothing else gave.
    for i in tidx:
        for j, name in enumerate(_TEACHER_Q):
            if name in usable and lab[name][i] < 0:
                lab[name][i] = int(teach_by[texts[i]][j] >= 0.5)
    X = np.hstack([np.vstack([emb_by[t] for t in texts]), np.ones((len(texts), 1), np.float32)])
    return X.astype(np.float32), agree


def _head_json(fitted: dict, args, encoder: str, dim: int, n_rows: int) -> dict:
    from neo import models

    return {
        "version": 2,
        "encoder": models.LAYA_REVISION,
        "subfolder": models.LAYA_SUBFOLDERS[encoder],
        "dim": dim,
        "split": _SPLIT,
        **fitted,
        "arch": "mlp-ensemble",
        "zs_below": args.zs_below,
        "ece": fitted["intent"]["ece"] if fitted["intent"] else None,
        "rows": n_rows,
    }


def _base_json(rows, X, lab, sl, fitted, new_json, args, encoder, dim) -> dict:
    """The head without any of your own rows (for the privacy erase), with the losses the full run
    chose. Without own rows it *is* the new head."""
    keep = ~sl["own"]
    if keep.all():
        return new_json
    print("\nbase head (none of your requests):")
    losses = {k: v["loss"] for k, v in fitted.items() if isinstance(v, dict) and "loss" in v}
    base = _head_json(_fit_heads(rows, X, lab, sl, keep, losses), args, encoder, dim, int(keep.sum()))
    # Its own anchor: after an erase, nothing measured on a head fit on your rows carries over.
    h = parse_head(base)
    idx = np.where(sl["base"])[0]  # never your rows
    pred = head_probs(h["intent"], (X[idx] - h["mu"]) / h["sd"]).argmax(1)
    base["anchor"] = {"base": float((pred == lab["intent"][idx]).mean()), "data": new_json["anchor"]["data"]}
    return base


def _write(path: Path, head: dict) -> None:
    tmp = path.with_suffix(".tmp")  # written whole, then swapped in: never a torn head
    tmp.write_text(json.dumps(head))
    tmp.replace(path)


def _show_mistakes(h: dict, X, lab, sl, texts, limit: int = 15) -> None:
    idx = np.where(sl["base"])[0]
    pred = head_probs(h["intent"], (X[idx] - h["mu"]) / h["sd"]).argmax(1)
    bad = [(texts[j], _INTENTS[lab["intent"][j]], _INTENTS[p]) for j, p in zip(idx, pred, strict=True) if p != lab["intent"][j]]
    for t, gold, got in bad[:limit]:
        print(f"  ✗ {t!r}: {gold} → {got}")


def _ask_teacher(agent, lock, texts: list[str], teach_by: dict[str, np.ndarray], quiet: bool) -> None:
    """Laya's zero-shot (destructive, needs_screen) for the texts not asked before (~25-60 ms each)."""
    todo = [t for t in dict.fromkeys(texts) if t not in teach_by]
    if not todo:
        return
    t0 = time.time()
    for n, t in enumerate(todo):
        with lock:
            ans = agent.system_one({"text": t}, _TEACHER_Q)["answers"]
        teach_by[t] = np.array([float(ans[k].get("noul", 0.0)) for k in _TEACHER_Q], np.float32)
        if n % 100 == 0 and not quiet:
            print(f"\r  teacher {n}/{len(todo)}", end="", file=sys.stderr)
    print(f"\n  {len(todo)} teacher labels in {time.time() - t0:.0f}s", file=sys.stderr)


def _teacher_probe(rows, lab, own, per: int = 60) -> np.ndarray:
    """Rows with NEO's own label for a teacher question, `per` of each answer (by a hash of the
    text): where the teacher is checked before it may label anything."""
    picked: set[int] = set()
    for name in _TEACHER_Q:
        for v in (0, 1):
            idx = np.where((lab[name] == v) & ~own)[0]
            picked |= set(sorted(idx, key=lambda j: _hash("probe:", rows[j]["text"]))[:per])
    return np.array(sorted(picked), dtype=int)


def _teacher_agreement(rows, lab, probe, teach_by) -> dict[str, float]:
    out = {}
    for j, name in enumerate(_TEACHER_Q):
        y = lab[name][probe]
        said = np.array([teach_by[rows[i]["text"]][j] >= 0.5 for i in probe], dtype=bool)
        pos, neg = y == 1, y == 0
        out[name] = float((said[pos].mean() + (~said[neg]).mean()) / 2) if pos.any() and neg.any() else 0.0
    return out


def _teacher_rows(rows, lab, own, cached, budget: int, questions: list[str]) -> np.ndarray:
    """Public rows missing a label the teacher may give, per intent class: those already answered
    first (new data mustn't force hundreds of fresh zero-shot calls), then by a hash of the text.
    Never your own rows: they are labelled by what actually happened."""
    if not questions:
        return np.zeros(0, dtype=int)
    need = np.any([lab[q] < 0 for q in questions], axis=0) & ~own
    per = max(1, budget // len(_INTENTS))
    picked: list[int] = []
    for i in range(len(_INTENTS)):
        idx = sorted(np.where(need & (lab["intent"] == i))[0], key=lambda j: _hash("teach:", rows[j]["text"]))
        done = [j for j in idx if rows[j]["text"] in cached][:per]
        picked += done + [j for j in idx if rows[j]["text"] not in cached][: per - len(done)]
    return np.array(sorted(picked), dtype=int)


# ---- the base head and the privacy erase ---------------------------------------------------------
def base_head_path() -> Path:
    return settings().data_dir / "reflex_head.base.json"


def reset_to_base() -> bool:
    """The privacy erase: the head trained without any of your requests replaces the current one,
    the one-step undo goes, and the running reflex switches over. With no base head (heads from
    before it existed) the head goes too: Laya answers zero-shot until it is retrained. Returns
    whether a base head was there."""
    from neo import reflex

    head = settings().data_dir / "reflex_head.json"
    base = base_head_path()
    head.with_suffix(".prev.json").unlink(missing_ok=True)
    had = base.exists()
    if had:
        tmp = head.with_suffix(".tmp")
        tmp.write_bytes(base.read_bytes())
        tmp.replace(head)
    else:
        head.unlink(missing_ok=True)
    if reflex._laya is not None:
        with reflex.laya_lock():
            reflex._laya.reload_head()
    return had


# ---- the per-text feature cache -------------------------------------------------------------------
def feature_cache() -> Path:
    return settings().data_dir / "reflex_features_v2.npz"


def _cache_key(encoder: str) -> str:
    from neo import models

    return f"{models.LAYA_REVISION}:{encoder}"


def _read_cache() -> tuple[str, dict[str, np.ndarray], dict[str, np.ndarray]]:
    cache = feature_cache()
    if not cache.exists():
        return "", {}, {}
    try:
        z = np.load(cache, allow_pickle=True)  # NEO's own file in ~/.neo (mode 700)
        # A cache from before the key was stored holds the English root's features.
        key = str(z["encoder"]) if "encoder" in z.files else _cache_key("english")
        emb_by = dict(zip([str(t) for t in z["texts"].tolist()], z["emb"], strict=True))
        if "teacher_texts" in z.files:
            t_texts = [str(t) for t in z["teacher_texts"].tolist()]
        else:  # first-format cache: teacher rows were indexed into `texts`
            t_texts = [str(z["texts"][i]) for i in z["teacher_idx"]]
        teach_by = {t: np.asarray(v, np.float32)[:2] for t, v in zip(t_texts, z["teacher"], strict=True)}
        return key, emb_by, teach_by
    except Exception as e:  # noqa: BLE001
        print(f"[reflex] feature cache unreadable ({str(e)[:80]}) — rebuilding it", file=sys.stderr)
        return "", {}, {}


def load_cache(encoder: str | None = None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """The cached embeddings and teacher labels; empty if there is none, it's unreadable (a cut-off
    write from an old version), or it holds another encoder's features: they are computed again."""
    key, emb_by, teach_by = _read_cache()
    if encoder is not None and key != _cache_key(encoder):
        return {}, {}
    return emb_by, teach_by


def save_cache(emb_by: dict[str, np.ndarray], teach_by: dict[str, np.ndarray], encoder: str | None = None) -> None:
    """Written to a temporary file and swapped in, so a crash mid-write never leaves a broken cache."""
    _write_cache(_cache_key(encoder or settings().reflex_encoder), emb_by, teach_by)


def _write_cache(key: str, emb_by: dict[str, np.ndarray], teach_by: dict[str, np.ndarray]) -> None:
    import os

    cache = feature_cache()
    tmp = cache.with_name(cache.name + ".tmp")
    dim = next(iter(emb_by.values())).shape[-1] if emb_by else 0
    with open(tmp, "wb") as f:
        np.savez(
            f,
            encoder=np.array(key),
            texts=np.array(list(emb_by), dtype=str),
            emb=np.vstack([emb_by[t] for t in emb_by]) if emb_by else np.zeros((0, dim), np.float32),
            teacher_texts=np.array(list(teach_by), dtype=str),
            teacher=np.vstack([teach_by[t] for t in teach_by]) if teach_by else np.zeros((0, 2), np.float32),
        )
    os.replace(tmp, cache)


def drop_from_cache(texts: set[str]) -> int:
    """Remove these sentences' features from the cache, whichever encoder's it is (for forget).
    Returns how many."""
    key, emb_by, teach_by = _read_cache()
    if not emb_by and not teach_by:
        return 0
    before = len(emb_by)
    emb_by = {t: e for t, e in emb_by.items() if t not in texts}
    teach_by = {t: v for t, v in teach_by.items() if t not in texts}
    _write_cache(key, emb_by, teach_by)
    return before - len(emb_by)


if __name__ == "__main__":
    main()
