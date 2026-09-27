"""Train the reflex heads: small MLP ensembles over one Laya encoder embedding.

One encoder pass feeds six heads: intent, destructive, needs_screen, reply (wants an answer?),
tool (the one NEO tool a quick action needs) and kind (neo2/core/decide's task kind). Each head is three
seeds averaged; its loss (cross-entropy, or spherical — which holds up under noisy labels) and its
temperature are chosen on rows it never trained on, and a new head replaces the current one only if
it passes the gate (`_gate`). Every run also fits a head without any of your own rows (nor their
labels on anyone else's rows), which the privacy erase falls back to (`reset_to_base`).

    python -m neo2.core.reflex.finetune             # extract (or reuse cache), train, gate, save
    python -m neo2.core.reflex.finetune --refresh   # ignore the feature cache
    python -m neo2.core.reflex.finetune --teacher 3000   # public rows that may get zero-shot labels

Data: templates (dataset.py) + reflex_generated.jsonl (generate.py) + reflex_hf.jsonl (hf_data.py)
+ reflex_extra.jsonl (your own requests). A row's own kind/destructive/screen/reply labels win over
derived ones; split=="eval" rows are never trained on (and evaluate.py drops from the benchmark any
sentence the training data has).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

from neo2.core.config import settings
from neo2.core.reflex.dataset import build
from neo2.core.reflex.features import (
    embed_only,
    forward,
    head_probs,
    load_encoder,
    load_head,
    parse_head,
    softmax,
)
from neo2.core.reflex.schema import (
    INTENT_CRITERIA,
    QUESTIONS,
    refers_to_screen,
    screen_label,
    sounds_destructive,
    wants_reply,
)

_INTENTS = list(INTENT_CRITERIA)
_STOP = _INTENTS.index("stop")
_ACTIONS = ("quick_action", "agent_task")
OWN, HF, GENERATED = "reflex_extra.jsonl", "reflex_hf.jsonl", "reflex_generated.jsonl"
_SEEDS = (0, 1, 2)  # a 3-seed ensemble: +1.1 points, and seed-to-seed spread 2.0 -> 0.35 points
_LOSSES = ("ce", "spherical")
_MIN_CLASS_ROWS = 12
_MIN_SLICE = 50  # fewer rows than this can't choose a temperature or a loss
_MIN_ANSWER = 5  # ...nor rows with fewer than two answers this common
# v2: your requests are held out too (the user and calibration slices), 1 in 20 public groups
# calibrates, and the base slice follows `group`. A head from the v1 split trained on every one of
# your rows, so it can't be judged on them: it counts as no comparable head.
_SPLIT = "hash5-v2"
_VALIDATION = ("validation", "val", "dev")
_EVAL = ("eval", "test")
_PUBLIC_CAL = 20  # 1 in this many public/NEO groups is the public validation (calibration) slice
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


# NEO tool -> the neo2/core/decide task kind it carries out; a row with any other tool is "other".
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
    from neo2.core.decide.engine import TASKS

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
        # For the heads whose labels your requests don't verify (destructive, needs_screen, reply:
        # derived from the words, or from a row with the same words), a request of yours weighs what
        # a public row does. At 10 they only shifted those heads' priors: a retrain on 500 requests
        # missed 23 more destructive rows (McNemar p<0.01) and was rightly rejected.
        "weight_derived": np.array([1.0 if r.get("source") == OWN else _weight(r) for r in rows], dtype=np.float32),
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
        lab["needs_screen"][i] = s if s >= 0 else int(screen_label(r["text"]))
    # Your request carries no destructive label (the ledger doesn't record one): a template or a
    # labelled public row with the same words gives it theirs ("delete this folder" stays destructive
    # after you said it, and "cancel my alarm" isn't taught as harmless at your rows' weight just
    # because it was a quick action).
    own = np.array([r.get("source") == OWN for r in rows], dtype=bool)
    if own.any():
        said: dict[str, int] = {}
        for r, o, d in zip(rows, own, lab["destructive"], strict=True):
            if not o and d >= 0 and (r.get("source") == "template" or _given(r, "destructive") >= 0):
                k = _norm(r["text"])
                said[k] = max(said.get(k, 0), int(d))
        for i in np.where(own)[0]:
            k = _norm(rows[i]["text"])
            if k in said and _given(rows[i], "destructive") < 0:
                lab["destructive"][i] = said[k]
    lab["destructive_trusted"] = lab["destructive"] >= 0
    lab["needs_screen_trusted"] = lab["needs_screen"] >= 0
    # The rows whose label is their own (a labelled public source, or a template labelled by hand):
    # not one NEO derived from words its rules also read. Only these judge destructive and
    # needs_screen: on a row the rule labelled, "head or rule" is a hit by construction.
    lab["destructive_labelled"] = np.array([_labelled(r, "destructive") for r in rows], dtype=bool)
    lab["needs_screen_labelled"] = np.array([_labelled(r, "screen") for r in rows], dtype=bool)
    return lab


def _labelled(r: dict, key: str) -> bool:
    return r.get("source") != OWN and any(r.get(k) is not None for k in _PUBLIC_LABELS) and _given(r, key) >= 0


# ---- slices -------------------------------------------------------------------------------------
def _hash(salt: str, text: str) -> int:
    return int(hashlib.md5((salt + text.strip().lower()).encode()).hexdigest(), 16)


def _held_out(text: str) -> bool:
    return _hash("", text) % 5 == 0


def slices(rows: list[dict]) -> dict[str, np.ndarray]:
    """Which rows train and which judge, fixed by hashes so every run agrees:

    * base: ~20% of the public/NEO rows (by `group` when a row has one, so a translation of a
      held-out sentence is held out too) — the general skill, and the fixed rows the anchors are
      measured on;
    * user: 20% of your verified requests (another hash) — did it learn *your* phrasing;
    * cal: 10% of your requests plus the public validation rows (a split named so, and 1 in 20
      public/NEO groups: the public sources fold their own validation splits into train) — fits
      temperatures, chooses losses, measures ECE;
    * train: the rest.

    None of the public rows' slices depends on your rows. `shadowed`: other rows with the same words
    as one of yours. Your verdict replaces theirs in the head (the same sentence mustn't train on one
    label and judge on another), but not in the base head, which never sees yours. `echo`: those of
    yours — a head from before you said it trained on the other copy, so they can't compare heads."""
    own = np.array([r.get("source") == OWN for r in rows], dtype=bool)
    own_texts = {_norm(r["text"]) for r, o in zip(rows, own, strict=True) if o}
    shadowed = ~own & np.array([_norm(r["text"]) in own_texts for r in rows], dtype=bool)
    other_texts = {_norm(r["text"]) for r, o in zip(rows, own, strict=True) if not o}
    echo = own & np.array([_norm(r["text"]) in other_texts for r in rows], dtype=bool)
    verified = own & np.array([bool(r.get("verified")) for r in rows], dtype=bool)
    user = verified & np.array([_hash("user:", r["text"]) % 5 == 0 for r in rows], dtype=bool)
    group = [str(r.get("group") or r["text"]) for r in rows]
    public_cal = np.array(
        [r.get("split") in _VALIDATION or _hash("cal:", g) % _PUBLIC_CAL == 0 for r, g in zip(rows, group, strict=True)],
        dtype=bool,
    )
    cal = (own & ~user & np.array([_hash("cal:", r["text"]) % 10 == 0 for r in rows], dtype=bool)) | (~own & public_cal)
    base = ~own & ~cal & np.array([_held_out(g) for g in group], dtype=bool)
    return {"own": own, "user": user, "cal": cal, "base": base, "train": ~(user | cal | base), "shadowed": shadowed, "echo": echo}


def _norm(text: str) -> str:
    """How dataset.py and learn.py match your sentences to others: case and spacing aside."""
    return " ".join(str(text).lower().split())


def training_rows(per_class: int, *, own: bool = True) -> list[dict]:
    """The data minus eval rows, in an order fixed by a hash: the head fit without your rows can't
    depend on them even through the shuffle. Built without dataset.py's own-row override (which
    relabels templates with your words and drops public rows with them): `shadowed` keeps those
    rows out of the head here, and the base head must not learn your labels through them.
    `own` False: none of your requests (learning is off), exactly the rows of a run without them."""
    got = build(per_class, override=False) if own else build(per_class, override=False, own=False)
    rows = [r for r in got if r.get("split") not in _EVAL and (own or r.get("source") != OWN)]
    # A total order (duplicates included): dataset.py's shuffle depends on how many rows you have.
    rows.sort(key=lambda r: (_hash("order:", r["text"]), str(r.get("source")), json.dumps(r, sort_keys=True, default=str)))
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
    sel = _chooser(y, np.where(user_m & have)[0], cal)
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
        + f"  T={T:.2f}  calibration rows {acc:.3f} (n={len(cal)}, ECE there {err:.3f})"
    )
    return {"members": [[[pack(W), pack(b)] for W, b in m] for m in members], "classes": classes, "T": T, "loss": loss, "ece": err}


def _chooser(y: np.ndarray, user: np.ndarray, cal: np.ndarray) -> np.ndarray:
    """The rows that choose a head's loss: your held-out requests, if they give at least two answers
    (_MIN_ANSWER rows each) among _MIN_SLICE — on rows that all say "no" (your requests carry no
    needs_screen label, the words decide), the loss that says "no" loudest wins, and that head then
    misses the screen requests (measured: recall 0.70 -> 0.48) — else the calibration rows."""
    if len(user) >= _MIN_SLICE and (np.bincount(y[user]) >= _MIN_ANSWER).sum() >= 2:
        return user
    return cal


def _fit_heads(rows, X, lab, sl, keep) -> dict:
    """All six heads on the rows in `keep`, each with the loss that does better on held-out rows in
    `keep` (nothing outside `keep` reaches the result: the base head must be exactly the head a run
    without your requests would fit)."""
    from neo2.core.decide.engine import TASKS

    mu, sd = standardise(X[keep])
    Xs = (X - mu) / sd
    masks = tuple(sl[k] & keep for k in ("train", "cal", "user", "base"))
    w, w_derived = lab["weight"], lab["weight_derived"]
    out = {"mu": mu.tolist(), "sd": sd.tolist()}
    out["intent"] = _fit("intent", Xs, lab["intent"], _INTENTS, w, masks, _LOSSES)
    for name, classes in (("destructive", ["no", "yes"]), ("needs_screen", ["no", "yes"]), ("reply", ["do it", "answer"])):
        out[name] = _fit(name, Xs, lab[name], classes, w_derived, masks, _LOSSES)

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
    out["tool"] = _fit("tool", Xs, y_tool, classes, w, masks, _LOSSES)

    k_counts = {c: int(((lab["kind"] == c) & keep).sum()) for c in TASKS}
    k_classes = [c for c in TASKS if k_counts[c] >= _MIN_CLASS_ROWS]
    y_kind = np.array([k_classes.index(c) if c in k_classes else -1 for c in lab["kind"]])
    out["kind"] = _fit("kind", Xs, y_kind, k_classes, w, masks, _LOSSES)
    return out


# ---- judging a head -----------------------------------------------------------------------------
# What the session cancels everything on besides the bare-stop rule (agent/session.py): a "stop" of
# at most this many words that the intent head is at least this sure of.
_STOP_WORDS, _STOP_SURE = 2, 0.9


def cancels(P: np.ndarray, rules: dict) -> np.ndarray:
    """Per row: would the session cancel everything? The bare-stop rule, or a short "stop" the
    intent head (probabilities `P`) is sure of — its top pick alone isn't enough."""
    return rules["stop"] | ((P.argmax(1) == _STOP) & rules["short"] & (P[:, _STOP] >= _STOP_SURE))


def score(h: dict, X: np.ndarray, lab: dict, sl: dict, rules: dict) -> dict:
    """A parsed head's numbers on the rows no head trains on (other sources' copies of your
    sentences aside). base, the per-intent recalls, reply, tool, kind and ECE: on the base slice,
    the fixed rows the anchors are measured on (none of which chose a temperature or a loss, so
    calibration is measured out of sample). What NEO acts on — destructive, stop, needs_screen — on
    the public held-out rows (fixed with the data too, so they anchor): how many positives it
    catches (the head or the rule behind it; for stop, what the session cancels on, over the stops
    it can cancel on at all) and how often it goes off by mistake (`<name> fa`: a short command
    cancelled, a row labelled "no" flagged; `fa chat`: on conversation), each with its per-row
    outcomes (`*_ok`) for a paired test. destructive and needs_screen count only rows whose label
    is their own (`*_labelled`): a label the rule wrote is a hit by construction."""
    held = (sl["base"] | sl["user"] | sl["cal"]) & ~sl["shadowed"]
    idx = np.where(held)[0]
    Xs = (X[idx] - h["mu"]) / h["sd"]
    y = lab["intent"][idx]
    P = head_probs(h["intent"], Xs)
    ok = P.argmax(1) == y
    # your slice: only your own phrasings (not an echo of a public row the current head trained on)
    base, user, public = sl["base"][idx], sl["user"][idx] & ~sl["echo"][idx], ~sl["own"][idx]
    rules = {k: v[idx] for k, v in rules.items()}

    def acc(mask):
        return float(ok[mask].mean()) if mask.any() else float("nan")

    m: dict = {"base": acc(base), "user": acc(user), "user_ok": ok[user], "n": {"base": int(base.sum()), "user": int(user.sum())}}
    judge = base if base.sum() >= _MIN_SLICE else np.ones(len(idx), dtype=bool)  # tiny data: every held-out row
    m["ece"], m["n"]["ece"] = ece(P[judge], y[judge]), int(judge.sum())
    for i, c in enumerate(_INTENTS):
        m[f"recall {c}"], m["n"][f"recall {c}"] = acc(base & (y == i)), int((base & (y == i)).sum())
    for s in _BY_SCRIPT:  # the base slice can hide a script a few hundred rows strong getting worse
        in_script = base & (lab["script"][idx] == s)
        m[s], m[f"{s}_ok"], m["n"][s] = acc(in_script), ok[in_script], int(in_script.sum())
    # The stops the session can act on, and the commands it could wrongly cancel on: a longer
    # sentence is never cancelled on the head's word.
    can = rules["stop"] | rules["short"]
    fired = cancels(P, rules)
    _rate(m, "stop", fired, public & can & (y == _STOP))
    _rate(m, "stop fa", fired, public & can & (y != _STOP))
    chat = y == _INTENTS.index("chat")
    for name in ("destructive", "needs_screen"):
        yes, own_label = lab[name][idx], lab[f"{name}_labelled"][idx]
        pos, neg, neg_chat = public & own_label & (yes == 1), public & own_label & (yes == 0), public & chat & (yes == 0)
        if h[name] is None:  # NaN: the head is gone
            for key, mask in ((name, pos), (f"{name} fa", neg), (f"{name} fa chat", neg_chat)):
                m[key], m["n"][key] = float("nan"), int(mask.sum())
            continue
        fired = np.maximum(head_probs(h[name], Xs)[:, 1], rules[name]) >= 0.5
        _rate(m, name, fired, pos)
        _rate(m, f"{name} fa", fired, neg)
        _rate(m, f"{name} fa chat", fired, neg_chat)
        if pos.any():
            m[f"{name}_head"] = float((head_probs(h[name], Xs[pos])[:, 1] >= 0.5).mean())
    have = base & (lab["reply"][idx] >= 0)
    m["reply"] = float((head_probs(h["reply"], Xs[have]).argmax(1) == lab["reply"][idx][have]).mean()) if h["reply"] is not None and have.any() else float("nan")
    for name in ("tool", "kind"):
        have = base & (lab[name][idx] != "")
        if h[name] is None or not have.any():
            m[name] = float("nan")
            continue
        pred = np.array(h[name]["classes"], dtype=object)[head_probs(h[name], Xs[have]).argmax(1)]
        m[name] = float((pred == lab[name][idx][have]).mean())
    return m


def _rate(m: dict, name: str, fired: np.ndarray, rows: np.ndarray) -> None:
    """Into `m`: how often it `fired` on `rows` (NaN without any), and per row whether it did what
    it should (`name_ok`: fired on a positive; stayed quiet where firing is a false alarm, " fa")."""
    m[name], m["n"][name] = float(fired[rows].mean()) if rows.any() else float("nan"), int(rows.sum())
    m[f"{name}_ok"] = ~fired[rows] if name.endswith((" fa", " fa chat")) else fired[rows]


def rule_signals(rows: list[dict]) -> dict[str, np.ndarray]:
    """What the rules behind the heads say for each row: a bare stop, destructive words, the screen
    (and whether it is short enough for the session to cancel on the head's "stop")."""
    from neo2.core.reflex import is_bare_stop

    return {
        "stop": np.array([is_bare_stop(r["text"]) for r in rows], dtype=bool),
        "short": np.array([len(r["text"].split()) <= _STOP_WORDS for r in rows], dtype=bool),
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


# Held to the first head's numbers on the base slice (a fixed anchor, see _gate), with this slack.
_ANCHORED = {"base": 0.01, "reply": 0.01, "tool": 0.02, "kind": 0.02}
# What NEO acts on: the positives each signal must catch, and where going off is a mistake — the
# session cancels every running job on a short "stop", sends conversation that needs the screen to
# the vision agent. A head that says yes more often catches more and must not look better for it.
_RECALLS = ("destructive", "stop", "needs_screen")
_FALSE_ALARMS = ("destructive fa", "destructive fa chat", "stop fa", "needs_screen fa", "needs_screen fa chat")
_SIGNALS = _RECALLS + _FALSE_ALARMS
# Intent accuracy on the base slice's Hinglish-looking and Devanagari rows, judged like the signals.
_BY_SCRIPT = ("latin-undecided", "devanagari")
_PAIRED = _BY_SCRIPT + _SIGNALS
_ANCHOR_KEYS = (*_ANCHORED, *_PAIRED)
# Against the current head, per retrain (your slice, and each signal on the same rows): at most
# max(_NOISE_ROWS, tolerance x rows) more rows wrong, and not significantly more (one-sided exact
# McNemar, p < _NOISE_P) — a row or two either way is luck, not a worse head. Against the first
# head's numbers (the anchor, fixed like the base slice's): at most max(tolerance, _NOISE_ROWS / n)
# worse, so losses too small to reject one at a time can't pile up over retrains.
_NOISE_ROWS, _NOISE_P = 2, 0.1
_TOL = {"user": 0.02, **dict.fromkeys(_BY_SCRIPT + _RECALLS, 0.02), **dict.fromkeys(_FALSE_ALARMS, 0.005)}
# Every head must clear these, each on at least _MIN_JUDGE rows. Measured on the public data (49.7k
# rows, multilingual encoder): base 0.90, each intent's recall 0.81-0.91, destructive with the rule
# 0.85 (the rule alone 0.41: the public sets label calls, bookings, orders and deletions), the stops
# the session cancels on 0.73-0.76 (the rule alone 0.55), the screen with the rule 0.45-0.55 on the
# rows a public source labelled so (the rule alone ~0.11), ECE 0.02. The floors sit well under
# those and above what the rules alone give: they catch a broken head, not a slightly worse one.
# ECE over _FLOORS["ece"] only warns, and only decides nothing when there is no head to keep: a head
# that isn't well calibrated still beats Laya's zero-shot (intent 0.37), and NEO must not stay
# without one.
_FLOORS = {"base": 0.85, "recall": 0.6, "destructive": 0.75, "stop": 0.6, "needs_screen": 0.35, "ece": 0.10}
# ...and go off by mistake no more than this. Measured: 2-3 of 209 short commands cancelled on,
# destructive on 4.1% of rows labelled not (the rule alone on 2.5%) and 0.8-0.9% of chat, the
# screen on 0.03% (none of 2047 chat rows). A head with its "yes" pushed up until it catches more
# (+50 on the screen head's logit) goes off on 4-6% of rows, 4% of chat.
_CEILINGS = {"stop fa": 0.03, "destructive fa": 0.10, "destructive fa chat": 0.05, "needs_screen fa": 0.02, "needs_screen fa chat": 0.01}
_MIN_JUDGE = 30


def _num(v) -> bool:
    return v is not None and v == v  # not NaN: measured


def _gate(new: dict, cur: dict | None, anchor: dict) -> tuple[bool, list[str]]:
    """Accept only if the head clears the floors and ceilings (_FLOORS, _CEILINGS: every time, not
    only for the first head) and, when there is a current head to compare to: your requests aren't
    worse, nor is Hinglish or Devanagari, nor any signal NEO acts on (each: not more than a row or
    two, or a couple of points, and not significantly — see _NOISE_ROWS), nor worse than the first
    head's numbers (`anchor`: fixed, so small losses can't pile up over retrains; base, reply, tool
    and kind by _ANCHORED, the rest by _TOL); no head is lost; ECE <= max(0.05, current + 0.01).
    `cur` None: nothing compares, so the floors and ceilings decide (ECE only warns)."""
    lines: list[str] = []
    ok = True

    def check(name: str, passed: bool, text: str, *, warn: bool = False) -> None:
        nonlocal ok
        ok &= passed or warn
        lines.append(f"  {'ok  ' if passed else 'WARN' if warn else 'FAIL'} {name:13s} {text}")

    n = new.get("n", {})
    bounds = [("base slice", "base", _FLOORS["base"], 1)] + [(f"recall {c}", f"recall {c}", _FLOORS["recall"], 1) for c in _INTENTS]
    bounds += [(k, k, _FLOORS[k], 1) for k in _RECALLS] + [(k, k, v, -1) for k, v in _CEILINGS.items()]
    for label, key, bound, sign in bounds:
        if not _num(new.get(key)) or n.get(key, _MIN_JUDGE) < _MIN_JUDGE:
            if cur is None:
                lines.append(f"  --   {label:13s} not judged (n={n.get(key, 0)} < {_MIN_JUDGE})")
            continue
        passed = new[key] >= bound if sign > 0 else new[key] <= bound
        check(label, passed, f"{new[key]:.3f} ({'>= floor' if sign > 0 else '<= ceiling'} {bound}, n={n.get(key, '?')})")
    if cur is None:
        if _num(new.get("ece")):
            e, floor = new["ece"], _FLOORS["ece"]
            check("ECE", e <= floor, f"{e:.3f} (<= floor {floor}, n={n.get('ece', '?')})"
                  + ("" if e <= floor else ": kept anyway, a head beats none"), warn=True)  # fmt: skip
        return ok, lines

    if len(new["user_ok"]) and len(new["user_ok"]) == len(cur["user_ok"]):
        passed, text = _paired(cur["user_ok"], new["user_ok"], _TOL["user"], "wrong")
        check("user slice", passed, f"{cur['user']:.3f} -> {new['user']:.3f} ({text})")
    for name in _PAIRED:
        if not _num(cur.get(name)):
            continue
        if not _num(new.get(name)):
            check(name, False, f"{cur[name]:.3f} -> none: the head is gone")
            continue
        was, now = cur.get(f"{name}_ok"), new.get(f"{name}_ok")
        alarm = name in _FALSE_ALARMS
        if was is not None and now is not None and len(was) == len(now):
            passed, text = _paired(was, now, _TOL[name], "false alarms" if alarm else "wrong" if name in _BY_SCRIPT else "missed")
            check(name, passed, f"{cur[name]:.3f} -> {new[name]:.3f} ({text})")
        if _num(anchor.get(name)):
            tol = max(_TOL[name], _NOISE_ROWS / max(1, n.get(name, len(now) if now is not None else 1)))
            a = anchor[name]
            passed = new[name] <= a + tol + 1e-9 if alarm else new[name] >= a - tol - 1e-9
            check(name, passed, f"{new[name]:.3f} ({'<=' if alarm else '>='} anchor {a:.3f} {'+' if alarm else '-'} {tol:.3f})")
    if _num(new.get("ece")) and _num(cur.get("ece")):
        limit = max(0.05, cur["ece"] + 0.01)
        check("ECE", new["ece"] <= limit, f"{cur['ece']:.3f} -> {new['ece']:.3f} (<= {limit:.3f})")
    for name, tol in _ANCHORED.items():
        if not _num(anchor.get(name)):
            continue
        passed = _num(new.get(name)) and new[name] >= anchor[name] - tol
        was = f"{cur[name]:.3f}" if _num(cur.get(name)) else "none"
        now = f"{new[name]:.3f}" if _num(new.get(name)) else "none (the head is gone)"
        check("base slice" if name == "base" else name, passed, f"{was} -> {now} (>= anchor {anchor[name]:.3f} - {tol})")
    return ok, lines


def _paired(was: np.ndarray, now: np.ndarray, tol: float, what: str) -> tuple[bool, str]:
    """The same rows under the current head (`was`) and the new one (`now`), True where each got
    it right: fine unless more than max(_NOISE_ROWS, tol x rows) more are wrong, or significantly
    more (one-sided exact McNemar p < _NOISE_P)."""
    lost = int(was.sum()) - int(now.sum())
    allowed = max(float(_NOISE_ROWS), tol * len(now))
    p = mcnemar_worse(was, now)
    return lost <= allowed and p >= _NOISE_P, f"n={len(now)}, {lost:+d} {what}, <= {allowed:.1f}; McNemar p={p:.2f}"


def anchor_rows(sl: dict[str, np.ndarray]) -> np.ndarray:
    """The rows the anchors are measured on: the base slice and the public calibration rows (what
    NEO acts on is judged on both), none of them yours."""
    return sl["base"] | (sl["cal"] & ~sl["own"])


def _fingerprint(rows: list[dict], mask: np.ndarray) -> str:
    """Identifies the rows the anchors are measured on (the base slice and the public calibration
    rows): the anchors only mean something on the same rows."""
    keys = sorted(f"{r['text']}\t{r['intent']}" for r, m in zip(rows, mask, strict=True) if m)
    return hashlib.md5("\n".join(keys).encode()).hexdigest()[:16]


def _plain(m: dict) -> dict:
    """A score's numbers for the head file: no per-row arrays, NaN as null."""
    return {k: v if _num(v) else None for k, v in m.items() if not k.endswith("_ok") and k != "n"}


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


def train(args: argparse.Namespace, *, agent=None, lock=None, cancelled=None, own: bool = True) -> dict:
    """Build the data, fill the feature cache, fit every head, save if it passes the gate.

    `own` False (learning is off): none of your requests, so the head is the one a run without
    them fits (and it is also the base head).

    `cancelled()` true (the user erased NEO's memory while this ran): nothing is written — the
    features and the head were computed from sentences that must no longer exist. It is checked
    right before every write, under the lock the erase's reset_to_base takes, and once more after.

    `agent`/`lock`: reuse a Laya instance that is already loaded (the running core) instead of
    loading a second copy; every call into it holds `lock` (MPS is not thread-safe). The heads are
    for that agent's encoder (the running reflex is on the English one while the multilingual one
    isn't downloaded)."""
    from neo2.core import models

    lock = lock or contextlib.nullcontext()
    quiet = agent is not None  # inside the core: no progress bars in the log
    s = settings()
    encoder = getattr(agent, "neo_encoder", None) or s.reflex_encoder
    rows = training_rows(args.per_class, own=own)
    texts = [r["text"] for r in rows]
    lab = labels(rows)
    sl = slices(rows)
    print(
        f"{len(rows)} rows ({encoder} encoder{'' if own else ', none of your requests: learning is off'}): "
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
    fp = _fingerprint(rows, anchor_rows(sl))

    head_path = s.data_dir / "reflex_head.json"
    try:
        cur_json = json.loads(head_path.read_text())
        cur_h = parse_head(cur_json, encoder, dim) if cur_json.get("split") == _SPLIT else None
    except (OSError, ValueError, AttributeError):
        cur_json, cur_h = {}, None
    # Judged on *this* data's held-out rows: scores from other data don't compare.
    cur = score(cur_h, X, lab, sl, rules) if cur_h is not None else None
    # The anchors are the first head's numbers on the base slice, re-set when the public data
    # changes (they then mean something else); a head gated against nothing is its own anchor.
    old = cur_json.get("anchor") or {}
    if cur is not None and old.get("data") == fp:
        anchor = {k: old[k] if _num(old.get(k)) else cur.get(k) for k in _ANCHOR_KEYS}
    else:
        anchor = {k: (cur or new).get(k) for k in _ANCHOR_KEYS}
    new_json["anchor"] = {**_plain(anchor), "data": fp}
    new_json["teacher"] = agree
    new_json["held_out"] = _plain(new)
    accepted, lines = _gate(new, cur, anchor)
    how = "against the current head" if cur else "(no comparable head): absolute floors"
    print(f"\ntrained in {time.time() - t0:.0f}s; gate {how}")
    print("\n".join(lines))
    report = {
        "rows": len(rows), "encoder": encoder, "intent": new["base"], "tool": new["tool"], "reply": new["reply"],
        "kind": new["kind"], "ece": new["ece"], "previous": cur["base"] if cur else None, "saved": False, "gate": lines,
    }  # fmt: skip
    if not accepted and not args.force:
        print("\nkept the current head")
        if load_head(base_head_path(), encoder, dim) is None:  # none for this encoder: an erase needs one
            base = _base_head(rows, X, lab, sl, rules, new, new_json, args, encoder, dim, fp)
            if base is not None and not _save(cancelled, None, base):
                return {**report, "cancelled": True}
        return report
    base = _base_head(rows, X, lab, sl, rules, new, new_json, args, encoder, dim, fp)
    if not _save(cancelled, new_json, base):
        return {**report, "cancelled": True}
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
    from neo2.core import models

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


def _base_head(rows, X, lab, sl, rules, new, new_json, args, encoder, dim, fp) -> dict | None:
    """The head without any of your own rows, for the privacy erase: fit on exactly what a run
    without your requests would fit on (their words on other rows keep those rows' labels), each loss
    chosen without them, and anchored on its own numbers — after an erase, nothing measured on a head
    fit on your rows carries over. Without own rows it *is* the new head (`new` its score). None if
    it doesn't clear the absolute floors: after an erase, no head (and a retrain) beats a bad one."""
    if sl["own"].any():
        print("\nbase head (none of your requests):")
        base = _head_json(_fit_heads(rows, X, lab, sl, ~sl["own"]), args, encoder, dim, int((~sl["own"]).sum()))
        # judged, and its numbers kept, on public rows only: nothing in its file comes from yours
        public = {**{k: v & ~sl["own"] for k, v in sl.items()}, "shadowed": np.zeros_like(sl["own"])}
        new = score(parse_head(base), X, lab, public, rules)
        base["anchor"] = {**_plain({k: new.get(k) for k in _ANCHOR_KEYS}), "data": fp}
        base["held_out"] = _plain(new)
    else:
        base = new_json
    passed, lines = _gate(new, None, {})
    if not passed:
        print("base head not kept (an erase would put it back):\n" + "\n".join(lines))
    return base if passed else None


# Head files are written, and put back by an erase (reset_to_base), one writer at a time.
_head_lock = threading.Lock()


def _save(cancelled, head: dict | None, base: dict | None) -> bool:
    """Write the head (the one it replaces kept as .prev: one step of undo) and the base head —
    unless the user erased NEO's memory meanwhile. That is checked under the lock reset_to_base
    takes, right before the writes, so a head fit on erased rows can't land after the erase put the
    base head back; and once more after, in case the erase started while this wrote. Returns
    whether it wrote (and nothing was erased)."""
    path = settings().data_dir / "reflex_head.json"
    with _head_lock:
        if cancelled and cancelled():
            return False
        if head is not None:
            if path.exists():
                _write(path.with_suffix(".prev.json"), path.read_bytes())
            _write(path, head)
        if base is not None:
            _write(base_head_path(), base)
    if cancelled and cancelled():
        reset_to_base()  # the erase may have run before these writes finished: undo them
        return False
    return True


def _write(path: Path, data: dict | bytes) -> None:
    """Written whole, then swapped in: never a torn head. The temporary file is this writer's own
    (a retrain from the command line may run beside the core's)."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_bytes(data if isinstance(data, bytes) else json.dumps(data).encode())
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


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
    the one-step undo goes (and any head file a cut-off write left behind), and the running reflex
    re-reads it before its next decision (one still loading too). With no base head (heads from
    before it existed) the head goes too: Laya answers zero-shot until it is retrained. Returns
    whether a base head was put back. Never raises: the rest of the erase must still run."""
    from neo2.core.reflex import laya_reflex

    head = settings().data_dir / "reflex_head.json"
    base = base_head_path()
    had = False
    try:
        with _head_lock:  # this process's writers hold it: none of theirs is still being written
            sweep_temporaries(heads_only=True)
            head.with_suffix(".prev.json").unlink(missing_ok=True)
            had = base.is_file()
            if had:
                _write(head, base.read_bytes())
            else:
                head.unlink(missing_ok=True)
    except Exception as e:  # noqa: BLE001
        print(f"[reflex] couldn't put back the head trained without your requests ({str(e)[:80]}): removing the head")
        had = False
        for p in (head, head.with_suffix(".prev.json")):
            with contextlib.suppress(OSError):
                p.unlink(missing_ok=True)
    laya_reflex.head_changed()
    return had


def sweep_temporaries(max_age_s: float = 0.0, *, heads_only: bool = False) -> int:
    """Remove the head files (and the feature cache's) a cut-off write left behind: quitting NEO
    mid-retrain skips the writer's clean-up, and no later write reuses a writer's own temporary
    name. They hold weights fit on your requests, or their words and embeddings. Only those older
    than `max_age_s` (0: every one — the erase), so a write still going on (a command-line retrain
    beside the core) is left alone. Returns how many went."""
    d = settings().data_dir
    found = list(d.glob("reflex_head*.tmp"))
    if not heads_only:
        found.append(feature_cache().with_name(feature_cache().name + ".tmp"))
    now, gone = time.time(), 0
    for p in found:
        try:
            if max_age_s and now - p.stat().st_mtime < max_age_s:
                continue
            p.unlink()
            gone += 1
        except OSError:
            continue
    return gone


# ---- the per-text feature cache -------------------------------------------------------------------
def feature_cache() -> Path:
    return settings().data_dir / "reflex_features_v2.npz"


def _cache_key(encoder: str) -> str:
    from neo2.core import models

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
    from neo2.core import log_to_stdout

    log_to_stdout()  # the providers' status lines, as this command always printed them
    main()
