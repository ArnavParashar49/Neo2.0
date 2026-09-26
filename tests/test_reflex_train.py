"""Reflex training and inference: per-row labels, slices, ensembles, the gate, the encoder check,
the base head behind the privacy erase, and the rule backstops. No Laya: a fake encoder."""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pytest

from neo.reflex import finetune
from neo.reflex import laya_reflex as lr

_DIM = 12
# word -> (intent, tool, kind-ish): the fake encoder puts each word's sentences in its own cluster
_WORDS = {
    "hello": ("chat", ""), "explain": ("chat", ""), "joke": ("chat", ""),
    "weather": ("quick_action", "weather"), "play": ("quick_action", "play_music"), "timer": ("quick_action", "timer"),
    "email": ("agent_task", "mail_send"), "organize": ("agent_task", ""), "research": ("agent_task", ""),
    "stop": ("stop", ""), "cancel": ("stop", ""), "quiet": ("stop", ""),
}  # fmt: skip


def _vec(text: str) -> np.ndarray:
    word = text.split()[0]
    rng = np.random.default_rng(int(hashlib.md5(text.encode()).hexdigest()[:8], 16))
    centre = np.zeros(_DIM, np.float32)
    centre[list(_WORDS).index(word) % _DIM] = 3.0
    v = centre + rng.normal(0, 0.3, _DIM).astype(np.float32)
    return v / np.linalg.norm(v)


class _Agent:
    """Stands in for Laya: embeddings from _vec, zero-shot says "not destructive" to everything."""

    def __init__(self):
        self.calls = 0

    def system_one(self, item, questions):
        self.calls += 1
        return {"answers": {k: {"noul": 0.0} for k in questions}}


def _rows(n_per_word: int = 40, own: int = 0) -> list[dict]:
    """Two words each: a "stop" that short is one the session cancels on when the head is sure."""
    rows = []
    for word, (intent, tool) in _WORDS.items():
        for i in range(n_per_word):
            rows.append({"text": f"{word} thing{i}", "intent": intent, "tool": tool, "source": "reflex_hf.jsonl",
                         "hf_intent": "", "destructive": word == "email", "lang": "en"})  # fmt: skip
    for i in range(own):
        word = list(_WORDS)[i % len(_WORDS)]
        rows.append({"text": f"{word} mine{i}", "intent": _WORDS[word][0], "tool": _WORDS[word][1], "source": finetune.OWN, "verified": True})
    return rows


def _setup(monkeypatch, rows):
    monkeypatch.setattr(finetune, "build", lambda per_class, override=True: [dict(r) for r in rows])
    monkeypatch.setattr(finetune, "embed_only", lambda agent, texts, progress=False: np.vstack([_vec(t) for t in texts]))
    monkeypatch.setattr(finetune, "_SEEDS", (0, 1))
    monkeypatch.setattr(finetune, "_MIN_SLICE", 10)
    monkeypatch.setattr(finetune, "_MIN_JUDGE", 5)
    # the toy data has 40 rows a word: the stop and destructive floors can't be the real ones
    monkeypatch.setattr(finetune, "_FLOORS", {**finetune._FLOORS, "destructive": 0.0, "stop": 0.5})
    import laya.lang

    monkeypatch.setattr(laya.lang, "analyse", lambda t: {"script": "latin", "language_undecided": False, "language": "en"})


def _args(**kw) -> argparse.Namespace:
    return argparse.Namespace(**{**vars(finetune.defaults()), "teacher": 60, **kw})


def test_training_saves_an_ensemble_head_for_this_encoder_and_a_base_head(monkeypatch):
    from neo.config import settings

    _setup(monkeypatch, _rows(own=60))
    report = finetune.train(_args(), agent=_Agent())
    assert report["saved"] and report["intent"] > 0.9
    h = json.loads((settings().data_dir / "reflex_head.json").read_text())
    assert (h["subfolder"], h["dim"]) == ("multilingual", _DIM) and len(h["mu"]) == _DIM + 1
    assert len(h["intent"]["members"]) == 2 and h["intent"]["loss"] in ("ce", "spherical")
    assert h["destructive"]["members"] and h["needs_screen"] is None  # every row said "no screen"
    assert h["kind"]["classes"] and set(h["kind"]["classes"]) <= {"play_media", "send_email", "other"}
    # nothing earlier compares: the absolute floors, and the head is its own anchor
    assert any("floor" in line for line in report["gate"]) and report["previous"] is None
    assert 0 <= h["ece"] < 1 and h["anchor"]["base"] == pytest.approx(report["intent"])
    assert set(h["anchor"]) == {"base", "reply", "tool", "kind", "data"} and h["split"] == "hash5-v2"
    base = json.loads(finetune.base_head_path().read_text())
    assert base["rows"] == len(_rows()) and base["mu"] != h["mu"]  # fit without the user's rows
    assert base["held_out"]["user"] is None  # nothing in its file is measured on your requests
    # the same data again: a comparable head, gated against it (your requests too), and accepted
    again = finetune.train(_args(), agent=_Agent())
    assert again["saved"] and again["previous"] is not None and any("user slice" in line for line in again["gate"])
    assert json.loads((settings().data_dir / "reflex_head.json").read_text())["anchor"] == h["anchor"]  # fixed


def test_the_base_head_is_the_head_you_would_have_without_your_requests(monkeypatch):
    from neo.config import settings

    _setup(monkeypatch, _rows(own=60))
    finetune.train(_args(), agent=_Agent())
    base = json.loads(finetune.base_head_path().read_text())
    for p in settings().data_dir.glob("reflex_head*.json"):
        p.unlink()
    monkeypatch.setattr(finetune, "build", lambda per_class, override=True: [dict(r) for r in _rows()])
    finetune.train(_args(), agent=_Agent())
    never = json.loads(finetune.base_head_path().read_text())
    # the same weights, losses, temperatures and anchors, bit for bit (plus that run's report)
    assert {k: never[k] for k in base} == base


def test_a_head_for_another_encoder_or_size_is_ignored(tmp_path):
    from neo import models

    def head(**kw):
        return {"version": 2, "mu": [0.0] * (_DIM + 1), "sd": [1.0] * (_DIM + 1),
                "intent": {"W": np.zeros((4, _DIM + 1)).tolist()}, "destructive": {"W": np.zeros((2, _DIM + 1)).tolist()},
                "needs_screen": {"W": np.zeros((2, _DIM + 1)).tolist()}, "encoder": models.LAYA_REVISION, **kw}  # fmt: skip

    p = tmp_path / "h.json"
    p.write_text(json.dumps(head()))  # an old file: no subfolder = the English root
    assert lr._load_head(p, "english", _DIM) is not None
    assert lr._load_head(p, "multilingual", _DIM) is None
    assert lr._load_head(p, "english", 1024) is None  # never a 13-d head on 1025-d features
    p.write_text(json.dumps(head(subfolder="multilingual")))
    assert lr._load_head(p, "multilingual", _DIM) is not None
    p.write_text(json.dumps(head(encoder="another-revision", subfolder="multilingual")))
    assert lr._load_head(p, "multilingual", _DIM) is None


def test_row_labels_win_and_the_teacher_only_fills_gaps(monkeypatch):
    rows = [
        {"text": "email the landlord now", "intent": "agent_task", "source": "reflex_hf.jsonl", "destructive": True, "screen": False, "kind": "send_email", "lang": "en"},
        {"text": "what's the capital of peru", "intent": "chat", "source": "reflex_hf.jsonl", "destructive": False, "lang": "en"},
        {"text": "set a timer please", "intent": "quick_action", "source": "reflex_hf.jsonl", "destructive": False, "tool": "timer"},
        {"text": "delete the report", "intent": "agent_task", "source": "template", "destructive": True},
        {"text": "email Sam the notes", "intent": "agent_task", "source": "template", "destructive": True},
        {"text": "send it to mom", "intent": "agent_task", "source": finetune.OWN, "destructive": False, "verified": True},
        {"text": "book me a cab to the airport", "intent": "agent_task", "source": "reflex_hf.jsonl", "destructive": False},
        {"text": "what does this error mean", "intent": "chat", "source": "reflex_hf.jsonl", "destructive": False},
    ]  # fmt: skip
    import laya.lang

    monkeypatch.setattr(laya.lang, "analyse", lambda t: {"script": "latin", "language_undecided": False, "language": "en"})
    lab = finetune.labels(rows)
    # a labelled public row's False is a label; an unlabelled agent task's (dataset.build's default)
    # isn't — but nothing but an agent task changes anything
    assert list(lab["destructive"]) == [1, 0, 0, 1, 1, -1, -1, 0]
    assert list(lab["needs_screen"]) == [0, 0, 0, 0, 0, 0, 0, 1]  # its own label, else its words
    assert lab["kind"][0] == "send_email" and lab["kind"][2] == "other" and lab["kind"][3] == "other" and lab["kind"][4] == "send_email"
    assert list(lab["weight"]) == [1.0, 1.0, 1.0, 3.0, 3.0, 10.0, 1.0, 1.0]
    assert list(lab["weight_derived"]) == [1.0, 1.0, 1.0, 3.0, 3.0, 1.0, 1.0, 1.0]  # yours verify intent, not these
    assert finetune.script_of({"text": "\u0917\u093e\u0928\u093e \u091a\u0932\u093e\u0913"}) == "devanagari"
    assert finetune.script_of({"text": "bas karo", "lang": "hinglish"}) == "latin-undecided"
    # the teacher never labels your rows, nor a row that already has the label it may give...
    own = np.array([r["source"] == finetune.OWN for r in rows])
    assert list(finetune._teacher_rows(rows, lab, own, {}, 100, ["destructive", "needs_screen"])) == [6]
    assert list(finetune._teacher_rows(rows, lab, own, {}, 100, [])) == []
    # ...and only teaches a question it answers like NEO's own labels do
    teach = {r["text"]: np.array([float(d == 1), 0.0], np.float32) for r, d in zip(rows, lab["destructive"], strict=True)}
    agree = finetune._teacher_agreement(rows, lab, finetune._teacher_probe(rows, lab, own), teach)
    assert agree == {"destructive": 1.0, "needs_screen": 0.5}  # it never says "needs the screen"


def test_eval_rows_never_train_and_your_verdict_replaces_other_sources(monkeypatch):
    rows = [
        {"text": "play jazz", "intent": "quick_action", "source": "reflex_hf.jsonl", "split": "eval"},
        {"text": "open notes", "intent": "quick_action", "source": "reflex_hf.jsonl"},
        {"text": "Open notes", "intent": "agent_task", "source": finetune.OWN},
        {"text": "what time is it", "intent": "quick_action", "source": "reflex_hf.jsonl", "split": "validation"},
    ]
    monkeypatch.setattr(finetune, "build", lambda per_class, override=True: [dict(r) for r in rows])
    monkeypatch.setattr(finetune, "_PUBLIC_CAL", 10**9)  # only the split named so calibrates here
    got = finetune.training_rows(640)
    assert sorted(r["text"] for r in got) == ["Open notes", "open notes", "what time is it"]
    sl = finetune.slices(got)
    shadowed = [r["text"] for r, c in zip(got, sl["shadowed"], strict=True) if c]
    assert shadowed == ["open notes"]  # out of the head, still in the base head (which never sees yours)
    cal = [r["text"] for r, c in zip(got, sl["cal"], strict=True) if c]
    assert cal == ["what time is it"] and not (sl["train"] & sl["cal"]).any()


def _m(**kw) -> dict:
    """A head's score: 100 of your requests, 100 positives for each thing NEO acts on."""
    m = {"base": 0.90, "user": 0.8, "user_ok": np.array([True] * 80 + [False] * 20), "ece": 0.03, "reply": 0.9, "tool": 0.85, "kind": 0.88}
    for c in ("chat", "quick_action", "agent_task", "stop"):
        m[f"recall {c}"] = 0.9
    for name in ("destructive", "stop", "needs_screen"):
        m[name], m[f"{name}_ok"] = 0.98, np.array([True] * 98 + [False] * 2)
    return {**m, **kw}


_ANCHOR = {"base": 0.90, "reply": 0.9, "tool": 0.85, "kind": 0.88}


def test_the_gate():
    ok, _ = finetune._gate(_m(), _m(), _ANCHOR)
    assert ok
    ok, lines = finetune._gate(_m(user_ok=np.array([True] * 60 + [False] * 40), user=0.6), _m(), _ANCHOR)
    assert not ok and any("FAIL user slice" in line for line in lines)
    ok, _ = finetune._gate(_m(base=0.892), _m(), _ANCHOR)  # within 1 pt of the anchor
    assert ok
    ok, _ = finetune._gate(_m(base=0.892), _m(), {**_ANCHOR, "base": 0.905})
    assert not ok  # the anchor, not the current head, sets the bar: no ratchet
    ok, _ = finetune._gate(_m(ece=0.049), _m(), _ANCHOR)
    assert ok  # under 0.05 always passes
    ok, lines = finetune._gate(_m(ece=0.07), _m(), _ANCHOR)
    assert not ok and any("FAIL ECE" in line for line in lines)  # with a head to keep, calibration counts
    assert finetune.mcnemar_worse(np.array([True] * 10), np.array([True] * 10)) == 1.0
    assert finetune.mcnemar_worse(np.array([True] * 10), np.array([False] * 10)) < 0.01


def test_with_nothing_to_compare_the_floors_decide_but_calibration_alone_never_rejects():
    ok, lines = finetune._gate(_m(), None, {})
    assert ok and any("floor" in line for line in lines) and not any("user slice" in line for line in lines)
    ok, lines = finetune._gate(_m(ece=0.3), None, {})  # badly calibrated: still better than no head
    assert ok and any("WARN ECE" in line and "floor 0.1" in line for line in lines)
    for bad in ({"base": 0.8}, {"recall agent_task": 0.4}, {"stop": 0.5}, {"destructive": 0.7}):
        ok, lines = finetune._gate(_m(**bad), None, {})
        assert not ok and any("FAIL" in line for line in lines), bad
    # a floor is only judged on enough rows
    ok, lines = finetune._gate(_m(stop=0.5, n={"stop": 12}), None, {})
    assert ok and any("stop" in line and "not judged" in line for line in lines)


def test_one_more_missed_row_is_noise_a_lost_head_is_not():
    two = np.array([True] * 96 + [False] * 4)  # 2 rows (2 pt) more missed: tolerated
    ok, _ = finetune._gate(_m(destructive=0.96, destructive_ok=two), _m(), _ANCHOR)
    assert ok  # a retrain better at everything else isn't rejected over a template row or two
    three = np.array([True] * 95 + [False] * 5)
    ok, lines = finetune._gate(_m(stop=0.95, stop_ok=three), _m(), _ANCHOR)
    assert not ok and any("FAIL stop" in line for line in lines)
    # few positives: one row is allowed, two aren't
    cur = _m(destructive=1.0, destructive_ok=np.array([True] * 20))
    ok, _ = finetune._gate(_m(destructive=0.95, destructive_ok=np.array([True] * 19 + [False])), cur, _ANCHOR)
    assert ok
    ok, _ = finetune._gate(_m(destructive=0.9, destructive_ok=np.array([True] * 18 + [False] * 2)), cur, _ANCHOR)
    assert not ok
    # within 2 pt on many rows, but only ever losing rows: McNemar says it is worse
    cur = _m(needs_screen=1.0, needs_screen_ok=np.array([True] * 400))
    ok, lines = finetune._gate(_m(needs_screen=0.98, needs_screen_ok=np.array([True] * 392 + [False] * 8)), cur, _ANCHOR)
    assert not ok and any("FAIL needs_screen" in line and "McNemar" in line for line in lines)
    lost = _m(destructive=float("nan"), needs_screen=float("nan"))  # score(): the head is gone
    del lost["destructive_ok"], lost["needs_screen_ok"]
    ok, lines = finetune._gate(lost, _m(), _ANCHOR)
    assert not ok and any("FAIL destructive" in line for line in lines) and any("FAIL needs_screen" in line for line in lines)


def test_reply_tool_and_kind_are_held_to_the_first_heads_numbers():
    # each retrain a little worse than the last, never 2 pt at once: the anchor stops the slide
    ok, lines = finetune._gate(_m(tool=0.825), _m(tool=0.84), _ANCHOR)
    assert not ok and any("FAIL tool" in line and "anchor 0.850" in line for line in lines)
    ok, _ = finetune._gate(_m(tool=0.84), _m(tool=0.87), _ANCHOR)  # 3 pt under the current, 1 under the anchor
    assert ok
    ok, lines = finetune._gate(_m(reply=0.885), _m(), _ANCHOR)
    assert not ok and any("FAIL reply" in line for line in lines)
    ok, lines = finetune._gate(_m(kind=float("nan")), _m(), _ANCHOR)  # the kind head is gone
    assert not ok and any("FAIL kind" in line for line in lines)


def test_stop_counts_what_the_session_cancels_on():
    # sure and short / short but unsure / sure but long / the bare-stop rule
    P = np.array([[0.0, 0.0, 0.05, 0.95], [0.2, 0.0, 0.0, 0.8], [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]])
    rules = {"stop": np.array([False, False, False, True]), "short": np.array([True, True, False, False])}
    assert list(finetune.cancels(P, rules)) == [True, False, False, True]
    assert list(finetune.rule_signals([{"text": "stop 3"}, {"text": "please stop it now"}])["short"]) == [True, False]


def test_the_first_head_with_your_requests_is_calibrated_on_public_validation_rows(monkeypatch):
    """With no head to compare, ECE is measured on the calibration slice: 10% of your requests and
    the public validation rows (1 in 20 public groups) — and even a bad ECE keeps the head."""
    _setup(monkeypatch, _rows(own=600))
    sl = finetune.slices(finetune.training_rows(640))
    assert (sl["cal"] & ~sl["own"]).sum() >= 10 and (sl["cal"] & sl["own"]).sum() >= 10
    assert not (sl["cal"] & (sl["base"] | sl["user"] | sl["train"])).any()
    monkeypatch.setattr(finetune, "ece", lambda P, y: 0.2)  # say calibration came out badly
    report = finetune.train(_args(), agent=_Agent())
    assert report["saved"] and any("WARN ECE" in line for line in report["gate"]), report["gate"]


def test_a_first_head_under_the_floors_is_not_kept(monkeypatch):
    from neo.config import settings

    _setup(monkeypatch, _rows(own=60))
    monkeypatch.setattr(finetune, "_FLOORS", {**finetune._FLOORS, "base": 1.01})  # nothing passes
    report = finetune.train(_args(), agent=_Agent())
    assert not report["saved"] and any("FAIL base slice" in line for line in report["gate"])
    assert not (settings().data_dir / "reflex_head.json").exists()
    assert not finetune.base_head_path().exists()  # the base head fails them too: an erase finds none


def test_a_head_from_the_old_split_is_not_judged_on_rows_it_trained_on(monkeypatch):
    from neo.config import settings

    _setup(monkeypatch, _rows(own=60))
    finetune.train(_args(), agent=_Agent())
    path = settings().data_dir / "reflex_head.json"
    old = json.loads(path.read_text())
    path.write_text(json.dumps({**old, "split": "hash5-v1"}))  # main's heads: trained on all your rows
    report = finetune.train(_args(), agent=_Agent())
    assert report["previous"] is None and any("floor" in line for line in report["gate"])


def test_the_feature_cache_belongs_to_one_encoder():
    finetune.save_cache({"hello there": np.ones(4, np.float32)}, {"hello there": np.zeros(2, np.float32)}, "english")
    assert finetune.load_cache("multilingual") == ({}, {})
    assert list(finetune.load_cache("english")[0]) == ["hello there"]
    assert finetune.drop_from_cache({"hello there"}) == 1
    assert finetune._read_cache()[0] == finetune._cache_key("english")  # the erase keeps its key


def test_a_rejected_head_still_leaves_the_erase_a_base_head_for_this_encoder(monkeypatch):
    _setup(monkeypatch, _rows(own=60))
    finetune.base_head_path().write_text('{"version": 2, "subfolder": ""}')  # the English checkpoint's
    monkeypatch.setattr(finetune, "_gate", lambda new, cur, anchor: (False, ["  FAIL everything"]) if anchor else (True, []))
    report = finetune.train(_args(), agent=_Agent())
    base = json.loads(finetune.base_head_path().read_text())
    assert not report["saved"] and base["subfolder"] == "multilingual" and base["rows"] == len(_rows())


def test_reset_to_base_swaps_in_the_head_without_your_rows():
    from neo.config import settings

    head = settings().data_dir / "reflex_head.json"
    head.write_text('{"mine": 1}')
    head.with_suffix(".prev.json").write_text('{"mine": 0}')
    epoch = lr._head_epoch
    assert not finetune.reset_to_base()  # no base head: the head fit on your rows still goes
    assert not head.exists() and not head.with_suffix(".prev.json").exists() and lr._head_epoch == epoch + 1
    finetune.base_head_path().write_text('{"public": 1}')
    head.write_text('{"mine": 1}')
    assert finetune.reset_to_base() and json.loads(head.read_text()) == {"public": 1}
    # an unreadable base head: no exception (the rest of the erase must run), no head fit on your rows
    finetune.base_head_path().unlink()
    finetune.base_head_path().mkdir()
    head.write_text('{"mine": 1}')
    assert not finetune.reset_to_base() and not head.exists()


def test_an_erase_racing_the_save_never_leaves_a_head_fit_on_your_rows(monkeypatch):
    import threading

    from neo.config import settings

    _setup(monkeypatch, _rows(own=60))
    finetune.train(_args(), agent=_Agent())  # a head and its base head
    erased, calls, eraser = [], [], []

    def erase():  # forget_everything: the retrain's cancelled() turns true, then the base head goes back
        erased.append(True)
        finetune.reset_to_base()

    def cancelled():
        calls.append(1)
        if len(calls) == 2:  # the check right before the writes: the erase lands just after it
            eraser.append(threading.Thread(target=erase))
            eraser[0].start()
            eraser[0].join(0.5)  # it can't finish: the writes hold the head lock
            return False
        return bool(erased)

    report = finetune.train(_args(), agent=_Agent(), cancelled=cancelled)
    eraser[0].join()
    head = json.loads((settings().data_dir / "reflex_head.json").read_text())
    assert report.get("cancelled") and not report["saved"]
    assert head["rows"] == len(_rows()) and not (settings().data_dir / "reflex_head.prev.json").exists()


def test_a_head_saved_before_the_erase_noticed_is_undone(monkeypatch):
    from neo.config import settings

    _setup(monkeypatch, _rows(own=60))
    finetune.train(_args(), agent=_Agent())
    state = {"n": 0}

    def cancelled():  # false up to and including the check before the writes, true after them
        state["n"] += 1
        return state["n"] > 2

    report = finetune.train(_args(), agent=_Agent(), cancelled=cancelled)
    head = json.loads((settings().data_dir / "reflex_head.json").read_text())
    assert report.get("cancelled") and head["rows"] == len(_rows())  # the base head, put back


def test_head_writers_never_share_a_temporary_file(monkeypatch):
    import threading
    from pathlib import Path

    from neo.config import settings

    temps = []
    real = Path.write_bytes

    def spy(self, data):
        temps.append(self.name)
        return real(self, data)

    finetune.base_head_path().write_bytes(b'{"public": 1}')
    monkeypatch.setattr(Path, "write_bytes", spy)
    head = settings().data_dir / "reflex_head.json"
    finetune._write(head, {"mine": 1})  # a retrain (this thread)...
    t = threading.Thread(target=finetune.reset_to_base)  # ...and the erase, on another
    t.start()
    t.join()
    assert len(temps) == 2 and temps[0] != temps[1] and head.name not in temps
    assert not list(settings().data_dir.glob("*.tmp"))


def test_a_reflex_still_loading_when_the_erase_ran_switches_to_the_base_head(monkeypatch):
    from neo import models
    from neo.config import settings

    def head(intent: int) -> dict:  # every embedding -> this intent
        W = np.zeros((4, 5))
        W[intent, 4] = 5.0
        return {"version": 2, "encoder": models.LAYA_REVISION, "subfolder": "multilingual", "mu": [0.0] * 5,
                "sd": [1.0] * 5, "intent": {"W": W.tolist()}}  # fmt: skip

    path = settings().data_dir / "reflex_head.json"
    path.write_text(json.dumps(head(2)))  # fit on your requests: agent_task
    finetune.base_head_path().write_text(json.dumps(head(0)))  # without them: chat
    r = lr.LayaReflex.__new__(lr.LayaReflex)
    r.encoder, r.dim, r._agent, r._head_path = "multilingual", 4, _Agent(), path
    r.reload_head()  # LayaReflex.__init__ has read the head; get_laya hasn't published it yet
    finetune.reset_to_base()  # the erase: reflex._laya is still None
    monkeypatch.setattr(lr.features, "embed", lambda agent, texts: np.ones((len(texts), 4), np.float32))
    assert r.decide("organize my files").intent == "chat"
    finetune.base_head_path().unlink()
    finetune.reset_to_base()  # nothing to go back to: no head, and the learn loop is told
    assert not r.has_head


def _reflex(head: dict, encoder: str = "multilingual") -> lr.LayaReflex:
    r = lr.LayaReflex.__new__(lr.LayaReflex)
    r.encoder, r.dim = encoder, len(head["mu"]) - 1
    r._agent = _Agent()
    r._head = finetune.parse_head(head)
    r.last_embedding, r.center = None, r._head["mu"][:-1]
    return r


def test_decide_averages_members_and_names_the_kind_of_an_action(monkeypatch):
    d = 4
    eye = np.eye(4, d + 1).tolist()
    member = [[eye, [0.0] * 4]]  # feature i -> intent i
    head = {
        "version": 2, "mu": [0.1] * d + [0.0], "sd": [1.0] * (d + 1),
        "intent": {"members": [member, [[(np.array(eye) * 3).tolist(), [0.0] * 4]]], "T": 0.2},
        "destructive": {"layers": [[np.zeros((2, d + 1)).tolist(), [0.0, 0.0]]]},
        "needs_screen": {"layers": [[np.zeros((2, d + 1)).tolist(), [0.0, 0.0]]]},
        "kind": {"members": [[[np.zeros((2, d + 1)).tolist(), [0.0, 2.0]]]], "classes": ["play_media", "other"], "T": 1.0},
    }  # fmt: skip
    r = _reflex(head)
    vec = {"hmm what": [1.0, 0, 0, 0], "organize my files": [0, 0, 1.0, 0]}
    monkeypatch.setattr(lr.features, "embed", lambda agent, texts: np.array([vec[t] for t in texts], np.float32))
    chat = r.decide("hmm what")
    assert chat.intent == "chat" and chat.kind == "" and chat.source == "laya+head"
    assert r.last_embedding.dtype == np.float32 and r.center.shape == r.last_embedding.shape
    act = r.decide("organize my files")
    assert act.intent == "agent_task" and act.kind == "other" and act.kind_p > 0.8
    assert r._agent.calls == 0  # no zero-shot call on the multilingual encoder


def test_rules_back_up_the_destructive_and_screen_heads(monkeypatch):
    from neo.reflex.schema import refers_to_screen, sounds_destructive

    for t in ("Priya ko email bhejo", "send it to mom", "delete the report", "file hata do", "ye jacket kharido",
              "khareed lo", "uninstall zoom", "email Sam the notes", "pay the electricity bill",
              # every Hinglish imperative ending (the old rule's \b stopped at "bhej de|na")
              "bhej do", "bhej dena", "Rahul ko paise bhejna", "ye app hata dena", "whatsapp pe msg bhej", "mita do",
              "mita dijiye", "delete kar do", "trash khaali kar do", "hata do", "kharid lo",
              # and Hindi: the head's destructive positives are English templates only
              "माँ को ईमेल भेजो", "भेज दो", "ये ऐप हटाओ", "मिटाओ", "फ़ाइल डिलीट करो", "जूते खरीदो", "जूते ख़रीद लो",
              "बिल पे करो", "ज़ूम इंस्टॉल करो", "पैसे ट्रांसफर करो",
              "make a transfer to John", "forward my email to Sam", "empty the trash", "wipe the drive",
              "overwrite the file"):  # fmt: skip
        assert sounds_destructive(t), t
    for t in ("check my email", "read the last message from Sam", "what format is this", "what time is it",
              # the nouns and look-alikes: shopping, a brain ("bheja"), sweets, a tree, "is it empty"
              "fast forward the video", "I look forward to it", "is the trash empty", "trash khaali hai kya",
              "kharidari karni hai", "bheja fry ho gaya", "my purchase history", "मिठाई खाओ", "खरीदारी करनी है",
              "मेरा भेजा खराब है", "मैंने भेजा था", "पेड़ लगाओ", "ईमेल पढ़ो", "मेरा ईमेल चेक करो"):  # fmt: skip
        assert not sounds_destructive(t), t
    assert refers_to_screen("what does this error mean") and refers_to_screen("what am I looking at")
    assert not refers_to_screen("play some jazz")

    d = 4
    no = [[np.zeros((2, d + 1)).tolist(), [3.0, 0.0]]]  # the heads say "no" to everything
    head = {"version": 2, "mu": [0.0] * (d + 1), "sd": [1.0] * (d + 1),
            "intent": {"layers": [[np.eye(4, d + 1).tolist(), [0.0] * 4]]}, "destructive": {"layers": no}, "needs_screen": {"layers": no}}  # fmt: skip
    r = _reflex(head)
    monkeypatch.setattr(lr.features, "embed", lambda agent, texts: np.array([[0, 0, 1.0, 0]], np.float32))
    a = r.decide("Priya ko email bhejo")
    assert a.destructive and a.destructive_p == 1.0 and not a.needs_screen
    b = r.decide("fix the error on this page")
    assert b.needs_screen and b.needs_screen_p == 1.0


def _snapshot(monkeypatch, root, checkpoints) -> list:
    """snapshot_download as it behaves offline: the revision's folder, with whatever was fetched."""
    import huggingface_hub

    for sub in checkpoints:
        for f in ("rl_agent_config.json", "model.safetensors", "encoder/config.json", "tokenizer/tokenizer.json"):
            (root / sub / f).parent.mkdir(parents=True, exist_ok=True)
            (root / sub / f).write_text("{}")
    seen = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, revision, allow_patterns: seen.append(allow_patterns) or str(root))
    return seen


def test_laya_dir_fetches_only_the_chosen_checkpoint(monkeypatch, tmp_path):
    from neo import models

    seen = _snapshot(monkeypatch, tmp_path, ("", "multilingual"))
    assert models.laya_dir("multilingual") == str(tmp_path / "multilingual")
    assert all(p.startswith("multilingual/") for p in seen[0]) and "multilingual/model.safetensors" in seen[0]
    assert models.laya_dir() == str(tmp_path) and "model.safetensors" in seen[1]


def test_a_checkpoint_that_never_came_down_is_a_clear_error_and_the_reflex_uses_the_english_one(monkeypatch, tmp_path):
    from neo import models
    from neo.config import settings

    _snapshot(monkeypatch, tmp_path, ("",))  # offline: only the English files were ever fetched
    with pytest.raises(FileNotFoundError, match="multilingual checkpoint isn't on this Mac"):
        models.laya_dir("multilingual")
    (tmp_path / "multilingual" / "tokenizer").mkdir(parents=True)  # an empty folder isn't a tokenizer
    for f in ("rl_agent_config.json", "model.safetensors", "encoder/config.json"):
        (tmp_path / "multilingual" / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "multilingual" / f).write_text("{}")
    with pytest.raises(FileNotFoundError, match="tokenizer"):
        models.laya_dir("multilingual")
    loaded = []

    def load(encoder, device):
        loaded.append(encoder)
        if encoder != "english":
            raise FileNotFoundError(f"Laya's {encoder} checkpoint isn't on this Mac yet")
        return "the English agent"

    monkeypatch.setattr(lr, "load_encoder", load)
    assert lr._load("multilingual", "cpu") == ("the English agent", "english") and loaded == ["multilingual", "english"]
    # a retrain inside the core, on that agent, trains heads for the English checkpoint
    _setup(monkeypatch, _rows())
    agent = _Agent()
    agent.neo_encoder = "english"
    assert finetune.train(_args(), agent=agent)["encoder"] == "english"
    assert json.loads((settings().data_dir / "reflex_head.json").read_text())["subfolder"] == ""


def test_the_encoder_setting_is_forgiving(monkeypatch):
    from neo.config import Settings

    assert Settings().reflex_encoder == "multilingual"
    monkeypatch.setenv("NEO_REFLEX_ENCODER", " English ")
    assert Settings().reflex_encoder == "english"
    monkeypatch.setenv("NEO_REFLEX_ENCODER", "hindi-ish")
    assert Settings().reflex_encoder == "multilingual"


def test_evaluate_reports_a_head_on_the_benchmark_and_held_out_rows(monkeypatch, capsys):
    import sys

    from neo.config import settings
    from neo.reflex import evaluate

    _setup(monkeypatch, _rows())
    finetune.train(_args(), agent=_Agent())
    bench = settings().data_dir / "reflex_eval.jsonl"
    bench.write_text(
        json.dumps({"text": "weather thing999", "intent": "quick_action", "tool": "weather", "lang": "en", "source": "x:y"}) + "\n"
        + json.dumps({"text": "stop thing999", "intent": "stop", "lang": "hinglish", "source": "x:z"}) + "\n"
        + json.dumps({"text": "play thing998", "intent": "quick_action"}) + "\n"  # no tool, kind, lang or source
        + json.dumps({"text": "Weather thing3!", "intent": "quick_action", "lang": "en", "source": "x:y"}) + "\n"  # trained on
        + "{cut off\n"
    )  # fmt: skip
    monkeypatch.setattr(evaluate, "load_encoder", lambda encoder, device: _Agent())
    monkeypatch.setattr(evaluate, "embed_only", lambda agent, texts: np.vstack([_vec(t) for t in texts]))
    monkeypatch.setattr(sys, "argv", ["evaluate", "--latency", "0"])
    capsys.readouterr()
    evaluate.main()
    out = capsys.readouterr().out
    assert "acc bench lang=hinglish" in out and "recall stop" in out and "tool acc" in out and "ECE intent" in out
    assert "neo_data/reflex_head.json (multilingual)" in out and "stop cancels (short)" in out
    assert "benchmark: 3 rows (1 left out: the training data has them too)" in out
    # no head there: a clear word, not a traceback
    monkeypatch.setattr(sys, "argv", ["evaluate", "--head", str(settings().data_dir / "nope.json")])
    with pytest.raises(SystemExit):
        evaluate.main()
    assert "no head there" in capsys.readouterr().out


def test_the_erase_puts_back_the_head_trained_without_your_requests():
    import asyncio

    from neo import privacy
    from neo.config import settings

    head = settings().data_dir / "reflex_head.json"
    head.write_text('{"fit on": "your requests"}')
    head.with_suffix(".prev.json").write_text('{"fit on": "your requests"}')
    finetune.base_head_path().write_text('{"fit on": "public data"}')
    asyncio.run(privacy.forget_everything())
    assert json.loads(head.read_text()) == {"fit on": "public data"} and not head.with_suffix(".prev.json").exists()


def test_without_a_head_the_multilingual_checkpoint_is_not_asked_about_the_screen():
    asked = []

    class Agent:
        def system_one(self, item, questions):
            asked.append(set(questions))
            said = {"intent": {"probabilities": {"chat": 0.9}}, "destructive": {"noul": 0.1}, "needs_screen": {"noul": 0.9}}
            return {"answers": {k: said[k] for k in questions}}

    r = lr.LayaReflex.__new__(lr.LayaReflex)
    r._agent, r._head, r.encoder = Agent(), None, "multilingual"
    d = r.decide("tell me about the moon")
    assert asked == [{"intent", "destructive"}] and not d.needs_screen and d.source == "laya" and d.margin == 1.0
    assert r.decide("what's on my screen").needs_screen  # the words still say so
    r.encoder = "english"
    assert r.decide("tell me about the moon").needs_screen and asked[-1] == {"intent", "destructive", "needs_screen"}


def test_your_request_keeps_the_destructive_label_of_a_template_with_its_words(monkeypatch):
    import laya.lang

    monkeypatch.setattr(laya.lang, "analyse", lambda t: {"script": "latin", "language_undecided": False, "language": "en"})
    rows = [
        {"text": "delete the report", "intent": "agent_task", "source": "template", "destructive": True},
        {"text": "Delete  the report", "intent": "agent_task", "source": finetune.OWN, "destructive": False, "verified": True},
        {"text": "organize the report", "intent": "agent_task", "source": finetune.OWN, "destructive": False, "verified": True},
        {"text": "cancel my alarm", "intent": "quick_action", "source": "reflex_hf.jsonl", "destructive": True, "lang": "en"},
        {"text": "cancel my alarm", "intent": "quick_action", "source": finetune.OWN, "destructive": False, "verified": True},
        {"text": "open notes", "intent": "quick_action", "source": "reflex_hf.jsonl"},  # no label: derived
        {"text": "open notes", "intent": "agent_task", "source": finetune.OWN, "destructive": False, "verified": True},
    ]
    assert list(finetune.labels(rows)["destructive"]) == [1, 1, -1, 1, 1, 0, -1]


def test_the_head_without_your_requests_never_sees_your_labels_through_the_dataset():
    """dataset.build relabels templates with your words and drops public rows with them; training
    builds without that, so the base head's rows and the anchors' fingerprint don't move."""
    import time

    from neo.config import settings

    def public(rows):
        sl = finetune.slices(rows)
        lab = finetune.labels(rows)
        keep = ~sl["own"]
        kept = sorted(
            (r["text"], r["intent"], r.get("tool", ""), r["source"], int(lab["destructive"][i]), str(lab["kind"][i]))
            for i, r in enumerate(rows) if keep[i]
        )  # fmt: skip
        return kept, finetune._fingerprint(rows, sl["base"]), sl

    rows = finetune.training_rows(40)
    before, fp, sl = public(rows)
    said = next(r for r, b in zip(rows, sl["base"], strict=True) if b and r["source"] == "template" and r["intent"] == "quick_action")
    row = {"text": said["text"], "intent": "agent_task", "tool": "", "source": "live", "ts": time.time(), "verified": True}
    (settings().data_dir / "reflex_extra.jsonl").write_text(json.dumps(row) + "\n")
    after, fp_after, sl = public(rows := finetune.training_rows(40))
    assert after == before and fp_after == fp
    shadowed = {r["text"] for r, s in zip(rows, sl["shadowed"], strict=True) if s}
    assert shadowed == {said["text"]}  # not in the head: your verdict is


def test_a_reflex_without_a_head_is_retrained_once(monkeypatch):
    import neo.reflex as reflex_mod
    from neo.reflex import finetune as ft
    from neo.reflex import learn

    class Laya:  # a new encoder, a first run, or an erase with no base head to go back to
        has_head = False
        encoder = "multilingual"
        _agent = None

        def reload_head(self):
            pass

    laya = Laya()
    monkeypatch.setattr(reflex_mod, "_laya", laya)
    monkeypatch.setattr(reflex_mod, "get_laya", lambda: laya)
    assert learn.due()  # nothing new to learn from, but zero-shot is far worse than any head
    tries = []

    def fails(*a, **kw):
        tries.append(1)
        raise RuntimeError("no luck")

    monkeypatch.setattr(ft, "train", fails)
    assert learn.retrain() is None and tries == [1]
    assert not learn.due()  # it failed: not again and again
    monkeypatch.setattr(ft, "train", lambda *a, **kw: {"saved": False, "intent": 0.9, "previous": None})
    Laya.encoder = "english"  # the user switched encoder: that one gets its own try
    assert learn.due() and learn.retrain() is not None and not learn.due()  # rejected: not retried either
    learn._write_state({"trained_at": 0.0})  # later states keep the record
    assert not learn.due()
    Laya.has_head = True
    Laya.encoder = "multilingual"
    assert not learn.due()
    learn.forget_examples()  # an erase starts over: one try again when it leaves no head
    Laya.has_head = False
    assert learn.due()


def test_the_erase_goes_on_when_the_base_head_cannot_be_put_back():
    import asyncio

    from neo import privacy
    from neo.config import settings

    s = settings()
    head = s.data_dir / "reflex_head.json"
    head.write_text('{"fit on": "your requests"}')
    finetune.base_head_path().mkdir()  # unreadable
    s.audit_log.write_text("an entry\n")
    asyncio.run(privacy.forget_everything())
    assert not head.exists() and not s.audit_log.exists()  # the rest of the erase ran


def test_your_requests_choose_a_heads_loss_only_when_they_give_more_than_one_answer():
    user, cal = np.arange(0, 60), np.arange(60, 120)
    y = np.zeros(120, dtype=int)
    assert list(finetune._chooser(y, user, cal)) == list(cal)  # all "no": the loudest "no" would win
    y[:10] = 1
    assert list(finetune._chooser(y, user, cal)) == list(user)
    assert list(finetune._chooser(y, user[:40], cal)) == list(cal)  # too few rows
