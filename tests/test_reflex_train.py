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
    rows = []
    for word, (intent, tool) in _WORDS.items():
        for i in range(n_per_word):
            rows.append({"text": f"{word} thing {i}", "intent": intent, "tool": tool, "source": "reflex_hf.jsonl",
                         "hf_intent": "", "destructive": word == "email", "lang": "en"})  # fmt: skip
    for i in range(own):
        word = list(_WORDS)[i % len(_WORDS)]
        rows.append({"text": f"{word} mine {i}", "intent": _WORDS[word][0], "tool": _WORDS[word][1], "source": finetune.OWN, "verified": True})
    return rows


def _setup(monkeypatch, rows):
    monkeypatch.setattr(finetune, "build", lambda per_class: [dict(r) for r in rows])
    monkeypatch.setattr(finetune, "embed_only", lambda agent, texts, progress=False: np.vstack([_vec(t) for t in texts]))
    monkeypatch.setattr(finetune, "_SEEDS", (0, 1))
    monkeypatch.setattr(finetune, "_MIN_SLICE", 10)
    monkeypatch.setattr(finetune, "_FLOORS", {"base": 0.5, "destructive": 0.0, "stop": 0.5, "ece": 1.0})
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
    assert 0 <= h["ece"] < 1 and h["anchor"]["base"] == pytest.approx(report["intent"])
    base = json.loads(finetune.base_head_path().read_text())
    assert base["rows"] == len(_rows()) and base["mu"] != h["mu"]  # fit without the user's rows
    # the same data again: a comparable head, gated against it (your requests too), and accepted
    again = finetune.train(_args(), agent=_Agent())
    assert again["saved"] and again["previous"] is not None and any("user slice" in line for line in again["gate"])


def test_the_base_head_is_the_head_you_would_have_without_your_requests(monkeypatch):
    from neo.config import settings

    _setup(monkeypatch, _rows(own=60))
    finetune.train(_args(), agent=_Agent())
    base = json.loads(finetune.base_head_path().read_text())
    for p in settings().data_dir.glob("reflex_head*.json"):
        p.unlink()
    monkeypatch.setattr(finetune, "build", lambda per_class: [dict(r) for r in _rows()])
    finetune.train(_args(), agent=_Agent())
    never = json.loads(finetune.base_head_path().read_text())
    assert {k: never[k] for k in base} == base  # the same weights, bit for bit (plus that run's report)


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
    monkeypatch.setattr(finetune, "build", lambda per_class: [dict(r) for r in rows])
    got = finetune.training_rows(640)
    assert sorted(r["text"] for r in got) == ["Open notes", "open notes", "what time is it"]
    sl = finetune.slices(got)
    shadowed = [r["text"] for r, c in zip(got, sl["shadowed"], strict=True) if c]
    assert shadowed == ["open notes"]  # out of the head, still in the base head (which never sees yours)
    cal = [r["text"] for r, c in zip(got, sl["cal"], strict=True) if c]
    assert cal == ["what time is it"] and not (sl["train"] & sl["cal"]).any()


def test_the_gate():
    base = {"base": 0.90, "user": 0.8, "user_ok": np.array([True] * 80 + [False] * 20), "destructive": 0.98,
            "stop": 0.99, "needs_screen": 1.0, "ece": 0.03, "reply": 0.9, "tool": 0.85, "kind": 0.88}  # fmt: skip
    ok, _ = finetune._gate(dict(base), dict(base), anchor=0.90)
    assert ok
    worse_user = {**base, "user_ok": np.array([True] * 60 + [False] * 40), "user": 0.6}
    ok, lines = finetune._gate(worse_user, base, anchor=0.90)
    assert not ok and any("FAIL user slice" in line for line in lines)
    ok, _ = finetune._gate({**base, "base": 0.892}, base, anchor=0.90)  # within 1 pt of the anchor
    assert ok
    ok, _ = finetune._gate({**base, "base": 0.892}, base, anchor=0.905)
    assert not ok  # the anchor, not the current head, sets the bar: no ratchet
    ok, _ = finetune._gate({**base, "destructive": 0.97}, base, anchor=0.90)
    assert not ok
    ok, _ = finetune._gate({**base, "ece": 0.049}, base, anchor=0.90)
    assert ok  # under 0.05 always passes
    ok, lines = finetune._gate(dict(base), None, anchor=0.90)  # nothing comparable: absolute floors
    assert ok and any("floor" in line for line in lines)
    assert finetune.mcnemar_worse(np.array([True] * 10), np.array([True] * 10)) == 1.0
    assert finetune.mcnemar_worse(np.array([True] * 10), np.array([False] * 10)) < 0.01


def test_the_feature_cache_belongs_to_one_encoder():
    finetune.save_cache({"hello there": np.ones(4, np.float32)}, {"hello there": np.zeros(2, np.float32)}, "english")
    assert finetune.load_cache("multilingual") == ({}, {})
    assert list(finetune.load_cache("english")[0]) == ["hello there"]
    assert finetune.drop_from_cache({"hello there"}) == 1
    assert finetune._read_cache()[0] == finetune._cache_key("english")  # the erase keeps its key


def test_reset_to_base_swaps_in_the_head_without_your_rows(monkeypatch):
    import neo.reflex as reflex_mod
    from neo.config import settings

    reloaded = []

    class Laya:
        def reload_head(self):
            reloaded.append(True)

    head = settings().data_dir / "reflex_head.json"
    head.write_text('{"mine": 1}')
    head.with_suffix(".prev.json").write_text('{"mine": 0}')
    monkeypatch.setattr(reflex_mod, "_laya", Laya())
    assert not finetune.reset_to_base()  # no base head: the head fit on your rows still goes
    assert not head.exists() and not head.with_suffix(".prev.json").exists() and reloaded == [True]
    finetune.base_head_path().write_text('{"public": 1}')
    head.write_text('{"mine": 1}')
    assert finetune.reset_to_base() and json.loads(head.read_text()) == {"public": 1} and reloaded == [True, True]


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
              "khareed lo", "uninstall zoom", "email Sam the notes", "pay the electricity bill"):  # fmt: skip
        assert sounds_destructive(t), t
    for t in ("check my email", "read the last message from Sam", "what format is this", "what time is it"):
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


def test_laya_dir_fetches_only_the_chosen_checkpoint(monkeypatch):
    import huggingface_hub

    from neo import models

    seen = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, revision, allow_patterns: seen.append(allow_patterns) or "/snap")
    assert models.laya_dir("multilingual") == "/snap/multilingual"
    assert all(p.startswith("multilingual/") for p in seen[0]) and "multilingual/model.safetensors" in seen[0]
    assert models.laya_dir() == "/snap" and "model.safetensors" in seen[1]


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
        json.dumps({"text": "weather thing 999", "intent": "quick_action", "tool": "weather", "lang": "en", "source": "x:y"}) + "\n"
        + json.dumps({"text": "stop thing 999", "intent": "stop", "lang": "hinglish", "source": "x:z"}) + "\n"
    )  # fmt: skip
    monkeypatch.setattr(evaluate, "load_encoder", lambda encoder, device: _Agent())
    monkeypatch.setattr(evaluate, "embed_only", lambda agent, texts: np.vstack([_vec(t) for t in texts]))
    monkeypatch.setattr(sys, "argv", ["evaluate", "--latency", "0"])
    capsys.readouterr()
    evaluate.main()
    out = capsys.readouterr().out
    assert "acc bench lang=hinglish" in out and "recall stop" in out and "tool acc" in out and "ECE intent" in out
    assert "neo_data/reflex_head.json (multilingual)" in out


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
    assert asked == [{"intent", "destructive"}] and not d.needs_screen and d.source == "laya"
    assert r.decide("what's on my screen").needs_screen  # the words still say so
    r.encoder = "english"
    assert r.decide("tell me about the moon").needs_screen and asked[-1] == {"intent", "destructive", "needs_screen"}
