"""The reflex learns UI commands as the resolver's lane needs them: clicks by name in English, Hinglish
and Hindi are click_text quick actions that don't need the screen (destructive when the label
commits), questions about a control are conversation, and a head that sends conversation to a UI
tool fails the retrain gate ("ui fa chat"). The evaluator reports tool accuracy per class and
script. Hermetic: no encoder, synthetic heads."""

from __future__ import annotations

import re

import numpy as np

from neo2.core.reflex import dataset, finetune, templates_hi


def _no_app_names(texts):
    apps = {a.lower() for a in dataset.APPS + templates_hi.APPS}
    # "Zoom In" / "Zoom Out" are a View menu's items, not the app
    return [t for t in texts if any(re.search(rf"\b{re.escape(a)}\b(?!\s+(?:in|out)\b)", t.lower()) for a in apps)]


def test_english_ui_rows_are_clicks_without_the_screen_and_questions_are_chat():
    rows = dataset.ui_rows()
    assert rows == dataset.ui_rows()  # deterministic
    clicks = [r for r in rows if r["intent"] == "quick_action"]
    asks = [r for r in rows if r["intent"] == "chat"]
    assert len(clicks) >= 150 and len(asks) >= 40
    assert all(r["tool"] == "click_text" and r["screen"] is False and r["source"] == "template" for r in clicks)
    commit = {c.lower() for c in dataset.COMMIT_LABELS}
    for r in clicks:
        named = any(c in r["text"].lower() for c in commit)
        assert r["destructive"] == named, r
    assert any(r["destructive"] for r in clicks) and not any(r["destructive"] for r in asks)
    assert all(r["tool"] == "" for r in asks)
    assert _no_app_names(r["text"] for r in rows) == []


def test_hinglish_and_hindi_ui_rows():
    rows = templates_hi.ui_rows()
    assert rows == templates_hi.ui_rows()
    by = {lang: [r for r in rows if r["lang"] == lang and r["intent"] == "quick_action"] for lang in ("hinglish", "hi")}
    assert len(by["hinglish"]) >= 80 and len(by["hi"]) >= 50
    assert all(re.search("[ऀ-ॿ]", r["text"]) for r in by["hi"])
    for r in by["hinglish"] + by["hi"]:
        assert r["tool"] == "click_text" and r["screen"] is False and r["kind"] == "other"
        assert all(isinstance(r[k], bool) for k in ("destructive", "screen", "reply"))
    assert any(r["destructive"] for r in by["hi"] if "भेजें" in r["text"])
    assert not any(r["destructive"] for r in by["hi"] if "सहेजें" in r["text"])
    assert {r["intent"] for r in rows if "kya karta hai" in r["text"] or "क्या करता है" in r["text"]} == {"chat"}
    assert _no_app_names(r["text"] for r in rows) == []


def test_the_training_set_carries_them():
    rows = dataset.build(20)
    texts = {r["text"] for r in rows}
    assert {r["text"] for r in dataset.ui_rows()} <= texts
    assert any(r["lang"] == "hi" and r.get("tool") == "click_text" for r in rows)


def _m(**kw) -> dict:
    m = {"base": 0.90, "user": 0.8, "user_ok": np.array([True] * 80 + [False] * 20), "ece": 0.03, "reply": 0.9,
         "tool": 0.85, "kind": 0.88, "n": {}}  # fmt: skip
    for c in ("chat", "quick_action", "agent_task", "stop"):
        m[f"recall {c}"] = 0.9
    for name in ("destructive", "stop", "needs_screen"):
        m[name], m[f"{name}_ok"] = 0.98, np.array([True] * 98 + [False] * 2)
    for name in ("destructive fa", "destructive fa chat", "stop fa", "needs_screen fa", "needs_screen fa chat"):
        m[name], m[f"{name}_ok"] = 0.01, np.array([True] * 198 + [False] * 2)
    return {**m, **kw}


def test_the_ui_false_alarm_ceiling():
    assert finetune._CEILINGS["ui fa chat"] == 0.01
    ok, lines = finetune._gate(_m(**{"ui fa chat": 0.04}), None, {})
    assert not ok and any("FAIL ui fa chat" in line and "ceiling" in line for line in lines)
    ok, _ = finetune._gate(_m(**{"ui fa chat": 0.005}), None, {})
    assert ok
    ok, _ = finetune._gate(_m(), None, {})  # a head without the measure is judged on the rest
    assert ok


def test_score_counts_conversation_the_tool_lane_would_press():
    rng = np.random.default_rng(0)
    rows = [{"text": f"tell me something about topic number {i}", "intent": "chat", "source": "reflex_hf.jsonl",
             "lang": "en", "hf_intent": "", "destructive": False, "screen": False} for i in range(600)]  # fmt: skip
    rows += [{"text": f"click button number {i}", "intent": "quick_action", "tool": "click_text", "source": "reflex_hf.jsonl",
              "lang": "en", "hf_intent": "", "destructive": False, "screen": False} for i in range(200)]  # fmt: skip
    d = 8
    X = np.hstack([rng.normal(size=(len(rows), d)), np.ones((len(rows), 1))]).astype(np.float32)
    qa = finetune._INTENTS.index("quick_action")

    def block(classes, favourite):
        W = np.zeros((len(classes), d + 1), np.float32)
        b = np.zeros(len(classes), np.float32)
        b[favourite] = 20.0
        return {"members": [[(W, b)]], "T": 1.0, "classes": classes}

    head = {"mu": np.zeros(d + 1, np.float32), "sd": np.ones(d + 1, np.float32), "intent": block(list(finetune._INTENTS), qa),
            "tool": block(["click_text", "none"], 0), **dict.fromkeys(("destructive", "needs_screen", "reply", "kind"))}  # fmt: skip
    sl = finetune.slices(rows)
    m = finetune.score(head, X, finetune.labels(rows), sl, finetune.rule_signals(rows))
    assert m["n"]["ui fa chat"] >= 30 and m["ui fa chat"] == 1.0
    ok, lines = finetune._gate(m, None, {})
    assert not ok and any("FAIL ui fa chat" in line for line in lines)
    quiet = {**head, "tool": block(["click_text", "none"], 1)}
    assert finetune.score(quiet, X, finetune.labels(rows), sl, finetune.rule_signals(rows))["ui fa chat"] == 0.0
