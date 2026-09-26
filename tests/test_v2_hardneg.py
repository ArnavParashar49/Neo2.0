"""Hard negatives: nearest public rows to a corrected request, by centred cosine."""

from __future__ import annotations

import numpy as np
import pytest

from neo.reflex.hardneg import hard_negatives
from neo.reflex.learn import EXTRA

D = 12
OFFSET = np.eye(D)[0] * 10.0  # the direction every Laya embedding shares
U, V = np.eye(D)[1], np.eye(D)[2]  # two meanings


def _data(seed: int = 0):
    """Four balanced clusters (±U strong, ±V faint) on a big shared offset, plus one corrected
    request near +U. Faint rows sit closest to the shared direction, so raw cosine prefers them."""
    rng = np.random.default_rng(seed)
    rows, vecs = [], []
    for name, centre in (("near", 3 * U), ("far", -3 * U), ("faint", 0.1 * V), ("faint2", -0.1 * V)):
        for i in range(6):
            rows.append({"text": f"{name} {i}", "intent": "chat" if name == "near" else "quick_action", "source": "reflex_hf.jsonl"})
            vecs.append(OFFSET + centre + rng.normal(0, 0.02, D))
    rows.append({"text": "my request", "intent": "agent_task", "wrong": ["chat"], "source": EXTRA})
    vecs.append(OFFSET + U)
    return rows, np.array(vecs, np.float32)


def _all(rows):
    return np.ones(len(rows), bool)


def test_centred_cosine_finds_the_real_neighbours_where_raw_cosine_finds_generic_rows():
    rows, emb = _data()
    got = hard_negatives(rows, emb, _all(rows), k=6)
    assert sorted(r["text"] for r in got) == [f"near {i}" for i in range(6)]
    raw = emb[:-1] @ emb[-1] / (np.linalg.norm(emb[:-1], axis=1) * np.linalg.norm(emb[-1]))
    assert all(rows[i]["text"].startswith("faint") for i in np.argsort(-raw)[:6])  # why centring matters


def test_rows_come_back_once_at_weight_two_with_their_own_labels():
    rows, emb = _data()
    rows.append({"text": "my other request", "intent": "chat", "wrong": ["quick_action"], "source": EXTRA})
    emb = np.vstack([emb, OFFSET + 1.1 * U])  # a second correction in the same region
    before = [dict(r) for r in rows]
    got = hard_negatives(rows, emb, _all(rows), k=6)
    assert len(got) == len({r["text"] for r in got}) == 6
    assert all(r["w"] == 2.0 and r["hardneg"] and r["intent"] == "chat" and "wrong" not in r for r in got)
    assert rows == before  # the training rows themselves are untouched


def test_only_corrections_seed_neighbours_and_the_users_own_rows_are_never_returned():
    rows, emb = _data()
    rows[-1]["wrong"] = []  # not a correction
    assert hard_negatives(rows, emb, _all(rows)) == []
    rows, emb = _data()
    rows.append({"text": "another thing I said", "intent": "chat", "source": EXTRA})  # right on top of it
    emb = np.vstack([emb, OFFSET + U])
    rows.append({"text": "My request ", "intent": "chat", "source": "reflex_generated.jsonl"})  # the same sentence
    emb = np.vstack([emb, OFFSET + U])
    texts = {r["text"] for r in hard_negatives(rows, emb, _all(rows), k=6)}
    assert texts == {f"near {i}" for i in range(6)}


def test_rows_held_out_to_judge_the_retrain_are_never_trained_on():
    rows, emb = _data()
    trainable = _all(rows)
    trainable[:3] = False  # "near 0-2" judge the new head
    got = {r["text"] for r in hard_negatives(rows, emb, trainable, k=6)}
    assert len(got) == 6 and got.isdisjoint({"near 0", "near 1", "near 2"})
    trainable[-1] = False  # the correction itself is held out: it picks nothing either
    assert hard_negatives(rows, emb, trainable, k=6) == []


def test_k_is_capped_by_the_pool_and_the_mean_can_be_given():
    rows, emb = _data()
    assert len(hard_negatives(rows, emb, _all(rows), k=500)) == 24  # every public row, once
    got = hard_negatives(rows, emb, _all(rows), k=3, mean=emb.mean(0))
    assert len(got) == 3 and all(r["text"].startswith("near") for r in got)
    with pytest.raises(ValueError):
        hard_negatives(rows, emb[:-1], _all(rows))
    with pytest.raises(ValueError):
        hard_negatives(rows, emb, _all(rows)[:-1])
