"""neo.decide.learned and the privacy erase of everything the decision layer learned: the
namespaced stores (private files, atomic, capped), the preferences in decide.json, the in-memory
receipts, and a write that started before an erase being dropped. Test data dir only."""

from __future__ import annotations

import asyncio
import json
import stat

import pytest

from neo import privacy
from neo.config import settings
from neo.decide import engine, learned, run


@pytest.fixture(autouse=True)
def _own_request():
    """begin_request() called outside a task sets the test's own context: undo it afterwards.
    Receipts and last actions are process-wide memory: none carries between tests."""
    token = run._gen.set(None)
    run.forget()
    yield
    run.forget()
    run._gen.reset(token)


def test_put_get_drop_and_private_files():
    assert learned.get("sites", "shop|in") is None
    learned.put("sites", "shop|in", {"host": "shop.example.in"})
    assert learned.get("sites", "shop|in") == {"host": "shop.example.in"}
    d = settings().data_dir / "decide_learned"
    assert stat.S_IMODE(d.stat().st_mode) == 0o700
    assert stat.S_IMODE((d / "sites.json").stat().st_mode) == 0o600
    assert not [p for p in d.iterdir() if p.suffix == ".tmp"]
    got = learned.get("sites", "shop|in")
    got["host"] = "changed"  # a copy: the store is untouched
    assert learned.get("sites", "shop|in") == {"host": "shop.example.in"}
    learned.drop("sites", "shop|in")
    learned.drop("sites", "never there")
    assert learned.get("sites", "shop|in") is None
    with pytest.raises(ValueError):
        learned.put("messages", "k", {"text": "never kept"})


def test_the_least_recently_learned_go_past_the_cap(monkeypatch):
    t = iter(range(100, 200))
    monkeypatch.setattr(learned.time, "time", lambda: next(t))
    for k in "abcde":
        learned.put("containers", k, {"id": k}, cap=3)
    assert [learned.get("containers", k) for k in "abcde"] == [None, None, {"id": "c"}, {"id": "d"}, {"id": "e"}]
    learned.put("containers", "c", {"id": "c2"}, cap=3)  # learned again: now the newest
    learned.put("containers", "f", {"id": "f"}, cap=3)
    assert learned.get("containers", "c") == {"id": "c2"} and learned.get("containers", "d") is None


def test_a_damaged_store_starts_fresh():
    learned.put("templates", "h", {"url": "https://h.example/?q={searchTerms}"})
    p = settings().data_dir / "decide_learned" / "templates.json"
    p.write_text('{"h": {"v": {"url": ')
    assert learned.get("templates", "h") is None
    p.write_text(json.dumps({"ok": {"v": {"a": 1}, "t": 1}, "bad": {"v": "x"}, "worse": 3}))
    assert learned.get("templates", "ok") == {"a": 1} and learned.get("templates", "bad") is None
    learned.put("templates", "h", {"url": "u"})
    assert learned.get("templates", "ok") == {"a": 1}


def test_a_write_from_a_request_that_began_before_an_erase_is_dropped(monkeypatch):
    run.begin_request()
    monkeypatch.setattr(privacy, "_generation", privacy.generation() + 1)
    learned.put("bindings", "k", {"contact_id": "c1", "label": "mobile"})
    engine.record_pref("send_message", "test.chat", True, about="h1")
    run.remember_last("send_message", "test.chat")
    assert learned.get("bindings", "k") is None and run.last("send_message") is None
    assert engine.pref_factor("send_message", "test.chat", "h1") == 1.0
    run.begin_request()  # a new request after the erase learns again
    learned.put("bindings", "k", {"contact_id": "c1", "label": "mobile"})
    assert learned.get("bindings", "k") is not None


def test_forget_everything_empties_every_decision_store():
    run.begin_request()
    for ns in learned.NAMESPACES:
        learned.put(ns, "k", {"x": 1})
    engine.record("write_note", "test.notebook", "script_make", True)
    engine.record_pref("send_message", "test.chat", True, about="h1")
    run.remember_last("write_note", "test.notebook")
    counts = asyncio.run(privacy.forget_everything())
    assert counts["learned places & people"] == len(learned.NAMESPACES)
    assert counts["app choices"] == 2  # one arm, one preference
    assert all(learned.get(ns, "k") is None for ns in learned.NAMESPACES)
    assert not (settings().data_dir / "decide_learned").exists()
    assert not engine.proven("write_note", "test.notebook") and engine.pref_factor("send_message", "test.chat", "h1") == 1.0
    assert run.last("write_note") is None
    assert learned.forget() == 0
