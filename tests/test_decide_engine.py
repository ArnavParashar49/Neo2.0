"""The decision engine: which installed app should do a request, Laya deciding, outcomes learned.

Laya is faked (a stub whose system_one returns probabilities, installed as the zero-shot model;
the reflex is a stub with a kind head); apps are built directly. Nothing here loads a model, or
launches, scripts or inspects a real app.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
import threading
import time
import types

import numpy as np
import pytest

from neo.decide import engine
from neo.reflex import zeroshot

try:
    from neo.decide.inventory import App

    _STUB_INVENTORY = None
except ImportError:  # the inventory module lands separately: a minimal stand-in until it does

    @dataclasses.dataclass(frozen=True)
    class App:  # type: ignore[no-redef]
        bundle_id: str
        name: str
        path: str
        category: str
        viewer_types: tuple
        editor_types: tuple
        schemes: tuple
        own_schemes: tuple
        handles: tuple
        script: object
        media_evidence: tuple
        web_start_url: str
        last_used: float | None
        use_count: int
        days_used_30: int
        running: bool
        in_dock: bool

    def _affinity(app, now=None):
        now = now or time.time()
        recent = 0.0 if not app.last_used else max(0.0, 1 - (now - app.last_used) / (90 * 86400))
        return min(1.0, 0.4 * recent + 0.4 * app.days_used_30 / 30 + 0.2 * app.running)

    def _named_in(text, apps=None):
        low = f" {text.lower()} "
        hits = [a for a in apps or [] if f" {a.name.lower()} " in low]
        return max(hits, key=lambda a: len(a.name)) if hits else None

    _STUB_INVENTORY = types.SimpleNamespace(
        App=App, affinity=_affinity, named_in=_named_in, inventory=lambda refresh=False: []
    )

NOW = time.time()
DAY = 86400.0


def mk(name: str, bid: str, **kw) -> App:
    f = {
        "bundle_id": bid,
        "name": name,
        "path": f"/Applications/{name}.app",
        "category": "",
        "viewer_types": (),
        "editor_types": (),
        "schemes": (),
        "own_schemes": (),
        "handles": (),
        "script": None,
        "media_evidence": (),
        "web_start_url": "",
        "last_used": None,
        "use_count": 0,
        "days_used_30": 0,
        "running": False,
        "in_dock": False,
    }
    f.update(kw)
    return App(**f)


HEAVY = {"last_used": NOW - 60, "use_count": 400, "days_used_30": 28, "running": True, "in_dock": True}
LIGHT = {"last_used": NOW - 20 * DAY, "use_count": 3, "days_used_30": 1}
MEDIA = ("declares a music category",)

# Fixture apps: names are made up, as any real Mac's would be unknown to the engine.
TUNEWAVE = mk("Tunewave", "test.tunewave", category="public.app-category.music", media_evidence=MEDIA, **HEAVY)
CLIPTIDE = mk("Cliptide", "test.cliptide", category="public.app-category.video", media_evidence=MEDIA, **LIGHT)
PODNEST = mk("Podnest", "test.podnest", media_evidence=("links a media framework",))  # installed, never used
NOTEPADLY = mk("Notepadly", "test.notepadly", category="public.app-category.productivity", **HEAVY)
BROWSA = mk(
    "Browsa", "test.browsa", schemes=("http", "https"), handles=("http", "https"),
    viewer_types=("public.html",), media_evidence=("plays web video",), **HEAVY,
)
CAPS = {"test.tunewave": frozenset({"media.search_play", "media.state"}), "test.notepadly": frozenset({"note.make"})}


class FakeLaya:
    """Answers every choice with ``pick(state, labels) -> {label: p}``; records what it was asked."""

    def __init__(self, pick):
        self.pick = pick
        self.calls: list[tuple[object, dict]] = []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        answers = {}
        for qid, q in questions.items():
            probs = self.pick(state, list(q["criteria"]))
            answers[qid] = {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs}
        return {"answers": answers}


def prefer(label: str | None, p: float = 0.85):
    def pick(state, labels):
        if label not in labels:
            return {k: 1 / len(labels) for k in labels}
        rest = (1 - p) / max(1, len(labels) - 1)
        return {k: (p if k == label else rest) for k in labels}

    return pick


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch):
    import neo.reflex

    if _STUB_INVENTORY is not None:
        monkeypatch.setitem(sys.modules, "neo.decide.inventory", _STUB_INVENTORY)
    monkeypatch.setattr(engine, "DRIVERS", {})  # generic methods: script / url / ax
    monkeypatch.setattr(engine, "_app_caps", lambda app: CAPS.get(app.bundle_id, frozenset()))
    monkeypatch.setattr(neo.reflex, "get_laya", lambda: None)
    monkeypatch.setattr(zeroshot, "_agent", None)  # Laya is off in tests: nothing ever loads
    engine._cache.clear()
    yield
    engine._cache.clear()


def use_laya(monkeypatch, pick) -> FakeLaya:
    """Install a fake as the zero-shot model (loaded, so no load is attempted)."""
    fake = FakeLaya(pick)
    monkeypatch.setattr(zeroshot, "_agent", fake)
    return fake


class FakeReflex:
    """The reflex with trained heads: its Decision carries the kind head's answer ('' = the head
    abstained, as it does for a chat). ``kind_head=False``: trained heads, but no kind head yet."""

    has_head = True

    def __init__(self, kind: str | None, p: float = 0.0, kind_head: bool = True):
        self.kind, self.p, self.has_kind_head = kind, p, kind_head
        self.texts: list[str] = []

    def decide(self, text):
        self.texts.append(text)
        return types.SimpleNamespace(intent="quick_action", kind=self.kind, kind_p=self.p)


def use_reflex(monkeypatch, kind: str | None, p: float = 0.0, kind_head: bool = True) -> FakeReflex:
    import neo.reflex

    fake = FakeReflex(kind, p, kind_head)
    monkeypatch.setattr(neo.reflex, "get_laya", lambda: fake)
    return fake


def order(cands) -> list[str]:
    return [c.app.name if c.app is not None else c.key for c in cands]


def choose(kind, text, apps):
    return asyncio.run(engine.choose(kind, text, apps))


APPS = [NOTEPADLY, PODNEST, CLIPTIDE, BROWSA, TUNEWAVE]


# ---- ordering -----------------------------------------------------------------------------------


def test_named_app_goes_first_even_when_laya_prefers_another(monkeypatch):
    use_laya(monkeypatch, prefer("Tunewave", 0.95))
    c = choose("play_media", "play some lofi on Cliptide", APPS)
    assert order(c)[0] == "Cliptide"
    assert "named by the user" in c[0].evidence
    assert order(c).count("Cliptide") == 1


def test_named_app_that_cannot_do_the_task_is_ignored_unless_asked_for_explicitly():
    photo = mk("Snapshot", "test.snapshot", category="public.app-category.photography", **HEAVY)
    apps = [*APPS, photo]
    # a song that happens to share an installed app's name: not a request for that app
    assert "Snapshot" not in order(choose("play_media", "play snapshot by the band", apps))
    # "... on <app>" is: the user gets what they asked for
    assert order(choose("play_media", "play the news on Snapshot", apps))[0] == "Snapshot"
    # a task any app could be meant for takes the named app as is
    assert order(choose("open_app", "open snapshot", apps)) == ["Snapshot", "web"]


@pytest.mark.skipif(_STUB_INVENTORY is not None, reason="needs the inventory's name matching")
def test_an_app_asked_for_by_a_short_name_is_used_even_if_it_does_not_fit():
    mailer = mk("Quillmail Desktop", "test.quill", **LIGHT)  # declares nothing mail-like
    apps = [*APPS, mailer]
    assert order(choose("send_email", "email bob the report with quillmail", apps)) == ["Quillmail Desktop"]
    assert order(choose("send_email", "email bob with the quillmail desktop app", apps)) == ["Quillmail Desktop"]
    # named, but not as the app to use: a word that only follows the preposition further on
    assert order(choose("send_email", "email bob with notes from quillmail", apps)) == []


def test_laya_orders_the_used_apps_when_it_is_sure(monkeypatch):
    fake = use_laya(monkeypatch, prefer("Cliptide", 0.85))
    c = choose("play_media", "play the trailer", APPS)
    assert order(c)[:2] == ["Cliptide", "Tunewave"]
    assert c[0].laya_p == pytest.approx(0.85)
    assert c[0].score > c[1].score

    state, q = fake.calls[-1]
    assert state == {"request": "play the trailer"}  # evidence lives in the options, once
    crit = q["app"]["criteria"]
    assert set(crit) == {"Tunewave", "Cliptide"}  # only used, fitting apps; no named/unused/web
    # each option is its app's short evidence: category, usage, NEO's record once there is one
    assert crit["Tunewave"] == "music; used 28 of last 30 days"
    assert crit["Cliptide"] == "video; used 1 of last 30 days"
    engine.record("play_media", "test.cliptide", "ax", True)
    choose("play_media", "play the trailer", APPS)
    assert fake.calls[-1][1]["app"]["criteria"]["Cliptide"].endswith("worked 1 of 1 tries")


def test_prior_orders_when_laya_is_unsure(monkeypatch):
    use_laya(monkeypatch, lambda s, labels: {k: (0.52 if k == "Cliptide" else 0.48) for k in labels})
    c = choose("play_media", "play something", APPS)
    # the app this user actually uses most; one opened once, weeks ago, after the web
    assert order(c)[:3] == ["Tunewave", "web", "Cliptide"]
    assert c[2].laya_p == pytest.approx(0.52)
    assert c[0].score > c[2].score


def test_prior_orders_when_laya_is_unavailable_broken_or_busy(monkeypatch):
    import neo.reflex

    assert order(choose("play_media", "play something", APPS))[:3] == ["Tunewave", "web", "Cliptide"]

    class Broken:
        def system_one(self, state, questions):
            raise RuntimeError("MPS fell over")

    monkeypatch.setattr(zeroshot, "_agent", Broken())
    assert order(choose("play_media", "play something", APPS))[:3] == ["Tunewave", "web", "Cliptide"]

    use_laya(monkeypatch, prefer("Cliptide", 0.99))
    monkeypatch.setattr(engine, "LAYA_LOCK_S", 0.05)
    lock = neo.reflex.laya_lock()
    with lock:  # the reflex (or a retrain) holds Laya: don't wait, decide by the prior
        t = time.perf_counter()
        c = choose("play_media", "play something", APPS)
        assert time.perf_counter() - t < 1.0
    assert order(c)[:3] == ["Tunewave", "web", "Cliptide"]
    assert order(choose("play_media", "play something", APPS))[0] == "Cliptide"  # lock free again


def test_a_busy_laya_costs_the_lock_wait_not_the_whole_timeout(monkeypatch):
    # the real get_laya() takes the shared lock itself (no timeout): the engine must not block in it
    import neo.reflex

    fake = use_laya(monkeypatch, prefer("Cliptide", 0.99))
    reflex = FakeReflex("play_media", 0.9)

    def get_laya_taking_the_lock():
        with neo.reflex.laya_lock():
            return reflex

    monkeypatch.setattr(neo.reflex, "get_laya", get_laya_taking_the_lock)
    monkeypatch.setattr(engine, "LAYA_LOCK_S", 0.05)
    monkeypatch.setattr(engine, "LAYA_TIMEOUT_S", 2.0)
    held, done = threading.Event(), threading.Event()

    def reflex_busy():  # another thread holds Laya (a retrain chunk, a load); lets go within 3 s
        with neo.reflex.laya_lock():
            held.set()
            done.wait(3)

    holder = threading.Thread(target=reflex_busy)
    holder.start()
    held.wait(1)
    try:
        t = time.perf_counter()
        c = choose("play_media", "play something", APPS)
        kind = asyncio.run(engine.decide_kind("play jazz"))
        dt = time.perf_counter() - t
    finally:
        done.set()
        holder.join()
    assert dt < 1.0
    assert order(c)[:3] == ["Tunewave", "web", "Cliptide"] and not fake.calls
    assert kind == ("other", 0.0, False) and not reflex.texts
    assert order(choose("play_media", "play something", APPS))[0] == "Cliptide"
    assert asyncio.run(engine.decide_kind("play jazz")) == ("play_media", pytest.approx(0.9), False)


def test_an_answer_that_came_too_late_is_never_computed(monkeypatch):
    import neo.reflex

    fake = FakeReflex("play_media", 0.9)
    gate = threading.Event()

    def slow_load():  # the model is still loading when the caller gives up
        gate.wait(5)
        return fake

    monkeypatch.setattr(neo.reflex, "get_laya", slow_load)
    monkeypatch.setattr(engine, "LAYA_TIMEOUT_S", 0.1)

    async def run():
        r = await engine.decide_kind("play jazz")
        gate.set()
        await asyncio.sleep(0.2)  # the worker finishes loading after the caller gave up
        return r

    assert asyncio.run(run()) == ("other", 0.0, False)
    assert not fake.texts


def test_web_fallback_after_the_apps_used_most_and_rarely_used_and_unused_apps_after_it():
    c = choose("play_media", "play a song", APPS)
    # the productivity app doesn't fit; the default browser is what the web fallback opens
    assert order(c) == ["Tunewave", "web", "Cliptide", "Podnest"]
    web = c[1]
    assert web.app is None and web.methods == ("web",) and web.key == "web"
    assert c[0].methods == ("script", "ax")  # scriptable search+play first; the AX tree after


def test_no_used_app_means_the_web_goes_first():
    assert order(choose("play_media", "play a song", [PODNEST, NOTEPADLY])) == ["web", "Podnest"]


def test_attempts_are_capped_and_the_fallback_keeps_its_slot():
    apps = [mk(f"Player{i}", f"test.p{i}", media_evidence=MEDIA, **HEAVY) for i in range(9)]
    c = choose("play_media", "play a song", [*apps, PODNEST])
    assert len(c) == engine.MAX_CANDIDATES
    assert c[-1].app is None


def test_a_local_fallback_comes_after_every_used_app(monkeypatch):
    # a file is what's left when the apps fail, never a rival to an app this user opens now and then
    notes = engine.Task("save a note", wants=("note.make",), fallback="file", laya_min_used=1)
    monkeypatch.setitem(engine.TASKS, "write_note", notes)
    rare = mk("Notepadly", "test.notepadly", last_used=NOW - 80 * DAY, use_count=1)  # opened once, long ago
    assert order(choose("write_note", "make a note", [rare])) == ["Notepadly", "file"]
    monkeypatch.setitem(engine.TASKS, "write_note", dataclasses.replace(notes, fallback="web"))
    assert order(choose("write_note", "make a note", [rare])) == ["web", "Notepadly"]  # the web's bar stays


def test_every_browser_is_left_to_the_web_fallback_unless_named():
    # not the default browser, but it claims http(s) and opens HTML: the web fallback covers it
    other = mk(
        "Surfer", "test.surfer", schemes=("http", "https", "surfer"), viewer_types=("public.html",),
        media_evidence=("web browser",), **HEAVY,
    )
    # claiming http alone doesn't make a browser (a player or chat app may)
    chat = mk("Asker", "test.asker", schemes=("http", "https"), media_evidence=MEDIA, **LIGHT)
    apps = [*APPS, other, chat]
    got = order(choose("play_media", "play a song", apps))
    assert "Surfer" not in got and "Browsa" not in got and "Asker" in got
    assert order(choose("play_media", "play a song in Surfer", apps))[0] == "Surfer"


def test_other_kinds_find_their_apps_from_declared_evidence():
    mailer = mk("Postbox", "test.mail", handles=("mailto",), **LIGHT)
    chat = mk("Chatter", "test.chat", category="public.app-category.social-networking", **HEAVY)
    cal = mk("Datebook", "test.cal", viewer_types=("com.example.ical.ics",), **LIGHT)
    apps = [*APPS, mailer, chat, cal]
    assert order(choose("send_email", "email mom", apps)) == ["Postbox"]
    assert order(choose("send_message", "text mom", apps)) == ["Chatter"]
    assert order(choose("add_event", "lunch friday at 1", apps)) == ["Datebook"]
    assert order(choose("write_note", "note: buy milk", apps)) == ["Notepadly"]
    assert order(choose("open_app", "open the thing", apps)) == ["web"]  # nothing named
    assert order(choose("web_search", "search for cats", apps)) == ["web"]
    assert order(choose("no_such_kind", "hmm", apps)) == []


def test_a_registered_driver_supplies_the_methods(monkeypatch):
    drv = types.SimpleNamespace(
        methods_for=lambda app: () if app.bundle_id == "test.cliptide" else ("ax_search_play", "learned_uri")
    )
    monkeypatch.setitem(sys.modules, "fake_media_driver", drv)
    monkeypatch.setattr(engine, "DRIVERS", {"play_media": "fake_media_driver"})
    c = choose("play_media", "play a song", APPS)
    assert order(c) == ["Tunewave", "web", "Podnest"]  # an app the driver can't drive is dropped
    assert c[0].methods == ("learned_uri", "ax_search_play")  # best prior first


def test_many_used_apps_go_through_laya_shortlist(monkeypatch):
    music = {"category": "public.app-category.music", "media_evidence": MEDIA}
    apps = [mk(f"Player{i:02d}", f"test.p{i}", **music, **LIGHT) for i in range(24)]
    film = mk("Reelio", "test.reelio", category="public.app-category.video", media_evidence=MEDIA, **LIGHT)
    apps.insert(17, film)

    from neo.memory import embed

    seen = []

    def minilm(texts):  # NEO's memory embedder, faked: "video" texts point one way, the rest the other
        seen.append(list(texts))
        return np.array([[1.0, 0.0] if "video" in t or "film" in t else [0.0, 1.0] for t in texts])

    monkeypatch.setattr(embed, "embed_texts", minilm)
    monkeypatch.setattr(embed, "ready", lambda: True)  # NEO's start loaded it
    fake = use_laya(monkeypatch, lambda s, labels: {k: (0.9 if k == "Reelio" else 0.1 / (len(labels) - 1)) for k in labels})
    c = asyncio.run(engine.choose("play_media", "play a film", apps))
    assert order(c)[:2] == ["Reelio", "web"]  # Laya's sure pick, then the web: the rest were opened once
    assert seen[0][0] == "play a film" and len(seen[0]) == 1 + len(apps)  # the request, then each app
    crit = fake.calls[-1][1]["app"]["criteria"]
    assert "Reelio" in crit and len(crit) == engine.LAYA_POOL  # the shortlist, not the most-used six
    assert len(c) == engine.MAX_CANDIDATES


def test_without_the_embedder_the_most_used_apps_go_to_laya(monkeypatch):
    from neo.memory import embed

    music = {"category": "public.app-category.music", "media_evidence": MEDIA}
    apps = [mk(f"Player{i:02d}", f"test.p{i}", **music, **LIGHT) for i in range(24)]
    fave = mk("Reelio", "test.reelio", category="public.app-category.video", media_evidence=MEDIA, **HEAVY)
    fake = use_laya(monkeypatch, prefer("Reelio", 0.9))
    monkeypatch.setattr(embed, "ready", lambda: True)
    c = asyncio.run(engine.choose("play_media", "play a film", [*apps, fave]))  # embed_texts -> None
    assert len(fake.calls[-1][1]["app"]["criteria"]) == engine.LAYA_POOL
    assert order(c)[0] == "Reelio"


def test_the_shortlist_never_waits_for_the_embedder_and_skips_when_laya_cannot_answer(monkeypatch):
    from neo.memory import embed

    music = {"category": "public.app-category.music", "media_evidence": MEDIA}
    apps = [mk(f"Player{i:02d}", f"test.p{i}", **music, **LIGHT) for i in range(24)]
    apps.append(mk("Reelio", "test.reelio", category="public.app-category.video", media_evidence=MEDIA, **HEAVY))
    calls, gate = [], threading.Event()

    def slow(texts):  # another caller's batch holds MiniLM
        calls.append(len(texts))
        gate.wait(3)
        return np.array([[0.0, 1.0]] * len(texts))

    monkeypatch.setattr(embed, "embed_texts", slow)
    monkeypatch.setattr(engine, "LAYA_TIMEOUT_S", 0.3)
    fake = use_laya(monkeypatch, prefer("Reelio", 0.9))

    async def timed():
        t = time.perf_counter()
        c = await engine.choose("play_media", "play a film", apps)
        dt = time.perf_counter() - t
        gate.set()  # let a stuck worker finish so the loop can close
        return c, dt

    # not loaded yet (NEO's start is still warming it): no shortlist, never a load from here
    c, dt = asyncio.run(timed())
    assert not calls and dt < 0.5 and order(c)[0] == "Reelio"
    assert len(fake.calls[-1][1]["app"]["criteria"]) == engine.LAYA_POOL
    # loaded but busy: the shortlist and the question share LAYA_TIMEOUT_S, then the prior decides
    gate.clear()
    monkeypatch.setattr(embed, "ready", lambda: True)
    c, dt = asyncio.run(timed())
    assert calls and dt < 0.8 and order(c)[0] == "Reelio"
    # zero-shot Laya is off: nothing would read a shortlist, so none is made
    calls.clear()
    monkeypatch.setattr(zeroshot, "_agent", None)
    c, dt = asyncio.run(timed())
    assert not calls and order(c)[0] == "Reelio"


# ---- learning ------------------------------------------------------------------------------------


def test_three_failures_demote_and_a_success_recovers():
    k = "test.tunewave"
    for m in ("script", "ax"):
        engine.record("play_media", k, m, False)
        engine.record("play_media", k, m, False)
    assert order(choose("play_media", "play a song", APPS))[0] == "Tunewave"  # two strikes: still first

    engine.record("play_media", k, "script", False)
    c = choose("play_media", "play a song", APPS)
    tw = next(x for x in c if x.app is not None and x.app.name == "Tunewave")
    assert tw.methods == ("ax", "script")  # the failing method goes last within the app

    engine.record("play_media", k, "ax", False)
    c = choose("play_media", "play a song", APPS)
    assert order(c) == ["web", "Cliptide", "Podnest", "Tunewave"]  # still tried, so it can recover
    assert "failed last 3 tries" in c[-1].evidence

    engine.record("play_media", k, "ax", True)
    c = order(choose("play_media", "play a song", APPS))
    assert c.index("Tunewave") < c.index("web")


def test_an_untried_method_does_not_hide_the_ones_that_keep_failing(monkeypatch):
    # a driver whose first method only runs when it has something to replay (so it is rarely
    # attempted): its optimistic seed must not keep a failing app ahead of the web
    drv = types.SimpleNamespace(methods_for=lambda app: ("learned_uri", "script_search_play", "ax_search_play"))
    monkeypatch.setitem(sys.modules, "fake_media_driver", drv)
    monkeypatch.setattr(engine, "DRIVERS", {"play_media": "fake_media_driver"})
    before = choose("play_media", "play a song", APPS)
    assert order(before)[0] == "Tunewave"
    engine.record("play_media", "test.tunewave", "script_search_play", False)
    after_one = choose("play_media", "play a song", APPS)
    assert after_one[0].score < before[0].score  # a real failure moves the app's prior
    for _ in range(2):
        engine.record("play_media", "test.tunewave", "script_search_play", False)
    for _ in range(3):
        engine.record("play_media", "test.tunewave", "ax_search_play", False)
    c = order(choose("play_media", "play a song", APPS))
    assert c.index("Tunewave") > c.index("web")
    # permission denials don't count as tries: they never demote the app
    engine._cache.clear()
    from neo.config import settings

    (settings().data_dir / "decide.json").unlink()
    for _ in range(5):
        engine.record("play_media", "test.tunewave", "script_search_play", False, "permission")
    assert order(choose("play_media", "play a song", APPS))[0] == "Tunewave"


def test_a_demotion_lapses_so_the_app_gets_probed_again(monkeypatch):
    for m in ("script", "ax"):
        for _ in range(3):
            engine.record("play_media", "test.tunewave", m, False)
    c = order(choose("play_media", "play", APPS))
    assert c.index("Tunewave") > c.index("Podnest")  # demoted: after even the unused apps
    later = time.time() + engine.DEMOTE_FOR_S + 60
    monkeypatch.setattr(engine, "_now", lambda: later)
    c = order(choose("play_media", "play", APPS))
    assert c.index("Tunewave") < c.index("web")


def test_success_and_failure_move_the_prior_and_blame_rules_hold():
    base = engine.prior("play_media", "test.x", "ax")
    assert base == pytest.approx(0.5)
    for reason in sorted(engine.NO_BLAME):
        for _ in range(4):
            engine.record("play_media", "test.x", "ax", False, reason)
    assert engine.prior("play_media", "test.x", "ax") == pytest.approx(base)  # not its fault
    engine.record("play_media", "test.x", "ax", False, "timeout")
    assert engine.prior("play_media", "test.x", "ax") < base
    engine.record("play_media", "test.x", "ax", True)
    engine.record("play_media", "test.x", "ax", True)
    assert engine.prior("play_media", "test.x", "ax") > base
    # a permission denial never demotes an app the user relies on
    for _ in range(5):
        engine.record("play_media", "test.tunewave", "script", False, "permission")
        engine.record("play_media", "test.tunewave", "ax", False, "login")
    assert order(choose("play_media", "play", APPS))[0] == "Tunewave"


def test_the_web_fallback_learns_like_any_arm():
    for _ in range(3):
        engine.record("play_media", engine.WEB, engine.WEB, True)
    assert engine.prior("play_media", "web", "web") > engine.prior("play_media", "never", "web")
    web = next(c for c in choose("play_media", "play", APPS) if c.app is None)
    assert "worked 3 of 3" in web.evidence


def test_an_app_with_a_track_record_counts_as_used_even_without_spotlight_data():
    engine.record("play_media", "test.podnest", "ax", True)
    c = order(choose("play_media", "play a podcast", APPS))
    assert c.index("Podnest") < c.index("web")


def test_decide_json_is_written_atomically(monkeypatch):
    from neo.config import settings

    engine.record("play_media", "test.tunewave", "ax", True)
    p = settings().data_dir / "decide.json"
    data = json.loads(p.read_text())
    arm = data["arms"]["play_media"]["test.tunewave"]["ax"]
    assert arm["ok"] == 1 and arm["fails"] == 0
    assert not [x for x in p.parent.iterdir() if x.suffix == ".tmp"]

    before = p.read_text()

    def crash(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(engine.os, "replace", crash)
    engine.record("play_media", "test.tunewave", "ax", False)  # must not raise
    assert p.read_text() == before  # the old file is intact, never half-written
    assert not [x for x in p.parent.iterdir() if x.suffix == ".tmp"]
    assert engine.prior("play_media", "test.tunewave", "ax") < engine._mean(arm)  # learned for the session


def test_decide_json_survives_corruption():
    from neo.config import settings

    p = settings().data_dir / "decide.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"arms": {"play_media": {"test.tunewave": {"ax": {"a": 1, "b"')
    assert engine.prior("play_media", "test.tunewave", "ax") == pytest.approx(0.5)
    assert order(choose("play_media", "play", APPS))[0] == "Tunewave"
    assert (p.parent / "decide.json.corrupt").exists()
    engine.record("play_media", "test.tunewave", "ax", True)
    assert json.loads(p.read_text())["arms"]["play_media"]["test.tunewave"]["ax"]["ok"] == 1

    engine._cache.clear()  # well-formed JSON with junk inside: the junk is dropped, the rest kept
    p.write_text(json.dumps({"arms": {"play_media": {"a": {"ax": {"a": "x"}}, "b": [], "c": {"ax": {"a": 5, "b": 1}}}}}))
    assert engine.prior("play_media", "a", "ax") == pytest.approx(0.5)
    assert engine.prior("play_media", "c", "ax") == pytest.approx(5 / 6)
    p.write_text("[]")
    assert engine.prior("play_media", "c", "ax") == pytest.approx(0.5)


def test_concurrent_records_are_not_lost():
    def worker(i):
        for _ in range(10):
            engine.record("play_media", f"test.t{i % 2}", "ax", True)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    from neo.config import settings

    data = json.loads((settings().data_dir / "decide.json").read_text())
    assert sum(data["arms"]["play_media"][k]["ax"]["ok"] for k in ("test.t0", "test.t1")) == 40


# ---- the task kind ------------------------------------------------------------------------------


def test_decide_kind_comes_from_the_reflex_kind_head(monkeypatch):
    fake = use_laya(monkeypatch, prefer("send_message", 0.99))
    reflex = use_reflex(monkeypatch, "play_media", 0.62)
    k = asyncio.run(engine.decide_kind(" put on some jazz "))
    assert k == ("play_media", pytest.approx(0.62), False) and not k.advisory
    assert reflex.texts == ["put on some jazz"] and not fake.calls  # no zero-shot question at all
    # the caller's own Decision for the text is used as is: the reflex isn't asked again
    d = types.SimpleNamespace(kind="add_event", kind_p=0.7)
    assert asyncio.run(engine.decide_kind("lunch friday", d)) == ("add_event", pytest.approx(0.7), False)
    assert reflex.texts == ["put on some jazz"] and not fake.calls


@pytest.fixture
def zs_asked(monkeypatch):
    """Every zero-shot question decide_kind asks (none is answered)."""
    asked = []
    monkeypatch.setattr(zeroshot, "ask", lambda state, questions, wait_s, stale=None: asked.append(state))
    return asked


def test_a_kind_head_that_is_busy_or_abstains_is_never_second_guessed_by_zero_shot(monkeypatch, zs_asked):
    import neo.reflex

    # a retrain holds Laya past LAYA_LOCK_S: the head may well know, so don't guess - and don't
    # start loading 1.7 GB of zero-shot model while memory is tightest
    reflex = use_reflex(monkeypatch, "play_media", 0.9)
    monkeypatch.setattr(engine, "LAYA_LOCK_S", 0.05)
    with neo.reflex.laya_lock():
        assert asyncio.run(engine.decide_kind("play jazz")) == ("other", 0.0, False)
    assert not zs_asked and not reflex.texts
    # the head abstains (not an action): that is its answer
    reflex = use_reflex(monkeypatch, "", 0.0)
    assert asyncio.run(engine.decide_kind("how are you")) == ("other", 0.0, False)
    assert reflex.texts == ["how are you"] and not zs_asked
    # the caller's Decision abstained, and the reflex has a kind head: not asked again either
    d = types.SimpleNamespace(intent="chat", kind="", kind_p=0.0)
    assert asyncio.run(engine.decide_kind("how are you", d)) == ("other", 0.0, False)
    assert reflex.texts == ["how are you"] and not zs_asked
    # only a reflex without a kind head falls back to zero-shot
    use_reflex(monkeypatch, None, kind_head=False)
    asyncio.run(engine.decide_kind("play jazz"))
    assert zs_asked == [{"text": "play jazz"}]


def test_a_reflex_that_only_has_its_head_file_is_read_for_a_kind_head(monkeypatch, zs_asked):
    import neo.reflex

    class Reflex:  # no has_kind_head flag: its loaded head file says
        has_head = True

        def __init__(self, head):
            self._head = head

        def decide(self, text):
            return types.SimpleNamespace(kind="play_media", kind_p=0.8)

    monkeypatch.setattr(neo.reflex, "get_laya", lambda: Reflex({"intent": {}, "kind": {"classes": []}}))
    assert asyncio.run(engine.decide_kind("play jazz")) == ("play_media", pytest.approx(0.8), False)
    assert not zs_asked
    monkeypatch.setattr(neo.reflex, "get_laya", lambda: Reflex({"intent": {}, "kind": None}))
    asyncio.run(engine.decide_kind("play jazz"))
    assert zs_asked == [{"text": "play jazz"}]


def test_decide_kind_takes_at_most_the_laya_timeout_in_all(monkeypatch):
    import neo.reflex

    gate = threading.Event()

    def slow_reflex():
        time.sleep(0.3)
        return FakeReflex(None, kind_head=False)

    monkeypatch.setattr(neo.reflex, "get_laya", slow_reflex)
    monkeypatch.setattr(zeroshot, "ask", lambda *a, **kw: gate.wait(3))  # the zero-shot model hangs
    monkeypatch.setattr(engine, "LAYA_TIMEOUT_S", 0.5)

    async def timed():
        t = time.perf_counter()
        r = await engine.decide_kind("play jazz")
        dt = time.perf_counter() - t
        gate.set()
        return r, dt

    r, dt = asyncio.run(timed())
    assert r == ("other", 0.0, False) and dt < 0.7  # the reflex's 0.3 s came out of the same budget


def test_zero_shot_kind_counts_only_when_very_sure_and_only_as_advice(monkeypatch):
    use_reflex(monkeypatch, None, kind_head=False)  # trained heads, but no kind head yet
    fake = use_laya(monkeypatch, prefer("play_media", 0.97))
    k = asyncio.run(engine.decide_kind("put on some jazz"))
    # a hint, never a route: even at this bar 9% of the kinds it gave were wrong on held-out requests
    assert k == ("play_media", pytest.approx(0.97), True) and k.advisory
    state, q = fake.calls[-1]
    assert state == {"text": "put on some jazz"}
    assert set(q["kind"]["criteria"]) == set(engine.TASKS)
    assert {"play_media", "send_message", "write_note", "add_reminder", "add_event", "send_email",
            "open_app", "web_search", "other"} <= set(engine.TASKS)
    # sure enough to order apps, not to pick the kind: zero-shot was measured 23% sure-and-wrong there
    for p in (0.9, 0.6):
        use_laya(monkeypatch, prefer("send_message", p))
        assert asyncio.run(engine.decide_kind("remind me to call mom")) == ("other", 0.0, False)
    # a Decision without a kind (the lite reflex, rules) falls back the same way, and so does a
    # label the engine has no task for
    use_laya(monkeypatch, prefer("add_reminder", 0.99))
    for d in (types.SimpleNamespace(intent="quick_action"), types.SimpleNamespace(kind="dance", kind_p=1.0)):
        got = asyncio.run(engine.decide_kind("remind me to call mom", d))
        assert got == ("add_reminder", pytest.approx(0.99), True)


def test_decide_kind_without_laya_is_other():
    assert asyncio.run(engine.decide_kind("put on some jazz")) == ("other", 0.0, False)
    assert asyncio.run(engine.decide_kind("  ")) == ("other", 0.0, False)


def test_decide_kind_gives_up_quickly_when_laya_hangs(monkeypatch):
    import neo.reflex

    gate = threading.Event()

    def slow():
        gate.wait(5)
        return None

    monkeypatch.setattr(neo.reflex, "get_laya", slow)
    monkeypatch.setattr(engine, "LAYA_TIMEOUT_S", 0.1)

    async def timed():
        t = time.perf_counter()
        r = await engine.decide_kind("play jazz")
        dt = time.perf_counter() - t
        gate.set()  # let the stuck worker finish so the loop can close
        return r, dt

    r, dt = asyncio.run(timed())
    assert r == ("other", 0.0, False) and dt < 1.0


def test_no_app_names_in_the_engine():
    import inspect

    src = inspect.getsource(engine).lower()
    for word in ("spotify", "itunes", "apple music", "youtube", "safari", "chrome", "com.apple.", "vlc", "whatsapp"):
        assert word not in src, word


# ---- sites, editors, NEO itself ---------------------------------------------------------------


@pytest.fixture
def dictionary(monkeypatch):
    """A tiny system word list: ordinary words vs. made-up site names."""
    import neo.decide.inventory as inv

    common = {"repeat", "loop", "speaker", "background", "shuffle"}
    monkeypatch.setattr(inv, "_common_words", lambda words: {w: w in common for w in words})


def test_a_named_website_is_a_hint_only_when_it_is_no_app_and_no_ordinary_word(dictionary):
    assert engine.site_hint("play 505 on vidzone", APPS) == "vidzone"
    assert engine.site_hint("play 505 on vidzone music.", APPS) == "vidzone music"
    assert engine.site_hint("play 505 from clips.example", APPS) == "clips.example"
    assert engine.site_hint("play 505 on repeat", APPS) == ""  # an ordinary word
    assert engine.site_hint("play 505 on my speaker", APPS) == ""
    assert engine.site_hint("play 505 on Cliptide", APPS) == ""  # an app on this Mac: named app
    assert engine.site_hint("play 505", APPS) == ""


def test_the_site_the_user_named_goes_first_then_the_apps(dictionary):
    c = choose("play_media", "play 505 on vidzone", APPS)
    assert order(c)[:2] == ["web", "Tunewave"]
    assert c[0].site == "vidzone" and "(vidzone)" in c[0].evidence
    plain = choose("play_media", "play 505 on repeat", APPS)
    assert order(plain)[:2] == ["Tunewave", "web"] and plain[1].site == ""


def test_editors_are_not_players_and_neo_never_picks_itself(monkeypatch):
    editor = mk(
        "Mixdesk", "test.mixdesk", category="public.app-category.music",
        media_evidence=("audio/video editor", "category=music"), **HEAVY,
    )
    me = mk("Orbit", "test.orbit", media_evidence=("media app",), **HEAVY)
    monkeypatch.setattr(engine, "_self_ids", lambda: frozenset({"test.orbit"}))
    got = order(choose("play_media", "play a song", [*APPS, editor, me]))
    assert "Mixdesk" not in got and "Orbit" not in got and got[0] == "Tunewave"
    assert order(choose("open_app", "open Orbit", [*APPS, me])) == ["web"]


def test_an_app_that_played_for_neo_before_goes_before_the_web():
    for _ in range(3):
        engine.record("play_media", "test.cliptide", "ax", True)
    c = order(choose("play_media", "play a song", APPS))
    assert c.index("Cliptide") < c.index("web")


def test_the_first_app_scan_runs_off_the_event_loop(monkeypatch):
    seen = []

    def scan():
        seen.append(threading.current_thread() is threading.main_thread())
        return list(APPS)

    monkeypatch.setattr(engine, "_inventory", scan)
    assert order(asyncio.run(engine.choose("play_media", "play a song", None)))[0] == "Tunewave"
    assert seen == [False]


def test_neo_start_scans_the_apps_then_makes_sure_the_zero_shot_checkpoint_is_here(monkeypatch):
    seen = []
    monkeypatch.setattr(engine, "_inventory", lambda: seen.append("scan") or [])
    monkeypatch.setattr(zeroshot, "prefetch", lambda: seen.append("fetch"))
    engine.warm()
    end = time.monotonic() + 2
    while len(seen) < 2 and time.monotonic() < end:
        time.sleep(0.01)
    assert seen == ["scan", "fetch"]


def test_the_privacy_erase_forgets_what_was_learned_about_apps():
    from neo.config import settings

    engine.record("play_media", "test.tunewave", "ax", True)
    (settings().data_dir / "decide.json.corrupt").write_text("{}")
    assert engine.forget() == 1
    assert not list(settings().data_dir.glob("decide.json*"))
    assert engine.prior("play_media", "test.tunewave", "ax") == pytest.approx(0.5)


# ---- outcomes for the reflex's learning ----------------------------------------------------------


@pytest.fixture
def ledger(monkeypatch):
    """A stand-in for neo.reflex.outcomes: what the engine reports."""
    import neo.reflex

    got = []
    mod = types.SimpleNamespace(decided=lambda text, kind, key, ok: got.append((text, kind, key, ok)))
    monkeypatch.setitem(sys.modules, "neo.reflex.outcomes", mod)
    monkeypatch.setattr(neo.reflex, "outcomes", mod, raising=False)
    return got


def test_settled_attempts_of_a_decision_are_reported_once_each(ledger):
    async def request():
        await engine.choose("play_media", "a song", APPS, said="play a song")
        engine.record("play_media", "test.tunewave", "script", False)  # blamed: reported
        engine.record("play_media", "test.tunewave", "ax", False)  # same app, same outcome: once
        engine.record("play_media", "test.cliptide", "ax", False, "permission")  # not its fault
        engine.record("play_media", "web:clips.example", "web", True)  # a per-site sub-arm, not a candidate
        engine.record("play_media", "web", "web", True)

    asyncio.run(request())
    assert ledger == [
        ("play a song", "play_media", "test.tunewave", False),
        ("play a song", "play_media", "web", True),
    ]
    assert engine.prior("play_media", "test.tunewave", "ax") < 0.5  # the store still learned all of it


def test_only_the_users_own_words_are_reported(ledger):
    async def request(text, **said):
        await engine.choose("play_media", text, APPS, **said)
        engine.record("play_media", "web", "web", True)

    # what a decision reads may be the model's query, or Live's arguments while the user's words
    # are still on the way: neither is ever reported
    asyncio.run(request("{'query': 'bohemian rhapsody'}"))
    asyncio.run(request("bohemian rhapsody", said=" "))
    assert ledger == []
    asyncio.run(request("bohemian rhapsody", said="neo play bohemian rhapsody"))
    assert ledger == [("neo play bohemian rhapsody", "play_media", "web", True)]


def test_a_decision_that_failed_leaves_nothing_to_file_a_later_outcome_under(ledger, monkeypatch):
    async def session_task():  # one long-lived task, two requests
        await engine.choose("play_media", "song a", APPS, said="play song a")
        engine.record("play_media", "test.tunewave", "script", True)
        monkeypatch.setattr(engine, "_self_ids", lambda: 1 / 0)
        with pytest.raises(ZeroDivisionError):
            await engine.choose("play_media", "song b", APPS, said="play song b")
        engine.record("play_media", "web", "web", True)  # play_best's own fallback after the failure

    asyncio.run(session_task())
    assert ledger == [("play song a", "play_media", "test.tunewave", True)]


def test_attempts_outside_a_decision_are_not_reported(ledger):
    engine.record("play_media", "test.tunewave", "ax", True)  # no choose() in this task

    async def other_kind():
        await engine.choose("play_media", "play a song", APPS, said="play a song")
        engine.record("send_email", "test.tunewave", "ax", True)

    asyncio.run(other_kind())

    async def elsewhere():  # a decision in one task says nothing about another task's attempts
        await engine.choose("play_media", "play a song", APPS, said="play a song")

    asyncio.run(elsewhere())
    engine.record("play_media", "test.tunewave", "ax", True)
    assert ledger == []


def test_no_ledger_or_a_broken_one_never_breaks_learning(monkeypatch):
    import neo.reflex

    monkeypatch.setitem(sys.modules, "neo.reflex.outcomes", None)  # not there: ImportError
    monkeypatch.delattr(neo.reflex, "outcomes", raising=False)

    async def request():
        await engine.choose("play_media", "play a song", APPS, said="play a song")
        engine.record("play_media", "test.tunewave", "ax", True)

    asyncio.run(request())
    broken = types.SimpleNamespace(decided=lambda *a: 1 / 0)
    monkeypatch.setitem(sys.modules, "neo.reflex.outcomes", broken)
    monkeypatch.setattr(neo.reflex, "outcomes", broken, raising=False)
    asyncio.run(request())
    monkeypatch.setattr(neo.reflex, "outcomes", types.SimpleNamespace(), raising=False)  # no decided()
    asyncio.run(request())
    assert engine.prior("play_media", "test.tunewave", "ax") > 0.5


# ---- every kind chosen by evidence: signals, routes, Laya's pool, preferences ----------------------

_REAL_APP_CAPS = engine._app_caps  # the fixture above stubs it per test
SOCIAL = "public.app-category.social-networking"
UNUSED = {"last_used": None, "use_count": 0, "days_used_30": 0}


def test_the_new_kinds_are_tasks():
    assert {"read_mail", "read_calendar"} <= set(engine.TASKS)
    assert engine.TASKS["send_message"].laya_min_used == 1 and engine.TASKS["send_message"].prefs
    assert engine.TASKS["play_media"].laya_min_used == 2 and not engine.TASKS["play_media"].prefs
    assert engine.TASKS["read_calendar"].wants == ("event.read",) and not engine.TASKS["read_calendar"].proven_fit
    assert {"ambiguous", "declined", "needs_user"} <= engine.NO_BLAME


def test_a_weak_social_signal_needs_use_and_no_call_handler():
    caller = mk("Ringer", "test.ringer", category=SOCIAL, handles=("facetime", "facetime-audio"), **HEAVY)
    chat = mk("Chatwave", "test.chatwave", category=SOCIAL, **HEAVY)
    idle = mk("Idlechat", "test.idle", category=SOCIAL, **UNUSED)
    apps = [*APPS, caller, chat, idle]
    assert order(choose("send_message", "text mom i'm late", apps)) == ["Chatwave"]
    assert engine.fits_kind("send_message", chat) and not engine.fits_kind("send_message", caller)
    engine.record("send_message", "test.idle", "ax", True)  # it did the job before: that counts as use
    assert "Idlechat" in order(choose("send_message", "text mom", apps))


def test_intents_app_intents_and_share_evidence_make_candidates():
    intents = mk("Pigeon", "test.pigeon", intents=("INSendMessageIntent",), **UNUSED)
    mailer = mk("Letterly", "test.letterly", ai_schemas=("protocol:SendMail",), **LIGHT)
    drafts = mk("Drafter", "test.drafter", ai_schemas=("schema:mail.CreateDraftIntent",), **LIGHT)
    sharer = mk("Sharer", "test.sharer", share_text=True, **LIGHT)
    apps = [intents, mailer, drafts, sharer]
    assert order(choose("send_message", "text mom", apps)) == ["Pigeon"]  # declared: no use needed
    assert set(order(choose("send_email", "email sam", apps))) == {"Letterly", "Drafter"}
    assert engine._signal(sharer, "share:text") and not engine._signal(mailer, "share:text")


def test_a_notes_library_is_a_weak_note_candidate_and_a_document_editor_is_not():
    library = mk("Jotter", "test.jotter", editor_types=("net.daringfireball.markdown",),
                 category="public.app-category.productivity", **HEAVY)
    editor = mk("Pageset", "test.pageset", editor_types=("public.plain-text",), doc_based=True, **HEAVY)
    unused = mk("Jotter2", "test.jotter2", editor_types=("public.plain-text",), **UNUSED)
    ide = mk("Codebox", "test.codebox", editor_types=("public.plain-text",), category="public.app-category.developer-tools", **HEAVY)
    got = order(choose("write_note", "note: buy milk", [library, editor, unused, ide, NOTEPADLY]))
    assert set(got) == {"Jotter", "Notepadly"}


def test_the_web_search_tag_counts_only_for_a_browser():
    from neo.decide.sdef import Command, Dictionary

    d = Dictionary(commands={"search the web": Command("search the web", "sfrisrch", "", (("for", "qury", "text"),))},
                   classes={}, properties={}, enums={})
    browser = mk("Browsa2", "test.browsa2", handles=("https",), script=d)
    library = mk("Photobox", "test.photobox", script=d)
    assert "web.search" in _REAL_APP_CAPS(browser) and "web.search" not in _REAL_APP_CAPS(library)


def test_an_app_that_did_the_kind_before_is_a_candidate_without_declared_evidence():
    quiet = mk("Quietnote", "test.quiet", **LIGHT)
    assert "Quietnote" not in order(choose("write_note", "note: x", [quiet]))
    engine.record("write_note", "test.quiet", "ax", True)
    assert order(choose("write_note", "note: x", [quiet])) == ["Quietnote"]
    assert engine.proven("write_note", "test.quiet") and not engine.proven("write_note", "test.other")
    engine.record("read_calendar", "test.quiet", "ax", True)  # read_calendar takes declared evidence only
    assert order(choose("read_calendar", "what's on today", [quiet])) == []


# ---- routes ------------------------------------------------------------------------------------------

DATEBOOK = mk("Datebook", "test.datebook", **LIGHT)
CAPS["test.datebook"] = frozenset({"event.make"})
EVENTS = engine.Task("put a meeting on my calendar", wants=("event.make",), routes=("eventkit.events",),
                     fallback="ics", laya_min_used=1, proven_fit=True, prefs=True)


def store(**kw):
    from neo.decide.inventory import App

    return App(bundle_id="route:eventkit.events", name="your Mac's calendar store", path="", kind="route",
               note="your Mac's calendar store (every calendar app here shows it): 2 writable calendars", **kw)


@pytest.fixture
def events(monkeypatch):
    monkeypatch.setitem(engine.TASKS, "add_event", EVENTS)
    got: list = []
    monkeypatch.setattr(engine, "_routes", lambda kind, task, exclude: [r for r in got if r[0].bundle_id not in exclude])
    return got


def test_a_store_not_read_yet_counts_as_much_as_the_most_used_app(events):
    events.append((store(), True))
    c = choose("add_event", "lunch friday at 1", [DATEBOOK, *APPS])
    assert order(c) == ["your Mac's calendar store", "Datebook", "ics"]
    route, app = c[0], c[1]
    assert route.methods == ("eventkit",) and route.key == "route:eventkit.events"
    assert route.score == pytest.approx(5 / 6 * (engine.AFF_FLOOR + (1 - engine.AFF_FLOOR) * engine._affinity(DATEBOOK, NOW)), rel=1e-3)
    assert app.score < route.score and "2 writable calendars" in route.evidence


def test_a_store_with_access_counts_the_users_own_use(events):
    events.append((store(days_used_30=0, last_used=None), False))  # granted, but empty
    assert order(choose("add_event", "lunch", [DATEBOOK]))[:2] == ["Datebook", "ics"]  # unused store: the tail
    events[0] = (store(days_used_30=12, last_used=NOW - 3600), False)
    assert order(choose("add_event", "lunch", [DATEBOOK]))[0] == "your Mac's calendar store"


def test_apps_that_write_into_the_store_step_aside_unless_named(events):
    from neo.decide import learned

    events.append((store(), True))
    learned.put("store_views", "add_event:test.datebook", {"seen": True})
    assert order(choose("add_event", "lunch", [DATEBOOK])) == ["your Mac's calendar store", "ics"]
    assert order(choose("add_event", "lunch in Datebook", [DATEBOOK]))[0] == "Datebook"
    events.clear()  # the store can't be used: the app is back
    assert order(choose("add_event", "lunch", [DATEBOOK])) == ["Datebook", "ics"]


def test_excluded_candidates_stay_out(events):
    events.append((store(), True))
    got = order(asyncio.run(engine.choose("add_event", "lunch", [DATEBOOK], exclude=frozenset({"route:eventkit.events", "ics"}))))
    assert got == ["Datebook"]


def test_routes_are_asked_off_the_loop_and_a_broken_one_is_no_candidate(monkeypatch):
    from neo.decide import routes

    seen = []

    def route_app(kind, route):
        seen.append(threading.current_thread() is threading.main_thread())
        if route == "eventkit.reminders":
            raise RuntimeError("store fell over")
        return store()

    monkeypatch.setattr(routes, "route_app", route_app)
    monkeypatch.setattr(routes, "pending", lambda app: True)
    task = dataclasses.replace(EVENTS, routes=("eventkit.events", "eventkit.reminders"))
    monkeypatch.setitem(engine.TASKS, "add_event", task)
    assert order(choose("add_event", "lunch", [DATEBOOK]))[0] == "your Mac's calendar store"
    assert seen == [False, False]
    assert engine._routes("add_event", task, frozenset({"route:eventkit.events"})) == []


def test_fits_kind_never_takes_a_route():
    assert not engine.fits_kind("add_event", store()) and not engine.fits_kind("no_such_kind", DATEBOOK)


# ---- Laya's pool ---------------------------------------------------------------------------------------


def test_laya_weighs_a_capable_unused_app_when_one_used_one_fits(monkeypatch):
    used = mk("Chatwave", "test.chatwave", category=SOCIAL, **HEAVY)
    unused = mk("Pigeon", "test.pigeon", intents=("INSendMessageIntent",), **UNUSED)
    fake = use_laya(monkeypatch, prefer("Pigeon", 0.9))
    c = choose("send_message", "message john on the pigeon thing", [used, unused])
    assert order(c) == ["Pigeon", "Chatwave"] and c[0].laya_p == pytest.approx(0.9)
    assert set(fake.calls[-1][1]["app"]["criteria"]) == {"Chatwave", "Pigeon"}
    use_laya(monkeypatch, lambda s, labels: {k: 0.5 for k in labels})  # unsure: the prior, unused last
    assert order(choose("send_message", "message john", [used, unused])) == ["Chatwave", "Pigeon"]
    fake = use_laya(monkeypatch, prefer("Podnest", 0.9))  # play_media still needs two used apps
    choose("play_media", "play a podcast", [TUNEWAVE, PODNEST])
    assert not fake.calls


# ---- preferences -----------------------------------------------------------------------------------------

A = mk("Chatwave", "test.chatwave", category=SOCIAL, **HEAVY)
B = mk("Talkie", "test.talkie", category=SOCIAL, **HEAVY)


def test_preferences_are_neutral_until_the_user_says_otherwise():
    assert engine.pref_factor("send_message", "test.chatwave") == 1.0
    assert engine.pref_factor("send_message", "test.chatwave", "h1") == 1.0
    before = [c.score for c in choose("send_message", "text mom", [A, B])]
    engine.record("send_message", "test.chatwave", "ax", True)  # reliability moves the prior...
    engine.record("send_message", "test.chatwave", "ax", False, "timeout")
    assert engine.pref_factor("send_message", "test.chatwave") == 1.0  # ...never the preference
    assert before[0] == pytest.approx(before[1])


def test_a_correction_moves_preference_not_reliability():
    engine.record("send_message", "test.chatwave", "ax", True)
    arm = engine.prior("send_message", "test.chatwave", "ax")
    assert order(choose("send_message", "text mom", [A, B]))[0] == "Chatwave"
    engine.correct("send_message", "test.chatwave")
    assert engine.prior("send_message", "test.chatwave", "ax") == arm
    c = choose("send_message", "text mom", [A, B])
    assert order(c)[0] == "Talkie" and "you corrected this once" in c[1].evidence


def test_a_per_person_choice_reorders_for_that_person_only():
    engine.record_pref("send_message", "test.talkie", True, about="h-priya")
    c = asyncio.run(engine.choose("send_message", "text priya", [A, B], about="h-priya"))
    assert order(c)[0] == "Talkie" and "used for this person 1 of 1 times" in c[0].evidence
    other = asyncio.run(engine.choose("send_message", "text sam", [A, B], about="h-sam"))
    assert other[0].score == pytest.approx(other[1].score)
    notes = choose("play_media", "play", [TUNEWAVE])  # a kind without preferences never reads them
    assert "person" not in notes[0].evidence


def test_version_1_files_load_and_are_saved_as_version_2():
    from neo.config import settings

    p = settings().data_dir / "decide.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    arm = {"a": 5, "b": 1, "t": time.time()}
    p.write_text(json.dumps({"version": 1, "arms": {"send_message": {"test.chatwave": {"ax": arm}}}}))
    assert engine.prior("send_message", "test.chatwave", "ax") == pytest.approx(5 / 6)
    assert engine.pref_factor("send_message", "test.chatwave") == 1.0
    engine.correct("send_message", "test.chatwave")
    data = json.loads(p.read_text())
    assert data["version"] == 2 and data["prefs"]["send_message"]["test.chatwave"]["down"] == 1
    assert data["arms"]["send_message"]["test.chatwave"]["ax"]["a"] == 5
    p.write_text(json.dumps({"arms": {}, "prefs": {"k": {"x": {"a": "bad"}}, "j": []}}))
    engine._cache.clear()
    assert engine.pref_factor("k", "x") == 1.0


def test_habitual_is_where_this_kind_last_went(monkeypatch):
    t = iter([100.0, 200.0, 300.0])
    monkeypatch.setattr(engine, "_now", lambda: next(t))
    assert not engine.habitual("write_note", "test.a")
    engine.record("write_note", "test.a", "script_make", True)
    engine.record("write_note", "test.b", "ax_new", True)
    assert engine.habitual("write_note", "test.b") and not engine.habitual("write_note", "test.a")
    engine.record("write_note", "test.b", "ax_new", False, "timeout")
    assert engine.habitual("write_note", "test.b")  # a later failure doesn't move where it last went


def test_a_wrong_route_is_demoted_at_once():
    from neo.config import settings

    engine.record("write_note", "test.a", "script_make", False, "route_bad")
    arm = json.loads((settings().data_dir / "decide.json").read_text())["arms"]["write_note"]["test.a"]["script_make"]
    assert arm["fails"] == engine.DEMOTE_AFTER and arm["n"] == 1


# ---- the kind, with app names masked -------------------------------------------------------------------


def test_decide_kind_reads_the_request_with_app_names_masked(monkeypatch, zs_asked):
    chat = mk("Chatwhiz", "test.chatwhiz", **HEAVY)
    reflex = use_reflex(monkeypatch, "open_app", 0.9)
    for text in ("open Chatwhiz", "chatwhiz kholo", "please switch to chatwhiz"):
        assert asyncio.run(engine.decide_kind(text, apps=[chat])) == ("open_app", 1.0, False), text
    assert reflex.texts == [] and not zs_asked  # no model needed
    reflex = use_reflex(monkeypatch, "send_message", 0.8)
    assert asyncio.run(engine.decide_kind("message john on Chatwhiz", apps=[chat])).kind == "send_message"
    assert reflex.texts == ["message john on an app"]
    assert engine.mask_apps("open chatwhiz and chatwhiz", [chat]) == "open an app and an app"
