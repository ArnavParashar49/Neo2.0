"""The decision engine: which installed app should do a request, Laya deciding, outcomes learned.

Laya is faked (a stub whose system_one returns probabilities); apps are built directly. Nothing
here launches, scripts or inspects a real app.
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
    engine._cache.clear()
    yield
    engine._cache.clear()


def use_laya(monkeypatch, pick) -> FakeLaya:
    import neo.reflex

    fake = FakeLaya(pick)
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
    assert state["request"] == "play the trailer"
    assert {x["label"] for x in state["candidates"]} == {"Tunewave", "Cliptide"}
    assert all({"label", "evidence", "affinity", "record"} <= set(x) for x in state["candidates"])
    crit = q["app"]["criteria"]
    assert set(crit) == {"Tunewave", "Cliptide"}  # only used, fitting apps; no named/unused/web
    # each option is described from evidence: category, what it can do, usage, NEO's record
    assert "music" in crit["Tunewave"] and "search play" in crit["Tunewave"]
    assert "used 28 of last 30 days" in crit["Tunewave"] and "not tried yet" in crit["Tunewave"]
    assert "video" in crit["Cliptide"]


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

    monkeypatch.setattr(neo.reflex, "get_laya", lambda: Broken())
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

    fake = FakeLaya(prefer("Cliptide", 0.99))

    def get_laya_taking_the_lock():
        with neo.reflex.laya_lock():
            return fake

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
        dt = time.perf_counter() - t
    finally:
        done.set()
        holder.join()
    assert dt < 1.0
    assert order(c)[:3] == ["Tunewave", "web", "Cliptide"] and not fake.calls
    assert order(choose("play_media", "play something", APPS))[0] == "Cliptide"


def test_an_answer_that_came_too_late_is_never_computed(monkeypatch):
    import neo.reflex

    fake = FakeLaya(prefer("play_media", 0.9))
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

    assert asyncio.run(run()) == ("other", 0.0)
    assert not fake.calls


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

    def embed(texts):  # "video" texts point one way, everything else the other
        return np.array([[1.0, 0.0] if "video" in t or "film" in t else [0.0, 1.0] for t in texts])

    monkeypatch.setattr(engine, "_embed_fn", lambda agent: embed)
    fake = use_laya(monkeypatch, lambda s, labels: {k: (0.9 if k == "Reelio" else 0.1 / (len(labels) - 1)) for k in labels})
    c = asyncio.run(engine.choose("play_media", "play a film", apps))
    assert order(c)[:2] == ["Reelio", "web"]  # Laya's sure pick, then the web: the rest were opened once
    assert len(fake.calls[-1][1]["app"]["criteria"]) <= engine.LAYA_POOL
    assert len(c) == engine.MAX_CANDIDATES


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


def test_decide_kind_asks_laya_over_the_task_descriptions(monkeypatch):
    fake = use_laya(monkeypatch, prefer("play_media", 0.9))
    kind, p = asyncio.run(engine.decide_kind("put on some jazz"))
    assert (kind, p) == ("play_media", pytest.approx(0.9))
    state, q = fake.calls[-1]
    assert state == {"text": "put on some jazz"}
    assert set(q["kind"]["criteria"]) == set(engine.TASKS)
    assert {"play_media", "send_message", "write_note", "add_reminder", "add_event", "send_email",
            "open_app", "web_search", "other"} <= set(engine.TASKS)


def test_decide_kind_without_laya_is_other():
    assert asyncio.run(engine.decide_kind("put on some jazz")) == ("other", 0.0)
    assert asyncio.run(engine.decide_kind("  ")) == ("other", 0.0)


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
    assert r == ("other", 0.0) and dt < 1.0


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


def test_the_privacy_erase_forgets_what_was_learned_about_apps():
    from neo.config import settings

    engine.record("play_media", "test.tunewave", "ax", True)
    (settings().data_dir / "decide.json.corrupt").write_text("{}")
    assert engine.forget() == 1
    assert not list(settings().data_dir.glob("decide.json*"))
    assert engine.prior("play_media", "test.tunewave", "ax") == pytest.approx(0.5)
