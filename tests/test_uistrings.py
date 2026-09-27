"""neo.decide.uistrings: the OS's own English↔Hindi UI strings, on synthetic string tables in a temp
folder (the real /System is never read here). The lazy reader decodes what a plain plist load would;
the index is built once per OS build and cached under the (per-test) data dir; library mode never
scans; an app's own strings stay in RAM; the gate hook is registered on import."""

from __future__ import annotations

import asyncio
import os
import plistlib
from pathlib import Path

import pytest

from neo import runtime
from neo.decide import uistrings as U
from neo.security import policy


@pytest.fixture(autouse=True)
def _clean_index():
    before, apps, empty = U._system, dict(U._apps), dict(U._no_tables)
    U.install(None)
    U._apps.clear()
    U._no_tables.clear()
    yield
    U.install(before)
    U._apps.clear()
    U._apps.update(apps)
    U._no_tables.clear()
    U._no_tables.update(empty)


def _table(path: Path, body: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        plistlib.dump(body, fh, fmt=plistlib.FMT_BINARY)
    return path


def _os_tables(root: Path) -> Path:
    fw = root / "Frameworks" / "Kit.framework" / "Versions" / "A" / "Resources"
    _table(
        fw / "Menus.loctable",
        {
            "en": {"send": "Send", "new": "New Game", "ok": "OK", "cancel": "Cancel", "fmt": "%@ items",
                   "undo": "This action cannot be undone.", "same": "Wi-Fi", "plural": {"NSStringLocalizedFormatKey": "%d"}},
            "hi": {"send": "भेजें", "new": "नया गेम", "ok": "ठीक", "cancel": "रद्द करें", "fmt": "%@ आइटम",
                   "undo": "इस क्रिया को पहले जैसा नहीं किया जा सकता है।", "same": "Wi-Fi", "plural": {"x": "y"}},
            "fr": {"send": "Envoyer", "new": "Nouvelle partie"},
            "LocProvenance": {"en": "x"},
        },
    )  # fmt: skip
    _table(root / "Apps" / "Other.app" / "Contents" / "Resources" / "L.loctable", {"English": {"k": "Start"}, "hi": {"k": "शुरू करें"}})
    _table(root / "Apps" / "NoHindi.loctable", {"en": {"k": "Zoom In"}, "de": {"k": "Vergrößern"}})
    (root / "Apps" / "Broken.loctable").write_bytes(b"bplist00 not really")
    (root / "Frameworks" / "Kit.framework" / "Versions" / "Current").symlink_to("A")  # the same files again
    return root


# ---- the reader ----------------------------------------------------------------------------------------


def test_the_lazy_reader_decodes_what_a_plist_load_does(tmp_path):
    body = {lang: {f"k{i}": f"{lang} {'हिन्दी' if lang == 'hi' else 'text'} {i}" for i in range(300)} for lang in ("en", "hi", "ja", "de")}
    body["en"]["nested"] = {"a": "b"}
    p = _table(tmp_path / "t.loctable", body)
    en, loc = U._table_langs(str(p), ("hi",))
    assert en == {k: v for k, v in body["en"].items() if isinstance(v, str)}
    assert loc == {"hi": body["hi"]}


def test_build_pairs_english_with_hindi_once_per_file(tmp_path):
    pairs = U.build([str(_os_tables(tmp_path))])
    assert pairs["Send"] == {"भेजें"} and pairs["New Game"] == {"नया गेम"} and pairs["Start"] == {"शुरू करें"}
    assert "%@ items" not in pairs and "Wi-Fi" not in pairs and "Zoom In" not in pairs  # formats, same text, no Hindi
    assert "This action cannot be undone." in pairs


def test_bench_agrees_with_a_plain_plist_load(tmp_path):
    got = U.bench(tables=4, keys=50, langs=12, root=str(tmp_path))
    assert got["same"] == 1.0


# ---- words and lookups -------------------------------------------------------------------------------


def test_norm_folds_spelling_variants():
    assert U.norm("  Save As…  ") == "save as"
    assert U.norm("ख़रीदें") == U.norm("खरीदें")  # nukta
    assert U.norm("हटाएँ") == U.norm("हटाएं")  # chandrabindu / anusvara
    assert U.norm("Don’t Save") == "don't save"


def test_lookups_both_ways(tmp_path):
    U.install(U.Index.of(U.build([str(_os_tables(tmp_path))]), "test"))
    assert U.ready()
    assert U.english_sources("भेजें") == ("Send",)
    assert U.english_sources("Send") == ()  # Latin script: English already
    assert U.localized("new game") == ("नया गेम",)
    assert U.english_sources("अनजान") == ()
    # a dialog's text: whole, else sentence by sentence
    text = "क्या आप निश्चित हैं? इस क्रिया को पहले जैसा नहीं किया जा सकता है।"
    assert U.english_sources(text) == ("This action cannot be undone.",)


def test_readable_fails_closed_without_the_index(tmp_path):
    assert U.non_latin("नया गेम") and not U.non_latin("Café au lait") and not U.non_latin("New Game")
    assert U.readable("New Game") and U.readable("Café")
    assert not U.readable("नया गेम")  # no index yet: NEO can't say what it is
    U.install(U.Index.of(U.build([str(_os_tables(tmp_path))]), "test"))
    assert U.readable("नया गेम") and not U.readable("अनजान बटन")


def test_os_match_both_directions(tmp_path):
    U.install(U.Index.of(U.build([str(_os_tables(tmp_path))]), "test"))
    assert U.os_match("नया गेम शुरू करो", "New Game")  # a Hindi request, an English menu
    assert U.os_match("new game karo", "नया गेम")  # an English request, a Hindi menu
    assert not U.os_match("नया गेम शुरू करो", "Cancel")
    assert not U.os_match("new game", "New Game")  # same script: the lexical stage's job


def test_the_gate_reads_labels_through_the_index(tmp_path):
    assert policy._back_translate is U.english_sources  # registered on import
    assert not policy.commits("भेजें")  # no index: nothing to read it with
    U.install(U.Index.of(U.build([str(_os_tables(tmp_path))]), "test"))
    assert policy.commits("भेजें") and not policy.commits("नया गेम")
    assert policy.commits("ठीक", "इस क्रिया को पहले जैसा नहीं किया जा सकता है।")


# ---- an app's own strings ----------------------------------------------------------------------------


def test_an_apps_own_strings_are_read_once_and_kept_in_ram_only(tmp_path, monkeypatch):
    app = tmp_path / "Some.app"
    res = app / "Contents" / "Resources"
    _table(res / "Localizable.loctable", {"en": {"n": "New Round"}, "hi": {"n": "नया राउंड"}})
    (res / "en.lproj").mkdir(parents=True)
    (res / "hi.lproj").mkdir(parents=True)
    (res / "en.lproj" / "Main.strings").write_text('/* c */\n"deal" = "Deal Again";\n"q" = "Say \\"hi\\"";\n', encoding="utf-16")
    (res / "hi.lproj" / "Main.strings").write_text('"deal" = "फिर से बाँटें";\n"q" = "\\"नमस्ते\\" कहें";\n', encoding="utf-8")
    data_dir = Path(os.environ["NEO_DATA_DIR"])
    before = sorted(data_dir.rglob("*")) if data_dir.exists() else []
    ix = U.for_bundle(str(app))
    assert ix is not None and U.english_sources("नया राउंड") == ("New Round",)
    assert U.english_sources("फिर से बाँटें") == ("Deal Again",) and U.localized('Say "hi"') == ('"नमस्ते" कहें',)
    assert U.for_bundle(str(app)) is ix  # read once
    assert (sorted(data_dir.rglob("*")) if data_dir.exists() else []) == before  # nothing written down
    for i in range(U._APPS_KEPT + 2):
        U.for_bundle(str(tmp_path / f"None{i}.app"))
    assert len(U._apps) == U._APPS_KEPT


# ---- once per OS build, at idle, never in library mode --------------------------------------------------


def test_built_once_per_os_build_then_loaded_from_the_cache(tmp_path, monkeypatch):
    root = _os_tables(tmp_path / "os")
    monkeypatch.setattr(U, "ROOTS", (str(root),))
    monkeypatch.setattr(U, "os_build", lambda: "26.0-TEST1")
    assert U.load_or_build()
    cache = U.cache_path("26.0-TEST1")
    assert cache.exists() and U.english_sources("भेजें") == ("Send",)
    U.install(None)

    def no_scan(*a, **k):
        raise AssertionError("scanned although the cache was there")

    real_build = U.build
    monkeypatch.setattr(U, "build", no_scan)
    assert U.load_or_build() and U.english_sources("भेजें") == ("Send",)
    # an OS update: a new build rebuilds, and the old cache goes
    monkeypatch.setattr(U, "build", real_build)
    monkeypatch.setattr(U, "os_build", lambda: "26.1-TEST2")
    U.install(None)
    assert U.load_or_build()
    assert U.cache_path("26.1-TEST2").exists() and not cache.exists()


def test_a_scan_waits_while_neo_is_busy_and_stops_when_told(tmp_path):
    import threading

    root = str(_os_tables(tmp_path))
    stop = threading.Event()
    busy_calls = []

    def busy():
        busy_calls.append(1)
        if len(busy_calls) > 2:
            stop.set()
        return True

    with pytest.raises(U.Stopped):
        U.build([root], stop=stop, busy=busy)
    assert len(busy_calls) >= 2


def test_library_mode_never_scans(monkeypatch):
    monkeypatch.setattr(runtime, "_library", True)
    monkeypatch.setattr(U, "build", lambda *a, **k: pytest.fail("a scan in library mode"))
    assert not U.may_scan()
    assert U.load_or_build() is False
    asyncio.run(asyncio.wait_for(U.idle_loop(lambda: True, idle_s=0, poll_s=0), 1))  # returns at once
    assert not U.ready()


async def test_the_idle_loop_builds_once_neo_is_idle(monkeypatch):
    idle = {"now": False}
    built = []

    def fake_build(**kw):
        built.append(kw)
        U.install(U.Index.of({"Send": ["भेजें"]}, "t"))
        return True

    monkeypatch.setattr(U, "load_or_build", fake_build)

    async def flip():
        await asyncio.sleep(0.05)
        assert not built  # not while NEO is busy
        idle["now"] = True

    await asyncio.wait_for(asyncio.gather(U.idle_loop(lambda: idle["now"], idle_s=0, poll_s=0.01), flip()), 2)
    assert len(built) == 1 and U.ready() and callable(built[0]["busy"])


# ---- review fixes ------------------------------------------------------------------------------------------


def test_readable_needs_every_sentence_read(tmp_path):
    """One sentence the OS knows doesn't make a sheet readable: the next one might say what's lost."""
    U.install(U.Index.of({"Choose the players.": ["खिलाड़ी चुनें।"], "New Game": ["नया गेम"]}, "t"))
    assert U.readable("खिलाड़ी चुनें।") and U.readable("नया गेम")
    assert not U.readable("खिलाड़ी चुनें। आपकी सारी प्रगति मिट जाएगी।")  # the second: no English
    assert not U.readable("नया गेम खिलाड़ी चुनें।")  # two strings joined with a space: no one string


def test_an_apps_tables_are_found_without_walking_its_whole_bundle(tmp_path, monkeypatch, capsys):
    """Hermetic: a bundle whose Resources hold thousands of files in deep folders (a big office
    app's) and no Hindi at all. The first contact reads only near the top, within a bound; an app
    with no tables is remembered, so the next contact costs nothing (after the RAM cache let it go)."""
    import time

    app = tmp_path / "Big.app"
    res = app / "Contents" / "Resources"
    _table(res / "Localizable.loctable", {"en": {"n": "New Round"}, "hi": {"n": "नया राउंड"}})
    for d in range(60):
        deep = res / f"pack{d}" / "a" / "b" / "c"
        deep.mkdir(parents=True)
        for f in range(40):
            (deep / f"f{f}.dat").write_bytes(b"x")
    _table(res / "pack0" / "a" / "b" / "c" / "Deep.loctable", {"en": {"d": "Deep"}, "hi": {"d": "गहरा"}})
    seen = []
    real_walk = os.walk

    def walk(top, *a, **k):
        for d, dirs, files in real_walk(top, *a, **k):
            seen.append(len(files) + len(dirs))
            yield d, dirs, files

    monkeypatch.setattr(U.os, "walk", walk)
    t0 = time.perf_counter()
    ix = U.for_bundle(str(app))
    first_ms = (time.perf_counter() - t0) * 1000
    assert ix is not None and U.english_sources("नया राउंड") == ("New Round",)
    assert U.english_sources("गहरा") == ()  # too deep to be the app's own UI tables: never walked to
    assert sum(seen) <= U._APP_MAX_ENTRIES + 64, sum(seen)
    empty = tmp_path / "NoTables.app" / "Contents" / "Resources"
    for d in range(3):
        (empty / f"d{d}").mkdir(parents=True)
    assert len(U.for_bundle(str(tmp_path / "NoTables.app"))) == 0
    U._apps.clear()  # the RAM cache let it go (other apps used since)
    monkeypatch.setattr(U, "_bundle_pairs", lambda *a, **k: pytest.fail("walked again"))
    assert len(U.for_bundle(str(tmp_path / "NoTables.app"))) == 0
    with capsys.disabled():
        print(f"\n[bench] first contact with a 2,400-file bundle: {first_ms:.1f} ms, {sum(seen)} entries looked at")
