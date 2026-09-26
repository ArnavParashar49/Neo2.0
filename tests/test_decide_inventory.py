"""neo.decide.inventory + neo.decide.sdef: finding the apps on a Mac, what their scripting
dictionaries let NEO do, which app the user named, and how much the user uses each one.

Everything runs on fixtures: fake app bundles in a temp dir (copied from tests/fixtures/decide),
and fake Spotlight, LaunchServices, process-table and Dock readers. Nothing on this Mac is
launched, opened or sent an Apple event, and the real app folders are never read.
"""

from __future__ import annotations

import ast
import os
import plistlib
import re
import shutil
import time
import urllib.parse
from collections import Counter
from pathlib import Path

import pytest

from neo.decide import inventory as inv
from neo.decide import sdef

FIX = Path(__file__).parent / "fixtures" / "decide"
DAY = 86400.0
NO_USE = (None, 0, 0, "", "")


# ------------------------------------------------------------------------------ helpers
def bundle(root: Path, name: str, rename: str | None = None) -> Path:
    """Materialise fixture app ``name`` as ``root/<rename or name>.app``; its scripting dictionary
    (and the standard suite it includes) goes into Contents/Resources."""
    dst = root / f"{rename or name}.app"
    shutil.copytree(FIX / "apps" / name, dst)
    plist = dst / "Contents" / "Info.plist"
    if plist.exists():
        osa = plistlib.loads(plist.read_bytes()).get("OSAScriptingDefinition")
        if osa:
            res = dst / "Contents" / "Resources"
            res.mkdir(exist_ok=True)
            fname = osa if osa.endswith(".sdef") else f"{osa}.sdef"
            shutil.copy(FIX / "sdef" / f"{osa.removesuffix('.sdef').lower()}.sdef", res / fname)
            shutil.copy(FIX / "sdef" / "standard.sdef", res / "standard.sdef")
    return dst


def make_app(root: Path, name: str, **info) -> Path:
    dst = root / f"{name}.app"
    (dst / "Contents").mkdir(parents=True)
    (dst / "Contents" / "Info.plist").write_bytes(plistlib.dumps(info))
    return dst


def load_fixture_sdef(tmp_path: Path, name: str) -> sdef.Dictionary | None:
    d = tmp_path / f"sdef-{name}"
    d.mkdir()
    shutil.copy(FIX / "sdef" / "standard.sdef", d / "standard.sdef")
    shutil.copy(FIX / "sdef" / f"{name}.sdef", d / f"{name}.sdef")
    return sdef.load_file(str(d / f"{name}.sdef"))


class FakeMac:
    """The OS readers inventory uses, replaced by plain data."""

    def __init__(self, tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tmp = tmp
        self.apps_dir = tmp / "Applications"
        self.user_dir = tmp / "UserApps"
        self.services = tmp / "CoreServices"
        for d in (self.apps_dir, self.user_dir, self.services):
            d.mkdir()
        self.spotlight: list[str] = []  # extra bundle paths Spotlight "finds"
        self.usage: dict[
            str, tuple
        ] = {}  # bundle stem -> (last_used, count, days30, store category, display name)
        self.running: set[str] = set()  # bundle paths
        self.dock: set[str] = set()  # bundle ids
        self.defaults: dict[str, str] = {}  # scheme -> bundle path
        self.handlers: dict[str, str] = {}  # url / uti -> bundle path (default_for)
        self.by_id: dict[str, str] = {}  # bundle id -> path LaunchServices knows
        self.calls: Counter = Counter()
        self.usage_reads: list[list[str]] = []
        m = monkeypatch
        m.setattr(inv, "APP_DIRS", (str(self.apps_dir), str(self.user_dir)))
        m.setattr(inv, "SERVICES_DIR", str(self.services))
        m.setattr(inv, "_spotlight_app_paths", self._spot)
        m.setattr(inv, "_launchservices_app_paths", lambda: [])
        m.setattr(inv, "_spotlight_usage", self._usage)
        m.setattr(inv, "_running_reals", lambda: {os.path.realpath(p) for p in self.running})
        m.setattr(inv, "_dock_ids", lambda: set(self.dock))
        m.setattr(inv, "_scheme_defaults", lambda: {k: os.path.realpath(v) for k, v in self.defaults.items()})
        m.setattr(inv, "_ls_default_path", lambda s: self.handlers.get(s))
        m.setattr(inv, "_ls_path_for_id", lambda bid: self.by_id.get(bid))
        use_fixture_words(m)
        inv._reset()

    def add(self, name: str, where: Path | None = None, rename: str | None = None) -> Path:
        return bundle(where or self.apps_dir, name, rename)

    def _spot(self) -> list[str]:
        self.calls["spotlight"] += 1
        return list(self.spotlight)

    def _usage(self, paths: list[str]) -> dict:
        self.usage_reads.append(sorted(Path(p).stem for p in paths))
        return {p: self.usage.get(Path(p).stem, NO_USE) for p in paths}

    def app(self, name: str) -> inv.App:
        hits = [a for a in inv.inventory() if a.name == name]
        assert len(hits) == 1, [a.name for a in inv.inventory()]
        return hits[0]


def use_fixture_words(m: pytest.MonkeyPatch, words: Path | None = FIX / "words.txt") -> None:
    m.setattr(inv, "_WORDS_FILE", str(words))
    m.setattr(inv, "_CONNECTIVES_FILE", str(FIX / "connectives.txt"))
    m.setattr(inv, "_words_known", {})
    m.setattr(inv, "_connectives", None)


@pytest.fixture
def mac(tmp_path, monkeypatch):
    fake = FakeMac(tmp_path, monkeypatch)
    yield fake
    inv._reset()


# ------------------------------------------------------------------------------ enumeration
def test_enumeration_keeps_real_apps_and_drops_helpers_agents_and_junk(mac):
    for n in ("Spotify", "Music", "WhatsApp", "VLC-like", "SaavnLike", "Menubar Tool", "Legacy Scripted"):
        mac.add(n)
    mac.add("Daemon")  # LSBackgroundOnly
    mac.add("NoId")  # no bundle id
    mac.add("Weather", rename=".Hidden")  # hidden
    (mac.apps_dir / "Suite").mkdir()
    mac.add("Writer", where=mac.apps_dir / "Suite")  # one level of sub-folder
    # a helper app nested inside another app, and bundles in caches/node_modules, which Spotlight also reports
    helper = make_app(
        mac.apps_dir / "Spotify.app" / "Contents" / "Frameworks",
        "Helper",
        CFBundleIdentifier="com.example.helper",
    )
    (mac.tmp / "Library" / "Caches").mkdir(parents=True)
    cached = make_app(mac.tmp / "Library" / "Caches", "Cached", CFBundleIdentifier="com.example.cached")
    (mac.tmp / "proj" / "node_modules").mkdir(parents=True)
    nm = make_app(mac.tmp / "proj" / "node_modules", "Tool", CFBundleIdentifier="com.example.nm")
    # a dev build elsewhere that Spotlight knows about is a real app
    (mac.tmp / "dist").mkdir()
    dev = bundle(mac.tmp / "dist", "Streamer")
    # the same bundle id twice: the app-folder copy wins; a symlink to an app is the same app
    (mac.tmp / "Downloads").mkdir()
    mac.add("Notes")
    stray_notes = bundle(mac.tmp / "Downloads", "Notes")
    os.symlink(mac.apps_dir / "Music.app", mac.user_dir / "Music Link.app")
    # OS service apps: only those that declare an app category are user-facing
    make_app(
        mac.services,
        "Finderish",
        CFBundleIdentifier="com.example.finderish",
        LSApplicationCategoryType="public.app-category.productivity",
    )
    make_app(mac.services, "Agentish", CFBundleIdentifier="com.example.agentish")
    mac.spotlight = [str(helper), str(cached), str(nm), str(dev), str(stray_notes)]

    apps = inv.inventory()
    names = sorted(a.name for a in apps)
    assert names == sorted(
        [
            "Spotify",
            "Music",
            "WhatsApp",
            "VLC-like",
            "SaavnLike",
            "Menubar Tool",
            "Legacy Scripted",
            "Writer",
            "Streamer",
            "Notes",
            "Finderish",
        ]
    )
    by_id = {a.bundle_id: a for a in apps}
    assert by_id["com.acme.Notes"].path == str(mac.apps_dir / "Notes.app")
    assert by_id["com.acme.Music"].path == str(mac.apps_dir / "Music.app")
    assert by_id["com.example-streamer.desktop"].path == str(dev)
    assert by_id["com.example.saavnlike"].path.endswith("SaavnLike.app")  # iPhone/iPad app wrapper
    assert by_id["com.example.legacy"].script is None  # scriptable flag but no dictionary: none
    assert [a.name for a in apps] == sorted((a.name for a in apps), key=str.casefold)


def test_bundle_fields_and_media_evidence(mac):
    for n in (
        "Spotify",
        "Music",
        "VLC-like",
        "Audacity-like",
        "GarageBand-like",
        "IINA-like",
        "Streamer",
        "Some Site",
        "Acme Browser",
        "SaavnLike",
        "Writer",
        "Mail",
        "Notes",
    ):
        mac.add(n)
    yt = mac.add("YouTube Music")
    (yt / "Contents" / "Frameworks" / "Electron Framework.framework").mkdir(parents=True)
    mac.usage["SaavnLike"] = (None, 0, 0, "public.app-category.music", "")
    mac.defaults = {
        "https": str(mac.apps_dir / "Acme Browser.app"),
        "http": str(mac.apps_dir / "Acme Browser.app"),
        "mailto": str(mac.apps_dir / "Mail.app"),
    }

    spotify = mac.app("Spotify")
    assert spotify.category == "music"
    assert spotify.schemes == ("spotify",) and spotify.own_schemes == ("spotify",)
    assert spotify.media_evidence[0] == "media app"
    assert {"category=music", "script: player state", "media-library-access", "script: play by URI"} <= set(
        spotify.media_evidence
    )
    assert {"media.play_uri", "media.item_uri", "media.state"} <= spotify.caps

    music = mac.app("Music")
    assert music.own_schemes == ("music", "musics")  # 'itms' is not spelled like the app
    assert "public.mp3" in music.editor_types and "public.mpeg-4-audio" in music.viewer_types
    assert "media.search_play" in music.caps

    vlc = mac.app("VLC-like")
    assert vlc.media_evidence[0] == "media file player"
    assert vlc.own_schemes == () and "https" in vlc.schemes  # claims web schemes but opens no HTML
    assert "public.mp3" in vlc.viewer_types and vlc.category == "video"
    assert mac.app("IINA-like").media_evidence[0] == "media file player"
    assert mac.app("IINA-like").own_schemes == ("iinalike",)
    assert mac.app("Audacity-like").media_evidence[0] == "audio/video editor"
    assert "mp3" in mac.app("Audacity-like").editor_types
    assert mac.app("GarageBand-like").media_evidence[0] == "audio/video editor"
    assert mac.app("Streamer").media_evidence[:2] == ("media app", "category=music")
    ytm = mac.app("YouTube Music")
    assert ytm.media_evidence[0] == "possible media app" and "electron" in ytm.media_evidence

    site = mac.app("Some Site")
    assert site.web_start_url == "https://music.example.com/"
    assert site.media_evidence[0] == "web app" and "web app for music.example.com" in site.media_evidence

    browser = mac.app("Acme Browser")
    assert browser.handles == ("https", "http") and browser.media_evidence[0] == "web browser"
    assert browser.web_start_url == ""  # an update-feed URL is not a start page
    assert mac.app("Mail").handles == ("mailto",) and mac.app("Notes").handles == ()

    saavn = mac.app("SaavnLike")
    assert saavn.category == "music"  # from Spotlight's App Store category
    assert "background-audio" in saavn.media_evidence

    assert mac.app("Writer").media_evidence == () and mac.app("Writer").caps == frozenset()


def test_usage_fields_come_from_spotlight_and_the_process_table(mac):
    spotify, weather = mac.add("Spotify"), mac.add("Weather")
    now = time.time()
    mac.usage = {
        "Spotify": (now - 3600, 98, 4, "", "Spotify"),
        "Weather": (None, 0, 0, "", "Weather Station"),
    }
    mac.running = {str(spotify)}
    mac.dock = {"com.acme.weather"}
    s, w = mac.app("Spotify"), mac.app("Weather Station")
    assert (s.last_used, s.use_count, s.days_used_30, s.running, s.in_dock) == (
        now - 3600,
        98,
        4,
        True,
        False,
    )
    assert (w.last_used, w.use_count, w.running, w.in_dock) == (None, 0, False, True)
    assert "Weather" in w.aliases  # the bundle's own name still names it
    assert weather.exists()


# ------------------------------------------------------------------------------ own schemes
@pytest.mark.parametrize(
    ("schemes", "bid", "name", "own"),
    [
        (["spotify"], "com.example.spotify", "Spotify", ("spotify",)),
        (["music", "musics", "itms", "daap"], "com.acme.Music", "Music", ("music", "musics")),
        (["com.acme.tv", "videos"], "com.acme.TV", "TV", ("com.acme.tv",)),  # the bundle id itself
        (["http", "https", "mailto", "rtsp"], "com.example.player", "Player", ()),  # protocols, not apps
        (["msauth.com.example.chat", "chat"], "com.example.chat", "Chat", ("chat",)),  # auth callback: no
        (["fb1234567", "whatsapp"], "net.example.WhatsApp", "WhatsApp", ("whatsapp",)),
        (["app", "desktop"], "com.example.app", "Thing", ()),  # generic words
    ],
)
def test_own_schemes(schemes, bid, name, own):
    assert inv.own_schemes(schemes, bid, name) == own


# ------------------------------------------------------------------------------ named_in
def _fixture_apps() -> list[inv.App]:
    A = inv.App
    return [
        A("com.example.spotify", "Spotify", "/A/Spotify.app", own_schemes=("spotify",)),
        A("net.example.WhatsApp", "WhatsApp", "/A/WhatsApp.app", own_schemes=("whatsapp",)),
        A("com.acme.Music", "Music", "/A/Music.app", own_schemes=("music", "musics")),
        A("com.acme.Notes", "Notes", "/A/Notes.app", own_schemes=("notes",)),
        A("com.acme.MobileSMS", "Messages", "/A/Messages.app", own_schemes=("messages",)),
        A("com.acme.weather", "Weather", "/A/Weather.app", own_schemes=("weather",)),
        A("com.acme.iCal", "Calendar", "/A/Calendar.app"),
        A("com.acme.Browser", "Acme Browser", "/A/Acme Browser.app"),
        A("com.example.musicstudio", "Music Studio 2025", "/A/Music Studio 2025.app"),
        A("com.example.vlclike", "VLC", "/A/VLC.app"),
        A("com.example.neo", "Neo", "/A/Neo.app"),
    ]


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("play 505 on spotify", "Spotify"),
        ("Neo, play 505 on Spotify", "Spotify"),  # the assistant being addressed is not an app
        ("hey neo play 505 on spotify", "Spotify"),
        ("PLAY 505 ON SPOTIFY!", "Spotify"),
        ("spotify's top songs", "Spotify"),  # a coined name counts anywhere
        ("open whatsapp", "WhatsApp"),
        ("send hi to mom on whatsapp", "WhatsApp"),
        ("play the trailer with vlc", "VLC"),
        ("play 505 in music", "Music"),  # a common word, but after 'in'
        ("open the music app", "Music"),
        ("play it in acme music", "Music"),  # vendor + name
        ("open music studio", "Music Studio 2025"),  # longest name wins; version dropped
        ("jot this down in notes", "Notes"),
        ("open chrome", None),
        ("open browser", "Acme Browser"),  # vendor-less form of the name
        ("play some music", None),  # common words are not app names
        ("take a note", None),
        ("add milk to my notes", None),
        ("send a message to mom", None),
        ("what's the weather", None),
        ("turn on music", None),  # phrasal 'turn on' is not 'on <app>'
        ("add lunch to my calendar", None),
        ("", None),
    ],
)
def test_named_in(monkeypatch, text, want):
    use_fixture_words(monkeypatch)
    app = inv.named_in(text, _fixture_apps())
    assert (app.name if app else None) == want


def test_named_in_without_a_word_list_is_only_stricter(monkeypatch, tmp_path):
    use_fixture_words(monkeypatch, tmp_path / "missing-words")
    apps = _fixture_apps()
    assert inv.named_in("play 505 on spotify", apps).name == "Spotify"
    assert inv.named_in("open whatsapp", apps).name == "WhatsApp"
    assert inv.named_in("spotify's top songs", apps) is None  # every single word now needs a cue
    assert inv.named_in("play some music", apps) is None


@pytest.mark.parametrize(
    ("text", "want"),
    [
        # a name that is a generic noun newer than the word list is still an ordinary word
        ("play a podcast about history", None),
        ("play some podcasts", None),
        ("play the latest episode on podcasts", "Podcasts"),
        # a name made only of ordinary words needs a cue too
        ("find my keys", None),
        ("open find my", "Find My"),
        ("find my", "Find My"),
        # a trailing descriptor word is dropped: 'quicktime player' -> 'quicktime'
        ("open quicktime", "QuickTime Player"),
        ("play this in quicktime", "QuickTime Player"),
        # a greeting before an ordinary word is not the assistant being addressed
        ("ok play music", None),
        ("ok open music", "Music"),
        ("hey neo play some music", None),
        ("hey neo, open music", "Music"),
    ],
)
def test_named_in_generic_nouns_phrases_and_greetings(monkeypatch, tmp_path, text, want):
    words = tmp_path / "words"
    words.write_text((FIX / "words.txt").read_text() + "find\n")
    use_fixture_words(monkeypatch, words)
    A = inv.App
    apps = [
        *_fixture_apps(),
        A("com.acme.podcasts", "Podcasts", "/A/Podcasts.app", own_schemes=("podcast", "podcasts")),
        A("com.acme.findmy", "Find My", "/A/Find My.app", own_schemes=("findmy",)),
        A("com.acme.QuickTimePlayerX", "QuickTime Player", "/A/QuickTime Player.app"),
    ]
    app = inv.named_in(text, apps)
    assert (app.name if app else None) == want


def test_named_in_uses_the_inventory_and_prefers_the_app_used_more(mac):
    for n in ("Spotify", "Music", "WhatsApp", "Notes", "Messages", "Weather"):
        mac.add(n)
    assert inv.named_in("Neo, play 505 on spotify").name == "Spotify"
    assert inv.named_in("open whatsapp").bundle_id == "net.example.WhatsApp"
    assert inv.named_in("play some music") is None
    a = inv.App("com.one.Player", "Player", "/A/Player.app", use_count=1)
    b = inv.App("com.two.Player", "Player", "/B/Player.app", use_count=500, running=True)
    assert inv.named_in("open player", [a, b]) is b  # same name: the one this user uses


# ------------------------------------------------------------------------------ sdef parsing + caps
def test_sdef_music_like(tmp_path):
    app = bundle(tmp_path, "Music")  # OSAScriptingDefinition names the file without ".sdef"
    d = sdef.load(str(app))
    assert d is not None
    assert "make" in d.commands and "quit" in d.commands  # XInclude (2003) of the standard suite
    assert d.commands["search"].params[0] == ("for", "pTrm", "text")
    assert d.commands["search"].direct == "playlist" and d.commands["search"].result == "track"
    assert d.commands["play"].direct == "specifier" and d.commands["play"].direct_optional
    assert d.commands["play"].optional == ("once",)
    assert d.classes["playlist"][0] == "cPly"
    assert "name" in d.classes["track"][1] and "artist" in d.classes["track"][1]  # inherited + own
    assert {"player state", "current track", "name"} <= set(d.classes["application"][1])
    assert d.properties["player state"] == "pPlS" and d.properties["current track"] == "pTrk"
    assert d.enums["ePlS"]["playing"] == "kPSP"
    assert d.container_path("playlist") == ["playlist"]
    assert d.prop_type("application", "current track") == "track"
    assert sdef.caps(d) == {"media.search_play", "media.resume", "media.state", "media.now_playing"}


def test_sdef_spotify_like(tmp_path):
    d = sdef.load(str(bundle(tmp_path, "Spotify")))
    assert d is not None
    assert "make" not in d.commands  # the include there is commented out
    assert {"player state", "current track", "name", "frontmost"} <= set(
        d.classes["application"][1]
    )  # merged
    assert d.commands["play track"].direct == "text" and d.commands["play"].direct == ""
    assert d.commands["play track"].params == (("in context", "cotx", "text"),)
    assert sdef.caps(d) == {
        "media.play_uri",
        "media.item_uri",
        "media.resume",
        "media.state",
        "media.now_playing",
    }


def test_sdef_notes_like(tmp_path):
    d = load_fixture_sdef(tmp_path, "notes")  # XInclude 2001
    assert d is not None and "make" in d.commands and d.classes["note"][0] == "note"
    assert sdef.caps(d) == {"note.make"}


def test_sdef_mail_like(tmp_path):
    d = load_fixture_sdef(tmp_path, "mail")
    assert d is not None
    # the standard suite was spliced into a suite, minus the commands the xpointer excludes
    assert "make" in d.commands and "duplicate" not in d.commands and "move" not in d.commands
    assert d.commands["delete"].code == "maildelo"  # the app's own delete, not the standard one
    assert sdef.caps(d) == {"mail.compose"}


def test_sdef_messages_calendar_reminders(tmp_path):
    msgs = load_fixture_sdef(tmp_path, "messages")
    assert msgs.commands["send"].direct == "file|text"
    assert msgs.commands["send"].params == (("to", "TO  ", "participant|chat"),)
    assert sdef.caps(msgs) == {"message.send"}
    assert sdef.caps(load_fixture_sdef(tmp_path, "calendar")) == {"event.make"}  # a hidden make still counts
    assert sdef.caps(load_fixture_sdef(tmp_path, "reminders")) == {"reminder.make"}


def test_sdef_negatives(tmp_path):
    writer = load_fixture_sdef(tmp_path, "writer")
    assert sdef.caps(writer) == frozenset()
    assert writer.commands["print"].code == "aevtpdoc"  # a hidden duplicate never hides the visible one
    assert sdef.caps(load_fixture_sdef(tmp_path, "slides")) == frozenset()  # "playing" but nothing plays
    assert sdef.caps(None) == frozenset()


def test_sdef_caps_by_codes_alone_and_names_alone():
    C, D = sdef.Command, sdef.Dictionary
    by_code = D(
        commands={
            "lookup": C("lookup", "xxxxSrch", "", (("term", "trm ", "text"),)),
            "start": C("start", "xxxxPlay", "media item"),
            "flip": C("flip", "xxxxPlPs"),
        },
        classes={"application": ("capp", ("status",))},
        properties={"status": "pPlS"},
        enums={},
    )
    assert sdef.caps(by_code) >= {"media.search_play", "media.resume", "media.state"}
    by_name = D(
        commands={
            "play/pause": C("play/pause", "abcdefgh"),
            "create note": C("create note", "abcdnote", "text"),
        },
        classes={},
        properties={},
        enums={},
    )
    assert sdef.caps(by_name) == {"media.resume", "note.make"}


def test_sdef_absolute_file_url_include_and_bad_input(tmp_path):
    lib = tmp_path / "Scripting Defs"
    lib.mkdir()
    shutil.copy(FIX / "sdef" / "standard.sdef", lib / "Standard.sdef")
    href = "file://localhost" + urllib.parse.quote(str(lib / "Standard.sdef"))
    main = tmp_path / "app.sdef"
    main.write_text(
        '<?xml version="1.0"?><dictionary xmlns:xi="http://www.w3.org/2001/XInclude">'
        f'<xi:include href="{href}" xpointer="xpointer(/dictionary/suite)"/>'
        '<suite name="S" code="ssss"><class name="note" code="note"/></suite></dictionary>'
    )
    d = sdef.load_file(str(main))
    assert d is not None and "make" in d.commands and sdef.caps(d) == {"note.make"}
    bad = tmp_path / "bad.sdef"
    bad.write_text("<dictionary><suite>")
    assert sdef.load_file(str(bad)) is None
    assert sdef.load_file(str(tmp_path / "missing.sdef")) is None


def test_sdef_load_from_bundles(tmp_path):
    assert sdef.load(str(bundle(tmp_path, "Legacy Scripted"))) is None  # scriptable, but no dictionary
    assert sdef.load(str(bundle(tmp_path, "WhatsApp"))) is None
    # scriptable with one undeclared sdef in Resources: that is its dictionary
    app = bundle(tmp_path, "Legacy Scripted", rename="Legacy Two")
    (app / "Contents" / "Resources").mkdir()
    shutil.copy(FIX / "sdef" / "messages.sdef", app / "Contents" / "Resources" / "Whatever.sdef")
    d = sdef.load(str(app))
    assert d is not None and sdef.caps(d) == {"message.send"}


# ------------------------------------------------------------------------------ affinity
def test_affinity_orders_by_how_much_this_user_uses_the_app():
    now = 1_800_000_000.0
    A = inv.App
    heavy = A(
        "a.heavy",
        "Heavy",
        "/h",
        last_used=now - 3600,
        use_count=200,
        days_used_30=15,
        running=True,
        in_dock=True,
        handles=("https",),
    )
    regular = A("a.regular", "Regular", "/r", last_used=now - 2 * DAY, use_count=40, days_used_30=6)
    stale = A("a.stale", "Stale", "/s", last_used=now - 60 * DAY, use_count=40, days_used_30=0)
    docked = A("a.docked", "Docked", "/d", in_dock=True)
    never = A("a.never", "Never", "/n")
    scores = [inv.affinity(x, now) for x in (heavy, regular, stale, docked, never)]
    assert scores == sorted(scores, reverse=True) and len(set(scores)) == 5
    assert all(0.0 <= s <= 1.0 for s in scores) and scores[-1] == 0.0
    assert inv.affinity(regular, now + 30 * DAY) < inv.affinity(regular, now)  # recency fades
    assert inv.affinity(never, now) < inv.affinity(A("a.run", "Run", "/x", running=True), now)


def test_affinity_from_inventory(mac):
    for n in ("Spotify", "Music", "WhatsApp"):
        mac.add(n)
    now = time.time()
    mac.usage = {"Spotify": (now - 7200, 98, 4, "", ""), "WhatsApp": (now - 20 * DAY, 5, 1, "", "")}
    apps = sorted(inv.inventory(), key=inv.affinity, reverse=True)
    assert [a.name for a in apps] == ["Spotify", "WhatsApp", "Music"]


# ------------------------------------------------------------------------------ caching
def test_cache_reuses_bundles_and_rereads_usage_cheaply(mac, monkeypatch):
    for n in ("Spotify", "Music", "Notes", "WhatsApp"):
        mac.add(n)
    built: list[str] = []
    real_build = inv._build_static
    monkeypatch.setattr(
        inv,
        "_build_static",
        lambda path, *a, **k: (built.append(Path(path).stem), real_build(path, *a, **k))[1],
    )

    first = inv.inventory()
    assert sorted(built) == ["Music", "Notes", "Spotify", "WhatsApp"] and mac.calls["spotlight"] == 1
    assert mac.usage_reads == [["Music", "Notes", "Spotify", "WhatsApp"]]

    # warm: nothing re-parsed, nothing re-enumerated, no usage re-read (nothing is running)
    t = time.perf_counter()
    second = inv.inventory()
    assert time.perf_counter() - t < 0.05
    assert second == first and len(built) == 4 and mac.calls["spotlight"] == 1 and len(mac.usage_reads) == 1

    # a running app's usage is re-read on every call (launching it changes it); the rest wait for the TTL
    mac.running = {str(mac.apps_dir / "Spotify.app")}
    assert mac.app("Spotify").running
    assert mac.usage_reads[-1] == ["Spotify"]
    mac.running = set()
    inv.inventory()
    assert mac.usage_reads[-1] == ["Spotify"]  # once more after it quits
    n = len(mac.usage_reads)
    inv.inventory()
    assert len(mac.usage_reads) == n
    monkeypatch.setattr(inv, "_USAGE_TTL", -1.0)
    inv.inventory()
    assert mac.usage_reads[-1] == ["Music", "Notes", "Spotify", "WhatsApp"]
    monkeypatch.setattr(inv, "_USAGE_TTL", 60.0)

    # an updated bundle (Info.plist changed) is re-read, and only that one
    plist = mac.apps_dir / "Music.app" / "Contents" / "Info.plist"
    info = plistlib.loads(plist.read_bytes())
    info["LSApplicationCategoryType"] = "public.app-category.video"
    plist.write_bytes(plistlib.dumps(info))
    st = os.stat(plist)
    os.utime(plist, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))
    assert mac.app("Music").category == "video"
    assert built[4:] == ["Music"]

    # an app dropped into an app folder shows up without a refresh; a removed one disappears
    mac.add("Weather")
    assert "Weather" in {a.name for a in inv.inventory()} and mac.calls["spotlight"] == 2
    shutil.rmtree(mac.apps_dir / "WhatsApp.app")
    assert "WhatsApp" not in {a.name for a in inv.inventory()}

    # refresh re-enumerates and re-reads every bundle
    before = len(built)
    inv.inventory(refresh=True)
    assert mac.calls["spotlight"] == 4 and len(built) - before == 4


def test_default_for_and_app_by_id(mac):
    mail, music = mac.add("Mail"), mac.add("Music")
    mac.add("Notes")
    (mac.tmp / "Elsewhere").mkdir()
    elsewhere = bundle(mac.tmp / "Elsewhere", "Acme Browser")  # installed where the scan doesn't look
    mac.handlers = {"mailto:": str(mail), "public.mp3": str(music), "https://": str(elsewhere)}
    assert inv.default_for("mailto:").name == "Mail"
    assert inv.default_for("public.mp3").bundle_id == "com.acme.Music"
    assert inv.default_for("https://").name == "Acme Browser"
    assert inv.default_for("nothing:") is None and inv.default_for("") is None
    assert inv.app_by_id("com.acme.notes").name == "Notes"  # case-insensitive
    assert inv.app_by_id("com.example.nowhere") is None
    mac.by_id = {"com.acme.Browser": str(elsewhere)}
    assert inv.app_by_id("com.acme.Browser").path == str(elsewhere)


def test_default_for_an_os_service_app_left_out_of_the_listing(mac):
    """An OS service app without a category isn't listed, but is still the answer when it is the
    default handler (the listing's category filter must not stick to the bundle)."""
    mounter = make_app(mac.services, "Mounter", CFBundleIdentifier="com.example.mounter")
    mac.handlers = {"dmg": str(mounter)}
    assert "Mounter" not in {a.name for a in inv.inventory()}
    got = inv.default_for("dmg")
    assert got is not None and got.bundle_id == "com.example.mounter"
    assert "Mounter" not in {a.name for a in inv.inventory()}  # and the listing is unchanged


# ------------------------------------------------------------------------------ the rule
_APP_NAMES = re.compile(
    r"spotify|whatsapp|youtube|itunes|chrome|safari|firefox|\bvlc\b|iina|tidal|deezer|jiosaavn|"
    r"com\.apple\.(music|tv|mail|notes|ical|reminders|mobilesms|safari|podcasts)|"
    r"[a-z0-9-]+\.(com|net|org|io|fm)(/|$)|https?://[a-z0-9]",
    re.I,
)


@pytest.mark.parametrize("module", [inv, sdef])
def test_no_app_names_bundle_ids_or_websites_in_logic(module):
    """String constants in the code (docstrings aside) must not name an app or a website."""
    tree = ast.parse(Path(module.__file__).read_text())
    docs = {
        id(n.body[0].value)
        for n in ast.walk(tree)
        if isinstance(n, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        and n.body
        and isinstance(n.body[0], ast.Expr)
        and isinstance(n.body[0].value, ast.Constant)
    }
    consts = [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs
    ]
    consts = [c for c in consts if not c.startswith("{")]  # XML namespace names ({uri}tag) are not sites
    assert [c for c in consts if _APP_NAMES.search(c)] == []


def test_includes_and_plists_that_are_not_regular_files_never_hang(tmp_path):
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    main = tmp_path / "app.sdef"
    main.write_text(
        '<?xml version="1.0"?><dictionary xmlns:xi="http://www.w3.org/2003/XInclude">'
        f'<xi:include href="file://{urllib.parse.quote(str(fifo))}"/><xi:include href="file:///dev/zero"/>'
        '<suite name="S" code="ssss"><command name="play" code="xxxxPlay"/></suite></dictionary>'
    )
    d = sdef.load_file(str(main))
    assert d is not None and set(d.commands) == {"play"}
    app = tmp_path / "Piped.app" / "Contents"
    app.mkdir(parents=True)
    os.mkfifo(app / "Info.plist")
    assert inv._plist_key(str(tmp_path / "Piped.app")) is None
    assert inv._read_info(str(tmp_path / "Piped.app")) == ({}, "")
    assert sdef.load(str(tmp_path / "Piped.app")) is None
