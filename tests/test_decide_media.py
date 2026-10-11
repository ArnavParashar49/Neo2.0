"""neo2.core.decide.media: the generic play-something driver, fully faked.

Nothing here launches an app, sends an Apple event, performs an AX action, plays audio or opens a
URL: osascript, the AX tree, CoreAudio, the video search, open_app, the engine and the inventory
are all fakes. App names below are fixtures.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from types import SimpleNamespace

import pytest

from neo2.core.decide import axkit, media
from neo2.core.decide.engine import Candidate
from neo2.core.decide.inventory import App
from neo2.core.decide.sdef import Command, Dictionary
from neo2.core.tools.computer import apps as apps_mod
from neo2.core.tools.computer import ax as ax_mod
from neo2.core.tools.computer import focus as focus_mod
from neo2.core.tools.computer import input as inp_mod

BID = "com.fixture.player"
APP_PID = 500
HELPER_PID = 501
NEO_PID = 900
BROWSER_BID = "com.fixture.browser"
BROWSER = App(bundle_id=BROWSER_BID, name="Fixture Browser", path="/Applications/Fixture Browser.app", handles=("https",))


# ---------------------------------------------------------------- fixture dictionaries and apps
def library_dict() -> Dictionary:
    """A library player: 'search <playlist> for <text>' + 'play <item>' (de-facto raw codes)."""
    return Dictionary(
        commands={
            "search": Command("search", "hookSrch", "playlist", (("for", "pTrm", "text"), ("only", "pAre", "eSrA"))),
            "play": Command("play", "hookPlay", "specifier", (), direct_optional=True),
        },
        classes={
            "application": ("capp", ("current track", "name")),
            "playlist": ("cPly", ("name",)),
            "track": ("cTrk", ("name", "artist")),
        },
        properties={"current track": "pTrk", "name": "pnam", "artist": "pArt"},
        enums={},
    )


def uri_dict() -> Dictionary:
    """A streaming player: 'play track <uri>' + a current track exposing its own URL."""
    return Dictionary(
        commands={
            "play track": Command(
                "play track", "fixrPCtx", "text", (("in context", "cotx", "text"),), direct_description="the URI of the track"
            ),
            "play": Command("play", "fixPlay_", "", ()),
        },
        classes={"application": ("capp", ("current track",)), "track": ("cTrk", ("name", "artist", "item url"))},
        properties={"current track": "pTrk", "name": "pnam", "artist": "pArt", "item url": "iurl"},
        enums={},
    )


def make_app(script=None, **kw) -> App:
    base = dict(bundle_id=BID, name="Fixture Player", path="/Applications/Fixture Player.app", script=script)
    base.update(kw)
    return App(**base)


# ---------------------------------------------------------------- fakes
class FakeAudio:
    """CoreAudio HAL clients: {object: [pid, client bundle id, output running]}."""

    def __init__(self, monkeypatch):
        self.clients: dict[int, list] = {}
        self.resp: dict[int, int] = {HELPER_PID: APP_PID, 901: NEO_PID, 601: 600}
        self.bids: dict[int, str] = {APP_PID: BID, NEO_PID: "com.fixture.neo", 600: BROWSER_BID}
        self.paths: dict[int, str] = {}
        self._ids = itertools.count(100)
        monkeypatch.setattr(media, "_hal_clients", lambda: [(o, c[0], c[1], c[2]) for o, c in list(self.clients.items())])
        monkeypatch.setattr(media, "responsible_pid", lambda pid: self.resp.get(pid, pid))
        monkeypatch.setattr(media, "_bundle_id_of_pid", lambda pid: self.bids.get(pid, ""))
        monkeypatch.setattr(media, "_proc_path", lambda pid: self.paths.get(pid, ""))
        monkeypatch.setattr(media, "_self_family", lambda: frozenset({NEO_PID}))

    def start(self, pid: int, client: str) -> int:
        obj = next(self._ids)
        self.clients[obj] = [pid, client, True]
        return obj

    def app_plays(self) -> None:  # the fixture app's audio helper opens an output stream
        self.start(HELPER_PID, BID + ".helper")

    def browser_plays(self) -> None:
        self.start(601, BROWSER_BID + ".helper")


class FakeOsa:
    """osascript: answers chosen by what the script is; every call is recorded."""

    def __init__(self):
        self.calls: list[tuple[str, list[str]]] = []
        self.rules: list[tuple[str, object]] = []
        self.now_playing = (0, "NOT_RUNNING", "")

    def on(self, marker: str, answer) -> None:
        self.rules.append((marker, answer))

    async def __call__(self, script: str, args: list[str], timeout: float):
        self.calls.append((script, list(args)))
        for marker, answer in self.rules:
            if marker in script:
                return answer() if callable(answer) else answer
        if "NOT_RUNNING" in script:
            return self.now_playing
        return 0, "OK", ""

    def scripts_with(self, marker: str) -> list[tuple[str, list[str]]]:
        return [c for c in self.calls if marker in c[0]]


class FakeEngine:
    def __init__(self, cands=()):
        self.cands = list(cands)
        self.records: list[tuple] = []
        self.priors: dict[str, float] = {}
        self.chosen_with: list[tuple[str, str, str]] = []  # (kind, text it reads, the user's words)

    async def choose(self, kind, text, apps=None, *, said=""):
        self.chosen_with.append((kind, text, said))
        return list(self.cands)

    def record(self, kind, key, method, ok, reason=""):
        self.records.append((kind, key, method, ok, reason))

    def prior(self, kind, key, method):
        return self.priors.get(key, 0.5)


class El:
    """A fake AX element."""

    def __init__(self, role, children=(), actions=("AXPress",), on_press=None, **attrs):
        self.attrs = {"AXRole": role, **attrs}
        self.children = list(children)
        self.actions = list(actions)
        self.on_press = on_press
        self.on_set: dict[str, object] = {}
        self.refuse: set[str] = set()  # attributes the app turns a write of down (an AX error)
        self.pressed = 0


class FakeAX:
    def __init__(self, monkeypatch, root: El):
        self.root = root
        self.sets: list[tuple[str, object]] = []
        self.returns: list[int] = []
        self.performed: list[tuple[El, str]] = []
        monkeypatch.setattr(media, "_attr", self.attr)
        monkeypatch.setattr(media, "_set", self.set)
        monkeypatch.setattr(media, "_perform", self.perform)
        monkeypatch.setattr(media, "_actions", lambda el: list(el.actions))
        monkeypatch.setattr(media, "_ax_app", lambda pid: self.root)
        monkeypatch.setattr(media, "_trusted", lambda: True)
        monkeypatch.setattr(media, "_post_return", self.returns.append)

    @staticmethod
    def attr(el, name, default=None):
        if el is None:
            return default
        if name == "AXChildren":
            return list(el.children)
        return el.attrs.get(name, default)

    def set(self, el, name, value):
        if el is None:
            return False
        self.sets.append((name, value))
        if name in el.refuse:
            return False
        el.attrs[name] = value
        if name in el.on_set:
            el.on_set[name](value)
        return True

    def perform(self, el, action):
        self.performed.append((el, action))
        if action in el.actions:
            if action == "AXPress":
                el.pressed += 1
                if el.on_press:
                    el.on_press()
            return True
        return False


def player_window(audio: FakeAudio | None, *, with_search: bool = True, results: bool = True):
    """(app root, the button that should be pressed): a toolbar search box, a transport bar with a
    bare 'Play', and results that appear once a query is typed."""
    field = El("AXTextField", actions=("AXConfirm",), AXSubrole="AXSearchField", AXPlaceholderValue="Search")
    listing = El("AXGroup")
    transport = El(
        "AXGroup", [El("AXButton", AXDescription="Previous"), El("AXButton", AXDescription="Play"), El("AXButton", AXDescription="Next")]
    )
    window = El("AXWindow", [El("AXToolbar", [field] if with_search else []), listing, transport], AXTitle="Fixture")
    root = El("AXApplication", [window], AXFocusedWindow=window, AXManualAccessibility=False)
    target = El("AXButton", on_press=(audio.app_plays if audio else None))

    def typed(value):
        if not results:
            return
        target.attrs["AXDescription"] = f"Play {value} by The Fixtures"
        listing.children = [
            El("AXButton", AXDescription="Play Something Else"),
            target,
            El("AXRow", [El("AXStaticText", AXValue=value)]),
        ]

    field.on_set["AXValue"] = typed
    return root, target, transport.children[1]


PLAYER = "Fixture Player"
OTHER, OTHER_PID = "Fixture Mail", 777
_CMD = 1 << 20  # kCGEventFlagMaskCommand


def search_field(root: El) -> El:
    return root.children[0].children[0].children[0]  # (window → toolbar → search box)


def focusable(root: El, el: El) -> None:
    """Accessibility can give ``el`` the keyboard focus: the app then says it is its focused element."""
    el.on_set["AXFocused"] = lambda v: root.attrs.__setitem__("AXFocusedUIElement", el if v else None)


def framed(el: El, x: float = 100.0, y: float = 200.0, w: float = 80.0, h: float = 30.0) -> El:
    el.attrs.update(AXPosition=(x, y), AXSize=(w, h))
    return el


class FakeScreen:
    """The window server and the pointer for a guarded click: which app is in front, what is at each
    point (the player's own elements by their frames, unless another window is over them), the apps
    brought forward and the clicks made. A click lands on whatever is at the point when it is made.
    Nothing real is looked at, brought forward or clicked."""

    def __init__(self, monkeypatch, *els: El):
        self.els = list(els)
        self.front = (PLAYER, APP_PID)
        self.over: tuple[str, int] | None = None  # another window over the player's elements
        self.over_stays = False  # …still there with the player in front (a floating panel)
        self.brought: list[int] = []
        self.clicks: list[tuple[float, float]] = []
        self.on_click: dict[int, object] = {}
        monkeypatch.setattr(ax_mod, "visible_frame", self.frame)
        monkeypatch.setattr(ax_mod, "hit", self.hit)
        monkeypatch.setattr(ax_mod, "is_neo", lambda pid, name="": pid == NEO_PID)
        monkeypatch.setattr(ax_mod, "app_names", lambda pid: set())
        monkeypatch.setattr(ax_mod, "windows_at", lambda x, y: [ax_mod.Win(APP_PID, PLAYER, 0, (0.0, 0.0, 1000.0, 800.0))])
        monkeypatch.setattr(focus_mod, "front_app", lambda: self.front)
        monkeypatch.setattr(apps_mod, "bring_pid_front", self.bring)
        monkeypatch.setattr(inp_mod, "click", self.click)

    @staticmethod
    def frame(el):
        pos, size = el.attrs.get("AXPosition"), el.attrs.get("AXSize")
        return (*pos, *size) if pos and size else None

    def at(self, x, y) -> El | None:
        for el in self.els:
            f = self.frame(el)
            if f and f[0] <= x <= f[0] + f[2] and f[1] <= y <= f[1] + f[3]:
                return el
        return None

    def hit(self, x, y):
        if self.over is not None:
            name, pid = self.over
            return ax_mod.Hit(pid=pid, app=name, label="Inbox", role="AXStaticText", regular=True, el=El("AXStaticText"))
        el = self.at(x, y)
        if el is None:
            return ax_mod.Hit(pid=APP_PID, app=PLAYER, role="AXGroup", regular=True, el=El("AXGroup"))
        label = str(el.attrs.get("AXDescription") or el.attrs.get("AXPlaceholderValue") or "")
        return ax_mod.Hit(pid=APP_PID, app=PLAYER, label=label, role=el.attrs["AXRole"], regular=True, el=el)

    async def bring(self, pid):  # (async, like apps.bring_pid_front)
        self.brought.append(pid)
        self.front = (PLAYER, pid)
        if not self.over_stays:
            self.over = None
        return True, PLAYER

    def click(self, x, y, **k):
        self.clicks.append((x, y))
        el = None if self.over is not None else self.at(x, y)
        if el is not None and id(el) in self.on_click:
            self.on_click[id(el)]()
        return "clicked"


class FakeKeys:
    """Keystrokes and typing NEO sends to the player's pid. Like the real thing they land in whatever
    element the app has focused, or nowhere: ⌘A selects its text, Delete deletes the selection (else
    one character), typing replaces the selection. Everything sent is recorded."""

    def __init__(self, monkeypatch, root: El):
        self.root = root
        self.keys: list[tuple[int, int]] = []
        self.typed: list[str] = []
        monkeypatch.setattr(media, "_post_key", self.key)
        monkeypatch.setattr(axkit, "_type", self.type)
        monkeypatch.setattr(axkit, "_pid_of", lambda el: APP_PID)
        monkeypatch.setattr(axkit, "_READBACK_S", 0.05)
        monkeypatch.setattr(media, "_whole_range", lambda el: ("all", len(str(el.attrs.get("AXValue", "")))))

    def _focused(self) -> El | None:
        return self.root.attrs.get("AXFocusedUIElement")

    def key(self, pid, keycode, flags=0):
        assert pid == APP_PID  # only ever to the player
        self.keys.append((keycode, flags))
        el = self._focused()
        if el is None:
            return
        if (keycode, flags) == (0, _CMD):
            el.attrs["AXSelectedTextRange"] = ("all", len(str(el.attrs.get("AXValue", ""))))
        elif keycode == 51:
            v = str(el.attrs.get("AXValue", ""))
            el.attrs["AXValue"] = "" if el.attrs.pop("AXSelectedTextRange", None) else v[:-1]

    def type(self, pid, text, newline):
        assert pid == APP_PID
        self.typed.append(text)
        el = self._focused()
        if el is None:
            return
        v = "" if el.attrs.pop("AXSelectedTextRange", None) else str(el.attrs.get("AXValue", ""))
        el.attrs["AXValue"] = v + text
        if "AXValue" in el.on_set:
            el.on_set["AXValue"](el.attrs["AXValue"])


# ---------------------------------------------------------------- shared setup
@pytest.fixture(autouse=True)
def _fast_and_hermetic(monkeypatch):
    monkeypatch.setattr(media, "_POLL_S", 0.01)
    monkeypatch.setattr(media, "_AX_STEP_S", 0.01)
    monkeypatch.setattr(
        media,
        "TIMEOUTS",
        {
            "learned_uri": (0.5, 0.5, 0.3),
            "script_search_play": (0.5, 0.5, 0.3),
            "ax_search_play": (0.5, 0.5, 0.3),
            "local_file": (0.5, 0.5, 0.3),
            "web": (0.5, 0.5, 0.3),
        },
    )
    monkeypatch.setattr(media, "_last_open", None)
    # These tests drive the choice among apps; the web-first default has its own tests below.
    from neo2.core import config

    monkeypatch.setenv("NEO_PLAY_DEFAULT", "apps")
    monkeypatch.setenv("NEO_PLAY_SITE", "")
    monkeypatch.setattr(config, "_settings", None)

    async def no_site(word):
        return ""

    monkeypatch.setattr(media, "_resolve_site", no_site)

    def no_text_search(q, n=10):
        raise AssertionError("no web search in this test")

    monkeypatch.setattr(media, "_text_search", no_text_search)
    from neo2.core.decide import sites

    async def no_discovery(host, prefix=""):
        return []

    async def no_fetch(url, limit=0, partial=False):
        raise AssertionError("no page fetch in this test")

    monkeypatch.setattr(sites, "discover", no_discovery)
    monkeypatch.setattr(sites, "_fetch", no_fetch)
    monkeypatch.setattr(media, "automation", lambda bid, ask=False: media.AE_ALLOWED)  # never ask macOS
    monkeypatch.setattr(media, "audio_confirmable", lambda: True)
    words = {
        "play": frozenset({"Play", "Lecture"}),
        "search": frozenset({"Search", "Rechercher"}),
        "find": frozenset({"Find"}),
        "pause": frozenset({"Pause"}),
    }
    monkeypatch.setattr(media, "lexicon", lambda: words)
    media._rx.cache_clear()
    monkeypatch.setattr(media, "_runtime", lambda path: "electron")

    async def no_launch_needed(app, ready_s):
        return APP_PID

    monkeypatch.setattr(media, "_launch", no_launch_needed)
    inv = SimpleNamespace(named_in=lambda text, apps=None: None, default_for=lambda s: BROWSER)
    monkeypatch.setattr(media, "_inventory", lambda: inv)

    async def no_open(*a, **k):
        raise AssertionError("nothing may be opened in this test")

    monkeypatch.setattr(apps_mod, "open_app", no_open)
    monkeypatch.setattr(apps_mod, "_run", no_open)

    def no_search(q, n=12):
        raise AssertionError("no web search in this test")

    monkeypatch.setattr(media, "_video_search", no_search)

    async def no_osascript(*a, **k):
        raise AssertionError("no Apple events in this test")

    def no_ax(*a, **k):
        raise AssertionError("no real AX actions or keystrokes in this test")

    monkeypatch.setattr(media, "_osascript", no_osascript)
    for name in ("_set", "_perform", "_post_return", "_post_key", "_ax_app", "_pid_of"):
        monkeypatch.setattr(media, name, no_ax)
    # Nothing on screen and no pointer unless a test sets up a fake one (FakeScreen).
    monkeypatch.setattr(ax_mod, "visible_frame", lambda el: None)
    monkeypatch.setattr(inp_mod, "click", no_ax)
    monkeypatch.setattr(media, "_PRESS_GRACE_S", 0.05)
    yield
    media._rx.cache_clear()


@pytest.fixture
def audio(monkeypatch):
    return FakeAudio(monkeypatch)


@pytest.fixture
def osa(monkeypatch):
    fake = FakeOsa()
    monkeypatch.setattr(media, "_osascript", fake)
    return fake


@pytest.fixture
def engine(monkeypatch):
    eng = FakeEngine()
    monkeypatch.setattr(media, "_engine", lambda: eng)
    return eng


# ---------------------------------------------------------------- which methods an app offers
def test_methods_for_come_from_what_the_app_declares():
    assert media.methods_for(make_app(library_dict())) == ("script_search_play", "ax_search_play")
    assert media.methods_for(make_app(uri_dict(), own_schemes=("fixtureplayer",))) == ("learned_uri", "ax_search_play")
    # a file player with no dictionary: the user's own files first, then its search box
    assert media.methods_for(make_app(None, viewer_types=("public.mp3", "mkv"))) == ("local_file", "ax_search_play")
    assert media.methods_for(make_app(None)) == ("ax_search_play",)


def test_scripts_use_raw_codes_and_argv_never_the_query():
    spec = media._search_play_spec(library_dict())
    src = media._search_play_script(spec)
    assert "«event hookSrch» («class cPly» 1) given «class pTrm»:q" in src
    assert "«event hookPlay» hits" in src and "item 2 of argv" in src
    assert "Fixture" not in src and "tell application id bid" in src
    # a code that isn't a plain four/eight-char code never reaches script source
    bad = library_dict()
    bad.commands["search"] = Command("search", 'hoo"Srch', "playlist", (("for", "pTrm", "text"),))
    assert media._search_play_spec(bad) is None


# ---------------------------------------------------------------- the scripting path
async def test_script_path_plays_and_confirms_by_sound(audio, osa, engine):
    osa.on("«event hookSrch»", lambda: (audio.app_plays(), (0, "OK", ""))[1])
    osa.now_playing = (0, "NOT_RUNNING", "")
    tricky = '5"05 & (do shell script "x") & y'
    out = await media.play(tricky, make_app(library_dict()), "script_search_play")
    assert out.ok and out.text.endswith("in Fixture Player"), out
    (script, args), = osa.scripts_with("«event hookSrch»")
    assert args == [BID, tricky]  # passed as argv, never interpolated
    assert "do shell script" not in script


async def test_script_ok_without_sound_is_not_success(audio, osa, engine):
    osa.on("«event hookSrch»", (0, "OK", ""))  # "the script said OK" is not proof
    out = await media.play("505", make_app(library_dict()), "script_search_play")
    assert not out.ok and out.reason == "audio_not_started" and out.blame


async def test_permission_denied_is_not_the_apps_fault(audio, osa, engine):
    osa.on(
        "«event hookSrch»",
        (1, "", "execution error: Not authorized to send Apple events to Fixture Player. (-1743)"),
    )
    out = await media.play("505", make_app(library_dict()), "script_search_play")
    assert out == media.Outcome(False, out.text, blame=False, reason="permission")


# ---------------------------------------------------------------- the AX path
async def test_ax_types_the_query_and_presses_the_best_match(monkeypatch, audio, osa, engine):
    root, target, transport_play = player_window(audio)
    ax = FakeAX(monkeypatch, root)
    out = await media.play("505", make_app(None), "ax_search_play")
    assert out.ok and out.text == "Playing 505 by The Fixtures in Fixture Player", out
    assert target.pressed == 1 and transport_play.pressed == 0  # never the bare transport Play
    others = [el for el in root.children[0].children[1].children if el is not target]
    assert all(el.pressed == 0 for el in others)
    # the web-content switch was turned on for the search and restored afterwards
    assert ("AXManualAccessibility", True) in ax.sets
    assert root.attrs["AXManualAccessibility"] is False
    assert osa.calls == []  # no dictionary: no Apple events at all


def test_find_play_controls_ranks_named_play_buttons_first(monkeypatch):
    # localized word ('Lecture') and a hover 'Play' on a card that names the item both count
    card = El("AXGroup", [El("AXStaticText", AXValue="505"), El("AXButton", AXDescription="Play")])
    root = El(
        "AXWindow",
        [
            El("AXGroup", [El("AXButton", AXDescription="Play"), El("AXButton", AXDescription="Next")]),
            El("AXButton", AXDescription="Play Other Song"),
            card,
            El("AXButton", AXDescription="Lecture 505"),
        ],
    )
    monkeypatch.setattr(media, "_attr", FakeAX.attr)
    hits = media.find_play_controls(root, "505")
    assert [h.label for h in hits] == ["Lecture 505", "Play"]
    assert hits[1].el is card.children[1]


async def test_ax_with_no_search_box_fails_and_restores(monkeypatch, audio, osa, engine):
    root, _target, _ = player_window(audio, with_search=False)
    FakeAX(monkeypatch, root)
    out = await media.play("505", make_app(None), "ax_search_play")
    assert not out.ok and "search box" in out.text and out.blame
    assert root.attrs["AXManualAccessibility"] is False


async def test_ax_sign_in_wall_is_handed_to_the_user(monkeypatch, audio, osa, engine):
    root, _target, _ = player_window(audio)
    root.children[0].children.append(El("AXTextField", AXSubrole="AXSecureTextField"))
    FakeAX(monkeypatch, root)
    out = await media.play("505", make_app(None), "ax_search_play")
    assert out.reason == "login" and not out.blame


async def test_ax_without_accessibility_permission(monkeypatch, audio, osa, engine):
    root, _target, _ = player_window(audio)
    FakeAX(monkeypatch, root)
    monkeypatch.setattr(media, "_trusted", lambda: False)
    out = await media.play("505", make_app(None), "ax_search_play")
    assert out.reason == "permission" and not out.blame


async def test_a_cancelled_ax_attempt_still_restores_the_switch(monkeypatch, audio, osa, engine):
    root, target, _ = player_window(audio, results=False)  # results never show up
    FakeAX(monkeypatch, root)
    monkeypatch.setitem(media.TIMEOUTS, "ax_search_play", (5.0, 5.0, 1.0))
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(media.play("505", make_app(None), "ax_search_play"), 0.3)
    deadline = time.monotonic() + 2
    while root.attrs["AXManualAccessibility"] is not False and time.monotonic() < deadline:
        await asyncio.sleep(0.02)
    assert root.attrs["AXManualAccessibility"] is False and target.pressed == 0


# ---------------------------------------------------------------- pressing: never a blind click
async def test_a_play_control_with_a_press_is_pressed_not_clicked(monkeypatch, audio, osa, engine):
    """Accessibility's press comes first, even for a control that shows on screen: no pointer, no
    app brought forward, nothing that could land on another window."""
    root, target, _ = player_window(audio)
    FakeAX(monkeypatch, root)
    screen = FakeScreen(monkeypatch, framed(target))
    out = await media.play("505", make_app(None), "ax_search_play")
    assert out.ok, out
    assert target.pressed == 1 and screen.clicks == [] and screen.brought == []


async def test_a_control_with_no_press_is_clicked_through_the_pointers_checks(monkeypatch, audio, osa, engine):
    root, target, _ = player_window(audio)
    target.actions = []  # nothing Accessibility can press
    FakeAX(monkeypatch, root)
    screen = FakeScreen(monkeypatch, framed(target))
    screen.on_click[id(target)] = audio.app_plays
    out = await media.play("505", make_app(None), "ax_search_play")
    assert out.ok, out
    assert screen.clicks == [(140.0, 215.0)]  # the middle of the control
    assert target.pressed == 0


async def test_a_player_behind_another_app_comes_forward_before_its_control_is_clicked(monkeypatch, audio, osa, engine):
    root, target, _ = player_window(audio)
    target.actions = []
    FakeAX(monkeypatch, root)
    screen = FakeScreen(monkeypatch, framed(target))
    screen.on_click[id(target)] = audio.app_plays
    screen.front = screen.over = (OTHER, OTHER_PID)  # another app in front, its window over the control
    out = await media.play("505", make_app(None), "ax_search_play")
    assert out.ok, out
    assert screen.brought == [APP_PID] and screen.clicks == [(140.0, 215.0)]


async def test_no_click_lands_on_a_window_that_stays_over_the_player(monkeypatch, audio, osa, engine):
    root, target, _ = player_window(audio)
    target.actions = []
    FakeAX(monkeypatch, root)
    screen = FakeScreen(monkeypatch, framed(target))
    screen.front = screen.over = (OTHER, OTHER_PID)
    screen.over_stays = True  # with the player in front, the other window is still over the control
    out = await media.play("505", make_app(None), "ax_search_play")
    assert not out.ok and out.reason == "play_control_not_pressed", out
    assert screen.clicks == []


async def test_neos_own_window_over_the_player_is_never_clicked_or_moved(monkeypatch, audio, osa, engine):
    root, target, _ = player_window(audio)
    target.actions = []
    FakeAX(monkeypatch, root)
    screen = FakeScreen(monkeypatch, framed(target))
    screen.over = ("NEO", NEO_PID)
    out = await media.play("505", make_app(None), "ax_search_play")
    assert not out.ok and out.reason == "play_control_not_pressed", out
    assert screen.clicks == [] and screen.brought == []


async def test_something_else_of_the_players_over_its_control_is_not_clicked(monkeypatch, audio, osa, engine):
    """The player's own popup (a menu, a tooltip) over the control: the click would press that, so
    there is none."""
    root, target, _ = player_window(audio)
    target.actions = []
    FakeAX(monkeypatch, root)
    popup = framed(El("AXMenu"), 120.0, 190.0, 200.0, 100.0)
    screen = FakeScreen(monkeypatch, popup, framed(target))
    out = await media.play("505", make_app(None), "ax_search_play")
    assert not out.ok and out.reason == "play_control_not_pressed", out
    assert screen.clicks == []


async def test_a_press_the_player_ignores_gets_one_guarded_click(monkeypatch, audio, osa, engine):
    """Some Electron players say a press worked and do nothing: still silent a moment later, with
    the control not showing Pause, it gets one click, through the same checks."""
    root, target, _ = player_window(None)  # its press plays nothing
    FakeAX(monkeypatch, root)
    screen = FakeScreen(monkeypatch, framed(target))
    screen.on_click[id(target)] = audio.app_plays
    out = await media.play("505", make_app(None), "ax_search_play")
    assert out.ok, out
    assert target.pressed == 1 and screen.clicks == [(140.0, 215.0)]


async def test_a_press_that_shows_pause_is_never_clicked_again(monkeypatch, audio, osa, engine):
    """A player that started (its control says Pause now) but whose sound hasn't come yet: a
    second press would pause it."""
    root, target, _ = player_window(None)
    target.on_press = lambda: target.attrs.update(AXDescription="Pause 505 by The Fixtures")
    FakeAX(monkeypatch, root)
    screen = FakeScreen(monkeypatch, framed(target))
    out = await media.play("505", make_app(None), "ax_search_play")
    assert not out.ok and out.reason == "audio_not_started", out
    assert target.pressed == 1 and screen.clicks == []


async def test_a_press_is_not_clicked_again_while_the_player_was_already_playing(monkeypatch, audio, osa, engine):
    """It was playing before the press: whether the press took can't be told by sound, so no click."""
    root, target, _ = player_window(None)
    FakeAX(monkeypatch, root)
    screen = FakeScreen(monkeypatch, framed(target))
    audio.app_plays()
    await media.play("505", make_app(None), "ax_search_play")
    assert target.pressed == 1 and screen.clicks == []


# ---------------------------------------------------------------- typing: only into the focused box
async def test_return_goes_only_while_the_search_box_has_the_focus_and_reads_the_query(monkeypatch, audio, osa, engine):
    root, _target, _ = player_window(audio)
    ax = FakeAX(monkeypatch, root)
    field = search_field(root)
    out = await media.play("505", make_app(None), "ax_search_play")
    assert out.ok, out
    # The box took the query, but the app never said it has the focus: no Return (it could land on
    # something else in the app); the box's own confirm action instead.
    assert ax.returns == [] and (field, "AXConfirm") in ax.performed

    audio.clients.clear()
    root, _target, _ = player_window(audio)
    ax = FakeAX(monkeypatch, root)
    field = search_field(root)
    focusable(root, field)
    out = await media.play("505", make_app(None), "ax_search_play")
    assert out.ok, out
    assert ax.returns == [APP_PID] and (field, "AXConfirm") not in ax.performed


async def test_a_box_that_turns_its_value_down_is_typed_into_once_it_has_the_focus(monkeypatch, audio, osa, engine):
    root, target, _ = player_window(audio)
    FakeAX(monkeypatch, root)
    field = search_field(root)
    field.refuse = {"AXValue"}
    focusable(root, field)
    keys = FakeKeys(monkeypatch, root)
    screen = FakeScreen(monkeypatch, framed(field, 300.0, 20.0, 200.0, 24.0))
    out = await media.play("505", make_app(None), "ax_search_play")
    assert out.ok, out
    assert keys.typed == ["505"] and keys.keys == [] and screen.clicks == []  # (empty: nothing to clear)
    assert field.attrs["AXValue"] == "505" and target.pressed == 1


async def test_the_old_query_is_selected_and_deleted_before_the_new_one_is_typed(monkeypatch, audio, osa, engine):
    root, _target, _ = player_window(audio)
    ax = FakeAX(monkeypatch, root)
    field = search_field(root)
    field.refuse = {"AXValue"}
    field.attrs["AXValue"] = "an old song"
    focusable(root, field)
    keys = FakeKeys(monkeypatch, root)
    out = await media.play("505", make_app(None), "ax_search_play")
    assert out.ok, out
    assert ("AXSelectedTextRange", ("all", 11)) in ax.sets
    assert keys.keys == [(51, 0)] and keys.typed == ["505"]  # selected through Accessibility: no ⌘A
    assert field.attrs["AXValue"] == "505"


async def test_without_a_selection_through_accessibility_select_all_goes_to_the_focused_box(monkeypatch, audio, osa, engine):
    root, _target, _ = player_window(audio)
    FakeAX(monkeypatch, root)
    field = search_field(root)
    field.refuse = {"AXValue", "AXSelectedTextRange"}
    field.attrs["AXValue"] = "an old song"
    focusable(root, field)
    keys = FakeKeys(monkeypatch, root)
    out = await media.play("505", make_app(None), "ax_search_play")
    assert out.ok, out
    assert keys.keys == [(0, _CMD), (51, 0)] and keys.typed == ["505"]
    assert field.attrs["AXValue"] == "505"


async def test_a_box_that_wont_take_the_focus_is_clicked_through_the_pointers_checks(monkeypatch, audio, osa, engine):
    root, _target, _ = player_window(audio)
    FakeAX(monkeypatch, root)
    field = search_field(root)
    field.refuse = {"AXValue", "AXFocused"}
    keys = FakeKeys(monkeypatch, root)
    screen = FakeScreen(monkeypatch, framed(field, 300.0, 20.0, 200.0, 24.0))
    screen.on_click[id(field)] = lambda: root.attrs.__setitem__("AXFocusedUIElement", field)
    out = await media.play("505", make_app(None), "ax_search_play")
    assert out.ok, out
    assert screen.clicks == [(400.0, 32.0)] and keys.typed == ["505"]


async def test_nothing_is_typed_while_another_window_is_over_the_search_box(monkeypatch, audio, osa, engine):
    root, _target, _ = player_window(audio)
    FakeAX(monkeypatch, root)
    field = search_field(root)
    field.refuse = {"AXValue", "AXFocused"}
    keys = FakeKeys(monkeypatch, root)
    screen = FakeScreen(monkeypatch, framed(field, 300.0, 20.0, 200.0, 24.0))
    screen.front = screen.over = (OTHER, OTHER_PID)
    screen.over_stays = True
    out = await media.play("505", make_app(None), "ax_search_play")
    assert not out.ok and out.reason == "search_box_input_failed", out
    assert screen.clicks == [] and keys.keys == [] and keys.typed == []


async def test_nothing_is_typed_when_a_click_leaves_the_focus_elsewhere(monkeypatch, audio, osa, engine):
    root, _target, _ = player_window(audio)
    FakeAX(monkeypatch, root)
    field = search_field(root)
    field.refuse = {"AXValue", "AXFocused"}
    field.attrs["AXValue"] = "an old song"
    root.attrs["AXFocusedUIElement"] = root.children[0].children[1]  # the results list keeps it
    keys = FakeKeys(monkeypatch, root)
    screen = FakeScreen(monkeypatch, framed(field, 300.0, 20.0, 200.0, 24.0))
    out = await media.play("505", make_app(None), "ax_search_play")
    assert not out.ok and out.reason == "search_box_input_failed", out
    assert screen.clicks == [(400.0, 32.0)]
    assert keys.keys == [] and keys.typed == [] and field.attrs["AXValue"] == "an old song"


async def test_a_password_field_is_never_typed_into(monkeypatch):
    root = El("AXApplication")
    field = El("AXTextField", AXSubrole="AXSecureTextField")
    root.attrs["AXFocusedUIElement"] = field
    FakeAX(monkeypatch, root)
    keys = FakeKeys(monkeypatch, root)
    assert media._set_query(field, "505", root) is False
    assert keys.typed == [] and keys.keys == [] and "AXValue" not in field.attrs


def test_the_whole_text_is_an_accessibility_range_in_utf16_units(monkeypatch):
    AS = pytest.importorskip("ApplicationServices")
    monkeypatch.setattr(media, "_attr", FakeAX.attr)
    rng = media._whole_range(El("AXTextField", AXValue="naïve 🎵"))
    ok, span = AS.AXValueGetValue(rng, AS.kAXValueCFRangeType, None)
    assert ok and tuple(span) == (0, 8)  # (the note is two UTF-16 units)


# ---------------------------------------------------------------- learned item URIs
async def test_learned_uri_miss_then_learn_then_hit(audio, osa, engine):
    app = make_app(uri_dict(), own_schemes=("fixtureplayer",))
    miss = await media.play("505", app, "learned_uri")
    assert not miss.ok and miss.reason == "not_found" and not miss.blame and osa.calls == []

    # a confirmed play (any method) teaches the app's own id for this request
    osa.now_playing = (0, "505\nThe Fixtures\nfixtureplayer:track:42", "")
    async with media._Watch([app]) as watch:
        audio.app_plays()
        learned = await media._confirmed(app, "505", watch, 0.5, "505", None)
    assert learned.ok and learned.text == "Playing 505 by The Fixtures in Fixture Player"
    assert media._uri_lookup(BID, "  505! ")["uri"] == "fixtureplayer:track:42"

    audio.clients.clear()
    osa.on("«event fixrPCtx»", lambda: (audio.app_plays(), (0, "OK", ""))[1])
    hit = await media.play("505", app, "learned_uri")
    assert hit.ok, hit
    (_script, args), = osa.scripts_with("«event fixrPCtx»")
    assert args == [BID, "fixtureplayer:track:42"]


async def test_learned_uri_that_no_longer_plays_is_forgotten(audio, osa, engine):
    app = make_app(uri_dict(), own_schemes=("fixtureplayer",))
    media._uri_store(BID, "505", "fixtureplayer:track:42", "505")
    osa.on("«event fixrPCtx»", (0, "OK", ""))  # accepted, but no sound
    out = await media.play("505", app, "learned_uri")
    assert not out.ok and media._uri_lookup(BID, "505") is None


async def test_nothing_is_learned_when_something_else_plays(audio, osa, engine):
    app = make_app(uri_dict(), own_schemes=("fixtureplayer",))
    osa.now_playing = (0, "An Advert\n\nfixtureplayer:ad:1", "")
    async with media._Watch([app]) as watch:
        audio.app_plays()
        await media._confirmed(app, "505", watch, 0.5, "505", None)
    assert media._uri_lookup(BID, "505") is None


# ---------------------------------------------------------------- confirmation by sound
def test_attribution_counts_the_apps_helpers_but_never_neo(audio):
    audio.start(HELPER_PID, BID + ".helper")  # the app's audio helper (responsible pid = the app)
    audio.start(901, "com.fixture.neo.gpu")  # NEO's own web view speaking
    audio.start(NEO_PID, "com.fixture.neo")  # NEO's backend (TTS)
    audio.start(700, "com.fixture.other")
    audio.bids[700] = "com.fixture.other"
    outs = media.AudioWatcher().outputs()
    assert outs == {BID: {HELPER_PID}, "com.fixture.other": {700}}
    procs = media.AudioWatcher().playing()
    app = make_app(None)
    assert [p.pid for p in procs if media._belongs(p, app)] == [HELPER_PID]
    # no responsibility info: the process's own path inside the app bundle still attributes it
    audio.start(555, "")
    audio.paths[555] = app.path + "/Contents/Frameworks/Fixture Helper.app/Contents/MacOS/Fixture Helper"
    assert any(p.pid == 555 and media._belongs(p, app) for p in media.AudioWatcher().playing())


async def test_watch_sees_new_streams_only(audio):
    app = make_app(None)
    async with media._Watch([app]) as watch:
        audio.start(901, "com.fixture.neo.gpu")  # NEO talking during the attempt
        assert await watch.wait(0.1) == "silent"
    async with media._Watch([app]) as watch:
        audio.app_plays()
        assert await watch.wait(0.5) == "started"


async def test_watch_counts_a_restarted_stream_of_an_app_already_playing(audio):
    app = make_app(None)
    obj = audio.start(HELPER_PID, BID + ".helper")
    async with media._Watch([app]) as watch:
        assert await watch.wait(0.05) == "unknown"  # still the old stream
        audio.clients[obj][2] = False  # the track switch stops the stream...
        await asyncio.sleep(0.05)
        audio.clients[obj][2] = True  # ...and a new one starts
        assert await watch.wait(0.5) == "started"


# ---------------------------------------------------------------- the web fallback
def _rows():
    return [
        {"content": "https://www.a.example/watch?v=1", "title": "505 - 1 hour loop", "duration": "47:57"},
        {"content": "https://www.b.example/watch?v=2", "title": "505 (Official Video)", "duration": "4:13"},
        {"content": "http://127.0.0.1/evil", "title": "505", "duration": "3:00"},
    ]


async def test_web_fallback_opens_a_fitting_video_and_learns_the_host(monkeypatch, audio, engine):
    opened = []

    async def open_app(url):
        opened.append(url)
        audio.browser_plays()
        return f"Opened {url} in Fixture Browser"

    monkeypatch.setattr(apps_mod, "open_app", open_app)
    monkeypatch.setattr(media, "_video_search", lambda q, n=12: _rows())
    out = await media.play("505", None, "web")
    assert out.ok and out.text == "Playing 505 (Official Video) in Fixture Browser", out
    assert opened == ["https://www.b.example/watch?v=2"]  # not the 48-minute loop, not a private host
    assert ("play_media", "web:b.example", "web", True, "") in engine.records


async def test_web_prefers_a_host_that_worked_here(monkeypatch, audio, engine):
    rows = [
        {"content": "https://c.example/v/1", "title": "505", "duration": "4:10"},
        {"content": "https://b.example/v/2", "title": "505", "duration": "4:13"},
    ]
    engine.priors = {"web:b.example": 0.9, "web:c.example": 0.3}
    assert media._pick_video(rows, "505")[0] == "https://b.example/v/2"
    engine.priors = {}
    assert media._pick_video(rows, "505")[0] == "https://c.example/v/1"  # no history: the top result


async def test_web_offline_is_not_blamed(monkeypatch, audio, engine):
    def down(q, n=12):
        raise OSError("network is unreachable")

    monkeypatch.setattr(media, "_video_search", down)
    out = await media.play("505", None, "web")
    assert out.reason == "offline" and not out.blame


# ---------------------------------------------------------------- play_best: the whole loop
async def test_play_best_moves_to_the_next_source_when_the_library_has_nothing(monkeypatch, audio, osa, engine):
    app = make_app(library_dict())
    engine.cands = [
        Candidate(app, 0.9, 0.0, "", ("script_search_play", "ax_search_play")),
        Candidate(None, 0.5, 0.0, "fallback: web", ("web",)),
    ]
    osa.on("«event hookSrch»", (0, "EMPTY", ""))
    root, target, _ = player_window(audio)
    FakeAX(monkeypatch, root)

    async def open_app(url):
        audio.browser_plays()
        return "Opened"

    monkeypatch.setattr(apps_mod, "open_app", open_app)
    monkeypatch.setattr(media, "_video_search", lambda q, n=12: _rows())
    out = await media.play_best("505", text="neo play 505")
    assert out == "Playing 505 (Official Video) in Fixture Browser"
    assert engine.chosen_with == [("play_media", "neo play 505", "neo play 505")]
    assert engine.records[0] == ("play_media", BID, "script_search_play", False, "empty")  # blamed: it learns
    assert [r[2] for r in engine.records if r[1] == BID] == ["script_search_play"]  # its search box: skipped
    assert target.pressed == 0


async def test_only_the_users_own_words_go_on_as_theirs(monkeypatch):
    from neo2.core.agent.registry import ToolContext, registry
    from neo2.core.tools import media as _tool  # noqa: F401 - registers play_music

    got = []

    async def play_best(query, text=""):
        got.append((query, text))
        return "Playing"

    monkeypatch.setattr(media, "play_best", play_best)
    args = {"query": "bohemian rhapsody"}
    # Gemini Live before the user's words arrive: its arguments stand in for them
    await registry().invoke("play_music", args, ToolContext(user_text=str(args)))
    await registry().invoke("play_music", args, ToolContext())
    await registry().invoke("play_music", args, ToolContext(user_text="neo play bohemian rhapsody"))
    assert got == [
        ("bohemian rhapsody", ""),
        ("bohemian rhapsody", ""),
        ("bohemian rhapsody", "neo play bohemian rhapsody"),
    ]


async def test_play_best_decides_by_the_query_when_the_words_are_unknown(monkeypatch, audio, engine):
    engine.cands = [Candidate(None, 0.5, 0.0, "fallback: web", ("web",))]

    def down(q, n=12):
        raise OSError("offline")

    monkeypatch.setattr(media, "_video_search", down)
    await media.play_best("505")
    assert engine.chosen_with == [("play_media", "505", "")]  # the query decides; nobody said it


async def test_play_best_searches_the_named_app_for_the_words_only(monkeypatch, audio, osa, engine):
    app = make_app(library_dict())
    engine.cands = [Candidate(app, 1.0, 0.0, "named by the user", ("script_search_play",))]
    monkeypatch.setattr(
        media, "_inventory", lambda: SimpleNamespace(named_in=lambda t, apps=None: app, default_for=lambda s: BROWSER)
    )
    osa.on("«event hookSrch»", lambda: (audio.app_plays(), (0, "OK", ""))[1])
    out = await media.play_best("505 on Fixture Player", text="play 505 on fixture player")
    assert out.startswith("Playing") and out.endswith("in Fixture Player")
    (_s, args), = osa.scripts_with("«event hookSrch»")
    assert args == [BID, "505"]


async def test_play_best_tries_everything_in_order_and_reports_what_it_tried(monkeypatch, audio, osa, engine):
    lib = make_app(library_dict())
    other = make_app(None, bundle_id="com.fixture.other", name="Other Player", path="/Applications/Other Player.app")
    engine.cands = [
        Candidate(lib, 0.9, 0.0, "", ("script_search_play",)),
        Candidate(other, 0.8, 0.0, "", ("ax_search_play",)),
        Candidate(None, 0.5, 0.0, "fallback: web", ("web",)),
    ]
    osa.on("«event hookSrch»", (0, "EMPTY", ""))
    root, _t, _ = player_window(audio, with_search=False)
    FakeAX(monkeypatch, root)

    def down(q, n=12):
        raise OSError("offline")

    monkeypatch.setattr(media, "_video_search", down)
    out = await media.play_best("505")
    assert out.startswith("Error: couldn't play 505 - tried Fixture Player (scripting:"), out
    assert out.index("Fixture Player") < out.index("Other Player (search box:") < out.index("the web (web:")
    assert [r[1:4] for r in engine.records] == [
        (BID, "script_search_play", False),
        ("com.fixture.other", "ax_search_play", False),
        ("web", "web", False),
    ]
    assert engine.records[-1][4] == "offline"


async def test_play_best_asks_the_user_when_the_page_opened_but_is_silent(monkeypatch, audio, engine):
    engine.cands = [Candidate(None, 0.5, 0.0, "fallback: web", ("web",))]

    async def open_app(url):
        return f"Opened {url} in Fixture Browser"  # no sound follows (autoplay blocked)

    monkeypatch.setattr(apps_mod, "open_app", open_app)
    monkeypatch.setattr(media, "_video_search", lambda q, n=12: _rows())
    out = await media.play_best("505")
    assert out.startswith("NEEDS_USER: I opened 505 (Official Video) in Fixture Browser") and " - " in out
    assert ("play_media", "web:b.example", "web", False, "timeout") in engine.records
    assert ("play_media", "web", "web", False, "timeout") in engine.records


async def test_play_best_asks_for_permission_when_that_is_what_blocked_it(monkeypatch, audio, osa, engine):
    engine.cands = [Candidate(make_app(library_dict()), 0.9, 0.0, "", ("script_search_play",))]
    osa.on("«event hookSrch»", (1, "", "Not authorized to send Apple events to Fixture Player. (-1743)"))

    def down(q, n=12):
        raise OSError("offline")

    monkeypatch.setattr(media, "_video_search", down)
    out = await media.play_best("505")
    assert out.startswith("NEEDS_USER: Fixture Player needs your OK") and "Automation" in out
    assert engine.records[0] == ("play_media", BID, "script_search_play", False, "permission")


async def test_play_best_skips_learned_uri_when_nothing_was_learned(monkeypatch, audio, osa, engine):
    app = make_app(uri_dict(), own_schemes=("fixtureplayer",))
    engine.cands = [Candidate(app, 0.9, 0.0, "", ("learned_uri", "ax_search_play"))]
    root, _target, _ = player_window(audio)
    FakeAX(monkeypatch, root)
    out = await media.play_best("505")
    assert out.startswith("Playing") and out.endswith("in Fixture Player")
    assert [r[2] for r in engine.records] == ["ax_search_play"]  # a cache miss is not an attempt


async def test_play_best_stops_at_the_overall_budget(monkeypatch, audio, osa, engine):
    lib = make_app(library_dict())
    engine.cands = [Candidate(lib, 0.9, 0.0, "", ("script_search_play",)) for _ in range(5)]

    async def slow(script, args, timeout):
        await asyncio.sleep(10)
        return 0, "OK", ""

    monkeypatch.setattr(media, "_osascript", slow)
    monkeypatch.setattr(media, "OVERALL_S", 1.5)
    t0 = time.monotonic()
    out = await media.play_best("505")
    assert time.monotonic() - t0 < 3.0
    assert out.startswith("Error:") and "ran out of time" in out
    assert all(r[4] == "timeout" for r in engine.records)


# ---------------------------------------------------------------- review fixes
def test_search_container_follows_the_element_tree():
    d = library_dict()
    tree = Dictionary(
        commands=d.commands,
        classes={**d.classes, "source": ("cSrc", ("name",))},
        properties=d.properties,
        enums={},
        elements={"application": ("source",), "source": ("playlist",), "playlist": ("track",)},
    )
    sp = media._search_play_spec(tree)
    assert sp.container == ("cSrc", "cPly")
    assert "(«class cPly» 1 of «class cSrc» 1) given «class pTrm»:q" in media._search_play_script(sp)
    # 'search <application> for <text>': the application itself is searched - no direct object
    on_app = Dictionary(
        commands={**d.commands, "search": Command("search", "hookSrch", "application", (("for", "pTrm", "text"),))},
        classes=tree.classes,
        properties=d.properties,
        enums={},
        elements=tree.elements,
    )
    sp = media._search_play_spec(on_app)
    assert sp.container == ()
    assert "set hits to «event hookSrch» given «class pTrm»:q" in media._search_play_script(sp)


async def test_only_an_item_id_the_app_can_be_handed_back_is_learned(audio, osa, engine):
    d = uri_dict()
    no_play_uri = Dictionary(
        commands={"play": d.commands["play"]}, classes=d.classes, properties=d.properties, enums={}
    )
    app = make_app(no_play_uri, own_schemes=("fixtureplayer",), schemes=("fixtureplayer",))
    osa.now_playing = (0, "505\nThe Fixtures\nhttps://fixture.example/track/42", "")
    async with media._Watch([app]) as watch:
        audio.app_plays()
        assert (await media._confirmed(app, "505", watch, 0.5, "505", None)).ok
    assert media._uri_lookup(BID, "505") is None  # its own scheme can't open an https id
    audio.clients.clear()
    osa.now_playing = (0, "505\nThe Fixtures\nfixtureplayer:track:42", "")
    async with media._Watch([app]) as watch:
        audio.app_plays()
        assert (await media._confirmed(app, "505", watch, 0.5, "505", None)).ok
    assert media._uri_lookup(BID, "505")["uri"] == "fixtureplayer:track:42"


async def test_play_best_notices_a_late_start_after_its_last_attempt(monkeypatch, audio, osa, engine):
    lib = make_app(library_dict())
    engine.cands = [
        Candidate(lib, 0.9, 0.0, "", ("script_search_play",)),
        Candidate(None, 0.5, 0.0, "fallback: web", ("web",)),
    ]
    osa.on("«event hookSrch»", (0, "OK", ""))  # accepted; the sound comes after its confirm budget

    def late_and_offline(q, n=12):
        audio.app_plays()  # the player finally starts while the web attempt fails
        raise OSError("offline")

    monkeypatch.setattr(media, "_video_search", late_and_offline)
    out = await media.play_best("505")
    assert out == "Playing 505 in Fixture Player", out
    assert [r[1:5] for r in engine.records] == [
        (BID, "script_search_play", False, "audio_not_started"),
        ("web", "web", False, "offline"),
        (BID, "script_search_play", True, ""),
    ]


# ---------------------------------------------------------------- sites, browsers, unconfirmed, permission
def test_methods_follow_what_kind_of_media_app_it_is():
    browser = make_app(None, media_evidence=("web browser",), viewer_types=("public.html",))
    assert media.methods_for(browser) == ("web",)  # named browser: the web method, in it
    player = make_app(None, media_evidence=("media file player", "opens 12 audio/video types"), viewer_types=("public.mp3",))
    assert media.methods_for(player) == ("local_file",)  # no catalog: only the user's files


def test_results_count_as_on_the_named_site_by_host():
    assert media._on_site("youtube.com", "youtube") and media._on_site("music.youtube.com", "youtube music")
    assert media._on_site("vimeo.com", "vimeo.com") and media._on_site("player.vimeo.com", "vimeo.com")
    assert not media._on_site("b.example", "youtube") and not media._on_site("notvimeo.com", "vimeo.com")
    assert media._without_site("505 on vidzone", "vidzone") == "505"


async def test_web_on_a_named_site_opens_only_that_site(monkeypatch, audio, engine):
    rows = [
        {"content": "https://b.example/watch?v=2", "title": "505", "duration": "4:13"},
        {"content": "https://www.vidzone.example/v/9", "title": "505 (live)", "duration": "4:20"},
    ]
    searched, opened = [], []
    monkeypatch.setattr(media, "_video_search", lambda q, n=12: (searched.append(q), rows)[1])

    async def open_app(url):
        opened.append(url)
        audio.browser_plays()
        return "Opened"

    monkeypatch.setattr(apps_mod, "open_app", open_app)
    out = await media.play("505 on vidzone.example", None, "web", "vidzone.example")
    assert out.ok and opened == ["https://www.vidzone.example/v/9"]
    assert searched == ["505 vidzone.example"]
    missing = await media.play("505", None, "web", "elsewhere")
    assert missing.reason == "not_found" and not missing.blame


async def test_play_best_falls_back_to_the_open_web_when_the_site_has_nothing(monkeypatch, audio, osa, engine):
    site = Candidate(None, 0.5, 0.0, "", ("web",), site="elsewhere")
    engine.cands = [site]
    opened = []

    async def open_app(url):
        opened.append(url)
        audio.browser_plays()
        return "Opened"

    monkeypatch.setattr(apps_mod, "open_app", open_app)
    monkeypatch.setattr(media, "_video_search", lambda q, n=12: _rows())
    out = await media.play_best("505 on elsewhere")
    assert out.startswith("Playing 505 (Official Video)") and len(opened) == 1


async def test_a_named_browser_opens_the_page_itself(monkeypatch, audio, engine):
    other = App(bundle_id="com.fixture.other", name="Other Browser", path="/Applications/Other Browser.app",
                media_evidence=("web browser",))  # fmt: skip
    audio.bids[602] = "com.fixture.other"
    ran = []

    async def fake_open(args, timeout):
        ran.append(args)
        audio.start(602, "com.fixture.other")
        return 0, "", ""

    monkeypatch.setattr(media, "_open", fake_open)
    monkeypatch.setattr(media, "_video_search", lambda q, n=12: _rows())
    out = await media.play("505", other, "web")
    assert out.ok and out.text.endswith("in Other Browser")
    assert ran == [["-b", "com.fixture.other", "https://www.b.example/watch?v=2"]]


async def test_web_that_cannot_be_confirmed_asks_and_learns_nothing(monkeypatch, audio, engine):
    audio.browser_plays()  # already playing something: a new tab keeps the same stream

    async def open_app(url):
        return "Opened"

    monkeypatch.setattr(apps_mod, "open_app", open_app)
    monkeypatch.setattr(media, "_video_search", lambda q, n=12: _rows())
    engine.cands = [Candidate(None, 0.5, 0.0, "", ("web",))]
    out = await media.play_best("505")
    assert out.startswith("NEEDS_USER: I opened 505 (Official Video) in Fixture Browser")
    assert not [r for r in engine.records if r[1].startswith("web:")]  # no host success or failure
    assert engine.records == [("play_media", "web", "web", False, "unconfirmed")]


async def test_without_coreaudio_an_app_counts_only_by_its_now_playing(monkeypatch, audio, osa, engine):
    monkeypatch.setattr(media, "audio_confirmable", lambda: False)
    lib = make_app(library_dict())
    osa.now_playing = (0, "505\nFixture Band\n", "")
    ok = await media.play("505", lib, "script_search_play")
    assert ok.ok, ok
    osa.now_playing = (0, "Something Else\nOther\n", "")
    unsure = await media.play("505", lib, "script_search_play")
    assert unsure.reason == "unconfirmed" and not unsure.blame
    # play_best stops there: it may well be playing, so no second source on top of it
    engine.cands = [Candidate(lib, 0.9, 0.0, "", ("script_search_play",)), Candidate(None, 0.5, 0.0, "", ("web",))]
    out = await media.play_best("505")
    assert out.startswith("NEEDS_USER: I asked Fixture Player to play 505")
    assert [r[2] for r in engine.records] == ["script_search_play"]


async def test_macos_is_asked_once_before_scripting_and_a_no_moves_on(monkeypatch, audio, osa, engine):
    asked = []

    def automation(bid, ask=False):
        asked.append(ask)
        return media.AE_NOT_ASKED if not ask else media.AE_DENIED

    monkeypatch.setattr(media, "automation", automation)
    app = make_app(library_dict())
    engine.cands = [Candidate(app, 0.9, 0.0, "", ("script_search_play", "ax_search_play"))]
    root, _target, _ = player_window(audio)
    FakeAX(monkeypatch, root)
    out = await media.play_best("505")
    assert asked[:2] == [False, True]  # checked quietly, then macOS's own question
    assert not osa.scripts_with("«event hookSrch»")  # no script once the user said no
    assert out.startswith("Playing") and out.endswith("in Fixture Player")  # its search box still works
    assert engine.records[0] == ("play_media", BID, "script_search_play", False, "permission")


async def test_a_not_running_app_is_started_in_the_background_before_the_check(monkeypatch, audio, osa, engine):
    states = iter([media.AE_NOT_RUNNING, media.AE_NOT_RUNNING, media.AE_ALLOWED])
    monkeypatch.setattr(media, "automation", lambda bid, ask=False: next(states, media.AE_ALLOWED))
    opened = []

    async def fake_open(args, timeout):
        opened.append(args)
        return 0, "", ""

    monkeypatch.setattr(media, "_open", fake_open)
    assert await media._control_gate(make_app(library_dict())) is None
    assert opened == [["-g", "-b", BID]]


async def test_now_playing_never_makes_macos_ask(monkeypatch, osa):
    monkeypatch.setattr(media, "automation", lambda bid, ask=False: media.AE_NOT_ASKED)
    assert await media._now_playing(make_app(library_dict())) is None
    assert not osa.calls


async def test_the_erase_forgets_learned_items_and_a_request_in_flight_learns_nothing(monkeypatch, engine):
    from neo2.core import privacy

    media._uri_store(BID, "505", "fixtureplayer:track:1", "505")
    assert media._uri_lookup(BID, "505") is not None
    assert media.forget() == 1 and media._uri_lookup(BID, "505") is None

    media._gen.set(privacy.generation())
    monkeypatch.setattr(privacy, "_generation", privacy.generation() + 1)  # erased mid-request
    media._uri_store(BID, "505", "fixtureplayer:track:1", "505")
    media._record(BID, "script_search_play", True)
    assert media._uri_lookup(BID, "505") is None and engine.records == []


# ---------------------------------------------------------------- web first (the default)
@pytest.fixture
def web_first(monkeypatch, audio):
    from neo2.core import config

    monkeypatch.setenv("NEO_PLAY_DEFAULT", "web")
    monkeypatch.setenv("NEO_PLAY_SITE", "vidzone")
    monkeypatch.setattr(config, "_settings", None)

    async def host(word):
        return "vidzone.example" if word == "vidzone" else ""

    monkeypatch.setattr(media, "_resolve_site", host)
    monkeypatch.setitem(media.TIMEOUTS, "web", (6.0, 0.5, 0.3))  # room for all three searches
    opened = []

    async def open_app(url):
        opened.append(url)
        audio.browser_plays()
        return "Opened"

    monkeypatch.setattr(apps_mod, "open_app", open_app)
    return opened


def _vid_rows():
    return [
        {"content": "https://b.example/watch?v=1", "title": "505", "duration": "4:13"},
        {"content": "https://www.vidzone.example/watch?v=abc", "title": "505 (Official)", "duration": "4:14"},
    ]


async def test_play_goes_to_the_preferred_site_and_opens_no_app(monkeypatch, engine, osa, web_first):
    app = make_app(library_dict(), days_used_30=20)
    engine.cands = [Candidate(app, 0.9, 0.0, "used 20 of last 30 days", ("script_search_play",)),
                    Candidate(None, 0.5, 0.0, "fallback: web", ("web",))]
    monkeypatch.setattr(media, "_video_search", lambda q, n=12: _vid_rows())
    out = await media.play_best("505", text="play 505")
    assert out == "Playing 505 (Official) in Fixture Browser"
    assert web_first == ["https://www.vidzone.example/watch?v=abc"]
    assert osa.calls == []  # no app was scripted, launched or even asked


async def test_a_refused_video_search_falls_back_to_a_site_search(monkeypatch, engine, web_first):
    engine.cands = [Candidate(None, 0.5, 0.0, "fallback: web", ("web",))]

    def refused(q, n=12):
        raise RuntimeError("202 Ratelimit")

    monkeypatch.setattr(media, "_video_search", refused)
    searched = []

    def text(q, n=10):
        searched.append(q)
        return [{"href": "https://vidzone.example/", "title": "home"}, {"href": "https://vidzone.example/watch?v=xyz", "title": "505"}]

    monkeypatch.setattr(media, "_text_search", text)
    from neo2.core.decide import sites

    async def no_templates(host, prefix=""):
        return []

    monkeypatch.setattr(sites, "discover", no_templates)  # the site publishes no search
    out = await media.play_best("505", text="play 505")
    assert out.startswith("Playing 505") and web_first == ["https://vidzone.example/watch?v=xyz"]
    assert searched == ["505 site:vidzone.example"]


async def test_with_both_searches_down_the_sites_own_search_page_finds_it(monkeypatch, engine, web_first):
    from neo2.core.decide import sites

    engine.cands = [Candidate(None, 0.5, 0.0, "fallback: web", ("web",))]

    def down(*a, **k):
        raise OSError("offline")

    monkeypatch.setattr(media, "_video_search", down)
    monkeypatch.setattr(media, "_text_search", down)
    monkeypatch.setattr(sites, "learned_template", lambda host, prefix="": sites.Template("https://vidzone.example/results?q={searchTerms}", "learned"))
    fetched = []

    async def fetch(url, limit=0, partial=False):
        fetched.append(url)
        html = '<a href="/results?q=x">more</a> {"url":"/watch?v=first\\u0026pp=1"} <a href="https://other.example/watch?v=no">x</a>'
        return sites._Page(url, html)

    monkeypatch.setattr(sites, "_fetch", fetch)
    out = await media.play_best("505", text="play 505")
    assert out.startswith("Playing") and web_first == ["https://vidzone.example/watch?v=first"]
    assert fetched == ["https://vidzone.example/results?q=505"]


async def test_an_app_the_user_named_goes_first_and_a_fallback_is_said(monkeypatch, engine, osa, web_first):
    app = make_app(library_dict())
    engine.cands = [Candidate(app, 1.0, 0.0, "named by the user", ("script_search_play",)),
                    Candidate(None, 0.5, 0.0, "fallback: web", ("web",))]
    osa.on("«event hookSrch»", (1, "", "Not authorized to send Apple events to Fixture Player. (-1743)"))
    monkeypatch.setattr(media, "_video_search", lambda q, n=12: _vid_rows())
    out = await media.play_best("505 on Fixture Player", text="play 505 on Fixture Player")
    assert out.say is True and out.ok
    assert out.text.startswith("Playing 505 (Official) in Fixture Browser (Fixture Player couldn't:")


def test_item_links_are_single_items_on_the_site_in_page_order():
    html = ('<a href="/">home</a> {"url":"/watch?v=a1\\u0026list=x"} <a href="//vidzone.example/watch?v=b2">b</a>'
            ' <a href="https://evil.example/watch?v=c3">c</a> <a href="/channel/abc">ch</a> "url":"/watch?v=a1"')
    assert media._item_links(html, "vidzone.example") == [
        "https://vidzone.example/watch?v=a1", "https://vidzone.example/watch?v=b2"]
