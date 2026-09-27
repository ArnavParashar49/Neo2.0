"""Nothing the suite (or a scratch script run with NEO_NO_EFFECTS=1) runs can act on the Mac.

neo/effects.py refuses every real effect at the lowest layer of its path. Here each choke point is
driven with nothing faked above it, its lowest primitive swapped for a sentinel that fails the test
if reached: the effect must be refused before it. The same paths in NEO.app's mode (not under
pytest, NEO_NO_EFFECTS unset) reach their sentinel: nothing is refused there, and reaching it shows
each path really leads to the primitive it guards.

A sentinel standing where a real PyObjC, ctypes or sounddevice API belongs poses as that API (its
module says so): a test's own fake of the primitive is let through on purpose, see
test_a_tests_own_fake_of_the_primitive_is_let_through.
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import weakref
import webbrowser
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from neo import effects
from neo.agent.registry import ToolContext


class Reached(BaseException):
    """A sentinel ran: the real effect would have happened here. (A BaseException, so no
    `except Exception` on the way can hide it.)"""


# Read-only programs a long path may run on its way (the incident test lets them through).
_PROBES = frozenset({"lsappinfo", "mdfind", "mdls", "ioreg", "sw_vers", "uname", "file", "sysctl"})


class Sentinels:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self.mp, self.tmp = monkeypatch, tmp_path
        self.hit: list[str] = []
        self._after: list[Callable[[], None]] = []
        self._servers: list[socket.socket] = []
        self._stop = threading.Event()

    def fn(self, name: str, *, poses_as: str = "") -> Callable:
        def reached(*a, **k):
            self.hit.append(name)
            raise Reached(name)

        if poses_as:
            reached.__module__ = poses_as  # the real API's module: effects must take it for the real one
        return reached

    def cls(self, name: str, *, poses_as: str) -> type:
        """A class posing as a real framework class (PyObjC's alloc, or a sounddevice stream)."""
        hit = self.fn(name)
        return type(
            name,
            (),
            {"__module__": poses_as, "__init__": lambda self, *a, **k: hit(), "alloc": staticmethod(hit)},
        )

    def put(self, obj: object, name: str, *, poses_as: str = "") -> None:
        self.mp.setattr(obj, name, self.fn(f"{getattr(obj, '__name__', obj)}.{name}", poses_as=poses_as))

    def spawn(self, *, probes: bool = False) -> None:
        """Every way Python starts a program, one level under the audit event. ``probes``: the
        read-only ones still run."""
        real_fork, real_spawn = subprocess._fork_exec, os.posix_spawn

        def fork(args, executables, *rest):
            if probes and os.path.basename(os.fsdecode(args[0])) in _PROBES:
                return real_fork(args, executables, *rest)
            self.hit.append(f"fork_exec {[os.fsdecode(a) for a in args]}")
            raise Reached("fork_exec")

        def posix_spawn(path, argv, *rest, **kw):
            if probes and os.path.basename(os.fsdecode(path)) in _PROBES:
                return real_spawn(path, argv, *rest, **kw)
            self.hit.append(f"posix_spawn {os.fsdecode(path)}")
            raise Reached("posix_spawn")

        self.mp.setattr(subprocess, "_fork_exec", fork)
        self.mp.setattr(os, "posix_spawn", posix_spawn)
        self.put(os, "posix_spawnp")

    def skylight(self) -> None:
        """The WindowServer's private calls behind bringing an app to the front."""
        from neo.tools.computer import focus

        sl = SimpleNamespace(
            _SLPSSetFrontProcessWithOptions=self.fn("_SLPSSetFrontProcessWithOptions", poses_as="ctypes"),
            SLPSPostEventRecordTo=self.fn("SLPSPostEventRecordTo", poses_as="ctypes"),
        )
        hi = SimpleNamespace(GetProcessForPID=lambda pid, psn: 0)  # a read: which process that is
        self.mp.setattr(focus, "_load", lambda: (sl, hi))

    def network(self) -> None:
        """Listeners on both loopbacks count every connection that arrives, and every IP address
        is taken for one off this Mac."""
        for host, family in (("127.0.0.1", socket.AF_INET), ("::1", socket.AF_INET6)):
            srv = socket.create_server((host, 0), family=family)
            srv.settimeout(0.05)
            self._servers.append(srv)
            threading.Thread(target=self._accept, args=(srv,), daemon=True).start()
        self.url4 = f"http://127.0.0.1:{self._servers[-2].getsockname()[1]}"
        self.url6 = f"http://[::1]:{self._servers[-1].getsockname()[1]}"
        self.mp.setattr(effects, "_outside", lambda sock, address: isinstance(address, tuple))
        self._after.append(lambda: time.sleep(0.1))  # a connection on its way lands first

    def search(self) -> None:
        """Every web search's HTTP client: primp, written in Rust (no Python socket under it, so the
        audit hook never sees its connections)."""
        import primp

        hit = self.fn("primp.Client.request")

        class Client:  # as ddgs makes and uses one
            def __init__(self, *a, **k):
                pass

            def headers_update(self, headers):
                pass

            def request(self, *a, **k):
                hit()

            get = post = request

        self.mp.setattr(primp, "Client", Client)

    def _accept(self, srv: socket.socket) -> None:
        while not self._stop.is_set():
            try:
                conn, peer = srv.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            self.hit.append(f"connection from {peer[0]}")
            conn.close()

    def after(self, check: Callable[[], None]) -> None:
        self._after.append(check)

    def reached(self) -> list[str]:
        for check in self._after:
            check()
        return self.hit

    def close(self) -> None:
        self._stop.set()
        for srv in self._servers:
            srv.close()


@pytest.fixture
def s(monkeypatch, tmp_path):
    sentinels = Sentinels(monkeypatch, tmp_path)
    yield sentinels
    sentinels.close()


@pytest.fixture
def neo_app(monkeypatch):
    """NEO.app's mode (NEO_NO_EFFECTS unset, effects on as if not under pytest) for this test's own
    thread and what it starts from there (its asyncio tasks, asyncio.to_thread). Every other thread
    (one an earlier test left running) keeps effects off."""
    monkeypatch.delenv("NEO_NO_EFFECTS", raising=False)
    token = effects._as_app.set(True)
    assert effects.on()
    yield
    effects._as_app.reset(token)


def _refusal(e: BaseException | None, depth: int = 0) -> effects.EffectRefused | None:
    """The refusal inside what a path raised: as is, in a group (anyio's connect), or as a cause."""
    if e is None or depth > 8:
        return None
    if isinstance(e, effects.EffectRefused):
        return e
    for sub in getattr(e, "exceptions", ()):
        if found := _refusal(sub, depth + 1):
            return found
    return _refusal(e.__cause__ or e.__context__, depth + 1)


# ---- the choke points: each arranges its sentinels and returns the action --------------------------

CASES: dict[str, tuple[str, Callable[[Sentinels], Callable[[], object]], bool]] = {}


def case(kind: str, *, app: bool = True):
    """A choke point refused as ``kind``. ``app``: also run in NEO.app's mode (not for a model
    provider's client, which retries a dropped connection for seconds there, nor for a path that
    reaches its primitive from a thread of its own, where NEO.app's mode doesn't reach)."""

    def register(arrange):
        CASES[arrange.__name__] = (kind, arrange, app)
        return arrange

    return register


# programs: the audit hook, under subprocess / asyncio / os.system / os.popen


@case("process")
def open_an_app(s):
    from neo.tools.computer import apps

    s.spawn()
    return lambda: asyncio.run(apps.open_app("Safari"))


@case("process")
def open_a_web_page(s):
    from neo.tools.computer import apps

    s.spawn()
    return lambda: asyncio.run(apps.open_url("https://example.com"))


@case("process")
def reveal_a_file_in_finder(s):
    from neo.tools.computer import apps

    s.spawn()
    return lambda: asyncio.run(apps.finder_reveal(str(s.tmp)))


@case("process")
def run_applescript(s):
    from neo.tools.computer import apps

    s.spawn()
    return lambda: asyncio.run(apps.osascript('tell application "Finder" to activate'))


@case("process")
def run_applescript_with_arguments(s):
    from neo.decide import media

    s.spawn()
    return lambda: asyncio.run(media._osascript("on run argv\nreturn item 1 of argv\nend run", ["x"], 5))


@case("process")
def open_a_file_in_the_app_that_imports_it(s):
    from neo.decide.kinds import schedule

    s.spawn()
    return lambda: asyncio.run(schedule._open_with("com.apple.iCal", str(s.tmp / "event.ics")))


@case("process")
def mute_the_mac(s):
    from neo.tools import system

    s.spawn()
    return lambda: asyncio.run(system.volume({"action": "mute"}, ToolContext()))


@case("process")
def sleep_the_display(s):
    from neo.tools import system

    s.spawn()
    return lambda: asyncio.run(system.sleep_display({}, ToolContext()))


@case("process")
def a_shell_command(s):
    from neo.tools.computer import shell

    s.spawn()
    return lambda: asyncio.run(shell.run("ls"))  # read-only, so no confirmation: straight to the shell


@case("process")
def capture_a_window(s):
    from neo.tools.computer import screen

    s.mp.setattr(screen, "can_capture", lambda: True)
    s.spawn()
    return lambda: screen.capture_window(1)


@case("process")
def say_it_aloud(s):
    from neo import app

    s.spawn()
    return lambda: app._say_aloud("hello")


@case("process")
def the_default_browser(s):
    s.mp.delenv("BROWSER", raising=False)
    s.mp.setattr(webbrowser, "_tryorder", None)  # found afresh: macOS's default browser, via osascript
    s.mp.setattr(webbrowser, "_browsers", {})
    s.spawn()
    return lambda: webbrowser.open("https://example.com")


@case("process")
def a_shell_through_os_system(s):
    marker = s.tmp / "ran"  # os.system has nothing in Python under its audit event: a harmless command
    s.after(lambda: marker.exists() and s.hit.append("os.system"))
    return lambda: os.system(f"touch '{marker}'")


@case("model")
def start_the_local_model_server(s):
    from neo.providers import local_server

    async def down(port=8080):
        return False

    s.mp.setattr(local_server, "is_up", down)
    s.spawn()
    return lambda: asyncio.run(local_server.ensure_running(wait_s=0.1))


@case("browser")
def start_the_page_reader(s):
    import playwright.async_api

    from neo.tools import render

    s.put(playwright.async_api, "async_playwright")
    return lambda: asyncio.run(render._start_playwright())


@case("browser")
def start_neos_browser(s):
    import playwright.async_api

    from neo.tools import browser

    s.put(playwright.async_api, "async_playwright")
    return lambda: asyncio.run(browser._Browser()._launch())


# the network: the socket, under every model provider's client and the web tools


def _ask(provider) -> Callable[[], object]:
    from neo.providers.base import Message

    return lambda: asyncio.run(asyncio.wait_for(provider.complete([Message("user", "hi")]), 30))


@case("network", app=False)
def the_openai_client(s):  # Groq's (and the local server's) protocol
    from neo.providers.openai_compat import OpenAICompatProvider

    s.network()
    return _ask(OpenAICompatProvider(name="groq", base_url=s.url6 + "/v1", api_key="test", model="m"))


@case("network", app=False)
def the_anthropic_client(s):
    from neo import config
    from neo.providers.claude import ClaudeProvider

    s.network()
    s.mp.setenv("ANTHROPIC_API_KEY", "test")
    s.mp.setenv("ANTHROPIC_BASE_URL", s.url4)
    s.mp.setattr(config, "_settings", None)
    return _ask(ClaudeProvider())


@case("network", app=False)
def the_gemini_client(s):
    from neo import config
    from neo.providers.gemini import GeminiProvider

    s.network()
    s.mp.setenv("GEMINI_API_KEY", "test")
    s.mp.setenv("GOOGLE_GEMINI_BASE_URL", s.url4)
    s.mp.setattr(config, "_settings", None)
    return _ask(GeminiProvider())


@case("network", app=False)
def a_gemini_live_session(s):
    from google import genai

    s.network()
    s.mp.setenv("GOOGLE_GEMINI_BASE_URL", s.url4)
    client = genai.Client(api_key="test")  # as neo/voice/live.py makes it

    async def connect():
        async with client.aio.live.connect(model="m", config={}):
            pass

    return lambda: asyncio.run(asyncio.wait_for(connect(), 30))


@case("network")
def the_web_tools_http_client(s):
    import httpx

    s.network()

    async def get():
        async with httpx.AsyncClient() as c:
            return await c.get(s.url4)

    return lambda: asyncio.run(get())


@case("network")
def a_download(s):
    import urllib.request

    s.network()
    return lambda: urllib.request.urlopen(s.url4, timeout=5)


# the network under every web search: ddgs's HTTP client, primp, has no Python socket. ddgs asks its
# engines from its own worker threads, where NEO.app's mode (a test's own thread) doesn't reach:
# test_neo_app_searches_the_web drives the same door in that mode.


@case("network", app=False)
def a_web_search(s):
    from neo.tools import web

    s.search()
    return lambda: web._one("despacito", 3, "duckduckgo")


@case("network", app=False)
def a_news_search(s):
    from neo.tools import web

    s.search()
    return lambda: web._news("despacito", 3)


@case("network", app=False)
def a_video_search(s):
    from neo.decide import media

    s.search()
    return lambda: media._video_search("despacito", 3)


@case("network", app=False)
def a_text_search(s):
    from neo.decide import media

    s.search()
    return lambda: media._text_search("despacito", 3)


@case("network", app=False)
def a_site_search(s):
    from neo.decide import sites

    s.search()
    return lambda: sites._search("youtube", "us-en")


# models: gigabytes read into memory, guard() before each load


def _fake_laya(s):
    s.mp.setitem(sys.modules, "laya", SimpleNamespace(load=s.fn("laya.load", poses_as="laya")))


@case("model")
def load_laya_for_a_question(s):
    from neo.reflex import zeroshot

    _fake_laya(s)
    return lambda: zeroshot._open("/nowhere/typed-decisions")


@case("model")
def load_laya_for_the_reflex(s):
    from neo import models
    from neo.reflex import features

    _fake_laya(s)
    s.mp.setattr(models, "laya_dir", lambda encoder="english": "/nowhere/laya")  # never the real cache
    return lambda: features.load_encoder("english", "cpu")


@case("model")
def load_the_on_device_voice(s):
    from neo import models
    from neo.voice import local

    s.mp.setitem(sys.modules, "mlx_audio.stt.utils", SimpleNamespace(load=s.fn("stt_load", poses_as="mlx_audio")))
    s.mp.setitem(
        sys.modules, "mlx_audio.tts.utils", SimpleNamespace(load_model=s.fn("tts_load", poses_as="mlx_audio"))
    )
    s.mp.setattr(models, "local_path", lambda repo: repo)  # never the real cache
    s.mp.setattr(local, "WakeSpotter", lambda rate: None)  # the wake word's own small model: not this case

    async def heard(text):
        pass

    return local.LocalVoice(heard)._load_models


@case("model")
def load_the_memory_embedder(s):
    from neo.memory import embed

    model = s.cls("SentenceTransformer", poses_as="sentence_transformers")
    s.mp.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=model))
    s.mp.setattr(embed, "_model", None)
    s.mp.setattr(embed, "_failed", False)
    return embed._get


# macOS calls with no program or socket under them: guard() before each


@case("input")
def a_click(s):
    import Quartz

    from neo.tools.computer import input as inp

    s.put(Quartz, "CGEventPost", poses_as="objc")
    return lambda: inp.click(5, 5)


@case("input")
def typing_into_another_app(s):
    import Quartz

    from neo.decide import axkit

    s.put(Quartz, "CGEventPostToPid", poses_as="objc")
    return lambda: axkit._type(os.getpid(), "hi", "return")


@case("input")
def return_in_another_app(s):
    import Quartz

    from neo.decide import media

    s.put(Quartz, "CGEventPostToPid", poses_as="objc")
    return lambda: media._post_return(os.getpid())


@case("accessibility")
def an_ax_value_set(s):
    import ApplicationServices

    from neo.decide import media

    s.put(ApplicationServices, "AXUIElementSetAttributeValue", poses_as="objc")
    return lambda: media._set(object(), "AXValue", "hi")


@case("accessibility")
def an_ax_press(s):
    import ApplicationServices

    from neo.decide import media

    s.put(ApplicationServices, "AXUIElementPerformAction", poses_as="objc")
    return lambda: media._perform(object(), "AXPress")


def _element():
    from neo.tools.computer import ax

    return ax.Element("e1", "AXButton", "OK", "", 0, 0, 20, 20, actions=["AXPress"], ref=object())


@case("accessibility")
def a_press_by_id(s):
    import ApplicationServices

    from neo.tools.computer import ax

    s.mp.setattr(ax, "_last_tree", {"e1": _element()})
    s.put(ApplicationServices, "AXUIElementPerformAction", poses_as="objc")
    return lambda: ax.press("e1")


@case("accessibility")
def a_value_set_by_id(s):
    import ApplicationServices

    from neo.tools.computer import ax

    s.mp.setattr(ax, "_last_tree", {"e1": _element()})
    s.put(ApplicationServices, "AXUIElementSetAttributeValue", poses_as="objc")
    return lambda: ax.set_value("e1", "hi")


@case("accessibility")
def a_press_of_the_element_in_hand(s):
    import ApplicationServices

    from neo.tools.computer import ax

    s.put(ApplicationServices, "AXUIElementPerformAction", poses_as="objc")
    return lambda: ax.press_element(_element())


@case("focus")
def the_windowserver_front_switch(s):
    from neo.tools.computer import focus

    s.skylight()
    return lambda: focus.force_front(os.getpid())


@case("focus")
def an_app_activated(s):
    from neo.tools.computer import apps, focus

    app = SimpleNamespace(
        localizedName=lambda: "Sentinel",
        processIdentifier=lambda: 424242,
        activateWithOptions_=s.fn("activateWithOptions_", poses_as="objc"),
    )

    async def never_front(pid, seconds):
        return False

    s.mp.setattr(apps, "_running", lambda name: app)  # reads: which app, and is it in front
    s.mp.setattr(apps, "_wait_front", never_front)
    s.mp.setattr(focus, "force_front", lambda pid: False)  # the WindowServer switch didn't take
    s.spawn()  # and the AppleScript `activate` after it
    return lambda: asyncio.run(apps.bring_front("Sentinel"))


@case("permission")
def the_screen_recording_question(s):
    import Quartz

    from neo.tools.computer import screen

    s.mp.setattr(screen, "can_capture", lambda: False)
    s.put(Quartz, "CGRequestScreenCaptureAccess", poses_as="objc")
    return screen.capture


@case("permission")
def the_accessibility_question(s):
    import ApplicationServices

    from neo import permissions

    s.mp.setattr(permissions, "missing", lambda: ["Accessibility"])
    s.mp.delenv("NEO_LAUNCHED_BY_APP", raising=False)
    s.put(ApplicationServices, "AXIsProcessTrustedWithOptions", poses_as="objc")
    return permissions.request_missing


@case("permission")
def the_automation_question(s):
    from neo.decide import media

    lib = SimpleNamespace(
        AECreateDesc=lambda *a: 0,
        AEDisposeDesc=lambda *a: 0,
        AEDeterminePermissionToAutomateTarget=s.fn(
            "AEDeterminePermissionToAutomateTarget", poses_as="ctypes"
        ),
    )
    s.mp.setattr(media, "_ae", lambda: lib)
    return lambda: media.automation("com.example.sentinel", ask=True)


@case("calendar")
def the_calendar_store(s):
    from neo.decide import eventkit

    s.mp.setattr(eventkit, "_store_obj", None)
    s.mp.setattr(
        eventkit, "_ek", lambda: SimpleNamespace(EKEventStore=s.cls("EKEventStore", poses_as="objc"))
    )
    return eventkit._the_store


@case("contacts")
def the_contacts_store(s):
    from neo.decide import people

    s.mp.setattr(
        people, "_cn", lambda: SimpleNamespace(CNContactStore=s.cls("CNContactStore", poses_as="objc"))
    )
    return people._store


@case("audio")
def the_speaker(s):
    from neo.voice import audio

    s.mp.setattr(audio.sd, "OutputStream", s.cls("OutputStream", poses_as="sounddevice"))
    return audio.Speaker().start


@case("audio")
def the_microphone(s):
    from neo.voice import audio

    s.mp.setattr(audio.sd, "RawInputStream", s.cls("RawInputStream", poses_as="sounddevice"))
    return audio.Mic().start


@case("files")
def a_note_file_in_the_users_documents(s):
    from neo import config
    from neo.decide.kinds import note
    from neo.decide.run import Req

    s.mp.delenv("NEO_NOTES_DIR", raising=False)  # the release default: ~/Documents/NEO Notes
    s.mp.setattr(config, "_settings", None)
    s.put(note, "_free_path")  # the folder's first touch
    req = Req("write_note", {"title": "Groceries", "body": "milk"}, "note groceries milk")
    return lambda: asyncio.run(note.fallback(req, "file"))


@pytest.mark.parametrize("name", list(CASES))
def test_a_real_effect_is_refused_before_its_primitive(name, s):
    kind, arrange, _app = CASES[name]
    act = arrange(s)
    with pytest.raises(BaseException) as got:
        act()
    refused = _refusal(got.value)
    assert refused is not None, f"not refused: {got.value!r}"
    assert refused.kind == kind
    assert s.reached() == []


@pytest.mark.parametrize("name", [n for n, (_k, _a, app) in CASES.items() if app])
def test_neo_app_is_refused_nothing(name, s, neo_app):
    _kind, arrange, _app = CASES[name]
    act = arrange(s)
    try:
        act()
    except BaseException as e:  # noqa: BLE001 - the sentinel stopping it, or what came after it failing
        assert _refusal(e) is None, f"refused in NEO.app: {e!r}"
    assert s.reached(), "the path never reached its primitive"


# ---- the switch itself ---------------------------------------------------------------------------------


def test_the_suite_runs_with_effects_off():
    assert effects.why_off() and not effects.on()
    assert effects._hooked  # installed when neo was first imported


def test_the_suite_widens_nothing():
    """Nothing is allowed outside a test's own opt-in, and neither Laya nor the voice is on, whatever
    the environment says (test_the_environment_cannot_widen_the_suite runs this with it saying so)."""
    from neo import config

    assert not effects._allowed
    assert (os.environ["NEO_REFLEX"], os.environ["NEO_VOICE"]) == ("off", "off")
    assert (config.settings().reflex, config.settings().voice) == ("off", "off")


def test_the_environment_cannot_widen_the_suite():
    """NEO_TEST_RENDER=1 left exported in a shell once let every test reach the network and start
    Chromium; NEO_REFLEX=laya would load the 1.7 GB model into any test that asked the reflex."""
    root = Path(effects.__file__).resolve().parent.parent
    env = {**os.environ, "NEO_TEST_RENDER": "1", "NEO_REFLEX": "laya", "NEO_VOICE": "local"}
    here = "tests/test_v2_effects.py::test_the_suite_widens_nothing"
    run = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", here]
    done = subprocess.run(run, cwd=root, env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]


def test_neo_apps_mode_is_this_tests_own_thread_only(neo_app, tmp_path):
    """Any other thread (one an earlier test left running) keeps effects off; what this test hands
    a worker itself (asyncio.to_thread) runs as NEO.app."""
    marker, out = tmp_path / "ran", {}

    def elsewhere():
        try:
            effects.guard("input", "post a keyboard event")
            out["guard"] = "passed"
        except effects.EffectRefused:
            out["guard"] = "refused"
        try:
            os.system(f"touch '{marker}'")
        except effects.EffectRefused:
            out["os.system"] = "refused"

    t = threading.Thread(target=elsewhere, daemon=True)
    t.start()
    t.join(10)
    assert out == {"guard": "refused", "os.system": "refused"} and not marker.exists()
    effects.guard("input", "post a keyboard event")  # this thread: NEO.app
    assert asyncio.run(asyncio.to_thread(effects.on))


def test_neo_app_searches_the_web(s, neo_app):
    """The door every web search goes through (ddgs's HTTP client) opens in NEO.app: the Rust
    client under it is reached."""
    from ddgs.http_client import HttpClient

    s.search()
    with pytest.raises(Reached):
        HttpClient().request("GET", "https://duckduckgo.com/")
    assert s.reached() == ["primp.Client.request"]


def test_allow_lasts_only_its_block():
    with effects.allow("input"):
        effects.guard("input", "post a keyboard event")
        with effects.allow("audio"):
            effects.guard("audio", "play a sound")
        with pytest.raises(effects.EffectRefused):
            effects.guard("audio", "play a sound")
    with pytest.raises(effects.EffectRefused):
        effects.guard("input", "post a keyboard event")
    assert not effects._allowed


# ---- signals and C functions that start programs ----------------------------------------------------


def test_a_signal_only_reaches_this_processs_own_children():
    """Called as the hook is (nothing is sent): the shell that started the suite, its process group
    and every process at once are refused; a probe (signal 0) and this process itself are not."""
    for event, args in (
        ("os.kill", (os.getppid(), signal.SIGTERM)),
        ("os.kill", (-1, signal.SIGKILL)),
        ("os.kill", (0, signal.SIGTERM)),
        ("os.killpg", (os.getpgid(0), signal.SIGTERM)),
        ("os.kill", (1, signal.SIGTERM)),  # launchd, whose own parent is 0
    ):
        with pytest.raises(effects.EffectRefused, match="signal"):
            effects._audit(event, args)
    effects._audit("os.kill", (os.getppid(), 0))
    effects._audit("os.kill", (os.getpid(), signal.SIGINT))


def test_a_child_this_process_started_can_be_stopped():
    with effects.allow("process"):
        child = subprocess.Popen(["/bin/sleep", "30"])
        done = subprocess.Popen(["/usr/bin/true"])
    try:
        child.terminate()  # Popen's own os.kill
        assert child.wait(10) == -signal.SIGTERM
        time.sleep(0.3)  # `true` has exited and waits to be reaped: a signal to it reaches nothing
        effects._audit("os.kill", (done.pid, signal.SIGTERM))
        done.wait(10)
        effects._audit("os.kill", (done.pid, signal.SIGTERM))  # reaped: no such process any more
    finally:
        for p in (child, done):
            if p.poll() is None:
                p.kill()
                p.wait(10)


def test_a_c_function_that_starts_programs_is_refused_through_ctypes():
    import ctypes

    libc = ctypes.CDLL(None)
    for name in ("system", "posix_spawn", "execv", "fork", "kill"):
        with pytest.raises(effects.EffectRefused, match=name):
            getattr(libc, name)  # looked up only: never called
    assert libc.getpid() == os.getpid()


def test_a_spawn_with_no_audit_event_is_checked_too(tmp_path):
    """multiprocessing starts its children through _posixsubprocess.fork_exec, which raises no audit
    event of its own."""
    import multiprocessing.util

    marker = tmp_path / "ran"
    with pytest.raises(effects.EffectRefused, match="touch"):
        multiprocessing.util.spawnv_passfds(b"/usr/bin/touch", [b"touch", os.fsencode(marker)], ())
    assert not marker.exists()


# ---- a test's fake of a real API, and a spy around the real one ------------------------------------


def _spies(real):
    return {
        "a closure": lambda tap, ev: real(tap, ev),
        "functools.wraps": functools.wraps(real)(lambda tap, ev: real(tap, ev)),
        "a partial": functools.partial(real),
        "a mock": mock.Mock(wraps=real),
        "a helper": lambda tap, ev: _call(real, tap, ev),
    }


def _call(fn, *args):
    return fn(*args)


@pytest.mark.parametrize("shape", ["a closure", "functools.wraps", "a partial", "a mock", "a helper"])
def test_a_spy_that_calls_the_real_api_is_no_fake(shape, s):
    import Quartz

    from neo.tools.computer import input as inp

    real = s.fn("CGEventPost", poses_as="objc")  # stands for PyObjC's own CGEventPost
    s.mp.setattr(Quartz, "CGEventPost", _spies(real)[shape])
    with pytest.raises(effects.EffectRefused):
        inp.click(5, 5)
    assert s.reached() == []


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS")
def test_a_clean_process_is_neo_app_and_the_switch_turns_it_off(tmp_path):
    """Detected for real, not patched: a process that isn't pytest and doesn't set NEO_NO_EFFECTS
    has effects on and no audit hook; with NEO_NO_EFFECTS=1 a program it starts is refused."""
    root = Path(effects.__file__).resolve().parent.parent
    env = {k: v for k, v in os.environ.items() if k not in ("PYTEST_CURRENT_TEST", "NEO_NO_EFFECTS")}
    env["PYTHONPATH"] = str(root)
    marker = tmp_path / "ran"
    code = (
        "import os, sys, neo.effects as e\n"
        "assert 'pytest' not in sys.modules and os.path.dirname(e.__file__).startswith(sys.argv[1])\n"
        "print(repr(e.why_off()), e._hooked)\n"
        "if e.why_off():\n"
        "    try:\n"
        '        os.system("touch \'" + sys.argv[2] + "\'")\n'
        "    except e.EffectRefused:\n"
        "        print('refused')\n"
    )
    run = [sys.executable, "-c", code, str(root), str(marker)]
    with effects.allow("process"):  # a Python child: refused unless meant, as here
        app = subprocess.run(run, env=env, capture_output=True, text=True, timeout=60)
        off = subprocess.run(run, env={**env, "NEO_NO_EFFECTS": "1"}, capture_output=True, text=True, timeout=60)
    assert app.stdout.split("\n")[0] == "'' False", app.stderr
    assert off.stdout.split("\n")[:2] == ["'NEO_NO_EFFECTS=1' True", "refused"], off.stderr
    assert not marker.exists()


def test_a_tests_own_fake_of_the_primitive_is_let_through(monkeypatch):
    import Quartz

    from neo.decide import eventkit
    from neo.tools.computer import input as inp
    from neo.voice import audio

    posted = []
    monkeypatch.setattr(Quartz, "CGEventPost", lambda tap, ev: posted.append(ev))
    inp.click(5, 5)
    assert posted  # the test's own fake did nothing real, so nothing was refused

    class Store:  # a fake Calendar framework, as tests/test_decide_eventkit.py installs one
        @classmethod
        def alloc(cls):
            return cls()

        def init(self):
            return self

    monkeypatch.setattr(eventkit, "_store_obj", None)
    monkeypatch.setattr(eventkit, "_ek", lambda: SimpleNamespace(EKEventStore=Store))
    assert isinstance(eventkit._the_store(), Store)

    class Stream:
        def __init__(self, **kw):
            pass

        def start(self):
            pass

    monkeypatch.setattr(audio.sd, "OutputStream", Stream)
    audio.Speaker()._open_stream()


def test_the_real_calendar_and_contacts_are_refused(monkeypatch):
    """The genuine frameworks, nothing swapped: refused before a store is made."""
    pytest.importorskip("EventKit")
    pytest.importorskip("Contacts")
    from neo.decide import eventkit, people

    monkeypatch.setattr(eventkit, "_store_obj", None)
    with pytest.raises(effects.EffectRefused, match="Calendar"):
        eventkit._the_store()
    assert eventkit._store_obj is None
    with pytest.raises(effects.EffectRefused, match="Contacts"):
        people._store()


def test_the_real_browser_opt_in_lets_playwrights_driver_run(monkeypatch):
    """The tests run with NEO_TEST_RENDER=1 allow the browser and the network (tests/conftest.py's
    `real_browser`): Playwright's own driver may start then, and nothing else."""
    node = "/venv/lib/python3.12/site-packages/playwright/driver/node"
    driver = ("subprocess.Popen", (node, [node, "package/cli.js", "run-driver"], None, None))
    monkeypatch.setattr(effects, "_allowed", set())
    with pytest.raises(effects.EffectRefused, match="run-driver") as got:
        effects._audit(*driver)
    assert got.value.kind == "browser"
    monkeypatch.setattr(effects, "_allowed", {"browser", "network"})
    effects._audit(*driver)
    effects.guard("browser", "start Playwright's headless Chromium")
    with pytest.raises(effects.EffectRefused, match="open -a Safari"):
        effects._audit("subprocess.Popen", ("open", ["open", "-a", "Safari"], None, None))


def test_allow_lets_one_kind_through(monkeypatch):
    monkeypatch.setattr(effects, "_allowed", {"input"})
    effects.guard("input", "post a keyboard event")
    with pytest.raises(effects.EffectRefused, match="refused to play a sound: real effects are off"):
        effects.guard("audio", "play a sound")


_OSA_RUNS_AS_IT_COMPILES = (
    'property p : (do shell script "touch ran")',  # a property's first value
    "script S\nproperty p : 1\nend script",  # a script object's properties
    'use framework "Foundation"',  # loads a framework's code
    'use script "Helpers"',
    'tell application "Finder" to activate',  # loads its dictionary: may launch it
    'using terms from app id "com.apple.mail"\nend using terms from',
    'tell «class capp» "Finder" to activate',
)


@pytest.mark.parametrize(
    "argv, refused",
    [
        (["lsappinfo", "front"], False),
        (["lsappinfo", "info", "-only", "pid", "-app", "com.apple.Safari"], False),
        (["lsappinfo", "list"], False),
        (["lsappinfo", "visibleProcessList"], False),
        (["lsappinfo", "launch", "execpath=/System/Applications/Calculator.app/Contents/MacOS/Calculator"], True),
        (["lsappinfo", "kill", "-force", "0x0-0x1"], True),
        (["lsappinfo", "setfront", "0x0-0x1"], True),
        (["lsappinfo", "setinfo", "0x0-0x1", "LSUIElement=1"], True),
        (["lsappinfo"], True),
        (["/usr/bin/mdfind", "kMDItemContentType == 'com.apple.application-bundle'"], False),
        (["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"], False),
        (["defaults", "read", "-g", "AppleInterfaceStyle"], False),
        (["defaults", "write", "com.apple.dock", "autohide", "-bool", "true"], True),
        (["sysctl", "-n", "hw.memsize"], False),
        (["sysctl", "-w", "kern.maxfiles=1"], True),
        (["sysctl", "kern.maxfiles=1"], True),
        (["sysctl", "-f", "settings.conf"], True),
        (["osacompile", "-o", "s.scpt", "-e", "on run argv\nreturn item 1 of argv\nend run"], False),
        (["osacompile", "-o", "s.scpt", "-e", "tell application id (item 1 of argv) to return name"], False),
        *[(["osacompile", "-o", "s.scpt", "-e", script], True) for script in _OSA_RUNS_AS_IT_COMPILES],
        (["osacompile", "-l", "JavaScript", "-o", "s.scpt", "-e", "1"], True),
        (["osacompile", "-o", "s.scpt"], True),  # the script on stdin: unseen
        ([sys.executable, "-m", "pytest"], False),  # a child test run: under this switch itself
        ([sys.executable, "-c", "import os; os.system('open -a Safari')"], True),  # no switch of its own
        (["python3", "-m", "webbrowser", "https://example.com"], True),
        (["git", "status"], True),  # `-c alias.x=!cmd` runs anything, `web--browse` opens a browser
        (["git", "-c", "alias.x=!touch ran", "x"], True),
        (["uv", "run", "--", "touch", "ran"], True),
        (["open", "-a", "Safari"], True),
        (["/usr/bin/osascript", "-e", 'tell application "Finder" to activate'], True),
        (["shortcuts", "run", "Morning"], True),
        (["say", "hello"], True),
        (["afplay", "/System/Library/Sounds/Glass.aiff"], True),
        (["screencapture", "-x", "shot.png"], True),
        (["pmset", "displaysleepnow"], True),
        (["killall", "Dock"], True),
        (["/bin/sh", "-c", "ls"], True),
    ],
)
def test_which_programs_still_run(argv, refused):
    assert (effects._program(argv[0], argv) is not None) is refused


@pytest.mark.parametrize("script", ["on run argv\nreturn item 1 of argv\nend run", *_OSA_RUNS_AS_IT_COMPILES])
def test_osacompile_reads_the_script_file_it_is_given(script, tmp_path):
    src = tmp_path / "s.applescript"
    src.write_text(script)
    runs = script != "on run argv\nreturn item 1 of argv\nend run"
    assert (effects._program("osacompile", ["osacompile", "-o", str(tmp_path / "s.scpt"), str(src)]) is not None) is runs
    assert effects._program("osacompile", ["osacompile", str(tmp_path / "missing.applescript")]) is not None


@pytest.mark.skipif(shutil.which("osacompile") is None, reason="macOS")
def test_osacompile_never_runs_a_script_as_it_compiles(tmp_path):
    marker = tmp_path / "ran"
    script = f"property p : (do shell script \"touch '{marker}'\")"
    with pytest.raises(effects.EffectRefused, match="osacompile"):
        subprocess.run(["osacompile", "-o", str(tmp_path / "s.scpt"), "-e", script], capture_output=True, timeout=30)
    assert not marker.exists()


@pytest.mark.parametrize(
    "family, address, outside",
    [
        (socket.AF_INET, ("142.250.74.10", 443), True),
        (socket.AF_INET, ("192.168.1.20", 8080), True),
        (socket.AF_INET, ("generativelanguage.googleapis.com", 443), True),
        (socket.AF_INET6, ("2001:db8::1", 443, 0, 0), True),
        # loopback, where nothing of this process listens: NEO.app's local model server (8080) and ws
        # port (8765), a local proxy app, AirPlay (5000, 7000)...
        (socket.AF_INET, ("127.0.0.1", 8080), True),
        (socket.AF_INET, ("localhost", 8765), True),
        (socket.AF_INET, ("0.0.0.0", 7000), True),
        (socket.AF_INET6, ("::1", 5000, 0, 0), True),
        (socket.AF_INET6, ("::ffff:127.0.0.1", 8080, 0, 0), True),
        (socket.AF_UNIX, "/tmp/neo.sock", False),
    ],
)
def test_where_a_connection_goes(family, address, outside):
    assert effects._outside(SimpleNamespace(family=family), address) is outside


def test_loopback_reaches_only_this_processs_own_listeners():
    v4, v6 = SimpleNamespace(family=socket.AF_INET), SimpleNamespace(family=socket.AF_INET6)
    mine = socket.create_server(("127.0.0.1", 0))
    port = mine.getsockname()[1]
    try:
        for sock, host in ((v4, "127.0.0.1"), (v4, "localhost"), (v6, "::1"), (v6, "::ffff:127.0.0.1")):
            assert not effects._outside(sock, (host, port)), host
        assert effects._outside(v4, ("142.250.74.10", port))
    finally:
        mine.close()
    assert effects._outside(v4, ("127.0.0.1", port))  # closed: whatever listens there now isn't this process


def test_a_refused_connection_says_where():
    v6 = SimpleNamespace(family=socket.AF_INET6)
    with pytest.raises(effects.EffectRefused, match=r"connect to \[2001:db8::1\]:443"):
        effects._audit("socket.connect", (v6, ("2001:db8::1", 443, 0, 0)))
    with pytest.raises(effects.EffectRefused, match=r"connect to 127\.0\.0\.1:8080 .*not this process"):
        effects._audit("socket.connect", (SimpleNamespace(family=socket.AF_INET), ("127.0.0.1", 8080)))
    mine = socket.create_server(("::1", 0), family=socket.AF_INET6)
    try:
        effects._audit("socket.connect", (v6, ("::1", mine.getsockname()[1], 0, 0)))  # this process's own
    finally:
        mine.close()


class Elsewhere:
    """A server on this Mac's loopback that this process didn't start (NEO.app's local model server, a
    local proxy app): it runs here, forgotten as this process's own. It answers any request as an
    OpenAI-compatible model server would, and records each one."""

    def __init__(self, monkeypatch) -> None:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self.seen: list[str] = []
        seen = self.seen

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                seen.append(self.requestline)
                self.rfile.read(int(self.headers.get("content-length", 0)))
                body = json.dumps(
                    {
                        "id": "c", "object": "chat.completion", "created": 0, "model": "m",
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant", "content": "A REAL LOCAL MODEL"}}],
                    }
                ).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_GET = do_CONNECT = do_POST

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        monkeypatch.setattr(effects, "_listening", weakref.WeakSet(), raising=False)  # as if elsewhere

    def close(self) -> None:
        self.srv.shutdown()
        self.srv.server_close()


@pytest.fixture
def elsewhere(monkeypatch):
    server = Elsewhere(monkeypatch)
    yield server
    server.close()


def test_the_offline_brain_is_refused_on_loopback(elsewhere, monkeypatch):
    """No keys in the suite: the brain chain ends at the local model server NEO.app runs on
    127.0.0.1:8080 - a real model."""
    from neo import providers
    from neo.providers import openai_compat
    from neo.providers.base import Message

    monkeypatch.setattr(openai_compat, "local", functools.partial(openai_compat.local, port=elsewhere.port))
    monkeypatch.setattr(providers, "_cache", {})
    monkeypatch.setattr(providers, "_demoted", {})
    with pytest.raises(BaseException) as got:
        asyncio.run(asyncio.wait_for(providers.brain("agent").complete([Message("user", "hi")]), 30))
    assert _refusal(got.value) is not None, repr(got.value)
    assert elsewhere.seen == []


def test_no_proxy_carries_a_client_past_the_checks(elsewhere, monkeypatch):
    """A proxy app (Clash, Surge, Proxyman, a corporate agent) listens on loopback and names itself
    in the environment or macOS's settings: every HTTP client would reach the internet through it."""
    import urllib.request

    import httpx

    assert os.environ.get("NO_PROXY") == "*"
    assert not [k for k in os.environ if k.lower().endswith("_proxy") and k.lower() != "no_proxy"]
    assert {k: v for k, v in urllib.request.getproxies().items() if k != "no"} == {}
    assert httpx._utils.get_environment_proxies() == {}

    monkeypatch.setenv("HTTPS_PROXY", f"http://127.0.0.1:{elsewhere.port}")  # set after neo was imported
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    async def get():
        async with httpx.AsyncClient(timeout=5) as c:
            return await c.get("https://generativelanguage.googleapis.com/v1beta/models")

    with pytest.raises(BaseException) as got:
        asyncio.run(get())
    assert _refusal(got.value) is not None, repr(got.value)
    assert elsewhere.seen == []


# ---- the user's keys ------------------------------------------------------------------------------------


def test_the_users_env_files_are_never_read_with_effects_off(monkeypatch, tmp_path):
    """~/.neo/.env holds the user's real API keys: a scratch script (NEO_NO_EFFECTS=1) must not
    pick them up, as the suite never does."""
    from neo import config

    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=key-from-the-users-env-file\n")
    monkeypatch.setitem(config.Settings.model_config, "env_file", (env,))  # the release setting's shape
    monkeypatch.setattr(config, "_settings", None)
    assert config.Settings().gemini_api_key == ""
    assert config.settings().gemini_api_key == ""


def test_neo_app_reads_its_env_file(monkeypatch, tmp_path, neo_app):
    from neo import config

    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=key-from-the-users-env-file\n")
    monkeypatch.setitem(config.Settings.model_config, "env_file", (env,))
    assert config.Settings().gemini_api_key == "key-from-the-users-env-file"


# ---- the incident -----------------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS")
def test_play_through_the_session_with_its_fast_paths_acts_on_nothing(s, monkeypatch):
    """What once opened a browser on the user's Mac: "play ..." through Session.handle with the fast
    paths on and the real play_music handler. Nothing is faked but the reflex; every lowest
    primitive is a sentinel (read-only probes still run, so the path gets as far as it can)."""
    import ApplicationServices
    import playwright.async_api
    import Quartz

    from neo import events
    from neo.agent.registry import registry
    from neo.agent.session import Session
    from neo.reflex.schema import Decision
    from neo.tools import load_all
    from neo.tools import media as media_tool

    class Reflex:
        async def decide(self, text):
            return Decision("quick_action", 0.9, False, 0.0, False, 0.0, "fake")

    load_all()
    monkeypatch.setattr(registry().get("play_music"), "handler", media_tool.play_music)
    monkeypatch.setattr(events, "_bus", events.EventBus())
    s.spawn(probes=True)
    s.network()
    s.search()  # the web searches on the way (site resolution, the video search): Rust, no Python socket
    s.put(playwright.async_api, "async_playwright")
    s.skylight()
    for mod, name in (
        (Quartz, "CGEventPost"),
        (Quartz, "CGEventPostToPid"),
        (ApplicationServices, "AXUIElementPerformAction"),
        (ApplicationServices, "AXUIElementSetAttributeValue"),
    ):
        s.put(mod, name, poses_as="objc")

    async def play():
        return await asyncio.wait_for(Session(reflex=Reflex()).handle("play despacito", from_user=True), 60)

    try:
        asyncio.run(play())
    except BaseException as e:  # noqa: BLE001 - a refusal on the way is the point
        assert _refusal(e) is not None, f"failed, but not refused: {e!r}"
    assert s.reached() == []
