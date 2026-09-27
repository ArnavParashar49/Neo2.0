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
import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

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
    monkeypatch.setattr(effects, "_allowed", set())  # NEO_TEST_RENDER=1 lets the browser through: not here
    sentinels = Sentinels(monkeypatch, tmp_path)
    yield sentinels
    sentinels.close()


@pytest.fixture
def neo_app(monkeypatch):
    """NEO.app's process: not under pytest, NEO_NO_EFFECTS unset."""
    monkeypatch.setattr(effects, "_under_pytest", lambda: False)
    monkeypatch.delenv("NEO_NO_EFFECTS", raising=False)
    assert effects.on()


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
    """A choke point refused as ``kind``. ``app``: also run in NEO.app's mode (a model provider's
    client retries a dropped connection for seconds there, so those only run with effects off)."""

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
    app = subprocess.run(run, env=env, capture_output=True, text=True, timeout=60)
    assert app.stdout.split("\n")[0] == "'' False", app.stderr
    off = subprocess.run(run, env={**env, "NEO_NO_EFFECTS": "1"}, capture_output=True, text=True, timeout=60)
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
    """NEO_TEST_RENDER=1 (tests/conftest.py) allows the browser and the network: Playwright's own
    driver may start then, and nothing else."""
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


@pytest.mark.parametrize(
    "argv, refused",
    [
        (["lsappinfo", "front"], False),
        (["/usr/bin/mdfind", "kMDItemContentType == 'com.apple.application-bundle'"], False),
        (["defaults", "read", "-g", "AppleInterfaceStyle"], False),
        (["osacompile", "-o", "s.scpt", "-e", "return 1"], False),
        ([sys.executable, "-m", "pytest"], False),
        (["git", "status"], False),
        (["uv", "pip", "list"], False),
        (["open", "-a", "Safari"], True),
        (["/usr/bin/osascript", "-e", 'tell application "Finder" to activate'], True),
        (["shortcuts", "run", "Morning"], True),
        (["say", "hello"], True),
        (["afplay", "/System/Library/Sounds/Glass.aiff"], True),
        (["screencapture", "-x", "shot.png"], True),
        (["pmset", "displaysleepnow"], True),
        (["killall", "Dock"], True),
        (["defaults", "write", "com.apple.dock", "autohide", "-bool", "true"], True),
        (["/bin/sh", "-c", "ls"], True),
    ],
)
def test_which_programs_still_run(argv, refused):
    assert (effects._program(argv[0], argv) is not None) is refused


@pytest.mark.parametrize(
    "family, address, outside",
    [
        (socket.AF_INET, ("127.0.0.1", 80), False),
        (socket.AF_INET, ("localhost", 80), False),
        (socket.AF_INET, ("0.0.0.0", 80), False),
        (socket.AF_INET6, ("::1", 80, 0, 0), False),
        (socket.AF_INET6, ("::ffff:127.0.0.1", 80, 0, 0), False),
        (socket.AF_UNIX, "/tmp/neo.sock", False),
        (socket.AF_INET, ("142.250.74.10", 443), True),
        (socket.AF_INET, ("192.168.1.20", 8080), True),
        (socket.AF_INET, ("generativelanguage.googleapis.com", 443), True),
        (socket.AF_INET6, ("2001:db8::1", 443, 0, 0), True),
    ],
)
def test_where_a_connection_goes(family, address, outside):
    assert effects._outside(SimpleNamespace(family=family), address) is outside


def test_a_refused_connection_says_where():
    v6 = SimpleNamespace(family=socket.AF_INET6)
    with pytest.raises(effects.EffectRefused, match=r"connect to \[2001:db8::1\]:443"):
        effects._audit("socket.connect", (v6, ("2001:db8::1", 443, 0, 0)))
    effects._audit("socket.connect", (v6, ("::1", 443, 0, 0)))  # loopback: fine


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
