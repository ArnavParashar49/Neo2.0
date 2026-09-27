"""App-level control: open/activate/quit apps, AppleScript, and first-class helpers for
Safari, Mail, Calendar, Notes, Reminders and Finder. Structured beats clicking."""

from __future__ import annotations

import asyncio
import re
import shlex
import time

from AppKit import NSWorkspace

from neo2.core import effects
from neo2.core.agent.confirm import ask_user
from neo2.core.tools.computer import wait


def _env() -> dict[str, str] | None:
    """An embedded agent's osascript (and whatever its `do shell script` runs) gets the shell
    tool's basic environment, never the host process's; NEO.app's inherits its own, as always."""
    from neo2.core import runtime

    if not runtime.library_bound():
        return None
    from neo2.core.tools.computer.shell import _env as basic

    return basic()


async def _run(cmd: list[str], timeout: float = 30) -> tuple[int, str, str]:
    p = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=_env()
    )
    try:
        out, err = await asyncio.wait_for(p.communicate(), timeout)
    except TimeoutError:
        p.kill()
        return 124, "", f"timed out after {timeout}s"
    return p.returncode or 0, out.decode(errors="replace").strip(), err.decode(errors="replace").strip()


async def osascript(script: str, timeout: float = 30) -> str:
    code, out, err = await _run(["osascript", "-e", script], timeout)
    if code != 0:
        return f"Error: {err or out or 'AppleScript failed'}"
    return out


async def _resolves(host: str, timeout: float = 2.0) -> bool:
    """DNS check so an unknown name doesn't open a dead page."""
    import socket

    loop = asyncio.get_running_loop()
    try:
        await asyncio.wait_for(loop.getaddrinfo(host, 443, type=socket.SOCK_STREAM), timeout)
        return True
    except (OSError, TimeoutError):
        return False


# ---- which app NEO is working in ----------------------------------------------------------
# NEO is a background process, and macOS (cooperative activation) may refuse to bring an app it
# opens to the front while the user is busy in another one. So every open/activate *verifies*
# the app is really frontmost, and remembers it as the target: keystrokes are only ever sent
# to the target (see computer/__init__.py::_guard_front), never to whatever else has focus.
_target: tuple[str, float] | None = None
_TARGET_S = 180.0  # after this long without NEO touching an app, "type X" means "where the cursor is"


def set_target(name: str) -> None:
    global _target
    _target = (name, time.time()) if name else None


def target() -> str | None:
    if _target and time.time() - _target[1] < _TARGET_S:
        return _target[0]
    return None


def front_name() -> str:
    from neo2.core.tools.computer.focus import front_app

    return front_app()[0]


# name → (pid, when): the running app found for a name a moment ago, while it still runs.
_RUNNING_TTL_S = 5.0
_running_cache: dict[str, tuple[int, float]] = {}


def _running(name: str):
    """The running app called `name` — asked fresh (NSWorkspace's cached list can't see apps
    launched after NEO started unless the Cocoa run loop is pumped), and kept for a few seconds
    while it still runs (not terminated, same name). Only an app called exactly that is kept: one
    found by part of its name is asked afresh, so the app called that is found once it runs."""
    from AppKit import NSRunningApplication

    from neo2.core.tools.computer.focus import pid_running

    want = name.lower().removesuffix(".app")
    kept = _running_cache.get(want)
    if kept is not None and time.monotonic() - kept[1] < _RUNNING_TTL_S:
        a = NSRunningApplication.runningApplicationWithProcessIdentifier_(kept[0])
        try:
            real = str(a.localizedName() or "").lower() if a is not None else ""
            if a is not None and not a.isTerminated() and real == want:
                return a
        except Exception:  # noqa: BLE001 - can't tell: ask afresh
            pass
        _running_cache.pop(want, None)
    for real, pid in pid_running(want):
        a = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
        if a is not None and (real.lower() == want or a.activationPolicy() == 0):
            if real.lower() == want:
                _running_cache[want] = (int(pid), time.monotonic())
            return a
    return None


_POLL_S = 0.012  # how often "is it in front yet?" is asked while an app comes forward


async def _wait_front(pid: int, seconds: float) -> bool:
    """Is the app with this pid in front — now, or within ``seconds`` (asked every ~12 ms, done at
    the first yes)? ``seconds`` 0: asked once, no waiting."""
    from neo2.core.tools.computer.focus import front_app

    return bool(
        await wait.until(lambda: front_app()[1] == pid, seconds, every=_POLL_S, thread=True)
    )


async def bring_front(name: str, *, wait: float = 1.5) -> tuple[bool, str]:
    """Make the app frontmost and confirm it. First the WindowServer switch window managers
    use (works from a background process while the user is busy elsewhere), then the public
    APIs. Returns (is_front, the app's real name)."""
    from neo2.core.tools.computer import wait as waits

    # a just-launched app takes a moment to register
    a = await waits.until(lambda: _running(name), wait, every=0.05, thread=True)
    if a is None:
        return False, name
    return await _raise(a)


async def bring_pid_front(pid: int) -> tuple[bool, str]:
    """:func:`bring_front` for the app with this pid (the owner of a point on screen: no name to
    look up, and two apps may share one)."""
    from AppKit import NSRunningApplication

    a = await asyncio.to_thread(NSRunningApplication.runningApplicationWithProcessIdentifier_, int(pid))
    if a is None:
        return False, ""
    return await _raise(a)


async def _raise(a) -> tuple[bool, str]:
    """Bring the app forward and confirm it, done at the first sign it is in front: one look first
    (no waiting — an app already in front is left as it is), then the WindowServer switch, polled
    every ~12 ms; the public APIs only when that doesn't take."""
    from neo2.core.tools.computer.focus import force_front, note_bring

    real, pid = a.localizedName(), int(a.processIdentifier())
    if await _wait_front(pid, 0.0):
        return True, real
    for _ in range(3):  # the app may still be opening its first window
        await asyncio.to_thread(force_front, pid)
        if await _wait_front(pid, 0.4):
            return True, real
    effects.guard("focus", f"bring {real} to the front", via=a.activateWithOptions_)
    note_bring()
    a.activateWithOptions_(1 << 1)  # NSApplicationActivateIgnoringOtherApps
    if await _wait_front(pid, 0.4):
        return True, real
    note_bring()
    await osascript(f"tell application {_q(real)} to activate")
    return await _wait_front(pid, 0.8), real


# ---- after `open`: until the app that took it is in front ------------------------------------------------
_OPEN_NEW_S = 2.0  # another app takes the file or link: at most this long for it to be in front…
_OPEN_SAME_S = 0.6  # …the app already in front takes it (or who takes it can't be told)


def _handler_for(target: str) -> tuple[str, int] | None:
    """(name, pid or 0) of the app macOS opens this file or link with (by what it is, never by a
    guess); None when it can't say."""
    try:
        import os

        from AppKit import NSURL, NSRunningApplication

        url = (
            NSURL.fileURLWithPath_(os.path.expanduser(target))
            if target.startswith(("/", "~"))
            else NSURL.URLWithString_(target)
        )
        if url is None:
            return None
        app_url = NSWorkspace.sharedWorkspace().URLForApplicationToOpenURL_(url)
        if app_url is None:
            return None
        from Foundation import NSBundle

        bundle = NSBundle.bundleWithURL_(app_url)
        bid = str(bundle.bundleIdentifier() or "") if bundle is not None else ""
        name = str(app_url.lastPathComponent() or "").removesuffix(".app")
        runs = NSRunningApplication.runningApplicationsWithBundleIdentifier_(bid) if bid else []
        if runs:
            return str(runs[0].localizedName() or name), int(runs[0].processIdentifier())
        return name, 0
    except Exception:  # noqa: BLE001
        return None


def _has_window(name: str) -> bool:
    """The app shows a window (Accessibility: a focused or main window)."""
    try:
        from neo2.core.tools.computer import ax

        return ax.focused_window(name)[1] is not None
    except Exception:  # noqa: BLE001
        return False


async def settle_opened(target: str, before: str, same_s: float = _OPEN_SAME_S, new_s: float = _OPEN_NEW_S) -> str:
    """The app in front once `open` has handed ``target`` over — polled, done as soon as another
    app than ``before`` is in front: with a window when another app takes it (at most ``new_s``:
    by then the handler has had its chance), at once otherwise (at most ``same_s``: the app already
    in front takes it, or who does can't be told). Never by the handler's name — its bundle's name
    is not always the one it runs under (a localized name, "Code"), and some handlers never come
    forward themselves (a disk image's: the Finder does). Returns the name of the app then in front."""
    handler = await asyncio.to_thread(_handler_for, target)
    other = handler is not None and handler[0] and handler[0] != before

    def settled() -> str:
        now = front_name()
        if not now or now == before:
            return ""
        return now if not other or _has_window(now) else ""

    got = await wait.until(settled, new_s if other else same_s, every=0.03, thread=True)
    return got or await asyncio.to_thread(front_name)


async def open_app(name: str) -> str:
    # URLs and file paths go straight to `open`; app names via -a.
    before = ""
    if "://" in name or name.startswith(("/", "~")):
        before = await asyncio.to_thread(front_name)
        code, _, err = await _run(["open", name])
    elif "." in name and " " not in name and not name.lower().endswith(".app"):
        code, _, err = await _run(["open", f"https://{name}"])
    else:
        code, _, err = await _run(["open", "-a", name])
        if code != 0 and " " not in name.strip() and name.isascii():
            # Not an installed app — a single word is almost always a website ("youtube", "github"),
            # but only open it if the domain actually resolves.
            host = f"www.{name.strip().lower()}.com"
            if await _resolves(host):
                site = f"https://{host}"
                code, _, err = await _run(["open", site])
                if code == 0:
                    browser = default_browser()
                    ok, real = await bring_front(browser) if browser else (False, "")
                    set_target(real or await asyncio.to_thread(front_name))  # the browser that took the URL
                    return f"No app called {name!r}; opened {site} in your browser"
            else:
                return f"Error: {name!r} isn't an installed app and {host} doesn't exist"
        if code == 0:
            ok, real = await bring_front(name)
            set_target(real)
            if not ok:
                front = await asyncio.to_thread(front_name)
                return ask_user(
                    f"{real} is open but macOS kept {front or 'another app'} in front — "
                    f"click {real} (I won't type anywhere else)."
                )
            return f"Opened {real}"
    if code != 0:
        return f"Error: couldn't open {name!r}: {err}"
    web = name.startswith(("http://", "https://")) or (
        "." in name and " " not in name and "://" not in name and not name.startswith(("/", "~"))
    )
    if web:  # a web page: bring the browser that got it to the front (not for files or app schemes)
        browser = default_browser()
        if browser:
            ok, real = await bring_front(browser)
            set_target(real)
            if ok:
                return f"Opened {name} in {real}"
            return f"Opened {name} in {real} (it may be behind {await asyncio.to_thread(front_name)})"
    set_target(await settle_opened(name, before))  # then whatever took it is where "type ..." goes next
    return f"Opened {name}"


def default_browser() -> str:
    """The app that opens https links (Safari, Chrome, Arc…)."""
    try:
        from AppKit import NSURL

        url = NSWorkspace.sharedWorkspace().URLForApplicationToOpenURL_(NSURL.URLWithString_("https://example.com"))
        if url is not None:
            return str(url.lastPathComponent()).removesuffix(".app")
    except Exception:  # noqa: BLE001
        pass
    return ""


async def activate(name: str) -> str:
    """Bring the app to the front and wait until it *is* frontmost, so the next ax_tree /
    type_text acts on it rather than on whatever was in front a moment ago."""
    if await asyncio.to_thread(_running, name) is None:
        return f"Error: {name} is not running"
    ok, real = await bring_front(name, wait=0.3)
    set_target(real)
    if ok:
        return f"Activated {real} (frontmost)"
    front = await asyncio.to_thread(front_name)
    return ask_user(f"macOS kept {front or 'another app'} in front instead of {real} — click {real}.")


async def quit_app(name: str) -> str:
    return await osascript(f"tell application {_q(name)} to quit") or f"Quit {name}"


async def open_url(url: str, browser_id: str = "", browser_name: str = "") -> str:
    """Open a web page: in the browser given by bundle id, else in the user's default browser
    (whatever macOS says that is), and bring that browser to the front."""
    if not re.match(r"^[a-z][\w+.-]*:", url, re.I) or re.match(r"^[\w.-]+:\d+(?:/|$)", url):
        url = "https://" + url  # "github.com", "localhost:3000"
    code, _, err = await _run(["open", "-b", browser_id, url] if browser_id else ["open", url])
    if code != 0:
        return f"Error: couldn't open {url}: {err}"
    browser = browser_name or default_browser()
    if browser:
        ok, real = await bring_front(browser)
        set_target(real)
        if ok:
            return f"Opened {url} in {real}"
        return f"Opened {url} in {real} (it may be behind {await asyncio.to_thread(front_name)})"
    return f"Opened {url}"


async def finder_reveal(path: str) -> str:
    code, _, err = await _run(["open", "-R", path])
    return f"Revealed {path} in Finder" if code == 0 else f"Error: {err}"


def _q(s: str) -> str:
    """Quote for AppleScript string literal."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def shell_quote(s: str) -> str:
    return shlex.quote(s)
