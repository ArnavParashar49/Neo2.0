"""App-level control: open/activate/quit apps, AppleScript, and first-class helpers for
Safari, Mail, Calendar, Notes, Reminders and Finder. Structured beats clicking."""

from __future__ import annotations

import asyncio
import re
import shlex
import time

from AppKit import NSWorkspace

from neo import effects
from neo.agent.confirm import ask_user


def _env() -> dict[str, str] | None:
    """An embedded agent's osascript (and whatever its `do shell script` runs) gets the shell
    tool's basic environment, never the host process's; NEO.app's inherits its own, as always."""
    from neo import runtime

    if not runtime.library_bound():
        return None
    from neo.tools.computer.shell import _env as basic

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
    from neo.tools.computer.focus import front_app

    return front_app()[0]


def _running(name: str):
    """The running app called `name` — asked fresh (NSWorkspace's cached list can't see apps
    launched after NEO started unless the Cocoa run loop is pumped)."""
    from AppKit import NSRunningApplication

    from neo.tools.computer.focus import pid_running

    want = name.lower().removesuffix(".app")
    for real, pid in pid_running(want):
        a = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
        if a is not None and (real.lower() == want or a.activationPolicy() == 0):
            return a
    return None


async def _wait_front(pid: int, seconds: float) -> bool:
    from neo.tools.computer.focus import front_app

    for _ in range(max(1, int(seconds / 0.05))):
        if await asyncio.to_thread(lambda: front_app()[1]) == pid:
            return True
        await asyncio.sleep(0.05)
    return False


async def bring_front(name: str, *, wait: float = 1.5) -> tuple[bool, str]:
    """Make the app frontmost and confirm it. First the WindowServer switch window managers
    use (works from a background process while the user is busy elsewhere), then the public
    APIs. Returns (is_front, the app's real name)."""
    from neo.tools.computer.focus import force_front

    a = None
    for _ in range(max(1, int(wait / 0.1))):  # a just-launched app takes a moment to register
        a = await asyncio.to_thread(_running, name)
        if a is not None:
            break
        await asyncio.sleep(0.1)
    if a is None:
        return False, name
    real, pid = a.localizedName(), int(a.processIdentifier())
    if await _wait_front(pid, 0.05):
        return True, real
    for _ in range(3):  # the app may still be opening its first window
        await asyncio.to_thread(force_front, pid)
        if await _wait_front(pid, 0.4):
            return True, real
    effects.guard("focus", f"bring {real} to the front", via=a.activateWithOptions_)
    a.activateWithOptions_(1 << 1)  # NSApplicationActivateIgnoringOtherApps
    if await _wait_front(pid, 0.4):
        return True, real
    await osascript(f"tell application {_q(real)} to activate")
    return await _wait_front(pid, 0.8), real


async def open_app(name: str) -> str:
    # URLs and file paths go straight to `open`; app names via -a.
    if "://" in name or name.startswith(("/", "~")):
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
                    set_target(real or front_name())  # the browser that took the URL
                    return f"No app called {name!r}; opened {site} in your browser"
            else:
                return f"Error: {name!r} isn't an installed app and {host} doesn't exist"
        if code == 0:
            ok, real = await bring_front(name)
            set_target(real)
            if not ok:
                return ask_user(
                    f"{real} is open but macOS kept {front_name() or 'another app'} in front — "
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
            return f"Opened {name} in {real}" if ok else f"Opened {name} in {real} (it may be behind {front_name()})"
    await asyncio.sleep(0.6)
    set_target(front_name())
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
    if _running(name) is None:
        return f"Error: {name} is not running"
    ok, real = await bring_front(name, wait=0.3)
    set_target(real)
    if ok:
        return f"Activated {real} (frontmost)"
    return ask_user(f"macOS kept {front_name() or 'another app'} in front instead of {real} — click {real}.")


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
        return f"Opened {url} in {real}" if ok else f"Opened {url} in {real} (it may be behind {front_name()})"
    return f"Opened {url}"


async def finder_reveal(path: str) -> str:
    code, _, err = await _run(["open", "-R", path])
    return f"Revealed {path} in Finder" if code == 0 else f"Error: {err}"


def _q(s: str) -> str:
    """Quote for AppleScript string literal."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def shell_quote(s: str) -> str:
    return shlex.quote(s)
