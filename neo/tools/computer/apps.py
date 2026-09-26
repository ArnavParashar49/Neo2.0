"""App-level control: open/activate/quit apps, AppleScript, and first-class helpers for
Safari, Mail, Calendar, Notes, Reminders and Finder. Structured beats clicking."""

from __future__ import annotations

import asyncio
import re
import shlex
import time

from AppKit import NSWorkspace


async def _run(cmd: list[str], timeout: float = 30) -> tuple[int, str, str]:
    p = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
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
    for attempt in range(3):  # the app may still be opening its first window
        await asyncio.to_thread(force_front, pid)
        if await _wait_front(pid, 0.4):
            return True, real
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
                    await asyncio.sleep(0.6)
                    set_target(front_name())  # the browser that took the URL
                    return f"No app called {name!r}; opened {site} in your browser"
            else:
                return f"Error: {name!r} isn't an installed app and {host} doesn't exist"
        if code == 0:
            ok, real = await bring_front(name)
            set_target(real)
            if not ok:
                return (
                    f"NEEDS_USER: {real} is open but macOS kept {front_name() or 'another app'} in front — "
                    f"click {real} (I won't type anywhere else)."
                )
            return f"Opened {real}"
    if code != 0:
        return f"Error: couldn't open {name!r}: {err}"
    await asyncio.sleep(0.6)
    set_target(front_name())
    return f"Opened {name}"


async def activate(name: str) -> str:
    """Bring the app to the front and wait until it *is* frontmost, so the next ax_tree /
    type_text acts on it rather than on whatever was in front a moment ago."""
    if _running(name) is None:
        return f"Error: {name} is not running"
    ok, real = await bring_front(name, wait=0.3)
    set_target(real)
    if ok:
        return f"Activated {real} (frontmost)"
    return f"NEEDS_USER: macOS kept {front_name() or 'another app'} in front instead of {real} — click {real}."


async def quit_app(name: str) -> str:
    return await osascript(f"tell application {_q(name)} to quit") or f"Quit {name}"


async def safari_open(url: str, new_tab: bool = True) -> str:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    script = (
        'tell application "Safari"\n activate\n'
        + (
            f" if (count of windows) = 0 then make new document\n tell window 1 to set current tab to (make new tab with properties {{URL:{_q(url)}}})\n"
            if new_tab
            else f" set URL of current tab of window 1 to {_q(url)}\n"
        )
        + "end tell"
    )
    r = await osascript(script)
    return r if r.startswith("Error") else f"Opened {url} in Safari"


async def safari_page_text(max_chars: int = 8000) -> str:
    js = "document.body.innerText"
    r = await osascript(f'tell application "Safari" to do JavaScript {_q(js)} in current tab of window 1')
    if r.startswith("Error") and "JavaScript" in r:
        return "Error: enable Safari → Develop → Allow JavaScript from Apple Events."
    return r[:max_chars]


async def safari_current_url() -> str:
    return await osascript('tell application "Safari" to get URL of current tab of window 1')


async def mail_send(to: str, subject: str, body: str, send: bool = False) -> str:
    script = f"""
tell application "Mail"
  set m to make new outgoing message with properties {{subject:{_q(subject)}, content:{_q(body)}, visible:true}}
  tell m to make new to recipient at end of to recipients with properties {{address:{_q(to)}}}
  {"send m" if send else "activate"}
end tell"""
    r = await osascript(script)
    return (
        r
        if r.startswith("Error")
        else (f"Sent email to {to}" if send else f"Drafted email to {to} (open in Mail)")
    )


async def mail_unread(limit: int = 10, scan: int = 150) -> str:
    """Newest unread messages. Scans only the newest `scan` messages: Mail's `whose read status
    is false` filter drops the connection (-609) on inboxes with thousands of unread mails."""
    script = f"""
tell application "Mail"
  set total to unread count of inbox
  set out to ""
  set n to 0
  set toScan to count of messages of inbox
  if toScan > {int(scan)} then set toScan to {int(scan)}
  repeat with i from 1 to toScan
    set m to message i of inbox
    if read status of m is false then
      set out to out & (sender of m) & " | " & (subject of m) & " | " & (date received of m as string) & linefeed
      set n to n + 1
      if n ≥ {int(limit)} then exit repeat
    end if
  end repeat
  return (total as string) & " unread in total. Newest:" & linefeed & out
end tell"""
    r = await osascript(script, timeout=45)
    return r or "No unread mail."


async def calendar_add(title: str, start_iso: str, end_iso: str, calendar: str = "") -> str:
    cal = f"calendar {_q(calendar)}" if calendar else "first calendar whose writable is true"
    script = f"""
tell application "Calendar"
  set sd to my parseISO({_q(start_iso)})
  set ed to my parseISO({_q(end_iso)})
  tell ({cal}) to make new event with properties {{summary:{_q(title)}, start date:sd, end date:ed}}
end tell
on parseISO(s)
  set d to current date
  set day of d to 1
  set year of d to (text 1 thru 4 of s) as integer
  set month of d to (text 6 thru 7 of s) as integer
  set day of d to (text 9 thru 10 of s) as integer
  set hours of d to (text 12 thru 13 of s) as integer
  set minutes of d to (text 15 thru 16 of s) as integer
  set seconds of d to 0
  return d
end parseISO"""
    r = await osascript(script)
    return r if r.startswith("Error") else f"Added event '{title}' {start_iso} → {end_iso}"


async def calendar_today() -> str:
    script = """
tell application "Calendar"
  set out to ""
  set t0 to current date
  set time of t0 to 0
  set t1 to t0 + 1 * days
  repeat with c in calendars
    repeat with e in (every event of c whose start date ≥ t0 and start date < t1)
      set out to out & (time string of (start date of e)) & " " & (summary of e) & linefeed
    end repeat
  end repeat
  return out
end tell"""
    r = await osascript(script, timeout=60)
    return r or "Nothing on the calendar today."


async def notes_create(title: str, body: str, folder: str = "") -> str:
    target = f"folder {_q(folder)}" if folder else "default account"
    script = f"""
tell application "Notes"
  tell {target} to make new note with properties {{name:{_q(title)}, body:{_q(body)}}}
end tell"""
    r = await osascript(script)
    return r if r.startswith("Error") else f"Created note '{title}'"


async def reminder_add(title: str, due_iso: str = "") -> str:
    due = f", due date:my parseISO({_q(due_iso)})" if due_iso else ""
    script = f"""
tell application "Reminders"
  tell default list to make new reminder with properties {{name:{_q(title)}{due}}}
end tell
on parseISO(s)
  set d to current date
  set day of d to 1
  set year of d to (text 1 thru 4 of s) as integer
  set month of d to (text 6 thru 7 of s) as integer
  set day of d to (text 9 thru 10 of s) as integer
  set hours of d to (text 12 thru 13 of s) as integer
  set minutes of d to (text 15 thru 16 of s) as integer
  set seconds of d to 0
  return d
end parseISO"""
    r = await osascript(script)
    return (
        r if r.startswith("Error") else f"Reminder added: {title}" + (f" (due {due_iso})" if due_iso else "")
    )


async def finder_reveal(path: str) -> str:
    code, _, err = await _run(["open", "-R", path])
    return f"Revealed {path} in Finder" if code == 0 else f"Error: {err}"


_RISKY_SCRIPT = re.compile(
    r"do shell script|\bdelete\b|empty (?:the )?trash|\bsend\b|\bquit\b|restart|shut ?down|log ?out|"
    r"\bkeystroke\b|key code|set volume|\bmove\b.*\bto (?:trash|folder)|\bduplicate\b|make new (?:outgoing )?message|"
    r"\bsave\b|\bclose\b.*saving|\berase\b|\bremove\b|\bkill\b",
    re.I | re.S,
)


def script_is_risky(script: str) -> bool:
    """AppleScript that changes state or drives the GUI must go through the confirm gate."""
    return bool(_RISKY_SCRIPT.search(script))


def _q(s: str) -> str:
    """Quote for AppleScript string literal."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def shell_quote(s: str) -> str:
    return shlex.quote(s)
