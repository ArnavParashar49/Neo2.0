"""App-level control: open/activate/quit apps, AppleScript, and first-class helpers for
Safari, Mail, Calendar, Notes, Reminders and Finder. Structured beats clicking."""

from __future__ import annotations

import asyncio
import re
import shlex

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


async def open_app(name: str) -> str:
    # URLs and file paths go straight to `open`; app names via -a.
    if "://" in name or name.startswith(("/", "~")):
        code, _, err = await _run(["open", name])
    elif "." in name and " " not in name and not name.lower().endswith(".app"):
        code, _, err = await _run(["open", f"https://{name}"])
    else:
        code, _, err = await _run(["open", "-a", name])
        if code != 0 and " " not in name.strip() and name.isascii():
            # Not an installed app — a single word is almost always a website ("youtube", "github").
            site = f"https://www.{name.strip().lower()}.com"
            code, _, err = await _run(["open", site])
            if code == 0:
                await asyncio.sleep(0.6)
                return f"No app called {name!r}; opened {site} in your browser"
    if code != 0:
        return f"Error: couldn't open {name!r}: {err}"
    await asyncio.sleep(0.6)
    return f"Opened {name}"


def activate(name: str) -> str:
    for a in NSWorkspace.sharedWorkspace().runningApplications():
        if (a.localizedName() or "").lower() == name.lower():
            a.activateWithOptions_(1 << 1)  # NSApplicationActivateIgnoringOtherApps
            return f"Activated {name}"
    return f"Error: {name} is not running"


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
