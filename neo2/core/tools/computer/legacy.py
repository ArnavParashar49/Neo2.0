"""Today's app- and site-specific code, moved verbatim from apps.py and web.py.

It runs only while a task kind's mode is `legacy` (neo2.core.config `decide_modes`) and goes away
once every kind has its evidence-chosen driver (neo2/core/decide). Nothing new belongs here: this is
the one place the names of particular apps and websites may still appear.
"""

from __future__ import annotations

from neo2.core.tools.computer.apps import _q, osascript

# ---- AppleScripts (from apps.py) ------------------------------------------------------------


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


# ---- website search pages (from web.py) -----------------------------------------------------
# The table lives in neo2.core.tools.sites_legacy (plain Python: the web tools load it without macOS).
from neo2.core.tools.sites_legacy import (  # noqa: E402, F401
    _AMAZON,
    _ENGINE_RE,
    _GENERIC_SITES,
    _SITE_RE,
    _SITES,
    SITE_NAMES,
    site_search_url,
)
