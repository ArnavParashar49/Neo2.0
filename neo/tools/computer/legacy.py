"""Today's app- and site-specific code, moved verbatim from apps.py and web.py.

It runs only while a task kind's mode is `legacy` (neo.config `decide_modes`) and goes away
once every kind has its evidence-chosen driver (neo/decide). Nothing new belongs here: this is
the one place the names of particular apps and websites may still appear.
"""

from __future__ import annotations

from neo.tools.computer.apps import _q, osascript

# ---- AppleScripts (from apps.py) ------------------------------------------------------------


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


# ---- website search pages (from web.py) -----------------------------------------------------

# "search X on amazon", "open it on google": a real search page on that site, in the browser.
_SITES: dict[str, str] = {
    "google": "https://www.google.com/search?q={q}",
    "google images": "https://www.google.com/search?tbm=isch&q={q}",
    "images": "https://www.google.com/search?tbm=isch&q={q}",
    "google maps": "https://www.google.com/maps/search/{q}",
    "maps": "https://www.google.com/maps/search/{q}",
    "youtube": "https://www.youtube.com/results?search_query={q}",
    "amazon": "https://www.{amazon}/s?k={q}",
    "flipkart": "https://www.flipkart.com/search?q={q}",
    "ebay": "https://www.ebay.com/sch/i.html?_nkw={q}",
    "wikipedia": "https://en.wikipedia.org/w/index.php?search={q}",
    "github": "https://github.com/search?q={q}&type=repositories",
    "reddit": "https://www.reddit.com/search/?q={q}",
    "stack overflow": "https://stackoverflow.com/search?q={q}",
    "stackoverflow": "https://stackoverflow.com/search?q={q}",
    "twitter": "https://x.com/search?q={q}",
    "x": "https://x.com/search?q={q}",
    "linkedin": "https://www.linkedin.com/search/results/all/?keywords={q}",

    "imdb": "https://www.imdb.com/find/?q={q}",
    "duckduckgo": "https://duckduckgo.com/?q={q}",
    "bing": "https://www.bing.com/search?q={q}",
    "news": "https://news.google.com/search?q={q}",
    "google news": "https://news.google.com/search?q={q}",
    "pinterest": "https://www.pinterest.com/search/pins/?q={q}",
    "instagram": "https://www.instagram.com/explore/search/keyword/?q={q}",
}
_AMAZON = {
    "IN": "amazon.in", "GB": "amazon.co.uk", "DE": "amazon.de", "FR": "amazon.fr", "IT": "amazon.it",
    "ES": "amazon.es", "CA": "amazon.ca", "JP": "amazon.co.jp", "AU": "amazon.com.au", "MX": "amazon.com.mx",
    "BR": "amazon.com.br", "NL": "amazon.nl", "AE": "amazon.ae", "SG": "amazon.sg", "SE": "amazon.se",
}
SITE_NAMES = sorted(_SITES, key=len, reverse=True)
_SITE_RE = "|".join(n.replace(" ", r"\s+") for n in SITE_NAMES)
# Sites named by a generic word ("maps", "news", "images", "x") need an explicit search verb —
# "open the news on …" / "find x on …" are too ambiguous for a fast path.
_GENERIC_SITES = {"maps", "news", "images", "x"}
_ENGINE_RE = "|".join(n.replace(" ", r"\s+") for n in SITE_NAMES if n not in _GENERIC_SITES)


def site_search_url(site: str, query: str) -> str:
    from urllib.parse import quote_plus

    from neo.tools.web import _country  # web.py imports this module: read at call time

    key = " ".join(site.lower().replace("www.", "").replace(".com", "").split())
    q = quote_plus(query.strip())
    if key in _SITES:
        return _SITES[key].format(q=q, amazon=_AMAZON.get(_country(), "amazon.com"))
    if "." in site:  # a domain: Google, limited to that site
        return f"https://www.google.com/search?q={quote_plus(f'site:{site.strip()} {query.strip()}')}"
    return f"https://www.google.com/search?q={quote_plus(f'{query.strip()} {site.strip()}')}"

