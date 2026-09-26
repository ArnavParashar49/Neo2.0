"""open_app: what the user asked to open, found by evidence, never by a guessed web address.

:func:`decide` picks, first rule that applies:

1. a URL, a path or a network address (an IP, "localhost", "<host>:<port>"): opened as given;
2. an installed app the words name (every name the inventory knows it by); a domain the words
   only partly name an app by ("<app>.com") is the website;
3. an everyday noun for a kind of app ("my email", "the browser"): the app this user uses for
   that kind (the engine's top candidate that is used), or the default handler of those links;
4. a website the word names (neo.decide.sites: a domain, a site learned here, an installed
   web-app's site, a region-aware web search), opened in the default browser; words with a ':'
   or a '/' are no site's name and are never looked up;
5. nothing, said honestly: there is no "www.<word>.com" guess.

:func:`open_target` carries it out. An app counts as opened once it is frontmost, a website once
the browser's tab shows that site (read through Accessibility); only then is the word learned as
that site (as one opened, which alone doesn't make "search X on <word>" that site's search).
:func:`shadow` reports what this path would open, opening nothing.
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from neo.decide import run
from neo.decide._objc import pooled
from neo.decide.run import Effect, Outcome, Result

if TYPE_CHECKING:
    from neo.decide.inventory import App
    from neo.decide.sites import SiteRes

__all__ = [
    "KIND",
    "Choice",
    "methods_for",
    "decide",
    "open_target",
    "start",
    "wait_pid",
    "bring_front",
    "shadow",
]

KIND = "open_app"
OPEN_S = 15.0  # `open` returns once LaunchServices has taken the request
PID_S = 5.0  # a just-launched app has registered its process by then
SETTLE_S = 0.6  # a file or a link's app has come to the front by then
# "mailto:x", "x-help:y": a link in some scheme (but "host:8080" is no scheme)
_SCHEME_URL = re.compile(r"^[a-z][a-z0-9+.-]*:(?!\d)\S*$", re.I)
_URL_IN_TEXT = re.compile(r"https?://[^\s,;)]+", re.I)
_HOST_PORT = re.compile(r"^([\w.-]+):\d+(?:/|$)")  # "localhost:3000", "<host>:8080/admin"


@dataclass(frozen=True)
class Choice:
    how: str  # url | path | app | noun | handler | site | none
    target: str  # the words, spaces normalised
    app: App | None = None  # app / noun / handler (None: no app fits the noun)
    url: str = ""  # url / path as it is opened; site: its front page
    site: SiteRes | None = None
    kind: str = ""  # noun: the task kind; handler: the link scheme


def methods_for(app: App, kind: str | None = None) -> tuple[str, ...]:
    """How the engine may do this kind with an app (engine.DRIVERS): any app can be opened."""
    return ("open",)


# ---- what the words mean -----------------------------------------------------------------------


def _handler_app(scheme: str, apps: list[App]) -> App | None:
    from neo.decide import inventory, workspace

    bid = workspace.default_handler(scheme)
    if not bid:
        return None
    return next((a for a in apps if a.bundle_id.lower() == bid.lower()), None) or inventory.app_by_id(bid)


def _used(kind: str, app: App, now: float) -> bool:
    """The user uses the app (the engine's own test), or it has done this kind for NEO."""
    from neo.decide import engine

    recent = bool(app.last_used) and now - float(app.last_used or 0) < engine.USED_WITHIN_S
    return bool(app.running or app.days_used_30 or recent or engine.proven(kind, app.bundle_id))


async def _used_app(kind: str, text: str, apps: list[App]) -> App | None:
    """The kind's top candidate among the apps this user uses (never a route or a fallback)."""
    from neo.decide import engine

    try:
        cands = await engine.choose(kind, text, apps)
    except Exception as e:  # noqa: BLE001 - no decision: no app
        print(f"[decide] choosing an app for {kind} failed ({str(e)[:60]})")
        return None
    now = time.time()
    return next(
        (c.app for c in cands if c.app is not None and c.app.kind == "app" and _used(kind, c.app, now)), None
    )


def _local(host: str) -> bool:
    """This Mac or a network of the user's: an IP that isn't a public one (loopback, private,
    link-local, a VPN's shared range), "localhost", a machine's name (no dot) or an mDNS name."""
    h = host.casefold()
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        return h == "localhost" or "." not in h or h.endswith(".local")
    return not ip.is_global


def _address(t: str) -> str | None:
    """The page for a network address ("192.168.1.1/admin", "localhost:3000", "<host>:8080"),
    port and path kept; None for anything else (a time, "12:30", has no host). This Mac and the
    local network over http (a dev server or a router rarely serves https, and a browser never
    falls back from an https link), anything else over https."""
    head, slash, rest = t.partition("/")
    m = _HOST_PORT.match(t)
    if (m and not m.group(1).isdigit()) or head.casefold() == "localhost":
        return f"{'http' if _local(m.group(1) if m else head) else 'https'}://{t}"
    try:
        ip = ipaddress.ip_address(head)
    except ValueError:
        return None
    return f"{'http' if _local(head) else 'https'}://{f'[{ip}]' if ip.version == 6 else ip}{slash}{rest}"


def _named(t: str, apps: list[App]) -> App | None:
    """The app the words name, unless they are a domain that only contains one of its names."""
    from neo.decide import inventory, sites

    hit = inventory.named_span(t, apps)
    if hit is None:
        return None
    app, start, end = hit
    whole = t[:start].strip(" .!?") == "" and t[end:].strip(" .!?") == ""
    return app if whole or sites._domain(t) is None else None


async def decide(target: str, apps: list[App] | None = None) -> Choice:
    """What ``target`` means on this Mac (see the module docstring). Opens nothing; a site word
    may cost one web search (cached)."""
    from neo.decide import inventory, kinds, sites

    t = " ".join(str(target or "").split())
    if not t:
        return Choice("none", t)
    if url := _address(t):  # before the scheme test: "fe80::1" isn't a link
        return Choice("url", t, url=url)
    if "://" in t or _SCHEME_URL.match(t):
        return Choice("url", t, url=t)
    if t.startswith(("/", "~")):
        return Choice("path", t, url=os.path.expanduser(t))
    head, slash, _ = t.partition("/")
    if slash and " " not in t and sites._domain(head):  # "<domain>/<page>": that page
        return Choice("url", t, url="https://" + t)
    if apps is None:
        apps = await asyncio.to_thread(inventory.inventory)
    if (app := await asyncio.to_thread(_named, t, apps)) is not None:
        return Choice("app", t, app)
    if kind := kinds.noun_kind(t):
        return Choice("noun", t, await _used_app(kind, t, apps), kind=kind)
    if scheme := kinds.HANDLER_NOUNS.get(re.sub(r"^(?:my|the)\s+", "", t.casefold())):
        return Choice("handler", t, await asyncio.to_thread(_handler_app, scheme, apps), kind=scheme)
    if ":" in t or "/" in t:
        return Choice("none", t)  # no site's name: never sent to a web search
    if sites.site_like(t) is not False and (res := await sites.resolve(t)) is not None:
        return Choice("site", t, url=f"https://{res.host}{res.prefix or '/'}", site=res)
    return Choice("none", t)


# ---- opening --------------------------------------------------------------------------------------


async def _open(args: list[str]) -> tuple[int, str]:
    """`open` with these arguments: (exit code, why not)."""
    from neo.tools.computer import apps

    code, _out, err = await apps._run(["open", *args], OPEN_S)
    return code, err


async def start(app: App) -> str:
    """Open the app by its bundle id (never by a name another app could share): '' or why not."""
    code, err = await _open(["-b", app.bundle_id])
    return f"couldn't open {app.name}: {err or 'macOS refused'}" if code else ""


async def wait_pid(bundle_id: str, wait_s: float = PID_S) -> int | None:
    """The app's pid once it runs (None if it hasn't started within ``wait_s``)."""
    from neo.decide import axkit

    deadline = time.monotonic() + wait_s
    while True:
        pid = await asyncio.to_thread(axkit.pid_of, bundle_id)
        if pid or time.monotonic() >= deadline:
            return pid
        await asyncio.sleep(0.2)


@pooled
def _running_name(pid: int) -> str:
    try:
        from AppKit import NSRunningApplication

        a = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
        return str(a.localizedName() or "") if a is not None else ""
    except Exception:  # noqa: BLE001
        return ""


async def bring_front(app: App) -> tuple[bool, str, int | None]:
    """Bring the running app to the front and confirm it: (it is frontmost, the name it runs
    under, its pid). Found by bundle id, so an app running under another name still counts."""
    from neo.tools.computer import apps

    pid = await wait_pid(app.bundle_id)
    name = (await asyncio.to_thread(_running_name, pid) if pid else "") or app.name
    ok, real = await apps.bring_front(name)
    return ok, real, pid


async def _open_given(ch: Choice) -> Result:
    from neo.tools.computer import apps

    if ch.url.lower().startswith(("http://", "https://")):
        text = await apps.open_url(ch.url)
        return Result(text, not text.startswith("Error"))
    code, err = await _open([ch.url])
    if code:
        return Result(f"Error: couldn't open {ch.target}: {err or 'macOS refused'}", False)
    await asyncio.sleep(SETTLE_S)  # then whatever took it is where "type ..." goes next
    apps.set_target(await asyncio.to_thread(apps.front_name))
    return Result(f"Opened {ch.target}", True)


async def _open_app(app: App) -> Result:
    from neo.agent.confirm import ask_user
    from neo.tools.computer import apps

    if why := await start(app):
        run.record(KIND, app.bundle_id, "open", Outcome(False, why))
        return Result(f"Error: {why}", False)
    front, real, _pid = await bring_front(app)
    apps.set_target(real)
    # kept behind another app is macOS's doing, not the app's
    run.record(KIND, app.bundle_id, "open", Outcome(front, "", reason="" if front else "needs_user"))
    if not front:
        behind = await asyncio.to_thread(apps.front_name) or "another app"
        ask = f"{real} is open but macOS kept {behind} in front — click {real} (I won't type anywhere else)."
        return Result(ask_user(ask), True)
    return Result(f"Opened {real}", True)


async def _open_site(word: str, url: str, res: SiteRes) -> Result:
    from neo.decide import sites
    from neo.decide.kinds import site as web
    from neo.tools.computer import apps

    browser = await asyncio.to_thread(web.default_browser)
    text = await apps.open_url(url, browser.bundle_id if browser else "", browser.name if browser else "")
    if text.startswith("Error"):
        run.record(KIND, res.key, "web", Outcome(False, text))
        return Result(text, False)
    verdict = "untrusted"
    if browser is not None and (pid := await wait_pid(browser.bundle_id)):
        verdict, _url = await web.landing(pid, browser, res.host, prefix=res.prefix)
    if verdict == "done":
        run.record(KIND, res.key, "web", Outcome(True, "", effect=Effect.DONE))
        if res.how in ("search", "learned"):  # a domain or a web app's site needs no learning
            sites.learn_landing(word, res.host, None, res.prefix, via="open")
    else:  # NEO couldn't see where it landed: nothing learned, nobody blamed
        run.record(KIND, res.key, "web", Outcome(False, verdict, reason="unconfirmed"))
    where, name = res.key, browser.name if browser else "your browser"
    if verdict == "off_site":
        return Result(
            f"Opened {where} in {name}, but it shows another site (a sign-in or a check page?)",
            True,
            say=True,
        )
    # a site found by a web search is said the first time: the user hears which one it was
    return Result(f"Opened {where} in {name}", True, say=True if res.how == "search" else None)


async def open_target(target: str) -> Result:
    """Open what ``target`` means (see :func:`decide`); the tool's gates ran before this."""
    run.begin_request()
    ch = await decide(target)
    if ch.how in ("url", "path"):
        return await _open_given(ch)
    if ch.how in ("app", "noun", "handler"):
        if ch.app is None:
            return Result(f"Error: I couldn't find an app you use for {ch.target}.", False, say=True)
        return await _open_app(ch.app)
    if ch.site is not None:
        return await _open_site(ch.target, ch.url, ch.site)
    return Result(f"Error: I couldn't find an app or a website called {ch.target}.", False, say=True)


# ---- shadow ---------------------------------------------------------------------------------------


def _agrees(ch: Choice, legacy: str) -> bool:
    """Did legacy (its reply) open the same thing?"""
    from neo.decide import sites

    url = _URL_IN_TEXT.search(legacy or "")
    failed = (legacy or "").startswith("Error")
    if ch.how == "site" and ch.site is not None:
        if url is not None:
            opened = url.group(0)
        else:  # legacy opens words written as a domain at https://<words> and names only the words
            host = None if failed else sites._domain(ch.target)
            opened = f"https://{host}" if host else ""
        return bool(opened) and sites.on_site(opened, ch.site.host, ch.site.prefix)
    if ch.how in ("app", "noun", "handler"):
        return (
            ch.app is not None and url is None and not failed and ch.app.name.casefold() in legacy.casefold()
        )
    if ch.how == "none":
        return failed
    return True  # a URL or a path: both open it as given


async def shadow(target: str, legacy: str = "") -> dict[str, str | bool]:
    """What this path would open for ``target``, opening nothing: how it chose, the app's bundle
    id or the site's host, and whether legacy (its reply ``legacy``) opened the same thing."""
    ch = await decide(target)
    key = ch.app.bundle_id if ch.app is not None else ch.site.key if ch.site is not None else ""
    return {"how": ch.how, "key": key, "agrees": _agrees(ch, legacy)}
