"""search_site and page_read: a search on the website the user named, in their own browser, and
the front browser tab read back.

The site word becomes a host by evidence (:mod:`neo.decide.sites`: a domain as written, a site
learned here, an installed web app's site, a region-aware web search). Its search comes from what
the site itself publishes, best first by NEO's record with that host: a learned template, then
OpenSearch, JSON-LD and a GET search form. One page is opened per request: when it isn't the
search, the same tab falls back to the site's own search box (``site_box``) and last to the
browser's address bar and the user's own search engine (``address_bar``). A site no evidence can
name goes to the address bar with the site word among the search words.

A landing is confirmed by the tab's address (Accessibility): on the site and holding the query's
words. Only then are the site word and its template learned (the template keeps the terms slot
alone; a query is never stored). Without Accessibility the page still opens and nothing is learned.

The fast-path checks (``claims_*``) read cached evidence only.
"""

from __future__ import annotations

import asyncio
import re
import threading
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import quote_plus

from neo.decide import run
from neo.decide._objc import pooled
from neo.decide.run import Effect, Outcome, Result

if TYPE_CHECKING:
    from neo.decide.inventory import App
    from neo.decide.sites import SiteRes, Template

__all__ = [
    "KIND",
    "methods_for",
    "search",
    "page_read",
    "landing",
    "default_browser",
    "carries_data",
    "claims_search_site",
    "claims_web_search",
    "shadow",
    "shadow_page_read",
]

KIND = "web_search"
LAND_S = 8.0  # a page NEO opened shows its address this soon
READY_S = 8.0  # the site's page, with its search box, is there this soon
BAR_S = 1.5  # the address field has the focus this soon after its menu command
READ_S = 3.0  # page_read: a web tree just switched on has filled in by then
STEP_S = 0.25
_METHOD = {
    "learned": "learned_template",
    "opensearch": "opensearch",
    "jsonld": "jsonld",
    "site_form": "site_form",
}
_FIELD_ROLES = ("AXTextField", "AXComboBox", "AXSearchField")
_NO_AX = "Accessibility access is off for NEO: System Settings › Privacy & Security › Accessibility"
# how a flow ended -> (what to say, the outcome's reason); '' = the method's own failure (blamed)
_VERDICTS = {
    "untrusted": ("Accessibility is off, so I couldn't check the page", "unconfirmed"),
    "blank": ("the page didn't show its address in time", "unconfirmed"),
    "off_site": ("it shows another site (a sign-in or a check page?)", "unconfirmed"),
    "login": ("it wants you to sign in first", "login"),
    "no_query": ("the page didn't show the search", ""),
    "no_box": ("there's no search box on the page", ""),
    "no_type": ("its search box didn't take the words", ""),
    "no_bar": ("the browser's address bar didn't open", ""),
}


# ---- the browser ---------------------------------------------------------------------------------


def _is_browser(app: App) -> bool:
    from neo.decide import engine

    return engine._is_browser(app)


def methods_for(app: App, kind: str | None = None) -> tuple[str, ...]:
    """How the engine may do this kind with an app (engine.DRIVERS): only a browser searches the
    web, through its address bar; any other app named in a search is no website."""
    return ("address_bar",) if _is_browser(app) else ()


@pooled
def default_browser() -> App | None:
    """The app that opens https links."""
    from neo.decide import inventory, workspace

    bid = workspace.default_handler("https")
    return (inventory.app_for_bundle(bid) or inventory.app_by_id(bid)) if bid else None


def _named_browser(word: str, apps: list[App]) -> App | None:
    from neo.decide import inventory

    app = inventory.named_in(word, apps) if word.strip() else None
    return app if app is not None and _is_browser(app) else None


async def _open(browser: App, url: str) -> tuple[int | None, str]:
    """Open ``url`` in the browser, brought to the front: (its pid, or None and why not). A page
    that didn't open is no method's fault: the caller stops with the reason 'not_found'."""
    from neo.decide.kinds import launch
    from neo.tools.computer import apps

    out = await apps.open_url(url, browser.bundle_id, browser.name)
    if out.startswith("Error"):
        return None, out.removeprefix("Error:").strip()
    pid = await launch.wait_pid(browser.bundle_id)
    return pid, "" if pid else f"{browser.name} didn't start"


# ---- watching the tab (worker threads; each restores the web-content switch it turned on) ------------


def _await_landing(pid: int, host: str, query: str, wait_s: float, stop: threading.Event) -> tuple[str, str]:
    """(verdict, address) once the tab is on ``host``'s site (any site when '') holding the words of
    ``query`` (any page when ''), or at the deadline: off_site, no_query or blank."""
    from neo.decide import axkit, sites

    deadline, url = time.monotonic() + wait_s, ""
    while True:
        url = axkit.web_url(pid) or url
        if url and (not host or sites.on_site(url, host)) and (not query or sites.carries(url, query)):
            return "done", url
        if stop.is_set() or time.monotonic() >= deadline:
            break
        time.sleep(STEP_S)
    if not url:
        return "blank", ""
    return ("off_site" if host and not sites.on_site(url, host) else "no_query"), url


@pooled
def _watch(
    pid: int, path: str, host: str, query: str, wait_s: float, stop: threading.Event
) -> tuple[str, str]:
    from neo.decide import axkit

    app_el = axkit.app_el(pid)
    changed = axkit.enable_web_tree(app_el, axkit.runtime(path))
    try:
        return _await_landing(pid, host, query, wait_s, stop)
    finally:
        axkit.restore_web_tree(app_el, changed)


@pooled
def _box_flow(
    pid: int, path: str, host: str, query: str, ready_s: float, land_s: float, stop: threading.Event
) -> tuple[str, str]:
    """The site's own search box, on the page the tab shows: the words typed, then Return."""
    from neo.decide import axkit, sites

    app_el = axkit.app_el(pid)
    changed = axkit.enable_web_tree(app_el, axkit.runtime(path))
    try:
        deadline, url, field = time.monotonic() + ready_s, "", None
        while field is None:
            url = axkit.web_url(pid) or url
            area = axkit.web_area(pid) if url and sites.on_site(url, host) else None
            if area is not None:
                field = next((h.el for h in axkit.find_search(area) if h.kind == "field"), None)
            if field is not None:
                break
            if stop.is_set() or time.monotonic() >= deadline:
                if not url:
                    return "blank", ""
                if not sites.on_site(url, host):
                    return "off_site", url
                win = axkit.window(app_el)
                return ("login" if win is not None and axkit.needs_sign_in(win) else "no_box"), url
            time.sleep(STEP_S)
        if not axkit.set_search(field, query):
            return "no_type", url
        axkit.post_return(pid)  # to that browser only
        return _await_landing(pid, host, query, land_s, stop)
    finally:
        axkit.restore_web_tree(app_el, changed)


@pooled
def _bar_flow(
    pid: int, path: str, text: str, query: str, land_s: float, stop: threading.Event
) -> tuple[str, str]:
    """The browser's address bar (its "open location" command, found by the ⌘L shortcut, not by a
    title): ``text`` typed for the user's own search engine, then Return."""
    from neo.decide import axkit

    app_el = axkit.app_el(pid)
    hit = axkit.menu_item(pid, "l", 0)
    if hit is None or not axkit.perform(hit[0], "AXPress"):
        return "no_bar", ""
    deadline, field = time.monotonic() + BAR_S, None
    while field is None:
        el = axkit.attr(app_el, "AXFocusedUIElement")
        if el is not None and str(axkit.attr(el, "AXRole") or "") in _FIELD_ROLES:
            field = el
        elif stop.is_set() or time.monotonic() >= deadline:
            return "no_bar", ""
        else:
            time.sleep(0.1)
    if not axkit.set_search(field, text):
        return "no_type", ""
    axkit.post_return(pid)
    changed = axkit.enable_web_tree(app_el, axkit.runtime(path))  # only now: the focus stays put
    try:
        return _await_landing(pid, "", query, land_s, stop)
    finally:
        axkit.restore_web_tree(app_el, changed)


async def _threaded(fn: Any, *args: Any) -> tuple[str, str]:
    """Run a flow in a worker; cancelled, it stops at its next step and restores what it changed."""
    stop = threading.Event()
    try:
        return await asyncio.to_thread(fn, *args, stop)
    finally:
        stop.set()


async def landing(pid: int, browser: App, host: str = "", query: str = "") -> tuple[str, str]:
    """How a page just opened in ``browser`` landed, read from its tab's address within LAND_S:
    ('done', url) when on ``host``'s site holding ``query``'s words; else 'off_site', 'no_query',
    'blank' (no address showed) or 'untrusted' (no Accessibility, so NEO can't tell)."""
    from neo.decide import axkit

    if not await asyncio.to_thread(axkit.trusted):
        return "untrusted", ""
    return await _threaded(_watch, pid, browser.path, host, query, LAND_S)


def _verdict(verdict: str, url: str = "", pid: int | None = None) -> Outcome:
    receipt = {"url": url, "pid": pid}
    if verdict == "done":
        return Outcome(True, "", effect=Effect.DONE, receipt=receipt)
    text, reason = _VERDICTS.get(verdict, (verdict, ""))
    return Outcome(False, text, reason=reason, receipt=receipt)


# ---- the methods ---------------------------------------------------------------------------------------


async def _template(browser: App, url: str, host: str, query: str) -> Outcome:
    pid, why = await _open(browser, url)
    if pid is None:
        return Outcome(False, why, reason="not_found")
    return _verdict(*await landing(pid, browser, host, query), pid)


async def _site_box(browser: App, res: SiteRes, query: str, pid: int | None) -> Outcome:
    """``pid``: the browser, when its tab already shows the site; else the site's page is opened."""
    from neo.decide import axkit

    if pid is None:
        pid, why = await _open(browser, f"https://{res.host}{res.prefix or '/'}")
        if pid is None:
            return Outcome(False, why, reason="not_found")
    if not await asyncio.to_thread(axkit.trusted):
        return Outcome(False, "Accessibility is off, so I couldn't type the search", reason="unconfirmed")
    return _verdict(*await _threaded(_box_flow, pid, browser.path, res.host, query, READY_S, LAND_S), pid)


async def _address_bar(browser: App, text: str, query: str) -> Outcome:
    from neo.agent.confirm import ask_user
    from neo.decide import axkit
    from neo.decide.kinds import launch

    if not await asyncio.to_thread(axkit.trusted):
        return Outcome(False, _NO_AX, reason="permission")
    if await asyncio.to_thread(axkit.pid_of, browser.bundle_id) is None and (
        why := await launch.start(browser)
    ):
        return Outcome(False, why, reason="not_found")
    front, real, pid = await launch.bring_front(browser)
    if not front or pid is None:
        ask = ask_user(f"{real} isn't in front, so I didn't type anything — click {real} and say it again.")
        return Outcome(False, ask, reason="needs_user")
    return _verdict(*await _threaded(_bar_flow, pid, browser.path, text, query, LAND_S), pid)


def _blamed(out: Outcome) -> bool:
    from neo.decide import engine

    return not out.ok and out.reason not in engine.NO_BLAME


def _ordered(host: str, templates: list[Template]) -> list[Template]:
    """Best first by NEO's record with this host (the seeds give discovery's own order)."""
    from neo.decide import engine

    return sorted(templates, key=lambda t: -engine.prior(KIND, host, _METHOD[t.how]))


def _said(out: Outcome, where: str, query: str, browser: str, first: bool) -> Result:
    """The reply: ``where`` is the site's host, '' for the web through the browser's own engine."""
    if out.ok:
        done = f"Searching {where} for “{query}”." if where else f"Searching for “{query}” in {browser}."
        return Result(done, True, say=True if first else None)
    if out.reason == "needs_user":
        return Result(out.text, True)  # NEO's own question to the user
    if out.reason in ("unconfirmed", "login"):  # the tab shows something: the user can take it from there
        return Result(f"I opened {where or 'the search'} in {browser}, but {out.text}.", True, say=True)
    return Result(f"Error: couldn't search {where or 'the web'}: {out.text}", False)


async def _on_site(site: str, res: SiteRes, query: str, browser: App) -> Result:
    from neo.decide import sites

    where, first, pid = f"{res.host}{res.prefix}", res.how == "search", None
    if templates := _ordered(res.host, await sites.discover(res.host, res.prefix)):
        t = templates[0]
        out = await _template(browser, sites.fill(t.url, query), res.host, query)
        run.record(KIND, res.host, _METHOD[t.how], out)
        if out.ok:
            sites.learn_landing(site, res.host, t.url, res.prefix)
            return _said(out, where, query, browser.name, first)
        if t.how == "learned" and _blamed(out):
            sites.template_failed(res.host)
        if not _blamed(out):
            return _said(out, where, query, browser.name, first)
        if sites.on_site(out.receipt.get("url") or "", res.host):
            pid = out.receipt.get("pid")  # the tab is on the site: its own search box is right there
    out = await _site_box(browser, res, query, pid)
    run.record(KIND, res.host, "site_box", out)
    if out.ok:
        sites.learn_landing(site, res.host, sites.sanitize(out.receipt["url"], query), res.prefix)
        return _said(out, where, query, browser.name, first)
    if not _blamed(out):
        return _said(out, where, query, browser.name, first)
    out = await _address_bar(browser, f"{query} site:{sites.registrable(res.host) or res.host}", query)
    run.record(KIND, res.host, "address_bar", out)
    if out.ok:
        return Result(f"Searched {where} for “{query}” through your search engine.", True, say=True)
    return _said(out, where, query, browser.name, first)


async def search(site: str, query: str, *, browser: str = "") -> Result:
    """Open a search for ``query`` on the website ``site`` names, in ``browser`` (the name the user
    said) or the default browser, and check it landed. The tool's gates ran before this."""
    from neo.decide import engine, inventory, sites

    run.begin_request()
    s, q = " ".join(str(site or "").split()), " ".join(str(query or "").split())
    if not s or not q:
        return Result("Error: search for what, on which website?", False)
    apps = await asyncio.to_thread(inventory.inventory)
    named = await asyncio.to_thread(inventory.named_in, s, apps)
    if named is not None and not named.web_start_url:  # a web app's name is its site's
        if not _is_browser(named):
            return Result(f"Error: {s} is an app on this Mac, not a website", False)
        out = await _address_bar(named, q, q)  # "search X on <browser>": the web, in that browser
        run.record(KIND, named.bundle_id, "address_bar", out)
        return _said(out, "", q, named.name, False)
    if sites.site_like(s) is False:  # "my notes", "the <x> app", a day: a place on this Mac, or a time
        return Result(f"Error: {s} isn't a website", False)
    br = await asyncio.to_thread(_named_browser, browser, apps) or await asyncio.to_thread(default_browser)
    if br is None:
        return Result("Error: there's no web browser on this Mac", False)
    res = await sites.resolve(s)
    if res is not None:
        return await _on_site(s, res, q, br)
    # no website NEO can tell by that name: the user's own search engine, with the site's word
    out = await _address_bar(br, f"{q} {s}", q)
    run.record(KIND, engine.WEB, "address_bar", out)
    if out.ok:
        return Result(f"I couldn't tell which website {s} is, so I searched for “{q} {s}”.", True, say=True)
    return _said(out, "", q, br.name, True)


# ---- page_read -----------------------------------------------------------------------------------------


@pooled
def _front_browser() -> tuple[App | None, int | None]:
    """The frontmost app when it shows web pages (a browser or a web app), else the default
    browser: (it, its pid when running)."""
    from neo.decide import axkit, inventory
    from neo.tools.computer.focus import front_app

    _name, pid = front_app()
    app = inventory.app_for_pid(pid) if pid else None
    if app is not None and (_is_browser(app) or app.web_start_url):
        return app, pid
    br = default_browser()
    return br, axkit.pid_of(br.bundle_id) if br is not None else None


@pooled
def _read_flow(pid: int, path: str, limit: int, wait_s: float, stop: threading.Event) -> tuple[str, str]:
    from neo.decide import axkit

    app_el = axkit.app_el(pid)
    changed = axkit.enable_web_tree(app_el, axkit.runtime(path))
    try:
        deadline = time.monotonic() + wait_s
        while axkit.web_area(pid) is None and not stop.is_set() and time.monotonic() < deadline:
            time.sleep(STEP_S)
        return axkit.web_url(pid), axkit.web_text(pid, limit)
    finally:
        axkit.restore_web_tree(app_el, changed)


async def page_read(limit: int = 8000) -> Result:
    """The address and visible text of the front browser tab (whichever browser that is)."""
    from neo.decide import axkit

    if not await asyncio.to_thread(axkit.trusted):
        return Result(f"Error: {_NO_AX}", False)
    browser, pid = await asyncio.to_thread(_front_browser)
    if browser is None or pid is None:
        return Result("Error: no browser is open", False)
    url, text = await _threaded(_read_flow, pid, browser.path, limit, READ_S)
    if not url and not text:
        return Result(f"Error: {browser.name} shows no web page", False)
    return Result(f"URL: {url}\n\n{text}", True)


# ---- the tool's gates and fast-path checks -------------------------------------------------------------


def carries_data(site: str, query: str) -> bool:
    """Would this search carry data out (the shape policy.url_carries_data guards: a long or
    random-looking query or host)? For the taint gate, before anything opens."""
    from neo.decide import sites
    from neo.tools.computer import policy

    return policy.url_carries_data(f"https://{sites._domain(site) or 'site.invalid'}/?q={quote_plus(query)}")


def claims_search_site(filled: dict) -> bool:
    """search_site's fast path, on cached evidence only: the site is one NEO knows is a website
    (a domain, a site learned here, a web app's site). Anything else goes to a model."""
    from neo.decide import sites

    return sites.site_like(str(filled.get("site") or "")) is True


_PREPS = frozenset({"in", "on", "at"})


def _place(words: str) -> str | None:
    """What ``words`` after "in/on/at" (or before "for") are, by local evidence: 'site', 'time',
    'app' (an installed app, "my <x>", "the <x> app"), or None when that can't be told."""
    from neo.decide import sites, when

    like = sites.site_like(words)
    if like:
        return "site"
    if like is False:
        return "time" if when.parse(words, datetime.now()) is not None else "app"
    return None


def _ordinary(words: str) -> bool:
    """Only dictionary words ("recipes", "cheap flights"): no brand or site name among them."""
    from neo.decide import inventory

    ws = set(re.findall(r"[a-z]+", words.casefold()))
    return bool(ws) and all(inventory._common_words(ws).values())


def claims_web_search(filled: dict) -> bool:
    """web_search's fast path, on cached evidence only. A query ending in "in <W>" is a web search
    unless W is a site or a place on this Mac ("restaurants in Delhi" is; "X in notes" isn't);
    one ending in "on/at <W>" only when W is a time ("the weather on monday"; "news on apple" and
    "2tb ssd on <site>" go to a model). "<W> for X" ("<site> for cats") only when W is ordinary
    words or a time. Anything else is claimed."""
    words = " ".join(str(filled.get("query") or "").split()).rstrip(" ?.!").split()
    for i in range(len(words) - 2, 0, -1):  # the last "in/on/at", with words on both sides
        if words[i].casefold() in _PREPS:
            tail = " ".join(words[i + 1 :])
            if len(words) - i - 1 > 3:
                break  # a long tail is a topic, not a place
            place = _place(tail)
            return place in (None, "time") if words[i].casefold() == "in" else place == "time"
    lead = next((i for i in range(1, min(4, len(words) - 1)) if words[i].casefold() == "for"), None)
    if lead is not None:
        w = " ".join(words[:lead])
        place = _place(w)
        return place == "time" or (place is None and _ordinary(w))
    return True


# ---- shadow ----------------------------------------------------------------------------------------------


async def shadow(site: str, legacy_url: str = "") -> dict[str, str | bool]:
    """What search_site's new path would do for ``site``, opening nothing (a site word may cost a
    web search, and discovery reads the site's front page): how the site was told ('app',
    'browser', 'unresolved', or how sites.resolve found it), its host, the first method, and
    whether legacy's search page ``legacy_url`` is on the same site. Never the query."""
    from neo.decide import inventory, sites

    s = " ".join(str(site or "").split())
    apps = await asyncio.to_thread(inventory.inventory)
    named = await asyncio.to_thread(inventory.named_in, s, apps)
    if named is not None and not named.web_start_url:
        browser = _is_browser(named)
        how, method = ("browser", "address_bar") if browser else ("app", "")
        return {"how": how, "key": named.bundle_id, "method": method, "agrees": False}
    res = await sites.resolve(s)
    if res is None:
        return {"how": "unresolved", "key": "", "method": "address_bar", "agrees": False}
    templates = _ordered(res.host, await sites.discover(res.host, res.prefix))
    method = _METHOD[templates[0].how] if templates else "site_box"
    agrees = bool(legacy_url) and sites.on_site(legacy_url, res.host)
    return {"how": res.how, "key": res.host, "method": method, "agrees": agrees}


async def shadow_page_read() -> dict[str, str | bool]:
    """Which browser page_read would read, reading nothing: its bundle id, whether it runs, whether
    Accessibility lets NEO read it, and its runtime (a Chromium one needs its web tree switched on)."""
    from neo.decide import axkit

    browser, pid = await asyncio.to_thread(_front_browser)
    return {
        "key": browser.bundle_id if browser is not None else "",
        "running": pid is not None,
        "trusted": await asyncio.to_thread(axkit.trusted),
        "runtime": axkit.runtime(browser.path) if browser is not None else "",
    }
