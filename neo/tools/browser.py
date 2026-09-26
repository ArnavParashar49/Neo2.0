"""NEO's own browser: Playwright + Chromium with a persistent profile, driven through the page's
structure.

A separate, visible Chromium window — never the user's own browser — whose profile lives in
<data_dir>/browser, so the logins the user makes there persist (the privacy erase removes it).
A page is read as a compact snapshot: its interactive elements, numbered, from the page's
accessibility tree, plus the visible text. The click and type tools take that number (or the
element's name). A page that is mostly pixels (a canvas, images) comes with a screenshot for the
vision brain, and browser_click then takes x and y.

By itself NEO waits for a page to settle, follows the tab a click of its own opened, and declines a
cookie banner (it never accepts one). Logins, one-time codes, card details and CAPTCHAs are the
user's: NEO never types a password, it hands the window over (NEEDS_USER) and the user says
"continue". It asks first (with the element and the page on the card) before anything that
submits, buys, sends, posts or deletes, and — after reading untrusted content — before a link
that carries data or reaches this Mac's own network.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from contextlib import suppress
from dataclasses import dataclass
from urllib.parse import quote_plus, urljoin, urlsplit

from neo.agent import confirm
from neo.agent.registry import ToolContext, ToolOutput, tool
from neo.config import settings
from neo.providers.base import ImagePart

_TIMEOUT_MS = 8000  # any one step on a page
_NAV_MS = 20000  # a page load
_IDLE_MS = 2500  # network quiet after a load: most pages get there, streaming ones never do
_CLICK_MS = 5000
_FOLLOW_S = 0.8  # how long a click gets to open a tab or start loading a page
_CHALLENGE_WAIT_S = 6.0  # an automatic "checking your browser" page usually clears by itself
_SETTLE_S = 0.3  # after a banner closes or a key goes in
_READ_CHARS = 6000  # one read fits the agent's per-result budget whole (nothing cut from the middle)
_MAX_ELEMENTS = 150
_MAX_LINKS = 3000
_VIEWPORT = {"width": 1280, "height": 860}


def _headless() -> bool:
    """Only for checks without a screen: NEO's browser is meant to be seen (the user steps in)."""
    return os.environ.get("NEO_BROWSER_HEADLESS", "").strip().lower() in ("1", "true", "yes")


# ---- the page as numbered elements ---------------------------------------------------------------

_INTERACTIVE = {
    "link", "button", "textbox", "searchbox", "combobox", "checkbox", "radio", "switch", "slider",
    "spinbutton", "tab", "menuitem", "menuitemcheckbox", "menuitemradio", "option", "listbox", "treeitem",
}  # fmt: skip
_VALUED = {"textbox", "searchbox", "combobox", "spinbutton", "slider"}
_STATES = ("checked", "disabled", "expanded", "selected", "pressed")
_LINE = re.compile(
    r'^(?P<indent> *)- (?P<role>[a-z][a-z-]*)(?: "(?P<name>(?:[^"\\]|\\.)*)")?'
    r"(?P<attrs>(?: \[[^\]]*\])*)(?P<colon>:)?(?: (?P<rest>.*))?$"
)
_URL_LINE = re.compile(r"^ *- /url: (?P<url>.+)$")
_ATTR = re.compile(r"\[([^\]=]+)(?:=([^\]]*))?\]")
# A field NEO never types into: passwords, one-time codes, card and bank numbers. ("PIN code" is a
# postal code in India, so a PIN only without "code".)
_SECRET_FIELD = re.compile(
    r"passw|passcode|\bpin\b(?!\s*-?\s*code)|one[- ]?time|\botp\b|\b2fa\b|two[- ]factor|verification code|"
    r"security code|auth(?:entication)? code|\bcvv|\bcvc\b|\bcsc\b|card ?number|credit ?card|debit ?card|"
    r"\biban\b|routing number|account number|social security|\bssn\b",
    re.I,
)
_SECRET_AUTOCOMPLETE = re.compile(r"password|one-time-code|\bcc-", re.I)


@dataclass
class _El:
    n: int
    role: str
    name: str = ""
    ref: str = ""  # Playwright's aria ref ("e12", "f1e3" in a frame) from the snapshot
    nth: int = 0  # which of the elements with this role and name (the fallback when refs are gone)
    value: str = ""
    url: str = ""
    states: tuple[str, ...] = ()

    def card(self) -> str:
        """How the element is named on a confirmation card and in results."""
        return f"[{self.n}] {self.role} “{self.name[:60]}”" if self.name else f"[{self.n}] {self.role}"

    def line(self) -> str:
        s = f"[{self.n}] {self.role}"
        if self.name:
            s += f' "{self.name[:100]}"'
        s += "".join(f" [{x}]" for x in self.states)
        if self.value:
            s += f' = "{self.value[:80]}"'
        if self.url and not self.name:
            s += f" → {self.url[:80]}"
        return s


def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] == '"':
        with suppress(ValueError):
            return str(json.loads(s))
        return s[1:-1]
    return s


def _parse(snapshot: str) -> list[_El | str]:
    """The snapshot's interactive elements, numbered in page order, with headings and dialogs kept
    as unnumbered context lines. Works on Playwright's "ai" snapshot (with refs) and the plain one."""
    out: list[_El | str] = []
    seen: dict[tuple[str, str], int] = {}
    last: _El | None = None
    n = 0
    for raw in snapshot.splitlines():
        if (u := _URL_LINE.match(raw)) and last is not None and last.role == "link" and not last.url:
            last.url = _unquote(u.group("url"))
            continue
        m = _LINE.match(raw)
        if not m:
            continue
        role = m.group("role")
        name = _unquote('"' + m.group("name") + '"') if m.group("name") is not None else ""
        attrs = dict((k.strip(), (v or "").strip()) for k, v in _ATTR.findall(m.group("attrs") or ""))
        rest = _unquote(m.group("rest") or "")
        if role == "heading" and name:
            out.append("#" * min(int(attrs.get("level") or 2), 3) + " " + name[:120])
            continue
        if role in ("dialog", "alertdialog"):
            out.append(f"(dialog{' “' + name[:80] + '”' if name else ''})")
            continue
        if (
            role == "option"
            and "selected" in attrs
            and last is not None
            and last.role in ("combobox", "listbox")
        ):
            last.value = last.value or name
        ref = attrs.get("ref", "")
        clickable = role not in _INTERACTIVE and attrs.get("cursor") == "pointer" and ref and (name or rest)
        if role == "option" and not ref:
            continue  # a <select>'s options: the combobox takes the text
        if role not in _INTERACTIVE and not clickable:
            continue
        if clickable:
            name = name or rest[:80]
        n += 1
        key = (role, name)
        el = _El(n, role, name, ref=ref, nth=seen.get(key, 0), states=tuple(s for s in _STATES if s in attrs))
        seen[key] = el.nth + 1
        if role in _VALUED and rest and not _SECRET_FIELD.search(name):
            el.value = rest
        out.append(el)
        last = el
    return out


def _render(
    url: str,
    title: str,
    notes: list[str],
    entries: list[_El | str],
    text: str,
    max_chars: int,
    find: str = "",
) -> str:
    head = f"URL: {url}\nTitle: {title}\n" + "".join(f"[NOTE] {x}\n" for x in notes)
    phrase = " ".join(find.lower().split())
    words = phrase.split()

    def hit(s: str) -> bool:  # the phrase, or every one of its words
        s = s.lower()
        return phrase in s or all(w in s for w in words)

    if words:
        entries = [e for e in entries if isinstance(e, _El) and hit(e.name)]
        lines = [ln.strip() for ln in text.splitlines()]
        keep = {j for i, ln in enumerate(lines) if hit(ln) for j in (i - 1, i, i + 1)}
        text = "\n".join(lines[i] for i in sorted(keep) if 0 <= i < len(lines) and lines[i])
        head += f"(only what mentions “{find[:60]}”)\n"
    budget = max(400, (max_chars - len(head)) // 2)
    shown: list[str] = []
    used, more = 0, 0
    total = sum(isinstance(e, _El) for e in entries)
    for e in entries:
        ln = e.line() if isinstance(e, _El) else e
        if used + len(ln) + 1 > budget or len(shown) >= _MAX_ELEMENTS:
            more = total - sum(1 for x in shown if x.startswith("["))
            break
        shown.append(ln)
        used += len(ln) + 1
    els = "\n".join(shown) or "(no interactive elements)"
    if more > 0:
        els += f'\n(+{more} more elements further down — browser_read find="…" to reach one)'
    room = max(0, max_chars - len(head) - len(els) - 30)
    body = text if len(text) <= room else text[:room].rstrip() + " …"
    return f"{head}[ELEMENTS] click/type by number\n{els}\n\n[TEXT]\n{body or '(no text)'}"


def _clean_text(text: str) -> str:
    out: list[str] = []
    for ln in text.splitlines():
        ln = " ".join(ln.split())
        if ln and (not out or out[-1] != ln):
            out.append(ln)
    return "\n".join(out)


# ---- the browser -----------------------------------------------------------------------------------


class _Browser:
    """One Chromium for NEO, started on first use. `current` is the tab NEO works in: the one it
    opened last, or the tab a click of its own just opened. `els` are the numbers of the latest
    read of that tab."""

    def __init__(self) -> None:
        self._pw = None
        self._ctx = None
        self._gone = False  # the context closed under NEO (the user closed the window)
        self._lock = asyncio.Lock()
        self.dialogs: list[str] = []  # the pages' own pop-ups since the last action reported them
        self.current = None
        self.els: list[_El] = []
        self.els_page = None
        self.els_url = ""
        # Links pages showed NEO (absolute): opening one of them is following the page, not a URL a
        # model put together — it can't carry anything of the user's out.
        self.links: dict[str, None] = {}

    async def page(self, *, new_tab: bool = False):
        async with self._lock:
            if self._ctx is not None and self._gone:
                await self._drop()  # the user closed NEO's browser window: start it again
            if self._ctx is None:
                await self._launch()
            if new_tab:
                self.current = await self._ctx.new_page()
                return self.current
            cur = self.current
            if cur is None or cur.is_closed():
                pages = self.pages()
                cur = self.current = pages[-1] if pages else await self._ctx.new_page()
            return cur

    async def _launch(self) -> None:
        from playwright.async_api import async_playwright

        profile = settings().data_dir / "browser"
        profile.mkdir(parents=True, exist_ok=True)
        pw = await async_playwright().start()
        try:
            ctx = await pw.chromium.launch_persistent_context(
                str(profile),
                headless=_headless(),
                viewport=_VIEWPORT,
                args=["--disable-blink-features=AutomationControlled"],
            )
        except Exception:
            await pw.stop()
            raise
        ctx.set_default_timeout(_TIMEOUT_MS)
        ctx.on("close", self._closed)
        ctx.on("page", self._watch)
        for p in ctx.pages:
            self._watch(p)
        self._pw, self._ctx, self._gone = pw, ctx, False

    def _closed(self, *_) -> None:
        self._gone = True

    def _watch(self, page) -> None:
        page.on("dialog", self._dialog)

    async def _dialog(self, dialog) -> None:
        """A page's own alert / confirm / prompt. Dismissed — "Cancel" is the choice that changes
        nothing — and told to the model with the next action's result. Leaving a page NEO was told
        to leave ("unsaved changes?") is accepted."""
        kind = str(getattr(dialog, "type", ""))
        with suppress(Exception):
            if kind == "beforeunload":
                await dialog.accept()
                return
            self.dialogs.append(" ".join(str(dialog.message).split())[:200])
            del self.dialogs[:-3]
            await dialog.dismiss()

    def popups(self) -> str:
        said = "; ".join(f"“{m}”" for m in self.dialogs)
        self.dialogs = []
        return f" The page showed a pop-up ({said}); NEO dismissed it." if said else ""

    async def _drop(self) -> None:
        if self._ctx is not None:
            with suppress(Exception):
                await self._ctx.close()
        if self._pw is not None:
            with suppress(Exception):
                await self._pw.stop()
        self._ctx = self._pw = self.current = self.els_page = None
        self.els, self.els_url, self._gone = [], "", False

    def pages(self) -> list:
        return [p for p in (self._ctx.pages if self._ctx is not None else []) if not p.is_closed()]

    def url(self) -> str:
        cur = self.current
        return "" if cur is None or cur.is_closed() else str(cur.url)

    async def follow(self, page, before: set[int], url0: str):
        """The tab an action just opened — now the current one — or None when it opened none
        (it navigated this tab, or changed the page in place)."""
        deadline = time.monotonic() + _FOLLOW_S
        while True:
            new = [p for p in self.pages() if id(p) not in before]
            if new:
                self.current = new[-1]
                with suppress(Exception):
                    await self.current.bring_to_front()
                return self.current
            if page.url != url0 or time.monotonic() >= deadline:
                return None
            await asyncio.sleep(0.1)

    def remember(self, page, entries: list[_El | str]) -> None:
        self.els = [e for e in entries if isinstance(e, _El)]
        self.els_page, self.els_url = page, str(page.url)
        for e in self.els:
            if e.url and not e.url.startswith("#"):
                with suppress(ValueError):
                    self.links[_no_fragment(urljoin(self.els_url, e.url))] = None
        for old in list(self.links)[: max(0, len(self.links) - _MAX_LINKS)]:
            del self.links[old]

    def element(self, page, n: int) -> tuple[_El | None, str]:
        if self.els_page is not page or not self.els:
            return None, "read this page first (browser_read) — those numbers were for another page"
        if _no_fragment(page.url) != _no_fragment(self.els_url):
            return None, "the page changed since it was read — read it again (browser_read) for new numbers"
        for e in self.els:
            if e.n == n:
                return e, ""
        return None, f"there's no element [{n}] on this page — read it again (browser_read)"

    def named(self, page, name: str, role: str = "") -> list[_El]:
        if self.els_page is not page or _no_fragment(page.url) != _no_fragment(self.els_url):
            return []
        want = " ".join(name.lower().split())
        pool = [e for e in self.els if not role or e.role == role.lower()]
        exact = [e for e in pool if " ".join(e.name.lower().split()) == want]
        return exact or [e for e in pool if want and want in e.name.lower()]

    async def close(self) -> None:
        async with self._lock:
            await self._drop()
            self.links, self.dialogs = {}, []


_b = _Browser()


def _no_fragment(url: str) -> str:
    return str(url).split("#", 1)[0]


def _norm_url(url: str) -> str:
    url = url.strip()
    if not re.match(r"^[a-z]+://", url):
        url = "https://" + url
    return url


def _host(url: str) -> str:
    return urlsplit(str(url)).hostname or "this page"


def _err(e: BaseException) -> str:
    return (str(e).splitlines() or [type(e).__name__])[0][:160]


async def _settle(page) -> None:
    """Loaded, then network-quiet if it gets there soon (pages that stream never do)."""
    with suppress(Exception):
        await page.wait_for_load_state("domcontentloaded", timeout=_NAV_MS)
    with suppress(Exception):
        await page.wait_for_load_state("networkidle", timeout=_IDLE_MS)


def _locator(page, target: str, role: str = ""):
    """Resolution for words that aren't in the latest read: role+name → label/placeholder →
    visible text → CSS."""
    t = target.strip()
    if role:
        return page.get_by_role(role, name=re.compile(re.escape(t), re.I)).first
    if t.startswith(("#", ".", "//", "css=", "xpath=", "[")) or re.match(r"^[a-z]+(\[|\.|#|$)", t):
        with suppress(Exception):  # not a selector after all
            return page.locator(t).first
    rx = re.compile(re.escape(t), re.I)
    return (
        page.get_by_role("button", name=rx)
        .or_(page.get_by_role("link", name=rx))
        .or_(page.get_by_label(rx))
        .or_(page.get_by_placeholder(rx))
        .or_(page.get_by_text(t, exact=False))
        .first
    )


async def _el_locator(page, el: _El):
    if el.ref:
        loc = page.locator(f"aria-ref={el.ref}")
        with suppress(Exception):
            if await loc.count():
                return loc
    if el.name:
        return page.get_by_role(el.role, name=el.name, exact=True).nth(el.nth)
    return None


async def _resolve(page, target: str, role: str = ""):
    """(locator, element from the latest read or None, error). A number is an element of the latest
    read; a name matching one element there is that element; two or more is the model's choice to
    make, by number."""
    t = str(target).strip()
    if m := re.fullmatch(r"\[?#?(\d{1,4})\]?", t):
        el, why = _b.element(page, int(m.group(1)))
        if el is None:
            return None, None, f"Error: {why}."
        loc = await _el_locator(page, el)
        if loc is None:
            return None, None, f"Error: [{el.n}] isn't on the page any more — read it again (browser_read)."
        return loc, el, ""
    found = _b.named(page, t, role)
    if len({e.ref or e.n for e in found}) > 1:
        which = ", ".join(e.card() for e in found[:8])
        return None, None, f"Error: several elements match “{t[:60]}”: {which}. Say which by number."
    if found:
        loc = await _el_locator(page, found[0])
        if loc is not None:
            return loc, found[0], ""
    return _locator(page, t, role), None, ""


# ---- what the page is: text, graphics, sign-in, checks -----------------------------------------------

_JS_PAGE = r"""() => {
  const vis = e => { const r = e.getBoundingClientRect(); if (r.width < 2 || r.height < 2) return false;
    const s = getComputedStyle(e); return s.visibility !== 'hidden' && s.display !== 'none' && +s.opacity !== 0; };
  const area = innerWidth * innerHeight || 1;
  let canvas = 0;
  for (const c of document.querySelectorAll('canvas')) if (vis(c)) {
    const r = c.getBoundingClientRect(); canvas = Math.max(canvas, (r.width * r.height) / area); }
  const inputs = [...document.querySelectorAll('input')].filter(vis);
  const CODE = /one[- ]?time|\botp\b|verification code|security code|\b2fa\b|two[- ]factor|passcode|auth(?:entication)? code/i;
  const words = i => [i.name, i.id, i.placeholder, i.getAttribute('aria-label')].join(' ');
  return {
    body: (document.body ? document.body.innerText : '').slice(0, 60000),
    main: [...document.querySelectorAll('main,[role=main]')].map(m => m.innerText).join('\n').slice(0, 60000),
    canvas,
    password: inputs.some(i => i.type === 'password'),
    code: inputs.some(i => (i.getAttribute('autocomplete') || '') === 'one-time-code' || CODE.test(words(i))),
    widget: [...document.querySelectorAll('iframe')].some(f => vis(f)
      && /captcha|challenge/i.test((f.title || '') + ' ' + (f.src || '')) && !/size=invisible/i.test(f.src || '')),
  };
}"""
# Words a human-check page says (on the page itself, not in a small widget's frame).
_CHALLENGE = re.compile(
    r"verify (?:that )?you(?:['’]| a)re (?:a )?human|i['’]?m not a robot|are you a (?:human|robot)|unusual traffic|"
    r"checking (?:if the site connection is secure|your browser)|press (?:and|&) hold|"
    r"complete the security check|(?:solve|complete) (?:the|this) (?:captcha|puzzle)|"
    r"(?:enter|type) the (?:characters|captcha|text) (?:you see|shown|below|in the image)",
    re.I,
)


async def _probe(page) -> dict:
    try:
        got = await page.evaluate(_JS_PAGE)
    except Exception:  # noqa: BLE001 — a page mid-navigation: read what the snapshot gives
        got = None
    return got if isinstance(got, dict) else {}


def _blocked(probe: dict) -> bool:
    """A page that is nothing but a human check ("unusual traffic", "checking your browser"). An
    article that merely mentions CAPTCHAs is long; a small widget on a form is a note, not a stop."""
    text = str(probe.get("body") or "")
    return bool(_CHALLENGE.search(text)) and len(text.strip()) < 2000


def _page_text(probe: dict, find: str) -> str:
    """The page's own content (its <main>, when that holds real text) — or all of it for a find."""
    body, main = str(probe.get("body") or ""), str(probe.get("main") or "")
    return _clean_text(main if len(main.strip()) >= 500 and not find else body)


async def _aria(page) -> str:
    try:
        return await page.aria_snapshot(mode="ai", timeout=5000)
    except Exception:  # noqa: BLE001 — an older Playwright: the plain snapshot, resolved by role + name
        with suppress(Exception):
            return await page.locator("body").aria_snapshot(timeout=5000)
    return ""


async def _shot(page) -> bytes:
    return await page.screenshot(type="jpeg", quality=70, scale="css")


async def _front(page) -> None:
    with suppress(Exception):
        await page.bring_to_front()


def _handoff(kind: str, url: str) -> confirm.Halt:
    """NEEDS_USER in NEO's own words: the site's host, never text from the page or a model."""
    host = _host(url)
    msg = {
        "login": f"Please log in to {host} in NEO's browser window, then say continue.",
        "captcha": f"{host} wants a human check (a CAPTCHA). Please do it in NEO's browser window, then say continue.",
        "code": f"{host} wants a verification code. Please enter it in NEO's browser window, then say continue.",
        "payment": f"Please enter your payment details on {host} yourself in NEO's browser window, then say continue.",
    }.get(kind, f"Please take over in NEO's browser window on {host}, then say continue.")
    return confirm.ask_user(msg)


async def _read(
    page, max_chars: int = _READ_CHARS, find: str = "", notes: list[str] | None = None
) -> ToolOutput:
    """The snapshot of `page` (its numbers become the ones click/type take). A page that is only a
    human check hands the window to the user; one that is mostly graphics brings a screenshot."""
    notes = list(notes or [])
    if said := await _decline_consent(page):
        notes.append(said)
    probe = await _probe(page)
    waited = 0.0
    while _blocked(probe) and waited < _CHALLENGE_WAIT_S:
        await asyncio.sleep(1.0)
        waited += 1.0
        probe = await _probe(page)
    if _blocked(probe):
        await _front(page)
        return ToolOutput(_handoff("captcha", page.url))
    entries = _parse(await _aria(page))
    _b.remember(page, entries)
    text = _page_text(probe, find)
    n_els = sum(isinstance(e, _El) for e in entries)
    if probe.get("widget"):
        notes.append(
            "A CAPTCHA is on this page. NEO can't solve it: if the job needs to get past it, call browser_handoff kind=captcha."
        )
    if probe.get("code"):
        notes.append(
            "This page asks for a verification code. NEO never types codes: call browser_handoff kind=code if the job needs it."
        )
    elif probe.get("password"):
        notes.append(
            "This page asks to sign in. NEO never types passwords: if the job needs the account, call "
            "browser_handoff kind=login; otherwise carry on with what is readable."
        )
    thin = (len(text) < 200 and n_els < 4) or float(probe.get("canvas") or 0) >= 0.35
    images: list[ImagePart] = []
    if thin:
        with suppress(Exception):
            images = [ImagePart(await _shot(page), "image/jpeg")]
        if images:
            notes.append(
                "The page is mostly graphics: a screenshot is attached. browser_click takes x and y from it."
            )
    tabs = _b.pages()
    if len(tabs) > 1:
        notes.append(f"{len(tabs)} tabs are open (browser_tabs lists them).")
    try:
        title = await page.title()
    except Exception:  # noqa: BLE001
        title = ""
    return ToolOutput(_render(page.url, title, notes, entries, text, max_chars, find), images=images)


def _then(done: str, out: ToolOutput) -> ToolOutput:
    """An action's result with the page it led to (a question for the user stays as it is)."""
    if isinstance(out.text, confirm.Halt):
        return out
    return ToolOutput(f"{done}\n{out.text}", images=out.images)


async def _after(page, before: set[int], url0: str, done: str) -> ToolOutput | str:
    """After a click, a submit or a key: follow a new tab, or wait for a new page and read it; an
    in-place change gets a moment to finish."""
    new = await _b.follow(page, before, url0)
    if new is not None:
        await _settle(new)
        return _then(f"{done}{_b.popups()} It opened a new tab, which NEO follows now.", await _read(new))
    if page.url != url0:
        await _settle(page)
        return _then(done + _b.popups(), await _read(page))
    with suppress(Exception):
        await page.wait_for_load_state("networkidle", timeout=1500)
    return f"{done}{_b.popups()} The page stayed at {page.url} (browser_read shows what changed)."


# ---- cookie banners: declined, never accepted ------------------------------------------------------

_JS_CONSENT = r"""() => {
  const BANNER = /cookie|consent|gdpr|tracking technolog|personali[sz]ed (?:ads|advertising|content)|einwilligung|datenschutz|consentement|consentimiento|consenso|toestemming/i;
  const REJECT = /^(?:reject(?: all)?(?: cookies)?|reject (?:all |non[- ]essential |optional |additional |unnecessary )cookies|decline(?: all)?(?: cookies)?|decline (?:optional|non[- ]essential) cookies|deny(?: all)?|refuse(?: all)?|(?:use |allow |accept )?(?:only )?(?:strictly )?(?:necessary|essential|required)(?: cookies)?(?: only)?|continue without accepting|(?:i )?do not accept|alle ablehnen|ablehnen|nur (?:notwendige|erforderliche|essenzielle)(?: cookies)?(?: (?:akzeptieren|zulassen|erlauben))?|tout refuser|refuser(?: tout)?|continuer sans accepter|rechazar(?: todo| todas)?(?: las cookies)?|rifiuta(?: tutto| tutti)?|(?:alles )?weigeren|rejeitar(?: tudo| todos)?|recusar(?: tudo)?)$/i;
  const ACCEPT = /^(?:accept|agree|allow|i agree|i accept|ok|okay|got it|akzeptieren|alle akzeptieren|tout accepter|aceptar|accetta)/i;
  const vis = e => { const r = e.getBoundingClientRect(); if (r.width < 2 || r.height < 2) return false;
    const s = getComputedStyle(e); return s.visibility !== 'hidden' && s.display !== 'none' && +s.opacity !== 0; };
  const label = e => ((e.tagName === 'INPUT' ? e.value : e.innerText) || e.getAttribute('aria-label') || '').replace(/\s+/g, ' ').trim();
  const framed = window.top !== window;
  const overlay = e => {
    for (let a = e, i = 0; a && i < 14; a = a.parentElement, i++) {
      const st = getComputedStyle(a);
      if (st.position === 'fixed' || st.position === 'sticky' || a.tagName === 'DIALOG'
          || /^(?:alert)?dialog$/.test(a.getAttribute('role') || '')) return a;
    }
    return framed ? document.body : null;
  };
  document.querySelectorAll('[data-neo-consent]').forEach(e => e.removeAttribute('data-neo-consent'));
  let accept = '';
  for (const b of document.querySelectorAll('button,[role=button],a,input[type=button],input[type=submit]')) {
    const t = label(b);
    if (!t || t.length > 60 || !vis(b)) continue;
    const box = overlay(b);
    if (!box) continue;
    const words = (box.innerText || '').slice(0, 6000);
    if (words.length >= 6000 || !BANNER.test(words)) continue;
    if (REJECT.test(t)) { b.setAttribute('data-neo-consent', '1'); return {reject: t}; }
    if (!accept && ACCEPT.test(t)) accept = t;
  }
  return accept ? {accept} : null;
}"""


def _frames(page) -> list:
    main = getattr(page, "main_frame", page)
    rest = [f for f in (getattr(page, "frames", None) or []) if f is not main]
    return [main, *rest[:8]]


async def _decline_consent(page) -> str:
    """Decline a cookie banner that offers it (the privacy-preserving choice). One that only
    offers "Accept" is left alone and mentioned. Returns what to tell the model ("" = nothing)."""
    accept = ""
    for fr in _frames(page):
        try:
            got = await fr.evaluate(_JS_CONSENT)
        except Exception:  # noqa: BLE001 — a frame that went away
            continue
        if not isinstance(got, dict):
            continue
        if got.get("reject"):
            try:
                await fr.locator("[data-neo-consent]").first.click(timeout=3000)
            except Exception:  # noqa: BLE001 — covered or animating: a script click still reaches it
                try:
                    await fr.evaluate(
                        "() => { const e = document.querySelector('[data-neo-consent]'); if (e) e.click(); }"
                    )
                except Exception:  # noqa: BLE001
                    continue
            await asyncio.sleep(_SETTLE_S)
            return f"Declined a cookie banner (“{str(got['reject'])[:40]}”)."
        accept = accept or str(got.get("accept") or "")
    if accept:
        return (
            f"A cookie banner is showing with no way to decline it here (only “{accept[:40]}”). NEO never "
            "accepts one: work around it, or tell the user."
        )
    return ""


# ---- one element: what it says and where it sits ----------------------------------------------------

_JS_INFO = r"""e => {
  const button = e.tagName === 'INPUT' && /^(?:submit|button|reset|image)$/i.test(e.type);
  const own = [e.tagName === 'INPUT' ? '' : e.innerText, e.getAttribute('aria-label'), button ? e.value : '',
               e.title, e.getAttribute('alt')].filter(x => typeof x === 'string' && x.trim()).join(' ');
  const labels = e.labels ? [...e.labels].map(l => l.innerText).join(' ') : '';
  const box = e.closest('dialog,[role=dialog],[role=alertdialog],form');
  const a = e.closest('a[href]');
  return {
    tag: e.tagName.toLowerCase(), type: (e.getAttribute('type') || '').toLowerCase(),
    autocomplete: (e.getAttribute('autocomplete') || '').toLowerCase(),
    words: [e.name, e.id, e.placeholder, e.getAttribute('aria-label'), labels].filter(Boolean).join(' ').slice(0, 300),
    label: own.replace(/\s+/g, ' ').trim().slice(0, 200),
    href: a ? a.href : '', around: box ? box.innerText.slice(0, 1500) : '',
  };
}"""
_JS_POINT = (
    "([x, y]) => { const hit = document.elementFromPoint(x, y); if (!hit) return null;"
    " const e = hit.closest('a,button,input,select,textarea,label,[role=button],[role=link],[onclick]') || hit;"
    " return (" + _JS_INFO + ")(e); }"
)
_JS_IS_SEARCH = r"""e => {
  if (e.type === 'password') return false;
  if (e.type === 'search' || ['search', 'searchbox'].includes(e.getAttribute('role'))) return true;
  if (e.form && e.form.getAttribute('role') === 'search') return true;
  if (e.closest('[role=search],search')) return true;
  const words = [e.name, e.id, e.placeholder, e.getAttribute('aria-label')].join(' ').toLowerCase();
  if (/\bsearch\b|\bsearch[_-]|^q$|^query$/.test(words) || e.name === 'q' || e.name === 'query') return true;
  const action = e.form ? (e.form.getAttribute('action') || '') : '';
  return /\/search\b|[?&]q=/.test(action);
}"""


async def _info(loc) -> dict:
    try:
        got = await loc.first.evaluate(_JS_INFO, timeout=1500)
    except Exception:  # noqa: BLE001 — the action itself reports a missing element
        return {}
    return got if isinstance(got, dict) else {}


def _secret_kind(info: dict, name: str = "") -> str:
    """What a field NEO must not type into wants from the user: "login" (a password), "code" (a
    one-time code), "payment" (card or bank details) — or "" for an ordinary field."""
    auto = str(info.get("autocomplete") or "")
    words = f"{info.get('words') or ''} {name}"
    if (
        info.get("type") != "password"
        and not _SECRET_AUTOCOMPLETE.search(auto)
        and not _SECRET_FIELD.search(words)
    ):
        return ""
    if auto.startswith("cc-") or re.search(
        r"card|\bcvv|\bcvc\b|\bcsc\b|\biban\b|routing|account number", words, re.I
    ):
        return "payment"
    if auto == "one-time-code" or re.search(r"one[- ]?time|\botp\b|\b2fa\b|two[- ]factor|code", words, re.I):
        return "code"
    return "login"


async def _is_search(loc) -> bool:
    """A search box: submitting it only searches. Judged by what the field says it is — not by
    the form's method (every form without method="post" reports "get")."""
    try:
        return bool(await loc.first.evaluate(_JS_IS_SEARCH, timeout=1500))
    except Exception:  # noqa: BLE001
        return False


async def _link_risk(href: str) -> str:
    """After reading untrusted content: why following this link needs the user's OK ("" = none).
    Only where it goes: the page wrote the link, so it can't carry anything of the user's out (what
    NEO types into a page is gated where it's typed)."""
    from neo.tools.computer import policy

    if not href.lower().startswith(("http://", "https://")):
        return ""
    return await asyncio.to_thread(policy.private_host, href)


# ---- the tools ---------------------------------------------------------------------------------------


def _p(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or []}


@tool(
    "browser_open",
    "Open a web page in NEO's own browser (a separate Chromium window; logins made there persist). "
    "Returns the page as numbered elements plus its text. new_tab=true keeps the current page open.",
    _p({"url": {"type": "string"}, "new_tab": {"type": "boolean"}}, ["url"]),
    category="browser",
    slow=True,
    quiet=True,
    untrusted=True,
    fast_path=[
        (
            # "open example.com in your browser", "go to example.org in NEO's own browser"
            r"^\s*(?:please\s+)?(?:open|go\s+to|visit)\s+(?P<url>(?:https?://)?[\w-]+(?:\.[\w-]+)+(?:/\S*)?)\s+"
            r"(?:in|on|with|using)\s+(?:your|neo'?s)\s+(?:own\s+)?browser\s*[.!?]*\s*$",
            {"url": "<url>"},
        )
    ],
)
async def browser_open(a: dict, c: ToolContext) -> str | ToolOutput:
    from neo.tools.computer import policy

    url = _norm_url(str(a["url"]))
    scheme = url.split(":", 1)[0].lower()
    if scheme not in ("http", "https"):
        return f"Error: NEO's browser only opens web pages, not {scheme}: links."
    why = ""
    if c.tainted():
        # After reading a page or an email, a link that smuggles data out — or reaches this Mac's
        # own services — is the user's call.
        if policy.url_carries_data(url) and _no_fragment(url) not in _b.links:
            why = "the link carries data"
        elif private := await asyncio.to_thread(policy.private_host, url):
            why = private
    if why:
        gate = confirm.gated(
            "browser_open", a, f"open {url[:80]} ({why}, after reading {c.tainted()})", detail=url
        )
        if isinstance(gate, str):
            return gate
    page = await _b.page(new_tab=confirm.as_bool(a.get("new_tab")))
    notes: list[str] = []
    try:
        resp = await page.goto(url, wait_until="domcontentloaded", timeout=_NAV_MS)
    except Exception as e:  # noqa: BLE001
        if "download" in str(e).lower():
            return f"Error: {url} is a file download, not a page (web_fetch reads PDFs)."
        return f"Error: couldn't open {url}: {_err(e)}"
    status = getattr(resp, "status", 0) or 0
    if status >= 400:
        notes.append(f"The site answered {status}.")
    await _settle(page)
    return await _read(page, notes=notes)


@tool(
    "browser_read",
    "Read NEO's browser page: URL, title, the interactive elements numbered for browser_click and "
    'browser_type, and the visible text. find="words" keeps only what mentions them.',
    _p({"max_chars": {"type": "integer"}, "find": {"type": "string"}}),
    parallel_safe=True,
    category="browser",
    untrusted=True,
)
async def browser_read(a: dict, c: ToolContext) -> ToolOutput:
    page = await _b.page()
    limit = max(1000, min(int(a.get("max_chars") or _READ_CHARS), 20000))
    return await _read(page, limit, str(a.get("find") or ""))


@tool(
    "browser_click",
    'Click an element in NEO\'s browser: target = its number from the latest read ("12"), or its name; '
    "optionally role (button, link, checkbox, tab…). On a screenshot, pass x and y instead.",
    _p(
        {
            "target": {"type": "string"},
            "role": {"type": "string"},
            "x": {"type": "number"},
            "y": {"type": "number"},
        }
    ),
    category="browser",
    quiet=True,
    untrusted=True,
)
async def browser_click(a: dict, c: ToolContext) -> str | ToolOutput:
    page = await _b.page()
    target = str(a.get("target") or "").strip()
    at = a.get("x") is not None and a.get("y") is not None and not target
    loc, el = None, None
    if at:
        try:
            x, y = float(a["x"]), float(a["y"])
        except (TypeError, ValueError):
            return "Error: x and y must be numbers (pixels on the screenshot)."
        try:
            got = await page.evaluate(_JS_POINT, [x, y])
        except Exception:  # noqa: BLE001
            got = None
        info = got if isinstance(got, dict) else {}
        desc = f"the point ({x:g}, {y:g})"
    elif target:
        loc, el, err = await _resolve(page, target, str(a.get("role") or ""))
        if err:
            return err
        info = await _info(loc)
        desc = el.card() if el else f"“{target[:60]}”"
    else:
        return (
            "Error: say what to click — its number from browser_read, its name, or x and y on a screenshot."
        )
    from neo.tools.computer import policy

    own = str(info.get("label") or "").strip()
    words = own or (el.name if el else "") or target
    summary = ""
    if policy.commits(f"{target} {own} {el.name if el else ''}") or policy.commits(
        words, str(info.get("around") or "")
    ):
        summary = f"click “{words[:60]}” on {_host(page.url)}"
    elif at and info.get("tag") == "iframe":
        summary = f"click inside an embedded frame at ({a['x']}, {a['y']}) on {_host(page.url)}"
    elif c.tainted() and (href := str(info.get("href") or "")) and (why := await _link_risk(href)):
        summary = f"follow the link “{words[:40]}” to {href[:80]} ({why}, after reading {c.tainted()})"
    if summary:
        gate = confirm.gated("browser_click", a, summary, detail=f"{page.url}\n{el.card() if el else desc}")
        if isinstance(gate, str):
            return gate
    before, url0 = {id(p) for p in _b.pages()}, page.url
    try:
        if at:
            await page.mouse.click(float(a["x"]), float(a["y"]))
        else:
            await loc.click(timeout=_CLICK_MS)
    except Exception as e:  # noqa: BLE001
        return f"Error: couldn't click {desc}: {_err(e)}"
    return await _after(page, before, url0, f"Clicked {desc}.")


@tool(
    "browser_type",
    "Type into a field in NEO's browser: target = its number from the latest read, or its label or "
    "placeholder. A dropdown gets the option with that text. submit=true presses Enter. Never for "
    "passwords, codes or card details (browser_handoff).",
    _p(
        {"target": {"type": "string"}, "text": {"type": "string"}, "submit": {"type": "boolean"}},
        ["target", "text"],
    ),
    category="browser",
    quiet=True,
    untrusted=True,
)
async def browser_type(a: dict, c: ToolContext) -> str | ToolOutput:
    from neo.tools.computer import policy

    page = await _b.page()
    loc, el, err = await _resolve(page, str(a["target"]))
    if err:
        return err
    info = await _info(loc)
    if kind := _secret_kind(info, el.name if el else ""):
        await _front(page)
        return _handoff(kind, page.url)
    text, submit = str(a["text"]), confirm.as_bool(a.get("submit"))
    desc = el.card() if el else f"“{str(a['target'])[:60]}”"
    host = _host(page.url)
    summary = ""
    if submit and not await _is_search(loc):
        summary = f"submit the form on {host}"
    elif c.tainted() and policy.url_carries_data(f"https://{host}/?q={quote_plus(text)}"):
        # After reading untrusted content, a long or odd-looking text into a page is data leaving.
        summary = f"type “{text[:40]}{'…' if len(text) > 40 else ''}” into {desc} on {host} (after reading {c.tainted()})"
    if summary:
        gate = confirm.gated("browser_type", a, summary, detail=f"{page.url}\n{desc}: {text}")
        if isinstance(gate, str):
            return gate
    if info.get("type") in ("checkbox", "radio"):
        return f"Error: {desc} is a {info['type']} — click it instead."
    before, url0 = {id(p) for p in _b.pages()}, page.url
    try:
        if info.get("tag") == "select":
            try:
                await loc.select_option(label=text, timeout=_CLICK_MS)
            except Exception:  # noqa: BLE001 — the option's value rather than its label
                await loc.select_option(text, timeout=_CLICK_MS)
        else:
            await loc.fill(text, timeout=_CLICK_MS)
        if submit:
            await loc.press("Enter")
    except Exception as e:  # noqa: BLE001
        return f"Error: couldn't type into {desc}: {_err(e)}"
    if submit:
        return await _after(page, before, url0, f"Typed into {desc} and submitted.")
    return f"Typed into {desc}."


_PASTE = re.compile(r"^(?:control|ctrl|meta|cmd|command)\+v$|^shift\+insert$", re.I)


@tool(
    "browser_press",
    "Press a key in NEO's browser (Enter, Tab, Escape, ArrowDown, Control+a…).",
    _p({"key": {"type": "string"}}, ["key"]),
    category="browser",
    quiet=True,
    untrusted=True,
)
async def browser_press(a: dict, c: ToolContext) -> str | ToolOutput:
    key = str(a["key"]).strip()
    if _PASTE.match(key.replace(" ", "")):
        # The clipboard is the user's (it can hold a password): it never goes into a web page.
        return "Error: NEO doesn't paste the clipboard into web pages — type the text with browser_type."
    page = await _b.page()
    if key.split("+")[-1].lower() in ("enter", "return", "numpadenter"):
        focused = page.locator("*:focus").first
        if await focused.count() and not await _is_search(focused):
            gate = confirm.gated(
                "browser_press", a, f"press Enter on {_host(page.url)} — it submits the form", detail=page.url
            )
            if isinstance(gate, str):
                return gate
        before, url0 = {id(p) for p in _b.pages()}, page.url
        await page.keyboard.press(key)
        return await _after(page, before, url0, f"Pressed {key}.")
    await page.keyboard.press(key)
    await asyncio.sleep(_SETTLE_S)
    return f"Pressed {key}."


@tool(
    "browser_scroll",
    "Scroll NEO's browser page. direction: down|up; amount in pixels (default 700).",
    _p({"direction": {"type": "string", "enum": ["down", "up"]}, "amount": {"type": "integer"}}),
    category="browser",
    quiet=True,
)
async def browser_scroll(a: dict, c: ToolContext) -> str:
    page = await _b.page()
    dy = int(a.get("amount") or 700) * (-1 if a.get("direction") == "up" else 1)
    await page.mouse.wheel(0, dy)
    await asyncio.sleep(_SETTLE_S)
    return f"Scrolled {a.get('direction', 'down')}."


@tool(
    "browser_back",
    "Go back one page in NEO's browser.",
    _p({}),
    category="browser",
    quiet=True,
    untrusted=True,
)
async def browser_back(a: dict, c: ToolContext) -> str | ToolOutput:
    page = await _b.page()
    url0 = page.url
    try:
        resp = await page.go_back(wait_until="domcontentloaded", timeout=_NAV_MS)
    except Exception as e:  # noqa: BLE001
        return f"Error: couldn't go back: {_err(e)}"
    if resp is None and page.url == url0:
        return "Nothing to go back to in this tab."
    await _settle(page)
    return _then("Went back.", await _read(page))


@tool(
    "browser_tabs",
    "NEO's browser tabs: action=list (default), switch (to tab `index`) or close (tab `index`).",
    _p({"action": {"type": "string", "enum": ["list", "switch", "close"]}, "index": {"type": "integer"}}),
    category="browser",
    quiet=True,
    untrusted=True,
)
async def browser_tabs(a: dict, c: ToolContext) -> str | ToolOutput:
    await _b.page()
    pages = _b.pages()
    action = str(a.get("action") or "list").lower()
    if action == "list":
        rows = []
        for i, p in enumerate(pages, 1):
            try:
                title = await p.title()
            except Exception:  # noqa: BLE001
                title = ""
            rows.append(f"[{i}] {title[:80]} — {p.url[:100]}{' (current)' if p is _b.current else ''}")
        return "\n".join(rows) or "No tabs are open."
    try:
        i = int(a.get("index") or 0)
    except (TypeError, ValueError):
        i = 0
    if not 1 <= i <= len(pages):
        return f"Error: there's no tab {a.get('index')!r} — tabs are numbered 1 to {len(pages)}."
    page = pages[i - 1]
    if action == "close":
        await page.close()
        return f"Closed tab {i}. {len(_b.pages())} left."
    _b.current = page
    await _front(page)
    return _then(f"Switched to tab {i}.", await _read(page))


@tool(
    "browser_screenshot",
    "Screenshot NEO's browser page (when the numbered elements aren't enough). Click by x and y from it.",
    _p({}),
    category="browser",
    slow=True,
    untrusted=True,
)
async def browser_screenshot(a: dict, c: ToolContext) -> ToolOutput:
    page = await _b.page()
    img = await _shot(page)
    return ToolOutput(
        f"Screenshot of {page.url} ({_VIEWPORT['width']}×{_VIEWPORT['height']}).",
        images=[ImagePart(img, "image/jpeg")],
    )


@tool(
    "browser_handoff",
    "Hand NEO's browser to the user for what NEO never does itself: log in (kind=login), solve a CAPTCHA "
    "(captcha), enter a one-time code (code) or card details (payment). Stops and asks them to say continue.",
    _p({"kind": {"type": "string", "enum": ["login", "captcha", "code", "payment", "other"]}}, ["kind"]),
    category="browser",
)
async def browser_handoff(a: dict, c: ToolContext) -> str:
    page = await _b.page()
    await _front(page)
    return _handoff(str(a.get("kind") or "other"), page.url)


@tool("browser_close", "Close NEO's browser window.", _p({}), category="browser", quiet=True)
async def browser_close(a: dict, c: ToolContext) -> str:
    await _b.close()
    return "Browser closed."


async def shutdown() -> None:
    await _b.close()
