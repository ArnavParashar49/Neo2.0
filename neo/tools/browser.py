"""NEO's own browser: Playwright + Chromium with a persistent profile, driven through the page's
structure.

NEO's own Chromium — never the user's own browser — whose profile lives in <data_dir>/browser, so
the logins the user makes there persist (the privacy erase removes it). By default it runs headless
and the user watches it live in the chat's browser card (browser_live.py), where they can take it
over or pop it out; with browser_view = "window" it is a separate, visible Chromium window.
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
import functools
import json
import os
import plistlib
import re
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus, urljoin, urlsplit

from neo.agent import confirm
from neo.agent.registry import ToolContext, ToolOutput, tool
from neo.config import settings
from neo.events import current_job
from neo.providers.base import ImagePart
from neo.tools.browser_live import live, quoted

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
_VIEWPORT = {"width": 1280, "height": 860}  # the separate window's page ("window" mode)
_TOOL_S = 120.0  # one browser tool's own work
# The registry's cap on a browser tool: while the user has the page in the chat, NEO's next step
# waits for them to hand it back — that's their time, not the tool's (the card can always stop it).
_HELD_S = 6 * 3600.0


def _in_chat() -> bool:
    return settings().browser_view == "chat"


def _headless() -> bool:
    """In the chat, always: the card shows the page. A separate window is headless only for checks
    without a screen."""
    if _in_chat():
        return True
    return os.environ.get("NEO_BROWSER_HEADLESS", "").strip().lower() in ("1", "true", "yes")


def _chrome_ua(pw) -> str | None:
    """The user agent a visible Chrome of this build sends: headless Chromium says "HeadlessChrome",
    which some sites turn away — the page NEO shows should be the page the user would see."""
    if sys.platform != "darwin":
        return None
    try:
        exe = Path(pw.chromium.executable_path)
        app = next(p for p in (exe, *exe.parents) if p.suffix == ".app")
        info = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
        major = str(info.get("CFBundleShortVersionString", "")).split(".")[0]
    except Exception:  # noqa: BLE001 — Chromium's own user agent then
        return None
    if not major.isdigit():
        return None
    return (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
        f"Chrome/{major}.0.0.0 Safari/537.36"
    )


async def _appearance() -> str:
    """"dark" or "light", as macOS is set now: sites that support both match the overlay."""
    if sys.platform != "darwin":
        return "light"
    try:
        proc = await asyncio.create_subprocess_exec(
            "defaults", "read", "-g", "AppleInterfaceStyle",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )  # fmt: skip
        out, _ = await asyncio.wait_for(proc.communicate(), 2)
    except Exception:  # noqa: BLE001
        return "light"
    return "dark" if out.strip().lower() == b"dark" else "light"


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
        self._closing = False  # NEO is closing it itself
        self._lock = asyncio.Lock()
        self.dialogs: list[str] = []  # the pages' own pop-ups since the last action reported them
        self._current = None
        self.els: list[_El] = []
        self.els_page = None
        self.els_url = ""
        # Links pages showed NEO (absolute): opening one of them is following the page, not a URL a
        # model put together — it can't carry anything of the user's out.
        self.links: dict[str, None] = {}
        self.shown: dict[int, str] = {}  # id(tab) → the URL NEO last showed a model from it

    @property
    def current(self):
        return self._current

    @current.setter
    def current(self, page) -> None:
        """The tab NEO works in — the one the chat's card shows."""
        changed = page is not self._current
        self._current = page
        if changed and page is not None:
            live.page_changed(page)

    def cold(self) -> bool:
        """Is NEO's browser not running yet (a first step starts it)?"""
        return self._ctx is None or self._gone

    def saw(self, page) -> None:
        self.shown[id(page)] = str(page.url)
        if len(self.shown) > 64:
            live = {id(p) for p in self.pages()}
            self.shown = {k: v for k, v in self.shown.items() if k in live}

    async def page(self, *, new_tab: bool = False):
        async with self._lock:
            if self._ctx is not None and self._gone:
                await self._drop()  # the user closed NEO's browser window: start it again
            if self._ctx is None:
                await self._launch()
                live.launched()
            if new_tab:
                self.current = await self._ctx.new_page()
                cur = self.current
            else:
                cur = self.current
                if cur is None or cur.is_closed():
                    pages = self.pages()
                    cur = self.current = pages[-1] if pages else await self._ctx.new_page()
        await live.bind(cur)  # the chat's card watches the tab NEO works in
        return cur

    async def _launch(self) -> None:
        from playwright.async_api import async_playwright

        profile = settings().data_dir / "browser"
        profile.mkdir(parents=True, exist_ok=True)
        pw = await async_playwright().start()
        try:
            if _in_chat():
                # Chromium's new headless mode — the full browser, rendering as a visible one would
                # — at the card's size, Retina-crisp; no downloads (nothing reaches the Mac's disk).
                kw: dict = {
                    "headless": True,
                    "viewport": {"width": live.vp[0], "height": live.vp[1]},
                    "device_scale_factor": 2,
                    "color_scheme": await _appearance(),
                    "accept_downloads": False,
                    # A real 2x surface: the screencast captures the surface, not the emulated page.
                    "args": ["--disable-blink-features=AutomationControlled", "--force-device-scale-factor=2"],
                }
                if ua := _chrome_ua(pw):
                    kw["user_agent"] = ua
                try:
                    ctx = await pw.chromium.launch_persistent_context(str(profile), channel="chromium", **kw)
                except Exception:  # noqa: BLE001 — only the headless shell is installed
                    ctx = await pw.chromium.launch_persistent_context(str(profile), **kw)
            else:
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
        self._pw, self._ctx, self._gone, self._closing = pw, ctx, False, False

    def _closed(self, *_) -> None:
        self._gone = True
        live.browser_down(unexpected=not self._closing)

    def _watch(self, page) -> None:
        page.on("dialog", self._dialog)
        page.on("close", lambda p: self.shown.pop(id(p), None))
        live.new_page(page)  # a tab the user opened while they have the page: the card follows it

    async def _dialog(self, dialog) -> None:
        """A page's own alert / confirm / prompt. While the user has the page in the chat, the card
        asks them. Otherwise it's dismissed — "Cancel" is the choice that changes nothing — and
        told to the model with the next action's result. Leaving a page NEO was told to leave
        ("unsaved changes?") is accepted."""
        if live.user_dialog(dialog):
            return
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
        self._closing = True
        if self._ctx is not None:
            with suppress(Exception):
                await self._ctx.close()
        if self._pw is not None:
            with suppress(Exception):
                await self._pw.stop()
        self._ctx = self._pw = self._current = self.els_page = None
        self.els, self.els_url, self._gone, self.shown = [], "", False, {}
        live.browser_down(unexpected=False)

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
# The values of the inputs that hold a secret — by what the input is (type, autocomplete), not by
# what it is called: the page's snapshot shows every input's value, a password's too, and a label
# can be in any language (or missing).
_JS_SECRETS = r"""() => {
  const CODE = /one[- ]?time|\botp\b|verification code|security code|\b2fa\b|two[- ]factor|passcode|auth(?:entication)? code|\bcvv|\bcvc\b/i;
  const words = i => [i.name, i.id, i.placeholder, i.getAttribute('aria-label')].join(' ');
  const out = [];
  const walk = root => {
    for (const e of root.querySelectorAll('*')) {
      if (e.shadowRoot) walk(e.shadowRoot);
      if (e.tagName !== 'INPUT' || !e.value || out.length >= 50) continue;
      const auto = (e.getAttribute('autocomplete') || '').toLowerCase();
      if (e.type === 'password' || /password|one-time-code|\bcc-/.test(auto) || CODE.test(words(e))) out.push(e.value);
    }
  };
  walk(document);
  return out;
}"""


async def _secret_values(page) -> set[str] | None:
    """What the secret inputs of the page and its frames hold (None = the page itself couldn't
    be asked)."""
    got: set[str] = set()
    for i, fr in enumerate(_frames(page)):
        try:
            vals = await fr.evaluate(_JS_SECRETS)
        except Exception:  # noqa: BLE001 — a frame that went away
            if i == 0:
                return None
            continue
        got |= {" ".join(str(v).split()) for v in vals or [] if str(v).strip()}
    return got


def _mask(entries: list[_El | str], secrets: set[str] | None, label: str = "filled") -> None:
    """No secret's value in a snapshot: a field holding one shows as [filled]. When the page
    couldn't say what its secrets are, no field shows its value."""
    for e in entries:
        if not isinstance(e, _El) or not e.value or e.role not in _VALUED:
            continue
        if secrets is None or " ".join(e.value.split()) in secrets:
            e.value = ""
            e.states = (*e.states, label)


# What the fields the user typed into in the chat's card hold (browser_live marks each one as they
# type): the user's words are theirs, NEO's reads show those fields as [filled by you].
_JS_USER_FILLED = r"""() => {
  const out = [];
  const walk = root => {
    for (const e of root.querySelectorAll('[data-neo-user-filled]')) {
      const v = e.isContentEditable ? e.innerText : e.value;
      if (v && out.length < 50) out.push(String(v));
    }
    for (const e of root.querySelectorAll('*')) if (e.shadowRoot) walk(e.shadowRoot);
  };
  walk(document);
  return out;
}"""


async def _user_values(page) -> set[str] | None:
    got: set[str] = set()
    for i, fr in enumerate(_frames(page)):
        try:
            vals = await fr.evaluate(_JS_USER_FILLED)
        except Exception:  # noqa: BLE001
            if i == 0:
                return None
            continue
        got |= {" ".join(str(v).split()) for v in vals or [] if str(v).strip()}
    return got


# Words a human-check page says (on the page itself, not in a small widget's frame).
_CHALLENGE = re.compile(
    r"verify (?:that )?you(?:['’]| a)re (?:a )?human|i['’]?m not a robot|are you a (?:human|robot)|unusual traffic|"
    r"checking (?:if the site connection is secure|your browser)|press (?:and|&) hold|"
    r"complete the security check|(?:solve|complete) (?:the|this) (?:captcha|puzzle)|"
    r"(?:enter|type) the (?:characters|captcha|text|letters) (?:you see|seen|shown|below|in the image)",
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
    if _in_chat() and live.user_touched():
        # After the user had the page: what they filled (and every password box) is blacked out.
        with suppress(Exception):
            mask = [page.locator("[data-neo-user-filled], input[type=password]")]
            return await page.screenshot(type="jpeg", quality=70, scale="css", mask=mask)
    return await page.screenshot(type="jpeg", quality=70, scale="css")


async def _front(page) -> None:
    with suppress(Exception):
        await page.bring_to_front()


def _handoff(kind: str, url: str) -> confirm.Halt:
    """NEEDS_USER in NEO's own words: the site's host, never text from the page or a model."""
    host = _host(url)
    if _in_chat():
        msg = {
            "login": f"Please log in to {host} in NEO's browser in the chat, then hand it back — NEO won't see what you type.",
            "captcha": f"{host} wants a human check (a CAPTCHA). Please do it in NEO's browser in the chat, then hand it back.",
            "code": f"{host} wants a verification code. Please enter it in NEO's browser in the chat, then hand it back.",
            "payment": f"Please enter your payment details on {host} yourself in NEO's browser in the chat, then hand it back.",
        }.get(kind, f"Please take over in NEO's browser in the chat on {host}, then hand it back.")
        return confirm.ask_user(msg)
    msg = {
        "login": f"Please log in to {host} in NEO's browser window, then say continue.",
        "captcha": f"{host} wants a human check (a CAPTCHA). Please do it in NEO's browser window, then say continue.",
        "code": f"{host} wants a verification code. Please enter it in NEO's browser window, then say continue.",
        "payment": f"Please enter your payment details on {host} yourself in NEO's browser window, then say continue.",
    }.get(kind, f"Please take over in NEO's browser window on {host}, then say continue.")
    return confirm.ask_user(msg)


_NEED = {"login": "login", "code": "2fa", "captcha": "captcha", "payment": "payment"}


async def _hand_over(kind: str, page, detail: str = "") -> confirm.Halt:
    """NEO stops for the user: in the chat the card turns to them (its prompt built from the page's
    real host); a separate window comes to the front."""
    if _in_chat():
        await live.need_user(_NEED.get(kind, "other"), page=page, detail=detail)
    else:
        await _front(page)
    return _handoff(kind, str(page.url))


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
        return ToolOutput(await _hand_over("captcha", page))
    entries = _parse(await _aria(page))
    _mask(entries, await _secret_values(page))
    if _in_chat() and live.user_touched():
        _mask(entries, await _user_values(page), label="filled by you")
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
    _b.saw(page)
    return ToolOutput(_render(page.url, title, notes, entries, text, max_chars, find), images=images)


async def _open_risk(url: str, c: ToolContext) -> str:
    """After reading untrusted content: why opening `url` is the user's call ("" = it isn't) — a
    link that smuggles data out, or one that reaches this Mac's own services."""
    from neo.tools.computer import policy

    if not c.tainted():
        return ""
    if policy.url_carries_data(url) and _no_fragment(url) not in _b.links:
        return "the link carries data"
    return await asyncio.to_thread(policy.private_host, url)


def _open_card(a: dict, url: str, why: str, c: ToolContext) -> dict | str:
    return confirm.gated("browser_open", a, f"open {url[:80]} ({why}, after reading {c.tainted()})", detail=url)


def _hostname(url: str) -> str:
    try:
        return (urlsplit(str(url)).hostname or "").lower()
    except ValueError:
        return ""


async def _stray(page, c: ToolContext, start: str) -> str:
    """Why NEO must not show `page` now ("" = it may): after reading untrusted content it ended up
    on this Mac or the local network — a redirect, a script, a tab a page opened — having started
    on a public page. (Where NEO is sent is checked before it goes; this is where it arrived.)"""
    from neo.tools.computer import policy

    url = str(page.url)
    if not c.tainted() or not url.lower().startswith(("http://", "https://")):
        return ""
    if start and _hostname(start) == _hostname(url):
        return ""  # still the site it started on
    why = await asyncio.to_thread(policy.private_host, url)
    if not why:
        return ""
    if start.lower().startswith(("http://", "https://")) and await asyncio.to_thread(policy.private_host, start):
        return ""  # it was on this network already: where the user (or their OK) sent it
    return why


async def _leave(page, url: str, how: str) -> None:
    """Take NEO's browser off a page it mustn't show: close the tab it just followed ("close"),
    step back ("back") or forward again ("forward") — or, failing that, a blank page."""
    if how == "none":
        return
    with suppress(Exception):
        if how == "close":
            await page.close()
            return
        if how == "forward":
            await page.go_forward(wait_until="domcontentloaded", timeout=_NAV_MS)
        else:
            await page.go_back(wait_until="domcontentloaded", timeout=_NAV_MS)
    if not page.is_closed() and str(page.url) == url:
        with suppress(Exception):
            await page.goto("about:blank")


async def _keep_off(page, c: ToolContext, start: str, leave: str = "back") -> confirm.Halt | None:
    """A stray page (see _stray) is left unread, and opening it becomes the user's call: the very
    card browser_open asks for that URL, so their yes opens it."""
    why = await _stray(page, c, start)
    if not why:
        return None
    url = str(page.url)
    gate = _open_card({"url": url}, url, why, c)
    if not isinstance(gate, str):
        return None  # confirmation is turned off
    await _leave(page, url, leave)
    return gate


async def _show(page, c: ToolContext, start: str, *, leave: str = "back", **read) -> ToolOutput:
    """`_read(page)` — unless it is a page NEO mustn't show (see _keep_off)."""
    if stop := await _keep_off(page, c, start, leave):
        return ToolOutput(stop, halt=True, staged=stop.action_id)
    return await _read(page, **read)


def _halted(out: ToolOutput) -> bool:
    return out.halt or isinstance(out.text, confirm.Halt)


def _then(done: str, out: ToolOutput) -> ToolOutput:
    """An action's result with the page it led to (a question for the user stays as it is)."""
    if _halted(out):
        return out
    return ToolOutput(f"{done}\n{out.text}", images=out.images)


async def _after(page, before: set[int], url0: str, done: str, c: ToolContext) -> ToolOutput | str:
    """After a click, a submit or a key: follow a new tab, or wait for a new page and read it; an
    in-place change gets a moment to finish."""
    new = await _b.follow(page, before, url0)
    if new is not None:
        await _settle(new)
        out = await _show(new, c, url0, leave="close")
        if _halted(out):
            _b.current = page  # the tab it opened is closed again
        return _then(f"{done}{_b.popups()} It opened a new tab, which NEO follows now.", out)
    if page.url != url0:
        await _settle(page)
        return _then(done + _b.popups(), await _show(page, c, url0))
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
  const field = e.tagName === 'TEXTAREA' || e.tagName === 'SELECT' || (e.tagName === 'INPUT'
    && !/^(?:submit|button|reset|image|checkbox|radio|file|range|color|hidden)$/i.test(e.type));
  return {
    tag: e.tagName.toLowerCase(), type: (e.getAttribute('type') || '').toLowerCase(),
    editable: field || e.isContentEditable || /^(?:textbox|searchbox|combobox)$/.test(e.getAttribute('role') || ''),
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


async def _timed(coro, name: str):
    try:
        return await asyncio.wait_for(coro, _TOOL_S)
    except TimeoutError:
        return f"Error: {name} took longer than {int(_TOOL_S)}s and was stopped"


# Tools that act on what the model saw of the page: after the user had the page they re-read it
# instead (the numbers, the page itself may have changed). The rest run as asked.
_ACTS = {"open", "click", "type", "press", "scroll", "back", "tabs"}
_USER_WAS_HERE = (
    "The user used the browser while you waited, so NEO didn't do that step. The page may have "
    "changed: act on this read, not an earlier one (fields the user filled show as [filled by you])."
)
_USER_READ = "The user used the browser while you waited (fields the user filled show as [filled by you])."


def _card(kind: str):
    """A NEO browser tool, as the chat's browser card sees it: the card opens for the job that
    calls it; while the user has the page the tool waits (then re-reads instead of acting on a page
    that may have changed); and it holds the page while it acts, so the user's input and NEO's never
    mix. In "window" mode the tool just runs."""

    def deco(fn):
        @functools.wraps(fn)
        async def run(a: dict, c: ToolContext):
            name = fn.__name__
            if not _in_chat():
                return await _timed(fn(a, c), name)
            job = current_job.get()
            url = _norm_url(str(a.get("url") or "")) if kind == "open" and a.get("url") else ""
            live.claim(job, url=url, starting=_b.cold())
            waited = await live.enter()
            try:
                if live.s is None:
                    live.claim(job, starting=_b.cold())  # the card closed while this step waited
                if waited and kind in _ACTS:
                    return await _timed(_reread(), name)
                out = await _timed(fn(a, c), name)
                if waited and kind == "read" and isinstance(out, ToolOutput) and not _halted(out):
                    return ToolOutput(f"{_USER_READ}\n{out.text}", images=out.images)
                return out
            finally:
                live.leave()

        return run

    return deco


async def _reread() -> ToolOutput:
    page = await _b.page()
    await live.intent("read", "Read the page again", present=False)
    out = await _read(page)
    live.done()
    if _halted(out):
        return out
    return ToolOutput(f"{_USER_WAS_HERE}\n{out.text}", images=out.images)


async def _box(loc) -> dict | None:
    """Where an element is on the screen (viewport CSS px), scrolled into view first."""
    if loc is None:
        return None
    with suppress(Exception):
        await loc.first.scroll_into_view_if_needed(timeout=2000)
    try:
        return await loc.first.bounding_box(timeout=2000)
    except Exception:  # noqa: BLE001 — no box: the card shows no highlight
        return None


def _gate_card(gate: confirm.Halt | str, step: str) -> None:
    """The step waits on the user's OK: the card spotlights its target."""
    if _in_chat() and isinstance(gate, confirm.Halt) and gate.action_id:
        live.waiting(gate.action_id, step)


@tool(
    "browser_open",
    "Open a web page in NEO's own browser (its own Chromium with its own logins, which the user can watch). "
    "Returns the page as numbered elements plus its text. new_tab=true keeps the current page open.",
    _p({"url": {"type": "string"}, "new_tab": {"type": "boolean"}}, ["url"]),
    category="browser",
    slow=True,
    quiet=True,
    untrusted=True,
    timeout=_HELD_S,
    fast_path=[
        (
            # "open example.com in your browser", "go to example.org in NEO's own browser"
            r"^\s*(?:please\s+)?(?:open|go\s+to|visit)\s+(?P<url>(?:https?://)?[\w-]+(?:\.[\w-]+)+(?:/\S*)?)\s+"
            r"(?:in|on|with|using)\s+(?:your|neo'?s)\s+(?:own\s+)?browser\s*[.!?]*\s*$",
            {"url": "<url>"},
        )
    ],
)
@_card("open")
async def browser_open(a: dict, c: ToolContext) -> str | ToolOutput:
    url = _norm_url(str(a["url"]))
    scheme = url.split(":", 1)[0].lower()
    if scheme not in ("http", "https"):
        return f"Error: NEO's browser only opens web pages, not {scheme}: links."
    host = _host(url)
    # After reading a page or an email, a link that smuggles data out — or reaches this Mac's own
    # services — is the user's call.
    if why := await _open_risk(url, c):
        gate = _open_card(a, url, why, c)
        if isinstance(gate, str):
            await live.intent("navigate", f"Asked to open {host}", url=url, present=False)
            _gate_card(gate, f"Asked to open {host}")
            return gate
    page = await _b.page(new_tab=confirm.as_bool(a.get("new_tab")))
    notes: list[str] = []
    await live.intent("navigate", f"Opened {host}", url=url)
    try:
        resp = await page.goto(url, wait_until="domcontentloaded", timeout=_NAV_MS)
    except Exception as e:  # noqa: BLE001
        live.fail(_err(e), f"Couldn't open {host}")
        if "download" in str(e).lower():
            return f"Error: {url} is a file download, not a page (web_fetch reads PDFs)."
        return f"Error: couldn't open {url}: {_err(e)}"
    status = getattr(resp, "status", 0) or 0
    if status >= 400:
        notes.append(f"The site answered {status}.")
    await _settle(page)
    live.done(f"Opened {_host(page.url)}")
    out = await _show(page, c, url, notes=notes)
    if c.origin in ("user", "live") and not _halted(out):
        # The user's own words through a fast path ("open example.com in your browser"): no model
        # reads this page, and the reply is kept as NEO's own words — so it's NEO's line, not the
        # site's text (declining its cookie banner and a human check still happened above).
        said = f" (it answered {status})" if status >= 400 else ""
        return ToolOutput(f"Opened {_host(page.url)} in NEO's browser{said}.", say=False)
    return out


@tool(
    "browser_read",
    "Read NEO's browser page: URL, title, the interactive elements numbered for browser_click and "
    'browser_type, and the visible text. find="words" keeps only what mentions them.',
    _p({"max_chars": {"type": "integer"}, "find": {"type": "string"}}),
    parallel_safe=True,
    category="browser",
    untrusted=True,
    timeout=_HELD_S,
)
@_card("read")
async def browser_read(a: dict, c: ToolContext) -> ToolOutput:
    page = await _b.page()
    limit = max(1000, min(int(a.get("max_chars") or _READ_CHARS), 20000))
    find = str(a.get("find") or "")
    await live.intent("read", f"Looked for {quoted(find)} on the page" if find.strip() else "Read the page", present=False)
    out = await _show(page, c, _b.shown.get(id(page), ""), max_chars=limit, find=find)
    live.done()
    return out


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
    timeout=_HELD_S,
)
@_card("click")
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
    said = quoted(words) if words else "the page"
    await live.intent(
        "click", f"Clicked {said}", rect=await _box(loc), x=float(a["x"]) if at else None,
        y=float(a["y"]) if at else None, label=words, role=(el.role if el else "") or str(info.get("tag") or ""),
        remeasure=(lambda: loc.first.bounding_box(timeout=2000)) if loc is not None else None,
    )  # fmt: skip
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
            _gate_card(gate, f"Asked to click {said}")
            return gate
    before, url0 = {id(p) for p in _b.pages()}, page.url
    try:
        if at:
            await page.mouse.click(float(a["x"]), float(a["y"]))
        else:
            await loc.click(timeout=_CLICK_MS)
    except Exception as e:  # noqa: BLE001
        live.fail(_err(e), f"Couldn't click {said}")
        return f"Error: couldn't click {desc}: {_err(e)}"
    live.done()
    return await _after(page, before, url0, f"Clicked {desc}.", c)


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
    timeout=_HELD_S,
)
@_card("type")
async def browser_type(a: dict, c: ToolContext) -> str | ToolOutput:
    from neo.tools.computer import policy

    page = await _b.page()
    loc, el, err = await _resolve(page, str(a["target"]))
    if err:
        return err
    info = await _info(loc)
    if kind := _secret_kind(info, el.name if el else ""):
        return await _hand_over(kind, page)
    text, submit = str(a["text"]), confirm.as_bool(a.get("submit"))
    desc = el.card() if el else f"“{str(a['target'])[:60]}”"
    host = _host(page.url)
    field = (el.name if el else "") or str(a["target"])
    choose = info.get("tag") == "select"
    step = (
        f"Chose {quoted(text)} in {field[:40]}"
        if choose
        else f"Typed {quoted(text)} in {field[:40]}" + (" and pressed Enter" if submit else "")
    )
    await live.intent(
        "select" if choose else "type", step, rect=await _box(loc), label=field, fieldname=field, text=text,
        role=el.role if el else "", remeasure=lambda: loc.first.bounding_box(timeout=2000),
    )  # fmt: skip
    summary = ""
    if submit and not await _is_search(loc):
        summary = f"submit the form on {host}"
    elif c.tainted() and policy.url_carries_data(f"https://{host}/?q={quote_plus(text)}"):
        # After reading untrusted content, a long or odd-looking text into a page is data leaving.
        summary = f"type “{text[:40]}{'…' if len(text) > 40 else ''}” into {desc} on {host} (after reading {c.tainted()})"
    if summary:
        gate = confirm.gated("browser_type", a, summary, detail=f"{page.url}\n{desc}: {text}")
        if isinstance(gate, str):
            _gate_card(gate, f"Asked to submit the form on {host}" if submit else f"Asked to type {quoted(text)}")
            return gate
    if info.get("type") in ("checkbox", "radio"):
        live.fail("not a text field")
        return f"Error: {desc} is a {info['type']} — click it instead."
    before, url0 = {id(p) for p in _b.pages()}, page.url
    try:
        if choose:
            try:
                await loc.select_option(label=text, timeout=_CLICK_MS)
            except Exception:  # noqa: BLE001 — the option's value rather than its label
                await loc.select_option(text, timeout=_CLICK_MS)
        else:
            await loc.fill(text, timeout=_CLICK_MS)
        if submit:
            await loc.press("Enter")
    except Exception as e:  # noqa: BLE001
        live.fail(_err(e), f"Couldn't type into {field[:40]}")
        return f"Error: couldn't type into {desc}: {_err(e)}"
    live.done()
    if submit:
        return await _after(page, before, url0, f"Typed into {desc} and submitted.", c)
    return f"Typed into {desc}."


_MODIFIERS = {
    "control": "control", "ctrl": "control", "meta": "meta", "cmd": "meta", "command": "meta",
    "controlormeta": "meta", "shift": "shift", "alt": "alt", "option": "alt",
}  # fmt: skip
# Keys that only move around a page: never a command, whatever a page asked for.
_MOVING = {
    "tab", "escape", "esc", "arrowup", "arrowdown", "arrowleft", "arrowright", "up", "down", "left",
    "right", "pageup", "pagedown", "home", "end", "shift", "control", "alt", "meta",
}  # fmt: skip


def _combo(key: str) -> tuple[set[str], str]:
    """(modifiers, key) of a key combo as Playwright takes it, however it's spelled: "Meta+KeyV",
    "ControlOrMeta+v", "cmd + V" and " " all come out plain ({"meta"}, "v"), (set(), "space")."""
    if not key.strip():
        return set(), "space" if key else ""
    raw = key.strip()
    if raw == "+" or raw.endswith("++"):
        parts, last = raw[:-2].split("+") if len(raw) > 1 else [], "+"
    else:
        *parts, last = raw.split("+")
    mods = {_MODIFIERS.get(p.strip().lower(), p.strip().lower()) for p in parts if p.strip()}
    last = last.strip()
    if re.fullmatch(r"(?:Key[A-Za-z]|Digit\d)", last):
        last = last[-1]
    last = last.lower()
    return mods, {"spacebar": "space", "return": "enter", "numpadenter": "enter", "del": "delete"}.get(last, last)


def _pastes(mods: set[str], last: str) -> bool:
    return (last == "v" and bool(mods & {"control", "meta"})) or (last == "insert" and "shift" in mods)


def _editable(info: dict) -> bool:
    if "editable" in info:
        return bool(info["editable"])
    tag, kind = info.get("tag"), str(info.get("type") or "")
    return tag in ("textarea", "select") or (
        tag == "input" and kind not in ("submit", "button", "reset", "image", "checkbox", "radio", "file")
    )


@tool(
    "browser_press",
    "Press a key in NEO's browser (Enter, Tab, Escape, ArrowDown, Control+a…).",
    _p({"key": {"type": "string"}}, ["key"]),
    category="browser",
    quiet=True,
    untrusted=True,
    timeout=_HELD_S,
)
@_card("press")
async def browser_press(a: dict, c: ToolContext) -> str | ToolOutput:
    from neo.tools.computer import policy

    key = str(a["key"])
    mods, last = _combo(key)
    key = key.strip() or "Space"
    if _pastes(mods, last):
        # The clipboard is the user's (it can hold a password): it never goes into a web page.
        return "Error: NEO doesn't paste the clipboard into web pages — type the text with browser_type."
    page = await _b.page()
    host = _host(page.url)
    focused = page.locator("*:focus").first
    has_focus = last not in _MOVING and bool(await focused.count())
    info = await _info(focused) if has_focus else {}
    field = _editable(info)
    label = "" if field else str(info.get("label") or "").strip()  # a field's "label" is its text
    await live.intent("key", f"Pressed {key[:20]}", key=key[:32], label=label)
    summary = ""
    if last == "enter":
        if has_focus and not await _is_search(focused):
            summary = (
                f"press Enter on “{label[:60]}” on {host}"
                if label
                else f"press Enter on {host} — it submits the form"
            )
    elif last == "space" and not field and not mods & {"control", "meta", "alt"}:
        # Space on a focused button presses it, like a click.
        if policy.commits(label, str(info.get("around") or "")):
            summary = f"press Space on “{label[:60]}” on {host}"
        elif info.get("tag") == "iframe":
            summary = f"press Space inside an embedded frame on {host}"
    elif (
        c.tainted()
        and (len(last) == 1 or last in ("delete", "backspace"))
        and not (len(last) == 1 and mods & {"control", "meta"})
        and not field
    ):
        # Outside a text field a letter is a web app's shortcut ("#" deletes, "l" likes, "r"
        # replies…): after reading a page or an email, the user's call.
        summary = f"press {key[:20]} on {host} — outside a text field a page can treat it as a command (after reading {c.tainted()})"
    if summary:
        gate = confirm.gated("browser_press", a, summary, detail=f"{page.url}\n{label[:80]}".strip())
        if isinstance(gate, str):
            _gate_card(gate, f"Asked to press {key[:20]}")
            return gate
    if last in ("enter", "space"):
        before, url0 = {id(p) for p in _b.pages()}, page.url
        await page.keyboard.press(key)
        live.done()
        return await _after(page, before, url0, f"Pressed {key}.", c)
    await page.keyboard.press(key)
    live.done()
    await asyncio.sleep(_SETTLE_S)
    return f"Pressed {key}."


@tool(
    "browser_scroll",
    "Scroll NEO's browser page. direction: down|up; amount in pixels (default 700).",
    _p({"direction": {"type": "string", "enum": ["down", "up"]}, "amount": {"type": "integer"}}),
    category="browser",
    quiet=True,
    timeout=_HELD_S,
)
@_card("scroll")
async def browser_scroll(a: dict, c: ToolContext) -> str:
    page = await _b.page()
    up = a.get("direction") == "up"
    amount = int(a.get("amount") or 700)
    await live.intent("scroll", f"Scrolled {'up' if up else 'down'}", direction="up" if up else "down", amount=amount, present=False)
    await page.mouse.wheel(0, amount * (-1 if up else 1))
    await asyncio.sleep(_SETTLE_S)
    live.done()
    return f"Scrolled {a.get('direction', 'down')}."


@tool(
    "browser_back",
    "Go back one page in NEO's browser.",
    _p({}),
    category="browser",
    quiet=True,
    untrusted=True,
    timeout=_HELD_S,
)
@_card("back")
async def browser_back(a: dict, c: ToolContext) -> str | ToolOutput:
    page = await _b.page()
    url0 = page.url
    await live.intent("navigate", "Went back", caption="Going back", present=False)
    try:
        resp = await page.go_back(wait_until="domcontentloaded", timeout=_NAV_MS)
    except Exception as e:  # noqa: BLE001
        live.fail(_err(e), "Couldn't go back")
        return f"Error: couldn't go back: {_err(e)}"
    if resp is None and page.url == url0:
        live.fail("nothing to go back to", "Nothing to go back to")
        return "Nothing to go back to in this tab."
    await _settle(page)
    live.done()
    return _then("Went back.", await _show(page, c, url0, leave="forward"))


@tool(
    "browser_tabs",
    "NEO's browser tabs: action=list (default), switch (to tab `index`) or close (tab `index`).",
    _p({"action": {"type": "string", "enum": ["list", "switch", "close"]}, "index": {"type": "integer"}}),
    category="browser",
    quiet=True,
    untrusted=True,
    timeout=_HELD_S,
)
@_card("tabs")
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
        await live.intent("navigate", f"Closed tab {i}", caption=f"Closing tab {i}", present=False)
        await page.close()
        live.done()
        return f"Closed tab {i}. {len(_b.pages())} left."
    if stop := await _keep_off(page, c, _b.shown.get(id(page), ""), leave="none"):
        return stop
    await live.intent("navigate", f"Switched to tab {i}", caption=f"Switching to tab {i}", present=False)
    _b.current = page
    await _front(page)
    await live.bind(page)
    live.done()
    return _then(f"Switched to tab {i}.", await _read(page))


@tool(
    "browser_screenshot",
    "Screenshot NEO's browser page (when the numbered elements aren't enough). Click by x and y from it.",
    _p({}),
    category="browser",
    slow=True,
    untrusted=True,
    timeout=_HELD_S,
)
@_card("shot")
async def browser_screenshot(a: dict, c: ToolContext) -> ToolOutput:
    page = await _b.page()
    if stop := await _keep_off(page, c, _b.shown.get(id(page), "")):
        return ToolOutput(stop, halt=True, staged=stop.action_id)
    await live.intent("read", "Looked at the page", caption="Looking at the page", present=False)
    img = await _shot(page)
    live.done()
    _b.saw(page)
    size = (getattr(page, "viewport_size", None) or _VIEWPORT) if _in_chat() else _VIEWPORT
    return ToolOutput(
        f"Screenshot of {page.url} ({size['width']}×{size['height']}).",
        images=[ImagePart(img, "image/jpeg")],
    )


@tool(
    "browser_handoff",
    "Hand NEO's browser to the user for what NEO never does itself: log in (kind=login), solve a CAPTCHA "
    "(captcha), enter a one-time code (code) or card details (payment). Stops and asks them to take over. "
    "For kind=other, detail says in a few words what they should do.",
    _p(
        {
            "kind": {"type": "string", "enum": ["login", "captcha", "code", "payment", "other"]},
            "detail": {"type": "string"},
        },
        ["kind"],
    ),
    category="browser",
    timeout=_HELD_S,
)
@_card("handoff")
async def browser_handoff(a: dict, c: ToolContext) -> str:
    page = await _b.page()
    kind = str(a.get("kind") or "other")
    detail = " ".join(str(a.get("detail") or "").split())[:120] if kind not in _NEED else ""
    return await _hand_over(kind, page, detail=detail)


@tool("browser_close", "Close NEO's browser.", _p({}), category="browser", quiet=True)
async def browser_close(a: dict, c: ToolContext) -> str:
    if _in_chat():
        await live.end("done", alive=False)
    await _b.close()
    return "Browser closed."


async def relaunch(url: str) -> None:
    """Start NEO's browser again (it closed or crashed under the card) at the page it was on."""
    await _b.close()
    page = await _b.page()
    if url.lower().startswith(("http://", "https://")):
        with suppress(Exception):
            await page.goto(url, wait_until="commit", timeout=_NAV_MS)


async def shutdown(reason: str = "") -> None:
    """NEO is quitting ("") or the user erased everything ("erased")."""
    with suppress(Exception):
        await live.shutdown(reason)
    await _b.close()
