"""A real browser tool (Playwright + Chromium, persistent profile).

Safari-via-AppleScript can't reliably click, fill forms or download. This gives the agent a
headed Chromium it drives directly: the profile lives in ~/.neo/browser so logins persist, and
the window is visible so the user can step in for CAPTCHAs or 2FA. Pages are read as an
accessibility (ARIA) snapshot — roles and names — which is what the click/type tools target.
"""

from __future__ import annotations

import asyncio
import re

from neo import runtime
from neo.agent.registry import ToolContext, ToolOutput, tool
from neo.config import settings
from neo.providers.base import ImagePart

_TIMEOUT_MS = 8000


class _Browser:
    def __init__(self) -> None:
        self._pw = None
        self._ctx = None
        self._lock = asyncio.Lock()

    async def page(self):
        async with self._lock:
            if self._ctx is None:
                from playwright.async_api import async_playwright

                self._pw = await async_playwright().start()
                s = settings()
                profile = s.data_dir / "browser"
                profile.mkdir(mode=0o700, parents=True, exist_ok=True)
                self._ctx = await self._pw.chromium.launch_persistent_context(
                    str(profile),
                    headless=s.browser_headless,
                    viewport={"width": 1280, "height": 860},
                    args=["--disable-blink-features=AutomationControlled"],
                )
                self._ctx.set_default_timeout(_TIMEOUT_MS)
            pages = self._ctx.pages
            return pages[-1] if pages else await self._ctx.new_page()

    async def close(self) -> None:
        async with self._lock:
            if self._ctx:
                await self._ctx.close()
            if self._pw:
                await self._pw.stop()
            self._ctx = self._pw = None


_b = _Browser()


def _browser() -> _Browser:
    """The bound agent's own browser (neo.runtime), else NEO.app's."""
    rt = runtime.current_runtime()
    if rt is not None:
        return rt.get_or_create("browser", _Browser)
    runtime.check_unbound("NEO's browser")
    return _b


def _norm_url(url: str) -> str:
    url = url.strip()
    if not re.match(r"^[a-z]+://", url):
        url = "https://" + url
    return url


def _locator(page, target: str, role: str = ""):
    """Best-effort element resolution: role+name → label/placeholder → visible text → CSS."""
    t = target.strip()
    if role:
        return page.get_by_role(role, name=re.compile(re.escape(t), re.I)).first
    if t.startswith(("#", ".", "//", "css=", "xpath=", "[")) or re.match(r"^[a-z]+(\[|\.|#|$)", t):
        try:
            return page.locator(t).first
        except Exception:  # noqa: BLE001 — not a selector after all
            pass
    return (
        page.get_by_role("button", name=re.compile(re.escape(t), re.I))
        .or_(page.get_by_role("link", name=re.compile(re.escape(t), re.I)))
        .or_(page.get_by_label(re.compile(re.escape(t), re.I)))
        .or_(page.get_by_placeholder(re.compile(re.escape(t), re.I)))
        .or_(page.get_by_text(t, exact=False))
        .first
    )


async def _snapshot(page, max_chars: int) -> str:
    title = await page.title()
    head = f"URL: {page.url}\nTitle: {title}\n"
    try:
        aria = await page.locator("body").aria_snapshot()
    except Exception:  # noqa: BLE001 — older Playwright
        aria = ""
    if aria:
        # Keep the interactive/structural lines; drop empty generic containers.
        lines = [ln for ln in aria.splitlines() if not re.search(r"^\s*- (generic|group)\s*:?\s*$", ln)]
        aria = "\n".join(lines)
    text = await page.evaluate("() => document.body ? document.body.innerText : ''")
    body = f"\n[ELEMENTS]\n{aria[: max_chars // 2]}\n\n[TEXT]\n{text[: max_chars // 2]}"
    return head + body


def _p(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or []}


@tool(
    "browser_open",
    "Open a URL in NEO's own browser (Chromium, logins persist). Returns the page's elements and text.",
    _p({"url": {"type": "string"}}, ["url"]),
    category="browser",
    slow=True,
    quiet=True,
    untrusted=True,
)
async def browser_open(a: dict, c: ToolContext) -> str:
    from neo.agent import confirm
    from neo.tools.computer import policy

    url = _norm_url(a["url"])
    scheme = url.split(":", 1)[0].lower()
    if scheme not in ("http", "https"):
        return f"Error: NEO's browser only opens web pages, not {scheme}: links."
    why = ""
    if c.tainted():
        # After reading a page or an email, a link that smuggles data out — or reaches this Mac's
        # own services — is the user's call.
        if policy.url_carries_data(url):
            why = "the link carries data"
        elif private := await asyncio.to_thread(policy.private_host, url):
            why = private
    if why:
        gate = confirm.gated("browser_open", a, f"open {url[:80]} ({why}, after reading {c.tainted()})", detail=url)
        if isinstance(gate, str):
            return gate
    page = await _browser().page()
    await page.goto(url, wait_until="domcontentloaded")
    await asyncio.sleep(0.4)
    return await _snapshot(page, 9000)


@tool(
    "browser_read",
    "Read the current browser page: URL, title, interactive elements (roles + names) and visible text.",
    _p({"max_chars": {"type": "integer"}}),
    parallel_safe=True,
    category="browser",
    untrusted=True,
)
async def browser_read(a: dict, c: ToolContext) -> str:
    page = await _browser().page()
    return await _snapshot(page, int(a.get("max_chars") or 9000))


@tool(
    "browser_click",
    "Click an element on the current page by its visible name/text (from browser_read), optionally with its "
    "role (button, link, checkbox, tab…), or a CSS selector.",
    _p({"target": {"type": "string"}, "role": {"type": "string"}}, ["target"]),
    category="browser",
    quiet=True,
)
async def browser_click(a: dict, c: ToolContext) -> str:
    from neo.agent import confirm
    from neo.tools.computer import policy

    page = await _browser().page()
    loc = _locator(page, a["target"], a.get("role", ""))
    own = await _label(loc)
    label = str(a["target"]) + " " + own
    if policy.commits(label) or policy.commits(own.strip() or str(a["target"]), await _surroundings(loc)):
        gate = confirm.gated("browser_click", a, f"click “{label.strip()[:60]}” on {_host(page.url)}", detail=page.url)
        if isinstance(gate, str):
            return gate
    try:
        await loc.click()
    except Exception as e:  # noqa: BLE001
        return f"Error: couldn't click {a['target']!r}: {str(e).splitlines()[0][:160]}"
    await asyncio.sleep(0.5)
    return f"Clicked {a['target']!r}. Now at {page.url}"


@tool(
    "browser_type",
    "Type into a field on the current page (by label, placeholder, name or CSS selector). submit=true presses Enter.",
    _p(
        {"target": {"type": "string"}, "text": {"type": "string"}, "submit": {"type": "boolean"}},
        ["target", "text"],
    ),
    category="browser",
    quiet=True,
)
async def browser_type(a: dict, c: ToolContext) -> str:
    from neo.agent import confirm

    page = await _browser().page()
    loc = _locator(page, a["target"])
    if confirm.as_bool(a.get("submit")) and not await _is_search(loc):
        gate = confirm.gated(
            "browser_type", a, f"submit the form on {_host(page.url)}", detail=f"{page.url}\n{a['target']}: {a['text']}"
        )
        if isinstance(gate, str):
            return gate
    try:
        await loc.fill(a["text"])
        if a.get("submit"):
            await loc.press("Enter")
            await asyncio.sleep(0.6)
    except Exception as e:  # noqa: BLE001
        return f"Error: couldn't type into {a['target']!r}: {str(e).splitlines()[0][:160]}"
    return f"Typed into {a['target']!r}" + (" and submitted." if a.get("submit") else ".")


@tool(
    "browser_press",
    "Press a key in the browser (Enter, Tab, Escape, ArrowDown, Control+a…).",
    _p({"key": {"type": "string"}}, ["key"]),
    category="browser",
    quiet=True,
)
async def browser_press(a: dict, c: ToolContext) -> str:
    from neo.agent import confirm

    page = await _browser().page()
    if str(a["key"]).split("+")[-1].lower() in ("enter", "return", "numpadenter"):
        focused = page.locator("*:focus").first
        if await focused.count() and not await _is_search(focused):
            gate = confirm.gated("browser_press", a, f"press Enter on {_host(page.url)} — it submits the form", detail=page.url)
            if isinstance(gate, str):
                return gate
    await page.keyboard.press(a["key"])
    return f"Pressed {a['key']}"


@tool(
    "browser_scroll",
    "Scroll the page. direction: down|up; amount in pixels (default 700).",
    _p({"direction": {"type": "string", "enum": ["down", "up"]}, "amount": {"type": "integer"}}),
    category="browser",
    quiet=True,
)
async def browser_scroll(a: dict, c: ToolContext) -> str:
    page = await _browser().page()
    dy = int(a.get("amount") or 700) * (-1 if a.get("direction") == "up" else 1)
    await page.mouse.wheel(0, dy)
    await asyncio.sleep(0.3)
    return f"Scrolled {a.get('direction', 'down')}."


@tool("browser_back", "Go back one page in the browser.", _p({}), category="browser", quiet=True)
async def browser_back(a: dict, c: ToolContext) -> str:
    page = await _browser().page()
    await page.go_back(wait_until="domcontentloaded")
    return f"Back at {page.url}"


@tool(
    "browser_screenshot",
    "Screenshot the current browser page (use when the ARIA snapshot isn't enough).",
    _p({}),
    category="browser",
    slow=True,
    untrusted=True,
)
async def browser_screenshot(a: dict, c: ToolContext) -> ToolOutput:
    page = await _browser().page()
    png = await page.screenshot(type="png")
    return ToolOutput(f"Browser screenshot of {page.url}", images=[ImagePart(png, "image/png")])


@tool("browser_close", "Close NEO's browser window.", _p({}), category="browser", quiet=True)
async def browser_close(a: dict, c: ToolContext) -> str:
    await _browser().close()
    return "Browser closed."


async def shutdown() -> None:
    await _browser().close()


def _host(url: str) -> str:
    from urllib.parse import urlsplit

    return urlsplit(url).hostname or "this page"


async def _label(loc) -> str:
    """The element's own words: text, aria-label, value (a button's caption can live in any)."""
    try:
        return await loc.first.evaluate(
            "e => [e.innerText, e.getAttribute('aria-label'), e.value, e.title].filter(Boolean).join(' ').slice(0, 200)",
            timeout=1500,
        )
    except Exception:  # noqa: BLE001 — the click itself reports a missing element
        return ""


async def _surroundings(loc) -> str:
    """The text of the dialog or form a button sits in ("Delete this account? … OK")."""
    try:
        return await loc.first.evaluate(
            "e => { const c = e.closest('dialog,[role=dialog],[role=alertdialog],form');"
            " return c ? c.innerText.slice(0, 1500) : ''; }",
            timeout=1500,
        )
    except Exception:  # noqa: BLE001
        return ""


async def _is_search(loc) -> bool:
    """A search box: submitting it only searches. Judged by what the field says it is — not by
    the form's method (every form without method="post" reports "get")."""
    try:
        return bool(
            await loc.first.evaluate(
                """e => {
                  if (e.type === 'password') return false;
                  if (e.type === 'search' || ['search', 'searchbox'].includes(e.getAttribute('role'))) return true;
                  if (e.form && e.form.getAttribute('role') === 'search') return true;
                  const words = [e.name, e.id, e.placeholder, e.getAttribute('aria-label')].join(' ').toLowerCase();
                  if (/\\bsearch\\b|\\bsearch[_-]|^q$|^query$/.test(words) || e.name === 'q' || e.name === 'query') return true;
                  const action = e.form ? (e.form.getAttribute('action') || '') : '';
                  return /\\/search\\b|[?&]q=/.test(action);
                }""",
                timeout=1500,
            )
        )
    except Exception:  # noqa: BLE001
        return False
