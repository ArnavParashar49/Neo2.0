"""NEO's own browser (neo/tools/browser.py) and the `browse` job runner (neo/tools/browse.py), on fake
Playwright pages: no Chromium starts, nothing reaches the network."""

from __future__ import annotations

import asyncio

import pytest

import neo.agent.session as sess_mod
from neo.agent import confirm
from neo.agent.registry import ToolContext, registry
from neo.providers.base import ToolCall, Turn
from neo.reflex.schema import Decision
from neo.tools import browser

SHOP = """- generic [active] [ref=e1]:
  - navigation [ref=e2]:
    - link "Home" [ref=e3] [cursor=pointer]:
      - /url: /
  - search [ref=e4]:
    - searchbox "Search the shop" [ref=e5]: mugs
    - button "Search" [ref=e6]
  - heading "Blue mug" [level=1] [ref=e7]
  - combobox "Size" [ref=e8]:
    - option "S"
    - option "M" [selected]
  - checkbox "Gift wrap" [checked] [ref=e9]
  - button "Place order" [ref=e10]
  - button "Add to cart" [ref=e11]
  - button "Add to cart" [ref=e12]
  - link "Reviews" [ref=e13] [cursor=pointer]:
    - /url: https://shop.example/mug/reviews
  - link [ref=e14] [cursor=pointer]:
    - /url: https://shop.example/help
  - generic [ref=e15] [cursor=pointer]: Show more
  - dialog "Sign in" [ref=e16]:
    - textbox "Password" [ref=e17]: hunter2
  - iframe [ref=e18]:
    - button "Pay" [ref=f1e2]
"""
TEXT = "Blue mug\nPrice: $12.99\n\nFree delivery on orders over $20.\nReviews (41)"


class Resp:
    def __init__(self, status: int) -> None:
        self.status = status


class Loc:
    def __init__(self, page: Page, key: str) -> None:
        self.page, self.key = page, key

    @property
    def first(self) -> Loc:
        return self

    def nth(self, i: int) -> Loc:
        return Loc(self.page, f"{self.key}#{i}")

    def or_(self, other: Loc) -> Loc:
        return self

    async def count(self) -> int:
        return 0 if self.key in self.page.gone else 1

    async def evaluate(self, script, arg=None, timeout=None):
        if script is browser._JS_INFO:
            return self.page.info.get(self.key, {"tag": "button", "label": ""})
        if script is browser._JS_IS_SEARCH:
            return self.key in self.page.search_fields
        raise AssertionError(f"unexpected script: {script[:40]}")

    async def click(self, timeout=None) -> None:
        self.page.log.append(("click", self.key))
        if (act := self.page.on_click.get(self.key)) and asyncio.iscoroutine(r := act()):
            await r  # the page's reaction happens while the click is in flight (a pop-up)

    async def fill(self, text: str, timeout=None) -> None:
        self.page.log.append(("fill", self.key, text))

    async def press(self, key: str) -> None:
        self.page.log.append(("press", self.key, key))
        if act := self.page.on_click.get(f"press:{self.key}"):
            act()

    async def select_option(self, *a, **kw) -> None:
        self.page.log.append(("select", self.key, kw.get("label") or a[0]))


class Mouse:
    def __init__(self, page: Page) -> None:
        self.page = page

    async def click(self, x, y) -> None:
        self.page.log.append(("mouse", x, y))

    async def wheel(self, dx, dy) -> None:
        self.page.log.append(("wheel", dy))


class Keyboard:
    def __init__(self, page: Page) -> None:
        self.page = page

    async def press(self, key: str) -> None:
        self.page.log.append(("key", key))


class Page:
    def __init__(self, ctx: Ctx, url="about:blank", snapshot="", text="", **probe) -> None:
        self.ctx, self.url, self.snapshot, self.text = ctx, url, snapshot, text
        self.probe = probe
        self.main = ""  # the text of the page's <main>
        self.page_title = "Blue mug — Shop"
        self.log: list[tuple] = []
        self.info: dict[str, dict] = {}
        self.search_fields: set[str] = set()
        self.gone: set[str] = set()
        self.on_click: dict = {}
        self.consent: dict | None = None
        self.point: dict | None = None
        self.status = 200
        self.closed = False
        self.frames = [self]
        self.main_frame = self
        self.mouse, self.keyboard = Mouse(self), Keyboard(self)
        self.sites: dict[str, tuple[str, str]] = {}  # url → (snapshot, text) that goto shows

    def is_closed(self) -> bool:
        return self.closed

    async def title(self) -> str:
        return self.page_title

    async def aria_snapshot(self, mode=None, timeout=None) -> str:
        assert mode == "ai"
        return self.snapshot

    async def evaluate(self, script, arg=None):
        if script is browser._JS_PAGE:
            return {"body": self.text, "main": self.main, **self.probe}
        if script is browser._JS_CONSENT:
            return self.consent
        if script is browser._JS_POINT:
            self.log.append(("point", tuple(arg)))
            return self.point
        raise AssertionError(f"unexpected script: {script[:40]}")

    def locator(self, sel: str) -> Loc:
        return Loc(self, sel)

    def get_by_role(self, role, name=None, exact=False) -> Loc:
        return Loc(self, f"role={role}:{getattr(name, 'pattern', name)}")

    def get_by_label(self, rx) -> Loc:
        return Loc(self, f"label:{rx.pattern}")

    get_by_placeholder = get_by_label

    def get_by_text(self, t, exact=False) -> Loc:
        return Loc(self, f"text:{t}")

    async def goto(self, url, wait_until=None, timeout=None) -> Resp:
        self.log.append(("goto", url))
        self.url = url
        if url in self.sites:
            self.snapshot, self.text = self.sites[url]
        return Resp(self.status)

    async def go_back(self, wait_until=None, timeout=None):
        return None

    async def wait_for_load_state(self, *a, **kw) -> None:
        pass

    async def screenshot(self, **kw) -> bytes:
        self.log.append(("shot", kw.get("type")))
        return b"jpeg-bytes"

    async def bring_to_front(self) -> None:
        self.log.append(("front",))

    async def close(self) -> None:
        self.closed = True


class Ctx:
    def __init__(self) -> None:
        self.pages: list[Page] = []
        self.closed = False

    async def close(self) -> None:
        self.closed = True

    async def new_page(self) -> Page:
        p = Page(self)
        self.pages.append(p)
        return p

    def add(self, page: Page) -> Page:
        self.pages.append(page)
        return page


@pytest.fixture
def world(monkeypatch):
    """A fake NEO's browser with one tab showing the shop page."""
    from neo import config
    from neo.tools import load_all

    monkeypatch.setattr(config, "_settings", None)
    load_all()
    confirm.cancel("test")
    b = browser._Browser()
    ctx = Ctx()
    b._ctx = ctx
    monkeypatch.setattr(browser, "_b", b)
    for name in ("_FOLLOW_S", "_CHALLENGE_WAIT_S", "_SETTLE_S"):
        monkeypatch.setattr(browser, name, 0)
    page = ctx.add(Page(ctx, "https://shop.example/mug", SHOP, TEXT))
    page.search_fields.add("aria-ref=e5")
    page.info["aria-ref=e10"] = {"tag": "button", "label": "Place order", "around": ""}
    page.info["aria-ref=e6"] = {"tag": "button", "label": "Search", "around": ""}
    page.info["aria-ref=e8"] = {"tag": "select", "label": "", "words": "size"}
    page.info["aria-ref=e13"] = {"tag": "a", "label": "Reviews", "href": "https://shop.example/mug/reviews"}
    yield b, ctx, page
    confirm.cancel("test")


def run(name: str, args: dict, ctx: ToolContext | None = None):
    return asyncio.run(registry().invoke(name, args, ctx or ToolContext()))


# ---- the page as numbered elements ---------------------------------------------------------------


def test_the_snapshot_numbers_what_can_be_clicked_or_typed():
    entries = browser._parse(SHOP)
    lines = [e.line() if isinstance(e, browser._El) else e for e in entries]
    assert lines[:4] == [
        '[1] link "Home"',
        '[2] searchbox "Search the shop" = "mugs"',
        '[3] button "Search"',
        "# Blue mug",
    ]
    assert '[4] combobox "Size" = "M"' in lines and '[5] checkbox "Gift wrap" [checked]' in lines
    assert "[10] link → https://shop.example/help" in lines  # an unnamed link shows where it goes
    assert '[11] generic "Show more"' in lines  # a clickable div with text
    assert "(dialog “Sign in”)" in lines
    assert '[12] textbox "Password"' in lines  # never its value
    els = {e.n: e for e in entries if isinstance(e, browser._El)}
    assert els[13].ref == "f1e2" and els[13].name == "Pay"  # inside a frame
    assert (els[7].ref, els[7].nth, els[8].nth) == ("e11", 0, 1)  # two "Add to cart", told apart


def test_the_plain_snapshot_without_refs_still_numbers_elements():
    plain = '- searchbox "Search"\n- button "Go"\n- option "x"\n- heading "Hi" [level=2]'
    els = [e for e in browser._parse(plain) if isinstance(e, browser._El)]
    assert [(e.n, e.role, e.ref) for e in els] == [(1, "searchbox", ""), (2, "button", "")]


def test_a_long_page_is_cut_with_a_count_and_find_narrows_it():
    snap = "\n".join(f'- link "Result number {i}" [ref=e{i}]' for i in range(1, 400))
    entries = browser._parse(snap)
    out = browser._render("https://x.test", "T", [], entries, "intro\nprice $5\nfooter", 3000)
    assert "more elements further down" in out and len(out) <= 3200
    found = browser._render(
        "https://x.test", "T", [], entries, "intro\nprice $5\nfooter", 3000, find="number 42"
    )
    assert '[42] link "Result number 42"' in found and 'Result number 7"' not in found
    found = browser._render(
        "https://x.test", "T", [], entries, "intro\nsome\nthing\nprice $5\nfooter", 3000, find="PRICE"
    )
    assert (
        "[TEXT]\nthing\nprice $5\nfooter" in found
        and "intro" not in found
        and "(no interactive elements)" in found
    )


def test_browser_read_is_untrusted_and_remembers_the_numbers(world):
    b, _, page = world
    out = run("browser_read", {})
    assert out.ok and not out.images
    assert out.text.startswith("URL: https://shop.example/mug\nTitle: Blue mug — Shop")
    assert '[6] button "Place order"' in out.text and "Price: $12.99" in out.text
    assert "[TEXT]" in out.text and b.els_page is page and len(b.els) == 13
    assert registry().get("browser_read").untrusted and registry().get("browser_click").untrusted


# ---- clicking --------------------------------------------------------------------------------------


def test_click_by_number_uses_the_elements_own_ref(world):
    _, _, page = world
    run("browser_read", {})
    out = run("browser_click", {"target": "3"})
    assert out.ok and ("click", "aria-ref=e6") in page.log
    assert "Clicked [3] button “Search”" in out.text and "stayed at" in out.text


def test_a_name_that_matches_two_elements_asks_for_a_number(world):
    _, _, page = world
    run("browser_read", {})
    out = run("browser_click", {"target": "Add to cart"})
    assert not out.ok and "[7] button “Add to cart”, [8] button “Add to cart”" in out.text
    assert not [x for x in page.log if x[0] == "click"]


def test_numbers_from_before_a_navigation_are_refused(world):
    _, _, page = world
    run("browser_read", {})
    page.url = "https://shop.example/elsewhere"
    out = run("browser_click", {"target": "3"})
    assert not out.ok and "read it again" in out.text and not [x for x in page.log if x[0] == "click"]


def test_a_click_that_orders_asks_first_with_the_element_and_the_page(world):
    _, _, page = world
    run("browser_read", {})
    out = run("browser_click", {"target": "6"})
    assert out.halt and out.text.startswith("NEEDS_CONFIRM") and not [x for x in page.log if x[0] == "click"]
    p = confirm.peek()
    assert p.tool == "browser_click" and p.summary == "click “Place order” on shop.example"
    assert p.detail == "https://shop.example/mug\n[6] button “Place order”"

    async def approve():
        with confirm.approved(p.action_id):
            return await registry().invoke("browser_click", {"target": "6"}, ToolContext(origin="approval"))

    done = asyncio.run(approve())
    assert done.ok and ("click", "aria-ref=e10") in page.log and confirm.peek() is None


def test_a_model_passing_confirm_is_not_an_approval(world):
    _, _, page = world
    run("browser_read", {})
    out = run("browser_click", {"target": "6", "confirm": True})
    assert out.halt and not [x for x in page.log if x[0] == "click"]


def test_a_click_that_opens_a_tab_is_followed_and_read(world):
    b, ctx, page = world
    run("browser_read", {})
    popup = Page(
        ctx,
        "https://shop.example/mug/reviews",
        '- heading "Reviews" [level=1]\n- link "Next" [ref=e2]',
        "Great mug",
    )
    page.on_click["aria-ref=e13"] = lambda: ctx.add(popup)
    out = run("browser_click", {"target": "9"})
    assert out.ok and b.current is popup and ("front",) in popup.log
    assert "opened a new tab" in out.text and "URL: https://shop.example/mug/reviews" in out.text
    assert '[1] link "Next"' in out.text and b.els_page is popup


def test_a_click_that_navigates_returns_the_new_page(world):
    _, _, page = world
    run("browser_read", {})

    def go():
        page.url, page.snapshot, page.text = (
            "https://shop.example/cart",
            '- button "Checkout" [ref=e2]',
            "Your cart",
        )

    page.on_click["aria-ref=e6"] = go
    out = run("browser_click", {"target": "3"})
    assert (
        "Clicked [3] button “Search”.\nURL: https://shop.example/cart" in out.text
        and "[1] button" in out.text
    )


def test_after_reading_a_page_a_link_to_the_local_network_asks_first(world, monkeypatch):
    from neo.tools.computer import policy

    _, _, page = world
    run("browser_read", {})
    # a long link the page wrote is just a link: it can't carry anything of the user's
    page.info["aria-ref=e13"]["href"] = "https://shop.example/reviews/" + "a-very-long-product-slug-" * 4
    monkeypatch.setattr(
        policy, "private_host", lambda u: "it's on this Mac or the local network" if "192.168" in u else ""
    )
    assert run("browser_click", {"target": "9"}, ToolContext(taint=["browser_read"])).ok
    page.info["aria-ref=e13"]["href"] = "http://192.168.1.1/admin"
    out = run("browser_click", {"target": "9"}, ToolContext(taint=["browser_read"]))
    assert out.halt and "local network" in confirm.peek().summary
    confirm.cancel("test")
    out = run("browser_click", {"target": "9"})  # nothing untrusted read in this request: just a link
    assert out.ok and page.log.count(("click", "aria-ref=e13")) == 2


def test_opening_a_link_a_page_showed_is_not_carrying_data(world, monkeypatch):
    from neo.tools.computer import policy

    _, _, page = world
    monkeypatch.setattr(policy, "private_host", lambda u: "")
    slug = "why-this-blue-mug-is-the-best-mug-for-most-people-in-2026"
    page.snapshot += f'  - link "Article" [ref=e30]:\n    - /url: /blog/{slug}\n'
    run("browser_read", {})
    tainted = ToolContext(taint=["browser_read"])
    assert run("browser_open", {"url": f"https://shop.example/blog/{slug}"}, tainted).ok
    out = run(
        "browser_open", {"url": f"https://shop.example/blog/{slug}-and-my-address-is-12-main-st"}, tainted
    )
    assert out.halt and "carries data" in confirm.peek().summary


def test_clicks_by_position_are_checked_like_any_other(world):
    _, _, page = world
    page.point = {"tag": "button", "label": "Pay now", "around": ""}
    out = run("browser_click", {"x": 400, "y": 300})
    assert out.halt and confirm.peek().summary == "click “Pay now” on shop.example"
    confirm.cancel("test")
    page.point = {"tag": "iframe", "label": ""}
    assert run("browser_click", {"x": 10, "y": 20}).halt  # can't see into a frame by position
    confirm.cancel("test")
    page.point = {"tag": "canvas", "label": ""}
    out = run("browser_click", {"x": 10, "y": 20})
    assert out.ok and ("mouse", 10.0, 20.0) in page.log


# ---- typing -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "info, words",
    [
        ({"tag": "input", "type": "password"}, "Please log in to shop.example"),
        ({"tag": "input", "autocomplete": "cc-number"}, "payment details"),
        ({"tag": "input", "autocomplete": "one-time-code"}, "verification code"),
        ({"tag": "input", "words": "Enter the 6-digit OTP"}, "verification code"),
        ({"tag": "input", "words": "UPI PIN"}, "log in"),
    ],
)
def test_never_types_passwords_codes_or_card_details(world, info, words):
    _, _, page = world
    run("browser_read", {})
    page.info["aria-ref=e5"] = info
    out = run("browser_type", {"target": "2", "text": "secret"})
    assert out.halt and out.text.startswith("NEEDS_USER") and words in out.text
    assert not [x for x in page.log if x[0] == "fill"] and ("front",) in page.log


def test_a_postal_pin_code_is_an_ordinary_field(world):
    _, _, page = world
    run("browser_read", {})
    page.info["aria-ref=e5"] = {"tag": "input", "words": "PIN code"}
    assert run("browser_type", {"target": "2", "text": "560001"}).ok
    assert ("fill", "aria-ref=e5", "560001") in page.log


def test_a_search_submits_without_asking_and_a_form_asks(world):
    _, _, page = world
    run("browser_read", {})

    def results():
        page.url, page.snapshot, page.text = (
            "https://shop.example/search?q=teapot",
            '- link "Teapot" [ref=e2]',
            "1 result",
        )

    page.on_click["press:aria-ref=e5"] = results
    out = run("browser_type", {"target": "2", "text": "teapot", "submit": True})
    assert (
        out.ok
        and ("fill", "aria-ref=e5", "teapot") in page.log
        and ("press", "aria-ref=e5", "Enter") in page.log
    )
    assert "submitted.\nURL: https://shop.example/search?q=teapot" in out.text

    page.info["aria-ref=e2"] = {"tag": "input", "words": "email"}
    page.snapshot, page.url = '- textbox "Email" [ref=e2]', "https://shop.example/newsletter"
    run("browser_read", {})
    out = run("browser_type", {"target": "1", "text": "me@example.com", "submit": True})
    assert out.halt and confirm.peek().summary == "submit the form on shop.example"
    assert confirm.peek().detail.endswith("[1] textbox “Email”: me@example.com")


def test_after_reading_a_page_long_text_into_it_asks_first(world):
    run("browser_read", {})
    out = run("browser_type", {"target": "2", "text": "a" * 70}, ToolContext(taint=["browser_read"]))
    assert out.halt and "after reading browser_read" in confirm.peek().summary
    confirm.cancel("test")
    assert run("browser_type", {"target": "2", "text": "mugs"}, ToolContext(taint=["browser_read"])).ok


def test_a_dropdown_gets_the_option_and_a_checkbox_is_clicked_not_typed(world):
    _, _, page = world
    run("browser_read", {})
    assert run("browser_type", {"target": "4", "text": "L"}).ok and ("select", "aria-ref=e8", "L") in page.log
    page.info["aria-ref=e9"] = {"tag": "input", "type": "checkbox"}
    out = run("browser_type", {"target": "5", "text": "yes"})
    assert not out.ok and "click it instead" in out.text


def test_keys_enter_in_a_form_asks_and_the_clipboard_never_goes_in(world):
    _, _, page = world
    assert "doesn't paste" in run("browser_press", {"key": "Meta+V"}).text
    assert "doesn't paste" in run("browser_press", {"key": "Control+v"}).text
    out = run("browser_press", {"key": "Enter"})
    assert out.halt and "submits the form" in confirm.peek().summary
    confirm.cancel("test")
    page.search_fields.add("*:focus")
    assert run("browser_press", {"key": "Enter"}).ok and ("key", "Enter") in page.log


# ---- what NEO does by itself: banners, graphics, checks ---------------------------------------------------


def test_a_cookie_banner_is_declined_and_one_without_a_decline_is_left(world):
    _, _, page = world
    page.consent = {"reject": "Reject all"}
    out = run("browser_read", {})
    assert (
        "click",
        "[data-neo-consent]",
    ) in page.log and "[NOTE] Declined a cookie banner (“Reject all”)." in out.text
    page.log.clear()
    page.consent = {"accept": "Accept all"}
    out = run("browser_read", {})
    assert not [x for x in page.log if x[0] == "click"] and "NEO never accepts one" in out.text


def test_accepting_cookies_by_hand_is_the_users_call(world):
    _, _, page = world
    page.snapshot = '- button "Accept all cookies" [ref=e2]'
    page.info["aria-ref=e2"] = {"tag": "button", "label": "Accept all cookies"}
    run("browser_read", {})
    assert run("browser_click", {"target": "1"}).halt


def test_a_page_of_pixels_comes_with_a_screenshot(world):
    _, _, page = world
    page.snapshot, page.text = "- generic [ref=e1]", ""
    out = run("browser_read", {})
    assert out.images and out.images[0].mime == "image/jpeg" and ("shot", "jpeg") in page.log
    assert "screenshot is attached" in out.text
    page.snapshot, page.text, page.probe = SHOP, TEXT, {"canvas": 0.6}
    assert run("browser_read", {}).images  # a big canvas, whatever the text says


def test_a_human_check_page_hands_the_window_to_the_user(world):
    _, _, page = world
    page.sites["https://search.example/"] = (
        "",
        "Our systems have detected unusual traffic from your computer network.",
    )
    out = run("browser_open", {"url": "search.example/"})
    assert out.halt and out.text.startswith("NEEDS_USER")
    assert "search.example wants a human check" in out.text and ("front",) in page.log


def test_sign_in_and_captcha_widgets_are_notes_not_stops(world):
    _, _, page = world
    page.probe = {"password": True, "widget": True}
    out = run("browser_read", {})
    assert out.ok and not out.halt
    assert "call browser_handoff kind=login" in out.text and "browser_handoff kind=captcha" in out.text


def test_an_article_about_captchas_is_just_an_article(world):
    _, _, page = world
    page.text = "The Turing test inspired CAPTCHA: a test to tell computers and humans apart. " * 40
    out = run("browser_read", {})
    assert out.ok and not out.halt and "CAPTCHA is on this page" not in out.text


def test_the_text_is_the_pages_main_content_and_find_searches_all_of_it(world):
    _, _, page = world
    page.text = "Menu\nSign in\n" + "Blue mug, hand-made. " * 40 + "\nContact: help@shop.example"
    page.main = "Blue mug, hand-made. " * 40
    out = run("browser_read", {})
    assert "[TEXT]\nBlue mug, hand-made." in out.text and "Contact" not in out.text
    assert "help@shop.example" in run("browser_read", {"find": "contact"}).text


def test_handoff_speaks_neos_own_words_about_the_site(world):
    _, _, page = world
    out = run("browser_handoff", {"kind": "login"})
    assert (
        out.halt
        and out.text
        == "NEEDS_USER: Please log in to shop.example in NEO's browser window, then say continue."
    )
    assert (
        "Please take over in NEO's browser window on shop.example"
        in run("browser_handoff", {"kind": "zzz"}).text
    )


# ---- opening, tabs -------------------------------------------------------------------------------------


def test_open_checks_the_scheme_and_after_reading_the_destination(world, monkeypatch):
    from neo.tools.computer import policy

    _, _, page = world
    assert "only opens web pages" in run("browser_open", {"url": "file:///etc/passwd"}).text
    assert run(
        "browser_open",
        {"url": "https://shop.example/?" + "q=" + "x" * 80},
        ToolContext(taint=["mail_unread"]),
    ).halt
    confirm.cancel("test")
    monkeypatch.setattr(policy, "private_host", lambda u: "it's on this Mac or the local network")
    out = run("browser_open", {"url": "http://192.168.1.1/"}, ToolContext(taint=["web_fetch"]))
    assert out.halt and "local network" in confirm.peek().summary
    confirm.cancel("test")
    assert run("browser_open", {"url": "http://192.168.1.1/"}).ok  # asked for by the user: fine
    page.status = 404
    assert "[NOTE] The site answered 404." in run("browser_open", {"url": "shop.example/nope"}).text


def test_tabs_list_switch_and_close(world):
    b, ctx, page = world
    run("browser_read", {})  # NEO works in the shop tab; a page the site opened by itself isn't followed
    second = ctx.add(Page(ctx, "https://shop.example/help", '- link "FAQ" [ref=e2]', "Help"))
    out = run("browser_tabs", {})
    assert "[1] Blue mug — Shop — https://shop.example/mug (current)" in out.text
    out = run("browser_tabs", {"action": "switch", "index": 2})
    assert b.current is second and '[1] link "FAQ"' in out.text
    assert "numbered 1 to 2" in run("browser_tabs", {"action": "switch", "index": 0}).text
    assert run("browser_tabs", {"action": "close", "index": 2}).ok and second.closed


def test_the_voice_shortcut_opens_a_site_in_neos_own_browser(world):
    from neo.agent import fastpath

    assert fastpath.match("open example.com in your browser") == ("browser_open", {"url": "example.com"})
    assert fastpath.match("go to docs.example.org/guide in NEO's own browser") == (
        "browser_open",
        {"url": "docs.example.org/guide"},
    )
    assert (fastpath.match("open example.com in my browser") or ("",))[0] != "browser_open"


def test_a_pages_own_pop_up_is_dismissed_and_reported(world):
    b, _, page = world

    class Dialog:
        def __init__(self, kind, message):
            self.type, self.message, self.did = kind, message, ""

        async def accept(self):
            self.did = "accept"

        async def dismiss(self):
            self.did = "dismiss"

    confirm_box, leave = Dialog("confirm", "Delete this   item?"), Dialog("beforeunload", "")
    run("browser_read", {})
    page.on_click["aria-ref=e6"] = lambda: b._dialog(confirm_box)

    async def click_and_leave():
        await b._dialog(leave)
        return await registry().invoke("browser_click", {"target": "3"}, ToolContext())

    out = asyncio.run(click_and_leave())
    assert confirm_box.did == "dismiss" and leave.did == "accept"
    assert "The page showed a pop-up (“Delete this item?”); NEO dismissed it." in out.text and b.dialogs == []


def test_a_browser_window_the_user_closed_is_started_again(world, monkeypatch):
    b, ctx, page = world
    fresh = Ctx()

    async def launch():
        b._ctx, b._gone = fresh, False

    monkeypatch.setattr(b, "_launch", launch)
    b._closed()  # Playwright's "close" event on the context
    out = run("browser_open", {"url": "shop.example"})
    assert out.ok and b._ctx is fresh and fresh.pages and b.current is fresh.pages[0]


def test_headless_only_when_asked(monkeypatch):
    monkeypatch.delenv("NEO_BROWSER_HEADLESS", raising=False)
    assert not browser._headless()
    monkeypatch.setenv("NEO_BROWSER_HEADLESS", "1")
    assert browser._headless()


def test_older_browser_views_are_trimmed_by_any_newer_one():
    from neo.agent.loop import _supersede_observations
    from neo.providers.base import Message, ToolResult

    msgs = [
        Message.user("go"),
        Message.tool([ToolResult("1", "browser_open", "A" * 3000)]),
        Message.tool([ToolResult("2", "browser_click", "B" * 3000)]),
        Message.tool([ToolResult("3", "web_search", "C" * 3000)]),
    ]
    _supersede_observations(msgs)
    assert "trimmed" in msgs[1].tool_results[0].content and msgs[2].tool_results[0].content == "B" * 3000
    assert msgs[3].tool_results[0].content == "C" * 3000


# ---- browse(goal) -------------------------------------------------------------------------------------


class Scripted:
    name = "fake"
    supports_vision = True
    supports_tools = True

    def __init__(self, turns: list[Turn]) -> None:
        self.turns = list(turns)
        self.calls: list[dict] = []

    async def complete(self, messages, *, system="", tools=None, effort="medium", max_tokens=4096):
        self.calls.append(
            {"messages": list(messages), "system": system, "tools": [t.name for t in tools or []]}
        )
        return self.turns.pop(0)

    async def stream(self, messages, **kw):
        yield ""


def _call(i: int, name: str, **args) -> Turn:
    return Turn(tool_calls=[ToolCall(f"c{i}", name, args)])


@pytest.fixture
def inner(monkeypatch):
    """Install the brain a browse run gets."""
    from neo import providers
    from neo.tools import browse

    browse._asked.clear()
    monkeypatch.setattr(browse, "_last", None)

    def use(turns: list[Turn]) -> Scripted:
        prov = Scripted(turns)
        monkeypatch.setattr(providers, "brain", lambda purpose="agent": prov)
        return prov

    return use


def test_browse_runs_a_focused_loop_on_the_browser_and_web_tools_only(world, inner):
    from neo.tools import browse

    _, _, page = world
    prov = inner([_call(1, "browser_open", url="shop.example/mug"), Turn(text="The blue mug is $12.99.")])
    ctx = ToolContext(origin="live")
    out = asyncio.run(
        registry().invoke("browse", {"goal": "find the price of the blue mug on shop.example"}, ctx)
    )
    assert out.ok and out.text == "The blue mug is $12.99."
    assert set(prov.calls[0]["tools"]) == set(browse.TOOLS)
    assert (
        "your own browser" in prov.calls[0]["system"]
        and f"about {browse.STEPS} steps" in prov.calls[0]["system"]
    )
    assert ("goto", "https://shop.example/mug") in page.log
    assert "browser_open" in ctx.taint  # what the run read gates what the caller does next
    t = registry().get("browse")
    assert t.slow and t.untrusted and t.category == "web" and t.timeout > browse.SECONDS


def test_a_browse_that_reaches_an_order_button_asks_with_that_very_card(world, inner):
    _, _, page = world
    prov = inner([_call(1, "browser_read"), _call(2, "browser_click", target="6")])
    out = asyncio.run(
        registry().invoke("browse", {"goal": "order the blue mug on shop.example"}, ToolContext())
    )
    p = confirm.peek()
    assert out.halt and out.staged == p.action_id and p.tool == "browser_click"
    assert out.text == "NEEDS_CONFIRM: click “Place order” on shop.example — Should I go ahead?"
    assert not [x for x in page.log if x[0] == "click"]
    # asked again for the same job while the card waits: the same question, no new run
    again = asyncio.run(
        registry().invoke("browse", {"goal": "order the blue mug on shop.example"}, ToolContext())
    )
    assert again.halt and again.staged == p.action_id and len(prov.calls) == 2
    from neo.tools import browse

    assert browse.asks(p) and not browse.asks(None)


def test_a_login_inside_browse_comes_back_as_the_users_job_and_continue_picks_it_up(world, inner):
    _, _, page = world
    inner([_call(1, "browser_read"), _call(2, "browser_handoff", kind="login")])
    out = asyncio.run(registry().invoke("browse", {"goal": "check my orders on shop.example"}, ToolContext()))
    assert (
        out.halt
        and out.text
        == "NEEDS_USER: Please log in to shop.example in NEO's browser window, then say continue."
    )
    prov = inner([Turn(text="Your mug ships Tuesday.")])
    out = asyncio.run(registry().invoke("browse", {"goal": "continue"}, ToolContext()))
    assert out.text == "Your mug ships Tuesday."
    first = prov.calls[0]["messages"][0].text
    assert "<earlier_run" in first and "check my orders on shop.example" in first and "Please log in" in first
    prov = inner([Turn(text="ok")])
    asyncio.run(registry().invoke("browse", {"goal": "find a teapot on another.example"}, ToolContext()))
    assert "<earlier_run" not in prov.calls[0]["messages"][0].text  # a different job starts fresh


def test_a_browse_out_of_steps_says_what_it_found(world, inner, monkeypatch):
    from neo.tools import browse

    monkeypatch.setattr(browse, "STEPS", 2)
    inner(
        [
            _call(1, "browser_scroll", amount=500),
            _call(2, "browser_scroll", amount=900),
            Turn(text="Two mugs so far."),
        ]
    )
    out = asyncio.run(
        registry().invoke("browse", {"goal": "compare all mugs on shop.example"}, ToolContext())
    )
    assert out.ok and out.text == "I couldn't finish that in NEO's browser in time. Two mugs so far."


def test_one_browse_at_a_time(world, inner):
    from neo.tools import browse

    async def two():
        async with browse._running:
            browse._now = "order a mug"
            return await registry().invoke("browse", {"goal": "find a teapot"}, ToolContext())

    out = asyncio.run(two())
    assert not out.ok and "busy with another job (order a mug)" in out.text
    assert "needs a goal" in run("browse", {"goal": "  "}).text


def test_the_users_yes_to_a_browse_card_clicks_and_the_agent_finishes(world, inner, monkeypatch):
    """Typed request → the agent calls browse → the run stops at "Place order" → the user's yes
    runs exactly that click → the agent reports."""
    from neo import events
    from neo.events import EventBus

    _, _, page = world
    monkeypatch.setattr(events, "_bus", EventBus())
    inner([_call(1, "browser_read"), _call(2, "browser_click", target="6")])
    outer = Scripted(
        [_call(9, "browse", goal="order the blue mug on shop.example"), Turn(text="Ordered the blue mug.")]
    )
    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": outer)

    class R:
        async def decide(self, t):
            return Decision("agent_task", 0.95, False, 0.0, False, 0.0, "laya+head", reply=False, reply_p=0.1)

    async def talk():
        s = sess_mod.Session(reflex=R())
        first = await s.handle("order the blue mug on shop.example", from_user=True)
        assert first.route == "confirm" and "Place order" in first.text
        assert not [x for x in page.log if x[0] == "click"]
        return await s.handle("yes", from_user=True)

    reply = asyncio.run(talk())
    assert ("click", "aria-ref=e10") in page.log and reply.text == "Ordered the blue mug."
