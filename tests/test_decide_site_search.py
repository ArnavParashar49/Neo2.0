"""neo.decide.kinds.site: search_site by evidence, page_read for any browser, and the web fast-path
checks. The browser is a fake that only records (pages "opened", words "typed", keys "pressed"),
site pages come from an httpx.MockTransport and the site search is a fake: nothing is launched,
no URL is opened, no key reaches a real app and no network is used. Sites are fixtures."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus

import httpx
import pytest

from neo.decide import axkit, engine, inventory, learned, sites, workspace
from neo.decide.inventory import App
from neo.decide.kinds import launch
from neo.decide.kinds import site as web
from neo.tools.computer import apps as apps_mod
from neo.tools.computer import focus, policy

WEBBY = App(bundle_id="test.webby", name="Webby", path="/Applications/Webby.app", handles=("https", "http"))
SURFER = App(
    bundle_id="test.surfer",
    name="Surfer",
    path="/Applications/Surfer.app",
    schemes=("http", "https"),
    viewer_types=("public.html",),
)
NOTEBOOK = App(bundle_id="test.notebook", name="Notebook", path="/Applications/Notebook.app")
VIDZONE = App(
    bundle_id="test.vidzone",
    name="Vidzone",
    path="/Applications/Vidzone.app",
    web_start_url="https://www.vidzone.tv/",
)
APPS = [WEBBY, SURFER, NOTEBOOK, VIDZONE]
HOST = "www.shopkart.in"


@dataclass
class Hit:
    kind: str
    el: object


class Browser:
    """A browser tab and what NEO did to it. ``lands`` maps an opened URL to the address the tab
    then shows; ``results`` maps (the field, its words) to the address after Return."""

    def __init__(
        self, monkeypatch, *, trusted=True, box=True, bar=True, sign_in=False, lands=None, results=None, front=True
    ):
        self.url, self.value, self.focus, self.front, self.lag = "", "", None, front, 0
        self.opened, self.typed, self.returns, self.pressed, self.tree, self.fronted = [], [], 0, [], [], []
        self.box, self.bar, self.sign_in = box, bar, sign_in
        self.lands = lands or (lambda url: url)
        self.results = results or (lambda field, words: f"https://{HOST}/find?q={quote_plus(words)}&sid=9")
        monkeypatch.setattr(inventory, "inventory", lambda refresh=False: list(APPS))
        monkeypatch.setattr(inventory, "snapshot", lambda: list(APPS))
        monkeypatch.setattr(
            inventory, "app_by_id", lambda bid: next((a for a in APPS if a.bundle_id == bid), None)
        )
        monkeypatch.setattr(
            inventory,
            "_common_words",
            lambda ws: {w: w in {"notebook", "recipes", "dinner", "cheap", "flights", "apple"} for w in ws},
        )
        monkeypatch.setattr(
            workspace, "default_handler", lambda scheme: WEBBY.bundle_id if scheme == "https" else None
        )
        monkeypatch.setattr(sites, "_country", lambda: "in")
        monkeypatch.setattr(sites, "_language", lambda: "en")
        monkeypatch.setattr(sites, "_country_name", lambda cc: "India")
        monkeypatch.setattr(policy, "private_host", lambda url: "")
        monkeypatch.setattr(apps_mod, "open_url", self._open_url)
        monkeypatch.setattr(apps_mod, "bring_front", self._bring_front)
        monkeypatch.setattr(apps_mod, "_run", self._run)
        monkeypatch.setattr(launch, "_running_name", lambda pid: "")
        monkeypatch.setattr(axkit, "trusted", lambda: trusted)
        monkeypatch.setattr(axkit, "pid_of", lambda bid: 42)
        monkeypatch.setattr(axkit, "app_el", lambda pid: "app")
        monkeypatch.setattr(axkit, "runtime", lambda path: "chromium")
        monkeypatch.setattr(
            axkit, "enable_web_tree", lambda el, rt: self.tree.append("on") or [("switch", False)]
        )
        monkeypatch.setattr(axkit, "restore_web_tree", lambda el, changed: self.tree.append("off"))
        monkeypatch.setattr(axkit, "web_url", lambda pid: self.url)
        monkeypatch.setattr(axkit, "web_area", lambda pid: "area" if self.url else None)
        monkeypatch.setattr(axkit, "web_text", lambda pid, limit=8000: "Deals of the day"[:limit])
        monkeypatch.setattr(
            axkit, "find_search", lambda root: [Hit("field", "box")] if self.box and root == "area" else []
        )
        monkeypatch.setattr(axkit, "set_search", self._set_search)
        monkeypatch.setattr(axkit, "post_return", self._post_return)
        monkeypatch.setattr(
            axkit,
            "menu_item",
            lambda pid, char, mods=0: (
                ("open-location", "Open Location…") if self.bar and (char, mods) == ("l", 0) else None
            ),
        )
        monkeypatch.setattr(axkit, "perform", self._perform)
        monkeypatch.setattr(axkit, "attr", self._attr)
        monkeypatch.setattr(axkit, "window", lambda el: "win")
        monkeypatch.setattr(axkit, "needs_sign_in", lambda win: self.sign_in)
        monkeypatch.setattr(web, "LAND_S", 0.05)
        monkeypatch.setattr(web, "READY_S", 0.05)
        monkeypatch.setattr(web, "STEP_S", 0.01)
        sites._searched.clear()

    async def _open_url(self, url, browser_id="", browser_name=""):
        self.opened.append((url, browser_id))
        self.url = self.lands(url)
        return f"Opened {url} in {browser_name}"

    async def _bring_front(self, name, wait=1.5):
        self.fronted.append(name)
        return self.front, name

    async def _run(self, cmd, timeout=30):
        raise AssertionError(f"nothing is launched here: {cmd}")

    def _set_search(self, el, text):
        self.typed.append((el, text))
        self.value, self.focus = text, el
        return True

    def _post_return(self, pid):
        self.returns += 1
        self.url = self.results(self.focus, self.value)

    def _perform(self, el, action):
        self.pressed.append((el, action))
        if el == "open-location":
            self.moved = "address"  # the focus gets there after ``lag`` more reads
        return True

    # the page's own text field sits in the web content; the address and find fields are the browser's
    _TREE = {  # noqa: RUF012
        "address": {"AXRole": "AXTextField"},
        "findbar": {"AXRole": "AXTextField"},
        "pagefield": {"AXRole": "AXTextField", "AXParent": "form"},
        "form": {"AXRole": "AXGroup", "AXParent": "webarea"},
        "webarea": {"AXRole": "AXWebArea", "AXParent": "window"},
    }

    def _attr(self, el, name, default=None):
        if el == "app" and name == "AXFocusedUIElement":
            if getattr(self, "moved", None):
                if self.lag <= 0:
                    self.focus, self.moved = self.moved, None
                self.lag -= 1
            return self.focus
        return self._TREE.get(el, {}).get(name, default)


class FakeSearch:
    def __init__(self, hrefs):
        self.hrefs, self.calls = hrefs, []

    def __call__(self, query, region):
        self.calls.append(query)
        return [{"href": h} for h in self.hrefs]


def serve(monkeypatch, pages: dict[str, str]):
    def handler(request: httpx.Request) -> httpx.Response:
        body = pages.get(str(request.url))
        return (
            httpx.Response(404)
            if body is None
            else httpx.Response(200, text=body, headers={"content-type": "text/html"})
        )

    monkeypatch.setattr(sites, "_transport", httpx.MockTransport(handler))


FORM_HOME = '<html><body><form role="search" action="/s"><input name="kw"></form></body></html>'


def search(site, query, **kw):
    return asyncio.run(web.search(site, query, **kw))


# ---- search_site ----------------------------------------------------------------------------------


def test_first_use_resolves_discovers_lands_and_learns(monkeypatch):
    b = Browser(monkeypatch)
    monkeypatch.setattr(sites, "_search", FakeSearch([f"https://{HOST}/"]))
    serve(monkeypatch, {f"https://{HOST}/": FORM_HOME})
    r = search("Shopkart", "2tb ssd")
    assert r.ok and r.say is True and r.text == f"Searching {HOST} for “2tb ssd”."
    assert b.opened == [(f"https://{HOST}/s?kw=2tb+ssd", "test.webby")] and b.typed == []
    assert learned.get("sites", "shopkart|in") == {"host": HOST, "prefix": "", "via": "search"}
    assert learned.get("templates", HOST)["url"] == f"https://{HOST}/s?kw={{searchTerms}}"
    assert b.tree == ["on", "off"]
    assert engine.proven("web_search", HOST)
    # the next time: the learned site and template, said no more
    r = search("shopkart", "usb hub")
    assert r.ok and r.say is None and b.opened[-1][0] == f"https://{HOST}/s?kw=usb+hub"


def test_the_query_is_never_stored(monkeypatch, tmp_path):
    Browser(monkeypatch)
    monkeypatch.setattr(sites, "_search", FakeSearch([f"https://{HOST}/"]))
    serve(monkeypatch, {f"https://{HOST}/": FORM_HOME})
    assert search("shopkart", "zebra crossing lamp").ok
    stored = " ".join(p.read_text() for p in (tmp_path / "neo_data").rglob("*.json"))
    assert "zebra" not in stored and HOST in stored


def test_a_template_that_lands_without_the_search_falls_back_to_the_sites_own_box(monkeypatch):
    b = Browser(monkeypatch, lands=lambda url: f"https://{HOST}/")  # the site ignores the parameter now
    sites.learn_landing("shopkart", HOST, f"https://{HOST}/old?k={{searchTerms}}")
    serve(monkeypatch, {})
    r = search("shopkart", "2tb ssd")
    assert r.ok and len(b.opened) == 1  # the same tab: no second page opened
    assert b.typed == [("box", "2tb ssd")] and b.returns == 1
    # the page it landed on, reduced to its terms slot (the session token dropped)
    assert learned.get("templates", HOST) == {"url": f"https://{HOST}/find?q={{searchTerms}}", "fails": 0}


def test_a_learned_template_that_keeps_failing_is_dropped(monkeypatch):
    Browser(monkeypatch, lands=lambda url: f"https://{HOST}/", box=False, bar=False)
    sites.learn_landing("shopkart", HOST, f"https://{HOST}/old?k={{searchTerms}}")
    serve(monkeypatch, {})
    assert not search("shopkart", "2tb ssd").ok
    assert learned.get("templates", HOST)["fails"] == 1
    search("shopkart", "2tb ssd")
    assert learned.get("templates", HOST) is None


def test_without_a_template_the_site_box_is_used_on_the_sites_page(monkeypatch):
    b = Browser(monkeypatch)
    monkeypatch.setattr(sites, "_search", FakeSearch([f"https://{HOST}/"]))
    serve(monkeypatch, {f"https://{HOST}/": "<html><body>no search form here</body></html>"})
    r = search("shopkart", "2tb ssd")
    assert r.ok and b.opened == [(f"https://{HOST}/", "test.webby")] and b.typed == [("box", "2tb ssd")]
    assert learned.get("templates", HOST)["url"] == f"https://{HOST}/find?q={{searchTerms}}"


def test_no_box_then_the_address_bar_with_the_users_own_engine(monkeypatch):
    engine_page = lambda field, words: f"https://engine.example/results?q={quote_plus(words)}"  # noqa: E731
    b = Browser(monkeypatch, box=False, results=engine_page)
    monkeypatch.setattr(sites, "_search", FakeSearch([f"https://{HOST}/"]))
    serve(monkeypatch, {})
    r = search("shopkart", "2tb ssd")
    assert r.ok and r.say is True and "through your search engine" in r.text
    assert b.pressed == [("open-location", "AXPress")]  # found by its shortcut
    assert b.typed == [("address", "2tb ssd site:shopkart.in")] and b.fronted
    assert learned.get("templates", HOST) is None  # the engine's page teaches nothing about the site
    assert learned.get("sites", "shopkart|in") is None


def test_a_site_nothing_can_name_goes_to_the_address_bar_with_its_word(monkeypatch):
    engine_page = lambda field, words: f"https://engine.example/results?q={quote_plus(words)}"  # noqa: E731
    b = Browser(monkeypatch, results=engine_page)
    monkeypatch.setattr(sites, "_search", FakeSearch(["https://blog.example/zorblax"]))
    r = search("zorblax", "blue shoes")
    assert r.ok and r.say is True and "couldn't tell which website zorblax is" in r.text
    assert b.opened == [] and b.typed == [("address", "blue shoes zorblax")]


def test_an_app_is_not_a_website_and_a_browser_searches_the_web(monkeypatch):
    b = Browser(monkeypatch, results=lambda field, words: f"https://engine.example/?q={quote_plus(words)}")
    r = search("notebook", "groceries")
    assert not r.ok and r.text == "Error: notebook is an app on this Mac, not a website" and b.opened == []
    r = search("surfer", "cat videos")  # "search cat videos on <browser>"
    assert r.ok and b.typed == [("address", "cat videos")] and b.fronted == ["Surfer"]


@pytest.mark.parametrize("site", ["my notes", "the vidzone app", "monday"])
def test_what_is_no_website_is_never_searched_or_typed(monkeypatch, site):
    b = Browser(monkeypatch)
    search_ = FakeSearch([f"https://{HOST}/"])
    monkeypatch.setattr(sites, "_search", search_)
    r = search(site, "milk")
    assert not r.ok and r.text == f"Error: {site} isn't a website"
    assert b.opened == [] and b.typed == [] and search_.calls == []


def test_a_web_apps_name_is_its_site(monkeypatch):
    b = Browser(monkeypatch)
    serve(
        monkeypatch,
        {"https://www.vidzone.tv/": '<form role="search" action="/results"><input name="v"></form>'},
    )
    assert search("vidzone", "cats").ok
    assert b.opened == [("https://www.vidzone.tv/results?v=cats", "test.webby")]


def test_a_browser_the_user_named_is_used(monkeypatch):
    b = Browser(monkeypatch)
    serve(monkeypatch, {f"https://{HOST}/": FORM_HOME})
    assert search("shopkart.in", "ssd", browser="surfer").ok
    assert b.opened == [("https://shopkart.in/", "test.surfer")] and b.typed == [("box", "ssd")]


def test_without_accessibility_the_page_opens_and_nothing_is_learned_or_blamed(monkeypatch):
    b = Browser(monkeypatch, trusted=False)
    monkeypatch.setattr(sites, "_search", FakeSearch([f"https://{HOST}/"]))
    serve(monkeypatch, {f"https://{HOST}/": FORM_HOME})
    r = search("shopkart", "2tb ssd")
    assert r.ok and "Accessibility is off" in r.text and len(b.opened) == 1 and b.typed == []
    assert learned.get("sites", "shopkart|in") is None and learned.get("templates", HOST) is None
    assert engine.prior("web_search", HOST, "site_form") == pytest.approx(0.625)  # the seed: no blame


def test_a_landing_on_another_site_stops_there(monkeypatch):
    b = Browser(monkeypatch, lands=lambda url: "https://accounts.elsewhere.example/challenge")
    monkeypatch.setattr(sites, "_search", FakeSearch([f"https://{HOST}/"]))
    serve(monkeypatch, {f"https://{HOST}/": FORM_HOME})
    r = search("shopkart", "2tb ssd")
    assert r.ok and r.say is True and "another site" in r.text
    assert len(b.opened) == 1 and b.typed == [] and learned.get("sites", "shopkart|in") is None


def test_a_sign_in_wall_instead_of_a_search_box(monkeypatch):
    b = Browser(monkeypatch, box=False, sign_in=True)
    monkeypatch.setattr(sites, "_search", FakeSearch([f"https://{HOST}/"]))
    serve(monkeypatch, {})
    r = search("shopkart", "orders")
    assert r.ok and "sign in" in r.text and b.pressed == []


def test_a_vague_request_is_refused(monkeypatch):
    Browser(monkeypatch)
    assert not search("shopkart", "  ").ok


def test_the_web_tree_is_restored_when_the_search_is_cancelled(monkeypatch):
    b = Browser(monkeypatch, lands=lambda url: f"https://{HOST}/")
    monkeypatch.setattr(web, "LAND_S", 5.0)
    sites.learn_landing("shopkart", HOST, f"https://{HOST}/s?k={{searchTerms}}")
    serve(monkeypatch, {})

    async def go():
        task = asyncio.create_task(web.search("shopkart", "2tb ssd"))
        while b.tree != ["on"]:
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        for _ in range(100):  # the worker notices at its next step
            if b.tree == ["on", "off"]:
                break
            await asyncio.sleep(0.01)

    asyncio.run(go())
    assert b.tree == ["on", "off"]


# ---- page_read ------------------------------------------------------------------------------------


def test_page_read_reads_the_front_browser_whichever_it_is(monkeypatch):
    b = Browser(monkeypatch)
    b.url = "https://docs.example.org/page"
    monkeypatch.setattr(focus, "front_app", lambda: ("Surfer", 42))
    monkeypatch.setattr(inventory, "app_for_pid", lambda pid: SURFER)
    r = asyncio.run(web.page_read())
    assert r.ok and r.text == "URL: https://docs.example.org/page\n\nDeals of the day"
    assert b.tree == ["on", "off"]
    # the front app shows no web pages: the default browser's front tab
    monkeypatch.setattr(inventory, "app_for_pid", lambda pid: NOTEBOOK)
    assert asyncio.run(web.shadow_page_read()) == {
        "key": "test.webby",
        "running": True,
        "trusted": True,
        "runtime": "chromium",
    }


def test_page_read_without_accessibility_or_a_page(monkeypatch):
    b = Browser(monkeypatch, trusted=False)
    assert asyncio.run(web.page_read()).text.startswith("Error: Accessibility access is off")
    b = Browser(monkeypatch)
    monkeypatch.setattr(focus, "front_app", lambda: ("Surfer", 42))
    monkeypatch.setattr(inventory, "app_for_pid", lambda pid: SURFER)
    monkeypatch.setattr(axkit, "web_text", lambda pid, limit=8000: "")
    monkeypatch.setattr(web, "READ_S", 0.02)
    r = asyncio.run(web.page_read())
    assert not r.ok and r.text == "Error: Surfer shows no web page" and b.tree == ["on", "off"]


# ---- the tool's gate and the fast-path checks -----------------------------------------------------------


def test_the_taint_gate_sees_a_search_that_carries_data():
    assert not web.carries_data("shopkart", "2tb ssd")
    assert web.carries_data("shopkart", "the api key is " + "x" * 60)
    assert web.carries_data("a1b2c3d4e5f6g7h8.evil.org", "hi")


@pytest.fixture
def evidence(monkeypatch, tmp_path):
    words = tmp_path / "words"
    words.write_text("notebook\nrecipe\ndinner\ncheap\nflight\napple\ndeal\nShopkart\n")
    monkeypatch.setattr(inventory, "_WORDS_FILE", str(words))
    monkeypatch.setattr(sites, "_lower_known", {})
    monkeypatch.setattr(inventory, "snapshot", lambda: list(APPS))
    monkeypatch.setattr(
        inventory,
        "_common_words",
        lambda ws: {w: w in {"notebook", "recipes", "dinner", "cheap", "flights", "apple"} for w in ws},
    )
    monkeypatch.setattr(sites, "_country", lambda: "in")
    sites.learn_landing("shopkart", HOST, None)


@pytest.mark.parametrize(
    "site, claims",
    [
        ("shopkart", True),
        ("shopkart.in", True),
        ("vidzone", True),
        ("youtube", False),
        ("notebook", False),
        ("monday", False),
        ("apple", False),
    ],
)
def test_search_site_claims_only_sites_it_knows(evidence, site, claims):
    assert web.claims_search_site({"site": site, "query": "x"}) is claims


@pytest.mark.parametrize(
    "query, claims",
    [
        ("restaurants in delhi", True),  # a place in the world
        ("things to do in new delhi", True),
        ("milk in notebook", False),  # an app on this Mac
        ("milk in my notes", False),
        ("2tb ssd in shopkart.in", False),  # a site: search_site's
        ("the weather on monday", True),  # a time
        ("flights at 5pm", True),
        ("news on apple", False),  # about apple, or on a site? a model decides
        ("articles on climate change", False),
        ("2tb ssd on shopkart", False),
        ("articles on the history of the roman empire", True),  # a topic, not a place
        ("recipes for dinner", False),  # one word before "for" may be a site's name
        ("cheap flights for tomorrow", True),  # a phrase of everyday words: a topic
        ("shopkart deals for 2tb ssd", False),
        ("tomorrow for flights", True),
        ("youtube for cats", False),  # "<site> for X": the site's search, a model decides
        ("shopkart for 2tb ssd", False),
        ("notebook for milk", False),
        ("how to tie a tie", True),
    ],
)
def test_web_search_claims_on_evidence(evidence, query, claims):
    assert web.claims_web_search({"query": query}) is claims


# ---- shadow ------------------------------------------------------------------------------------------


def test_shadow_opens_nothing_and_compares_hosts(monkeypatch):
    b = Browser(monkeypatch)
    monkeypatch.setattr(sites, "_search", FakeSearch([f"https://{HOST}/"]))
    serve(monkeypatch, {f"https://{HOST}/": FORM_HOME})
    got = asyncio.run(web.shadow("shopkart", f"https://{HOST}/s?k=2tb+ssd"))
    assert got == {"how": "search", "key": HOST, "method": "site_form", "agrees": True}
    got = asyncio.run(web.shadow("shopkart", "https://www.search.example/search?q=2tb+ssd+shopkart"))
    assert got["agrees"] is False
    assert asyncio.run(web.shadow("notebook"))["how"] == "app"
    assert asyncio.run(web.shadow("surfer"))["method"] == "address_bar"
    assert b.opened == [] and b.typed == [] and b.tree == []
    assert learned.get("sites", "shopkart|in") is None
    assert "2tb" not in json.dumps(got)


def test_a_page_the_browser_wouldnt_open_blames_no_template(monkeypatch):
    b = Browser(monkeypatch)
    sites.learn_landing("shopkart", HOST, f"https://{HOST}/s?k={{searchTerms}}")
    serve(monkeypatch, {})

    async def refused(url, browser_id="", browser_name=""):
        b.opened.append((url, browser_id))
        return "Error: couldn't open it: LaunchServices said no"

    monkeypatch.setattr(apps_mod, "open_url", refused)
    r = search("shopkart", "2tb ssd")
    assert not r.ok and "LaunchServices said no" in r.text and len(b.opened) == 1
    assert learned.get("templates", HOST)["fails"] == 0


def test_the_address_bar_starts_a_browser_that_isnt_running(monkeypatch):
    b = Browser(monkeypatch, results=lambda field, words: f"https://engine.example/?q={quote_plus(words)}")
    started = []
    running = {"test.surfer": False}

    async def run_(cmd, timeout=30):
        started.append(cmd)
        running["test.surfer"] = True
        return 0, "", ""

    monkeypatch.setattr(apps_mod, "_run", run_)
    monkeypatch.setattr(axkit, "pid_of", lambda bid: 42 if running.get(bid, True) else None)
    assert search("surfer", "cat videos").ok
    assert started == [["open", "-b", "test.surfer"]] and b.typed == [("address", "cat videos")]


def test_the_engine_gets_these_drivers_methods(monkeypatch):
    monkeypatch.setitem(engine.DRIVERS, "web_search", "neo.decide.kinds.site")
    monkeypatch.setitem(engine.DRIVERS, "open_app", "neo.decide.kinds.launch")
    cands = asyncio.run(engine.choose("web_search", "search cats on surfer", APPS))
    assert [(c.key, c.methods) for c in cands] == [("test.surfer", ("address_bar",)), ("web", ("web",))]
    cands = asyncio.run(engine.choose("web_search", "search cats in notebook", APPS))
    assert [c.key for c in cands] == ["web"]  # an app that isn't a browser is no website
    cands = asyncio.run(engine.choose("open_app", "open notebook", APPS))
    assert [(c.key, c.methods) for c in cands][0] == ("test.notebook", ("open",))


# ---- review fixes ---------------------------------------------------------------------------------------

ZENTRIX = App(bundle_id="test.zentrix", name="Zentrix", path="/Applications/Zentrix.app")
SECTION = "https://www.shopkart.com/maps/search?q={searchTerms}"


def test_a_sites_sections_keep_their_own_templates(monkeypatch):
    b = Browser(monkeypatch)
    monkeypatch.setattr(
        sites, "_search", FakeSearch(["https://www.shopkart.com/maps/place/x", "https://www.shopkart.com/"])
    )
    serve(
        monkeypatch,
        {
            "https://www.shopkart.com/maps/": '<form role="search" action="search"><input name="q"></form>',
            "https://www.shopkart.com/": '<form role="search" action="/s"><input name="k"></form>',
        },
    )
    r = search("shopkart maps", "pizza")
    assert r.ok and r.text == "Searching www.shopkart.com/maps for “pizza”."
    assert b.opened[-1][0] == "https://www.shopkart.com/maps/search?q=pizza"
    assert engine.proven("web_search", "www.shopkart.com/maps")
    r = search("shopkart", "weather")  # the whole site: its own search, not the section's
    assert r.ok and b.opened[-1][0] == "https://www.shopkart.com/s?k=weather"
    assert learned.get("templates", "www.shopkart.com")["url"] == "https://www.shopkart.com/s?k={searchTerms}"
    assert learned.get("templates", "www.shopkart.com/maps")["url"] == SECTION


def test_a_sections_search_that_lands_outside_it_is_no_landing(monkeypatch):
    b = Browser(monkeypatch, lands=lambda url: "https://www.shopkart.com/search?q=pizza", box=False, bar=False)
    sites.learn_landing("shopkart maps", "www.shopkart.com", SECTION, "/maps")
    serve(monkeypatch, {})
    r = search("shopkart maps", "pizza")
    assert not r.ok and b.typed == []
    # the tab left the section: the section's own page is opened for its search box
    assert [u for u, _ in b.opened] == ["https://www.shopkart.com/maps/search?q=pizza", "https://www.shopkart.com/maps"]
    assert learned.get("templates", "www.shopkart.com/maps")["fails"] == 1


def test_a_learned_template_opens_without_reading_the_site_first(monkeypatch):
    b = Browser(monkeypatch)
    sites.learn_landing("shopkart", HOST, f"https://{HOST}/s?k={{searchTerms}}")
    fetched = []

    def handler(request):
        fetched.append(str(request.url))
        return httpx.Response(404)

    monkeypatch.setattr(sites, "_transport", httpx.MockTransport(handler))
    assert search("shopkart", "2tb ssd").ok and b.opened[-1][0] == f"https://{HOST}/s?k=2tb+ssd"
    assert asyncio.run(web.shadow("shopkart"))["method"] == "learned_template"
    assert fetched == []


def test_a_domain_that_only_contains_an_apps_name_is_a_website(monkeypatch):
    b = Browser(monkeypatch)
    monkeypatch.setattr(inventory, "inventory", lambda refresh=False: [*APPS, ZENTRIX])
    monkeypatch.setattr(inventory, "snapshot", lambda: [*APPS, ZENTRIX])
    serve(monkeypatch, {"https://zentrix.so/": FORM_HOME.replace("/s", "https://zentrix.so/s")})
    assert web.claims_search_site({"site": "zentrix.so"})
    r = search("zentrix.so", "roadmap")
    assert r.ok and b.opened == [("https://zentrix.so/s?kw=roadmap", "test.webby")]
    assert asyncio.run(web.shadow("zentrix.so"))["how"] == "domain"
    assert search("zentrix", "roadmap").text == "Error: zentrix is an app on this Mac, not a website"


def test_the_address_bar_types_nothing_while_the_browser_is_behind(monkeypatch):
    b = Browser(monkeypatch, front=False)
    r = search("surfer", "cat videos")
    assert r.ok and r.text.startswith("NEEDS_USER") and "click Surfer" in r.text
    assert b.pressed == [] and b.typed == [] and b.returns == 0


@pytest.mark.parametrize("focused", ["pagefield", "findbar"])
def test_the_address_bar_is_the_field_its_command_focuses(monkeypatch, focused):
    engine_page = lambda field, words: f"https://engine.example/?q={quote_plus(words)}"  # noqa: E731
    b = Browser(monkeypatch, results=engine_page)
    b.focus, b.lag = focused, 1  # the command's focus change lands a read later
    assert search("surfer", "cat videos").ok
    assert b.typed == [("address", "cat videos")] and b.returns == 1


def test_web_content_is_never_taken_for_the_address_bar(monkeypatch):
    b = Browser(monkeypatch, bar=True)
    monkeypatch.setattr(web, "BAR_S", 0.3)
    b.focus, b.lag = "pagefield", 10**6  # the command never moves the focus
    r = search("surfer", "cat videos")
    assert not r.ok and "address bar didn't open" in r.text
    assert b.typed == [] and b.returns == 0


@pytest.mark.skipif(not Path(inventory._WORDS_FILE).exists(), reason="needs the system word list")
@pytest.mark.parametrize(
    "query, claims",
    [
        ("amazon for 2tb ssd", False),  # a name the word list knows only capitalised
        ("amazon prime for movies", False),
        ("twitter for elon musk", False),  # one word before "for": maybe a site, whatever the list says
        ("target for tv stands", False),
        ("cheap flights for tomorrow", True),
        ("best shoes for running", True),
        ("restaurants in delhi", True),
    ],
)
def test_web_search_claims_with_the_real_word_list(monkeypatch, query, claims):
    monkeypatch.setattr(inventory, "snapshot", lambda: list(APPS))
    monkeypatch.setattr(sites, "_country", lambda: "in")
    monkeypatch.setattr(sites, "_lower_known", {})
    assert web.claims_web_search({"query": query}) is claims
