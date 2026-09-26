"""neo.decide.kinds.launch: open_app by evidence. Apps are fixtures, `open` and bringing an app to
the front are fakes that only record, the browser tab's address comes from a fake Accessibility
layer and the site search is a fake: nothing is launched, no URL is opened, no network is used."""

from __future__ import annotations

import asyncio

import pytest

from neo.decide import axkit, engine, inventory, learned, sites, workspace
from neo.decide.inventory import App
from neo.decide.kinds import launch
from neo.decide.kinds import site as web
from neo.tools.computer import apps as apps_mod

NOTEBOOK = App(bundle_id="test.notebook", name="Notebook", path="/Applications/Notebook.app", running=True)
CHAT = App(bundle_id="test.chatio", name="Chat.io", path="/Applications/Chat.io.app")
POSTBOX = App(
    bundle_id="test.postbox",
    name="Postbox",
    path="/Applications/Postbox.app",
    handles=("mailto",),
    days_used_30=6,
)
OLDMAIL = App(bundle_id="test.oldmail", name="Oldmail", path="/Applications/Oldmail.app", schemes=("mailto",))
WEBBY = App(bundle_id="test.webby", name="Webby", path="/Applications/Webby.app", handles=("https", "http"))
APPS = [NOTEBOOK, CHAT, POSTBOX, OLDMAIL, WEBBY]


class Mac:
    """What NEO did to this fake Mac: commands run, pages opened, apps brought forward."""

    def __init__(self, monkeypatch, *, trusted=True, lands=lambda url: url, front=True):
        self.runs, self.opened, self.fronted, self.targets, self.tree = [], [], [], [], []
        self.url, self.lands, self.front = "", lands, front
        monkeypatch.setattr(inventory, "inventory", lambda refresh=False: list(APPS))
        monkeypatch.setattr(inventory, "snapshot", lambda: list(APPS))
        monkeypatch.setattr(
            inventory, "app_by_id", lambda bid: next((a for a in APPS if a.bundle_id == bid), None)
        )
        monkeypatch.setattr(
            inventory, "_common_words", lambda ws: {w: w in {"notebook", "browser", "email"} for w in ws}
        )
        monkeypatch.setattr(
            workspace, "default_handler", lambda scheme: {"https": WEBBY.bundle_id}.get(scheme)
        )
        monkeypatch.setattr(sites, "_country", lambda: "in")
        monkeypatch.setattr(sites, "_language", lambda: "en")
        monkeypatch.setattr(sites, "_country_name", lambda cc: "India")
        monkeypatch.setattr(apps_mod, "_run", self._run)
        monkeypatch.setattr(apps_mod, "open_url", self._open_url)
        monkeypatch.setattr(apps_mod, "bring_front", self._bring_front)
        monkeypatch.setattr(apps_mod, "set_target", self.targets.append)
        monkeypatch.setattr(apps_mod, "front_name", lambda: "Finder")
        monkeypatch.setattr(launch, "_running_name", lambda pid: f"{pid}-name")
        monkeypatch.setattr(axkit, "pid_of", lambda bid: 7)
        monkeypatch.setattr(axkit, "trusted", lambda: trusted)
        monkeypatch.setattr(axkit, "app_el", lambda pid: "app")
        monkeypatch.setattr(axkit, "runtime", lambda path: "chromium")
        monkeypatch.setattr(
            axkit, "enable_web_tree", lambda el, rt: self.tree.append("on") or [("switch", False)]
        )
        monkeypatch.setattr(axkit, "restore_web_tree", lambda el, changed: self.tree.append("off"))
        monkeypatch.setattr(axkit, "web_url", lambda pid: self.url)
        monkeypatch.setattr(web, "LAND_S", 0.05)
        monkeypatch.setattr(web, "STEP_S", 0.01)
        sites._searched.clear()

    async def _run(self, cmd, timeout=30):
        self.runs.append(cmd)
        return 0, "", ""

    async def _open_url(self, url, browser_id="", browser_name=""):
        self.opened.append((url, browser_id))
        self.url = self.lands(url)
        return f"Opened {url} in {browser_name or 'the browser'}"

    async def _bring_front(self, name, wait=1.5):
        self.fronted.append(name)
        return self.front, name


class FakeSearch:
    def __init__(self, hrefs):
        self.hrefs, self.calls = hrefs, []

    def __call__(self, query, region):
        self.calls.append(query)
        return [{"href": h} for h in self.hrefs]


def go(target):
    return asyncio.run(launch.open_target(target))


def test_a_named_app_opens_by_bundle_id_and_counts_once_it_is_in_front(monkeypatch):
    mac = Mac(monkeypatch)
    r = go("notebook")
    assert r.ok and r.text == "Opened 7-name"
    assert mac.runs == [["open", "-b", "test.notebook"]]  # never `open -a <name>`
    assert mac.fronted == ["7-name"] and mac.targets == ["7-name"]  # by the name it runs under
    assert engine.proven("open_app", "test.notebook")


def test_an_app_kept_behind_asks_the_user_and_blames_nobody(monkeypatch):
    Mac(monkeypatch, front=False)
    r = go("Notebook")
    assert r.text.startswith("NEEDS_USER") and "click 7-name" in r.text
    assert not engine.proven("open_app", "test.notebook")
    assert engine.prior("open_app", "test.notebook", "open") == pytest.approx(0.8)  # the seed, untouched


def test_a_domain_that_only_contains_an_apps_name_is_the_website(monkeypatch):
    Mac(monkeypatch)
    assert asyncio.run(launch.decide("notebook.com")).how == "site"
    ch = asyncio.run(launch.decide("chat.io"))  # the app's own name is that domain
    assert (ch.how, ch.app) == ("app", CHAT)


def test_a_kind_noun_opens_the_app_this_user_uses_for_it(monkeypatch):
    mac = Mac(monkeypatch)
    ch = asyncio.run(launch.decide("my email"))
    assert (ch.how, ch.kind, ch.app) == ("noun", "read_mail", POSTBOX)
    assert go("email").ok and mac.runs == [["open", "-b", "test.postbox"]]


def test_a_kind_noun_nobody_uses_an_app_for_is_said(monkeypatch):
    mac = Mac(monkeypatch)
    monkeypatch.setattr(inventory, "inventory", lambda refresh=False: [NOTEBOOK, OLDMAIL, WEBBY])
    r = go("my email")
    assert not r.ok and "an app you use for my email" in r.text and mac.runs == []


def test_the_browser_noun_is_whatever_opens_web_links(monkeypatch):
    mac = Mac(monkeypatch)
    ch = asyncio.run(launch.decide("the browser"))
    assert (ch.how, ch.kind, ch.app) == ("handler", "https", WEBBY)
    assert go("browser").ok and mac.runs == [["open", "-b", "test.webby"]]


def test_a_site_word_is_resolved_opened_confirmed_and_then_learned(monkeypatch):
    mac = Mac(monkeypatch)
    search = FakeSearch(["https://review.example/shopkart", "https://www.shopkart.in/"])
    monkeypatch.setattr(sites, "_search", search)
    assert sites.site_like("shopkart") is None  # unknown before
    r = go("shopkart")
    assert r.ok and r.say is True and r.text == "Opened www.shopkart.in in Webby"
    assert mac.opened == [("https://www.shopkart.in/", "test.webby")]
    assert mac.tree == ["on", "off"]  # the web tree was switched back
    assert learned.get("sites", "shopkart|in") == {"host": "www.shopkart.in", "prefix": "", "via": "open"}
    r = go("shopkart")  # learned: no search, said no more
    assert r.ok and r.say is None and len(search.calls) == 1


def test_a_word_opened_as_a_site_is_no_claim_on_searches(monkeypatch):
    Mac(monkeypatch)
    monkeypatch.setattr(sites, "_search", FakeSearch(["https://www.shopkart.in/"]))
    assert go("shopkart").ok
    # "search news on <word>" may be about it: only a search that landed there makes it the site's
    assert sites.site_like("shopkart") is None
    assert not web.claims_search_site({"site": "shopkart", "query": "news"})
    assert not web.claims_web_search({"query": "news on shopkart"})


def test_a_landing_nobody_could_see_teaches_nothing(monkeypatch):
    mac = Mac(monkeypatch, trusted=False)
    monkeypatch.setattr(sites, "_search", FakeSearch(["https://www.shopkart.in/"]))
    r = go("shopkart")
    assert r.ok and mac.opened and mac.tree == []
    assert learned.get("sites", "shopkart|in") is None


def test_a_site_that_lands_elsewhere_is_said_and_not_learned(monkeypatch):
    Mac(monkeypatch, lands=lambda url: "https://accounts.elsewhere.example/login")
    monkeypatch.setattr(sites, "_search", FakeSearch(["https://www.shopkart.in/"]))
    r = go("shopkart")
    assert r.ok and r.say is True and "another site" in r.text
    assert learned.get("sites", "shopkart|in") is None


def test_an_unknown_word_is_an_honest_error_and_never_a_guessed_address(monkeypatch):
    mac = Mac(monkeypatch)
    monkeypatch.setattr(sites, "_search", FakeSearch(["https://blog.example/zorblax-review"]))
    r = go("zorblax")
    assert not r.ok and r.text == "Error: I couldn't find an app or a website called zorblax."
    assert mac.runs == [] and mac.opened == []


def test_a_time_word_is_never_looked_up_as_a_site(monkeypatch):
    Mac(monkeypatch)
    search = FakeSearch(["https://www.tomorrow.example/"])
    monkeypatch.setattr(sites, "_search", search)
    assert asyncio.run(launch.decide("tomorrow")).how == "none" and search.calls == []


@pytest.mark.parametrize(
    "target, how, url",
    [
        ("https://docs.example.org/a?b=c", "url", "https://docs.example.org/a?b=c"),
        ("docs.example.org/guide", "url", "https://docs.example.org/guide"),
        ("mailto:someone@example.org", "url", "mailto:someone@example.org"),
        ("/Applications", "path", "/Applications"),
    ],
)
def test_urls_and_paths_open_as_given(monkeypatch, target, how, url):
    mac = Mac(monkeypatch)
    monkeypatch.setattr(launch, "SETTLE_S", 0)
    ch = asyncio.run(launch.decide(target))
    assert (ch.how, ch.url) == (how, url)
    assert go(target).ok
    assert mac.opened == [(url, "")] if url.startswith("https") else mac.runs == [["open", url]]


@pytest.mark.parametrize(
    "target, url",
    [
        # this Mac and the local network: http (a dev server or a router rarely serves https)
        ("192.168.1.1", "http://192.168.1.1"),
        ("192.168.1.1/admin", "http://192.168.1.1/admin"),
        ("127.0.0.1:8000", "http://127.0.0.1:8000"),
        ("localhost:3000/app", "http://localhost:3000/app"),
        ("localhost", "http://localhost"),
        ("nas:5000", "http://nas:5000"),
        ("printer.local:631/jobs", "http://printer.local:631/jobs"),
        ("::1", "http://[::1]"),
        ("fe80::1", "http://[fe80::1]"),
        ("100.64.0.7:8080", "http://100.64.0.7:8080"),  # a VPN's shared range
        # anything else: https, the port kept
        ("docs.example.org:8443", "https://docs.example.org:8443"),
        ("docs.example.org:8443/guide", "https://docs.example.org:8443/guide"),
        ("1.2.3.4", "https://1.2.3.4"),
    ],
)
def test_a_network_address_opens_as_given_and_is_never_looked_up(monkeypatch, target, url):
    mac = Mac(monkeypatch)
    search = FakeSearch(["https://www.shopkart.in/"])
    monkeypatch.setattr(sites, "_search", search)
    ch = asyncio.run(launch.decide(target))
    assert (ch.how, ch.url) == ("url", url)
    assert go(target).ok and mac.opened == [(url, "")] and search.calls == []


def test_words_with_a_colon_or_a_slash_are_no_site_word(monkeypatch):
    Mac(monkeypatch)
    search = FakeSearch(["https://www.shopkart.in/"])
    monkeypatch.setattr(sites, "_search", search)
    assert asyncio.run(launch.decide("shopkart/deals")).how == "none" and search.calls == []
    assert asyncio.run(launch.decide("12:30")).how == "none" and search.calls == []  # a time, no host


def test_a_home_path_is_expanded(monkeypatch):
    Mac(monkeypatch)
    ch = asyncio.run(launch.decide("~/Documents"))
    assert ch.how == "path" and not ch.url.startswith("~")


def test_shadow_opens_nothing_and_compares_with_legacy(monkeypatch):
    mac = Mac(monkeypatch)
    monkeypatch.setattr(sites, "_search", FakeSearch(["https://www.shopkart.in/"]))
    got = asyncio.run(
        launch.shadow("shopkart", "No app called 'shopkart'; opened https://www.shopkart.com in your browser")
    )
    assert got == {"how": "site", "key": "www.shopkart.in", "agrees": False}  # legacy guessed .com
    got = asyncio.run(
        launch.shadow("shopkart", "No app called 'shopkart'; opened https://m.shopkart.in in Webby")
    )
    assert got["agrees"] is True
    # a domain: legacy opens https://<words> and names only the words (and where it went)
    for said in ("Opened shopkart.in in Webby", "Opened shopkart.in in Webby (it may be behind Notebook)"):
        assert asyncio.run(launch.shadow("shopkart.in", said)) == {"how": "site", "key": "shopkart.in", "agrees": True}
    assert asyncio.run(launch.shadow("shopkart.in", "Error: couldn't open 'shopkart.in': no"))["agrees"] is False
    assert asyncio.run(launch.shadow("notebook", "Opened Notebook")) == {
        "how": "app",
        "key": "test.notebook",
        "agrees": True,
    }
    assert asyncio.run(launch.shadow("zorblax", "Error: nope"))["agrees"] is True
    assert mac.runs == [] and mac.opened == [] and mac.fronted == []
    assert learned.get("sites", "shopkart|in") is None


def test_an_erase_mid_request_learns_nothing(monkeypatch):
    from neo import privacy

    Mac(monkeypatch)
    monkeypatch.setattr(sites, "_search", FakeSearch(["https://www.shopkart.in/"]))
    real = web.landing

    async def erased_while_loading(*a, **k):
        monkeypatch.setattr(privacy, "_generation", privacy.generation() + 1)
        return await real(*a, **k)

    monkeypatch.setattr(web, "landing", erased_while_loading)
    assert go("shopkart").ok
    assert learned.get("sites", "shopkart|in") is None
    assert engine.prior("open_app", "www.shopkart.in", "web") == engine.prior("open_app", "never.seen", "web")


def test_the_no_names_guard_scans_these_drivers():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "guard", Path(__file__).with_name("test_decide_no_names.py")
    )
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    for mod in (launch, web, sites):
        path = Path(mod.__file__).resolve()
        assert path in guard.SCOPE and guard._hits(path) == []
