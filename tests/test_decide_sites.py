"""neo2.core.decide.sites: a site word resolved by evidence and its search found in what the site itself
publishes. The web search is a fake, pages come from an httpx.MockTransport, and the private-host
check is a fake too: no network is used. Site names here are fixtures."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from neo2.core.decide import inventory, learned, sites
from neo2.core.decide.inventory import App
from neo2.core.tools.computer import policy

NOTEBOOK = App(bundle_id="test.notebook", name="Notebook", path="/Applications/Notebook.app")
WRAPPER = App(bundle_id="test.vidzone", name="Vidzone", path="/Applications/Vidzone.app", web_start_url="https://www.vidzone.tv/")


@pytest.fixture(autouse=True)
def here(monkeypatch):
    monkeypatch.setattr(sites, "_country", lambda: "in")
    monkeypatch.setattr(sites, "_language", lambda: "en")
    monkeypatch.setattr(sites, "_country_name", lambda cc: {"in": "India", "gb": "United Kingdom"}.get(cc, ""))
    monkeypatch.setattr(inventory, "snapshot", lambda: [NOTEBOOK, WRAPPER])
    monkeypatch.setattr(inventory, "_words_known", {})
    monkeypatch.setattr(inventory, "_common_words", lambda words: {w: w in {"notebook", "climate", "change"} for w in words})
    monkeypatch.setattr(policy, "private_host", lambda url: "it's on the local network" if "10.0.0." in url else "")
    sites._searched.clear()
    yield
    sites._searched.clear()


@pytest.mark.parametrize(
    "word, want",
    [
        ("shopkart.in", True),
        ("en.wiki.example.org", True),
        ("vidzone", True),  # an installed web-app wrapper's site
        ("Notebook", False),  # an app on this Mac
        ("my notes", False),
        ("the vidzone app", False),
        ("monday", False),
        ("tomorrow", False),
        ("3 oct", False),
        ("", False),
        ("youtube", None),  # local evidence can't tell: the model decides
        ("apple", None),
        ("climate change", None),
    ],
)
def test_site_like(word, want):
    assert sites.site_like(word) is want


def test_a_learned_site_word_counts_for_its_country_only(monkeypatch):
    sites.learn_landing("shopkart", "www.shopkart.in", None)
    assert sites.site_like("Shopkart") is True
    monkeypatch.setattr(sites, "_country", lambda: "us")
    assert sites.site_like("shopkart") is None


class FakeSearch:
    def __init__(self, hrefs):
        self.hrefs, self.calls = hrefs, []

    def __call__(self, query, region):
        self.calls.append((query, region))
        return [{"href": h, "title": "t"} for h in self.hrefs]


def test_resolve_order_domain_learned_wrapper_then_search(monkeypatch):
    s = FakeSearch(["https://www.shopkart.com/", "https://www.shopkart.in/deals"])
    monkeypatch.setattr(sites, "_search", s)
    assert asyncio.run(sites.resolve("docs.example.org")) == sites.SiteRes("docs.example.org", "domain", "in")
    assert asyncio.run(sites.resolve("vidzone")) == sites.SiteRes("www.vidzone.tv", "webapp", "in")
    got = asyncio.run(sites.resolve("ShopKart"))
    assert got == sites.SiteRes("www.shopkart.in", "search", "in")  # the store for this country
    assert s.calls == [("ShopKart official website India", "in-en")]  # its own site, in this country
    asyncio.run(sites.resolve("shopkart"))
    assert len(s.calls) == 1  # cached
    sites.learn_landing("shopkart", "www.shopkart.in", None)
    assert asyncio.run(sites.resolve("shopkart")).how == "learned"


def test_the_search_regions_quirk_and_no_local_store(monkeypatch):
    s = FakeSearch(["https://blog.other.example/shopkart-review", "https://www.shopkart.com/"])
    monkeypatch.setattr(sites, "_search", s)
    monkeypatch.setattr(sites, "_country", lambda: "gb")
    assert asyncio.run(sites.resolve("shopkart")).host == "www.shopkart.com"
    assert s.calls[-1][1] == "uk-en"


def test_the_domain_most_results_agree_on_at_its_front_host(monkeypatch):
    s = FakeSearch([
        "https://en.wiki.example.org/wiki/Shopkart",
        "https://shopkart.blog/",  # the site's blog, one result
        "https://corporate.shopkart.net/about",
        "https://careers.shopkart.com/",
        "https://www.shopkart.com/login",
        "https://www.shopkart.com/",
    ])
    monkeypatch.setattr(sites, "_search", s)
    monkeypatch.setattr(sites, "_country", lambda: "")
    assert asyncio.run(sites.resolve("shopkart")) == sites.SiteRes("www.shopkart.com", "search", "")
    assert s.calls == [("shopkart official website", "wt-wt")]  # no region known: no country in the words
    sites._searched.clear()
    monkeypatch.setattr(sites, "_search", FakeSearch(["https://shopkart.blog/", "https://corporate.shopkart.net/"]))
    assert asyncio.run(sites.resolve("shopkart")).host == "shopkart.blog"  # equals: the first


def test_pages_are_fetched_in_the_users_languages(monkeypatch):
    got = []
    sites._accept_language.cache_clear()

    class Locale:
        @staticmethod
        def preferredLanguages():
            return ["en-IN", "hi-IN", "en-GB", "fr"]

    import Foundation

    monkeypatch.setattr(Foundation, "NSLocale", Locale)

    def handler(request):
        got.append(request.headers.get("accept-language"))
        return httpx.Response(404)

    monkeypatch.setattr(sites, "_transport", httpx.MockTransport(handler))
    asyncio.run(sites.discover("shop.example.in"))
    sites._accept_language.cache_clear()
    assert got == ["en-IN, hi-IN;q=0.9, en-GB;q=0.8"]


def test_multi_word_sites_by_subdomain_or_path(monkeypatch):
    monkeypatch.setattr(sites, "_search", FakeSearch(["https://maps.shopkart.com/"]))
    assert asyncio.run(sites.resolve("shopkart maps")) == sites.SiteRes("maps.shopkart.com", "search", "in")
    sites._searched.clear()
    monkeypatch.setattr(sites, "_search", FakeSearch(["https://www.shopkart.com/maps/place/x"]))
    assert asyncio.run(sites.resolve("shopkart maps")) == sites.SiteRes("www.shopkart.com", "search", "in", "/maps")


def test_no_match_or_no_network_is_not_a_site(monkeypatch):
    monkeypatch.setattr(sites, "_search", FakeSearch(["https://news.example/climate-change"]))
    assert asyncio.run(sites.resolve("climate change")) is None

    def offline(q, r):
        raise OSError("no route to host")

    sites._searched.clear()
    monkeypatch.setattr(sites, "_search", offline)
    assert asyncio.run(sites.resolve("shopkart")) is None


# ---- discovery ---------------------------------------------------------------------------------------

HOST = "shop.example.in"
OSD = """<?xml version="1.0"?>
<OpenSearchDescription xmlns="http://a9.com/-/spec/opensearch/1.1/">
  <Url type="application/x-suggestions+json" template="https://shop.example.in/suggest?q={searchTerms}"/>
  <Url type="text/html" template="https://shop.example.in/s?page={startPage?}&amp;k={searchTerms}&amp;lang={language?}"/>
</OpenSearchDescription>"""


def serve(monkeypatch, pages: dict[str, httpx.Response | str]):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        got = pages.get(str(request.url))
        if got is None:
            return httpx.Response(404)
        return got if isinstance(got, httpx.Response) else httpx.Response(200, text=got, headers={"content-type": "text/html"})

    monkeypatch.setattr(sites, "_transport", httpx.MockTransport(handler))
    return seen


def home(head: str = "", body: str = "") -> str:
    return f"<html><head>{head}</head><body>{body}</body></html>"


def discover(host=HOST):
    return asyncio.run(sites.discover(host))


def test_opensearch_relative_href_and_optional_params_dropped(monkeypatch):
    serve(monkeypatch, {
        f"https://{HOST}/": home('<link rel="search" type="application/opensearchdescription+xml" href="/osd.xml">'),
        f"https://{HOST}/osd.xml": OSD,
    })
    assert discover() == [sites.Template("https://shop.example.in/s?k={searchTerms}", "opensearch")]


def test_the_page_is_parsed_off_the_event_loop(monkeypatch):
    import threading

    parsed_on: list[bool] = []
    real = sites._read_page

    def read_page(page, host):
        parsed_on.append(threading.current_thread() is threading.main_thread())
        return real(page, host)

    monkeypatch.setattr(sites, "_read_page", read_page)
    serve(monkeypatch, {f"https://{HOST}/": home(body='<form action="/p"><input type="search" name="w"></form>')})
    assert discover() == [sites.Template("https://shop.example.in/p?w={searchTerms}", "site_form")]
    assert parsed_on == [False]  # the loop (Live's audio runs on it) never parses up to MAX_PAGE of HTML


def test_opensearch_off_domain_dtd_or_extra_params_are_refused(monkeypatch):
    serve(monkeypatch, {
        f"https://{HOST}/": home('<link rel="search" type="application/opensearchdescription+xml" href="https://evil.example/osd.xml">'),
        "https://evil.example/osd.xml": OSD,
    })
    assert discover() == []
    dtd = '<?xml version="1.0"?><!DOCTYPE x [<!ENTITY e SYSTEM "file:///etc/passwd">]>' + OSD.split("?>", 1)[1]
    need = OSD.replace("&amp;lang={language?}", "&amp;sid={sessionId}")
    for osd in (dtd, need):
        serve(monkeypatch, {
            f"https://{HOST}/": home('<link rel="search" type="application/opensearchdescription+xml" href="/osd.xml">'),
            f"https://{HOST}/osd.xml": osd,
        })
        assert discover() == []


@pytest.mark.parametrize(
    "target",
    ['"https://shop.example.in/find?text={search_term_string}"',
     '{"@type": "EntryPoint", "urlTemplate": "https://shop.example.in/find?text={search_term_string}"}'],
)
def test_json_ld_search_action(monkeypatch, target):
    action = f'{{"@type": "SearchAction", "target": {target}, "query-input": "required name=search_term_string"}}'
    graph = f'{{"@context": "https://schema.org", "@graph": [{{"@type": "WebSite", "potentialAction": {action}}}]}}'
    ld = f'<script type="application/ld+json">{graph}</script>'
    serve(monkeypatch, {f"https://{HOST}/": home(ld)})
    assert discover() == [sites.Template("https://shop.example.in/find?text={searchTerms}", "jsonld")]


@pytest.mark.parametrize(
    "form, want",
    [
        ('<div role="search"><form action="/find"><input type="hidden" name="cat" value="all"><input name="qq"></form></div>',
         "https://shop.example.in/find?cat=all&qq={searchTerms}"),
        ('<form action="https://shop.example.in/x"><input type="search" name="zz"></form>', "https://shop.example.in/x?zz={searchTerms}"),
        ('<form action="/p"><input type="text" name="w" placeholder="Search products"></form>', "https://shop.example.in/p?w={searchTerms}"),
        ('<search><form action="/g"><input name="t"></form></search>', "https://shop.example.in/g?t={searchTerms}"),
    ],
)
def test_search_forms_are_found_by_structure(monkeypatch, form, want):
    serve(monkeypatch, {f"https://{HOST}/": home(body=form)})
    assert discover() == [sites.Template(want, "site_form")]


@pytest.mark.parametrize(
    "form",
    [
        '<form action="/find"><input name="q"></form>',  # a conventional name alone means nothing: no name list
        '<form role="search" method="post" action="/find"><input name="q"></form>',
        '<form role="search" action="/login"><input name="user"><input name="pass"></form>',
        '<form role="search" action="https://evil.example/find"><input name="q"></form>',
    ],
)
def test_forms_that_are_not_a_search_here(monkeypatch, form):
    serve(monkeypatch, {f"https://{HOST}/": home(body=form)})
    assert discover() == []


def test_a_learned_template_is_read_without_the_network_and_discovery_gives_each_url_once(monkeypatch):
    seen = serve(monkeypatch, {})
    sites.learn_landing("shop", HOST, "https://shop.example.in/s?k={searchTerms}")
    assert sites.learned_template(HOST) == sites.Template("https://shop.example.in/s?k={searchTerms}", "learned")
    assert seen == []
    ld = ('<script type="application/ld+json">{"@type": "SearchAction", "target": '
          '"https://shop.example.in/find?q={s}", "query-input": "required name=s"}</script>')
    serve(monkeypatch, {f"https://{HOST}/": home(ld, '<form role="search" action="/find"><input name="q"></form>')})
    assert discover() == [sites.Template("https://shop.example.in/find?q={searchTerms}", "jsonld")]


def test_private_hosts_and_redirects_there_are_never_fetched(monkeypatch):
    seen = serve(monkeypatch, {f"https://{HOST}/": httpx.Response(302, headers={"location": "https://10.0.0.1/admin"})})
    assert discover() == [] and seen == [f"https://{HOST}/"]  # the private hop was refused before it went
    seen = serve(monkeypatch, {"https://10.0.0.5/": home()})
    assert discover("10.0.0.5") == [] and seen == []


@pytest.mark.parametrize(
    "page",
    ["<!-- nothing here -->", "<?php ?>",
     home('<script type="application/ld+json">' + "[" * 100_000 + "</script>",
          '<form role="search" action="/find"><input name="q"></form>')],
)
def test_a_page_the_parsers_choke_on_gives_what_was_found(monkeypatch, page):
    serve(monkeypatch, {f"https://{HOST}/": page})
    assert [t.how for t in discover()] == (["site_form"] if "<form" in page else [])


def test_oversized_insecure_and_failed_pages(monkeypatch):
    monkeypatch.setattr(sites, "MAX_PAGE", 100)
    serve(monkeypatch, {f"https://{HOST}/": home(body="x" * 500)})
    assert discover() == []
    monkeypatch.setattr(sites, "MAX_PAGE", 1_500_000)
    serve(monkeypatch, {f"https://{HOST}/": httpx.Response(302, headers={"location": f"http://{HOST}/plain"}),
                        f"http://{HOST}/plain": home(body='<form role="search"><input name="q"></form>')})
    assert discover() == []  # downgraded to http: not used
    serve(monkeypatch, {})
    assert discover() == []


# ---- templates from a landing -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url, query, want",
    [
        ("https://shop.example.in/s?k=2tb+ssd&ref=nb_sb_noss&session=abc123", "2tb ssd", "https://shop.example.in/s?k={searchTerms}"),
        ("https://shop.example.in/s?crid=X&field-q=2TB%20SSD&sprefix=2tb", "2tb ssd", "https://shop.example.in/s?field-q={searchTerms}"),
        ("https://videos.example/search/cats%20dancing?utm_source=x", "cats dancing", "https://videos.example/search/{searchTerms}"),
        ("https://shop.example.in/s?k=2tb+ssd&also=2tb+ssd", "2tb ssd", None),  # two slots: which one?
        ("https://shop.example.in/s?k=other", "2tb ssd", None),
        ("http://shop.example.in/s?k=2tb+ssd", "2tb ssd", None),
        # the browser's own URL: a session in a path parameter or credentials in the host never stay
        ("https://shop.example.in/search;jsessionid=ABC123?q=2tb+ssd&sid=9", "2tb ssd", "https://shop.example.in/search?q={searchTerms}"),
        ("https://videos.example/search;s=1/cats%20dancing;jsessionid=X", "cats dancing", "https://videos.example/search/{searchTerms}"),
        ("https://user:pw@shop.example.in:8443/find?q=ssd", "ssd", "https://shop.example.in:8443/find?q={searchTerms}"),
        ("https://shop.example.in/s/ref=nb_sb_noss_2?k=2tb+ssd", "2tb ssd", None),  # a token in the path: not learned
    ],
)
def test_sanitize_keeps_only_the_terms_slot(url, query, want):
    assert sites.sanitize(url, query) == want


def test_learning_a_landing_and_dropping_a_failing_template():
    sites.learn_landing("shop", HOST, "https://shop.example.in/s?k={searchTerms}")
    assert learned.get("sites", "shop|in") == {"host": HOST, "prefix": "", "via": "search"}
    assert learned.get("templates", HOST)["url"] == "https://shop.example.in/s?k={searchTerms}"
    sites.learn_landing("other", "other.example", "https://evil.example/s?q={searchTerms}")  # off-site: not kept
    assert learned.get("templates", "other.example") is None and learned.get("sites", "other|in") is not None
    sites.template_failed(HOST)
    assert learned.get("templates", HOST)["fails"] == 1
    sites.template_failed(HOST)
    assert learned.get("templates", HOST) is None
    sites.template_failed(HOST)  # nothing left to drop


# ---- a search page, and whether a landing is one ----------------------------------------------------------


@pytest.mark.parametrize(
    "template, query, want",
    [
        ("https://shop.example.in/s?k={searchTerms}", "2tb ssd & more", "https://shop.example.in/s?k=2tb+ssd+%26+more"),
        ("https://videos.example.org/search/{searchTerms}?hl=en", "cats/dogs", "https://videos.example.org/search/cats%2Fdogs?hl=en"),
        ("https://shop.example.in/s?k={searchTerms}", "नमस्ते", "https://shop.example.in/s?k=%E0%A4%A8%E0%A4%AE%E0%A4%B8%E0%A5%8D%E0%A4%A4%E0%A5%87"),
    ],
)
def test_fill_encodes_the_query_for_where_the_slot_sits(template, query, want):
    assert sites.fill(template, query) == want


def test_a_landing_is_on_the_site_and_holds_the_query():
    assert sites.registrable("https://www.shop.example.co.in/x") == "example.co.in"
    assert sites.registrable("10.0.0.1") == "" and sites.registrable("") == ""
    assert sites.on_site("https://m.shopkart.in/s?q=x", "www.shopkart.in")
    assert not sites.on_site("https://accounts.other.in/login", "www.shopkart.in")
    assert not sites.on_site("https://10.0.0.1/", "10.0.0.1")  # no registrable domain: never "the same site"
    assert sites.carries("https://shop.example.in/s?k=2TB+SSD&ref=x", "2tb ssd")
    assert sites.carries("https://videos.example.org/results#q=cats%20dancing", "Cats dancing")
    assert sites.carries(sites.fill("https://shop.example.in/s?k={searchTerms}", "नमस्ते दोस्त"), "नमस्ते दोस्त")
    assert not sites.carries("https://shop.example.in/", "2tb ssd")
    assert not sites.carries("https://shop.example.in/s?k=2tb", "2tb ssd")
    assert not sites.carries("https://shop.example.in/s?k=x", "  ")


# ---- review fixes: a site's sections, words in any script, how a word was learned ---------------------

MAPS = "https://www.shopkart.com/maps/search?q={searchTerms}"


def test_a_sites_section_keeps_its_own_template():
    sites.learn_landing("shopkart maps", "www.shopkart.com", MAPS, "/maps")
    assert sites.learned_template("www.shopkart.com", "/maps") == sites.Template(MAPS, "learned")
    assert sites.learned_template("www.shopkart.com") is None  # the whole site has none of its own
    sites.template_failed("www.shopkart.com")
    sites.template_failed("www.shopkart.com")
    assert sites.learned_template("www.shopkart.com", "/maps") is not None  # nobody blamed it
    sites.learn_landing("shopkart", "www.shopkart.com", MAPS)  # a section's search for the whole site
    assert sites.learned_template("www.shopkart.com") == sites.Template(MAPS, "learned")
    sites.learn_landing("shopkart maps", "www.shopkart.com", "https://www.shopkart.com/s?k={searchTerms}", "/maps")
    assert sites.learned_template("www.shopkart.com", "/maps").url == MAPS  # outside the section: not kept


def test_a_landing_on_a_section_must_be_inside_it():
    assert sites.on_site("https://m.shopkart.com/maps/search/pizza", "www.shopkart.com", "/maps")
    assert sites.on_site("https://www.shopkart.com/Maps", "www.shopkart.com", "/maps")
    assert not sites.on_site("https://www.shopkart.com/search?q=pizza", "www.shopkart.com", "/maps")
    assert not sites.on_site("https://www.shopkart.com/mapsearch", "www.shopkart.com", "/maps")
    assert sites.on_site("https://www.shopkart.com/search?q=pizza", "www.shopkart.com")


def test_a_sections_search_is_one_inside_it(monkeypatch):
    seen = serve(monkeypatch, {
        "https://www.shopkart.com/maps/": home(body='<form role="search" action="/s"><input name="k"></form>'),
    })
    assert asyncio.run(sites.discover("www.shopkart.com", "/maps")) == []  # the whole site's search
    assert seen == ["https://www.shopkart.com/maps/"]
    serve(monkeypatch, {
        "https://www.shopkart.com/maps/": home(body='<form role="search" action="search"><input name="q"></form>'),
    })
    assert asyncio.run(sites.discover("www.shopkart.com", "/maps")) == [sites.Template(MAPS, "site_form")]


def test_words_in_any_script_are_their_own_site_words(monkeypatch):
    search = FakeSearch([])
    monkeypatch.setattr(sites, "_search", search)
    assert sites._key("यूट्यूब", "in") != sites._key("फ्लिपकार्ट", "in")
    sites.learn_landing("यूट्यूब", "www.vidzone.tv", None)
    assert sites.site_like("यूट्यूब") is True
    assert sites.site_like("फ्लिपकार्ट") is None  # not the one word learned for every such word
    sites.learn_landing("?!", "www.shopkart.in", None)  # no letters or digits: nothing to learn
    assert sites.site_like("…") is None and asyncio.run(sites.resolve("…")) is None
    assert search.calls == [] and learned.get("sites", "") is None


def test_a_word_opened_as_a_site_is_no_claim_on_searches(monkeypatch):
    monkeypatch.setattr(sites, "_search", FakeSearch([]))
    sites.learn_landing("shopkart", "www.shopkart.in", None, via="open")
    assert sites.site_like("shopkart") is None  # "search news on <word>": a model decides
    assert asyncio.run(sites.resolve("shopkart")) == sites.SiteRes("www.shopkart.in", "learned", "in")
    sites.learn_landing("shopkart", "www.shopkart.in", None)  # a confirmed search on it
    assert sites.site_like("shopkart") is True
    sites.learn_landing("shopkart", "www.shopkart.in", None, via="open")  # opening it again keeps that
    assert sites.site_like("shopkart") is True


def test_everyday_words_are_the_ones_the_word_list_has_in_lower_case(monkeypatch, tmp_path):
    words = tmp_path / "words"
    words.write_text("cheap\nflight\nShopkart\nrecipe\n")
    monkeypatch.setattr(inventory, "_WORDS_FILE", str(words))
    monkeypatch.setattr(sites, "_lower_known", {})
    assert sites.everyday("cheap flights") and sites.everyday("Recipes")
    assert not sites.everyday("shopkart") and not sites.everyday("cheap shopkart")  # a name, capitalised
    assert not sites.everyday("") and not sites.everyday("2tb")
    sites.forget()
    monkeypatch.setattr(inventory, "_WORDS_FILE", str(tmp_path / "missing"))
    assert not sites.everyday("cheap")  # no word list: nothing is known to be an everyday word
