"""Which website a word means, and how to search it: by evidence, never from a table of sites.

* :func:`site_like` - from cached evidence only (fast-path checks): True for a domain, a site
  word a search here landed on or an installed web-app wrapper's site; False for an installed
  app, a time or "my <x>"; None when local evidence can't tell ("apple", "climate change", or a
  word only ever opened: "search news on <it>" may be about it).
* :func:`resolve` - the word as a host: a domain as given, a learned site, a web-app wrapper's
  site, else a web search for its website in this country: of the results whose own domain
  spells the word, the domain most of them agree on (this country's first), at its front host.
* :func:`learned_template` - the search template a landing on this site (or section) confirmed.
* :func:`discover` - the site's search URL template from what the site itself publishes: an
  OpenSearch description, a JSON-LD SearchAction, or a GET search form found by structure (a
  search role, a search input, the OS's own word for "Search"), with no list of parameter names.
  A section's ("<brand> maps" -> /maps) templates are the ones inside it.
* :func:`sanitize` - a search page's URL reduced to its terms slot (every other parameter, such
  as a session or tracking token, dropped).
* :func:`fill`, :func:`on_site`, :func:`carries` - a template's search page for a query, and
  whether a page a browser shows is on a site (inside its section) and holds a query (how a
  landing is confirmed).
* :func:`everyday` - whether words are everyday words, not a name (the system word list).

A site or template is learned only after a confirmed landing (:func:`learn_landing`), templates
per site and section (:attr:`SiteRes.key`); a search query is never stored. The network is
reached only through :func:`_fetch` (and the web search), whose transport tests replace.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Any
from urllib.parse import (
    parse_qsl,
    quote,
    quote_plus,
    unquote,
    unquote_plus,
    urlencode,
    urljoin,
    urlsplit,
    urlunsplit,
)

__all__ = ["SiteRes", "Template", "site_like", "resolve", "learned_template", "discover", "sanitize", "fill",
           "registrable", "on_site", "carries", "everyday", "learn_landing", "template_failed", "forget"]  # fmt: skip

TERMS = "{searchTerms}"
FETCH_S = 5.0
MAX_PAGE = 1_500_000
MAX_OSD = 256_000
SEARCH_TTL_S = 600.0
DROP_AFTER = 2  # blamed failures that drop a learned template
_transport: Any = None  # None = the network (tests put an httpx.MockTransport here)


@dataclass(frozen=True)
class SiteRes:
    host: str
    how: str  # domain | learned | webapp | search
    country: str
    prefix: str = ""  # a path the site lives under ("<brand> maps" -> /maps)

    @property
    def key(self) -> str:
        """The site as NEO keeps what it learns about it: a section apart from its whole site."""
        return f"{self.host}{self.prefix}"


@dataclass(frozen=True)
class Template:
    url: str  # https, on the site's own domain, exactly one {searchTerms}
    how: str  # learned | opensearch | jsonld | site_form


@dataclass(frozen=True)
class _Page:
    url: str
    text: str


# ---- evidence --------------------------------------------------------------------------------------


def _country() -> str:
    """The Mac's region ('in', 'gb', ...); '' when unknown."""
    try:
        from Foundation import NSLocale

        return str(NSLocale.currentLocale().countryCode() or "").lower()
    except Exception:  # noqa: BLE001
        return ""


def _language() -> str:
    try:
        from Foundation import NSLocale

        return str(NSLocale.currentLocale().languageCode() or "en").lower()
    except Exception:  # noqa: BLE001
        return "en"


def _country_name(cc: str) -> str:
    """The region's name in English ('in' -> 'India'), as a search engine reads it; '' if unknown."""
    if not cc:
        return ""
    try:
        from Foundation import NSLocale

        return str(NSLocale.alloc().initWithLocaleIdentifier_("en").localizedStringForCountryCode_(cc.upper()) or "")
    except Exception:  # noqa: BLE001
        return ""


@lru_cache(maxsize=1)
def _accept_language() -> str:
    """The user's languages, as a browser would send them: a site's search template must not
    carry a language the site guessed from where the request came from."""
    try:
        from Foundation import NSLocale

        langs = [str(x) for x in (NSLocale.preferredLanguages() or [])][:3]
    except Exception:  # noqa: BLE001
        langs = []
    return ", ".join(f"{lang};q={1 - n / 10:.1f}" if n else lang for n, lang in enumerate(langs)) or "en"


def _words(word: str) -> list[str]:
    """The word's letters and digits, in any script."""
    return re.findall(r"[^\W_]+", (word or "").casefold())


def _key(word: str, cc: str) -> str:
    """Where a site word is learned for this country; '' for a word with no letters or digits,
    which is never learned (every such word would share it)."""
    words = _words(word)
    return f"{' '.join(words)}|{cc}" if words else ""


def _learned_site(word: str, cc: str) -> dict | None:
    from neo2.core.decide import learned

    k = _key(word, cc)
    return learned.get("sites", k) if k else None


def _parts(url: str) -> Any:
    import tld

    return tld.get_tld(url if "://" in url else "https://" + url, as_object=True, fail_silently=True)


def _domain(word: str) -> str | None:
    """The host when ``word`` is written as a domain ('github.com', 'en.wikipedia.org')."""
    w = (word or "").strip().casefold()
    if "." not in w or " " in w:
        return None
    r = _parts(w)
    return r.parsed_url.hostname if r is not None else None


def _matches(word: str, url: str) -> tuple[str, str] | None:
    """(host, path prefix) when the site at ``url`` is the one ``word`` names: the registrable
    domain's first label spells the word, or spells its first word while the rest is a
    subdomain or the first path segment ("<brand> maps")."""
    r = _parts(url)
    words = _words(word)
    if r is None or not words:
        return None
    label, host = r.domain.casefold(), r.parsed_url.hostname or ""
    if "".join(words) == label:
        return host, ""
    rest = "".join(words[1:])
    if words[0] != label or not rest:
        return None
    if rest in [s.casefold() for s in r.subdomain.split(".") if s]:
        return host, ""
    seg = next((s for s in r.parsed_url.path.split("/") if s), "")
    return (host, f"/{seg}") if seg.casefold() == rest else None


def _country_tld(host: str) -> str:
    r = _parts(host)
    return r.tld.rsplit(".", 1)[-1] if r is not None else ""


def _front(host: str) -> bool:
    """A site's front host ('example.org', 'www.', 'm.'), not one of its side sites ('blog.')."""
    r = _parts(host)
    return r is not None and r.subdomain.casefold() in ("", "www", "m")


def _pick(word: str, rows: list, cc: str) -> tuple[str, str] | None:
    """(host, path prefix) of the site the results agree ``word`` names: of the registrable
    domains spelling it, one in this country if any, the one most results are on (the first on a
    tie), at its front host."""
    groups: dict[str, list[tuple[str, str]]] = {}
    for r in rows:
        if isinstance(r, dict) and (m := _matches(word, str(r.get("href") or ""))):
            groups.setdefault(registrable(m[0]), []).append(m)
    local = [g for g in groups.values() if cc and _country_tld(g[0][0]) == ("uk" if cc == "gb" else cc)]
    best = max(local or groups.values(), key=len, default=None)  # max keeps the first of equals
    return next((m for m in best if _front(m[0])), best[0]) if best else None


def _webapp(word: str, apps: list) -> tuple[str, str] | None:
    return next((m for a in apps if a.web_start_url and (m := _matches(word, a.web_start_url))), None)


def site_like(word: str) -> bool | None:
    """Is ``word`` a website? From cached evidence only (see the module docstring)."""
    from neo2.core.decide import inventory, when

    w = " ".join((word or "").split())
    if not w:
        return False
    low = w.casefold()
    if low.startswith("my ") or low.endswith((" app", " application")):
        return False
    apps = inventory.snapshot() or []
    searched = (_learned_site(w, _country()) or {}).get("via") == "search"
    # a site-specific browser is an app too, but the word names its site
    if _domain(w) is not None or searched or _webapp(w, apps):
        return True
    if inventory.named_in(w, apps) is not None or when.parse(w, datetime.now()) is not None:
        return False
    return None


def _search(query: str, region: str) -> list[dict]:
    """A web search's results ({"href": ...}); the one other place the network is reached."""
    from ddgs import DDGS

    return list(DDGS().text(query, region=region, max_results=10) or [])


_searched: dict[str, tuple[float, SiteRes | None]] = {}


def forget() -> None:
    """The privacy erase: the site words searched for lately and the words looked up in the word
    list (memory only) go too."""
    _searched.clear()
    _lower_known.clear()


async def resolve(word: str) -> SiteRes | None:
    """The website ``word`` names, first rule that applies: a domain as given, a site learned
    here, an installed web-app wrapper's site, then a web search (cached 10 min). The search asks
    for the word's website in this country: a bare name mostly finds pages about the site (an
    encyclopedia, its blog) rather than the site, and a store has one site per country."""
    from neo2.core.decide import inventory

    cc = _country()
    if host := _domain(word):
        return SiteRes(host, "domain", cc)
    if hit := _learned_site(word, cc):
        return SiteRes(str(hit.get("host", "")), "learned", cc, str(hit.get("prefix", "")))
    if m := _webapp(word, inventory.snapshot() or []):
        return SiteRes(m[0], "webapp", cc, m[1])
    if not (k := _key(word, cc)):
        return None  # no letters or digits: no site's name
    if (seen := _searched.get(k)) and time.monotonic() - seen[0] < SEARCH_TTL_S:
        return seen[1]
    # the search engine's own vocabulary names Great Britain's region 'uk'
    region = f"{'uk' if cc == 'gb' else cc or 'wt'}-{_language() if cc else 'wt'}"
    try:
        query = " ".join((*word.split(), "official website", _country_name(cc))).strip()
        rows = await asyncio.to_thread(_search, query, region)
    except Exception as e:  # noqa: BLE001 - no network: not a site as far as NEO can tell
        print(f"[decide] site search failed ({str(e)[:60]})")
        return None
    best = _pick(word, rows, cc)
    res = SiteRes(best[0], "search", cc, best[1]) if best else None
    _searched[k] = (time.monotonic(), res)
    return res


# ---- the site's own search ------------------------------------------------------------------------


async def _fetch(url: str, limit: int = MAX_PAGE, *, partial: bool = False) -> _Page | None:
    """GET a public https page (every hop checked: never this Mac or the local network), within
    FETCH_S and ``limit`` bytes, with no cookies kept. ``partial``: a longer page is cut at
    ``limit`` rather than refused (a results page lists its first items early)."""
    import httpx

    from neo2.core.tools.computer import policy
    from neo2.core.tools.web import _UA

    async def each_hop(request: httpx.Request) -> None:
        if why := await asyncio.to_thread(policy.private_host, str(request.url)):
            raise httpx.RequestError(f"not fetching {request.url.host}: {why}", request=request)

    try:
        async with (
            httpx.AsyncClient(
                follow_redirects=True,
                timeout=FETCH_S,
                headers={"User-Agent": _UA, "Accept-Language": _accept_language()},
                event_hooks={"request": [each_hop]},
                max_redirects=5,
                transport=_transport,
            ) as cl,
            cl.stream("GET", url) as r,
        ):
            if r.status_code != 200 or r.url.scheme != "https":
                return None
            body = bytearray()
            async for chunk in r.aiter_bytes():
                body += chunk
                if len(body) > limit:
                    if not partial:
                        return None
                    del body[limit:]
                    break
            return _Page(str(r.url), body.decode(r.encoding or "utf-8", errors="replace"))
    except (httpx.HTTPError, httpx.InvalidURL, UnicodeError, ValueError) as e:
        print(f"[decide] couldn't read {urlsplit(url).hostname} ({str(e)[:60]})")
        return None


def _same_site(url: str, host: str) -> bool:
    a, b = _parts(url), _parts(host)
    return a is not None and b is not None and urlsplit(url).scheme == "https" and a.fld == b.fld


def _under(url: str, prefix: str) -> bool:
    """Is ``url``'s path inside the section ``prefix`` ('' = the whole site)?"""
    if not prefix:
        return True
    try:
        path = urlsplit(url).path.casefold()
    except ValueError:
        return False
    p = prefix.casefold().rstrip("/")
    return path == p or path.startswith(p + "/")


def _one_slot(t: str) -> bool:
    return t.count(TERMS) == 1


def _from_osd(xml: str, host: str) -> str | None:
    """The HTML search template of an OpenSearch description (no DTDs, no entities, no network)."""
    from lxml import etree

    if "<!doctype" in xml[:2048].casefold() or "<!entity" in xml.casefold():
        return None  # a DTD could expand or fetch things: refused
    try:
        root = etree.fromstring(
            xml.encode(), etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False, huge_tree=False)
        )
    except etree.XMLSyntaxError:
        return None
    for el in root.iter("{*}Url", "Url"):
        if (el.get("type") or "").split(";")[0].strip() != "text/html":
            continue
        t = _drop_optional(el.get("template") or "")
        if _one_slot(t) and not re.search(r"\{(?!searchTerms\})[^}]*\}", t) and _same_site(t, host):
            return t  # any other required parameter would need a value NEO doesn't have
    return None


def _drop_optional(t: str) -> str:
    """An OpenSearch template without its optional parameters ({startPage?} and the like)."""
    base, _, query = t.partition("?")
    kept = [kv for kv in query.split("&") if kv and not re.fullmatch(r"[^=]*=\{[\w:]+\?\}", kv)]
    return base + ("?" + "&".join(kept) if kept else "")


def _jsonld(doc: Any, host: str) -> str | None:
    """A WebSite's SearchAction template from the page's JSON-LD."""
    for node in _walk(doc):
        types = node.get("@type")
        if "SearchAction" not in (types if isinstance(types, list) else [types]):
            continue
        target = node.get("target")
        target = target.get("urlTemplate") if isinstance(target, dict) else target
        qi = node.get("query-input") or node.get("query")
        name = re.search(r"name=([\w-]+)", qi if isinstance(qi, str) else "")
        if isinstance(target, str) and name and "{" + name.group(1) + "}" in target:
            t = target.replace("{" + name.group(1) + "}", TERMS)
            if _one_slot(t) and _same_site(t, host) and "{" not in t.replace(TERMS, ""):
                return t
    return None


def _walk(x: Any):
    if isinstance(x, dict):
        yield x
        for v in x.values():
            yield from _walk(v)
    elif isinstance(x, list):
        for v in x:
            yield from _walk(v)


def _visible_text_inputs(form: Any) -> list[Any]:
    return [
        i
        for i in form.iter("input")
        if (i.get("type") or "text").lower() in ("text", "search") and i.get("name") and i.get("hidden") is None
    ]


def _search_form(page: _Page, doc: Any, host: str) -> str | None:
    """A GET form that is a search by its structure: a search role or element around it, or one
    visible text input that is a search input or labelled with the OS's word for "Search"."""
    from neo2.core.decide import axkit

    words = [w.casefold() for w in axkit.lexicon().get("search", ()) if w]
    for form in doc.iter("form"):
        if (form.get("method") or "get").lower() != "get":
            continue
        inputs = _visible_text_inputs(form)
        if len(inputs) != 1:
            continue
        box = inputs[0]
        roles = [a for a in (form, *form.iterancestors()) if a.get("role") == "search" or a.tag == "search"]
        labelled = any(
            re.search(rf"(?<!\w){re.escape(w)}(?!\w)", (box.get(a) or "").casefold())
            for a in ("placeholder", "aria-label", "title")
            for w in words
        )
        if not (roles or (box.get("type") or "").lower() == "search" or labelled):
            continue
        action = urljoin(page.url, form.get("action") or page.url)
        keep = [
            (i.get("name"), i.get("value") or "")
            for i in form.iter("input")
            if (i.get("type") or "").lower() == "hidden" and i.get("name")
        ]
        u = urlsplit(action)
        query = urlencode(keep) + ("&" if keep else "") + f"{quote(box.get('name'))}={TERMS}"
        t = urlunsplit((u.scheme, u.netloc, u.path or "/", query, ""))
        if _same_site(t, host) and _one_slot(t):
            return t
    return None


def learned_template(host: str, prefix: str = "") -> Template | None:
    """The search template a landing on this site (or its section) confirmed; no network."""
    from neo2.core.decide import learned

    hit = learned.get("templates", f"{host}{prefix}")
    url = str((hit or {}).get("url", ""))
    return Template(url, "learned") if _one_slot(url) else None


def _read_page(page: _Page, host: str) -> tuple[list[str], list[Template]]:
    """(the site's OpenSearch description URLs, the templates the page itself holds: JSON-LD, a
    search form). Parsing up to MAX_PAGE of HTML takes tens of ms: called off the event loop."""
    from lxml import etree, html

    try:
        doc = html.fromstring(page.text)
    except (etree.LxmlError, ValueError, TypeError):  # a page with no elements ("Document is empty")
        return [], []
    osd: list[str] = []
    for link in doc.iter("link"):
        rel = (link.get("rel") or "").lower().split()
        if "search" in rel and (link.get("type") or "").lower() == "application/opensearchdescription+xml":
            osd_url = urljoin(page.url, link.get("href") or "")
            if _same_site(osd_url, host):
                osd.append(osd_url)
    found: list[Template] = []
    for script in doc.iter("script"):
        if (script.get("type") or "").lower() != "application/ld+json":
            continue
        try:
            t = _jsonld(json.loads(script.text or ""), host)
        except (ValueError, RecursionError):  # not JSON, or nested too deep to read
            continue
        if t:
            found.append(Template(t, "jsonld"))
            break
    if t := _search_form(page, doc, host):
        found.append(Template(t, "site_form"))
    return osd, found


async def discover(host: str, prefix: str = "") -> list[Template]:
    """What the site publishes about its search, best first: OpenSearch, JSON-LD, a search form
    (for a section, only those inside it). Reads the site's page: :func:`learned_template` first."""
    out: list[Template] = []
    page = await _fetch(f"https://{host}{prefix}/")
    if page is None or not page.text.strip():
        return out
    osd_urls, found = await asyncio.to_thread(_read_page, page, host)
    for osd_url in osd_urls:
        if (osd := await _fetch(osd_url, MAX_OSD)) and (t := await asyncio.to_thread(_from_osd, osd.text, host)):
            out.append(Template(t, "opensearch"))
            break
    out += found
    unique: dict[str, Template] = {}
    for t in out:
        if _under(t.url, prefix):  # a section's page may offer the whole site's search
            unique.setdefault(t.url, t)  # the same URL again from a lesser source adds nothing
    return list(unique.values())


def sanitize(url: str, query: str) -> str | None:
    """The template of a search page NEO landed on: the one parameter (or path segment) holding
    the query becomes {searchTerms} and everything else is dropped (other parameters, ';' path
    parameters, credentials in the host); None if there isn't exactly one such slot, another
    path segment carries a parameter ("ref=..."), or the page isn't https."""
    u = urlsplit(url)
    want = " ".join(query.split()).casefold()
    if u.scheme != "https" or not want:
        return None
    params = [(k, v) for k, v in parse_qsl(u.query, keep_blank_values=True) if " ".join(v.split()).casefold() == want]
    path = [s.split(";", 1)[0] for s in u.path.split("/")]  # ";jsessionid=...": session state
    segs = [i for i, s in enumerate(path) if " ".join(unquote(s).replace("+", " ").split()).casefold() == want]
    if len(params) + len(segs) != 1:
        return None
    if params:
        query_part = f"{quote(params[0][0])}={TERMS}"
    else:
        path[segs[0]], query_part = TERMS, ""
    if any("=" in s for s in path):
        return None  # a token or a tracking value in the path: which parts the site needs can't be told
    netloc = u.netloc.rpartition("@")[2]  # never a user name or password
    return urlunsplit((u.scheme, netloc, "/".join(path), query_part, ""))


def fill(template: str, query: str) -> str:
    """The search page for ``query``: the template's terms slot filled in, encoded for where it
    sits (a query parameter or a path segment)."""
    base, sep, params = template.partition("?")
    if TERMS in params:
        return base + sep + params.replace(TERMS, quote_plus(query))
    return base.replace(TERMS, quote(query, safe="")) + sep + params


def registrable(url_or_host: str) -> str:
    """The registrable domain ('shop.example.co.in' -> 'example.co.in'); '' when there is none."""
    r = _parts(url_or_host)
    return str(r.fld).casefold() if r is not None else ""


def on_site(url: str, host: str, prefix: str = "") -> bool:
    """Is the page at ``url`` on the site ``host`` belongs to (the same registrable domain), and
    inside its section ``prefix`` when it has one?"""
    mine = registrable(host)
    return bool(mine) and registrable(url) == mine and _under(url, prefix)


_lower_known: dict[str, bool] = {}
_lower_lock = threading.Lock()


def everyday(words: str) -> bool:
    """Is every word of ``words`` an everyday word: in the system word list in lower case (plurals
    included)? A name the list has only capitalised is none, and without the list nothing is."""
    from neo2.core.decide import inventory

    ws = set(_words(words))
    with _lower_lock:
        known = {w: _lower_known[w] for w in ws if w in _lower_known}
        if todo := ws - known.keys():
            want: dict[str, set[str]] = {}
            for w in todo:
                for v in inventory._variants(w):
                    want.setdefault(v, set()).add(w)
            hit: set[str] = set()
            try:  # one streaming pass, as inventory reads the list, but case kept
                with open(inventory._WORDS_FILE, encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if owners := want.get(line.strip()):
                            hit |= owners
            except OSError:
                pass
            if len(_lower_known) > 4096:
                _lower_known.clear()
            _lower_known.update(found := {w: w in hit for w in todo})
            known.update(found)
    return bool(ws) and all(known.values())


def carries(url: str, query: str) -> bool:
    """Does the page's address hold every word of ``query`` (decoded, any case, any script)?"""
    words = re.findall(r"\w+", query.casefold())
    try:
        u = urlsplit(url)
    except ValueError:
        return False
    shown = unquote_plus(f"{u.path} {u.query} {u.fragment}").casefold()
    return bool(words) and all(w in shown for w in words)


def learn_landing(word: str, host: str, template: str | None, prefix: str = "", via: str = "search") -> None:
    """A page landed where it should: remember the site for this word here, how it got there
    (``via`` 'search', or 'open' for the site's page alone, which a search later outranks), and
    the site's (or section's) template."""
    from neo2.core.decide import learned

    cc = _country()
    if k := _key(word, cc):
        old = _learned_site(word, cc) or {}
        same = (old.get("host"), old.get("prefix", "")) == (host, prefix)
        searched = via == "search" or (same and old.get("via") == "search")
        learned.put("sites", k, {"host": host, "prefix": prefix, "via": "search" if searched else via})
    if template and _one_slot(template) and _same_site(template, host) and _under(template, prefix):
        learned.put("templates", f"{host}{prefix}", {"url": template, "fails": 0})


def template_failed(host: str, prefix: str = "") -> None:
    """A learned template's search didn't land (the site changed): dropped after DROP_AFTER."""
    from neo2.core.decide import learned

    key = f"{host}{prefix}"
    hit = learned.get("templates", key)
    if hit is None:
        return
    fails = int(hit.get("fails", 0)) + 1
    if fails >= DROP_AFTER:
        learned.drop("templates", key)
    else:
        learned.put("templates", key, {**hit, "fails": fails})
