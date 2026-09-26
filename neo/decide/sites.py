"""Which website a word means, and how to search it: by evidence, never from a table of sites.

* :func:`site_like` - from cached evidence only (fast-path checks): True for a domain, a site
  word NEO learned here or an installed web-app wrapper's site; False for an installed app, a
  time or "my <x>"; None when local evidence can't tell ("apple", "climate change").
* :func:`resolve` - the word as a host: a domain as given, a learned site, a web-app wrapper's
  site, else a region-aware web search whose result's own domain spells the word.
* :func:`discover` - the site's search URL template from what the site itself publishes: an
  OpenSearch description, a JSON-LD SearchAction, or a GET search form found by structure (a
  search role, a search input, the OS's own word for "Search"), with no list of parameter names.
* :func:`sanitize` - a search page's URL reduced to its terms slot (every other parameter, such
  as a session or tracking token, dropped).

A site or template is learned only after a confirmed landing (:func:`learn_landing`); a search
query is never stored. The network is reached only through :func:`_fetch` (and the web search),
whose transport tests replace.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import parse_qsl, quote, unquote, urlencode, urljoin, urlsplit, urlunsplit

__all__ = ["SiteRes", "Template", "site_like", "resolve", "discover", "sanitize", "learn_landing", "template_failed"]

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


def _words(word: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (word or "").casefold())


def _key(word: str, cc: str) -> str:
    return f"{' '.join(_words(word))}|{cc}"


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


def _webapp(word: str, apps: list) -> tuple[str, str] | None:
    return next((m for a in apps if a.web_start_url and (m := _matches(word, a.web_start_url))), None)


def site_like(word: str) -> bool | None:
    """Is ``word`` a website? From cached evidence only (see the module docstring)."""
    from neo.decide import inventory, learned, when

    w = " ".join((word or "").split())
    if not w:
        return False
    low = w.casefold()
    if low.startswith("my ") or low.endswith((" app", " application")):
        return False
    apps = inventory.snapshot() or []
    # a site-specific browser is an app too, but the word names its site
    if _domain(w) is not None or learned.get("sites", _key(w, _country())) is not None or _webapp(w, apps):
        return True
    if inventory.named_in(w, apps) is not None or when.parse(w, datetime.now()) is not None:
        return False
    return None


def _search(query: str, region: str) -> list[dict]:
    """A web search's results ({"href": ...}); the one other place the network is reached."""
    from ddgs import DDGS

    return list(DDGS().text(query, region=region, max_results=8) or [])


_searched: dict[str, tuple[float, SiteRes | None]] = {}


async def resolve(word: str) -> SiteRes | None:
    """The website ``word`` names, first rule that applies: a domain as given, a site learned
    here, an installed web-app wrapper's site, then a region-aware web search (cached 10 min)."""
    from neo.decide import inventory, learned

    cc = _country()
    if host := _domain(word):
        return SiteRes(host, "domain", cc)
    if hit := learned.get("sites", _key(word, cc)):
        return SiteRes(str(hit.get("host", "")), "learned", cc, str(hit.get("prefix", "")))
    if m := _webapp(word, inventory.snapshot() or []):
        return SiteRes(m[0], "webapp", cc, m[1])
    k = _key(word, cc)
    if (seen := _searched.get(k)) and time.monotonic() - seen[0] < SEARCH_TTL_S:
        return seen[1]
    # the search engine's own vocabulary names Great Britain's region 'uk'
    region = f"{'uk' if cc == 'gb' else cc or 'wt'}-{_language() if cc else 'wt'}"
    try:
        rows = await asyncio.to_thread(_search, " ".join(word.split()), region)
    except Exception as e:  # noqa: BLE001 - no network: not a site as far as NEO can tell
        print(f"[decide] site search failed ({str(e)[:60]})")
        return None
    hits = [m for r in rows if isinstance(r, dict) and (m := _matches(word, str(r.get("href") or "")))]
    local = [m for m in hits if cc and _country_tld(m[0]) == ("uk" if cc == "gb" else cc)]
    best = (local or hits or [None])[0]  # the store for this country when the site has one
    res = SiteRes(best[0], "search", cc, best[1]) if best else None
    _searched[k] = (time.monotonic(), res)
    return res


# ---- the site's own search ------------------------------------------------------------------------


async def _fetch(url: str, limit: int = MAX_PAGE) -> _Page | None:
    """GET a public https page (every hop checked: never this Mac or the local network), within
    FETCH_S and ``limit`` bytes, with no cookies kept."""
    import httpx

    from neo.tools.computer import policy
    from neo.tools.web import _UA

    async def each_hop(request: httpx.Request) -> None:
        if why := await asyncio.to_thread(policy.private_host, str(request.url)):
            raise httpx.RequestError(f"not fetching {request.url.host}: {why}", request=request)

    try:
        async with (
            httpx.AsyncClient(
                follow_redirects=True,
                timeout=FETCH_S,
                headers={"User-Agent": _UA},
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
                    return None
            return _Page(str(r.url), body.decode(r.encoding or "utf-8", errors="replace"))
    except (httpx.HTTPError, httpx.InvalidURL, UnicodeError, ValueError) as e:
        print(f"[decide] couldn't read {urlsplit(url).hostname} ({str(e)[:60]})")
        return None


def _same_site(url: str, host: str) -> bool:
    a, b = _parts(url), _parts(host)
    return a is not None and b is not None and urlsplit(url).scheme == "https" and a.fld == b.fld


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
    from neo.decide import axkit

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


async def discover(host: str, prefix: str = "") -> list[Template]:
    """The site's search templates, best first: learned, OpenSearch, JSON-LD, a search form."""
    from lxml import html

    from neo.decide import learned

    out: list[Template] = []
    if (hit := learned.get("templates", host)) and _one_slot(str(hit.get("url", ""))):
        out.append(Template(str(hit["url"]), "learned"))
    page = await _fetch(f"https://{host}{prefix}/")
    if page is None or not page.text.strip():
        return out
    try:
        doc = html.fromstring(page.text)
    except (ValueError, TypeError):
        return out
    for link in doc.iter("link"):
        rel = (link.get("rel") or "").lower().split()
        if "search" in rel and (link.get("type") or "").lower() == "application/opensearchdescription+xml":
            osd_url = urljoin(page.url, link.get("href") or "")
            if _same_site(osd_url, host) and (osd := await _fetch(osd_url, MAX_OSD)) and (t := _from_osd(osd.text, host)):
                out.append(Template(t, "opensearch"))
                break
    for script in doc.iter("script"):
        if (script.get("type") or "").lower() != "application/ld+json":
            continue
        try:
            t = _jsonld(json.loads(script.text or ""), host)
        except ValueError:
            continue
        if t:
            out.append(Template(t, "jsonld"))
            break
    if t := _search_form(page, doc, host):
        out.append(Template(t, "site_form"))
    unique: dict[str, Template] = {}
    for t in out:
        unique.setdefault(t.url, t)  # the same URL again from a lesser source adds nothing
    return list(unique.values())


def sanitize(url: str, query: str) -> str | None:
    """The template of a search page NEO landed on: the one parameter (or path segment) holding
    the query becomes {searchTerms} and everything else is dropped; None if there isn't exactly
    one such slot, or the page isn't https."""
    u = urlsplit(url)
    want = " ".join(query.split()).casefold()
    if u.scheme != "https" or not want:
        return None
    params = [(k, v) for k, v in parse_qsl(u.query, keep_blank_values=True) if " ".join(v.split()).casefold() == want]
    segs = [s for s in u.path.split("/") if " ".join(unquote(s).replace("+", " ").split()).casefold() == want]
    if len(params) + len(segs) != 1:
        return None
    if params:
        return urlunsplit((u.scheme, u.netloc, u.path, f"{quote(params[0][0])}={TERMS}", ""))
    path = "/".join(TERMS if s == segs[0] else s for s in u.path.split("/"))
    return urlunsplit((u.scheme, u.netloc, path, "", ""))


def learn_landing(word: str, host: str, template: str | None, prefix: str = "") -> None:
    """A search landed where it should: remember the site for this word here, and its template."""
    from neo.decide import learned

    learned.put("sites", _key(word, _country()), {"host": host, "prefix": prefix})
    if template and _one_slot(template) and _same_site(template, host):
        learned.put("templates", host, {"url": template, "fails": 0})


def template_failed(host: str) -> None:
    """A learned template's search didn't land (the site changed): dropped after DROP_AFTER."""
    from neo.decide import learned

    hit = learned.get("templates", host)
    if hit is None:
        return
    fails = int(hit.get("fails", 0)) + 1
    if fails >= DROP_AFTER:
        learned.drop("templates", host)
    else:
        learned.put("templates", host, {**hit, "fails": fails})
