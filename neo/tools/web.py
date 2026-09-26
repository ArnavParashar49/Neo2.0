"""Web search (DuckDuckGo, no key) + page fetch/extract (trafilatura). Free and fast."""

from __future__ import annotations

import asyncio

import httpx

from neo.agent.registry import ToolContext, tool

# "search X on amazon", "open it on google": a real search page on that site, in the browser.
_SITES: dict[str, str] = {
    "google": "https://www.google.com/search?q={q}",
    "google images": "https://www.google.com/search?tbm=isch&q={q}",
    "images": "https://www.google.com/search?tbm=isch&q={q}",
    "google maps": "https://www.google.com/maps/search/{q}",
    "maps": "https://www.google.com/maps/search/{q}",
    "youtube": "https://www.youtube.com/results?search_query={q}",
    "amazon": "https://www.{amazon}/s?k={q}",
    "flipkart": "https://www.flipkart.com/search?q={q}",
    "ebay": "https://www.ebay.com/sch/i.html?_nkw={q}",
    "wikipedia": "https://en.wikipedia.org/w/index.php?search={q}",
    "github": "https://github.com/search?q={q}&type=repositories",
    "reddit": "https://www.reddit.com/search/?q={q}",
    "stack overflow": "https://stackoverflow.com/search?q={q}",
    "stackoverflow": "https://stackoverflow.com/search?q={q}",
    "twitter": "https://x.com/search?q={q}",
    "x": "https://x.com/search?q={q}",
    "linkedin": "https://www.linkedin.com/search/results/all/?keywords={q}",
    "spotify": "https://open.spotify.com/search/{q}",
    "netflix": "https://www.netflix.com/search?q={q}",
    "imdb": "https://www.imdb.com/find/?q={q}",
    "duckduckgo": "https://duckduckgo.com/?q={q}",
    "bing": "https://www.bing.com/search?q={q}",
    "news": "https://news.google.com/search?q={q}",
    "google news": "https://news.google.com/search?q={q}",
    "pinterest": "https://www.pinterest.com/search/pins/?q={q}",
    "instagram": "https://www.instagram.com/explore/search/keyword/?q={q}",
}
_AMAZON = {
    "IN": "amazon.in", "GB": "amazon.co.uk", "DE": "amazon.de", "FR": "amazon.fr", "IT": "amazon.it",
    "ES": "amazon.es", "CA": "amazon.ca", "JP": "amazon.co.jp", "AU": "amazon.com.au", "MX": "amazon.com.mx",
    "BR": "amazon.com.br", "NL": "amazon.nl", "AE": "amazon.ae", "SG": "amazon.sg", "SE": "amazon.se",
}
SITE_NAMES = sorted(_SITES, key=len, reverse=True)
_SITE_RE = "|".join(n.replace(" ", r"\s+") for n in SITE_NAMES)
_VAGUE_QUERY = r"(?!(?:it|this|that|these|those|them|one|the\s+same|the\s+first\s+one|the\s+cheaper\s+one)\s+(?:on|in)\b)"


def _country() -> str:
    try:
        from Foundation import NSLocale

        return str(NSLocale.currentLocale().countryCode() or "")
    except Exception:  # noqa: BLE001
        return ""


def site_search_url(site: str, query: str) -> str:
    from urllib.parse import quote_plus

    key = " ".join(site.lower().replace("www.", "").replace(".com", "").split())
    q = quote_plus(query.strip())
    if key in _SITES:
        return _SITES[key].format(q=q, amazon=_AMAZON.get(_country(), "amazon.com"))
    if "." in site:  # a domain: Google, limited to that site
        return f"https://www.google.com/search?q={quote_plus(f'site:{site.strip()} {query.strip()}')}"
    return f"https://www.google.com/search?q={quote_plus(f'{query.strip()} {site.strip()}')}"


_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"


def _search(query: str, n: int) -> str:
    from ddgs import DDGS

    with DDGS() as d:
        rows = list(d.text(query, max_results=n))
    if not rows:
        return "No results."
    return "\n".join(
        f"{i + 1}. {r.get('title', '')}\n   {r.get('href', '')}\n   {r.get('body', '')[:220]}"
        for i, r in enumerate(rows)
    )


def _news(query: str, n: int) -> str:
    from ddgs import DDGS

    with DDGS() as d:
        rows = list(d.news(query, max_results=n))
    return (
        "\n".join(
            f"{i + 1}. {r.get('title', '')} — {r.get('source', '')} ({r.get('date', '')[:10]})\n   {r.get('url', '')}"
            for i, r in enumerate(rows)
        )
        or "No news found."
    )


@tool(
    "web_search",
    "Search the web. Returns titles, URLs and snippets. Use web_fetch to read a result.",
    {
        "type": "object",
        "properties": {"query": {"type": "string"}, "n": {"type": "integer"}, "news": {"type": "boolean"}},
        "required": ["query"],
    },
    parallel_safe=True,
    category="web",
    slow=True,
    chain=False,  # only as the whole utterance: a later clause means 'in that app'
    fast_path=[
        (
            # "search for X", "google X", "look up X" — but not "search for X in Notes"
            r"^\s*(?:please\s+)?(?:search(?:\s+the\s+(?:web|internet))?(?:\s+for)?|google|look\s+up|"
            r"web\s+search(?:\s+for)?|find\s+(?:me\s+)?(?:info(?:rmation)?\s+(?:on|about)|out\s+about))\s+"
            rf"(?!.*\b(?:in|on)\s+(?:notes|spotify|finder|mail|messages|slack|my\s+\w+|the\s+\w+\s+app|{_SITE_RE})\b)"
            rf"(?!(?:{_SITE_RE})\s+for\b)"  # "search amazon for X" opens the site's search
            r"(?P<query>.{3,}?)\s*[?.!]*\s*$",
            {"query": "<query>"},
        )
    ],
    payload=True,
)
async def web_search(a: dict, c: ToolContext) -> str:
    n = int(a.get("n") or 6)
    fn = _news if a.get("news") else _search
    return await asyncio.to_thread(fn, a["query"], n)


@tool(
    "web_fetch",
    "Fetch a URL and return its main text content (articles, docs, pages).",
    {
        "type": "object",
        "properties": {"url": {"type": "string"}, "max_chars": {"type": "integer"}},
        "required": ["url"],
    },
    parallel_safe=True,
    category="web",
    slow=True,
)
async def web_fetch(a: dict, c: ToolContext) -> str:
    url = a["url"]
    if not url.startswith("http"):
        url = "https://" + url
    limit = int(a.get("max_chars") or 12000)
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=20, headers={"User-Agent": _UA}) as cl:
            r = await cl.get(url)
            r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        return f"Error fetching {url}: {e}"
    ctype = r.headers.get("content-type", "")
    if "pdf" in ctype:
        import io

        from pypdf import PdfReader

        rd = PdfReader(io.BytesIO(r.content))
        return "\n".join((p.extract_text() or "") for p in rd.pages[:30])[:limit]
    import trafilatura

    text = trafilatura.extract(r.text, include_links=False, include_tables=True, favor_recall=True) or ""
    if not text.strip():
        text = trafilatura.html2txt(r.text)
    return text[:limit] or "(no readable text)"


@tool(
    "search_site",
    "Open a search for `query` on a website in the browser — google, youtube, amazon (the user's "
    "country store), flipkart, maps, wikipedia, github, reddit, spotify, netflix… or any domain. "
    "Use for 'search X on amazon', 'open it on google', 'find X on youtube'. The query must be the "
    "actual thing (e.g. the product name from the conversation), never 'it'.",
    {
        "type": "object",
        "properties": {"site": {"type": "string"}, "query": {"type": "string"}},
        "required": ["site", "query"],
    },
    category="web",
    quiet=True,
    payload=True,
    chain=True,
    fast_path=[
        (
            # "search 2tb ssd on amazon", "look up cat videos on youtube", "find X in google maps"
            rf"^\s*(?:please\s+)?(?:search(?:\s+for)?|look\s+up|find|open|show\s+me|google)\s+{_VAGUE_QUERY}"
            rf"(?P<query>.+?)\s+(?:on|in|at)\s+(?P<site>{_SITE_RE})\s*[.!?]*\s*$",
            {"site": "<site>", "query": "<query>"},
        ),
        (
            # "search amazon for 2tb ssd", "search youtube for cat videos"
            rf"^\s*(?:please\s+)?(?:search|look\s+up|check)\s+(?P<site>{_SITE_RE})\s+for\s+(?P<query>.+?)\s*[.!?]*\s*$",
            {"site": "<site>", "query": "<query>"},
        ),
        (
            # "open youtube and search for cat videos"
            rf"^\s*(?:please\s+)?(?:open|go\s+to)\s+(?P<site>{_SITE_RE})\s+(?:and|then)\s+(?:search|look\s+up|find)(?:\s+for)?\s+"
            r"(?P<query>.+?)\s*[.!?]*\s*$",
            {"site": "<site>", "query": "<query>"},
        ),
    ],
)
async def search_site(a: dict, c: ToolContext) -> str:
    from neo.tools.computer import apps

    site, query = str(a["site"]).strip(), str(a["query"]).strip()
    if not query or query.lower() in ("it", "this", "that", "them", "one"):
        return "Error: search for what? Name the thing to search for."
    url = site_search_url(site, query)
    out = await apps.open_app(url)
    return out if out.startswith(("Error", "NEEDS_")) else f"Searching {site} for “{query}”."
