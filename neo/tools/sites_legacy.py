"""web_search's legacy: the table of known sites and their search pages ("search X on amazon").

Moved verbatim from neo/tools/computer/legacy.py, which re-exports it: plain Python, so the web
tools import it without loading the Mac tools (or any macOS framework). It runs only while the
web_search kind is `legacy`/`shadow` (neo.config `decide_modes`) and goes away with the rest of
legacy.py once that kind's evidence-chosen driver is on everywhere.
"""

from __future__ import annotations

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
# Sites named by a generic word ("maps", "news", "images", "x") need an explicit search verb —
# "open the news on …" / "find x on …" are too ambiguous for a fast path.
_GENERIC_SITES = {"maps", "news", "images", "x"}
_ENGINE_RE = "|".join(n.replace(" ", r"\s+") for n in SITE_NAMES if n not in _GENERIC_SITES)


def site_search_url(site: str, query: str) -> str:
    from urllib.parse import quote_plus

    from neo.tools.web import _country  # web.py imports this module: read at call time

    key = " ".join(site.lower().replace("www.", "").replace(".com", "").split())
    q = quote_plus(query.strip())
    if key in _SITES:
        return _SITES[key].format(q=q, amazon=_AMAZON.get(_country(), "amazon.com"))
    if "." in site:  # a domain: Google, limited to that site
        return f"https://www.google.com/search?q={quote_plus(f'site:{site.strip()} {query.strip()}')}"
    return f"https://www.google.com/search?q={quote_plus(f'{query.strip()} {site.strip()}')}"

