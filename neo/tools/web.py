"""Web search (DuckDuckGo, no key) + page fetch/extract (trafilatura). Free and fast."""

from __future__ import annotations

import asyncio
import logging
import re
from urllib.parse import urlsplit

import httpx

from neo.agent.registry import ToolContext, tool
from neo.config import decide_mode
from neo.tools import render

# The table of known sites and their search pages: web_search's legacy, until the kind is on.
from neo.tools.sites_legacy import _ENGINE_RE, _SITE_RE, site_search_url

# "search for it on amazon", "open that one on google": the thing comes from the conversation,
# which only the model has — never a fast path.
_VAGUE_QUERY = (
    r"(?!(?:for\s+)?(?:it|this|that|these|those|them|one|the\s+same(?:\s+\w+)?|(?:the|this|that)\s+(?:\w+\s+)?one)"
    r"\s+(?:on|in|at)\b)"
)
_VAGUE_QUERY_AFTER_FOR = r"(?!(?:for\s+)?(?:it|this|that|these|those|them|one|(?:the|this|that)\s+(?:\w+\s+)?one)\s*[.!?]*\s*$)"
# a site as the user says it: a name or a domain, up to three words ("<brand> maps")
_SITE = r"(?P<site>[\w.+-]+(?:\s+[\w.+-]+){0,2})"

log = logging.getLogger(__name__)


# While web_search is on, which site a search means is told by evidence (neo.decide.kinds.site):
# the patterns with a site table give way to generic ones whose checks claim known sites only.
def _legacy_fp(filled: dict) -> bool:
    return decide_mode("web_search") != "on"


def _claims_web(filled: dict) -> bool:
    if decide_mode("web_search") != "on":  # before the import: legacy never loads neo.decide
        return False
    from neo.decide.kinds import site

    return site.claims_web_search(filled)


def _claims_site(filled: dict) -> bool:
    if decide_mode("web_search") != "on":
        return False
    from neo.decide.kinds import site

    return site.claims_search_site(filled)


def _country() -> str:
    try:
        from Foundation import NSLocale

        return str(NSLocale.currentLocale().countryCode() or "")
    except Exception:  # noqa: BLE001
        return ""


_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"


# The library's "auto" backend tries engines one after another (~2 s). Racing three engines and
# taking the first that answers with results is ~0.6–0.8 s — and an empty answer from one engine
# costs nothing, because the others are already running.
_RACE = ("duckduckgo", "brave", "bing")
_RACE_TIMEOUT_S = 4.0


def _one(query: str, n: int, backend: str) -> list[dict]:
    from ddgs import DDGS

    with DDGS() as d:
        return list(d.text(query, max_results=n, backend=backend))


def _race(query: str, n: int) -> list[dict]:
    import concurrent.futures as cf

    pool = cf.ThreadPoolExecutor(max_workers=len(_RACE))
    try:
        futs = [pool.submit(_one, query, n, b) for b in _RACE]
        for f in cf.as_completed(futs, timeout=_RACE_TIMEOUT_S):
            try:
                rows = f.result()
            except Exception:  # noqa: BLE001 — that engine failed; the others may not have
                continue
            if rows:
                return rows
    except cf.TimeoutError:
        pass
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return []


def _search(query: str, n: int) -> str:
    rows = _race(query, n)
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
            _legacy_fp,
        ),
        (
            # the same verbs; what the query ends with ("in <W>", "on <W>") is judged by evidence
            r"^\s*(?:please\s+)?(?:search(?:\s+the\s+(?:web|internet))?(?:\s+for)?|google|look\s+up|"
            r"web\s+search(?:\s+for)?|find\s+(?:me\s+)?(?:info(?:rmation)?\s+(?:on|about)|out\s+about))\s+"
            r"(?P<query>.{3,}?)\s*[?.!]*\s*$",
            {"query": "<query>"},
            _claims_web,
        ),
    ],
    payload=True,
    untrusted=True,
    read_only=True,
)
async def web_search(a: dict, c: ToolContext) -> str:
    n = int(a.get("n") or 6)
    fn = _news if a.get("news") else _search
    return await asyncio.to_thread(fn, a["query"], n)


@tool(
    "web_fetch",
    "Fetch a URL and return its main text content (articles, docs, pages — also ones built by JavaScript).",
    {
        "type": "object",
        "properties": {"url": {"type": "string"}, "max_chars": {"type": "integer"}},
        "required": ["url"],
    },
    parallel_safe=True,
    category="web",
    slow=True,
    untrusted=True,
    read_only=True,
)
async def web_fetch(a: dict, c: ToolContext) -> str:
    from neo.agent import confirm
    from neo.security import policy

    url = str(a["url"]).strip()
    if not re.match(r"^[a-z][\w+.-]*://", url, re.I):
        url = "https://" + url
    # A background fetch only goes to the public internet: never this Mac's own services, the
    # router or other machines on the network (a page could ask for them).
    if why := await asyncio.to_thread(policy.private_host, url):
        return f"Error: not fetching {url} — {why}."
    if c.tainted() and policy.url_carries_data(url):
        gate = confirm.gated(
            "web_fetch", a, f"fetch a link that carries data (after reading {c.tainted()})", detail=url
        )
        if isinstance(gate, str):
            return gate
    limit = int(a.get("max_chars") or 12000)

    async def _each_hop(request: httpx.Request) -> None:
        if why := await asyncio.to_thread(policy.private_host, str(request.url)):
            raise httpx.RequestError(f"redirected to {request.url.host} — {why}", request=request)

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=20,
            headers={"User-Agent": _UA},
            event_hooks={"request": [_each_hop]},
            max_redirects=5,
        ) as cl:
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
    html = r.text if _is_html(ctype, r.text) else ""
    if not html:
        text = await asyncio.to_thread(_main_text, r.text) or await asyncio.to_thread(_plain_text, r.text)
        return text[:limit] or "(no readable text)"
    main, full, shell = await asyncio.to_thread(_judge, html)
    text = main if main.strip() else full
    # Built by its scripts: what the server sent is a shell. Never a link that carries data (a
    # token…): the page's scripts would run, and could act on it.
    if shell and not policy.url_carries_data(str(r.url)):
        sent = max(len(main.strip()), len(full.strip()))
        text = await _rendered(str(r.url), policy, sent=sent) or text
    return text[:limit] or "(no readable text)"


def _is_html(ctype: str, body: str) -> bool:
    if "html" in ctype.lower():
        return True
    return not ctype and body.lstrip()[:200].lower().startswith(("<!doctype html", "<html"))


def _main_text(html: str) -> str:
    import trafilatura

    return trafilatura.extract(html, include_links=False, include_tables=True, favor_recall=True) or ""


def _plain_text(body: str) -> str:
    import trafilatura

    return trafilatura.html2txt(body) or ""


def _texts(html: str) -> tuple[str, str]:
    """The page's main text, and all of its text (only worked out when the main text is short —
    that's when it's needed)."""
    import trafilatura

    main = _main_text(html)
    full = (trafilatura.html2txt(html) or "") if len(main) < 4000 else main
    return main, full


def _judge(html: str) -> tuple[str, str, bool]:
    """The page's main text, all of its text, and whether its scripts build it — all off NEO's
    loop (a hostile page can be slow to read)."""
    main, full = _texts(html)
    return main, full, render.needs_render(html, main, full)


async def _rendered(url: str, policy, *, sent: int) -> str:
    """The text of `url` once its scripts have built it, from a headless browser nobody sees —
    or "" when that's no better than the `sent` characters the server sent (it couldn't be
    rendered, it ended on an error page, or it shows no more): then the page as sent stands."""
    try:
        page = await render.render(url, policy.private_host)
    except render.RenderError as e:
        why = re.sub(r"[a-z][\w+.-]*://\S+", "<url>", str(e))  # the log stays content-free
        log.info(f"[web] a script-built page wasn't rendered: {why[:120]}")
        return ""
    except Exception as e:  # noqa: BLE001 — whatever went wrong, the page as sent stands
        log.info(f"[web] a script-built page wasn't rendered: {type(e).__name__}")
        return ""
    log.info(f"[web] rendered a script-built page in {page.seconds:.1f} s")
    if not 200 <= page.status < 400:  # a bot check, a server error…
        return ""
    main = (await asyncio.to_thread(_main_text, page.html)).strip() if page.html else ""
    shown = page.text.strip()
    if max(len(main), len(shown)) <= sent:  # a wall, a spinner: no more than the server sent
        return ""
    # The article extractor may find little in an app's markup (a feed, a thread): then what the
    # page shows.
    text = main if len(main) >= max(render.THIN_CHARS, len(shown) // 4) else shown
    there = _site(page.url)
    return f"Redirected to {there}:\n\n{text}" if there and there != _site(url) else text


def _site(url: str) -> str:
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""
    return host.removeprefix("www.")


@tool(
    "search_site",
    "Open a search for `query` on a website in the browser — google, youtube, amazon (the user's "
    "country store), flipkart, maps, wikipedia, github, reddit… or any domain. To PLAY music or "
    "video, use play_music instead. "
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
            # "search 2tb ssd on amazon", "look up cat videos on youtube", "find X in google maps".
            # One thing only: "open notes and search … on amazon" is two steps for the planner.
            rf"^\s*(?:please\s+)?(?:search(?:\s+for)?|look\s+up|google)\s+{_VAGUE_QUERY}(?!.*\b(?:and|then)\b)"
            rf"(?P<query>.+?)\s+(?:on|in|at)\s+(?P<site>{_SITE_RE})\s*[.!?]*\s*$",
            {"site": "<site>", "query": "<query>"},
            _legacy_fp,
        ),
        (
            # "open"/"find"/"show me" … on a named site (not a generic word like "maps" or "news")
            rf"^\s*(?:please\s+)?(?:open|find|show\s+me)\s+{_VAGUE_QUERY}(?!.*\b(?:and|then)\b)"
            rf"(?P<query>.+?)\s+(?:on|in|at)\s+(?P<site>{_ENGINE_RE})\s*[.!?]*\s*$",
            {"site": "<site>", "query": "<query>"},
            _legacy_fp,
        ),
        (
            # "search amazon for 2tb ssd", "search youtube for cat videos"
            rf"^\s*(?:please\s+)?(?:search|look\s+up|check)\s+(?P<site>{_SITE_RE})\s+for\s+{_VAGUE_QUERY_AFTER_FOR}(?P<query>.+?)\s*[.!?]*\s*$",
            {"site": "<site>", "query": "<query>"},
            _legacy_fp,
        ),
        (
            # "open youtube and search for cat videos"
            rf"^\s*(?:please\s+)?(?:open|go\s+to)\s+(?P<site>{_ENGINE_RE})\s+(?:and|then)\s+(?:search|look\s+up|find)(?:\s+for)?\s+"
            rf"{_VAGUE_QUERY_AFTER_FOR}(?P<query>.+?)\s*[.!?]*\s*$",
            {"site": "<site>", "query": "<query>"},
            _legacy_fp,
        ),
        # The same four shapes for any site NEO knows is a website (web_search on). The greedy
        # query makes the site the words after the last "on/in/at".
        (
            rf"^\s*(?:please\s+)?(?:search(?:\s+for)?|look\s+up|google)\s+{_VAGUE_QUERY}(?!.*\b(?:and|then)\b)"
            rf"(?P<query>.+)\s+(?:on|in|at)\s+{_SITE}\s*[.!?]*\s*$",
            {"site": "<site>", "query": "<query>"},
            _claims_site,
        ),
        (
            rf"^\s*(?:please\s+)?(?:open|find|show\s+me)\s+{_VAGUE_QUERY}(?!.*\b(?:and|then)\b)"
            rf"(?P<query>.+)\s+(?:on|in|at)\s+{_SITE}\s*[.!?]*\s*$",
            {"site": "<site>", "query": "<query>"},
            _claims_site,
        ),
        (
            rf"^\s*(?:please\s+)?(?:search|look\s+up|check)\s+{_SITE}\s+for\s+{_VAGUE_QUERY_AFTER_FOR}"
            r"(?P<query>.+?)\s*[.!?]*\s*$",
            {"site": "<site>", "query": "<query>"},
            _claims_site,
        ),
        (
            rf"^\s*(?:please\s+)?(?:open|go\s+to)\s+{_SITE}\s+(?:and|then)\s+(?:search|look\s+up|find)(?:\s+for)?\s+"
            rf"{_VAGUE_QUERY_AFTER_FOR}(?P<query>.+?)\s*[.!?]*\s*$",
            {"site": "<site>", "query": "<query>"},
            _claims_site,
        ),
    ],
    idempotent=True,
    requires=frozenset({"macos"}),  # opens the Mac's own browser
)
async def search_site(a: dict, c: ToolContext):
    from neo.tools.computer import apps

    site, query = str(a["site"]).strip(), str(a["query"]).strip()
    q = " ".join(query.lower().split()).removeprefix("for ").strip(" .!?")
    if not q or q in ("it", "this", "that", "them", "those", "these", "one", "the same") or q.endswith(" one"):
        return "Error: search for what? Name the thing to search for."
    query = query.strip()[4:].strip() if query.lower().startswith("for ") else query
    m = decide_mode("web_search")
    if m == "on":
        from neo.agent import confirm
        from neo.decide import run
        from neo.decide.kinds import site as sk

        if c.tainted() and sk.carries_data(site, query):
            gate = confirm.gated(
                "search_site",
                a,
                f"search {site} for something that carries data (after reading {c.tainted()})",
                detail=f"{site}: {query}",
            )
            if isinstance(gate, str):
                return gate
        return run.to_output(await sk.search(site, query, browser=str(a.get("browser") or "")))
    url = site_search_url(site, query)
    out = await apps.open_app(url)
    if m == "shadow":
        from neo.decide import shadow

        with shadow.starting("web_search"):
            from neo.decide.kinds import site as sk

            ok = not out.startswith(("Error", "NEEDS_"))
            shadow.later("web_search", sk.shadow(site, url), legacy_ok=ok)
    return out if out.startswith(("Error", "NEEDS_")) else f"Searching {site} for “{query}”."
