"""Web search (DuckDuckGo, no key) + page fetch/extract (trafilatura). Free and fast."""

from __future__ import annotations

import asyncio

import httpx

from neo.agent.registry import ToolContext, tool

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
            r"(?!.*\b(?:in|on)\s+(?:notes|spotify|finder|mail|messages|slack|my\s+\w+|the\s+\w+\s+app)\b)"
            r"(?P<query>.{3,}?)\s*[?.!]*\s*$",
            {"query": "<query>"},
        )
    ],
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
