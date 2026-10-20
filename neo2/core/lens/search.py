"""Searching beyond the Mac for a Search Screen area: "Search with Google" (or the site the user set
in NEO_LENS_SEARCH), "Search “<the answer's query>”", and the search rows for a product or a place.

Image search, on the user's click only. Google documents no URL to upload a picture to, and its
"search by URL" needs the picture on a public server (never). What its own browser does — the
New Tab Page's Lens button, and SumatraPDF's "Search image on Google" after it — is a form POST of
the picture to lens.google.com. NEO does exactly that, in the user's own browser:

1. the area is made smaller (at most 1000 px on its long side, white behind any transparency,
   JPEG q85: the size Chromium sends, text still legible);
2. a page of NEO's own is written to $TMPDIR/neo-lens/<random>.html (the folder this user's alone,
   the file too): no outside resource, no referrer, a content policy that lets it post only to the
   provider, a hidden form and six lines of script that put the picture in it and submit it;
3. it is opened in the browser that opens https links on this Mac (the system's own answer: never
   a browser NEO picked by name, never whatever opens .html files);
4. the file is deleted 90 seconds later, and any left over is swept when the core starts.

Only the provider in NEO_LENS_SEARCH gets the picture: "google" (Google Lens, the default), "bing"
(Bing's visual search). Any other site word ("duckduckgo", a domain) gets a text search for the
answer's query instead, and "off" offers neither. Google is named in words only — no logo.

Text search goes through neo2.core.tools.web.open_search (search_site's own way to the site: the
site and the browser from evidence where web_search is on). The links for a product or a place are
search results for the answer's query (web._race), never links a model wrote; the page is shown
each row's title and full host, and opens one only by its id.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import html
import io
import os
import secrets
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from PIL import Image

from neo2.core import logger

log = logger(__name__)

MAX_EDGE = 1000
JPEG_QUALITY = 85
KEEP_S = 90.0
LINKS_FOR = {"product": 5, "place": 3, "plant": 3, "animal": 3, "food": 3, "art": 3}
_SEARCH_N = 8


@dataclass(frozen=True)
class Recipe:
    """How a provider takes a picture: the form's action, its file field, and whether the picture
    goes as a file (multipart) or as base64 text in a field."""

    word: str
    label: str
    action: str
    field: str
    as_file: bool
    host: str


RECIPES: dict[str, Recipe] = {
    "google": Recipe("google", "Google", "https://lens.google.com/v3/upload?hl={hl}&st={st}", "encoded_image", True, "lens.google.com"),
    "bing": Recipe("bing", "Bing", "https://www.bing.com/images/search?view=detailv2&iss=sbiupload", "imageBin", False, "www.bing.com"),
}  # fmt: skip
_NAMES = {"duckduckgo": "DuckDuckGo", "ddg": "DuckDuckGo", "yandex": "Yandex", "brave": "Brave", "ecosia": "Ecosia",
          "kagi": "Kagi", "baidu": "Baidu", "yahoo": "Yahoo"}  # fmt: skip


def folder() -> Path:
    return Path(tempfile.gettempdir()) / "neo-lens"


def _word() -> str:
    try:
        from neo2.core.config import settings

        return str(getattr(settings(), "lens_search", "google") or "google")
    except Exception:  # noqa: BLE001
        return "google"


def provider(word: str | None = None) -> dict[str, Any] | None:
    """What the page offers: {word, label, image, text} — None when NEO_LENS_SEARCH is off."""
    w = (word or _word()).lower()
    if w == "off":
        return None
    r = RECIPES.get(w)
    if r is not None:
        return {"word": w, "label": r.label, "image": True, "text": True}
    name = _NAMES.get(w) or (w.split(".")[0].capitalize() if w else "the web")
    return {"word": w, "label": name, "image": False, "text": True}


def shrink(img: Image.Image) -> bytes:
    """The area as the provider gets it: at most MAX_EDGE px long, white behind transparency, JPEG."""
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.getchannel("A"))
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    if max(img.size) > MAX_EDGE:
        k = MAX_EDGE / max(img.size)
        img = img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=JPEG_QUALITY, subsampling=0)
    return buf.getvalue()


def page(jpeg: bytes, r: Recipe, *, hl: str = "en", st: int | None = None) -> str:
    """The self-submitting page for ``r`` (see the module's doc). Nothing on it loads from anywhere."""
    st = int(time.time() * 1000) if st is None else st
    action = html.escape(r.action.format(hl="".join(c for c in hl if c.isalnum() or c == "-")[:12] or "en", st=st))
    b64 = base64.b64encode(jpeg).decode("ascii")
    if r.as_file:
        field = f'<input type="file" name="{r.field}" id="f" hidden>'
        script = (
            f'const b=atob("{b64}"),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);'
            'const d=new DataTransfer();d.items.add(new File([u],"screen.jpg",{type:"image/jpeg"}));'
            'document.getElementById("f").files=d.files;document.getElementById("s").submit();'
        )
    else:
        field = f'<input type="hidden" name="{r.field}" value="{b64}">'
        script = 'document.getElementById("s").submit();'
    csp = f"default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; form-action https://{r.host}"
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        '<meta name="referrer" content="no-referrer">'
        f'<meta http-equiv="Content-Security-Policy" content="{csp}">'
        f"<title>Opening {html.escape(r.label)}…</title>"
        "<style>body{font:15px -apple-system,system-ui,sans-serif;color:#555;display:grid;place-items:center;height:90vh}</style>"
        f"</head><body><p>Opening {html.escape(r.label)}…</p>"
        f'<form id="s" method="post" enctype="multipart/form-data" action="{action}">{field}</form>'
        f"<script>{script}</script></body></html>"
    )


def _write(text: str) -> Path:
    d = folder()
    d.mkdir(mode=0o700, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(d, 0o700)
    p = d / f"{secrets.token_hex(8)}.html"
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    return p


def _delete_later(p: Path) -> None:
    def gone() -> None:
        with contextlib.suppress(OSError):
            p.unlink()

    try:
        asyncio.get_running_loop().call_later(KEEP_S, gone)
    except RuntimeError:  # (no loop: nothing to wait on; the next start sweeps it)
        pass


def sweep() -> int:
    """Delete any page an earlier run left in the folder (the core is starting). Returns how many."""
    d = folder()
    n = 0
    if not d.is_dir():
        return 0
    for p in d.glob("*.html"):
        with contextlib.suppress(OSError):
            p.unlink()
            n += 1
    return n


def _browser() -> str:
    """The bundle id of the app that opens https links ("" when there is none)."""
    try:
        from neo2.core.decide import workspace

        return workspace.default_handler("https") or ""
    except Exception:  # noqa: BLE001
        return ""


def _hl() -> str:
    from neo2.core.lens import local

    pref = local.preferred()
    return local.base(pref[0]).split("-")[0] if pref else "en"


async def image_search(img: Image.Image, *, word: str | None = None) -> str:
    """The area uploaded to the provider in the user's browser: "opened", or why not ("off",
    "no_image" — a provider that takes text only, "no_browser", "failed")."""
    w = (word or _word()).lower()
    if w == "off":
        return "off"
    r = RECIPES.get(w)
    if r is None:
        return "no_image"
    browser = await asyncio.to_thread(_browser)
    if not browser:
        return "no_browser"
    jpeg = await asyncio.to_thread(shrink, img)
    p = await asyncio.to_thread(_write, page(jpeg, r, hl=_hl()))
    _delete_later(p)
    from neo2.core.tools.computer import apps

    try:
        out = await apps.open_url(p.as_uri(), browser_id=browser)
    except Exception as e:  # noqa: BLE001
        log.info(f"[lens] image search didn't open ({type(e).__name__})")
        return "failed"
    ok = not str(out).startswith("Error")
    log.info(f"[lens] image search with {r.label}: {'opened' if ok else 'failed'} ({len(jpeg) // 1024} KB)")
    return "opened" if ok else "failed"


async def text_search(query: str, *, word: str | None = None) -> str:
    """A search for ``query`` on the provider's site: "opened", or why not ("off", "failed")."""
    w = (word or _word()).lower()
    q = " ".join(str(query or "").split())[:120]
    if w == "off" or not q:
        return "off" if w == "off" else "failed"
    from neo2.core.tools import web

    try:
        out = await web.open_search(w, q)
    except Exception as e:  # noqa: BLE001
        log.info(f"[lens] text search didn't open ({type(e).__name__})")
        return "failed"
    text = getattr(out, "text", out)
    ok = getattr(out, "ok", True) and not str(text).startswith(("Error", "NEEDS_"))
    return "opened" if ok else "failed"


def wanted(kind: str, shop: bool) -> int:
    """How many search rows an answer of this kind gets (0: none — text, math, code…)."""
    if shop:
        return LINKS_FOR["product"]
    return LINKS_FOR.get(kind, 0)


def links(query: str, n: int) -> list[dict[str, str]]:
    """Up to ``n`` search rows for ``query``, one per host: {title, host, href}. [] offline or when
    the web extra isn't there."""
    q = " ".join(str(query or "").split())[:100]
    if not q or n <= 0:
        return []
    from neo2.core.tools import web

    if not web._has_ddgs():
        return []
    try:
        rows = web._race(q, _SEARCH_N)
    except Exception as e:  # noqa: BLE001 — no rows: the answer stands alone
        log.info(f"[lens] no search rows ({type(e).__name__})")
        return []
    out: list[dict[str, str]] = []
    hosts: set[str] = set()
    for r in rows or []:
        href = str(r.get("href") or r.get("url") or "")
        try:
            u = urlsplit(href)
        except ValueError:
            continue
        host = (u.hostname or "").lower()
        if u.scheme not in ("http", "https") or not host or host in hosts or len(href) > 2048:
            continue
        hosts.add(host)
        title = " ".join(str(r.get("title") or host).split())[:90]
        out.append({"title": title, "host": host, "href": href})
        if len(out) >= n:
            break
    return out


__all__ = ["RECIPES", "folder", "image_search", "links", "page", "provider", "shrink", "sweep", "text_search", "wanted"]
