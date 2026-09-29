"""Naming a song from a few seconds of audio: Shazam's fingerprint of it, one request to Shazam.

shazamio (MIT) does the Shazam part. shazamio_core (Rust) turns a WAV built here in memory into
Shazam's signature: a fingerprint of the sound's strongest frequencies, from which the audio can't be
played back. shazamio's ``Shazam.recognize`` sends that signature (with its length and a timestamp,
nothing else) to Shazam's recognition endpoint. The audio itself never leaves the Mac, never touches
the disk and is never logged: no temporary file, and always a well-formed WAV, which the Rust decoder
reads itself (anything it can't read, shazamio_core would hand to an ``ffmpeg`` program).

The request goes through NEO's own client, not shazamio's (whose retries go on for up to 20 attempts):
one attempt, at most ``REQUEST_S``, only to Shazam, no proxy. The answer becomes the card's result:
title, artist, album, year, genre and links (https only: the song on Apple Music, and on Shazam). The
cover art is fetched here, not by the UI: https from Shazam's image hosts only, at most
``COVER_MAX_BYTES``, made at most ``COVER_PX`` wide and handed over as a ``data:image/jpeg`` URL of at
most ``COVER_DATA_MAX`` characters.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import importlib.util
import io
import math
import re
import threading
import warnings
import wave
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import httpx
import numpy as np

from neo2.core import logger

log = logger(__name__)

RATE = 16000
MAX_SIGNATURE_S = 12  # Shazam's signatures cover at most this much audio
REQUEST_S = 6.0  # one recognition request, all of it
COVER_S = 3.0  # fetching the cover art, all of it
COVER_PX = 400
COVER_MAX_BYTES = 1_500_000  # the downloaded picture
COVER_DATA_MAX = 300_000  # the data URL the card gets
_MAX_PIXELS = 25_000_000  # a picture bigger than this is no cover (a decompression bomb)
_SHAZAM = ("shazam.com",)  # where the recognition request goes, and the song's Shazam page
_IMAGE_HOSTS = ("mzstatic.com", "shazam.com")  # Shazam's cover art (Apple's image CDN serves most)
_APPLE_MUSIC = ("music.apple.com",)
_INSTALL = "Song recognition isn't installed: install NEO's music extra (shazamio)."

# Tests: an httpx transport in place of the network (httpx.MockTransport).
transport: httpx.AsyncBaseTransport | None = None


class Failed(Exception):
    """No answer from Shazam (offline, a timeout, an error status, not JSON)."""


def available() -> str:
    """ "" when shazamio is installed, else what to install (for the user)."""
    return (
        "" if importlib.util.find_spec("shazamio") and importlib.util.find_spec("shazamio_core") else _INSTALL
    )


def _shazamio() -> Any:
    """shazamio, imported quietly (pydub, under it, warns about ffmpeg, which NEO never uses)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import shazamio
        import shazamio_core  # noqa: F401
    return shazamio


def warm_soon() -> None:
    """Import shazamio in the background (~0.3 s) while NEO starts listening."""
    if available():
        return

    def warm() -> None:
        with contextlib.suppress(Exception):
            _shazamio()

    threading.Thread(target=warm, name="neo-song-warm", daemon=True).start()


def wav(pcm: np.ndarray, rate: int = RATE) -> bytes:
    """16-bit mono PCM as a WAV file's bytes, in memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.asarray(pcm, dtype="<i2").tobytes())
    return buf.getvalue()


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=transport, trust_env=False, follow_redirects=False)


def _https(url: Any, hosts: tuple[str, ...]) -> str | None:
    """``url`` when it is https on one of ``hosts`` (or their subdomains), else None."""
    if not isinstance(url, str) or len(url) > 2048 or any(c.isspace() for c in url):
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    host = (parts.hostname or "").lower()
    if parts.scheme != "https" or parts.username or parts.password or parts.port not in (None, 443):
        return None
    return url if any(host == h or host.endswith("." + h) for h in hosts) else None


def safe_link(url: Any) -> str | None:
    """A result link the card may open: https to Apple Music or Shazam."""
    return _https(url, _APPLE_MUSIC) or _https(url, _SHAZAM)


class _OneShot:
    """shazamio's HTTP client interface, as one bounded request (see the module doc)."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def request(self, method: str, url: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if method.upper() != "POST" or not _https(url, _SHAZAM):
            raise Failed("not the recognition request")  # (shazamio asks nothing else of it here)
        try:
            r = await self.client.post(
                url, json=kwargs.get("json"), headers=kwargs.get("headers"), timeout=REQUEST_S
            )
        except httpx.HTTPError as e:
            raise Failed(type(e).__name__) from None
        if r.status_code >= 400:
            raise Failed(f"HTTP {r.status_code}")
        try:
            data = r.json()
        except ValueError:
            raise Failed("the answer wasn't JSON") from None
        if not isinstance(data, dict):
            raise Failed("an answer of the wrong shape")
        return data


def _text(v: Any, limit: int = 200) -> str:
    return (
        " ".join(str(v).split())[:limit]
        if isinstance(v, str | int | float) and not isinstance(v, bool)
        else ""
    )


@dataclass
class Song:
    title: str
    artist: str
    album: str | None = None
    year: str | None = None
    genre: str | None = None
    links: dict[str, str | None] = field(default_factory=lambda: {"apple_music": None, "shazam": None})
    cover_url: str | None = None  # where the cover art is (fetched by :func:`cover`, never by the UI)

    def result(self, cover: str | None = None) -> dict[str, Any]:
        """The card's result (the UI contract)."""
        return {
            "title": self.title, "artist": self.artist, "album": self.album, "year": self.year,
            "genre": self.genre, "cover": cover, "links": dict(self.links),
        }  # fmt: skip

    def sentence(self) -> str:
        """What NEO says: "That's Blinding Lights by The Weeknd, from After Hours, 2020." """
        said = f"That's {self.title}" + (f" by {self.artist}" if self.artist else "")
        album = re.sub(r"\s+-\s+(?:Single|EP)$", "", self.album or "").strip()
        if album and album.casefold() != self.title.casefold():
            said += f", from {album}"
        if self.year:
            said += f", {self.year}"
        return said + "."


def _apple_music(track: dict) -> str | None:
    """The song on Apple Music: the "open" action of Shazam's hub (its first https link there)."""
    hub = track.get("hub") if isinstance(track.get("hub"), dict) else {}
    actions: list[Any] = []
    for opt in hub.get("options") or []:
        if isinstance(opt, dict):
            actions += opt.get("actions") or []
    actions += hub.get("actions") or []
    urls = [a.get("uri") for a in actions if isinstance(a, dict)]
    ranked = [
        a.get("uri") for a in actions if isinstance(a, dict) and a.get("type") == "applemusicopen"
    ] + urls
    return next((u for u in ranked if _https(u, _APPLE_MUSIC)), None)


def parse(resp: Any) -> Song | None:
    """Shazam's answer as a :class:`Song` (None: no match)."""
    if not isinstance(resp, dict) or not resp.get("matches"):
        return None
    track = resp.get("track")
    if not isinstance(track, dict):
        return None
    title, artist = _text(track.get("title")), _text(track.get("subtitle"))
    if not title:
        return None
    meta: dict[str, str] = {}
    for section in track.get("sections") or []:
        if isinstance(section, dict) and section.get("type") == "SONG":
            for m in section.get("metadata") or []:
                if isinstance(m, dict) and (k := _text(m.get("title")).lower()):
                    meta.setdefault(k, _text(m.get("text")))
    year = re.search(r"\b(1[89]\d\d|2\d{3})\b", meta.get("released", ""))
    genres = track.get("genres") if isinstance(track.get("genres"), dict) else {}
    images = track.get("images") if isinstance(track.get("images"), dict) else {}
    share = track.get("share") if isinstance(track.get("share"), dict) else {}
    cover = next(
        (
            u
            for u in (images.get("coverart"), images.get("coverarthq"), share.get("image"))
            if _https(u, _IMAGE_HOSTS)
        ),
        None,
    )
    return Song(
        title=title,
        artist=artist,
        album=meta.get("album") or None,
        year=year.group(1) if year else None,
        genre=_text(genres.get("primary"), 80) or None,
        links={
            "apple_music": _apple_music(track),
            "shazam": _https(track.get("url"), _SHAZAM) or _https(share.get("href"), _SHAZAM),
        },
        cover_url=cover,
    )


async def identify(pcm: np.ndarray) -> Song | None:
    """The song in these 16 kHz mono samples (at most the latest ``MAX_SIGNATURE_S`` count), None
    when Shazam knows none. Raises :class:`Failed` when Shazam couldn't be asked."""
    pcm = np.asarray(pcm, dtype=np.int16)[-MAX_SIGNATURE_S * RATE :]
    if pcm.size < RATE:
        return None  # under a second: nothing Shazam could match
    shazamio = _shazamio()
    from shazamio_core import SearchParams

    seconds = min(MAX_SIGNATURE_S, math.ceil(pcm.size / RATE))  # all of it: the latest audio is the song now
    async with _client() as client:
        shazam = shazamio.Shazam(http_client=_OneShot(client), segment_duration_seconds=seconds)
        try:
            resp = await asyncio.wait_for(
                shazam.recognize(wav(pcm), options=SearchParams(segment_duration_seconds=seconds)),
                REQUEST_S + 1.0,
            )
        except Failed:
            raise
        except TimeoutError:
            raise Failed("timed out") from None
        except Exception as e:  # noqa: BLE001 — the signature, or shazamio itself: never the user's problem to read
            raise Failed(type(e).__name__) from None
    if not isinstance(resp.get("matches"), list):  # (no match is {"matches": [], "tagid", "retryms"})
        raise Failed("an answer without matches")
    return parse(resp)


async def cover(url: str) -> str | None:
    """The cover art at ``url`` as a data URL (see the module doc for its bounds), None when it isn't
    one of Shazam's pictures or can't be had in time."""
    if not _https(url, _IMAGE_HOSTS):
        return None
    try:
        async with _client() as client:
            raw = await asyncio.wait_for(_download(client, url), COVER_S)
    except (TimeoutError, httpx.HTTPError):
        return None
    if raw is None:
        return None
    return await asyncio.to_thread(data_url, raw)


async def _download(client: httpx.AsyncClient, url: str) -> bytes | None:
    async with client.stream("GET", url, timeout=COVER_S) as r:
        if r.status_code != 200 or not r.headers.get("content-type", "").lower().startswith("image/"):
            return None
        with contextlib.suppress(ValueError):
            if int(r.headers.get("content-length") or 0) > COVER_MAX_BYTES:
                return None
        buf = bytearray()
        async for chunk in r.aiter_bytes():
            buf += chunk
            if len(buf) > COVER_MAX_BYTES:
                return None
    return bytes(buf)


def data_url(raw: bytes) -> str | None:
    """A picture's bytes as a JPEG data URL at most ``COVER_PX`` wide and ``COVER_DATA_MAX`` long."""
    from PIL import Image

    try:
        with Image.open(io.BytesIO(raw)) as img:
            if img.width * img.height > _MAX_PIXELS:
                return None
            img.draft("RGB", (COVER_PX, COVER_PX))  # (a JPEG decodes smaller straight away)
            pic = img.convert("RGB")
        pic.thumbnail((COVER_PX, COVER_PX))
        for quality in (85, 75, 65, 55):
            out = io.BytesIO()
            pic.save(out, "JPEG", quality=quality, optimize=True)
            url = "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode()
            if len(url) <= COVER_DATA_MAX:
                return url
    except Exception:  # noqa: BLE001 — not a picture PIL can read
        return None
    return None
