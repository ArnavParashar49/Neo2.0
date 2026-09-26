"""write_note: what the user said, saved as a note in the notes app this user uses.

The engine picks where (``choose("write_note")``: the app the user named, else the apps they use
that can make notes, else a file) and :func:`neo.decide.run.run_best` tries them in order. Per
app, best first:

* ``script_make`` - its own dictionary makes the note (in the folder the user named, else the
  app's default one) and reads it back through the id the app gave it. A make that timed out is
  looked for, never repeated: found is done, not found is "have a look there";
* ``ax_new`` - its New command (found by the ⌘N shortcut, not by a title) and the new, empty
  editor that takes the focus, filled and read back;
* ``file`` (the fallback) - a Markdown file in ``settings().notes_dir``, never over another file.

Nothing here keeps a note's text: the only memory is the engine's record of which way worked.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import re
import sys
import time
import unicodedata
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from neo.config import settings
from neo.decide import aescript, axkit, engine, run
from neo.decide.run import Effect, Outcome, Req, Result

if TYPE_CHECKING:
    from neo.decide.engine import Candidate
    from neo.decide.inventory import App

__all__ = [
    "KIND",
    "write",
    "preview",
    "methods_for",
    "accepts",
    "describe",
    "attempt",
    "fallback",
    "doc_like",
    "top_app",
]

KIND = "write_note"
OVERALL_S = 30.0  # every app and the file, gates' questions excluded
BUDGET_S = {"script_make": 17.0, "ax_new": 10.0, "file": 3.0}
LIST_S, MAKE_S, SEARCH_S = 4.0, 8.0, 4.0  # folders listed; made and read back; looked for again
OPEN_S, NEW_S = 5.0, 3.0  # the app showing a window; its new editor taking the focus
CHECK = 40  # this many characters of the body must read back
TITLE_WORDS = 6  # a note without a title is named by its first words


# ---- the request ------------------------------------------------------------------------------------


def _one_line(s: object) -> str:
    return " ".join(str(s or "").split())


def _req(fields: dict, said: str) -> Req:
    """The note as the tool got it. What the engine and Laya read is the user's command without
    the note itself (an app's name inside a note is not where to put it)."""
    title, folder, named = (_one_line(fields.get(k)) for k in ("title", "folder", "app"))
    body = str(fields.get("body") or "").strip()
    text = _one_line(said)
    for part in (body, title):
        if part and part in text:
            head, _sep, tail = text.rpartition(part)
            text = _one_line(f"{head} {tail}")
    text = text or "save a note"
    if named and named.casefold() not in text.casefold():
        text = f"{text} in {named}"
    digest = hashlib.sha256("\0".join((title, body, folder, named)).encode()).hexdigest()[:16]
    return Req(
        KIND, {"title": title, "body": body, "folder": folder}, text, named=named, sig=("note", digest)
    )


def _parts(req: Req) -> tuple[str, str, bool]:
    """(title, body, whether the user gave the title). A note without a body is its title."""
    title, body = req.fields.get("title", ""), req.fields.get("body", "")
    if not body:
        return title, title, False
    return title or " ".join(body.split()[:TITLE_WORDS]), body, bool(title)


def _norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFC", s or "").split()).casefold()


def _holds(r: aescript.NoteRoute, name: str, text: str, title: str, body: str) -> bool:
    """The note read back holds what was written: the start of the body (or the title)."""
    if r.html and r.text == r.body:  # read back as HTML: compare its text
        text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    want = _norm(body)[:CHECK] or _norm(title)[:CHECK]
    return bool(want) and want in _norm(f"{name} {text}")


def _pad(row: list[str], n: int) -> list[str]:
    return (list(row) + [""] * n)[:n]


# ---- the driver --------------------------------------------------------------------------------------


def methods_for(app: App, kind: str | None = None) -> tuple[str, ...]:
    """Its dictionary's make when it has one; its New command for any app with a window, except a
    browser or a chat app (a line typed there could be sent)."""
    out = []
    if aescript.note_route(app.script) is not None:
        out.append("script_make")
    chat = engine.fits_kind("send_message", app)
    if app.path and app.kind == "app" and not engine._is_browser(app) and not chat:
        out.append("ax_new")
    return tuple(out)


def accepts(app: App, method: str, req: Req) -> bool:
    return method in methods_for(app)


def describe(app: App) -> str:
    how = {"script_make": "makes notes by script", "ax_new": "makes notes through its window"}
    return "; ".join(how[m] for m in methods_for(app))


async def attempt(req: Req, cand: Candidate, method: str) -> Outcome:
    app = cand.app
    if app is None:
        return Outcome(False, "no app to write in", reason="route_bad")
    if method == "script_make":
        return await _script_make(req, app)
    if method == "ax_new":
        return await _ax_new(req, app)
    return Outcome(False, f"{app.name} can't make notes that way", reason="route_bad")


async def fallback(req: Req, key: str) -> Outcome | None:
    return await asyncio.to_thread(_to_file, req) if key == "file" else None


# ---- script_make ----------------------------------------------------------------------------------------


def _unsure(app: App, what: str = "") -> Outcome:
    """Made, or maybe made: never a second note. The user is told where to look."""
    text = what or f"I may have saved it in {app.name} but couldn't check"
    return Outcome(
        False, f"{text}. Have a look there.", reason="unconfirmed", effect=Effect.UNKNOWN, say=True
    )


def _saved(app: App, req: Req, oid: str, where: str, folder_id: str) -> Outcome:
    folder = req.fields.get("folder", "")
    receipt = {"id": oid, "where": app.name, "container": where, "verified": True}
    if folder and not folder_id:
        into = f"“{where}”" if where else "its default folder"
        text = f"There's no folder called “{folder}” in {app.name}, so I put it in {into}."
        return Outcome(True, text, effect=Effect.DONE, receipt=receipt, say=True)
    text = f"Saved in {app.name} (“{where}”)." if folder and where else f"Saved in {app.name}."
    return Outcome(True, text, effect=Effect.DONE, receipt=receipt)


async def _pick_folder(named: str, options: list[tuple[str, str]], said: str) -> str:
    """The id of the folder the user named: the same name, else the one folder holding every word
    of it; between several such, only Laya's sure choice. '' when none."""
    want = _norm(named)
    if hit := next((fid for name, fid in options if _norm(name) == want), ""):
        return hit
    words = set(want.split())
    some = {name: fid for name, fid in options if words and words <= set(_norm(name).split())}
    if len(some) <= 1:
        return next(iter(some.values()), "")
    q = {
        "folder": {
            "type": "choice",
            "instructions": f"Which folder did the user mean by “{named}”?",
            "criteria": {name: f"the folder called {name}" for name in some},
        }
    }
    probs = engine._choice(await engine._ask_laya({"request": said}, q), "folder")
    if not engine._sure(probs):
        return ""
    return some.get(max(probs, key=lambda k: probs[k]), "")


async def _folder_id(req: Req, app: App, r: aescript.NoteRoute, folder: str) -> str:
    rendered = aescript.render_folders(r, app.bundle_id)
    if rendered is None:
        return ""
    code, out, _err = await aescript.run(*rendered, LIST_S)
    if code != 0:
        return ""  # the make that follows says why, if it's anything but this listing
    rows = [_pad(row, 2) for row in aescript.rows(out)[1]]
    return await _pick_folder(folder, [(n, i) for n, i in rows if n and i], req.text)


async def _search_back(req: Req, app: App, r: aescript.NoteRoute, since: datetime, folder_id: str) -> Outcome:
    """A make that may have happened: the note it would have made, found again, is done."""
    title, body, _titled = _parts(req)
    rendered = aescript.render_note_search(r, app.bundle_id, since, title, folder_id)
    if rendered is not None:
        code, out, _err = await aescript.run(*rendered, SEARCH_S)
        for row in aescript.rows(out)[1] if code == 0 else ():
            oid, name, text = _pad(row, 3)
            if _holds(r, name, text, title, body):
                return _saved(app, req, oid, "", folder_id)
    return _unsure(app)


async def _script_make(req: Req, app: App) -> Outcome:
    r = aescript.note_route(app.script)
    if r is None:
        return Outcome(False, f"{app.name} can't make notes by script", reason="route_bad")
    title, body, _titled = _parts(req)
    folder = req.fields.get("folder", "")
    folder_id = await _folder_id(req, app, r, folder) if folder else ""
    since = datetime.now() - timedelta(seconds=2)
    script, argv = aescript.render_note(r, app.bundle_id, title, body, folder_id)
    code, out, err = await run.commit(aescript.run(script, argv, MAKE_S))
    if code != 0:
        failed = aescript.fail(app.name, code, err, committed=True)
        return (
            await _search_back(req, app, r, since, folder_id) if failed.effect is Effect.UNKNOWN else failed
        )
    status, rows = aescript.rows(out)
    oid, name, text, where = _pad(rows[0] if rows else [], 4)
    if status != "OK" or not oid:  # made, but not read back: look for it rather than make another
        return await _search_back(req, app, r, since, folder_id)
    if not _holds(r, name, text, title, body):
        return _unsure(app, f"I saved it in {app.name} but couldn't check what it says")
    return _saved(app, req, oid, where, folder_id)


# ---- ax_new ----------------------------------------------------------------------------------------------


async def _new_editor(pid: int):
    """The empty text input that took the focus after the New command, within NEW_S."""
    deadline = time.monotonic() + NEW_S
    while True:
        el = await asyncio.to_thread(axkit.focused_editable, pid)
        if el is not None and await asyncio.to_thread(axkit.is_empty, el):
            return el
        if time.monotonic() >= deadline:
            return None
        await asyncio.sleep(0.15)


async def _ax_new(req: Req, app: App) -> Outcome:
    pid = await axkit.window_pid(app, OPEN_S)
    if pid is None:
        return Outcome(False, f"{app.name} didn't open")
    item = await asyncio.to_thread(axkit.menu_item, pid, "n", 0)
    if item is None:
        return Outcome(False, f"{app.name} has no New command", reason="route_bad")
    if not await asyncio.to_thread(axkit.perform, item[0], "AXPress"):
        return Outcome(False, f"{app.name}'s New command didn't respond")
    el = await _new_editor(pid)
    if el is None:  # whatever it made is empty: harmless, and said only if nothing else works
        return Outcome(False, f"{app.name} didn't give me an empty note to write in", effect=Effect.PARTIAL)
    title, body, titled = _parts(req)
    text = f"{title}\n{body}" if titled and not _norm(body).startswith(_norm(title)) else body
    if "\n" in text and not axkit.multiline(el):
        return Outcome(False, f"{app.name}'s new note has a one-line field", effect=Effect.PARTIAL)
    # a line break goes in as shift+return: never a key that could send or submit something
    if await run.commit(asyncio.to_thread(axkit.set_text, el, text, newline="shift+return")):
        receipt = {"where": app.name, "verified": True}
        return Outcome(True, f"Wrote it in {app.name}.", effect=Effect.DONE, receipt=receipt)
    if await asyncio.to_thread(axkit.is_empty, el):
        return Outcome(False, f"{app.name} didn't take the text", effect=Effect.PARTIAL)
    return _unsure(app, f"I may have written it in {app.name} but couldn't check")


# ---- the file fallback --------------------------------------------------------------------------------


def _slug(title: str) -> str:
    """A file name from the title: no path separators, control or reserved characters, no
    leading dot (a hidden file), at most 60 characters."""
    s = re.sub(r"[\x00-\x1f\x7f/\\:*?\"<>|]+", " ", unicodedata.normalize("NFC", title))
    return " ".join(s.split())[:60].strip(" .") or "Note"


def _shown(folder: Path) -> str:
    """Where the file went, as the user would find it ("Documents › NEO Notes")."""
    try:
        return " › ".join(folder.relative_to(Path.home()).parts) or str(folder)
    except ValueError:
        return str(folder)


def _to_file(req: Req) -> Outcome:
    from neo.tools.computer import policy

    title, body, titled = _parts(req)
    folder = settings().notes_dir
    stem = f"{date.today().isoformat()} {_slug(title)}"
    content = f"# {title}\n\n{body}\n" if titled and body != title else f"{body}\n"
    path = None
    for n in range(1, 100):
        p = folder / (f"{stem}.md" if n == 1 else f"{stem}-{n}.md")
        if p.exists():
            continue
        if why := policy.write_risk(p):
            return Outcome(False, f"I couldn't save it as a file: {why}", reason="permission")
        try:
            folder.mkdir(parents=True, exist_ok=True)
            with open(p, "x", encoding="utf-8") as f:  # "x": never over a file that appeared meanwhile
                f.write(content)
        except FileExistsError:
            continue
        except OSError as e:
            return Outcome(False, f"I couldn't save it as a file ({e.strerror or e})")
        path = p
        break
    if path is None:
        return Outcome(False, "There are too many notes with that name today")
    try:
        back = path.read_text(encoding="utf-8")
    except OSError:
        back = None
    if back != content:
        return _unsure_file(folder)
    where = _shown(folder)
    receipt = {"where": "file", "path": str(path), "verified": True}
    return Outcome(True, f"Saved it as a file in {where}.", effect=Effect.DONE, receipt=receipt)


def _unsure_file(folder: Path) -> Outcome:
    text = f"I saved a file in {_shown(folder)} but couldn't read it back. Have a look there."
    return Outcome(False, text, reason="unconfirmed", effect=Effect.UNKNOWN, say=True)


# ---- entry points --------------------------------------------------------------------------------------


def _installed(named: str) -> bool | None:
    """Whether an app goes by that name here (None before the first app scan)."""
    from neo.decide import inventory

    apps = inventory.snapshot()
    return None if apps is None else inventory.named_in(named, apps) is not None


async def write(fields: dict, said: str = "") -> Result:
    """Save the note (``notes_create``'s arguments: body, title, folder, app); ``said`` is the
    user's own words for the request, when known."""
    req = _req(fields, said)
    if not req.fields["body"] and not req.fields["title"]:
        return Result("Error: there's nothing to write in the note.", False)
    got = await run.run_best(sys.modules[__name__], req)
    if req.named and got.ok and _installed(req.named) is False:
        return Result(f"{req.named} isn't on this Mac. {got.text}", True, say=True)
    return got


def _first(cands: list[Candidate], req: Req) -> tuple[str, str]:
    for c in cands:
        if c.app is None:
            return c.key, c.key
        if m := next((m for m in c.methods if accepts(c.app, m, req)), ""):
            return c.key, m
    return "", ""


async def preview(fields: dict, said: str = "") -> dict[str, str | int | bool]:
    """What the new path would do with this note, for the shadow log: app keys, methods and
    counts only. Nothing is opened, scripted, asked or learned."""
    req = _req(fields, said)
    try:
        cands = await engine.choose(KIND, req.text)
    except Exception as e:  # noqa: BLE001 - a shadow run must never disturb the real one
        return {"error": type(e).__name__}
    key, method = _first(cands, req)
    return {
        "app": key,
        "method": method,
        "candidates": len(cands),
        "keys": ",".join(c.key for c in cands),
        "folder": bool(req.fields["folder"]),
        "named": bool(req.named),
    }


# Content types (type vocabulary, matched whole or as a UTI's last part) of text a document
# editor opens as its own: plain text, rich text, Markdown, HTML.
_TEXT_TYPES = ("text", "plain-text", "utf8-plain-text", "txt", "rtf", "rtfd", "markdown", "md", "html")
# App Intents schemas (the OS's own domains) for making a new document, board or sheet.
_NEW_DOC_SCHEMA = re.compile(
    r"^schema:(?:wordProcessor|spreadsheet|presentation|whiteboard)\.Create\w*Intent$"
)


def doc_like(app: App) -> bool:
    """An app whose New command makes a fresh note or document to type into: it makes notes,
    edits a text-like content type, or declares an App Intents schema for a new document."""
    if app.kind != "app":
        return False
    edits = {str(t).lower() for t in app.editor_types}
    return (
        any(t == x or t.endswith("." + x) for t in edits for x in _TEXT_TYPES)
        or any(_NEW_DOC_SCHEMA.match(s) for s in app.ai_schemas)
        or engine.fits_kind(KIND, app)
    )


async def top_app(said: str = "") -> App | None:
    """The app a "new note" goes to: this kind's first candidate that is an installed app (never
    the file fallback)."""
    cands = await engine.choose(KIND, said or "new note")
    return next((c.app for c in cands if c.app is not None and c.app.kind == "app"), None)
