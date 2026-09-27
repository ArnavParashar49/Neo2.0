"""The user's mail, through the mail app this user uses. This half reads (read_mail); sending
(send_email) joins it later.

The engine picks the app (``choose("read_mail")``: one whose dictionary reads an inbox, or the
default mail handler when it is no browser) and :func:`neo2.core.decide.run.run_best` tries them in
order. Per app, best first:

* ``script_read`` - its dictionary: the inbox's unread count, then the newest messages by index,
  keeping the unread ones (sender, subject, date received). Never a ``whose`` filter (it drops
  the connection on a big inbox) and never a property set, so nothing is marked read;
* ``badge`` - the count on its Dock icon: a hint when the messages themselves can't be read (no
  count there is no answer: badges can be off, and the read's own failure is what gets said).

What is read is third-party text: the tool hands it to the model fenced as untrusted. NEO never
reads the mail store on disk.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from typing import TYPE_CHECKING

from neo2.core.decide import aescript, axkit, engine, run
from neo2.core.decide.run import Effect, Outcome, Req, Result

if TYPE_CHECKING:
    from neo2.core.decide.engine import Candidate
    from neo2.core.decide.inventory import App

__all__ = ["READ", "unread", "preview", "methods_for"]

LIMIT, MAX_LIMIT = 10, 50  # unread messages listed
SCAN, RETRY_SCAN = 150, 50  # newest messages looked at; fewer after a dropped connection
READ_S, RETRY_S, PREVIEW_S = 25.0, 12.0, 20.0
_DROPPED = re.compile(r"\(-609\)\s*$")  # the app dropped the connection mid-read
_BADGE = re.compile(r"^\d+\+?$")  # a count, maybe capped ("99+")


def _one_line(s: str, n: int = 160) -> str:
    s = " ".join(str(s or "").split()).replace(" | ", " / ")  # a field never looks like the separator
    return s if len(s) <= n else s[: n - 1] + "…"


def _when(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%a %d %b %Y %H:%M")
    except ValueError:
        return iso


def _listing(total: str, rows: list[list[str]]) -> str:
    """The answer the model reads: the unread count, then sender | subject | date per message."""
    n = int(total) if total.strip().isdigit() else len(rows)
    if n == 0 and not rows:
        return "No unread mail."
    lines = [f"{n} unread in total. Newest:"]
    for row in rows:
        sender, subject, when = (list(row) + ["", "", ""])[:3]
        lines.append(f"{_one_line(sender)} | {_one_line(subject)} | {_when(when)}")
    return "\n".join(lines)


async def _read(app: App, limit: int, timeout: float, *, only_running: bool = False) -> Outcome:
    """One read of the app's inbox through its dictionary (once more, over fewer messages, when
    the app drops the connection)."""
    r = aescript.unread_route(app.script)
    if r is None:
        return Outcome(False, f"{app.name} can't list mail by script", reason="route_bad")
    code, out, err = 0, "", ""
    for scan, wait in ((SCAN, timeout), (RETRY_SCAN, min(RETRY_S, timeout))):
        script, argv = aescript.render_unread(r, app.bundle_id, limit, scan, only_running=only_running)
        code, out, err = await aescript.run(script, argv, wait)
        if code == 0 or not _DROPPED.search(err or ""):
            break
    if code != 0:
        return aescript.fail(app.name, code, err, committed=False)
    status, rows = aescript.rows(out)
    if status == "EMPTY":
        return Outcome(False, f"{app.name} isn't running", reason="not_found")
    if status != "OK":
        return Outcome(False, f"{app.name} gave an answer I couldn't read")
    total, found = (rows[0][0], rows[1:]) if rows else ("", [])
    text = _listing(total, found)
    receipt = {"where": app.name, "unread": total, "listed": len(found), "verified": True}
    return Outcome(True, text, effect=Effect.DONE, receipt=receipt, data=text)


async def _badge(app: App) -> Outcome | None:
    """The Dock count. Without one, after a read by script it is no attempt at all (None): what
    the user hears is why the read failed, not a line that sounds like "no unread mail"."""
    label = (await asyncio.to_thread(axkit.dock_badge, app.name) or "").strip()
    if not _BADGE.match(label):
        if aescript.unread_route(app.script) is not None:
            return None
        return Outcome(False, f"{app.name}'s Dock icon shows no unread count", reason="not_found")
    text = f"{app.name} shows {label} unread (the count on its Dock icon; I couldn't read the messages)."
    return Outcome(True, text, effect=Effect.DONE, receipt={"where": app.name, "unread": label}, data=text)


class _Read:
    """The read_mail driver (see :class:`neo2.core.decide.run.Driver`)."""

    KIND = "read_mail"
    OVERALL_S = 45.0
    BUDGET_S = {"script_read": 40.0, "badge": 5.0}

    def methods_for(self, app: App, kind: str | None = None) -> tuple[str, ...]:
        """Its dictionary's inbox when it has one; the Dock count for any mail app with a window
        (a browser that handles mail links has neither: it is left out)."""
        if app.kind != "app" or engine._is_browser(app):
            return ()
        out = ["script_read"] if aescript.unread_route(app.script) is not None else []
        if app.path:
            out.append("badge")
        return tuple(out)

    def accepts(self, app: App, method: str, req: Req) -> bool:
        if method == "badge":
            return axkit.trusted()  # the Dock is read through Accessibility
        return method in self.methods_for(app)

    def describe(self, app: App) -> str:
        how = {"script_read": "reads its inbox by script", "badge": "shows an unread count on its Dock icon"}
        return "; ".join(how[m] for m in self.methods_for(app))

    async def attempt(self, req: Req, cand: Candidate, method: str) -> Outcome | None:
        app = cand.app
        if app is None:
            return Outcome(False, "no mail app to read", reason="route_bad")
        if method == "script_read":
            return await _read(app, req.fields["limit"], READ_S)
        if method == "badge":
            return await _badge(app)
        return Outcome(False, f"{app.name} can't be read that way", reason="route_bad")

    async def fallback(self, req: Req, key: str) -> Outcome | None:
        return None  # nothing else holds the user's mail


READ = _Read()


def methods_for(app: App, kind: str | None = None) -> tuple[str, ...]:
    """The engine's lookup for this module's kinds (reading is the only one so far)."""
    return READ.methods_for(app, kind)


def _req(fields: dict, said: str) -> Req:
    try:
        limit = max(1, min(MAX_LIMIT, int(fields.get("limit") or LIMIT)))
    except (TypeError, ValueError):
        limit = LIMIT
    named = " ".join(str(fields.get("app") or "").split())
    text = " ".join(said.split()) or "check my email"
    if named and named.casefold() not in text.casefold():
        text = f"{text} in {named}"
    return Req(READ.KIND, {"limit": limit}, text, named=named)


async def unread(fields: dict, said: str = "") -> Result:
    """The unread inbox (``mail_unread``'s arguments: limit, app); ``said`` is the user's own
    words for the request, when known."""
    return await run.run_best(READ, _req(fields, said))


_LEGACY_TOTAL = re.compile(r"^\s*(\d+) unread in total")


async def preview(fields: dict, said: str = "", legacy: str = "") -> dict[str, str | int | bool]:
    """What the new path would do, for the shadow log: app keys, methods and counts only. When
    NEO may already script the chosen app and it is running, the new read runs too and its
    unread count is set against the one in ``legacy`` (the old tool's answer); nothing is
    launched, asked, marked read or learned."""
    req = _req(fields, said)
    try:
        cands = await engine.choose(READ.KIND, req.text)
    except Exception as e:  # noqa: BLE001 - a shadow run must never disturb the real one
        return {"error": type(e).__name__}
    top = next((c for c in cands if c.app is not None and c.methods), None)
    method = next((m for m in top.methods if READ.accepts(top.app, m, req)), "") if top else ""
    facts: dict[str, str | int | bool] = {
        "app": top.key if top else "",
        "method": method,
        "candidates": len(cands),
        "keys": ",".join(c.key for c in cands),
    }
    if top and method == "script_read" and await asyncio.to_thread(aescript.may_script, top.app.bundle_id):
        got = await _read(top.app, req.fields["limit"], PREVIEW_S, only_running=True)
        facts["read"] = got.ok
        if got.ok and str(got.receipt.get("unread", "")).isdigit():
            facts["unread"] = int(got.receipt["unread"])
            facts["listed"] = int(got.receipt["listed"])
    if m := _LEGACY_TOTAL.match(legacy or ""):
        facts["legacy_unread"] = int(m.group(1))
        if "unread" in facts:
            facts["same"] = facts["unread"] == facts["legacy_unread"]
    return facts
