"""neo2.core.decide.kinds.mail, the read half: the unread inbox through a mail app's own dictionary
(newest messages by index, never `whose`, never a property set), once more over fewer messages
after a dropped connection, the Dock badge as a hint, and a shadow preview that reads only an app
NEO may already script and that is running. osascript and the Dock are faked; every generated
script is compiled (darwin), never run."""

from __future__ import annotations

import asyncio
import dataclasses
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import pytest

from neo2.core.decide import aescript, axkit, engine, media, run, sdef
from neo2.core.decide.engine import Candidate
from neo2.core.decide.inventory import App
from neo2.core.decide.kinds import mail

US, RS = aescript.US, aescript.RS

INBOX = """
<class name="application" code="capp">
  <element type="mailbox"/>
  <property name="inbox" code="inmb" type="mailbox" access="r"/>
</class>
<class name="mailbox" code="mbxp">
  <element type="message"/>
  <property name="name" code="pnam" type="text"/>
  <property name="unread count" code="mbuc" type="integer" access="r"/>
</class>
<class name="message" code="mssg">
  <property name="sender" code="sndr" type="text" access="r"/>
  <property name="subject" code="subj" type="text" access="r"/>
  <property name="date received" code="rdrc" type="date" access="r"/>
  <property name="read status" code="isrd" type="boolean"/>
</class>
"""


def parse(body: str) -> sdef.Dictionary:
    return sdef.parse(ET.fromstring(f'<dictionary><suite name="S" code="SSSS">{body}</suite></dictionary>'))


D = parse(INBOX)
POSTBOX = App(
    bundle_id="test.postbox", name="Postbox", path="/Applications/Postbox.app", script=D,
    handles=("mailto",), days_used_30=5, last_used=time.time(),
)
PIGEON = App(bundle_id="test.pigeon", name="Pigeon", path="/Applications/Pigeon.app", handles=("mailto",))
TWO = (
    f"OK{US}12{RS}Priya Sharma <p@example.test>{US}Rent  due{US}2026-09-26T10:04:00"
    f"{RS}Bank{US}Your | statement{US}2026-09-25T18:30:00"
)


@pytest.fixture(autouse=True)
def _own_request():
    token = run._gen.set(None)
    run.forget()
    yield
    run.forget()
    run._gen.reset(token)


@pytest.fixture
def world(monkeypatch):
    """Fixed candidates, every gate open, osascript and the Dock faked."""
    w = SimpleNamespace(chosen=[], calls=[], answers=[], badge=None)

    async def choose(kind, text, apps=None, *, said="", about="", exclude=frozenset()):
        return list(w.chosen)

    async def allowed(app):
        return None

    async def osascript(script, argv, timeout):
        w.calls.append((script, list(argv), timeout))
        return w.answers.pop(0) if w.answers else (0, "OK" + US + "0", "")

    monkeypatch.setattr(engine, "choose", choose)
    monkeypatch.setattr(media, "_control_gate", allowed)
    monkeypatch.setattr(media, "_trusted", lambda: True)
    monkeypatch.setattr(media, "_osascript", osascript)
    monkeypatch.setattr(axkit, "dock_badge", lambda name: w.badge)
    return w


def cand(app, *methods):
    return Candidate(app, 0.5, 0.0, "", methods)


def read(**fields):
    return asyncio.run(mail.unread(fields, "check my email"))


# ---- the dictionary route ---------------------------------------------------------------------------


def test_the_route_comes_from_the_dictionary():
    assert sdef.read_caps(D) >= {"mail.read", "mail.unread_count"}
    r = aescript.unread_route(D)
    assert (r.inbox, r.count, r.message, r.read) == ("inmb", "mbuc", "mssg", "isrd")
    assert (r.sender, r.subject, r.received) == ("sndr", "subj", "rdrc")
    renamed = parse(INBOX.replace('code="inmb"', 'code="INBX"').replace('code="mbuc"', 'code="UNRD"'))
    assert (aescript.unread_route(renamed).inbox, aescript.unread_route(renamed).count) == ("INBX", "UNRD")
    for missing in ('<property name="read status" code="isrd" type="boolean"/>',
                    '<property name="unread count" code="mbuc" type="integer" access="r"/>',
                    '<property name="inbox" code="inmb" type="mailbox" access="r"/>'):
        assert aescript.unread_route(parse(INBOX.replace(missing, ""))) is None
    no_fields = INBOX.replace('<property name="sender" code="sndr" type="text" access="r"/>', "").replace(
        '<property name="subject" code="subj" type="text" access="r"/>', "")
    assert aescript.unread_route(parse(no_fields)) is None and aescript.unread_route(None) is None


def test_the_read_never_filters_never_sets_and_takes_its_numbers_from_argv():
    r = aescript.unread_route(D)
    script, argv = aescript.render_unread(r, "test.postbox", 7, 150)
    assert argv == ["test.postbox", "7", "150"]
    assert " whose " not in script and not re.search(r"set\s+«property", script)  # nothing marked read
    assert "test.postbox" not in script and "is not running" not in script
    guarded, _ = aescript.render_unread(r, "test.postbox", 7, 150, only_running=True)
    assert 'if application id bid is not running then return "EMPTY"' in guarded


@pytest.mark.skipif(sys.platform != "darwin" or shutil.which("osacompile") is None, reason="macOS only")
@pytest.mark.parametrize("only_running", [False, True])
def test_the_read_script_compiles(only_running, tmp_path):
    no_date = dataclasses.replace(aescript.unread_route(D), received="")
    for r in (aescript.unread_route(D), no_date):
        script, _ = aescript.render_unread(r, "b", 10, 150, only_running=only_running)
        p = subprocess.run(["osacompile", "-o", str(tmp_path / "s.scpt"), "-e", script], capture_output=True, text=True)
        assert p.returncode == 0, p.stderr


# ---- reading --------------------------------------------------------------------------------------------


def test_the_unread_list(world):
    world.chosen.append(cand(POSTBOX, "script_read", "badge"))
    world.answers.append((0, TWO, ""))
    r = read(limit=5)
    assert r.ok and r.say is None  # the information is the answer
    assert r.text == (
        "12 unread in total. Newest:\n"
        "Priya Sharma <p@example.test> | Rent due | Sat 26 Sep 2026 10:04\n"
        "Bank | Your / statement | Fri 25 Sep 2026 18:30"
    )
    assert world.calls[0][1] == ["test.postbox", "5", "150"]
    assert engine.proven("read_mail", "test.postbox")


def test_nothing_unread(world):
    world.chosen.append(cand(POSTBOX, "script_read"))
    world.answers.append((0, f"OK{US}0", ""))
    assert (read().text, read().ok) == ("No unread mail.", True)


def test_a_dropped_connection_is_read_again_over_fewer_messages(world):
    world.chosen.append(cand(POSTBOX, "script_read"))
    world.answers += [(1, "", "Connection is invalid. (-609)"), (0, TWO, "")]
    assert read().ok
    assert [c[1][2] for c in world.calls] == ["150", "50"]
    world.calls.clear()
    world.answers += [(1, "", "AppleEvent timed out. (-1712)")]
    assert not read().ok and len(world.calls) == 1  # only a dropped connection is retried


@pytest.mark.parametrize("limit, sent", [(None, "10"), (500, "50"), (0, "10"), ("x", "10")])
def test_the_limit_is_kept_in_bounds(world, limit, sent):
    world.chosen.append(cand(POSTBOX, "script_read"))
    read(limit=limit)
    assert world.calls[0][1][1] == sent


def test_permission_is_named_and_the_badge_is_only_a_hint(world, monkeypatch):
    world.chosen.append(cand(POSTBOX, "script_read", "badge"))
    world.answers.append((1, "", "Not authorized to send Apple events to Postbox. (-1743)"))
    world.badge = "3"
    r = read()
    assert r.ok and r.text == "Postbox shows 3 unread (the count on its Dock icon; I couldn't read the messages)."
    assert engine.prior("read_mail", "test.postbox", "script_read") == pytest.approx(0.75)  # no blame
    world.answers.append((1, "", "Not authorized to send Apple events to Postbox. (-1743)"))
    monkeypatch.setattr(media, "_trusted", lambda: False)  # no Accessibility: no Dock either
    r = read()
    assert not r.ok and r.text == "Postbox needs your OK before NEO can control it"


@pytest.mark.parametrize(
    "failed, said",
    [((124, "", "timed out after 25s"), "Postbox didn't answer in time"),
     ((1, "", "Postbox got an error. (-2700)"), "Postbox's scripting failed (-2700)")],
)
def test_no_dock_count_after_a_failed_read_leaves_the_reads_reason(world, failed, said):
    # no count there is no "no unread mail" (badges may be off): the read's failure is what's said
    world.chosen.append(cand(POSTBOX, "script_read", "badge"))
    world.answers.append(failed)
    r = read()
    assert (r.ok, r.text) == (False, said)
    assert engine._load()["arms"].get("read_mail", {}).get("test.postbox", {}).get("badge") is None  # no attempt


@pytest.mark.parametrize("label, ok", [("3", True), ("99+", True), ("!", False), (None, False), ("", False)])
def test_badge_labels(world, label, ok):
    world.chosen.append(cand(PIGEON, "badge"))
    world.badge = label
    r = read()
    assert r.ok is ok
    if not ok:
        assert "no unread count" in r.text and engine.prior("read_mail", "test.pigeon", "badge") == pytest.approx(0.5)


def test_methods_come_from_evidence(monkeypatch):
    assert mail.methods_for(POSTBOX) == ("script_read", "badge")
    assert mail.methods_for(PIGEON) == ("badge",)  # the mail handler with no dictionary
    browser = dataclasses.replace(PIGEON, bundle_id="test.browser", handles=("mailto", "https"))
    assert mail.methods_for(browser) == ()  # a browser that opens mail links reads no inbox
    assert mail.methods_for(dataclasses.replace(PIGEON, path="")) == ()
    assert "by script" in mail.READ.describe(POSTBOX)
    req = mail._req({}, "")
    monkeypatch.setattr(media, "_trusted", lambda: False)
    assert not mail.READ.accepts(PIGEON, "badge", req) and mail.READ.accepts(POSTBOX, "script_read", req)


def test_the_real_engine_asks_the_mail_apps_and_drops_a_browser(monkeypatch):
    browser = dataclasses.replace(PIGEON, bundle_id="test.browser", handles=("mailto", "https"), days_used_30=9)
    monkeypatch.setitem(engine.DRIVERS, "read_mail", "neo2.core.decide.kinds.mail")
    monkeypatch.setattr(engine, "_zero_shot_can_answer", lambda: False)
    monkeypatch.setattr(engine, "_inventory", lambda: [POSTBOX, browser])
    cands = asyncio.run(engine.choose("read_mail", "check my email"))
    assert [(c.key, c.methods) for c in cands] == [("test.postbox", ("script_read", "badge"))]


# ---- the shadow preview -----------------------------------------------------------------------------


LEGACY = "12 unread in total. Newest:\nPriya | Rent | Saturday\n"


def test_preview_reads_only_an_app_neo_may_already_script(world, monkeypatch):
    world.chosen.append(cand(POSTBOX, "script_read", "badge"))
    gates = []

    async def gate(app):
        gates.append(app.bundle_id)

    monkeypatch.setattr(media, "_control_gate", gate)
    monkeypatch.setattr(media, "automation", lambda bid, ask=False: media.AE_NOT_RUNNING)
    facts = asyncio.run(mail.preview({}, "check my email", LEGACY))
    assert facts == {"app": "test.postbox", "method": "script_read", "candidates": 1, "keys": "test.postbox",
                     "legacy_unread": 12}
    assert world.calls == [] and gates == []  # not running: nothing started, nothing asked
    monkeypatch.setattr(media, "automation", lambda bid, ask=False: media.AE_ALLOWED if not ask else pytest.fail())
    world.answers.append((0, TWO, ""))
    facts = asyncio.run(mail.preview({}, "check my email", LEGACY))
    assert facts["read"] is True and (facts["unread"], facts["listed"], facts["same"]) == (12, 2, True)
    assert 'is not running then return "EMPTY"' in world.calls[0][0] and gates == []
    assert not (engine.proven("read_mail", "test.postbox"))  # nothing learned from a shadow read
    world.answers.append((0, "EMPTY", ""))  # it quit in between: still nothing started
    facts = asyncio.run(mail.preview({}, "", ""))
    assert facts["read"] is False and "legacy_unread" not in facts and "same" not in facts


def test_the_read_takes_every_field_the_old_script_read(monkeypatch):
    """Equivalence with the legacy tool: each inbox, mailbox and message property its script read
    (by the dictionary's own names) is read here too, by code."""
    from neo2.core.tools.computer import legacy

    seen = []

    async def osascript(script, timeout=15):
        seen.append(script)
        return ""

    monkeypatch.setattr(legacy, "osascript", osascript)
    asyncio.run(legacy.mail_unread())
    props = {**D.class_props["application"], **D.class_props["mailbox"], **D.class_props["message"]}
    used = {code for name, (code, _t) in props.items() if re.search(rf"\b{re.escape(name)}\b", seen[0])}
    assert used == {"inmb", "mbuc", "isrd", "sndr", "subj", "rdrc"}
    script, _argv = aescript.render_unread(aescript.unread_route(D), "b", 10, 150)
    assert used <= set(re.findall(r"«property (\w{4})»", script))


def test_through_the_real_engine(monkeypatch):
    monkeypatch.setitem(engine.DRIVERS, "read_mail", "neo2.core.decide.kinds.mail")
    monkeypatch.setattr(engine, "_zero_shot_can_answer", lambda: False)
    monkeypatch.setattr(engine, "_inventory", lambda: [POSTBOX])

    async def allowed(app):
        return None

    async def osascript(script, argv, timeout):
        return 0, TWO, ""

    monkeypatch.setattr(media, "_control_gate", allowed)
    monkeypatch.setattr(media, "_osascript", osascript)
    r = asyncio.run(mail.unread({"limit": 2}, "check my email"))
    assert r.ok and r.text.startswith("12 unread in total. Newest:\nPriya Sharma")
