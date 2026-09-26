"""neo.decide.kinds.note: write_note through a notes app's own dictionary (made, read back, looked
for after a timeout, never made twice), its New command (⌘N found by shortcut), and a Markdown
file that never replaces another. osascript, the AX tree and the app launch are faked; the file
goes to the test's own folder. Every generated script is compiled (darwin), never run."""

from __future__ import annotations

import asyncio
import dataclasses
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime
from types import SimpleNamespace

import pytest

from neo import config
from neo.decide import aescript, engine, inventory, media, run, sdef
from neo.decide.engine import Candidate
from neo.decide.inventory import App
from neo.decide.kinds import note

US, RS = aescript.US, aescript.RS

NOTES = """
<command name="make" code="corecrel">
  <parameter name="new" code="kocl" type="type"/>
  <parameter name="at" code="insh" type="location specifier" optional="yes"/>
  <parameter name="with properties" code="prdt" type="record" optional="yes"/>
  <result type="specifier"/>
</command>
<class name="application" code="capp">
  <element type="account"/><element type="folder"/><element type="note"/>
  <property name="default account" code="dfac" type="account"/>
</class>
<class name="account" code="acct">
  <element type="folder"/><element type="note"/>
  <property name="default folder" code="dfol" type="folder"/>
  <property name="name" code="pnam" type="text"/>
</class>
<class name="folder" code="cfol">
  <element type="note"/>
  <property name="name" code="pnam" type="text"/>
  <property name="id" code="ID  " type="text" access="r"/>
</class>
<class name="note" code="note">
  <property name="name" code="pnam" type="text"/>
  <property name="id" code="ID  " type="text" access="r"/>
  <property name="body" code="body" type="text" description="the HTML content of the note"/>
  <property name="plaintext" code="text" type="text" access="r"/>
  <property name="creation date" code="ascd" type="date" access="r"/>
</class>
"""
# the same, kept at the application itself: no default properties, no folders, no creation date
FLAT = """
<command name="make" code="corecrel">
  <parameter name="new" code="kocl" type="type"/>
  <parameter name="with properties" code="prdt" type="record" optional="yes"/>
</command>
<class name="application" code="capp"><element type="note"/></class>
<class name="note" code="note">
  <property name="title" code="ttl1" type="text"/>
  <property name="content" code="body" type="text" description="plain text"/>
</class>
"""


def parse(body: str) -> sdef.Dictionary:
    return sdef.parse(ET.fromstring(f'<dictionary><suite name="S" code="SSSS">{body}</suite></dictionary>'))


D = parse(NOTES)
NOTEBOOK = App(
    bundle_id="test.notebook", name="Notebook", path="/Applications/Notebook.app", script=D,
    days_used_30=9, last_used=time.time(),
)
JOTTER = App(bundle_id="test.jotter", name="Jotter", path="/Applications/Jotter.app")


@pytest.fixture(autouse=True)
def _own_request(tmp_path, monkeypatch):
    """Each test starts its own request memory, and notes files go to its own folder."""
    monkeypatch.setenv("NEO_NOTES_DIR", str(tmp_path / "notes"))
    monkeypatch.setattr(config, "_settings", None)
    token = run._gen.set(None)
    run.forget()
    yield
    run.forget()
    run._gen.reset(token)


class FakeOsa:
    """media._osascript: each script answered by the first rule whose marker it contains."""

    def __init__(self, **answers):
        self.calls: list[tuple[str, list[str]]] = []
        self.answers = answers

    def kind(self, script: str) -> str:
        if "«event corecrel»" in script:
            return "make"
        if " whose " in script:
            return "search"
        if "every «class cfol»" in script:
            return "folders"
        return "other"

    async def __call__(self, script, argv, timeout):
        self.calls.append((script, list(argv)))
        got = self.answers.get(self.kind(script), (0, "OK", ""))
        return got(script, argv) if callable(got) else got

    def of(self, kind: str) -> list[list[str]]:
        return [argv for script, argv in self.calls if self.kind(script) == kind]


@pytest.fixture
def world(monkeypatch):
    """Fixed candidates, every gate open, osascript and the window faked."""
    chosen: list[Candidate] = []
    asked: list[tuple[str, str]] = []

    async def choose(kind, text, apps=None, *, said="", about="", exclude=frozenset()):
        asked.append((kind, text))
        return list(chosen)

    async def allowed(app):
        return None

    osa = FakeOsa()
    monkeypatch.setattr(engine, "choose", choose)
    monkeypatch.setattr(media, "_control_gate", allowed)
    monkeypatch.setattr(media, "_trusted", lambda: True)
    monkeypatch.setattr(media, "_osascript", osa)
    return SimpleNamespace(chosen=chosen, osa=osa, asked=asked)


def cand(app, *methods):
    return Candidate(app, 0.5, 0.0, "", methods)


def made(oid="n1", name="Shopping", text="Shopping\nbuy milk and eggs", where="Notes"):
    return 0, f"OK{US}{oid}{US}{name}{US}{text}{US}{where}", ""


def write(**fields):
    return asyncio.run(note.write({"body": "buy milk and eggs", **fields}, "make a note saying buy milk and eggs"))


# ---- the dictionary route -------------------------------------------------------------------------------


def test_the_route_comes_from_the_dictionarys_own_codes():
    r = aescript.note_route(D)
    assert (r.klass, r.make, r.p_new, r.p_at, r.p_props) == ("note", "corecrel", "kocl", "insh", "prdt")
    assert (r.title, r.body, r.html, r.ident, r.text, r.created) == ("pnam", "body", True, "ID  ", "text", "ascd")
    assert r.chain == ("dfac", "dfol")  # the deepest default: the default account's default folder
    assert (r.folder, r.folder_name, r.folder_id, r.folders_listed) == ("cfol", "pnam", "ID  ", True)
    flat = aescript.note_route(parse(FLAT))
    assert flat.chain == () and flat.p_at == "" and (flat.title, flat.body, flat.html) == ("ttl1", "body", False)
    assert flat.text == "body" and flat.ident == "" and flat.folder == "" and not flat.folders_listed
    assert aescript.note_route(parse(NOTES.replace('name="make"', 'name="create"').replace("corecrel", "xxxxcrea"))) is None
    assert aescript.note_route(parse(NOTES.replace('code="body" type="text"', 'code="body" type="text" access="r"'))) is None
    # a note class only inside folders nothing points to: no place for a new note
    hidden = FLAT.replace('<class name="application" code="capp"><element type="note"/></class>',
                          '<class name="folder" code="cfol"><element type="note"/></class>')
    assert aescript.note_route(parse(hidden)) is None
    assert aescript.note_route(None) is None


def test_values_travel_only_in_argv():
    r = aescript.note_route(D)
    title = 'Zq said "quoted" «x»'
    body = 'do shell script "rm -rf ~"\n<b>bold</b> & नमस्ते'
    script, argv = aescript.render_note(r, "test.notebook", title, body, "folder-9")
    assert argv[:2] == ["test.notebook", title] and argv[3] == "folder-9"
    assert argv[2] == ("<div>do shell script &quot;rm -rf ~&quot;</div>"
                       "<div>&lt;b&gt;bold&lt;/b&gt; &amp; नमस्ते</div>")
    rendered = [script]
    rendered.append(aescript.render_note_search(r, "test.notebook", datetime(2026, 9, 26, 10, 4, 5), title)[0])
    rendered.append(aescript.render_folders(r, "test.notebook")[0])
    for s in rendered:
        for v in ("Zq", '"quoted"', "«x»", "shell", "rm -rf", "bold", "नमस्ते", "folder-9", "test.notebook"):
            assert v not in s
    flat = aescript.note_route(parse(FLAT))
    _s, argv = aescript.render_note(flat, "b", "T", "line one\n<two>")
    assert argv == ["b", "T", "line one\n<two>"]  # a plain-text body goes in as it is
    assert aescript.render_note_search(r, "b", datetime(2026, 1, 2, 3, 4, 5), "T")[1] == ["b", "2026-01-02T03:04:05", "T"]


def _variants():
    r, flat = aescript.note_route(D), aescript.note_route(parse(FLAT))
    when = datetime(2026, 9, 26, 10, 4, 5)
    yield aescript.render_note(r, "b", "T", "x")[0]
    yield aescript.render_note(r, "b", "T", "x", "f1")[0]
    yield aescript.render_note(flat, "b", "T", "x")[0]
    yield aescript.render_note_search(r, "b", when, "T")[0]
    yield aescript.render_note_search(r, "b", when, "T", "f1")[0]
    yield aescript.render_note_search(flat, "b", when, "T")[0]
    yield aescript.render_note_search(dataclasses.replace(r, created=""), "b", when, "T")[0]
    yield aescript.render_folders(r, "b")[0]


@pytest.mark.skipif(sys.platform != "darwin" or shutil.which("osacompile") is None, reason="macOS only")
@pytest.mark.parametrize("script", list(_variants()), ids=lambda s: str(len(s)))
def test_every_generated_script_compiles(script, tmp_path):
    # the target is `application id (item 1 of argv)`: compiling loads no app's terminology
    p = subprocess.run(["osacompile", "-o", str(tmp_path / "s.scpt"), "-e", script], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr


# ---- script_make ------------------------------------------------------------------------------------------


def test_made_and_read_back_then_quiet_where_notes_always_go(world):
    world.chosen.append(cand(NOTEBOOK, "script_make"))
    world.osa.answers["make"] = made()
    r = write(title="Shopping")
    assert (r.ok, r.text, r.say) == (True, "Saved in Notebook.", True)  # the first time: say where
    assert world.osa.of("make") == [["test.notebook", "Shopping", "<div>buy milk and eggs</div>"]]
    assert world.asked == [("write_note", "make a note saying")]  # the command, not the note itself
    assert write(title="Shopping").text == "Already added a moment ago."  # a repeat within 120 s
    assert len(world.osa.of("make")) == 1
    world.osa.answers["make"] = made(text="Shopping\nbread")
    r = asyncio.run(note.write({"body": "bread", "title": "Shopping"}))
    assert (r.ok, r.say) == (True, False)  # the habitual app: nothing said
    assert engine.proven("write_note", "test.notebook")


def test_a_note_without_a_title_is_named_by_its_first_words(world):
    world.chosen.append(cand(NOTEBOOK, "script_make"))
    world.osa.answers["make"] = made(name="one two three four five six", text="one two three four five six seven")
    assert asyncio.run(note.write({"body": "one two three four five six seven"})).ok
    assert world.osa.of("make")[0][1] == "one two three four five six"
    assert asyncio.run(note.write({"body": " ", "title": ""})).text.startswith("Error:")


@pytest.mark.parametrize("found", [True, False])
def test_a_timed_out_make_is_looked_for_and_never_made_again(world, found):
    world.chosen += [cand(NOTEBOOK, "script_make"), cand(JOTTER, "ax_new"), cand(None, "file")]
    world.osa.answers["make"] = (124, "", "timed out after 8s")
    hit = f"{RS}n7{US}Shopping{US}Shopping\nbuy milk and eggs" if found else ""
    world.osa.answers["search"] = (0, f"OK{hit}", "")
    r = write(title="Shopping")
    assert len(world.osa.of("make")) == 1 and len(world.osa.of("search")) == 1
    search = world.osa.of("search")[0]
    assert search[0] == "test.notebook" and search[2] == "Shopping"
    assert datetime.fromisoformat(search[1]) <= datetime.now()
    if found:
        assert (r.ok, r.text) == (True, "Saved in Notebook.")
    else:
        assert not r.ok and r.say is True and "may have saved it in Notebook" in r.text
        assert engine.prior("write_note", "test.notebook", "script_make") == pytest.approx(0.75)  # no blame
    assert not config.settings().notes_dir.exists()  # no second copy anywhere


def test_a_note_made_but_not_read_back_is_looked_for(world):
    world.chosen += [cand(NOTEBOOK, "script_make"), cand(None, "file")]
    world.osa.answers["make"] = (0, "OK", "")
    world.osa.answers["search"] = (0, f"OK{RS}n2{US}buy milk{US}buy milk and eggs", "")
    assert write().text == "Saved in Notebook."


def test_a_note_that_reads_back_wrong_is_not_made_again(world):
    world.chosen += [cand(NOTEBOOK, "script_make"), cand(None, "file")]
    world.osa.answers["make"] = made(text="something else entirely")
    r = write()
    assert not r.ok and "couldn't check what it says" in r.text
    assert not config.settings().notes_dir.exists()  # no copy in a file either


@pytest.mark.parametrize(
    "err, blamed",
    [("Can't get folder. (-1728)", False), ("Not authorized to send Apple events. (-1743)", True)],
)
def test_a_make_that_never_ran_falls_through(world, err, blamed):
    world.chosen += [cand(NOTEBOOK, "script_make"), cand(None, "file")]
    world.osa.answers["make"] = (1, "", err)
    r = write(title="Shopping")
    assert r.ok and r.text.startswith("Saved it as a file in")
    assert engine.prior("write_note", "test.notebook", "script_make") == pytest.approx(0.75)  # no blame either way


FOLDERS = (0, f"OK{RS}Work{US}f1{RS}Notes{US}f2{RS}Goa Trip{US}f3{RS}Trip Photos{US}f4", "")


def test_a_named_folder_by_name_then_by_whole_words(world):
    world.chosen.append(cand(NOTEBOOK, "script_make"))
    world.osa.answers.update(folders=FOLDERS, make=made(where="Work"))
    r = write(folder="work")
    assert world.osa.of("make")[0][3] == "f1" and r.text == "Saved in Notebook (“Work”)."
    world.osa.answers["make"] = made(where="Goa Trip", text="x y")
    assert write(folder="goa", body="x y").ok and world.osa.of("make")[-1][3] == "f3"


def test_a_missing_folder_goes_to_the_default_and_is_said(world):
    world.chosen.append(cand(NOTEBOOK, "script_make"))
    world.osa.answers.update(folders=FOLDERS, make=made(where="Notes"))
    r = write(folder="garden")
    assert len(world.osa.of("make")[0]) == 3  # no folder id: the default folder
    assert (r.ok, r.say) == (True, True)
    assert r.text == "There's no folder called “garden” in Notebook, so I put it in “Notes”."


@pytest.mark.parametrize("probs, fid", [({"Goa Trip": 0.9, "Trip Photos": 0.1}, "f3"), ({"Goa Trip": 0.5, "Trip Photos": 0.5}, None)])
def test_a_folder_tie_is_laya_s_only_when_sure(world, monkeypatch, probs, fid):
    world.chosen.append(cand(NOTEBOOK, "script_make"))
    world.osa.answers.update(folders=FOLDERS, make=made(where="x"))
    asked = []

    async def laya(state, questions, timeout=None):
        asked.append(sorted(questions["folder"]["criteria"]))
        return {"answers": {"folder": {"probabilities": probs}}}

    monkeypatch.setattr(engine, "_ask_laya", laya)
    write(folder="trip")
    assert asked == [["Goa Trip", "Trip Photos"]]  # only the tied folders, by their own titles
    argv = world.osa.of("make")[0]
    assert (argv[3] if len(argv) > 3 else None) == fid


# ---- ax_new ---------------------------------------------------------------------------------------------


class El:
    def __init__(self, role, children=(), **attrs):
        self.attrs = {"AXRole": role, **attrs}
        self.children = list(children)


class FakeUI:
    """An app whose New menu item (⌘N) puts the focus in ``editor``."""

    def __init__(self, monkeypatch, editor: El | None, *, value_sticks: bool = True):
        self.new = El("AXMenuItem", AXTitle="New Page", AXMenuItemCmdChar="N", AXMenuItemCmdModifiers=0)
        other = El("AXMenuItem", AXTitle="New Folder", AXMenuItemCmdChar="N", AXMenuItemCmdModifiers=1)
        bar = El("AXMenuBar", [El("AXMenuBarItem", [El("AXMenu", [other, self.new])])])
        self.app = El("AXApplication", AXMenuBar=bar, AXFocusedUIElement=El("AXTextArea", AXValue="an old note"))
        self.editor, self.value_sticks, self.pressed, self.launched = editor, value_sticks, [], []

        async def launch(app, ready_s):
            self.launched.append(app.bundle_id)
            return 42

        monkeypatch.setattr(media, "_launch", launch)
        monkeypatch.setattr(media, "_attr", self.attr)
        monkeypatch.setattr(media, "_set", self.set)
        monkeypatch.setattr(media, "_perform", self.perform)
        monkeypatch.setattr(media, "_ax_app", lambda pid: self.app)
        monkeypatch.setattr(note, "NEW_S", 0.2)
        monkeypatch.setattr(media, "lexicon", lambda: {"search": frozenset({"Search"}), "find": frozenset({"Find"})})

    @staticmethod
    def attr(el, name, default=None):
        if el is None:
            return default
        return list(el.children) if name == "AXChildren" else el.attrs.get(name, default)

    def set(self, el, name, value):
        if name != "AXValue" or self.value_sticks:
            el.attrs[name] = value
        return True

    def perform(self, el, action):
        self.pressed.append(el.attrs.get("AXTitle"))
        if el is self.new and self.editor is not None:
            self.app.attrs["AXFocusedUIElement"] = self.editor
        return True


def test_the_new_command_by_its_shortcut_and_the_text_read_back(world, monkeypatch):
    world.chosen += [cand(JOTTER, "ax_new"), cand(None, "file")]
    box = El("AXTextArea", AXValue="")
    ui = FakeUI(monkeypatch, box)
    r = write(title="Shopping")
    assert (r.ok, r.text) == (True, "Wrote it in Jotter.")
    assert ui.launched == ["test.jotter"] and ui.pressed == ["New Page"]  # ⌘N, not ⇧⌘N
    assert box.attrs["AXValue"] == "Shopping\nbuy milk and eggs"


def test_a_line_break_is_typed_as_shift_return_never_return(world, monkeypatch):
    world.chosen.append(cand(JOTTER, "ax_new"))
    box = El("AXTextArea", AXValue="")
    FakeUI(monkeypatch, box, value_sticks=False)
    from neo.decide import axkit

    typed = []

    def type_(pid, text, newline):
        typed.append((pid, newline))
        box.attrs["AXValue"] = text

    monkeypatch.setattr(axkit, "_pid_of", lambda el: 42)
    monkeypatch.setattr(axkit, "_type", type_)
    assert write(title="Shopping").ok and typed == [(42, "shift+return")]


@pytest.mark.parametrize(
    "editor, why",
    [(None, "no editor"), (El("AXTextArea", AXValue="someone's old note"), "not empty"),
     (El("AXTextField", AXValue=""), "one line")],
)
def test_no_empty_editor_is_partial_and_the_next_way_is_tried(world, monkeypatch, editor, why):
    world.chosen += [cand(JOTTER, "ax_new"), cand(None, "file")]
    FakeUI(monkeypatch, editor)
    before = editor.attrs["AXValue"] if editor is not None else None
    r = write(title="Shopping")
    assert r.ok and r.text.startswith("Saved it as a file"), why
    assert editor is None or editor.attrs["AXValue"] == before  # nothing typed into it


def test_text_that_went_in_but_reads_back_wrong_is_unknown(world, monkeypatch):
    world.chosen += [cand(JOTTER, "ax_new"), cand(None, "file")]
    box = El("AXTextArea", AXValue="")
    FakeUI(monkeypatch, box)

    def cut(el, name, value):  # a field that keeps only the first few characters
        el.attrs[name] = value[:5] if name == "AXValue" else value
        return True

    monkeypatch.setattr(media, "_set", cut)
    monkeypatch.setattr("neo.decide.axkit._pid_of", lambda el: None)
    r = write(title="Shopping")
    assert not r.ok and "may have written it in Jotter" in r.text
    assert not config.settings().notes_dir.exists()


def test_no_new_command_is_the_routes_fault(world, monkeypatch):
    world.chosen += [cand(JOTTER, "ax_new"), cand(None, "file")]
    ui = FakeUI(monkeypatch, El("AXTextArea", AXValue=""))
    ui.app.attrs["AXMenuBar"] = El("AXMenuBar")
    assert write().ok
    arm = engine._load()["arms"]["write_note"]["test.jotter"]["ax_new"]
    assert engine._demoted(arm, time.time())  # at once, not after three tries


# ---- the file ---------------------------------------------------------------------------------------------


def test_the_file_is_new_never_over_another_and_read_back(world):
    world.chosen.append(cand(None, "file"))
    folder = config.settings().notes_dir
    folder.mkdir(parents=True)
    first = folder / f"{date.today().isoformat()} Shopping.md"
    first.write_text("the user's own file")
    r = write(title="Shopping")
    assert r.ok and r.text == f"Saved it as a file in {folder}."
    assert first.read_text() == "the user's own file"
    second = folder / f"{date.today().isoformat()} Shopping-2.md"
    assert second.read_text() == "# Shopping\n\nbuy milk and eggs\n"
    r = asyncio.run(note.write({"body": "call the plumber\ntomorrow"}))
    made_ = folder / f"{date.today().isoformat()} call the plumber tomorrow.md"
    assert r.ok and made_.read_text() == "call the plumber\ntomorrow\n"  # no title given: none added


def test_a_file_name_is_never_a_path_or_hidden():
    assert note._slug("../../.ssh/keys: a/b") == "ssh keys a b"
    assert note._slug("a/b\\c") == "a b c" and note._slug(".hidden") == "hidden"
    assert note._slug("   ") == "Note" and len(note._slug("x" * 200)) == 60


def test_a_folder_notes_may_not_go_to_is_refused(world, monkeypatch, tmp_path):
    world.chosen.append(cand(None, "file"))
    monkeypatch.setenv("NEO_NOTES_DIR", str(tmp_path / ".hidden" / "notes"))
    monkeypatch.setattr(config, "_settings", None)
    r = write()
    assert not r.ok and "hidden" in r.text and not (tmp_path / ".hidden").exists()


# ---- choosing -------------------------------------------------------------------------------------------


def test_methods_come_from_evidence():
    assert note.methods_for(NOTEBOOK) == ("script_make", "ax_new")
    assert note.methods_for(JOTTER) == ("ax_new",)
    browser = dataclasses.replace(JOTTER, bundle_id="test.browser", handles=("https",))
    chat = dataclasses.replace(JOTTER, bundle_id="test.chat", handles=("sms",))
    assert note.methods_for(browser) == () and note.methods_for(chat) == ()  # a line there could be sent
    assert note.methods_for(dataclasses.replace(JOTTER, path="")) == ()
    assert "by script" in note.describe(NOTEBOOK) and "window" in note.describe(JOTTER)


@pytest.mark.parametrize(
    "app, yes",
    [
        (dataclasses.replace(JOTTER, editor_types=("public.plain-text",)), True),
        (dataclasses.replace(JOTTER, editor_types=("net.daringfireball.markdown",)), True),
        (dataclasses.replace(JOTTER, editor_types=("org.example.rtfd",)), True),
        (dataclasses.replace(JOTTER, ai_schemas=("schema:wordProcessor.CreateWordProcessorDocumentIntent",)), True),
        (dataclasses.replace(JOTTER, ai_schemas=("schema:whiteboard.CreateCanvasBoardIntent",)), True),
        (dataclasses.replace(JOTTER, ai_schemas=("schema:journal.CreateJournalEntryIntent",)), False),
        (dataclasses.replace(JOTTER, editor_types=("public.png", "com.adobe.pdf")), False),
        (dataclasses.replace(JOTTER, viewer_types=("public.plain-text",)), False),  # only opens text
        (NOTEBOOK, True),  # makes notes
        (JOTTER, False),
        (dataclasses.replace(NOTEBOOK, kind="route"), False),
    ],
)
def test_doc_like(app, yes):
    assert note.doc_like(app) is yes


def test_the_real_engine_puts_the_notes_app_first_and_the_file_last(monkeypatch):
    monkeypatch.setitem(engine.TASKS, "write_note", dataclasses.replace(engine.TASKS["write_note"], fallback="file"))
    monkeypatch.setitem(engine.DRIVERS, "write_note", "neo.decide.kinds.note")
    monkeypatch.setattr(engine, "_zero_shot_can_answer", lambda: False)
    monkeypatch.setattr(engine, "_inventory", lambda: [NOTEBOOK, JOTTER])
    cands = asyncio.run(engine.choose("write_note", "make a note saying"))
    assert [(c.key, c.methods) for c in cands] == [("test.notebook", ("script_make", "ax_new")), ("file", ("file",))]
    assert asyncio.run(note.top_app()) is NOTEBOOK
    osa = FakeOsa()
    monkeypatch.setattr(media, "_osascript", osa)
    facts = asyncio.run(note.preview({"body": "milk", "folder": "Work"}, "note milk"))
    assert facts == {"app": "test.notebook", "method": "script_make", "candidates": 2,
                     "keys": "test.notebook,file", "folder": True, "named": False}
    assert osa.calls == [] and not (config.settings().data_dir / "decide.json").exists()  # nothing run or learned


def test_an_app_the_user_named_that_isnt_here_is_said(world, monkeypatch):
    world.chosen.append(cand(NOTEBOOK, "script_make"))
    world.osa.answers["make"] = made()
    monkeypatch.setattr(inventory, "snapshot", lambda: [NOTEBOOK])
    r = asyncio.run(note.write({"body": "buy milk and eggs", "app": "Scribbler"}, "note buy milk and eggs"))
    assert r.ok and r.say is True and r.text == "Scribbler isn't on this Mac. Saved in Notebook."
    assert world.asked[-1] == ("write_note", "note in Scribbler")
    assert world.osa.of("make")[0][0] == "test.notebook"
    world.osa.answers["make"] = made(text="bread")
    r = asyncio.run(note.write({"body": "bread", "app": "Notebook"}, "note bread in notebook"))
    assert r.text == "Saved in Notebook." and world.asked[-1] == ("write_note", "note in notebook")


def test_a_shadow_preview_touches_no_app(world):
    world.chosen += [cand(JOTTER, "ax_new"), cand(None, "file")]
    facts = asyncio.run(note.preview({"body": "milk"}))
    assert facts["app"] == "test.jotter" and facts["method"] == "ax_new" and world.osa.calls == []
    world.chosen[:] = [cand(None, "file")]
    assert asyncio.run(note.preview({"body": "milk"}))["method"] == "file"
    assert not config.settings().notes_dir.exists()


def test_the_dictionary_route_sets_everything_the_old_script_set(monkeypatch):
    """Equivalence with the legacy tool: every property its record set is set here too, by code."""
    from neo.tools.computer import legacy

    seen = []

    async def osascript(script, timeout=15):
        seen.append(script)
        return ""

    monkeypatch.setattr(legacy, "osascript", osascript)
    asyncio.run(legacy.notes_create("T", "B"))
    record = re.search(r"with properties \{(.*)\}", seen[0]).group(1)
    names = {k.strip() for k in re.findall(r"([a-z][a-z ]*):", record)}
    codes = {D.class_props["note"][n][0] for n in names}
    assert names == {"name", "body"} and codes == {"pnam", "body"}
    script, _argv = aescript.render_note(aescript.note_route(D), "b", "T", "B")
    record = re.search(r"«class prdt»:\{(.*)\}", script).group(1)
    assert codes <= set(re.findall(r"«class (\w{4})»:", record))


def test_through_the_real_engine_to_the_file_when_the_app_cannot(monkeypatch):
    monkeypatch.setitem(engine.TASKS, "write_note", dataclasses.replace(engine.TASKS["write_note"], fallback="file"))
    monkeypatch.setitem(engine.DRIVERS, "write_note", "neo.decide.kinds.note")
    monkeypatch.setattr(engine, "_zero_shot_can_answer", lambda: False)
    monkeypatch.setattr(engine, "_inventory", lambda: [NOTEBOOK])

    async def allowed(app):
        return None

    monkeypatch.setattr(media, "_control_gate", allowed)
    monkeypatch.setattr(media, "_trusted", lambda: True)
    osa = FakeOsa(make=(1, "", "Can't get default folder. (-1728)"))
    monkeypatch.setattr(media, "_osascript", osa)
    ui = FakeUI(monkeypatch, El("AXTextArea", AXValue=""))
    ui.app.attrs["AXMenuBar"] = El("AXMenuBar")  # no New command either
    r = asyncio.run(note.write({"body": "call the plumber"}, "note call the plumber"))
    assert r.ok and r.say is True and r.text == f"Saved it as a file in {config.settings().notes_dir}."
    assert len(osa.of("make")) == 1 and ui.launched == ["test.notebook"]
