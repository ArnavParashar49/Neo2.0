"""neo.decide.axkit: the AX facade over media's primitives, and the new finders and writers, on a
fake element tree patched into media (the same targets media's own tests patch). No real UI is
read, no key is pressed and the pasteboard is never touched."""

from __future__ import annotations

import pytest

from neo.decide import axkit, media


class El:
    def __init__(self, role, children=(), **attrs):
        self.attrs = {"AXRole": role, **attrs}
        self.children = list(children)
        self.pressed = 0

    def __repr__(self):
        return f"El({self.attrs.get('AXRole')}, {self.attrs.get('AXTitle') or self.attrs.get('AXValue')!r})"


class FakeAX:
    def __init__(self, monkeypatch, root: El, *, value_sticks: bool = True):
        self.root, self.sets, self.value_sticks = root, [], value_sticks
        monkeypatch.setattr(media, "_attr", self.attr)
        monkeypatch.setattr(media, "_set", self.set)
        monkeypatch.setattr(media, "_perform", self.perform)
        monkeypatch.setattr(media, "_actions", lambda el: ["AXPress"])
        monkeypatch.setattr(media, "_ax_app", lambda pid: self.root)
        monkeypatch.setattr(media, "_trusted", lambda: True)

    @staticmethod
    def attr(el, name, default=None):
        if el is None:
            return default
        if name == "AXChildren":
            return list(el.children)
        return el.attrs.get(name, default)

    def set(self, el, name, value):
        self.sets.append((name, value))
        if name == "AXValue" and not self.value_sticks:
            return True  # a web input that ignores a value set
        el.attrs[name] = value
        return True

    @staticmethod
    def perform(el, action):
        el.pressed += 1
        return True


@pytest.fixture(autouse=True)
def _words(monkeypatch):
    axkit.lexicon.cache_clear()
    monkeypatch.setattr(media, "lexicon", lambda: {"play": frozenset({"Play"}), "search": frozenset({"Search"}),
                                                   "find": frozenset({"Find"}), "pause": frozenset({"Pause"})})
    monkeypatch.setattr(axkit, "_MORE_SOURCES", ())
    yield
    axkit.lexicon.cache_clear()


def test_the_facade_follows_medias_primitives(monkeypatch):
    el = El("AXButton", AXTitle="Go")
    fake = FakeAX(monkeypatch, El("AXApplication", [el]))
    assert axkit.attr(el, "AXTitle") == "Go" and axkit.attr(el, "Nope", 3) == 3
    assert axkit.set_attr(el, "AXFocused", True) and fake.sets == [("AXFocused", True)]
    assert axkit.perform(el, "AXPress") and el.pressed == 1
    assert axkit.actions(el) == ["AXPress"] and axkit.app_el(1) is fake.root and axkit.trusted()
    returns = []
    monkeypatch.setattr(media, "_post_return", returns.append)
    axkit.post_return(42)
    assert returns == [42]


def test_lexicon_adds_send_and_new_to_medias_words():
    words = axkit.lexicon()
    assert "Send" in words["send"] and "New" in words["new"] and "Play" in words["play"]


def menu_bar():
    new = El("AXMenuItem", AXTitle="New Note", AXMenuItemCmdChar="N", AXMenuItemCmdModifiers=0)
    new_folder = El("AXMenuItem", AXTitle="New Folder", AXMenuItemCmdChar="N", AXMenuItemCmdModifiers=1)
    off = El("AXMenuItem", AXTitle="Send Later", AXMenuItemCmdChar="D", AXMenuItemCmdModifiers=1, AXEnabled=False)
    return El("AXMenuBar", [El("AXMenuBarItem", [El("AXMenu", [new_folder, new, off])])]), new, new_folder


def test_menu_items_are_found_by_their_shortcut_not_their_title(monkeypatch):
    bar, new, new_folder = menu_bar()
    FakeAX(monkeypatch, El("AXApplication", AXMenuBar=bar))
    assert axkit.menu_item(1, "n") == (new, "New Note")
    assert axkit.menu_item(1, "n", 1) == (new_folder, "New Folder")
    assert axkit.menu_item(1, "d", 1) is None  # disabled
    assert axkit.menu_item(1, "t") is None


def test_focused_editable_is_never_a_password_or_search_field(monkeypatch):
    app = El("AXApplication")
    FakeAX(monkeypatch, app)
    for el, ok in (
        (El("AXTextArea"), True),
        (El("AXTextField", AXPlaceholderValue="Title"), True),
        (El("AXGroup", AXEditableAncestor=object()), True),  # contenteditable in a web view
        (El("AXTextField", AXSubrole="AXSecureTextField"), False),
        (El("AXTextField", AXSubrole="AXSearchField"), False),
        (El("AXTextField", AXPlaceholderValue="Search messages"), False),
        (El("AXButton"), False),
    ):
        app.attrs["AXFocusedUIElement"] = el
        assert (axkit.focused_editable(1) is el) is ok, el


def chat_window():
    search = El("AXTextField", AXSubrole="AXSearchField")
    rows = [El("AXRow", [El("AXStaticText", AXValue="Priya Sharma")]), El("AXRow", [El("AXStaticText", AXValue="Priyanka")]),
            El("AXRow", [El("AXStaticText", AXValue="Priya Sharma")]), El("AXCell", AXTitle="Book club: Priya, Sam")]
    transcript = El("AXGroup", [El("AXStaticText", AXValue="running  late"), El("AXStaticText", AXValue="ok")])
    composer = El("AXTextArea", AXValue="")
    send = El("AXButton", AXDescription="Send")
    heading = El("AXHeading", AXValue="Priya Sharma")
    win = El("AXWindow", [search, El("AXTable", rows), heading, transcript, composer, send], AXTitle="")
    return win, search, rows, composer, send


def test_composer_rows_header_transcript_and_send_button(monkeypatch):
    win, search, rows, composer, send = chat_window()
    FakeAX(monkeypatch, El("AXApplication", [win]))
    assert axkit.composer(win) is composer and axkit.composer(win, exclude=(composer,)) is None
    got = axkit.rows_matching(win, "priya")
    assert [label for _el, label in got] == ["Priya Sharma", "Book club: Priya, Sam"]  # whole words, one per label
    assert got[0][0] is rows[0]
    assert axkit.rows_matching(win, "priya sharma")[0][1] == "Priya Sharma" and axkit.rows_matching(win, " ") == []
    assert axkit.header_title(win) == "Priya Sharma"
    win.attrs["AXTitle"] = "Chat with Sam"
    assert axkit.header_title(win) == "Chat with Sam"
    assert axkit.transcript_count(win, "Running late") == 1 and axkit.transcript_count(win, "") == 0
    assert axkit.commit_button(win) is send
    win.children.append(El("AXButton", AXTitle="send"))
    assert axkit.commit_button(win) is None  # two: not sure which


def test_set_text_by_value_and_by_typing_into_that_process_only(monkeypatch):
    box = El("AXTextArea", AXValue="")
    fake = FakeAX(monkeypatch, El("AXApplication", [box]))
    assert axkit.set_text(box, "hi\nthere") and box.attrs["AXValue"] == "hi\nthere"
    assert ("AXFocused", True) in fake.sets
    # a web input ignores the value set: type into its process, lines joined by the chord asked for
    web = El("AXTextArea", AXValue="")
    fake = FakeAX(monkeypatch, El("AXApplication", [web]), value_sticks=False)
    typed = []
    monkeypatch.setattr(axkit, "_pid_of", lambda el: 321)

    def type_(pid, text, newline):
        typed.append((pid, text, newline))
        web.attrs["AXValue"] = text

    monkeypatch.setattr(axkit, "_type", type_)
    assert axkit.set_text(web, "नमस्ते\ndost", newline="shift+return")
    assert typed == [(321, "नमस्ते\ndost", "shift+return")]
    # never into a password field; never on top of a half-set value
    assert not axkit.set_text(El("AXTextField", AXSubrole="AXSecureTextField"), "x")
    web.attrs["AXValue"] = "partial"
    typed.clear()
    assert not axkit.set_text(web, "hello") and typed == []


def test_clear_text_only_takes_back_neos_own_text(monkeypatch):
    box = El("AXTextArea", AXValue="running late")
    FakeAX(monkeypatch, El("AXApplication", [box]))
    assert not axkit.clear_text(box, "something else") and box.attrs["AXValue"] == "running late"
    assert axkit.clear_text(box, "running  late") and box.attrs["AXValue"] == ""


def test_web_url_and_text_of_the_front_window(monkeypatch):
    class Url:
        def absoluteString(self):
            return "https://example.test/search?q=x"

    area = El("AXWebArea", [El("AXStaticText", AXValue="Hello"), El("AXGroup", [El("AXStaticText", AXValue=" world ")])], AXURL=Url())
    win = El("AXWindow", [area])
    FakeAX(monkeypatch, El("AXApplication", [win], AXFocusedWindow=win))
    assert axkit.web_url(1) == "https://example.test/search?q=x"
    assert axkit.web_text(1) == "Hello\nworld" and axkit.web_text(1, limit=3) == "Hel"
    FakeAX(monkeypatch, El("AXApplication", [El("AXWindow")], AXFocusedWindow=El("AXWindow")))
    assert axkit.web_url(1) == "" and axkit.web_text(1) == ""


def test_dock_badge(monkeypatch):
    items = El("AXList", [El("AXDockItem", AXSubrole="AXApplicationDockItem", AXTitle="Postbox", AXStatusLabel="3"),
                          El("AXDockItem", AXSubrole="AXApplicationDockItem", AXTitle="Notebook")])
    FakeAX(monkeypatch, El("AXApplication", [items]))
    monkeypatch.setattr(axkit, "_dock_pid", lambda: 10)
    assert axkit.dock_badge("postbox") == "3" and axkit.dock_badge("Notebook") is None and axkit.dock_badge("Nope") is None
    monkeypatch.setattr(axkit, "_dock_pid", lambda: None)
    assert axkit.dock_badge("Postbox") is None
